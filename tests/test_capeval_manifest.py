"""Manifest schema validation, hashing, sidecar/layer
validation, and the checked-in capeval manifests."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.manifest import ManifestError

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "capeval"
MANIFESTS = ROOT / "configs" / "capeval" / "manifests"


def _raw_fixture() -> dict:
    return yaml.safe_load((FIXTURES / "manifest_synthetic.yaml").read_text())


def _manifest_from_raw(raw: dict) -> cm.Manifest:
    target = raw.get("target")
    return cm.Manifest(
        sweep=raw["sweep"], mode=raw["mode"], base=raw["base"],
        base_revision=raw["base_revision"], engine=raw["engine"],
        tiers=raw["tiers"], confirmatory=raw.get("confirmatory", False),
        margins=raw.get("margins", {}),
        target=None if target is None else cm.TargetSpec(
            primaries=[cm.PrimarySpec(**p) for p in target["primaries"]],
            secondary=[cm.PrimarySpec(**p) for p in target.get("secondary", [])],
            off_target_exclusions=target.get("off_target_exclusions", []),
            relevant_corpus=cm.CorpusRef(**target["relevant_corpus"]),
        ),
        ablations=[cm.AblationSpec(**a) for a in raw["ablations"]],
    )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_load_manifest_fixture_roundtrip():
    m = cm.load_manifest(FIXTURES / "manifest_synthetic.yaml")
    expected = _manifest_from_raw(_raw_fixture())
    # the loader resolves the relative artifact against the manifest's own
    # directory and stores the ABSOLUTE path in the spec (review-fix
    # contract); everything else round-trips verbatim
    expected.ablations[0].artifact = str(
        (FIXTURES / "vector_synthetic.safetensors").resolve())
    assert m == expected


def test_artifact_resolution_is_manifest_dir_anchored(tmp_path, monkeypatch):
    """Relative artifact paths resolve against the MANIFEST's directory —
    never the cwd — and the resolved absolute path is stored in the spec
    (review-fix contract, simplify-bug #5)."""
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    shutil.copy(FIXTURES / "vector_synthetic.safetensors",
                mdir / "vec.safetensors")
    shutil.copy(FIXTURES / "vector_synthetic.json", mdir / "vec.json")
    raw = _raw_fixture()
    raw["ablations"][0]["artifact"] = "vec.safetensors"
    p = mdir / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    # cwd holds a DECOY file of the same name — it must never win
    decoy_cwd = tmp_path / "elsewhere"
    decoy_cwd.mkdir()
    (decoy_cwd / "vec.safetensors").write_bytes(b"decoy")
    monkeypatch.chdir(decoy_cwd)
    m = cm.load_manifest(p)
    art = Path(m.ablations[0].artifact)
    assert art.is_absolute()
    assert art == (mdir / "vec.safetensors").resolve()
    # a relative path that exists ONLY in the cwd refuses: manifest-dir
    # anchoring has no cwd fallback
    raw["ablations"][0]["artifact"] = "cwd_only.safetensors"
    (decoy_cwd / "cwd_only.safetensors").write_bytes(b"decoy")
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ManifestError, match="does not exist"):
        cm.load_manifest(p)


def test_load_manifest_refuses_multidoc(tmp_path):
    raw = (FIXTURES / "manifest_synthetic.yaml").read_text()
    p = tmp_path / "multi.yaml"
    p.write_text(raw + "\n---\n" + raw)
    with pytest.raises(ManifestError, match="load_manifests"):
        cm.load_manifest(p)


def test_unknown_keys_refused_at_every_level(tmp_path):
    for mutate in (
        lambda r: r.update(unknown_top=1),
        lambda r: r["target"].update(unknown_target=1),
        lambda r: r["target"]["primaries"][0].update(unknown_primary=1),
        lambda r: r["target"]["relevant_corpus"].update(unknown_corpus=1),
        lambda r: r["ablations"][0].update(unknown_arm=1),
    ):
        raw = _raw_fixture()
        mutate(raw)
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(raw))
        with pytest.raises(ManifestError, match="unknown key"):
            cm.load_manifest(p)


def test_missing_required_keys_refused(tmp_path):
    for key in ("sweep", "mode", "base", "base_revision", "engine",
                "tiers", "ablations"):
        raw = _raw_fixture()
        del raw[key]
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(raw))
        with pytest.raises(ManifestError, match=key):
            cm.load_manifest(p)


# --------------------------------------------------------------------------
# validation refusals (structural — validate_manifest)
# --------------------------------------------------------------------------


def _valid() -> cm.Manifest:
    return _manifest_from_raw(_raw_fixture())


def _refuses(raw_mutator, match):
    raw = _raw_fixture()
    raw_mutator(raw)
    with pytest.raises(ManifestError, match=match):
        cm.validate_manifest(_manifest_from_raw(raw))


def test_validate_ok():
    cm.validate_manifest(_valid())  # no raise


def test_confirmatory_true_refused():
    _refuses(lambda r: r.update(confirmatory=True), "confirmatory")


def test_mode_refusals():
    _refuses(lambda r: r.update(mode="audit"), "mode")
    # F20 both ways
    _refuses(lambda r: r.update(mode="control"), "FORBID")
    _refuses(lambda r: r.update(mode="calibration"), "FORBID")

    raw = _raw_fixture()
    raw["target"] = None
    with pytest.raises(ManifestError, match="REQUIRES"):
        cm.validate_manifest(_manifest_from_raw(raw))


def test_base_revision_refusals():
    _refuses(lambda r: r.update(base_revision="main"), "40-hex")
    _refuses(lambda r: r.update(base_revision="0123abc"), "40-hex")
    _refuses(lambda r: r.update(base_revision="z" * 40), "40-hex")


def test_engine_refusals():
    _refuses(lambda r: r.update(engine="sglang"), "engine")
    # vllm accepts kind: none only (v0)
    _refuses(lambda r: r.update(engine="vllm"), "vllm")


def test_tiers_refusals():
    _refuses(lambda r: r.update(tiers=[]), "tiers")
    _refuses(lambda r: r.update(tiers=[0, 2]), "tiers")
    _refuses(lambda r: r.update(tiers=[0, 0]), "tiers")


def test_target_refusals():
    _refuses(lambda r: r["target"].update(primaries=[]), "primaries")
    _refuses(lambda r: r["target"]["primaries"][0].update(direction="sideways"),
             "direction")
    _refuses(lambda r: r["target"]["primaries"][0].update(min_effect=0),
             "min_effect")
    # relevant_corpus addressing: exactly one of path/repo_id
    _refuses(lambda r: r["target"]["relevant_corpus"].update(
        repo_id="org/data"), "exactly one")
    _refuses(lambda r: r["target"]["relevant_corpus"].update(
        path=None), "exactly one")
    _refuses(lambda r: r["target"]["relevant_corpus"].update(
        revision="a" * 40), "meaningless")

    raw = _raw_fixture()
    raw["target"]["relevant_corpus"] = {
        "repo_id": "org/data", "slice": {"first": 2, "field": "prompt"}}
    with pytest.raises(ManifestError, match="revision"):
        cm.validate_manifest(_manifest_from_raw(raw))


def test_margin_refusals():
    _refuses(lambda r: r.update(margins={"math": -0.1}), "margins")


def test_arm_refusals():
    steer = dict(_raw_fixture()["ablations"][0])

    def with_arm(**kw):
        def mut(r):
            arm = {**steer, **kw}
            r["ablations"] = [arm]
        return mut

    _refuses(with_arm(artifact=None), "artifact")
    _refuses(with_arm(layer=None), "layer")
    _refuses(with_arm(layer=-3), "layer")
    _refuses(with_arm(coeffs=[]), "coeff")
    _refuses(with_arm(coeffs=["big"]), "numbers")
    _refuses(with_arm(seed=3), "seed")
    _refuses(with_arm(params={"x": 1}), "params")
    _refuses(with_arm(kind="lora_merge"), "unknown kind")
    _refuses(with_arm(name="a__b"), "reserve")
    _refuses(with_arm(name="a/b"), "reserve")

    # duplicate names
    def dup(r):
        r["ablations"] = [steer, dict(steer)]
    _refuses(dup, "duplicate")

    # none-kind constraints
    def none_arm(**kw):
        def mut(r):
            r["mode"] = "calibration"
            r["target"] = None
            arm = {"name": "none", "kind": "none", **kw}
            r["ablations"] = [arm]
        return mut

    raw = _raw_fixture()
    none_arm()(raw)
    cm.validate_manifest(_manifest_from_raw(raw))  # valid baseline
    _refuses(none_arm(coeffs=[1.0]), "none")
    _refuses(none_arm(artifact="x"), "none")
    _refuses(none_arm(seed=1), "none")

    # control-kind constraints
    ctrl = dict(_raw_fixture()["ablations"][1])

    def with_ctrl(**kw):
        def mut(r):
            r["ablations"] = [{**ctrl, **kw}]
        return mut

    _refuses(with_ctrl(seed=None), "seed")
    _refuses(with_ctrl(artifact="x.safetensors"), "no top-level artifact")
    _refuses(with_ctrl(layer=15), "params")
    _refuses(with_ctrl(coeffs=[]), "coeff")
    _refuses(with_ctrl(kind="control:"), "empty control kind")

    # empty ablations
    def empty(r):
        r["ablations"] = []
    _refuses(empty, "non-empty")


# --------------------------------------------------------------------------
# artifact / sidecar (loader-side, F9)
# --------------------------------------------------------------------------


def test_absent_artifact_refused(tmp_path):
    raw = _raw_fixture()
    raw["ablations"][0]["artifact"] = str(tmp_path / "nope.safetensors")
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ManifestError, match="does not exist"):
        cm.load_manifest(p)


def test_missing_sidecar_refused_for_steering(tmp_path):
    art = tmp_path / "vec.safetensors"
    shutil.copy(FIXTURES / "vector_synthetic.safetensors", art)
    raw = _raw_fixture()
    raw["ablations"][0]["artifact"] = str(art)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ManifestError, match="sidecar"):
        cm.load_manifest(p)


def test_sidecar_layer_mismatch_refused(tmp_path):
    art = tmp_path / "vec.safetensors"
    shutil.copy(FIXTURES / "vector_synthetic.safetensors", art)
    side = json.loads((FIXTURES / "vector_synthetic.json").read_text())
    side["layer"] = 14
    (tmp_path / "vec.json").write_text(json.dumps(side))
    raw = _raw_fixture()
    raw["ablations"][0]["artifact"] = str(art)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ManifestError, match="does not match"):
        cm.load_manifest(p)
    # both values named in the refusal
    try:
        cm.load_manifest(p)
    except ManifestError as e:
        assert "15" in str(e) and "14" in str(e)


def test_sidecar_layers_list_accepted():
    arm = cm.AblationSpec(name="a", kind="steering", artifact="x", layer=15,
                          coeffs=[1.0])
    cm.validate_layer_against_sidecar(arm, {"layers": [15]})
    with pytest.raises(ManifestError, match="does not match"):
        cm.validate_layer_against_sidecar(arm, {"layers": [3, 7]})
    with pytest.raises(ManifestError, match="neither"):
        cm.validate_layer_against_sidecar(arm, {"norms": {}})


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def test_canonical_json_stable_and_loud():
    assert (cm.canonical_json({"b": 1, "a": [1, 2]})
            == cm.canonical_json({"a": [1, 2], "b": 1})
            == '{"a":[1,2],"b":1}')
    with pytest.raises(ManifestError):
        cm.canonical_json({"x": float("nan")})
    with pytest.raises(ManifestError):
        cm.canonical_json({"x": object()})


def test_canonical_json_names_the_key_path_of_a_non_finite_value():
    """§PW-nanfix-1. The encoder's own message ("Out of range float values
    are not JSON compliant: nan") says a payload holds one SOMEWHERE, which
    is where a wave of Llama cells died: the NaN was a scorer's unscored
    sentinel three levels down in a row's ``extra``, and nothing in the
    traceback said which key. The refusal now names the path, derived by
    walking the same object the encoder rejected."""
    payload = {"task": "inspect/or_bench_hard_1k", "n": 200,
               "extra": {"scores": {"model_graded_qa": float("nan")}}}
    with pytest.raises(ManifestError) as ei:
        cm.canonical_json(payload)
    assert "extra.scores.model_graded_qa" in str(ei.value)
    assert "non-finite" in str(ei.value)
    # a list index, and infinities, are named the same way
    with pytest.raises(ManifestError, match=r"rows\[1\].v"):
        cm.canonical_json({"rows": [{"v": 1.0}, {"v": float("inf")}]})
    with pytest.raises(ManifestError, match="'<root>'"):
        cm.canonical_json(float("-inf"))
    # a value that is merely unserializable keeps the generic message —
    # only a non-finite float gets a key path
    with pytest.raises(ManifestError, match="unserializable object"):
        cm.canonical_json({"x": object()})
    assert cm._first_non_finite({"a": {"b": 1.0}}) is None


def test_sha256_hex16():
    assert cm.sha256_hex16(b"capeval") == cm.sha256_hex16(b"capeval")
    assert len(cm.sha256_hex16(b"capeval")) == 16
    assert cm.sha256_hex16(b"a") != cm.sha256_hex16(b"b")
    with pytest.raises(ManifestError):
        cm.sha256_hex16("not bytes")


def test_intervention_hash_steering_sensitivity():
    art = (FIXTURES / "vector_synthetic.safetensors").read_bytes()
    side = (FIXTURES / "vector_synthetic.json").read_bytes()
    spec = {"kind": "steering", "layer": 15, "coeff": 2.0, "skip_bos": True}
    h = cm.intervention_hash(spec, art, side)
    assert len(h) == 16
    # stable across key order
    spec2 = {"skip_bos": True, "coeff": 2.0, "layer": 15, "kind": "steering"}
    assert cm.intervention_hash(spec2, art, side) == h
    # artifact byte flip changes it
    flipped = bytes([art[0] ^ 1]) + art[1:]
    assert cm.intervention_hash(spec, flipped, side) != h
    # sidecar byte change changes it
    assert cm.intervention_hash(spec, art, side + b" ") != h
    # coeff change changes it
    assert cm.intervention_hash({**spec, "coeff": 4.0}, art, side) != h
    # byte args REQUIRED for steering kinds
    with pytest.raises(ManifestError, match="BOTH"):
        cm.intervention_hash(spec)
    # blob kinds pass b"" explicitly and it hashes fine
    h_blob = cm.intervention_hash({**spec, "kind": "sisteer_object"}, art, b"")
    assert len(h_blob) == 16 and h_blob != h


def test_intervention_hash_control_kinds():
    spec = {"kind": "control:random_vector", "seed": 7, "params": {"d": 8}}
    h = cm.intervention_hash(spec)
    # control hash is a pure function of {kind, seed, params} — resolved
    # extras don't move it
    assert cm.intervention_hash({**spec, "extra": "ignored"}) == h
    assert cm.intervention_hash({**spec, "seed": 8}) != h
    assert cm.intervention_hash(
        {**spec, "params": {"d": 9}}) != h
    with pytest.raises(ManifestError, match="seed"):
        cm.intervention_hash({"kind": "control:x", "params": {}})
    with pytest.raises(ManifestError, match="no byte"):
        cm.intervention_hash(spec, b"x", b"y")


def test_intervention_hash_none_kind():
    h = cm.intervention_hash({"kind": "none", "model": "m"})
    assert len(h) == 16
    with pytest.raises(ManifestError, match="no byte"):
        cm.intervention_hash({"kind": "none"}, b"x", b"")
    with pytest.raises(ManifestError, match="unknown kind"):
        cm.intervention_hash({"kind": "mystery"})


# --------------------------------------------------------------------------
# multi-document manifests (load_manifests)
# --------------------------------------------------------------------------


def _minimal_sweep_doc(arm_name: str, sweep: str = "multi") -> dict:
    raw = _raw_fixture()
    raw["sweep"] = sweep
    # the docs get written under tmp_path, so the fixture artifact must be
    # addressed absolutely (relative would anchor to tmp_path and refuse)
    raw["ablations"] = [dict(
        raw["ablations"][0], name=arm_name,
        artifact=str((FIXTURES / "vector_synthetic.safetensors").resolve()))]
    return raw


def test_load_manifests_multidoc(tmp_path):
    docs = [_minimal_sweep_doc("arm-a"), _minimal_sweep_doc("arm-b")]
    p = tmp_path / "multi.yaml"
    p.write_text("---\n".join(yaml.safe_dump(d) for d in docs))
    ms = cm.load_manifests(p)
    assert [m.ablations[0].name for m in ms] == ["arm-a", "arm-b"]
    assert all(m.sweep == "multi" for m in ms)

    # sweep-name disagreement refused
    docs[1]["sweep"] = "other"
    p.write_text("---\n".join(yaml.safe_dump(d) for d in docs))
    with pytest.raises(ManifestError, match="sweep"):
        cm.load_manifests(p)

    # arm-name collision across documents refused
    docs = [_minimal_sweep_doc("arm-a"), _minimal_sweep_doc("arm-a")]
    p.write_text("---\n".join(yaml.safe_dump(d) for d in docs))
    with pytest.raises(ManifestError, match="collide"):
        cm.load_manifests(p)


# --------------------------------------------------------------------------
# the checked-in capeval manifests
# --------------------------------------------------------------------------


def test_calibration_manifest_loads():
    m = cm.load_manifest(MANIFESTS / "calibration.yaml")
    assert m.mode == "calibration" and m.target is None
    assert m.engine == "hf" and m.tiers == [0, 1]
    assert m.base_revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    [arm] = m.ablations
    assert arm.kind == "none" and arm.name == "none" and arm.coeffs == []


def test_controls_manifest_loads():
    m = cm.load_manifest(MANIFESTS / "controls.yaml")
    assert m.mode == "control" and m.target is None
    kinds = {a.name: a.kind for a in m.ablations}
    assert kinds == {
        "zero-coeff": "control:zero_coeff",
        "sham-hook": "control:sham_hook",
        "rand-vec": "control:random_vector",
        "noise-lora": "control:noise_lora",
        "weight-noise": "control:weight_noise",
    }
    by_name = {a.name: a for a in m.ablations}
    assert by_name["zero-coeff"].coeffs == [0.0]
    assert by_name["zero-coeff"].params["artifact"].endswith("vector.safetensors")
    assert by_name["rand-vec"].coeffs == [0.5, 1.0, 2.0, 4.0]
    assert by_name["rand-vec"].params["tier1_coeffs"] == [1.0, 4.0]
    assert by_name["rand-vec"].params["norm"]["source"] == "median_sidecar_norm"
    assert len(by_name["rand-vec"].params["norm"]["sidecars"]) == 4
    assert by_name["noise-lora"].coeffs == [1e-4, 3e-4, 1e-3, 3e-3]
    assert by_name["weight-noise"].coeffs == [3e-3]
    # every arm carries a seed and no top-level artifact (control contract)
    assert all(a.seed is not None and a.artifact is None for a in m.ablations)

