"""The twosi-fits Modal app's image routing (fake-modal + source-level).

Every model key maps to exactly ONE image variant (std / qwen / nemotron)
through _variant(), and every routing site — function-object picks and
deployed-function names alike — derives from it. Pinned after the nemotron
key moved off image_qwen onto image_nemotron (the hybrid-stack recipe plus
the pinned mamba-ssm fused kernels, 2026-09-02): the qwen and std tables
must not move, a nemotron key must never fall back to a kernel-less image,
and --b200 on it must refuse rather than quietly take a binding that does
not exist.

`modal` is not installed in the test env; a minimal duck-typed stub stands
in so the module imports (the app objects are never exercised here).
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC_PATH = REPO / "scripts" / "modal" / "twosi_fits.py"
SRC = SRC_PATH.read_text()


class _Chain:
    """Any method returns self — enough for the Image builder chains."""

    def __getattr__(self, name):
        def _method(*a, **k):
            return self
        return _method


def _fake_modal() -> types.ModuleType:
    fake = types.ModuleType("modal")

    class _App:
        def __init__(self, name):
            self.name = name

        def function(self, *a, **k):
            return lambda fn: fn

        def local_entrypoint(self, *a, **k):
            return lambda fn: fn

    fake.App = _App
    fake.Image = SimpleNamespace(debian_slim=lambda **k: _Chain(),
                                 from_registry=lambda *a, **k: _Chain())
    fake.Volume = SimpleNamespace(
        from_name=lambda *a, **k: SimpleNamespace(
            listdir=lambda *aa, **kk: [], commit=lambda: None))
    fake.Secret = SimpleNamespace(from_name=lambda n: ("secret", n))
    fake.Function = SimpleNamespace(from_name=lambda *a, **k: None)
    fake.FunctionCall = SimpleNamespace(from_id=lambda i: None)
    fake.exception = SimpleNamespace(OutputExpiredError=Exception)
    return fake


@pytest.fixture(scope="module")
def tf():
    # ALWAYS this file's own stub: another test module may have left ITS
    # fake ``modal`` in sys.modules (test_grokmt_served_resume installs one at
    # import and never removes it), and a foreign stub without
    # Image.debian_slim made every test here ERROR at setup when both files
    # ran in one suite (2026-09-03). Whatever was there is restored after.
    prev = sys.modules.get("modal")
    sys.modules["modal"] = _fake_modal()
    spec = importlib.util.spec_from_file_location("_twosi_fits_test",
                                                  SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_twosi_fits_test"] = mod
    spec.loader.exec_module(mod)
    yield mod
    if prev is None:
        sys.modules.pop("modal", None)
    else:
        sys.modules["modal"] = prev


def test_key_tables_and_variants(tf):
    assert tf.QWEN_IMAGE_KEYS == ("qwen38_27b", "qwen35_9b", "qwen35_4b")
    assert tf.NEMOTRON_IMAGE_KEYS == ("nemotron3nano4b",)
    # glm46vflash rides image_std: a dense full-attention VLM used
    # text-only, nothing hybrid to fuse (its registration commit).
    assert {k: tf._variant(k) for k in tf.MODELS} == {
        "gemma4e4b": "std", "gemma4e2b": "std", "llama8b": "std",
        "glm46vflash": "std",
        "qwen38_27b": "qwen", "qwen35_9b": "qwen", "qwen35_4b": "qwen",
        "nemotron3nano4b": "nemotron"}


def test_function_object_routing(tf):
    assert tf._fit_fn("nemotron3nano4b") is tf.fit_h200_nemotron
    assert tf._fit_fn("qwen38_27b") is tf.fit_h200_qwen
    assert tf._fit_fn("qwen38_27b", b200=True) is tf.fit_b200_qwen
    assert tf._fit_fn("llama8b") is tf.fit_h200_std
    # historical std behavior kept: --b200 is ignored, not refused
    assert tf._fit_fn("llama8b", b200=True) is tf.fit_h200_std
    assert tf._memprobe_fn("nemotron3nano4b") is tf.memprobe_h200_nemotron
    assert tf._memprobe_fn("qwen35_4b", b200=True) is tf.memprobe_b200_qwen
    assert tf._memprobe_fn("qwen35_4b") is tf.memprobe_h200_qwen
    assert tf._memprobe_fn("gemma4e4b") is tf.memprobe_h200_std
    assert tf._teacher_fn("nemotron3nano4b") is tf.teacher_h200_nemotron
    assert tf._teacher_fn("qwen35_9b") is tf.teacher_h200_qwen
    assert tf._teacher_fn("gemma4e2b") is tf.teacher_h200_std
    assert tf._HPO_FNS == {"std": tf.hpo_h200_std,
                           "qwen": tf.hpo_h200_qwen,
                           "nemotron": tf.hpo_h200_nemotron}


def test_deployed_function_names_follow_the_variant(tf):
    assert tf._fit_fname("nemotron3nano4b") == "fit_h200_nemotron"
    assert tf._fit_fname("qwen38_27b") == "fit_h200_qwen"
    assert tf._fit_fname("qwen38_27b", True) == "fit_b200_qwen"
    assert tf._fit_fname("llama8b") == "fit_h200_std"
    assert tf._fit_fname("llama8b", True) == "fit_h200_std"


def test_b200_refuses_on_the_nemotron_image(tf):
    for pick in (lambda: tf._fit_fn("nemotron3nano4b", b200=True),
                 lambda: tf._memprobe_fn("nemotron3nano4b", b200=True),
                 lambda: tf._fit_fname("nemotron3nano4b", True)):
        with pytest.raises(SystemExit, match="no binding on the nemotron"):
            pick()


_BINDING = (r"@app\.function\((?P<deco>[^)]*)\)\s*\ndef {name}\("
            r".*?\n    return (?P<impl>\w+)\(")


def test_every_qwen_gpu_binding_has_a_nemotron_sibling():
    """Modal binds image/gpu statically per function: every *_h200_qwen
    binding has a *_h200_nemotron twin on image_nemotron, H200, wrapping
    the same implementation."""
    stems = set(re.findall(r"^def (\w+)_h200_qwen\(", SRC, flags=re.M))
    assert stems == {"teacher", "memprobe", "hpomem", "hpo", "fit"}
    for stem in stems:
        impl = {}
        for variant, image in (("qwen", "image_qwen"),
                               ("nemotron", "image_nemotron")):
            m = re.search(_BINDING.format(name=f"{stem}_h200_{variant}"),
                          SRC, flags=re.S)
            assert m, (stem, variant)
            assert f"image={image}" in m["deco"], (stem, variant)
            assert 'gpu="H200"' in m["deco"], (stem, variant)
            impl[variant] = m["impl"]
        assert impl["qwen"] == impl["nemotron"], stem
    # no B200 binding on the nemotron image, by design (the 4B fits H200)
    assert "_b200_nemotron" not in SRC.replace("fit_b200_nemotron / "
                                               "memprobe_b200_nemotron", "")


def test_entrypoint_routes_through_variant_only():
    """No routing site in the entrypoint string-matches the qwen table (the
    qwen35 transfer fits once landed on image_std that way); every pick
    goes through _variant()-derived helpers."""
    main_src = SRC.split("def main(", 1)[1]
    assert "in QWEN_IMAGE_KEYS" not in main_src
    assert "in NEMOTRON_IMAGE_KEYS" not in main_src
    assert "_fit_fname(model, b200)" in main_src
    assert "_teacher_fn(model)" in main_src
    assert 'fns[_variant(r["model"])]' in main_src
    assert 'f"hpo_h200_{v}")' in main_src and "for v in _HPO_FNS" in main_src
    assert "_HPO_FNS[v], items_by_variant[v]" in main_src
