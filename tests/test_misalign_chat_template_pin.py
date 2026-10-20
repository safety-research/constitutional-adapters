"""The organisms transfer target `llama8b` is served under Meta's STOCK Llama-3.1 chat template (vendored
byte-exact), pinned by the misalign manifest; every other target keeps the pinned tokenizer's own template."""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
STOCK_SHA16 = "e10ca381b1ccc5cf"     # meta-llama/Llama-3.1-8B-Instruct == unsloth/Llama-3.1-8B-Instruct, 4,614 bytes
MINIMAL_SHA16 = "b48c47f644389271"   # NousResearch/Meta-Llama-3.1-8B-Instruct @d10aef79: 348 bytes, no system block


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_misalign_manifest", REPO / "scripts/sdfbase/full22/misalign/make_misalign_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stock_template_is_vendored_byte_exact_and_pinned_for_llama8b_org_only():
    m = _generator()
    sha, path = m.chat_template_pin("llama8b_org")
    assert path.is_file() and path.stat().st_size == 4614
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()[:16] == STOCK_SHA16
    text = path.read_text(encoding="utf-8")
    assert "Cutting Knowledge Date: December 2023" in text and 'date_string = "26 Jul 2024"' in text
    for other in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b", "llama8b_it", "gemma4e4b_it", "qwen35_4b_it",
                  "gemma4e4b_bl", "qwen35_4b_bl", "nemotron3nano4b_bl", "llama8b_bl", "qwen35_4b_psr", "qwen35_4b_allpos",
                  "qwen35_4b_org"):
        assert m.chat_template_pin(other) == (None, None)


def test_llama8b_org_manifest_pins_the_stock_hash_and_the_agentic_llama8b_keys_keep_the_minimal_one():
    org = (REPO / "configs/capeval/manifests/misalign_full22_llama8b_org.yaml").read_text(encoding="utf-8")
    assert f"  chat_template_hash: {STOCK_SHA16}" in org and "llama31_instruct_stock.jinja" in org
    for key in ("llama8b", "llama8b_it"):
        text = (REPO / f"configs/capeval/manifests/misalign_full22_{key}.yaml").read_text(encoding="utf-8")
        assert f"  chat_template_hash: {MINIMAL_SHA16}" in text
