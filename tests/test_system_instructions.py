"""``system_instructions/{safety,constitution}.md`` are BYTE-IDENTICAL to the constants the
harness prompts with: ``SAFETY_SI.text`` and ``CONSTITUTION_PROMPT``; the
registry ``si:<sha256>`` ids the two virtual
objects of ``objects/MANIFEST.json`` carry, and the sha-equality of the two vendored copies of
the FULL constitution (harness passthrough cell vs corpus-generation snapshot)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from front_door_testlib import assert_no_tokens

from sisteer.data.instructions import SAFETY_SI
from sisteer.data.safety_texts import CONSTITUTION_PROMPT, CONSTITUTION_SI_WORDS
from sisteer.registry.identity import object_id

REPO = Path(__file__).resolve().parents[1]
SI_DIR = REPO / "system_instructions"

#: The registry ids of the two system-instruction objects (``v_paper_objects``, in_paper = 1).
SAFETY_SI_OBJECT_ID = "si:ac3a58f61d53cf2412c02b8bb56063b9683a9c1154ef1f0978efc1449a836e08"
CONSTITUTION_SI_OBJECT_ID = "si:21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize(
    "name, text",
    [("safety.md", SAFETY_SI.text), ("constitution.md", CONSTITUTION_PROMPT)],
    ids=["safety", "constitution"],
)
def test_file_bytes_equal_constant(name, text):
    path = SI_DIR / name
    assert path.is_file(), f"{path} missing"
    on_disk = path.read_bytes()
    assert on_disk == text.encode("utf-8"), f"{name} differs from its constant"
    assert not on_disk.endswith(b"\n"), f"{name}: a trailing newline breaks the byte identity"


def test_registry_si_object_ids():
    assert object_id("system_instruction", si_text_sha256=_sha((SI_DIR / "safety.md").read_bytes())) == SAFETY_SI_OBJECT_ID
    assert object_id("system_instruction", si_text_sha256=_sha((SI_DIR / "constitution.md").read_bytes())) == CONSTITUTION_SI_OBJECT_ID


def test_constitution_excerpt_word_count_matches_the_measured_constant():
    assert len((SI_DIR / "constitution.md").read_text(encoding="utf-8").split()) == CONSTITUTION_SI_WORDS == 4896


def test_full_constitution_copies_are_sha_equal():
    harness = REPO / "src/sisteer/data/constitution_full.md"
    vendored = REPO / "corpus_generator/constitution/constitution.md"
    snap = json.loads((REPO / "corpus_generator/constitution/SNAPSHOT.json").read_text(encoding="utf-8"))
    assert _sha(harness.read_bytes()) == _sha(vendored.read_bytes()) == snap["sha256"]


def test_readme_names_both_ids_and_no_personal_identifiers():
    readme = (SI_DIR / "README.md").read_text(encoding="utf-8")
    assert SAFETY_SI_OBJECT_ID in readme and CONSTITUTION_SI_OBJECT_ID in readme
    for p in SI_DIR.iterdir():   # README.md included
        assert_no_tokens(p.read_text(encoding="utf-8"), p.name)
