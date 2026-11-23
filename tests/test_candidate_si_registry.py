"""The candidate-SI loader (sisteer.data.candidate_sis) and its env-guarded registry widening (instructions.py).

All fixtures here are SYNTHETIC: invented tiny SI texts under the candidate set's id grammar (A1..A20, B1..B20, C1..C20). The
real 60 candidate texts (B1 = the full ~142k-char constitution) are never committed; the only real texts a passing fixture
carries are B2/C3, which the loader requires byte-equal to CONSTITUTION_PROMPT/IRRELEVANT_PROMPT, both repo constants imported
from safety_texts rather than copied.

The snapshot is not part of this repository and the loader names no default location: a caller passes a path or sets
SISTEER_CANDIDATE_SIS, and without either the load refuses by name. The tests against the REAL snapshot (its provenance, and
the C16 text the sikl4 refits distil: 517 characters, sha256 ba214825...) run only when SISTEER_CANDIDATE_SIS names it, so a
checkout without it is unaffected.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.data import instructions
from sisteer.data.candidate_sis import (
    CATEGORY_BY_PREFIX,
    EXPECTED_IDS,
    N_EXPECTED,
    N_PER_CATEGORY,
    PATH_ENV,
    default_path,
    load_candidate_instructions,
)
from sisteer.data.instructions import Instruction
from sisteer.data.safety_texts import CONSTITUTION_PROMPT, IRRELEVANT_PROMPT
from sisteer.eval import checkers

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synth_entries() -> list[dict]:
    """60 invented entries under the candidate id grammar, with the two
    entries the loader cross-checks (B2/C3) set to the repo constants."""
    cats = {"A": "safety", "B": "constitution", "C": "irrelevant"}
    entries = []
    for prefix in "ABC":
        for n in range(1, 21):
            sid = f"{prefix}{n}"
            entries.append({
                "id": sid,
                "category": cats[prefix],
                "text": (f"Synthetic {cats[prefix]} instruction {sid}: "
                         f"always mention the token {sid.lower()}-marker."),
            })
    by_id = {e["id"]: e for e in entries}
    by_id["B2"]["text"] = CONSTITUTION_PROMPT
    by_id["C3"]["text"] = IRRELEVANT_PROMPT
    return entries


def _write(tmp_path: Path, entries: list[dict], name: str = "cand.json") -> str:
    p = tmp_path / name
    p.write_text(
        json.dumps({"meta": {"purpose": "synthetic fixture"},
                    "system_instructions": entries}),
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def synth_path(tmp_path):
    return _write(tmp_path, _synth_entries())


# --- happy path -----------------------------------------------------------------


def test_loads_60_frozen_external_instructions(synth_path):
    instrs = load_candidate_instructions(synth_path)
    assert len(instrs) == N_EXPECTED == 60
    assert {i.id for i in instrs} == set(EXPECTED_IDS)
    assert all(isinstance(i, Instruction) for i in instrs)
    assert all(i.checker == "external" for i in instrs)
    assert all(i.checker_kwargs == {"suite": "align"} for i in instrs)
    assert all(i.anti_text is None for i in instrs)
    assert all(i.trigger_set is None for i in instrs)
    # file order preserved (A1..A20, B1..B20, C1..C20)
    assert [i.id for i in instrs[:3]] == ["A1", "A2", "A3"]
    assert instrs[20].id == "B1" and instrs[40].id == "C1"
    # 60 distinct text hashes — text identity is the intervention identity
    assert len({hashlib.sha256(i.text.encode()).hexdigest()
                for i in instrs}) == 60
    # frozen: the served text can't be mutated after load
    with pytest.raises(dataclasses.FrozenInstanceError):
        instrs[0].text = "mutated"


def test_bare_list_form_accepted(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps(_synth_entries()), encoding="utf-8")
    assert len(load_candidate_instructions(str(p))) == 60


def test_default_path_resolution(synth_path, monkeypatch):
    monkeypatch.setenv(PATH_ENV, synth_path)
    assert default_path() == synth_path
    assert len(load_candidate_instructions()) == 60
    # no fallback location: the snapshot is not part of this repository, so an unset variable refuses by name
    monkeypatch.delenv(PATH_ENV)
    with pytest.raises(FileNotFoundError, match=PATH_ENV):
        default_path()
    with pytest.raises(FileNotFoundError, match=PATH_ENV):
        load_candidate_instructions()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidate_instructions(str(tmp_path / "nope.json"))


# --- the "external" checker sentinel passes test_data's checker-exists gate ------


def test_external_checker_is_deferred_and_fails_loudly(synth_path):
    from test_data import DEFERRED_CHECKERS  # the gate's own exemption set

    instrs = load_candidate_instructions(synth_path)
    assert all(i.checker in DEFERRED_CHECKERS for i in instrs)
    # the sentinel means "externally judged": dispatching it is a loud error,
    # never a quietly-wrong number
    with pytest.raises(KeyError):
        checkers.get_checker("external")


# --- expect_sha -----------------------------------------------------------------


def test_expect_sha_match_and_mismatch(synth_path):
    raw = Path(synth_path).read_bytes()
    good = hashlib.sha256(raw).hexdigest()
    assert len(load_candidate_instructions(synth_path, expect_sha=good)) == 60
    with pytest.raises(ValueError, match="sha mismatch"):
        load_candidate_instructions(synth_path, expect_sha="0" * 64)


# --- integrity rejections -------------------------------------------------------


def test_double_underscore_id_rejected(tmp_path):
    entries = _synth_entries()
    entries[0]["id"] = "A__1"
    with pytest.raises(ValueError, match="'__'"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_duplicate_id_rejected(tmp_path):
    entries = _synth_entries()
    entries[1]["id"] = "A1"  # A2 -> A1
    with pytest.raises(ValueError, match="duplicate id 'A1'"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_wrong_id_set_rejected(tmp_path):
    entries = _synth_entries()
    entries[-1]["id"] = "X1"  # C20 -> X1
    with pytest.raises(ValueError, match="C20"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_wrong_count_rejected(tmp_path):
    entries = _synth_entries()[:-1]  # 59
    with pytest.raises(ValueError, match="60"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_duplicate_text_rejected(tmp_path):
    entries = _synth_entries()
    by_id = {e["id"]: e for e in entries}
    by_id["A5"]["text"] = by_id["A4"]["text"]
    with pytest.raises(ValueError, match="not distinct"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_pointer_sentinel_rejected(tmp_path):
    # B1 must be the inlined constitution, never the retired pointer form
    entries = _synth_entries()
    by_id = {e["id"]: e for e in entries}
    by_id["B1"]["text"] = "<<FULL_CONSTITUTION_VERBATIM>>"
    with pytest.raises(ValueError, match="pointer"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_missing_field_rejected(tmp_path):
    entries = _synth_entries()
    del entries[7]["text"]
    with pytest.raises(ValueError, match="missing/empty"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_mislabeled_category_rejected(tmp_path):
    # the id prefix BINDS the category — an A-id filed as
    # "constitution" would compete in the wrong promotion pool.
    entries = _synth_entries()
    by_id = {e["id"]: e for e in entries}
    by_id["A7"]["category"] = "constitution"
    with pytest.raises(ValueError, match="prefix 'A' binds it to 'safety'"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_every_prefix_binding_direction_rejected(tmp_path):
    # each prefix, mislabeled to each OTHER category, is refused
    for sid, wrong in (("A1", "irrelevant"), ("B9", "safety"),
                       ("C20", "constitution")):
        entries = _synth_entries()
        {e["id"]: e for e in entries}[sid]["category"] = wrong
        with pytest.raises(ValueError, match="prefix-category binding"):
            load_candidate_instructions(_write(tmp_path, entries))


def test_loader_categories_are_exactly_20_per_category(synth_path):
    # the counts invariant consumers lean on, asserted by the
    # loader itself (and re-checked here on the loaded objects).
    instrs = load_candidate_instructions(synth_path)
    counts: dict[str, int] = {}
    for i in instrs:
        counts[i.category] = counts.get(i.category, 0) + 1
    assert counts == {c: N_PER_CATEGORY for c in CATEGORY_BY_PREFIX.values()}
    assert all(i.category == CATEGORY_BY_PREFIX[i.id[0]] for i in instrs)


def test_b2_must_equal_constitution_prompt(tmp_path):
    entries = _synth_entries()
    by_id = {e["id"]: e for e in entries}
    by_id["B2"]["text"] = "Synthetic drifted B2 text that is not the constitution."
    with pytest.raises(ValueError, match="CONSTITUTION_PROMPT"):
        load_candidate_instructions(_write(tmp_path, entries))


def test_c3_must_equal_irrelevant_prompt(tmp_path):
    entries = _synth_entries()
    by_id = {e["id"]: e for e in entries}
    by_id["C3"]["text"] = "Synthetic drifted C3 text that is not the control."
    with pytest.raises(ValueError, match="IRRELEVANT_PROMPT"):
        load_candidate_instructions(_write(tmp_path, entries))


# --- collision with the static registry ------------------------------------------


def test_existing_id_with_different_content_rejected(synth_path, monkeypatch):
    # No A1..C20 id exists in the static registry today; simulate a future
    # drift where one does, with different content.
    squatter = Instruction(id="A7", category="style", text="squatter",
                           anti_text=None, checker="external",
                           checker_kwargs={"suite": "align"})
    monkeypatch.setitem(instructions.BY_ID, "A7", squatter)
    with pytest.raises(ValueError, match="already registered"):
        load_candidate_instructions(synth_path)


def test_identical_preregistration_is_idempotent(synth_path, monkeypatch):
    # Re-loading after the env-guarded registration must not trip the
    # collision check: identical content under a known id is a no-op.
    [a1] = [i for i in load_candidate_instructions(synth_path) if i.id == "A1"]
    monkeypatch.setitem(instructions.BY_ID, "A1", a1)
    assert len(load_candidate_instructions(synth_path)) == 60


# --- the env-guarded REGISTRY extension (fresh interpreter, real import path) ----


def _fresh_import(env_extra: dict[str, str], code: str, *,
                  check: bool = True) -> subprocess.CompletedProcess:
    """``code`` in a fresh interpreter on this checkout's src, the two guard
    variables unset unless ``env_extra`` sets them."""
    env = os.environ.copy()
    env.pop("SISTEER_LOAD_CANDIDATE_SIS", None)
    env.pop(PATH_ENV, None)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=REPO_ROOT,
        capture_output=True, text=True, check=check,
    )


def test_env_guard_registers_sixty(synth_path):
    out = _fresh_import(
        {"SISTEER_LOAD_CANDIDATE_SIS": synth_path},
        "from sisteer.data import instructions as I\n"
        "assert len({i.id for i in I.REGISTRY}) == len(I.REGISTRY)\n"  # unique
        "assert all(f'{p}{n}' in I.BY_ID for p in 'ABC' for n in range(1, 21))\n"
        "assert I.resolve_si_id('A1') == 'A1'\n"  # no SI_ALIASES shadowing
        "assert I.get('B1').checker == 'external'\n"
        "from sisteer.data.safety_texts import CONSTITUTION_PROMPT\n"
        "assert I.BY_ID['B2'].text == CONSTITUTION_PROMPT\n"
        "print('n_registry', len(I.REGISTRY))\n",
    ).stdout
    assert "n_registry" in out


def test_env_guard_flag_value_1_uses_default_path(synth_path):
    # SISTEER_LOAD_CANDIDATE_SIS=1 defers to SISTEER_CANDIDATE_SIS/default
    _fresh_import(
        {"SISTEER_LOAD_CANDIDATE_SIS": "1", PATH_ENV: synth_path},
        "from sisteer.data import instructions as I\n"
        "assert 'C20' in I.BY_ID\n",
    )


def test_bare_import_is_unchanged_when_env_unset():
    _fresh_import(
        {},
        "from sisteer.data import instructions as I\n"
        "assert not any(f'{p}{n}' in I.BY_ID for p in 'ABC' "
        "for n in range(1, 21))\n"
        "assert len({i.id for i in I.REGISTRY}) == len(I.REGISTRY)\n",
    )


def test_candidate_sis_importable_first_with_guard_enabled(synth_path):
    # with SISTEER_LOAD_CANDIDATE_SIS set, importing candidate_sis
    # BEFORE instructions used to raise ImportError (candidate_sis's top-level
    # `from .instructions import ...` triggered instructions' env-guarded
    # tail, which re-imported the half-initialized candidate_sis). The lazy
    # import makes BOTH orders work; this pins the order that broke.
    _fresh_import(
        {"SISTEER_LOAD_CANDIDATE_SIS": synth_path},
        "import sisteer.data.candidate_sis as C\n"          # candidate_sis FIRST
        "from sisteer.data import instructions as I\n"       # tail runs here
        "assert all(f'{p}{n}' in I.BY_ID for p in 'ABC' for n in range(1, 21))\n"
        f"assert len(C.load_candidate_instructions({synth_path!r})) == 60\n",
    )


def test_env_guard_reload_is_idempotent(synth_path):
    # A second explicit load+register in the same interpreter is a no-op
    # (_register_instruction idempotency), not a duplicate or a collision error.
    _fresh_import(
        {"SISTEER_LOAD_CANDIDATE_SIS": synth_path},
        "from sisteer.data import instructions as I\n"
        "from sisteer.data.candidate_sis import load_candidate_instructions\n"
        "n = len(I.REGISTRY)\n"
        f"for x in load_candidate_instructions({synth_path!r}):\n"
        "    I._register_instruction(x)\n"
        "assert len(I.REGISTRY) == n\n",
    )


# --- the REAL snapshot (only when SISTEER_CANDIDATE_SIS names it) ------------------

REAL = os.environ.get(PATH_ENV, "")
needs_real = pytest.mark.skipif(not (REAL and Path(REAL).is_file()),
                                reason=f"{PATH_ENV} does not name the candidate-SI snapshot")
#: the snapshot's file sha256 and the C16 text's sha256 the registry's si_text row carries
SNAPSHOT_SHA256 = "ba8efe82ece5a35a97e558e31ca9c2660c220bc3677a02193e4b4a126560d60e"
C16_TEXT_SHA256 = "ba214825889190f8b83ab705187e4eeb66174da3c4af36f95fd60d77e34d139a"


@needs_real
def test_real_snapshot_provenance():
    instrs = load_candidate_instructions(REAL, expect_sha=SNAPSHOT_SHA256)
    assert len(instrs) == 60
    assert len({hashlib.sha256(i.text.encode()).hexdigest()
                for i in instrs}) == 60
    by_id = {i.id: i for i in instrs}
    # B1 is the inlined full constitution, not a pointer
    b1 = by_id["B1"].text
    assert "<<" not in b1
    assert len(b1) > 100_000
    # the loader already asserted B2/C3 byte-equality; pin their sha16s too
    sha16 = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
    assert sha16(by_id["B2"].text) == "21e06f482a6559b3"
    assert sha16(by_id["C3"].text) == "71ff67aa2bc000f1"
    assert {i.category for i in instrs} == {"safety", "constitution",
                                            "irrelevant"}
    # C16, the short irrelevant control the refits distil
    c16 = by_id["C16"]
    assert c16.category == "irrelevant" and len(c16.text) == 517
    assert hashlib.sha256(c16.text.encode()).hexdigest() == C16_TEXT_SHA256


@needs_real
def test_c16_resolves_through_the_trainers_resolver_only_under_the_guard():
    """The fit's own resolver (sisteer.eval.jailbreak.resolve_instruction) names C16 when the guard is set, to the text whose
    sha256 the registry's si_text row carries; without the guard C16 is unknown."""
    code = ("import hashlib\n"
            "from sisteer.eval.jailbreak import resolve_instruction\n"
            "print(hashlib.sha256(resolve_instruction('C16').text.encode()).hexdigest())\n")
    out = _fresh_import({"SISTEER_LOAD_CANDIDATE_SIS": REAL}, code).stdout
    assert out.strip() == C16_TEXT_SHA256
    r = _fresh_import({}, code, check=False)
    assert r.returncode != 0 and "unknown system-instruction id 'C16'" in r.stderr


def test_without_the_snapshot_the_guard_refuses_by_name(tmp_path):
    """A named snapshot that does not exist, or the guard's "1" with no SISTEER_CANDIDATE_SIS, fails the import loudly, naming
    what is missing; it never registers a partial set."""
    missing = str(tmp_path / "absent.snapshot.json")
    for extra, needle in (({"SISTEER_LOAD_CANDIDATE_SIS": missing}, "absent.snapshot.json"),
                          ({"SISTEER_LOAD_CANDIDATE_SIS": "1"}, PATH_ENV)):
        r = _fresh_import(extra, "from sisteer.data import instructions", check=False)
        assert r.returncode != 0 and needle in r.stderr, (extra, r.stderr[-400:])
