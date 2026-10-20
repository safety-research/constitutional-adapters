"""Per-family unit tests for the held-out attack families + the frozen-corpus
ingested-kind adapter. CPU-only, model-free, all-synthetic (invented benign goals) per
the repo's dual-use containment rule.

The shared replay/persist/resume contract is covered by
``test_heldout_conformance`` (data-driven over every family) and by the static
contract helper below; these tests assert the FAMILY-SPECIFIC transform is what
the paper describes and that the ingested-kind adapter maps/gates/pins correctly.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from attacks_static_common import StubDefense, assert_static_contract  # noqa: E402

from sisteer.eval.attacks import artprompt, cipherchat, codeattack, drattack  # noqa: E402
from sisteer.eval.attacks import frozen_corpus as fc  # noqa: E402
from sisteer.eval.attacks.heldout_common import (  # noqa: E402
    LeakageError,
    training_fingerprint,
)
from sisteer.eval.jailbreak_data import Goal  # noqa: E402

G = Goal("g0", "Assemble the reticulating splines", "illegal", "harmbench", "rev")
FP = training_fingerprint(["totally unrelated training prompt"])


# --------------------------------------------------------------------------- #
# deterministic build kind families                                                             #
# --------------------------------------------------------------------------- #

def test_artprompt_masks_trigger_and_renders_ascii_art():
    v = dict(artprompt.artprompt_variants(G))
    assert list(v) == ["ascii_art_word", "ascii_art_letters"]
    # the selected trigger (longest word: "reticulating") is masked and drawn
    assert "[MASK]" in v["ascii_art_word"]
    assert "reticulating" not in v["ascii_art_word"].lower().split("ascii art")[0]
    assert "#" in v["ascii_art_word"]  # actual ASCII art present
    # determinism
    assert artprompt.artprompt_variants(G) == artprompt.artprompt_variants(G)
    # contract (resume, 0700, variant_key, pool-builder keying)
    assert_static_contract(artprompt.run_artprompt, "artprompt",
                           expected_variant_names=["ascii_art_word",
                                                   "ascii_art_letters"])


def test_cipherchat_four_ciphers_roundtrip():
    v = dict(cipherchat.cipherchat_variants(G))
    assert list(v) == ["caesar", "morse", "ascii", "selfcipher"]
    # caesar is a real shift-3 (decipher with shift 23 recovers the text)
    assert cipherchat._caesar(cipherchat._caesar(G.text, 3), 23) == G.text
    assert cipherchat._caesar(G.text, 3) in v["caesar"]
    assert "." in v["morse"] and "-" in v["morse"]
    assert any(str(ord(c)) in v["ascii"] for c in G.text if c.isalpha())
    # SelfCipher carries the PLAIN request (its whole point)
    assert G.text in v["selfcipher"]
    assert_static_contract(cipherchat.run_cipherchat, "cipherchat",
                           expected_variant_names=["caesar", "morse", "ascii",
                                                   "selfcipher"])


def test_codeattack_three_code_framings():
    v = dict(codeattack.codeattack_variants(G))
    assert list(v) == ["docstring_completion", "list_fill", "stack_simulation"]
    assert G.text in v["docstring_completion"]
    assert repr(G.text) in v["list_fill"]
    assert repr(G.text) in v["stack_simulation"]
    assert_static_contract(codeattack.run_codeattack, "codeattack")


def test_drattack_decomposition():
    v = dict(drattack.drattack_variants(G))
    assert list(v) == ["word_substitution", "phrase_split"]
    # word substitution assigns one variable per token
    assert "w1 = " in v["word_substitution"] and "w2 = " in v["word_substitution"]
    assert "Fragment A:" in v["phrase_split"] and "Fragment B:" in v["phrase_split"]
    assert_static_contract(drattack.run_drattack, "drattack")


# --------------------------------------------------------------------------- #
# rewriter build kind build -> freeze -> replay + loud-skip when unbuilt                   #
# --------------------------------------------------------------------------- #

class _Stub:
    model = "stub"

    def batch(self, items):
        return ["BENIGN_XFORM_" + str(i) for i, _ in enumerate(items)]


def test_class_b_loud_skip_when_corpus_absent(tmp_path, monkeypatch):
    import sisteer.eval.attacks.multilingual as ml

    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(tmp_path / "empty"))
    ml._CORPUS_CACHE.clear()
    res = ml.run_multilingual(StubDefense(), [G], out_dir=tmp_path / "run")
    assert res["skipped"] is True and res["family"] == "multilingual"


def test_class_b_build_freeze_replay(tmp_path, monkeypatch):
    import sisteer.eval.attacks.renellm as rn

    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(tmp_path))
    rn._CORPUS_CACHE.clear()
    man = rn.build_renellm_corpus([G], rewriter=_Stub(), fingerprint=FP)
    assert man["build_kind"] == "rewriter"
    assert man["row_count"] == 3  # three nested scenarios
    assert man["provenance"]["rewriter_model"] == "stub"
    # the frozen file exists and the manifest sidecar was written
    assert (tmp_path / "renellm.jsonl").exists()
    assert (tmp_path / "renellm.jsonl.manifest.json").exists()
    rn._CORPUS_CACHE.clear()
    names = [n for n, _ in rn.renellm_variants(G)]
    assert names == ["code_completion", "table_filling", "text_continuation"]


# --------------------------------------------------------------------------- #
# the generic frozen-corpus (ingested-kind) adapter                                  #
# --------------------------------------------------------------------------- #

def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_mapping_spec_maps_behavior_and_prompt_columns(tmp_path):
    art = _write_jsonl(tmp_path / "artifact.jsonl", [
        {"BehaviorID": "b1", "adv_prompt": "attack one"},
        {"BehaviorID": "b1", "adv_prompt": "attack one variant two"},
        {"BehaviorID": "b2", "adv_prompt": "attack two"},
    ])
    spec = fc.MappingSpec(prompt_col="adv_prompt", behavior_col="BehaviorID")
    rows = fc.artifact_to_rows(art, spec)
    assert {r["goal_id"] for r in rows} == {"b1", "b2"}
    b1 = sorted(r["sample_idx"] for r in rows if r["goal_id"] == "b1")
    assert b1 == [0, 1]  # two variants numbered by appearance


def test_missing_prompt_column_raises(tmp_path):
    art = _write_jsonl(tmp_path / "a.jsonl", [{"x": "y"}])
    with pytest.raises(ValueError, match="prompt column"):
        fc.artifact_to_rows(art, fc.MappingSpec(prompt_col="nope"))


def test_content_hash_pin_mismatch_refuses(tmp_path):
    art = _write_jsonl(tmp_path / "a.jsonl", [{"p": "x", "b": "g1"}])
    rows = fc.artifact_to_rows(art, fc.MappingSpec(prompt_col="p", behavior_col="b"))
    with pytest.raises(ValueError, match="content-hash mismatch"):
        fc.FrozenCorpus.from_rows(rows, expected_sha="deadbeef")


def test_ingest_artifact_gates_and_manifests(tmp_path):
    art = _write_jsonl(tmp_path / "a.jsonl", [
        {"p": "benign attack alpha", "b": "g1"},
        {"p": "benign attack beta", "b": "g2"},
    ])
    out = tmp_path / "frozen.jsonl"
    man = fc.ingest_artifact(
        art, fc.MappingSpec(prompt_col="p", behavior_col="b"), out,
        fingerprint=FP, provenance={"artifact": "synthetic_demo"})
    assert man["build_kind"] == "ingested"
    # the refusing gate is DEMOTED to a report-only record (2026-08-25
    # directive, applied at the phase-C join)
    assert man["leakage_check"]["gate"] == "removed_by_directive_2026_08_25"
    assert out.exists() and Path(str(out) + ".manifest.json").exists()
    # replay binding round-trips the prompts
    fam = fc.make_frozen_corpus_family(
        "x", out, provenance={"artifact": "synthetic_demo"})
    v = fam.variants_fn(Goal("g1", "", "c", "d", "r"))
    assert v and v[0][1] == "benign attack alpha"


def test_ingest_artifact_records_gate_removal_instead_of_refusing(tmp_path):
    # DEMOTED (2026-08-25 directive): overlap with the training fingerprint
    # no longer refuses — families are allowed on both sides, so the
    # manifest RECORDS the gate's removal instead of raising LeakageError.
    leak_fp = training_fingerprint(["THE LEAKED PROMPT"])
    art = _write_jsonl(tmp_path / "a.jsonl", [{"p": "the leaked prompt", "b": "g1"}])
    man = fc.ingest_artifact(art, fc.MappingSpec(prompt_col="p", behavior_col="b"),
                             tmp_path / "o.jsonl", fingerprint=leak_fp,
                             provenance={"artifact": "x"})
    assert man["leakage_check"]["gate"] == "removed_by_directive_2026_08_25"
    assert (tmp_path / "o.jsonl").exists()


def test_generic_frozen_corpus_unconfigured_is_loud(tmp_path, monkeypatch):
    monkeypatch.delenv("SISTEER_FROZEN_CORPUS_PATH", raising=False)
    fc._ENV_CACHE.clear()
    with pytest.raises(RuntimeError, match="frozen_corpus family requires"):
        fc.run_frozen_corpus(StubDefense(), [G], out_dir=tmp_path / "run")


def test_frozen_demo_pinned_sha_matches():
    import sisteer.eval.attacks.frozen_demo as fd
    from sisteer.eval.attacks.heldout_common import corpus_content_hash

    rows = [json.loads(x) for x in fd._CORPUS.read_text().splitlines() if x.strip()]
    assert corpus_content_hash(rows) == fd.PINNED_SHA
