"""The Petri TARGET decode contract is a MANIFEST declaration the
standup hashes into cells_served.json, so the generator emits it — `target_decode: {mode: vendor, policy: '<ref@v1>'}` after
`prompted_prefix_hashes:` — for exactly the roster views that declare `petri_target_decode: '<ref>'` (the lane's three seed views of
qwen35_4b), and the lane's misalign manifests (derived from those) carry it too. Pinned here: the three s-view manifests + their misalign
twins load through the manifest loader as the vendor block with the exact ref; EVERY other generated manifest carries no block and renders
byte-identically whether or not the knob exists in the roster; the knob refuses a value that is not a versioned `petri` decode-policy
ref (a bon row, a versionless ref, a bare mode word, a non-string) by name; a knob on a foreign view renders the block there and nowhere
else. Everything read off the tree's roster + the generator, nothing typed twice."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import make_full22_configs as gen  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402

ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text(encoding="utf-8"))
PINS = gen.load_pins() if hasattr(gen, "load_pins") else __import__("json").loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
TARGET = "qwen35_4b"
SEED_VIEWS = ("s1", "s2", "s3")
POLICY = "qwen35_4b_it|petri|off|vendor_preset@v1"
BLOCK = ("  target_decode:                  # the vendor row on every TARGET request; a harness-hash input\n"
         "    mode: vendor\n"
         f"    policy: '{POLICY}'\n")
#: the seed views' OTHER per-view Petri knob: the judge fallback chain, rendered as its own block after the auditor chain
JUDGE_KNOB = "petri_judge_fallback_models"
AUDITOR_KNOB = "petri_auditor_fallback_models"
JUDGE_BLOCK = ("  judge_fallback_models:          # grok-4.6 backs the JUDGE too (its returned content_filter refusal switches legs); a harness-hash input\n"
               "    - anthropic/claude-opus-4-8\n"
               "    - openrouter/x-ai/grok-4.6\n")
#: the seed views' auditor chain REPLACES the template's one-leg grok chain -- opus-4-8 before grok, order pinned
AUDITOR_BLOCK = ("  auditor_fallback_models:        # claude-opus-4-8 BEFORE grok-4.6, order pinned; a harness-hash input\n"
                 "    - anthropic/claude-opus-4-8\n"
                 "    - openrouter/x-ai/grok-4.6\n")
DEFAULT_AUDITOR_BLOCK = "  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n"


def _every_view():
    for target, t in ROSTER["targets"].items():
        for view in (t.get("views") or {}):
            yield target, view


def test_the_three_seed_views_declare_the_knob_and_their_manifests_carry_the_block_the_loader_reads():
    for view in SEED_VIEWS:
        v = ROSTER["targets"][TARGET]["views"][view]
        assert v["petri_target_decode"] == POLICY == gen.view_petri_target_decode(ROSTER, TARGET, view)
        text = (REPO / v["manifest"]).read_text(encoding="utf-8")
        assert gen.render_manifest(ROSTER, TARGET, view) == text                      # the committed bytes are the generator's
        assert text.count(BLOCK) == 1 and text.index("  prompted_prefix_hashes:") < text.index(BLOCK) < text.index("  bloom_options:")
        spec = cm.load_petri_spec(REPO / v["manifest"])                                # the manifest loader: the block, the exact ref
        assert spec.target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=None)
        assert cm.petri_target_decode_mode(spec) == "vendor"
        # the config is untouched by the knob (the manifest is the authority the standup hashes)
        assert gen.render_config(ROSTER, PINS, TARGET, view) == (REPO / v["config"]).read_text(encoding="utf-8")
        assert "target_decode" not in (REPO / v["config"]).read_text(encoding="utf-8")
        # the lane's misalign twin is derived from this manifest's text and carries the same block
        twin = REPO / f"configs/capeval/manifests/misalign_full22_{TARGET}_{view}.yaml"
        assert twin.read_text(encoding="utf-8").count(BLOCK) == 1
        assert cm.load_petri_spec(twin).target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=None)


def test_every_other_manifest_carries_no_block_and_is_byte_identical_with_or_without_the_knob():
    without = copy.deepcopy(ROSTER)
    for view in SEED_VIEWS:
        del without["targets"][TARGET]["views"][view]["petri_target_decode"]
        del without["targets"][TARGET]["views"][view][JUDGE_KNOB]
        del without["targets"][TARGET]["views"][view][AUDITOR_KNOB]
    n = 0
    for target, view in _every_view():
        if target == TARGET and view in SEED_VIEWS:
            continue
        assert gen.view_petri_target_decode(ROSTER, target, view) is None, (target, view)
        assert gen.view_petri_judge_fallback_models(ROSTER, target, view) is None, (target, view)
        assert gen.view_petri_auditor_fallback_models(ROSTER, target, view) is None, (target, view)
        text = (REPO / ROSTER["targets"][target]["views"][view]["manifest"]).read_text(encoding="utf-8")
        assert "target_decode" not in text and "judge_fallback_models" not in text, (target, view)
        # every other manifest keeps the template's one-leg grok AUDITOR chain (the two-leg chain is the seed views' alone;
        # `scenarios_model: anthropic/claude-opus-4-8` is a different field every manifest carries)
        assert DEFAULT_AUDITOR_BLOCK in text and AUDITOR_BLOCK not in text, (target, view)
        assert gen.render_manifest(ROSTER, target, view) == gen.render_manifest(without, target, view) == text, (target, view)
        assert cm.load_petri_spec(REPO / ROSTER["targets"][target]["views"][view]["manifest"]).target_decode is None
        n += 1
    assert n == sum(1 for _ in _every_view()) - 3 >= 60
    # ... and the three seed views WITHOUT the knobs render exactly the committed text minus the two blocks (the knobs add their
    # blocks, nothing else); each knob alone adds exactly its own block
    for view in SEED_VIEWS:
        committed = (REPO / ROSTER["targets"][TARGET]["views"][view]["manifest"]).read_text(encoding="utf-8")
        assert committed.count(BLOCK) == 1 and committed.count(JUDGE_BLOCK) == 1 and committed.count(AUDITOR_BLOCK) == 1
        bare = committed.replace(BLOCK, "").replace(JUDGE_BLOCK, "").replace(AUDITOR_BLOCK, DEFAULT_AUDITOR_BLOCK)
        assert gen.render_manifest(without, TARGET, view) == bare
        only_decode = copy.deepcopy(ROSTER)
        del only_decode["targets"][TARGET]["views"][view][JUDGE_KNOB]; del only_decode["targets"][TARGET]["views"][view][AUDITOR_KNOB]
        assert gen.render_manifest(only_decode, TARGET, view) == committed.replace(JUDGE_BLOCK, "").replace(AUDITOR_BLOCK, DEFAULT_AUDITOR_BLOCK)
        only_chains = copy.deepcopy(ROSTER); del only_chains["targets"][TARGET]["views"][view]["petri_target_decode"]
        assert gen.render_manifest(only_chains, TARGET, view) == committed.replace(BLOCK, "")
        spec = cm.load_petri_spec(REPO / ROSTER["targets"][TARGET]["views"][view]["manifest"])
        assert spec.judge_fallback_models == spec.auditor_fallback_models == ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"]
    # every other misalign manifest on the tree carries no block either
    for p in sorted((REPO / "configs/capeval/manifests").glob("misalign_full22_*.yaml")):
        if p.name in {f"misalign_full22_{TARGET}_{v}.yaml" for v in SEED_VIEWS}:
            continue
        assert "target_decode" not in p.read_text(encoding="utf-8"), p.name


@pytest.mark.parametrize("bad, phrase", [
    ("qwen35_4b_it|bon|off|vendor_preset@v1", "not a versioned `petri` decode-policy ref"),       # another eval's row
    ("qwen35_4b_it|petri|off|vendor_preset", "not a versioned `petri` decode-policy ref"),         # no @v<N>
    ("vendor", "not a versioned `petri` decode-policy ref"),                                       # a bare mode word
    ("qwen35_4b_it|petri|off|vendor_preset@v0", "not a versioned `petri` decode-policy ref"),      # v0 is not a version
    ("", "non-empty string"),
    (1, "non-empty string"),
    ({"mode": "vendor"}, "non-empty string"),
])
def test_the_knob_refuses_anything_but_a_versioned_petri_ref_by_name(bad, phrase):
    r = copy.deepcopy(ROSTER)
    r["targets"][TARGET]["views"]["s1"]["petri_target_decode"] = bad
    with pytest.raises(SystemExit, match=phrase):
        gen.view_petri_target_decode(r, TARGET, "s1")
    with pytest.raises(SystemExit, match=phrase):
        gen.render_manifest(r, TARGET, "s1")


def test_a_knob_on_another_view_renders_the_block_there_and_the_loader_admits_it(tmp_path):
    r = copy.deepcopy(ROSTER)
    r["targets"][TARGET]["views"]["it"]["petri_target_decode"] = POLICY
    text = gen.render_manifest(r, TARGET, "it")
    assert text.count(BLOCK) == 1 and text.replace(BLOCK, "") == gen.render_manifest(ROSTER, TARGET, "it")
    p = tmp_path / "it.yaml"
    p.write_text(text, encoding="utf-8")
    assert cm.load_petri_spec(p).target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=None)
    # the generator's constants are the loader's vocabulary
    assert gen.PETRI_TARGET_DECODE_MODE == cm.PETRI_TARGET_DECODE_VENDOR == "vendor"



@pytest.mark.parametrize("bad", [[], "openrouter/x-ai/grok-4.6", [""], [None], ["grok"], ["openrouter/x-ai/grok-4.6", "openrouter/x-ai/grok-4.6"]])
def test_the_judge_chain_knob_refuses_a_malformed_value_by_name(bad):
    """The judge-chain knob mirrors the decode knob's stance: a value that is not a NON-EMPTY list of `<provider>/<model>` strings (or repeats
    a model) refuses naming the view, so a mistyped roster never renders a block the loader would refuse at standup."""
    for knob in (JUDGE_KNOB, AUDITOR_KNOB):
        r = copy.deepcopy(ROSTER)
        r["targets"][TARGET]["views"]["s1"][knob] = bad
        with pytest.raises(SystemExit, match=rf"s1: views\.s1\.{knob}"):
            gen.render_manifest(r, TARGET, "s1")
