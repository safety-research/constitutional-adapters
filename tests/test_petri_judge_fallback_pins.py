"""Petri JUDGE fallback chain: the hash-semantics pins, written AHEAD of the field.

The implementation landed separately (181ac97f4, cherry-picked onto the assembly branch); this file pinned
the contract it was delegated under BEFORE the field existed (as ``xfail(strict=True)``) so the cherry-pick was
checked, not trusted -- the marks came off when it landed and every pin below runs plain.

The delegated semantics (mirrored from ``auditor_fallback_models``):
  1. a NON-EMPTY judge chain is a harness-hash input (``role_models.judge_fallbacks``): a manifest with the chain
     keys differently from one without -> a fresh ``petri_run_key`` namespace (the auditor change already re-keys,
     both land in ONE tree, the wave re-keys once);
  2. ADDITIVE: an EMPTY chain enters neither the inputs nor the hash, so a spec declaring ``[]`` hashes
     byte-identically to a pre-field spec (every other campaign's frozen run keys stand);
  3. OMISSION = NO chain: no default factory re-instates one (that is how the abliterated Qwen would creep back);
     the chain is explicit in the manifest or absent;
  4. the loader accepts the key and the chain is stamped into the eval-log metadata beside the auditor's.

Also pinned: every Petri manifest the seedrep launch plan names carries ``auditor_fallback_models:
[openrouter/x-ai/grok-4.6]`` AND ``judge_fallback_models: [openrouter/x-ai/grok-4.6]``, and a
deterministic-refusing stub JUDGE run through the REAL chain code is scored by grok with its own provenance
record (``petri_judge_chain``), never silently."""
from __future__ import annotations

import asyncio
import copy
import re
import sys
from dataclasses import fields as dc_fields
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import auditor_fallback as af

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "modal"))
import seedrep_launch_plan as lp  # noqa: E402

from test_petri_auditor_fallback import _cell_under, _harness_inputs  # noqa: E402
from test_petri_store_shape import PROMPT_FILES, world  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
GROK = "openrouter/x-ai/grok-4.6"
OPUS48 = "anthropic/claude-opus-4-8"
#: opus-4-8 BEFORE grok in BOTH chains on every plan manifest, order pinned
PLAN_CHAIN = [OPUS48, GROK]
JUDGE_CHAIN = [GROK]          # the one-leg chain the contract pins were written against (still a valid chain)
FIELD = "judge_fallback_models"
#: the Petri manifests the seedrep launch plan names (three view engines + the Arditi engine)
PLAN_MANIFESTS = tuple(REPO / lp.VIEW_MANIFEST_REL.format(view=v) for v in ("s1", "s2", "s3")) + (
    REPO / lp.ARDITI_MANIFEST_REL,)



# --- contract pins (the semantics, checked against the landed field) ------------------------------------------------------------------

def test_judge_chain_is_a_harness_input_and_rekeys(world):
    """Pin 1: the chain enters ``role_models.judge_fallbacks`` beside ``auditor_fallbacks``; harness hash and run
    key both change -- the re-key named as the consequence."""
    base = world.spec
    chained = replace(base, **{FIELD: list(JUDGE_CHAIN)})
    cell_b = _cell_under(base, world)
    cell_c = _cell_under(chained, world)
    inputs = _harness_inputs(chained, world, cell_c)
    assert inputs["role_models"]["judge_fallbacks"] == JUDGE_CHAIN
    assert inputs["role_models"]["judge"] == base.judge_model
    assert cell_c.harness_config_hash != cell_b.harness_config_hash
    assert cell_c.petri_run_key != cell_b.petri_run_key
    without = copy.deepcopy(inputs)
    del without["role_models"]["judge_fallbacks"]
    assert without == _harness_inputs(base, world, cell_b), "the chain is the ONE difference"


def test_empty_judge_chain_keys_byte_identically_to_the_pre_field_inputs(world):
    """Pin 2 (ADDITIVE): ``[]`` enters neither the inputs nor the hash -- the inputs equal the pre-field mapping
    and the cell keys to the sha over that same mapping, so every recorded run key without a judge chain stands."""
    empty = replace(world.spec, **{FIELD: []})
    cell = _cell_under(empty, world)
    inputs = _harness_inputs(empty, world, cell)
    assert "judge_fallbacks" not in inputs["role_models"]
    pre_field = _harness_inputs(world.spec, world, cell)
    pre_field["role_models"].pop("judge_fallbacks", None)
    assert inputs == pre_field
    pre_field_hash = cm.sha256_hex16(cm.canonical_json(pre_field).encode("utf-8"))
    assert cell.harness_config_hash == pre_field_hash
    assert cell.petri_run_key == cm.petri_run_key(pre_field_hash, cell.cell_id, cell.intervention_hash)


def test_omitting_the_judge_chain_means_no_chain(world):
    """Pin 3: a spec built WITHOUT the field carries NO judge chain -- no default factory re-instates one -- and
    hashes identically to an explicit ``[]``."""
    omitted = world.spec  # the fixture never names the field
    assert getattr(omitted, FIELD) == []
    cell = _cell_under(omitted, world)
    assert _harness_inputs(omitted, world, cell) == _harness_inputs(replace(omitted, **{FIELD: []}), world, cell)


def test_loader_accepts_the_chain_and_it_is_stamped_beside_the_auditors(world, tmp_path):
    """Pin 4: a manifest declaring the chain loads with it; the eval-log metadata carries it beside the auditor
    chain, and a spec without it stamps NO judge-chain key (the same by-meaning rule as the auditor's stamp)."""
    from sisteer.capeval.tiers import petri_runner as pr

    src = PLAN_MANIFESTS[0]
    raw = yaml.safe_load(src.read_text())
    raw["petri"][FIELD] = list(JUDGE_CHAIN)
    p = tmp_path / "with_judge_chain.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    spec = cm.load_petri_spec(p)
    assert getattr(spec, FIELD) == JUDGE_CHAIN
    assert spec.auditor_fallback_models == PLAN_CHAIN
    cell = _cell_under(replace(world.spec, **{FIELD: list(JUDGE_CHAIN)}), world)
    md = pr.eval_log_metadata(replace(world.spec, **{FIELD: list(JUDGE_CHAIN)}), cell, "dir", world.bf, PROMPT_FILES)
    assert md[pr.AUDITOR_FALLBACK_METADATA_KEY] == world.spec.auditor_fallback_models
    judge_keys = [k for k, v in md.items() if v == JUDGE_CHAIN and k != pr.AUDITOR_FALLBACK_METADATA_KEY]
    assert judge_keys, f"no metadata key carries the judge chain: {sorted(md)}"
    md_none = pr.eval_log_metadata(world.spec, cell, "dir", world.bf, PROMPT_FILES)
    assert not any(k in md_none for k in judge_keys), "an absent chain stamps nothing (additive)"


@pytest.mark.parametrize("declaration", ["false", "null", '""', "{}", GROK, f"[{GROK}, '']", f"[{GROK}, 7]"])
def test_loader_refuses_a_judge_chain_that_is_not_a_list_of_models(tmp_path, declaration):
    """The auditor field's validation, mirrored: an explicitly present value must be a list of non-empty model
    strings; ``false`` / ``null`` / "" / {} / a scalar / a bad member are refusals naming the field."""
    src = PLAN_MANIFESTS[0]
    text = src.read_text()
    block = re.search(rf"^  {FIELD}:[^\n]*\n(?:    - [^\n]*\n)+", text, flags=re.M)
    assert block, "the plan manifest spells the judge chain"
    p = tmp_path / "bad.yaml"
    p.write_text(text[:block.start()] + f"  {FIELD}: {declaration}\n" + text[block.end():])
    # the MIRRORED message, not merely the field's name: today's unknown-key refusal also names the field, and
    # only the typed-value refusal proves the auditor field's validation was carried over
    with pytest.raises(cm.ManifestError, match=f"{FIELD} must be a list of non-empty model strings"):
        cm.load_petri_spec(p)


# --- pins that hold today --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("manifest", PLAN_MANIFESTS, ids=lambda p: p.name)
def test_every_plan_manifest_carries_the_grok_chain_for_both_roles(manifest):
    """sonnet -> grok for the auditor and opus -> grok for the judge on every Petri manifest the
    launch plan names -- each by its GENERATOR's rule, never a hand edit: the three view engines through the roster
    knob `views.<view>.petri_judge_fallback_models` (make_full22_configs), the Arditi engine through
    PETRI_FALLBACK_CHAINS (render_arditi_seedrep_configs)."""
    spec = cm.load_petri_spec(manifest)
    assert spec.auditor_model == "anthropic/claude-sonnet-5"
    assert spec.auditor_fallback_models == PLAN_CHAIN
    assert spec.judge_model == "anthropic/claude-opus-5"
    assert spec.judge_fallback_models == PLAN_CHAIN
    assert spec.judge_fallback_models[0] == OPUS48 and spec.judge_fallback_models[1] == GROK   # the ORDER is the pin (M11)


# --- a deterministic-refusing stub JUDGE through the REAL chain code ------------------------------------------------

class _StubJudge:
    """A judge leg that behaves deterministically: ``outcomes`` in order (an output object, or an exception)."""

    def __init__(self, name, outcomes):
        self.name, self.outcomes, self.calls = name, list(outcomes), []

    async def generate(self, input, **kwargs):
        self.calls.append((input, kwargs))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _refused():
    from types import SimpleNamespace
    return SimpleNamespace(stop_reason="content_filter", completion="I can't help with scoring this.", metadata=None)


def _verdict():
    from types import SimpleNamespace
    return SimpleNamespace(stop_reason="stop", completion='{"score": 7}', metadata=None)


def test_a_deterministically_refusing_judge_is_scored_by_grok_with_provenance_never_silently(world):
    """The landed chain (181ac97f4): opus-5 returns the SAME content_filter refusal on every call of
    one transcript (deterministic, like the classifier refusals found earlier); the chain switches to grok in
    place, grok's verdict is what the judge receives, the switch is recorded under the judge's OWN store key with
    the refusing model and the scoring model named, the auditor's record is untouched, and the switched leg stays
    current for the sample's next turn. A second transcript on the same controller starts on opus again only via a
    new sample state -- here the store is per sample, so provenance is per sample."""
    opus = _StubJudge("anthropic/claude-opus-5", [_refused(), _refused()])
    grok = _StubJudge(GROK, [_verdict(), _verdict()])
    store: dict = {}
    ctl = af.AuditorFallbackController(
        [af.AuditorLeg(opus, "anthropic/claude-opus-5", af.auditor_label("anthropic/claude-opus-5")),
         af.AuditorLeg(grok, GROK, af.auditor_label(GROK))],
        state_getter=lambda: ("scenario-1/epoch-1", store), role="judge",
        store_key=af.JUDGE_CHAIN_STORE_KEY, switch_on_refusal_output=True)
    out = asyncio.run(ctl.generate(["<transcript>"]))
    assert out.stop_reason == "stop" and out.completion == '{"score": 7}'
    rec = store[af.JUDGE_CHAIN_STORE_KEY]
    assert af.AUDITOR_CHAIN_STORE_KEY not in store
    assert rec["display"].startswith("opus-5") and "grok-4.6" in rec["display"]
    assert [sw["reason"] for sw in rec["switches"]] == ["refusal_output"]
    assert rec["failures"][0]["model"] == "anthropic/claude-opus-5"
    assert "content_filter" in rec["failures"][0]["error"]
    assert out.metadata[af.JUDGE_CHAIN_STORE_KEY]["current_index"] == 1
    # the next turn of the same sample stays on grok: opus is never re-asked to refuse again
    asyncio.run(ctl.generate(["<transcript>", "<turn 2>"]))
    assert len(opus.calls) == 1 and len(grok.calls) == 2
    # and when grok refuses too, the refusal is handed back AS IS (a terminal judge refusal, filed by name), never a score
    opus2 = _StubJudge("anthropic/claude-opus-5", [_refused()])
    grok2 = _StubJudge(GROK, [_refused()])
    store2: dict = {}
    ctl2 = af.AuditorFallbackController(
        [af.AuditorLeg(opus2, "anthropic/claude-opus-5", "opus-5"), af.AuditorLeg(grok2, GROK, "grok-4.6")],
        state_getter=lambda: ("scenario-2/epoch-1", store2), role="judge",
        store_key=af.JUDGE_CHAIN_STORE_KEY, switch_on_refusal_output=True)
    out2 = asyncio.run(ctl2.generate(["<transcript>"]))
    assert out2.stop_reason == "content_filter"
    assert store2[af.JUDGE_CHAIN_STORE_KEY]["final_failure"]["all_legs_exhausted"] is True


def test_metadata_carries_both_chains_for_the_plan_manifests(world):
    """The eval-log stamp names BOTH chains on a plan manifest's spec, beside each other; the auditor's key is the
    one every prior log verified under, the judge's is new and absent when the chain is."""
    from dataclasses import replace as _replace
    from sisteer.capeval.tiers import petri_runner as pr

    spec = _replace(world.spec, auditor_fallback_models=[GROK], **{FIELD: list(JUDGE_CHAIN)})
    cell = _cell_under(spec, world)
    md = pr.eval_log_metadata(spec, cell, "dir", world.bf, PROMPT_FILES)
    assert md[pr.AUDITOR_FALLBACK_METADATA_KEY] == [GROK]
    assert md[pr.JUDGE_FALLBACK_METADATA_KEY] == JUDGE_CHAIN
    plain = _replace(world.spec, **{FIELD: []})
    assert pr.JUDGE_FALLBACK_METADATA_KEY not in pr.eval_log_metadata(plain, _cell_under(plain, world), "dir", world.bf, PROMPT_FILES)
