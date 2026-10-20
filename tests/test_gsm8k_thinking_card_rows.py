"""The GSM8K front door's `--sampling thinking_card` table (``contracts.THINKING_SAMPLING_BY_MODEL_KEY``) and the served driver's
model-key pin behind it (ruling `decode-thinking-card-rows-2026-09-09`).

The u0 gsm8k units of a front-door model's wave once stood up cleanly and then had their DRIVER refuse within seconds:
`--sampling thinking_card: model key '<unpinned>' has no verified sampling row -- no row in THINKING_SAMPLING_BY_MODEL_KEY`.
Two rules, pinned here in EXECUTION form (the driver's own functions, the committed export):

1. `served_capability_run._sampling_model_key` reads BOTH geometry maps -- the sweep's `MODEL_HF_IDS` first, then the front door's
   `ONBOARDED_MODEL_HF_IDS` -- exact and case-insensitive, never a mirror or a family guess. The front door's demo key
   (``qwen35_4b_demo``, reproduce/models.yaml) serves the paper's own Qwen3.5-4B checkpoint, so its served HF id pins the PAPER
   key ``qwen35_4b`` and the driver resolves the 4B's thinking_card row for it; a checkpoint no key pins stays ``None``.
2. The table is the export's thinking_card mapping RESTRICTED to the keys contracts documents (its provenance prose): exactly the
   thinking-ON experiment's four. A model with no `sampling_mode_policy (model_key, thinking_card)` mapping refuses by name; the
   demo key refuses too even though the registry now maps it (the registry owner's follow-up of 2026-10-09 seeded its `(qwen35_4b_demo,
   thinking_card)` twin of the 4B's entry): the table is restricted to the documented keys, and the served driver never asks for it (see 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.registry import decode_policy as DP  # noqa: E402

EXPORT = REPO / "configs" / "registry" / "decode_policy.json"
PACKAGE_EXPORT = REPO / "src" / "sisteer" / "registry" / "decode_policy.json"

#: the experiment's four -- their rows must not move
FOUR = ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
#: the front door's demo key and the paper's key whose checkpoint it serves
DEMO_KEY, PAPER_KEY, SERVED_HF = "qwen35_4b_demo", "qwen35_4b", "Qwen/Qwen3.5-4B"
GSM = tc.GSM8K_STANDARD_TASK


@pytest.fixture(scope="module")
def pf():
    return DP.PolicyFile.load(str(EXPORT))


# --------------------------------------------------------------------------------------------
# 1. the driver's model-key pin
# --------------------------------------------------------------------------------------------


def test_the_drivers_model_key_pin_reads_both_geometry_maps_and_the_paper_key_wins_a_shared_checkpoint():
    assert DEMO_KEY in sg.ONBOARDED_MODEL_HF_IDS and DEMO_KEY not in sg.MODEL_HF_IDS and DEMO_KEY in sg.KNOWN_MODEL_KEYS
    assert sg.ONBOARDED_MODEL_HF_IDS[DEMO_KEY] == sg.MODEL_HF_IDS[PAPER_KEY] == SERVED_HF      # the demo serves the paper's checkpoint
    assert srun._sampling_model_key(SERVED_HF) == PAPER_KEY                                   # the sweep's map is read first
    assert srun._sampling_model_key(SERVED_HF.upper()) == PAPER_KEY                           # case-insensitive, as before
    # a front-door key whose HF id is NOT a paper model's pins ITSELF through the second map (the rule the demo key rides on)
    for key, hf in sg.ONBOARDED_MODEL_HF_IDS.items():
        if hf not in sg.MODEL_HF_IDS.values():
            assert srun._sampling_model_key(hf) == key, key
    # the sweep's keys resolve exactly as before
    assert srun._sampling_model_key("google/gemma-4-E4B-it") == "gemma4e4b"
    assert srun._sampling_model_key("meta-llama/Llama-3.1-8B") is None                        # not a pinned id: never a near-match
    assert srun._sampling_model_key("Qwen/Qwen3.5-4B-Base") is None                           # the demo's FIT checkpoint pins no key
    assert srun._sampling_model_key(None) is None and srun._sampling_model_key("") is None


def test_the_drivers_own_sampling_resolution_returns_the_papers_row_for_the_demo_keys_served_checkpoint():
    args = SimpleNamespace(sampling=tc.SAMPLING_THINKING_CARD, samples=None)
    r = srun._resolve_sampling(args, {"model": SERVED_HF}, [tc.CONTROL_GSM8K], {GSM: "on"})
    assert r.mode == tc.SAMPLING_THINKING_CARD and r.model_key == PAPER_KEY
    assert r.row is tc.THINKING_SAMPLING_BY_MODEL_KEY[PAPER_KEY] is tc.sampling_for(PAPER_KEY, tc.SAMPLING_THINKING_CARD)
    assert (r.row.temperature, r.row.top_p, r.row.top_k, r.row.min_p, r.row.presence_penalty) == (0.6, 0.95, 20, 0.0, 0.0)
    assert r.samples_by_task == {GSM: 1} and r.samples == 1


def test_an_unpinned_model_and_a_key_with_no_mapping_still_refuse_by_name():
    args = SimpleNamespace(sampling=tc.SAMPLING_THINKING_CARD, samples=None)
    with pytest.raises(SystemExit) as e:
        srun._resolve_sampling(args, {"model": "someone/not-a-pinned-model"}, [tc.CONTROL_GSM8K], {GSM: "on"})
    assert "'<unpinned>'" in str(e.value) and "THINKING_SAMPLING_BY_MODEL_KEY" in str(e.value)
    # the FIT checkpoint is not the served model: it pins no key and refuses the same way
    with pytest.raises(SystemExit, match="<unpinned>"):
        srun._resolve_sampling(args, {"model": "Qwen/Qwen3.5-4B-Base"}, [tc.CONTROL_GSM8K], {GSM: "on"})
    # a key with no thinking_card mapping refuses by name -- the demo key among them (no registry mapping on this tree)
    for foreign in ("llama8b", "qwen35_9b", "gemma4e2b", DEMO_KEY, "<unpinned>"):
        with pytest.raises(tc.SamplingUnverified, match="THINKING_SAMPLING_BY_MODEL_KEY"):
            tc.sampling_for(foreign, tc.SAMPLING_THINKING_CARD)
    # and no other table-backed mode carries a row for the demo key (nothing seeded, nothing invented)
    for mode in (tc.SAMPLING_CARD, tc.SAMPLING_GENERAL, tc.SAMPLING_BATTERY, tc.SAMPLING_VENDOR_PRESET):
        with pytest.raises(tc.SamplingUnverified):
            tc.sampling_for(DEMO_KEY, mode)


# --------------------------------------------------------------------------------------------
# 2. the table: the four, restricted from the export by the documented keys
# --------------------------------------------------------------------------------------------


def test_the_thinking_card_table_is_exactly_the_four_and_the_restriction_rule_holds(pf):
    t = tc.THINKING_SAMPLING_BY_MODEL_KEY
    assert set(t) == set(FOUR) == set(tc._THINKING_SOURCES)
    # the export's thinking_card mapping RESTRICTED to the documented keys: the registry may keep rows for models the harness no
    # longer launches, and such a key is not a harness row; every documented key IS mapped (the builder refuses otherwise)
    mapped = set(pf.mode_rows(tc.SAMPLING_THINKING_CARD))
    assert set(t) <= mapped and set(t) == mapped & set(tc._THINKING_SOURCES)
    assert DEMO_KEY in mapped and DEMO_KEY not in t                       # the registry maps the demo key (its twin of the 4B's entry); the
    assert pf.mode_rows(tc.SAMPLING_THINKING_CARD)[DEMO_KEY].policy_id == f"{DEMO_KEY}|capability|on|thinking_card@v1"   # restriction keeps it out
    assert (t["gemma4e4b"].temperature, t["gemma4e4b"].top_p, t["gemma4e4b"].top_k) == (1.0, 0.95, 64)
    assert (t["qwen35_4b"].temperature, t["qwen35_4b"].top_p, t["qwen35_4b"].top_k,
            t["qwen35_4b"].min_p, t["qwen35_4b"].presence_penalty) == (0.6, 0.95, 20, 0.0, 0.0)
    assert (t["glm46vflash"].temperature, t["glm46vflash"].top_p, t["glm46vflash"].top_k) == (0.8, 0.6, 2)
    assert (t["nemotron3nano4b"].temperature, t["nemotron3nano4b"].top_p, t["nemotron3nano4b"].top_k) == (1.0, 0.95, None)
    for key in FOUR:
        assert t[key].repetition_penalty is None
        assert t[key].source.strip() and len(t[key].source) > 40, key          # provenance prose, never blank
    # the identity hashes NUMBERS, never the prose: the 4B's thinking-precise row is a different instrument from gemma's
    one = {GSM: 1}
    assert tc.sampling_identity(t["gemma4e4b"], one) != tc.sampling_identity(t["qwen35_4b"], one)


def test_both_export_copies_agree_and_every_harness_row_is_mapped_by_the_export():
    with open(EXPORT, encoding="utf-8") as fh, open(PACKAGE_EXPORT, encoding="utf-8") as gh:
        a, b = json.load(fh), json.load(gh)
    assert a == b and a["header"]["policy_set_sha256"] == DP.policy_set_sha256(a)
    modes = {str(m.get("model_key") or m["model_id"]) for m in a["sampling_mode_policy"] if m["sampling_mode"] == tc.SAMPLING_THINKING_CARD}
    assert set(tc.THINKING_SAMPLING_BY_MODEL_KEY) == set(FOUR) <= modes          # every harness row is mapped by the export
    assert not (modes - set(tc.THINKING_SAMPLING_BY_MODEL_KEY)) & set(tc._THINKING_SOURCES)   # and no documented key is left out of the table
