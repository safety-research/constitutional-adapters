"""A dose ladder BORROWS the s1 static store and matched its sampled configs to this tree's s1 `sampled_decode` block byte for byte
except `seed_base`, so the regeneration of `configs/sdfbase_full22_qwen35_4b_s{1,2,3}_sampled_t0.yaml` may change EXACTLY two keys of
that block versus the committed predecessor (commit 5fa280ef): `seed_base` (the retired hashed bases 4057844608 / 2226795507 /
2554970601 -> the literal training seeds 1 / 2 / 3) and `n_trials` (2 -> 1; the `_t1` files deleted). T 0.7 / top_p 0.8 / top_k 20 /
min_p 0.0 / presence_penalty 1.5 / repetition_penalty 1.0 / static cap 4096 / controls cap 512 / arm vendor_recommended / source /
applies_to / the readouts sub-block are byte-identical. The predecessor blocks are VENDORED here as literals (read off
`git show 5fa280ef:<path>`), never re-derived — a test that read them from git would skip on a shallow clone; the known-bad is a
regeneration that also alters `presence_penalty`, which the diff must name."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "full22" / "sampled", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import make_sampled_configs as msc  # noqa: E402
import eval_seed as es  # noqa: E402

PRESETS = msc.load_presets()
ALLOWED_CHANGED = {"seed_base", "n_trials"}
SOURCE = ("Qwen/Qwen3.5-4B @ 851bf6e8: NO generation_config.json ships (.no_exist marker); model card README \"Best Practices\", "
          "instruct/non-thinking mode, GENERAL tasks -- temperature 0.7, top_p 0.8, top_k 20, min_p 0.0, presence_penalty 1.5, "
          "repetition_penalty 1.0 (the card's stated no-op value); the only preset of the four with an anti-repetition term; read on "
          "cluster A 2026-09-09 (DESIGN.md 'Verified sampler presets')")
THINKING_SOURCE = ("Qwen/Qwen3.5-4B @ 851bf6e8: model card README \"Best Practices\", THINKING mode, general tasks -- temperature 1.0, "
                   "top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0 (the card's stated no-op value); the "
                   "vendor's thinking row, as reported to the project (adopted by the second amendment, 2026-09-09)")
#: the `sampled_decode` block of each committed predecessor at 5fa280ef (W), AS DATA — the base t0 config of each seed view
PREDECESSOR = {
    s: {"arm": "vendor_recommended", "trial": 0, "n_trials": 2, "seed_base": base,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        "source": SOURCE, "source_kind": "model_card", "applies_to": ["static_families", "controls"],
        "static_cap": 4096, "controls_cap": 512,
        "readouts": {"gsm8k": {"cap": 8192, "thinking": "on",
                               "preset": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
                               "source": THINKING_SOURCE},
                     "mmlu": {"instrument": "letter_logprob_argmax",
                              "note": "sampler-moot: teacher-forced letter-logprob argmax; the sampled run measures server-to-server logprob nondeterminism only"}}}
    for s, base in ((1, 4057844608), (2, 2226795507), (3, 2554970601))
}


def changed_keys(before: dict, after: dict) -> set:
    """The top-level keys of the block whose values differ (a key present on one side only counts as changed)."""
    return {k for k in set(before) | set(after) if before.get(k) != after.get(k)}


@pytest.mark.parametrize("s", (1, 2, 3))
def test_the_regenerated_s_view_block_differs_from_its_predecessor_in_seed_base_and_n_trials_only(s):
    tag = f"qwen35_4b_s{s}"
    path = REPO / f"configs/sdfbase_full22_{tag}_sampled_t0.yaml"
    after = yaml.safe_load(path.read_text(encoding="utf-8"))["sampled_decode"]
    assert after == msc.expected_block(PRESETS, tag, 0)                         # the committed bytes are the generator's block
    before = PREDECESSOR[s]
    assert changed_keys(before, after) == ALLOWED_CHANGED, changed_keys(before, after)
    assert (before["seed_base"], after["seed_base"]) == (PREDECESSOR[s]["seed_base"], s) and after["seed_base"] == es.seed_base(s)
    assert (before["n_trials"], after["n_trials"]) == (2, 1)
    # the keys the peer matched byte for byte, named one by one
    for k in ("arm", "trial", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "source", "source_kind",
              "applies_to", "static_cap", "controls_cap", "readouts"):
        assert after[k] == before[k], k
    assert (after["temperature"], after["top_p"], after["top_k"], after["min_p"], after["presence_penalty"], after["repetition_penalty"]) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)
    assert after["static_cap"] == 4096 and after["controls_cap"] == 512 and after["arm"] == "vendor_recommended"
    # the block's key ORDER is unchanged too (the drivers' validators read the keys positionally in the header)
    assert list(after) == list(before)
    # the two rung configs of the same trial carry the same block + the chain fields, and no _t1 file of the TAG exists
    for cap, prev in ((8192, msc.sampled_name(tag, 0, False)), (16384, msc.sampled_name(tag, 0, 8192))):
        rung = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}_sampled_t0_cap{cap}.yaml").read_text(encoding="utf-8"))["sampled_decode"]
        # the rung's block is the predecessor's AT THE RUNG's cap plus the chain field — so the same two keys move and nothing else
        assert changed_keys({**before, "static_cap": cap}, {k: v for k, v in rung.items() if k != "escalated_from"}) == ALLOWED_CHANGED
        assert rung["static_cap"] == cap and rung["escalated_from"]["private_run"] == prev
    assert not list((REPO / "configs").glob(f"sdfbase_full22_{tag}_sampled_t1*.yaml"))


def test_the_diff_names_a_regeneration_that_also_moved_presence_penalty_as_the_known_bad():
    """The KNOWN-BAD: a block that moved presence_penalty beside the two allowed keys must fail the same assertion, naming the key."""
    before = PREDECESSOR[1]
    bad = copy.deepcopy(msc.expected_block(PRESETS, "qwen35_4b_s1", 0))
    bad["presence_penalty"] = 0.0
    got = changed_keys(before, bad)
    assert got == ALLOWED_CHANGED | {"presence_penalty"} and got != ALLOWED_CHANGED
    with pytest.raises(AssertionError):
        assert changed_keys(before, bad) == ALLOWED_CHANGED
    # ... and so must one that moved the static cap, or dropped the readouts sub-block
    capped = copy.deepcopy(msc.expected_block(PRESETS, "qwen35_4b_s1", 0)); capped["static_cap"] = 8192
    assert changed_keys(before, capped) == ALLOWED_CHANGED | {"static_cap"}
    dropped = copy.deepcopy(msc.expected_block(PRESETS, "qwen35_4b_s1", 0)); del dropped["readouts"]
    assert changed_keys(before, dropped) == ALLOWED_CHANGED | {"readouts"}


def test_every_other_view_of_the_scope_still_renders_the_campaigns_two_trials_and_base():
    for tag in msc.sampled_tags():
        if tag in msc.seedrep_tags():
            continue
        block = msc.expected_block(PRESETS, tag, 1)
        assert block["n_trials"] == 2 and block["seed_base"] == 20260909 and block["trial"] == 1
