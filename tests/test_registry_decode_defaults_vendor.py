"""The standing decode ruling, enforced GLOBALLY (2026-09-17; ruling
`decode-vendor-preset-default-2026-09-17`): "always use the vendor presets unless specifically instructed otherwise".

Every `is_default_for_lookup 1` row of `configs/registry/decode_policy_seed.yaml` (live: not superseded) is `mode vendor_preset` UNLESS a
ruling instructs otherwise. The instructed exceptions are ENUMERATED here by the ruling id the row cites (:data:`EXCUSED_BY_RULING`), plus
the one canonical-battery ruling the seed carries as prose rather than as a ruling id (MMLU's greedy letter-logprob read,
:data:`EXCUSED_BY_REF`). A default with no such ruling FAILS BY NAME -- except the defaults the seed carried before the ruling was filed,
which are listed in :data:`EXPECTED_VIOLATIONS` with a TODO: each is a per-row decision (flip to the vendor preset, or cite the
ruling that instructs otherwise), never this test's to flip silently. The list is exact in both directions: a NEW unexcused default fails,
and a row that leaves the list (flipped or excused) fails until the list is updated, so the artifact always sees the current set.

The same check runs over the committed export (`configs/registry/decode_policy.json`), the file every production sampling entry point reads.

The ruling of 2026-09-21 (`decode-vendor-default-all-evals-2026-09-21`, :data:`RULING_ALL_EVALS`): the
agentic runner's T = 1.0 default was a MISINTERPRETATION of the standing rule, not an exemption -- the vendor preset is the default for EVERY
family. So `decode-agentic-runner-T1-2026-09-11` no longer excuses a default (it left :data:`EXCUSED_BY_RULING`; the T1 rows are non-default,
kept and unsuperseded as the recorded policy of every existing agentic run), the `<model>|agentic|<thinking>|vendor` rows are the defaults, and
the Llama em_organism T1 default left :data:`EXPECTED_VIOLATIONS` the same way. Those rows spell the vendor preset `mode vendor` -- the misalign
runner's SAMPLER_VENDOR label, resolved by EXACT ref (`runner.vendor_policy_ref`), so the spelling is identity and is not renamed -- which
:data:`VENDOR_MODES` accepts beside `vendor_preset`.

Llama's `not_available` twins (ruling `decode-llama-thinking-not-available-2026-09-18`, :data:`THINKING_EQUIVALENCE_RULING`): a
`llama8b_it|…|not_available|…` row bound to that ruling is the SAME decode decision as its `off` row re-keyed for the arm Llama's runs file
(Llama has no thinking channel), never a new default -- so it is judged AS its `off` row (the sibling's mode and ruling; a violation is
reported under the sibling's name, which the lists above already carry). A twin whose sibling is missing, or whose mode / source /
default flag differ from the sibling's, fails by name: the equivalence is checked, not assumed.

The worked example's twins (ruling `decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09`, :data:`DEMO_MIRROR_RULING`; the front-door
session's ruling of 2026-10-09): a `qwen35_4b_demo|…` row bound to that ruling is the paper's `qwen35_4b_it|…` row under the worked
example's own key (the same checkpoint, the same pins, the same presets -- tests/test_registry_seeds_demo_instruments.py), never a new
default -- so it is judged AS its 4B row by the same twin rule, and a violation is reported under the 4B row's name (the lists above
already carry `qwen35_4b_it|capability|off|card`).
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Mapping, Optional, Tuple

import pytest

from sisteer.registry import decode_policy as DP
from sisteer.registry import synth
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
EXPORT = os.path.join(ROOT, "configs", "registry", "decode_policy.json")
PROD = synth.producer("tests.registry_decode_defaults_vendor")

RULING = "decode-vendor-preset-default-2026-09-17"
#: the rule of 2026-09-21: the vendor preset is the default for EVERY family; the agentic T = 1.0 default was a misinterpretation
RULING_ALL_EVALS = "decode-vendor-default-all-evals-2026-09-21"
T1_RULING = "decode-agentic-runner-T1-2026-09-11"
VENDOR_MODE = "vendor_preset"
#: the two spellings of the vendor preset a default row may carry: the harness's `vendor_preset` and the misalign runner's `vendor`
#: (SAMPLER_VENDOR; `<model>|agentic|<thinking>|vendor@v1` is resolved by exact ref, so the label is identity)
VENDOR_MODES = (VENDOR_MODE, "vendor")
#: the rule of 2026-09-18: for Llama-8B (no thinking channel) `thinking_mode not_available` == `off`; the seed carries one
#: `not_available` twin per `llama8b_it` `off` row, identical in every parameter, bound to this ruling (tests/test_registry_decode_llama_thinking_na.py)
THINKING_EQUIVALENCE_RULING = "decode-llama-thinking-not-available-2026-09-18"
#: the front-door session's ruling of 2026-10-09: the worked example `qwen35_4b_demo` runs the paper's Qwen3.5-4B checkpoint under the
#: paper's presets, so every explicit `qwen35_4b_it` row has a `qwen35_4b_demo` twin identical in every parameter, bound to this ruling
#: (tests/test_registry_seeds_demo_instruments.py)
DEMO_MIRROR_RULING = "decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09"
DEMO_MODEL, DEMO_TEMPLATE = "qwen35_4b_demo", "qwen35_4b_it"

#: ruling id -> what it instructs INSTEAD of the vendor preset (the seed's `rulings:` block). A default row citing one of these is excused.
EXCUSED_BY_RULING: Dict[str, str] = {
    # GSM8K / MATH-500 / capability thinking-ON: the vendor's THINKING row (shipped generation_config where one exists, Qwen's thinking
    # 'precise' card row where none ships) -- the thinking arm has its own vendor numbers, labelled `thinking_card`
    "decode-thinking-card-rows-2026-09-09": "thinking-ON capability legs decode at the vendor's THINKING row (math500-thinking-on-full22 D1 / D26)",
    # BigCodeBench / BFCL: each benchmark's OWN reference decoding (greedy T 0; T 0.001 under the pinned clamp), model-independent
    "decode-protocol-rows-2026-09-11": "protocol rows: each benchmark's own reference decoding, per task, model-independent (bcb-bfcl-200-widen)",
    # NOT here since 2026-09-21: decode-agentic-runner-T1-2026-09-11 (the agentic runner's explicit T = 1.0). That default was a
    # misinterpretation of the standing rule, not an exemption (RULING_ALL_EVALS); its rows stay, non-default, as the recorded policy of the runs
    # the 4,096-cap multi-turn re-runs send nothing per turn but max_tokens: the model's shipped decode as a MECHANISM (the engine fills
    # from generation_config.json) -- the vendor numbers, reached by omission rather than by sending them
    "decode-multiturn-cap4096-generation-config-2026-09-11": "multi-turn targets send nothing but max_tokens; the engine fills from the shipped generation_config (R2-D253)",
    # Qwen's vLLM SERVER DEFAULT is the vendor instruct row composed into --override-generation-config: a ROUTE row (the values ARE
    # the vendor row minus the presence penalty, which can never be a server default), not a decode choice
    "decode-qwen-server-default-2026-09-11": "the omit_params route: Qwen's vendor instruct row pinned as the vLLM server default (PR #34)",
}
#: policy_ref pattern -> the canonical-battery ruling the seed cites as PROSE (no ruling id on the row). MMLU is a greedy letter-logprob
#: prefill read (top_logprobs 64): nothing is sampled, so no preset applies (CLAUDE.md "The canonical battery", user ruling 2026-09-11;
#: instruments.yaml mmlu_letter_prefill_200).
EXCUSED_BY_REF: Dict[str, str] = {
    "*|mmlu|off|greedy": "MMLU letter-logprob prefill read: greedy, nothing sampled (CLAUDE.md 'The canonical battery', user ruling 2026-09-11; instruments.yaml mmlu_letter_prefill_200)",
}

#: TODO (a per-row decision, NOT flipped here): defaults that are neither `vendor_preset` nor excused by a ruling as of 2026-09-17.
#: Each stays as filed until it is either flipped to the vendor preset (a new row / a re-pointed default) or the ruling that instructs
#: otherwise is named (then it moves to EXCUSED_BY_RULING). Reported in the registry-em-organisms report.
EXPECTED_VIOLATIONS: Dict[str, str] = {
    # the thinking-OFF capability `card` rows (contracts.sampling_for mode `card`; nemotron has none -- it refuses): each states its
    # generation_config or card numbers under the mode label `card` with no ruling id; the two Qwen rows are the hashed legacy contracts row
    # (0.7 / 0.95 / 20, matching NO vendor row -- README contradiction #1)
    "gemma4e4b_it|capability|off|card": "mode card (generation_config numbers), no ruling",
    "qwen35_4b_it|capability|off|card": "mode card = the legacy contracts row 0.7 / 0.95 / 20 (matches no vendor row), no ruling",
    "glm46vflash|capability|off|card": "mode card (generation_config numbers), no ruling (descoped model)",
    "gemma4e2b_it|capability|off|card": "mode card (a 2026-09-04 cluster read, not re-verified), no ruling (reference model)",
    "qwen35_9b_it|capability|off|card": "mode card = the legacy contracts row (matches no vendor row), no ruling (reference model)",
    "llama3b_it|capability|off|card": "mode card inherited from the llama8b card, decode_source unverified, no ruling (reference model)",
    "llama8b_it|capability|not_available|card": "mode card (the shipped 0.6 / 0.9 -- the vendor numbers under the `card` label), no ruling",
    "llama8b_it|gsm8k|not_available|card": "mode card (Llama has no thinking row; the shipped 0.6 / 0.9 under the `card` label), no ruling",
    "llama8b_it|math500|not_available|card": "mode card (as gsm8k), no ruling",
    # Petri targets: nothing sent but max_tokens (the engine fills from generation_config) -- the same mechanism the multi-turn ruling
    # instructs, but NO ruling id on these rows (CLAUDE.md's Petri row is cited as prose)
    "gemma4e4b_it|petri|off|generation_config": "Petri target sends nothing (generation_config mechanism), no ruling id on the row",
    "nemotron3nano4b|petri|off|generation_config": "Petri target sends nothing, no ruling id on the row",
    "llama8b_it|petri|off|generation_config": "Petri target sends nothing, no ruling id on the row",
    "glm46vflash|petri|off|generation_config": "Petri target sends nothing, no ruling id on the row (descoped model)",
    # gone 2026-09-21: llama8b_it|em_organism|off|runner_forced_T1 (the Llama organisms' em8q at the runner's T = 1.0) -- RULING_ALL_EVALS makes
    # llama8b_it|em_organism|off|vendor the default; the T1 row stays non-default as the recorded policy of the Llama organisms panels
}


# --------------------------------------------------------------------------------------------
# the check (pure)
# --------------------------------------------------------------------------------------------


def _live_default(payload: Mapping[str, object], extra: Mapping[str, object]) -> bool:
    return int(payload.get("is_default") or 0) == 1 and not payload.get("superseded_by_policy_id") and not extra.get("superseded_by_unresolved")


def _excused(ref: str, ruling_ref: Optional[str]) -> Optional[str]:
    """The excuse a default row has for not being the vendor preset, or None."""
    if ruling_ref in EXCUSED_BY_RULING:
        return EXCUSED_BY_RULING[str(ruling_ref)]
    for pattern, why in EXCUSED_BY_REF.items():
        if pattern.startswith("*|") and ref.split("|", 1)[1] == pattern.split("|", 1)[1]:
            return why
        if ref == pattern:
            return why
    return None


def twin_sibling_id(pid: str, extra: Mapping[str, object]) -> Optional[str]:
    """For a Llama `not_available` twin (bound to :data:`THINKING_EQUIVALENCE_RULING`), the policy_id of the `off` row it re-keys; for a
    worked-example twin (bound to :data:`DEMO_MIRROR_RULING`), the policy_id of the `qwen35_4b_it` row it re-keys; else None."""
    ruling = extra.get("decode_ruling_ref")
    if ruling == THINKING_EQUIVALENCE_RULING and "|not_available|" in pid:
        return pid.replace("|not_available|", "|off|")
    if ruling == DEMO_MIRROR_RULING and pid.startswith(DEMO_MODEL + "|"):
        return DEMO_TEMPLATE + pid[len(DEMO_MODEL):]
    return None


def twin_faults(rows: Mapping[str, Tuple[Mapping[str, object], Mapping[str, object]]]) -> Dict[str, str]:
    """``{policy_id: why}`` for every twin that cannot stand in for its `off` row: no sibling, or a differing mode / decode_source / default flag /
    supersession state. Empty when every twin IS its sibling's decision re-keyed."""
    out: Dict[str, str] = {}
    for pid, (p, e) in rows.items():
        sib = twin_sibling_id(pid, e)
        if sib is None:
            continue
        if sib not in rows:
            out[pid] = f"twin without its `off` row {sib}"
            continue
        sp, se = rows[sib]
        for what, mine, theirs in (("mode", e.get("mode"), se.get("mode")), ("decode_source", p.get("decode_source"), sp.get("decode_source")),
                                   ("is_default", int(p.get("is_default") or 0), int(sp.get("is_default") or 0)),
                                   ("live", _live_default(p, e), _live_default(sp, se))):
            if mine != theirs:
                out[pid] = f"{what} {mine!r} differs from the `off` row's {theirs!r}"
                break
    return out


def vendor_default_violations(rows: Mapping[str, Tuple[Mapping[str, object], Mapping[str, object]]]) -> Dict[str, str]:
    """``{policy_ref: why}`` for every LIVE default row that is neither `mode vendor_preset` nor excused by a ruling. ``rows`` is the
    loader's ``PolicyIndex.rows`` shape: policy_id -> (payload, extra). A Llama `not_available` twin is judged as its `off` row (the same
    decision re-keyed): its sibling's mode and ruling decide, and a violation is reported under the sibling's ref."""
    out: Dict[str, str] = {}
    for pid, (p, e) in rows.items():
        sib = twin_sibling_id(pid, e)
        if sib is not None and sib in rows:
            p, e = rows[sib]
        if not _live_default(p, e):
            continue
        mode = e.get("mode")
        if mode in VENDOR_MODES:
            continue
        ref = str(p["policy_ref"])
        if _excused(ref, e.get("decode_ruling_ref")) is not None:
            continue
        out[ref] = f"mode {mode!r}, decode_source {p.get('decode_source')!r}, ruling {e.get('decode_ruling_ref')!r}"
    return out


def _unwildcard(ref: str, seeds: D.Seeds) -> str:
    """An expanded wildcard row's ref -> the seed's `*|...` spelling, so the lists above name the seed row, not five expansions."""
    for m in D.served_model_ids(seeds.models):
        if ref.startswith(m + "|"):
            star = "*" + ref[len(m):]
            if any(str(r["policy_ref"]) == star for r in seeds.decode["policies"]):
                return star
    return ref


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def idx(seeds):
    return D.build_records(seeds, PROD, now="2026-09-17T12:00:00.000Z").policies


# --------------------------------------------------------------------------------------------
# the ruling is in the seed and binds at least one default
# --------------------------------------------------------------------------------------------


def test_the_ruling_is_seeded_and_binds_a_vendor_preset_default(seeds, idx):
    assert RULING in seeds.decode["rulings"]
    text = seeds.decode["rulings"][RULING]
    assert "the vendor presets are always used unless specifically instructed otherwise" in text and "not yet confirmed" in text
    bound = [pid for pid, (p, e) in idx.rows.items() if e.get("decode_ruling_ref") == RULING]
    assert bound == ["qwen35_4b_it|em8q|off|vendor_preset@v1"]
    p, e = idx.rows[bound[0]]
    assert p["is_default"] == 1 and e["mode"] == VENDOR_MODE
    # every excuse names a ruling the seed carries (an excuse for a ruling that does not exist excuses nothing)
    assert set(EXCUSED_BY_RULING) <= set(seeds.decode["rulings"]), sorted(set(EXCUSED_BY_RULING) - set(seeds.decode["rulings"]))
    for pattern in EXCUSED_BY_REF:
        assert any(str(r["policy_ref"]) == pattern for r in seeds.decode["policies"]), pattern


# --------------------------------------------------------------------------------------------
# the global rule over the seed
# --------------------------------------------------------------------------------------------


def test_every_live_default_is_the_vendor_preset_unless_a_ruling_instructs_otherwise(seeds, idx):
    """The rule, over the loader's expanded rows: violations == EXPECTED_VIOLATIONS exactly (by name, both directions)."""
    found = {_unwildcard(ref, seeds): why for ref, why in vendor_default_violations(idx.rows).items()}
    new = sorted(set(found) - set(EXPECTED_VIOLATIONS))
    assert not new, ("defaults that are neither `vendor_preset` nor excused by a ruling, and not in EXPECTED_VIOLATIONS -- the user's ruling "
                     "of 2026-09-17 (decode-vendor-preset-default-2026-09-17) makes the vendor preset the default unless a ruling instructs "
                     "otherwise: cite the ruling, flip the row, or (the user's call) add it to EXPECTED_VIOLATIONS with a TODO: "
                     + "; ".join(f"{r}: {found[r]}" for r in new))
    gone = sorted(set(EXPECTED_VIOLATIONS) - set(found))
    assert not gone, f"rows no longer in violation (flipped or excused): remove them from EXPECTED_VIOLATIONS so the artifact sees the current set: {gone}"
    # the excused defaults are exactly the rows citing the enumerated rulings (or the MMLU greedy read); every other live default IS vendor_preset
    for pid, (p, e) in idx.rows.items():
        if twin_sibling_id(pid, e) is not None:
            continue                       # a Llama not_available twin is judged as its `off` row (twin_faults proves the sibling is sound)
        if not _live_default(p, e):
            continue
        ref = _unwildcard(str(p["policy_ref"]), seeds)
        if e.get("mode") in VENDOR_MODES:
            continue
        assert _excused(str(p["policy_ref"]), e.get("decode_ruling_ref")) is not None or ref in EXPECTED_VIOLATIONS, pid


def test_llama_not_available_twins_are_judged_as_their_off_rows_and_add_no_default(seeds, idx):
    """The ruling of 2026-09-18 re-keys Llama's `off` rows for the arm its runs file; it rules no new default. So: every twin has a sound
    sibling (twin_faults empty), the twins are exactly the llama8b_it not_available rows bound to the ruling, and dropping them from the index
    changes the violation set not at all (they add no name, they remove none)."""
    assert THINKING_EQUIVALENCE_RULING in seeds.decode["rulings"]
    twins = {pid for pid, (p, e) in idx.rows.items() if twin_sibling_id(pid, e) is not None and e.get("decode_ruling_ref") == THINKING_EQUIVALENCE_RULING}
    assert twins and all(idx.rows[t][0]["model_id"] == "llama8b_it" and idx.rows[t][0]["thinking_mode"] == "not_available" for t in twins)
    assert twin_faults(idx.rows) == {}
    without = {pid: pe for pid, pe in idx.rows.items() if pid not in twins}
    assert vendor_default_violations(idx.rows) == vendor_default_violations(without)
    # judged by the sibling: a twin of an excused `off` row is excused, a twin of a listed violation reports under the sibling's name
    for t in sorted(twins):
        p, e = idx.rows[t]
        sp, se = idx.rows[twin_sibling_id(t, e)]
        one = vendor_default_violations({t: (p, e), twin_sibling_id(t, e): (sp, se)})
        assert set(one) <= {str(sp["policy_ref"])}, (t, one)
    # a twin whose sibling is gone, or whose mode differs, is a fault by name (never silently excused)
    t = sorted(twins)[0]
    p, e = idx.rows[t]
    assert twin_faults({t: (p, e)}) == {t: f"twin without its `off` row {twin_sibling_id(t, e)}"}
    sib = twin_sibling_id(t, e)
    assert "mode" in twin_faults({t: (p, dict(e, mode="card")), sib: idx.rows[sib]})[t]


def test_the_worked_examples_twins_are_judged_as_their_4b_rows_and_add_no_default(seeds, idx):
    """The front-door session's ruling of 2026-10-09 re-keys the paper's qwen35_4b_it rows for the worked example's key; it rules no new default. So:
    every qwen35_4b_demo row IS a twin with a sound 4B sibling (twin_faults empty), the twins are exactly the demo rows bound to the ruling, and
    dropping them from the index changes the violation set not at all; a violation on a twin reports under the 4B row's name."""
    assert DEMO_MIRROR_RULING in seeds.decode["rulings"]
    twins = {pid for pid, (p, e) in idx.rows.items() if twin_sibling_id(pid, e) is not None and e.get("decode_ruling_ref") == DEMO_MIRROR_RULING}
    demo_rows = {pid for pid, (p, e) in idx.rows.items() if p["model_id"] == DEMO_MODEL}
    assert twins and twins == demo_rows, (sorted(demo_rows - twins), sorted(twins - demo_rows))
    assert twin_faults(idx.rows) == {}
    without = {pid: pe for pid, pe in idx.rows.items() if pid not in twins}
    assert vendor_default_violations(idx.rows) == vendor_default_violations(without)
    for t in sorted(twins):
        p, e = idx.rows[t]
        sib = twin_sibling_id(t, e)
        assert sib in idx.rows and idx.rows[sib][0]["model_id"] == DEMO_TEMPLATE, (t, sib)
        one = vendor_default_violations({t: (p, e), sib: idx.rows[sib]})
        assert set(one) <= {str(idx.rows[sib][0]["policy_ref"])}, (t, one)
    # the one non-vendor, unexcused 4B default (the legacy `card` row) reports ONCE, under the 4B's name, twin or no twin
    card = f"{DEMO_MODEL}|capability|off|card@v1"
    assert card in twins and vendor_default_violations({card: idx.rows[card], twin_sibling_id(card, idx.rows[card][1]): idx.rows[twin_sibling_id(card, idx.rows[card][1])]}) == {
        f"{DEMO_TEMPLATE}|capability|off|card": vendor_default_violations(idx.rows)[f"{DEMO_TEMPLATE}|capability|off|card"]}
    assert not [ref for ref in vendor_default_violations(idx.rows) if ref.startswith(DEMO_MODEL + "|")]


def test_every_excused_ruling_is_actually_cited_by_a_live_default(idx):
    """An excuse nobody uses is dead text; each enumerated ruling excuses at least one live non-vendor default."""
    cited = {e.get("decode_ruling_ref") for p, e in idx.rows.values() if _live_default(p, e) and e.get("mode") not in VENDOR_MODES}
    assert set(EXCUSED_BY_RULING) <= cited, sorted(set(EXCUSED_BY_RULING) - cited)
    # ...and the withdrawn T1 excuse is cited by NO live default: the ruling still names the rows' values, never a default
    assert T1_RULING not in EXCUSED_BY_RULING and T1_RULING not in cited


# --------------------------------------------------------------------------------------------
# the ruling of 2026-09-21: the vendor preset is the default for EVERY family, agentic included
# --------------------------------------------------------------------------------------------

#: every `<model>|agentic|<thinking>|vendor` / `llama8b_it|em_organism|<thinking>|vendor` `off` row bound to RULING_ALL_EVALS (the Llama
#: not_available twins are bound to the thinking-equivalence ruling, as every twin is)
BOUND_ALL_EVALS = (
    "gemma4e4b_it|agentic|off|vendor@v1", "glm46vflash|agentic|off|vendor@v1", "llama8b_it|agentic|off|vendor@v1",
    "llama8b_it|em_organism|off|vendor@v1", "nemotron3nano4b|agentic|off|vendor@v1", "qwen35_4b_it|agentic|off|vendor@v1",
    "qwen35_9b_it|agentic|off|vendor@v1",
)
#: the runner_forced_T1 rows: NON-default since 2026-09-21, kept and unsuperseded (the recorded policy every existing agentic run matched)
T1_ROWS = (
    "gemma4e4b_it|agentic|off|runner_forced_T1@v1", "glm46vflash|agentic|off|runner_forced_T1@v1",
    "llama8b_it|agentic|not_available|runner_forced_T1@v1", "llama8b_it|agentic|off|runner_forced_T1@v1",
    "llama8b_it|em_organism|not_available|runner_forced_T1@v1", "llama8b_it|em_organism|off|runner_forced_T1@v1",
    "nemotron3nano4b|agentic|off|runner_forced_T1@v1", "qwen35_4b_it|agentic|off|runner_forced_T1@v1",
    # the worked example's twin of the 4B row (2026-10-09, DEMO_MIRROR_RULING): the same values under its own key, judged as its sibling below
    "qwen35_4b_demo|agentic|off|runner_forced_T1@v1",
)
#: (model, thinking) -> the served model's static vendor row the agentic default must restate (9B is a reference model with no static row:
#: its numbers are the 4B instruct row's, as the runner's VENDOR_SAMPLING always said)
AGENTIC_KEYS = (("gemma4e4b_it", "off"), ("glm46vflash", "off"), ("llama8b_it", "off"), ("llama8b_it", "not_available"),
                ("nemotron3nano4b", "off"), ("qwen35_4b_it", "off"), ("qwen35_9b_it", "off"))


def test_the_all_evals_ruling_is_seeded_and_binds_exactly_the_vendor_rows(seeds, idx):
    assert RULING_ALL_EVALS in seeds.decode["rulings"]
    text = seeds.decode["rulings"][RULING_ALL_EVALS]
    for needle in ("MISINTERPRETATION", "the vendor preset is the default unless the authors explicitly overrule it", "2026-09-21",
                   "NOT superseded", "misalign_vendor"):
        assert needle in text, needle
    bound = sorted(pid for pid, (p, e) in idx.rows.items() if e.get("decode_ruling_ref") == RULING_ALL_EVALS)
    assert bound == sorted(BOUND_ALL_EVALS)
    for pid in bound:
        p, e = idx.rows[pid]
        assert p["is_default"] == 1 and e["mode"] == "vendor" and p.get("superseded_by_policy_id") is None and p["ruling_id"] is None, pid
        assert "the vendor preset is the default" in str(p["source_citation"]), pid
        assert e.get("decode_ruling_text") == text
    # the T1 rows: non-default, kept, NOT superseded, values untouched (temperature 1.0 alone; nothing else pinned), their own ruling kept
    t1 = sorted(pid for pid, (p, e) in idx.rows.items() if e.get("mode") == "runner_forced_T1")
    assert t1 == sorted(T1_ROWS)
    for pid in t1:
        p, e = idx.rows[pid]
        assert p["is_default"] == 0 and p.get("superseded_by_policy_id") is None and not e.get("superseded_by_unresolved"), pid
        assert p["temperature"] == 1.0 and all(p[k] is None for k in ("top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")), pid
        assert p["seed_rule"] == "none" and e["params_sent"] == ["temperature", "max_tokens"], pid
        if pid.startswith(DEMO_MODEL + "|"):
            # the worked example's twin: bound to the mirror ruling, its notes name the 4B row; the history checks below are its sibling's
            assert e.get("decode_ruling_ref") == DEMO_MIRROR_RULING and twin_sibling_id(pid, e) in t1, pid
            p, e = idx.rows[twin_sibling_id(pid, e)]
        assert RULING_ALL_EVALS in str(p.get("notes")) and "NOT superseded" in str(p.get("notes")), pid
        if "|not_available|" in pid:
            assert e.get("decode_ruling_ref") == THINKING_EQUIVALENCE_RULING, pid
        elif "|em_organism|" in pid:
            assert e.get("decode_ruling_ref") is None, pid                       # inferred by analogy; never ruled
        else:
            assert e.get("decode_ruling_ref") == T1_RULING, pid                  # the values' ruling stands; only the default left
    # every key that has agentic / em_organism rows has exactly ONE live default, and it is the vendor row
    keys = {}
    for pid, (p, e) in idx.rows.items():
        if p.get("eval_category") in ("agentic", "em_organism") and not p.get("superseded_by_policy_id"):
            keys.setdefault((p["model_id"], p["eval_category"], p["thinking_mode"]), []).append((pid, int(p["is_default"])))
    for key, rows in sorted(keys.items()):
        defaults = [pid for pid, d in rows if d == 1]
        assert len(defaults) == 1 and defaults[0].endswith("|vendor@v1"), (key, rows)
    assert idx.no_default_keys == []


def test_the_export_resolves_agentic_to_the_vendor_row_on_every_served_model_and_keeps_the_t1_rows_by_name():
    pf = DP.PolicyFile.load(EXPORT)
    for model, thinking in AGENTIC_KEYS:
        pol = pf.resolve(model, eval_category="agentic", thinking_mode=thinking)
        assert pol.policy_ref == f"{model}|agentic|{thinking}|vendor" and pol.mode == "vendor" and pol.is_default == 1, (model, thinking)
        assert pol.ruling_ref == (THINKING_EQUIVALENCE_RULING if thinking == "not_available" else RULING_ALL_EVALS), (model, thinking)
        if model == "qwen35_9b_it":
            static = pf.resolve("qwen35_4b_it", eval_category="static_attack", thinking_mode="off")
        else:
            static = pf.resolve(model, eval_category="static_attack", thinking_mode=thinking)
        assert pol.numbers() == static.numbers(), (model, thinking, pol.numbers(), static.numbers())   # the model's vendor preset, restated
        t1 = pf.by_ref(f"{model}|agentic|{thinking}|runner_forced_T1") if model != "qwen35_9b_it" else None
        if t1 is not None:
            assert t1.is_default == 0 and t1.superseded_by_policy_id is None and t1.request_params() == {"temperature": 1.0}
    for thinking in ("off", "not_available"):
        em = pf.resolve("llama8b_it", eval_category="em_organism", thinking_mode=thinking)
        assert em.policy_ref == f"llama8b_it|em_organism|{thinking}|vendor" and em.numbers() == pf.resolve("llama8b_it", eval_category="agentic", thinking_mode=thinking).numbers()
    # the eval-scoped em8q default is untouched by the category rows (an eval row wins, C18)
    assert pf.resolve("qwen35_4b_it", eval_id="em8q").policy_id == "qwen35_4b_it|em8q|off|vendor_preset@v1"
    # the runner's T1_TEMPERATURE reads the T1 rows model-independently, default flag or not
    assert pf.model_independent(eval_category="agentic", thinking_mode="off", mode="runner_forced_T1").numbers()["temperature"] == 1.0


def test_the_committed_export_agrees_with_the_seed(seeds):
    """The file every entry point reads: the same live defaults, the same violations, the same excused set."""
    pf = DP.PolicyFile.load(EXPORT)
    rows = {}
    for row in pf.policies:
        payload = {"policy_ref": row["policy_ref"], "is_default": row.get("is_default", 1), "superseded_by_policy_id": row.get("superseded_by_policy_id"),
                   "decode_source": row.get("decode_source")}
        extra = {"mode": row.get("mode"), "decode_ruling_ref": row.get("ruling_ref"), "superseded_by_unresolved": row.get("superseded_by_unresolved")}
        rows[str(row["policy_id"])] = (payload, extra)
    assert twin_faults(rows) == {} and any(twin_sibling_id(pid, e) for pid, (p, e) in rows.items())   # the twins are in the export, each with its sibling
    found = {_unwildcard(ref, seeds): why for ref, why in vendor_default_violations(rows).items()}
    assert set(found) == set(EXPECTED_VIOLATIONS), (sorted(set(found) ^ set(EXPECTED_VIOLATIONS)))
    # and the eval-level em8q default resolves to the vendor preset bound by the ruling
    p = pf.resolve("qwen35_4b_it", eval_id="em8q", thinking_mode="off")
    assert p.mode == VENDOR_MODE and p.ruling_ref == RULING
    # the agentic defaults are the `vendor` rows bound to the 2026-09-21 ruling (mode `vendor`, accepted as the vendor preset)
    bound = sorted(str(r["policy_id"]) for r in pf.policies if r.get("ruling_ref") == RULING_ALL_EVALS)
    assert bound == sorted(BOUND_ALL_EVALS)


# --------------------------------------------------------------------------------------------
# known-bad: the check fires
# --------------------------------------------------------------------------------------------


def test_known_bad_default_is_reported_by_name(seeds):
    """A new default row with a non-vendor mode and no ruling is reported; the same row citing an excusing ruling is not; a superseded
    non-vendor row is never a violation (it is not the default)."""
    s = copy.deepcopy(seeds)
    bad = {"policy_ref": "qwen35_4b_it|xstest|off|card", "policy_version": 1, "model_id": "qwen35_4b_it", "eval_id": "xstest", "thinking_mode": "off",
           "mode": "card", "is_default_for_lookup": 1, "decode_source": "model_card", "temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": None,
           "repetition_penalty": None, "presence_penalty": None, "max_new_tokens_rule": "synthetic", "seed_rule": "none",
           "params_sent": ["temperature", "top_p", "top_k"], "params_not_sent": {}, "source_citation": "synthetic known-bad row (this test)",
           "ruling_id": None, "effective_from_utc": "2026-09-17T00:00:00Z"}
    s.decode["policies"].append(bad)
    idx = D.build_records(s, PROD).policies
    found = vendor_default_violations(idx.rows)
    assert "qwen35_4b_it|xstest|off|card" in found and "mode 'card'" in found["qwen35_4b_it|xstest|off|card"]
    # excused by an enumerated ruling: not a violation
    s2 = copy.deepcopy(seeds)
    s2.decode["policies"].append(dict(bad, ruling_id="decode-protocol-rows-2026-09-11"))
    assert "qwen35_4b_it|xstest|off|card" not in vendor_default_violations(D.build_records(s2, PROD).policies.rows)
    # superseded (non-default) rows are outside the rule
    s3 = copy.deepcopy(seeds)
    s3.decode["policies"].append(dict(bad, is_default_for_lookup=0, superseded_by_policy_id="qwen35_4b_it|static|off|vendor_preset"))
    assert "qwen35_4b_it|xstest|off|card" not in vendor_default_violations(D.build_records(s3, PROD).policies.rows)
    # and the base seed's own violations are unchanged by the additions
    base = {_unwildcard(r, seeds) for r in vendor_default_violations(D.build_records(seeds, PROD).policies.rows)}
    assert base == set(EXPECTED_VIOLATIONS)
