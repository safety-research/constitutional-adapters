#!/usr/bin/env python
"""Generate the full22 wave's transfer configs AND Petri manifests from the roster.

Inputs (all committed):
  configs/sdfbase_full22_roster.yaml   the cell roster: targets, per-config bank VIEWS
                                       (config cell id -> staged id), blocked / pending cells
  configs/sdfbase_full22_pins.json     object_sha256 per (target, staged_id), copied from the
                                       clusters' staging_manifest.json by ``--pins-from``; a FRONT-DOOR
                                       derived object's pin is its REGISTRY object id, read
                                       from --registry-root / SISTEER_REGISTRY_ROOT — and so is a
                                       front-door refusal direction's (the row 04's fit emits;
                                       ``--derived-pins-from-registry``, gap-refusal-pin)
  configs/sdfbase_transfer_<slug>_r3.yaml   the battery body (gemma4e4b, qwen35_4b) — every key
                                       byte-for-byte as data, the mt1 precedent
  configs/two_si_<key>.yaml            the model pins of a target with no sdfbase lineage
                                       (nemotron3nano4b, glm46vflash), substituted into the
                                       qwen35_4b _r3 body

Outputs (regenerated; the test requires byte-identity with the committed files):
  configs/sdfbase_full22_<key>[_<view>].yaml
  configs/capeval/manifests/petri_full22_<key>[_<view>].yaml

SERVING ENGINE (2026-09-13, native-serving-default). The wave is served NATIVELY by default (every LoRA cell a vLLM
native adapter; `mt_serving_up.py --serving-path lora_native`, the stacked lens only behind `--serving-path lens` /
`F22_SERVING_PATH=lens`; CLAUDE.md "Serving default: native LoRA engine"). The MANIFESTS say so (the
`serving_launch_flags` comment, the derived twin `none__lora0` / `none__lens0`). The CONFIG header's serving sentence
("on the vLLM-served lens target under the STACKED reduction mode" and the "Serving:" paragraph) is the LENS-ERA
wording, kept BYTE-IDENTICAL on purpose: tests/test_sdfbase_ifeval_overlays.py pins the llama8b main / it campaign
configs by git blob sha, and every downstream generator (sampled, overlays, misalign, alphasteer, bcbw)
embeds the config bytes. The engine a wave ran on is read from the standup's stamps (`endpoint.json.engine`,
`serving_engine` on every row), never from a config header; re-word the header only together with those pins.

DOSE VIEWS (the strength frontier, 2026-09-10; the roster's DOSE LADDER block). A view that carries a
``dose:`` key is a dose-ladder view: its name must spell that dose (``d<tok>`` / ``it_d<tok>``,
``dose_token``); a dose-named view without the key, or a keyed view whose name spells another dose, is
refused (``view_dose``). Its config is the body's under exactly the same keys as every other view — in
particular ``bon.max_new_tokens`` STAYS the body's 512 (superseding the earlier native-4096 override:
best-of-N follows Scale's 512 -> 4096 climb, whose tooling refuses a store unless the cap is 512).
``_refuse_bon_drift`` re-reads EVERY rendered config and refuses a ``bon`` block, a BoN cap, or an
``attacks.target_max_new_tokens`` that differs from the body's / 512.

PER-VIEW BoN CAP (2026-09-15). The roster grammar admits an
optional ``views.<view>.caps: {bon: N}`` on a target that declares NO caps table: that view's
``bon.max_new_tokens`` is WRITTEN to N (the NATIVE best-of-75 pass at N — no 512 store, no climb; the
sdfbase-llama8b route) through the same ``_sub_line`` the caps table uses, and every other view of the
target keeps the body's 512 byte-for-byte (a target-level cap would move every existing config of the target).
``view_declared_bon_cap`` is the grammar — it REFUSES a view cap on a target that also declares ``caps.bon``
(two authorities for one cap), on a dose view (a dose view moves the object, never a cap), and any key
but ``bon``; ``_view_bon_cap`` = the view's cap, else the target's, else BON_MAX_NEW_TOKENS, and
``_refuse_bon_drift`` holds every rendered config to THAT value. ``bon_served_run.py`` reads the key, so the
view's best-of-N request stream carries ``max_tokens N``. The qwen35_4b ``s1`` / ``s2`` / ``s3`` views (the
batch-64 seed replicates) are the first users.

OBJECT FAMILIES (the ReLU ladder, 2026-09-16). A dose view's name may carry a trailing OBJECT-FAMILY segment
(``d0p25_relu``, ``it_d1p5_relu``; ``_DOSE_VIEW_RE``'s ``objfam``), and for every NEW dose view it MUST: the bare
spelling is CLOSED to the frozen allow-list ``BARE_DOSE_VIEWS_FROZEN`` (the measured LoRA ladders, family ``lora``
implicitly). A named family is declared beside the dose (``object_family: relu``), must agree with the name, must be
one of ``OBJECT_FAMILIES``, and every cell the view serves must satisfy that family's predicate (``lora`` -> a
``lora_r4`` id, ``relu`` -> a ``psr_`` id) — ``view_object_family`` refuses each disagreement, and ``view_dose``
calls it, so no dose consumer reads a view whose family does not hold. Why: a cell of another family added to a
measured dose view would re-serve its rows against a new same-pass anchor and leave two passes per dose in the
record; the family lives in the NAME so that can never happen by roster edit.

CERTIFICATION-ONLY ROSTERS (2026-09-14). ``--roster PATH`` renders another roster's
views through the SAME renderers — the certification roster ``configs/sdfbase_full22_roster_psrcert.yaml``
names llama8b's ``psrcert`` view (the staged all-positions PSR cell beside its LoRA twin; no production view of
llama8b carries a PSR cell) and renders ``configs/sdfbase_full22_llama8b_psrcert.yaml`` +
``configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml`` exactly the way the production pair is
rendered; ``--check --roster PATH`` holds them byte-identical. The rendered header names the roster it came
from (the production spelling is unchanged, so every production file stays byte-identical) and, for a
non-production roster, says so in one paragraph. The production roster is never edited for a certification.

Usage:
  python scripts/sdfbase/make_full22_configs.py                    # write every output
  python scripts/sdfbase/make_full22_configs.py --check            # exit 1 on drift
  python scripts/sdfbase/make_full22_configs.py --pins-from <staging_manifest.json>
                                                                   # refresh the pins, then write
  python scripts/sdfbase/make_full22_configs.py [--check] --roster configs/sdfbase_full22_roster_psrcert.yaml
                                                                   # the certification roster's outputs only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
# the ONE reader of a (target, view)'s served cells (view cells + enabled lanes): every script
# that consumes the roster resolves through it, so a lane flag flip moves them all together
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
from f22_roster import (CAP_LEGS, LEGS, MULTITURN_FAMILIES, is_prompted, lanes_of, load_roster, pending_decision_of,  # noqa: E402
                        pending_fit_of, served_cells, steered_wave_view, STEERED_WAVE_VIEW_RE, target_capability_omitted, target_caps, target_multiturn_families,
                        target_windows, target_xstest_subset, view_cell_universe, view_legs, view_legs_omitted, view_multiturn_windows,
                        view_prompted_only, view_serving, view_windows)
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
PINS = REPO / "configs" / "sdfbase_full22_pins.json"
MANIFEST_DIR = REPO / "configs" / "capeval" / "manifests"

#: TrainTest SI id -> Petri harness SI id (registry_adapter.HARNESS_TO_TRAINTEST_SI inverted).
#: Restated here so the generator stays import-light; tests/test_sdfbase_full22_wave.py asserts
#: this table against registry_adapter.harness_cell_id for every cell the roster serves.
TT_TO_HARNESS_SI = {
    "constv2-mix-full": "constv2_mix_full",
    "constv2ctl2-full": "constv2ctl2_full",
    "safety_refusal": "safety",
    "si_constitution": "constitution",
    "si_irrelevant": "irrelevant",
    # the FULL constitution (sdfbase-llama8b, 2026-09-10): the whole document as a prompted
    # baseline (prompted__si_constitution_full <-> prompted__constitution_full); `si_constitution`
    # is the compressed EXCERPT and is never described as the constitution
    "si_constitution_full": "constitution_full",
    # the candidate-SI triplet (lane `candsi`, behind lane_flags.candsi): the candidate-si-refit
    # lineage's harness ids (registry_adapter.HARNESS_TO_TRAINTEST_SI there). Reachable only while
    # that flag is on; the wave test refuses the flag until THIS tree's registry_adapter agrees.
    "A10": "safety_a10",
    "B3": "constitution_b3",
    "C16": "irrelevant_c16",
}

#: BARE Class B cell ids — an arm with no SI, spelled the SAME in both vocabularies
#: (``registry_adapter._harness_base_cell_id`` returns such an arm unchanged). Restated here like
#: TT_TO_HARNESS_SI above to keep the generator import-light; tests/test_sdfbase_full22_wave.py
#: asserts this tuple equals ``defense_cells.CLASS_B_IDS + SERVED_ONLY_CLASS_B_IDS``. ``arditi``
#: and ``arditi_allpos`` are the SERVED-ONLY ones (no trainer on this tree; their objects are the
#: arditi lane's, named through the roster's cross-bank pin form). The two
#: are the SAME fitted direction served in DIFFERENT configurations — ``arditi`` prefill-only at a
#: fraction of the residual norm, ``arditi_allpos`` the paper's own raw coefficient at every
#: position of both phases — and never aliases of each other (arditi-qwen-all-positions, 2026-09-16).
BARE_CLASS_B_IDS = ("antidote", "alphasteer", "caa_static_r", "arditi", "arditi_allpos")


def _yaml_flag(tok: str) -> str:
    """A launch-flag token as a YAML scalar: a JSON value (`{"image":0,...}`) would otherwise parse
    as a flow mapping, so anything that is not a plain word is single-quoted (YAML single quotes
    need no escaping for double quotes, braces or colons)."""
    if tok.startswith("--") or all(c.isalnum() or c in "_-." for c in tok):
        return tok
    assert "'" not in tok, tok
    return f"'{tok}'"


#: Per-model serving pins for the Petri manifests. chat_template_hash = sha256_hex16 over the
#: pinned tokenizer's chat_template bytes (the standup re-hashes the served template and refuses
#: a mismatch, so a stale value here fails loudly before a GPU minute is spent); tool_parser = the
#: pinned vLLM 0.25.1's parser for the model's tool-call format. Values from the completed audits'
#: manifests on this tree (petri_gemma4e4b.yaml) and the candidate-si-refit lineage's per-model
#: manifests (qwen / glm / nemotron), which served these exact (model, revision) pins.
MODEL_SERVING = {
    "gemma4e4b": {
        "chat_template_hash": "0a2c8073c878ab1d", "tool_parser": "gemma4",
        "flags": ["--enforce-eager", "--chat-template-content-format", "string"],
        "note": ("gemma-4 is multimodal (Gemma4ForConditionalGeneration): content-format=string "
                 "keeps the system block on the template's string branch (petri_gemma4e4b.yaml, "
                 "2026-08-30); tool parser `gemma4` (llama3_json 500s on this tokenizer)")},
    "qwen35_4b": {
        "chat_template_hash": "a4aee8afcf2e0711", "tool_parser": "qwen3_xml",
        "flags": ["--enforce-eager", "--chat-template-content-format", "string",
                  "--limit-mm-per-prompt", '{"image":0,"video":0}'],
        "note": ("Qwen3.5 is multimodal too (Qwen3_5ForConditionalGeneration): the same "
                 "content-format pin; qwen3_xml parses its <tool_call><function=..> format; "
                 "prefix caching + thinking-off template kwargs are wrapper-owned. TEXT-ONLY "
                 "(--limit-mm-per-prompt image=0,video=0): vLLM 0.25.1 then skips the multimodal "
                 "encoder profiling (multimodal/registry.py: all limits 0 -> no MM infrastructure), "
                 "whose dummy max-size images sized off the encoder-cache budget (= S x L; 43 items "
                 "at 148 x 4864 / 82 x 8704, 32 at 256 x 2048) were OOM-killed by the 200G host-RAM "
                 "cgroup at the full22 launch smoke (Slurm OUT_OF_MEMORY, engine core dead by signal "
                 "within 1 s of the profiling line; smokes 421951 / 422285 on cluster A; the mt1 "
                 "precedent survived 43 items under --mem=256G). This wave sends text only.")},
    "nemotron3nano4b": {
        "chat_template_hash": "ab7813c3abdd9cb6", "tool_parser": "qwen3_xml",
        "flags": ["--enforce-eager"],
        "note": ("NemotronH Mamba2 hybrid: HYBRID_MODEL_KEYS (geometry.py) caps max_num_seqs at "
                 "256 per slot and the composer owns --mamba-ssm-cache-dtype; transformers "
                 "5.15.0 (pyproject exclude-newer-package) for the cache-layer mapping")},
    "llama8b": {
        # sha256_hex16 of the pinned NousResearch @ d10aef79 tokenizer's chat_template (348 chars):
        # RECOMPUTED locally from the cached snapshot 2026-09-10 = petri_twosi.yaml's frozen value.
        # The Nous mirror's template renders EXACTLY the messages given (meta-llama's injects a
        # 25-token default system block), so the `none` anchor is unprompted — configs/stage2_jailbreak.yaml
        "chat_template_hash": "b48c47f644389271", "tool_parser": "llama3_json",
        "flags": ["--enforce-eager"],
        "note": ("Llama-3.1-8B-Instruct (dense, text-only; no thinking mode — the 4-id assistant header "
                 "<|start_header_id|>assistant<|end_header_id|>\\n\\n and no switch, so there is nothing to "
                 "declare for a thinking regime): the sdfbase-llama8b lane's served target. Its budget row "
                 "is the MEASURED 475,136 = 29 x 16384 (the 2026-09-10 serving gate: the production stacked-lens "
                 "standup + a full-window load probe, bisected to the KV-coverage wall at 31 x 16384; one card-size "
                 "of KV margin below the 30 that also passed), so every window's S is budget // L — 54 multi-turn, "
                 "97 static, 64 best-of-N, 232 controls, 29 Petri, 10 at the full-constitution view's 45,056. Tool "
                 "parser llama3_json (vLLM's Llama-3.1 parser; petri_twosi.yaml)")},
    "glm46vflash": {
        "chat_template_hash": "bf6cfe131f9af1cd", "tool_parser": "glm45",
        "flags": ["--enforce-eager", "--chat-template-content-format", "string",
                  "--limit-mm-per-prompt", '{"image":0,"video":0}'],
        "note": ("GLM-4.6V-Flash (Glm4vForConditionalGeneration, image + video encoders): served "
                 "TEXT-ONLY (--limit-mm-per-prompt image=0,video=0), the same pin qwen35_4b needed — "
                 "its multimodal profiling (dummy max-size images sized off the S x L encoder-cache "
                 "budget) OOM-killed the 200G host-RAM cgroup at the full22 launch smoke on qwen "
                 "(Slurm OUT_OF_MEMORY, engine core dead by signal within 1 s of the profiling line), "
                 "and GLM's encoder is the larger one; with all limits 0, vLLM 0.25.1 builds no "
                 "multimodal infrastructure and the identity geometry stands. M-RoPE positions are "
                 "served through capeval/serving/vllm_patches.py (lens_worker applies the "
                 "interleaved-layout patch at worker import; the neox-rotation kernel bug, main "
                 "bd89108). ATTESTED at every standup since 2026-09-08: mt_serving_up.mrope_attest asks "
                 "every worker sisteer_mrope_report over the authenticated collective_rpc route, records "
                 "it as endpoint.json `mrope_attestation`, and REFUSES to serve unless the report shows "
                 "MRotaryEmbedding / is_neox_style=False / dispatch forward_cuda / patch_applied; "
                 "common.sh re-reads the stamp and refuses traffic without it; mrope_attestation.json "
                 "records routed_native_calls again at STOP, after traffic")},
}

#: THE FRONT DOOR: a target of the models table
#: ``reproduce/models.yaml`` with PHASE views (``u0``, ``s<N>``, ``d<tok>[_relu]``; PHASE_VIEW_RE, the grammar every other generator
#: admits) is rendered from the TABLE, never from a hand-listed entry here: ``render_config`` substitutes the body's ``model`` /
#: ``model_revision`` with the roster block's (held equal to the table's ``serve.checkpoint``), and ``render_manifest`` takes the
#: Petri serving pins from ``front_door_serving`` — ``chat_template_hash`` = the table's ``serve.chat_template_hash`` (sha256_hex16 of
#: the pinned tokenizer's chat_template bytes, what the standup re-hashes and refuses a mismatch on; a row without it refuses by
#: name, never a placeholder), ``tool_parser`` + launch ``flags`` by the FAMILY rule (FAMILY_SERVING, the paper's row for the
#: architecture family). MODEL_SERVING stays the paper's five, byte-identical; a key in BOTH refuses at import (one authority).
#: `s<N>r` = the ReLU-parent subset of phase N (`s2r` = control_relu + constitution_relu at alpha 1; a view outside this grammar
#: renders nothing here while the planner still plans it) — the same grammar in every copy, pinned equal by tests
#: `s<N>rl` = the LeCun-init (random_lecun_1e-3) REFIT of phase N's ReLU line, served from the sibling bank full22_lecun
#: (`s2rl` = control_relu + constitution_relu + difference_relu refits at alpha 1, steered-only, cells bound by object id)
PHASE_VIEW_RE = re.compile(r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
MODELS_TABLE = Path(os.environ.get("SISTEER_MODELS_TABLE") or (REPO / "reproduce" / "models.yaml"))
MODELS_TABLE_READER = REPO / "reproduce" / "lib" / "models.py"     # the ONE reader of the table (duplicate keys refused; review A9)
_QWEN_TEXT_ONLY_FLAGS = ["--enforce-eager", "--chat-template-content-format", "string", "--limit-mm-per-prompt", '{"image":0,"video":0}']
#: models.yaml ``architecture.family`` -> (tool_parser, serving_launch_flags): the paper's MODEL_SERVING rows restated PER FAMILY
#: (tests/test_full22_front_door_targets.py holds each paper model's family row equal to its literal).
FAMILY_SERVING = {
    "qwen3_5_hybrid": ("qwen3_xml", _QWEN_TEXT_ONLY_FLAGS),
    "gemma4_dense": ("gemma4", ["--enforce-eager", "--chat-template-content-format", "string"]),
    "nemotron_h_hybrid": ("qwen3_xml", ["--enforce-eager"]),
    "llama_dense": ("llama3_json", ["--enforce-eager"]),
}


def _models_reader():
    """``reproduce/lib/models.py`` loaded by path (the ONE reader; duplicate keys refused); a table without its reader beside this tree is refused."""
    if not MODELS_TABLE_READER.is_file():
        raise SystemExit(f"models table reader {MODELS_TABLE_READER} is not on this tree — refusing to read the table another way")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_full22_reproduce_models_table", MODELS_TABLE_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_models_table(path: Path) -> dict:
    """``reproduce/models.yaml`` through its ONE reader; a table without its reader beside this tree is refused, never read another way."""
    return _models_reader().load_models(str(path))


def front_door_models(table_path: "Path | str | None" = None) -> dict:
    """``{key: the table's model row}`` for every model of the models table that names PHASE views (the front door's targets); an
    ABSENT table contributes nothing (the paper's tree before the front door); the paper's models (``main`` / ``it`` only) contribute
    nothing; a key MODEL_SERVING already names refuses (one authority for a target's serving pins)."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return {}
    out = {}
    reader = _models_reader()
    for key, m in reader.load_models(str(p)).items():
        # the reader's ONE classification (models.phase_views_of: the grammar's views plus the row's declared `wave` — demo-import M3)
        if not isinstance(m, dict) or not reader.phase_views_of(m):
            continue
        if key in MODEL_SERVING:
            raise SystemExit(f"{p.name}: model {key!r} names phase views but is a MODEL_SERVING target — one authority for a target's "
                             "serving pins; refusing")
        out[str(key)] = m
    return out


FRONT_DOOR_MODELS = front_door_models()


def front_door_model(target: str) -> "dict | None":
    """The models-table row of a FRONT-DOOR target, None for every other target (the paper's, a certification roster's)."""
    return FRONT_DOOR_MODELS.get(target)


def front_door_serving(target: str) -> dict:
    """MODEL_SERVING's shape for a front-door target, DERIVED from its table row: chat_template_hash (serve.chat_template_hash, 16 hex,
    refused when unrecorded), tool_parser + flags (FAMILY_SERVING by architecture.family, refused for a family this generator does not
    know), and a note naming the sources."""
    m = front_door_model(target)
    if m is None:
        raise SystemExit(f"{target!r}: not a MODEL_SERVING target and not a front-door model of {MODELS_TABLE.name} (phase views) — "
                         f"the manifest's serving pins have no source (MODEL_SERVING: {sorted(MODEL_SERVING)}; front door: {sorted(FRONT_DOOR_MODELS)})")
    serve = m.get("serve") or {}
    ck = serve.get("checkpoint") or {}
    fam = (m.get("architecture") or {}).get("family")
    if fam not in FAMILY_SERVING:
        raise SystemExit(f"{target}: architecture.family {fam!r} has no serving row (FAMILY_SERVING: {sorted(FAMILY_SERVING)}) — add the "
                         "family's tool parser and launch flags there, deliberately, before a manifest names it")
    h = serve.get("chat_template_hash")
    if not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{16}", h):
        raise SystemExit(f"{target}: {MODELS_TABLE.name} serve.chat_template_hash is not recorded ({h!r}) — the Petri manifest pins "
                         "sha256_hex16 of the pinned tokenizer's chat_template bytes (chat_template.jinja at serve.checkpoint.revision) "
                         "and the standup refuses a mismatch, so no placeholder may ride here; record it from the pinned snapshot")
    tool_parser, flags = FAMILY_SERVING[fam]
    ref = next((k for k, mm in _load_models_table(MODELS_TABLE).items() if k in MODEL_SERVING and (mm.get("architecture") or {}).get("family") == fam), None)
    return {"chat_template_hash": h, "tool_parser": tool_parser, "flags": list(flags),
            "note": (f"{m.get('display', target)} ({fam}) — the FRONT DOOR's served target ({MODELS_TABLE.name} `{target}` serve.checkpoint "
                     f"{ck.get('hf_repo')} @ {str(ck.get('revision', ''))[:8]}): launch flags and tool parser by the architecture-family rule "
                     f"(make_full22_configs.FAMILY_SERVING" + (f", the paper's {ref} row for this family" if ref else "") + "); chat_template_hash = "
                     "sha256_hex16 of the pinned snapshot's chat_template.jinja bytes (models.yaml serve.chat_template_hash; the standup "
                     "re-hashes the served template and refuses a mismatch)")}


BEHAVIORS = ["cyber_bio_chem_capabilities", "self_preservation", "delusion_sycophancy",
             "defer_to_users", "reasoning_unfaithfulness", "instruction_evasion"]
BUNDLE = "8812ba083bb5473c"
#: sha256_hex16 over the EXACT SI text bytes (model-independent; petri_gemma4e4b.yaml, verified
#: on this tree 2026-08-30; `safety` re-verified 2026-09-08 against resolve_instruction's bytes).
#: The values ride along in every manifest so one that gains a prompted cell verifies per request
#: (H8) without a re-freeze — nemotron3nano4b/main serves `prompted__safety` since 2026-09-08.
PREFIX_HASHES = {"safety": "ac3a58f61d53cf24", "constitution": "21e06f482a6559b3",
                 "irrelevant": "71ff67aa2bc000f1",
                 # sha256_hex16 of data/instructions.CONSTITUTION_FULL_PROMPT (the vendored
                 # constitution_full.md, 182,219 chars; = the file's sha256 prefix), 2026-09-10
                 "constitution_full": "82f88c5532e4939e"}
#: The hashes EVERY manifest carries (the three that were in every full22 manifest when its
#: Petri legs ran). petri_harness_config_inputs hashes the WHOLE prompted_prefix_hashes
#: mapping, so a key added to a manifest re-keys every completed leg's harness_config_hash
#: (H10) — the `irrelevant` widening (2026-08-28) predated every full22 audit; the
#: `constitution_full` one (2026-09-10) does not. A hash outside this set therefore rides
#: ONLY the manifests that serve a prompted cell over its SI (render_manifest), and every
#: manifest that does not is byte-identical to the one its legs ran under.
PREFIX_HASHES_ALWAYS = ("safety", "constitution", "irrelevant")

#: the campaign's BoN completion cap, which EVERY view keeps — dose views included: Scale's
#: 512 -> 4096 climb tooling refuses a store unless bon.max_new_tokens == 512, so the
#: generator refuses a rendered config whose cap is anything else
BON_MAX_NEW_TOKENS = 512
#: `d<tok>` / `it_d<tok>`, optionally `_<object family>` (the ReLU ladder, 2026-09-16; OBJECT_FAMILIES below). The
#: dose token is group 2 and must still spell the view's `dose:` exactly (`view_dose`); the family segment is `objfam`.
_DOSE_VIEW_RE = re.compile(r"^(it_)?d(m?\d+(?:p\d+)?)(?:_(?P<objfam>[a-z][a-z0-9]*))?$")
#: D0P75RELUL: a dose view may serve the copy of a SIBLING LANE's parent (the LeCun-init refit `difference_relu_lecun`,
#: lane lecun_derived, bank full22_lecun — the same cell id as the zero-init composition's). The lane rides the view name as a marker after
#: the family segment (`d0p75_relul`: family relu, lane lecun — as `s2rl` is `s2r`'s LeCun lane) and the staged id as the same marker after
#: the dose token (`d0p75l__<cell>`, the `d<dose>it__` qualifier shape); the roster declares it (`object_lane: lecun`). lane -> marker
DOSE_LANES = {"lecun": "l"}
#: the models-table lane a dose copy's PARENT lives in, per dose lane (None = the zero-init / default lane)
DOSE_PARENT_LANE = {None: "derived", "lecun": "lecun_derived"}
#: a DOSE COPY's staged id (stage_strength.py / scale_psr_dose.py): `d<tok>[<lane marker>]__<cell id>` — the dose token (+ the lane marker)
#: in front of the parent's cell id
_DOSE_SID_RE = re.compile(r"^d(?P<tok>m?\d+(?:p\d+)?)(?P<lane>l)?__(?P<cell>.+)$")
#: OBJECT FAMILIES (the ReLU ladder, 2026-09-16): the family segment a dose view's name may carry (`d<tok>_<family>`,
#: `it_d<tok>_<family>`) -> (the predicate a cell id must satisfy to be served by a view of that family, the words the
#: refusal uses). A dose view serves ONE object family against its own same-pass `none` anchor: the roster's bare `d<tok>`
#: views are the LoRA ladders, already measured, and a cell of another family added to one of them would re-serve the
#: measured rows against a new anchor and leave two passes per dose in the record. So the family is spelled in the NAME
#: (a view of another family is a new view, with a new anchor), declared beside `dose:` (`object_family:`), and every cell
#: the view serves is held to it (`view_object_family`). A family not in this table is refused by name; add one here,
#: with its predicate, before naming it in a view.
OBJECT_FAMILIES = {
    "lora": (lambda cid: "lora_r4" in cid, "a `lora_r4` cell id"),
    "relu": (lambda cid: "psr_" in cid, "a `psr_` cell id"),
    # THE BATCH-64 SEED-REPLICATE CONTROL PARENTS AT STRENGTH -1 (ctl-neg1, 2026-09-25): the `constv2ctl2-full` control
    # objects of the s1 / s2 / s3 views, write tensor x -1, one view per (fit seed, class) -- `dm1_ctllora64s<k>` (the control
    # LoRA, unit the write basis) and `dm1_ctlrelu64s<k>` (the control PSR, unit the direction norm). The predicates are
    # narrowed to the CONTROL cell ids on purpose: a constitution or a difference cell can never ride a control-family view.
    # SEEDLESS families: a fit seed is not a different object family (same predicate, same unit, same axis), so the seed rides
    # the view name as a trailing `s<N>` and `family_of_segment` strips it -- `ctllora64s1` / `s2` / `s3` are ONE family at
    # three seeds, never three families that can drift apart (the strength-dose-s1 precedent for `loradiff64`).
    "ctllora64": (lambda cid: "constv2ctl2" in cid and "lora_r4" in cid, "a `constv2ctl2` control `lora_r4` cell id"),
    "ctlrelu64": (lambda cid: "constv2ctl2" in cid and "psr_" in cid, "a `constv2ctl2` control `psr_` cell id"),
}


def family_of_segment(segment: str) -> str:
    """The family a view-name segment names, with any trailing fit-seed suffix removed.

    `ctllora64s2` -> `ctllora64`. Only strips when the seedless stem is a known family, so a family whose
    real name ends in s<digits> is never truncated into something else; a segment that is itself a family
    (`lora`, `relu`) is returned unchanged."""
    m = re.match(r"^(?P<stem>[a-z][a-z0-9]*?)s(?P<seed>[1-9][0-9]*)$", segment or "")
    if m and m.group("stem") in OBJECT_FAMILIES:
        return m.group("stem")
    return segment


def split_family_lane(segment: str) -> tuple:
    """D0P75RELUL: `(family, lane)` of a dose view's family segment — `relul` -> (`relu`, `lecun`); a known family (seed-suffixed or not)
    -> (itself, None); anything else unchanged with no lane (view_object_family refuses it by name as before)."""
    if segment in OBJECT_FAMILIES or family_of_segment(segment) in OBJECT_FAMILIES:
        return segment, None
    for lane, mark in DOSE_LANES.items():
        stem = segment[:-len(mark)] if segment.endswith(mark) else None
        if stem and (stem in OBJECT_FAMILIES or family_of_segment(stem) in OBJECT_FAMILIES):
            return stem, lane
    return segment, None
#: FROZEN AND CLOSED (2026-09-16). The bare dose views — `d<tok>` / `it_d<tok>`, NO object-family segment — that the
#: production roster carried, per target, when the family segment became REQUIRED: every one is a LoRA-only view (family
#: `lora`, implicitly, by this list) and is already measured against its own same-pass anchor. This literal is the whole
#: allow-list and NOTHING is ever added to it: a new dose view names its object family in its name (`d0p25_relu`), a new
#: LoRA rung included (`d2_lora`), so the record can never carry two passes of a measured dose under one view. A bare
#: dose view not listed here refuses (`view_object_family`); the wave test holds this list equal to the roster's bare
#: dose views, so neither can grow without the other.
BARE_DOSE_VIEWS_FROZEN = {
    "gemma4e4b": ("it_dm1", "it_d0p1", "it_d0p25", "it_d0p5", "it_d1", "it_d1p5",
                  "dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5", "d2", "d4"),
    "qwen35_4b": ("it_dm1", "it_d0p1", "it_d0p25", "it_d0p5", "it_d1", "it_d1p5",
                  "dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5"),
    "nemotron3nano4b": ("dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5"),
    "llama8b": ("it_dm1", "it_d0p1", "it_d0p25", "it_d0p5", "it_d1", "it_d1p5",
                "dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5"),
    "glm46vflash": ("dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5"),
    # THE FRONT DOOR: a
    # front-door target of reproduce/models.yaml declares its LoRA dose view BARE (`d0p75`: cells none + difference_lora, dose_unit
    # lora_write_basis_multiplier) and its ReLU twin NAMED (`d0p75_relu`) — the paper's own grammar for the two families — so the
    # bare view joins here, cited, and NOTHING of the paper's five lists above moves (the wave test holds each front-door entry to
    # the table's bare LoRA dose views, and the paper's five to the production roster's, as before).
    "qwen35_4b_demo": ("d0p75",),   # the worked example: the one bare LoRA dose view its models.yaml row declares
}
#: the ONE unit a LoRA-ladder dose is measured in: the multiplier on the
#: served write basis B (B_pos / B_neg together for the subtraction), which is exactly the multiplier on the served
#: o_proj edit. Every dose view DECLARES it (`dose_unit:`) beside its `dose:` so the dose axis a reader plots is
#: named by the roster and not inferred from the producer; the generator refuses a dose view without it, with
#: another unit, or a unit on a non-dose view. (The AlphaSteer ladder's unit, `alphasteer_base_scale`, lives in its
#: own roster and never appears here.)
#: THE UNIT IS PER OBJECT FAMILY, because the families' dose axes are different quantities and must never be read as
#: one. A LoRA dose multiplies the WRITE BASIS `B` of the o_proj edit. A ReLU (PSR) dose multiplies the DIRECTION `d`
#: and leaves the gate byte-identical: a gated PSR has the gauge freedom `(w, b, d) -> (c w, c b, d / c)` under the
#: ReLU's positive homogeneity, so `||d||` is the dose and scaling the gate would move its FIRING RATE instead
#: (scripts/sdfbase/scale_psr_dose.py). A view declares the unit its family names, and no other.
DOSE_UNITS = {"lora": "lora_write_basis_multiplier", "relu": "psr_direction_norm_multiplier",
              # the seed-replicate control parents (ctl-neg1): the same two axes, keyed by the SEEDLESS family
              # (`family_of_segment`) -- a fit seed does not change the axis
              "ctllora64": "lora_write_basis_multiplier", "ctlrelu64": "psr_direction_norm_multiplier"}
#: the frozen bare dose views are family `lora` implicitly (BARE_DOSE_VIEWS_FROZEN), so this stays their unit
DOSE_UNIT = DOSE_UNITS["lora"]


def dose_token(dose: float) -> str:
    """The view-name spelling of a dose: `p` for the point, `m` for the minus sign, no trailing
    `.0` (0.1 -> 0p1, 0.25 -> 0p25, 0.5 -> 0p5, 1 -> 1, 1.5 -> 1p5, -1 -> m1)."""
    d = float(dose)
    if d == 0:
        raise SystemExit("a dose view's dose must be nonzero (dose 0 is the `none` anchor, never a view)")
    s = str(int(d)) if d == int(d) else repr(d)
    return s.replace("-", "m").replace(".", "p")


DOSE_FAMILIES = ("it", "main")


def dose_families(roster: dict, target: str) -> tuple:
    """The ladder FAMILIES a target carries (`dose_families:`): `it` = instruct-fit objects under `it_d<tok>`,
    `main` = base-fit objects under `d<tok>`. Default: `it` iff the target has an instruct-direct `it` view, else
    `main`. A declared family must be one of DOSE_FAMILIES and the target must have the view it anchors to."""
    t = roster["targets"][target]
    fams = t.get("dose_families")
    if fams is None:
        return ("it",) if "it" in t["views"] else ("main",)
    if not isinstance(fams, list) or not fams or any(f not in DOSE_FAMILIES for f in fams) or len(set(fams)) != len(fams):
        raise SystemExit(f"{target}: dose_families must be a non-empty list drawn from {DOSE_FAMILIES} without repeats, got {fams!r}")
    for f in fams:
        if f not in t["views"]:
            raise SystemExit(f"{target}: dose_families names {f!r} but the target has no `{f}` view to anchor that family")
    return tuple(fams)


def view_object_family(roster: dict, target: str, view: str) -> str:
    """The ONE object family a dose view serves — `lora` for a view on the frozen bare allow-list, else the family its
    name spells — REFUSING every way the name, the `object_family:` key and the cells can disagree: a bare dose view
    that is not on BARE_DOSE_VIEWS_FROZEN (name the family in the view name), a key on a bare view, a family segment
    with no key or with a key that names another family, a family this generator does not know, and a SERVED cell
    (lane cells included) that is not of the view's family — the contamination the grammar exists to prevent. Called
    by `view_dose` once the dose spelling holds, so a dose view is never rendered, or read by a dose consumer, without
    its family holding."""
    t = roster["targets"][target]
    v = t["views"][view]
    m = _DOSE_VIEW_RE.match(view)
    segment, lane = split_family_lane(m.group("objfam")) if (m and m.group("objfam")) else (None, None)
    declared = v.get("object_family")
    declared_lane = v.get("object_lane")
    if (lane is not None or declared_lane is not None) and lane != declared_lane:     # D0P75RELUL: the name's lane marker and the key agree
        raise SystemExit(f"{target}/{view}: the view name spells dose lane {lane!r} but the view declares `object_lane: {declared_lane!r}` — "
                         f"the two must agree (a LeCun-lane dose view is `d<tok>_<family>{DOSE_LANES['lecun']}` with `object_lane: lecun`)")
    if segment is None:
        if view not in BARE_DOSE_VIEWS_FROZEN.get(target, ()):
            raise SystemExit(f"{target}/{view}: a NEW bare dose view — the bare `d<tok>` / `it_d<tok>` spelling is closed to "
                             "every view that is not on the frozen allow-list (BARE_DOSE_VIEWS_FROZEN: the measured LoRA "
                             f"ladders): name the object family in the view name (`{view}_lora`, `{view}_relu`) and declare "
                             "`object_family:` to match, so a measured dose is never re-served beside another family's cells")
        if declared is not None:
            raise SystemExit(f"{target}/{view}: declares `object_family: {declared}` but its name carries no object-family "
                             "segment — a frozen bare dose view is the LoRA ladder (family `lora`) by the allow-list and "
                             "declares nothing; a view of another family is `d<tok>_<family>`")
        fam = "lora"
    else:
        if segment not in OBJECT_FAMILIES and family_of_segment(segment) not in OBJECT_FAMILIES:
            raise SystemExit(f"{target}/{view}: object family `{segment}` is not one this generator knows — the families are "
                             f"{tuple(OBJECT_FAMILIES)} (OBJECT_FAMILIES); add a family there, with its cell-id predicate, "
                             "before naming it in a view")
        if declared is None:
            raise SystemExit(f"{target}/{view}: the view name names object family `{segment}` but the view declares no "
                             f"`object_family:` — declare `object_family: {segment}` beside `dose:` (the roster states what "
                             "the name spells; neither is inferred from the other)")
        if declared != segment:
            raise SystemExit(f"{target}/{view}: the view name names object family `{segment}` but the view declares "
                             f"`object_family: {declared}` — the two must agree; rename the view or fix the key")
        fam = segment
    # the seed suffix is not part of the family identity: look the predicate and the unit up SEEDLESS
    fam = family_of_segment(fam)
    is_member, words = OBJECT_FAMILIES[fam]
    for cid in served_cells(roster, target, view):
        if not is_member(cid):
            raise SystemExit(f"{target}/{view}: object family `{fam}` but serves `{cid}`, which is not {words} — a dose view "
                             "serves ONE object family against its own same-pass anchor; give the cell a "
                             f"`{'it_' if view.startswith('it_') else ''}d<tok>_<its family>` view of its own")
    return fam


def view_object_lane(roster: dict, target: str, view: str):
    """D0P75RELUL: the dose LANE a dose view serves (`lecun` for `d0p75_relul`, None for every other dose view and every non-dose view),
    read from the name's marker and held equal to the roster's `object_lane:` by view_object_family."""
    m = _DOSE_VIEW_RE.match(view)
    if not m or not m.group("objfam"):
        return None
    view_object_family(roster, target, view)
    return split_family_lane(m.group("objfam"))[1]


_PAIRED_VIEW_RE = re.compile(r"^(?P<seed>s[1-9][0-9]*)p$")


def view_paired_of(roster: dict, target: str, view: str):
    """The seed view a PAIRED view is an alias of — the roster's `paired_of:` key, the twin marker — or None for every other view,
    REFUSING the ways the name and the key can disagree: a paired-named view (`s<s>p`) with no key; a key on a view not named
    `<seed>p`; a key naming a view the target does not carry; a paired view that re-serves nothing from its seed; and a re-served
    cell whose staged id is not the seed view's — the re-serve is a STANDUP replicate of the seed view's objects, so a differing
    id would be a third object hiding under a twin's name."""
    t = roster["targets"][target]
    v = t["views"][view]
    m = _PAIRED_VIEW_RE.match(view)
    if "paired_of" not in v:
        if m:
            raise SystemExit(f"{target}/{view}: named like a paired view but carries no `paired_of:` key — a paired view declares "
                             "the seed view it re-serves (the twin marker)")
        return None
    seed = v["paired_of"]
    if not m or m.group("seed") != seed:
        raise SystemExit(f"{target}/{view}: `paired_of: {seed!r}` but a paired view is named `<seed>p` — the alias must be readable "
                         f"from the name (`{seed}p`)")
    if seed not in t["views"]:
        raise SystemExit(f"{target}/{view}: `paired_of: {seed}` names a view {target} does not carry")
    seed_cells = t["views"][seed].get("cells") or {}
    own = v.get("cells") or {}
    reserved = {cid: sid for cid, sid in own.items() if cid in seed_cells}
    if not reserved:
        raise SystemExit(f"{target}/{view}: `paired_of: {seed}` but no cell is re-served from it — a paired view that shares nothing "
                         "with its seed view is not an alias")
    bad = {cid: (sid, seed_cells[cid]) for cid, sid in reserved.items() if sid != seed_cells[cid]}
    if bad:
        raise SystemExit(f"{target}/{view}: re-served cell(s) differ from {seed}'s staged ids: {bad} — a paired view re-serves the "
                         "seed view's OBJECTS, never a third one under the twin's name")
    return seed


def paired_reserved_cells(roster: dict, target: str, view: str) -> dict:
    """``{config cell id: staged id}`` a paired view RE-SERVES from its seed view (empty for every other view)."""
    seed = view_paired_of(roster, target, view)
    if seed is None:
        return {}
    seed_cells = roster["targets"][target]["views"][seed].get("cells") or {}
    return {cid: sid for cid, sid in (roster["targets"][target]["views"][view].get("cells") or {}).items() if cid in seed_cells}

def view_dose(roster: dict, target: str, view: str):
    """The dose a view serves, or None for a non-dose view — REFUSING the two ways the name and the
    `dose:` key can disagree: a dose-named view (`d<tok>` / `it_d<tok>`) with no key, and a keyed view
    whose name spells another dose (or none). The `it_` prefix is required exactly on a target that has
    an instruct-direct (`it`) view, so a chat-checkpoint dose view cannot borrow the instruct spelling.
    REFUSAL ORDER, most specific diagnosis first: the ladder-family spelling (`it_` / bare vs `dose_families`),
    then the dose spelling (name vs `dose:`), then the OBJECT family (`view_object_family`: the frozen bare
    allow-list, the `object_family:` key, the cells), then the unit — which is the family's (DOSE_UNITS), so
    it cannot be checked before the family holds. An `it_` view on a chat-checkpoint target is told the target
    has no instruct family, never that its spelling is off the frozen allow-list."""
    t = roster["targets"][target]
    v = t["views"][view]
    m = _DOSE_VIEW_RE.match(view)
    if "dose" not in v:
        if m:
            raise SystemExit(f"{target}/{view}: named like a dose view but carries no `dose:` key — a dose "
                             "view declares its dose (the DOSE LADDER block in the roster)")
        if "dose_unit" in v:
            raise SystemExit(f"{target}/{view}: carries `dose_unit:` but no `dose:` — a unit without a dose names nothing")
        if "object_family" in v:
            raise SystemExit(f"{target}/{view}: carries `object_family:` but no `dose:` — the object family is a dose view's "
                             "declaration (`d<tok>_<family>`); a non-dose view names its cells outright")
        return None
    dose = v["dose"]
    if isinstance(dose, bool) or not isinstance(dose, (int, float)):
        raise SystemExit(f"{target}/{view}: dose must be a number, got {dose!r}")
    # WHICH FAMILY the spelling names. `it_d<tok>` is the INSTRUCT-DIRECT family (objects fit on the instruct
    # checkpoint, served natively); bare `d<tok>` is the BASE-FIT family (objects fit on the base checkpoint —
    # served natively on a chat-checkpoint target, TRANSFERRED onto the instruct target on llama8b). A target's
    # ladder families are a roster fact (`dose_families:`; default: the instruct family iff the target has an
    # `it` view, else the base family), so a target that carries both (llama8b) says so, and a target that
    # carries one cannot borrow the other's spelling.
    families = dose_families(roster, target)
    it_prefixed = view.startswith("it_")
    if it_prefixed and "it" not in families:
        raise SystemExit(f"{target}/{view}: `it_` spelling but this target's ladder families are {families} — "
                         "a chat-checkpoint dose view is `d<tok>`")
    if not it_prefixed and "main" not in families:
        raise SystemExit(f"{target}/{view}: bare `d<tok>` spelling but this target's ladder families are {families} — "
                         "an instruct-direct dose view is `it_d<tok>`")
    segment = m.group("objfam") if m else None     # the OBJECT-FAMILY segment, kept in the spelling the dose must match
    want = ("it_" if it_prefixed else "") + "d" + dose_token(dose) + (f"_{segment}" if segment else "")
    if view != want:
        raise SystemExit(f"{target}/{view}: dose {dose} spells view `{want}`, not `{view}` — the dose must be "
                         "parseable from the view name (roster DOSE LADDER naming rule)")
    # The OBJECT family LAST among the name checks (2026-09-17), so the most specific diagnosis wins: only a view whose
    # ladder spelling and dose spelling already hold is held to the frozen bare allow-list, its `object_family:` key and
    # its cells. Before this ordering an `it_d<tok>` view on a chat-checkpoint target was refused as "a NEW bare dose
    # view — name the object family", sending the author to the wrong fix. The unit is the FAMILY's (DOSE_UNITS), so it
    # is checked once the family holds and never before.
    fam = view_object_family(roster, target, view)   # the family the name spells holds, key and cells alike (seedless)
    want_unit = DOSE_UNITS[family_of_segment(fam)]
    if v.get("dose_unit") != want_unit:
        raise SystemExit(f"{target}/{view}: a dose view declares `dose_unit: {want_unit}` (got {v.get('dose_unit')!r}) — "
                         "the unit the dose axis is measured in is a roster fact, not an inference (D66), and it is the "
                         "unit of the view's OBJECT FAMILY: a LoRA dose scales the write basis, a ReLU dose scales the "
                         "direction, and the two axes are not interchangeable")
    return float(dose)


#: the ONE per-VIEW eval-time cap the roster grammar admits (`views.<view>.caps: {bon: N}`; 2026-09-15): a view of a
#: target WITHOUT a caps table may write its own BoN completion cap — the NATIVE best-of-75 pass at N (no 512 store, no climb:
#: the sdfbase-llama8b route) — so a lane can take the native route on ITS views while every other view of the target keeps
#: the campaign's 512 store byte-for-byte. Any other key is refused: a view declares no other cap (the multi-turn / static /
#: controls caps are the roster's wave-wide values).
VIEW_CAP_KEYS = ("bon",)


#: the Petri TARGET decode contract a VIEW may declare (`views.<view>.petri_target_decode`): the EXACT registered
#: decode-policy ref of a `petri` row WITH its version — `<model_id>|petri|<thinking>|<mode>@v<N>`
#: (sisteer.registry.decode_policy's ref grammar; PolicyFile.by_ref is the ONLY lookup, at hash time). The manifest renders it as
#: `target_decode: {mode: vendor, policy: '<ref>'}` after `prompted_prefix_hashes` (manifest.PetriTargetDecode; a harness-hash input
#: the standup and petri_run.py hash identically). A view without the knob renders no block = today's harness decode, byte for byte.
_PETRI_DECODE_REF_RE = re.compile(r"^[a-z0-9_]+\|petri\|(?:off|on)\|[a-z0-9_]+@v[1-9][0-9]*$")
PETRI_TARGET_DECODE_KEY = "petri_target_decode"
PETRI_TARGET_DECODE_MODE = "vendor"


def view_petri_target_decode(roster: dict, target: str, view: str) -> str | None:
    """The Petri vendor-decode ref a VIEW declares (`views.<view>.petri_target_decode`), or None when the view declares none.
    Refuses (SystemExit naming the view) a value that is not a string, or a string that is not a versioned `petri` decode-policy ref —
    a `bon` / `multi_turn` row, a ref without `@v<N>`, a bare mode word — so a mistyped knob never renders a block the loader or the
    registry would refuse at standup."""
    t = roster["targets"].get(target) or {}
    v = (t.get("views") or {}).get(view) or {}
    ref = v.get(PETRI_TARGET_DECODE_KEY)
    if ref is None:
        return None
    where = f"{target}/{view}: views.{view}.{PETRI_TARGET_DECODE_KEY}"
    if not isinstance(ref, str) or not ref.strip():
        raise SystemExit(f"{where} must be the exact registered ref of a `petri` decode-policy row (a non-empty string), got {ref!r}")
    if not _PETRI_DECODE_REF_RE.match(ref):
        raise SystemExit(f"{where}={ref!r} is not a versioned `petri` decode-policy ref (`<model_id>|petri|<off|on>|<mode>@v<N>`, "
                         "e.g. 'qwen35_4b_it|petri|off|vendor_preset@v1') — the manifest block names the exact row PolicyFile.by_ref resolves")
    return ref


#: the Petri JUDGE fallback chain a VIEW may declare (`views.<view>.petri_judge_fallback_models` /
#: manifest.PetriSpec.judge_fallback_models, landed 181ac97f4): a non-empty list of inspect model strings, rendered as the
#: manifest's `judge_fallback_models:` block right after `auditor_fallback_models:` (a harness-hash input when non-empty --
#: role_models.judge_fallbacks). A view without the knob renders NO block = no judge chain (the field's omission rule), byte
#: for byte. The chain is spelled in the ROSTER, never hand-typed into a generated manifest: the generator holds every committed
#: manifest byte-identical to its render, so a hand edit is drift and a regeneration would erase it (the Arditi lane's generator
#: learnt the same lesson the same day: PETRI_FALLBACK_CHAINS in render_arditi_seedrep_configs.py).
PETRI_JUDGE_FALLBACK_KEY = "petri_judge_fallback_models"
#: the AUDITOR chain a VIEW may declare (`views.<view>.petri_auditor_fallback_models`; claude-opus-4-8 BEFORE grok-4.6 in
#: BOTH chains, order pinned): the same grammar as the judge knob; it REPLACES the
#: template's default auditor chain (DEFAULT_AUDITOR_CHAIN, the one-leg grok chain every other manifest keeps, byte for
#: byte) on that view's manifest. Spelled in the roster, never hand-typed into a generated manifest.
PETRI_AUDITOR_FALLBACK_KEY = "petri_auditor_fallback_models"
DEFAULT_AUDITOR_CHAIN = ["openrouter/x-ai/grok-4.6"]
_INSPECT_MODEL_RE = re.compile(r"^[a-z][a-z0-9_-]*/[A-Za-z0-9._/-]+$")


def _view_petri_chain(roster: dict, target: str, view: str, key: str) -> list[str] | None:
    """A fallback chain a VIEW declares under ``key``, or None when the view declares none. Refuses (SystemExit naming
    the view) a value that is not a NON-EMPTY list of distinct `<provider>/<model>` strings -- `[]`, a scalar, a null
    member, a repeat -- so a mistyped knob never renders a block the loader would refuse at standup."""
    t = roster["targets"].get(target) or {}
    v = (t.get("views") or {}).get(view) or {}
    chain = v.get(key)
    if chain is None:
        return None
    where = f"{target}/{view}: views.{view}.{key}"
    if not isinstance(chain, list) or not chain or not all(isinstance(m, str) and _INSPECT_MODEL_RE.match(m) for m in chain):
        raise SystemExit(f"{where} must be a NON-EMPTY list of inspect model strings (`<provider>/<model>`, e.g. "
                         f"'openrouter/x-ai/grok-4.6'); omit the key for no judge chain; got {chain!r}")
    if len(set(chain)) != len(chain):
        raise SystemExit(f"{where} repeats a model: {chain!r}")
    return list(chain)


def view_petri_judge_fallback_models(roster: dict, target: str, view: str) -> list[str] | None:
    """The judge fallback chain a VIEW declares (`views.<view>.petri_judge_fallback_models`), or None (= no judge chain)."""
    return _view_petri_chain(roster, target, view, PETRI_JUDGE_FALLBACK_KEY)


def view_petri_auditor_fallback_models(roster: dict, target: str, view: str) -> list[str] | None:
    """The auditor fallback chain a VIEW declares (`views.<view>.petri_auditor_fallback_models`), or None (= the template's
    DEFAULT_AUDITOR_CHAIN)."""
    return _view_petri_chain(roster, target, view, PETRI_AUDITOR_FALLBACK_KEY)


def _petri_auditor_fallback_block(roster: dict, target: str, view: str) -> str:
    """The manifest's `auditor_fallback_models:` block: the view's declared chain or the template's default."""
    chain = view_petri_auditor_fallback_models(roster, target, view)
    if chain is None:
        return "  auditor_fallback_models:\n" + "".join(f"    - {m}\n" for m in DEFAULT_AUDITOR_CHAIN)
    return ("  auditor_fallback_models:        # claude-opus-4-8 BEFORE grok-4.6, order pinned; a harness-hash input\n"
            + "".join(f"    - {m}\n" for m in chain))


def _petri_judge_fallback_block(roster: dict, target: str, view: str) -> str:
    """The manifest's `judge_fallback_models:` block for a view that declares the knob (indented under `petri:`), '' otherwise."""
    chain = view_petri_judge_fallback_models(roster, target, view)
    if chain is None:
        return ""
    return ("  judge_fallback_models:          # grok-4.6 backs the JUDGE too (its returned content_filter refusal switches legs); a harness-hash input\n"
            + "".join(f"    - {m}\n" for m in chain))


#: the PER-REQUEST refusal fallback a VIEW may declare for each Petri role (`views.<view>.petri_auditor_refusal_fallback_models` /
#: `views.<view>.petri_judge_refusal_fallback_models`; 2026-10-01: manifest.PetriSpec.*_refusal_fallback_models): a
#: NON-EMPTY list of first-party `anthropic/...` ids, rendered as the manifest's `auditor_refusal_fallback_models:` /
#: `judge_refusal_fallback_models:` blocks right after `judge_model:`. ADDITIVE -- never a harness-hash input (the role's default
#: model is the instrument; the fallback serves a refused request once, with the serving model recorded per message), so a view
#: gaining the knobs keeps every cell's petri_run_key. A view without them renders NO block, byte for byte.
PETRI_AUDITOR_REFUSAL_FALLBACK_KEY = "petri_auditor_refusal_fallback_models"
PETRI_JUDGE_REFUSAL_FALLBACK_KEY = "petri_judge_refusal_fallback_models"
_REFUSAL_FALLBACK_PROVIDER = "anthropic/"
#: the JUDGE key may END with exactly ONE OpenRouter leg (Grok 4.6, the terminal judge after every
#: Claude leg refused) -- mirrors manifest.PETRI_REFUSAL_FALLBACK_TERMINAL_{ROLE,PROVIDER}; the auditor key stays first-party
_REFUSAL_FALLBACK_TERMINAL_PROVIDER = "openrouter/"
_REFUSAL_FALLBACK_TERMINAL_TAIL = " -- the terminal non-Claude leg (OpenRouter) serves after every Claude leg refused"


def _petri_refusal_fallback_blocks(roster: dict, target: str, view: str) -> str:
    """The manifest's `auditor_refusal_fallback_models:` / `judge_refusal_fallback_models:` blocks for a view that declares the
    knobs (indented under `petri:`), '' otherwise. A non-first-party id is refused by name (the Anthropic API's server-side
    fallback cannot serve it)."""
    out = ""
    for key, field in ((PETRI_AUDITOR_REFUSAL_FALLBACK_KEY, "auditor_refusal_fallback_models"),
                       (PETRI_JUDGE_REFUSAL_FALLBACK_KEY, "judge_refusal_fallback_models")):
        chain = _view_petri_chain(roster, target, view, key)
        if chain is None:
            continue
        bad = [m for m in chain if not m.startswith(_REFUSAL_FALLBACK_PROVIDER)]
        # ONE terminal OpenRouter leg is admitted for the JUDGE key; two non-Claude legs, a non-terminal one, another provider
        # or any non-Claude AUDITOR leg refuse by name here too, so a mistyped roster never renders a block the loader refuses
        terminal_ok = (key == PETRI_JUDGE_REFUSAL_FALLBACK_KEY and len(bad) == 1 and bad[0] == chain[-1]
                       and chain[-1].startswith(_REFUSAL_FALLBACK_TERMINAL_PROVIDER))
        if terminal_ok and not (len(chain[-1]) > 9 and chain[-1][-9] == "-" and chain[-1][-8:].isdigit()):
            # the terminal leg is the IMMUTABLE dated build (OpenRouter's canonical slug), never a mutable alias
            raise SystemExit(f"{target}/{view}: views.{view}.{key} terminal leg {chain[-1]!r} is a mutable alias: name the IMMUTABLE "
                             f"DATED build (…-YYYYMMDD, e.g. openrouter/x-ai/grok-4.6-20260810) (D896)")
        if bad and not terminal_ok:
            raise SystemExit(f"{target}/{view}: views.{view}.{key} names {bad}: the per-request refusal fallback is the "
                             f"Anthropic API's server-side fallback and takes first-party `{_REFUSAL_FALLBACK_PROVIDER}...` ids only, "
                             f"except ONE TERMINAL `{_REFUSAL_FALLBACK_TERMINAL_PROVIDER}...` leg for the judge (D869)")
        tail = _REFUSAL_FALLBACK_TERMINAL_TAIL if terminal_ok else ""
        out += (f"  {field}:  # the Anthropic API's PER-REQUEST refusal fallback -- a request the "
                "default declines with a safety refusal is served once by this model, the next request is at the default; the serving "
                f"model is recorded on every message; ADDITIVE, not a harness-hash input{tail}\n"
                + "".join(f"    - {m}\n" for m in chain))
    return out


def _petri_target_decode_block(roster: dict, target: str, view: str) -> str:
    """The manifest's `target_decode:` block for a view that declares the knob (indented under `petri:`), '' otherwise."""
    ref = view_petri_target_decode(roster, target, view)
    if ref is None:
        return ""
    return (f"  target_decode:                  # the vendor row on every TARGET request; a harness-hash input\n"
            f"    mode: {PETRI_TARGET_DECODE_MODE}\n"
            f"    policy: '{ref}'\n")


def view_declared_bon_cap(roster: dict, target: str, view: str) -> int | None:
    """The BoN cap a VIEW declares for itself (`views.<view>.caps.bon`), or None when the view declares none.

    Three refusals, each a SystemExit naming the view: (1) the view's target also declares a caps table — its
    `caps.bon` is WRITTEN for every view of the target, so a view value would be a second authority for one
    cap; (2) the view is a dose view (named `d<tok>` / `it_d<tok>` or carrying `dose:`) — a dose view moves the
    OBJECT, never a cap; (3) the block carries a key outside VIEW_CAP_KEYS, names no `bon`, or its value
    is not a positive int (a `pending` marker is not a cap)."""
    t = roster["targets"].get(target) or {}
    v = (t.get("views") or {}).get(view) or {}
    caps = v.get("caps")
    if caps is None:
        return None
    where = f"{target}/{view}: views.{view}.caps"
    if not isinstance(caps, dict):
        raise SystemExit(f"{where} must be a mapping {{bon: <positive int>}} (got {caps!r})")
    unknown = sorted(set(caps) - set(VIEW_CAP_KEYS))
    if unknown:
        raise SystemExit(f"{where} carries unknown keys {unknown} — a view may declare only {list(VIEW_CAP_KEYS)} "
                         "(every other cap is the roster's wave-wide value or the target's caps table)")
    if "bon" not in caps:
        raise SystemExit(f"{where} names no cap ({list(VIEW_CAP_KEYS)})")
    cap = caps["bon"]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise SystemExit(f"{where}.bon={cap!r} must be a positive int (a pending entry is not a cap)")
    if t.get("caps") is not None:
        raise SystemExit(f"{where}.bon={cap}: the target {target} declares a caps table whose caps.bon is WRITTEN for every "
                         "view of it (D194) — a view of a caps-table target may not carry a second value for one cap")
    if "dose" in v or _DOSE_VIEW_RE.match(view):
        raise SystemExit(f"{where}.bon={cap}: a dose view moves the OBJECT, never a cap (D33) — the per-view BoN cap is "
                         "refused on a dose view")
    return int(cap)


def _view_bon_cap(roster: dict, target: str, view: str) -> int:
    """The `bon.max_new_tokens` a rendered (target, view) config MUST carry: the view's own `caps.bon`
    (view_declared_bon_cap, with its refusals), else the target's caps-table `caps.bon`, else the
    campaign's BON_MAX_NEW_TOKENS (512). The renderer writes this value and `_refuse_bon_drift` holds the
    rendered config to it — one function, both ends."""
    vcap = view_declared_bon_cap(roster, target, view)
    if vcap is not None:
        return vcap
    tcap = ((roster["targets"].get(target) or {}).get("caps") or {}).get("bon")
    return int(tcap) if tcap is not None else BON_MAX_NEW_TOKENS


def _refuse_bon_drift(target: str, view: str, rendered: str, body_text: str, roster: dict | None = None) -> None:
    """The generator's own refusal for the two caps a dose lane once moved and must not: parse what
    was rendered and hold it to the body for EVERY view, dose views included — the `bon` block identical,
    its cap `_view_bon_cap` (the body's BON_MAX_NEW_TOKENS unless the target's caps table or the view's own
    `caps.bon` writes it; Scale's 512 -> 4096 climb tooling refuses a store otherwise), and
    `attacks.target_max_new_tokens` the body's (multi-turn and the judged controls read it)."""
    cfg, body = yaml.safe_load(rendered), yaml.safe_load(body_text)
    # The canonical best-of-N landing (main, 2026-09-12) ADDS two climb keys to every rendered view's `bon`
    # that the BODY does not carry, so an exact-equality test now refuses every view including `main`.
    # The invariant is unchanged and is still what this guard enforces — a dose lane must not move a cap.
    # So strip exactly those two, holding them to the roster's wave-wide values (the same statement main's
    # own wave test makes), and hold everything else to the body byte for byte.
    roster = roster if roster is not None else load_roster()   # the caller's roster when rendering (a test may move a cap in memory)
    bon, bon_body = dict(cfg.get("bon") or {}), dict(body.get("bon") or {})
    # A TARGET whose roster caps table WRITES the BoN cap (sdfbase-llama8b: the NATIVE 4,096 best-of-N pass — no 512 store, no
    # climb) is held to THAT cap on every one of its views instead of the body's 512: the declaration is per target in the roster,
    # so a view still cannot move a cap on its own — EXCEPT through the one per-view grammar, `views.<view>.caps.bon`
    # (view_declared_bon_cap; 2026-09-15), admitted only on a target with NO caps table and never on a dose view. Both ends of the
    # generator read the cap through _view_bon_cap, so the value written and the value held to are one function.
    target_bon_cap = ((roster["targets"].get(target) or {}).get("caps") or {}).get("bon")
    view_bon_cap = view_declared_bon_cap(roster, target, view)     # its refusals fire here too: a caps-table target, a dose view
    want_cap = _view_bon_cap(roster, target, view)
    written = target_bon_cap is not None or view_bon_cap is not None
    if written:
        if bon.get("max_new_tokens") != want_cap:
            who = (f"the target's caps.bon {want_cap} (the caps table WRITES this cap for every view of {target}; D194)"
                   if target_bon_cap is not None else
                   f"the view's caps.bon {want_cap} (views.{view}.caps writes this cap for THIS view only: the native best-of-N "
                   "route; every other view of the target keeps the body's cap)")
            raise SystemExit(f"{target}/{view}: bon.max_new_tokens={bon.get('max_new_tokens')!r} != {who}")
        bon.pop("max_new_tokens", None); bon_body.pop("max_new_tokens", None)
    for key, want in (("climb_cap", int(roster["bon_climb_cap"])),
                      ("climb_decode", str(roster["bon_climb_decode"]))):
        got, in_body = bon.pop(key, None), bon_body.pop(key, None)
        if got != want:
            raise SystemExit(f"{target}/{view}: bon.{key}={got!r} != the roster's {want!r} — the climb is "
                             "wave-wide; a view may not carry its own")
        if in_body is not None and in_body != want:
            raise SystemExit(f"{target}/{view}: the BODY's bon.{key}={in_body!r} != the roster's {want!r}")
    if bon != bon_body:
        raise SystemExit(f"{target}/{view}: bon block drifted from the body — every view keeps the campaign's "
                         "BoN block apart from the wave-wide climb keys (D33)")
    if not written and (cfg.get("bon") or {}).get("max_new_tokens") != BON_MAX_NEW_TOKENS:
        raise SystemExit(f"{target}/{view}: bon.max_new_tokens is not {BON_MAX_NEW_TOKENS} — Scale's 512 -> 4096 "
                         "climb tooling refuses the store (D33)")
    # The cap invariant, restated against the ROSTER rather than the body text. Two things changed under
    # this guard on 2026-09-12: the multi-turn change moved `target_max_new_tokens` to the roster's wave-wide
    # 4,096, and the judged controls got their OWN key, `controls_max_new_tokens` (cell_controls.controls_cap
    # is the one reader). The generator SUBSTITUTES both into the body it renders, so comparing the rendered
    # value to the PRE-substitution body text compares a value against its own template and refuses every
    # view. The invariant was never about the template: it is that a view may not carry a cap of its own.
    for key in ("target_max_new_tokens", "controls_max_new_tokens"):
        want = int(roster[key])
        if cfg["attacks"].get(key) != want:
            raise SystemExit(f"{target}/{view}: attacks.{key}={cfg['attacks'].get(key)!r} != the roster's "
                             f"wave-wide {want} — a view may not carry its own cap (D33)")


def roster_rel(roster_path: Path = ROSTER) -> str:
    """The roster's path as the rendered headers name it: repo-relative POSIX
    (``configs/sdfbase_full22_roster.yaml`` for the default — the literal every
    committed header carries), or the absolute path for a roster outside the repo."""
    p = Path(roster_path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def _cert_roster_note(roster_path: Path) -> str:
    """The one header paragraph a NON-production roster's outputs carry (empty for the
    production roster, whose files stay byte-identical): what the file is and what reads it."""
    rel = roster_rel(roster_path)
    if rel == roster_rel(ROSTER):
        return ""
    return (f"#   * a CERTIFICATION-ONLY view: rendered from the roster {rel}, NOT the\n"
            f"#     production roster {roster_rel(ROSTER)} (which names no such view; it is never edited for a\n"
            "#     certification). Consumed by the certification driver and by the smoke run with\n"
            f"#     F22_ROSTER={rel} (common.sh passes it to every roster reader); never a production wave's input.\n")


def load_pins() -> dict:
    return json.loads(PINS.read_text(encoding="utf-8")) if PINS.is_file() else {}


def pin_form_facts(target: str, cell_id: str, mapping: dict) -> tuple[str, str, str, str]:
    """``(staged_id, source bank, object_sha256, pin reference)`` for a cell the roster names
    through the CROSS-BANK PIN FORM (``{pins: <file>, pin: <model>.<cell>}``) — read from the
    named pin file, which is the authority for every one of them.

    The generator only ECHOES these into the config header; the bank-view builder
    (``full22/make_bank_view.py``) is what holds the bytes to the pin. Both refuse the same
    shapes by name, and neither ever reads a staged id from the roster: the form has no field
    for one, so a config can never claim a rung the pin did not choose."""
    unknown = sorted(set(mapping) - {"pins", "pin"})
    if unknown or not mapping.get("pins") or not mapping.get("pin"):
        raise SystemExit(f"{target}/{cell_id}: the roster's mapping form takes exactly "
                         f"['pins', 'pin'] (unknown {unknown}) — every identity fact comes from "
                         "the pin, so there is no staged_id field to name a rung with")
    pins_path = REPO / str(mapping["pins"])
    if not pins_path.is_file():
        raise SystemExit(f"{target}/{cell_id}: no pins file at {pins_path}")
    ref = str(mapping["pin"])
    parts = ref.split(".")
    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"{target}/{cell_id}: pin reference {ref!r} is not `<model key>.<cell>`")
    doc = json.loads(pins_path.read_text(encoding="utf-8"))
    pin = (doc.get(parts[0]) or {}).get(parts[1])
    if not isinstance(pin, dict):
        raise SystemExit(f"{target}/{cell_id}: {mapping['pins']} has no {ref} entry")
    missing = [k for k in ("staged_id", "bank", "object_sha256") if not pin.get(k)]
    if missing:
        raise SystemExit(f"{target}/{cell_id}: pin {ref} of {mapping['pins']} lacks {missing}")
    return str(pin["staged_id"]), str(pin["bank"]), str(pin["object_sha256"]), ref


DERIVED_LANE = "derived"
REGISTRY_CAMPAIGN = "sdfbase_full22"


def registry_campaign_default() -> str:
    """The registry campaign the front door's pins are read under when no caller names one — 04_fit_objects.sh's OWN resolution, restated
    (its `CAMPAIGN="${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"` line, which tests/test_front_door_campaign_completion.py and
    test_fit_env_name_plumbing.py pin, mirrored here by tests/test_refusal_direction_pin.py): the site's SISTEER_REGISTRY_CAMPAIGN (config.env,
    exported into every stage's environment by repro_load_config, so the generator 05 runs sees it), else the job-env spelling REGISTRY_CAMPAIGN,
    else the paper frame's sdfbase_full22 (an empty value is unset, as `${X:-}` reads it). 04 files EVERY fit under that value — the refusal
    direction's row included (job_env hands REGISTRY_CAMPAIGN to the fitter's --campaign) — and reads each unit's done-ness under it
    (registry_done's `campaign='$CAMPAIGN'`, refusal_registry_ids' `--campaign "$CAMPAIGN"`), so the pins writers must look where 04 wrote
    (arditi_pins_from_registry read sdfbase_full22 alone, so a site naming another campaign fitted and registered its
    refusal direction, `--derived-pins-from-registry` and 05's `--pins-from` found no row and wrote no pin, and 05 refused to stage the
    completed fit). Every registry-backed pins writer below (derived, declared base, refusal direction, dose copies, the parent check) takes
    `campaign` and resolves None through this function; main's --campaign names it for every mode, and 05 names it explicitly, as it names
    the root. The campaign a pin was read under is spelled in its `source` (registry:<campaign>/…) and in every `pins:` line."""
    return os.environ.get("SISTEER_REGISTRY_CAMPAIGN") or os.environ.get("REGISTRY_CAMPAIGN") or REGISTRY_CAMPAIGN


def registry_root_default() -> Path:
    """The registry root the reproduction reads (SISTEER_REGISTRY_ROOT with `~` expanded — reproduce/lib/common.sh repro_registry_root,
    the root 04_fit_objects.sh registers into), else the registry package's own default."""
    env = os.environ.get("SISTEER_REGISTRY_ROOT")
    if env:
        return Path(os.path.expanduser(env))
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import db as DB
    return Path(DB.DEFAULT_ROOT)


def derived_registry_rows(root: "Path | str", *, fit_model_id: str, object_kind: str, paper_label: str, campaign: str = REGISTRY_CAMPAIGN,
                          base_scale: "float | None" = None) -> list:
    """The registry's reference / approved object rows for ONE derived identity, read-only ([] when it holds none). The object row
    carries no cell id: the identity is the model's FIT checkpoint (models.yaml fit.checkpoint.model_id — the id 04_fit_objects.sh
    registers under), the derived entry's `object_kind`, the cell's paper label (its models-table name) and the campaign — the set
    04's registry_done reads — AND the strength lane: a dose-scaled COPY of a derived object (derivation dose_scale,
    `base_scale` = its dose; the qwen35_4b relu ladder, the front door's d0p75_relu) shares every identity column with its parent, so the
    parent's pin (the base label: `base_scale` NULL — a composition / subtraction at full strength) and a dose view's pin (`base_scale` ==
    the view's dose) are resolved as DIFFERENT identities and never compete; two rows in ONE lane stay a refusal by name (the caller's
    "one object must be THE reference"). A registry that is not on disk refuses by name (never read as empty)."""
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import db as DB
    path = DB.db_path(str(root))
    if not Path(path).is_file():
        raise SystemExit(f"registry {path} is not on disk — a derived pin is written from the REGISTRY object row (D1319); "
                         "point SISTEER_REGISTRY_ROOT / --registry-root at the grading host's registry")
    conn = DB.connect(path, readonly=True)
    try:
        lane_sql, lane_args = ("AND base_scale IS NULL", ()) if base_scale is None else ("AND base_scale = ?", (float(base_scale),))
        rows = conn.execute("SELECT object_id, status, object_kind, paper_label, fit_model_id, campaign, base_scale, derivation, parent_object_ids_json FROM object "
                            "WHERE campaign=? AND fit_model_id=? AND object_kind=? AND paper_label=? AND status IN ('reference', 'approved') "
                            + lane_sql + " ORDER BY status, object_id", (campaign, fit_model_id, object_kind, paper_label, *lane_args)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def registry_object_statuses(root: "Path | str", ids: "list[str]") -> dict:
    """{object_id: [status, ...]} for every id the registry holds an object row for, read-only, one chunked query (a pin is
    registry-backed or not written — a staging manifest row whose digest has no reference/approved row is not a pin). A registry that is not
    on disk refuses by name (never read as empty), as derived_registry_rows does."""
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import db as DB
    path = DB.db_path(str(root))
    if not Path(path).is_file():
        raise SystemExit(f"registry {path} is not on disk — a derived pin is written from the REGISTRY object row (D1319); "
                         "point SISTEER_REGISTRY_ROOT / --registry-root at the grading host's registry")
    out: dict = {}
    ids = sorted({str(i) for i in ids if i})
    conn = DB.connect(path, readonly=True)
    try:
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            for oid, status in conn.execute(f"SELECT object_id, status FROM object WHERE object_id IN ({','.join('?' * len(chunk))})", chunk).fetchall():
                out.setdefault(str(oid), []).append(str(status))
    finally:
        conn.close()
    return out


#: demo-import (M1, Q12 ruled NARROW): a staging manifest row whose `source_path` is a COMMITTED object of the checkout (stage_objects.py's
#: `committed:<repo-relative dir>` source; reproduce/tools/import_objects.py cells writes the rows). The ONE lookup bypassed for such a row is
#: `derived_registry_rows` — the `(campaign=<this run's: the site's, sdfbase_full22 by default>, fit_model_id=<the front-door FIT id>, object_kind, paper_label[, base_scale])` query,
#: which can never find a paper row (filed under the paper's fit checkpoint and its own campaign: `reseed` for the LoRA difference) — at its three
#: committed-path call sites (the derived loop; dose_pins_from_registry's copy and parent lookups). Every refusal those lookups fed is KEPT,
#: re-keyed on the digest's OWN live row: reference|approved status, derivation (lora_subtraction | composition for a difference, dose_scale for a
#: copy), parents == the committed sidecar's, a copy's base_scale == the view's dose and its parent == the pinned parent of the same block.
COMMITTED_SOURCE_PREFIX = "committed:"


def is_committed_row(mrow: "dict | None") -> bool:
    """A manifest row staged from a committed object directory (its `source_path`, the field the stager writes — never a `source` key)."""
    return bool(mrow) and str(mrow.get("source_path") or "").startswith(COMMITTED_SOURCE_PREFIX)


#: the `source` spellings of a committed PIN entry (import_objects.py pins: `objects/DEMO.json:<label>`, `objects/MANIFEST.json:<id> (dose … copy …)`;
#: this writer's committed branches: `committed:<dir> (registry …)`)
COMMITTED_PIN_SOURCES = ("objects/DEMO.json", "objects/MANIFEST.json", COMMITTED_SOURCE_PREFIX)


def is_committed_pin(entry: "dict | None") -> bool:
    return bool(entry) and str(entry.get("source") or "").startswith(COMMITTED_PIN_SOURCES)


def committed_pin_still_backed(entry: dict, *, registry_root: "Path | str", ident: str) -> None:
    """The rule for a committed pin entry the staging manifest does not carry (kept verbatim): the registry must still stand behind it — by
    the digest's OWN row (reference|approved), the identity a committed object is filed under (never the demo-keyed lookup, which can never find a
    paper row); a demoted / withdrawn / superseded / absent row refuses by name."""
    oid = str(entry.get("object_sha256") or "")
    row = live_object_identity(registry_root, oid)
    if row is None or str(row.get("status")) not in ("reference", "approved"):
        raise SystemExit(f"{ident}: the committed pin {oid[:12]} ({entry.get('source')}) has NO reference/approved registry row"
                         + (f" (status {row.get('status')!r})" if row else "") + " — the pin IS the registered object, and a pin the registry no longer stands "
                         "behind cannot be written as if it did; restore the row (import the committed objects) or retire the pin; refusing to write the pins")


def live_object_identity(root: "Path | str", object_id: str) -> "dict | None":
    """The registry's object row for ONE id with its identity columns (read-only); None when it holds none. A registry that is not on disk
    refuses by name, as derived_registry_rows does."""
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import db as DB
    path = DB.db_path(str(root))
    if not Path(path).is_file():
        raise SystemExit(f"registry {path} is not on disk — a committed pin is held to the REGISTRY object row (D1319); "
                         "point SISTEER_REGISTRY_ROOT / --registry-root at the grading host's registry")
    conn = DB.connect(path, readonly=True)
    try:
        r = conn.execute("SELECT object_id, status, object_kind, paper_label, fit_model_id, campaign, base_scale, derivation, parent_object_ids_json "
                         "FROM object WHERE object_id = ?", (object_id,)).fetchone()
        return dict(r) if r is not None else None
    finally:
        conn.close()


def committed_sidecar(mrow: dict, ident: str) -> dict:
    """The committed row's identity sidecar as staged in the bank (`local_path`): derived_meta.json, else train_budget_meta.json; a row
    without a readable one refuses by name (the parents a committed pin is held to are the sidecar's)."""
    lp = mrow.get("local_path")
    if not lp:
        raise SystemExit(f"{ident}: the committed staging row carries no local_path — its sidecar's parents cannot be read; refusing to write the pins")
    for name in ("derived_meta.json", "train_budget_meta.json"):
        p = Path(str(lp)) / name
        if p.is_file():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise SystemExit(f"{ident}: {p} is not JSON ({exc}); refusing to write the pins")
            if isinstance(doc, dict):
                return doc
    raise SystemExit(f"{ident}: no derived_meta.json / train_budget_meta.json under {lp} — the committed row's parents cannot be read; refusing to write the pins")


def committed_derived_pin(mrow: dict, *, registry_root: "Path | str", key: str, sid: str, cell: str, kind: str, label: str, have: "dict | None") -> dict:
    """The pin of a committed DIFFERENCE row (lane derived), held to the digest's own live row: reference|approved, derivation lora_subtraction |
    composition, object_kind / paper_label the table's, parents == the committed derived_meta.json's {parent_pos, parent_neg}; a committed entry
    with the same id is kept VERBATIM (import_objects.py pins' spelling), else the row's facts are written."""
    msha = str(mrow.get("object_sha256") or "")
    ident = f"{key}/{sid} ({cell}; committed {mrow.get('source_path')}, {kind})"
    row = live_object_identity(registry_root, msha)
    if row is None or str(row.get("status")) not in ("reference", "approved"):
        raise SystemExit(f"{ident}: the committed object {msha[:12]} has no reference/approved registry row"
                         + (f" (status {row.get('status')!r})" if row else "") + " — a pin is registry-backed or not written (S2RLFIX3); import the "
                         "committed objects first (reproduce/tools/import_objects.py registry); refusing to write the pins")
    if str(row.get("derivation") or "") not in ("lora_subtraction", "composition"):
        raise SystemExit(f"{ident}: the registry row {msha[:12]} has derivation {str(row.get('derivation'))!r}, not a difference's (lora_subtraction | composition); "
                         "refusing to write the pins")
    if str(row.get("object_kind")) != kind or str(row.get("paper_label")) != label:
        raise SystemExit(f"{ident}: the registry row {msha[:12]} is {row.get('object_kind')}/{row.get('paper_label')}, the table declares {kind}/{label}; refusing to write the pins")
    dm = committed_sidecar(mrow, ident)
    want = [str(((dm.get(r) or {}) if isinstance(dm.get(r), dict) else {}).get("object_sha256") or "") for r in ("parent_pos", "parent_neg")]
    if not all(want):
        raise SystemExit(f"{ident}: the committed derived_meta.json names no parent_pos / parent_neg object_sha256; refusing to write the pins")
    got = [str(x) for x in (_json_col(row, "parent_object_ids_json", []) or [])]
    if sorted(got) != sorted(want):
        raise SystemExit(f"{ident}: the registry row {msha[:12]} names parents {sorted(x[:12] for x in got)}, the committed derived_meta.json's are "
                         f"{sorted(x[:12] for x in want)}; refusing to write the pins")
    if have and str(have.get("object_sha256")) == msha:
        return dict(have)
    return {"object_sha256": msha, "source": str(mrow.get("source_path")) + f" (registry {row.get('campaign')}:{row.get('status')})",
            "verified_cluster_a": bool(mrow.get("cluster_a_verified")), "verified_cluster_b": bool(mrow.get("cluster_b_verified")),
            "sidecar_object_sha256_recorded": (mrow.get("recorded_hash_check") or {}).get("status")}


def pins_from_staging_manifest(path: Path, *, registry_root: "Path | str | None" = None, models_table: "Path | str | None" = None,
                               roster: "dict | None" = None, committed: "dict | None" = None, allow_drop: "str | None" = None,
                               targets: "set | None" = None, campaign: "str | None" = None) -> dict:
    """{target: {staged_id: {object_sha256, source, verified_cluster_a, verified_cluster_b, sidecar_object_sha256_recorded}}}.

    `targets` (demo-import): the ONE-TARGET write — only the manifest rows and the front-door registry lookups of the named
    targets are read and (re)written through every check below; a committed entry of any OTHER key is kept verbatim, never re-validated here
    (the file aggregates several banks' pins, and a registry that holds one lane's objects alone — a fresh registry after
    `reproduce/tools/import_objects.py registry` — must not be asked to stand behind another lane's pins before that lane is imported).
    None (the default, `--pins-from`) is the whole-file write as before.

    Fitted / arditi / paper-model rows: from the staging agent's manifest — an entry that carries an object sha (as before).
    DERIVED cells of a FRONT-DOOR target (the models table's `objects` with lane `derived`) are pinned by their REGISTRY id:
    exactly one reference / approved row for (fit checkpoint, object_kind, paper label, campaign) writes the pin — whether or not
    the object is staged yet, which is what lets 05 stage a new derived object the first time — in the order 04 builds + registers
    (+ the fold) -> this generator writes the pin from the row -> 05 stages by the pin (stage_objects holds the pulled bytes to it).
    Zero rows: no pin, said so (05 then refuses to stage it); two or more: refused by name (a decision settles which is the
    reference). The staging manifest records what WAS staged and is never a derived pin's source: a front-door derived manifest row
    the registry does not know, or whose digest is not the registry id, refuses by name — as does any
    such row when no registry is given (a library caller without `registry_root` reads the manifest only and touches no registry).
    The paper's models (roster views, no phase views) predate the by-pin stager: their derived rows stay the manifest copies they are.
    The committed file AGGREGATES several banks' pins (copies staged on the grading host,
    cluster_a:/… relu_paired and reseed paired writes, twosi-fits:, arditi_allpos:, registry-pinned ids), so one bank's staging manifest can only
    ever carry a subset. With `committed` (the committed pins; main passes load_pins()): (1) a committed entry the manifest does not carry is KEPT
    VERBATIM (another bank's pin, or a fitted pin pinned ahead of its staging) and said so — never dropped, never refused; `allow_drop` (a stated
    reason, `--allow-dropping-committed-pins`) is the one explicit prune path and prints every entry it prunes; (2) a manifest row whose digest
    differs from the committed entry's id REFUSES naming both ids — a committed pin's identity changes only through its registry re-pin (the
    derived / dose writers), never from a staging row (the row's own expectation can agree with its digest and still not be the committed id).
    With a registry, a manifest row whose digest has NO reference/approved object row is NOT
    pinned and is named (another lane's staged-but-unregistered object: register it first); a committed entry for such an id is kept verbatim
    as any other. A committed FRONT-DOOR derived / dose entry (the registry-owned lanes) absent from the manifest is carried only while its
    registry row is reference/approved — a demoted, withdrawn or superseded row refuses by name (the rule the registry-only writer
    applies), never resurrected by the keep-verbatim branch. Without a registry (a library caller) the manifest is read as before.
    A DECLARED registry-owned base pin (a front-door base-lane object with an object_kind — the s1
    LoRA parents) is diverted to base_pins_from_registry with the manifest row's staging facts, so the manifest path and the registry-only
    path spell it identically (registry:…:<status> (staged from <path>)); its refusals (stale committed id, stale staged copy, row gone) fire
    before the keep-verbatim branch. An undeclared base object (the ReLU parents, the lecun fits) stays a plain stager row.
    `campaign`: the registry campaign every registry-backed lookup below reads under (the derived loop, the base / refusal-direction / dose
    writers) — None resolves to the site's, as 04 resolves it (registry_campaign_default), so the pins are read where 04 registered;
    05 names it explicitly (--campaign), as it names the root."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    fd = front_door_models(models_table) if models_table is not None else FRONT_DOOR_MODELS
    if targets is not None:
        # the one-target write: this manifest's rows of the named targets, and the front-door lookups of those targets alone
        rows = [r for r in rows if str(r.get("target")) in targets]
        fd = {k: m for k, m in fd.items() if k in targets}
    out: dict = {}
    dose_rows: list = []
    # S2RLFIX4: a DECLARED registry-owned base pin (a front-door BASE-lane object with an object_kind: the s1 LoRA parents) is written by
    # base_pins_from_registry from its top-level row — never as a plain stager row, so both writers spell it identically
    base_owned = {(key, f"{BASE_LANE}__{o['cell_id']}") for key, m in fd.items() for cell, o in (m.get("objects") or {}).items()
                  if isinstance(o, dict) and o.get("lane") == BASE_LANE and o.get("object_kind")}
    base_rows: list = []
    # gap-refusal-pin (2026-10-12): a front-door REFUSAL DIRECTION the stager staged from 04's publication (a non-committed row) is written by
    # arditi_pins_from_registry from the row its fit emitted — never as a plain stager row, so both writers spell it identically; the committed
    # sequence's row (source_path committed:…) stays the plain path's, byte-identical
    arditi_owned = {(key, f"{o['lane']}__{o['cell_id']}") for key, m in fd.items() for cell, o in (m.get("objects") or {}).items()
                    if isinstance(o, dict) and o.get("source") == ARDITI_SOURCE}
    arditi_rows: list = []
    changed: list = []   # (target, staged_id, committed id, manifest digest) where a row would change a committed pin's identity
    for r in rows:
        sha = r.get("object_sha256")
        # s3r-fix: a row the stager refused for its digest, or whose recorded digest differs from the registry id it was held to,
        # never reaches the pins — the pin rewrite is the last door a stale object.pt could walk through
        if r.get("status") == "digest_mismatch" or (r.get("object_sha256_expected") and sha and str(r["object_sha256_expected"]) != str(sha)):
            raise SystemExit(f"{path}: {r.get('target')}/{r.get('staged_id')}: the staging manifest row's digest {str(sha or r.get('object_sha256_found'))[:12]} "
                             f"does not match the registered id {str(r.get('object_sha256_expected'))[:12]} (status {r.get('status')}) — refusing to rewrite the pins from it")
        if not sha:
            continue
        have_c = ((committed or {}).get(str(r.get("target"))) or {}).get(str(r.get("staged_id")))
        if have_c and str(have_c.get("object_sha256")) != str(sha):
            changed.append((str(r.get("target")), str(r.get("staged_id")), str(have_c.get("object_sha256")), str(sha))); continue
        if (str(r.get("target")), str(r.get("staged_id"))) in base_owned:
            base_rows.append(r); continue      # S2RLFIX4: a declared registry-owned base pin is written by base_pins_from_registry below
        if (str(r.get("target")), str(r.get("staged_id"))) in arditi_owned and not is_committed_row(r):
            arditi_rows.append(r); continue    # gap-refusal-pin: a fitted refusal direction is written by arditi_pins_from_registry below (its registry row)
        if str(r.get("target")) in fd and _DOSE_SID_RE.match(str(r.get("staged_id") or "")):
            dose_rows.append(r); continue      # dose-pin-fix: a front-door DOSE copy is pinned by its registry row, never from the manifest
        out.setdefault(r["target"], {})[r["staged_id"]] = {
            "object_sha256": sha,
            "source": r.get("volume_path") or r.get("source_path"),
            "verified_cluster_a": bool(r.get("cluster_a_verified")),
            "verified_cluster_b": bool(r.get("cluster_b_verified")),
            "sidecar_object_sha256_recorded": (r.get("recorded_hash_check") or {}).get("status"),
        }
    # -- the front door's DERIVED pins come from the registry --------------------------------------------------------
    if changed:
        raise SystemExit(f"{path}: pins would CHANGE the identity of {len(changed)} committed entr{'y' if len(changed) == 1 else 'ies'} — "
                         + ", ".join(f"{t}/{s}: committed {c[:12]} vs the manifest row's digest {m[:12]}" for t, s, c, m in changed)
                         + " — a committed pin's id changes only through its registry re-pin (the derived / dose writers), never from a staging manifest row "
                         "(S2RLFIX2, D1980 (2)); refusing to write the pins")
    if registry_root is not None:
        # A pin is registry-backed or not written — every manifest-derived entry's digest must have a reference/approved
        # object row; a row without one (another lane's staged, unregistered object: the alphasteer_f22 cells on 2026-10-04) is not pinned and is named
        backed = registry_object_statuses(registry_root, [e["object_sha256"] for ents in out.values() for e in ents.values()])
        for t in sorted(out):
            for sid in sorted(out[t]):
                sha = str(out[t][sid]["object_sha256"]); statuses = backed.get(sha, [])
                if not any(s in ("reference", "approved") for s in statuses):
                    del out[t][sid]
                    print(f"pins: {t}/{sid}: no registry row for {sha}" + (f" (rows: {', '.join(sorted(statuses))}, none reference/approved)" if statuses else "")
                          + " — not pinned (register the object first)")
        for t in [t for t in out if not out[t]]:
            del out[t]
    if base_rows and registry_root is None:
        r0 = base_rows[0]
        raise SystemExit(f"{path}: {r0.get('target')}/{r0.get('staged_id')}: a front-door declared base pin is written from its REGISTRY top-level row (D1425 / "
                         f"S2RLFIX4), never from the staging manifest ({str(r0.get('object_sha256'))[:12]}) — this read was given no registry; run the generator with "
                         f"--registry-root / SISTEER_REGISTRY_ROOT")
    if arditi_rows and registry_root is None:
        r0 = arditi_rows[0]
        raise SystemExit(f"{path}: {r0.get('target')}/{r0.get('staged_id')}: a front-door refusal direction is pinned from its REGISTRY object row (the row its fit emits; "
                         f"D1319), never from the staging manifest ({str(r0.get('object_sha256'))[:12]}) — this read was given no registry; run the generator with "
                         f"--registry-root / SISTEER_REGISTRY_ROOT")
    if registry_root is not None:
        out = base_pins_from_registry(out, registry_root=registry_root, models_table=models_table, manifest_rows=rows, committed=committed,
                                      targets=targets, campaign=campaign)   # S2RLFIX4: one writer, one spelling
        out = arditi_pins_from_registry(out, registry_root=registry_root, models_table=models_table, manifest_rows=rows, committed=committed,
                                        targets=targets, campaign=campaign)   # gap-refusal-pin: the refusal direction by the row its fit emitted, one spelling, under 04's campaign
    by_row = {(str(r.get("target")), str(r.get("staged_id"))): r for r in rows}
    for key, m in sorted(fd.items()):
        fit_model_id = str(((m.get("fit") or {}).get("checkpoint") or {}).get("model_id") or "")
        objs = m.get("objects") or {}
        for cell, o in sorted(objs.items()):
            if not isinstance(o, dict) or o.get("source") != "derived_bank":      # S2RL: every lane the derived build publishes (derived, lecun_derived), never the arditi bank
                continue
            sid = f"{o['lane']}__{o['cell_id']}"; mrow = by_row.get((key, sid)); msha = str((mrow or {}).get("object_sha256") or "")
            label = str(o.get("paper_label") or cell)                                    # S2RL: the registry's paper label (a lecun object spells its own name)
            kind = o.get("object_kind")
            ident = f"{key}/{sid} ({cell}; campaign {campaign}, fit {fit_model_id}, {kind or 'object_kind UNDECLARED'})"
            if registry_root is None:
                if msha:
                    raise SystemExit(f"{path}: {ident}: a front-door derived object is pinned by its REGISTRY id, never from the staging manifest "
                                     f"(D1319) — this read was given no registry; run the generator with --registry-root / SISTEER_REGISTRY_ROOT")
                continue
            if not kind:
                if msha:
                    raise SystemExit(f"{path}: {ident}: the models table declares no object_kind for this derived cell, so its registry identity "
                                     f"cannot be formed, and the staging manifest ({msha[:12]}) is never a derived pin's source (D1319) — declare it")
                print(f"pins: {ident}: no object_kind declared — no registry identity, no pin (05 refuses to stage it)")
                continue
            if is_committed_row(mrow):
                # demo-import (Q12, narrow): a COMMITTED row skips the (campaign = this run's, demo-fit-id) lookup alone — the paper's row sits under
                # its own fit checkpoint and campaign — and is held to the digest's own live row (status, derivation, kind/label, the sidecar's parents)
                have_c = ((committed or {}).get(key) or {}).get(sid)
                out.setdefault(key, {})[sid] = committed_derived_pin(mrow, registry_root=registry_root, key=key, sid=sid, cell=cell, kind=str(kind), label=label, have=have_c)
                print(f"pins: {ident}: committed {msha[:12]} ({mrow.get('source_path')}) — registry-backed by its own row; {'kept' if have_c else 'ADDED'}")
                continue
            found = derived_registry_rows(registry_root, fit_model_id=fit_model_id, object_kind=str(kind), paper_label=label, campaign=campaign)
            found = _narrow_by_declared_parents(found, o, key, objs, out, load_pins())   # S2RL: the declared parents' pinned ids pick the lane's row
            if len(found) > 1:
                raise SystemExit(f"{ident}: {len(found)} reference/approved registry rows — "
                                 + ", ".join(f"{x['object_id'][:12]} ({x['status']})" for x in found)
                                 + " — one object must be THE reference before it can be pinned (declare `parents:` on the object to tell lanes apart); refusing to write the pins")
            if not found:
                if msha:
                    raise SystemExit(f"{path}: {ident}: staged (manifest digest {msha[:12]}) but the registry holds no reference/approved row — the "
                                     f"staging manifest is never a derived pin's source (D1319): 04 registers the object first; refusing to write the pins")
                have_c = ((committed or {}).get(key) or {}).get(sid)
                if have_c and is_committed_pin(have_c):
                    # demo-import: a committed pin entry the manifest does not carry is backed by the digest's OWN row (the paper's identity), checked, then kept
                    # verbatim — into the block NOW, so a committed dose copy staged beside it finds its parent's pin
                    committed_pin_still_backed(have_c, registry_root=registry_root, ident=ident)
                    out.setdefault(key, {})[sid] = dict(have_c)
                    print(f"pins: {ident}: committed pin {str(have_c.get('object_sha256'))[:12]} ({have_c.get('source')}) not in this staging manifest — registry-backed by its own row; kept")
                    continue
                if have_c:
                    # A committed registry-owned pin whose row is demoted / withdrawn / superseded is NOT carried
                    # back by the keep-verbatim branch as if the registry still stood behind it — the same refusal the registry-only writer applies
                    raise SystemExit(f"{ident}: the committed pin {str(have_c.get('object_sha256'))[:12]} has NO reference/approved registry row any more — the pin IS the "
                                     f"registered object, and a pin the registry no longer stands behind cannot be written as if it did; restore the row or retire the pin; "
                                     f"refusing to write the pins")
                print(f"pins: {ident}: no reference/approved registry row yet — no pin; 04 builds + registers it, then this generator writes the pin and 05 stages by it")
                continue
            oid = str(found[0]["object_id"])
            if msha and msha != oid:
                raise SystemExit(f"{path}: {ident}: the staging manifest's digest {msha[:12]} is not the registry id {oid[:12]} ({found[0]['status']}) — "
                                 f"a stale staged copy; refusing to write the pins")
            out.setdefault(key, {})[sid] = {
                "object_sha256": oid,
                "source": (f"registry:{campaign}/{fit_model_id}/{kind}/{label}:{found[0]['status']}"
                           + (f" (staged from {mrow.get('volume_path') or mrow.get('source_path')})" if mrow else " (not staged yet)")),
                "verified_cluster_a": bool((mrow or {}).get("cluster_a_verified")),
                "verified_cluster_b": bool((mrow or {}).get("cluster_b_verified")),
                "sidecar_object_sha256_recorded": ((mrow or {}).get("recorded_hash_check") or {}).get("status"),
            }
    # -- the front door's DOSE copies come from the registry too: the strength lane of the PARENT's identity ----------
    if dose_rows and registry_root is None:
        r0 = dose_rows[0]
        raise SystemExit(f"{path}: {r0.get('target')}/{r0.get('staged_id')}: a front-door dose copy is pinned by its REGISTRY row (the parent's identity at "
                         f"base_scale = the view's dose), never from the staging manifest ({str(r0.get('object_sha256'))[:12]}) — this read was given no "
                         f"registry; run the generator with --registry-root / SISTEER_REGISTRY_ROOT")
    if registry_root is not None:
        out = dose_pins_from_registry(out, registry_root=registry_root, roster=roster, models_table=models_table, manifest_rows=rows, manifest_path=path,
                                      committed=committed, targets=targets, campaign=campaign)   # S2RLFIX3: a committed dose pin whose row is gone refuses here, never resurrected below
    if committed:
        absent = sorted((k, sid) for k, ents in committed.items() for sid in (ents or {}) if sid not in (out.get(k) or {}))
        for k, sid in absent:
            if allow_drop:
                print(f"pins: {k}/{sid}: committed entry PRUNED (not in this staging manifest) — allowed by: {allow_drop}")
            else:
                out.setdefault(k, {})[sid] = dict(committed[k][sid])
                print(f"pins: {k}/{sid}: committed entry kept verbatim (not in this staging manifest: another bank's pin, or pinned ahead of its staging — S2RLFIX2, D1980 (1))")
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _declared_parent_pins(o: dict, key: str, objs: dict, *pin_sets: dict) -> "set | None":
    """S2RL: the pinned ids of a derived object's DECLARED parents (`parents:` on its models-table entry — the parents' own
    staged ids `<lane>__<cell_id>` looked up in the given pin sets, the first that carries each), or None when the entry declares no parents.
    A declared parent without a pin refuses by name (pin the parents before the derived object)."""
    parents = o.get("parents")
    if not parents:
        return None
    want = set()
    for pc in parents:
        po = objs.get(pc)
        if not isinstance(po, dict) or not po.get("lane") or not po.get("cell_id"):
            raise SystemExit(f"{key}: derived object declares parent {pc!r}, which is not an object of the models table")
        psid = f"{po['lane']}__{po['cell_id']}"
        pid = next((str(((ps.get(key) or {}).get(psid) or {}).get("object_sha256") or "") for ps in pin_sets if ((ps.get(key) or {}).get(psid) or {}).get("object_sha256")), "")
        if not pid:
            raise SystemExit(f"{key}: derived object's declared parent {pc} ({psid}) carries no pin — pin the parents before the derived object")
        want.add(pid)
    return want


def _narrow_by_declared_parents(found: list, o: dict, key: str, objs: dict, *pin_sets: dict) -> list:
    """Two or more reference / approved rows for ONE derived identity (the zero-init and the LeCun-refit compositions of the same cell share
    (campaign, fit, kind, label)): the row whose `parent_object_ids_json` is exactly the pinned ids of the object's declared parents is
    THE row, however many rows were found — a lone row recording other parents is another lane's, never adopted; several matching rows go
    back for the caller's refusal by name; none matching is NO row for this object (a note says which rows were seen); no declared parents
    leaves `found` as it is."""
    want = _declared_parent_pins(o, key, objs, *pin_sets)
    if want is None or not found:
        return found
    def _parents(r):
        return {str(x) for x in (_json_col(r, "parent_object_ids_json", []) or [])}
    hits = [r for r in found if _parents(r) == want]
    if not hits:
        print(f"pins: {key}/{o['lane']}__{o['cell_id']}: {len(found)} reference/approved row(s) for the identity record other parents than the declared "
              f"parents' pinned ids {sorted(x[:12] for x in want)} — " + ", ".join(f"{str(r['object_id'])[:12]} ({r['status']}; parents {sorted(x[:12] for x in _parents(r))})" for r in found)
              + " — none is this object's (a row built from other parents is another lane's)")
    return hits


def derived_pins_from_registry(pins: dict, *, registry_root: "Path | str", models_table: "Path | str | None" = None,
                               campaign: "str | None" = None) -> dict:
    """The committed pins with ONLY the front-door DERIVED entries rewritten from the registry object rows: for every derived cell of
    a front-door target (the models table's `objects` with lane `derived` and an `object_kind`), exactly one reference / approved row for
    (campaign, fit checkpoint, object_kind, paper label) writes the entry — `object_sha256` = the row's id, `source` = registry:…:<status>
    (with the staged copy's source when the committed entry already carries the same id, else "not staged yet"), the committed entry's
    verification facts kept when its id is the row's, false otherwise; a committed entry whose id differs from the row, or two rows, refuses
    by name; zero rows leaves the file as it is. Every other entry is byte-identical: the paper pins were written from the Modal staging
    manifest and are NEVER regenerated from the clusters' bank manifest (a dry --pins-from over the 248-record manifest of 2026-10-02
    rewrote 1,754 / 1,780 lines of this file — every entry that manifest did not hold). `campaign`: the registry campaign the rows are read
    under — None resolves to the site's, as 04 resolves it (registry_campaign_default), so the pins are read where 04 registered."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    out = {k: dict(v) for k, v in pins.items()}
    fd = front_door_models(models_table) if models_table is not None else FRONT_DOOR_MODELS
    for key, m in sorted(fd.items()):
        fit_model_id = str(((m.get("fit") or {}).get("checkpoint") or {}).get("model_id") or "")
        objs = m.get("objects") or {}
        for cell, o in sorted(objs.items()):
            if not isinstance(o, dict) or o.get("source") != "derived_bank" or not o.get("object_kind"):   # S2RL: every derived-bank lane
                continue
            sid = f"{o['lane']}__{o['cell_id']}"; kind = str(o["object_kind"]); label = str(o.get("paper_label") or cell)   # S2RL: the registry's paper label
            ident = f"{key}/{sid} ({cell}; campaign {campaign}, fit {fit_model_id}, {kind})"
            found = derived_registry_rows(registry_root, fit_model_id=fit_model_id, object_kind=kind, paper_label=label, campaign=campaign)
            found = _narrow_by_declared_parents(found, o, key, objs, out)             # S2RL: the declared parents' pinned ids pick the lane's row
            if len(found) > 1:
                raise SystemExit(f"{ident}: {len(found)} reference/approved registry rows — " + ", ".join(f"{x['object_id'][:12]} ({x['status']})" for x in found)
                                 + " — one object must be THE reference before it can be pinned (declare `parents:` on the object to tell lanes apart); refusing to write the pins")
            have = (out.get(key) or {}).get(sid)
            if not found:
                if have and is_committed_pin(have):
                    # gap-refusal-pin (2026-10-12): a COMMITTED pin (import_objects.py pins — the paper's object under the worked example's key) is backed by the
                    # digest's OWN row, the identity a committed object is filed under (never the lookup above, keyed on the key's fit checkpoint, which finds no
                    # paper row), and kept verbatim — the rule pins_from_staging_manifest already applies to the same entry; before this the README's
                    # `--derived-pins-from-registry` line refused here on every tree carrying the worked example's block, so no fit-your-own pin was ever written
                    committed_pin_still_backed(have, registry_root=registry_root, ident=ident)
                    print(f"pins: {ident}: committed pin {str(have.get('object_sha256'))[:12]} ({have.get('source')}) — registry-backed by its own row; kept")
                    continue
                if have:
                    # a committed pin the registry no longer stands behind (its row demoted, withdrawn or superseded — zero
                    # reference/approved rows) is NOT left standing as if it did, and the configs are not written over it; the no-row / no-pin bootstrap stays
                    raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} has NO reference/approved registry row any more — the pin IS the "
                                     f"registered object, and a pin the registry no longer stands behind cannot be written as if it did; restore the row or retire the pin; "
                                     f"refusing to write the pins")
                print(f"pins: {ident}: no reference/approved registry row and no committed entry — not written (04 registers the object first)")
                continue
            oid = str(found[0]["object_id"])
            if have and str(have.get("object_sha256")) != oid:
                if is_committed_pin(have):
                    # gap-refusal-pin: a fit under the worked example's key beside its committed (paper) pin — two objects for one cell; the committed identity
                    # never changes by a fit (the demo key serves the committed sequence's objects), so the collision is named, never resolved silently
                    raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} ({have.get('source')}) is the paper's committed object under this key, "
                                     f"and the registry's row {oid[:12]} ({found[0]['status']}) is another object — a {kind} fitted under this key; a committed pin's identity "
                                     "never changes by a fit: fit your own objects under a key of your own (reproduce/README.md, Adding a model: copy the row WITHOUT "
                                     "committed_objects_of — this key's row declares it, so 04 refuses to fit under it and 05 --from-committed imports the committed "
                                     "objects), or run the committed sequence under this key; refusing to write the pins")
                raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} is not the registry id {oid[:12]} ({found[0]['status']}) — "
                                 f"the pin IS the registered object; refusing to write the pins")
            staged_from = (have or {}).get("source") if have else None
            if staged_from and staged_from.startswith("registry:") and "(staged from " in staged_from:
                staged_from = staged_from.split("(staged from ", 1)[1].rstrip(")")
            elif staged_from and staged_from.startswith("registry:"):
                staged_from = None
            out.setdefault(key, {})[sid] = {
                "object_sha256": oid,
                "source": f"registry:{campaign}/{fit_model_id}/{kind}/{label}:{found[0]['status']}" + (f" (staged from {staged_from})" if staged_from else " (not staged yet)"),
                "verified_cluster_a": bool((have or {}).get("verified_cluster_a")),
                "verified_cluster_b": bool((have or {}).get("verified_cluster_b")),
                "sidecar_object_sha256_recorded": (have or {}).get("sidecar_object_sha256_recorded"),
            }
            print(f"pins: {ident}: {oid[:12]} ({found[0]['status']}) {'kept' if have else 'ADDED'}")
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


#: A BASE-lane object the models table declares with an `object_kind` (the s1 LoRA parents) is pinned from
#: the registry like a derived one — by its TOP-LEVEL fitted row. The rows for (campaign, fit checkpoint, object_kind, paper label) include the fit's
#: epoch children (04 registers them under the same kind + label: artifact under epoch_ckpts/, extra.parent_object_id = the top-level id); the top-level
#: row is the one carrying NEITHER marker, the two markers must agree on every row, and exactly one top-level row writes the pin. A derived object's
#: registered parents must then be exactly the pinned top-level ids of its parent kind (lora_subtraction <- lora), or the generator refuses by name.
BASE_LANE = "base"
EPOCH_CHILD_PATH_MARK = "/epoch_ckpts/"
DERIVED_PARENT_KINDS = {"lora_subtraction": "lora", "composed": "relu_steer"}

#: the front door's REFUSAL DIRECTION (2026-10-12 — reproduce/README.md's Known gap "no stage writes the pin of a fresh refusal direction"): a
#: models-table object of `source: arditi_bank` (lane `arditi`, kind `derived`: never staged by directory discovery) is pinned like the derived
#: objects, from the registry object row its fit emits — scripts/arditi/fit_refusal_direction.py -> sisteer.registry.hooks.emit_object_from_sidecar:
#: kind vector_all, role baseline_published -> paper label arditi_allpos, fit_model_id the SERVED checkpoint's model_id (models.yaml
#: serve.checkpoint.model_id — the sbatch's <model_key>, the fitter's --model-key AND the publication directory <refusal root>/<served model id>/
#: <staged id>: the object's identity names its directory, so two table keys serving one checkpoint share the publication, the row and the pin, as
#: two keys sharing a fit checkpoint share the fit bank's cells; the table key need not equal it, the paper's qwen35_4b serves qwen35_4b_it), the
#: campaign 04 states (its CAMPAIGN: the site's SISTEER_REGISTRY_CAMPAIGN, else REGISTRY_CAMPAIGN, else sdfbase_full22 — read here by the SAME
#: resolution, registry_campaign_default, or named by --campaign; reading sdfbase_full22 alone would leave a site naming another campaign with its
#: fit registered and no pin written), object_id = the sha256 of the object.pt the fit wrote. The identity carries no coefficient column, so a
#: ladder's rungs share it (the paper's qwen35_4b_it rows: c = 0.1 … 1.5): the declared cell `ar_c<c>__arditi_allpos` picks the row by the row's
#: OWN statement of its rung — extra.staged_id (the fitter's spelling), else base_scale (the committed import, the adapter-filed ladder), else
#: extra.coefficient / extra.c — and a row stating none is NAMED, never adopted. 04's done-ness reads the rung through the SAME function —
#: `arditi_registered_ids`, the `--arditi-registered-ids` CLI mode: the identity's rows selected broadly in SQL (no rung predicate) and classified
#: in Python by `arditi_row_rung`, the one reading of a row's rung — never through an SQL twin of it (the twin matched only the two canonical
#: staged-id spellings while the Python parses any decimal in the grammar, so a row spelled `ar_cm1p0__arditi_allpos` was coefficient -1 here and
#: absent to 04 at c=-1, which then submitted a second fit of the registered identity and rung, and this writer refused the two reference rows).
#: The constants restate the fitter's (tests/test_refusal_direction_pin.py holds them equal to fit_refusal_direction's and to hooks.paper_label_for,
#: and 04's reading equal to the writer's over every spelling the grammar admits). The CELL is the one identity: 04 read its rung from
#: SISTEER_REFUSAL_COEFFICIENT alone while this writer picked the pin's rung from the cell, so a knob the cell did not restate fitted and registered
#: a row no pin was written from, and a numerically equal cell spelled `ar_c0p30` was pinned at 0.3 while 04 published under `ar_c0p3` and 05 read
#: the literal `ar_c0p30` directory. Now 04 derives its coefficient and staged id from the cell through `arditi_declared_cell` (the `--arditi-cell`
#: CLI mode; the knob may only restate it), and `arditi_cell_coefficient` — the function this writer picks the pin's rung with, and 05's cells_json
#: judges the cell with — accepts a cell only in the FITTER'S spelling (fit_refusal_direction.staged_id, imported by path, never restated): the
#: cell is the directory 04 publishes, the directory 05 stages from and the pin's key, so it has one spelling; a ROW's spelling stays lenient
#: (arditi_row_rung).
ARDITI_LANE = "arditi"
ARDITI_SOURCE = "arditi_bank"
ARDITI_OBJECT_KIND = "vector_all"
ARDITI_OBJECT_ROLE = "baseline_published"
ARDITI_PAPER_LABEL = "arditi_allpos"
ARDITI_TABLE_CELL = "refusal_direction"   # the models-table view cell 04's unit of that name fits: objects.refusal_direction (an arditi_bank object)
ARDITI_FITTER = REPO / "scripts" / "arditi" / "fit_refusal_direction.py"   # the ONE owner of the staged id's spelling (staged_id); stdlib at import
_ARDITI_CELL_RE = re.compile(r"^ar_c(?P<tok>[m-]?\d+(?:p\d+)?)__(?P<label>.+)$")


def _arditi_tok_value(tok: str) -> float:
    """`0p3` -> 0.3, `m1p0` / `-1p0` -> -1.0 (fit_refusal_direction.staged_id: `ar_c{c:g}` with `.` as `p`)."""
    t = str(tok).replace("p", ".")
    if t[:1] in ("m", "-"):
        t = "-" + t[1:]
    return float(t)


_ARDITI_FITTER_MOD = None


def arditi_fitter_staged_id(coefficient: float) -> str:
    """THE spelling of a refusal-direction cell at a rung — scripts/arditi/fit_refusal_direction.staged_id, imported by path (never restated
    here: the fitter names the directory it publishes, <out_root>/<served id>/<staged id>, and the staged id its row states), the string 04
    publishes under, 05 reads from, and the pin is keyed by (`arditi__<staged id>`)."""
    global _ARDITI_FITTER_MOD
    if _ARDITI_FITTER_MOD is None:
        if not ARDITI_FITTER.is_file():
            raise SystemExit(f"the refusal-direction fitter {ARDITI_FITTER} is not on this tree — a refusal-direction cell's spelling is the fitter's "
                             "(staged_id), refusing to spell it another way")
        import importlib.util
        spec = importlib.util.spec_from_file_location("_full22_fit_refusal_direction", ARDITI_FITTER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ARDITI_FITTER_MOD = mod
    return str(_ARDITI_FITTER_MOD.staged_id(float(coefficient)))


def arditi_cell_coefficient(cell_id: str) -> float:
    """The coefficient a refusal-direction CELL id spells (`ar_c0p3__arditi_allpos` -> 0.3: the fitter's staged-id grammar
    `ar_c<c with . as p>__<paper label>`); a cell id outside the grammar, or of another label, refuses by name — and so does a cell that is
    not the FITTER'S OWN spelling of its coefficient (`ar_c0p30`, the ladder's `ar_cm1p0`): a cell is a directory name and a pin key as well
    as a rung — 04 publishes at the fitter's spelling (<refusal root>/<served id>/staged_id(c)), 05 reads the literal cell directory, and this
    writer keys the pin `arditi__<cell_id>` — so a cell spelled any other way is pinned here but never staged. A
    ROW's spelling stays lenient (arditi_row_rung: a hand-filed row is read at its rung); the cell is the table's declaration and has one
    spelling. Returns the coefficient of a canonical cell."""
    m = _ARDITI_CELL_RE.match(str(cell_id))
    if not m or m.group("label") != ARDITI_PAPER_LABEL:
        raise SystemExit(f"refusal direction cell {cell_id!r} does not spell `ar_c<coefficient>__{ARDITI_PAPER_LABEL}` (fit_refusal_direction.staged_id's "
                         "grammar) — refusing to form its registry identity")
    c = _arditi_tok_value(m.group("tok"))
    canonical = arditi_fitter_staged_id(c)
    if str(cell_id) != canonical:
        raise SystemExit(f"refusal direction cell {cell_id!r} spells coefficient {c:g} but is not the fitter's spelling of it, {canonical!r} "
                         "(scripts/arditi/fit_refusal_direction.staged_id: the directory 04 publishes under <refusal root>/<served id>/, the directory 05 "
                         f"stages from, the staged id the registry row states and the pin's key) — a cell spelled otherwise is pinned but never staged; spell the cell {canonical!r}")
    return c


def arditi_declared_cell(models_table: "Path | str | None", key: str) -> "tuple[str, float]":
    """(cell_id, coefficient) of ONE model's refusal direction as the models table DECLARES it — objects.refusal_direction (ARDITI_TABLE_CELL), an
    `arditi_bank` object of lane `arditi`, its cell the fitter's spelling of the coefficient (arditi_cell_coefficient) — the one identity 04 fits
    and publishes, this writer pins and 05 stages (04 read its rung from SISTEER_REFUSAL_COEFFICIENT alone while the pin was written from the
    cell, so a knob set to 0.5 over the copied example's `ar_c0p3` cell fitted and registered a 0.5 the pin writer never pinned and 05 never
    staged). A key the table lacks, a row without the cell, or a cell that is not an arditi_bank object refuses by name."""
    p = Path(models_table) if models_table is not None else MODELS_TABLE
    if not p.is_file():
        raise SystemExit(f"models table {p} is not on disk — refusing to form the refusal direction's identity without its declaration")
    models = _models_reader().load_models(str(p))
    if key not in models or not isinstance(models[key], dict):
        raise SystemExit(f"{p.name}: no model {key!r} (keys: {sorted(models)}) — refusing to form the refusal direction's identity")
    objs = models[key].get("objects") or {}
    o = objs.get(ARDITI_TABLE_CELL)
    if not isinstance(o, dict) or not o.get("cell_id"):
        raise SystemExit(f"{p.name}: {key}: objects.{ARDITI_TABLE_CELL}.cell_id is not declared (objects: {sorted(k for k, v in objs.items() if isinstance(v, dict))}) — "
                         f"04's {ARDITI_TABLE_CELL} unit fits the cell the table declares (`ar_c<c>__{ARDITI_PAPER_LABEL}`, the fitter's spelling); refusing to form its identity")
    if o.get("source") != ARDITI_SOURCE or o.get("lane") != ARDITI_LANE:
        raise SystemExit(f"{p.name}: {key}: objects.{ARDITI_TABLE_CELL} ({o.get('cell_id')}) declares source {o.get('source')!r} / lane {o.get('lane')!r}, not an "
                         f"{ARDITI_SOURCE} object of lane {ARDITI_LANE} — the refusal direction 04 fits is published outside the fit bank and pinned from its registry row; refusing to form its identity")
    cell_id = str(o["cell_id"])
    return cell_id, arditi_cell_coefficient(cell_id)


def arditi_registry_rows(root: "Path | str", *, fit_model_id: str, campaign: str = REGISTRY_CAMPAIGN) -> list:
    """EVERY object row of the refusal direction's identity for ONE served checkpoint — (campaign, fit_model_id, vector_all, arditi_allpos,
    baseline_published), ANY status, so the caller can name a candidate / withdrawn fit rather than read it as "no fit" — with the columns the
    rung is read from (base_scale, extra_json). Read-only; a registry that is not on disk refuses by name (never read as empty)."""
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import db as DB
    path = DB.db_path(str(root))
    if not Path(path).is_file():
        raise SystemExit(f"registry {path} is not on disk — the refusal direction's pin is written from the REGISTRY object row its fit emits (D1319); "
                         "point SISTEER_REGISTRY_ROOT / --registry-root at the grading host's registry")
    conn = DB.connect(path, readonly=True)
    try:
        rows = conn.execute("SELECT object_id, status, object_kind, paper_label, fit_model_id, campaign, base_scale, derivation, extra_json, artifact_locations_json "
                            "FROM object WHERE campaign=? AND fit_model_id=? AND object_kind=? AND paper_label=? AND object_role=? ORDER BY status, object_id",
                            (campaign, fit_model_id, ARDITI_OBJECT_KIND, ARDITI_PAPER_LABEL, ARDITI_OBJECT_ROLE)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def arditi_row_rung(row: dict) -> "tuple[str | None, float | None]":
    """(where, coefficient) as the row ITSELF states its rung: extra.staged_id (the in-repo fitter's spelling) first, else base_scale (the
    committed import and the adapter-filed ladder rows), else extra.coefficient / extra.c; (None, None) when it states nothing."""
    extra = _json_col(row, "extra_json", {})
    extra = extra if isinstance(extra, dict) else {}
    sid = extra.get("staged_id")
    if isinstance(sid, str):
        m = _ARDITI_CELL_RE.match(sid)
        if m:
            return "extra.staged_id", _arditi_tok_value(m.group("tok"))
    if row.get("base_scale") is not None:
        try:
            return "base_scale", float(row["base_scale"])
        except (TypeError, ValueError):
            pass
    for k in ("coefficient", "c"):
        v = extra.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"extra.{k}", float(v)
    return None, None


ARDITI_PINNABLE_STATUSES = ("reference", "approved")   # the statuses a row writes the pin from — and 04 reads as registered (one rule)


def arditi_rows_at_rung(rows: list, coefficient: float) -> "tuple[list, list]":
    """ONE partition of the identity's rows by the rung each states (arditi_row_rung) against `coefficient` (within 1e-9), shared by the pins
    writer (arditi_pins_from_registry) and 04's done-ness read (arditi_registered_ids) so the two cannot disagree on a spelling: `mine` = the
    reference / approved rows at this rung, the rows a pin is written from; `seen` = every other row NAMED, one line each — `<id> (<status>;
    <where> <c>)`, or `no rung stated` — another rung's, a candidate's / withdrawn one's, one stating nothing."""
    c = float(coefficient)
    mine, seen = [], []
    for r in rows:
        where, rc = arditi_row_rung(r)
        ok_rung = rc is not None and abs(rc - c) < 1e-9
        if ok_rung and str(r.get("status")) in ARDITI_PINNABLE_STATUSES:
            mine.append(r)
        else:
            seen.append(f"{str(r['object_id'])[:12]} ({r.get('status')}; " + (f"{where} {rc:g}" if rc is not None else "no rung stated") + ")")
    return mine, seen


def arditi_registered_ids(root: "Path | str", *, fit_model_id: str, coefficient: float, campaign: str = REGISTRY_CAMPAIGN) -> "tuple[list, list]":
    """What 04_fit_objects.sh reads as THE registered refusal direction of the SERVED checkpoint at the rung (`--arditi-registered-ids`):
    the identity's rows selected broadly in SQL — campaign, fit_model_id = the served id, vector_all / arditi_allpos /
    baseline_published, ANY status, NO rung predicate (arditi_registry_rows) — then classified in Python by arditi_row_rung through
    arditi_rows_at_rung, the pins writer's own partition. -> (the distinct reference / approved object ids at the rung, sorted; the other
    rows named). Zero ids: not registered at this rung. TWO or more: refused by name — the state the pins writer refuses (one refusal
    direction must be THE reference), so 04 must never read it as done or as unregistered. A registry not on disk refuses by name
    (arditi_registry_rows), never reads as empty."""
    rows = arditi_registry_rows(root, fit_model_id=fit_model_id, campaign=campaign)
    mine, seen = arditi_rows_at_rung(rows, coefficient)
    ids = sorted({str(r["object_id"]) for r in mine})
    if len(ids) > 1:
        raise SystemExit(f"refusal direction of {fit_model_id} at coefficient {float(coefficient):g} (campaign {campaign}): {len(ids)} reference/approved registry rows "
                         "at this coefficient — " + ", ".join(f"{str(x['object_id'])[:12]} ({x['status']}; {arditi_row_rung(x)[0]})" for x in mine)
                         + " — one refusal direction must be THE reference before it can be pinned (the pins writer refuses the same rows); 04 can "
                         "neither read this as done nor fit a third; withdraw the rows that are not it, then re-run")
    return ids, seen


def arditi_pins_from_registry(pins: dict, *, registry_root: "Path | str", models_table: "Path | str | None" = None,
                              manifest_rows: "list | None" = None, committed: "dict | None" = None, targets: "set | None" = None,
                              campaign: "str | None" = None) -> dict:
    """The pins with ONLY the front-door REFUSAL-DIRECTION entries (the models table's `objects` with `source: arditi_bank`) written from the
    registry object row the fit emits (2026-10-12): `object_sha256` = the row's id (the sha256 of the object.pt the fit wrote —
    never a digest this generator computes or a staging manifest records), `source` = registry:<campaign>/<served model id>/vector_all/
    arditi_allpos coefficient <c>:<status> (the staged copy's source kept when the entry already carries the row's id, else "not staged yet"), the
    verification facts the derived writer's shape. The rows of the identity are partitioned by the rung each states (arditi_row_rung) against
    the cell's `ar_c<c>`; among the cell's rows exactly one reference / approved row writes the pin. ZERO such rows: a committed pin
    (import_objects.py pins — the paper's refusal direction under the key) is held to the digest's OWN row and kept verbatim; any other
    committed entry refuses (a pin the registry no longer stands behind); else nothing is written and the note names every row seen
    (another rung's, a candidate's, one stating no rung) and the order — 04 --object refusal_direction fits + registers, this generator writes
    the pin, 05 stages by it (05 refuses the cell until then). TWO or more: refused by name. A committed pin whose id is not the row's refuses by
    name — for the worked example's committed (paper) pin, naming the fit under this key as another object (fit under a key of your own; a
    committed pin's identity never changes by a fit). A staging manifest row (`manifest_rows`, the --pins-from path: a row the stager staged
    from 04's publication) supplies the staging facts; its digest must be the row's id (a stale staged copy refuses) and a staged digest with
    no row refuses (the manifest is never the pin's source). A COMMITTED staging row (source_path committed:…, the committed sequence) is not
    this lane's write and is left to the manifest path's own committed branches. `targets`: the one-target write (pins_from_staging_manifest).
    `campaign`: the registry campaign the identity's rows are read under — None resolves to the site's, as 04 resolves it
    (registry_campaign_default), so this writer looks where 04 registered the fit (reading sdfbase_full22 alone while 04 fitted, registered
    and read done-ness under SISTEER_REGISTRY_CAMPAIGN left a site naming another campaign with no pin and 05 refusing the completed fit);
    the campaign is spelled in the pin's source and the `pins:` line. Every other entry is byte-identical."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    out = {k: dict(v) for k, v in pins.items()}
    fd = front_door_models(models_table) if models_table is not None else FRONT_DOOR_MODELS
    if targets is not None:
        fd = {k: m for k, m in fd.items() if k in targets}
    by_row = {(str(r.get("target")), str(r.get("staged_id"))): r for r in (manifest_rows or [])}
    for key, m in sorted(fd.items()):
        serve_id = str(((m.get("serve") or {}).get("checkpoint") or {}).get("model_id") or "")
        for cell, o in sorted((m.get("objects") or {}).items()):
            if not isinstance(o, dict) or o.get("source") != ARDITI_SOURCE:
                continue
            sid = f"{o['lane']}__{o['cell_id']}"
            label = str(o.get("paper_label") or ARDITI_PAPER_LABEL)
            if label != ARDITI_PAPER_LABEL:
                raise SystemExit(f"{key}/{sid} ({cell}): the models table declares paper_label {label!r} for an {ARDITI_SOURCE} object, but the fitter files the "
                                 f"refusal direction as {ARDITI_PAPER_LABEL} (hooks.paper_label_for({ARDITI_OBJECT_KIND}, {ARDITI_OBJECT_ROLE})); refusing to form its registry identity")
            c = arditi_cell_coefficient(str(o["cell_id"]))
            ident = f"{key}/{sid} ({cell}; campaign {campaign}, served {serve_id or 'UNDECLARED'}, {ARDITI_OBJECT_KIND}/{ARDITI_PAPER_LABEL} coefficient {c:g})"
            mrow = by_row.get((key, sid)); msha = str((mrow or {}).get("object_sha256") or "")
            if is_committed_row(mrow):
                continue   # the committed sequence's row (import_objects.py cells): the manifest path's committed branches own it, this lane never rewrites it
            have = (out.get(key) or {}).get(sid) or ((committed or {}).get(key) or {}).get(sid)
            if not serve_id:
                raise SystemExit(f"{ident}: the models table declares no serve.checkpoint.model_id — the refusal direction is an object of the SERVED checkpoint "
                                 "(its registry row's fit_model_id, the id 04's registry_done reads); refusing to form its registry identity")
            rows = arditi_registry_rows(registry_root, fit_model_id=serve_id, campaign=campaign)   # under the campaign 04 registered, never the default alone
            mine, seen = arditi_rows_at_rung(rows, c)   # the partition 04's done-ness reads with too (arditi_registered_ids): one rule, one reading of a rung
            seen_txt = (f" ({len(seen)} other row(s) of the identity, none this cell's reference/approved: " + ", ".join(seen) + ")") if seen else ""
            if len(mine) > 1:
                raise SystemExit(f"{ident}: {len(mine)} reference/approved registry rows at this coefficient — " + ", ".join(f"{str(x['object_id'])[:12]} ({x['status']})" for x in mine)
                                 + " — one refusal direction must be THE reference before it can be pinned; refusing to write the pins")
            if not mine:
                if msha:
                    raise SystemExit(f"{ident}: staged (manifest digest {msha[:12]}) but the registry holds no reference/approved row of this cell{seen_txt} — the staging "
                                     "manifest is never the refusal direction's pin's source (D1319): 04 --object refusal_direction registers it first; refusing to write the pins")
                if have and is_committed_pin(have):
                    # the worked example's committed pin (import_objects.py pins: the paper's refusal direction under this key) is backed by the digest's OWN row —
                    # the identity a committed object is filed under (never the served-key lookup above, which finds no paper row) — then kept verbatim
                    committed_pin_still_backed(have, registry_root=registry_root, ident=ident)
                    out.setdefault(key, {})[sid] = dict(have)
                    print(f"pins: {ident}: committed pin {str(have.get('object_sha256'))[:12]} ({have.get('source')}) — registry-backed by its own row; kept")
                    continue
                if have:
                    raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} has NO reference/approved registry row any more{seen_txt} — the pin IS "
                                     "the registered object, and a pin the registry no longer stands behind cannot be written as if it did; restore the row or retire the pin; "
                                     "refusing to write the pins")
                print(f"pins: {ident}: no reference/approved registry row{seen_txt} — no pin; 04_fit_objects.sh --object refusal_direction fits + registers it (the row the fitter "
                      "emits, object_id = its object.pt's sha256), then `make_full22_configs.py --derived-pins-from-registry` writes the pin and 05 stages by it (05 refuses "
                      "to stage the cell until then)")
                continue
            row = mine[0]; oid = str(row["object_id"])
            if have and str(have.get("object_sha256")) != oid:
                if is_committed_pin(have):
                    raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} ({have.get('source')}) is the paper's committed refusal direction under "
                                     f"this key, and the registry's row {oid[:12]} ({row['status']}) is another object — a refusal direction fitted under this key; a committed "
                                     "pin's identity never changes by a fit: fit your own objects under a key of your own (reproduce/README.md, Adding a model: copy the row "
                                     "WITHOUT committed_objects_of — this key's row declares it, so 04 refuses to fit under it and 05 --from-committed imports the committed "
                                     "objects), or run the committed sequence under this key; refusing to write the pins")
                raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} is not the registry id {oid[:12]} ({row['status']}) — "
                                 "the pin IS the registered object; refusing to write the pins")
            if msha and msha != oid:
                raise SystemExit(f"{ident}: the staging manifest's digest {msha[:12]} is not the registry id {oid[:12]} ({row['status']}) — a stale staged copy; refusing to write the pins")
            if mrow:
                staged_from = mrow.get("volume_path") or mrow.get("source_path")
                facts = {"verified_cluster_a": bool(mrow.get("cluster_a_verified")), "verified_cluster_b": bool(mrow.get("cluster_b_verified")),
                         "sidecar_object_sha256_recorded": ((mrow.get("recorded_hash_check") or {}).get("status"))}
            else:
                staged_from = (have or {}).get("source") if have else None
                if staged_from and staged_from.startswith("registry:") and "(staged from " in staged_from:
                    staged_from = staged_from.split("(staged from ", 1)[1].rstrip(")")
                elif staged_from and staged_from.startswith("registry:"):
                    staged_from = None
                facts = {"verified_cluster_a": bool((have or {}).get("verified_cluster_a")), "verified_cluster_b": bool((have or {}).get("verified_cluster_b")),
                         "sidecar_object_sha256_recorded": (have or {}).get("sidecar_object_sha256_recorded")}
            out.setdefault(key, {})[sid] = {
                "object_sha256": oid,
                "source": f"registry:{campaign}/{serve_id}/{ARDITI_OBJECT_KIND}/{ARDITI_PAPER_LABEL} coefficient {c:g}:{row['status']}"
                          + (f" (staged from {staged_from})" if staged_from else " (not staged yet)"),
                **facts,
            }
            print(f"pins: {ident}: {oid[:12]} ({row['status']}) {'kept' if have else 'ADDED'}" + (" — staging facts from the manifest row" if mrow else ""))
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def front_door_object_rows(root: "Path | str", *, fit_model_id: str, object_kind: str, paper_label: str, campaign: str = REGISTRY_CAMPAIGN) -> list:
    """Every reference / approved object row for ONE (campaign, fit checkpoint, object_kind, paper label) WITH the columns the top-level rule
    and the parent check read (artifact_locations_json, extra_json, parent_object_ids_json), read-only; a registry not on disk refuses by name."""
    sys.path.insert(0, str(REPO / "src"))
    from sisteer.registry import db as DB
    path = DB.db_path(str(root))
    if not Path(path).is_file():
        raise SystemExit(f"registry {path} is not on disk — a front-door pin is written from the REGISTRY object row (D1319 / D1425); "
                         "point SISTEER_REGISTRY_ROOT / --registry-root at the grading host's registry")
    conn = DB.connect(path, readonly=True)
    try:
        rows = conn.execute("SELECT object_id, status, object_kind, paper_label, fit_model_id, campaign, artifact_locations_json, extra_json, parent_object_ids_json "
                            "FROM object WHERE campaign=? AND fit_model_id=? AND object_kind=? AND paper_label=? AND status IN ('reference', 'approved') "
                            "ORDER BY status, object_id", (campaign, fit_model_id, object_kind, paper_label)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _json_col(row: dict, col: str, default):
    raw = row.get(col)
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        raise SystemExit(f"registry row {str(row.get('object_id'))[:12]}: {col} is not JSON ({str(raw)[:60]!r}) — refusing to read the row")


def top_level_rows(rows: list, *, ident: str) -> list:
    """The rows carrying NEITHER epoch-child marker (an artifact under epoch_ckpts/; extra.parent_object_id). A row carrying exactly one of the
    two refuses by name — the markers are the fit's own two statements of the same fact and must agree before either is trusted."""
    out = []
    for r in rows:
        paths = _json_col(r, "artifact_locations_json", []); extra = _json_col(r, "extra_json", {})
        by_path = any(EPOCH_CHILD_PATH_MARK in str(p) for p in (paths if isinstance(paths, list) else [paths]))
        by_parent = bool((extra or {}).get("parent_object_id")) if isinstance(extra, dict) else False
        if by_path != by_parent:
            raise SystemExit(f"{ident}: registry row {str(r['object_id'])[:12]} ({r.get('status')}) is an epoch child by {'its artifact path' if by_path else 'extra.parent_object_id'} "
                             f"but not by {'extra.parent_object_id' if by_path else 'its artifact path'} — the two markers disagree; refusing to pick a top-level row")
        if not by_path:
            out.append(r)
    return out


def base_pins_from_registry(pins: dict, *, registry_root: "Path | str", models_table: "Path | str | None" = None,
                            manifest_rows: "list | None" = None, committed: "dict | None" = None, targets: "set | None" = None,
                            campaign: "str | None" = None) -> dict:
    """The pins with ONLY the front-door BASE-lane entries whose object DECLARES an `object_kind` rewritten from the registry's TOP-LEVEL row:
    `object_sha256` = the row's id, `source` = registry:<campaign>/<fit>/<kind>/<label>:<status> (the staged copy's source kept when
    the committed entry already carries the row's id, else "not staged yet"), the verification facts kept when the id is the committed one, false
    otherwise — the derived writer's shape (test_sdfbase_full22_newmodels holds every pin to it). Zero rows leave the entry as it is; two top-level
    rows, a committed id that is not the row's, or disagreeing markers refuse by name. An undeclared base object (the ReLU parents, pinned from
    the Modal staging manifest) is untouched. Every other entry is byte-identical.
    The manifest path (`--pins-from`) shares this writer instead of writing a declared base pin as a
    plain stager row — `manifest_rows` supplies the staging facts (the staged-from path, the verification flags, the sidecar check) for the
    rows it carries, `committed` the pin on record when `pins` lacks the entry (the manifest path diverts these rows), so both writers spell the
    pin identically and a row whose digest is not the registry id refuses as a stale staged copy; a committed pin whose row is gone refuses
    before the manifest path's keep-verbatim branch could carry it back. `targets`: the one-target write (pins_from_staging_manifest) —
    only the named targets' entries are read and written. `campaign`: the registry campaign the rows are read under — None resolves to the
    site's, as 04 resolves it (registry_campaign_default), so the pins are read where 04 registered."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    out = {k: dict(v) for k, v in pins.items()}
    fd = front_door_models(models_table) if models_table is not None else FRONT_DOOR_MODELS
    if targets is not None:
        fd = {k: m for k, m in fd.items() if k in targets}
    by_row = {(str(r.get("target")), str(r.get("staged_id"))): r for r in (manifest_rows or [])}
    for key, m in sorted(fd.items()):
        fit_model_id = str(((m.get("fit") or {}).get("checkpoint") or {}).get("model_id") or "")
        for cell, o in sorted((m.get("objects") or {}).items()):
            if not isinstance(o, dict) or o.get("lane") != BASE_LANE or not o.get("object_kind"):
                continue
            sid = f"{BASE_LANE}__{o['cell_id']}"; kind = str(o["object_kind"])
            ident = f"{key}/{sid} ({cell}; campaign {campaign}, fit {fit_model_id}, {kind}, the top-level row)"
            rows = front_door_object_rows(registry_root, fit_model_id=fit_model_id, object_kind=kind, paper_label=cell, campaign=campaign)
            have = (out.get(key) or {}).get(sid) or ((committed or {}).get(key) or {}).get(sid)   # S2RLFIX4: the manifest path diverts these rows; the committed entry is the pin on record
            mrow = by_row.get((key, sid)); msha = str((mrow or {}).get("object_sha256") or "")
            if not rows:
                if have:
                    # a committed pin the registry no longer stands behind (its row demoted, withdrawn or superseded —
                    # zero reference/approved rows) is NOT left standing as if it did; the no-row / no-pin bootstrap (04 registers the fit first) stays
                    raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} has NO reference/approved registry row any more — the pin IS the "
                                     f"registered object, and a pin the registry no longer stands behind cannot be written as if it did; restore the row or retire the pin; "
                                     f"refusing to write the pins")
                print(f"pins: {ident}: no reference/approved registry row and no committed entry — not written (04 registers the fit first)")
                continue
            tops = top_level_rows(rows, ident=ident)
            if len(tops) != 1:
                raise SystemExit(f"{ident}: {len(tops)} top-level registry row(s) among {len(rows)} ({', '.join(str(x['object_id'])[:12] + ' (' + str(x['status']) + ')' for x in tops) or 'none'}) — "
                                 f"exactly one fitted object is the pin; refusing to write the pins")
            oid = str(tops[0]["object_id"])
            if have and str(have.get("object_sha256")) != oid:
                raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} is not the registry id {oid[:12]} ({tops[0]['status']}) — "
                                 f"the pin IS the registered object; refusing to write the pins")
            if msha and msha != oid:
                raise SystemExit(f"{ident}: the staging manifest's digest {msha[:12]} is not the registry id {oid[:12]} ({tops[0]['status']}) — "
                                 f"a stale staged copy; refusing to write the pins")
            if mrow:   # S2RLFIX4: the staging facts come from the manifest row when it carries this pin
                staged_from = mrow.get("volume_path") or mrow.get("source_path")
                facts = {"verified_cluster_a": bool(mrow.get("cluster_a_verified")), "verified_cluster_b": bool(mrow.get("cluster_b_verified")),
                         "sidecar_object_sha256_recorded": ((mrow.get("recorded_hash_check") or {}).get("status"))}
            else:
                staged_from = (have or {}).get("source") if have else None
                if staged_from and staged_from.startswith("registry:") and "(staged from " in staged_from:
                    staged_from = staged_from.split("(staged from ", 1)[1].rstrip(")")
                elif staged_from and staged_from.startswith("registry:"):
                    staged_from = None
                facts = {"verified_cluster_a": bool((have or {}).get("verified_cluster_a")), "verified_cluster_b": bool((have or {}).get("verified_cluster_b")),
                         "sidecar_object_sha256_recorded": (have or {}).get("sidecar_object_sha256_recorded")}
            out.setdefault(key, {})[sid] = {
                "object_sha256": oid,
                "source": f"registry:{campaign}/{fit_model_id}/{kind}/{cell}:{tops[0]['status']}" + (f" (staged from {staged_from})" if staged_from else " (not staged yet)"),
                **facts,
            }
            print(f"pins: {ident}: {oid[:12]} ({tops[0]['status']}; {len(rows) - 1} epoch child row(s) beside it) {'kept' if have else 'ADDED'}"
                  + (" — staging facts from the manifest row" if mrow else ""))
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def check_derived_parents(pins: dict, *, registry_root: "Path | str", models_table: "Path | str | None" = None,
                          campaign: "str | None" = None) -> dict:
    """For every front-door DERIVED object whose kind names a parent kind (DERIVED_PARENT_KINDS) and whose parents are DECLARED base-lane
    objects of that kind, the registry row's `parent_object_ids_json` must equal, as a set, the pinned top-level ids of those parents — a derived
    object built from another fit's parents (an epoch child, a superseded fit) would otherwise be pinned beside parents it was not made of.
    Refuses by name; returns {derived staged id: sorted parent ids} for what it checked. A derived object with no committed pin, no registry row,
    or whose parent kind no base object declares (the ReLU parents) is skipped and said so. `campaign`: the registry campaign the rows
    are read under — None resolves to the site's, as 04 resolves it (registry_campaign_default)."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    checked: dict = {}
    fd = front_door_models(models_table) if models_table is not None else FRONT_DOOR_MODELS
    for key, m in sorted(fd.items()):
        fit_model_id = str(((m.get("fit") or {}).get("checkpoint") or {}).get("model_id") or "")
        objs = m.get("objects") or {}
        for cell, o in sorted(objs.items()):
            if not isinstance(o, dict) or o.get("source") != "derived_bank" or str(o.get("object_kind") or "") not in DERIVED_PARENT_KINDS:
                continue
            kind = str(o["object_kind"]); pkind = DERIVED_PARENT_KINDS[kind]; sid = f"{o['lane']}__{o['cell_id']}"
            if o.get("parents"):   # S2RL: DECLARED parents, whatever their lane and whether or not they declare an object_kind (the ReLU parents do not)
                parents = {f"{objs[pc]['lane']}__{objs[pc]['cell_id']}": pc for pc in o["parents"] if isinstance(objs.get(pc), dict)}
            else:
                parents = {f"{BASE_LANE}__{po['cell_id']}": pc for pc, po in objs.items()
                           if isinstance(po, dict) and po.get("lane") == BASE_LANE and str(po.get("object_kind") or "") == pkind}
            ident = f"{key}/{sid} ({cell}; {kind} <- {pkind})"
            pin = (pins.get(key) or {}).get(sid)
            if not parents or not pin:
                print(f"pins: {ident}: parent check skipped ({'no base-lane object declares object_kind ' + pkind if not parents else 'no committed pin'})")
                continue
            rows = [r for r in front_door_object_rows(registry_root, fit_model_id=fit_model_id, object_kind=kind, paper_label=str(o.get("paper_label") or cell), campaign=campaign) if str(r["object_id"]) == str(pin.get("object_sha256"))]   # S2RL: the paper label
            if not rows:
                print(f"pins: {ident}: parent check skipped (the pinned id {str(pin.get('object_sha256'))[:12]} has no reference/approved registry row)")
                continue
            want = {str((pins.get(key) or {}).get(psid, {}).get("object_sha256") or "") for psid in parents}
            missing = sorted(psid for psid in parents if not (pins.get(key) or {}).get(psid, {}).get("object_sha256"))
            if missing:
                raise SystemExit(f"{ident}: its parents {missing} carry no pin — pin the parents (their top-level rows) before the derived object")
            got = {str(x) for x in (_json_col(rows[0], "parent_object_ids_json", []) or [])}
            if got != want:
                raise SystemExit(f"{ident}: the registry row {str(rows[0]['object_id'])[:12]} names parents {sorted(x[:12] for x in got)}, the pinned "
                                 f"{pkind} parents are {sorted(x[:12] for x in want)} ({', '.join(f'{pc}={psid}' for psid, pc in sorted(parents.items()))}) — "
                                 f"the derived object was not built from the pinned parents; refusing to write the pins")
            checked[sid] = sorted(got)
            print(f"pins: {ident}: parents {sorted(x[:12] for x in got)} are the pinned {pkind} top-level rows ({', '.join(sorted(parents.values()))})")
    return checked


def dose_view_cells(roster: dict, key: str, m: dict) -> list:
    """Every cell of every DOSE view of front-door target `key`, as (view, dose, cell id, staged id, paper label, object_kind) — refusing by name
    a cell whose staged id does not spell `d<dose token>__<cell id>` (the copy would be served under another dose's name) and a cell that is
    not a DECLARED derived object of the models table (lane derived + object_kind: the registry identity a dose copy inherits is its PARENT's)."""
    out = []
    objs = m.get("objects") or {}
    # S2RL: a cell id is shared by the zero-init object and its LeCun refit (lanes derived / lecun_derived); a dose copy `d<tok>__<cid>`
    # is scaled from the DERIVED-lane composition, so that object — not the last one declared under the cell id — is the parent a dose cell resolves to
    by_cell: dict = {}
    for label, o in objs.items():
        if isinstance(o, dict) and o.get("cell_id"):
            cid_ = str(o["cell_id"])
            if cid_ not in by_cell or (o.get("lane") == DERIVED_LANE and by_cell[cid_][1].get("lane") != DERIVED_LANE):
                by_cell[cid_] = (label, o)
    for view in sorted((roster["targets"].get(key) or {}).get("views") or {}):
        dose = view_dose(roster, key, view)
        if dose is None:
            continue
        # D0P75RELUL: a LeCun-lane dose view (`d<tok>_<family>l`, `object_lane: lecun`) serves `d<tok>l__<cid>`, scaled from the lecun_derived parent
        lane = view_object_lane(roster, key, view)
        mark = DOSE_LANES.get(lane, "") if lane else ""
        parent_lane = DOSE_PARENT_LANE[lane]
        for cid, sid in sorted(((roster["targets"][key]["views"][view].get("cells") or {}).items())):
            want = f"d{dose_token(dose)}{mark}__{cid}"
            if str(sid) != want:
                raise SystemExit(f"{key}/{view}: cell {cid} is staged as {sid!r}, which does not spell the view's dose {dose:g}"
                                 f"{' and lane ' + lane if lane else ''} ({want!r}) — a dose copy is "
                                 f"named by its dose in front of its parent's cell id; refusing to resolve the pins")
            label, o = (by_cell.get(str(cid), (None, None)) if parent_lane == DERIVED_LANE else
                        next(((lb, po) for lb, po in objs.items() if isinstance(po, dict) and str(po.get("cell_id")) == str(cid) and po.get("lane") == parent_lane), (None, None)))
            if o is None or o.get("lane") != parent_lane or not o.get("object_kind"):
                raise SystemExit(f"{key}/{view}: cell {cid} (staged {sid}) is not a DECLARED derived object of the models table (lane {parent_lane!r} with an "
                                 f"object_kind) — a dose copy's registry identity is its parent's, so the parent must be declared before its dose can be pinned")
            # the REGISTRY label — the object's declared paper_label (the LeCun lane's `difference_relu_lecun` is registered as
            # `difference_relu`), else its table name
            out.append((view, float(dose), str(cid), str(sid), str(o.get("paper_label") or label), str(o["object_kind"])))
    return out


def dose_pins_from_registry(pins: dict, *, registry_root: "Path | str", roster: "dict | None" = None, models_table: "Path | str | None" = None,
                            manifest_rows: "list | None" = None, manifest_path: "Path | None" = None, committed: "dict | None" = None,
                            targets: "set | None" = None, campaign: "str | None" = None) -> dict:
    """The pins with the front door's DOSE-COPY entries (`d<tok>__<cell>`, the roster's views with a
    `dose:` key) validated against — and, when absent, ADDED from — the registry object row in the PARENT's identity at the view's dose: for
    (campaign, fit checkpoint, the parent's object_kind, the parent's paper label, base_scale == dose) exactly one reference / approved row,
    with derivation `dose_scale` and `parent_object_ids_json` naming exactly the pinned parent id, whose own pin must be its registry row.
    A committed entry whose id is the row's is kept VERBATIM (its staged source and verification facts are the stager's); an absent entry is
    added as `registry:…` (with the staging manifest's facts when `manifest_rows` carry the copy, else "not staged yet"); a committed entry
    whose id differs from the row, two rows in the lane, a committed pin with no row left (demoted, withdrawn, superseded), a row in another
    lane only (the copy at another dose), a row whose parents are not the pinned parent, an unpinned or mis-pinned parent, and a manifest
    digest that is not the row's id all refuse by name. No row and no committed entry: nothing written, said so. Every other entry is
    byte-identical. Before this, `derived_pins_from_registry` walked the models table's derived objects only, so the strength lane the
    reader grew was never asked and a dose view's staged pin stood on its surgical write alone. `targets`: the one-target write
    (pins_from_staging_manifest) — only the named targets' dose views are read and written. `campaign`: the registry campaign the copies'
    and the parents' rows are read under — None resolves to the site's, as 04 resolves it (registry_campaign_default)."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    out = {k: dict(v) for k, v in pins.items()}
    roster = roster if roster is not None else load_roster()
    fd = front_door_models(models_table) if models_table is not None else FRONT_DOOR_MODELS
    if targets is not None:
        fd = {k: m for k, m in fd.items() if k in targets}
    by_row = {(str(r.get("target")), str(r.get("staged_id"))): r for r in (manifest_rows or [])}
    for key, m in sorted(fd.items()):
        fit_model_id = str(((m.get("fit") or {}).get("checkpoint") or {}).get("model_id") or "")
        for view, dose, cid, sid, label, kind in dose_view_cells(roster, key, m):
            parent_lane = DOSE_PARENT_LANE[view_object_lane(roster, key, view)]   # D0P75RELUL: the lane the copy was scaled from (derived / lecun_derived)
            parent_sid = f"{parent_lane}__{cid}"
            ident = f"{key}/{sid} ({view}: {label} x{dose:g} of {parent_sid}; campaign {campaign}, fit {fit_model_id}, {kind}, base_scale {dose:g})"
            have = (out.get(key) or {}).get(sid) or ((committed or {}).get(key) or {}).get(sid)   # S2RLFIX3: the manifest path diverts dose rows, so the committed entry is the pin on record
            mrow_c = by_row.get((key, sid))
            if is_committed_row(mrow_c):
                # demo-import (Q12, narrow): a COMMITTED dose copy skips BOTH `derived_registry_rows` lookups (the copy's and the parent's, keyed on the demo
                # fit id that can never match a paper row) and is held to the digest's own live row: dose_scale, base_scale == the view's dose,
                # reference|approved, parents == {the parent's pin in THIS block} == the sidecar's dose_scaled_from.parent_object_sha256
                out.setdefault(key, {})[sid] = committed_dose_pin(mrow_c, registry_root=registry_root, key=key, sid=sid, view=view, dose=dose, parent_sid=parent_sid,
                                                                 kind=kind, label=label, block=out.get(key) or {}, have=have, manifest_path=manifest_path)
                print(f"pins: {ident}: committed {str(mrow_c.get('object_sha256'))[:12]} ({mrow_c.get('source_path')}) — registry-backed by its own row; {'kept' if have else 'ADDED'}")
                continue
            found = derived_registry_rows(registry_root, fit_model_id=fit_model_id, object_kind=kind, paper_label=label, base_scale=dose, campaign=campaign)
            if not found and have and is_committed_pin(have):
                # demo-import: a committed copy's pin entry the manifest does not carry is backed by the digest's OWN row, then kept verbatim
                committed_pin_still_backed(have, registry_root=registry_root, ident=ident)
                out.setdefault(key, {})[sid] = dict(have)
                print(f"pins: {ident}: committed pin {str(have.get('object_sha256'))[:12]} ({have.get('source')}) not in this staging manifest — registry-backed by its own row; kept")
                continue
            if not found:
                if have:
                    raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} has NO reference/approved registry row any more at base_scale "
                                     f"{dose:g} — the pin IS the registered dose copy, and a pin the registry no longer stands behind cannot be written as if it did; "
                                     f"restore the row (or re-mint + register the copy) or retire the pin and the view; refusing to write the pins")
                print(f"pins: {ident}: no reference/approved registry row at base_scale {dose:g} and no committed entry — not written (the copy is minted + registered first)")
                continue
            # the lane's REGISTRY parent is resolved once a copy EXISTS — a dose cell with no registered copy and no committed entry stays the
            # soft skip it was on the tip (the pins-rewrite worlds carry no LeCun parent row) — and BEFORE the copies are narrowed:
            # the zero-init and the LeCun-init compositions' dose copies share every identity column at
            # base_scale 0.75 (the identity collision at the dose rung), so the copy whose declared parents are THIS lane's registry parent is the lane's;
            # same-parent duplicates still refuse below, and a missing parent row / pin refuses as before, ahead of the copies
            all_parents = derived_registry_rows(registry_root, fit_model_id=fit_model_id, object_kind=kind, paper_label=label, base_scale=None, campaign=campaign)   # every full-strength row of the identity: the lanes' compositions
            parents = list(all_parents)
            pobj = next((po for po in (m.get("objects") or {}).values() if isinstance(po, dict) and po.get("lane") == parent_lane and str(po.get("cell_id")) == cid), None)
            if pobj is not None:   # S2RL: the zero-init composition's row among the lanes' (the LeCun refit shares the identity), by its declared parents
                parents = _narrow_by_declared_parents(parents, pobj, key, m.get("objects") or {}, out, load_pins())
            if len(parents) != 1:
                raise SystemExit(f"{ident}: the parent {parent_sid} has {len(parents)} reference/approved registry rows (base_scale NULL)"
                                 + (" — " + ", ".join(f"{x['object_id'][:12]} ({x['status']})" for x in parents) if parents else "")
                                 + " — the parent is pinned from exactly one row before its dose copy can be; refusing to write the pins")
            parent_id = str(parents[0]["object_id"])
            # lane-aware (the identity collision at the dose rung): the lane's copies are the rows whose declared parents are ITS registry parent. A dose
            # whose rows are ALL other lanes' copies (parents within the other compositions' ids) has no row of this lane → the not-found path (the committed-pin
            # refusal by name, else the soft skip) — a lone other-lane row is never adopted. A row recording parents that
            # are no lane's composition keeps the tip's single-row refusal below ("names parents […], the pinned parent is …").
            mine = [r for r in found if {str(x) for x in (_json_col(r, "parent_object_ids_json", []) or [])} == {parent_id}]
            if found and not mine:
                lanes = {str(x["object_id"]) for x in all_parents} - {parent_id}
                foreign = [r for r in found if (lambda ps: bool(ps) and ps <= lanes)({str(x) for x in (_json_col(r, "parent_object_ids_json", []) or [])})]
                if len(foreign) == len(found):
                    others = ", ".join(f"{x['object_id'][:12]} ({x['status']}; parents {sorted(str(y)[:12] for y in (_json_col(x, 'parent_object_ids_json', []) or []))})" for x in found)
                    if have:
                        raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} has NO reference/approved registry row any more at base_scale "
                                         f"{dose:g} — the {len(found)} row(s) at this dose are other lanes' copies ({others}); the pin IS the registered dose copy; register the copy "
                                         f"(or retire the pin and the view); refusing to write the pins")
                    print(f"pins: {ident}: no reference/approved registry row of THIS lane at base_scale {dose:g} ({len(found)} other-lane row(s): {others}) and no committed entry — not written (the copy is minted + registered first)")
                    continue
            found = mine or found
            if len(found) > 1:
                raise SystemExit(f"{ident}: {len(found)} reference/approved registry rows in the lane — " + ", ".join(f"{x['object_id'][:12]} ({x['status']})" for x in found)
                                 + " — one dose copy must be THE approved one before it can be pinned; refusing to write the pins")
            row = found[0]; oid = str(row["object_id"])
            if str(row.get("derivation") or "") != "dose_scale":
                raise SystemExit(f"{ident}: the registry row {oid[:12]} ({row['status']}) at base_scale {dose:g} has derivation {str(row.get('derivation'))!r}, not "
                                 f"'dose_scale' — a scaled copy is registered as one; refusing to write the pins")
            parent_pin = str(((out.get(key) or {}).get(parent_sid) or {}).get("object_sha256") or "")
            if not parent_pin:
                raise SystemExit(f"{ident}: the parent {parent_sid} carries no pin — pin the parent (its registry row {parent_id[:12]}) before its dose copy")
            if parent_pin != parent_id:
                raise SystemExit(f"{ident}: the parent's committed pin {parent_pin[:12]} is not its registry row {parent_id[:12]} ({parents[0]['status']}) — "
                                 f"the dose copy would be pinned beside a parent the registry does not stand behind; run the derived rewrite first")
            got = {str(x) for x in (_json_col(row, "parent_object_ids_json", []) or [])}
            if got != {parent_id}:
                raise SystemExit(f"{ident}: the registry row {oid[:12]} names parents {sorted(x[:12] for x in got)}, the pinned parent is {parent_id[:12]} — "
                                 f"the copy was not scaled from the pinned parent; refusing to write the pins")
            mrow = by_row.get((key, sid)); msha = str((mrow or {}).get("object_sha256") or "")
            if msha and msha != oid:
                raise SystemExit(f"{manifest_path}: {ident}: the staging manifest's digest {msha[:12]} is not the registry id {oid[:12]} ({row['status']}) — "
                                 f"a stale staged copy; refusing to write the pins")
            if have and str(have.get("object_sha256")) != oid:
                raise SystemExit(f"{ident}: the committed pin {str(have.get('object_sha256'))[:12]} is not the registry id {oid[:12]} ({row['status']}) — "
                                 f"the pin IS the registered dose copy; refusing to write the pins")
            if have:
                out.setdefault(key, {})[sid] = dict(have)   # S2RLFIX3: in the manifest path the entry came from `committed` — carried explicitly, verbatim
                print(f"pins: {ident}: {oid[:12]} ({row['status']}) kept (parent {parent_id[:12]})")
                continue
            out.setdefault(key, {})[sid] = {
                "object_sha256": oid,
                "source": (f"registry:{campaign}/{fit_model_id}/{kind}/{label} base_scale {dose:g}:{row['status']} (dose-scaled copy of {parent_id[:12]}"
                           + (f"; staged from {mrow.get('volume_path') or mrow.get('source_path')})" if mrow else "; not staged yet)")),
                "verified_cluster_a": bool((mrow or {}).get("cluster_a_verified")),
                "verified_cluster_b": bool((mrow or {}).get("cluster_b_verified")),
                "sidecar_object_sha256_recorded": ((mrow or {}).get("recorded_hash_check") or {}).get("status"),
            }
            print(f"pins: {ident}: {oid[:12]} ({row['status']}) ADDED (parent {parent_id[:12]})")
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def committed_dose_pin(mrow: dict, *, registry_root: "Path | str", key: str, sid: str, view: str, dose: float, parent_sid: str, kind: str, label: str,
                       block: dict, have: "dict | None", manifest_path: "Path | None") -> dict:
    """The pin of a committed DOSE copy (`d<tok>__<cell>`), held to the digest's own live row (demo-import, Q12 narrow): derivation dose_scale,
    base_scale == the view's dose, reference|approved, kind / label the parent's declared ones, parents == {the parent's pin in the SAME block},
    that pin present and equal to the committed sidecar's dose_scaled_from.parent_object_sha256; a committed entry with the same id is kept
    VERBATIM, else the row's facts are written. A registered copy of another parent, at another dose, or not a dose_scale row refuses by name."""
    msha = str(mrow.get("object_sha256") or "")
    ident = f"{key}/{sid} ({view}: {label} x{dose:g} of {parent_sid}; committed {mrow.get('source_path')}, {kind}, base_scale {dose:g})"
    row = live_object_identity(registry_root, msha)
    if row is None or str(row.get("status")) not in ("reference", "approved"):
        raise SystemExit(f"{ident}: the committed copy {msha[:12]} has no reference/approved registry row" + (f" (status {row.get('status')!r})" if row else "")
                         + " — a pin is registry-backed or not written (S2RLFIX3); import the committed objects first; refusing to write the pins")
    if str(row.get("derivation") or "") != "dose_scale":
        raise SystemExit(f"{ident}: the registry row {msha[:12]} ({row['status']}) has derivation {str(row.get('derivation'))!r}, not 'dose_scale' — a scaled copy is "
                         "registered as one; refusing to write the pins")
    bs = row.get("base_scale")
    if bs is None or abs(float(bs) - float(dose)) > 1e-9:
        raise SystemExit(f"{ident}: the registry row {msha[:12]} is a copy at base_scale {bs!r}, the view's dose is {dose:g}; refusing to write the pins")
    if str(row.get("object_kind")) != kind or str(row.get("paper_label")) != label:
        raise SystemExit(f"{ident}: the registry row {msha[:12]} is {row.get('object_kind')}/{row.get('paper_label')}, the parent's identity is {kind}/{label}; refusing to write the pins")
    parent_pin = str(((block or {}).get(parent_sid) or {}).get("object_sha256") or "")
    if not parent_pin:
        raise SystemExit(f"{ident}: the parent {parent_sid} carries no pin in this block — pin the parent before its dose copy")
    dm = committed_sidecar(mrow, ident)
    stamp = dm.get("dose_scaled_from") if isinstance(dm.get("dose_scaled_from"), dict) else {}
    sparent = str(stamp.get("parent_object_sha256") or "")
    if not sparent:
        raise SystemExit(f"{ident}: the committed sidecar carries no dose_scaled_from.parent_object_sha256; refusing to write the pins")
    if sparent != parent_pin:
        raise SystemExit(f"{ident}: the committed copy was scaled from {sparent[:12]}, the pinned parent is {parent_pin[:12]} — the copy does not belong to the "
                         "pinned parent; refusing to write the pins")
    got = {str(x) for x in (_json_col(row, "parent_object_ids_json", []) or [])}
    if got != {parent_pin}:
        raise SystemExit(f"{ident}: the registry row {msha[:12]} names parents {sorted(x[:12] for x in got)}, the pinned parent is {parent_pin[:12]} — "
                         f"the copy was not scaled from the pinned parent; refusing to write the pins")
    if have and str(have.get("object_sha256")) != msha:
        raise SystemExit(f"{manifest_path}: {ident}: the committed pin {str(have.get('object_sha256'))[:12]} is not the staged copy {msha[:12]}; refusing to write the pins")
    if have:
        return dict(have)
    return {"object_sha256": msha, "source": str(mrow.get("source_path")) + f" (registry {row.get('campaign')}:{row.get('status')}; dose-scaled copy of {parent_pin[:12]})",
            "verified_cluster_a": bool(mrow.get("cluster_a_verified")), "verified_cluster_b": bool(mrow.get("cluster_b_verified")),
            "sidecar_object_sha256_recorded": (mrow.get("recorded_hash_check") or {}).get("status")}


def dump_pins_like_committed(pins: dict, like: Path) -> str:
    """The pins serialised in the COMMITTED file's own shape (its indentation, sorted keys, one trailing newline), so a registry-only rewrite
    changes nothing but the entries it names — the committed pins.json is indented by one space (the Modal-era writer), the manifest writer
    uses two; whichever the file carries is kept."""
    text = like.read_text(encoding="utf-8") if like.is_file() else ""
    indent = 2
    for line in text.splitlines()[1:3]:
        stripped = line.lstrip(" ")
        if stripped.startswith('"'):
            indent = len(line) - len(stripped); break
    return json.dumps(pins, indent=indent, sort_keys=True) + "\n"


def harness_cell_id(tt_id: str) -> str:
    if tt_id == "none":
        return "none"
    if is_prompted(tt_id):
        # a prompted baseline: TrainTest `prompted__<si>` <-> harness `prompted__<harness si>`
        # (registry_adapter.harness_cell_id's own rule; the wave test pins the two equal)
        si = tt_id.split("__", 1)[1]
        if si not in TT_TO_HARNESS_SI:
            raise SystemExit(f"{tt_id!r}: SI {si!r} has no Petri harness translation in this "
                             "generator (TT_TO_HARNESS_SI) — extend it deliberately")
        if TT_TO_HARNESS_SI[si] not in PREFIX_HASHES:
            raise SystemExit(f"{tt_id!r}: harness SI {TT_TO_HARNESS_SI[si]!r} has no recorded SI "
                             "prefix hash (PREFIX_HASHES) — a prompted leg is verified per request "
                             "against it (H8); record it deliberately")
        return f"prompted__{TT_TO_HARNESS_SI[si]}"
    if "__" not in tt_id:
        # a BARE Class B id (`arditi`): no SI, and the harness spells it identically
        # (registry_adapter's own rule). Refused by name when unregistered, so a typo can never
        # ride into a manifest as a cell nothing can serve.
        if tt_id not in BARE_CLASS_B_IDS:
            raise SystemExit(f"{tt_id!r}: not a cell id this generator knows — expected "
                             f"`<si>__<arm>`, `prompted__<si>`, or a bare Class B id "
                             f"{list(BARE_CLASS_B_IDS)}")
        return tt_id
    si, arm = tt_id.split("__", 1)
    if si not in TT_TO_HARNESS_SI:
        raise SystemExit(f"{tt_id!r}: SI {si!r} has no Petri harness translation in this "
                         "generator (TT_TO_HARNESS_SI) — extend it deliberately")
    return f"si_{TT_TO_HARNESS_SI[si]}__{arm}"


# --------------------------------------------------------------------------- configs

def _sub_line(body: str, key_regex: str, new_line: str, *, count: int = 1) -> str:
    n = len(re.findall(key_regex, body, flags=re.M))
    if n != count:
        raise SystemExit(f"expected {count} line(s) matching {key_regex!r}, found {n}")
    return re.sub(key_regex, new_line, body, count=count, flags=re.M)


def _pin_value(pins_text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}: (.*)$", pins_text, re.M)
    if not m:
        raise SystemExit(f"pins config has no top-level {key!r}")
    return m.group(1).rstrip()


def render_config(roster: dict, pins: dict, target: str, view: str, *, roster_path: Path = ROSTER) -> str:
    rrel = roster_rel(roster_path)     # the roster the header names (the production literal by default)
    t = roster["targets"][target]
    v = t["views"][view]
    cells = served_cells(roster, target, view)
    dose = view_dose(roster, target, view)   # None, or the dose this view serves (refuses a name/key mismatch)
    paired_seed = view_paired_of(roster, target, view)   # None, or the seed view this PAIRED view re-serves from (the twin marker)
    reserved = paired_reserved_cells(roster, target, view)
    body_text = (REPO / t["body_config"]).read_text(encoding="utf-8")
    marker = "# -- everything below:"
    body = body_text[body_text.index(marker):]
    body = re.sub(r"^# -- everything below: .*$",
                  f"# -- everything below: {Path(t['body_config']).name}'s data, unchanged, EXCEPT "
                  "the keys the header names --", body, count=1, flags=re.M)
    tag = target + ("" if view == "main" else f"_{view}")
    # the enumeration every artefact of this view carries — `none` + the served cells for an ordinary view, EXACTLY the
    # served cells for a steered-wave view (s<N>r; no unsteered leg is generated for it — the standup serves the
    # base route implicitly for its health checks and attestation, never listed, never measured)
    universe = view_cell_universe(roster, target, view)
    cells_intro = (f"#   * `explicit_cells` = EXACTLY the {len(cells)} cell(s) of the roster's `{view}` view — a STEERED-WAVE view "
                   + ("(s<N>r)" if STEERED_WAVE_VIEW_RE.match(str(view)) else "(a front-door view other than u0)") + "\n"
                   "#     lists NO `none`: it measures no unsteered leg (the u0 wave is the baseline). The standup\n"
                   "#     serves the base route implicitly (mt_serving_up select_cells) for its health checks and attestation — served, never\n"
                   "#     listed, never registered or scored. Each cell is served from the staged"
                   if steered_wave_view(view, target) else
                   f"#   * `explicit_cells` = [none, the {len(cells)} cell(s) of the roster's `{view}` view]. `none`\n"
                   "#     first is the SAME-WAVE bare anchor (greedy generations differ run-to-run from numeric noise\n"
                   "#     alone, so every contrast pairs against a bare run generated in THIS wave on THIS serving")
    body = _sub_line(body, r"^defenses_dir: .*$", f"defenses_dir: out/full22/{target}/cells_view/{view}")
    body = _sub_line(body, r"^out_dir: .*$", f"out_dir: out/sdfbase_full22_{tag}")
    body = _sub_line(body, r"^private_run: .*$", f"private_run: sdfbase_full22_{tag}")
    pins_note = ""
    if t.get("pins_config"):
        ptxt = (REPO / t["pins_config"]).read_text(encoding="utf-8")
        for key in ("model", "model_revision", "layer", "max_len", "teacher_prefix_cache",
                    "pool_manifest"):
            body = _sub_line(body, rf"^{key}: .*$", f"{key}: {_pin_value(ptxt, key)}")
        store = re.search(r"^  store_dir: (.*)$", ptxt, re.M)
        if not store:
            raise SystemExit(f"{t['pins_config']} has no pool.store_dir")
        body = _sub_line(body, r"^  store_dir: .*$", f"  store_dir: {store.group(1).rstrip()}")
        pins_note = (f"#   * model / model_revision / layer / max_len / teacher_prefix_cache / "
                     f"pool_manifest / pool.store_dir are {t['pins_config']}'s (this target has no\n"
                     f"#     sdfbase transfer lineage; the battery body is {t['body_config']}'s).\n")
    fd = front_door_model(target)
    if fd is not None and not t.get("pins_config"):
        # THE FRONT DOOR (module note above MODEL_SERVING): the roster block's served checkpoint IS the table's serve.checkpoint (held
        # equal — a block pasted from another table refuses by name) and replaces the body's top-level model / model_revision; the
        # body's FIT-SIDE keys (layer / max_len / teacher_prefix_cache / pool_manifest / pool.store_dir — what a pins_config
        # substitutes for a two_si-lineage target) stay the battery body's data: the wave's serving and eval path reads none of them,
        # and a front-door target's fits are the sdfbase document fits (configs/sdfbase_<fit model_id>_fits.yaml), not a two_si pool
        ck = (fd.get("serve") or {}).get("checkpoint") or {}
        if (t["model"], t["model_revision"]) != (ck.get("hf_repo"), ck.get("revision")):
            raise SystemExit(f"{target}: the roster block pins {t['model']}@{t['model_revision']} but {MODELS_TABLE.name} serve.checkpoint is "
                             f"{ck.get('hf_repo')}@{ck.get('revision')} — one served checkpoint; re-print the block (05_stage_roster.sh --step roster)")
        body = _sub_line(body, r"^model: .*$", f"model: {t['model']}")
        body = _sub_line(body, r"^model_revision: .*$", f"model_revision: {t['model_revision']}")
        pins_note = (f"#   * model / model_revision are the FRONT DOOR's served checkpoint ({MODELS_TABLE.name} `{target}` serve.checkpoint\n"
                     f"#     = this roster block's; make_full22_configs.front_door_model). The battery body is {t['body_config']}'s; its\n"
                     "#     fit-side keys (layer / max_len / teacher_prefix_cache / pool_manifest / pool.store_dir) are that file's data — the\n"
                     "#     wave's serving and eval path reads none of them, and this target's fits are the sdfbase document fits\n"
                     f"#     (configs/sdfbase_{((fd.get('fit') or {}).get('checkpoint') or {}).get('model_id', '<fit model_id>')}_fits.yaml), not a two_si pool.\n")
    if t["model"] != _pin_value(body, "model") or t["model_revision"] != _pin_value(body, "model_revision"):
        raise SystemExit(f"{target}: roster pins {t['model']}@{t['model_revision']} but the body "
                         f"serves {_pin_value(body, 'model')}@{_pin_value(body, 'model_revision')}")
    smnt = int(roster["static_max_new_tokens"])
    tmnt = int(roster["target_max_new_tokens"])
    cmnt = int(roster["controls_max_new_tokens"])
    # the MULTI-TURN target's per-reply cap: the roster's wave-wide value replaces the body's
    # (the mt1 body says 512; the canonical instrument since 2026-09-11 is 4,096)
    body = _sub_line(body, r"^  target_max_new_tokens: \d+$",
                     "  # full22: the multi-turn TARGET's per-reply cap (Crescendo / PAIR / ActorAttack;\n"
                     "  # jailbreak_run._target_max_new_tokens) -- the roster's target_max_new_tokens. 512 was\n"
                     "  # mt1's and every filed 512-cap store's; 4,096 is the canonical instrument (2026-09-11)\n"
                     f"  target_max_new_tokens: {tmnt}\n"
                     "  # full22: the judged OVER-REFUSAL controls' completion cap (XSTest / OR-Bench;\n"
                     "  # sisteer.eval.cell_controls.controls_cap) -- the roster's controls_max_new_tokens: 512,\n"
                     "  # the sampled ladder's root and every filed greedy control store's cap. Its own key\n"
                     "  # since 2026-09-11, so the multi-turn cap above moves without moving the controls\n"
                     f"  controls_max_new_tokens: {cmnt}")
    # THE CANONICAL BEST-OF-N: the 512 store + the 4,096 climb (the roster's wave-wide values). The BASE
    # cap `bon.max_new_tokens` is untouched -- the climb escalates it, never replaces it. The climb's SAMPLER is
    # the store's (`as_drawn`, bon_climb.CLIMB_DECODE): best-of-N draws at each served
    # model's vendor preset (the registry's `<model>|bon|off|vendor_preset` row), so no per-lane temperature
    # is rendered here -- a store drawn at the grandfathered 1.0 is climbed at 1.0, a vendor-preset store at
    # its preset, and a native pass draws at the preset.
    ccap, cdec = int(roster["bon_climb_cap"]), str(roster["bon_climb_decode"])
    body = _sub_line(body, r"^  max_new_tokens: (\d+)$",
                     "  # the BASE best-of-N cap: the 512 draws the canonical instrument climbs FROM\n"
                     r"  max_new_tokens: \1" + "\n"
                     "  # THE CANONICAL BEST-OF-N (2026-09-11) = this store PLUS the climb of its cap-coherent\n"
                     "  # at-cap draws to climb_cap with the SAME per-draw seed and the SAME SAMPLER the store\n"
                     "  # records (climb_decode as_drawn: a store drawn at the grandfathered T 1.0 climbs at 1.0,\n"
                     "  # a vendor-preset store at its preset; the 512 reading is a FLOOR: an at-cap draw is\n"
                     "  # judged jailbroken ~19x more often than a terminated one). A cell with no base store\n"
                     "  # draws NATIVELY at climb_cap, at the served model's vendor preset (the registry's\n"
                     "  # <model>|bon|off|vendor_preset row; user ruling 2026-09-17). Loop-flagged draws are\n"
                     "  # never climbed -- an accepted, reported residual floor.\n"
                     f"  climb_cap: {ccap}\n"
                     f"  climb_decode: {cdec}")
    body = _sub_line(body, r"^  attacker_max_new_tokens: 1024$",
                     "  attacker_max_new_tokens: 1024\n"
                     "  # full22: the static/replayable single-shot families (pasttense, codeattack,\n"
                     "  # renellm, grok420) decode to THIS cap (jailbreak_run._static_max_new_tokens)\n"
                     f"  static_max_new_tokens: {smnt}")
    body = _sub_line(body, r"^  attn_implementation: sdpa$",
                     "  attn_implementation: sdpa\n"
                     "  # full22: the attacker's serving defaults, IDENTICAL for every cell of every\n"
                     "  # target (configs/attacker_serving.json on this tree; the sbatch passes them\n"
                     "  # explicitly and attacker_engine.json records them per shard)\n"
                     f"  prefix_caching: {str(roster['attacker_prefix_caching']).lower()}\n"
                     f"  quantization: {roster['attacker_quantization']}")
    # THE TOKEN CAPS (the sdfbase-llama8b lane, 2026-09-10; roster targets.<key>.caps).
    # The multi-turn target cap becomes attacks.multiturn_target_max_new_tokens — a per-family key the multi-turn path
    # reads (jailbreak_run._multiturn_target_max_new_tokens) beside main's wave-wide attacks.target_max_new_tokens (4,096;
    # the two agree on every family since 2026-09-11); the controls' cap is the roster's
    # attacks.controls_max_new_tokens (cell_controls.controls_cap; the caps table's controls_ladder asserts it);
    # every other cap must EQUAL the value the body / the roster already carries for its leg — the table is the
    # authority and a disagreement REFUSES (never two values for one leg, never a silent default). The multi-turn
    # serving window is DERIVED from the key (geometry.multiturn_window), by the attacks sbatch and the printers.
    caps = target_caps(roster, target)
    cap_lines = []
    if caps:
        bdata = yaml.safe_load(body)
        have = {"attacker": ("attacks.attacker_max_new_tokens", bdata["attacks"]["attacker_max_new_tokens"]),
                # the judged controls' cap rides its OWN key since 2026-09-11 (attacks.controls_max_new_tokens, the roster's;
                # cell_controls.controls_cap) — attacks.target_max_new_tokens is the multi-turn cap (4,096) on every config
                "controls_ladder": ("the roster's controls_max_new_tokens (attacks.controls_max_new_tokens)", cmnt),
                "static": ("the roster's static_max_new_tokens", smnt)}
        for leg, (where, val) in have.items():
            if int(val) != int(caps[leg]):
                raise SystemExit(f"{target}: caps.{leg}={caps[leg]} but {where} is {val} — the caps table is "
                                 "authoritative; move the body / roster value deliberately, never a second value")
        # bon.max_new_tokens is the ONE cap the table WRITES rather than asserts (2026-09-11): the sdfbase-llama8b BoN leg is a
        # NATIVE best-of-75 pass at the table's cap (4,096), fresh samples — the other lanes' 512 store + the climb to 4,096 is a
        # different route to the same estimand — so the body's campaign cap (512) is replaced here, once, from the roster; the
        # load check and bon_served_run read this key
        body = _sub_line(body, r"^  max_new_tokens: \d+$",
                         "  # the sdfbase-llama8b lane: the NATIVE best-of-75 pass at the roster's\n"
                         "  # caps.bon, WRITTEN by the generator (the body's 512 is the campaign's cap; route native_4096: no 512 store, no climb)\n"
                         f"  max_new_tokens: {int(caps['bon'])}")
        # PER FAMILY (2026-09-10: Crescendo / PAIR 4,096, ActorAttack at its filed 512): the
        # key is a map family -> cap; jailbreak_run resolves the family it runs and REFUSES a config lacking that family's entry
        fam = caps["multiturn_target"]
        fam_src = " ".join(str(caps["multiturn_target_source"]).split())
        mt_block = "  multiturn_target_max_new_tokens:\n" + "\n".join(f"    {f}: {int(fam[f])}" for f in MULTITURN_FAMILIES)
        body = _sub_line(body, r"^  controls_max_new_tokens: \d+$",
                         f"  controls_max_new_tokens: {cmnt}\n"
                         "  # the sdfbase-llama8b lane (the lane's caps, PER FAMILY: Crescendo / PAIR 4,096, ActorAttack at its filed 512 at\n"
                         "  # first, every family 4,096 since 2026-09-11): the MULTI-TURN target reply cap per attack family, read by the\n"
                         "  # Crescendo / ActorAttack / PAIR\n"
                         "  # path (jailbreak_run._multiturn_target_max_new_tokens resolves the family it runs and REFUSES a config lacking\n"
                         "  # that family's entry; an explicit --target-max-new-tokens wins) — FLAT per family, no ladder (a truncated reply\n"
                         "  # feeds the next attacker turn); target_max_new_tokens above is the roster's wave-wide 4,096 (the same cap\n"
                         "  # through main's reader) and controls_max_new_tokens the controls ladder's root (the caps table's\n"
                         "  # controls_ladder). Each family's multi-turn serving window is DERIVED from ITS value\n"
                         "  # (geometry.multiturn_window(cap)) or PINNED below, never a literal; the load check sizes its probe from the\n"
                         "  # family(ies) the standup serves.\n"
                         f"  # source: {fam_src}\n"
                         + mt_block)
        cap_lines = [f"#   * the TOKEN CAPS of this lane (roster targets.{target}.caps; source: {caps['source']}): the generator",
                     "#     applied multiturn_target as the PER-FAMILY map attacks.multiturn_target_max_new_tokens (the multi-turn path's cap,",
                     "#     resolved for the family it runs; each family's serving window is DERIVED from ITS cap, geometry.multiturn_window)",
                     "#     WROTE bon.max_new_tokens (the NATIVE best-of-75 pass at the table's cap) and asserted",
                     "#     every other cap equals the body's / roster's:",
                     "#       multiturn_target " + " / ".join(f"{f} {fam[f]}" for f in MULTITURN_FAMILIES) + f" (source: {fam_src})",
                     "#       " + "  ".join(f"{leg} {caps[leg]}" for leg in CAP_LEGS if leg != "multiturn_target")]
        # the keep-or-drop switch: a dropped multi-turn family is a recorded omission on the config, never an inferred absence
        for f_om, why_om in target_multiturn_families(roster, target)[1].items():
            cap_lines.append(f"#     multi-turn family OMITTED {f_om}: {why_om}")
        # THE PINNED WINDOWS (2026-09-10; roster caps.windows): the multi-turn serving window and the over-window
        # re-run window PINNED WIDER than the derivation — validated HERE through geometry's own rule (pin_window: refused below
        # the derived minimum or off the grid; a re-run window not wider than the first pass refused), so a bad pin refuses at
        # generation, never at a standup — and written as attacks.multiturn_max_model_len / multiturn_rerun_max_model_len, the keys
        # target_attacks.sbatch and the re-run pass read (S = the model's budget // L through the standup's strict identity). A
        # target without pins carries neither key and derives exactly as before.
        # ... the VIEW's first-pass overrides laid over the target's pins (f22_roster.view_multiturn_windows — the llama8b shortconst
        # view widens PAIR to 18,688 because its ~5.9k-token prefix pushes the measured maximum past the target's 16,384; only
        # wider, and still below the family's re-run pin): THIS view's config carries the served pins
        win = view_multiturn_windows(roster, target, view)
        if win:
            from sisteer.capeval.serving.geometry import multiturn_rerun_window, multiturn_window
            # PER FAMILY: each pinned family's windows at ITS cap through geometry's rule (a family without a block derives and
            # carries no entry in the maps; a family absent from the map is refused by the readers, never defaulted)
            L_mt, L_re, srcs = {}, {}, {}
            for f in MULTITURN_FAMILIES:
                b = win.get(f)
                if not b:
                    continue
                L_mt[f] = multiturn_window(int(fam[f]), b.get("multiturn"))
                L_re[f] = multiturn_rerun_window(int(fam[f]), b.get("multiturn_rerun"), window=L_mt[f])
                srcs[f] = " ".join(str(b["source"]).split())
            if body.count(mt_block) != 1:
                raise SystemExit(f"{target}: expected one multiturn_target_max_new_tokens map to hang the pinned windows on, found {body.count(mt_block)}")
            body = body.replace(mt_block, mt_block + "\n"
                                "  # the multi-turn SERVING WINDOWS PER FAMILY, PINNED by the roster's caps.windows (2026-09-10):\n"
                                "  # geometry.multiturn_window(cap, pin) / multiturn_rerun_window(cap, pin) honour each\n"
                                "  # family's pin over ITS derived value and refuse a pin below the derived minimum (p99 prompt + cap; measured max call +\n"
                                "  # cap); S = the model's budget // L. Read by target_attacks.sbatch (the family's entry; a combined job serves at the max\n"
                                "  # over its families) and the over-window re-run pass; the load check sizes its probe from the family's cap. Sources:\n"
                                + "\n".join(f"  #   {f}: {srcs[f]}" for f in MULTITURN_FAMILIES if f in srcs) + "\n"
                                + "  multiturn_max_model_len:\n" + "\n".join(f"    {f}: {L_mt[f]}" for f in MULTITURN_FAMILIES if f in L_mt) + "\n"
                                + "  multiturn_rerun_max_model_len:\n" + "\n".join(f"    {f}: {L_re[f]}" for f in MULTITURN_FAMILIES if f in L_re), 1)
            cap_lines.append("#     the multi-turn serving windows are PINNED PER FAMILY (roster caps.windows): "
                             + "; ".join(f"{f} first pass {L_mt[f]} ({'pinned' if win[f].get('multiturn') else 'derived'}), over-window re-run {L_re[f]} "
                                         f"({'pinned' if win[f].get('multiturn_rerun') else 'derived'})" for f in MULTITURN_FAMILIES if f in L_mt)
                             + " — attacks.multiturn_max_model_len / multiturn_rerun_max_model_len (per-family maps)")
            for f in MULTITURN_FAMILIES:
                if f in srcs:
                    cap_lines.append(f"#       {f} windows pinned on: {srcs[f]}")
        for ctl, why in target_capability_omitted(roster, target).items():
            cap_lines.append(f"#     capability control OMITTED {ctl}: {why}")
    # THE PER-VIEW BoN CAP (2026-09-15): a view of a target with NO caps table
    # may declare `caps: {bon: N}` — the NATIVE best-of-75 pass at N (no 512 store, no climb: the sdfbase-llama8b route)
    # for THIS view only, written through the same _sub_line the caps table uses. view_declared_bon_cap refuses it on a
    # caps-table target (two authorities for one cap) and on a dose view; every other view of the target keeps the
    # body's 512 byte-for-byte. bon_served_run.py reads this key, so the view's request stream carries max_tokens N.
    vcap = view_declared_bon_cap(roster, target, view)
    if vcap is not None:
        # the climb substitution above labelled the body's line "the 512 draws the canonical instrument climbs FROM"; on a
        # native-route view that label is false, so BOTH lines are replaced (one match, or the substitution refuses)
        body = _sub_line(body, r"^  # the BASE best-of-N cap: the 512 draws the canonical instrument climbs FROM\n  max_new_tokens: \d+$",
                         f"  # bon.max_new_tokens {vcap}: the view's caps.bon (native best-of-75 pass, no 512 store, no climb — the sdfbase-llama8b route)\n"
                         f"  max_new_tokens: {vcap}")
        cap_lines += [f"#   * bon.max_new_tokens {vcap}: the view's caps.bon (native best-of-75 pass, no 512 store, no climb — the sdfbase-llama8b route)",
                      f"#     ({rrel} views.{view}.caps; make_full22_configs.view_declared_bon_cap — admitted only on a target with NO caps",
                      f"#     table and never on a dose view; bon_served_run.py reads this key, so this view's best-of-N request stream carries",
                      f"#     max_tokens {vcap}; every other view of {target} keeps the body's {BON_MAX_NEW_TOKENS} store + climb byte-for-byte)"]
    tp = pins.get(target, {})
    cell_lines = []
    for cid, sid in cells.items():
        if sid is None:   # a prompted baseline: the SI text as the system prompt, no object, no pin
            hsi = TT_TO_HARNESS_SI[cid.split("__", 1)[1]]
            cell_lines.append(f"#       {cid:<32} <- (prompted baseline: the SI as the system prompt; no object)")
            cell_lines.append(f"#       {'':<32}    si_prefix sha16 {PREFIX_HASHES[hsi]} (verified per request, H8)")
            continue
        if isinstance(sid, dict):
            # the CROSS-BANK PIN FORM: the cell names a PIN in
            # another authority and every identity fact — the staged id included — comes from it,
            # so the header echoes the pin, not a roster-chosen rung
            staged_id, bank, sha, ref = pin_form_facts(target, cid, sid)
            cell_lines.append(f"#       {cid:<32} <- {staged_id}  (bank {bank}, THROUGH the pin "
                              f"{sid['pins']} {ref})")
            cell_lines.append(f"#       {'':<32}    object_sha256 {sha}")
            continue
        sha = tp.get(sid, {}).get("object_sha256") or "UNPINNED (not in sdfbase_full22_pins.json)"
        cell_lines.append(f"#       {cid:<32} <- {sid}")
        cell_lines.append(f"#       {'':<32}    object_sha256 {sha}")
        if cid in reserved:
            cell_lines.append(f"#       {'':<32}    RE-SERVED from view `{paired_seed}` (the same staged id in a second standup: a STANDUP replicate of that view's run, never a seed replicate)")
    pending = pending_fit_of(roster, target, view)   # target-level + this view's own pending fits
    pending_stage = t.get("pending_stage") or {}
    legs = view_legs(roster, target, view)
    omitted = view_legs_omitted(roster, target, view)
    serving = view_serving(roster, target, view)
    prompted_only = view_prompted_only(roster, target, view)
    if prompted_only and not cells:
        raise SystemExit(f"{target}/{view}: a prompted_only view must serve at least one prompted baseline")
    # the legs/serving block is emitted ONLY for a view that departs from the full battery at the
    # table's windows (the sdfbase-llama8b fullconst view): every earlier view's config stays
    # byte-identical (the sampled configs embed a wave config's bytes and record its sha)
    leg_lines = []
    if legs != LEGS or omitted or serving or prompted_only:
        leg_lines = [f"#   * the view's LEGS and serving ({rrel} views.{view}; f22_roster.view_legs /",
                     "#     view_serving — the printers, the sbatch scripts and the wave manifest read the same functions):",
                     f"#     legs: {', '.join(legs)}" + ("  — every other leg OMITTED, with its reason:" if omitted else "  — EVERY leg of the battery (no omission)")]
        leg_lines += [f"#       omitted {leg:<11} {why}" for leg, why in omitted.items()]
    if serving.get("max_model_len"):
        leg_lines.append(f"#     serving: max_model_len {serving['max_model_len']} for EVERY standup of this view "
                         "(common.sh passes it; S = budget_for(model) // L, the standup's strict identity)")
    if serving.get("windows"):
        vw = view_windows(roster, target, view)
        mt_over = dict(serving["windows"].get("multiturn") or {})
        leg_lines.append(f"#     serving: PER-LEG windows (roster views.{view}.serving.windows; f22_roster.view_windows — the sbatch scripts read")
        leg_lines.append("#       them through roster_view.py `window <leg>` / `rung_window <rung>`, the printers the same function; S = budget_for(model) // L):")
        leg_lines.append(f"#       static + BoN {vw['static']}; controls (MMLU letter prefill + xstest / or_bench) {vw['controls']}; ladder rungs "
                         + ", ".join(f"{r} -> {vw['rungs'][r]}" for r in sorted(vw['rungs']))
                         + (f"; multi-turn first-pass overrides over the target's caps.windows pins: " + ", ".join(f"{f} {L}" for f, L in mt_over.items()) if mt_over else "")
                         + "; Petri at the certified 16,384")
        leg_lines.append(f"#       windows measured on: {' '.join(str(vw['source'] or '').split())}")
    if prompted_only:
        leg_lines.append("#     prompted_only: `none` + the prompted baseline(s) above, NO object cell by design — the")
        leg_lines.append("#       bank view for this view is empty and the count guard expects zero staged entries")
    xsub = target_xstest_subset(roster, target)
    if xsub:
        leg_lines.append(f"#     xstest_subset: {xsub} — the over-refusal control is the 250 SAFE XSTest prompts only (the")
        leg_lines.append("#       contrast half is a different measurement); the key is ADDED top-level, the drivers select the")
        leg_lines.append("#       half BEFORE the seeded subset / item hashes / ladder inputs, and a 250-row store keys apart")
    pending_decision = pending_decision_of(roster, target)
    lane_lines = []
    for lane, enabled, lcells in lanes_of(roster, target, view):
        if enabled:
            continue    # an enabled lane's cells are in `cells` above, not here
        for cid, sid in lcells.items():
            lane_lines.append(f"#   lane {lane} OFF  {cid:<32} <- {sid}  (lane_flags.{lane}.enabled: false)")
    not_served = t.get("not_served") or []
    dose_note = ""
    if dose is not None:
        # the two caps a dose lane once moved and must not: name what each IS on this target, truthfully — a
        # caps-table target (sdfbase-llama8b) has its BoN cap WRITTEN from the table on every view, the rest keep
        # the body's; `attacks.target_max_new_tokens` is the roster's wave-wide value since main's 2026-09-11 landing
        if caps and caps.get("bon") is not None:
            bon_line = (f"#     else moves: `bon.max_new_tokens` is the target's caps-table {int(caps['bon'])} on every view (the NATIVE\n"
                        f"#     best-of-N pass — no 512 store, no climb) and `attacks.target_max_new_tokens` the roster's wave-wide {tmnt};\n")
        else:
            bon_line = (f"#     else moves: `bon.max_new_tokens` STAYS the body's {BON_MAX_NEW_TOKENS} (since 2026-09-10, superseding an\n"
                        f"#     earlier native-4096 override) — best-of-N follows Scale's 512 -> 4096 climb, whose tooling\n"
                        f"#     refuses a store unless the cap is 512; `attacks.target_max_new_tokens` is the roster's wave-wide {tmnt};\n")
        fam_line = ""
        if len(dose_families(roster, target)) == 2:
            fam_line = ("#     This target serves BOTH ladder families on ONE model: the `it_d*` views the instruct-fit objects natively,\n"
                        "#     the `d*` views the base-fit objects TRANSFERRED onto it.\n")
        objfam = _DOSE_VIEW_RE.match(view).group("objfam")
        objfam, objlane = split_family_lane(objfam) if objfam else (None, None)   # `relul` = family relu on the LeCun lane
        if objlane:
            fam_line += (f"#     A LANE view (`object_lane: {objlane}`; the name's trailing `{DOSE_LANES[objlane]}`): the copy is scaled from the\n"
                         f"#     {DOSE_PARENT_LANE[objlane]}-lane parent (the LeCun-init refit, bank full22_lecun), never the zero-init composition's.\n")
        if objfam:      # a NAMED object family (the ReLU ladder, 2026-09-16); the frozen bare LoRA views say nothing new
            fam_line += (f"#     An OBJECT-FAMILY view (`{objfam}`: the view name's family segment, the roster's `object_family:`): every\n"
                         f"#     cell it serves is {OBJECT_FAMILIES[family_of_segment(objfam)][1]}, held by the generator, and it shares neither a view nor an\n"
                         "#     anchor with the frozen bare LoRA ladder (BARE_DOSE_VIEWS_FROZEN) — one object family per dose view.\n")
        dose_note = (f"#   * a DOSE view (dose {dose:g}; the strength frontier's ladder, the roster's DOSE LADDER block). NOTHING\n"
                     + bon_line + "#     a dose view moves the OBJECT, never a cap.\n" + fam_line)
    paired_note = ""
    if paired_seed is not None:
        paired_note = (f"#   * a PAIRED view (`paired_of: {paired_seed}` — the twin marker, ReLU's PR #300 review): the composed arm is this view's OWN;\n"
                       f"#     {', '.join(f'`{c}`' for c in reserved)} {'is' if len(reserved) == 1 else 'are'} RE-SERVED from view `{paired_seed}` — the same staged\n"
                       f"#     id(s) in a second standup, so their runs here are STANDUP replicates of `{paired_seed}`'s, never seed replicates: a\n"
                       f"#     reader keys on object sha and never pools the twins as independent samples. The GSM8K matrix HOLDS these rows on\n"
                       f"#     the paired views (measured on `{paired_seed}`); misalign measures them (the in-view difference read).\n")
    header = f"""# sdfbase_full22_{tag}.yaml — the full22 wave (sdf-base-steering-r2): the whole
# battery — Crescendo / ActorAttack / PAIR (5x3, forward_cap 150) over the 259-goal confirmatory
# pool, the static families pasttense / codeattack / renellm / grok420 at the {smnt}-token
# completion cap, best-of-{'75'} on the 100 JBB goals, the served controls, and the Petri audit —
# on the vLLM-served lens target under the STACKED reduction mode, with a FRESH same-wave bare
# anchor. GENERATED by scripts/sdfbase/make_full22_configs.py from
# {rrel} (target {target}, view {view}) — never hand-edited; the test
# requires byte-identity. DERIVED from {t['body_config']}: every key below is that file's,
# byte-for-byte as data, EXCEPT —
#
{cells_intro}
#     path). Cell ids are the code's grammar (<si>__<arm>); each is served from the staged
#     directory the roster maps it to, through the bank view
#     $SHARED/exp/objects/sdfbase/full22/{target}/cells_view/{view}/<cell_id> (make_bank_view.py,
#     sha-verified against the pins below):
{chr(10).join(cell_lines) if cell_lines else '#       (none — `none` only until the roster gains cells for this view)'}
{(chr(10).join(leg_lines) + chr(10)) if leg_lines else ''}{(chr(10).join(cap_lines) + chr(10)) if cap_lines else ''}{_cert_roster_note(roster_path)}#   * `defenses_dir` names that view (on the cluster it is the TT_TWO_SI_DEFENSES_DIR export every
#     job sets; the defenses-dir resolution order).
#   * `attacks.target_max_new_tokens` is the roster's ({tmnt}: the multi-turn target's per-reply
#     cap, the canonical 4,096 instrument of 2026-09-11), `attacks.controls_max_new_tokens` ({cmnt}: the
#     judged over-refusal controls' cap, its own key since 2026-09-11), `attacks.static_max_new_tokens` ({smnt})
#     and `attacker.prefix_caching` / `attacker.quantization` are ADDED (the roster's wave-wide values).
#   * out_dir / private_run get the fresh `sdfbase_full22_{tag}` namespace.
{dose_note}{paired_note}{pins_note}#
# Roster entries NOT in this config ({rrel} names what each waits on):
{chr(10).join(f'#   pending_fit      {cid:<32} <- {sid}' for cid, sid in pending.items()) if pending else '#   pending_fit      (none)'}
{chr(10).join(f'#   pending_stage    {cid:<32} <- {sid}' for cid, sid in pending_stage.items()) if pending_stage else '#   pending_stage    (none)'}
{chr(10).join(f"#   pending_decision {cid:<32} <- {e['staged_id']}  [waits on {e['waits_on']}] {' '.join(str(e['state']).split())}" for cid, e in pending_decision.items()) if pending_decision else '#   pending_decision (none)'}
{chr(10).join(lane_lines) if lane_lines else '#   (no lane off)'}
{chr(10).join(f"#   {str(e.get('status')):<13} {e['staged_id']}" for e in not_served) if not_served else ''}#
# Serving: scripts/sdfbase/mt_serving_up.py --mode attacks|petri --reduction-mode stacked
# --require-attested-mode stacked (one vLLM+lens server per (target, view) serving every cell as a
# virtual model; objects at their recorded fit leg, so drivers pass NO --positions); every driver
# runs --target-engine served with SISTEER_LENS_STACKED_REDUCTIONS=1 exported client-side too.
# Attacker: the config's huihui-ai/Huihui-Qwen3.6-27B-abliterated on vLLM (--max-model-len 49152,
# thinking stripped, prefix caching {str(roster['attacker_prefix_caching']).lower()}) from the shared pool; PAIR_N_STREAMS=5 asserted per driver.
#
# serving target: {t['model']} @ {t['model_revision']}

# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---
explicit_cells:
"""
    for cid in universe:
        header += f"- {cid}\n"
    if xsub:
        header += f"# the over-refusal control's XSTest half (cell_controls_run.XSTEST_SUBSET_KEY; roster targets.{target}.xstest_subset)\n"
        header += f"xstest_subset: {xsub}\n"
    header += "\n"
    _refuse_bon_drift(target, view, header + body, body_text, roster=roster)
    return header + body


# --------------------------------------------------------------------------- manifests

def render_manifest(roster: dict, target: str, view: str, *, roster_path: Path = ROSTER) -> str:
    rrel = roster_rel(roster_path)
    cert_note = _cert_roster_note(roster_path)
    t = roster["targets"][target]
    v = t["views"][view]
    ms = MODEL_SERVING[target] if target in MODEL_SERVING else front_door_serving(target)
    tag = target + ("" if view == "main" else f"_{view}")
    cells = [harness_cell_id(c) for c in view_cell_universe(roster, target, view)]   # s3r-fix: no unconditional `none` (a steered-wave view lists none)
    flags = "\n".join(f"    - {_yaml_flag(f)}" for f in ms["flags"])
    cell_lines = "\n".join(f"    - {c}" for c in cells)
    prompted = [c for c in cells if c.startswith("prompted__")]
    # the historical three always; any other SI's hash only when this manifest serves a
    # prompted cell over it (PREFIX_HASHES_ALWAYS: the hash mapping is a harness-hash input)
    in_play = [c[len("prompted__"):] for c in prompted]
    emit = [k for k in PREFIX_HASHES if k in PREFIX_HASHES_ALWAYS or k in in_play]
    hashes = "\n".join(f"    {k}: {PREFIX_HASHES[k]}" for k in emit)
    prompted_note = (f"verified per request (H8) for the prompted cell(s) {', '.join(prompted)}"
                     if prompted else "no prompted cell in this wave")
    return f"""# Petri Bloom misalignment audit — full22 wave, target {target}, view `{view}`
# (sdf-base-steering-r2). GENERATED by scripts/sdfbase/make_full22_configs.py from
# {rrel} — never hand-edited (byte-identity is tested). The cells are
# EXACTLY the harness ids of {Path(v['config']).name}'s explicit_cells (scripts/sdfbase/
# mt_serving_up.py refuses a manifest/config cell mismatch): TrainTest `<si>__<arm>` <-> harness
# `si_<si>__<arm>` through registry_adapter (constv2-* are OBJECT-ONLY pseudo-instructions —
# never a prompted leg; safety_refusal <-> safety). The path-matched no-op control leg
# (`none__lora0` on a native standup, `none__lens0` on a lens one) is DERIVED by
# defense_cells.enumerate_cells, never listed.
{cert_note}#
# SAME audit world as the completed gemma-4-E4B-it two-SI audit and the mt1 wave: the frozen
# scenario bundle {BUNDLE} (six behaviors x 10 scenarios), auditor claude-sonnet-5 with the
# fallback chain (grok-4.6, then the abliterated Qwen3.6-27B; auditor_refused sidecars +
# auditor_chain provenance), judge claude-opus-5, the verbatim petri-bloom 0.2.6 prompt bytes,
# bloom_options unchanged. Every leg of every target audits identical scenarios.
#
# Model serving: {ms['note']}.
#
# Concurrency: the grading host driver (scripts/sdfbase/full22/petri_drive_full22.sh) runs every leg
# with PETRI_EVAL_MAX_CONNECTIONS=1 so the TOTAL of concurrent auditor conversations across all
# legs of the wave stays <= 50 (48 legs planned; concurrency is not a harness-hash input).
petri:
  audit: petri_full22_{tag}
  behaviors:
{chr(10).join(f'    - {b}' for b in BEHAVIORS)}
  cells:
{cell_lines}
  scenario_bundle_hash: {BUNDLE}   # durable 0700 store: $SHARED/petri/scenarios/{BUNDLE} (cluster A); the grading host copy $STATE/petri/scenarios/{BUNDLE}
  auditor_model: anthropic/claude-sonnet-5
{_petri_auditor_fallback_block(roster, target, view)}{_petri_judge_fallback_block(roster, target, view)}  judge_model: anthropic/claude-opus-5
{_petri_refusal_fallback_blocks(roster, target, view)}  scenarios_model: anthropic/claude-opus-4-8   # the frozen bundle's author (harness-hash input)
  prompt_provenance: override@c7fb238c0b17a812   # VERBATIM petri-bloom 0.2.6 auditor/judge bytes (configs/capeval/petri/prompts)
  target_model: {t['model']}
  target_revision: {t['model_revision']}
  serving_backend: proxy_vllm     # committed production primary (B1)
  serving_mode: interleaved       # exact_replay dropped 2026-08-27 (see petri_twosi.yaml)
  serving_packages:               # the serving-venv pins (configs/capeval/vllm-requirements.txt)
    vllm: 0.25.1+cu129
    vllm-lens: 1.2.1
    torch: 2.11.0+cu129
  serving_launch_flags:           # applied on BOTH engines: the native default (--serving-path lora_native) strips --enforce-eager and records it as manifest_flags_dropped; the LENS escape hatch (--serving-path lens) takes them verbatim; never --enable-lora here, the native composer owns it (docs/native_multilora_serving.md §1)
{flags}
  chat_template_hash: {ms['chat_template_hash']}   # sha256_hex16 of the pinned tokenizer's chat_template bytes; the standup re-hashes and refuses a mismatch
  tool_parser: {ms['tool_parser']}
  serving_impl_commit: {roster['serving_impl_commit']}   # the last commit touching capeval/serving/*.py + tiers/petri_runner.py on the wave tree; RE-PINNED at landing (README)
  prompted_prefix_hashes:         # sha256_hex16 over the EXACT SI text bytes (H8); {prompted_note}
{hashes}
{_petri_target_decode_block(roster, target, view)}  bloom_options:
    max_turns: 15                 # auditor turn budget per scenario (2026-08-18)
    enable_prefill: false         # defaults pinned OFF
    enable_rollback: false
    compaction: false
    realism_filter: false
    epochs: 1                     # ~10 scenarios x 1 epoch per behavior
    seed: 0                       # the ONE audit-wide evaluation seed
"""


def front_door_pins_from_registry(pins: dict, *, registry_root: "Path | str", roster: dict, models_table: "Path | str | None" = None,
                                  campaign: "str | None" = None) -> dict:
    """The registry-only rewrite `--derived-pins-from-registry` performs (the fit-your-own sequence's pins step, reproduce/README.md), as ONE
    function so a test can build a pins file in a temp dir through the generator: the front door's DERIVED entries from their registry rows
, the declared BASE-lane entries from their top-level fitted row, the derived parents held to the pinned parent ids,
    the REFUSAL-DIRECTION entries from the row the fit emitted (gap-refusal-pin, 2026-10-12), then the dose views' copies by the parent's identity
    at base_scale = dose. Every other entry byte-identical; a committed (paper) entry is held to its own row and kept verbatim.
    `campaign`: the ONE registry campaign every writer of the chain reads under — None resolves to the site's, as 04 resolves it
    (registry_campaign_default), resolved here once so the chain cannot read two campaigns."""
    campaign = campaign if campaign is not None else registry_campaign_default()
    out = derived_pins_from_registry(pins, registry_root=registry_root, models_table=models_table, campaign=campaign)
    out = base_pins_from_registry(out, registry_root=registry_root, models_table=models_table, campaign=campaign)
    check_derived_parents(out, registry_root=registry_root, models_table=models_table, campaign=campaign)
    out = arditi_pins_from_registry(out, registry_root=registry_root, models_table=models_table, campaign=campaign)
    return dose_pins_from_registry(out, registry_root=registry_root, roster=roster, models_table=models_table, campaign=campaign)


def outputs(roster: dict) -> list[tuple[Path, str, str, str]]:
    out = []
    for target, t in roster["targets"].items():
        for view, v in t["views"].items():
            out.append((REPO / v["config"], "config", target, view))
            out.append((REPO / v["manifest"], "manifest", target, view))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed file drifts")
    ap.add_argument("--pins-from", type=Path, default=None,
                    help="a clusters' staging_manifest.json: rewrite configs/sdfbase_full22_pins.json")
    ap.add_argument("--registry-root", type=Path, default=None,
                    help="the registry a FRONT-DOOR derived pin is read from (D1319; default SISTEER_REGISTRY_ROOT, else the registry's own default)")
    ap.add_argument("--allow-dropping-committed-pins", metavar="REASON", default=None,
                    help="S2RLFIX2 (D1980): PRUNE the committed entries this staging manifest does not carry, for this stated reason, printing each (default: keep them verbatim — the file aggregates several banks' pins)")
    ap.add_argument("--derived-pins-from-registry", action="store_true",
                    help="rewrite ONLY the front-door DERIVED entries of configs/sdfbase_full22_pins.json from the registry object rows (D1371), the BASE-lane "
                         "entries whose object declares an object_kind from their top-level fitted row (D1425), the REFUSAL-DIRECTION entries from the row the fit "
                         "emitted (04 --object refusal_direction), then hold a derived object's registered parents to the pinned parent ids (D1427) and the dose "
                         "views' copies to their parent's identity (D1642); every other entry byte-identical — the paper pins are never regenerated from the "
                         "clusters' bank manifest")
    ap.add_argument("--roster", type=Path, default=ROSTER,
                    help="the roster to render (default: the production configs/sdfbase_full22_roster.yaml; a "
                         "certification-only roster renders ITS views only, e.g. configs/sdfbase_full22_roster_psrcert.yaml)")
    ap.add_argument("--arditi-registered-ids", action="store_true",
                    help="04_fit_objects.sh's done-ness read for the refusal direction (gap-refusal-pin): print, one per line, the reference/approved "
                         "registry object id(s) of the SERVED checkpoint's refusal direction (--served-model-id, --campaign) AT the rung --coefficient, "
                         "classified by the pins writer's own row reading (arditi_row_rung over the identity's rows: no SQL rung predicate); no "
                         "line when none; two or more distinct objects at the identity and rung REFUSE by name (exit 1), as the pins writer does; "
                         "a registry not on disk refuses by name. Renders nothing")
    ap.add_argument("--served-model-id", default=None, help="with --arditi-registered-ids: the served checkpoint's registry model id (models.yaml serve.checkpoint.model_id — the row's fit_model_id)")
    ap.add_argument("--coefficient", type=float, default=None,
                    help="with --arditi-registered-ids: the rung (the cell's coefficient, which 04 reads through --arditi-cell); with --arditi-cell: a knob "
                         "(SISTEER_REFUSAL_COEFFICIENT) that may only RESTATE the cell's coefficient — a disagreement refuses by name")
    ap.add_argument("--campaign", default=None,
                    help="the registry campaign the object rows are read under — with --arditi-registered-ids (04's done-ness read), --pins-from and "
                         "--derived-pins-from-registry alike (every registry-backed pins writer: derived, declared base, refusal direction, dose copies, "
                         f"the parent check). Default: the site's SISTEER_REGISTRY_CAMPAIGN, else REGISTRY_CAMPAIGN, else {REGISTRY_CAMPAIGN} — "
                         "04_fit_objects.sh's own resolution (registry_campaign_default), so the pins are read where 04 registered the fits")
    ap.add_argument("--arditi-cell", action="store_true",
                    help="04_fit_objects.sh's identity read for the refusal direction (gap-refusal-pin; the review on ea634b165): print `<coefficient:g> <cell id>` of "
                         "--model-key's objects.refusal_direction cell in --models-table — the ONE identity 04 fits and publishes, the pins writer pins and 05 "
                         "stages — refusing by name (exit 1) a cell that is not the fitter's spelling of its coefficient (`ar_c0p30`, `ar_cm1p0`: pinned but never "
                         "staged), a row without the cell or with a non-arditi_bank one, and a --coefficient that does not restate the cell's. Renders nothing")
    ap.add_argument("--models-table", type=Path, default=None, help=f"with --arditi-cell: the models table (default $SISTEER_MODELS_TABLE, else reproduce/models.yaml: {MODELS_TABLE})")
    ap.add_argument("--model-key", default=None, help="with --arditi-cell: the table key whose refusal_direction cell is read")
    args = ap.parse_args(argv)
    # ONE campaign for every registry read of this run — named, else the site's, as 04 resolves it (registry_campaign_default)
    campaign = args.campaign if args.campaign is not None else registry_campaign_default()
    if args.arditi_cell:
        # a READ of the table, not a render: the one answer 04 acts on goes to stdout (`<c:g> <cell id>`), every refusal to stderr (exit 1)
        if not args.model_key:
            raise SystemExit("--arditi-cell needs --model-key <table key> (and --models-table <path>, default the environment's)")
        if args.arditi_registered_ids or args.pins_from is not None or args.derived_pins_from_registry or args.check:
            raise SystemExit("--arditi-cell is a table read on its own: it renders and writes nothing (drop --arditi-registered-ids / --pins-from / --derived-pins-from-registry / --check)")
        cell_id, c = arditi_declared_cell(args.models_table, args.model_key)
        if args.coefficient is not None and abs(float(args.coefficient) - c) >= 1e-9:
            raise SystemExit(f"SISTEER_REFUSAL_COEFFICIENT={args.coefficient:g} contradicts the models table: {args.model_key}'s objects.{ARDITI_TABLE_CELL} cell "
                             f"{cell_id!r} spells coefficient {c:g}, and the cell is the declaration 04 publishes under (<refusal root>/<served id>/{cell_id}), the pins "
                             f"writer pins (`arditi__{cell_id}`) and 05 stages — a fit at {args.coefficient:g} would register a row no pin is written from and 05 "
                             f"could never stage. Unset SISTEER_REFUSAL_COEFFICIENT (or set it to {c:g}); to fit another rung, change the cell to "
                             f"{arditi_fitter_staged_id(float(args.coefficient))!r} (the fitter's spelling) so every stage reads the same identity")
        print(f"{c:g} {cell_id}")
        return 0
    if args.arditi_registered_ids:
        # a READ, not a render: the one answer 04 acts on goes to stdout (ids, one per line; nothing when none), everything else to stderr
        if not args.served_model_id or args.coefficient is None:
            raise SystemExit("--arditi-registered-ids needs --served-model-id <served registry model id> and --coefficient <rung>")
        if args.pins_from is not None or args.derived_pins_from_registry or args.check:
            raise SystemExit("--arditi-registered-ids is a registry read on its own: it renders and writes nothing (drop --pins-from / --derived-pins-from-registry / --check)")
        root = args.registry_root if args.registry_root is not None else registry_root_default()
        ids, seen = arditi_registered_ids(root, fit_model_id=args.served_model_id, coefficient=args.coefficient, campaign=campaign)
        if seen:
            print(f"refusal direction of {args.served_model_id} at coefficient {args.coefficient:g} (campaign {campaign}): {len(seen)} other row(s) of the identity, "
                  "none this rung's reference/approved: " + ", ".join(seen), file=sys.stderr)
        for oid in ids:
            print(oid)
        return 0
    if not args.roster.is_file():
        raise SystemExit(f"no roster at {args.roster}")
    roster = load_roster(args.roster)
    if args.pins_from is not None and args.derived_pins_from_registry:
        raise SystemExit("--pins-from and --derived-pins-from-registry are exclusive: the manifest read rewrites the whole file, the registry read only the front door's derived entries")
    if args.pins_from is not None:
        pins = pins_from_staging_manifest(args.pins_from, registry_root=args.registry_root if args.registry_root is not None else registry_root_default(),
                                          committed=load_pins(), allow_drop=args.allow_dropping_committed_pins, roster=roster, campaign=campaign)   # S2RLFIX1: never drop a committed pin silently
        PINS.write_text(dump_pins_like_committed(pins, PINS), encoding="utf-8")   # S2RLFIX3: the committed file's own shape — a no-op rewrite is byte-identical
        print(f"wrote {PINS} ({sum(len(v) for v in pins.values())} pinned objects; registry reads under campaign {campaign})")
    if args.derived_pins_from_registry:
        root = args.registry_root if args.registry_root is not None else registry_root_default()
        pins = front_door_pins_from_registry(load_pins(), registry_root=root, roster=roster, campaign=campaign)
        PINS.write_text(dump_pins_like_committed(pins, PINS), encoding="utf-8")
        print(f"wrote {PINS} (front-door derived, declared base-lane and refusal-direction entries from the registry under campaign {campaign}; "
              f"{sum(len(v) for v in pins.values())} pinned objects)")
    pins = load_pins()
    rc = 0
    for dst, kind, target, view in outputs(roster):
        text = (render_config(roster, pins, target, view, roster_path=args.roster) if kind == "config"
                else render_manifest(roster, target, view, roster_path=args.roster))
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.write_text(text, encoding="utf-8"); print(f"wrote {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
