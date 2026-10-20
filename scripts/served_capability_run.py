"""Served-path capability controls (Phase 1): generate XSTest / OR-Bench
(and, since Phase 2, score MMLU letter-logprobs) against ONE already-standing
steered vLLM server, through its RoutingProxy, for ONE two-SI DefenseCell.

GSM8K (PW3, user instruction U2 2026-09-03; item set flipped by U4(a),
2026-09-04 — PW17): the bespoke 0-shot leg this driver used to score inline
(``s2parity/gsm8k_0shot``, the ``####`` parser) is RETIRED. ``--controls
gsm8k`` names the STANDARD instrument — inspect_evals' stock ``gsm8k``
restricted to the <=200 items matched by question text to the retired
instrument's Platinum rows, dispatched to ``inspect_served.run_inspect_tasks``
like every tinycap control (so ``--thinking`` is required; row task
``inspect/gsm8k_paired200``; greedy, max_tokens 512 under thinking-OFF). Its
``controls_summary.json`` slot payload names the instrument (``task``,
``item_set: paired200``). ``gsm8k_full`` is the IN-DEPTH full 1,319-item test
set (``inspect/gsm8k``), never the slot; the retained spelling
``inspect_cmp:gsm8k`` resolves to ``gsm8k``. A request for the bespoke leg by
its old names refuses with the pointer (``contracts.gsm8k_retired_message``).

This is the capability twin of the served jailbreak/BoN path: the SAME
proxy, the SAME routing table, the SAME per-cell cache_salt isolation and
steering-bites gates — so capability numbers can finally be filed from the
SAME serving the attack numbers came from, instead of the HF-only path
(``cell_controls_run.py``), whose stores this driver's layout mirrors.

Phases (mirroring the HF controls driver):

  generate  — runs on the GPU node against the live proxy. Fans the pinned
              row sets out over ``POST <proxy>/chat/completions`` (async
              aiohttp, bounded concurrency) and writes the judged controls'
              raw generations into the standard ``GenSpool`` files.
  (judging) — NOT implemented here. XSTest / OR-Bench spools are graded
              later by the EXISTING judged-tier grader
              (``sisteer.capeval.tiers.judged.judge_xstest`` /
              ``judge_or_bench``) — the spool header, store layout, item
              ids, and manifest written here are exactly what that phase
              consumes.
  verify    — a CPU container.
              After ``generate --verification-phase generate`` left a
              BigCodeBench task GENERATED (its unscored ``.eval`` under the
              store, a generation sidecar + pending marker in the row dir),
              ``verify`` scores that log in owned Modal sandboxes, BINDS the
              scores onto the GPU phase's log by ``(id, epoch)`` and files the
              task through the SAME tail the coupled path files through
              (``sisteer.capeval.tinycap.cpu_verify``).

THE LOAD-BEARING SUBTLETY (double-templating). The RoutingProxy exposes an
OpenAI-compatible chat endpoint that takes ``messages`` and applies the
chat template SERVER-SIDE. The HF runners (``judged.generate``,
``parity._run_gsm8k``) render prompts CLIENT-side via ``build_chat_prompt``
— reusing them against the proxy would wrap an already-rendered chat string
in a second template. So this driver sends the RAW user text as the user
message and never calls ``build_chat_prompt``: the judged controls' user
text is the pinned row's ``prompt``, verbatim. A compiled (psr/lens)
cell sends NO system turn — the object replaces steering, and the proxy
refuses an unexpected registered prefix on a non-prompted vmodel (H8).

POSITIVE CONTROL (must be able to FAIL). Before any generation, the driver
proves the route actually steers: ``assert_cell_salt_isolation`` over the
FULL route set, then ``assert_steering_bites`` with a steered-vs-unsteered
greedy probe through the SAME proxy against the path-matched scale-0 twin
(``<cell>__lens0`` / ``none__lens0``). This check was RED during the real
mask bug; ``intervention_hash``-of-spec and ``effective_scale`` stayed
green through it and are deliberately NOT used as controls here. The result
(verified yes/no + the per-arm divergence diagnostic) is recorded in
``evidence.json`` alongside the served engine stamp (decode_mechanism
``position_mask``, top-level ``mask="generated"``), the served response
header ids, and — when ``--vllm-log`` is given — the server's prefix-cache
hit rate (expected ~0% on a steered lens leg BY DESIGN: the proxy salts
those requests per-request so a steered prefill can never resume from
cache).

MMLU, SERVED (Phase 2 — the served-loglikelihood landing). ``mmlu_letter``
(s2parity) is not a generation task; on the HF path a response-mask arm
scores it via ``parity._response_letter_nlls``: ONE prefill of
``build_chat_prompt + "Answer:"`` inside the arm ctx with the
response boundary re-pointed at the LOCATED header start, reading all
four letter logprobs from the FINAL position's next-token distribution
(one softmax denominator == the S2 argmin rule). The served analogue is
that exact forward, request-shaped: one chat request per item whose final
message is the assistant SEED ``"Answer:"`` (``continue_final_message``,
``add_generation_prompt: false``), ``max_tokens: 1``, greedy, with
``logprobs`` + ``top_logprobs`` + ``return_tokens_as_token_ids`` — the
four letter logprobs are read off the first generated position's raw
top-K (the pinned engine's ``logprobs_mode`` default is raw_logprobs =
log-softmax of the raw logits, the same quantity HF reads). A letter id
absent from the returned top-K is CENSORED, never guessed: it sits below
every present letter, so the pick over the present letters is exact and
the censorship is recorded per item and counted per cell
(``_run_mmlu_served``); only an item with NO letter in the top-K is a
PER-ITEM hard error (nothing to rank — collapse, not a read-width knob),
and even that item is RECORDED, never dropped: a ``topk_undeterminable``
row files the letters, their ids, the window and the smallest returned
logprob (the K-th largest, an upper bound on every absent option), so it
is a recoverable bracket rather than a gap whose evidence died with the
exception. Every MMLU row names its case in ``extra["letter_read"]``.

THE MASK-COVERAGE GUARANTEE (the crux this phase exists to close): a
masked steer only affects logprobs INSIDE the mask, and the serving-side
response mask used to require the prompt to END at the assistant header
— a seeded render refused (fatally, in-forward). The request therefore
DECLARES the seed via the proxy's ``petri_response_prefill`` gate
(``proxy.RESPONSE_PREFILL_KEY``): the proxy verifies the declaration
matches the request shape and threads it as a selector fact into
``HookContext.response_prefill``, and ``lens_adapters._response_start``
then locates the boundary at the LAST header occurrence — header + seed
+ generated all masked, ``parity._response_boundary``'s exact semantics —
so the scored distribution's predictor row is inside the mask BY
CONSTRUCTION whenever the request is served at all (a mismatch refuses;
nothing falls back to width arithmetic). The declaration changes only
where a response op's boundary is located within one declared request;
the position MODE each op serves is baked in the worker registry, so the
fit/serve position-regime guard is untouched.

THE LOGLIKELIHOOD POSITIVE CONTROL (must be able to fail, and does):
before any MMLU request is scored, ``_logprob_bites_evidence`` reads the
letter logprobs for probe items through the SAME seeded scoring path from
the steered vmodel, its scale-0 twin, AND the twin a second time — the
repeat measures the engine's own numeric noise so that "the logprobs
differ" cannot be satisfied by noise alone (the text-diff
hypersensitivity finding, inverted). ``fidelity.check_logprob_steering_
bites`` refuses bit-identical steered-vs-twin logprobs (the
silently-unsteered failure) AND a divergence that does not clear
``max(abs_floor, min_ratio x measured noise ceiling)``. As everywhere in
this lane, ``intervention_hash`` and ``effective_scale`` are deliberately
NOT controls — both stayed green through the real mask bug.

ENDPOINT BINDING AND SERVED IDENTITY (tinycap runs; Evals E10 N2 / N3).
The standup's ``endpoint.json`` beside ``--routing-table`` is the record the
manifest copies its served facts from — so before anything is read off it,
the record is BOUND to the endpoints this run was pointed at:
``endpoint.proxy_url`` must equal ``--proxy-url`` and
``endpoint.routing_table`` must resolve to ``--routing-table``; with a
``--ledger-url`` the ledger's own record (``ledger_endpoint.json``, the
ledger's ``--endpoint-json``, beside the table) must name that URL and an
``upstream`` equal to the standup's proxy. Two live standups with the same
virtual-model names on different loopback ports pass ``/v1/models``; this
binding is what refuses to file standup B's numbers under standup A's
manifest. What the standup SERVED that changes outputs — the composed
reasoning parser, the tool-call mode, the WINDOW ``max_model_len``, the
served S ``max_num_seqs`` and the client's request width ``--concurrency``
(PW5-4: the served-path owner's 2026-09-04 re-check ruled S and the client
width identity — under bf16 a greedy argmax at a near-tie moves with the
batch composition, and a resume at another S would mix two scheduling
regimes in one store) — is hashed into the store identity
(``resolved_config.tinycap.served``) and compared against an existing cell
manifest before any resume skip. So are the FRAMEWORKS (PW5-2): every
requested framework's imported version must equal its client-lock pin
(``contracts.FRAMEWORK_PINS``; ``FrameworkPinMismatch``
refuses before any store is touched), and the frameworks block — versions +
``client_lock_sha256`` — is hashed (``resolved_config.tinycap.frameworks``),
so a dependency change selects a new store rather than resuming an old one's
markers. The manifest also names its PRODUCER tree (``producer.tree_sha``,
read from the ``TREE_SHA`` file the launcher writes at upload, else the git
HEAD of a checkout) so the later EvalPlus merge can require the same tree.
Every tinycap runner then verifies the proxy's attestation stamps on every
response it retains (``proxy_engine.assert_response_stamps``).

THE SAMPLED RE-RUN + THE CAP LADDER (scripts/sdfbase/full22/sampled/
DESIGN.md; merged onto main 2026-09-11 as the CANONICAL over-refusal
instrument). A config with a top-level ``sampled_decode`` block applying to
``controls`` turns ``controls_gen_config`` into the vendor preset's gen
(do_sample, temperature, top_p, [top_k], [min_p], [presence_penalty],
[repetition_penalty], arm, seed_base, trial), which re-keys the store; each
judged draw is its own request carrying exactly those keys plus a
cell-independent per-item ``sampling_seed`` (``sampling_seed``), and a
sidecar ``private/gens_<name>.meta.jsonl`` records per item its seed, cap,
finish_reason, token counts and at-cap bit (ids and numbers only; ``GenMeta``).
``--controls-cap N --escalate-from-root ROOT`` runs one RUNG: the same cell's
completed store one rung below is located under ROOT (never written), every
item it left at cap is re-drawn at N with the same seed (plus a
cell-independent ``--floor-sample`` of below-cap items, the re-draw noise
floor), every other item's text is copied forward unchanged, and the result
is a complete standard store under --out-root with its own identity
(``escalated_from`` in the gen) that ``served_capability_judge.py`` reads
unchanged. Greedy configs write byte-identical files to before: no sidecar,
no extra gen_stats keys, the same request bodies. The sampled lineage's
bespoke GSM8K readout was NOT ported: ``gsm8k`` on this driver is the
standard Inspect paired-200 leg (PW3) and the paper uses that one instrument.

Example (GPU node, proxy already standing)::

  PYTHONPATH=src python scripts/served_capability_run.py generate \\
    --proxy-url http://127.0.0.1:8814/v1 \\
    --routing-table /path/to/routing_table.json \\
    --cell safety_refusal__psr_kl_all \\
    --config configs/two_si.yaml \\
    --defenses-dir $SHARED/exp/jailbreak/two_si_defenses/cells \\
    --out-root out/two_si/served_controls \\
    --controls mmlu,xstest,or_bench_hard_1k --limit 100 --concurrency 24

  # the standard GSM8K (a tinycap control: --thinking is required)
  PYTHONPATH=src python scripts/served_capability_run.py generate ... \\
    --controls gsm8k --thinking off
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import copy
import functools
import contextvars
import hashlib
import json
import uuid
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import cell_controls_run as ccr  # the HF controls driver: conventions reused
import jailbreak_run  # _load_cfg — the lane's one config loader
import unified_sweep_run as usr  # _served_route_for / _served_probe_fn

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving.fidelity import FidelityError
from sisteer.capeval.serving import geometry as sg
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sdfbase' / 'full22'))
import low_dose_liveness_policy as _low_dose_liveness  # noqa: E402  (the low-dose liveness seam; scripts/, never src/serving — serving_impl_commit unmoved)
from sisteer.capeval.serving.geometry import geometry_for
from sisteer.capeval.serving.proxy import (ProxyClientEngine,
                                           RESPONSE_PREFILL_KEY,
                                           _canonical_v1_base,
                                           load_routing_table,
                                           proxy_client_token)
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as tc
from sisteer.capeval.tinycap import cpu_verify as _CV   # landing 2: the two-phase BigCodeBench contract's files + verify
from sisteer.eval import sweep as sweep_mod
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.cell_controls import (SAMPLED_DECODE_SAMPLER_KEYS,
                                        SampledDecodeError,
                                        controls_gen_config)
from sisteer.eval import sampled_controls as scp   # the sampled-controls machinery: ONE home, two drivers
from sisteer.eval import controls_events as cev    # the sampled controls' durable per-draw stream (seedrep §7.4)
# every name of the sampled re-run's machinery this driver uses, or a test reaches through it,
# imported by name from its module — the driver defines none of them (2026-09-12 extraction;
# tests/test_sampled_controls_module.py pins that the driver's names ARE the module's objects)
from sisteer.eval.sampled_controls import (  # noqa: F401
    CONTROLS_CAP_LADDER, FLOOR_SAMPLE_DEFAULT, FLOOR_SAMPLE_RULE, META_SUFFIX,
    META_ROW_KEYS, META_REGEN_KEYS, META_COPIED_KEYS, META_EXCLUDED_KEY, META_EXCLUSIONS,
    sampling_seed, floor_rank_key, floor_sample_ids, _decode_of, _decode_block, _text_sha,
    GenMeta, _first_divergence, _fork_fields, _meta_stats, Escalation, SourceControl,
    _ladder_decide, _read_items_files, _apply_items, _locate_source_store, _decoder_of,
    _apply_escalation, _SOURCE_IDENTITY_KEYS, _validate_source_identity, _read_source_spool,
    _load_source_control, _run_judged_served)

REPO = Path(__file__).resolve().parents[1]

#: The LEGACY controls this driver serves itself (mmlu landed with Phase 2
#: — module docstring). ``gsm8k`` LEFT this tuple with PW3: the bespoke
#: ``s2parity/gsm8k_0shot`` leg is retired and ``gsm8k`` is now the tinycap
#: control ``contracts.CONTROL_GSM8K`` (inspect_evals' stock task, dispatched
#: to inspect_served like math500 / bfcl).
SERVED_CONTROLS = ("mmlu", "xstest", "or_bench_hard_1k")
#: The ``resolved_config["controls"]`` value — FROZEN at the historical
#: ``sorted(SERVED_CONTROLS)`` so every existing served store (the scale-12
#: and qwen capability waves) keeps its ``harness_config_hash``: the list is
#: store IDENTITY (it says "this is the served driver", never which controls
#: a run requested — the manifest's ``controls`` does that), and dropping
#: ``gsm8k`` from it would fork every served store's key for no new
#: information. A tinycap run keys apart anyway via ``resolved["tinycap"]``.
RESOLVED_CONTROLS_IDENTITY = ("gsm8k", "mmlu", "or_bench_hard_1k", "xstest")
#: Spellings of the RETIRED bespoke served leg; each refuses with the pointer.
RETIRED_GSM8K_TOKENS = ("gsm8k_0shot", tc.GSM8K_RETIRED_TASK, "gsm8k_legacy", "gsm8k_bespoke")

#: The served capability geometry (sisteer.capeval.serving.geometry, the
#: per-eval table): the client's default fan-out is ITS max_num_seqs, so a
#: driver never fans out wider than the server the table launches (the old
#: literal 24 was a width copied in by hand).
CAPABILITY_GEOMETRY = geometry_for("capability")

# --------------------------------------------------------------------------- #
# the sampled re-run (scripts/sdfbase/full22/sampled/DESIGN.md)                #
# --------------------------------------------------------------------------- #

#: The judged controls the sampled re-run's controls preset covers (DESIGN.md
#: "Scope"): the sampler in ``sampled_decode`` is THEIR preset at THEIR cap.
SAMPLED_CONTROLS = ("xstest", "or_bench_hard_1k")
#: mmlu under a sampled config runs UNCHANGED (LADDER_RULES.md): it is a
#: teacher-forced letter-logprob argmax over ``top_logprobs`` — ONE greedy
#: token, temperature 0 — so no sampler key can move it; it is recorded as
#: unaffected in gen_stats, never sampled, and has no cap ladder.
SAMPLED_UNAFFECTED_CONTROLS = ("mmlu",)

#: The MMLU instrument's assistant seed — the exact string
#: ``parity._mmlu_prompt`` appends after the chat render, pinned equal by
#: tests/test_served_capability.py so the served render and the HF
#: instrument's can never drift apart.
MMLU_ANSWER_SEED = "Answer:"
#: decode-recording (the registry owner, 2026-10-13): the ONE sampler object the legacy MMLU scoring request spreads into its body
#: (``_mmlu_scoring_payload``) AND the manifest states under ``tc.MANIFEST_KEY_REQUEST_SAMPLERS["mmlu"]`` -- one object, so the
#: manifest and the request cannot be two numbers.
MMLU_SCORING_SAMPLER = {"temperature": 0.0}
#: decode-recording (Major 3): the ONE sampler object the GREEDY judged-control request spreads into its body
#: (``_chat_payload`` with no ``decode`` -- xstest / or_bench_hard_1k under the ``--greedy`` opt-in; the body bytes are those of every
#: greedy store ever drawn) AND the manifest states under ``request_samplers[<control>]`` for such a control. The greedy request DOES
#: name a sampler key (``temperature: 0.0``), so the manifest states it; ``{}`` attested a request shape the body never had.
GREEDY_CHAT_SAMPLER = {"temperature": 0.0}
#: top-K letter-logprob read width. The capctrl standup launches the
#: server with ``--max-logprobs`` at least this (the engine default is
#: 20); a letter outside the returned top-K is censored and recorded (the
#: pick over the present letters is exact); an item with NO letter inside
#: it is a per-item hard error that still files its bound
#: (``MMLU_UNDETERMINABLE_METRIC``). Not a knob: a letter below the top-64
#: is the model off the letter vocabulary, and widening the read would
#: only hide that.
MMLU_TOP_LOGPROBS = 64
#: the per-item row filed for an MMLU item whose EVERY letter fell outside
#: the returned top-K (value 1.0; ``extra`` carries the letters, ids, the
#: window and the returned-logprob bound — module docstring). Not an "acc"
#: row, so ``parity._done_acc`` leaves the item un-done and a later run
#: re-attempts it; readers pairing stores filter on metric "acc".
MMLU_UNDETERMINABLE_METRIC = "topk_undeterminable"
#: the per-item case label on every MMLU row's ``extra`` (the analysis
#: tells a point from a bracket by it): all four letters inside the top-K;
#: some outside (scored from the present ones, the pick mapped back to its
#: option, the window recorded under ``incomplete_topk``); every letter
#: outside (the ``MMLU_UNDETERMINABLE_METRIC`` row). A legacy row without
#: the key is all-present by construction: no scorer before the key ever
#: filed a censored row without ``censored_letters``.
MMLU_LETTER_READ_KEY = "letter_read"
MMLU_LETTER_READ_ALL_PRESENT = "all_present"
MMLU_LETTER_READ_PARTIAL = "partial_present"
MMLU_LETTER_READ_ALL_ABSENT = "all_absent"
#: loglikelihood positive-control geometry (module docstring); the probe
#: count is the contracts'.
LOGPROB_BITES_PROBES = tc.BITES_PROBE_ITEMS
LOGPROB_BITES_MIN_RATIO = 2.0
LOGPROB_BITES_ABS_FLOOR = 1e-3
#: THE LOGPROB LIVENESS GATE of a GENERATION cell's positive control
#: (contracts.OOS_EVIDENCE_BITES_LIVENESS_GATE says why). Under
#: ``--lens-liveness logprob`` a cell of kind ``alphasteer`` / ``static_r`` (contracts.LOGPROB_LIVENESS_KINDS) is ruled
#: live or dead by ``fidelity.check_logprob_steering_bites`` — the ladder's own instrument — over the FIRST generated
#: token's top-K logprobs on the same five bites prompts (``unified_sweep_run._SERVED_PROBE_PROMPTS``), three legs per
#: prompt (steered / scale-0 twin / the twin AGAIN: the measured noise floor), compared on the COMMON top-K support of the
#: three legs (the AlphaSteer rung records' ``liveness`` block reads exactly this: top-20, ``n_common`` per probe). The
#: text bites statistic still runs and is REPORTED beside it. Constants:
#:   top-K 20 = vLLM's default ``--max-logprobs`` (the MATH-500 standup passes none: STANDUP_POLICY ``max_logprobs=none``),
#:   so a wider read would be refused by the server; the rung records used 20 as well.
#:   min common tokens 2 = the instrument's own minimum (``check_logprob_steering_bites`` refuses a <2-vector); a probe
#:   whose three legs share fewer top-K ids is INCONCLUSIVE (recorded with its support sizes and whether the supports are
#:   disjoint — itself the shape of a large edit — never fed to the statistic as a guess).
#:   min conclusive probes 3 of 5 = the rung records' rule (``min_conclusive_probes`` = half the probe set, 4 of 8); fewer
#:   conclusive probes is an UNEVALUABLE control, refused by name (fail closed), never a text-only pass.
LOGPROB_LIVENESS_TOP_LOGPROBS = 20
LOGPROB_LIVENESS_MIN_COMMON_TOKENS = 2
LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES = 3

#: aiohttp total timeout (the served clients' one constant, proxy_engine's).
#: The 600 s client default is a known bite: at high server load a queued
#: request can sit far longer than its own decode.
AIOHTTP_TOTAL_TIMEOUT_S = pe.AIOHTTP_TOTAL_TIMEOUT_S

DRIVER_VERSION = "served_capability_run_v2"


def parse_controls(spec: str) -> list[str]:
    """Validate the --controls csv. The LEGACY served set (``SERVED_CONTROLS``,
    mmlu included since Phase 2 — module docstring) is checked FIRST and is
    untouched; the RETIRED bespoke GSM8K's spellings (``RETIRED_GSM8K_TOKENS``) refuse with
    the pointer at the standard. Any other token must be a tinycap control
    (``contracts.expand_control``: ``math500`` / ``humaneval_plus`` / ``mbpp_plus`` /
    ``gsm8k`` (the standard paired GSM8K, PW3 / U4(a)) /
    ``gsm8k_full`` (the in-depth full set) / ``inspect_cmp[:<task>]``) and is
    expanded to its canonical atoms. Order preserved, duplicates dropped."""
    tokens = [c.strip() for c in spec.split(",") if c.strip()]
    if not tokens:
        raise SystemExit("[served-cap] --controls is empty")
    out: list[str] = []
    for c in tokens:
        if c in RETIRED_GSM8K_TOKENS:
            raise SystemExit(
                f"[served-cap] --controls {c} refused: "
                + tc.gsm8k_retired_message("the served driver's inline-scored leg"))
        if c in SERVED_CONTROLS:
            atoms = (c,)
        else:
            try:
                atoms = tc.expand_control(c)
            except tc.ControlNameError as e:
                raise SystemExit(
                    f"[served-cap] unknown control {c!r}; served controls: "
                    f"{list(SERVED_CONTROLS)}; tinycap controls: "
                    f"{list(tc.TINYCAP_CONTROLS)} ({e})") from None
        for a in atoms:
            if a not in out:
                out.append(a)
    return out


def tinycap_atoms(controls: list[str]) -> list[str]:
    """The tinycap atoms among parsed controls — everything that is not a
    legacy served control."""
    return [c for c in controls if c not in SERVED_CONTROLS]


#: The tinycap control FAMILIES this driver dispatches (fix wave W1 / R1:
#: every frozen family): ``math500`` / ``gsm8k`` / ``gsm8k_full`` / ``inspect_cmp`` / ``bfcl_200`` /
#: ``bigcodebench_200`` / ``bigcodebench_full`` / ``bfcl_full`` →
#: ``inspect_served.run_inspect_tasks``; ``humaneval_plus`` / ``mbpp_plus``
#: → ``evalplus_runner.run_served``. Pinned equal to EVERY frozen family by
#: the assert below (and by the tests); ``parse_controls`` already refuses
#: any name outside ``contracts.TINYCAP_CONTROLS`` through
#: ``expand_control``, so no runtime gate re-derives this.
TINYCAP_DISPATCHED = frozenset(tc.TINYCAP_CONTROLS)
assert set(tc.INSPECT_CONTROLS) | set(tc.EVALPLUS_CONTROLS) == TINYCAP_DISPATCHED


# --------------------------------------------------------------------------- #
# pinned-row seams (monkeypatchable in the offline test; real loaders here)    #
# --------------------------------------------------------------------------- #


def _mmlu_rows(pins, *, full: bool = False) -> list[dict]:
    """The MMLU rows this run scores: the s2parity PARITY_N-item head (the
    default, every historical store) or — under ``--mmlu-full`` — the whole
    pinned pool (contracts.MMLU_FULL_N). ONE resolver for the width
    (``contracts.mmlu_parity_n``), shared with the identity stamp
    (``contracts.mmlu_item_set``), so the rows that run and the item set the
    store claims cannot disagree. The full pool is the SAME seed-0
    permutation the head is cut from, so the head is its prefix item for
    item (parity.mmlu_parity_rows)."""
    from sisteer.capeval.tiers import parity

    return parity.mmlu_parity_rows(pins, n=tc.mmlu_parity_n(full=full))


def _xstest_rows(pins) -> list[dict]:
    from sisteer.capeval.tiers import judged

    return judged.xstest_rows(pins)


def _orbench_rows(pins) -> list[dict]:
    from sisteer.capeval.tiers import judged

    return judged.or_bench_rows(pins)


# --------------------------------------------------------------------------- #
# the chat surface: raw messages, templated SERVER-side                        #
# --------------------------------------------------------------------------- #


def _chat_payload(virtual_model: str, user_text: str, *, system,
                  max_tokens: int, decode: Optional[dict] = None,
                  seed: int = 0) -> dict:
    """One OpenAI chat request through the proxy: RAW user text (the proxy
    templates server-side — never a build_chat_prompt render), greedy and
    seed 0 by default — byte-identical to what every greedy store was drawn
    with. ``system`` is the SI text for a prompted/stacked cell (verified
    by the proxy against the registered prefix hash, H8) and None for
    compiled cells and the none baseline. ``cache_salt`` and the selector
    keys are proxy-owned and never sent (FORBIDDEN_CLIENT_KEYS).

    With ``decode`` (the sampled re-run, DESIGN.md) the body carries exactly
    the preset's keys — temperature, top_p, [top_k], [min_p],
    [presence_penalty], [repetition_penalty]; vLLM's chat API takes them
    top-level and the proxy forwards them — plus the per-item ``seed``
    (``sampling_seed``). Never ``n``: every draw is its own request."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user_text})
    if decode is None:
        # the greedy body: GREEDY_CHAT_SAMPLER's one key at the same position (the bytes unchanged; one object, so the manifest's
        # request_samplers statement and the request cannot be two numbers)
        return {"model": virtual_model, "messages": messages,
                "temperature": GREEDY_CHAT_SAMPLER["temperature"], "max_tokens": int(max_tokens), "seed": 0}
    body = {"model": virtual_model, "messages": messages}
    for k in SAMPLED_DECODE_SAMPLER_KEYS:
        if k in decode:
            body[k] = decode[k]
    body["max_tokens"] = int(max_tokens)
    body["seed"] = int(seed)
    return body


def _mmlu_user_text(row) -> str:
    """The MMLU user text — ``parity._mmlu_prompt``'s exact lines (its
    A-D letter zip truncates to 4 choices, matching the scoring side's
    ``_letters(min(k, 4))``), WITHOUT the chat render: the proxy templates
    server-side, and the assistant seed travels as its own message. The
    reconstruction ``build_chat_prompt + MMLU_ANSWER_SEED ==
    parity._mmlu_prompt(...)`` is pinned by test."""
    from sisteer.eval.capability_ext import _letters

    lines = [row["question"].strip()]
    for letter, choice in zip(_letters(4), row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)


def _mmlu_scoring_payload(virtual_model: str, user_text: str, *, system,
                          top_logprobs: int, response_prefill: bool) -> dict:
    """One teacher-forced letter-scoring request (module docstring): the
    assistant SEED as a continued final message, ONE greedy token, raw
    top-K logprobs keyed by token id. ``response_prefill`` sets the
    proxy's declaration key — required on a lens route (the proxy refuses
    an undeclared continued message there) and FORBIDDEN elsewhere (the
    proxy refuses the declaration on a maskless route)."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": MMLU_ANSWER_SEED})
    body = {"model": virtual_model, "messages": messages,
            "temperature": MMLU_SCORING_SAMPLER["temperature"], "max_tokens": 1, "seed": 0,
            "continue_final_message": True, "add_generation_prompt": False,
            "logprobs": True, "top_logprobs": int(top_logprobs),
            "return_tokens_as_token_ids": True}
    if response_prefill:
        body[RESPONSE_PREFILL_KEY] = True
    return body


def request_samplers_block(controls, gen: dict) -> dict:
    """``manifest[tc.MANIFEST_KEY_REQUEST_SAMPLERS]`` (decode-recording, 2026-10-13): keyed by the PRODUCER spelling of every legacy
    control this run was asked for, holding exactly the sampler keys that control's request body names -- ``mmlu`` the scoring request's
    :data:`MMLU_SCORING_SAMPLER` (the object the body spreads), a sampled over-refusal control ``_decode_of(gen)``, and under the
    ``--greedy`` opt-in :data:`GREEDY_CHAT_SAMPLER` (the object ``_chat_payload``'s greedy body spreads: the request DOES send
    ``temperature: 0.0``). PRESENT on every manifest of this driver (empty when no legacy control ran: the
    tinycap atoms' request samplers are the standup recorder's evidence, ``resolved_sampler.json``). Top-level and UNHASHED, never under
    ``resolved_config`` -- see ``tc.MANIFEST_KEY_REQUEST_SAMPLERS``."""
    out: dict = {}
    for c in controls:
        if c == "mmlu":
            out[c] = dict(MMLU_SCORING_SAMPLER)
        elif c in SAMPLED_CONTROLS:
            decode = _decode_of(gen)
            out[c] = dict(decode) if decode is not None else dict(GREEDY_CHAT_SAMPLER)
    return out


def _letter_logprobs_from_choice(resp, letter_ids, *, what: str):
    """The per-letter LOGPROBS at the scored position, read from one
    scoring response's first generated token. Returns
    ``(logprobs, missing_ids)``. THE SHAPE, which the scorer's index
    mapping depends on: ``logprobs`` is COMPACTED — one entry per letter
    id that IS in the returned top-K, in ``letter_ids`` order, so
    ``len(logprobs) == len(letter_ids) - len(missing_ids)`` and it is
    aligned with ``letter_ids`` ONLY when nothing is missing;
    ``missing_ids`` holds the absent letters' token IDS (not positions),
    in ``letter_ids`` order. A caller must map a position in ``logprobs``
    back through the present letter positions, never index ``letter_ids``
    by it directly (a missing letter ahead of the pick shifts it). Never a
    guessed value: the bites control refuses any missing letter (it must
    compare all four), the MMLU scorer ranks the present letters and
    records the censored ones. The first-position
    parse is ``proxy_engine.first_position_logprobs_by_id``: it refuses — ``ProxyEngineError``, a
    RuntimeError — a response without logprobs at that position, a token
    not in ``token_id:<int>`` form (``return_tokens_as_token_ids`` not
    honored; letters are never matched by surface string) and a
    non-finite logprob."""
    by_id = pe.first_position_logprobs_by_id(resp, what=what)
    out, missing = [], []
    for lid in letter_ids:
        if int(lid) in by_id:
            out.append(by_id[int(lid)])
        else:
            missing.append(int(lid))
    return out, missing


def _first_position_window(resp, *, what: str):
    """Everything one scoring response returned at the first generated
    position, ``({token_id: logprob}, sampled_token_id)``: the map is the
    shared ``pe.first_position_logprobs_by_id`` read (the top-K entries
    plus the sampled token, shape- and form-checked — the same entries the
    letter read consulted), and the sampled token's id is read off that
    same, already-checked entry. The payload is greedy, so the sampled
    token is the argmax and sits inside the top-K: the returned set IS the
    K largest logprobs, and its minimum bounds every absent token from
    above."""
    by_id = pe.first_position_logprobs_by_id(resp, what=what)
    token = resp["choices"][0]["logprobs"]["content"][0]["token"]
    return by_id, int(token[len(pe.TOKEN_ID_PREFIX):])


def _mmlu_window_record(resp, *, letters, letter_ids, missing, top_logprobs,
                        what: str) -> dict:
    """The record filed for an MMLU item with >= 1 letter outside the
    returned top-K (``extra["incomplete_topk"]`` on a scored row; the body
    of a ``MMLU_UNDETERMINABLE_METRIC`` row): the absent letters and their
    ids, the requested width, how many entries came back, the smallest
    returned logprob — the K-th largest, which bounds EVERY absent option
    from above — the largest, and the sampled token. The inputs a later
    reading needs to place the item; the same keys the AlphaSteer path's
    copy of this driver files, so one reader serves both stores."""
    returned, sampled_id = _first_position_window(resp, what=what)
    absent = {int(m) for m in missing}
    return {"missing_letters": [letters[i] for i, lid in enumerate(letter_ids)
                                if int(lid) in absent],
            "missing_ids": [int(m) for m in missing],
            "top_logprobs": int(top_logprobs),
            "n_returned": len(returned),
            "returned_min_logprob": min(returned.values()),
            "returned_max_logprob": max(returned.values()),
            "sampled_token_id": sampled_id}


@contextlib.asynccontextmanager
async def _proxy_endpoint(proxy_url: str, path: str):
    """An async ``payload -> response dict`` poster to ONE proxy endpoint
    (``<canonical /v1 base>/<path>``) over one aiohttp session with the long
    total timeout — the shape both :func:`_proxy_poster` (chat completions,
    modelled on the BoN throughput sweep's aiohttp poster) and
    a ``/v1/tokenize`` poster yield. The optional ``headers`` are a leg's
    ledger headers (P27), forwarded verbatim so the usage ledger can
    attribute the row; the legacy legs pass none. A non-200 raises (no
    client retry). Yielded as a callable so the offline tests can replace
    either name wholesale with a canned poster."""
    import aiohttp

    url = f"{_canonical_v1_base(proxy_url)}/{path}"
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_TOTAL_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def post(req, headers=None):
            async with session.post(url, json=req, headers=headers) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"proxy {url} returned {r.status}: {body[:400]}")
                return json.loads(body)

        yield post


def _proxy_poster(proxy_url: str):
    """:func:`_proxy_endpoint` on ``/v1/chat/completions``."""
    return _proxy_endpoint(proxy_url, "chat/completions")


def _seedrep_rung_redraw():
    """``scripts/sdfbase/seedrep/rung_redraw.py`` by path, loaded ONLY when ``--from-materialised``
    is given — the streaming ladder's assembly seam; nothing here changes without the flag."""
    import importlib.util as _ilu
    import sys as _sys
    mod = _sys.modules.get("seedrep_rung_redraw")
    if mod is None:
        spec = _ilu.spec_from_file_location("seedrep_rung_redraw", REPO / "scripts" / "sdfbase" / "seedrep" / "rung_redraw.py")
        mod = _sys.modules["seedrep_rung_redraw"] = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def _import_version(module: str, dist: "str | None" = None) -> str:
    """The version of a framework a REQUESTED control family runs on, read
    from the actually imported module (``__version__``, else the
    distribution metadata). A family whose framework cannot be imported
    cannot run, so this refuses loudly rather than recording null (R1: the
    manifest's frameworks block is None only for a family not requested)."""
    import importlib
    import importlib.metadata as md

    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise SystemExit(
            f"[served-cap] framework {module!r} is required by a requested control "
            f"but cannot be imported ({type(e).__name__}: {e}) — install the client "
            f"lock ({tc.CLIENT_LOCK_TXT}) into this environment") from None
    v = getattr(mod, "__version__", None)
    if not v:
        try:
            v = md.version(dist or module)
        except md.PackageNotFoundError:
            raise SystemExit(
                f"[served-cap] framework {module!r} imported but carries no version "
                f"(no __version__, no distribution {dist or module!r}) — refusing to "
                "record a frameworks block that cannot name what ran") from None
    return str(v)


def _client_lock_sha256(repo: Path = REPO) -> "str | None":
    """sha256 of the compiled client lock (``contracts.CLIENT_LOCK_TXT``) in
    the tree this driver runs from; None when the tree carries none (a
    checkout that predates the lock — recorded as absent, never guessed)."""
    p = Path(repo) / tc.CLIENT_LOCK_TXT
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


#: frameworks-block key -> the module imported for it (``_import_version``'s
#: argument; the distribution name equals the module name for all three).
FRAMEWORK_MODULES: dict = {
    "inspect_ai": "inspect_ai", "inspect_evals": "inspect_evals",
    "evalplus": "evalplus"}
#: The pins the frameworks block is CHECKED against (PW5-2, re-review F2) —
#: the contracts' client-lock pins, aliased here so a test can stub version
#: discovery and the pins with ONE helper (never by skipping the check).
FRAMEWORK_PINS = tc.FRAMEWORK_PINS


class FrameworkPinMismatch(SystemExit):
    """A REQUESTED framework's imported version is not the client lock's pin
    (PW5-2 / F2).
    The pinned constants ``INSPECT_TASK_VERSION`` / ``EVALPLUS_TASK_VERSION``
    are stamped into every row, so an unpinned environment would label rows
    with a version that did not run; the driver refuses before any store is
    touched instead of recording an attestation nobody enforces."""


def _check_framework_pins(versions: dict) -> None:
    """Refuse (:class:`FrameworkPinMismatch`) unless every framework the
    block names (a requested one — None is "not requested") is at its
    ``FRAMEWORK_PINS`` version. Names every disagreement."""
    bad: list[str] = []
    for key, pin in FRAMEWORK_PINS.items():
        have = versions.get(key)
        if have is not None and have != pin:
            bad.append(f"{key}: imported {have!r}, the client lock pins {pin!r}")
    if bad:
        raise FrameworkPinMismatch(
            "[served-cap] framework pin mismatch — " + "; ".join(bad) + f". Every row of a "
            f"tinycap run is stamped with the PINNED framework version, so this environment "
            f"cannot produce rows that say what ran; install the client lock "
            f"({tc.CLIENT_LOCK_TXT}) into this environment (PW5-2)")


def _framework_versions(families) -> dict:
    """``manifest["frameworks"]`` (``contracts.FRAMEWORKS_BLOCK_FIELDS``): the
    version of every framework a requested family runs on, from the imported
    module; None ONLY for a family not requested; plus the client lock's
    sha256 when the tree carries it. Every named version is CHECKED against
    its pin (:func:`_check_framework_pins`) — the block is store identity
    (``resolved_config.tinycap.frameworks``, PW5-2), never attestation only."""
    fam = set(families)
    out: dict = {k: None for k in tc.FRAMEWORKS_BLOCK_FIELDS}
    requested: list[str] = []
    if fam & set(tc.INSPECT_CONTROLS):
        requested += ["inspect_ai", "inspect_evals"]
    if fam & set(tc.EVALPLUS_CONTROLS):
        requested += ["evalplus"]
    for key in requested:
        out[key] = _import_version(FRAMEWORK_MODULES[key])
    _check_framework_pins(out)
    out["client_lock_sha256"] = _client_lock_sha256()
    return out


def _producer_block(repo: "Path | None" = None) -> dict:
    """``manifest["producer"]`` (``contracts.PRODUCER_BLOCK_FIELDS``; PW5-5,
    Evals item 3): the tree this driver ran from, so the later EvalPlus
    merge can require the same tree. ``tree_sha`` is the content of
    ``<repo>/TREE_SHA`` (``contracts.TREE_SHA_FILENAME`` — the launcher
    writes it at upload; a container has no ``.git``); absent that, the git
    HEAD of a checkout; neither refuses naming the file — a store that
    cannot say which tree produced it is not mergeable later."""
    import subprocess

    root = Path(repo) if repo is not None else REPO
    stamp = root / tc.TREE_SHA_FILENAME
    if stamp.is_file():
        sha = stamp.read_text(encoding="utf-8").strip()
        if not sha or len(sha.split()) != 1:
            raise SystemExit(
                f"[served-cap] {stamp} does not hold a single sha token — refusing to "
                "record a producer tree the merge could not compare")
        return {"tree_sha": sha, "driver": tc.PRODUCER_DRIVER_SERVED}
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        r = None
        why = f"{type(e).__name__}: {e}"
    else:
        why = r.stderr.strip()
    if r is None or r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(
            f"[served-cap] no {tc.TREE_SHA_FILENAME} file at {root} and it is not a git "
            f"checkout ({why or 'git rev-parse HEAD failed'}) — the manifest must name the "
            "tree that produced it (producer.tree_sha; the later EvalPlus merge requires "
            "the same tree); the launcher writes the file at upload")
    return {"tree_sha": r.stdout.strip(), "driver": tc.PRODUCER_DRIVER_SERVED}


def _seeds_block(families, samples_by_task: Mapping, *, inspect_seed_policy=None, inspect_seed_base=None) -> dict:
    """``manifest["seeds"]``: the seed RULE and its constants per requested
    family (the per-(item, sample) values themselves ride in every row's
    ``extra_json`` and are recomputable from the rule — PB2), and, since U8
    / PW-sampling, each decoding family's ``samples_per_item`` as the
    ``{row task: k}`` slice of the run's map (the same values the identity
    hashes as ``samples_by_task``)."""
    fam = set(families)
    out: dict = {}
    if fam & set(tc.INSPECT_CONTROLS):
        from sisteer.capeval.tiers import inspect_served

        rules = tc.seed_rules_for(samples_by_task, inspect_seed_policy=inspect_seed_policy, inspect_seed_base=inspect_seed_base)
        out["inspect"] = {"rule": rules["inspect"],
                          "samples_per_item": {t: int(k) for t, k in sorted(samples_by_task.items())
                                               if t in tc.INSPECT_TASKS}}
        policy, base = tc.check_inspect_seed_policy(inspect_seed_policy, inspect_seed_base)
        if policy == tc.INSPECT_SEED_POLICY_REPLICATE:
            # the Qwen base seed-replicate battery: the OPT-IN replicate rule and its base, recorded beside the
            # rule string the identity hashes; a default run's block is byte-identical (this key is absent)
            out["inspect"]["replicate"] = {"policy": policy, "seed_base": base,
                                           "derivation": "contracts.inspect_replicate_seed (inspect_served.seed_for_task)"}
        ifeval_task = tc.inspect_task(tc.IFEVAL_TASK_NAME)
        if ifeval_task in samples_by_task:
            # the canonical IFEval battery: IFEval's OWN rule and its
            # constants, recorded beside the Inspect rule for exactly the runs
            # that file it (every other manifest's block is byte-identical) —
            # the same conditional `seed_rules_for` the identity hashes
            out["inspect"]["ifeval"] = {"rule": rules[ifeval_task],
                                        "seed_base": tc.IFEVAL_SEED_BASE,
                                        "readout": tc.IFEVAL_SEED_READOUT,
                                        "derivation": "inspect_served.ifeval_sample_seed"}
    if fam & set(tc.EVALPLUS_CONTROLS):
        from sisteer.capeval.tiers import evalplus_runner

        out["evalplus"] = {"rule": evalplus_runner.SEED_RULE,
                           "samples_per_item": {t: int(k) for t, k in sorted(samples_by_task.items())
                                                if t in tc.EVALPLUS_TASKS}}
    return out


def _sandbox_block(families) -> "dict | None":
    """``manifest["sandbox"]`` (P18): the execution-sandbox constants from
    the contracts when an EvalPlus family is requested; None otherwise (no
    generated code of this run will ever execute)."""
    if not (set(families) & set(tc.EVALPLUS_CONTROLS)):
        return None
    return {"image_digest": tc.SANDBOX_IMAGE_DIGEST, "cpu": tc.SANDBOX_CPU,
            "memory_mib": tc.SANDBOX_MEMORY_MIB, "timeout_s": tc.SANDBOX_TIMEOUT_S,
            "block_network": tc.SANDBOX_BLOCK_NETWORK}


#: The launcher's persisted geometry record (PI5), written beside the cells
#: under the driver's ``--out-root`` the moment the standup is up: it carries
#: the PASS number (and the budget override) the standup's own record cannot.
SERVED_GEOMETRY_FILENAME = "served_geometry.json"


def _launcher_geometry_record(out_root) -> "dict | None":
    """The ``geometry`` block of the launcher's ``served_geometry.json``
    beside the cells (PI5) — the record that carries the PASS number the
    standup's own record cannot — or None when there is none (a hand run
    without the launcher). The ONE reader: the hashed identity's pass
    (:func:`_launcher_pass`) and the manifest's geometry block
    (:func:`_geometry_block`) are both read off it, so they cannot
    disagree. A record without a geometry block is malformed and refuses."""
    rec_path = (Path(out_root) / SERVED_GEOMETRY_FILENAME) if out_root else None
    if rec_path is None or not rec_path.is_file():
        return None
    rec = json.loads(rec_path.read_text())
    g = rec.get("geometry") if isinstance(rec, dict) else None
    if not isinstance(g, dict):
        raise SystemExit(
            f"[served-cap] {rec_path} carries no geometry block — the launcher's record is "
            "malformed; refusing to resolve a pass from it")
    return g


def _launcher_pass(record: "dict | None") -> "int | None":
    """The geometry PASS this run executes under — ``geometry.pass`` of the
    launcher's record (:func:`_launcher_geometry_record`); None for a hand
    run without one. HASHED into the tinycap identity (PW8 / E16 A2): until
    then the pass lived only in the manifest's top-level geometry block,
    which a resume rewrites, so a pass-2 run whose window and S resolved to
    pass 1's values keyed onto the pass-1 store and relabelled its rows. A
    recorded pass that is not one of ``GEOMETRY_PASSES`` refuses."""
    if record is None:
        return None
    p = record.get(tc.GEOMETRY_PASS_FIELD)
    if p is None:
        return None
    if isinstance(p, bool) or not isinstance(p, int) or p not in tc.GEOMETRY_PASSES:
        raise SystemExit(
            f"[served-cap] {SERVED_GEOMETRY_FILENAME} records {tc.GEOMETRY_PASS_FIELD}={p!r}, "
            f"not one of {list(tc.GEOMETRY_PASSES)} — refusing to hash an unknown pass into "
            "the store identity")
    return int(p)


def _endpoint_is_native(endpoint: object, *, source: str) -> bool:
    """``contracts.endpoint_is_native`` over a standup record — the ONE engine
    reading every native branch of this driver keys on (``engine ==
    lora_native``, or the row-facing ``serving_engine`` the native core writes
    beside it; the ``/v1/models`` identity block's ``{"kind": …}`` spelling
    too; a lens record carries neither key and reads False). A record whose
    two spellings disagree is torn and REFUSES here (``SystemExit`` naming
    ``source``) rather than being read as either engine — the same reading
    the tinycap launcher's ``endpoint_engine`` and the wave's ``f22_standup``
    make."""
    try:
        return tc.endpoint_is_native(endpoint)
    except ValueError as e:
        raise SystemExit(f"[served-cap] {source}: {e}") from None


def _served_reduction_mode(endpoint: Mapping, *, source: str) -> str:
    """The ``reduction_mode`` the served identity and the manifest ``route``
    block record, ENGINE-AWARE: on a NATIVE record
    ``contracts.served_reduction_mode`` — ``not_applicable``, an explicit value
    (so the launcher's resume guard ``route.get(reduction_mode, per_row)``
    never falls to ``per_row`` on a native store); a native record that ALSO
    states a ``lens_reduction_mode`` is torn and refuses. On a LENS record
    ``contracts.resolve_reduction_mode`` over the record's stated
    ``lens_reduction_mode`` exactly as before (absent = ``per_row``, E25) —
    byte-identical for every lens store, its ``ValueError`` on an unknown
    mode included."""
    if _endpoint_is_native(endpoint, source=source):
        try:
            return tc.served_reduction_mode(endpoint, source=source)
        except ValueError as e:
            raise SystemExit(f"[served-cap] {e}") from None
    return tc.resolve_reduction_mode(endpoint.get(STANDUP_REDUCTION_MODE_KEY), sources=(source,))


def _hold_serving_path_expectation(endpoint: Mapping, expect: "str | None", *, source: str) -> str:
    """The serving ENGINE a standup record attests — ``lora_native`` iff
    :func:`_endpoint_is_native`, else ``lens`` (a lens standup's record has no
    ``engine`` key: lens BY ERA, the ``REDUCTION_MODE_LEGACY`` pattern) — held
    to ``expect`` (:data:`FLAG_SERVING_PATH`) when one was stated: a
    disagreement is ``contracts.REC_SERVING_ENGINE_MISMATCH``, refused before
    any served fact is copied off the record (the launcher's own post-standup
    rule). Returns the engine."""
    engine = (tc.SERVING_ENGINE_LORA_NATIVE if _endpoint_is_native(endpoint, source=source)
              else tc.SERVING_ENGINE_LENS)
    if expect is not None and expect != engine:
        raise SystemExit(
            f"[served-cap] {tc.REC_SERVING_ENGINE_MISMATCH}: {source} attests {tc.ENDPOINT_KEY_ENGINE}="
            f"{engine!r} but this run's {FLAG_SERVING_PATH} {expect} expects {expect!r} — the standup did "
            "not serve the engine this run was pointed at; refusing rather than recording either "
            "(native-serving-default plan §1.3)")
    return engine


def _cells_skipped_of(ep: Mapping, path) -> "list[dict]":
    """``endpoint.json.cells_skipped`` (:data:`STANDUP_CELLS_SKIPPED_KEY`) as the
    native core writes it — a list of ``{cell_id, kind, reason}`` — validated:
    a present value that is not such a list refuses (a malformed claim about
    what was served is never read around). Absent reads ``[]``: the core
    writes the key on every record, and a record without it states no skipped
    cell — nothing to honour, not a refusal."""
    raw = ep.get(STANDUP_CELLS_SKIPPED_KEY)
    if raw is None:
        return []
    if not isinstance(raw, list) or any(
            not isinstance(s, Mapping) or not isinstance(s.get("cell_id"), str) or not s.get("cell_id")
            for s in raw):
        raise SystemExit(
            f"[served-cap] standup record {path}: {STANDUP_CELLS_SKIPPED_KEY} is {raw!r}, not a list of "
            "{cell_id, kind, reason} records (registry_adapter.SkippedCell) — the standup's own statement "
            "of what it skipped is malformed; refusing to read around it")
    return [dict(s) for s in raw]


def _read_native_standup_facts(ep: dict, path, *, cell: "str | None") -> None:
    """The NATIVE branch of :func:`_standup_endpoint`: the record attests
    ``engine == lora_native`` (the native multi-LoRA core's
    ``LoraStandup.endpoint()``), so

    * the LENS reduction-mode machinery is NOT consulted — no
      ``lens_reduction_mode`` is expected on the record (one that states it is
      torn: ``contracts.served_reduction_mode`` refuses), ``resolve_reduction_mode``
      / ``lens_worker.endpoint_reduction_mode`` / ``check_reduction_mode_expectation``
      never run, and the served identity records ``reduction_mode =
      not_applicable``; the manifest's route block says the expectation was
      :data:`REDUCTION_MODE_EXPECTATION_NOT_CONSULTED`;
    * ``cells_skipped`` is HONOURED: every skipped cell is printed by name with
      its kind and reason, and a run whose ``--cell`` is one of them (in either
      spelling: the TrainTest id or its harness translation) REFUSES here — no
      route serves it, the standup said so, and the honest path for that cell is
      a lens standup (``--serving-path lens``): never a second standup started
      here, never a silent drop.
    """
    try:
        tc.served_reduction_mode(ep, source=f"standup record {path}")
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e} (native-serving-default plan §1.3)") from None
    skipped = _cells_skipped_of(ep, path)
    print(f"[served-cap] standup record attests {tc.ENDPOINT_KEY_ENGINE}={tc.SERVING_ENGINE_LORA_NATIVE!r}: "
          f"the native multi-LoRA engine — the served identity records "
          f"{tc.REDUCTION_MODE_FIELD}={tc.REDUCTION_MODE_NOT_APPLICABLE!r} (no lens hook; "
          f"{tc.LENS_STACKED_REDUCTIONS_ENV} {REDUCTION_MODE_EXPECTATION_NOT_CONSULTED}); "
          f"{len(skipped)} cell(s) skipped by the standup", flush=True)
    for s in skipped:
        print(f"[served-cap] the standup SKIPPED cell {s['cell_id']!r} (kind {s.get('kind')!r}): "
              f"{s.get('reason')!r}", flush=True)
    if cell is None or not skipped:
        return
    spellings = {str(cell)}
    from sisteer.capeval.petri.registry_adapter import RegistryAdapterError, harness_cell_id
    try:
        spellings.add(harness_cell_id(str(cell)))
    except RegistryAdapterError:
        pass   # a cell the harness grammar cannot spell is compared under its own id only
    hit = [s for s in skipped if s["cell_id"] in spellings]
    if hit:
        s = hit[0]
        raise SystemExit(
            f"[served-cap] cell {cell!r} was SKIPPED by the native standup (standup record {path} "
            f"{STANDUP_CELLS_SKIPPED_KEY}: kind {s.get('kind')!r}, reason {s.get('reason')!r}) — no route "
            "serves it on this engine; a cell with no PEFT form belongs on a lens standup "
            "(--serving-path lens), never on this one; refusing before any store is touched")


def _native_lens_position_record() -> dict:
    """Predicate (B')'s evidence block on a NATIVE record: ``not_applicable``,
    stated — the lens-position attestation reads the served-positions map, the
    attested resumed-prefill policy and the lens salt triple, none of which a
    native LoRA standup has a surface for (its record carries none of the keys;
    the adapter applies at every position). Never derived from the absent lens
    keys, never read as verified."""
    return {"status": tc.NOT_APPLICABLE, "verified": None,
            "reason": (f"{tc.ENDPOINT_KEY_ENGINE}={tc.SERVING_ENGINE_LORA_NATIVE!r}: the lens-position "
                       "attestation (predicate B') reads lens facts — the served-positions map, the "
                       "resumed-prefill policy, the lens salt triple — that a native LoRA standup does not "
                       "have; the adapter applies at every position (native-serving-default plan §1.3)")}


def _native_identity_seqs(geom: Mapping) -> int:
    """The IDENTITY S of a native record's geometry block — the S the block's
    OWN recorded ``budget`` buys at its window under the resolver's rule:
    ``ModelBudget.seqs_at(L, budget=<the block's>)`` (``budget // L``; a
    hybrid-recurrent family's :data:`geometry.HYBRID_SEQS_CAP` and measured
    window ceiling on top), the bare quotient (``geometry.identity_seqs``)
    when the block names no model — what the §1.8 kernel step was applied TO.
    Read off the block, never the live table: a standup under a provisional
    budget contracts THAT budget's S (the GSM8K / MATH-500 validators
    re-derive it the same way). Refuses a block naming no positive-int budget
    / window, or a model the geometry table does not know.

    The derivation itself is ``geometry.recorded_identity_seqs`` — the ONE
    function the tinycap launcher's production reading of a ladder rung
    calls too, so the probe that certifies a rung and this driver cannot
    disagree on what the record's identity S is; this wrapper only names
    the refusals."""
    try:
        return int(sg.recorded_identity_seqs(geom))
    except sg.UnknownModelKey as e:
        raise SystemExit(f"[served-cap] {STANDUP_ENDPOINT_FILENAME}: geometry.model_key: {e}") from None
    except ValueError as e:
        raise SystemExit(f"[served-cap] {STANDUP_ENDPOINT_FILENAME}: {e} — "
                         "the native record's identity S cannot be re-derived; refusing") from None


def _native_geometry_problem(geom: Mapping, lora_config: "Mapping | None" = None) -> "str | None":
    """Why a NATIVE record's geometry block fails the §1.8 contract, or None
    (``geometry.geometry_adjustment_problem`` with ``native=True``: the kernel
    step PRESENT IFF the formula lowers — re-derived from the model's widest
    adapted module, L and the identity S — the KV step from the previous
    step's S, the served S the last step's; a bare smaller S with no record
    is named, never accepted)."""
    S_identity = _native_identity_seqs(geom)
    model_key = geom.get("model_key")
    # the standup's ATTESTED adapter class (endpoint.json lora_config: max_lora_rank / max_loras), when the record carries it: a
    # `launch_class` step must be THIS standup's class, not a class that happens to re-derive (geometry-fix-2, 2026-10-02)
    served_lora = None
    if isinstance(lora_config, Mapping):
        r_, n_ = lora_config.get("max_lora_rank"), lora_config.get("max_loras")
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in (r_, n_)):
            served_lora = (int(r_), int(n_))
    return sg.geometry_adjustment_problem(
        geom, model_key=(model_key if isinstance(model_key, str) else None),
        L=int(geom["max_model_len"]), S_identity=S_identity,
        S_served=geom.get("max_num_seqs"), native=True, served_lora=served_lora)


def _geometry_block(endpoint: dict, out_root) -> "dict | None":
    """``manifest["geometry"]`` = the standup record's ``geometry`` block
    (``ResolvedGeometry.as_dict()`` as ``capctrl_serving_up.py`` wrote it —
    the window and S actually served) + ``pass`` from the launcher's
    ``served_geometry.json`` beside the cells when present (None for a hand
    run without the launcher). The two records must agree on the window
    and S; a disagreement is a launcher/standup mismatch and refuses. The
    pass written here is the SAME value the hashed identity carries
    (:func:`_launcher_pass` over the same record); this block is not
    identity and only ever a cross-check on it (``completion.merge_source``)."""
    geom = endpoint.get("geometry")
    if not isinstance(geom, dict):
        return None   # _standup_endpoint refuses this for a real run
    g = _launcher_geometry_record(out_root)
    block = dict(geom)
    # A truncation-ladder rung's DELIBERATE reduction rides the LAUNCHER's record
    # (``served_geometry.json`` geometry [``geometry.LADDER_REDUCTION_KEY``] — the standup
    # knows no ladder and records only what it was asked), so it is composed into the block
    # BEFORE the rule below: the chain then re-derives from the budget's identity S through
    # the requested rung the standup was asked to the standup's own steps, and the
    # manifest's block carries it. The probe that certified the rung ran this same reading
    # on this same record (``tinycap_served.native_ladder_geometry_problem``). A reduction
    # beside a LENS record is a wiring error and refuses; a lens record without one is
    # copied as before.
    reduction = g.get(sg.LADDER_REDUCTION_KEY) if isinstance(g, dict) else None
    if _endpoint_is_native(endpoint, source=STANDUP_ENDPOINT_FILENAME):
        # native-serving-default §1.8 — the ONE geometry-adjustment contract: a
        # native record's ``geometry.adjustments`` (``[]`` / ``[kernel]`` / ``[kv]``
        # / ``[kernel, kv]``) is RE-DERIVED before the block is copied whole
        # (``geometry.geometry_adjustment_problem(native=True)``: the kernel step
        # from the model's widest adapted module, L and the identity S — present
        # iff it lowers, gemma 121 -> 107 / nemotron 161 -> 86 at 4864 — the KV
        # step from the previous step's S, the served S the last step's), so a
        # manifest never carries "a smaller S" without a named, validated reason.
        # A lens record is copied exactly as before (no rule runs here: its S is
        # the launcher's, and the validators hold it to the budget).
        if reduction is not None:
            block[sg.LADDER_REDUCTION_KEY] = dict(reduction)
        problem = _native_geometry_problem(block, endpoint.get("lora_config"))
        if problem is not None:
            raise SystemExit(
                f"[served-cap] {STANDUP_ENDPOINT_FILENAME}: the native standup's geometry block does not "
                f"hold the §1.8 adjustment contract — {problem}; refusing to copy it into the manifest "
                "(a served S below the budget's is accepted only through a named, re-derivable "
                "adjustment, never as a smaller S)")
    elif reduction is not None:
        raise SystemExit(
            f"[served-cap] {Path(out_root) / SERVED_GEOMETRY_FILENAME} records a {sg.LADDER_REDUCTION_KEY} beside a "
            "LENS standup record — the reduction record is the native ladder's (the lens ladder's rung is the "
            "launcher's S, held to the budget by the validators); refusing")
    block[tc.GEOMETRY_PASS_FIELD] = None
    block["source"] = STANDUP_ENDPOINT_FILENAME
    if g is not None:
        for k in ("max_model_len", "max_num_seqs"):
            if k in g and g[k] != geom.get(k):
                raise SystemExit(
                    f"[served-cap] {Path(out_root) / SERVED_GEOMETRY_FILENAME} records "
                    f"{k}={g[k]!r} but the standup's {STANDUP_ENDPOINT_FILENAME} says "
                    f"{geom.get(k)!r} — the launcher's geometry record and the served "
                    "geometry disagree; refusing")
        block[tc.GEOMETRY_PASS_FIELD] = _launcher_pass(g)
        if "budget_override" in g:
            block["budget_override"] = g["budget_override"]
        block["source"] = f"{STANDUP_ENDPOINT_FILENAME}+{SERVED_GEOMETRY_FILENAME}"
    return block


#: The standup's record, written ATOMICALLY beside ``routing_table.json``
#: by ``capctrl_serving_up.py`` (both land in its ``--out`` dir) and unlinked
#: at teardown — so while the proxy answers, the record is there.
STANDUP_ENDPOINT_FILENAME = "endpoint.json"
#: The serving-MODALITY keys the standup records (PB13): what the server was
#: actually launched with, as opposed to what a launcher meant to ask for.
STANDUP_MODALITY_KEYS = ("reasoning_parser", "tool_calls")
#: The proxy's Petri-facing agent-loop entry (tinycap U4(b) / PW18): REQUIRED
#: on the record — it decides whether a lens route serves BFCL's tool turns
#: (the last-header regime) and is hashed into the served identity, so a
#: standup that cannot say is refused like one that cannot say its parser.
STANDUP_ENTRY_KEY = tc.AGENT_LOOP_ENTRY_KEY
#: The lens reduction mode (E25 / PW18; E46 once the stacked landing merged):
#: OPTIONAL on the record. A record with NO such key is a PRE-LANDING standup
#: — every store frozen before the landing — and means ``per_row``, never
#: null. A post-landing standup always writes the key, from the workers' own
#: steering attestation, and writes ``None`` there only when it could not
#: determine what it served; it says why in
#: :data:`STANDUP_REDUCTION_MODE_ERROR_KEY`, and that case is REFUSED rather
#: than read as the ``per_row`` default (E29 / PW18b: nothing is recorded into
#: identity on a disagreement).
STANDUP_REDUCTION_MODE_KEY = tc.PROXY_IDENTITY_REDUCTION_MODE
#: Why the standup attested no reduction mode (``capctrl_serving_up``'s
#: ``lens_reduction_mode_error``): a missing or unreadable attestation, a
#: worker that stamps no mode, workers that disagree. Present only on
#: post-landing records.
STANDUP_REDUCTION_MODE_ERROR_KEY = "lens_reduction_mode_error"
#: ``endpoint.json.cells_skipped`` — the
#: view's cells a NATIVE standup skipped BY NAME (a stacked cell, a position-masked
#: LoRA fit — PSR / vector cells serve on the residual slot since native-psr-serving;
#: ``registry_adapter.SkippedCell``: ``{cell_id, kind, reason}``),
#: written by the native core beside ``engine``. A lens record has no such key.
#: The driver prints one line per skipped cell and REFUSES a run whose
#: ``--cell`` is among them — no route serves it, and the honest path for such
#: a cell is ``--serving-path lens`` on the standup — rather than failing later
#: on a missing route.
STANDUP_CELLS_SKIPPED_KEY = "cells_skipped"
#: The manifest ``route`` block's UNHASHED record of how the lens reduction-mode
#: EXPECTATION (``SISTEER_LENS_STACKED_REDUCTIONS``, E29 / PW18b) was handled on
#: a NATIVE record: not consulted — the native engine has no lens hook,
#: ``reduction_mode`` is recorded ``not_applicable`` (an explicit value), and
#: neither ``resolve_reduction_mode`` nor ``check_reduction_mode_expectation``
#: runs. PRESENT IFF native (a lens route block is byte-identical).
ROUTE_KEY_REDUCTION_MODE_EXPECTATION = "reduction_mode_expectation"
REDUCTION_MODE_EXPECTATION_NOT_CONSULTED = "not_consulted"
#: ``--serving-path {lens,lora_native}``: the serving ENGINE this run EXPECTS
#: the standup to have served, held to the record's own ``engine`` (absent =
#: lens by era) before any served fact is read — a cross-check the wave shell
#: or a launcher may state, never an input: the record is the authoritative
#: source and an unstated flag leaves it to say (SERVING_ENGINE_MISMATCH on a
#: disagreement). Judgement call (W-C2): the brief's "defaulting to the
#: registry adapter's DEFAULT_STANDUP_SERVING_PATH" would refuse every existing
#: lens invocation (the goldens, and ``contracts.driver_argv``, which composes
#: no such flag), so the default here is UNSTATED and the engine is always READ
#: off the record, never assumed.
FLAG_SERVING_PATH = "--serving-path"
#: 16-hex sha256 over the chat template the standup LAUNCHED the server with
#: (``capctrl_serving_up.Standup.chat_template_hash``; E45 M5). OPTIONAL, like
#: the reduction mode: a standup that predates the key records nothing, and
#: the mixed-arm header block then states the comparison was unavailable
#: rather than assuming it passed. Its one consumer is that block — it is what
#: makes a locally-measured render provably (or not) the served one.
#: Equals ``contracts.ENDPOINT_KEY_CHAT_TEMPLATE_HASH``; a test pins this literal spelling.
STANDUP_TEMPLATE_HASH_KEY = "chat_template_hash"
#: The standup's ``geometry`` block (``ResolvedGeometry.as_dict()``): the
#: window and S actually served — the manifest's ``geometry`` block (R4).
STANDUP_GEOMETRY_KEY = "geometry"
#: The record's BINDING keys (E10 N2): the proxy it vouches for and the
#: table it was exported with — compared against ``--proxy-url`` /
#: ``--routing-table`` before any served fact is copied off the record.
STANDUP_BINDING_KEYS = ("proxy_url", "routing_table")
STANDUP_REQUIRED_KEYS = (STANDUP_BINDING_KEYS + STANDUP_MODALITY_KEYS
                         + (STANDUP_GEOMETRY_KEY, STANDUP_ENTRY_KEY))
#: The manifest's ``binding`` block (tinycap runs): the endpoints this run
#: was VERIFIED against — canonical proxy URL, the resolved table path, and
#: (with a ledger) the ledger URL, its recorded upstream and its record.
MANIFEST_KEY_BINDING = "binding"
BINDING_FIELDS = ("proxy_url", "routing_table", "ledger_url", "ledger_upstream",
                  "ledger_endpoint")


class EndpointBindingMismatch(SystemExit):
    """The standup record beside ``--routing-table`` does not vouch for the
    proxy, table or ledger this run was pointed at (E10 N2): its
    ``proxy_url`` is not ``--proxy-url``, its ``routing_table`` is not the
    table given, or the ledger's record names another URL / another
    upstream. Filing numbers under that record would be scoring one
    standup under another's manifest, so the run refuses before any store
    is touched. A ``SystemExit`` so the CLI exits the way every other
    driver refusal does; typed so a caller can tell it apart."""


class ServingIdentityMismatch(SystemExit):
    """An existing cell manifest records a served identity — reasoning
    parser, tool-call mode, window (``contracts.SERVED_IDENTITY_FIELDS``) —
    other than this standup's (R4 / E10 N3). The identity is hashed into
    the cell key, so this fires only on a collision, a copied or hand-edited
    store; it names old vs new per field and refuses the resume."""


def _standup_endpoint(routing_table, *, cell: "str | None" = None,
                      expect_serving_path: "str | None" = None) -> dict:
    """Read the standup's ``endpoint.json`` next to ``--routing-table`` (the
    two are siblings by the standup's construction; ``endpoint.json`` even
    names the table's path). The manifest's ``thinking.reasoning_parser``,
    ``route.tool_calls``, ``route.agent_loop_entry``, ``route.reduction_mode``
    and ``geometry`` are copied from it, so they state what was SERVED (PB13 /
    R4 / PW18) — never the launcher's intent, never a literal; the lens
    reduction mode is the record's ATTESTED ``lens_reduction_mode``, read
    through ``lens_worker.endpoint_reduction_mode`` (absent = a pre-landing
    server = per_row; null WITH a stated reason refuses), and a
    ``SISTEER_LENS_STACKED_REDUCTIONS``
    set in this process is held to it (E29 / PW18b). Refuses (``SystemExit``) when the file is absent or lacks a
    modality key or the geometry block: a standup that predates the record
    cannot vouch for what it composed, and a guessed ``reasoning_parser``
    is exactly the silent failure PB12 describes (a wrong parser returns
    200s with the scratchpad in ``content``).

    ENGINE-AWARE: the record's
    ``engine`` is read FIRST (:func:`_endpoint_is_native`; absent = lens by
    era) and held to ``expect_serving_path`` when the run stated one
    (:data:`FLAG_SERVING_PATH`). A NATIVE record takes
    :func:`_read_native_standup_facts` — no ``lens_reduction_mode`` expected,
    no lens reader consulted, ``cells_skipped`` honoured for ``cell`` — and a
    LENS record takes :func:`_hold_lens_reduction_mode`, the paragraph above,
    byte for byte."""
    path = Path(routing_table).resolve().parent / STANDUP_ENDPOINT_FILENAME
    if not path.is_file():
        raise SystemExit(
            f"[served-cap] no standup record at {path}: capctrl_serving_up.py "
            f"writes {STANDUP_ENDPOINT_FILENAME} beside routing_table.json "
            "while the proxy is up; the tinycap manifest records the served "
            "reasoning_parser / tool_calls / geometry from it and refuses to guess (PB13)")
    ep = json.loads(path.read_text())
    missing = [k for k in STANDUP_REQUIRED_KEYS if k not in ep]
    if missing:
        raise SystemExit(
            f"[served-cap] standup record {path} lacks {missing}: a standup "
            "that predates the modality/geometry record (capctrl_serving_up.py at "
            "Lane B's tip) cannot say what it served — refusing rather than "
            "recording a literal (PB13)")
    if not isinstance(ep[STANDUP_ENTRY_KEY], bool):
        raise SystemExit(
            f"[served-cap] standup record {path}: {STANDUP_ENTRY_KEY} is "
            f"{ep[STANDUP_ENTRY_KEY]!r}, not a JSON boolean — the proxy's agent-loop "
            "entry is a stated fact (served identity, PW18), never a truthy coercion")
    source = f"standup record {path}"
    _hold_serving_path_expectation(ep, expect_serving_path, source=source)
    if _endpoint_is_native(ep, source=source):
        _read_native_standup_facts(ep, path, cell=cell)
    else:
        _hold_lens_reduction_mode(ep, path)
    geom = ep[STANDUP_GEOMETRY_KEY]
    for k, why in (("max_model_len", "the served window is store identity (N3)"),
                   ("max_num_seqs", "the served S is store identity (PW5-4)")):
        v = geom.get(k) if isinstance(geom, dict) else None
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise SystemExit(
                f"[served-cap] standup record {path}: geometry.{k} is {v!r}, "
                f"not a positive integer — {why} and is never guessed; refusing")
    return ep


def _hold_lens_reduction_mode(ep: dict, path) -> None:
    """The LENS branch of :func:`_standup_endpoint`, MOVED here verbatim when
    the native branch landed so the two
    engines' readings sit side by side: the attested ``lens_reduction_mode``
    through ``lens_worker.endpoint_reduction_mode`` (absent = a pre-landing
    server = per_row; null WITH a stated reason refuses), the vocabulary
    reading first, and the ``SISTEER_LENS_STACKED_REDUCTIONS`` expectation of
    this process held to it (E29 / PW18b). Every refusal reads exactly as it
    did before the move."""
    # E46 (the stacked landing merged): a post-landing standup ALWAYS writes
    # the key and writes None + a reason when it could not read the workers'
    # attestation. "No mode stated" is keyless — a pre-landing server, per_row
    # — but only when the standup did not also say WHY it has none; a stated
    # reason means the standup could not determine what it served, and that
    # must never be read as the per_row default and recorded as identity.
    #
    # That last reading used to be an inline copy right here. It is not any
    # more: main took the rule from this read site into
    # lens_worker.endpoint_reduction_mode (39399acc), so the second main merge
    # calls it back rather than leaving one rule with two implementations. What
    # stays here is what the resolver has no notion of — contracts'
    # resolve_reduction_mode, the launcher's VOCABULARY reading, shared with
    # the three identity-block read sites below and refusing an unknown mode in
    # this launcher's own words; and the environment expectation. The
    # vocabulary reading therefore runs FIRST, so an unknown mode still refuses
    # with the launcher's wording and the resolver only ever sees a value this
    # launcher already accepts.
    #
    # The import is function-local to keep torch — lens_worker's dependency,
    # and nothing this driver otherwise loads — out of the module's import
    # closure, the way unified_sweep_run already imports it locally. Main's
    # capctrl_served keeps a deliberate SECOND COPY of the resolver for a
    # hazard that cannot arise here: that driver is mounted BESIDE a staged
    # candidate tree of another sha, while this read site and lens_worker ship
    # in one tree at one sha and are merged or absent together.
    from sisteer.capeval.serving import lens_worker as lw
    stated_mode = ep.get(STANDUP_REDUCTION_MODE_KEY)
    keyless = stated_mode is None
    try:
        attested = tc.resolve_reduction_mode(stated_mode,
                                             sources=(f"standup record {path}",))
        resolved = lw.endpoint_reduction_mode(ep, source=f"standup record {path}")
        if resolved != attested:
            # unreachable while the two agree on the vocabulary and on what an
            # unstated mode means — which is the point of checking it at the
            # read site rather than trusting a test to notice the day they part
            raise ValueError(
                f"standup record {path}: this launcher reads "
                f"{STANDUP_REDUCTION_MODE_KEY}={attested!r} from the record and "
                f"lens_worker.endpoint_reduction_mode reads {resolved!r} — the two readings "
                "of one record have drifted; refusing rather than recording either")
        # E29 / PW18b: the attestation is the truth; the environment of THIS
        # process (the container's, shared with the standup — the launcher sets
        # it to one canonical value) yields the serving parser's EXPECTATION,
        # which must agree: a disagreement is refused, never recorded, and an
        # expectation of stacked against a keyless (pre-landing) server refuses
        tc.check_reduction_mode_expectation(
            attested, tc.reduction_mode_from_env(os.environ),
            expected_source=f"this container's {tc.LENS_STACKED_REDUCTIONS_ENV}", keyless=keyless)
    except lw.ReductionModeUnattested as e:
        # the ported rule, raised by its ONE implementation; the marker is this
        # launcher's, so the refusal reads exactly as it did before the merge
        raise SystemExit(f"[served-cap] {e} (E29 / PW18b)") from None
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    if keyless:
        print(f"[served-cap] standup record states no {STANDUP_REDUCTION_MODE_KEY}: a pre-landing "
              f"server — the served identity records {tc.REDUCTION_MODE_FIELD}={attested!r} (E29)",
              flush=True)


def _bind_endpoint(endpoint: dict, *, proxy_url: str, routing_table,
                   ledger_url: str = "") -> dict:
    """Bind the standup record to the endpoints THIS run was pointed at
    (E10 N2), refusing (:class:`EndpointBindingMismatch`, naming both
    values) on any disagreement:

    * ``endpoint.proxy_url`` == ``--proxy-url`` (both canonicalized to the
      ``<root>/v1`` form the clients use);
    * ``endpoint.routing_table`` resolves to the same file as
      ``--routing-table``;
    * with ``--ledger-url``: the ledger's record
      ``<table dir>/<contracts.LEDGER_ENDPOINT_FILENAME>`` (its
      ``--endpoint-json``, which the launcher writes beside the table) must
      exist, its ``ledger_url`` must equal the flag and its ``upstream``
      must equal the standup's proxy — the ledger that receives the tinycap
      traffic forwards to the proxy the manifest vouches for.

    Returns the manifest's ``binding`` block (:data:`BINDING_FIELDS`). The
    positive controls deliberately stay on ``--proxy-url`` (PW4-3): with this
    chain verified the two URLs are one serving, control probes are not task
    calls and must not enter the ledger's geometry basis, and every task
    response is stamp-verified against its route regardless."""
    have_proxy = endpoint.get("proxy_url")
    if not isinstance(have_proxy, str) or not have_proxy:
        raise EndpointBindingMismatch(
            f"[served-cap] endpoint binding: the standup record carries no proxy_url "
            f"({have_proxy!r}) — it cannot vouch for any proxy; refusing (N2)")
    have_proxy = _canonical_v1_base(have_proxy)
    want_proxy = _canonical_v1_base(proxy_url)
    if have_proxy != want_proxy:
        raise EndpointBindingMismatch(
            f"[served-cap] endpoint binding: --proxy-url is {want_proxy!r} but the standup "
            f"record beside the routing table names proxy_url {have_proxy!r} — the record "
            "does not vouch for the proxy that would receive this run's traffic (another "
            "live standup? a stale record?); refusing before any store is touched (N2)")
    have_table = endpoint.get("routing_table")
    if not isinstance(have_table, str) or not have_table:
        raise EndpointBindingMismatch(
            f"[served-cap] endpoint binding: the standup record carries no routing_table "
            f"({have_table!r}); refusing (N2)")
    have_table = Path(have_table).resolve()
    want_table = Path(routing_table).resolve()
    if have_table != want_table:
        raise EndpointBindingMismatch(
            f"[served-cap] endpoint binding: --routing-table resolves to {str(want_table)!r} "
            f"but the standup record names routing_table {str(have_table)!r} — this record "
            "was written for another table; refusing (N2)")
    binding = {"proxy_url": have_proxy, "routing_table": str(have_table),
               "ledger_url": None, "ledger_upstream": None, "ledger_endpoint": None}
    if ledger_url:
        want_ledger = _canonical_v1_base(ledger_url)
        rec_path = want_table.parent / tc.LEDGER_ENDPOINT_FILENAME
        if not rec_path.is_file():
            raise EndpointBindingMismatch(
                f"[served-cap] endpoint binding: --ledger-url {want_ledger!r} but no ledger "
                f"record at {rec_path}: the usage ledger writes it (its --endpoint-json) "
                "beside routing_table.json once its probe passed — the launcher does so; a "
                f"hand run passes --endpoint-json <that dir>/{tc.LEDGER_ENDPOINT_FILENAME}. "
                "Without it the ledger's upstream cannot be bound to this standup's proxy; "
                "refusing (N2)")
        rec = json.loads(rec_path.read_text())
        for key in ("ledger_url", "upstream"):
            if not isinstance(rec.get(key), str) or not rec.get(key):
                raise EndpointBindingMismatch(
                    f"[served-cap] endpoint binding: ledger record {rec_path} lacks "
                    f"{key!r} ({rec.get(key)!r}) — refusing (N2)")
        have_ledger = _canonical_v1_base(rec["ledger_url"])
        if have_ledger != want_ledger:
            raise EndpointBindingMismatch(
                f"[served-cap] endpoint binding: --ledger-url is {want_ledger!r} but the "
                f"ledger record {rec_path} names ledger_url {have_ledger!r} — another "
                "ledger's record; refusing (N2)")
        upstream = _canonical_v1_base(rec["upstream"])
        if upstream != have_proxy:
            raise EndpointBindingMismatch(
                f"[served-cap] endpoint binding: the ledger at {have_ledger!r} forwards to "
                f"upstream {upstream!r} but the standup record's proxy is {have_proxy!r} — "
                "the tinycap traffic would be served by a proxy this manifest does not "
                "vouch for; refusing (N2)")
        binding.update(ledger_url=have_ledger, ledger_upstream=upstream,
                       ledger_endpoint=str(rec_path))
    assert tuple(binding) == BINDING_FIELDS
    return binding


def _proxy_upstream_timeout(endpoint: Mapping) -> "float | None":
    """The lens proxy's UPSTREAM (proxy -> vLLM) timeout the standup recorded
    (``contracts.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S``; the value it built its
    ``HTTPBackend`` with), held to this driver's own request budget
    (``pe.AIOHTTP_TOTAL_TIMEOUT_S`` == ``contracts.PROXY_UPSTREAM_TIMEOUT_S``,
    the number the ledger and the Inspect provider wait too). A proxy that
    gives up on vLLM BEFORE the client does turns every draw longer than its
    timeout into a 504 the client's retry RE-SAMPLES — differential censoring
    of the slow (steered, cap-bound) draws; at the ladder's 16,384 rung a
    steered draw at ~2.7 tok/s worst case needs ~6,100 s, inside the 7,200 s
    budget and ten times the proxy's old 600 s default. Present and shorter
    REFUSES (``SystemExit``) before any store is touched; present and a number
    >= the budget is returned for the manifest; absent or null (a standup before
    the knob, or one started without the flag) is returned as ``None`` and
    STATED — never read as any value, and never silently accepted as sufficient."""
    key = tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S
    value = endpoint.get(key)
    budget = float(pe.AIOHTTP_TOTAL_TIMEOUT_S)
    if value is None:
        print(f"[served-cap] standup record carries no {key} (a standup before the knob, or "
              f"started without {tc.FLAG_UPSTREAM_TIMEOUT_S}): the proxy's upstream timeout is "
              f"UNKNOWN to this run and is recorded null — a cap-bound draw longer than the "
              "proxy's own default is 504'd and re-sampled (D73 (d))", flush=True)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"[served-cap] standup record {key}={value!r} is not a number of seconds — a record "
            "that cannot state the proxy's upstream timeout cannot vouch for it; refusing (D73 (d))")
    if float(value) < budget:
        raise SystemExit(
            f"[served-cap] the standup's proxy upstream timeout {key}={float(value):g} s is SHORTER "
            f"than this driver's request budget {budget:g} s (contracts.PROXY_UPSTREAM_TIMEOUT_S): a "
            "draw longer than the proxy's timeout would be 504'd by the proxy and RE-SAMPLED by the "
            "client's retry while the client would still have waited — differential censoring of "
            "cap-bound draws on the slower (steered) route (D73 (d)). Stand the proxy up with "
            f"{tc.FLAG_UPSTREAM_TIMEOUT_S} {budget:g} (the launcher passes it on every tinycap "
            "standup); refusing before any store is touched")
    return float(value)


def _bcb_sandbox_route_block(args, thinking_by_task: Mapping) -> "dict | None":
    """The UNHASHED sandbox facts of a run that
    files BigCodeBench, for the manifest's route block — the provider, image
    and resources the thin task holds the compose to, the sandbox lifetime, the
    verifier timeout, ``max_sandboxes`` (= this run's served S, the driver's
    ``--concurrency``), the app, the OWNER tuple the container declared
    (``TINYCAP_OWNER_*``; null on a hand run, which the provider then refuses to
    create sandboxes under) and the durable record path. None when the run
    files no BigCodeBench row."""
    from sisteer.capeval.tinycap import sandbox_scope as _scope

    if not tc.files_any_bigcodebench(thinking_by_task):      # the 200 pin OR the full split (capeval-full-tasks)
        return None
    owner = _scope.owner_from_env()
    return {**tc.bcb_sandbox_facts(), "max_sandboxes": int(args.concurrency),
            "owner": owner.as_dict() if owner is not None else None,
            "sandbox_record": os.environ.get(_scope.SANDBOX_RECORD_ENV) or None}


def _served_identity(endpoint: dict, *, concurrency: int) -> dict:
    """``resolved_config.tinycap.served`` (``contracts.SERVED_IDENTITY_FIELDS``):
    what the standup SERVED that changes outputs, read off its record — the
    composed ``reasoning_parser`` and the ``tool_calls`` mode; the
    WINDOW ``max_model_len`` (it bounds every request — which
    prompts are 400-rejected, where the once-only clamp lands, which items
    are scorable at all — so a store filled under one window never resumes
    under another); and the served S ``max_num_seqs`` together with
    the client's request width ``concurrency`` (``--concurrency``): under
    bf16 a greedy argmax at a near-tie moves with the batch COMPOSITION, and
    (S, client width) fixes the scheduling regime the batches are drawn from
    — a resume at another S would mix two regimes in one store, and the
    scale-0-repeat leg measures the noise WITHIN one regime only. Also the
    proxy's agent-loop entry ``agent_loop_entry`` — it decides which positions
    a lens route steers on a tool turn (BFCL's regime) — and the lens
    ``reduction_mode`` (``per_row`` when the record states none; ``stacked``
    once the stacked reduction landed): ulp-level logit differences on
    every decode step, so a store under one arithmetic never resumes under
    another. NOT ``gpu_mem_util`` (KV capacity), never pid / port / paths."""
    geom = endpoint[STANDUP_GEOMETRY_KEY]
    ident = {"reasoning_parser": endpoint["reasoning_parser"],
             "tool_calls": bool(endpoint["tool_calls"]),
             "max_model_len": int(geom["max_model_len"]),
             "max_num_seqs": int(geom["max_num_seqs"]),
             "concurrency": int(concurrency),
             tc.AGENT_LOOP_ENTRY_KEY: bool(endpoint[STANDUP_ENTRY_KEY]),
             tc.REDUCTION_MODE_FIELD: _served_reduction_mode(endpoint, source="standup record")}
    native = _endpoint_is_native(endpoint, source="standup record")
    if native:
        # The ENGINE is served identity, PRESENT
        # IFF native — a passthrough leg (`none`, a prompted cell) carries the same
        # `proxy_vllm+passthrough+…` identity on both engines, so without this key a
        # lens-era store and a native-era store of one cell would share a cell key
        # and a resume. The lens key set is byte-identical
        # (contracts.served_identity_fields; the launcher derives the same set).
        ident[tc.SERVED_ENGINE_FIELD] = tc.SERVING_ENGINE_LORA_NATIVE
    # the protocol sampling CLAMP OVERRIDE the standup launched under
    # (endpoint.json `sampling_clamp_override`, PRESENT IFF the launch lifted
    # vLLM's _MAX_TEMP so BFCL's 0.001 executes) — copied VERBATIM into the
    # hashed served block, so a server WITH it never resumes into or pools with
    # one WITHOUT (the card identity, without the key, is byte-identical);
    # `_check_served_clamp_override` holds its presence to the run's sampling mode
    override = endpoint.get(tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE)
    if override is not None:
        if not isinstance(override, Mapping) or set(override) != set(tc.SAMPLING_CLAMP_OVERRIDE):
            raise SystemExit(
                f"[served-cap] standup record: {tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE} {override!r} is not the "
                f"contract's record shape {sorted(tc.SAMPLING_CLAMP_OVERRIDE)} — refusing to hash it")
        ident[tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE] = {k: override[k] for k in tc.SAMPLING_CLAMP_OVERRIDE}
    assert tuple(ident) == tc.served_identity_fields(protocol_sampling=override is not None, native=native)
    return ident


def _check_served_clamp_override(served: dict, *, sampling_mode: str) -> None:
    """The present-iff rule at the driver: a ``protocol`` run must be served
    by a standup that LIFTED the clamp (the record present and EXACTLY
    ``contracts.SAMPLING_CLAMP_OVERRIDE`` — else BFCL's 0.001 would execute at
    0.01 behind a store labelled protocol: the silent substitution the mode
    refuses), and every other mode by one that did NOT (a lifted clamp under
    ``card`` is a server this run never asked for). Refused BEFORE any store is
    read or GPU request sent."""
    have = served.get(tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE)
    if sampling_mode == tc.SAMPLING_PROTOCOL:
        if have is None:
            raise SystemExit(
                f"[served-cap] {tc.FLAG_SAMPLING} {tc.SAMPLING_PROTOCOL}: the standup record carries no "
                f"{tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE} — the server was launched WITHOUT the D13 clamp "
                f"override (capctrl_serving_up.py {tc.FLAG_SAMPLING_CLAMP_OVERRIDE} {tc.SAMPLING_CLAMP_OVERRIDE_ARG}), "
                f"so BFCL's {tc.BFCL_PROTOCOL_TEMPERATURE} would execute at vLLM's {tc.VLLM_MAX_TEMP_PINNED}; "
                "refusing the silent substitution — stand the server up with the override")
        if dict(have) != dict(tc.SAMPLING_CLAMP_OVERRIDE):
            raise SystemExit(
                f"[served-cap] {tc.FLAG_SAMPLING} {tc.SAMPLING_PROTOCOL}: the standup's "
                f"{tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE} {dict(have)!r} is not the pinned record "
                f"{dict(tc.SAMPLING_CLAMP_OVERRIDE)!r} — another override is another instrument; refusing")
    elif have is not None:
        raise SystemExit(
            f"[served-cap] {tc.FLAG_SAMPLING} {sampling_mode}: the standup record carries "
            f"{tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE} {dict(have)!r} — the server was launched with the D13 "
            f"clamp override, which only {tc.FLAG_SAMPLING} {tc.SAMPLING_PROTOCOL} asks for; a {sampling_mode!r} "
            "store must never be filled from a server whose clamp was lifted. Stand it up without the flag")


def _hold_native_route_binding(ident: Mapping, *, route, table: Mapping) -> dict:
    """Bind the LIVE native proxy's inventory to the ROUTE this run drives and
    to the leg its positive controls compare against (the rule
    ``petri_serving_gate.hold_live_inventory`` applies to every route of a
    standup, applied here to the two legs of ONE run; ``ident`` is the
    ``/v1/models`` ``petri_proxy`` block, ``table`` the loaded routing table):

    * a LORA route: the live ``engine.adapters`` names the cell; the adapter's
      ``petri_intervention_hash`` equals the route's (the zero adapter's is
      ``None`` by construction and is exempt), its file digests equal the
      table's adapter block for the cell, and the two zero flags agree;
    * a PASSTHROUGH route (``none``, a prompted cell): NO live adapter of the
      cell's name (an adapter there would steer a leg whose identity says
      nothing steers it); ``base_model`` equal to the table's; and — a
      prompted route — the route's H8 ``si_prefix_hash`` bound to the table's
      own ``prompted_prefixes`` spec for its vmodel (record-internal by
      necessity: the identity block exposes no prefix facts and the prefix
      BYTES never reach the proxy, by containment); a base route has
      neither a hash nor a spec;
    * a LENS route on a native standup refuses by name (base served under a
      steered identity — the proxy refuses one at start; this names it here).

    Every problem is collected and refused at once
    (:class:`EndpointBindingMismatch`). Returns the evidence of what was
    bound (the legs, by path)."""
    from sisteer.capeval.serving import proxy as _px

    engine = ident.get(tc.ENDPOINT_KEY_ENGINE)
    adapters = engine.get("adapters") if isinstance(engine, Mapping) else None
    problems: list = []
    if not isinstance(adapters, Mapping):
        problems.append(f"the live {tc.PROXY_IDENTITY_KEY}.{tc.ENDPOINT_KEY_ENGINE}.adapters block is missing "
                        f"({engine!r}) — the proxy attests no adapter inventory")
        adapters = {}
    blocks = table.get(_px.ADAPTERS_KEY) or {}
    prefix_by_vmodel = {getattr(s, "virtual_model", None): s
                        for s in (table.get("prompted_prefixes") or ())}
    legs = [("route", route)]
    twin = pe.scale0_twin(route, table["routes"]) if bs.is_steered(route) else None
    if twin is not None:
        legs.append(("twin", twin))
    bound = []
    for role, leg in legs:
        path = cm.parse_serving_identity(leg.serving_identity).path
        entry = adapters.get(leg.cell_id)
        bound.append({"role": role, "cell_id": leg.cell_id, "virtual_model": leg.virtual_model, "path": path})
        if path == "lora":
            block = blocks.get(leg.cell_id)
            if not isinstance(entry, Mapping):
                problems.append(f"lora {role} {leg.cell_id!r}: no live adapter of that name (live adapters: "
                                f"{sorted(adapters)})")
                continue
            if not isinstance(block, Mapping):
                problems.append(f"lora {role} {leg.cell_id!r}: the routing table carries no adapter block for it "
                                f"(blocks: {sorted(blocks)}) — the live adapter's digests cannot be held to anything")
                continue
            if bool(block.get("petri_zero_adapter")) != bool(entry.get("petri_zero_adapter")):
                problems.append(f"lora {role} {leg.cell_id!r}.petri_zero_adapter: table "
                                f"{block.get('petri_zero_adapter')!r} != live {entry.get('petri_zero_adapter')!r}")
            if not block.get("petri_zero_adapter") and entry.get("petri_intervention_hash") != leg.intervention_hash:
                problems.append(f"lora {role} {leg.cell_id!r}.intervention_hash {leg.intervention_hash!r} != the live "
                                f"adapter's petri_intervention_hash {entry.get('petri_intervention_hash')!r} — rows "
                                "would be stamped with one intervention's hash while another's weights serve")
            for key in ("config_sha256", "weights_sha256"):
                if block.get(key) != entry.get(key):
                    problems.append(f"lora {role} {leg.cell_id!r}.{key}: table {block.get(key)!r} != live "
                                    f"{entry.get(key)!r}")
            continue
        if path != "passthrough":
            problems.append(f"{role} {leg.cell_id!r} carries serving path {path!r} — a native standup serves LORA "
                            "and PASSTHROUGH routes only (a lens route there would be base served under a steered "
                            "identity)")
            continue
        if isinstance(entry, Mapping):
            problems.append(f"passthrough {role} {leg.cell_id!r}: the live proxy serves an ADAPTER of that name "
                            f"(petri_intervention_hash {entry.get('petri_intervention_hash')!r}) — a passthrough leg "
                            "selects no adapter, so the record and the server disagree about what steers it")
        spec = prefix_by_vmodel.get(leg.virtual_model)
        if leg.si_prefix_hash is not None:
            if spec is None:
                problems.append(f"prompted {role} {leg.cell_id!r} carries si_prefix_hash {leg.si_prefix_hash!r} but the "
                                f"table has no prompted_prefixes spec for {leg.virtual_model!r} — the served prefix "
                                "cannot be bound to anything")
            elif getattr(spec, "sha16", None) != leg.si_prefix_hash:
                problems.append(f"prompted {role} {leg.cell_id!r}.si_prefix_hash {leg.si_prefix_hash!r} != the table's "
                                f"prompted_prefixes[{leg.virtual_model!r}].sha16 {getattr(spec, 'sha16', None)!r} — "
                                "the record disagrees with itself about the system prefix this leg serves")
        elif spec is not None:
            problems.append(f"passthrough {role} {leg.cell_id!r} carries NO si_prefix_hash but the table has a "
                            f"prompted_prefixes spec for {leg.virtual_model!r} — an unprompted leg with a prefix spec "
                            "is a mislabelled cell")
    if ident.get("base_model") != table.get("base_model"):
        problems.append(f"base_model: live {ident.get('base_model')!r} != the table's {table.get('base_model')!r}")
    if problems:
        raise EndpointBindingMismatch(
            f"[served-cap] {tc.REC_SERVING_ENGINE_MISMATCH}: the live native proxy's inventory does not bind the "
            f"route this run drives ({route.cell_id!r}) to the standup's records (routing_table.json + the "
            "attested engine block); refusing before any store is touched: " + "; ".join(problems))
    return {"held": True, "legs": bound, "base_model": ident.get("base_model")}


def _check_live_proxy_identity(client, endpoint: dict, served: dict, *, route=None, table=None) -> dict:
    """The LIVE proxy's ``/v1/models`` identity block (captured by
    ``ProxyClientEngine.load``, the authoritative source — the way
    ``petri_runner.require_agent_loop_entry`` reads it) must agree with what
    the standup record vouched for (N2, PW18): ``petri_agent_loop_entry``
    equals the record's ``agent_loop_entry`` (already hashed into the served
    identity), and a ``lens_reduction_mode`` the block states equals the
    identity's ``reduction_mode`` (the record and the block are two spellings
    of one server; a block that states none leaves the record's reading).
    No block at all is a proxy from before the multi-turn landing, or not the
    petri proxy — refused (:class:`EndpointBindingMismatch`) rather than
    scored under an identity nobody vouched for. Returns the block.

    ROUTE-AWARE: the two documents' ENGINE must agree on every run
    (``endpoint.json.engine`` against the block's ``engine.kind`` — both absent
    on the lens path, byte-identical);
    on a NATIVE record the block must be a native proxy's
    (``proxy_engine.served_engine_problem``: the engine block, NO
    ``lens_reduction_mode`` — no lens reader is consulted) and, given ``route``
    + ``table``, the run's route and its path-matched twin are BOUND to the
    live inventory (:func:`_hold_native_route_binding`: a lora route's adapter
    by intervention hash and digests, a passthrough route by the absence of an
    adapter and the H8 prefix spec). A lens record with ``route``/``table``
    given runs exactly as before."""
    ident = getattr(client, "proxy_identity", None)
    if not isinstance(ident, Mapping):
        raise EndpointBindingMismatch(
            f"[served-cap] the live proxy exposes no {tc.PROXY_IDENTITY_KEY!r} identity "
            f"block on /v1/models ({ident!r}) — cannot confirm the agent-loop entry / lens "
            "reduction mode the standup record vouches for (a build older than the "
            "multi-turn landing, or not the petri proxy); refusing (N2, PW18)")
    live_entry = ident.get(tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY)
    if live_entry != served[tc.AGENT_LOOP_ENTRY_KEY]:
        raise EndpointBindingMismatch(
            f"[served-cap] the live proxy reports {tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY}="
            f"{live_entry!r} but the standup record declares {STANDUP_ENTRY_KEY}="
            f"{endpoint.get(STANDUP_ENTRY_KEY)!r} — the record does not vouch for the "
            "proxy answering; refusing before any store is touched (N2, PW18)")
    # the ENGINE the two documents attest must agree for EVERY run —
    # `endpoint.json.engine` (absent on a lens record) against the live
    # block's `engine.kind` (absent on a lens proxy): None == None on the
    # lens path, byte-identical; lora_native == lora_native on the native
    # one; anything else is a proxy the record does not vouch for
    native = _endpoint_is_native(endpoint, source="standup record")
    live_engine = ident.get(tc.ENDPOINT_KEY_ENGINE)
    live_kind = live_engine.get("kind") if isinstance(live_engine, Mapping) else live_engine
    record_engine = endpoint.get(tc.ENDPOINT_KEY_ENGINE)
    if live_kind != record_engine:
        raise EndpointBindingMismatch(
            f"[served-cap] {tc.REC_SERVING_ENGINE_MISMATCH}: the live proxy's /v1/models "
            f"{tc.PROXY_IDENTITY_KEY}.{tc.ENDPOINT_KEY_ENGINE}.kind is {live_kind!r} but the standup record "
            f"attests {tc.ENDPOINT_KEY_ENGINE}={record_engine!r} — the record does not vouch for the proxy "
            "answering; refusing before any store is touched (native-serving-default plan §1.3)")
    if native:
        # a native proxy carries the engine block and NO lens_reduction_mode — the
        # ONE helper the wave drivers hold their proxies with; no lens reader runs
        why = pe.served_engine_problem(ident, expect=pe.ENGINE_EXPECT_LORA_NATIVE)
        if why is not None:
            raise EndpointBindingMismatch(
                f"[served-cap] {tc.REC_SERVING_ENGINE_MISMATCH}: {why} — the live proxy is not the native "
                "multi-LoRA standup the record attests; refusing before any store is touched")
    else:
        try:
            tc.resolve_reduction_mode(endpoint.get(STANDUP_REDUCTION_MODE_KEY),
                                      ident.get(tc.PROXY_IDENTITY_REDUCTION_MODE),
                                      sources=("standup record", "live proxy identity"))
        except ValueError as e:
            raise EndpointBindingMismatch(f"[served-cap] {e} (N2, PW18)") from None
        live_mode = ident.get(tc.PROXY_IDENTITY_REDUCTION_MODE)
        if live_mode is not None and live_mode != served[tc.REDUCTION_MODE_FIELD]:
            raise EndpointBindingMismatch(
                f"[served-cap] the live proxy reports {tc.PROXY_IDENTITY_REDUCTION_MODE}="
                f"{live_mode!r} but this run's served identity records "
                f"{tc.REDUCTION_MODE_FIELD}={served[tc.REDUCTION_MODE_FIELD]!r}; refusing (N2, PW18)")
    # fit-support transfer (contract §2.5.1, §2.5.7): the SAME per-cell
    # declarations map rides endpoint.json and /v1/models.petri_proxy — two
    # spellings of one server, compared whole; the record's own compare
    # (_check_serve_attestation) has already required the record's key
    if tc.PROXY_IDENTITY_SERVE_DECLARATIONS in endpoint:
        want, live = endpoint[tc.PROXY_IDENTITY_SERVE_DECLARATIONS], ident.get(
            tc.PROXY_IDENTITY_SERVE_DECLARATIONS)
        if live != want:
            raise EndpointBindingMismatch(
                f"[served-cap] the live proxy's {tc.PROXY_IDENTITY_KEY}."
                f"{tc.PROXY_IDENTITY_SERVE_DECLARATIONS} is {live!r} but the standup record "
                f"attests {want!r} — the record does not vouch for the proxy answering "
                "(contract §2.5.1: the same map in both); refusing before any store is touched")
    if native and route is not None:
        if table is None:
            raise EndpointBindingMismatch(
                "[served-cap] a native run's live-proxy check needs the loaded routing table to bind the "
                f"route {route.cell_id!r} (its adapter block / prefix spec); none was given")
        _hold_native_route_binding(ident, route=route, table=table)
    return dict(ident)


# --------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 — the driver's half of the contract: the #
# attestation compare, the header certification, the probe's body and the     #
# live prompt-budget pass — every literal from contracts, every refusal a REC  #
# --------------------------------------------------------------------------- #


def _check_serve_attestation(endpoint: dict, *, cell: str, arm: str, out_of_support: bool,
                             expected_object_sha256: "str | None",
                             budget: "Mapping | None", model_key: str,
                             off_cap_override: bool = False,
                             header_located: bool = True) -> dict:
    """Hold the standup record to the CONTRACT before any store is touched —
    a thin wrapper over :func:`contracts.check_serve_attestation`, the ONE
    compare sequence the launcher's worker (``oos_attestation_refusal``) runs
    too, so the two doors reach the same verdict on the same record through
    the same code (review round B, P2-2): the declarations map by own key (+
    the twin on a declared run); on a lens run the artifact digest(s) against
    :data:`contracts.FLAG_EXPECT_OBJECT_SHA256` (an out-of-support lens run
    told none refuses); ``lens_salt_policy``; the authorised psr leg's attested
    per-cell policy; on an out-of-support thinking-ON run (the base route's ON
    leg included) ``chat_template_hash`` against the prompt-budget artifact.
    ``out_of_support`` here is the caller's RECORD POSTURE (``held_to_record``:
    either override — the presence rules of the landed record), never a render
    fact; ``off_cap_override`` holds a comparator configuration to the
    artifact's template too.

    Returns the UNHASHED facts for the manifest ``route`` block. Only the TYPED
    refusals are caught: an :class:`contracts.OutOfSupportAttestationRefusal`
    exits with its REC (its ``str()`` starts with it); a bare ``ValueError``
    from a caller bug is never relabelled as a contract mismatch (the digest
    argument is validated at argparse, ``arm`` by the authorisation check that
    runs first)."""
    try:
        return tc.check_serve_attestation(
            endpoint, cell=cell, arm=arm, out_of_support=out_of_support, model_key=model_key,
            expected_object_sha256=expected_object_sha256, budget=budget,
            source=f"standup record {STANDUP_ENDPOINT_FILENAME}",
            off_cap_override=bool(off_cap_override), header_located=bool(header_located))
    except tc.OutOfSupportAttestationError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    except tc.CellNameError as e:
        raise SystemExit(f"[served-cap] {e}") from None


def _attestation_route_block(attest: "dict | None") -> dict:
    """The manifest ``route`` block's additive, UNHASHED attestation facts
    (contract §2.5.1 <ROUTE_KEY>: "copy the whole map"; no derived scalar):
    the raw ``lens_serve_declarations`` map, the served artifact digests, the
    salt policy and this cell's attested resumed-prefill policy. Null-valued
    when no attestation was compared (a legacy run)."""
    attest = attest or {}
    digest = attest.get("object_digest") or {}
    decl = attest.get("declarations") or {}
    out = {tc.ROUTE_KEY_SERVE_DECLARATIONS: decl.get("raw"),
           tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: digest.get("scalar"),
           tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: digest.get("by_cell"),
           tc.ENDPOINT_KEY_LENS_SALT_POLICY: attest.get(tc.ENDPOINT_KEY_LENS_SALT_POLICY),
           tc.ROUTE_KEY_RESUMED_PREFILL_POLICY: attest.get(tc.ROUTE_KEY_RESUMED_PREFILL_POLICY)}
    if attest.get(tc.LENS_SERVED_POSITIONS_KEY) is not None:
        # the served-positions fact ``check_serve_attestation`` resolved under the
        # cell's OWN key (the declared `@thinking` spelling on a declared run, the
        # bare id otherwise) rides the route block too — the same value the
        # driver's two-key lookup (``_standup_served_positions``) writes there on
        # an identity-v2 run. A record that STATES no value adds no key: the
        # identity-v1 route block never carried one (the golden manifests pin
        # that shape), and version 2 records the null itself at its write site
        out[tc.LENS_SERVED_POSITIONS_KEY] = list(attest[tc.LENS_SERVED_POSITIONS_KEY])
    return out


def _standup_served_positions(endpoint: "Mapping | None", *, cell: str) -> "tuple[list | None, str | None]":
    """The standup record's served-positions fact for THIS lens cell, read
    under the spelling the standup serves the cell as (contract §2.5.2 /
    §2.5.10 door table): a DECLARED standup keys ``lens_served_positions`` by
    the variant harness id (``<harness>@thinking``), an undeclared one by the
    bare harness id. Both spellings are looked up and EXACTLY ONE may be
    present. Returns ``(modes, key)``; ``(None, None)`` when neither names the
    cell, so the caller's fail-closed ``header_located(None)`` reading stands.
    Refuses by name (:data:`contracts.REC_OOS_DECLARATION_MISMATCH`) a map
    naming BOTH spellings (not one standup's shape) and a declared-key hit
    whose modes are NOT header-located (the standup declared a cell that no
    header-located mask asks it to declare).

    Before this lookup the driver read the BARE key only. On a declared standup
    that miss returned None — read fail-closed as header-located, so the serve
    proceeded on the declared path exactly as designed — and the None was
    written into the manifest three times (``lens_served_positions``, the route
    block, ``transfer_provenance.served_positions``) on every declared cell of
    the gsm8k-paired200 wave at 8f5f1495; the analysis recovers those four from
    the standup record beside each job (served-positions-null-diagnosis.md).

    ENGINE-NEUTRAL since native-positions-resolve: the record's
    ``served_positions`` map (:data:`contracts.SERVED_POSITIONS_KEY` — the lens
    legs' stated modes AND every lora route's attested ``petri_position_mode``,
    one shape, stamped by ``proxy.export_routing_table`` and copied into
    endpoint.json by both standups) is read FIRST under the same two-spelling
    discipline; the lens map is the FALLBACK for an older record that carries
    no neutral map. Before this the driver read the lens map ONLY, which a
    native standup writes empty, so every natively served fitted cell recorded
    ``served_positions: null`` and a transfer-provenance block whose
    ``header_located`` / ``header_check_class`` were derived from that null,
    fail closed (the GSM8K reader's ``RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY``);
    a route attesting no mode under either map still returns ``(None, None)``
    — never a default from the engine's name — and the reader refuses it by
    name. Both maps naming the cell must AGREE (one standup's two records)."""
    declared_key = tc.cell_spellings(cell, declared=True).harness
    bare_key = (declared_key[: -len(tc.OOS_STANDUP_CELL_SUFFIX)] if cell.endswith(tc.OOS_STANDUP_CELL_SUFFIX)
                else tc.cell_spellings(cell, declared=False).harness)
    found: "dict[str, tuple[str, list]]" = {}
    for map_key, reader in ((tc.SERVED_POSITIONS_KEY, tc.recorded_served_positions_of_cell),
                            (tc.LENS_SERVED_POSITIONS_KEY, tc.served_positions_of_cell)):
        hits = {k: reader(endpoint, k) for k in (declared_key, bare_key)}
        present = [k for k, v in hits.items() if v is not None]
        if len(present) == 2:
            raise SystemExit(
                f"[served-cap] {tc.REC_OOS_DECLARATION_MISMATCH}: the standup record's {map_key} names "
                f"cell {cell!r} under BOTH its declared spelling {declared_key!r} ({hits[declared_key]}) and its bare "
                f"spelling {bare_key!r} ({hits[bare_key]}) — one standup serves a cell under ONE spelling (contract "
                "§2.5.10); refusing rather than choosing")
        if present:
            found[map_key] = (present[0], list(hits[present[0]]))
    if not found:
        return None, None
    if len(found) == 2 and found[tc.SERVED_POSITIONS_KEY] != found[tc.LENS_SERVED_POSITIONS_KEY]:
        raise SystemExit(
            f"[served-cap] {tc.REC_OOS_DECLARATION_MISMATCH}: the standup record's {tc.SERVED_POSITIONS_KEY} map "
            f"({found[tc.SERVED_POSITIONS_KEY]}) and its {tc.LENS_SERVED_POSITIONS_KEY} map "
            f"({found[tc.LENS_SERVED_POSITIONS_KEY]}) disagree for cell {cell!r} — one standup's two records; "
            "refusing rather than choosing")
    key, modes = found.get(tc.SERVED_POSITIONS_KEY) or found[tc.LENS_SERVED_POSITIONS_KEY]
    unclassified = tc.unclassified_native_modes(modes)
    if unclassified:
        # a native-vocabulary mode the lens gate cannot classify: header_located(modes) would read it header-located, fail
        # closed, and the manifest would stamp a WRONG POSITIVE — the null bug one vocabulary over. Refuse by name instead;
        # nothing is stamped (whether such a mode locates a header is decided outside this gate)
        raise SystemExit(
            f"[served-cap] {tc.NATIVE_POSITION_MODE_UNCLASSIFIED}: {unclassified[0]!r} — the standup record resolves cell {cell!r} "
            f"({key!r}) to served positions {modes}, a native-vocabulary mode outside the served vocabulary "
            f"{list(tc.HEADER_LOCATED_MODES)} / {list(_proxy_position_modes())} that contracts.header_located would read as "
            "header-located (fail closed); refusing rather than stamp a header fact the gate cannot derive")
    if key == declared_key and not tc.header_located(modes):
        raise SystemExit(
            f"[served-cap] {tc.REC_OOS_DECLARATION_MISMATCH}: the standup serves cell {cell!r} DECLARED "
            f"({declared_key!r}) with served positions {modes} that are not header-located "
            f"({list(tc.HEADER_LOCATED_MODES)}) — a declaration is required by, and only by, a header-located "
            "mask (plan §4.3 item 1b); the standup declared a cell that needs none")
    return modes, key


def _proxy_position_modes() -> "tuple[str, ...]":
    """The lens gate's served vocabulary (``proxy.POSITION_MODES``), through the contracts module's own proxy import."""
    return tuple(tc._proxy.POSITION_MODES)


def _models_payload_of(client) -> "dict | None":
    """The ``/v1/models`` payload as the contract's ONE ``thinking_header``
    source reads it: ``ProxyClientEngine.load`` captured the ``petri_proxy``
    identity block off that listing, so the payload is rebuilt around it under
    :data:`contracts.PROXY_IDENTITY_KEY` (== ``THINKING_HEADER_SOURCE[0]``)."""
    ident = getattr(client, "proxy_identity", None)
    return {tc.PROXY_IDENTITY_KEY: dict(ident)} if isinstance(ident, Mapping) else None


def _header_measured_live(header_arms: Mapping) -> bool:
    """Whether the header block's ids stand as LIVE for the certifications' "measured live" check: wholly served,
    or the mix — the ON arm served and the OFF arm a local REFERENCE the block records as accepted under the
    ON-pinned rule (``off_arm_accepted_under`` set). The block's own ``source`` / ``source_by_arm`` still label the
    mix; this is the reading, not a relabel. A wholly local measurement, a mix the rule did not admit, or a block
    whose measurement ERRORED (``error`` set — its ``source`` names the intended source, not a measurement) is not live."""
    if header_arms.get("error") is not None:
        return False
    source = header_arms.get(tc.OOS_EVIDENCE_HEADER_SOURCE)
    if source == HEADER_SOURCE_SERVED:
        return True
    return (source == HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
            and header_arms.get(HEADER_KEY_OFF_ARM_ACCEPTED_UNDER) is not None)


def _certify_header(header_arms: dict, client, *, family: object = None) -> "tuple[dict, str | None]":
    """Certify the measured per-arm header against the family's
    declared render fact and the live proxy's attested
    ``/v1/models.petri_proxy.thinking_header`` (contract §2.5.1, §2.5.7,
    §2.5.12) through :func:`contracts.certify_thinking_header`. Returns the
    evidence fields to merge into ``thinking_header_by_arm`` and the refusal
    text (``None`` when certified): every mismatch is recorded FIRST (all three
    certification flags false, every list and attested value with its type),
    then the caller exits — the record first, the refusal second."""
    payload = _models_payload_of(client)
    # the block VERBATIM (contracts.raw_thinking_header): the certification
    # judges its shape itself, so a non-mapping block yields the same COMPLETE
    # evidence block as any other mismatch (every OOS_EVIDENCE_* key, typed
    # nulls, the problem named) — never a partial one
    try:
        return tc.certify_thinking_header(
            header_ids_by_arm=header_arms.get(tc.OOS_EVIDENCE_HEADER_IDS_BY_ARM) or {},
            measured_live=_header_measured_live(header_arms),
            attested=tc.raw_thinking_header(payload),
            attested_error=tc.attested_thinking_header_error(payload),
            # the family's expected id tuples (the Qwen constants when unknown)
            family=(family if (family is not None and tc.render_fact_for(family) is not None) else None)), None
    except tc.ThinkingHeaderCertificationError as e:
        return dict(e.evidence), f"{tc.REC_THINKING_UNSERVABLE}: {e}"


def _probe_base_route_refuses_declaration(chat, base_vmodel: str) -> "tuple[dict, str | None]":
    """The base route's refusal of the declaration key,
    VERIFIED rather than assumed (spec: "the base route keeps refusing the
    declaration key"): ONE short body to the base vmodel — ``max_tokens 1``,
    greedy, seeded — carrying ``render_fragment(lens=True, arm=on)`` (the ON
    ``chat_template_kwargs`` + ``petri_thinking_on: true``); the landed proxy
    must answer 400 ``thinking_on_not_lens`` (contract §2.5.5, ``proxy.py``)
    BEFORE any forward. Returns ``(record, error)``: the record is the
    evidence block :data:`contracts.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED`
    (``vmodel``, ``status``, ``code``, ``ok``, plus the reply's ``message``);
    the error names :data:`contracts.REC_OOS_DECLARATION_MISMATCH` when the proxy
    forwarded the body (a 2xx) or refused it under any other status / code —
    a lens-side divergence, refused before generation, never re-rolled. Run on
    the comparator's ``none`` configuration only (the base route; gated on the
    comparator flag, never the record posture — the 2x2's legs are frozen)."""
    from sisteer.capeval.serving import proxy as _px

    body = {"model": base_vmodel,
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "temperature": 0.0, "max_tokens": 1, "seed": 0,
            **tc.render_fragment(lens=True, arm=tc.THINKING_ON)}
    rec: dict = {"vmodel": base_vmodel, "status": None, "code": None, "message": None, "ok": False,
                 "expected": {"status": 400, "code": _px.THINKING_ON_NOT_LENS}}
    try:
        reply = chat(body)
    except _px.ProxyError as e:
        m = re.search(r"returned (\d{3}): (.*)\Z", str(e), flags=re.DOTALL)
        if m is None:
            rec["message"] = str(e)[:500]
            return rec, (f"{tc.REC_OOS_DECLARATION_MISMATCH}: the base route's declaration probe "
                         f"({base_vmodel}) failed without an HTTP status: {e}")
        rec["status"] = int(m.group(1))
        try:
            doc = json.loads(m.group(2))
            err = doc.get("error") if isinstance(doc, Mapping) else None
            rec["code"] = err.get("code") if isinstance(err, Mapping) else None
            rec["message"] = err.get("message") if isinstance(err, Mapping) else m.group(2)[:500]
        except ValueError:
            rec["message"] = m.group(2)[:500]
    else:
        rec["status"] = 200
        rec["message"] = json.dumps(reply, sort_keys=True)[:500]
    rec["ok"] = rec["status"] == 400 and rec["code"] == _px.THINKING_ON_NOT_LENS
    if rec["ok"]:
        return rec, None
    return rec, (f"{tc.REC_OOS_DECLARATION_MISMATCH}: the base route {base_vmodel!r} did not refuse a "
                 f"{tc.OOS_THINKING_DECLARATION_KEY} declaration with 400 {_px.THINKING_ON_NOT_LENS!r} "
                 f"(contract §2.5.5) — it answered status {rec['status']}, code {rec['code']!r}; a "
                 "base route that forwards or otherwise refuses a declared body is a lens-side "
                 "divergence; refusing before any generation (no re-roll)")


def _probe_extra_body(render_arm: "str | None", *, header_located: bool = True) -> "dict | None":
    """What the steering-bites probe adds to EVERY payload on
    BOTH legs (contract §2.5.3, "ONE shape"): the arm's ``chat_template_kwargs``
    (explicit ``enable_thinking: false`` on the OFF probe, so its stamp reads
    ``thinking_off`` — §2.5.11 rows 3–4) plus, on a thinking-ON standup of a
    HEADER-LOCATED lens cell, the ``petri_thinking_on: true`` declaration —
    the twin is a declared lens cell there. ``header_located``: an undeclared
    all-position cell's probe sends NO declaration on either leg (the driver
    also renders that probe thinking-OFF — the probe measures whether the
    object BITES, not the render).
    ``None`` for a legacy run with no thinking arm (byte-identical probe).
    Refuses BEFORE sending should the declaration constants read ``None`` (an
    ON probe without its declaration would certify the OFF render)."""
    if render_arm is None:
        return None
    try:
        return tc.render_fragment(lens=True, arm=render_arm, header_located=bool(header_located))
    except tc.OutOfSupportDeclarationUnreconciled as e:
        raise SystemExit(
            f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: the declaration constants are not landed "
            f"— an ON probe would certify the OFF render ({e})") from None


class _RequestEvidence:
    """The versioned request-evidence writer: ONE record per HTTP ATTEMPT —
    Inspect task calls and driver probes alike — appended to
    ``<out_root>/request_evidence.jsonl``
    (``contracts.REQUEST_EVIDENCE_FIELDS``, version 2), written FROM THE HTTP
    BOUNDARY: the Inspect provider's httpx request hook and the sync probe
    client's ``on_request`` hook each hand over the EXACT bytes and headers
    about to go on the wire, after the attempt's ``X-Tinycap-Request-Id`` was
    minted onto them. The usage ledger writes the same id and the same digest
    on its row, so the two inventories reconcile by identity
    (``contracts.request_evidence_reconciliation``). Never buffered: a run
    that dies mid-way leaves every record it sent."""

    def __init__(self, path: Path, *, attempt_id: "str | None" = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self._probe_counts: dict = {}
        #: the attempt this driver invocation runs as — stamped on EVERY
        #: record it writes (appended after the version-2 fields; None = no
        #: stamp, the record byte-identical)
        self.attempt_id = attempt_id

    def record_wire(self, raw: bytes, *, headers: Mapping) -> dict:
        """The one writer — the Inspect provider's wire hook calls it as its
        sink, the sync probe client through :meth:`attach`: the record of one
        attempt from its wire bytes + headers."""
        rec = tc.request_evidence_record(raw, headers=headers)
        if self.attempt_id is not None:
            rec = {**rec, tc.REQUEST_EVIDENCE_ATTEMPT_FIELD: self.attempt_id}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.n += 1
        return rec

    # -- the sync probe client -------------------------------------------------
    def attach(self, client) -> None:
        """Observe ``client``'s HTTP boundary: every ``/chat/completions`` POST it
        sends is recorded from the exact bytes (``ProxyClientEngine.on_request``)."""
        def on_request(url: str, data: bytes, headers: Mapping) -> None:
            if str(url).rstrip("/").endswith("/chat/completions"):
                self.record_wire(data, headers=headers)
        client.on_request = on_request

    def labelled(self, client, kind: str):
        """A ``chat(payload)`` callable over ``client`` whose every call is a
        LABELLED probe attempt (FC4 (2)): ``X-Tinycap-Task: probe/<kind>``,
        ``X-Tinycap-Sample: <vmodel>/<n>`` (n per kind and route, from 0) and a
        fresh ``X-Tinycap-Request-Id`` — so the ledger rows it and the record
        written at the client's boundary carries the same labels and id."""
        label = tc.probe_task_label(kind)

        def chat(payload):
            vmodel = str(payload.get("model")) if isinstance(payload, Mapping) else "?"
            n = self._probe_counts.get((kind, vmodel), 0)
            self._probe_counts[(kind, vmodel)] = n + 1
            headers = {tc.HEADER_TASK: label, tc.HEADER_SAMPLE: f"{vmodel}/{n}",
                       tc.HEADER_REQUEST_ID: uuid.uuid4().hex}
            return client.chat_completions(payload, headers=headers)
        return chat

    def records(self) -> list:
        return _jsonl_rows(self.path)


def _jsonl_rows(path) -> list:
    """Every record of a JSONL file — the usage ledger (``usage_ledger.jsonl``)
    and the request-evidence file — or ``[]`` when there is none: the post-run
    reader of the NAMED proxy refusals and of the request-evidence gate."""
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _utc_now() -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` — the one timestamp shape the driver's records wear."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mapping_get(block, key: str):
    """``block[key]`` when ``block`` is a mapping, else None (a manifest block read defensively)."""
    return block.get(key) if isinstance(block, Mapping) else None


def _check_served_family(model_id, model_revision, *, family, model_key) -> None:
    """Refuse a run whose SERVED model's family (``vllm_server.model_family`` = the served config's ``model_type``) is
    not the sampling model key's: the render fact the run is planned and certified against is the key's family's, and
    it must be the server's. Shared by the thinking-ON and the no-switch header measurements.

    The served ``model_type`` is compared to the key's family BY NAME: a model_type that is not the family refuses — never a
    prefix guess, never a fold (a served architecture joins a family's render fact only through its own measured header)."""
    from sisteer.capeval.serving import vllm_server as _vllm_server
    served_model_type = _vllm_server.model_family(model_id, model_revision)
    served_family = None if served_model_type is None else str(served_model_type)
    if served_family is not None and served_family != family:
        raise SystemExit(f"[served-cap] the served model {model_id!r} is family {served_family!r} (the config's model_type) "
                         f"but the sampling model key {model_key!r} names family {family!r} — refusing to read one "
                         "family's render fact for another's server")


def _probe_reply(resp) -> dict:
    """ONE proxy chat completion as the four-key reply record the predicates
    read (``contracts.lens_on_bite_verdict``; ``render_certified`` rule 3): the
    first choice's content and reasoning channel, its finish reason and the
    response stamp's ``render_regime`` — each None / empty where the response
    does not carry it."""
    stamp = resp.get(tc.STAMP_BLOCK_KEY) if isinstance(resp, Mapping) else None
    choice = (resp.get("choices") or [{}])[0] if isinstance(resp, Mapping) else {}
    msg = choice.get("message") or {}
    # the server-separated reasoning channel rides as `reasoning_content` on older OpenAI-compatible servers and as
    # `reasoning` on this vLLM (0.25.x) — read both, as the diagnostic smoke's reader does (gemma's certified ON probe
    # answered "43" with its reasoning under `reasoning`, and a reader of `reasoning_content` alone saw none)
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
            "finish_reason": choice.get("finish_reason"),
            "render_regime": (stamp.get(tc.STAMP_KEY_RENDER_REGIME) if isinstance(stamp, Mapping) else None)}


#: predicate (A)'s render probe: a SUBSTANTIVE one-line arithmetic question — a thinking model opens a short
#: reasoning block and answers, so the ON reply carries the family's think shape within the budget; a one-word
#: instruction at temperature 0 degenerated into a channel-less whitespace loop on gemma-4 at 64 tokens. The budgets:
#: 512 tokens, one retry at 2,048 when the ON reply is cut at the cap with neither channel.
RENDER_PROBE_PROMPT = "What is 17 + 26? Reply with just the number."
RENDER_PROBE_MAX_TOKENS = 512
RENDER_PROBE_RETRY_MAX_TOKENS = 2048
#: the INCONCLUSIVE-draw retry prompt: gemma-4 answered the arithmetic question "43" with NO thought block and
#: finish_reason `stop` — the MODEL declining to think on a trivial prompt (the diagnostic smoke's `no_thinking_emitted`,
#: ~1 in 24 real MATH draws at temperature 1.0), not a render defect. The retry asks a problem that takes several
#: steps (four products, two sums, a difference: 9*8 + 11*15 - (12*8 + 7*15) = 36) and STILL asks for a bare number, so
#: the working has nowhere to go but the thought channel; the model can only decline it by answering blind.
RENDER_PROBE_REASONING_PROMPT = ("A bookshop sells paperbacks for 8 dollars each and hardcovers for 15 dollars each. "
                                 "On Monday it sold 12 paperbacks and 7 hardcovers; on Tuesday it sold 9 paperbacks "
                                 "and 11 hardcovers. How many dollars more did the shop take in on Tuesday than on "
                                 "Monday? Reply with just the number.")
#: ON draws per probe, all told: the first (arithmetic, seed 0), then up to two more — the budget retry of a channel-less
#: cap-cut (same prompt and seed, 2,048 tokens) and/or the reasoning prompt after a declined draw (a fresh seed each)
RENDER_PROBE_MAX_ATTEMPTS = 3
RENDER_PROBE_PROMPT_ARITHMETIC = "arithmetic"
RENDER_PROBE_PROMPT_REASONING = "reasoning"
RENDER_PROBE_PROMPTS = {RENDER_PROBE_PROMPT_ARITHMETIC: RENDER_PROBE_PROMPT,
                        RENDER_PROBE_PROMPT_REASONING: RENDER_PROBE_REASONING_PROMPT}
#: what ONE ON probe draw says about the family's render (:func:`_classify_on_probe_reply`) — the retry policy keys on
#: it; the certification (``contracts.render_certified``) stays the one reader that PASSES or REFUSES
PROBE_CLASS_THINK_SHAPE = "think_shape"                  # carries the family's think shape -> the certifying draw
PROBE_CLASS_LEAKED_TAG = "leaked_tag"                    # a raw think tag where the render forbids one -> stands, no retry
PROBE_CLASS_NO_THINKING_EMITTED = "no_thinking_emitted"  # finished, no reasoning, no tag: the MODEL declined -> retry
PROBE_CLASS_UNCLOSED_BLOCK = "unclosed_block"            # implicit_open: finished INSIDE the prompt-opened block, no close -> retry
PROBE_CLASS_AT_CAP_CHANNEL_LESS = "at_cap_channel_less"  # cut at the cap with NEITHER channel -> the budget retry
PROBE_CLASS_NO_THINK_SHAPE = "no_think_shape"            # shape absent and no decline to blame -> stands, no retry
PROBE_CLASS_NO_RENDER_FACT = "no_render_fact"            # no family fact to read the shape against -> stands
PROBE_CLASSES = (PROBE_CLASS_THINK_SHAPE, PROBE_CLASS_LEAKED_TAG, PROBE_CLASS_NO_THINKING_EMITTED,
                 PROBE_CLASS_UNCLOSED_BLOCK, PROBE_CLASS_AT_CAP_CHANNEL_LESS, PROBE_CLASS_NO_THINK_SHAPE,
                 PROBE_CLASS_NO_RENDER_FACT)
#: the INCONCLUSIVE classes the ON arm re-draws after (the reasoning prompt, a fresh seed): the model's per-draw
#: choice NOT to emit the shape — declining to open the block where the MODEL opens it (record_only / explicit_tags),
#: or ending its turn inside the block the PROMPT opened without closing it (implicit_open; gsm8k-paired200 rows 27 /
#: 30: nemotron answered "17 plus 26 is 43. 43." then `stop` with no `</think>`, and the one-draw probe refused a
#: render whose prompt-side header predicate (A) rule 1 had just certified live)
PROBE_DECLINED_CLASSES = (PROBE_CLASS_NO_THINKING_EMITTED, PROBE_CLASS_UNCLOSED_BLOCK)


def _classify_on_probe_reply(fact, rec: Mapping) -> str:
    """ONE ON probe draw's class (:data:`PROBE_CLASSES`) under the family's render fact — the shape itself is read by
    ``contracts._think_shape``, the SAME rule ``render_certified`` applies to the certifying draw, so the retry policy
    can never call a draw conclusive that the certification would not.

    Precedence: cut at the cap with NEITHER channel (no text at all: inconclusive) -> a LEAKED tag (record_only:
    any think delimiter in content, the server parser missed a block; implicit_open: a stray opening tag inside the
    prompt-opened block) -> the shape present -> a finished draw with no reasoning and no tag under a render where the
    MODEL opens the block (record_only, explicit_tags): the model declined (inconclusive) -> a finished
    implicit-open draw with no close tag: the model ended its turn INSIDE the block the prompt opened
    (:data:`PROBE_CLASS_UNCLOSED_BLOCK`, inconclusive — the prompt-side render is certified by rule 1's live header
    ids, and the close tag is the model's per-draw choice exactly as the open tag is under explicit_tags; an at-cap
    unclosed block is already the shape) -> anything else (a cap-cut that carried text but no shape; an explicit-tags
    close without an open): the shape is absent and stands."""
    content = str(rec.get("content") or "")
    reasoning = rec.get("reasoning")
    finish = rec.get("finish_reason")
    at_cap = finish in tc.AT_CAP_FINISH_REASONS
    if at_cap and not content.strip() and not (reasoning or "").strip():
        # nothing in either channel (whitespace is nothing: the gemma probe looped on whitespace at the cap)
        return PROBE_CLASS_AT_CAP_CHANNEL_LESS
    if fact is None:
        return PROBE_CLASS_NO_RENDER_FACT
    present, stray_open, _why = tc._think_shape(fact, content=content, reasoning=reasoning, finish_reason=finish)
    has_tag = fact.think_open in content or fact.think_close in content
    if (fact.split_mode == tc.THINK_SPLIT_RECORD_ONLY and has_tag) or stray_open:
        return PROBE_CLASS_LEAKED_TAG
    if present:
        return PROBE_CLASS_THINK_SHAPE
    if fact.split_mode in (tc.THINK_SPLIT_RECORD_ONLY, tc.THINK_SPLIT_EXPLICIT_TAGS) and not at_cap and not has_tag:
        return PROBE_CLASS_NO_THINKING_EMITTED
    if fact.split_mode == tc.THINK_SPLIT_IMPLICIT_OPEN and not at_cap:
        # not present under implicit_open == not closed and not cut at the cap: the draw finished inside the block
        return PROBE_CLASS_UNCLOSED_BLOCK
    return PROBE_CLASS_NO_THINK_SHAPE


def _probe_decode(sampling_row) -> dict:
    """A live probe's decode block: the cell's frozen :class:`contracts.SamplingRow` (its temperature plus
    every optional term the row sets), or the legacy greedy probe (temperature 0) when the caller passes none.
    Shared by :func:`_render_probe` and :func:`_no_switch_render_probe`."""
    if sampling_row is None:
        return {"temperature": 0.0}
    decode = {"temperature": float(sampling_row.temperature)}
    for key in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        val = getattr(sampling_row, key, None)
        if val is not None:
            decode[key] = val
    return decode


def _render_probe(chat, vmodel: str, *, lens: bool, header_located: bool, system=None, sampling_row=None,
                  family=None, standup_thinking_mode=None) -> dict:
    """PREDICATE (A)'s live probes: ONE 1-item greedy completion on the
    cell's route under EACH arm — explicit JSON booleans, the declaration
    only where a HEADER-LOCATED lens cell needs it — returning the
    ``render_regime`` stamp per arm and each arm's reply record
    (:func:`_probe_reply`; the ON one feeds the think-shape check). The base
    route accepts an undeclared ON kwarg; an undeclared all-position lens cell
    does too (Evals' refinement); a refusal here is a NAMED proxy refusal,
    surfaced as the driver's.

    ``standup_thinking_mode`` is the standup record's attested
    ``thinking_mode`` (``endpoint.json``, :data:`contracts.STANDUP_THINKING_MODE_KEY`
    — read off the bound record, never argv). On a standup attesting ``on`` the
    proxy refuses the OFF render BY DESIGN (``400 thinking_regime_mismatch`` —
    direct from chat, or from the usage ledger off its pre-tokenize with
    ``via: ledger_tokenize``), so that EXACT typed refusal to the deliberate OFF
    probe is recorded as the OFF-arm evidence: ``regimes[off] ==
    contracts.PROBE_REGIME_REFUSED_REGIME_MISMATCH``, ``responses[off]`` a reply
    record with the refusal beside it, and ``off_arm_refusal`` the refusal
    body (:func:`contracts.off_arm_refusal_evidence`). Any OTHER ProxyError on
    either arm — a 502, an untyped 400, another code — still exits
    ``THINKING_UNSERVABLE``; so does the typed refusal when the record attests
    anything but ``on`` or carries no ``thinking_mode`` (the OFF arm must then be
    measured; :func:`contracts.off_arm_refusal_accepted` names why). The ON arm
    is always measured through the proxy.

    ``system`` is the cell's SI text for a PROMPTED (or stacked) cell, exactly
    as :func:`_chat_payload` sends it on every real request: the proxy verifies
    the first message of every request on such a route against the registered
    SI prefix (H8) and REFUSES a bare user turn — the first production prompted
    cell died on this probe (math500-thinking-on-full22). None for a bare
    or object cell, whose routes carry no prefix and would refuse one.

    ``sampling_row`` is the cell's frozen :class:`contracts.SamplingRow`:
    the probe decodes under the SAME temperature / top_p / top_k (and the
    optional penalty terms when the row sets them) as the production draws,
    seeded — argmax decoding on a steered route looped on the opening think
    tag (qwen's object cell: ``<think>\n<think>\n…`` for 64 tokens), a
    degenerate draw the production sampling row never produces. None keeps
    the legacy greedy probe (temperature 0).

    ``family`` selects the render fact the ON draws are CLASSIFIED against
    (:func:`_classify_on_probe_reply`); the ON arm draws up to
    :data:`RENDER_PROBE_MAX_ATTEMPTS` times — a channel-less cap-cut re-sends the
    same prompt and seed at :data:`RENDER_PROBE_RETRY_MAX_TOKENS` (the same
    draw with room to close its block), a DECLINED draw (finished, no reasoning,
    no tag) or an UNCLOSED implicit-open draw (finished inside the prompt-opened
    block with no close tag — :data:`PROBE_DECLINED_CLASSES`) re-asks with
    :data:`RENDER_PROBE_REASONING_PROMPT` under a fresh seed (the draw's index) —
    and stops at the first draw that is conclusive: the shape present, a leaked
    tag, or a shape absent that no such choice explains.
    Every ON draw is recorded in ``on_attempts`` (prompt id, seed, budget,
    finish reason, channel lengths, class); ``responses[on]`` is the LAST draw,
    the one the certification reads. The OFF arm is one draw, as ever. With no
    ``family`` (or a family without a fact) only the cap-cut retry applies and
    the certification refuses the missing fact by name."""
    from sisteer.capeval.serving import proxy as _px

    decode = _probe_decode(sampling_row)

    fact = tc.render_fact_for(family)
    out: dict = {"regimes": {}, "responses": {}, "on_attempts": [], "off_arm_refusal": None}
    for arm in (tc.THINKING_OFF, tc.THINKING_ON):
        prompt_id, max_tokens, seed = RENDER_PROBE_PROMPT_ARITHMETIC, RENDER_PROBE_MAX_TOKENS, 0
        rec = None
        for attempt in range(RENDER_PROBE_MAX_ATTEMPTS):
            messages = ([{"role": "system", "content": system}] if system else []) + \
                [{"role": "user", "content": RENDER_PROBE_PROMPTS[prompt_id]}]
            body = {"model": vmodel, "messages": messages, **decode, "max_tokens": max_tokens, "seed": seed,
                    **tc.render_fragment(lens=lens, arm=arm, header_located=bool(header_located))}
            try:
                resp = chat(body)
            except _px.ProxyError as e:
                typed = _off_arm_typed_refusal(e) if arm == tc.THINKING_OFF else None
                if typed is not None:
                    # the deliberate OFF probe drew the proxy's BY-DESIGN refusal of an OFF render on a thinking-ON
                    # standup. Evidence ONLY when the bound standup record attests thinking_mode 'on' — else by name
                    accepted, why_not = tc.off_arm_refusal_accepted(typed, standup_thinking_mode=standup_thinking_mode)
                    if not accepted:
                        raise SystemExit(
                            f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: predicate (A)'s {arm!r} render probe on "
                            f"{vmodel!r} was REFUSED by the proxy as a thinking-ON standup's OFF render "
                            f"({typed['status']} {typed['code']}"
                            f"{', via ' + typed['via'] if typed.get('via') else ''}), and that refusal is not admissible "
                            f"as OFF-arm evidence here: {why_not} — a named refusal, never worked around ({e})") from None
                    print(f"[served-cap] predicate (A) OFF render probe on {vmodel!r}: the proxy refused the OFF render "
                          f"{typed['status']} {typed['code']}{' via ' + typed['via'] if typed.get('via') else ''} — the "
                          f"standup attests {tc.STANDUP_THINKING_MODE_KEY} '{tc.THINKING_ON}', so the typed refusal is "
                          "the OFF-arm evidence (D105); the ON arm is measured next", flush=True)
                    rec = {"content": "", "reasoning": None, "finish_reason": None,
                           "render_regime": tc.PROBE_REGIME_REFUSED_REGIME_MISMATCH, "refusal": typed}
                    out["off_arm_refusal"] = typed
                    break
                raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: predicate (A)'s {arm!r} render probe on "
                                 f"{vmodel!r} was REFUSED by the proxy ({e}) — a named refusal, never worked around") from None
            rec = _probe_reply(resp)
            if arm == tc.THINKING_OFF:
                break
            cls = _classify_on_probe_reply(fact, rec)
            out["on_attempts"].append({
                "attempt": attempt + 1, "prompt_id": prompt_id, "seed": seed, "max_tokens": max_tokens,
                "finish_reason": rec["finish_reason"], "reasoning_chars": len(rec["reasoning"] or ""),
                "content_chars": len(rec["content"] or ""), "class": cls})
            draws_left = attempt + 1 < RENDER_PROBE_MAX_ATTEMPTS
            if cls == PROBE_CLASS_AT_CAP_CHANNEL_LESS and max_tokens < RENDER_PROBE_RETRY_MAX_TOKENS and draws_left:
                # An ON reply cut at the cap with NEITHER channel is INCONCLUSIVE about the render (gemma's greedy
                # 64-token probe on a one-word instruction came back channel-less — the parser flushes an open thought
                # block at the cap, so "nothing in either channel" is a degenerate draw, not a shape). One retry with a
                # larger budget, the SAME prompt and seed; a second channel-less cap-cut then stands and the
                # certification refuses it by name.
                print(f"[served-cap] predicate (A) ON render probe on {vmodel!r}: cut at the cap ({max_tokens}) with "
                      f"neither channel — retrying once at {RENDER_PROBE_RETRY_MAX_TOKENS}", flush=True)
                max_tokens = RENDER_PROBE_RETRY_MAX_TOKENS
                continue
            if cls in PROBE_DECLINED_CLASSES and draws_left:
                # A FINISHED ON reply with no reasoning and no tag is the MODEL declining to think on that draw
                # (gemma-4 answered "43" with no thought block; the smoke reads the same as no_thinking_emitted,
                # inconclusive, never a failure) — re-ask with a problem that needs the working, under a fresh seed.
                # The same for an implicit-open draw that FINISHED inside the prompt-opened block without closing it
                # (nemotron, gsm8k-paired200 rows 27 / 30: the template opened `<think>`, the model answered in the
                # block and emitted `stop` with no `</think>` — a per-draw choice under the sampling row, not a render
                # defect; rule 1 certified the prompt-side header live). Either way the certification still reads the
                # LAST draw and refuses unless it carries the shape.
                what = ("finished with no thought block — the model declined" if cls == PROBE_CLASS_NO_THINKING_EMITTED
                        else f"finished inside the prompt-opened block without {fact.think_close!r} — the model ended "
                             "its turn in the block")
                print(f"[served-cap] predicate (A) ON render probe on {vmodel!r}: draw {attempt + 1} "
                      f"({prompt_id!r}, seed {seed}) {what}; retrying with the reasoning prompt (seed {attempt + 1})",
                      flush=True)
                prompt_id, max_tokens, seed = RENDER_PROBE_PROMPT_REASONING, RENDER_PROBE_MAX_TOKENS, attempt + 1
                continue
            break
        out["regimes"][arm] = rec["render_regime"]
        out["responses"][arm] = rec
    return out


def _render_probe_declined_note(probe: Mapping) -> str:
    """The clause the render refusal gains when the CERTIFYING ON draw is the model's per-draw choice not to emit
    the shape (:data:`PROBE_DECLINED_CLASSES`): declining to open a thought block
    (:data:`PROBE_CLASS_NO_THINKING_EMITTED`), or ending its turn inside the prompt-opened block without closing it
    (:data:`PROBE_CLASS_UNCLOSED_BLOCK`) — how many draws, of how many, on which prompts — so the refusal reads as
    what it is and the artifact can show it. Empty otherwise."""
    attempts = list(probe.get("on_attempts") or [])
    if not attempts or attempts[-1].get("class") not in PROBE_DECLINED_CLASSES:
        return ""
    cls = attempts[-1].get("class")
    declined = [a for a in attempts if a.get("class") == cls]
    count = (f"{len(declined)} draws" if len(declined) == len(attempts)
             else f"{len(declined)} of {len(attempts)} draws")
    prompt_ids = list(dict.fromkeys(str(a.get("prompt_id")) for a in attempts))
    prompts = "; ".join(f"{pid}: {RENDER_PROBE_PROMPTS.get(pid, '?')!r}" for pid in prompt_ids)
    if cls == PROBE_CLASS_UNCLOSED_BLOCK:
        return (f" — the model ended its turn inside the prompt-opened thought block without closing it on {count} "
                f"(each finished with no close tag and no stray open tag; prompts {prompts}): the prompt-side render "
                "is certified by the live header ids (rule 1), and leaving the block unclosed is the model's choice on "
                "the draw under the sampling row, not a render defect — the probe has no draws left")
    return (f" — the model declined to open a thought block on {count} (each finished with no reasoning channel and no "
            f"think tag; prompts {prompts}): a decline is the model's choice on the draw, not a render defect (D115, "
            f"D143), and the probe has no draws left")


#: the evidence field naming the CERTIFYING ON draw's class (:data:`PROBE_CLASSES`) — the certification passes only
#: when it is :data:`PROBE_CLASS_THINK_SHAPE`
CERTIFYING_DRAW_CLASS_FIELD = "certifying_draw_class"


def _render_certification(probe: Mapping, *, family, header_ids_by_arm: Mapping, measured_live: bool,
                          header_source_by_arm: "Mapping | None" = None, standup_thinking_mode=None
                          ) -> tuple[dict, str | None]:
    """PREDICATE (A)'s verdict on a :func:`_render_probe` result: ``contracts.render_certified`` over the per-arm
    regime stamps and the CERTIFYING ON draw (``responses[on]``, the last one), the declined note
    (:func:`_render_probe_declined_note`) appended to a refusal. Returns ``(record, error)`` — the record is the
    certification's own (every flag false and every problem named on a refusal), written before the refusal.

    ``header_source_by_arm`` is the header block's per-arm provenance (:data:`HEADER_KEY_SOURCE_BY_ARM`),
    ``standup_thinking_mode`` the bound record's attested ``thinking_mode``, and the probe's ``off_arm_refusal`` (the
    typed OFF refusal, when the OFF arm was refused by design) rides into the certification — which accepts the
    three together ONLY on a standup attesting ``on`` and records every acceptance (``off_arm_evidence``,
    ``off_arm_refusal``, ``off_header_source``, ``off_header_role``, ``off_arm_accepted_under``, ``render_source_by_arm``).

    THE ACCEPTING CLASS IS THE ONLY PATH: the certifying draw's recorded class (``on_attempts[-1].class``,
    :func:`_classify_on_probe_reply`) must be :data:`PROBE_CLASS_THINK_SHAPE`. ``contracts._think_shape`` reads an
    implicit-open completion cut at the cap as "inside the block" whatever the channels hold, so a probe whose
    last draw was a channel-less cap-cut (:data:`PROBE_CLASS_AT_CAP_CHANNEL_LESS` — nothing emitted) would
    otherwise certify the render on NO text: reachable on an implicit-open family once an unclosed first draw is
    retried. A probe with no recorded ON draw certifies nothing either."""
    attempts = list(probe.get("on_attempts") or [])
    final_cls = attempts[-1].get("class") if attempts and isinstance(attempts[-1], Mapping) else None
    try:
        record = tc.render_certified(family=family, header_ids_by_arm=header_ids_by_arm, measured_live=measured_live,
                                     probe_regimes=probe["regimes"], on_completion=probe["responses"][tc.THINKING_ON],
                                     header_source_by_arm=header_source_by_arm,
                                     standup_thinking_mode=standup_thinking_mode,
                                     off_arm_refusal=probe.get("off_arm_refusal"))
    except tc.RenderCertificationError as e:
        return {**dict(e.evidence), CERTIFYING_DRAW_CLASS_FIELD: final_cls}, str(e) + _render_probe_declined_note(probe)
    record = {**record, CERTIFYING_DRAW_CLASS_FIELD: final_cls}
    if final_cls != PROBE_CLASS_THINK_SHAPE:
        problem = (f"the certifying ON draw's class is {final_cls!r}, not {PROBE_CLASS_THINK_SHAPE!r}"
                   + (" — no ON draw was recorded" if final_cls is None else
                      " — a channel-less cap-cut (nothing emitted in either channel) is not the family's think "
                      "shape however the finish reason reads" if final_cls == PROBE_CLASS_AT_CAP_CHANNEL_LESS else
                      " — only a draw carrying the family's think shape certifies the render"))
        for flag in (tc.OOS_EVIDENCE_HEADER_CERTIFIED, "render_regime_certified", "think_shape_certified",
                     "render_certified"):
            if record.get(flag) is not None:
                record[flag] = False
        record["certification_problems"] = [*list(record.get("certification_problems") or []), problem]
        return record, (f"{tc.REC_THINKING_UNSERVABLE}: predicate (A) — the live base-route render of family "
                        f"{family!r} could not be certified against its render fact: {problem}"
                        + _render_probe_declined_note(probe))
    return record, None


#: predicate (C)'s completion cap — greedy, seed 0, no other sampler key. Named so the resolved-sampler
#: recorder's probe-kind table (scripts/sdfbase/record_resolved_sampler.py PROBE_KIND_TABLE["lens_on_bite"])
#: can be pinned equal to the code that fires the probe, like RENDER_PROBE_MAX_TOKENS above.
LENS_ON_BITE_MAX_TOKENS = 256


def _no_switch_render_probe(chat, vmodel: str, *, system=None, sampling_row=None) -> dict:
    """PREDICATE (A″)'s live probe (llama-paper-rows P18): ONE 1-item completion on
    the cell's route under the family's ONE render — NO ``chat_template_kwargs``,
    no declaration (``contracts.render_fragment`` under the no-switch arm is
    empty) — returning the reply record (:func:`_probe_reply`) and the stamp's
    ``render_regime`` (expected ``unspecified``: the proxy's literal for an absent
    switch). ``system`` / ``sampling_row`` as for :func:`_render_probe` (a
    prompted cell's SI, the model's card row). A proxy refusal is a NAMED
    refusal, surfaced as the driver's."""
    from sisteer.capeval.serving import proxy as _px

    decode = _probe_decode(sampling_row)
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": RENDER_PROBE_PROMPT}]
    body = {"model": vmodel, "messages": messages, **decode, "max_tokens": RENDER_PROBE_MAX_TOKENS, "seed": 0,
            **tc.render_fragment(lens=True, arm=tc.THINKING_NOT_AVAILABLE, header_located=False)}
    assert tc.CHAT_TEMPLATE_KWARGS_KEY not in body and tc.OOS_THINKING_DECLARATION_KEY not in body
    try:
        resp = chat(body)
    except _px.ProxyError as e:
        raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: predicate (A″)'s render probe on {vmodel!r} was "
                         f"REFUSED by the proxy ({e}) — a named refusal, never worked around") from None
    rec = _probe_reply(resp)
    return {"regimes": {tc.THINKING_NOT_AVAILABLE: rec["render_regime"]}, "responses": {tc.THINKING_NOT_AVAILABLE: rec},
            "on_attempts": []}


def _no_switch_render_certification(probe: Mapping, *, family, header_ids: "Sequence[int]", measured_live: bool
                                    ) -> "tuple[dict, str | None]":
    """PREDICATE (A″)'s verdict on a :func:`_no_switch_render_probe` result:
    ``contracts.no_switch_render_certified`` over the once-measured header, the
    probe's ``unspecified`` stamp and its tag-free, channel-free completion.
    Returns ``(record, error)`` — the record is the certification's own (every
    flag false and every problem named on a refusal), written before the refusal."""
    rec = probe["responses"][tc.THINKING_NOT_AVAILABLE]
    try:
        record = tc.no_switch_render_certified(family=family, header_ids=header_ids, measured_live=measured_live,
                                               probe_regime=probe["regimes"][tc.THINKING_NOT_AVAILABLE],
                                               probe_completion=rec)
    except tc.RenderCertificationError as e:
        return {**dict(e.evidence), CERTIFYING_DRAW_CLASS_FIELD: PROBE_CLASS_NO_THINKING_EMITTED}, str(e)
    return {**record, CERTIFYING_DRAW_CLASS_FIELD: PROBE_CLASS_NO_THINKING_EMITTED}, None


def _lens_on_bite(chat, *, lens_vmodel: str, base_vmodel: str, family: str,
                  logprob_liveness: "Mapping | None" = None) -> "tuple[dict, str | None]":
    """PREDICATE (C) — the lens-ON bite: ONE lens-routed request to the
    object cell with ``enable_thinking: true`` and NO declaration, and the SAME
    prompt on the base route under the same render; the lens reply must carry a
    reasoning block (the ON render reached the lens route) and differ from the
    base reply (the object bit under ON) — ``contracts.lens_on_bite_verdict``.
    Returns ``(record, error)``; the record is written before any refusal.

    ``logprob_liveness``: the same run's
    ``steering_bites.logprob_liveness`` block — offered by the caller ONLY when
    that gate ruled the cell — which the verdict may take as its liveness
    evidence in place of the token bite when it is CONCLUSIVE
    (``contracts.lens_on_bite_verdict``). An admitted record carries the fact
    on the row and is logged; a refusal with an offered-but-inconclusive block
    records WHY the substitution did not apply beside the unchanged error."""
    from sisteer.capeval.serving import proxy as _px

    def _one(vmodel: str) -> dict:
        return _probe_reply(chat({
            "model": vmodel,
            "messages": [{"role": "user", "content": "In one short paragraph, why is the sky blue?"}],
            "temperature": 0.0, "max_tokens": LENS_ON_BITE_MAX_TOKENS, "seed": 0,
            **tc.render_fragment(lens=True, arm=tc.THINKING_ON, header_located=False)}))

    try:
        lens_reply = _one(lens_vmodel)
        base_reply = _one(base_vmodel)
    except _px.ProxyError as e:
        rec = {"applicable": True, "verified": False, "error": f"proxy refusal: {e}"}
        return rec, f"{tc.REC_THINKING_UNSERVABLE}: predicate (C) lens-ON bite — the proxy refused ({e})"
    try:
        rec = tc.lens_on_bite_verdict(lens_reply=lens_reply, base_reply=base_reply, family=family,
                                      logprob_liveness=logprob_liveness)
    except tc.LensOnBiteError as e:
        rec = {"applicable": True, "verified": False, "error": str(e),
               "lens_finish_reason": lens_reply.get("finish_reason"), "base_finish_reason": base_reply.get("finish_reason"),
               "lens_render_regime": lens_reply.get("render_regime")}
        if logprob_liveness is not None:
            # the refusal is the result; say why the offered verdict could not stand in — diagnosable after
            # the container is gone, the error text itself unchanged
            rec["logprob_liveness_substitution"] = {
                "offered": True, "admitted": False,
                "problems": tc.logprob_liveness_conclusive_problems(logprob_liveness) or
                ["the token bite was not the only failing rule — the logprob verdict stands in for the token bite alone"]}
        return rec, str(e)
    rec["lens_vmodel"], rec["base_vmodel"] = lens_vmodel, base_vmodel
    if rec.get(tc.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY) == tc.LENS_ON_BITE_EVIDENCE_LOGPROB:
        ref = rec.get(tc.LENS_ON_BITE_LOGPROB_REF_KEY) or {}
        print(f"[served-cap] predicate (C) admitted by LOGPROB liveness (class E, D95): {tc.LENS_ON_BITE_LOGPROB_SENTENCE} "
              f"— median divergence {ref.get('median_divergence')} vs measured floor {ref.get('measured_noise_floor')}, "
              f"bar {ref.get('bar')} ({ref.get('binding')}), conclusive {ref.get('n_conclusive')}/{ref.get('n_probes')}; "
              f"identity (B′) checked as before", flush=True)
    return rec, None


def _probe_stamp_problem(stamps_by_leg: Mapping, render_arm: str, *,
                         out_of_support: bool, policies: Mapping,
                         errors_by_leg: "Mapping | None" = None,
                         lens_by_leg: "Mapping | None" = None,
                         mode_by_leg: "Mapping | None" = None) -> "str | None":
    """Every leg of the probe carried EXACTLY ONE stamp triple and it is
    the leg's expected row —
    ``contracts.expected_stamp_regimes(lens=lens_by_leg[leg], arm, policy=policies[leg])``
    PER LEG: a LENS leg's row is
    the §2.5.11 one (request-unique / its attested policy); a NATIVE leg — a
    LoRA adapter, or the zero adapter it is compared against — is the row the
    proxy stamps on every non-lens leg (``cell-stable``, ``not-consulted``,
    the render), so a healthy native bite verifies instead of reading as a
    lens leg with the wrong salt regime. ``lens_by_leg`` None (every caller
    before the native engine) reads every leg as lens — byte-identical — with
    each leg's ``resumed_prefill_policy`` label from
    ``contracts.expected_stamp_policy`` — the STRICT §2.5.11 rows (1–2 under
    ON, 3–4 under OFF) on the authorised experiment's legs, the leg's ATTESTED
    label (the steered cell's under its own key, the twin's under
    ``none__lens0``) on every other run (review round B, P1: an all-position
    cell attested ``always`` probes cell-stable, and that is its row). ``None``
    when every leg holds, else the refusal text naming the leg and both triples.

    The additive-key rule, per field of the one triple: a
    PRESENT value that is not the row's refuses on every run; a field the
    proxy did not stamp at all (today's pre-landing proxy: every response reads
    ``None`` there) refuses only under ``out_of_support`` — fail closed against
    an uncertified render — and is tolerated in support, where the recorded
    ``stamps_by_leg`` says so. A field stamped on some prompts and not others
    is a disagreement and refuses everywhere. ``errors_by_leg`` — the
    extraction refusals the probe RECORDED (a PRESENT malformed stamp:
    ``contracts.stamp_regimes``) — are reported FIRST, on every run whatever
    ``out_of_support``: an incompatible proxy is never read as unstamped."""
    for leg, errors in (errors_by_leg or {}).items():
        if errors:
            return (f"probe leg {leg!r}: {len(errors)} response(s) carried a PRESENT malformed regime "
                    f"stamp — {errors[0]}")
    for leg, stamps in stamps_by_leg.items():
        if leg not in policies:
            return (f"probe leg {leg!r}: no expected resumed_prefill_policy label was derived for it "
                    f"(labels for {sorted(policies)}) — the leg cannot be held to a row")
        if lens_by_leg is not None and leg not in lens_by_leg:
            return (f"probe leg {leg!r}: no engine reading (lens / native) was derived for it "
                    f"(readings for {sorted(lens_by_leg)}) — the leg cannot be held to a row")
        lens = True if lens_by_leg is None else bool(lens_by_leg[leg])
        try:
            # a NATIVE leg at a restricted position mode (`response` /
            # `prompt_last` / `prefill_all`) is minted request-unique — its expected row follows
            # the route's own mode (`mode_by_leg`; None = an all-position / LoRA / lens leg)
            expected = tc.expected_stamp_regimes(lens=lens, arm=render_arm,
                                                 policy=policies[leg],
                                                 position_mode=(mode_by_leg or {}).get(leg))._asdict()
        except ValueError as e:
            return f"probe leg {leg!r}: no expected row can be derived — {e}"
        distinct = {json.dumps(s, sort_keys=True) for s in stamps}
        if not stamps:
            return f"probe leg {leg!r}: no response carried a stamp"
        if len(distinct) != 1:
            return (f"probe leg {leg!r}: {len(distinct)} distinct regime stamps across its "
                    f"{len(stamps)} prompts ({sorted(distinct)}) — a leg wears ONE triple")
        [only] = [json.loads(d) for d in distinct]
        if set(only) != set(expected):
            return (f"probe leg {leg!r}: regime stamp {only} does not name the fields of the "
                    f"expected row {expected} (contract §2.5.11)")
        absent = sorted(f for f, v in only.items() if v is None)
        differing = sorted(f for f, v in only.items() if v is not None and v != expected[f])
        if differing:
            return (f"probe leg {leg!r}: regime stamp {only} != the expected row for a "
                    f"{'lens' if lens else 'non-lens (native adapter)'} leg "
                    f"under the {render_arm!r} render {expected} (fields {differing}; contract "
                    "§2.5.11)")
        if absent and out_of_support:
            return (f"probe leg {leg!r}: regime stamp {only} carries no value for {absent} (a "
                    f"pre-landing proxy?) — an out-of-support run requires the landed proxy's "
                    f"row {expected} ({tc.FLAG_ALLOW_OOS_THINKING}; contract §2.5.11)")
    return None


def _served_prompt_budget(tokenize, vmodel: str, messages_list, *, arm: str,
                          artifact: Mapping, max_model_len: int, cap: int,
                          concurrency: int = 1,
                          expected_n: int = tc.GSM8K_PAIRED_N) -> "tuple[dict, str | None]":
    """The LIVE prompt-budget pass: tokenize the 200 EXACT rendered
    GSM8K message lists through the proxy's read-only ``/v1/tokenize`` under
    the run's render (the same body shape as the header measurement —
    ``add_generation_prompt: true`` + the arm's ``chat_template_kwargs``; NO
    declaration on tokenize, Q12) and assert (i) ``max + cap <=`` the served
    ``max_model_len``, (ii) the live lengths equal the CPU artifact's item by
    item (same ``chat_template_hash`` => same render; a difference is a
    tokenizer divergence that is not the template), (iii) ``n == 200 ==`` the
    artifact's. Returns ``(record, error)``: the record is the evidence block
    ``prompt_budget_live`` (``max`` / ``p99`` under
    :data:`contracts.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX` / ``_P99``; ``n``,
    ``p50``, ``window``, ``cap``, ``margin``, ``source``, the artifact's max /
    p99 / digest, ``n_items_differing``), written FIRST; the caller exits on the
    error before the first generation request. ``concurrency`` bounds the
    tokenize posts in flight (1..32; each is its own read-only request); the
    answers are validated in item order, so the record does not depend on it."""
    kwargs = tc.chat_template_kwargs(arm)
    bodies = [{"model": vmodel, "messages": [dict(m) for m in msgs],
               "add_generation_prompt": True, **kwargs}
              for msgs in messages_list]
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(concurrency), 32))) as pool:
        responses = list(pool.map(tokenize, bodies))
    lengths: list = []
    problems: list = []
    for i, resp in enumerate(responses):
        toks = resp.get("tokens") if isinstance(resp, Mapping) else None
        if not isinstance(toks, (list, tuple)) or not toks \
                or not all(isinstance(t, int) and not isinstance(t, bool) for t in toks):
            problems.append(f"item {i}: /v1/tokenize answered with no usable 'tokens' list ({resp!r})")
            continue
        lengths.append(len(toks))
    art_lengths = [int(x) for x in artifact.get("lengths") or []]
    rec: dict = {
        "n": len(lengths),
        tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX: (max(lengths) if lengths else None),
        tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99: (tc.nearest_rank_percentile(lengths, 0.99)
                                                 if lengths else None),
        "p50": (tc.nearest_rank_percentile(lengths, 0.50) if lengths else None),
        "window": int(max_model_len), "cap": int(cap),
        "margin": (int(max_model_len) - (max(lengths) + int(cap)) if lengths else None),
        "arm": arm, "source": HEADER_SOURCE_SERVED,
        "artifact_max": artifact.get("max"), "artifact_p99": artifact.get("p99"),
        "artifact_n": artifact.get("n"),
        "artifact_chat_template_hash": artifact.get("chat_template_hash"),
        "n_items_differing": sum(1 for a, b in zip(lengths, art_lengths) if a != b)
                             + abs(len(lengths) - len(art_lengths)),
    }
    if not problems:
        if len(lengths) != int(expected_n) or len(lengths) != int(artifact.get("n") or -1):
            problems.append(f"n = {len(lengths)} live message lists, expected the frozen "
                            f"{int(expected_n)} (artifact n = {artifact.get('n')!r})")
        if rec["margin"] is not None and rec["margin"] < 0:
            problems.append(f"prompt max {rec[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX]} + cap {cap} = "
                            f"{rec[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX] + cap} > the served "
                            f"max_model_len {max_model_len} (over by {-rec['margin']}) — an "
                            "over-window request is a vLLM 400 and Inspect refuses the task")
        if lengths != art_lengths:
            problems.append(f"the live lengths differ from the CPU artifact's on "
                            f"{rec['n_items_differing']} item(s) (live max {rec[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX]} / "
                            f"p99 {rec[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99]}, artifact max "
                            f"{artifact.get('max')} / p99 {artifact.get('p99')}) under one "
                            f"chat_template_hash {artifact.get('chat_template_hash')} — a tokenizer "
                            "divergence that is not the template (pinned-revision drift)")
    error = None
    if problems:
        error = (f"{tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: the live prompt-budget pass over the "
                 f"served render refused: " + "; ".join(problems))
    rec["ok"] = error is None
    return rec, error


# --------------------------------------------------------------------------- #
# per-task thinking under ONE standup (U5-4; PW20b-D)                          #
# --------------------------------------------------------------------------- #


def _override_max_tokens_by_task(inspect_atoms, thinking_by_task: Mapping, *,
                                 out_of_support: bool, off_cap_override: bool,
                                 oos_experiment: "str | None" = None,
                                 on_cap_override: "Mapping | None" = None,
                                 cap_overrides_by_task: "Mapping[str, int] | None" = None,
                                 off_cap_setter: "Mapping | None" = None) -> dict:
    """The hashed ``max_tokens_by_task`` of a run under EITHER named override
    (ONE derivation for both): ``{row task: max_tokens}`` over
    EVERY Inspect task the run files, each cap from the plan builder's OWN rule
    (``inspect_served.task_max_tokens`` at that task's arm, under
    ``out_of_support`` — the OOS experiment's raised ON cap — or
    ``off_cap_override`` — the comparator's raised OFF cap) — the same
    derivation ``plans_for_atoms`` applies to the plans the run executes, never
    a copy of the table, so an 8192 store can never share a key with, or resume
    onto, a 2048 (or a 512) one.

    COUPLING (review round B): the launcher derives its PURE expectation of the
    same map from the tuple alone (``tinycap_served.oos_caps_by_task`` ->
    ``contracts.oos_max_tokens_on``), omitting any ON task the tuple names no
    cap for, where this derivation would fill the table's cap. They agree
    today ONLY because the authorised controls are GSM8K-only; the two-doors
    test holds them equal over the AUTHORISED controls by execution — and the
    comparator's map (``offcap_caps_by_task``) the same way.

    ``on_cap_override``: the bounded
    ON-cap SETTER, handed through to the same rule so the hashed map carries
    the executed 8,192 and the statement check holds as written.

    ``cap_overrides_by_task`` (bcb-bfcl-200 §S2b): on a truncation-ladder
    rung-2 run the VALIDATED rung (``_ladder_args``) REPLACES the table's cap
    for each fired leg — exactly what ``plan_task`` does with the same map
    (``cap_override`` -> ``CAP_SOURCE_OVERRIDE``) — so the ONE map this returns
    is the cap the run EXECUTES: version 2's ``max_tokens_by_task``, the
    ``--max-tokens-by-task`` statement check and the hashed
    ``cap_overrides_by_task`` can never disagree (before this the map carried
    the TABLE's cap beside a rung that planned another, and the correct
    statement was refused). A rung not above the table's cap refuses HERE,
    before any store is touched — the planner's own rule, stated once more at
    the driver's door. ``None`` (no ladder) leaves the derivation
    byte-identical to what it was."""
    from sisteer.capeval.tiers import inspect_served as _isv

    out: dict = {}
    for atom in inspect_atoms:
        for row_task in tc.TASKS_OF_CONTROL[atom]:
            cap, _source = _isv.task_max_tokens(
                _isv._task_name(row_task), thinking=thinking_by_task[row_task],
                out_of_support=bool(out_of_support), off_cap_override=bool(off_cap_override),
                oos_experiment=oos_experiment, on_cap_override=on_cap_override,
                off_cap_setter=off_cap_setter)
            out[row_task] = cap
    for row_task, rung in dict(cap_overrides_by_task or {}).items():
        if row_task not in out:
            raise SystemExit(f"[served-cap] {tc.FLAG_CAP_OVERRIDE} names {row_task!r}, which this run does not "
                             f"plan (tasks {sorted(out)}) — a rung for a leg the run never fires")
        table = out[row_task]
        if table is not None and int(rung) <= int(table):
            raise SystemExit(f"[served-cap] {tc.FLAG_CAP_OVERRIDE}: the ladder cap {int(rung)} for {row_task} is "
                             f"not above the table's cap {int(table)} — a rung that does not raise the cap "
                             "re-measures the same truncation (the task planner refuses it too; plan §S2b)")
        out[row_task] = int(rung)
    return {k: out[k] for k in sorted(out)}


def _item_subset_problem(doc: Mapping, *, executed_caps: Mapping, out_root: "str | Path | None" = None
                         ) -> "str | None":
    """Why a validated ``--item-subset`` file is NOT this run's, or
    ``None``: its task must be one the run plans at EXACTLY the file's ``cap``
    (the setter's rung cap); this run's ``out_root`` must NOT be the source
    root the file names (both resolved, symlink-aware — a rung generates into a
    NEW root beside its source, never into it: the two would share one usage
    ledger and request-evidence file, the rung's ``evidence.json`` would
    overwrite the source's, and two cell dirs under one ``cells/`` make
    ``inspect_rung.cell_dir_of`` refuse both; integration review H-P2); and
    the SOURCE it names must be bound — the source root holds ONE cell whose
    manifest carries ``harness_config_hash == source_identity_hash`` and
    ``max_tokens_by_task[task] == source_cap``. A source that cannot be read,
    or that disagrees, refuses: a rung re-draws for exactly one store. A
    SEPARATE rung root that already exists (the planner's ``ladder_state.json``,
    an interrupted attempt's files) is a legitimate resume and passes."""
    task = str(doc["task"])
    if task not in executed_caps:
        return f"names task {task!r}, which this run does not plan (tasks {sorted(executed_caps)})"
    if executed_caps[task] != int(doc["cap"]):
        # the rung cap's setter is the task's arm's: the ON setter for the paired
        # GSM8K, the OFF setter for IFEval (ifeval-canonical-battery §5.2)
        name = task[len(tc.INSPECT_TASK_PREFIX):] if task.startswith(tc.INSPECT_TASK_PREFIX) else task
        flag = tc.FLAG_OFF_CAP_BY_TASK if name in tc.OFF_CAP_BY_TASK_ALLOWED else tc.FLAG_ON_CAP_BY_TASK
        return (f"was written for cap {doc['cap']} but this run resolves {executed_caps[task]} for {task} — "
                f"state the rung cap with {flag} {name}={doc['cap']}")
    root = Path(str(doc["source_out_root"]))
    if out_root is not None and Path(out_root).resolve() == root.resolve():
        return (f"names source_out_root {doc['source_out_root']!r}, which resolves to THIS run's --out-root "
                f"({Path(out_root).resolve()}) — a rung generates into a NEW root beside its source, never into "
                f"it: one root would share the source's {tc.LEDGER_FILENAME} and {tc.REQUEST_EVIDENCE_FILENAME}, "
                "overwrite its evidence.json, and hold two cell dirs under cells/ (inspect_rung.cell_dir_of "
                "refuses both). Point --out-root at a separate rung root (plan §4.4: <source>__rung<cap>); "
                "nothing was written")
    cells_dir = root / "cells"
    cells = sorted(p for p in cells_dir.iterdir() if p.is_dir()) if cells_dir.is_dir() else []
    if len(cells) != 1:
        return f"source root {root} holds {len(cells)} cell dir(s) under cells/, not exactly one"
    try:
        manifest = json.loads((cells[0] / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        return f"the source store's manifest {cells[0] / 'manifest.json'} is unreadable ({type(e).__name__}: {e})"
    have_hash = manifest.get("harness_config_hash")
    if have_hash != doc["source_identity_hash"]:
        return (f"source_identity_hash {doc['source_identity_hash']!r} is not the source store's harness_config_hash "
                f"{have_hash!r} ({cells[0]}) — the subset was planned against another store")
    caps = ((manifest.get("resolved_config") or {}).get(tc.MANIFEST_KEY_TINYCAP) or {}).get(
        tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK) or {}
    if caps.get(task) != int(doc["source_cap"]):
        return (f"source_cap {doc['source_cap']} is not the source store's executed cap {caps.get(task)!r} for "
                f"{task} — the subset's at-cap facts were read at another cap")
    return None


def _prior_generation_attempt(prior_attempts, records) -> "str | None":
    """The LATEST recorded prior attempt that GENERATED (sent a task call —
    :func:`contracts.attempt_generation_calls` over the request-evidence file),
    or ``None``: the attempt whose complete retained ``.eval`` this
    run would FINALISE if it reuses one. Every prior entry must carry its id
    (a pre-contract entry refuses, ``AttemptScopeError``)."""
    gens = []
    for i, a in enumerate(prior_attempts):
        if not isinstance(a, Mapping) or a.get(tc.ATTEMPT_ID_FIELD) is None:
            raise tc.AttemptScopeError(
                f"resume_attempts[{i}] carries no {tc.ATTEMPT_ID_FIELD} — a store whose attempts predate the "
                f"attempt contract cannot be attributed under {tc.FLAG_ATTEMPT_ID}")
        aid = str(a[tc.ATTEMPT_ID_FIELD])
        if tc.attempt_generation_calls(records, aid) > 0:
            gens.append(aid)
    return gens[-1] if gens else None


def _math500_max_tokens(executed_caps: Mapping) -> int:
    """The hashed ``math500_max_tokens`` of a run that files MATH-500: the
    FIRST-PASS cap ``inspect/math`` is generated at, read off
    the run's ONE executed-cap map (:func:`_override_max_tokens_by_task` — the
    plan builder's OWN rule, ``inspect_served.task_max_tokens``, at the task's
    arm under this run's support state, override and matched tuple; the same
    derivation ``plans_for_atoms`` applies to the plan the run executes, never
    a copy of the table) — so a 4,096-cap store and an 8,192-cap store can
    never share a key or resume onto each other, whatever the retry field
    says, and the scalar can never disagree with version 2's
    ``max_tokens_by_task["inspect/math"]``, which is the same map's entry
    (m4 merge caveat). Called only where ``contracts.files_math500`` holds; a
    map without the MATH row is a caller bug and refuses.

    COUPLING: the launcher derives the same number purely
    (``contracts.math500_max_tokens_for``); ``tests/test_inspect_served.py``
    holds the two equal over every arm, both support states and every
    registry tuple."""
    math = tc.inspect_task("math")
    if math not in executed_caps or executed_caps[math] is None:
        raise SystemExit(
            f"[served-cap] {tc.IDENTITY_KEY_MATH500_MAX_TOKENS}: the executed-cap map "
            f"{dict(executed_caps)} carries no cap for {math} — the run files MATH-500 "
            "but the plan builder resolved no first-pass cap for it")
    return int(executed_caps[math])


def _resolve_thinking_by_task(atoms, thinking: str) -> dict:
    """The run's ``{row task: arm}`` map (``contracts.run_thinking_by_task``),
    recorded as identity in EVERY mode.

    Under ``per-task`` each task takes ``contracts.THINKING_BY_TASK``'s arm and
    a task the policy does not map is REFUSED here, at plan time, naming it —
    before a standup is read, a store is touched or a GPU minute is spent.
    Under ``on`` / ``off`` every task takes that one arm, so the map is still
    the single place a reader asks "what arm was this task run under"."""
    try:
        return tc.run_thinking_by_task(atoms, thinking)
    except KeyError as e:
        raise SystemExit(
            f"[served-cap] {tc.FLAG_THINKING} {thinking}: {e.args[0]} — the per-task "
            "policy is total over contracts.TINYCAP_TASKS, so an unmapped task means "
            "the policy and the task list have drifted apart; add the task's arm to "
            f"contracts.THINKING_BY_TASK, or run it under an explicit "
            f"{tc.FLAG_THINKING} {'|'.join(tc.THINKING_MODES)}") from None


def _sampling_model_key(model_id) -> "str | None":
    """The canonical model key (``geometry.KNOWN_MODEL_KEYS``: the sweep's
    eight AND the onboarded models) whose pinned HF id IS ``model_id`` —
    exact, case-insensitive, no mirror or family guessing: the sampling row is
    frozen per model and a near-match would hash another model's card values
    into this store. None when no key pins that id (``geometry.model_key_for``
    does the same lookup but WARNS about budget sizing, which is not the
    question here).

    Both geometry maps are read: the front door's onboarded models live in
    ``geometry.ONBOARDED_MODEL_HF_IDS``, not ``MODEL_HF_IDS``, and a lookup
    over the sweep's map alone handed ``contracts.sampling_for`` the key
    ``'<unpinned>'`` for a served onboarded model — every gsm8k unit of its
    first wave died on that refusal after a clean standup."""
    if not model_id:
        return None
    wanted = str(model_id).strip().lower()
    for table in (sg.MODEL_HF_IDS, sg.ONBOARDED_MODEL_HF_IDS):
        for key, hf in table.items():
            if hf.lower() == wanted:
                return key
    return None


def _resolve_sampling(args, cfg: dict, atoms, thinking_by_task: dict) -> SimpleNamespace:
    """The run's sampling contract (U8 item 1, PW-sampling), resolved BEFORE
    anything is read or touched: the ``--sampling`` mode, the model key the
    row is looked up under (the config's ``model`` against
    ``geometry.MODEL_HF_IDS`` — exact), the row itself
    (``contracts.sampling_for``: the model's frozen card row under ``card``,
    ``GREEDY_SAMPLING`` under ``greedy``; a rowless model REFUSES under
    ``card``, never a default temperature), the ``{row task: k}`` map over the
    tasks this run files (``contracts.run_samples_by_task`` over the thinking
    map's keys, so the two per-task maps range over the same tasks) and the
    MATH-500 k recorded as the identity's ``samples`` (``--samples`` explicit
    wins in either mode; else 4 under ``card``, 1 under ``greedy``)."""
    mode = tc.check_sampling_arg(args.sampling)
    model_key = _sampling_model_key(cfg.get("model"))
    # The Qwen base seed-replicate battery: the OPT-IN Inspect seed policy — validated HERE, before anything is read; a
    # replicate policy with no Inspect task to seed (or under `protocol`, whose identity hashes the frozen rules) refuses
    try:
        seed_policy, seed_base = tc.check_inspect_seed_policy(getattr(args, "inspect_seed_policy", None),
                                                              getattr(args, "inspect_seed_base", None))
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    seeded_tasks = [t for t in thinking_by_task if t in tc.INSPECT_TASKS and t != tc.inspect_task(tc.IFEVAL_TASK_NAME)]
    if seed_policy == tc.INSPECT_SEED_POLICY_REPLICATE and not seeded_tasks:
        raise SystemExit(f"[served-cap] {tc.FLAG_INSPECT_SEED_POLICY} {seed_policy}: this run files no Inspect task the rule "
                         f"applies to (tasks {sorted(thinking_by_task)}) — nothing to seed; drop the flag")
    if seed_policy == tc.INSPECT_SEED_POLICY_REPLICATE and mode == tc.SAMPLING_PROTOCOL:
        raise SystemExit(f"[served-cap] {tc.FLAG_INSPECT_SEED_POLICY} {seed_policy} is not defined under {tc.FLAG_SAMPLING} "
                         f"{tc.SAMPLING_PROTOCOL} (its identity hashes the frozen rules)")
    seed_kw = dict(inspect_seed_policy=seed_policy, inspect_seed_base=seed_base)
    if not atoms:
        return SimpleNamespace(mode=mode, model_key=model_key, row=None,
                               samples_by_task={}, samples=None,
                               sampling_by_task=None, protocol_by_task=None, effective_by_task=None, **seed_kw)
    if mode == tc.SAMPLING_PROTOCOL:
        # PER TASK and MODEL-INDEPENDENT —
        # no single row (`row` None); every filed task must have a protocol row
        # (a prefill / EvalPlus / any other task refuses HERE, by name); the
        # EXECUTED rows are what the provider sends and the identity hashes, the
        # declarations ride the manifest's by_task; the in-image check's record
        # (--protocol-check) supplies the RESOLVED `effective` values
        try:
            protocol_by_task = tc.protocol_rows_by_task(thinking_by_task)
        except tc.ProtocolSamplingUnavailable as e:
            raise SystemExit(f"[served-cap] {e}") from None
        if not getattr(args, "protocol_check", None):
            raise SystemExit(
                f"[served-cap] {tc.FLAG_SAMPLING} {tc.SAMPLING_PROTOCOL} needs {tc.FLAG_PROTOCOL_CHECK} <path>: "
                "the preflight's check_protocol_sampling record (the RESOLVED SamplingParams per task, WITH the "
                "clamp override) is what the manifest's by_task.effective records — without it the executed "
                "values would be assumed, never resolved (P1-2); the launcher writes it into the out root")
        try:
            record = json.loads(Path(args.protocol_check).read_text())
            effective_by_task = tc.check_protocol_effective(record)
        except (OSError, ValueError) as e:
            raise SystemExit(f"[served-cap] {tc.FLAG_PROTOCOL_CHECK} {args.protocol_check}: {e}") from None
        missing = sorted(t for t in protocol_by_task if t not in effective_by_task)
        if missing:
            raise SystemExit(
                f"[served-cap] {tc.FLAG_PROTOCOL_CHECK} {args.protocol_check}: the record resolves "
                f"{sorted(effective_by_task)} but this run files {missing} under {tc.SAMPLING_PROTOCOL} — "
                "the check must cover every filed leg")
        try:
            by_task = tc.run_samples_by_task(thinking_by_task, mode, math_samples=args.samples)
            samples = tc.samples_of_task(mode, tc.inspect_task("math"), math_samples=args.samples)
        except ValueError as e:
            raise SystemExit(f"[served-cap] {tc.FLAG_SAMPLES}: {e}") from None
        return SimpleNamespace(mode=mode, model_key=model_key, row=None,
                               samples_by_task=by_task, samples=int(samples),
                               sampling_by_task={t: r.executed for t, r in protocol_by_task.items()},
                               protocol_by_task=protocol_by_task,
                               effective_by_task={t: effective_by_task[t] for t in protocol_by_task}, **seed_kw)
    if model_key is None and mode == tc.SAMPLING_CARD:
        raise SystemExit(
            f"[served-cap] {tc.FLAG_SAMPLING} {mode}: the config's model {cfg.get('model')!r} is "
            f"not the pinned HF id of any canonical model key ({', '.join(sg.KNOWN_MODEL_KEYS)}; "
            "geometry.MODEL_HF_IDS / ONBOARDED_MODEL_HF_IDS), so no frozen sampling row can be looked up for it — "
            f"refusing to invent a temperature. Pin the model under a key, or run "
            f"{tc.FLAG_SAMPLING} {tc.SAMPLING_GREEDY}")
    try:
        row = tc.sampling_for(model_key if model_key is not None else "<unpinned>", mode)
    except tc.SamplingUnverified as e:
        raise SystemExit(f"[served-cap] {e}") from None
    math_task = tc.inspect_task("math")
    try:
        by_task = tc.run_samples_by_task(thinking_by_task, mode, math_samples=args.samples)
        samples = tc.samples_of_task(mode, math_task, math_samples=args.samples)
    except ValueError as e:
        raise SystemExit(f"[served-cap] {tc.FLAG_SAMPLES}: {e}") from None
    return SimpleNamespace(mode=mode, model_key=model_key, row=row,
                           samples_by_task=by_task, samples=int(samples),
                           sampling_by_task=None, protocol_by_task=None, effective_by_task=None, **seed_kw)


def _sampling_identity(sampling: SimpleNamespace) -> dict:
    """The HASHED sampling block of this run: the row's
    ``contracts.sampling_identity`` under every pre-existing mode (byte for
    byte what it was); under ``protocol``,
    ``contracts.protocol_sampling_identity`` — the three run-level scalars NULL
    plus the per-task EXECUTED map ``sampling_by_task``."""
    if sampling.mode == tc.SAMPLING_PROTOCOL:
        return tc.protocol_sampling_identity(sampling.sampling_by_task, sampling.samples_by_task)
    return tc.sampling_identity(sampling.row, sampling.samples_by_task, **_seed_policy_kw(sampling))


def _seed_policy_kw(sampling: SimpleNamespace) -> dict:
    """qwen-base-seed-replicates: the run's Inspect seed policy as `contracts.sampling_identity` / `seed_rules_for`
    kwargs (default: both None — the pre-existing identity, byte for byte)."""
    return dict(inspect_seed_policy=getattr(sampling, "inspect_seed_policy", None),
                inspect_seed_base=getattr(sampling, "inspect_seed_base", None))


def _sampling_block(sampling: SimpleNamespace) -> dict:
    """``manifest["sampling"]`` (``contracts.SAMPLING_BLOCK_FIELDS``): the
    hashed sampling identity's mirror plus what it does not hash — the mode,
    the model key the row was looked up under, and the row's source prose.

    Under ``protocol`` the block carries the
    five run-level fields (scalars null), the joined per-task provenance as
    ``source``, and ``by_task`` — per filed task ``{executed, protocol_temperature,
    executed_reason, effective}`` (``contracts.protocol_manifest_by_task``; the
    ``effective`` values are the preflight check's RESOLVED parameters). The
    hashed ``sampling_by_task`` map is NOT duplicated here: ``by_task[t].executed``
    mirrors it, and the launcher's store-trust check compares the two."""
    if sampling.mode == tc.SAMPLING_PROTOCOL:
        tasks = sorted(sampling.protocol_by_task)
        ident = tc.protocol_sampling_identity(sampling.sampling_by_task, sampling.samples_by_task)
        block = {"mode": sampling.mode, "model_key": sampling.model_key,
                 **{f: ident[f] for f in tc.SAMPLING_IDENTITY_FIELDS},
                 "source": tc.protocol_source(tasks),
                 tc.SAMPLING_BLOCK_KEY_BY_TASK: tc.protocol_manifest_by_task(tasks, sampling.effective_by_task)}
        assert set(block) == set(tc.sampling_block_fields(None, mode=tc.SAMPLING_PROTOCOL))
        return block
    block = {"mode": sampling.mode, "model_key": sampling.model_key,
             **tc.sampling_identity(sampling.row, sampling.samples_by_task, **_seed_policy_kw(sampling)),
             "source": sampling.row.source}
    # the optional terms join the block ONLY when the row sets them
    assert set(block) == set(tc.sampling_block_fields(sampling.row))
    return block


def _check_manifest_sampling(store, *, expected: dict) -> None:
    """Resume guard on the SAMPLING (U8 / PW-sampling), the sibling of
    :func:`_check_manifest_thinking_map`.

    The five sampling fields are hashed identity, so a different draw normally
    lands in a different cell dir; this is the check the 8-hex key prefix
    cannot make (a collision, a copied or hand-edited store). It refuses
    naming every field that moved — the failure it exists to stop is one
    store holding rows drawn at two temperatures, or greedy rows beside
    sampled ones, all called ``acc``. A manifest with no ``sampling`` block
    predates the contract and cannot vouch for how its rows were drawn, so it
    is refused like a manifest without the window."""
    if not store.manifest_path.exists():
        return
    rec = json.loads(store.manifest_path.read_text())
    block = rec.get(tc.MANIFEST_KEY_SAMPLING)
    # a legacy block (the base fields) and a block with the row's optional
    # terms are both the contract's shape; anything else is not.
    # Under `protocol` the block carries by_task too — accepted for a protocol
    # EXPECTATION only (a card store carrying it, or a protocol store lacking
    # it, is another instrument's manifest)
    base_fields = set(tc.SAMPLING_BLOCK_FIELDS)
    protocol = expected.get("mode") == tc.SAMPLING_PROTOCOL
    allowed_extra = set(tc.SAMPLING_OPTIONAL_IDENTITY_FIELDS) | ({tc.SAMPLING_BLOCK_KEY_BY_TASK} if protocol else set())
    shape_ok = (isinstance(block, Mapping) and base_fields <= set(block)
                and set(block) - base_fields <= allowed_extra
                and (not protocol or tc.SAMPLING_BLOCK_KEY_BY_TASK in block))
    if not shape_ok:
        raise ServingIdentityMismatch(
            f"[served-cap] {store.manifest_path.parent}: the cell manifest carries no "
            f"{tc.MANIFEST_KEY_SAMPLING!r} block of the contract's shape "
            f"({list(tc.SAMPLING_BLOCK_FIELDS)}) — a store written before the sampling "
            "contract (U8) cannot vouch for how its rows were drawn; refusing to resume onto "
            "it. Generate into a fresh out root")
    compared = ("mode", *tc.SAMPLING_IDENTITY_FIELDS,
                *(f for f in tc.SAMPLING_OPTIONAL_IDENTITY_FIELDS if f in expected or f in block))
    moved = [f"{f}: the manifest records {block.get(f)!r}, this run is {expected.get(f)!r}"
             for f in compared if block.get(f) != expected.get(f)]
    if protocol:
        # bcb-bfcl-200-widen: the per-task executed rows, declarations and reasons
        # must be this run's, task for task (`effective` — the check's resolved
        # record — is informational here: the served override is identity already)
        have_bt = block.get(tc.SAMPLING_BLOCK_KEY_BY_TASK) or {}
        want_bt = expected.get(tc.SAMPLING_BLOCK_KEY_BY_TASK) or {}
        strip = lambda m: {t: {k: v for k, v in dict(b).items() if k != "effective"} for t, b in dict(m).items()}
        if strip(have_bt) != strip(want_bt):
            moved.append(f"{tc.SAMPLING_BLOCK_KEY_BY_TASK}: the manifest records {strip(have_bt)!r}, "
                         f"this run is {strip(want_bt)!r}")
    if moved:
        raise ServingIdentityMismatch(
            f"[served-cap] {store.manifest_path.parent}: resume mismatch on the sampling "
            f"contract — {'; '.join(moved)}; a store drawn one way never resumes under another "
            "(U8: temperature, top_p, top_k, the per-task k and the seed rule are identity)")


def _leg_arm(by_task: dict, tasks, leg: str, thinking: str) -> str:
    """The ONE arm a non-Inspect leg runs at — the EvalPlus generator takes a
    single ``thinking`` argument, so its tasks' arms must agree. They do under
    both U5's policy (every EvalPlus task OFF) and any whole-configuration
    mode; a future map that split them would be refused HERE rather than
    silently serving one arm for both."""
    arms = {by_task[t] for t in tasks if t in by_task}
    if len(arms) > 1:
        raise SystemExit(
            f"[served-cap] the {leg} leg takes ONE thinking arm but this run's per-task "
            f"map gives its tasks {sorted(arms)} "
            f"({ {t: by_task[t] for t in tasks if t in by_task} }) — that leg cannot "
            "serve two arms in one pass; split the run or align the map")
    return arms.pop() if arms else (tc.THINKING_OFF if tc.is_per_task(thinking) else thinking)


class HeaderMeasurementError(RuntimeError):
    """The per-arm assistant header could not be MEASURED (E45 M5): neither
    the live server's ``/v1/tokenize`` nor a local tokenizer produced a pair
    of renders the header can be read off, or the local tokenizer that DID
    answer renders a different chat template than the server was launched
    with. Typed rather than a bare ``SystemExit`` so a caller can tell an
    unmeasurable precondition from a measured refusal — the CLI records the
    failed measurement in ``evidence.json`` and only then exits."""


class MixedArmHeaderMoved(HeaderMeasurementError):
    """MEASURED and refused: a mixed-arm run on a LENS route whose served
    assistant header MOVES between the arms. Carries both renders (ids and
    widths per arm) in its message, because the two renders ARE the
    finding."""


#: the probe conversation both header measurements render (one user turn —
#: the trailing assistant header does not depend on its content)
HEADER_PROBE_MESSAGES = ({"role": "user", "content": "x"},)
#: ``thinking_header_by_arm.source``: measured through the LIVE server's
#: ``/v1/tokenize`` (the render the model is actually served) …
HEADER_SOURCE_SERVED = tc.RENDER_SOURCE_SERVED_TOKENIZE
#: … or off a locally loaded tokenizer, the recorded FALLBACK. Never silent:
#: the block always states which one it is and why.
HEADER_SOURCE_LOCAL = tc.RENDER_SOURCE_LOCAL_TOKENIZER
#: … or the MIX on an ON-pinned standup: the ON arm live, the OFF arm a local REFERENCE (the proxy refuses the OFF
#: render by design). The block's ``source_by_arm`` names each arm's source; ``source`` is never wholly ``served_tokenize``
#: when an arm was local.
HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF = tc.RENDER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
#: the header block's per-arm provenance and OFF-reference fields (additive)
HEADER_KEY_SOURCE_BY_ARM = "source_by_arm"
HEADER_KEY_OFF_HEADER_SOURCE = "off_header_source"
HEADER_KEY_OFF_HEADER_ROLE = "off_header_role"
HEADER_KEY_OFF_HEADER_FALLBACK_REASON = "off_header_fallback_reason"
HEADER_KEY_OFF_ARM_REFUSAL = "off_arm_refusal"
HEADER_KEY_OFF_ARM_ACCEPTED_UNDER = "off_arm_accepted_under"

#: ``ProxyClientEngine._http_json`` folds an HTTP error into ONE message: ``proxy <url> returned <status>: <body>``
#: (the body is the proxy's JSON error object, ``proxy._error``; the ledger's relay adds ``via``). The class does not
#: survive HTTP and the client attaches no fields, so the WIRE SHAPE is read back out of that message — the status
#: and the JSON object — never the prose of the refusal itself.
_PROXY_ERROR_STATUS_BODY = re.compile(r" returned (\d{3}): (.*)\Z", re.DOTALL)


def _proxy_refusal_of(e: BaseException) -> "dict | None":
    """The typed refusal a client-side ``ProxyError`` carries, as ``{"status": int, "error": {...}}`` (the proxy's
    error object verbatim: ``message`` / ``type`` / ``code`` and any extra such as the ledger's ``via``); ``None``
    when the error carries no such body — a transport failure, a non-JSON body, a body with no ``error`` object —
    so the caller treats it exactly as before. Feed the result to :func:`contracts.off_arm_refusal_evidence`
    for the reading; this helper takes no view on the code."""
    m = _PROXY_ERROR_STATUS_BODY.search(str(e))
    if m is None:
        return None
    try:
        payload = json.loads(m.group(2))
    except ValueError:
        return None
    err = payload.get("error") if isinstance(payload, Mapping) else None
    if not isinstance(err, Mapping):
        return None
    return {"status": int(m.group(1)), "error": dict(err)}


def _off_arm_typed_refusal(e: BaseException) -> "dict | None":
    """:func:`contracts.off_arm_refusal_evidence` over :func:`_proxy_refusal_of`: the exact ``400
    thinking_regime_mismatch`` (direct or ``via: ledger_tokenize``) as the record the evidence carries, else None."""
    wire = _proxy_refusal_of(e)
    return tc.off_arm_refusal_evidence(wire["status"], wire["error"]) if wire is not None else None


def _header_ids_from_renders(full: tuple, head: tuple, arm: str, source: str) -> tuple:
    """The appended generation-prompt header: ``add_generation_prompt`` True
    minus False, refusing when False is not a token PREFIX of True (the
    header cannot then be read off the pair at all)."""
    if full[:len(head)] != head:
        raise HeaderMeasurementError(
            f"the served template's add_generation_prompt=False render is not a token "
            f"prefix of the True render under the {arm!r} arm (source {source}) — the "
            "appended assistant header cannot be read off the renders, so the per-task "
            "arms cannot be compared")
    return full[len(head):]


def _render_token_ids(out, *, arm: str, what: str) -> tuple:
    """The token ids of ONE ``apply_chat_template(..., tokenize=True)`` render, whatever SHAPE the tokenizer
    returned them in: a list / tuple of ints (transformers 4.x's default), or a mapping carrying ``input_ids``
    (``BatchEncoding`` — transformers 5.x defaults ``return_dict=True``; ``tuple()`` over it yields its KEYS,
    ``('input_ids', 'attention_mask')``, identical for both renders, so the header difference read as EMPTY —
    the GSM8K ``none`` child's ``'off': []``). Anything else refuses by name: a shape the header cannot
    be read off is never compared."""
    if isinstance(out, Mapping):
        ids = out.get("input_ids")
        if isinstance(ids, (list, tuple)) and ids and all(isinstance(r, (list, tuple)) for r in ids):
            ids = ids[0] if len(ids) == 1 else None          # a batch of one render (return_tensors-less batching)
    else:
        ids = out
    if not isinstance(ids, (list, tuple)) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
        raise HeaderMeasurementError(
            f"the local tokenizer's {what} render under the {arm!r} arm is not a list of token ids "
            f"({type(out).__name__}{': keys ' + str(sorted(out)) if isinstance(out, Mapping) else ''}) — "
            "the appended assistant header cannot be read off it (F-AJ: transformers 5.x returns a BatchEncoding "
            "by default, whose iteration yields its keys); refusing rather than compare a shape")
    return tuple(int(t) for t in ids)


def _header_ids_under_arm(tok, arm: str) -> tuple:
    """The generation-prompt assistant header ids a LOCAL tokenizer appends
    under ONE thinking arm — the same difference
    ``chat_markers.generation_header_ids`` takes (``add_generation_prompt``
    True minus False), but with that arm's ``chat_template_kwargs`` passed
    into the render, which that helper does not accept. Each render's ids are
    read through :func:`_render_token_ids` (a list under transformers 4.x, a
    ``BatchEncoding`` under 5.x — F-AJ)."""
    kw = dict(tc.chat_template_kwargs(arm).get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {})   # EMPTY under the no-switch arm
    probe = [dict(m) for m in HEADER_PROBE_MESSAGES]
    render = functools.partial(tok.apply_chat_template, probe, tokenize=True, **kw)
    return _header_ids_from_renders(
        _render_token_ids(render(add_generation_prompt=True), arm=arm, what="add_generation_prompt=True"),
        _render_token_ids(render(add_generation_prompt=False), arm=arm, what="add_generation_prompt=False"),
        arm, HEADER_SOURCE_LOCAL)


def _served_header_ids_under_arm(tokenize, vmodel: str, arm: str) -> tuple:
    """The same header, measured through the LIVE server's ``/v1/tokenize``
    (the proxy's read-only pass-through, ``ProxyClientEngine.tokenize``): the
    render the model is ACTUALLY served, template pins and all, rather than a
    second derivation from a separately loaded tokenizer."""
    def ids(agp: bool) -> tuple:
        body = {"model": vmodel,
                "messages": [dict(m) for m in HEADER_PROBE_MESSAGES],
                "add_generation_prompt": bool(agp),
                **tc.chat_template_kwargs(arm)}
        resp = tokenize(body)
        toks = resp.get("tokens") if isinstance(resp, Mapping) else None
        if not isinstance(toks, (list, tuple)) or not toks \
                or not all(isinstance(t, int) and not isinstance(t, bool) for t in toks):
            raise HeaderMeasurementError(
                f"/v1/tokenize answered the {arm!r} arm's "
                f"add_generation_prompt={bool(agp)} render with no usable 'tokens' list "
                f"({resp!r}) — the served render cannot be read")
        return tuple(int(t) for t in toks)

    return _header_ids_from_renders(ids(True), ids(False), arm,
                                    HEADER_SOURCE_SERVED)


def _local_template_hash(tok) -> object:
    from sisteer.models import chat_template_hash
    try:
        return chat_template_hash(tok)
    except Exception:                                        # noqa: BLE001
        return None


#: Which side of the local-vs-served chat-template comparison was unavailable
#: (E50 #6) — named in the record and in the refusal, because "restore
#: /v1/tokenize" and "re-stand-up on a tree that records the hash" are
#: different remedies and a reader must not have to guess which applies.
TEMPLATE_SIDE_LOCAL = "local_chat_template_hash"
TEMPLATE_SIDE_SERVED = "served_chat_template_hash"


def template_comparison(*, required: bool, local_hash, served_hash) -> dict:
    """Did the local-vs-served chat-template comparison actually RUN, and what
    did it say (E50 #6)?

    The comparison is what licenses a LOCAL header measurement to stand for the
    SERVED render: it is ``required`` exactly when the measurement fell back to
    a local tokenizer on a LENS route (a served measurement IS the served
    render and needs no comparison; a non-lens route has no mask to misalign).

    Before this it was written inline as ``local is not None and served is not
    None and local != served`` — three conditions in one expression, so the two
    NULL cases (``/v1/tokenize`` down AND a standup that predates
    ``chat_template_hash`` in ``endpoint.json``) fell out of the refusal
    silently and the record still read ``verified: true``, off locally
    invariant headers alone. That is a green tick over a comparison that never
    happened. ``ran`` is now the fact the caller keys on, and a null on either
    side names the SIDE (:data:`TEMPLATE_SIDE_LOCAL` /
    :data:`TEMPLATE_SIDE_SERVED`) rather than reporting an undifferentiated
    failure."""
    unavailable = [side for side, value in ((TEMPLATE_SIDE_LOCAL, local_hash),
                                            (TEMPLATE_SIDE_SERVED, served_hash))
                   if value is None]
    ran = not unavailable
    reason = None
    if unavailable:
        reason = (f"{' and '.join(unavailable)} is null — "
                  + " ".join(
                      "the local tokenizer's chat template could not be hashed"
                      if side == TEMPLATE_SIDE_LOCAL else
                      "the standup's endpoint.json records no chat_template_hash "
                      "(a server stood up before that key existed)"
                      for side in unavailable))
    return {"required": bool(required), "ran": ran,
            "equal": (local_hash == served_hash) if ran else None,
            "unavailable": unavailable, "reason": reason}


def measure_thinking_header_by_arm(by_task: dict, *, lens: bool, tok=None,
                                   tokenize=None, vmodel=None,
                                   served_template_hash=None,
                                   allow_out_of_support: bool = False,
                                   measure_both_arms: bool = False,
                                   header_located: bool = True,
                                   standup_thinking_mode=None):
    """MEASURE the per-arm assistant header and say whether this run may mix
    arms — returning ``(block, error)`` rather than raising, so the CLI can
    record a FAILED measurement in ``evidence.json`` before it exits (E45 M5;
    the same discipline the steering-bites control already follows).

    A LENS route may mix thinking arms in ONE standup only if the served
    template's assistant header is the SAME under both (U5-4; measured
    2026-09-04, the thinking-header measurement).

    Why this is a precondition and not a nicety: a fitted object's response
    mask is keyed on ``response_header_len`` — the trailing assistant header —
    as measured AT FIT TIME, and every fitted object was derived under a
    thinking-OFF render (``models.pin_reasoning_template_off`` fires for any
    template that mentions ``enable_thinking``, gemma-4's included). If the
    header width moves with the arm, serving the other arm masks a DIFFERENT
    span than was fit: a different intervention, with entirely plausible
    numbers. Measured: gemma-4 E4B / E2B signal thinking with a LEADING system
    turn and leave the trailing header at H=3 in both arms — invariant, which
    is the case this lane targets; Qwen3.5 moves H=7 (OFF: an empty
    ``<think></think>`` inside the header) to H=5 (ON: left open), so a mixed
    standup there would steer two tokens of the model's own thinking as if they
    were the response. Refused BY MEASUREMENT rather than by a model
    allow-list, which would go stale on the next family. A non-lens route has
    no mask to misalign and is exempt — the block records that it was checked.

    WHERE THE MEASUREMENT COMES FROM (E45 M5). ``/v1/tokenize`` through the
    live proxy FIRST: that is the render the server will actually apply,
    including the thinking-OFF template pin the standup wrote and launched
    with. A locally loaded ``AutoTokenizer`` is a SECOND derivation of the
    same thing and can legitimately differ (``pin_reasoning_template_off`` is
    a no-op on llama / gemma-4 and is NOT on Qwen3.5 / GLM-4.6V), so it is
    only a FALLBACK and never a silent one: the block records ``source``,
    ``fallback_reason``, and both template digests, and the fallback REFUSES
    on a lens route when its own template hash disagrees with the
    ``chat_template_hash`` the standup recorded in ``endpoint.json`` — a
    measurement of a template nobody is serving proves nothing about the mask.

    E50 #6: it also refuses when that comparison CANNOT BE MADE. Either digest
    may be null — ``/v1/tokenize`` down and a standup that predates the
    ``chat_template_hash`` key is exactly the pair that produces both — and the
    comparison used to be a single ``local is not None and served is not None
    and local != served`` expression, so a null on either side skipped the
    refusal and the record was filed ``verified: true`` on locally invariant
    headers alone. ``verified`` is now true ONLY when a required comparison
    actually ran and agreed; the ``template_comparison`` sub-record says
    whether it was required, whether it ran, what it said, and which SIDE was
    unavailable (:func:`template_comparison`), and ``main`` — the consumer that
    requires verification — refuses on the returned error. Not lifted by
    ``allow_out_of_support``: like its two siblings (an unmeasurable header, a
    local template that disagrees) it is about measuring nothing, which the
    override is not for.

    Returns ``(None, None)`` when the map holds a single arm — nothing to
    compare. Otherwise ``block`` always carries the ids it managed to measure
    and ``verified``; ``error`` is a :class:`HeaderMeasurementError` (or its
    :class:`MixedArmHeaderMoved` subclass) when the run must refuse.

    PW-oos (U10 item 5). ``allow_out_of_support`` keeps the MEASUREMENT and
    drops only the REFUSAL, and it does one more thing: it measures BOTH arms
    even on a SINGLE-arm run. A whole-arm thinking-ON standup has nothing to
    compare by the rule above, so without this it would record no header at
    all — and the two widths ARE the finding the authorised experiment exists
    to state (Qwen3.5 H=7 OFF vs H=5 ON, against an object fit at 7). The
    block then carries ``header_moves``, both arms' ids, and the refusal that
    was overridden, and ``error`` stays None for the moved-header case. The
    OTHER refusals stand: an unmeasurable header and a local tokenizer whose
    template is not the served one are still errors, because the override is
    about serving out of support, not about measuring nothing.

    On the ON-pinned standup, ``standup_thinking_mode`` is the bound
    record's attested ``thinking_mode``. The live ``/v1/tokenize`` is tried
    PER ARM. A live measurement that FAILS is an error, recorded and returned
    — never silently replaced by the local tokenizer (the broad fallback this
    replaces is why an earlier run died at the render probe and not here).
    ONE narrow exception: on a standup attesting ``on`` the proxy refuses the
    OFF arm's render by design (``400 thinking_regime_mismatch``, direct or
    from the usage ledger); that exact refusal sends the OFF arm — and only
    the OFF arm — to the local tokenizer as a recorded REFERENCE
    (``source_by_arm``, ``off_header_source: local_tokenizer``,
    ``off_header_role: local_reference``, ``off_header_fallback_reason``,
    ``off_arm_refusal``, ``off_arm_accepted_under``); the block's ``source``
    then reads :data:`HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF`, never
    wholly ``served_tokenize``. The same refusal on a record attesting
    anything else (or none) is an error by name; the ON arm never falls back.
    With no tokenize client at all the whole measurement is the recorded
    local fallback, as before (an explicit configuration, not a masked
    failure)."""
    arms = sorted(set(by_task.values()))
    mixed = len(arms) > 1
    if allow_out_of_support or measure_both_arms:
        # both arms, whatever this run's map says — see the docstring; also
        # a thinking-ON run measures BOTH arms so the live render can be certified
        # against the family fact (`measure_both_arms`), a MEASUREMENT, never a
        # refusal on a single-arm run
        arms = sorted({tc.THINKING_OFF, tc.THINKING_ON, *arms})
    if len(arms) < 2:
        return None, None
    fallback_reason = None
    ids: dict = {}
    source_by_arm: dict = {}
    off_fallback: "dict | None" = None      # The OFF arm's by-design refusal, when it sent that arm local
    error = None
    if tokenize is not None and vmodel:
        for arm in arms:
            try:
                ids[arm] = _served_header_ids_under_arm(tokenize, vmodel, arm)
                source_by_arm[arm] = HEADER_SOURCE_SERVED
            except HeaderMeasurementError as e:
                # the live render answered but the header cannot be read off it — a measured refusal, as ever
                error = e
                break
            except Exception as e:                               # noqa: BLE001
                typed = _off_arm_typed_refusal(e) if arm == tc.THINKING_OFF else None
                if typed is not None:
                    accepted, why_not = tc.off_arm_refusal_accepted(typed, standup_thinking_mode=standup_thinking_mode)
                    if accepted:
                        source_by_arm[arm] = HEADER_SOURCE_LOCAL
                        off_fallback = {"refusal": typed,
                                        "reason": (f"the live /v1/tokenize refused the {arm!r} arm's render by design "
                                                   f"({typed['status']} {typed['code']}"
                                                   f"{', via ' + typed['via'] if typed.get('via') else ''}): {typed['message']}")}
                        continue
                    error = HeaderMeasurementError(
                        f"the live /v1/tokenize REFUSED the {arm!r} arm's render as a thinking-ON standup's "
                        f"({typed['status']} {typed['code']}"
                        f"{', via ' + typed['via'] if typed.get('via') else ''}), and that refusal is not admissible as "
                        f"OFF-arm evidence here: {why_not} — a local tokenizer is not silently substituted (D105)")
                    break
                error = HeaderMeasurementError(
                    f"the served header could not be measured through the live /v1/tokenize under the {arm!r} arm "
                    f"({type(e).__name__}: {e}) — a failed live measurement is refused, never silently replaced by "
                    "a local tokenizer (D105); restore the proxy's /v1/tokenize")
                break
    else:
        fallback_reason = ("no live /v1/tokenize was available to this run "
                           "(no tokenize client or no virtual model)")
        source_by_arm = {arm: HEADER_SOURCE_LOCAL for arm in arms}
    local_arms = [a for a in arms if source_by_arm.get(a) == HEADER_SOURCE_LOCAL]
    if error is None and source_by_arm and all(source_by_arm.get(a) == HEADER_SOURCE_SERVED for a in arms):
        source = HEADER_SOURCE_SERVED
    elif error is None and off_fallback is not None and local_arms == [tc.THINKING_OFF]:
        source = HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF
        fallback_reason = off_fallback["reason"]
    elif error is None and local_arms:
        source = HEADER_SOURCE_LOCAL
    else:
        # a failed live measurement: what WAS the intended source, with the error beside it (no fallback happened)
        source = HEADER_SOURCE_SERVED if (tokenize is not None and vmodel) else HEADER_SOURCE_LOCAL
    local_hash = None
    if error is None and local_arms:
        if tok is None:
            error = HeaderMeasurementError(
                "the per-arm assistant header could not be measured: the live "
                f"/v1/tokenize was unusable for {local_arms} ({fallback_reason}) and no local tokenizer "
                "was supplied to fall back to")
        else:
            local_hash = _local_template_hash(tok)
            try:
                for arm in local_arms:
                    ids[arm] = _header_ids_under_arm(tok, arm)
                    if off_fallback is not None and arm == tc.THINKING_OFF and not ids[arm]:
                        # under the OFF-reference rule the OFF header is a REFERENCE the certification compares
                        # against the family fact — an EMPTY reference (the local template appended no assistant
                        # header under the OFF kwargs) certifies nothing and is refused by name, never carried on
                        # as `off: []`
                        raise HeaderMeasurementError(
                            f"the local tokenizer appended NO assistant header under the {arm!r} arm (the "
                            "add_generation_prompt=True and =False renders are identical) — the OFF header reference "
                            "the D105 rule admits must be the template's OFF header (non-empty, equal to the family "
                            "fact's off_ids); an empty reference is never compared. Check the local tokenizer's chat "
                            "template and its thinking pin (sisteer.models.pin_reasoning_template_off)")
            except HeaderMeasurementError as e:
                error = e
    off_local_reference = (error is None and source == HEADER_SOURCE_SERVED_ON_LOCAL_REFERENCE_OFF)
    block = {"lens": bool(lens), "source": source,
             "fallback_reason": fallback_reason,
             "served_chat_template_hash": served_template_hash,
             "local_chat_template_hash": local_hash,
             "header_len_by_arm": {a: len(v) for a, v in ids.items()},
             "header_ids_by_arm": {a: [int(i) for i in v] for a, v in ids.items()},
             "invariant": bool(ids) and len(set(ids.values())) == 1,
             # PW-oos: the positive statement of the same fact, because
             # "header_moves: true" is what an out-of-support record is FOR and
             # a reader should not have to negate a field to find it
             "header_moves": (None if not ids else len(set(ids.values())) > 1),
             "run_arms": sorted(set(by_task.values())),
             "measured_arms": list(arms),
             # additive: PER-ARM provenance, and the OFF arm's reference status when it was sent local by rule
             HEADER_KEY_SOURCE_BY_ARM: {a: source_by_arm.get(a) for a in arms},
             HEADER_KEY_OFF_HEADER_SOURCE: source_by_arm.get(tc.THINKING_OFF),
             HEADER_KEY_OFF_HEADER_ROLE: (tc.OFF_HEADER_ROLE_LOCAL_REFERENCE if off_local_reference
                                          else tc.OFF_HEADER_ROLE_SERVED
                                          if source_by_arm.get(tc.THINKING_OFF) == HEADER_SOURCE_SERVED else None),
             HEADER_KEY_OFF_HEADER_FALLBACK_REASON: (off_fallback["reason"] if off_fallback is not None else None),
             HEADER_KEY_OFF_ARM_REFUSAL: (off_fallback["refusal"] if off_fallback is not None else None),
             HEADER_KEY_OFF_ARM_ACCEPTED_UNDER: (tc.OFF_ARM_ACCEPTED_UNDER_ON_PINNED if off_local_reference else None),
             "standup_thinking_mode": standup_thinking_mode,
             # E50 #6: whether the local-vs-served template comparison was
             # REQUIRED, whether it actually RAN, and what it said — so
             # `verified` can never be a green tick over a comparison that
             # never happened (below)
             "template_comparison": template_comparison(
                 required=(bool(local_arms) and bool(lens)),
                 local_hash=local_hash, served_hash=served_template_hash)}
    if allow_out_of_support:
        block["out_of_support_thinking"] = True
        block["out_of_support_thinking_note"] = tc.OUT_OF_SUPPORT_THINKING_NOTE
    comparison = block["template_comparison"]
    if error is None and comparison["required"] and not comparison["ran"]:
        error = HeaderMeasurementError(
            f"the per-arm assistant header was measured off a LOCAL tokenizer "
            f"({fallback_reason}), and whether that tokenizer renders what the server "
            f"renders CANNOT BE CHECKED: {comparison['reason']}. On a lens route that "
            "comparison is the only thing standing between a local measurement and a "
            "claim about the response mask a fitted object was fit under, so an "
            "unavailable comparison is refused rather than assumed to pass. Restore the "
            "proxy's /v1/tokenize (the primary source, which needs no comparison), stand "
            "the server up again on a tree that records chat_template_hash in "
            "endpoint.json, or run this cell under one arm "
            f"({tc.FLAG_THINKING} {'|'.join(tc.THINKING_MODES)})")
    if error is None and comparison["required"] and comparison["ran"] \
            and not comparison["equal"]:
        error = HeaderMeasurementError(
            f"the per-arm assistant header was measured off a LOCAL tokenizer "
            f"({fallback_reason}), but that tokenizer's chat template hashes "
            f"{local_hash} while the standup launched the server with "
            f"{served_template_hash} (endpoint.json chat_template_hash) — the renders "
            "measured are not the renders served, so they prove nothing about the "
            "response mask a fitted object was fit under. Restore the proxy's "
            "/v1/tokenize, or run this cell under one arm "
            f"({tc.FLAG_THINKING} {'|'.join(tc.THINKING_MODES)})")
    # the MIXED-arm refusal keys on a run that genuinely MIXES arms on a
    # HEADER-LOCATED lens mask (the mask that could move) — never on `lens`
    # alone; a single-arm ON run that measured both arms records
    # `header_moves` and refuses nothing (the override keeps its
    # recorded-as-overridden shape on every run, as before)
    if error is None and lens and header_located and not block["invariant"] and (mixed or allow_out_of_support):
        on_tasks = sorted(t for t, a in by_task.items() if a == tc.THINKING_ON)
        moved = MixedArmHeaderMoved(
            "this run mixes thinking arms on a LENS route, but the served template's "
            f"assistant header MOVES between them: {block['header_len_by_arm']} "
            f"({block['header_ids_by_arm']}, measured via {source}). Every fitted object "
            "was derived under a thinking-OFF render, so its response mask is keyed on "
            f"the OFF header; serving the ON tasks ({on_tasks[:3]}) under a different "
            "header masks a different span than was fit — a different intervention with "
            f"plausible numbers. Run this cell under one arm ({tc.FLAG_THINKING} "
            f"{'|'.join(tc.THINKING_MODES)}), or serve the ON tasks in their own standup "
            "(measured 2026-09-04: gemma-4 invariant at H=3, Qwen3.5 moves 7 -> 5)")
        if allow_out_of_support:
            # PW-oos: MEASURED and RECORDED, not refused — the override lifts
            # this one refusal and nothing else, and the message it would have
            # raised travels into evidence.json so the store says what it did
            block["overridden_refusal"] = str(moved)
        else:
            error = moved
    # `verified` stays the honest answer to "does the invariance hold?", so an
    # overridden run reads verified=false with the refusal beside it — never a
    # green tick over a precondition that was waived (PW-oos) or over a
    # comparison that never ran (E50 #6)
    block["verified"] = (error is None and not block.get("overridden_refusal")
                         and not (comparison["required"] and not comparison["ran"]))
    if error is not None:
        block["error"] = str(error)
    return block, error


#: how the NO-SWITCH class read the served header (``thinking_header_by_arm.construction``): the generation-prompt
#: difference (``add_generation_prompt`` True minus False — a template that GATES its assistant header on the flag),
#: or the trailing header of a template that appends it UNCONDITIONALLY (the flag is never consulted, so the
#: difference is EMPTY where the header is always present — ``chat_markers.derive_assistant_markers`` names that
#: diff the FORBIDDEN comparison; the NousResearch Llama-3.1 mirror is exactly this template, measured 2026-09-12
#: by the GSM8K no-switch smoke, job 490518)
HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF = "generation_prompt_diff"
HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING = "unconditional_trailing_header"


def _served_no_switch_header(tokenize, vmodel: str) -> "tuple[tuple, dict]":
    """The NO-SWITCH class's served header, through the LIVE ``/v1/tokenize`` with NO ``chat_template_kwargs``
    (``contracts.chat_template_kwargs(not_available)`` is ``{}``), by the construction the template's own shape
    admits — ``chat_markers.derive_assistant_markers``' rule, served:

    1. render the probe user turn with ``add_generation_prompt`` True (``r_true``) and False (``r_false``); False
       must be a token prefix of True (else the header cannot be read off the pair — refused, as before);
    2. GATED template (``r_true != r_false``): the header IS the difference (:data:`HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF`);
    3. agp-INVARIANT template (``r_true == r_false`` — the flag is never consulted): the content-substitution probe —
       the same user turn plus an assistant turn carrying each of two sentinels sharing no words, rendered WITHOUT the
       flag; the first differing token is where assistant content begins (``content_start``), the common suffix is
       the completed turn's closing scaffold (its terminator, then whatever the template appends after every turn).
       The header is unconditional iff the served generation prompt ends EXACTLY where the content begins
       (``len(r_true) == content_start`` and ``r_true`` is the probe render's prefix — chat_markers' acceptance rule:
       a template that ignores the flag WITHOUT appending the header is refused there and here) and the closing
       scaffold carries a trailing header after its terminator which ``r_true`` ENDS with, by tuple
       (:data:`HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING`). Anything else refuses by name.

    Returns ``(header_ids, facts)``; ``facts`` names the construction, ``agp_invariant``, both render lengths and (under
    3.) ``content_start`` / ``closing_scaffold`` — recorded on the evidence block so a reader sees WHICH construction
    certified the render. Raises :class:`HeaderMeasurementError`."""
    from sisteer import chat_markers as cm

    arm = tc.THINKING_NOT_AVAILABLE
    kwargs = tc.chat_template_kwargs(arm)
    assert not kwargs

    def ids(messages, agp: bool) -> tuple:
        body = {"model": vmodel, "messages": [dict(m) for m in messages], "add_generation_prompt": bool(agp), **kwargs}
        resp = tokenize(body)
        toks = resp.get("tokens") if isinstance(resp, Mapping) else None
        if not isinstance(toks, (list, tuple)) or not toks \
                or not all(isinstance(t, int) and not isinstance(t, bool) for t in toks):
            raise HeaderMeasurementError(
                f"/v1/tokenize answered the {arm!r} arm's add_generation_prompt={bool(agp)} render of "
                f"{len(messages)} message(s) with no usable 'tokens' list ({resp!r}) — the served render cannot be read")
        return tuple(int(t) for t in toks)

    r_true, r_false = ids(HEADER_PROBE_MESSAGES, True), ids(HEADER_PROBE_MESSAGES, False)
    facts: dict = {"render_len_by_agp": {"true": len(r_true), "false": len(r_false)}}
    if r_true[:len(r_false)] != r_false:
        raise HeaderMeasurementError(
            f"the served template's add_generation_prompt=False render is not a token prefix of the True render under "
            f"the {arm!r} arm (source {HEADER_SOURCE_SERVED}) — the appended assistant header cannot be read off the renders")
    if r_true != r_false:
        return r_true[len(r_false):], {**facts, "construction": HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF, "agp_invariant": False}
    # the flag is never consulted: chat_markers' content-substitution probe, served (its private sentinels and
    # `divergence` are the ONE source of that construction)
    probe = list(HEADER_PROBE_MESSAGES)
    fa = ids(probe + [{"role": "assistant", "content": cm._SENTINEL_A}], False)
    fb = ids(probe + [{"role": "assistant", "content": cm._SENTINEL_B}], False)
    if fa == fb:
        raise HeaderMeasurementError(
            "the served template ignores add_generation_prompt, and substituting the assistant content changed nothing "
            "in the render — no assistant turn can be located, so no header can be read")
    s = cm.divergence(fa, fb)
    q = 0
    while q < min(len(fa), len(fb)) - s and fa[len(fa) - 1 - q] == fb[len(fb) - 1 - q]:
        q += 1
    closing = fa[len(fa) - q:]
    facts.update(content_start=s, closing_scaffold=[int(t) for t in closing])
    if s != len(r_true) or fa[:s] != r_true:
        raise HeaderMeasurementError(
            f"the served template ignores add_generation_prompt but its generation prompt ({len(r_true)} tokens) does not "
            f"end where a completed assistant turn's content begins (token {s}) — a template that ignores the flag WITHOUT "
            "appending the assistant header unconditionally (or boundary retokenization across the header); the header "
            "cannot be read (chat_markers.derive_assistant_markers refuses this shape too)")
    if len(closing) < 2:
        raise HeaderMeasurementError(
            f"the served template ignores add_generation_prompt and the completed turn's closing scaffold {list(closing)!r} "
            "carries nothing after its terminator — no unconditional trailing header to read")
    header = tuple(closing[1:])
    if r_true[len(r_true) - len(header):] != header:
        raise HeaderMeasurementError(
            f"the served template ignores add_generation_prompt but its generation prompt does not END with the closing "
            f"scaffold's trailing header {list(header)!r} (it ends {list(r_true[-len(header):])!r}) — the unconditional "
            "header cannot be read off the renders")
    return header, {**facts, "construction": HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING, "agp_invariant": True}


def measure_no_switch_header(*, lens: bool, tokenize, vmodel, served_template_hash=None):
    """The NO-SWITCH class's header measurement (llama-paper-rows P18, predicate
    (A″) rule 1): the appended assistant header measured ONCE through the live
    ``/v1/tokenize`` with NO ``chat_template_kwargs`` — the ONE render a family
    with no thinking switch has — by :func:`_served_no_switch_header` (the
    generation-prompt difference on a gated template; the unconditional trailing
    header on a template that never consults the flag). Returns ``(block, error)``
    in the shape of :func:`measure_thinking_header_by_arm`'s block (``header_ids_by_arm``
    keyed by the one arm, ``measured_arms [not_available]``, ``invariant`` True,
    ``header_moves`` False, ``cert_class no_switch``) plus the construction facts
    (``construction`` / ``agp_invariant`` / ``render_len_by_agp`` …) so evidence.json
    readers see one field list and WHICH construction read the render; a live
    measurement is REQUIRED (no local fallback certifies the served render), so an
    unusable ``/v1/tokenize`` is the error, recorded first."""
    arm = tc.THINKING_NOT_AVAILABLE
    error = None
    ids: dict = {}
    facts: dict = {}
    if tokenize is None or not vmodel:
        error = HeaderMeasurementError("the no-switch header could not be measured: no live /v1/tokenize client "
                                       "(the no-switch class certifies the SERVED render only — no local fallback)")
    else:
        try:
            header, facts = _served_no_switch_header(tokenize, vmodel)
            ids = {arm: header}
        except Exception as e:                               # noqa: BLE001
            error = HeaderMeasurementError(f"the no-switch header could not be measured through the live /v1/tokenize: "
                                           f"{type(e).__name__}: {e}")
    block = {"lens": bool(lens), "source": HEADER_SOURCE_SERVED, "fallback_reason": None,
             "served_chat_template_hash": served_template_hash, "local_chat_template_hash": None,
             "header_len_by_arm": {a: len(v) for a, v in ids.items()},
             "header_ids_by_arm": {a: [int(i) for i in v] for a, v in ids.items()},
             "invariant": True, "header_moves": False,
             "run_arms": [arm], "measured_arms": [arm], "cert_class": tc.CERT_CLASS_NO_SWITCH,
             "template_comparison": template_comparison(required=False, local_hash=None, served_hash=served_template_hash),
             "verified": error is None, **facts}
    if error is not None:
        block["error"] = str(error)
    return block, error


def assert_header_invariant_across_arms(tok, by_task: dict, *, lens: bool, **kw):
    """:func:`measure_thinking_header_by_arm`, raising. Kept for callers that
    have only a tokenizer and want the refusal as an exception; ``main`` uses
    the measuring form so a failure reaches ``evidence.json`` first."""
    block, error = measure_thinking_header_by_arm(by_task, lens=lens, tok=tok, **kw)
    if error is not None:
        raise error
    return block


#: ``steering_bites.probe_render_arm_reason`` values — WHICH rule chose the render the bites probe was sent under
PROBE_RENDER_REASON_RUN_ARM = "run_arm"                                   # the run's own arm (a declared / OFF / non-lens probe)
PROBE_RENDER_REASON_D22_OFF = "d22_undeclared_all_position_off"           # An undeclared all-position LENS cell renders OFF
PROBE_RENDER_REASON_ATTESTED_ON = "attested_thinking_on_native_standup"   # the native standup attests 'on'; OFF is refused by design


def _bites_probe_render_arm(run_arm: "str | None", *, lens_cell: bool, header_located: bool,
                            standup_native: bool, standup_thinking_mode: object) -> "tuple[str | None, str | None]":
    """The thinking arm the steering-bites probe renders under, and WHY (``(arm, reason)``; ``(None, None)`` for a
    legacy run with no arm — the probe is then sent byte-identical to before).

    On an UNDECLARED all-position cell the bites probe renders thinking-OFF on both legs — the LENS
    twin (``none__lens0``) has no served-positions entry and an undeclared ON request to it would refuse; the probe
    measures whether the object BITES (its mask is every row under either render), the ON render is certified
    separately (predicates (A) and (C)).

    On a NATIVE standup whose bound record attests ``thinking_mode: on``
    (:data:`contracts.STANDUP_THINKING_MODE_KEY`, the key the readiness contract holds the row to), the OFF render
    is exactly what the proxy refuses BY DESIGN (``400 thinking_regime_mismatch`` on chat and on the ledger's
    pre-tokenize), and the rationale above is void there: the twin is ``none__lora0``, a non-lens route that admits an
    undeclared ON render, and ``header_located`` is the engine's fact, not a missing record. So the probe renders in
    the ATTESTED regime — ON, with the certified ON arm's kwargs (``render_fragment(lens=True, arm=on,
    header_located=False)`` == ``{"chat_template_kwargs": {"enable_thinking": true}}``, no declaration) on BOTH legs,
    so the bite compares an ON render with an ON render (``unified_sweep_run._probe_channel_policy`` then compares
    reasoning + content re-joined on each leg). A record attesting ``off`` or carrying no ``thinking_mode``, and every
    lens standup, keep the OFF-render rule byte for byte."""
    if run_arm is None:
        return None, None
    if lens_cell and not header_located and tc.enable_thinking(run_arm):
        if standup_native and standup_thinking_mode == tc.THINKING_ON:
            return tc.THINKING_ON, PROBE_RENDER_REASON_ATTESTED_ON
        return tc.THINKING_OFF, PROBE_RENDER_REASON_D22_OFF
    return run_arm, PROBE_RENDER_REASON_RUN_ARM


def _render_arm(thinking: str, by_task: dict) -> str:
    """The arm the manifest's ``thinking.render`` label describes (P6/P25): the
    whole-configuration arm, and under ``per-task`` OFF (hence ``native``) — the
    arm the prefill-scored tasks the label was defined for took under U5, kept
    so a per-task run's hashed label is unchanged now that no row task is
    prefill-scored. ``by_task`` is accepted for the callers' signature."""
    if not tc.is_per_task(thinking):
        return thinking
    return tc.THINKING_OFF


def _render_label(thinking: str) -> str:
    """The ``thinking.render`` label of an arm: ``native`` when the arm sends no
    thinking switch ON, else ``flag_only`` (the switch alone, no thought block
    in the declared message)."""
    if not tc.enable_thinking(thinking):
        return tc.THINKING_RENDER_NATIVE
    return tc.THINKING_RENDER_FLAG_ONLY


def _check_served_reasoning_parser(atoms, thinking: str, endpoint: dict,
                                   by_task=None, *,
                                   allow_out_of_support: bool = False,
                                   render_fact=None,
                                   require_certified: bool = True) -> None:
    """The served reasoning parser must be consistent with the thinking arm
    (PB12 / R4):

    * thinking-ON with a GENERATION task and NO parser served is refused —
      the thought text would land in ``content`` and every extractor
      (ANSWER:, numeric match, choice, code fence) would score the
      scratchpad, with 200s all the way;
    * thinking-ON with a parser served: it must be one of the family pins
      (``contracts.REASONING_PARSER_BY_FAMILY``) — an unpinned name is a
      standup nobody reviewed;
    * thinking-OFF must have served NO parser: the published suite is the
      thinking-OFF render, and a parser splitting ``content`` there is a
      standup that does not match the arm it serves.

    PER-TASK (PW20b-D, U5-4). ``by_task`` is the run's ``{row task: arm}``
    map; the rules above are then applied to THE ATOMS THAT ARE ON, because a
    mixed run under ONE standup needs exactly the parser its ON tasks need.
    What that relaxes, said plainly: the "OFF serves no parser" rule now binds
    only when NOTHING in the run is ON (a parser serving nobody). A mixed
    run's OFF tasks ARE served with a parser composed — the parser is a SERVER
    fact, not an arm fact, so the provider's neither-channel refusal covers
    them exactly as it covers the ON ones, and an OFF task simply emits no
    thought for the parser to split. The alternative (two standups per cell so
    the OFF tasks see no parser) is the whole-configuration design U5-4
    replaced.

    ``allow_out_of_support`` (PW-oos, U10 item 5) turns the FIRST rule —
    thinking-ON with a generation task and no parser served — into a printed
    statement instead of a refusal: a family with no VERIFIED parser is exactly
    the family the authorised experiment serves ON, and the thought text
    landing in ``content`` is one of the things it measures (the unparseable
    rate). The other two rules are untouched — an UNPINNED parser is still
    refused (a standup nobody reviewed is not made reviewable by this flag),
    and so is a parser serving an all-OFF run."""
    served = endpoint["reasoning_parser"]
    if by_task is not None:
        on_atoms = sorted(a for a in atoms
                          if any(by_task.get(t) == tc.THINKING_ON
                                 for t in tc.TASKS_OF_CONTROL[a]))
        if not on_atoms:
            if served:
                raise SystemExit(
                    "[served-cap] every requested task's per-task arm is thinking-OFF but "
                    f"the standup composed reasoning parser {served!r} (endpoint.json): "
                    "the parser would serve nobody and the OFF arm is the published "
                    "render — stand up without --reasoning-parser for this run")
            return
        atoms, thinking = on_atoms, tc.THINKING_ON
    if not tc.enable_thinking(thinking):
        if served:
            raise SystemExit(
                f"[served-cap] thinking-OFF but the standup composed reasoning parser "
                f"{served!r} (endpoint.json): the OFF arm is the published render and "
                "takes no parser — stand up without --reasoning-parser for this arm")
        return
    generating = sorted(atoms)
    if generating and not served:
        # a family whose VERIFIED harness-side split mode is CERTIFIED
        # (`render_fact`) has a parser — the split — so thinking-ON with no
        # server parser is admitted in support, by the fact, never by the
        # override; a `candidate` fact refuses production by name (the dry
        # run's `pre` stage passes require_certified=False)
        fact_why = None
        if render_fact is not None:
            if render_fact.thinking_switch is False:
                # llama-paper-rows (P7 / P18): the driver-side twin of the planner's rule (inspect_served._plan_task) — a
                # family whose template carries NO thinking switch has no thinking-ON render, so its fact (certified for the
                # `not_available` arm; split mode `none`) can never stand in for a parser under thinking-ON. Before the fact
                # was certified (2026-09-12) the `candidate` refusal below covered this by accident; a certified fact needs
                # the rule itself, else a thinking-ON run on the family is admitted and walks on to the tinycap preflight.
                fact_why = (f"family {render_fact.family!r} carries no thinking switch (contracts.NO_THINKING_CHANNEL_FAMILIES; "
                            f"its render fact's split mode is {render_fact.split_mode!r}, its arm {tc.THINKING_NOT_AVAILABLE!r}) — "
                            "a no-switch fact certifies the `not_available` arm, never a thinking-ON render")
            else:
                fact_why = tc.render_fact_refusal(render_fact.family, require_certified=require_certified)
            if fact_why is None:
                print(f"[served-cap] thinking-ON with generation controls {generating} and NO server "
                      f"reasoning parser: the family's render fact ({render_fact.family!r}, split mode "
                      f"{render_fact.split_mode!r}, {render_fact.evidence_status}) IS the parser — the "
                      "scorer splits the think block under that mode (plan §4.3 item 1)", flush=True)
                return
        if allow_out_of_support and fact_why is None:
            print(f"[served-cap] {tc.FLAG_ALLOW_OOS_THINKING}: thinking-ON with generation "
                  f"controls {generating} and NO reasoning parser served — the thought text "
                  "stays in 'content' where every extractor sees it, which is part of what "
                  f"this run measures. {tc.OUT_OF_SUPPORT_THINKING_NOTE}", flush=True)
        else:
            raise SystemExit(
                f"[served-cap] thinking-ON with generation controls {generating} "
                "but the standup composed NO reasoning parser (endpoint.json "
                "reasoning_parser is null): the thought text would be scored as "
                "the answer (PB12) — stand up with --reasoning-parser "
                f"<contracts.REASONING_PARSER_BY_FAMILY[family]> "
                f"({dict(tc.REASONING_PARSER_BY_FAMILY)})"
                + (f"; and the family's render fact does not stand in for one: {fact_why}"
                   if fact_why else ""))
    if served and served not in tc.REASONING_PARSER_BY_FAMILY.values():
        raise SystemExit(
            f"[served-cap] thinking-ON: the standup composed reasoning parser {served!r}, "
            f"which is not a pinned family parser ({dict(tc.REASONING_PARSER_BY_FAMILY)}) "
            "— refusing to record an unreviewed parser as the arm's")


class _AbsentIdentityField:
    """The value a resume guard reads for an identity field the cell manifest never recorded (repr'd by name in the mismatch)."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "<absent from the cell manifest>"

    def __eq__(self, other) -> bool:
        return isinstance(other, _AbsentIdentityField)

    def __hash__(self) -> int:
        return hash("_AbsentIdentityField")


_ABSENT = _AbsentIdentityField()


def _check_manifest_serving_identity(store, endpoint: dict, *, concurrency: int,
                                     pass_no: "int | None") -> None:
    """Resume guard (R4 / E10 N3 / PW5-4 / PW8): an existing cell manifest
    must record the SAME served identity as this run — ``reasoning_parser``
    (``thinking`` block), ``tool_calls`` (``route`` block), the window
    ``max_model_len`` and the served S ``max_num_seqs`` (``geometry`` block)
    and the client width (``concurrency``), i.e. :func:`_served_identity` —
    and the SAME geometry pass (``geometry.pass`` against this run's
    ``pass_no``, E16 A2). All six are hashed into the store identity
    (``resolved_config.tinycap``), so a different standup or pass normally
    lands in a different cell dir; this check catches what the 8-hex key
    prefix cannot (a collision, a copied or hand-edited store) and refuses
    (:class:`ServingIdentityMismatch`) naming old vs new per field. A
    manifest that predates the geometry block records no window and is
    refused too: it cannot vouch for the one it was filled under."""
    if not store.manifest_path.exists():
        return
    rec = json.loads(store.manifest_path.read_text())
    geom = rec.get(tc.MANIFEST_KEY_GEOMETRY) or {}
    route_block = rec.get(tc.MANIFEST_KEY_ROUTE) or {}
    have = {"reasoning_parser": (rec.get(tc.MANIFEST_KEY_THINKING) or {}).get("reasoning_parser"),
            "tool_calls": route_block.get("tool_calls"),
            "max_model_len": geom.get("max_model_len"),
            "max_num_seqs": geom.get("max_num_seqs"),
            "concurrency": rec.get(tc.MANIFEST_KEY_CONCURRENCY),
            # PW18: a manifest without the entry cannot vouch for the regime its
            # tool turns were steered under (None -> named mismatch, like a
            # missing window); a manifest without the reduction mode is LEGACY
            # and maps to per_row — every store before Evals' landing was
            # served per-row (E25), so that mapping is a fact, not a guess
            tc.AGENT_LOOP_ENTRY_KEY: route_block.get(tc.AGENT_LOOP_ENTRY_KEY),
            tc.REDUCTION_MODE_FIELD: route_block.get(tc.REDUCTION_MODE_FIELD,
                                                     tc.REDUCTION_MODE_DEFAULT),
            tc.GEOMETRY_PASS_FIELD: geom.get(tc.GEOMETRY_PASS_FIELD)}
    want = {**_served_identity(endpoint, concurrency=concurrency), tc.GEOMETRY_PASS_FIELD: pass_no}
    if tc.SERVED_ENGINE_FIELD in want:
        # native-serving-default: a NATIVE run's identity carries the engine; the
        # manifest's route block states it (ABSENT on every lens store = lens by
        # era -> a named mismatch, never a resume of a lens store under the native
        # engine); the reduction mode names the same fact a second way
        have[tc.SERVED_ENGINE_FIELD] = route_block.get(tc.SERVED_ENGINE_FIELD)
    if tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE in want:
        # a PROTOCOL standup's identity carries the sampling
        # clamp override; the manifest records it VERBATIM in the hashed served block
        # (`resolved_config.tinycap.served`, written from _served_identity), so that is
        # where a resume reads it back. Before this line the check never read it, and
        # `have[k]` raised KeyError on the first in-container resume under a protocol
        # standup (llama8b_it difference LoRA, BCB-full, acb7434: the whole BCB leg was
        # generated and then lost to a traceback instead of a resume). A manifest whose
        # served block lacks the key is a NAMED mismatch below, never a crash.
        served_block = ((rec.get("resolved_config") or {}).get("tinycap") or {}).get("served") or {}
        have[tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE] = served_block.get(tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE, _ABSENT)
    # every identity key `want` carries is compared, a key the manifest never recorded
    # reads as `_ABSENT` (a named mismatch: the manifest cannot vouch for it) — a new
    # identity field must never turn a resume into a KeyError again
    diff = {k: (have.get(k, _ABSENT), want[k]) for k in want if have.get(k, _ABSENT) != want[k]}
    if diff:
        detail = "; ".join(
            f"{k}: the cell manifest records {old!r} but this run is pass {new!r}"
            if k == tc.GEOMETRY_PASS_FIELD else
            f"{k}: the cell manifest records {old!r} but this standup served {new!r}"
            for k, (old, new) in diff.items())
        raise ServingIdentityMismatch(
            f"[served-cap] {store.cell_dir}: resume mismatch on the served identity — "
            f"{detail}; a store produced under one reasoning parser / tool-call mode / "
            "window / S / client width / agent-loop entry / lens reduction mode / serving engine / "
            "geometry pass never resumes under another (R4, N3, PW5-4, PW8, PW18)")


def _cell_quality(gen_stats: Mapping) -> dict:
    """The run's PER-CELL quality roll-up (U5-5; ``contracts.QUALITY_BLOCK_FIELDS``)
    pooled from the per-task summaries in ``gen_stats``.

    Pooled from the COUNTS, never by averaging rates: tasks have very different
    item counts, so a mean of rates is not the run's rate. A task that does not
    record a measure (MATH-500's sympy scorer records no answer, so it cannot
    say whether extraction failed) is listed by name under
    ``tasks_not_recorded`` and left out of the denominator — "nobody said" and
    "0 %" are different facts and this block keeps them apart. A measure no
    task recorded has ``rate: null``.

    Reads only summaries that CARRY the keys, so the EvalPlus and legacy legs
    (which record neither) are simply not in it."""
    out: dict = {}
    for measure, n_key, d_key in (("unparseable", "n_unparseable", "n_answered"),
                                  ("at_cap", "n_at_cap", "n_stop_reason")):
        n = d = 0
        absent: list = []
        for task, s in sorted(gen_stats.items()):
            if not isinstance(s, Mapping) or d_key not in s:
                continue
            denom = int(s.get(d_key) or 0)
            if denom <= 0:
                absent.append(task)
                continue
            n += int(s.get(n_key) or 0)
            d += denom
        out[measure] = {"n": n if d else None, "denominator": d,
                        "rate": (n / d) if d else None,
                        "tasks_not_recorded": absent}
        assert set(out[measure]) == set(tc.QUALITY_MEASURE_FIELDS)
    out["by_task"] = {
        task: {"unparseable_rate": s.get("unparseable_rate"),
               "at_cap_rate": s.get("at_cap_rate")}
        for task, s in sorted(gen_stats.items())
        if isinstance(s, Mapping) and "unparseable_rate" in s}
    assert set(out) == set(tc.QUALITY_BLOCK_FIELDS)
    return out


def _print_cell_quality(quality: dict) -> None:
    """The run's stdout summary of the two quality measures (U5-5: format
    drift is its own deviation and must be VISIBLE, not buried in a file)."""
    for measure, what in (("unparseable", "unparseable (scorer found no answer)"),
                          ("at_cap", "at the token cap")):
        b = quality[measure]
        if b["denominator"]:
            print(f"[served-cap] quality: {what}: {b['n']}/{b['denominator']} = "
                  f"{b['rate']:.2%}", flush=True)
        else:
            print(f"[served-cap] quality: {what}: not recorded by any task", flush=True)
        if b["tasks_not_recorded"]:
            print(f"[served-cap]   not recorded by: {', '.join(b['tasks_not_recorded'])}",
                  flush=True)
    for task, rates in quality["by_task"].items():
        parts = [f"{k.replace('_rate', '')} "
                 + ("n/a" if v is None else f"{v:.2%}") for k, v in rates.items()]
        print(f"[served-cap]   {task}: {', '.join(parts)}", flush=True)


def _check_manifest_thinking_map(store, *, thinking: str, by_task: dict) -> None:
    """Resume guard on the ARM (PW20b-D, U5-4), the sibling of
    :func:`_check_manifest_serving_identity`.

    ``thinking_by_task`` is hashed identity, so a different map normally
    lands in a different cell dir; this is the check the 8-hex key prefix
    cannot make (a collision, a copied or hand-edited store, a manifest
    written before the map existed). It refuses naming the tasks whose arm
    moved — the failure it exists to stop is a store holding rows generated
    at two different arms for one task, which no aggregate can tell apart
    because both are called ``acc``.

    A manifest that predates the map cannot vouch for the arms its rows were
    generated at, so it is refused like a manifest without the window."""
    if not store.manifest_path.exists():
        return
    rec = json.loads(store.manifest_path.read_text())
    block = rec.get(tc.MANIFEST_KEY_THINKING) or {}
    if "by_task" not in block or "mode" not in block:
        raise ServingIdentityMismatch(
            f"[served-cap] {store.manifest_path.parent}: the cell manifest's "
            f"{tc.MANIFEST_KEY_THINKING!r} "
            "block records no per-task thinking map (a store written before PW20b-D) — it "
            "cannot vouch for the arm its rows were generated at; refusing to resume onto "
            "it. Generate into a fresh out root")
    have_mode, have = block["mode"], dict(block["by_task"] or {})
    moved = {t: (have.get(t), by_task[t]) for t in by_task if have.get(t) != by_task[t]}
    dropped = sorted(set(have) - set(by_task))
    if have_mode != thinking or moved or dropped:
        # EVERY disagreement is named. The parenthesisation matters: a bare
        # `[A] if c else [] + B + C` binds the ternary loosest, so it would
        # report only the mode whenever the mode moved and hide the arms.
        parts = ([f"mode: the manifest records {have_mode!r}, this run is {thinking!r}"]
                 if have_mode != thinking else [])
        parts += [f"{t}: the manifest records arm {old!r}, this run is {new!r}"
                  for t, (old, new) in sorted(moved.items())]
        if dropped:
            parts.append(f"the manifest also records {dropped}, which this run does not")
        raise ServingIdentityMismatch(
            f"[served-cap] {store.manifest_path.parent}: resume mismatch on the per-task "
            f"thinking map — {'; '.join(parts)}; one arm per task (U5), and a store filled "
            "under one map never resumes under another (PW20b-D)")


def _served_stamp(*, cfg: dict, model_id: str, model_revision, route, tok_ev: dict) -> dict:
    """The served path's serving-identity stamp for the controls-summary
    provenance — the fields ``sisteer.models.serving_identity`` records on
    the HF path (requested/resolved model + revision, the LIVE tokenizer's
    chat-template hash) plus the route that actually served: the virtual
    model, the serving identity and the engine."""
    return {"requested_model": cfg.get("model"),
            "requested_revision": cfg.get("model_revision"),
            "resolved_model_id": model_id,
            "resolved_revision": model_revision,
            "chat_template_hash": tok_ev["served_template_hash"],
            "virtual_model": route.virtual_model,
            "serving_identity": route.serving_identity,
            "engine": "vllm"}


def _tinycap_manifest(*, atoms, thinking: str, args, route, routes,
                      identity, endpoint, out_root=None, provenance=None,
                      sampling=None, attestation=None, bank_view=None) -> dict:
    """The additive cell-manifest blocks for a tinycap run
    (``contracts.TINYCAP_MANIFEST_KEYS``, every one filled from an actual
    input — R4): thinking / geometry / concurrency / route / declaration /
    seeds / frameworks / usage_ledger / sandbox / tinycap, plus the
    ``provenance`` block the controls-summary merge writes under.
    ``route.grader_virtual_model`` is READ from the table's ``none`` entry
    (P20) and is None when the table has no such route;
    ``thinking.reasoning_parser``, ``route.tool_calls`` and ``geometry`` are
    READ from the standup's ``endpoint`` record (PB13: what was served, not
    what was asked for); ``usage_ledger`` is the DURABLE ledger path under
    ``out_root`` (PD5), never the ephemeral URL; ``frameworks`` is the
    identity's block (``identity["frameworks"]``: the imported, pin-checked
    versions for the requested families + the client lock sha — computed
    ONCE at startup by :func:`_framework_versions` and hashed, PW5-2).

    PW20b-D: the ``thinking`` block also carries ``mode`` (the driver's
    ``--thinking`` value) and ``by_task`` (the mirror of the hashed per-task
    map read by :func:`_check_manifest_thinking_map`); ``enable_thinking`` is
    NULL under ``per-task``, where no one per-request boolean exists — the map
    is the answer to "was this thinking?" and the block says so by refusing to
    invent a single bit.

    PW-sampling (U8): ``sampling`` (:func:`_resolve_sampling`'s record) fills
    the ``sampling`` block — the hashed sampling identity's mirror plus the
    mode, the model key and the row's source — and the ``seeds`` block's
    per-task ``samples_per_item`` maps."""
    families = {tc.control_family(a) for a in atoms}
    native = _endpoint_is_native(endpoint, source="standup record")
    grader = pe.routes_by_cell(routes).get(tc.ROUTE_NONE_CELL)
    by_task = dict(identity[tc.IDENTITY_KEY_THINKING_BY_TASK])
    render = _render_label(_render_arm(thinking, by_task))
    # no tinycap row task declares a response prefill on the lens route (the
    # prefill-scored family that did is removed); the block keeps its key
    declared: list = []
    ledger_path = (str(Path(out_root) / tc.LEDGER_FILENAME)
                   if (args.ledger_url and out_root is not None) else None)
    out = {
        tc.MANIFEST_KEY_THINKING: {
            # the switch value every request sent: None under per-task (mixed) and under a
            # no-switch family (no request carried one — `by_task` says not_available)
            "enable_thinking": (None if tc.is_per_task(thinking) or not tc.sends_thinking_kwargs(thinking)
                                or any(not tc.sends_thinking_kwargs(a) for a in (by_task or {}).values())
                                else tc.enable_thinking(thinking)),
            "render": render,
            "reasoning_parser": endpoint["reasoning_parser"],
            "mode": thinking,
            "by_task": by_task},
        tc.MANIFEST_KEY_GEOMETRY: _geometry_block(endpoint, out_root),
        tc.MANIFEST_KEY_CONCURRENCY: int(args.concurrency),
        tc.MANIFEST_KEY_ROUTE: {
            "virtual_model": route.virtual_model,
            "grader_virtual_model": (grader.virtual_model if grader is not None else None),
            "serving_identity": route.serving_identity,
            "tool_calls": bool(endpoint["tool_calls"]),
            # PW18: the proxy's agent-loop entry and the lens reduction mode the
            # standup served (endpoint.json; absent mode = per_row) — mirrors of
            # the hashed served identity, read by the resume guard
            tc.AGENT_LOOP_ENTRY_KEY: bool(endpoint[STANDUP_ENTRY_KEY]),
            tc.REDUCTION_MODE_FIELD: _served_reduction_mode(endpoint, source="standup record"),
            # the served ENGINE, UNHASHED here
            # (the hashed copy is tinycap.served.serving_engine), PRESENT IFF native —
            # what the stored-record consumers read a store's engine from
            # (contracts.read_store_reduction_mode: pairing, cell_controls, the
            # launcher's output-boundary conflict); a lens route block is
            # byte-identical (absent = lens by era), and the lens reduction-mode
            # EXPECTATION is recorded not consulted on this engine
            **({tc.SERVED_ENGINE_FIELD: tc.SERVING_ENGINE_LORA_NATIVE,
                ROUTE_KEY_REDUCTION_MODE_EXPECTATION: REDUCTION_MODE_EXPECTATION_NOT_CONSULTED}
               if native else {}),
            # fit-support transfer (contract §2.5.1 <ROUTE_KEY>): the RAW
            # lens_serve_declarations map the standup attested, copied whole,
            # UNHASHED (the hashed identity carries the expectation literal),
            # with the served digests / salt policy / attested policy beside it
            **_attestation_route_block(attestation),
            # bcb-bfcl-200: the BigCodeBench sandbox facts + the owner tuple,
            # UNHASHED (the image digest and the verifier timeout ARE hashed,
            # under tinycap.bcb_sandbox) — PRESENT ONLY on a run that files the
            # leg, so every other store's route block is byte-identical
            **({"bcb_sandbox": _bcb_sandbox_route_block(args, by_task)}
               if tc.files_any_bigcodebench(by_task) else {}),
            # llama-paper-rows: the bank view the run's objects come from (`--bank-view`), UNHASHED,
            # PRESENT ONLY when the flag was given — every other store's route block is byte-identical
            **({"bank_view": bank_view} if bank_view is not None else {})},
        tc.MANIFEST_KEY_DECLARATION: {"response_prefill_used": declared},
        tc.MANIFEST_KEY_SEEDS: _seeds_block(families, sampling.samples_by_task, **_seed_policy_kw(sampling)),
        tc.MANIFEST_KEY_SAMPLING: _sampling_block(sampling),
        tc.MANIFEST_KEY_FRAMEWORKS: dict(identity[tc.MANIFEST_KEY_FRAMEWORKS]),
        tc.MANIFEST_KEY_USAGE_LEDGER: ledger_path,
        tc.MANIFEST_KEY_SANDBOX: _sandbox_block(families),
        tc.MANIFEST_KEY_TINYCAP: dict(identity),
    }
    if provenance is not None:
        out[tc.MANIFEST_KEY_PROVENANCE] = dict(provenance)
    return out


#: THE CONTROLS PUBLICATION SINK: the
#: ``controls_events.ControlsEventSink`` whose ``publish`` the driver's :func:`_fanout` hands
#: ``proxy_engine.bounded_gather`` as its completion-order ``on_done`` while a sampled controls
#: leg that asked for the durable per-draw stream (``--controls-events``) is generating. A
#: ContextVar, not a parameter, on purpose: the controls writer
#: ``sampled_controls._run_judged_served`` is the SHARED body the vendor contract pins byte for
#: byte (``tests/test_sampled_controls_drift.py``; the vendored copy on another lane cannot be
#: re-mirrored from here), and it calls the seam's ``_fanout(post, work, concurrency=...)`` with
#: no hook — so the driver sets the sink around that call and its ``_fanout`` reads it. Unset
#: (the default) = today's fan-out, byte for byte; the MMLU leg's ``_fanout_responses`` never
#: reads it.
_CONTROLS_EVENT_SINK: "contextvars.ContextVar[Optional[cev.ControlsEventSink]]" = \
    contextvars.ContextVar("controls_event_sink", default=None)


async def _fanout_responses(post, work, *, concurrency: int, on_done=None):
    """The bounded fan-out (``proxy_engine.bounded_gather``: a semaphore,
    every failure COLLECTED, no client-side retry — a proxy refusal such as
    a 409 salt-discipline violation must surface, not be re-rolled) for
    legs that need the FULL response dict (the MMLU letter scorer reads
    logprobs, not message text). ``work`` is ``[(item_id, payload)]``;
    returns ``(responses, errors)`` with ``responses = {item_id: response}``.
    ``on_done`` (additive; default None) is passed through to
    ``bounded_gather`` untouched: ``on_done(item_id, response_or_exception)``
    per request, in completion order."""
    return await pe.bounded_gather(
        [(item_id, functools.partial(post, payload)) for item_id, payload in work],
        concurrency=concurrency, on_done=on_done)


async def _fanout(post, work, *, concurrency: int, on_done=None):
    """:func:`_fanout_responses` plus the text read: ``(results, errors,
    finish_reasons, meta)`` with ``results = {item_id: text}`` and ``meta =
    {item_id: {finish_reason, completion_tokens, prompt_tokens, max_tokens,
    reasoning_content}}`` (the response's finish_reason and ``usage`` counts —
    None where the server sent none — plus the request's cap; the sampled
    re-run's sidecar reads them); a response without a readable choice is a
    collected ``malformed response`` error. The callers refuse to summarize a
    partial run (spool/rows resume, so a re-invoke fills only the gaps).

    ``on_done`` (additive; default None): the completion-order callback
    handed to ``bounded_gather``. When none is given and a controls
    publication sink is set (:data:`_CONTROLS_EVENT_SINK`, the sampled
    controls leg under ``--controls-events``), the sink publishes one
    ``controls_events.jsonl`` event per completed request from the request's
    own payload (seed, cap) and its response. The maps returned are the same
    either way."""
    if on_done is None:
        sink = _CONTROLS_EVENT_SINK.get()
        if sink is not None:
            on_done = sink.on_done_for(work)
    responses, errors = await _fanout_responses(post, work, concurrency=concurrency,
                                                on_done=on_done)
    payloads = dict(work)
    results: dict = {}
    finish: dict = {}
    meta: dict = {}
    for item_id, resp in responses.items():
        try:
            choice = resp["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            fr = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as e:
            errors.append((item_id, f"malformed response: {e!r}"))
            continue
        usage = resp.get("usage") if isinstance(resp, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        results[item_id] = text
        finish[fr] = finish.get(fr, 0) + 1
        meta[item_id] = {"finish_reason": fr,
                         "completion_tokens": usage.get("completion_tokens"),
                         "prompt_tokens": usage.get("prompt_tokens"),
                         "max_tokens": (payloads.get(item_id) or {}).get("max_tokens"),
                         # a reasoning-parser server returns the thought
                         # channel here, apart from the answer
                         "reasoning_content": (choice.get("message") or {}).get(
                             "reasoning_content")}
    return results, errors, finish, meta


# --------------------------------------------------------------------------- #
# store identity: cell_controls_run's layout, engine="vllm"                    #
# --------------------------------------------------------------------------- #


def _gen_cfg(cfg: dict) -> dict:
    """The judged/parity gen config, IDENTICAL to the HF driver's
    (_judged_specs): controls_gen_config + batch_size 8. batch_size is
    meaningless for the async fan-out, but it is part of the spool header's
    gen_cfg — the judge phase rebuilds the header from the same recipe, so
    dropping it here would make judging refuse every spool we write."""
    from sisteer.eval.cell_controls import controls_gen_config

    return {**controls_gen_config(cfg), "batch_size": 8}


def _grader_max_tokens(explicit: "int | None" = None) -> int:
    """The model GRADER's completion cap (E56 (3)) — ``inspect_served``'s
    number, imported HERE because that module is deliberately loaded lazily
    (it pulls the eval stack) and a module-level default would import it at
    startup. ONE resolver, so the value hashed into the store identity and
    the value the grader is actually built with cannot be two numbers."""
    if explicit is not None:
        return int(explicit)
    from sisteer.capeval.tiers import inspect_served as _isv

    return int(_isv.GRADER_MAX_TOKENS)


def _resolved_config(cell, cfg: dict, system, limit, judge_model: str, *,
                     route, model_id: str, model_revision,
                     top_logprobs: int = MMLU_TOP_LOGPROBS,
                     grader_max_tokens: "int | None" = None,
                     tinycap: "dict | None" = None, prov=None,
                     serving_engine: "str | None" = None,
                     mmlu_full: bool = False) -> dict:
    """This driver's OWN config identity (hashed into the cell key). It
    deliberately DIFFERS from cell_controls_run's — engine and serving
    route are identity here — so a served store can never resume onto (or
    be mistaken for) an HF store of the same cell.

    ``prov`` (the cell's fit-time provenance, None for a baseline) adds
    ``sweep.FIT_PROVENANCE_TEMPLATE_CHECK_KEY`` — the CLASS of what the
    served==trained template check did (sweep.TEMPLATE_CHECK_BY_STATUS) —
    whenever it is NOT in sweep.TEMPLATE_CHECK_UNHASHED ("compared" /
    "baseline"). Two stores emitting identical numbers under different
    provenance must not be equal witnesses at merge:
    the fit KIND lives only in the object's sidecar (train_budget_meta.json
    / derived_meta.json), never inside the intervention hash (H(artifact ||
    object.json sidecar || resolved spec)), so a document-fit / impossible /
    unverified row and a chat-fit row of the same cell would otherwise share
    a store key. GATED so every existing store's chash is byte-identical —
    an unconditional identity field would re-key the 57-cell tiny-capability
    wave and every two-SI store (R18-A). The TOTAL class ("compared" spelled
    out) is stated top-level on every manifest, unhashed, and a store whose
    manifest carries neither is dated by tinycap.contracts.store_era, never
    read as "compared" (sweep.template_check_of_manifest).

    ``serving_engine``: the ENGINE the run is served by — ``lora_native`` adds
    ``serving.serving_engine`` to the hashed identity, PRESENT IFF native. A
    passthrough leg (``none``, a prompted cell) spells the same
    ``proxy_vllm+passthrough+…`` identity on both engines, and the legacy
    controls (XSTest / OR-Bench / MMLU) have no ``tinycap.served`` block to
    carry the engine, so without this key a native-era store of such a cell
    would key like — and resume onto — the lens-era one. ``None`` / ``lens``
    adds nothing: every lens chash is byte-identical."""
    if serving_engine not in (None, *tc.SERVING_ENGINES):
        raise ValueError(f"serving_engine {serving_engine!r} is not one of {list(tc.SERVING_ENGINES)}")
    resolved = {
        "driver": DRIVER_VERSION,
        "engine": "vllm",
        "serving": {"virtual_model": route.virtual_model,
                    "serving_identity": route.serving_identity,
                    **({tc.SERVED_ENGINE_FIELD: serving_engine}
                       if serving_engine == tc.SERVING_ENGINE_LORA_NATIVE else {})},
        "cell_id": cell.id,
        "model_id": model_id,
        "model_revision": model_revision,
        "gen": _gen_cfg(cfg),
        "controls": list(RESOLVED_CONTROLS_IDENTITY),
        "order_seed": 0,
        "limit": limit or None,
        "scale": float(cfg.get("scale", 1.0)),
        "system_sha": (cm.sha256_hex16(system.encode("utf-8"))
                       if system else None),
        "judge_model": judge_model,
        "subset_seed": ccr.SUBSET_SEED if limit else None,
    }
    # The served lens leg steers the position mask (assistant header at the
    # prompt tail + every generated position) — cell_controls_run's
    # --steer-positions 'response', the sweep's mask='generated' +
    # decode_mechanism='position_mask'. Recorded exactly like the HF driver
    # records a non-default policy (identity, not decoration).
    if bs._is_lens(route):
        resolved["steer_positions"] = "response"
    elif bs.is_steered(route):
        # a NATIVE steered route records the position mode the SLOT actually
        # serves it at — the route's own `petri_position_mode`
        # (native-psr-serving §1.8; the routing table's adapter block and the
        # engine attestation carry the same value, and the proxy's KV-namespace
        # decision is made from it). Stage 1 admitted `all` alone — a plain LoRA
        # adapter applies at every token, and the disposition SKIPPED every
        # residual cell whose mode was anything else — so a route from a table
        # that carries no mode is all-position BY CONSTRUCTION, not by guess:
        # this is the same "by era" reading `_run_serving_engine` makes of a
        # record with no `engine` key. Identity, on NEW rows only: every stage-1
        # native row still records "all", and a lens route still records
        # "response" above.
        _mode = getattr(route, "petri_position_mode", None)
        resolved["steer_positions"] = "all" if _mode is None else str(_mode)
    # the MMLU scoring surface is identity too: the seed, the top-K read
    # width, and whether the seeded render was DECLARED (lens routes).
    # `response_prefill` stays `_is_lens`: it is the lens mask's RE-LOCATION
    # DECLARATION, not a statement about steering.
    # The proxy REFUSES the declaration key on a non-lens route
    # (`response_prefill_not_lens`), so a lora route sends the same body without
    # it and records False — the seeded read itself is identical on both engines.
    resolved["mmlu"] = {"answer_seed": MMLU_ANSWER_SEED,
                        "top_logprobs": int(top_logprobs),
                        "response_prefill": bs._is_lens(route)}
    # mmlu-full-1531: the item-set WIDTH is identity too —
    # 200 items and the whole 1,531-row pinned pool are two instruments (a ±6.6 pp
    # and a ±2.4 pp read), never one at two resolutions, so a full-pool store
    # keys apart and can never resume onto or pool with a 200-item one. Recorded
    # ONLY on a full-pool run (contracts.IDENTITY_KEY_MMLU_ITEM_SET), so every
    # existing store's chash is byte-identical: the default width is the head
    # of the same permutation and stays the instrument every historical row is
    # tied to (the xstest_subset / parity_n precedent).
    if mmlu_full:
        resolved["mmlu"][tc.IDENTITY_KEY_MMLU_ITEM_SET] = tc.mmlu_item_set(full=True)
    # tinycap identity (contracts.TINYCAP_IDENTITY_FIELDS) ONLY when a
    # tinycap control is requested: the legacy chash stays byte-identical.
    if tinycap:
        resolved["tinycap"] = dict(tinycap)
        # E56 (3): the model GRADER's completion cap. Identity, by the same
        # argument as mmlu.top_logprobs (§PW-nanfix-2 item 5): the number
        # changes what the instrument produces, `harness_config_hash` covers
        # this dict WHOLE, so a store graded uncapped and one graded at a cap
        # land in different cell-key directories and can never resume onto or
        # be mistaken for each other. Recorded here rather than inside
        # `tinycap` so `contracts.tinycap_identity_fields()` — which is an
        # EXACT key list, asserted at both the driver and the launcher —
        # stays R19-A's to change; gated on `tinycap` for the same reason the
        # block itself is, so a non-tinycap run's chash is byte-identical to
        # what it was.
        resolved["inspect_grader"] = {"max_tokens": int(_grader_max_tokens(grader_max_tokens))}
    check = _template_check_class(getattr(prov, "status", None) if prov is not None else None)
    if check not in sweep_mod.TEMPLATE_CHECK_UNHASHED:
        resolved[sweep_mod.FIT_PROVENANCE_TEMPLATE_CHECK_KEY] = check
    # xstest_subset: recorded ONLY when set (cell_controls_run's precedent), so every historical
    # 450-row served store keeps its key and a 250-row SAFE store is a DIFFERENT store
    xstest_subset = ccr.xstest_subset_of(cfg)
    if xstest_subset:
        resolved[ccr.XSTEST_SUBSET_KEY] = xstest_subset
    return resolved


def _refuse_resume_across_provenance(store, manifest: dict) -> None:
    """Evidence is written ONCE by its producer and never upgraded by a later
    run. Reproduced: a store generated under
    partial provenance, resumed by a run under verified provenance, kept its
    identity (the "compared" class is not hashed, and the old hash covered
    partial and override too), skipped every generation and had its
    evidence rewritten to template_check="compared" — absence at the
    historical boundary became an asserted comparison.

    So a store may be RESUMED only when it positively states the SAME
    template-check class this run would write: the recorded manifest must
    carry the top-level class (a store that states none cannot be dated by
    this driver and is refused — its producer wrote it before the field, or
    the manifest is torn) and it must equal this run's. Any other case
    refuses, naming both, and the operator generates into a fresh out root:
    a re-run under different provenance is a NEW store, not an amendment.
    `validate_resume`'s producer guard (E14/E16 B) runs right after this and
    refuses a legacy store holding artifacts when this run names a producer;
    this check is the provenance-specific half of the same rule."""
    if not store.manifest_path.exists():
        return
    recorded = json.loads(store.manifest_path.read_text())
    key = sweep_mod.FIT_PROVENANCE_TEMPLATE_CHECK_KEY
    # P2 (fourth review): the RECORDED provenance is validated by the canonical
    # validator before any cached generation is trusted — comparing one
    # top-level field let a store whose hashed class contradicted its stated
    # one resume, make zero requests and be rewritten with the contradiction
    # erased. Two operations, kept apart: a store that already HOLDS work
    # (generations, evidence, markers) gets the full check
    # (sweep.store_provenance_problems: hashed-vs-stated class agreement, a
    # complete witness, a fit_provenance block that agrees); a positively
    # EMPTY pre-evidence store (a manifest and nothing else — the run died
    # before its evidence step) is RECOVERED, which needs the class to read
    # consistently and the witness to be complete, but not the block the run
    # never reached.
    # A recorded fit_provenance block is itself evidence that the run reached
    # its evidence step, so a manifest carrying one is never treated as a
    # pre-evidence store whatever the traversal sees (fifth review): the
    # block is VALIDATED, not ignored.
    if store._content_artifacts() or sweep_mod.MANIFEST_KEY_FIT_PROVENANCE in recorded:
        problems = sweep_mod.store_provenance_problems(recorded)
        if problems:
            raise SystemExit(
                f"[served-cap] {store.cell_dir}: the recorded store holds generations but its "
                f"provenance does not read consistently ({'; '.join(problems)}) — a record the "
                "canonical validator rejects is never trusted for its cached work and never "
                "rewritten into apparent validity. Repair it explicitly, or generate into a "
                "fresh out root (P2).")
    else:
        try:
            sweep_mod.template_check_of_manifest(recorded)
        except ValueError as e:
            raise SystemExit(
                f"[served-cap] {store.cell_dir}: the empty recorded store's template-check "
                f"class does not read ({e}); refusing to recover onto it (P2).")
        w = recorded.get(sweep_mod.MANIFEST_KEY_FIT_WITNESS)
        missing = [f for f in sweep_mod.FIT_WITNESS_FIELDS
                   if not isinstance(w, dict) or f not in w]
        if missing:
            raise SystemExit(
                f"[served-cap] {store.cell_dir}: the empty recorded store's fit witness is "
                f"incomplete (missing {missing}); refusing to recover onto it (P2).")
    # the ONE age mechanism (E54/E55): a store not positively written under the
    # template-check contract — historical (an older producer) or unknown (a
    # torn / undatable record) — is never resumed by a writer of it
    era, why = tc.store_era(recorded, contract=sweep_mod.CONTRACT_TEMPLATE_CHECK)
    if era != tc.STORE_ERA_CURRENT:
        raise SystemExit(
            f"[served-cap] {store.cell_dir}: the recorded store was not written under "
            f"the {sweep_mod.CONTRACT_TEMPLATE_CHECK} contract ({era}: {why}); its "
            "evidence was written once by its producer and is never re-stated by a "
            "later run — refusing to resume onto it. Generate into a fresh out root "
            "(a NEW store), never amend the old one.")
    # P1-2: the COMPLETE immutable fit witness, compared before anything is
    # touched — era and class alone let a same-class resume under an altered
    # sidecar (a changed trained template hash and model revision, unchanged
    # bytes) keep the store and replace the recorded fit facts in place
    diff = sweep_mod.fit_witness_diff(recorded.get(sweep_mod.MANIFEST_KEY_FIT_WITNESS),
                                      manifest[sweep_mod.MANIFEST_KEY_FIT_WITNESS])
    if diff:
        rec_w = recorded.get(sweep_mod.MANIFEST_KEY_FIT_WITNESS) or {}
        cur_w = manifest[sweep_mod.MANIFEST_KEY_FIT_WITNESS]
        raise SystemExit(
            f"[served-cap] {store.cell_dir}: the recorded fit witness differs from this "
            f"run's on {diff}: recorded "
            f"{ {f: rec_w.get(f) for f in diff if f != '<absent>'} } vs this run "
            f"{ {f: cur_w.get(f) for f in diff if f != '<absent>'} } — a store's fit "
            "provenance is written once by its producer; a changed or missing witness "
            "means a fresh store (generate into a fresh out root) or an explicit "
            "history-preserving rebind, never an amendment. Refusing to resume; the "
            "recorded manifest is untouched.")
    if key not in recorded:
        raise SystemExit(
            f"[served-cap] {store.cell_dir}: the recorded manifest states no {key} "
            "— written under the template-check contract yet stating no class: a "
            "torn manifest, whose evidence cannot be re-stated under this run's "
            f"class {manifest[key]!r}; refusing to resume onto it. "
            "Evidence is written once by its producer; generate into a fresh out "
            "root (a new store), never amend the old one.")
    if recorded[key] != manifest[key]:
        raise SystemExit(
            f"[served-cap] {store.cell_dir}: the recorded manifest states "
            f"{key}={recorded[key]!r} but this run would write {manifest[key]!r} — "
            "a provenance change is not a resume; refusing to rewrite the store's "
            "evidence. Generate into a fresh out root.")


def _registry_run_identity():
    """The planned-run identity ``$SISTEER_REGISTRY_RUN_IDENTITY`` names (``hooks.identity_from_env``) —
    the ONE switch for the two additive behaviours of :func:`_store_for` (qwen-seedrep-native-battery):
    the store's registry recorder and the production seed on its row template. Genuinely UNSET =
    today's store, byte for byte (no recorder, ``seed`` 0, no notice). PRESENT means this child was
    planned, so the door must open: a nonempty path, readable JSON, and a usable identity (one
    identity, or a ``{task: identity}`` map with at least one route, EVERY route usable by the
    recorder's own rule — ``hooks.store_recorder`` drops the whole map when any route lacks a
    ``run_id`` and has ``missing_identity_fields``) — else REFUSE. ``identity_from_env`` reads each
    load failure as None with a notice, and the recorder reads an incomplete route as "not recorded"
    with a notice; reading either as absence once stamped seed 0 on an UNREGISTERED store whose
    requests still drew at s (a silent fail-open after an expensive child)."""
    from sisteer.registry import hooks as _registry_hooks   # stdlib-only

    env = _registry_hooks.RUN_IDENTITY_ENV
    if env not in os.environ:
        return None
    path = os.environ[env]
    identity = _registry_hooks.identity_from_env()
    routes = _registry_hooks.normalise_identities(identity)
    if routes:
        # the recorder's rule, its function: a route with a run_id is registered already; one without must be complete
        incomplete = {r: _registry_hooks.missing_identity_fields(i) for r, i in routes.items()
                      if not i.get("run_id") and _registry_hooks.missing_identity_fields(i)}
        if incomplete:
            raise SystemExit(f"[served-cap] {env}={path!r} names a planned run the registry recorder would refuse as "
                             "incomplete — " + "; ".join(f"route {r!r} lacks {m}" for r, m in incomplete.items())
                             + " — a planned child never runs as an unregistered seed-0 store; fix the plan file")
        return identity
    if not path:
        why = "is empty"
    else:
        try:
            json.loads(Path(path).read_text(encoding="utf-8"))
            why = "names no planned run (neither an identity with eval_id/run_id nor a {task: identity} map)"
        except OSError as exc:
            why = f"cannot be read ({exc})"
        except ValueError as exc:
            why = f"is not JSON ({exc})"
    raise SystemExit(f"[served-cap] {env}={path!r} is exported but {why} — a planned child never runs as an "
                     "unregistered seed-0 store; fix the plan file, or unset the variable for an unplanned run")


#: the identity field that carries the plan's literal training seed s — the ``run`` row's own column,
#: what ``registry_plans.identity_for`` exports (``seed_base: s``). NOT ``seed``: that spelling
#: appeared only in hand-written test identities and hid a production seed of 0.
IDENTITY_SEED_FIELD = "seed_base"


def _production_seed(identity, gen: Optional[Mapping], *, inspect_seed_base: Optional[int] = None) -> Optional[int]:
    """The seed the store's rows (and the registry ``run.seed_base``) carry when a registry identity
    is exported: the identity's declared :data:`IDENTITY_SEED_FIELD` (the plan's literal training seed
    s; every route of a per-task map must agree), held to EVERY base this run actually draws under —
    the config's ``sampled_decode.seed_base`` when the judged controls draw under one, the Inspect
    replicate base (``--inspect-seed-base``; GSM8K) when one is in force: one value across all of
    them, else REFUSE naming each source (a plan saying 2 over a config saying 1 is never a silent
    1). Nothing declared and nothing in force -> None (the template keeps its 0). An identity that
    spells the seed as ``seed`` and not ``seed_base`` REFUSES: that is a fixture spelling no producer
    emits, and reading past it is how a GSM8K run once stamped seed 0."""
    from sisteer.registry import hooks as _registry_hooks   # stdlib-only

    declared = set()
    for route, ident in _registry_hooks.normalise_identities(identity).items():
        if ident.get("seed") is not None and ident.get(IDENTITY_SEED_FIELD) is None:
            raise SystemExit(f"[served-cap] the registry identity (route {route!r}) declares seed={ident['seed']!r} but "
                             f"no {IDENTITY_SEED_FIELD}; the run's seed field is {IDENTITY_SEED_FIELD} — refusing to "
                             "guess which field the plan meant")
        if ident.get(IDENTITY_SEED_FIELD) is not None:
            declared.add(int(ident[IDENTITY_SEED_FIELD]))
    if len(declared) > 1:
        raise SystemExit(f"[served-cap] the registry identity map declares several {IDENTITY_SEED_FIELD} values "
                         f"{sorted(declared)} for one store; one store carries one seed base — refusing")
    bases = {}          # source -> the seed base it states; all of them must agree
    if declared:
        bases[f"the registry identity's {IDENTITY_SEED_FIELD}"] = next(iter(declared))
    if isinstance(gen, Mapping) and gen.get("do_sample") and gen.get("seed_base") is not None:
        bases["the config's sampled_decode.seed_base"] = int(gen["seed_base"])
    if inspect_seed_base is not None:
        bases[tc.FLAG_INSPECT_SEED_BASE] = int(inspect_seed_base)
    if len(set(bases.values())) > 1:
        raise SystemExit("[served-cap] the seed base disagrees across "
                         + ", ".join(f"{src} {v}" for src, v in bases.items()) + " — refusing to stamp either")
    return next(iter(bases.values()), None)


def _leaves(obj):
    """Every scalar inside a nested mapping / sequence."""
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


#: the probe recorder's identity and construction — one run, no spool, disabled (its atexit hook is then a no-op),
#: no git fork — shared with the test that drives main's REAL registry package through the same construction
#: (the fallback at 72faab6c, and main before it), so what the tree is probed with is what main was tested with
_PROBE_IDENTITY = {"run_id": "probe", "campaign": "probe", "eval_id": "probe", "instrument_id": "probe"}
_PROBE_RECORDER_KWARGS = dict(spool=None, disabled=True, code_sha="probe", session="",
                              script="served_capability_run._recorder_row_constant_reads")


def _template_shape(template: Mapping) -> tuple:
    """The probe's memo key: the row template's ``((key, type name), …)``, sorted."""
    return tuple(sorted((k, type(v).__name__) for k, v in template.items()))


def _probe_sentinels(shape: tuple) -> dict:
    """One distinct sentinel per row constant, of the key's own type (so an ``int()`` read never trips);
    ``shape`` is the row template's ``((key, type name), …)``."""
    return {k: 7919 + i if t == "int" else 7919.5 + i if t == "float" else f"probe-{k}-7919"
            for i, (k, t) in enumerate(shape)}


def _covered_reads(sentinels: Mapping, payload: Mapping) -> frozenset:
    """The row constants whose sentinel surfaces anywhere in the probe's ``run`` payload — read by the
    recorder through ``row_constants``, wherever it lands them."""
    leaves = set(_leaves(payload))
    return frozenset(k for k, s in sentinels.items() if s in leaves)


@functools.lru_cache(maxsize=None)
def _recorder_row_constant_reads(shape: tuple) -> frozenset:
    """The row constants the tree's ``StoreRecorder.run_payload`` reads THROUGH the manifest's nested
    ``row_constants`` on its own (main's ``_seed_base`` fallback reads ``seed`` there), found by
    PROBING the recorder — a spec whose row constants sit only under ``row_constants``, one distinct
    sentinel per key, any sentinel that surfaces anywhere in the ``run`` payload marking its key as read —
    never by assuming which registry state the tree carries, and without knowing where each field lands.
    Once per process per template shape."""
    from sisteer.registry import hooks as _registry_hooks   # stdlib-only

    sentinels = _probe_sentinels(shape)
    probe = _registry_hooks.StoreRecorder("probe", {"row_constants": sentinels}, {"*": _PROBE_IDENTITY},
                                          **_PROBE_RECORDER_KWARGS)
    return _covered_reads(sentinels, probe.run_payload("*"))


def _manifest_row_constant_lift(template: Mapping) -> dict:
    """The row constants to lift to the manifest's top level under a registry identity: every one
    the recorder does not read through ``row_constants`` itself (:func:`_recorder_row_constant_reads`) —
    on a tree with no fallback the whole template, on a complete fallback only what the recorder never
    reads at all (harmless under ``setdefault``)."""
    covered = _recorder_row_constant_reads(_template_shape(template))
    return {k: v for k, v in template.items() if k not in covered}


def _store_for(cell, cfg: dict, out_root: Path, *, ihash: str, chash: str,
               model_id: str, model_revision, serving_engine: "str | None" = None,
               seed: "int | None" = None, registry_identity=None, resolved: "Mapping | None" = None,
               inspect_seed_base: "int | None" = None):
    """cell_controls_run._store_for's layout — one CellStore for the
    controls, store_key_token + cell_key — replicated because that template
    hardcodes engine='hf' (cell_controls_run.py:363); this one stamps
    engine='vllm' into every row and the resume identity. ``serving_engine``:
    the row schema (``store.ResultRow``) has no engine-NAME column and lives
    in another module, so the served engine does NOT ride the row constants —
    it rides every row's ``harness_config_hash`` (the hashed identity carries
    ``serving.serving_engine`` iff native, so a native store's rows never key
    like a lens store's) and the manifest (top-level ``serving_engine``, the
    tinycap ``route`` block the stored-record consumers read). Accepted here
    for the signature the caller composes; a lens run's rows and template are
    byte-identical.

    ``seed`` / ``registry_identity`` / ``resolved`` (ADDITIVE, qwen-seedrep-native-battery — the
    served-controls store was registry-blind): when the caller exports
    ``$SISTEER_REGISTRY_RUN_IDENTITY`` it passes the identity it names (:func:`_registry_run_identity`),
    the production seed (:func:`_production_seed`: the plan's literal s) and this run's resolved
    config, and the store gets (a) a registry recorder — on a tree whose ``CellStore`` does not resolve one
    itself, ``hooks.store_recorder`` attached here, the same recorder
    ``runtime.resolve_cell`` hands its stores; on a self-registering tree (main @ 525607e3+) nothing is attached
    and the store resolves main's default — so the wave's controls/GSM8K stores emit native ``run`` /
    ``item_result`` records (``run.seed_base`` = the template's ``seed``) — and (b) ``seed`` on its row
    template instead of the hardcoded 0 — what main still lacks. ``registry_identity`` defaults to the plan
    the child's environment exports (:func:`_registry_run_identity`) and ``seed`` to the production seed that
    identity implies (:func:`_production_seed` over ``_gen_cfg(cfg)`` and ``inspect_seed_base``), so the factory
    ITSELF carries the production seed for any caller under the child env — main's recorder files ``run.seed_base``
    from this template before it looks at the identity. Identity absent: no recorder,
    ``seed`` 0, the template byte for byte (the additive-only default every existing store was written under)."""
    from sisteer.capeval import store as cs
    from sisteer.eval.cell_controls import store_key_token

    if registry_identity is None:
        registry_identity = _registry_run_identity()          # the child's exported plan, when the caller threads none
    if seed is None and registry_identity is not None:
        seed = _production_seed(registry_identity, _gen_cfg(cfg), inspect_seed_base=inspect_seed_base)
    token = store_key_token(cell.id)
    key = cs.cell_key(token, "controls", 0, 0, ihash, chash)
    template = {
        "sweep": "two_si_controls_served",
        "run_id": f"served-controls-{token}",
        "model_id": model_id,
        "model_revision": model_revision,
        "arm_name": token,
        "kind": "cell",
        "coeff": float(cfg.get("scale", 1.0)),
        "intervention_hash": ihash,
        "engine": "vllm",
        "trial_id": 0,
        "seed": 0 if seed is None else int(seed),
        "attempt": 0,
        "harness_config_hash": chash,
        "env_digest": cs.env_digest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    recorder = None
    # main @ 525607e3 (#207, native emission ON by default): `CellStore(recorder=None)` RESOLVES the default
    # recorder itself (`hooks.default_store_recorder` off $SISTEER_REGISTRY_RUN_IDENTITY) and exports the
    # `NO_RECORDER` sentinel for callers that decline one. On such a tree this factory attaches nothing and lets
    # the store resolve (one recorder, main's). On a tree WITHOUT self-registration (the W cut this branch is
    # based on) the identity is attached here explicitly, so the wave's stores record either way.
    if registry_identity is not None and getattr(cs, "NO_RECORDER", None) is None:
        try:
            from sisteer.registry import hooks as _registry_hooks   # stdlib-only
            recorder = _registry_hooks.store_recorder(
                key, {**dict(resolved or {}), **template, "row_constants": dict(template)}, registry_identity,
                script="served_capability_run._store_for")
        except Exception as e:  # noqa: BLE001 — a lost record never costs a run (runtime.resolve_cell's rule)
            print(f"[registry] served-controls store {key}: recorder not built ({type(e).__name__}: {e}); "
                  "the run continues", flush=True)
            recorder = None
    return cs.CellStore(out_root, key, row_constants=template, recorder=recorder), template


def _judged_specs(cfg: dict, controls: list[str], template: dict, system,
                  limit: int, judge_model: str, pins,
                  max_goals: int) -> dict:
    """control -> JudgedSpec, rows from the SAME pinned loaders the HF path
    uses and the SAME seeded --limit subset (cell_controls_run's
    _seeded_subset) — so item ids match the HF stores and the downstream
    grader byte-for-byte."""
    from sisteer.capeval.tiers import judged

    loaders = {"xstest": _xstest_rows, "or_bench_hard_1k": _orbench_rows}
    xsub = ccr.xstest_subset_of(cfg)
    gen = _gen_cfg(cfg)
    specs: dict = {}
    for control in controls:
        if control not in loaders:
            continue
        rows = loaders[control](pins)
        if control == "xstest" and xsub:
            # the SAFE half only when the config says so — selected on the loader's rows BEFORE the seeded
            # subset, the head cap, the spec, the item hashes and every cap-ladder input
            rows = judged.xstest_subset_rows(rows, xsub)
        rows = ccr._seeded_subset(rows, limit)
        rows = _head_cap(rows, max_goals, control)
        specs[control] = judged.JudgedSpec(
            task=ccr.JUDGED_TASK_OF[control], template=dict(template),
            domain_of=dict(ccr.DOMAIN_OF), rows=rows, order_seed=0,
            system=system, judge_model=judge_model, gen=dict(gen))
    return specs


def _head_cap(rows: list, max_goals: int, control: str) -> list:
    """--max-goals: a smoke-only HEAD cap applied AFTER the seeded subset.
    Loud on use: the resulting item set no longer matches the HF store's,
    so a capped store must never enter a paired comparison."""
    if not max_goals or max_goals >= len(rows):
        return rows
    print(f"[served-cap] WARNING {control}: --max-goals {max_goals} head-cap "
          f"applied ({len(rows)} -> {max_goals} items) — SMOKE ONLY; this "
          "store's item set no longer matches the HF stores")
    return rows[:max_goals]


# --------------------------------------------------------------------------- #
# served identity + evidence (the positive control lives here)                 #
# --------------------------------------------------------------------------- #


def _mirror_provenance(cell, *, expected_object_sha256=None):
    """Fit-time provenance for a COMPILED cell (sweep.resolve_training_mirror,
    keyed on the object sha; loud refusal on an unrecorded object). None for
    baselines — nothing was fit. ``expected_object_sha256`` is the run's
    ``--expect-object-sha256`` pin when given: a fit record that carries no
    ``object_sha256`` of its own (the Arditi writer's) is BOUND to the bytes
    through it inside the reader, and a pin that disagrees with the bytes
    refuses there too — the mirror is the fourth door the launcher's pin
    passes through, not a way around it."""
    if getattr(cell, "object_path", None) is None:
        return None
    return sweep_mod.resolve_training_mirror(cell.id, cell.object_path,
                                             expected_object_sha256=expected_object_sha256)


def _model_identity(cfg: dict, table: dict, prov):
    """(model_id, model_revision) of the SERVED model. The routing table's
    base_model is authoritative for WHAT is being served; the revision comes
    from whichever record names that same model (config first, then the fit
    provenance). A mismatch is recorded loudly, never silently re-pinned."""
    base = table["base_model"]
    if cfg.get("model") == base:
        return base, cfg.get("model_revision")
    if prov is not None and getattr(prov, "model_id", None) == base:
        return base, getattr(prov, "revision", None)
    print(f"[served-cap] WARNING: config model {cfg.get('model')!r} != "
          f"routing-table base_model {base!r} and no provenance record names "
          "it — recording the SERVED model with revision=None. The "
          "served==trained template assert below still gates the serving.")
    return base, None


def _make_client(proxy_url: str, route) -> ProxyClientEngine:
    """The sync proxy client for the pre-flight + the bites probe. load()
    verifies the endpoint IS the petri proxy and serves this cell's virtual
    model — a dead tunnel or stale table fails HERE, before any request."""
    client = ProxyClientEngine(proxy_url, identity=route.serving_identity)
    client.load(require_virtual_models=[route.virtual_model])
    return client


def _inspect_api_key(standup_endpoint: "dict | None",
                     environ: "Mapping[str, str]" = os.environ) -> str:
    """THE bearer the Inspect and EvalPlus legs hand their clients
    (``inspect_served.run_inspect_tasks(api_key=)`` /
    ``evalplus_runner.run_served(api_key=)``): ``$TINYCAP_API_KEY``
    (:data:`tc.INSPECT_API_KEY_ENV`) when the environment names one — the
    cluster sbatch and ``tinycap_served`` spell it there as
    ``endpoint.proxy_token or "petri"`` and keep that path byte-identical —
    else the SAME rule the judged legs' :class:`ProxyClientEngine`
    (:func:`_make_client`) already follows against the SAME proxy,
    :func:`proxy_client_token`: the standup record's ``proxy_token`` when it
    names one, else ``PROXY_CLIENT_BEARER_DEFAULT``. One bearer source (the
    standup record this run already read), one rule, for every leg.

    The seedrep app composes no bearer variable of ANY name into a child's
    env — its lora-native standup records ``proxy_token: None`` (the routing
    proxy checks no token; the ENGINE's per-launch ``VLLM_API_KEY`` never
    leaves the standup process) — so ``os.environ.get(TINYCAP_API_KEY)`` was
    None, :func:`build_model` passed None through, and inspect_ai's
    ``OpenAICompatibleAPI.__init__`` (which wants SOME non-empty key whether
    or not the server checks one) refused every GSM8K child with
    ``PrerequisiteError: No TINYCAP_API_KEY defined in the environment``. The
    controls children on the same standup passed because their client never
    reads the env: it sends the default bearer. Never on argv, never logged;
    the value is what every client of this proxy already sends."""
    named = environ.get(tc.INSPECT_API_KEY_ENV) or ""
    if named:
        return named
    return proxy_client_token(standup_endpoint)


@functools.lru_cache(maxsize=2)
def _load_served_tok(model_id: str, revision):
    """The served model's tokenizer, loaded ONCE per (model, revision): the
    template evidence and the MMLU letter ids read the same read-only object
    (before, up to three ``from_pretrained`` loads
    of a ~17 MB tokenizer.json per run). The tests replace this name.

    PINNED TO THE SERVED RENDER REGIME (O27, 2026-09-06). A fresh
    ``from_pretrained`` carries no thinking pin, so on a reasoning template
    it renders thinking-ON while the server this driver talks to renders
    thinking-OFF (``vllm_server.THINKING_OFF_TEMPLATE_KWARGS``, the pin the
    fits were derived under). On Qwen3.5 the two disagree by two tokens —
    the generation prompt leaves ``<think>`` open (H=5) where the served,
    fit-side render carries the closed empty block (H=7) — so the header
    ids this driver writes into the manifest would
    describe a render the server never produces. chat_markers refuses an
    unpinned reasoning tokenizer rather than pick a width, which is how the
    smoke surfaced this; the FIX is the pin, the identical one
    lens_worker's install preflight took for the same failure (job 255399,
    2026-08-30) and that every other header call site already applies
    (capctrl_serving_up, lens_gate_standup, object_identity, the proxy).
    A no-op on templates without the switch (llama), and inert on gemma-4,
    whose trailing assistant header is H=3 under both arms.

    NOT the per-arm measurement: ``_header_ids_under_arm`` passes each
    arm's ``chat_template_kwargs`` into ``apply_chat_template`` explicitly
    and never reads this pin, so mixed-arm header comparison is unchanged.
    """
    from transformers import AutoTokenizer

    from sisteer.models import pin_reasoning_template_off

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    pin_reasoning_template_off(tok)
    return tok


def _tok_evidence(model_id: str, revision) -> dict:
    """Template + response-boundary facts measured from the SERVED model's
    live tokenizer: the served chat-template hash, the generation-prompt
    assistant header ids (the id vocabulary of the serve-time response
    boundary — the mask's support), and the render kwargs the probe (and
    the server) pinned, so evidence.json says WHICH render the header ids
    describe."""
    from sisteer.chat_markers import generation_header_ids
    from sisteer.models import chat_template_hash

    tok = _load_served_tok(model_id, revision)
    header_ids = [int(t) for t in generation_header_ids(tok)]
    return {"served_template_hash": chat_template_hash(tok),
            "serve_header_ids": header_ids,
            "serve_header_len": len(header_ids),
            "template_kwargs": dict(
                getattr(tok, "_sisteer_template_kwargs", None) or {})}


def _scrape_vllm_log(log_text: str) -> dict:
    """RESOLVED serving facts from the vLLM server log (copied from the BoN
    throughput sweep's scraper — phrase-matched, robust to the several vLLM
    log spellings of the prefix-cache line)."""
    out: dict = {}
    m = re.search(r"GPU KV cache size[:=]?\s*([\d,]+)\s*tokens", log_text)
    if m:
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    gen = re.findall(r"generation throughput:\s*([\d.]+)\s*tokens/s", log_text)
    if gen:
        out["last_gen_throughput_toks_s"] = float(gen[-1])
    hit = re.findall(
        r"(?i)(?:gpu )?prefix cache hit rate:\s*(?:gpu:\s*)?([\d.]+)\s*%",
        log_text)
    if hit:
        out["prefix_cache_hit_pct_last"] = float(hit[-1])
        out["prefix_cache_hit_pct_series_len"] = len(hit)
    run = re.findall(r"Running:\s*(\d+)\s*reqs", log_text)
    if run:
        out["running_series_max"] = max(int(x) for x in run)
    return out


def _template_check_class(status) -> str:
    """sweep.template_check_class — the ONE table (sweep.TEMPLATE_CHECK_BY_STATUS)
    the evidence block and the store identity both read."""
    return sweep_mod.template_check_class(status)


def _fit_provenance_block(prov, stamp: dict, *, model_id: str,
                          model_revision) -> dict:
    """evidence.json's ``fit_provenance``: the fit-time record the stamp's
    provenance fields were read from, spelled out beside the SERVED model
    so a reader of a capability result can see, per cell, (a) what the
    object was fit on vs what it is served on (``transfer``: a base-model
    document fit served on its instruct sibling is the sdf-base design, and
    a row must say so), and (b) whether the served==trained template check
    RAN (``template_check``, sweep.TEMPLATE_CHECK_BY_STATUS, with the prose
    in ``template_check_reason``). A baseline (nothing was fit) carries the
    block too, with class "baseline" — the field is TOTAL so the manifest's
    hashed class always has a mirror to be checked against."""
    status = stamp.get("provenance_status")
    check = _template_check_class(status)
    # the block is the IMMUTABLE witness (sweep.fit_witness_of — the same
    # dict the manifest's fit_witness holds, so the two agree by
    # construction) plus the REFRESHABLE served side
    witness = sweep_mod.fit_witness_of(prov)
    fit_model, fit_rev = witness["model_id"], witness["revision"]
    return {
        **witness,
        "served_model_id": model_id,
        "served_revision": model_revision,
        "served_template_hash": stamp.get("served_template_hash"),
        "transfer": (fit_model is not None
                     and (fit_model, fit_rev) != (model_id, model_revision)),
        "template_check": check,
        "template_check_reason": sweep_mod.TEMPLATE_CHECK_REASON[check],
    }


def _object_kind_of(cell) -> "str | None":
    """The served object's KIND for a compiled cell — the blob's own stored kind, read off the arm through the ONE map the
    serving adapter dispatches on (``registry_adapter.ARM_OBJECT_KINDS``; a Class B cell's arm is its id: ``alphasteer``
    -> ``alphasteer``, ``arditi`` -> ``static_r``). ``None`` for a baseline or an arm the map does not name — the caller
    then keeps the text gate (never a guess at a kind)."""
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    arm = getattr(cell, "arm", None)
    return ARM_OBJECT_KINDS.get(arm) if isinstance(arm, str) else None


def _logprob_liveness_evidence(chat, *, route, twin, kind: str, prompts, extra_body=None,
                               top_logprobs: int = LOGPROB_LIVENESS_TOP_LOGPROBS,
                               min_ratio: float = LOGPROB_BITES_MIN_RATIO,
                               abs_floor: float = LOGPROB_BITES_ABS_FLOOR,
                               min_common_tokens: int = LOGPROB_LIVENESS_MIN_COMMON_TOKENS,
                               min_conclusive_probes: int = LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES) -> tuple[dict, "str | None"]:
    """The LOGPROB liveness statistic on a GENERATION cell (``LOGPROB_LIVENESS_TOP_LOGPROBS`` says why it exists): for
    every bites prompt, ONE greedy token with ``logprobs`` + ``top_logprobs`` at the first generated position from the
    steered vmodel, the scale-0 twin, and the twin AGAIN (``contracts.BITES_LEGS``, sequential per prompt so the two
    scale-0 draws never share a batch); the three legs' top-K read on their COMMON token ids; the verdict
    ``fidelity.check_logprob_steering_bites``'s over the conclusive probes — the steered divergence must clear
    max(abs_floor, min_ratio x the measured repeat-noise ceiling), and bit-identical steered/twin logprobs refuse.
    ``extra_body`` is the bites probe's own (the render arm's kwargs; the declaration on a header-located ON cell), merged
    into every payload on every leg so the statistic is read under the SAME render as the text bite. Returns
    ``(block, error_or_None)``: the block is recorded either way (a failed control is durable AND loud), with the measured
    floor, the divergence, which bound of the bar bound, every probe's support, and the not-a-dose-axis sentence."""
    from sisteer.capeval.serving.fidelity import FidelityError, check_logprob_steering_bites

    extra = dict(extra_body or {})
    legs = tuple(zip(tc.BITES_LEGS, (route.virtual_model, twin.virtual_model, twin.virtual_model)))

    def one_token(vmodel: str, prompt: str, leg: str, index: int) -> dict:
        resp = chat({"model": vmodel, "messages": [{"role": "user", "content": prompt}],
                     "temperature": 0.0, "max_tokens": tc.LOGPROB_BITES_MAX_TOKENS, "seed": 0,
                     "logprobs": True, "top_logprobs": int(top_logprobs),
                     "return_tokens_as_token_ids": True, **extra})
        return pe.first_position_logprobs_by_id(resp, what=f"logprob_liveness probe {index} leg {leg!r}")

    per_probe: list = []
    items: list = []
    error = None
    try:
        for i, prompt in enumerate(prompts):
            by_leg = {leg: one_token(vm, prompt, leg, i) for leg, vm in legs}
            supports = [set(d) for d in by_leg.values()]
            common = sorted(set.intersection(*supports))
            rec = {"item_id": f"probe{i:02d}", "prompt_index": i, "n_common": len(common),
                   "support_sizes": {leg: len(by_leg[leg]) for leg, _ in legs},
                   "supports_disjoint": not (set(by_leg[tc.BITES_LEGS[0]]) & set(by_leg[tc.BITES_LEGS[1]])),
                   "conclusive": len(common) >= int(min_common_tokens)}
            per_probe.append(rec)
            if rec["conclusive"]:
                items.append({"item_id": rec["item_id"], "common_token_ids": common,
                              **{leg: [by_leg[leg][t] for t in common] for leg, _ in legs}})
        if len(items) < int(min_conclusive_probes):
            error = (f"logprob_liveness: only {len(items)} of {len(per_probe)} probe(s) are conclusive (>= "
                     f"{min_common_tokens} common top-{top_logprobs} token ids across the three legs); the control needs "
                     f">= {min_conclusive_probes} — UNEVALUABLE, refused (never a text-only pass); per probe: "
                     f"{[(r['item_id'], r['n_common'], r['supports_disjoint']) for r in per_probe]}")
    except Exception as e:  # noqa: BLE001 — an unreadable instrument is a FAILED control, recorded, never a pass
        error = f"logprob_liveness: the statistic could not be read — {type(e).__name__}: {e}"
    diag: dict = {}
    if error is None:
        try:
            check_logprob_steering_bites({"items": items}, diag, min_ratio=min_ratio, abs_floor=abs_floor)
        except FidelityError as e:
            error = str(e)
    noise = diag.get("noise_ceiling")
    bar = diag.get("bar")
    median = diag.get("steered_divergence_median")
    block = {
        "applicable": True, "verified": error is None, "kind": kind, "twin": twin.cell_id,
        "statistic": ("sisteer.capeval.serving.fidelity.check_logprob_steering_bites over the COMMON top-K token ids of the "
                      "three legs at the first generated position, on the bites probe prompts (the AlphaSteer ladder's own "
                      "liveness instrument)"),
        "legs": list(tc.BITES_LEGS), "leg_order": "sequential per prompt: steered, scale0, scale0_repeat",
        "n_probes": len(prompts), "n_conclusive": len(items),
        "min_conclusive_probes": int(min_conclusive_probes), "min_common_tokens": int(min_common_tokens),
        "top_logprobs": int(top_logprobs), "max_tokens": tc.LOGPROB_BITES_MAX_TOKENS, "probe_extra_body": extra,
        "per_probe": per_probe, "probe_items": items, "diag": diag,
        "noise_floor": {
            "measured": noise,
            "how": ("the scale-0 twin against ITSELF: the same request to the same twin route a second time, in this run, on "
                    "these probe prompts — max over probes of the max abs logprob difference (fidelity's noise_ceiling)"),
            "min_ratio": float(min_ratio), "abs_floor": float(abs_floor), "bar": bar,
            "binding": (None if bar is None else
                        ("measured" if float(min_ratio) * float(noise) >= float(abs_floor) else "abs_floor")),
            "note": ("the bar is max(abs_floor, min_ratio x the measured repeat-noise ceiling): abs_floor is the "
                     "instrument's degenerate-case guard for a bit-stable engine (noise 0), not a threshold of its own; "
                     "`binding` says which bound applied on this run"),
        },
        "divergence": {"steered_vs_scale0_median": median, "clears_bar": (None if bar is None or median is None else median >= bar)},
        "not_a_dose_axis": tc.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS,
    }
    if error is not None:
        block["error"] = error
    return block, error


def _served_evidence(*, cell, route, routes, chat, model_id, model_revision,
                     prov, vllm_log, tok_ev=None,
                     probe_render_arm: "str | None" = None,
                     out_of_support: bool = False,
                     probe_stamp_policies: "Mapping | None" = None,
                     probe_header_located: bool = True,
                     lens_liveness: str = tc.LENS_LIVENESS_TEXT,
                     logprob_chat=None,
                     probe_render_arm_reason: "str | None" = None,
                     standup_thinking_mode: object = None) -> tuple[dict, "str | None"]:
    """Assemble evidence.json's contents and run the positive controls.
    ``tok_ev`` is the served tokenizer's template evidence when the caller
    already measured it (the tinycap provenance needs it BEFORE the
    manifest); measured here otherwise. ``probe_render_arm`` (fit-support
    transfer) is the thinking arm the bites probe renders under
    on a tinycap run — its payloads then carry the arm's
    ``chat_template_kwargs`` (+ the declaration on a thinking-ON lens standup)
    on BOTH legs and every leg's stamp triple is held to its contract row
    (``steering_bites.stamps_by_leg`` / ``probe_render_arm`` are recorded);
    ``None`` (a legacy run) sends the probe byte-identical to before.

    Returns ``(evidence, bites_error)``. ``bites_error`` is non-None when
    the steering-bites probe FAILED (an inert steered route) — the caller
    records the evidence and then aborts, so the failure is both durable
    and loud. Salt-isolation violations raise immediately: they are
    config-level and nothing should run on top of them.

    ``lens_liveness`` (contracts.LENS_LIVENESS_MODES; the ``--lens-liveness``
    flag) selects the statistic that RULES on a lens cell: ``text`` (the
    default; byte-identical to before) or ``logprob`` — which applies only to a
    cell whose object kind is in ``contracts.LOGPROB_LIVENESS_KINDS``
    (``alphasteer`` / ``static_r``): there ``_logprob_liveness_evidence`` rules,
    through ``logprob_chat`` (the ``probe/logprob_bites``-labelled client; the
    bites ``chat`` when None), and the text statistic is reported beside it
    under ``steering_bites.text_divergence``. Every other kind keeps the text
    gate under either mode. ``steering_bites.liveness_gate`` names the ruler.

    Deliberately NOT used as controls: intervention_hash-of-spec and
    effective_scale — both stayed green through the real mask bug. The
    steered-vs-unsteered generation diff and the mask/decode_mechanism
    stamps ARE the control."""
    lens = bs._is_lens(route)
    # the bites proof is owed by every STEERED route — a lens hook OR a native
    # LoRA adapter (`bon_served.is_steered`) — against its PATH-MATCHED scale-0
    # twin; only the salt regime and the position mask (`decode_mechanism`,
    # `mask`) stay lens facts (`lens`)
    steered = bs.is_steered(route)

    # 1. per-cell KV isolation over the FULL route set (raises on violation).
    bs.assert_cell_salt_isolation(routes)
    salt = bs.cell_salt_isolation_report(routes)
    this_salt = salt["per_cell"].get(route.virtual_model)

    # 1b. the served engine stamp + the served==trained template invariant —
    #     BEFORE the first steered request. A wrongly-provenanced object must be
    #     refused before it is served and probed, not after: the probe below
    #     used to run first, so an object the assertion would refuse was steered
    #     once on the way to the refusal. Every input is bound already: the
    #     provenance record, the served tokenizer's template evidence, the route.
    if tok_ev is None:
        tok_ev = _tok_evidence(model_id, model_revision)
    stamp = sweep_mod.engine_stamp(
        target_engine="vllm",
        serving_path=cm.parse_serving_identity(route.serving_identity).path,
        served_template_hash=tok_ev["served_template_hash"],
        trained_template_hash=getattr(prov, "trained_template_hash", None),
        object_sha=getattr(prov, "object_sha", None),
        decode_mechanism="position_mask" if lens else None,
        provenance_status=getattr(prov, "status", None),
        provenance_source=getattr(prov, "source", None))
    sweep_mod.assert_served_matches_trained(
        stamp, what=f"{cell.id} served capability generate")
    fit_prov = _fit_provenance_block(prov, stamp, model_id=model_id,
                                     model_revision=model_revision)

    # 2. steering-bites through the SAME serving path, vs the scale-0 twin.
    bites: dict
    bites_error = None
    if not steered:
        bites = {"applicable": False,
                 "reason": f"route {route.cell_id!r} carries no steering "
                           "intervention (an unsteered route: serving path "
                           f"{cm.parse_serving_identity(route.serving_identity).path!r}, "
                           "neither a lens hook nor a native adapter) — no "
                           "steered arm to probe"}
    else:
        # the PATH-MATCHED twin (proxy_engine.scale0_twin): a lens route's `__lens0`
        # leg, a lora route's `__lora0` zero adapter — the ENGINE's own zero cell,
        # never the other engine's control (which would prove nothing about the
        # machinery under test)
        twin = pe.scale0_twin(route, routes)
        twin_suffix = pe.scale0_twin_suffix(route)
        if twin is None:
            raise bs.BonServedError(
                f"steered cell {route.cell_id!r} has no scale-0 twin route "
                f"(neither its own {twin_suffix} nor "
                f"{tc.ROUTE_NONE_CELL}{twin_suffix}) in the routing "
                "table — a steered serve without a bites probe is refused, "
                "never run on faith")
        arm = cell.id.split("__")[-1]
        # the twin route the table exports must be the registry's derived
        # control leg under its ONE rule (VMODEL_PREFIX + the ENGINE's zero cell:
        # none__lens0 on a lens route, none__lora0 on a lora route —
        # contracts.twin_cell_id_for / twin_vmodel_for) — the same rule the
        # launcher's dry run names the probe's leg by
        if twin.cell_id == tc.twin_cell_id_for(route) and twin.virtual_model != tc.twin_vmodel_for(route):
            raise SystemExit(
                f"[served-cap] the routing table's twin {twin.cell_id!r} is served as "
                f"{twin.virtual_model!r} but the registry's rule names it {tc.twin_vmodel_for(route)!r} "
                "— the probe would compare against a leg the launcher does not expect; refusing")
        probe = usr._served_probe_fn(chat, route.virtual_model,
                                     twin.virtual_model,
                                     extra_body=_probe_extra_body(probe_render_arm, header_located=probe_header_located))
        try:
            diag = bs.assert_steering_bites((arm,), probe)
            arm_diag = diag["arms"].get(arm, {})
            bites = {"applicable": True, "verified": True,
                     "twin": twin.cell_id, "arm": arm,
                     "n_probes": arm_diag.get("n"),
                     "n_differing": arm_diag.get("n_differing"),
                     "diag": diag}
        except FidelityError as e:
            bites_error = str(e)
            bites = {"applicable": True, "verified": False,
                     "twin": twin.cell_id, "arm": arm, "error": bites_error}
        # THE LIVENESS GATE (contracts.OOS_EVIDENCE_BITES_LIVENESS_GATE): text by default; the LOGPROB statistic for the
        # two Class B deployment-shape kinds when the run asks for it — then the text result above is an OBSERVATION
        # recorded beside the ruling statistic, never the ruler and never `error` at the block's top level
        kind = _object_kind_of(cell)
        gate = (tc.LENS_LIVENESS_LOGPROB if (lens_liveness == tc.LENS_LIVENESS_LOGPROB and kind in tc.LOGPROB_LIVENESS_KINDS)
                else tc.LENS_LIVENESS_TEXT)
        bites[tc.OOS_EVIDENCE_BITES_LIVENESS_GATE] = gate
        bites["object_kind"] = kind
        if gate == tc.LENS_LIVENESS_LOGPROB:
            n_text = (bites.get("n_probes") if bites.get("n_probes") is not None else len(usr._SERVED_PROBE_PROMPTS))
            bites[tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE] = {
                "verified": bites_error is None, "n_probes": n_text,
                "n_differing": bites.get("n_differing", 0 if bites_error is not None else None),
                "error": bites_error,
                "role": ("REPORTED beside the logprob liveness statistic, which rules for this kind — the greedy-text "
                         "divergence of five 48-token bites is a step-function readout that a small correct edit need "
                         "not move; a cell live by logprob and inert by text is a fact about DOSE kept on the record")}
            # the text probe fired on every prompt of both legs before its verdict, so the counts the analysis's
            # inventory reads are known even when the text statistic refused (the except-branch shape carries none)
            bites.setdefault("n_probes", n_text)
            if bites.get("n_differing") is None:
                bites["n_differing"] = 0
            bites.pop("error", None)
            lp_block, lp_err = _logprob_liveness_evidence(
                logprob_chat or chat, route=route, twin=twin, kind=kind,
                prompts=usr._SERVED_PROBE_PROMPTS, extra_body=probe.extra_body)
            bites[tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS] = lp_block
            bites["verified"] = lp_err is None
            bites_error = lp_err
            if lp_err is not None:
                bites["error"] = lp_err
        # the render the probe was sent under, the body it added, and
        # every leg's stamp triples (one per prompt) — then the leg check: ONE
        # triple per leg, equal to the §2.5.11 row for a lens leg under that
        # render; a disagreement fails the control like an inert steer (recorded
        # first, aborted by the caller)
        bites[tc.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = probe_render_arm
        # F-AI, additive: WHICH rule chose that render (`_bites_probe_render_arm`) and the standup's attested
        # regime it was keyed on — so a reader sees an ON probe on a native ON-pinned standup for what it is
        bites["probe_render_arm_reason"] = probe_render_arm_reason
        bites["standup_thinking_mode"] = standup_thinking_mode
        bites["probe_channel_policy"] = getattr(probe, "channel_policy", None)
        bites["probe_extra_body"] = dict(probe.extra_body)
        bites["stamps_by_leg"] = {leg: list(stamps) for leg, stamps in probe.stamps_by_leg.items()}
        bites["stamp_errors_by_leg"] = {leg: list(errs) for leg, errs in probe.stamp_errors_by_leg.items()}
        # what each probe completion carried per channel (counts + finish reason): a
        # thinking-ON probe answers in the reasoning channel with content None, and the
        # bite is compared on both re-joined (usr._served_probe_fn) — recorded so a
        # channel-less draw reads off the evidence
        bites["channels_by_leg"] = {leg: list(rows) for leg, rows in getattr(probe, "channels_by_leg", {}).items()}
        if probe_render_arm is not None:
            stamp_problem = _probe_stamp_problem(probe.stamps_by_leg, probe_render_arm,
                                                 out_of_support=bool(out_of_support),
                                                 policies=dict(probe_stamp_policies or {}),
                                                 errors_by_leg=probe.stamp_errors_by_leg,
                                                 # each leg's expected row follows ITS engine:
                                                 # the steered route's and the twin's
                                                 lens_by_leg={usr.PROBE_LEG_STEERED: lens,
                                                              usr.PROBE_LEG_SCALE0: bs._is_lens(twin)},
                                                 # the route's NATIVE position mode
                                                 mode_by_leg={usr.PROBE_LEG_STEERED: getattr(route, "petri_position_mode", None),
                                                              usr.PROBE_LEG_SCALE0: getattr(twin, "petri_position_mode", None)})
            if stamp_problem is not None:
                bites["verified"] = False
                bites["stamp_error"] = stamp_problem
                bites_error = (f"{bites_error + '; ' if bites_error else ''}{stamp_problem}")

    # 3. (the served engine stamp + the served==trained template invariant ran
    #    BEFORE step 2 — see there; ``stamp`` / ``fit_prov`` are already bound)

    # 4. prefix-cache facts from the server log (expected ~0% on a steered
    #    lens leg BY DESIGN: per-request cache_salt defeats the cache).
    if vllm_log:
        p = Path(vllm_log)
        cache = (_scrape_vllm_log(p.read_text(errors="replace"))
                 if p.is_file() else {"unavailable": f"no file at {p}"})
    else:
        cache = {"unavailable": "--vllm-log not given"}

    evidence = {
        "driver": DRIVER_VERSION,
        "cell_id": cell.id,
        "harness_cell_id": route.cell_id,
        "virtual_model": route.virtual_model,
        "serving_identity": route.serving_identity,
        "intervention_hash": route.intervention_hash,
        "engine": "vllm",
        # sweep vocabulary: mask='generated' + decode_mechanism=
        # 'position_mask' == cell_controls' steer_positions='response'.
        "mask": "generated" if lens else None,
        "engine_stamp": stamp,
        # the fit side of the stamp, spelled out: what the object was fit ON
        # vs what it is served on, and whether the template check ran
        "fit_provenance": fit_prov,
        "serve_header": tok_ev,
        "salt_isolation": {"ok": True, "n_routes": len(routes),
                           "this_cell": this_salt},
        "steering_bites": bites,
        "prefix_cache": cache,
        "written_at": _utc_now(),
    }
    return evidence, bites_error


def _logprob_bites_evidence(chat, *, route, routes, rows, tok, system,
                            top_logprobs: int,
                            n_probes: int = LOGPROB_BITES_PROBES,
                            min_ratio: float = LOGPROB_BITES_MIN_RATIO,
                            abs_floor: float = LOGPROB_BITES_ABS_FLOOR):
    """The LOGLIKELIHOOD positive control (module docstring): letter
    logprobs for the first ``n_probes`` MMLU rows through the SAME seeded
    scoring path, from the steered vmodel, its scale-0 twin, and the twin
    AGAIN (the numeric-noise leg), ruled on by
    ``fidelity.check_logprob_steering_bites``. Returns
    ``(evidence_block, error_or_None)`` — the caller records the block
    either way, so a failure is durable AND loud."""
    from sisteer.capeval.serving.fidelity import (
        FidelityError, check_logprob_steering_bites)
    from sisteer.eval.capability_ext import _letter_token_ids, _letters

    # the PATH-MATCHED twin (proxy_engine.scale0_twin): `__lens0` for a lens
    # route, the `__lora0` zero adapter for a lora route
    twin = pe.scale0_twin(route, routes)
    if twin is None:
        twin_suffix = pe.scale0_twin_suffix(route)
        raise bs.BonServedError(
            f"steered cell {route.cell_id!r} has no scale-0 twin route "
            f"(neither its own {twin_suffix} nor "
            f"{tc.ROUTE_NONE_CELL}{twin_suffix}) in the routing "
            "table — a served loglikelihood without its positive control "
            "is refused, never run on faith")
    probes = list(rows[:max(1, int(n_probes))])
    # the lens mask's re-location DECLARATION (`petri_response_prefill`) rides a
    # LENS route's seeded body only — the proxy refuses it on a non-lens route
    # (`response_prefill_not_lens`); a lora route sends the same seeded body
    # without the key, on all three legs
    declare_prefill = bs._is_lens(route)
    # the three legs under their own vmodels (contracts.BITES_LEGS): legs SEQUENTIAL, so the two scale-0 legs never share a
    # batch (they measure the engine's own noise floor); the probes of one
    # leg CONCURRENT — independent items through the sync client, where the
    # 24 blocking round trips used to run strictly one at a time
    legs = tuple(zip(tc.BITES_LEGS,
                     (route.virtual_model, twin.virtual_model, twin.virtual_model)))
    letter_ids_of = {
        row["item_id"]: [int(t) for t in _letter_token_ids(
            tok, _letters(min(len(row["choices"]), 4)))]
        for row in probes}

    def probe_one(leg: str, vmodel: str, row) -> list:
        resp = chat(_mmlu_scoring_payload(
            vmodel, _mmlu_user_text(row), system=system,
            top_logprobs=top_logprobs, response_prefill=declare_prefill))
        lps, missing = _letter_logprobs_from_choice(
            resp, letter_ids_of[row["item_id"]],
            what=f"logprob-bites {row['item_id']} {leg}")
        if missing:
            raise FidelityError(
                f"logprob_steering_bites: probe {row['item_id']!r} "
                f"leg {leg!r}: letter token id(s) {missing} absent "
                f"from the top-{top_logprobs} logprobs — the "
                "control cannot read the letters it must compare; "
                "raise --top-logprobs (and the server's "
                "--max-logprobs)")
        return lps

    items = []
    error = None
    try:
        per_leg = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(probes)) as ex:
            for leg, vmodel in legs:
                per_leg[leg] = list(ex.map(functools.partial(probe_one, leg, vmodel), probes))
        items = [{"item_id": row["item_id"], "letter_ids": letter_ids_of[row["item_id"]],
                  **{leg: per_leg[leg][k] for leg, _ in legs}}
                 for k, row in enumerate(probes)]
    except FidelityError as e:
        # a probe-collection failure (top-K coverage, malformed response)
        # is a FAILED control, recorded durably in evidence.json — never
        # an uncaught abort that dies before the evidence is written
        # (bitten on gemma4e4b x2, 2026-09-01: the letter fell outside
        # top-64 and the run died with NO evidence on disk).
        error = str(e)
    diag: dict = {}
    if error is None:
        try:
            check_logprob_steering_bites({"items": items}, diag,
                                         min_ratio=min_ratio,
                                         abs_floor=abs_floor)
        except FidelityError as e:
            error = str(e)
    block = {"applicable": True, "verified": error is None,
             "twin": twin.cell_id, "n_probes": len(items),
             "diag": diag, "probe_items": items}
    if error is not None:
        block["error"] = error
    return block, error


def _write_evidence(out_root: Path, store, evidence: dict) -> None:
    """evidence.json at the out-root (the task's named location) AND inside
    the cell dir (so it travels with the store)."""
    blob = json.dumps(evidence, indent=2, default=str)
    (out_root / "evidence.json").write_text(blob)
    store.cell_dir.mkdir(parents=True, exist_ok=True)
    (store.cell_dir / "evidence.json").write_text(blob)


# --------------------------------------------------------------------------- #
# the two generation legs                                                      #
# --------------------------------------------------------------------------- #


async def _run_mmlu_served(post, *, route, rows, tok, store, template,
                           system, concurrency, top_logprobs) -> dict:
    """s2parity/mmlu_letter through the proxy (module docstring): one
    seeded scoring request per item, the four letter logprobs read from
    the first generated position's raw top-K, pick = argmin NLL (== the
    S2 panel's argmax-letter-logit rule; one forward, one softmax
    denominator — ``parity._response_letter_nlls``'s exact readout,
    served). Emits the SAME per-item rows the HF store carries (metric
    "acc", extra pick/gold) so the stores pair item-for-item, plus the
    served NLLs as diagnostics.

    A letter id absent from the returned top-K is CENSORED, never
    guessed and never filled in: every token outside the top-K has a
    logprob no higher than the lowest token inside it, so an absent
    letter cannot out-rank any PRESENT letter and the argmax over the
    four letters is decided by the present ones alone. The pick is
    therefore EXACT with any letter present (only an exact tie at the
    top-K boundary could differ, a measure-zero float event) — and so is
    the accuracy: a censored gold letter scores 0, which is the correct
    verdict (the model ranked it below at least one distractor), and a
    censored distractor changes nothing. The censorship is recorded on
    the item row (``censored_letters`` / ``n_censored``; the censored
    letter's ``nlls`` entry is ``None``, not a number) and counted on
    the aggregate (``censored_items``, ``censored_letter_counts``): a
    cell whose letters fall off the top-K is a finding — the model has
    drifted off the letter vocabulary — and must read as one, never as
    a quiet accuracy. Refused as a hard per-item error (collected,
    aborting the aggregate, exactly as before) only when NO letter is
    in the top-K: nothing can be ranked and the pick is undefined; that
    is collapse, and raising the read width is not the answer (decided
    2026-09-07). The scorer was refusing EVERY censored item
    before 2026-09-08 — provably too broad; it stopped four gemma cells
    on 1-6 items of 200 (448328).

    Even the refused item is RECORDED: a ``MMLU_UNDETERMINABLE_METRIC`` row
    files the letters, their ids, the window and the smallest returned
    logprob — the K-th largest, an upper bound on every absent option — plus
    the sampled token, so the item is a recoverable bracket, not a gap whose
    evidence died with the exception (a guard that refuses on a missing
    input must keep the inputs it had). The row does not block resume:
    ``_done_acc`` reads metric "acc" only, so a later run with the letters
    back inside the window scores the item. Every row names its case in
    ``extra["letter_read"]`` (all_present / partial_present on an "acc"
    row; all_absent on the bracket row), and a partial row carries the
    same window record under ``incomplete_topk``. An item with every
    letter present is scored by the same expressions as before and its
    value / pick / nlls / censorship markers are unchanged (pinned)."""
    from sisteer.capeval.tiers import parity, tier0
    from sisteer.eval.capability_ext import _letter_token_ids, _letters

    t0 = time.time()
    em = tier0._Emitter(store, dict(template), dict(ccr.DOMAIN_OF),
                        parity.TASK_VERSION)
    done = parity._done_acc(store, parity.PARITY_TASK_MMLU)
    ordered = parity._permuted(rows, 0)  # order_seed 0, the driver constant
    todo = [(k, r) for k, r in enumerate(ordered)
            if r["item_id"] not in done]
    # `declare_prefill`: whether the seeded render is DECLARED to the proxy
    # (`petri_response_prefill`) — the lens mask's re-location declaration, a
    # lens-route fact the proxy refuses on a non-lens route
    # (`response_prefill_not_lens`). The seeded teacher-forced letter read
    # itself is IDENTICAL on lens and lora routes: a lora route sends the same
    # body without the key and records False; the evidence block `mmlu_scoring`
    # says where the steering acted.
    declare_prefill = bs._is_lens(route)
    letters_of = {}
    letter_ids_of = {}
    work = []
    for _, r in todo:
        letters = _letters(min(len(r["choices"]), 4))
        letters_of[r["item_id"]] = letters
        letter_ids_of[r["item_id"]] = [int(t) for t in
                                       _letter_token_ids(tok, letters)]
        work.append((r["item_id"], _mmlu_scoring_payload(
            route.virtual_model, _mmlu_user_text(r), system=system,
            top_logprobs=top_logprobs, response_prefill=declare_prefill)))
    responses, errors = await _fanout_responses(post, work,
                                                concurrency=concurrency)

    acc_by_item = {iid: m["acc"] for iid, m in done.items()}
    undeterminable = 0
    for order_idx, row in todo:
        resp = responses.get(row["item_id"])
        if resp is None:
            continue  # its error is in `errors`; the item reruns on resume
        letters = letters_of[row["item_id"]]
        letter_ids = letter_ids_of[row["item_id"]]
        what = f"mmlu {row['item_id']}"
        try:
            lps, missing = _letter_logprobs_from_choice(
                resp, letter_ids, what=what)
        except RuntimeError as e:
            errors.append((row["item_id"], str(e)))
            continue
        # the window record (the bound every absent letter sits under) is
        # read from the same entries the letter read just parsed
        window = (_mmlu_window_record(
            resp, letters=letters, letter_ids=letter_ids, missing=missing,
            top_logprobs=top_logprobs, what=what) if missing else None)
        if len(missing) == len(letter_ids):
            # every letter absent: nothing can be ranked — but the guard
            # does not refuse empty-handed. The response DID carry the K
            # largest logprobs, whose minimum bounds every option from
            # above; filing them keeps the item a recoverable bracket
            # instead of a gap whose evidence died with the exception.
            undeterminable += 1
            em.emit(task=parity.PARITY_TASK_MMLU, item_id=row["item_id"],
                    metric=MMLU_UNDETERMINABLE_METRIC, value=1.0, tier=1,
                    position_order=order_idx,
                    extra={MMLU_LETTER_READ_KEY: MMLU_LETTER_READ_ALL_ABSENT,
                           "gold": letters[row["answer"]],
                           "letters": list(letters),
                           "letter_ids": [int(t) for t in letter_ids],
                           "present_letter_logprobs": {},
                           **window})
            errors.append((row["item_id"],
                           f"all {len(letter_ids)} letter token ids "
                           f"{missing} absent from the top-{top_logprobs} "
                           "logprobs — no letter can be ranked, the pick "
                           "is undefined; this is letter-vocabulary "
                           "collapse, not a read-width knob; the item is "
                           "left unscored and the returned-logprob bound "
                           "is filed under metric "
                           f"{MMLU_UNDETERMINABLE_METRIC!r}"))
            continue
        if len(lps) + len(missing) != len(letter_ids):
            errors.append((row["item_id"],
                           f"letter read returned {len(lps)} logprobs + "
                           f"{len(missing)} missing for {len(letter_ids)} "
                           "letters — cannot align them; refusing"))
            continue
        # the read returns the PRESENT letters' logprobs in letter order;
        # re-align to the four letters, None where the letter is censored
        present = iter(lps)
        nlls = [None if lid in missing else -next(present)
                for lid in letter_ids]
        ranked = [i for i, v in enumerate(nlls) if v is not None]
        pick = min(ranked, key=nlls.__getitem__)  # argmin NLL over the
        # present letters == argmax letter logprob of ONE forward — the S2
        # panel's rule; with all four present this is the same min over
        # range(4) it always was, so an uncensored item's row is unchanged
        censored = [letters[i] for i, v in enumerate(nlls) if v is None]
        acc = float(pick == row["answer"])
        extra = {"pick": letters[pick], "gold": letters[row["answer"]],
                 "nlls": [None if v is None else round(v, 6)
                          for v in nlls],
                 "top_logprobs": int(top_logprobs),
                 "censored_letters": censored,
                 "n_censored": len(censored),
                 MMLU_LETTER_READ_KEY: (MMLU_LETTER_READ_PARTIAL if censored
                                        else MMLU_LETTER_READ_ALL_PRESENT)}
        if censored:
            # the bound travels with the row: how far below the window
            # the censored letters sit is readable without the response
            extra["incomplete_topk"] = window
        em.emit(task=parity.PARITY_TASK_MMLU, item_id=row["item_id"],
                metric="acc", value=acc, tier=1, position_order=order_idx,
                extra=extra)
        acc_by_item[row["item_id"]] = acc
    if errors:
        raise SystemExit(
            f"[served-cap] mmlu: {len(errors)} of {len(todo)} items FAILED "
            f"(first: {errors[:3]}) — {len(acc_by_item)} items are scored "
            "in rows.jsonl and resume; re-run generate to fill the gaps "
            "rather than summarizing a partial run. Of the failures, "
            f"{undeterminable} are undeterminable (every letter outside the "
            f"top-{top_logprobs}; the returned-logprob bound is filed under "
            f"metric {MMLU_UNDETERMINABLE_METRIC!r}, the item unscored)")
    n = len(acc_by_item)
    if n != len(rows):
        raise SystemExit(f"[served-cap] mmlu: {n} scored items != "
                         f"{len(rows)} rows — refusing a partial aggregate")
    censored_items, censored_letter_counts = _mmlu_censorship_on_file(
        store, parity.PARITY_TASK_MMLU)
    summary = {"acc": sum(acc_by_item.values()) / n, "n": n,
               "top_logprobs": int(top_logprobs),
               "response_prefill": declare_prefill,
               "censored_items": censored_items,
               "censored_letter_counts": censored_letter_counts,
               "wall_clock_s": time.time() - t0}
    em.emit(task=parity.PARITY_TASK_MMLU, item_id="__aggregate__",
            metric=parity.PARITY_AGG_METRIC, value=summary["acc"], tier=1,
            extra={k: v for k, v in summary.items() if k != "acc"},
            wall_clock_s=summary["wall_clock_s"])
    print(f"[served-cap] mmlu done: acc={summary['acc']:.4f} n={n} "
          f"censored_items={censored_items} "
          f"censored_letter_counts={censored_letter_counts} "
          f"(seeded scoring, top-{top_logprobs} letter read)")
    return summary


def _mmlu_censorship_on_file(store, task: str) -> "tuple[int, dict]":
    """The cell's censorship, read from the STORE after every item is on
    file (the aggregate's authoritative source, this run's rows and a
    resumed store's alike): ``(items with >= 1 censored letter,
    {letter: how many items censored it})``. A row without the marker
    was written by the scorer that REFUSED every censored item, so it is
    uncensored by construction — the marker's absence is not read as
    unknown here because no scorer ever filed a censored row without
    it. A store that cannot be read back is a refusal, never a zero."""
    parsed = getattr(store, "_parsed_rows", None)
    if parsed is None:
        raise RuntimeError(
            "mmlu aggregate: the store exposes no parsed rows to count "
            "censored items from; refusing to file a count it cannot read")
    items = 0
    counts: dict = {}
    for r in parsed():
        if (r.get("task") != task or r.get("metric") != "acc"
                or r.get("item_id") == "__aggregate__"):
            continue
        letters = json.loads(r.get("extra_json") or "{}").get(
            "censored_letters") or []
        if letters:
            items += 1
            for letter in letters:
                counts[letter] = counts.get(letter, 0) + 1
    return items, dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# generate — the one subcommand                                                #
# --------------------------------------------------------------------------- #


#: the standup's record beside --routing-table (mt_serving_up.py writes both into --out)
STANDUP_ENDPOINT_FILENAME = "endpoint.json"


def _run_serving_engine(args, standup_endpoint: "dict | None") -> str:
    """The serving ENGINE this run is served by (``lens`` | ``lora_native``),
    read off the standup record — the strict :func:`_standup_endpoint` read
    when a tinycap run made one, else the record beside ``--routing-table``
    read LENIENTLY (a legacy controls run: XSTest / OR-Bench / MMLU never
    required the record) — and held to ``--serving-path`` when the run stated
    one. NO record: ``lens`` by era — only the native core writes ``engine``,
    and it publishes its record before it serves a request, so a native
    standup without one cannot be answering; a stated expectation with no
    record to hold it to refuses by name."""
    flag = "unstated" if args.serving_path is None else f"{FLAG_SERVING_PATH} {args.serving_path}"
    if standup_endpoint is not None:
        engine = _hold_serving_path_expectation(standup_endpoint, args.serving_path, source="standup record")
        source = "the standup record (strict read)"
    else:
        path = Path(args.routing_table).resolve().parent / STANDUP_ENDPOINT_FILENAME
        if not path.is_file():
            if args.serving_path is not None:
                raise SystemExit(
                    f"[served-cap] {FLAG_SERVING_PATH} {args.serving_path}: no standup record at {path} to hold the "
                    "expectation to — a native standup always publishes endpoint.json beside its table; refusing")
            engine, source = tc.SERVING_ENGINE_LENS, f"no standup record at {path}: lens by era"
        else:
            ep = json.loads(path.read_text())
            if not isinstance(ep, Mapping):
                raise SystemExit(f"[served-cap] standup record {path} is not a JSON object; no engine can be read from it")
            engine = _hold_serving_path_expectation(ep, args.serving_path, source=f"standup record {path}")
            source = f"standup record {path}"
    # ONE line naming the engine every row and manifest of this run is stamped
    # with — on the legacy controls path (no tinycap atoms) this is the only
    # place the engine is said out loud
    print(f"[served-cap] {tc.SERVED_ENGINE_FIELD}={engine!r} (read off {source}; {flag})", flush=True)
    return engine


def _cap_ladder_standup_endpoint(routing_table) -> Optional[dict]:
    """The standup's ``endpoint.json`` next to ``--routing-table`` (siblings by
    the standup's construction: ``mt_serving_up.py --out`` holds both, and
    ``endpoint.json`` names the table's path). The transport guard of the
    standing truncation rule reads the proxy's upstream hop timeout from it
    (``upstream_timeout_s``; cap_ladder.ENDPOINT_UPSTREAM_TIMEOUT_KEYS) —
    never from the proxy's compiled default. ``None`` when absent: the guard
    then refuses any climb as undeterminable (absent evidence is not evidence
    of adequacy), which is the loud outcome, not a crash of the base pass."""
    path = Path(routing_table).resolve().parent / STANDUP_ENDPOINT_FILENAME
    if not path.is_file():
        print(f"[served-cap] no standup record at {path}: the cap ladder's transport guard "
              "reads the proxy upstream timeout from it; without it any escalation is refused "
              "as undeterminable", flush=True)
        return None
    return json.loads(path.read_text())


def _measure_judged_ladder(*, spool, spec, control: str, cell_id: str, tok,
                           hop_timeouts_s=None):
    """The standing truncation rule's CHECK after one judged control's pass
    (``tiers/cap_ladder`` via ``cell_controls_run.measure_cap_ladder``): per-item
    sidecar, rung verdict, one loud line. On this tree a rung is the sampled
    run's ``--controls-cap/--escalate-from-root`` re-draw (its own meta rows
    carry the redraw record), so the check here measures the pass at ITS cap;
    the carry-over bundle branch did not travel to this tree."""
    from sisteer.capeval.tiers import cap_ladder as cl

    try:
        return ccr.measure_cap_ladder(spool, spec, tok=tok, control=control,
                                      cell_id=cell_id, prefix="[served-cap]",
                                      hop_timeouts_s=hop_timeouts_s)
    except cl.CapLadderError as e:
        raise SystemExit(f"[served-cap] cap-ladder {control}: {e}") from None


# --------------------------------------------------------------------------- #
# generate — the one subcommand                                                #
# --------------------------------------------------------------------------- #


def _ladder_args(args, thinking_by_task: Mapping[str, str]):
    """``(cap_overrides_by_task, item_selection)`` from ``--cap-override`` +
    ``--item-selection-json``, or None when neither was given. Refuses
    (``SystemExit``): one flag without the other (a selection at the table's
    cap re-draws the capped items for nothing; a raised cap over the whole leg
    is not a rung); a smoke cap beside them (``--limit`` / ``--inspect-limit``
    / ``--max-goals`` — a rung is not a smoke, and its only limit is the
    selection); the two maps naming different tasks; tasks that are not
    EXACTLY the row tasks this run files (a ladder run serves the fired legs
    alone); a selection that does not recompute
    (``contracts.parse_item_selection_json``)."""
    cap_spec = str(getattr(args, "cap_override", "") or "")
    sel_json = str(getattr(args, "item_selection_json", "") or "")
    if not cap_spec and not sel_json:
        return None
    if not (cap_spec and sel_json):
        raise SystemExit(f"[served-cap] {tc.FLAG_CAP_OVERRIDE} and {tc.FLAG_ITEM_SELECTION_JSON} travel together "
                         "(the truncation ladder's rung 2); one without the other is not a rung")
    if args.limit or args.inspect_limit or getattr(args, "max_goals", 0):
        raise SystemExit(f"[served-cap] a ladder rung takes no --limit / {tc.FLAG_INSPECT_LIMIT} / --max-goals; "
                         f"its only limit is {dict(tc.LADDER_LIMITS)}")
    try:
        caps = tc.parse_cap_overrides(cap_spec)
        selection = tc.parse_item_selection_json(sel_json)
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    if set(caps) != set(selection):
        raise SystemExit(f"[served-cap] {tc.FLAG_CAP_OVERRIDE} names {sorted(caps)} but "
                         f"{tc.FLAG_ITEM_SELECTION_JSON} names {sorted(selection)} — the two must name the same legs")
    if set(caps) != set(thinking_by_task):
        raise SystemExit(f"[served-cap] the ladder names {sorted(caps)} but this run files "
                         f"{sorted(thinking_by_task)} — a ladder run serves the fired legs alone")
    return caps, selection


def cmd_generate(args) -> None:
    from sisteer.capeval.tiers import cap_ladder as cl  # transport guard input
    # Pin the SDPA backend BEFORE _store_for records env_digest(): this
    # driver serves through vLLM and never flips the flag via an HF model
    # load, so unpinned it would fingerprint the torch-default backend
    # state (cell_controls_run.cmd_generate's discipline). Idempotent.
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()
    from sisteer.capeval import corpora
    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import judged

    cfg = jailbreak_run._load_cfg(args.config)
    controls = parse_controls(args.controls)
    # PW8 (E16 A3): the operator's per-entry source resolution — parsed
    # BEFORE any store is touched, so a malformed pair fails here, not at
    # the merge after a full run
    try:
        replace_source = tc.parse_replace_source(",".join(args.replace_source or []))
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    # -- tinycap controls: dispatch gate + the REQUIRED thinking arm ------- #
    atoms = tinycap_atoms(controls)
    thinking = getattr(args, "thinking", None)
    if atoms and thinking is None:
        gsm8k_note = (
            f" `{tc.CONTROL_GSM8K}` is the STANDARD GSM8K now (inspect_evals' stock "
            f"task on the paired-{tc.GSM8K_PAIRED_N} subset, row task {tc.GSM8K_STANDARD_TASK}) "
            "and takes the thinking arm like every Inspect control; the bespoke inline leg "
            "it replaced is retired (PW3)."
            if tc.CONTROL_GSM8K in atoms else "")
        raise SystemExit(
            f"[served-cap] {tc.FLAG_THINKING} "
            # the two ARMS first, then per-task: the arms are what an operator
            # reaches for, and the historical wording ("--thinking on|off") is
            # what existing refusal tests read
            f"{'|'.join(tc.THINKING_MODES)}|{tc.THINKING_PER_TASK} is REQUIRED when any tinycap "
            f"control is requested ({atoms[:3]}) — the thinking arm is never "
            f"implicit (P33).{gsm8k_note}")
    # U5-4 / PW20b-D: the {row task: arm} map this run FILES, resolved before
    # anything is read or touched — under `per-task` an unmapped task refuses
    # HERE, naming it
    thinking_by_task = _resolve_thinking_by_task(atoms, thinking) if atoms else {}
    # U8 / PW-sampling: the draw every decoding task is made under, resolved
    # before anything is read — a rowless model under `card` refuses HERE
    sampling = _resolve_sampling(args, cfg, atoms, thinking_by_task)
    # qwen-base-seed-replicates: the Inspect seed policy into THIS (sync) context BEFORE the event loop starts, so the
    # provider's request builder, the log adapter and every post-run reader dispatch the same rule (inspect_served.seed_for_task)
    from sisteer.capeval.tiers import inspect_served as _isv_policy
    _isv_policy.set_inspect_seed_policy(sampling.inspect_seed_policy, sampling.inspect_seed_base)
    # the identity VERSION (1 = the legacy serialisation, byte for byte; 2 =
    # the experiment's keys), the operator's cap STATEMENT (held equal to the
    # resolved caps below), the diagnostic marker, the FAMILY render fact
    # (torch-free, off the model key — verified against the served config's
    # family once the table is read), and the ledger requirement: a control in
    # LEDGER_REQUIRED_CONTROLS (MATH-500) never runs with usage_ledger null
    try:
        identity_version = tc.check_identity_version_arg(getattr(args, "tinycap_identity_version", 1))
        max_tokens_stmt = tc.parse_max_tokens_by_task(getattr(args, "max_tokens_by_task", None) or [])
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    # the ONE version test every experiment gate below keys on
    identity_v2 = identity_version == tc.TINYCAP_IDENTITY_VERSION_2
    diagnostic = bool(getattr(args, "diagnostic", False))
    if diagnostic and not identity_v2:
        raise SystemExit(f"[served-cap] {tc.FLAG_DIAGNOSTIC} hashes `diagnostic: true` into the identity, which "
                         f"only {tc.FLAG_TINYCAP_IDENTITY_VERSION} {tc.TINYCAP_IDENTITY_VERSION_2} carries; a "
                         "diagnostic store must never share a key with a production one (plan §5.2)")
    # the bounded ON-cap SETTER (the statement's grammar; its bounds apply where
    # each task's arm and table resolution are known, below), the rung's ITEM
    # SUBSET (validated file) and the ATTEMPT id — each an identity-version-2
    # mechanism, refused under version 1 by name like the statement; all parsed
    # BEFORE any store is touched
    try:
        on_cap_stmt = tc.parse_on_cap_by_task(getattr(args, "on_cap_by_task", None) or [])
        # the OFF-arm rung setter, the same grammar
        off_cap_stmt = tc.parse_off_cap_by_task(getattr(args, "off_cap_by_task", None) or [])
        attempt_id = (tc.check_attempt_id_arg(args.attempt_id)
                      if getattr(args, "attempt_id", None) else None)
        item_subset_path = str(getattr(args, "item_subset", "") or "")
        item_subset = tc.read_item_subset(item_subset_path) if item_subset_path else None
    except ValueError as e:        # OnCapSetterRefused / AttemptScopeError / ItemSubsetError are ValueErrors
        raise SystemExit(f"[served-cap] {e}") from None
    for flag, given in ((tc.FLAG_ON_CAP_BY_TASK, bool(on_cap_stmt)),
                        (tc.FLAG_OFF_CAP_BY_TASK, bool(off_cap_stmt)),
                        (tc.FLAG_ITEM_SUBSET, item_subset is not None),
                        (tc.FLAG_ATTEMPT_ID, attempt_id is not None)):
        if given and not identity_v2:
            raise SystemExit(f"[served-cap] {flag} is an identity-version-2 mechanism "
                             f"({tc.FLAG_TINYCAP_IDENTITY_VERSION} {tc.TINYCAP_IDENTITY_VERSION_2}); version 1 "
                             "writes no such key and reads the whole ledger (plan §3.1 / §4.4 / §4.5)")
    # The row runner's per-cell record — read and held to
    # contracts.ROW_ENGINE_MANIFEST_FIELDS BEFORE any store is touched (a wrong key set refuses
    # naming the missing / unknown keys); copied into the manifest UNHASHED, below. Absent (every
    # solo run), nothing about this run changes — the manifest is byte for byte what it was
    row_engine_record_path = str(getattr(args, "row_engine_record", None) or "")
    row_engine_block = None
    if row_engine_record_path:
        try:
            row_engine_block = tc.read_row_engine_record(row_engine_record_path)
        except tc.RowEngineRecordError as e:
            raise SystemExit(f"[served-cap] {tc.FLAG_ROW_ENGINE_RECORD}: {e}") from None
    # The GPU-side verification phase —
    # `coupled` (the default, byte-identical) or `generate` (every BigCodeBench task generated
    # WITHOUT scoring: the unscored .eval stays under the store, the generation sidecar + the
    # pending marker go to the ROW dir, `verification_pass` reads pending, RESULT reads GENERATED;
    # the CPU `verify` subcommand scores and files it later). `generate` needs the row dir (the
    # record's) and at least one BigCodeBench task, else it would silently defer nothing.
    verification_phase = str(args.verification_phase)   # argparse: coupled | generate (verify is its own subcommand)
    if verification_phase == tc.VERIFICATION_PHASE_GENERATE:
        if row_engine_block is None:
            raise SystemExit(f"[served-cap] {tc.FLAG_VERIFICATION_PHASE} {tc.VERIFICATION_PHASE_GENERATE} needs "
                             f"{tc.FLAG_ROW_ENGINE_RECORD} — the generation sidecar and the pending marker live in "
                             "the row dir the record names (plan §2.2.3)")
        if not tc.files_any_bigcodebench(thinking_by_task):
            raise SystemExit(f"[served-cap] {tc.FLAG_VERIFICATION_PHASE} {tc.VERIFICATION_PHASE_GENERATE}: this run "
                             f"files no BigCodeBench task ({sorted(thinking_by_task)}) — nothing to defer to a CPU "
                             "verifier; refusing a phase that would silently do nothing")
    generation_phase = None   # built below, once out_root is known (the row dir's files are keyed by the store)
    family = tc.family_of_model_key(sampling.model_key) if atoms else None
    render_fact = tc.render_fact_for(family) if family is not None else None
    if atoms and thinking in tc.THINKING_MODES:
        # PR-A (P7): the FAMILY gate on the whole-arm value, BEFORE any store is touched — the
        # no-switch arm `not_available` is admitted for a no-thinking-channel family only and refused
        # by name elsewhere (a family with a switch has an OFF arm; "not available" would mislabel it)
        try:
            tc.check_thinking_arg_for_family(thinking, family)
        except ValueError as e:
            raise SystemExit(f"[served-cap] {e}") from None
    if atoms and family is not None and identity_v2 and tc.inspect_task(tc.IFEVAL_TASK_NAME) in thinking_by_task:
        # a family with NO thinking switch (contracts.NO_THINKING_CHANNEL_FAMILIES)
        # runs its OFF arm as `not_available` — the hashed map, the manifest's
        # by_task, every row's `thinking`, the outcome header and the summary record
        # it; the provider sends no switch; the evidence gate requires its ABSENCE.
        # An ON arm there refuses. SCOPED to the IFEval experiment's runs (identity
        # version 2 AND the IFEval task filed — the same scoping as its other
        # mechanisms): a legacy run, and a version-2 run of another instrument, of a
        # no-switch family keeps its recorded `off` and its inert switch, so every
        # existing llama8b store resumes unchanged and the version-1 / version-2
        # goldens hold; an explicit `--thinking not_available` is the per-launch
        # opt-in for every other run.
        try:
            thinking_by_task = tc.apply_no_switch_policy(thinking_by_task, family)
            if thinking in tc.THINKING_MODES:
                # a WHOLE-arm mode is an arm, so the hashed MODE label
                # (`tinycap_identity.thinking.mode`, the manifest's `thinking.mode`) resolves through
                # the same derivation — a llama IFEval store launched with the frozen matrix line
                # `--thinking off` and one launched with `--thinking not_available` hash ONE
                # identity (`mode: not_available`, `by_task: {inspect/ifeval: not_available}`)
                thinking = tc.effective_arm_for_family(thinking, family)
        except tc.NoThinkingChannel as e:
            raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: {e}") from None
    # llama-paper-rows (P7 / P18): `no_switch_run` selects the NO-SWITCH certification class everywhere below (the family's
    # fact must say the template carries no switch: contracts.cert_class_for); the production preflight seam:
    # a CANDIDATE no-switch fact refuses the run by name until the diagnostic smoke certifies it; `--diagnostic` (the smoke
    # itself) admits it
    no_switch_run = False
    if atoms and thinking == tc.THINKING_NOT_AVAILABLE:
        try:
            tc.cert_class_for(thinking, render_fact)      # refuses a switched family's fact by name
        except ValueError as e:
            raise SystemExit(f"[served-cap] {e}") from None
        no_switch_run = True
        if not getattr(args, "diagnostic", False):
            try:
                tc.check_render_fact_certified(family)
            except tc.RenderFactUnverified as e:
                raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: {e}") from None
    # `--bank-view` (the cross-eval guard reads ROWS): the bank view the run's objects come from, recorded
    # UNHASHED on the manifest's route block and on every emitted row beside the arm; absent = rows
    # byte-identical to before
    try:
        bank_view = tc.check_bank_view_arg(args.bank_view) if getattr(args, "bank_view", None) else None
    except ValueError as e:
        raise SystemExit(f"[served-cap] {e}") from None
    # under identity version 2 the driver's probes go THROUGH the usage ledger
    # when one is given (labelled rows, one per attempt), like the Inspect
    # traffic; a version-1 run keeps its probes on the proxy (byte-identical:
    # no untagged rows in a legacy ledger)
    probe_base_url = args.ledger_url if (identity_v2 and args.ledger_url) else args.proxy_url
    if atoms and args.ledger_url == "" and identity_v2 \
            and any(a in tc.LEDGER_REQUIRED_CONTROLS for a in atoms):
        # scoped to the EXPERIMENT (identity version 2, like every other new gate
        # — LA15/LA16): a version-1 standard-battery MATH-500 run needs no ledger,
        # exactly as before this experiment. ifeval-canonical-battery: the
        # IFEval atom is ledger-required too — its request-evidence gate below
        # REFUSES an empty ledger, so an invocation without the flag must refuse
        # HERE, before a request, a marker or a store, not after 1,082 draws
        required = sorted(a for a in atoms if a in tc.LEDGER_REQUIRED_CONTROLS)
        raise SystemExit(
            f"[served-cap] {tc.FLAG_LEDGER_URL} is REQUIRED for {required} "
            f"under {tc.FLAG_TINYCAP_IDENTITY_VERSION} {tc.TINYCAP_IDENTITY_VERSION_2}: a run of {required} in this "
            "experiment with usage_ledger null has no per-call record to hold the cell to (plan §4.3 item 9, D5 P1-1; "
            "the IFEval request-evidence gate refuses an empty ledger, so the run is refused before any request); "
            "start the usage ledger in front of the proxy and pass its /v1 base")
    # E50 #3: the override is not a family-wide licence — it authorises ONE
    # experiment (contracts.AUTHORISED_OOS_THINKING), and this driver is a
    # separate entry point from the launcher, so it checks the tuple itself.
    # Here, before any store is touched: the flag also stamps
    # out_of_support_thinking into the hashed identity below, and a store keyed
    # that way by an unauthorised run is indistinguishable from the real one.
    # Both doors hand the contracts the COMPLETE parsed `controls`, never the
    # tinycap `atoms` (the code review's H1): `tinycap_atoms` drops the legacy
    # served controls (xstest, mmlu, or_bench_hard_1k) that `_judged_specs` and
    # the mmlu leg still EXECUTE from the original list, so `--controls
    # gsm8k,xstest` under either flag passed as the GSM8K-only experiment and
    # ran workloads outside the tuple on paths the render / stamp enforcement
    # never sees. The launcher and its worker already check the complete list
    # (`ctl` / `wanted`); the contracts refuse a legacy token by name.
    # The two named overrides are mutually exclusive — refused
    # BEFORE either authorisation check, naming both flags
    try:
        tc.check_override_flags_exclusive(
            out_of_support_thinking=bool(args.allow_out_of_support_thinking),
            off_cap_override=bool(args.allow_off_cap_override), where="driver run")
    except tc.OverrideFlagsExclusive as e:
        raise SystemExit(f"[served-cap] {tc.REC_OFFCAP_UNAUTHORISED}: {e}") from None
    oos_experiment = None
    if args.allow_out_of_support_thinking:
        # this door STATES the served identity. The registry holds two GSM8K pairs
        # that differ from each other ONLY in reduction_mode — the lens-era
        # `gsm8k_paired200_<model>_<cap>` (stacked) and the native battery's
        # `…_native_<cap>` (not_applicable) — so a door that leaves the mode unstated
        # matches both for every cell and refuses `ambiguous authorisation`; and THIS
        # door's return is the name the manifest records
        # (MANIFEST_KEY_AUTHORISED_EXPERIMENT) and the raised cap is read under, so it
        # must be the one that decides. The mode is a PURE READ of the standup record
        # beside --routing-table through the same reader the served identity below uses
        # (_served_reduction_mode: native record -> not_applicable; lens record -> its
        # attested lens_reduction_mode, absent = per_row), so the three doors state one
        # value. The full _standup_endpoint read below keeps every refusal it has and
        # its order; a record not yet present leaves the mode UNSTATED here, exactly as
        # before, and that later read refuses in its own words. Not a store: nothing is
        # written before this door decides.
        first_door_record = Path(args.routing_table).resolve().parent / STANDUP_ENDPOINT_FILENAME
        first_door_reduction_mode = (
            _served_reduction_mode(json.loads(first_door_record.read_text()), source="standup record")
            if first_door_record.is_file() else None)
        try:
            # The ONE registry-aware
            # checker returns the MATCHED tuple's name (recorded, unhashed); every
            # tuple's render fact agrees with its family's before any is relied on
            tc.check_authorised_render_facts()
            oos_experiment = tc.check_out_of_support_thinking_authorised(
                model=sampling.model_key, cell=args.cell, controls=controls,
                mode=thinking, sampling_mode=sampling.mode, samples=sampling.samples,
                reduction_mode=first_door_reduction_mode,
                # The cap STATEMENT, compared
                # when stated — the 8,192 and 16,384 GSM8K tuples are two doors
                max_tokens_by_task=(dict(max_tokens_stmt) if max_tokens_stmt else None),
                where="driver run")
        except tc.OutOfSupportThinkingUnauthorised as e:
            # BEFORE the attestation compare, and under its own REC: an
            # unauthorised mode (per-task, say) exits OOS_UNAUTHORISED, never
            # a declaration MISMATCH the compare would have filed for it
            raise SystemExit(f"[served-cap] {tc.REC_OOS_UNAUTHORISED}: {e}") from None
        except tc.RenderFactUnverified as e:
            raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: {e}") from None
    if args.allow_off_cap_override:
        # The SECOND named override authorises ONE comparator
        # (contracts.AUTHORISED_OFF_CAP_COMPARATOR); a `--thinking on` or
        # `per-task` under the flag refuses HERE by name (`mode differs`)
        try:
            tc.check_off_cap_override_authorised(
                model=sampling.model_key, cell=args.cell, controls=controls,
                mode=thinking, sampling_mode=sampling.mode, samples=sampling.samples,
                where="driver run")
        except tc.OffCapOverrideUnauthorised as e:
            raise SystemExit(f"[served-cap] {tc.REC_OFFCAP_UNAUTHORISED}: {e}") from None
    # The override and the experiment's DECLARED render fact enter the Inspect
    # task kwargs from HERE, once — without them a
    # `--allow-out-of-support-thinking --thinking on` run stamped its store out
    # of support yet planned GSM8K at the in-support 2,048 floor with
    # `explicit_tags` and trusted a bare done marker. The fact is the tuple's
    # `prompt_opens_think_on`, applied only when this run serves an ON task (it
    # is a fact about the ON render; an OFF task splits `record_only`
    # regardless). Its RUNTIME certification against the live header is
    # _certify_header's, recorded in evidence.json as
    # prompt_opened_think_certified beside the declared fact.
    # THREE names that cannot be confused:
    #   oos_run        = the RENDER AUTHORISATION ("this run's render is out of the
    #                    fit support": the ON render on a thinking-OFF-fit object);
    #                    False for the comparator, whose render is the fit render.
    #                    It gates everything about the RENDER (parser composition,
    #                    header measurement / certification, the OOS identity keys)
    #                    AND is the ONLY out_of_support the TASK PLANNER ever sees.
    #   offcap_run     = the CAP OVERRIDE (`--allow-off-cap-override`): the executed
    #                    thinking-OFF cap of the named tasks + the hashed cap key.
    #   held_to_record = oos_run or offcap_run — the ATTESTATION POSTURE ("this run
    #                    is held to the LANDED record, fail-closed"): the record
    #                    compares and the stamp enforcement ONLY. Never a render
    #                    fact; never handed to a planner.
    oos_run = bool(args.allow_out_of_support_thinking)
    offcap_run = bool(args.allow_off_cap_override)
    held_to_record = oos_run or offcap_run
    any_on = any(tc.enable_thinking(arm) for arm in thinking_by_task.values())
    if render_fact is not None:
        # The render fact is the FAMILY's, for ANY ON run
        # (in support or out) — never the authorisation's
        oos_prompt_opened_think = bool(render_fact.prompt_opens_think_on and any_on)
    else:
        oos_prompt_opened_think = bool(
            oos_run and tc.AUTHORISED_OOS_THINKING.prompt_opens_think_on and any_on)
    # A thinking-ON out-of-support
    # run of the LENS cell is the DECLARED configuration — the standup serves
    # the cell under its `@thinking` spelling (the route's cell id, the vmodel,
    # the attestation key), the CPU prompt-budget artifact is required and the
    # header is certified; every other configuration keeps today's spellings
    # ONE scope for the prompt-budget artifact (both doors, review round B):
    # every out-of-support thinking-ON run — the base route's ON leg included,
    # whose 200 prompts + the raised cap must fit the served window too — needs
    # the CPU artifact, is held to its template digest after the standup and
    # re-asserts its lengths through the live /v1/tokenize pass
    oos_on_run = bool(oos_run and thinking in tc.THINKING_MODES and tc.enable_thinking(thinking))
    # `lens_cell`: a FITTED cell — the lens-route thinking/prefill DECLARATION
    # contract's predicate (contracts.is_lens_cell reads the cell grammar's
    # kind, so a natively served LoRA cell reads True here too), NOT the
    # steering gate: bites / logprob-bites / the twin key on `bs.is_steered`.
    # Under a native record the declared branches below are closed by
    # `header_located = False` (the engine's fact).
    lens_cell = tc.is_lens_cell(args.cell)
    # the declaration predicate is the cell's SERVED mask, read off the
    # standup record's lens_served_positions once it is loaded below
    # (absent -> header-located, fail closed — the parent's declared path);
    # a non-lens route has no mask
    header_located = bool(lens_cell)
    served_positions = None
    oos_declared_run = bool(oos_on_run and lens_cell and header_located)
    # whether THIS run serves a fitted intervention outside its fit support is
    # a fact about the ROUTE (a lens cell) and the RENDER (any thinking-ON
    # task) read off the standup record below — never about the override flag,
    # which only acknowledges it. Set where the served positions are known;
    # False on every non-lens route
    fitted_on_why = None
    object_on_run = False
    oos_budget = None
    request_evidence = None
    row_identity = None   # ifeval-canonical-battery P42: the per-row object identity of a --bank-view run
    # the rung's ids (set once the subset is bound to its source) and the
    # attempt scope (set once the evidence writer exists); the rung_subset/2
    # DRAWS beside the ids on a draw-level (k = 2) rung, None on an
    # item-level one
    item_subset_ids = None
    item_subset_draws = None
    attempt_scope = None
    sampling_ref = None
    attestation = None
    stamp_policy = twin_stamp_policy = None
    # U10 item 4: the OR-Bench item set, derived from the CONTRACTS (pure —
    # positions only), never from the corpus; None when no OR-Bench row is filed
    or_bench_item_set = (tc.run_or_bench_item_set(atoms, full=bool(args.or_bench_full))
                         if atoms else None)
    tinycap_identity = None
    standup_endpoint = None
    standup_native = False   # the record's engine bit, read ONCE below; no record reads False
    binding = None
    # bcb-bfcl-200 §S2b: the truncation ladder's rung-2 tuple — both flags or
    # neither, recomputed, naming EXACTLY the row tasks this run files, no smoke
    # cap beside it; None on every other run
    ladder = _ladder_args(args, thinking_by_task)
    # the ACTIVE limits (contracts.LIMIT_KINDS): identity, and the reason no
    # task of a limited run may write a done marker (R2); a ladder run is
    # LIMITED by exactly {"item_selection": 1} (a selection run ends PARTIAL)
    limits = completion.active_limits(limit=args.limit, inspect_limit=args.inspect_limit,
                                      max_goals=args.max_goals,
                                      item_selection=(1 if ladder is not None else None))
    inspect_atoms = [a for a in atoms if tc.control_family(a) in tc.INSPECT_CONTROLS]
    evalplus_atoms = [a for a in atoms if tc.control_family(a) in tc.EVALPLUS_CONTROLS]
    out_root = ccr._out_root(args)
    if row_engine_block is not None:
        # the row's generation facts the Inspect leg reads —
        # the record's row dir, this cell, the container's FunctionCall id (the pending marker's
        # `generation_fcid`; None on a hand run), the driver's concurrency (S // K on a K-concurrent row) and
        # the store root's basename (the row dir's generation files are keyed by it: the in-run repeat runs
        # the same --cell into another store)
        from sisteer.capeval.tinycap import sandbox_scope as _scope
        gen_owner = None
        if verification_phase == tc.VERIFICATION_PHASE_GENERATE:
            # the owner tuple is read under the GENERATE phase alone — it supplies the marker's generation_fcid;
            # a coupled row writes no marker and must not refuse on an environment it never reads
            try:
                gen_owner = _scope.owner_from_env()
            except ValueError as e:
                raise SystemExit(f"[served-cap] {e}") from None
        generation_phase = _CV.GenerationPhase(
            row_dir=Path(row_engine_block["row_dir"]), cell=str(args.cell),
            generation_fcid=(gen_owner.function_call_id if gen_owner is not None else None),
            concurrency=int(args.concurrency), store=Path(out_root).name)
    # PW8 (E16 A2): the geometry pass this run executes under, from the
    # launcher's record beside the cells (None for a hand run) — identity,
    # so pass 1 and pass 2 are two stores even when they resolve to one
    # window and S; the manifest's geometry block copies the same value
    pass_no = _launcher_pass(_launcher_geometry_record(out_root))
    if atoms:
        # what the standup actually SERVED (PB13) — refuses before any
        # store is touched when the record is absent or, under thinking-ON
        # with generation controls, names no reasoning parser (PB12)
        standup_endpoint = _standup_endpoint(args.routing_table, cell=args.cell,
                                             expect_serving_path=args.serving_path)
        # the record's ENGINE, read once: a pure read of the same record every native
        # branch below keys on (a torn record already refused inside _standup_endpoint)
        standup_native = _endpoint_is_native(standup_endpoint, source="standup record")
        # bcb-bfcl-200-widen: the clamp override the standup launched
        # under is held to this run's sampling mode — present iff protocol —
        # BEFORE any served fact is hashed (a protocol run on an unmodified
        # server would file 0.01 draws under a 0.001 label; a card run on a
        # lifted server is a server it never asked for)
        _check_served_clamp_override(standup_endpoint, sampling_mode=sampling.mode)
        # ... and BOUND to the proxy / table / ledger this run was pointed
        # at (N2) before any served fact is copied off it
        binding = _bind_endpoint(standup_endpoint, proxy_url=args.proxy_url,
                                 routing_table=args.routing_table,
                                 ledger_url=args.ledger_url)
        if lens_cell:
            # the standup keys its served-positions map by the spelling it serves
            # the cell as — the declared `@thinking` harness id on a declared
            # standup, the bare id otherwise — so both are looked up and exactly
            # one may hit (the bare-key-only read returned None on every declared
            # cell of the gsm8k-paired200 wave and recorded it as null; the
            # fail-closed reading below was right, the record was not)
            served_positions, served_positions_key = _standup_served_positions(standup_endpoint, cell=args.cell)
            header_located = tc.header_located(served_positions)
            if standup_native:
                # a NATIVE standup records no served-positions map — a native adapter
                # applies at EVERY position (the fit's `positions == "all"` is what
                # `native_lora_unservable_reason` admits at enumeration), so "not
                # header-located" is the ENGINE's fact, never a missing record read
                # fail-closed; a fitted cell takes the UNDECLARED all-position path
                # (the bites probe renders OFF; no declaration is sent; a declared
                # configuration refuses by name in contracts.check_serve_attestation)
                header_located = False
            oos_declared_run = bool(oos_on_run and header_located)
            # The driver USED to refuse a HEADER-LOCATED cell on a header-INVARIANT family
            # under thinking-ON BY NAME here (the declaration was required at the proxy and
            # refused as vacuous at the standup). WITHDRAWN: a header that does not move
            # under the thinking switch is no reason not to serve or steer thinking-ON — the
            # standup, export, worker and shim now admit the EQUAL second header, and such a
            # cell proceeds on the declared path like any other header-located cell (in support:
            # contracts.fitted_thinking_on_out_of_support). Stated, so a reader sees which.
            if any_on and header_located and family in tc.HEADER_INVARIANT_THINKING_FAMILIES:
                print(f"[served-cap] cell {args.cell!r} on {sampling.model_key}: header-located (served positions "
                      f"{served_positions}) on the header-INVARIANT family {family!r} — the declared thinking-ON "
                      "render locates the same assistant header as OFF; served on the declared path (the user's "
                      "ruling 2026-09-10; the plan §3.3 by-name refusal is withdrawn)", flush=True)
            if oos_on_run and header_located and served_positions is not None:
                # the served mask IS header-located: this configuration must be
                # DECLARED (the parent's path) — stated, so a reader sees which
                print(f"[served-cap] cell {args.cell!r} serves header-located positions {served_positions}: "
                      "the thinking-ON declaration is REQUIRED on every lens-route request (contract §2.5.4)",
                      flush=True)
            if any_on:
                # a FITTED cell under the thinking-ON render is out of its fit
                # support by the served positions and the family
                # (contracts.fitted_thinking_on_out_of_support) — not by the flag.
                # Such a run REQUIRES the override (the acknowledgement), a
                # matching registry tuple and the artifact digest, and runs the
                # predicates; without the flag it REFUSES here, by name, before
                # the parser admission, any budget pass, any store. Before this
                # gate the certified-render admission let an unflagged fitted ON
                # run through as `in_support` with no tuple, digest, (B')/(C) or
                # request-evidence gate
                fitted_on_why = tc.fitted_thinking_on_out_of_support(family, header_located=header_located)
                object_on_run = fitted_on_why is not None
                if object_on_run and not oos_run:
                    raise SystemExit(
                        f"[served-cap] {tc.REC_OOS_UNAUTHORISED}: cell {args.cell!r} is a FITTED lens route "
                        f"served under the thinking-ON render, which is outside its fit support — "
                        f"{fitted_on_why}. This run REQUIRES {tc.FLAG_ALLOW_OOS_THINKING} (an explicit "
                        "acknowledgement: the route and the render are the trigger, not the flag), a "
                        "registry tuple in contracts.AUTHORISED_OOS_EXPERIMENTS matching this run, and "
                        f"{tc.FLAG_EXPECT_OBJECT_SHA256} <digest> tying the authorised intervention to the "
                        "served artifact; the certified render fact admits the RENDER on bare and prompted "
                        "routes, never a fitted intervention under it (plan §4.3 item 1b, §4.4)")
        _check_served_reasoning_parser(
            atoms, thinking, standup_endpoint,
            by_task=thinking_by_task if tc.is_per_task(thinking) else None,
            allow_out_of_support=bool(args.allow_out_of_support_thinking),
            render_fact=render_fact, require_certified=not diagnostic)
        # fit-support transfer: the prompt-budget artifact (REQUIRED on a
        # thinking-ON out-of-support run — the worker hands its path over as
        # --oos-prompt-budget) and the standup record held to the CONTRACT —
        # the declarations map by own key (+ the twin on a declared run), the
        # served artifact digest(s), the salt policy, the attested per-cell
        # policy and the template digest — before any store is touched.
        # The artifact is REQUIRED on a thinking-ON out-of-support
        # run AND on an off-cap-override run (both cells: the window fit at the
        # raised cap must hold on the base route too)
        if oos_on_run or offcap_run:
            if not args.oos_prompt_budget:
                raise SystemExit(
                    f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: a thinking-ON "
                    f"out-of-support run, or a thinking-OFF cap-override run, needs the CPU "
                    f"prompt-budget artifact ({tc.FLAG_OOS_PROMPT_BUDGET} "
                    f"<path to {tc.OOS_PROMPT_BUDGET_FILENAME}>, the document the launcher's preflight "
                    f"wrote under rep[{tc.OOS_PROMPT_BUDGET_REPORT_FIELD!r}]) — its chat_template_hash is "
                    "compared to the standup's and its 200 lengths to the live /v1/tokenize pass; a "
                    "comparison that cannot be made is refused, never passed (D16 P2-a)")
            try:
                oos_budget = tc.read_oos_prompt_budget(args.oos_prompt_budget)
            except tc.OutOfSupportPromptBudgetError as e:
                raise SystemExit(f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: {e}") from None
        # the record compare takes the POSTURE (`held_to_record`): under either
        # override the landed record's presence rules; for the comparator's OFF
        # arm the declaration expectation it derives is still "undeclared"
        # (contracts fact 3) — nothing about the render enters here
        attestation = _check_serve_attestation(
            standup_endpoint, cell=args.cell, arm=thinking, out_of_support=held_to_record,
            expected_object_sha256=(args.expect_object_sha256 or None), budget=oos_budget,
            model_key=sampling.model_key, off_cap_override=offcap_run,
            header_located=header_located)
        # the resumed_prefill_policy label this leg's responses must wear (the
        # provider, the summary check, the probe's steered leg) and the probe
        # twin's — the STRICT §2.5.11 row on the authorised experiment's legs,
        # the cell's ATTESTED label off the record on every other run (review
        # round B, P1); derived ONCE here, from the record just held to the
        # contract, so no consumer re-derives it from the arm alone
        try:
            # the posture again (`conditional` / `not-consulted` either way on the
            # authorised legs; the cell's attested label on every other run)
            stamp_policy = tc.expected_stamp_policy(
                standup_endpoint, cell=args.cell, arm=thinking, out_of_support=held_to_record,
                model_key=sampling.model_key, header_located=header_located)
            # the probe twin's key is the ENGINE's zero cell (none__lens0 on a lens
            # record, none__lora0 on a native one — contracts.twin_cell_id_for's
            # rule, read off the record here because the route is resolved further
            # down); on a native record expected_stamp_policy reads not-consulted
            # before it looks any key up
            twin_key = (tc.TWIN_CELL_ID_LORA if standup_native
                        else tc.TWIN_CELL_ID)
            twin_stamp_policy = tc.expected_stamp_policy(
                standup_endpoint, cell=args.cell, arm=thinking, out_of_support=held_to_record,
                model_key=sampling.model_key, harness_key=twin_key, header_located=header_located)
        except tc.OutOfSupportAttestationError as e:
            raise SystemExit(f"[served-cap] {e}") from None
        # PW5-2: every requested framework must be at its client-lock pin
        # (FrameworkPinMismatch otherwise) — at startup, before any store
        # is touched; the block is hashed identity below
        frameworks = _framework_versions({tc.control_family(a) for a in atoms})
        # ONE cap resolution for every Inspect task the run files — the plan
        # builder's own rule at each task's arm, under this run's support state,
        # named override and matched tuple (`inspect_served.task_max_tokens`, the
        # same derivation `plans_for_atoms` applies to the plans the run
        # executes). It fills the MATH-500 first-pass scalar, the override
        # maps and version 2's `max_tokens_by_task` alike, so the scalar and the
        # map's `inspect/math` entry can never part (m4 merge caveat)
        from sisteer.capeval.tiers import inspect_served as _isv_caps
        try:
            executed_caps = _override_max_tokens_by_task(
                inspect_atoms, thinking_by_task, out_of_support=oos_run, off_cap_override=offcap_run,
                oos_experiment=oos_experiment,
                # the bounded setter, through the same rule — its bounds refuse
                # HERE by name (the arm, the table)
                on_cap_override=(dict(on_cap_stmt) if on_cap_stmt else None),
                # bcb-bfcl-200 §S2b: on a ladder rung-2 run the VALIDATED rung replaces the table's
                # cap for the fired legs, so the map, the cap statement and `cap_overrides_by_task`
                # are one number
                cap_overrides_by_task=(ladder[0] if ladder is not None else None),
                # ifeval-canonical-battery §5.2: the OFF-arm rung setter, through the same rule
                off_cap_setter=(dict(off_cap_stmt) if off_cap_stmt else None))
        except _isv_caps.InspectServedError as e:
            raise SystemExit(f"[served-cap] {e}") from None
        for flag, stmt in ((tc.FLAG_ON_CAP_BY_TASK, on_cap_stmt), (tc.FLAG_OFF_CAP_BY_TASK, off_cap_stmt)):
            for name in stmt:
                if tc.inspect_task(name) not in executed_caps:
                    raise SystemExit(f"[served-cap] {flag} names {name!r}, which this run does not "
                                     f"plan (tasks {sorted(executed_caps)})")
        if item_subset is not None:
            # the rung's subset is THIS run's (task planned at the file's
            # cap) and bound to its SOURCE store (identity hash + executed cap)
            why = _item_subset_problem(item_subset, executed_caps=executed_caps, out_root=out_root)
            if why is not None:
                raise SystemExit(f"[served-cap] {tc.FLAG_ITEM_SUBSET} {item_subset_path}: {why}")
            item_subset_ids = [str(i) for i in item_subset["ids"]]
            if tc.item_subset_is_draw_level(item_subset):
                # a rung_subset/2 file narrows PER (item, trial) — only the listed
                # draws are re-drawn, each with its pass-1 seed; the ids above are
                # their distinct items
                item_subset_draws = tc.item_subset_draws(item_subset)
        # whether this run's identity carries the declaration expectation — a
        # declared (header-located) lens cell, any non-lens route, or an arm
        # that is not thinking-ON; an UNDECLARED all-position object under ON
        # does not. ONE predicate for the key's presence in the dict below and
        # for the exact-keys assertion
        declaration_expected = oos_declared_run or not lens_cell or not oos_on_run
        tinycap_identity = {
            "tasks": sorted(atoms),
            "thinking": {"mode": thinking,
                         "render": _render_label(_render_arm(thinking, thinking_by_task))},
            # U5-4 / PW20b-D: the per-task arm map is HASHED — under one
            # standup serving mixed flags, `thinking.mode` alone no longer
            # says what arm a task ran under
            tc.IDENTITY_KEY_THINKING_BY_TASK: dict(thinking_by_task),
            # the MATH-500 truncation-retry cap, recorded whether or not the
            # retry is armed (0 = off) — a store with the retry and one
            # without are different instruments. Stamped on EVERY store, so a
            # run that files no MATH-500 row keeps the legacy literal and its
            # key (the rule is contracts.math500_retry_identity_value)
            tc.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS: tc.math500_retry_identity_value(
                thinking_by_task, int(args.math500_retry_max_tokens)),
            # U10 item 4 / PW-battery: WHICH OR-Bench items this run filed, by
            # POSITION (None when it files no OR-Bench rows). 200 frozen items
            # and the full 1,319 are two refusal rates, not one at two
            # resolutions, so they must never share a store
            tc.IDENTITY_KEY_OR_BENCH_ITEM_SET: or_bench_item_set,
            "versions": {"tinycap": tc.TINYCAP_VERSION},
            # R4 / N3 / PW5-4: what the standup served is identity — a store
            # under one parser / tool-call mode / window / S / client width
            # can never resume onto another's (_served_identity)
            "served": _served_identity(standup_endpoint, concurrency=args.concurrency),
            # R2: a limited store never shares a key with the full one
            "limits": dict(limits),
            # the MATH-500 k this run uses (--samples explicit, else the mode's)
            "samples": int(sampling.samples),
            # U8 / PW-sampling: the run's draw — the row's temperature / top_p /
            # top_k, the per-task k map and the seed rules — is HASHED, so a
            # sampled store never mixes with a greedy one (the row's source
            # prose is not; it rides in the manifest's sampling block).
            # Under `protocol` the three scalars
            # are null and the per-task EXECUTED map `sampling_by_task` is hashed
            **_sampling_identity(sampling),
            # PW5-2: the imported framework versions + the client lock sha —
            # a dependency change selects a new store
            tc.MANIFEST_KEY_FRAMEWORKS: frameworks,
            # PW8 (E16 A2): the geometry pass — a pass-2 run never keys
            # onto a pass-1 store, however their windows resolve
            tc.GEOMETRY_PASS_FIELD: pass_no,
            # MATH-500 one-pass: the FIRST-PASS cap MATH is generated
            # at, PRESENT ONLY when this run files inspect/math — a non-MATH
            # identity stays byte-identical (the goldens), and a MATH store
            # generated at 4,096 never keys with, or resumes onto, one
            # generated at 8,192 whatever the retry field records
            **({tc.IDENTITY_KEY_MATH500_MAX_TOKENS: _math500_max_tokens(executed_caps)}
               if tc.files_math500(thinking_by_task) else {}),
            # PW-oos (U10 item 5): PRESENT ONLY under the override, so an
            # in-support store hashes exactly as it did before the flag
            # existed and the two can never resume onto each other. Fit-support
            # transfer adds, under the same override: the EXECUTED per-task
            # cap map (the plan builder's own rule, inspect_served.task_max_tokens,
            # so 8192 never resumes onto 2048) and the CONTRACT literal the
            # declared standup must attest (never the attestation itself — the
            # launcher derives the same value with no server)
            **({tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: True,
                tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK: executed_caps,
                # the declaration expectation is PRESENT only on a declared
                # (header-located) configuration — an undeclared all-position
                # object writes NO key, never null
                **({tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: tc.expected_serve_out_of_support(
                        lens=lens_cell, arm=thinking, out_of_support=True, header_located=header_located)}
                   if declaration_expected else {})}
               if oos_run else {}),
            # the comparator's ONE optional key — the EXECUTED
            # per-task cap map from the same derivation (the plan builder's own
            # rule under the cap override), so an OFF/8192 store never keys like,
            # or resumes onto, the campaign's OFF/512 store; no marker key beside
            # it (the present hashed cap IS the positive producer marker)
            **({tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK: executed_caps}
               if offcap_run else {}),
        }
        # version 2 adds `identity_version`, the EXECUTED per-task cap map
        # (every Inspect task of the run, in support too — held equal to the
        # operator's statement), `diagnostic` when marked, and — once the
        # served header is measured below — `thinking_header_ids`; version 1
        # writes NONE of them
        if identity_v2:
            for name, stated in max_tokens_stmt.items():
                row_task = tc.inspect_task(name)
                if row_task not in executed_caps:
                    raise SystemExit(f"[served-cap] {tc.FLAG_MAX_TOKENS_BY_TASK} names {name!r}, which this run does "
                                     f"not plan (tasks {sorted(executed_caps)})")
                if executed_caps[row_task] != stated:
                    raise SystemExit(f"[served-cap] {tc.FLAG_MAX_TOKENS_BY_TASK} {name}={stated} but the plan builder "
                                     f"resolves {executed_caps[row_task]} for {row_task} — the statement must EQUAL "
                                     "the resolved cap (a statement, never a setter; plan §4.3 item 5)")
            tinycap_identity[tc.IDENTITY_KEY_IDENTITY_VERSION] = tc.TINYCAP_IDENTITY_VERSION_2
            tinycap_identity.setdefault(tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK, executed_caps)
            if diagnostic:
                tinycap_identity[tc.IDENTITY_KEY_DIAGNOSTIC] = True
            if item_subset is not None:
                # a RUNG store's hashed binding to its source — present only
                # under --item-subset
                tinycap_identity[tc.IDENTITY_KEY_CAP_RUNG] = tc.cap_rung_identity(item_subset)
        elif max_tokens_stmt:
            raise SystemExit(f"[served-cap] {tc.FLAG_MAX_TOKENS_BY_TASK} is an identity-version-2 statement "
                             f"({tc.FLAG_TINYCAP_IDENTITY_VERSION} 2); version 1 writes no such key")
        sampling_opt = tuple(f for f in tc.SAMPLING_OPTIONAL_ROW_FIELDS if getattr(sampling.row, f, None) is not None)
        if offcap_run:
            # the SERVED reduction mode is part of the authorised comparator
            # tuple too — held to it here, where the attested mode is first known
            try:
                tc.check_off_cap_override_authorised(
                    model=sampling.model_key, cell=args.cell, controls=controls,
                    mode=thinking, sampling_mode=sampling.mode, samples=sampling.samples,
                    reduction_mode=tinycap_identity["served"][tc.REDUCTION_MODE_FIELD],
                    where="driver run")
            except tc.OffCapOverrideUnauthorised as e:
                raise SystemExit(f"[served-cap] {tc.REC_OFFCAP_UNAUTHORISED}: {e}") from None
        # the two pinned legs' HASHED item sets and the
        # BigCodeBench sandbox facts, PRESENT ONLY when this run files the leg
        # (contracts.files_bigcodebench_200 / files_bfcl_200 — the launcher's
        # predicate too), from the VERIFIED pins on this tree (a pin that does
        # not regenerate from its rule refuses here, before any request)
        if tc.files_bigcodebench_200(thinking_by_task):
            tinycap_identity[tc.IDENTITY_KEY_BCB_ITEM_SET] = tc.item_set_identity(tc.load_bcb_pin(REPO))
            tinycap_identity[tc.IDENTITY_KEY_SANDBOX] = tc.bcb_sandbox_identity()
        if tc.files_bfcl_200(thinking_by_task):
            tinycap_identity[tc.IDENTITY_KEY_BFCL_ITEM_SET] = tc.item_set_identity(tc.load_bfcl_pin(REPO))
        # capeval-full-tasks (2026-09-23): the FULL siblings' own item-set keys from their
        # verified full pins; the sandbox facts key rides on EITHER BigCodeBench leg
        if tc.files_bigcodebench_full(thinking_by_task):
            tinycap_identity[tc.IDENTITY_KEY_BCB_FULL_ITEM_SET] = tc.item_set_identity(tc.load_bcb_full_pin(REPO))
            tinycap_identity[tc.IDENTITY_KEY_SANDBOX] = tc.bcb_sandbox_identity()
        if tc.files_bfcl_full(thinking_by_task):
            tinycap_identity[tc.IDENTITY_KEY_BFCL_FULL_ITEM_SET] = tc.item_set_identity(tc.load_bfcl_full_pin(REPO))
        # the ladder's rung-2 tuple, PRESENT ONLY on a
        # ladder run — the caps per fired task and {n, ids_sha256} over the
        # selected PINNED ids (contracts.item_selection_identity), so a rung-2
        # store never keys with, or resumes onto, the rung-1 store or another
        # selection's rerun
        if ladder is not None:
            caps_by_task, selection = ladder
            tinycap_identity[tc.IDENTITY_KEY_CAP_OVERRIDES] = dict(caps_by_task)
            tinycap_identity[tc.IDENTITY_KEY_ITEM_SELECTION] = tc.item_selection_identity(
                {t: b["ids"] for t, b in selection.items()})
        if args.allow_out_of_support_thinking:
            # the SERVED reduction mode is part of the authorised
            # tuple — held to it here, where the attested mode is first known
            try:
                tc.check_out_of_support_thinking_authorised(
                    model=sampling.model_key, cell=args.cell, controls=controls,
                    mode=thinking, sampling_mode=sampling.mode, samples=sampling.samples,
                    reduction_mode=tinycap_identity["served"][tc.REDUCTION_MODE_FIELD],
                    max_tokens_by_task=(dict(max_tokens_stmt) if max_tokens_stmt else None),
                    where="driver run")
            except tc.OutOfSupportThinkingUnauthorised as e:
                raise SystemExit(f"[served-cap] {tc.REC_OOS_UNAUTHORISED}: {e}") from None
    print(f"[served-cap] concurrency={args.concurrency} "
          f"(table default {CAPABILITY_GEOMETRY.max_num_seqs} = the "
          f"'capability' geometry's max_num_seqs at max_model_len "
          f"{CAPABILITY_GEOMETRY.max_model_len}; basis: "
          f"{CAPABILITY_GEOMETRY.basis.note})")
    cell = ccr._pick_cell(cfg, args)

    table = load_routing_table(Path(args.routing_table))
    routes = table["routes"]
    # the route under the spelling this standup serves the cell as: the
    # declared `@thinking` harness id on the out-of-support thinking-ON lens
    # leg (contract §2.5.2 / §2.5.10 door table), the bare translation otherwise
    route = usr._served_route_for(
        cell.id, routes,
        harness_id=(tc.cell_spellings(args.cell, declared=True).harness
                    if oos_declared_run else None))
    prov = _mirror_provenance(cell, expected_object_sha256=(args.expect_object_sha256 or None))
    model_id, model_revision = _model_identity(cfg, table, prov)
    system = ccr._cell_system(cell, cfg)
    judge_model = ccr._judge_model(args)

    served_model_key = _sampling_model_key(model_id)
    if atoms and sampling.mode in tc.SAMPLING_TABLE_MODES and served_model_key != sampling.model_key:
        # the row was looked up under the CONFIG's model; the SERVED model (the
        # routing table's base_model) must pin EXACTLY the same key, or the
        # table values hashed into this store would be another model's. Every
        # table-backed mode (card, thinking_card, general, battery): the family
        # cross-check further down cannot tell Qwen3.5-4B from 9B, and the
        # authorised tuple names a model KEY, not a family
        raise SystemExit(
            f"[served-cap] {tc.FLAG_SAMPLING} {sampling.mode}: the sampling row was resolved for "
            f"model key {sampling.model_key!r} (config model {cfg.get('model')!r}) but the SERVED "
            f"model {model_id!r} (the routing table's base_model) pins key {served_model_key!r} — "
            f"refusing to hash one model's {sampling.mode} row over another model's outputs; the "
            "served model must be the exact pinned HF id of the row's key (D47 harness (3))")
    if atoms and oos_experiment is not None:
        # the authorisation was matched on the CONFIG's model key before the
        # served model was known; it is BOUND here to the RESOLVED served
        # identity — the same registry compare over the served model's key
        # (None when the served id pins no key), so a tuple can never
        # authorise a run on a model other than the one actually served
        try:
            tc.check_out_of_support_thinking_authorised(
                model=served_model_key, cell=args.cell, controls=controls,
                mode=thinking, sampling_mode=sampling.mode, samples=sampling.samples,
                reduction_mode=tinycap_identity["served"][tc.REDUCTION_MODE_FIELD],
                max_tokens_by_task=(dict(max_tokens_stmt) if max_tokens_stmt else None),
                where="driver run (served model)")
        except tc.OutOfSupportThinkingUnauthorised as e:
            raise SystemExit(
                f"[served-cap] {tc.REC_OOS_UNAUTHORISED}: the SERVED model {model_id!r} (key "
                f"{served_model_key!r}) is not the model the matched tuple {oos_experiment!r} authorises "
                f"— {e}") from None
    ihash = route.intervention_hash
    # THE SAMPLED RE-RUN (module docstring): the controls cap ladder rewrites an
    # in-memory copy of the config BEFORE the gen and the store identity are
    # computed, so a rung store gets its own key; the sampled scope is the
    # LEGACY judged controls (xstest / or_bench_hard_1k; mmlu unaffected) — the
    # tinycap atoms draw under their own contracts rows (--sampling), untouched
    legacy_controls = [c for c in controls if c in SERVED_CONTROLS]
    # THE DEFAULT (sisteer.eval.sampled_decode): the
    # judged over-refusal controls decode under the served model's VENDOR preset by
    # default — a config with no block gets the data file's row for the SERVED model
    # (the routing table's identity, revision included) injected here, so the store
    # keys, sidecar and ladder behave exactly as the sampled re-run's; ``--greedy``
    # is the explicit opt-in for argmax (the greedy store identity, byte for byte),
    # recorded in gen_stats.json; a served model with no row REFUSES naming the flag.
    # Runs with no sampled control requested (mmlu alone, tinycap atoms) inject nothing.
    decode_resolution: dict
    if any(c in SAMPLED_CONTROLS for c in legacy_controls):
        from sisteer.eval import sampled_decode as sdd
        from sisteer.eval.cell_controls import controls_cap as _controls_cap_of
        try:
            cfg, decode_resolution = sdd.resolve_default_decode(
                cfg, greedy=bool(getattr(args, "greedy", False)), trial=int(getattr(args, "trial", 0) or 0),
                # the controls' OWN cap (attacks.controls_max_new_tokens: 512 on every wave config),
                # never the multi-turn target's 4,096
                controls_cap=_controls_cap_of(cfg),
                model=model_id, revision=model_revision)
        except (sdd.DecodePresetUnavailable, sdd.DecodePresetsMalformed) as e:
            raise SystemExit(f"[served-cap] {e}") from None
        # the roster's capability omissions ride on the injected default block's readouts (capability_omissions)
        from sisteer.eval import capability_omissions as _omit
        cfg = _omit.apply_to_config(cfg, decode_resolution.get("preset_key"))
        print(f"[served-cap] decode resolution: {decode_resolution['decode_default']} "
              f"({decode_resolution.get('preset_key') or 'no preset'}) — {decode_resolution['reason']}")
    else:
        # no sampled control requested (mmlu alone, the tinycap atoms): the default does not
        # apply and NOTHING is recorded — every such store's gen_stats.json stays byte-identical
        decode_resolution = None
    if getattr(args, "controls_cap", None) is not None:
        no_ladder = [c for c in legacy_controls if c in SAMPLED_UNAFFECTED_CONTROLS]
        if no_ladder:
            raise SystemExit(f"[served-cap] --controls {no_ladder} on a rung: a teacher-forced "
                             "one-token readout has no cap ladder; refusing")
    cfg, esc = _apply_escalation(args, cfg, cell_id=cell.id, ihash=ihash,
                                 out_root=out_root, controls=legacy_controls)
    try:
        gen = _gen_cfg(cfg)
    except SampledDecodeError as e:
        raise SystemExit(f"[served-cap] {args.config}: {e}")
    decode = _decode_of(gen)
    if getattr(args, "controls_events", False) and (
            decode is None or not any(c in SAMPLED_CONTROLS for c in legacy_controls)):
        # the stream is the SAMPLED ladder's producer record (an at-cap draw is what it feeds
        # forward); a greedy store has no ladder and a run with no judged control has no draw
        raise SystemExit(f"[served-cap] {cev.CONTROLS_EVENTS_FLAG} rides a SAMPLED judged-controls run "
                         f"(a sampled_decode config with --controls naming {list(SAMPLED_CONTROLS)}); "
                         f"decode={'sampled' if decode is not None else 'greedy'}, controls={legacy_controls}")
    if decode is not None:
        off_scope = [c for c in legacy_controls
                     if c not in SAMPLED_CONTROLS + SAMPLED_UNAFFECTED_CONTROLS]
        if off_scope:
            raise SystemExit(
                f"[served-cap] --controls {off_scope} under a sampled_decode config: "
                f"the sampled re-run covers {list(SAMPLED_CONTROLS)} (sampled) and "
                f"{list(SAMPLED_UNAFFECTED_CONTROLS)} (unaffected, run unchanged) here")
        print(f"[served-cap] SAMPLED decode in force: {_decode_block(gen)}"
              + (f"; rung {esc.source_cap} -> {esc.cap} from {esc.root} "
                 f"(floor sample k={esc.floor_k}"
                 f"{', items ' + str(esc.items_files) if esc.items_files else ''})" if esc else ""))
        unaffected = [c for c in legacy_controls if c in SAMPLED_UNAFFECTED_CONTROLS]
        if unaffected:
            print(f"[served-cap] {unaffected} run UNCHANGED under the sampled config: a "
                  "teacher-forced letter-logprob argmax (one greedy token) that no sampler key "
                  "can move (LADDER_RULES.md)")
    # under identity version 2 a
    # thinking-ON run HASHES the served header id tuples per arm, so the header is
    # measured NOW — through the live /v1/tokenize, both arms — before the store
    # is chosen; a measurement that fails refuses before any store is touched
    early_client = None
    early_header = None
    if atoms and identity_v2 and (no_switch_run or any_on):
        # either certified-render class measures the served header EARLY (before the store is chosen), through one
        # probe client, once the served model's family is confirmed to be the sampling key's — the render fact read
        # below is that family's. The two classes are exclusive: `any_on` is False under the no-switch arm.
        if family is not None:
            _check_served_family(model_id, model_revision, family=family, model_key=sampling.model_key)
        early_client = _make_client(probe_base_url, route)
    if atoms and identity_v2 and no_switch_run:
        # llama-paper-rows (P18, predicate (A″) rule 1): the NO-SWITCH class measures the ONE render's
        # header once through the live /v1/tokenize with no kwargs; it is NOT hashed (no `thinking_header_ids`
        # key: the identity's rule is "present iff any task is ON"), it is evidence
        early_header, header_error_early = measure_no_switch_header(
            # `lens=`: the lens-route render DECLARATION contract, not steering
            lens=bs._is_lens(route), tokenize=getattr(early_client, "tokenize", None), vmodel=route.virtual_model,
            served_template_hash=standup_endpoint.get(STANDUP_TEMPLATE_HASH_KEY))
        if header_error_early is not None:
            raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: the served header could not be measured "
                             f"for the no-switch class — {header_error_early}")
    if atoms and identity_v2 and any_on:
        header_arms_early, header_error_early = measure_thinking_header_by_arm(
            # `lens=`: the lens-route thinking/prefill DECLARATION contract, not steering
            thinking_by_task, lens=bs._is_lens(route),
            tok=_load_served_tok(model_id, model_revision),
            tokenize=getattr(early_client, "tokenize", None), vmodel=route.virtual_model,
            served_template_hash=standup_endpoint.get(STANDUP_TEMPLATE_HASH_KEY),
            allow_out_of_support=bool(args.allow_out_of_support_thinking),
            measure_both_arms=True, header_located=header_located,
            # the bound record's attested thinking class — the ON-pinned rule keys on it (never argv)
            standup_thinking_mode=standup_endpoint.get(tc.STANDUP_THINKING_MODE_KEY))
        if header_error_early is not None or not header_arms_early:
            raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: the served header could not be measured "
                             f"for the version-2 identity — {header_error_early}")
        early_header = header_arms_early
        tinycap_identity[tc.IDENTITY_KEY_THINKING_HEADER_IDS] = {
            arm: [int(i) for i in ids] for arm, ids in sorted(header_arms_early["header_ids_by_arm"].items())}
    if tinycap_identity is not None:
        # the exact-keys assertion, at the point the identity is COMPLETE (the
        # version-2 thinking-ON header ids were the last key in): ONE derivation
        # for the driver's own check and the launcher's store-trust check
        expected_fields = set(tc.tinycap_identity_fields(
            out_of_support_thinking=oos_run, off_cap_override=offcap_run,
            files_math500=tc.files_math500(thinking_by_task),
            sampling_optional=sampling_opt, identity_version=identity_version,
            thinking_on=bool(identity_v2 and any_on), max_tokens_by_task=identity_v2, diagnostic=diagnostic,
            declares=declaration_expected, cap_rung=item_subset is not None,
            # the two pinned legs' keys and the ladder tuple
            files_bigcodebench_200=tc.files_bigcodebench_200(thinking_by_task),
            files_bfcl_200=tc.files_bfcl_200(thinking_by_task),
            # capeval-full-tasks: the full siblings' keys, iff filed
            files_bigcodebench_full=tc.files_bigcodebench_full(thinking_by_task),
            files_bfcl_full=tc.files_bfcl_full(thinking_by_task),
            ladder=ladder is not None,
            # the per-task executed map, iff protocol
            protocol_sampling=sampling.mode == tc.SAMPLING_PROTOCOL))
        assert set(tinycap_identity) == expected_fields, (sorted(set(tinycap_identity) ^ expected_fields))
    # E56 (3): resolved ONCE, then hashed into the store identity below AND
    # handed to run_inspect_tasks, which builds the grader with it — the
    # number that was served and the number the identity claims are one.
    grader_max_tokens = _grader_max_tokens()
    # the ENGINE this run is served by, read off
    # the standup record — the strict read above on a tinycap run; the record beside
    # the table, read leniently, on a legacy controls run (absent = lens by era: only
    # the native core writes `engine`, and it publishes the record before it serves)
    # — held to --serving-path when stated. Hashed into the identity, stamped on
    # every row and the manifest, PRESENT IFF native.
    serving_engine = _run_serving_engine(args, standup_endpoint)
    resolved = _resolved_config(cell, cfg, system, args.limit, judge_model,
                                route=route, model_id=model_id,
                                model_revision=model_revision,
                                top_logprobs=args.top_logprobs,
                                grader_max_tokens=grader_max_tokens,
                                tinycap=tinycap_identity, prov=prov,
                                serving_engine=serving_engine,
                                mmlu_full=bool(args.mmlu_full))
    chash = cs.harness_config_hash(resolved)
    # the registry identity the wave exports (None = today's store, byte for byte); the factory derives the
    # production seed it implies — the plan's literal s (the identity's seed_base), held to every base this run
    # draws under: the config's sampled seed_base and the Inspect replicate base passed here
    registry_identity = _registry_run_identity()
    store, template = _store_for(cell, cfg, out_root, ihash=ihash,
                                 chash=chash, model_id=model_id,
                                 model_revision=model_revision,
                                 serving_engine=serving_engine,
                                 registry_identity=registry_identity, resolved=resolved,
                                 inspect_seed_base=sampling.inspect_seed_base)

    # -- tinycap provenance: the HF driver's controls_provenance stamp, with
    #    the SERVED stamp (live tokenizer template hash + the route) in the
    #    serving slot; written into the manifest, the one provenance every
    #    controls_summary.json writer merges under (R3) -----------------------
    tok_ev = None
    provenance = None
    header_arms = None
    header_error = None
    # the sync proxy client, built BEFORE the pre-generation checks that need
    # it (E45 M5: the mixed-arm header is measured through its /v1/tokenize).
    # load() verifies the endpoint IS the petri proxy and serves this cell's
    # virtual model — a dead tunnel or stale table fails HERE, before any
    # request and before the manifest is written.
    client = early_client if early_client is not None else _make_client(probe_base_url, route)
    if atoms:
        from sisteer.eval.cell_controls import controls_provenance

        tok_ev = _tok_evidence(model_id, model_revision)
        provenance = controls_provenance(
            cell, cfg, intervention_hash=ihash,
            serving_identity=_served_stamp(cfg=cfg, model_id=model_id,
                                           model_revision=model_revision, route=route,
                                           tok_ev=tok_ev),
            judge_model=judge_model)
        # a store that exists must have been served under the SAME parser /
        # tool-call mode / window / S / client width (R4, N3, PW5-4) and the
        # same geometry pass (PW8) — checked before anything is skipped
        _check_manifest_serving_identity(store, standup_endpoint,
                                         concurrency=args.concurrency, pass_no=pass_no)
        # ... and at the SAME per-task arms (PW20b-D): one arm per task (U5)
        _check_manifest_thinking_map(store, thinking=thinking, by_task=thinking_by_task)
        # ... and under the SAME draw (U8 / PW-sampling): mode, the row's three
        # numbers, the per-task k and the seed rule
        _check_manifest_sampling(store, expected=_sampling_block(sampling))
        # PW20b-D: mixing arms in ONE standup is only safe while the served
        # template's assistant header does not move between them — a fitted
        # object's mask is keyed on the OFF header it was fit under. Measured
        # through the LIVE server's /v1/tokenize (E45 M5), before anything is
        # generated; a local tokenizer is the RECORDED fallback. The refusal
        # is deferred to just after evidence.json is written (below) so a
        # failed precondition is durable, exactly like the bites control.
        if early_header is not None:
            header_arms, header_error = early_header, None
        else:
            header_arms, header_error = measure_thinking_header_by_arm(
                # `lens=`: the lens-route thinking/prefill DECLARATION contract, not steering
                thinking_by_task, lens=bs._is_lens(route),
                tok=_load_served_tok(model_id, model_revision),
                tokenize=getattr(client, "tokenize", None),
                vmodel=route.virtual_model,
                served_template_hash=standup_endpoint.get(STANDUP_TEMPLATE_HASH_KEY),
                # PW-oos: MEASURE both arms and record them; refuse only if the
                # override was not named (U10 item 5)
                allow_out_of_support=bool(args.allow_out_of_support_thinking),
                header_located=header_located,
                standup_thinking_mode=standup_endpoint.get(tc.STANDUP_THINKING_MODE_KEY))

    pins = corpora.load_pins()
    specs = _judged_specs(cfg, controls, template, system, args.limit,
                          judge_model, pins, args.max_goals)
    # the standing truncation rule's config-time half: an under-sized cap is
    # refused here, before the store exists (tiers/cap_ladder; --cap-sizing)
    sizing_block = ccr.cap_sizing_gate(args, specs)
    mmlu_rows = served_tok = None
    if "mmlu" in controls:
        mmlu_rows = _head_cap(
            ccr._seeded_subset(_mmlu_rows(pins, full=bool(args.mmlu_full)), args.limit),
            args.max_goals, "mmlu")
        served_tok = _load_served_tok(model_id, model_revision)
    # a rung: every requested control's source data is loaded and validated
    # BEFORE this store exists on disk (nothing to clean up on a refusal); the
    # served tokenizer measures the first divergence token of each re-draw
    sources: Dict[str, SourceControl] = {}
    fork_tok = None
    if esc is not None:
        _validate_source_identity(esc, resolved, model_revision)
        for control, spec in specs.items():
            sources[control] = _load_source_control(esc, control, spec, gen=gen)
            src = sources[control]
            print(f"[served-cap] {control}: source {esc.src_cell_dir.name} @ cap "
                  f"{esc.source_cap}: {len(src.at_cap_ids)} at cap, "
                  f"{len(src.excluded)} excluded by the decision "
                  f"({dict(Counter(src.excluded.values()))}) -> regenerate "
                  f"{len(set(src.at_cap_ids) - set(src.excluded))} at {esc.cap}; floor sample "
                  f"{len(src.floor_ids)} (k={esc.floor_k}); "
                  f"{len(spec.rows) - len(src.regen_ids)} copied forward")
        fork_tok = _load_served_tok(model_id, model_revision)
    manifest = {
        "cell_id": cell.id,  # the TRUE id; the store key carries the token
        "harness_cell_id": route.cell_id,
        "virtual_model": route.virtual_model,
        "intervention_hash": ihash,
        "harness_config_hash": chash,
        "model_id": model_id,
        "model_revision": model_revision,
        "engine": "vllm",
        "serving_path": cm.parse_serving_identity(
            route.serving_identity).path,
        "serving_identity": route.serving_identity,
        # native-serving-default: the served ENGINE, UNHASHED, PRESENT IFF native (a
        # lens manifest is byte-identical; absent = lens by era) — beside the hashed
        # copies in resolved_config.serving and tinycap.served
        **({tc.SERVED_ENGINE_FIELD: serving_engine}
           if serving_engine == tc.SERVING_ENGINE_LORA_NATIVE else {}),
        "proxy_url": args.proxy_url,
        "routing_table": str(args.routing_table),
        "steer_positions": resolved.get("steer_positions", "all"),
        "row_constants": template,
        "resolved_config": resolved,
        "controls": list(controls),
        # decode-recording: the request sampler per legacy control, UNHASHED (a fact about the request, never store identity)
        tc.MANIFEST_KEY_REQUEST_SAMPLERS: request_samplers_block(controls, gen),
    }
    if registry_identity is not None:
        # under a registry identity, the row constants the tree's recorder does not read through row_constants
        # itself (_manifest_row_constant_lift) also ride the top level it reads — never overriding a key the
        # manifest has; without the identity the manifest is byte for byte what it was
        for k, v in _manifest_row_constant_lift(template).items():
            manifest.setdefault(k, v)
    # The served==trained verdict class, TOTAL and unhashed (top-level, like
    # `quality`): every store STATES it, "compared" and "baseline" included,
    # so a reader never infers a class from absence (the hashed copy in
    # resolved_config exists only for the non-compared classes; see
    # _resolved_config). sweep.template_check_of_manifest is the one reader.
    manifest[sweep_mod.FIT_PROVENANCE_TEMPLATE_CHECK_KEY] = _template_check_class(
        getattr(prov, "status", None) if prov is not None else None)
    # P1-2: the IMMUTABLE fit witness, with the store's first manifest and
    # compared in full on every resume (sweep.MANIFEST_KEY_FIT_WITNESS)
    manifest[sweep_mod.MANIFEST_KEY_FIT_WITNESS] = sweep_mod.fit_witness_of(prov)
    # E54, on EVERY store — not only tinycap runs: the legacy XSTest/OR-Bench
    # path stamped neither, so a store of it could not be dated, and a later
    # run under different provenance could resume onto it and rewrite its
    # evidence. producer names the tree; contract_version
    # names the contract the writer wrote under. Top-level, unhashed.
    manifest[tc.MANIFEST_KEY_PRODUCER] = _producer_block()
    manifest[tc.MANIFEST_KEY_CONTRACT_VERSION] = tc.CONTRACT_VERSION
    if row_engine_block is not None:
        # the row engine's per-cell record — which
        # standup served this cell, beside which cells, under which adapter budget and
        # compile-cache verdict — top-level and UNHASHED like `producer`: facts about the
        # CONTAINER, not about which outputs are produced, so the served identity, the
        # cell key and the resume guard (_check_manifest_serving_identity) never read it,
        # and a row-engine store keys exactly as a solo store of the same cell would.
        # Exactly contracts.ROW_ENGINE_MANIFEST_FIELDS (the reader validated the key set;
        # the store writer sorts keys on disk); PRESENT IFF --row-engine-record was given —
        # every other manifest is byte-identical (the golden in
        # tests/test_served_capability_row_engine.py)
        block = dict(row_engine_block)
        if store.manifest_path.exists():
            # landing 2, the door: `verification_pass` is the ONE field of the block the two phases
            # MOVE on the store (pending after generate, verified after the CPU verifier) and the record always
            # says None — so a RESUME carries the recorded store's state forward through every manifest write
            # of this run (the fit-provenance rewrite below lands BEFORE the legs). Without this a re-run that
            # refused mid-way, or a re-run over a verified store, reset a pending / verified store to None
            recorded_block = json.loads(store.manifest_path.read_text()).get(tc.MANIFEST_KEY_ROW_ENGINE)
            if isinstance(recorded_block, Mapping) and recorded_block.get("verification_pass") is not None:
                block["verification_pass"] = recorded_block["verification_pass"]
        manifest[tc.MANIFEST_KEY_ROW_ENGINE] = block
    if atoms:
        manifest.update(_tinycap_manifest(
            atoms=atoms, thinking=thinking, args=args, route=route,
            routes=routes, identity=tinycap_identity,
            endpoint=standup_endpoint, out_root=out_root, provenance=provenance,
            sampling=sampling, attestation=attestation, bank_view=bank_view))
        # the ephemeral ledger URL, beside proxy_url (usage_ledger is the
        # DURABLE file path — PD5), and the endpoints this run was VERIFIED
        # against (N2): canonical proxy, resolved table, ledger + its upstream
        manifest["ledger_url"] = args.ledger_url or None
        manifest[MANIFEST_KEY_BINDING] = dict(binding)
        # the proxy's recorded UPSTREAM timeout is held to this driver's own
        # request budget BEFORE the manifest is written and any store is
        # touched (a shorter one 504s and re-samples the slow draws); the
        # value rides the manifest beside ledger_url, UNHASHED (it does not
        # change which outputs are produced, only which draws survive to be
        # scored). A record WITHOUT the key (a standup before the knob) is
        # stated on stdout and stamps NO leaf — the manifest shape of every
        # store from such a standup, the two identity golden fixtures
        # included, stays byte-identical; 7283933f wrote a null leaf there
        # instead, which would have moved every manifest
        upstream_timeout_s = _proxy_upstream_timeout(standup_endpoint)
        if upstream_timeout_s is not None:
            manifest[tc.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S] = upstream_timeout_s
        if identity_v2:
            # the UNHASHED
            # experiment record — the render fact the plans were built on, the
            # matched authorisation, the support label, the served-positions fact
            # and (fitted cells) the transfer-provenance block; version 1 writes none
            manifest[tc.MANIFEST_KEY_RENDER_FACT_EVIDENCE] = (render_fact.as_dict() if render_fact else None)
            manifest[tc.MANIFEST_KEY_AUTHORISED_EXPERIMENT] = oos_experiment
            manifest["support"] = tc.support_label(fitted=lens_cell, out_of_support=oos_run)
            manifest["header_located"] = bool(header_located) if lens_cell else None
            manifest[tc.LENS_SERVED_POSITIONS_KEY] = served_positions
            manifest[tc.MANIFEST_KEY_ROUTE][tc.LENS_SERVED_POSITIONS_KEY] = served_positions
            # every run of this store, appended (the analysis' completeness gate
            # attributes extra ledger rows to recorded resumes)
            prior_attempts = []
            if store.manifest_path.exists():
                try:
                    prior_attempts = list(json.loads(store.manifest_path.read_text()).get(
                        tc.MANIFEST_KEY_RESUME_ATTEMPTS) or [])
                except (OSError, ValueError):
                    prior_attempts = []
            # FC4 (3): the entry is written BEFORE any probe (below, on a resume
            # too) with `completed: false`, and flipped to true only when the run
            # passes — a resume whose preflight refuses leaves its attempt on
            # record, so its probe rows are attributable
            attempt_entry = {"ts": _utc_now(),
                             "tree_sha": manifest[tc.MANIFEST_KEY_PRODUCER].get("tree_sha"),
                             "attempt": len(prior_attempts) + 1, "completed": False}
            if attempt_id is not None:
                # the attempt-scoped entry — its id now; kind / calls / finalised_from /
                # ended / terminal when the run ends (contracts.resolve_attempt_kinds);
                # every PRIOR entry must carry its id and this id must be fresh (the
                # wrapper mints one per run)
                if any(not isinstance(a, Mapping) or a.get(tc.ATTEMPT_ID_FIELD) is None for a in prior_attempts):
                    raise SystemExit(
                        f"[served-cap] {tc.FLAG_ATTEMPT_ID}: this store's resume_attempts carry an entry without "
                        f"{tc.ATTEMPT_ID_FIELD} — a store whose attempts predate the attempt contract cannot be "
                        "attributed; every attempt of a store is recorded with its id")
                if any(str(a.get(tc.ATTEMPT_ID_FIELD)) == attempt_id for a in prior_attempts):
                    raise SystemExit(
                        f"[served-cap] {tc.FLAG_ATTEMPT_ID} {attempt_id} is already recorded on this store — the "
                        "wrapper mints a FRESH id per driver invocation (a re-used id would merge two attempts' rows)")
                attempt_entry.update({f: None for f in tc.RESUME_ATTEMPT_SCOPED_FIELDS})
                attempt_entry.update(attempt_id=attempt_id, started=attempt_entry["ts"])
            manifest[tc.MANIFEST_KEY_RESUME_ATTEMPTS] = [*prior_attempts, attempt_entry]
            if on_cap_stmt:
                # the setter map, UNHASHED, beside the hashed executed caps
                manifest[tc.MANIFEST_KEY_ON_CAP_BY_TASK] = dict(on_cap_stmt)
            if off_cap_stmt:
                # ifeval-canonical-battery §5.2: the OFF setter map, UNHASHED, the same way
                manifest[tc.MANIFEST_KEY_OFF_CAP_BY_TASK] = dict(off_cap_stmt)
            if getattr(args, "bank_view", None):
                # P42: the bank view every row of this run is stamped with, in the route block
                manifest[tc.MANIFEST_KEY_ROUTE][tc.MANIFEST_ROUTE_KEY_BANK_VIEW] = str(args.bank_view)
            if item_subset is not None:
                # the subset's ids, path and source, UNHASHED
                manifest[tc.MANIFEST_KEY_ITEM_SUBSET] = tc.item_subset_block(item_subset, path=item_subset_path)
            if lens_cell:
                manifest[tc.MANIFEST_KEY_TRANSFER_PROVENANCE] = tc.transfer_provenance_block(
                    fit_kind=getattr(prov, "status", None), fit_template_hash=getattr(prov, "trained_template_hash", None),
                    fit_response_header_len=getattr(prov, "response_header_len", None),
                    served_positions=served_positions, served_render_arm=_render_arm(thinking, thinking_by_task),
                    served_header_ids_by_arm=(early_header or {}).get("header_ids_by_arm") or {}, family=family)
            request_evidence = _RequestEvidence(Path(out_root) / tc.REQUEST_EVIDENCE_FILENAME, attempt_id=attempt_id)
            manifest[tc.MANIFEST_KEY_REQUEST_EVIDENCE] = str(request_evidence.path)
            if attempt_id is not None:
                # what the Inspect runner needs to scope a task's
                # ledger-derived field — this attempt, and the latest PRIOR attempt
                # that generated (the request evidence on disk proves which), whose
                # complete .eval a reuse would finalise
                try:
                    prior_gen = _prior_generation_attempt(prior_attempts, request_evidence.records())
                except tc.AttemptScopeError as e:
                    raise SystemExit(f"[served-cap] {tc.FLAG_ATTEMPT_ID}: {e}") from None
                attempt_scope = tc.AttemptScope(attempt_id, prior_gen, str(request_evidence.path))
            # every probe the driver sends from here on is recorded from the client's
            # HTTP boundary (the exact bytes), labelled per kind below (FC4 (2))
            request_evidence.attach(client)
        if identity_v2 or sampling_opt or sampling.mode == tc.SAMPLING_PROTOCOL:
            # the sampling block's digest rides on every row of a version-2
            # run, and of any run whose row carries an optional term; and of
            # every `protocol` run (the block now carries by_task, so a row is
            # never read against the wrong per-task table)
            sampling_ref = tc.sampling_ref(_sampling_block(sampling))
        if getattr(args, "bank_view", None):
            # the cross-eval guard: EVERY row of this run names its object — the
            # roster pin the standup was held to (a lens cell needs
            # --expect-object-sha256; bare / prompted cells stamp a stated null),
            # the route kind read off the cell grammar, the bank view, the store's
            # producer tree and the sampling mode. Built ONCE, validated, handed
            # to the Inspect runner; a run without the flag stamps nothing
            # (byte-identical rows)
            try:
                row_identity = tc.row_identity_block(
                    cell=args.cell, view=str(args.bank_view),
                    object_artifact_sha256=(args.expect_object_sha256 or None) if lens_cell else None,
                    producer_tree_sha=manifest[tc.MANIFEST_KEY_PRODUCER].get("tree_sha"),
                    sampling_mode=sampling.mode)
            except tc.RowIdentityError as e:
                raise SystemExit(f"[served-cap] {tc.FLAG_BANK_VIEW}: {e}") from None
        # PW5-5 / E54: producer + contract_version are stamped above, on
        # every store (they used to be stamped only here — P1-B).
        if or_bench_item_set is not None:
            # U10 item 4: the hashed spec PLUS the ids its positions resolve to.
            # Unhashed and not in TINYCAP_MANIFEST_KEYS, like `quality` — the
            # identity is the positions (pure, launcher-derivable); the ids are
            # the human- and analysis-facing resolution of them, so a reader can
            # check that two cells ran the same prompts without re-deriving
            # anything. Resolving here also RE-CHECKS the pinned corpus size
            # before any GPU minute (or_bench_selected_rows refuses a drift).
            from sisteer.capeval.tiers import inspect_served as _isv
            manifest[tc.MANIFEST_KEY_OR_BENCH_ITEM_SET] = {
                **or_bench_item_set,
                "item_ids": _isv.or_bench_item_ids(or_bench_item_set, repo_root=REPO),
            }
    if args.limit:
        manifest["subset"] = {
            "seed": ccr.SUBSET_SEED,
            "limit": args.limit,
            "item_ids": {
                **({"mmlu": [r["item_id"] for r in mmlu_rows]}
                   if mmlu_rows is not None else {}),
                **{c: [r["item_id"] for r in spec.rows]
                   for c, spec in specs.items()},
            },
        }
    if mmlu_rows is not None and args.mmlu_full:
        # mmlu-full-1531: the hashed spec (contracts.mmlu_item_set — the same object
        # the identity carries) PLUS the item ids this run files, unhashed and beside
        # the OR-Bench block's precedent, so a reader can check two cells scored the
        # same items — and that the 200-item stores' ids are this list's head —
        # without re-deriving the selection. Under --limit / --max-goals the ids are
        # the rows that actually ran (the `subset` block above says why fewer).
        manifest[tc.MANIFEST_KEY_MMLU_ITEM_SET] = {
            **tc.mmlu_item_set(full=True),
            "item_ids": [r["item_id"] for r in mmlu_rows],
        }
    # P1-2: an EXISTING manifest is never replaced before validation succeeds
    # — the first write used to land here, before the served==trained check,
    # with a mapping that lacked the fit_provenance block, so a run the
    # identity check then refused had already stripped the store's fit record
    # while its class still read "compared". A fresh store is written now
    # (nothing to destroy); a resumed one is re-written only after its
    # evidence step succeeded.
    is_resume = store.manifest_path.exists()
    _refuse_resume_across_provenance(store, manifest)
    store.validate_resume(manifest)
    if not is_resume:
        store.write_cell_manifest(manifest)
    elif identity_v2:
        # FC4 (3): under version 2 a RESUME puts its attempt on record BEFORE the
        # probes (the identity was just validated equal; the entry says
        # completed: false until the run passes) — by patching ONLY
        # resume_attempts into the RECORDED manifest, never by writing this run's
        # pre-evidence manifest over it (that would drop the recorded
        # fit_provenance / evidence-step blocks the canonical validator reads on
        # the next resume); the full rewrite follows the evidence step as before.
        # Version 1 keeps the parent's late rewrite, byte for byte
        recorded = json.loads(store.manifest_path.read_text())
        store.write_cell_manifest({**recorded,
                                   tc.MANIFEST_KEY_RESUME_ATTEMPTS: manifest[tc.MANIFEST_KEY_RESUME_ATTEMPTS]})
    print(f"[served-cap] generate cell={cell.id} vmodel={route.virtual_model} "
          f"controls={controls} -> {store.cell_dir}")

    # -- positive controls FIRST: never generate on an unproven serving ---- #
    # (`client` was built above, before the header measurement that uses it)
    if atoms:
        # PW18 (N2): the LIVE proxy's identity block must agree with the
        # record the served identity was read from — the agent-loop entry
        # and, when it states one, the lens reduction mode
        _check_live_proxy_identity(client, standup_endpoint, tinycap_identity["served"],
                                   route=route, table=table)
    # fit-support transfer: a tinycap run's bites probe renders
    # under the run's arm EXPLICITLY (the kwargs on both legs; the declaration
    # too on a thinking-ON standup, where the twin is a declared lens cell) and
    # its legs' stamps are held to their §2.5.11 rows; a legacy run (no arm)
    # sends the probe byte-identical to before
    probe_render_arm, probe_render_arm_reason = _bites_probe_render_arm(
        _render_arm(thinking, thinking_by_task) if (atoms and thinking) else None,
        lens_cell=lens_cell, header_located=header_located, standup_native=standup_native,
        # F-AI: the bound record's attested regime — on a native standup attesting 'on' the proxy refuses the
        # OFF render by design, so the probe renders in the attested regime (never argv)
        standup_thinking_mode=standup_endpoint.get(tc.STANDUP_THINKING_MODE_KEY) if atoms else None)
    # each probe family is a LABELLED attempt through the client's boundary
    # (task probe/<kind>, sample <vmodel>/<n>, a fresh request id) so the ledger
    # rows it and the record reconciles by identity; a version-1 run keeps the
    # bare client (byte-identical)
    def _probe_chat(kind):
        return request_evidence.labelled(client, kind) if request_evidence is not None else client.chat_completions
    evidence, bites_error = _served_evidence(
        cell=cell, route=route, routes=routes,
        chat=_probe_chat(tc.PROBE_KIND_BITES), model_id=model_id,
        model_revision=model_revision, prov=prov, vllm_log=args.vllm_log,
        tok_ev=tok_ev, probe_render_arm=probe_render_arm,
        probe_render_arm_reason=probe_render_arm_reason,
        standup_thinking_mode=(standup_endpoint.get(tc.STANDUP_THINKING_MODE_KEY) if atoms else None),
        # an unstamped probe leg — today's proxy — refuses
        # under EITHER override (the record posture); in support it is recorded
        # and tolerated
        out_of_support=held_to_record,
        # each probe leg's expected resumed_prefill_policy label (resolved off
        # the standup record above): the steered cell's and the twin's own
        probe_stamp_policies=({usr.PROBE_LEG_STEERED: stamp_policy,
                               usr.PROBE_LEG_SCALE0: twin_stamp_policy}
                              if stamp_policy is not None else None),
        probe_header_located=header_located,
        # the liveness gate the run asked for (text unless --lens-liveness logprob), and
        # the labelled client its one-token logprob probes go through (kind probe/logprob_bites, cap 1)
        lens_liveness=getattr(args, "lens_liveness", tc.LENS_LIVENESS_TEXT),
        logprob_chat=_probe_chat(tc.PROBE_KIND_LOGPROB_BITES))
    fp = evidence.get("fit_provenance")
    if fp.get("status") is not None:
        print(f"[served-cap] fit provenance: status={fp['status']} "
              f"fit_on={fp['model_id']}@{str(fp['revision'])[:12]} "
              f"served_on={fp['served_model_id']}@"
              f"{str(fp['served_revision'])[:12]} "
              f"transfer={fp['transfer']} trained_template="
              f"{fp['template_hash']} served_template="
              f"{fp['served_template_hash']} template_check="
              f"{fp['template_check']}")
    # at LOW DOSE a correct object changes no greedy token on a neutral probe, so the
    # text-divergence probe reports "not live" when steering is live and merely small. Liveness is then established
    # NUMERICALLY (a distinct intervention + a measured effect above the engine's own repeat noise, written by
    # scripts/alphasteer_f22/screen/asscr_liveness.py and named by SISTEER_LOW_DOSE_LIVENESS_RECORD, bound to THIS cell and
    # THIS routing table). With the env var unset — every other job — this block does nothing and the gate is unchanged.
    if bites_error is not None:
        _ld_rec, _ld_why = _low_dose_liveness.consult(cell.id, args.routing_table)
        if _ld_rec is not None:
            evidence["steering_bites"]["text_divergence_is_observation"] = True
            evidence["steering_bites"]["text_divergence_refusal"] = bites_error
            evidence["steering_bites"]["low_dose_liveness"] = _low_dose_liveness.summary(_ld_rec)
            evidence["steering_bites"]["verified"] = True
            evidence["steering_bites"]["liveness_basis"] = "numeric (D87)"
            print(f"[served-cap] LOW-DOSE LIVENESS (D87): the steered route produced no greedy-text divergence from its "
                  f"scale-0 twin, and that is an OBSERVATION here, not a refusal — {_ld_why}. Recorded in evidence.json; "
                  f"generation proceeds.", flush=True)
            bites_error = None
        else:
            print(f"[served-cap] text-divergence refusal stands: {_ld_why}", flush=True)
    bites = evidence["steering_bites"]
    bites_line = ("n/a (unsteered route)" if not bites.get("applicable")
                  else f"verified={bites.get('verified')} "
                       f"n_differing={bites.get('n_differing')} "
                       f"twin={bites.get('twin')} "
                       f"gate={bites.get(tc.OOS_EVIDENCE_BITES_LIVENESS_GATE)}")
    if bites.get(tc.OOS_EVIDENCE_BITES_LIVENESS_GATE) == tc.LENS_LIVENESS_LOGPROB:
        _lp = bites.get(tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS) or {}
        _nf = _lp.get("noise_floor") or {}
        _tx = bites.get(tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE) or {}
        print(f"[served-cap] logprob liveness (kind {bites.get('object_kind')}): verified={_lp.get('verified')} "
              f"median_divergence={(_lp.get('divergence') or {}).get('steered_vs_scale0_median')} "
              f"measured_noise_floor={_nf.get('measured')} bar={_nf.get('bar')} (binding={_nf.get('binding')}) "
              f"conclusive={_lp.get('n_conclusive')}/{_lp.get('n_probes')} top_logprobs={_lp.get('top_logprobs')}; "
              f"text bites beside it: {_tx.get('n_differing')}/{_tx.get('n_probes')} differing "
              f"(text verified={_tx.get('verified')}); logprob distance is not a dose axis", flush=True)
    lp_error = None
    if mmlu_rows is not None:
        # the loglikelihood positive control is owed by every STEERED route
        # (lens or native LoRA), against its path-matched twin; the seeded
        # scoring body is identical on both engines minus the lens declaration
        # key, and the UNHASHED `mmlu_scoring` block says where the steering
        # acted at the scored position (the MMLU decision)
        route_path = cm.parse_serving_identity(route.serving_identity).path
        evidence["mmlu_scoring"] = {
            "route_path": route_path,
            "declaration_sent": bs._is_lens(route),
            "steering_at_scored_position": (
                "lens response mask" if route_path == "lens"
                else "adapter, all positions" if route_path == "lora"
                else "none (unsteered route)")}
        if bs.is_steered(route):
            lp_block, lp_error = _logprob_bites_evidence(
                client.chat_completions, route=route, routes=routes,
                rows=mmlu_rows, tok=served_tok, system=system,
                top_logprobs=args.top_logprobs)
        else:
            lp_block = {"applicable": False,
                        "reason": f"route {route.cell_id!r} carries no "
                                  "steering intervention (an unsteered route: "
                                  f"serving path {route_path!r}) — no steer to "
                                  "prove at the scored positions"}
        evidence["logprob_steering_bites"] = lp_block
    certification_error = None
    render_error = bite_error = position_error = None
    if atoms and identity_v2 and no_switch_run and header_arms is not None and header_error is None:
        # llama-paper-rows (P18): THE NO-SWITCH CERTIFICATION CLASS — the emit gate widened from `any_on` to
        # "a certified-render arm": (A″) the live render under the family's ONE render on EVERY cell;
        # on a fitted undeclared lens route (B′) verbatim and (C″) the EXPLICIT lens-ON-bite inapplicability
        # record (never an absent block); the ordinary steering-bites control ran above under the single
        # render (probe_render_arm not_available). Written first, refused by name second.
        probe = _no_switch_render_probe(_probe_chat(tc.PROBE_KIND_RENDER), route.virtual_model,
                                        system=system, sampling_row=sampling.row)
        evidence["render_certified"], render_error = _no_switch_render_certification(
            probe, family=family,
            header_ids=(header_arms.get("header_ids_by_arm") or {}).get(tc.THINKING_NOT_AVAILABLE) or [],
            measured_live=(header_arms.get("source") == HEADER_SOURCE_SERVED))
        evidence["render_probe"] = {**probe["responses"], "on_attempts": probe["on_attempts"]}
        if lens_cell and not header_located and render_error is None:
            if standup_native:
                # native-serving-default (W-C2 judgement call): predicate (B') is the
                # LENS-position attestation — the served-positions map, the resumed-
                # prefill policy, the lens salt triple — facts a native adapter has no
                # surface for (the record carries none of them). Recorded
                # not_applicable by name, never derived from the absent lens keys.
                evidence["lens_position_attested"] = _native_lens_position_record()
            else:
                try:
                    evidence["lens_position_attested"] = tc.lens_position_attested(
                        standup_endpoint, cell=args.cell, expected_sha256=(args.expect_object_sha256 or None),
                        model_key=sampling.model_key, arm=tc.THINKING_NOT_AVAILABLE)
                except tc.OutOfSupportAttestationError as e:
                    evidence["lens_position_attested"] = {"verified": False, "error": str(e)}
                    position_error = str(e)
            evidence[tc.LENS_ON_BITE_KEY] = tc.no_switch_lens_on_bite_record()
        elif lens_cell and header_located:
            render_error = render_error or (
                f"{tc.REC_THINKING_UNSERVABLE}: cell {args.cell!r} is served at HEADER-LOCATED positions "
                f"({served_positions}) under the no-switch arm — the no-switch class certifies all-position objects only "
                "(there is no declared render on a family with no switch); refusing")
    if atoms and identity_v2 and any_on and header_arms is not None \
            and header_error is None:
        # PREDICATE (A) — the live base-route render certification, on EVERY
        # thinking-ON cell: the measured header tuples against the
        # family fact, the 1-item probes' render_regime stamps, the ON probe's
        # think shape; reads NOTHING from the proxy's thinking_header attestation
        # `lens=`: the lens-route render DECLARATION contract, not steering
        # the bound standup record's attested thinking class rides into the probe and the certification — on a
        # record attesting 'on' the proxy's typed OFF-arm refusal is the OFF-arm evidence; anything else refuses by name
        standup_thinking_mode = standup_endpoint.get(tc.STANDUP_THINKING_MODE_KEY)
        probe = _render_probe(_probe_chat(tc.PROBE_KIND_RENDER), route.virtual_model, lens=bs._is_lens(route),
                              header_located=header_located, system=system, sampling_row=sampling.row,
                              family=family, standup_thinking_mode=standup_thinking_mode)
        evidence["render_certified"], render_error = _render_certification(
            probe, family=family, header_ids_by_arm=header_arms.get("header_ids_by_arm") or {},
            measured_live=(header_arms.get("source") == HEADER_SOURCE_SERVED),
            header_source_by_arm=header_arms.get(HEADER_KEY_SOURCE_BY_ARM),
            standup_thinking_mode=standup_thinking_mode)
        # the per-arm probe completions (the ON one is the CERTIFYING draw) beside every ON draw's record
        evidence["render_probe"] = {**probe["responses"], "on_attempts": probe["on_attempts"]}
        if object_on_run and not header_located and render_error is None:
            # PREDICATE (B′) — the lens-position attestation: the gate is the
            # ROUTE + RENDER fact, never the flag alone; the served-positions
            # fact, no declaration, the digest, the attested policy
            if standup_native:
                # native-serving-default (W-C2 judgement call): the lens-position
                # attestation has no native surface — recorded not_applicable (above)
                evidence["lens_position_attested"] = _native_lens_position_record()
            else:
                try:
                    evidence["lens_position_attested"] = tc.lens_position_attested(
                        standup_endpoint, cell=args.cell, expected_sha256=(args.expect_object_sha256 or None),
                        model_key=sampling.model_key)
                except tc.OutOfSupportAttestationError as e:
                    evidence["lens_position_attested"] = {"verified": False, "error": str(e)}
                    position_error = str(e)
            # PREDICATE (C) — the lens-ON bite: one undeclared ON request on
            # the lens route must carry a reasoning block and differ from the base
            # route's reply to the same prompt under the same render (engine-neutral:
            # a native adapter's ON reply must differ from the base route's too)
            base = pe.routes_by_cell(routes).get(tc.ROUTE_NONE_CELL)
            if base is None:
                bite_error = f"{tc.REC_THINKING_UNSERVABLE}: predicate (C) needs the base route ({tc.ROUTE_NONE_CELL}) in the table"
                evidence[tc.LENS_ON_BITE_KEY] = {"applicable": True, "verified": False, "error": bite_error}
            elif position_error is None:
                # class E: the same run's LOGPROB liveness verdict is offered as (C)'s liveness evidence ONLY
                # when that gate ruled the cell; the verdict itself decides whether it is conclusive. (B′) above is
                # untouched: a position / identity refusal never reaches this branch.
                evidence[tc.LENS_ON_BITE_KEY], bite_error = _lens_on_bite(
                    _probe_chat(tc.PROBE_KIND_LENS_ON_BITE), lens_vmodel=route.virtual_model,
                    base_vmodel=base.virtual_model, family=str(family),
                    logprob_liveness=(bites.get(tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS)
                                      if bites.get(tc.OOS_EVIDENCE_BITES_LIVENESS_GATE) == tc.LENS_LIVENESS_LOGPROB else None))
                if bite_error is None and evidence[tc.LENS_ON_BITE_KEY].get(tc.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY) \
                        == tc.LENS_ON_BITE_EVIDENCE_LOGPROB:
                    # the fact on the DURABLE record too (additive; a cell with a token bite writes no such key)
                    manifest[tc.MANIFEST_KEY_LENS_ON_BITE_LIVENESS] = tc.lens_on_bite_liveness_manifest_block(
                        evidence[tc.LENS_ON_BITE_KEY])
    if atoms and header_arms is not None:
        # PW20b-D: the measured per-arm assistant header of a MIXED-arm run —
        # the fact that made composing the two arms in one standup admissible
        evidence["thinking_header_by_arm"] = header_arms
        if oos_run:
            # the render fact the plans were built on (declared) beside
            # its RUNTIME certification against the live header and the proxy's
            # attested thinking_header — the certification flags read true or
            # false, never null, once the measurement itself succeeded; a
            # failed measurement leaves them null (not certified — never a pass)
            header_arms["prompt_opened_think_declared"] = oos_prompt_opened_think
            header_arms.setdefault(tc.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED, None)
            # PREDICATE (B) — the lens-boundary attestation against the proxy's
            # thinking_header — applies to a DECLARED header-located cell only;
            # an UNDECLARED all-position cell is admitted by (A) + (B′) and
            # NEVER refused on a partial lens fact (Nemotron's `on_ids: null`
            # standup) — nothing copies the fact table into an attestation, and
            # nothing here reads it
            if header_error is None and (oos_declared_run or not lens_cell):
                cert, certification_error = _certify_header(header_arms, client, family=family)
                header_arms.update(cert)
            elif header_error is None:
                header_arms["lens_boundary_attestation"] = (
                    "not applicable: an undeclared all-position cell — predicates (A) and (B') apply "
                    "(plan §4.4); the proxy's thinking_header is recorded, not asserted")
                models_payload = _models_payload_of(client)   # ONE /v1/models read for both facts
                header_arms["attested_thinking_header"] = tc.raw_thinking_header(models_payload)
                header_arms["thinking_header_error"] = tc.attested_thinking_header_error(models_payload)
    # the fit side, durable on the manifest (evidence.json is diagnostics and
    # may be absent or torn without the store being wrong); the identity
    # fields have not moved, so the re-write is additive. A RESUMED store's
    # manifest is written only now — after the served==trained validation
    # inside _served_evidence and every positive control passed, the header certification included (P1-2) — and
    # BEFORE evidence.json lands: write_cell_manifest re-validates against the
    # recorded manifest, and an adopted producer-less empty store must gain
    # its producer block before the store holds any artifact (E16/E17 B).
    manifest[sweep_mod.MANIFEST_KEY_FIT_PROVENANCE] = evidence["fit_provenance"]
    if not is_resume or (header_error is None and bites_error is None
                         and lp_error is None and certification_error is None
                         and render_error is None and position_error is None and bite_error is None):
        store.write_cell_manifest(manifest)
    _write_evidence(out_root, store, evidence)
    print(f"[served-cap] evidence written: salt isolation ok "
          f"({len(routes)} routes); steering bites: {bites_line}")
    if header_error is not None:
        # E45 M5: the measurement that refused this run is now ON DISK
        # (evidence.json thinking_header_by_arm, verified=false with the
        # reason), so the refusal is diagnosable after the container is gone —
        # the same reason the bites control records before it aborts. Raised
        # BEFORE the bites/logprob aborts because it is the older fact: it was
        # measured before any probe was sent, and nothing was generated either
        # way.
        raise SystemExit(f"[served-cap] {header_error}")
    if certification_error is not None:
        # the failed certification is on disk (all three flags false, every
        # list and attested value with its type) — now refuse the leg
        raise SystemExit(f"[served-cap] {certification_error}")
    for named in (render_error, position_error, bite_error):
        if named is not None:
            # predicates (A) / (B′) / (C): recorded in evidence.json first, refused by name second
            raise SystemExit(f"[served-cap] {named}")
    if bites_error is not None:
        if bites.get(tc.OOS_EVIDENCE_BITES_LIVENESS_GATE) == tc.LENS_LIVENESS_LOGPROB:
            _tx = bites.get(tc.OOS_EVIDENCE_BITES_TEXT_DIVERGENCE) or {}
            raise SystemExit(
                f"[served-cap] POSITIVE CONTROL FAILED for {cell.id}: the LOGPROB liveness statistic (kind "
                f"{bites.get('object_kind')}) could not certify the steered route against its scale-0 twin — the "
                "first-token divergence sits inside the MEASURED repeat-noise floor, or the statistic could not be "
                f"read; a dead or unreadable route, nothing was generated (text bites beside it: "
                f"{_tx.get('n_differing')}/{_tx.get('n_probes')} differing). {bites_error}")
        raise SystemExit(
            f"[served-cap] POSITIVE CONTROL FAILED for {cell.id}: the "
            f"steered route produced NO divergence from its scale-0 twin — "
            f"an inert serve; nothing was generated. {bites_error}")
    if lp_error is not None:
        raise SystemExit(
            f"[served-cap] LOGLIKELIHOOD POSITIVE CONTROL FAILED for "
            f"{cell.id}: the steered letter logprobs at the scored "
            "position are not distinguishable from the scale-0 twin's — "
            "every served loglikelihood would silently be the base "
            f"model's; no MMLU was scored. {lp_error}")
    # llama-paper-rows (P7 / P20): the NO-SWITCH arm takes the SAME CPU prompt-budget artifact + live /v1/tokenize pass as a
    # thinking-ON run (the artifact's arm must be `not_available`, its cap the run's)
    v2_on_budget = bool(identity_v2 and (any_on or no_switch_run) and inspect_atoms and thinking in tc.THINKING_MODES)
    # ifeval-canonical-battery follow-up (plan Appendix A; merge finding M6): a thinking-OFF
    # identity-version-2 IFEval run HANDED the artifact (the Modal container writes one per cell)
    # takes the SAME live pass — the 541 exact renders of its own cell class through /v1/tokenize,
    # max + cap held to the served window, the lengths to the artifact's — opt-in by the flag's
    # presence (the entries are sized max-prompt + cap, so a run without it proceeds as before)
    ifeval_budget_run = bool(identity_v2 and not any_on and args.oos_prompt_budget
                             and tc.IFEVAL_TASK in tinycap_identity[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK])
    if (v2_on_budget or ifeval_budget_run) and not (oos_on_run or offcap_run):
        # under identity version 2 EVERY thinking-ON Inspect run — the
        # in-support bare and prompted cells included — takes the CPU
        # prompt-budget artifact and re-tokenizes its OWN task's exact
        # prompts live
        if not args.oos_prompt_budget:
            raise SystemExit(
                f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: a thinking-ON Inspect run under "
                f"{tc.FLAG_TINYCAP_IDENTITY_VERSION} 2 needs the CPU prompt-budget artifact ({tc.FLAG_OOS_PROMPT_BUDGET} "
                f"<path to {tc.OOS_PROMPT_BUDGET_FILENAME}>) — its prompts + the cap are held to the served window "
                "before the first generation (plan §7.3)")
        try:
            oos_budget = tc.read_oos_prompt_budget(args.oos_prompt_budget)
        except tc.OutOfSupportPromptBudgetError as e:
            raise SystemExit(f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: {e}") from None
        why = tc.prompt_budget_template_mismatch(
            oos_budget, standup_endpoint.get(tc.ENDPOINT_KEY_CHAT_TEMPLATE_HASH),
            served_source=f"standup record {STANDUP_ENDPOINT_FILENAME} {tc.ENDPOINT_KEY_CHAT_TEMPLATE_HASH}")
        if why is not None:
            raise SystemExit(f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: {why}")
    if oos_on_run or offcap_run or v2_on_budget or ifeval_budget_run:
        # the LIVE prompt-budget pass — the 200 exact rendered GSM8K
        # messages through the proxy's /v1/tokenize under the run's render
        # (ON on the out-of-support legs, OFF on the comparator's: explicit
        # `enable_thinking: false`, no declaration),
        # max + cap held to the served window and the lengths to the CPU
        # artifact's, item by item — BEFORE the first generation request;
        # the record is written first, the refusal follows
        from sisteer.capeval.tiers import inspect_served as _isv_budget

        # the budget is measured on THIS run's Inspect task — the ONE task a
        # thinking-ON run of this experiment files (MATH-500's 500 prompts; the
        # parent's GSM8K paired-200 unchanged); a run with several tasks names
        # the one the artifact's `cap_task` records, and a render of another
        # task refuses
        caps_map = tinycap_identity[tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK]
        gen_tasks = list(caps_map)
        budget_task = None
        art_task = (oos_budget or {}).get("cap_task") if isinstance(oos_budget, Mapping) else None
        if isinstance(art_task, str) and art_task in caps_map:
            budget_task = art_task
        elif tc.GSM8K_STANDARD_TASK in caps_map and not identity_v2:
            budget_task = tc.GSM8K_STANDARD_TASK
        elif len(gen_tasks) == 1:
            budget_task = gen_tasks[0]
        if budget_task is None:
            raise SystemExit(
                f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: the live prompt-budget pass cannot tell "
                f"which of this run's tasks {sorted(caps_map)} the artifact measured (cap_task {art_task!r}) — one "
                "artifact per task; the budget must be measured on this run's task, never another's")
        cap = caps_map.get(budget_task)
        if cap is None:
            raise SystemExit(
                f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: the run plans no "
                f"{budget_task} cap (max_tokens_by_task = {caps_map}) — the live prompt-budget pass has no cap to add")
        task_name = _isv_budget._task_name(budget_task)
        # the artifact's OWN arm and cap must be the run's — an ON artifact on
        # the OFF comparator, or an OFF artifact written for the table's 512,
        # refuses by name before the pass; the WINDOW stays the live pass's
        # (its record is written first, then the refusal — the durability the
        # mutation tests pin)
        try:
            tc.prompt_budget_matches_run(oos_budget, arm=thinking, cap=int(cap),
                                         where=f"the {tc.FLAG_OOS_PROMPT_BUDGET} artifact")
        except tc.OutOfSupportPromptBudgetError as e:
            raise SystemExit(f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: {e}") from None
        try:
            messages_list = _isv_budget.task_messages(task_name, repo_root=REPO, system=system)
        except _isv_budget.InspectServedError as e:
            raise SystemExit(f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: {e}") from None
        live, live_error = _served_prompt_budget(
            client.tokenize, route.virtual_model, messages_list,
            arm=thinking, artifact=oos_budget,
            max_model_len=tinycap_identity["served"]["max_model_len"], cap=int(cap),
            concurrency=int(args.concurrency), expected_n=len(messages_list))
        live["task"] = budget_task
        evidence[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK] = live
        _write_evidence(out_root, store, evidence)
        # the single-pass window pre-flight over the ARTIFACT (its max + the
        # cap vs the served window) — after the live record is on disk, before
        # the refusal (the record first, as every gate)
        try:
            live["artifact_window_check"] = _isv_budget.assert_budget_fits_window(
                budget_task, oos_budget, cap=int(cap), max_model_len=tinycap_identity["served"]["max_model_len"])
        except _isv_budget.InspectServedError as e:
            # the live record is already on disk (written above); refuse by name
            raise SystemExit(f"[served-cap] {tc.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH}: {e}") from None
        _write_evidence(out_root, store, evidence)
        print(f"[served-cap] live prompt budget ({len(messages_list)} exact {task_name} prompts via "
              f"/v1/tokenize, {thinking} render): max {live[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX]} "
              f"p99 {live[tc.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99]} + cap {cap} vs window "
              f"{live['window']} -> margin {live['margin']}; artifact max {live['artifact_max']}",
              flush=True)
        if live_error is not None:
            raise SystemExit(f"[served-cap] {live_error}")
    if offcap_run and not tc.is_lens_cell(args.cell):
        # the base route's refusal of the declaration key, verified live on
        # the comparator's `none` configuration ONLY (gated on the comparator
        # flag, never the posture: the 2x2's legs are frozen) — the record is
        # written first, the refusal follows, before generation
        probe_rec, probe_error = _probe_base_route_refuses_declaration(
            client.chat_completions, route.virtual_model)
        evidence[tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED] = probe_rec
        _write_evidence(out_root, store, evidence)
        print(f"[served-cap] base-route declaration probe ({route.virtual_model}): status "
              f"{probe_rec['status']} code {probe_rec['code']!r} ok={probe_rec['ok']}", flush=True)
        if probe_error is not None:
            raise SystemExit(f"[served-cap] {probe_error}")

    # the fit side, durable on the manifest (evidence.json is diagnostics and
    # may be absent or torn without the store being wrong); the identity
    # fields have not moved, so the re-write is additive
    manifest[sweep_mod.MANIFEST_KEY_FIT_PROVENANCE] = evidence["fit_provenance"]
    store.write_cell_manifest(manifest)

    # -- generation --------------------------------------------------------- #
    gen_stats: dict = {}
    ladder_statuses: dict = {}
    # the sampled controls' durable per-draw stream (--controls-events; refused above unless a
    # sampled judged control runs): one file per store beside gen_stats.json, fsync per event,
    # numbering continued across a resume
    events_writer = (cev.ControlsEventWriter(store.cell_dir / cev.CONTROLS_EVENTS_FILENAME, cell_id=cell.id)
                     if getattr(args, "controls_events", False) else None)
    # the standing truncation rule measures every judged spool with the SERVED
    # tokenizer; the transport guard's proxy hop comes from the standup's record
    # (read leniently here — None when absent — so the base pass itself never
    # crashes on a missing record; the tinycap path's strict _standup_endpoint
    # read above is a different reader with different duties)
    ladder_tok = (fork_tok if fork_tok is not None
                  else served_tok if served_tok is not None        # the mmlu rows' load: the same served tokenizer
                  else (_load_served_tok(model_id, model_revision) if specs else None))
    ladder_endpoint = (standup_endpoint if standup_endpoint is not None
                       else (_cap_ladder_standup_endpoint(args.routing_table) if specs else None))
    # tinycap traffic goes through the usage ledger when one is given (P36);
    # the legacy legs stay on --proxy-url
    tinycap_url = args.ledger_url or args.proxy_url

    async def _run():
        async with _proxy_poster(args.proxy_url) as post:
            if mmlu_rows is not None:
                gen_stats["mmlu"] = await _run_mmlu_served(
                    post, route=route, rows=mmlu_rows, tok=served_tok,
                    store=store, template=template, system=system,
                    concurrency=args.concurrency,
                    top_logprobs=args.top_logprobs)
                # mmlu-full-1531: WHICH item set this store scored, stated on the store
                # (both widths, so a reader never infers the width from `n` alone)
                gen_stats["mmlu"]["item_set"] = tc.mmlu_item_set(full=bool(args.mmlu_full))
                if decode is not None:
                    # LADDER_RULES.md: the sampler cannot move a teacher-forced
                    # one-token argmax; say so ON the store (no `decode` key, so
                    # at_cap_controls never reads mmlu as a sampled control)
                    gen_stats["mmlu"]["sampled_decode_applies"] = False
                    gen_stats["mmlu"]["sampled_decode_note"] = (
                        "teacher-forced letter-logprob argmax over top_logprobs: one greedy "
                        "token at temperature 0; temperature/top-k/penalties cannot move it. "
                        "Any trial-to-trial difference is server-to-server logprob "
                        "nondeterminism, not a sampled letter")
            if getattr(args, "from_materialised", None):  # the streaming ladder's controls assembly — default off
                if esc is None or not getattr(args, "results", None):
                    raise SystemExit("[served-cap] --from-materialised rides an escalation run (--controls-cap/"
                                     "--escalate-from-root) and needs --results <the results Dict snapshot>")
                post = _seedrep_rung_redraw().wrap_controls_post(
                    post, ladder_root=args.from_materialised, results=args.results, cell_id=cell.id, seed_base=int(gen["seed_base"]),
                    cap=int(esc.cap), decode=decode, specs=specs, sources=sources, trial=int(gen["trial"]),
                    log=lambda m: print(f"[served-cap] {m}", flush=True))
            for control, spec in specs.items():
                spool = ccr._spool_for(judged, store, spec, ihash=ihash,
                                       model_revision=model_revision,
                                       create=True)
                meta = GenMeta.for_spool(spool) if decode is not None else None
                # the durable per-draw stream (--controls-events): the sink is set for THIS
                # control's fan-out only — bound to its cap and its source rung — and reset
                # after, so no other leg's requests can be published under its name
                sink_token = None
                if events_writer is not None:
                    sink_token = _CONTROLS_EVENT_SINK.set(events_writer.sink(
                        control=control, cap=int(spec.gen["max_new_tokens"]),
                        escalated_from_cap=(int(esc.source_cap) if esc is not None else None)))
                try:
                    gen_stats[control] = await _run_judged_served(
                        post, route=route, control=control, spec=spec,
                        spool=spool, concurrency=args.concurrency, meta=meta,
                        source=sources.get(control), tok=fork_tok)
                finally:
                    if sink_token is not None:
                        _CONTROLS_EVENT_SINK.reset(sink_token)
                # the generated items of a judged task, told to the store's registry recorder -- one null-valued
                # item_result per generation, so the run's completion union holds the plan's count (the rows are the judge's)
                store.note_generated(spec.task, sorted(spool.entries()), str(spool.path))
                if events_writer is not None:
                    gen_stats[control][cev.GEN_STATS_EVENTS_KEY] = str(events_writer.path)
                # the standing truncation rule (tiers/cap_ladder): CHECK the
                # pass — per-item sidecar, rung verdict, loud line. The
                # transport guard's input: the driver's own client timeout and
                # the proxy's upstream timeout AS THE STANDUP RECORDED IT
                # (None when unrecorded -> the rung is refused)
                ladder_statuses[control] = _measure_judged_ladder(
                    spool=spool, spec=spec, control=control, cell_id=cell.id,
                    tok=ladder_tok,
                    hop_timeouts_s=cl.served_hop_timeouts(ladder_endpoint,
                                                          AIOHTTP_TOTAL_TIMEOUT_S))
        if inspect_atoms:
            # the entry point: the SI is injected by the provider, the grader
            # is the table's `none` route, the parser is the one the standup
            # SERVED, the ledger URL is the base
            from sisteer.capeval.tiers import inspect_served

            try:
                gen_stats.update(await inspect_served.run_inspect_tasks(
                    atoms=inspect_atoms, route=route, routes=routes, system=system,
                    store=store, template=template, thinking=thinking,
                    base_url=tinycap_url,
                    api_key=_inspect_api_key(standup_endpoint),
                    concurrency=args.concurrency, samples=sampling.samples,
                    inspect_limit=args.inspect_limit, repo_root=REPO,
                    reasoning_parser=standup_endpoint["reasoning_parser"],
                    limits=limits,
                    # PW18 (U4(b)): the two standup facts that decide whether
                    # BFCL can run on this route — refused at plan time
                    agent_loop_entry=bool(standup_endpoint[STANDUP_ENTRY_KEY]),
                    tool_calls=bool(standup_endpoint["tool_calls"]),
                    # the per-task arms this ONE standup serves, the MATH-500
                    # truncation retry and the served window the retry must fit
                    # ... and whenever the served FAMILY's no-switch policy moved an
                    # arm off the mode's (llama: off -> not_available), so the provider
                    # sends no switch and every plan / row carries the resolved arm
                    thinking_by_task=(dict(thinking_by_task)
                                      if tc.is_per_task(thinking)
                                      or any(a != thinking for a in thinking_by_task.values()) else None),
                    math500_retry_max_tokens=int(args.math500_retry_max_tokens),
                    max_model_len=tinycap_identity["served"]["max_model_len"],
                    # U8 / PW-sampling: the run's draw (the TARGET's row; the
                    # grader is pinned greedy inside) and every task's own k
                    sampling=sampling.row, samples_by_task=dict(sampling.samples_by_task),
                    # under `protocol` the row is None and every task decodes
                    # under ITS executed row (the provider selects it per
                    # request); the declared temperature rides the per-task
                    # summary as protocol_temperature
                    sampling_by_task=(dict(sampling.sampling_by_task) if sampling.sampling_by_task else None),
                    protocol_temperature_by_task=(
                        {t: float(r.protocol_temperature) for t, r in sampling.protocol_by_task.items()}
                        if sampling.protocol_by_task else None),
                    # U10 item 4: the item set the identity above hashed — the
                    # rows that run are resolved from exactly those positions
                    or_bench_item_set=or_bench_item_set,
                    # E56 (3): the cap hashed into this store's identity above
                    grader_max_tokens=grader_max_tokens,
                    # the RENDER authorisation and the declared render fact —
                    # the ONE place they enter the task kwargs (resolved above);
                    # never the record posture: run_inspect_tasks derives its own
                    # held_to_record from the two flags
                    out_of_support=oos_run,
                    prompt_opened_think=oos_prompt_opened_think,
                    # the CAP override, kept apart from the render flag along
                    # the whole path
                    off_cap_override=offcap_run,
                    # the label every response of this leg must wear (resolved
                    # above off the standup record; the base route's grader
                    # consults none)
                    stamp_policy=stamp_policy,
                    # the durable ledger file this run's manifest
                    # records (usage_ledger), so every task summary carries
                    # the ledger's count of its rejected calls; None = no ledger
                    # the family render fact, the declaration predicate, the
                    # certification requirement, the matched tuple, the sampling
                    # digest and the request-evidence sink — every default the
                    # pre-existing run
                    family=family, header_located=header_located,
                    require_certified_fact=not diagnostic, oos_experiment=oos_experiment,
                    sampling_ref=sampling_ref,
                    request_evidence_sink=(request_evidence.record_wire if request_evidence is not None else None),
                    # gsm8k-paired200-paper-objects: the bounded ON-cap setter (§3.1),
                    # a rung's item subset (§4.4) and the attempt scope (§4.5)
                    on_cap_override=(dict(on_cap_stmt) if on_cap_stmt else None),
                    item_subset=item_subset_ids, attempt_scope=attempt_scope,
                    # ifeval-canonical-battery §5.4: the draw list of a rung_subset/2 rung
                    item_subset_draws=item_subset_draws,
                    # ifeval-canonical-battery: the OFF-arm rung setter (§5.2) and the
                    # per-row object identity of a --bank-view run (§1.5, P42)
                    off_cap_setter=(dict(off_cap_stmt) if off_cap_stmt else None),
                    row_identity=row_identity,
                    # the GPU-side verification phase and the row's generation
                    # facts (None on a solo run)
                    verification_phase=verification_phase, generation=generation_phase,
                    ledger_path=(Path(out_root) / tc.LEDGER_FILENAME
                                 if args.ledger_url else None),
                    # bcb-bfcl-200 §S2b: the ladder's rung-2 tuple per task —
                    # the raised cap and the selected items in LOG vocabulary
                    # (plans_for_atoms refuses a stray or an unpaired one)
                    cap_overrides_by_task=(dict(ladder[0]) if ladder is not None else None),
                    item_selection_by_task=({t: tc.ladder_log_ids(t, b["ids"]) for t, b in ladder[1].items()}
                                            if ladder is not None else None),
                    # qwen-base-seed-replicates: the OPT-IN Inspect seed policy (None / default = byte-identical)
                    inspect_seed_policy=sampling.inspect_seed_policy, inspect_seed_base=sampling.inspect_seed_base))
            except inspect_served.InspectServedError as e:
                raise SystemExit(f"[served-cap] {e}") from None

    asyncio.run(_run())
    # the tasks this run GENERATED without scoring (their gen_stats block reads
    # status GENERATED — never an OK status); [] on every other run
    generated_tasks = _generated_tasks(gen_stats)
    if generated_tasks and verification_phase != tc.VERIFICATION_PHASE_GENERATE:
        raise SystemExit(f"[served-cap] {generated_tasks} came back {tc.REC_STATUS_GENERATED} under "
                         f"{tc.FLAG_VERIFICATION_PHASE} {verification_phase} — a coupled run files every task")
    if evalplus_atoms:
        # Lane C's entry point (R1; PC9): SYNC — its engine runs its own
        # event loop — so it runs after the async legs; generation only,
        # execution + rows are the sandbox orchestrator's. The window it is
        # handed is the one the standup served (PC18a: provenance for the
        # once-only clamp; the server's stated window governs) — the SAME
        # validated geometry value the identity and the N3 guard use
        # (PW5-3: never the record's unvalidated top-level field)
        from sisteer.capeval.tiers import evalplus_runner

        datasets = tuple(tc.EVALPLUS_DATASET_OF_CONTROL[a] for a in evalplus_atoms)
        try:
            gen_stats.update(evalplus_runner.run_served(
                base_url=tinycap_url, route=route, system=system,
                # EvalPlus takes ONE arm: its tasks' shared arm under this
                # run's map (OFF under U5's policy — a code-generation prefill)
                thinking=_leg_arm(thinking_by_task, tc.EVALPLUS_TASKS,
                                  "EvalPlus", thinking),
                concurrency=args.concurrency, cell_dir=store.cell_dir,
                datasets=datasets, limit=args.limit,
                api_key=_inspect_api_key(standup_endpoint),
                max_model_len=tinycap_identity["served"]["max_model_len"],
                reasoning_parser=standup_endpoint["reasoning_parser"], limits=limits,
                # U8 / PW-sampling: the run's draw and every EvalPlus task's k
                sampling=sampling.row, samples_by_task=dict(sampling.samples_by_task)))
        except evalplus_runner.EvalPlusSeamError as e:
            raise SystemExit(f"[served-cap] {e}") from None
    if decode is not None:
        # the CLIENT hop's timeout, recorded where the hop_timeout_guard can read it back beside the
        # endpoint's upstream_timeout_s: a transport-killed draw reads as a short completion, so the
        # store says what could have killed one
        for blk in gen_stats.values():
            if isinstance(blk, dict):
                blk["client_timeout_s"] = AIOHTTP_TOTAL_TIMEOUT_S
                blk["client_timeout_source"] = "scripts/served_capability_run.py:AIOHTTP_TOTAL_TIMEOUT_S"
    # HOW the decode was chosen (sisteer.eval.sampled_decode): the vendor-preset default, the
    # --greedy opt-in, or the config's own block — stated on every store that ran a sampled
    # control (the judged over-refusal controls); absent where the default does not apply
    if decode_resolution is not None:
        gen_stats["decode_resolution"] = decode_resolution
    (store.cell_dir / "gen_stats.json").write_text(
        json.dumps(gen_stats, indent=2, default=str))
    # the standing truncation rule's record: the judged legs measured above —
    # cap_ladder.json + the manifest block (an OUTCOME block, never identity)
    if ladder_statuses:
        ccr.write_cap_ladder(
            store, manifest, cell_id=cell.id, engine="vllm",
            cap=int(gen["max_new_tokens"]),        # gen IS gsm8k_gen or _gen_cfg(cfg) (bound above)
            statuses=ladder_statuses, sizing=sizing_block,
            extra={"climbed": esc is not None, "carry_over_bundle_sha256": None,
                   "source_cap": (int(esc.source_cap) if esc is not None else None)})
    # the ledger, read ONCE for the two post-run gates below (the named refusals,
    # the request-evidence reconciliation); [] when the run had no ledger
    ledger_rows = _jsonl_rows(Path(out_root) / tc.LEDGER_FILENAME) if (atoms and args.ledger_url) else []
    # the rows and records the post-run gates ADMIT: the whole ledger / every
    # record (byte-identical) — or, under --attempt-id, the accepted generation
    # attempt's and this attempt's own
    admitted_rows = ledger_rows
    admitted_records = request_evidence.records() if request_evidence is not None else []
    n_records = request_evidence.n if request_evidence is not None else 0
    if atoms and attempt_id is not None and request_evidence is not None:
        records = admitted_records   # every record; admission narrows it below
        try:
            attempts, accepted = tc.resolve_attempt_kinds(
                manifest.get(tc.MANIFEST_KEY_RESUME_ATTEMPTS) or [], records,
                current_attempt_id=attempt_id, ended=_utc_now())
            ev_by_id = tc.evidence_by_request_id(records)
            admitted_ids = {accepted, attempt_id}
            admitted_rows = tc.rows_of_attempts(ledger_rows, admitted_ids, evidence_by_id=ev_by_id)
            admitted_records = tc.records_of_attempts(records, admitted_ids)
            attribution = tc.attempt_attribution(ledger_rows, attempts=attempts, admitted=admitted_ids,
                                                 evidence_by_id=ev_by_id)
        except tc.AttemptScopeError as e:
            raise SystemExit(f"[served-cap] {tc.FLAG_ATTEMPT_ID}: {e}") from None
        n_records = len(admitted_records)
        manifest[tc.MANIFEST_KEY_RESUME_ATTEMPTS] = attempts
        manifest[tc.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] = accepted
        manifest[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION] = attribution
        evidence[tc.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT] = accepted
        evidence[tc.MANIFEST_KEY_ATTEMPT_ATTRIBUTION] = attribution
        cur = attempts[-1]
        print(f"[served-cap] attempt {attempt_id}: kind {cur['kind']} ({cur['generation_calls']} task calls"
              + (f", finalised from {cur['finalised_from_attempt']}" if cur["finalised_from_attempt"] else "")
              + f"); accepted generation attempt {accepted}; admitted {len(admitted_rows)} of {len(ledger_rows)} "
              f"ledger rows; attributed to other attempts: "
              f"{ {k: v['n_rows'] for k, v in attribution.items()} }", flush=True)
    if atoms and args.ledger_url:
        # the NAMED proxy refusals off the ledger's `error_code` column — any
        # non-zero count refuses the cell BY NAME (contained refusals, never
        # silent corruption); the counts ride the evidence and the manifest
        # either way
        refusals = tc.ledger_thinking_refusals(admitted_rows)
        evidence["ledger_thinking_refusals"] = refusals
        manifest["ledger_thinking_refusals"] = refusals
        _write_evidence(out_root, store, evidence)
        if refusals:
            store.write_cell_manifest(manifest)
            raise SystemExit(f"[served-cap] {tc.REC_THINKING_UNSERVABLE}: the usage ledger records NAMED thinking "
                             f"refusals on this cell — {refusals} (error_code; plan §4.3 item 9) — the cell is "
                             "refused by name, never read around")
    if atoms and request_evidence is not None and ((object_on_run and not header_located)
                                                   or (no_switch_run and lens_cell and not header_located)):
        # (B′) rule 3: every record shows NO declaration and the ON render —
        # gated on the ROUTE + RENDER fact, never the flag alone — and on a
        # fitted NO-SWITCH cell the same inventory reconciliation under the
        # class's rule — no declaration and NO switch on any call
        # (contracts.request_evidence_no_switch_absent) — and the evidence
        # inventory IS the ledger's attempt inventory: every ledger attempt has
        # its record and every record its attempt, by request_id, agreeing on
        # the wire digest and the labels; every filed item has a recorded
        # attempt. Missing, duplicate-without-attempt or mismatched records
        # refuse by name. The inventory is the store's ITEMS
        # (`filed_item_pairs`): the runner also files two per-task aggregate
        # rows under `AGG_ITEM_ID`, which correspond to no request — counted as
        # filed items they refused every complete leg
        filed = tc.filed_item_pairs(store.completed_keys())
        # llama-paper-rows (P17 / P18): the NO-SWITCH class reads the evidence through `request_evidence_no_switch_absent`
        # (no switch on any task call) — `expected_enable_thinking=None` is that class's spelling on the shared gate
        why = tc.request_evidence_reconciliation(admitted_records, admitted_rows, filed=filed,
                                                 expected_enable_thinking=(None if no_switch_run else True))
        evidence["request_evidence_gate"] = {"n_records": n_records, "n_ledger_rows": len(admitted_rows),
                                             "n_filed_items": len(filed), "problem": why}
        _write_evidence(out_root, store, evidence)
        if why is not None:
            raise SystemExit(f"[served-cap] {tc.REC_OOS_DECLARATION_MISMATCH}: request evidence — {why}")
    ifeval_task = tc.inspect_task(tc.IFEVAL_TASK_NAME)
    if atoms and request_evidence is not None and ifeval_task in thinking_by_task and sampling.row is not None:
        # on EVERY IFEval cell — bare, prompted or fitted — the version-3
        # request evidence must show, per TARGET task call, the run's frozen
        # sampling row ON THE WIRE (Qwen's presence_penalty 1.5 included: the
        # term no server default supplies), the IFEval seed rule's value for
        # its (item, trial), the OFF render, and the same request-id / digest
        # inventory as the ledger. A version-2 record is "not recorded", never
        # a pass. Refused by name here so a wire defect is caught in the
        # container, not at analysis.
        from sisteer.capeval.tiers import inspect_served as _isv_seed
        filed = tc.filed_item_pairs(store.completed_keys())
        ifeval_arm = thinking_by_task[ifeval_task]
        # under the no-switch arm the gate requires the ABSENCE of the switch on
        # every task call (None); under a switch arm its value (True / False)
        expected_switch = tc.enable_thinking(ifeval_arm) if tc.sends_thinking_kwargs(ifeval_arm) else None
        why = tc.request_evidence_reconciliation(
            admitted_records, admitted_rows, filed=filed,
            expected_enable_thinking=expected_switch,
            expected_sampling=sampling.row, expected_seed=_isv_seed.seed_for_task,
            target_vmodel=route.virtual_model)
        evidence["request_evidence_sampling_gate"] = {
            "n_records": len(admitted_records), "n_ledger_rows": len(admitted_rows), "n_filed_items": len(filed),
            "request_evidence_version": tc.REQUEST_EVIDENCE_VERSION, "sampling_mode": sampling.mode,
            "seed_rule": tc.seed_rules_for(thinking_by_task, **_seed_policy_kw(sampling)).get(ifeval_task), "problem": why,
            # the arm the task ran under and what the gate required of the switch on the wire
            "thinking_arm": ifeval_arm, "expected_enable_thinking": expected_switch}
        _write_evidence(out_root, store, evidence)
        if why is not None:
            raise SystemExit(f"[served-cap] {tc.REC_OOS_DECLARATION_MISMATCH}: IFEval request evidence — {why}")
    if atoms:
        # U5-5: the per-CELL unparseable / at-cap roll-up — printed (format
        # drift must be VISIBLE) and written into the manifest beside the run
        # it describes. Written AFTER generation because it is an outcome, not
        # an input: it is not identity and never selects a store, so the
        # re-write only adds this block (validate_resume re-checks the identity
        # fields, which have not moved).
        quality = _cell_quality(gen_stats)
        _print_cell_quality(quality)
        manifest[tc.MANIFEST_KEY_QUALITY] = quality
        if generated_tasks:
            # the row-engine block's `verification_pass` reads PENDING naming
            # the generated tasks — the ONE unhashed field the GPU phase moves;
            # the CPU verifier rewrites it `verified`
            block = dict(manifest[tc.MANIFEST_KEY_ROW_ENGINE])
            block["verification_pass"] = _CV.pending_block(generated_tasks)
            manifest[tc.MANIFEST_KEY_ROW_ENGINE] = tc.validate_row_engine_record(
                block, where=f"{tc.FLAG_VERIFICATION_PHASE} {tc.VERIFICATION_PHASE_GENERATE} pending stamp")
        attempts = manifest.get(tc.MANIFEST_KEY_RESUME_ATTEMPTS)
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], Mapping) \
                and "completed" in attempts[-1]:
            # FC4 (3): this run passed every gate above — its attempt is complete
            attempts[-1] = {**attempts[-1], "completed": True}
        store.write_cell_manifest(manifest)
    if atoms:
        # controls_summary.json (R3): ONE writer, provenance-checked; only a
        # COMPLETE run has done markers to merge — a limited run writes nothing
        if limits:
            print(f"[served-cap] limited run ({limits}): partial markers only — no "
                  "done markers, no controls_summary.json")
        elif item_subset is not None:
            # a RUNG store's population is its subset, whole — its done marker and
            # outcome record are at that size — but the two-SI `gsm8k` SLOT's
            # acceptance rule (contracts.GSM8K_SLOT_ACCEPTANCE) is the full paired
            # population, so a rung store fills no slot: its reading is MERGED
            # analysis-side with its source's (per item the rung draw where
            # redrawn, else the source's)
            print(f"[served-cap] rung store ({tc.FLAG_ITEM_SUBSET}, {len(item_subset_ids or [])} items"
                  + (f", {len(item_subset_draws)} draws" if item_subset_draws is not None else "")
                  + "): done markers and outcome record at the subset's size; no controls_summary.json — a rung "
                  "fills no controls slot, its reading is merged analysis-side with the source store's (plan §4.4"
                  + ("; per (item, trial) — ifeval-canonical-battery §5.4" if item_subset_draws is not None else "")
                  + ")")
        else:
            path = completion.merge_controls_summary(
                store.cell_dir, replace_source=replace_source)
            print(f"[served-cap] controls_summary.json: {path}" if path else
                  "[served-cap] controls_summary.json: nothing complete to merge yet")
    if generated_tasks:
        # the store root's RESULT reads GENERATED (not COMPLETE) until the CPU
        # verifier appends its COMPLETE line
        line = _CV.result_line(name=str(args.cell), status=tc.REC_STATUS_GENERATED,
                               tree_sha=manifest[tc.MANIFEST_KEY_PRODUCER]["tree_sha"],
                               extra={"phase": tc.VERIFICATION_PHASE_GENERATE, "tasks": ",".join(generated_tasks)})
        _CV.append_result(out_root, line)
        print(f"[served-cap] {tc.REC_STATUS_GENERATED}: {generated_tasks} generated without scoring — awaiting the CPU "
              f"verifier (`served_capability_run.py verify`); RESULT: {line}", flush=True)
    print(f"[served-cap] generate complete for {cell.id}; XSTest/OR-Bench "
          "spools are graded by the existing judged-tier grader "
          "(judge_xstest / judge_or_bench) downstream")


def _generated_tasks(gen_stats: Mapping) -> list:
    """The row tasks whose ``gen_stats`` block reads status GENERATED (tinycap-shared-standup
    landing 2: a task generated without scoring, awaiting its CPU verifier), sorted."""
    return sorted(str(t) for t, blk in gen_stats.items()
                  if isinstance(blk, Mapping) and blk.get("status") == tc.REC_STATUS_GENERATED)


def cmd_verify(args) -> None:
    """The CPU half of BigCodeBench verification —
    score a GENERATED store's unscored ``.eval`` in owned Modal sandboxes, BIND the scores onto the
    GPU phase's log by ``(id, epoch)`` and file the task through the SAME tail the coupled path
    files through (``cpu_verify.verify_store``). Every refusal names what departs and leaves the
    pending marker in place; nothing is generated."""
    from sisteer.capeval.tiers import inspect_served
    from sisteer.capeval.tinycap import sandbox_scope as _scope

    controls = parse_controls(args.controls)
    atoms = tinycap_atoms(controls)
    tasks = sorted({t for a in atoms for t in tc.TASKS_OF_CONTROL.get(a, ())})
    sandboxed = [t for t in tasks if inspect_served._task_name(t) in inspect_served.SANDBOXED_TASK_NAMES]
    if len(tasks) != 1 or sandboxed != tasks:
        raise SystemExit(f"[served-cap] verify: --controls {args.controls!r} plans {tasks} — the verify phase files "
                         f"exactly ONE sandboxed BigCodeBench task ({list(inspect_served.SANDBOXED_TASK_NAMES)}); "
                         "BFCL and every other leg were filed by the GPU phase")
    task = tasks[0]
    fcid = str(args.function_call_id or os.environ.get(_scope.OWNER_ENV_FUNCTION_CALL_ID) or "")
    if not fcid:
        raise SystemExit("[served-cap] verify: --function-call-id (or the TINYCAP_OWNER_FUNCTION_CALL_ID environment) "
                         "is required — the verifier's sandboxes are tagged with its own FunctionCall id so the "
                         "watcher can sweep them after a hard death")
    try:
        report = _CV.verify_store(
            out_root=Path(args.out_root), cell=str(args.cell), task=task, row_dir=Path(args.row_dir),
            row_engine_record=Path(args.row_engine_record), mechanism=str(args.mechanism), function_call_id=fcid,
            run_id=(str(args.run_id) if args.run_id else None), max_sandboxes=args.max_sandboxes,
            tree=(Path(args.tree) if args.tree else None), controls=controls,
            repo_root=REPO, quality_fn=_cell_quality)
    except (_CV.CpuVerifyError, inspect_served.InspectServedError, completion.CompletionError, ValueError) as e:
        raise SystemExit(f"[served-cap] verify: {e}") from None
    if report.get("already_verified") and not report.get("tail_completed"):
        # the door: a store the CPU phase already verified — exit 0, nothing moved
        vp = report["verification_pass"]
        print(f"[served-cap] verify: already verified — {report['task']} for {args.cell} was filed by "
              f"{vp['function_call_id']} via {vp['mechanism']} ({vp['n_items']} items); nothing to do; RESULT: "
              f"{report['result_line']}", flush=True)
        return
    what = ("tail completed" if report.get("tail_completed")
            else f"complete (resumed the scoring of {report.get('scoring_function_call_id')})" if report.get("resumed_scoring")
            else "complete")
    print(f"[served-cap] verify {what} for {args.cell}: {report['task']} filed via {report['mechanism']} "
          f"({report['n_items']} items, max_sandboxes {report['max_sandboxes']}); RESULT: {report['result_line']}",
          flush=True)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _sha256_arg(value: str) -> str:
    """argparse ``type`` of :data:`contracts.FLAG_EXPECT_OBJECT_SHA256`: the ONE
    64-hex test (``contracts.is_sha256_hex``) at the parser, so a malformed
    digest is an argparse error naming the flag — never a ``ValueError`` inside
    the attestation compare that a REC could be hung on."""
    if not tc.is_sha256_hex(value):
        raise argparse.ArgumentTypeError(
            f"{tc.FLAG_EXPECT_OBJECT_SHA256} takes the 64-hex sha256 of the fitted object.pt (the "
            f"launcher's own value), got {value!r}")
    return value


def _bank_view_arg(value: str) -> str:
    """``--bank-view``: ``main`` | ``it`` or a full22 roster view (``contracts.bank_views_admitted`` —
    the seed-replicate ``s1`` / ``s2`` / ``s3`` rows sit on that view itself), refused AT PARSE TIME
    as the former ``choices=`` did, so an unknown view never reaches a standup."""
    try:
        return tc.check_bank_view_arg(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def build_parser() -> argparse.ArgumentParser:
    """The CLI (one subcommand, ``generate``); ``main`` parses through it, and
    the tests read the argparse seam off it."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="GPU node, against the live proxy")
    g.add_argument("--cap-sizing", dest="cap_sizing", default="",
                   help="the standing truncation rule's config-time gate "
                        "(tiers/cap_ladder): a cap_sizing/1 JSON from "
                        "`scripts/cap_ladder.py size`; the run's cap must be at or "
                        "above the measured recommendation for every judged control "
                        "it generates, else it refuses before any store is touched. "
                        "Every generate pass measures its spools under the rule "
                        "regardless (cap_ladder.json, manifest cap_ladder)")
    g.add_argument("--proxy-url", required=True,
                   help="the RoutingProxy endpoint (e.g. "
                        "http://127.0.0.1:PORT/v1, from the serving "
                        "launch's endpoint.json) — never a bare vLLM URL; "
                        "the proxy owns cache_salt isolation and prefix "
                        "verification")
    g.add_argument("--routing-table", required=True,
                   help="the serving launch's export_routing_table JSON "
                        "(endpoint.json's routing_table path)")
    g.add_argument("--cell", required=True,
                   help="two-SI DefenseCell id, e.g. safety_refusal__psr_kl_all")
    g.add_argument("--config", default="configs/two_si.yaml")
    g.add_argument("--defenses-dir", default="",
                   help="explicit compiled-objects dir (wins over the config "
                        "value and the TT_TWO_SI_DEFENSES_DIR env var)")
    g.add_argument("--out-root", default="out/two_si/served_controls",
                   help="served stores' root — deliberately NOT the HF "
                        "controls out-root; the layout mirrors it "
                        "(cells/<token>__controls__...) with engine=vllm")
    g.add_argument("--controls", default=",".join(SERVED_CONTROLS),
                   help="csv of legacy served controls "
                        f"({', '.join(SERVED_CONTROLS)}) and/or tinycap "
                        f"tokens ({', '.join(tc.TINYCAP_CONTROLS)}); "
                        "`gsm8k` is the STANDARD GSM8K (inspect_evals' stock "
                        "task on the paired-200 subset; `gsm8k_full` = the "
                        "in-depth full set) and needs --thinking; the "
                        "retired bespoke leg's spellings refuse (PW3)")
    g.add_argument("--limit", type=int, default=0,
                   help="per-control item cap (0 = full): the SAME seeded "
                        "random subset as cell_controls_run (seed 0), so "
                        "the served item set matches the HF store's")
    g.add_argument("--concurrency", type=int,
                   default=CAPABILITY_GEOMETRY.max_num_seqs,
                   help="max in-flight chat requests; default = the "
                        "'capability' entry's max_num_seqs in "
                        "sisteer.capeval.serving.geometry (right-size to the "
                        "server's resolved concurrency when the standup was "
                        "given a different width)")
    g.add_argument("--max-goals", type=int, default=0,
                   help="SMOKE ONLY: head-cap after the seeded subset; the "
                        "item set stops matching the HF stores")
    g.add_argument("--judge-model", default="",
                   help="binds into the resolved config / store key exactly "
                        "like the HF driver (default: the capeval judging "
                        "default) — the downstream judge phase refuses a "
                        "mismatched store")
    g.add_argument("--top-logprobs", type=int, default=MMLU_TOP_LOGPROBS,
                   help="letter-logprob read width for --controls mmlu; "
                        "must be <= the server's --max-logprobs (the "
                        "capctrl standup launches with 64) — a letter "
                        "outside the returned top-K is censored and "
                        "recorded (the pick over the present letters is "
                        "exact, never a guess); an item with NO letter "
                        "inside it is a hard per-item error")
    # -- the vendor-preset default and its opt-out (module docstring) ---------- #
    g.add_argument("--greedy", action="store_true",
                   help="argmax decoding (temperature 0) for the judged over-refusal controls "
                        f"({', '.join(SAMPLED_CONTROLS)}) — the EXPLICIT opt-in; the default is the "
                        "served model's vendor sampling preset from configs/sdfbase_sampled_decode.yaml "
                        "(the user's ruling 2026-09-11), and a config carrying its own sampled_decode "
                        "block ignores this flag. Recorded in gen_stats.json as decode_resolution. BoN "
                        "(scripts/sdfbase/full22/bon_served_run.py) is deliberately OUTSIDE this default: it "
                        "draws at the grandfathered temperature 1.0, and a vendor row would make new draws "
                        "incomparable with every existing BoN store")
    g.add_argument("--trial", type=int, default=0,
                   help="the sampled trial this run IS (its own complete run: seeds and store), for the "
                        "default-injected preset; a config carrying its own sampled_decode block names "
                        "its trial there")
    # -- the sampled re-run's controls cap ladder (module docstring) ---------- #
    g.add_argument("--controls-cap", type=int, default=None,
                   help="the controls cap LADDER (DESIGN.md; sampled_decode "
                        f"configs only): a rung of {CONTROLS_CAP_LADDER} above "
                        "the base. With --escalate-from-root (both or "
                        "neither): the items the source rung left at cap are "
                        "re-drawn at this cap with the SAME seed, every other "
                        "item is copied forward, and the result is a complete "
                        "standard store under --out-root with its own key")
    g.add_argument("--escalate-from-root", default="",
                   help="the source rung's out-root (exactly one rung below "
                        "--controls-cap) holding this cell's completed sampled "
                        "store; read only, never written")
    g.add_argument("--floor-sample", type=int, default=None,
                   help="escalation runs only: re-draw this many BELOW-cap "
                        "items of the source rung at the new cap too (the "
                        "re-draw noise floor), chosen cell-independently "
                        f"(default {FLOOR_SAMPLE_DEFAULT}; 0 disables)")
    g.add_argument("--items", action="append", default=None, metavar="FILE",
                   help="escalation runs only (LADDER_RULES.md): the at-cap "
                        "disposition file ladder_decide.py wrote for the SOURCE "
                        "rung of one (cell, control) — repeatable, one per "
                        "control. Only the items it marks escalate are re-drawn; "
                        "every other at-cap item is copied forward with "
                        f"'{META_EXCLUDED_KEY}' (loop|repetitive|boundary) on "
                        "its meta row. Without it every at-cap item is re-drawn")
    g.add_argument(cev.CONTROLS_EVENTS_FLAG, dest="controls_events", action="store_true",
                   help="sampled judged-controls runs only: publish ONE event per completed draw, "
                        "in completion order, fsync per event, to "
                        f"<cell_dir>/{cev.CONTROLS_EVENTS_FILENAME} (item_id, sampling_seed, "
                        "finish_reason, at_cap, text_sha256, completion_id; a failed request is an "
                        "event with `error`) — the durable stream a concurrently running ladder rung "
                        "tails. The store's own files are unchanged; default off writes no file")
    g.add_argument("--from-materialised", default=None, metavar="LADDER_ROOT",
                   help="escalation runs only — seedrep streaming ladder (plan §7.3, WS2-E): the CLOSED queue root "
                        ".../ladder/<leg>/<view>/<cap>; a re-draw whose item has an elected, VERIFIED materialised "
                        "draw is answered from it, every other request goes to the engine. Default off: without it "
                        "the rung is byte-identical to before")
    g.add_argument("--results", default=None, metavar="SNAPSHOT",
                   help="with --from-materialised: the results Dict snapshot (rung_redraw.py snapshot)")
    g.add_argument("--vllm-log", default="",
                   help="the vLLM server log to scrape prefix-cache facts "
                        "from into evidence.json (expected ~0%% hit rate on "
                        "a steered lens leg, by design)")
    g.add_argument(tc.REPLACE_SOURCE_FLAG, dest="replace_source", action="append",
                   default=[], metavar="ENTRY=CELL_KEY",
                   help="controls_summary.json: for ENTRY (a tinycap row task or the slot it "
                        "fills), the store whose numbers the campaign means — this store's "
                        "cell key (it takes the entry, the displaced source recorded as "
                        "superseded) or the standing store's (it keeps it) — when two "
                        "stores claim one entry at the same (or an unrecorded) geometry "
                        "pass (PW7 / E14 A; per entry since PW8 / E16 A3; repeatable, or "
                        "comma-separated); naming the STANDING store itself refreshes ENTRY from "
                        "that store's CHANGED summary, the refresh recorded with both digests "
                        "(PW16 / E24). Default: the merge refuses and names both "
                        "sources — which store the campaign means is not the driver's call")
    # -- tinycap flags (contracts.TINYCAP_DRIVER_FLAGS; legacy runs ignore) -- #
    g.add_argument(tc.FLAG_THINKING, dest="thinking",
                   choices=list(tc.LAUNCHER_THINKING_MODES), default=None,
                   help="the thinking arm — REQUIRED with any tinycap control "
                        "(never implicit, P33); travels on every request as "
                        "chat_template_kwargs.enable_thinking. `per-task` "
                        "(U5) takes each row task's own arm from "
                        "contracts.THINKING_BY_TASK and serves the whole "
                        "battery under ONE standup, mixing flags per request; "
                        "an unmapped task refuses at plan time. `not_available` "
                        "(llama-paper-rows P7) is the NO-SWITCH arm of a family whose "
                        "chat template carries no thinking switch: no chat_template_kwargs "
                        "on any request, the mode recorded as not available (never OFF); "
                        "refused by name on any other family")
    g.add_argument(tc.FLAG_SAMPLING, dest="sampling", choices=list(tc.SAMPLING_MODES),
                   default=tc.SAMPLING_DEFAULT,
                   help="how every DECODING task's completions are drawn (U8): "
                        "`card` (default, the ruling) = k seeded samples per item at "
                        "the model's frozen card temperature / top_p / top_k "
                        "(contracts.SAMPLING_BY_MODEL_KEY; a model with no verified "
                        "row REFUSES) with k = contracts.SAMPLES_BY_TASK; `greedy` = "
                        "argmax x 1 — a DIFFERENT store identity (temperature, "
                        "top_p, top_k, the per-task k and the seed rule are hashed); "
                        "`protocol` (bcb-bfcl-200-widen) = each pinned leg under its OWN "
                        "reference decoding, per task and model-independent "
                        "(contracts.PROTOCOL_SAMPLING_BY_TASK: BigCodeBench-200 greedy T 0; "
                        "BFCL-200 T 0.001 executed literally under the served stack's clamp "
                        "override, D13) with top_p 1.0 / top_k 0 sent explicitly — the "
                        "run-level scalars hash null and the per-task executed map "
                        "`sampling_by_task` is hashed; any other task under it refuses; "
                        f"needs {tc.FLAG_PROTOCOL_CHECK} and a standup launched with the override")
    g.add_argument(tc.FLAG_PROTOCOL_CHECK, dest="protocol_check", default=None,
                   help="bcb-bfcl-200-widen (plan §S2): the path of the preflight's "
                        "check_protocol_sampling record (protocol_check.json) — the RESOLVED "
                        "vLLM SamplingParams per protocol task, WITH the clamp override; the "
                        "manifest's sampling.by_task.effective is copied from it (P1-2: effective "
                        "values, never assumed). Required under --sampling protocol; refused "
                        "when the record is not a passing one or does not cover every filed leg")
    g.add_argument(tc.FLAG_INSPECT_SEED_POLICY, dest="inspect_seed_policy", choices=list(tc.INSPECT_SEED_POLICIES),
                   default=tc.INSPECT_SEED_POLICY_DEFAULT,
                   help="qwen-base-seed-replicates (plan §8 / §13): the Inspect family's per-(item, sample) seed rule — "
                        "`default` = contracts.SEED_RULE_INSPECT (every run before the policy existed, byte-identical); "
                        "`replicate` = contracts.SEED_RULE_INSPECT_REPLICATE at --inspect-seed-base, hashed into the store "
                        "identity IN PLACE of the default rule (two bases are two stores; item ids and the scaffold untouched; "
                        "IFEval keeps its own rule). Recorded in the manifest's seeds.inspect block and on every row's seed")
    g.add_argument(tc.FLAG_INSPECT_SEED_BASE, dest="inspect_seed_base", type=int, default=None,
                   help="the replicate's seed base (scripts/sdfbase/seedrep/eval_seed.py seed_base(s)); REQUIRED under "
                        "`replicate`, REFUSED under `default`")
    g.add_argument(tc.FLAG_SAMPLES, dest="samples", type=int, default=None,
                   help="MATH-500 samples per item (independent requests, "
                        "distinct seeds — never n=; Lane B). Default: the "
                        "sampling mode's (4 under card, 1 under greedy); an "
                        "explicit value wins in either mode and is recorded")
    g.add_argument(tc.FLAG_MATH500_RETRY_MAX_TOKENS, dest="math500_retry_max_tokens",
                   type=int, default=tc.MATH500_RETRY_MAX_TOKENS,
                   help="MATH-500 only (U5): re-request ONCE, at this cap, any "
                        "sample that stopped at the pass cap "
                        f"({tc.MATH500_MAX_TOKENS_DEFAULT}), with the same seed "
                        f"and sample index. Default {tc.MATH500_RETRY_MAX_TOKENS} = "
                        "no retry (the one-pass instrument); a value at or below "
                        "the pass cap is refused. Recorded in a MATH run's store "
                        "identity either way (a run that files no MATH-500 row "
                        f"records the legacy {tc.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY}); "
                        "refused when the retry cannot fit the served window")
    g.add_argument(tc.FLAG_OR_BENCH_FULL, dest="or_bench_full",
                   action="store_true", default=False,
                   help="run the OR-Bench leg over the WHOLE pinned corpus "
                        f"({tc.OR_BENCH_FULL_N} items) instead of the battery's "
                        f"frozen, seeded {tc.OR_BENCH_SUBSET_N}-item subset (the "
                        "default; the user, 2026-09-06). A DIFFERENT store "
                        "identity either way — the item set is hashed by "
                        "position — so the two never pool or resume into each "
                        "other")
    g.add_argument(tc.FLAG_MMLU_FULL, dest="mmlu_full",
                   action="store_true", default=False,
                   help="score the MMLU letter-prefill read over the WHOLE pinned pool "
                        f"({tc.MMLU_FULL_N} cais/mmlu 'all' validation rows) instead of the "
                        f"s2parity {tc.PARITY_N}-item head (the default; the user, 2026-09-21). "
                        "The same seed-0 permutation either way, so the head is the pool's "
                        "first rows item for item (nested widths) — but a DIFFERENT store "
                        "identity (resolved_config.mmlu.item_set, recorded only here), so the "
                        "two never pool or resume into each other; the manifest's "
                        f"`{tc.MANIFEST_KEY_MMLU_ITEM_SET}` block lists the ids that ran")
    g.add_argument(tc.FLAG_ALLOW_OOS_THINKING, dest="allow_out_of_support_thinking",
                   action="store_true", default=False,
                   help="PW-oos (U10 item 5, the user 2026-09-06): the EXPLICIT "
                        "override for the ONE authorised out-of-support experiment "
                        "(contracts.AUTHORISED_OOS_THINKING; any other model / cell / "
                        "controls / mode / sampling / samples / served reduction mode "
                        "refuses by name) — serve thinking-ON on a family whose fitted "
                        "objects were derived under the pinned thinking-OFF render. It "
                        "LIFTS the thinking-ON-with-no-reasoning-parser refusal and the "
                        "mixed-arm header-invariance REFUSAL; it KEEPS the header "
                        "MEASUREMENT (both arms, through the live /v1/tokenize, into "
                        "evidence.json) and every other positive control. Under it: the "
                        "thinking-ON GSM8K cap is the experiment's 8192 and the scorer "
                        "splits the think block as `implicit_open` (the family's template "
                        "opens <think> in the prompt; an unterminated block is unscored, "
                        "never a wrong answer); every task writes its all-attempted "
                        f"outcome record ({tc.TASK_OUTCOMES_DIR}/<task>.json), which a "
                        "done marker is trusted only WITH; the hashed identity gains "
                        f"{tc.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING}=true, "
                        f"{tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK} (the executed per-task cap) and "
                        f"{tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT} (the CONTRACT literal the "
                        "declared standup must attest), so such a store can never resume "
                        "into or merge with an in-support or other-cap one; the standup "
                        f"record's {tc.PROXY_IDENTITY_SERVE_DECLARATIONS} is compared by own "
                        "key (the cell's @thinking spelling AND the none__lens0 twin on the "
                        "thinking-ON lens leg), its error key must be null, "
                        f"{tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256} must equal "
                        f"{tc.FLAG_EXPECT_OBJECT_SHA256}, {tc.ENDPOINT_KEY_LENS_SALT_POLICY} "
                        f"must be {tc.LENS_SALT_POLICY_ATTESTED!r} and the cell's attested "
                        f"{tc.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS} entry `conditional`; "
                        "the live /v1/models thinking_header must certify the measured ON/OFF "
                        "header ids (boolean switch/moves) and the prompt-opened <think> render "
                        "fact; the 200 exact GSM8K prompts are re-tokenized live under the ON "
                        f"render and held to the served window minus the cap ({tc.FLAG_OOS_PROMPT_BUDGET} "
                        "is REQUIRED on the thinking-ON leg); every response's render / "
                        "cache-salt regime stamp is recorded in the task summary and held to "
                        "its contract row. Absent, nothing about this run changes")
    g.add_argument(tc.FLAG_ALLOW_OFF_CAP_OVERRIDE, dest="allow_off_cap_override",
                   action="store_true", default=False,
                   help=("offcap plan §4.1: the SECOND named override — the thinking-OFF "
                         "comparator at the thinking-ON cap (contracts.AUTHORISED_OFF_CAP_COMPARATOR; "
                         "any other model / cell / controls / mode / sampling / samples / served "
                         f"reduction mode refuses by name as {tc.REC_OFFCAP_UNAUTHORISED}); mutually "
                         f"exclusive with {tc.FLAG_ALLOW_OOS_THINKING}. " + tc.OFF_CAP_OVERRIDE_NOTE)),
    g.add_argument(tc.FLAG_EXPECT_OBJECT_SHA256, dest="expect_object_sha256", default=None,
                   type=_sha256_arg,
                   help="fit-support transfer (contract §2.5.7): the 64-hex sha256 of the "
                        "fitted object.pt the launcher held the standup to (its own "
                        "--expect-object-sha256 value); the standup record's "
                        f"{tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256} (and, when present, "
                        f"{tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL}) must equal it before "
                        "any store is touched. REQUIRED on an out-of-support lens run; a lens "
                        "run without it is unchanged (no compare)")
    g.add_argument(FLAG_SERVING_PATH, dest="serving_path", choices=list(tc.SERVING_ENGINES), default=None,
                   help="native-serving-default (plan §1.3): the serving ENGINE this run expects the standup to "
                        f"have served ({' | '.join(tc.SERVING_ENGINES)}), held to the standup record's own "
                        f"`{tc.ENDPOINT_KEY_ENGINE}` (absent = lens by era) before any served fact is read — a "
                        f"disagreement refuses as {tc.REC_SERVING_ENGINE_MISMATCH}. Unstated (the default): the "
                        "record decides. The engine is always READ off the record, never asked for: a native "
                        "record hashes serving_engine into the store identity, records reduction_mode "
                        f"{tc.REDUCTION_MODE_NOT_APPLICABLE!r}, honours its cells_skipped, and binds the live "
                        "proxy's adapter inventory to the run's route")
    g.add_argument(tc.FLAG_LENS_LIVENESS, dest="lens_liveness", choices=list(tc.LENS_LIVENESS_MODES),
                   default=tc.LENS_LIVENESS_TEXT,
                   help="math-eight-arms-extension W4: the statistic that RULES the steering-bites positive control on a "
                        f"lens cell whose object kind is in {list(tc.LOGPROB_LIVENESS_KINDS)}: `text` (the default; the "
                        "historical greedy-text divergence, byte-identical) or `logprob` — fidelity."
                        "check_logprob_steering_bites over the first generated token's common top-"
                        f"{LOGPROB_LIVENESS_TOP_LOGPROBS} logprobs on the same probe prompts, three legs (steered / "
                        "scale-0 twin / the twin again = the MEASURED noise floor), the text result reported beside it. "
                        "Every other kind keeps the text gate under either value. Recorded as "
                        f"steering_bites.{tc.OOS_EVIDENCE_BITES_LIVENESS_GATE}")
    g.add_argument(tc.FLAG_OOS_PROMPT_BUDGET, dest="oos_prompt_budget", default="",
                   help=f"fit-support transfer: the path of the {tc.OOS_PROMPT_BUDGET_FILENAME} "
                        "the launcher's preflight wrote (scripts/tinycap_oos_prompt_budget.py; the "
                        f"worker copies rep[{tc.OOS_PROMPT_BUDGET_REPORT_FIELD!r}] beside the run's usage "
                        "ledger). Its chat_template_hash must equal the standup's, and its 200 "
                        "per-item lengths the live /v1/tokenize pass under the ON render. REQUIRED "
                        "on the thinking-ON out-of-support leg; ignored otherwise")
    g.add_argument(tc.FLAG_INSPECT_LIMIT, dest="inspect_limit", type=int,
                   default=0, help="smoke cap on Inspect tasks (0 = full; Lane B)")
    # -- MATH-500 thinking-ON ------------------------------------------------ #
    g.add_argument(tc.FLAG_TINYCAP_IDENTITY_VERSION, dest="tinycap_identity_version", type=int,
                   choices=list(tc.TINYCAP_IDENTITY_VERSIONS), default=tc.TINYCAP_IDENTITY_VERSION_LEGACY,
                   help="the tinycap identity VERSION: 1 (default) = the legacy serialisation, byte for byte; "
                        "2 = adds identity_version, the EXECUTED per-task cap map max_tokens_by_task, "
                        "thinking_header_ids (the measured served header per arm, iff any task is ON) and "
                        "diagnostic (with --diagnostic) — a v1 and a v2 store never share a key; under 2 every "
                        "thinking-ON cell is certified live against its family render fact (predicate A), an "
                        "undeclared all-position object cell against the served-positions fact (B') and the "
                        "lens-ON bite (C), every thinking-ON Inspect run takes --oos-prompt-budget, and every "
                        "attempted call writes request_evidence.jsonl")
    g.add_argument(tc.FLAG_MAX_TOKENS_BY_TASK, dest="max_tokens_by_task", action="append", default=[],
                   metavar="TASK=CAP",
                   help="a STATEMENT of an Inspect task's completion cap (e.g. math=8192; repeatable or "
                        "comma-separated) that must EQUAL the cap the plan builder resolves, else the run refuses "
                        f"by name — never a setter; {tc.FLAG_TINYCAP_IDENTITY_VERSION} 2 only")
    # -- GSM8K paired-200 on the paper's objects ----------------------------- #
    g.add_argument(tc.FLAG_ON_CAP_BY_TASK, dest="on_cap_by_task", action="append", default=[],
                   metavar="TASK=CAP",
                   help="the bounded thinking-ON cap SETTER for an IN-SUPPORT Inspect task (e.g. "
                        "gsm8k_paired200=8192; repeatable or comma-separated): consulted AFTER an authorised "
                        "tuple's raised cap (which it must then EQUAL) and BEFORE the ON table / floor; only the "
                        "tasks and values of contracts.ON_CAP_BY_TASK_ALLOWED, only an ON task, only ABOVE the "
                        f"table / floor and at or below the ladder's ceiling {tc.ON_CAP_CEILING}; the executed cap "
                        f"hashes through max_tokens_by_task and {tc.FLAG_MAX_TOKENS_BY_TASK} stays a statement of "
                        f"it; {tc.FLAG_TINYCAP_IDENTITY_VERSION} 2 only")
    g.add_argument(tc.FLAG_OFF_CAP_BY_TASK, dest="off_cap_by_task", action="append", default=[],
                   metavar="TASK=CAP",
                   help="ifeval-canonical-battery (plan §5.2): the bounded thinking-OFF cap SETTER — the IFEval "
                        "ladder's rungs (ifeval=4096 | 8192; repeatable or comma-separated): consulted AFTER the "
                        "authorised comparator's raised OFF cap (mutually exclusive with it by name) and BEFORE the "
                        "OFF table; only the tasks and values of contracts.OFF_CAP_BY_TASK_ALLOWED, only an OFF "
                        f"task, only ABOVE the table and at or below the ladder's ceiling {tc.IFEVAL_CAP_CEILING}; "
                        f"the executed cap hashes through max_tokens_by_task and {tc.FLAG_MAX_TOKENS_BY_TASK} stays "
                        f"a statement of it; recorded unhashed as manifest.{tc.MANIFEST_KEY_OFF_CAP_BY_TASK}; "
                        f"{tc.FLAG_TINYCAP_IDENTITY_VERSION} 2 only")
    g.add_argument(tc.FLAG_BANK_VIEW, dest="bank_view", default=None, type=_bank_view_arg,
                   help="ifeval-canonical-battery (plan §1.5, P42 — the cross-eval comparability guard): the bank "
                        "VIEW this cell's object comes from — `main` | `it`, or a full22 roster view such as the "
                        "seed-replicate lane's `s1` (contracts.bank_views_admitted; an unknown view is refused at "
                        "parse time); when given, EVERY emitted row's extra_json carries the "
                        f"object identity {list(tc.ROW_IDENTITY_FIELDS)} + {tc.ROW_SAMPLING_MODE_FIELD} (a lens cell "
                        f"needs {tc.FLAG_EXPECT_OBJECT_SHA256}; bare / prompted cells stamp a stated null sha), and "
                        f"the manifest route block records it as {tc.MANIFEST_ROUTE_KEY_BANK_VIEW}. UNHASHED: a run "
                        "without it emits byte-identical rows")
    g.add_argument(tc.FLAG_ITEM_SUBSET, dest="item_subset", default="",
                   help=f"a RUNG run: the ladder planner's {tc.ITEM_SUBSET_SCHEMA} file naming the eligible item "
                        "ids to re-draw at the rung cap (stated with --on-cap-by-task) into THIS new root beside "
                        f"the source store, or its {tc.ITEM_SUBSET_SCHEMA_V2} twin naming (item, trial) DRAWS "
                        "(the IFEval k = 2 ladder, --off-cap-by-task): only the listed draws are re-drawn, each with "
                        "its pass-1 seed; the file's task must be planned at its cap and its source store's "
                        "identity hash and executed cap must match; the hashed identity gains cap_rung "
                        f"({', '.join(tc.CAP_RUNG_FIELDS)} — the digest over ids for v1, over draws for v2); "
                        f"completeness is the subset's size (items AND draws); {tc.FLAG_TINYCAP_IDENTITY_VERSION} 2 only")
    g.add_argument(tc.FLAG_ATTEMPT_ID, dest="attempt_id", default=None, metavar="a<k>",
                   help="the attempt-scoped producer contract: this invocation's attempt id (the wrapper mints "
                        "a<k>), stamped on every request-evidence record and this run's resume_attempts entry; "
                        "the post-run gates and the per-task rejected count are then computed over the ACCEPTED "
                        "generation attempt's rows (+ this attempt's own), every other row attributed to its "
                        "recorded attempt in the manifest's attempt_attribution block or refused by name; an "
                        "interruption after generation FINALISES from the complete retained .eval / done marker "
                        f"with zero task calls; {tc.FLAG_TINYCAP_IDENTITY_VERSION} 2 only; absent = the whole "
                        "ledger, as before")
    # -- the shared standup: the row engine record ------------------------- #
    g.add_argument(tc.FLAG_ROW_ENGINE_RECORD, dest="row_engine_record", default=None, metavar="PATH",
                   help="tinycap-shared-standup (plan §1.1.4): the row runner's per-cell record "
                        "(srv/cells/<cell>/row_engine.json) of the ONE standup a model row shares — read and "
                        f"held to contracts.ROW_ENGINE_MANIFEST_FIELDS {list(tc.ROW_ENGINE_MANIFEST_FIELDS)} "
                        "before any store is touched (a wrong key set refuses by name), then copied into the "
                        "manifest's top-level, UNHASHED `row_engine` block; the served identity, the cell key "
                        "and the resume guard never read it. Absent (every solo run): the manifest is "
                        "byte-identical")
    # -- BigCodeBench verification as a CPU phase -------------------------- #
    g.add_argument(tc.FLAG_VERIFICATION_PHASE, dest="verification_phase",
                   choices=[tc.VERIFICATION_PHASE_COUPLED, tc.VERIFICATION_PHASE_GENERATE],
                   default=tc.VERIFICATION_PHASE_COUPLED,
                   help="tinycap-shared-standup landing 2 (plan §2.2.3): BigCodeBench verification as a CPU phase. "
                        f"`{tc.VERIFICATION_PHASE_COUPLED}` (default) generates AND scores every task here, "
                        f"byte-identical to before; `{tc.VERIFICATION_PHASE_GENERATE}` builds every BigCodeBench "
                        "task generation-only (sandbox dropped after every check, scorer kept, eval score=False, no "
                        "sandbox kwargs, no D73 transport), keeps the UNSCORED .eval under the store, writes "
                        f"<row_dir>/generation/<cell>__<task>__<store>.json (the generation sidecar) and pending_<cell>__"
                        f"<task>__<store>.json (exactly {list(tc.PENDING_MARKER_FIELDS)}; <store> = the --out-root "
                        "basename), stamps manifest.row_engine."
                        f"verification_pass {{state: {tc.VERIFICATION_PASS_PENDING}, tasks}}, files BFCL as today and "
                        f"appends RESULT result={tc.REC_STATUS_GENERATED} status={tc.REC_STATUS_GENERATED}; needs "
                        f"{tc.FLAG_ROW_ENGINE_RECORD}. `{tc.VERIFICATION_PHASE_VERIFY}` is the `verify` subcommand")
    g.add_argument(tc.FLAG_DIAGNOSTIC, dest="diagnostic", action="store_true", default=False,
                   help="a DIAGNOSTIC run (plan §5.2): hashes diagnostic=true into the version-2 identity so the "
                        "store can never share a key with, resume onto, or be read by the analysis as a production "
                        "store; also admits a family whose render fact is still `candidate` (the smoke that "
                        "produces the completions G-PARSER certifies)")
    g.add_argument(tc.FLAG_CAP_OVERRIDE, dest="cap_override", default="",
                   help="bcb-bfcl-200 §S2b (the truncation ladder's rung 2): <row_task>=<cap>[,...] — "
                        f"the raised cap per fired leg ({', '.join(tc.LADDER_ROW_TASKS)} at {tc.LADDER_CAP_TO}), "
                        "above the table's; ALWAYS with --item-selection-json; the run is then LIMITED by "
                        f"exactly {dict(tc.LADDER_LIMITS)} (every task PARTIAL by contract) and hashes "
                        f"{list(tc.TINYCAP_IDENTITY_LADDER_FIELDS)}; no --limit / --inspect-limit / --max-goals beside it")
    g.add_argument(tc.FLAG_ITEM_SELECTION_JSON, dest="item_selection_json", default="",
                   help="the capped items to re-run, inline JSON {row_task: {ids, n, ids_sha256}} in the pinned "
                        "id vocabulary (the launcher composes it from ladder_selection.json); recomputed here; "
                        "ALWAYS with --cap-override")
    g.add_argument(tc.FLAG_LEDGER_URL, dest="ledger_url", default="",
                   help="the usage-ledger /v1 base: ALL tinycap traffic goes "
                        "through it (P36); the positive controls and legacy "
                        "controls stay on --proxy-url")

    # -- the CPU verify phase --------------------------------------------------- #
    v = sub.add_parser("verify", help="CPU container: score a GENERATED store's BigCodeBench log in owned "
                                      "sandboxes, bind the scores onto the GPU phase's log and file the task")
    v.add_argument("--out-root", required=True, help="the GENERATED cell store's root (the GPU phase's --out-root)")
    v.add_argument("--cell", required=True, help="the cell the store is (the manifest's cell_id)")
    v.add_argument("--controls", default=tc.CONTROL_BIGCODEBENCH_FULL,
                   help=f"exactly ONE sandboxed BigCodeBench control (default {tc.CONTROL_BIGCODEBENCH_FULL})")
    v.add_argument("--config", default="configs/two_si.yaml", help="the overlay the GPU phase ran (recorded)")
    v.add_argument("--tree", default=None, help="the staged tree this verifier runs; its TREE_SHA stamp is RECORDED as "
                                                "verification_pass.verifier_tree_sha beside the store's producer tree "
                                                "(generation_tree_sha) — never held equal, so a later landed tree verifies "
                                                "a store an earlier tree generated (tinycap-verify-rate-limit)")
    v.add_argument("--row-dir", dest="row_dir", required=True,
                   help="the row dir (the row-engine record's row_dir): generation/ holds the sidecar + the pending "
                        "marker; watch/ receives the verifier's attempt pointer and sandbox record")
    v.add_argument(tc.FLAG_ROW_ENGINE_RECORD, dest="row_engine_record", required=True, metavar="PATH",
                   help="the row runner's per-cell record the GPU phase copied into the manifest — held equal to the "
                        "store's row_engine block (verification_pass aside)")
    v.add_argument("--mechanism", choices=list(tc.CPU_VERIFY_MECHANISMS), default=None,
                   help=f"the scoring mechanism (plan §2.2.2 (i)); default {_CV.DEFAULT_MECHANISM} "
                        "until the G5 probe records another")
    v.add_argument("--max-sandboxes", dest="max_sandboxes", type=int, default=None,
                   help="live sandboxes; default min(<the GPU phase's concurrency, already S // K>, 8) (plan §2.2.3); the "
                        "harness passes the marker's value, or its --verify-max-sandboxes override (recorded on the "
                        "verified block as max_sandboxes) — Modal creates at most 5 sandboxes/s per workspace")
    v.add_argument("--function-call-id", dest="function_call_id", default=None,
                   help="THIS CPU FunctionCall's id (default: the TINYCAP_OWNER_FUNCTION_CALL_ID environment) — the "
                        "verifier's owner tuple (run_id, cell, fcid); its pointer and sandbox record wear it")
    v.add_argument("--run-id", dest="run_id", default=None,
                   help="the launcher run id — the harness passes NONE (it exports the owner env instead); default: "
                        "TINYCAP_OWNER_RUN_ID, else the GPU phase's attempt pointer under the row dir")

    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "verify" and args.mechanism is None:
        args.mechanism = _CV.DEFAULT_MECHANISM
    {"generate": cmd_generate, "verify": cmd_verify}[args.cmd](args)


# THE SAMPLED-CONTROLS SEAM (sisteer.eval.sampled_controls.DRIVER_GLOBALS): the module's bodies
# call this driver's request builder and fan-out by their names, use the sibling `cell_controls_run`
# module, and compare the escalation identity on DRIVER_VERSION. Bound ONCE, here — AFTER every
# bound object exists and BEFORE the __main__ guard, so a process that runs this file as a script
# binds exactly as an import does (a bind placed after `main()` once left the production process
# unbound while every import, and every test, saw it bound). The module refuses a rebind to
# different objects; REPO is deliberately NOT bound — the module locates the repo root from its
# own file, and this driver only checks that the two agree.
scp.bind_driver_globals(_chat_payload=_chat_payload, _fanout=_fanout, ccr=ccr,
                        DRIVER_VERSION=DRIVER_VERSION)
assert scp.REPO == REPO, (scp.REPO, REPO)

if __name__ == "__main__":
    main()
