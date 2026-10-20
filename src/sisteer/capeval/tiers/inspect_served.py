"""Inspect + inspect_evals on the served, steerable vLLM path (Lane B).

Every NAME here comes from :mod:`sisteer.capeval.tinycap.contracts`;
nothing is retyped.

WHAT THIS IS. inspect_evals owns the task definitions we want (MATH,
BFCL, HellaSwag, MMLU, GSM8K, XSTest) and Inspect owns the harness that
runs them. Neither knows anything about a steering lens. The only way a
STEERED number can be produced is through the RoutingProxy's
``/v1/chat/completions`` on the cell's virtual model (E6: a steered
number that did not go through the served path is an unsteered number),
and the proxy has rules Inspect's stock OpenAI provider breaks in four
separate ways. So the integration is a PROVIDER, not a wrapper script:

1. **One leading system message, SI first.** Inspect's ``system_message``
   solver inserts after existing system messages and the ``system_message``
   config PREPENDS a separate one, so a naive "put the SI in front"
   yields TWO leading system turns on HellaSwag / XSTest / MMLU / GSM8K.
   The proxy hashes the first ``byte_length`` bytes of
   ``messages[0].content`` against the registered prefix, so the second
   system turn is not a cosmetic problem — it is a 400, or worse, a
   prompted cell served without its SI. The provider collapses every
   LEADING system message into exactly one (SI first, then the task's own
   texts, joined by a blank line) and REFUSES anything it cannot collapse
   (a system turn after a user turn; a prompted cell whose first message
   does not start with the SI bytes).
2. **No forbidden keys, no streaming, no ``n``.** E3's five
   ``FORBIDDEN_CLIENT_KEYS`` are a 400 ``selector_smuggling``;
   ``stream: true`` is a 400; and E4 forbids ``n`` on a lens route
   outright — a fork may run UNSTEERED, which is unverified and exactly
   the silent-no-steer failure this lane refuses. avg@k is therefore k
   INDEPENDENT requests with distinct seeds, never ``num_choices``.
3. **Thinking is a per-request field.** The served template default is
   thinking-OFF (that is what the published suite is); the arm travels as
   ``chat_template_kwargs.enable_thinking`` on EVERY request, which vLLM
   0.25.1 merges over the server default. The provider sets it from its
   own mode so no task can forget it.
4. **Every call is ledgered.** The two ``X-Tinycap-*`` headers (P27) let
   the usage ledger in front of the proxy attribute a row to a task and a
   sample, including BFCL's dynamic turns, which are separate HTTP
   requests.

5. **Both channels are recorded; scoring is on ``content``** (Evals
   2026-09-03 21:33, binding). Under a composed reasoning parser
   (thinking-ON, ``--reasoning-parser gemma4``) vLLM splits the assistant
   message into ``reasoning_content`` and ``content``. inspect_ai 0.3.260
   does NOT drop the thought: ``chat_message_assistant_from_openai``
   (``model/_openai.py``) turns ``reasoning_content`` into a
   ``ContentReasoning(internal="reasoning_content")`` at index 0 of the
   assistant message's content list, with the answer as the ``ContentText``
   after it; ``ChatMessageBase.text`` and ``ModelOutput.completion`` join
   the TEXT parts only, so every scorer sees the answer alone (verified
   live, not read off the docs). The adapter therefore reads both off the
   log's ``sample.output`` (:func:`output_channels`) and files them in the
   row's ``extra_json`` as ``content`` / ``reasoning``, flags a row whose
   content is empty while reasoning is present
   (``content_empty_with_reasoning: true`` — no answer within
   ``max_tokens`` IS the measurement; it scores as-is) and counts those per
   task (``n_content_empty_with_reasoning`` in the summary). The one
   REFUSAL is a response carrying NEITHER channel (and no tool call) under
   a composed parser: the provider raises it live, which Inspect files as
   the sample's error (retried, then the eval fails — PB8's discipline;
   refusing at adapt time would deadlock resume on a ``success`` log).
6. **Every response is stamp-verified against ITS model's route** (Evals
   E10 N2, binding). The proxy stamps each completion with ``model`` +
   ``petri_intervention_hash`` + ``petri.cell_id`` +
   ``petri.serving_identity``; ``/v1/models`` listing a virtual-model name
   cannot tell two live standups (or a table one generation behind the
   proxy) apart, so the provider is built with its ROUTE (the table entry,
   target and grader each their own) and runs the ONE checker
   ``proxy_engine.assert_response_stamps`` in ``on_response`` — the
   ``OpenAICompatibleAPI`` hook that receives ``completion.model_dump()``
   before any choice is read (the openai SDK keeps the proxy's extra
   top-level fields as model extras, so the dict carries them verbatim).
   A mismatch is raised there — OUTSIDE the SDK's send, so it is never
   classed as a connection error nor retried by the client — and Inspect
   files it as the sample's error (PB8's discipline: retried, then the eval
   fails and its log is not reusable). The grader's responses are checked
   against the GRADER route.

Plus a **preflight** (P20): before a GPU minute of evaluation, ``GET
/v1/models`` must list every virtual model this configuration will use —
the target AND the grader, whose name is READ from the routing table's
``none`` entry, never typed.

WHAT RUNS. ``math`` restricted to the canonical 500 MATH-500 ids selected
INSIDE inspect_evals' own task (P2 — the stock task is MATH-lighteval's
5,000 shuffled with no seed); the two GSM8K item sets; the two pinned
bcb-bfcl-200 legs; and the two comparison tasks, of which
``or_bench_hard_1k`` (P3: not in inspect_evals 0.18.0 at all) is a thin
in-repo task under ``configs/capeval/inspect_tasks/``. The full 3,981-item
``bfcl``, Inspect's generative letter-choice ``hellaswag`` and the two
generated-letter MMLU variants were RETIRED with the rule of
2026-09-11 (``contracts.RETIRED_INSPECT_TASK_NAMES``): no plan builds them.

WHAT IS NOT HERE. Scoring rules, prompts and datasets are inspect_evals'
— we select items and route requests, we never re-implement a task. The
one exception the spec forces is the GRADER (P4): ``math``'s
model-graded equivalence defaults to the model UNDER EVALUATION and
``xstest``'s to ``openai/gpt-4o``; both are replaced by the unsteered
base route on the same proxy, and the primary MATH metric is the
judge-free ``expression_exact_match_sympy``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib.util
import json
import uuid
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

from sisteer.capeval.serving import proxy_engine as pe
from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.serving.proxy import _canonical_v1_base
from sisteer.capeval.tiers.cap_ladder import text_sha256
from sisteer.capeval.tiers.think_split import message_channels, output_channels  # used here; also re-exported
from sisteer.capeval.tiers import think_split as ts  # GSM8K-SCORER-fix: the no-think-block key
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import cpu_verify as _CV  # landing 2: the sidecar / pending-marker writer and readers
from sisteer.capeval.tinycap.cpu_verify import GenerationPhase  # landing 2: the row's generation-phase facts
from sisteer.eval import request_evidence_client as rev  # request-evidence response side

REPO_ROOT = Path(__file__).resolve().parents[4]
#: Where the thin in-repo tasks live (P3 / P8). Loaded by PATH, not by
#: package import: they are configuration next to the pins they read, not
#: library code, and inspect_evals' own tasks are imported normally.
INSPECT_TASKS_DIR = REPO_ROOT / "configs" / "capeval" / "inspect_tasks"


def _task_name(row_task: str) -> str:
    """``inspect/<name>`` -> ``<name>`` (the inspect_evals / thin-task name)."""
    return row_task[len(C.INSPECT_TASK_PREFIX):]

#: The scorer whose value becomes the row's ``acc`` when a task files
#: several. MATH ships three (P4): the model-graded equivalence is the
#: one that needs a judge, so the judge-free sympy exact match is
#: primary and the other two ride in ``extra_json["scores"]``.
PRIMARY_SCORER: Mapping[str, str] = {
    "math": "expression_exact_match_sympy",
    # bcb-bfcl-200: the stock BigCodeBench task files ONE scorer, `verify`
    # (the sandbox execution); named so the row's `acc` is that verdict by
    # decision, not by default — and asserted against the log's scorer key
    "bigcodebench_200": "verify",
    # capeval-full-tasks: the same stock task over the whole split files the same scorer
    "bigcodebench_full": "verify",
    # ifeval-canonical-battery (P5): the stock task files ONE scorer,
    # `instruction_following`, whose value is a DICT of five IFEval numbers;
    # named so the plan asserts the log's scorer key and the row's `acc` is
    # read off it by PRIMARY_VALUE_KEY — the scorer itself stays byte-untouched
    "ifeval": "instruction_following",
}
#: ifeval-canonical-battery (P5): the KEY of a DICT-valued primary score that
#: becomes the row's ``acc`` — IFEval's ``prompt_level_strict`` (the paper's
#: headline IFEval number; bool -> 0 / 1). Every other task's primary score is a
#: scalar and names no key; a dict-valued score whose task names none still
#: REFUSES (:func:`score_to_value`). The four other values ride as row facts
#: (``ROW_FACTS_OF_TASK``) and as ``scores["instruction_following/<key>"]``.
PRIMARY_VALUE_KEY: Mapping[str, str] = {"ifeval": "prompt_level_strict"}

#: ``max_tokens`` per task per arm. Inspect's task configs pin caps sized
#: for a model that does not think (``xstest`` pins 256): under thinking-ON those caps truncate the THOUGHT
#: and the arm would measure truncation rather than capability. So the
#: thinking-ON arm raises every task to a floor (PB5), and the value is
#: recorded per row.
#:
#: Under thinking-OFF a task that pins NOTHING — inspect_evals 0.18.0's
#: ``gsm8k`` and ``bfcl`` carry no ``GenerateConfig`` at all
#: — would run at vLLM's remaining-window cap, so its completion length (and
#: the usage ledger's geometry basis, which reads ``max_tokens`` off the
#: body) would be undefined. Every task is therefore pinned:
#: ``gsm8k_paired200`` (the STANDARD paired set since U4(a)) and ``gsm8k``
#: (the in-depth full set) 512 (a step-by-step answer ending ``ANSWER: N`` —
#: the 512 completion budget the retired 0-shot instrument ran at, kept so
#: the paired set compares like for like with the published numbers; PW3);
#: ``bfcl_200`` 1024 (ONE turn of tool calls — several parallel calls with
#: long string arguments fit with margin, well under the ON floor); ``math`` :data:`contracts.
#: MATH500_MAX_TOKENS_DEFAULT` (16,384) in BOTH arms — ONE pass
#: (2026-09-09, at 8,192; raised the same day to 16,384,
#: the standing truncation rule's ceiling; before it U5's 4,096 default with a
#: per-item retry at 8,192, and before U5 2,048 OFF / 8,192 ON). The retry
#: mechanism is intact but RETIRED by default (``MATH500_RETRY_MAX_TOKENS`` = 0); a
#: re-armed retry must sit above the pass cap. What replaces the retry's
#: window check is :func:`assert_pass_fits_window`: a cap-bound sample the
#: served window clamped below the cap is a false at-cap reading, refused.
#: ``xstest`` and ``or_bench_hard_1k`` pin :data:`MODEL_GRADED_MAX_TOKENS`
#: (256) in their own task config; since E56 (3) the SAME number is pinned
#: here too, so the plan, the summary payload and every row STATE the cap
#: rather than recording ``null`` and leaving a reader to find it in a task
#: file. Same value, so the request on the wire is unchanged — verified
#: against the wave's ledgers, where every OR-Bench TARGET call already
#: carried ``max_tokens_requested: 256``.
#: The cap the two model-graded refusal tasks (XSTest, OR-Bench-Hard-1K) run
#: their TARGET generation at — inspect_evals' XSTest number, which the
#: vendored OR-Bench task copies.
MODEL_GRADED_MAX_TOKENS = 256
#: ``bigcodebench_200`` 4,096 OFF — a whole function
#: body with imports (the dataset's canonical solutions run to ~850 chars at
#: p95, well under; the cap leaves room for a model that explains first) —
#: and ``bfcl_200`` 1,024 OFF, exactly ``bfcl``'s (the same one-turn tool call).
#: The ON entries (8,192 / 4,096) exist so the tables stay total; both legs
#: run OFF by policy (contracts.THINKING_BY_TASK).
MAX_TOKENS_OFF: Mapping[str, Optional[int]] = {
    "math": C.MATH500_MAX_TOKENS_DEFAULT, "gsm8k": 512, "gsm8k_paired200": 512,
    "xstest": MODEL_GRADED_MAX_TOKENS, "or_bench_hard_1k": MODEL_GRADED_MAX_TOKENS,
    "bigcodebench_200": C.BCB200_MAX_TOKENS_OFF, "bfcl_200": C.BFCL200_MAX_TOKENS_OFF,
    # capeval-full-tasks: the full siblings at the SAME caps (the cap is the eval's)
    "bigcodebench_full": C.BCB_FULL_MAX_TOKENS_OFF, "bfcl_full": C.BFCL_FULL_MAX_TOKENS_OFF,
    # IFEval: pass 1 of the 2,048 -> 4,096 -> 8,192
    # ladder; the rungs above it come through the OFF-arm cap setter
    # (contracts.FLAG_OFF_CAP_BY_TASK), never by editing this table
    C.IFEVAL_TASK_NAME: C.IFEVAL_MAX_TOKENS_OFF}
THINKING_MAX_TOKENS_FLOOR = 2048
MAX_TOKENS_ON: Mapping[str, Optional[int]] = {"math": C.MATH500_MAX_TOKENS_DEFAULT,
                                              "bigcodebench_200": 8192, "bfcl_200": 4096,
                                              "bigcodebench_full": 8192, "bfcl_full": 4096}
#: The Inspect task names whose requests carry ``tools`` (bcb-bfcl-200: the
#: pinned BFCL subset beside the full leg) — the provider-side twin of
#: ``contracts.TOOL_CALL_CONTROLS``, pinned equal by test.
TOOL_CALL_TASK_NAMES: Tuple[str, ...] = tuple(
    _task_name(t) for fam in C.TOOL_CALL_CONTROLS for t in C.TASKS_OF_CONTROL[fam])
#: The tasks that execute in a SANDBOX (bcb-bfcl-200): their plans carry the
#: sandbox facts, their rows the BCB row facts, their summaries the counts.
SANDBOXED_TASK_NAMES: Tuple[str, ...] = ("bigcodebench_200", "bigcodebench_full")
#: A plan's ``cap_source`` when the truncation ladder's ``--cap-override``
#: replaced the table's cap.
CAP_SOURCE_OVERRIDE = "cap_override"

#: The tasks whose scorer calls a MODEL — the grader is a second generation
#: per item, on the unsteered base route (P4), and is the traffic
#: :data:`GRADER_MAX_TOKENS` bounds. MATH's model-graded equivalence rides
#: beside the sympy primary scorer; XSTest and OR-Bench-Hard-1K are graded by
#: it outright.
#: MATH LEFT this tuple —
#: its plan carries ONLY the judge-free ``expression_exact_match_sympy`` (wrapped
#: think-aware, registered under its own name) and constructs no grader; a
#: MATH-only run builds no grader model at all (``run_inspect_tasks``).
MODEL_GRADED_TASKS: Tuple[str, ...] = ("xstest", "or_bench_hard_1k")
#: The GRADER's completion cap (E56 (3), the served-path owner's flag). The
#: target legs have always been capped; the GRADER was not, and that is where
#: the wave's six HTTP 504s live — all six on the ``petri__base`` grader route
#: with ``max_tokens_requested: null``, none on a target call, at 600.1-600.9 s
#: against a 600 s gateway. An unbounded grader on a 16k window decodes until
#: the window runs out, which is exactly what the timing-out calls did.
#:
#: 4,096 is read off the wave's own ledgers (13,601 OR-Bench grader calls
#: across 69 pulled stores, six model families): p50 346, p90 551, p99 1,193,
#: p99.9 ~3,606 completion tokens, and a clean gap above — the largest grader
#: that TERMINATED ran 7,785 tokens, while 28 calls sat at ~15,7xx with
#: ``finish_reason: length``, i.e. they ran the window out and never stopped.
#: 4,096 truncates 3 of those 13,601 (0.022%) and cuts the whole degenerate
#: tail. Sized against the clock as well as the distribution: at the measured
#: 1st-percentile grader decode rate (16.1 tok/s) 4,096 tokens is ~255 s, a
#: 2.4x margin under the 600 s gateway, where 8,192 would be ~510 s and buy
#: back only those 3 calls.
#:
#: WHAT 3/13,601 DOES AND DOES NOT ESTABLISH (E58 note (b)). It bounds
#: EXPOSURE — how many gradings the cap can touch — and nothing else. It is
#: NOT evidence that the cap is neutral in DIRECTION. A grader the cap
#: truncates emits no parseable ``GRADE:``, so it files inspect's unscored
#: sentinel (§PW-nanfix-1); ``n_unscored`` counts it and ``acc`` is the mean
#: over the rest, so the item is EXCLUDED from the number rather than scored
#: wrongly. Exclusions are only harmless if they are random, and these are
#: not drawn at random: the calls that run long are the ones the grader found
#: hardest to adjudicate, which plausibly correlate with unusual or borderline
#: answers. The bound on the damage is arithmetic, not empirical: excluding k
#: of n binary items moves the mean by at most k/n, so ONE excluded item on a
#: 200-item cell is at most ~0.5 pp (1/n per item). The measured rate makes
#: even one unlikely — 0.022% of 200 is ~0.04 items expected — but "unlikely
#: and small when it happens" is a different claim from "neutral", so the right
#: description of a capped store is a CHANGED INSTRUMENT with potentially
#: non-random exclusions, never a demonstrated-neutral one. That is why the cap is store identity
#: rather than a tuning knob, and why every OR-Bench summary must report
#: ``n_unscored`` beside its compliance rate: the payload already carries it,
#: and a reader comparing a capped cell with an uncapped one has to see it.
#: The alternative it is being weighed against is not "no cap": it is an
#: unbounded grader that spends ten GPU-minutes and can take the whole cell
#: down with a gateway timeout, which excludes items too and says nothing
#: about it.
#:
#: Applied to the GRADER MODEL, not at eval level, and that distinction is
#: load-bearing: inspect_ai 0.3.260's ``Model._resolve_config`` merges the
#: eval-level config into the ACTIVE model only, and for every other model
#: inherits the operational keys alone (``max_connections`` / ``max_retries`` /
#: ``timeout`` / ``cache``). The grader is never the active model, so a task's
#: own cap has never reached it and never will.
GRADER_MAX_TOKENS = 4096

#: The ``ModelOutput.stop_reason`` value that means AT THE CAP — the ONE
#: definition of "at cap" in this module and in the summaries it writes (PW20b-D).
#: inspect_ai 0.3.260 normalizes the provider's OpenAI ``finish_reason`` through
#: ``as_stop_reason``: ``"length" -> "max_tokens"`` (verified against the pinned
#: client venv; ``"model_length"`` is a CONTEXT overflow, a different failure and
#: deliberately not counted here). The alternative reading —
#: ``completion_tokens >= max_tokens`` — is NOT used: it cannot be computed for a
#: task that pins its cap inside its solver, and it cannot tell
#: a completion that ENDED at the cap from one that was CUT there. The comparison
#: tool's ``at_token_cap`` keeps that other reading under its own name, on rows
#: whose producer recorded no stop reason (W20a / PW20a-4).
STOP_REASON_AT_CAP = "max_tokens"
#: The row task the MATH-500 truncation retry applies to (U5-3): the in-depth
#: MATH-500 leg and nothing else — the retry is a property of that instrument's
#: design (avg@k over long derivations), not a general policy.
RETRY_ROW_TASK = C.inspect_task("math")
#: Sampling (U8 item 1, PW-sampling): every plan carries the run's
#: :class:`contracts.SamplingRow` — ``temperature`` / ``top_p`` / ``top_k`` —
#: set at EVAL level (an eval-level value beats a task's own config, PB5's
#: verified precedence, so inspect_evals' ``math`` task-pinned 0.5 no longer
#: governs anything: it was Inspect's default, never a decision) AND applied
#: by the provider on every body (:meth:`TinycapAPI.tinycap_config`, the ONE
#: place a request's sampling parameters are resolved; ``top_k`` travels in
#: ``extra_body`` because inspect_ai's OpenAI params do not carry it). With no
#: row the plan is :data:`contracts.GREEDY_SAMPLING` — 0.0, no nucleus — what
#: every pre-U8 decoding task sent. Recorded per task in the plan, the
#: summary and the aggregate row.
GREEDY_TEMPERATURE = C.GREEDY_TEMPERATURE   # from the registry's decode policy export (contracts reads it)
#: BFCL categories inspect_evals 0.18.0's ``bfcl()`` REFUSES without API
#: execution (its ``invalid_category_errors`` table): ``rest`` — "REST category
#: requires API execution and is not yet implemented". The plan used to pass
#: the raw V1 u V2 union, so the 2026-09-04 prompted GPU runs died at plan time
#: on exactly that ValueError. :func:`bfcl_categories` drops this set and
#: cross-checks it against inspect_evals' own ``DEFAULT_CATEGORIES`` (which
#: leaves out the same names), so a pin bump that moves the refusal table
#: refuses loudly HERE instead of inside ``bfcl()``.
BFCL_EXCLUDED_CATEGORIES: frozenset = frozenset({"rest"})

#: The per-(item, sample) seed rule (:func:`sample_seed`), for the cell
#: manifest's ``seeds`` block — the contract's :data:`contracts.SEED_RULE_INSPECT`
#: (U8 item 1: the rule is hashed identity, so it is named in ONE place).
SEED_RULE = C.SEED_RULE_INSPECT

#: Inspect's own default is retry-forever on 5xx/connection errors, which
#: turns a dead proxy into a hang; a refusal-class 400 is never re-rolled.
class InspectServedError(RuntimeError):
    """A tinycap Inspect integration refusal. Never a soft degrade: a
    task that cannot be run correctly is reported blocked, never skipped
    and never summarized from a partial item set."""


# --------------------------------------------------------------------------- #
# per-request identity: which task, which item, which avg@k sample            #
# --------------------------------------------------------------------------- #

#: The ROW task name (``inspect/math``) of the eval currently running, set
#: by :func:`run_inspect_tasks` around each ``eval_async`` call. A
#: ContextVar rather than an attribute on the provider because ONE model
#: object serves every task in a container, and because contextvars are
#: copied into the tasks Inspect spawns, so each request sees the value
#: that was set when its eval began.
_TASK: "ContextVar[Optional[str]]" = ContextVar("tinycap_task", default=None)
#: The avg@k sample index (P28). 0 for every task but MATH-500.
_SAMPLE_IDX: "ContextVar[int]" = ContextVar("tinycap_sample_idx", default=0)
#: The NO-SWITCH RESPONSE-CONVERSION POLICY's hand-off: the RAW
#: assistant content of each choice (string + sha256), captured in ``on_response`` BEFORE the
#: pinned SDK converts the response (``chat_choices_from_openai`` -> ``parse_reasoning_content``
#: lifts any complete ``<think>…</think>`` block out of the text into ``ContentReasoning``
#: client-side, with no server parser) and read back in ``generate`` AFTER it. A ContextVar,
#: never an attribute: it is set and consumed inside ONE ``generate`` call with no await
#: between, and a copied context can never see another request's capture.
_NO_SWITCH_RAW: "ContextVar[Optional[List[Dict[str, Any]]]]" = ContextVar("tinycap_no_switch_raw", default=None)
#: ``ModelOutput.metadata[<this key>]``: the per-choice lift record the policy files
#: (``contracts.ROW_LIFT_FIELDS``), read onto the row by :func:`results_from_log`
NO_SWITCH_METADATA_KEY = "tinycap_no_switch"
#: The ``(request_id, completion_id)`` of the
#: HTTP attempt whose response the SDK returned, captured by the provider's httpx RESPONSE hook
#: (``_wire_response_hook``: ``response.request.headers[X-Tinycap-Request-Id]`` + the body's ``id``)
#: and read back in ``generate`` onto ``ModelOutput.metadata[WIRE_BINDING_METADATA_KEY]``, from where
#: :func:`results_from_log` copies it onto the row (``request_id`` / ``completion_id``, riding ONLY when
#: recorded). A ContextVar like ``_NO_SWITCH_RAW``: set inside the SDK's send, consumed in the same
#: ``generate`` call; a retried attempt's hook overwrites it, so the binding is the attempt that ANSWERED.
_WIRE_BINDING: "ContextVar[Optional[Dict[str, Optional[str]]]]" = ContextVar("tinycap_wire_binding", default=None)
WIRE_BINDING_METADATA_KEY = "tinycap_wire_binding"


def sample_seed(item_id: str, sample_idx: int = 0) -> int:
    """The per-(item, sample) seed (E4) — the ONE implementation of
    :data:`SEED_RULE` (= ``contracts.SEED_RULE_INSPECT``, the hashed identity
    string; the derivation itself lives HERE, with the request builder that
    applies it, not in the contracts).

    Deterministic and RECOMPUTABLE: the adapter derives the same number
    from the log's sample id rather than trusting a value carried through
    the harness, so the ``seed`` in every row is verifiable from the row
    itself; distinct across sample indices by construction, so k samples of
    one item are k draws. Reproducible in DISTRIBUTION only — bf16 batch
    invariance means the same seed does not give the same bytes at a
    different batch composition, which is why avg@k is a mean over k
    requests and not a bit-for-bit replay."""
    if isinstance(sample_idx, bool) or not isinstance(sample_idx, int) or sample_idx < 0:
        raise ValueError(f"sample_idx must be a non-negative int, got {sample_idx!r}")
    h = hashlib.sha256(f"{item_id}\0{int(sample_idx)}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31 - 1)


def ifeval_sample_seed(item_key: str, trial: int = 0) -> int:
    """The IFEval per-(item, trial) seed — the ONE implementation of
    ``contracts.SEED_RULE_IFEVAL``: the
    static / over-refusal sampled re-run's derivation, verbatim in substance —
    ``int.from_bytes(sha256(f"{seed_base}|{trial}|{readout}|{item_key}")[:4],
    "big")`` with ``seed_base`` :data:`contracts.IFEVAL_SEED_BASE` (20260909),
    ``readout`` :data:`contracts.IFEVAL_SEED_READOUT` (the control atom,
    ``ifeval``), ``item_key`` the Inspect sample id (the vendor's own row key)
    and ``trial`` the sample index — 4 bytes big-endian, range [0, 2^32), NO
    modulus. The key names no cell, cap or model: the same (item, trial) draws
    the same seed in every lane and cell, and a rung re-draw reuses it. A pure
    delegate: the derivation AND its argument checks (a non-negative int trial, a
    non-empty str key) live on ``contracts.ifeval_sample_seed`` — the one place
    both the adapter and the merged name resolve to."""
    return C.ifeval_sample_seed(item_key, trial)


#: The run's OPT-IN Inspect seed policy — ``None`` (every run
#: before the policy existed: ``SEED_RULE_INSPECT`` through :func:`sample_seed`) or ``("replicate", base)``
#: (``contracts.inspect_replicate_seed`` at the replicate's base). Set ONCE per run by the driver
#: (:func:`set_inspect_seed_policy` before its event loop starts, so the tasks Inspect spawns inherit it)
#: or by :func:`run_inspect_tasks` from its kwargs; read by :func:`seed_for_task`, the ONE dispatch, so
#: the request builder, the log adapter and the mean row's ``seeds`` agree by construction. A
#: ContextVar like ``_TASK`` — never an environment variable, never a per-call-site table.
_INSPECT_SEED_POLICY: "ContextVar[Optional[tuple]]" = ContextVar("tinycap_inspect_seed_policy", default=None)


def set_inspect_seed_policy(policy: Optional[str] = None, seed_base: Optional[int] = None):
    """Set the run's Inspect seed policy (validated by ``contracts.check_inspect_seed_policy``) in the
    CURRENT context and return the token (``_INSPECT_SEED_POLICY.reset(token)`` restores). ``default``
    (or None) stores None, so the default run is byte-identical to before the policy existed."""
    pol, base = C.check_inspect_seed_policy(policy, seed_base)
    return _INSPECT_SEED_POLICY.set((pol, base) if pol == C.INSPECT_SEED_POLICY_REPLICATE else None)


def inspect_seed_policy() -> Optional[tuple]:
    """The active policy: ``None`` (default) or ``("replicate", base)``."""
    return _INSPECT_SEED_POLICY.get()


def seed_for_task(row_task: Optional[str], item_id: str, sample_idx: int = 0, *,
                  policy: Optional[tuple] = None) -> int:
    """The per-(item, sample) seed of ``row_task``: :func:`ifeval_sample_seed`
    for ``inspect/ifeval`` (``contracts.SEED_RULE_IFEVAL``), :func:`sample_seed`
    (``SEED_RULE_INSPECT``) for every other Inspect task and for ``None`` (a
    request outside a task) — the ONE dispatch every call site goes through
    (the request builder, the log adapter, the mean row's ``seeds``).

    qwen-base-seed-replicates: under the run's ``replicate`` policy (the context's, or an explicit
    ``policy=("replicate", base)`` for an out-of-band reader) every task but IFEval draws
    ``contracts.inspect_replicate_seed(item, idx, base)``; IFEval keeps its own rule in every case."""
    if row_task == C.IFEVAL_TASK:
        return ifeval_sample_seed(item_id, sample_idx)
    pol = policy if policy is not None else _INSPECT_SEED_POLICY.get()
    if pol is not None:
        name, base = pol
        if name != C.INSPECT_SEED_POLICY_REPLICATE:
            raise InspectServedError(f"unknown Inspect seed policy {name!r} (one of {C.INSPECT_SEED_POLICIES})")
        return C.inspect_replicate_seed(item_id, sample_idx, base)
    return sample_seed(item_id, sample_idx)


def _raw_choice_contents(response: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The no-switch policy's capture (P21): per choice of the RAW response dict
    (``completion.model_dump()``, before the SDK converts it), the assistant
    ``content`` verbatim (None when the server sent none) and its sha256 over
    the UTF-8 bytes (the empty digest for None), in choice order."""
    out: List[Dict[str, Any]] = []
    for choice in (response.get("choices") or []):
        msg = choice.get("message") if isinstance(choice, Mapping) else None
        content = msg.get("content") if isinstance(msg, Mapping) else None
        text = content if isinstance(content, str) else None
        out.append({"content": text, "sha256": text_sha256(text)})
    return out


def current_request_identity() -> Tuple[Optional[str], Optional[str], int]:
    """``(row task, item id, sample idx)`` for the request being built.

    The item id is read from Inspect's OWN active-sample record — the
    authoritative source — rather than reconstructed from the message
    list. Outside a sample (a bare ``get_model().generate``) the item is
    ``None`` and the headers say so instead of guessing."""
    item: Optional[str] = None
    try:
        from inspect_ai.log._samples import sample_active

        active = sample_active()
    except Exception:  # noqa: BLE001 — no active-sample machinery: no id
        active = None
    if active is not None and getattr(active, "sample", None) is not None:
        sid = getattr(active.sample, "id", None)
        if sid is not None:
            item = str(sid)
    return _TASK.get(), item, _SAMPLE_IDX.get()


# --------------------------------------------------------------------------- #
# the one thing the provider must get right: leading system messages          #
# --------------------------------------------------------------------------- #


def collapse_system_messages(messages: Sequence[Any], si: Optional[str]) -> List[Any]:
    """Exactly ONE leading system message, the SI first.

    Prompted cells are served by the proxy hashing the first
    ``byte_length`` bytes of ``messages[0].content`` against the
    registered prefix — prefix semantics, so ``SI + "\\n\\n" + <task
    system text>`` passes and a SECOND system turn carrying the task text
    does not exist as far as that check is concerned. Fitted cells and
    the base route send the task's own system text unchanged.

    Refusals, all hard (never retried, never degraded):

    * a system message AFTER the first non-system message — the message
      list is not a shape this collapse can express, and silently
      dropping or re-ordering it would change the prompt;
    * a prompted cell whose collapsed first message does not start with
      the SI bytes — that is the exact condition the proxy checks, caught
      here where the error names the cause.

    Idempotent in the SI: a list whose leading system text already starts
    with the SI (a caller that pre-prepended it) is not double-prefixed.
    """
    from inspect_ai.model import ChatMessageSystem

    msgs = list(messages)
    if not msgs:
        raise InspectServedError(
            "tinycap provider: empty message list — nothing to send")
    lead = 0
    while lead < len(msgs) and isinstance(msgs[lead], ChatMessageSystem):
        lead += 1
    rest = msgs[lead:]
    stray = [i for i, m in enumerate(rest) if isinstance(m, ChatMessageSystem)]
    if stray:
        raise InspectServedError(
            f"tinycap provider: system message at position {lead + stray[0]} "
            f"follows a non-system message ({len(msgs)} messages, {lead} "
            "leading system). The served prompted route hashes the SI off "
            "messages[0] only, so a mid-conversation system turn cannot be "
            "collapsed into it without changing the prompt — refusing "
            "rather than re-ordering the task's own messages")
    texts = [t for t in (str(m.text) for m in msgs[:lead]) if t.strip()]
    joined = "\n\n".join(texts)
    if si:
        merged = joined if joined.startswith(si) else (
            f"{si}\n\n{joined}" if joined else si)
    else:
        merged = joined
    out = ([ChatMessageSystem(content=merged)] if merged else []) + rest
    if si:
        first = out[0] if out else None
        if not isinstance(first, ChatMessageSystem) or not str(
                first.text).startswith(si):
            raise InspectServedError(
                "tinycap provider: the collapsed first message does not "
                "start with the SI bytes — the prompted route's prefix hash "
                "would refuse it, and a cell served without its SI is an "
                "unprompted number wearing a prompted cell's name")
    return out


# --------------------------------------------------------------------------- #
# the provider                                                                 #
# --------------------------------------------------------------------------- #


def _register_provider():
    """Define and register ``@modelapi("tinycap")`` once, on first use.

    Deferred so that importing this module for its pure helpers (the id
    computations, the adapter, the preflight) does not require the client
    venv's inspect_ai — and so the registration cannot happen twice."""
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    from inspect_ai.model import GenerateConfig
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
    from inspect_ai.model._registry import modelapi

    class TinycapAPI(OpenAICompatibleAPI):
        """``tinycap/<virtual model>`` — the RoutingProxy as an
        OpenAI-compatible service, with the proxy's rules enforced on the
        way out rather than discovered as 400s.

        ``si`` is the system instruction of a PROMPTED cell (``None`` for
        a fitted cell, the base route, and every grader); ``thinking`` is
        the arm. Both are constructor kwargs rather than module state
        because one container serves a prompted TARGET and an unprompted
        GRADER at the same time — a module-level SI would leak the
        prompted cell's instruction into the judge. ``reasoning_parser``
        is the parser the standup COMPOSED (a SERVER fact — the driver
        reads it off ``endpoint.json``; it applies to the grader's
        requests too), None when none was: it arms the neither-channel
        refusal (module docstring item 5). ``route`` is REQUIRED: the
        routing-table entry this model is served on (a ``RouteEntry`` or
        the table's dict), whose four stamp facts every response is
        verified against (item 6) — a provider that cannot verify what it
        was served must not serve results.

        PER-TASK THINKING (U5-4, PW20b-D). ``thinking_by_task`` is the
        ``{row task: arm}`` map of a run under ONE standup serving mixed
        flags; when it is given, ``thinking`` is ignored and EVERY request
        takes the arm of the task the request belongs to, read off the
        ``_TASK`` ContextVar at build time (:meth:`request_thinking`). The
        arm is therefore a property of the REQUEST, never of the process:
        nothing here caches a prompt, a body or a rendered template across
        requests, so two tasks at opposite arms on one model object cannot
        contaminate each other. A task the map does not name is REFUSED
        rather than served under a default — the driver has already refused
        it at plan time, so reaching here means the ContextVar carries
        something the plan never planned. The GRADER model is built with
        NO map and ``thinking=off``: it is an instrument, and an instrument
        that varies with the task it is grading is not one."""

        def __init__(self, model_name: str, base_url: Optional[str] = None,
                     api_key: Optional[str] = None,
                     config: Optional[Any] = None,
                     si: Optional[str] = None,
                     thinking: str = C.THINKING_OFF,
                     reasoning_parser: Optional[str] = None,
                     route: Any = None,
                     thinking_by_task: Optional[Mapping[str, str]] = None,
                     sampling: Optional[C.SamplingRow] = None,
                     out_of_support: bool = False,
                     stamp_policy: Optional[str] = None,
                     header_located: bool = True,
                     request_evidence_sink: Optional[Callable[..., Any]] = None,
                     client_timeout: Optional[float] = None,
                     sampling_by_task: Optional[Mapping[str, C.SamplingRow]] = None,
                     response_evidence_sink: Optional[Callable[..., Any]] = None,
                     **model_args: Any) -> None:
            C.enable_thinking(thinking)   # refuses an unknown arm
            self.si = si or None
            self.thinking = thinking
            #: whether this route's SERVED mask is header-located — the
            #: declaration predicate. True (the default, the parent's
            #: ``positions: response`` cell) sends ``petri_thinking_on`` on
            #: every lens-route ON request as before; False (a
            #: ``positions: all`` object) sends NONE — the proxy accepts the
            #: undeclared ON render on the served-positions fact
            self.header_located = bool(header_located)
            #: WIRE-BOUND: the driver's request-evidence writer,
            #: called from the httpx request hook with the EXACT body bytes and
            #: headers of every HTTP attempt this provider sends (a retry is a
            #: new attempt), after the hook minted the attempt's
            #: ``X-Tinycap-Request-Id``; None = no record. Installed on the
            #: SDK's own httpx client in :meth:`initialize` (re-installed with
            #: the client, idempotently)
            self.request_evidence_sink = request_evidence_sink
            #: the RESPONSE side of the same boundary — called from
            #: the httpx response hook as ``sink(request_id, status=, resp=, error=, headers=)``
            #: (``request_evidence_client.RequestEvidenceClient.response_sink``) for every chat attempt
            #: this provider sends, so the filed row binds to ONE engine-log request; None = no response
            #: record and no row binding (byte-identical rows). Needs the request sink: without the minted
            #: id there is nothing to bind.
            if response_evidence_sink is not None and request_evidence_sink is None:
                raise InspectServedError(
                    "tinycap provider: a response_evidence_sink needs the request_evidence_sink that mints the "
                    "attempt's X-Tinycap-Request-Id — a response cannot bind to an attempt nobody recorded")
            self.response_evidence_sink = response_evidence_sink
            self._wire_hook_installed_on: Any = None
            #: fit-support transfer (review round B, P1): the
            #: ``resumed_prefill_policy`` label every response of THIS leg must
            #: wear — the driver resolves it ONCE off the standup record
            #: (``contracts.expected_stamp_policy``: the STRICT §2.5.11 row on
            #: the authorised experiment's legs, the cell's ATTESTED label on
            #: every other run) and hands it here; ``None`` = the authorised
            #: experiment's contract row for the route's path
            #: (``contracts.expected_stamp_regimes`` with no policy)
            self.stamp_policy = stamp_policy
            #: fit-support transfer: whether this run is under
            #: ``--allow-out-of-support-thinking``. It decides ONE thing
            #: here — whether a response with NO regime stamp (today's
            #: pre-landing proxy) refuses (the override: fail closed) or is
            #: tolerated and recorded null (in support) — the additive-key rule
            #: the attestation compares apply (contracts.check_serve_declarations)
            self.out_of_support = bool(out_of_support)
            # U8 (PW-sampling): the row this model's requests are drawn under —
            # the TARGET's card row, or GREEDY_SAMPLING for the grader (an
            # instrument samples nothing); None = leave the caller's config
            # alone (a provider built outside the runner)
            if sampling is not None and not isinstance(sampling, C.SamplingRow):
                raise InspectServedError(
                    f"tinycap provider: sampling must be a contracts.SamplingRow or None, "
                    f"got {type(sampling).__name__}")
            self.sampling = sampling
            # Under `protocol` the row is a property of the REQUEST's task —
            # `{row task: SamplingRow}` (the EXECUTED rows,
            # contracts.protocol_executed_by_task); `sampling_params` selects by the
            # task it already receives and refuses a task the map does not name.
            # ONE source per model: a single row AND a map together refuse.
            if sampling_by_task is not None:
                if sampling is not None:
                    raise InspectServedError(
                        "tinycap provider: sampling (one frozen row) and sampling_by_task (a per-task map) were "
                        "both given — a request's row has exactly one source")
                bad = sorted(t for t, r in dict(sampling_by_task).items() if not isinstance(r, C.SamplingRow))
                if bad:
                    raise InspectServedError(
                        f"tinycap provider: sampling_by_task[{bad}] must be contracts.SamplingRow values")
            self.sampling_by_task: Optional[Dict[str, C.SamplingRow]] = (
                {str(t): r for t, r in dict(sampling_by_task).items()} if sampling_by_task is not None else None)
            # the map's VALUES are per-task ARMS (contracts.THINKING_ARMS: the two switch
            # arms and the no-switch `not_available`), never the mode token
            # (`check_thinking_arm` admits arms only, so `per-task` refuses here by construction)
            try:
                self.thinking_by_task = (
                    {str(k): C.check_thinking_arm(v) for k, v in thinking_by_task.items()}
                    if thinking_by_task else None)
            except ValueError as e:
                raise InspectServedError(
                    f"tinycap provider: thinking_by_task carries a value that is not an arm — the map's VALUES are arms "
                    f"({C.THINKING_ARMS}), never the mode token {C.THINKING_PER_TASK!r}: {e}") from None
            if route is None:
                raise InspectServedError(
                    "tinycap provider: `route` (the routing-table entry this model is "
                    "served on) is required — every response's attestation stamps are "
                    "verified against it (E10 N2); a provider that cannot verify what "
                    "it was served must not serve results")
            self.route = stamp_route(route)
            #: whether the route is served through the lens
            #: (``pe.is_lens_route`` parses the serving identity) — read ONCE
            #: per model, lazily at the first request, where the parse's
            #: refusal fires today (construction accepts any non-empty identity
            #: string); the expected regime row per arm is memoised the same
            #: way, never precomputed, so a contracts module that cannot say
            #: the row still refuses at the first response, as today
            self._lens: Optional[bool] = None
            self._expected_regimes: Dict[str, C.StampRegimes] = {}
            #: per row task, the COUNT of
            #: each regime stamp triple this model's accepted responses carried
            #: (``on_response``); ``run_one_task`` files the one value and the
            #: distinct count of each field into the task summary
            self.regime_stamps: Dict[Optional[str], Counter] = {}
            if reasoning_parser is not None and (
                    not isinstance(reasoning_parser, str)
                    or not reasoning_parser.strip()):
                raise InspectServedError(
                    f"tinycap provider: reasoning_parser={reasoning_parser!r} "
                    "must be a non-empty parser name or None")
            self.reasoning_parser = reasoning_parser
            for forbidden in ("stream", "responses_api"):
                if model_args.pop(forbidden, None):
                    raise InspectServedError(
                        f"tinycap provider: {forbidden}=True is refused — the "
                        "RoutingProxy 400s `stream: true` and serves only "
                        "/v1/chat/completions")
            # the per-REQUEST timeout, resolved HERE so the provider is
            # safe by construction whoever builds it (`build_model`, a test's
            # direct construction): None = the ONE served-request budget every
            # hop shares (contracts.INSPECT_TIMEOUT_S == PROXY_UPSTREAM_TIMEOUT_S),
            # a value below the proxy's upstream wait is refused. The parent
            # (inspect_ai 0.3.260 OpenAICompatibleAPI.__init__) stores it as
            # `self.client_timeout` and builds BOTH clients from it — the SDK's
            # httpx client (`_create_http_client`: httpx2.Timeout(timeout=…,
            # connect=5.0)) and the AsyncOpenAI client (`_create_client`:
            # timeout=…), the per-request default the SDK stamps on every
            # request — and rebuilds a closed http client from the same
            # attribute in `initialize()`. Left unset, the SDK's own default
            # (600 s read) bounded every request: see `resolve_client_timeout`.
            client_timeout = resolve_client_timeout(client_timeout)
            super().__init__(
                model_name=model_name, base_url=base_url, api_key=api_key,
                config=config if config is not None else GenerateConfig(),
                client_timeout=client_timeout,
                service=C.INSPECT_PROVIDER_NAME,
                api_key_var=C.INSPECT_API_KEY_ENV,
                # NON-NEGOTIABLE, not defaults: the proxy refuses streaming
                # and exposes no /v1/responses.
                stream=False, responses_api=False,
                strict_tools=C.INSPECT_STRICT_TOOLS,
                **model_args)
            if self.service_model_name() != self.route.virtual_model:
                raise InspectServedError(
                    f"tinycap provider: model name {model_name!r} names virtual model "
                    f"{self.service_model_name()!r} but the route given is "
                    f"{self.route.virtual_model!r} (cell {self.route.cell_id!r}) — the "
                    "stamps would be checked against a route the requests never go to")

        def initialize(self) -> None:
            super().initialize()
            self._install_wire_hook()

        def _install_wire_hook(self) -> None:
            """Put :meth:`_wire_request_hook` on the SDK's httpx client (the ONE
            place the final body exists as bytes), once per client instance,
            keeping inspect_ai's own hooks."""
            client = getattr(self, "http_client", None)
            if client is None or self._wire_hook_installed_on is client:
                return
            hooks = {k: list(v) for k, v in dict(client.event_hooks).items()}
            hooks.setdefault("request", []).append(self._wire_request_hook)
            # the response hook is installed beside it and is INERT without a response sink (it returns before
            # reading the body), so a sink set after construction — `initialize()` already ran — still takes effect
            hooks.setdefault("response", []).append(self._wire_response_hook)
            client.event_hooks = hooks
            self._wire_hook_installed_on = client

        async def _wire_request_hook(self, request: Any) -> None:
            """The HTTP boundary: for every POST to ``/chat/completions``
            this provider sends, mint the attempt's
            ``X-Tinycap-Request-Id`` onto the outgoing headers and hand the
            EXACT body bytes + headers to the driver's request-evidence writer.
            httpx calls request hooks right before the transport sends, with the
            object the transport sends, so the id and the bytes recorded are
            the id and the bytes on the wire; a retry passes here again as a
            new attempt with a new id."""
            if self.request_evidence_sink is None:
                return
            if str(request.method).upper() != "POST" or not str(request.url.path).endswith("/chat/completions"):
                return
            request.headers[C.HEADER_REQUEST_ID] = uuid.uuid4().hex
            self.request_evidence_sink(bytes(request.content), headers=dict(request.headers.items()))

        async def _wire_response_hook(self, response: Any) -> None:
            """The RESPONSE side of the HTTP boundary: for every
            ``/chat/completions`` reply to an attempt :meth:`_wire_request_hook` minted,
            read the body (cached — the SDK reads the same bytes after), hand ``(request_id,
            completion_id, status)`` to :attr:`response_evidence_sink`, and set this context's
            binding for :meth:`generate` to stamp onto the output. httpx runs response hooks per
            attempt, so a retried request's successful attempt is the binding that survives."""
            if self.response_evidence_sink is None:
                return
            request = response.request
            if str(request.method).upper() != "POST" or not str(request.url.path).endswith("/chat/completions"):
                return
            hdrs = dict(request.headers.items())
            rid = next((v for k, v in hdrs.items() if k.lower() == C.HEADER_REQUEST_ID.lower()), None)
            if not rid:
                return   # not an attempt this provider minted (no sink at send time)
            await response.aread()
            try:
                body = json.loads(bytes(response.content).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                body = None
            ok = 200 <= int(response.status_code) < 300
            body = body if (ok and isinstance(body, Mapping)) else None
            self.response_evidence_sink(str(rid), status=int(response.status_code), resp=body,
                                        error=(None if ok else bytes(response.content)[:2000].decode("utf-8", "replace")),
                                        headers=hdrs)
            cid = body.get("id") if body is not None else None
            _WIRE_BINDING.set({rev.ROW_REQUEST_ID_FIELD: str(rid),
                               rev.ROW_COMPLETION_ID_FIELD: (str(cid) if isinstance(cid, str) and cid else None)})

        def should_stream(self, config: Any) -> bool:
            return False

        @property
        def lens(self) -> bool:
            """True when this model's route is served through the lens."""
            if self._lens is None:
                self._lens = pe.is_lens_route(self.route)
            return self._lens

        def expected_regimes(self, arm: str) -> C.StampRegimes:
            """This leg's expected regime stamp row under ``arm``
            (``contracts.expected_stamp_regimes`` at :attr:`stamp_policy` — the
            render from the arm, the salt pair from the leg's label), computed
            once per arm."""
            row = self._expected_regimes.get(arm)
            if row is None:
                row = self._expected_regimes[arm] = C.expected_stamp_regimes(
                    lens=self.lens, arm=arm, policy=self.stamp_policy)
            return row

        def on_response(self, response: Dict[str, Any]) -> None:
            """Every chat completion — target and grader alike — before
            Inspect reads a choice (module docstring item 6): the proxy's
            four attestation stamps must equal THIS model's route.
            ``OpenAICompatibleAPI.generate`` calls this hook with
            ``completion.model_dump()``, which carries the proxy's extra
            top-level fields verbatim (pydantic keeps model extras;
            verified on the pinned openai SDK). Raised OUTSIDE the SDK's
            send, so a mismatch is never a "connection error" and never
            re-sent; Inspect files it as the sample's error (PB8).

            The same hook reads the response's regime stamp
            triple — ``petri.cache_salt_regime`` / ``petri.resumed_prefill_policy``
            / ``petri.render_regime`` (``contracts.stamp_regimes``) — and holds
            it to THIS request's expected row (``contracts.expected_stamp_regimes``
            over the route's path and the request's arm): a PRESENT value
            outside the contract's vocabulary, a render regime that is not the
            arm's, or any other row refuses the sample at once, on every run
            (fail fast); a field the stamp does NOT carry (today's pre-landing
            proxy stamps no ``render_regime``) refuses only under the
            out-of-support override (:attr:`out_of_support`) and is otherwise
            tolerated — the additive-key rule the attestation compares apply.
            The accepted triple (null fields included) is COUNTED per task on
            :attr:`regime_stamps`, which ``run_one_task`` reads into the
            summary's ``render_regime`` / ``salt_regime`` / ``resumed_policy``
            and their distinct-value counts (a null field reads null / 0)."""
            task, item, idx = current_request_identity()
            where = (f"tinycap provider: {task or '<no task>'} item {item!r} sample {idx} "
                     f"on {self.route.virtual_model!r}")
            try:
                pe.assert_response_stamps(response, self.route)
            except pe.StampMismatch as e:
                raise InspectServedError(f"{where}: {e}") from e
            arm = self.request_thinking()
            try:
                seen = C.stamp_regimes(response)
            except C.OutOfSupportAttestationError as e:
                # a PRESENT malformed stamp (null / empty / non-string /
                # off-vocabulary): an incompatible proxy, refused on EVERY run
                # before any row comparison — never read as an absent field
                raise InspectServedError(f"{where}: {e}") from e
            expected = self.expected_regimes(arm)
            problems = regime_stamp_problems(seen, expected, out_of_support=self.out_of_support)
            if problems:
                raise InspectServedError(
                    f"{where}: the response's regime stamp is not this leg's contract row "
                    f"(expected {expected._asdict()}, arm {arm!r}): " + "; ".join(problems)
                    + " — a mixed, off-vocabulary or unexpected regime refuses the task "
                    "(fit-support transfer plan §3.2; contract §2.5.11)")
            self.regime_stamps.setdefault(task, Counter())[seen] += 1
            # The completion the SDK parsed IS the one the response hook
            # bound — a disagreeing id would file a row against another attempt's engine request
            binding = _WIRE_BINDING.get()
            if binding is not None and binding.get(rev.ROW_COMPLETION_ID_FIELD) != response.get("id"):
                raise InspectServedError(
                    f"{where}: the response hook bound completion {binding.get(rev.ROW_COMPLETION_ID_FIELD)!r} to "
                    f"request {binding.get(rev.ROW_REQUEST_ID_FIELD)!r} but the SDK returned completion "
                    f"{response.get('id')!r} — the filed row would bind to another attempt; refusing")
            if self.no_switch_arm(arm):
                # llama-paper-rows (P21): the NO-SWITCH policy's capture — the RAW assistant
                # content of every choice, verbatim, BEFORE the SDK's conversion can lift a
                # quoted think block out of it; `generate` restores it after
                _NO_SWITCH_RAW.set(_raw_choice_contents(response))

        @staticmethod
        def no_switch_arm(arm: str) -> bool:
            """Whether ``arm`` is the no-switch arm (:data:`contracts.THINKING_NOT_AVAILABLE`):
            the response-conversion policy (P21) applies to exactly these requests."""
            return arm == C.THINKING_NOT_AVAILABLE

        async def generate(self, input, tools, tool_choice, config):  # noqa: ANN001
            arm = self.request_thinking()
            token = _NO_SWITCH_RAW.set(None)
            bind_token = _WIRE_BINDING.set(None)
            try:
                out = await super().generate(
                    collapse_system_messages(input, self.si), tools, tool_choice,
                    self.tinycap_config(config))
                if self.no_switch_arm(arm):
                    out = self.restore_no_switch_content(out, _NO_SWITCH_RAW.get())
                # The attempt binding onto the output's metadata,
                # ONLY when the response hook recorded one (no sink = no key, byte-identical rows)
                out = self.stamp_wire_binding(out, _WIRE_BINDING.get())
            finally:
                _NO_SWITCH_RAW.reset(token)
                _WIRE_BINDING.reset(bind_token)
            self.assert_channels(out)
            return out

        @staticmethod
        def stamp_wire_binding(out: Any, binding: "Optional[Dict[str, Optional[str]]]") -> Any:
            """``ModelOutput.metadata[WIRE_BINDING_METADATA_KEY] = {request_id, completion_id}`` when the
            response hook bound this generate's answering attempt; ``out`` untouched otherwise."""
            if binding is None:
                return out
            from inspect_ai.model import ModelOutput

            output = out[0] if isinstance(out, tuple) else out
            if not isinstance(output, ModelOutput):
                return out
            meta = dict(output.metadata or {})
            meta[WIRE_BINDING_METADATA_KEY] = dict(binding)
            output.metadata = meta
            return out

        def restore_no_switch_content(self, out: Any, raw: "Optional[List[Dict[str, Any]]]") -> Any:
            """THE NO-SWITCH RESPONSE-CONVERSION POLICY (llama-paper-rows P21).

            The pinned inspect_ai 0.3.260 ``OpenAICompatibleAPI`` converts every
            response through ``chat_choices_from_openai`` ->
            ``parse_reasoning_content``, which lifts any COMPLETE
            ``<think>…</think>`` block out of the assistant text into a
            ``ContentReasoning`` part — client-side, even when no server reasoning
            parser is composed (the GLM / steered-Qwen artefact of MATH-500). On a
            family with NO thinking channel that lift would hand the scorer a
            MODIFIED text and discard the very evidence (a quoted think block) the
            reader must refuse as a stray. So, for a request under the no-switch
            arm: (i) ``on_response`` captured each choice's raw content verbatim;
            (ii) here every choice's message becomes exactly ONE
            ``ContentText(raw)`` — every lifted ``ContentReasoning`` part removed;
            (iii) the lift's firing is RECORDED, never dropped, in
            ``ModelOutput.metadata[NO_SWITCH_METADATA_KEY]`` (per choice:
            ``client_lift_fired``, ``lifted_reasoning`` — the text the SDK had
            lifted, kept as evidence — and ``raw_content_sha256``;
            ``contracts.ROW_LIFT_FIELDS``), which the log adapter copies onto the
            row; (iv) :meth:`assert_channels` then REFUSES a surviving reasoning
            part or a text that differs from the raw by one byte. The raw HTTP body
            stays in the eval log's model-call attachment as before. A response
            with no capture (a bad-request ``ModelOutput`` the SDK synthesised, or
            an exception) is returned untouched — there was no conversion to undo."""
            from inspect_ai._util.content import ContentText
            from inspect_ai.model import ModelOutput

            output = out[0] if isinstance(out, tuple) else out
            if not isinstance(output, ModelOutput) or not output.choices or not raw:
                return out
            if len(raw) != len(output.choices):
                task, item, idx = current_request_identity()
                raise InspectServedError(
                    f"tinycap provider: {task or '<no task>'} item {item!r} sample {idx}: the no-switch policy captured "
                    f"{len(raw)} raw choice(s) but the SDK converted {len(output.choices)} — the raw content cannot be "
                    "matched to its choice; refusing to score a text that may not be the model's")
            records: List[Dict[str, Any]] = []
            for choice, cap in zip(output.choices, raw):
                msg = choice.message
                raw_text = cap["content"] if cap["content"] is not None else ""
                lifted = [str(c.reasoning) for c in msg.content_list
                          if getattr(c, "type", None) == "reasoning" and getattr(c, "reasoning", None)]
                converted_text = str(msg.text or "")
                fired = bool(lifted) or converted_text != raw_text
                msg.content = [ContentText(text=raw_text)]
                records.append({"client_lift_fired": fired,
                                "lifted_reasoning": ("\n".join(lifted) if lifted else None),
                                "raw_content_sha256": cap["sha256"]})
            # `ModelOutput.completion` is a STORED field the SDK set from the converted first choice at
            # construction (not a live view of the message) — and it is what the scorer reads
            # (`state.output.completion`), so it is restored too
            output.completion = str(output.choices[0].message.text or "")
            meta = dict(output.metadata or {})
            meta[NO_SWITCH_METADATA_KEY] = records
            output.metadata = meta
            return out

        def request_thinking(self) -> str:
            """The arm THIS request is served under (PW20b-D).

            With no ``thinking_by_task`` map it is the constructor's arm —
            the whole-configuration run, unchanged. With a map it is the
            CURRENT task's arm, so one model object serves mixed flags under
            one standup. An unmapped (or absent) task is refused, never
            defaulted: a request whose task the plan never planned would
            otherwise be served under a silently-chosen arm, which is the
            one thing the four-way matrix exists to make impossible."""
            if not self.thinking_by_task:
                return self.thinking
            task, item, idx = current_request_identity()
            try:
                return self.thinking_by_task[task]
            except KeyError:
                raise InspectServedError(
                    f"tinycap provider: request for task {task!r} (item {item!r} "
                    f"sample {idx}) has no arm in this run's per-task thinking map "
                    f"{sorted(self.thinking_by_task)} — refusing to pick one; the "
                    "driver refuses an unmapped task at plan time, so this is a "
                    "task the run never planned") from None

        def assert_channels(self, out: Any) -> None:
            """The neither-channel refusal (module docstring item 5): with
            a parser composed, a message with no text, no reasoning and no
            tool call cannot be told apart from text the parser lost, so
            it is raised here — Inspect files it as the sample's error.
            Without a composed parser an empty answer is an ordinary
            (wrong) answer; a tool-call-only message (BFCL) is normal.

            The parser is a SERVER fact, so under per-task thinking this
            refusal covers the OFF tasks of a mixed run too: the standup
            composed the parser for the ON tasks, and an empty-both-channels
            response on an OFF task is exactly as unreadable as on an ON
            one. That is deliberate — the arm does not change what the
            server's parser can lose.

            llama-paper-rows (P21): under the NO-SWITCH arm the check is the
            restoration's own — after :meth:`restore_no_switch_content` NO
            choice may carry a reasoning part and every choice's text must
            EQUAL the raw content the policy captured (sha256); either is a
            conversion the scorer must not read."""
            from inspect_ai.model import ModelOutput

            output = out[0] if isinstance(out, tuple) else out
            if isinstance(output, ModelOutput) and output.choices and self.no_switch_arm(self.request_thinking()):
                records = list((output.metadata or {}).get(NO_SWITCH_METADATA_KEY) or [])
                task, item, idx = current_request_identity()
                where = f"tinycap provider: {task or '<no task>'} item {item!r} sample {idx}"
                if len(records) != len(output.choices):
                    raise InspectServedError(
                        f"{where}: the no-switch policy recorded {len(records)} choice(s) against {len(output.choices)} "
                        "converted — the restoration did not run over every choice; refusing to score")
                for k, (choice, rec) in enumerate(zip(output.choices, records)):
                    m = choice.message
                    surviving = [c for c in m.content_list if getattr(c, "type", None) == "reasoning"]
                    if surviving:
                        raise InspectServedError(
                            f"{where}: choice {k} still carries {len(surviving)} reasoning part(s) after the no-switch "
                            "restoration — a family with no thinking channel emits none; refusing to score a converted text")
                    got = text_sha256(str(m.text or ""))
                    if got != rec["raw_content_sha256"]:
                        raise InspectServedError(
                            f"{where}: choice {k}'s text sha256 {got[:16]} != the raw content's {str(rec['raw_content_sha256'])[:16]} — "
                            "the scorer would read a text the model did not return; refusing")
                if str(output.completion or "") != str(output.choices[0].message.text or ""):
                    raise InspectServedError(
                        f"{where}: ModelOutput.completion (the text the scorer reads) differs from the first choice's restored "
                        "text — the stored field was not restored with the message; refusing to score a converted text")
            if not self.reasoning_parser:
                return
            if not isinstance(output, ModelOutput) or not output.choices:
                return
            msg = output.choices[0].message
            content, reasoning = message_channels(msg)
            if pe.channel_verdict(content, reasoning, has_tool_call=bool(msg.tool_calls),
                                  reasoning_parser=self.reasoning_parser) != pe.ERR_EMPTY_RESPONSE:
                return
            task, item, idx = current_request_identity()
            usage = output.usage
            raise InspectServedError(
                f"tinycap provider: {task or '<no task>'} item {item!r} sample "
                f"{idx}: the response carries neither content nor reasoning "
                f"under the composed reasoning parser {self.reasoning_parser!r} "
                f"(stop_reason={output.stop_reason!r}, output_tokens="
                f"{getattr(usage, 'output_tokens', None)!r}) — cannot tell an "
                "empty answer from text the parser lost; refusing to score it")

        # -- request shaping, split out so the tests can read one body --- #

        def tinycap_config(self, config):  # noqa: ANN001
            """The proxy-safe :class:`GenerateConfig` for one call: thinking
            kwargs, a per-(item, sample) seed, the ledger headers — and the
            three refusals that must never reach the wire.

            The thinking flag is resolved PER REQUEST
            (:meth:`request_thinking`), so a mixed-arm run sends the right
            ``chat_template_kwargs.enable_thinking`` on every body."""
            if config.num_choices is not None:
                raise InspectServedError(
                    f"tinycap provider: num_choices={config.num_choices} would "
                    "send `n` on a served route, which E4 forbids: a fork may "
                    "run UNSTEERED (unverified), so avg@k is k independent "
                    "seeded requests, never one n-way call")
            extra = dict(config.extra_body or {})
            bad = sorted(set(extra) & set(C.FORBIDDEN_CLIENT_KEYS))
            if bad:
                raise InspectServedError(
                    f"tinycap provider: extra_body carries {bad}, which the "
                    "proxy refuses as selector smuggling (E3) — the serving "
                    "geometry and the object are the standup's to choose, "
                    "never a client's")
            if extra.get("stream"):
                raise InspectServedError(
                    "tinycap provider: extra_body sets stream — the proxy 400s it")
            arm = self.request_thinking()
            have = dict(extra.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {})
            if not C.sends_thinking_kwargs(arm):
                # a family with NO thinking switch is served at its shipped render — no
                # `chat_template_kwargs.enable_thinking` rides the body (an inert kwarg
                # would still be a recorded switch), and a caller's own switch is refused:
                # there is nothing for it to flip
                if C.ENABLE_THINKING_KEY in have:
                    raise InspectServedError(
                        f"tinycap provider: a caller set {C.ENABLE_THINKING_KEY}={have[C.ENABLE_THINKING_KEY]!r} on a "
                        f"request under the {arm!r} arm — the served family has no thinking switch "
                        f"(contracts.NO_THINKING_CHANNEL_FAMILIES); no switch is sent, none may be set")
                if have:
                    extra[C.CHAT_TEMPLATE_KWARGS_KEY] = have
                else:
                    extra.pop(C.CHAT_TEMPLATE_KWARGS_KEY, None)
            else:
                want = C.chat_template_kwargs(arm)[C.CHAT_TEMPLATE_KWARGS_KEY]
                if (C.ENABLE_THINKING_KEY in have
                        and have[C.ENABLE_THINKING_KEY] != want[C.ENABLE_THINKING_KEY]):
                    raise InspectServedError(
                        "tinycap provider: a caller set "
                        f"{C.ENABLE_THINKING_KEY}={have[C.ENABLE_THINKING_KEY]!r} "
                        f"against this request's {arm!r} arm — the arm is the whole "
                        "point of the four-way matrix and is never overridden per "
                        "request")
                have.update(want)
                extra[C.CHAT_TEMPLATE_KWARGS_KEY] = have
            # fit-support transfer (contract §2.5.3–§2.5.5): the
            # thinking-ON declaration iff this request sends enable_thinking:
            # true on a LENS route — never on the base route (400
            # thinking_on_not_lens), never under OFF, never with false/null; a
            # caller's own value for the key is refused rather than merged
            try:
                declaration = C.thinking_on_declaration(lens=self.lens, arm=arm,
                                                        header_located=self.header_located)
            except C.OutOfSupportDeclarationUnreconciled as e:
                raise InspectServedError(
                    f"tinycap provider: refusing a thinking-ON lens-route request — {e} (an ON "
                    "request without its declaration would certify the OFF render)") from None
            if C.OOS_THINKING_DECLARATION_KEY in extra:
                # ANY caller-supplied value — the right one included: the
                # declaration is derived from the route and the arm here, and a
                # caller that spells it has bypassed that derivation
                raise InspectServedError(
                    f"tinycap provider: extra_body carries {C.OOS_THINKING_DECLARATION_KEY}="
                    f"{extra[C.OOS_THINKING_DECLARATION_KEY]!r} — the declaration is the provider's "
                    "to add (the JSON boolean true, on a lens route, under enable_thinking: true) "
                    "and never a caller's, whatever the value")
            extra.update(declaration)
            task, item, idx = current_request_identity()
            headers = dict(config.extra_headers or {})
            if task:
                headers[C.HEADER_TASK] = task
            if item is not None:
                headers[C.HEADER_SAMPLE] = C.sample_header_value(item, idx)
            update: Dict[str, Any] = {"extra_body": extra,
                                      "extra_headers": headers or None}
            if config.seed is None and item is not None:
                # the task's seed rule (ifeval-canonical-battery: IFEval's own)
                update["seed"] = seed_for_task(task, item, idx)
            if self.sampling is not None or self.sampling_by_task is not None:
                update.update(self.sampling_params(config, extra, task=task, item=item, idx=idx))
            # the request-evidence record is NOT written here: this is the
            # fragment; the record is written from the EXACT bytes at the HTTP
            # boundary (`_wire_request_hook`)
            return config.model_copy(update=update)

        def request_sampling_row(self, task: Optional[str], *, where: str = "") -> C.SamplingRow:
            """The :class:`contracts.SamplingRow` THIS request decodes under:
            the one frozen row (every pre-existing mode), or — with a per-task
            map (``protocol``) — the CURRENT task's executed row. A task the
            map does not name (or no task at all) is REFUSED, never
            defaulted: the driver refuses an unmapped task at plan time, so
            reaching here means a request the run never planned."""
            if self.sampling_by_task is None:
                if self.sampling is None:
                    raise InspectServedError(
                        f"tinycap provider: {where}: no sampling row and no per-task map on this provider")
                return self.sampling
            row = self.sampling_by_task.get(task) if task is not None else None
            if row is None:
                raise InspectServedError(
                    f"tinycap provider: {where}: request for task {task!r} has no row in this run's per-task "
                    f"sampling map {sorted(self.sampling_by_task)} — refusing to pick one ({C.SAMPLING_PROTOCOL}: "
                    "every filed task decodes under its OWN reference row, and a task with none was refused "
                    "at plan time, so this is a request the run never planned)")
            return row

        def sampling_params(self, config, extra: Dict[str, Any], *, task, item, idx
                            ) -> Dict[str, Any]:
            """The row's three numbers onto ONE request (U8, PW-sampling) — the
            one place a request's sampling parameters are resolved.

            ``temperature`` and ``top_p`` ride as ``GenerateConfig`` fields
            (inspect_ai's OpenAI params send them); ``top_k`` rides in
            ``extra_body`` because those params do not carry it (verified on
            the pinned client venv: ``openai_completion_params`` sends seed /
            temperature / top_p and no top_k) and vLLM reads it off the body.
            A value the row leaves None is REMOVED (``top_p`` None, no
            ``top_k`` key), so a task's own pin cannot survive underneath. A
            caller's DIFFERENT explicit value is refused, never overridden
            silently: the row is frozen per model and the eval-level values
            the runner sets are the row's, so a disagreement here is a task
            or solver trying to sample its own way — the four-way matrix
            collapsing to whatever it asked for.

            With a per-task map the row is THIS request's task's
            (:meth:`request_sampling_row`); the refusal below then holds the
            caller's values to THAT row, and a request on a task the map does
            not name refuses — never a default."""
            where = f"{task or '<no task>'} item {item!r} sample {idx}"
            row = self.request_sampling_row(task, where=where)
            for name, want in (("temperature", float(row.temperature)),
                               ("top_p", None if row.top_p is None else float(row.top_p)),
                               ("top_k", None if row.top_k is None else int(row.top_k))):
                have = getattr(config, name, None)
                if have is not None and (want is None or float(have) != float(want)):
                    raise InspectServedError(
                        f"tinycap provider: {where}: a caller set {name}={have!r} against this "
                        f"model's frozen sampling row ({name}={want!r}; {row.source[:60]}...) — "
                        "the row is per model and never overridden per request (U8)")
            if "top_k" in extra and extra["top_k"] != row.top_k:
                raise InspectServedError(
                    f"tinycap provider: {where}: extra_body carries top_k={extra['top_k']!r} against "
                    f"this model's frozen row (top_k={row.top_k!r}) — never overridden per request (U8)")
            if row.top_k is None:
                extra.pop("top_k", None)
            else:
                extra["top_k"] = int(row.top_k)
            out: Dict[str, Any] = {"temperature": float(row.temperature),
                                   "top_p": (None if row.top_p is None else float(row.top_p)),
                                   "top_k": (None if row.top_k is None else int(row.top_k))}
            # the OPTIONAL terms ride ONLY when the row sets them —
            # `presence_penalty` as a GenerateConfig field (inspect_ai's OpenAI
            # params send it), `min_p` / `repetition_penalty` in extra_body
            # (vLLM reads them off the body); a legacy row (all None) leaves
            # the request byte-identical
            pp = getattr(row, "presence_penalty", None)
            if pp is not None:
                have = getattr(config, "presence_penalty", None)
                if have is not None and float(have) != float(pp):
                    raise InspectServedError(
                        f"tinycap provider: {where}: a caller set presence_penalty={have!r} against this "
                        f"model's frozen row (presence_penalty={pp!r}) — never overridden per request")
                out["presence_penalty"] = float(pp)
            for name in ("min_p", "repetition_penalty"):
                want = getattr(row, name, None)
                if name in extra and (want is None or float(extra[name]) != float(want)):
                    raise InspectServedError(
                        f"tinycap provider: {where}: extra_body carries {name}={extra[name]!r} against this "
                        f"model's frozen row ({name}={want!r}) — never overridden per request")
                if want is not None:
                    extra[name] = float(want)
            return out

    # `modelapi` registers a WRAPPER that instantiates the class; the
    # class itself is untouched, so registering out-of-line lets callers
    # (and the tests) hold the real class while `get_model("tinycap/...")`
    # still resolves.
    modelapi(name=C.INSPECT_PROVIDER_NAME)(TinycapAPI)
    _PROVIDER = TinycapAPI
    return _PROVIDER


_PROVIDER: Optional[type] = None


#: Each regime stamp field's closed vocabulary (contract §2.5.11), by the
#: summary's field name — what a PRESENT stamp value is judged against.
#: the contract's closed vocabulary per stamp field — contracts' ONE mapping, by reference
_STAMP_VOCABULARY: Mapping[str, Tuple[str, ...]] = C.STAMP_VOCABULARY


def regime_stamp_problems(seen: C.StampRegimes, expected: C.StampRegimes, *,
                          out_of_support: bool = False) -> List[str]:
    """Why one response's regime stamp triple is NOT the leg's expected row —
    ``[]`` when it is. Per field, BY NAME: a PRESENT value outside the
    contract's closed vocabulary (``SALT_REGIME_LITERALS`` /
    ``RESUMED_POLICY_LABELS`` / ``RENDER_REGIME_LITERALS`` — ``unspecified``
    included: the harness always sends the kwarg) or one that is not the
    expectation's refuses on EVERY run; a field the stamp does not carry
    (today's pre-landing proxy) refuses only under ``out_of_support`` — the
    additive-key rule: an absent fact fails closed under the override and is
    tolerated, recorded null, in support, where nothing about the run depends
    on it."""
    problems: List[str] = []
    for name in C.EXPECTED_STAMP_TRIPLE_FIELDS:
        got, want = getattr(seen, name), getattr(expected, name)
        if got is None:
            if out_of_support:
                problems.append(
                    f"{name}: the stamp carries no value (a pre-landing proxy?) — an out-of-support "
                    f"run requires the landed proxy's stamp ({C.FLAG_ALLOW_OOS_THINKING})")
        elif got not in _STAMP_VOCABULARY[name]:
            problems.append(
                f"{name}: {got!r} is outside the contract's vocabulary {list(_STAMP_VOCABULARY[name])}")
        elif got != want:
            problems.append(f"{name}: {got!r} != expected {want!r}")
    return problems


def _regime_fields_from_prior(prior: Optional[Mapping[str, Any]], *,
                              expected: Optional[C.StampRegimes], task: str,
                              out_of_support: bool, fields: Sequence[str],
                              counts: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The six regime fields carried from a task's EXISTING summary when every
    ``.eval`` log was reused (:func:`regime_summary_fields`): its present
    values are held to the expected row (a store from another regime refuses);
    ``None`` when ``prior`` does not carry all six, so the caller decides."""
    if not (isinstance(prior, Mapping) and all(k in prior for k in (*fields, *counts))):
        return None
    carried = C.StampRegimes(**{f: prior[f] for f in fields})
    problems = (regime_stamp_problems(carried, expected, out_of_support=out_of_support)
                if expected is not None else [])
    if problems:
        raise InspectServedError(
            f"{task}: every .eval log was reused and the task's existing summary carries "
            f"regime fields that are not the leg's contract row (expected "
            f"{expected._asdict() if expected else None}): " + "; ".join(problems)
            + " (fit-support transfer plan §3.2; contract §2.5.11)")
    print(f"[inspect-served] {task}: every .eval log was reused — the regime fields are "
          "carried from the task's existing summary (stamped at the live run that wrote "
          "the logs)", flush=True)
    return {**{f: prior[f] for f in fields}, **{c: int(prior[c]) for c in counts}}


def regime_summary_fields(observed: Optional[Mapping[C.StampRegimes, int]], *,
                          expected: Optional[C.StampRegimes], task: str,
                          out_of_support: bool,
                          prior: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The summary's six regime fields (``contracts.SUMMARY_PAYLOAD_FIELDS``'
    last six) over the stamp triples a task's responses
    carried: the ONE value of each field and its distinct-value count
    (``contracts.STAMP_REGIME_COUNT_FIELD``), which must read 1 and equal the
    leg's expected row — else the task is REFUSED here, after generation, as
    the belt to ``on_response``'s braces (that hook refuses each response as
    it arrives; this re-reads the accumulated set, so a leg that somehow
    filed two rows cannot be summarised as one).

    ``observed`` None means the model exposes NO stamp accumulator (a caller
    without the tinycap provider — a test stub); the six fields then read
    ``None`` / 0 and the payload says so by construction (the analysis reads a
    null regime as PENDING, never a pass).

    The additive-key rule, per field: a PRESENT value
    that disagrees, or a leg whose responses carry the field on some responses
    and not others, refuses on every run; a field ABSENT from every response
    (today's pre-landing proxy) refuses only under ``out_of_support`` and reads
    ``None`` / 0 in support.

    An accumulator that saw NO response for this task (every ``.eval`` log
    reused) reads the fields off ``prior`` — the task's EXISTING summary, which
    was stamped at the live run that wrote the logs — when it carries all six
    (its present values are still held to the expected row; a store from
    another regime refuses); with no such prior the stamps cannot be read, so
    an out-of-support run refuses by name and an in-support one reads
    ``None`` / 0 and says so."""
    fields = list(C.EXPECTED_STAMP_TRIPLE_FIELDS)
    counts = [C.STAMP_REGIME_COUNT_FIELD[f] for f in fields]
    null = {**{f: None for f in fields}, **{c: 0 for c in counts}}
    if observed is None:
        print(f"[inspect-served] {task}: no regime stamps observed (the model exposes no "
              "tinycap provider accumulator) — the summary's regime fields read null",
              flush=True)
        return null
    if not observed:
        carried = _regime_fields_from_prior(prior, expected=expected, task=task,
                                            out_of_support=out_of_support,
                                            fields=fields, counts=counts)
        if carried is not None:
            return carried
        if out_of_support:
            raise InspectServedError(
                f"{task}: no response of this task passed through the provider (every .eval log "
                "was reused) and no existing summary of the task carries its regime fields, so "
                "they cannot be read — the render / cache-salt regime is certified off LIVE "
                "responses (or a summary stamped from them) only; an out-of-support run refuses "
                "rather than file null regimes; remove the retained log dir(s) so the task "
                "re-runs (fit-support transfer plan §3.2)")
        print(f"[inspect-served] {task}: every .eval log was reused and no existing summary "
              "carries the regime fields — an in-support run records them null", flush=True)
        return null
    distinct = {f: sorted({getattr(t, f) for t in observed}, key=str) for f in fields}
    problems = []
    for f, v in distinct.items():
        if len(v) != 1:
            if None in v:
                problems.append(f"{f}: carried by some responses and not others {v}")
            else:
                problems.append(f"{f}: {len(v)} distinct values {v}")
    if not problems and expected is not None:
        only = C.StampRegimes(**{f: distinct[f][0] for f in fields})
        problems = regime_stamp_problems(only, expected, out_of_support=out_of_support)
    if problems:
        raise InspectServedError(
            f"{task}: the responses' regime stamps are not ONE value each equal to the leg's "
            f"contract row (expected {expected._asdict() if expected else None}): "
            + "; ".join(problems) + " (fit-support transfer plan §3.2; contract §2.5.11)")
    return {**{f: distinct[f][0] for f in fields},
            **{C.STAMP_REGIME_COUNT_FIELD[f]: (len(distinct[f]) if distinct[f][0] is not None else 0)
               for f in fields}}


def provider_of(model: Any) -> Optional[Any]:
    """The tinycap provider instance behind an inspect ``Model`` (its ``api``),
    when it is one — the object that carries ``regime_stamps`` and the route;
    ``None`` for a stub / a bare ``None`` model."""
    api = getattr(model, "api", None)
    if api is None or not hasattr(api, "regime_stamps") or not hasattr(api, "route"):
        return None
    return api


def stamp_route(route: Any) -> SimpleNamespace:
    """The four routing-table facts the proxy stamps a response with
    (``proxy_engine.STAMP_FIELDS``: ``virtual_model``, ``intervention_hash``,
    ``cell_id``, ``serving_identity``), read off a ``RouteEntry`` OR a
    table dict, as the attribute object ``proxy_engine.expected_stamps``
    reads. Refuses an entry missing any of them — an unverifiable route is
    not a route to serve on."""
    names = pe.ROUTE_STAMP_ATTRS
    vals = {}
    for name in names:
        try:
            vals[name] = _route_field(route, name)
        except (AttributeError, KeyError):
            vals[name] = None
    missing = [n for n in names if not isinstance(vals[n], str) or not vals[n]]
    if missing:
        raise InspectServedError(
            f"tinycap provider: route {route!r} lacks {missing} — the response stamps "
            "cannot be verified against it (E10 N2)")
    return SimpleNamespace(**vals)


#: The eval-level OPERATIONAL keys every Inspect eval this module launches
#: carries (``eval_async(max_retries=…, timeout=…)``). ONE mapping feeds the
#: call in :func:`run_one_task` AND the record every task files
#: (:func:`client_record`), so a store can never say one number while the run
#: used another. ``timeout`` here is inspect_ai 0.3.260's RETRY-LOOP budget —
#: ``Model.generate`` hands ``config.timeout`` to ``model_retry_config``
#: (``_retry.py``), where it is tenacity's stop-after-delay over a generate
#: call's attempts — and NOTHING ELSE: it never reaches a request. The
#: per-REQUEST bound is :func:`resolve_client_timeout`.
INSPECT_EVAL_OPERATIONAL: Mapping[str, int] = {
    "max_retries": C.INSPECT_MAX_RETRIES, "timeout": C.INSPECT_TIMEOUT_S}
#: The summary-payload / aggregate-row key the per-REQUEST client timeout is
#: recorded under (seconds — the number the target's OpenAI client was BUILT
#: with; null for a model with no tinycap provider, which had no client).
#: Recorded UNHASHED beside ``max_retries``: neither is store identity
#: (``contracts.tinycap_identity_fields`` names no operational key in any
#: shape), so every existing store's key is untouched.
CLIENT_TIMEOUT_FIELD = "client_timeout_s"


def resolve_client_timeout(client_timeout: Optional[float]) -> float:
    """The per-REQUEST timeout, in seconds, the provider's OpenAI client is
    built with — the bound on ONE HTTP round trip to the proxy.

    Read off the MATH-500 wave's retained ``.eval`` logs and usage
    ledgers: the provider passed none, so the OpenAI SDK's default applied
    (``openai._constants.DEFAULT_TIMEOUT``: 600 s read). A thinking-ON
    completion to the 16,384 cap on the eager served path takes 600–2,200 s;
    the SDK gave up at 600 s while the proxy finished the completion (the
    ledger holds it: HTTP 200, exactly 600 s apart per re-send), the SDK
    retried twice, Inspect twice more, the sample was re-run twice — up to 27
    requests per sample, none of the timed-out completions scored. That is
    differential censoring of the SLOW draws at the client layer, plus a GPU
    burning through phantom completions. ``contracts.INSPECT_TIMEOUT_S``
    existed but bounds only inspect_ai's retry loop (``GenerateConfig.timeout``
    → ``model_retry_config``), never a request.

    RELATIONSHIP TO THE PROXY: the client must not time out before the proxy's
    upstream does. The lens proxy waits ``contracts.PROXY_UPSTREAM_TIMEOUT_S``
    on vLLM, so the default here is that same budget
    (``INSPECT_TIMEOUT_S == int(PROXY_UPSTREAM_TIMEOUT_S)`` — one number every
    hop shares, so no hop can sit below the others unnoticed), and a value
    BELOW it is refused: a shorter client wait re-creates exactly this defect
    one hop further out. Longer is allowed. ``None`` = the default; a
    non-number, zero, a negative or a non-finite value refuses — a request
    with no bound is not the design either."""
    if client_timeout is None:
        return float(C.INSPECT_TIMEOUT_S)
    if isinstance(client_timeout, bool) or not isinstance(client_timeout, (int, float)):
        raise InspectServedError(
            f"tinycap provider: client_timeout={client_timeout!r} must be a number of "
            "seconds or None (the shared served-request budget, contracts.INSPECT_TIMEOUT_S)")
    value = float(client_timeout)
    if not math.isfinite(value) or value <= 0:
        raise InspectServedError(
            f"tinycap provider: client_timeout={client_timeout!r} must be a positive, finite "
            "number of seconds — a request with no bound is not the design (one budget "
            "every hop shares, contracts.PROXY_UPSTREAM_TIMEOUT_S)")
    floor = float(C.PROXY_UPSTREAM_TIMEOUT_S)
    if value < floor:
        raise InspectServedError(
            f"tinycap provider: client_timeout={value:g} s is below the proxy's upstream wait "
            f"(contracts.PROXY_UPSTREAM_TIMEOUT_S = {floor:g} s) — the client must not time "
            "out before the proxy's upstream does: a shorter client wait re-sends a "
            "completion the server is still finishing and censors the slow draws at the "
            "client (D153)")
    return value


def client_record(api: Any) -> Dict[str, Any]:
    """The OPERATIONAL keys a task's summary payload and aggregate row record
    beside its results: inspect_ai's retry count (the eval-level
    ``max_retries`` the run was launched with, :data:`INSPECT_EVAL_OPERATIONAL`
    — the same mapping the ``eval_async`` call reads) and the per-REQUEST
    client timeout the target model's OpenAI client was BUILT with, read off the
    provider (``api.client_timeout``, the attribute ``OpenAICompatibleAPI``
    builds its httpx and AsyncOpenAI clients from) — never off the constant, so
    the store says what the client did. A model with no tinycap provider (a
    stub, ``None``) had no client at all: ``None``, never a guessed default."""
    timeout = getattr(api, "client_timeout", None) if api is not None else None
    return {"max_retries": int(INSPECT_EVAL_OPERATIONAL["max_retries"]),
            CLIENT_TIMEOUT_FIELD: float(timeout) if timeout is not None else None}


def build_model(virtual_model: str, *, base_url: str, api_key: str,
                si: Optional[str], thinking: str, concurrency: int,
                max_retries: int = C.INSPECT_MAX_RETRIES,
                timeout_s: int = C.INSPECT_TIMEOUT_S,
                reasoning_parser: Optional[str] = None,
                route: Any = None,
                thinking_by_task: Optional[Mapping[str, str]] = None,
                sampling: Optional[C.SamplingRow] = None,
                max_tokens: Optional[int] = None,
                out_of_support: bool = False,
                stamp_policy: Optional[str] = None,
                header_located: bool = True,
                request_evidence_sink: Optional[Callable[..., Any]] = None,
                sampling_by_task: Optional[Mapping[str, C.SamplingRow]] = None,
                response_evidence_sink: Optional[Callable[..., Any]] = None):
    """A ``tinycap/<vmodel>`` Model bound to this container's proxy.

    ``sampling_by_task`` (the ``protocol`` mode): the TARGET's
    ``{row task: executed SamplingRow}`` map — the provider then
    selects every request's row by its task (``sampling`` is None); None on
    every other run, byte-identical to before.

    Built as an OBJECT, not a name string, because the grader is a
    DIFFERENT cell on the same proxy: a task that resolves its grader by
    name would get ``get_model``'s memoized default, i.e. the prompted
    target grading its own answers under its own SI. ``reasoning_parser``
    is the standup's composed parser (None = none), passed to the
    provider. ``route`` is the routing-table entry ``virtual_model`` is
    served on — REQUIRED, the provider verifies every response's stamps
    against it (E10 N2). ``thinking_by_task`` (PW20b-D) makes the arm a
    per-REQUEST fact read off the running task — the TARGET's map under one
    standup; never the grader's. ``sampling`` (U8) is the row every request
    of this model is drawn under — the TARGET's card row, ``GREEDY_SAMPLING``
    for the grader; None leaves the caller's config alone.

    ``max_tokens`` (E56 (3)) is THIS model's completion cap, carried on the
    model's own ``GenerateConfig``. It exists for the GRADER: the target's cap
    is a per-task decision made at eval level, while the grader is never the
    active model, so the eval config's cap never reaches it (inspect_ai
    0.3.260 ``Model._resolve_config``) and its own config is the only place a
    cap can live. None leaves the model uncapped, as every model was before.

    ``out_of_support`` (fit-support transfer) is the run's
    override, handed to the provider: under it a response with no regime stamp
    refuses; in support it is tolerated and recorded null (``on_response``).
    ``stamp_policy`` is the ``resumed_prefill_policy`` label this model's
    responses must wear (``contracts.expected_stamp_policy``, resolved by the
    driver off the standup record); ``None`` = the authorised experiment's
    contract row for the route's path.

    ``timeout_s`` is ONE budget handed to BOTH bounds: inspect_ai's
    retry-loop budget (``GenerateConfig.timeout`` → ``model_retry_config``,
    tenacity's stop-after-delay over a generate call's attempts, which never
    reaches a request) AND the per-REQUEST client timeout (``client_timeout``,
    which ``OpenAICompatibleAPI`` builds its httpx and AsyncOpenAI clients
    from). Before this the second was never passed, so the SDK's 600 s default
    bounded every request under a 7,200 s loop budget — see
    :func:`resolve_client_timeout`, which also refuses a value below the
    proxy's upstream wait."""
    if route is None:
        raise InspectServedError(
            f"build_model({virtual_model!r}): a route is required — the provider "
            "verifies every response's attestation stamps against it (E10 N2)")
    _register_provider()
    from inspect_ai.model import GenerateConfig, get_model

    cfg = GenerateConfig(max_connections=int(concurrency),
                         max_retries=int(max_retries),
                         timeout=int(timeout_s))
    if max_tokens is not None:
        cfg.max_tokens = int(max_tokens)
    return get_model(
        C.inspect_model_name(virtual_model),
        base_url=base_url, api_key=api_key,
        config=cfg,
        # the SAME number as the loop budget above, so the two bounds cannot
        # drift; the provider refuses one below the proxy's upstream wait
        client_timeout=float(timeout_s),
        memoize=False, si=si, thinking=thinking,
        reasoning_parser=reasoning_parser, route=route,
        thinking_by_task=(dict(thinking_by_task) if thinking_by_task else None),
        sampling=sampling, out_of_support=bool(out_of_support), stamp_policy=stamp_policy,
        header_located=bool(header_located), request_evidence_sink=request_evidence_sink,
        # bcb-bfcl-200-widen: the per-task rows of a `protocol` run (None otherwise)
        sampling_by_task=(dict(sampling_by_task) if sampling_by_task is not None else None),
        # the response side (None = no row binding, byte-identical)
        response_evidence_sink=response_evidence_sink)


# --------------------------------------------------------------------------- #
# preflight (P20): the names come off the routing table, never a keyboard     #
# --------------------------------------------------------------------------- #


def _route_field(route: Any, name: str) -> Any:
    return route.get(name) if isinstance(route, Mapping) else getattr(route, name)


def is_lens_route(route: Any) -> bool:
    """True when this leg is served through the steering lens."""
    from sisteer.capeval import manifest as cm

    return cm.parse_serving_identity(
        _route_field(route, "serving_identity")).path == "lens"


def _bfcl_requested(names: Sequence[str]) -> bool:
    """Any TOOL-CALLING task among ``names`` (``bfcl`` or the pinned
    ``bfcl_200``; :data:`TOOL_CALL_TASK_NAMES`) — the guard is about requests
    that carry ``tools``, not about one task's name."""
    return any(n in TOOL_CALL_TASK_NAMES for n in names)


def assert_route_can_serve(names: Sequence[str], route: Any, *,
                           agent_loop_entry: Optional[bool] = None,
                           tool_calls: Optional[bool] = None) -> None:
    """Refuse a task this ROUTE, as stood up, cannot serve correctly — at
    plan time, before a standup's worth of GPU minutes is spent.

    Today that is BFCL, whose requests carry ``tools``. Two facts of the
    STANDUP decide it, both read off its ``endpoint.json`` by the driver
    (never inferred here):

    * ``tool_calls`` — the standup composed ``--tool-calls`` (vLLM's
      auto-tool-choice + the family's tool parser). Without it the pinned
      vLLM 400s every tools-carrying request, on ANY route; a BFCL plan
      without it is refused.
    * ``agent_loop_entry`` — the proxy's Petri-facing entry
      (``RoutingProxy.petri_agent_loop_entry``; ``--petri-agent-loop``, on
      by default). On a LENS route it is what serves a tool TURN (an
      assistant tool call awaiting its result, or a tool result awaiting
      the model's continuation): the proxy declares ``petri_agent_loop`` for
      that structure when the render does not end at the header, and the
      worker steers from the LAST header — the completed gemma-E4B Petri
      audit's regime. Undeclared, such a turn refuses at the adapter as a
      contained 500. So BFCL on a lens route is served only under an armed
      entry, which supersedes both the earlier blanket refusal and the
      earlier prompted-only scoping. The client never declares the key
      itself: the declaration stays the served path's business, and the
      proxy verifies it against the render on every turn.

    What the single-turn BFCL categories the plan runs (V1 u V2, P9)
    actually SEND is one generation-prompt request per item — system /
    user (a live entry may carry a plain user / assistant history) plus
    ``tools``, the model's tool calls captured and AST-scored, never
    executed and never sent back (inspect_evals ``single_turn_solve``:
    ``generate(tool_calls="none")``). Those are ordinary tail renders that
    the lens serves undeclared like any generation task, with the response
    mask = header + every generated position (the tool call). The armed
    entry is therefore the regime a tool TURN would be served under
    (V3's ``tool_calls="loop"`` — excluded today) and the guard keeps the
    client's requirement equal to the served path's; it is not what makes
    the V1 u V2 requests serve (tests pin both facts against the proxy's
    classifier and the real gemma-4 render)."""
    if not _bfcl_requested(names):
        return
    cell = _route_field(route, "cell_id")
    if tool_calls is not True:
        raise InspectServedError(
            f"BFCL was requested on route {cell!r} but the standup composed no "
            f"--tool-calls (endpoint.json tool_calls={tool_calls!r}): every BFCL "
            "request carries `tools`, which the pinned vLLM 400s without "
            "--enable-auto-tool-choice + the family's tool parser; stand up with "
            "--tool-calls --manifest <family pin> (the launcher does when bfcl is "
            "among the controls)")
    if is_lens_route(route) and agent_loop_entry is not True:
        raise InspectServedError(
            f"BFCL was requested on lens route {cell!r} but the standup does not "
            f"declare agent-loop turns (endpoint.json agent_loop_entry="
            f"{agent_loop_entry!r}): a tool round-trip is an agent-loop turn, which "
            f"a lens route serves only when the request declares {C.AGENT_LOOP_KEY!r} "
            "— this client does not, and should not (the declaration is the served "
            "path's business: the proxy's Petri-facing entry attaches and verifies "
            "it). Start the proxy with the entry armed (the default; never "
            f"{C.FLAG_NO_PETRI_AGENT_LOOP} for a BFCL run), or run BFCL on the "
            "prompted cell")


def assert_live_agent_loop_entry(names: Sequence[str], route: Any, models_doc: Mapping[str, Any],
                                 *, declared: Optional[bool], base_url: str) -> None:
    """The LIVE half of the BFCL-on-lens gate (the way
    ``petri_runner.require_agent_loop_entry`` reads it): the proxy's own
    ``/v1/models`` identity block (``petri_proxy.petri_agent_loop_entry``)
    must say ``True`` for a lens route running BFCL, and must AGREE with
    what the standup record declared (``declared``, endpoint.json's
    ``agent_loop_entry``) — a record that vouches for one proxy while
    another answers is an N2 binding failure, refused before any task. A
    non-lens route, or a plan without BFCL, consults nothing."""
    if not _bfcl_requested(names) or not is_lens_route(route):
        return
    cell = _route_field(route, "cell_id")
    ident = models_doc.get(C.PROXY_IDENTITY_KEY)
    if not isinstance(ident, Mapping):
        raise InspectServedError(
            f"BFCL on lens route {cell!r}: {base_url}/models carries no "
            f"{C.PROXY_IDENTITY_KEY!r} identity block ({ident!r}) — cannot confirm the "
            "proxy declares agent-loop turns for lens routes (a build older than the "
            "multi-turn landing, or not the petri proxy); refusing before the first "
            "request")
    live = ident.get(C.PROXY_IDENTITY_AGENT_LOOP_ENTRY)
    if live is not True or live != declared:
        raise InspectServedError(
            f"BFCL on lens route {cell!r}: the live proxy's identity block reports "
            f"{C.PROXY_IDENTITY_AGENT_LOOP_ENTRY}={live!r} while the standup record "
            f"declares {C.AGENT_LOOP_ENTRY_KEY}={declared!r} — "
            + ("the record does not vouch for the proxy answering (N2); refusing"
               if live is True else
               "this proxy does not declare the agent-loop turns a lens route needs "
               "for BFCL's tool round-trips (an opted-out launch); refusing before a "
               "ledger full of contained 500s"))


def grader_route(routes: Sequence[Any]) -> Any:
    """The UNSTEERED base route's table entry — the grader for MATH's
    equivalence scorer and XSTest/OR-Bench's refusal grader (P4), and the
    route its provider verifies every grading response against (E10 N2).

    Read from the routing table's ``none`` entry. The repo spells its
    virtual model ``petri__base`` today, but the spelling is the table's
    business: a typed name is a class of bug the preflight below cannot
    even see, because a wrong name resolves to a Model object just fine
    and only fails when the proxy 404s, half an eval in."""
    for r in routes:
        if _route_field(r, "cell_id") == C.ROUTE_NONE_CELL:
            return r
    raise InspectServedError(
        f"routing table exports no {C.ROUTE_NONE_CELL!r} cell, so there is no "
        "unsteered base route to grade with — a steered cell grading its own "
        "answers is not a grader. Stand the server up with the base cell "
        "exported (capctrl_serving_up always does)")


def grader_virtual_model(routes: Sequence[Any]) -> str:
    """The grader's virtual model (:func:`grader_route`'s ``virtual_model``)."""
    return str(_route_field(grader_route(routes), "virtual_model"))


def get_models(*, base_url: str, api_key: Optional[str], timeout_s: float = 60.0
               ) -> Dict[str, Any]:
    """``GET /v1/models`` — the listing AND the proxy's ``petri_proxy``
    identity block, in one round trip (the usage ledger passes the call
    through verbatim, so the same document comes back through either URL)."""
    url = _canonical_v1_base(base_url) + "/models"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise InspectServedError(
            f"tinycap preflight: GET {url} failed ({type(e).__name__}: {e}) — "
            "refusing to start an evaluation against a proxy that cannot even "
            "list its models") from None
    if not isinstance(doc, Mapping):
        raise InspectServedError(
            f"tinycap preflight: GET {url} returned {type(doc).__name__}, not an object")
    return dict(doc)


def preflight_models(*, base_url: str, api_key: Optional[str],
                     required: Sequence[str], timeout_s: float = 60.0,
                     doc: Optional[Mapping[str, Any]] = None) -> List[str]:
    """``GET /v1/models`` (or the ``doc`` a caller already fetched through
    :func:`get_models`); every name in ``required`` must be listed.

    Runs BEFORE any task, so a mistyped or unexported virtual model costs
    one HTTP round trip rather than a GPU container's worth of 404s
    discovered at the first request."""
    url = _canonical_v1_base(base_url) + "/models"
    if doc is None:
        doc = get_models(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
    listed = [str(m.get("id")) for m in (doc.get("data") or [])
              if isinstance(m, Mapping)]
    missing = [n for n in dict.fromkeys(required) if n not in listed]
    if missing:
        raise InspectServedError(
            f"tinycap preflight: {url} does not list {missing} (listed: "
            f"{listed}) — every target and grader virtual model must resolve "
            "before a task runs; a missing name is a standup that exported "
            "different cells than this configuration asks for")
    return listed


# --------------------------------------------------------------------------- #
# task plans                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskPlan:
    """One row-task's worth of work: the Inspect Task, the item filter,
    and everything the adapter needs to turn its log into rows."""

    row_task: str
    name: str
    task: Any
    sample_ids: Optional[List[str]] = None
    paired_ids: Mapping[str, str] = field(default_factory=dict)
    samples_per_item: int = 1
    #: the eval-level ``limit`` (None when the plan selects by ``sample_ids``
    #: — Inspect refuses both together — or the task heads itself)
    limit: Optional[int] = None
    max_tokens: Optional[int] = None
    primary_scorer: Optional[str] = None
    #: the frozen FULL item count (R2c) — never narrowed by a smoke cap
    expected_n: Optional[int] = None
    #: the item count THIS run selects: ``expected_n`` unless a smoke cap
    #: narrowed it (``limit_plan``); None = unknown (treated as expected_n)
    n_run: Optional[int] = None
    #: the run's sampling row (U8), set at eval level AND applied by the
    #: provider: ``temperature`` always a number (the model's card value under
    #: ``--sampling card``, 0.0 under ``greedy`` — never None: the task's own
    #: config must not govern), ``top_p`` / ``top_k`` None when the row sets
    #: none (no key on the wire; vLLM's defaults apply)
    temperature: float = GREEDY_TEMPERATURE
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    #: which item set of a task with several this plan runs (GSM8K:
    #: ``paired200`` = the standard and the two-SI slot, ``full`` = the
    #: in-depth full set — PW3, roles flipped by U4(a) / PW17); None for a
    #: task with one item set. Recorded in the summary and the aggregate row.
    item_set: Optional[str] = None
    #: THIS task's thinking arm (PW20b-D): under ``per-task`` every plan of a
    #: run can carry a different one, and ``max_tokens`` above was chosen from
    #: it. The plan is where the arm lives because the arm and the cap must
    #: never disagree — they are one decision (:func:`_plan_task`).
    thinking: str = C.THINKING_OFF
    #: The cap a truncated sample of THIS task is re-requested at once (U5-3);
    #: 0 = no retry. Only MATH-500 (:data:`RETRY_ROW_TASK`) ever carries one.
    retry_max_tokens: int = 0
    #: The GRADER's completion cap for this task (E56 (3)): the number the
    #: grader MODEL was built with, or None for a task whose scorer calls no
    #: model (:data:`MODEL_GRADED_TASKS`). Recorded in the summary payload and
    #: the aggregate row, so a refusal rate always says what instrument
    #: produced it — an uncapped grader and a capped one are two instruments,
    #: and a pre-cap payload simply carries no key.
    grader_max_tokens: Optional[int] = None
    #: Where ``max_tokens`` came from —
    #: ``oos_override`` (the authorised experiment's raised ON cap,
    #: ``contracts.oos_max_tokens_on``), ``on_table`` (``MAX_TOKENS_ON``),
    #: ``on_floor`` (``THINKING_MAX_TOKENS_FLOOR``), ``off_table``
    #: (``MAX_TOKENS_OFF``) or ``offcap_override`` (the authorised comparator's
    #: raised OFF cap, ``contracts.off_cap_override_max_tokens``) — so a
    #: summary's cap is never a mystery number.
    cap_source: Optional[str] = None
    #: The think-block split the scorer applies (``contracts.THINK_SPLIT_MODES``),
    #: derived ONCE here beside the cap from the arm, the composed reasoning
    #: parser and the DECLARED render fact (``contracts.think_split_mode``) —
    #: the plan is where the arm lives, so the mode is decided where the cap is.
    think_split_mode: str = C.THINK_SPLIT_RECORD_ONLY
    #: The execution-sandbox facts of a task that runs its samples in one —
    #: provider, image digest, resources, the sandbox lifetime and the
    #: verifier timeout (``contracts.BCB_SANDBOX_*``); None for every task
    #: that executes nothing. Recorded in the manifest's route block; the
    #: image + verify timeout are hashed identity.
    sandbox: Optional[Mapping[str, Any]] = None
    #: The GPU half of the two-phase BigCodeBench contract — the task was
    #: built ``generation_only`` (its sandbox dropped AFTER every check ran;
    #: the stock scorer kept, never run), so :func:`run_one_task` passes
    #: ``score=False``, omits the sandbox kwargs, skips the transport install
    #: and, instead of filing, writes the generation sidecar + the pending
    #: marker into the row dir (``cpu_verify``). ``sandbox`` above STAYS the
    #: leg's facts (the filing tail the CPU verifier runs reads them); this
    #: flag alone says which phase the plan is. False = coupled,
    #: byte-identical.
    generation_only: bool = False
    #: the ladder's item SELECTION this plan was
    #: narrowed to (the capped items, log vocabulary), or None (the leg's whole
    #: pinned set). When set, ``sample_ids`` is exactly it and ``n_run`` its
    #: size; ``expected_n`` stays the frozen full count so the task ends PARTIAL.
    selection: Optional[List[str]] = None
    #: the DRAW-level rung —
    #: the ``rung_subset/2`` draws this plan re-draws, as sorted ``(item_id,
    #: sample_idx)`` pairs; None on every plan that is not a draw-subset rung
    #: (the item-level GSM8K rung included). When set, ``sample_ids`` is the
    #: DISTINCT items (the union of the draws), ``expected_n = n_run`` their
    #: count, ``expected_draws`` the draw count, and ``sample_ids_by_sample``
    #: names, per sample index, EXACTLY the items re-drawn at that trial — an
    #: index with none is never evaluated (no ``eval_async`` call). The runner
    #: files per-sample ``acc@<idx>`` rows + the outcome record + the summary and
    #: SKIPS the per-item mean rows by contract (a mean over one of an item's
    #: two trials is not avg@k).
    draw_subset: Optional[List[Tuple[str, int]]] = None
    sample_ids_by_sample: Optional[Dict[int, List[str]]] = None
    expected_draws: Optional[int] = None
    note: str = ""
    #: the row's OPTIONAL terms, None on
    #: every legacy row (the summary carries them only when set)
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    #: under ``protocol`` the benchmark's DECLARED
    #: reference temperature for THIS task (``contracts.ProtocolRow.
    #: protocol_temperature``) — recorded beside the EXECUTED ``temperature`` in
    #: the summary and the aggregate row (never hashed); None under every other
    #: mode (the summary then carries no such key — byte-identical)
    protocol_temperature: Optional[float] = None
    #: the FAMILY whose render fact fed this plan's split mode, and that fact's
    #: evidence status; None when planned without one
    family: Optional[str] = None
    render_evidence_status: Optional[str] = None
    #: the registered name of the scorer whose value is the row's ``acc``
    #: (``primary_scorer``) — recorded so a wrap under the inner name is visible
    scorer_wrapper: Optional[str] = None

    @property
    def n_expected_run(self) -> Optional[int]:
        return self.n_run if self.n_run is not None else self.expected_n

    def ids_for(self, sample_idx: int) -> Optional[List[str]]:
        """The item ids sample index ``sample_idx`` evaluates: under a draw
        subset exactly the items re-drawn at that trial (``[]`` = the trial is
        skipped); otherwise ``sample_ids`` (every trial draws the same items)."""
        if self.sample_ids_by_sample is not None:
            return list(self.sample_ids_by_sample.get(int(sample_idx), []))
        return list(self.sample_ids) if self.sample_ids is not None else None


#: :func:`plan_facts` leaves out exactly these :class:`TaskPlan` fields — the built
#: inspect ``Task`` object (rebuilt by the CPU verifier from the pin + the log's ids).
PLAN_FACTS_EXCLUDED: Tuple[str, ...] = ("task",)


def plan_facts(plan: TaskPlan) -> Dict[str, Any]:
    """Every :class:`TaskPlan` field but the ``Task`` object, JSON-clean — what the
    generation sidecar carries so
    the CPU verifier files the task under the SAME plan the GPU phase generated it
    under (the cap, the sampling row, the arm, the split mode, the note, the
    counts …) without re-deriving any of it from a config it does not have.
    Refuses a plan the two-phase contract does not cover (a draw subset, several
    samples per item, a truncation retry): BigCodeBench is k = 1, one pass."""
    if plan.draw_subset is not None or plan.sample_ids_by_sample is not None:
        raise InspectServedError(f"{plan.row_task}: a draw-subset plan has no generation sidecar (plan §2.2.3 covers "
                                 "the one-pass BigCodeBench leg alone)")
    if int(plan.samples_per_item) != 1 or plan.retry_max_tokens:
        raise InspectServedError(f"{plan.row_task}: samples_per_item={plan.samples_per_item}, retry_max_tokens="
                                 f"{plan.retry_max_tokens} — the two-phase contract covers a k = 1, no-retry task")
    out: Dict[str, Any] = {}
    for f in fields(TaskPlan):
        if f.name in PLAN_FACTS_EXCLUDED:
            continue
        v = getattr(plan, f.name)
        if isinstance(v, Mapping):
            v = {str(k): x for k, x in v.items()}
        elif isinstance(v, (list, tuple)):
            v = list(v)
        out[f.name] = v
    # JSON-clean by construction: the round trip must give the same plan back
    json.dumps(out, allow_nan=False)
    return out


def plan_from_facts(facts: Mapping[str, Any], *, task: Any) -> TaskPlan:
    """The inverse of :func:`plan_facts`: a :class:`TaskPlan` over ``task`` with every
    other field from the sidecar — EXACTLY the field set (a missing or unknown key
    refuses by name: a sidecar written by another tree is not this plan)."""
    names = [f.name for f in fields(TaskPlan) if f.name not in PLAN_FACTS_EXCLUDED]
    missing = sorted(set(names) - set(facts))
    unknown = sorted(set(facts) - set(names))
    if missing or unknown:
        raise InspectServedError(f"generation sidecar plan facts are not this tree's TaskPlan: missing {missing}, "
                                 f"unknown {unknown}")
    kw = {k: facts[k] for k in names}
    if kw.get("draw_subset") is not None:
        kw["draw_subset"] = [tuple(x) for x in kw["draw_subset"]]
    if kw.get("sample_ids_by_sample") is not None:
        kw["sample_ids_by_sample"] = {int(k): list(v) for k, v in kw["sample_ids_by_sample"].items()}
    return TaskPlan(task=task, **kw)


def load_repo_task(module_name: str):
    """Import a thin in-repo Inspect task by PATH from
    ``configs/capeval/inspect_tasks/`` and return its task function."""
    path = INSPECT_TASKS_DIR / f"{module_name}.py"
    if not path.is_file():
        raise InspectServedError(
            f"in-repo Inspect task {path} does not exist — the thin tasks "
            "(P3/P8) ship with this branch; a missing one is a broken "
            "checkout, not a reason to fall back to a stock task with a "
            "different item set")
    spec = importlib.util.spec_from_file_location(
        f"sisteer_inspect_tasks.{module_name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    try:
        return getattr(mod, module_name)
    except AttributeError:
        raise InspectServedError(
            f"{path} defines no task named {module_name!r}") from None


# -- MATH-500 selection (P2) ------------------------------------------------ #


def math500_problems(*, revision: Optional[str] = None) -> List[str]:
    """The canonical 500 MATH-500 problem strings, in dataset order."""
    from datasets import load_dataset

    ds = load_dataset(C.MATH500_DATASET, split="test", revision=revision)
    return [str(r["problem"]) for r in ds]


def math500_sample_ids(*, revision: Optional[str] = None) -> List[str]:
    """The inspect_evals sample ids of the 500 canonical problems.

    ``math``'s ``record_to_sample`` ids a sample as
    ``math_<md5(problem)[:8]>``, so the canonical 500 are addressable
    INSIDE inspect_evals' own task by ``sample_id=`` — no re-implemented
    task, no re-implemented scorers, and the 500 are the published 500
    rather than ``--limit 500`` off an unseeded shuffle."""
    return [C.math500_sample_id(p) for p in math500_problems(revision=revision)]


def verify_math500_selection(sample_ids: Sequence[str]) -> Dict[str, Any]:
    """Cross-check the selection against the task's OWN dataset ids.

    Returns ``{matched, missing, n_source}``. A shortfall is a refusal at
    plan time, not a quiet short run: a "MATH-500" number computed over
    486 problems is not MATH-500, and the failure is invisible in the
    aggregate."""
    from inspect_evals.math.math import DATASET_PATH, MATH_DATASET_REVISION
    from datasets import load_dataset

    ds = load_dataset(DATASET_PATH, split="test", name="default",
                      revision=MATH_DATASET_REVISION)
    have = {C.math500_sample_id(str(r["problem"])) for r in ds}
    missing = [s for s in sample_ids if s not in have]
    return {"matched": len(sample_ids) - len(missing), "missing": missing,
            "n_source": len(have)}


# -- pairing for the comparison tasks --------------------------------------- #


def _paired_by_text(rows: Sequence[Mapping[str, Any]], *, text_key: str,
                    sample_id: Callable[[str], str]) -> Dict[str, str]:
    """``{inspect sample id -> our item_id}`` keyed on the prompt TEXT.

    Text, not index: our corpora and Inspect's are different snapshots of
    the same benchmark (natolambert's XSTest copy vs walledai's;
    gsm8k-platinum vs openai/gsm8k), so an index join would silently pair
    different questions. An item Inspect has and we do not simply gets no
    ``paired_id``, and the comparison reports it as unmatched."""
    out: Dict[str, str] = {}
    for r in rows:
        out[sample_id(str(r[text_key]))] = str(r["item_id"])
    return out


# -- the OR-Bench battery item set (U10 item 4) --------- #


def or_bench_selected_rows(spec: Mapping[str, Any], *, pins: Any = None,
                           repo_root: Path = REPO_ROOT
                           ) -> Tuple[List[dict], Optional[List[int]]]:
    """The pinned OR-Bench rows an item-set spec selects, and the position
    list to hand the thin task (``None`` = the whole corpus).

    The spec is ``contracts.or_bench_item_set``'s — the one the run HASHED —
    so the corpus is resolved from exactly the positions the identity claims.
    The corpus size is re-checked against ``contracts.OR_BENCH_FULL_N`` here,
    at PLAN time and before any GPU minute: a frozen set of POSITIONS says
    nothing about which prompts ran if the rows underneath them moved, and the
    launcher derived those positions without ever seeing the file."""
    from sisteer.capeval.tiers.judged import or_bench_rows

    rows = or_bench_rows(pins if pins is not None else _pins(repo_root))
    if len(rows) != C.OR_BENCH_FULL_N:
        raise InspectServedError(
            f"or_bench_hard_1k: the pinned corpus has {len(rows)} rows, not the "
            f"{C.OR_BENCH_FULL_N} contracts.OR_BENCH_FULL_N froze the battery subset "
            "against — the positions in this run's identity would select different "
            "prompts than the ones they were drawn from. Re-pin the corpus and re-derive "
            "the subset in the same change; refusing to run.")
    if int(spec["of"]) != len(rows):
        raise InspectServedError(
            f"or_bench_hard_1k: the item-set spec was drawn over {spec['of']} items but the "
            f"pinned corpus has {len(rows)} — refusing a subset drawn against another corpus")
    positions = spec.get("positions")
    if positions is None:
        return list(rows), None
    want = sorted(int(i) for i in positions)
    if len(want) != int(spec["n"]):
        raise InspectServedError(
            f"or_bench_hard_1k: the spec says n={spec['n']} but carries {len(want)} positions")
    return [rows[i] for i in want], want


def or_bench_item_ids(spec: Mapping[str, Any], *, pins: Any = None,
                      repo_root: Path = REPO_ROOT) -> List[str]:
    """The Inspect sample ids an item-set spec resolves to, in position order —
    what the manifest records unhashed beside the hashed positions, so a reader
    can check by eye that two cells ran the same prompts."""
    rows, _ = or_bench_selected_rows(spec, pins=pins, repo_root=repo_root)
    return [C.inspect_sample_id("or_bench_hard_1k", str(r["prompt"])) for r in rows]


@functools.lru_cache(maxsize=4)
def _pins(repo_root: Path = REPO_ROOT):
    """pins.yaml — the pinned dataset revisions every corpus loader reads,
    parsed once per repo root (a run plans up to six tasks off it)."""
    from sisteer.capeval.corpora import load_pins

    return load_pins(Path(repo_root) / "configs" / "capeval" / "pins.yaml")


def _dataset_ids(task: Any) -> Optional[set]:
    """The sample ids of a built Inspect ``Task``'s dataset (a sized
    ``Dataset``), or None when the object exposes none (a test stub)."""
    ds = getattr(task, "dataset", None)
    if ds is None:
        return None
    try:
        return {str(s.id) for s in ds}
    except Exception:  # noqa: BLE001 — not a Dataset: no frozen count
        return None


def _dataset_size(task: Any) -> Optional[int]:
    ids = _dataset_ids(task)
    return len(ids) if ids is not None else None


def verify_selection(task: Any, sample_ids: Sequence[str]) -> Dict[str, Any]:
    """Cross-check a ``sample_id`` selection against the built task's OWN
    dataset ids (the HellaSwag twin of :func:`verify_math500_selection`):
    ``{matched, missing, n_source}``. A miss is a plan-time refusal — with
    Inspect's ``sample_id`` filter an unmatched id neither warns nor
    errors, so a wrong pairing would otherwise run silently on fewer items
    (or on none: the pairing once read the tiny dataset's unlabelled
    ``test`` split, and 0 of its ids exist in the task)."""
    have = _dataset_ids(task)
    if have is None:
        raise InspectServedError(
            f"cannot verify a sample_id selection: {type(task).__name__} exposes no dataset")
    missing = [s for s in sample_ids if s not in have]
    return {"matched": len(sample_ids) - len(missing), "missing": missing,
            "n_source": len(have)}


def limit_plan(plan: TaskPlan, inspect_limit: int) -> TaskPlan:
    """Apply a smoke cap (``--inspect-limit``) to a plan — narrowing the RUN,
    never the frozen full count: ``expected_n`` stays the full set and
    ``n_run`` says what this run selects, so a capped run ends with a
    PARTIAL marker (``n_expected_full`` / ``n_items``) rather than a done
    one. Inspect refuses ``sample_id`` together with ``limit``
    (``eval.py``: "You cannot specify both"), so a plan that selects by id
    takes the FIRST N of its planned ids and passes no ``limit``; a plan
    over the task's own order passes ``limit=N``; a plan whose builder
    already narrowed itself (``n_run`` set: the thin OR-Bench task heads
    its own pinned order) is left as built."""
    if plan.n_run is not None:
        return plan
    n = int(inspect_limit or 0)
    if n <= 0:
        return replace(plan, n_run=plan.expected_n)
    if plan.sample_ids is not None:
        ids = list(plan.sample_ids)[:n]
        return replace(plan, sample_ids=ids, limit=None, n_run=len(ids))
    n_run = min(n, plan.expected_n) if plan.expected_n is not None else n
    return replace(plan, limit=n, n_run=n_run)


def bfcl_categories() -> List[str]:
    """The single-turn BFCL categories the plan passes to inspect_evals'
    ``bfcl()``: V1 u V2 (P9 — V3 multi-turn stays out) minus
    :data:`BFCL_EXCLUDED_CATEGORIES`, sorted. Cross-checked against the task
    module's ``DEFAULT_CATEGORIES`` — inspect_evals' own statement of every
    category ``bfcl()`` accepts — so the names it leaves out of V1 u V2 must
    be EXACTLY our excluded set; a disagreement (a pin bump moved the
    refusal table) is a refusal naming both, never a ValueError inside the
    task after the standup."""
    import importlib

    from inspect_evals.bfcl.utils.task_categories import (V1_CATEGORIES,
                                                          V2_CATEGORIES)

    accepted = set(importlib.import_module("inspect_evals.bfcl.bfcl").DEFAULT_CATEGORIES)
    single_turn = set(V1_CATEGORIES) | set(V2_CATEGORIES)
    refused = single_turn - accepted
    if refused != set(BFCL_EXCLUDED_CATEGORIES):
        raise InspectServedError(
            f"BFCL: inspect_evals' bfcl() accepts every V1 u V2 category except "
            f"{sorted(refused)}, but this module excludes {sorted(BFCL_EXCLUDED_CATEGORIES)} "
            "— the pinned task's category table moved; re-pin BFCL_EXCLUDED_CATEGORIES")
    return sorted(single_turn - set(BFCL_EXCLUDED_CATEGORIES))


def plan_task(name: str, *, grader_model: Any, thinking: str,
              inspect_limit: int = 0, samples: int = C.MATH500_SAMPLES_PER_ITEM,
              repo_root: Path = REPO_ROOT,
              verify_math500: bool = True, verify_pairs: bool = True,
              retry_max_tokens: int = 0,
              sampling: Optional[C.SamplingRow] = None,
              samples_per_item: Optional[int] = None,
              or_bench_item_set: Optional[Mapping[str, Any]] = None,
              grader_max_tokens: int = GRADER_MAX_TOKENS,
              out_of_support: bool = False,
              prompt_opened_think: bool = False,
              reasoning_parser: Optional[str] = None,
              off_cap_override: bool = False,
              family: Optional[str] = None,
              require_certified_fact: bool = True,
              oos_experiment: Optional[str] = None,
              on_cap_override: Optional[Mapping[str, int]] = None,
              item_subset: Optional[Sequence[str]] = None,
              cap_override: Optional[int] = None,
              selection: Optional[Sequence[str]] = None,
              off_cap_setter: Optional[Mapping[str, int]] = None,
              item_subset_draws: Optional[Sequence[Sequence[object]]] = None,
              protocol_temperature: Optional[float] = None,
              generation_only: bool = False) -> TaskPlan:
    """The :class:`TaskPlan` for one Inspect task name, with the smoke cap
    applied by :func:`limit_plan` (the full count is kept; the run is
    narrowed). ``verify_pairs`` cross-checks an id-paired selection
    (HellaSwag) — and the two GSM8K counts: the in-depth full split's
    ``contracts.GSM8K_FULL_N`` and the STANDARD paired set's matched count
    ``contracts.GSM8K_PAIRED_N`` (the slot's acceptance rule) — against the
    built task's dataset, like ``verify_math500`` does for MATH-500.
    ``thinking`` is THIS task's arm (never a mode token): it picks the cap
    and is recorded on the plan. ``retry_max_tokens`` is the MATH-500
    truncation-retry cap (0 = no retry, the default since the one-pass
    instrument); it is attached only to
    :data:`RETRY_ROW_TASK` — naming it on any other task is silently no-op
    by construction, so the retry can never appear on an instrument it was
    not requested on — and a value at or below the pass cap refuses.

    ``sampling`` (U8, PW-sampling) is the run's :class:`contracts.SamplingRow`
    — the model's card row under ``--sampling card``, ``GREEDY_SAMPLING``
    when None (the pre-U8 draw) — and ``samples_per_item`` THIS task's k from
    the run's ``samples_by_task`` map; when None the legacy rule applies
    (``samples`` for MATH-500, 1 for every other task), so a caller that
    predates the map plans exactly what it always did.

    ``or_bench_item_set`` (U10 item 4, PW-battery) is the run's HASHED
    OR-Bench spec (``contracts.or_bench_item_set``); None = the battery
    default, the frozen seeded 200-item subset. Ignored by every other task.

    ``grader_max_tokens`` (E56 (3)) is the cap the caller BUILT the grader
    model with; it is recorded on the plan of every
    :data:`MODEL_GRADED_TASKS` task and ignored by the rest. It is threaded
    rather than re-read from :data:`GRADER_MAX_TOKENS` here so the number the
    grader was built with and the number the summary reports cannot drift
    (``run_inspect_tasks`` passes ONE value to both).

    ``out_of_support`` / ``prompt_opened_think`` / ``reasoning_parser`` (S2.1):
    the override in force, the experiment's DECLARED render fact and the
    composed parser — together they pick the thinking-ON cap
    (``contracts.oos_max_tokens_on`` for the tasks it names) and the think-split
    mode (``contracts.think_split_mode``). The defaults plan the in-support run
    exactly as before.

    gsm8k-paired200-paper-objects: ``on_cap_override`` is the bounded ON-cap
    setter (:func:`task_max_tokens`); ``item_subset``
    narrows the STANDARD paired GSM8K plan to a rung's eligible ids — after the
    frozen-200 check, ``expected_n = n_run = len(ids)`` so the rung's
    completeness is subset-sized — and refuses on every other task.

    ``cap_override`` / ``selection`` (the truncation
    ladder — the two pinned legs only): the 16,384 completion cap that replaces
    the table's (``cap_source`` = ``cap_override``) and the capped items (log
    vocabulary) the plan is narrowed to (``sample_ids`` = exactly them, ``n_run``
    = their count, ``expected_n`` the frozen 200 — the task ends PARTIAL by
    contract). A selection without an override, or either on a task outside
    the two legs, refuses."""
    plan = _plan_task(name, grader_model=grader_model, thinking=thinking,
                      inspect_limit=inspect_limit, samples=samples,
                      repo_root=repo_root, verify_math500=verify_math500,
                      verify_pairs=verify_pairs, retry_max_tokens=retry_max_tokens,
                      sampling=sampling, samples_per_item=samples_per_item,
                      or_bench_item_set=or_bench_item_set,
                      grader_max_tokens=grader_max_tokens,
                      out_of_support=out_of_support, prompt_opened_think=prompt_opened_think,
                      reasoning_parser=reasoning_parser, off_cap_override=off_cap_override,
                      family=family, require_certified_fact=require_certified_fact,
                      oos_experiment=oos_experiment, on_cap_override=on_cap_override,
                      item_subset=item_subset,
                      cap_override=cap_override, selection=selection,
                      off_cap_setter=off_cap_setter, item_subset_draws=item_subset_draws,
                      protocol_temperature=protocol_temperature,
                      generation_only=generation_only)
    return limit_plan(plan, inspect_limit)


def task_max_tokens(name: str, *, thinking: str, out_of_support: bool = False,
                    off_cap_override: bool = False,
                    oos_experiment: Optional[str] = None,
                    on_cap_override: Optional[Mapping[str, int]] = None,
                    off_cap_setter: Optional[Mapping[str, int]] = None) -> Tuple[Optional[int], str]:
    """The completion cap of Inspect task ``name`` under arm ``thinking`` and
    where it came from — the ONE cap rule: under ``on`` and
    ``out_of_support`` the authorised experiment's raised cap for the tasks it
    names (``contracts.oos_max_tokens_on``, ``oos_override``); else the ON
    table (``on_table``) or the ON floor (``on_floor``); under ``off`` and
    ``off_cap_override`` the authorised comparator's raised cap for the tasks
    IT names (``contracts.off_cap_override_max_tokens``, ``offcap_override``);
    else under ``off`` the OFF table (``off_table``; None = the task pins its
    own). The override is an OFF fact: naming it under the ON arm refuses.
    :func:`_plan_task` applies it to every plan and the served driver hashes
    the same answer per row task into ``max_tokens_by_task``, so the
    identity's cap and the executed cap are one derivation.

    ``on_cap_override`` is the operator's bounded SETTER
    (``contracts.FLAG_ON_CAP_BY_TASK``, ``{Inspect task name: cap}``),
    consulted AFTER ``oos_override`` — a fitted cell's tuple cap wins and the
    setter must EQUAL it (one argv shape for every row) — and BEFORE
    ``on_table`` / ``on_floor``, within ``contracts.check_on_cap_setter``'s
    bounds (an allowed task and value, the ON arm, above the table / floor
    resolution, at or below the ceiling); ``cap_source``
    ``contracts.CAP_SOURCE_ON_CAP_EXPLICIT``. Under the OFF arm a setter naming
    the task refuses. ``None`` / a map not naming ``name`` = exactly the rule
    above (every existing plan byte-identical).

    ``off_cap_setter`` is the OFF-arm twin
    (``contracts.FLAG_OFF_CAP_BY_TASK``, ``{Inspect task name: cap}`` — the
    IFEval ladder's 4,096 / 8,192 rungs), consulted under ``off`` AFTER the
    authorised comparator's ``off_cap_override`` (the two are mutually exclusive
    by name: both naming one task refuses) and BEFORE ``MAX_TOKENS_OFF``, within
    ``contracts.check_off_cap_setter``'s bounds (an allowed task and value, the
    OFF arm, above the table, at or below the ceiling); ``cap_source``
    ``contracts.CAP_SOURCE_OFF_CAP_EXPLICIT``. Under the ON arm a setter naming
    the task refuses. ``None`` / a map not naming ``name`` = the rule above."""
    if name not in C.INSPECT_TASK_NAMES:
        raise InspectServedError(
            f"{name!r} is not a tinycap Inspect task: {list(C.INSPECT_TASK_NAMES)}")
    setter = on_cap_override.get(name) if on_cap_override else None
    off_setter = off_cap_setter.get(name) if off_cap_setter else None
    if thinking == C.THINKING_NOT_AVAILABLE and off_setter is None and name != C.IFEVAL_TASK_NAME:
        # (the IFEval battery's no-switch rows are that lane's POLICY — its `off` resolves to `not_available` and the cap is the
        # OFF table's or its OFF setter's, pinned by tests/test_ifeval_no_switch_llama.py — so the IFEval task takes the OFF path below)
        # llama-paper-rows (P7): a NO-SWITCH arm has NO table of its own — the OFF table's cap is the retired
        # instrument's and the ON table / floor a thinking budget — so a bounded setter is REQUIRED and IS the cap:
        # the ON setter here (an allowed task and value, at or below the ceiling; contracts.check_on_cap_setter), or the
        # OFF-arm rung setter below (the IFEval battery's `--off-cap-by-task`, which serves its no-switch llama rows through
        # the OFF path); the two switch-arm OVERRIDES refuse under it by name
        if out_of_support or off_cap_override:
            raise InspectServedError(
                f"{name}: {'out_of_support' if out_of_support else 'off_cap_override'} was passed under the "
                f"{thinking!r} arm — a family with no thinking switch has no render to be out of support of and no OFF "
                "cap to override; refusing to plan on it")
        if setter is None:
            raise InspectServedError(
                f"{name}: the {thinking!r} arm (a no-thinking-channel family) has no cap table — state the cap with "
                f"{C.FLAG_ON_CAP_BY_TASK} {name}=<cap> (the setter is REQUIRED and is the cap; P7)")
        try:
            C.check_on_cap_setter(name, int(setter), arm=thinking, resolved_cap=None, cap_source="none")
        except C.OnCapSetterRefused as e:
            raise InspectServedError(f"{name}: {e}") from None
        return int(setter), C.CAP_SOURCE_ON_CAP_EXPLICIT_NOT_AVAILABLE
    if off_setter is not None and C.enable_thinking(thinking):
        # the OFF setter is a thinking-OFF cap: naming an ON task refuses by name
        try:
            C.check_off_cap_setter(name, int(off_setter), arm=thinking, resolved_cap=MAX_TOKENS_ON.get(name),
                                   cap_source="on_table")
        except C.OffCapSetterRefused as e:
            raise InspectServedError(f"{name}: {e}") from None
        raise AssertionError("unreachable: check_off_cap_setter admits no ON arm")   # pragma: no cover
    if C.enable_thinking(thinking):
        if off_cap_override:
            raise InspectServedError(
                f"{name}: off_cap_override is the authorised comparator's thinking-OFF cap fact "
                f"(contracts.AUTHORISED_OFF_CAP_COMPARATOR, modes {list(C.AUTHORISED_OFF_CAP_COMPARATOR.modes)}) "
                f"and was passed under the ON arm {thinking!r} — there is no ON cap it overrides; "
                "refusing to plan on it")
        # the cap map of the MATCHED registry tuple when the driver names one;
        # the parent's tuple otherwise
        oos_cap = C.oos_max_tokens_on(name, oos_experiment) if out_of_support else None
        if oos_cap is not None:
            if setter is not None and int(setter) != int(oos_cap):
                raise InspectServedError(
                    f"{name}: {C.FLAG_ON_CAP_BY_TASK} {name}={int(setter)} but the matched authorised tuple "
                    f"raises the thinking-ON cap to {int(oos_cap)} — on a fitted cell the setter must EQUAL the "
                    "tuple's cap (one argv shape for fitted and in-support rows; plan §3.1)")
            return oos_cap, "oos_override"
        table_cap, table_source = ((MAX_TOKENS_ON[name], "on_table") if name in MAX_TOKENS_ON
                                   else (THINKING_MAX_TOKENS_FLOOR, "on_floor"))
        if setter is not None:
            try:
                C.check_on_cap_setter(name, int(setter), arm=thinking, resolved_cap=table_cap,
                                      cap_source=table_source)
            except C.OnCapSetterRefused as e:
                raise InspectServedError(f"{name}: {e}") from None
            return int(setter), C.CAP_SOURCE_ON_CAP_EXPLICIT
        return table_cap, table_source
    if setter is not None:
        # the setter is a thinking-ON cap: naming an OFF task refuses by name
        try:
            C.check_on_cap_setter(name, int(setter), arm=thinking, resolved_cap=MAX_TOKENS_OFF.get(name),
                                  cap_source="off_table")
        except C.OnCapSetterRefused as e:
            raise InspectServedError(f"{name}: {e}") from None
        raise AssertionError("unreachable: check_on_cap_setter admits no OFF arm")   # pragma: no cover
    if off_cap_override:
        cap = C.off_cap_override_max_tokens(name)
        if cap is not None:
            if off_setter is not None:
                raise InspectServedError(
                    f"{name}: {C.FLAG_OFF_CAP_BY_TASK} {name}={int(off_setter)} beside the authorised comparator's "
                    f"raised OFF cap {int(cap)} — the two OFF overrides are mutually exclusive by name; a rung "
                    "run is not a comparator run")
            return cap, "offcap_override"
    if off_setter is not None:
        # the OFF-arm rung setter, bounded
        try:
            C.check_off_cap_setter(name, int(off_setter), arm=thinking, resolved_cap=MAX_TOKENS_OFF.get(name),
                                   cap_source="off_table")
        except C.OffCapSetterRefused as e:
            raise InspectServedError(f"{name}: {e}") from None
        return int(off_setter), C.CAP_SOURCE_OFF_CAP_EXPLICIT
    return MAX_TOKENS_OFF.get(name), "off_table"


def _plan_task(name: str, *, grader_model: Any, thinking: str,
               inspect_limit: int, samples: int, repo_root: Path,
               verify_math500: bool, verify_pairs: bool = True,
               retry_max_tokens: int = 0,
               sampling: Optional[C.SamplingRow] = None,
               samples_per_item: Optional[int] = None,
               or_bench_item_set: Optional[Mapping[str, Any]] = None,
               grader_max_tokens: int = GRADER_MAX_TOKENS,
               out_of_support: bool = False,
               prompt_opened_think: bool = False,
               reasoning_parser: Optional[str] = None,
               off_cap_override: bool = False,
               family: Optional[str] = None,
               require_certified_fact: bool = True,
               oos_experiment: Optional[str] = None,
               on_cap_override: Optional[Mapping[str, int]] = None,
               item_subset: Optional[Sequence[str]] = None,
               cap_override: Optional[int] = None,
               selection: Optional[Sequence[str]] = None,
               off_cap_setter: Optional[Mapping[str, int]] = None,
               item_subset_draws: Optional[Sequence[Sequence[object]]] = None,
               protocol_temperature: Optional[float] = None,
               generation_only: bool = False) -> TaskPlan:
    if name not in C.INSPECT_TASK_NAMES:
        raise InspectServedError(
            f"{name!r} is not a tinycap Inspect task: {list(C.INSPECT_TASK_NAMES)}")
    if generation_only and name not in SANDBOXED_TASK_NAMES:
        # the generation-only build is the SANDBOXED legs' contract alone — on any
        # other task the knob would silently do nothing, which is the worse failure
        raise InspectServedError(
            f"{name}: generation_only is the sandboxed legs' two-phase contract ({list(SANDBOXED_TASK_NAMES)}); "
            f"{name!r} verifies nothing in a sandbox, so there is no verification to defer")
    if item_subset_draws is not None:
        # the DRAW list rides beside the item list of the SAME subset, and only
        # IFEval's k = 2 rung (`rung_subset/2`) has one — GSM8K's rung is
        # item-level by contract
        if item_subset is None:
            raise InspectServedError(
                f"{name}: item_subset_draws given without item_subset — a draw subset narrows the ITEMS of "
                f"a {C.FLAG_ITEM_SUBSET} rung per trial; it is never a subset of its own")
        if name != C.IFEVAL_TASK_NAME:
            raise InspectServedError(
                f"{name}: a draw-level subset ({C.ITEM_SUBSET_SCHEMA_V2}) is the k = {C.IFEVAL_SAMPLES_PER_ITEM} "
                f"IFEval rung's contract alone; {name!r} re-draws whole items ({C.ITEM_SUBSET_SCHEMA})")
    if item_subset is not None and name not in C.ITEM_SUBSET_TASKS:
        # a RUNG run generates ONLY the listed ids of the paired instrument —
        # every other task of such a run refuses (a rung run with another task
        # in it is not a rung run); IFEval is the second rung task
        raise InspectServedError(
            f"{name}: {C.FLAG_ITEM_SUBSET} narrows a rung task's plan alone "
            f"({list(C.ITEM_SUBSET_TASKS)}) — a rung run plans that one task; refusing to plan {name!r} "
            "beside a rung subset")
    if off_cap_override and out_of_support:
        # the truly contradictory request — "plan the
        # ON-render out-of-support cap AND the OFF-render comparator cap for one
        # task". The ONLY producer is a DIRECT caller of plan_task /
        # plans_for_atoms / run_inspect_tasks passing both flags (a test, or a
        # future driver bug): the production driver's startup exclusivity check
        # makes `oos_run and offcap_run` impossible, and it hands the planner
        # `out_of_support=oos_run` (the RENDER authorisation), never the record
        # posture `held_to_record`, which run_inspect_tasks derives for itself.
        raise InspectServedError(
            f"{name}: off_cap_override and out_of_support were both passed to the task planner — "
            "the two named overrides are mutually exclusive (contracts.check_override_flags_exclusive) "
            "and only a direct caller of plan_task / plans_for_atoms / run_inspect_tasks can produce "
            "the pair; the driver passes the render authorisation alone (`out_of_support=oos_run`)")
    # bcb-bfcl-200 §S2b: the ladder's two knobs are for the two pinned legs
    # only, always together (a selection re-run at the SAME cap would re-draw
    # the capped items and call it a ladder)
    ladder_names = tuple(_task_name(t) for t in C.LADDER_ROW_TASKS)
    if (cap_override is not None or selection is not None) and name not in ladder_names:
        raise InspectServedError(
            f"{name}: cap_override / selection are the truncation ladder's knobs for {list(ladder_names)} "
            "only (plan §S2b) — refusing them on another task")
    if selection is not None and cap_override is None:
        raise InspectServedError(
            f"{name}: an item selection without a cap override is not a ladder rung — the capped items "
            "re-drawn at the same cap would measure the same truncation; pass both")
    if cap_override is not None and (isinstance(cap_override, bool) or not isinstance(cap_override, int)
                                     or cap_override <= 0):
        raise InspectServedError(f"{name}: cap_override must be a positive int, got {cap_override!r}")
    on = C.enable_thinking(thinking)
    row = sampling if sampling is not None else C.GREEDY_SAMPLING
    if not isinstance(row, C.SamplingRow):
        raise InspectServedError(
            f"{name}: sampling must be a contracts.SamplingRow, got {type(row).__name__}")
    # the render fact comes from the FAMILY (`contracts.FAMILY_RENDER_FACTS`)
    # for ANY ON run — in support or out — and an explicit
    # `prompt_opened_think` must EQUAL it; a thinking-ON generation plan with
    # NO server parser on a family with no fact (or, in production, a
    # `candidate` fact) refuses by name. Without a family the parent's rule
    # stands: the fact is the OOS tuple's alone.
    fact = C.render_fact_for(family) if family is not None else None
    if family is not None and fact is None and on:
        raise InspectServedError(
            f"{name}: family {family!r} has no render fact in contracts.FAMILY_RENDER_FACTS — "
            "what it renders and emits under thinking-ON is unknown; refusing to plan on a guess")
    if thinking == C.THINKING_NOT_AVAILABLE:
        # llama-paper-rows (P7 / P8): the NO-SWITCH arm is planned on the FAMILY's fact alone — a fact
        # is REQUIRED, it must say the template carries no switch (contracts.cert_class_for), and it must
        # be certified for production exactly like a thinking-ON family's (the dry run's `pre` reading
        # passes require_certified_fact=False)
        if fact is None:
            raise InspectServedError(
                f"{name}: the {thinking!r} arm needs the family's render fact (contracts.FAMILY_RENDER_FACTS) and "
                f"family {family!r} has none — a no-thinking-channel family is planned on its fact, never on a guess")
        try:
            C.cert_class_for(thinking, fact)
        except ValueError as e:
            raise InspectServedError(f"{name}: {e}") from None
        if prompt_opened_think:
            raise InspectServedError(f"{name}: prompt_opened_think=True under the {thinking!r} arm — a family with no "
                                     "thinking switch opens no block; refusing to plan on a contradiction")
        if reasoning_parser:
            raise InspectServedError(f"{name}: a reasoning parser {reasoning_parser!r} under the {thinking!r} arm — a family "
                                     "with no thinking channel composes none; refusing to plan on a contradiction")
        why = C.render_fact_refusal(family, require_certified=require_certified_fact)
        if why is not None:
            raise InspectServedError(f"{name}: {why}")
    if fact is not None:
        if on and fact.thinking_switch is False:
            # llama-paper-rows (P7 / P18): the planner-side twin of contracts.cert_class_for's refusal — a family whose
            # template carries NO thinking switch has no ON render to plan: its arm is `not_available` and its split is
            # `none`; a thinking-ON plan on it would compose a split its own fact contradicts. Until the fact was
            # certified (2026-09-12) the candidate refusal below stood in for this; a certified fact needs the rule itself.
            raise InspectServedError(
                f"{name}: family {family!r} carries no thinking switch (contracts.NO_THINKING_CHANNEL_FAMILIES; its render "
                f"fact's split mode is {fact.split_mode!r}) — there is no thinking-ON render to plan; its arm is "
                f"{C.THINKING_NOT_AVAILABLE!r}")
        if prompt_opened_think and not fact.prompt_opens_think_on:
            raise InspectServedError(
                f"{name}: prompt_opened_think=True was passed but FAMILY_RENDER_FACTS[{family!r}]"
                f".prompt_opens_think_on is False — the render fact is the family's, never a "
                "caller's; refusing to plan on a contradiction")
        prompt_opened_think = bool(fact.prompt_opens_think_on)
        if on and not reasoning_parser:
            why = C.render_fact_refusal(family, require_certified=require_certified_fact)
            if why is not None:
                raise InspectServedError(f"{name}: {why}")
    elif prompt_opened_think and not out_of_support:
        raise InspectServedError(
            f"{name}: prompt_opened_think is the authorised out-of-support experiment's "
            "DECLARED render fact and was passed without out_of_support — an in-support run "
            "has no such fact (its ON arm is parser-separated); refusing to plan on it")
    # THIS task's k: the run's map when given; else the legacy rule (MATH-500
    # takes `samples`; every other task its contracts k, `SAMPLES_BY_TASK` —
    # IFEval's 2, ifeval-canonical-battery P2, one otherwise). Never a product
    # of the two.
    if samples_per_item is not None:
        k = int(samples_per_item)
    elif name == "math":
        k = int(samples)
    else:
        k = int(C.SAMPLES_BY_TASK[C.inspect_task(name)])
    if isinstance(samples_per_item, bool) or k < 1:
        raise InspectServedError(f"{name}: samples per item must be a positive int, got {samples_per_item!r}")
    # the cap and where it came from (plan S2.1): under `on and out_of_support`
    # the authorised experiment's raised cap wins for the tasks it names
    # (MAX_TOKENS_ON itself is never edited); everything else as before —
    # ONE rule, shared with the driver's hashed `max_tokens_by_task`
    max_tokens, cap_source = task_max_tokens(name, thinking=thinking, out_of_support=out_of_support,
                                             off_cap_override=off_cap_override,
                                             oos_experiment=oos_experiment,
                                             on_cap_override=on_cap_override,
                                             off_cap_setter=off_cap_setter)
    if cap_override is not None:
        # bcb-bfcl-200 §S2b: the ladder's rung replaces the table's cap and SAYS so
        if max_tokens is not None and int(cap_override) <= int(max_tokens):
            raise InspectServedError(
                f"{name}: the ladder cap override {cap_override} is not above the table's cap "
                f"{max_tokens} — a rung that does not raise the cap re-measures the same truncation")
        max_tokens, cap_source = int(cap_override), CAP_SOURCE_OVERRIDE
    # the think-block split, decided beside the cap (plan S2.1 / S1.3): from
    # the arm, the composed parser and the DECLARED render fact — never content
    split_mode = C.think_split_mode(arm=thinking, reasoning_parser=reasoning_parser,
                                    prompt_opened_think=bool(prompt_opened_think))
    row_task = C.inspect_task(name)
    retry = int(retry_max_tokens or 0) if row_task == RETRY_ROW_TASK else 0
    if retry and max_tokens is not None and retry <= max_tokens:
        raise InspectServedError(
            f"{row_task}: the truncation-retry cap {retry} is not above the pass cap "
            f"{max_tokens} — a re-request at or below the cap that truncated the sample "
            "would truncate it again; raise it or disable the retry (0)")
    # eval-level `limit` is provisional here: `limit_plan` moves a cap onto
    # `sample_ids` for id-selected tasks (Inspect refuses both together)
    common = dict(row_task=row_task, name=name,
                  max_tokens=max_tokens, cap_source=cap_source,
                  think_split_mode=split_mode,
                  grader_max_tokens=(int(grader_max_tokens)
                                     if name in MODEL_GRADED_TASKS else None),
                  temperature=float(row.temperature), top_p=row.top_p, top_k=row.top_k,
                  min_p=getattr(row, "min_p", None),
                  presence_penalty=getattr(row, "presence_penalty", None),
                  repetition_penalty=getattr(row, "repetition_penalty", None),
                  # bcb-bfcl-200-widen: the DECLARED protocol temperature (None outside protocol)
                  protocol_temperature=(None if protocol_temperature is None else float(protocol_temperature)),
                  family=family,
                  render_evidence_status=(fact.evidence_status if fact is not None else None),
                  samples_per_item=k,
                  primary_scorer=PRIMARY_SCORER.get(name),
                  thinking=thinking, retry_max_tokens=retry,
                  limit=inspect_limit or None)

    if name == "math":
        from inspect_evals.math.math import math as math_task

        ids = math500_sample_ids()
        if verify_math500:
            check = verify_math500_selection(ids)
            if check["missing"]:
                raise InspectServedError(
                    f"MATH-500 selection: only {check['matched']}/{len(ids)} of "
                    f"the canonical problems are addressable inside "
                    f"inspect_evals' `math` task ({check['n_source']} source "
                    f"rows; first missing {check['missing'][:3]}) — refusing to "
                    "run a 'MATH-500' number over a subset. Fall back to the "
                    "in-repo thin task over MATH-500 (P2 option b)")
        # the task is built with NO grader model and its
        # scorer list REPLACED by the judge-free `expression_exact_match_sympy`
        # alone, wrapped think-aware under the plan's split mode and registered
        # under the INNER scorer's name so the log's score key stays
        # PRIMARY_SCORER["math"]; the model-graded equivalence companion
        # (thinking-OFF greedy on the base route) is dropped — no grader request
        # ever leaves this plan. A stub task without a `scorer` attribute (a
        # test double) is left alone.
        task = math_task(shuffle=False, grader_model=None)
        wrapper: Optional[str] = None
        if hasattr(task, "scorer"):
            task.scorer = [_math_scorer(split_mode)]
            wrapper = f"think_aware/{split_mode}"
        common["scorer_wrapper"] = wrapper
        return TaskPlan(task=task,
                        sample_ids=ids,
                        expected_n=len(ids),
                        note="MATH-500 ids selected inside inspect_evals' math "
                             "task; primary metric expression_exact_match_sympy — the ONLY "
                             "scorer (judge-free; the model-graded equivalence companion is "
                             "dropped, D5 P1-6), wrapped think-aware under the family's split "
                             "mode and registered under the sympy scorer's own name; NO grader "
                             "model; the run's sampling row replaces the task's own 0.5 "
                             "(Inspect's default, never a decision — U8); k = samples_per_item",
                        **common)

    if name in ("gsm8k", "gsm8k_paired200"):
        # ONE builder, two item sets (PW3 — never a second GSM8K adapter; the
        # roles flipped by U4(a), PW17): `gsm8k_paired200` = the
        # STANDARD instrument (control `gsm8k`), inspect_evals' stock task
        # narrowed to the <=200 items paired by question text to the retired
        # instrument's Platinum rows — the headline, fills the two-SI `gsm8k`
        # slot; `gsm8k` = the same task over its full test set — the IN-DEPTH
        # control `gsm8k_full`, never the slot. Both annotate `paired_id` on
        # the matched items, so a full run pairs with the old rows too
        # (compare_served_hf_task.py --allow-unmatched).
        from inspect_evals.gsm8k.gsm8k import gsm8k as gsm8k_task

        item_set = C.GSM8K_ITEM_SET_OF_TASK[C.inspect_task(name)]
        # the paired rows through the ONE derivation the launcher's pre-standup
        # probe shares (gsm8k_paired200_selection_ids): READING the retired
        # instrument's row set is allowed — only its GENERATION entry points
        # are gated (PW3)
        paired = _gsm8k_paired_by_text(repo_root)
        task = gsm8k_task()
        # plan S2.4: the stock `match(numeric=True)` scorer, wrapped so it reads
        # the ANSWER of a completion under this plan's think-split mode and
        # records the split on every score — in BOTH arms and every mode
        # (`record_only` is a pure pass-through with metadata), on both GSM8K
        # tasks. The wrap is applied here, where the plan's mode is known.
        task.scorer = [_gsm8k_scorer(split_mode)]
        present = _dataset_ids(task)
        if item_set == C.GSM8K_ITEM_SET_FULL:
            size = len(present) if present is not None else None
            if verify_pairs and size is not None and size != C.GSM8K_FULL_N:
                raise InspectServedError(
                    f"the stock gsm8k test split has {size} items, not the frozen "
                    f"{C.GSM8K_FULL_N} (contracts.GSM8K_FULL_N) — a dataset drift; "
                    "refusing to file a 'full GSM8K' number over a different set")
            n_paired = (len(paired) if present is None
                        else sum(1 for i in paired if i in present))
            return TaskPlan(task=task, paired_ids=paired, expected_n=size,
                            item_set=item_set,
                            note="the IN-DEPTH full GSM8K (U4(a); control gsm8k_full — "
                                 "PW3's standard, never the two-SI slot now): "
                                 "inspect_evals' stock task over the full openai/gsm8k "
                                 f"test split; {n_paired} items also pair by text to the "
                                 "retired s2parity/gsm8k_0shot platinum rows (paired_id)",
                            **common)
        # the frozen count is the MATCHED set: an id absent from Inspect's
        # split (a platinum-edited question) is never requested, so an
        # unmatched item cannot read as an unscored one — and since U4(a) the
        # slot's acceptance rule expects exactly GSM8K_PAIRED_N, so a drift in
        # the pinned inputs REFUSES here rather than filing a shorter standard
        # set every loader would then move to the legacy key
        ids = sorted(paired) if present is None else sorted(i for i in paired if i in present)
        n_unmatched = len(paired) - len(ids)
        if verify_pairs and len(ids) != C.GSM8K_PAIRED_N:
            raise InspectServedError(
                f"the standard (paired) GSM8K matched {len(ids)} of {len(paired)} Platinum "
                f"questions in the stock test split ({n_unmatched} unmatched), not the frozen "
                f"{C.GSM8K_PAIRED_N} (contracts.GSM8K_PAIRED_N) the `gsm8k` slot's acceptance "
                "rule expects — a pinned-input drift; refusing to file a shorter standard set")
        if item_subset is not None:
            # the RUNG plan — the eligible truncating ids of the source store,
            # AFTER the 200-drift check accepted the population they must come
            # from; every id must be one of the plan's paired ids; the frozen
            # count of THIS store is the subset's size, so its done marker is
            # written at that size (subset-sized completeness) and
            # `n_paired == n_run`; `item_set` stays `paired200` (the same
            # instrument's items, fewer of them)
            wanted = [str(i) for i in item_subset]
            if not wanted:
                raise InspectServedError(f"{row_task}: {C.FLAG_ITEM_SUBSET} names no ids — a rung with nothing to "
                                         "re-draw is not a rung (the planner writes no subset for it)")
            outside = sorted(set(wanted) - set(ids))
            if outside:
                raise InspectServedError(
                    f"{row_task}: {len(outside)} of the {len(wanted)} item-subset ids are not in the plan's "
                    f"{len(ids)} paired ids (first {outside[:3]}) — a rung re-draws the SOURCE store's own "
                    "items only (plan §4.4)")
            if inspect_limit:
                raise InspectServedError(
                    f"{row_task}: a rung plan ({C.FLAG_ITEM_SUBSET}) takes no smoke cap — its population is "
                    "the subset, whole")
            keep = set(wanted)
            ids = [i for i in ids if i in keep]
            return TaskPlan(task=task, sample_ids=ids, paired_ids=paired, expected_n=len(ids), n_run=len(ids),
                            item_set=item_set,
                            note="the STANDARD GSM8K's RUNG (gsm8k-paired200-paper-objects §4.4): the "
                                 f"{len(ids)} ELIGIBLE truncating items of the source store re-drawn at "
                                 f"cap {max_tokens} — the same paired instrument narrowed to its subset; "
                                 "the frozen count of this store IS the subset's size",
                            **{k: v for k, v in common.items() if k != "limit"}, limit=None)
        return TaskPlan(task=task, sample_ids=ids, paired_ids=paired, expected_n=len(ids),
                        item_set=item_set,
                        note="the STANDARD GSM8K (U4(a); PW3's paired-200 comparison "
                             "bridge, now the headline and the two-SI `gsm8k` slot): our "
                             "gsm8k-platinum questions located in Inspect's openai/gsm8k "
                             "test split by text and scored against the ORIGINAL gsm8k "
                             "gold (Platinum's answer corrections are not applied); "
                             "platinum-EDITED questions simply do not match and are "
                             f"reported unmatched ({n_unmatched}) rather than paired to a "
                             "different one",
                        **common)

    if name == "xstest":
        from inspect_evals.xstest.xstest import xstest as xstest_task
        from sisteer.capeval.tiers.judged import xstest_rows

        rows = [r for r in xstest_rows(_pins(repo_root)) if r["subset"] == "safe"]
        paired = _paired_by_text(
            rows, text_key="prompt",
            sample_id=lambda t: C.inspect_sample_id("xstest", t))
        task = xstest_task(subset="safe", scorer_model=grader_model)
        return TaskPlan(task=task, paired_ids=paired, expected_n=_dataset_size(task),
                        note="stock XSTest safe-250, graded by the unsteered "
                             "base route (P4: the default grader is "
                             "openai/gpt-4o, which this spec forbids) — the "
                             "refusal rate is labelled Gemma-base-judged",
                        **common)

    if name == "or_bench_hard_1k":
        build = load_repo_task("or_bench_hard_1k")

        pins = _pins(repo_root)
        spec = (dict(or_bench_item_set) if or_bench_item_set is not None
                else C.or_bench_item_set())
        rows, positions = or_bench_selected_rows(spec, pins=pins)
        paired = _paired_by_text(
            rows, text_key="prompt",
            sample_id=lambda t: C.inspect_sample_id("or_bench_hard_1k", t))
        # the thin task heads its own order (subset or full), so a smoke run's
        # items are a PREFIX of that run's; eval-level limit would head the
        # same order twice, so it is left off here (n_run set: limit_plan
        # leaves the plan as built)
        common["limit"] = None
        n_run = min(int(inspect_limit), len(rows)) if inspect_limit else len(rows)
        subset = spec["item_set"] == C.OR_BENCH_ITEM_SET_SUBSET
        note = ("P3: vendored thin task (OR-Bench-Hard-1K is NOT "
                "in inspect_evals 0.18.0; its METADATA lists it as "
                "an external package) over the SAME pinned CSV our "
                "judged control uses — full item overlap")
        note += (f"; item_set={spec['item_set']}: the frozen seeded subset, "
                 f"{len(rows)} of {spec['of']} at seed {spec['seed']} (the user, "
                 "2026-09-06), positions hashed into the run identity"
                 if subset else
                 f"; item_set={spec['item_set']}: the whole pinned corpus, {len(rows)} items")
        return TaskPlan(task=build(scorer_model=grader_model,
                                   limit=int(inspect_limit or 0), pins=pins,
                                   positions=positions),
                        paired_ids=paired, expected_n=len(rows), n_run=n_run,
                        item_set=spec["item_set"],
                        note=note,
                        **common)

    if name == "bigcodebench_200":
        # the in-repo thin task over inspect_evals'
        # stock bigcodebench, narrowed to the PIN (or the ladder's selection, a
        # subset of it in pin order), every sample executed in an OWNED Modal
        # sandbox. The pin is loaded through the contracts' VERIFIED reader
        # (a pin that does not regenerate from its rule refuses here).
        build = load_repo_task("bigcodebench_200")
        pin = C.load_bcb_pin(repo_root)
        log_ids = _pinned_subset("bigcodebench_200", [C.bcb_log_id(i) for i in pin["ids"]], selection)
        # a smoke cap heads the pin BEFORE building (fewer sandboxes are
        # created at all); n_run set, so limit_plan leaves the plan as built
        # and no eval-level `limit` rides beside the id selection
        if inspect_limit and inspect_limit < len(log_ids):
            log_ids = log_ids[:int(inspect_limit)]
        common["limit"] = None
        task = _build_bcb_task(build, log_ids=log_ids, repo_root=repo_root, pin=pin, generation_only=generation_only)
        got = [str(s.id) for s in task.dataset]
        if got != log_ids:
            raise InspectServedError(
                f"bigcodebench_200: the built task's ids {got[:5]}... != the planned {log_ids[:5]}...")
        scorer_names = _scorer_names(task)
        if scorer_names is not None and scorer_names != [PRIMARY_SCORER["bigcodebench_200"]]:
            raise InspectServedError(
                f"bigcodebench_200: the stock task files scorer(s) {scorer_names}, not exactly "
                f"[{PRIMARY_SCORER['bigcodebench_200']!r}] — the pinned scorer moved")
        return TaskPlan(task=task, sample_ids=list(log_ids), expected_n=int(pin["n"]), n_run=len(log_ids),
                        item_set="pinned200", sandbox=C.bcb_sandbox_facts(), generation_only=bool(generation_only),
                        selection=(list(log_ids) if selection is not None else None),
                        note=(f"bcb-bfcl-200: inspect_evals' stock bigcodebench ({C.BCB_SPLIT} complete "
                              f"split) over the pinned {pin['n']} non-Hard ids (seed {pin['seed']} of "
                              f"{pin['of']}), each sample executed in an owned Modal sandbox "
                              f"({C.BCB_SANDBOX_PROVIDER}, image {C.BCB_SANDBOX_IMAGE[-16:]}, "
                              f"verify timeout {C.BCB_VERIFY_TIMEOUT_S} s)"
                              + (f"; ladder selection of {len(log_ids)} capped items at cap {max_tokens}"
                                 if selection is not None else "")),
                        **common)

    if name == "bfcl_200":
        # the stock bfcl over the battery's categories, narrowed by sample_id
        # to the PIN (or the ladder's selection), the ids cross-checked against
        # the built task's own dataset (verify_selection)
        from inspect_evals.bfcl.bfcl import bfcl as bfcl_task

        pin = C.load_bfcl_pin(repo_root)
        ids = _pinned_subset("bfcl_200", pin["ids"], selection)
        cats = bfcl_categories()
        task = bfcl_task(categories=cats)
        n_source = _dataset_size(task)
        if n_source != C.BFCL_FULL_N:
            raise InspectServedError(
                f"bfcl_200: the task built over {len(cats)} categories exposes {n_source} samples, not the "
                f"pin's population {C.BFCL_FULL_N} (contracts.BFCL_FULL_N) — a dataset drift; refusing")
        check = verify_selection(task, ids)
        if check["missing"]:
            raise InspectServedError(
                f"bfcl_200: only {check['matched']}/{len(ids)} pinned ids exist in inspect_evals' bfcl "
                f"task ({check['n_source']} source rows; first missing {check['missing'][:3]}) — refusing "
                "to run a 'bfcl_200' number over a subset of the pin")
        return TaskPlan(task=task, sample_ids=ids, expected_n=int(pin["n"]),
                        n_run=(len(ids) if selection is not None else None),
                        item_set="pinned200",
                        selection=(list(ids) if selection is not None else None),
                        note=(f"bcb-bfcl-200: BFCL V1 u V2 single-turn ({len(cats)} categories) narrowed to "
                              f"the pinned {pin['n']} ids (Hamilton by category size, floor 1, seed "
                              f"{pin['seed']} of {pin['of']}); needs a --tool-calls standup"
                              + (f"; ladder selection of {len(ids)} capped items at cap {max_tokens}"
                                 if selection is not None else "")),
                        **common)

    if name == C.CONTROL_BIGCODEBENCH_FULL:
        # capeval-full-tasks: the SAME thin task over the
        # FULL v0.1.2 complete split (1,140 ids, Hard INCLUDED), read through the
        # contracts' verified FULL pin; every sample executed in an owned Modal
        # sandbox exactly as the 200 leg. No ladder knobs (refused above: the
        # ladder is the 200 legs'); a smoke cap heads the pin BEFORE building.
        build = load_repo_task(C.CONTROL_BIGCODEBENCH_FULL)
        pin = C.load_bcb_full_pin(repo_root)
        log_ids = [C.bcb_log_id(i) for i in pin["ids"]]
        if inspect_limit and inspect_limit < len(log_ids):
            log_ids = log_ids[:int(inspect_limit)]
        common["limit"] = None
        task = _build_bcb_task(build, log_ids=log_ids, repo_root=repo_root, pin=pin, generation_only=generation_only)
        got = [str(s.id) for s in task.dataset]
        if got != log_ids:
            raise InspectServedError(
                f"{name}: the built task's ids {got[:5]}... != the planned {log_ids[:5]}...")
        scorer_names = _scorer_names(task)
        if scorer_names is not None and scorer_names != [PRIMARY_SCORER[name]]:
            raise InspectServedError(
                f"{name}: the stock task files scorer(s) {scorer_names}, not exactly "
                f"[{PRIMARY_SCORER[name]!r}] — the pinned scorer moved")
        return TaskPlan(task=task, sample_ids=list(log_ids), expected_n=int(pin["n"]), n_run=len(log_ids),
                        item_set="full", sandbox=C.bcb_sandbox_facts(), generation_only=bool(generation_only),
                        note=(f"capeval-full-tasks: inspect_evals' stock bigcodebench ({C.BCB_SPLIT} complete "
                              f"split) over ALL {pin['n']} ids (Hard included: {len(pin['hard_ids'])} listed beside), "
                              f"no seed, each sample executed in an owned Modal sandbox "
                              f"({C.BCB_SANDBOX_PROVIDER}, image {C.BCB_SANDBOX_IMAGE[-16:]}, "
                              f"verify timeout {C.BCB_VERIFY_TIMEOUT_S} s); harness limits "
                              f"contracts.FULL_TASK_LIMITS[{row_task!r}]"),
                        **common)

    if name == C.CONTROL_BFCL_FULL:
        # capeval-full-tasks: the stock bfcl over the battery's categories, narrowed
        # by sample_id to the FULL pin (every item — the narrowing is a no-op by
        # construction, asserted through verify_selection so a dataset drift refuses)
        from inspect_evals.bfcl.bfcl import bfcl as bfcl_task

        pin = C.load_bfcl_full_pin(repo_root)
        ids = [str(i) for i in pin["ids"]]
        cats = bfcl_categories()
        task = bfcl_task(categories=cats)
        n_source = _dataset_size(task)
        if n_source != C.BFCL_FULL_N:
            raise InspectServedError(
                f"{name}: the task built over {len(cats)} categories exposes {n_source} samples, not the "
                f"pin's population {C.BFCL_FULL_N} (contracts.BFCL_FULL_N) — a dataset drift; refusing")
        check = verify_selection(task, ids)
        if check["missing"]:
            raise InspectServedError(
                f"{name}: only {check['matched']}/{len(ids)} pinned ids exist in inspect_evals' bfcl "
                f"task ({check['n_source']} source rows; first missing {check['missing'][:3]}) — refusing "
                f"to run a {name!r} number over a subset of the pin")
        return TaskPlan(task=task, sample_ids=ids, expected_n=int(pin["n"]), n_run=None,
                        item_set="full",
                        note=(f"capeval-full-tasks: BFCL V1 u V2 single-turn ({len(cats)} categories), ALL "
                              f"{pin['n']} ids (no seed; per_category = the population counts); needs a "
                              f"--tool-calls standup; harness limits contracts.FULL_TASK_LIMITS[{row_task!r}]"),
                        **common)

    if name == C.IFEVAL_TASK_NAME:
        # inspect_evals 0.18.0's STOCK task
        # — dataset google/IFEval @ IFEVAL_DATASET_REVISION, solver [generate()],
        # scorer `instruction_following` — built untouched (P5: "at-cap rows are
        # scored EXACTLY as IFEval scores them" is literal, and the retained
        # .eval log keeps inspect_evals' own if_metric numbers for gate I4). The
        # value is a dict of five numbers; the row's `acc` reads
        # PRIMARY_VALUE_KEY["ifeval"] and the five values + the instruction ids
        # ride as row facts (ifeval_row_facts). k = IFEVAL_SAMPLES_PER_ITEM seeded
        # trials (the run's k map; SEED_RULE_IFEVAL through seed_for_task).
        from inspect_evals.ifeval.ifeval import ifeval as ifeval_task

        # P6: the response-language checker calls bare langdetect.detect(),
        # stochastic unseeded; pinned so a resumed cell agrees with itself
        import langdetect
        langdetect.DetectorFactory.seed = C.IFEVAL_LANGDETECT_SEED
        task = ifeval_task()
        present = _dataset_ids(task)
        if verify_pairs and present is not None and len(present) != C.IFEVAL_N:
            raise InspectServedError(
                f"ifeval: the stock task's dataset has {len(present)} samples, not the frozen "
                f"{C.IFEVAL_N} (contracts.IFEVAL_N; google/IFEval @ {C.IFEVAL_DATASET_REVISION[:8]}) — "
                "a dataset drift; refusing to file an 'IFEval' number over a different set")
        scorer_names = _scorer_names(task)
        if scorer_names is not None and scorer_names != [PRIMARY_SCORER[C.IFEVAL_TASK_NAME]]:
            raise InspectServedError(
                f"ifeval: the stock task files scorer(s) {scorer_names}, not exactly "
                f"[{PRIMARY_SCORER[C.IFEVAL_TASK_NAME]!r}] — the pinned scorer moved")
        ids = sorted(present) if present is not None else None
        n_run: Optional[int] = None
        note = (f"inspect_evals ifeval {C.IFEVAL_TASK_VERSION} (google/IFEval @ "
                f"{C.IFEVAL_DATASET_REVISION[:8]}, {C.IFEVAL_N} prompts), stock solver and scorer; "
                f"k = {k} seeded trials (SEED_RULE_IFEVAL); thinking {thinking}; "
                f"langdetect seed {C.IFEVAL_LANGDETECT_SEED}")
        if item_subset is not None:
            # the RUNG plan — the listed items of the source store
            # re-drawn at the rung cap; every id must be one of the task's own
            # (the draw-level narrowing per sample index is lane 2's
            # rung_subset/2 contract; the item set is the union of its draws)
            wanted = [str(i) for i in item_subset]
            if not wanted:
                raise InspectServedError(f"{row_task}: {C.FLAG_ITEM_SUBSET} names no ids — a rung with nothing to "
                                         "re-draw is not a rung (the planner writes no subset for it)")
            if ids is not None:
                outside = sorted(set(wanted) - set(ids))
                if outside:
                    raise InspectServedError(
                        f"{row_task}: {len(outside)} of the {len(wanted)} item-subset ids are not in the task's "
                        f"{len(ids)} sample ids (first {outside[:3]}) — a rung re-draws the SOURCE store's own "
                        "items only (plan §5.4)")
            if inspect_limit:
                raise InspectServedError(
                    f"{row_task}: a rung plan ({C.FLAG_ITEM_SUBSET}) takes no smoke cap — its population is "
                    "the subset, whole")
            ids = sorted(set(wanted))
            if item_subset_draws is None:
                return TaskPlan(task=task, sample_ids=ids, expected_n=len(ids), n_run=len(ids),
                                item_set="ifeval541",
                                note=note + f"; RUNG: {len(ids)} items of the source store re-drawn at cap {max_tokens}",
                                **{kk: v for kk, v in common.items() if kk != "limit"}, limit=None)
            # the DRAW contract: ONLY the listed (item, trial) pairs are
            # re-drawn — the terminated sibling of a truncated draw is never re-requested.
            # The draws' items must be exactly the subset's ids, every trial one of the
            # task's k; a disagreement is a torn subset, refused by name
            pairs = sorted({(str(i), int(t)) for i, t in item_subset_draws})
            if len(pairs) != len(list(item_subset_draws)):
                raise InspectServedError(f"{row_task}: the draw subset repeats a draw")
            draw_items = sorted({i for i, _ in pairs})
            if draw_items != ids:
                raise InspectServedError(
                    f"{row_task}: the draw subset's items ({len(draw_items)}) are not the item subset's ids "
                    f"({len(ids)}) — one torn rung_subset file; refusing")
            bad_trials = sorted({t for _, t in pairs if not 0 <= t < k})
            if bad_trials:
                raise InspectServedError(
                    f"{row_task}: draw sample indices {bad_trials} are outside [0, {k}) (the task's k) — the draws "
                    "were planned against another k")
            by_sample = {t: sorted(i for i, tt in pairs if tt == t) for t in range(k)}
            return TaskPlan(task=task, sample_ids=ids, expected_n=len(ids), n_run=len(ids),
                            item_set="ifeval541", draw_subset=pairs,
                            sample_ids_by_sample=by_sample, expected_draws=len(pairs),
                            note=note + (f"; RUNG (draws): {len(pairs)} of the source store's (item, trial) draws "
                                         f"over {len(ids)} items re-drawn at cap {max_tokens}, per trial "
                                         f"{ {t: len(v) for t, v in by_sample.items()} }"),
                            **{kk: v for kk, v in common.items() if kk != "limit"}, limit=None)
        if inspect_limit and ids is not None and inspect_limit < len(ids):
            # a smoke cap heads the sorted ids BEFORE eval (n_run set, so
            # limit_plan leaves the plan as built; no eval-level `limit` rides
            # beside the id selection)
            ids = ids[:int(inspect_limit)]
            n_run = len(ids)
            common["limit"] = None
        return TaskPlan(task=task, sample_ids=ids, expected_n=C.IFEVAL_N, n_run=n_run,
                        item_set="ifeval541", note=note, **common)

    raise InspectServedError(f"no plan for inspect task {name!r}")  # pragma: no cover


def _build_bcb_task(build: Any, *, log_ids: Sequence[str], repo_root: Path, pin: Mapping[str, Any],
                    generation_only: bool) -> Any:
    """The thin BigCodeBench task over ``log_ids`` (LOG vocabulary) through the leg's
    builder — the coupled build (a ``Task``), or, under ``generation_only``, the
    builder's ``(task, compose facts)`` with the sandbox already dropped: the plan
    carries the task; the facts are the compose's, which ``contracts.bcb_sandbox_facts``
    already states for the manifest, so nothing here re-records them."""
    ids = [C.bcb_pinned_id(i) for i in log_ids]
    if not generation_only:
        return build(ids=ids, sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=repo_root, pin=pin)
    built = build(ids=ids, sandbox=C.BCB_SANDBOX_PROVIDER, repo_root=repo_root, pin=pin, generation_only=True)
    if not (isinstance(built, tuple) and len(built) == 2):
        raise InspectServedError(
            "the BigCodeBench builder did not return (task, compose facts) under generation_only=True — the "
            "thin task predates plan §2.2.1; refusing to plan a generation-only leg over it")
    task, _facts = built
    if getattr(task, "sandbox", None) is not None:
        raise InspectServedError(
            f"the generation-only BigCodeBench task still carries sandbox {task.sandbox!r} — the builder did not "
            "drop it (plan §2.2.1); a generation-only eval would allocate sandboxes on the GPU container")
    return task


def _pinned_subset(name: str, pinned: Sequence[str], selection: Optional[Sequence[str]]) -> List[str]:
    """The ids a pinned leg plans: the whole pin, or the ladder's ``selection``
    — a SET of pinned ids (no foreign id, no duplicate) — in PIN order. Refuses
    anything else: the ladder selects capped PINNED items and nothing else."""
    pinned = [str(i) for i in pinned]
    if selection is None:
        return pinned
    sel = [str(i) for i in selection]
    foreign = [i for i in sel if i not in pinned]
    if foreign or len(set(sel)) != len(sel):
        raise InspectServedError(
            f"{name}: the selection is not a set of pinned ids (foreign {foreign[:5]}, "
            f"{len(sel) - len(set(sel))} duplicate(s)) — the ladder selects capped PINNED items")
    order = {i: k for k, i in enumerate(pinned)}
    return sorted(sel, key=order.__getitem__)


def _scorer_names(task: Any) -> Optional[List[str]]:
    """The registry names of a built task's scorers, or None when the object
    exposes none in Inspect's shape (a test stub)."""
    from inspect_ai._util.registry import registry_info, registry_unqualified_name

    scorers = getattr(task, "scorer", None)
    if scorers is None:
        return None
    try:
        return [registry_unqualified_name(registry_info(s).name) for s in scorers]
    except Exception:  # noqa: BLE001 — not registered scorers: nothing to hold
        return None


def plans_for_atoms(atoms: Sequence[str], *,
                    thinking_by_task: Optional[Mapping[str, str]] = None,
                    sampling: Optional[C.SamplingRow] = None,
                    samples_by_task: Optional[Mapping[str, int]] = None,
                    cap_overrides_by_task: Optional[Mapping[str, int]] = None,
                    item_selection_by_task: Optional[Mapping[str, Sequence[str]]] = None,
                    sampling_by_task: Optional[Mapping[str, C.SamplingRow]] = None,
                    protocol_temperature_by_task: Optional[Mapping[str, float]] = None,
                    generation_only_tasks: Optional[Sequence[str]] = None,
                    **kw) -> List[TaskPlan]:
    """The plans a list of canonical control atoms asks for, in order.

    ``generation_only_tasks``: the ROW TASKS this run generates without scoring
    (the GPU phase of the two-phase BigCodeBench contract) — each named task's
    plan is built ``generation_only``; a name this run does not plan refuses (a
    knob that would silently do nothing), and a non-sandboxed task refuses
    inside :func:`_plan_task`. None = every plan coupled, byte-identical.

    ``sampling_by_task`` / ``protocol_temperature_by_task`` (the ``protocol``
    mode): each planned task takes ITS executed row and its declared temperature
    from the two maps (``sampling`` is then None — there is no single run row);
    a planned task either map does not name refuses (never a default row), and
    a map naming a task this run does not plan refuses too (a knob that silently
    did nothing).

    ``thinking_by_task`` (PW20b-D) gives each row task its OWN arm — the
    per-task run under one standup; a task the map does not name is refused
    here rather than served under ``kw["thinking"]``, so a hole in the map
    can never become a silently-chosen arm. Without the map every plan takes
    the single ``thinking`` arm, as before.

    ``sampling`` / ``samples_by_task`` (U8, PW-sampling) give every plan the
    run's sampling row and its OWN k (``contracts.run_samples_by_task``);
    a task the k-map does not name is refused the same way — never planned
    at a default k. Without the map the legacy rule applies (MATH-500 at
    ``kw["samples"]``, every other task 1).

    ``cap_overrides_by_task`` / ``item_selection_by_task``: the ladder's
    ``{row task: cap}`` and ``{row task: [ids]}`` maps — each named task's
    plan takes its override and its selection; a selected task with no
    override, or a map naming a task this run does not plan,
    refuses (a knob that silently did nothing would be the worse failure)."""
    names: List[str] = []
    row_tasks: Dict[str, str] = {}
    for atom in atoms:
        fam = C.control_family(atom)
        if C.DISPATCH_MODULE_OF_CONTROL.get(fam) != __name__:
            raise InspectServedError(
                f"control {atom!r} does not dispatch to this module "
                f"({C.DISPATCH_MODULE_OF_CONTROL.get(fam)})")
        for row_task in C.TASKS_OF_CONTROL[atom]:
            name = _task_name(row_task)
            row_tasks[name] = row_task
            if name not in names:
                names.append(name)
    if samples_by_task is not None:
        holes = sorted(row_tasks[n] for n in names if row_tasks[n] not in samples_by_task)
        if holes:
            raise InspectServedError(
                f"sampling: {holes} have no samples-per-item entry in this run's map "
                f"{sorted(samples_by_task)} — refusing to plan them at a default k")
    caps = dict(cap_overrides_by_task or {})
    sels = dict(item_selection_by_task or {})
    planned = {row_tasks[n] for n in names}
    stray = sorted((set(caps) | set(sels)) - planned)
    if stray:
        raise InspectServedError(
            f"ladder: cap_overrides / item_selection name {stray}, which this run does not plan "
            f"({sorted(planned)}) — refusing a knob that would silently do nothing")
    unpaired = sorted(set(sels) - set(caps))
    if unpaired:
        raise InspectServedError(
            f"ladder: item_selection names {unpaired} with no cap override — the capped items re-drawn "
            "at the same cap would measure the same truncation; pass --cap-override for each")
    gen_only = {str(t) for t in (generation_only_tasks or ())}
    stray_gen = sorted(gen_only - planned)
    if stray_gen:
        raise InspectServedError(
            f"{C.FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_GENERATE}: generation_only_tasks name {stray_gen}, "
            f"which this run does not plan ({sorted(planned)}) — refusing a phase that would silently do nothing")

    def _k(n: str) -> Optional[int]:
        return None if samples_by_task is None else int(samples_by_task[row_tasks[n]])

    def _ladder(n: str) -> Dict[str, Any]:
        rt = row_tasks[n]
        out: Dict[str, Any] = {}
        if rt in caps:
            out["cap_override"] = int(caps[rt])
        if rt in sels:
            out["selection"] = [str(i) for i in sels[rt]]
        return out

    # The per-task rows of a `protocol` run —
    # every planned task must have one (and the map must name only planned
    # tasks); with no map every plan takes the ONE run row, as before
    rows_bt = dict(sampling_by_task) if sampling_by_task is not None else None
    decl_bt = dict(protocol_temperature_by_task) if protocol_temperature_by_task is not None else None
    if rows_bt is not None and sampling is not None:
        raise InspectServedError(
            "sampling (one run row) and sampling_by_task (per-task rows) were both passed — a plan's row "
            "has exactly one source")
    for label, m in (("sampling_by_task", rows_bt), ("protocol_temperature_by_task", decl_bt)):
        if m is None:
            continue
        holes = sorted(row_tasks[n] for n in names if row_tasks[n] not in m)
        if holes:
            raise InspectServedError(
                f"{label}: {holes} have no entry in this run's per-task map {sorted(m)} — refusing to plan "
                f"them under a default ({C.SAMPLING_PROTOCOL}: every filed task decodes under its own row)")
        stray_bt = sorted(set(m) - planned)
        if stray_bt:
            raise InspectServedError(
                f"{label} names {stray_bt}, which this run does not plan ({sorted(planned)}) — refusing a "
                "row that would silently apply to nothing")

    def _row(n: str) -> Dict[str, Any]:
        rt = row_tasks[n]
        out: Dict[str, Any] = {"sampling": rows_bt[rt] if rows_bt is not None else sampling}
        if decl_bt is not None:
            out["protocol_temperature"] = float(decl_bt[rt])
        if rt in gen_only:
            out["generation_only"] = True
        return out

    if thinking_by_task is None:
        return [plan_task(n, samples_per_item=_k(n), **_row(n), **_ladder(n), **kw) for n in names]
    missing = sorted(row_tasks[n] for n in names if row_tasks[n] not in thinking_by_task)
    if missing:
        raise InspectServedError(
            f"per-task thinking: {missing} have no arm in this run's map "
            f"{sorted(thinking_by_task)} — refusing to plan them under a default arm")
    kw = {k: v for k, v in kw.items() if k != "thinking"}
    return [plan_task(n, thinking=thinking_by_task[row_tasks[n]],
                      samples_per_item=_k(n), **_row(n), **_ladder(n), **kw) for n in names]


def _gsm8k_scorer(split_mode: str) -> Any:
    """The GSM8K scorer: inspect_ai's stock ``match(numeric=True)`` (what
    inspect_evals' task ships), wrapped by ``think_split.think_aware_scorer``
    under the plan's mode (its default policy is the one in force,
    ``think_split.UNTERMINATED_POLICY``). The ONE wrap that opts into the
    no-think-block rule: under ``implicit_open`` a draw that stopped
    NATURALLY without closing the prompt-opened block is scored whole — a
    lifted close as a present block, an inline answer as ``no_think_block``
    (``n_no_think_block``); MATH-500's wrap does not opt in (its reader
    classifies such draws from the raw body)."""
    from inspect_ai.scorer import match

    from sisteer.capeval.tiers.think_split import think_aware_scorer

    return think_aware_scorer(match(numeric=True), mode=split_mode, no_think_block_when_stopped=True)


def _math_scorer(split_mode: str) -> Any:
    """MATH-500's ONE scorer: inspect_evals' judge-free
    ``expression_exact_match_sympy``, wrapped by ``think_split.think_aware_scorer``
    under the plan's mode and REGISTERED UNDER THE INNER SCORER'S NAME,
    so ``results_from_log`` finds ``PRIMARY_SCORER["math"]`` on every
    sample; the wrapper's identity rides in ``Score.metadata["scorer_wrapper"]``.
    The wrap applies the ``math_delimiters`` answer normalization: GLM's
    answer-box markers dropped and ``\\( \\)`` / ``\\[ \\]`` rewritten to ``$ $``,
    the only delimiters inspect_evals' extractor reads — the mathematics untouched."""
    from inspect_evals.math.math import expression_exact_match_sympy

    from sisteer.capeval.tiers.think_split import ANSWER_NORMALIZATION_MATH, inner_scorer_name, think_aware_scorer

    inner = expression_exact_match_sympy()
    name = inner_scorer_name(inner)
    if name != PRIMARY_SCORER["math"]:
        raise InspectServedError(
            f"MATH-500's sympy scorer is registered as {name!r}, not PRIMARY_SCORER['math'] "
            f"{PRIMARY_SCORER['math']!r} — the wrap would file its score under a key results_from_log "
            "does not read; refusing")
    # G8KSCORERFIX2: NO `no_think_block_when_stopped` here — a stopped unclosed MATH-500 draw stays unterminated -> unscored,
    # the class its paper reader (scripts/tinycap_math500_thinking.py: eos_in_thoughts / lifted_close off the raw body) files it under
    return think_aware_scorer(inner, mode=split_mode, name=name, answer_normalization=ANSWER_NORMALIZATION_MATH)


def _render_task_messages(plan: TaskPlan, ids: Sequence[str], *, what: str) -> List[List[Dict[str, str]]]:
    """Render ``plan``'s samples ``ids`` through the task's OWN solver chain with
    ``generate`` a no-op — the exact OpenAI-shaped message lists the eval sends
    (the few-shot / templated turns as the eval renders them, never a
    re-implementation). Refuses a sample the dataset lacks or non-text content."""
    from inspect_ai._eval.task.util import sample_messages
    from inspect_ai.model import ModelName
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    by_id = {str(smp.id): smp for smp in plan.task.dataset}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise InspectServedError(
            f"{what}: {len(missing)} planned ids are not in the task's dataset (first {missing[:3]})")

    async def _noop_generate(state: TaskState, *a: Any, **kw: Any) -> TaskState:
        return state

    async def _render(sample: Any) -> List[Dict[str, str]]:
        state = TaskState(model=ModelName("tinycap/prompt-budget"), sample_id=sample.id,
                          epoch=1, input=sample.input, target=Target(sample.target),
                          choices=sample.choices, messages=sample_messages(sample),
                          metadata=sample.metadata if sample.metadata else {})
        state = await plan.task.solver(state, _noop_generate)
        out: List[Dict[str, str]] = []
        for m in state.messages:
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                raise InspectServedError(
                    f"{what} {sample.id}: a {m.role} message carries non-text content "
                    f"({type(content).__name__}) — the budget tokenizes text only")
            out.append({"role": str(m.role), "content": content})
        return out

    async def _all() -> List[List[Dict[str, str]]]:
        return [await _render(by_id[i]) for i in ids]

    return asyncio.run(_all())


def math500_messages(repo_root: Path = REPO_ROOT, *, system: Optional[str] = None
                     ) -> List[List[Dict[str, str]]]:
    """The exact chat messages the MATH-500 plan sends for each of its 500 items,
    in plan order: the problems as ``inspect_evals.math``'s prompt template
    renders them through the task's own solver chain, plus — for a PROMPTED
    cell — the cell's system prompt as the provider's leading-system-message
    collapse would place it (``system`` first, ONE system turn). The
    prompt-budget script tokenizes these; the driver's live ``/v1/tokenize``
    pass re-asserts them. Refuses a plan whose item count is not 500."""
    plan = plan_task("math", grader_model=None, thinking=C.THINKING_OFF, repo_root=repo_root)
    ids = list(plan.sample_ids or [])
    if len(ids) != 500:
        raise InspectServedError(f"math500 planned {len(ids)} items, not 500")
    rendered = _render_task_messages(plan, ids, what="math500")
    if system:
        rendered = [collapse_system_messages_dicts(msgs, system) for msgs in rendered]
    return rendered


def collapse_system_messages_dicts(messages: Sequence[Mapping[str, str]], si: Optional[str]
                                   ) -> List[Dict[str, str]]:
    """The provider's leading-system-message collapse over PLAIN dict messages
    (:func:`collapse_system_messages` works on inspect_ai messages): the SI
    first, then every leading system message's text, joined into ONE system
    turn; the rest verbatim. ``si`` None returns the messages unchanged."""
    if not si:
        return [dict(m) for m in messages]
    msgs = [dict(m) for m in messages]
    leading: List[str] = []
    while msgs and msgs[0].get("role") == "system":
        leading.append(str(msgs.pop(0).get("content") or ""))
    text = "\n\n".join([str(si), *[t for t in leading if t]])
    return [{"role": "system", "content": text}, *msgs]


def ifeval_messages(repo_root: Path = REPO_ROOT, *, system: Optional[str] = None
                    ) -> List[List[Dict[str, str]]]:
    """The exact chat messages the IFEval plan sends for each of its 541
    prompts, in plan order: the
    stock task's samples rendered through its OWN solver chain (``[generate()]``
    — one user turn, no system turn, no few-shot) with ``generate`` a no-op,
    plus — for a PROMPTED cell — the cell's system prompt folded the way the
    provider's leading-system-message collapse serves it (ONE leading system
    turn). The prompt-budget CLI's IFEval branch tokenizes these (lane 2's
    G-LENGTHS measurement); the driver's live ``/v1/tokenize`` pass re-asserts
    them. Refuses a plan whose item count is not ``contracts.IFEVAL_N``."""
    plan = plan_task(C.IFEVAL_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, repo_root=repo_root)
    ids = list(plan.sample_ids or [])
    if len(ids) != C.IFEVAL_N:
        raise InspectServedError(f"ifeval planned {len(ids)} items, not the frozen {C.IFEVAL_N}")
    rendered = _render_task_messages(plan, ids, what="ifeval")
    if system:
        rendered = [collapse_system_messages_dicts(msgs, system) for msgs in rendered]
    return rendered


#: task name -> the renderer of its EXACT prompts (the live prompt-budget
#: preflight's source): ``gsm8k_paired200`` the parent's,
#: ``math`` this experiment's, ``ifeval`` the canonical battery's. A run whose
#: task has no renderer refuses the preflight by name rather than rendering
#: another task's prompts.
TASK_MESSAGES_RENDERERS: Mapping[str, Callable[..., List[List[Dict[str, str]]]]] = {
    "gsm8k_paired200": lambda repo_root, system=None: gsm8k_paired200_messages(repo_root, system=system),
    "math": lambda repo_root, system=None: math500_messages(repo_root, system=system),
    C.IFEVAL_TASK_NAME: lambda repo_root, system=None: ifeval_messages(repo_root, system=system),
}


def task_messages(task_name: str, *, repo_root: Path = REPO_ROOT,
                  system: Optional[str] = None) -> List[List[Dict[str, str]]]:
    """The EXACT rendered message lists of ``task_name`` (an Inspect task name)
    — :data:`TASK_MESSAGES_RENDERERS`; a task with no renderer refuses (a
    GSM8K render on a MATH run is "the budget was measured on another task")."""
    fn = TASK_MESSAGES_RENDERERS.get(task_name)
    if fn is None:
        raise InspectServedError(
            f"no prompt renderer for Inspect task {task_name!r} (renderers: "
            f"{sorted(TASK_MESSAGES_RENDERERS)}) — the live prompt budget must be measured on THIS "
            "run's task, never another's")
    return fn(repo_root, system=system)


def assert_budget_fits_window(task: str, budget: Mapping[str, Any], *, cap: int,
                              max_model_len: Optional[int]) -> Dict[str, Any]:
    """The single-pass window PRE-flight: the
    prompt-budget artifact's ``max`` prompt length + the pass ``cap`` must fit
    the SERVED window — before the first generation request, over the whole
    prompt set the artifact measured. Refuses an unknown window, an artifact
    without ``max``, or ``max + cap > max_model_len`` naming the overshoot.
    Returns ``{max, cap, window, margin}``. (S1's ``assert_pass_fits_window`` is
    the POST-hoc per-item check over the pass's measured prompts; this is the
    pre-flight over the artifact.)"""
    if max_model_len is None:
        raise InspectServedError(
            f"{task}: the served window (max_model_len) is unknown — the single pass at cap {cap} "
            "cannot be checked to fit; a fit that cannot be checked is not a fit")
    mx = budget.get("max") if isinstance(budget, Mapping) else None
    if not isinstance(mx, int) or isinstance(mx, bool):
        raise InspectServedError(
            f"{task}: the prompt-budget artifact carries no integer 'max' prompt length ({mx!r})")
    margin = int(max_model_len) - (int(mx) + int(cap))
    if margin < 0:
        raise InspectServedError(
            f"{task}: prompt max {mx} + cap {cap} = {mx + cap} > the served max_model_len "
            f"{max_model_len} (over by {-margin}) — an over-window request is a vLLM 400; refusing "
            "before the first generation")
    return {"max": int(mx), "cap": int(cap), "window": int(max_model_len), "margin": margin}


def gsm8k_paired200_messages(repo_root: Path = REPO_ROOT, *, system: Optional[str] = None
                             ) -> List[List[Dict[str, str]]]:
    """The exact chat messages the standard GSM8K plan sends for each of its
    200 paired items, in plan order — one OpenAI-shaped ``{"role", "content"}``
    list per item — rendered through the task's OWN solver chain with
    ``generate`` a no-op, so the few-shot system turn and the templated user
    turn are what the eval renders, never a re-implementation (the prompt-budget
    script tokenizes these; the driver's live ``/v1/tokenize`` pass re-asserts
    them). For a PROMPTED cell ``system`` is the cell's system prompt, folded
    the way the provider's leading-system-message collapse serves it — the SI
    first, then the task's few-shot system turn, ONE system turn — exactly as
    :func:`math500_messages` does, so the live prompt-budget pass measures the
    prompts the prompted route sends (the CPU artifact folds the same way);
    ``None`` leaves the rendered messages unchanged. Refuses a plan whose item
    count is not ``contracts.GSM8K_PAIRED_N``, a sample the dataset lacks, or
    non-text content."""
    plan = plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF,
                     repo_root=repo_root)
    ids = list(plan.sample_ids or [])
    if len(ids) != C.GSM8K_PAIRED_N:
        raise InspectServedError(
            f"gsm8k_paired200 planned {len(ids)} items, not the frozen {C.GSM8K_PAIRED_N}")
    rendered = _render_task_messages(plan, ids, what="gsm8k_paired200")
    if system:
        rendered = [collapse_system_messages_dicts(msgs, system) for msgs in rendered]
    return rendered


# --------------------------------------------------------------------------- #
# results adapter: an Inspect log -> CellStore rows                            #
# --------------------------------------------------------------------------- #

_SCORE_LETTERS: Mapping[str, float] = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def is_unscored(value: float) -> bool:
    """True iff ``value`` is inspect_ai's UNSCORED sentinel — a NaN.

    ``Score.unscored()`` (inspect_ai 0.3.260, ``scorer/_metric.py``) sets
    ``value`` to NaN and documents it as "the canonical sentinel that
    aggregate metrics and reducers skip". ``model_graded_qa`` files it
    whenever the grader's completion carries no ``GRADE: <letter>`` the
    pattern matches — i.e. "the instrument returned no verdict on this
    item", which is a producer FACT, not a score of zero."""
    return isinstance(value, float) and math.isnan(value)


def score_to_value(score: Any, *, value_key: Optional[str] = None) -> float:
    """One Inspect ``Score`` as a row value.

    ``CORRECT``/``INCORRECT``/``PARTIAL``/``NOANSWER`` are the letters
    ``C``/``I``/``P``/``N``; a numeric score passes through (XSTest's
    grader files 1.0 / 0.5 / 0.0 and its refusal-rate metric reads the
    halves, so rounding here would erase the partial class). A dict- or
    list-valued score has no single number and is refused rather than
    reduced by a guess — UNLESS the task names the key to read
    (``value_key``, :data:`PRIMARY_VALUE_KEY`; ifeval-canonical-battery P5):
    then ``value[value_key]`` is the reading (bool -> 0 / 1, numeric through),
    a dict WITHOUT that key refuses by name, and a non-dict value with a key
    named refuses too (the scorer's grammar moved).

    NaN passes through DELIBERATELY: it is inspect_ai's unscored sentinel
    (:func:`is_unscored`), and mapping it to a number here would invent the
    verdict the grader declined to give. Every consumer of the value must
    ask :func:`is_unscored` first — :meth:`ItemResult.extra` serializes it
    as ``null`` and :func:`run_one_task` leaves it out of the mean."""
    v = getattr(score, "value", score)
    if value_key is not None:
        if not isinstance(v, Mapping):
            raise InspectServedError(
                f"score value of type {type(v).__name__} is not a dict, but the task names the value "
                f"key {value_key!r} to read — the scorer's grammar moved")
        if value_key not in v:
            raise InspectServedError(
                f"dict-valued score carries no {value_key!r} (keys {sorted(v)}) — the scorer's "
                "grammar moved; refusing to guess a 0/1")
        return score_to_value(v[value_key])
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return _SCORE_LETTERS[v]
        except KeyError:
            raise InspectServedError(
                f"score value {v!r} is not one of {sorted(_SCORE_LETTERS)} and "
                "is not numeric — refusing to guess a 0/1") from None
    raise InspectServedError(
        f"score value of type {type(v).__name__} has no single numeric "
        "reading; name a primary scorer that does")


def unparseable_from_answer(answer: Any) -> Optional[bool]:
    """The row's ``unparseable`` from the primary scorer's ``answer``
    (PW20b-D; ``contracts.INSPECT_EXTRA_FIELDS``).

    ONE derivation, stated once: a scorer that records an answer writes
    ``""`` when its extractor found nothing (inspect_ai's ``match_str``
    behind GSM8K's ``match(numeric=True)`` and the choice scorers), so
    empty-or-whitespace is TRUE and anything else FALSE. A scorer that
    records NO answer — MATH-500's ``expression_exact_match_sympy`` files
    ``Score(value=...)`` with no ``answer`` — cannot express extraction
    failure at all, and gets ``None``: "the producer does not record it",
    never 100 % unparseable (the reading W20a had to guess at from the
    outside). A non-string answer is likewise ``None``: it is not the
    convention this derivation reads."""
    if answer is None or not isinstance(answer, str):
        return None
    return not answer.strip()


@dataclass(frozen=True)
class ItemResult:
    """One scored Inspect sample, ready to become a row."""

    item_id: str
    value: float
    sample_idx: int
    seed: int
    answer: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    paired_id: Optional[str]
    scores: Mapping[str, float]
    #: the two channels of the sample's final ``output`` (module docstring
    #: item 5): ``content`` is what the scorer saw (``""`` for an empty
    #: answer; None when the sample has no output message), ``reasoning``
    #: the thought (None when the response carried none).
    content: Optional[str] = None
    reasoning: Optional[str] = None
    #: True iff reasoning is present and content empty — a flagged row that
    #: still scores as-is.
    content_empty_with_reasoning: bool = False
    #: inspect_ai's ``ModelOutput.stop_reason`` VERBATIM (None when the sample
    #: carries no output). :data:`STOP_REASON_AT_CAP` is "at the cap".
    stop_reason: Optional[str] = None
    #: :func:`unparseable_from_answer` of the primary scorer's answer — True /
    #: False / None ("this scorer records no answer").
    unparseable: Optional[bool] = None
    #: which generation filed this result: 1, or 2 for a MATH-500 sample
    #: re-requested at the retry cap (U5-3). NOT the row's top-level
    #: ``attempt`` column (the store's retry counter).
    attempt: int = 1
    #: the cap THIS generation was requested at (None = the task pinned its
    #: own inside its solver).
    max_tokens_used: Optional[int] = None
    #: Fit-support transfer (plan S2.5): the scorer's think-block split, read
    #: off the primary score's metadata (``think_split.SPLIT_METADATA_FIELDS``;
    #: False when the scorer recorded none — a log written before the wrap, or
    #: a task that is not wrapped). ``think_block_unterminated`` under the
    #: implicit-open render = the draw ran out of cap inside its thoughts;
    #: ``stray_open_tag`` = a literal ``<think>`` inside a completion whose
    #: block the PROMPT had opened (an anomaly count).
    think_block_present: bool = False
    think_block_unterminated: bool = False
    stray_open_tag: bool = False
    #: GSM8K-SCORER-fix: the draw STOPPED under the implicit-open render without ever closing the prompt-opened block — a
    #: response with no think block, scored whole (``think_split.NO_THINK_BLOCK_FIELD``); counted as ``n_no_think_block``. Never in
    #: the row's ``extra_json`` nor the outcomes header (both shapes are pinned and compared against old stores); the summary.json carries it.
    no_think_block: bool = False
    #: Per-TASK producer facts a task's row-fact
    #: reader derived from the sample (``ROW_FACTS_OF_TASK``): for BigCodeBench
    #: ``code_source`` / ``sandbox_status`` / ``verifier_explanation``
    #: (``contracts.BCB_ROW_FACT_FIELDS``). Empty on every other task; merged
    #: into ``extra_json`` by :meth:`extra`.
    row_facts: Mapping[str, Any] = field(default_factory=dict)
    #: llama-paper-rows (P19 / P21) — the NO-SWITCH facts, PRESENT ONLY on a
    #: draw of a family with no reasoning channel (None = not recorded, the
    #: switched families' rows byte-identical): the ``none`` split's close-tag
    #: stray, and the provider's response-conversion record (``client_lift_fired``,
    #: ``lifted_reasoning``, ``raw_content_sha256`` — ``contracts.ROW_LIFT_FIELDS``)
    stray_close_tag: Optional[bool] = None
    client_lift_fired: Optional[bool] = None
    lifted_reasoning: Optional[str] = None
    raw_content_sha256: Optional[str] = None
    #: The HTTP attempt this draw's response
    #: came from: the client-minted ``X-Tinycap-Request-Id`` and the engine's external completion ``id``
    #: (``ModelOutput.metadata[WIRE_BINDING_METADATA_KEY]``, set by the provider's response hook). PRESENT
    #: ONLY on a draw whose run recorded them (None = not recorded, the row byte-identical)
    request_id: Optional[str] = None
    completion_id: Optional[str] = None

    def extra(self, **more: Any) -> Dict[str, Any]:
        """``extra_json`` for this row: the contract keys
        (:data:`contracts.INSPECT_EXTRA_FIELDS`) always — including the four
        PW20b-D producer facts ``stop_reason`` / ``unparseable`` /
        ``attempt`` / ``max_tokens_used`` and §PW-nanfix's ``unscored`` —
        plus every scorer's value so a task with a secondary metric (MATH's
        model-graded equivalence) is recoverable without re-running, plus
        both channels and the flag (``content`` / ``reasoning`` /
        ``content_empty_with_reasoning``).

        An UNSCORED scorer (inspect's NaN sentinel, :func:`is_unscored`)
        serializes as ``null`` in ``scores`` — "this scorer returned no
        verdict", the same reading ``unparseable: null`` already has — and
        never as NaN: the row is hashed through
        ``manifest.canonical_json``, which refuses non-finite floats, so a
        NaN here killed the whole cell (§PW-nanfix-1)."""
        out: Dict[str, Any] = {
            "sample_idx": self.sample_idx, "seed": self.seed,
            "answer": self.answer, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "paired_id": self.paired_id,
            "scores": {k: (None if is_unscored(v) else v)
                       for k, v in self.scores.items()},
            "content": self.content, "reasoning": self.reasoning,
            "content_empty_with_reasoning": self.content_empty_with_reasoning,
            "stop_reason": self.stop_reason, "unparseable": self.unparseable,
            "unscored": self.unscored,
            "attempt": self.attempt, "max_tokens_used": self.max_tokens_used,
            "think_block_present": self.think_block_present,
            "think_block_unterminated": self.think_block_unterminated,
            "stray_open_tag": self.stray_open_tag}
        # llama-paper-rows (P19 / P21): the no-switch facts ride ONLY when recorded — a switched
        # family's row is byte-identical to before
        if self.stray_close_tag is not None:
            out["stray_close_tag"] = bool(self.stray_close_tag)
        if self.client_lift_fired is not None:
            out["client_lift_fired"] = bool(self.client_lift_fired)
            out["lifted_reasoning"] = self.lifted_reasoning
            out["raw_content_sha256"] = self.raw_content_sha256
        # The attempt binding rides ONLY when recorded
        if self.request_id is not None:
            out[rev.ROW_REQUEST_ID_FIELD] = self.request_id
            out[rev.ROW_COMPLETION_ID_FIELD] = self.completion_id
        out.update(self.row_facts)
        out.update(more)
        return out

    def sample_provenance(self) -> Dict[str, Any]:
        """The COMPACT producer facts of this one (item, sample) draw, as an
        entry of a mean row's ``per_sample`` list (E54 #4).

        A mean row is an aggregate over k draws, so its singular
        :data:`contracts.INSPECT_EXTRA_FIELDS` can only describe ONE of them;
        this is how the other k-1 stay recoverable. It matters most for an
        UNSCORED draw, which files no ``acc@<i>`` row of its own (a row is a
        measurement) and would otherwise leave no trace but a count.

        ``value`` is ``None`` for an unscored draw — the same reading its
        ``scores`` entry already has, and never NaN. The full ``content`` /
        ``reasoning`` channels are deliberately NOT copied here: every scored
        draw already ships them on its own ``acc@<i>`` row, and duplicating k
        completions into every mean row would multiply the store for a fact
        that is already stored."""
        out = {"sample_idx": self.sample_idx, "seed": self.seed,
                "value": (None if self.unscored else self.value),
                "answer": self.answer, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "stop_reason": self.stop_reason, "unparseable": self.unparseable,
                "unscored": self.unscored, "attempt": self.attempt,
                "max_tokens_used": self.max_tokens_used}
        # The attempt binding rides ONLY when recorded
        if self.request_id is not None:
            out[rev.ROW_REQUEST_ID_FIELD] = self.request_id
            out[rev.ROW_COMPLETION_ID_FIELD] = self.completion_id
        return out

    @property
    def at_cap(self) -> bool:
        """True iff the provider said this generation stopped AT THE CAP
        (:data:`STOP_REASON_AT_CAP`) — the one definition."""
        return self.stop_reason == STOP_REASON_AT_CAP

    @property
    def unscored(self) -> bool:
        """True iff the PRIMARY scorer returned no verdict for this (item,
        sample) — inspect's ``Score.unscored()`` NaN sentinel
        (:func:`is_unscored`). Such a result carries real producer facts
        (its answer, its stop reason, its token counts) and is filed with
        them; it simply contributes no number to any mean."""
        return is_unscored(self.value)


# `message_channels` / `output_channels` — the (content, reasoning) readers of
# inspect_ai's split-response shape — live in `think_split` (the scorer reads
# the same shape) and are re-exported from this module's imports above.


def sample_usage(sample: Any, target_model: Optional[str]
                 ) -> Tuple[Optional[int], Optional[int]]:
    """``(prompt_tokens, completion_tokens)`` of ONE sample's TARGET
    generation (E40 H3).

    inspect_ai accumulates usage PER MODEL in ``EvalSample.model_usage``,
    keyed by ``f"{ModelName(model)}"`` — ``tinycap/<virtual model>`` here. A
    model-graded task (MATH-500's equivalence scorer, XSTest / OR-Bench's
    refusal grader) therefore files TWO entries per sample: the target's and
    the grader's. Summing them and calling the total "this item's prompt
    length" is wrong in the one place it matters most:
    :func:`assert_retry_fits_window` treats ``prompt_tokens`` as the length
    of the prompt the retry will re-send, so the GRADER's prompt could push
    an at-cap sample over ``max_model_len`` and refuse exactly the retry the
    sample needs.

    ``target_model`` is that key. When it names an entry, that entry ALONE is
    the row's usage. When the sample records usage but not under that key,
    this refuses: something generated and it cannot be attributed, which is
    the same failure wearing a different mask. ``None`` (no target named)
    falls back to the historical sum — right only when one model generated,
    and the reason :func:`run_one_task` refuses to arm the retry without a
    target (a fit that cannot be checked is not a fit).

    RESIDUAL, stated rather than hidden: when the target IS the grader route
    (the ``none`` cell — ``grader_route`` returns the unsteered base route,
    which that cell also serves), both generations land under ONE key and no
    reading of ``model_usage`` can separate them. The count is then a
    CEILING, so the window check errs toward refusing a retry rather than
    admitting one that would truncate twice."""
    usage = dict(getattr(sample, "model_usage", None) or {})
    if target_model is not None and usage:
        if target_model not in usage:
            raise InspectServedError(
                f"Inspect sample {getattr(sample, 'id', None)!r} records model usage for "
                f"{sorted(usage)} but nothing under the TARGET {target_model!r} — the "
                "row's prompt/completion tokens (and the truncation retry's window "
                "check) are the target generation's, and an unattributable usage map "
                "cannot supply them")
        usage = {target_model: usage[target_model]}
    entries = list(usage.values())
    pt = sum(int(getattr(u, "input_tokens", 0) or 0) for u in entries) or None
    ct = sum(int(getattr(u, "output_tokens", 0) or 0) for u in entries) or None
    return pt, ct


#: The HTTP status a provider error message carries, if it names one.
#: inspect_ai surfaces an OpenAI-compatible failure as its client exception's
#: message, which spells the code one of two ways depending on the layer that
#: raised it. Nothing INFERS a status from the absence of a match: no match
#: means the error was not an HTTP one (a transport drop, a limit, a bug) and
#: :attr:`ItemError.status` stays None, which reads as "not an HTTP failure"
#: rather than as a zero.
_HTTP_STATUS_RE = re.compile(r"(?:Error code|status[ _]code)\D{0,3}(\d{3})\b", re.I)
#: How much of an error message travels into the accounting. Long enough to
#: identify the failure, short enough that a provider echoing a request body
#: cannot pour it into a store.
ERROR_MESSAGE_CHARS = 200


def error_http_status(message: Optional[str]) -> Optional[int]:
    """The HTTP status named in an inspect ``EvalError`` message, or None."""
    if not message:
        return None
    m = _HTTP_STATUS_RE.search(str(message))
    if m is None:
        return None
    code = int(m.group(1))
    return code if 100 <= code < 600 else None


@dataclass(frozen=True)
class ItemError:
    """One item that produced NO generation — inspect ran it, every retry it
    was allowed failed, and the sample is in the log with an ``error`` instead
    of a score (E56 (3)).

    Distinct from :class:`ItemResult`'s ``unscored`` and deliberately so.
    Unscored means the SCORER returned no verdict on a real generation
    (§PW-nanfix-1); this means there is no generation to score. The two have
    opposite remedies — an unscored item reproduces exactly on a re-run
    (the grader is greedy), an errored one usually does not (the wave's six
    were 600-second gateway timeouts that the client's own retry then
    answered) — so they are counted apart and a task carrying either can never
    be read as a task carrying the other."""

    item_id: str
    sample_idx: int
    #: the HTTP status the message names, or None when it names none
    status: Optional[int] = None
    #: :data:`ERROR_MESSAGE_CHARS` of inspect's ``EvalError.message``
    message: str = ""
    #: how many earlier attempts of this sample also failed
    #: (``EvalSample.error_retries``), so a doubly-failed item is visibly
    #: doubly-failed and not just "failed"
    retries: int = 0


def sample_errors(log: Any, *, sample_idx: int = 0) -> List[ItemError]:
    """Every sample of one ``EvalLog`` that ERRORED, as :class:`ItemError`.

    Cancellations are excluded, and that exclusion is the point. Under
    inspect's default ``fail_on_error=True`` the FIRST real error aborts the
    eval, and every sibling still in flight is then logged with a backend
    cancellation exception — so a log of one 504 and thirty-seven cancelled
    neighbours describes ONE failure, not thirty-eight. inspect_ai ships the
    discriminator (``is_cancellation_message``), so this reads its rule rather
    than matching exception names of its own."""
    from inspect_ai._util.error import is_cancellation_message

    out: List[ItemError] = []
    for s in list(getattr(log, "samples", None) or []):
        err = getattr(s, "error", None)
        msg = getattr(err, "message", None) if err is not None else None
        if err is None or is_cancellation_message(msg):
            continue
        text = str(msg or "")
        out.append(ItemError(
            item_id=str(getattr(s, "id", "")),
            sample_idx=int(sample_idx),
            status=error_http_status(text),
            message=text[:ERROR_MESSAGE_CHARS],
            retries=len(getattr(s, "error_retries", None) or [])))
    return out


def cancelled_sample_count(log: Any) -> int:
    """Samples the eval's own teardown cancelled — reported beside the real
    errors so a refusal cannot be read as "38 items failed"."""
    from inspect_ai._util.error import is_cancellation_message

    return sum(1 for s in list(getattr(log, "samples", None) or [])
               if getattr(s, "error", None) is not None
               and is_cancellation_message(getattr(s.error, "message", None)))


def _errored_refusal(errs: Sequence[ItemError], *, n_cancelled: int,
                     n_samples: int) -> str:
    shown = "; ".join(
        f"{e.item_id} draw {e.sample_idx} (HTTP "
        f"{e.status if e.status is not None else 'none'}"
        f"{f', {e.retries} earlier attempts also failed' if e.retries else ''}: "
        f"{e.message[:120]})" for e in errs[:3])
    tail = "" if n_cancelled == 0 else (
        f" {n_cancelled} further samples carry a CANCELLATION, which is the eval "
        "tearing down after the first error, not that many failures.")
    return (
        f"{len(errs)} of {n_samples} Inspect samples produced NO generation after every "
        f"retry they were allowed: {shown}.{tail} Those draws have no row and no score, "
        "so scoring the survivors would silently shrink the denominator — re-run to fill "
        "the gaps")


#: The exact explanation grammar of inspect_evals 0.18.0's BigCodeBench
#: ``verify()`` scorer, reconstructed
#: rather than pattern-matched: ``preamble(answer, test) + (CORRECT_SUFFIX |
#: INCORRECT_PREFIX + stderr)``. Pinned against the scorer's own source by test.
BCB_EXPLANATION_HEAD = "The following verification code was executed:\n\n"
BCB_EXPLANATION_FENCE_OPEN = "```python\n\n"
BCB_EXPLANATION_MAIN = "\nif __name__ == '__main__':\n    unittest.main()"
BCB_EXPLANATION_FENCE_CLOSE = "\n```\n"
BCB_CORRECT_SUFFIX = "The submission was correct."
BCB_INCORRECT_PREFIX = "The submission was incorrect with the following error: \n\n"
BCB_VERIFY_TIMEOUT_STDERR = "Verification timed out."


def bcb_verify_preamble(answer: str, test: str) -> str:
    """Exactly what the pinned scorer prepends to every explanation: the
    verification program it executed, fenced."""
    code = "".join([str(answer), "\n", str(test), "\n", BCB_EXPLANATION_MAIN])
    return "".join([BCB_EXPLANATION_HEAD, BCB_EXPLANATION_FENCE_OPEN, code, BCB_EXPLANATION_FENCE_CLOSE])


def bcb_code_source(completion: str, answer: str) -> str:
    """Which of the stock ``find_code`` fallbacks produced ``answer``
    (``contracts.BCB_CODE_SOURCES``): the first ```` ```python ```` fence, else
    the first bare ```` ``` ```` fence, else the whole completion. A FORMATTING
    diagnostic — the stock extractor accepts bare whole-completion Python as
    code, so ``whole_completion`` is never ``unparseable``."""
    completion, answer = str(completion), str(answer)
    if answer == completion:
        return "whole_completion"
    if f"```python\n{answer}```" in completion:
        return "python_fence"
    if f"```\n{answer}```" in completion:
        return "bare_fence"
    # the scorer's own precedence: python fences first, then bare — an answer
    # that came from neither fence and is not the whole completion is a
    # scorer that moved
    raise InspectServedError(
        f"bigcodebench_200: the scored answer ({len(answer)} chars) is neither the first python "
        f"fence, the first bare fence nor the whole completion ({len(completion)} chars) — the pinned "
        "scorer's find_code grammar moved; refusing to classify")


def bcb_row_facts(sample: Any, primary: Any) -> Dict[str, Any]:
    """The BigCodeBench row's three facts (``contracts.BCB_ROW_FACT_FIELDS``)
    from the primary ``verify`` score and the sample's metadata, by
    RECONSTRUCTING the scorer's exact explanation grammar: the explanation must
    start with :func:`bcb_verify_preamble` over the score's ``answer`` and the
    sample's ``metadata["test"]``; the rest is exactly the correct suffix
    (``sandbox_status: ok``, explanation null) or the incorrect prefix + stderr
    (``verify_timeout`` when stderr is exactly "Verification timed out.", else
    ``ok`` with the stderr kept as ``verifier_explanation``). Anything else
    REFUSES — the pinned scorer moved, and a guessed status would be a wrong
    number filed quietly."""
    answer = getattr(primary, "answer", None)
    explanation = getattr(primary, "explanation", None)
    meta = getattr(sample, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    if answer is None or explanation is None or "test" not in meta:
        raise InspectServedError(
            f"bigcodebench_200 sample {getattr(sample, 'id', '?')}: the verify score carries answer="
            f"{'present' if answer is not None else 'None'}, explanation="
            f"{'present' if explanation is not None else 'None'}, metadata.test="
            f"{'present' if 'test' in meta else 'absent'} — cannot reconstruct the verifier's verdict")
    preamble = bcb_verify_preamble(answer, meta["test"])
    if not str(explanation).startswith(preamble):
        raise InspectServedError(
            f"bigcodebench_200 sample {getattr(sample, 'id', '?')}: the verify explanation does not begin "
            "with the verification program the pinned scorer executes — the scorer's grammar moved; "
            "refusing to classify the sandbox status")
    rest = str(explanation)[len(preamble):]
    if rest == BCB_CORRECT_SUFFIX:
        status, verifier = "ok", None
    elif rest.startswith(BCB_INCORRECT_PREFIX):
        stderr = rest[len(BCB_INCORRECT_PREFIX):]
        status = "verify_timeout" if stderr == BCB_VERIFY_TIMEOUT_STDERR else "ok"
        verifier = stderr[-2000:]
    else:
        raise InspectServedError(
            f"bigcodebench_200 sample {getattr(sample, 'id', '?')}: the verify explanation's verdict "
            f"{rest[:60]!r} is neither the correct suffix nor the incorrect prefix — the scorer moved")
    output = getattr(sample, "output", None)
    completion = getattr(output, "completion", None)
    if completion is None:
        content, _reasoning = output_channels(output)
        completion = content if content is not None else ""
    facts = {"code_source": bcb_code_source(completion, answer), "sandbox_status": status,
             "verifier_explanation": verifier}
    assert set(facts) == set(C.BCB_ROW_FACT_FIELDS)
    assert facts["code_source"] in C.BCB_CODE_SOURCES and facts["sandbox_status"] in C.BCB_SANDBOX_STATUSES
    return facts


def ifeval_row_facts(sample: Any, primary: Any) -> Dict[str, Any]:
    """The IFEval row's six facts (``contracts.IFEVAL_ROW_FACT_FIELDS``):
    the stock ``instruction_following`` score's ``value`` dict — EXACTLY its
    five keys, the two prompt-level ones bool, the three instruction-level
    ones int — and the sample's ``metadata["instruction_id_list"]``. Anything
    else REFUSES: the pinned scorer's grammar moved, and a guessed number
    would be filed quietly."""
    value = getattr(primary, "value", None)
    want = tuple(C.IFEVAL_ROW_FACT_FIELDS[:5])
    sid = getattr(sample, "id", "?")
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(sorted(want)):
        raise InspectServedError(
            f"ifeval sample {sid}: the instruction_following score's value is "
            f"{sorted(value) if isinstance(value, Mapping) else type(value).__name__}, not exactly the five "
            f"pinned keys {sorted(want)} — the scorer's grammar moved; refusing to file the row facts")
    for k in ("prompt_level_strict", "prompt_level_loose"):
        if not isinstance(value[k], bool):
            raise InspectServedError(f"ifeval sample {sid}: {k} is {value[k]!r}, not a bool — the scorer moved")
    for k in ("inst_level_strict", "inst_level_loose", "num_instructions"):
        if isinstance(value[k], bool) or not isinstance(value[k], int) or value[k] < 0:
            raise InspectServedError(f"ifeval sample {sid}: {k} is {value[k]!r}, not a non-negative int — the scorer moved")
    if not (value["inst_level_strict"] <= value["num_instructions"]
            and value["inst_level_loose"] <= value["num_instructions"] and value["num_instructions"] >= 1):
        raise InspectServedError(
            f"ifeval sample {sid}: instruction counts {dict(value)} are not coherent (followed <= total >= 1)")
    meta = getattr(sample, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    ids = meta.get("instruction_id_list")
    if (not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids
            or any(not isinstance(i, str) or not i for i in ids)):
        raise InspectServedError(
            f"ifeval sample {sid}: metadata.instruction_id_list is {ids!r}, not a non-empty list of ids")
    if len(ids) != value["num_instructions"]:
        raise InspectServedError(
            f"ifeval sample {sid}: {len(ids)} instruction ids but num_instructions {value['num_instructions']}")
    facts = {"prompt_level_strict": bool(value["prompt_level_strict"]),
             "prompt_level_loose": bool(value["prompt_level_loose"]),
             "inst_level_strict": int(value["inst_level_strict"]),
             "inst_level_loose": int(value["inst_level_loose"]),
             "num_instructions": int(value["num_instructions"]),
             "instruction_id_list": [str(i) for i in ids]}
    assert tuple(facts) == C.IFEVAL_ROW_FACT_FIELDS
    return facts


#: task name -> the per-row fact reader applied to its samples
ROW_FACTS_OF_TASK: Mapping[str, Callable[[Any, Any], Dict[str, Any]]] = {
    "bigcodebench_200": bcb_row_facts,
    "bigcodebench_full": bcb_row_facts,   # capeval-full-tasks: the same scorer, the same facts
    # ifeval-canonical-battery: the five IFEval values + the instruction ids
    C.IFEVAL_TASK_NAME: ifeval_row_facts,
}


def results_from_log(log: Any, *, sample_idx: int = 0,
                     primary_scorer: Optional[str] = None,
                     paired_ids: Optional[Mapping[str, str]] = None,
                     attempt: int = 1,
                     max_tokens_used: Optional[int] = None,
                     target_model: Optional[str] = None,
                     row_facts: Optional[Callable[[Any, Any], Dict[str, Any]]] = None,
                     primary_value_key: Optional[str] = None,
                     row_task: Optional[str] = None
                     ) -> List[ItemResult]:
    """Every scored sample of one ``EvalLog``.

    ``row_task`` names the task whose seed RULE recomputes every row's ``seed``
    (:func:`seed_for_task`; ifeval-canonical-battery P1 — IFEval's rule differs
    from the Inspect rule); ``None`` = the Inspect rule, byte-identical.

    ``primary_value_key`` (ifeval-canonical-battery P5, :data:`PRIMARY_VALUE_KEY`):
    the key of a DICT-valued primary score that is the row's value; every
    dict-valued score of the sample is then FLATTENED into ``scores`` as
    ``"<scorer>/<key>"`` (each leaf through :func:`score_to_value`), so the four
    other IFEval numbers are recoverable off the row. ``None`` = the scalar
    reading, byte-identical for every existing task.

    Refuses a log that did not finish, and refuses on an errored sample: a
    store never summarizes a partial item set, and an eval that lost samples
    to errors is a gap to fill on resume, not a smaller denominator.
    ``attempt`` / ``max_tokens_used`` label which generation
    this log IS (PW20b-D): the pass, or the MATH-500 truncation retry.
    ``target_model`` is the ``model_usage`` key of the TARGET generation
    (``contracts.inspect_model_name(<virtual model>)``); every row's
    ``prompt_tokens`` / ``completion_tokens`` is then that entry's alone and
    never the grader's — see :func:`sample_usage` (E40 H3).

    E56 (3). The errored-sample refusal now NAMES what it refuses — each
    item, the HTTP status its message carried, how many earlier attempts of
    that sample also failed, and how many further samples merely got
    cancelled when the eval tore down — because "6 samples errored" and
    "1 sample 504'd twice and 37 were cancelled behind it" call for different
    actions. It is also checked BEFORE the status gate: under inspect's
    ``fail_on_error=True`` an errored sample makes the log status ``error``,
    so the generic status refusal used to fire first and the evidence never
    reached anyone.

    There is NO opt-in that completes such a log. One
    (`--allow-errored-items`) was deferred: a completion path has to be
    honest about DRAWS, not just items, and about how an error-PARTIAL store
    becomes DONE, and neither was true of that flag — see "Deferred:
    errored-item completion" in ``docs/tinycap_interfaces.md`` §2 for what a
    re-introduction owes."""
    errs = sample_errors(log, sample_idx=sample_idx)
    samples = list(getattr(log, "samples", None) or [])
    if errs:
        raise InspectServedError(
            _errored_refusal(errs, n_cancelled=cancelled_sample_count(log),
                             n_samples=len(samples)))
    status = getattr(log, "status", None)
    if status is not None and status != "success":
        err = getattr(log, "error", None)
        raise InspectServedError(
            f"Inspect eval finished with status {status!r} ({err}) — no rows "
            "are filed from a run that did not complete")
    if not samples:
        raise InspectServedError(
            "Inspect eval log carries no samples — refusing to file an empty "
            "task (a zero-item 'complete' set is how a silently-filtered "
            "dataset gets recorded as a result)")
    paired = dict(paired_ids or {})
    out: List[ItemResult] = []
    for s in samples:
        item = str(s.id)
        scores = dict(getattr(s, "scores", None) or {})
        if not scores:
            raise InspectServedError(f"Inspect sample {item} carries no score")
        if primary_scorer is not None:
            if primary_scorer not in scores:
                raise InspectServedError(
                    f"primary scorer {primary_scorer!r} is not among sample "
                    f"{item}'s scorers {sorted(scores)}")
            primary = scores[primary_scorer]
        elif len(scores) == 1:
            primary = next(iter(scores.values()))
        else:
            raise InspectServedError(
                f"sample {item} carries {len(scores)} scores ({sorted(scores)}) "
                "and no primary scorer was named — which one is `acc` is a "
                "decision, not a default")
        # the TARGET generation's usage alone (E40 H3): a grader entry in the
        # same map is another model's prompt, never this item's
        pt, ct = sample_usage(s, target_model)
        # both channels of the sample's FINAL generation (the one scored;
        # BFCL's earlier tool turns stay in the private .eval log)
        output = getattr(s, "output", None)
        content, reasoning = output_channels(output)
        answer = getattr(primary, "answer", None)
        # the scorer's think-block split (plan S2.5): the primary score's
        # metadata, written by `think_split.think_aware_scorer`; a score
        # carrying none (an older log, an unwrapped task) reads False
        meta = getattr(primary, "metadata", None)
        meta = meta if isinstance(meta, Mapping) else {}
        # llama-paper-rows (P21): the provider's no-switch response-conversion record, when the
        # sample's output carries one (per choice; the scored draw is choice 0), onto the row
        lift = None
        out_meta = getattr(output, "metadata", None) if output is not None else None
        if isinstance(out_meta, Mapping) and isinstance(out_meta.get(NO_SWITCH_METADATA_KEY), list) \
                and out_meta[NO_SWITCH_METADATA_KEY]:
            lift = dict(out_meta[NO_SWITCH_METADATA_KEY][0])
        # The provider's attempt binding, when the response hook recorded one
        binding = out_meta.get(WIRE_BINDING_METADATA_KEY) if isinstance(out_meta, Mapping) else None
        binding = dict(binding) if isinstance(binding, Mapping) and binding.get(rev.ROW_REQUEST_ID_FIELD) else None
        out.append(ItemResult(
            item_id=item, value=score_to_value(primary, value_key=primary_value_key), sample_idx=sample_idx,
            seed=seed_for_task(row_task, item, sample_idx),
            answer=answer,
            prompt_tokens=pt, completion_tokens=ct,
            paired_id=paired.get(item),
            scores=_flatten_scores(scores, primary_value_key=primary_value_key),
            content=content, reasoning=reasoning,
            content_empty_with_reasoning=(
                pe.channel_verdict(content, reasoning) == pe.CHANNEL_FLAGGED),
            # the producer's own facts, never inferred (PW20b-D): what the
            # provider said it stopped for, and whether the scorer's own
            # extractor came back empty
            stop_reason=getattr(output, "stop_reason", None),
            unparseable=unparseable_from_answer(answer),
            attempt=int(attempt), max_tokens_used=max_tokens_used,
            think_block_present=bool(meta.get("think_block_present", False)),
            think_block_unterminated=bool(meta.get("think_block_unterminated", False)),
            stray_open_tag=bool(meta.get("stray_open_tag", False)),
            no_think_block=bool(meta.get(ts.NO_THINK_BLOCK_FIELD, False)),
            # llama-paper-rows (P19 / P21): the `none` split's close-tag stray (recorded under that
            # mode only) and the no-switch lift record — None when neither was recorded
            stray_close_tag=(bool(meta["stray_close_tag"]) if "stray_close_tag" in meta else None),
            client_lift_fired=(bool(lift.get("client_lift_fired")) if lift is not None else None),
            lifted_reasoning=(lift.get("lifted_reasoning") if lift is not None else None),
            raw_content_sha256=(lift.get("raw_content_sha256") if lift is not None else None),
            request_id=(str(binding[rev.ROW_REQUEST_ID_FIELD]) if binding is not None else None),
            completion_id=(binding.get(rev.ROW_COMPLETION_ID_FIELD) if binding is not None else None),
            # bcb-bfcl-200: the task's own producer facts (BigCodeBench's
            # code_source / sandbox_status / verifier_explanation), refusing
            # when the pinned scorer's grammar is not what the reader expects
            row_facts=(row_facts(s, primary) if row_facts is not None else {})))
    return out


def _flatten_scores(scores: Mapping[str, Any], *, primary_value_key: Optional[str]) -> Dict[str, float]:
    """``extra_json["scores"]``: every scorer's value through
    :func:`score_to_value`. With a ``primary_value_key`` named (a task whose
    primary score is a dict — IFEval), a dict-valued score is flattened to
    ``"<scorer>/<key>"`` leaves; without one the legacy scalar mapping, exactly."""
    if primary_value_key is None:
        return {k: score_to_value(v) for k, v in scores.items()}
    out: Dict[str, float] = {}
    for name, sc in scores.items():
        v = getattr(sc, "value", sc)
        if isinstance(v, Mapping):
            for k in sorted(v):
                out[f"{name}/{k}"] = score_to_value(v[k])
        else:
            out[name] = score_to_value(sc)
    return out


# --------------------------------------------------------------------------- #
# the runner                                                                   #
# --------------------------------------------------------------------------- #


def _log_dir(store: Any, row_task: str, sample_idx: int, attempt: int = 1) -> Path:
    """The log dir of one (task, sample, attempt), created.
    ``contracts.inspect_log_dir`` spells the ONE rule (attempt 1 keeps the
    historical name so existing stores resume unchanged; the MATH-500 retry
    gets its own ``__a2`` dir, since a retry log holds a DIFFERENT, shorter
    item set and beside the pass's log ``reusable_log`` would read one as the
    other); this adds the mkdir."""
    d = C.inspect_log_dir(store.cell_dir, row_task, sample_idx, attempt)
    d.mkdir(parents=True, exist_ok=True)
    return d


#: The suffix an ABANDONED retained log is renamed to (``<name>.eval.abandoned-<attempt id>``) — invisible to
#: every ``*.eval`` reader (``reusable_log``, ``retained_log_reason``, the analyses' log readers), kept on disk as the
#: durable trace of what the all-or-nothing rule set aside and which attempt did it
ABANDONED_LOG_SUFFIX = ".abandoned-"


def abandon_task_logs(store: Any, row_task: str, *, samples_per_item: int, attempt_id: str) -> List[Path]:
    """The ALL-OR-NOTHING reuse contract under ``--attempt-id``: rename every retained ``.eval`` of ``row_task`` —
    every trial's pass log AND its truncation-retry log — to ``*.eval.abandoned-<attempt_id>`` so that no ``*.eval``
    reader sees it and this attempt generates the task WHOLE. Returns the renamed paths (in log-dir order). Nothing is
    deleted: the abandoned logs stay beside the store as the record of the resume that set them aside, and a torn
    abandon (a rename that failed) raises rather than leaving a half-visible set."""
    out: List[Path] = []
    for idx in range(int(samples_per_item)):
        for attempt in (1, 2):
            d = C.inspect_log_dir(store.cell_dir, row_task, idx, attempt)
            if not d.is_dir():
                continue
            for q in sorted(d.glob("*.eval")):
                if not q.is_file():
                    continue
                target = q.with_name(f"{q.name}{ABANDONED_LOG_SUFFIX}{attempt_id}")
                q.rename(target)
                out.append(target)
    return out


class _MixedEvidence(Exception):
    """This attempt was about to GENERATE after having REUSED a retained log of the same task — the mixed
    evidence the all-or-nothing rule forbids; the caller abandons every retained log and runs the task whole."""


def log_sample_ids(log: Any) -> List[str]:
    """Every sample id of an ``EvalLog``, spelled as :func:`results_from_log`
    keys them (``str(sample.id)``), duplicates PRESERVED — the log's id
    multiset, in log order."""
    return [str(s.id) for s in (getattr(log, "samples", None) or [])]


def id_multiset_mismatch(got: Sequence[str], expected: Sequence[str]) -> Optional[str]:
    """Why the id multiset ``got`` is not EXACTLY ``expected`` — ``None`` when
    it is. Names the expected ids the log lacks, the ids it carries that were
    never selected, and the selected ids it carries MORE times than selected.
    A multiset, never a set: the set comparison that used to guard the retry
    could not see one requested id missing behind another one duplicated, and
    no count check can see a same-sized log over a different selection."""
    g, e = Counter(str(i) for i in got), Counter(str(i) for i in expected)
    if g == e:
        return None
    absent = sorted(i for i in e if g[i] < e[i])
    unexpected = sorted(i for i in g if i not in e)
    duplicated = sorted(i for i in g if i in e and g[i] > e[i])
    parts = [f"{len(got)} sample ids against {len(expected)} selected"]
    if absent:
        parts.append(f"{len(absent)} selected id(s) absent (first {absent[:3]})")
    if unexpected:
        parts.append(f"{len(unexpected)} id(s) never selected (first {unexpected[:3]})")
    if duplicated:
        parts.append(f"{len(duplicated)} selected id(s) duplicated (first {duplicated[:3]})")
    return "; ".join(parts)


def _refuse_log_id_mismatch(task: str, idx: int, log: Any, expected_ids: Optional[Sequence[str]],
                            *, what: str, extra: str = "") -> None:
    """Refuse a ``what`` log (the pass, the retry) whose sample-id multiset is
    not ``expected_ids`` — run on EVERY log, reused or fresh, BEFORE any
    outcome is derived from it. ``None`` expected ids (a task selected by
    ``limit`` rather than by id) is the one case nothing can be compared."""
    if expected_ids is None:
        return
    why = id_multiset_mismatch(log_sample_ids(log), expected_ids)
    if why is not None:
        raise InspectServedError(
            f"{task} sample {idx}: the {what} log does not carry this task's {what} selection "
            f"({why}) — refusing to derive any outcome from it{extra}")


def log_max_tokens(log: Any) -> Optional[int]:
    """The completion cap a retained ``.eval`` log was GENERATED at:
    ``log.plan.config.max_tokens`` — the eval-level ``GenerateConfig`` Inspect
    records (the task's config merged with the ``max_tokens`` this tier
    passes ``eval_async``; what Inspect's own ``eval_retry`` reads back to
    reproduce a run). ``None`` when the log records no cap, or carries no
    plan / config at all — which a caller holding the log to a plan's cap
    treats as "cannot be held", never as a match."""
    config = getattr(getattr(log, "plan", None), "config", None)
    cap = getattr(config, "max_tokens", None)
    return int(cap) if isinstance(cap, int) and not isinstance(cap, bool) else None


def _log_cap_mismatch(log: Any, expected_max_tokens: Optional[int]) -> Optional[str]:
    """Why ``log`` cannot be reused as evidence of a plan capped at
    ``expected_max_tokens`` — ``None`` when it can, or when the plan pins no
    cap of its own (``None``: the task's own config governs and there is
    nothing to hold the log to). Within one store the identity now pins the
    cap, so a log at another cap can only be a copied-in or pre-identity
    one — and a same-count, same-selection log generated at another cap
    reproduces every count and would otherwise be relabelled
    ``max_tokens_used = <this plan's cap>`` on reuse."""
    if expected_max_tokens is None:
        return None
    recorded = log_max_tokens(log)
    if recorded is None:
        return (f"records no completion cap, so it cannot be held to this plan's "
                f"{int(expected_max_tokens)}")
    if recorded != int(expected_max_tokens):
        return (f"generated at completion cap {recorded}, not this plan's "
                f"{int(expected_max_tokens)} (a log at another cap is another instrument's evidence)")
    return None


def _eval_logs_newest_first(log_dir: str | Path) -> List[Path]:
    """The ``.eval`` files under ``log_dir``, newest first — regular files only (``is_file``
    follows symlinks, so a dangling link and a directory named ``*.eval`` are passed over
    rather than crashing the mtime sort: ``stat()`` raises on a dangling link)."""
    return sorted((q for q in Path(log_dir).glob("*.eval") if q.is_file()),
                  key=lambda q: q.stat().st_mtime, reverse=True)


def reusable_log(log_dir: Path, *, expected_n: Optional[int],
                 expected_ids: Optional[Sequence[str]] = None,
                 expected_max_tokens: Optional[int] = None):
    """A completed ``.eval`` already in this (task, sample) log dir, or
    ``None``.

    Resume without re-spending: an Inspect log that finished successfully
    with the expected sample count is the same evidence a re-run would
    produce, and re-running it costs the GPU minutes this whole spec is
    trying to measure.

    E56 (3): a log carrying an ERRORED sample is never that evidence, however
    it reads otherwise — the hole is the whole reason to re-run. Under the
    only mode this tier runs (``fail_on_error=True``) such a log also fails
    the status filter, so today the check is defence in depth; it is kept
    because a log that finishes ``success`` at the full sample count with a
    hole still in it is exactly what an error-tolerant mode would write, and
    status plus count alone would resume onto a task that never measured
    those items.

    ``expected_ids`` (the plan's ``sample_ids``, or the retry's requested ids):
    a log whose sample-id MULTISET is not exactly that selection is not this
    plan's evidence either, however its count reads — a same-sized log over a
    different selection is what status plus count would have resumed onto —
    and is passed over like a wrong count (:func:`id_multiset_mismatch`).

    ``expected_max_tokens`` (the plan's cap for this attempt — the pass cap,
    or the retry cap for an ``__a2`` log): a log GENERATED at another cap, or
    recording none, is not this plan's evidence either (reuse once checked
    selection and status only, so a 4,096-cap log resumed under an 8,192
    plan was relabelled ``max_tokens_used = 8192``) and is passed over
    like a wrong count (:func:`_log_cap_mismatch`, reading
    :func:`log_max_tokens`). ``None`` = the plan pins no cap of its own, so
    nothing is held."""
    from inspect_ai.log import read_eval_log

    for p in _eval_logs_newest_first(log_dir):
        # the header first (status + completed count, no samples): a failed
        # attempt's log — every sample, with its full content — is never
        # deserialized just to be rejected. The header is a FILTER only; the
        # full log's own status and sample count below stay the verdict.
        try:
            head = read_eval_log(str(p), header_only=True)
        except Exception:  # noqa: BLE001 — a torn log is simply not reusable
            continue
        if getattr(head, "status", None) != "success":
            continue
        n_head = getattr(getattr(head, "results", None), "completed_samples", None)
        if expected_n is not None and isinstance(n_head, int) and n_head != expected_n:
            continue
        # off the header too (`plan.config` rides in the header): a log
        # RECORDING another cap is passed over before its samples are read; a
        # header recording none falls through to the full log's own verdict
        head_cap = log_max_tokens(head)
        if expected_max_tokens is not None and head_cap is not None and head_cap != int(expected_max_tokens):
            continue
        try:
            log = read_eval_log(str(p))
        except Exception:  # noqa: BLE001
            continue
        if getattr(log, "status", None) != "success":
            continue
        n = len(list(getattr(log, "samples", None) or []))
        if expected_n is not None and n != expected_n:
            continue
        if sample_errors(log):
            continue
        if expected_ids is not None and id_multiset_mismatch(log_sample_ids(log), expected_ids):
            continue
        if _log_cap_mismatch(log, expected_max_tokens) is not None:
            continue
        if n:
            return log
    return None


def retained_log_reason(log_dir: Path, *, expected_n: Optional[int],
                        expected_ids: Optional[Sequence[str]] = None,
                        expected_max_tokens: Optional[int] = None) -> Optional[str]:
    """Why NO retained ``.eval`` under ``log_dir`` is a completed log of this
    (task, sample) — ``None`` when one is (exactly :func:`reusable_log`'s
    verdict, the reader the driver's repair uses). The launcher asks this
    BEFORE a standup (its former check counted any ``*.eval`` path, a
    dangling symlink or a one-byte file included, and the container paid the
    standup before the driver's repair refused): every candidate path is
    named with what disqualifies it — a dangling symlink or a non-file, an
    unreadable (torn / truncated) log, a status other than ``success``, a
    sample count other than ``expected_n``, an errored sample, an id multiset
    that is not ``expected_ids``, a recorded completion cap that is not
    ``expected_max_tokens`` (or none recorded, when the plan pins one) — or
    the dir holds no ``.eval`` at all. A read-only question: nothing is
    written, nothing is deserialized beyond what the reader would read to
    reuse it."""
    from inspect_ai.log import read_eval_log

    log_dir = Path(log_dir)
    if reusable_log(log_dir, expected_n=expected_n, expected_ids=expected_ids,
                    expected_max_tokens=expected_max_tokens) is not None:
        return None
    paths = sorted(log_dir.glob("*.eval")) if log_dir.is_dir() else []
    if not paths:
        return f"no retained .eval log under {log_dir}"
    problems: List[str] = []
    for p in paths:
        if p.is_symlink() and not p.exists():
            problems.append(f"{p.name}: a dangling symlink")
            continue
        if not p.is_file():
            problems.append(f"{p.name}: not a regular file")
            continue
        try:
            log = read_eval_log(str(p))
        except Exception as e:  # noqa: BLE001 — the reader's refusal is the reason
            problems.append(f"{p.name}: unreadable ({type(e).__name__}: {str(e)[:120]}) — torn or truncated")
            continue
        status = getattr(log, "status", None)
        if status != "success":
            problems.append(f"{p.name}: status {status!r}, not 'success'")
            continue
        n = len(list(getattr(log, "samples", None) or []))
        if expected_n is not None and n != expected_n:
            problems.append(f"{p.name}: {n} samples, expected {expected_n}")
            continue
        if sample_errors(log):
            problems.append(f"{p.name}: carries errored samples")
            continue
        if expected_ids is not None:
            why = id_multiset_mismatch(log_sample_ids(log), expected_ids)
            if why is not None:
                problems.append(f"{p.name}: not this task's item selection ({why})")
                continue
        cap_why = _log_cap_mismatch(log, expected_max_tokens)
        if cap_why is not None:
            problems.append(f"{p.name}: {cap_why}")
            continue
        problems.append(f"{p.name}: not reusable")   # pragma: no cover — reusable_log said no
    return f"no completed .eval log under {log_dir}: " + "; ".join(problems)


def gsm8k_paired200_selection_ids(repo_root: Path = REPO_ROOT) -> List[str]:
    """The standard GSM8K plan's item selection, derivable from the PINNED
    corpus alone: the sorted inspect sample ids of the retired instrument's
    paired platinum rows (:func:`_gsm8k_paired_by_text`) — exactly
    ``plan.sample_ids`` whenever the plan is valid (the plan REFUSES a build in
    which the stock split lacks any of them, so the two never part on a store a
    driver wrote). Read by the launcher's pre-standup repair-eligibility probe,
    which has the pins but must not build the Inspect task."""
    return sorted(_gsm8k_paired_by_text(repo_root))


def _gsm8k_paired_by_text(repo_root: Path) -> Dict[str, str]:
    """``{inspect sample id -> our item_id}`` for the paired-200 GSM8K rows —
    the ONE derivation the plan builder and :func:`gsm8k_paired200_selection_ids`
    share (READING the retired instrument's row set is allowed; only its
    generation entry points are gated, PW3)."""
    from sisteer.capeval.tiers.parity import PARITY_N, gsm8k_parity_rows

    rows = gsm8k_parity_rows(_pins(repo_root), PARITY_N)
    return _paired_by_text(rows, text_key="question", sample_id=C.gsm8k_sample_id)


def bcb200_selection_ids(repo_root: Path = REPO_ROOT) -> List[str]:
    """The BigCodeBench-200 plan's item selection in LOG vocabulary (``"17"``),
    from the verified pin alone (bcb-bfcl-200): exactly ``plan.sample_ids``
    whenever the plan is valid. For the launcher's pre-standup probe."""
    return [C.bcb_log_id(i) for i in C.load_bcb_pin(repo_root)["ids"]]


def bfcl200_selection_ids(repo_root: Path = REPO_ROOT) -> List[str]:
    """The BFCL-200 plan's item selection (gorilla record ids) from the
    verified pin alone (bcb-bfcl-200)."""
    return [str(i) for i in C.load_bfcl_pin(repo_root)["ids"]]


def bcb_full_selection_ids(repo_root: Path = REPO_ROOT) -> List[str]:
    """The FULL BigCodeBench plan's item selection (log ids ``"17"``) from the
    verified full pin alone (capeval-full-tasks)."""
    return [C.bcb_log_id(i) for i in C.load_bcb_full_pin(repo_root)["ids"]]


def bfcl_full_selection_ids(repo_root: Path = REPO_ROOT) -> List[str]:
    """The FULL BFCL plan's item selection (gorilla record ids) from the
    verified full pin alone (capeval-full-tasks)."""
    return [str(i) for i in C.load_bfcl_full_pin(repo_root)["ids"]]


#: the tasks whose item selection the pre-standup probe can derive without the
#: Inspect task: row task -> the selection function
_SELECTION_IDS_OF_TASK: Mapping[str, Callable[[Path], List[str]]] = {
    C.GSM8K_STANDARD_TASK: gsm8k_paired200_selection_ids,
    C.inspect_task("bigcodebench_200"): bcb200_selection_ids,
    C.inspect_task("bfcl_200"): bfcl200_selection_ids,
    # capeval-full-tasks: the full siblings, from their own verified pins
    C.inspect_task(C.CONTROL_BIGCODEBENCH_FULL): bcb_full_selection_ids,
    C.inspect_task(C.CONTROL_BFCL_FULL): bfcl_full_selection_ids,
}


def retained_logs_report(spec: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """The launcher's pre-standup probe, run in the CLIENT interpreter against
    the staged tree (the launcher venv has no ``inspect_ai``; the reader is the
    authoritative one). ``spec`` = ``{"repo_root": <tree>, "checks": [{"log_dir",
    "row_task", "expected_n"}, ...]}`` -> ``{log_dir: reason | None}`` through
    :func:`retained_log_reason`, with the task's item selection compared where
    it is derivable from the pins (:data:`_SELECTION_IDS_OF_TASK`). A task with
    NO registered derivation leaves the selection to the driver's repair (which
    compares it against the plan after the standup) and says so in any reason;
    a REGISTERED derivation that fails (the pinned corpus unavailable) is
    ITSELF a refusal reason for every check of that task — fail closed, naming
    the error — since a complete, count-matching log of foreign ids would
    otherwise read as repairable. Never writes."""
    repo_root = Path(spec.get("repo_root") or REPO_ROOT)
    out: Dict[str, Optional[str]] = {}
    ids_by_task: Dict[str, Optional[Sequence[str]]] = {}
    notes_by_task: Dict[str, str] = {}
    #: tasks whose REGISTERED selection could not be derived: every one of their
    #: checks refuses (fail closed) — a complete, count-matching log of FOREIGN
    #: ids must not read as repairable because the ids could not be compared
    underivable: Dict[str, str] = {}
    for check in spec.get("checks") or []:
        task = str(check["row_task"])
        if task not in ids_by_task:
            derive = _SELECTION_IDS_OF_TASK.get(task)
            if derive is None:
                ids_by_task[task] = None
                notes_by_task[task] = "the item selection is the driver's to compare after the standup"
            else:
                try:
                    ids_by_task[task] = list(derive(repo_root))
                except Exception as e:  # noqa: BLE001 — named, never a silent skip
                    ids_by_task[task] = None
                    underivable[task] = (
                        f"the authorised task's item selection could not be derived here "
                        f"({type(e).__name__}: {str(e)[:120]}) — fail closed: a retained log cannot be "
                        "held to the plan's selection, so none counts as repairable before the standup")
        n = check.get("expected_n")
        why = retained_log_reason(Path(check["log_dir"]),
                                  expected_n=(int(n) if n is not None else None),
                                  expected_ids=ids_by_task[task])
        if task in underivable:
            why = underivable[task] if why is None else f"{why}; {underivable[task]}"
        elif why is not None and task in notes_by_task:
            why = f"{why} ({notes_by_task[task]})"
        out[str(check["log_dir"])] = why
    return out


def task_quality(results: Sequence[ItemResult]) -> Dict[str, Any]:
    """The task's two U5-5 quality numbers over the SCORED results (the
    final attempt of every (item, sample)), in the exact shape the summary
    payload and the aggregate row carry (``contracts.SUMMARY_PAYLOAD_FIELDS``).

    ``unparseable_rate = n_unparseable / n_answered`` over the results whose
    primary scorer RECORDED an answer; ``at_cap_rate = n_at_cap /
    n_stop_reason`` over the results carrying a stop reason, at-cap being
    ``stop_reason == STOP_REASON_AT_CAP``. Either rate is ``None`` when its
    denominator is 0 — a scorer that records no answer (MATH's sympy grader)
    reads as NOT RECORDED, never as 100 % unparseable, and the counts travel
    beside the rates so a per-cell roll-up pools them exactly instead of
    averaging rates over unequal denominators."""
    answered = [r for r in results if r.unparseable is not None]
    with_reason = [r for r in results if r.stop_reason is not None]
    n_unparseable = sum(1 for r in answered if r.unparseable)
    n_at_cap = sum(1 for r in with_reason if r.at_cap)
    return {
        "unparseable_rate": (n_unparseable / len(answered)) if answered else None,
        "n_unparseable": n_unparseable if answered else None,
        "n_answered": len(answered),
        "at_cap_rate": (n_at_cap / len(with_reason)) if with_reason else None,
        "n_at_cap": n_at_cap if with_reason else None,
        "n_stop_reason": len(with_reason),
    }


def _p99(values: Sequence[int]) -> Optional[int]:
    """``contracts.nearest_rank_percentile`` at 0.99 — an OBSERVED value, never
    an interpolation; ``None`` on an empty list."""
    return C.nearest_rank_percentile(values, 0.99) if values else None


def think_split_counts(results: Sequence[ItemResult], *, plan: TaskPlan) -> Dict[str, Any]:
    """The think-block and token-statistic summary fields
    (``contracts.SUMMARY_PAYLOAD_FIELDS``) over the FINAL attempt of every
    ATTEMPTED (item, sample) — the same list :func:`task_quality` runs over,
    never the emitted rows: the two block counts, ``n_unterminated_not_at_cap``
    (unterminated but NOT stopped by the cap — the analysis gate's anomaly), the
    split mode and policy labels, and mean / p99 completion tokens over the
    draws with recorded usage (``n_completion_tokens`` the denominator; ``None``,
    never 0, when none recorded)."""
    from sisteer.capeval.tiers.think_split import UNTERMINATED_POLICY

    tokens = [int(r.completion_tokens) for r in results if r.completion_tokens is not None]
    return {
        "n_think_block_present": sum(1 for r in results if r.think_block_present),
        "n_think_block_unterminated": sum(1 for r in results if r.think_block_unterminated),
        "n_unterminated_not_at_cap": sum(
            1 for r in results if r.think_block_unterminated and not r.at_cap),
        # GSM8K-SCORER-fix: the draws that ended their turn without a think block under implicit_open — scored whole, a
        # behavioural count the paper reports (the LeCun constitution ReLU answered 50 of 200 GSM8K items this way)
        "n_no_think_block": sum(1 for r in results if r.no_think_block),
        "think_split_mode": plan.think_split_mode,
        "unterminated_policy": UNTERMINATED_POLICY,
        "mean_completion_tokens": (sum(tokens) / len(tokens)) if tokens else None,
        "p99_completion_tokens": _p99(tokens),
        "n_completion_tokens": len(tokens),
    }


def outcome_record(results: Sequence[ItemResult], *, plan: TaskPlan, task: str,
                   thinking: str, summary: Mapping[str, Any],
                   split_counts: Mapping[str, Any],
                   producer_tree_sha: Optional[str],
                   sampling_ref: Optional[str] = None) -> Dict[str, Any]:
    """The ALL-ATTEMPTED outcome record of one task (plan S1.7 / S2.7;
    ``contracts.TASK_OUTCOMES_HEADER_FIELDS`` / ``TASK_OUTCOME_ITEM_FIELDS``):
    one record per (item, sample) FINAL attempt in ``results`` — no completion
    text — under a header whose counts are copied from the SAME summary the
    task files, so ``completion.verify_outcomes_against_summary`` holds by
    construction. ``n_attempted_items`` is the distinct item count (the
    summary's ``n``), ``n_attempted_samples`` the draw count (``len(items)``)."""
    items = [{
        "item_id": r.item_id, "sample_idx": int(r.sample_idx), "attempt": int(r.attempt),
        "scored": not r.unscored,
        "value": None if r.unscored else float(r.value),
        "unparseable": r.unparseable, "stop_reason": r.stop_reason, "at_cap": bool(r.at_cap),
        "think_block_present": bool(r.think_block_present),
        "think_block_unterminated": bool(r.think_block_unterminated),
        "stray_open_tag": bool(r.stray_open_tag),
        "content_empty_with_reasoning": bool(r.content_empty_with_reasoning),
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "seed": int(r.seed),
        # The attempt binding the ItemResult carries (the provider's response hook) rides the outcome item
        # too — the GSM8K reader binds draws off THIS record, and a shared engine files the same (item,
        # sample) for every arm and seed base, so a record without it can only be bound by a (task, sample)
        # lookup that is ambiguous by construction. ONLY when recorded (a default-off run's record is
        # byte-identical); the completion validator admits each shape with or without the pair
        # (contracts.TASK_OUTCOME_ITEM_BINDING_FIELDS)
        **({rev.ROW_REQUEST_ID_FIELD: r.request_id, rev.ROW_COMPLETION_ID_FIELD: r.completion_id}
           if r.request_id is not None else {}),
        # The six IFEval row facts on THIS task's items only
        # (contracts.TASK_OUTCOME_ITEM_FIELDS_IFEVAL) — the analysis reads the
        # four metrics per draw off the record, never off rows; every other
        # task's record is byte-identical; sampling_ref stays LAST
        **_ifeval_item_facts(r, task),
        **_ref_extra(sampling_ref),
        # llama-paper-rows (P19 / P21): the `none` split's close-tag stray, durable on the record
        # of a NO-SWITCH store ONLY (contracts.TASK_OUTCOME_ITEM_FIELDS_NO_SWITCH); every other
        # store's record keeps its shape
        **({"stray_close_tag": bool(r.stray_close_tag)} if plan.think_split_mode == C.THINK_SPLIT_NONE else {}),
    } for r in results]
    header: Dict[str, Any] = {
        "task": task, "item_set": plan.item_set, "thinking": thinking,
        "think_split_mode": split_counts["think_split_mode"],
        "unterminated_policy": split_counts["unterminated_policy"],
        "max_tokens": plan.max_tokens, "samples_per_item": int(plan.samples_per_item),
        "n_attempted_items": len({r.item_id for r in results}),
        "n_attempted_samples": len(items),
        "n_scored": summary["n_scored"], "n_unscored": summary["n_unscored"],
        "n_scored_samples": summary["n_scored_samples"],
        "n_unscored_samples": summary["n_unscored_samples"],
        "n_stop_reason": summary["n_stop_reason"], "n_at_cap": summary["n_at_cap"],
        "n_think_block_present": split_counts["n_think_block_present"],
        "n_think_block_unterminated": split_counts["n_think_block_unterminated"],
        "n_unterminated_not_at_cap": split_counts["n_unterminated_not_at_cap"],
        "n_completion_tokens": split_counts["n_completion_tokens"],
        "producer_tree_sha": producer_tree_sha,
        "items": items,
    }
    assert set(header) == set(C.TASK_OUTCOMES_HEADER_FIELDS)
    return header


def _ifeval_item_facts(r: ItemResult, task: str) -> Dict[str, Any]:
    """The IFEval outcome item's six row facts (``contracts.IFEVAL_ROW_FACT_FIELDS``)
    when ``task`` is ``inspect/ifeval``, else ``{}``. A result of that task
    WITHOUT the facts refuses (the row-fact reader ran on every sample)."""
    if task != C.IFEVAL_TASK:
        return {}
    missing = [k for k in C.IFEVAL_ROW_FACT_FIELDS if k not in r.row_facts]
    if missing:
        raise InspectServedError(
            f"{task} item {r.item_id} sample {r.sample_idx}: the result carries no IFEval row facts "
            f"({missing}) — ROW_FACTS_OF_TASK['ifeval'] must have read every sample")
    return {k: r.row_facts[k] for k in C.IFEVAL_ROW_FACT_FIELDS}


def ifeval_summary_fields(plan: TaskPlan, results: Sequence[ItemResult]) -> Dict[str, Any]:
    """The IFEval summary's extra fields (``contracts.IFEVAL_SUMMARY_FIELDS``):
    the stock task's own version, the pinned langdetect
    seed, and the PER-TRIAL means of the four IFEval metrics over the SCORED
    results of each sample index (a list indexed by trial) — prompt-level =
    the share of draws whose every instruction passed, instruction-level =
    followed / total instructions (IFEval's own definition, ``if_metric``'s) —
    so the retained ``.eval`` log's numbers reconcile against the adapter's
    (gate I4). ``{}`` for every other task (payloads byte-identical)."""
    if plan.name != C.IFEVAL_TASK_NAME:
        return {}
    by_idx: Dict[int, List[ItemResult]] = {}
    for r in results:
        if not r.unscored:
            by_idx.setdefault(int(r.sample_idx), []).append(r)
    trials = list(range(int(plan.samples_per_item)))

    def prompt_mean(key: str) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for t in trials:
            rs = by_idx.get(t, [])
            out.append(sum(1.0 for r in rs if r.row_facts[key]) / len(rs) if rs else None)
        return out

    def inst_mean(key: str) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for t in trials:
            rs = by_idx.get(t, [])
            total = sum(int(r.row_facts["num_instructions"]) for r in rs)
            out.append(sum(int(r.row_facts[key]) for r in rs) / total if total else None)
        return out

    fields = {"ifeval_task_version": C.IFEVAL_TASK_VERSION,
              "langdetect_seed": C.IFEVAL_LANGDETECT_SEED,
              "prompt_strict_mean_by_trial": prompt_mean("prompt_level_strict"),
              "prompt_loose_mean_by_trial": prompt_mean("prompt_level_loose"),
              "inst_strict_mean_by_trial": inst_mean("inst_level_strict"),
              "inst_loose_mean_by_trial": inst_mean("inst_level_loose")}
    assert tuple(fields) == C.IFEVAL_SUMMARY_FIELDS
    return fields


def _store_producer_tree_sha(cell_dir: Any) -> Optional[str]:
    """The store's own ``manifest.producer.tree_sha`` — the tree that wrote it
    (the driver records the block before any task runs) — or ``None`` when the
    store has no manifest yet or the block records none (a hand run, a test
    stub). Provenance on the outcome record, never verified against it."""
    manifest = completion.read_store_manifest(cell_dir)
    producer = manifest.get(C.MANIFEST_KEY_PRODUCER) if manifest else None
    sha = producer.get("tree_sha") if isinstance(producer, Mapping) else None
    return sha if isinstance(sha, str) and sha else None


def _retry_ids(results: Sequence[ItemResult]) -> List[str]:
    """The items of one sample whose generation stopped AT THE CAP, in the
    order they were scored — the MATH-500 retry's item set (U5-3)."""
    return [r.item_id for r in results if r.at_cap]


def _cap_bound_window_overflow(results: Sequence[ItemResult], *, budget: int,
                               max_model_len: int) -> Tuple[List[str], List[Tuple[str, int]]]:
    """The two ways a CAP-BOUND sample can fail to fit the served window, over
    one pass's rows: ``unmeasured`` — the ids of at-cap rows whose
    ``prompt_tokens`` is None (no usable target usage, E50 #5) — and ``over``
    — ``(item_id, prompt_tokens)`` of the measured at-cap rows with
    ``prompt + budget > max_model_len``. Rows that stopped normally are in
    neither. One definition shared by :func:`assert_pass_fits_window`
    (``budget`` = the pass cap) and :func:`assert_retry_fits_window`
    (``budget`` = the retry cap), so the two guards cannot disagree on the
    arithmetic."""
    budget, window = int(budget), int(max_model_len)
    at_cap = [r for r in results if r.at_cap]
    unmeasured = [r.item_id for r in at_cap if r.prompt_tokens is None]
    over = [(r.item_id, int(r.prompt_tokens)) for r in at_cap
            if r.prompt_tokens is not None and int(r.prompt_tokens) + budget > window]
    return unmeasured, over


def assert_retry_fits_window(task: str, results: Sequence[ItemResult], *,
                             retry_max_tokens: int, max_model_len: Optional[int]) -> None:
    """Refuse a truncation retry that the SERVED WINDOW cannot hold (U5-3).

    ``max_model_len`` is what the standup served (``endpoint.json`` — the same
    number the store identity and the resume guard use), and the check is
    EXACT rather than a table lookup: attempt 1 measured each item's
    ``prompt_tokens``, so ``prompt + retry_max_tokens > window`` is a fact
    about that item, not an estimate. That measurement is the TARGET
    generation's alone (:func:`sample_usage`, E40 H3) — MATH-500 runs with a
    separate grader model on the same proxy, and summing its prompt into the
    item's would refuse exactly the at-cap retries this exists to allow.
    This is a real reachable case, not
    defensive padding: ``serving/geometry.py``'s ``capability:gen_long`` is
    sized 4,608 (measured at a 2,048 cap), so a pass-2 run serving MATH-500
    there could not hold even the earlier 8,192-token completion — vLLM would
    clamp it and the retry would truncate a second time while LOOKING like a
    longer generation. MATH-500's own entry ``capability:gen_long_math``
    (17,664) holds prompt + the 16,384 pass from every measured prompt; a
    re-armed retry (above 16,384) fits NO registered window and refuses here.

    A retry with the window UNKNOWN is refused too — a fit that cannot be
    checked is not a fit (the driver always knows it).

    E50 #5: so is a retry whose PROMPT is unknown, and that is the same
    sentence read from the other end. ``prompt_tokens`` is ``None`` whenever
    the sample recorded no usable ``model_usage`` — an empty map, a missing
    attribute, or entries that summed to zero (:func:`sample_usage`) — and the
    ``prompt + retry > window`` arithmetic simply SKIPPED those rows, so an
    at-cap item with no usage passed a check it was never subjected to and was
    then re-requested into a window nobody had measured it against. Nothing
    upstream guarantees the usage is there: inspect_ai fills ``model_usage``
    from whatever the provider returned, and a provider that omits ``usage``
    (or a cached / replayed log) leaves it empty. An unmeasured at-cap row is
    therefore refused with the same remedy as an unknown window — never
    silently admitted, and never counted as a fit."""
    if not retry_max_tokens:
        return
    if max_model_len is None:
        if any(r.at_cap for r in results):
            raise InspectServedError(
                f"{task}: {sum(1 for r in results if r.at_cap)} sample(s) stopped at the cap "
                f"and would be re-requested at {retry_max_tokens} tokens, but the SERVED "
                "window was not supplied — refusing to re-request into a window nobody "
                "checked (pass the standup's max_model_len, or disable the retry with 0)")
        return
    unmeasured, over = _cap_bound_window_overflow(results, budget=retry_max_tokens,
                                                  max_model_len=max_model_len)
    if unmeasured:
        raise InspectServedError(
            f"{task}: {len(unmeasured)} sample(s) stopped at the cap and would be "
            f"re-requested at {retry_max_tokens} tokens, but their PROMPT LENGTH was not "
            f"measured (first: {unmeasured[:3]}) — the sample recorded no usable "
            "model_usage for the target generation, so prompt + retry cannot be compared "
            f"against the served window max_model_len={max_model_len}. An unmeasured "
            "at-cap row is refused, never skipped: skipping it re-requests into a window "
            "nobody checked, which is the failure this check exists to prevent. Re-run the "
            "task so the provider records usage, or disable the retry with 0")
    if over:
        worst = max(over, key=lambda p: p[1])
        raise InspectServedError(
            f"{task}: the truncation retry at {retry_max_tokens} completion tokens does "
            f"not fit the SERVED window max_model_len={max_model_len} for "
            f"{len(over)} item(s) — worst {worst[0]!r} needs {worst[1]} prompt + "
            f"{retry_max_tokens} = {worst[1] + retry_max_tokens} tokens. vLLM would "
            "clamp the completion and the retry would truncate a second time while "
            "reading as a longer generation; refusing. Serve this task at a window "
            f">= prompt + {retry_max_tokens} (serving/geometry.py: "
            "capability:gen_long_math is 17,664 — MATH-500's own window in both arms; "
            "capability:gen_long_thinking 10,752, capability:gen_long 4,608), "
            "or disable the retry with 0")


def _expected_draws(plan: TaskPlan) -> Optional[int]:
    """The DRAW count a plan is complete at: the subset's
    ``expected_draws`` on a draw-subset rung; ``expected_n x k`` on the IFEval
    task otherwise (pass 1: 541 x 2 — the k = 2 instrument's completeness is a
    draw fact too, and every IFEval marker carries the two counts); ``None`` for
    every other task, whose markers stay byte-identical."""
    if plan.expected_draws is not None:
        return int(plan.expected_draws)
    if plan.row_task == C.IFEVAL_TASK and plan.expected_n is not None:
        return int(plan.expected_n) * int(plan.samples_per_item)
    return None


def _ref_extra(sampling_ref: Optional[str]) -> Dict[str, Any]:
    """``{"sampling_ref": <digest>}`` when the run has one, else ``{}`` — legacy
    rows and summaries byte-identical."""
    return {C.SAMPLING_REF_FIELD: sampling_ref} if sampling_ref else {}


def _optional_terms(plan: TaskPlan) -> Dict[str, Any]:
    """The plan's OPTIONAL sampling terms, only those set — the row
    fields of ``contracts.SAMPLING_OPTIONAL_ROW_FIELDS``, in that order."""
    return {name: getattr(plan, name) for name in C.SAMPLING_OPTIONAL_ROW_FIELDS
            if getattr(plan, name, None) is not None}


def _protocol_term(plan: TaskPlan) -> Dict[str, Any]:
    """``{"protocol_temperature": <declared T>}`` under ``protocol`` — the
    benchmark's declared temperature recorded beside the EXECUTED
    ``temperature``; ``{}`` under every other mode, so a card / greedy summary
    is byte-identical."""
    t = getattr(plan, "protocol_temperature", None)
    return {} if t is None else {"protocol_temperature": float(t)}


def assert_pass_fits_window(task: str, results: Sequence[ItemResult], *,
                            max_tokens: Optional[int], max_model_len: Optional[int]) -> None:
    """Refuse a PASS whose cap-bound samples the SERVED WINDOW could not hold
    (MATH-500 at ONE draw of the cap, 16,384 — the check that
    replaces :func:`assert_retry_fits_window` as the instrument's window
    guard).

    EXACT and post hoc over what the pass measured. Every sample that stopped
    AT THE CAP carries the target generation's ``prompt_tokens``
    (:func:`sample_usage`, E40 H3), and ``prompt + max_tokens > max_model_len``
    means vLLM clamped that request's completion budget to the window's
    remainder: the sample stopped at the WINDOW, not at the cap, while its
    row would read ``at_cap`` at ``max_tokens_used = max_tokens``. That is a
    false cap-bound reading — the cap the instrument claims is not the cap
    that ran — so the task refuses BY NAME (the items, the worst prompt, the
    two numbers) rather than filing it; nothing is emitted. Reachable, not
    padding: MATH's 16,384 cap overflows ``capability:gen_long`` (4,608) and
    ``capability:gen_long_thinking`` (10,752) from ANY prompt (the sizing is
    stated on ``contracts.GEOMETRY_CLASS_GEN_LONG_MATH``; MATH's own entries
    are per PROMPT CLASS — ``capability:gen_long_math`` 17,664 = prompt max
    1,041 + the cap for the bare / safety-SI cells, and
    ``capability:gen_long_math_longsi`` 24,064 = prompt max 7,622 + the cap
    for the constitution / irrelevant SIs), and this check is what
    catches a standup that served MATH-500 anywhere smaller than the entry
    its prompt class resolves to. A cap-bound sample whose prompt was NOT
    measured (no usable target usage — ``prompt_tokens`` None) cannot be told
    from a clamped one and is refused with the same remedy (E50 #5's
    discipline: never skipped, never counted as a fit). Rows that stopped
    normally are not the check's business — an early stop fit by construction.

    Nothing is checked when ``max_model_len`` is None (no standup record —
    a direct caller; the driver always supplies the served window) or
    ``max_tokens`` is None (a task that pins its cap inside its solver: the
    plan states no cap to compare)."""
    if max_model_len is None or max_tokens is None:
        return
    unmeasured, over = _cap_bound_window_overflow(results, budget=max_tokens,
                                                  max_model_len=max_model_len)
    if unmeasured:
        raise InspectServedError(
            f"{task}: {len(unmeasured)} sample(s) stopped at the {max_tokens}-token cap but "
            f"their PROMPT LENGTH was not measured (first: {unmeasured[:3]}) — the sample "
            "recorded no usable model_usage for the target generation, so prompt + cap cannot "
            f"be compared against the served window max_model_len={max_model_len}, and a "
            "cap-bound sample the window clamped below the cap cannot be told from one that "
            "reached it. Refused, never skipped (E50 #5): re-run the task so the provider "
            "records usage")
    if over:
        worst = max(over, key=lambda p: p[1])
        raise InspectServedError(
            f"{task}: the pass cap {max_tokens} does not fit the SERVED window "
            f"max_model_len={max_model_len} for {len(over)} cap-bound item(s) — worst "
            f"{worst[0]!r} needs {worst[1]} prompt + {max_tokens} = {worst[1] + max_tokens} "
            "tokens. vLLM clamped those completions to the window's remainder, so they "
            "stopped at the WINDOW while reading as cap-bound draws at the pass cap; refusing "
            "to file them. Serve this task at a window >= prompt + cap (serving/geometry.py: "
            "capability:gen_long_math is 17,664 — MATH-500's own window for the bare / safety-SI "
            "prompt classes in both arms, capability:gen_long_math_longsi 24,064 for the "
            "constitution / irrelevant SIs; capability:gen_long_thinking 10,752 and "
            "capability:gen_long 4,608 cannot hold it)")


def _plan_readers(plan: TaskPlan) -> Dict[str, Any]:
    """The per-task reader kwargs ``results_from_log`` takes for ``plan``'s task
    — its row-fact reader, its primary value key and its row task (the seed
    rule's dispatch key) — spelled once for the four call sites."""
    return dict(row_facts=ROW_FACTS_OF_TASK.get(plan.name),
                primary_value_key=PRIMARY_VALUE_KEY.get(plan.name),
                row_task=plan.row_task)


async def run_one_task(plan: TaskPlan, *, model: Any, store: Any, template: Mapping[str, Any],
                       thinking: Optional[str] = None, concurrency: int = 24,
                       route_virtual_model: Optional[str] = None,
                       emitter: Any = None,
                       reasoning_parser: Optional[str] = None,
                       limits: Optional[Mapping[str, int]] = None,
                       max_model_len: Optional[int] = None,
                       out_of_support: bool = False,
                       ledger_path: Optional[Path] = None,
                       sampling_ref: Optional[str] = None,
                       attempt_scope: Optional[C.AttemptScope] = None,
                       row_identity: Optional[Mapping[str, Any]] = None,
                       generation: Optional["GenerationPhase"] = None) -> Dict[str, Any]:
    """Run one :class:`TaskPlan` end to end and file its rows.

    ``generation``: the row's generation-phase facts
    (``cpu_verify.GenerationPhase``: the row dir, the
    cell, the GPU FunctionCall id, the driver's concurrency). REQUIRED for a
    ``plan.generation_only`` task, whose whole product is the UNSCORED ``.eval``
    under the store's task log dir plus the generation SIDECAR and the PENDING
    MARKER in ``<row_dir>/generation/`` (``cpu_verify.write_generation_pending``) —
    no ``results_from_log``, no rows, no summary, no marker in the store; the CPU
    ``verify`` phase (``served_capability_run.py verify``) scores the log, binds the
    scores back by ``(id, epoch)`` and runs the SAME filing tail
    (:func:`_file_task_results`) the coupled path runs here. On a coupled plan it
    is only READ: a pending marker for this task in the row dir refuses the task
    by name (the verifier owns it; it is never re-generated). ``None`` on every
    solo run, byte-identical.

    ``row_identity`` (the cross-eval guard): the per-row object identity block
    (``contracts.row_identity_block``: ``object_artifact_sha256``, ``route_kind``,
    ``view``, ``producer_tree_sha``, ``sampling_mode``) stamped into the
    ``extra_json`` of EVERY emitted row — each per-sample ``acc@<idx>`` row, the
    per-item mean ``acc`` row and the two aggregate rows — beside the existing
    per-row ``thinking`` and ``sampling_ref``. ``None`` (a run without
    ``--bank-view``) stamps nothing: byte-identical rows.

    ``attempt_scope`` (the attempt-scoped
    producer contract): when the driver runs under
    ``--attempt-id``, the ledger-derived ``n_http_rejected_retried`` is counted
    over ONE attempt's rows — this attempt's when the task GENERATED here, the
    recorded prior generation attempt's when a complete retained ``.eval`` was
    REUSED (a finalisation: the log is that attempt's evidence) — through
    ``contracts.scoped_rejected_count``; a reuse with no recorded generation
    attempt refuses. ``None`` = the whole-ledger count, byte-identical.

    ``ledger_path`` is the run's usage ledger file — the durable
    ``<out_root>/usage_ledger.jsonl`` the driver's manifest records — when
    the run has one: the summary then carries ``n_http_rejected_retried``,
    the ledger's count of THIS task's rejected calls (:func:`_ledger_rejected_count`),
    so the summary can never claim no draw was re-sampled while the ledger
    holds the upstream's refusals. ``None`` (no ledger) files null, never 0.

    k samples per item (P28; since U8 / PW-sampling EVERY task with
    ``plan.samples_per_item > 1``, not MATH-500 alone) is k separate
    ``eval_async`` calls with the sample index in the ContextVar, so every
    request carries its own seed and no ``n`` is ever sent (E4: a lens
    fork may run unsteered; and a steered prefill cannot be cached, so k
    IS k requests); the per-SAMPLE rows
    take the metric ``acc@<k>`` and the per-ITEM mean keeps the bare
    ``acc``, because CellStore completion identity is ``(task, item_id,
    metric)`` and four samples of one item cannot all be ``acc``. Every
    per-sample row states its own ``stop_reason`` / ``unparseable`` /
    ``seed``, and the U5-5 quality rates are over the (item, sample)
    results, so at-cap and unparseable are accounted PER SAMPLE. A
    model-graded refusal-rate task (OR-Bench, XSTest) samples too: its
    per-item value is the mean refusal over k. ``reasoning_parser`` is recorded
    (summary + aggregate row) so the per-task ``n_content_empty_with_reasoning``
    count reads against the parser that produced it. ``limits`` is the
    driver's active limit set: with any limit active, or a run narrower
    than ``plan.expected_n``, the task ends with a PARTIAL marker (R2),
    never a done one; the completeness check is against ``plan.n_run``
    (what this run selected), the marker records the frozen full count.

    ``thinking`` defaults to ``plan.thinking`` — the arm the plan's cap was
    chosen from, and under per-task thinking the only one that is right for
    this task (PW20b-D). An explicit value still wins, for callers that
    already pass one.

    TRUNCATION RETRY (U5-3, MATH-500 only). When ``plan.retry_max_tokens`` is
    set, every sample of the pass whose ``stop_reason`` is
    :data:`STOP_REASON_AT_CAP` is re-requested ONCE, at that cap, with the
    SAME sample index (hence the same seed, ``sample_seed(item, idx)``) and
    the same items; the retry's own log lives in a ``__a2`` dir and resumes
    like any other. A per-(item, sample) value is the FINAL attempt's, so the
    avg@k mean is over final attempts; every row records ``attempt`` and
    ``max_tokens_used``, and the per-item mean row also records ``n_retried``
    (how many of that item's samples were retried) and, when any was, the
    largest cap that contributed. ``max_model_len`` is the served window the
    retry must fit (:func:`assert_retry_fits_window`) — and, retry or not,
    the window every cap-bound sample of the PASS must have fit
    (:func:`assert_pass_fits_window`: a sample the window clamped
    below the cap is a false at-cap reading and refuses the task).

    ERRORED ITEMS (E56 (3)). An item inspect could not generate for,
    after every retry it was allowed, REFUSES this task, naming each item, the
    draw it was lost on and the HTTP status its message carried: a row that
    vanishes is invisible to every count downstream, and a payload's
    ``n_rejected: 0`` cannot show an item that never reached the payload. There
    is no opt-in that completes such a task — inspect's ``fail_on_error`` stays
    ON, so the first real error aborts the eval, and nothing is filed and no
    marker is written. Every task this tier files therefore has
    ``n == n_scored + n_unscored`` over a full attempted set. The deferred
    completion path and what a re-introduction owes are in
    ``docs/tinycap_interfaces.md`` §2, "Deferred: errored-item completion".

    ``route_virtual_model`` is recorded in the summary AND is load-bearing
    (E40 H3): it names the TARGET's ``model_usage`` key, so every row's
    ``prompt_tokens`` is the target generation's rather than the sum over the
    target and whatever grader the task ran. Without it the retry cannot be
    armed at all."""
    # bcb-bfcl-200 §S2b: a ladder rung-2 task ALREADY validly
    # filed in this store — the ONE strict validator (rung_probe.ladder_task_
    # partial_reason: PARTIAL over exactly the selection at the cap, the
    # outcomes record agreeing) says so — is skipped without a request, so a
    # resumed two-leg rung (the driver is always handed the FULL plan and the
    # full hashed tuple, never a trimmed one that would key another store)
    # re-generates nothing it has
    if plan.selection is not None:
        from sisteer.capeval.tinycap import rung_probe as _RP
        why = _RP.ladder_task_partial_reason(store.cell_dir, plan.row_task, log_ids=plan.selection,
                                             cap=int(plan.max_tokens), n_expected_full=plan.expected_n)
        if why is None:
            prior = completion.read_task_summary(store.cell_dir, plan.row_task) or {}
            return {**prior, "skipped": "ladder rung complete (a valid PARTIAL over exactly the selection)"}
    # A DRAW-subset rung ALREADY
    # validly filed — the ONE strict validator (rung_probe.draw_subset_task_reason:
    # DONE over exactly the subset's draws at the cap, the marker's n_draws ==
    # n_expected_draws, the outcome record agreeing) says so — is skipped without a
    # request, so a resumed rung container issues ZERO requests for a complete store
    if plan.draw_subset is not None:
        from sisteer.capeval.tinycap import rung_probe as _RP
        why = _RP.draw_subset_task_reason(store.cell_dir, plan.row_task, draws=plan.draw_subset,
                                          cap=int(plan.max_tokens))
        if why is None:
            prior = completion.read_task_summary(store.cell_dir, plan.row_task) or {}
            return {**prior, "skipped": "draw-subset rung complete (DONE over exactly the subset's draws)"}
        if completion.done_marker_path(store.cell_dir, plan.row_task).is_file():
            # a DONE marker that is NOT a valid rung over these draws (an item-level
            # producer's rung, a torn store, another cap) refuses BY NAME before any
            # request — generating over it would re-write the marker, a silent widen
            raise InspectServedError(
                f"{plan.row_task}: the store at {store.cell_dir} holds a DONE marker that is NOT a valid draw-subset "
                f"rung over the subset's {len(plan.draw_subset)} draws at cap {plan.max_tokens} ({why}) — repair = "
                "delete this rung store and re-run; a rung is never widened or re-read around (plan §5.4)")

    from inspect_ai import eval_async

    from sisteer.capeval.tiers import tier0

    if plan.generation_only:
        # the generation-only leg needs somewhere to leave its sidecar +
        # pending marker — the row dir the driver's --row-engine-record names;
        # a hand run without one refuses
        if generation is None:
            raise InspectServedError(
                f"{plan.row_task}: a generation-only plan needs the row's generation facts (the row dir the "
                f"sidecar and the pending marker are written to) — pass {C.FLAG_ROW_ENGINE_RECORD} beside "
                f"{C.FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_GENERATE}")
        if int(plan.samples_per_item) != 1 or plan.retry_max_tokens or plan.draw_subset is not None:
            raise InspectServedError(
                f"{plan.row_task}: the two-phase contract covers the one-pass, k = 1 BigCodeBench leg — this plan "
                f"carries samples_per_item={plan.samples_per_item}, retry_max_tokens={plan.retry_max_tokens}, "
                f"draw_subset={'set' if plan.draw_subset is not None else None}")
    elif generation is not None and plan.sandbox is not None:
        # a COUPLED sandboxed task under a row whose generation phase left this task
        # pending: the CPU verifier owns it — never re-generated here
        pending = _CV.pending_marker_path(generation.row_dir, generation.cell, plan.row_task, generation.store)
        if pending.is_file():
            raise InspectServedError(
                f"{plan.row_task}: the row dir carries the pending marker {pending} — this task was GENERATED and "
                f"awaits its CPU verifier (`served_capability_run.py verify`); a coupled run never re-generates "
                "or scores over it (plan §2.2.3)")

    if plan.sandbox is not None and not plan.generation_only:
        # bcb-bfcl-200-widen D73: the sandboxed task (SANDBOXED_TASK_NAMES = bigcodebench_200
        # alone) verifies each item by `sandbox().exec(["python", "-c", code])`; on rung 2 a
        # 16k-token completion overflows Modal's 65,536-byte ARG_MAX client-side, before the
        # sandbox runs anything, and Inspect's retries repeat the same deterministic
        # overflow (ladder_report §4b). Installed HERE — in the client-venv driver process,
        # once, before the task's first eval_async — on the provider class the thin task
        # names, so an oversize program rides the provider's write_file channel instead of
        # argv; every exec below the threshold is byte-identical (verifier_transport). A plan
        # naming another provider refuses: the wrapper would not cover the class Inspect
        # instantiates, and an unprotected verifier is the §4b failure again.
        from sisteer.capeval.tiers import sandbox_owned as _SO
        from sisteer.capeval.tinycap import verifier_transport as _VT
        if plan.sandbox.get("provider") != _SO.PROVIDER_NAME:
            raise InspectServedError(
                f"{plan.row_task}: the plan names sandbox provider {plan.sandbox.get('provider')!r}, not "
                f"{_SO.PROVIDER_NAME!r} — the verifier transport (D73) wraps that provider's exec and "
                "would not cover another; refusing to run an unprotected verifier")
        _VT.install_exec_spill(_SO.OwnedModalSandboxEnvironment, ledger_path=_VT.ledger_path_for(store))

    t0 = time.time()
    if thinking is not None and thinking == C.THINKING_PER_TASK:
        raise InspectServedError(
            f"{plan.row_task}: run_one_task was given the {C.THINKING_PER_TASK!r} MODE as "
            "an arm — a task runs under one arm; the per-task map is resolved into "
            "plan.thinking at plan time (plans_for_atoms)")
    # a per-task ARM (contracts.THINKING_ARMS — the no-switch `not_available` included)
    arm = plan.thinking if thinking is None else C.check_thinking_arm(thinking)
    thinking = arm
    em = emitter if emitter is not None else tier0._Emitter(
        store, dict(template), dict(C.TINYCAP_DOMAIN_OF), C.INSPECT_TASK_VERSION)
    task = plan.row_task
    n_expect = plan.n_expected_run
    # E40 H3: the model_usage key whose entry IS this task's prompt/completion
    # tokens. A model-graded task files the grader's usage in the same map, and
    # the truncation retry's window check reads prompt_tokens as the length it
    # is about to re-send — so the retry cannot be armed without knowing which
    # entry is the target's (a fit that cannot be checked is not a fit).
    target_usage_model = (C.inspect_model_name(route_virtual_model)
                          if route_virtual_model else None)
    if plan.retry_max_tokens and target_usage_model is None:
        raise InspectServedError(
            f"{task}: the truncation retry is armed ({plan.retry_max_tokens} tokens) but no "
            "route_virtual_model was passed, so the TARGET's entry in a sample's model_usage "
            "cannot be told from the grader's — and the window check would measure the sum. "
            "Pass the standup's virtual model (run_inspect_tasks does), or disable the retry "
            "with 0")
    # P42: the per-row identity stamp (validated by its builder; {} = none)
    ident: Dict[str, Any] = dict(row_identity) if row_identity else {}
    by_item: Dict[str, List[float]] = {}
    per_sample: Dict[int, List[ItemResult]] = {}
    reused = 0
    n_retried_requests = 0
    generated = {"n": 0}     # task calls THIS run issued (a reuse-only task is a finalisation)
    gen_logs: Dict[int, Tuple[Any, Path]] = {}   # the unscored pass log per trial (generation-only)
    # the ALL-OR-NOTHING reuse contract under the attempt contract: a task's retained evidence is
    # ONE attempt's. `contracts.resolve_attempt_kinds` accepts exactly one generation attempt per store and the
    # analysis' I1 / I2 hold that attempt's rows and records to EVERY draw, so a resume that reused trial 0's `.eval`
    # (a1's) and generated trial 1 (a2's) filed 1,082 outcomes against 541 accepted requests — a store no
    # finalisation could repair. So under an attempt scope: (1) BEFORE any request, every trial's pass log is
    # checked for reuse; a PARTIAL set (some trials retained, some not) is ABANDONED whole (renamed, never deleted)
    # and this attempt generates every trial; (2) if a trial's truncation-retry log is missing while its pass log was
    # reused, the same rule applies (`_MixedEvidence` -> abandon -> run the task whole); (3) a full set is REUSED whole
    # (a finalisation, as before). Without an attempt scope nothing changes: partial reuse is harmless where no
    # attempt is attributed. The re-draw uses the same per-(item, trial) seeds — the same instrument.
    allow_reuse = True
    if attempt_scope is not None:
        have: Dict[int, bool] = {}
        for idx in range(plan.samples_per_item):
            ids_idx = plan.ids_for(idx)
            if plan.sample_ids_by_sample is not None and not ids_idx:
                continue
            n_expect_idx = len(ids_idx) if plan.sample_ids_by_sample is not None else n_expect
            have[idx] = reusable_log(C.inspect_log_dir(store.cell_dir, task, idx, 1), expected_n=n_expect_idx,
                                     expected_ids=ids_idx, expected_max_tokens=plan.max_tokens) is not None
        if any(have.values()) and not all(have.values()):
            gone = abandon_task_logs(store, task, samples_per_item=plan.samples_per_item, attempt_id=attempt_scope.attempt_id)
            allow_reuse = False
            print(f"[inspect-served] {task}: attempt {attempt_scope.attempt_id} found a PARTIAL set of retained logs "
                  f"(trials {sorted(i for i, ok in have.items() if ok)} reusable, {sorted(i for i, ok in have.items() if not ok)} "
                  f"not) — the attempt contract keeps a task's evidence to ONE attempt, so {len(gone)} retained log(s) were "
                  f"set aside (renamed *.eval{ABANDONED_LOG_SUFFIX}{attempt_scope.attempt_id}) and every trial is generated "
                  "by this attempt (review S1: all-or-nothing reuse)", flush=True)

    async def _eval(log_dir: Path, *, sample_ids, limit, max_tokens):
        if attempt_scope is not None and allow_reuse and reused:
            # about to GENERATE after a REUSE of this task under the attempt contract: mixed evidence
            raise _MixedEvidence()
        generated["n"] += 1
        if sample_ids is not None and limit is not None:
            raise InspectServedError(
                f"{task}: plan carries both sample_ids and limit={limit} — "
                "Inspect refuses the pair; a smoke cap on an id-selected task "
                "takes the first N ids (limit_plan)")
        kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        # the run's sampling row at EVAL level (PB5: beats a task's own config —
        # inspect_evals' math 0.5 included); the provider applies the same row
        # on every body, top_k via extra_body (sampling_params)
        if plan.temperature is not None:
            kwargs["temperature"] = float(plan.temperature)
        if plan.top_p is not None:
            kwargs["top_p"] = float(plan.top_p)
        if plan.top_k is not None:
            kwargs["top_k"] = int(plan.top_k)
        # the row's OPTIONAL presence penalty at eval level too (a
        # GenerateConfig field); min_p / repetition_penalty ride in extra_body
        # from the provider — a legacy row sets nothing here
        if plan.presence_penalty is not None:
            kwargs["presence_penalty"] = float(plan.presence_penalty)
        if plan.sandbox is not None and not plan.generation_only:
            # a SANDBOXED task bounds its live
            # sandboxes at the served S (the provider defines no default
            # semaphore — unbounded otherwise) and always cleans up; every
            # other task's eval call is byte-identical to before
            kwargs["max_sandboxes"] = int(concurrency)
            kwargs["sandbox_cleanup"] = True
        if plan.generation_only:
            # the GPU phase GENERATES
            # only — no scorer runs, no sandbox exists on the task, no sandbox kwarg
            # rides; the CPU verifier scores the retained log (`cpu_verify`)
            kwargs["score"] = False
        logs = await eval_async(
            plan.task, model=model, log_dir=str(log_dir),
            log_format="eval", sample_id=sample_ids,
            limit=limit, max_connections=int(concurrency),
            # the eval-level operational keys (max_retries / the retry-loop
            # timeout) — the ONE mapping the task's record reads too
            **INSPECT_EVAL_OPERATIONAL,
            # a transient sample error is retried by Inspect; a
            # persistent one FAILS the eval, whose log is then
            # not reusable and whose rows are never filed. The
            # alternative (collect the errors, refuse at adapt
            # time) deadlocks resume: the log reads `success`,
            # so every re-run reuses it and refuses again.
            retry_on_error=2,
            # E56 (3): always ON. A sample inspect could not
            # generate aborts the eval, `results_from_log` refuses naming it,
            # and the task is re-run — there is no mode that completes a task
            # with a hole in it.
            fail_on_error=True, **kwargs)
        return logs[0]

    tok = _TASK.set(task)
    try:
      for _round in (1, 2):
        try:
          for idx in range(plan.samples_per_item):
            # under a draw subset each trial draws ITS items alone; a
            # trial the subset names no draw of is skipped — no log dir, no call
            ids_idx = plan.ids_for(idx)
            if plan.sample_ids_by_sample is not None and not ids_idx:
                continue
            n_expect_idx = len(ids_idx) if plan.sample_ids_by_sample is not None else n_expect
            itok = _SAMPLE_IDX.set(idx)
            try:
                d = _log_dir(store, task, idx)
                # a retained log is this plan's evidence only at this
                # plan's cap — never relabelled max_tokens_used on reuse
                log = (reusable_log(d, expected_n=n_expect_idx, expected_ids=ids_idx,
                                    expected_max_tokens=plan.max_tokens) if allow_reuse else None)
                if log is not None:
                    reused += 1
                else:
                    log = await _eval(d, sample_ids=ids_idx, limit=plan.limit,
                                      max_tokens=plan.max_tokens)
                # the pass log's id multiset IS the plan's selection — reused or
                # fresh — before any outcome is read off it
                _refuse_log_id_mismatch(task, idx, log, ids_idx, what="pass")
                if plan.generation_only:
                    # the UNSCORED log is the GPU phase's whole product;
                    # nothing is read off it here (its samples carry no score yet)
                    gen_logs[idx] = (log, d)
                    continue
                try:
                    res = results_from_log(
                        log, sample_idx=idx, primary_scorer=plan.primary_scorer,
                        paired_ids=plan.paired_ids, attempt=1,
                        max_tokens_used=plan.max_tokens,
                        target_model=target_usage_model,
                        **_plan_readers(plan))
                except InspectServedError:
                    # bcb-bfcl-200, strict: a sandboxed task's
                    # errored log is REFUSED like any other — but its diagnostics
                    # (which items, which errors) are preserved beside the store
                    # before the refusal propagates, so a creation outage is
                    # diagnosable after the container is gone
                    if plan.sandbox is not None:
                        write_sandbox_failures(store, task, log, sample_idx=idx)
                    raise
                # -- the one-pass fit: a cap-bound sample the served
                #    window clamped BELOW the cap is a false at-cap reading —
                #    refused by name before anything is filed. Only where the
                #    target's usage is attributable (the driver always names
                #    route_virtual_model): without it prompt_tokens is the
                #    target+grader SUM (sample_usage), which over-refuses
                if target_usage_model is not None:
                    assert_pass_fits_window(task, res, max_tokens=plan.max_tokens,
                                            max_model_len=max_model_len)
                # -- the truncation retry (U5-3): the samples the cap cut,
                #    re-requested ONCE at the retry cap, same seed, same items
                if plan.retry_max_tokens:
                    assert_retry_fits_window(
                        task, res, retry_max_tokens=plan.retry_max_tokens,
                        max_model_len=max_model_len)
                    ids = _retry_ids(res)
                    if ids:
                        rd = _log_dir(store, task, idx, attempt=2)
                        rlog = (reusable_log(rd, expected_n=len(ids), expected_ids=ids,
                                             expected_max_tokens=plan.retry_max_tokens) if allow_reuse else None)
                        if rlog is not None:
                            reused += 1
                        else:
                            print(f"[inspect-served] {task} sample {idx}: {len(ids)} of "
                                  f"{len(res)} samples stopped at the {plan.max_tokens}-token "
                                  f"cap — re-requesting them once at {plan.retry_max_tokens}",
                                  flush=True)
                            rlog = await _eval(rd, sample_ids=ids, limit=None,
                                               max_tokens=plan.retry_max_tokens)
                            n_retried_requests += len(ids)
                        # the retry log's id multiset IS the requested retry set: a
                        # set difference could not see one requested id missing
                        # behind another duplicated (the missing item would have kept
                        # its at-cap pass result under a row claiming a retry)
                        _refuse_log_id_mismatch(task, idx, rlog, ids, what="retry")
                        retried = {r.item_id: r for r in results_from_log(
                            rlog, sample_idx=idx, primary_scorer=plan.primary_scorer,
                            paired_ids=plan.paired_ids, attempt=2,
                            max_tokens_used=plan.retry_max_tokens,
                            target_model=target_usage_model,
                            **_plan_readers(plan))}
                        res = [retried.get(r.item_id, r) for r in res]
                per_sample[idx] = res
            finally:
                _SAMPLE_IDX.reset(itok)
          break
        except _MixedEvidence:
            # a reuse then a generation of one task under the attempt contract — abandon every retained log of
            # the task (nothing was generated yet: the rule fires BEFORE the first task call) and run it WHOLE, once
            assert generated["n"] == 0 and _round == 1
            gone = abandon_task_logs(store, task, samples_per_item=plan.samples_per_item, attempt_id=attempt_scope.attempt_id)
            print(f"[inspect-served] {task}: attempt {attempt_scope.attempt_id} would have GENERATED after REUSING "
                  f"{reused} retained log(s) of this task (a missing truncation-retry log beside a reused pass log) — "
                  f"{len(gone)} retained log(s) set aside (*.eval{ABANDONED_LOG_SUFFIX}{attempt_scope.attempt_id}); the task "
                  "is generated whole by this attempt (review S1: all-or-nothing reuse)", flush=True)
            allow_reuse, reused, n_retried_requests = False, 0, 0
            per_sample.clear()
    finally:
        _TASK.reset(tok)

    if plan.generation_only:
        # the GPU phase ends HERE —
        # the sidecar carries every generation-side fact the filing tail needs
        # (the regime stamps read off the live provider NOW, the client record,
        # the ledger's rejected count at generation end, the plan) and the pending
        # marker says the task awaits its CPU verifier; nothing is filed
        assert generation is not None
        return _generation_exit(plan, store=store, task=task, gen_logs=gen_logs, generation=generation,
                                model=model, thinking=thinking, route_virtual_model=route_virtual_model,
                                target_usage_model=target_usage_model, reasoning_parser=reasoning_parser,
                                limits=limits, out_of_support=out_of_support, ledger_path=ledger_path,
                                sampling_ref=sampling_ref, attempt_scope=attempt_scope, ident=ident,
                                reused=reused, generated=generated, t0=t0, max_model_len=max_model_len,
                                n_retried_requests=n_retried_requests)

    draw_rung = plan.draw_subset is not None
    if draw_rung:
        # completeness per TRIAL against exactly the items the subset
        # names at that trial; a trial that drew more or fewer is a torn store
        for idx, res in per_sample.items():
            want_ids = plan.ids_for(idx) or []
            if sorted(r.item_id for r in res) != sorted(want_ids):
                raise InspectServedError(
                    f"{task} sample {idx}: {len(res)} items != the {len(want_ids)} draws the subset names at "
                    f"trial {idx} — a rung's population is its draws, whole; refusing")
    elif n_expect is not None:
        for idx, res in per_sample.items():
            # every attempted item reached `res`: a draw that errored refused
            # the whole task upstream (E56 (3)), so there is no second
            # population to add in here.
            if len(res) != n_expect:
                raise InspectServedError(
                    f"{task} sample {idx}: {len(res)} items != the {n_expect} the plan "
                    f"selected (full set {plan.expected_n}) — refusing a partial task "
                    "rather than filing an aggregate over a short set")
    # avg@k means are only means if every sample scored the SAME items; an
    # item present in one sample and absent from another would otherwise
    # get a mean over however many samples happened to include it, and the
    # row would look exactly like a k-sample one. A DRAW-subset rung is exempt
    # BY CONTRACT: its trials draw different item sets on purpose,
    # and it files no per-item mean row at all (below).
    item_sets = {idx: [r.item_id for r in res] for idx, res in per_sample.items()}
    if not draw_rung and len({tuple(v) for v in item_sets.values()}) > 1:
        sizes = {k: len(v) for k, v in item_sets.items()}
        raise InspectServedError(
            f"{task}: the avg@{plan.samples_per_item} samples scored different "
            f"item sets ({sizes}) — a per-item mean over a varying denominator "
            "is not avg@k; re-run to fill the gaps")

    return _file_task_results(plan, store=store, task=task, per_sample=per_sample, em=em, thinking=thinking,
                              ident=ident, sampling_ref=sampling_ref, reused=reused,
                              n_retried_requests=n_retried_requests, t0=t0, model=model,
                              route_virtual_model=route_virtual_model, reasoning_parser=reasoning_parser,
                              limits=limits, out_of_support=out_of_support, ledger_path=ledger_path,
                              attempt_scope=attempt_scope, generated=generated, generation=None)


def _live_generation_facts(api: Any, task: str, *, thinking: str, out_of_support: bool, reused: int,
                           prior: Callable[[], Optional[Mapping[str, Any]]], ledger_path: Optional[Path],
                           attempt_scope: Optional[C.AttemptScope], generated: Mapping[str, int]
                           ) -> Tuple[Dict[str, Any], Any, Any]:
    """The three generation-side facts read off the LIVE run — ONE derivation for the
    coupled filing tail and the generation-only exit (tinycap-shared-standup landing 2),
    which differ only in where a reuse-only pass's PRIOR regime fields come from (the
    task's existing summary / the existing sidecar; ``prior`` is called only then):

    * the regime fields (an absent stamp follows the additive-key rule — refused under
      ``out_of_support``, null in support — and a task whose every log was reused carries
      the fields stamped at the live run that wrote the logs, never re-derived from nothing).
    * the client record (the operational keys this task ran under — inspect_ai's
      retry count and the per-REQUEST client timeout the target's client was BUILT with —
      recorded UNHASHED on the aggregate row and the summary, read off the provider, null
      for a model with none, never off the constant).
    * the LEDGER's count of this task's rejected calls: ``n_rejected`` is the
      client's reading (a rejection the client's retry answered never reaches the results,
      so it reads 0 on every finished task); the ledger in front of the proxy saw every
      attempt, and a task that finished with rejected rows scored the client's SECOND draw
      of those items."""
    regimes = regime_summary_fields(
        api.regime_stamps.get(task, Counter()) if api is not None else None,
        expected=(C.expected_stamp_regimes(lens=pe.is_lens_route(api.route), arm=thinking,
                                           policy=getattr(api, "stamp_policy", None))
                  if api is not None else None),
        task=task, out_of_support=bool(out_of_support),
        prior=(prior() if api is not None and reused and not api.regime_stamps.get(task) else None))
    client = client_record(api)
    n_http_rejected_retried = _ledger_rejected_count(ledger_path, task, attempt_scope=attempt_scope,
                                                     generated=bool(generated["n"]))
    return regimes, client, n_http_rejected_retried


def _generation_exit(plan: TaskPlan, *, store: Any, task: str, gen_logs: Mapping[int, Tuple[Any, Path]],
                     generation: "GenerationPhase", model: Any, thinking: str,
                     route_virtual_model: Optional[str], target_usage_model: Optional[str],
                     reasoning_parser: Optional[str], limits: Optional[Mapping[str, int]], out_of_support: bool,
                     ledger_path: Optional[Path], sampling_ref: Optional[str],
                     attempt_scope: Optional[C.AttemptScope], ident: Mapping[str, Any], reused: int,
                     generated: Mapping[str, int], t0: float, max_model_len: Optional[int],
                     n_retried_requests: int = 0) -> Dict[str, Any]:
    """The GPU phase's end of a generation-only task: the UNSCORED pass log is
    checked (one trial, no score on any sample, its model key the target's), then
    the generation SIDECAR (``cpu_verify.GENERATION_SIDECAR_FIELDS`` — every
    generation-side fact the filing tail needs: the regime fields read off the LIVE
    provider now, the client record, the ledger's rejected count at generation end,
    the plan's facts, the target's ``model_usage`` key, the log's ids) and the
    PENDING MARKER (exactly ``contracts.PENDING_MARKER_FIELDS``) are written into
    the row dir. Returns the task's ``gen_stats`` block: status ``GENERATED`` (never
    an OK status), the counts, the three paths. Nothing is filed into the store.

    A re-run over a store whose unscored log is retained REUSES it (``reusable_log``
    admits an unscored ``success`` log at the plan's count, ids and cap) and
    rewrites the sidecar — the regime fields then carry over from the EXISTING
    sidecar exactly as the coupled tail carries them from an existing summary
    (``regime_summary_fields(prior=…)``): a reuse-only pass saw no live response."""
    if sorted(gen_logs) != [0]:
        raise InspectServedError(
            f"{task}: the generation-only leg produced logs for trials {sorted(gen_logs)}, not exactly [0] — the "
            "two-phase contract is the one-pass, k = 1 BigCodeBench leg")
    log, log_dir = gen_logs[0]
    location = getattr(log, "location", None)
    log_path = Path(str(location)) if location else None
    if log_path is None or not log_path.is_file() or log_path.resolve().parent != Path(log_dir).resolve():
        cands = _eval_logs_newest_first(log_dir)
        if not cands:
            raise InspectServedError(f"{task}: no .eval log under {log_dir} after the generation-only eval "
                                     f"(log.location={location!r}) — nothing for the CPU verifier to score")
        log_path = cands[0]
    log_path = log_path.resolve()
    sample_ids = log_sample_ids(log)
    scored = [str(getattr(s, "id", "?")) for s in list(getattr(log, "samples", None) or []) if getattr(s, "scores", None)]
    if scored:
        raise InspectServedError(
            f"{task}: the generation-only eval returned a log whose samples {scored[:5]} carry scores — "
            "`score=False` did not hold (or a scored log was reused as the unscored one); refusing to mark it "
            "pending verification")
    eval_model = str(getattr(getattr(log, "eval", None), "model", "") or "")
    if target_usage_model is not None and eval_model and eval_model != target_usage_model:
        raise InspectServedError(
            f"{task}: the unscored log's model key {eval_model!r} is not the target's {target_usage_model!r} "
            "(contracts.inspect_model_name(route_virtual_model)) — the verifier's `results_from_log` would read "
            "the wrong model_usage entry; refusing")
    target_model = eval_model or target_usage_model
    if not target_model:
        raise InspectServedError(f"{task}: neither the log nor the route names the target model_usage key")
    wall = time.time() - t0
    # the generated log's bytes, so the verifier can tell the original in place from an earlier verify
    # attempt's merged log (whose original the archive then holds) — and refuse anything else by name
    log_sha256 = _CV.sha256_file(log_path)
    # the existing sidecar is read unconditionally: a torn or foreign one refuses BEFORE it is overwritten
    prior = _CV.read_generation_sidecar_if_present(generation.row_dir, generation.cell, task, generation.store)
    marker_path = _CV.pending_marker_path(generation.row_dir, generation.cell, task, generation.store)
    now = time.time()
    if reused:
        # the door (the lost-verifier rule): a RETAINED unscored log is this task's evidence
        # only while its pending marker AND its sidecar stand and describe it — the generation happened once, by
        # the fcid and at the time the sidecar records, and that is what the marker keeps saying. With either
        # file gone, or a sidecar naming other bytes, the store is TORN: the harness parks it by design, and a
        # re-run that re-marked it here would heal in place what nothing verified. Refused by name, never
        # regenerated; the remedy is a deliberate regeneration (delete the retained log dir and re-run).
        if not marker_path.is_file() or prior is None:
            missing = ("pending marker " + str(marker_path)) if not marker_path.is_file() else (
                "generation sidecar " + str(_CV.generation_sidecar_path(generation.row_dir, generation.cell, task, generation.store)))
            raise InspectServedError(
                f"{task}: torn — the unscored log {log_path} is retained under the store but its {missing} is "
                "missing, so the task is neither awaiting its CPU verifier nor generated by this run; a re-run "
                "never re-marks a retained log (the harness parks a torn store — plan §2.2.3). Repair = a "
                "deliberate regeneration: delete the task's log dir and re-run")
        if Path(str(prior["log"])).resolve() != log_path or str(prior["log_sha256"]) != log_sha256:
            raise InspectServedError(
                f"{task}: torn — the generation sidecar names the log {prior['log']} (sha256 "
                f"{str(prior['log_sha256'])[:12]}), the retained unscored log is {log_path} (sha256 "
                f"{log_sha256[:12]}); the two are not one generation, so neither is this task's evidence")
        generated_at, generation_fcid = prior["generated_at"], prior["generation_fcid"]
    else:
        generated_at, generation_fcid = now, generation.generation_fcid
    regimes, client, n_http_rejected_retried = _live_generation_facts(
        provider_of(model), task, thinking=thinking, out_of_support=out_of_support, reused=reused,
        prior=lambda: (dict(prior["regimes"]) if prior is not None else None),
        ledger_path=ledger_path, attempt_scope=attempt_scope, generated=generated)
    n_items = len(sample_ids)
    sidecar = {
        "task": task, "cell": generation.cell, "store": generation.store, "log": str(log_path),
        "log_sha256": log_sha256, "n_items": n_items,
        "n_expected_full": plan.expected_n, "sample_ids": list(sample_ids),
        "pin_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "generated_at": generated_at, "generation_fcid": generation_fcid,
        # the coupled tail's request accounting (`_file_task_results`: the pass's items + the truncation retry's
        # requests), carried as generation-side facts so the verifier files the same field set AND value
        "n_requests": n_items + int(n_retried_requests), "n_retried_requests": int(n_retried_requests),
        "generated_calls": int(generated["n"]), "wall_clock_s": round(wall, 3),
        "logs_reused": int(reused), "regimes": dict(regimes), "client": dict(client),
        "n_http_rejected_retried": n_http_rejected_retried,
        "route_virtual_model": route_virtual_model, "target_model": target_model,
        "sampling_ref": sampling_ref, "thinking": thinking, "reasoning_parser": reasoning_parser,
        "limits": dict(limits or {}), "out_of_support": bool(out_of_support),
        "row_identity": dict(ident), "attempt_id": (attempt_scope.attempt_id if attempt_scope is not None else None),
        "concurrency": int(generation.concurrency), "max_model_len": max_model_len,
        "plan": plan_facts(plan), "producer_tree_sha": _store_producer_tree_sha(store.cell_dir),
        "inspect_ai_version": _CV.inspect_ai_version(),
    }
    marker = {k: sidecar[k] for k in C.PENDING_MARKER_FIELDS}      # the marker is the sidecar's projection
    if reused:
        # the standing marker must already BE that projection (it was written from the sidecar the facts above
        # were carried from); it is held, not rewritten — no new marker for a generation that did not happen
        standing = _CV.read_pending_marker(generation.row_dir, generation.cell, task, generation.store)
        if standing != marker:
            raise InspectServedError(
                f"{task}: torn — the standing pending marker {marker_path} reads {standing}, the retained log and its "
                f"sidecar say {marker}; refusing to rewrite a marker that disagrees with the generation it names")
        sc_path, mk_path = _CV.write_generation_sidecar(generation.row_dir, sidecar=sidecar), marker_path
    else:
        sc_path, mk_path = _CV.write_generation_pending(generation.row_dir, sidecar=sidecar, marker=marker)
    print(f"[inspect-served] {task}: GENERATED {n_items} items (unscored; {reused} log(s) reused) — pending its CPU "
          f"verifier: sidecar {sc_path}, marker {mk_path}{' (standing, held equal)' if reused else ''}; nothing filed "
          "(plan §2.2.3)", flush=True)
    return {"status": C.REC_STATUS_GENERATED, "verification_phase": C.VERIFICATION_PHASE_GENERATE, "task": task,
            "n": n_items, "n_items": n_items, "n_expected_full": plan.expected_n, "n_requests": sidecar["n_requests"],
            "wall_clock_s": round(wall, 3), "logs_reused": int(reused), "log": str(log_path),
            "sidecar": str(sc_path), "pending_marker": str(mk_path)}


def _file_task_results(plan: TaskPlan, *, store: Any, task: str, per_sample: Dict[int, List[ItemResult]],
                       em: Any, thinking: str, ident: Mapping[str, Any], sampling_ref: Optional[str],
                       reused: int, n_retried_requests: int, t0: Optional[float], model: Any,
                       route_virtual_model: Optional[str], reasoning_parser: Optional[str],
                       limits: Optional[Mapping[str, int]], out_of_support: bool,
                       ledger_path: Optional[Path], attempt_scope: Optional[C.AttemptScope],
                       generated: Mapping[str, int],
                       generation: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The ONE filing tail of an Inspect task (lifted out of
    :func:`run_one_task` verbatim): the per-sample and
    per-item rows, the two aggregate rows, the summary payload, the all-attempted
    outcome record and the DONE / PARTIAL marker through ``completion.finish_task``.
    ``per_sample`` is ``{trial: [ItemResult, ...]}`` off the task's scored log(s).

    ``generation`` (the CPU ``verify`` phase): the GPU phase's generation sidecar
    (``cpu_verify.GENERATION_SIDECAR_FIELDS``) — its ``wall_clock_s``, ``regimes``,
    ``client`` and ``n_http_rejected_retried`` stand in for what the coupled path
    reads off the live provider, the clock and the run's ledger, and the caller
    passes its ``logs_reused`` as ``reused``. ``None`` = the coupled path, which
    calls this with every value it computed itself — byte-identical to before the
    lift (the goldens in ``tests/test_inspect_served_verification_phase.py`` pin it)."""
    draw_rung = plan.draw_subset is not None
    by_item: Dict[str, List[float]] = {}
    n_requests = 0
    n_rows_emitted = 0
    by_item_results: Dict[str, List[ItemResult]] = {}
    for idx, res in sorted(per_sample.items()):
        for r in res:
            # every item is a KEY (it was attempted), but only a SCORED
            # result contributes a value: inspect files NaN when a scorer
            # returned no verdict, and a mean that swallowed it would report
            # the grader's silence as a number (§PW-nanfix-1)
            by_item.setdefault(r.item_id, [])
            if not r.unscored:
                by_item[r.item_id].append(r.value)
            by_item_results.setdefault(r.item_id, []).append(r)
            n_requests += 1
            if plan.samples_per_item > 1 and not r.unscored:
                em.emit(task=task, item_id=r.item_id,
                        metric=C.sample_metric(C.METRIC_ACC, idx),
                        value=r.value, tier=C.ROW_TIER,
                        extra=r.extra(thinking=thinking,
                                      max_tokens=plan.max_tokens, **_ref_extra(sampling_ref), **ident))
                n_rows_emitted += 1
            elif plan.samples_per_item > 1:
                # qwen-seedrep-native-battery F-AU: an UNSCORED sample files its TYPED OUTCOME row
                # (`unscored@<i>`, value 1.0 = an indicator, never a score; the type in `extra.outcome`) so
                # every attempted (item, sample) is filed exactly once -- §PW-nanfix-1 still files no acc@<i>
                em.emit(task=task, item_id=r.item_id,
                        metric=C.sample_metric(C.METRIC_UNSCORED, idx),
                        value=1.0, tier=C.ROW_TIER,
                        extra=r.extra(thinking=thinking, max_tokens=plan.max_tokens,
                                      **{C.OUTCOME_TYPE_FIELD: C.outcome_type(scored=False, at_cap=r.at_cap, unparseable=r.unparseable)},
                                      **_ref_extra(sampling_ref), **ident))
                n_rows_emitted += 1
    # every retry was a real request on top of the pass's
    n_requests += n_retried_requests
    # the paired, per-ITEM metric: the bare acc, one row per item — SKIPPED on
    # a draw-subset rung: a mean over ONE of an item's trials is not
    # avg@k; the rung store's content is the per-sample rows, the outcome record
    # and the summary, and the analysis merges per (item, trial) with the source
    first = per_sample.get(0, []) if not draw_rung else []
    seeds_by_item = {r.item_id: [seed_for_task(task, r.item_id, k)
                                 for k in range(plan.samples_per_item)]
                     for r in first}
    for order, r0 in enumerate(first):
        vals = by_item[r0.item_id]
        rs = by_item_results[r0.item_id]
        if not vals:
            # no sample of this item was scored — there is no acc to file.
            # The item still counts in `n` (it was attempted) and in
            # `n_unscored`; a 0.0 row here would be the grader's silence
            # written down as a refusal (or a wrong answer). This is also the
            # ONLY shape in which EVERY sample is unscored, which is what makes
            # `unscored: false` on an emitted mean row a derived fact rather
            # than an inherited one (E54 #4).
            #
            # qwen-seedrep-native-battery F-AU: the item is still FILED —
            # as a TYPED OUTCOME row (`unscored`, value 1.0 = an indicator, never
            # a score) whose `extra.outcome` names why (at the cap / unparseable /
            # no verdict), so the store files exactly one row per attempted item
            # and the done marker's `n_items` counts what is filed. Before it the
            # gsm8k smoke filed 196 acc rows for 200 attempted items and the
            # audit read the outcomes record's 4 at-cap draws as undeclared.
            r_u = rs[0]
            em.emit(task=task, item_id=r0.item_id, metric=C.METRIC_UNSCORED, value=1.0, tier=C.ROW_TIER,
                    position_order=order,
                    extra=r_u.extra(thinking=thinking, max_tokens=plan.max_tokens, unscored=True,
                                    **{C.OUTCOME_TYPE_FIELD: C.outcome_type(scored=False, at_cap=r_u.at_cap, unparseable=r_u.unparseable)},
                                    **({"samples_per_item": plan.samples_per_item, "n_unscored_samples": len(rs),
                                        "per_sample": [x.sample_provenance() for x in rs]} if plan.samples_per_item > 1 else {}),
                                    **_ref_extra(sampling_ref), **ident))
            n_rows_emitted += 1
            continue
        # E54 #4: a mean row's metadata is the AGGREGATE's, never sample
        # ZERO's. Its singular producer facts (seed, answer, token counts,
        # stop reason) come from the first SCORED sample — the first draw that
        # actually contributed to the value — and `provenance_sample_index`
        # names which one, so a k=2 item whose sample 0 was unscored no longer
        # files sample 0's answer and `unscored: true` beside a value that
        # came from sample 1. Every other draw stays recoverable in
        # `per_sample`, including the unscored ones, which file no `acc@<i>`
        # row of their own.
        prov = next(x for x in rs if not x.unscored)
        item_unscored = sum(1 for x in rs if x.unscored)
        more: Dict[str, Any] = {
            "thinking": thinking, "max_tokens": plan.max_tokens,
            # DERIVED, not inherited: true only if every sample was unscored,
            # which files no row at all — so this reads false on every emitted
            # mean row, by construction rather than by luck.
            "unscored": all(x.unscored for x in rs),
            # the sampling block's digest, on every row of a run that has one
            **_ref_extra(sampling_ref),
            # ifeval-canonical-battery P42: the object identity, on every row of a --bank-view run
            **ident}
        if plan.samples_per_item > 1:
            more.update(samples_per_item=plan.samples_per_item,
                        seeds=seeds_by_item[r0.item_id], values=vals,
                        # SAMPLES, not items — the aggregate row and the
                        # summary spell the item count `n_unscored`, and one
                        # name for two denominators is how the two got
                        # confused in the first place
                        n_unscored_samples=item_unscored,
                        provenance_sample_index=prov.sample_idx,
                        per_sample=[x.sample_provenance() for x in rs])
        if plan.retry_max_tokens:
            # the mean is over the item's k FINAL attempts, which may be a
            # mix; the row says how many were retried and reports the LARGEST
            # attempt / cap that contributed, so `attempt == 2` on a mean row
            # reads as "at least one sample of this item was retried"
            more.update(n_retried=sum(1 for x in rs if x.attempt > 1),
                        attempt=max(x.attempt for x in rs),
                        max_tokens_used=max((x.max_tokens_used for x in rs
                                             if x.max_tokens_used is not None),
                                            default=None))
        em.emit(task=task, item_id=r0.item_id, metric=C.METRIC_ACC,
                value=sum(vals) / len(vals), tier=C.ROW_TIER,
                position_order=order, extra=prov.extra(**more))
        n_rows_emitted += 1

    n = len(by_item)
    # landing 2: under a generation sidecar the wall clock is the GPU phase's (the
    # verifier's own time is on the verified block), else this run's
    wall = (float(generation["wall_clock_s"]) if generation is not None else time.time() - t0)
    # §PW-nanfix-1: `acc` is the mean over the items the scorer RETURNED A
    # VERDICT FOR, and `n_scored` is that denominator — never `n`, which
    # counts the items ATTEMPTED. The two differ exactly when a scorer filed
    # inspect's unscored sentinel (a model grader whose reply carried no
    # parseable GRADE), and shipping both is what keeps "the grader said
    # nothing" from reading as a score.
    scored_items = {k: v for k, v in by_item.items() if v}
    n_scored = len(scored_items)
    # TWO denominators, and the tier files no task with a third: an item that
    # produced no generation at all refused this task upstream (E56 (3)), so
    # n == n_scored + n_unscored by construction here.
    n_unscored = n - n_scored
    # E54 #4: ITEMS and SAMPLES are two denominators, and at k > 1 both are
    # meaningful — an item counts as scored when ANY of its k draws was, so
    # `n_unscored` (items) can read 0 while draws were lost. `*_samples`
    # counts the (item, sample) draws. The two pairs coincide at k = 1.
    every_result = [r for res in per_sample.values() for r in res]
    n_unscored_samples = sum(1 for r in every_result if r.unscored)
    n_scored_samples = len(every_result) - n_unscored_samples
    if n and not n_scored:
        raise InspectServedError(
            f"{task}: no item of the {n} attempted has a score ({n_unscored} unscored) — "
            "there is no accuracy to file. Unscored means the scorer returned inspect's "
            "sentinel on a real generation: on a model-graded task the grader route never "
            "produced a parseable GRADE, so fix the grader (or its token cap). Not a mean "
            "over nothing")
    acc = (sum(sum(v) / len(v) for v in scored_items.values()) / n_scored
           if n_scored else 0.0)
    if n_unscored or n_unscored_samples:
        print(f"[inspect-served] {task}: {n_unscored} of {n} items UNSCORED "
              f"({n_unscored_samples} of {len(every_result)} draws; the "
              "scorer returned no verdict; on a model-graded task, no parseable "
              f"GRADE) — acc is the mean over the other {n_scored}, and the counts "
              "travel with it", flush=True)
    # flagged rows across every sample: reasoning present, content empty
    n_flagged = sum(1 for res in per_sample.values() for r in res
                    if r.content_empty_with_reasoning)
    # U5-5: the two quality numbers, over the FINAL attempt of every
    # (item, sample) — the results that were scored. Each rate carries its
    # count and its own denominator, so a per-cell roll-up pools exactly and
    # `null` reads as "the producer does not record it", never as zero.
    quality = task_quality(every_result)
    # Fit-support transfer (plan S2.6): the think-block counts and
    # the token statistics, over the SAME attempted population `n` and the
    # quality rates count — never over the emitted rows, which omit unscored
    # items. `mean` / `p99` are None (never 0) when no draw recorded usage.
    split_counts = think_split_counts(every_result, plan=plan)
    # a SANDBOXED task's sandbox-status and code-source counts over the same
    # attempted population; absent on every other task's payload
    # (contracts.BCB_SUMMARY_COUNT_FIELDS)
    sandbox_counts = bcb_summary_counts(every_result) if plan.sandbox is not None else {}
    ifeval_fields = ifeval_summary_fields(plan, every_result)
    # the regime stamp triple
    # this leg's LIVE responses carried — one value of each field, its
    # distinct count 1, equal to the leg's contract row — read off the
    # provider's accumulator (`on_response`); a model without one (a stub)
    # records null, a leg whose stamps are mixed or off-row refuses here;
    # an absent stamp follows the additive-key rule —
    # refused under `out_of_support`, null in support — and a task whose every
    # log was reused carries the fields from its existing summary (stamped at
    # the live run that wrote the logs), never re-derived from nothing.
    # Under a generation sidecar the regime fields, the
    # client record and the ledger's rejected count were read at GENERATION end,
    # where the live provider and the run's ledger were — the CPU verifier has
    # neither; the coupled path reads them off the live run here
    if generation is not None:
        regimes, client = dict(generation["regimes"]), dict(generation["client"])
        n_http_rejected_retried = generation["n_http_rejected_retried"]
    else:
        regimes, client, n_http_rejected_retried = _live_generation_facts(
            provider_of(model), task, thinking=thinking, out_of_support=out_of_support, reused=reused,
            prior=lambda: completion.read_task_summary(store.cell_dir, task),
            ledger_path=ledger_path, attempt_scope=attempt_scope, generated=generated)
    em.emit(task=task, item_id=C.AGG_ITEM_ID, metric=C.METRIC_ACC_AGGREGATE,
            value=acc, tier=C.ROW_TIER, wall_clock_s=wall,
            extra={"n": n, "thinking": C.enable_thinking(thinking),
                   "render": C.THINKING_RENDER_NATIVE,
                   "samples_per_item": plan.samples_per_item,
                   "temperature": plan.temperature, "top_p": plan.top_p, "top_k": plan.top_k,
                   **_optional_terms(plan), **_protocol_term(plan),
                   "reasoning_parser": reasoning_parser,
                   "n_content_empty_with_reasoning": n_flagged,
                   "item_set": plan.item_set,
                   "n_scored": n_scored, "n_unscored": n_unscored,
                   "n_scored_samples": n_scored_samples,
                   "n_unscored_samples": n_unscored_samples,
                   **quality, **split_counts, **sandbox_counts,
                   "n_retried": sum(1 for r in every_result if r.attempt > 1),
                   "retry_max_tokens": plan.retry_max_tokens or None,
                   # E56 (3): the GRADER's cap, None for a task with no model
                   # grader — a refusal rate names the instrument that produced it
                   "grader_max_tokens": plan.grader_max_tokens,
                   **client,
                   # ifeval-canonical-battery: the task version, the langdetect
                   # seed and the per-trial IFEval means (IFEval payloads only)
                   **ifeval_fields,
                   # P42: the object identity on the aggregate rows too
                   **ident,
                   "note": plan.note})
    em.emit(task=task, item_id=C.AGG_ITEM_ID, metric=C.METRIC_N,
            value=float(n), tier=C.ROW_TIER, wall_clock_s=wall, **({"extra": dict(ident)} if ident else {}))
    # controls_summary payloads are counts / fracs / SHORT whitespace-free
    # strings only (`cell_controls._assert_counts_only`), so the plan's
    # prose note rides in the aggregate row's extra_json above, never here.
    summary = {C.METRIC_ACC: acc, C.METRIC_N: n, "n_requests": n_requests,
               "n_rejected": 0, "n_http_rejected_retried": n_http_rejected_retried,
               "samples_per_item": plan.samples_per_item,
               # the payload names its instrument (PW3): the row task and,
               # for a task with several item sets, which one ran
               "task": task, "item_set": plan.item_set,
               "framework": C.INSPECT_TASK_VERSION, "engine": "vllm",
               "route": route_virtual_model,
               "thinking": thinking, "render": C.THINKING_RENDER_NATIVE,
               "wall_clock_s": round(wall, 3), "logs_reused": reused,
               "n_paired": sum(1 for r in first if r.paired_id),
               "reasoning_parser": reasoning_parser,
               "n_content_empty_with_reasoning": n_flagged,
               # §PW-nanfix-1: acc's OWN denominator. `n` is the items
               # attempted; `n_scored` is the items the scorer gave a verdict
               # for, and acc is the mean over those. They differ only when a
               # scorer filed inspect's unscored sentinel.
               "n_scored": n_scored, "n_unscored": n_unscored,
               # E54 #4: the same two facts over (item, sample) DRAWS. An item
               # is scored when ANY of its k draws was, so at k > 1 the item
               # pair cannot express how many draws the grader dropped; at
               # k = 1 the two pairs are equal.
               "n_scored_samples": n_scored_samples,
               "n_unscored_samples": n_unscored_samples,
               # U5-5: the unparseable rate is its own number, with the count
               # and the denominator beside it (contracts.SUMMARY_PAYLOAD_FIELDS)
               **quality,
               # fit-support transfer (plan S2.6): the think-block counts, the
               # token statistics, the split mode and the unterminated policy
               **split_counts,
               # bcb-bfcl-200: the sandbox-status / code-source counts of a
               # sandboxed task (absent on every other payload)
               **sandbox_counts,
               # the leg's regime stamp
               # triple (render / salt / resumed policy) and the distinct count
               # of each over its responses (must be 1)
               **regimes,
               # the pinned generation knobs this run sent: the cap
               # (None = the task pins its own inside its solver) and the
               # sampling row's three numbers (U8; top_p / top_k None = no key)
               "max_tokens": plan.max_tokens, "temperature": plan.temperature,
               "top_p": plan.top_p, "top_k": plan.top_k,
               # the OPTIONAL terms and the sampling digest, only when set;
               # bcb-bfcl-200-widen: the DECLARED protocol temperature beside the executed
               # one, only under `protocol` (every other summary is byte-identical)
               **_optional_terms(plan), **_protocol_term(plan), **_ref_extra(sampling_ref),
               # E56 (3): the GRADER's completion cap on a model-graded task
               # (None on every other). An uncapped grader and a capped one are
               # two instruments producing one number, so the payload says
               # which; a payload written before the cap carries no key at all,
               # which is how a merge tells the two apart.
               "grader_max_tokens": plan.grader_max_tokens,
               # the operational keys the task ran under (`max_retries`,
               # `client_timeout_s`), unhashed; a payload written before the
               # client timeout was passed carries no key — those stores ran
               # under the SDK's 600 s default, which is how a reader tells the
               # censored population from the fixed one
               **client,
               # the stock task's own
               # version, the langdetect seed and the per-trial means of the four
               # IFEval metrics — IFEval payloads only, every other byte-identical
               **ifeval_fields}
    # the ALL-ATTEMPTED outcome record (plan S2.7): one record per (item,
    # sample) FINAL attempt, built from the SAME list `task_quality` and the
    # counts above were computed from, so header and summary agree by
    # construction (`finish_task` verifies before it writes anything)
    outcomes = outcome_record(every_result, plan=plan, task=task, thinking=thinking,
                              summary=summary, split_counts=split_counts,
                              producer_tree_sha=_store_producer_tree_sha(store.cell_dir),
                              sampling_ref=sampling_ref)
    # done iff unlimited AND n == the frozen full count; else partial (R2)
    fin = completion.finish_task(
        store.cell_dir, task=task,
        # `n` counts the items ATTEMPTED, and every attempted item was
        # measured: an UNSCORED one counts toward completion on purpose
        # (a greedy grader's silence reproduces exactly, so re-running buys
        # nothing), and an ERRORED one never gets here at all (the task is
        # refused). So a hole can never be sealed under a done marker by way
        # of this count. Every attempted item also FILES exactly one row
        # (acc, or the typed `unscored` row), so `n_items` is the filed item
        # count as well.
        n_items=n,
        n_expected_full=plan.expected_n, limits=limits,
        # the per-item / per-sample rows actually FILED (since F-AU an unscored
        # result files its typed `unscored` row), not the shape the plan would
        # have produced. Counted the way the old formula counted: the two
        # aggregate rows are not in it
        n_rows=n_rows_emitted,
        # F-AU: the plan's count on the done marker beside the filed one
        n_planned=plan.expected_n,
        n_requests=n_requests, n_rejected=0, wall_clock_s=round(wall, 3), summary=summary,
        # written BETWEEN the summary and the marker (S2.7)
        outcomes=outcomes,
        # completion is a DRAW fact too on
        # the k = 2 instrument — the run's own draw count against the expected one
        # (items x trials at pass 1, the subset's n_draws on a rung); both ride the
        # marker (contracts.DONE_MARKER_DRAW_FIELDS). Every other task passes None
        # and writes a byte-identical marker
        n_draws=len(every_result), n_expected_draws=_expected_draws(plan))
    if not fin["complete"]:
        print(f"[inspect-served] {task}: PARTIAL — {n} measured of "
              f"{plan.expected_n} items under limits {dict(limits or {})}; "
              "no done marker written", flush=True)
    return summary


def _ledger_rejected_count(ledger_path: Optional[Path], task: str, *,
                           attempt_scope: Optional[C.AttemptScope] = None,
                           generated: bool = True) -> Optional[int]:
    """The usage ledger's count of ``task``'s REJECTED calls
    (``usage_ledger.rejected_rows_for_task`` — rows naming the task with
    ``rejected`` true; every row is fsync'd before its response returns, so
    the count is complete at task end). ``None`` when the run has no ledger
    (a hand run without ``--ledger-url``): the summary then says the producer
    recorded nothing, never 0. A ledger path GIVEN but absent at task end
    refuses: the ledger process wrote no row (or wrote elsewhere), and a
    summary claiming a count it cannot read would be the very disagreement
    this field exists to end.

    ``attempt_scope``: the count
    over ONE attempt's rows — this attempt's when ``generated``, else the
    recorded prior generation attempt's (a reused ``.eval`` is that attempt's
    evidence; none recorded refuses) — rows attributed by their own
    ``attempt_id`` column or their request-evidence record
    (``contracts.ledger_row_attempt``). ``None`` = the whole ledger, as before."""
    if ledger_path is None:
        return None
    path = Path(ledger_path)
    if not path.is_file():
        raise InspectServedError(
            f"{task}: the run's usage ledger {path} does not exist at task end — the ledger "
            "process never wrote a row (or wrote elsewhere), so the task's rejected-call count "
            "cannot be recorded; refusing to file a summary that would claim it (D73 (d))")
    if attempt_scope is None:
        return int(ul.rejected_rows_for_task(path, task))
    if generated:
        wanted = {attempt_scope.attempt_id}
    else:
        if attempt_scope.prior_generation_attempt is None:
            raise InspectServedError(
                f"{task}: the retained .eval was REUSED (attempt {attempt_scope.attempt_id!r} sent no task "
                "call — a finalisation) but no prior attempt of this store is recorded as having generated, so "
                "the ledger-derived summary field cannot be scoped to the attempt that produced the log "
                "(plan §4.5); refusing")
        wanted = {attempt_scope.prior_generation_attempt}
    rows = _jsonl_records(path)
    records = _jsonl_records(Path(attempt_scope.evidence_path)) if attempt_scope.evidence_path else []
    try:
        return int(C.scoped_rejected_count(rows, task, wanted, evidence_by_id=C.evidence_by_request_id(records)))
    except C.AttemptScopeError as e:
        raise InspectServedError(f"{task}: {e}") from None


def _jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Every parsed row of the usage ledger or the request-evidence file
    (``usage_ledger.read_rows``: blank and torn trailing lines skipped);
    ``[]`` when the file is absent."""
    p = Path(path)
    return ul.read_rows(p) if p.is_file() else []


def bcb_summary_counts(results: Sequence[ItemResult]) -> Dict[str, int]:
    """The BigCodeBench summary's count fields (``contracts.BCB_SUMMARY_COUNT_FIELDS``)
    over the FINAL attempt of every attempted (item, sample): how many verdicts
    the verifier reached (``n_sandbox_ok``), how many timed out
    (``n_verify_timeout``), and where the scored code came from."""
    out = {f: 0 for f in C.BCB_SUMMARY_COUNT_FIELDS}
    for r in results:
        status = r.row_facts.get("sandbox_status")
        if status == "ok":
            out["n_sandbox_ok"] += 1
        elif status == "verify_timeout":
            out["n_verify_timeout"] += 1
        src = r.row_facts.get("code_source")
        if src in C.BCB_CODE_SOURCES:
            out[f"n_code_source_{src}"] += 1
    return out


SANDBOX_FAILURES_DIR = "diagnostics"
SANDBOX_FAILURES_FILENAME = "sandbox_failures.json"


def write_sandbox_failures(store: Any, task: str, log: Any, *, sample_idx: int = 0) -> Path:
    """``<cell_dir>/diagnostics/sandbox_failures.json`` (bcb-bfcl-200): the
    errored items of a SANDBOXED task's refused log — item, draw, HTTP
    status, retries, the message head — and the cancelled count, appended per
    (task, sample, attempt) so a persistent creation failure is diagnosable
    after the container is gone. Diagnostics only: never a row, never a marker."""
    errs = sample_errors(log, sample_idx=sample_idx)
    rec = {"task": task, "sample_idx": int(sample_idx), "at": time.time(),
           "log_status": getattr(log, "status", None),
           "n_samples": len(list(getattr(log, "samples", None) or [])),
           "n_cancelled": cancelled_sample_count(log),
           "errors": [{"item_id": e.item_id, "sample_idx": e.sample_idx, "status": e.status,
                       "retries": e.retries, "message": e.message[:ERROR_MESSAGE_CHARS]}
                      for e in errs]}
    path = Path(store.cell_dir) / SANDBOX_FAILURES_DIR / SANDBOX_FAILURES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    prior: List[Dict[str, Any]] = []
    if path.is_file():
        try:
            prior = list(json.loads(path.read_text()).get("attempts") or [])
        except ValueError:
            prior = []
    path.write_text(json.dumps({"attempts": [*prior, rec]}, indent=1, sort_keys=True, default=str))
    return path


def attempted_counts(results: Sequence[ItemResult]) -> Dict[str, int]:
    """The five population counts of a task over its FINAL-attempt results —
    ``n`` (items attempted), ``n_scored`` / ``n_unscored`` (an item is scored
    when ANY of its draws was), ``n_scored_samples`` / ``n_unscored_samples``
    (draws) — the same definitions :func:`run_one_task` files in the summary
    (§PW-nanfix-1, E54 #4), re-derived from a log so a reconstructed record
    can be VERIFIED against a stored summary rather than copied from it."""
    scored_by_item: Dict[str, bool] = {}
    for r in results:
        scored_by_item[r.item_id] = scored_by_item.get(r.item_id, False) or not r.unscored
    n = len(scored_by_item)
    n_scored = sum(1 for v in scored_by_item.values() if v)
    n_unscored_samples = sum(1 for r in results if r.unscored)
    return {"n": n, "n_scored": n_scored, "n_unscored": n - n_scored,
            "n_scored_samples": len(results) - n_unscored_samples,
            "n_unscored_samples": n_unscored_samples}


def _rows_disagree_with_results(store: Any, task: str, results: Sequence[ItemResult]
                                ) -> Optional[str]:
    """Why the store's OWN per-item ``acc`` rows of ``task`` are not the rows
    ``results`` would have filed — ``None`` when they are, or when the store
    exposes no parsed rows (a stub). The rows a resumed regeneration would
    have left in place are exactly what this compares: a row set or value
    that differs from the retained log is a store whose rows and evidence
    disagree."""
    parsed = getattr(store, "_parsed_rows", None)
    if parsed is None:
        return None
    filed: Dict[str, float] = {}
    for r in parsed():
        if r.get("task") != task or r.get("metric") != C.METRIC_ACC or r.get("item_id") == C.AGG_ITEM_ID:
            continue
        filed[str(r["item_id"])] = float(r["value"])
    vals: Dict[str, List[float]] = {}
    for r in results:
        if not r.unscored:
            vals.setdefault(r.item_id, []).append(float(r.value))
    want = {i: sum(v) / len(v) for i, v in vals.items()}
    only_rows = sorted(set(filed) - set(want))
    only_log = sorted(set(want) - set(filed))
    differ = sorted(i for i in set(filed) & set(want) if filed[i] != want[i])
    problems = []
    if only_rows:
        problems.append(f"{len(only_rows)} rows for items the log leaves unscored or never "
                        f"attempted (first {only_rows[:3]})")
    if only_log:
        problems.append(f"{len(only_log)} scored items of the log with no row (first {only_log[:3]})")
    if differ:
        problems.append(f"{len(differ)} items whose row value differs from the log's "
                        f"(first {differ[:3]})")
    return "; ".join(problems) or None


def repair_task_outcomes(plan: TaskPlan, *, store: Any, thinking: Optional[str] = None,
                         route_virtual_model: Optional[str] = None) -> Dict[str, Any]:
    """REBUILD a DONE task's ``outcomes/<task>.json`` from its retained
    ``.eval`` logs — no new draw — or refuse by name (plan S2.8 (ii), as
    corrected after review).

    A done marker whose outcome record is missing or invalid used to send the
    task back through :func:`run_one_task`. That path regenerates when the log
    is gone, and its rows go through the store's ``(task, item_id, metric)``
    dedup (``tier0._Emitter``): every previously filed key is SUPPRESSED, so a
    regenerated draw leaves the OLD rows beneath a NEW summary / record /
    marker — a store whose rows and summary come from two different
    generations, which the verifier then accepts. So a repair is a
    RECONSTRUCTION: every sample's completed log must still be under
    ``private/inspect_logs/`` (:func:`reusable_log`; the MATH-500 retry's
    ``__a2`` log too when the pass stopped anything at the cap) and must carry
    EXACTLY the plan's selection as an id multiset (:func:`id_multiset_mismatch`
    — a same-sized log over another selection reproduces every count and every
    row and would otherwise be certified; the retry log must equal the ids the
    pass stopped at the cap), the record is
    rebuilt through the same ``results_from_log`` -> :func:`outcome_record`
    path the runner uses with its counts RE-DERIVED from the log
    (:func:`attempted_counts`, :func:`task_quality`,
    :func:`think_split_counts`) — never copied from the summary — and it is
    written only if it VERIFIES against the stored summary
    (``completion.verify_outcomes_against_summary``) and the store's own rows
    are the log's scored values. Anything else refuses in place, naming the
    cause and the two ways out: a fresh store (a new out-suffix) or an explicit
    operator repair. The summary and the marker are left exactly as they are;
    nothing is emitted. Returns the stored summary plus a ``repaired_outcomes``
    block."""
    arm = plan.thinking if thinking is None else C.check_thinking_arg(thinking)
    if arm == C.THINKING_PER_TASK:
        raise InspectServedError(
            f"{plan.row_task}: repair_task_outcomes was given the {C.THINKING_PER_TASK!r} MODE as "
            "an arm — a task runs under one arm")
    task = plan.row_task
    cell_dir = Path(store.cell_dir)
    ways_out = ("a fresh store (a new out-suffix) or an explicit operator repair; never a re-run "
                "into this store, whose skip-on-key row emission would leave the old rows under a "
                "new summary")
    if not completion.done_marker_path(cell_dir, task).is_file():
        raise InspectServedError(
            f"{task}: no done marker under {cell_dir} — nothing to repair (a task without one runs)")
    why = completion.task_outcomes_reason(cell_dir, task)
    if why is None:
        raise InspectServedError(
            f"{task}: the outcome record under {cell_dir} is valid — nothing to repair")
    summary = completion.read_task_summary(cell_dir, task)
    if summary is None:
        raise InspectServedError(
            f"{task}: a done marker with no {C.SUMMARY_DIR}/{C.marker_filename(task)} under "
            f"{cell_dir} — a torn store with nothing to verify a rebuilt record against; {ways_out}")
    n_expect = plan.n_expected_run
    target_usage_model = (C.inspect_model_name(route_virtual_model)
                          if route_virtual_model else None)
    per_sample: Dict[int, List[ItemResult]] = {}
    logs_read = 0
    for idx in range(plan.samples_per_item):
        d = _log_dir(store, task, idx)
        # held to this plan's cap too — a log generated at another cap is
        # not the evidence a record labelled max_tokens_used = plan.max_tokens
        # may be rebuilt from
        log = reusable_log(d, expected_n=n_expect, expected_max_tokens=plan.max_tokens)
        if log is None:
            raise InspectServedError(
                f"{task}: the done marker under {cell_dir} has no valid outcome record ({why}) and "
                f"no completed .eval log for sample {idx} survives under {d} (expected "
                f"{n_expect} samples at this plan's cap {plan.max_tokens}) — the record cannot be "
                f"rebuilt without a new draw; refusing: {ways_out}")
        logs_read += 1
        # the retained pass log's id MULTISET must be the plan's selection before
        # anything is derived from it: a same-sized log over a different selection
        # (the unscored ids substituted, the scored ones kept) reproduces every
        # count and every row and would have been certified — refused by name
        _refuse_log_id_mismatch(
            task, idx, log, plan.sample_ids, what="pass",
            extra=(f"; the retained log under {d} is not this task's evidence, so no record is "
                   f"written; {ways_out}"))
        res = results_from_log(
            log, sample_idx=idx, primary_scorer=plan.primary_scorer, paired_ids=plan.paired_ids,
            attempt=1, max_tokens_used=plan.max_tokens, target_model=target_usage_model,
            **_plan_readers(plan))
        if plan.retry_max_tokens:
            ids = _retry_ids(res)
            if ids:
                rd = _log_dir(store, task, idx, attempt=2)
                rlog = reusable_log(rd, expected_n=len(ids), expected_max_tokens=plan.retry_max_tokens)
                if rlog is None:
                    raise InspectServedError(
                        f"{task}: sample {idx} stopped {len(ids)} items at the cap and the retry's "
                        f"log under {rd} does not survive — the FINAL attempts cannot be rebuilt "
                        f"without a new draw; refusing: {ways_out}")
                logs_read += 1
                # the retry log's id multiset must EQUAL the ids the pass stopped at the
                # cap — a missing one hidden behind a duplicate is not this task's retry
                _refuse_log_id_mismatch(
                    task, idx, rlog, ids, what="retry",
                    extra=f"; the retained retry log under {rd} is not this task's; {ways_out}")
                retried = {r.item_id: r for r in results_from_log(
                    rlog, sample_idx=idx, primary_scorer=plan.primary_scorer,
                    paired_ids=plan.paired_ids, attempt=2, max_tokens_used=plan.retry_max_tokens,
                    target_model=target_usage_model,
                    **_plan_readers(plan))}
                res = [retried.get(r.item_id, r) for r in res]
        per_sample[idx] = res
    every_result = [r for res in per_sample.values() for r in res]
    # the counts come from the LOG (the same definitions run_one_task files),
    # so verify_outcomes_against_summary is a real comparison — a record whose
    # counts were copied from the summary would verify against anything
    derived = {**attempted_counts(every_result), **task_quality(every_result)}
    split_counts = think_split_counts(every_result, plan=plan)
    outcomes = outcome_record(every_result, plan=plan, task=task, thinking=arm, summary=derived,
                              split_counts=split_counts,
                              producer_tree_sha=_store_producer_tree_sha(cell_dir))
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)
    except completion.CompletionError as e:
        raise InspectServedError(
            f"{task}: the retained .eval log(s) under {cell_dir} do not reproduce the stored "
            f"summary — {e}; the store's evidence and its summary disagree, so no record is "
            f"written; {ways_out}") from e
    rows_problem = _rows_disagree_with_results(store, task, every_result)
    if rows_problem:
        raise InspectServedError(
            f"{task}: the store's rows.jsonl is not what the retained .eval log(s) would have "
            f"filed — {rows_problem}; a record rebuilt from that log would sit over rows from "
            f"another generation, so none is written; {ways_out}")
    completion.write_task_outcomes(cell_dir, task, outcomes)
    print(f"[inspect-served] {task}: outcome record REBUILT from {logs_read} retained .eval "
          f"log(s) under {cell_dir} (no new draw); it was invalid because: {why}", flush=True)
    return {**summary, "repaired_outcomes": {"logs_read": logs_read, "was_invalid_because": why,
                                             "n_attempted_items": outcomes["n_attempted_items"]}}


def write_done_marker(store: Any, task: str, payload: Mapping[str, Any]) -> Path:
    """``<cell_dir>/done/<task>.json`` (P34) — written only on a COMPLETE
    item set, so its presence is what resume trusts. The shared writer
    (``tinycap.completion``) asserts the contract's exact field set."""
    return completion.write_done_marker(store.cell_dir, task, payload)


def request_evidence_sinks(request_evidence_sink: Optional[Callable[..., Any]],
                           response_evidence_sink: Optional[Callable[..., Any]]
                           ) -> Tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]:
    """The ``(request, response)`` evidence sinks the TARGET provider records through.

    OFF (no ``SISTEER_REQUEST_EVIDENCE_DIR``): the caller's two sinks, untouched — production's
    ``_RequestEvidence.record_wire`` (``scripts/served_capability_run.py``) keeps writing exactly what it wrote
    (byte-identical records; no response record, no row binding).

    ON: BOTH sides are the lane client's (``request_evidence_client.from_env()``) — ``record_request`` (the v3 record
    + ``extra.s``, the seed-base discriminator ``audit_root`` requires of every task record on the Arditi
    engine, + production's top-level ``attempt_id`` stamp) and ``response_sink`` — WHATEVER request sink the caller
    passed: production's writer never writes ``extra.s``, so routing only the response side through the lane left
    every Arditi GSM8K request record unattributable to its seed base. The caller's request sink is NOT called for
    the provider's requests (one attempt, one record — no double-recording); it must append to the SAME
    ``request_evidence.jsonl`` and run as the same attempt (``RequestEvidenceClient.take_over_request_sink``
    refuses otherwise, by name). A caller-given response sink is kept."""
    client = rev.from_env()
    if client is None:
        return request_evidence_sink, response_evidence_sink
    try:
        request_sink = client.take_over_request_sink(request_evidence_sink)
    except rev.RequestEvidenceError as e:
        raise InspectServedError(f"request evidence: {e}") from None
    return request_sink, (response_evidence_sink if response_evidence_sink is not None else client.response_sink)


async def run_inspect_tasks(*, atoms: Sequence[str], route: Any,
                            routes: Sequence[Any], system: Optional[str],
                            store: Any, template: Mapping[str, Any],
                            thinking: str, base_url: str,
                            api_key: Optional[str] = None,
                            concurrency: int = 24,
                            samples: int = C.MATH500_SAMPLES_PER_ITEM,
                            inspect_limit: int = 0,
                            repo_root: Path = REPO_ROOT,
                            verify_math500: bool = True,
                            reasoning_parser: Optional[str] = None,
                            limits: Optional[Mapping[str, int]] = None,
                            agent_loop_entry: Optional[bool] = None,
                            tool_calls: Optional[bool] = None,
                            thinking_by_task: Optional[Mapping[str, str]] = None,
                            math500_retry_max_tokens: int = 0,
                            max_model_len: Optional[int] = None,
                            sampling: Optional[C.SamplingRow] = None,
                            samples_by_task: Optional[Mapping[str, int]] = None,
                            or_bench_item_set: Optional[Mapping[str, Any]] = None,
                            grader_max_tokens: int = GRADER_MAX_TOKENS,
                            out_of_support: bool = False,
                            prompt_opened_think: bool = False,
                            stamp_policy: Optional[str] = None,
                            ledger_path: Optional[Path] = None,
                            off_cap_override: bool = False,
                            family: Optional[str] = None,
                            header_located: bool = True,
                            require_certified_fact: bool = True,
                            oos_experiment: Optional[str] = None,
                            sampling_ref: Optional[str] = None,
                            request_evidence_sink: Optional[Callable[..., Any]] = None,
                            # the TARGET's response-side evidence sink, beside the
                            # request sink it pairs with (request_evidence_sinks); keyword-only, so its position is free
                            response_evidence_sink: Optional[Callable[..., Any]] = None,
                            on_cap_override: Optional[Mapping[str, int]] = None,
                            item_subset: Optional[Sequence[str]] = None,
                            attempt_scope: Optional[C.AttemptScope] = None,
                            cap_overrides_by_task: Optional[Mapping[str, int]] = None,
                            item_selection_by_task: Optional[Mapping[str, Sequence[str]]] = None,
                            off_cap_setter: Optional[Mapping[str, int]] = None,
                            row_identity: Optional[Mapping[str, Any]] = None,
                            item_subset_draws: Optional[Sequence[Sequence[object]]] = None,
                            sampling_by_task: Optional[Mapping[str, C.SamplingRow]] = None,
                            protocol_temperature_by_task: Optional[Mapping[str, float]] = None,
                            row_provenance: Optional[Mapping[str, Any]] = None,
                            inspect_seed_policy: Optional[str] = None,
                            inspect_seed_base: Optional[int] = None,
                            verification_phase: str = C.VERIFICATION_PHASE_COUPLED,
                            generation: Optional["GenerationPhase"] = None
                            ) -> Dict[str, Any]:
    """Every Inspect control atom this cell asked for (Lane A's dispatch
    entry point).

    ``verification_phase``
    is the driver's ``--verification-phase`` — ``coupled`` (the default; every plan
    generated AND scored here, byte-identical) or ``generate`` (every SANDBOXED
    task's plan is built ``generation_only``: the GPU phase writes its UNSCORED
    ``.eval``, a generation sidecar and a pending marker into the row dir and files
    nothing; the CPU ``verify`` subcommand scores and files it later). ``generate``
    needs ``generation`` (``cpu_verify.GenerationPhase``: the row dir, the cell, the
    GPU FunctionCall id, the concurrency) and at least one sandboxed task among the
    atoms — a phase that would defer nothing refuses. ``verify`` is NOT a value
    here: it is the CPU subcommand, which never enters this dispatch. Under
    ``coupled`` a ``generation`` block is read only for the pending-marker refusal
    (:func:`run_one_task`).

    qwen-base-seed-replicates: ``inspect_seed_policy`` / ``inspect_seed_base`` (the driver's
    ``--inspect-seed-policy`` / ``--inspect-seed-base``) select the Inspect seed rule for this run's
    context (:func:`set_inspect_seed_policy`; ``replicate`` = ``contracts.inspect_replicate_seed`` on
    every non-IFEval task). Both None = ``SEED_RULE_INSPECT``, byte-identical to before.

    ``item_subset_draws`` —
    the ``rung_subset/2`` DRAWS ``[[item_id, sample_idx], …]`` beside
    ``item_subset``'s distinct ids — reaches the IFEval plan through
    :func:`plans_for_atoms`, which narrows each trial to ITS items
    (``TaskPlan.sample_ids_by_sample``); a draw-subset plan already validly filed
    is skipped through the ONE strict validator
    (``rung_probe.draw_subset_task_reason``) and a DONE store whose content is
    not the subset's draws refuses BY NAME (repair = delete the store and re-run;
    never a silent widen). ``None`` = the pre-existing behaviour, byte-identical.

    ifeval-canonical-battery: ``off_cap_setter`` (the OFF-arm rung setter,
    ``contracts.FLAG_OFF_CAP_BY_TASK``) reaches every plan through
    :func:`plans_for_atoms`; ``row_identity`` (the per-row object identity
    block of a ``--bank-view`` run, ``contracts.row_identity_block``) reaches
    :func:`run_one_task`, which stamps it on every emitted row.
    Both default to None — the pre-existing behaviour, byte-identical.

    The ``protocol`` mode: ``sampling_by_task``
    is the TARGET's ``{row task: executed SamplingRow}`` map and
    ``protocol_temperature_by_task`` the benchmarks' declared temperatures —
    handed to :func:`build_model` (the provider selects each request's row by
    its task) and :func:`plans_for_atoms` (each plan carries its own row and
    declaration); ``sampling`` is then None and NO greedy default is
    substituted for the target. Both None on every other run — byte-identical
    behaviour. The GRADER stays greedy in every mode.

    ``row_provenance`` (llama-paper-rows): the ``contracts.row_provenance``
    fragment every emitted row carries under ``--bank-view``; None = rows
    byte-identical to before (:func:`run_one_task`).

    gsm8k-paired200-paper-objects: ``on_cap_override`` (the bounded ON-cap
    setter) and ``item_subset`` (a rung's eligible ids) reach every plan
    through :func:`plans_for_atoms`; ``attempt_scope`` reaches
    :func:`run_one_task`. Every default is the pre-existing behaviour.

    ``family`` feeds every plan's render fact (the split mode from
    ``contracts.FAMILY_RENDER_FACTS``); ``require_certified_fact`` (production
    True) refuses a family whose fact is only ``candidate``; ``header_located``
    is the target route's declaration predicate (False on an undeclared
    all-position object: no ``petri_thinking_on`` on any request);
    ``oos_experiment`` names the MATCHED authorisation tuple whose cap map the
    ON plans read; ``sampling_ref`` rides on every row. The GRADER model is
    built ONLY when a planned task is model-graded (``MODEL_GRADED_TASKS``) —
    a MATH-only run constructs no grader. Every default is the pre-existing
    behaviour.

    ``ledger_path``: the run's durable usage-ledger file when the
    run has one (the driver's ``<out_root>/usage_ledger.jsonl``, the same
    path its manifest records) — every task's summary then carries the
    ledger's count of that task's rejected calls (:func:`run_one_task`).

    ``cap_overrides_by_task`` / ``item_selection_by_task``: the truncation
    ladder's rung-2 tuple, handed to
    :func:`plans_for_atoms` as given — the raised cap per fired row task and its
    capped items (log vocabulary); both None on every other run.

    ``stamp_policy`` (fit-support transfer, review round B P1): the
    ``resumed_prefill_policy`` label the TARGET's responses must wear, resolved
    by the driver off the standup record (``contracts.expected_stamp_policy``)
    and handed to the target's provider; the grader is the base route and
    consults no policy. ``None`` = the authorised experiment's contract row.

    ``out_of_support`` (the driver's ``--allow-out-of-support-thinking``): a
    thinking-ON GSM8K plan takes the authorised raised cap, and a done marker is
    trusted only WITH a valid ``outcomes/<task>.json`` beside it
    (``completion.is_done(require_outcomes)``); a torn store's record is REBUILT
    from its retained ``.eval`` logs or refused by name
    (:func:`repair_task_outcomes`) — never re-run into the store, whose
    skip-on-key row emission would leave the old rows under a new summary.
    ``prompt_opened_think``: the experiment's DECLARED render
    fact — with no composed parser the GSM8K scorer splits as ``implicit_open``.
    Both default to the in-support behaviour.

    ``off_cap_override`` (the driver's ``--allow-off-cap-override``): the CAP
    override — a thinking-OFF GSM8K plan takes the authorised comparator's
    raised cap. Two flags, two meanings, kept apart:
    ``out_of_support`` is the RENDER authorisation and reaches the task
    PLANNER unchanged (False for the comparator, whose render is the fit
    render); ONE derived scope, ``held_to_record = out_of_support or
    off_cap_override``, governs the record enforcement — ``build_model`` (the
    provider's fail-closed stamp check, target and grader), ``is_done``'s
    ``require_outcomes``, the repair branch and ``run_one_task``'s stamp fields
    — so a comparator store whose DONE task has a torn or missing outcome
    record is REPAIRED from its retained ``.eval`` log or REFUSED by name, never
    re-run through the dedup emitter. Nothing hands ``held_to_record`` to a
    planner; nothing hands ``out_of_support`` to a record compare.

    ``base_url`` is the usage ledger's ``/v1`` when one is running and the
    RoutingProxy's otherwise (P36); ``system`` is the prompted cell's SI
    (``None`` for a fitted cell or the base route) and is injected by the
    provider, never by an Inspect solver. ``reasoning_parser`` is the
    parser the standup composed (the driver reads it off ``endpoint.json``;
    None when none was) — a server fact, so it reaches the grader's model
    as well as the target's. ``limits`` is the driver's active limit set
    (``completion.active_limits``; ``inspect_limit`` is ALSO applied here as
    the smoke cap) — any active limit makes every task end PARTIAL (R2).
    ``agent_loop_entry`` / ``tool_calls`` are the standup's two served facts
    that decide whether BFCL can run on this route (the driver reads them
    off ``endpoint.json``; :func:`assert_route_can_serve`), and the live
    proxy's identity block must agree (:func:`assert_live_agent_loop_entry`).
    A task with a done marker is skipped and its persisted summary
    (``summary/<task>.json``) returned, so a resumed run reports the same
    numbers as the run that finished it.

    PW20b-D. ``thinking_by_task`` is the run's ``{row task: arm}`` map: when
    given, every plan takes its own task's arm and the TARGET model serves the
    flag per request under ONE standup (``thinking`` is then only the mode
    token for the record; the GRADER stays thinking-OFF either way, because an
    instrument that varies with the task it grades is not one).
    ``math500_retry_max_tokens`` arms the MATH-500 truncation retry (U5-3; 0 =
    off) and ``max_model_len`` is the served window it must fit.

    PW-sampling (U8). ``sampling`` is the run's :class:`contracts.SamplingRow`
    (the driver resolves it: the model's card row under ``--sampling card``,
    ``GREEDY_SAMPLING`` under ``greedy``; None here = greedy, the pre-U8
    draw) — the TARGET is built with it and every plan carries it; the
    GRADER is ALWAYS built with ``GREEDY_SAMPLING``: an instrument that
    sampled at the target's temperature is not one (before this the grader's
    temperature was never pinned client-side at all — it rode on the
    server's default). ``samples_by_task`` is the run's ``{row task: k}``
    map (``contracts.run_samples_by_task``); every plan takes its own k
    from it, and a task it does not name is refused at plan time.

    PW-battery (U10 item 4). ``or_bench_item_set`` is the run's HASHED
    OR-Bench spec (``contracts.or_bench_item_set``): the battery's frozen,
    seeded 200-item subset by default, the whole pinned corpus under the
    driver's ``--or-bench-full``. Passing the spec the identity RECORDED —
    rather than re-deriving it here — is what makes the rows that ran and the
    rows the identity claims the same rows; ``None`` falls back to the battery
    default, for a caller that predates the contract.

    E56 (3). ``grader_max_tokens`` is the GRADER's completion cap
    (:data:`GRADER_MAX_TOKENS`), and this is the ONE place it is chosen: the
    same number builds the grader model and rides on every model-graded
    plan, so the cap that was served and the cap the summary reports are one
    value by construction. It is hashed store identity through the driver's
    resolved config, so a pre-cap store and a capped one are different
    stores."""
    C.check_thinking_arg(thinking)
    # The two GPU-side phases; the sandboxed row tasks are the ones a `generate`
    # run defers to the CPU verifier
    if verification_phase not in (C.VERIFICATION_PHASE_COUPLED, C.VERIFICATION_PHASE_GENERATE):
        raise InspectServedError(
            f"{C.FLAG_VERIFICATION_PHASE} {verification_phase!r} is not a GPU-side phase "
            f"({C.VERIFICATION_PHASE_COUPLED!r} | {C.VERIFICATION_PHASE_GENERATE!r}) — "
            f"{C.VERIFICATION_PHASE_VERIFY!r} is the CPU subcommand `served_capability_run.py verify`, never a "
            "value the generation dispatch takes")
    generation_only_tasks: Optional[List[str]] = None
    if verification_phase == C.VERIFICATION_PHASE_GENERATE:
        if generation is None:
            raise InspectServedError(
                f"{C.FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_GENERATE} needs the row's generation facts "
                f"(the row dir the sidecar and pending marker are written to) — the driver derives them from "
                f"{C.FLAG_ROW_ENGINE_RECORD}")
        every_row_task = [t for a in atoms for t in C.TASKS_OF_CONTROL.get(a, ())]
        generation_only_tasks = [rt for rt in dict.fromkeys(every_row_task) if _task_name(rt) in SANDBOXED_TASK_NAMES]
        if not generation_only_tasks:
            raise InspectServedError(
                f"{C.FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_GENERATE}: the atoms {list(atoms)} plan no "
                f"sandboxed task ({list(SANDBOXED_TASK_NAMES)}) — nothing to defer to a CPU verifier; refusing a "
                "phase that would silently do nothing")
    # qwen-base-seed-replicates: the run's Inspect seed policy into THIS context (the tasks Inspect spawns copy it);
    # `default` stores None — the dispatch is byte-identical to before the policy existed
    set_inspect_seed_policy(inspect_seed_policy, inspect_seed_base)
    # The ONE record-enforcement scope, computed once.
    # `out_of_support` (the render authorisation) goes to the planner alone;
    # `held_to_record` (the attestation posture) to admission, repair and the
    # stamp checks. Never the other way round.
    held_to_record = bool(out_of_support or off_cap_override)
    # E56 (3), before a route is read or a model is built: there is no "off".
    # An unbounded verdict generation is the defect this cap exists to remove.
    if int(grader_max_tokens) <= 0:
        raise InspectServedError(
            f"grader_max_tokens must be a positive completion cap, got {grader_max_tokens!r} "
            "— an uncapped model grader is what E56 (3) refuses; there is no 'off'")
    # Under a per-task map (`protocol`) the target has NO single row — never
    # the greedy default — and the provider selects each request's row by its
    # task
    if sampling_by_task is not None and sampling is not None:
        raise InspectServedError(
            "run_inspect_tasks: sampling (one run row) and sampling_by_task (per-task rows) were both "
            "passed — the target's row has exactly one source")
    target_sampling = (None if sampling_by_task is not None
                       else sampling if sampling is not None else C.GREEDY_SAMPLING)
    if C.is_per_task(thinking) and not thinking_by_task:
        raise InspectServedError(
            f"thinking={thinking!r} but no thinking_by_task map was passed — "
            f"{C.THINKING_PER_TASK!r} is an instruction to read each task's arm off "
            "the map, not an arm; the driver resolves the map and passes it")
    api_key = api_key if api_key is not None else os.environ.get(
        C.INSPECT_API_KEY_ENV)
    target_vmodel = str(_route_field(route, "virtual_model"))
    g_route = grader_route(routes)
    grader_vmodel = str(_route_field(g_route, "virtual_model"))
    models_doc = get_models(base_url=base_url, api_key=api_key)
    preflight_models(base_url=base_url, api_key=api_key,
                     required=[target_vmodel, grader_vmodel], doc=models_doc)
    task_names = [_task_name(t) for a in atoms for t in C.TASKS_OF_CONTROL.get(a, ())]
    # the route, AS STOOD UP, must be able to serve every task (U4(b)):
    # refused here, before any model is built or GPU minute spent
    assert_route_can_serve(task_names, route, agent_loop_entry=agent_loop_entry,
                           tool_calls=tool_calls)
    assert_live_agent_loop_entry(task_names, route, models_doc,
                                 declared=agent_loop_entry, base_url=base_url)

    # each model is built WITH its route: the target's responses are
    # stamp-verified against the target's entry, the grader's against the
    # grader's (E10 N2)
    # under a per-task map the ARM is a request-time fact (the map), not a
    # constructor one; `thinking` may then be the mode token, which the
    # provider's constructor would refuse, so the base arm is only passed
    # without a map
    # the Qwen seed-replicate native battery: the TARGET's request AND response sides —
    # the caller's sinks with the lane switch off (byte-identical records, no response record, no row binding);
    # the lane client's BOTH when SISTEER_REQUEST_EVIDENCE_DIR is set (request_evidence_sinks docstring)
    request_evidence_sink, response_evidence_sink = request_evidence_sinks(request_evidence_sink, response_evidence_sink)
    target = build_model(target_vmodel, base_url=base_url, api_key=api_key,
                         si=system,
                         thinking=(C.THINKING_OFF if thinking_by_task else thinking),
                         concurrency=concurrency,
                         reasoning_parser=reasoning_parser, route=route,
                         thinking_by_task=thinking_by_task,
                         sampling=target_sampling,
                         out_of_support=held_to_record,
                         stamp_policy=stamp_policy,
                         header_located=bool(header_located),
                         request_evidence_sink=request_evidence_sink,
                         # bcb-bfcl-200-widen: the per-task executed rows under `protocol`
                         sampling_by_task=sampling_by_task,
                         response_evidence_sink=response_evidence_sink)
    if provider_of(target) is None:
        # fit-support transfer: the regime stamps are read off the
        # provider's accumulator; a target without one could only file null
        # regime fields, so a production run refuses rather than degrade
        raise InspectServedError(
            f"the target model for {target_vmodel!r} exposes no tinycap provider (no "
            "`api.regime_stamps` / `api.route`) — its responses' regime stamps could not be "
            "read into the task summaries; refusing")
    # The grader is the UNSTEERED base route, never prompted, never
    # thinking and never SAMPLING: it is an instrument, and an instrument
    # that varies with the arm or draws at the target's temperature is not
    # one. (Its judgments stay comparable across all four configurations
    # for exactly that reason.) It is built with NO per-task map for the
    # same reason: its arm must not follow the task it grades.
    # ...and never uncapped (E56 (3)): the grader is not the active model, so
    # no task's eval-level cap has ever reached it, and an unbounded verdict
    # generation on a 16k window is where the wave's six gateway timeouts came
    # from. The cap lives on the grader's own config; ONE number also rides on
    # every model-graded plan, so what was served and what is reported agree.
    # a MATH-ONLY run constructs NO
    # grader model at all (a spy on build_model sees one call) — its plan's
    # scorer is judge-free, so no grader request can ever leave it; every
    # other run builds the grader exactly as before (the pins on GSM8K / BFCL
    # runs, which never used it either, stay byte-identical)
    grader = None
    math_only = bool(task_names) and all(n == "math" for n in task_names)
    if not math_only:
        grader = build_model(grader_vmodel, base_url=base_url, api_key=api_key,
                             si=None, thinking=C.THINKING_OFF,
                             concurrency=concurrency,
                             reasoning_parser=reasoning_parser, route=g_route,
                             sampling=C.GREEDY_SAMPLING,
                             max_tokens=int(grader_max_tokens),
                             out_of_support=held_to_record,
                             # the grader's attempts are ledger rows too, so
                             # they carry their record like the target's
                             request_evidence_sink=request_evidence_sink)

    plans = plans_for_atoms(atoms, grader_model=grader, thinking=thinking,
                            thinking_by_task=thinking_by_task,
                            sampling=target_sampling, samples_by_task=samples_by_task,
                            inspect_limit=inspect_limit, samples=samples,
                            repo_root=repo_root, verify_math500=verify_math500,
                            retry_max_tokens=math500_retry_max_tokens,
                            or_bench_item_set=or_bench_item_set,
                            grader_max_tokens=int(grader_max_tokens),
                            # fit-support transfer (S2.1): the RENDER
                            # authorisation, the declared render fact and the
                            # composed parser pick the ON cap and the
                            # think-split mode per plan; the CAP override
                            # (offcap §3.3.2) picks the comparator's OFF cap.
                            # The planner never sees `held_to_record`.
                            out_of_support=bool(out_of_support),
                            prompt_opened_think=bool(prompt_opened_think),
                            reasoning_parser=reasoning_parser,
                            off_cap_override=bool(off_cap_override),
                            family=family, require_certified_fact=bool(require_certified_fact),
                            oos_experiment=oos_experiment,
                            on_cap_override=on_cap_override, item_subset=item_subset,
                            # bcb-bfcl-200 §S2b: the ladder tuple, per task
                            cap_overrides_by_task=cap_overrides_by_task,
                            item_selection_by_task=item_selection_by_task,
                            # ifeval-canonical-battery §5.2: the OFF-arm rung setter;
                            # §5.4: the rung_subset/2 draws beside the item subset
                            off_cap_setter=off_cap_setter, item_subset_draws=item_subset_draws,
                            # bcb-bfcl-200-widen §S2: the protocol rows + declarations, per task
                            sampling_by_task=sampling_by_task,
                            protocol_temperature_by_task=protocol_temperature_by_task,
                            # tinycap-shared-standup landing 2 §2.2.1: the deferred (generate-only) tasks
                            generation_only_tasks=generation_only_tasks)
    out: Dict[str, Any] = {}
    for plan in plans:
        if getattr(plan, "draw_subset", None) is not None:      # getattr: a test's stub plan carries no field
            # a DRAW-subset rung's done marker is trusted through
            # the strict validator ALONE (run_one_task's first step: a valid store is
            # skipped with zero requests; a DONE marker that is not one refuses by
            # name) — never through the generic is_done skip below
            out[plan.row_task] = await run_one_task(
                plan, model=target, store=store, template=template,
                thinking=plan.thinking, concurrency=concurrency,
                route_virtual_model=target_vmodel,
                reasoning_parser=reasoning_parser, limits=limits,
                max_model_len=max_model_len, out_of_support=held_to_record,
                ledger_path=ledger_path, sampling_ref=sampling_ref, attempt_scope=attempt_scope,
                row_identity=row_identity, generation=generation)
            continue
        # S2.8 / offcap §3.3.2: under EITHER override a done marker is trusted
        # only WITH its valid outcome record
        if completion.is_done(store.cell_dir, plan.row_task,
                              require_outcomes=held_to_record):
            prior = completion.read_task_summary(store.cell_dir, plan.row_task)
            marker = json.loads(
                completion.done_marker_path(store.cell_dir, plan.row_task).read_text())
            out[plan.row_task] = {"skipped": "done marker present",
                                  **(prior if prior is not None else marker)}
            continue
        if held_to_record and completion.done_marker_path(store.cell_dir, plan.row_task).is_file():
            # a DONE task whose record is missing or invalid is REBUILT from its
            # retained .eval logs (no new draw) or refused by name — never
            # re-run: a regeneration would emit through the store's
            # (task, item, metric) dedup and leave the old rows under a new
            # summary (repair_task_outcomes)
            out[plan.row_task] = repair_task_outcomes(
                plan, store=store, thinking=plan.thinking, route_virtual_model=target_vmodel)
            continue
        out[plan.row_task] = await run_one_task(
            plan, model=target, store=store, template=template,
            thinking=plan.thinking, concurrency=concurrency,
            route_virtual_model=target_vmodel,
            reasoning_parser=reasoning_parser, limits=limits,
            max_model_len=max_model_len, out_of_support=held_to_record,
            ledger_path=ledger_path, sampling_ref=sampling_ref, attempt_scope=attempt_scope,
            row_identity=row_identity, generation=generation)
    return out
