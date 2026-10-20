"""request_evidence_client — the SHARED HTTP-boundary hook every served TARGET driver
carries.

WHAT IT RECORDS, AND WHY HERE. The vendor row on every request and a seed on every
draw are VERIFIED, not declared — read from the WIRE, never from the engine's
post-default ``SamplingParams`` log: a key the client omitted and the server filled
reads identically there, and ``seed`` is not unique on an engine.
So every target request is recorded at the ONE place the final body exists as bytes
— the HTTP boundary — as a version-3 request-evidence record
(:func:`contracts.request_evidence_record`: the sampling terms AS SENT,
``presence_penalty`` presence + value, the trial identity parsed off the sample
header), bound to the usage ledger's row of the same attempt by the client-minted
``X-Tinycap-Request-Id`` and the sha256 of the exact bytes, and bound to the ENGINE
by the response's ``id`` (the external completion id the engine log carries with
one ``-<8 hex>`` suffix, ``serving_attestation.is_engine_id_of``). The response side
is written beside it (``response_evidence.jsonl``) and handed back to the row writer
so every filed row carries ``request_id`` + ``completion_id``.

THE TWO TINYCAP HEADERS ARE THE STORE'S OWN LABELS.
``contracts.request_evidence_reconciliation`` requires the record's ``task`` to
equal the ledger row's and, for COVERAGE, to equal the store's FILED pair's task
exactly (``filed_item_pairs`` keeps the store's own task string); it reads the
coverage item as the sample header's FIRST slash component while the v3 parser
reads ``item_id`` / ``sample_idx`` by ``rpartition``. Hence:

* ``X-Tinycap-Task`` = the store's CANONICAL row task — :func:`store_task_name`,
  derived from each store's OWN writer (never a plan table): GSM8K
  ``inspect/gsm8k_paired200`` (``contracts.inspect_task``), the sampled controls
  ``judged/xstest_v2`` / ``judged/or_bench_hard_1k`` (``tiers.judged.TASK_*`` — the
  ``JudgedSpec.task`` the spool header and every judged row carry), MMLU
  ``s2parity/mmlu_letter`` (``tiers.parity.PARITY_TASK_MMLU``), Petri
  ``petri/<behaviour>`` (``tiers.petri_runner.PETRI_TASK_PREFIX``); and, for the four
  stores with no ``completed_keys()`` (static, BoN, multi-turn, misalign), the ONE
  lane spelling derived from the store's own row keys — ``static/<family>``,
  ``strongreject_bon``, ``multi_turn/<attack>``, ``<family>/<behavior>`` — used by the
  header HERE and by the audit's filed-pairs builders (:func:`filed_pairs_static` …)
  so the two cannot drift.
* ``X-Tinycap-Sample`` = ``<slash-free item_id>/<numeric idx>``
  (``proxy_engine.ledger_headers`` over ``contracts.sample_header_value``): the
  goal / item / scenario id and the variant index, draw index, or TURN (multi-turn
  and Petri: one request per goal per turn, so the turn is the numeric index and the
  seed closure reads it as such). An item id carrying ``/`` is REFUSED — coverage
  would read the wrong component.
* the ARM binds through the body's ``model`` (the route's virtual model, ``vmodel``);
  leg / arm / seed / attempt ride as EXTRA fields of the record and the row
  (``extra``), which the reconciliation ignores — never in the two headers.

DEFAULT OFF, ADDITIVE. Nothing here runs unless the lane asks: the process-level
switch is the environment variable :data:`ENV_DIR` (the directory the two evidence
files are written to; the Modal launcher exports it per attempt root). With it
absent :func:`from_env` is ``None`` and every ``maybe_*`` entry point returns its
argument untouched, so every existing driver sends byte-identical requests and
writes byte-identical rows. :data:`ENV_EXTRA` (a JSON object — ``leg``, ``arm``,
``s``, ``attempt_id``) is stamped on every record and response row as ``extra``.

HOW A DRIVER USES IT (one import + guarded calls; nothing else in a driver
changes): the driver that KNOWS the item identity sets the label around its send —
``with request_evidence_client.labelled(task, item_id, idx): …`` (a ContextVar,
copied into ``asyncio.to_thread`` workers and each ``create_task``); the TRANSPORT
is wrapped once where it is built — :meth:`RequestEvidenceClient.attach_proxy_client`
(the sync ``ProxyClientEngine`` behind ``ServedChatTarget`` / BoN's ``ServedDefense``:
its ``on_request`` hook sees the exact ``urllib`` bytes), :meth:`~RequestEvidenceClient.install_httpx`
(the Inspect providers' ``httpx`` client — the Petri target; the tinycap provider
records its own request side and takes only the response hook), :meth:`~RequestEvidenceClient.aiohttp_trace_config`
(the controls / MMLU aiohttp poster: ``on_request_chunk_sent`` is the exact bytes),
:meth:`~RequestEvidenceClient.wrap_post_json` (the misalign runner's ``httpx.Client``
transport). A labelled send mints a fresh id per HTTP attempt (a retry is a new
attempt), records the request from the bytes on the wire, and records the response;
the row writer then reads ``row_fields(resp)`` (by the response's ``id``) or the
``ServedTurn`` / ``last_request_binding`` the target carried back.

An UNLABELLED send is refused by name — except where the call site declares its
sends are the driver's PROBES (``unlabelled_kind``: the positive controls behind
``build_served_target`` are ``probe/bites`` attempts, labelled ``<vmodel>/<n>`` like
the tinycap driver's own).
"""
from __future__ import annotations

import contextlib
import base64
import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextvars import ContextVar, Token
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Set, Tuple)

from sisteer.capeval.serving.proxy_engine import ledger_headers
from sisteer.capeval.tinycap import contracts as C

__all__ = [
    "ENV_DIR", "ENV_EXTRA", "ENV_SEED_BASE", "EXTRA_SEED_BASE_KEY", "RESPONSE_EVIDENCE_FILENAME", "RESPONSE_EVIDENCE_FIELDS",
    "RESPONSE_EVIDENCE_FIELDS_V1", "RESPONSE_EVIDENCE_VERSION", "REQUEST_BODY_B64_FIELD", "TEXT_SHA256_OF_CONTENT",
    "TEXT_SHA256_OF_BODY", "response_text_sha256", "check_response_row", "read_response_evidence",
    "check_request_record", "read_request_evidence",
    "ROW_REQUEST_ID_FIELD", "ROW_COMPLETION_ID_FIELD", "BINDING_FIELDS", "EXTRA_FIELD",
    "LEG_GSM8K", "LEG_XSTEST", "LEG_OR_BENCH", "LEG_MMLU", "LEG_STATIC", "LEG_BON", "LEG_MULTI_TURN",
    "LEG_MISALIGN", "LEG_PETRI", "LEGS", "TABLE_LEGS", "STATIC_FAMILIES", "MULTITURN_ATTACKS", "CONTROL_LEGS",
    "MISALIGN_BEHAVIORS", "MISALIGN_FAMILY_AGENTIC", "PROBE_KIND_BITES",
    "RequestEvidenceError", "Label", "RequestEvidenceClient",
    "store_task_name", "check_item_id", "label_headers",
    "filed_pairs_static", "filed_pairs_bon", "filed_pairs_multiturn", "filed_pairs_misalign",
    "filed_pairs_petri", "filed_pairs_tinycap", "filed_pairs_judged",
    "labelled", "labelled_or_none", "set_label", "reset_label", "current_label", "require_label",
    "from_env", "enabled", "reset_env_cache",
    "maybe_attach_proxy_client", "maybe_wrap_post_json", "maybe_instrument_inspect_target", "maybe_install_petri_target_hooks",
    "maybe_aiohttp_trace_configs", "row_fields", "turn_fields", "last_binding_fields",
    "multiturn_label_fn", "petri_turn_index", "petri_turn_headers",
    "maybe_request_headers_fn", "MisalignRequestEvidence", "maybe_misalign_request_evidence",
    "EXTRA_ATTEMPT_ID_KEY", "PETRI_TARGET_TURNS_STORE_KEY", "PETRI_TARGET_TURNS_ROW_KEY",
    "petri_sample_turn_sink", "petri_target_turns_of_sample", "petri_target_turn_draws", "PetriTurnDraws",
    "check_sample_idx", "make_label",
]

# --------------------------------------------------------------------------- #
# the lane switch and the files                                                #
# --------------------------------------------------------------------------- #

#: The process-level switch: the directory ``request_evidence.jsonl`` /
#: ``response_evidence.jsonl`` are appended to (the attempt root in a Modal
#: container). ABSENT = off: every ``maybe_*`` entry point is the identity.
ENV_DIR = "SISTEER_REQUEST_EVIDENCE_DIR"
#: OPTIONAL: a JSON object of extra fields (``leg``, ``arm``, ``s``, ``attempt_id`` …)
#: stamped on every record and every response row under :data:`EXTRA_FIELD`.
ENV_EXTRA = "SISTEER_REQUEST_EVIDENCE_EXTRA"
#: the seed base ``s`` of the run (the LITERAL training seed of the view, or the client-side base an
#: Arditi driver was launched with — the rendered Arditi configs' ``sampled_decode.seed_base``, the
#: drivers' ``--seed-base`` / ``--decode-seed-base`` / ``--target-seed-base``); stamped as ``extra.s`` on
#: EVERY record. REQUIRED: the audit refuses an Arditi root whose task records carry no ``extra.s``
#: (three seed bases share one task on that engine), so a client is never built without it — either this
#: variable or ``s`` inside :data:`ENV_EXTRA` (both given must agree).
ENV_SEED_BASE = "SISTEER_REQUEST_EVIDENCE_SEED_BASE"
#: the ``extra`` key the seed base rides under (``request_evidence_audit.record_extra_s`` reads it)
EXTRA_SEED_BASE_KEY = "s"
#: the ``extra`` key the attempt id rides under (the lane's ``{"leg", "arm", "s", "attempt_id"}``); when present it is
#: ALSO stamped top-level on every request record as ``contracts.REQUEST_EVIDENCE_ATTEMPT_FIELD`` — production's
#: stamp (``scripts/served_capability_run.py``'s ``_RequestEvidence``), which ``contracts.records_of_attempts``
#: (``--attempt-id`` scoping) requires of EVERY record of a flagged run
EXTRA_ATTEMPT_ID_KEY = "attempt_id"
#: the Inspect SAMPLE STORE key the Petri response hook files each TARGET turn's attempt binding under
#: (``inspect_ai`` persists ``TaskState.store`` into the eval log's ``sample.store``; review fix P1-3): a list, in
#: wire order, of ``{turn, request_id, completion_id, status, task}`` — a retry (a 5xx, then its retry) is two
#: entries at one turn, a rollback a later entry at an earlier turn; nothing is collapsed. ``petri_runner.
#: convert_eval_log`` copies the list onto every row of the scenario under :data:`PETRI_TARGET_TURNS_ROW_KEY`
PETRI_TARGET_TURNS_STORE_KEY = "tinycap_target_turns"
#: ... the ``extra_json`` key it rides under on the Petri store rows (one Draw per successful entry for the audit:
#: :func:`petri_target_turn_draws`)
PETRI_TARGET_TURNS_ROW_KEY = "target_turns"
#: the response side, beside ``contracts.REQUEST_EVIDENCE_FILENAME``
RESPONSE_EVIDENCE_FILENAME = "response_evidence.jsonl"
#: one row per HTTP attempt that returned (or died): the attempt's request id, the
#: engine's external completion id (null on a rejected / dead attempt), the HTTP
#: status (null when the transport died), the body's ``model`` and first choice's
#: ``finish_reason``, the two labels the attempt carried, and the error text of a
#: failed attempt; ``extra`` (:data:`EXTRA_FIELD`) is appended when the lane set any
RESPONSE_EVIDENCE_FIELDS_V1: Tuple[str, ...] = (
    "ts", "request_id", "completion_id", "status", "vmodel", "finish_reason", "task", "sample", "error")
#: version 2 (additive — the determinism check): the row names its
#: version first and carries ``text_sha256`` + ``text_sha256_of`` after ``finish_reason`` — the sha256 (64 lowercase
#: hex) of the UTF-8 bytes of the EXACT response text this client extracted, ``choices[0].message.content`` when it
#: is a string (``text_sha256_of = "content"``: the same text the stores persist and the replay hashes through
#: ``ac.text_sha256(completion_text(resp))``), else of the canonical JSON of the FULL body when there is no single
#: text field (``"body"``: no choices, or a first choice whose content is not a string — a replay's content hash can
#: never equal it, so such a row reads as a divergence, never as a match; a body with SEVERAL choices hashes the
#: first's content, exactly as the replay reads it); both None when no body was recorded (a failed or dead
#: attempt). A thinking-ON reply's ``reasoning_content`` is NOT part of the content hash — the same limit as the
#: stores' text and the replay's comparison. Lane-on only by construction: the file exists only when the
#: switch is set.
RESPONSE_EVIDENCE_VERSION = 2
RESPONSE_EVIDENCE_FIELDS: Tuple[str, ...] = (
    "response_evidence_version", "ts", "request_id", "completion_id", "status", "vmodel", "finish_reason",
    "text_sha256", "text_sha256_of", "task", "sample", "error")
TEXT_SHA256_OF_CONTENT = "content"
TEXT_SHA256_OF_BODY = "body"
#: the exact wire bytes of every recorded request, base64, as an ADDITIVE field ON the request record — the version-3
#: record itself carries the body's sha256 and its LENGTH (``body_bytes``), not the bytes, and a byte-identical replay
#: needs them; the determinism check reads this field (``ac.REQUEST_BODY_B64_FIELD``,
#: ``bodies_from_records``) and holds it to ``body_sha256`` and ``body_bytes`` before re-sending. Lane-on only, like
#: every record.
REQUEST_BODY_B64_FIELD = "body_b64"
#: the two ADDITIVE row fields every filed row of a labelled run carries
ROW_REQUEST_ID_FIELD = "request_id"
ROW_COMPLETION_ID_FIELD = "completion_id"
BINDING_FIELDS: Tuple[str, str] = (ROW_REQUEST_ID_FIELD, ROW_COMPLETION_ID_FIELD)
#: the record / row key the lane's leg / arm / seed / attempt ride under (ignored by
#: ``request_evidence_reconciliation``; never in the two tinycap headers)
EXTRA_FIELD = "extra"
#: the probe kind of the positive controls ``build_served_target`` runs through the
#: SAME client before any task request (``served_evidence`` / ``probe_fn``: greedy
#: steering-bites requests on the cell and its scale-0 twin)
PROBE_KIND_BITES = C.PROBE_KIND_BITES
_CHAT_SUFFIX = "/chat/completions"


class RequestEvidenceError(RuntimeError):
    """A labelling / recording refusal — loud, never a silently unlabelled request."""


# --------------------------------------------------------------------------- #
# the canonical row task per leg, derived from each store's OWN writer         #
# --------------------------------------------------------------------------- #

LEG_GSM8K = "gsm8k"
LEG_XSTEST = "xstest"
LEG_OR_BENCH = "or_bench_hard_1k"
LEG_MMLU = "mmlu"
LEG_STATIC = "static"
LEG_BON = "bon"
LEG_MULTI_TURN = "multi_turn"
LEG_MISALIGN = "misalign"
LEG_PETRI = "petri"
LEGS: Tuple[str, ...] = (LEG_GSM8K, LEG_XSTEST, LEG_OR_BENCH, LEG_MMLU, LEG_STATIC, LEG_BON,
                         LEG_MULTI_TURN, LEG_MISALIGN, LEG_PETRI)
#: the battery's 15 TABLE legs as the audit spells them (``audit_common.LEGS``):
#: :func:`store_task_name` accepts these spellings too — a family / attack / control NAMES its leg, the
#: misalign behaviours ride as ``misalign_<behavior>`` — so the audit's drift check
#: (``check_client_task_names``) can call ``store_task_name(leg[, sub])`` with them
STATIC_FAMILIES: Tuple[str, ...] = ("pasttense", "codeattack", "renellm", "grok420")
MULTITURN_ATTACKS: Tuple[str, ...] = ("pair", "crescendo", "actorattack")
CONTROL_LEGS: Tuple[str, ...] = (LEG_XSTEST, LEG_OR_BENCH)
MISALIGN_BEHAVIORS: Tuple[str, ...] = ("leaking", "murder")
_MISALIGN_LEG_PREFIX = "misalign_"
TABLE_LEGS: Tuple[str, ...] = (*STATIC_FAMILIES, LEG_BON, *MULTITURN_ATTACKS, *CONTROL_LEGS, LEG_MMLU, LEG_GSM8K,
                               LEG_PETRI, *(f"{_MISALIGN_LEG_PREFIX}{b}" for b in MISALIGN_BEHAVIORS))
assert len(TABLE_LEGS) == 15
#: the misalign runner's family this battery runs (its rows carry ``family`` + ``behavior``)
MISALIGN_FAMILY_AGENTIC = "agentic"
#: the four legs whose store has NO ``completed_keys()``: the lane DEFINES the task
#: string from the store's own row keys, spelled ONCE here (the header and the audit's
#: filed-pairs builders both call :func:`store_task_name`)
_STATIC_PREFIX = "static/"
_BON_TASK = "strongreject_bon"
_MULTI_TURN_PREFIX = "multi_turn/"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")


def _plain_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise RequestEvidenceError(f"{what} must be a plain name ([A-Za-z0-9_.-], no '/'), got {value!r}")
    return value


def store_task_name(leg: str, name: Optional[str] = None, *, family: Optional[str] = None) -> str:
    """The store's CANONICAL row task for ``leg`` — the ``X-Tinycap-Task`` every target
    request of that leg carries AND the task the audit's filed pairs are keyed by.
    ``name`` is the leg's sub-key where the store has one: the static ``family``
    (``pasttense`` …), the multi-turn ``attack`` (``pair`` / ``crescendo`` /
    ``actorattack``), the misalign ``behavior`` (``leaking`` / ``murder``; ``family``
    defaults to ``agentic``), the Petri ``behaviour``. Every string is read off the
    writer that files the store's rows, never restated here. The 15 TABLE-leg
    spellings (:data:`TABLE_LEGS`) are accepted as ``leg`` with no ``name`` —
    ``store_task_name("pasttense")``, ``("pair")``, ``("misalign_leaking")`` — the
    audit's own vocabulary."""
    if leg in STATIC_FAMILIES and name is None:
        return store_task_name(LEG_STATIC, leg)
    if leg in MULTITURN_ATTACKS and name is None:
        return store_task_name(LEG_MULTI_TURN, leg)
    if leg.startswith(_MISALIGN_LEG_PREFIX) and name is None:
        return store_task_name(LEG_MISALIGN, leg[len(_MISALIGN_LEG_PREFIX):], family=family)
    if leg == LEG_GSM8K:
        return C.inspect_task(C.GSM8K_PAIRED_TASK_NAME)
    if leg in (LEG_XSTEST, LEG_OR_BENCH):
        from sisteer.capeval.tiers import judged
        return judged.TASK_XSTEST if leg == LEG_XSTEST else judged.TASK_ORBENCH
    if leg == LEG_MMLU:
        from sisteer.capeval.tiers import parity
        return parity.PARITY_TASK_MMLU
    if leg == LEG_STATIC:
        return _STATIC_PREFIX + _plain_name(name, "the static family")
    if leg == LEG_BON:
        return _BON_TASK
    if leg == LEG_MULTI_TURN:
        return _MULTI_TURN_PREFIX + _plain_name(name, "the multi-turn attack")
    if leg == LEG_MISALIGN:
        fam = _plain_name(family if family is not None else MISALIGN_FAMILY_AGENTIC, "the misalign family")
        return f"{fam}/{_plain_name(name, 'the misalign behavior')}"
    if leg == LEG_PETRI:
        from sisteer.capeval.tiers.petri_runner import PETRI_TASK_PREFIX
        return PETRI_TASK_PREFIX + _plain_name(name, "the Petri behaviour")
    raise RequestEvidenceError(f"unknown leg {leg!r}; legs: {list(LEGS)} or a table leg {list(TABLE_LEGS)}")


def check_item_id(item_id: object) -> str:
    """The slash-free, non-empty item id a sample header carries: the coverage check
    reads the header's FIRST slash component, so a ``/`` inside the id would make the
    filed item unfindable — refused here, at the label."""
    if not isinstance(item_id, str) or not item_id or "/" in item_id:
        raise RequestEvidenceError(
            f"item id {item_id!r} cannot label a request: it must be a non-empty string without '/' "
            "(X-Tinycap-Sample = <item_id>/<idx>; the reconciliation reads the first slash component)")
    return item_id


def check_sample_idx(sample_idx: object) -> int:
    """The non-negative INTEGER index a sample header carries, validated on the ORIGINAL value: a bool
    (``int(True) == 1``), a numeric string (``"2"``), a float (``2.0``, ``2.5``) or a negative number is
    refused — an adapter that coerced with ``int(...)`` first would alias one draw to ANOTHER valid sample's
    header, and the reconciliation would file the wrong item. Every adapter goes through here (or through
    :func:`make_label`) before a label or a header exists."""
    if isinstance(sample_idx, bool) or not isinstance(sample_idx, int) or sample_idx < 0:
        raise RequestEvidenceError(
            f"sample index must be a non-negative int, got {sample_idx!r} ({type(sample_idx).__name__}) — a bool, a "
            "numeric string or a float is refused rather than coerced (it would alias another sample)")
    return int(sample_idx)


def label_headers(task: str, item_id: str, sample_idx: int) -> Dict[str, str]:
    """The two labels exactly as the tinycap driver spells them
    (``proxy_engine.ledger_headers`` over ``contracts.sample_header_value``), after the
    slash-free check and the ORIGINAL-value index check (:func:`check_sample_idx`)."""
    idx = check_sample_idx(sample_idx)
    if not isinstance(task, str) or not task:
        raise RequestEvidenceError(f"a request label needs a task, got {task!r}")
    return ledger_headers(task, check_item_id(item_id), idx)


# -- the audit's filed-pairs builders: the SAME task strings -------------------------- #

def filed_pairs_static(rows: Iterable[Mapping[str, Any]]) -> Set[Tuple[str, str]]:
    """``(static/<family>, goal_id)`` from a sampled static store's own rows (``responses.jsonl``
    / ``progress.jsonl``: ``family`` + ``goal_id``)."""
    return {(store_task_name(LEG_STATIC, str(r["family"])), str(r["goal_id"])) for r in rows}


def filed_pairs_bon(rows: Iterable[Mapping[str, Any]]) -> Set[Tuple[str, str]]:
    """``(strongreject_bon, goal_id)`` from BoN's ``responses.jsonl`` rows."""
    return {(store_task_name(LEG_BON), str(r["goal_id"])) for r in rows}


def filed_pairs_multiturn(rows: Iterable[Mapping[str, Any]], attack: str) -> Set[Tuple[str, str]]:
    """``(multi_turn/<attack>, goal_id)`` from a multi-turn run's ``progress.jsonl`` /
    ``responses.jsonl`` rows (the attack is the run's, not a row field)."""
    return {(store_task_name(LEG_MULTI_TURN, attack), str(r["goal_id"])) for r in rows}


def filed_pairs_misalign(rows: Iterable[Mapping[str, Any]]) -> Set[Tuple[str, str]]:
    """``(<family>/<behavior>, item_id)`` from the runner's ``<cell>/responses.jsonl`` rows."""
    return {(store_task_name(LEG_MISALIGN, str(r["behavior"]), family=str(r["family"])), str(r["item_id"]))
            for r in rows}


def filed_pairs_petri(completed_keys: Iterable[Tuple[str, str, str]]) -> Set[Tuple[str, str]]:
    """Petri DOES file ``completed_keys()``: ``(petri/<behaviour>, scenario_id)`` through the
    contract's own :func:`contracts.filed_item_pairs` (rev 6)."""
    return C.filed_item_pairs(completed_keys)


def filed_pairs_tinycap(completed_keys: Iterable[Tuple[str, str, str]]) -> Set[Tuple[str, str]]:
    """GSM8K / MMLU (a tinycap ``CellStore``): the contract's own inventory."""
    return C.filed_item_pairs(completed_keys)


def filed_pairs_judged(task: str, rows: Iterable[Mapping[str, Any]]) -> Set[Tuple[str, str]]:
    """The sampled controls: ``(spec.task, item_id)`` over the judged generation spool's /
    store's rows, ``task`` = the spool header's ``task`` (``judged.spool_header``)."""
    return {(str(task), str(r["item_id"])) for r in rows}


# --------------------------------------------------------------------------- #
# the label (a ContextVar: copied into to_thread workers and created tasks)     #
# --------------------------------------------------------------------------- #

class Label(NamedTuple):
    task: str
    item_id: str
    sample_idx: int


_LABEL: "ContextVar[Optional[Label]]" = ContextVar("sisteer_request_evidence_label", default=None)
#: the binding of the LAST response this context observed (the httpx / proxy-client
#: paths set it; the tinycap provider reads it back onto ``ModelOutput.metadata``)
_LAST_BINDING: "ContextVar[Optional[Dict[str, Optional[str]]]]" = ContextVar(
    "sisteer_request_evidence_last_binding", default=None)


def make_label(task: str, item_id: str, sample_idx: object) -> Label:
    """The ONE constructor of a :class:`Label`: the slash-free item id and the index validated on their
    ORIGINAL values (:func:`check_item_id`, :func:`check_sample_idx`) and the header refusals run
    (:func:`label_headers`) BEFORE the tuple exists — ``set_label`` / ``labelled``, ``multiturn_label_fn``,
    ``ServedChatTarget.set_request_context`` and the other seams all build labels here; none coerces first."""
    lab = Label(task, check_item_id(item_id), check_sample_idx(sample_idx))
    label_headers(*lab)   # the same refusals as the header itself
    return lab


def set_label(task: str, item_id: str, sample_idx: object) -> Token:
    return _LABEL.set(make_label(task, item_id, sample_idx))


def reset_label(token: Token) -> None:
    _LABEL.reset(token)


def current_label() -> Optional[Label]:
    return _LABEL.get()


@contextlib.contextmanager
def labelled(task: str, item_id: str, sample_idx: int) -> Iterator[Label]:
    """``with labelled(task, item_id, idx):`` — every send inside carries these labels."""
    tok = set_label(task, item_id, sample_idx)
    try:
        yield _LABEL.get()  # type: ignore[misc]
    finally:
        _LABEL.reset(tok)


@contextlib.contextmanager
def labelled_or_none(lab: Optional[Tuple[str, str, int]]) -> Iterator[Optional[Label]]:
    """:func:`labelled` when ``lab`` is a triple, a no-op when it is ``None`` (the hook
    off, or a driver path with nothing to label)."""
    if lab is None:
        yield None
        return
    with labelled(*lab) as out:
        yield out


def require_label(what: str) -> Optional[Label]:
    """The task-request guard the served targets call before a send: with the hook
    ENABLED, a send that no caller labelled is refused by name (an unlabelled task
    request would be recorded as nothing the reconciliation can cover); with the hook
    off it is a no-op returning ``None``."""
    if not enabled():
        return None
    lab = _LABEL.get()
    if lab is None:
        raise RequestEvidenceError(
            f"{what}: request evidence is ON ({ENV_DIR} is set) but this task request carries no label — the "
            "caller must set one (generate_detailed(labels=…), a label_fn, or `with request_evidence_client."
            "labelled(task, item_id, idx)`) so the ledger row and the record name the filed item")
    return lab


def multiturn_label_fn(attack: str, goal_ref: Mapping[str, Any], key: str = "gid"
                       ) -> Optional[Callable[[int], Tuple[str, str, int]]]:
    """The pyrit adapter's ``label_fn(turn) -> (task, goal_id, turn)`` beside its ``seed_fn``:
    the goal is read off the driver's per-goal cursor at call time (``goal_ref[key]``), the
    turn is the forward index the seed rule uses. ``None`` when the hook is off."""
    if not enabled():
        return None
    task = store_task_name(LEG_MULTI_TURN, attack)
    return lambda turn: (task, str(goal_ref[key]), check_sample_idx(turn))


# --------------------------------------------------------------------------- #
# the client                                                                    #
# --------------------------------------------------------------------------- #

#: the contract's own case-insensitive header reader (public ``header_value`` once the promotion lands,
#: the private spelling until then) — ONE reader on both sides of the gate, never a third copy
_header: Callable[[Mapping[str, object], str], Optional[str]] = getattr(C, "header_value", None) or C._header_value


_HEX64 = re.compile(r"[0-9a-f]{64}")


def response_text_sha256(body: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """``(text_sha256, text_sha256_of)`` of one RECORDED response body (see :data:`RESPONSE_EVIDENCE_FIELDS`):
    ``("<sha256 of choices[0].message.content as UTF-8>", "content")`` when the body has choices whose first
    message carries a string ``content`` — the ``completion_text`` / ``ac.text_sha256`` reading, byte for byte;
    ``("<sha256 of canonical JSON of the whole body>", "body")`` otherwise (no single text field). Never ``(None,
    None)`` for a mapping: a recorded body always hashes to something a replay can be held to."""
    if not isinstance(body, Mapping):
        raise RequestEvidenceError(f"response_text_sha256 needs the parsed body mapping, got {type(body).__name__}")
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        msg = choices[0].get("message")
        content = msg.get("content") if isinstance(msg, Mapping) else None
        if isinstance(content, str):
            return hashlib.sha256(content.encode("utf-8")).hexdigest(), TEXT_SHA256_OF_CONTENT
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), TEXT_SHA256_OF_BODY


def check_response_row(row: Any, *, where: str = "response_evidence row") -> Dict[str, Any]:
    """The STRICT reader of one ``response_evidence.jsonl`` row at :data:`RESPONSE_EVIDENCE_VERSION`: a mapping whose
    key set EQUALS :data:`RESPONSE_EVIDENCE_FIELDS` + ``extra`` (a version-1 row — no ``response_evidence_version``,
    no ``text_sha256`` — is refused by name, never read as "no text"); the version this module writes; ``request_id``
    a non-empty str; ``text_sha256_of`` one of ``content`` / ``body`` / None with ``text_sha256`` a 64-hex sha EXACTLY
    when it is not None; a row that names a ``completion_id`` (a body was seen) must carry the sha. Returns the row."""
    if not isinstance(row, Mapping):
        raise RequestEvidenceError(f"{where}: not a mapping ({type(row).__name__})")
    want = set(RESPONSE_EVIDENCE_FIELDS) | {EXTRA_FIELD}
    missing, unknown = sorted(want - set(row)), sorted(set(row) - want)
    if missing or unknown:
        raise RequestEvidenceError(
            f"{where}: key set is not response-evidence version {RESPONSE_EVIDENCE_VERSION}'s (missing {missing}, unknown "
            f"{unknown}) — a version-1 row carries no text_sha256 and is refused, never read as 'no text'")
    v = row["response_evidence_version"]
    if isinstance(v, bool) or v != RESPONSE_EVIDENCE_VERSION:
        raise RequestEvidenceError(f"{where}: response_evidence_version {v!r}, this reader is version {RESPONSE_EVIDENCE_VERSION}")
    rid = row["request_id"]
    if not isinstance(rid, str) or not rid:
        raise RequestEvidenceError(f"{where}: request_id {rid!r} is not a non-empty str")
    sha, of = row["text_sha256"], row["text_sha256_of"]
    if of not in (TEXT_SHA256_OF_CONTENT, TEXT_SHA256_OF_BODY, None):
        raise RequestEvidenceError(f"{where}: text_sha256_of {of!r} is not content / body / null")
    if (of is None) != (sha is None):
        raise RequestEvidenceError(f"{where}: text_sha256 {sha!r} and text_sha256_of {of!r} must be both present or both null")
    if sha is not None and (not isinstance(sha, str) or not _HEX64.fullmatch(sha)):
        raise RequestEvidenceError(f"{where}: text_sha256 {sha!r} is not a 64-hex sha256")
    if row["completion_id"] is not None and sha is None:
        raise RequestEvidenceError(f"{where}: completion_id {row['completion_id']!r} recorded without a text_sha256 — the body was seen")
    return dict(row)


def check_request_record(rec: Any, *, where: str = "request_evidence record") -> Dict[str, Any]:
    """The STRICT reader of one ``request_evidence.jsonl`` record of a LANE-ON run: a mapping with a non-empty
    ``request_id``, the version-3 fields ``body_sha256`` / ``body_bytes``, and :data:`REQUEST_BODY_B64_FIELD` — a
    valid base64 string whose decoded bytes hash to ``body_sha256`` and have length ``body_bytes``. A record WITHOUT
    ``body_b64`` (a pre-fix record, or another writer's — production's ``_RequestEvidence``) is refused by name, never
    read as "no bytes"; a retained body that does not hash to the record's digest is refused
    (``bodies_from_records`` holds the same). Returns the record."""
    if not isinstance(rec, Mapping):
        raise RequestEvidenceError(f"{where}: not a mapping ({type(rec).__name__})")
    rid = rec.get("request_id")
    if not isinstance(rid, str) or not rid:
        raise RequestEvidenceError(f"{where}: request_id {rid!r} is not a non-empty str")
    if REQUEST_BODY_B64_FIELD not in rec:
        raise RequestEvidenceError(
            f"{where}: record {rid!r} carries no {REQUEST_BODY_B64_FIELD} — the exact wire bytes were not retained (a pre-fix "
            "record, or a writer other than the lane client); refused, never read as 'no bytes'")
    b64 = rec[REQUEST_BODY_B64_FIELD]
    if not isinstance(b64, str):
        raise RequestEvidenceError(f"{where}: record {rid!r}: {REQUEST_BODY_B64_FIELD} is {type(b64).__name__}, not a base64 str")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError) as exc:   # binascii.Error is a ValueError
        raise RequestEvidenceError(f"{where}: record {rid!r}: {REQUEST_BODY_B64_FIELD} is not base64 ({exc})") from None
    sha, n = rec.get("body_sha256"), rec.get("body_bytes")
    if C.wire_body_sha256(raw) != sha:
        raise RequestEvidenceError(
            f"{where}: record {rid!r}: the retained body hashes to {C.wire_body_sha256(raw)} but body_sha256 says {sha!r} — "
            "not the wire bytes")
    if isinstance(n, bool) or not isinstance(n, int) or n != len(raw):
        raise RequestEvidenceError(f"{where}: record {rid!r}: the retained body is {len(raw)} bytes but body_bytes says {n!r}")
    return dict(rec)


def read_request_evidence(path: "Path | str") -> List[Dict[str, Any]]:
    """Every record of a lane-on ``request_evidence.jsonl`` through :func:`check_request_record` (a line that is not
    JSON refuses, naming it)."""
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RequestEvidenceError(f"{path}:{i + 1}: not JSON ({exc})") from None
        out.append(check_request_record(rec, where=f"{path}:{i + 1}"))
    return out


def read_response_evidence(path: "Path | str") -> List[Dict[str, Any]]:
    """Every row of a ``response_evidence.jsonl`` through :func:`check_response_row` (an unreadable line refuses)."""
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RequestEvidenceError(f"{path}:{i + 1}: not JSON ({exc})") from None
        out.append(check_response_row(row, where=f"{path}:{i + 1}"))
    return out


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _status_of_error(exc: BaseException) -> Optional[int]:
    """The HTTP status a transport exception names: ``HttpError.status`` (the misalign
    runner), ``ProxyError('… returned 4xx: …')`` (the proxy client), else None."""
    st = getattr(exc, "status", None)
    if isinstance(st, int) and not isinstance(st, bool):
        return st
    m = re.search(r"\breturned (\d{3})\b", str(exc))
    return int(m.group(1)) if m else None


def _json_or_none(raw: bytes) -> Any:
    try:
        return json.loads(bytes(raw).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


class RequestEvidenceClient:
    """The recorder: two append-only, fsynced JSONL files under ``out_dir`` and the
    per-process ``{completion_id: request_id}`` binding table the row writers read.
    ``extra`` is stamped on every record / response row under :data:`EXTRA_FIELD`.
    Thread-safe (the misalign runner sends from a thread pool, the pyrit adapter from
    ``to_thread`` workers)."""

    def __init__(self, out_dir: "Path | str", *, extra: Optional[Mapping[str, Any]] = None,
                 seed_base: Optional[int] = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = self.out_dir / C.REQUEST_EVIDENCE_FILENAME
        self.response_path = self.out_dir / RESPONSE_EVIDENCE_FILENAME
        self.extra: Dict[str, Any] = dict(extra or {})
        # the seed base is REQUIRED on every record (extra.s): the audit partitions the Arditi engine's one task
        # by it and refuses a task record without it; given twice, the two must agree
        in_extra = self.extra.get(EXTRA_SEED_BASE_KEY)
        s = seed_base if seed_base is not None else in_extra
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise RequestEvidenceError(
                f"request evidence needs the run's seed base as a non-negative int (extra.{EXTRA_SEED_BASE_KEY}, "
                f"or {ENV_SEED_BASE}); got {s!r} — the audit refuses an Arditi task record without extra.s")
        if in_extra is not None and seed_base is not None and int(in_extra) != int(seed_base):
            raise RequestEvidenceError(f"seed base given twice and disagreeing: extra.{EXTRA_SEED_BASE_KEY}={in_extra!r} vs {seed_base!r}")
        self.extra[EXTRA_SEED_BASE_KEY] = int(s)
        self.seed_base = int(s)
        # the attempt id, when the extra names one: stamped top-level on every record too (EXTRA_ATTEMPT_ID_KEY)
        aid = self.extra.get(EXTRA_ATTEMPT_ID_KEY)
        self.attempt_id: Optional[str] = str(aid) if aid is not None and str(aid) else None
        self._lock = threading.Lock()
        self._bindings: Dict[str, str] = {}
        self._probe_counts: Dict[Tuple[str, str], int] = {}
        #: every request id this client has recorded an attempt under: a caller that presents one of them again
        #: (a hook re-minting one draw's id for its clamped retry) is refused — one attempt, one id, one ledger row
        self._seen_ids: Set[str] = set()
        self.n_requests = 0
        self.n_responses = 0

    # -- labels + the per-attempt id ------------------------------------------------ #

    def headers_for(self, body: Any, *, headers: Optional[Mapping[str, Any]] = None,
                    unlabelled_kind: Optional[str] = None) -> Dict[str, str]:
        """The headers ONE HTTP attempt goes out with: the caller's (kept verbatim), the
        two labels — the caller's if it set them, else the current :func:`labelled`
        context's, else a ``probe/<unlabelled_kind>`` + ``<vmodel>/<n>`` label when the
        call site declared its unlabelled sends are probes, else a REFUSAL — and a fresh
        ``X-Tinycap-Request-Id`` unless the caller minted one (a retry that re-enters here
        gets a new id: a retry is a new attempt)."""
        out: Dict[str, str] = {str(k): str(v) for k, v in (headers or {}).items()}
        has_task, has_sample = _header(out, C.HEADER_TASK), _header(out, C.HEADER_SAMPLE)
        if has_task is None or has_sample is None:
            if has_task is not None or has_sample is not None:
                raise RequestEvidenceError(
                    f"a request carried one tinycap label but not the other (task {has_task!r}, sample "
                    f"{has_sample!r}) — the two travel together")
            lab = _LABEL.get()
            if lab is not None:
                out.update(label_headers(*lab))
            elif unlabelled_kind is not None:
                vmodel = str(body.get("model")) if isinstance(body, Mapping) and body.get("model") else "?"
                with self._lock:
                    n = self._probe_counts.get((unlabelled_kind, vmodel), 0)
                    self._probe_counts[(unlabelled_kind, vmodel)] = n + 1
                out[C.HEADER_TASK] = C.probe_task_label(unlabelled_kind)
                out[C.HEADER_SAMPLE] = f"{vmodel}/{n}"
            else:
                raise RequestEvidenceError(
                    "an unlabelled target request under request evidence: no X-Tinycap-Task / X-Tinycap-Sample "
                    "on the call and no `labelled(task, item_id, idx)` context — the driver that knows the item "
                    "must label the send (a probe call site declares unlabelled_kind instead)")
        rid = _header(out, C.HEADER_REQUEST_ID)
        if rid is None:
            out[C.HEADER_REQUEST_ID] = uuid.uuid4().hex
        else:
            with self._lock:
                seen = rid in self._seen_ids
            if seen:
                raise RequestEvidenceError(
                    f"request id {rid!r} was already sent on an earlier attempt — a retry is a NEW attempt and mints a fresh "
                    "X-Tinycap-Request-Id (one id on two ledger rows cannot be reconciled); the hook re-minted it")
        return out

    # -- the two records ------------------------------------------------------------- #

    def _append(self, path: Path, row: Mapping[str, Any]) -> None:
        line = json.dumps(row, sort_keys=True) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def record_request(self, raw: bytes, headers: Mapping[str, Any]) -> Dict[str, Any]:
        """The version-3 record of one attempt from the EXACT bytes on the wire + its headers
        (:func:`contracts.request_evidence_record`), plus the bytes themselves under
        :data:`REQUEST_BODY_B64_FIELD`, plus ``extra`` (always: it carries ``s``); appended + fsynced."""
        raw = bytes(raw)
        rec: Dict[str, Any] = dict(C.request_evidence_record(raw, headers=headers))
        # the exact wire bytes beside their sha and length (REQUEST_BODY_B64_FIELD): the replay re-sends them.
        # The writer holds the three to each other BEFORE the record exists on disk — the bytes it retains ARE the bytes
        # the digest and the length describe (never a record whose retained body a reader must distrust)
        rec[REQUEST_BODY_B64_FIELD] = base64.b64encode(raw).decode("ascii")
        rec[EXTRA_FIELD] = dict(self.extra)
        if self.attempt_id is not None:
            rec[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] = self.attempt_id
        # ONE rule, the reader's (`247.0 == 247` passed an ad-hoc `!=` here and then failed the reader) —
        # the completed record goes through check_request_record before it is appended: the retained bytes must hash
        # to body_sha256 and body_bytes must be an int (not a bool) equal to their length, on the original types
        check_request_record(rec, where=f"record_request({rec.get('request_id')!r}): refusing to write a record that misdescribes its own body;")
        with self._lock:
            if rec["request_id"] in self._seen_ids:
                raise RequestEvidenceError(
                    f"request id {rec['request_id']!r} is being recorded a second time — two attempts under one id")
            self._seen_ids.add(str(rec["request_id"]))
        self._append(self.request_path, rec)
        with self._lock:
            self.n_requests += 1
        return rec

    def record_response(self, request_id: str, *, status: Optional[int], resp: Optional[Mapping[str, Any]] = None,
                        error: Optional[str] = None, headers: Optional[Mapping[str, Any]] = None
                        ) -> Dict[str, Any]:
        """The response side of one attempt (version :data:`RESPONSE_EVIDENCE_VERSION`): ``(request_id,
        completion_id)`` — the engine binding — with the status, the body's ``model`` / first ``finish_reason``,
        the response TEXT's sha256 + what it hashes (:func:`response_text_sha256`), the labels, and the error text
        of a failed attempt. Registers the binding for :meth:`binding_of` and as this context's last binding."""
        if not isinstance(request_id, str) or not request_id:
            raise RequestEvidenceError(f"record_response needs the attempt's request id, got {request_id!r}")
        cid: Optional[str] = None
        vmodel: Optional[str] = None
        finish: Optional[str] = None
        sha: Optional[str] = None
        sha_of: Optional[str] = None
        if isinstance(resp, Mapping):
            cid = resp.get("id") if isinstance(resp.get("id"), str) and resp.get("id") else None
            vmodel = resp.get("model") if isinstance(resp.get("model"), str) else None
            choices = resp.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                fr = choices[0].get("finish_reason")
                finish = str(fr) if fr is not None else None
            sha, sha_of = response_text_sha256(resp)
        hdrs = dict(headers or {})
        row: Dict[str, Any] = {
            "response_evidence_version": RESPONSE_EVIDENCE_VERSION,
            "ts": _utc_now(), "request_id": request_id, "completion_id": cid,
            "status": (int(status) if isinstance(status, int) and not isinstance(status, bool) else None),
            "vmodel": vmodel, "finish_reason": finish,
            "text_sha256": sha, "text_sha256_of": sha_of,
            "task": _header(hdrs, C.HEADER_TASK), "sample": _header(hdrs, C.HEADER_SAMPLE),
            "error": (str(error)[:2000] if error is not None else None),
        }
        assert tuple(row) == RESPONSE_EVIDENCE_FIELDS
        row[EXTRA_FIELD] = dict(self.extra)
        self._append(self.response_path, row)
        with self._lock:
            self.n_responses += 1
            if cid is not None:
                self._bindings[cid] = request_id
        binding = {ROW_REQUEST_ID_FIELD: request_id, ROW_COMPLETION_ID_FIELD: cid}
        _LAST_BINDING.set(binding)
        return row

    #: the sink signature the Inspect provider's response hook calls
    #: (``response_evidence_sink(request_id, status=, resp=, error=, headers=)``)
    response_sink = record_response

    # -- what the row writers read ---------------------------------------------------- #

    def binding_of(self, resp: Any) -> Optional[Dict[str, Optional[str]]]:
        """``{request_id, completion_id}`` for a response dict this client observed (by
        its ``id``); ``None`` for a response it did not."""
        cid = resp.get("id") if isinstance(resp, Mapping) else None
        if not isinstance(cid, str) or not cid:
            return None
        with self._lock:
            rid = self._bindings.get(cid)
        return None if rid is None else {ROW_REQUEST_ID_FIELD: rid, ROW_COMPLETION_ID_FIELD: cid}

    def row_fields(self, resp: Any) -> Dict[str, Optional[str]]:
        """:meth:`binding_of` as row fields, ``{}`` when unknown (so ``row.update(...)`` is safe)."""
        return self.binding_of(resp) or {}

    def bindings(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._bindings)

    # -- the Inspect provider's request sink (review fix P1-1) --------------------------- #

    def adopt_attempt_id(self, attempt_id: Optional[str]) -> None:
        """Take the run's attempt id from the driver's own writer when this client was built without one
        (``extra.attempt_id`` absent); a client that already names one REFUSES a different id — one run, one
        attempt, and the two stamps (production's top-level field, the lane's ``extra``) must agree."""
        if attempt_id is None or not str(attempt_id):
            return
        if self.attempt_id is None:
            self.attempt_id = str(attempt_id)
            self.extra[EXTRA_ATTEMPT_ID_KEY] = str(attempt_id)
            return
        if self.attempt_id != str(attempt_id):
            raise RequestEvidenceError(
                f"the driver's request-evidence writer runs as attempt {attempt_id!r} but this client's extra names "
                f"{self.attempt_id!r} ({ENV_EXTRA}) — one run has one attempt id; refusing to record under two")

    def take_over_request_sink(self, sink: Optional[Callable[..., Any]]) -> Callable[..., Any]:
        """The REQUEST sink the Inspect provider records through while this client is on: :meth:`record_request`, in
        place of the caller's ``sink`` — production's ``_RequestEvidence.record_wire``, which writes the v3 record +
        ``attempt_id`` and never ``extra.s`` (``audit_root`` refuses every Arditi task record without it).
        The caller's sink is NOT called for the provider's requests (one attempt, one record). When it is a writer
        with a ``path``, that path must be THIS client's ``request_evidence.jsonl`` (one inventory in one file —
        ``find_one`` refuses an attempt root with two), and its ``attempt_id`` is adopted or held to
        (:meth:`adopt_attempt_id`), so production's ``--attempt-id`` scoping sees every record stamped."""
        if sink is None:
            return self.record_request
        owner = getattr(sink, "__self__", None)
        if owner is None or owner is self:
            return self.record_request
        path = getattr(owner, "path", None)
        if isinstance(path, (str, Path)) and Path(path).resolve() != self.request_path.resolve():
            raise RequestEvidenceError(
                f"the driver's request-evidence writer appends to {path} but {ENV_DIR} names {self.request_path} — "
                "two request_evidence.jsonl under one attempt root is an ambiguous root (WS3-G's find_one refuses "
                f"it); point {ENV_DIR} at the driver's out root")
        self.adopt_attempt_id(getattr(owner, "attempt_id", None))
        return self.record_request

    # -- transports -------------------------------------------------------------------- #

    def attach_proxy_client(self, client: Any, *, unlabelled_kind: Optional[str] = None) -> Any:
        """Observe and label a sync ``ProxyClientEngine``: its ``on_request`` hook records
        every ``/chat/completions`` POST from the EXACT bytes ``urllib`` sends; its
        ``chat_completions`` is replaced by a wrapper that mints the attempt's headers
        (:meth:`headers_for`), calls through, and records the response (a ``ProxyError``
        is recorded with its status, then re-raised). Returns the same client."""
        if getattr(client, "_sisteer_request_evidence", None) is self:
            return client
        orig = client.chat_completions
        prev_on_request = getattr(client, "on_request", None)
        rec = self

        def on_request(url: str, data: bytes, headers: Mapping[str, Any]) -> None:
            if str(url).rstrip("/").endswith(_CHAT_SUFFIX) and _header(headers, C.HEADER_REQUEST_ID):
                rec.record_request(data, headers)
            if prev_on_request is not None:
                prev_on_request(url, data, headers)

        def chat_completions(payload: Mapping[str, Any], *, headers: Optional[Mapping[str, Any]] = None
                             ) -> Dict[str, Any]:
            hdrs = rec.headers_for(payload, headers=headers, unlabelled_kind=unlabelled_kind)
            rid = hdrs[C.HEADER_REQUEST_ID]
            try:
                resp = orig(payload, headers=hdrs)
            except Exception as exc:  # noqa: BLE001 — recorded, then re-raised unchanged
                rec.record_response(rid, status=_status_of_error(exc), error=f"{type(exc).__name__}: {exc}",
                                    headers=hdrs)
                raise
            rec.record_response(rid, status=200, resp=resp if isinstance(resp, Mapping) else None, headers=hdrs)
            return resp

        client.on_request = on_request
        client.chat_completions = chat_completions
        client._sisteer_request_evidence = self
        return client

    def httpx_hooks(self, *, on_binding: Optional[Callable[[Mapping[str, Any]], None]] = None
                    ) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
        """``(request_hook, response_hook)`` for an ``httpx.AsyncClient``'s ``event_hooks``:
        the request hook mints the labels + id onto a ``/chat/completions`` POST that carries
        none (the Petri target's ``openai/`` provider) and records the record from the exact
        ``request.content``; the response hook reads the response body (cached for the SDK),
        records ``(request_id, completion_id)`` and sets this context's last binding, then hands
        the response row to ``on_binding`` when given — the durable home of the binding beyond the
        two files (the Petri sample store, :func:`petri_sample_turn_sink`)."""
        rec = self

        def _is_chat(request: Any) -> bool:
            return (str(getattr(request, "method", "")).upper() == "POST"
                    and str(request.url.path).endswith(_CHAT_SUFFIX))

        async def request_hook(request: Any) -> None:
            if not _is_chat(request):
                return
            body = _json_or_none(bytes(request.content))
            minted = rec.headers_for(body, headers=dict(request.headers.items()))
            for name in (C.HEADER_TASK, C.HEADER_SAMPLE, C.HEADER_REQUEST_ID):
                if _header(dict(request.headers.items()), name) is None:
                    request.headers[name] = minted[name]
            rec.record_request(bytes(request.content), headers=dict(request.headers.items()))

        async def response_hook(response: Any) -> None:
            request = response.request
            if not _is_chat(request):
                return
            hdrs = dict(request.headers.items())
            rid = _header(hdrs, C.HEADER_REQUEST_ID)
            if rid is None:
                return   # not an attempt this evidence labels
            await response.aread()
            body = _json_or_none(bytes(response.content))
            ok = 200 <= int(response.status_code) < 300
            row = rec.record_response(rid, status=int(response.status_code),
                                      resp=body if (ok and isinstance(body, Mapping)) else None,
                                      error=(None if ok else bytes(response.content)[:2000].decode("utf-8", "replace")),
                                      headers=hdrs)
            if on_binding is not None:
                on_binding(row)

        return request_hook, response_hook

    def install_httpx(self, http_client: Any, *, request: bool = True, response: bool = True,
                      on_binding: Optional[Callable[[Mapping[str, Any]], None]] = None) -> None:
        """Put :meth:`httpx_hooks` on ``http_client.event_hooks`` once per client (idempotent;
        inspect_ai's own hooks are kept); ``on_binding`` rides the response hook."""
        marker = "_sisteer_request_evidence_hooks"
        installed = getattr(http_client, marker, None)
        if installed is self:
            return
        req_hook, resp_hook = self.httpx_hooks(on_binding=on_binding)
        hooks = {k: list(v) for k, v in dict(http_client.event_hooks).items()}
        if request:
            hooks.setdefault("request", []).append(req_hook)
        if response:
            hooks.setdefault("response", []).append(resp_hook)
        http_client.event_hooks = hooks
        try:
            setattr(http_client, marker, self)
        except Exception:  # noqa: BLE001 — a client that refuses attributes still has the hooks
            pass

    def instrument_inspect_target(self, model: Any, *, task: str) -> Any:
        """A per-TASK copy of an inspect_ai ``Model`` (the Petri target) whose every
        ``api.generate`` runs under the label ``(task, <active sample id>, <turn>)`` — the
        turn is :func:`petri_turn_index` of the request transcript (1 + its assistant
        messages, a staged prefill counted: ``petri_runner.target_turn_index``, the
        index the per-scenario-turn seed rule reads, so the header and the seed closure agree
        by construction) — over the shared ``http_client`` carrying :meth:`httpx_hooks`. The
        copy shares the transport; the original is untouched (inspect's own copy-then-configure
        role-model idiom). When the runner carries ``TARGET_TURN_HEADERS_HOOK`` the
        labels ALSO ride the request's ``extra_headers`` (:func:`petri_turn_headers`); the two
        spell the same values and :meth:`headers_for` keeps a label already on the request."""
        api = copy.copy(model.api)
        http_client = getattr(api, "http_client", None)
        if http_client is None:
            raise RequestEvidenceError(
                f"the target model's api ({type(api).__name__}) exposes no http_client — request evidence "
                "needs the httpx boundary to record the wire bytes")
        self.install_httpx(http_client)
        inner = api.generate

        async def generate(input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:  # noqa: A002
            sid, _epoch = _active_sample_identity()
            if sid is None:
                raise RequestEvidenceError(
                    f"a target generate for {task!r} ran outside an Inspect sample — the label needs the "
                    "scenario id (inspect_ai.log._samples.sample_active)")
            with labelled(task, sid, petri_turn_index(input)):
                return await inner(input=input, tools=tools, tool_choice=tool_choice, config=config)

        api.generate = generate
        out = copy.copy(model)
        out.api = api
        return out

    def aiohttp_trace_config(self) -> Any:
        """An ``aiohttp.TraceConfig`` for a ``ClientSession(trace_configs=[…])``: on
        ``/chat/completions`` POSTs it mints the labels (the caller's ``headers=`` labels, else
        the :func:`labelled` context) + the id onto the outgoing headers at
        ``on_request_start`` (the very ``CIMultiDict`` the request is built from), collects the
        EXACT bytes from ``on_request_chunk_sent``, and at ``on_request_end`` /
        ``on_request_exception`` writes the request record and the response row."""
        import aiohttp

        rec = self
        tc = aiohttp.TraceConfig()

        def _is_chat(params: Any) -> bool:
            return (str(params.method).upper() == "POST"
                    and str(params.url.path).endswith(_CHAT_SUFFIX))

        async def on_start(session: Any, ctx: Any, params: Any) -> None:
            ctx.sisteer_active = _is_chat(params)
            if not ctx.sisteer_active:
                return
            hdrs = params.headers
            minted = rec.headers_for(None, headers=dict(hdrs.items()))
            for name in (C.HEADER_TASK, C.HEADER_SAMPLE, C.HEADER_REQUEST_ID):
                if _header(dict(hdrs.items()), name) is None:
                    hdrs[name] = minted[name]
            ctx.sisteer_headers = dict(hdrs.items())
            ctx.sisteer_chunks = []
            ctx.sisteer_recorded = False

        async def on_chunk(session: Any, ctx: Any, params: Any) -> None:
            if getattr(ctx, "sisteer_active", False):
                ctx.sisteer_chunks.append(bytes(params.chunk))

        def _record_request(ctx: Any) -> None:
            if not ctx.sisteer_recorded:
                rec.record_request(b"".join(ctx.sisteer_chunks), ctx.sisteer_headers)
                ctx.sisteer_recorded = True

        async def on_end(session: Any, ctx: Any, params: Any) -> None:
            if not getattr(ctx, "sisteer_active", False):
                return
            _record_request(ctx)
            resp = params.response
            raw = await resp.read()
            body = _json_or_none(raw)
            ok = 200 <= int(resp.status) < 300
            rec.record_response(_header(ctx.sisteer_headers, C.HEADER_REQUEST_ID) or "", status=int(resp.status),
                                resp=body if (ok and isinstance(body, Mapping)) else None,
                                error=(None if ok else raw[:2000].decode("utf-8", "replace")),
                                headers=ctx.sisteer_headers)

        async def on_exc(session: Any, ctx: Any, params: Any) -> None:
            if not getattr(ctx, "sisteer_active", False):
                return
            if ctx.sisteer_chunks:
                _record_request(ctx)
            rec.record_response(_header(ctx.sisteer_headers, C.HEADER_REQUEST_ID) or "", status=None,
                                error=f"{type(params.exception).__name__}: {params.exception}",
                                headers=ctx.sisteer_headers)

        tc.on_request_start.append(on_start)
        tc.on_request_chunk_sent.append(on_chunk)
        tc.on_request_end.append(on_end)
        tc.on_request_exception.append(on_exc)
        return tc

    def wrap_post_json(self, http: Any) -> Any:
        """The misalign runner's transport (``HttpxJson``: ``get_json`` / ``post_json`` / ``close``)
        wrapped so every ``/chat/completions`` ``post_json`` is ONE recorded attempt: the labels from the
        caller's ``headers`` (the ``request_evidence.headers_for`` seam) or the :func:`labelled`
        context, a FRESH id per HTTP attempt (``_post_with_retries`` re-enters here on a 5xx: a new
        attempt, never one id on two ledger rows), the record from the bytes the inner transport
        serialises (``json.dumps(dict(payload)).encode("utf-8")`` — ``HttpxJson.post_json``'s own
        expression; the ledger's ``body_sha256`` check refuses any divergence), the response row from what
        it returns or raises. The inner call: ``post_json(url, payload, headers=)`` when the transport
        accepts ``headers`` (the runner), else — a transport predating it — the same ``httpx`` post
        through its own ``_client`` / ``_wrap`` with the headers added."""
        import inspect as _inspect

        rec = self
        try:
            accepts_headers = "headers" in _inspect.signature(http.post_json).parameters
        except (TypeError, ValueError):
            accepts_headers = False

        class _Labelled:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def get_json(self, url: str) -> Any:
                return self._inner.get_json(url)

            def close(self) -> None:
                return self._inner.close()

            def _send(self, url: str, payload: Mapping[str, Any], raw: bytes, hdrs: Mapping[str, str]) -> Any:
                if accepts_headers:
                    return self._inner.post_json(url, payload, headers=dict(hdrs))
                client, wrap = getattr(self._inner, "_client", None), getattr(self._inner, "_wrap", None)
                if client is None or wrap is None:
                    raise RequestEvidenceError(
                        f"the misalign transport {type(self._inner).__name__} accepts no request headers and exposes no "
                        "httpx client to send them through — the labelled request cannot be sent")
                httpx_mod = getattr(self._inner, "_httpx", None)
                try:
                    return wrap(client.post(url, content=raw, headers={"Content-Type": "application/json", **dict(hdrs)}))
                except Exception as e:  # noqa: BLE001 — the transport's own error type, as HttpxJson.post_json maps it
                    if httpx_mod is not None and isinstance(e, httpx_mod.HTTPError):
                        from sisteer.capeval.misalign.runner import HttpError
                        raise HttpError(f"POST {url}: {type(e).__name__}: {e}", status=None) from e
                    raise

            def post_json(self, url: str, payload: Mapping[str, Any], *, headers: Optional[Mapping[str, str]] = None
                          ) -> Dict[str, Any]:
                if not str(url).rstrip("/").endswith(_CHAT_SUFFIX):
                    return (self._inner.post_json(url, payload, headers=dict(headers)) if (headers and accepts_headers)
                            else self._inner.post_json(url, payload))
                hdrs = rec.headers_for(payload, headers=headers)
                raw = json.dumps(dict(payload)).encode("utf-8")
                rec.record_request(raw, hdrs)
                rid = hdrs[C.HEADER_REQUEST_ID]
                try:
                    resp = self._send(url, payload, raw, hdrs)
                except Exception as exc:  # noqa: BLE001 — recorded, then re-raised unchanged
                    rec.record_response(rid, status=_status_of_error(exc), error=f"{type(exc).__name__}: {exc}",
                                        headers=hdrs)
                    raise
                rec.record_response(rid, status=200, resp=resp if isinstance(resp, Mapping) else None,
                                    headers=hdrs)
                return resp

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        return _Labelled(http)


def petri_turn_index(messages: Iterable[Any]) -> int:
    """The Petri TARGET turn of a request transcript: ``1 + the assistant messages in it`` (a
    staged prefill counts) — the ``petri_runner.target_turn_index`` definition, the index
    ``petri_target_seed(s, behaviour, scenario_id, turn)`` reads, restated here so the sample
    header's numeric index and the seed closure name the same turn. Messages are inspect_ai
    ``ChatMessage`` objects (``.role``) or mappings (``["role"]``)."""
    n = 0
    for m in messages or ():
        role = m.get("role") if isinstance(m, Mapping) else getattr(m, "role", None)
        if role == "assistant":
            n += 1
    return 1 + n


def petri_turn_headers(turn: Any) -> Dict[str, str]:
    """The ``petri_runner.TARGET_TURN_HEADERS_HOOK`` body: the two labels of one target
    turn from its ``TargetTurn`` (``task`` = ``petri/<behaviour>``, ``item_id`` = the scenario id,
    ``sample_idx`` = the turn) — :func:`label_headers`, nothing else (the per-attempt id is minted
    at the HTTP boundary, :meth:`RequestEvidenceClient.httpx_hooks`)."""
    return label_headers(str(turn.task), str(turn.item_id), turn.sample_idx)   # validated there, never coerced


def _inspect_sample_state() -> Any:
    """Inspect's ACTIVE ``TaskState`` (``inspect_ai.solver.sample_state``) — whose ``store`` the eval log persists
    as ``sample.store`` — or ``None`` outside a sample (or without inspect_ai)."""
    try:
        from inspect_ai.solver._task_state import sample_state
    except Exception:  # noqa: BLE001 — no inspect_ai: no sample
        return None
    return sample_state()


def _is_petri_task(task: Any) -> bool:
    try:
        from sisteer.capeval.tiers.petri_runner import PETRI_TASK_PREFIX as prefix
    except Exception:  # noqa: BLE001 — the runner is the caller; its spelling is the authority when importable
        prefix = "petri/"
    return isinstance(task, str) and task.startswith(prefix)


def petri_sample_turn_sink(row: Mapping[str, Any]) -> None:
    """The ``on_binding`` sink :func:`maybe_install_petri_target_hooks` installs: one recorded
    TARGET attempt — the response row :meth:`RequestEvidenceClient.record_response` wrote — filed on the ACTIVE
    Inspect sample's store under :data:`PETRI_TARGET_TURNS_STORE_KEY`, so the binding outlives the process inside
    the eval log (``sample.store``) and ``petri_runner.convert_eval_log`` can put it on every row of the scenario.
    The entry keeps the TURN the request's sample label named (``<scenario_id>/<turn>`` — the ``TargetTurn``
    through ``TARGET_TURN_HEADERS_HOOK``), the attempt's id, the completion id (None on a failed attempt) and the
    status: a retry is a second entry at the same turn, a rollback a later entry at an earlier turn — nothing is
    collapsed. A Petri-labelled attempt outside a sample, or one whose label names another scenario than the
    active sample, is REFUSED (a binding with no home, or the wrong one, is never dropped quietly). Rows of any
    other task: a no-op (the hook is generic)."""
    task = row.get("task")
    if not _is_petri_task(task):
        return
    sample = row.get("sample")
    item_id, _sep, idx = str(sample).rpartition("/") if isinstance(sample, str) else ("", "", "")
    if not item_id or not idx.isdigit():
        raise RequestEvidenceError(
            f"a Petri target attempt (task {task!r}) carries the sample label {sample!r}, not <scenario_id>/<turn> — "
            "its turn cannot be filed")
    state = _inspect_sample_state()
    if state is None:
        raise RequestEvidenceError(
            f"a Petri target attempt ({task} {sample}, request {row.get('request_id')!r}) was answered outside an "
            "Inspect sample — its binding has no durable home (the eval log's sample store); refusing to drop it")
    active = getattr(state, "sample_id", None)
    if active is None or str(active) != item_id:
        raise RequestEvidenceError(
            f"a Petri target attempt is labelled scenario {item_id!r} but the active Inspect sample is {active!r} — "
            "refusing to file the binding on another scenario's record")
    entry = {"turn": int(idx), ROW_REQUEST_ID_FIELD: row.get("request_id"),
             ROW_COMPLETION_ID_FIELD: row.get("completion_id"), "status": row.get("status"), "task": task}
    turns = list(state.store.get(PETRI_TARGET_TURNS_STORE_KEY) or [])
    turns.append(entry)
    state.store.set(PETRI_TARGET_TURNS_STORE_KEY, turns)


_TARGET_TURN_KEYS = frozenset({"turn", ROW_REQUEST_ID_FIELD, ROW_COMPLETION_ID_FIELD, "status", "task"})


def _check_target_turn_entry(e: Any, *, where: str, task: str) -> Dict[str, Any]:
    """ONE target-turn entry validated on its ORIGINAL types (shared by the converter's reader
    and the audit's draws): a mapping whose key set EQUALS the sink's five (a missing key is not a recorded
    None); ``turn`` a non-bool int >= 1;
    ``request_id`` a non-empty str; ``status`` a non-bool int or None; ``completion_id`` a non-empty str EXACTLY
    when the status is 2xx — a successful attempt without its completion, or a failed / status-less attempt
    carrying one, is torn; ``task`` this row's. Nothing is coerced, nothing is skipped."""
    if not isinstance(e, Mapping):
        raise RequestEvidenceError(f"{where}: not a mapping ({type(e).__name__})")
    # EXACT key-set equality: a key that is MISSING is not a key recorded as None — an entry stripped of
    # `status` + `completion_id` would otherwise read as a failed attempt and lose its draw while every count still
    # balanced; both the missing and the unknown keys are named
    missing, unknown = sorted(_TARGET_TURN_KEYS - set(e)), sorted(set(e) - _TARGET_TURN_KEYS)
    if missing or unknown:
        raise RequestEvidenceError(
            f"{where}: key set {sorted(e)} is not the sink's {sorted(_TARGET_TURN_KEYS)} (missing {missing}, unknown "
            f"{unknown}) — a missing key is never read as a recorded None")
    turn, rid, cid = e["turn"], e[ROW_REQUEST_ID_FIELD], e[ROW_COMPLETION_ID_FIELD]
    status, etask = e["status"], e["task"]
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise RequestEvidenceError(f"{where}: turn {turn!r} ({type(turn).__name__}) is not a positive int")
    if not isinstance(rid, str) or not rid:
        raise RequestEvidenceError(f"{where}: request_id {rid!r} is not a non-empty str")
    if status is not None and (isinstance(status, bool) or not isinstance(status, int)):
        raise RequestEvidenceError(f"{where}: status {status!r} ({type(status).__name__}) is not an int or None")
    ok = status is not None and 200 <= status < 300
    if ok and (not isinstance(cid, str) or not cid):
        raise RequestEvidenceError(f"{where}: a {status} attempt without a completion_id ({cid!r}) — torn")
    if not ok and cid is not None:
        raise RequestEvidenceError(f"{where}: a failed attempt (status {status!r}) carrying completion_id {cid!r} — torn")
    if etask != task:
        raise RequestEvidenceError(
            f"{where}: filed under task {etask!r} but this row's task is {task!r} — another behaviour's binding")
    return {"turn": turn, ROW_REQUEST_ID_FIELD: rid, ROW_COMPLETION_ID_FIELD: cid, "status": status, "task": etask}


def _check_target_turn_list(raw: Any, *, where: str, task: str) -> List[Dict[str, Any]]:
    """The whole list: a list (not a str, not a mapping), every entry through :func:`_check_target_turn_entry`,
    request ids unique within it (an attempt is recorded once)."""
    if not isinstance(raw, list):
        raise RequestEvidenceError(f"{where}: {PETRI_TARGET_TURNS_ROW_KEY} is {type(raw).__name__}, not a list")
    out = [_check_target_turn_entry(e, where=f"{where}[{i}]", task=task) for i, e in enumerate(raw)]
    seen: Set[str] = set()
    for i, e in enumerate(out):
        if e[ROW_REQUEST_ID_FIELD] in seen:
            raise RequestEvidenceError(f"{where}[{i}]: request_id {e[ROW_REQUEST_ID_FIELD]!r} recorded twice in one list")
        seen.add(e[ROW_REQUEST_ID_FIELD])
    return out


def petri_target_turns_of_sample(sample: Mapping[str, Any], *, task: str, scenario_id: str
                                 ) -> Optional[List[Dict[str, Any]]]:
    """The TARGET-turn bindings an eval-log sample (dict form) carries under
    ``sample.store[PETRI_TARGET_TURNS_STORE_KEY]``, validated for the rows they will ride
    (:func:`_check_target_turn_list`: every entry names THIS row's ``task``, original types, completion iff 2xx,
    unique request ids). ``None`` when the sample carries none (the hook off, a pre-lane log): the converter then
    writes no key."""
    store = sample.get("store")
    raw = store.get(PETRI_TARGET_TURNS_STORE_KEY) if isinstance(store, Mapping) else None
    if raw is None:
        return None
    return _check_target_turn_list(raw, where=f"scenario {scenario_id!r} {PETRI_TARGET_TURNS_STORE_KEY}", task=task)


class PetriTurnDraws(NamedTuple):
    """:func:`petri_target_turn_draws`'s result: the audit's per-turn draws and the counts a consumer holds them to —
    ``len(draws) == n_successful`` by construction (assert it), ``n_failed`` the recorded attempts without a completion
    (a 5xx, a transport error; the ledger and the reconciliation account for them), ``n_scenarios`` the filed
    scenarios the rows named (every one contributed >= 1 draw, or the call refused)."""

    draws: List[Dict[str, Any]]
    n_successful: int
    n_failed: int
    n_scenarios: int


def _row_target_turns(r: Mapping[str, Any], *, where: str) -> Any:
    """A Petri store row's ``target_turns`` value from its ``extra_json`` (a canonical-JSON str, or a mapping) or
    ``extra`` mapping — REFUSING a row without a readable extra block or without the key (a row that binds to no
    attempt is never a zero-draw row: a pre-lane row, or the hook was off when it was filed)."""
    extra: Any = r.get("extra_json")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError as exc:
            raise RequestEvidenceError(f"{where}: extra_json is not JSON ({exc})") from None
    if not isinstance(extra, Mapping):
        extra = r.get("extra")
    if not isinstance(extra, Mapping):
        raise RequestEvidenceError(f"{where}: no readable extra_json / extra block — the row carries no target bindings")
    if PETRI_TARGET_TURNS_ROW_KEY not in extra:
        raise RequestEvidenceError(
            f"{where}: carries no {PETRI_TARGET_TURNS_ROW_KEY} — a row that binds to no attempt (a pre-lane row, or "
            "the request-evidence hook was off when it was filed); never a zero-draw scenario")
    return extra[PETRI_TARGET_TURNS_ROW_KEY]


def petri_target_turn_draws(rows: Iterable[Mapping[str, Any]]) -> PetriTurnDraws:
    """The audit's per-TURN draws of Petri store rows (``read_petri_store``: one ``Draw`` per successful
    ``(task, scenario, turn, request_id)``) — FAIL-CLOSED: every non-aggregate row must carry a
    valid ``extra_json.target_turns`` list (:func:`_check_target_turn_list` — original types, completion iff 2xx,
    unique request ids, this row's task); every dimension row of one scenario must carry the IDENTICAL ordered
    list (the converter writes one list per scenario onto each of its rows); every scenario must have >= 1
    successful attempt (a filed scenario whose target never answered is refused, not silently zero draws); a row
    without the key, a non-list value, a malformed entry, a divergent sibling — each REFUSES by name. Nothing is
    skipped except the per-behaviour aggregate rows (``contracts.is_aggregate_item``: the ``__aggregate__``
    means, which carry no attempts by construction). Returns :class:`PetriTurnDraws`; a consumer asserts
    ``len(result.draws) == result.n_successful`` and that ``n_scenarios`` equals the scenarios it filed."""
    rows = list(rows)
    if not rows:
        raise RequestEvidenceError("petri_target_turn_draws: no rows — nothing to draw from (an empty store is refused, not zero draws)")
    per_scenario: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    first_row_of: Dict[Tuple[str, str], int] = {}
    for i, r in enumerate(rows):
        task, item = r.get("task"), r.get("item_id")
        if not isinstance(task, str) or not task or not isinstance(item, str) or not item:
            raise RequestEvidenceError(f"row {i}: task {task!r} / item_id {item!r} are not non-empty strings")
        if C.is_aggregate_item(item):
            continue   # the per-behaviour mean row: no attempts by construction, never a scenario
        where = f"row {i} ({task} {item} {r.get('metric')!r})"
        turns = _check_target_turn_list(_row_target_turns(r, where=where), where=f"{where} {PETRI_TARGET_TURNS_ROW_KEY}", task=task)
        key = (task, item)
        prev = per_scenario.get(key)
        if prev is None:
            per_scenario[key] = turns
            first_row_of[key] = i
        elif prev != turns:
            raise RequestEvidenceError(
                f"{where}: its {PETRI_TARGET_TURNS_ROW_KEY} differs from row {first_row_of[key]}'s for the same scenario — the "
                "converter writes ONE list per scenario onto every dimension row; a divergent copy is a torn store")
    if not per_scenario:
        raise RequestEvidenceError("petri_target_turn_draws: only aggregate rows — no scenario row to draw from")
    draws: List[Dict[str, Any]] = []
    n_failed = 0
    for (task, item), turns in per_scenario.items():
        ok = [e for e in turns if e["status"] is not None and 200 <= e["status"] < 300]
        n_failed += len(turns) - len(ok)
        if not ok:
            raise RequestEvidenceError(
                f"scenario {item!r} ({task}): {len(turns)} target attempt(s) recorded and none successful — a filed scenario "
                "whose target never answered; refusing to emit zero draws for it")
        for e in ok:
            draws.append({"task": task, "item_id": item, "sample_idx": e["turn"],
                          ROW_REQUEST_ID_FIELD: e[ROW_REQUEST_ID_FIELD], ROW_COMPLETION_ID_FIELD: e[ROW_COMPLETION_ID_FIELD]})
    return PetriTurnDraws(draws=draws, n_successful=len(draws), n_failed=n_failed, n_scenarios=len(per_scenario))


def _active_sample_identity() -> Tuple[Optional[str], int]:
    """``(sample id, epoch)`` of Inspect's ACTIVE sample — the authoritative record, never
    a reconstruction — or ``(None, 1)`` outside one."""
    try:
        from inspect_ai.log._samples import sample_active
    except Exception:  # noqa: BLE001 — no inspect_ai: no active sample
        return None, 1
    active = sample_active()
    if active is None or getattr(active, "sample", None) is None:
        return None, 1
    sid = getattr(active.sample, "id", None)
    epoch = getattr(active, "epoch", 1)
    return (None if sid is None else str(sid)), (int(epoch) if isinstance(epoch, int) else 1)


# --------------------------------------------------------------------------- #
# process-level enablement: the ENV switch, then the guarded entry points       #
# --------------------------------------------------------------------------- #

_ENV_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], RequestEvidenceClient] = {}
_ENV_LOCK = threading.Lock()


def from_env(env: Optional[Mapping[str, str]] = None) -> Optional[RequestEvidenceClient]:
    """The process's client when :data:`ENV_DIR` is set (one instance per (dir, extra) —
    the binding table must be shared by the transport and the row writers), else ``None``."""
    env = os.environ if env is None else env
    out_dir = env.get(ENV_DIR)
    if not out_dir:
        return None
    extra_raw = env.get(ENV_EXTRA) or None
    seed_raw = env.get(ENV_SEED_BASE) or None
    key = (out_dir, extra_raw, seed_raw)
    with _ENV_LOCK:
        client = _ENV_CACHE.get(key)
        if client is None:
            extra: Dict[str, Any] = {}
            if extra_raw:
                parsed = json.loads(extra_raw)
                if not isinstance(parsed, Mapping):
                    raise RequestEvidenceError(f"{ENV_EXTRA} must be a JSON object, got {parsed!r}")
                extra = dict(parsed)
            seed_base: Optional[int] = None
            if seed_raw is not None:
                if not re.fullmatch(r"[0-9]+", seed_raw):
                    raise RequestEvidenceError(f"{ENV_SEED_BASE} must be a non-negative integer, got {seed_raw!r}")
                seed_base = int(seed_raw)
            client = RequestEvidenceClient(out_dir, extra=extra, seed_base=seed_base)
            _ENV_CACHE[key] = client
    return client


def enabled() -> bool:
    return bool(os.environ.get(ENV_DIR))


def reset_env_cache() -> None:
    """Tests: forget the cached per-env clients."""
    with _ENV_LOCK:
        _ENV_CACHE.clear()


def maybe_attach_proxy_client(client: Any, *, unlabelled_kind: Optional[str] = None) -> Any:
    rec = from_env()
    return client if rec is None else rec.attach_proxy_client(client, unlabelled_kind=unlabelled_kind)


def maybe_wrap_post_json(http: Any) -> Any:
    rec = from_env()
    return http if rec is None else rec.wrap_post_json(http)


def maybe_instrument_inspect_target(model: Any, *, task: str) -> Any:
    """The Petri runner's one guarded call: off = ``model`` itself; on = the labelled per-task copy,
    and ``petri_runner.TARGET_TURN_HEADERS_HOOK`` (when that seam exists on the tree and is
    unset) is pointed at :func:`petri_turn_headers` so the labels also ride ``extra_headers``."""
    rec = from_env()
    if rec is None:
        return model
    try:
        from sisteer.capeval.tiers import petri_runner as _pr
    except Exception:  # noqa: BLE001 — the runner is the caller; an import problem is its own
        _pr = None
    if _pr is not None and hasattr(_pr, "TARGET_TURN_HEADERS_HOOK") and _pr.TARGET_TURN_HEADERS_HOOK is None:
        _pr.TARGET_TURN_HEADERS_HOOK = petri_turn_headers
    return rec.instrument_inspect_target(model, task=task)


def maybe_install_petri_target_hooks(model: Any) -> Any:
    """The Petri runner's ONE guarded call (position-stable: after its `if not tasks` guard, before the
    per-behaviour targets are built): off = ``model`` untouched; on = the request + response httpx hooks on
    the target api's ``http_client`` (shared by every per-behaviour copy the runner builds from ``model``,
    ``make_seeded_target`` included), and ``petri_runner.TARGET_TURN_HEADERS_HOOK`` pointed at
    :func:`petri_turn_headers` when that seam exists and is unset — so each target turn's request carries
    ``petri/<behaviour>`` + ``<scenario_id>/<turn>`` from the runner's own ``TargetTurn`` and the hook mints
    the id. On a tree WITHOUT the seam nothing labels the turns and the first target request is
    refused by name (an unlabelled task request) — loud, never a silently unlabelled Petri leg."""
    rec = from_env()
    if rec is None:
        return model
    http_client = getattr(getattr(model, "api", None), "http_client", None)
    if http_client is None:
        raise RequestEvidenceError(
            f"the Petri target model ({type(model).__name__}) exposes no api.http_client — request evidence needs the "
            "httpx boundary to record the wire bytes")
    # the response hook ALSO files each target turn's binding on the active Inspect sample's store (review fix
    # P1-3: the two files bind attempts to the engine; the sample store binds them to the FILED scenario rows)
    rec.install_httpx(http_client, on_binding=petri_sample_turn_sink)
    try:
        from sisteer.capeval.tiers import petri_runner as _pr
    except Exception:  # noqa: BLE001 — the runner is the caller
        _pr = None
    if _pr is not None and hasattr(_pr, "TARGET_TURN_HEADERS_HOOK") and _pr.TARGET_TURN_HEADERS_HOOK is None:
        _pr.TARGET_TURN_HEADERS_HOOK = petri_turn_headers
    return model


def maybe_request_headers_fn(leg: str, name: Optional[str] = None) -> Optional[Callable[[Any, int, int], Dict[str, str]]]:
    """The BoN seam (``bon_served_run.py`` ``request_headers_fn`` -> ``bon_generate_served(request_headers_fn=)``):
    ``(goal_id, sample_idx, trial) -> headers`` of THAT draw — the two labels (``store_task_name(leg[, name])``,
    ``<goal_id>/<sample_idx>``) and a fresh ``X-Tinycap-Request-Id`` (their writer requires it and copies it onto
    the row; the attached transport keeps a caller-minted id). ``None`` when the hook is off (their default:
    no headers, rows byte-identical)."""
    rec = from_env()
    if rec is None:
        return None
    task = store_task_name(leg, name)

    def fn(goal_id: Any, sample_idx: int, trial: int) -> Dict[str, str]:  # noqa: ARG001 — the trial is the store's, not a label
        return rec.headers_for(None, headers=label_headers(task, str(goal_id), sample_idx))   # validated there, never coerced
    return fn


class MisalignRequestEvidence:
    """The misalign seam (``runner.run_family(request_evidence=)``): ``headers_for(*, family, behavior,
    item_id, sample_idx, payload)`` returns the two LABELS only (the id is minted per HTTP attempt by the
    attached transport, :meth:`RequestEvidenceClient.wrap_post_json` — a 5xx retry is a new attempt);
    ``record(*, payload, headers, response)`` returns the request id of the attempt that ANSWERED (the
    transport's binding by the response's completion id), which their writer puts on the row as
    ``request_id``; ``None`` when the response bound to nothing (the audit then refuses that row)."""

    def __init__(self, client: RequestEvidenceClient) -> None:
        self.client = client

    def headers_for(self, *, family: str, behavior: Any, item_id: Any, sample_idx: int, payload: Any = None
                    ) -> Dict[str, str]:  # noqa: ARG002 — the payload is the transport's to record
        return label_headers(store_task_name(LEG_MISALIGN, str(behavior), family=str(family)), str(item_id), sample_idx)

    def record(self, *, payload: Any = None, headers: Any = None, response: Any = None) -> Optional[str]:  # noqa: ARG002
        binding = self.client.binding_of(response)
        return None if binding is None else binding.get(ROW_REQUEST_ID_FIELD)


def maybe_misalign_request_evidence() -> Optional[MisalignRequestEvidence]:
    """The misalign CLI's one plug-in value: ``run_family(..., request_evidence=maybe_misalign_request_evidence())``
    — ``None`` when the hook is off (the default: no headers, no record, no field)."""
    rec = from_env()
    return None if rec is None else MisalignRequestEvidence(rec)


def maybe_aiohttp_trace_configs() -> List[Any]:
    """``trace_configs=`` for an ``aiohttp.ClientSession``: ``[]`` when off."""
    rec = from_env()
    return [] if rec is None else [rec.aiohttp_trace_config()]


def row_fields(resp: Any) -> Dict[str, Optional[str]]:
    """``{request_id, completion_id}`` of a response dict the process's client observed,
    ``{}`` otherwise (and always ``{}`` with the hook off) — for ``row.update(...)``."""
    rec = from_env()
    return {} if rec is None else rec.row_fields(resp)


def turn_fields(turn: Any) -> Dict[str, Optional[str]]:
    """The binding an object carries as attributes (``ServedTurn.request_id`` /
    ``.completion_id``), as row fields when the request id is set; else ``{}``."""
    rid = getattr(turn, ROW_REQUEST_ID_FIELD, None)
    if rid is None:
        return {}
    return {ROW_REQUEST_ID_FIELD: rid, ROW_COMPLETION_ID_FIELD: getattr(turn, ROW_COMPLETION_ID_FIELD, None)}


def last_binding_fields(defense: Any) -> Dict[str, Optional[str]]:
    """The binding of the LAST response a served target returned (``ServedChatTarget.
    last_request_binding``), as row fields — the multi-turn writers persist a reply's text
    in an ``on_response(text)`` callback and read the target's last block beside it (the
    ``last_decode`` precedent); ``{}`` when none."""
    b = getattr(defense, "last_request_binding", None)
    return dict(b) if isinstance(b, Mapping) and b.get(ROW_REQUEST_ID_FIELD) else {}
