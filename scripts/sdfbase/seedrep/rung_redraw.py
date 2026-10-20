#!/usr/bin/env python
"""rung_redraw.py — the seedrep ladder's RUNG CONSUMER and the four drivers' ASSEMBLY seams.

Two halves, one protocol (``scripts/modal/seedrep_ladder.py``, loaded here BY PATH — ``scripts/modal`` is also the
directory that shadows the ``modal`` package on ``sys.path``, so nothing imports it by bare name):

THE CONSUMER (``consume`` / the ``consume`` CLI; runs inside a ``run_rung`` container beside the rung engine).
  Loop until the queue is CLOSED and every accepted entry has an elected result: reload the volume, list the
  ACCEPTED projection (every declared producer's accepted attempt's files — ``--speculative`` adds the live
  entries of not-yet-accepted attempts, drawn to a TENTATIVE ``speculative/`` file that is promoted and elected only
  once that attempt is accepted; a known loser's entry is never drawn, a dropped tentative draw's slot is re-drawn
  from the accepted record — P1-5),
  skip items that already have an elected result, CLAIM each remaining item with ONE ``skip_if_exists`` put (held
  -> skip; no takeover, ever), DRAW it through the proxy at ``cap_next`` with the SAME ``sampling_seed`` under the
  record's own sampling row (a re-draw, never a continuation), WRITE the attempt-unique materialised file, commit,
  ELECT with ONE ``skip_if_exists`` put (a loser records itself under ``duplicates/<attempt_id>.jsonl``), and —
  when this rung's draw is itself at cap and a rung exists above — enqueue it for the NEXT queue with this rung
  attempt as the producer. 16,384 is the ceiling: the draw is recorded ``at_ceiling`` and NEVER queued. The
  container finishes its producer side with ``seedrep_ladder.finish_producer`` (inventory -> acceptance -> CLOSED
  marker) exactly as a base container does.

THE ASSEMBLY SEAMS (``--from-materialised <ladder root> --results <kv snapshot>`` on the existing drivers, each a
  one-call addition, default off, behaviour byte-identical without the flags; each seam's docstring is the authority):
  * static   — ``wrap_static_target`` (``escalate_static.py``: the served target answers eligible items from elected rows);
  * controls — ``wrap_controls_post`` (``served_capability_run.py``: the proxy ``post`` seam answers at-cap re-draws);
  * misalign — ``seed_misalign_root`` (the runner's NEW root is seeded before it draws; ``read_done`` skips the rows);
  * GSM8K    — ``gsm8k_subset_from_queue`` (the Inspect driver draws over ``--item-subset``; the queue is cross-checked;
    no GSM8K rung container pre-draws — ``consume`` refuses ``--kind gsm8k``).

Every verification is fail-closed: the elected row's ``(sampling_seed, cap, preset, base_row_sha256, producer
attempt)`` must match the ACCEPTED base row (``seedrep_ladder.verify_elected``) and the file's ``text_sha256``
must match both the Dict value and the text (``seedrep_ladder.load_elected``).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence

REPO = Path(__file__).resolve().parents[3]


def _load_ladder():
    """``scripts/modal/seedrep_ladder.py`` by path, registered ONCE in ``sys.modules`` so every caller (a driver, a
    test, this module) shares one module object — one ``LadderRefusal`` class, one ``ElectedMismatch``."""
    mod = sys.modules.get("seedrep_ladder")
    if mod is not None:
        return mod
    path = REPO / "scripts" / "modal" / "seedrep_ladder.py"
    spec = importlib.util.spec_from_file_location("seedrep_ladder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seedrep_ladder"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load_ladder()


# --------------------------------------------------------------------------- #
# the consumer's route to the Modal SDK                                       #
# --------------------------------------------------------------------------- #

#: the interpreter that CARRIES the Modal SDK, named by the served app in the consumer child's env (its own ``sys.executable``;
#: ``seedrep_served.RUNG_SDK_PYTHON_ENV`` spells the same literal, pinned by test). The rung container runs this consumer under the
#: DRIVER venv (``uv sync --frozen --extra cu128``) -- no ``modal`` distribution -- with ``PYTHONPATH=<tree>/src:<tree>/scripts``, so a
#: bare ``import modal`` there answers with the repo's ``scripts/modal/`` NAMESPACE package (no ``Volume``, no ``Dict``) instead of
#: ModuleNotFoundError: all 11 seed-1 rungs died ``AttributeError: module 'modal' has no attribute 'Volume'`` at ``_consume_volume``
#:, and ``_kv`` would have died the same way one line later. The wave's Dicts and the mount's commit / reload therefore
#: cross ONE helper process (:class:`SdkBridge`, ``modal_sdk_bridge.py``) under the interpreter this names; without it the direct import
#: is the CHECKED :func:`seedrep_ladder.modal_sdk`, which refuses that shape by name (the grading host reconciler's route, and the tests').
SDK_PYTHON_ENV = L.SDK_PYTHON_ENV
#: the helper beside this file (stdlib + modal only), run ``<python> -I <helper>`` -- isolated: no PYTHONPATH, no user site, no script
#: directory on ``sys.path`` -- so the repo's scripts/ never shadows anything there; it reaches the SDK by prepending the import roots the
#: app hands the consumer child in ``$SEEDREP_MODAL_SDK_PATH`` (fx1b: in a container the SDK lives at /pkg + /__modal/deps, not in site-packages)
SDK_BRIDGE_HELPER = Path(__file__).resolve().with_name("modal_sdk_bridge.py")
_BRIDGE_NO_DEFAULT = object()


class SdkBridgeError(RuntimeError):
    """The bridge could not start, died, or answered with an error the KV protocol does not spell (a KeyError is re-raised as one)."""


class SdkBridge:
    """ONE helper process serving the Modal SDK verbs the consumer needs -- ``modal.Dict`` put / get / pop / contains / items and
    ``modal.Volume`` commit / reload -- over newline-delimited JSON on its stdin / stdout (``modal_sdk_bridge.py`` documents the
    protocol). Requests are serialised under one lock: a Dict verb is one network round trip, and the consumer's parallelism is its
    draws through the proxy, not these. The helper inherits this process's environment (the container's Modal task credentials) and
    stderr (its complaints land in the consumer's log); it exits on EOF, so :meth:`close` -- or this process's end -- ends it. The
    handshake (``hello``) runs in the constructor: an interpreter without the SDK fails HERE, before any Dict is opened or draw made."""

    def __init__(self, python: str, *, helper: Path = SDK_BRIDGE_HELPER, popen: Callable[..., Any] = subprocess.Popen) -> None:
        self.python, self.helper = str(python), Path(helper)
        self._lock = threading.Lock()
        self._next = 0
        self._closed = False
        try:
            self._p = popen([self.python, "-I", str(self.helper)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            raise SdkBridgeError(f"Modal SDK bridge could not start `{self.python} -I {self.helper}`: {type(e).__name__}: {e}") from e
        hello = self.call("hello")       # fx1b: an SDK the helper cannot import is an ok:false reply here -> SdkBridgeError naming the roots tried
        #: the handshake: the SDK version + file + import roots as the helper saw them, ``python`` = the interpreter THIS end was given
        self.hello = {**(hello if isinstance(hello, dict) else {"hello": hello}), "python": self.python, "helper": str(self.helper)}

    def call(self, op: str, **params: Any) -> Any:
        with self._lock:
            if self._closed:
                raise SdkBridgeError(f"Modal SDK bridge is closed ({op})")
            self._next += 1
            rid = self._next
            line = ""
            try:
                self._p.stdin.write(json.dumps({"id": rid, "op": op, **params}) + "\n")
                self._p.stdin.flush()
                line = self._p.stdout.readline()
            except (OSError, ValueError):       # a broken pipe / a closed stream: the helper is gone
                line = ""
            if not line:
                try:
                    rc = self._p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._p.kill()
                    rc = self._p.wait()
                self._closed = True
                raise SdkBridgeError(f"Modal SDK bridge `{self.python} -I {self.helper}` exited rc={rc} during {op!r}; its stderr is in this log")
            try:
                rep = json.loads(line)
            except ValueError as e:
                raise SdkBridgeError(f"Modal SDK bridge answered {op!r} with a non-JSON line: {line[:200]!r}") from e
        if rep.get("id") != rid:
            raise SdkBridgeError(f"Modal SDK bridge answered request {rid} ({op}) with id {rep.get('id')!r}")
        if rep.get("ok"):
            return rep.get("result")
        if rep.get("type") == "KeyError":
            raise KeyError(params.get("key"))
        raise SdkBridgeError(f"Modal SDK bridge {op} {params.get('name', '')}: {rep.get('type')}: {rep.get('error')}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for stream in (self._p.stdin,):
                try:
                    stream.close()
                except OSError:
                    pass
            try:
                self._p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._p.wait()
            try:
                self._p.stdout.close()
            except OSError:
                pass

    def kv(self, name: str) -> "BridgeKV":
        return BridgeKV(self, name)

    def volume_verbs(self, name: str) -> tuple[Callable[[], None], Callable[[], None]]:
        """``(commit, reload)`` of the named Volume, bound for :class:`seedrep_ladder.LocalVolume` over the mount."""
        return (lambda: self.call("volume.commit", name=name)), (lambda: self.call("volume.reload", name=name))


class BridgeKV:
    """:class:`seedrep_ladder.KV` over the bridge -- the four verbs of ``modal.Dict`` the protocol uses, with ``MemoryKV``'s semantics
    (``put(skip_if_exists=True)`` -> False on a held key; ``pop`` -> KeyError when absent unless a default is given). Values ride as
    JSON, which they already are (``snapshot_kv`` writes them; ``_JsonKV`` stores them)."""

    def __init__(self, bridge: SdkBridge, name: str) -> None:
        self._b, self.name = bridge, name

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        return bool(self._b.call("dict.put", name=self.name, key=key, value=value, skip_if_exists=bool(skip_if_exists)))

    def get(self, key: str, default: Any = None) -> Any:
        return self._b.call("dict.get", name=self.name, key=key, default=default)

    def pop(self, key: str, default: Any = _BRIDGE_NO_DEFAULT) -> Any:
        if default is _BRIDGE_NO_DEFAULT:
            return self._b.call("dict.pop", name=self.name, key=key)
        return self._b.call("dict.pop", name=self.name, key=key, default=default)

    def contains(self, key: str) -> bool:
        return bool(self._b.call("dict.contains", name=self.name, key=key))

    def items(self) -> Iterable[tuple[str, Any]]:
        return iter([(k, v) for k, v in (self._b.call("dict.items", name=self.name) or [])])

#: the misalign runner's row shape (``_run_cell.one``): every key a seeded row must carry
MISALIGN_ROW_KEYS = ("family", "cell_id", "virtual_model", "item_id", "sample_idx", "behavior", "prompt_sha",
                     "response", "finish_reason", "usage", "model_reported")


# --------------------------------------------------------------------------- #
# the consumer                                                                  #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Draw:
    """One re-draw's outcome: the leg-shaped ``record`` (``text``, ``finish_reason``, ``completion_tokens``,
    ``prompt_tokens``, ``max_tokens``, ``sampling_seed`` — the ``turn_record`` shape the static ladder consumes),
    the text, the raw completion ``response`` (what the controls seam hands back), and the request / completion
    ids the request-evidence audit binds on."""

    record: dict
    text: str
    response: Any
    request_id: Optional[str]
    completion_id: Optional[str]


DrawFn = Callable[[Mapping[str, Any]], Draw]


def request_body(record: Mapping[str, Any]) -> dict:
    """The re-draw request of a queue record: ``{model, messages, <sampling row>, max_tokens: cap_next, seed}`` —
    the record's own sampling row (the six vendor keys as the base leg sent them), the SAME seed, the rung's cap."""
    redraw = record.get("redraw")
    if not isinstance(redraw, Mapping) or not redraw.get("vmodel") or not redraw.get("messages"):
        raise L.LadderRefusal(f"queue record {record.get('item_key')!r} carries no redraw block (vmodel, messages); "
                              "this leg's rung cannot re-issue the request from it")
    body = {"model": redraw["vmodel"], "messages": [dict(m) for m in redraw["messages"]]}
    body.update(L.check_redraw_sampling(record, where=f"queue record {record.get('item_key')!r} before the send"))   # P2-3
    body["max_tokens"] = int(record["cap_next"])
    if record.get("sampling_seed") is not None:
        body["seed"] = int(record["sampling_seed"])
    return body


def misalign_row_from_response(record: Mapping[str, Any], response: Mapping[str, Any], *, request_id: Optional[str],
                               completion_id: Optional[str]) -> dict:
    """The misalign RUNNER's row (``_run_cell.one``: ``family, cell_id, virtual_model, item_id, sample_idx, behavior,
    prompt_sha`` + ``_extract``'s ``response, finish_reason, usage, model_reported``) from a rung draw — through the
    runner's OWN ``_extract`` and ``messages_sha``, so the materialised record IS what the runner would have written
    and ``seed_misalign_root`` seeds it as-is. The identity comes from the queue record; ``prompt_sha``
    is recomputed from the messages actually re-issued and must equal the base row's (the same prompt was re-drawn);
    the per-response ``seed`` (the row key) and the request provenance ``request_id`` / ``completion_id`` ride on
    the row (additive keys, what the request-evidence client stamps on rung rows)."""
    from sisteer.capeval.misalign import runner as R  # the runner's own extractor: one row shape

    ident = record.get("identity") or {}
    redraw = record.get("redraw") or {}
    for k in ("family", "cell_id", "item_id", "sample_idx"):
        if ident.get(k) is None:
            raise L.LadderRefusal(f"{record.get('item_key')!r}: the misalign queue record's identity lacks {k!r}")
    try:
        got = R._extract(response, cell_id=str(ident["cell_id"]), item_id=str(ident["item_id"]), sample_idx=int(ident["sample_idx"]))
    except R.MisalignRunnerError as e:
        raise L.LadderRefusal(f"{record.get('item_key')!r}: {e}") from None
    if got.get("finish_reason") is None or not isinstance(got.get("usage"), Mapping):
        raise L.LadderRefusal(f"{record.get('item_key')!r}: the completion carries no finish_reason / usage; the ladder keys on them")
    psha = R.messages_sha(list(redraw.get("messages") or []))
    if ident.get("prompt_sha") is not None and psha != ident["prompt_sha"]:
        raise L.LadderRefusal(f"{record.get('item_key')!r}: the re-issued messages hash to {psha}, the base row's prompt_sha is "
                              f"{ident['prompt_sha']}; a rung re-draws the SAME prompt")
    row = {"family": ident["family"], "cell_id": ident["cell_id"], "virtual_model": redraw.get("vmodel") or ident.get("virtual_model"),
           "item_id": str(ident["item_id"]), "sample_idx": int(ident["sample_idx"]), "behavior": ident.get("behavior"),
           "prompt_sha": psha, **got}
    if record.get("sampling_seed") is not None:
        row["seed"] = int(record["sampling_seed"])
    row["request_id"], row["completion_id"] = request_id, completion_id
    return row


def record_from_response(record: Mapping[str, Any], response: Mapping[str, Any], *, request_id: Optional[str] = None,
                         completion_id: Optional[str] = None) -> dict:
    """The materialised ``record`` of a draw, in the LEG's own row shape: the misalign runner's row for a misalign
    record (:func:`misalign_row_from_response`), else ``turn_record``'s fields (static / controls). Refuses a response
    without a readable choice, a finish_reason or usage counts (the ladder keys on them)."""
    if record.get("kind") == "misalign":
        return misalign_row_from_response(record, response, request_id=request_id, completion_id=completion_id)
    from sisteer.eval.attacks import bon_served as bs  # the ONE turn_record shape (ServedChatTarget.generate_detailed's)

    try:
        choice = response["choices"][0]
        text = (choice.get("message") or {}).get("content")
        fr = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise L.LadderRefusal(f"{record.get('item_key')!r}: malformed completion ({e!r})") from None
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    if fr is None or usage.get("completion_tokens") is None or usage.get("prompt_tokens") is None:
        raise L.LadderRefusal(f"{record.get('item_key')!r}: the completion carries no finish_reason / usage counts; "
                              "the ladder keys on them — refusing to materialise it")
    turn = bs.ServedTurn(text=str(text if text is not None else ""), finish_reason=fr, prompt_tokens=int(usage["prompt_tokens"]),
                         completion_tokens=int(usage["completion_tokens"]), max_tokens=int(record["cap_next"]))
    return bs.turn_record(turn, sampling_seed=record.get("sampling_seed"))


#: The additive row fields every filed row of a labelled run carries (``request_evidence_client.BINDING_FIELDS``):
#: set on every materialised record here so the assembly's writer finds the binding on the row it copies
ROW_REQUEST_ID_FIELD, ROW_COMPLETION_ID_FIELD = "request_id", "completion_id"
#: The lane switch (``request_evidence_client.ENV_DIR``), spelled here too so an UNSET environment is decided without
#: importing the client (a smoke on a tree without it); :func:`_evidence_module` refuses if the two spellings ever drift
EVIDENCE_ENV_DIR = "SISTEER_REQUEST_EVIDENCE_DIR"


def _evidence_module():
    """The shared request-evidence client, ``sisteer.eval.request_evidence_client`` — the API used here as read at
    its tip 67355126 (``ENV_DIR`` / ``ENV_EXTRA`` / ``ENV_SEED_BASE`` / ``EXTRA_SEED_BASE_KEY``,
    ``RequestEvidenceClient(out_dir, extra=, seed_base=)`` with ``headers_for`` / ``record_request`` /
    ``record_response`` / ``n_requests`` / ``n_responses``, ``store_task_name``, ``label_headers``, ``LEG_STATIC`` /
    ``LEG_MISALIGN``). Lazy: the assembly seams and a smoke need none of it; a tree without the module refuses by name."""
    try:
        from sisteer.eval import request_evidence_client as REC
    except ImportError as e:
        raise L.LadderRefusal("WS1-D′'s sisteer.eval.request_evidence_client is not on this tree; a rung's draws cannot be "
                              f"recorded ({e}) — pass --no-request-evidence ONLY for a smoke") from None
    if REC.ENV_DIR != EVIDENCE_ENV_DIR:
        raise L.LadderRefusal(f"request_evidence_client.ENV_DIR is {REC.ENV_DIR!r}, this consumer spells it {EVIDENCE_ENV_DIR!r}; the two drifted")
    return REC


def evidence_label(record: Mapping[str, Any]) -> tuple[str, str, int]:
    """``(task, item_id, sample_idx)`` — the two tinycap labels a queue record's re-draw carries, from the
    authoritative task-label API (``store_task_name`` is derived from each store's OWN writer, never a plan table):
    static ``static/<family>`` + ``goal_id/sample_idx``; controls the judged control's task + ``item_id/trial`` (one
    draw per item — the trial IS the store's index, as the record's identity spells it); misalign
    ``<family>/<behavior>`` + ``item_id/sample_idx``. GSM8K has no rung request (Inspect draws)."""
    REC = _evidence_module()
    ident = record.get("identity") or {}
    kind = record.get("kind")
    try:
        if kind == "static":
            return REC.store_task_name(REC.LEG_STATIC, str(ident["family"])), str(ident["goal_id"]), int(ident["sample_idx"])
        if kind == "controls":
            return REC.store_task_name(str(ident["control"])), str(ident["item_id"]), int(ident.get("trial", 0))
        if kind == "misalign":
            return (REC.store_task_name(REC.LEG_MISALIGN, str(ident["behavior"]), family=str(ident["family"])),
                    str(ident["item_id"]), int(ident["sample_idx"]))
    except KeyError as e:
        raise L.LadderRefusal(f"{record.get('item_key')!r}: the {kind!r} record's identity lacks {e} — its request cannot be labelled") from None
    raise L.LadderRefusal(f"{record.get('item_key')!r}: a {kind!r} record has no rung request to label")


class RungEvidence:
    """The rung consumer's request-evidence side: every send goes through the shared
    ``RequestEvidenceClient`` — the two labels from :func:`evidence_label`, a fresh ``X-Tinycap-Request-Id`` per HTTP
    attempt, the EXACT bytes handed to ``urllib`` recorded as the version-3 request record, the response (or the
    failure) recorded beside it — so the rung attempt root carries the ``request_evidence.jsonl`` +
    ``response_evidence.jsonl`` the audit and S3 read for every rung store. ONE client per ``(arm, s)`` of
    the records drawn, all appending to the same two files under the attempt root (``ENV_DIR``), each stamping its own
    ``extra = {leg, arm, s, attempt_id}``: the Arditi consumer multiplexes s1–s3 in one container, so ``extra.s`` is
    selected PER REQUEST from the queue record (the audit partitions that engine's records by it). A seed base the
    environment pins (``ENV_SEED_BASE`` / ``extra.s``) or an ``arm`` / ``leg`` / ``attempt_id`` it names must agree
    with every record — a disagreement refuses, never a mis-stamped record."""

    _PINNED = ("leg", "arm", "attempt_id")

    def __init__(self, out_dir: "Path | str", *, leg: str, attempt_id: str, base_extra: Optional[Mapping[str, Any]] = None,
                 pinned_seed_base: Optional[int] = None) -> None:
        self.REC = _evidence_module()
        self.out_dir = Path(out_dir)
        self.leg, self.attempt_id = str(leg), str(attempt_id)
        extra = dict(base_extra or {})
        self.pinned_arm: Optional[str] = None if extra.get("arm") is None else str(extra["arm"])
        for k in (*self._PINNED, self.REC.EXTRA_SEED_BASE_KEY):
            extra.pop(k, None)
        self.base_extra = extra
        self.pinned_seed_base = None if pinned_seed_base is None else int(pinned_seed_base)
        self._clients: dict[tuple[str, int], Any] = {}
        self._lock = threading.Lock()      # the client table
        self._io = threading.Lock()        # one appender at a time: the per-(arm, s) clients share the two files

    @classmethod
    def from_env(cls, *, leg: str, attempt_id: str, env: Optional[Mapping[str, str]] = None) -> Optional["RungEvidence"]:
        """The lane switch: ``None`` when ``SISTEER_REQUEST_EVIDENCE_DIR`` is unset (the caller decides whether that
        is allowed); else the recorder over that directory, with ``SISTEER_REQUEST_EVIDENCE_EXTRA`` (a JSON object) and
        ``SISTEER_REQUEST_EVIDENCE_SEED_BASE`` read exactly as the client reads them — an ``extra`` naming this consumer's
        ``leg`` / ``attempt_id`` otherwise refuses."""
        env = os.environ if env is None else env
        out_dir = env.get(EVIDENCE_ENV_DIR)
        if not out_dir:
            return None                                   # decided WITHOUT the client: a smoke on a tree without it is not refused here
        REC = _evidence_module()
        extra_raw = env.get(REC.ENV_EXTRA) or None
        extra: dict = {}
        if extra_raw:
            parsed = json.loads(extra_raw)
            if not isinstance(parsed, Mapping):
                raise L.LadderRefusal(f"{REC.ENV_EXTRA} must be a JSON object, got {parsed!r}")
            extra = dict(parsed)
        pinned: Optional[int] = None
        seed_raw = env.get(REC.ENV_SEED_BASE) or None
        if seed_raw is not None:
            if not seed_raw.isdigit():
                raise L.LadderRefusal(f"{REC.ENV_SEED_BASE} must be a non-negative integer, got {seed_raw!r}")
            pinned = int(seed_raw)
        in_extra = extra.get(REC.EXTRA_SEED_BASE_KEY)
        if in_extra is not None:
            if pinned is not None and int(in_extra) != pinned:
                raise L.LadderRefusal(f"seed base given twice and disagreeing: {REC.ENV_EXTRA} s={in_extra!r} vs {REC.ENV_SEED_BASE}={pinned}")
            pinned = int(in_extra)
        for k, mine in (("leg", str(leg)), ("attempt_id", str(attempt_id))):
            if extra.get(k) is not None and str(extra[k]) != mine:
                raise L.LadderRefusal(f"{REC.ENV_EXTRA} names {k}={extra[k]!r}, this consumer is {mine!r}; refusing to record under another's")
        return cls(out_dir, leg=leg, attempt_id=attempt_id, base_extra=extra, pinned_seed_base=pinned)

    def client_for(self, record: Mapping[str, Any]) -> Any:
        """The client stamping THIS record's ``(arm, s)`` — built on first use, then reused."""
        arm, s = str(record["arm"]), int(record["s"])
        if self.pinned_seed_base is not None and s != self.pinned_seed_base:
            raise L.LadderRefusal(f"{record.get('item_key')!r}: the evidence environment pins seed base {self.pinned_seed_base} but this "
                                  f"record is s={s} — a rung consumer that multiplexes seed bases (Arditi) must be launched without "
                                  f"{self.REC.ENV_SEED_BASE} / extra.s; extra.s is stamped per request from the record")
        if self.pinned_arm is not None and arm != self.pinned_arm:
            raise L.LadderRefusal(f"{record.get('item_key')!r}: the evidence environment pins arm {self.pinned_arm!r} but this record "
                                  f"is {arm!r} — a rung consumer serves every arm of its engine; leave arm out of {self.REC.ENV_EXTRA}")
        with self._lock:
            client = self._clients.get((arm, s))
            if client is None:
                client = self.REC.RequestEvidenceClient(
                    self.out_dir, extra={**self.base_extra, "leg": self.leg, "arm": arm, self.REC.EXTRA_SEED_BASE_KEY: s,
                                         "attempt_id": self.attempt_id}, seed_base=s)
                self._clients[(arm, s)] = client
        return client

    def headers_for(self, client: Any, record: Mapping[str, Any], body: Mapping[str, Any], headers: Mapping[str, str]) -> dict:
        """The caller's headers + the record's two labels + a fresh ``X-Tinycap-Request-Id`` (the client mints it)."""
        task, item_id, sample_idx = evidence_label(record)
        return client.headers_for(body, headers={**dict(headers), **self.REC.label_headers(task, item_id, sample_idx)})

    def record_request(self, client: Any, raw: bytes, headers: Mapping[str, Any]) -> None:
        with self._io:
            client.record_request(raw, headers)

    def record_response(self, client: Any, request_id: str, **kw: Any) -> None:
        with self._io:
            client.record_response(request_id, **kw)

    def summary(self) -> dict:
        with self._lock:
            clients = list(self._clients.items())
        return {"dir": str(self.out_dir), "n_requests": sum(c.n_requests for _, c in clients),
                "n_responses": sum(c.n_responses for _, c in clients), "arm_s": sorted(L.arm_s(a, s) for (a, s), _ in clients)}


def http_draw_fn(proxy_url: str, *, api_key: str = "petri-proxy", timeout_s: float = 7500.0,
                 extra_headers: Optional[Mapping[str, str]] = None, evidence: Optional[RungEvidence] = None) -> DrawFn:
    """A :data:`DrawFn` over the proxy's ``/v1/chat/completions`` (stdlib ``urllib``; one request per item, no
    client retry — a proxy refusal must surface). With ``evidence`` (the production rung), every send is labelled and
    recorded through the shared client (:class:`RungEvidence`): the two tinycap labels of the record, a fresh
    ``X-Tinycap-Request-Id`` per attempt, the EXACT request bytes as the version-3 record BEFORE the send, the
    response — or the proxy's refusal / the transport's death — as the response record after it. Without it, only the
    request id is minted (``contracts.HEADER_REQUEST_ID``; a smoke). The completion ``id`` is read back and both ids are
    set on the materialised record (:data:`ROW_REQUEST_ID_FIELD` / :data:`ROW_COMPLETION_ID_FIELD`), so the row the
    assembly copies binds to its engine-log request."""
    import urllib.error
    import urllib.request

    from sisteer.capeval.misalign.runner import v1_base
    from sisteer.capeval.tinycap.contracts import HEADER_REQUEST_ID

    url = f"{v1_base(proxy_url)}/chat/completions"

    def draw(record: Mapping[str, Any]) -> Draw:
        body = request_body(record)
        raw = json.dumps(body).encode("utf-8")                     # the EXACT bytes handed to urllib — and recorded
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        headers.update(dict(extra_headers or {}))
        client = None
        if evidence is not None:
            client = evidence.client_for(record)
            headers = evidence.headers_for(client, record, body, headers)
            rid = headers[HEADER_REQUEST_ID]
            evidence.record_request(client, raw, headers)
        else:
            rid = uuid.uuid4().hex
            headers[HEADER_REQUEST_ID] = rid
        req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                status, raw_resp = int(r.status), r.read()
        except urllib.error.HTTPError as e:
            err = e.read()[:2000]
            if client is not None:
                evidence.record_response(client, rid, status=int(e.code), error=err.decode("utf-8", "replace"), headers=headers)
            raise L.LadderRefusal(f"{record.get('item_key')!r}: proxy returned {e.code}: {err[:400]!r}") from None
        except (urllib.error.URLError, OSError) as e:                # the transport died: a dead attempt, recorded, then raised
            if client is not None:
                evidence.record_response(client, rid, status=None, error=f"{type(e).__name__}: {e}", headers=headers)
            raise L.LadderRefusal(f"{record.get('item_key')!r}: the request to {url} died before a response ({type(e).__name__}: {e})") from None
        response = json.loads(raw_resp.decode("utf-8"))
        if client is not None:
            evidence.record_response(client, rid, status=status, resp=response if isinstance(response, Mapping) else None, headers=headers)
        cid = response.get("id") if isinstance(response, Mapping) else None
        rec = record_from_response(record, response, request_id=rid, completion_id=cid)
        rec[ROW_REQUEST_ID_FIELD], rec[ROW_COMPLETION_ID_FIELD] = rid, cid
        return Draw(record=rec, text=L.record_text(rec), response=response, request_id=rid, completion_id=cid)

    return draw


def next_queue_record(spec, next_spec, *, record: Mapping[str, Any], text: str, finish_reason: Optional[str], producer,
                      producer_attempt_id: str, materialised_path: str) -> dict:
    """An ELECTED at-cap draw of this rung as the NEXT queue's entry: same identity / item_key / seed, ``base_cap`` =
    this rung's cap, ``base_row_sha256`` = the normalised identity of the elected materialised record (``text``,
    ``finish_reason``), ``base_store`` = the elected file; the redraw block travels unchanged."""
    return L.make_queue_record(
        next_spec, arm=record["arm"], s=int(record["s"]), kind=record["kind"], identity=record["identity"],
        item_key_=record["item_key"], sampling_seed=record.get("sampling_seed"), base_cap=int(spec.cap_next),
        base_row_sha256=L.base_row_identity_sha256(text=text, sampling_seed=record.get("sampling_seed"),
                                                   finish_reason=finish_reason, cap=int(spec.cap_next)),
        base_store=materialised_path, producer=producer, producer_attempt_id=producer_attempt_id,
        redraw=record.get("redraw"))


class Candidate(NamedTuple):
    """A queue entry the consumer may draw: its record and whether it is the ACCEPTED projection's (else a live,
    not-yet-accepted producer's entry under ``--speculative``, drawn tentatively)."""

    rec: dict
    accepted: bool


class Tentative(NamedTuple):
    """A speculative draw awaiting its producer's acceptance: the ``speculative/`` file and its document (the queue
    record and the drawn record ride inside the document)."""

    path: str
    doc: dict


def known_loser(rec: Mapping[str, Any], acc: Mapping[str, Optional[str]]) -> bool:
    """A live entry whose shard already has an accepted attempt that is NOT this entry's — or whose shard is not a
    declared producer — can never be consumed (P1-5: exclude known losers)."""
    shard, pa = rec["producer"]["shard"], rec["producer"]["attempt_id"]
    holder = acc.get(shard)
    return shard not in acc or (holder is not None and holder != pa)


def adopt_found_results(vol, *, found: Mapping[str, tuple], spec, attempt_id: str, call_id: str, shard: Optional[str],
                        attempt_root: Optional[str], adopt_into: Optional[Path], evidence_dir: Optional[Path], evidence_on: bool,
                        retries: int = 3, poll_s: float = 30.0, sleep: Callable[[float], Any] = time.sleep,
                        log: Callable[[str], Any] = print) -> dict:
    """The elected results this consumer relies on but did not draw are its predecessors' — Modal re-executed
    the SAME FunctionCall (a new attempt id, ``call_id`` equal) or the slot was re-spawned (``shard`` equal) — and the
    accepted attempt must carry THEIR request pipeline, or the audit cannot bind the store rows their draws produced. Groups
    ``found`` (``slot -> (queue record, elected value)``) by claimant: ours (a re-entry) need nothing; a claimant of the SAME
    call or the SAME shard is adopted — its root (the claimant's ``attempt_root``, else the sibling ``attempts/<id>`` of ours)
    is read on the volume and :func:`seedrep_ladder.adopt_evidence` merges exactly the covering records into ``adopt_into``,
    retrying ``retries`` times ``poll_s`` apart for a predecessor whose TreeSyncer may still be shipping; a claimant of ANOTHER
    shard is that slot's own business (``foreign``, its accepted root carries it); a claimant whose slot cannot be told and
    whose call is not ours REFUSES (never silently someone else's evidence). With the evidence side off (a smoke) the
    adoption is skipped and counted. Returns ``{adopted: [index entries], foreign: [slots], own: n, skipped: n}``."""
    groups: dict[str, list[dict]] = {}
    roots: dict[str, Optional[str]] = {}
    foreign: list[str] = []
    own = sum(1 for _, v in found.values() if str(v.get("attempt_id")) == str(attempt_id))
    others = {str(v.get("attempt_id")) for _, v in found.values()} - {str(attempt_id)}
    if not evidence_on:                            # a smoke: no request pipeline anywhere, so nothing to adopt — counted, never classified
        if others:
            log(f"[rung-redraw] {len(others)} other attempt(s) elected results this consumer relies on; the evidence side is off (a smoke): "
                "adoption skipped")
        return {"adopted": [], "foreign": [], "own": own, "skipped": len(others)}
    for slot, (rec, value) in sorted(found.items()):
        cl_attempt = str(value.get("attempt_id"))
        if cl_attempt == str(attempt_id):
            continue
        cl_call, cl_shard, cl_root = value.get("call_id"), value.get("shard"), value.get("attempt_root")
        if cl_call is None or cl_shard is None or cl_root is None:          # an older value: the claimant block of the file
            el = L.load_elected_one(vol, None, spec, rec["arm_s"], rec["item_key"], value=value)
            cl = el.doc.get("claimant") or {}
            cl_call = cl_call if cl_call is not None else cl.get("call_id")
            cl_shard = cl_shard if cl_shard is not None else cl.get("shard")
            cl_root = cl_root if cl_root is not None else cl.get("attempt_root")
        same_call = bool(call_id) and cl_call == str(call_id)
        same_shard = shard is not None and cl_shard is not None and str(cl_shard) == str(shard)
        if not (same_call or same_shard):
            if cl_shard is None:
                raise L.LadderRefusal(f"{slot}: elected by attempt {cl_attempt!r} of call {cl_call!r}, whose slot is unknown (no shard on "
                                      "the claimant) and whose call is not this consumer's — the result's evidence cannot be attributed; "
                                      "refusing to accept over it")
            foreign.append(slot)
            continue
        if cl_root is None and attempt_root:
            cl_root = str(PurePosixPath(attempt_root).parent / cl_attempt)
        if cl_root is None:
            raise L.LadderRefusal(f"{slot}: elected by predecessor {cl_attempt!r} whose attempt root is unknown (no attempt_root on the "
                                  "claimant, none derivable from ours); its evidence cannot be adopted")
        roots[cl_attempt] = cl_root
        groups.setdefault(cl_attempt, []).append({"slot": slot, "item_key": rec["item_key"], "arm_s": rec["arm_s"],
                                                  "request_id": value.get("request_id"), "completion_id": value.get("completion_id"),
                                                  "materialised_path": value.get("path")})
    out = {"adopted": [], "foreign": foreign, "own": own, "skipped": 0}
    if not groups:
        return out
    if adopt_into is None or evidence_dir is None:
        raise L.LadderRefusal(f"{len(groups)} predecessor attempt(s) ({sorted(groups)}) elected results this consumer relies on, but the "
                              "attempt root to merge their evidence into is unknown — pass --attempt-root-local (or put ATTEMPT.json at the "
                              "root above SISTEER_REQUEST_EVIDENCE_DIR)")
    adopter = {"attempt_id": str(attempt_id), "call_id": str(call_id), **({"shard": str(shard)} if shard is not None else {})}
    for src_attempt, items in sorted(groups.items()):
        for k in range(int(retries) + 1):
            try:
                vol.reload()
                manifest = L.adopt_evidence(vol, into_root=adopt_into, evidence_dir=evidence_dir, source_root=roots[src_attempt],
                                            source_attempt=src_attempt, items=items, adopter=adopter)
                break
            except L.LadderRefusal as e:
                if k >= int(retries):
                    raise
                log(f"[rung-redraw] adoption of {src_attempt!r}'s evidence not yet possible ({e}); the predecessor may still be syncing "
                    f"— retry {k + 1}/{int(retries)} in {poll_s:g}s")
                sleep(poll_s)
        out["adopted"].append({"attempt_id": src_attempt, "n_items": len(items), "manifest": f"{L.ADOPTED_DIR}/{src_attempt}/{L.ADOPTED_MANIFEST}",
                               "sha256": L.sha256_hex((L.canonical_json(manifest) + "\n").encode("utf-8"))})
        log(f"[rung-redraw] adopted {len(items)} elected result(s) of predecessor {src_attempt!r}: request pipeline merged into {adopt_into} "
            f"(manifest {out['adopted'][-1]['manifest']})")
    return out


def consume(vol, *, accepted_kv, claims_kv, results_kv, spec, attempt_id: str, call_id: str, draw: DrawFn,
            next_producer=None, speculative: bool = False, width: int = 1, arm_filter: Optional[Callable[[str], bool]] = None,
            poll_s: float = 30.0, deadline_s: Optional[float] = None, sleep: Callable[[float], Any] = time.sleep,
            clock: Callable[[], float] = time.monotonic, log: Callable[[str], Any] = print,
            max_rounds: Optional[int] = None, shard: Optional[str] = None, attempt_root: Optional[str] = None,
            adopt_into: Optional[Path] = None, evidence_dir: Optional[Path] = None, adopt_retries: int = 3) -> dict:
    """The rung consumer loop (module docstring). Per round: reload; ONE walk of ``queue/`` (records cached across
    rounds — the files are immutable) and ONE scan of the results Dict; the accepted projection (plus, with
    ``speculative``, the live entries of not-yet-accepted producers that are not known losers); tentative draws
    whose producer has since been accepted are promoted, elected and forwarded, a loser's dropped; every unclaimed,
    unelected candidate is claimed right before its draw and drawn at ``width`` in flight (the ATTESTED rung width,
    the launcher's ``--concurrency S``): write the attempt-unique file, commit (ONE Volume commit at a time), elect,
    forward an at-cap winner to the next queue (also from an elected at-cap result found on entry). Once CLOSED and
    covered, the elected results this attempt relies on but did not draw are ADOPTED (:func:`adopt_found_results`):
    ``shard`` / ``attempt_root`` are this container's slot and volume-relative root (stamped on its own draws for a
    later execution), ``adopt_into`` / ``evidence_dir`` the local attempt root and evidence directory the predecessors'
    request pipeline is merged into. Returns the tally dict built below (every key is read by a test)."""
    if spec.kind == "gsm8k":
        raise L.LadderRefusal("GSM8K rungs are drawn by the Inspect driver over the queue-derived subset "
                              "(inspect_rung.subset_from_materialised_queue); no rung container pre-draws them")
    if next_producer is not None and next_producer.spec.base_cap != int(spec.cap_next):
        raise L.LadderRefusal(f"next_producer feeds a {next_producer.spec.cap_next} queue whose predecessor is "
                              f"{next_producer.spec.base_cap}, not this rung's {spec.cap_next}")
    width = int(width)
    if width < 1:
        raise L.LadderRefusal(f"width {width}: the rung dispatches at least one draw at a time (the attested S_rung)")
    ceiling = L.next_cap(spec.kind, spec.cap_next) is None
    results_prefix = L.key_str(spec.queue_key) + "|"
    tally = {"drawn": 0, "won": 0, "lost": 0, "skipped_held": 0, "claims_resumed": 0, "already_elected": 0, "enqueued_next": 0,
             "forwarded_found": 0, "unforwardable": 0, "at_ceiling": 0, "speculative_drawn": 0, "speculative_promoted": 0,
             "speculative_dropped": 0, "speculative_losers_skipped": 0, "speculative_pending": 0, "rounds": 0, "width": width,
             "closed": False, "covered": False, "adopted": [], "foreign_results": 0, "adoption_skipped": 0}
    handled: set[str] = set()                  # slots this attempt published (won / lost) or found elected: a result exists
    found: dict[str, tuple] = {}               # accepted slots whose elected result this attempt did not draw -> (record, elected value)
    tentative: dict[str, Tentative] = {}       # slots drawn speculatively, awaiting their producer's acceptance
    cache: dict[str, dict] = {}                # queue records by path (immutable files: read once)
    lock = threading.Lock()                    # the tally + handled + tentative
    commit_lock = threading.Lock()             # one Volume commit at a time

    def bump(key: str, n: int = 1) -> None:
        with lock:
            tally[key] += n

    def commit() -> None:
        with commit_lock:
            vol.commit()

    def forward(slot: str, rec: Mapping[str, Any], record: Mapping[str, Any], path: str, *, found: bool) -> None:
        """The next queue's entry for an ELECTED at-cap result — ours, or one found on entry (P1-7: a rung attempt
        that elected an at-cap draw and died before publishing it must not lose the 16,384 work; its replacement
        publishes from the validated elected result under ITS OWN attempt id — attempt-unique paths make that
        idempotent across attempts; within one attempt a slot is published once by construction and
        ``Producer.enqueue`` refuses a repeat). At the ceiling: recorded, never queued. Committed at once: the next
        rung's spin condition and its speculative consumer read the COMMITTED queue, and a forward is at most the
        at-cap share of draws, so its own commit costs nothing measurable."""
        if record.get("finish_reason") != "length":
            return
        if ceiling:
            bump("at_ceiling")
            return
        if next_producer is None:
            return
        nq = next_queue_record(spec, next_producer.spec, record=rec, text=L.record_text(record), finish_reason=record["finish_reason"],
                               producer=next_producer.ref, producer_attempt_id=attempt_id, materialised_path=path)
        next_producer.enqueue(nq)
        commit()
        bump("enqueued_next")
        if found:
            bump("forwarded_found")

    def forward_found(slot: str, rec: Mapping[str, Any], value: Mapping[str, Any]) -> None:
        """An elected result found on entry: read and VERIFY it (file, sha, claimant), require its queue record to
        describe the same world as the accepted record held (``same_world``) — else it is another world's result,
        left for the assembly to refuse — then forward it."""
        el = L.load_elected_one(vol, None, spec, rec["arm_s"], rec["item_key"], value=value)
        if not L.same_world(el.doc.get("queue_record") or {}, rec):
            bump("unforwardable")
            log(f"[rung-redraw] {slot}: the elected result's queue record is not the accepted record's world "
                f"({(el.doc.get('queue_record') or {}).get('producer')!r} vs {rec['producer']!r}); not forwarding it")
            return
        forward(slot, rec, el.doc.get("record") or {}, el.path, found=True)

    def publish(slot: str, rec: Mapping[str, Any], record: Mapping[str, Any], rel: str, doc: Mapping[str, Any]) -> None:
        """Elect the committed draw; an at-cap winner feeds the next queue (or is recorded at the ceiling); a loser
        records itself (evidence the audit reads; committed, so a lost container leaves it on the volume)."""
        if L.elect_result(results_kv, spec, doc, rel):
            bump("won")
            forward(slot, rec, record, rel, found=False)
        else:
            winner = results_kv.get(doc["key"]) or {}
            with lock:                         # the attempt's ONE duplicates file: appended under the lock
                L.record_duplicate(vol, spec, attempt_id, doc, rel, winner)
            commit()
            bump("lost")
        with lock:
            handled.add(slot)

    def claim(rec: Mapping[str, Any]) -> bool:
        """ONE skip_if_exists put; a claim already held by THIS attempt (an earlier call of this container, before a
        deadline) or by THIS Modal FunctionCall under an earlier attempt id (a lost container re-executed by Modal —
        the launcher mints the re-execution a new attempt id). Held by another call -> False."""
        if L.claim_item(claims_kv, spec, rec["arm_s"], rec["item_key"], attempt_id=attempt_id, call_id=call_id):
            return True
        holder = claims_kv.get(L.item_scoped_key(spec.queue_key, rec["arm_s"], rec["item_key"])) or {}
        if L.claim_is_this_calls(holder, attempt_id=attempt_id, call_id=call_id):
            if holder.get("attempt_id") != attempt_id:
                bump("claims_resumed")
            return True
        bump("skipped_held")
        return False

    def work(slot: str, cand: Candidate) -> None:
        # the claim is taken HERE, right before the draw, so a dying container leaves at most `width` claimed-undrawn
        # items for the reconciler to clear (never a whole round's worth)
        if not claim(cand.rec):
            return
        d = draw(cand.rec)
        record = dict(d.record)
        if record.get("finish_reason") == "length" and ceiling:
            record["at_ceiling"] = True
        rel, doc = L.write_materialised(vol, spec, queue_record=cand.rec, attempt_id=attempt_id, call_id=call_id, record=record,
                                        request_id=d.request_id, completion_id=d.completion_id, response=d.response,
                                        tentative=not cand.accepted, shard=shard, attempt_root=attempt_root)
        commit()
        bump("drawn")
        if cand.accepted:
            publish(slot, cand.rec, record, rel, doc)
        else:
            bump("speculative_drawn")
            with lock:
                tentative[slot] = Tentative(rel, doc)       # P1-5: tentative until its producer is accepted

    def settle_tentative(acc: Mapping[str, Optional[str]], candidates: Mapping[str, Candidate]) -> None:
        """Every tentative draw whose producer attempt is now ACCEPTED — and whose slot the accepted projection carries
        as the same world — is promoted to the canonical path, committed and elected / forwarded; one whose producer
        is a known loser (or whose world the projection does not carry) is dropped: its file stays as inert
        evidence, and the slot, still claimed by this attempt, is re-drawn from the accepted record; the rest wait."""
        accepted_ids = {a for a in acc.values() if a}
        for slot, t in list(tentative.items()):
            rec = t.doc["queue_record"]
            cand = candidates.get(slot)
            pa = rec["producer"]["attempt_id"]
            if cand is not None and cand.accepted and L.same_world(cand.rec, rec):
                rel, doc = L.promote_materialised(vol, spec, t.doc, t.path)
                commit()
                with lock:
                    del tentative[slot]
                bump("speculative_promoted")
                publish(slot, cand.rec, t.doc["record"], rel, doc)
            elif pa in accepted_ids or known_loser(rec, acc):
                with lock:
                    del tentative[slot]
                bump("speculative_dropped")
                log(f"[rung-redraw] {slot}: speculative draw from producer attempt {pa!r} dropped (not the accepted world); "
                    f"{t.path} stays as evidence, the accepted record is drawn")

    def dispatch(jobs: Sequence[tuple[str, Candidate]]) -> None:
        """``width`` draws in flight; the first failure cancels what has not started and propagates after the
        running draws finish (a real container dies on it; nothing half-published stays elected)."""
        if not jobs:
            return
        if width == 1 or len(jobs) == 1:
            for j in jobs:
                work(*j)
            return
        with ThreadPoolExecutor(max_workers=min(width, len(jobs)), thread_name_prefix="rung-draw") as pool:
            pending_f = {pool.submit(work, *j) for j in jobs}
            while pending_f:
                done, pending_f = wait(pending_f, return_when=FIRST_EXCEPTION)
                for f in done:
                    exc = f.exception()
                    if exc is not None:
                        for q in pending_f:
                            q.cancel()
                        raise exc

    t0 = clock()
    while True:
        tally["rounds"] += 1
        vol.reload()
        closed = L.read_closed(vol, spec)
        acc = L.accepted_producer_attempts(accepted_kv, spec)
        files = L._queue_files(vol, spec)                       # ONE walk per round
        candidates: dict[str, Candidate] = {slot: Candidate(rec, True)
                                            for slot, rec in L.accepted_queue_entries(vol, None, spec, acc, files=files, cache=cache).items()}
        if speculative:
            accepted_ids = {a for a in acc.values() if a}
            for slot, recs in L.live_queue_entries(vol, spec, files=files, cache=cache).items():
                if slot in candidates:
                    continue
                losers = [r for r in recs if known_loser(r, acc)]
                tally["speculative_losers_skipped"] += len(losers)
                live = [r for r in recs if r["producer"]["attempt_id"] not in accepted_ids and r not in losers]
                if len(live) == 1:      # two live attempts of one item: wait for the acceptance to pick one
                    candidates[slot] = Candidate(live[0], False)
        elected = {k: v for k, v in results_kv.items() if k.startswith(results_prefix)}   # ONE Dict scan per round
        if tentative:
            settle_tentative(acc, candidates)
        jobs: list[tuple[str, Candidate]] = []
        for slot in sorted(candidates):
            cand = candidates[slot]
            if arm_filter is not None and not arm_filter(cand.rec["arm_s"]):
                continue
            rkey = L.item_scoped_key(spec.queue_key, cand.rec["arm_s"], cand.rec["item_key"])
            if rkey in elected:
                if slot not in handled:
                    tally["already_elected"] += 1
                    handled.add(slot)
                    if cand.accepted:
                        found[slot] = (cand.rec, dict(elected[rkey]))    # relied on, not drawn here -> adopted at the close
                        forward_found(slot, cand.rec, elected[rkey])   # P1-7: the successor queue is derived from the elected results
                continue
            if slot in handled or slot in tentative:
                continue                # lost (the winner's result is on its way) or drawn tentatively (awaiting acceptance)
            jobs.append((slot, cand))
        dispatch(jobs)
        tally["speculative_pending"] = len(tentative)
        tally["closed"] = closed is not None
        accepted_slots = [sl for sl, c in candidates.items() if c.accepted]
        covered = all(sl in handled or L.item_scoped_key(spec.queue_key, candidates[sl].rec["arm_s"], candidates[sl].rec["item_key"]) in elected
                      for sl in accepted_slots)
        tally["covered"] = covered
        if closed is not None and covered:
            adoption = adopt_found_results(vol, found=found, spec=spec, attempt_id=attempt_id, call_id=call_id, shard=shard,
                                           attempt_root=attempt_root, adopt_into=adopt_into, evidence_dir=evidence_dir,
                                           evidence_on=evidence_dir is not None, retries=adopt_retries, poll_s=poll_s, sleep=sleep, log=log)
            tally["adopted"], tally["foreign_results"], tally["adoption_skipped"] = adoption["adopted"], len(adoption["foreign"]), adoption["skipped"]
            if next_producer is not None:
                next_producer.adopted = list(adoption["adopted"])      # the inventory lists the adoptions
            log(f"[rung-redraw] {spec.ladder_root}: CLOSED and covered — {tally}")
            return tally
        if deadline_s is not None and clock() - t0 >= deadline_s:
            log(f"[rung-redraw] {spec.ladder_root}: deadline after {tally['rounds']} rounds — {tally}")
            return tally
        if max_rounds is not None and tally["rounds"] >= max_rounds:
            return tally
        sleep(poll_s)


# --------------------------------------------------------------------------- #
# the assembly: loading a CLOSED queue's elected results from a ladder root      #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Assembly:
    """A CLOSED queue as the assembly reads it: the spec rebuilt from ``CLOSED.json``, the accepted producer
    attempts, a :class:`seedrep_ladder.LocalVolume` rooted so the records' volume-relative paths resolve, and the
    results snapshot."""

    ladder_root: Path
    vol: Any
    spec: Any
    closed: dict
    results: Any
    accepted: dict[str, str]          # {shard: accepted attempt_id} from CLOSED.json.producers (never the live Dict)

    @property
    def accepted_attempts(self) -> tuple[str, ...]:
        return tuple(self.accepted.values())

    def arm_slots(self) -> list[str]:
        names = set()
        for sub in ("materialised", "queue"):
            names.update(self.vol.listdir(f"{self.spec.ladder_root}/{sub}"))
        return sorted(names)

    def slot_for(self, arm: str, seed_base: int, *, store_stamp: Optional[int] = None) -> Optional[str]:
        """EXACTLY ``<arm>__s<seed_base>`` — the source store's own (arm, seed base), never discovered from the root
        (the Arditi view serves s1..s3 on one route, so one ladder root legitimately holds three seed slots of one
        arm). ``store_stamp`` is the seed base the source store itself records (a static row's
        ``decode.seed_base``, a misalign cell's ``meta.seed_base``; ``None`` where the store stamps none): a caller's
        seed base that disagrees with it refuses — the ONE place that check lives. ``None`` when the root has no
        queue / materialised entry under the slot."""
        if store_stamp is not None and int(store_stamp) != int(seed_base):
            raise L.LadderRefusal(f"{self.ladder_root}: the source store's rows carry seed_base {int(store_stamp)}, this assembly "
                                  f"was given {int(seed_base)}; refusing to read another seed's slot")
        want = L.arm_s(arm, int(seed_base))
        return want if want in self.arm_slots() else None

    def elected(self, arm_s_: str) -> dict:
        return L.load_elected(self.vol, self.results, self.spec, arm_s_)


def load_assembly(ladder_root: Path, results_path: Path) -> Assembly:
    """Open ``<ladder root>`` (``…/ladder/<leg>/<view>/<cap_next>``, mounted or pulled with its volume-relative
    prefix intact) + the results snapshot. Refuses an open queue (``CLOSED.json`` absent), a root whose path does
    not end in the queue root ``CLOSED.json`` names (the records' paths could not resolve), a snapshot of another
    Dict's shape."""
    ladder_root = Path(ladder_root).resolve()
    closed_p = ladder_root / "CLOSED.json"
    if not closed_p.is_file():
        raise L.LadderRefusal(f"{ladder_root}: no CLOSED.json — the assembly runs with the queue CLOSED (plan §7.3); "
                              "the reconciler closes it once every declared producer's accepted attempt has committed")
    closed = json.loads(closed_p.read_text(encoding="utf-8"))
    if closed.get("schema") != L.SCHEMA_CLOSED:
        raise L.LadderRefusal(f"{closed_p}: not a {L.SCHEMA_CLOSED} record")
    qroot = str(closed["queue_root"])
    as_posix = ladder_root.as_posix()
    if not as_posix.endswith("/" + qroot) and as_posix != qroot:
        raise L.LadderRefusal(f"{ladder_root} does not end in the queue root {qroot!r} its CLOSED.json names; pull the ladder "
                              "root with its volume-relative prefix intact so the records' paths resolve")
    vol_root = Path(as_posix[: -len(qroot)]) if as_posix != qroot else Path("/")
    key, _ = L.parse_key(closed["key"])
    shards = [p[0] for p in closed["producers"]]
    spec = L.QueueSpec.build(key.W7, key.wave_uuid, key.leg, key.view, closed["kind"], int(closed["cap_next"]), shards)
    if spec.ladder_root != qroot:
        raise L.LadderRefusal(f"{closed_p}: queue_root {qroot!r} is not the spec's {spec.ladder_root!r}")
    results = L.load_kv_snapshot(results_path)
    return Assembly(ladder_root=ladder_root, vol=L.LocalVolume(vol_root), spec=spec, closed=closed, results=results,
                    accepted={str(p[0]): str(p[1]) for p in closed["producers"]})


def _open_assembly(ladder_root, results, *, kind: str, cap: Optional[int] = None) -> Assembly:
    """:func:`load_assembly` + the refusal every seam starts with: the queue must be this leg kind's and, where the
    seam has a rung cap, at that cap."""
    asm = load_assembly(Path(ladder_root), Path(results))
    if asm.spec.kind != kind or (cap is not None and int(asm.spec.cap_next) != int(cap)):
        raise L.LadderRefusal(f"{ladder_root}: a {asm.spec.kind} queue at {asm.spec.cap_next}; this rung is {kind}"
                              + (f" at {int(cap)}" if cap is not None else ""))
    return asm


def _verified(elected: Mapping[str, Any], summary: dict, lookup: Callable[[str], Any], bind: Callable[[Any], dict],
              accepted: Iterable[str]):
    """The seams' shared verification loop: for every elected item, ``lookup(item_key)`` names the eligible row it
    answers (``None`` -> ``extra``, ignored with a note), ``bind(row)`` gives ``verify_elected``'s binding to the
    ACCEPTED base row (and may itself raise :class:`seedrep_ladder.ElectedMismatch`); a mismatch is ``refused``
    (the driver re-draws it). Yields ``(item_key, elected, row)`` for the survivors; the caller counts
    ``n_verified`` after its own leg-specific checks."""
    for ik, el in sorted(elected.items()):
        row = lookup(ik)
        if row is None:
            summary["extra"].append(ik)
            continue
        try:
            L.verify_elected(el, accepted_producers=accepted, **bind(row))
        except L.ElectedMismatch as e:
            summary["refused"].append(str(e))
            continue
        yield ik, el, row


def _log_refused(log: Callable[[str], Any], summary: Mapping[str, Any]) -> None:
    for r in summary["refused"]:
        log(f"[rung-redraw] REFUSED elected row: {r}")


def _preset_of(decode: Optional[Mapping[str, Any]]) -> Optional[dict]:
    """The request-body preset of a decode block — ``static_common.sampler_preset_of``, the driver's own reading."""
    if decode is None:
        return None
    from sisteer.eval.attacks.static_common import sampler_preset_of
    return sampler_preset_of(decode)


# --------------------------------------------------------------------------- #
# seam 1 — static (escalate_static.py)                                          #
# --------------------------------------------------------------------------- #


class _MaterialisedStaticTarget:
    """A ``ServedChatTarget`` stand-in for ``escalate_static.regenerate``: ``generate_detailed`` answers a
    ``(prompt, seed)`` it holds an elected, verified record for; everything else delegates to the real target.
    Every other attribute (``evidence``, ``virtual_model``, ``n_target_forwards``, ``route_info``, the clamp
    counters the manifest writer reads) is the real target's."""

    def __init__(self, target, hits: Mapping[tuple[str, int], dict], *, cap: int, preset: Optional[Mapping[str, Any]],
                 summary: dict) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_hits", dict(hits))
        object.__setattr__(self, "_cap", int(cap))
        object.__setattr__(self, "_preset", None if preset is None else dict(preset))
        object.__setattr__(self, "summary", summary)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def generate_detailed(self, prompts: Sequence[str], *, max_new_tokens: int = 512,
                          decode: Optional[Mapping[str, Any]] = None, seeds: Optional[Sequence[int]] = None,
                          labels: Optional[Sequence[tuple]] = None) -> list[dict]:
        """``ServedChatTarget.generate_detailed``'s surface, ``labels`` included (request evidence): the items
        this stand-in answers from an elected record send nothing (nothing to label); the rest go to the real target
        WITH their labels, aligned to the sub-list actually sent (an un-labelled delegate under the hook dies in
        ``require_label`` exactly like the static children). None when the caller passed none."""
        prompts = list(prompts)
        seed_list = list(seeds) if seeds is not None else [None] * len(prompts)
        if len(seed_list) != len(prompts):
            raise RuntimeError("generate_detailed: one seed per prompt")
        label_list = list(labels) if labels is not None else None
        if label_list is not None and len(label_list) != len(prompts):
            raise RuntimeError("generate_detailed: one label per prompt")
        out: list[Optional[dict]] = [None] * len(prompts)
        todo: list[int] = []
        usable = (int(max_new_tokens) == self._cap and (self._preset is None or _preset_of(decode) == self._preset))
        for i, (p, s) in enumerate(zip(prompts, seed_list)):
            hit = self._hits.get((p, None if s is None else int(s))) if usable else None
            if hit is not None:
                out[i] = copy.deepcopy(hit)
                self.summary["n_materialised"] += 1
            else:
                todo.append(i)
        if todo:
            recs = self._target.generate_detailed([prompts[i] for i in todo], max_new_tokens=max_new_tokens, decode=decode,
                                                  seeds=None if seeds is None else [seed_list[i] for i in todo],
                                                  **({} if label_list is None else {"labels": [label_list[i] for i in todo]}))
            for i, rec in zip(todo, recs):
                out[i] = rec
            self.summary["n_drawn_by_assembly"] += len(todo)
        return [r for r in out if r is not None]


def wrap_static_target(target, regen_items: Sequence[tuple[int, Mapping[str, Any]]], *, ladder_root, results, family: str,
                       defense: str, seed_base: int, cap_to: int, preset: Optional[Mapping[str, Any]],
                       log: Callable[[str], Any] = print):
    """``escalate_static.py --from-materialised <ladder root> --results <snapshot>``: wrap the served target so the
    eligible items (``regen_items`` — the driver's own at-cap − excluded set, computed as today) that have an
    elected, VERIFIED materialised record are answered from it. Verification per item: the elected
    row's queue record binds to the base row's normalised identity, the SAME seed, this rung's cap and preset, and
    an accepted producer attempt; a mismatch is refused (the item falls through to the engine and is re-drawn),
    an elected item outside the eligible set is ignored with a note, an eligible item with no elected record is
    drawn by the engine. ``seed_base`` is the SOURCE STORE's (``resolve_escalation(cfg)["seed_base"]``, the literal
    training seed of the view): the slot read is exactly ``<defense>__s<seed_base>`` (P1-3), and a regen row whose
    own ``decode.seed_base`` disagrees refuses. The summary lands on ``target.evidence["seedrep_assembly"]`` (the
    manifest carries it)."""
    asm = _open_assembly(ladder_root, results, kind="static", cap=cap_to)
    seed_base = int(seed_base)
    stamps = {int(v) for _, row in regen_items for v in [(row.get("decode") or {}).get("seed_base")] if v is not None}
    if len(stamps) > 1:
        raise L.LadderRefusal(f"{ladder_root}: the source store's rows carry several seed bases {sorted(stamps)}; refusing")
    summary = {"ladder_root": str(asm.ladder_root), "arm": defense, "seed_base": seed_base, "family": family, "cap": int(cap_to),
               "n_eligible": len(regen_items), "n_elected": 0, "n_verified": 0, "refused": [], "extra": [],
               "ambiguous": [], "n_materialised": 0, "n_drawn_by_assembly": 0}
    slot = asm.slot_for(defense, seed_base, store_stamp=next(iter(stamps), None))
    if slot is None:
        log(f"[rung-redraw] {asm.ladder_root}: no materialised draws for {L.arm_s(defense, seed_base)!r}; every eligible item is drawn here")
        summary["arm_s"] = None
        target.evidence["seedrep_assembly"] = summary
        return target
    summary["arm_s"] = slot
    elected = asm.elected(slot)
    summary["n_elected"] = len(elected)
    by_key = {L.item_key("static", family, row["goal_id"], int(row["sample_idx"])): row for _, row in regen_items}
    index: dict[tuple[str, int], str] = {}
    dup: set[tuple[str, int]] = set()
    for ik, row in by_key.items():
        pair = (row["attack_prompt"], int(row["sampling_seed"]))
        if pair in index:
            dup.add(pair)
        index[pair] = ik
    for pair in dup:                                  # two eligible items with one (prompt, seed): never guess
        index.pop(pair, None)
        summary["ambiguous"].append(list(pair))

    def bind(row):
        return dict(base_row_sha256=L.base_row_identity_sha256(
            text=row.get("response"), sampling_seed=int(row["sampling_seed"]), finish_reason=row.get("finish_reason"),
            cap=int((row.get("decode") or {}).get("cap", asm.spec.base_cap))),
            sampling_seed=int(row["sampling_seed"]), cap=int(cap_to), preset=preset)

    hits: dict[tuple[str, int], dict] = {}
    for ik, el, row in _verified(elected, summary, by_key.get, bind, asm.accepted_attempts):
        pair = (row["attack_prompt"], int(row["sampling_seed"]))
        if pair not in index:
            continue
        rec = dict(el.doc["record"])
        rec.pop("at_ceiling", None)
        rec.setdefault("sampling_seed", int(row["sampling_seed"]))
        rec["max_tokens"] = int(cap_to)
        hits[pair] = rec
        summary["n_verified"] += 1
    log(f"[rung-redraw] {asm.ladder_root}/{slot}: {len(regen_items)} eligible, {len(elected)} elected, "
        f"{summary['n_verified']} verified, {len(summary['refused'])} refused (re-drawn here), {len(summary['extra'])} extra "
        "(ignored)")
    _log_refused(log, summary)
    target.evidence["seedrep_assembly"] = summary
    return _MaterialisedStaticTarget(target, hits, cap=cap_to, preset=preset, summary=summary)


# --------------------------------------------------------------------------- #
# seam 2 — controls (served_capability_run.py --controls-cap … --from-materialised)
# --------------------------------------------------------------------------- #


def _completion_from_record(doc: Mapping[str, Any]) -> dict:
    """A chat completion rebuilt from a materialised record when the raw response was not kept."""
    rec = doc["record"]
    return {"id": doc.get("completion_id"), "choices": [{"index": 0, "message": {"role": "assistant", "content": rec["text"]},
                                                          "finish_reason": rec["finish_reason"]}],
            "usage": {"prompt_tokens": rec["prompt_tokens"], "completion_tokens": rec["completion_tokens"],
                      "total_tokens": int(rec["prompt_tokens"]) + int(rec["completion_tokens"])}}


def wrap_controls_post(post, *, ladder_root, results, cell_id: str, seed_base: int, cap: int, decode: Optional[Mapping[str, Any]],
                       specs: Mapping[str, Any], sources: Mapping[str, Any], trial: int,
                       log: Callable[[str], Any] = print):
    """``served_capability_run.py --controls-cap N --escalate-from-root R --from-materialised <ladder root> --results
    <snapshot>``: wrap the proxy ``post`` so a re-draw request whose item has an elected, VERIFIED materialised
    record is answered from it (the raw completion, or one rebuilt from the record); every other request goes to
    the engine. The item is located from the request's user prompt through ``specs[control].rows`` and its
    ACCEPTED base row is ``sources[control].meta[item_id]`` (the source rung's bound meta sidecar); the identity
    bound is ``(text_sha256, sampling_seed, finish_reason, cap)``, the seed and cap must equal the request's, the
    request's sampler keys this rung's ``decode``. ``seed_base`` is the source gen block's (``gen["seed_base"]``):
    the slot read is exactly ``<cell_id>__s<seed_base>`` (P1-3)."""
    asm = _open_assembly(ladder_root, results, kind="controls", cap=cap)
    preset = _preset_of(decode)
    summary = {"ladder_root": str(asm.ladder_root), "arm": cell_id, "seed_base": int(seed_base), "cap": int(cap), "n_elected": 0,
               "n_verified": 0, "refused": [], "extra": [], "n_materialised": 0, "n_drawn_by_assembly": 0, "arm_s": None}
    slot = asm.slot_for(cell_id, int(seed_base))
    if slot is None:
        log(f"[rung-redraw] {asm.ladder_root}: no materialised draws for {L.arm_s(cell_id, int(seed_base))!r}; every request goes to the engine")
        wrap_controls_post.last_summary = summary
        return post
    summary["arm_s"] = slot
    elected = asm.elected(slot)
    summary["n_elected"] = len(elected)
    regen = {control: set(getattr(src, "regen_ids", ())) for control, src in sources.items()}
    by_key: dict[str, tuple[str, str, str]] = {}
    for control, spec in specs.items():
        for r in spec.rows:
            by_key[L.item_key("controls", control, r["item_id"], int(trial))] = (control, r["item_id"], r["prompt"])

    def lookup(ik):
        """(control, item_id, prompt, the source's meta row) — None unless this rung regenerates the item."""
        loc = by_key.get(ik)
        if loc is None:
            return None
        control, iid, prompt = loc
        src = sources.get(control)
        meta = src.meta.get(iid) if src is not None else None
        if meta is None or iid not in regen.get(control, ()):
            return None                                 # elected but not in this rung's regenerate set
        return control, iid, prompt, meta

    def bind(row):
        meta = row[3]
        return dict(base_row_sha256=L.base_row_identity_sha256(
            text=None, text_sha256=meta["text_sha256"], sampling_seed=int(meta["sampling_seed"]),
            finish_reason=meta.get("finish_reason"), cap=int(meta["max_tokens"])),
            sampling_seed=int(meta["sampling_seed"]), cap=int(cap), preset=preset)

    hits: dict[tuple[str, int], dict] = {}
    for ik, el, (control, iid, prompt, meta) in _verified(elected, summary, lookup, bind, asm.accepted_attempts):
        response = el.doc.get("response")
        if not isinstance(response, Mapping) or "choices" not in response:
            response = _completion_from_record(el.doc)
        hits[(prompt, int(meta["sampling_seed"]))] = copy.deepcopy(response)
        summary["n_verified"] += 1
    log(f"[rung-redraw] {asm.ladder_root}/{slot}: {len(elected)} elected, {summary['n_verified']} verified, "
        f"{len(summary['refused'])} refused (re-drawn here), {len(summary['extra'])} extra (ignored)")
    _log_refused(log, summary)
    wrap_controls_post.last_summary = summary

    async def wrapped(req, headers=None):
        try:
            prompt = req["messages"][-1]["content"]
            seed = req.get("seed")
        except (KeyError, IndexError, TypeError):
            prompt, seed = None, None
        hit = None
        if prompt is not None and seed is not None and int(req.get("max_tokens", -1)) == int(cap):
            sent = {k: req[k] for k in (preset or {}) if k in req}
            if preset is None or sent == preset:
                hit = hits.get((prompt, int(seed)))
        if hit is not None:
            summary["n_materialised"] += 1
            return copy.deepcopy(hit)
        summary["n_drawn_by_assembly"] += 1
        return await (post(req) if headers is None else post(req, headers=headers))

    return wrapped


wrap_controls_post.last_summary = None


# --------------------------------------------------------------------------- #
# seam 3 — misalign (sisteer.capeval.misalign.runner --rerun-truncated-from … --from-materialised)
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict]:
    """The runner's responses.jsonl: split on ``\\n`` ONLY (``runner.split_jsonl`` — ``str.splitlines`` also breaks on
    U+2028 / NEL a model may emit inside a string); an absent file is an empty source."""
    from sisteer.capeval.misalign.runner import split_jsonl

    if not path.exists():
        return []
    return [json.loads(l) for l in split_jsonl(path.read_text(encoding="utf-8")) if l.strip()]


def seed_misalign_root(out_root, *, ladder_root, results, src_root, cap: int, only: Mapping[str, Iterable[tuple[str, int]]],
                       cell_dirname: Callable[[str], str], read_done: Callable[[Path], set], seed_base: Optional[int] = None,
                       log: Callable[[str], Any] = print) -> dict:
    """``python -m sisteer.capeval.misalign.runner --rerun-truncated-from SRC --out NEW --from-materialised <ladder
    root> --results <snapshot>``: BEFORE the runner draws, write every elected, VERIFIED materialised row of the
    selected ``(item_id, sample_idx)`` pairs into ``NEW/<cell>/responses.jsonl`` (the runner's exact row shape,
    :data:`MISALIGN_ROW_KEYS`; provenance beside it in ``materialised_seeded.jsonl``), so the runner's own
    ``read_done`` skips them and draws only the rest. The ACCEPTED base row is the SOURCE root's row of the same
    pair; its identity bound is ``(response text, seed, finish_reason, the source meta's max_tokens)``; the elected
    row must be this cell's and this pair's. A mismatch is refused and left for the runner to draw. ``seed_base`` is
    the source store's view seed (the runner's ``--seed-base``); when not given it is read from the SOURCE
    cell's ``meta.json`` (``seed_base``, the identity field the runner stamps) and a source that names none refuses — the
    slot read is exactly ``<cell dir>__s<seed_base>`` (P1-3)."""
    asm = _open_assembly(ladder_root, results, kind="misalign", cap=cap)
    out_root, src_root = Path(out_root), Path(src_root)
    summary: dict[str, Any] = {"ladder_root": str(asm.ladder_root), "cap": int(cap), "seed_base": seed_base, "cells": {}}
    for cell_id, pairs in sorted(only.items()):
        cdir = cell_dirname(cell_id)
        want = {(str(i), int(s)) for i, s in pairs}
        cs = {"arm_s": None, "n_selected": len(want), "n_elected": 0, "n_verified": 0, "n_seeded": 0,
              "n_already_present": 0, "refused": [], "extra": []}
        summary["cells"][cell_id] = cs
        meta_p = src_root / cdir / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        s_cell = seed_base if seed_base is not None else meta.get("seed_base")
        if s_cell is None:
            raise L.LadderRefusal(f"{src_root / cdir}: no seed_base — neither given (--seed-base) nor in the source meta.json; "
                                  "the assembly cannot pick this cell's seed slot")
        cs["seed_base"] = int(s_cell)
        slot = asm.slot_for(cdir, int(s_cell), store_stamp=meta.get("seed_base"))
        if slot is None:
            log(f"[rung-redraw] {asm.ladder_root}: no materialised draws for {L.arm_s(cdir, int(s_cell))!r} (cell {cell_id!r})")
            continue
        cs["arm_s"] = slot
        elected = asm.elected(slot)
        cs["n_elected"] = len(elected)
        src_rows = {(str(r["item_id"]), int(r["sample_idx"])): r for r in _read_jsonl(src_root / cdir / "responses.jsonl")}
        src_cap = (meta.get("params") or {}).get("max_tokens")
        if src_cap is None:
            src_cap = asm.spec.base_cap
        by_key = {L.item_key("misalign", i, s): (i, s) for i, s in want}
        dest = out_root / cdir / "responses.jsonl"
        done = read_done(dest)

        def bind(pair):
            src = src_rows.get(pair)
            if src is None:
                raise L.ElectedMismatch(f"{L.item_key('misalign', *pair)}: the source root has no row for {pair}")
            seed = L._opt_int(src.get("seed"))
            return dict(base_row_sha256=L.base_row_identity_sha256(text=src.get("response"), sampling_seed=seed,
                                                                   finish_reason=src.get("finish_reason"), cap=int(src_cap)),
                        sampling_seed=seed, cap=int(cap), preset=None)

        seeded_rows: list[str] = []
        prov: list[str] = []
        for ik, el, pair in _verified(elected, cs, by_key.get, bind, asm.accepted_attempts):
            row = el.doc.get("record") or {}
            missing = [k for k in MISALIGN_ROW_KEYS if k not in row]
            if missing or str(row.get("item_id")) != pair[0] or int(row.get("sample_idx", -1)) != pair[1] \
                    or row.get("cell_id") != cell_id:
                cs["refused"].append(f"{ik}: the materialised record is not this cell's runner row for {pair} "
                                     f"(missing {missing}, cell {row.get('cell_id')!r})")
                continue
            cs["n_verified"] += 1
            if pair in done:
                cs["n_already_present"] += 1
                continue
            seeded_rows.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
            prov.append(json.dumps({"item_id": pair[0], "sample_idx": pair[1], "attempt_id": el.value.get("attempt_id"),
                                    "request_id": el.value.get("request_id"), "completion_id": el.value.get("completion_id"),
                                    "path": el.path, "text_sha256": el.value.get("text_sha256")}))
            done.add(pair)
        if seeded_rows:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "a", encoding="utf-8") as f:
                f.write("\n".join(seeded_rows) + "\n")
                f.flush()
                os.fsync(f.fileno())
            with open(dest.parent / "materialised_seeded.jsonl", "a", encoding="utf-8") as f:
                f.write("\n".join(prov) + "\n")
        cs["n_seeded"] = len(seeded_rows)
        log(f"[rung-redraw] {asm.ladder_root}/{slot}: {cs['n_selected']} selected, {cs['n_elected']} elected, "
            f"{cs['n_verified']} verified, {cs['n_seeded']} seeded into {dest}, {len(cs['refused'])} refused (drawn by the "
            f"runner), {len(cs['extra'])} extra (ignored)")
        _log_refused(log, cs)
    return summary


# --------------------------------------------------------------------------- #
# seam 4 — GSM8K (sisteer.capeval.tiers.inspect_rung.subset_from_materialised_queue)
# --------------------------------------------------------------------------- #


def gsm8k_subset_from_queue(ladder_root, results, *, source_out_root, arm: str, seed_base: int, task: Optional[str] = None,
                            classify_store: Optional[Callable[..., tuple[dict, Optional[dict]]]] = None,
                            log: Callable[[str], Any] = print, **classify_kw) -> tuple[dict, Optional[dict], dict]:
    """The GSM8K assembly: ``(record, subset | None, verdict)``. The eligible set is ``inspect_rung.classify_store``'s
    (recomputed from the filed outcomes as today — the ONE at-cap authority); the CLOSED queue's accepted entries
    OF THIS SOURCE STORE's slot — exactly ``<arm>__s<seed_base>`` (the store's cell id and its ``--inspect-seed-base``;
    one queue serves every arm the base container ran, each with its own at-cap set) — are cross-checked
    against the accepted source store's outcomes: an entry naming an item the source did not file at cap REFUSES (a
    forged / foreign entry), an entry the eligible set excludes (a loop) is ``extra`` (noted), an eligible item the
    queue lacks is ``missing_from_queue`` (noted; the driver draws the whole subset). The subset returned is the
    adapter's own, unchanged, for ``--item-subset``."""
    from sisteer.capeval.tiers import inspect_rung as IR   # the adapter (its own constants; it is this seam's caller)

    asm = _open_assembly(ladder_root, results, kind="gsm8k")
    classify_store = classify_store or IR.classify_store
    record, subset = classify_store(Path(source_out_root), task=task or IR.GSM8K_TASK, **classify_kw)
    slot = L.arm_s(arm, int(seed_base))
    entries = {sl: rec for sl, rec in L.accepted_queue_entries(asm.vol, None, asm.spec, accepted=asm.accepted).items()
               if rec["arm_s"] == slot}
    queued = {str(rec["identity"]["item_id"]) for rec in entries.values()}
    at_cap = {str(r["item_id"]) for r in record.get("item_facts", []) if r.get("at_cap")}
    foreign = sorted(queued - at_cap)
    if foreign:
        raise L.LadderRefusal(f"{asm.ladder_root}: {len(foreign)} queued item(s) the accepted source store never filed at "
                              f"cap (first: {foreign[:3]}); refusing the queue as this store's")
    eligible = set(map(str, record.get(IR.RECORD_FIELD_ELIGIBLE_IDS, [])))
    verdict = {"exit": record.get("exit"), "arm_s": slot, "n_eligible": len(eligible), "n_queued": len(queued),
               "missing_from_queue": sorted(eligible - queued), "extra_in_queue": sorted(queued - eligible),
               "subset_ids": (list(subset["ids"]) if subset else None), "cap": int(asm.spec.cap_next)}
    if subset is not None and int(subset.get("cap", -1)) != int(asm.spec.cap_next):
        raise L.LadderRefusal(f"{asm.ladder_root}: the adapter's subset is at cap {subset.get('cap')}, this queue's rung is "
                              f"{asm.spec.cap_next}")
    log(f"[rung-redraw] {asm.ladder_root}/{slot}: exit {verdict['exit']}, {len(eligible)} eligible, {len(queued)} queued, "
        f"{len(verdict['missing_from_queue'])} missing from the queue, {len(verdict['extra_in_queue'])} extra in it")
    return record, subset, verdict


# --------------------------------------------------------------------------- #
# CLI: consume | reconcile | snapshot                                           #
# --------------------------------------------------------------------------- #


def _csv(text: Optional[str]) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _spec_from_args(a) -> Any:
    return L.QueueSpec.build(a.W7, a.wave_uuid, a.leg, a.view, a.kind, int(a.cap_next), _csv(a.producers))


class _JsonKV(L.MemoryKV):
    """A JSON-file-backed :class:`seedrep_ladder.MemoryKV` for smoke runs without Modal (one process; NOT a
    distributed store): loaded at construction, written back by :meth:`flush`."""

    def __init__(self, name: str, path: Path) -> None:
        super().__init__(name)
        self.path = Path(path)
        if self.path.exists():
            for k, v in json.loads(self.path.read_text()).items():
                self.put(k, v)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(self.items()), indent=1, sort_keys=True))


def _kv(a, purpose: str, bridge: Optional[SdkBridge] = None):
    """The wave's Dict for ``purpose``: a JSON stand-in (``--memory-kv-dir``), the bridge's :class:`BridgeKV` (a rung container: fx1), or
    the direct, CHECKED SDK handle (the grading host reconciler / snapshot under a venv that installs it)."""
    name = L.dict_name(a.W7, a.wave_uuid, purpose)
    if a.memory_kv_dir:
        return _JsonKV(name, Path(a.memory_kv_dir) / f"{name}.json")
    if bridge is not None:
        return bridge.kv(name)
    return L.ModalDictKV.from_name(name)


def _sdk_bridge(a) -> Optional[SdkBridge]:
    """The consumer's SDK route (fx1): ``$SEEDREP_MODAL_SDK_PYTHON`` (set by the served app on the consumer child) -> ONE
    :class:`SdkBridge`, handshaken here; unset -> None and every handle comes from the direct, CHECKED import (``L.modal_sdk``: the
    grading host reconciler under a venv that carries the SDK, or a test with a stubbed ``modal``). A smoke without Modal (``--memory-kv-dir``)
    never opens one."""
    python = os.environ.get(SDK_PYTHON_ENV, "").strip()
    if a.memory_kv_dir or not python:
        return None
    bridge = SdkBridge(python)
    print(f"[rung-redraw] Modal SDK bridge: {bridge.hello}", flush=True)
    return bridge


def _flush(*kvs) -> None:
    for kv in kvs:
        if isinstance(kv, _JsonKV):
            kv.flush()


#: the consumer's exit when the queue was not CLOSED and covered by the deadline / round cap (P1-1): the container
#: reports INCOMPLETE and does NOT finish its producer side — a rung that has not covered its queue is not a producer
EXIT_INCOMPLETE = 3


def build_parser() -> argparse.ArgumentParser:
    """The rung consumer's CLI (the driver composes argv against THIS parser; kept stable)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--vol-root", default=None, help="consume: the MOUNTED volume root (/vol); reconcile: a local pull of it")
    common.add_argument("--volume", default=None, metavar="NAME",
                        help="the Modal Volume (e.g. twosi-fits): consume binds its real commit/reload onto the mount; reconcile "
                             "reads/writes it through the Volume API (RemoteVolume). Required unless --memory-kv-dir (a smoke without Modal)")
    common.add_argument("--W7", required=True)
    common.add_argument("--wave-uuid", required=True)
    common.add_argument("--leg", required=True)
    common.add_argument("--view", required=True)
    common.add_argument("--kind", required=True, choices=sorted(L.LADDERS))
    common.add_argument("--cap-next", required=True, type=int)
    common.add_argument("--producers", required=True, help="comma-separated declared producer shards of this queue")
    common.add_argument("--memory-kv-dir", default=None, help="tests / smokes without Modal: JSON-backed KV stand-ins")
    c = sub.add_parser("consume", parents=[common], help="the rung consumer loop (inside a run_rung container)")
    c.add_argument("--attempt-id", required=True)
    c.add_argument("--call-id", required=True)
    c.add_argument("--proxy-url", required=True)
    c.add_argument("--api-key", default="petri-proxy")
    c.add_argument("--width", "--concurrency", dest="width", type=int, required=True,
                   help="draws in flight at once = the ATTESTED rung width S_rung (WS2-D's --concurrency S); never 1 by default")
    c.add_argument("--rung-shard", default=None, help="this rung container's shard label (its producer id for the next queue)")
    c.add_argument("--rung-shards", default=None, help="every declared shard of this rung (the next queue's producers)")
    c.add_argument("--attempt-root", default=None, help="this rung attempt's root (queue_inventory.json lands there)")
    c.add_argument("--speculative", action="store_true",
                   help="also draw live entries of not-yet-accepted producers, tentatively (promoted + elected only once accepted)")
    c.add_argument("--poll-s", type=float, default=30.0)
    c.add_argument("--deadline-s", type=float, default=None)
    c.add_argument("--finish-producer", action="store_true",
                   help="after the loop, ONLY if CLOSED and covered: inventory -> acceptance -> CLOSED marker for the next queue")
    c.add_argument("--shard", default=None,
                   help="this container's SHARD label — its slot of the rung (default: --rung-shard). A later execution of the slot adopts "
                        "the elected results' evidence of an earlier one (the third review's P1-1); pass it at the ceiling rung too")
    c.add_argument("--attempt-root-local", default=None, metavar="DIR",
                   help="the container-LOCAL attempt root the adopted request pipeline is merged into (WS2-D: /workspace/attempt); default: "
                        "the nearest ancestor of SISTEER_REQUEST_EVIDENCE_DIR holding ATTEMPT.json")
    c.add_argument("--adopt-retries", type=int, default=3,
                   help="how many times (--poll-s apart) to re-read a predecessor's root whose evidence does not yet cover its elected "
                        "results — its TreeSyncer may still be shipping — before refusing")
    c.add_argument("--no-request-evidence", action="store_true",
                   help="draw WITHOUT WS1-D′'s request-evidence client. A production rung (--volume) REFUSES to start unless "
                        "SISTEER_REQUEST_EVIDENCE_DIR names its attempt root or this flag is given (its store would be unauditable)")
    r = sub.add_parser("reconcile", parents=[common], help="one reconciler pass (the grading host): exactly one of --volume / --vol-root")
    r.add_argument("--out", default=None)
    s_ = sub.add_parser("snapshot", help="snapshot a Dict to JSON (the assembly's --results)")
    s_.add_argument("--W7", required=True)
    s_.add_argument("--wave-uuid", required=True)
    s_.add_argument("--purpose", default="results", choices=L.PURPOSES)
    s_.add_argument("--out", required=True)
    s_.add_argument("--memory-kv-dir", default=None)
    return ap


def _consume_volume(ap: argparse.ArgumentParser, a, bridge: Optional[SdkBridge] = None) -> Any:
    """The rung container's mount with the Volume's REAL commit / reload bound (without them nothing a consumer
    writes is durable and nothing a producer commits becomes visible); a plain directory only for a smoke without
    Modal (``--memory-kv-dir``). The verbs come over the :class:`SdkBridge` when the app named an SDK interpreter (the
    driver venv has no SDK and a bare ``import modal`` here answered with the repo's scripts/modal); otherwise from the direct,
    CHECKED import, which refuses that shape by name instead of dying on ``modal.Volume``."""
    if not a.vol_root:
        ap.error("consume needs --vol-root (the mounted volume root, /vol)")
    if a.memory_kv_dir:
        return L.LocalVolume(Path(a.vol_root))
    if not a.volume:
        ap.error("consume needs --volume <name> (the mounted Modal Volume, whose commit/reload are bound) unless --memory-kv-dir")
    if bridge is not None:
        commit, reload = bridge.volume_verbs(a.volume)
        return L.LocalVolume(Path(a.vol_root), commit=commit, reload=reload)
    v = L.modal_sdk("Volume").Volume.from_name(a.volume)
    return L.LocalVolume(Path(a.vol_root), commit=v.commit, reload=v.reload)


def _reconcile_volume(ap: argparse.ArgumentParser, a) -> Any:
    """The grading host reconciler's view: the Volume API (``--volume``, :class:`seedrep_ladder.RemoteVolume`) or a LOCAL
    PULL (``--vol-root``; as fresh as the pull) — exactly one."""
    if bool(a.volume) == bool(a.vol_root):
        ap.error("reconcile takes exactly one of --volume <name> (the Volume API) / --vol-root <a local pull>")
    return L.RemoteVolume.from_name(a.volume) if a.volume else L.LocalVolume(Path(a.vol_root))


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        kv = _kv(a, a.purpose)
        L.snapshot_kv(kv, Path(a.out), name=L.dict_name(a.W7, a.wave_uuid, a.purpose))
        print(a.out)
        return 0
    spec = _spec_from_args(a)
    if a.cmd == "reconcile":
        vol = _reconcile_volume(ap, a)
        accepted, claims, results = _kv(a, "accepted"), _kv(a, "claims"), _kv(a, "results")
        lease = None if a.memory_kv_dir else L.modal_lease_state     # registry_volume._lease_state's rule, verbatim
        status = L.reconcile_queue(vol, accepted, spec, claims_kv=claims, results_kv=results, lease_state=lease)
        _flush(accepted, claims, results)
        out = json.dumps(status, indent=1, default=str)
        if a.out:
            Path(a.out).write_text(out)
        print(out)
        return 0
    bridge = _sdk_bridge(a)
    try:
        return _consume_main(ap, a, spec, bridge)
    finally:
        if bridge is not None:
            bridge.close()


def _consume_main(ap: argparse.ArgumentParser, a, spec, bridge: Optional[SdkBridge]) -> int:
    vol = _consume_volume(ap, a, bridge)
    # the request-evidence side, after the Volume argument checks: a production rung without it is refused before
    # any Dict is opened or a draw is made
    evidence = None if a.no_request_evidence else RungEvidence.from_env(leg=a.leg, attempt_id=a.attempt_id)
    if evidence is None and not a.no_request_evidence and not a.memory_kv_dir:
        ap.error(f"consume: {EVIDENCE_ENV_DIR} is not set — a production rung records every draw for the request-evidence audit "
                 "(plan §8 S3, §9.1); point it at this attempt root, or pass --no-request-evidence for a smoke")
    accepted, claims, results = _kv(a, "accepted", bridge), _kv(a, "claims", bridge), _kv(a, "results", bridge)
    next_producer = None
    nxt = spec.next_spec(_csv(a.rung_shards or a.rung_shard)) if a.rung_shard else None
    if nxt is not None:
        ref = next(p for p in nxt.producers if p.shard == a.rung_shard)
        root = a.attempt_root or L.attempt_root_for(a.W7, leg=a.leg, view=a.view, stage=str(spec.cap_next), shard=a.rung_shard, attempt_id=a.attempt_id)
        next_producer = L.Producer(vol, nxt, ref, a.attempt_id, root)
    shard = a.shard or a.rung_shard
    attempt_root_vol = a.attempt_root or (L.attempt_root_for(a.W7, leg=a.leg, view=a.view, stage=str(spec.cap_next), shard=shard, attempt_id=a.attempt_id)
                                          if shard else None)      # this rung container's own root: rungs/<its cap>/shard<NN>/ (the app's spelling)
    adopt_into = (Path(a.attempt_root_local) if a.attempt_root_local
                  else (L.attempt_root_marker(evidence.out_dir) if evidence is not None else None))
    tally = consume(vol, accepted_kv=accepted, claims_kv=claims, results_kv=results, spec=spec, attempt_id=a.attempt_id,
                    call_id=a.call_id, draw=http_draw_fn(a.proxy_url, api_key=a.api_key, evidence=evidence), next_producer=next_producer,
                    speculative=a.speculative, width=a.width, poll_s=a.poll_s, deadline_s=a.deadline_s, shard=shard, attempt_root=attempt_root_vol,
                    adopt_into=adopt_into, evidence_dir=(evidence.out_dir if evidence is not None else None), adopt_retries=a.adopt_retries)
    if evidence is not None:
        tally["evidence"] = evidence.summary()
    if bridge is not None:
        tally["sdk_bridge"] = dict(bridge.hello)                 # fx1: which interpreter served the SDK verbs, and which SDK it carried
    complete = bool(tally["closed"] and tally["covered"])
    if complete and a.finish_producer and next_producer is not None:
        tally["producer"] = L.finish_producer(next_producer, accepted)
    _flush(accepted, claims, results)
    print(json.dumps(tally, indent=1, default=str))
    if not complete:
        print(f"[rung-redraw] INCOMPLETE: {spec.ladder_root} closed={tally['closed']} covered={tally['covered']} after "
              f"{tally['rounds']} round(s); the producer side was NOT finished (exit {EXIT_INCOMPLETE})", file=sys.stderr, flush=True)
        return EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    sys.exit(main())
