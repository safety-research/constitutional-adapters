"""The sampled over-refusal controls' DURABLE per-draw event stream, and the controls ladder's
ancestry rule stated once.

WHY A STREAM. The controls writer (``sisteer.eval.sampled_controls._run_judged_served``) only
iterates its results after the fan-out returns, and the fan-out (``proxy_engine.bounded_gather``)
returns the complete map after ALL requests finish — so nothing at the writer can see completion
order, and a ladder that wants to re-draw at-cap items on a concurrently running rung engine has
nothing to tail until the whole base leg is done. ``bounded_gather`` therefore takes an additive
``on_done`` callback fired per thunk in completion order, the driver's ``_fanout`` hands it a
:class:`ControlsEventSink`, and the sink appends ONE event per completed draw to
``<cell_dir>/controls_events.jsonl`` — fsync per event — the moment the response lands. The
ladder tailer reads that file; the store's own bytes (spool, meta sidecar, gen_stats.json,
manifest) are untouched.

WHAT AN EVENT SAYS (:data:`CONTROLS_EVENT_FIELDS`): which draw (``cell_id``, ``control``, ``cap``,
``escalated_from_cap`` = the immediate predecessor rung of a rung store, null on a base store,
``item_id``, ``sampling_seed`` AS SENT), what came back (``finish_reason``, ``at_cap`` =
``finish_reason == "length"``, the usage counts, ``text_sha256`` of the text — never the text),
and how it binds to the wire (``request_id`` from the request-evidence client when one is wired,
``completion_id`` = the response's ``id``). A draw whose request FAILED is an event too
(``error`` set, every response field null): a tailer must see the hole, not infer it from
silence. ``seq`` is monotonic per file and continues across a resume, so a re-drawn item (the
process died between its event and its spool append) yields a second event; readers take the
LAST event per (control, item_id) — :func:`latest_by_item`.

THE LADDER'S ANCESTRY RULE, stated once for the tailer and the assembly: the controls ladder is
``sampled_controls.CONTROLS_CAP_LADDER`` (512 -> 2,048 -> 8,192 -> 16,384), a rung's source is
the IMMEDIATELY preceding rung (``_apply_escalation`` refuses anything else), its store records
``escalated_from = {cap: <that rung>, root}``, and its noise-floor sample is drawn from that
COMPLETED predecessor's below-cap population (``_load_source_control`` -> ``floor_sample_ids``;
a source with no ``gen_stats.json`` — a run that never completed — is refused). The helpers here
(:func:`ladder_transitions`, :func:`predecessor_cap`, :func:`check_escalated_from`) let a reader
assert a rung store against that rule without re-deriving it.

Stdlib + ``sampled_controls`` only; nothing here imports torch or the serving tree.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sisteer.eval.sampled_controls import CONTROLS_CAP_LADDER, _text_sha

#: the stream's file name, beside the store's ``gen_stats.json``
CONTROLS_EVENTS_FILENAME = "controls_events.jsonl"
CONTROLS_EVENT_VERSION = 1
#: the one event kind today: a completed (or failed) draw request
CONTROLS_EVENT_KIND_DRAW = "draw"
#: every event's keys, in write order (ids, numbers and hashes only — never text)
CONTROLS_EVENT_FIELDS: Tuple[str, ...] = (
    "v", "kind", "seq", "ts_utc", "cell_id", "control", "cap", "escalated_from_cap",
    "item_id", "sampling_seed", "finish_reason", "at_cap", "completion_tokens", "prompt_tokens",
    "text_sha256", "request_id", "completion_id", "error")
#: the driver flag that turns the stream on (default off: no file, today's store byte for byte)
CONTROLS_EVENTS_FLAG = "--controls-events"
#: the gen_stats.json key a store that wrote the stream carries per control (the file's path)
GEN_STATS_EVENTS_KEY = "controls_events"


class ControlsEventsError(RuntimeError):
    """A malformed event, a torn stream a reader met, a sink bound to the wrong cap, a rung
    whose ``escalated_from`` breaks the ancestry rule."""


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


#: the request-evidence hook point (``request_evidence_client``): given the request body
#: and the response (or exception), the request id the client minted for that request — None when
#: no client is wired, which is the default
RequestIdOf = Callable[[Mapping[str, Any], Any], Optional[str]]


class ControlsEventWriter:
    """The append-only, fsync-per-event writer of one store's ``controls_events.jsonl``.

    Opened on an EXISTING file it continues the ``seq`` numbering after the last complete event;
    a torn final line (a crash mid-append) is truncated away exactly as the misalign runner's
    ``read_done`` treats its responses file — that event never completed, and its draw either
    re-runs (no spool row yet) or is re-published by the resume's re-draw; a corrupt EARLIER line
    is file damage and refuses."""

    def __init__(self, path: os.PathLike | str, *, cell_id: str) -> None:
        self.path = Path(path)
        self.cell_id = str(cell_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = self._resume_seq()
        self.n_written = 0

    def _resume_seq(self) -> int:
        if not self.path.exists():
            return 0
        text = self.path.read_text(encoding="utf-8")
        if not text:
            return 0
        lines = text.split("\n")
        if lines[-1] == "":
            lines = lines[:-1]
        last_seq = -1
        torn_at: Optional[int] = None
        for i, line in enumerate(lines):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not text.endswith("\n"):
                    torn_at = i
                    break
                raise ControlsEventsError(
                    f"{self.path}: corrupt event at line {i + 1} (not a torn tail) — file damage; "
                    "refusing to append to it")
            last_seq = int(ev["seq"])
        if torn_at is not None:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text("".join(l + "\n" for l in lines[:torn_at]), encoding="utf-8")
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            print(f"[controls-events] {self.path}: truncated a torn final line (interrupted "
                  "append); that draw's event is re-published by its re-draw", file=sys.stderr)
        return last_seq + 1

    def append(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        """Stamp ``v`` / ``kind`` / ``seq`` / ``ts_utc`` / ``cell_id`` and append ONE line, fsynced.
        ``event`` must carry exactly the other fields of :data:`CONTROLS_EVENT_FIELDS`."""
        stamped = {"v": CONTROLS_EVENT_VERSION, "kind": CONTROLS_EVENT_KIND_DRAW, "seq": self._seq,
                   "ts_utc": _utc_now(), "cell_id": self.cell_id}
        own = set(stamped)
        want = [k for k in CONTROLS_EVENT_FIELDS if k not in own]
        got = set(event)
        if got != set(want):
            raise ControlsEventsError(
                f"{self.path}: event fields {sorted(got ^ set(want))} differ from the contract "
                f"{want} (the writer stamps {sorted(own)})")
        row = {k: (stamped[k] if k in stamped else event[k]) for k in CONTROLS_EVENT_FIELDS}
        line = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        self.n_written += 1
        return row

    def sink(self, *, control: str, cap: int, escalated_from_cap: Optional[int] = None,
             request_id_of: Optional[RequestIdOf] = None) -> "ControlsEventSink":
        """The ``on_done`` adapter for ONE control's fan-out at ONE cap."""
        return ControlsEventSink(self, control=control, cap=cap,
                                 escalated_from_cap=escalated_from_cap, request_id_of=request_id_of)


class ControlsEventSink:
    """Turns a fan-out completion ``(item_id, request payload, outcome)`` into one appended
    event. Bound to one (control, cap): a payload whose ``max_tokens`` is not this cap is a
    wiring error (the sink was set for the wrong control) and REFUSES rather than mislabel."""

    def __init__(self, writer: ControlsEventWriter, *, control: str, cap: int,
                 escalated_from_cap: Optional[int], request_id_of: Optional[RequestIdOf]) -> None:
        self.writer = writer
        self.control = str(control)
        self.cap = int(cap)
        self.escalated_from_cap = None if escalated_from_cap is None else int(escalated_from_cap)
        self.request_id_of = request_id_of

    def publish(self, item_id: Any, payload: Optional[Mapping[str, Any]], outcome: Any) -> Dict[str, Any]:
        if payload is None:
            raise ControlsEventsError(f"{self.writer.path}: no request payload for item {item_id!r} — "
                                      "the sink must be handed the fan-out's own work list")
        if int(payload.get("max_tokens", -1)) != self.cap:
            raise ControlsEventsError(
                f"{self.writer.path}: item {item_id!r} was requested at max_tokens="
                f"{payload.get('max_tokens')!r} but this sink is bound to {self.control} @ cap "
                f"{self.cap} — the sink is set for the wrong control; refusing to mislabel the draw")
        ev: Dict[str, Any] = {"control": self.control, "cap": self.cap,
                              "escalated_from_cap": self.escalated_from_cap,
                              "item_id": str(item_id), "sampling_seed": payload.get("seed"),
                              "finish_reason": None, "at_cap": None, "completion_tokens": None,
                              "prompt_tokens": None, "text_sha256": None, "request_id": None,
                              "completion_id": None, "error": None}
        if isinstance(outcome, BaseException):
            ev["error"] = f"{type(outcome).__name__}: {outcome}"
        else:
            resp = outcome if isinstance(outcome, Mapping) else {}
            try:
                choice = resp["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                fr = choice.get("finish_reason")
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                ev["error"] = f"malformed response: {e!r}"
            else:
                usage = resp.get("usage") if isinstance(resp.get("usage"), Mapping) else {}
                ev.update({"finish_reason": fr, "at_cap": (fr == "length") if fr is not None else None,
                           "completion_tokens": usage.get("completion_tokens"),
                           "prompt_tokens": usage.get("prompt_tokens"),
                           "text_sha256": _text_sha(text), "completion_id": resp.get("id")})
        if self.request_id_of is not None:
            ev["request_id"] = self.request_id_of(payload, outcome)
        return self.writer.append(ev)

    # the on_done signature `bounded_gather` calls, when a caller binds the payloads itself
    def on_done_for(self, work: Sequence[Tuple[Any, Mapping[str, Any]]]) -> Callable[[Any, Any], None]:
        payloads = dict(work)
        return lambda key, outcome: self.publish(key, payloads.get(key), outcome)


# --------------------------------------------------------------------------- readers


def read_controls_events(path: os.PathLike | str) -> List[dict]:
    """Every event of a stream, strictly: each line a complete JSON object with exactly
    :data:`CONTROLS_EVENT_FIELDS`, ``v == 1``, ``seq`` strictly increasing. A torn or corrupt
    line REFUSES (a reader never repairs the producer's file)."""
    p = Path(path)
    if not p.is_file():
        raise ControlsEventsError(f"{p}: no controls events stream")
    out: List[dict] = []
    last = -1
    for i, line in enumerate(p.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            raise ControlsEventsError(f"{p}: corrupt or torn event at line {i + 1}: {e}") from e
        if not isinstance(ev, dict) or tuple(ev) != CONTROLS_EVENT_FIELDS:
            raise ControlsEventsError(f"{p}: line {i + 1} does not carry exactly {CONTROLS_EVENT_FIELDS}")
        if ev["v"] != CONTROLS_EVENT_VERSION or ev["kind"] != CONTROLS_EVENT_KIND_DRAW:
            raise ControlsEventsError(f"{p}: line {i + 1}: v={ev['v']!r} kind={ev['kind']!r} is not this reader's")
        if not isinstance(ev["seq"], int) or ev["seq"] <= last:
            raise ControlsEventsError(f"{p}: line {i + 1}: seq {ev['seq']!r} does not increase past {last}")
        last = ev["seq"]
        out.append(ev)
    return out


def latest_by_item(events: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str], dict]:
    """``{(control, item_id): the LAST event}`` — a resume's re-draw supersedes the earlier event."""
    out: Dict[Tuple[str, str], dict] = {}
    for ev in events:
        out[(str(ev["control"]), str(ev["item_id"]))] = dict(ev)
    return out


def at_cap_items(events: Iterable[Mapping[str, Any]]) -> List[dict]:
    """The latest event per item whose draw completed AT CAP (``at_cap`` true, no error) — the
    tailer's enqueue set, in ``seq`` order."""
    latest = latest_by_item(events)
    return sorted((ev for ev in latest.values() if ev["error"] is None and ev["at_cap"] is True),
                  key=lambda ev: ev["seq"])


# --------------------------------------------------------------------------- the ancestry rule


def ladder_transitions(ladder: Sequence[int] = CONTROLS_CAP_LADDER) -> Tuple[Tuple[int, int], ...]:
    """Every (source cap, rung cap) transition of the ladder, in order: ((512, 2048), (2048, 8192),
    (8192, 16384)) on the controls ladder. A rung is fed ONLY by the transition ending at it."""
    caps = [int(c) for c in ladder]
    if len(caps) < 2 or caps != sorted(set(caps)):
        raise ControlsEventsError(f"ladder {list(ladder)} is not a strictly increasing list of at least two caps")
    return tuple(zip(caps[:-1], caps[1:]))


def predecessor_cap(cap: int, ladder: Sequence[int] = CONTROLS_CAP_LADDER) -> int:
    """The IMMEDIATELY preceding rung of ``cap`` — the only admissible source of a rung at ``cap``
    (``_apply_escalation``'s rule). Refuses the base cap (it has no predecessor) and a cap off
    the ladder."""
    for src, dst in ladder_transitions(ladder):
        if dst == int(cap):
            return src
    if int(cap) == int(ladder[0]):
        raise ControlsEventsError(f"cap {cap} is the ladder's base {list(ladder)}: a base store has no source rung")
    raise ControlsEventsError(f"cap {cap} is not a rung of the ladder {list(ladder)}")


def check_escalated_from(gen: Mapping[str, Any], *, cap: int,
                         ladder: Sequence[int] = CONTROLS_CAP_LADDER) -> Dict[str, Any]:
    """Assert a rung store's gen block against the ancestry rule and return its
    ``escalated_from``: ``gen["max_new_tokens"] == cap`` and ``escalated_from.cap`` is the
    IMMEDIATE predecessor of ``cap`` — never an earlier rung, never absent on a rung, never
    present on a base store."""
    got_cap = int(gen.get("max_new_tokens", -1))
    if got_cap != int(cap):
        raise ControlsEventsError(f"the store's cap is {got_cap}, not the rung {cap} it is read as")
    esc = gen.get("escalated_from")
    if int(cap) == int(ladder[0]):
        if esc is not None:
            raise ControlsEventsError(f"a base store at {cap} carries escalated_from={esc!r}; a base has no source")
        return {}
    if not isinstance(esc, Mapping) or "cap" not in esc:
        raise ControlsEventsError(f"a rung store at {cap} carries no escalated_from.cap — its source rung is unknown; refusing")
    want = predecessor_cap(cap, ladder)
    if int(esc["cap"]) != want:
        raise ControlsEventsError(
            f"rung {cap} records escalated_from.cap={esc['cap']} but its immediate predecessor on "
            f"{list(ladder)} is {want} — a rung is fed by the rung directly below it, never an earlier one")
    return dict(esc)
