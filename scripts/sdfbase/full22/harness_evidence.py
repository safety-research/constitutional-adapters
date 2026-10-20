#!/usr/bin/env python
"""full22: the STANDUP's own requests carry the tinycap driver's kind tag, so the resolved-sampler
recorder classifies them by a DECLARED identity (recorder-load-check-kinds, 2026-09-13).

A served target answers more than the eval rows. The standup fires the representative LOAD CHECK
(``load_check.py``: S concurrent greedy requests at the RUN's cap — the leg's own client cap, by
design) and the steer-is-live POSITIVE CONTROL (``smoke_positive_control.py``: greedy requests at
``POSITIVE_CONTROL_MAX_TOKENS`` on the steered cell, the anchor and the scale-0 twin, plus
``served_target.probe_fn``'s steering-bites probes at ``PROBE_MAX_TOKENS``). Read off the server's
log alone, the load check is 46 greedy requests at the client cap inside a sampled client class
(job 97376: a false DIVERGED on a valid leg), and the positive control is 48 greedy requests at a
cap nothing declared (UNCLASSIFIED). The cap cannot identify the load check — it shares the client
cap by construction — so identity must come from the SOURCE.

The mechanism is the driver's, byte for byte (``served_capability_run._RequestEvidence.labelled``):
every attempt carries ``X-Tinycap-Task: probe/<kind>`` (``contracts.probe_task_label``),
``X-Tinycap-Sample: <vmodel>/<n>`` (n per kind and route, from 0) and a fresh
``X-Tinycap-Request-Id``; the record of the attempt is written from the EXACT wire bytes + headers
(``contracts.request_evidence_record``, ``REQUEST_EVIDENCE_FIELDS``: ``task``, ``max_tokens``, …)
to a JSONL file beside the script's own ``--out`` — ``<out stem>.request_evidence.jsonl``
(:func:`evidence_path_for`), a file per emitter so two standup scripts never append to one file.
``scripts/sdfbase/record_resolved_sampler.py --probe-expect kinds --probe-tags <file>`` reads it (the
flag is repeatable: one per emitter). A lane whose standup runs a frozen sha WITHOUT this module
declares the same identity explicitly instead (``--declare-kind <kind>:cap=<C>[:count=<N>|:count=<N>!]``).

Never buffered: a run that dies mid-way leaves every record it sent.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Callable, Mapping

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

#: the sidecar's suffix: ``load_check.json`` -> ``load_check.request_evidence.jsonl``
EVIDENCE_SUFFIX = ".request_evidence.jsonl"


def evidence_path_for(out: "Path | str") -> Path:
    """The request-evidence sidecar beside a script's ``--out``: its stem + :data:`EVIDENCE_SUFFIX`."""
    out = Path(out)
    return out.with_name(out.stem + EVIDENCE_SUFFIX)


class HarnessEvidence:
    """The standup's labelled-request writer: :meth:`headers` mints one attempt's labels,
    :meth:`record` appends its record from the wire bytes + headers, :meth:`on_request` is the
    ``ProxyClientEngine.on_request`` hook form of the same, :meth:`labelled` wraps a sync
    ``chat(body, headers)`` callable so every call is a labelled attempt of one kind."""

    def __init__(self, path: "Path | str") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self._counts: dict = {}

    def headers(self, kind: str, vmodel: str) -> dict:
        """``X-Tinycap-Task: probe/<kind>``, ``X-Tinycap-Sample: <vmodel>/<n>`` (n per kind and
        route, from 0), a fresh ``X-Tinycap-Request-Id`` — the driver's labels exactly."""
        n = self._counts.get((kind, vmodel), 0)
        self._counts[(kind, vmodel)] = n + 1
        return {tc.HEADER_TASK: tc.probe_task_label(kind), tc.HEADER_SAMPLE: f"{vmodel}/{n}",
                tc.HEADER_REQUEST_ID: uuid.uuid4().hex}

    def record(self, raw: bytes, headers: Mapping) -> dict:
        rec = tc.request_evidence_record(raw, headers=headers)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.n += 1
        return rec

    def on_request(self, url: str, data: bytes, headers: Mapping) -> None:
        """``ProxyClientEngine(on_request=...)``: every ``/chat/completions`` POST, from its exact bytes."""
        if str(url).rstrip("/").endswith("/chat/completions"):
            self.record(data, headers)

    def labelled(self, chat_fn: Callable[[dict, dict], dict], kind: str) -> Callable[[dict], dict]:
        """A ``chat(body)`` callable whose every call is a labelled attempt of ``kind`` through
        ``chat_fn(body, headers)`` — the shape ``served_target.probe_fn`` and the greedy legs take."""
        def chat(body: dict) -> dict:
            vmodel = str(body.get("model")) if isinstance(body, Mapping) else "?"
            return chat_fn(body, self.headers(kind, vmodel))
        return chat
