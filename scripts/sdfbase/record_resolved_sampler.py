#!/usr/bin/env python
"""record_resolved_sampler.py — WHAT THE ENGINE RESOLVED, read off the server's
own log and compared WITHIN REQUEST CLASSES against what each class was sent.

    record_resolved_sampler.py --vllm-log <vllm_server.log> --out <record.json>
        [--sent-static <responses.jsonl> ...] [--sent-controls <gen_stats.json|meta.jsonl> ...]
        [--expect temperature=0.7 --expect top_p=0.8 ...]
        [--client-cap N ...] [--probe-cap N ...] [--probe-expect greedy|card|kinds]
        [--probe-tags <request_evidence.jsonl|usage_ledger.jsonl> ...]
        [--declare-kind <kind>:cap=<C>[:count=<N>|:count=<N>!] ...] [--strict | --require-wired]

Lane-agnostic: it reads any vLLM server log. On this tree the log comes from
``scripts/sdfbase/mt_serving_up.py``, whose ``--enable-log-requests`` opt-in
(``f22_standup`` forwards it through ``f22_standup_extra_reset --enable-log-requests``)
is what makes the request lines below exist.

WHY THIS EXISTS. The resolved sampler is a property of the SERVING STACK, not
of the model: vLLM fills every sampling key a request omits, from the model's
``generation_config.json`` when it ships one and from its own neutral defaults
when it does not. Two lanes on two stacks can therefore resolve the same
omitted key differently and produce a divergence no summary statistic shows.
So the resolved values are recorded as data and exchanged, the way a
classifier hash is.

WHAT IS AUTHORITATIVE HERE, and what is not:

* ``resolved`` — parsed from vLLM's own ``Received request <id>: params:
  SamplingParams(...)`` INFO lines (0.25.1 ``entrypoints/serve/utils/
  request_logger.py:63``), which the server emits only under
  ``--enable-log-requests``. This is the engine's object AFTER its defaults
  were merged, i.e. the real answer. The prompt is a separate DEBUG record and
  is NOT in these lines, so nothing harmful travels (read off the 0.25.1
  source; re-verify on a version bump).
* the FLAG ATTESTATION — the same module logs ONCE at server start, before any
  request exists, that the flag is set: at INFO when requests will be logged
  ("...higher than DEBUG. Only limited information will be logged...") and at
  WARNING when they will NOT ("...higher than INFO. No request information
  will be logged."). Read so that silence has three readings, not one.
* ``generation_config_override`` — vLLM logs "Default vLLM sampling parameters
  have been overridden by the model's `generation_config.json`: {...}" ONLY
  when such a file exists and differs. Its ABSENCE is therefore positive
  evidence that no key could fall back to a vendor file. Recorded either way,
  because "absent" is a measurement here and not a missing value.
* ``sent`` — the sampler and the cap each CLIENT row recorded. Not
  authoritative about the engine; it is the other half of the comparison.

REQUEST CLASSES (job 487306, 2026-09-12). A served target answers more than
one kind of request: the eval rows (5,180, the vendor row) and the standup's
own greedy steering-bites / hook-attestation probes (180, ``temperature 0.0``
at ``max_tokens 48``). A single-class comparison read the probes as four
diverged keys — the instrument's own calibration requests filed as a defect in
the thing measured — and, worse, that greedy tuple resembles vLLM's
neutral-default signature, so one comparison could not tell the two worlds
apart. Requests are therefore partitioned on a fact the PRODUCERS declare,
never on how a row looks: its ``max_tokens``.

* ``client``: ``max_tokens`` in the caps the ``--sent-*`` rows recorded (plus
  any ``--client-cap``), compared against the rows' recorded sampler (or
  ``--expect``). With no client cap known at all, every non-probe request is a
  client request.
* ``probe``: ``max_tokens`` in ``--probe-cap`` (default: the served target's
  ``sisteer.eval.attacks.served_target.PROBE_MAX_TOKENS``, 48 — the code that
  fires the probes), compared against what that code sends: ``temperature
  0.0``, ``seed 0``, every other sampler key omitted, so each resolves to
  vLLM's neutral default. A controls leg also fires greedy scaffolding at
  other caps (MMLU scoring at ``max_tokens 1``, the thinking-ON bite probe at
  256 — ``scripts/served_capability_run.py``); declare those with
  ``--probe-cap`` so they are compared, not left unclassified.
* ``unclassified``: neither. Reported with its tuples, never folded into a
  class it does not belong to.

A client cap equal to a probe cap is REFUSED: the classes would be
indistinguishable and no verdict is possible. Only a WITHIN-class mismatch is
a divergence; a class internally uniform and matching its reference AGREES.

THE PROBE CLASS'S REFERENCE (``--probe-expect``, 2026-09-12). The probe class
is compared against one of two references, and the record says which:

* ``greedy`` (the default; today's reading, byte for byte): the body the
  scaffolding code sends — ``temperature 0.0`` + vLLM's neutral default for
  every other key.
* ``card``: the SAME ``--expect`` values the client class is checked against —
  the launcher's numeric projection of the PINNED card row. For a driver whose
  render probe decodes under the row's card (the no-switch render probe of the
  GSM8K driver), the greedy reference filed the instrument's own probe as a
  divergence on every leg; under ``card`` such a probe class AGREES when it
  equals the pinned row and DIVERGES when it does not — greedy probes included.
  The expectation comes from the pinned table alone, NEVER from the driver's
  runtime declaration of what it decoded under (a gate must not carry its own
  answer), so ``--probe-expect card`` without any ``--expect`` is REFUSED
  (exit 2). ``card`` does not fold the probes into the client class: they stay
  their own class, with their own count and their own verdict.

* ``kinds`` (2026-09-12): the probe class is PARTITIONED BY KIND and each
  kind is compared against ITS OWN pinned reference. The GSM8K wave's probe
  class is mixed — 10 greedy steering-bites probes at cap 48 plus ONE render
  probe at cap 512 that decodes under the card row — so a single reference
  reads one half or the other as a divergence whichever is chosen (``greedy``
  filed the render probe, ``card`` files the bites). The kinds are the
  driver's own vocabulary (``contracts.PROBE_KINDS``): ``render`` -> the
  PINNED card row (``--expect``, the launcher's projection of the pinned
  table, never the driver's runtime ``sampling_row``); ``bites`` and
  ``lens_on_bite`` -> greedy (``temperature 0.0`` + vLLM's neutral default
  for every other key) at their pinned caps (:data:`PROBE_KIND_TABLE`,
  mirrored from the code that fires each kind and pinned equal by test).

  THE TAG CLASSIFIES; THE SHAPE CROSS-CHECKS. The driver stamps every probe
  attempt with its kind at emission — ``X-Tinycap-Task: probe/<kind>``
  (``served_capability_run._RequestEvidence.labelled``) — and writes it with
  the attempt's ``max_tokens`` into ``<out>/request_evidence.jsonl``
  (``contracts.REQUEST_EVIDENCE_FIELDS``; the usage ledger rows the same
  label as ``task`` / ``max_tokens_requested``). ``--probe-tags`` reads those
  rows (REQUIRED in this mode; exit 2 without it — a kind is never inferred
  from how a line looks alone). The tag rows' ``(kind, max_tokens)`` pairs
  are this leg's cap->kind map, and each params line at a probe cap takes
  the kind the tag names for its cap. vLLM mints its own request id
  (``chatcmpl-<hex>``) and the routing proxy forwards no client id, so no
  tag row meets a params line by identity: the join is per KIND, the
  multiset of tagged caps against the multiset of logged caps. The shape
  must then AGREE with the tag twice over — every tagged cap is one the
  kind's code sends (the pinned caps), and per kind the tagged and logged
  multisets are equal — or the kind is not classified.

  CLASSIFICATION NEVER DEFAULTS. A probe that matches no kind is a THIRD
  outcome, ``UNCLASSIFIED``: a params line no tag names (no tag rows, or none
  at its cap), a cap two kinds claim, a tag whose kind the table does not
  know, a tag/shape disagreement in cap or in count. Such requests are
  reported with their tuples and the reason, never folded into a kind and
  never compared against a guess — and they FAIL the probe class (the
  top-level verdict is ``UNCLASSIFIED_REQUESTS``, not AGREES, not DIVERGED).
  The record reports the count per kind (``n_by_kind``), the tag-side
  counts, and a verdict per kind (``classes.probe.kinds.<kind>``).

THE STANDUP'S TWO HARNESS KINDS (job 97376, 2026-09-13). The standup fires two more
kinds of request around the served target, on EVERY lane that uses the shared
standup: the representative LOAD CHECK (``scripts/sdfbase/full22/load_check.py``:
S concurrent greedy requests at the RUN's cap — the leg's own CLIENT cap, by
design) and the steer-is-live POSITIVE CONTROL (``smoke_positive_control.py``:
greedy legs at ``POSITIVE_CONTROL_MAX_TOKENS`` 64 on the steered cell, the anchor
and the scale-0 twin; its steering-bites step is the ``bites`` kind at 48). On the
Llama run's 8,192 rung the client class held 632 sampled draws PLUS 46 greedy
load-check requests at the same cap (a false DIVERGED), and 48 greedy requests at
64 sat UNCLASSIFIED. Two kinds join the table: ``load_check`` (greedy reference;
cap = the client cap, so the cap-multiset join alone CANNOT separate it from client
traffic — it needs the tag or an explicit declaration) and ``positive_control``
(greedy; cap 64, pinned to the code that fires it). Both scripts now stamp the
same ``X-Tinycap-Task: probe/<kind>`` label and write ``<out stem>.request_evidence
.jsonl`` (``harness_evidence.py``), one file per emitter — pass each to
``--probe-tags``.

DECLARATIONS ARE FIRST-CLASS IDENTITY (``--declare-kind``, the same footing as the
tag — never a fallback, never dropped): a standup running a frozen sha stamps no
tag, so the operator declares what it fired: ``<kind>:cap=<C>[:count=<N>|:count=
<N>!]`` (``contracts.DECLARE_KIND_GRAMMAR``). ``kind`` and ``cap`` are REQUIRED;
``count=N`` is a MAXIMUM (fewer passes, 0 -> the kind is EMPTY, like bites on a
non-lens leg; more -> UNCLASSIFIED, unexpected extra traffic stays loud);
``count=N!`` is exact; ``count`` is optional for a kind whose cap differs from the
client cap (the cap plus the verified greedy shape IS the identity: the operator
asserts the standup is the only source at that cap). ``load_check`` at the CLIENT
cap: under a SAMPLED client reference (``--expect temperature`` > 0) the greedy
shape separates it, so ``count`` is optional / a maximum; under a GREEDY client
reference only the count separates them, so a ``load_check`` declaration without
an exact ``count=N!`` is REFUSED (exit 2, reason in the record), and with one the
kind takes N by count from the greedy-shaped requests at that cap ONLY when they
are identical in every compared key (the choice is then immaterial), else
UNCLASSIFIED. Every declared kind is cross-checked EXACTLY against the log — the
kind's pinned cap, the count, the greedy shape (temperature 0.0) — and any
mismatch is UNCLASSIFIED; a classified kind gets its own verdict against its
pinned reference. Declaring the driver's ``render`` kind, or a kind twice, is
refused; a declaration for a kind the tags already name must AGREE with the tag
(``basis: tag+declared``) or the kind is UNCLASSIFIED. A TAG BINDS, A DECLARATION
COUNTS: a tag is per request (this request is this kind); a declaration asserts N
requests of kind X at cap C exist and the recorder decides WHICH by the
cap-multiset join, so a declared count can be satisfied by the wrong requests if
the traffic mix shifts, where a tag cannot — every per-kind block names its
``basis`` (``tag`` / ``declared_count`` / ``tag+declared``), and where the multiset
does not uniquely determine the assignment (several greedy kinds at one cap,
split by count) the block says ``join_ambiguous: true`` with the competing kinds:
the verdict is still given, but it counts rather than binds. TWO KINDS AT ONE CAP
(the MATH-500 run, job 493001: the driver's 2 tagged card-row render probes at
512 beside the standup's 32-44 greedy load-check requests at the SAME cap — its
load check runs at the CONTROLS cap 512, not the client cap) are UNCLASSIFIED
only when the recorder cannot separate them by tag-count plus shape: the tagged
card-reference kind takes the card-shaped requests (exactly its tag count), the
greedy kind(s) the greedy-shaped ones; a single claim at a cap still takes every
request there (a tagged probe that decoded non-greedy stays a DIVERGENCE).

THE CLIENT CLASS after the carve holds ONLY untagged, undeclared requests at the
client cap. A greedy-shaped request left there under a sampled client reference —
no tag, no declaration — is UNCLASSIFIED and fails the leg (a load check nobody
declared, or a client request that decoded greedy: not told apart, NOT folded
back into the client class as a divergence). Under a greedy client reference an
undeclared load check is indistinguishable from client traffic and stays there
(the record says so); declare it with ``count=N!`` to carve it.

THE ZERO-TAG DRIVER: ``NOT_INSTRUMENTED``. A probe-class request no tag and no
declaration names has two readings, decided by a count the recorder already has.
(a) ZERO tagged requests in the whole run — the tag source absent, or present with
no ``probe/`` row — is a driver that cannot tag (a frozen sha): the untagged,
undeclared REMAINDER reads ``NOT_INSTRUMENTED`` (``reason: untagged_driver``), on
the exit-0 side, reported DESCRIPTIVELY (counts per (cap, greedy / sampled) and
the distinct tuples) and compared against NOTHING — no shape-based fallback, no
manufactured verdict at lower assurance; the record spells the consequence out:
"probe-class attestation unavailable on this driver; client class unaffected". (b)
SOME requests tagged and these not -> ``UNCLASSIFIED``, a genuine gap, exactly as
before. Declarations remain valid on a zero-tag driver: they carve harness
traffic OUT of the client class by explicit identity so the client verdict stays
comparable, and each declared kind is compared and reported per kind.

RUN-LEVEL PROBE VERDICT PRECEDENCE (``contracts.RESOLVED_SAMPLER_PROBE_VERDICT_
PRECEDENCE``): DIVERGED > UNCLASSIFIED > per-kind AGREES, with the zero-tag
remainder's NOT_INSTRUMENTED reported BESIDE an AGREES ("probe: AGREES
(load_check 46, positive_control 48); remainder NOT_INSTRUMENTED (80 greedy at 48,
untagged driver)") and the class verdict itself only when no kind was classified.
A fully declared zero-tag lane reads AGREES per kind with an empty remainder. At
the top level a non-empty NOT_INSTRUMENTED remainder beside an agreeing client
class reads ``PROBE_NOT_INSTRUMENTED`` (never plain ``AGREES``: that word keeps
meaning "everything attested"), exit 0 in both modes.

The record carries ``probe_reference: {kind, expect, source}`` (in ``kinds``
mode also ``kinds: {<kind>: {caps, reference, source}}``), ``declarations``, and
every class block its ``reference`` / ``reference_source`` / ``reference_kind``
(per kind also ``identity: {source, claims, regime, selection}``), so a reader
knows what each class was compared against without re-deriving it.

PER-TASK CLIENT REFERENCES (``--expect-tasks``, 2026-09-20). A served
target's CLIENT traffic can itself be more than one sampler by design: a
controls leg sends the over-refusal tasks (xstest / or_bench) under the
vendor row at the controls cap AND the MMLU letter readout greedy at
``max_tokens 1``; a single ``--expect`` (or the rows' one agreed sampler)
then does not exist, and every such row read UNCOMPARED -- correctly, and
uselessly. The producer therefore declares its expectations PER TASK in a
small JSON document::

    {"schema": "resolved_sampler_task_expectations/1",
     "tasks": {"judged/xstest_v2":      {"caps": [512], "sampler": {"temperature": 0.7, ...}, "source": "<where the row came from>"},
               "judged/or_bench_hard_1k": {"caps": [512], "sampler": {...}, "source": "..."},
               "s2parity/mmlu_letter":  {"caps": [1],   "sampler": {"temperature": 0.0}, "source": "..."},
               "some/task":             {"caps": [2048], "sampler": null, "source": "...", "why": "no reference could be resolved"}}}

The vLLM line carries no task label, so the join is the recorder's standing
one -- BY CAP: the client class is partitioned into cap groups, each group is
compared against the reference of the task(s) that declare that cap, and every
task gets its OWN verdict (``classes.client.tasks.<task>``: ``AGREES`` /
``DIVERGED`` / ``UNCOMPARED`` / ``EMPTY``), mirrored at the top level as
``client_task_verdicts``. Two tasks that share a cap share a verdict (the block
says ``shares_cap_with``); two tasks at one cap with DIFFERENT non-null samplers
are REFUSED (exit 2: indistinguishable). A task declared with ``sampler: null``
is compared against nothing and reads ``UNCOMPARED`` BY NAME -- never folded
into a neighbour's reference, never silently AGREES -- and so does a client
cap no task declares. The class verdict is DIVERGED if any group diverged,
else UNCOMPARED if any group has no reference, else AGREES; the top-level
``verdict`` follows as before. Only the CLIENT class changes: the probe class
and its modes are untouched, and without ``--expect-tasks`` the record is
byte-for-byte what it was. ``--expect`` beside ``--expect-tasks`` is refused
under the default probe mode (two references for one class); under ``card`` /
``kinds`` it is the probe (render) reference only, as before.

THE COMPARISON. Every distinct resolved tuple is reported per class, not just
the modal one: one control decoding differently from the rest is exactly the
defect this is for. A key that resolves to something other than its class's
reference is a ``divergence``; a key that landed exactly on vLLM's neutral
default is flagged ``equals_vllm_neutral_default`` — the signature of a key
that never reached the engine (0.25.1 ``_DEFAULT_SAMPLING_PARAMS``:
temperature 1.0, top_p 1.0, top_k 0, min_p 0.0, presence_penalty 0.0,
repetition_penalty 1.0).

VERDICTS (``contracts.RESOLVED_SAMPLER_VERDICTS``). ``AGREES``; ``DIVERGED`` (a
within-class mismatch, named per class); ``UNCOMPARED`` (client requests read,
but no single reference: pass ``--expect`` or rows that agree);
``UNCLASSIFIED_REQUESTS`` (every compared class agrees, but some requests match no
declared cap, or under ``kinds`` some request / tag / declaration matched no
kind); ``PROBE_NOT_INSTRUMENTED`` (``kinds``: the client class and every tagged /
declared kind agree, and the probe class has an untagged, undeclared remainder
on a driver that stamps no tag at all — descriptive, exit 0); ``FLAG_SET_NO_REQUESTS_YET``
(the server attests the flag and has logged no request — a standup-time
call, says nothing about the sampler either way); ``UNREADABLE`` (no readable
log, or silence WITHOUT the attestation, or the flag attested MUTED because
the log level is above INFO — nothing will ever be logged). Silence is never
agreement. ``REFUSED`` when the declarations make the question unanswerable.

EXIT CODE — settled 2026-09-12: a verdict is a FINDING about the serving
stack, not a defect in the evaluation data, so by default the exit code is 0
for every verdict that produced a record. Job 487306 exited non-zero on a
false DIVERGED alone and a complete, valid run read as a dead one; a record
is loud on its own (the JSON, the stdout lines, the per-class tuples).
``--strict`` maps EVERY verdict to an exit code (DIVERGED 1, UNREADABLE /
UNCOMPARED / UNCLASSIFIED_REQUESTS 3, FLAG_SET_NO_REQUESTS_YET 4) for a gate
whose purpose IS the sampler (a smoke). Exit 2 is reserved for a refused
argument or refused declarations — an operator error, never a finding.
CALLERS THAT CONSUME THE EXIT CODE must read the record's ``verdict`` field
or pass ``--require-wired`` (next paragraph) — by default the recorder
exits 0 for every verdict that produced a record, UNREADABLE included.

``--require-wired`` (2026-09-12) is the strict SUBSET a launcher needs. The
caller DECLARES that it requested request logging and is reading AFTER the
legs, so a record proving the instrument was never wired is a FAILURE:
UNREADABLE 3 (all three of its readings — no readable log; silence with no
attestation, i.e. launched WITHOUT the flag; the flag attested but MUTED, so
nothing will ever be logged) and FLAG_SET_NO_REQUESTS_YET 4 (a post-leg read
that finds no request is an ordering failure, not the standup-time non-answer
it is by default). Every finding ABOUT the sampler stays 0: AGREES, DIVERGED,
UNCOMPARED, UNCLASSIFIED_REQUESTS, PROBE_NOT_INSTRUMENTED. It is separate
from ``--strict`` because ``--strict`` is too blunt for this use: the one
thing the launcher needs to catch is a declared opt-in that was never
honoured — a launcher set the variable that requests the flag, the checkout
it ran read no such variable, the server came up without it, the record said
UNREADABLE, the exit was 0, and the launcher's if/else printed "resolved
sampler recorded" for a rung where nothing had been — and making DIVERGED
fatal to catch it would reintroduce exactly the false failure the default
exists to prevent (job 487306, and a second instance of the same family: the
standup's load check at the client cap, job 97376 above). PROBE_NOT_INSTRUMENTED
at 0 is load-bearing, not incidental: a checkout whose driver stamps no probe
tag reads it on EVERY leg, permanently, and it says nothing about the wiring
(the instrument recorded the requests; it is the TAG that is absent). The two
flags together are REFUSED (exit 2): they name two different gates over one
exit code.

NOT COVERED, stated rather than implied: ``max_tokens`` is COMPUTED by the
server (``get_max_tokens``) and can be clamped below what was asked, so a
difference there is not a sampler divergence and is tallied apart per class
(``distinct_max_tokens``) — it is the CLASS KEY, not a compared key; keys
absent from ``SamplingParams.__repr__`` cannot be read this way at all (such
a key is reported with ``resolved: null`` and a note, never as agreeing); and
this reads the ENGINE's line, so a rewrite between client and engine by the
routing proxy would show up here as a divergence without naming the proxy as
its cause.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

# the repo's src, so the probe cap can be read from the code that fires the probes
# even when this runs under a bare python3 from a checkout
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: the sampler keys this record compares
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "repetition_penalty")
#: vLLM 0.25.1's neutral fallbacks — what an OMITTED key becomes when the
#: model ships no generation_config.json
VLLM_NEUTRAL_DEFAULTS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                         "presence_penalty": 0.0, "repetition_penalty": 1.0}

# vLLM 0.25.1 entrypoints/serve/utils/request_logger.py:21-33 — the flag attested at
# server start (INFO: requests will be logged; WARNING: the level is above INFO and
# nothing will be). The backticks are in the source; tolerated either way.
_FLAG_ATTEST = re.compile(r"`?--enable-log-requests`? is set but")
_FLAG_MUTED = re.compile(r"`?--enable-log-requests`? is set but\s+the minimum log level "
                         r"is higher than INFO")
_REQ = re.compile(r"Received request ([^\s:]+): params: SamplingParams\((.*?)\), "
                  r"lora_request:", re.S)
_OVERRIDE = re.compile(
    r"Default vLLM sampling parameters have been overridden by (.+?): `(.+?)`")

#: what the served target's probe code SENDS (sisteer.eval.attacks.served_target.probe_fn,
#: its ``_greedy`` body, fired at the cell and its scale-0 twin before anything is scored):
#: temperature 0.0, seed 0, max_tokens PROBE_MAX_TOKENS, and NO other sampler key — so
#: every other key resolves to vLLM's neutral default
PROBE_SENT = {"temperature": 0.0}
#: NOTE on the greedy reference (contracts.GREEDY_REFERENCE_RESOLVED, pinned equal by test): greedy requests
#: resolve to top_p 1.0 (and top_k 0, min_p 0.0) on this vLLM EVEN WHEN the model's generation_config.json
#: declares top_p 0.9 — SamplingParams.__post_init__ normalises the greedy path after the vendor defaults are
#: merged (measured on job 97376: override line {'temperature': 0.6, 'top_p': 0.9}, every
#: greedy request logged top_p=1.0). A greedy-class reference pinned at the card's top_p would therefore be
#: WRONG in a way that looks exactly like sampler drift (a top_p divergence flagged equals_vllm_neutral_default).
PROBE_REFERENCE_SOURCE = ("sisteer.eval.attacks.served_target.probe_fn (_greedy): temperature=0.0, "
                          "seed=0, max_tokens=PROBE_MAX_TOKENS; every other sampler key "
                          "omitted, so it resolves to vLLM's neutral default")
#: --probe-expect card: the probe class compared against the launcher's --expect values — the
#: pinned card row, the same reference the client class is checked against
PROBE_CARD_REFERENCE_SOURCE = ("--expect (the pinned card row: the same values the client class is "
                               "checked against; --probe-expect card)")
PROBE_EXPECT_CHOICES = ("greedy", "card", "kinds")
#: the body a GREEDY client leg sends (scripts/served_capability_run.py, decode None):
#: temperature 0.0, seed 0, max_tokens = the leg's cap, nothing else
GREEDY_SENT = {"temperature": 0.0}

#: --probe-expect kinds: the probe class partitioned by KIND — the driver's own tag, read back — each kind
#: against ITS pinned reference. The kind NAMES are ``contracts.PROBE_KINDS``: the value the driver stamps on
#: every probe attempt (``X-Tinycap-Task: probe/<kind>``, ``served_capability_run._RequestEvidence.labelled``)
#: and writes with the attempt's ``max_tokens`` into ``<out>/request_evidence.jsonl``
#: (``contracts.REQUEST_EVIDENCE_FIELDS``; the usage ledger rows the same label). The CAPS are the SHAPE side:
#: what each kind's code sends, mirrored here for a stdlib-only run and pinned equal by test.
PROBE_TAG_PREFIX = "probe/"                      # contracts.PROBE_TASK_PREFIX
PROBE_KIND_RENDER = "render"                     # contracts.PROBE_KIND_RENDER
PROBE_KIND_BITES = "bites"                       # contracts.PROBE_KIND_BITES
PROBE_KIND_LENS_ON_BITE = "lens_on_bite"         # contracts.PROBE_KIND_LENS_ON_BITE
PROBE_KIND_LOAD_CHECK = "load_check"             # contracts.PROBE_KIND_LOAD_CHECK  (the standup's, 2026-09-13)
PROBE_KIND_POSITIVE_CONTROL = "positive_control"  # contracts.PROBE_KIND_POSITIVE_CONTROL
PROBE_KIND_LOGPROB_BITES = "logprob_bites"       # contracts.PROBE_KIND_LOGPROB_BITES
PROBE_KIND_STANDUP_CONTROLS = "standup_controls"  # contracts.PROBE_KIND_STANDUP_CONTROLS (the native standup's positive controls; fg1 / F-AV)
#: the logprob liveness probe's cap: ONE token (contracts.LOGPROB_BITES_MAX_TOKENS), pinned equal by test
LOGPROB_BITES_MAX_TOKENS = 1
#: smoke_positive_control.POSITIVE_CONTROL_MAX_TOKENS == contracts.POSITIVE_CONTROL_MAX_TOKENS (pinned by test)
POSITIVE_CONTROL_MAX_TOKENS = 64
#: a kind whose cap is the RUN's, decided per leg — the client cap on the static / sampled lanes, the CONTROLS cap
#: (512) on the MATH-500 lane — so no constant can pin it: the tag / declaration names it per leg
CAP_PER_LEG = "per-leg"
#: a kind whose cap is the STANDUP's own `--probe-max-tokens` (endpoint.json.probe_max_tokens; 48 by default), stated by the
#: standup's records and declared per standup -- no constant pins it either (fg1 / F-AV)
CAP_PER_STANDUP = "per-standup"
#: the two files a tinycap child writes side by side (contracts.REQUEST_EVIDENCE_FILENAME / LEDGER_FILENAME, pinned equal by test):
#: a `--probe-tags request_evidence.jsonl` is joined to its sibling ledger by request id, so a tagged attempt the ledger REJECTED
#: (the proxy / its pre-tokenize refused it: it never reached the engine) is skipped exactly as a ledger-spelled tag row is
REQUEST_EVIDENCE_FILENAME = "request_evidence.jsonl"
#: the native standup's own tag sidecar beside its vllm_server.log (native_standup.STANDUP_REQUEST_EVIDENCE_FILE, pinned equal by test)
STANDUP_EVIDENCE_FILENAME = "positive_controls.request_evidence.jsonl"
#: the standup's readiness record beside its log and the key that names the CURRENT attempt (native_standup.ENDPOINT_FILE /
#: ENDPOINT_ATTEMPT_KEY, pinned equal by test): the standup's tag rows are filtered to that attempt
STANDUP_ENDPOINT_FILENAME = "endpoint.json"
STANDUP_ENDPOINT_ATTEMPT_KEY = "standup_attempt_id"
LEDGER_FILENAME = "usage_ledger.jsonl"
PROBE_REFERENCE_CARD = "card"
PROBE_REFERENCE_GREEDY = "greedy"
PROBE_KIND_TABLE = {
    PROBE_KIND_RENDER: {
        "caps": (512, 2048), "reference": PROBE_REFERENCE_CARD,
        "source": ("scripts/served_capability_run.py _render_probe (predicate A): ONE 1-item probe per thinking arm, "
                   "decoding under the cell's frozen sampling row at RENDER_PROBE_MAX_TOKENS 512 (an ON retry at "
                   "RENDER_PROBE_RETRY_MAX_TOKENS 2048); compared against --expect — the launcher's projection of the "
                   "PINNED card row — never the driver's own sampling_row")},
    PROBE_KIND_BITES: {
        "caps": (48,), "reference": PROBE_REFERENCE_GREEDY,
        "source": ("sisteer.eval.attacks.served_target.probe_fn / scripts/unified_sweep_run._served_probe_fn (_greedy): "
                   "the steering-bites probe on the cell and its scale-0 twin — temperature 0.0, seed 0, max_tokens "
                   "PROBE_MAX_TOKENS 48, every other sampler key omitted, so it resolves to vLLM's neutral default")},
    PROBE_KIND_LENS_ON_BITE: {
        "caps": (256,), "reference": PROBE_REFERENCE_GREEDY,
        "source": ("scripts/served_capability_run.py _lens_on_bite (predicate C): one lens-ON request + its base twin — "
                   "temperature 0.0, seed 0, max_tokens LENS_ON_BITE_MAX_TOKENS 256, every other sampler key omitted, "
                   "so it resolves to vLLM's neutral default")},
    PROBE_KIND_LOGPROB_BITES: {
        "caps": (LOGPROB_BITES_MAX_TOKENS,), "reference": PROBE_REFERENCE_GREEDY,
        "source": ("scripts/served_capability_run.py _logprob_liveness_evidence (math-eight-arms-extension W4, the "
                   "Orchestrator's ruling 2026-09-13): the LOGPROB liveness statistic on an alphasteer / static_r cell "
                   "under --lens-liveness logprob — ONE greedy token with logprobs + top_logprobs at the first generated "
                   "position, temperature 0.0, seed 0, max_tokens LOGPROB_BITES_MAX_TOKENS 1, on the cell and TWICE on its "
                   "scale-0 twin per bites prompt (the repeat is the measured noise floor), every other sampler key "
                   "omitted (logprobs is not a sampler key), so it resolves to vLLM's neutral default")},
    PROBE_KIND_LOAD_CHECK: {
        "caps": CAP_PER_LEG, "reference": PROBE_REFERENCE_GREEDY,
        "source": ("scripts/sdfbase/full22/load_check.py _fire: the standup's representative load check — S concurrent "
                   "requests, temperature 0.0, seed i, max_tokens = the RUN's cap (the leg's own CLIENT cap on the static / "
                   "sampled lanes; the CONTROLS cap, 512, on the MATH-500 lane — the window must carry S requests at the cap "
                   "the run will ask for), every other sampler key omitted, so it resolves to vLLM's neutral default; tagged "
                   "probe/load_check at the source, else declared")},
    PROBE_KIND_POSITIVE_CONTROL: {
        "caps": (POSITIVE_CONTROL_MAX_TOKENS,), "reference": PROBE_REFERENCE_GREEDY,
        "source": ("scripts/sdfbase/full22/smoke_positive_control.py greedy(): the steer-is-live positive control — "
                   "temperature 0.0, seed 0, max_tokens POSITIVE_CONTROL_MAX_TOKENS 64 on the steered cell, the bare "
                   "anchor and the scale-0 twin (each prompt twice; the count is decided at run time), every other "
                   "sampler key omitted, so it resolves to vLLM's neutral default; its steering-bites step fires "
                   "served_target.probe_fn (kind bites, 48); tagged probe/positive_control at the source, else declared")},
    PROBE_KIND_STANDUP_CONTROLS: {
        "caps": CAP_PER_STANDUP, "reference": PROBE_REFERENCE_GREEDY,
        "source": ("sisteer.capeval.serving.native_standup.run_positive_controls via LoraStandup.greedy_ids: the NATIVE STANDUP's own "
                   "positive controls before READY -- noop_certificate (the base and the zero adapter over the probe set), steering_bites "
                   "(served_target.probe_fn on every lora cell and its scale-0 twin), selection_honored (the mixed batches and their solo "
                   "references) -- temperature 0, max_tokens = the standup's --probe-max-tokens (endpoint.json.probe_max_tokens == "
                   "positive_controls.json.probe_set.max_tokens; 48 by default), every other sampler key omitted, so it resolves to vLLM's "
                   "neutral default; UNTAGGED (the standup stamps no X-Tinycap-Task), so DECLARED per standup from the root's own records "
                   "(seedrep_served.probe_declaration_from_root) with a MAXIMUM count summed from those records -- the gsm8k S3 smoke: 609 "
                   "such requests at 48 beside the drivers' 40 tagged bites at the same cap (fg1 / F-AV)")},
}
PROBE_KINDS = tuple(PROBE_KIND_TABLE)


def _pinned_caps(kind: str):
    """The caps ``kind``'s code sends, or None for a kind that fires at the leg's client cap (load_check) or at the standup's own
    probe cap (standup_controls): the tag / declaration names it."""
    caps = PROBE_KIND_TABLE[kind]["caps"]
    return None if caps in (CAP_PER_LEG, CAP_PER_STANDUP) else tuple(caps)


def _caps_label(kind: str):
    caps = _pinned_caps(kind)
    return PROBE_KIND_TABLE[kind]["caps"] if caps is None else sorted(caps)


#: a cap two kinds pinned would make the shape check ambiguous by construction — refused at import
_PINNED = [c for k in PROBE_KINDS for c in (_pinned_caps(k) or ())]
assert len(set(_PINNED)) == len(_PINNED), "PROBE_KIND_TABLE: two kinds pin one cap"
PROBE_KINDS_REFERENCE_SOURCE = (
    "per kind (--probe-expect kinds): render -> --expect (the pinned card row); bites, lens_on_bite, load_check, "
    "positive_control, standup_controls -> greedy + vLLM's neutral defaults; a kind is classified by the driver's tag (X-Tinycap-Task: "
    "probe/<kind>, read off --probe-tags) or by an explicit --declare-kind, with its shape cross-checked against "
    "PROBE_KIND_TABLE, else UNCLASSIFIED; the untagged, undeclared remainder on a zero-tag driver is NOT_INSTRUMENTED")
#: --declare-kind (contracts.DECLARE_KIND_GRAMMAR): the identity a standup that stamps no tag declares
DECLARE_KIND_GRAMMAR = "<kind>:cap=<C>[:count=<N>|:count=<N>!]"
#: the probe class's untagged-undeclared REMAINDER on a zero-tag driver (contracts.PROBE_VERDICT_NOT_INSTRUMENTED, ...)
VERDICT_NOT_INSTRUMENTED = "NOT_INSTRUMENTED"
NOT_INSTRUMENTED_REASON_UNTAGGED_DRIVER = "untagged_driver"
NOT_INSTRUMENTED_CONSEQUENCE = "probe-class attestation unavailable on this driver; client class unaffected"
#: the probe class's run-level verdict is the first of these any kind / the remainder reads
PROBE_VERDICT_PRECEDENCE = ("DIVERGED", "UNCLASSIFIED", "AGREES", VERDICT_NOT_INSTRUMENTED, "EMPTY")
#: what a kind's identity RESTS ON (contracts.KIND_BASIS_*): a TAG is per request — it binds this request to this kind;
#: a DECLARATION is a COUNT — it asserts N requests of kind X at cap C exist and the recorder decides WHICH by the
#: cap-multiset join, so a declared count can be satisfied by the wrong requests if the traffic mix shifts, where a tag
#: cannot. Every per-kind block names its basis; where the multiset does not uniquely determine the assignment (several
#: greedy kinds at one cap, split by count) the block says `join_ambiguous: true` with the competing kinds — the verdict is
#: still given, but the reader sees that it counts rather than binds.
KIND_BASIS_TAG = "tag"
KIND_BASIS_DECLARED_COUNT = "declared_count"
KIND_BASIS_TAG_AND_DECLARED = "tag+declared"
#: what binds a tag row to a params line, stated in the record
PROBE_TAG_JOIN = ("per kind, by the multiset of max_tokens: the tag rows' (kind, max_tokens) against the log's "
                  "params lines at those caps — vLLM mints its own request id (chatcmpl-<hex>) and the proxy "
                  "forwards no client id, so no tag row meets a params line by identity")

_EXIT_STRICT = {"AGREES": 0, "DIVERGED": 1, "UNREADABLE": 3, "UNCOMPARED": 3,
                "UNCLASSIFIED_REQUESTS": 3, "FLAG_SET_NO_REQUESTS_YET": 4,
                # kinds: the zero-tag remainder is descriptive, never a failure (exit-0 side in both modes)
                "PROBE_NOT_INSTRUMENTED": 0}
#: --require-wired: the caller DECLARES it requested request logging and is reading after the legs, so the exit code
#: answers ONE question — was the instrument wired? — and nothing else. Non-zero only for the two verdicts that are
#: evidence of unwiring: UNREADABLE (no readable log; silence with no attestation = launched WITHOUT the flag; attested
#: but MUTED = nothing will ever be logged) and FLAG_SET_NO_REQUESTS_YET (a post-leg read with no request logged is an
#: ordering failure, not a standup-time non-answer). Every finding ABOUT the sampler stays 0 — DIVERGED included, so
#: this map can never re-manufacture the job-487306 false failure the default exists to prevent. PROBE_NOT_INSTRUMENTED
#: at 0 is LOAD-BEARING, not incidental: a checkout whose driver stamps no probe tag reads that verdict on EVERY leg,
#: permanently, and a map that failed on it would fail every rung of such a lane forever while saying nothing about the
#: wiring (the instrument recorded the requests; it is the TAG that is absent). Same vocabulary as _EXIT_STRICT, and a
#: strict SUBSET of its failures — pinned by test.
_EXIT_REQUIRE_WIRED = {"AGREES": 0, "DIVERGED": 0, "UNCOMPARED": 0, "UNCLASSIFIED_REQUESTS": 0,
                       "PROBE_NOT_INSTRUMENTED": 0,
                       "UNREADABLE": 3, "FLAG_SET_NO_REQUESTS_YET": 4}


def _parse_params(body: str) -> dict:
    """``SamplingParams(k=v, ...)`` -> the SAMPLER_KEYS (+ seed, max_tokens, n)
    it states. Values are read as Python literals; anything that is not a
    literal is kept as its source text rather than evaluated."""
    out: dict = {}
    for m in re.finditer(r"(\w+)=", body):
        key = m.group(1)
        if key not in (*SAMPLER_KEYS, "seed", "max_tokens", "n"):
            continue
        rest = body[m.end():]
        # up to the next ", <word>=" at this nesting level
        nxt = re.search(r",\s+\w+=", rest)
        raw = (rest[:nxt.start()] if nxt else rest).strip().rstrip(",")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def read_log(path: Path) -> dict:
    text = Path(path).read_text(errors="replace")
    resolved = [(rid, _parse_params(body)) for rid, body in _REQ.findall(text)]
    overrides = [{"source": a, "params": b} for a, b in _OVERRIDE.findall(text)]
    return {
        "vllm_log": str(path),
        "n_resolved_lines": len(resolved),
        "resolved": resolved,
        "log_requests_flag_attested": bool(_FLAG_ATTEST.search(text)),
        "log_requests_muted": bool(_FLAG_MUTED.search(text)),
        "generation_config_override": {
            "present": bool(overrides),
            "lines": overrides,
            "reading": (
                "a generation_config.json governs the defaults for any key a "
                "request omits" if overrides else
                "ABSENT: the served snapshot ships no generation_config.json "
                "(or it does not differ), so an omitted key falls to vLLM's "
                "own neutral defaults"),
        },
    }


def _sent_sampler(d: dict) -> "dict | None":
    """The sampler a recorded decode/sampler block says was SENT: its sampler
    keys; a greedy block (``do_sample`` false, no sampler keys) sent exactly
    ``temperature 0.0`` (the drivers' greedy body) and nothing else."""
    s = {k: d[k] for k in SAMPLER_KEYS if k in d}
    if not s and d.get("do_sample") is False:
        return dict(GREEDY_SENT)
    return s or None


def _sampler_of(row: dict) -> "dict | None":
    """The sampler a CLIENT row recorded: the controls meta sidecar's
    ``sampler`` block, or a static row's ``decode`` block minus its
    non-sampler metadata."""
    for key in ("sampler", "decode"):
        if isinstance(row.get(key), dict):
            return _sent_sampler(row[key])
    return None


def _cap_of(row: dict):
    """The cap a CLIENT row asked for: a top-level ``max_tokens`` / ``cap``,
    else its ``decode``/``sampler`` block's ``cap``."""
    for k in ("max_tokens", "cap"):
        if row.get(k) is not None:
            return str(row[k])
    for key in ("decode", "sampler"):
        d = row.get(key)
        if isinstance(d, dict) and d.get("cap") is not None:
            return str(d["cap"])
    return None


def _gen_stats_blocks(path: Path) -> list:
    """The canonical controls store records its decode ONCE per control in
    ``gen_stats.json[control]["decode"]`` — ``{do_sample, <preset keys>, cap,
    arm, ...}``, or ``{do_sample: false, cap, arm}`` for a greedy leg. Returns
    one (sampler-or-None, n_items, cap) per control block."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for _control, blk in d.items():
        if not isinstance(blk, dict) or not isinstance(blk.get("decode"), dict):
            continue
        dec = blk["decode"]
        n = int(blk.get("n_total") or blk.get("n") or 1)
        out.append((_sent_sampler(dec), n, dec.get("cap")))
    return out


def read_sent(paths) -> dict:
    counts: Counter = Counter()
    caps: Counter = Counter()
    n_rows = 0
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            continue
        if p.name == "gen_stats.json":
            for s, n, cap in _gen_stats_blocks(p):
                if s is not None:
                    counts[json.dumps(s, sort_keys=True)] += n
                    n_rows += n
                if cap is not None:
                    caps[str(cap)] += n
            continue
        for line in p.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            s = _sampler_of(row)
            cap = _cap_of(row)
            if s is not None:
                n_rows += 1
                counts[json.dumps(s, sort_keys=True)] += 1
            if cap is not None:
                caps[cap] += 1
    return {"n_rows": n_rows,
            "distinct": [{"sampler": json.loads(k), "n": v}
                         for k, v in counts.most_common()],
            "distinct_max_tokens": [{"max_tokens": k, "n": v} for k, v in caps.most_common()]}


def default_probe_cap():
    """The served target's probe cap, from the code that fires the probes; None
    (reported loudly) when sisteer is not importable."""
    try:
        from sisteer.eval.attacks.served_target import PROBE_MAX_TOKENS
        return int(PROBE_MAX_TOKENS)
    except Exception:  # noqa: BLE001 — no sisteer on path
        return None


def probe_reference() -> dict:
    ref = {k: VLLM_NEUTRAL_DEFAULTS[k] for k in SAMPLER_KEYS}
    ref.update(PROBE_SENT)
    return ref


def kind_reference(kind: str, expect: dict) -> dict:
    """ONE probe kind's pinned reference: the ``--expect`` values (the pinned card
    row) for a ``card`` kind, greedy + neutral defaults for a ``greedy`` kind."""
    spec = PROBE_KIND_TABLE[kind]
    return dict(expect) if spec["reference"] == PROBE_REFERENCE_CARD else probe_reference()


def probe_reference_block(kind: str, expect: dict) -> dict:
    """What the PROBE class is compared against, written into the record:
    ``greedy`` = the scaffolding's own body (temperature 0.0 + neutral defaults);
    ``card`` = the ``--expect`` values, i.e. the pinned card row the client class
    is checked against; ``kinds`` = one pinned reference PER KIND (the render
    probe the card row, the greedy bites their own body), each named with its
    caps and source. Never derived from the driver's runtime declaration."""
    if kind == "card":
        return {"kind": "card", "expect": dict(expect), "source": PROBE_CARD_REFERENCE_SOURCE}
    if kind == "kinds":
        return {"kind": "kinds", "expect": dict(expect), "source": PROBE_KINDS_REFERENCE_SOURCE,
                "kinds": {k: {"caps": _caps_label(k), "reference_kind": s["reference"],
                              "reference": kind_reference(k, expect), "source": s["source"]}
                          for k, s in PROBE_KIND_TABLE.items()}}
    return {"kind": "greedy", "expect": probe_reference(), "source": PROBE_REFERENCE_SOURCE}


def read_probe_tags(paths, *, standup_attempt: Optional[str] = None) -> dict:
    """The driver's KIND TAGS: every ``probe/<kind>`` row of a
    ``request_evidence.jsonl`` (``contracts.REQUEST_EVIDENCE_FIELDS``: ``task``,
    ``max_tokens``) or a ``usage_ledger.jsonl`` (``task``,
    ``max_tokens_requested``; a ``rejected`` row never reached the engine and is
    skipped, counted). Row tasks (an Inspect task label, or none) are not tags.
    Returns the tag rows as ``(kind, cap)`` plus the bookkeeping the record
    carries; a missing file is recorded, never invented."""
    rows: list = []
    id_rows: list = []          # (kind, cap, engine_request_id | None, meta): tag rows joined by ENGINE IDENTITY — every accepted row of the
                                # standup's kind (legacy rows without the field included: they fail closed), never a count claim
    n_rows = n_probe = n_rejected = n_other_attempt = 0
    readable: list = []
    missing: list = []
    ledger_join: dict = {"joined": [], "absent": [], "n_rejected_by_ledger": 0, "rejected": {},
                         "rule": (f"a {REQUEST_EVIDENCE_FILENAME} tag row is written at the client's HTTP boundary BEFORE the request is sent, so an "
                                  f"attempt the proxy refused is tagged although it never reached the engine; the sibling {LEDGER_FILENAME} (the same "
                                  "directory, the same X-Tinycap-Request-Id) rows every attempt with `rejected` -- a tag row whose ledger row is "
                                  "rejected is skipped and counted, exactly as a ledger-spelled `rejected` tag row is (fg1 / F-AV: the deliberate "
                                  "OFF-arm render probe on a thinking-ON standup, 400 thinking_regime_mismatch via the ledger's pre-tokenize)")}
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            missing.append(str(p))
            continue
        readable.append(str(p))
        rejected_ids: dict = {}
        if p.name == REQUEST_EVIDENCE_FILENAME or p.name.endswith("." + REQUEST_EVIDENCE_FILENAME):
            # any `<stem>.request_evidence.jsonl` beside a usage ledger joins it (the standup's sidecar too)
            sib = p.with_name(LEDGER_FILENAME)
            if sib.exists():
                rejected_ids = _ledger_rejected_ids(sib)
                ledger_join["joined"].append(str(sib))
            else:
                ledger_join["absent"].append(str(sib))
        # split on "\n", not splitlines(): a U+2028 / NEL inside a JSON string is not a row boundary
        for line in p.read_text(errors="replace").split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            n_rows += 1
            task = row.get("task")
            if not (isinstance(task, str) and task.startswith(PROBE_TAG_PREFIX)):
                continue
            n_probe += 1
            if row.get("rejected") is True:
                n_rejected += 1
                continue
            # the STANDUP's own rows are this ATTEMPT's alone: the server log appends across restarts in one serving
            # dir, so a row of another attempt (a stale sidecar) is skipped and counted — the requests it would have bound stay UNCLASSIFIED
            if standup_attempt is not None and task == f"{PROBE_TAG_PREFIX}{PROBE_KIND_STANDUP_CONTROLS}" \
                    and row.get("attempt_id") is not None and str(row.get("attempt_id")) != str(standup_attempt):
                n_other_attempt += 1
                continue
            rid = row.get("request_id")
            if rejected_ids and isinstance(rid, str) and rid in rejected_ids:
                n_rejected += 1
                ledger_join["n_rejected_by_ledger"] += 1
                ledger_join["rejected"][f"{task}@{row.get('max_tokens')}"] = ledger_join["rejected"].get(f"{task}@{row.get('max_tokens')}", 0) + 1
                continue
            cap = row.get("max_tokens", row.get("max_tokens_requested"))
            try:
                cap_s = None if cap is None else str(int(cap))
            except (TypeError, ValueError):
                cap_s = str(cap)
            kind_of_row = task[len(PROBE_TAG_PREFIX):]
            if "engine_request_id" in row or kind_of_row == PROBE_KIND_STANDUP_CONTROLS:
                # an IDENTITY row: EVERY accepted row of the standup's kind — and any row that names the
                # engine request it bound — is joined by identity, never by cap count; a standup row WITHOUT the field (a sidecar written
                # by a standup older than the identity join, still running through a rolling upgrade) fails closed with the row named
                eid = row.get("engine_request_id")
                meta = {"attempt_id": row.get("attempt_id"), "request_id": row.get("request_id"),
                        "engine_id_field": ("present" if "engine_request_id" in row else "absent")}
                id_rows.append((kind_of_row, cap_s, (str(eid) if isinstance(eid, str) and eid else None), meta))
                continue
            rows.append((kind_of_row, cap_s))
    by_kind: dict = {}
    for kind, cap in rows:
        by_kind.setdefault(kind, Counter())[str(cap)] += 1
    for kind, cap, _eid, _meta in id_rows:
        by_kind.setdefault(kind, Counter())[str(cap)] += 1
    return {"sources": readable, "missing": missing, "n_rows": n_rows, "n_probe_rows": n_probe,
            "n_rejected_skipped": n_rejected, "n_other_attempt_skipped": n_other_attempt, "standup_attempt": standup_attempt, "rows": rows,
            "id_rows": id_rows, "n_id_rows": len(id_rows),
            "by_kind": {k: [{"max_tokens": c, "n": n} for c, n in v.most_common()] for k, v in by_kind.items()},
            "tag_field": "task = probe/<kind> (X-Tinycap-Task), max_tokens (request_evidence.jsonl) / "
                         "max_tokens_requested (usage_ledger.jsonl)",
            "join": PROBE_TAG_JOIN, "ledger_join": ledger_join}


def _ledger_rejected_ids(ledger: Path) -> dict:
    """``{request_id: error_code}`` of every ``rejected: true`` row of a child's ``usage_ledger.jsonl`` (an attempt that never reached the
    engine); a row without a request id cannot be joined and is not returned."""
    out: dict = {}
    for line in ledger.read_text(errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("rejected") is True and isinstance(row.get("request_id"), str):
            out[row["request_id"]] = row.get("error_code")
    return out


def parse_declaration(text: str) -> dict:
    """One ``--declare-kind`` value (:data:`DECLARE_KIND_GRAMMAR`): ``<kind>:cap=<C>[:count=<N>|:count=<N>!]``.
    ``kind`` must be a GREEDY kind of the table (the tinycap driver tags its own ``render``); ``cap`` is
    required; ``count`` is a maximum, ``count=N!`` exact. Raises ValueError with the reason."""
    parts = [x.strip() for x in text.split(":")]
    kind = parts[0]
    greedy_kinds = [k for k in PROBE_KINDS if PROBE_KIND_TABLE[k]["reference"] == PROBE_REFERENCE_GREEDY]
    if kind not in PROBE_KIND_TABLE:
        raise ValueError(f"unknown kind {kind!r} — the kinds are {list(PROBE_KINDS)}")
    if kind not in greedy_kinds:
        raise ValueError(f"{kind!r} is the tinycap driver's own kind and that driver tags it (X-Tinycap-Task: "
                         f"probe/{kind}); a declaration names the standup's GREEDY kinds only: {greedy_kinds}")
    cap = count = None
    exact = False
    seen: set = set()
    for field in parts[1:]:
        k, sep, v = field.partition("=")
        if not sep or not v or k in seen:
            raise ValueError(f"bad field {field!r}")
        seen.add(k)
        try:
            if k == "cap":
                cap = int(v)
            elif k == "count":
                exact = v.endswith("!")
                count = int(v[:-1] if exact else v)
            else:
                raise ValueError(f"unknown field {k!r} (fields: cap, count)")
        except ValueError as e:
            if str(e).startswith("unknown field"):
                raise
            raise ValueError(f"{k} must be a non-negative integer, got {v!r}") from None
        if (k == "cap" and cap <= 0) or (k == "count" and count < 0):
            raise ValueError(f"{k} must be {'positive' if k == 'cap' else 'non-negative'}, got {v!r}")
    if cap is None:
        raise ValueError("cap is required (a declared identity always names the cap it fires at)")
    return {"kind": kind, "cap": str(cap), "count": count, "exact": exact, "raw": text}


def _count_check(n_found: int, claim: dict) -> "str | None":
    """The count half of an identity claim: exact must equal, a maximum must not be exceeded."""
    c = claim["count"]
    if c is None:
        return None
    if claim["exact"]:
        if n_found != c:
            return (f"the {claim['source']} says {c}! (exact) {claim['kind']} at max_tokens {claim['cap']} but "
                    f"{n_found} logged — identity and shape disagree in count")
    elif n_found > c:
        return (f"the {claim['source']} allows at most {c} {claim['kind']} at max_tokens {claim['cap']} but "
                f"{n_found} logged — unexpected extra traffic")
    return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_greedy(row) -> bool:
    return _num(row[1].get("temperature")) == 0.0


def _row_cap(row) -> str:
    return str(row[1].get("max_tokens"))


def _row_tuple(row) -> str:
    return json.dumps({k: row[1][k] for k in SAMPLER_KEYS if k in row[1]}, sort_keys=True)


def _consume_by_count(cap: str, pool: list, claims: list):
    """One shape's rows at one cap shared by SEVERAL greedy claims, assigned by COUNT — exact claims first, then
    maxima (min(N, remaining)), then the at-most-one countless claim (the rest). Nothing here binds a request to
    a kind: it counts (``join_ambiguous``). Returns ``(assignment {kind: rows}, problem-or-None, leftover)``."""
    kinds = [c["kind"] for c in claims]
    exact = [c for c in claims if c["count"] is not None and c["exact"]]
    maxes = [c for c in claims if c["count"] is not None and not c["exact"]]
    free = [c for c in claims if c["count"] is None]
    if len(free) > 1:
        return {}, (f"kinds {kinds} share max_tokens {cap} and {[c['kind'] for c in free]} carry no count — the split "
                    f"cannot be decided; give count=N or count=N! to all but one"), list(pool)
    need = sum(c["count"] for c in exact)
    if need > len(pool):
        return {}, (f"kinds {kinds} share max_tokens {cap}: the exact counts {[(c['kind'], c['count']) for c in exact]} "
                    f"sum to {need} but {len(pool)} greedy-shaped request(s) are logged there"), list(pool)
    remaining, out = list(pool), {}
    for c in exact:
        out[c["kind"]], remaining = remaining[:c["count"]], remaining[c["count"]:]
    for c in maxes:
        k = min(c["count"], len(remaining))
        out[c["kind"]], remaining = remaining[:k], remaining[k:]
    for c in free:
        out[c["kind"]], remaining = remaining, []
    return out, None, remaining


def classify_kinds(probe_rows, client_rows, *, tags: dict, declarations, expect: dict, client_caps: set) -> dict:
    """The PROBE class by KIND (``--probe-expect kinds``), identity from the TAG or a DECLARATION.

    THE IDENTITY CLASSIFIES: a tag row (``probe/<kind>`` at ``max_tokens``) is an exact CLAIM ``(kind, cap, n)``
    that BINDS its requests; a ``--declare-kind`` is a claim that COUNTS (``basis: declared_count``) — the
    recorder decides which requests by the cap-multiset join. A declaration for a kind the tags already name
    must agree with the tag (``basis: tag+declared``), else the kind is UNCLASSIFIED. THE SHAPE CROSS-CHECKS
    and must agree: a claimed cap is one the kind's code sends (:data:`PROBE_KIND_TABLE`; ``load_check`` fires
    at the run's per-leg cap), the count holds, a DECLARED cap's requests are greedy-shaped. Assignment is per
    CAP: with ONE claim the kind takes every request at the cap (a tagged probe that decoded non-greedy is a
    DIVERGENCE of its kind); with SEVERAL, the shape separates the one card-reference (tagged) kind from the
    greedy kinds — the tag's count must equal the card-shaped requests exactly — and several greedy kinds at
    one cap are split by COUNT (``join_ambiguous: true``, the competing kinds named: the multiset counts, it does
    not bind). At the CLIENT cap the greedy shape separates a kind from a SAMPLED client class; under a greedy
    client reference only an exact count does, by count from identical tuples. Anything else is UNCLASSIFIED —
    reported with its tuples and its reason, never folded into a kind, never compared against a guess. A
    classified kind is compared against ITS reference (:func:`kind_reference`).

    THE REMAINDER — probe-class requests no claim names — is NOT_INSTRUMENTED when the run carries ZERO tagged
    requests (a driver that cannot tag: descriptive counts per (cap, greedy / sampled), compared against nothing)
    and UNCLASSIFIED when some requests are tagged and these are not, or when a claimed cap's claims do not
    account for them. A greedy-shaped request left at the CLIENT cap under a sampled reference, unclaimed, is
    UNCLASSIFIED (not folded back). Returns the probe-class block plus ``client_rows`` (the client class after
    the carve)."""
    client_ref = dict(expect)
    t = _num(client_ref.get("temperature")) if "temperature" in client_ref else None
    regime = None if t is None else ("greedy" if t == 0.0 else "sampled")
    id_rows = list(tags.get("id_rows") or [])
    zero_tags = not tags["rows"] and not id_rows
    tagged = {k: Counter() for k in PROBE_KINDS}
    unknown_kind_tags: Counter = Counter()
    for kind, cap in tags["rows"]:               # the COUNT-join tags (the drivers'); identity rows are joined below, per request
        if kind in PROBE_KIND_TABLE:
            tagged[kind][str(cap)] += 1
        else:
            unknown_kind_tags[f"{kind}@{cap}"] += 1
    problems = {k: [] for k in PROBE_KINDS}
    basis: dict = {}
    claims: list = []
    decl_by_kind = {d["kind"]: d for d in declarations}
    for kind in PROBE_KINDS:
        tcaps, d = tagged[kind], decl_by_kind.get(kind)
        if tcaps:
            claims += [{"kind": kind, "cap": cap, "count": n, "exact": True, "source": "tag"} for cap, n in tcaps.items()]
            if d is None:
                basis[kind] = KIND_BASIS_TAG
            elif d["cap"] not in tcaps:
                problems[kind].append(f"the declaration says {kind} at max_tokens {d['cap']} but the tag names "
                                      f"{sorted(tcaps)} — identity sources disagree")
                basis[kind] = None
            elif d["count"] is not None and ((d["exact"] and d["count"] != tcaps[d["cap"]])
                                             or (not d["exact"] and tcaps[d["cap"]] > d["count"])):
                problems[kind].append(f"the declaration says {kind} count={d['count']}{'!' if d['exact'] else ''} at "
                                      f"max_tokens {d['cap']} but the tag binds {tcaps[d['cap']]} — identity sources disagree")
                basis[kind] = None
            else:
                basis[kind] = KIND_BASIS_TAG_AND_DECLARED      # the tag governs (it binds); the declaration agrees
        elif d is not None:
            claims.append({**d, "source": "declaration"})
            basis[kind] = KIND_BASIS_DECLARED_COUNT
        else:
            basis[kind] = None
    rows_of = {k: [] for k in PROBE_KINDS}
    selection = {k: [] for k in PROBE_KINDS}
    regimes = {k: set() for k in PROBE_KINDS}
    join_amb = {k: [] for k in PROBE_KINDS}
    probe_left = set(range(len(probe_rows)))
    # ---- THE IDENTITY JOIN: a tag row that names the ENGINE request it bound (the standup's accepted controls
    # record the response id, which vLLM logs as `Received request <id>`) binds exactly THAT logged request; an id absent from the
    # parsed log (a partial / muted log, a parser miss), a duplicate id, an id logged more than once, an id whose logged cap is not
    # the tag's, or an accepted row without an id is a NAMED PROBLEM of its kind — the kind is UNCLASSIFIED and the requests it did
    # not bind stay in the remainder; it NEVER falls back to cap equality (that is how an accepted tag consumed an untagged request)
    by_rid: dict = {}
    for i, rr in enumerate(probe_rows):
        by_rid.setdefault(str(rr[0]), []).append(i)
    id_seen = Counter(eid for _k, _c, eid, _m in id_rows if eid)
    n_id_bound = Counter()
    for kind, cap, eid, meta in id_rows:
        if kind not in PROBE_KIND_TABLE:
            unknown_kind_tags[f"{kind}@{cap}"] += 1
            continue
        tagged[kind][str(cap)] += 1
        regimes[kind].add("probe_cap")
        if not eid:
            absent = (meta or {}).get("engine_id_field") == "absent"
            problems[kind].append(f"an accepted {kind} tag at max_tokens {cap} (attempt {(meta or {}).get('attempt_id')!r}, request "
                                  f"{(meta or {}).get('request_id')!r}) carries no engine id — "
                                  + ("the field is ABSENT: a sidecar written by a standup older than the identity join (a rolling upgrade)"
                                     if absent else "the response carried no `id`")
                                  + "; it cannot be joined to a logged request; never a cap-count fallback")
            continue
        if id_seen[eid] > 1:
            problems[kind].append(f"the tag names engine request {eid} {id_seen[eid]} times — a duplicate identity binds nothing")
            continue
        hits = [i for i in by_rid.get(eid, []) if i in probe_left]
        if not hits:
            problems[kind].append(f"the tag names engine request {eid} at max_tokens {cap}, ABSENT from the parsed server log (a partial or muted "
                                  "log, a parser miss, a request logged under another id) — it binds nothing; never a cap-count fallback")
            continue
        if len(hits) > 1:
            problems[kind].append(f"engine request {eid} is logged {len(hits)} times — the identity is not unique; it binds nothing")
            continue
        i = hits[0]
        if _row_cap(probe_rows[i]) != str(cap):
            problems[kind].append(f"the tag says max_tokens {cap} for engine request {eid} but the log says {_row_cap(probe_rows[i])} — "
                                  "identity and shape disagree")
            continue
        rows_of[kind].append(probe_rows[i])
        probe_left.discard(i)
        n_id_bound[kind] += 1
    for kind, n in n_id_bound.items():
        selection[kind].append(f"{n} request(s) bound BY ENGINE REQUEST ID (the response id each accepted control recorded, matched to its "
                               "`Received request` line): per request, never by cap count")
        if basis.get(kind) is None and not problems[kind]:
            basis[kind] = KIND_BASIS_TAG
    client_taken: set = set()
    carved: Counter = Counter()
    claimed_caps = sorted({c["cap"] for c in claims}, key=lambda x: int(x) if x.isdigit() else 0)
    for cap in claimed_caps:
        at = [c for c in claims if c["cap"] == cap]
        valid, off = [], []
        for c in at:
            pinned = _pinned_caps(c["kind"])
            (off if pinned is not None and cap not in {str(x) for x in pinned} else valid).append(c)
        for c in off:
            problems[c["kind"]].append(f"the {c['source']} says {c['kind']} at max_tokens {cap} but {c['kind']}'s code "
                                       f"sends {sorted(_pinned_caps(c['kind']))} — identity and shape disagree")
        is_client = cap in client_caps
        src = client_rows if is_client else probe_rows
        pool = ([i for i, r in enumerate(client_rows) if i not in client_taken and _row_cap(r) == cap] if is_client
                else [i for i in sorted(probe_left) if _row_cap(probe_rows[i]) == cap])
        for c in valid:
            regimes[c["kind"]].add("client_cap" if is_client else "probe_cap")

        def take(kind: str, idxs: list) -> None:
            rows_of[kind].extend(src[i] for i in idxs)
            if is_client:
                client_taken.update(idxs)
                if idxs:
                    carved[kind] += len(idxs)
            else:
                probe_left.difference_update(idxs)

        if not valid:
            if off and not is_client:
                take(off[0]["kind"], pool)          # reported under the off-table kind (today's reading)
            continue
        card = [c for c in valid if PROBE_KIND_TABLE[c["kind"]]["reference"] == PROBE_REFERENCE_CARD]
        greedy_claims = [c for c in valid if c not in card]
        if len(card) > 1:
            for c in valid:
                problems[c["kind"]].append(f"max_tokens {cap} is tagged under more than one card-reference kind "
                                           f"{sorted(x['kind'] for x in card)} — the shape cannot separate them")
            continue
        if is_client and card:
            for c in card:
                problems[c["kind"]].append(f"{c['kind']} decodes under the card row and cannot be told from client "
                                           f"traffic at the client cap {cap}")
            card = []
        if len(valid) == 1 and not is_client:
            # ONE claim: the kind takes every request at the cap (a tagged probe that decoded non-greedy is a divergence)
            c = valid[0]
            if c["source"] == "tag":
                if len(pool) != c["count"]:
                    problems[c["kind"]].append(f"tagged {{{cap!r}: {c['count']}}} vs logged {{{cap!r}: {len(pool)}}} — "
                                               f"tag and shape disagree in count")
            else:
                non_greedy = [i for i in pool if not _is_greedy(src[i])]
                if non_greedy:
                    problems[c["kind"]].append(f"{len(non_greedy)} of {len(pool)} request(s) at max_tokens {cap} are not "
                                               f"greedy-shaped (temperature != 0.0) — a declaration claims a cap for the "
                                               f"standup's GREEDY traffic only")
                why = _count_check(len(pool), c)
                if why:
                    problems[c["kind"]].append(why)
                if pool and not non_greedy and not why:
                    selection[c["kind"]].append(f"every request at max_tokens {cap} ({len(pool)}), greedy-shaped: a declared "
                                                f"cap the standup alone fires at")
            take(c["kind"], pool)
            continue
        # SEVERAL claims at one cap, or the client cap: the shape separates card from greedy, counts split the greedy pool
        greedy = [i for i in pool if _is_greedy(src[i])]
        shaped = [i for i in pool if not _is_greedy(src[i])]
        if card:
            c = card[0]
            if len(shaped) != c["count"]:
                problems[c["kind"]].append(f"the tag binds {c['count']} {c['kind']} at max_tokens {cap} but {len(shaped)} "
                                           f"card-shaped (non-greedy) request(s) are logged there beside {len(greedy)} "
                                           f"greedy-shaped — tag-count and shape do not separate the kinds")
            else:
                selection[c["kind"]].append(f"{len(shaped)} card-shaped request(s) at max_tokens {cap}, the tag's count exactly; "
                                            f"the {len(greedy)} greedy-shaped request(s) at the same cap are "
                                            f"{[x['kind'] for x in greedy_claims]}'s (separated by tag-count plus shape)")
            take(c["kind"], shaped)
        if not greedy_claims:
            continue
        if is_client and regime != "sampled":
            # greedy / temperature-less client reference: exact counts only (declarations were refused without one),
            # and the requests must be identical for a by-count carve to be immaterial
            tuples = {_row_tuple(src[i]) for i in greedy}
            need = sum(int(c["count"]) for c in greedy_claims)
            if len(greedy) < need:
                for c in greedy_claims:
                    problems[c["kind"]].append(f"the {c['source']} says {c['count']}! {c['kind']} at the client cap {cap} but "
                                               f"only {len(greedy)} greedy-shaped request(s) are logged there")
                take(greedy_claims[0]["kind"], greedy)
                continue
            if len(tuples) > 1:
                for c in greedy_claims:
                    problems[c["kind"]].append(f"{c['count']}! {c['kind']} at the client cap {cap} under a "
                                               f"{'GREEDY' if regime == 'greedy' else 'temperature-less'} client reference, but the "
                                               f"{len(greedy)} greedy-shaped requests there resolve to {len(tuples)} distinct tuples "
                                               f"— which {c['count']} are the {c['kind']} cannot be told; nothing folded")
                take(greedy_claims[0]["kind"], greedy)
                continue
            for c in greedy_claims:
                n = int(c["count"])
                take(c["kind"], greedy[:n])
                greedy = greedy[n:]
                selection[c["kind"]].append(f"{n} identical greedy-shaped request(s) at the client cap {cap}, by count: the client "
                                            f"reference is greedy too, so the {c['kind']}'s requests are indistinguishable from "
                                            f"the client's and the choice is immaterial")
                if len(greedy_claims) > 1:
                    join_amb[c["kind"]] = sorted(x["kind"] for x in greedy_claims if x is not c)
            continue
        if len(greedy_claims) == 1:
            c = greedy_claims[0]
            why = _count_check(len(greedy), c)
            if why:
                problems[c["kind"]].append(why)
            elif is_client:
                selection[c["kind"]].append(f"{len(greedy)} greedy-shaped request(s) at the client cap {cap}: the client reference "
                                            f"is sampled (temperature {t}), so the greedy shape separates the {c['kind']}")
            elif greedy:
                selection[c["kind"]].append(f"{len(greedy)} greedy-shaped request(s) at max_tokens {cap}; the card-shaped ones are "
                                            f"{card[0]['kind']}'s by its tag (separated by tag-count plus shape)")
            take(c["kind"], greedy)
            continue
        assign, why, _left = _consume_by_count(cap, greedy, greedy_claims)
        if why:
            for c in greedy_claims:
                problems[c["kind"]].append(why)
            take(greedy_claims[0]["kind"], greedy)
            continue
        for c in greedy_claims:
            idxs = assign.get(c["kind"], [])
            take(c["kind"], idxs)
            join_amb[c["kind"]] = sorted(x["kind"] for x in greedy_claims if x is not c)
            selection[c["kind"]].append(f"{len(idxs)} of {len(greedy)} greedy-shaped request(s) at max_tokens {cap} by COUNT, "
                                        f"shared with {join_amb[c['kind']]}: the multiset counts, it does not bind")
        # counts that leave greedy-shaped requests unassigned: they stay in the pool (remainder / client leftover)
    kinds_out: dict = {}
    uncl: list = []
    n_uncl_tags = sum(unknown_kind_tags.values())
    n_uncl_decl = 0
    for kind in PROBE_KINDS:
        spec = PROBE_KIND_TABLE[kind]
        krows, kproblems = rows_of[kind], problems[kind]
        kclaims = [c for c in claims if c["kind"] == kind]
        declared = decl_by_kind.get(kind)
        n_tagged = sum(tagged[kind].values())
        n_decl = 1 if declared is not None else 0
        logged = Counter(_row_cap(r) for r in krows)
        if not kclaims and not krows and not kproblems:
            verdict, cmp_, ref_kind = "EMPTY", compare([], [], {}), None
        elif kproblems:
            verdict, ref_kind = "UNCLASSIFIED", None
            cmp_ = compare(krows, [], {})            # the tuples, no reference: never compared against a guess
            uncl.extend((rid, p, f"{kind}: " + "; ".join(kproblems)) for rid, p in krows)
            n_uncl_tags += n_tagged
            n_uncl_decl += n_decl
        elif not krows:
            verdict, cmp_, ref_kind = "EMPTY", compare([], [], {}), None      # declared (a maximum), nothing fired
            selection[kind].append("declared, none fired (0 <= the maximum)")
        else:
            ref_kind = spec["reference"]
            if ref_kind == PROBE_REFERENCE_CARD:
                cmp_ = compare(krows, [], expect)
            else:
                cmp_ = compare(krows, [{"sampler": probe_reference(), "n": len(krows)}], {})
            verdict = "DIVERGED" if cmp_["divergences"] else "AGREES"
        for d in cmp_["divergences"]:
            d["kind"] = kind
        cmp_["reference_source"] = spec["source"] if ref_kind is not None else None
        kinds_out[kind] = {"n": len(krows), "n_tagged": n_tagged, "n_declared": n_decl, "verdict": verdict,
                           "reference_kind": ref_kind, "caps_pinned": _caps_label(kind),
                           "basis": basis.get(kind) if verdict != "UNCLASSIFIED" else None,
                           "join_ambiguous": bool(join_amb[kind]), "competing_kinds": join_amb[kind],
                           "identity": {"source": ("tag" if n_tagged else "declaration" if n_decl else None),
                                        "basis": basis.get(kind),
                                        "claims": [{k: c[k] for k in ("cap", "count", "exact", "source")} for c in kclaims]
                                        + ([{k: declared[k] for k in ("cap", "count", "exact")} | {"source": "declaration"}]
                                           if declared is not None and n_tagged else []),
                                        "regime": sorted(regimes[kind]), "selection": selection[kind]},
                           "tag_check": {"agrees": not kproblems, "tagged": dict(tagged[kind]),
                                         "logged": dict(logged), "problems": kproblems},
                           **cmp_}
    # ---- the remainder: probe-class requests no claim names (or a claimed cap's claims did not account for)
    rem = [probe_rows[i] for i in sorted(probe_left)]
    claimed = set(claimed_caps)
    rem_claimed = [r for r in rem if _row_cap(r) in claimed]
    rem_plain = [r for r in rem if _row_cap(r) not in claimed]
    for r in rem_claimed:
        cap = _row_cap(r)
        uncl.append((r[0], r[1], f"unassigned at max_tokens {cap}: the claims there "
                                 f"({sorted({c['kind'] for c in claims if c['cap'] == cap})}) do not account for it "
                                 f"({'greedy' if _is_greedy(r) else 'sampled'}-shaped)"))
    if rem_plain and zero_tags:
        rem_verdict = VERDICT_NOT_INSTRUMENTED
    elif rem_plain or rem_claimed:
        rem_verdict = "UNCLASSIFIED"
        uncl.extend((r[0], r[1], f"no tag or declaration names max_tokens {_row_cap(r)}") for r in rem_plain)
    else:
        rem_verdict = "EMPTY"
    rem_desc = rem_plain if rem_verdict == VERDICT_NOT_INSTRUMENTED else rem_plain + rem_claimed
    shape_counts = Counter((_row_cap(r), "greedy" if _is_greedy(r) else "sampled") for r in rem_desc)
    rem_cmp = compare(rem_desc, [], {})
    remainder = {"n": len(rem_desc), "verdict": rem_verdict,
                 "reason": (NOT_INSTRUMENTED_REASON_UNTAGGED_DRIVER if rem_verdict == VERDICT_NOT_INSTRUMENTED else
                            "no tag or declaration names these requests" if rem_desc else None),
                 "consequence": NOT_INSTRUMENTED_CONSEQUENCE if rem_verdict == VERDICT_NOT_INSTRUMENTED else None,
                 "compared_against": None,
                 "shape_counts": [{"max_tokens": cap, "shape": shape, "n": n}
                                  for (cap, shape), n in sorted(shape_counts.items(),
                                                                key=lambda kv: (int(kv[0][0]) if kv[0][0].isdigit() else 0, kv[0][1]))],
                 "distinct_resolved": rem_cmp["distinct_resolved"], "distinct_max_tokens": rem_cmp["distinct_max_tokens"],
                 "tag_source": {"named": bool(tags["sources"] or tags["missing"]), "sources": tags["sources"],
                                "missing": tags["missing"], "n_probe_rows": tags["n_probe_rows"]}}
    # ---- the client class after the carve: a greedy-shaped leftover under a sampled reference is UNCLASSIFIED
    leftover: set = set()
    if regime == "sampled":
        for i, r in enumerate(client_rows):
            if i not in client_taken and _is_greedy(r) and _row_cap(r) in client_caps:
                leftover.add(i)
                uncl.append((r[0], r[1], f"greedy-shaped at the client cap {_row_cap(r)} with no tag and no declaration "
                                         f"naming it — a load check (declare --declare-kind {PROBE_KIND_LOAD_CHECK}:cap="
                                         f"{_row_cap(r)}) or a client request that decoded greedy: not told apart, not "
                                         f"folded into the client class"))
        if leftover:
            carved["unclassified"] = len(leftover)
    client_left = [r for i, r in enumerate(client_rows) if i not in client_taken and i not in leftover]
    client_note = None
    if regime == "greedy" and not any(c["kind"] == PROBE_KIND_LOAD_CHECK for c in claims):
        client_note = ("the client reference is greedy: an undeclared load check at the client cap is indistinguishable "
                       f"from client traffic and stays in the client class — declare it with --declare-kind "
                       f"{PROBE_KIND_LOAD_CHECK}:cap=<C>:count=<N>! to carve it")
    # ---- the probe class's verdict: DIVERGED > UNCLASSIFIED > AGREES > NOT_INSTRUMENTED > EMPTY
    n_uncl = len(uncl)
    kverd = {k: kinds_out[k]["verdict"] for k in PROBE_KINDS}
    if not probe_rows and not claims and not carved and not any(problems.values()):
        verdict = "EMPTY"
    elif any(v == "DIVERGED" for v in kverd.values()):
        verdict = "DIVERGED"
    elif n_uncl or n_uncl_tags or n_uncl_decl:
        verdict = "UNCLASSIFIED"
    elif any(v == "AGREES" for v in kverd.values()):
        verdict = "AGREES"
    elif rem_verdict == VERDICT_NOT_INSTRUMENTED:
        verdict = VERDICT_NOT_INSTRUMENTED
    else:
        verdict = "EMPTY"
    all_rows = probe_rows + [client_rows[i] for i in sorted(client_taken | leftover)]
    whole = compare(all_rows, [], {})
    stray = compare([(rid, p) for rid, p, _why in uncl], [], {})
    return {"n": len(all_rows), "verdict": verdict, "reference": None, "reference_source": "per kind — see `kinds`",
            "reference_kind": "kinds",
            "distinct_resolved": whole["distinct_resolved"], "distinct_max_tokens": whole["distinct_max_tokens"],
            "divergences": [d for k in PROBE_KINDS for d in kinds_out[k]["divergences"]],
            "kinds": kinds_out,
            "n_by_kind": {k: kinds_out[k]["n"] for k in PROBE_KINDS},
            "basis_by_kind": {k: kinds_out[k]["basis"] for k in PROBE_KINDS if kinds_out[k]["n"] or kinds_out[k]["basis"]},
            "join_ambiguous_kinds": [k for k in PROBE_KINDS if kinds_out[k]["join_ambiguous"]],
            "remainder": remainder,
            "verdict_precedence": list(PROBE_VERDICT_PRECEDENCE),
            "n_carved_from_client": dict(carved),
            "client_regime": regime, "client_note": client_note,
            "n_unclassified": n_uncl, "n_unclassified_tags": n_uncl_tags, "n_unclassified_declarations": n_uncl_decl,
            "unclassified": {"n": n_uncl,
                             "reasons": [{"reason": r, "n": n} for r, n in
                                         Counter(why for _rid, _p, why in uncl).most_common()],
                             "distinct_resolved": stray["distinct_resolved"],
                             "distinct_max_tokens": stray["distinct_max_tokens"],
                             "unknown_kind_tags": dict(unknown_kind_tags)},
            "tag_join": PROBE_TAG_JOIN,
            "client_rows": client_left}


def partition(resolved, client_caps: set, probe_caps: set) -> dict:
    """Requests by CLASS, keyed on the cap each producer declares — never on how
    a row looks. Refuses when a client cap equals a probe cap: the classes
    would be indistinguishable."""
    both = sorted(client_caps & probe_caps, key=str)
    if both:
        raise ValueError(f"max_tokens {both} is both a client cap (a --sent-* row / "
                         f"--client-cap) and a probe cap (--probe-cap / "
                         f"served_target.PROBE_MAX_TOKENS) — the request classes cannot "
                         f"be told apart, so no verdict is possible")
    classes = {"client": [], "probe": [], "unclassified": []}
    for rid, p in resolved:
        cap = str(p.get("max_tokens"))
        if cap in client_caps:
            classes["client"].append((rid, p))
        elif cap in probe_caps:
            classes["probe"].append((rid, p))
        elif not client_caps:
            classes["client"].append((rid, p))   # no client cap known: everything non-probe is client
        else:
            classes["unclassified"].append((rid, p))
    return classes


def compare(resolved, sent_distinct, expect: dict) -> dict:
    """Distinct resolved tuples, and every key that resolved to something
    other than the class's reference (``--expect``, else the single sampler
    the rows agree on). ``max_tokens`` is tallied apart and never enters
    ``divergences``: the server computes it and may clamp it."""
    counts: Counter = Counter()
    caps: Counter = Counter()
    for _rid, p in resolved:
        counts[json.dumps({k: p[k] for k in SAMPLER_KEYS if k in p},
                          sort_keys=True)] += 1
        if "max_tokens" in p:
            caps[str(p["max_tokens"])] += 1
    distinct = [{"sampler": json.loads(k), "n": v} for k, v in counts.most_common()]
    reference = dict(expect) if expect else None
    ref_source = "--expect" if expect else None
    if reference is None and len(sent_distinct) == 1:
        reference = sent_distinct[0]["sampler"]
        ref_source = "the client rows' recorded sampler"
    divergences = []
    if reference is not None:
        for block in distinct:
            got = block["sampler"]
            for k, want in reference.items():
                if k not in got:
                    divergences.append(
                        {"key": k, "sent": want, "resolved": None,
                         "note": "the engine's params line does not state this "
                                 "key — it is outside SamplingParams.__repr__, "
                                 "so it cannot be read back this way",
                         "n_requests": block["n"]})
                elif float(got[k]) != float(want):
                    divergences.append(
                        {"key": k, "sent": want, "resolved": got[k],
                         "n_requests": block["n"],
                         "equals_vllm_neutral_default":
                             (k in VLLM_NEUTRAL_DEFAULTS
                              and float(got[k]) == float(VLLM_NEUTRAL_DEFAULTS[k]))})
    return {"distinct_resolved": distinct,
            "distinct_max_tokens": [{"max_tokens": k, "n": v}
                                    for k, v in caps.most_common()],
            "reference": reference, "reference_source": ref_source,
            "divergences": divergences}


EXPECT_TASKS_SCHEMA = "resolved_sampler_task_expectations/1"
_TASK_KEYS = ("caps", "sampler", "source")            # `why` optional: the producer's reason a task has no reference


def read_task_expectations(path: Path) -> dict:
    """The producer's per-task expectation document (``--expect-tasks``), validated and normalised: ``{"tasks": {task: {"caps": [int, ...],
    "sampler": {key: float} | None, "source": str, "why": str | None}}}`` plus the document's other top-level keys verbatim (they ride into the
    record). Raises ValueError naming the refusal: not JSON / not the schema / no tasks; a task with no caps or a non-int cap; a sampler key
    outside SAMPLER_KEYS or a non-numeric value; two tasks at one cap with DIFFERENT non-null samplers (indistinguishable on the wire)."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"--expect-tasks {p}: unreadable / not JSON ({type(e).__name__}: {e})")
    if not isinstance(doc, dict) or doc.get("schema") != EXPECT_TASKS_SCHEMA:
        raise ValueError(f"--expect-tasks {p}: not a {EXPECT_TASKS_SCHEMA} document (schema {doc.get('schema') if isinstance(doc, dict) else None!r})")
    tasks = doc.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"--expect-tasks {p}: `tasks` must be a non-empty object (task -> {{caps, sampler, source}})")
    norm: dict = {}
    for task, spec in tasks.items():
        if not isinstance(spec, dict) or any(k not in spec for k in _TASK_KEYS):
            raise ValueError(f"--expect-tasks {p}: task {task!r} must carry {list(_TASK_KEYS)}")
        caps = spec["caps"]
        if not isinstance(caps, list) or not caps or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in caps):
            raise ValueError(f"--expect-tasks {p}: task {task!r} caps {caps!r} must be a non-empty list of positive ints (the max_tokens its requests carry)")
        sampler = spec["sampler"]
        if sampler is not None:
            if not isinstance(sampler, dict) or not sampler:
                raise ValueError(f"--expect-tasks {p}: task {task!r} sampler must be a non-empty object of sampler keys, or null (no reference)")
            bad = [k for k in sampler if k not in SAMPLER_KEYS]
            if bad:
                raise ValueError(f"--expect-tasks {p}: task {task!r} sampler key(s) {bad} are not sampler keys {SAMPLER_KEYS}")
            try:
                sampler = {k: float(v) for k, v in sampler.items()}
            except (TypeError, ValueError):
                raise ValueError(f"--expect-tasks {p}: task {task!r} sampler values must be numeric ({sampler!r})")
        norm[str(task)] = {"caps": sorted({str(int(c)) for c in caps}, key=int), "sampler": sampler, "source": str(spec.get("source") or ""),
                           "why": (str(spec["why"]) if spec.get("why") is not None else None)}
    by_cap: dict = {}
    for task, spec in norm.items():
        for c in spec["caps"]:
            by_cap.setdefault(c, []).append(task)
    for c, names in by_cap.items():
        refs = {json.dumps(norm[t]["sampler"], sort_keys=True) for t in names if norm[t]["sampler"] is not None}
        if len(refs) > 1:
            raise ValueError(f"--expect-tasks {p}: tasks {names} all declare max_tokens {c} with DIFFERENT samplers -- their requests cannot be told "
                             "apart on the wire, so no per-task verdict is possible (declare distinct caps, or one shared reference)")
    return {**{k: v for k, v in doc.items() if k != "tasks"}, "tasks": norm, "by_cap": {c: sorted(names) for c, names in by_cap.items()}}


def compare_by_task(resolved, expectations: dict) -> dict:
    """The CLIENT class compared PER TASK (``--expect-tasks``): the requests grouped by cap, each cap group held to the reference of the task(s)
    declaring that cap (`compare`), every task given its own verdict -- ``EMPTY`` (no request at its caps), ``UNCOMPARED`` (a null reference, or
    a shared cap whose other task has none), ``DIVERGED`` / ``AGREES``. A client cap no task declares is its own UNCOMPARED group, named. Returns
    the class block: the whole-class tuples, ``reference`` None (there is no single one), ``divergences`` flattened and stamped with the task(s)
    and cap, ``tasks`` and ``by_cap``."""
    tasks, by_cap = expectations["tasks"], expectations["by_cap"]
    groups: dict = {}
    for rid, p in resolved:
        groups.setdefault(str(p.get("max_tokens")), []).append((rid, p))
    caps = sorted(set(groups) | set(by_cap), key=lambda c: int(c) if str(c).isdigit() else -1)
    by_cap_out: dict = {}
    divergences: list = []
    for cap in caps:
        rows = groups.get(cap, [])
        names = list(by_cap.get(cap, []))
        refs = [tasks[t]["sampler"] for t in names if tasks[t]["sampler"] is not None]
        unref = [t for t in names if tasks[t]["sampler"] is None]
        reference = refs[0] if refs and not unref else None
        cmp_ = compare(rows, [], reference or {})
        if reference is None:
            cmp_["reference"], cmp_["reference_source"] = None, None
            why = (f"task(s) {unref} declare no sampler reference" + (f" ({'; '.join(str(tasks[t]['why']) for t in unref if tasks[t]['why'])})" if any(tasks[t]["why"] for t in unref) else "")
                   if names else f"no task declares max_tokens {cap}")
        else:
            src_tasks = [t for t in names if tasks[t]["sampler"] is not None]
            cmp_["reference_source"] = "--expect-tasks: " + ", ".join(f"{t} ({tasks[t]['source'] or 'no source stated'})" for t in src_tasks)
            why = None
        if not rows:
            verdict = "EMPTY"
        elif reference is None:
            verdict = "UNCOMPARED"
        elif cmp_["divergences"]:
            verdict = "DIVERGED"
        else:
            verdict = "AGREES"
        for d in cmp_["divergences"]:
            divergences.append({**d, "tasks": names, "max_tokens": cap})
        by_cap_out[cap] = {"tasks": names, "n": len(rows), "verdict": verdict, "why": why, **cmp_}
    task_blocks: dict = {}
    for t, spec in tasks.items():
        gs = [by_cap_out[c] for c in spec["caps"]]
        n = sum(g["n"] for g in gs)
        verdicts = {g["verdict"] for g in gs if g["n"]}
        if not n:
            v = "EMPTY"
        elif "DIVERGED" in verdicts:
            v = "DIVERGED"
        elif "UNCOMPARED" in verdicts:
            v = "UNCOMPARED"
        else:
            v = "AGREES"
        distinct: Counter = Counter()
        for g in gs:
            for b in g["distinct_resolved"]:
                distinct[json.dumps(b["sampler"], sort_keys=True)] += b["n"]
        task_blocks[t] = {"caps": list(spec["caps"]), "n": n, "verdict": v, "reference": spec["sampler"], "reference_source": (spec["source"] or None),
                          "why": (spec["why"] if spec["sampler"] is None else None),
                          "distinct_resolved": [{"sampler": json.loads(k), "n": c} for k, c in distinct.most_common()],
                          "distinct_max_tokens": [{"max_tokens": c, "n": by_cap_out[c]["n"]} for c in spec["caps"]],
                          "divergences": [d for d in divergences if t in d["tasks"]],
                          "shares_cap_with": sorted({o for c in spec["caps"] for o in by_cap_out[c]["tasks"] if o != t})}
    whole = compare(resolved, [], {})
    return {"distinct_resolved": whole["distinct_resolved"], "distinct_max_tokens": whole["distinct_max_tokens"],
            "reference": None, "reference_source": "--expect-tasks: one reference per task (classes.client.tasks)",
            "divergences": divergences, "tasks": task_blocks, "by_cap": by_cap_out}


def _client_tasks_verdict(block: dict, n_rows: int) -> str:
    if not n_rows:
        return "EMPTY"
    vs = {g["verdict"] for g in block["by_cap"].values() if g["n"]}
    if "DIVERGED" in vs:
        return "DIVERGED"
    if "UNCOMPARED" in vs:
        return "UNCOMPARED"
    return "AGREES"


def _fmt(sampler: dict) -> str:
    return " ".join(f"{k}={sampler[k]}" for k in SAMPLER_KEYS if k in sampler) or "(no sampler key)"


def _write(out: Path, rec: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=False, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="EXIT CODE: 0 for EVERY verdict that produced a record (UNREADABLE included); 2 for a "
                                        "refused argument / refused declarations; --strict maps every verdict to an exit code; "
                                        "--require-wired fails ONLY when the record proves the instrument was never wired "
                                        "(UNREADABLE 3, FLAG_SET_NO_REQUESTS_YET 4) and keeps every finding about the sampler "
                                        "at 0. A caller that consumes the exit code must read the record's `verdict` field or "
                                        "pass --require-wired; the two flags together are refused.")
    ap.add_argument("--vllm-log", required=True, type=Path, action="append",
                    help="a vLLM server log (repeatable)")
    ap.add_argument("--sent-static", type=Path, action="append", default=[],
                    help="a static family responses.jsonl (repeatable)")
    ap.add_argument("--sent-controls", type=Path, action="append", default=[],
                    help="a judged control store's gen_stats.json (the canonical layout: the "
                         "decode block per control) or a legacy per-item meta.jsonl (repeatable)")
    ap.add_argument("--expect", action="append", default=[],
                    help="key=value the CLIENT class must equal (repeatable; wins over the rows)")
    ap.add_argument("--expect-tasks", type=Path, default=None,
                    help=f"a {EXPECT_TASKS_SCHEMA} JSON document: the CLIENT class's expected sampler PER TASK, each task with the max_tokens its "
                         "requests carry (`caps`), its `sampler` (or null = no reference: that task reads UNCOMPARED by name) and its `source`. "
                         "The client class is partitioned by cap and each task gets its own verdict (classes.client.tasks, client_task_verdicts); "
                         "two tasks at one cap with different samplers are refused. Beside --expect only under --probe-expect card/kinds (where "
                         "--expect is the probe / render reference)")
    ap.add_argument("--client-cap", type=int, action="append", default=[],
                    help="a max_tokens the client legs send, in addition to the caps the --sent-* "
                         "rows recorded (repeatable; e.g. the controls driver's render probe, 512)")
    ap.add_argument("--probe-cap", type=int, action="append", default=None,
                    help="a max_tokens the GREEDY scaffolding sends (repeatable). Default: the served "
                         "target's PROBE_MAX_TOKENS alone; a controls leg also declares its MMLU "
                         "scoring cap (1) and the thinking-ON bite probe (256). Under --probe-expect kinds "
                         f"the default is every cap the probe-kind table pins ({', '.join(str(c) for c in sorted(_PINNED))}) plus any "
                         "--declare-kind cap that is not a client cap")
    ap.add_argument("--probe-expect", choices=PROBE_EXPECT_CHOICES, default="greedy",
                    help="what the PROBE class is compared against: 'greedy' (default) = the scaffolding's "
                         "own body, temperature 0.0 + vLLM's neutral defaults; 'card' = the SAME --expect "
                         "values the client class is checked against (the launcher's projection of the "
                         "pinned card row) — for a driver whose render probe decodes under the row's card. "
                         "'card' without any --expect is refused (exit 2): the reference comes from the "
                         "pinned table, never from the driver's declaration. 'kinds' = the probe class "
                         "partitioned by KIND — the driver's tag (--probe-tags) or an explicit --declare-kind, "
                         "one identity source per kind — each kind against its own pinned reference (render -> "
                         "--expect, the pinned card row; bites, lens_on_bite, load_check, positive_control -> "
                         "greedy), the shape cross-checked against the kind table; a request matching no kind is "
                         "UNCLASSIFIED and fails the probe class, EXCEPT that on a driver with ZERO tagged requests "
                         "the untagged, undeclared remainder is NOT_INSTRUMENTED (descriptive, exit 0)")
    ap.add_argument("--probe-tags", type=Path, action="append", default=[],
                    help="(--probe-expect kinds) a request log carrying the kind tag per attempt: the driver's "
                         "<out>/request_evidence.jsonl (task probe/<kind>, max_tokens), the standup's "
                         "load_check.request_evidence.jsonl / positive_control_<tag>.request_evidence.jsonl, or "
                         "usage_ledger.jsonl (task, max_tokens_requested) — repeatable, one per EMITTER; never the "
                         "evidence file AND the ledger for the same requests (the counts would double). A named file "
                         "that is missing is recorded as such: with no tagged request in the whole run the probe "
                         "remainder reads NOT_INSTRUMENTED")
    ap.add_argument("--declare-kind", action="append", default=[], metavar=DECLARE_KIND_GRAMMAR,
                    help="(--probe-expect kinds) the identity of harness traffic a standup that stamps NO tag fired — "
                         "FIRST-CLASS, on the same footing as the tag. kind and cap are required; count=N is a MAXIMUM "
                         "(fewer passes; 0 -> the kind is EMPTY; more -> UNCLASSIFIED), count=N! is exact; count is "
                         "optional for a kind whose cap differs from the client cap (cap + the verified greedy shape is "
                         "the identity). load_check at the CLIENT cap: with a sampled client reference (--expect "
                         "temperature > 0) the greedy shape separates it (count optional / a maximum); with a greedy one "
                         "an exact count=N! is REQUIRED (refused otherwise). Any mismatch with the log (the kind's pinned "
                         "cap, count, greedy shape) is UNCLASSIFIED, never folded. Repeatable, one per kind; render (the "
                         "driver's own tagged kind) is refused; a declaration for a kind the tags already name must AGREE "
                         "with the tag (basis tag+declared) or the kind is UNCLASSIFIED. A tag BINDS a request to a kind; a "
                         "declaration COUNTS (the recorder decides which requests by the cap-multiset join) — each kind's "
                         "block names its basis, and several greedy kinds split by count at one cap read join_ambiguous. "
                         "E.g. load_check:cap=8192  positive_control:cap=64:count=48  bites:cap=48:count=80!")
    ap.add_argument("--strict", action="store_true",
                    help="map the verdict to the exit code (DIVERGED 1; UNREADABLE / UNCOMPARED / "
                         "UNCLASSIFIED_REQUESTS 3; FLAG_SET_NO_REQUESTS_YET 4; PROBE_NOT_INSTRUMENTED 0). Default: exit 0 "
                         "for every recorded verdict — a finding, not a job failure")
    ap.add_argument("--require-wired", action="store_true",
                    help="the caller DECLARES it requested request logging (--enable-log-requests) and is reading AFTER "
                         "the legs, so a record proving the instrument was never wired is a failure: UNREADABLE 3 (no "
                         "readable log; silence with no attestation = launched without the flag; attested but MUTED); "
                         "FLAG_SET_NO_REQUESTS_YET 4 (a post-leg read that finds no request is an ordering failure). "
                         "Every finding ABOUT the sampler stays exit 0 — AGREES, DIVERGED, UNCOMPARED, "
                         "UNCLASSIFIED_REQUESTS, PROBE_NOT_INSTRUMENTED — deliberately: a fatal DIVERGED would reintroduce "
                         "the false failure the default exists to prevent, and PROBE_NOT_INSTRUMENTED is the permanent "
                         "reading of a driver that stamps no probe tag. Refused together with --strict (exit 2)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.strict and a.require_wired:
        print("REFUSED: --strict and --require-wired name two different gates over one exit code — --strict maps EVERY "
              "non-agreeing verdict to a failure (DIVERGED included); --require-wired fails ONLY when the record proves "
              "the instrument was never wired (UNREADABLE, FLAG_SET_NO_REQUESTS_YET) and keeps every finding about the "
              "sampler at 0. Pass one", file=sys.stderr)
        return 2

    expect = {}
    for kv in a.expect:
        k, _, v = kv.partition("=")
        if k not in SAMPLER_KEYS:
            print(f"REFUSED: --expect {kv!r}: {k!r} is not a sampler key "
                  f"{SAMPLER_KEYS}", file=sys.stderr)
            return 2
        expect[k] = float(v)
    task_exp = None
    if a.expect_tasks is not None:
        if expect and a.probe_expect == "greedy":
            print("REFUSED: --expect-tasks and --expect both name the CLIENT class's reference (under --probe-expect greedy --expect is the client "
                  "reference); pass one -- --expect is admitted beside --expect-tasks only as the probe / render reference (card / kinds)", file=sys.stderr)
            return 2
        try:
            task_exp = read_task_expectations(a.expect_tasks)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
    if a.probe_expect in ("card", "kinds") and not expect:
        print(f"REFUSED: --probe-expect {a.probe_expect} needs at least one --expect k=v — the "
              + ("render kind's" if a.probe_expect == "kinds" else "probe class's")
              + " reference is the pinned card row the launcher passes, never the driver's own "
              "declaration of what it decoded under", file=sys.stderr)
        return 2
    # THE STANDUP'S OWN TAG SIDECAR (lane B2, 2026-09-30): native_standup tags every request it sends (probe/standup_controls) and
    # writes `positive_controls.request_evidence.jsonl` beside its vllm_server.log; under kinds it is an identity source like any
    # --probe-tags file and is DISCOVERED beside each --vllm-log when the caller did not name it (recorded as `discovered`), so a
    # launcher that passes the driver's and the load check's sidecars alone no longer reads the standup's controls as UNCLASSIFIED
    discovered = []
    standup_attempt = None
    if a.probe_expect == "kinds":
        named = {str(Path(p).resolve()) for p in a.probe_tags}
        for lg in a.vllm_log:
            cand = Path(lg).parent / STANDUP_EVIDENCE_FILENAME
            if cand.is_file() and str(cand.resolve()) not in named:
                a.probe_tags.append(cand)
                discovered.append(str(cand))
            # the CURRENT attempt's identity beside the log (endpoint.json standup_attempt_id): the standup's rows are filtered to it
            ep = Path(lg).parent / STANDUP_ENDPOINT_FILENAME
            if ep.is_file():
                try:
                    v = json.loads(ep.read_text(errors="replace")).get(STANDUP_ENDPOINT_ATTEMPT_KEY)
                    standup_attempt = str(v) if v else standup_attempt
                except (ValueError, OSError, AttributeError):
                    pass
    if a.probe_expect == "kinds" and not a.probe_tags and not a.declare_kind:
        print("REFUSED: --probe-expect kinds needs an identity source — --probe-tags <request_evidence.jsonl|usage_ledger.jsonl> "
              "(the driver's own tag, X-Tinycap-Task: probe/<kind>, read back) and/or --declare-kind "
              f"{DECLARE_KIND_GRAMMAR} (a standup that stamps no tag declares what it fired); a kind is never "
              "inferred from how a params line looks alone (classification never defaults)", file=sys.stderr)
        return 2
    if a.probe_tags and a.probe_expect != "kinds":
        print(f"REFUSED: --probe-tags classifies the probe class by kind and is read only under "
              f"--probe-expect kinds (got --probe-expect {a.probe_expect}); drop it or switch modes — a "
              "tag file passed and silently ignored would read as a check that ran", file=sys.stderr)
        return 2
    if a.declare_kind and a.probe_expect != "kinds":
        print(f"REFUSED: --declare-kind declares a probe KIND and is read only under --probe-expect kinds (got "
              f"--probe-expect {a.probe_expect}); a declaration passed and silently ignored would read as a "
              "check that ran", file=sys.stderr)
        return 2
    declarations = []
    for text in a.declare_kind:
        try:
            declarations.append(parse_declaration(text))
        except ValueError as e:
            print(f"REFUSED: --declare-kind {text!r}: {e} (grammar: {DECLARE_KIND_GRAMMAR})", file=sys.stderr)
            return 2
    dup = [k for k, n in Counter(d["kind"] for d in declarations).items() if n > 1]
    if dup:
        print(f"REFUSED: --declare-kind names {dup} more than once — one declaration per kind (a kind fires at one "
              "cap per standup; a second cap is a second standup and a second record)", file=sys.stderr)
        return 2

    if a.probe_cap is not None:
        probe_caps = {str(int(c)) for c in a.probe_cap}
        probe_cap_source = "--probe-cap"
    elif a.probe_expect == "kinds":
        probe_caps = {str(c) for c in _PINNED}
        probe_cap_source = ("the probe-kind table (--probe-expect kinds): " + "; ".join(
            f"{k} {_caps_label(k) if _pinned_caps(k) is not None else '[the per-leg cap]'}" for k in PROBE_KINDS))
    else:
        dflt = default_probe_cap()
        probe_caps = {str(dflt)} if dflt is not None else set()
        probe_cap_source = ("sisteer.eval.attacks.served_target.PROBE_MAX_TOKENS" if dflt is not None
                            else "NONE: sisteer is not importable and no --probe-cap was given — there "
                                 "is no probe class; greedy probes, if any, are read as client requests")

    logs = [read_log(p) for p in a.vllm_log if Path(p).exists()]
    missing = [str(p) for p in a.vllm_log if not Path(p).exists()]
    resolved_all = [r for lg in logs for r in lg["resolved"]]
    sent = read_sent(list(a.sent_static) + list(a.sent_controls))
    row_caps = {str(c["max_tokens"]) for c in sent["distinct_max_tokens"]}
    client_caps = row_caps | {str(int(c)) for c in a.client_cap}
    client_cap_source = ("the --sent-* rows' recorded caps" if row_caps else "") + (
        (" + " if row_caps and a.client_cap else "") + ("--client-cap" if a.client_cap else ""))
    if task_exp is not None:
        task_caps = {c for spec in task_exp["tasks"].values() for c in spec["caps"]}
        client_caps |= task_caps
        client_cap_source = (client_cap_source + " + " if client_cap_source else "") + f"--expect-tasks ({len(task_exp['tasks'])} task(s) at max_tokens {sorted(task_caps, key=int)})"
    # a declared cap that is not a client cap IS a producer's declaration of a probe cap
    declared_probe_caps = sorted({d["cap"] for d in declarations if d["cap"] not in client_caps}, key=int)
    if declared_probe_caps:
        probe_caps |= set(declared_probe_caps)
        probe_cap_source += f" + --declare-kind caps {declared_probe_caps}"
    class_rule = (f"client: max_tokens in {sorted(client_caps, key=int) or 'ANY non-probe cap (no client cap known)'}"
                  f" ({client_cap_source or 'none declared'}); probe: max_tokens in "
                  f"{sorted(probe_caps, key=int)} ({probe_cap_source}); a class is compared only "
                  f"against ITS reference; anything else is unclassified, never folded in")

    base = {"n_logs": len(logs), "missing_logs": missing,
            "n_resolved_requests": len(resolved_all),
            "class_rule": class_rule,
            "client_caps": sorted(client_caps, key=int), "client_cap_source": client_cap_source or None,
            "probe_caps": sorted(probe_caps, key=int), "probe_cap_source": probe_cap_source,
            "probe_reference": probe_reference_block(a.probe_expect, expect),
            "vllm_neutral_defaults": VLLM_NEUTRAL_DEFAULTS,
            "exit_policy": ("--strict: the verdict sets the exit code" if a.strict else
                            ("--require-wired: exit 0 for every finding ABOUT the sampler (AGREES, DIVERGED, "
                             "UNCOMPARED, UNCLASSIFIED_REQUESTS, PROBE_NOT_INSTRUMENTED); non-zero ONLY when the "
                             "record proves the instrument the caller declared was never wired (UNREADABLE 3, "
                             "FLAG_SET_NO_REQUESTS_YET 4); 2 only for a refused argument or refused declarations")
                            if a.require_wired else
                            ("recorded, not propagated: exit 0 for every recorded verdict (a finding "
                             "about the serving stack, not a defect in the evaluation data); 2 only "
                             "for a refused argument or refused declarations")),
            "generation_config_override": [lg["generation_config_override"] for lg in logs],
            "sent": sent,
            "logs": [{k: v for k, v in lg.items() if k != "resolved"} for lg in logs]}
    tags = None
    if a.probe_expect == "kinds":
        tags = read_probe_tags(a.probe_tags, standup_attempt=standup_attempt)
        # the tag source in the record (its rows are folded into the per-kind blocks, not repeated)
        base["probe_tags"] = {"named": bool(a.probe_tags), "discovered": discovered, **{k: v for k, v in tags.items() if k != "rows"}}
        base["declarations"] = declarations
        base["declare_kind_grammar"] = DECLARE_KIND_GRAMMAR
        t_ref = _num(expect.get("temperature")) if "temperature" in expect else None
        regime = None if t_ref is None else ("greedy" if t_ref == 0.0 else "sampled")
        need_exact = [d for d in declarations if d["cap"] in client_caps and not d["exact"] and regime != "sampled"]
        if need_exact:
            why = (f"--declare-kind {[d['raw'] for d in need_exact]}: a declaration at the CLIENT cap "
                   f"{sorted({d['cap'] for d in need_exact}, key=int)} under a "
                   + ("GREEDY client reference (--expect temperature 0.0)" if regime == "greedy" else
                      "client reference that states no temperature (--expect names none)")
                   + " needs an exact count=N! — the greedy shape cannot separate the kind from client traffic there, "
                     "only the count can; with count=N! the kind takes N by count from identical greedy-shaped requests")
            rec = {"verdict": "REFUSED", "why": why, **base}
            _write(a.out, rec)
            print(f"REFUSED: {why}", file=sys.stderr)
            print(f"[resolved-sampler] REFUSED -> {a.out}")
            return 2

    try:
        classes = partition(resolved_all, client_caps, probe_caps)
    except ValueError as e:
        rec = {"verdict": "REFUSED", "why": str(e), **base}
        _write(a.out, rec)
        print(f"REFUSED: {e}", file=sys.stderr)
        print(f"[resolved-sampler] REFUSED -> {a.out}")
        return 2

    kinds_block = None
    if a.probe_expect == "kinds":
        # by KIND: the tag or a declaration classifies, the shape cross-checks, each kind against its own pinned
        # reference; the carve also rewrites the client class (load_check lives at the client cap)
        kinds_block = classify_kinds(classes["probe"], classes["client"], tags=tags, declarations=declarations,
                                     expect=expect, client_caps=client_caps)
        classes["client"] = kinds_block.pop("client_rows")

    per_class = {}
    for name, rows in classes.items():
        if name == "client" and task_exp is not None:
            # per TASK -- the class partitioned by the caps the tasks declare, each group against ITS task's reference
            cmp_ = compare_by_task(rows, task_exp)
            cmp_["reference_kind"] = "tasks"
            per_class[name] = {"n": len(rows), "verdict": _client_tasks_verdict(cmp_, len(rows)), **cmp_}
            continue
        if name == "client":
            cmp_ = compare(rows, sent["distinct"], expect)
            cmp_["reference_kind"] = ("expect" if expect else
                                      "rows" if cmp_["reference"] is not None else None)
        elif name == "probe" and a.probe_expect == "kinds":
            per_class[name] = kinds_block
            continue
        elif name == "probe" and a.probe_expect == "card":
            # the pinned card row: the SAME --expect values the client class is checked against
            cmp_ = compare(rows, [], expect)
            cmp_["reference_source"] = PROBE_CARD_REFERENCE_SOURCE if rows else None
            cmp_["reference_kind"] = "card"
        elif name == "probe":
            cmp_ = compare(rows, [{"sampler": probe_reference(), "n": len(rows)}], {})
            cmp_["reference_source"] = PROBE_REFERENCE_SOURCE if rows else None
            cmp_["reference_kind"] = "greedy"
        else:
            cmp_ = compare(rows, [], {})
            cmp_["reference_kind"] = None
            # never folded in, but SAID: does each stray tuple equal the client reference?
            # (a server- or proxy-clamped max_tokens lands a client request here — the logged
            # cap is not the sent cap — and this is how a reader tells that from a stray sampler)
            crefs = ([per_class["client"]["reference"]] if per_class["client"].get("reference") else
                     [t["reference"] for t in per_class["client"].get("tasks", {}).values() if t.get("reference")])
            for block in cmp_["distinct_resolved"]:
                got = block["sampler"]
                block["matches_client_reference"] = any(
                    all(k in got and float(got[k]) == float(v) for k, v in cref.items()) for cref in crefs if cref)
        if not rows:
            v = "EMPTY"
        elif name == "unclassified":
            v = "UNCLASSIFIED"
        elif cmp_["reference"] is None:
            v = "UNCOMPARED"
        elif cmp_["divergences"]:
            v = "DIVERGED"
        else:
            v = "AGREES"
        per_class[name] = {"n": len(rows), "verdict": v, **cmp_}
    n_client = per_class["client"]["n"]
    n_probe = per_class["probe"]["n"]
    n_uncl_stray = per_class["unclassified"]["n"]
    # kinds mode: a probe no kind classified is unclassified too — counted, named, never folded
    n_uncl_probe = int(per_class["probe"].get("n_unclassified") or 0)
    n_uncl_tags = int(per_class["probe"].get("n_unclassified_tags") or 0)
    n_uncl_decl = int(per_class["probe"].get("n_unclassified_declarations") or 0)
    n_uncl = n_uncl_stray + n_uncl_probe
    remainder = per_class["probe"].get("remainder") if a.probe_expect == "kinds" else None
    remainder_ni = bool(remainder) and remainder["verdict"] == VERDICT_NOT_INSTRUMENTED
    diverged = [n for n, c in per_class.items() if c["verdict"] == "DIVERGED"]

    def _diverged_clause(name: str) -> str:
        c = per_class[name]
        if name == "client" and task_exp is not None:
            return "client: " + "; ".join(
                f"task {t}: {len(tb['divergences'])} key(s) resolved to a value other than its reference ({tb['reference_source']})"
                for t, tb in c["tasks"].items() if tb["verdict"] == "DIVERGED")
        if name == "probe" and a.probe_expect == "kinds":
            return "probe: " + "; ".join(
                f"kind {k}: {len(kb['divergences'])} key(s) resolved to a value other than its reference "
                f"({kb['reference_source']})" for k, kb in c["kinds"].items() if kb["verdict"] == "DIVERGED")
        return (f"{name}: {len(c['divergences'])} key(s) resolved to a value other than its reference "
                f"({c['reference_source']})")

    def _kinds_unclassified_clause() -> str:
        c = per_class["probe"]
        parts = []
        if n_uncl_probe:
            reasons = "; ".join(f"{r['n']} x {r['reason']}" for r in c["unclassified"]["reasons"])
            parts.append(f"{n_uncl_probe} request(s) matched no kind ({reasons})")
        if n_uncl_tags:
            parts.append(f"{n_uncl_tags} tag row(s) bound to no classified kind"
                         + (f" (unknown kinds: {c['unclassified']['unknown_kind_tags']})"
                            if c["unclassified"]["unknown_kind_tags"] else ""))
        if n_uncl_decl:
            parts.append(f"{n_uncl_decl} declaration(s) bound to no classified kind")
        return ("the probe class could not be classified by kind: " + "; ".join(parts)
                + " — nothing folded into a kind, nothing compared against a guess (see "
                  "`classes.probe.kinds` / `classes.probe.unclassified`)")

    def _kinds_agree_clause() -> str:
        kb = per_class["probe"]["kinds"]
        named = [k for k in PROBE_KINDS if kb[k]["n"]]
        if not named:
            return ""
        return "; probe " + str(n_probe) + " request(s) by kind: " + ", ".join(
            f"{k} {kb[k]['n']} = " + ("the pinned card row (--expect)" if kb[k]["reference_kind"] == "card"
                                      else "greedy + neutral defaults")
            + f" ({kb[k]['basis']}{'; join_ambiguous' if kb[k]['join_ambiguous'] else ''})" for k in named) + (
            " (--probe-expect kinds; every tag / declaration agrees with its shape)")

    def _remainder_clause() -> str:
        r = remainder
        shapes = ", ".join(f"{x['n']} {x['shape']} at max_tokens {x['max_tokens']}" for x in r["shape_counts"])
        src = r["tag_source"]
        how = ("no --probe-tags named" if not src["named"] else
               f"tag source missing: {src['missing']}" if src["missing"] and not src["sources"] else
               f"tag source carries no probe/ row: {src['sources']}")
        return (f"probe remainder NOT_INSTRUMENTED: {r['n']} untagged, undeclared request(s) ({shapes}) on a driver that "
                f"stamps no kind tag ({how}; reason {r['reason']}) — {r['consequence']}; reported descriptively, compared "
                f"against nothing (no shape-based fallback); declare the standup's kinds with --declare-kind to attest them")

    if not logs:
        verdict = "UNREADABLE"
        why = f"no vLLM log was readable (missing: {missing})"
    elif not resolved_all and any(lg["log_requests_muted"] for lg in logs):
        verdict = "UNREADABLE"
        why = ("the server attests `--enable-log-requests` but MUTED — its log level is above "
               "INFO ('No request information will be logged'), so no request will ever be "
               "recorded. Set VLLM_LOGGING_LEVEL=INFO. Silence here is NOT agreement")
    elif not resolved_all and any(lg["log_requests_flag_attested"] for lg in logs):
        verdict = "FLAG_SET_NO_REQUESTS_YET"
        why = ("the server attests `--enable-log-requests` (request_logger.py, at start) but has "
               "logged no request yet — this readback ran before any request was sent (a "
               "standup-time call). It says nothing about the sampler either way; read again "
               "after the legs")
    elif not resolved_all:
        verdict = "UNREADABLE"
        why = ("the server logged no `Received request ... params: SamplingParams(...)` line "
               "and never attested the flag — it was launched WITHOUT --enable-log-requests, so "
               "what resolved is unrecorded. Silence here is NOT agreement")
    elif diverged:
        verdict = "DIVERGED"
        why = ("within-class mismatch in " + ", ".join(diverged) + ": "
               + "; ".join(_diverged_clause(n) for n in diverged) + " — see `classes`"
               + (("; ALSO " + _kinds_unclassified_clause()) if (n_uncl_probe or n_uncl_tags or n_uncl_decl) else "")
               + (("; ALSO " + _remainder_clause()) if remainder_ni else ""))
    elif n_client and per_class["client"]["verdict"] == "UNCOMPARED" and task_exp is not None:
        verdict = "UNCOMPARED"
        unc = {c: g for c, g in per_class["client"]["by_cap"].items() if g["n"] and g["verdict"] == "UNCOMPARED"}
        why = ("client requests were read for which no per-task reference exists: " + "; ".join(
            f"{g['n']} request(s) at max_tokens {c}: {g['why']}" for c, g in unc.items())
               + " -- UNCOMPARED by task, never folded into a neighbour's reference (see `classes.client.tasks`)")
    elif n_client and per_class["client"]["verdict"] == "UNCOMPARED":
        verdict = "UNCOMPARED"
        why = ("client requests were read, but there is no single reference to compare them "
               "against (pass --expect, or point --sent-* at rows that agree on one sampler)")
    elif n_uncl_probe or n_uncl_tags or n_uncl_decl:
        verdict = "UNCLASSIFIED_REQUESTS"
        why = _kinds_unclassified_clause() + (
            f"; and {n_uncl_stray} request(s) match neither a client cap {sorted(client_caps, key=int)} nor a "
            f"probe cap {sorted(probe_caps, key=int)} (see `classes.unclassified`)" if n_uncl_stray else "") + (
            ("; ALSO " + _remainder_clause()) if remainder_ni else "")
    elif n_uncl:
        verdict = "UNCLASSIFIED_REQUESTS"
        why = (f"every compared class agrees, but {n_uncl} request(s) match neither a client cap "
               f"{sorted(client_caps, key=int)} nor a probe cap {sorted(probe_caps, key=int)}; "
               "nothing is folded into a class it does not belong to — declare their cap with "
               "--client-cap or --probe-cap (see `classes.unclassified`; a server- or "
               "proxy-clamped max_tokens lands here, and each stray tuple says whether it "
               f"matches the client reference: "
               f"{sum(b['n'] for b in per_class['unclassified']['distinct_resolved'] if b.get('matches_client_reference'))}"
               f" of {n_uncl} do)") + (("; ALSO " + _remainder_clause()) if remainder_ni else "")
    elif remainder_ni:
        verdict = "PROBE_NOT_INSTRUMENTED"
        why = ("the client class is internally uniform and matches its reference: "
               f"client {n_client} request(s) = {per_class['client']['reference_source']}"
               + (" (per task: " + ", ".join(f"{t} {tb['verdict']}" for t, tb in per_class["client"]["tasks"].items()) + ")" if task_exp is not None else "")
               + _kinds_agree_clause()
               + "; " + _remainder_clause())
    else:
        verdict = "AGREES"
        if a.probe_expect == "kinds" and n_probe:
            probe_clause = _kinds_agree_clause()
        else:
            probe_clause = (f"; probe {n_probe} request(s) at max_tokens {sorted(probe_caps, key=int)} = "
                            + ("the pinned card row (--expect; --probe-expect card)"
                               if a.probe_expect == "card" else "greedy + neutral defaults")
                            if n_probe else "")
        if task_exp is not None:
            client_clause = f"client {n_client} request(s) per task: " + "; ".join(
                f"{t} {tb['n']} at max_tokens {tb['caps']} = {tb['reference_source'] or 'reference stated'}" + (" (EMPTY: no request)" if tb["verdict"] == "EMPTY" else "")
                for t, tb in per_class["client"]["tasks"].items())
        else:
            client_clause = f"client {n_client} request(s) = {per_class['client']['reference_source']}"
        why = "every class is internally uniform and matches its reference: " + client_clause + probe_clause
        if a.probe_expect == "kinds" and per_class["probe"].get("client_note"):
            why += "; note: " + per_class["probe"]["client_note"]

    rec = {"verdict": verdict, "why": why,
           "classes": per_class,
           "n_client_requests": n_client, "n_probe_requests": n_probe,
           "n_unclassified_requests": n_uncl,
           # --expect-tasks: the client verdict PER TASK, and the producer's expectation document verbatim
           **({"client_task_verdicts": {t: tb["verdict"] for t, tb in per_class["client"]["tasks"].items()},
               "client_task_expectations": {k: v for k, v in task_exp.items() if k != "by_cap"}} if task_exp is not None else {}),
           # kinds mode only: the counts per kind the ruling asks for, at the top level
           **({"n_probe_requests_by_kind": per_class["probe"]["n_by_kind"],
               "n_probe_unclassified_requests": n_uncl_probe,
               "n_probe_unclassified_tags": n_uncl_tags,
               "n_probe_unclassified_declarations": n_uncl_decl,
               "probe_verdict_by_kind": {k: per_class["probe"]["kinds"][k]["verdict"] for k in PROBE_KINDS},
               "probe_remainder_verdict": remainder["verdict"], "n_probe_remainder_requests": remainder["n"],
               "n_carved_from_client": per_class["probe"]["n_carved_from_client"],
               "probe_basis_by_kind": per_class["probe"]["basis_by_kind"],
               "probe_join_ambiguous_kinds": per_class["probe"]["join_ambiguous_kinds"]}
              if a.probe_expect == "kinds" else {}),
           **base,
           # the client class at top level too: what earlier readers cite
           **{k: per_class["client"][k] for k in
              ("distinct_resolved", "distinct_max_tokens", "reference", "reference_source", "divergences")}}
    _write(a.out, rec)
    print(f"[resolved-sampler] {verdict}: {why}")
    for name in ("client", "probe", "unclassified"):
        c = per_class[name]
        if name == "probe" and a.probe_expect == "kinds":
            for k in PROBE_KINDS:
                kb = c["kinds"][k]
                for block in kb["distinct_resolved"]:
                    caps_s = ",".join(x["max_tokens"] for x in kb["distinct_max_tokens"])
                    print(f"[resolved-sampler]   probe/{k:<16s} {kb['verdict']:16s} n={block['n']:6d}  "
                          f"{_fmt(block['sampler'])}  max_tokens={caps_s}  tagged={kb['n_tagged']} declared={kb['n_declared']}")
            u = c["unclassified"]
            for block in u["distinct_resolved"]:
                caps_s = ",".join(x["max_tokens"] for x in u["distinct_max_tokens"])
                print(f"[resolved-sampler]   probe/{'(no kind)':<16s} {'UNCLASSIFIED':16s} n={block['n']:6d}  "
                      f"{_fmt(block['sampler'])}  max_tokens={caps_s}")
            r = c["remainder"]
            for block in r["distinct_resolved"]:
                caps_s = ",".join(x["max_tokens"] for x in r["distinct_max_tokens"])
                print(f"[resolved-sampler]   probe/{'(remainder)':<16s} {r['verdict']:16s} n={block['n']:6d}  "
                      f"{_fmt(block['sampler'])}  max_tokens={caps_s}  reason={r['reason']}")
            continue
        for block in c["distinct_resolved"]:
            caps_s = ",".join(x["max_tokens"] for x in c["distinct_max_tokens"])
            print(f"[resolved-sampler]   {name:12s} {c['verdict']:11s} n={block['n']:6d}  "
                  f"{_fmt(block['sampler'])}  max_tokens={caps_s}")
        if name == "client" and task_exp is not None:
            for t, tb in c["tasks"].items():
                print(f"[resolved-sampler]   client/task {tb['verdict']:11s} n={tb['n']:6d}  {t}  max_tokens={','.join(tb['caps'])}  "
                      f"reference={_fmt(tb['reference']) if tb['reference'] else 'NONE (' + str(tb['why'] or 'undeclared') + ')'}")
    for lg in logs:
        g = lg["generation_config_override"]
        print(f"[resolved-sampler]   generation_config override: {g['present']} ({g['reading']})")
    print(f"[resolved-sampler] -> {a.out}")
    if a.strict:
        return _EXIT_STRICT.get(verdict, 3)
    if a.require_wired:
        # an unmapped verdict is a stale map, loud rather than a silent 0 (unreachable while the
        # vocabulary pin holds: both maps cover contracts.RESOLVED_SAMPLER_VERDICTS minus REFUSED)
        return _EXIT_REQUIRE_WIRED.get(verdict, 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
