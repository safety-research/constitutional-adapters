#!/usr/bin/env python
"""The native-multi-LoRA certification's ONE admission function and its
probe arithmetic — pure, stdlib-only, importable in any venv.

Everything that decides a target's verdict lives here so that the driver
(``multilora_cert.py``), the report (``multilora_cert_report.py``) and the
RESULT line read ONE output and compute nothing themselves:

* :func:`probe_agreement` — per-probe greedy token agreement with the EOS /
  length semantics stated, not implied: ``d = first_divergence_index(a, b)``
  (``None`` iff identical; a pure length difference diverges at ``min(len)``),
  ``denom = max(len(a), len(b))``. Identical outputs of ANY length agree fully
  (two identical early-EOS outputs of 17 tokens are agreement 1.0, never
  17/128 — ``max_tokens`` is a cap, not a normaliser); index 0 is a REAL
  divergence (the earlier draft's ``first_divergence_index(...) or min(len)``
  turned it into full agreement because ``0`` is falsy); one output ending
  while the other continues is a divergence at that index.
* :func:`pooled_agreement` — the per-leg-pair pooled metrics of §1.6.
* :func:`first_token_ulps` — the DESCRIPTIVE first-token statistic: the
  float32 ulp distance between the two legs' logprob of the LENS-STACKED argmax
  token; a native top-5 that does not list that token is ``n_missing`` and
  contributes nothing to the token metrics (the id comparison already
  registers it as ``d == 0`` — never double counted).
* :func:`admission_verdict` — THE FUNCTIONAL RULE (``functional-2026-09-13``,
  schema 5). Production decodes by SAMPLING, so the greedy-token
  disagreement between the native and lens engines is a DIAGNOSTIC, never an
  admission input. A standup is ``admitted`` iff every functional gate holds:
  ``readiness.attested`` (the standup's ``serving_attestation.json`` read by
  the STRICT reader — graphs on, plugin out, adapters bound by asserted name),
  the five positive controls (``noop_certificate`` = the SOLO zero-adapter
  certificate, ``salt_isolation``, ``steering_bites``, ``selection_honored``
  — structural — and ``adapter_digests``), the §10 gate canary against
  the fp32 oracle, and ``head_to_head.not_slower_at_S``; ``not_admitted``
  otherwise, with ``reasons`` naming each failed gate
  (``readiness_unattested`` / ``controls_failed`` / ``gate_canary_{failed,
  missing}`` / ``slower_at_S`` / ``head_to_head_{missing,invalid:…}``). The
  verdict is TWO-WAY — no gate is ever inconclusive. Zero-adapter exactness
  (``zero_adapter_exact``) IS the solo certificate (``zero_adapter.source ==
  "noop_certificate"``), recorded with its exact scope; the concurrent 43 × 128
  zero-vs-none legs' disagreement is RECORDED beside it and refuses nothing.
  Everything the earlier rule computed is still computed — the agreement
  table, the primary per_row-vs-stacked floor, Δ = native − primary with its
  prompt-resampling band (:func:`paired_bootstrap_delta`), the three-way
  ``numerics.outcome`` at the literal m = 0, the planning numbers, the repeat
  pairs, the ulps — into the ``numerics`` block, which carries ``gate: false``
  and the note :data:`NUMERICS_NOTE`; no reader may turn it into a verdict.
* :func:`readmit` — the OFFLINE re-evaluation: verifies the BYTES of every
  numerical input (each leg's JSONL against its DONE.json ``sha256_of_jsonl``,
  every inventoried file against ``INVENTORY.json``; absent server logs are
  reported as ``absent_diagnostic_logs``, never as a pass), rebuilds
  :class:`AdmissionInputs` through the SAME loader the job uses
  (:func:`admission_inputs_from_unit`; the readiness input re-read from the
  attestation on disk through the strict reader), recomputes the verdict
  under the current rule (:func:`reevaluate` does exactly this and writes
  NOTHING) and writes ``unit/admission.readmitted.json`` beside the job's own
  ``admission.json`` — never over it — carrying the original's sha256, the
  code tree that re-evaluated it and the rule; an existing readmit under an
  OLDER rule (:data:`RULE_ORDER`) is superseded (recorded), a newer one is
  never overwritten, an unknown rule refuses.
  ``python multilora_admission.py readmit <record_dir>``.
* :func:`validate_admission_record` — validates a record against ITS OWN
  ``schema_version`` (:data:`SCHEMA_RULES`): the three historical schema-4
  sidecars keep validating beside schema-5 records, so a report over a
  mixed directory renders every record under the rule it was computed under.

History. An earlier rule (2026-09-12) raised the floor to ``max(primary,
native noise)`` and admitted on the LOWER bound ≤ 0; a later analysis
reversed it: a lower-bound rule refuses only demonstrably worse
implementations and cannot establish "no larger than", and ``max(...)``
changed the requirement after seeing the records — against the PRIMARY
floor gemma's excess IS resolved (Δ +8.26 pp, 90% band [+1.26, +16.33])
and nemotron's is inconclusive (+5.37, [−4.37, +15.51]); that analysis
also read the concurrent zero-vs-none legs (8/43 gemma, 6/43 nemotron,
19/43 qwen diverging while the solo 11 × 48 certificate passed) as the
authoritative exactness instrument and refused all three records. Since
2026-09-13 the rule is that production decodes by sampling, so greedy-token
disagreement is a diagnostic, never a gate, and zero-adapter exactness is
the solo certificate: the functional rule above. The three existing records
are NOT rewritten (their schema-4 sidecars stand; the report shows each
record's own rule); under the functional rule each would read ``admitted``,
provable offline with :func:`reevaluate` without writing a byte. The rule is
a pure function of the record, so records are re-evaluated offline
(:func:`readmit`) without a GPU; the in-job admission at the final tip stays
authoritative.

Row shape (one per probe, per leg; ``multilora_cert.py`` writes them):
``{"probe_index": int, "token_ids": [int, ...], "first_token": {"id": int|None,
"logprob": float|None, "top5": [[id|None, logprob], ...], "token": str|None,
"top5_tokens": [str, ...]}}`` plus whatever else the driver records (vmodel,
stamps, finish_reason) — this module reads only those keys.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import secrets
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

#: 5 = the FUNCTIONAL rule (2026-09-13): the two-way verdict over the four
#: functional gates (``gates``, ``readiness``), the greedy-token numerics
#: RECORDED whole under ``numerics`` with ``gate: false``, the zero adapter
#: decided by the solo certificate with the concurrent legs recorded;
#: 4 = D111 (2026-09-12): the THREE-WAY verdict against the primary floor with
#: both one-sided bounds (``numerics.outcome``, ``numerics.prompt_resampling_
#: sensitivity``, ``numerics.primary_point_inequality``, ``floor.margin``,
#: ``floor.diagnostics``), the concurrent zero-exactness reading with the solo
#: certificate recorded (``zero_adapter.{concurrent, solo_certificate}``);
#: 3 = D102 (reversed: the ``max(primary, native noise)`` floor and the
#: lower-bound admission); 2 = the point rule. :func:`validate_admission_record`
#: validates a record against ITS OWN schema (:data:`SCHEMA_RULES`).
SCHEMA_VERSION = 5

#: the nine probe legs of §1.6, in record order
LEGS: Tuple[str, ...] = (
    "lora_native", "lora_native_rep", "lora_zero", "lora_none",
    "lens_stacked", "lens_stacked_rep", "lens_stacked_compA", "lens_stacked_compB",
    "lens_per_row",
)
#: legs the verdict cannot be computed without (a missing one is a named reason)
REQUIRED_LEGS: Tuple[str, ...] = ("lora_native", "lora_zero", "lora_none",
                                  "lens_stacked", "lens_per_row")
#: the six agreement pairs — the five of §1.6 plus ``native_zero_vs_none`` —
#: (pair name) -> (leg A, leg B); the metric is symmetric in A/B except for
#: ``first_token_disagreement_rate``'s reading, so the order is fixed here once.
#: ``native_zero_vs_none`` is the concurrent zero-adapter-vs-base pair on the
#: native engine (43 probes x 128 tokens, submitted together as production
#: does): under D111 it IS the zero-adapter exactness instrument —
#: ``zero_adapter_exact`` is true iff every row of the pair is identical
#: (:data:`ZERO_ADAPTER_SOURCE`) — and it is also recorded under
#: ``floor.diagnostics`` with its own band against the primary. The standup's
#: solo ``noop_certificate`` (one request at a time, 11 x 48) is recorded
#: beside it with its scope; which instrument the no-op requirement names —
#: the solo or the concurrent one — is an open contract question.
PAIRS: Dict[str, Tuple[str, str]] = {
    "native_vs_stacked": ("lora_native", "lens_stacked"),
    "per_row_vs_stacked": ("lens_per_row", "lens_stacked"),
    "stacked_compA_vs_compB": ("lens_stacked_compA", "lens_stacked_compB"),
    "native_rep_vs_native": ("lora_native_rep", "lora_native"),
    "stacked_rep_vs_stacked": ("lens_stacked_rep", "lens_stacked"),
    "native_zero_vs_none": ("lora_zero", "lora_none"),
}
#: the three descriptive ulps pairs — (pair name) -> (leg whose first-token
#: top-5 is searched, the LENS-STACKED-side leg whose argmax is the token)
ULPS_PAIRS: Dict[str, Tuple[str, str]] = {
    "native_vs_stacked": ("lora_native", "lens_stacked"),
    "per_row_vs_stacked": ("lens_per_row", "lens_stacked"),
    "compA_vs_compB": ("lens_stacked_compA", "lens_stacked_compB"),
}
FLOOR_PRIMARY = "per_row_vs_stacked"
FLOOR_FALLBACK = "gate_canary_neartie_oracle"
#: every value ``floor.source`` may take: the primary, or the fallback
FLOOR_SOURCES: Tuple[str, ...] = (FLOOR_PRIMARY, FLOOR_FALLBACK)
#: the pairs RECORDED against the primary with their own bands (``floor.
#: diagnostics``), never a gate (no pre-registered bound on them)
FLOOR_DIAGNOSTIC_PAIRS: Tuple[str, ...] = ("native_rep_vs_native", "native_zero_vs_none",
                                           "stacked_rep_vs_stacked", "stacked_compA_vs_compB")
#: the native engine's own variability vs the lens's own — a stated FINDING
NATIVE_VARIABILITY_PAIRS: Tuple[str, ...] = ("native_rep_vs_native", "native_zero_vs_none")
LENS_VARIABILITY_PAIRS: Tuple[str, ...] = ("stacked_rep_vs_stacked", "stacked_compA_vs_compB")
#: the rule every record names itself by (the FUNCTIONAL rule of 2026-09-13);
#: :data:`RULE_ORDER` is the total order the offline :func:`readmit`
#: supersedes by — an older-rule sidecar is superseded, a newer one never
#: overwritten, an UNKNOWN rule refused by name (the earlier ``D<n>`` integer
#: parse cannot order a rule that is not spelled ``D<n>``)
ADMISSION_RULE = "functional-2026-09-13"
RULE_ORDER: Tuple[str, ...] = ("D102", "D111", ADMISSION_RULE)
#: which rules a record of each schema may name (:func:`validate_admission_record`
#: validates a record against ITS OWN schema, so the three historical schema-4
#: sidecars keep validating beside schema-5 records)
SCHEMA_RULES: Dict[int, Tuple[str, ...]] = {4: ("D111",), 5: (ADMISSION_RULE,)}
#: the functional verdict is TWO-WAY (no gate is inconclusive); the schema-4
#: D111 vocabulary was three-way; ``numerics.outcome`` keeps D111's names for
#: the RECORDED three-way comparison against the primary floor (a diagnostic)
VERDICTS: Tuple[str, ...] = ("admitted", "not_admitted")
VERDICTS_V4: Tuple[str, ...] = ("admitted", "not_admitted", "inconclusive")
#: a certification whose Target says ``numerics="not_measured"`` runs NO probe legs
#: and NO lens per_row leg by DESIGN — no greedy-token agreement measurement of any
#: kind for PSR; its record says so with this literal — distinct from ``unmeasured``,
#: which means "attempted and missing"
NUMERICS_DIAGNOSTIC = "diagnostic"
NUMERICS_NOT_MEASURED = "not_measured"
NUMERICS_MODES: Tuple[str, ...] = (NUMERICS_DIAGNOSTIC, NUMERICS_NOT_MEASURED)
OUTCOMES: Tuple[str, ...] = ("admitted", "not_admitted", "inconclusive", "degenerate_gate_canary", "unmeasured",
                             NUMERICS_NOT_MEASURED)
#: the four functional gates, in the order the verdict names them
FUNCTIONAL_GATES: Tuple[str, ...] = ("readiness_attested", "controls_all_passed", "gate_canary_passed",
                                     "not_slower_at_S")
#: what decides ``zero_adapter_exact`` under the functional rule: the standup's
#: SOLO certificate (``noop_certificate``, one request at a time, 11 x 48),
#: recorded with its exact scope, in force since 2026-09-13. The concurrent
#: lora_zero vs lora_none legs (43 x 128, the instrument the superseded
#: schema-4 rule read as authoritative — :data:`ZERO_ADAPTER_SOURCE_V4`)
#: are RECORDED beside it under ``zero_adapter.concurrent`` and refuse nothing.
ZERO_ADAPTER_SOURCE = "noop_certificate"
ZERO_ADAPTER_SOURCE_V4 = "concurrent_native_zero_vs_none"
ZERO_ADAPTER_CONTROL = "noop_certificate"
#: the ONE sentence the ``numerics`` block carries (and every rendering of it):
#: why the greedy-token statistics are recorded and never read by the verdict
NUMERICS_NOTE = ("diagnostic: production decodes by sampling; greedy-token disagreement is not an admission "
                 "input (user ruling 2026-09-13)")
#: the threshold the RECORDED three-way outcome compares Δ against: the literal
#: m = 0. There is no admission margin under the functional rule (the numerics
#: gate nothing), so no option sets it and no source can amend it; the key is
#: kept under ``floor.margin`` / ``numerics.margin`` so schema-4 and schema-5
#: records render through one reader.
DIAGNOSTIC_MARGIN: Dict[str, Any] = {
    "m": 0.0,
    "source": ("literal m = 0: the threshold of the RECORDED three-way outcome (numerics.gate is false; no "
               "admission margin exists under the functional rule, user ruling 2026-09-13)")}
#: where the readiness input comes from (the driver reads it in-job; the
#: readmit re-reads it from the unit on disk through the same reader)
READINESS_SOURCE = ("lora_native/serving_attestation.json through serving_attestation.read_serving_attestation "
                    "(the strict reader: kind lora_native, schema 1, every key present, name_binding asserted "
                    "per name, engine.enforce_eager False)")
#: the paired bootstrap: resamples and the one-sided level
BOOTSTRAP_N = 2000
BOOTSTRAP_ALPHA = 0.05
#: the ONE sentence every table and figure that shows a band carries
BOOTSTRAP_COVERAGE_NOTE = ("Bands measure sensitivity to resampling this fixed 43-prompt probe set; they do not "
                           "include shared scheduling or between-run uncertainty, and their nominal coverage is "
                           "unvalidated.")
#: the ONE sentence every rendering of the planning numbers carries
PLANNING_NOTE = ("The planning numbers estimate the prompts of this kind needed for 80% power to DETECT a 5-pp "
                 "excess under the conditional approximation — not a promise that so many prompts would "
                 "establish zero-margin noninferiority, and no help with the separate zero-adapter requirement.")
#: what the prompt-resampling band does NOT cover; copied into every record
BOOTSTRAP_CAVEATS: Tuple[str, ...] = (
    "the probe set is fixed and structured (43 prompts: JBB goals + the gate prompts), not a random sample of "
    "production traffic; resampling it estimates sensitivity to WHICH of these prompts were drawn, conditional "
    "on the set",
    "shared scheduler effects are not resampled: every leg ran as one concurrent submission, so batch composition "
    "and prefill grouping are a common cause across prompts that the prompt bootstrap treats as fixed",
    "coverage is unvalidated: no simulation or replicate run has checked that the 90% band covers the true "
    "delta 90% of the time on this engine",
    "one-sided 95% bounds are the 5th and 95th percentiles of the resampled delta (percentile method, linear "
    "interpolation)",
)
#: the offline re-evaluation's output, written BESIDE ``admission.json`` under
#: ``unit/`` and never over it. A post-hoc derived file: it is not unit
#: evidence, so :func:`_walk_unit_files` skips it — never inventoried, never
#: copied on a reuse, never "unlisted" by :func:`unit_inventory_problems`
READMITTED_ADMISSION_FILE = "admission.readmitted.json"
UNIT_POST_HOC_FILES: Tuple[str, ...] = (READMITTED_ADMISSION_FILE,)
N_CONTROLS = 6
#: the sixth control (O49) may be recorded ``not_applicable`` ONLY on a standup that served no
#: restricted-mode cell (its record states ``restricted_cells: []``); ``of`` then counts the
#: APPLICABLE controls and the summary names it under ``not_applicable``
BOUNDARY_CONTROL = "boundary_refusal"
NOT_APPLICABLE = "not_applicable"
#: the five positive controls by the PRODUCER's names —
#: E's ``lora_serving_up.CONTROL_NAMES``, in run order, pinned equal by
#: tests/test_multilora_cert_records.py. :func:`controls_summary` is BOUND to
#: them (code-review item 9, 2026-09-11): the earlier reader counted any five
#: boolean verdicts, so a ``positive_controls.json`` carrying the wrong five
#: names read as 5/5.
CONTROL_NAMES: Tuple[str, ...] = ("noop_certificate", "salt_isolation", "steering_bites",
                                  "selection_honored", "adapter_digests", "boundary_refusal")
GATE_RECORD = "gate_record.json"
GATE_EVIDENCE = "gate_evidence.json"
GATE_FAILED = "gate_failed.json"
GATE_UNAVAILABLE = "gate_unavailable.json"
#: the gate's transactional certificate store (petri_serving_gate.CertificateStore)
GATE_CERTIFICATES = "noop_certificates.json"
#: the ATTEMPT binding: the driver writes ``ATTEMPT.json`` (a nonce + the
#: unit's identity) BEFORE the gate child runs, after moving every earlier
#: attempt's outputs aside, and ``canary_run.json`` (the same nonce, the
#: child's rc, the sha256 of the record it left) AFTER;
#: :func:`gate_canary_summary` accepts a ``gate_record.json`` only when both
#: are present, agree, the rc is 0 and the record is the one the current run
#: left. Before this a rerun into the same out dir kept an earlier attempt's
#: successful record beside the failure/unavailable marker of the current one,
#: and the reader returned the record first — a failed or timed-out canary
#: supplied the required fallback evidence from an earlier attempt.
GATE_ATTEMPT = "ATTEMPT.json"
GATE_RUN = "canary_run.json"
GATE_ATTEMPT_OUTPUTS: Tuple[str, ...] = (GATE_RECORD, GATE_EVIDENCE, GATE_FAILED, GATE_UNAVAILABLE,
                                         GATE_CERTIFICATES, GATE_RUN, GATE_ATTEMPT)
#: the record's ``recorded_at`` may precede the attempt's start by this much
#: (one host, one clock, sub-second scheduling skew): the sha256 binding and
#: the move-aside are what tie a record to its attempt; this stamp check only
#: catches a record older than the attempt by more than any skew could be
GATE_RECORD_SKEW_S = 5.0
#: ONE certification attempt's records under ``unit/``: ``identity.json``,
#: ``DONE.json`` and ``admission.json`` each carry the driver attempt's
#: ``attempt_id`` (the nonce ``multilora_cert`` mints at the start of every
#: run, before any fallible phase, and stamps into ``manifest.json`` too),
#: and ``DONE.json`` carries the sha256 of the ``admission.json`` bytes the
#: same attempt wrote. :func:`unit_binding_problems` is the ONE rule both the
#: driver (a whole-unit reuse, its own RESULT) and the report (what a record
#: dir presents) read a unit through. Before this a rerun into a successful
#: out dir that failed in preflight or the oracle left the earlier unit in
#: place, and the FAILED attempt's RESULT and the report both presented that
#: unit's ``admitted`` as the rerun's (reproduced).
ATTEMPT_ID_KEY = "attempt_id"
DONE_ADMISSION_SHA_KEY = "admission_sha256"
UNIT_ATTEMPT_RECORDS: Tuple[str, ...] = ("identity.json", "DONE.json", "admission.json")
#: a whole-unit REUSE is SELF-CONTAINED: the accepting attempt
#: copies the source unit's records VERBATIM into its own ``reused_unit/``
#: and records, in ``unit_reuse.json``, the accepted source ``attempt_id``
#: and the sha256 of every copied file. Every reader (the report, the RESULT)
#: resolves a reused unit at THE COPY, bound to those digests through
#: :func:`reused_unit_binding_problems` — never by following
#: ``unit_reused_from`` to a directory a later rerun of the SOURCE replaces
#: (``begin_attempt`` archives its unit and the new attempt writes another at
#: the same path, possibly with different code, probe settings and verdict).
#: Reproduced before this: B reused A's ``not_admitted`` unit; A was rerun in
#: place and came out ``admitted``; the report on B followed the path and read
#: ``admitted`` with no binding problem while B's own RESULT said
#: ``not_admitted``.
REUSED_UNIT_DIR = "reused_unit"
UNIT_REUSE_FILE = "unit_reuse.json"
#: THE EVIDENCE INVENTORY of a complete unit: ``unit/INVENTORY.json`` is
#: written at unit completion, just before ``DONE.json`` (which carries its
#: sha256), and lists EVERY file under ``unit/`` with its sha256 and size plus
#: the REQUIRED evidence list (:func:`unit_required_evidence`). A whole-unit
#: reuse snapshots the entire inventory (:func:`unit_inventory_problems` on
#: the bytes), copies it whole into ``reused_unit/`` and binds by digest; an
#: incomplete source refuses by naming the missing entries. Before this the
#: reuse required and copied four summaries only, so a summaries-only source
#: (no gate record, no attestation, no probe rows, no windows) reused with
#: zero standups as ``admitted`` and the report saw no problem.
UNIT_INVENTORY_FILE = "INVENTORY.json"
DONE_INVENTORY_SHA_KEY = "inventory_sha256"
#: the files DONE.json / INVENTORY.json themselves are the binding, not entries
UNIT_INVENTORY_EXCLUDED: Tuple[str, ...] = ("DONE.json", UNIT_INVENTORY_FILE)
THROUGHPUT_LEGS: Tuple[str, ...] = ("lora_native", "lens_stacked", "lens_per_row")
#: the throughput legs a ``not_measured`` unit runs: the head-to-head's two sides
#: (``not_slower_at_S`` compares same-job numbers); the per_row leg is a numerics leg
THROUGHPUT_LEGS_CORE: Tuple[str, ...] = ("lora_native", "lens_stacked")
#: evidence every complete unit holds (paths relative to ``unit/``) — the CORE:
#: the functional gates' inputs (split out from the numerics evidence)
UNIT_EVIDENCE_CORE: Tuple[str, ...] = (
    "identity.json", "admission.json", "head_to_head.json", "run_identity.json",
    "controls_summary.json", "lora_native_endpoint.json", "oracle_reference.json",
    "native_standup_attempts.json",
    "lora_native/serving_attestation.json", "lora_native/cells_served.json",
    "lora_native/positive_controls.json", "lora_native/routing_table.json", "lora_native/CANARY_OK",
    "gate_canary/ATTEMPT.json", "gate_canary/canary_run.json", "gate_canary/canary_summary.json",
    "lens_stacked/steering_attestation.json", "lens_stacked/reduction_mode_check.json",
)
#: the NUMERICS evidence: the lens per_row leg's standup records (its probe and throughput
#: legs are listed per leg by :func:`unit_required_evidence`); absent by design from a
#: ``not_measured`` unit
UNIT_EVIDENCE_NUMERICS: Tuple[str, ...] = (
    "lens_per_row/steering_attestation.json", "lens_per_row/reduction_mode_check.json",
)
#: the historical name: a DIAGNOSTIC unit's fixed evidence (core + numerics), byte for byte
UNIT_EVIDENCE_FIXED: Tuple[str, ...] = UNIT_EVIDENCE_CORE + UNIT_EVIDENCE_NUMERICS
#: the gate canary's record + evidence: required whenever the admission says the canary passed
UNIT_EVIDENCE_GATE_PASSED: Tuple[str, ...] = ("gate_canary/gate_record.json", "gate_canary/gate_evidence.json")
#: a throughput window's VALIDITY: a window with any request error, fewer
#: than this many successful completions, or an achieved concurrency (the
#: engine's ``num_requests_running`` mean) below this fraction of B is
#: INVALID and cannot establish ``not_slower_at_S`` — engine tokens counted
#: before a proxy post-check rejection are still tokens on the counter, so an
#: errored lens window can manufacture a passing ratio.
MIN_REQUESTS_DONE = 1
MIN_RUNNING_MEAN_FRAC = 0.5
#: the ONE admissible evidence for a throughput window: the ENGINE's own
#: counters — ``server_tokens`` off ``vllm:generation_tokens_total`` and
#: ``running_mean`` off ``vllm:num_requests_running`` — read through
#: ``/metrics``. A window that lacks either (the driver's client-completion
#: fallback when ``/metrics`` was unreadable: ``throughput_source ==
#: "client_fallback"``) is INVALID for certification whatever its client
#: count says; the fallback stays in the record as a diagnostic
#: (``client_tok_s``). Before this the concurrency guard ran only when
#: ``running_mean`` was present, so a leg whose first ``/metrics`` request
#: failed had every window pass with no evidence that the engine ever held B
#: requests (reproduced: ``invalid reasons: []``, ``not_slower_at_S: True``
#: off client counts alone).
THROUGHPUT_SOURCE_SERVER = "server"
THROUGHPUT_SOURCE_CLIENT_FALLBACK = "client_fallback"


class AdmissionError(ValueError):
    """Malformed inputs — a row without ``token_ids``, a leg keyed twice on
    one probe. Never a verdict: the verdict records reasons, this refuses."""


# --------------------------------------------------------------------------- #
# per-probe arithmetic                                                          #
# --------------------------------------------------------------------------- #

def first_divergence_index(a: Sequence[Any], b: Sequence[Any]) -> Optional[int]:
    """Index of the first differing element, or ``None`` when identical; a
    pure length difference diverges at ``min(len(a), len(b))``. Byte-for-byte
    the semantics of ``sisteer.capeval.serving.fidelity.first_divergence_index``
    (restated here so this module needs nothing from the package; the test
    pins the two equal on every case)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


class ProbeAgreement(NamedTuple):
    """One probe's agreement between two legs."""

    d: Optional[int]        # first divergence index; None iff identical
    exact: bool
    agree_prefix: int       # tokens agreeing before the divergence (len(a) when exact)
    denom: int              # max(len(a), len(b))
    frac: float             # agree_prefix / denom; 1.0 for identical outputs of any length
    empty: bool             # both outputs empty (identical, counted separately)


def probe_agreement(a: Sequence[int], b: Sequence[int]) -> ProbeAgreement:
    """§2.F.7's per-probe rule with EXPLICIT ``None`` handling: ``d is None``
    (identical, any length, both-empty included) -> exact, ``agree_prefix =
    len(a)``, ``frac = 1.0``; otherwise ``exact = False``, ``agree_prefix = d``
    (``d == 0`` is a real first-token divergence), ``frac = d / denom`` with
    ``denom = max(len(a), len(b))``."""
    a = list(a)
    b = list(b)
    d = first_divergence_index(a, b)
    denom = max(len(a), len(b))
    if d is None:
        return ProbeAgreement(d=None, exact=True, agree_prefix=len(a), denom=denom,
                              frac=1.0, empty=(denom == 0))
    return ProbeAgreement(d=d, exact=False, agree_prefix=d, denom=denom,
                          frac=(d / denom) if denom else 1.0, empty=False)


def pooled_token_agreement(per: Sequence[ProbeAgreement]) -> float:
    """THE pooled statistic: ``sum(agree_prefix) / sum(denom)`` over a set of
    per-probe agreements (every pair empty — ``sum_denom == 0`` — is
    "identical outputs", 1.0). One definition, read by the agreement table
    (:func:`pool_probe_agreements`) and by every bootstrap resample
    (:func:`paired_bootstrap_delta`), so the resampled statistic is the
    table's statistic on a resampled prompt set — a sum over prompts, never an
    average of per-prompt rates."""
    sum_prefix = sum(p.agree_prefix for p in per)
    sum_denom = sum(p.denom for p in per)
    return (sum_prefix / sum_denom) if sum_denom else 1.0


def pool_probe_agreements(per: Sequence[ProbeAgreement]) -> Dict[str, Any]:
    """The §1.6 pooled metrics over per-probe agreements: ``token_agreement``
    (:func:`pooled_token_agreement`), ``disagreement = 1 - token_agreement``,
    ``exact_match_rate = mean(exact)``, ``n_diverging = sum(not exact)``,
    ``first_token_disagreement_rate = mean(d == 0)``, ``n_empty`` (identical
    empty pairs, counted and listed in the verdict's reasons), ``n``. An empty
    input yields ``n == 0`` and ``None`` metrics (the caller names the missing
    leg; nothing here guesses a number)."""
    per = list(per)
    n = len(per)
    if n == 0:
        return {"n": 0, "token_agreement": None, "disagreement": None,
                "exact_match_rate": None, "n_diverging": 0,
                "first_token_disagreement_rate": None, "n_empty": 0,
                "sum_agree_prefix": 0, "sum_denom": 0}
    sum_prefix = sum(p.agree_prefix for p in per)
    sum_denom = sum(p.denom for p in per)
    token_agreement = pooled_token_agreement(per)
    return {
        "n": n,
        "token_agreement": token_agreement,
        "disagreement": 1.0 - token_agreement,
        "exact_match_rate": sum(1 for p in per if p.exact) / n,
        "n_diverging": sum(1 for p in per if not p.exact),
        "first_token_disagreement_rate": sum(1 for p in per if p.d == 0) / n,
        "n_empty": sum(1 for p in per if p.empty),
        "sum_agree_prefix": sum_prefix,
        "sum_denom": sum_denom,
    }


def pooled_agreement(pairs: Iterable[Tuple[Sequence[int], Sequence[int]]]) -> Dict[str, Any]:
    """:func:`pool_probe_agreements` over ``[(a, b), ...]`` token-id pairs."""
    return pool_probe_agreements([probe_agreement(a, b) for a, b in pairs])


def float32_ulps(x: float, y: float) -> int:
    """The monotone int32 bit-pattern distance between two values rounded to
    float32 (the "ulps" of §1.6): 0 iff the two float32s are the same number
    (``-0.0`` and ``0.0`` included), 1 for adjacent representable values,
    sign-crossing distances counted through zero. Refuses a non-finite input
    (the caller files it as missing)."""
    if not (math.isfinite(x) and math.isfinite(y)):
        raise AdmissionError(f"ulps of a non-finite value: {x!r}, {y!r}")

    def mono(v: float) -> int:
        u = struct.unpack("<I", struct.pack("<f", v))[0]
        return u if u < 0x8000_0000 else 0x8000_0000 - u

    return abs(mono(x) - mono(y))


# --------------------------------------------------------------------------- #
# rows                                                                          #
# --------------------------------------------------------------------------- #

def rows_by_probe(rows: Iterable[Mapping[str, Any]], *, leg: str = "?"
                  ) -> Dict[int, Mapping[str, Any]]:
    """``{probe_index: row}``; refuses a duplicate index or a row without an
    integer ``probe_index`` / a list ``token_ids`` (malformed evidence is a
    refusal, never a silent zero)."""
    out: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        idx = row.get("probe_index") if isinstance(row, Mapping) else None
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise AdmissionError(f"leg {leg}: row without an integer probe_index: {row!r}"[:300])
        ids = row.get("token_ids")
        if not isinstance(ids, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in ids):
            raise AdmissionError(f"leg {leg}: probe {idx} has no integer token_ids list")
        if idx in out:
            raise AdmissionError(f"leg {leg}: probe_index {idx} appears twice")
        out[idx] = row
    return out


def _pairs_by_probe(legs: Mapping[str, Dict[int, Mapping[str, Any]]], a: str, b: str
                    ) -> Tuple[Dict[int, Tuple[List[int], List[int]]], Dict[str, Any]]:
    """``{probe_index: (ids_a, ids_b)}`` over the probes BOTH legs carry (in
    probe order), plus a note on any asymmetry (a probe one leg has and the
    other lacks is recorded, never silently dropped into the denominator)."""
    ra, rb = legs.get(a), legs.get(b)
    if ra is None or rb is None:
        return {}, {"missing_legs": [x for x, r in ((a, ra), (b, rb)) if r is None]}
    common = sorted(set(ra) & set(rb))
    note: Dict[str, Any] = {}
    only_a, only_b = sorted(set(ra) - set(rb)), sorted(set(rb) - set(ra))
    if only_a or only_b:
        note["probe_set_mismatch"] = {a: only_a, b: only_b}
    return {i: (list(ra[i]["token_ids"]), list(rb[i]["token_ids"])) for i in common}, note


def _pairs_for(legs: Mapping[str, Dict[int, Mapping[str, Any]]], a: str, b: str
               ) -> Tuple[List[Tuple[List[int], List[int]]], Dict[str, Any]]:
    """:func:`_pairs_by_probe` as the list of pairs (probe order)."""
    by_probe, note = _pairs_by_probe(legs, a, b)
    return list(by_probe.values()), note


def per_probe_agreements(legs: Mapping[str, Dict[int, Mapping[str, Any]]], a: str, b: str
                         ) -> Tuple[Dict[int, ProbeAgreement], Dict[str, Any]]:
    """One pair's per-probe agreements keyed by probe index — what the
    agreement table pools (:func:`pool_probe_agreements`) AND what the paired
    bootstrap resamples (:func:`paired_bootstrap_delta`), so the two read one
    set of objects — cross-checked through the loader, not beside it."""
    by_probe, note = _pairs_by_probe(legs, a, b)
    return {i: probe_agreement(x, y) for i, (x, y) in by_probe.items()}, note


def _first_token(row: Mapping[str, Any]) -> Mapping[str, Any]:
    ft = row.get("first_token")
    return ft if isinstance(ft, Mapping) else {}


def first_token_ulps(search_rows: Mapping[int, Mapping[str, Any]],
                     argmax_rows: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    """The descriptive statistic for one pair: for every probe both legs
    carry, take the ``argmax_rows`` leg's first token (its id, and its
    logprob) and find that token in the ``search_rows`` leg's first-token
    top-5 — by id when the top-5 carries ids, else by token string — then the
    float32 ulp distance between the two logprobs. Not found (or a missing /
    non-finite logprob) -> ``n_missing += 1``. Returns ``{max, p50, n_present,
    n_missing, match_basis: {id, token}}``; ``max`` / ``p50`` are ``None``
    when nothing is present."""
    dists: List[int] = []
    n_missing = 0
    basis = {"id": 0, "token": 0}
    for idx in sorted(set(search_rows) & set(argmax_rows)):
        want = _first_token(argmax_rows[idx])
        have = _first_token(search_rows[idx])
        want_id, want_lp, want_tok = want.get("id"), want.get("logprob"), want.get("token")
        top5 = have.get("top5") if isinstance(have.get("top5"), list) else []
        top5_tokens = have.get("top5_tokens") if isinstance(have.get("top5_tokens"), list) else []
        found_lp: Optional[float] = None
        if want_id is not None:
            for ent in top5:
                if isinstance(ent, (list, tuple)) and len(ent) >= 2 and ent[0] == want_id:
                    found_lp = ent[1]
                    basis["id"] += 1
                    break
        if found_lp is None and want_tok is not None and top5_tokens:
            for ent, tok in zip(top5, top5_tokens):
                if tok == want_tok and isinstance(ent, (list, tuple)) and len(ent) >= 2:
                    found_lp = ent[1]
                    basis["token"] += 1
                    break
        if found_lp is None or want_lp is None:
            n_missing += 1
            continue
        try:
            dists.append(float32_ulps(float(want_lp), float(found_lp)))
        except (AdmissionError, TypeError, ValueError):
            n_missing += 1
    return {"max": (max(dists) if dists else None),
            "p50": (statistics.median(dists) if dists else None),
            "n_present": len(dists), "n_missing": n_missing,
            "match_basis": basis}


# --------------------------------------------------------------------------- #
# the evidence summaries the verdict consumes (file readers; pure over dicts)   #
# --------------------------------------------------------------------------- #

def _write_json_atomic(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2, sort_keys=True, default=str))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def move_aside(dir_: Path, names: Sequence[str], *, tag: str) -> Optional[Path]:
    """Move whichever of ``names`` exist under ``dir_`` into a fresh
    ``dir_/stale-<tag>/`` (a rename, never a deletion — an earlier attempt's
    record stays readable for a human, out of every reader's path). Returns
    the stale dir, or ``None`` when nothing was there."""
    dir_ = Path(dir_)
    present = [n for n in names if (dir_ / n).exists()]
    if not present:
        return None
    dest = dir_ / f"stale-{tag}"
    i = 0
    while dest.exists():
        i += 1
        dest = dir_ / f"stale-{tag}-{i}"
    dest.mkdir(parents=True)
    for n in present:
        os.replace(dir_ / n, dest / n)
    return dest


def begin_gate_attempt(gate_dir: Path, *, unit: Mapping[str, Any]) -> Dict[str, Any]:
    """Open ONE gate-canary attempt under ``gate_dir``: every earlier attempt's
    outputs (:data:`GATE_ATTEMPT_OUTPUTS`) are moved aside first, then
    ``ATTEMPT.json`` is written with a fresh nonce, the start time and the
    unit's identity — BEFORE the gate child runs, so the child can never be
    read against a record it did not write. Returns the attempt record."""
    gate_dir = Path(gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(16)
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + nonce[:8]
    stale = move_aside(gate_dir, GATE_ATTEMPT_OUTPUTS, tag=tag)
    attempt = {"nonce": nonce, "started_at": time.time(), "unit": dict(unit),
               "stale_moved_to": (str(stale) if stale is not None else None)}
    _write_json_atomic(gate_dir / GATE_ATTEMPT, attempt)
    return attempt


def finish_gate_attempt(gate_dir: Path, attempt: Mapping[str, Any], *, rc: int) -> Dict[str, Any]:
    """Close the attempt: ``canary_run.json`` carries the attempt's nonce, the
    child's exit code, the finish time and the sha256 of whatever
    ``gate_record.json`` the child left (``None`` when none) — what
    :func:`gate_canary_summary` binds the record to."""
    gate_dir = Path(gate_dir)
    rec = gate_dir / GATE_RECORD
    now = time.time()
    run = {"nonce": attempt["nonce"], "started_at": attempt["started_at"], "finished_at": now,
           "elapsed_s": now - float(attempt["started_at"]), "rc": int(rc),
           "record_sha256": (_sha256_file(rec) if rec.is_file() else None),
           "unit": dict(attempt.get("unit") or {})}
    _write_json_atomic(gate_dir / GATE_RUN, run)
    return run


def _read_attempt_binding(gate_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """``(attempt, run, problem)``: both binding files read, or the problem."""
    docs: Dict[str, Any] = {}
    for name in (GATE_ATTEMPT, GATE_RUN):
        p = gate_dir / name
        if not p.is_file():
            return None, None, f"no {name} under {gate_dir} — the gate canary is not bound to a current attempt"
        try:
            doc = json.loads(p.read_text())
        except (ValueError, OSError) as e:
            return None, None, f"{name} unreadable: {e}"
        if not isinstance(doc, Mapping) or not isinstance(doc.get("nonce"), str) or not doc["nonce"]:
            return None, None, f"{name} carries no nonce"
        docs[name] = dict(doc)
    attempt, run = docs[GATE_ATTEMPT], docs[GATE_RUN]
    if run["nonce"] != attempt["nonce"]:
        return attempt, run, (f"{GATE_RUN} nonce {run['nonce'][:8]} != {GATE_ATTEMPT} nonce "
                              f"{attempt['nonce'][:8]} — a stale run record beside a newer attempt")
    if not isinstance(run.get("rc"), int) or isinstance(run.get("rc"), bool):
        return attempt, run, f"{GATE_RUN} records no exit code"
    return attempt, run, None


def gate_canary_summary(gate_dir: Optional[Path], *, cell: str) -> Dict[str, Any]:
    """The gate canary's verdict as the admission reads it
    (``gate_canary``): ``gate_record.json`` with ``status == "completed"``, the
    cell's ``lora`` path among ``kinds[cell].passed_paths``, both no-op
    certificates (``path in {passthrough, lora}``) recorded and
    ``multi_lora_checked`` -> ``passed``. A ``gate_failed.json`` /
    ``gate_unavailable.json`` (or no record at all) is ``passed = False`` with
    its ``status`` named; the comparator's ``margin_atol_nats`` is copied from
    ``gate_evidence.json`` when present. Never raises on a missing dir: absence
    is a recorded outcome ("missing"), which the verdict names.

    BOUND TO THE CURRENT ATTEMPT: the record counts only when
    ``ATTEMPT.json`` and ``canary_run.json`` are present with one nonce, the
    current child exited 0, and the record's bytes are the ones that run left
    (``record_sha256``) and were written after the attempt began. A record
    without that binding, beside a non-zero exit, or older than the attempt is
    ``passed = False`` with the reason named — an earlier attempt's success
    never stands in for the current one."""
    out: Dict[str, Any] = {"ran": False, "status": "missing", "passed": False,
                           "lora_canary_passed": False, "noop_lora_passed": False,
                           "noop_passthrough_passed": False, "multi_lora_checked": False,
                           "margin_atol_nats": None, "record": None, "cell": cell,
                           "reason": None, "bound": False, "attempt": None}
    if gate_dir is None:
        out["reason"] = "no gate canary directory"
        return out
    gate_dir = Path(gate_dir)
    attempt, run, problem = _read_attempt_binding(gate_dir)
    if problem is not None:
        rec_path = gate_dir / GATE_RECORD
        if rec_path.is_file():
            out.update(status="unbound", ran=True, record=str(rec_path))
        out["reason"] = problem
        return out
    assert attempt is not None and run is not None
    out["bound"] = True
    out["attempt"] = {"nonce": attempt["nonce"], "started_at": attempt.get("started_at"),
                      "finished_at": run.get("finished_at"), "rc": run["rc"],
                      "unit": dict(attempt.get("unit") or {})}
    ev_path = gate_dir / GATE_EVIDENCE
    if ev_path.is_file():
        try:
            ev = json.loads(ev_path.read_text())
            out["margin_atol_nats"] = (ev.get("cross_engine_comparator") or {}).get("margin_atol_nats")
        except (ValueError, OSError, AttributeError):
            pass
    rec_path = gate_dir / GATE_RECORD
    if rec_path.is_file():
        try:
            rec = json.loads(rec_path.read_text())
        except (ValueError, OSError) as e:
            out.update(status="unreadable", ran=True, record=str(rec_path),
                       reason=f"{GATE_RECORD} unreadable: {e}")
            return out
        out.update(ran=True, record=str(rec_path), status=str(rec.get("status")))
        if run["rc"] != 0:
            out["reason"] = (f"the current attempt exited rc={run['rc']} — a {GATE_RECORD} beside a "
                             "non-zero exit is not this attempt's completion")
            return out
        got_sha = _sha256_file(rec_path)
        if run.get("record_sha256") != got_sha:
            out["reason"] = (f"{GATE_RECORD} sha256 {got_sha[:12]} != the current run's recorded "
                             f"{str(run.get('record_sha256'))[:12]} — not the record this attempt left")
            return out
        recorded_at = rec.get("recorded_at")
        started_at = attempt.get("started_at")
        if (isinstance(recorded_at, (int, float)) and isinstance(started_at, (int, float))
                and recorded_at < started_at - GATE_RECORD_SKEW_S):
            out["reason"] = (f"{GATE_RECORD} recorded_at {recorded_at} predates the attempt's start "
                             f"{started_at} by more than {GATE_RECORD_SKEW_S}s — an earlier attempt's record")
            return out
        kinds = rec.get("kinds") or {}
        k = kinds.get(cell) if isinstance(kinds, Mapping) else None
        out["lora_canary_passed"] = bool(isinstance(k, Mapping)
                                         and "lora" in (k.get("passed_paths") or []))
        paths = {c.get("path") for c in (rec.get("certificates") or []) if isinstance(c, Mapping)}
        out["noop_lora_passed"] = "lora" in paths
        out["noop_passthrough_passed"] = "passthrough" in paths
        out["multi_lora_checked"] = bool(rec.get("multi_lora_checked"))
        out["passed"] = (rec.get("status") == "completed" and out["lora_canary_passed"]
                         and out["noop_lora_passed"] and out["noop_passthrough_passed"]
                         and out["multi_lora_checked"])
        if not out["passed"]:
            out["reason"] = ("gate record completed but " + ", ".join(
                n for n, v in (("lora canary not passed", not out["lora_canary_passed"]),
                               ("no lora no-op certificate", not out["noop_lora_passed"]),
                               ("no passthrough no-op certificate", not out["noop_passthrough_passed"]),
                               ("multi_lora not checked", not out["multi_lora_checked"]),
                               (f"status {rec.get('status')!r}", rec.get("status") != "completed"))
                if v))
        return out
    for marker, status in ((GATE_FAILED, "failed"), (GATE_UNAVAILABLE, "unavailable")):
        p = gate_dir / marker
        if p.is_file():
            try:
                reason = json.loads(p.read_text()).get("reason")
            except (ValueError, OSError):
                reason = f"{marker} unreadable"
            out.update(ran=True, status=status, record=str(p), reason=reason)
            return out
    if run["rc"] != 0:
        # a child that died or timed out (rc 124) before writing any marker
        out.update(ran=True, status="failed",
                   reason=f"the current attempt exited rc={run['rc']} and left no "
                          f"{GATE_RECORD} / {GATE_FAILED} / {GATE_UNAVAILABLE}")
        return out
    out["reason"] = f"no {GATE_RECORD} / {GATE_FAILED} / {GATE_UNAVAILABLE} under {gate_dir}"
    return out


def unit_required_evidence(admission: Any) -> List[str]:
    """The REQUIRED evidence of a complete unit (A2), relative to ``unit/``:
    :data:`UNIT_EVIDENCE_FIXED`; the gate canary's record + evidence when the
    admission says the canary passed; every probe leg's JSONL + DONE; every
    throughput leg's record + DONE and at least one gauges file and one util
    CSV under ``throughput/raw/`` (entries with ``*`` are globs that must
    match at least one inventoried file)."""
    not_measured = numerics_mode_of(admission) == NUMERICS_NOT_MEASURED
    req: List[str] = list(UNIT_EVIDENCE_CORE if not_measured else UNIT_EVIDENCE_FIXED)
    gate = (admission or {}).get("gate_canary") if isinstance(admission, Mapping) else None
    if isinstance(gate, Mapping) and gate.get("passed"):
        req += list(UNIT_EVIDENCE_GATE_PASSED)
    if not not_measured:
        for leg in LEGS:
            req += [f"probes/{leg}.jsonl", f"probes/{leg}.DONE.json"]
    for leg in (THROUGHPUT_LEGS_CORE if not_measured else THROUGHPUT_LEGS):
        req += [f"throughput/{leg}.json", f"throughput/{leg}.DONE.json",
                f"throughput/raw/gauges_{leg}_*.json", f"throughput/raw/util_{leg}_*.csv"]
    return req


def numerics_mode_of(admission: Any) -> str:
    """The record's numerics MODE (``numerics.mode``): :data:`NUMERICS_NOT_MEASURED`
    when the record says so, else :data:`NUMERICS_DIAGNOSTIC` (every record written
    before native-psr-serving carries no ``mode`` and measured its numerics)."""
    if isinstance(admission, Mapping):
        nm = admission.get("numerics")
        if isinstance(nm, Mapping) and nm.get("mode") == NUMERICS_NOT_MEASURED:
            return NUMERICS_NOT_MEASURED
    return NUMERICS_DIAGNOSTIC


def numerics_evidence_present(files: Iterable[str]) -> List[str]:
    """The NUMERICS evidence among ``files`` (relative unit paths): probe legs and
    the lens per_row leg's records — what a ``not_measured`` unit must NOT carry
    (the known-bad case: a record that says not_measured beside measured legs is
    two claims about one unit)."""
    out: List[str] = []
    for rel in files:
        r = str(rel)
        if r.startswith(("probes/", "lens_per_row/")) or "lens_per_row" in r.rsplit("/", 1)[-1]:
            out.append(r)
    return sorted(out)


def evidence_missing(required: Sequence[str], files: Iterable[str]) -> List[str]:
    """The entries of ``required`` no file in ``files`` satisfies (a ``*``
    entry is satisfied by any fnmatch)."""
    names = list(files)
    missing: List[str] = []
    for entry in required:
        if "*" in entry:
            if not any(fnmatch.fnmatch(n, entry) for n in names):
                missing.append(entry)
        elif entry not in names:
            missing.append(entry)
    return missing


def _walk_unit_files(unit_dir: Path) -> List[str]:
    """Every regular file under ``unit_dir`` as a POSIX relative path, the
    binding files (:data:`UNIT_INVENTORY_EXCLUDED`), the post-hoc derived
    files (:data:`UNIT_POST_HOC_FILES`: an offline ``readmit`` output is not
    unit evidence and is never inventoried, copied or "unlisted") and
    dot-files (the atomic writers' temp names) left out, sorted."""
    unit_dir = Path(unit_dir)
    out: List[str] = []
    for root, _dirs, files in os.walk(unit_dir):
        for name in files:
            if name.startswith("."):
                continue
            rel = Path(root, name).relative_to(unit_dir).as_posix()
            if rel in UNIT_INVENTORY_EXCLUDED or rel in UNIT_POST_HOC_FILES:
                continue
            out.append(rel)
    return sorted(out)


def build_unit_inventory(unit_dir: Path, *, attempt_id: Optional[str]) -> Dict[str, Any]:
    """The ``INVENTORY.json`` document for a unit about to complete (A2):
    every file under ``unit/`` (:func:`_walk_unit_files`) with sha256 + size,
    the required-evidence list for this unit's admission, and whatever of it
    is missing (the caller refuses to complete on a non-empty ``missing``)."""
    unit_dir = Path(unit_dir)
    files: Dict[str, Dict[str, Any]] = {}
    for rel in _walk_unit_files(unit_dir):
        p = unit_dir / rel
        files[rel] = {"sha256": _sha256_file(p), "size": p.stat().st_size}
    admission = None
    if (unit_dir / "admission.json").is_file():
        try:
            admission = json.loads((unit_dir / "admission.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            admission = None
    required = unit_required_evidence(admission)
    return {"schema": "multilora_cert.inventory/1", ATTEMPT_ID_KEY: attempt_id,
            "written_at": time.time(), "n_files": len(files), "files": files,
            "required": required, "missing": evidence_missing(required, files)}


#: a diagnostic server log an inventory lists: absent from a PULLED record by design
def is_diagnostic_log(rel: str) -> bool:
    return str(rel).endswith(".log")


def unit_inventory_problems_from_bytes(raw: Mapping[str, bytes], *, present: Iterable[str],
                                       absent_logs: Iterable[str] = ()) -> List[str]:
    """:func:`unit_inventory_problems` over bytes already in hand: ``raw``
    maps ``INVENTORY.json``, ``DONE.json`` and every inventoried relative
    path to the file's bytes; ``present`` is every regular file actually
    under the unit (for the unlisted check); ``absent_logs`` names inventoried
    diagnostic logs the caller allowed to be absent (a pulled record) — they
    are neither read nor a problem. Empty = the inventory binds."""
    problems: List[str] = []
    allowed_absent = set(absent_logs)
    for name in (UNIT_INVENTORY_FILE, "DONE.json"):
        if name not in raw:
            problems.append(f"no {name}")
    if problems:
        return problems
    try:
        inv = json.loads(raw[UNIT_INVENTORY_FILE].decode("utf-8"))
        done = json.loads(raw["DONE.json"].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return [f"{UNIT_INVENTORY_FILE} / DONE.json unreadable: {e}"]
    if not isinstance(inv, Mapping) or not isinstance(inv.get("files"), Mapping):
        return [f"{UNIT_INVENTORY_FILE} is not an inventory document"]
    if not isinstance(done, Mapping):
        return ["DONE.json is not a JSON object"]
    got = hashlib.sha256(raw[UNIT_INVENTORY_FILE]).hexdigest()
    if done.get(DONE_INVENTORY_SHA_KEY) != got:
        problems.append(f"DONE.json {DONE_INVENTORY_SHA_KEY} {str(done.get(DONE_INVENTORY_SHA_KEY))[:12]} != "
                        f"the {UNIT_INVENTORY_FILE} on hand {got[:12]}")
    if inv.get(ATTEMPT_ID_KEY) != done.get(ATTEMPT_ID_KEY) or not inv.get(ATTEMPT_ID_KEY):
        problems.append(f"{UNIT_INVENTORY_FILE} attempt {str(inv.get(ATTEMPT_ID_KEY))[:8]} != DONE.json's "
                        f"{str(done.get(ATTEMPT_ID_KEY))[:8]}")
    files: Mapping[str, Any] = inv["files"]
    for rel, meta in sorted(files.items()):
        if rel not in raw:
            if rel in allowed_absent:
                continue
            problems.append(f"inventoried file missing: {rel}")
            continue
        want = meta.get("sha256") if isinstance(meta, Mapping) else None
        have = hashlib.sha256(raw[rel]).hexdigest()
        if want != have:
            problems.append(f"{rel}: sha256 {have[:12]} != inventoried {str(want)[:12]}")
    unlisted = sorted(set(present) - set(files) - set(UNIT_INVENTORY_EXCLUDED))
    if unlisted:
        problems.append(f"files under the unit that the inventory does not list: {unlisted[:8]}"
                        + (" …" if len(unlisted) > 8 else ""))
    required = inv.get("required")
    if not isinstance(required, list):
        problems.append(f"{UNIT_INVENTORY_FILE} records no required-evidence list")
    else:
        missing = evidence_missing(required, files)
        if missing:
            problems.append(f"required evidence missing: {missing}")
        if inv.get("missing"):
            problems.append(f"{UNIT_INVENTORY_FILE} itself records missing evidence: {inv.get('missing')}")
    # a unit whose admission says its numerics were NOT measured must carry no
    # numerics evidence — probe legs or a lens per_row leg beside that literal are two
    # claims about one unit, refused by name
    if "admission.json" in raw:
        try:
            adm_rec = json.loads(raw["admission.json"].decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            adm_rec = None
        if numerics_mode_of(adm_rec) == NUMERICS_NOT_MEASURED:
            stray = numerics_evidence_present(list(files) + list(present))
            if stray:
                problems.append(f"admission says numerics {NUMERICS_NOT_MEASURED!r} but the unit carries "
                                f"numerics evidence: {stray[:8]}" + (" …" if len(stray) > 8 else ""))
    return problems


def read_unit_snapshot(unit_dir: Path, *, allow_absent_logs: bool = False
                       ) -> Tuple[Dict[str, bytes], List[str], List[str]]:
    """Every record of a unit read ONCE, verbatim, through its inventory:
    ``(snapshot, present, problems)`` — the snapshot maps ``INVENTORY.json``,
    ``DONE.json`` and every inventoried relative path to its bytes;
    ``present`` is every regular file under the unit. A unit without an
    inventory, or one whose inventoried file is missing or unreadable, comes
    back with the problem named and a partial snapshot. ``allow_absent_logs``
    (a PULLED record, whose server logs were deliberately not copied) lets an
    inventoried ``*.log`` be absent without a problem — the driver's reuse
    keeps the strict default, the report and the readmit relax it."""
    unit_dir = Path(unit_dir)
    snap: Dict[str, bytes] = {}
    problems: List[str] = []
    present = _walk_unit_files(unit_dir)
    for name in (UNIT_INVENTORY_FILE, "DONE.json"):
        p = unit_dir / name
        if not p.is_file():
            problems.append(f"{unit_dir} has no {name}")
            continue
        try:
            snap[name] = p.read_bytes()
        except OSError as e:
            problems.append(f"{p} unreadable: {e}")
    if problems:
        return snap, present, problems
    try:
        inv = json.loads(snap[UNIT_INVENTORY_FILE].decode("utf-8"))
        files = inv["files"] if isinstance(inv, Mapping) else None
        if not isinstance(files, Mapping):
            raise ValueError("no files map")
    except (ValueError, UnicodeDecodeError, KeyError) as e:
        return snap, present, [f"{unit_dir / UNIT_INVENTORY_FILE} is not an inventory document: {e}"]
    for rel in sorted(files):
        p = unit_dir / rel
        if not p.is_file():
            if allow_absent_logs and is_diagnostic_log(rel):
                continue
            problems.append(f"inventoried file missing: {rel}")
            continue
        try:
            snap[rel] = p.read_bytes()
        except OSError as e:
            problems.append(f"{p} unreadable: {e}")
    return snap, present, problems


def absent_diagnostic_logs(unit_dir: Path) -> List[str]:
    """The inventoried ``*.log`` files NOT present under ``unit_dir`` (a pulled
    record); ``[]`` when there is no readable inventory."""
    unit_dir = Path(unit_dir)
    p = unit_dir / UNIT_INVENTORY_FILE
    try:
        files = json.loads(p.read_text(encoding="utf-8")).get("files") or {}
    except (OSError, ValueError, AttributeError):
        return []
    return sorted(rel for rel in files if is_diagnostic_log(rel) and not (unit_dir / rel).is_file())


def unit_inventory_problems(unit_dir: Path, *, allow_absent_logs: bool = False) -> List[str]:
    """The inventory binding of a unit on disk (A2): ``INVENTORY.json`` present
    and bound by ``DONE.json``'s sha256 and attempt id, every inventoried file
    present with its digest, no unlisted file, and no required evidence
    missing. Empty = bound. ``allow_absent_logs``: see :func:`read_unit_snapshot`
    (an absent diagnostic log is reported by :func:`absent_diagnostic_logs`,
    never a pass and never a problem)."""
    snap, present, problems = read_unit_snapshot(unit_dir, allow_absent_logs=allow_absent_logs)
    if problems:
        return problems
    return unit_inventory_problems_from_bytes(snap, present=present,
                                              absent_logs=(absent_diagnostic_logs(unit_dir) if allow_absent_logs else ()))


def unit_binding_problems(unit_dir: Path, *, expected_attempt_id: Optional[str] = None) -> List[str]:
    """The binding of a unit's three attempt-bound records
    (:data:`UNIT_ATTEMPT_RECORDS`) to ONE attempt (F1): every one present,
    readable, carrying a non-empty :data:`ATTEMPT_ID_KEY`; the three ids
    equal (and equal ``expected_attempt_id`` when given — the manifest's id
    for a unit measured by that record's own attempt; ``None`` for a reused
    unit, whose records must agree among THEMSELVES under the prior attempt);
    and ``DONE.json``'s :data:`DONE_ADMISSION_SHA_KEY` the sha256 of the
    ``admission.json`` bytes on disk. Empty = bound. Fail closed: a record
    without an attempt id predates the binding and is nobody's. Reads the
    files ONCE and judges the bytes (:func:`unit_binding_problems_from_bytes`)."""
    unit_dir = Path(unit_dir)
    problems: List[str] = []
    docs: Dict[str, bytes] = {}
    for name in UNIT_ATTEMPT_RECORDS:
        path = unit_dir / name
        if not path.is_file():
            problems.append(f"no {name} under {unit_dir}")
            continue
        try:
            docs[name] = path.read_bytes()
        except OSError as e:
            problems.append(f"{name} unreadable: {e}")
    if problems:
        return problems
    return unit_binding_problems_from_bytes(docs, expected_attempt_id=expected_attempt_id)


def unit_binding_problems_from_bytes(raw: Mapping[str, bytes], *,
                                     expected_attempt_id: Optional[str] = None) -> List[str]:
    """:func:`unit_binding_problems` over record BYTES already in hand — the
    rule the driver applies to the snapshot it is about to copy on a reuse,
    so what it validates is exactly what it keeps (no second read of a
    source that could change underneath). ``raw`` maps each of
    :data:`UNIT_ATTEMPT_RECORDS` to the file's bytes."""
    problems: List[str] = []
    docs: Dict[str, Mapping[str, Any]] = {}
    for name in UNIT_ATTEMPT_RECORDS:
        if name not in raw:
            problems.append(f"no {name} in the snapshot")
            continue
        try:
            doc = json.loads(raw[name].decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            problems.append(f"{name} unreadable: {e}")
            continue
        if not isinstance(doc, Mapping):
            problems.append(f"{name} is not a JSON object")
            continue
        docs[name] = doc
    if problems:
        return problems
    ids = {name: docs[name].get(ATTEMPT_ID_KEY) for name in UNIT_ATTEMPT_RECORDS}
    for name, aid in ids.items():
        if not isinstance(aid, str) or not aid:
            problems.append(f"{name} carries no {ATTEMPT_ID_KEY} — a record from before the attempt "
                            "binding, or not the driver's")
    if problems:
        return problems
    if len(set(ids.values())) != 1:
        problems.append("attempt ids disagree across the unit's records: "
                        + ", ".join(f"{n}={str(i)[:8]}" for n, i in ids.items()))
    if expected_attempt_id is not None and ids["DONE.json"] != expected_attempt_id:
        problems.append(f"the unit's attempt {str(ids['DONE.json'])[:8]} != the expected attempt "
                        f"{str(expected_attempt_id)[:8]} (this record's manifest)")
    want = docs["DONE.json"].get(DONE_ADMISSION_SHA_KEY)
    got = hashlib.sha256(raw["admission.json"]).hexdigest()
    if want != got:
        problems.append(f"DONE.json {DONE_ADMISSION_SHA_KEY} {str(want)[:12]} != the admission.json "
                        f"{got[:12]} — not the admission that attempt completed with")
    return problems


def reused_unit_binding_problems(record_dir: Path, *, record_attempt_id: Optional[str],
                                 unit_reused_from: Optional[str]) -> List[str]:
    """The binding of a record that REUSED a unit:
    its :data:`UNIT_REUSE_FILE` is this record's attempt's and names the same
    source the manifest names (``unit_reused_from``, provenance only — never
    followed); it records the accepted source ``attempt_id`` and a sha256 per
    copied file; the self-contained copy under :data:`REUSED_UNIT_DIR` is
    present, binds to the source attempt (:func:`unit_binding_problems`), and
    every copied file hashes to its recorded digest; and the source attempt is
    not this record's own. Empty = bound. A source rerun in place (a new unit
    at the old path) changes nothing here; a copy or a reuse record altered
    after the acceptance is named."""
    record_dir = Path(record_dir)
    problems: List[str] = []
    p = record_dir / UNIT_REUSE_FILE
    if not p.is_file():
        return [f"no {UNIT_REUSE_FILE} under {record_dir} — the reuse was never recorded"]
    try:
        reuse = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return [f"{UNIT_REUSE_FILE} unreadable: {e}"]
    if not isinstance(reuse, Mapping):
        return [f"{UNIT_REUSE_FILE} is not a JSON object"]
    if reuse.get("unit_reused_from") != unit_reused_from:
        problems.append(f"{UNIT_REUSE_FILE} names unit_reused_from {reuse.get('unit_reused_from')!r}, "
                        f"the manifest {unit_reused_from!r}")
    if not record_attempt_id or reuse.get(ATTEMPT_ID_KEY) != record_attempt_id:
        problems.append(f"{UNIT_REUSE_FILE} attempt {str(reuse.get(ATTEMPT_ID_KEY))[:8]} is not this record's "
                        f"attempt {str(record_attempt_id)[:8]}")
    src = reuse.get("source")
    if (not isinstance(src, Mapping) or not isinstance(src.get(ATTEMPT_ID_KEY), str)
            or not src.get(ATTEMPT_ID_KEY) or not isinstance(src.get("sha256"), Mapping)):
        problems.append(f"{UNIT_REUSE_FILE} records no accepted source (attempt_id + per-file sha256) — "
                        "a reuse from before the self-contained copy")
        return problems
    src_attempt = src[ATTEMPT_ID_KEY]
    if src_attempt == record_attempt_id:
        problems.append("the accepted source attempt is this record's own attempt")
    copy = record_dir / REUSED_UNIT_DIR
    if not copy.is_dir():
        problems.append(f"no self-contained copy under {copy}")
        return problems
    problems += unit_binding_problems(copy, expected_attempt_id=src_attempt)
    # the WHOLE inventory (A2): the copy carries the source's INVENTORY.json +
    # DONE.json binding, every inventoried file with its digest, and the
    # required evidence; and every copied file hashes to what was accepted
    problems += [f"copy inventory: {p}" for p in unit_inventory_problems(copy)]
    digests: Mapping[str, Any] = src["sha256"]
    for name in (UNIT_INVENTORY_FILE, "DONE.json", *UNIT_ATTEMPT_RECORDS, "head_to_head.json"):
        if name not in digests:
            problems.append(f"{UNIT_REUSE_FILE} records no sha256 for {name}")
    copied = set(_walk_unit_files(copy)) | {n for n in UNIT_INVENTORY_EXCLUDED if (copy / n).is_file()}
    for name in sorted(copied - set(digests)):
        problems.append(f"{REUSED_UNIT_DIR}/{name} was not accepted (no recorded sha256)")
    for name, want in sorted(digests.items()):
        f = copy / name
        if not f.is_file():
            problems.append(f"{REUSED_UNIT_DIR}/{name} is missing")
            continue
        got = _sha256_file(f)
        if got != want:
            problems.append(f"{REUSED_UNIT_DIR}/{name} sha256 {got[:12]} != the accepted {str(want)[:12]}")
    if src.get(DONE_ADMISSION_SHA_KEY) != digests.get("admission.json"):
        problems.append(f"{UNIT_REUSE_FILE} source.{DONE_ADMISSION_SHA_KEY} disagrees with source.sha256[admission.json]")
    return problems


def resolve_reused_path(path: Any, *, source_unit_dir: Any, copy_dir: Path) -> Optional[Path]:
    """A path a reused record's copied documents name (the admission's
    ``gate_canary.record``, the controls summary's, …) resolved to the LOCAL
    copy: a path under the source unit maps to the same relative path under
    ``copy_dir``; anything else (or a non-path) is ``None``. The copies stay
    byte-verbatim (they are bound by digest), so references are resolved at
    read time, never rewritten in place."""
    if not isinstance(path, str) or not path or not source_unit_dir:
        return None
    try:
        rel = Path(path).relative_to(Path(str(source_unit_dir)))
    except ValueError:
        return None
    return Path(copy_dir) / rel


def _verdict_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, Mapping):
        for key in ("passed", "ok", "pass"):
            if isinstance(v.get(key), bool):
                return bool(v[key])
        st = v.get("status") or v.get("verdict")
        if isinstance(st, str):
            if st.lower() in ("passed", "pass", "ok", "admitted"):
                return True
            if st.lower() in ("failed", "fail", "refused", "not_admitted", "error"):
                return False
    return None


#: the selection control's structural verdict: ``serving_attestation``
#: names it once; the standup writes it; this reader holds it
SELECTION_VERDICT_ROUTING_ATTESTED = "routing_attested"


def selection_control_problems(entry: Any) -> List[str]:
    """The STRUCTURAL shape of a PASSED ``selection_honored`` record (gemma
    canary job 489308): ``verdict == "routing_attested"`` (the verdict came
    from the engine's per-request LoRA bookkeeping, not an output comparison),
    ``steps.n_steps_mixed >= 1`` (the batches were genuinely mixed), every
    submitted request observed (``n_requests >= 1``, ``n_observed ==
    n_requests``, ``n_missing == 0``), and the output diagnostics present as a
    non-gating ``numerics`` block — unless the record says ``output_diagnostics:
    false``, in which case NO numerics block may be present. Anything else is
    named; a record in the older output-based shape (``rule: equal-size`` at
    the top level, no verdict) is not the structural control and fails
    closed."""
    problems: List[str] = []
    if not isinstance(entry, Mapping):
        return [f"selection_honored is {type(entry).__name__}, not the control's record — a bare "
                "verdict carries no structural evidence and fails closed"]
    verdict = entry.get("verdict")
    if verdict != SELECTION_VERDICT_ROUTING_ATTESTED:
        problems.append(f"selection_honored verdict={verdict!r}, expected "
                        f"{SELECTION_VERDICT_ROUTING_ATTESTED!r} (D88: the structural verdict "
                        "from the engine's per-request LoRA bookkeeping)")
    steps = entry.get("steps")
    if not isinstance(steps, Mapping):
        problems.append("selection_honored carries no steps block")
    else:
        mixed = steps.get("n_steps_mixed")
        if isinstance(mixed, bool) or not isinstance(mixed, int) or mixed < 1:
            problems.append(f"selection_honored steps.n_steps_mixed={mixed!r}, expected >= 1 "
                            "(no engine step co-hosted two lora ids: the mixed claim is vacuous)")
    n_req = entry.get("n_requests")
    n_obs = entry.get("n_observed")
    n_missing = entry.get("n_missing")
    if isinstance(n_req, bool) or not isinstance(n_req, int) or n_req < 1:
        problems.append(f"selection_honored n_requests={n_req!r}, expected >= 1")
    if n_obs != n_req:
        problems.append(f"selection_honored n_observed={n_obs!r} != n_requests={n_req!r} "
                        "(every submitted request must be observed)")
    if n_missing != 0:
        problems.append(f"selection_honored n_missing={n_missing!r}, expected 0")
    numerics = entry.get("numerics")
    if entry.get("output_diagnostics") is False:
        # the control ran with the output diagnostics OFF (a residual-slot cell served, or the
        # certification's not_measured row) — the structural verdict stands on its own and NO
        # numerics block exists by design; one that is present contradicts the record
        if numerics is not None:
            problems.append("selection_honored says output_diagnostics false but carries a numerics block "
                            "(two claims about one control)")
    elif not isinstance(numerics, Mapping):
        problems.append("selection_honored carries no numerics block (the recorded output "
                        "diagnostics)")
    elif numerics.get("gating") is not False:
        problems.append(f"selection_honored numerics.gating={numerics.get('gating')!r}, expected "
                        "False (output comparisons are recorded, never the verdict)")
    return problems


def controls_summary(positive_controls: Any) -> Dict[str, Any]:
    """The five positive controls read from the
    standup's ``positive_controls.json`` BY NAME (:data:`CONTROL_NAMES`, E's
    own tuple): the record is ``{"controls": {name: {"passed": bool | None,
    ...}}, "passed": bool, "failed_control": ..., ...}`` (``lora_serving_up.
    run_positive_controls``); a bare ``{name: verdict}`` mapping is accepted
    too. Every one of the five names must carry a bool verdict — a control
    E recorded as not run (``passed: None``, "an earlier control failed") is
    named as such — and any OTHER verdict-shaped entry is a problem (a record
    with the wrong five names is not the five controls; code-review item 9).
    A record-level ``passed: False`` is a problem in its own right. A PASSED
    ``selection_honored`` must be the STRUCTURAL record
    (:func:`selection_control_problems`) — a bare ``True`` or any
    evidence-free form fails closed. Fail closed:
    ``all_passed`` only when the five named verdicts are all True and nothing
    is wrong."""
    out: Dict[str, Any] = {"passed": 0, "of": N_CONTROLS, "all_passed": False,
                           "verdicts": {}, "problems": [], "not_applicable": []}
    pc = positive_controls
    record_passed: Any = None
    legacy_five = False
    if isinstance(pc, Mapping) and isinstance(pc.get("controls"), Mapping):
        record_passed = pc.get("passed")
        if pc.get("failed_control") is not None:
            out["problems"].append(f"the record names failed_control={pc.get('failed_control')!r}")
        # a record E produced BEFORE the sixth control existed (its recorded run order is exactly the five
        # pre-O49 names and it carries no boundary entry): boundary_refusal reads not_applicable (legacy) and
        # its verdict is not listed, so the record's own summary block re-derives byte-equal. A fresh record
        # always records the six-name order, so a missing sixth control there is a problem, as below.
        legacy_five = (list(pc.get("order") or []) == list(CONTROL_NAMES[:5])
                       and BOUNDARY_CONTROL not in pc["controls"])
        pc = pc["controls"]
    if not isinstance(pc, Mapping):
        out["problems"].append(f"positive controls are {type(pc).__name__}, not a mapping")
        return out
    verdicts: Dict[str, Optional[bool]] = {}
    not_applicable: List[str] = []
    for name in CONTROL_NAMES:
        if name not in pc:
            if name == BOUNDARY_CONTROL and legacy_five:
                not_applicable.append(name)                     # pre-O49 record: the control did not exist
                continue
            out["problems"].append(f"control {name!r} is missing from the record")
            verdicts[name] = None
            continue
        entry = pc[name]
        if (name == BOUNDARY_CONTROL and isinstance(entry, Mapping)
                and entry.get("verdict") == NOT_APPLICABLE):
            # O49: acceptable ONLY with an EMPTY restricted-cell set stated in the record — a standup
            # that served a response-boundary cell must have run it
            if entry.get("restricted_cells") == []:
                verdicts[name] = None
                not_applicable.append(name)
            else:
                out["problems"].append(
                    f"control {name!r} is recorded not_applicable but the record does not state an EMPTY "
                    f"restricted-cell set (restricted_cells={entry.get('restricted_cells')!r}) — a standup "
                    "serving a restricted-mode cell must run the boundary-refusal control")
                verdicts[name] = None
            continue
        b = _verdict_bool(entry)
        verdicts[name] = b
        if b is None:
            out["problems"].append(f"control {name!r} carries no verdict "
                                   f"({pc[name]!r} — not run, or not a bool)")
        elif b is True and name == "selection_honored":
            # EVERY passed selection verdict must be the structural record (a bare
            # `selection_honored: true` bypassed the rule and admission_verdict read
            # `admitted` with no reasons); a bool, or any evidence-free form, fails
            # closed here
            out["problems"].extend(selection_control_problems(pc[name]))
    unexpected = sorted(str(n) for n in pc if n not in CONTROL_NAMES
                        and _verdict_bool(pc[n]) is not None)
    if unexpected:
        out["problems"].append(f"unexpected control verdicts {unexpected} — the controls are "
                               f"{list(CONTROL_NAMES)} (E's lora_serving_up.CONTROL_NAMES)")
    if record_passed is not None and record_passed is not True:
        out["problems"].append(f"the record's own passed={record_passed!r}")
    out["verdicts"] = verdicts
    out["not_applicable"] = not_applicable
    out["of"] = N_CONTROLS - len(not_applicable)               # the APPLICABLE controls
    out["passed"] = sum(1 for v in verdicts.values() if v is True)
    failed = sorted(n for n, v in verdicts.items() if v is False)
    if failed:
        out["problems"].append("failed: " + ", ".join(failed))
    out["all_passed"] = (out["passed"] == out["of"] and not out["problems"])
    return out


def _finite_number(v: Any) -> bool:
    return (not isinstance(v, bool) and isinstance(v, (int, float))
            and math.isfinite(float(v)))


def point_invalid_reasons(point: Mapping[str, Any]) -> List[str]:
    """Why one throughput window (a ``Point`` dict of ``throughput_leg``)
    cannot count: ``[]`` when it can. A window with any request error is
    invalid (engine tokens counted before a proxy post-check rejection are
    still on the counter); so is one with fewer than
    :data:`MIN_REQUESTS_DONE` successful completions, an achieved concurrency
    (``running_mean``, the engine's ``num_requests_running`` gauge) below
    :data:`MIN_RUNNING_MEAN_FRAC` of B, no ``tok_s``, or a field it cannot be
    judged without (fail closed: a point that does not record its errors or
    completions is not a measurement).

    F2: the CONCURRENCY and COUNTER evidence must be PRESENT, not merely
    consistent when present. A ``running_mean`` that is missing or not a
    finite number, a missing ``server_tokens`` (the engine counter delta), or
    a ``throughput_source`` other than :data:`THROUGHPUT_SOURCE_SERVER` (the
    driver's client-completion fallback, or an unrecorded source) each make
    the window invalid by name — the client count is a diagnostic, never the
    headline predicate's evidence."""
    reasons: List[str] = []
    errors = point.get("errors")
    if isinstance(errors, bool) or not isinstance(errors, int):
        reasons.append(f"errors unrecorded ({errors!r})")
    elif errors > 0:
        sample = point.get("error_sample")
        reasons.append(f"errors={errors}" + (f" ({sample})" if sample else ""))
    done = point.get("requests_done")
    if isinstance(done, bool) or not isinstance(done, int):
        reasons.append(f"requests_done unrecorded ({done!r})")
    elif done < MIN_REQUESTS_DONE:
        reasons.append(f"requests_done={done} < {MIN_REQUESTS_DONE}")
    B = point.get("B")
    b_ok = isinstance(B, int) and not isinstance(B, bool) and B > 0
    if not b_ok:
        reasons.append(f"B unrecorded ({B!r})")
    rm = point.get("running_mean")
    if rm is None:
        reasons.append("running_mean unrecorded (None): the engine's num_requests_running gauge "
                       "was never sampled, so the achieved concurrency is unattested")
    elif not _finite_number(rm):
        reasons.append(f"running_mean unreadable ({rm!r})")
    elif b_ok and rm < MIN_RUNNING_MEAN_FRAC * B:
        reasons.append(f"running_mean={float(rm):.1f} < {MIN_RUNNING_MEAN_FRAC} x B={B} "
                       "(the engine never held the requested concurrency)")
    st = point.get("server_tokens")
    if st is None:
        reasons.append("server_tokens unrecorded (None): the engine counter "
                       "vllm:generation_tokens_total was not read across the window")
    elif not _finite_number(st) or st < 0:
        reasons.append(f"server_tokens unreadable ({st!r})")
    src = point.get("throughput_source")
    if src != THROUGHPUT_SOURCE_SERVER:
        reasons.append(f"throughput_source={src!r}: only the engine counter ({THROUGHPUT_SOURCE_SERVER!r}) "
                       "is certification evidence; the client completion count is a recorded diagnostic")
    if point.get("tok_s") is None:
        reasons.append("tok_s missing")
    return reasons


def head_to_head(points_by_leg: Mapping[str, Sequence[Mapping[str, Any]]], *,
                 headline_B: int, ready_s: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``unit/head_to_head.json``: per B the mean tok/s
    (each point's ``tok_s`` — the ENGINE counter; a window whose counter was
    unreadable carries ``tok_s None`` and is invalid, F2) of ``lens_stacked`` and
    ``lora_native`` over their repeats, the ratio native / lens, both ms/step,
    the SM duty; ``not_slower_at_S = ratio(S) >= 1.0`` at ``headline_B`` (the
    table's S, or the lowered S when the hybrid retry fired — the caller names
    it); ``lens_per_row_tok_s_at_S`` as context. A B with no measurement on
    either side has ``ratio None`` and, at the headline, ``not_slower_at_S
    False`` with the gap named.

    ONLY VALID WINDOWS COUNT: every point is judged by
    :func:`point_invalid_reasons` first; an invalid one is excluded from every
    mean and named under ``per_B[B].invalid[leg]``. A headline leg with no
    valid point has ``ratio None``, ``not_slower_at_S False`` and its reasons
    under ``headline_invalid`` — so an errored lens window (engine tokens
    counted before a proxy rejection) can never manufacture a passing ratio."""
    legs = ("lens_stacked", "lora_native", "lens_per_row")
    invalid: Dict[str, Dict[int, Dict[int, List[str]]]] = {leg: {} for leg in legs}
    valid: Dict[str, List[Mapping[str, Any]]] = {leg: [] for leg in legs}
    for leg in legs:
        for p in points_by_leg.get(leg, ()):
            if p.get("B") is None:
                continue
            reasons = point_invalid_reasons(p)
            if reasons:
                invalid[leg].setdefault(int(p["B"]), {})[int(p.get("rep", len(invalid[leg].get(int(p["B"]), {}))))] = reasons
            else:
                valid[leg].append(p)

    def _mean(leg: str, B: int, key: str) -> Optional[float]:
        vals = [p.get(key) for p in valid.get(leg, ())
                if p.get("B") == B and p.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    def _n(leg: str, B: int) -> int:
        return sum(1 for p in points_by_leg.get(leg, ()) if p.get("B") == B)

    def _n_valid(leg: str, B: int) -> int:
        return sum(1 for p in valid.get(leg, ()) if p.get("B") == B)

    bs = sorted({int(p["B"]) for pts in points_by_leg.values() for p in pts if p.get("B") is not None})
    per_B: Dict[str, Any] = {}
    for B in bs:
        lens = _mean("lens_stacked", B, "tok_s")
        native = _mean("lora_native", B, "tok_s")
        per_B[str(B)] = {
            "lens_stacked_tok_s": lens, "lora_native_tok_s": native,
            "ratio": ((native / lens) if (lens and native) else None),
            "lens_ms_step": _mean("lens_stacked", B, "ms_per_step"),
            "native_ms_step": _mean("lora_native", B, "ms_per_step"),
            "sm_util": {"lens_stacked": _mean("lens_stacked", B, "sm_util_mean"),
                        "lora_native": _mean("lora_native", B, "sm_util_mean"),
                        "lens_per_row": _mean("lens_per_row", B, "sm_util_mean")},
            "n_rep": {"lens_stacked": _n("lens_stacked", B), "lora_native": _n("lora_native", B)},
            "n_valid": {"lens_stacked": _n_valid("lens_stacked", B),
                        "lora_native": _n_valid("lora_native", B),
                        "lens_per_row": _n_valid("lens_per_row", B)},
            "invalid": {leg: {str(rep): reasons for rep, reasons in sorted(invalid[leg].get(B, {}).items())}
                        for leg in legs if invalid[leg].get(B)},
        }
    S = int(headline_B)
    at_s = per_B.get(str(S), {})
    ratio_s = at_s.get("ratio")
    # a headline leg with measurements but NO valid one: the ratio is None and
    # the reasons are named (distinct from a leg that was never measured)
    headline_invalid = {leg: sorted({r for reasons in invalid[leg].get(S, {}).values() for r in reasons})
                        for leg in ("lens_stacked", "lora_native")
                        if _n(leg, S) > 0 and _n_valid(leg, S) == 0}
    return {
        "per_B": per_B,
        "headline_B": S,
        "ratio_at_S": ratio_s,
        "not_slower_at_S": bool(ratio_s is not None and ratio_s >= 1.0),
        "headline_missing": (ratio_s is None),
        "headline_invalid": headline_invalid,
        "validity": {"min_requests_done": MIN_REQUESTS_DONE,
                     "min_running_mean_frac": MIN_RUNNING_MEAN_FRAC,
                     "throughput_source_required": THROUGHPUT_SOURCE_SERVER,
                     "rule": "a window with any request error, too few completions, an achieved "
                             "concurrency far below B, or WITHOUT the engine's own evidence "
                             "(running_mean off num_requests_running, server_tokens off "
                             "generation_tokens_total; throughput_source == 'server') is invalid "
                             "and enters no mean — the client completion count is a diagnostic"},
        "lens_per_row_tok_s_at_S": _mean("lens_per_row", S, "tok_s"),
        "ready_s": dict(ready_s or {}),
    }


# --------------------------------------------------------------------------- #
# the paired bootstrap                                                          #
# --------------------------------------------------------------------------- #

def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """The ``q``-quantile of an already-sorted sequence by linear
    interpolation between order statistics (numpy's default method): with
    B = 2000 the choice of method moves the bound by less than one resample."""
    n = len(sorted_vals)
    if n == 0:
        raise AdmissionError("quantile of an empty sample")
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


#: the seed namespace is the rule the paired bootstrap was SPECIFIED under
#: (D111), not the current admission rule: the functional rule records the
#: same diagnostic bands D111 computed for the same attempt, so a schema-5
#: re-evaluation of a schema-4 record reproduces its sidecar's numbers exactly
#: (the seed used to embed ADMISSION_RULE; changing the rule would have moved
#: every recorded band by resampling noise for no reason)
BOOTSTRAP_SEED_NAMESPACE = "D111"


def bootstrap_seed(attempt_id: Optional[str]) -> int:
    """The bootstrap's seed, a pure function of the attempt id (the nonce the
    driver mints before any probe runs, so it cannot be chosen against the
    data) — the in-job admission and an offline :func:`readmit` of the same
    record draw the same resamples, under any rule
    (:data:`BOOTSTRAP_SEED_NAMESPACE`). ``None`` (a caller without an attempt)
    seeds from the empty id, still reproducibly."""
    h = hashlib.sha256(f"multilora-admission:{BOOTSTRAP_SEED_NAMESPACE}:{attempt_id or ''}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def paired_bootstrap_delta(per_a: Mapping[int, ProbeAgreement], per_b: Mapping[int, ProbeAgreement], *,
                           seed: int, n_boot: int = BOOTSTRAP_N, alpha: float = BOOTSTRAP_ALPHA
                           ) -> Dict[str, Any]:
    """The comparison WITH its sampling uncertainty: a paired-over-prompts
    bootstrap of ``Δ = disagreement(a) − disagreement(b)`` over the probes both
    pairs carry. Each of ``n_boot`` resamples draws the probe indices with
    replacement (stdlib ``random.Random(seed)``) and recomputes BOTH pairs'
    pooled disagreement on the resampled prompts from their per-prompt
    ``(agree_prefix, denom)`` contributions through
    :func:`pooled_token_agreement` — a sum over the resampled prompts, never an
    average of per-prompt rates — so the two statistics move together on the
    same prompts (a prompt that is hard for both pairs is drawn for both).
    Returns ``delta_point`` (the full-sample Δ), ``delta_ci95_low`` (the 5th
    percentile: the one-sided 95% lower bound the verdict reads),
    ``delta_ci95_high`` (the 95th percentile: the matching one-sided upper
    bound; together the central 90% band), a few more quantiles,
    ``frac_boot_delta_positive`` (how often a resample had native above the
    floor — descriptive), ``n_prompts`` (paired), ``n_prompts_differing``
    (prompts whose contributions differ between the two pairs: the only ones
    that can move Δ; 0 makes Δ identically 0), ``n_boot``, ``seed``. An empty
    paired set returns ``None`` bounds (the caller names what is unmeasured).

    What this is: the sensitivity of Δ to which of the fixed probe prompts were
    drawn, conditional on the probe set; :data:`BOOTSTRAP_CAVEATS` names what
    it does not cover (the structured set, shared scheduler effects,
    unvalidated coverage). The caller records both one-sided bounds and reads
    the UPPER one for admission and the LOWER one for refusal."""
    common = sorted(set(per_a) & set(per_b))
    n = len(common)
    A = [per_a[i] for i in common]
    B = [per_b[i] for i in common]
    out: Dict[str, Any] = {
        "method": ("paired-over-prompts percentile bootstrap of delta = disagreement(a) - disagreement(b); "
                   "prompt indices resampled with replacement; both pooled statistics recomputed per "
                   "resample as sum(agree_prefix)/sum(denom) over the drawn prompts (pooled_token_agreement); "
                   "bounds = linear-interpolated quantiles; delta_ci95_low is the one-sided 95% lower bound"),
        "n_prompts": n, "n_boot": int(n_boot), "seed": int(seed), "alpha": float(alpha),
        "delta_point": None, "delta_ci95_low": None, "delta_ci95_high": None,
        "quantiles": {}, "frac_boot_delta_positive": None, "n_prompts_differing": 0,
        "only_a": sorted(set(per_a) - set(per_b)), "only_b": sorted(set(per_b) - set(per_a)),
    }
    if n == 0:
        return out

    def delta_of(idx: Sequence[int]) -> float:
        dis_a = 1.0 - pooled_token_agreement([A[i] for i in idx])
        dis_b = 1.0 - pooled_token_agreement([B[i] for i in idx])
        return dis_a - dis_b

    out["delta_point"] = delta_of(range(n))
    out["n_prompts_differing"] = sum(1 for x, y in zip(A, B)
                                     if (x.agree_prefix, x.denom) != (y.agree_prefix, y.denom))
    rng = random.Random(int(seed))
    deltas = sorted(delta_of([rng.randrange(n) for _ in range(n)]) for _ in range(int(n_boot)))
    out["delta_ci95_low"] = _quantile(deltas, alpha)
    out["delta_ci95_high"] = _quantile(deltas, 1.0 - alpha)
    out["quantiles"] = {f"p{q * 100:g}": _quantile(deltas, q) for q in (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)}
    out["frac_boot_delta_positive"] = sum(1 for d in deltas if d > 0.0) / len(deltas)
    out["sd_boot"] = statistics.pstdev(deltas)
    return out


#: one-sided alpha 0.05 and power 0.80 normal quantiles for the planning number
_Z_ONE_SIDED_95 = 1.6448536269514722
_Z_POWER_80 = 0.8416212335729143


def influence_planning(per_a: Mapping[int, ProbeAgreement], per_b: Mapping[int, ProbeAgreement], *,
                       margins: Sequence[float] = (0.05, 0.10)) -> Optional[Dict[str, Any]]:
    """Planning numbers from the paired per-prompt INFLUENCE on Δ:
    Δ = R_b − R_a with R = sum(agree_prefix) / sum(denom) (so dis(a) − dis(b));
    the linearised (delta-method) influence of prompt i is
    ``(p_b,i − R_b d_b,i) / mean(d_b) − (p_a,i − R_a d_a,i) / mean(d_a)``, its
    sample sd over prompts gives ``se(Δ) ≈ sd / sqrt(n)``, and the prompts
    needed to detect a margin m at one-sided α = 0.05 with power 0.80 are
    ``((z_0.95 + z_0.80) sd / m)^2``. A planning APPROXIMATION under the same
    conditional-on-the-probe-set reading as the bootstrap (whose ``sd_boot``
    the caller records beside ``se_delta_linearised`` as a cross-check); it
    says how many prompts of THIS kind, not that more prompts of this kind
    would be the right instrument. ``None`` when fewer than two paired
    prompts or a zero denominator makes it undefined."""
    common = sorted(set(per_a) & set(per_b))
    n = len(common)
    if n < 2:
        return None
    A = [per_a[i] for i in common]
    B = [per_b[i] for i in common]
    Da, Db = sum(x.denom for x in A), sum(x.denom for x in B)
    if Da == 0 or Db == 0:
        return None
    Ra, Rb = sum(x.agree_prefix for x in A) / Da, sum(x.agree_prefix for x in B) / Db
    dbar_a, dbar_b = Da / n, Db / n
    infl = [(y.agree_prefix - Rb * y.denom) / dbar_b - (x.agree_prefix - Ra * x.denom) / dbar_a
            for x, y in zip(A, B)]
    sd = statistics.stdev(infl)
    z = _Z_ONE_SIDED_95 + _Z_POWER_80
    return {
        "method": ("delta-method influence of each prompt on delta (ratios of sums); n(m) = ((z_0.95 + z_0.80) * "
                   "sd_influence / m)^2 at one-sided alpha 0.05, power 0.80 — a planning approximation conditional "
                   "on prompts of this kind, not a guarantee"),
        "reads": PLANNING_NOTE,
        "n_prompts_now": n,
        "sd_influence": sd,
        "se_delta_linearised": sd / math.sqrt(n),
        "n_prompts_for_margin": {f"{m:g}": (int(math.ceil((z * sd / m) ** 2)) if sd > 0 else 0) for m in margins},
    }


# --------------------------------------------------------------------------- #
# the readiness input (the functional rule's first gate)                        #
# --------------------------------------------------------------------------- #

#: the attestation fields the readiness record copies (all stdlib values), so a
#: reader of admission.json sees WHAT was attested, not only that it was
READINESS_ENGINE_KEYS: Tuple[str, ...] = ("enforce_eager", "cudagraph_mode", "cudagraph_specialize_lora",
                                          "lens_plugin_imported")


def readiness_record(attestation: Optional[Mapping[str, Any]], *, problem: Optional[str] = None,
                     path: Optional[str] = None) -> Dict[str, Any]:
    """The functional rule's READINESS input from a strict-read serving
    attestation (``serving_attestation.read_serving_attestation``'s return
    value — the driver reads it in-job at ``_read_native_attestation``, the
    readmit re-reads the unit's file): ``attested`` is True iff the strict
    read SUCCEEDED (``attestation`` given, ``problem`` None); the engine bits
    (:data:`READINESS_ENGINE_KEYS`), the adapter names, ``name_binding`` and
    the kind are COPIED so the record says what was attested. A refused read
    is ``attested: False`` with the reader's ``problem`` verbatim — never a
    silent False. Pure: takes the parsed document, imports nothing."""
    out: Dict[str, Any] = {"attested": False, "source": READINESS_SOURCE, "path": path, "problem": problem,
                           "kind": None, "engine": None, "adapters": None, "name_binding": None,
                           "reads": ("attested iff the strict reader accepted the standup's serving attestation: "
                                     "graphs on (enforce_eager False), the lens plugin out, adapters bound by "
                                     "asserted name")}
    if attestation is None or problem is not None:
        if problem is None:
            out["problem"] = "no serving attestation was read"
        return out
    if not isinstance(attestation, Mapping):
        out["problem"] = f"attestation is {type(attestation).__name__}, not a mapping"
        return out
    engine = attestation.get("engine") if isinstance(attestation.get("engine"), Mapping) else {}
    adapters = attestation.get("adapters") if isinstance(attestation.get("adapters"), Mapping) else {}
    nb = attestation.get("name_binding") if isinstance(attestation.get("name_binding"), Mapping) else {}
    out.update(attested=True, kind=attestation.get("kind"),
               engine={k: engine.get(k) for k in READINESS_ENGINE_KEYS},
               adapters=sorted(str(a) for a in adapters),
               name_binding={"asserted": nb.get("asserted"), "basis": nb.get("basis")})
    return out


def _strict_attestation_reader():
    """The strict reader, imported lazily from the repo's ``src/`` (this module
    stays importable with the stdlib alone; only the readiness re-read of an
    offline re-evaluation needs the package). Returns ``(reader, error_class,
    None)`` or ``(None, None, why)`` when the package is not importable — the
    caller files WHY as a PROBLEM (an environment fact), never as
    ``attested: False`` (an evidence fact)."""
    try:
        import sys
        src = Path(__file__).resolve().parents[2] / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from sisteer.capeval.serving import serving_attestation as sa  # noqa: WPS433
        return sa.read_serving_attestation, sa.ServingAttestationError, None
    except Exception as e:  # noqa: BLE001 — an ImportError, or a broken package
        return None, None, f"the strict serving-attestation reader is not importable here: {type(e).__name__}: {e}"


#: where the unit keeps the standup's attestation (relative to ``unit/``)
UNIT_ATTESTATION_REL = "lora_native/serving_attestation.json"


def readiness_from_unit(unit_dir: Path, *, reader=None) -> Tuple[Dict[str, Any], Optional[str]]:
    """``(readiness_record, environment_problem)`` for a unit on disk: the
    attestation under :data:`UNIT_ATTESTATION_REL` read through the STRICT
    reader (``reader`` = ``(read_fn, error_class)`` when the caller supplies
    one; else the package's, imported lazily). A refused or absent
    attestation is a readiness record with ``attested: False`` and the
    reader's problem (an EVIDENCE fact: the verdict reads
    ``readiness_unattested``); a reader that cannot be imported is the
    second element (an ENVIRONMENT fact: the caller refuses to re-evaluate
    rather than filing a false refusal)."""
    unit_dir = Path(unit_dir)
    path = unit_dir / UNIT_ATTESTATION_REL
    if reader is None:
        read_fn, err_cls, why = _strict_attestation_reader()
        if read_fn is None:
            return readiness_record(None, problem="strict reader unavailable", path=str(path)), why
    else:
        read_fn, err_cls = reader
    try:
        att = read_fn(path)
    except err_cls as e:
        return readiness_record(None, problem=f"{type(e).__name__}: {e}", path=str(path)), None
    except (OSError, ValueError) as e:
        return readiness_record(None, problem=f"{type(e).__name__}: {e}", path=str(path)), None
    return readiness_record(att, path=str(path)), None


# --------------------------------------------------------------------------- #
# the verdict                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AdmissionInputs:
    """Everything :func:`admission_verdict` reads. ``legs`` maps a leg name to
    its rows; ``gate_canary`` is :func:`gate_canary_summary`'s output;
    ``controls`` :func:`controls_summary`'s; ``head_to_head``
    :func:`head_to_head`'s (only ``not_slower_at_S`` / ``ratio_at_S`` /
    ``headline_B`` are read)."""

    legs: Mapping[str, Sequence[Mapping[str, Any]]]
    gate_canary: Mapping[str, Any]
    controls: Mapping[str, Any]
    head_to_head: Mapping[str, Any]
    n_probes: int
    max_tokens: int
    #: the driver attempt that produced these inputs (F1): copied into the record
    attempt_id: Optional[str] = None
    #: the geometry the unit ran at — ``{"L", "S", "lowered": None | {...}}`` — so the
    #: record LABELS a lowered S (the LoRA kernel index ceiling, the Mamba retry)
    geometry: Optional[Mapping[str, Any]] = None
    #: E's ``noop_certificate`` record (``positive_controls.json`` → controls →
    #: noop_certificate: ``{passed, n_prompts, n_tokens, all_exact, ...}``): the
    #: SOLO zero-adapter certificate, which DECIDES ``zero_adapter_exact`` under the
    #: functional rule and is recorded with its exact scope under
    #: ``zero_adapter.solo_certificate``
    noop_certificate: Optional[Mapping[str, Any]] = None
    #: the cap the solo certificate ran at (the driver's ``--control-probe-max-tokens``)
    control_max_tokens: Optional[int] = None
    #: the READINESS input (:func:`readiness_record`): the standup's serving
    #: attestation as the strict reader accepted it. ``None`` = not recorded,
    #: which the verdict reads as ``readiness_unattested`` (fail closed).
    readiness: Optional[Mapping[str, Any]] = None
    #: the numerics MODE (:data:`NUMERICS_MODES`): ``diagnostic`` = the probe legs
    #: were run and are recorded; ``not_measured`` = the Target says no
    #: greedy-token agreement is measured (no probe legs, no per_row leg) — the record
    #: states ``numerics.mode`` / ``numerics.outcome`` = ``not_measured`` and names no
    #: missing leg as a finding
    numerics_mode: str = NUMERICS_DIAGNOSTIC


def admission_verdict(inputs: AdmissionInputs) -> Dict[str, Any]:
    """The ONE authoritative admission (plan native-serving-default §1.4;
    ``admission.json`` schema 5; the FUNCTIONAL rule):

    * ``verdict`` is TWO-WAY over the four functional gates (``gates``):
      ``admitted`` iff ``readiness.attested`` (the strict-read serving
      attestation: graphs on, plugin out, adapters bound by asserted name)
      and ``controls.all_passed`` (5/5 by E's names — ``noop_certificate`` IS
      the solo zero-adapter certificate, its scope recorded; ``salt_isolation``;
      ``steering_bites``; ``selection_honored``, structural; ``adapter_digests``)
      and ``gate_canary.passed`` (the §10 fp32-oracle canary) and
      ``head_to_head.not_slower_at_S``; ``not_admitted`` otherwise, with
      ``reasons`` naming each failed gate: ``readiness_unattested`` /
      ``controls_failed`` / ``gate_canary_failed`` / ``gate_canary_missing`` /
      ``slower_at_S`` / ``head_to_head_missing`` / ``head_to_head_invalid:…``.
      Nothing else refuses.
    * ``zero_adapter_exact`` — the SOLO certificate (``zero_adapter.source ==
      "noop_certificate"``): E's ``noop_certificate`` record's ``passed``, else
      the control's verdict in ``controls``; its scope (``n_prompts``,
      ``max_tokens``) recorded. The concurrent ``lora_zero`` vs ``lora_none``
      legs (43 × 128) are RECORDED under ``zero_adapter.concurrent`` — pooled
      disagreement, ``n_diverging``, whether they disagree with the solo
      instrument — and refuse nothing (``zero_adapter.instruments_disagree``
      is a stated finding, not a reason).
    * ``numerics`` — everything measured, RECORDED with ``gate: false``
      and :data:`NUMERICS_NOTE`: the agreement table over :data:`PAIRS`; the
      PRIMARY ``agreement(lens_per_row, lens_stacked)`` floor (``degenerate``
      iff zero diverging prompts); Δ = disagreement(native_vs_stacked) −
      disagreement(primary) with its paired prompt-resampling band
      (:func:`paired_bootstrap_delta`, B = :data:`BOOTSTRAP_N`, seeded from the
      attempt id) and the three-way ``numerics.outcome`` at the literal m = 0
      (``admitted`` iff the one-sided 95% UPPER bound ≤ 0, ``not_admitted``
      iff the LOWER bound > 0, else ``inconclusive``; ``degenerate_gate_canary``
      when the floor is degenerate; ``unmeasured`` when a leg is missing);
      ``numerics.reasons`` are the diagnostic FINDINGS in the numerics
      vocabulary (``native_disagreement_exceeds_floor``, ``leg_missing:…``,
      ``zero_adapter_inexact:concurrent_legs``, …) and never enter ``reasons``;
      the point inequality; the planning numbers; ``floor.diagnostics`` (the
      native engine's own variability pairs and the lens's, each with its own
      band against the primary).
    * ``diagnostics.native_deterministic`` / ``stacked_deterministic`` — RECORDED
      ONLY; the ulps — descriptive. Neither is read by anything."""
    mode = str(inputs.numerics_mode)
    if mode not in NUMERICS_MODES:
        raise AdmissionError(f"numerics_mode {mode!r} is not one of {list(NUMERICS_MODES)}")
    not_measured = mode == NUMERICS_NOT_MEASURED
    legs: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for leg, rows in inputs.legs.items():
        if rows is None:
            continue
        legs[leg] = rows_by_probe(rows, leg=leg)
    if not_measured and any(legs.values()):
        # the known-bad, at the verdict too: a not_measured unit that hands in probe legs
        raise AdmissionError(f"numerics_mode {NUMERICS_NOT_MEASURED!r} with probe legs present "
                             f"({sorted(k for k, v in legs.items() if v)}): a unit is one claim or the other")
    numerics_reasons: List[str] = []
    notes: Dict[str, Any] = {}
    if not not_measured:
        for leg in REQUIRED_LEGS:
            if leg not in legs or not legs[leg]:
                numerics_reasons.append(f"leg_missing:{leg}")

    # agreement per pair — the table and every bootstrap read the SAME per-probe objects
    per_pair: Dict[str, Dict[int, ProbeAgreement]] = {}
    agreement: Dict[str, Any] = {}
    for name, (a, b) in PAIRS.items():
        per, note = per_probe_agreements(legs, a, b)
        per_pair[name] = per
        agreement[name] = pool_probe_agreements(list(per.values()))
        if note:
            notes[name] = note
            if "probe_set_mismatch" in note and a in REQUIRED_LEGS and b in REQUIRED_LEGS:
                numerics_reasons.append(f"probe_set_mismatch:{name}")

    def _stat(pair: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: pair[k] for k in ("disagreement", "n_diverging", "n")}

    margin = dict(DIAGNOSTIC_MARGIN)                  # the RECORDED outcome's threshold: the literal m = 0
    m = float(margin["m"])
    controls = dict(inputs.controls)

    # ---- the zero adapter: the SOLO certificate decides; the concurrent legs are recorded
    zn = agreement["native_zero_vs_none"]
    concurrent_measured = bool(zn["n"] > 0 and "probe_set_mismatch" not in notes.get("native_zero_vs_none", {}))
    concurrent_exact = bool(concurrent_measured and zn["n_diverging"] == 0)
    cert = inputs.noop_certificate if isinstance(inputs.noop_certificate, Mapping) else None
    verdicts = controls.get("verdicts") if isinstance(controls.get("verdicts"), Mapping) else {}
    solo_passed: Optional[bool] = None
    if cert is not None and isinstance(cert.get("passed"), bool):
        solo_passed = bool(cert["passed"])
    elif isinstance(verdicts.get(ZERO_ADAPTER_CONTROL), bool):
        solo_passed = bool(verdicts[ZERO_ADAPTER_CONTROL])
    solo_n = cert.get("n_prompts") if cert is not None and isinstance(cert.get("n_prompts"), int) else None
    solo_cap: Optional[int] = None
    solo_cap_basis: Optional[str] = None
    if isinstance(inputs.control_max_tokens, int) and not isinstance(inputs.control_max_tokens, bool):
        solo_cap, solo_cap_basis = int(inputs.control_max_tokens), "control_probe_max_tokens (the cap the control ran at)"
    elif cert is not None and isinstance(cert.get("n_tokens"), list) and cert["n_tokens"]:
        try:
            solo_cap, solo_cap_basis = int(max(cert["n_tokens"])), "observed max n_tokens (the cap itself was not recorded)"
        except (TypeError, ValueError):
            solo_cap, solo_cap_basis = None, None
    instruments = {
        "concurrent_legs": ("unmeasured" if not concurrent_measured else ("passed" if concurrent_exact else "failed")),
        "solo_certificate": ("not_recorded" if solo_passed is None else ("passed" if solo_passed else "failed")),
    }
    # the concurrent instrument's reading is a diagnostic FINDING (numerics), never a reason
    # (under not_measured the concurrent legs are absent by design — no finding to name)
    if not concurrent_measured:
        if not not_measured:
            numerics_reasons.append("zero_adapter_unmeasured:concurrent_legs")
    elif not concurrent_exact:
        numerics_reasons.append("zero_adapter_inexact:concurrent_legs")
    zero_exact = solo_passed is True                       # THE zero-adapter verdict: the solo certificate
    instruments_disagree = bool(concurrent_measured and solo_passed is not None and concurrent_exact != solo_passed)
    zero_adapter = {
        "exact": zero_exact,
        "source": ZERO_ADAPTER_SOURCE,
        "rule": ADMISSION_RULE,
        "solo_certificate": {"passed": solo_passed, "n_prompts": solo_n, "max_tokens": solo_cap,
                             "max_tokens_basis": solo_cap_basis,
                             "label": (f"solo zero-adapter token-exact on {solo_n if solo_n is not None else '?'} probes, "
                                       f"cap {solo_cap if solo_cap is not None else '?'}: "
                                       f"{'passed' if solo_passed else ('FAILED' if solo_passed is False else 'not recorded')}"),
                             "source": f"{ZERO_ADAPTER_CONTROL} (standup control a, D1 one request at a time)",
                             "decides": True},
        "concurrent": {**_stat(zn), "max_tokens": int(inputs.max_tokens), "measured": concurrent_measured,
                       "exact": concurrent_exact, "recorded_only": True,
                       "label": (f"concurrent zero-adapter vs base, {zn['n']} probes x cap {int(inputs.max_tokens)} "
                                 f"tokens: {zn['n_diverging']} diverge, pooled disagreement "
                                 f"{(zn['disagreement'] if zn['disagreement'] is not None else float('nan')):.4f}"
                                 " (recorded; production decodes by sampling)")},
        "instruments": instruments,
        "failed_instruments": [k for k, v in instruments.items() if v == "failed"],
        "instruments_disagree": instruments_disagree,
        "note": ("user ruling 2026-09-13: zero-adapter exactness IS the solo certificate (D1, one request at a time), "
                 "recorded with its exact scope; the concurrent 43 x 128 legs' greedy-token disagreement is recorded "
                 "beside it as a diagnostic and refuses nothing. A failed solo certificate fails the noop_certificate "
                 "control, so it reaches the verdict as controls_failed."),
    }

    # ---- determinism: recorded, never a precondition
    rep_n = agreement["native_rep_vs_native"]
    rep_s = agreement["stacked_rep_vs_stacked"]
    diagnostics = {
        "native_deterministic": (rep_n["n"] > 0 and rep_n["n_diverging"] == 0) if rep_n["n"] else None,
        "stacked_deterministic": (rep_s["n"] > 0 and rep_s["n_diverging"] == 0) if rep_s["n"] else None,
        "note": "recorded only (D27): neither flag is read by numerics.admitted or the verdict",
    }

    # ---- ulps: descriptive
    ulps: Dict[str, Any] = {}
    for name, (search, argmax) in ULPS_PAIRS.items():
        if search in legs and argmax in legs:
            ulps[name] = first_token_ulps(legs[search], legs[argmax])
        else:
            ulps[name] = {"max": None, "p50": None, "n_present": 0, "n_missing": 0,
                          "match_basis": {"id": 0, "token": 0},
                          "missing_legs": [x for x in (search, argmax) if x not in legs]}

    # ---- the floor: the PRIMARY, its margin, and the recorded diagnostics
    primary = agreement[FLOOR_PRIMARY]
    native_pair = agreement["native_vs_stacked"]
    floor_measured = primary["n"] > 0
    degenerate = bool(floor_measured and primary["n_diverging"] == 0)
    seed = bootstrap_seed(inputs.attempt_id)
    diag_pairs: Dict[str, Any] = {}
    for name in FLOOR_DIAGNOSTIC_PAIRS:
        pr = agreement[name]
        entry: Dict[str, Any] = {**_stat(pr), "delta_vs_primary": None}
        if floor_measured and not degenerate and pr["n"] > 0:
            b = paired_bootstrap_delta(per_pair[name], per_pair[FLOOR_PRIMARY],
                                       seed=bootstrap_seed(f"{inputs.attempt_id or ''}:{name}"))
            entry["delta_vs_primary"] = {k: b[k] for k in ("delta_point", "delta_ci95_low", "delta_ci95_high",
                                                            "n_prompts", "n_prompts_differing", "n_boot", "seed")}
        diag_pairs[name] = entry

    def _max_pair(names: Sequence[str]) -> Dict[str, Any]:
        measured = [k for k in names if agreement[k]["n"] > 0]
        if not measured:
            return {"pair": None, "disagreement": None}
        best = max(measured, key=lambda k: agreement[k]["disagreement"])
        return {"pair": best, "disagreement": agreement[best]["disagreement"]}

    native_var = _max_pair(NATIVE_VARIABILITY_PAIRS)
    lens_var = _max_pair(LENS_VARIABILITY_PAIRS)
    exceeds: Optional[bool] = None
    if native_var["disagreement"] is not None and lens_var["disagreement"] is not None:
        exceeds = bool(native_var["disagreement"] > lens_var["disagreement"])
    floor = {
        "rule": ADMISSION_RULE,
        "primary": FLOOR_PRIMARY,
        **_stat(primary),
        "degenerate": degenerate,
        "source": (FLOOR_FALLBACK if degenerate else FLOOR_PRIMARY),
        "margin": dict(margin),
        "diagnostics": {
            "pairs": diag_pairs,
            "native_variability": native_var,
            "lens_variability": lens_var,
            "native_variability_exceeds_lens": exceeds,
            "note": ("RECORDED, never a gate (D111): each pair's own paired band against the primary; no "
                     "pre-registered bound exists on the engine's own variability, so the verdict reads the "
                     "primary floor only. 'native variability exceeds the lens's own' is a stated finding."),
        },
    }

    # ---- the three-way comparison against the primary
    gate = dict(inputs.gate_canary)
    gate_passed = bool(gate.get("passed"))
    gate_status = gate.get("status")
    within_floor: Optional[bool] = None
    point_ineq: Optional[bool] = ((native_pair["disagreement"] <= primary["disagreement"])
                                  if (floor_measured and native_pair["n"] > 0) else None)
    outcome = "unmeasured"
    prs: Dict[str, Any] = {
        "rule": ADMISSION_RULE, "pair": "native_vs_stacked", "floor_pair": FLOOR_PRIMARY, "margin_m": m,
        "delta_point": None, "delta_ci95_low": None, "delta_ci95_high": None,
        "bounds": {"lower95_one_sided": None, "upper95_one_sided": None},
        "n_boot": BOOTSTRAP_N, "seed": seed, "alpha": BOOTSTRAP_ALPHA, "skipped": None,
        "reads": "admitted iff upper95_one_sided <= m; not_admitted iff lower95_one_sided > m; else inconclusive",
        "caveats": list(BOOTSTRAP_CAVEATS),
        "planning": None,
    }
    if not_measured:
        outcome = NUMERICS_NOT_MEASURED
        prs["skipped"] = ("not measured by design (native-psr-serving D8): no probe leg and no lens per_row leg "
                          "were run for this certification; the functional gates are the whole verdict")
    elif not floor_measured:
        if "leg_missing:lens_per_row" not in numerics_reasons and \
                "leg_missing:lens_stacked" not in numerics_reasons:
            numerics_reasons.append("floor_unmeasured")
        prs["skipped"] = "floor unmeasured"
    elif degenerate:
        outcome = "degenerate_gate_canary"
        if not gate_passed:
            numerics_reasons.append(
                "floor_degenerate_and_gate_canary_"
                + ("missing" if gate_status in (None, "missing") else "failed"))
        prs["skipped"] = ("primary floor degenerate (0 diverging prompts): the gate canary's D39 near-tie comparator "
                          "vs the fp32 oracle is the yardstick (D10)")
    else:
        if native_pair["n"] == 0:
            numerics_reasons.append("native_vs_stacked_unmeasured")
            prs["skipped"] = "native_vs_stacked unmeasured"
        else:
            b = paired_bootstrap_delta(per_pair["native_vs_stacked"], per_pair[FLOOR_PRIMARY], seed=seed)
            prs.update(b)
            low, high = b["delta_ci95_low"], b["delta_ci95_high"]
            prs["bounds"] = {"lower95_one_sided": low, "upper95_one_sided": high}
            planning = influence_planning(per_pair["native_vs_stacked"], per_pair[FLOOR_PRIMARY])
            if planning is not None:
                planning["se_delta_bootstrap"] = b["sd_boot"]
            prs["planning"] = planning
            if high is not None and high <= m:
                outcome, within_floor = "admitted", True
            elif low is not None and low > m:
                outcome, within_floor = "not_admitted", False
                numerics_reasons.append("native_disagreement_exceeds_floor")
            else:
                outcome, within_floor = "inconclusive", False
                numerics_reasons.append("native_disagreement_inconclusive")
        if not gate_passed:
            numerics_reasons.append(
                "gate_canary_missing" if gate_status in (None, "missing") else "gate_canary_failed")
    structural_ok = not any(r.startswith(("leg_missing:", "probe_set_mismatch:")) for r in numerics_reasons)
    # de-duplicate while keeping order
    seen: set = set()
    numerics_reasons = [r for r in numerics_reasons if not (r in seen or seen.add(r))]
    n_empty_total = sum(v.get("n_empty", 0) for v in agreement.values())
    if n_empty_total:
        notes["n_empty"] = {k: v.get("n_empty", 0) for k, v in agreement.items() if v.get("n_empty")}

    # ---- the FUNCTIONAL gates: the only inputs the verdict reads
    readiness = (dict(inputs.readiness) if isinstance(inputs.readiness, Mapping)
                 else readiness_record(None, problem="readiness not recorded in the inputs"))
    readiness_ok = readiness.get("attested") is True
    controls_ok = bool(controls.get("all_passed"))
    h2h = dict(inputs.head_to_head)
    not_slower = bool(h2h.get("not_slower_at_S"))
    reasons: List[str] = []
    if not readiness_ok:
        reasons.append("readiness_unattested")
    if not controls_ok or solo_passed is False:
        # a failed solo certificate fails the noop_certificate control; named once
        reasons.append("controls_failed")
    if not gate_passed:
        reasons.append("gate_canary_missing" if gate_status in (None, "missing") else "gate_canary_failed")
    if not not_slower:
        if h2h.get("headline_invalid"):
            # a headline leg measured but every window invalid (errors, too few
            # completions, concurrency never reached): named, never "missing"
            reasons.append("head_to_head_invalid:" + ",".join(sorted(h2h["headline_invalid"])))
        else:
            reasons.append("slower_at_S" if h2h.get("ratio_at_S") is not None else "head_to_head_missing")
    gates = {"readiness_attested": readiness_ok, "controls_all_passed": controls_ok and solo_passed is not False,
             "gate_canary_passed": gate_passed, "not_slower_at_S": not_slower}
    assert tuple(gates) == FUNCTIONAL_GATES
    verdict = "admitted" if all(gates.values()) else "not_admitted"
    assert (verdict == "admitted") == (not reasons)
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": ADMISSION_RULE,
        "attempt_id": inputs.attempt_id,
        "geometry": (dict(inputs.geometry) if inputs.geometry is not None else None),
        "n_probes": int(inputs.n_probes),
        "max_tokens": int(inputs.max_tokens),
        "zero_adapter_exact": zero_exact,
        "zero_adapter": zero_adapter,
        "readiness": readiness,
        "gates": {**gates, "rule": ("admitted iff every gate is true: readiness attested (strict), five controls "
                                    "passed, gate canary passed, not slower than the lens at S; the numerics block "
                                    "is recorded and never read")},
        "diagnostics": diagnostics,
        "agreement": agreement,
        "first_token_ulps": ulps,
        "first_token_ulps_note": "descriptive (D11): never read by the verdict",
        "floor": floor,
        "gate_canary": {k: gate.get(k) for k in ("ran", "passed", "status", "lora_canary_passed",
                                                 "noop_lora_passed", "noop_passthrough_passed",
                                                 "multi_lora_checked", "margin_atol_nats",
                                                 "record", "reason", "bound", "attempt")},
        "numerics": {"gate": False,
                     "mode": mode,
                     "mode_note": ("diagnostic: the D111 legs were run and recorded; not_measured: NO greedy-token "
                                   "agreement was measured for this certification (native-psr-serving D8 — the "
                                   "user's ruling for PSR), so every agreement/floor/ulps entry below is empty by "
                                   "design and outcome reads not_measured"),
                     "note": NUMERICS_NOTE,
                     "outcome": outcome,
                     "outcome_note": ("D111's three-way comparison of native-vs-stacked against the primary "
                                      "per_row-vs-stacked floor at m = 0, RECORDED as a diagnostic: its vocabulary "
                                      "(admitted / not_admitted / inconclusive) names the comparison's outcome, not "
                                      "the standup's verdict"),
                     "reasons": numerics_reasons,
                     "reasons_note": "diagnostic findings in D111's vocabulary; none of them enters the verdict",
                     "structural_ok": structural_ok,
                     "floor_measured": floor_measured,
                     "within_floor": within_floor,
                     "primary_point_inequality": point_ineq,
                     "primary_point_inequality_note": ("the point inequality native.disagreement <= "
                                                       "primary.disagreement (the pre-bootstrap reading), RECORDED "
                                                       "and never read"),
                     "margin": dict(margin),
                     "prompt_resampling_sensitivity": prs},
        "controls": {"passed": int(controls.get("passed", 0)), "of": int(controls.get("of", N_CONTROLS)),
                     "all_passed": controls_ok, "problems": list(controls.get("problems", []))},
        "head_to_head": {"not_slower_at_S": not_slower, "ratio_at_S": h2h.get("ratio_at_S"),
                         "headline_B": h2h.get("headline_B")},
        "verdict": verdict,
        "verdict_note": ("functional (user ruling 2026-09-13): admitted iff readiness attested, five controls passed, "
                         "gate canary passed, not slower at S; the greedy-token numerics are recorded diagnostics "
                         "(numerics.gate is false)"),
        "reasons": reasons,
        "notes": notes,
    }

# --- schema 4, FROZEN as it was: the three historical sidecars validate against these
REQUIRED_ADMISSION_KEYS_V4: Tuple[str, ...] = (
    "schema_version", "rule", "attempt_id", "n_probes", "max_tokens", "zero_adapter_exact", "zero_adapter",
    "diagnostics", "agreement", "first_token_ulps", "floor", "gate_canary", "numerics", "controls",
    "head_to_head", "verdict", "reasons")
REQUIRED_FLOOR_KEYS_V4: Tuple[str, ...] = ("primary", "disagreement", "n_diverging", "n", "degenerate", "source",
                                           "margin", "diagnostics")
REQUIRED_NUMERICS_KEYS_V4: Tuple[str, ...] = ("admitted", "outcome", "reasons", "within_floor",
                                              "primary_point_inequality", "margin", "prompt_resampling_sensitivity")
REQUIRED_SENSITIVITY_KEYS_V4: Tuple[str, ...] = ("floor_pair", "margin_m", "delta_point", "delta_ci95_low",
                                                 "delta_ci95_high", "bounds", "n_boot", "seed", "caveats")
REQUIRED_ZERO_KEYS_V4: Tuple[str, ...] = ("exact", "source", "concurrent", "solo_certificate", "instruments")
# --- schema 5 (the functional rule): the schema-4 blocks kept whole, plus readiness + gates, the numerics
# block marked ``gate: false`` (no ``admitted`` in it), the two-way verdict
REQUIRED_ADMISSION_KEYS_V5: Tuple[str, ...] = REQUIRED_ADMISSION_KEYS_V4 + ("readiness", "gates")
REQUIRED_FLOOR_KEYS_V5: Tuple[str, ...] = REQUIRED_FLOOR_KEYS_V4
REQUIRED_NUMERICS_KEYS_V5: Tuple[str, ...] = ("gate", "note", "outcome", "reasons", "within_floor",
                                              "primary_point_inequality", "margin", "prompt_resampling_sensitivity")
REQUIRED_SENSITIVITY_KEYS_V5: Tuple[str, ...] = REQUIRED_SENSITIVITY_KEYS_V4
REQUIRED_ZERO_KEYS_V5: Tuple[str, ...] = REQUIRED_ZERO_KEYS_V4 + ("instruments_disagree",)
REQUIRED_READINESS_KEYS: Tuple[str, ...] = ("attested", "source", "problem")
REQUIRED_GATES_KEYS: Tuple[str, ...] = FUNCTIONAL_GATES
#: the CURRENT schema's tables under the unsuffixed names (what a producer of new records reads)
REQUIRED_ADMISSION_KEYS = REQUIRED_ADMISSION_KEYS_V5
REQUIRED_FLOOR_KEYS = REQUIRED_FLOOR_KEYS_V5
REQUIRED_NUMERICS_KEYS = REQUIRED_NUMERICS_KEYS_V5
REQUIRED_SENSITIVITY_KEYS = REQUIRED_SENSITIVITY_KEYS_V5
REQUIRED_ZERO_KEYS = REQUIRED_ZERO_KEYS_V5


def _validate_common_blocks(rec: Mapping[str, Any], *, floor_keys: Tuple[str, ...], numerics_keys: Tuple[str, ...],
                            sensitivity_keys: Tuple[str, ...], zero_keys: Tuple[str, ...], zero_source: str,
                            verdicts: Tuple[str, ...]) -> List[str]:
    """The blocks schema 4 and schema 5 share, checked against the tables the
    caller names for its schema: every :data:`PAIRS` name under ``agreement``,
    the three ulps pairs, the floor's keys and source vocabulary
    (:data:`FLOOR_SOURCES`) with its margin and diagnostics, the numerics'
    three-way ``outcome`` and ``prompt_resampling_sensitivity`` block, the
    zero-adapter block with both instruments and ITS schema's source, the
    verdict vocabulary, the determinism flags."""
    problems: List[str] = []
    ag = rec.get("agreement") or {}
    problems += [f"agreement lacks {p}" for p in PAIRS if p not in ag]
    ul = rec.get("first_token_ulps") or {}
    problems += [f"first_token_ulps lacks {p}" for p in ULPS_PAIRS if p not in ul]
    fl = rec.get("floor") or {}
    problems += [f"floor lacks {k}" for k in floor_keys if k not in fl]
    if fl.get("source") not in FLOOR_SOURCES:
        problems.append(f"floor.source {fl.get('source')!r}")
    mg = fl.get("margin")
    if not isinstance(mg, Mapping) or not _finite_number(mg.get("m")) or not mg.get("source"):
        problems.append("floor.margin must carry a finite m and a source")
    dg = fl.get("diagnostics")
    if not isinstance(dg, Mapping) or not isinstance(dg.get("pairs"), Mapping):
        problems.append("floor.diagnostics lacks pairs")
    else:
        problems += [f"floor.diagnostics.pairs lacks {p}" for p in FLOOR_DIAGNOSTIC_PAIRS if p not in dg["pairs"]]
        if "native_variability_exceeds_lens" not in dg:
            problems.append("floor.diagnostics lacks native_variability_exceeds_lens")
    nm = rec.get("numerics") or {}
    problems += [f"numerics lacks {k}" for k in numerics_keys if k not in nm]
    if nm.get("outcome") not in OUTCOMES:
        problems.append(f"numerics.outcome {nm.get('outcome')!r}")
    prs = nm.get("prompt_resampling_sensitivity")
    if not isinstance(prs, Mapping):
        problems.append("numerics lacks prompt_resampling_sensitivity")
    else:
        problems += [f"numerics.prompt_resampling_sensitivity lacks {k}" for k in sensitivity_keys if k not in prs]
    za = rec.get("zero_adapter")
    if not isinstance(za, Mapping):
        problems.append("zero_adapter missing")
    else:
        problems += [f"zero_adapter lacks {k}" for k in zero_keys if k not in za]
        if za.get("source") != zero_source:
            problems.append(f"zero_adapter.source {za.get('source')!r} != {zero_source!r}")
        sc = za.get("solo_certificate")
        if not isinstance(sc, Mapping) or not {"passed", "n_prompts", "max_tokens", "label"} <= set(sc):
            problems.append("zero_adapter.solo_certificate lacks passed / n_prompts / max_tokens / label")
    if rec.get("verdict") not in verdicts:
        problems.append(f"verdict {rec.get('verdict')!r} not in {list(verdicts)}")
    for k in ("native_deterministic", "stacked_deterministic"):
        if k not in (rec.get("diagnostics") or {}):
            problems.append(f"diagnostics lacks {k}")
    return problems


def _validate_v4(rec: Mapping[str, Any]) -> List[str]:
    problems = [f"missing key {k}" for k in REQUIRED_ADMISSION_KEYS_V4 if k not in rec]
    return problems + _validate_common_blocks(
        rec, floor_keys=REQUIRED_FLOOR_KEYS_V4, numerics_keys=REQUIRED_NUMERICS_KEYS_V4,
        sensitivity_keys=REQUIRED_SENSITIVITY_KEYS_V4, zero_keys=REQUIRED_ZERO_KEYS_V4,
        zero_source=ZERO_ADAPTER_SOURCE_V4, verdicts=VERDICTS_V4)


def _validate_v5(rec: Mapping[str, Any]) -> List[str]:
    problems = [f"missing key {k}" for k in REQUIRED_ADMISSION_KEYS_V5 if k not in rec]
    problems += _validate_common_blocks(
        rec, floor_keys=REQUIRED_FLOOR_KEYS_V5, numerics_keys=REQUIRED_NUMERICS_KEYS_V5,
        sensitivity_keys=REQUIRED_SENSITIVITY_KEYS_V5, zero_keys=REQUIRED_ZERO_KEYS_V5,
        zero_source=ZERO_ADAPTER_SOURCE, verdicts=VERDICTS)
    nm = rec.get("numerics") or {}
    if nm.get("gate") is not False:
        problems.append(f"numerics.gate {nm.get('gate')!r} != False (the numerics are recorded, never a gate)")
    if "admitted" in nm:
        problems.append("numerics carries an `admitted` verdict (schema 5 records none: the numerics gate nothing)")
    mode = nm.get("mode", NUMERICS_DIAGNOSTIC)
    if mode not in NUMERICS_MODES:
        problems.append(f"numerics.mode {mode!r} not in {list(NUMERICS_MODES)}")
    elif mode == NUMERICS_NOT_MEASURED:
        # the literal is one claim — the outcome must read it too and no leg may be named missing
        if nm.get("outcome") != NUMERICS_NOT_MEASURED:
            problems.append(f"numerics.mode {NUMERICS_NOT_MEASURED!r} but outcome {nm.get('outcome')!r}")
        stray = [r for r in (nm.get("reasons") or []) if str(r).startswith(("leg_missing:", "floor_unmeasured",
                                                                            "zero_adapter_unmeasured"))]
        if stray:
            problems.append(f"numerics.mode {NUMERICS_NOT_MEASURED!r} names missing legs as findings: {stray}")
    elif nm.get("outcome") == NUMERICS_NOT_MEASURED:
        problems.append(f"numerics.outcome {NUMERICS_NOT_MEASURED!r} under mode {mode!r}")
    rd = rec.get("readiness")
    if not isinstance(rd, Mapping):
        problems.append("readiness missing")
    else:
        problems += [f"readiness lacks {k}" for k in REQUIRED_READINESS_KEYS if k not in rd]
        if not isinstance(rd.get("attested"), bool):
            problems.append(f"readiness.attested {rd.get('attested')!r} is not a bool")
    gt = rec.get("gates")
    if not isinstance(gt, Mapping):
        problems.append("gates missing")
    else:
        problems += [f"gates lacks {k}" for k in REQUIRED_GATES_KEYS if k not in gt]
        held = [k for k in REQUIRED_GATES_KEYS if isinstance(gt.get(k), bool)]
        if len(held) == len(REQUIRED_GATES_KEYS):
            want = "admitted" if all(gt[k] for k in REQUIRED_GATES_KEYS) else "not_admitted"
            if rec.get("verdict") in VERDICTS and rec.get("verdict") != want:
                problems.append(f"verdict {rec.get('verdict')!r} does not follow the gates {dict((k, gt[k]) for k in REQUIRED_GATES_KEYS)}")
    return problems


def validate_admission_record(rec: Mapping[str, Any]) -> List[str]:
    """Schema problems of an ``admission.json`` (empty = valid), judged against
    the record's OWN ``schema_version``: an unknown schema refuses by name; a
    rule that is not one of that schema's (:data:`SCHEMA_RULES`) refuses — a
    schema-4 record claiming the functional rule, or a schema-5 record
    claiming the legacy rule; then the schema's required keys and
    vocabularies (schema 4: the legacy tables, frozen; schema 5: the
    functional tables, ``numerics.gate`` false and no numerics verdict,
    ``readiness`` and ``gates`` present and the verdict following the gates).
    The three historical schema-4 sidecars therefore keep validating beside
    the schema-5 records a new job writes."""
    if not isinstance(rec, Mapping):
        return [f"record is {type(rec).__name__}, not a JSON object"]
    sv = rec.get("schema_version")
    if isinstance(sv, bool) or not isinstance(sv, int) or sv not in SCHEMA_RULES:
        return [f"schema_version {sv!r} is not a known schema (known: {sorted(SCHEMA_RULES)})"]
    problems: List[str] = []
    rule = rec.get("rule")
    if rule not in SCHEMA_RULES[sv]:
        problems.append(f"rule {rule!r} is not a schema-{sv} rule (schema {sv} names {list(SCHEMA_RULES[sv])})")
    return problems + (_validate_v4(rec) if sv == 4 else _validate_v5(rec))


# --------------------------------------------------------------------------- #
# the leg loader — ONE for the job and the offline readmit                      #
# --------------------------------------------------------------------------- #

def read_leg_rows(unit_dir: Path, leg: str) -> Optional[List[Dict[str, Any]]]:
    """One probe leg's rows off ``unit/probes/<leg>.jsonl`` — only when its
    ``<leg>.DONE.json`` is beside it (a leg without its DONE was never
    completed and is ``None``, which the verdict names as ``leg_missing``).
    The driver's ``read_leg`` and :func:`readmit` both read through here.
    Lines split on ``\n`` only: ``str.splitlines`` also breaks on U+2028 /
    U+0085, which a JSON string may carry."""
    unit_dir = Path(unit_dir)
    path = unit_dir / "probes" / f"{leg}.jsonl"
    if not path.is_file() or not (unit_dir / "probes" / f"{leg}.DONE.json").is_file():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def admission_inputs_from_unit(unit_dir: Path, *, gate_canary: Mapping[str, Any], controls: Mapping[str, Any],
                               head_to_head: Mapping[str, Any], n_probes: int, max_tokens: int,
                               attempt_id: Optional[str], geometry: Optional[Mapping[str, Any]],
                               noop_certificate: Optional[Mapping[str, Any]] = None,
                               control_max_tokens: Optional[int] = None,
                               readiness: Optional[Mapping[str, Any]] = None,
                               numerics_mode: str = NUMERICS_DIAGNOSTIC) -> AdmissionInputs:
    """The :class:`AdmissionInputs` of a unit on disk: every leg of
    :data:`LEGS` that :func:`read_leg_rows` finds complete, plus the
    summaries the caller holds (``readiness`` = :func:`readiness_record`'s
    output; ``None`` fails closed in the verdict). The driver's ``finish_unit``
    and :func:`readmit` build their inputs through this one function, so an
    offline re-evaluation reads the legs exactly as the job did."""
    legs: Dict[str, List[Dict[str, Any]]] = {}
    for leg in LEGS:
        rows = read_leg_rows(unit_dir, leg)
        if rows is not None:
            legs[leg] = rows
    return AdmissionInputs(legs=legs, gate_canary=dict(gate_canary), controls=dict(controls),
                           head_to_head=dict(head_to_head), n_probes=int(n_probes), max_tokens=int(max_tokens),
                           attempt_id=attempt_id, geometry=(dict(geometry) if geometry is not None else None),
                           noop_certificate=(dict(noop_certificate) if isinstance(noop_certificate, Mapping) else None),
                           control_max_tokens=control_max_tokens,
                           readiness=(dict(readiness) if isinstance(readiness, Mapping) else None),
                           numerics_mode=str(numerics_mode))


# --------------------------------------------------------------------------- #
# the offline re-evaluation                                                     #
# --------------------------------------------------------------------------- #

#: gate_canary keys compared between the recorded summary and its recomputation
#: (``record`` is an absolute path that differs once the record is pulled off
#: the cluster; ``attempt`` is a nested copy of the same files)
_GATE_COMPARE_KEYS: Tuple[str, ...] = ("ran", "passed", "status", "lora_canary_passed", "noop_lora_passed",
                                       "noop_passthrough_passed", "multi_lora_checked", "margin_atol_nats",
                                       "reason", "bound", "cell")
#: the agreement fields the readmit holds the rebuilt legs to (integers: the
#: same probe rows give the same counts; the floats are derived from them)
_AGREEMENT_PIN_KEYS: Tuple[str, ...] = ("n", "sum_agree_prefix", "sum_denom", "n_diverging", "n_empty")


def resolve_unit_dir(path: Path) -> Path:
    """The unit a record path denotes: the path itself when it holds an
    ``admission.json``; a record dir's ``reused_unit/`` when its manifest
    names ``unit_reused_from`` (a reused record is read at its own bound copy,
    never at the source path); else the record dir's ``unit/``. Refuses a path
    with no admission under any of those."""
    p = Path(path)
    if (p / "admission.json").is_file():
        return p
    manifest: Any = None
    if (p / "manifest.json").is_file():
        try:
            manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            manifest = None
    if isinstance(manifest, Mapping) and manifest.get("unit_reused_from") and (p / REUSED_UNIT_DIR).is_dir():
        return p / REUSED_UNIT_DIR
    if (p / "unit" / "admission.json").is_file():
        return p / "unit"
    raise AdmissionError(f"{p}: no admission.json here, under unit/ or under {REUSED_UNIT_DIR}/")


def verify_unit_input_bytes(unit_dir: Path) -> Dict[str, Any]:
    """The BYTES of every numerical input a re-evaluation reads:
    each probe leg's ``probes/<leg>.jsonl`` hashes to its ``DONE.json``'s
    ``sha256_of_jsonl`` with the recorded ``n_rows``; ``INVENTORY.json`` is
    bound by ``DONE.json``'s ``inventory_sha256``; every inventoried file
    that is present hashes to its recorded digest; an inventoried file that
    is ABSENT is a problem unless it is a server log (``*.log``), which is
    reported under ``absent_diagnostic_logs`` — never as a pass. Returns
    ``{problems, legs: {leg: {sha256, n_rows}}, inventory_listed,
    inventory_verified, absent_diagnostic_logs, done_binds_inventory}``.
    Before this the readmit held only the marginal summaries: a re-allocation
    of per-prompt contributions that preserved every aggregate flipped a
    verdict and passed."""
    unit_dir = Path(unit_dir)
    out: Dict[str, Any] = {"problems": [], "legs": {}, "inventory_listed": 0, "inventory_verified": 0,
                           "absent_diagnostic_logs": [], "done_binds_inventory": None}
    problems: List[str] = out["problems"]
    for leg in LEGS:
        jsonl = unit_dir / "probes" / f"{leg}.jsonl"
        done_p = unit_dir / "probes" / f"{leg}.DONE.json"
        if not jsonl.is_file() and not done_p.is_file():
            continue                     # a leg never run: the verdict names it as leg_missing
        if not jsonl.is_file() or not done_p.is_file():
            problems.append(f"probes/{leg}: {'DONE.json' if jsonl.is_file() else 'jsonl'} present without its partner")
            continue
        try:
            done = json.loads(done_p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            problems.append(f"probes/{leg}.DONE.json unreadable: {e}")
            continue
        raw = jsonl.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        want = done.get("sha256_of_jsonl") if isinstance(done, Mapping) else None
        n_rows = sum(1 for line in raw.decode("utf-8", errors="replace").split("\n") if line.strip())
        if want != got:
            problems.append(f"probes/{leg}.jsonl sha256 {got[:12]} != DONE.json sha256_of_jsonl {str(want)[:12]}")
        if isinstance(done, Mapping) and done.get("n_rows") != n_rows:
            problems.append(f"probes/{leg}.jsonl has {n_rows} rows, DONE.json records {done.get('n_rows')!r}")
        out["legs"][leg] = {"sha256": got, "n_rows": n_rows}
    inv_p = unit_dir / UNIT_INVENTORY_FILE
    done_p = unit_dir / "DONE.json"
    if not inv_p.is_file():
        problems.append(f"no {UNIT_INVENTORY_FILE} under {unit_dir}: the numerical inputs are unbound")
        return out
    raw_inv = inv_p.read_bytes()
    try:
        inv = json.loads(raw_inv.decode("utf-8"))
        files = inv["files"]
        if not isinstance(files, Mapping):
            raise ValueError("no files map")
    except (ValueError, UnicodeDecodeError, KeyError) as e:
        problems.append(f"{UNIT_INVENTORY_FILE} is not an inventory document: {e}")
        return out
    if done_p.is_file():
        try:
            done = json.loads(done_p.read_text(encoding="utf-8"))
            binds = isinstance(done, Mapping) and done.get(DONE_INVENTORY_SHA_KEY) == hashlib.sha256(raw_inv).hexdigest()
        except (ValueError, OSError):
            binds = False
        out["done_binds_inventory"] = bool(binds)
        if not binds:
            problems.append(f"DONE.json {DONE_INVENTORY_SHA_KEY} does not bind the {UNIT_INVENTORY_FILE} on disk")
    else:
        problems.append("no DONE.json: the inventory is unbound")
    out["inventory_listed"] = len(files)
    for rel, meta in sorted(files.items()):
        p = unit_dir / rel
        want = meta.get("sha256") if isinstance(meta, Mapping) else None
        if not p.is_file():
            if rel.endswith(".log"):
                out["absent_diagnostic_logs"].append(rel)
            else:
                problems.append(f"inventoried input missing: {rel}")
            continue
        got = _sha256_file(p)
        if got != want:
            problems.append(f"{rel}: sha256 {got[:12]} != inventoried {str(want)[:12]}")
        else:
            out["inventory_verified"] += 1
    for leg in out["legs"]:
        for name in (f"probes/{leg}.jsonl", f"probes/{leg}.DONE.json"):
            if name not in files:
                problems.append(f"{name} is not inventoried")
    return out


#: the readiness keys the readmit holds a recorded (schema-5 job) readiness block to
_READINESS_COMPARE_KEYS: Tuple[str, ...] = ("attested", "kind", "engine", "adapters")


def load_readmission_inputs(unit_dir: Path, *, attestation_reader=None
                            ) -> Tuple[Optional[AdmissionInputs], Dict[str, Any], List[str]]:
    """Rebuild a unit's :class:`AdmissionInputs` from its records, the way the
    job built them, and cross-check each rebuilt input against what the job
    recorded: ``(inputs, provenance, problems)`` — ``inputs`` is ``None`` when
    a problem stops the rebuild; any problem means the unit on disk is not the
    one the job admitted and :func:`readmit` refuses.

    * the BYTES of every numerical input verify (:func:`verify_unit_input_bytes`);
    * the unit binds to ONE attempt (:func:`unit_binding_problems`);
    * ``controls`` := :func:`controls_summary` over ``controls_summary.json``'s
      verbatim ``positive_controls`` (E's record), held equal to the recorded
      ``summary`` on its verdicts; the raw ``noop_certificate`` record rides
      along as ``noop_certificate`` (the solo certificate: its scope is recorded
      and it decides ``zero_adapter_exact``);
    * ``gate_canary`` := the recorded summary (the object the job passed),
      held equal on every verdict key to a fresh :func:`gate_canary_summary`
      over ``unit/gate_canary/``;
    * ``readiness`` := :func:`readiness_from_unit` — the attestation on disk
      RE-READ through the strict reader (``attestation_reader`` = ``(read_fn,
      error_class)`` to inject one; else the package's), held equal on
      :data:`_READINESS_COMPARE_KEYS` to ``controls_summary.json``'s recorded
      ``readiness`` when the job recorded one (a schema-5 job; a schema-4 job
      recorded none — its readiness is what the file on disk says). A reader
      that cannot be imported is a PROBLEM (the re-evaluation refuses), never
      an unattested readiness;
    * ``head_to_head`` := ``head_to_head.json`` verbatim;
    * ``n_probes`` / ``max_tokens`` / ``attempt_id`` / ``geometry`` := the
      original ``admission.json``'s, held equal to ``identity.json``'s;
      ``control_max_tokens`` := ``identity.json``'s ``config.control_probe_max_tokens``;
    * the legs := :func:`admission_inputs_from_unit` (the job's loader)."""
    unit_dir = Path(unit_dir)
    problems: List[str] = []
    prov: Dict[str, Any] = {"unit_dir": str(unit_dir)}
    adm_path = unit_dir / "admission.json"
    if not adm_path.is_file():
        return None, prov, [f"no admission.json under {unit_dir}"]
    raw = adm_path.read_bytes()
    prov["original_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        original = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return None, prov, [f"admission.json unreadable: {e}"]
    if not isinstance(original, Mapping):
        return None, prov, ["admission.json is not a JSON object"]
    prov["original"] = dict(original)
    verified = verify_unit_input_bytes(unit_dir)
    prov["inputs_verified"] = {k: v for k, v in verified.items() if k != "problems"}
    problems += [f"input bytes: {p}" for p in verified["problems"]]
    problems += [f"unit binding: {p}" for p in unit_binding_problems(unit_dir)]

    def _doc(name: str) -> Optional[Mapping[str, Any]]:
        p = unit_dir / name
        if not p.is_file():
            problems.append(f"no {name} under {unit_dir}")
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            problems.append(f"{name} unreadable: {e}")
            return None
        if not isinstance(d, Mapping):
            problems.append(f"{name} is not a JSON object")
            return None
        return d

    cs = _doc("controls_summary.json")
    h2h = _doc("head_to_head.json")
    identity = _doc("identity.json")
    controls: Optional[Dict[str, Any]] = None
    gate: Optional[Dict[str, Any]] = None
    noop_cert: Optional[Mapping[str, Any]] = None
    if cs is not None:
        pc = cs.get("positive_controls")
        if pc is None:
            problems.append("controls_summary.json carries no positive_controls record")
        else:
            controls = controls_summary(pc)
            ctrls = pc.get("controls") if isinstance(pc, Mapping) else None
            if isinstance(ctrls, Mapping) and isinstance(ctrls.get(ZERO_ADAPTER_CONTROL), Mapping):
                noop_cert = ctrls[ZERO_ADAPTER_CONTROL]
            recorded = cs.get("summary")
            if isinstance(recorded, Mapping):
                for key in ("verdicts", "passed", "all_passed"):
                    if recorded.get(key) != controls.get(key):
                        problems.append(f"controls_summary.json summary.{key} {recorded.get(key)!r} != recomputed "
                                        f"{controls.get(key)!r}")
            else:
                problems.append("controls_summary.json carries no summary block to cross-check against")
        rec_gate = cs.get("gate_canary")
        if not isinstance(rec_gate, Mapping):
            problems.append("controls_summary.json carries no gate_canary summary")
        else:
            gate = dict(rec_gate)
            gdir = unit_dir / "gate_canary"
            fresh = gate_canary_summary(gdir if gdir.is_dir() else None, cell=str(rec_gate.get("cell")))
            diff = [k for k in _GATE_COMPARE_KEYS if fresh.get(k) != rec_gate.get(k)]
            if diff:
                problems.append("gate_canary recomputed from unit/gate_canary differs from the recorded summary on "
                                + ", ".join(f"{k}: {fresh.get(k)!r} != {rec_gate.get(k)!r}" for k in diff))
    if h2h is not None and "not_slower_at_S" not in h2h:
        problems.append("head_to_head.json carries no not_slower_at_S")
    # the READINESS input: the attestation on disk, re-read strictly (the evidence); a job that
    # recorded its readiness (schema 5) must agree with what the file says now
    readiness, env_problem = readiness_from_unit(unit_dir, reader=attestation_reader)
    if env_problem is not None:
        problems.append(f"readiness: {env_problem} — cannot re-evaluate without the strict reader")
    if cs is not None and isinstance(cs.get("readiness"), Mapping):
        rec_rd = cs["readiness"]
        diff = [k for k in _READINESS_COMPARE_KEYS if rec_rd.get(k) != readiness.get(k)]
        if diff:
            problems.append("readiness re-read from the attestation on disk differs from the recorded one on "
                            + ", ".join(f"{k}: {readiness.get(k)!r} != {rec_rd.get(k)!r}" for k in diff))
    prov["readiness"] = dict(readiness)
    n_probes = original.get("n_probes")
    max_tokens = original.get("max_tokens")
    attempt_id = original.get(ATTEMPT_ID_KEY)
    geometry = original.get("geometry")
    if isinstance(n_probes, bool) or not isinstance(n_probes, int):
        problems.append(f"admission.json n_probes {n_probes!r}")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        problems.append(f"admission.json max_tokens {max_tokens!r}")
    control_max_tokens: Optional[int] = None
    if identity is not None:
        for key, val in (("n_probes", n_probes), ("max_tokens", max_tokens), (ATTEMPT_ID_KEY, attempt_id)):
            if key in identity and identity.get(key) != val:
                problems.append(f"identity.json {key} {identity.get(key)!r} != admission.json {val!r}")
        cfg = identity.get("config")
        cmt = cfg.get("control_probe_max_tokens") if isinstance(cfg, Mapping) else None
        if isinstance(cmt, int) and not isinstance(cmt, bool):
            control_max_tokens = cmt
    if problems or controls is None or gate is None or h2h is None:
        return None, prov, problems
    # the numerics MODE is the ORIGINAL record's: a not_measured unit re-evaluates as
    # not_measured — its probe legs are absent by design, never "not complete on disk"
    mode = numerics_mode_of(original)
    inputs = admission_inputs_from_unit(unit_dir, gate_canary=gate, controls=controls, head_to_head=h2h,
                                        n_probes=int(n_probes), max_tokens=int(max_tokens),  # type: ignore[arg-type]
                                        attempt_id=(attempt_id if isinstance(attempt_id, str) else None),
                                        geometry=(geometry if isinstance(geometry, Mapping) else None),
                                        noop_certificate=noop_cert, control_max_tokens=control_max_tokens,
                                        readiness=readiness, numerics_mode=mode)
    if mode != NUMERICS_NOT_MEASURED:
        for leg in REQUIRED_LEGS:
            if leg not in inputs.legs:
                problems.append(f"leg {leg} is not complete on disk (probes/{leg}.jsonl + .DONE.json)")
    if problems:
        return None, prov, problems
    return inputs, prov, []


def _agreement_pin_mismatch(old: Mapping[str, Any], new: Mapping[str, Any]) -> List[str]:
    return [f"{k}: {old.get(k)!r} != {new.get(k)!r}" for k in _AGREEMENT_PIN_KEYS if old.get(k) != new.get(k)]


def _git_tree(module_path: Path, *, claimed_sha: Optional[str] = None) -> Dict[str, Any]:
    """The code tree running the readmit, BOUND to this module's bytes:
    ``sha`` (git HEAD, or ``claimed_sha`` when the caller names the tree),
    ``dirty`` (tracked changes under the worktree), ``module_sha256`` (this
    module's bytes — provenance that needs no git), ``module_sha256_at_tree``
    (the sha256 of ``git show <sha>:<module path>``) and ``module_matches_tree``
    (the two equal: the sha names the code that ran). When git is unavailable
    ``bind_skipped`` records why and the tree fields are ``None``."""
    out: Dict[str, Any] = {"sha": None, "dirty": None, "module_sha256": _sha256_file(module_path),
                           "module_sha256_at_tree": None, "module_matches_tree": None, "bind_skipped": None}
    cwd = str(module_path.parent)
    try:
        sha = claimed_sha or subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True,
                                            timeout=20, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", cwd, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        top = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             timeout=20, check=True).stdout.strip()
        rel = module_path.resolve().relative_to(Path(top).resolve()).as_posix()
        shown = subprocess.run(["git", "-C", cwd, "show", f"{sha}:{rel}"], capture_output=True, timeout=20, check=True)
        at_tree = hashlib.sha256(shown.stdout).hexdigest()
        out.update(sha=sha, dirty=bool(dirty), module_sha256_at_tree=at_tree,
                   module_matches_tree=(at_tree == out["module_sha256"]))
    except (OSError, ValueError, subprocess.SubprocessError) as e:  # pragma: no cover - environment
        out["bind_skipped"] = f"git unavailable or the tree could not be read: {type(e).__name__}: {e}"[:240]
        out.update(sha=claimed_sha, dirty=None)
    return out


def _readmit_provenance_problems(tree: Mapping[str, Any], *, allow_dirty: bool) -> List[str]:
    """Why a readmit may NOT record its tree provenance (P2a): a DIRTY
    tree without ``allow_dirty`` (the tree sha would not name the code that
    ran); a tree claimed clean whose committed module does not hash to the
    module on disk. A skipped bind (no git) is recorded, not a refusal."""
    problems: List[str] = []
    if tree.get("bind_skipped"):
        return problems
    if tree.get("dirty") and not allow_dirty:
        problems.append("the code tree is DIRTY (tracked changes): the tree sha would not name the code that "
                        "re-evaluated this record — commit first, or pass --allow-dirty to write a file MARKED dirty")
    if not tree.get("dirty") and tree.get("module_matches_tree") is False:
        problems.append(f"the tree reads clean at {str(tree.get('sha'))[:8]} but git show of this module there hashes to "
                        f"{str(tree.get('module_sha256_at_tree'))[:12]}, not the module on disk "
                        f"{str(tree.get('module_sha256'))[:12]} — the claimed tree is not the code that ran")
    return problems


def result_sidecar_line(new: Mapping[str, Any], *, original_verdict: Any, tree: Mapping[str, Any],
                        written_at: float) -> str:
    """The one-line ``RESULT.readmitted`` beside a record's ``RESULT``, in the
    driver's RESULT vocabulary: the rule, the functional verdict,
    the four gates (``readiness`` / ``controls`` / ``gate`` / ``not_slower``),
    the RECORDED three-way outcome (``numerics=diagnostic``), the solo zero
    certificate's verdict, the job's own verdict, and the code that
    re-evaluated (module sha12, tree sha8, dirty)."""
    za = new.get("zero_adapter") or {}
    zero = "exact" if za.get("exact") else ("inexact:" + ",".join(za.get("failed_instruments") or ["unmeasured"]))
    dirty = " dirty=true" if tree.get("dirty") else ""
    rd = new.get("readiness") or {}
    ctl = new.get("controls") or {}
    gc = new.get("gate_canary") or {}
    h2h = new.get("head_to_head") or {}
    nm = new.get("numerics") or {}
    return (f"rule={new.get('rule')} verdict={new.get('verdict')} "
            f"readiness={'attested' if rd.get('attested') is True else 'unattested'} "
            f"controls={ctl.get('passed', 0)}/{ctl.get('of', N_CONTROLS)} "
            f"gate={'passed' if gc.get('passed') else (gc.get('status') or 'missing')} "
            f"not_slower={h2h.get('not_slower_at_S')} "
            f"numerics={'diagnostic' if nm.get('gate') is False else nm.get('admitted')} outcome={nm.get('outcome')} "
            f"zero={zero} original_verdict={original_verdict} module={str(tree.get('module_sha256'))[:12]} "
            f"tree={str(tree.get('sha') or 'unknown')[:8]}{dirty} "
            f"written={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(written_at))}")


RESULT_SIDECAR = "RESULT.readmitted"


def rule_rank(rule: Any) -> Optional[int]:
    """A rule's position in :data:`RULE_ORDER` (``"D102" -> 0``, ``"D111" ->
    1``, the functional rule ``-> 2``); ``None`` for an UNKNOWN rule — which
    :func:`readmit` refuses to supersede or overwrite (the earlier ``D<n>``
    integer parse could not place ``functional-2026-09-13`` and would have
    read any ``D<big>`` as newer)."""
    if isinstance(rule, str) and rule.strip() in RULE_ORDER:
        return RULE_ORDER.index(rule.strip())
    return None


def reevaluate(path: Path, *, attestation_reader=None) -> Dict[str, Any]:
    """Re-evaluate one certification record under the current rule IN MEMORY
    — writes NOTHING. Verifies the BYTES of every numerical input, rebuilds
    the inputs through the job's loader (:func:`load_readmission_inputs`,
    the readiness re-read from the attestation on disk), recomputes
    :func:`admission_verdict` and holds the rebuilt legs to the original
    ``admission.json``'s agreement table and ulps, and the rebuilt gate /
    controls / head-to-head verdicts to the original's. Returns ``{new,
    original, unit_dir, original_sha256, inputs_verified, problems}``: ``new``
    is the schema-5 record (``None`` when the rebuild refused), ``problems``
    every reason the unit on disk is not the one the job admitted. This is
    how a historical record is PROVEN to read ``admitted`` under the
    functional rule without rewriting it; :func:`readmit` is this plus the
    sidecar write."""
    try:
        unit_dir = resolve_unit_dir(path)
    except AdmissionError as e:
        return {"new": None, "original": None, "unit_dir": None, "original_sha256": None,
                "inputs_verified": None, "problems": [str(e)]}
    inputs, prov, problems = load_readmission_inputs(unit_dir, attestation_reader=attestation_reader)
    out: Dict[str, Any] = {"new": None, "original": prov.get("original"), "unit_dir": str(unit_dir),
                           "original_sha256": prov.get("original_sha256"),
                           "inputs_verified": prov.get("inputs_verified"), "problems": problems}
    if inputs is None or problems:
        return out
    original: Mapping[str, Any] = prov["original"]
    try:
        new = admission_verdict(inputs)
    except AdmissionError as e:
        problems.append(f"admission refused: {e}")
        return out
    # the rebuilt legs must reproduce the original's table on every pair it recorded
    old_ag = original.get("agreement") if isinstance(original.get("agreement"), Mapping) else {}
    for pair, old in sorted(old_ag.items()):
        if pair in new["agreement"] and isinstance(old, Mapping):
            mism = _agreement_pin_mismatch(old, new["agreement"][pair])
            if mism:
                problems.append(f"agreement[{pair}] rebuilt from the legs on disk differs from the original: "
                                + "; ".join(mism))
    old_ul = original.get("first_token_ulps") if isinstance(original.get("first_token_ulps"), Mapping) else {}
    for pair, old in sorted(old_ul.items()):
        if pair in new["first_token_ulps"] and isinstance(old, Mapping):
            got = new["first_token_ulps"][pair]
            diff = [k for k in ("max", "p50", "n_present", "n_missing") if old.get(k) != got.get(k)]
            if diff:
                problems.append(f"first_token_ulps[{pair}] rebuilt differs from the original on {diff}")
    if (original.get("gate_canary") or {}).get("passed") != new["gate_canary"].get("passed"):
        problems.append("gate_canary.passed rebuilt differs from the original")
    if (original.get("controls") or {}).get("passed") != new["controls"].get("passed"):
        problems.append("controls.passed rebuilt differs from the original")
    if (original.get("head_to_head") or {}).get("not_slower_at_S") != new["head_to_head"].get("not_slower_at_S"):
        problems.append("head_to_head.not_slower_at_S rebuilt differs from the original")
    out["new"] = new
    return out


def readmit(path: Path, *, tree_sha: Optional[str] = None, allow_dirty: bool = False,
            attestation_reader=None) -> Dict[str, Any]:
    """The OFFLINE re-evaluation of one certification record under the
    current rule (:data:`ADMISSION_RULE`): :func:`reevaluate` (the bytes
    verified, the inputs rebuilt through the job's loader, the verdict
    recomputed and held to the original's table) and then write
    ``unit/admission.readmitted.json`` beside the job's own file — NEVER over
    it; ``admission.json`` and the ``DONE.json`` binding are untouched. An
    existing readmitted file under an OLDER rule (:data:`RULE_ORDER`) is
    overwritten and named under ``superseded``; one under a NEWER rule is
    never overwritten; one under an UNKNOWN rule refuses by name. The output
    carries ``readmitted_from``: the original's sha256, schema, rule, verdict,
    reasons and numerics, the verified input digests, the code tree that
    re-evaluated it, the rule and the time. PROVENANCE (P2a): the tree
    sha is recorded only when the tree is clean and ``git show
    <sha>:<module>`` hashes to the module that ran (``module_matches_tree``);
    a dirty tree refuses unless ``allow_dirty``, and then the file is MARKED
    (``tree_dirty: true``, a ``provenance_note``); ``tree_sha`` names a tree
    to bind against instead of HEAD. Beside the record's ``RESULT`` an atomic
    one-line ``RESULT.readmitted`` sidecar is written
    (:func:`result_sidecar_line`); the original line is preserved. Returns
    ``{written, sidecar, unit_dir, verdict, outcome, problems, ...}``;
    ``written`` is ``None`` on refusal with every problem named."""
    module_path = Path(__file__).resolve()
    ev = reevaluate(path, attestation_reader=attestation_reader)
    problems: List[str] = ev["problems"]
    result: Dict[str, Any] = {"written": None, "unit_dir": ev["unit_dir"], "problems": problems,
                              "original_sha256": ev.get("original_sha256"), "verdict": None, "outcome": None,
                              "inputs_verified": ev.get("inputs_verified"), "superseded": None}
    if ev["new"] is None or problems:
        return result
    new: Dict[str, Any] = ev["new"]
    original: Mapping[str, Any] = ev["original"]
    unit_dir = Path(ev["unit_dir"])
    # an existing readmit: supersede an older rule, never a newer one, never an unknown one
    out_path = unit_dir / READMITTED_ADMISSION_FILE
    superseded: Optional[str] = None
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev_rule = prev.get("rule") or (prev.get("readmitted_from") or {}).get("rule")
        except (ValueError, OSError, AttributeError):
            prev_rule = None
        prev_n, cur_n = rule_rank(prev_rule), rule_rank(ADMISSION_RULE)
        assert cur_n is not None
        if prev_n is None:
            problems.append(f"existing {READMITTED_ADMISSION_FILE} carries an unknown rule {prev_rule!r} (known: "
                            f"{list(RULE_ORDER)}); not overwriting")
        elif prev_n > cur_n:
            problems.append(f"existing {READMITTED_ADMISSION_FILE} carries rule {prev_rule} NEWER than "
                            f"{ADMISSION_RULE}; not overwriting")
        elif prev_rule != ADMISSION_RULE:
            superseded = str(prev_rule)
    tree = _git_tree(module_path, claimed_sha=tree_sha)
    problems += _readmit_provenance_problems(tree, allow_dirty=allow_dirty)
    result["tree"] = {k: tree.get(k) for k in ("sha", "dirty", "module_sha256", "module_matches_tree", "bind_skipped")}
    if problems:
        return result
    written_at = time.time()
    new["readmitted_from"] = {
        "admission_json": str(unit_dir / "admission.json"),
        "sha256_of_original_admission_json": ev["original_sha256"],
        "original_schema_version": original.get("schema_version"),
        "original_rule": original.get("rule", "point comparison (schema 2)"),
        "original_verdict": original.get("verdict"),
        "original_reasons": list(original.get("reasons") or []),
        "original_numerics": dict(original.get("numerics") or {}),
        "original_zero_adapter_exact": original.get("zero_adapter_exact"),
        "original_floor": dict(original.get("floor") or {}),
        "inputs_verified": ev.get("inputs_verified"),
        "tree_sha_of_readmit": tree.get("sha"),
        "tree_dirty": tree.get("dirty"),
        "module_sha256": tree.get("module_sha256"),
        "module_sha256_at_tree": tree.get("module_sha256_at_tree"),
        "module_matches_tree": tree.get("module_matches_tree"),
        "tree_bind_skipped": tree.get("bind_skipped"),
        "provenance_note": ("DIRTY TREE: written with --allow-dirty; the tree sha names the last commit, not "
                            "necessarily the code that ran (module_sha256 does)" if tree.get("dirty") else None),
        "rule": ADMISSION_RULE,
        "superseded": superseded,
        "readmitted_at": written_at,
        "note": ("offline re-evaluation under the current rule from the record's own legs and summaries, every "
                 "numerical input's bytes verified; admission.json is untouched and stays the job's authoritative "
                 "record — the in-job admission at the final tip supersedes this file"),
    }
    _write_json_atomic(out_path, new)
    # the RESULT sidecar beside the record's own RESULT (never over it)
    sidecar: Optional[str] = None
    record_dir = unit_dir.parent
    if (record_dir / "manifest.json").is_file() or (record_dir / "RESULT").is_file():
        line = result_sidecar_line(new, original_verdict=original.get("verdict"), tree=tree, written_at=written_at)
        sc_path = record_dir / RESULT_SIDECAR
        tmp = sc_path.with_name(f".{sc_path.name}.tmp-{os.getpid()}")
        tmp.write_text(line + "\n", encoding="utf-8")
        os.replace(tmp, sc_path)
        sidecar = str(sc_path)
    result.update(written=str(out_path), sidecar=sidecar, verdict=new["verdict"], outcome=new["numerics"]["outcome"],
                  reasons=new["reasons"], superseded=superseded, gates=new["gates"], readiness=new["readiness"],
                  original_verdict=original.get("verdict"), original_reasons=list(original.get("reasons") or []),
                  floor=new["floor"], sensitivity=new["numerics"]["prompt_resampling_sensitivity"],
                  within_floor=new["numerics"]["within_floor"],
                  primary_point_inequality=new["numerics"]["primary_point_inequality"],
                  zero_adapter_exact=new["zero_adapter_exact"], zero_adapter=new["zero_adapter"])
    return result


#: readmit exit codes
RC_READMIT_WRITTEN = 0
RC_READMIT_REFUSED = 2
RC_READMIT_NOT_ADMITTED = 3     # written, but the verdict is not `admitted` (only with --require-admitted)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="native-multi-LoRA admission: pure functions + the offline readmit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("readmit", help=f"re-evaluate one certification record under the current rule "
                                       f"({ADMISSION_RULE}); writes unit/{READMITTED_ADMISSION_FILE}, never touches "
                                       "admission.json; exit 0 = written, 2 = refused, 3 = written but not admitted "
                                       "(with --require-admitted); the last stdout line is `verdict=... outcome=...`. "
                                       "No margin option exists: the numerics are recorded, never a gate.")
    r.add_argument("record_dir", type=Path, help="a <target>_<jobid> record dir, or its unit/ dir")
    r.add_argument("--tree-sha", default=None, help="record this as tree_sha_of_readmit instead of git HEAD")
    r.add_argument("--require-admitted", action="store_true",
                   help="exit 3 when the re-evaluation was written but the verdict is not `admitted`")
    r.add_argument("--allow-dirty", action="store_true",
                   help="write even when the code tree has tracked changes; the file is MARKED tree_dirty "
                        "(default: refuse, because the tree sha would not name the code that ran)")
    a = ap.parse_args(argv)
    if a.cmd == "readmit":
        out = readmit(a.record_dir, tree_sha=a.tree_sha, allow_dirty=a.allow_dirty)
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
        print(f"verdict={out.get('verdict')} outcome={out.get('outcome')} written={out.get('written')}")
        if not out.get("written"):
            return RC_READMIT_REFUSED
        if a.require_admitted and out.get("verdict") != "admitted":
            return RC_READMIT_NOT_ADMITTED
        return RC_READMIT_WRITTEN
    return RC_READMIT_REFUSED  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
