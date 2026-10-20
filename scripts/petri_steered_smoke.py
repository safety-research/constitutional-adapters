#!/usr/bin/env python
"""First steered tokens through the vLLM lens serving path (petri-bloom
Phase-2; the first execution of lens_worker's GPU-VERIFY list).

WHY THIS EXISTS. ``lens_worker.SisteerWorkerExtension`` — the in-worker
steering wiring that serves the registry's hook ops per request — has
NEVER executed against a real vLLM engine. The 173012 probe verified the
vllm-lens x vllm 0.25.1+cu129 SUBSTRATE (plugin loads, hooks fire per
token, mixed batches isolate); this job is the first run of OUR wiring
on top of it: RealWorkerView's attribute surface, the collective_rpc
install/begin_run/finish_run glue, the split-residual write, the
lens_batch plan lattice + PrefillLedger, and the per-kind adapters
serving two REAL landed objects. If this run fails, the substrate is
already exonerated — the fault is this wiring or this script.

THE ARMS (real objects on shared storage, adapted by the committed
``lens_adapters`` — never a script-local translation):

- ``static_r_matched`` (kind static_r): the matched CAA vector, 10
  layers, site='pre', steer_positions='prefill_all', DEPLOYED AT
  lambda=0.05. The blob stores ``base_scale=1.0`` (its harness applies
  lambda at request time), and the registry path has exactly one scale
  slot — the object's own baked ``base_scale``, which the adapter reads
  into ``HookOp.coeff`` — so the parent rewrites ``base_scale`` to the
  deployment lambda in a DERIVED artifact (original + derived sha16s
  recorded; the tensors are untouched). Its resid_pre fit means the
  adapter serves every layer ONE BLOCK EARLIER — the install report's
  hook-layer set is asserted against the expected shifted set, so the
  index shift is validated end to end.
- ``alphasteer_official`` (kind alphasteer): prefill-only last-token
  matmul broadcast over prompt positions, 10 layers, site='pre', served
  at its stored ``base_scale`` (=1.0). Its metadata characterizes
  deployment at strengths -0.1..-0.5, so an optional ADVISORY arm
  (``--extra-alphasteer-scale``, default -0.3) serves the same object at
  a characterized strength via the same base_scale derivation. Advisory
  arms are excluded from the pass/fail steering_bites evidence — they
  inform, never gate.
- ``static_r_matched_response`` (kind static_r, ADVISORY): THE
  CROSS-PHASE CELL, and the first time POSITIONS_RESPONSE has ever
  executed on hardware. It is the static_r arm's OWN DERIVED BYTES
  (identical ``served_sha16`` — asserted, and both recorded) served
  under a resolved spec carrying ``positions: "response"``. Nothing
  else differs: same layers, same site shift, same deployment lambda.
  ``d1851eb`` made the served position mode come from the RESOLVED SPEC
  rather than the artifact precisely so an all-position artifact can be
  served under the response mask, and that combination IS the
  cross-phase cell; the intervention hash covers the resolved spec, so
  the same bytes at ``all`` and at ``response`` are correctly two
  distinct cells (both hashes recorded).

  Three facts are ASSERTED about its adapted ops, on CPU before any GPU
  spend, and recorded either way: every op carries
  ``positions == "response"``; every op carries a NON-EMPTY
  ``header_ids`` equal to the derived assistant header; and every op's
  phase is ``both``. The phase is the one that matters most — a
  prefill-only op would leave generation entirely unsteered while every
  other signal looked healthy, which is the specific silent failure
  this arm exists to exercise.

  It is ADVISORY, and that is enforced in FOUR places rather than
  asserted once, because a lens worker CONDEMNS ITSELF on any steering
  failure (``LensWorkerSteering._condemn``: every later call refuses),
  so a mask refusal inside the hook does not stay local to one request:
  (i) the CPU probe above drops the arm on a refusal or an unexpected
  op shape, with a loud recorded status and a ``!!`` banner, never a
  smoke_failed; (ii) every prompt the arm will be served — probe,
  filler, and each throughput set — is checked ON CPU to END with the
  assistant header ids, because the mask is LOCATED by matching those
  ids at prefill, so the likely cause is caught before the expensive
  path can fire and is reported as the template/render disagreement it
  is; (iii) it is served in an ENGINE OF ITS OWN (``--phase
  response``), so a condemnation costs those runs and nothing else;
  and (iv) it is kept OUT of the mixed batch, so the gate's own batch
  stays the exact 15-row composition that passed. Within its own
  engine it gets a full-set steered run, its scale-0 twin, the
  ALL-position arm served alongside for comparison, and the unsteered
  baseline — which yields its bite count, a scale-0-equals-unsteered
  count, a bite count against the ALL-position serving of the same
  bytes (identical there would mean the mask did nothing), its
  completions in an eyeball section judged on the same two axes, and
  its own throughput legs on both suites.

NEGATIVE CONTROL (CPU, cheap, recorded): ``alphasteer`` and
``cosine_gated_r`` must REFUSE a ``positions: "response"`` override —
both compute their edit from the LAST PROMPT TOKEN and broadcast it, so
re-masking relocates the edit rather than restricting it (``d1851eb``'s
DERIVATION LOCALITY rule). The harness adapts each under the override
and records that it raised and that the message names the rule. A
control that does NOT refuse is printed with a ``!!`` banner and
recorded; it does not fail the run, because it is a statement about
committed adapter code rather than about this serving run.

If either REQUIRED object fails to load or adapt, that is THE FINDING
(smoke_failed.json + exit nonzero naming the arm) — never a
substituted object.

MODEL PIN — meta-llama, not the NousResearch mirror (the arms' owner,
2026-08-21). The two repos ship byte-identical weights but DIFFERENT
chat templates (43 vs 18 prompt tokens on a plain user prompt). Both
arms steer EVERY prompt position, so a 25-token render difference is a
different intervention, and the objects were fit on meta-llama-rendered
activations. The manifest still pins NousResearch for the audit; the
divergence is recorded as an explicit model_repin note, and a mismatch
between the configured and RESOLVED (live engine config) model is a
loud refusal.

CHUNKED PREFILL — sized out, not switched off. vLLM V1 forces chunked
prefill on with no flag to disable it; every engine here is constructed
with ``max_num_batched_tokens = max_num_seqs * max_model_len`` so the
scheduler's token budget can NEVER bind mid-prompt and no prompt is
ever split. alphasteer's edit is computed once from the LAST prompt
token and broadcast to every prompt position — under a chunked prefill
the value needed to steer position 0 does not exist until the final
chunk — so this sizing is what makes that arm servable at all
(static_r is a fixed vector and chunk-safe either way; the two arms'
verdicts are reported separately). The RESOLVED scheduler values are
read back from the live config and a silent clamp is a refusal.

TWO GEOMETRIES, SAME GUARANTEE, SAME BUDGET. The fidelity gate and the
legacy throughput suite run on the CERTIFIED geometry — max_model_len
2048, max_num_seqs 32, so max_num_batched_tokens 65536 — byte-for-byte
what job 173580 passed under; nothing about the gate changed. The
shared-prefix throughput suite cannot fit a ~6500-token preamble in
2048, so it runs on a second geometry: max_model_len 8192,
max_num_seqs 8 (which is the consumer's own batch size), so
max_num_batched_tokens is 8 x 8192 = 65536 — THE SAME NUMBER. The
product is re-derived rather than reused, the no-chunking guarantee
holds by the identical argument on both, and vLLM's memory-profiling
dummy forward is the same size it was on the certified run, so raising
max_model_len does not quietly change how much KV cache the engine
gets. Which geometry a leg ran under is recorded per leg AND per
suite, so a reader never has to infer that two phases ran under
different scheduler settings.

WHAT THE RUN ESTABLISHES, in order of importance — evaluated by the
committed ``fidelity.evaluate_fidelity`` over exactly its documented
evidence schema (five checks), plus smoke-level extras:

1. STEERING BITES, per required arm (steered vs same-arm-at-scale-0).
   An arm that changes nothing is the HEADLINE failure — vllm-lens's
   documented failure mode is silently doing nothing — and is named
   per arm in the failure marker and the job log.
2. SCALE-0 NO-OP: the arm's donor-backed scale-0 twin is token-
   identical to no selection at all (armed but inert — a different
   claim from engine determinism; the two are separate checks).
3. ENGINE DETERMINISM: the identical mixed batch submitted twice is
   bit-identical. Its failure INVALIDATES everything else.
4. BATCH INVARIANCE + PER-REQUEST SELECTION: every request in one
   mixed batch (different requests selecting different objects, some
   none) equals what that same request produces in a SAME-SIZE
   reference batch whose other members are unsteered filler — see
   SAME-SIZE REFERENCES below, which is not a detail.

   SELECTION IS ALSO OBSERVED, NOT ONLY INFERRED. Comparing output
   token ids between compositions concludes that the right object was
   loaded because the output looks like that arm's output; that
   inference holds only while outputs are composition-stable, which is
   exactly what ``VLLM_BATCH_INVARIANT=1`` provides. Selection is a
   DISCRETE FACT, so the worker now records which object it resolved
   and bound per request (``lens_worker.SelectionRecord``, read over
   ``sisteer_selection_record``) and the harness compares that against
   the intended selection: ``selection_direct`` in the record, reported
   ALONGSIDE the output comparison and never instead of it. Direct
   passes + output fails = the right objects were loaded and the
   outputs moved numerically (expected at BI=0); direct FAILS = a real
   routing bug, outranking everything else; both pass = the strongest
   form of the claim. The worker record ships its own coverage
   statement: it observes routing and binding IDENTITY (cell id,
   expect_hash, effective_scale, and whether applying it changed the
   request's rows), NOT op-level mis-binding downstream of selection.
5. SENSIBLE OUTPUT: steered and unsteered completions are printed side
   by side in the job log (never in the JSON). Judged on TWO separate
   axes: static_r_matched at lambda=0.05 legitimately REFUSES ~21% of
   benign prompts (XSTest 0.788 vs 0.924 unsteered) — a refusal is the
   arm working; the bug signal is DEGENERATE text (repetition loops,
   word salad). Numeric fluency diagnostics (distinct-token fraction,
   max token run, max repeated 4-gram) go in the record.
6. THROUGHPUT — a SINGLE-FLAG LADDER, not a bundle, measured over two
   request geometries. Every leg records which policies were in force
   (prefix caching, CUDA graphs, batch invariance, shared preamble and
   its MEASURED token length) so no number has to be interpreted from
   its name, and a leg that cannot be measured is recorded as
   ``MISSING: <reason>`` — never omitted, never folded into another.

   WHY A LADDER. The earlier harness compared stock (graphs + prefix
   caching + VLLM_BATCH_INVARIANT=0) against eager + no-prefix-cache +
   batch-invariant, and reported the 3.1x between them. That is THREE
   policy changes in one step, so the number could not be attributed —
   and the three are recoverable by completely different means: prefix
   caching is prefill-side and could be restored with a per-request
   cache salt; batch invariance is an AUDIT requirement rather than a
   serving one and could simply be dropped in production; enforce_eager
   is decode-side and the plugin forces it, so it cannot be recovered
   at all. A consumer choosing between optimisation work and a
   two-server architecture needs to know which rung dominates. So the
   base comparators now walk from stock to the lens engine's own
   configuration ONE FLAG AT A TIME:

     base_graphs_prefix_cache        graphs, cache, BI=0   (stock)
     base_graphs_prefix_cache_bi     graphs, cache, BI=1   (- BI)
     base_eager_prefix_cache         eager,  cache, BI=1   (- graphs)
     base_eager_no_prefix_cache      eager,  no cache, BI=1
     lens_unselected / lens_steered  the lens engine itself

   Each adjacent pair differs in exactly one policy, so each policy's
   cost is a subtraction rather than an inference. (If batch invariance
   turns out to force eager internally, rung 2's own config assertion
   refuses and it is recorded MISSING with that reason — which is
   itself the finding that the two are inseparable.)

   WHY TWO REQUEST GEOMETRIES. The prefix-cache rung is only meaningful
   against a workload that HAS a reusable prefix. The legacy suite is 8
   short distinct prompts x 4 replicas at 128 generated tokens: almost
   nothing to reuse per distinct prompt, and prefill a large fraction
   of wall time, so its cache rung understates the cache in one
   direction and overstates it in the other. It is kept unchanged — it
   is the control, and it is what the passing run measured — and a
   second suite matches the consumer's real geometry:

     SHARED-PREFIX SUITE: batch 8, greedy, 256 generated tokens, a
     distinct ~150-token suffix per request, and a SHARED SYNTHETIC
     SYSTEM PREAMBLE at two sizes — ~6500 tokens (the expensive case)
     and ~90 tokens (the cheap case) — plus a matched NO-PREAMBLE
     control that differs from them in nothing else. The preamble is
     generated in this file from a fixed rule template with no
     randomness and no data file (:func:`build_preamble`); its MEASURED
     token length and the batch's measured longest-common-token-prefix
     are both recorded, so the numbers are attributable to what
     actually ran rather than to a nominal target. The warmup for these
     legs uses the SAME preamble with DIFFERENT suffixes, so a
     cache-enabled leg measures the steady state a long-lived server
     actually sees (system prompt cached, each request's own text new)
     rather than a full-prompt cache hit.

   The two suites run under DIFFERENT SCHEDULER GEOMETRY, deliberately
   and separately recorded (``throughput.geometries``): the certified
   one (max_model_len 2048) cannot hold a 6500-token preamble. See
   CHUNKED PREFILL below — the no-chunking guarantee is preserved on
   both, and ``max_num_batched_tokens`` is 65536 either way.

SAME-SIZE REFERENCES, NEVER SOLO ONES — the correction job 173152 paid
for. Checks 4 and 5 ask exactly one question: does a request's output
depend on WHO ELSE shares its batch? Batch SIZE is a different
variable, and on this stack it is not one the engine controls.
Measured in job 173152 and reproduced on a BARE vLLM 0.25.1 engine with
no steering code and vllm-lens not installed, ``VLLM_BATCH_INVARIANT=1``
does NOT deliver batch-size-1-versus-N bit-invariance on H200 (SM90):
sizes 6, 7, 8, 11, 15 and 16 agreed bit-exactly with each other, while
size 1 differed, flipping one near-tie token mid-generation. (The
flag's SM90 path installs cuBLAS workspace configs rather than
shape-pinned matmul kernels, so a residual kernel still varies with M;
a bf16 GEMM probe at the same settings reproduces it directly — row 0
of an Mx4096 @ 4096x14336 matmul is bit-equal across M in {6, 8, 11,
15, 16} and differs at M=1.) An earlier version of this harness
compared each mixed-batch row against a SOLO serving, and reported
cross-request contamination that was not there: the batched outputs
were bit-perfect — one of them bit-equalled a no-selector baseline
serving, which a contaminated row cannot do — and it was the batch-of-1
references that had moved. A false positive here accuses the serving
path of the exact failure this gate exists to detect, so it is
believed; it cost a full debugging cycle.

So every reference serving is a batch of the SAME SIZE as the mixed
batch, differing ONLY in composition: the row under test keeps its own
prompt, its own selection and its own row position, and all N-1
neighbours become UNSTEERED FILLER (``FILLER_PROMPTS``, fixed and
disjoint from the probe prompts, so a filler can never be read as a row
under test). Two further consequences, both load-bearing:
- A size mismatch between a reference and its mixed batch is a REFUSAL
  — in the worker as it serves, and again in the parent as it assembles
  evidence — never a number the reader has to notice. Both sizes are
  recorded (``batch_sizing`` in the record, ``reference_batches`` in
  the worker JSON) so the check is auditable without rerunning.
- The size-fixed batches pin EVERY row to exactly ``SMOKE_MAX_TOKENS``
  generated tokens (``ignore_eos``), because a batch that loses rows to
  EOS is a batch whose size shrinks mid-generation — the running batch
  would fall to small M in one composition and not the other, which is
  the variable being held fixed. The baseline/steered/scale-0 batches
  keep EOS so the eyeball section still judges real completions.
``REFERENCE_SIZING_NOTE`` states all of this in the JSON record itself,
so the next reader tempted to "simplify" the harness back to solo
servings finds the reason attached to the artifact and not only in a
commit message.

THE CERTIFIED CONFIGURATION IS THE PRODUCTION ONE — BI=0. The user
dropped ``serving_mode: exact_replay`` for the Petri audit (2026-08-27):
``VLLM_BATCH_INVARIANT=1`` costs 2.97x on the ladder — by far the
largest single term (enforce_eager is 1.14x, the prefix cache 1.33x at a
6.5k shared prefix and ~1.0 with none) — and two measured findings say
it was buying much less than its name: on H200/SM90 it does NOT deliver
batch-size-1-versus-N invariance at all (SAME-SIZE REFERENCES below), so
it buys SAME-SIZE composition invariance only, while an audit's live
batch size fluctuates continuously; and the consuming lane's chunking is
deterministic by construction with an aggregate-agreement bar rather
than a byte-identity one, so it needs no composition invariance. What
that lane DOES depend on is a different, weaker property:
REPEAT-DETERMINISM — same input, same composition, same order, run
twice, byte-identical output. BI governs invariance ACROSS compositions;
repeat-determinism lives WITHIN one, and ``engine_determinism`` is
exactly its test.

WHAT IS LOST, stated plainly (``EXACT_REPLAY_DROPPED_NOTE``, written
into the record, the marker, and the console): re-running the audit
will NOT reproduce its transcripts bit-for-bit — a request's output may
vary with the composition of the batch it happens to share. The user
accepted that trade knowingly; this harness records it rather than
quietly dropping the claim.

The manifest pins ``serving_mode: interleaved`` (whose composed launch
sets no VLLM_BATCH_INVARIANT), and THE GATE CERTIFIES WHAT PRODUCTION
RUNS, so the gating phases run at BI=0 and the BI=1 phase is now the
evidence-only probe — the exact inverse of the pre-2026-08-27 roles:

  ``bi0_certified``  the gate, at the PRODUCTION configuration
                     (serving_mode=interleaved). Governs the exit
                     status together with the PETRI-geometry phase.
  ``bi1_probe``      the same five checks at ``VLLM_BATCH_INVARIANT=1``
                     — the DROPPED exact_replay configuration, kept as
                     cheap evidence (the lens-engine BI-cost
                     comparison; the composition-invariance reference).
                     EVIDENCE ONLY: it writes no marker, cannot change
                     any verdict, and CANNOT make the job exit nonzero.

Both are recorded under ``fidelity_phases`` with their resolved
``VLLM_BATCH_INVARIANT``, all five verdicts, the per-check divergence
diagnostics (counts AND first-divergence indices — a repeat-determinism
failure at token 0 means something quite different from one at token
40), the direct selection half, and the lens engine's own throughput
legs at each setting, so the 2.97x is CONFIRMED on the lens engine
rather than carried over from the plugin-disabled ladder. Every phase
entry carries ``certifies``; the top-level ``fidelity`` key is, as it
always was, the gating phase's report and never the probe's.

WHAT GATES AT BI=0 — the waiver, and why it is not a loophole
(``split_gating_failures``). ``engine_determinism`` (the headline),
``scale0_noop`` and ``steering_bites`` are same-composition properties
production depends on: they gate as hard as ever, as does
``selection_direct`` (the worker's own routing record, composition-
independent) and every structural/instrument failure (missing evidence,
malformed evidence, unknown keys). ``batch_invariance`` and
``selection_honored``'s output half compare one request across
DIFFERENT compositions — exactly the property given up with
exact_replay — so at BI=0 their failures are WAIVED: still measured,
still reported, tagged ``expected_failures``/``waived_failures`` with
the reason attached (``BI0_EXPECTED_FAILURE_REASON``) in the record and
the console, but not job failures, because a gate must not block
production on a property nobody claims. A PASS of those checks is still
recorded as a pass (the 229597 BI=0 probe passed all five). A failure
of any OTHER check at BI=0 is a real finding and fails the job.

Flags: ``--fidelity-batch-invariant {0,1}`` sets the GATING phases
(default 0 — the production configuration; 1 marks every gating phase
``is_the_certified_configuration: false`` and certifies nothing);
``--fidelity-bi1-probe {on,skip}`` (default on) runs the BI=1 evidence
phase; ``--petri32-probe-utils`` (default skip) re-measures the 32-way
geometry — see THE 32-WAY RE-MEASURE below.

THE 32-WAY RE-MEASURE (``--petri32-probe-utils``, EVIDENCE ONLY).
32 x 16384 measurably did not fit one H200 at BI=1 (jobs 229194 +
229390 — KV coverage and warmup headroom are zero-sum in utilization,
and the window between under-coverage at 0.95 and the 0.48 GiB warmup
OOM at 0.97 is ~1% wide). BI=0 runs different kernels and may profile a
different peak, so that negative result does not transfer to the
setting production now runs: each requested utilization gets one
non-gating engine at 32 x 16384, its resolved KV coverage (or init
failure) recorded under ``petri_32way_probe``. A pass with real slack
moves PETRI_MAX_NUM_SEQS/DEFAULT_MAX_NUM_SEQS to 32 via a follow-up
commit and a fresh certification — never mid-job.

OUTCOME DISCIPLINE (petri_lens_probe's D13 pattern, verbatim):
- imports/GPU/engine-init failure → ``smoke_unavailable.json`` + exit 0
  (a launch blocker to escalate, not a verdict on the wiring);
- object load/adapt failure, config-assert failure, a raised request
  step, or ANY GATING fidelity failure (a BI=0-waived composition check
  is recorded, never gating — WHAT GATES AT BI=0 above) →
  ``smoke_failed.json`` + exit NONZERO naming every failure;
- success → ``smoke_record.json`` (numbers and verdicts only; token
  ids are numbers, generated TEXT never enters any JSON).
``smoke_evidence.json`` (children's raw gatherings + the fidelity
evidence + report) is always written.

ARCHITECTURE — one parent + twelve sequential children (plus one per
requested 32-way probe utilization), one engine per process (the
probe's clean-GPU-handoff pattern). A child never
outlives its engine, so a geometry change, a policy change, or an arm
that might condemn a worker is a new process rather than a mutated
one:
  smoke (parent)            CPU only: adapt-validates the real objects,
                            probes the response override (and the two
                            refusals that are its negative control),
                            builds + CPU-loads the worker registry,
                            derives the assistant header and checks
                            every prompt ends with it, launches
                            children, evaluates fidelity, writes the
                            outcome.
  worker-steered            the lens engine + SisteerWorkerExtension,
    --phase fidelity        certified geometry, CERTIFIED ARMS ONLY:
    (BI=0, GATING)          all fidelity evidence + the legacy lens
                            throughput legs. THE GATE, at the
                            production configuration.
  worker-steered            the same gate at the PETRI geometry the
    --phase fidelity        audit actually serves (PETRI_* constants),
    (BI=0, PETRI geometry,  KV coverage asserted off the worker's own
     GATING)                cache_config. GATES like the first.
  worker-steered x 0..N     the 32-way re-measure at the production BI
    --phase fidelity        (THE 32-WAY RE-MEASURE above), one engine
    (32 x 16384, EVIDENCE)  per requested utilization; never gates.
  worker-steered            the identical 2048-geometry child at
    --phase fidelity        VLLM_BATCH_INVARIANT=1 — the dropped
    (BI=1, EVIDENCE ONLY)   exact_replay configuration. Its own engine;
                            certifies nothing.
  worker-steered            the cross-phase cell alone, certified
    --phase response        geometry, in an engine of its own so a mask
                            refusal cannot condemn anything else.
  worker-base x 4           the single-flag ladder, certified geometry,
    --tp-suite legacy       plugin disabled (VLLM_LENS_DISABLE=1):
                            graphs/BI=0, graphs/BI=1, eager+cache,
                            eager+no-cache.
  worker-base x 3           the same three policy points at the
    --tp-suite shared_prefix  consumer geometry, over all three
                            preamble variants.
  worker-steered            the lens engine again, consumer geometry,
    --phase throughput      throughput only — no fidelity evidence is
                            gathered there, so the gate can never
                            depend on the new geometry.

CHAT-MARKERS FALLBACK (recorded, self-disabling). ``lens_worker``
preflight requires ``sisteer.chat_markers.derive_assistant_markers``,
which has not yet landed on this branch (it lives on tier0-mask-fix /
sweep-stage3-integration). When the real module is importable it is
used untouched; when absent, a shim is registered that derives the
header as the ``add_generation_prompt`` True/False render diff — which
is EXACTLY what the real module documents its ``header`` to equal on
the meta-llama Llama-3.1 template (verified locally: H=4,
``[128006, 78191, 128007, 271]``) — cross-checked over two probe
contents and refusing a zero-diff (the NousResearch trap) or a header
that does not decode through "assistant". ``chat_markers_shim: true``
in the record marks any run that used it.

PROMPTS are synthetic and harmless, defined inline, chat-templated and
pre-tokenized (the ledger is armed with the EXACT served token
lengths). Generated text goes to the job log only.

Run on the cluster via ``scripts/sbatch/petri_steered_smoke.sbatch``
from the immutable commit-addressed checkout. Results land under
``--out-dir`` (durable, on $SHARED).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # script-mode import path
    sys.path.insert(0, str(REPO_ROOT / "src"))

# --- smoke constants (pre-registered; part of what the record means) -----------

#: fixed, harmless, synthetic probe prompts (chat-templated per request).
PROMPTS = (
    "What is 17 times 24?",
    "Name three coastal cities and one interesting fact about each.",
    "Explain in two sentences why bread rises.",
    "Write a haiku about a lighthouse.",
    "List four everyday uses for a paperclip.",
    "What is the capital of Australia, and what is it known for?",
    "Describe briefly how a bicycle pump works.",
    "Give me a simple recipe for lemonade.",
)

#: UNSTEERED FILLER: the neighbours of a row under test in its same-size
#: reference batch. Never a row under test, never carrying a selection.
#: Chosen on four criteria, each doing work:
#:  (a) FIXED and pre-registered, exactly like PROMPTS — a reference
#:      batch's composition is part of what the record means, so it may
#:      not be sampled, shuffled or otherwise vary between runs;
#:  (b) DISJOINT from PROMPTS as a set (asserted at import, below), so a
#:      filler can never be confused with a row under test in any record
#:      or any log line — the two prompt pools share no string;
#:  (c) the SAME REGISTER and roughly the same rendered length as the
#:      probes (one short, harmless, self-contained instruction each), so
#:      a reference batch differs from the mixed batch in WHO the
#:      neighbours are rather than in how many tokens the prefill
#:      carries — composition is the variable, and nothing else should
#:      drift with it;
#:  (d) ENOUGH of them (16) to fill every position of the default 15-row
#:      mixed batch without repeating a prompt. A longer plan cycles,
#:      which is harmless (prefix caching is off and filler output is
#:      never evidence).
FILLER_PROMPTS = (
    "Name a common houseplant that tolerates low light.",
    "How many minutes are in three hours?",
    "Suggest a title for a short story about a train journey.",
    "What tool would you use to tighten a loose screw?",
    "Give one reason people keep a calendar.",
    "Describe the colour teal in one sentence.",
    "What is a good way to label moving boxes?",
    "Name two instruments in a string quartet.",
    "What does a thermometer measure?",
    "Suggest one topping for a baked potato.",
    "How would you explain a map legend to a child?",
    "Name a board game that uses dice.",
    "What is the plural of the word cactus?",
    "Give one tip for storing fresh herbs.",
    "What season follows spring?",
    "Name a unit used to measure distance.",
)

assert not (set(PROMPTS) & set(FILLER_PROMPTS)), (
    "filler prompts must be disjoint from the probe prompts: a filler "
    "that equals a probe could be read as a row under test")

#: Stated as a plain fact IN THE ARTIFACT (smoke_record.json and the
#: worker JSON), not only in a commit message — the next reader tempted
#: to "simplify" the harness back to solo servings must find the reason
#: attached to the thing they are reading.
REFERENCE_SIZING_NOTE = (
    "Reference servings are batches of the SAME SIZE as the mixed batch, "
    "differing only in composition: the request under test keeps its own "
    "prompt, selection and row position, and every other row is unsteered "
    "filler carrying no selection. BATCH-SIZE-1 (SOLO) REFERENCES ARE "
    "INVALID ON THIS ENGINE AND HARDWARE AND MUST NOT BE REINSTATED. "
    "Measured on vLLM 0.25.1 / H200 (SM90) in job 173152, and reproduced "
    "on a bare engine with no steering code and vllm-lens not installed: "
    "VLLM_BATCH_INVARIANT=1 does not deliver batch-size-1-versus-N "
    "bit-invariance. Batch sizes 6, 7, 8, 11, 15 and 16 agreed bit-exactly "
    "with each other; batch size 1 differed, flipping a single near-tie "
    "token mid-generation, because the flag's SM90 path installs cuBLAS "
    "workspace configs rather than shape-pinned matmul kernels and a "
    "residual kernel still varies with M. Comparing a batched row against "
    "a solo serving therefore reports cross-request contamination that is "
    "not there: it did exactly that in job 173152, whose batched outputs "
    "were bit-perfect, and cost a debugging cycle. For the same reason "
    "every row of a size-fixed batch is pinned to the full token budget "
    "with ignore_eos, so the running batch cannot shrink toward small M "
    "in one composition and not the other."
)

#: greedy budget for the fidelity probes. The baseline/steered/scale-0
#: batches let EOS stop them early (their completions are what the
#: eyeball section reads); equality comparisons handle unequal lengths
#: and every run generates >= 1 token. The SIZE-FIXED batches (the mixed
#: batch and its same-size references) instead pin every row to exactly
#: this many tokens — see SIZE_FIXED_IGNORE_EOS.
SMOKE_MAX_TOKENS = 48
#: the mixed batch and its references run with ignore_eos so the running
#: batch never shrinks mid-generation: a row that stops at EOS removes
#: itself from the batch, and the neighbours of a row under test differ
#: between the two compositions by construction, so their EOS timings do
#: too. Without this the two batches would drift to DIFFERENT sizes part
#: way through the very comparison that holds size fixed (module
#: docstring, SAME-SIZE REFERENCES; and see REFERENCE_SIZING_NOTE).
SIZE_FIXED_IGNORE_EOS = True
#: throughput legs: ignore_eos pins every request to exactly this many
#: generated tokens so tokens/s is comparable across configurations.
TP_MAX_TOKENS = 128
TP_REPLICAS = 4          # throughput request set = PROMPTS x replicas
TP_WARMUP_TOKENS = 8

#: engine geometry. max_num_batched_tokens = max_num_seqs * max_model_len
#: is LOAD-BEARING: vLLM V1 forces chunked prefill on with no disable
#: flag, and this sizing means the scheduler's budget can never bind
#: mid-prompt, so no prompt is ever split (module docstring: alphasteer
#: is unservable under a chunked prefill). Do not shrink it.
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 32
MAX_NUM_BATCHED_TOKENS = MAX_NUM_SEQS * MAX_MODEL_LEN

#: gpu_memory_utilization for every engine this smoke has always built
#: (jobs 173580/173664/227308 all ran at 0.85). The PETRI-geometry gate
#: below is the one engine that needs more; everything else keeps this.
SMOKE_GPU_MEMORY_UTILIZATION = 0.85

# --- the PETRI serving geometry (what the audit actually runs) ----------------
# The 2048/32 gate above certifies the serving path, but NOT the
# geometry the audit runs in: build_server_launch defaults
# max_model_len to 16384 and calls 16k a FLOOR (the CONSTITUTION_SI
# prefix alone is 5,872 tokens on the audit pin, the target-side turns
# carry no max_tokens, and the manifest's 15-turn budget projects a
# median constitution-cell conversation to ~13.4k tokens), with
# max_num_seqs defaulting to 24 for concurrent modes. Certifying the
# audit's concurrency only at 2048 leaves the production geometry
# uncertified — job 227308 did exactly that, and its own NOTE said the
# 8192/8 legs carried no fidelity evidence. This phase runs the SAME
# five-check gate at the geometry the audit will actually serve.
#
# WHY 24 AND NOT 32 — a MEASURED negative result (jobs 229194 + 229390,
# two H200 nodes): 32 x 16384 does not fit one H200 under the
# no-chunking identity. Fixed costs: weights ~15 GiB + full-coverage KV
# 32 x 16384 x 128 KiB/token (Llama-3.1-8B: 32 layers x 8 KV heads x
# 128 head_dim x 2 x bf16) = 64.0 GiB + the warmup peak of a forward
# over the whole 524,288-token budget, which includes one MLP
# intermediate of 524,288 x 14,336 x 2 B = 14.0 GiB — against a 139.81
# GiB card. Utilization is zero-sum against it (vLLM sizes KV from what
# the profiled peak leaves, so raising util grows KV and starves the
# warmup): 0.95 under-covered KV by 2,768 tokens (521,520 = 31.83x; the
# coverage gate below refused), 0.97 covered KV (544,432 = 33.23x) and
# then OOMed by 0.48 GiB on the 14.0 GiB warmup tensor. The window
# between is ~1% wide and failed on two different nodes — do not retry
# 32. At 24 the same numbers are KV 48.0 GiB + MLP 10.5 GiB + weights
# ~15 GiB: ~25 GiB of true slack.
#
# RE-TESTED AT BI=0 (job 230089, the --petri32-probe-utils re-measure,
# after exact_replay was dropped): SAME WALL at every utilization —
# 0.90 resolved KV 463,472 (28.29x), 0.95 resolved 520,736 (31.78x,
# 3,552 short; BI=1 gave 521,520 — essentially identical), 0.97 could
# not init the engine. Structural, not a batch-invariance kernel
# artifact; the question is CLOSED at both BI settings. 24 x 16384
# certified at BI=0 with resolved KV 574,064 = 35.04x (job 230089).
PETRI_MAX_MODEL_LEN = 16384
PETRI_MAX_NUM_SEQS = 24
PETRI_MAX_NUM_BATCHED_TOKENS = PETRI_MAX_NUM_SEQS * PETRI_MAX_MODEL_LEN
#: gpu_memory_utilization for the PETRI-geometry gate. NOT tuned to the
#: maximum that boots — 0.90 leaves genuine headroom on both sides: the
#: claim (0.90 x 139.81 = 125.8 GiB) hands KV everything after weights
#: and the profiled peak (comfortably above the 48.0 GiB the
#: 393,216-token budget needs for full 24-way coverage), and the
#: unclaimed ~14 GiB absorbs the post-KV warmup allocations that sank
#: the 32-way attempt at 0.97. max_num_seqs is still a scheduler
#: CEILING, not a KV grant — vLLM schedules fewer and preempts on a
#: shortfall instead of erroring — so the worker ASSERTS the resolved
#: coverage off its own cache_config rather than trusting this
#: arithmetic (kv_coverage_problem).
PETRI_GPU_MEMORY_UTILIZATION = 0.90

#: the 32-way RE-MEASURE at the production batch-invariance setting
#: (--petri32-probe-utils; EVIDENCE ONLY, never a gate). The negative
#: result above was taken at VLLM_BATCH_INVARIANT=1; BI=0 runs different
#: kernels (the flag's SM90 path installs cuBLAS workspace configs) and
#: may profile a different peak, so with the audit now serving BI=0
#: (exact_replay dropped, 2026-08-27) the "do not retry 32" conclusion
#: has to be re-taken at the setting production actually runs. Each
#: requested utilization gets its own engine; resolved KV coverage (or
#: the init failure) is recorded either way, and a pass here moves the
#: constants only via a follow-up commit + re-certification, never
#: mid-job.
#:
#: ANSWERED (job 230089, utils 0.90/0.95/0.97): NOT reachable at BI=0
#: either — same wall as BI=1 (0.95: 520,736 vs 521,520 tokens), so the
#: limit is structural. The flag stays for any future stack change that
#: plausibly moves the MEMORY profile (new vLLM pin, new model), but
#: the default is skip and the sbatch no longer passes it.
PETRI32_PROBE_SEQS = 32

# --- the shared-prefix throughput suite (the consumer's own geometry) ---------
# The legacy suite above cannot answer "what does the prefix cache buy
# me": 8 short distinct prompts have essentially nothing to reuse, so
# its cache rung measures an artifact of the probe rather than a fact
# about any workload. These constants are the consumer's actual serving
# geometry, so the legs come out in their cost model instead of as a
# ratio somebody has to translate.

#: requests per shared-prefix leg — the consumer's batch size, and also
#: this suite's max_num_seqs, which is what keeps the no-chunking
#: guarantee an identity rather than a property of the request set.
SP_BATCH = 8
#: generated tokens per request (their cap is 512; 256 is the working
#: point). Pinned with ignore_eos like every other throughput leg.
SP_GEN_TOKENS = 256
#: the DISTINCT per-request suffix, in tokens — the part a prefix cache
#: can never reuse. Approximate: built up to the first line that
#: reaches this count, and the MEASURED length is what is recorded.
SP_SUFFIX_TOKENS = 150
#: the SHARED system preamble, at two sizes. The pair is the point: the
#: contrast between them says how much of the cache rung is specific to
#: long-prefix conditions rather than general, which is otherwise an
#: estimate. ``None`` is the matched no-preamble control — same
#: suffixes, same batch, same token budget, nothing shared.
SP_PREAMBLE_TARGETS: Tuple[Tuple[str, Optional[int]], ...] = (
    ("nopreamble", None), ("preamble_short", 90), ("preamble_long", 6500))

#: the shared-prefix suite's scheduler geometry. max_model_len must hold
#: the largest preamble + its suffix + the full generation, with room
#: for the consumer's own 512-token cap:
#:   6500 + 200 + 512 = 7212 < 8192.
#: max_num_batched_tokens is RE-DERIVED from this suite's own
#: max_num_seqs x max_model_len (never inherited), so the no-chunking
#: guarantee holds by the identical argument as on the certified
#: geometry — and 8 x 8192 lands on exactly 65536, the same budget the
#: certified engines run, so vLLM's memory-profiling dummy forward is
#: unchanged and raising max_model_len does not quietly shrink the KV
#: cache.
SP_MAX_MODEL_LEN = 8192
SP_MAX_NUM_SEQS = SP_BATCH
SP_MAX_NUM_BATCHED_TOKENS = SP_MAX_NUM_SEQS * SP_MAX_MODEL_LEN
assert SP_MAX_NUM_BATCHED_TOKENS == MAX_NUM_BATCHED_TOKENS, (
    "the two geometries are deliberately sized to the SAME token budget "
    "so the engine's memory profile is unchanged between them; if that "
    "stops being true, say so in the record rather than silently "
    "re-profiling")

#: the seed sentence the synthetic preamble is grown from. Everything
#: else is generated by :func:`_preamble_line` from an index — no
#: randomness, no data file, no copied text, so the same preamble comes
#: out on every node and every run.
PREAMBLE_SEED_SENTENCE = (
    "You are a careful assistant working through a long list of house "
    "rules before you answer.")

#: the vocabulary the synthetic filler rotates through. Deliberately
#: dull, concrete and harmless: this text is only ever a token count.
_FILLER_LEXICON = (
    "amber", "basket", "candle", "driftwood", "envelope", "ferry",
    "granite", "harbour", "inkwell", "jigsaw", "kettle", "lantern",
    "mosaic", "notebook", "orchard", "pebble", "quilt", "ribbon",
    "sundial", "trellis", "umbrella", "velvet", "willow", "zither")
#: how the rules describe themselves — varied so the text is not one
#: line repeated, which some tokenizers and some caches treat specially.
_FILLER_QUALITIES = ("brief", "concrete", "orderly", "plain", "precise",
                     "calm", "literal", "tidy")

#: the model repin (the arms' owner, 2026-08-21): meta-llama, NOT the
#: manifest's NousResearch mirror — byte-identical weights, different
#: chat template, and these arms steer every prompt position.
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
#: where the two REAL objects live is the site's to say, never a default
#: here: these settings (src/sisteer/site.py setting: the environment, then
#: the site config; a value may spell $SHARED) answer when no
#: --static-r-object / --alphasteer-object is given, and with neither the
#: run refuses by name before anything is read.
STATIC_R_OBJECT_KEY = "SISTEER_PETRI_SMOKE_STATIC_R_OBJECT"
ALPHASTEER_OBJECT_KEY = "SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT"
DEFAULT_STATIC_R_SCALE = 0.05     # the deployment lambda (task brief)
DEFAULT_EXTRA_ALPHASTEER = "-0.3"  # advisory arm; "skip" disables
#: the cross-phase cell (module docstring): the SAME static_r bytes
#: served with positions='response'. Advisory; "skip" disables.
DEFAULT_RESPONSE_ARM = "on"
#: its cell id — the all-position arm's id plus a suffix, so the two are
#: adjacent in every listing and obviously the same object.
RESPONSE_ARM_SUFFIX = "_response"

UNAVAILABLE_MARKER = "smoke_unavailable.json"
FAILURE_MARKER = "smoke_failed.json"
SMOKE_RECORD = "smoke_record.json"
EVIDENCE_FILE = "smoke_evidence.json"

# --- the fidelity phases (module docstring, THE CERTIFIED CONFIGURATION) ------

#: the CERTIFIED phase's ``VLLM_BATCH_INVARIANT`` — "0", THE PRODUCTION
#: CONFIGURATION. The user dropped ``serving_mode: exact_replay`` for
#: the Petri audit (2026-08-27): its live strategy, batch_invariant
#: (``VLLM_BATCH_INVARIANT=1``), costs 2.97x on the ladder — the largest
#: single term by far — and on H200/SM90 it delivers only SAME-SIZE
#: composition invariance, never batch-size-1-versus-N (SAME-SIZE
#: REFERENCES above), while an audit's live batch size fluctuates
#: continuously. The manifest now pins ``serving_mode: interleaved``,
#: whose composed launch sets no VLLM_BATCH_INVARIANT at all, and THE
#: GATE MUST CERTIFY WHAT PRODUCTION RUNS — certifying BI=1 while the
#: audit serves BI=0 would be exactly the certified-but-not-run gap the
#: PETRI-geometry phase closed for geometry. The historical BI=1
#: certifications (jobs 173664, 227308, 229597) remain valid FOR THAT
#: configuration; they no longer describe production.
#: Anything else here means the gating phase is not the production
#: configuration, and the record says so in as many words rather than
#: leaving a reader to notice the env line.
CERTIFIED_BATCH_INVARIANT = "0"
#: the manifest serving mode the certified BI setting corresponds to —
#: tested against the committed manifest (test_petri_bi0_gating), so the
#: manifest and this gate cannot silently disagree about what
#: production runs.
CERTIFIED_SERVING_MODE = "interleaved"
#: WHAT WAS GIVEN UP, stated where every reader of the record will see
#: it (the record, the marker, and the console print all carry it).
EXACT_REPLAY_DROPPED_NOTE = (
    "EXACT REPLAY IS DROPPED (the user's decision, 2026-08-27): the audit "
    "serves serving_mode=interleaved with no VLLM_BATCH_INVARIANT, so "
    "RE-RUNNING THE AUDIT WILL NOT REPRODUCE ITS TRANSCRIPTS BIT-FOR-BIT — "
    "a request's output may vary with the composition of the batch it "
    "happens to share. That is a real reduction of the artifact's "
    "guarantee, accepted knowingly because what BI=1 actually bought was "
    "weaker than advertised: 2.97x throughput for SAME-SIZE composition "
    "invariance only (H200/SM90 never had batch-size-1-versus-N "
    "invariance, job 173152), on a serving path whose live batch size "
    "fluctuates continuously. What IS still certified at the production "
    "configuration: repeat-determinism (same batch submitted twice is "
    "byte-identical), scale-0 no-op, steering-bites per arm, and DIRECT "
    "selection routing (the worker's own record of which object it bound). "
    "Composition invariance is measured and reported but no longer claimed "
    "or gated.")
#: the two phase keys, fixed so a reader never has to work out which
#: block is which. Both appear in every record, even when the probe was
#: skipped (recorded MISSING with the reason — never omitted).
PHASE_CERTIFIED = "bi0_certified"
PHASE_BI1 = "bi1_probe"
#: the PETRI-geometry gate's phase key (a THIRD phase since the 16k
#: certification): the same five checks, the same arms, the same
#: VLLM_BATCH_INVARIANT as the gating phase, at the geometry the audit
#: actually serves (PETRI_MAX_MODEL_LEN x PETRI_MAX_NUM_SEQS) with its
#: KV coverage asserted. GATING: it fails the job exactly like the
#: certified phase does.
PHASE_PETRI = "bi0_certified_petri16k"
#: the checks whose failure at BI=0 is an EXPECTED RESULT, not a
#: regression. Both compare ONE REQUEST ACROSS DIFFERENT BATCH
#: COMPOSITIONS (the mixed batch against a same-size reference whose
#: other members are unsteered filler), which is precisely the property
#: VLLM_BATCH_INVARIANT=1 was buying. See BI0_EXPECTED_FAILURE_REASON.
#: Since the certified configuration IS BI=0 (exact_replay dropped),
#: these two are WAIVED in the gating phases: measured, reported, tagged
#: — never suppressed — but a failure of them cannot fail the job,
#: because composition invariance is not a property production claims
#: (EXACT_REPLAY_DROPPED_NOTE). selection_honored's DIRECT half
#: (selection_direct) is composition-independent and still gates.
BI0_EXPECTED_FAILURES = ("batch_invariance", "selection_honored")
BI0_EXPECTED_FAILURE_REASON = (
    "EXPECTED AT VLLM_BATCH_INVARIANT=0, NOT A REGRESSION. batch_invariance "
    "and selection_honored each compare one request's output across two "
    "DIFFERENT BATCH COMPOSITIONS — its row in the mixed batch against its "
    "row in a same-size reference batch whose other members are unsteered "
    "filler. Composition invariance is exactly what VLLM_BATCH_INVARIANT=1 "
    "provides on this stack, so removing it can move those outputs for "
    "purely numerical reasons with nothing wrong in the serving path. A "
    "failure of these two at BI=0 therefore says the flag was doing "
    "something, not that the serving path broke; the BI=1 phase in the same "
    "record is the verdict on the serving path. The checks that do NOT "
    "depend on composition — engine_determinism (the same batch submitted "
    "twice), scale0_noop and steering_bites (same-composition comparisons) "
    "— are expected to hold at BI=0, and a failure of any of THEM is a real "
    "finding. selection_honored additionally carries a DIRECT half "
    "(selection_direct) that does not depend on outputs at all, so when the "
    "output half fails here the direct half says whether routing was still "
    "correct.")
#: THE HEADLINE of every BI=0 phase — and now the gate's own headline,
#: since BI=0 is the certified production configuration.
BI0_HEADLINE_NOTE = (
    "REPEAT-DETERMINISM is the headline at BI=0, and it is a "
    "different and weaker property than batch invariance: same prompts, "
    "same batch composition, same order, submitted twice, byte-identical "
    "output. BI governs invariance ACROSS compositions; engine_determinism "
    "measures it WITHIN one. Production (serving_mode=interleaved, no "
    "VLLM_BATCH_INVARIANT) depends on repeat-determinism and NOT on "
    "composition invariance — the consuming lane's chunking is "
    "deterministic by construction and its bar is aggregate agreement "
    "with a reported divergence rate, not byte-identity — which is why "
    "engine_determinism GATES at BI=0 while batch_invariance is waived "
    "(EXACT_REPLAY_DROPPED_NOTE). If engine_determinism does NOT hold at "
    "BI=0, that is the headline finding and it fails the job: any "
    "downstream equivalence gate that compares generations byte-for-byte "
    "would be passing on luck.")

#: What ``selection_direct`` is, stated in the artifact. The direct half
#: is an OBSERVATION (lens_worker's SelectionRecord, read over its own
#: RPC); the output-comparison half is an INFERENCE. They answer
#: different questions and are both reported.
SELECTION_DIRECT_NOTE = (
    "THE DIRECT HALF of selection_honored, and it does not replace the "
    "output-comparison half — both are reported, because they answer "
    "different questions and disagreement between them is itself "
    "informative. The gate's own selection_honored check concludes the "
    "right object was loaded because the output looks like that arm's "
    "output; that inference is sound only while outputs are "
    "composition-stable, which is what VLLM_BATCH_INVARIANT=1 was "
    "providing. This half instead reads the worker's OWN record of which "
    "object it resolved and bound for each request, so it holds at either "
    "BI setting. Read the pair together: direct passes and output fails = "
    "the right objects were loaded and the outputs moved for numerical "
    "reasons (expected at BI=0); direct FAILS = a real routing bug, which "
    "outranks everything else in this record; both pass = the strongest "
    "form of the claim. The worker record ships its own `coverage` string "
    "stating what it does not cover — notably op-level mis-binding "
    "downstream of selection, which would read as a clean pass here. "
    "ONE UNVERIFIED ASSUMPTION, and it degrades loudly rather than "
    "silently: the PER-ROW half correlates a submitted row to the worker's "
    "record through RequestOutput.request_id, which should equal the id in "
    "the runner's input_batch but has never been checked on hardware. If it "
    "does not hold, every selected row comes back `unobserved` and the "
    "verdict is MISSING — never a pass — while the AGGREGATE half (every "
    "observation the worker made bound what was asked for, plus the set of "
    "cells actually resolved) still applies, because it needs no "
    "correlation at all.")

#: child exit codes — the parent's classification contract (probe's).
EXIT_ENGINE_INIT = 3   # engine could not come up   -> unavailable path
EXIT_REQUESTS = 4      # engine up, a step raised    -> failed path
CHILD_TIMEOUT_S = 2700

_SAFE_ID = re.compile(r"[A-Za-z0-9_.\-]{1,80}")


# --- pure helpers (CPU-exercised; no vllm imports) ------------------------------


def sha16(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()[:16]


def canonical_model_identity(resolved: str) -> Tuple[str, Optional[str]]:
    """``(repo_id, revision)`` for either a hub id or a local HF-cache
    snapshot path — so identity checks compare MODELS, not spellings.

    With a warm cache under HF_HUB_OFFLINE, vLLM resolves a hub id to
    its snapshot path (``.../models--Org--Name/snapshots/<sha>``);
    comparing that string against the configured id can never pass
    (jobs 255287/255288, 2026-08-30), while a genuinely different model
    or revision must still refuse. A plain hub id canonicalizes to
    itself with no revision claim. (HF cache dirs mangle ``/`` to
    ``--``; an org or name containing a literal ``--`` would be
    ambiguous here — none of this campaign's pins do.)"""
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})",
                  str(resolved))
    if m:
        return f"{m.group(1)}/{m.group(2)}", m.group(3)
    return str(resolved), None


def render_ids(tok: Any, messages: Sequence[Mapping[str, str]],
               add_generation_prompt: bool) -> List[int]:
    """``apply_chat_template`` normalized to list[int] (transformers 4.x
    list / 5.x BatchEncoding-or-nested — the real chat_markers module's
    own normalization)."""
    # honor the thinking-off pin (models.pin_reasoning_template_off),
    # exactly like the real chat_markers.render_ids: a thinking-ON render
    # on a reasoning template derives the WRONG assistant header and
    # false-fails every response-mask arm at prefill.
    out = tok.apply_chat_template(list(messages), tokenize=True,
                                  add_generation_prompt=add_generation_prompt,
                                  **getattr(tok, "_sisteer_template_kwargs",
                                            {}))
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(i) for i in out]


def derive_header_ids_fallback(tok: Any) -> Tuple[int, ...]:
    """The assistant header as the add_generation_prompt render diff —
    exactly what the real chat_markers module documents its ``header``
    to equal on the meta-llama Llama-3.1 template (its docstring: the
    marker-located header "match[es] the 4-token add_generation_prompt
    diff on the meta-llama template"). Refuses the NousResearch trap
    (zero diff: that template appends the header unconditionally), a
    non-prefix render pair, a content-dependent header, and a header
    that does not decode through "assistant"."""
    headers: List[Tuple[int, ...]] = []
    for content in ("Probe content one.", "A different probe, two?"):
        msgs = [{"role": "user", "content": content}]
        base = render_ids(tok, msgs, False)
        gen = render_ids(tok, msgs, True)
        if len(gen) <= len(base) or gen[:len(base)] != base:
            raise RuntimeError(
                "chat_markers fallback: the add_generation_prompt=True "
                "render is not a strict token extension of the =False "
                f"render ({len(base)} -> {len(gen)} tokens) — either the "
                "template ignores the flag (the NousResearch trap: the "
                "header is appended unconditionally, so the diff is 0) or "
                "the template is not header-terminal. The fallback cannot "
                "derive a header here; land sisteer.chat_markers instead")
        headers.append(tuple(gen[len(base):]))
    if headers[0] != headers[1]:
        raise RuntimeError(
            f"chat_markers fallback: header depends on message content "
            f"({len(headers[0])} vs {len(headers[1])} tokens) — not a "
            "content-independent assistant header; refusing")
    decoded = tok.decode(list(headers[0]))
    if "assistant" not in decoded:
        raise RuntimeError(
            "chat_markers fallback: the derived render diff does not "
            "decode through 'assistant' — not an assistant header; "
            "refusing")
    return headers[0]


def ensure_chat_markers() -> bool:
    """Use the real ``sisteer.chat_markers`` when importable; otherwise
    register the fallback shim (module docstring). Returns True iff the
    shim was installed — recorded in the smoke record, and printed, so
    no run silently depends on it. Registered in ``sys.modules`` AND as
    the package attribute so both import forms resolve; the in-process
    worker (VLLM_ENABLE_V1_MULTIPROCESSING=0) shares the interpreter,
    so lens_steer's ``from sisteer.chat_markers import ...`` inside
    preflight sees the same module."""
    try:
        import sisteer.chat_markers  # noqa: F401  (the real one wins)
        return False
    except ImportError:
        pass
    import sisteer

    mod = types.ModuleType("sisteer.chat_markers")
    mod.__doc__ = ("petri_steered_smoke's recorded fallback for the "
                   "not-yet-landed sisteer.chat_markers — header = the "
                   "add_generation_prompt render diff (see "
                   "derive_header_ids_fallback)")

    def derive_assistant_markers(tok: Any) -> Any:
        return types.SimpleNamespace(header=derive_header_ids_fallback(tok))

    mod.derive_assistant_markers = derive_assistant_markers

    def generation_header_ids(tok: Any, markers: Any = None) -> Any:
        # the fallback's header IS the add_generation_prompt render diff,
        # i.e. exactly the generation-prompt convention.
        return tuple(derive_header_ids_fallback(tok))

    mod.generation_header_ids = generation_header_ids
    sys.modules["sisteer.chat_markers"] = mod
    setattr(sisteer, "chat_markers", mod)
    return True


def scale0_cell_id(cell_id: str) -> str:
    return f"{cell_id}__scale0"


def production_install_check(registry_dir: Any, rpc_results: Any
                             ) -> Dict[str, Any]:
    """Validate a ``sisteer_install_steering`` RPC result the way the
    PRODUCTION serving path does — ``vllm_server.expected_install_state``
    (the launcher's independent reading of the registry) held against
    EVERY worker's report by ``vllm_server.validate_install_reports`` —
    and return ``{"expected": ..., "reports": [...]}``. Raises the same
    ``VllmServerError`` readiness raises on any disagreement.

    THE smokes' install gate, on purpose. The smokes used to assert a
    bespoke subset of the report here, and the LoRA one went green (job
    232307) while the wave's real readiness comparison refused the very
    same install (job 232410: ``worker 0: layers = [], expected
    [0..31]``) — a smoke that validates anything other than the
    production comparison can pass while production is broken, which is
    worse than no smoke. Bespoke assertions may ride ON TOP of this
    check; they must never replace it."""
    from sisteer.capeval.serving.vllm_server import (
        expected_install_state, validate_install_reports)
    expected = expected_install_state(Path(registry_dir))
    results = (list(rpc_results) if isinstance(rpc_results, (list, tuple))
               else rpc_results)
    reports = validate_install_reports(results, expected)
    return {"expected": expected, "reports": reports}


def response_arm_cell_id(cell_id: str) -> str:
    return f"{cell_id}{RESPONSE_ARM_SUFFIX}"


def header_tail_report(prompt_ids: Sequence[Sequence[int]],
                       header_ids: Sequence[int], label: str
                       ) -> Dict[str, Any]:
    """Does every prompt END with the assistant header's exact token
    ids? That is the precondition the response mask is LOCATED by: the
    hook matches these ids against the request's own tokens at prefill
    and REFUSES on any mismatch, so a template/render disagreement
    surfaces as an error rather than as a wrong mask.

    Checking it HERE, on CPU, turns that refusal into a named
    configuration finding reported before any GPU spend, instead of an
    opaque mid-run exception that reads as a generic arm failure. Pure;
    returns a report and never raises."""
    hdr = tuple(int(t) for t in header_ids)
    bad: List[Dict[str, Any]] = []
    for i, row in enumerate(prompt_ids):
        row = [int(t) for t in row]
        if len(row) < len(hdr) or tuple(row[len(row) - len(hdr):]) != hdr:
            bad.append({"index": i, "n_prompt_tokens": len(row),
                        "tail": row[-len(hdr):] if hdr else []})
    return {
        "label": label, "n_prompts": len(prompt_ids),
        "header_ids": list(hdr), "n_header_ids": len(hdr),
        "all_end_with_header": not bad and bool(hdr),
        "n_mismatched": len(bad), "mismatches": bad[:8],
    }


#: HARD FLOOR of the mixed-batch check (2026-08-30): the accepted passing
#: precedent (Llama gate 245237) ran 11 rows / 5 distinct selections, and
#: anything below that is a weaker check wearing the same name. These are
#: CONSTANTS of the check itself, deliberately not launch parameters —
#: weakening the floor requires editing this file, never a flag. The
#: geometry-honest gates (45985/45986) proved the converse failure mode:
#: a too-big plan splits across scheduler steps and the size-fixed checks
#: are undefined, which is the existing too-big refusal below.
MIXED_MIN_ROWS = 11
MIXED_MIN_SELECTIONS = 5


def validate_mixed_plan(plan: Sequence[Tuple[int, Optional[str]]],
                        max_num_seqs: int,
                        arm_hook_sites: Optional[Mapping[str, Sequence[str]]]
                        = None,
                        all_arm_ids: Optional[Sequence[str]] = None) -> None:
    """The mixed-batch check's own admission rules — ceiling AND floor.

    - CEILING: the plan must fit ``max_num_seqs`` in ONE scheduler step
      (else neither it nor its same-size references have the fixed size
      the checks hold constant).
    - FLOOR: >= :data:`MIXED_MIN_ROWS` rows and
      >= :data:`MIXED_MIN_SELECTIONS` distinct selections (arms + twins +
      unselected), so a subset flag can never quietly shrink the check
      below the accepted precedent.
    - SITE COVERAGE: when ``arm_hook_sites`` shows the full canary set
      (``all_arm_ids``) spans more than one hook site (block / oproj),
      the MIXED arms must span them too — a single-site subset would
      stop exercising cross-site isolation while both sites serve.
    Refusals are loud; there is no truncation path."""
    n = len(plan)
    if n > int(max_num_seqs):
        raise RuntimeError(
            f"the mixed batch carries {n} rows but max_num_seqs "
            f"resolved to {max_num_seqs} — the scheduler would split it "
            "across steps, so neither it nor its references would have "
            "the single fixed size these checks hold constant; refusing")
    if n < MIXED_MIN_ROWS:
        raise RuntimeError(
            f"the mixed batch carries {n} rows, below the check's own "
            f"floor of {MIXED_MIN_ROWS} (the accepted precedent's size) — "
            "a smaller plan is a weaker check wearing the same name; "
            "refusing (edit MIXED_MIN_ROWS deliberately, never a flag)")
    selections = {sel for _, sel in plan}
    if len(selections) < MIXED_MIN_SELECTIONS:
        raise RuntimeError(
            f"the mixed batch carries {len(selections)} distinct "
            f"selections, below the check's own floor of "
            f"{MIXED_MIN_SELECTIONS} — refusing (edit "
            "MIXED_MIN_SELECTIONS deliberately, never a flag)")
    if arm_hook_sites:
        mixed_arms = {sel for _, sel in plan
                      if sel is not None and sel in arm_hook_sites}
        all_sites = {site for a in (all_arm_ids or arm_hook_sites)
                     for site in arm_hook_sites.get(a, ())}
        mixed_sites = {site for a in mixed_arms
                       for site in arm_hook_sites.get(a, ())}
        if len(all_sites) > 1 and mixed_sites != all_sites:
            raise RuntimeError(
                f"the canary set spans hook sites {sorted(all_sites)} but "
                f"the mixed batch's arms span only {sorted(mixed_sites)} — "
                "a single-site mixed batch stops exercising cross-site "
                "isolation while both sites serve; refusing")


def build_mixed_plan(arm_ids: Sequence[str], n_prompts: int
                     ) -> List[Tuple[int, Optional[str]]]:
    """The ONE mixed batch: (prompt_index, selection cell id or None)
    rows mixing every arm, its scale-0 twin, and unselected requests.
    Prompt 0 is served under None AND every arm, so selection_honored's
    distinguishability refusal has a shared prompt whose REFERENCE
    outputs separate selections (fidelity.py's documented
    requirement)."""
    if not arm_ids:
        raise ValueError("build_mixed_plan needs at least one arm")
    if n_prompts < 8:
        raise ValueError(f"build_mixed_plan needs >= 8 prompts, got "
                         f"{n_prompts}")
    plan: List[Tuple[int, Optional[str]]] = [
        (0, None), (3 % n_prompts, None), (7 % n_prompts, None)]
    for i, arm in enumerate(arm_ids):
        plan.append((0, arm))
        plan.append(((1 + i) % n_prompts, arm))
        plan.append(((4 + i) % n_prompts, arm))
        plan.append(((2 + i) % n_prompts, scale0_cell_id(arm)))
    return plan


def build_reference_rows(plan: Sequence[Tuple[int, Optional[str]]],
                         index: int, n_probe_prompts: int,
                         n_filler_prompts: int
                         ) -> List[Tuple[int, Optional[str]]]:
    """The SAME-SIZE reference batch for row ``index`` of an N-row mixed
    ``plan``: N rows, of which row ``index`` is the row under test —
    unchanged prompt, unchanged selection, unchanged POSITION — and
    every other row is an UNSTEERED FILLER request (selection None,
    prompt drawn from ``FILLER_PROMPTS``, whose ids live at
    ``n_probe_prompts + k`` in the served prompt table).

    Position j always carries filler ``j % n_filler_prompts``, so the N
    reference batches differ from one another only in which position
    holds the probe, and the reference differs from the mixed batch only
    in WHO the neighbours are — never in how many of them there are, and
    never in where the row under test sits. That is the whole point: see
    the module docstring's SAME-SIZE REFERENCES, and never replace this
    with a solo serving."""
    n = len(plan)
    if not 0 <= index < n:
        raise ValueError(f"reference row index {index} out of range for a "
                         f"{n}-row mixed plan")
    if n_probe_prompts < 1 or n_filler_prompts < 1:
        raise ValueError(
            f"build_reference_rows needs >= 1 probe prompt and >= 1 filler "
            f"prompt, got {n_probe_prompts} and {n_filler_prompts} — with "
            "no filler there is nothing to hold the batch size at N")
    rows: List[Tuple[int, Optional[str]]] = []
    for j in range(n):
        if j == index:
            rows.append((int(plan[j][0]), plan[j][1]))
        else:
            rows.append((n_probe_prompts + (j % n_filler_prompts), None))
    return rows


def fluency_stats(tokens: Sequence[int]) -> Dict[str, Any]:
    """Numeric degeneracy diagnostics for one completion (numbers only;
    the text itself never enters a JSON). High max_token_run or
    max_repeated_4gram with low distinct_frac = a repetition loop."""
    toks = [int(t) for t in tokens]
    n = len(toks)
    if n == 0:
        return {"n_tokens": 0, "distinct_frac": 0.0, "max_token_run": 0,
                "max_repeated_4gram": 0}
    longest = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    grams: Dict[Tuple[int, ...], int] = {}
    for i in range(max(0, n - 3)):
        g = tuple(toks[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    return {"n_tokens": n,
            "distinct_frac": round(len(set(toks)) / n, 4),
            "max_token_run": longest,
            "max_repeated_4gram": max(grams.values()) if grams else 0}


# --- the shared-prefix throughput suite: synthetic text, built here -----------
# Generated from an index, never sampled and never pasted: the same
# bytes come out on every node, on every run, with no data file to keep
# in step with the script. The text is only ever a token count — it is
# never read, never judged, and never enters the eyeball section.


def _preamble_line(i: int) -> str:
    """Rule line ``i`` of the synthetic shared preamble. Deterministic
    in ``i`` alone; the three rotations are coprime-ish strides through
    the lexicon so consecutive lines do not repeat and the whole thing
    is not one line duplicated."""
    n, q = len(_FILLER_LEXICON), len(_FILLER_QUALITIES)
    return (f"Rule {i + 1:04d}: when the topic is "
            f"{_FILLER_LEXICON[(i * 7 + 3) % n]}, keep the answer "
            f"{_FILLER_QUALITIES[(i * 5 + 1) % q]}, and do not let it "
            f"drift toward {_FILLER_LEXICON[(i * 11 + 5) % n]}.")


def _suffix_line(request: int, k: int) -> str:
    """Context line ``k`` of request ``request``'s DISTINCT suffix. The
    request index enters every rotation, so two requests never share a
    line — the suffix is the part a prefix cache must not be able to
    reuse, and the prompt sets assert pairwise distinctness."""
    n = len(_FILLER_LEXICON)
    return (f"Context {request + 1}.{k + 1}: the "
            f"{_FILLER_LEXICON[(request * 17 + k * 5 + 1) % n]} on the "
            f"shelf sits beside the "
            f"{_FILLER_LEXICON[(request * 23 + k * 3 + 7) % n]}, which "
            "matters only for keeping this note the length it is.")


def _n_tokens(tok: Any, text: str) -> int:
    """Token count of raw text under the served tokenizer, with no
    special tokens — the chat template adds those.

    Normalized across the tokenizer surfaces this runs against: the
    parent holds an HF ``PreTrainedTokenizer``, the children hold
    whatever ``llm.get_tokenizer()`` returns, and a fast tokenizer's
    ``encode`` can hand back an ``Encoding`` rather than a list."""
    try:
        out = tok.encode(text, add_special_tokens=False)
    except TypeError:
        out = tok.encode(text)
    if hasattr(out, "ids"):          # tokenizers.Encoding
        out = out.ids
    if hasattr(out, "keys"):         # BatchEncoding
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return len(out)


def _grow_to_tokens(tok: Any, line_of: Any, target: int, first: str = "",
                    limit: int = 40000) -> Tuple[str, int]:
    """Join generated lines until the text reaches ``target`` tokens,
    then trim back to the shortest line count that still reaches it.

    Grown in blocks and trimmed line by line so the tokenizer is called
    tens of times rather than once per line — the 6500-token preamble
    would otherwise cost hundreds of full re-tokenizations. Returns the
    text and its MEASURED token length, which is what gets recorded:
    the target is a request, the measurement is the fact."""
    lines: List[str] = [first] if first else []
    i = 0
    n = _n_tokens(tok, "\n".join(lines)) if lines else 0
    while n < target:
        for _ in range(16):
            lines.append(line_of(i))
            i += 1
        if len(lines) > limit:
            raise SmokeConfigError(
                f"synthetic filler did not reach {target} tokens within "
                f"{limit} lines — refusing to spin")
        n = _n_tokens(tok, "\n".join(lines))
    while len(lines) > 1 and _n_tokens(tok, "\n".join(lines[:-1])) >= target:
        lines.pop()
    text = "\n".join(lines)
    return text, _n_tokens(tok, text)


def build_preamble(tok: Any, target_tokens: int) -> Tuple[str, int]:
    """The SHARED synthetic system preamble, grown to ``target_tokens``
    under the served tokenizer. Identical for every request in the leg —
    that is the whole point — and its measured length is recorded."""
    return _grow_to_tokens(tok, _preamble_line, target_tokens,
                           first=PREAMBLE_SEED_SENTENCE)


def build_sp_suffix(tok: Any, request: int) -> Tuple[str, int]:
    """Request ``request``'s DISTINCT suffix: ~``SP_SUFFIX_TOKENS`` of
    context lines that belong to this request alone, then one of the
    ordinary probe prompts as the actual instruction, so the completion
    is a real answer rather than a continuation of filler."""
    body, _ = _grow_to_tokens(
        tok, lambda k: _suffix_line(request, k), SP_SUFFIX_TOKENS)
    text = f"{body}\n\n{PROMPTS[request % len(PROMPTS)]}"
    return text, _n_tokens(tok, text)


def common_prefix_len(rows: Sequence[Sequence[int]]) -> int:
    """The batch's longest common TOKEN prefix — the part a prefix cache
    can actually reuse. Measured rather than assumed: a shared preamble
    only becomes a shared token prefix if the render puts it first and
    the tokenizer does not merge across the boundary."""
    if not rows:
        return 0
    first = list(rows[0])
    n = len(first)
    for row in rows[1:]:
        row = list(row)
        n = min(n, len(row))
        k = 0
        while k < n and row[k] == first[k]:
            k += 1
        n = k
        if n == 0:
            break
    return n


def build_sp_prompt_sets(tok: Any) -> Dict[str, Dict[str, Any]]:
    """The three shared-prefix prompt sets — no preamble, ~90-token
    preamble, ~6500-token preamble — each a batch of ``SP_BATCH``
    requests differing ONLY in whether (and how much) system preamble
    precedes the same distinct suffixes.

    A separate WARMUP batch per set carries the SAME preamble and
    DIFFERENT suffixes, so a cache-enabled leg measures the steady state
    a long-lived server sees (system prompt already cached, each
    request's own text new) instead of a full-prompt cache hit, which
    would flatter the cache rung into meaninglessness.

    Refuses rather than reports if the geometry does not hold: prompts
    must fit ``SP_MAX_MODEL_LEN`` with their generation, the batch's
    total prompt tokens must fit ``SP_MAX_NUM_BATCHED_TOKENS`` (the
    no-chunking guarantee, checked against the request set that will
    actually be submitted), and the suffixes must be pairwise
    distinct."""
    measured = [build_sp_suffix(tok, i) for i in range(SP_BATCH * 2)]
    suffixes = [t for t, _ in measured[:SP_BATCH]]
    warmups = [t for t, _ in measured[SP_BATCH:]]
    if len(set(suffixes)) != SP_BATCH:
        raise SmokeConfigError(
            "the shared-prefix suffixes are not pairwise distinct — the "
            "suffix is the part a prefix cache must NOT be able to reuse; "
            "refusing to measure a cache leg on duplicated text")

    def render(preamble: Optional[str], body: str) -> List[int]:
        msgs = ([] if preamble is None
                else [{"role": "system", "content": preamble}])
        return render_ids(tok, msgs + [{"role": "user", "content": body}],
                          True)

    sets: Dict[str, Dict[str, Any]] = {}
    for name, target in SP_PREAMBLE_TARGETS:
        if target is None:
            preamble, p_tokens = None, 0
        else:
            preamble, p_tokens = build_preamble(tok, target)
        ids = [render(preamble, s) for s in suffixes]
        warm = [render(preamble, s) for s in warmups]
        longest = max(len(r) for r in ids + warm)
        total = sum(len(r) for r in ids)
        if longest + SP_GEN_TOKENS > SP_MAX_MODEL_LEN:
            raise SmokeConfigError(
                f"shared-prefix set {name!r}: longest prompt is {longest} "
                f"tokens and the leg generates {SP_GEN_TOKENS}, which "
                f"exceeds max_model_len {SP_MAX_MODEL_LEN}; refusing")
        if total > SP_MAX_NUM_BATCHED_TOKENS:
            raise SmokeConfigError(
                f"shared-prefix set {name!r}: the batch carries {total} "
                f"prompt tokens against a {SP_MAX_NUM_BATCHED_TOKENS}-token "
                "scheduler budget — the prefill would be CHUNKED, which "
                "alphasteer cannot survive and which would make these legs "
                "incomparable to the certified ones; refusing")
        sets[name] = {
            "prompt_ids": ids,
            "warmup_ids": warm,
            "preamble_target_tokens": target,
            "preamble_measured_tokens": p_tokens,
            "shared_token_prefix": common_prefix_len(ids),
            "prompt_token_lens": [len(r) for r in ids],
            "prompt_tokens_total": total,
            "n_requests": len(ids),
            "generated_tokens_per_request": SP_GEN_TOKENS,
        }
    return sets


def assemble_fidelity_evidence(child: Mapping[str, Any]
                               ) -> Dict[str, Any]:
    """Map the steered worker's raw run outputs onto EXACTLY the
    evidence schema ``fidelity.evaluate_fidelity`` documents (five
    checks, keys == check names). Advisory arms contribute to
    scale-0/selection evidence (equality claims, safe regardless of
    bite) but are EXCLUDED from steering_bites, whose per-arm silence
    is a gate failure only the REQUIRED arms may trigger.

    THE EQUAL-SIZE RULE is enforced HERE as well as at serving time: the
    reference row for plan row i must have come from a batch of the SAME
    SIZE as the mixed batch, at the same position. A mismatch refuses
    loudly rather than quietly producing the false
    cross-request-contamination verdict a smaller reference produces on
    this engine (module docstring, SAME-SIZE REFERENCES). Raises
    RuntimeError; the caller turns that into smoke_failed.json."""
    runs = child["runs"]

    def ids(name: str) -> List[List[int]]:
        if name not in runs:
            raise RuntimeError(
                f"run {name!r} is MISSING from the steered worker's "
                "output — a real evidence absence (every fidelity-served "
                "arm gets its own steered__/scale0__ batch regardless of "
                "mixed-batch membership; mixed non-membership is "
                "legitimate sampling and is recorded separately as "
                "batch_invariance_sampled_arms, never as a missing run); "
                "refusing to assemble")
        return [[int(t) for t in row["tokens"]] for row in runs[name]]

    required = list(child["required_arm_ids"])
    all_arms = list(child["arm_ids"])
    plan = [(int(p), s) for p, s in child["mixed_plan"]]
    batched = ids("mixed_run1")
    reference = ids("reference")
    n_mixed = int(child["mixed_batch_size"])
    if not (len(plan) == len(batched) == len(reference) == n_mixed):
        raise RuntimeError(
            f"mixed plan carries {len(plan)} rows, mixed_run1 has "
            f"{len(batched)}, the reference servings yielded "
            f"{len(reference)} rows and the recorded mixed batch size is "
            f"{n_mixed} — misaligned evidence; refusing to assemble")

    ref_batches = [dict(m) for m in child["reference_batches"]]
    if len(ref_batches) != n_mixed:
        raise RuntimeError(
            f"{len(ref_batches)} reference servings recorded for a "
            f"{n_mixed}-row mixed batch — every row needs exactly one "
            "same-size reference; refusing to assemble")
    wrong_size = [(int(m.get("plan_row", -1)), int(m.get("batch_size", -1)))
                  for m in ref_batches
                  if int(m.get("batch_size", -1)) != n_mixed]
    if wrong_size:
        raise RuntimeError(
            "THE EQUAL-SIZE RULE IS VIOLATED: reference serving(s) "
            f"(plan_row, batch_size) {wrong_size[:8]} are not the mixed "
            f"batch's size {n_mixed}. Refusing to assemble evidence that "
            "would be believed. " + REFERENCE_SIZING_NOTE)
    misplaced = [
        [i, int(m.get("plan_row", -1)), int(m.get("row_under_test_index", -1))]
        for i, m in enumerate(ref_batches)
        if int(m.get("plan_row", -1)) != i
        or int(m.get("row_under_test_index", -1)) != i]
    if misplaced:
        raise RuntimeError(
            f"reference servings are not aligned to the mixed plan: "
            f"[list index, plan_row, row_under_test_index] {misplaced[:8]} "
            "— row i's reference must carry plan row i at position i, or "
            "the comparison is between two different requests; refusing "
            "to assemble")

    scale0_rows: List[List[int]] = []
    unsteered_rows: List[List[int]] = []
    for arm in all_arms:
        scale0_rows.extend(ids(f"scale0__{arm}"))
        unsteered_rows.extend(ids("baseline"))

    arm_set = set(all_arms)
    steered_i = [i for i, (_p, s) in enumerate(plan) if s in arm_set]
    none_i = [i for i, (_p, s) in enumerate(plan) if s is None]

    # NB: the returned mapping's keys must be EXACTLY the five check
    # names (evaluate_fidelity refuses unknown keys). The per-cell
    # batch-invariance SAMPLING record (mixed membership) therefore
    # lives in gate_evidence.json (the gate writes it), never here.
    return {
        "engine_determinism": {"run1_ids": ids("mixed_run1"),
                               "run2_ids": ids("mixed_run2")},
        "scale0_noop": {"scale0_ids": scale0_rows,
                        "unsteered_ids": unsteered_rows},
        "steering_bites": {"arms": {
            arm: {"steered_ids": ids(f"steered__{arm}"),
                  "scale0_ids": ids(f"scale0__{arm}")}
            for arm in required}},
        "batch_invariance": {
            "steered_reference_ids": [reference[i] for i in steered_i],
            "steered_batched_ids": [batched[i] for i in steered_i],
            "unsteered_reference_ids": [reference[i] for i in none_i],
            "unsteered_batched_ids": [batched[i] for i in none_i]},
        "selection_honored": {"requests": [
            {"arm": s, "prompt_index": p,
             "reference_ids": reference[i], "batched_ids": batched[i]}
            for i, (p, s) in enumerate(plan)]},
    }


def reference_sizing_facts(child: Mapping[str, Any]) -> Dict[str, Any]:
    """The sizes a reader must be able to check without reading code:
    the mixed batch's size, EVERY reference serving's size, and the
    plain statement of why a batch-of-1 reference is invalid here. Goes
    into smoke_record.json and smoke_evidence.json."""
    n_mixed = int(child["mixed_batch_size"])
    sizes = [int(m["batch_size"]) for m in child["reference_batches"]]
    return {
        "mixed_batch_size": n_mixed,
        "n_reference_servings": len(sizes),
        "reference_batch_sizes": sizes,
        "all_references_equal_mixed": bool(
            sizes and all(s == n_mixed for s in sizes)),
        "n_unsteered_filler_per_reference": n_mixed - 1,
        "n_filler_prompts_available": len(FILLER_PROMPTS),
        "size_fixed_batches_ignore_eos": SIZE_FIXED_IGNORE_EOS,
        "size_fixed_generated_tokens_per_row": SMOKE_MAX_TOKENS,
        "solo_references_are_invalid_here": True,
        "note": REFERENCE_SIZING_NOTE,
    }


def selection_direct_report(plan: Sequence[Sequence[Any]],
                            mixed_rows: Sequence[Mapping[str, Any]],
                            record: Optional[Mapping[str, Any]],
                            reset_status: str) -> Dict[str, Any]:
    """Compare the WORKER'S OWN record of which object it bound to each
    request against what each row of the mixed batch asked for.

    Pure, so it is exercised on CPU. Three verdicts per row, and the
    third is not folded into either of the others:
    ``match`` / ``MISMATCH`` / ``unobserved`` — an instrument that did
    not see a row is not the same claim as a row that resolved
    correctly, and reporting it as one would be the silent fallback this
    harness refuses everywhere else. A row that asked for NOTHING is
    checked too: the record's silence about it is the evidence that no
    object was bound to an unselected request."""
    rows = list(mixed_rows or [])
    requests = dict((record or {}).get("requests") or {})
    out_rows: List[Dict[str, Any]] = []
    n_match = n_mismatch = n_unobserved = n_selected = 0
    for i, entry in enumerate(plan):
        p, sel = int(entry[0]), entry[1]
        rid = rows[i].get("req_id") if i < len(rows) else None
        rec = requests.get(str(rid)) if rid is not None else None
        if sel is None:
            if rec is None:
                verdict, resolved = "match_none", None
                n_match += 1
            else:
                verdict = ("MISMATCH: a request carrying NO selection had an "
                           "object bound to it")
                resolved = rec.get("applied_cell")
                n_mismatch += 1
        else:
            n_selected += 1
            if rec is None:
                verdict, resolved = ("unobserved: the worker's record has no "
                                     "entry for this request id"), None
                n_unobserved += 1
            elif rec.get("applied_cell") == sel:
                verdict, resolved = "match", rec.get("applied_cell")
                n_match += 1
            else:
                verdict = "MISMATCH: bound object != requested object"
                resolved = rec.get("applied_cell")
                n_mismatch += 1
        out_rows.append({
            "row": i, "prompt_index": p, "intended": sel, "req_id": rid,
            "resolved": resolved, "verdict": verdict,
            "applied_expect_hash": (rec or {}).get("applied_expect_hash"),
            "applied_effective_scale": (rec or {}).get(
                "applied_effective_scale"),
            "n_applications": (rec or {}).get("n_applications"),
            "n_applications_modifying": (rec or {}).get(
                "n_applications_modifying"),
            "stable": (rec or {}).get("stable")})

    # The request-id-INDEPENDENT view: even when the row correlation
    # fails, "every observation the worker made bound what was asked for"
    # is a real check, and the resolved-cell set says whether every
    # intended arm was actually reached.
    pairs = dict((record or {}).get("pairs") or {})
    resolved_cells = sorted({str(a) for by in pairs.values() for a in by})
    intended_cells = sorted({str(s) for _p, s in
                             ((int(e[0]), e[1]) for e in plan) if s})
    n_obs = int((record or {}).get("n_observations") or 0)
    n_bad = int((record or {}).get("n_mismatched") or 0)
    aggregate_ok = bool(n_obs > 0 and n_bad == 0)

    if record is None:
        verdict = f"MISSING: {reset_status}"
    elif n_mismatch or n_bad:
        verdict = "FAILED"
    elif n_selected and n_unobserved == n_selected:
        verdict = ("MISSING: not one selected row could be correlated to the "
                   "worker's record by request id, so the per-row half did "
                   "not run. The aggregate half below still applies.")
    elif n_unobserved:
        verdict = "PARTIAL"
    elif aggregate_ok:
        verdict = "passed"
    else:
        verdict = ("MISSING: the worker recorded no observations for this "
                   "batch at all")
    return {
        "verdict": verdict,
        "note": SELECTION_DIRECT_NOTE,
        "arm_status": reset_status,
        "n_rows": len(out_rows),
        "n_rows_with_a_selection": n_selected,
        "n_matched": n_match,
        "n_mismatched": n_mismatch,
        "n_unobserved": n_unobserved,
        "aggregate_all_observations_matched": aggregate_ok,
        "aggregate_n_observations": n_obs,
        "aggregate_n_mismatched": n_bad,
        "worker_record_errors": int((record or {}).get("n_record_errors")
                                    or 0),
        "n_unstable_requests": int((record or {}).get("n_unstable_requests")
                                   or 0),
        "intended_cells": intended_cells,
        "resolved_cells": resolved_cells,
        "resolved_cells_cover_intended": resolved_cells == intended_cells,
        "rows": out_rows,
        "worker_record": dict(record) if record is not None else None,
    }


def selection_direct_failure(
        selection_direct: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The GATING consequence of the direct-selection report, for the
    gating phase alone: a ``FAILED`` verdict means the worker's own
    record shows a request bound to the WRONG object (or an unselected
    request bound to one) — a routing failure that outranks every
    token-level check, so it must fail the smoke (failure marker +
    nonzero), never ride along as a diagnostic. ``MISSING``/``PARTIAL``
    stay ADVISORY: an instrument that did not observe a row is not a
    row that resolved wrongly, and failing the run on lost telemetry
    would manufacture routing failures out of correlation gaps."""
    sd = dict(selection_direct or {})
    if str(sd.get("verdict") or "") != "FAILED":
        return None
    return (
        "selection_direct FAILED — the worker's own selection record "
        "contradicts what the mixed batch asked for: "
        f"{sd.get('n_mismatched')} per-row mismatch(es) across "
        f"{sd.get('n_rows')} rows, "
        f"{sd.get('aggregate_n_mismatched')} aggregate mismatch(es) over "
        f"{sd.get('aggregate_n_observations')} worker observations. A "
        "mis-bound object means some arm's rows were served by a "
        "different intervention; nothing downstream of routing can be "
        "trusted (see selection_direct.rows in the record)")


def check_state(verdict: str) -> str:
    """One fidelity check's verdict, reduced to passed / indeterminate /
    failed. ``evaluate_fidelity`` writes "passed", an "indeterminate: ..."
    line when engine_determinism invalidated the report, or the failure
    message itself — and the three must not be collapsed (an
    indeterminate check is not a failed one, and reporting it as failed
    at BI=0 would manufacture failures)."""
    v = str(verdict)
    if v == "passed":
        return "passed"
    if v.startswith("indeterminate"):
        return "indeterminate"
    return "failed"


def phase_check_split(report_dict: Mapping[str, Any], *,
                      batch_invariant: str, gating: bool) -> Dict[str, Any]:
    """Per-check states, plus — for ANY phase at BI=0, gating or not —
    the split between failures that are EXPECTED at BI=0 and failures
    that are not. The split exists so nobody reads the run as the
    serving path breaking; it never suppresses a failure, which is
    reported either way. For a GATING BI=0 phase the expected side is
    also the WAIVED side (split_gating_failures): recorded and printed,
    but not a job failure, because composition invariance is not a
    property production claims (EXACT_REPLAY_DROPPED_NOTE)."""
    checks = dict(report_dict.get("checks") or {})
    states = {n: check_state(v) for n, v in checks.items()}
    failed = sorted(n for n, s in states.items() if s == "failed")
    bi0 = str(batch_invariant) == "0"
    expected = [n for n in failed if bi0 and n in BI0_EXPECTED_FAILURES]
    unexpected = [n for n in failed if n not in expected]
    return {
        "check_states": states,
        "failed_checks": failed,
        "expected_failures": expected,
        "unexpected_failures": unexpected,
        "expected_failure_reason": (BI0_EXPECTED_FAILURE_REASON if bi0
                                    else None),
        "all_expected": bool(bi0 and failed and not unexpected),
    }


#: failure texts that mean the INSTRUMENT broke rather than the outputs
#: moving — never waivable (split_gating_failures). These are
#: evaluate_fidelity's own normalized messages; the coupling is pinned
#: by test_petri_bi0_gating.
_INSTRUMENT_FAILURE_MARKS = (
    "no evidence gathered",
    "evidence must be a mapping",
    "evidence could not be decoded",
)


def split_gating_failures(failures: Sequence[str], *, batch_invariant: str
                          ) -> Tuple[List[str], List[str]]:
    """Split a gating phase's fidelity failures into ``(gating,
    waived)``. A failure string produced by ``evaluate_fidelity`` starts
    with its check's name; at BI=0 — the certified production
    configuration — failures of the composition-dependent checks
    (:data:`BI0_EXPECTED_FAILURES`) are WAIVED: recorded, printed, and
    tagged in the record, but not job failures, because composition
    invariance is exactly the property that was given up with
    exact_replay (:data:`EXACT_REPLAY_DROPPED_NOTE`). Everything else —
    engine_determinism, scale0_noop, steering_bites, and structural
    ``evidence:`` failures — always gates; so does selection_honored's
    DIRECT half, which is enforced separately
    (:func:`selection_direct_failure`) and is composition-independent.
    At any other BI setting nothing is waived.

    INSTRUMENT failures are never waived, even on a waivable check: the
    waiver's whole justification is "the output moved for numerical
    reasons", which requires the comparison to have RUN. A probe that
    gathered no evidence, malformed evidence, or an unreadable record is
    the harness breaking, not a composition effect — those strings
    (``evaluate_fidelity``'s own normalizations) gate at any BI."""
    waivable = (set(BI0_EXPECTED_FAILURES)
                if str(batch_invariant) == "0" else set())
    gating: List[str] = []
    waived: List[str] = []
    for f in failures:
        text = str(f)
        name = text.split(":", 1)[0].strip()
        instrument = any(mark in text for mark in _INSTRUMENT_FAILURE_MARKS)
        (gating if (name not in waivable or instrument)
         else waived).append(text)
    return gating, waived


def parse_probe_utils(raw: Any) -> Tuple[float, ...]:
    """``--petri32-probe-utils``: a comma list of gpu_memory_utilization
    values for the 32-way re-measure, or ``skip`` (also ``off``/``none``
    /empty) to run none. Malformed values REFUSE — a typo'd probe list
    silently measuring nothing would read as "32 was re-tested"."""
    text = str(raw if raw is not None else "").strip().lower()
    if text in ("skip", "off", "none", ""):
        return ()
    utils: List[float] = []
    for part in text.split(","):
        part = part.strip()
        try:
            u = float(part)
        except ValueError:
            raise SystemExit(
                f"--petri32-probe-utils: {part!r} is not a float (got "
                f"{raw!r}; expected e.g. '0.90,0.95,0.97' or 'skip')")
        if not 0.5 <= u <= 0.99:
            raise SystemExit(
                f"--petri32-probe-utils: {u} outside [0.5, 0.99] — vLLM "
                "cannot give KV+weights+warmup a utilization outside that "
                "range on this card; a typo would burn an engine init")
        utils.append(u)
    if len(set(utils)) != len(utils):
        raise SystemExit(
            f"--petri32-probe-utils: duplicate utilization in {raw!r}")
    return tuple(utils)


def lens_leg_rates(legs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """tokens/s per lens leg, for the BI=1-vs-BI=0 comparison on the LENS
    engine itself — so the batch-invariance cost is confirmed where the
    gate runs rather than inferred from the plugin-disabled ladder."""
    return {k: v.get("tokens_per_s")
            for k, v in sorted((legs or {}).items())
            if isinstance(v, Mapping) and v.get("tokens_per_s") is not None}


def batch_invariance_cost(bi1: Mapping[str, Any],
                          bi0: Mapping[str, Any]) -> Dict[str, Any]:
    """BI=0 tokens/s divided by BI=1 tokens/s, per lens leg present in
    both. A leg missing on either side is named rather than dropped."""
    out: Dict[str, Any] = {}
    for leg in sorted(set(bi1) | set(bi0)):
        a, b = bi1.get(leg), bi0.get(leg)
        if a and b:
            out[leg] = {"bi1_tokens_per_s": a, "bi0_tokens_per_s": b,
                        "speedup_bi0_over_bi1": round(b / a, 3)}
        else:
            out[leg] = {"bi1_tokens_per_s": a, "bi0_tokens_per_s": b,
                        "speedup_bi0_over_bi1": None,
                        "missing": "MISSING: the leg was not measured at "
                                   + ("BI=1" if not a else "BI=0")}
    return out


def advisory_bite_counts(child: Mapping[str, Any]) -> Dict[str, Any]:
    """Diagnostic-only bite counts for advisory arms (never a gate)."""
    runs = child["runs"]
    out: Dict[str, Any] = {}
    for arm in child["arm_ids"]:
        if arm in child["required_arm_ids"]:
            continue
        steered = [r["tokens"] for r in runs[f"steered__{arm}"]]
        zero = [r["tokens"] for r in runs[f"scale0__{arm}"]]
        out[arm] = {"n": len(steered),
                    "n_differing": sum(1 for a, b in zip(steered, zero)
                                       if list(a) != list(b))}
    return out


# --- markers (the probe's D13 discipline, smoke names) ---------------------------


def _write_marker(out_dir: Path, name: str, payload: Mapping[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / name
    marker.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                                 default=str))
    return marker


def record_unavailable(out_dir: Path, reason: str, **extra: Any) -> Path:
    return _write_marker(out_dir, UNAVAILABLE_MARKER, {
        "status": "smoke unavailable", "reason": reason,
        "recorded_at": time.time(), **extra})


def record_failure(out_dir: Path, reason: str, **extra: Any) -> Path:
    return _write_marker(out_dir, FAILURE_MARKER, {
        "status": "smoke FAILED (steered serving did not hold)",
        "reason": reason, "recorded_at": time.time(), **extra})


# --- artifact derivation + registry build (parent, CPU) --------------------------


class SmokeConfigError(RuntimeError):
    """A smoke-configuration refusal (bad object, bad blob, bad spec)."""


def load_arm_artifact(cell_id: str, kind: str, path: Path,
                      deploy_scale: Optional[float]
                      ) -> Tuple[bytes, Dict[str, Any]]:
    """The arm's artifact bytes, with ``base_scale`` rewritten to the
    deployment scale when one is given (module docstring: the registry
    path's one scale slot is the blob's own base_scale, which the
    adapter bakes into HookOp.coeff). Tensors are never touched; the
    derivation is recorded (original + served sha16 and base_scale)."""
    import io

    import torch

    raw = Path(path).read_bytes()
    rec: Dict[str, Any] = {"cell_id": cell_id, "kind": kind,
                           "path": str(path), "original_sha16": sha16(raw),
                           "deploy_scale": deploy_scale, "derived": False}
    blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or blob.get("kind") != kind:
        raise SmokeConfigError(
            f"arm {cell_id!r}: {path} stores kind "
            f"{blob.get('kind') if isinstance(blob, dict) else type(blob).__name__!r},"
            f" expected {kind!r} — wrong object file; refusing")
    orig_scale = blob.get("base_scale")
    rec["original_base_scale"] = (float(orig_scale)
                                  if isinstance(orig_scale, (int, float))
                                  else None)
    meta = blob.get("metadata")
    if isinstance(meta, Mapping) and meta.get("deployed_strength") is not None:
        rec["metadata_deployed_strength"] = str(meta["deployed_strength"])
    if deploy_scale is not None and rec["original_base_scale"] != float(
            deploy_scale):
        derived = dict(blob)
        derived["base_scale"] = float(deploy_scale)
        buf = io.BytesIO()
        torch.save(derived, buf)
        raw = buf.getvalue()
        rec["derived"] = True
    rec["served_base_scale"] = (float(deploy_scale)
                                if deploy_scale is not None
                                else rec["original_base_scale"])
    rec["served_sha16"] = sha16(raw)
    return raw, rec


def op_facts(ops: Sequence[Any]) -> Dict[str, Any]:
    """The three things that decide whether a cross-phase cell is really
    a cross-phase cell — position mode, phase, and the header ids the
    mask is located by — read off the ops the REAL adapter returned."""
    hdrs = {op.header_ids for op in ops}
    one = hdrs.pop() if len(hdrs) == 1 else None
    return {"n_ops": len(ops),
            "positions": sorted({op.positions for op in ops}),
            "phases": sorted({op.phase for op in ops}),
            "header_ids": list(one) if one else None,
            "n_header_ids": len(one) if one else 0}


def probe_response_mode_arm(cell_id: str, artifact_bytes: bytes,
                            header_ids: Sequence[int]) -> Dict[str, Any]:
    """Adapt the static_r bytes under ``positions: "response"`` with the
    REAL committed adapter, and check the three facts that make the
    result the cross-phase cell rather than something that merely
    loaded (module docstring, THE ARMS):

      positions == "response"  — the mask is actually in force;
      header_ids non-empty     — it has a boundary to locate;
      phase == "both"          — it is applied at prefill AND at every
                                 decode step. A prefill-only op would
                                 leave generation unsteered while
                                 everything else looked healthy, which
                                 is the exact silent failure this arm
                                 exists to exercise.

    Never raises: an adapter refusal or an unexpected op shape comes
    back as a ``status`` the caller reports loudly and drops the arm
    over, because this arm is advisory and must not be able to fail the
    run while the mechanism is new."""
    from sisteer.capeval.serving import lens_adapters as la

    spec = {"kind": "static_r", "positions": "response"}
    rec: Dict[str, Any] = {"cell_id": cell_id, "resolved_spec": dict(spec),
                           "ok": False}
    try:
        ops = la.get_adapter("static_r")(spec, artifact_bytes, b"",
                                         header_ids=tuple(header_ids) or None)
    except Exception as e:
        rec["status"] = (
            f"MISSING: the committed static_r adapter REFUSED the response "
            f"override — {type(e).__name__}: {e}. The cross-phase cell is "
            "unservable on this branch; the arm is dropped and nothing "
            "else in the run is affected")
        return rec
    rec.update(op_facts(ops))
    problems: List[str] = []
    if rec["positions"] != ["response"]:
        problems.append(
            f"ops carry positions {rec['positions']}, not ['response'] — the "
            "resolved spec's override did not reach the served ops, so this "
            "would have measured the ALL-position arm under a second name")
    if rec["phases"] != ["both"]:
        problems.append(
            f"ops carry phase {rec['phases']}, not ['both'] — a "
            "prefill-only op leaves every generated position UNSTEERED "
            "while the arm still looks installed and still bites at "
            "prefill; that is the specific silent failure this arm exists "
            "to exercise")
    if not rec["header_ids"]:
        problems.append(
            "ops carry no single non-empty header_ids — the response mask "
            "is LOCATED by matching those ids against the request's own "
            "tokens, so without them there is no boundary to check")
    elif rec["header_ids"] != [int(t) for t in header_ids]:
        problems.append(
            f"ops carry header_ids {rec['header_ids']} but the served "
            f"template derives {[int(t) for t in header_ids]} — the mask "
            "would be located against a header this server does not emit")
    if problems:
        rec["status"] = "UNEXPECTED: " + " | ".join(problems)
        return rec
    rec["ok"] = True
    rec["status"] = ("ok — all-position artifact adapted under the response "
                     "mask, phase both, header located by token ids")
    return rec


def response_override_refusals(alphasteer_bytes: bytes,
                               header_ids: Sequence[int]) -> Dict[str, Any]:
    """THE NEGATIVE CONTROL, on CPU and for free: the two arms whose
    edit is computed from the LAST PROMPT TOKEN and broadcast must
    REFUSE a ``positions: "response"`` override, because re-masking
    relocates that edit rather than restricting it (``d1851eb``'s
    DERIVATION LOCALITY rule). Without this, "the override works" is
    only ever half-tested — the half that says yes.

    ``cosine_gated_r`` is checked with EMPTY artifact bytes on purpose:
    its refusal is the first statement in the adapter and precedes any
    read of the blob, so the control needs no object file. If that ever
    stops being true the control fails loudly on a blob error, which is
    the right way for it to break."""
    from sisteer.capeval.serving import lens_adapters as la

    out: Dict[str, Any] = {}
    for kind, data, note in (
            ("alphasteer", alphasteer_bytes,
             "the real landed object's bytes"),
            ("cosine_gated_r", b"",
             "empty bytes: the refusal precedes any artifact read, so this "
             "control needs no object file")):
        spec = {"kind": kind, "positions": "response"}
        try:
            la.get_adapter(kind)(spec, data, b"",
                                 header_ids=tuple(header_ids) or None)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            out[kind] = {
                "refused": True,
                "names_the_rule": "DERIVATION LOCALITY" in str(e),
                "error": msg[:400], "note": note}
        else:
            out[kind] = {
                "refused": False, "names_the_rule": False, "note": note,
                "error": ("ADAPTED — the response override was NOT refused. "
                          "This arm's edit is computed from the last prompt "
                          "token and broadcast, so a response mask "
                          "RELOCATES it onto generated positions rather "
                          "than restricting it: the served function is one "
                          "nobody has defined")}
    return out


def build_registry(registry_dir: Path, arms: Sequence[Mapping[str, Any]],
                   header_ids: Sequence[int]) -> Dict[str, Any]:
    """Write the worker registry dir: one canonical entry per arm plus
    its donor-backed scale-0 twin — the exact index schema
    ``lens_steer.WorkerObjectRegistry.load`` verifies (hash bindings,
    donor field-equality, header ids). Built directly (a full
    ``export_worker_registry`` needs a PetriCellSet/CellSource this
    smoke has no business inventing); the LOAD path — the committed
    verification — runs unmodified, in the parent AND in the worker.

    Each arm's ops are adapted here first via the REAL per-kind adapter,
    both to fail an unadaptable required arm before any GPU spend and to
    compute the expected served (site-shifted) hook layers the install
    report is asserted against. Returns the per-arm expectation map.

    An arm may carry ``spec_extra`` — extra RESOLVED-SPEC keys, which is
    how the cross-phase cell is expressed: the same bytes with
    ``positions: "response"`` in the spec. It is not smuggled, because
    ``petri_intervention_hash`` covers the resolved spec, so the two
    cells hash differently and the registry's own selector binding
    proves they are distinct."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lens_batch as lb

    registry_dir = Path(registry_dir)
    if registry_dir.exists():
        shutil.rmtree(registry_dir)
    registry_dir.mkdir(parents=True)

    entries: List[Dict[str, Any]] = []
    expects: Dict[str, Any] = {}
    for arm in arms:
        cell_id, kind, data = arm["cell_id"], arm["kind"], arm["bytes"]
        if not _SAFE_ID.fullmatch(cell_id):
            raise SmokeConfigError(f"arm cell id {cell_id!r} is not "
                                   "identifier-safe")
        spec: Dict[str, Any] = {"kind": kind}
        spec.update(dict(arm.get("spec_extra") or {}))
        ops = la.get_adapter(kind)(spec, data, b"",
                                   header_ids=tuple(header_ids) or None)
        ihash = cm.petri_intervention_hash(spec, artifact_bytes=data,
                                           sidecar_bytes=b"")
        rel = f"{cell_id}/artifact.bin"
        p = registry_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        canonical = {
            "cell_id": cell_id, "kind": kind, "resolved_spec": spec,
            "expect_hash": ihash, "artifact": rel, "sidecar": None,
            "artifact_sha16": cm.sha256_hex16(data),
            "artifact_intervention_hash": ihash,
            "forced_scale": None, "donor_cell": None, "validated_by": None,
        }
        twin = dict(canonical)
        twin.update(cell_id=scale0_cell_id(cell_id),
                    expect_hash=cm.sha256_hex16(
                        f"{ihash}:scale0".encode("utf-8")),
                    forced_scale=0.0, donor_cell=cell_id)
        entries.extend([canonical, twin])
        expects[cell_id] = {
            "expect_hash": ihash,
            "resolved_spec": dict(spec),
            "artifact_sha16": canonical["artifact_sha16"],
            "scale0_cell": twin["cell_id"],
            "scale0_hash": twin["expect_hash"],
            "served_layers": sorted({op.layer for op in ops}),
            "op_kinds": sorted({op.op for op in ops}),
            "coeffs": sorted({float(op.coeff) for op in ops}),
            "needs_ledger": lb.ops_need_plan(ops),  # at scale 1.0
            **op_facts(ops),
        }
    index = {"version": 1,
             "assistant_header_ids": list(header_ids) or None,
             "entries": entries}
    (registry_dir / "registry.json").write_text(
        json.dumps(index, indent=2, sort_keys=True))
    return expects


# --- shared engine-evidence helpers (children) ------------------------------------


def _reported_packages() -> Dict[str, str]:
    import importlib.metadata as md
    out = {}
    for pkg in ("vllm", "vllm-lens", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except Exception as e:
            out[pkg] = f"UNIMPORTABLE({type(e).__name__})"
    return out


def _patch_state() -> Dict[str, Any]:
    """Which vllm entry points carry the lens patches — read from the
    live classes (the probe's helper, verbatim)."""
    out: Dict[str, Any] = {}
    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        out["generate_patched"] = (
            getattr(LLM.generate, "__module__", "")
            == "vllm_lens._activations_plugin")
        out["engine_args_patched"] = (
            getattr(EngineArgs.create_engine_config, "__module__", "")
            == "vllm_lens._activations_plugin")
    except Exception as e:  # pragma: no cover - live-only
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _live_engine_evidence(llm: Any) -> Dict[str, Any]:
    """Resolved engine configuration, read from the LIVE config
    (authoritative — what actually runs; the probe's walker, extended
    with the scheduler geometry this smoke must assert). Unreadable
    fields record None; the caller decides which Nones refuse."""
    out: Dict[str, Any] = {"enforce_eager": None,
                           "prefix_caching_enabled": None,
                           "worker_extension_cls": None,
                           "max_num_batched_tokens": None,
                           "max_num_seqs": None,
                           "max_model_len": None,
                           "model": None,
                           "tokenizer": None}
    cfg = None
    for path in ("llm_engine.vllm_config", "llm_engine.engine_config",
                 "llm_engine.model_config"):
        obj = llm
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            cfg = obj
            break
        except AttributeError:
            continue
    mc = getattr(cfg, "model_config", cfg)
    for key, attr, cast in (("enforce_eager", "enforce_eager", bool),
                            ("max_model_len", "max_model_len", int),
                            ("model", "model", str),
                            ("tokenizer", "tokenizer", str)):
        try:
            out[key] = cast(getattr(mc, attr))
        except Exception:
            pass
    try:
        cc = getattr(cfg, "cache_config", None)
        if cc is not None:
            out["prefix_caching_enabled"] = bool(
                getattr(cc, "enable_prefix_caching"))
    except Exception:
        pass
    try:
        pc = getattr(cfg, "parallel_config", None)
        if pc is not None:
            out["worker_extension_cls"] = getattr(
                pc, "worker_extension_cls", None)
    except Exception:
        pass
    try:
        sc = getattr(cfg, "scheduler_config", None)
        if sc is not None:
            out["max_num_batched_tokens"] = int(
                getattr(sc, "max_num_batched_tokens"))
            out["max_num_seqs"] = int(getattr(sc, "max_num_seqs"))
    except Exception:
        pass
    return out


def _worker_kv_cache_facts(worker: Any) -> Dict[str, Any]:
    """Runs INSIDE each worker over ``collective_rpc`` (a callable RPC:
    the pinned 0.25.1's ``serial_utils.run_method`` binds the worker
    object as the first argument). Reads the RESOLVED KV cache size off
    the worker's own ``cache_config`` — ``gpu_worker.initialize_cache``
    wrote ``num_gpu_blocks`` there after memory profiling — PLUS the
    KV-cache GROUP structure and the allocator's own max-concurrency
    verdict via the SHARED ``lens_worker.kv_group_report`` (the same
    fragment every lens launch's readiness reads over the
    ``sisteer_scheduler_config`` RPC). The group half is what makes the
    coverage verdict correct on hybrids: on Qwen3.5-style models
    ``num_gpu_blocks * block_size`` is NOT token capacity (LCM
    scheduler block size; Mamba/GDN state pages), and a gate that
    certifies coverage from it is quietly wrong — worse than no gate.
    No fallback: an unreadable value must refuse upstream, never pass
    as coverage (kv_group_report raises, the call site records the
    reason in ``read_error``, and the group-less facts refuse)."""
    from sisteer.capeval.serving.lens_worker import kv_group_report
    cc = worker.vllm_config.cache_config
    out = {"num_gpu_blocks": getattr(cc, "num_gpu_blocks", None),
           "block_size": getattr(cc, "block_size", None)}
    out.update(kv_group_report(worker.vllm_config,
                               getattr(worker, "model_runner", None)))
    return out


def kv_coverage_problem(kv_facts: Mapping[str, Any], *, max_num_seqs: int,
                        max_model_len: int) -> Optional[str]:
    """The KV-COVERAGE HALF OF A GEOMETRY, as a gate-shaped verdict:
    ``None`` when the resolved KV cache holds the FULL no-chunking
    budget (``max_num_seqs`` sequences at ``max_model_len`` each — every
    scheduled sequence can be resident at full length simultaneously),
    else a one-line problem naming both numbers. max_num_seqs and the
    token budget are scheduler CEILINGS, not grants: vLLM runs a
    32 x 16384 geometry over a smaller KV cache without erroring, it
    just schedules fewer sequences and preempts (the production server
    at that geometry resolved 486,944 KV tokens = 29.72x — under 32 —
    at the default utilization). Unreadable facts are a problem too,
    never a pass: a coverage that cannot be verified must not certify.

    DELEGATES to the launcher's ``vllm_server.kv_coverage_problem`` —
    the single shared verdict, two-branch: a UNIFORM cache (one KV
    cache group) keeps the original num_gpu_blocks x block_size
    arithmetic verbatim; a HYBRID cache is held to the allocator's own
    ``kv_max_concurrency`` instead, because the uniform formula
    misstates capacity there in either direction with everything still
    green. This function was the THIRD copy of the formula (readiness,
    the in-proc smoke's assert, and here) — the copy count is how the
    hybrid miscount survived two fixes, so the smoke now adapts its
    legacy fact keys (``num_gpu_blocks``/``block_size``, kept unprefixed
    for continuity with recorded job logs) onto the report names and
    calls the shared verdict rather than re-deriving it."""
    from sisteer.capeval.serving.vllm_server import (
        kv_coverage_problem as _shared_verdict)
    rep = dict(kv_facts)
    rep["kv_num_gpu_blocks"] = kv_facts.get("num_gpu_blocks")
    rep["kv_block_size"] = kv_facts.get("block_size")
    return _shared_verdict(rep, max_num_seqs=max_num_seqs,
                           max_model_len=max_model_len)


def _greedy_params(max_tokens: int, *, ignore_eos: bool = False,
                   selector: Optional[Mapping[str, Any]] = None) -> Any:
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                          ignore_eos=ignore_eos,
                          extra_args=dict(selector) if selector else None)


def _chat_prompt_ids(tok: Any, prompts: Sequence[str]) -> List[List[int]]:
    """Prompts chat-templated + tokenized ONCE with the served
    tokenizer; requests are submitted as token ids so the lengths the
    ledger is armed with are exactly the lengths the engine serves."""
    return [render_ids(tok, [{"role": "user", "content": p}], True)
            for p in prompts]


def _selection_record_rpc(llm: Any, *, reset: bool = False) -> Dict[str, Any]:
    """The worker's OWN record of which object it bound to each request
    (``lens_worker.SelectionRecord``), over the extension's RPC.

    THE AUTHORITATIVE SOURCE for "which object was selected": the worker
    wrote it as it served. Raises on absence rather than reconstructing
    anything — the caller records the failure as MISSING and keeps the
    inferred output-comparison half separate, never substituting one for
    the other."""
    rpc = llm.collective_rpc("sisteer_selection_record", args=(bool(reset),))
    out = rpc[0] if isinstance(rpc, (list, tuple)) else rpc
    if not isinstance(out, Mapping):
        raise RuntimeError(
            f"sisteer_selection_record returned {type(out).__name__}, not a "
            "mapping — the worker's selection record cannot be read")
    return dict(out)


# --- throughput leg records: every policy on every leg ----------------------------
# A leg's name is a label, not evidence. Each record carries the three
# engine policies (prefix caching, CUDA graphs, batch invariance), the
# scheduler geometry it ran under, and the request geometry including
# the MEASURED preamble length — so the ladder can be read, and
# subtracted, without knowing how this file spells its tags.


def leg_policy(*, suite: str, prompt_set: str, engine: str,
               prefix_caching: Optional[bool], cuda_graphs: Optional[bool],
               batch_invariant: str, geometry: Mapping[str, Any],
               request_set: Mapping[str, Any]) -> Dict[str, Any]:
    return {"suite": suite, "prompt_set": prompt_set, "engine": engine,
            "prefix_caching": prefix_caching, "cuda_graphs": cuda_graphs,
            "batch_invariant": batch_invariant,
            "scheduler_geometry": dict(geometry),
            "request_geometry": dict(request_set)}


def throughput_leg(outs: Sequence[Any], wall: float, n_requests: int,
                   policy: Mapping[str, Any]) -> Dict[str, Any]:
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"n_requests": n_requests, "generated_tokens": gen,
            "wall_s": round(wall, 3),
            "tokens_per_s": round(gen / wall, 2) if wall > 0 else None,
            "prompts_per_min": (round(n_requests * 60.0 / wall, 2)
                                if wall > 0 else None),
            **dict(policy)}


def missing_leg(reason: str, policy: Mapping[str, Any]) -> Dict[str, Any]:
    """A leg that could not be measured, recorded AS A LEG. Never
    omitted (a reader would not know to look for it) and never folded
    into a neighbour (which would silently change what that number
    means)."""
    return {"missing": f"MISSING: {reason}", **dict(policy)}


def _sp_request_geometry(s: Mapping[str, Any]) -> Dict[str, Any]:
    return {"batch_size": s["n_requests"],
            "generated_tokens_per_request": s["generated_tokens_per_request"],
            "shared_preamble_target_tokens": s["preamble_target_tokens"],
            "shared_preamble_measured_tokens": s["preamble_measured_tokens"],
            "measured_shared_token_prefix": s["shared_token_prefix"],
            "prompt_token_lens": s["prompt_token_lens"],
            "prompt_tokens_total": s["prompt_tokens_total"],
            "warmup": ("same preamble, DIFFERENT suffixes — a cache leg "
                       "therefore measures the steady state (system prompt "
                       "cached, each request's own text new), not a "
                       "full-prompt cache hit")}


def _lens_shared_prefix_legs(tok: Any, run_batch: Any, arm_ids: Sequence[str],
                             response_arm_ids: Sequence[str],
                             header_ids: Sequence[int],
                             geometry: Mapping[str, Any],
                             result: Dict[str, Any]) -> Dict[str, Any]:
    """The lens engine's shared-prefix throughput legs: unselected, then
    each arm, over each preamble variant.

    TWO PASSES, and the order is load-bearing. A steering failure
    CONDEMNS the worker — every later call refuses — so one raise ends
    the child. Response-mode arms have never run on hardware, so all of
    the certified arms' legs are measured across all three prompt sets
    FIRST, and the response arms' legs come afterwards. A refusal then
    costs only the response legs, which are recorded MISSING with the
    reason, instead of the whole suite.

    They are additionally gated on a CPU header check of each set's own
    prompts, because the mask is located by matching the assistant
    header at prefill and refuses on mismatch — the cheap check catches
    the likely cause before the expensive one can fire."""
    sets = build_sp_prompt_sets(tok)
    result["shared_prefix_sets"] = {
        name: {k: v for k, v in s.items()
               if k not in ("prompt_ids", "warmup_ids")}
        for name, s in sets.items()}
    plain = [a for a in arm_ids if a not in response_arm_ids]
    response = [a for a in arm_ids if a in response_arm_ids]
    header_reports: Dict[str, Any] = {}
    legs: Dict[str, Any] = {}
    warmed: Dict[str, bool] = {}

    def policy_for(set_name: str, s: Mapping[str, Any]) -> Dict[str, Any]:
        return leg_policy(
            suite="shared_prefix", prompt_set=set_name,
            engine=("lens (VLLM_LENS_DISABLE unset, "
                    "SisteerWorkerExtension installed)"),
            prefix_caching=False, cuda_graphs=False,
            batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "?"),
            geometry=geometry, request_set=_sp_request_geometry(s))

    for arms, is_response in ((plain, False), (response, True)):
        tags = ([] if is_response else [("lens_unselected", None)]) + [
            (f"lens_steered__{a}", a) for a in arms]
        if not tags:
            continue
        for set_name, s in sets.items():
            table = list(s["prompt_ids"]) + list(s["warmup_ids"])
            n_req = len(s["prompt_ids"])
            pol = policy_for(set_name, s)
            hdr = header_reports.get(set_name)
            if hdr is None:
                hdr = header_tail_report(s["prompt_ids"], header_ids,
                                         f"shared_prefix:{set_name}")
                header_reports[set_name] = hdr
            if not warmed.get(set_name):
                try:
                    run_batch(f"sp_warmup__{set_name}",
                              [(n_req + i, None)
                               for i in range(len(s["warmup_ids"]))],
                              max_tokens=TP_WARMUP_TOKENS, ignore_eos=True,
                              table=table)
                    warmed[set_name] = True
                except Exception as e:
                    for tag, _sel in tags:
                        legs[f"{set_name}__{tag}"] = missing_leg(
                            f"the warmup for prompt set {set_name!r} raised "
                            f"{type(e).__name__}: {e}", pol)
                    continue
                finally:
                    result.setdefault("runs", {}).pop(
                        f"sp_warmup__{set_name}", None)
            for tag, sel in tags:
                key = f"{set_name}__{tag}"
                if is_response and not hdr["all_end_with_header"]:
                    legs[key] = missing_leg(
                        f"{hdr['n_mismatched']}/{hdr['n_prompts']} prompts "
                        f"in set {set_name!r} do not END with the assistant "
                        f"header {hdr['header_ids']}, so the response mask "
                        "cannot be located — a template/render "
                        "disagreement, not a steering failure; skipped "
                        "rather than raising and condemning the worker",
                        pol)
                    continue
                try:
                    outs, wall = run_batch(f"sp__{key}",
                                           [(i, sel) for i in range(n_req)],
                                           max_tokens=SP_GEN_TOKENS,
                                           ignore_eos=True, table=table)
                    legs[key] = throughput_leg(outs, wall, n_req, pol)
                except Exception as e:
                    legs[key] = missing_leg(f"{type(e).__name__}: {e}", pol)
                finally:
                    # token ids of throughput runs are never evidence.
                    result.setdefault("runs", {}).pop(f"sp__{key}", None)
    result["shared_prefix_header_check"] = header_reports
    return legs


# --- worker-steered: the lens engine + SisteerWorkerExtension ---------------------


def cmd_worker_steered(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    result: Dict[str, Any] = {"stage": "preflight",
                              "packages": _reported_packages(),
                              "env": {k: os.environ.get(k) for k in (
                                  "VLLM_BATCH_INVARIANT",
                                  "VLLM_USE_V2_MODEL_RUNNER",
                                  "VLLM_ENABLE_V1_MULTIPROCESSING",
                                  "VLLM_LENS_DISABLE")}}

    def dump(code: int, **extra: Any) -> int:
        result.update(extra)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       default=str))
        return code

    result["chat_markers_shim"] = ensure_chat_markers()
    from sisteer.capeval.serving import lens_worker as lw

    plugin: Dict[str, Any] = {"pre_load": _patch_state()}
    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        plugin["forced_load_called"] = True
    except Exception as e:
        plugin["forced_load_called"] = False
        plugin["forced_load_error"] = f"{type(e).__name__}: {e}"
    plugin.update(_patch_state())
    result["plugin"] = plugin

    arms_cfg = json.loads(Path(args.arms_json).read_text())
    arm_ids: List[str] = list(arms_cfg["arm_ids"])
    required: List[str] = list(arms_cfg["required_arm_ids"])
    mixed_arm_ids: List[str] = list(arms_cfg.get("mixed_arm_ids") or arm_ids)
    response_arm_ids: List[str] = list(arms_cfg.get("response_arm_ids") or [])
    header_ids: List[int] = [int(t) for t in
                             (arms_cfg.get("assistant_header_ids") or [])]
    selectors: Dict[str, Dict[str, Any]] = dict(arms_cfg["selectors"])
    ledger_cells = set(arms_cfg["expected_ledger_cells"])
    result["arm_ids"] = arm_ids
    result["required_arm_ids"] = required
    result["mixed_arm_ids"] = mixed_arm_ids
    result["response_arm_ids"] = response_arm_ids

    # THE GEOMETRY IS A PARAMETER, and both callers pass it explicitly
    # (module docstring, TWO GEOMETRIES): the fidelity phase runs the
    # certified 2048/32 and the shared-prefix throughput phase runs
    # 8192/8. max_num_batched_tokens is RE-DERIVED from this engine's
    # own two numbers rather than inherited, so the no-chunking argument
    # is the same argument on both.
    max_model_len = int(args.max_model_len)
    max_num_seqs = int(args.max_num_seqs)
    max_num_batched_tokens = max_num_seqs * max_model_len
    gpu_memory_utilization = float(args.gpu_memory_utilization)
    result["geometry"] = {"max_model_len": max_model_len,
                          "max_num_seqs": max_num_seqs,
                          "max_num_batched_tokens": max_num_batched_tokens,
                          "gpu_memory_utilization": gpu_memory_utilization,
                          "phase": args.phase}

    result["stage"] = "engine_init"
    try:
        from vllm import LLM
        llm = LLM(
            model=args.model, revision=args.revision,
            tokenizer_revision=args.revision, dtype="bfloat16", seed=1234,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            max_num_seqs=max_num_seqs,
            # LOAD-BEARING (do not "tidy" this into a smaller number):
            # vLLM V1 forces chunked prefill on and offers no flag to
            # disable it. max_num_batched_tokens = max_num_seqs *
            # max_model_len means the scheduler's token budget can never
            # bind mid-prompt, so no prompt is ever split. alphasteer's
            # edit is computed ONCE from the LAST prompt token and
            # broadcast to every prompt position — under a chunked
            # prefill the value needed to steer position 0 does not
            # exist until the final chunk — so this sizing is what makes
            # that arm servable at all. The wiring refuses (never
            # mis-steers) if a chunk happens anyway, but a refused run
            # is still a wasted run. Resolved values are asserted below.
            max_num_batched_tokens=max_num_batched_tokens,
            # ours FIRST: the vllm-lens plugin injects its own class
            # only `if not self.worker_extension_cls`, so this occupies
            # vLLM's single slot while inheriting the stock RPC surface.
            worker_extension_cls=lw.WORKER_EXTENSION_CLS,
            # enforce_eager deliberately NOT passed — the live config
            # must show the PLUGIN forced it (and lens_worker preflight
            # refuses otherwise).
        )
    except Exception as e:
        return dump(EXIT_ENGINE_INIT, error=f"{type(e).__name__}: {e}")
    Path(str(out_path) + ".engine_ready").write_text(str(time.time()))

    engine = _live_engine_evidence(llm)
    result["engine"] = engine
    problems: List[str] = []

    # THE KV-COVERAGE HALF OF THE GEOMETRY (kv_coverage_problem): read
    # the RESOLVED cache size off the workers themselves and assert the
    # full no-chunking budget is KV-backed — trivially true at 2048/32
    # (793,952 tokens against a 65,536 budget in job 227308), and the
    # entire point at the PETRI geometry, where the requested 32-way
    # concurrency at the default utilization exists in argv only.
    try:
        rpc = llm.collective_rpc(_worker_kv_cache_facts)
        kv_facts = dict(rpc[0] if isinstance(rpc, (list, tuple)) else rpc)
    except Exception as e:
        kv_facts = {"num_gpu_blocks": None, "block_size": None,
                    "read_error": f"{type(e).__name__}: {e}"}
    kv_problem = kv_coverage_problem(kv_facts, max_num_seqs=max_num_seqs,
                                     max_model_len=max_model_len)
    kv_block: Dict[str, Any] = dict(kv_facts)
    n_groups = kv_facts.get("kv_num_cache_groups")
    n_b, b_s = kv_facts.get("num_gpu_blocks"), kv_facts.get("block_size")
    mc = kv_facts.get("kv_max_concurrency")
    if (n_groups == 1 and isinstance(n_b, int)
            and not isinstance(n_b, bool) and n_b > 0
            and isinstance(b_s, int) and not isinstance(b_s, bool)
            and b_s > 0):
        # recorded whether coverage holds or not — a REFUSED phase must
        # still show the resolved numbers (job 229194's refusal printed
        # "None tokens" here while the real 521,520 lived only in the
        # problem string). Uniform cache only: here the token
        # arithmetic is real.
        kv_tokens = n_b * b_s
        kv_block.update({
            "kv_cache_tokens": kv_tokens,
            "full_length_concurrency": round(kv_tokens / max_model_len, 2)})
    elif (isinstance(n_groups, int) and not isinstance(n_groups, bool)
            and n_groups > 1 and isinstance(mc, (int, float))
            and not isinstance(mc, bool) and 0 < mc < float("inf")):
        # hybrid cache: num_gpu_blocks x block_size is NOT token
        # capacity (LCM scheduler block size; Mamba/GDN state pages) —
        # record the allocator's own verdict under the same field
        # names, so the numbers a reader sees are the honest ones.
        kv_block.update({
            "kv_cache_tokens": int(mc * max_model_len),
            "full_length_concurrency": round(float(mc), 2)})
    kv_block["covers_full_budget"] = kv_problem is None
    if kv_problem is not None:
        problems.append(kv_problem)
    result["geometry"]["kv_cache"] = kv_block

    if engine["model"] is not None:
        got_id, got_rev = canonical_model_identity(engine["model"])
        want_id, _ = canonical_model_identity(args.model)
        if got_id != want_id:
            problems.append(
                f"resolved model identity {got_id!r} != configured "
                f"{want_id!r} — refusing to serve a model nobody pinned")
        elif got_rev is not None and args.revision and got_rev != args.revision:
            problems.append(
                f"resolved model revision {got_rev} != pinned "
                f"{args.revision} — same repo, WRONG snapshot")
    if engine["worker_extension_cls"] != lw.WORKER_EXTENSION_CLS:
        problems.append(
            f"worker_extension_cls resolved to "
            f"{engine['worker_extension_cls']!r}, not "
            f"{lw.WORKER_EXTENSION_CLS!r} — the steering extension did not "
            "occupy vLLM's single extension slot")
    if engine["enforce_eager"] is not True:
        problems.append(f"enforce_eager resolved to "
                        f"{engine['enforce_eager']!r}, not True — the lens "
                        "plugin did not force eager; layer hooks would "
                        "silently never fire")
    if engine["prefix_caching_enabled"] is not False:
        problems.append(f"prefix caching resolved to "
                        f"{engine['prefix_caching_enabled']!r}, not False")
    if engine["max_num_batched_tokens"] != max_num_batched_tokens:
        problems.append(
            f"max_num_batched_tokens resolved to "
            f"{engine['max_num_batched_tokens']!r}, not the requested "
            f"{max_num_batched_tokens} — a silent clamp re-enables "
            "mid-prompt chunking, which alphasteer cannot survive")
    if engine["max_num_seqs"] != max_num_seqs:
        problems.append(f"max_num_seqs resolved to "
                        f"{engine['max_num_seqs']!r}, not {max_num_seqs}")
    if engine["max_model_len"] != max_model_len:
        problems.append(f"max_model_len resolved to "
                        f"{engine['max_model_len']!r}, not {max_model_len}")
    if problems:
        return dump(EXIT_REQUESTS, error="engine config assertions failed: "
                    + " | ".join(problems))

    result["stage"] = "install"
    try:
        rpc = llm.collective_rpc("sisteer_install_steering",
                                 args=(str(args.registry),))
        # the PRODUCTION readiness comparison first (launcher expectation
        # vs every worker's report — the exact functions launch_server
        # runs); the bespoke config cross-checks below ride on top of it,
        # never instead of it (production_install_check's docstring says
        # why).
        checked = production_install_check(args.registry, rpc)
        result["install_expected"] = checked["expected"]
        install = checked["reports"][0]
        result["install"] = install
        got_layers = sorted(int(l) for l in install["layers"])
        want_layers = sorted(int(l) for l in arms_cfg["expected_hook_layers"])
        if got_layers != want_layers:
            return dump(EXIT_REQUESTS, error=(
                f"install hooked layers {got_layers} != expected "
                f"{want_layers} (the site-shifted set the adapters "
                "computed at export) — the resid_pre index shift did not "
                "survive the worker path"))
        if sorted(install["ledger_cells"]) != sorted(ledger_cells):
            return dump(EXIT_REQUESTS, error=(
                f"install ledger_cells {sorted(install['ledger_cells'])} != "
                f"expected {sorted(ledger_cells)}"))
    except Exception as e:
        import traceback
        return dump(EXIT_REQUESTS, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    result["stage"] = "requests"
    runs: Dict[str, Any] = {}
    result["runs"] = runs
    try:
        tok = llm.get_tokenizer()
        # the engine's tokenizer is FRESH — the parent process's
        # thinking-off pin does not travel into this worker, and an
        # unpinned reasoning template (Qwen3.5) renders thinking-ON,
        # deriving prompts whose tails do not match the fits' H=7
        # assistant header. Pin it here, same seam as everywhere else.
        from sisteer.models import pin_reasoning_template_off as _pin_think
        result["thinking_pin_applied"] = _pin_think(tok)
        # ONE served prompt table: the probe prompts first (indices
        # 0..len(PROMPTS)-1 — the only indices any plan or any piece of
        # evidence ever names), then the unsteered filler at
        # len(PROMPTS)+k, which only reference batches ever request.
        probe_ids = _chat_prompt_ids(tok, PROMPTS)
        filler_ids = _chat_prompt_ids(tok, FILLER_PROMPTS)
        prompt_ids = probe_ids + filler_ids
        result["prompt_token_lens"] = [len(p) for p in probe_ids]
        result["filler_token_lens"] = [len(p) for p in filler_ids]

        def run_batch(name: str, rows: Sequence[Tuple[int, Optional[str]]],
                      *, max_tokens: int = SMOKE_MAX_TOKENS,
                      ignore_eos: bool = False,
                      table: Optional[Sequence[Sequence[int]]] = None
                      ) -> Tuple[List[Any], float]:
            # ``table`` is the prompt id table the row indices name. It
            # defaults to the fidelity table; the shared-prefix
            # throughput legs pass their own, so one runner serves both
            # phases and the ledger arming, the served-length check and
            # the timing are the same code in each.
            tbl = prompt_ids if table is None else table
            tracked = [len(tbl[p]) for p, s in rows
                       if s is not None and s in ledger_cells]
            prompts = [{"prompt_token_ids": list(tbl[p])}
                       for p, _s in rows]
            params = [_greedy_params(max_tokens, ignore_eos=ignore_eos,
                                     selector=selectors.get(s) if s else None)
                      for _p, s in rows]
            if tracked:
                llm.collective_rpc("sisteer_begin_run", args=(tracked,))
            try:
                t0 = time.perf_counter()
                outs = llm.generate(prompts, params)
                wall = time.perf_counter() - t0
                if tracked:
                    llm.collective_rpc("sisteer_finish_run")
            except BaseException:
                # Exception-path cleanup ONLY (never the success path):
                # a generate() that died for a NON-steering reason would
                # otherwise leave the run's credits armed, and the next
                # tracked run_batch would fail with a misleading
                # "begin_run while a run is already armed" instead of
                # its own real error (the response phase continues past
                # per-arm failures, so this leak compounded). disarm
                # drops credits WITHOUT the run-end check and never
                # condemns; a genuine steering failure has already
                # condemned the worker, which abort_run deliberately
                # does not touch — and the abort RPC itself is
                # best-effort (a condemned/odd engine must not mask the
                # original exception).
                if tracked:
                    try:
                        llm.collective_rpc("sisteer_abort_run")
                    except Exception:
                        pass
                raise
            rec = []
            for (p, _s), o in zip(rows, outs):
                n_prompt = len(o.prompt_token_ids)
                if n_prompt != len(tbl[p]):
                    raise RuntimeError(
                        f"run {name}: engine reports {n_prompt} prompt "
                        f"tokens for a {len(tbl[p])}-token "
                        "TokensPrompt — the engine re-tokenized or altered "
                        "the prompt; the ledger arming would be wrong")
                rec.append({"tokens": [int(t) for t in o.outputs[0].token_ids],
                            "n_prompt": n_prompt,
                            # the engine's own id for this row, which is
                            # what the worker's selection record is keyed
                            # by — the only way to correlate a submitted
                            # row with what the worker BOUND to it.
                            "req_id": getattr(o, "request_id", None)})
            runs[name] = rec
            return outs, wall

        if args.phase == "throughput":
            # THROUGHPUT ONLY, on the shared-prefix geometry. No fidelity
            # evidence is gathered here at all, so the gate can never
            # depend on a geometry it was not certified under.
            result["stage"] = "throughput"
            result["throughput_legs"] = _lens_shared_prefix_legs(
                tok, run_batch, arm_ids, response_arm_ids, header_ids,
                result["geometry"], result)
            result["runs"] = {}   # throughput token ids are never evidence
            return dump(0, stage="completed")

        all_rows = [(i, None) for i in range(len(PROMPTS))]

        if args.phase == "response":
            # THE CROSS-PHASE CELL, IN A PROCESS OF ITS OWN. This is not
            # tidiness — it is the only way "advisory" can be true. A
            # steering failure CONDEMNS the worker
            # (LensWorkerSteering._condemn): every later call refuses,
            # so a response-mask refusal inside the hook would take down
            # whatever else that engine was going to serve. The response
            # mask has never run on hardware; putting it in its own
            # engine means the worst case costs these runs and nothing
            # else, and the fidelity gate keeps the exact composition it
            # passed with.
            result["stage"] = "response_arm"
            run_batch("baseline", all_rows)
            served: List[str] = []
            bites: Dict[str, Any] = {}
            twins: Dict[str, Any] = {}
            errors: Dict[str, Any] = {}
            donors = dict(arms_cfg.get("response_arm_donors") or {})
            for arm in response_arm_ids:
                donor = donors.get(arm)
                try:
                    run_batch(f"steered__{arm}",
                              [(i, arm) for i in range(len(PROMPTS))])
                    run_batch(f"scale0__{arm}",
                              [(i, scale0_cell_id(arm))
                               for i in range(len(PROMPTS))])
                    if donor and f"steered__{donor}" not in runs:
                        run_batch(f"steered__{donor}",
                                  [(i, donor) for i in range(len(PROMPTS))])
                except Exception as e:
                    import traceback
                    errors[arm] = {"error": f"{type(e).__name__}: {e}",
                                   "traceback": traceback.format_exc()}
                    print(f"\n!! RESPONSE-MODE ARM {arm} RAISED on its first "
                          f"hardware serving:\n   {type(e).__name__}: {e}\n"
                          "   An assistant-header mismatch at prefill is a "
                          "template/render disagreement, not a steering "
                          "failure. This engine is now condemned, but it "
                          "serves NOTHING ELSE — the fidelity gate ran in a "
                          "different process and is unaffected.\n",
                          flush=True)
                    break
                served.append(arm)
                st = [r["tokens"] for r in runs[f"steered__{arm}"]]
                z0 = [r["tokens"] for r in runs[f"scale0__{arm}"]]
                base = [r["tokens"] for r in runs["baseline"]]
                bites[arm] = {
                    "n": len(st),
                    "n_differing": sum(1 for a, b in zip(st, z0) if a != b),
                    "scale0_identical_to_unsteered": sum(
                        1 for a, b in zip(z0, base) if a == b)}
                if donor:
                    ap = [r["tokens"] for r in runs[f"steered__{donor}"]]
                    twins[arm] = {
                        "donor": donor, "n": len(st),
                        "n_differing": sum(1 for a, b in zip(st, ap)
                                           if a != b)}
            result["response_served_arm_ids"] = served
            result["response_bites"] = bites
            result["response_vs_allposition"] = twins
            result["response_errors"] = errors
            result["fluency"] = {
                name: [fluency_stats(r["tokens"]) for r in runs[name]]
                for name in sorted(runs)}

            print("\n" + "=" * 72)
            print("POSITIONS_RESPONSE — FIRST EXECUTION ON GPU. Same bytes "
                  "as the all-position arm,")
            print("served under the response mask (assistant header at "
                  "prefill + every generated")
            print("position). Judge on the SAME TWO AXES as the main "
                  "eyeball section: a refusal is")
            print("the arm working; repetition or word salad is the bug.")
            print("=" * 72)
            for i in range(min(4, len(PROMPTS))):
                print(f"\nPROMPT: {PROMPTS[i]}")
                print("  BASE        : "
                      f"{tok.decode(runs['baseline'][i]['tokens'], skip_special_tokens=True).strip()[:240]}")
                for arm in served:
                    donor = donors.get(arm)
                    if donor and f"steered__{donor}" in runs:
                        print(f"  {donor[:12]:<12}: "
                              f"{tok.decode(runs[f'steered__{donor}'][i]['tokens'], skip_special_tokens=True).strip()[:240]}")
                    print(f"  {'response':<12}: "
                          f"{tok.decode(runs[f'steered__{arm}'][i]['tokens'], skip_special_tokens=True).strip()[:240]}")
            print("\n" + "=" * 72, flush=True)
            for arm, d in bites.items():
                print(f"[smoke] response arm {arm}: {d['n_differing']}/"
                      f"{d['n']} differ from its scale-0 twin; "
                      f"{d['scale0_identical_to_unsteered']}/{d['n']} "
                      "scale-0 completions are identical to unsteered",
                      flush=True)
            for arm, d in twins.items():
                verdict = ("distinct" if d["n_differing"] else
                           "IDENTICAL — the mask changed nothing relative "
                           "to the all-position serving")
                print(f"[smoke] response arm {arm} vs all-position "
                      f"{d['donor']}: {d['n_differing']}/{d['n']} differ "
                      f"({verdict})", flush=True)

            # its legacy-suite throughput leg, so the ladder has a row
            # for it at the certified geometry too.
            tp_resp: Dict[str, Any] = {}
            n_req = len(PROMPTS) * TP_REPLICAS
            for arm in served:
                pol = leg_policy(
                    suite="legacy_short_distinct", prompt_set="legacy",
                    engine="lens (SisteerWorkerExtension installed)",
                    prefix_caching=False, cuda_graphs=False,
                    batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT",
                                                   "?"),
                    geometry=result["geometry"],
                    request_set={"batch_size": n_req,
                                 "generated_tokens_per_request":
                                     TP_MAX_TOKENS,
                                 "n_distinct_prompts": len(PROMPTS),
                                 "replicas": TP_REPLICAS,
                                 "shared_preamble_measured_tokens": 0})
                try:
                    rows = [(i % len(PROMPTS), arm) for i in range(n_req)]
                    run_batch(f"tp_warm__{arm}",
                              [(i % len(PROMPTS), None)
                               for i in range(n_req)],
                              max_tokens=TP_WARMUP_TOKENS, ignore_eos=True)
                    outs, wall = run_batch(f"tp__{arm}", rows,
                                           max_tokens=TP_MAX_TOKENS,
                                           ignore_eos=True)
                    tp_resp[f"lens_steered__{arm}"] = throughput_leg(
                        outs, wall, n_req, pol)
                except Exception as e:
                    tp_resp[f"lens_steered__{arm}"] = missing_leg(
                        f"{type(e).__name__}: {e}", pol)
                runs.pop(f"tp_warm__{arm}", None)
                runs.pop(f"tp__{arm}", None)
            result["throughput_legs"] = tp_resp
            return dump(0, stage="completed")

        # THE FIDELITY PHASE: per-arm evidence (steered__/scale0__ runs,
        # batch = len(PROMPTS) rows) spans EVERY fidelity-served arm —
        # the full arm set minus the response-isolated ones (those are
        # served by --phase response in a separate process; they load
        # here but nothing selects one). The ONE mixed batch is the only
        # thing that uses ``mixed_arm_ids`` — a floor-validated SUBSET
        # when the geometry cannot hold 3+4k rows (validate_mixed_plan).
        # This phase MUST NOT narrow arm_ids to the mixed subset: it did
        # once (2026-08-30, gate 46058), which silently dropped the
        # per-arm steered/scale-0 evidence for non-mixed arms and the
        # assembler refused on the missing steered__<arm> key — the
        # right refusal, wrong cause. steering_bites' per-arm evidence
        # and mixed-batch membership are DIFFERENT properties; only the
        # second is legitimately sampled.
        arm_ids = [a for a in arm_ids if a not in response_arm_ids]
        result["arm_ids"] = arm_ids
        stray_mixed = sorted(set(mixed_arm_ids) - set(arm_ids))
        if stray_mixed:
            return dump(EXIT_REQUESTS, error=(
                f"mixed_arm_ids {stray_mixed} are not fidelity-served "
                "arms (response-isolated or unknown) — the mixed batch "
                "must be a subset of the arms this engine serves"))

        # R0: unselected baseline (hooks armed, nothing selected).
        run_batch("baseline", all_rows)
        # per arm: full-set steered + full-set scale-0 twin.
        for arm in arm_ids:
            run_batch(f"steered__{arm}",
                      [(i, arm) for i in range(len(PROMPTS))])
            run_batch(f"scale0__{arm}",
                      [(i, scale0_cell_id(arm)) for i in range(len(PROMPTS))])
        # the mixed batch, twice (identical submission -> determinism),
        # then one SAME-SIZE reference batch per row. NEVER a solo
        # serving: on this engine a batch-of-1 reference reports
        # contamination that is not there (module docstring, SAME-SIZE
        # REFERENCES; REFERENCE_SIZING_NOTE, which ships in the record).
        #
        # Built from ``mixed_arm_ids``, which EXCLUDES the response-mode
        # arm: it is advisory and brand new, and holding the gate's own
        # batch at the exact composition that passed means a first-ever
        # mask execution cannot perturb engine_determinism,
        # batch_invariance or selection_honored. The arm still gets its
        # own steered and scale-0 runs above, so its bite count and its
        # scale-0 no-op claim are both real evidence.
        plan = build_mixed_plan(mixed_arm_ids, len(PROMPTS))
        n_mixed = len(plan)
        validate_mixed_plan(
            plan, max_num_seqs,
            arm_hook_sites=arms_cfg.get("arm_hook_sites"),
            all_arm_ids=arm_ids)
        result["mixed_plan"] = [[p, s] for p, s in plan]
        result["mixed_batch_size"] = n_mixed
        result["reference_sizing"] = REFERENCE_SIZING_NOTE
        # ignore_eos on the size-fixed batches: a row that stops at EOS
        # leaves the running batch, and the two compositions' EOS
        # timings differ by construction, so without this the batch
        # would shrink to DIFFERENT sizes part way through the very
        # comparison that holds size fixed.
        # THE DIRECT HALF of selection_honored (SELECTION_DIRECT_NOTE):
        # arm the worker's own selection record, serve the mixed batch,
        # then read back WHICH OBJECT the worker resolved and bound for
        # each row. Scoped to mixed_run1 by the reset, so the read is
        # about this batch and not the whole run. Everything here is
        # advisory-by-construction on the serving side — the record is
        # observation only in the worker, and a failure to read it is
        # recorded MISSING rather than substituted for by the inferred
        # half.
        select_arm = "armed: the worker's selection record was reset"
        try:
            _selection_record_rpc(llm, reset=True)
        except Exception as e:
            select_arm = (f"MISSING: the worker's selection record could not "
                          f"be reset before the mixed batch — "
                          f"{type(e).__name__}: {e}")
            print(f"[smoke] {select_arm}", flush=True)
        run_batch("mixed_run1", plan, ignore_eos=SIZE_FIXED_IGNORE_EOS)
        mixed_record: Optional[Dict[str, Any]] = None
        try:
            mixed_record = _selection_record_rpc(llm, reset=True)
        except Exception as e:
            if not select_arm.startswith("MISSING"):
                select_arm = (f"MISSING: the worker's selection record could "
                              f"not be read after the mixed batch — "
                              f"{type(e).__name__}: {e}")
            print(f"[smoke] {select_arm}", flush=True)
        result["selection_direct"] = selection_direct_report(
            plan, runs.get("mixed_run1") or [], mixed_record, select_arm)

        run_batch("mixed_run2", plan, ignore_eos=SIZE_FIXED_IGNORE_EOS)
        ref_rows: List[Dict[str, Any]] = []
        ref_batches: List[Dict[str, Any]] = []
        for i in range(n_mixed):
            rows = build_reference_rows(plan, i, len(PROMPTS),
                                        len(FILLER_PROMPTS))
            if len(rows) != n_mixed or tuple(rows[i]) != tuple(plan[i]):
                raise RuntimeError(
                    f"reference batch for plan row {i} is {len(rows)} rows "
                    f"against a {n_mixed}-row mixed batch, or does not "
                    "carry that row at that position — the equal-size rule "
                    "is the whole point of this comparison; refusing")
            outs, _ = run_batch(f"reference__{i}", rows,
                                ignore_eos=SIZE_FIXED_IGNORE_EOS)
            # the N-1 filler rows are scaffolding, never evidence; only
            # the row under test is kept (the pop also keeps the JSON to
            # N rows instead of N^2). Same idiom as the tp__ runs below.
            # outs is in submission order, and run_batch has already
            # checked EVERY row's served prompt length against the row it
            # submitted — which is also what would catch a permuted outs
            # before position i were read as the row under test.
            runs.pop(f"reference__{i}", None)
            o = outs[i]
            ref_rows.append({"tokens": [int(t) for t in
                                        o.outputs[0].token_ids],
                             "n_prompt": len(o.prompt_token_ids)})
            ref_batches.append({"plan_row": i, "row_under_test_index": i,
                                "batch_size": len(rows),
                                "n_unsteered_filler": len(rows) - 1})
        runs["reference"] = ref_rows
        result["reference_batches"] = ref_batches

        # the SAME direct check over everything the comparison rests on
        # besides the mixed batch itself — mixed_run2 and every same-size
        # reference serving. A reference batch that mis-routed would make
        # selection_honored's output comparison meaningless in the other
        # direction, and this costs one RPC to rule out.
        try:
            after = _selection_record_rpc(llm, reset=True)
            result["selection_direct_references"] = {
                "scope": ("mixed_run2 plus every same-size reference "
                          "serving — the other side of the comparison "
                          "selection_honored makes"),
                "n_observations": after.get("n_observations"),
                "n_mismatched": after.get("n_mismatched"),
                "all_observations_matched": after.get(
                    "all_observations_matched"),
                "n_record_errors": after.get("n_record_errors"),
                "n_unstable_requests": after.get("n_unstable_requests"),
                "pairs": after.get("pairs")}
        except Exception as e:
            result["selection_direct_references"] = {
                "scope": "mixed_run2 plus every same-size reference serving",
                "status": f"MISSING: {type(e).__name__}: {e}"}

        # fluency diagnostics (numbers only).
        result["fluency"] = {
            name: [fluency_stats(r["tokens"]) for r in runs[name]]
            for name in ["baseline"] + [f"steered__{a}" for a in arm_ids]}

        # eyeball section — job log ONLY, never any JSON.
        print("\n" + "=" * 72)
        print("STEERED vs UNSTEERED COMPLETIONS (log-only; judge on TWO "
              "separate axes:")
        print("  1. refusal — static_r_matched at lambda=0.05 legitimately "
              "refuses ~21% of")
        print("     benign prompts (XSTest 0.788 vs 0.924 unsteered); a "
              "refusal is the arm")
        print("     WORKING, not a bug;")
        print("  2. degeneracy — repetition loops, truncation into noise, "
              "word salad. THAT")
        print("     is the real failure, and the two verdicts must not be "
              "collapsed.)")
        print("=" * 72)

        def text(name: str, i: int) -> str:
            toks = runs[name][i]["tokens"]
            return tok.decode(toks, skip_special_tokens=True).strip()

        for i in range(min(4, len(PROMPTS))):
            print(f"\nPROMPT: {PROMPTS[i]}")
            print(f"  BASE        : {text('baseline', i)[:240]}")
            for arm in arm_ids:
                print(f"  {arm[:12]:<12}: {text(f'steered__{arm}', i)[:240]}")
        print("\n" + "=" * 72, flush=True)

        for arm in arm_ids:
            steered = [r["tokens"] for r in runs[f"steered__{arm}"]]
            zero = [r["tokens"] for r in runs[f"scale0__{arm}"]]
            n_diff = sum(1 for a, b in zip(steered, zero) if a != b)
            print(f"[smoke] arm {arm}: {n_diff}/{len(steered)} completions "
                  f"changed vs its scale-0 twin", flush=True)

    except Exception as e:
        import traceback
        return dump(EXIT_REQUESTS, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())

    # the LEGACY lens throughput legs — unchanged from the passing run
    # (8 distinct short prompts x 4 replicas, 128 generated tokens, on
    # the certified geometry), so they stay comparable to job 173580.
    # Secondary: an error here is recorded, not fatal.
    result["stage"] = "throughput"
    n_req = len(PROMPTS) * TP_REPLICAS
    legacy_geom = {
        "batch_size": n_req,
        "generated_tokens_per_request": TP_MAX_TOKENS,
        "n_distinct_prompts": len(PROMPTS),
        "replicas": TP_REPLICAS,
        "shared_preamble_target_tokens": None,
        "shared_preamble_measured_tokens": 0,
        "prompt_token_lens": result.get("prompt_token_lens"),
        "warmup": ("the SAME request set, so a cache-enabled comparator "
                   "on this suite sees full-prompt hits — an upper bound "
                   "on reuse at a ~40-token prompt, not the consumer's "
                   "steady state; that is what the shared_prefix suite "
                   "is for")}

    def legacy_policy() -> Dict[str, Any]:
        return leg_policy(
            suite="legacy_short_distinct", prompt_set="legacy",
            engine="lens (SisteerWorkerExtension installed)",
            prefix_caching=False, cuda_graphs=False,
            batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "?"),
            geometry=result["geometry"], request_set=legacy_geom)

    tp: Dict[str, Any] = {}
    try:
        tp_rows_for = lambda sel: [(i % len(PROMPTS), sel)  # noqa: E731
                                   for i in range(n_req)]
        run_batch("tp_warmup", tp_rows_for(None),
                  max_tokens=TP_WARMUP_TOKENS, ignore_eos=True)
        for tag, sel in ([("lens_unselected", None)]
                         + [(f"lens_steered__{a}", a) for a in arm_ids]):
            try:
                outs, wall = run_batch(f"tp__{tag}", tp_rows_for(sel),
                                       max_tokens=TP_MAX_TOKENS,
                                       ignore_eos=True)
                tp[tag] = throughput_leg(outs, wall, n_req, legacy_policy())
            except Exception as e:
                # one arm's leg failing must not cost the others theirs
                # — the response-mode arm is new, and its mask can raise.
                tp[tag] = missing_leg(f"{type(e).__name__}: {e}",
                                      legacy_policy())
            runs.pop(f"tp__{tag}", None)  # token ids of tp runs: not evidence
        runs.pop("tp_warmup", None)
    except Exception as e:
        result["throughput_error"] = f"{type(e).__name__}: {e}"
    result["throughput_legs"] = tp
    # the pre-existing key, kept so anything reading the old record
    # shape still finds the legacy lens legs where they have always been.
    result["throughput"] = tp

    return dump(0, stage="completed")


# --- worker-base: the plugin-disabled comparator engines --------------------------


def cmd_worker_base(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    mode = args.mode
    result: Dict[str, Any] = {"stage": "preflight", "mode": mode,
                              "packages": _reported_packages(),
                              "env": {k: os.environ.get(k) for k in (
                                  "VLLM_BATCH_INVARIANT",
                                  "VLLM_LENS_DISABLE")}}

    def dump(code: int, **extra: Any) -> int:
        result.update(extra)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       default=str))
        return code

    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()  # must no-op under VLLM_LENS_DISABLE=1
    except Exception as e:
        result["forced_load_error"] = f"{type(e).__name__}: {e}"
    result["plugin"] = _patch_state()
    if result["plugin"].get("generate_patched") or result["plugin"].get(
            "engine_args_patched"):
        return dump(EXIT_REQUESTS, error=(
            "the lens plugin is PATCHED in a VLLM_LENS_DISABLE=1 process — "
            "this comparator would not measure base vLLM; refusing"))

    # THE SINGLE-FLAG LADDER (module docstring, THROUGHPUT). Three
    # policy points, each one flag from its neighbour, so the walk from
    # stock vLLM to the lens engine's configuration can be SUBTRACTED
    # rather than inferred. Batch invariance is the fourth flag and is
    # set by the parent through the environment, not here.
    want_eager = mode in ("eager", "eager_cache")
    want_pc = mode in ("graphs", "eager_cache")

    max_model_len = int(args.max_model_len)
    max_num_seqs = int(args.max_num_seqs)
    # RE-DERIVED, never inherited: the no-chunking guarantee is
    # max_num_batched_tokens = max_num_seqs * max_model_len, and it has
    # to hold on whichever geometry this comparator was asked for.
    max_num_batched_tokens = max_num_seqs * max_model_len
    geometry = {"max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "suite": args.tp_suite}
    result["geometry"] = geometry

    result["stage"] = "engine_init"
    kwargs: Dict[str, Any] = dict(
        model=args.model, revision=args.revision,
        tokenizer_revision=args.revision, dtype="bfloat16", seed=1234,
        max_model_len=max_model_len, gpu_memory_utilization=0.85,
        max_num_seqs=max_num_seqs,
        # same scheduler geometry as the lens engine running the same
        # suite, so the comparator differs ONLY in the policy axes it
        # names.
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=want_eager, enable_prefix_caching=want_pc)
    try:
        from vllm import LLM
        llm = LLM(**kwargs)
    except Exception as e:
        return dump(EXIT_ENGINE_INIT, error=f"{type(e).__name__}: {e}")
    Path(str(out_path) + ".engine_ready").write_text(str(time.time()))

    engine = _live_engine_evidence(llm)
    result["engine"] = engine
    problems: List[str] = []
    if engine["model"] is not None and engine["model"] != args.model:
        problems.append(f"resolved model {engine['model']!r} != configured "
                        f"{args.model!r}")
    if engine["enforce_eager"] is not want_eager:
        problems.append(
            f"mode {mode} asked for enforce_eager={want_eager} but the live "
            f"config resolved to {engine['enforce_eager']!r} — this rung of "
            "the ladder would not be the comparator it claims. Refusing "
            "rather than mislabeling it: if VLLM_BATCH_INVARIANT forced "
            "eager here, that IS the finding (batch invariance and CUDA "
            "graphs are then not separable on this stack)")
    if engine["prefix_caching_enabled"] is not want_pc:
        problems.append(f"mode {mode} asked for prefix caching {want_pc} but "
                        "the live config resolved to "
                        f"{engine['prefix_caching_enabled']!r}")
    if engine["max_num_batched_tokens"] != max_num_batched_tokens:
        problems.append(f"max_num_batched_tokens resolved to "
                        f"{engine['max_num_batched_tokens']!r}, not "
                        f"{max_num_batched_tokens}")
    if engine["max_num_seqs"] != max_num_seqs:
        problems.append(f"max_num_seqs resolved to "
                        f"{engine['max_num_seqs']!r}, not {max_num_seqs}")
    if engine["max_model_len"] != max_model_len:
        problems.append(f"max_model_len resolved to "
                        f"{engine['max_model_len']!r}, not {max_model_len}")
    if problems:
        return dump(EXIT_REQUESTS, error="comparator config assertions "
                    "failed: " + " | ".join(problems))

    binv = os.environ.get("VLLM_BATCH_INVARIANT", "?")
    result["stage"] = "throughput"
    legs: Dict[str, Any] = {}
    try:
        tok = llm.get_tokenizer()
        # the engine's tokenizer is FRESH — the parent process's
        # thinking-off pin does not travel into this worker, and an
        # unpinned reasoning template (Qwen3.5) renders thinking-ON,
        # deriving prompts whose tails do not match the fits' H=7
        # assistant header. Pin it here, same seam as everywhere else.
        from sisteer.models import pin_reasoning_template_off as _pin_think
        result["thinking_pin_applied"] = _pin_think(tok)

        def timed(prompt_rows: Sequence[Sequence[int]],
                  warm_rows: Sequence[Sequence[int]],
                  max_tokens: int) -> Tuple[List[Any], float]:
            warm = [{"prompt_token_ids": list(r)} for r in warm_rows]
            llm.generate(warm, [_greedy_params(TP_WARMUP_TOKENS,
                                               ignore_eos=True)
                                for _ in warm])
            rows = [{"prompt_token_ids": list(r)} for r in prompt_rows]
            params = [_greedy_params(max_tokens, ignore_eos=True)
                      for _ in rows]
            t0 = time.perf_counter()
            outs = llm.generate(rows, params)
            return outs, time.perf_counter() - t0

        if args.tp_suite == "legacy":
            # UNCHANGED from the passing run, deliberately: 8 distinct
            # short prompts x 4 replicas at 128 tokens, warmed on the
            # same set. It is the control the shared-prefix suite is
            # contrasted against, and the only thing new about it here
            # is that the record now says which policies it ran under.
            prompt_ids = _chat_prompt_ids(tok, PROMPTS)
            rows = [prompt_ids[i % len(PROMPTS)]
                    for i in range(len(PROMPTS) * TP_REPLICAS)]
            outs, wall = timed(rows, rows, TP_MAX_TOKENS)
            legs["legacy"] = throughput_leg(outs, wall, len(rows), leg_policy(
                suite="legacy_short_distinct", prompt_set="legacy",
                engine=f"base vLLM, plugin disabled (mode {mode})",
                prefix_caching=want_pc, cuda_graphs=not want_eager,
                batch_invariant=binv, geometry=geometry,
                request_set={
                    "batch_size": len(rows),
                    "generated_tokens_per_request": TP_MAX_TOKENS,
                    "n_distinct_prompts": len(PROMPTS),
                    "replicas": TP_REPLICAS,
                    "shared_preamble_target_tokens": None,
                    "shared_preamble_measured_tokens": 0,
                    "prompt_token_lens": [len(p) for p in prompt_ids],
                    "warmup": ("the SAME request set — a cache-enabled rung "
                               "here sees full-prompt hits on ~40-token "
                               "prompts, which is an upper bound on reuse "
                               "rather than the consumer's steady state")}))
        else:
            sets = build_sp_prompt_sets(tok)
            result["shared_prefix_sets"] = {
                name: {k: v for k, v in s.items()
                       if k not in ("prompt_ids", "warmup_ids")}
                for name, s in sets.items()}
            for set_name, s in sets.items():
                policy = leg_policy(
                    suite="shared_prefix", prompt_set=set_name,
                    engine=f"base vLLM, plugin disabled (mode {mode})",
                    prefix_caching=want_pc, cuda_graphs=not want_eager,
                    batch_invariant=binv, geometry=geometry,
                    request_set=_sp_request_geometry(s))
                try:
                    outs, wall = timed(s["prompt_ids"], s["warmup_ids"],
                                       SP_GEN_TOKENS)
                    legs[set_name] = throughput_leg(
                        outs, wall, len(s["prompt_ids"]), policy)
                except Exception as e:
                    legs[set_name] = missing_leg(f"{type(e).__name__}: {e}",
                                                 policy)
    except Exception as e:
        import traceback
        return dump(EXIT_REQUESTS, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())
    result["throughput_legs"] = legs
    if args.tp_suite == "legacy" and "legacy" in legs:
        # the pre-existing key, where the old record shape put it.
        result["throughput"] = legs["legacy"]
    return dump(0, stage="completed")


# --- the parent orchestrator -------------------------------------------------------


def _child_env(*, disable_lens: bool, batch_invariant: str) -> Dict[str, str]:
    env = dict(os.environ)
    # vllm filters general plugins by ENTRY-POINT NAME ("activations");
    # an inherited VLLM_PLUGINS=vllm_lens would silently filter the
    # plugin OUT (the probe's documented trap) — pop it.
    env.pop("VLLM_PLUGINS", None)
    env.pop("VLLM_LENS_DISABLE", None)
    if disable_lens:
        env["VLLM_LENS_DISABLE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process worker
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"        # the wiring reads V1
    env["VLLM_BATCH_INVARIANT"] = batch_invariant
    env["TOKENIZERS_PARALLELISM"] = "false"
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (src + ":" + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else src)
    return env


def _run_child(subcmd: str, argv: List[str], env: Dict[str, str],
               out_json: Path) -> Tuple[Optional[int], Dict[str, Any]]:
    for stale in (out_json, Path(str(out_json) + ".engine_ready")):
        if stale.exists():
            stale.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()), subcmd, *argv]
    print(f"[smoke] launching {subcmd}: {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, env=env, timeout=CHILD_TIMEOUT_S)
        rc: Optional[int] = proc.returncode
    except subprocess.TimeoutExpired:
        rc = None
    data: Dict[str, Any] = {}
    if out_json.is_file():
        try:
            data = json.loads(out_json.read_text())
        except Exception as e:
            data = {"stage": "unreadable", "error": str(e)}
    return rc, data


#: the gate's checks in the order the console prints them (the module
#: docstring's order of importance, with engine_determinism — the
#: repeat-determinism test, and the BI=0 headline — first).
FIDELITY_CHECK_ORDER = ("engine_determinism", "scale0_noop",
                        "steering_bites", "batch_invariance",
                        "selection_honored")


def _fidelity_child_argv(args: argparse.Namespace, registry_dir: Path,
                         arms_json_path: Path, out_json: Path, *,
                         max_model_len: int = MAX_MODEL_LEN,
                         max_num_seqs: int = MAX_NUM_SEQS,
                         gpu_memory_utilization: float =
                         SMOKE_GPU_MEMORY_UTILIZATION) -> List[str]:
    """The fidelity child's argv — IDENTICAL between the two BI phases
    (only VLLM_BATCH_INVARIANT differs in the environment, which is the
    whole point: one variable, nothing else). The PETRI-geometry phase
    passes its own geometry explicitly and differs in exactly these
    three flags — same registry, same arms, same prompts."""
    return ["--model", args.model, "--revision", args.revision,
            "--registry", str(registry_dir),
            "--arms-json", str(arms_json_path), "--out", str(out_json),
            "--phase", "fidelity", "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", str(gpu_memory_utilization)]


def run_fidelity_phase(*, args: argparse.Namespace, out_dir: Path,
                       registry_dir: Path, arms_json_path: Path,
                       batch_invariant: str, out_name: str, label: str,
                       purpose: str, gating: bool,
                       max_model_len: int = MAX_MODEL_LEN,
                       max_num_seqs: int = MAX_NUM_SEQS,
                       gpu_memory_utilization: float =
                       SMOKE_GPU_MEMORY_UTILIZATION
                       ) -> Tuple[str, Dict[str, Any], Any, Any,
                                  Dict[str, Any]]:
    """One whole fidelity phase: launch the lens child at a given
    ``VLLM_BATCH_INVARIANT``, assemble its gatherings into the committed
    evidence schema, and evaluate the five-check gate over them.

    Returns ``(outcome, phase, report, evidence, child)`` where
    ``outcome`` is one of ``unavailable`` / ``child_failed`` /
    ``evidence_failed`` / ``evaluated``. THE CALLER decides what an
    outcome costs: the gating phase keeps the outcome discipline it has
    always had (marker + exit code), and a probe phase (BI=1, the
    32-way re-measure) records the same structure and can never change
    either. Nothing in here writes a
    marker or exits."""
    from sisteer.capeval.serving.fidelity import evaluate_fidelity

    json_path = out_dir / out_name
    print(f"\n[smoke] FIDELITY PHASE [{label}] — "
          f"VLLM_BATCH_INVARIANT={batch_invariant}", flush=True)
    rc, child = _run_child(
        "worker-steered",
        _fidelity_child_argv(args, registry_dir, arms_json_path, json_path,
                             max_model_len=max_model_len,
                             max_num_seqs=max_num_seqs,
                             gpu_memory_utilization=gpu_memory_utilization),
        _child_env(disable_lens=False, batch_invariant=str(batch_invariant)),
        json_path)
    engine_up = Path(str(json_path) + ".engine_ready").exists()
    # the RESOLVED setting, read back out of the child's own environment
    # rather than assumed from what the parent asked for.
    resolved = (child.get("env") or {}).get("VLLM_BATCH_INVARIANT")
    phase: Dict[str, Any] = {
        "label": label,
        "purpose": purpose,
        "gating": bool(gating),
        "certifies": bool(gating),
        "governs_exit_status": bool(gating),
        "requested_batch_invariant": str(batch_invariant),
        "resolved_batch_invariant": resolved,
        "batch_invariant_as_requested": resolved == str(batch_invariant),
        "is_the_certified_configuration": (
            bool(gating)
            and str(batch_invariant) == CERTIFIED_BATCH_INVARIANT),
        "child_json": out_name,
        "child_exit": rc,
        "engine_reached_ready": engine_up,
        "geometry": child.get("geometry"),
        "resolved_engine": child.get("engine"),
        "lens_throughput_legs": child.get("throughput_legs") or {},
        "lens_tokens_per_s": lens_leg_rates(child.get("throughput_legs")),
        "selection_direct": child.get("selection_direct"),
        "selection_direct_references": child.get(
            "selection_direct_references"),
    }
    err = (f"child exit {rc}, stage {child.get('stage')}: "
           f"{child.get('error')}")
    if rc == EXIT_ENGINE_INIT or (rc is None and not engine_up):
        phase["outcome"] = "MISSING: the lens-active engine could not init"
        phase["error"] = err
        return "unavailable", phase, None, None, child
    if rc != 0:
        phase["outcome"] = ("MISSING: the engine came up but the run failed "
                            "before any evidence existed")
        phase["error"] = err
        return "child_failed", phase, None, None, child
    try:
        fid_ev = assemble_fidelity_evidence(child)
        phase["batch_sizing"] = reference_sizing_facts(child)
    except Exception as e:
        phase["outcome"] = ("MISSING: the worker's gatherings could not be "
                            "assembled into fidelity evidence")
        phase["error"] = f"{type(e).__name__}: {e}"
        return "evidence_failed", phase, None, None, child

    report = evaluate_fidelity(fid_ev)
    rd = report.as_dict()
    phase["outcome"] = "evaluated"
    phase["passed"] = rd["passed"]
    phase["invalidated"] = rd["invalidated"]
    phase["checks"] = rd["checks"]
    # counts AND first-divergence indices, per check — a repeat-
    # determinism failure that diverges at token 0 means something quite
    # different from one that diverges at token 40.
    phase["diagnostics"] = rd["diagnostics"]
    phase["failures"] = rd["failures"]
    phase.update(phase_check_split(rd, batch_invariant=str(batch_invariant),
                                   gating=gating))
    try:
        phase["advisory_arm_bites"] = advisory_bite_counts(child)
    except Exception as e:
        phase["advisory_arm_bites"] = {
            "status": f"MISSING: {type(e).__name__}: {e}"}
    return "evaluated", phase, report, fid_ev, child


def _diag_line(name: str, diagnostics: Mapping[str, Any]) -> str:
    """One check's divergence diagnostics, compressed to one line: how
    many comparisons diverged, and WHERE the first divergence fell in
    each (``[pair index, token index]``)."""
    d = dict((diagnostics or {}).get(name) or {})
    if name in ("engine_determinism", "scale0_noop"):
        return (f"{d.get('n_divergent')}/{d.get('n')} pairs differ; first "
                f"divergences [pair, token] "
                f"{list(d.get('first_divergences') or [])[:6]}")
    if name == "steering_bites":
        arms = dict(d.get("arms") or {})
        return "; ".join(
            f"{a} {v.get('n_differing')}/{v.get('n')} differ"
            for a, v in sorted(arms.items())) or "no arms probed"
    if name == "batch_invariance":
        out = []
        for direction in ("steered", "unsteered"):
            s = dict(d.get(direction) or {})
            out.append(f"{direction} {s.get('n_divergent')}/{s.get('n')} "
                       f"first {list(s.get('first_divergences') or [])[:4]}")
        return "; ".join(out)
    if name == "selection_honored":
        return (f"{d.get('n_mismatched')}/{d.get('n_requests')} rows differ "
                f"across compositions over {d.get('n_selections')} distinct "
                f"selections; first [row, token] "
                f"{list(d.get('mismatches') or [])[:4]}")
    return ""


def print_fidelity_phase(phase: Mapping[str, Any],
                         required_arms: Sequence[str]) -> None:
    """One phase's block in the console summary. The two phases print
    the SAME block under different headers, so they can be read side by
    side, and every line says which phase it belongs to."""
    print("  " + "-" * 70)
    print(f"  FIDELITY PHASE  [{phase['label']}]")
    warn = ("" if (phase.get("batch_invariant_as_requested")
                   or phase.get("outcome") != "evaluated")
            else "   !! NOT THE REQUESTED SETTING")
    print(f"    VLLM_BATCH_INVARIANT resolved in the child: "
          f"{phase.get('resolved_batch_invariant')!r} (requested "
          f"{phase.get('requested_batch_invariant')!r}){warn}")
    if phase.get("certifies"):
        print("    THIS PHASE CERTIFIES: it is a gate, and its failures "
              "(minus any BI=0 composition waivers) fail the job."
              + ("" if phase.get("is_the_certified_configuration") else
                 "\n    !! but it is NOT the certified configuration "
                 f"(VLLM_BATCH_INVARIANT="
                 f"{CERTIFIED_BATCH_INVARIANT} — production, "
                 f"serving_mode={CERTIFIED_SERVING_MODE} — is) — this run "
                 "certifies nothing"))
    else:
        print("    THIS PHASE CERTIFIES NOTHING. Evidence only: it cannot "
              "change any verdict above and cannot make the job exit "
              "nonzero.")
    g = dict(phase.get("geometry") or {})
    if g:
        kvc = dict(g.get("kv_cache") or {})
        cov = kvc.get("covers_full_budget")
        print(f"    geometry: {g.get('max_model_len')} x "
              f"{g.get('max_num_seqs')} (budget "
              f"{g.get('max_num_batched_tokens')}), gpu_memory_utilization "
              f"{g.get('gpu_memory_utilization')}")
        if kvc:
            print(f"    KV coverage (resolved off the worker): "
                  f"{kvc.get('kv_cache_tokens')} tokens = "
                  f"{kvc.get('full_length_concurrency')}x at full length; "
                  f"covers the full budget: {cov}"
                  + ("" if cov else
                     "   !! the requested max_num_seqs is NOT KV-backed"))
    if phase.get("outcome") != "evaluated":
        print(f"    {phase.get('outcome')}\n      {phase.get('error')}")
        return

    bites = dict((phase.get("diagnostics") or {}).get(
        "steering_bites", {}).get("arms") or {})
    for arm in required_arms:
        d = dict(bites.get(arm) or {})
        nd, n = d.get("n_differing"), d.get("n")
        verdict = ("BITES" if isinstance(nd, int) and nd > 0
                   else "SILENT — HEADLINE FAILURE")
        print(f"    [{verdict}] required arm {arm}: {nd}/{n} probe "
              "completions differ from its scale-0 twin")
    for arm, d in sorted((phase.get("advisory_arm_bites") or {}).items()):
        if isinstance(d, Mapping) and "n" in d:
            print(f"    [advisory] arm {arm}: {d['n_differing']}/{d['n']} "
                  "differ (diagnostic only, never a gate)")

    states = dict(phase.get("check_states") or {})
    expected = set(phase.get("expected_failures") or ())
    unexpected = set(phase.get("unexpected_failures") or ())
    tags = {"passed": "PASS", "failed": "FAIL", "indeterminate": "INDET"}
    at_bi0 = str(phase.get("requested_batch_invariant")) == "0"
    for name in FIDELITY_CHECK_ORDER:
        state = states.get(name, "?")
        mark = ""
        if name in expected and phase.get("gating"):
            mark = ("   <- EXPECTED AT BI=0 and WAIVED: composition "
                    "invariance is not a production guarantee "
                    "(exact_replay dropped)")
        elif name in expected:
            mark = "   <- EXPECTED AT BI=0, NOT A REGRESSION"
        elif name in unexpected and at_bi0:
            mark = "   <- NOT expected at BI=0 — a REAL finding"
        print(f"    [{tags.get(state, state.upper())}] {name}{mark}")
        print(f"           {_diag_line(name, phase.get('diagnostics') or {})}")
    if phase.get("invalidated"):
        print("    !! engine_determinism FAILED in THIS phase — every other "
              "verdict in this block is INDETERMINATE (each is a single-run "
              "equality comparison, and this engine did not repeat itself)")
    if expected:
        print(f"    EXPECTED-FAILURE NOTE for {sorted(expected)}: "
              f"{BI0_EXPECTED_FAILURE_REASON}")

    sd = phase.get("selection_direct")
    if isinstance(sd, Mapping):
        print(f"    selection_honored, DIRECT half: {sd.get('verdict')} — "
              f"{sd.get('n_matched')} matched, {sd.get('n_mismatched')} "
              f"mismatched, {sd.get('n_unobserved')} unobserved of "
              f"{sd.get('n_rows')} rows; the worker's own record made "
              f"{sd.get('aggregate_n_observations')} observations with "
              f"{sd.get('aggregate_n_mismatched')} bound to the wrong "
              "object")
        if str(sd.get("verdict")) == "FAILED":
            print("    !! DIRECT SELECTION FAILED — the worker bound an "
                  "object the request did not ask for. This is a ROUTING "
                  "BUG and it outranks every other line in this record; it "
                  "is NOT a numerical effect of the batch-invariance "
                  "setting.")
        for row in (sd.get("rows") or []):
            if str(row.get("verdict", "")).startswith(("MISMATCH",
                                                       "unobserved")):
                print(f"       row {row.get('row')} asked for "
                      f"{row.get('intended')!r}, worker bound "
                      f"{row.get('resolved')!r} — {row.get('verdict')}")
    else:
        print("    selection_honored, DIRECT half: MISSING — the worker's "
              "selection record was not read for this phase")


def cmd_smoke(args: argparse.Namespace) -> int:
    from sisteer.capeval import manifest as cm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (UNAVAILABLE_MARKER, FAILURE_MARKER, SMOKE_RECORD):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    spec = cm.load_petri_spec(Path(args.manifest))
    script = Path(__file__).resolve()
    provenance: Dict[str, Any] = {
        "served_model": args.model,
        "served_revision": args.revision,
        "manifest_target_model": spec.target_model,
        "manifest_target_revision": spec.target_revision,
        "model_repin": {
            "repinned": (args.model != spec.target_model
                         or args.revision != spec.target_revision),
            "note": ("deliberate: the arms were fit on meta-llama-rendered "
                     "activations and steer every prompt position; the "
                     "manifest's NousResearch mirror ships byte-identical "
                     "weights under a DIFFERENT chat template (43 vs 18 "
                     "prompt tokens on a plain user prompt), which is a "
                     "different intervention for prefill-position arms "
                     "(the arms' owner, 2026-08-21)"),
        },
        "expected_packages": dict(spec.serving_packages),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "started_at": time.time(),
        "engine_geometry": {"max_model_len": MAX_MODEL_LEN,
                            "max_num_seqs": MAX_NUM_SEQS,
                            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS},
        "petri_engine_geometry": {
            "max_model_len": PETRI_MAX_MODEL_LEN,
            "max_num_seqs": PETRI_MAX_NUM_SEQS,
            "max_num_batched_tokens": PETRI_MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": PETRI_GPU_MEMORY_UTILIZATION},
    }
    try:
        provenance["checkout_sha"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:
        provenance["checkout_sha"] = None

    evidence_out: Dict[str, Any] = {"provenance": provenance}

    def flush_evidence() -> None:
        (out_dir / EVIDENCE_FILE).write_text(json.dumps(
            evidence_out, indent=2, sort_keys=True, default=str))

    # ---- preflight: importability + GPU (unavailable, never a crash) ----
    missing = [m for m in ("vllm", "vllm_lens")
               if importlib.util.find_spec(m) is None]
    if missing:
        flush_evidence()
        marker = record_unavailable(
            out_dir, f"not importable in this venv: {missing} — the smoke "
            "runs only in the pinned GPU serving venv "
            "(configs/capeval/vllm-requirements.txt)", stage="imports")
        print(f"SMOKE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0
    import torch
    if not torch.cuda.is_available():
        flush_evidence()
        marker = record_unavailable(out_dir, "torch.cuda unavailable — the "
                                    "smoke needs a GPU node", stage="gpu")
        print(f"SMOKE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0

    # ---- the arms: load + derive + adapt on CPU, before any GPU spend ----
    arms_cfg: List[Dict[str, Any]] = [
        {"cell_id": "static_r_matched", "kind": "static_r",
         "path": Path(args.static_r_object),
         "deploy_scale": float(args.static_r_scale), "required": True},
        {"cell_id": "alphasteer_official", "kind": "alphasteer",
         "path": Path(args.alphasteer_object), "deploy_scale": None,
         "required": True},
    ]
    extra = str(args.extra_alphasteer_scale).strip().lower()
    if extra not in ("skip", "none", ""):
        val = float(args.extra_alphasteer_scale)
        cell = re.sub(r"[^A-Za-z0-9_.\-]", "_",
                      f"alphasteer_official_scale{val}")
        arms_cfg.append({"cell_id": cell, "kind": "alphasteer",
                         "path": Path(args.alphasteer_object),
                         "deploy_scale": val, "required": False})

    derivations: List[Dict[str, Any]] = []
    for arm in arms_cfg:
        if not arm["path"].is_file():
            flush_evidence()
            marker = record_unavailable(
                out_dir, f"arm {arm['cell_id']!r}: no object file at "
                f"{arm['path']} — a missing artifact is a launch blocker, "
                "not a serving verdict", stage="objects")
            print(f"SMOKE UNAVAILABLE — recorded at {marker}",
                  file=sys.stderr)
            return 0
    try:
        for arm in arms_cfg:
            arm["bytes"], rec = load_arm_artifact(
                arm["cell_id"], arm["kind"], arm["path"],
                arm["deploy_scale"])
            rec["required"] = arm["required"]
            derivations.append(rec)
    except Exception as e:
        evidence_out["arm_derivations"] = derivations
        flush_evidence()
        marker = record_failure(
            out_dir, f"a REQUIRED arm failed to load: {type(e).__name__}: "
            f"{e} — this is the finding; no object is substituted",
            stage="objects")
        print(f"SMOKE FAILED — recorded at {marker}", file=sys.stderr)
        return 1
    evidence_out["arm_derivations"] = derivations

    # ---- header + registry (the committed load path verifies it) ----
    shim_used = ensure_chat_markers()
    from sisteer.chat_markers import generation_header_ids
    from transformers import AutoTokenizer

    from sisteer.models import pin_reasoning_template_off
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # reasoning templates (Qwen3.5) render thinking-OFF everywhere in
    # this harness — the fitted objects' response headers were derived
    # under that pin (H=7), so the smoke must probe under it too.
    pin_reasoning_template_off(tok)
    header_ids = tuple(int(t) for t in generation_header_ids(tok))
    template = getattr(tok, "chat_template", None) or ""
    evidence_out["chat_markers_shim"] = shim_used
    evidence_out["assistant_header_ids"] = list(header_ids)
    evidence_out["served_chat_template_sha16"] = sha16(
        template.encode("utf-8"))
    evidence_out["manifest_chat_template_hash"] = spec.chat_template_hash
    if shim_used:
        print("[smoke] NOTE: sisteer.chat_markers is not on this branch — "
              "using the recorded fallback header derivation "
              "(chat_markers_shim=true in the record)", flush=True)

    # ---- the CROSS-PHASE CELL, and the two refusals that bound it ----
    # THE NEGATIVE CONTROL FIRST: "the response override works" is only
    # half a claim unless the arms that must NOT take it are shown to
    # refuse. Both are CPU-only and cost nothing.
    alphasteer_bytes = next(
        (a["bytes"] for a in arms_cfg if a["kind"] == "alphasteer"), b"")
    refusals = response_override_refusals(alphasteer_bytes, header_ids)
    evidence_out["response_override_refusals"] = refusals
    for kind, r in sorted(refusals.items()):
        if r["refused"] and r["names_the_rule"]:
            print(f"[smoke] negative control: {kind} REFUSES a response "
                  "override, naming DERIVATION LOCALITY — as designed",
                  flush=True)
        elif r["refused"]:
            print(f"[smoke] negative control: {kind} refused a response "
                  "override, but the message does not name DERIVATION "
                  f"LOCALITY — {r['error'][:160]}", flush=True)
        else:
            print(f"\n!! NEGATIVE CONTROL FAILED: {kind} ADAPTED under a "
                  "response override instead of refusing it. Its edit is "
                  "computed from the last prompt token and broadcast, so a "
                  "response mask RELOCATES that edit onto generated "
                  "positions rather than restricting it. This is a finding "
                  "about committed adapter code, not about this serving "
                  "run, so the run continues and it is recorded in "
                  "response_override_refusals.\n", flush=True)

    # THE ARM ITSELF: the static_r arm's own derived bytes, served with
    # positions='response' in the resolved spec. Advisory throughout —
    # every way this can go wrong drops the arm and records why, and
    # none of them fails the run.
    response_probe: Dict[str, Any] = {"status": "MISSING: not requested"}
    response_arm_ids: List[str] = []
    response_arm_donors: Dict[str, str] = {}
    if str(args.response_arm).strip().lower() not in ("skip", "none", "off",
                                                      "0", ""):
        donor = next((a for a in arms_cfg
                      if a["kind"] == "static_r" and not a.get("response_of")),
                     None)
        probe_texts = list(PROMPTS) + list(FILLER_PROMPTS)
        hdr_report = header_tail_report(
            _chat_prompt_ids(tok, probe_texts), header_ids,
            "fidelity prompts (probe + filler)")
        evidence_out["response_arm_header_check"] = hdr_report
        if donor is None:
            response_probe = {"status": (
                "MISSING: no static_r arm to serve under the response mask")}
        elif not hdr_report["all_end_with_header"]:
            response_probe = {"status": (
                f"MISSING: {hdr_report['n_mismatched']}/"
                f"{hdr_report['n_prompts']} served prompts do not END with "
                f"the assistant header {list(header_ids)}. The response mask "
                "is LOCATED by matching those ids against each request's own "
                "tokens at prefill and REFUSES on mismatch, so this is a "
                "template/render disagreement — reported as itself here "
                "rather than as an opaque arm failure mid-run"),
                "header_check": hdr_report}
        else:
            cell = response_arm_cell_id(donor["cell_id"])
            response_probe = probe_response_mode_arm(cell, donor["bytes"],
                                                     header_ids)
            if response_probe["ok"]:
                arms_cfg.append({
                    "cell_id": cell, "kind": "static_r",
                    "path": donor["path"],
                    "deploy_scale": donor["deploy_scale"],
                    "required": False, "response_of": donor["cell_id"],
                    "spec_extra": {"positions": "response"},
                    "bytes": donor["bytes"]})
                response_arm_ids.append(cell)
                response_arm_donors[cell] = donor["cell_id"]
                derivations.append({
                    "cell_id": cell, "kind": "static_r",
                    "path": str(donor["path"]), "required": False,
                    "response_of": donor["cell_id"],
                    "resolved_spec": {"kind": "static_r",
                                      "positions": "response"},
                    "served_base_scale": donor["deploy_scale"],
                    "served_sha16": sha16(donor["bytes"]),
                    "note": ("THE CROSS-PHASE CELL: byte-identical to "
                             f"{donor['cell_id']} (same served_sha16), "
                             "served under the response mask. The spec is "
                             "what differs, and the intervention hash covers "
                             "the spec, so they are correctly two cells")})
                print(f"[smoke] response-mode arm {cell}: adapted — "
                      f"positions {response_probe['positions']}, phase "
                      f"{response_probe['phases']}, header "
                      f"{response_probe['header_ids']} "
                      f"({response_probe['n_header_ids']} ids), "
                      f"{response_probe['n_ops']} ops. Same bytes as "
                      f"{donor['cell_id']} (sha16 "
                      f"{sha16(donor['bytes'])})", flush=True)
        if not response_arm_ids:
            print("\n!! RESPONSE-MODE ARM NOT SERVED — "
                  f"{response_probe['status']}\n   (advisory: the run "
                  "continues and every other arm is unaffected)\n",
                  flush=True)
    evidence_out["response_mode_arm"] = response_probe
    evidence_out["arm_derivations"] = derivations

    registry_dir = out_dir / "registry"
    from sisteer.capeval.serving import lens_steer as ls
    from sisteer.capeval.serving import lens_worker as lw

    def build_and_load(arms: List[Dict[str, Any]]
                       ) -> Tuple[Dict[str, Any], List[str]]:
        """Write the registry, then run the WORKER's own load path over
        it on CPU — hash bindings, donor equality, adapters, header
        validation — before any GPU spend."""
        exp = build_registry(registry_dir, arms, header_ids)
        reg = ls.WorkerObjectRegistry.load(registry_dir)
        cells = sorted(lw.cells_needing_ledger(reg))
        want = sorted(a["cell_id"] for a in arms
                      if exp[a["cell_id"]]["needs_ledger"])
        if cells != want:
            raise SmokeConfigError(
                f"CPU registry load: cells_needing_ledger {cells} != "
                f"expected {want}")
        return exp, cells

    try:
        expects, ledger_cells = build_and_load(arms_cfg)
    except Exception as e:
        if response_arm_ids:
            # the advisory cross-phase cell must never be able to fail
            # the run: drop it, record why, and build the registry the
            # certified arms alone would have produced.
            reason = (f"MISSING: the registry rejected the cross-phase cell "
                      f"— {type(e).__name__}: {e}")
            print(f"\n!! RESPONSE-MODE ARM DROPPED at registry build — "
                  f"{reason}\n   (advisory: rebuilding without it and "
                  "continuing)\n", flush=True)
            arms_cfg = [a for a in arms_cfg
                        if a["cell_id"] not in response_arm_ids]
            derivations = [d for d in derivations
                           if d["cell_id"] not in response_arm_ids]
            response_probe = dict(response_probe or {})
            response_probe["status"] = reason
            response_probe["ok"] = False
            evidence_out["response_mode_arm"] = response_probe
            evidence_out["arm_derivations"] = derivations
            response_arm_ids, response_arm_donors = [], {}
            try:
                expects, ledger_cells = build_and_load(arms_cfg)
            except Exception as e2:
                e = e2
            else:
                e = None
        if e is not None:
            flush_evidence()
            marker = record_failure(
                out_dir, f"an arm failed to ADAPT or to load through the "
                f"committed lens_steer/lens_adapters path: "
                f"{type(e).__name__}: {e} — this is the finding; no "
                "translation is guessed", stage="adapt")
            print(f"SMOKE FAILED — recorded at {marker}", file=sys.stderr)
            return 1
    evidence_out["registry_expectations"] = expects

    hook_layers = sorted({l for e in expects.values()
                          for l in e["served_layers"]})
    arms_json = {
        "arm_ids": [a["cell_id"] for a in arms_cfg],
        "required_arm_ids": [a["cell_id"] for a in arms_cfg
                             if a["required"]],
        # the mixed batch deliberately EXCLUDES response-mode arms, so
        # the fidelity gate's own batch keeps the exact composition it
        # passed with (worker: build_mixed_plan's call site).
        "mixed_arm_ids": [a["cell_id"] for a in arms_cfg
                          if not a.get("response_of")],
        "response_arm_ids": list(response_arm_ids),
        "response_arm_donors": dict(response_arm_donors),
        "assistant_header_ids": list(header_ids),
        "selectors": {},
        "expected_hook_layers": hook_layers,
        "expected_ledger_cells": ledger_cells,
    }
    for a in arms_cfg:
        e = expects[a["cell_id"]]
        arms_json["selectors"][a["cell_id"]] = {
            "petri_object": a["cell_id"], "expect_hash": e["expect_hash"],
            "scale": 1.0}
        arms_json["selectors"][e["scale0_cell"]] = {
            "petri_object": e["scale0_cell"],
            "expect_hash": e["scale0_hash"], "scale": 0.0}
    arms_json_path = out_dir / "smoke_arms.json"
    arms_json_path.write_text(json.dumps(arms_json, indent=2,
                                         sort_keys=True))
    flush_evidence()

    # ---- child 1: THE GATING FIDELITY PHASE, certified geometry ----
    # The gate, at the PRODUCTION configuration (module docstring, THE
    # CERTIFIED CONFIGURATION IS THE PRODUCTION ONE): it runs at the
    # certified VLLM_BATCH_INVARIANT (=0, serving_mode=interleaved) and
    # governs the exit status together with the PETRI-geometry phase
    # below. The BI=1 phase further down is evidence and can never
    # certify or fail anything; composition-check failures here are
    # waived at BI=0 (split_gating_failures), never suppressed.
    gating_bi = str(args.fidelity_batch_invariant)
    certified = gating_bi == CERTIFIED_BATCH_INVARIANT
    gating_key = (PHASE_CERTIFIED if certified
                  else f"bi{gating_bi}_gating_NOT_certified")
    gating_label = (
        f"CERTIFIED, GATING — VLLM_BATCH_INVARIANT={gating_bi} "
        f"(the PRODUCTION configuration: serving_mode="
        f"{CERTIFIED_SERVING_MODE}, exact_replay dropped)"
        if certified else
        f"GATING but NOT CERTIFIED — VLLM_BATCH_INVARIANT={gating_bi}, "
        f"and the certified setting is {CERTIFIED_BATCH_INVARIANT}")
    petri_key = (PHASE_PETRI if certified
                 else f"bi{gating_bi}_petri16k_gating_NOT_certified")
    probe_utils = parse_probe_utils(args.petri32_probe_utils)
    probe_keys = {u: f"bi{gating_bi}_petri16k_x{PETRI32_PROBE_SEQS}"
                     f"_probe_util{u:g}" for u in probe_utils}
    fidelity_block: Dict[str, Any] = {
        "gating_phase_key": gating_key,
        "certified_phase_key": PHASE_CERTIFIED if certified else None,
        "petri_geometry_phase_key": petri_key,
        "certified_petri_phase_key": PHASE_PETRI if certified else None,
        "exit_status_governed_by": [gating_key, petri_key],
        "exact_replay_dropped": EXACT_REPLAY_DROPPED_NOTE,
        "note": ("THE CERTIFIED CONFIGURATION IS THE PRODUCTION ONE: "
                 f"serving_mode={CERTIFIED_SERVING_MODE}, no "
                 "VLLM_BATCH_INVARIANT (exact_replay dropped, the user "
                 "2026-08-27 — see exact_replay_dropped). Two phases GATE, "
                 "both at BI=0: the 2048-geometry serving-path gate and "
                 "the PETRI-geometry gate at the geometry the audit "
                 f"actually serves ({PETRI_MAX_MODEL_LEN} x "
                 f"{PETRI_MAX_NUM_SEQS}, gpu_memory_utilization "
                 f"{PETRI_GPU_MEMORY_UTILIZATION}) with its KV coverage "
                 "asserted. In gating phases the composition-dependent "
                 "checks (batch_invariance, selection_honored's output "
                 "half) are measured and reported but WAIVED on failure — "
                 "composition invariance is not a property production "
                 "claims; engine_determinism, scale0_noop, steering_bites "
                 "and selection_direct gate as hard as ever. The BI=1 "
                 "phase, when run, is EVIDENCE ONLY: the dropped "
                 "exact_replay configuration, one environment variable "
                 "apart and nothing else, kept for the lens-engine BI-cost "
                 "comparison; it cannot change a verdict, write a marker, "
                 "or make this job exit nonzero."),
        "headline": BI0_HEADLINE_NOTE,
        "expected_failures_at_bi0": list(BI0_EXPECTED_FAILURES),
        "expected_failure_reason": BI0_EXPECTED_FAILURE_REASON,
        # ALL keys always exist, so a reader never has to wonder whether
        # a phase was skipped or the writer forgot it. Overwritten below.
        "phases": {
            gating_key: {"outcome": "MISSING: not reached"},
            petri_key: {
                "label": (f"CERTIFIED, GATING — the PETRI geometry "
                          f"{PETRI_MAX_MODEL_LEN}x{PETRI_MAX_NUM_SEQS}"),
                "gating": True, "certifies": bool(certified),
                "governs_exit_status": True,
                "outcome": ("MISSING: not reached — the 2048-geometry "
                            "gating phase did not pass, so no further GPU "
                            "time was spent")},
            PHASE_BI1: {
                "label": "PROBE — CERTIFIES NOTHING "
                         "(VLLM_BATCH_INVARIANT=1, the dropped "
                         "exact_replay configuration)",
                "gating": False, "certifies": False,
                "governs_exit_status": False,
                "requested_batch_invariant": "1",
                "resolved_batch_invariant": None,
                "batch_invariant_as_requested": True,
                "is_the_certified_configuration": False,
                "outcome": ("MISSING: not reached — the gating phases did "
                            "not pass, so no further GPU time was spent")},
            **{key: {
                "label": (f"PROBE — CERTIFIES NOTHING (the 32-way "
                          f"re-measure at BI={gating_bi}: "
                          f"{PETRI_MAX_MODEL_LEN}x{PETRI32_PROBE_SEQS} at "
                          f"gpu_memory_utilization {u:g})"),
                "gating": False, "certifies": False,
                "governs_exit_status": False,
                "outcome": ("MISSING: not reached — the gating phases did "
                            "not pass, so no further GPU time was spent")}
               for u, key in probe_keys.items()},
        },
    }
    evidence_out["fidelity_phases"] = fidelity_block
    if not certified:
        print(f"\n!! THE GATING PHASE IS RUNNING AT VLLM_BATCH_INVARIANT="
              f"{gating_bi}, NOT the certified "
              f"{CERTIFIED_BATCH_INVARIANT} (= production, serving_mode="
              f"{CERTIFIED_SERVING_MODE}). Nothing this run produces "
              "certifies the serving path; the record says so too "
              "(is_the_certified_configuration=false).\n", flush=True)

    outcome, phase_gating, report, fid_ev, steered = run_fidelity_phase(
        args=args, out_dir=out_dir, registry_dir=registry_dir,
        arms_json_path=arms_json_path, batch_invariant=gating_bi,
        out_name="worker_steered.json", label=gating_label, gating=True,
        purpose=("THE GATE. All five checks over the committed evidence "
                 "schema, on the certified geometry, at the PRODUCTION "
                 "batch-invariance setting (exact_replay dropped). "
                 "Composition-dependent failures are waived at BI=0 "
                 "(split_gating_failures); everything else here decides "
                 "the job's exit status together with the PETRI-geometry "
                 "phase."))
    fidelity_block["phases"][gating_key] = phase_gating
    evidence_out["steered"] = steered
    evidence_out["steered_exit"] = phase_gating["child_exit"]
    flush_evidence()
    if outcome == "unavailable":
        marker = record_unavailable(
            out_dir, "the lens-active steered engine cannot init "
            f"({phase_gating['error']})", stage="steered_engine_init",
            fidelity_phases=fidelity_block)
        print(f"SMOKE UNAVAILABLE — recorded at {marker}", file=sys.stderr)
        return 0
    if outcome == "child_failed":
        marker = record_failure(
            out_dir, "the steered engine came up but the run failed "
            f"({phase_gating['error']})", stage=str(steered.get("stage")),
            fidelity_phases=fidelity_block)
        print(f"SMOKE FAILED — recorded at {marker}", file=sys.stderr)
        return 1
    if outcome == "evidence_failed":
        # a size/alignment refusal above all (THE EQUAL-SIZE RULE): the
        # gate must never be handed evidence whose reference batches are
        # not the mixed batch's size, because the verdict it would
        # produce reads as cross-request contamination and is believed.
        marker = record_failure(
            out_dir, "the steered worker's gatherings could not be "
            f"assembled into fidelity evidence: {phase_gating['error']}",
            stage="evidence", fidelity_phases=fidelity_block)
        print(f"SMOKE FAILED — recorded at {marker}", file=sys.stderr)
        return 1

    # pinned-package cross-check (the probe's provenance discipline).
    pkg_problems = [
        f"{pkg}: reported {steered.get('packages', {}).get(pkg)!r}, "
        f"pinned {want!r}"
        for pkg, want in spec.serving_packages.items()
        if steered.get("packages", {}).get(pkg) != want]

    evidence_out["batch_sizing"] = phase_gating["batch_sizing"]
    evidence_out["fidelity_evidence"] = fid_ev
    evidence_out["fidelity_report"] = report.as_dict()
    evidence_out["advisory_arm_bites"] = phase_gating["advisory_arm_bites"]
    flush_evidence()

    print("=" * 72)
    print(f"steered vLLM serving smoke — {args.model} @ "
          f"{args.revision[:12]}")
    print_fidelity_phase(phase_gating, arms_json["required_arm_ids"])
    bs = evidence_out["batch_sizing"]
    print(f"  batch sizes: mixed {bs['mixed_batch_size']}; "
          f"{bs['n_reference_servings']} reference servings, all of size "
          f"{sorted(set(bs['reference_batch_sizes']))} "
          f"({bs['n_unsteered_filler_per_reference']} unsteered filler rows "
          "each). Composition varies, SIZE does not — a batch-of-1 "
          "reference is invalid on this engine (see batch_sizing.note in "
          "the record).")

    # THE WAIVER (split_gating_failures): at BI=0 — the certified
    # production configuration — failures of the composition-dependent
    # checks are recorded and printed but do not gate, because
    # composition invariance is exactly what was given up with
    # exact_replay (EXACT_REPLAY_DROPPED_NOTE). Everything else gates.
    gating_failures, waived = split_gating_failures(
        report.failures, batch_invariant=gating_bi)
    phase_gating["waived_failures"] = list(waived)
    if waived:
        print(f"\n!! WAIVED (not gating, at BI={gating_bi}): "
              f"{[w.split(':', 1)[0] for w in waived]} failed the "
              "composition comparison. Composition invariance is NOT a "
              "production guarantee (exact_replay dropped); recorded "
              "under waived_failures.\n", flush=True)
    failures = gating_failures + [f"serving package pin mismatch — "
                                  f"{p}" for p in pkg_problems]
    # THE DIRECT HALF GATES TOO (selection_direct_failure): a FAILED
    # verdict is the worker's own record showing mis-bound routing, and
    # it enters the gating failures exactly like a fidelity check —
    # composition-independent, so the BI=0 waiver never touches it.
    # MISSING/PARTIAL verdicts stay advisory (lost telemetry is not a
    # routing failure) — and only the GATING phases' reports are
    # consulted; the BI=1 probe's selection_direct remains evidence.
    sd_failure = selection_direct_failure(phase_gating.get("selection_direct"))
    if sd_failure:
        failures.append(sd_failure)
    if failures:
        reason = "\n\n".join(failures)
        marker = record_failure(out_dir, reason,
                                failed_checks=[n for n, v in
                                               report.checks.items()
                                               if v != "passed"],
                                selection_direct_failed=bool(sd_failure),
                                invalidated=report.invalidated,
                                fidelity_phases=fidelity_block)
        print(f"SMOKE FAILED — steered serving did NOT hold; recorded at "
              f"{marker}\n{reason}", file=sys.stderr)
        return 1

    # ---- child 1a: THE SAME GATE AT THE PETRI GEOMETRY (16384 x 24) ----
    # The 2048 phase above certifies the serving path; this one
    # certifies it AT THE GEOMETRY THE AUDIT ACTUALLY SERVES
    # (build_server_launch's 16k floor x its concurrent-mode default —
    # 24, the widest that FITS one H200 under no-chunking as measured at
    # BI=1; see the PETRI_* constants' negative-result note, and the
    # --petri32-probe-utils re-measure at the production BI=0), where
    # the requested concurrency is not free: the worker
    # asserts the resolved KV cache covers the full
    # PETRI_MAX_NUM_BATCHED_TOKENS budget (kv_coverage_problem).
    # GATING: any failure here fails the job exactly like the phase
    # above (with the same BI=0 composition waiver). It runs before the
    # probes and the ladder so a failed certification spends no further
    # GPU time.
    outcome_p, phase_petri, report_p, fid_ev_p, steered_p = (
        run_fidelity_phase(
            args=args, out_dir=out_dir, registry_dir=registry_dir,
            arms_json_path=arms_json_path, batch_invariant=gating_bi,
            out_name="worker_steered_petri16k.json",
            label=(f"CERTIFIED, GATING — the PETRI geometry "
                   f"{PETRI_MAX_MODEL_LEN}x{PETRI_MAX_NUM_SEQS}, "
                   f"VLLM_BATCH_INVARIANT={gating_bi}"
                   if certified else
                   f"GATING but NOT CERTIFIED — the PETRI geometry "
                   f"{PETRI_MAX_MODEL_LEN}x{PETRI_MAX_NUM_SEQS} at "
                   f"VLLM_BATCH_INVARIANT={gating_bi}, and the certified "
                   f"setting is {CERTIFIED_BATCH_INVARIANT}"),
            gating=True,
            purpose=("THE GATE AT THE PETRI GEOMETRY. The same five "
                     "checks, the same arms, the same batch-invariance "
                     "setting as the gating phase above, at "
                     f"max_model_len {PETRI_MAX_MODEL_LEN} x max_num_seqs "
                     f"{PETRI_MAX_NUM_SEQS} (the no-chunking budget "
                     f"{PETRI_MAX_NUM_BATCHED_TOKENS}) with "
                     f"gpu_memory_utilization "
                     f"{PETRI_GPU_MEMORY_UTILIZATION}, and the resolved "
                     "KV coverage asserted in the worker. This is the "
                     "geometry the audit's servers run in; certifying 32 "
                     "only at 2048 left it uncertified."),
            max_model_len=PETRI_MAX_MODEL_LEN,
            max_num_seqs=PETRI_MAX_NUM_SEQS,
            gpu_memory_utilization=PETRI_GPU_MEMORY_UTILIZATION))
    fidelity_block["phases"][petri_key] = phase_petri
    evidence_out["steered_petri16k"] = steered_p
    evidence_out["steered_petri16k_exit"] = phase_petri["child_exit"]
    flush_evidence()
    print_fidelity_phase(phase_petri, arms_json["required_arm_ids"])
    if outcome_p != "evaluated":
        # NOT the bootstrap-unavailable path: the 2048 engine initialized
        # and passed in THIS job, so an engine that cannot come up (or a
        # child that dies on its geometry assertions) at the PETRI
        # geometry IS the certification failing — most likely the memory
        # claim — never a launch blocker to wave through with exit 0.
        marker = record_failure(
            out_dir, "the PETRI-geometry gating phase did not produce a "
            f"verdict ({outcome_p}): {phase_petri.get('error')}",
            stage=f"petri_geometry_{outcome_p}",
            fidelity_phases=fidelity_block)
        print(f"SMOKE FAILED — the PETRI geometry "
              f"({PETRI_MAX_MODEL_LEN}x{PETRI_MAX_NUM_SEQS}) did not "
              f"certify; recorded at {marker}", file=sys.stderr)
        return 1
    failures_p, waived_p = split_gating_failures(
        report_p.failures, batch_invariant=gating_bi)
    phase_petri["waived_failures"] = list(waived_p)
    if waived_p:
        print(f"\n!! WAIVED (not gating, at BI={gating_bi}) in the PETRI-"
              f"geometry phase: {[w.split(':', 1)[0] for w in waived_p]} — "
              "composition invariance is NOT a production guarantee "
              "(exact_replay dropped); recorded under waived_failures.\n",
              flush=True)
    sd_failure_p = selection_direct_failure(
        phase_petri.get("selection_direct"))
    if sd_failure_p:
        failures_p.append(sd_failure_p)
    if failures_p:
        reason = "\n\n".join(failures_p)
        marker = record_failure(
            out_dir, "PETRI-geometry gating phase: " + reason,
            failed_checks=[n for n, v in report_p.checks.items()
                           if v != "passed"],
            selection_direct_failed=bool(sd_failure_p),
            invalidated=report_p.invalidated,
            stage="petri_geometry_checks",
            fidelity_phases=fidelity_block)
        print(f"SMOKE FAILED — steered serving did NOT hold at the PETRI "
              f"geometry; recorded at {marker}\n{reason}", file=sys.stderr)
        return 1
    evidence_out["fidelity_evidence_petri16k"] = fid_ev_p
    evidence_out["fidelity_report_petri16k"] = report_p.as_dict()
    flush_evidence()

    # ---- children 1b: THE 32-WAY RE-MEASURE AT THE PRODUCTION BI ----
    # NOT A GATE. 32 x 16384 measurably did not fit one H200 at BI=1
    # (jobs 229194 + 229390: KV coverage and warmup headroom are
    # zero-sum in utilization, and the window between under-coverage at
    # 0.95 and the 0.48 GiB warmup OOM at 0.97 is ~1% wide). BI=0 uses
    # different kernels and may profile a different peak, so the
    # negative result does not transfer — these probes re-take the
    # measurement at the setting production now runs, one engine per
    # requested utilization, recording resolved KV coverage (or the
    # init failure) either way. If one passes WITH REAL SLACK, the
    # PETRI_* constants and DEFAULT_MAX_NUM_SEQS move to 32 in a
    # follow-up commit and the gate re-certifies there; nothing in this
    # job's verdict depends on them.
    petri32_summary: Dict[str, Any] = {
        "requested_utils": [f"{u:g}" for u in probe_utils],
        "max_num_seqs": PETRI32_PROBE_SEQS,
        "max_model_len": PETRI_MAX_MODEL_LEN,
        "budget": PETRI32_PROBE_SEQS * PETRI_MAX_MODEL_LEN,
        "batch_invariant": gating_bi,
        "note": ("EVIDENCE ONLY — the BI=1 negative result (32 x 16384 "
                 "does not fit; jobs 229194/229390) re-measured at the "
                 "production batch-invariance setting. reachable means: "
                 "engine booted, resolved KV covered the full "
                 "no-chunking budget, and the phase evaluated."),
        "outcomes": {},
    }
    if not probe_utils:
        petri32_summary["outcomes"] = {
            "MISSING": "not requested (--petri32-probe-utils skip)"}
    for u in probe_utils:
        key = probe_keys[u]
        _op, phase_32, _rp, _ep, child_32 = run_fidelity_phase(
            args=args, out_dir=out_dir, registry_dir=registry_dir,
            arms_json_path=arms_json_path, batch_invariant=gating_bi,
            out_name=f"worker_steered_petri16k_x{PETRI32_PROBE_SEQS}"
                     f"_util{u:g}.json",
            label=(f"PROBE — CERTIFIES NOTHING (32-way re-measure: "
                   f"{PETRI_MAX_MODEL_LEN}x{PETRI32_PROBE_SEQS} at "
                   f"gpu_memory_utilization {u:g}, "
                   f"VLLM_BATCH_INVARIANT={gating_bi})"),
            gating=False,
            purpose=("EVIDENCE ONLY. Re-measures whether "
                     f"{PETRI32_PROBE_SEQS} x {PETRI_MAX_MODEL_LEN} fits "
                     "one H200 under the no-chunking identity at the "
                     "production batch-invariance setting — the BI=1 "
                     "negative result may not transfer. An init OOM or a "
                     "KV-coverage refusal here is the measurement, not a "
                     "job failure."),
            max_model_len=PETRI_MAX_MODEL_LEN,
            max_num_seqs=PETRI32_PROBE_SEQS,
            gpu_memory_utilization=u)
        fidelity_block["phases"][key] = phase_32
        evidence_out[f"steered_petri16k_x{PETRI32_PROBE_SEQS}_util{u:g}"] = (
            child_32)
        flush_evidence()
        print_fidelity_phase(phase_32, arms_json["required_arm_ids"])
        kv32 = ((phase_32.get("geometry") or {}).get("kv_cache") or {})
        petri32_summary["outcomes"][f"{u:g}"] = {
            "outcome": phase_32.get("outcome"),
            "error": phase_32.get("error"),
            "engine_reached_ready": phase_32.get("engine_reached_ready"),
            "kv_cache_tokens": kv32.get("kv_cache_tokens"),
            "full_length_concurrency": kv32.get("full_length_concurrency"),
            "covers_full_budget": kv32.get("covers_full_budget"),
            "reachable": bool(phase_32.get("outcome") == "evaluated"
                              and kv32.get("covers_full_budget")),
        }
    evidence_out["petri_32way_probe"] = petri32_summary
    flush_evidence()

    # ---- child 1c: THE SAME FIDELITY PHASE AT VLLM_BATCH_INVARIANT=1 ----
    # NOT A GATE, and structurally incapable of becoming one: it writes
    # no marker, returns no exit code, and its verdicts live under their
    # own phase key labelled "certifies: false". This is the DROPPED
    # exact_replay configuration, kept as cheap evidence: it confirms
    # the ~2.97x BI cost on the lens engine itself (the ladder's rung is
    # plugin-disabled), and it shows what the dropped mode would have
    # bought on this hardware — same-size composition invariance — so
    # the trade stays measured rather than remembered. It no longer
    # governs anything (it certified until 2026-08-27; see
    # EXACT_REPLAY_DROPPED_NOTE).
    bi1_requested = str(args.fidelity_bi1_probe).strip().lower()
    phase_bi1: Dict[str, Any] = dict(fidelity_block["phases"][PHASE_BI1])
    if bi1_requested in ("skip", "off", "none", "0", ""):
        phase_bi1["outcome"] = ("MISSING: not requested "
                                "(--fidelity-bi1-probe skip)")
    elif gating_bi == "1":
        phase_bi1["outcome"] = (
            "MISSING: the GATING phase already ran at "
            "VLLM_BATCH_INVARIANT=1, so a second BI=1 phase would measure "
            "the same thing twice — see the gating phase above")
    else:
        _o, phase_bi1, _r1, _e1, bi1_child = run_fidelity_phase(
            args=args, out_dir=out_dir, registry_dir=registry_dir,
            arms_json_path=arms_json_path, batch_invariant="1",
            out_name="worker_steered_bi1.json",
            label=("PROBE — CERTIFIES NOTHING (VLLM_BATCH_INVARIANT=1, "
                   "the dropped exact_replay configuration)"),
            gating=False,
            purpose=("EVIDENCE ONLY. The same five checks, the same arms, "
                     "the same certified geometry, with VLLM_BATCH_INVARIANT "
                     "set to 1 and NOTHING else changed — the configuration "
                     "exact_replay used to pin before it was dropped "
                     "(2026-08-27). Kept for the lens-engine BI-cost "
                     "comparison and as the composition-invariance "
                     "reference; it can never change a verdict."))
        evidence_out["steered_bi1"] = bi1_child
        evidence_out["steered_bi1_exit"] = phase_bi1["child_exit"]
    fidelity_block["phases"][PHASE_BI1] = phase_bi1
    flush_evidence()

    print_fidelity_phase(phase_bi1, arms_json["required_arm_ids"])
    # THE SIDE-BY-SIDE LINE, so the two phases are read as a pair and the
    # BI=1 column can never be mistaken for a verdict.
    det = {k: (v.get("check_states") or {}).get("engine_determinism")
           for k, v in fidelity_block["phases"].items()}
    print("  " + "-" * 70)
    print(f"  REPEAT-DETERMINISM SIDE BY SIDE (engine_determinism): "
          f"{gating_key}={det.get(gating_key)}  |  "
          f"{PHASE_BI1}={det.get(PHASE_BI1)}")
    print(f"  {BI0_HEADLINE_NOTE}")
    print(f"  Exit status is governed by {gating_key} and {petri_key}; "
          f"nothing in {PHASE_BI1} or the 32-way probes can change it.")

    # ---- child 2: THE CROSS-PHASE CELL, in an engine of its own ----
    # After the gate, and in its own process, because a steering failure
    # CONDEMNS a lens worker: a response-mask refusal inside the hook
    # would end whatever else that engine was serving. Its own engine is
    # what makes "advisory" true rather than nearly true. Everything
    # below is recorded and printed; none of it can change a verdict.
    response_child: Dict[str, Any] = {}
    served_response: List[str] = []
    if response_arm_ids:
        resp_json = out_dir / "worker_steered_response.json"
        rc_r, response_child = _run_child(
            "worker-steered",
            ["--model", args.model, "--revision", args.revision,
             "--registry", str(registry_dir),
             "--arms-json", str(arms_json_path), "--out", str(resp_json),
             "--phase", "response", "--max-model-len", str(MAX_MODEL_LEN),
             "--max-num-seqs", str(MAX_NUM_SEQS)],
            # the gating BI (production configuration): its comparisons
            # are all same-composition, valid at either setting.
            _child_env(disable_lens=False, batch_invariant=gating_bi),
            resp_json)
        evidence_out["response_child"] = response_child
        evidence_out["response_child_exit"] = rc_r
        served_response = list(
            response_child.get("response_served_arm_ids") or [])
        if rc_r != 0 and not served_response:
            response_probe = dict(response_probe or {})
            response_probe["gpu_status"] = (
                f"MISSING: the response-mode child exited {rc_r} at stage "
                f"{response_child.get('stage')}: {response_child.get('error')}")
    evidence_out["response_mode_arm"] = response_probe

    # ALWAYS VISIBLE in the console summary — served, refused, or never
    # built. A silent absence would look exactly like a pass.
    if served_response:
        for arm in served_response:
            d = (response_child.get("response_bites") or {}).get(arm, {})
            twin = (response_child.get("response_vs_allposition")
                    or {}).get(arm, {})
            print(f"  [POSITIONS_RESPONSE] {arm}: FIRST EXECUTION ON GPU — "
                  f"positions {response_probe.get('positions')}, phase "
                  f"{response_probe.get('phases')}, header "
                  f"{response_probe.get('n_header_ids')} ids "
                  f"{response_probe.get('header_ids')}")
            print(f"       bites its own scale-0 twin: "
                  f"{d.get('n_differing')}/{d.get('n')}; differs from the "
                  f"ALL-position {twin.get('donor')}: "
                  f"{twin.get('n_differing')}/{twin.get('n')} "
                  "(identical there would mean the mask changed nothing); "
                  f"scale-0 == unsteered on "
                  f"{d.get('scale0_identical_to_unsteered')}/{d.get('n')}")
            print("       advisory — served in its OWN engine, kept out of "
                  "the mixed batch, and unable to affect any verdict above")
    else:
        why = (response_probe.get("gpu_status")
               or response_probe.get("status")) if isinstance(
                   response_probe, dict) else None
        errs = (response_child.get("response_errors") or {})
        for arm, c in errs.items():
            why = f"MISSING: {arm} raised on its first GPU serving — " \
                  f"{c.get('error')}"
        print(f"  [POSITIONS_RESPONSE] NOT SERVED — {why}")

    # ---- children 3..10: the throughput ladder (never gates fidelity) ----
    # Two suites, two geometries, one policy change per rung. Nothing
    # below can change a fidelity verdict: the gate has already been
    # evaluated and its failure path has already returned.
    throughput: Dict[str, Any] = {
        "note": ("A SINGLE-FLAG LADDER over two request geometries. Each "
                 "leg records its own prefix_caching / cuda_graphs / "
                 "batch_invariant policy, its scheduler geometry, and its "
                 "request geometry including the MEASURED shared-preamble "
                 "length — so no number has to be read off its name, and "
                 "each policy's cost is a subtraction between adjacent "
                 "rungs rather than an inference from a bundle."),
        "how_to_read": {
            "batch_invariance_cost":
                "base_graphs_prefix_cache -> base_graphs_prefix_cache_bi",
            "cuda_graph_cost":
                "base_graphs_prefix_cache_bi -> base_eager_prefix_cache",
            "prefix_cache_cost":
                "base_eager_prefix_cache -> base_eager_no_prefix_cache",
            "lens_wiring_cost":
                "base_eager_no_prefix_cache -> lens_unselected",
            "steering_cost": "lens_unselected -> lens_steered__<arm>",
            "caveat": ("the prefix-cache rung is only meaningful against a "
                       "workload that HAS a reusable prefix — read it off "
                       "the shared_prefix suite, and treat the legacy "
                       "suite's version as the no-reuse control"),
        },
        "geometries": {
            "certified": {"max_model_len": MAX_MODEL_LEN,
                          "max_num_seqs": MAX_NUM_SEQS,
                          "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                          "used_by": ["the fidelity gate",
                                      "the legacy_short_distinct suite"],
                          "note": ("unchanged from the passing run — the "
                                   "gate was certified here and nothing "
                                   "about it moved")},
            "certified_petri": {
                "max_model_len": PETRI_MAX_MODEL_LEN,
                "max_num_seqs": PETRI_MAX_NUM_SEQS,
                "max_num_batched_tokens": PETRI_MAX_NUM_BATCHED_TOKENS,
                "gpu_memory_utilization": PETRI_GPU_MEMORY_UTILIZATION,
                "used_by": ["the fidelity gate's PETRI-geometry phase"],
                "note": ("the geometry the audit actually serves "
                         "(build_server_launch: 16k floor, concurrent-mode "
                         "max_num_seqs 32). Its no-chunking budget is 8x "
                         "the other two, so the memory-profiling forward "
                         "is NOT unchanged here — which is exactly why the "
                         "phase asserts resolved KV coverage instead of "
                         "assuming the flags were honored. No throughput "
                         "suite runs on this geometry.")},
            "consumer_shared_prefix": {
                "max_model_len": SP_MAX_MODEL_LEN,
                "max_num_seqs": SP_MAX_NUM_SEQS,
                "max_num_batched_tokens": SP_MAX_NUM_BATCHED_TOKENS,
                "used_by": ["the shared_prefix suite"],
                "note": ("a ~6500-token preamble does not fit in "
                         f"{MAX_MODEL_LEN}. max_num_batched_tokens is "
                         "RE-DERIVED as max_num_seqs * max_model_len, so "
                         "the no-chunking guarantee holds by the identical "
                         "argument — and it lands on the same 65536, so the "
                         "engine's memory profile is unchanged. NO FIDELITY "
                         "EVIDENCE IS GATHERED ON THIS GEOMETRY.")},
        },
        "suites": {
            "legacy_short_distinct": {
                "request_set": {"n_requests": len(PROMPTS) * TP_REPLICAS,
                                "n_distinct_prompts": len(PROMPTS),
                                "max_tokens": TP_MAX_TOKENS,
                                "ignore_eos": True,
                                "shared_preamble": None},
                "geometry": "certified", "legs": {}},
            "shared_prefix": {
                "request_set": {"n_requests": SP_BATCH,
                                "max_tokens": SP_GEN_TOKENS,
                                "ignore_eos": True,
                                "suffix_target_tokens": SP_SUFFIX_TOKENS,
                                "preamble_targets": dict(
                                    SP_PREAMBLE_TARGETS)},
                "geometry": "consumer_shared_prefix", "legs": {}},
        },
    }
    legacy_legs = throughput["suites"]["legacy_short_distinct"]["legs"]
    sp_legs = throughput["suites"]["shared_prefix"]["legs"]

    lens_legacy = steered.get("throughput_legs") or steered.get("throughput")
    if isinstance(lens_legacy, dict) and lens_legacy:
        legacy_legs.update(lens_legacy)
    else:
        legacy_legs["lens_unselected"] = {"missing": "MISSING: " + str(
            steered.get("throughput_error", "not measured"))}
    # the response arm's legacy leg was measured in its own child.
    legacy_legs.update(response_child.get("throughput_legs") or {})
    # the SAME lens legs from the BI=1 probe phase, under their own
    # ``__bi1`` keys so the batch-invariance cost is confirmed ON THE
    # LENS ENGINE rather than inferred from the plugin-disabled ladder,
    # and so a BI=1 number can never overwrite the certified (BI=0)
    # one. (Each leg's own record already carries batch_invariant, so
    # the JSON is unambiguous with or without the suffix.)
    for leg_key, leg in sorted(
            (phase_bi1.get("lens_throughput_legs") or {}).items()):
        legacy_legs[f"{leg_key}__bi1"] = leg
    fidelity_block["comparison"] = {
        "note": ("The batch-invariance cost measured ON THE LENS ENGINE "
                 "ITSELF (both phases: same engine configuration, same "
                 "legacy request set, one environment variable apart), "
                 "beside the plugin-disabled ladder's own BI rung "
                 "(base_graphs_prefix_cache -> base_graphs_prefix_cache_bi). "
                 "speedup_bi0_over_bi1 > 1 means BI=0 is faster by that "
                 "factor — the throughput production recovered by dropping "
                 "exact_replay."),
        "lens_legacy_suite": batch_invariance_cost(
            phase_bi1.get("lens_tokens_per_s") or {},
            phase_gating.get("lens_tokens_per_s") or {}),
        "engine_determinism": {
            k: (v.get("check_states") or {}).get("engine_determinism")
            for k, v in fidelity_block["phases"].items()},
        "selection_direct_verdict": {
            k: ((v.get("selection_direct") or {}).get("verdict")
                if isinstance(v.get("selection_direct"), Mapping) else None)
            for k, v in fidelity_block["phases"].items()},
    }

    # THE LADDER. Each row is one child process: one engine, one policy
    # point, one geometry. `binv` is the batch-invariance rung and is
    # set through the child's environment, never inside the engine.
    ladder = (
        # tag, mode, VLLM_BATCH_INVARIANT, suite, max_model_len, max_num_seqs
        ("base_graphs_prefix_cache", "graphs", "0", "legacy",
         MAX_MODEL_LEN, MAX_NUM_SEQS),
        ("base_graphs_prefix_cache_bi", "graphs", "1", "legacy",
         MAX_MODEL_LEN, MAX_NUM_SEQS),
        ("base_eager_prefix_cache", "eager_cache", "1", "legacy",
         MAX_MODEL_LEN, MAX_NUM_SEQS),
        ("base_eager_no_prefix_cache", "eager", "1", "legacy",
         MAX_MODEL_LEN, MAX_NUM_SEQS),
        ("base_graphs_prefix_cache", "graphs", "0", "shared_prefix",
         SP_MAX_MODEL_LEN, SP_MAX_NUM_SEQS),
        ("base_eager_prefix_cache", "eager_cache", "1", "shared_prefix",
         SP_MAX_MODEL_LEN, SP_MAX_NUM_SEQS),
        ("base_eager_no_prefix_cache", "eager", "1", "shared_prefix",
         SP_MAX_MODEL_LEN, SP_MAX_NUM_SEQS),
    )
    for tag, mode, binv, suite, mml, mns in ladder:
        child_key = f"base_{mode}_bi{binv}_{suite}"
        base_json = out_dir / f"worker_{child_key}.json"
        rc_b, base = _run_child(
            "worker-base",
            ["--model", args.model, "--revision", args.revision,
             "--mode", mode, "--tp-suite", suite,
             "--max-model-len", str(mml), "--max-num-seqs", str(mns),
             "--out", str(base_json)],
            _child_env(disable_lens=True, batch_invariant=binv), base_json)
        evidence_out[child_key] = base
        evidence_out[f"{child_key}_exit"] = rc_b
        target = legacy_legs if suite == "legacy" else sp_legs
        got = base.get("throughput_legs") or {}
        if rc_b == 0 and got:
            for set_name, leg in got.items():
                key = tag if suite == "legacy" else f"{set_name}__{tag}"
                target[key] = leg
        else:
            reason = (f"comparator child exit {rc_b}, stage "
                      f"{base.get('stage')}: {base.get('error')}")
            if suite == "legacy":
                target[tag] = {"missing": f"MISSING: {reason}"}
            else:
                for set_name, _t in SP_PREAMBLE_TARGETS:
                    target[f"{set_name}__{tag}"] = {
                        "missing": f"MISSING: {reason}"}
            print(f"[smoke] MISSING comparator {tag} ({suite}): {reason}",
                  flush=True)

    # the lens engine again, on the consumer geometry, throughput only.
    sp_lens_json = out_dir / "worker_steered_shared_prefix.json"
    rc_sp, sp_lens = _run_child(
        "worker-steered",
        ["--model", args.model, "--revision", args.revision,
         "--registry", str(registry_dir),
         "--arms-json", str(arms_json_path), "--out", str(sp_lens_json),
         "--phase", "throughput", "--tp-suite", "shared_prefix",
         "--max-model-len", str(SP_MAX_MODEL_LEN),
         "--max-num-seqs", str(SP_MAX_NUM_SEQS)],
        # the gating BI (production configuration), so the consumer-
        # geometry throughput is the number production actually gets.
        _child_env(disable_lens=False, batch_invariant=gating_bi),
        sp_lens_json)
    evidence_out["lens_shared_prefix"] = sp_lens
    evidence_out["lens_shared_prefix_exit"] = rc_sp
    if rc_sp == 0 and sp_lens.get("throughput_legs"):
        sp_legs.update(sp_lens["throughput_legs"])
        throughput["suites"]["shared_prefix"]["prompt_sets"] = sp_lens.get(
            "shared_prefix_sets")
        throughput["suites"]["shared_prefix"]["response_header_check"] = (
            sp_lens.get("shared_prefix_header_check"))
    else:
        reason = (f"lens shared-prefix child exit {rc_sp}, stage "
                  f"{sp_lens.get('stage')}: {sp_lens.get('error')}")
        for set_name, _t in SP_PREAMBLE_TARGETS:
            sp_legs[f"{set_name}__lens_unselected"] = {
                "missing": f"MISSING: {reason}"}
        print(f"[smoke] MISSING lens shared-prefix legs: {reason}",
              flush=True)
    # the pre-existing top-level keys, so anything that read the old
    # record shape still finds the three legs it knew about.
    throughput["lens"] = {k: v for k, v in legacy_legs.items()
                          if k.startswith("lens")}
    for legacy_key in ("base_graphs_prefix_cache",
                       "base_eager_no_prefix_cache"):
        throughput[legacy_key] = legacy_legs.get(
            legacy_key, {"missing": "MISSING: not measured"})
    evidence_out["throughput"] = throughput
    flush_evidence()

    def _print_legs(title: str, legs: Mapping[str, Any]) -> None:
        print(f"  {title} (tokens/s | prompts/min):")
        for key in sorted(legs):
            t = legs[key] or {}
            if t.get("tokens_per_s") is not None:
                pre = t.get("request_geometry", {}).get(
                    "shared_preamble_measured_tokens")
                extra = (f"  [preamble {pre} tok, shared prefix "
                         f"{t['request_geometry']['measured_shared_token_prefix']}]"
                         if pre else "")
                print(f"    {key:<46} {t['tokens_per_s']:>9} | "
                      f"{t['prompts_per_min']}{extra}")
            else:
                print(f"    {key:<46} {t.get('missing', 'MISSING: no record')}")

    print("  THROUGHPUT LADDER — one policy per rung; see "
          "throughput.how_to_read in the record for the subtractions.")
    _print_legs(f"legacy suite (certified geometry "
                f"{MAX_MODEL_LEN}/{MAX_NUM_SEQS}/"
                f"{MAX_NUM_BATCHED_TOKENS}; 8 short distinct prompts x "
                f"{TP_REPLICAS}, {TP_MAX_TOKENS} tokens)", legacy_legs)
    _print_legs(f"shared-prefix suite (consumer geometry "
                f"{SP_MAX_MODEL_LEN}/{SP_MAX_NUM_SEQS}/"
                f"{SP_MAX_NUM_BATCHED_TOKENS}; batch {SP_BATCH}, "
                f"{SP_GEN_TOKENS} tokens)", sp_legs)
    kv_petri = ((steered_p.get("geometry") or {}).get("kv_cache") or {})
    print(f"  NOTE: the fidelity gate ran on BOTH certified geometries — "
          f"{MAX_MODEL_LEN}/{MAX_NUM_SEQS}/{MAX_NUM_BATCHED_TOKENS} and "
          f"the PETRI geometry {PETRI_MAX_MODEL_LEN}/{PETRI_MAX_NUM_SEQS}/"
          f"{PETRI_MAX_NUM_BATCHED_TOKENS} at gpu_memory_utilization "
          f"{PETRI_GPU_MEMORY_UTILIZATION} (resolved KV "
          f"{kv_petri.get('kv_cache_tokens')} tokens = "
          f"{kv_petri.get('full_length_concurrency')}x at full length). "
          "The shared-prefix legs ran on a DIFFERENT scheduler geometry "
          "(above), which carries no fidelity evidence.")

    marker = _write_marker(out_dir, SMOKE_RECORD, {
        "status": "completed",
        # THE CERTIFIED VERDICT, unchanged in shape and meaning: this key
        # has always been the gating phase's report and still is — now
        # the BI=0/production phase's (exact_replay dropped 2026-08-27).
        # It is NEVER a probe phase's, and NOT the PETRI-geometry
        # phase's (which lives under fidelity_petri_geometry below).
        "fidelity": report.as_dict(),
        # any composition failures the BI=0 waiver carried past the gate
        # (split_gating_failures): if these are non-empty, the report
        # above says passed=false while the job still completed — that
        # is the waiver working, not a contradiction, and the reason
        # lives in exact_replay_dropped below.
        "fidelity_waived_failures": list(
            phase_gating.get("waived_failures") or []),
        # THE PETRI-GEOMETRY VERDICT: the same five checks at the
        # geometry the audit actually serves, gating exactly like
        # "fidelity". Its resolved KV coverage rides in
        # certified_geometries below and in fidelity_phases.
        "fidelity_petri_geometry": report_p.as_dict(),
        "fidelity_petri_geometry_waived_failures": list(
            phase_petri.get("waived_failures") or []),
        "certified_geometries": {
            "fidelity_2048": {
                "max_model_len": MAX_MODEL_LEN,
                "max_num_seqs": MAX_NUM_SEQS,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "gpu_memory_utilization": SMOKE_GPU_MEMORY_UTILIZATION,
                "kv_cache": ((steered.get("geometry") or {})
                             .get("kv_cache"))},
            "fidelity_petri_16384": {
                "max_model_len": PETRI_MAX_MODEL_LEN,
                "max_num_seqs": PETRI_MAX_NUM_SEQS,
                "max_num_batched_tokens": PETRI_MAX_NUM_BATCHED_TOKENS,
                "gpu_memory_utilization": PETRI_GPU_MEMORY_UTILIZATION,
                "kv_cache": kv_petri or None},
        },
        # EVERY phase, side by side, each labelled with whether it
        # certifies anything. Written even when a probe was skipped
        # (recorded MISSING with the reason).
        "fidelity_phases": fidelity_block,
        "advisory_arm_bites": evidence_out["advisory_arm_bites"],
        # the DIRECT half of selection_honored, at the top level of the
        # gating phase's record because a routing failure outranks
        # everything else here.
        "selection_direct": phase_gating.get("selection_direct"),
        "arm_derivations": derivations,
        "install": steered.get("install"),
        "resolved_engine": steered.get("engine"),
        "resolved_engine_petri_geometry": steered_p.get("engine"),
        "selection_direct_petri_geometry": phase_petri.get(
            "selection_direct"),
        "resolved_engine_shared_prefix": sp_lens.get("engine"),
        # THE EQUAL-SIZE RULE, stated in the artifact: both the mixed
        # batch's size and every reference batch's size, plus why a
        # batch-of-1 reference is invalid on this engine and hardware.
        "batch_sizing": evidence_out["batch_sizing"],
        "prompt_token_lens": steered.get("prompt_token_lens"),
        "filler_token_lens": steered.get("filler_token_lens"),
        "fluency": steered.get("fluency"),
        "throughput": throughput,
        "chat_markers_shim": shim_used,
        "assistant_header_ids": list(header_ids),
        # THE CROSS-PHASE CELL: what was asserted about its ops, whether
        # it served, and how it compared with its own scale-0 twin and
        # with the ALL-position serving of the same bytes.
        "response_mode_arm": {
            "requested": str(args.response_arm),
            "probe": response_probe,
            "served_arm_ids": served_response,
            "bites": response_child.get("response_bites") or {},
            "vs_all_position": (response_child.get("response_vs_allposition")
                                or {}),
            "serving_errors": response_child.get("response_errors") or {},
            "fluency": response_child.get("fluency"),
            "header_check": evidence_out.get("response_arm_header_check"),
            "resolved_engine": response_child.get("engine"),
            "excluded_from_mixed_batch": True,
            "served_in_its_own_engine": True,
            "note": ("ADVISORY BY CONSTRUCTION, three ways: the CPU probe "
                     "asserts positions/phase/header before any GPU spend "
                     "and drops the arm on anything unexpected; every "
                     "served prompt is checked on CPU to end with the "
                     "assistant header, because the mask is located by "
                     "matching those ids at prefill and refuses on "
                     "mismatch; and it is served in an ENGINE OF ITS OWN, "
                     "because a steering failure condemns a lens worker "
                     "and would otherwise end whatever else that process "
                     "was serving. It is also kept out of the mixed batch, "
                     "so the gate's composition is the one that passed."),
        },
        "response_override_refusals": refusals,
        # WHAT WAS GIVEN UP, at the record's top level where a reader of
        # the results sees it before any number: the audit is not
        # exactly replayable, and why the trade was made.
        "exact_replay_dropped": EXACT_REPLAY_DROPPED_NOTE,
        # the 32-way re-measure's summary (evidence, never a gate) —
        # full phase blocks live under fidelity_phases.phases.
        "petri_32way_probe": petri32_summary,
        "batch_invariant_modes": {
            "lens_fidelity_gating": gating_bi,
            "lens_fidelity_petri_geometry": gating_bi,
            "lens_fidelity_bi1_probe": phase_bi1.get(
                "resolved_batch_invariant"),
            "lens_petri_32way_probes": (gating_bi if probe_utils
                                        else None),
            "lens_response": gating_bi, "lens_shared_prefix": gating_bi,
            "base_eager": "1", "base_eager_cache": "1",
            "base_graphs": "0", "base_graphs_bi": "1"},
        "provenance": provenance,
        "recorded_at": time.time(),
    })
    bi1_state = (phase_bi1.get("check_states") or {}).get(
        "engine_determinism", phase_bi1.get("outcome"))
    waived_any = sorted({w.split(":", 1)[0]
                         for w in (list(phase_gating.get("waived_failures")
                                        or []) +
                                   list(phase_petri.get("waived_failures")
                                        or []))})
    print(f"SMOKE PASS — steered tokens served through the lens path at "
          f"the PRODUCTION configuration (serving_mode="
          f"{CERTIFIED_SERVING_MODE}, VLLM_BATCH_INVARIANT={gating_bi}); "
          f"every gating check held in ({gating_key})"
          + (f" — composition checks waived: {waived_any}" if waived_any
             else " — all 5 checks passed outright")
          + f"; record at {marker}")
    print(f"           the gate ALSO held at the PETRI geometry "
          f"({petri_key}: {PETRI_MAX_MODEL_LEN}x{PETRI_MAX_NUM_SEQS}, "
          f"gpu_memory_utilization {PETRI_GPU_MEMORY_UTILIZATION}) with "
          f"resolved KV coverage {kv_petri.get('kv_cache_tokens')} tokens "
          f"= {kv_petri.get('full_length_concurrency')}x at full length "
          f"(>= {PETRI_MAX_NUM_SEQS} required)")
    print(f"           NOTE {EXACT_REPLAY_DROPPED_NOTE}")
    if probe_utils:
        print(f"           32-way re-measure at BI={gating_bi} "
              f"(evidence only): "
              + "; ".join(
                  f"util {u}: reachable={o.get('reachable')} "
                  f"(KV {o.get('kv_cache_tokens')} tok = "
                  f"{o.get('full_length_concurrency')}x, outcome "
                  f"{o.get('outcome')})"
                  for u, o in sorted(petri32_summary["outcomes"].items())))
    print(f"           the BI=1 probe (the dropped exact_replay "
          f"configuration) is evidence only and certified nothing — its "
          f"engine_determinism verdict was: {bi1_state}")
    return 0


# --- CLI ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("smoke", help="the parent orchestrator (GPU node)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--static-r-object", default=None,
                   help="the static_r_matched object.pt (default: the "
                        f"setting {STATIC_R_OBJECT_KEY})")
    p.add_argument("--alphasteer-object", default=None,
                   help="the alphasteer_official object.pt (default: the "
                        f"setting {ALPHASTEER_OBJECT_KEY})")
    p.add_argument("--static-r-scale", type=float,
                   default=DEFAULT_STATIC_R_SCALE,
                   help="the static_r deployment lambda baked into the "
                        "derived artifact's base_scale")
    p.add_argument("--extra-alphasteer-scale",
                   default=DEFAULT_EXTRA_ALPHASTEER,
                   help="advisory alphasteer arm at this base_scale "
                        "('skip' to disable); advisory arms never gate")
    p.add_argument("--response-arm", default=DEFAULT_RESPONSE_ARM,
                   help="serve the static_r arm's own bytes a second time "
                        "with positions='response' — THE CROSS-PHASE CELL "
                        "('skip' to disable). Advisory: it can never fail "
                        "the run")
    p.add_argument("--fidelity-batch-invariant",
                   choices=("0", "1"), default=CERTIFIED_BATCH_INVARIANT,
                   help="VLLM_BATCH_INVARIANT for the GATING fidelity "
                        f"phases (default {CERTIFIED_BATCH_INVARIANT} — "
                        "the PRODUCTION configuration: serving_mode="
                        f"{CERTIFIED_SERVING_MODE}, exact_replay dropped "
                        "2026-08-27). Setting it to 1 does NOT make the "
                        "run certify at BI=1: the record marks the phase "
                        "is_the_certified_configuration=false and the "
                        "console says so")
    p.add_argument("--fidelity-bi1-probe", choices=("on", "skip"),
                   default="on",
                   help="also run the WHOLE fidelity phase a second time "
                        "at VLLM_BATCH_INVARIANT=1 — the DROPPED "
                        "exact_replay configuration — in its own engine, "
                        "as EVIDENCE ONLY (default on). It certifies "
                        "nothing, writes no marker and can never make the "
                        "job exit nonzero; it exists for the lens-engine "
                        "BI-cost comparison and as the composition-"
                        "invariance reference")
    p.add_argument("--petri32-probe-utils", default="skip",
                   help="comma list of gpu_memory_utilization values at "
                        f"which to RE-MEASURE {PETRI32_PROBE_SEQS} x "
                        f"{PETRI_MAX_MODEL_LEN} at the gating BI setting "
                        "(EVIDENCE ONLY, one engine per value; default "
                        "'skip'). The BI=1 negative result (jobs "
                        "229194/229390) may not transfer to BI=0 — see "
                        "PETRI32_PROBE_SEQS")
    p.set_defaults(fn=cmd_smoke)

    w = sub.add_parser("worker-steered", help="internal: the lens engine")
    w.add_argument("--model", required=True)
    w.add_argument("--revision", required=True)
    w.add_argument("--registry", required=True)
    w.add_argument("--arms-json", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--phase", choices=("fidelity", "response", "throughput"),
                   default="fidelity",
                   help="fidelity gathers all the gate's evidence on the "
                        "certified geometry, serving the certified arms "
                        "only; response serves the cross-phase cell alone, "
                        "in its own engine, because a steering failure "
                        "condemns a lens worker; throughput measures the "
                        "shared-prefix legs and gathers no evidence at all")
    w.add_argument("--tp-suite", choices=("legacy", "shared_prefix"),
                   default="legacy")
    w.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    w.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    w.add_argument("--gpu-memory-utilization", type=float,
                   default=SMOKE_GPU_MEMORY_UTILIZATION,
                   help="engine gpu_memory_utilization. The PETRI-geometry "
                        "fidelity phase passes PETRI_GPU_MEMORY_UTILIZATION "
                        "(sized so the 393,216-token budget's KV coverage "
                        "AND its warmup peak both fit — see the PETRI_* "
                        "constants); every other engine keeps the default")
    w.set_defaults(fn=cmd_worker_steered)

    b = sub.add_parser("worker-base",
                       help="internal: VLLM_LENS_DISABLE=1 comparator")
    b.add_argument("--model", required=True)
    b.add_argument("--revision", required=True)
    # the three engine-policy rungs of the ladder: graphs+cache,
    # eager+cache, eager+no-cache. Batch invariance is the fourth rung
    # and rides in through the environment, not through a mode.
    b.add_argument("--mode", choices=("eager", "eager_cache", "graphs"),
                   required=True)
    b.add_argument("--tp-suite", choices=("legacy", "shared_prefix"),
                   default="legacy")
    b.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    b.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_worker_base)

    args = ap.parse_args(argv)
    # the two objects: the flag, else the site's setting (src/sisteer/site.py), any $SHARED expanded here, so an object
    # nothing names, or a root that does not resolve, refuses by name before anything is read
    if args.cmd == "smoke":
        from sisteer import site as _site
        for attr, key in (("static_r_object", STATIC_R_OBJECT_KEY),
                          ("alphasteer_object", ALPHASTEER_OBJECT_KEY)):
            flag = "--" + attr.replace("_", "-")
            try:
                value = getattr(args, attr) or _site.setting(key)
                if not value:
                    ap.error(f"{flag} is not given and {key} is not set: name the object this gate serves "
                             "(the site config's key, or the job environment's)")
                setattr(args, attr, str(_site.expand_path(value, key=flag if getattr(args, attr) else key)))
            except (_site.PlaceholderError, _site.SiteConfigError) as exc:
                ap.error(str(exc))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
