#!/usr/bin/env python
"""lens_vectorize_gate — the GPU fidelity (bit-identity + bounded reduction
tolerance) + timing gate for the vectorized lens apply (lens-vectorize,
2026-09-04; stacked reductions the served default since lens-grouped-scope,
2026-09-04), per served FAMILY, between TWO CHECKOUTS.

WHAT IT PROVES. The lens hook's per-request apply loop was replaced by one
apply per (object, declared facts) group of decode rows, the per-forward
classification is computed once per forward from the runner's CPU mirrors,
and the op lookup is precomputed (lens_worker / lens_adapters / lens_steer
module docstrings). The ELEMENTWISE work of that is bit-identical to the
per-request path at any row count by construction. The REDUCTIONS over the
hidden dim inside a group (the psr gate gemv, the LoRA rank-1 / rank-K GEMM
pair, the LoReFT affine ``F.linear`` pair and its back-projection) are NOT:
job 382416 (2026-09-04) measured a stacked ``[B, K]`` fp32 matmul reducing
in an M-dependent order (unequal from B=7 at K=2560/3136, for rank-4 LoRA
GEMMs at every K; always one bf16 ulp), and the user DECIDED (2026-09-04)
that the stacked call — ONE ``[B, K]`` reduction per group, 2-3.5x faster per
steered decode step at B=32-64 — is the DEFAULT served arithmetic on every
family and every reduction kind (``lens_adapters._row_reduction``;
``SISTEER_LENS_STACKED_REDUCTIONS=0`` is the per-row opt-out), accepting
bf16-ulp-level logit differences. So this gate no longer asks for bit
identity everywhere: it holds the served arithmetic to THE REDUCTION
CONTRACT (below) under the mode it runs (``--reduction-mode``, default
``stacked``) and stays EXACT for everything the contract does not cover.
No CPU test can settle kernel behaviour, so this gate stands the PRODUCTION
server up TWICE per
served-op case on one GPU — once from a SEPARATE CHECKOUT OF THE BASE COMMIT
(``--reference-repo``, held to ``--reference-sha``: the whole old
implementation, worker extension included — the REFERENCE) and once from the
candidate checkout (``--repo-dir``) — through the same tree-local standup
helper (scripts/lens_gate_standup.py), sends IDENTICAL greedy requests to both
(temperature 0, fixed prompts of mixed lengths, max_tokens 64, logprobs +
top_logprobs + token ids on) at B in {1, 8, 32} plus a MIXED batch (steered +
passthrough + the scale-0 control cell in one window), and asserts token ids
AND logprobs are bit-identical per request. The env-var per-row mode of the
candidate (``SISTEER_LENS_PER_ROW_APPLY=1``) is a SECONDARY A/B
(``--legacy-ab``), never the reference.

THE REDUCTION CONTRACT (applied by ``reduction_tolerance`` — pure, unit-
tested — inside the ``shadow_compare`` check; the report header and the
exit-code footer state it in the same words): reduction ops (stacked mode):
bf16-kernel-variance equivalence to the per-row path, bounded IN
CONCENTRATION for every reduction op and, per op CLASS, either ON THE EDIT or
AT THE GATE. Over all counted reduction-op comparisons of a case the
fraction of unequal ELEMENTS
is <= 1e-4 (``REDUCTION_FRAC_TOL``; measured 0 to 3.9e-5 on microbench
388414, 0 to 1.35e-5 on the A/B 388415), and in ANY SINGLE GROUP <= 2e-3
(``REDUCTION_GROUP_FRAC_TOL``) and in ANY SINGLE ROW <= 2e-2
(``REDUCTION_ROW_FRAC_TOL``) — the two concentration bounds are MEASURED,
not mechanism: 10x the largest values the 1600 microbench trials and the
A/B by-rows tables produced (per group 9.8e-5 on a trial, 1.95e-4 for one
flip in a 2-row group at d=2560; per row at most 5 unequal elements in any
one row, 5/2560 = 2.0e-3), and they exist because a whole row's edit
dropped where the edit is within one ulp of the output is invisible per
element (it IS a one-ulp move of the output) and invisible to the case-wide
fraction (3136 elements of 7.8e8) — but lights up nearly every element of
its row and 1/B of its group. For a LINEAR reduction op
(``lens_adapters.LINEAR_REDUCTION_OPS``: the LoRA rank-1 / rank-K GEMM pair
``lora_oproj``, the LoReFT ``affine_subspace`` pair) every unequal element's
stacked edit is within 4 bf16 ulps of the per-row edit beyond one bf16 ulp
of the per-row output, i.e.
|applied - ref| <= 4 * ulp(|ref - base|) + 1 * ulp(|ref|)
(``REDUCTION_ULPS_EDIT_TOL``, ``lens_adapters.edit_excess_ulps``; the
mechanism — a few fp32 ulps on the edit, then one bf16 rounding of the edit
plus one of the sum — predicts <= 2; a dropped, halved or misplaced edit
violates it wherever the edit exceeds a couple of output ulps; all five
families read <= 1). For a GATED reduction op
(``lens_adapters.GATED_REDUCTION_OPS``: ``relu_gated_vector`` /
``linear_gated_vector`` and ``lowrank_relu_gated`` / ``lowrank_linear_gated``)
that unit is WRONG (contract change 2026-09-06, the user's decision on the
ff35913 evidence — docs/lens_stacked_reductions_scoping.md section 9): the
reduction there produces a GATE SCALAR that scales a fixed direction, kernel
variance perturbs it by a few fp32 ulps of its PARTIAL SUMS sum_i |h_i w_i|
whatever the gate's size, and on a near-threshold ReLU gate (1e-5 .. 1e-4)
that is a 2-11 % change of the whole row's edit, which reads as 5-23 "edit
ulps" wherever the activation is exactly zero (qwen35_4b and
nemotron3nano4b's fitted PSR objects) while gemma4e4b, with the LARGEST
absolute gate deviation of the three, read 1. So the gated bound is AT THE
GATE, per gate element (per row; per row x latent for the low-rank pair):
|g_stacked - g_per_row| <= C * 2^-24 * sum_i |h_i w_i| with C = 16
(``REDUCTION_GATE_C``; the worker's ``_gate_stats`` records the ratio as
``gate_max_diff_rel_sum`` for every unequal gated group; 0.4-0.8 x 2^-24 at
the worst-by-edit-ulps element of each family, up to 4.2 x 2^-24 among the
recorded first-unequal groups; a misplaced or wrong gate is ~1e5-1e6 in
these units), and a ReLU flip is allowed ONLY when the side that fired is
within that same bound (``gate_max_flip_rel_sum``; a threshold crossing by
less than the reduction's own rounding). THE GATES ARE OBSERVED, NOT
RECOMPUTED (review P1, 2026-09-06): the shadow compare arms a
``lens_adapters.GateCapture`` around the grouped apply and around each
per-row reference apply, and ``lens_worker._gate_stats_observed`` compares
the gate tensors the two paths actually built their edits from — a
recomputation from the rows certifies the arithmetic in the abstract and is
blind to a grouped-only defect that hands row 0 row 1's gate (with a sparse
direction that defect also hides under every concentration bound). The same
observation binds the gate delta to the OUTPUT delta: for every element
``applied - ref`` must equal ``eff * (g_stacked - g_per_row) * v`` (the
low-rank write for the low-rank pair) up to the bf16 roundings of the two
edits and the two sums — ``gate_edit_consistency_max_ulps``, in units of
ulp(|ref|) + ulp(|edit|), bounded at 4
(``REDUCTION_GATE_CONSISTENCY_ULPS_TOL``): a wrong WRITE breaks it while the
gate delta stays small. Provenance is ENFORCED, not assumed (review P1-a):
every gate record carries ``gate_source``, the worker counts the unequal
groups whose stats are ``'observed'`` (``gate_n_groups_observed``, per key
and case-wide), and a gated key passes only when that count equals its
``n_unequal`` — a recomputed, unsourced or mixed-source record is missing
evidence however good its numbers look. And EXACTNESS WHERE THE GATES AGREE
(review P1-b): after an equal observed gate a gated-VECTOR op is elementwise,
so on a row whose gate delta is exactly zero for every vector op writing it —
or which every gated op masks off — the two writes must be bit-identical;
``gate_exact_rows_unequal_elems`` counts unequal elements in such rows and
any is a FAIL (``REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL`` = 0): the 4-ulp
identity tolerance alone let a grouped-only one-ulp sparse write through. A
low-rank write ``z @ Bᵀ`` is itself a k-term reduction, so its zero-delta
rows cannot promise exactness and are judged by the LINEAR unit instead
(``gate_lowrank_zero_delta_max_ulps_edit`` <= ``REDUCTION_ULPS_EDIT_TOL``).
And WRITE RECONSTRUCTION (round-4 review P1): a legitimate nonzero gate
delta exempts its row from that exactness rule, and the 4-ulp identity
envelope then hides an unrelated sparse wrong write on the row — one ulp
written where the direction is exactly ZERO reads a residual of 1. So for a
gated-VECTOR group the worker rebuilds each side's output from that side's
OBSERVED gates with the apply's own write (``lens_adapters.
gated_vector_write``, one code path — ``base + ((eff g).unsqueeze(-1) v
[* mask]).to(dtype)``, replayed in op order) and counts the elements of
``applied`` / ``ref`` that are not the reconstruction
(``gate_write_recon_mismatch_elems``; ``gate_n_groups_recon`` groups
rebuilt): any is a FAIL (``REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL`` = 0),
whatever the gate delta — a nonzero gate delta cannot write where the
direction is zero, nor by other than eff * delta g * v where it is not. A
vector-gated key whose unequal groups were not all rebuilt (a group with an
interleaved non-gated op — no registered kind has one) is missing evidence.
The low-rank pair is NOT reconstructed — its write ``z @ Bᵀ`` is a k-term
reduction whose kernel variance is exactly what the linear unit tolerates
— so for a low-rank key the identity residual (<= 4) and the zero-delta
rows' linear unit remain the rule: a low-rank row that RECEIVES the edit is
never held exact, identical latents or not — it is BOUNDED (<= 4 edit ulps;
a one-output-ulp flip reads 0 excess and is the accepted class) — and
exactness-at-zero belongs to the vector pair alone (plus any row a low-rank
op masks off, which no path may touch).
The recomputed pair rides along as ``recomputed_*``
for reading only. The observed per-row gates of EVERY compared gated group
— equal or unequal (``gate_hist_n_groups`` of them; round-4 review P2-a: a
histogram over the unequal groups alone is not a histogram over the served
gates) — also feed a GATE-MAGNITUDE HISTOGRAM (``gate_hist_zero`` / ``near``
/ ``mid`` / ``far`` in units of the noise floor f = 2^-24 sum|h_i w_i|,
``gate_dose_share_near_threshold`` — the share of the served dose |eff g|
||v|| on gates within 10 f of the ReLU — and the min nonzero / max g/f),
aggregated apart from the unequal-only stats, reported in the case line and
the FAIL brief and NEVER ruled on: the measurement behind the question
whether the near-threshold gates the evidence run found matter. Unarmed,
the seam costs the production apply, per gated-VECTOR op, one module-global
lookup and one branch, and per LOW-RANK op one module-global lookup and three
branches in the stacked mode (B + 2 in the per-row mode, the latent guard
running once per row) — no allocation, no context manager; not zero, and
counted executably by the test suite.
``max_ulps_edit`` is still
recorded and reported for gated keys but is NOT a rule there. Elementwise
ops, and EVERY op under the per-row mode: exact (n_unequal == 0). Any
modified-flag disagreement, shadow error, record error, an unequal
reduction-op group whose record lacks the tolerance fields
(n_elems_compared == 0, max_ulps_edit / max_frac_row / max_frac_group
missing), an unequal GATED group whose record lacks observed gate stats
(``gate_n_groups_with_stats`` below the key's ``n_unequal``, no
``gate_max_diff_rel_sum``, no ``gate_edit_consistency_max_ulps`` — a
recomputation-only record — or a ReLU flip without its
``gate_max_flip_rel_sum``) or any ``gate_error`` is a FAIL — the tolerance is for kernel variance,
never for a defect or for missing evidence. A shadow key's op token
(``kind:op[+op]:shape:positions:skipbosN:phase:B<rows>``) classifies it
through ``lens_adapters.reduction_op_class`` (``shadow_key_op_class``; the
refinement of ``op_reduces_in_group`` into the two reduction classes): gated
when ANY op on the key is gated (the edit-ulp unit is unreadable on such a
key, and its linear op, if any, is still held by the concentration bounds —
no registered kind produces a mixed key), linear when any reduces and none
is gated, elementwise otherwise; a key whose op the adapters do not classify
is missing evidence — FAIL, never a default class. The gate SETS the mode on
every candidate-tree
standup (``standup_env``: ``SISTEER_LENS_STACKED_REDUCTIONS=1`` for stacked,
``0`` for per_row — never inherited) and scrubs it from the base-commit
reference, which does not read it; the worker attests the mode right after
the ``vectorized`` token of its apply path and as ``lens_reduction_mode`` in
its SelectionRecord, and the apply-path and ``reduction_mode_*`` hard checks
hold both to the gate's mode, so a standup can never be judged under the
other mode's contract. Earlier still, ``start_standup`` REFUSES a
candidate-tree standup whose endpoint (the helper's ``lens_reduction_mode``,
read from every worker's scheduler-config report) does not attest the mode
the gate set (``reduction_mode_endpoint_ok`` — a tree refusal's cousin, exit
2 through the case error, never a verdict); the base-commit reference is
asked for nothing.

HARD CHECKS (each a FAIL -> exit 2; pure functions, unit-tested; the test
suite pins that every check key ``case_checks`` emits is named here):
  reference_apply_path   the reference's attested path is "per_row", or the
                         RPC is ABSENT and the reference is the base commit
                         (which has no sisteer_apply_path);
  candidate_apply_path   the candidate's is "vectorized+<mode>" (lens_worker.
                         apply_path_string — "vectorized+stacked" by default);
  shadow_apply_path      the shadow standup's is "vectorized+<mode>+shadow(
                         serves per_row)"; legacy_ab_apply_path "per_row" when
                         run;
  reduction_mode_*       the CANDIDATE's and the SHADOW standup's SelectionRecord
                         attest ``lens_reduction_mode`` EQUAL to the gate's mode
                         (a missing key — a worker from a tree that predates the
                         attestation — or a disagreement between workers fails;
                         the base-commit reference is never asked);
  evidence_*             both sides' SelectionRecord: every observation
                         matched its request AND n_modifying > 0 (steering
                         happened, by the worker's own count);
  routing_*              the CANDIDATE's and the SHADOW standup's SelectionRecord
                         PER-PATH counters agree with the case's ROUTING CLASS
                         (COVERAGE below): a vectorized case modified rows on
                         the grouped path; a per-slot case modified rows on the
                         per-slot path and NOTHING on the grouped path; a record
                         without per-path counters, or with unattributed
                         applications, fails; routing_class fails when the
                         case id does not yield a class at all;
  steering_bites_B1_*    the B=1 steered singleton DIFFERS from the
                         passthrough-only oracle in at least one logprob (a
                         "steered" output identical to passthrough = nothing
                         was applied);
  controls_B1_*          the B=1 passthrough AND scale-0 singletons are
                         bit-identical to the passthrough-only oracle (every
                         standup runs with PREFIX CACHING OFF, below, so the
                         three compare cold prefills with cold prefills);
  pure_B1_present        the B=1 reference-vs-candidate verdict was actually
                         filed as a hard verdict — B=1 runs OUTSIDE the
                         configurable ladder (``--batches`` names the B>1
                         evidence sizes only; ``gate_batches``) and cannot be
                         configured away;
  batch_invariant_*      every standup's worker attests the resolved
                         VLLM_BATCH_INVARIANT state the launcher intended (env
                         AND active both equal to the intent, no read error;
                         the base-commit reference alone may carry no fields);
  shadow_compare         the in-worker shadow compare held to THE REDUCTION
                         CONTRACT (above; the mechanism below) — in plain
                         words: reduction ops under the stacked mode must be
                         bf16-kernel-variance-equivalent to the per-row path,
                         bounded on the edit (every unequal element's stacked
                         edit within 4 bf16 ulps of the per-row edit beyond
                         one ulp of the output) and in concentration (unequal
                         ELEMENTS <= 1e-4 of those compared over the case,
                         <= 2e-3 in any one group, <= 2e-2 in any one row);
                         elementwise ops, and EVERY op under the per-row
                         mode, must be exact (n_unequal == 0); any
                         modified-flag disagreement, shadow error, record
                         error, or unequal reduction-op group whose record
                         lacks the tolerance fields is a FAIL — the tolerance
                         is for kernel variance, never for a defect or for
                         missing evidence;
  signatures             every response carried non-empty token ids, one
                         logprob record per token, a numeric sampled-token
                         logprob and top-logprobs at every position — missing
                         evidence is never a comparable signature;
  resumed_prefill        THE SHARED-PREFILL CASE (below), by the served
                         object's resumed-prefill POLICY: an ``always`` case's
                         warm steered request was served, attested RESUMED by
                         the worker (cached >= 1, computed >= 1, a modifying
                         application on the resumed rows), differs from the
                         warm passthrough, its cold twin attested NOT resumed,
                         and the identical-prompt repeat and the multi-turn
                         resume were served and attested too; a
                         ``conditional`` case's safe (single-turn) warm request
                         likewise, and its unsafe (multi-turn) one CONTAINED —
                         engine 200, the worker's ResumedPrefillRefusal
                         claimed, the engine serving afterwards; a ``never``
                         case's warm steered request was REFUSED with the
                         worker's typed refusal in the server log; ``None``
                         (inconclusive) when the case did not run
                         (``--no-resumed-prefill``); ok with "not applicable"
                         on a hybrid-recurrent family, where prefix caching is
                         structurally OFF;
  trees                  reference HEAD == --reference-sha, candidate HEAD !=
                         it, both clean; recorded in the JSON (a violation is
                         a tree refusal, exit 2, before any standup).

THE AUTHORITATIVE IDENTITY CHECK IS IN THE WORKER (review #5, 2026-09-04).
A cross-server token/logprob comparison at B > 1 cannot certify the hook at
any batch-invariance setting: at production BI=0 the two standups' admission
order and batch composition are uncontrolled, so bf16 outputs at B > 1 differ
between ANY two servers regardless of the hook (the 378517 run: the reference
itself repeat-unstable at B=32 on llama8b / gemma4e4b, pure_B8 FAIL with
pure_B1 PASS on glm / nemotron); and BI=1 swaps in different cuBLAS / kernel
configurations for exactly the ``@`` / ``F.linear`` ops whose grouped-[N,d]
versus singleton-[1,d] behaviour is the thing under test (the repo's own
measurement, fidelity.py: BI=1 on 0.25.1 / H200 does not give B=1-vs-N bit
identity), so a BI=1 PASS proves nothing about production arithmetic and a
BI=1 FAIL can reject a production-safe op. So every standup runs at
PRODUCTION BI=0 (``--batch-invariant`` is an opt-in experiment flag, default
0, attested per standup by the worker's resolved state) and the identity
claim rests on the IN-WORKER SHADOW COMPARE: a third standup per case, the
candidate with ``SISTEER_LENS_SHADOW_COMPARE=1`` (lens_worker.
ShadowCompareRecord) — for every vectorized group at every hooked layer the
worker also runs the legacy per-row path (the very function the
``SISTEER_LENS_PER_ROW_APPLY=1`` reference runs, dense masks) on a clone of
the SAME rows, compares exactly (``torch.equal``) with the production
kernels in one process (no admission confound), records the deviation of
every unequal group (elements differed / elements compared, max |diff|, the
largest excess of |stacked edit - per-row edit| over one output rounding in
bf16 ulps of the per-row edit, the largest unequal fraction of any one row
and of any one group), serves
the per-row result, and counts per (kind, op, shape, positions, skip_bos,
phase, rows). The gate reads it over ``sisteer_shadow_compare`` and applies
THE REDUCTION CONTRACT: exact for elementwise ops and for every op under the
per-row mode; the bounded bf16-kernel-variance tolerance for reduction ops
under the stacked mode; any shadow error, record error, modified-flag
disagreement, or unequal reduction-op group without the tolerance fields
FAILS the case (exit 2). COVERAGE IS ATTRIBUTED TO THE SUBMITTED WAVE, never
to group size:
the record is snapshotted before and after every wave and the difference is
that wave's comparisons (``shadow_wave_delta``), so a VECTORIZED case needs
>= 1 counted (modified) comparison in some wave at every ladder size
(re-driven up to three times; none -> INCONCLUSIVE_SHADOW, exit 3), while the
actual group row counts (the ``B<rows>`` key tail) are kept as KERNEL-SHAPE
EVIDENCE only (``rows_compared``) — admission splitting legitimately yields
several smaller groups for one HTTP wave, so a key equal to the HTTP request
count is never required. A PER-SLOT case (ROUTING CLASS, COVERAGE below) is
never vectorized — its decode groups are unmodified on both sides
(``equal=None``, not a comparison) — so it needs NO counted comparison: its
hard check is the fallback path itself (``routing_*``), and a counted
comparison appearing for it is a classification disagreement, FAIL. The
multigroup case must additionally show a layer call that compared groups of
>= 2 distinct cells (combined with the CoScheduleRecord).

The cross-server comparison stays, RE-SCOPED: at B=1 (one request per step —
no composition confound; 378517 passed it everywhere) it is a HARD verdict,
and the B=1 singleton controls are hard too (a passthrough and a scale-0
request on the steered standup equal the passthrough-only oracle; the steered
one differs; the multigroup cells differ from each other); at B > 1 every
token comparison — reference vs candidate, control rows vs oracle, bites,
cross-cell distinctness — is recorded as EVIDENCE with the reference's
repeat stability noted, never a FAIL and never a PASS on its own.

PREFIX CACHING is OFF on every FIDELITY standup (the helper composes
``prefix_caching=False``; ``start_standup`` refuses a standup whose
endpoint does not carry composed False AND the launcher's RESOLVED
``enable_prefix_caching`` read as boolean False from the attestation's
``scheduler_geometry.expected`` and every worker report — a read error,
a missing or non-boolean field or a None is a refusal, never a pass —
``prefix_caching_ok``); the ONE caching-ON standup is the shared-prefill
case's (``candidate_resumed``, below), held to ON by the same strictness
(``prefix_caching_on_ok``). Job array 382416 ran the dense families
at their production default (caching ON) and controls_B1 read "scale0
differs" at position 0 (llama / gemma-4 logprobs) or token 37 (GLM) on BOTH
trees for every case, while the two hybrids (Qwen3.5, Nemotron — caching
resolves OFF there) passed: the scale-0 ``none__lens0`` request is a LENS
route and carries the proxy's request-unique ``cache_salt`` (a steered
prefill never resumes from cache), so it ran a full, cold prefill, while
the passthrough singleton and the oracle resumed prompt 0 from the cache
the B>1 waves had just filled — a cached-prefix prefill and a full prefill
are different attention computations on the same tokens. The hook is a
bit-exact no-op at scale 0; the control compared two cache states. Cold
everywhere, the three roles run the same prefill; the shadow compare and
the steered B=1 verdicts never depended on caching (steered requests are
cold in production too).

THE SHARED-PREFILL CASE (resumed prefills for per-position ops — the lens
half of the EvalCheck lane's 2026-09-06 request; lens_adapters module
docstring, RESUMED PREFILLS). vLLM keys KV blocks on token ids + the request
``cache_salt`` and never on the steering selector, so a prompt prefix cached
under the SAME cell and intervention carries exactly the edit a cold prefill
would have given it, position by position, iff every op of the cell computes
each row's edit from that row alone at every position in both phases
(``lens_adapters.resumed_prefill_ok``: the fixed vector, the gated vectors,
the affine subspace, the low-rank gates and the o_proj LoRA at
``positions="all"`` / ``phase="both"``; NOT the two last-prompt-token
broadcast ops, ``positions="last"`` / ``"response"``, or any phase-restricted
op). The worker now serves such a cell's prefix-cache resume as a RESUMED
PREFILL and attests it per request (``lens_worker.SelectionRecord``:
``prefill_resumed`` / ``prefill_cached_positions`` /
``prefill_computed_positions``); every other cell keeps the engine-fatal
salt-discipline refusal. Every single-op case therefore stands the CANDIDATE
up ONCE MORE — ``candidate_resumed``, with PREFIX CACHING ON (the helper's
``--prefix-caching default``: the production default, resolved per model;
SKIPPED, recorded "not applicable", on the hybrid-recurrent families where
caching is structurally OFF — ``vllm_server.hybrid_recurrent_model``) — and
drives it DIRECTLY at vLLM: the routing proxy mints REQUEST-UNIQUE lens salts
and 409s any collision until the proxy half of the landing (EvalCheck's:
a stable per-(cell, intervention) salt for resumable cells) lands, so the
gate mirrors ``proxy._rewrite``'s upstream shape itself — the selector in
``vllm_xargs``, the namespace in ``cache_salt``. Two prompts A and B share
the longest gate passage (~1200 tokens) as a prefix and differ in a short
tail. The drive: passthrough cold B / writer A / warm B under one
passthrough salt — the CACHING-NOISE FLOOR, because a resumed prefill and a
cold prefill of the same tokens are different attention computations on
this stack (382416: even the UNHOOKED warm-vs-cold pair differed at position
0 on llama / gemma-4 and token 37 on GLM), so warm-vs-cold is NOT
bit-identical even without a hook and this gate does not pretend it is;
then steered cold B (a unique salt), steered writer A (the cell-stable
salt), steered warm B (the SAME salt — it resumes A's shared blocks; the
SelectionRecord is reset before it and read after, so its one entry IS the
warm request), the identical B again (the avg@k shape: whatever full
blocks the first warm request wrote are resumed too), and a MULTI-TURN pair
(two two-turn prompts sharing the first turn — user, assistant, the second
user turn diverging late — so the first turn's assistant header sits deep
inside the cached region: a prior writer's response boundary). HARD
(``resumed_prefill``), by the served object's RESUMED-PREFILL POLICY
(``lens_adapters.resumed_prefill_policy`` over its adapted ops,
cross-checked against the case id's op / positions / phase — a
disagreement refuses before any standup): an ``always`` case (per-position
ops at all positions) has the warm steered request SERVED and attested
RESUMED (``prefill_resumed``, ``cached >= 1``, ``computed >= 1``, ``>= 1``
modifying application on the resumed rows), the cold one attested NOT
resumed, the warm steered output DIFFERING from the warm passthrough
(steering bit on the resumed prefill — the worker's modifying count says so
too), and the repeat AND the multi-turn resume served and attested — where
SERVED means the PRODUCTION CONTRACT for a lens request (the proxy's
``_refuse_poisoned_lens_response``): HTTP 200 with a full signature AND the
worker's per-request failure ledger claimed for it
(``sisteer_take_request_failure``, made for EVERY steered response the gate
sends — a worker-contained hook failure still returns 200 from vLLM and the
proxy turns the claimed failure into a 500 ``lens_request_failed``) and
EMPTY, so a failed writer can never seed a cache the gate then validates
reuse of, and an unclaimed steered entry is missing evidence; a
``conditional`` case (per-position ops under the response mask — the
scale-12 production cells) has the SAFE shape — the single-turn pair, which
diverges before the final header, so the cached region is header-free —
served and attested exactly like ``always``, and the UNSAFE shape — the
multi-turn pair, a header in the cached region — CONTAINED: the engine
replies 200 (the request served UNSTEERED), the worker's
``ResumedPrefillRefusal`` is claimed over ``sisteer_take_request_failure``
exactly as the proxy claims it (``RESUMED_CONTAINED_ERROR_TYPE``), and a
fresh steered request afterwards is served with no failure of its own —
never engine-fatal; the repeat is EVIDENCE there (whether the identical
prompt resumes safely depends on where the block boundary falls against the
header); a ``never`` case has the warm steered request REFUSED and the
worker's typed refusal (``RESUMED_REFUSAL_MARKER``) in the standup's
``vllm_server.log`` — the refusal is engine-fatal BY DESIGN (the
cache-isolation gate's step 5), so it is the LAST request of that standup
and the standup is torn down afterwards. EVIDENCE (recorded, never
a verdict — ``closeness``): the warm-vs-cold STEERED pair's first token
divergence and position-0 top-logprob deviation next to the passthrough
pair's, the floor; a numeric rule on that ratio needs a measured run
first, like every tolerance in this gate. The repeat's cached count is
block-aligned (vLLM caches full blocks only), so ``computed`` is the
prompt's remainder past the last full block, or 1 when the prompt is
block-aligned — recorded, never assumed.

TIMING, all at BI=0: the reference (the base commit's per-row path), the
candidate with the shadow OFF (the production vectorized path — their ratio
is the production speedup) and the candidate with the shadow ON (so the gate's
own overhead is visible), ms/step at B in {8, 32, 64}, next to the
passthrough floor.

VERDICTS (B=1, hard). B=1 runs for EVERY case whatever ``--batches`` /
``GATE_BATCHES`` says (the ladder configures the B>1 evidence sizes; a
ladder entry that is not a positive integer refuses at startup). Every batch
is submitted ``--repeats`` times per standup and the verdict is:
PASS = the reference repeats identically, the candidate repeats identically,
and the two agree; FAIL = both repeat identically and DISAGREE (a real
numerical difference: keep that case on the per-row path —
``SISTEER_LENS_PER_ROW_APPLY=<op names or kinds>`` — and investigate);
UNSTABLE_VECTORIZED = the reference repeats but the candidate does not
(FAIL); UNSTABLE_REFERENCE = the reference itself does not repeat
(inconclusive — never PASS; exit 3). Every B=1 check — ``pure_B1``,
``controls_B1_*``, ``steering_bites_B1_*`` — stays HARD and EXACT under
EITHER reduction mode; the reduction tolerance never reaches them. A one-row
group is one call in either mode (``lens_adapters._row_reduction``'s
``x.shape[0] <= 1`` branch takes the single call before the mode is even
consulted), so stacking is inert at B=1 by construction, and the 382416
stacked-candidate gate run filed pure_B1=PASS on all 26 cases that reached a
verdict across the five families. A B=1 difference is therefore a defect,
never kernel variance.

COVERAGE. The required cases are DERIVED, not listed: for every arm of the
production Petri vocabulary whose kind has a lens adapter
(registry_adapter.ARM_OBJECT_KINDS x defense_cells.CLASS_A_METHODS/
CLASS_B_IDS minus lens_adapters.UNSERVABLE_KIND_REASONS — native LoRA exports
are empty, so every object cell is lens-served) the gate writes a tiny object
in the kind's own save format for EVERY value of the fields that format
carries a choice for (``positions`` over the hooks vocabulary, ``skip_bos``
both ways) and keeps exactly the variants the REAL adapter accepts — so
``vector_all`` yields all/response x skip_bos on/off, psr and both LoRA forms
yield all and response, ``last`` is dropped where the adapter refuses it,
antidote/alphasteer/loreft keep the one regime their classes fix — filed by
``kind:op:shape:positions:skip_bos:phase`` (``served_inventory``). Per model
that set is UNIONED with the adapted signatures of every real production
artifact found FOR THIS FAMILY (the scale-12 bank, the extra dirs, the
candsi_refit fits). Per case the gate serves a real object whose adapted
signature IS the case (else a synthetic written WITH that signature, marked
``objects=synthetic``). Real-object selection is PER MODEL: a bank / candsi
dir is keyed by the family; an extra dir's object serves only the family its
own fit-time record names (``extra_object_provenance``: fit_provenance.json /
train_budget_meta.json / the save sidecar, or the dir's CONFIGURED origin
when the object carries no record) — an object fit on another family is
never loaded here (it is a ``__foreign__`` note; the case falls to a
synthetic), and an object nobody can attribute refuses at resolution time,
before any standup.
The live ``multigroup`` case (below) is mandatory too. ``complete`` is true
only when every derived case AND the multigroup case have a verdict; the exit
code is 3 whenever it is false, whatever else passed.

ROUTING CLASS. Each case is ``vectorized`` or ``per_slot``, derived from the
case id's op PHASE through ``lens_worker.phase_vectorizes_in_decode`` — the
worker's own rule (``build_slot_groups`` routes prefill slots per slot and
decode slots into groups; ``apply_hook_ops`` skips an op whose phase excludes
the forward's), never a hand-written list, so the gate and the worker cannot
drift (a worker test pins the predicate against the served behaviour). Every
op with phase ``both`` / ``decode`` is ``vectorized`` (psr, both LoRA forms,
vector_all, antidote — whose ``last`` mask is all-ones in decode); an object
whose every op is ``phase="prefill"`` (alphasteer, loreft) is ``per_slot``:
applied on the legacy per-request path by design, verified through the
SelectionRecord's per-path counters (``routing_*``) rather than through a
shadow comparison that can never happen.

KIND COVERAGE (round-8 review). The inventory above is derived from the
PETRI ARM vocabulary (registry_adapter.ARM_OBJECT_KINDS over
defense_cells.CLASS_A_METHODS / CLASS_B_IDS) because a standup can only
serve a cell a Petri id can name. The lens adapters register MORE kinds
than that — the earlier development branch's lineage (``loreft_both``: the LoReFT
affine subspace in BOTH phases, ``lowrank_gated``), the TrainTest-only
``alphakl`` ladder, the SteeringVector container arms (``vector``),
``static_r``, ``cosine_gated_r`` — and no cell id names them, so the
five-family gate cannot stand them up. ``kind_coverage`` therefore derives
the decode-capable KINDS from the adapters' registered table itself: every
registered kind that is neither internal nor documented-unservable must
either be covered by a Petri case or carry a ``NON_PETRI_KINDS`` entry
(a reason) AND a synthetic mint (``write_synthetic_kind``) probed over the
kind's adapter-supported VARIANT DOMAIN (``kind_variant_axes`` /
``probe_kind_variants``, the twin of ``probe_variants``: static_r's two
``steer_positions`` x the response-mode override that turns it
phase=both, both ``clamp_gate`` classes of lowrank_gated and alphakl, rank 1
and rank k for the two low-rank kinds — distinct kernel-shape classes in the
case schema — the container arms' positions x skip_bos), keeping EVERY distinct case
signature the adapter accepts — decode-capability and routing are computed
from all of them (round-9 review: one fixture per kind filed static_r as
prefill-only while its response-mode serving is a grouped class). A
registered kind with neither a Petri arm nor an entry refuses, and so does
a kind whose probe yields zero cases (a Petri-armed kind absent from the
served inventory, or a mint every variant of which the adapter refuses),
so a supported grouped signature can never go unfiled silently; every
unseen decode-capable SIGNATURE is printed in the report with its reason.
What such a signature has is the CPU call-shape / torch.equal tests — they
certify the call shapes and exact integer arithmetic of the stacked call
against the per-row one — and nothing on hardware: stacked is the default,
so its bf16 deviation is UNMEASURED there, and the reduction-op tolerance is
certified on hardware only for the cases this gate sees.

MULTIGROUP. Every single-object standup exports ``none`` + ONE cell, so every
nonzero steered row of its mixed batch resolves to the same object — a defect
that merged groups, swapped their row sets, or applied one group's op to
another's rows would pass. The ``multigroup`` case stands up two or three
DISTINCT nonzero cells at once (the psr object, the lora_r4 object, and a psr
object of another SI where one exists — synthetic seed-1 otherwise) and
drives ONE interleaved batch of passthrough, scale-0, each cell's plain
decode rows, and each cell's DECLARED prefill-seeded rows (``petri_response_
prefill`` — a distinct DecodeRows grouping key), every prompt appearing twice
under different roles; every row is compared to the base-checkout reference,
the passthrough/scale-0 rows to the passthrough-only oracle, each cell's rows
must differ from the oracle, rows of the SAME prompt under DIFFERENT cells
must differ from each other, and the worker's SelectionRecord must show
n_modifying > 0 for every nonzero cell, 0 for the scale-0 twin, and every
observation matched. (Two scales of one object are not servable: the
registry fixes one effective scale per cell — 1.0 arm, 0.0 twin — and
``select`` refuses any other, so the scale axis is exactly those two.)

CO-SCHEDULING ATTESTATION. All of the above is satisfiable by a sequence of
SINGLE-group forwards — HTTP admission under BI=0 scheduling can spread a
wave across steps — so the candidate worker must ATTEST that distinct
groups shared a forward: its ``CoScheduleRecord`` (lens_worker; read over
``sisteer_coschedule_record``, reset before every wave) counts, per
classified forward, the distinct NONZERO grouping keys; the multigroup case
requires ``n_forwards_with_same_render_pair >= 1`` (one forward held two
distinct cells under the SAME render — both plain or both seeded). To make
that likely the wave is one simultaneous submission of equal-length twin
prompts that all decode ``max_tokens`` (ignore_eos), and an unattested wave
is re-driven up to three times; never attested -> the case is INCONCLUSIVE
(exit 3), never PASS; a candidate whose worker lacks the RPC FAILS (exit 2).
The REFERENCE tree (the base commit) cannot attest and need not: the per-row
path's correctness does not depend on batch composition, so the reference
only supplies the outputs. The multigroup case runs FIRST, so a routing or
config fault surfaces in minutes. Cells are TrainTest ids on the object side
and HARNESS ids on the served side (``safety_refusal__psr_kl_all`` serves as
``si_safety__psr_kl_all``); the helper's endpoint.json carries the explicit
``cell_map`` and every route lookup, request, row check and evidence check
goes through it.

THE DECLARED-RENDER CASES (fit-support-transfer, 2026-09-07; the
lens side of serving a fitted object OUTSIDE its fit support — a thinking-ON
render, every assistant turn, every position — under a HASHED, NAMED cell
declaration, ``serve_declaration.SERVE_VARIANTS``). Three cases, selectable
by name through ``--cases`` and OPT-IN (never part of the derived inventory,
never counted uncovered unless named): ``declared_thinking`` (qwen35_4b),
``all_positions_carveout`` and ``all_turns_render_check`` (gemma4e4b) — a
case named for another family refuses at startup. Each stands the CANDIDATE
up ONCE (``candidate_declared``; there is no reference standup: the base
commit cannot spell a variant cell) with ``none`` + the REAL scale-12 bank's
``safety_refusal__psr_kl_all`` (a response-fit object; no synthetic stand-in,
no other SI) + that cell's ``@<variant>`` spelling, and drives every leg
THROUGH the routing proxy — the proxy's validation IS what is proved. The
EVIDENCE PER REQUEST is the shared-prefill case's: HTTP 200 + a full
signature + an EMPTY claimed failure ledger + the SelectionRecord entry the
worker filed for it (the record reset before, read after; exactly ONE entry
for a served lens request — carrying ``facts`` and ``prefill_masked_rows`` —
and NONE for a refused one, so a 400 is shown to have happened at the proxy).
Every served lens leg of EVERY standup carries that worker evidence through
ONE validator, ``lens_leg_checks`` (code review S-G1 for the carve-out's
forced-policy legs, AG-1 for the all-turns twin and refusal-probe controls:
legs once held to HTTP 200 + the proxy stamp, or to ``_served`` alone, which
an UNSTEERED or MISROUTED request would have passed): ``served``, ``one_entry``
(exactly one SelectionRecord entry, keyed by this request's ENGINE id — the
completion id, or it + vLLM's one ``-<8 hex>`` internal suffix), ``selection``
(``requested`` == the harness cell, ``matched`` and ``stable`` True, >= 1
application), ``masked_rows`` (the derived geometry), ``modifying`` (>= 1 on
a steered cell, 0 on the twin) and, on the twin, ``scale``
(``applied_effective_scale == 0``). ``POSITIVE_LENS_LEGS`` (and the forced
standup's ``FORCED_LENS_LEGS``) name every such leg per case and the test
suite pins the map complete against the drive's served lens requests, so no
leg whose success is evidence can be held to less.
Expected masked-row counts are DERIVED from the served template's ids
(``thinking_header.on_ids`` / ``off_ids``, the ``/v1/tokenize`` prompt
length, the gate's own ``span_locator`` arithmetic over the rendered ids),
never literals; the attested map is read from ``endpoint.json`` AND
``/v1/models`` and must equal the case's expectation EXACTLY (the base cell
absent); every id is derived through the standup's ``cell_map`` and routing
table. A run naming these cases only skips the passthrough oracle standup.
Hard checks (``declared_render_checks``; each ``{"ok": bool, "detail":
<the observed literal>}``; a missing fact is a FAIL; every key is named here
in full and the test suite pins that):
  declared_thinking      the ``@thinking`` cell's declared ON request (kwargs
                         true + ``petri_thinking_on`` true): thinking_on_served,
                         thinking_on_one_entry, thinking_on_selection,
                         thinking_on_facts (thinking_on True),
                         thinking_on_masked_rows (== len(on_ids), 5 on
                         Qwen3.5), thinking_on_modifying (>= 1),
                         thinking_on_stamp (request-unique,
                         declared-request-unique, thinking_on); the same shape
                         on ``petri__none__lens0`` — the twin is a DECLARED
                         cell here, its donor the ``@thinking`` cell
                         (thinking_twin_donor), modifying NOTHING:
                         thinking_twin_served, thinking_twin_one_entry,
                         thinking_twin_selection, thinking_twin_facts,
                         thinking_twin_masked_rows, thinking_twin_modifying
                         (0), thinking_twin_scale (0), thinking_twin_stamp;
                         the base cell under kwargs false: thinking_off_served,
                         thinking_off_one_entry, thinking_off_selection,
                         thinking_off_facts (False), thinking_off_masked_rows
                         (== len(off_ids), 7), thinking_off_modifying,
                         thinking_off_stamp (conditional, thinking_off);
                         thinking_bites_on (the ON
                         steered output differs from the twin's under the same
                         render); the §2.5.6 refusals, 400 with the code and
                         nothing at the worker: thinking_refuse_undeclared,
                         thinking_refuse_mismatch,
                         thinking_refuse_kwargs_invalid,
                         thinking_refuse_cell_not_declared,
                         thinking_refuse_not_lens; kwargs true on
                         ``petri__base`` undeclared (200, cell-stable,
                         not-consulted, thinking_on):
                         thinking_passthrough_on_served,
                         thinking_passthrough_on_stamp;
                         thinking_attestation_map (exactly ``{@thinking:
                         {"serve_out_of_support": ["thinking"]}, none__lens0:
                         same}`` in endpoint.json and /v1/models);
                         thinking_artifact_sha256 (Q8: the endpoint's scalar
                         == the bank artifact's sha256, the per-cell map on
                         the base and the variant — the EXACT map in
                         endpoint.json AND in /v1/models — bound at export
                         and published by the proxy; S-G2);
                         thinking_header_matches_tokenizer (on/off ids == the
                         gate's own derivation, which certified — no reason
                         beside it —, moves True, /v1/models equal);
                         thinking_header_attested (the four-key fact and the
                         thinking_header_error beside it agree between
                         endpoint.json and /v1/models and PAIR: a partial
                         fact carries a reason, a complete one None; S-G3);
                         thinking_claims_empty; thinking_signatures;
  all_positions_carveout the fit/serve guard through the REAL adapter on the
                         real bytes, in-process: carveout_guard_admits_declared
                         (the ``@all_positions`` overlay adapts to ["all"]),
                         carveout_guard_refuses_undeclared (``positions="all"``
                         WITHOUT the name is refused naming the carve-out); the
                         single-turn request on the variant:
                         carveout_variant_served, carveout_variant_one_entry,
                         carveout_variant_selection,
                         carveout_variant_masked_rows (== P from /v1/tokenize,
                         minus the BOS row iff the object skips it),
                         carveout_variant_modifying, carveout_variant_stamp
                         (declared-request-unique under the default policy);
                         on the base: carveout_base_served,
                         carveout_base_one_entry, carveout_base_selection,
                         carveout_base_masked_rows (== the header's width),
                         carveout_base_modifying, carveout_base_stamp
                         (conditional); on the twin, whose donor is the BASE
                         (carveout_twin_donor — a position variant is never a
                         donor): carveout_twin_served, carveout_twin_one_entry,
                         carveout_twin_selection, carveout_twin_masked_rows
                         (the header's width), carveout_twin_modifying (0),
                         carveout_twin_scale (0), carveout_twin_stamp
                         (conditional);
                         carveout_bites; carveout_attestation_map (exactly
                         ``{@all_positions: {"serve_experiment":
                         "fit_response_serve_all_positions"}}``);
                         carveout_attested_policy_endpoint;
                         carveout_artifact_sha256 and
                         carveout_thinking_header_attested (as the thinking
                         case's, on this standup);
                         carveout_degeneracy_instrument (distinct-3-gram ran on
                         the served text — evidence, never a verdict); the
                         SECOND standup under ``--lens-salt-policy
                         request_unique_all``: carveout_forced_policy_endpoint
                         (endpoint + /v1/models say request_unique_all),
                         carveout_forced_variant_stamp,
                         carveout_forced_base_stamp, carveout_forced_twin_stamp
                         (request-unique, forced-request-unique), and per
                         forced leg the validator's evidence (S-G1):
                         carveout_forced_variant_served,
                         carveout_forced_base_served,
                         carveout_forced_twin_served,
                         carveout_forced_variant_one_entry,
                         carveout_forced_base_one_entry,
                         carveout_forced_twin_one_entry (exactly one
                         SelectionRecord entry, this request's id),
                         carveout_forced_variant_selection,
                         carveout_forced_base_selection,
                         carveout_forced_twin_selection (requested == the
                         harness cell, matched, stable, >= 1 application),
                         carveout_forced_variant_masked_rows (== P of the
                         FORCED standup's own /v1/tokenize render, minus BOS
                         iff skipped), carveout_forced_base_masked_rows,
                         carveout_forced_twin_masked_rows (== the header's
                         width: the twin's donor is the base),
                         carveout_forced_variant_modifying,
                         carveout_forced_base_modifying (>= 1),
                         carveout_forced_twin_modifying (0),
                         carveout_forced_twin_scale (0);
                         carveout_forced_attestation_map (the base still
                         absent); carveout_forced_artifact_sha256;
                         carveout_forced_thinking_header_attested;
                         carveout_claims_empty; carveout_signatures;
  all_turns_render_check the 3-turn conversation on the variant:
                         all_turns_convo_served, all_turns_convo_one_entry,
                         all_turns_convo_selection, all_turns_check_label_ok
                         (``petri.all_turns_check == "ok"``),
                         all_turns_convo_masked_rows (== the gate's own
                         ``assistant_turn_spans`` sum over the /v1/tokenize
                         ids), all_turns_convo_modifying, all_turns_stamp
                         (declared-request-unique); on the base:
                         all_turns_base_served, all_turns_base_one_entry,
                         all_turns_base_selection, all_turns_base_masked_rows
                         (== the header's width), all_turns_base_modifying,
                         all_turns_base_unlabelled (no all_turns_check label
                         on the base route); on the twin (AG-1 — the leg the
                         bites comparison rests on): all_turns_twin_served,
                         all_turns_twin_one_entry, all_turns_twin_selection,
                         all_turns_twin_masked_rows (the header's width),
                         all_turns_twin_modifying (0), all_turns_twin_scale
                         (0), all_turns_twin_stamp (conditional);
                         all_turns_refuse_quoted (a historical turn quoting
                         ``<turn|>\\n``: 400 all_turns_render_refused on the
                         variant); all_turns_refuse_fake (a COMPLETE fake
                         boundary inside an assistant message: 400
                         all_turns_expected_unavailable naming the
                         ChatMarkerError — revision 1, finding 7); the two
                         probes' base CONTROLS, 200 with the validator's
                         evidence (AG-1): all_turns_quoted_base_served,
                         all_turns_quoted_base_one_entry,
                         all_turns_quoted_base_selection,
                         all_turns_quoted_base_masked_rows,
                         all_turns_quoted_base_modifying,
                         all_turns_quoted_base_stamp, all_turns_fake_base_served,
                         all_turns_fake_base_one_entry,
                         all_turns_fake_base_selection,
                         all_turns_fake_base_masked_rows,
                         all_turns_fake_base_modifying,
                         all_turns_fake_base_stamp;
                         all_turns_attestation_map (exactly
                         ``{@all_turns: {"serve_experiment":
                         "fit_response_serve_all_turns"}}``);
                         all_turns_scaffold (the registry's turn-end ids ==
                         the gate's own derivation); all_turns_bites;
                         all_turns_twin_donor (the base);
                         all_turns_artifact_sha256;
                         all_turns_thinking_header_attested;
                         all_turns_claims_empty; all_turns_signatures.
Standup plumbing (``lens_gate_standup.py``): the helper derives the ON header
/ turn-end scaffold iff a served cell needs them, derives ``thinking_header``
HARD for a declaring cell and NON-FATALLY otherwise (the partial fact and its
``thinking_header_error`` kept, ``ChatMarkerError`` only — S-G3), records the
pair in the routing table, binds every fitted cell's artifact digest, reads
the attested ``lens_serve_declarations`` and hands them, the header fact and
its reason, the digests and ``--lens-salt-policy`` to the proxy — each only
when the SERVED tree's API takes it, so the base-commit reference is
untouched; the endpoint publishes the digest map ONLY when the proxy received
it. The declared-render standups are spawned with ``--declared-render``
(S-G2): a served tree that drops the digest binding or its publication
refuses before any launch, naming the kwarg.

EXIT CODES (``overall_exit_code``; ``render_report``'s footer states the same):
  0  every case PASS, every hard check ok, ``complete`` true;
  2  any FAIL / UNSTABLE_VECTORIZED verdict, any hard check ``ok`` False (a
     failed check, missing evidence, an unreadable batch-invariance state, a
     routing disagreement, a missing pure_B1, a reduction mode the worker
     does not attest, a shadow record outside THE REDUCTION CONTRACT, a
     shared-prefill case served / refused against its class or unattested),
     a signature error, a tree refusal — and, under the opt-in BI=1
     experiment, an UNSTABLE_REFERENCE;
  3  no failure but something inconclusive or missing: UNSTABLE_REFERENCE at
     BI=0, INCONCLUSIVE_COSCHEDULE / INCONCLUSIVE_SHADOW, a check ``ok`` None,
     an errored case, an uncovered inventory case, ``complete`` false.

Runs from the pinned serving venv; ``scripts/sbatch/lens_vectorize_gate.sbatch``
clones the reference from the cluster mirror and submits one array task per
family. The CPU tests import this module without vllm / aiohttp / a GPU —
every runtime import is lazy.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PER_ROW_ENV = "SISTEER_LENS_PER_ROW_APPLY"
#: the base commit the reference standup must run from
DEFAULT_REFERENCE_SHA = "09be969fb676db26a532917e7eb1673de40605f5"
REFERENCE, CANDIDATE, LEGACY, PASSTHROUGH = "reference", "candidate", "legacy_ab", "passthrough"
INCONCLUSIVE_VERDICTS = ("UNSTABLE_REFERENCE", "INCONCLUSIVE_COSCHEDULE", "INCONCLUSIVE_SHADOW")
PASS, FAIL, UNSTABLE_REF, UNSTABLE_VEC = ("PASS", "FAIL", "UNSTABLE_REFERENCE",
                                          "UNSTABLE_VECTORIZED")
#: the multigroup outputs agreed but the candidate never attested that two
#: distinct groups shared a forward (module docstring): exit 3, never PASS
INCONCLUSIVE = "INCONCLUSIVE_COSCHEDULE"
#: a B > 1 cross-server comparison: recorded, never a verdict (module docstring)
EVIDENCE = "EVIDENCE"
#: the candidate-with-shadow standup and its env switch (lens_worker)
SHADOW = "candidate_shadow"
SHADOW_ENV = "SISTEER_LENS_SHADOW_COMPARE"
#: the candidate standup with PREFIX CACHING ON that drives THE SHARED-PREFILL
#: CASE (module docstring) — the one standup the gate holds to caching ON
RESUMED = "candidate_resumed"
#: the standup helper's ``--prefix-caching`` regimes
PREFIX_CACHING_OFF, PREFIX_CACHING_DEFAULT = "off", "default"
#: the worker's typed refusal of a prefix-cache resume on a cell whose ops
#: cannot resume — lens_batch.check_tracked_segment and lens_worker.
#: build_request_slots both spell it; the negative protocol greps it out of
#: the standup's vllm_server.log (the refusal is engine-fatal, so the log is
#: the only surface left). A literal mirror; tests/test_lens_resumed_prefill_
#: gate.py pins it against the refusals the modules actually raise.
RESUMED_REFUSAL_MARKER = "resumed_prefill_ok is False"
#: the upstream selector key (proxy.UPSTREAM_EXTRA_ARGS_KEY — a literal mirror
#: for the same lazy-import reason; pinned equal by the test suite)
UPSTREAM_EXTRA_ARGS_KEY = "vllm_xargs"
#: the SelectionRecord request-entry fields the resumed attestation reads
#: (lens_worker.SelectionRecord; pinned by the test suite)
RESUMED_ATTEST_FIELDS = ("prefill_resumed", "prefill_cached_positions", "prefill_computed_positions",
                         "n_applications_modifying", "n_applications_modifying_resumed")
#: the three-valued resumed-prefill POLICY vocabulary (lens_adapters.RESUMED_*;
#: literal mirrors, pinned by the test suite): ``never`` (a broadcast, last or
#: phase-restricted op — the engine-fatal refusal), ``always`` (per-position ops
#: at all positions), ``conditional`` (per-position ops under the response mask
#: — decided per request; an unsafe hit is a CONTAINED per-request refusal)
RESUMED_NEVER, RESUMED_ALWAYS, RESUMED_CONDITIONAL = "never", "always", "conditional"
RESUMED_POLICIES = (RESUMED_NEVER, RESUMED_ALWAYS, RESUMED_CONDITIONAL)
#: the error_type of the CONTAINED refusal the worker records for a conditional
#: cell's unsafe resume (lens_adapters.ResumedPrefillRefusal), claimed over
#: ``sisteer_take_request_failure`` exactly as the proxy claims it
RESUMED_CONTAINED_ERROR_TYPE = "ResumedPrefillRefusal"
#: the shared-prefill case's ONE request through the routing proxy
#: (:func:`proxy_regime_probe`): short and DISTINCT from the case's A/B pair,
#: so reading the live proxy's decision cannot seed the cache the case
#: measures.
PROXY_PROBE_MESSAGES = [{"role": "user", "content": "Reply with the word ok."}]
#: the REDUCTION MODE switch (lens_adapters.STACKED_REDUCTIONS_ENV; stacked is
#: the default since lens-grouped-scope, 2026-09-04): the gate SETS it
#: explicitly on every candidate-tree standup — "1" for the stacked mode, "0"
#: for per_row (``standup_env``; never inherited) — and scrubs it from the
#: base-commit reference, which does not read it. The mode names, the env
#: name and the attestation key mirror lens_adapters / lens_worker (pinned by
#: the test suite; this module keeps every runtime import lazy, so they are
#: literals here).
STACKED_ENV = "SISTEER_LENS_STACKED_REDUCTIONS"
REDUCTION_MODE_STACKED = "stacked"
REDUCTION_MODE_PER_ROW = "per_row"
REDUCTION_MODES = (REDUCTION_MODE_STACKED, REDUCTION_MODE_PER_ROW)
DEFAULT_REDUCTION_MODE = REDUCTION_MODE_STACKED
#: the env seam main() reads when --reduction-mode is not given (the sbatch's
#: body-env seams, GATE_BATCHES and friends, reach the gate the same way)
REDUCTION_MODE_ENV = "GATE_REDUCTION_MODE"
#: the SelectionRecord key the worker attests its mode under
#: (lens_worker.LENS_REDUCTION_MODE_KEY)
LENS_REDUCTION_MODE_KEY = "lens_reduction_mode"
#: THE REDUCTION CONTRACT's four bounds (module docstring). Per element: the
#: excess of |stacked edit - per-row edit| over one bf16 rounding of the
#: output, in bf16 ulps of the per-row edit (lens_adapters.edit_excess_ulps;
#: the mechanism — a few fp32 ulps on the edit, one bf16 rounding of the edit,
#: one of the sum — predicts <= 2; a dropped / halved / misplaced edit reads
#: tens to thousands wherever the edit exceeds a couple of output ulps).
#: Concentration: the fraction of unequal ELEMENTS over all counted
#: reduction-op comparisons of a case (microbench 388414 measured 0 to 3.9e-5,
#: the A/B 388415 0 to 1.35e-5), in any single group and in any single row —
#: the last two MEASURED, not mechanism: 10x the largest values the 1600
#: microbench trials and the A/B by-rows tables produced (per group 9.8e-5 on
#: a trial, 1.95e-4 for one flip in a 2-row group at d=2560; per row at most
#: 5 unequal elements in any one row, 5/2560 = 2.0e-3). They exist because a
#: whole row's edit dropped where the edit is within one ulp of the output is
#: a one-ulp move per element and 4e-6 of the case — and nearly every
#: element of its row.
REDUCTION_FRAC_TOL = 1e-4
REDUCTION_GROUP_FRAC_TOL = 2e-3
REDUCTION_ROW_FRAC_TOL = 2e-2
REDUCTION_ULPS_EDIT_TOL = 4.0
#: THE GATED-OP BOUND (contract change 2026-09-06): |g_stacked - g_per_row| <=
#: REDUCTION_GATE_C * 2^-24 * sum_i |h_i w_i| per gate element, and a ReLU flip
#: only when the fired side is within the same bound. 2^-24 is one fp32
#: rounding of a partial sum of the reduction's own magnitude; C = 16 is 20x
#: the 0.8 measured at the worst-by-edit-ulps element of each family on the
#: ff35913 evidence run (and ~4x the 4.2 x 2^-24 the recorded first-unequal
#: groups reached); a misplaced or wrong gate is ~1e5-1e6 in these units
REDUCTION_GATE_C = 16.0
REDUCTION_GATE_UNIT = 2.0 ** -24
REDUCTION_GATE_REL_SUM_TOL = REDUCTION_GATE_C * REDUCTION_GATE_UNIT
#: THE GATE-EDIT IDENTITY: the gates the rule reads
#: are OBSERVED on the two applies (lens_adapters.GateCapture), and for every
#: element the stacked-minus-per-row edit must equal eff * (g_stacked -
#: g_per_row) * v (the low-rank write for the low-rank pair) up to the bf16
#: roundings of the two edits and the two sums — the worker's
#: ``gate_edit_consistency_max_ulps``, in units of ulp(|ref|) + ulp(|edit|);
#: kernel variance reads <= ~1, a wrong WRITE (a transposed operand, another
#: row's direction) breaks it while leaving the gate delta small
REDUCTION_GATE_CONSISTENCY_ULPS_TOL = 4.0
#: EXACTNESS WHERE THE GATES AGREE: on a row whose observed gate
#: delta is exactly zero for every gated-vector op writing it, or which every
#: gated op masks off, the two writes are elementwise-identical by
#: construction — ANY unequal element there is a wrong write, never kernel
#: variance (the identity's 4-ulp tolerance alone let a one-ulp sparse write
#: through). A low-rank write is itself a k-term reduction, so its zero-delta
#: rows are judged by the LINEAR unit (REDUCTION_ULPS_EDIT_TOL) instead.
REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL = 0
#: WRITE RECONSTRUCTION: for a gated-VECTOR group each
#: side's output is rebuilt bit-for-bit from that side's OBSERVED gates with
#: the apply's own write (lens_adapters.gated_vector_write), and any element
#: that is not the reconstruction is a wrong write — whatever the gate delta
#: (a legitimate nonzero delta exempted its row from the exactness rule, and
#: the 4-ulp identity envelope then hid a one-ulp write where the direction
#: is exactly zero). The low-rank pair is not reconstructed: its write is a
#: k-term reduction, so the identity residual and the linear unit rule there.
REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL = 0
#: the three op CLASSES of a shadow key (``shadow_key_op_class``) — literal
#: mirrors of lens_adapters.OP_CLASS_* (this module keeps its imports lazy);
#: tests/test_lens_vectorize_gate.py pins them equal
OP_CLASS_REDUCTION_LINEAR = "reduction_linear"
OP_CLASS_REDUCTION_GATED = "reduction_gated"
OP_CLASS_ELEMENTWISE = "elementwise"
OP_CLASSES_REDUCTION = (OP_CLASS_REDUCTION_LINEAR, OP_CLASS_REDUCTION_GATED)
#: literal mirrors of lens_adapters.GATED_REDUCTION_OPS / LINEAR_REDUCTION_OPS
#: (the ONE definition lives there; the test suite pins these equal) — the
#: contract text names them, the classification itself goes through the
#: adapters' ``reduction_op_class``
GATED_REDUCTION_OPS = frozenset(("relu_gated_vector", "linear_gated_vector",
                                 "lowrank_relu_gated", "lowrank_linear_gated",
                                 "relu_gated_vector_pair"))   # the paired write (two gated gemvs)
LINEAR_REDUCTION_OPS = frozenset(("affine_subspace", "lora_oproj"))
#: the contract in one sentence — the report header and the exit-code footer
#: carry it, the module docstring states it at length
REDUCTION_CONTRACT = (
    "reduction ops (stacked mode): bf16-kernel-variance equivalence to the per-row path, bounded in "
    "concentration for every reduction op and, per op class, on the edit or at the gate — over all counted "
    f"reduction-op comparisons of a case the fraction of unequal ELEMENTS is <= {REDUCTION_FRAC_TOL:g}, in "
    f"any single group <= {REDUCTION_GROUP_FRAC_TOL:g} and in any single row <= {REDUCTION_ROW_FRAC_TOL:g} "
    "(the two concentration bounds are 10x the measured maxima, not mechanism); LINEAR reduction ops "
    f"({', '.join(sorted(LINEAR_REDUCTION_OPS))}): every unequal element's stacked edit is within "
    f"{REDUCTION_ULPS_EDIT_TOL:g} bf16 ulps of the per-row edit beyond one bf16 ulp of the per-row output "
    "(|applied - ref| <= 4 ulp(|ref - base|) + ulp(|ref|)); GATED reduction ops "
    f"({', '.join(sorted(GATED_REDUCTION_OPS))}): the stacked gate is within {REDUCTION_GATE_C:g} x 2^-24 of "
    "the per-row gate in units of the reduction's own magnitude, per gate element "
    f"(|g_stacked - g_per_row| <= {REDUCTION_GATE_C:g} * 2^-24 * sum_i |h_i w_i|), a ReLU flip only "
    "when the side that fired is within that same bound, the gates OBSERVED from the two applies on every "
    "unequal group (never recomputed), each gated-vector side's output the BIT-EXACT reconstruction base + eff * g "
    "* v from that side's observed gate (any other element is a wrong write, whatever the gate delta), the "
    "stacked-minus-per-row edit equal to eff * (g_stacked - g_per_row) * v within "
    f"{REDUCTION_GATE_CONSISTENCY_ULPS_TOL:g} bf16 ulps of the output + the edit, and, for the gated-VECTOR pair, "
    "the two writes EXACT on every row whose observed gates agree or which is masked off (the low-rank pair is "
    "not reconstructed and never held exact where it writes — its write reduces over k — so its zero-delta rows "
    f"are BOUNDED at the linear {REDUCTION_ULPS_EDIT_TOL:g}-edit-ulp unit, a one-output-ulp flip reading 0 excess, "
    "and the identity residual rules the group) — the edit-ulp unit is recorded there but is not a rule; "
    "elementwise ops, and EVERY op under the per-row mode: exact "
    "(n_unequal == 0). Any modified-flag disagreement, shadow error, record error, an unequal reduction-op "
    "group whose record lacks the tolerance fields, an unequal gated group whose record lacks observed gate "
    "stats, or a gate_error is a FAIL — the tolerance is for kernel variance, never for a defect or for "
    "missing evidence")
#: the attested apply paths (lens_worker.apply_path_string builds them; the
#: test suite pins these literals to it): a grouped worker names its reduction
#: mode right after the "vectorized" token, so a standup can never read as the
#: other mode; the all-per-row reference forms no group and stays "per_row"
APPLY_PATH_VECTORIZED = "vectorized"                    # the token
APPLY_PATH_PER_ROW = "per_row"
APPLY_PATH_VECTORIZED_STACKED = "vectorized+stacked"
APPLY_PATH_VECTORIZED_PER_ROW = "vectorized+per_row"
APPLY_PATH_SHADOW_SUFFIX = "+shadow(serves per_row)"
APPLY_PATH_SHADOW = APPLY_PATH_VECTORIZED_STACKED + APPLY_PATH_SHADOW_SUFFIX
APPLY_PATH_SHADOW_PER_ROW = APPLY_PATH_VECTORIZED_PER_ROW + APPLY_PATH_SHADOW_SUFFIX
#: a case's ROUTING CLASS (module docstring): whether its object is ever
#: applied on the vectorized (grouped) path — derived from the op phase
#: through lens_worker.phase_vectorizes_in_decode, the worker's own rule
ROUTING_VECTORIZED = "vectorized"
ROUTING_PER_SLOT = "per_slot"
#: the batch-invariance fields a worker's attestation record carries
BI_FIELDS = ("batch_invariant_env", "batch_invariant_active", "batch_invariant_error",
             "batch_invariant_surface")
#: the SelectionRecord's record-level PER-PATH counters (lens_worker.SelectionRecord):
#: which apply path made each application — the ``routing_*`` check's input
PATH_COUNTERS = ("n_applications_per_slot", "n_applications_grouped", "n_applications_unattributed",
                 "n_modifying_per_slot", "n_modifying_grouped", "n_modifying_unattributed")
#: re-drives of a shadow / multigroup wave until every ladder size / the
#: two-cell layer call shows up in the record
SHADOW_ATTEMPTS = 3
#: how many times an unattested multigroup wave is re-driven
COSCHEDULE_ATTEMPTS = 3
#: vLLM's batch-invariance switch (vllm_server.BATCH_INVARIANT_ENV)
BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
#: the harness SI staged / synthetic Class A objects are filed under (a label
#: — the object bytes are what is served; the SI text never enters a lens cell)
STAGE_SI = "si_constitution"
BANK_SIS = ("si_constitution", "safety_refusal", "si_irrelevant")
CANDSI_IDS = ("A10", "B3", "C16")
DEFAULT_BATCHES = (1, 8, 32)
DEFAULT_TIMING_BATCHES = (8, 32, 64)
PROMPT_TOKEN_TARGETS = (48, 96, 200, 400, 800, 1200)
FILLER = ("The river runs quietly past the old stone mill while the miller counts sacks "
          "of grain and the afternoon light moves slowly across the wooden floor. ")
#: the gate's roots under the cluster's shared root, the site's ``$SHARED`` (:func:`root_path`)
DEFAULT_RESULTS_ROOT = "$SHARED/exp/lens-vectorize-gate"
DEFAULT_OBJECTS_ROOT = "$SHARED/exp/objects/scale-12"
DEFAULT_CANDSI_ROOT = "$SHARED/exp/twosweep/fits/candsi_refit"
#: extra bank dirs searched after the scale-12 bank -> the gate family whose
#: objects they hold (the CONFIGURED origin; ``--extra-objects-dir PATH=<key>``).
#: stage3-wave lanes A and C are the 2026-08-24 stage-3 wave's Llama-3.1-8B
#: fits (configs/swp/conditions_wave_{A,C}.tsv: NousResearch@d10aef79, every
#: cell's train_budget_meta.json names that model) — llama8b's, and no other
#: family's. The mapping is a cross-check on (and, for a record-less cell, the
#: only substitute for) the object's own fit-time record; the re-gate at
#: aa2b8a1 served lane A's rank-1 LoRA to all five families because the dirs
#: were searched unkeyed (extra_object_provenance).
DEFAULT_EXTRA_OBJECTS: Dict[str, str] = {
    "$SHARED/exp/objects/stage3-wave/A": "llama8b",
    "$SHARED/exp/objects/stage3-wave/C": "llama8b",
}


def root_path(spec: str, user: str, *, strict: bool = True) -> str:
    """A root the gate reads or writes: ``{user}`` filled in, then the site's ``$SHARED`` (the job's SISTEER_SHARED_ROOT, else the
    site config's cluster block) expanded through ``scripts/site_paths.py``. An unresolvable ``$SHARED`` ends the run by name, or,
    with ``strict=False`` (a default directory the gate only searches), leaves the placeholder spelling, which names no directory."""
    spec = spec.format(user=user)
    if "$" not in spec:
        return spec
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import site_paths
    try:
        return str(site_paths.site_path(spec, "the gate's shared root"))
    except site_paths.SitePathError:
        if strict:
            raise
        return spec
#: inventory entries that are NOTES about real artifacts, never cases:
#: ``__unadaptable__`` (present but does not adapt) and ``__foreign__``
#: (another family's object under an extra dir — never served here)
INVENTORY_NOTES = ("__unadaptable__", "__foreign__")
#: synthetic-object dims used ONLY to derive the inventory (tiny, CPU);
#: ``oproj_shapes`` = {attention layer: (o_proj in_features, out_features)}
INVENTORY_DIMS = {"hidden_size": 16, "num_hidden_layers": 4,
                  "oproj_shapes": {1: (12, 16), 3: (12, 16)}}
#: the live multi-object case (module docstring) — mandatory for complete=true
MULTIGROUP = "multigroup"
#: the assistant seed of a declared prefill-seeded render
SEED_TEXT = "Certainly. The passage describes"
#: multigroup role labels; cellC maps to cellA when only two cells exist
MG_ROLES = ("passthrough", "scale0", "cellA", "cellB", "cellC",
            "cellA+seeded", "cellB+seeded", "scale0+seeded")
#: the role a prompt's SECOND row takes (multigroup_spec): always a different
#: role, and for every nonzero cell a DIFFERENT nonzero cell with the same
#: render, so every prompt yields a same-prompt cross-cell pair
MG_TWIN_ROLE = {"passthrough": "cellA", "scale0": "cellB", "cellA": "cellB", "cellB": "cellC",
                "cellC": "cellA", "cellA+seeded": "cellB+seeded", "cellB+seeded": "cellA+seeded",
                "scale0+seeded": "cellA+seeded"}
#: timing is measured on the production regime of the production kinds
#: (response-masked psr / LoRA / vector) and on the multigroup case; the
#: other variants get bit-identity verdicts only (timing is a property of
#: the op kind, not of the mask variant — and 29 standups must fit a job)
TIMING_KINDS = ("psr", "lora_all", "lora", "vector_all")


class GateError(RuntimeError):
    """A gate refusal (bad trees, unreadable evidence) — loud, never a verdict."""


class SignatureError(GateError):
    """A response without the evidence a bit comparison needs."""


# --------------------------------------------------------------------------- #
# The families                                                                 #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ModelServe:
    key: str
    hf: str
    rev: str
    cfg: str
    bank: str
    branch: str


MODELS: Dict[str, ModelServe] = {
    "llama8b": ModelServe(
        "llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "d10aef7999a2b5ba950ab3974312feeedbfe0b77", "configs/two_si.yaml", "llama8b",
        branch="plain: one FlashAttention metadata group, 1-D positions, model.layers"),
    "gemma4e4b": ModelServe(
        "gemma4e4b", "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "configs/two_si_gemma4e4b_reduced.yaml", "gemma4e4b",
        branch="gemma nested: sliding + full attention KV groups (two bearing metadata "
               "objects aliasing one buffer), the conditional-generation decoder accessor"),
    "qwen35_4b": ModelServe(
        "qwen35_4b", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "configs/two_si_qwen35_4b.yaml", "qwen35_4b",
        branch="hybrid GDN + M-RoPE: gated-deltanet non_spec_query_start_loc cross-check, "
               "(3, T) positions reduced from mrope_positions.cpu"),
    "glm46vflash": ModelServe(
        "glm46vflash", "zai-org/GLM-4.6V-Flash", "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "configs/two_si_glm46vflash.yaml", "glm46vflash",
        branch="dense M-RoPE: (3, T) positions from mrope_positions.cpu, language_model-nested "
               "decoder, vllm_patches M-RoPE layout"),
    "nemotron3nano4b": ModelServe(
        "nemotron3nano4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", "configs/two_si_nemotron3nano4b.yaml",
        "nemotron3nano4b",
        branch="hybrid Mamba2: Mamba2 metadata seq_lens cross-check (no num_actual_tokens / "
               "query_start_loc on the recurrent group), attention o_proj on 4 of 42 blocks"),
}
MODEL_ORDER: Tuple[str, ...] = tuple(MODELS)


# --------------------------------------------------------------------------- #
# Signatures + verdicts (pure)                                                 #
# --------------------------------------------------------------------------- #


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def signature(choice: Mapping[str, Any]) -> Dict[str, Any]:
    """The bit-identity signature of one chat-completion choice — or a
    :class:`SignatureError`. REQUIRED evidence: a non-empty list of int
    ``token_ids`` (``return_token_ids``), a ``logprobs.content`` list of
    exactly one record per token with a string ``token``, a finite numeric
    sampled-token ``logprob`` and a non-empty ``top_logprobs`` list of
    (token, finite logprob) pairs at EVERY position. Anything less is not a
    comparable signature (a gate over missing evidence would compare
    nothing to nothing and pass). Floats are kept as the JSON gave them
    (repr round-trips exactly), so ``==`` is a bit comparison."""
    ids = choice.get("token_ids")
    if not isinstance(ids, list) or not ids \
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise SignatureError(
            f"response carries no usable token_ids ({type(ids).__name__}, "
            f"{len(ids) if isinstance(ids, list) else 'n/a'} entries) — the server did "
            "not honor return_token_ids; token ids are required evidence")
    lp = (choice.get("logprobs") or {}).get("content")
    if not isinstance(lp, list) or len(lp) != len(ids):
        raise SignatureError(
            f"logprobs.content has {len(lp) if isinstance(lp, list) else 'no'} records for "
            f"{len(ids)} token ids — one aligned record per token is required")
    tokens, logprobs, top = [], [], []
    for i, rec in enumerate(lp):
        if not isinstance(rec, Mapping) or not isinstance(rec.get("token"), str):
            raise SignatureError(f"logprobs.content[{i}] is not a token record")
        v = rec.get("logprob")
        if not _finite_number(v):
            raise SignatureError(f"logprobs.content[{i}].logprob is {v!r}, not a finite number")
        tl = rec.get("top_logprobs")
        if not isinstance(tl, list) or not tl:
            raise SignatureError(f"logprobs.content[{i}] carries no top_logprobs")
        pairs = []
        for t in tl:
            if not isinstance(t, Mapping) or not isinstance(t.get("token"), str) \
                    or not _finite_number(t.get("logprob")):
                raise SignatureError(f"logprobs.content[{i}].top_logprobs has a malformed entry")
            pairs.append((t["token"], t["logprob"]))
        tokens.append(rec["token"])
        logprobs.append(v)
        top.append(sorted(pairs))
    return {"token_ids": list(ids), "tokens": tokens, "logprobs": logprobs, "top": top,
            "finish_reason": choice.get("finish_reason"),
            "text": (choice.get("message") or {}).get("content")}


def signatures_equal(a: Sequence[Mapping[str, Any]],
                     b: Sequence[Mapping[str, Any]]) -> Tuple[bool, Optional[str]]:
    if len(a) != len(b):
        return False, f"batch sizes differ: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        for key in ("token_ids", "tokens", "logprobs", "top", "finish_reason"):
            if x.get(key) != y.get(key):
                xs, ys = x.get(key), y.get(key)
                pos = None
                if isinstance(xs, list) and isinstance(ys, list):
                    pos = next((j for j, (u, v) in enumerate(zip(xs, ys)) if u != v),
                               min(len(xs), len(ys)))
                return False, f"request {i}: {key} differs at position {pos}"
    return True, None


def signatures_differ_somewhere(a: Sequence[Mapping[str, Any]],
                                b: Sequence[Mapping[str, Any]]) -> bool:
    """True when at least one request differs in at least one sampled-token
    logprob (or token) — the "steering bites" test."""
    for x, y in zip(a, b):
        if x.get("logprobs") != y.get("logprobs") or x.get("token_ids") != y.get("token_ids"):
            return True
    return False


def stable(runs: Sequence[Sequence[Mapping[str, Any]]]) -> Tuple[bool, Optional[str]]:
    if len(runs) < 2:
        return False, "fewer than two runs"
    for k, run in enumerate(runs[1:], start=2):
        ok, why = signatures_equal(runs[0], run)
        if not ok:
            return False, f"run 1 vs run {k}: {why}"
    return True, None


def verdict(ref_runs: Sequence[Sequence[Mapping[str, Any]]],
            cand_runs: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    ref_ok, ref_why = stable(ref_runs)
    cand_ok, cand_why = stable(cand_runs)
    if not ref_ok:
        return {"verdict": UNSTABLE_REF, "reference_stable": False,
                "candidate_stable": cand_ok, "detail": ref_why}
    if not cand_ok:
        return {"verdict": UNSTABLE_VEC, "reference_stable": True,
                "candidate_stable": False, "detail": cand_why}
    eq, why = signatures_equal(ref_runs[0], cand_runs[0])
    return {"verdict": PASS if eq else FAIL, "reference_stable": True,
            "candidate_stable": True, "detail": why}


def evidence_verdict(ref_runs: Sequence[Sequence[Mapping[str, Any]]],
                     cand_runs: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """A B > 1 cross-server comparison as EVIDENCE (module docstring): the
    underlying equality / stability facts, tagged so no exit code is ever
    derived from it."""
    v = verdict(ref_runs, cand_runs)
    return {"verdict": EVIDENCE, "evidence": True, "equal": v["verdict"] == PASS,
            "underlying": v["verdict"], "reference_stable": v["reference_stable"],
            "candidate_stable": v["candidate_stable"], "detail": v["detail"],
            "note": "B>1 cross-server comparison at BI=0 is composition-confounded: "
                    "recorded, never a verdict"}


def compared_at(snapshot: Mapping[str, Any], B: int) -> int:
    """Groups the shadow record compared at exactly ``B`` ROWS (keys end in
    ``:B<rows>``) — KERNEL-SHAPE EVIDENCE (which stacked shape the production
    kernels ran at), never wave coverage: admission splitting legitimately
    turns one HTTP wave of B requests into several smaller groups."""
    return sum(int(e.get("n_compared") or 0) for k, e in (snapshot.get("by_key") or {}).items()
               if isinstance(e, Mapping) and k.endswith(f":B{int(B)}"))


_ROWS_TAIL = re.compile(r":B(\d+)$")


def rows_compared(snapshot: Any) -> List[int]:
    """The distinct group row counts with a counted comparison in ``snapshot``
    (the ``:B<rows>`` tail of every key with ``n_compared > 0``) — the
    kernel-shape evidence the report carries next to each case."""
    if not isinstance(snapshot, Mapping):
        return []
    rows = set()
    for k, e in (snapshot.get("by_key") or {}).items():
        if isinstance(e, Mapping) and int(e.get("n_compared") or 0) > 0:
            m = _ROWS_TAIL.search(str(k))
            if m:
                rows.add(int(m.group(1)))
    return sorted(rows)


def shadow_wave_delta(before: Any, after: Any) -> Dict[str, Any]:
    """What ONE submitted wave added to the cumulative shadow record: the
    counter differences (``compared`` = counted, i.e. modified-on-some-side,
    comparisons; ``unequal``; ``unmodified``; ``shadow_errors``) plus the group
    row counts compared IN THAT WAVE (per-key ``n_compared`` differences). This
    is how a comparison is attributed to the wave that produced it."""
    def _n(snap, k):
        return int(snap.get(k) or 0) if isinstance(snap, Mapping) else 0
    bk_before = (before.get("by_key") or {}) if isinstance(before, Mapping) else {}
    bk_after = (after.get("by_key") or {}) if isinstance(after, Mapping) else {}
    rows = set()
    for k, e in bk_after.items():
        if not isinstance(e, Mapping):
            continue
        prev = bk_before.get(k) or {}
        d = int(e.get("n_compared") or 0) - int(prev.get("n_compared") or 0)
        if d > 0:
            m = _ROWS_TAIL.search(str(k))
            if m:
                rows.add(int(m.group(1)))
    return {"compared": _n(after, "n_groups_compared") - _n(before, "n_groups_compared"),
            "unequal": _n(after, "n_unequal") - _n(before, "n_unequal"),
            "unmodified": _n(after, "n_unmodified_groups") - _n(before, "n_unmodified_groups"),
            "shadow_errors": _n(after, "n_shadow_errors") - _n(before, "n_shadow_errors"),
            "rows_compared": sorted(rows)}


def case_routing(case_id: str) -> str:
    """The case's ROUTING CLASS (module docstring): ``vectorized`` when its op
    phase lets the worker apply it on a decode group, ``per_slot`` when every
    op is prefill-only — decided by ``lens_worker.phase_vectorizes_in_decode``,
    the worker's own rule, from the phase field the case id carries. A
    malformed id or an unknown phase is a GateError, never a default class."""
    from sisteer.capeval.serving import lens_worker as lw
    try:
        phase = parse_case_id(case_id)["phase"]
        vectorized = lw.phase_vectorizes_in_decode(phase)
    except (ValueError, lw.LensWorkerError) as e:
        raise GateError(f"case {case_id!r}: cannot derive its routing class: {e}")
    return ROUTING_VECTORIZED if vectorized else ROUTING_PER_SLOT


def compared_per_wave_size(waves: Any, batches: Sequence[int]) -> Dict[int, int]:
    """Counted shadow comparisons attributed to the submitted waves of each
    ladder size (summed over every wave of that size, re-driven or not)."""
    out = {int(B): 0 for B in batches}
    for w in waves or []:
        if not isinstance(w, Mapping):
            continue
        try:
            B = int(w.get("B"))
        except (TypeError, ValueError):
            continue
        if B in out:
            out[B] += int(w.get("compared_in_wave") or 0)
    return out


def check_reduction_mode(mode: Any) -> str:
    """``mode`` when it is one of :data:`REDUCTION_MODES`; GateError otherwise
    (a mode the gate does not know is a programming error, never a default)."""
    if mode not in REDUCTION_MODES:
        raise GateError(f"unknown reduction mode {mode!r}; expected one of {REDUCTION_MODES}")
    return str(mode)


def parse_reduction_mode_spec(spec: Any) -> str:
    """``--reduction-mode`` / :data:`REDUCTION_MODE_ENV`: unset or empty = the
    default (``stacked``); a mode name, or the env value the gate itself
    writes to :data:`STACKED_ENV` (``1`` = stacked, ``0`` = per_row); anything
    else refuses loudly — a typo must never pick a mode silently."""
    if spec is None or not str(spec).strip():
        return DEFAULT_REDUCTION_MODE
    v = str(spec).strip().lower()
    if v in (REDUCTION_MODE_STACKED, "1"):
        return REDUCTION_MODE_STACKED
    if v in (REDUCTION_MODE_PER_ROW, "0"):
        return REDUCTION_MODE_PER_ROW
    raise GateError(f"reduction mode {spec!r} is not one of {REDUCTION_MODES} (or 1 / 0)")


def expected_apply_path(mode: str, *, shadow: bool = False) -> str:
    """The apply path a grouped candidate standup must attest in reduction
    ``mode`` — ``lens_worker.apply_path_string``, the worker's OWN builder, so
    the gate's expectation cannot drift from what the worker writes."""
    from sisteer.capeval.serving import lens_worker as lw
    return lw.apply_path_string(check_reduction_mode(mode), shadow=bool(shadow))


def shadow_key_op_class(key: str) -> str:
    """:data:`OP_CLASS_REDUCTION_GATED` | :data:`OP_CLASS_REDUCTION_LINEAR` |
    :data:`OP_CLASS_ELEMENTWISE` from a shadow key's op token (index 1 of
    ``kind:op[+op]:shape:positions:skipbosN:phase:B<rows>``, several ops joined
    by ``+``), each op classified by ``lens_adapters.reduction_op_class`` (the
    adapters' own classification): GATED when ANY op on the key is a gated
    reduction op (the gate bound is the only rule that can read such a key;
    a linear op sharing it stays under the concentration bounds — no
    registered kind produces a mixed key), LINEAR when any op reduces and
    none is gated, ELEMENTWISE otherwise. A key that does not have the
    schema, names no op, or names an op the adapters do not classify is a
    GateError — never a default class."""
    from sisteer.capeval.serving import lens_adapters as la
    parts = str(key).split(":")
    if len(parts) < 7 or not _ROWS_TAIL.search(str(key)):
        raise GateError(f"shadow key {key!r} is not kind:op:shape:positions:skipbosN:phase:B<rows>; "
                        "cannot classify its op")
    ops = [o for o in parts[1].split("+") if o and o != "-"]
    if not ops:
        raise GateError(f"shadow key {key!r} names no op; cannot classify it")
    try:
        classes = [str(la.reduction_op_class(o)) for o in ops]
    except la.AdapterError as e:
        raise GateError(f"shadow key {key!r}: {e}")
    if OP_CLASS_REDUCTION_GATED in classes:
        return OP_CLASS_REDUCTION_GATED
    if OP_CLASS_REDUCTION_LINEAR in classes:
        return OP_CLASS_REDUCTION_LINEAR
    return OP_CLASS_ELEMENTWISE


def fmt_gate_units(x: Any) -> str:
    """A gate deviation ratio (|delta gate| / sum|h_i w_i|) in the contract's
    unit: ``0.8 x 2^-24``; ``n/a`` when unknown."""
    v = _opt_num(x)
    return "n/a" if v is None else f"{v / REDUCTION_GATE_UNIT:.3g} x 2^-24"


def _opt_num(x: Any) -> Optional[float]:
    """A record field as a float; None when missing or not a number."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(float(x)) else None


GATE_SUMMARY_KEYS = ("gate_max_abs_diff", "gate_max_rel_diff", "gate_min_abs_at_diff",
                     "gate_n_rows_unequal", "gate_n_relu_flips", "gate_max_diff_rel_sum",
                     "gate_max_flip_rel_sum", "gate_edit_consistency_max_ulps",
                     "gate_n_groups_with_stats", "gate_n_groups_observed", "gate_n_errors",
                     "gate_exact_rows_n", "gate_exact_rows_unequal_elems",
                     "gate_lowrank_zero_delta_max_ulps_edit",
                     "gate_n_groups_recon", "gate_write_recon_mismatch_elems",
                     # the gate-magnitude histogram (measurement only; never a rule)
                     "gate_hist_n_groups", "gate_hist_zero", "gate_hist_near", "gate_hist_mid", "gate_hist_far",
                     "gate_dose_share_near_threshold", "gate_g_over_f_min_nonzero", "gate_g_over_f_max")
#: the histogram fields, read from the case-wide summary and rendered by
#: ``gate_hist_brief`` in the report's case line and the FAIL brief
GATE_HIST_KEYS = ("gate_hist_n_groups", "gate_hist_zero", "gate_hist_near", "gate_hist_mid", "gate_hist_far",
                  "gate_dose_share_near_threshold", "gate_g_over_f_min_nonzero", "gate_g_over_f_max")


def shadow_key_has_vector_gate(key: str) -> bool:
    """Whether the shadow key's op token names a gated-VECTOR op
    (``relu_gated_vector`` / ``linear_gated_vector``) — the keys the write
    reconstruction applies to. False for a key that cannot be parsed (the
    classifier is what refuses those)."""
    from sisteer.capeval.serving import lens_adapters as la
    parts = str(key).split(":")
    if len(parts) < 7:
        return False
    return any(o in la._GATED_OPS for o in parts[1].split("+") if o and o != "-")


def gate_hist_brief(gate: Any) -> str:
    """The gate-magnitude histogram of a gate summary (case-wide or one
    group), in one clause — INFORMATIONAL (2026-09-06; the user's question
    about small gates): how many observed per-row gate elements sit at zero,
    within 10 noise floors of the ReLU (f = 2^-24 sum|h_i w_i|), between 10
    and 1000, and beyond; the share of the served dose |eff g| ||v|| that the
    near-threshold elements carry; and the smallest nonzero / largest g / f.
    Empty when the summary has no histogram (a worker before it existed)."""
    if not isinstance(gate, Mapping) or gate.get("gate_hist_zero") is None:
        return ""
    share = gate.get("gate_dose_share_near_threshold")
    return (f"gate magnitudes over {gate.get('gate_hist_n_groups')} compared group(s), equal and unequal "
            f"(per-row, in noise floors f): zero {gate.get('gate_hist_zero')}, "
            f"near (0 < g <= 10 f) {gate.get('gate_hist_near')}, mid (<= 1000 f) {gate.get('gate_hist_mid')}, "
            f"far {gate.get('gate_hist_far')}; near-threshold dose share "
            f"{'n/a' if _opt_num(share) is None else format(float(share), '.3g')}; g/f nonzero min "
            f"{_fmt_g(gate.get('gate_g_over_f_min_nonzero'))}, max {_fmt_g(gate.get('gate_g_over_f_max'))}")


def worst_brief(worst: Any, gate: Any) -> str:
    """The evidence tail of a FAIL detail (2026-09-05): where the unit peaked
    — the worst element's values and edits, its row's edit ratio / cosine, and
    that group's gate-scalar read — plus the case-wide gate summary. Empty
    when the record carries neither (a worker before the fields existed).
    Reading only; the rules above are the verdict."""
    parts: List[str] = []
    if isinstance(worst, Mapping):
        we, g = worst.get("worst_elem"), worst.get("gate")
        head = (f"worst (ulps_edit {_fmt_g(worst.get('max_ulps_edit'))}): key {worst.get('key')} layer "
                f"{worst.get('layer')} cell {worst.get('cell')} rows {worst.get('rows')}")
        if isinstance(we, Mapping) and "error" not in we:
            head += (f"; element [{we.get('row')}, {we.get('col')}] base {_fmt_g(we.get('base'))} per-row "
                     f"{_fmt_g(we.get('ref'))} stacked {_fmt_g(we.get('applied'))} (edits {_fmt_g(we.get('edit_ref'))} "
                     f"vs {_fmt_g(we.get('edit_applied'))}; |diff| {_fmt_g(we.get('abs_diff'))} = "
                     f"{_fmt_g(we.get('ulps_out'))} output ulps); its row: {we.get('row_n_unequal')}/"
                     f"{we.get('row_n_elems')} elements differ, ||per-row edit|| {_fmt_g(we.get('row_edit_norm'))}, "
                     f"||stacked edit||/||per-row edit|| {_fmt_g(we.get('row_edit_ratio'))}, cosine "
                     f"{_fmt_g(we.get('row_edit_cos'))}")
        if isinstance(g, Mapping) and "gate_error" not in g:
            head += (f"; gate ({','.join(g.get('gate_ops') or [])}): {g.get('gate_n_rows_unequal')}/"
                     f"{g.get('gate_n_rows')} rows differ, max |delta| {_fmt_g(g.get('gate_max_abs_diff'))} "
                     f"(rel {_fmt_g(g.get('gate_max_rel_diff'))}; smallest gate that differed "
                     f"{_fmt_g(g.get('gate_min_abs_at_diff'))}; |delta gate|/sum|hw| "
                     f"{fmt_gate_units(g.get('gate_max_diff_rel_sum'))}), ReLU flips {g.get('gate_n_relu_flips')}"
                     + (f" (fired side {fmt_gate_units(g.get('gate_max_flip_rel_sum'))} of sum|hw|)"
                        if g.get("gate_max_flip_rel_sum") is not None else "")
                     + (f", gate-edit identity {_fmt_g(g.get('gate_edit_consistency_max_ulps'))} ulps"
                        f" (worst {g.get('gate_edit_consistency_worst')})"
                        if g.get("gate_edit_consistency_max_ulps") is not None else "")
                     + (f", exact rows {g.get('gate_exact_rows_n')} with {g.get('gate_exact_rows_unequal_elems')} "
                        "unequal element(s)" if g.get("gate_exact_rows_n") is not None else "")
                     + (f", write reconstruction {g.get('gate_write_recon_mismatch_elems')} mismatch(es) of "
                        f"{g.get('gate_write_recon_elems')}" if g.get("gate_write_recon_mismatch_elems") is not None
                        else (f", write reconstruction skipped: {g.get('gate_write_recon_skipped')}"
                              if g.get("gate_write_recon_skipped") else ""))
                     + (f", low-rank zero-delta rows {_fmt_g(g.get('gate_lowrank_zero_delta_max_ulps_edit'))} edit ulps"
                        if g.get("gate_lowrank_zero_delta_max_ulps_edit") is not None else "")
                     + (f", gates {g.get('gate_source')}" if g.get("gate_source") else ", gates UNSOURCED")
                     + f", samples [per_row, stacked, sum|hw|] {g.get('gate_rows_unequal_sample')}")
        hb = gate_hist_brief(g)
        if hb:
            head += "; " + hb
        parts.append(head)
    if isinstance(gate, Mapping) and gate:
        def _one(k: str, v: Any) -> str:
            if k in ("gate_max_diff_rel_sum", "gate_max_flip_rel_sum") and v is not None:
                return f"{k[5:]} {fmt_gate_units(v)}"
            return f"{k[5:]} {_fmt_g(v) if isinstance(v, (int, float)) else v}"
        parts.append("case-wide gate: " + ", ".join(_one(k, v) for k, v in gate.items() if k not in GATE_HIST_KEYS))
        hb = gate_hist_brief(gate)
        if hb:
            parts.append("case-wide " + hb)
    return ("; " + "; ".join(parts)) if parts else ""


def reduction_tolerance(snapshot: Any, *, reduction_mode: str = DEFAULT_REDUCTION_MODE,
                        frac_tol: float = REDUCTION_FRAC_TOL) -> Dict[str, Any]:
    """THE REDUCTION CONTRACT (module docstring) applied to one case's shadow
    record, pure: the per-case verdict facts. ``ok`` True / False (None only
    when there is no record to read), a ``detail`` naming the numbers and the
    bounds in the same units, the pooled reduction-op numbers
    (``n_unequal_elems`` / ``n_elems_compared`` over EVERY reduction-class key
    of either class, equal groups included, ``frac_unequal_elems``,
    ``max_frac_row``, ``max_frac_group``), the edit unit split by class
    (``max_ulps_edit`` over every reduction key — reported; ``max_ulps_edit_
    linear`` — the LINEAR rule's input; ``max_ulps_edit_gated`` —
    informational), the gate rule's inputs (``max_gate_diff_rel_sum``,
    ``max_gate_flip_rel_sum``, ``n_gate_relu_flips`` over the GATED keys), a
    ``per_key`` table (key, class, counts, the tolerance fields, the gate
    fields), the unequal keys by class (``reduction_keys_unequal`` = both
    classes, ``linear_keys_unequal``, ``gated_keys_unequal``,
    ``elementwise_keys_unequal``) and ``n_flag_disagreements``.

    Rules, in order: a key whose op cannot be classified, an unequal group
    attributed to no key (``n_keys_dropped``), or any modified-flag
    disagreement -> FAIL; no unequal group anywhere -> ok ("exact") in either
    mode; per_row mode with any unequal group -> FAIL; stacked mode: an
    elementwise-class key with an unequal group -> FAIL; an unequal
    reduction-class key (either class) whose record lacks the tolerance fields
    (``n_elems_compared`` 0, ``n_unequal_elems`` None / 0, ``max_ulps_edit``
    None, or ``max_frac_row`` / ``max_frac_group`` None / 0.0 while elements
    differed) -> FAIL ("lacks the tolerance fields"); an unequal GATED key
    whose record lacks gate stats on every unequal group
    (``gate_n_groups_with_stats`` != its ``n_unequal``, no
    ``gate_max_diff_rel_sum``, or ReLU flips without ``gate_max_flip_rel_sum``)
    -> FAIL ("lacks gate stats"); any ``gate_error`` (per key or case-wide) ->
    FAIL — all three are missing evidence, never kernel variance; then, with
    every field present: pooled fraction > :data:`REDUCTION_FRAC_TOL`, any
    group's fraction > :data:`REDUCTION_GROUP_FRAC_TOL`, any row's fraction >
    :data:`REDUCTION_ROW_FRAC_TOL` -> FAIL (both classes); a LINEAR key's
    ``max_ulps_edit`` > :data:`REDUCTION_ULPS_EDIT_TOL` -> FAIL; a GATED key's
    ``gate_max_diff_rel_sum`` > :data:`REDUCTION_GATE_REL_SUM_TOL` (C x 2^-24)
    or a ReLU flip whose fired side exceeds the same bound -> FAIL (a gated
    key's ``max_ulps_edit`` is reported, never judged); else ok. (A
    ``max_ulps_edit`` of exactly 0.0 with elements differing is legitimate: a
    dropped or flipped one-output-ulp edit reads 0 per element — the
    concentration bounds are what judge it.)

    ``frac_tol`` is the pooled-fraction bound, :data:`REDUCTION_FRAC_TOL` by
    default; the gate's own callers never pass it. That constant is a
    CASE-WIDE bound whose measured basis is cases of thousands of groups. The
    pooled fraction is the element-weighted mean of the per-group fractions,
    so over a handful of groups it is a few host-dependent flip counts, not a
    rate — how many bf16 output roundings a fp32 gate delta flips follows the
    host's BLAS summation order, and one byte-identical 2-group bf16 fixture
    read 1.68e-4 on one CPU and below 1e-4 on another. A caller judging such
    a record passes the scale honest for its size; the verdict records the
    bound it was judged at (``frac_tol``). The bound may never exceed
    :data:`REDUCTION_GROUP_FRAC_TOL` (above it the clause is vacuous: the
    pooled fraction never exceeds the largest group's), nor be non-positive
    — GateError otherwise."""
    mode = check_reduction_mode(reduction_mode)
    frac_tol = float(frac_tol)
    if not (0.0 < frac_tol <= REDUCTION_GROUP_FRAC_TOL):
        raise GateError(f"frac_tol must lie in (0, {REDUCTION_GROUP_FRAC_TOL:g}] — the pooled fraction never "
                        f"exceeds the largest group's, so a looser bound is vacuous; got {frac_tol!r}")
    out: Dict[str, Any] = {
        "ok": None, "detail": "", "reduction_mode": mode,
        "frac_tol": frac_tol, "ulps_edit_tol": REDUCTION_ULPS_EDIT_TOL,
        "group_frac_tol": REDUCTION_GROUP_FRAC_TOL, "row_frac_tol": REDUCTION_ROW_FRAC_TOL,
        "gate_c": REDUCTION_GATE_C, "gate_unit": REDUCTION_GATE_UNIT,
        "gate_rel_sum_tol": REDUCTION_GATE_REL_SUM_TOL,
        "gate_consistency_ulps_tol": REDUCTION_GATE_CONSISTENCY_ULPS_TOL,
        "gate_exact_rows_unequal_tol": REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL,
        "gate_write_recon_mismatch_tol": REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL,
        "n_groups_compared": 0, "n_unequal": 0,
        "n_reduction_compared": 0, "n_reduction_unequal": 0,
        "n_unequal_elems": 0, "n_elems_compared": 0, "frac_unequal_elems": None,
        "max_ulps_edit": None, "max_ulps_edit_linear": None, "max_ulps_edit_gated": None,
        "max_frac_row": None, "max_frac_group": None,
        "max_gate_diff_rel_sum": None, "max_gate_flip_rel_sum": None, "n_gate_relu_flips": 0,
        "max_gate_edit_consistency_ulps": None,
        "n_gate_groups_observed": 0, "gate_exact_rows_n": 0, "gate_exact_rows_unequal_elems": 0,
        "max_gate_lowrank_zero_delta_ulps_edit": None,
        "n_gate_groups_recon": 0, "gate_write_recon_mismatch_elems": 0, "vector_gated_keys_unequal": [],
        "max_abs_diff": None, "n_flag_disagreements": 0,
        "per_key": [], "reduction_keys_unequal": [], "linear_keys_unequal": [], "gated_keys_unequal": [],
        "elementwise_keys_unequal": [],
        "unclassified_keys": [], "keys_lacking_fields": [], "keys_lacking_gate_stats": [],
        "keys_with_gate_errors": [],
        # evidence for READING a violation (2026-09-05), never a rule: the unequal
        # group where max_ulps_edit peaks (with its worst_elem / gate sub-records)
        # and the record's gate summary (whose rel_sum / flip / n_groups / n_errors
        # totals ARE, since 2026-09-06, the gated rule's case-wide inputs)
        "worst_unequal": None, "gate": {}}
    if not isinstance(snapshot, Mapping) or snapshot.get("rpc") == "absent":
        out["detail"] = "no sisteer_shadow_compare record to read"
        return out
    out["worst_unequal"] = (dict(snapshot["worst_unequal"])
                            if isinstance(snapshot.get("worst_unequal"), Mapping) else None)
    out["gate"] = {k: snapshot.get(k) for k in GATE_SUMMARY_KEYS if k in snapshot}
    n_unequal = int(snapshot.get("n_unequal") or 0)
    n_cmp = int(snapshot.get("n_groups_compared") or 0)
    flags = int(snapshot.get("n_flag_disagreements") or 0)
    case_gate_errors = int(_opt_num(snapshot.get("gate_n_errors")) or 0)
    mad = snapshot.get("max_abs_diff")
    first = snapshot.get("first_unequal")
    out.update(n_groups_compared=n_cmp, n_unequal=n_unequal, n_flag_disagreements=flags,
               max_abs_diff=mad)
    keyed_unequal = 0
    red_cmp = red_ue = red_ue_elems = red_elems = 0
    all_max_ulps: Optional[float] = None
    lin_max_ulps: Optional[float] = None
    gated_max_ulps: Optional[float] = None
    red_max_row: Optional[float] = None
    red_max_group: Optional[float] = None
    gate_max_rs: Optional[float] = None
    gate_max_fr: Optional[float] = None
    gate_max_cons: Optional[float] = None
    gate_max_lz: Optional[float] = None
    n_gate_flips = n_gate_obs = 0
    gate_exact_n = gate_exact_ne = 0
    n_recon = recon_ne = 0
    for k, e in (snapshot.get("by_key") or {}).items():
        if not isinstance(e, Mapping):
            continue
        n_c, n_u = int(e.get("n_compared") or 0), int(e.get("n_unequal") or 0)
        keyed_unequal += n_u
        n_el = int(_opt_num(e.get("n_elems_compared")) or 0)
        n_ue = _opt_num(e.get("n_unequal_elems"))
        mue = _opt_num(e.get("max_ulps_edit"))
        mfr = _opt_num(e.get("max_frac_row"))
        mfg = _opt_num(e.get("max_frac_group"))
        g_rs = _opt_num(e.get("gate_max_diff_rel_sum"))
        g_fr = _opt_num(e.get("gate_max_flip_rel_sum"))
        g_cons = _opt_num(e.get("gate_edit_consistency_max_ulps"))
        g_obs = int(_opt_num(e.get("gate_n_groups_observed")) or 0)
        g_exn = _opt_num(e.get("gate_exact_rows_n"))
        g_exne = _opt_num(e.get("gate_exact_rows_unequal_elems"))
        g_lz = _opt_num(e.get("gate_lowrank_zero_delta_max_ulps_edit"))
        g_nrec = int(_opt_num(e.get("gate_n_groups_recon")) or 0)
        g_rne = _opt_num(e.get("gate_write_recon_mismatch_elems"))
        g_ns = int(_opt_num(e.get("gate_n_groups_with_stats")) or 0)
        g_ne = int(_opt_num(e.get("gate_n_errors")) or 0)
        g_fl = int(_opt_num(e.get("gate_n_relu_flips")) or 0)
        try:
            cls: Optional[str] = shadow_key_op_class(k)
        except GateError as ex:
            cls = None
            out["unclassified_keys"].append(str(ex))
        out["per_key"].append({"key": str(k), "class": cls, "n_compared": n_c, "n_unequal": n_u,
                               "n_unequal_elems": (int(n_ue) if n_ue is not None else None),
                               "n_elems_compared": n_el, "max_ulps_edit": mue,
                               "max_frac_row": mfr, "max_frac_group": mfg,
                               "max_abs_diff": e.get("max_abs_diff"),
                               "gate_max_rel_diff": _opt_num(e.get("gate_max_rel_diff")),
                               "gate_n_rows_unequal": e.get("gate_n_rows_unequal"),
                               "gate_n_relu_flips": e.get("gate_n_relu_flips"),
                               "gate_max_diff_rel_sum": g_rs, "gate_max_flip_rel_sum": g_fr,
                               "gate_edit_consistency_max_ulps": g_cons,
                               "gate_n_groups_with_stats": e.get("gate_n_groups_with_stats"),
                               "gate_n_groups_observed": e.get("gate_n_groups_observed"),
                               "gate_exact_rows_n": g_exn, "gate_exact_rows_unequal_elems": g_exne,
                               "gate_lowrank_zero_delta_max_ulps_edit": g_lz,
                               "gate_n_groups_recon": g_nrec, "gate_write_recon_mismatch_elems": g_rne,
                               "gate_n_errors": e.get("gate_n_errors")})
        if cls in OP_CLASSES_REDUCTION:
            red_cmp += n_c
            red_elems += n_el
            if n_u > 0:
                out["reduction_keys_unequal"].append(str(k))
                out["linear_keys_unequal" if cls == OP_CLASS_REDUCTION_LINEAR else "gated_keys_unequal"].append(str(k))
                red_ue += n_u
                red_ue_elems += int(n_ue or 0)
                # max_ulps_edit may legitimately be 0.0 (a one-output-ulp move); the
                # concentration fields cannot be 0 when an element differed
                if n_el == 0 or not n_ue or mue is None or not mfr or not mfg:
                    out["keys_lacking_fields"].append(str(k))
                    continue
                red_max_row = max(red_max_row or 0.0, mfr)
                red_max_group = max(red_max_group or 0.0, mfg)
                all_max_ulps = max(all_max_ulps or 0.0, mue)
                if cls == OP_CLASS_REDUCTION_LINEAR:
                    lin_max_ulps = max(lin_max_ulps or 0.0, mue)
                    continue
                gated_max_ulps = max(gated_max_ulps or 0.0, mue)
                if g_ne > 0:
                    out["keys_with_gate_errors"].append(str(k))
                # every unequal gated group must carry OBSERVED gate stats — gate_source
                # 'observed' on every one (the provenance count, not the worst record's
                # label), the ratio the rule reads, the gate-edit identity residual and the
                # exact-rows count (absent from a recomputation-only record), and a ReLU
                # flip's fired-side ratio
                # ... and a gated-VECTOR key must have every unequal group's two outputs rebuilt
                # from the observed gates (the write reconstruction; the low-rank pair is never
                # rebuilt — its write reduces — and needs none)
                vector_gated = shadow_key_has_vector_gate(k)
                if vector_gated:
                    out["vector_gated_keys_unequal"].append(str(k))
                if (g_ns != n_u or g_obs != n_u or g_rs is None or g_cons is None or g_exne is None
                        or (g_fl > 0 and g_fr is None)
                        or (vector_gated and (g_nrec != n_u or g_rne is None))):
                    out["keys_lacking_gate_stats"].append(str(k))
                    continue
                n_gate_obs += g_obs
                n_recon += g_nrec
                recon_ne += int(g_rne or 0)
                gate_max_rs = max(gate_max_rs or 0.0, g_rs)
                gate_max_cons = max(gate_max_cons or 0.0, g_cons)
                gate_exact_n += int(g_exn or 0)
                gate_exact_ne += int(g_exne)
                if g_lz is not None:
                    gate_max_lz = max(gate_max_lz or 0.0, g_lz)
                if g_fr is not None:
                    gate_max_fr = max(gate_max_fr or 0.0, g_fr)
                n_gate_flips += g_fl
        elif cls == OP_CLASS_ELEMENTWISE and n_u > 0:
            out["elementwise_keys_unequal"].append(str(k))
    frac = (red_ue_elems / red_elems) if red_elems else None
    out.update(n_reduction_compared=red_cmp, n_reduction_unequal=red_ue, n_unequal_elems=red_ue_elems,
               n_elems_compared=red_elems, frac_unequal_elems=frac, max_ulps_edit=all_max_ulps,
               max_ulps_edit_linear=lin_max_ulps, max_ulps_edit_gated=gated_max_ulps,
               max_frac_row=red_max_row, max_frac_group=red_max_group,
               max_gate_diff_rel_sum=gate_max_rs, max_gate_flip_rel_sum=gate_max_fr,
               max_gate_edit_consistency_ulps=gate_max_cons, n_gate_relu_flips=n_gate_flips,
               n_gate_groups_observed=n_gate_obs, gate_exact_rows_n=gate_exact_n,
               gate_exact_rows_unequal_elems=gate_exact_ne, max_gate_lowrank_zero_delta_ulps_edit=gate_max_lz,
               n_gate_groups_recon=n_recon, gate_write_recon_mismatch_elems=recon_ne)
    fails: List[str] = []
    if out["unclassified_keys"]:
        fails.append(f"{len(out['unclassified_keys'])} shadow key(s) whose op the adapters do not "
                     f"classify (missing evidence, never a default class): {out['unclassified_keys'][:2]}")
    if flags > 0:
        fails.append(f"{flags} unequal group(s) where the two paths disagreed on whether the rows were "
                     "modified at all — a routing/masking defect, never kernel variance")
    if n_unequal > keyed_unequal:
        fails.append(f"{n_unequal - keyed_unequal} unequal group(s) attributed to no key "
                     f"({snapshot.get('n_keys_dropped')} key(s) dropped) — unclassifiable")
    if case_gate_errors > 0:
        fails.append(f"{case_gate_errors} gate recomputation(s) raised (gate_error; first: "
                     f"{snapshot.get('first_gate_error')}) — missing evidence, never kernel variance")
    if n_unequal == 0 and not fails:
        out["ok"] = True
        out["detail"] = f"exact ({mode} mode): every compared group equal"
        return out
    C = REDUCTION_GATE_C
    if mode == REDUCTION_MODE_PER_ROW and n_unequal > 0:
        fails.append(f"per_row mode: {n_unequal} unequal group(s) of {n_cmp} — every op must be EXACT "
                     "under the per-row mode (the same single-row call as the per-request path)")
    elif mode == REDUCTION_MODE_STACKED:
        if out["elementwise_keys_unequal"]:
            fails.append(f"{len(out['elementwise_keys_unequal'])} ELEMENTWISE-op key(s) with unequal "
                         f"groups (exact required in every mode): {out['elementwise_keys_unequal'][:3]}")
        if out["keys_lacking_fields"]:
            fails.append(f"{len(out['keys_lacking_fields'])} unequal reduction-op key(s) whose record "
                         "lacks the tolerance fields (n_elems_compared / n_unequal_elems / "
                         f"max_ulps_edit / max_frac_row / max_frac_group): {out['keys_lacking_fields'][:3]} "
                         "— missing evidence, never kernel variance")
        if out["keys_lacking_gate_stats"]:
            fails.append(f"{len(out['keys_lacking_gate_stats'])} unequal GATED reduction-op key(s) whose "
                         "record lacks gate stats on every unequal group (gate_n_groups_with_stats == n_unequal "
                         "AND gate_n_groups_observed == n_unequal — gates observed on every unequal group, not a "
                         "recomputation — gate_max_diff_rel_sum, gate_edit_consistency_max_ulps, "
                         "gate_exact_rows_unequal_elems, gate_max_flip_rel_sum for any ReLU flip, and for a "
                         "gated-vector key gate_n_groups_recon == n_unequal with gate_write_recon_mismatch_elems — "
                         "every unequal group's two outputs rebuilt from the observed gates): "
                         f"{out['keys_lacking_gate_stats'][:3]} — missing evidence, never kernel variance")
        if out["keys_with_gate_errors"]:
            fails.append(f"{len(out['keys_with_gate_errors'])} gated reduction-op key(s) with a gate_error "
                         f"(the gate recomputation raised): {out['keys_with_gate_errors'][:3]} — missing "
                         "evidence, never kernel variance")
        lacking = bool(out["keys_lacking_fields"] or out["keys_lacking_gate_stats"]
                       or out["keys_with_gate_errors"] or case_gate_errors)
        if not lacking and red_ue > 0:
            if frac is None or frac > frac_tol:
                fails.append(f"fraction of unequal reduction-op elements "
                             f"{'n/a' if frac is None else format(frac, '.2e')} > {frac_tol:g} "
                             f"({red_ue_elems} of {red_elems} elements)")
            if red_max_group is None or red_max_group > REDUCTION_GROUP_FRAC_TOL:
                fails.append(f"largest unequal fraction in one GROUP "
                             f"{'n/a' if red_max_group is None else format(red_max_group, '.2e')} > "
                             f"{REDUCTION_GROUP_FRAC_TOL:g} (concentration: a group-level defect, not kernel "
                             "variance)")
            if red_max_row is None or red_max_row > REDUCTION_ROW_FRAC_TOL:
                fails.append(f"largest unequal fraction in one ROW "
                             f"{'n/a' if red_max_row is None else format(red_max_row, '.2e')} > "
                             f"{REDUCTION_ROW_FRAC_TOL:g} (concentration: a row's edit dropped, scaled or "
                             "misplaced, not kernel variance)")
            if out["linear_keys_unequal"] and (lin_max_ulps is None or lin_max_ulps > REDUCTION_ULPS_EDIT_TOL):
                fails.append(f"largest unequal element of a LINEAR reduction op {lin_max_ulps:g} bf16 ulps of "
                             f"the per-row edit beyond one output ulp > {REDUCTION_ULPS_EDIT_TOL:g} (the stacked "
                             "edit is not the per-row edit)" if lin_max_ulps is not None
                             else "no max_ulps_edit recorded for the unequal linear reduction-op groups")
            if out["gated_keys_unequal"]:
                if gate_max_rs is None or gate_max_rs > REDUCTION_GATE_REL_SUM_TOL:
                    fails.append(f"largest |delta gate|/sum|hw| of a GATED reduction op "
                                 f"{fmt_gate_units(gate_max_rs)} > {C:g} x 2^-24 (the stacked gate is not the "
                                 "per-row gate within the reduction's own rounding; edit-ulps "
                                 f"{_fmt_g(gated_max_ulps)}, informational)")
                if gate_max_fr is not None and gate_max_fr > REDUCTION_GATE_REL_SUM_TOL:
                    fails.append(f"a ReLU flip of a GATED reduction op whose fired side is "
                                 f"{fmt_gate_units(gate_max_fr)} of sum|hw| > {C:g} x 2^-24 (a threshold crossing "
                                 f"beyond the reduction's own rounding; {n_gate_flips} flip(s))")
                if gate_max_cons is None or gate_max_cons > REDUCTION_GATE_CONSISTENCY_ULPS_TOL:
                    fails.append(f"the gate-edit identity of a GATED reduction op breaks by "
                                 f"{_fmt_g(gate_max_cons)} bf16 ulps (of the output + the edit) > "
                                 f"{REDUCTION_GATE_CONSISTENCY_ULPS_TOL:g}: applied - ref is not "
                                 "eff * (g_stacked - g_per_row) * v (a misplaced gate or a wrong write, not "
                                 "kernel variance)")
                if recon_ne > REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL:
                    fails.append(f"{recon_ne} element(s) of a GATED reduction op where the grouped or the per-row output "
                                 "is not the bit-exact reconstruction base + eff * g * v from that side's OBSERVED gate "
                                 f"(write consistency, {n_recon} group(s) rebuilt): a wrong write, never kernel variance "
                                 "— a nonzero gate delta cannot write where the direction is zero, nor by other than "
                                 "eff * delta g * v where it is not")
                if gate_exact_ne > REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL:
                    fails.append(f"{gate_exact_ne} unequal element(s) of a GATED reduction op in rows whose observed "
                                 f"gates are bit-identical or masked off ({gate_exact_n} such row(s)): after an equal "
                                 "gate a gated-vector op is elementwise, so the two writes must be exact — a wrong "
                                 "write, never kernel variance")
                if gate_max_lz is not None and gate_max_lz > REDUCTION_ULPS_EDIT_TOL:
                    fails.append(f"the low-rank write on rows whose observed latents are bit-identical departs by "
                                 f"{gate_max_lz:g} bf16 ulps of the per-row edit beyond one output ulp > "
                                 f"{REDUCTION_ULPS_EDIT_TOL:g} (the write's own k-term reduction reads <= ~1; the "
                                 "LINEAR unit applies there)")
    if fails:
        out["ok"] = False
        out["detail"] = ("; ".join(fails) + f"; {n_unequal} unequal group(s) of {n_cmp} compared, "
                         f"max |diff| {mad}; first: {first}" + worst_brief(out["worst_unequal"], out["gate"]))
        return out
    out["ok"] = True
    by_class = ""
    if out["linear_keys_unequal"]:
        by_class += (f"; linear ops: largest {lin_max_ulps:g} bf16 ulps of the per-row edit beyond one output "
                     f"ulp (<= {REDUCTION_ULPS_EDIT_TOL:g})")
    if out["gated_keys_unequal"]:
        by_class += (f"; gated ops: |delta gate|/sum|hw| = {fmt_gate_units(gate_max_rs)} <= {C:g} x 2^-24, "
                     + (f"{n_gate_flips} ReLU flip(s) with fired side {fmt_gate_units(gate_max_fr)} <= {C:g} x 2^-24"
                        if n_gate_flips else "no ReLU flips")
                     + f", gate-edit identity within {_fmt_g(gate_max_cons)} ulps "
                     f"(<= {REDUCTION_GATE_CONSISTENCY_ULPS_TOL:g}; gates observed on all {n_gate_obs} unequal "
                     f"group(s)), writes exact on the {gate_exact_n} row(s) whose gates agree"
                     + (f", both outputs the bit-exact reconstruction from the observed gates on all {n_recon} "
                        "vector-gated group(s)" if n_recon else "")
                     + (f", low-rank zero-delta rows within {gate_max_lz:g} edit ulps (<= {REDUCTION_ULPS_EDIT_TOL:g})"
                        if gate_max_lz is not None else "")
                     + f" (edit-ulps {_fmt_g(gated_max_ulps)} recorded, not a rule)")
    out["detail"] = (f"within the reduction contract ({mode} mode): {red_ue} unequal reduction-op "
                     f"group(s) of {red_cmp}, {red_ue_elems} of {red_elems} elements unequal (frac "
                     f"{frac:.2e} <= {frac_tol:g}; max in one group {red_max_group:.2e} <= "
                     f"{REDUCTION_GROUP_FRAC_TOL:g}, in one row {red_max_row:.2e} <= "
                     f"{REDUCTION_ROW_FRAC_TOL:g}){by_class}; elementwise groups exact; "
                     f"max |diff| {mad}; first: {first}")
    return out


def reduction_mode_evidence_ok(ev: Any, reduction_mode: str) -> Tuple[bool, str]:
    """The ``reduction_mode_*`` hard check: the SelectionRecord evidence
    (``selection_evidence``) attests ``lens_reduction_mode`` EQUAL to the
    gate's mode. A missing key (a worker from a tree that predates the
    attestation), a None, or workers that disagree fails — never read as
    either mode."""
    mode = check_reduction_mode(reduction_mode)
    got = ev.get(LENS_REDUCTION_MODE_KEY) if isinstance(ev, Mapping) else None
    if got is None:
        seen = ev.get("lens_reduction_modes_seen") if isinstance(ev, Mapping) else None
        return False, (f"the worker's SelectionRecord attests no {LENS_REDUCTION_MODE_KEY} "
                       + (f"(workers disagree: {seen}) " if seen and len(seen) > 1 else
                          "(a worker from a tree that predates the reduction-mode attestation) ")
                       + f"— the gate runs the {mode!r} mode")
    return got == mode, (f"the worker attests {LENS_REDUCTION_MODE_KEY}={got!r}; the gate runs the "
                         f"{mode!r} mode")


def reduction_mode_endpoint_ok(endpoint: Any, reduction_mode: Optional[str]) -> Tuple[bool, str]:
    """The standup-time refusal (``start_standup``, like the prefix-caching
    one): a standup the gate SET to ``reduction_mode`` must have its helper
    attest that very mode in its endpoint (``lens_reduction_mode``, read by
    scripts/lens_gate_standup.py from every worker's scheduler-config report
    through ``lens_worker.attested_reduction_mode``; None + ``lens_reduction_
    mode_error`` when the workers attest none or disagree). A None, an
    unknown value or the other mode refuses — a standup whose arithmetic is
    not the one it will be judged under does not run. ``reduction_mode``
    None (the base-commit reference, whose workers predate the attestation)
    requires nothing and records what is there."""
    ep = endpoint if isinstance(endpoint, Mapping) else {}
    got, err, raw = ep.get(LENS_REDUCTION_MODE_KEY), ep.get("lens_reduction_mode_error"), \
        ep.get("stacked_reductions_env")
    if reduction_mode is None:
        return True, (f"no reduction mode required of this standup (the base-commit reference); the "
                      f"endpoint attests {LENS_REDUCTION_MODE_KEY}={got!r}"
                      + (f" ({err})" if err else ""))
    mode = check_reduction_mode(reduction_mode)
    if not isinstance(endpoint, Mapping):
        return False, f"no endpoint record to read {LENS_REDUCTION_MODE_KEY} from (gate mode {mode!r})"
    if got == mode:
        return True, (f"the standup attests {LENS_REDUCTION_MODE_KEY}={got!r} = the gate's mode "
                      f"({STACKED_ENV}={raw!r})")
    return False, (f"the gate set the {mode!r} mode ({STACKED_ENV} was to be "
                   f"{'1' if mode == REDUCTION_MODE_STACKED else '0'}; the helper saw {raw!r}) but the "
                   f"standup attests {LENS_REDUCTION_MODE_KEY}={got!r}"
                   + (f" ({err})" if err else "")
                   + " — a standup whose served arithmetic is not the one it is judged under does not run")


def shadow_check(snapshot: Any, batches: Sequence[int], *, routing: Any,
                 waves: Any = None, reduction_mode: str = DEFAULT_REDUCTION_MODE
                 ) -> Tuple[Optional[bool], str]:
    """The shadow compare verdict for one case. False (FAIL) when the record
    is outside THE REDUCTION CONTRACT (``reduction_tolerance`` under
    ``reduction_mode``: exact for elementwise ops and for every op under the
    per-row mode, the bounded bf16-kernel-variance tolerance for reduction ops
    under the stacked mode; flag disagreements and records without the
    tolerance fields fail), on any shadow error / record error, when the
    standup did not run the shadow at all, or when a PER-SLOT case produced a
    counted comparison (the gate's routing class and the worker's grouping
    disagree). For a VECTORIZED case: None (INCONCLUSIVE) unless some
    submitted wave at EVERY ladder size carried >= 1 counted comparison
    (``waves`` — the per-wave attribution ``drive_shadow`` records); True when
    it did and the record is inside the contract, the detail naming the
    pooled numbers and the bounds whenever a group was unequal. For a
    PER-SLOT case: True with zero comparisons (the op is applied on the
    per-request path by design; ``routing_*`` is its hard check). Group row
    counts are reported as kernel-shape evidence only."""
    if not isinstance(snapshot, Mapping) or snapshot.get("rpc") == "absent":
        return False, "no sisteer_shadow_compare record from the shadow standup"
    if not snapshot.get("enabled"):
        return False, "the shadow standup did not enable the shadow compare"
    tol = reduction_tolerance(snapshot, reduction_mode=reduction_mode)
    if tol["ok"] is not True:
        return False, tol["detail"]
    if int(snapshot.get("n_shadow_errors") or 0) > 0:
        return False, f"shadow errors: {snapshot.get('first_shadow_error')}"
    if int(snapshot.get("n_record_errors") or 0) > 0:
        return False, f"record errors: {snapshot.get('first_record_error')}"
    n_cmp = int(snapshot.get("n_groups_compared") or 0)
    rows = rows_compared(snapshot)
    if routing == ROUTING_PER_SLOT:
        if n_cmp > 0:
            return False, (f"a per-slot-routed case produced {n_cmp} counted shadow comparison(s) "
                           f"(group rows {rows}) — the gate's routing class disagrees with the "
                           "worker's grouping; refusing")
        return True, (f"per-slot routing (every op phase=prefill): applied on the per-request "
                      f"path by design, no vectorized group to compare — "
                      f"{snapshot.get('n_unmodified_groups')} decode group(s) unmodified on both "
                      "sides; the fallback path is the routing_* check")
    if routing != ROUTING_VECTORIZED:
        return False, f"unknown routing class {routing!r}"
    if not isinstance(waves, list):
        return None, ("no per-wave shadow attribution recorded (shadow_waves); cannot attribute "
                      f"the {n_cmp} comparison(s) to submitted waves")
    per_B = compared_per_wave_size(waves, batches)
    missing = [B for B, n in per_B.items() if n == 0]
    if missing:
        return None, (f"no counted shadow comparison in any submitted wave at B={missing} "
                      f"(counted per wave size: {per_B}; group rows compared overall: {rows})")
    return True, (f"{n_cmp} groups compared, {tol['detail']}; counted comparisons per submitted wave "
                  f"size: {per_B}; group rows compared (kernel-shape evidence): {rows}")


def routing_evidence_ok(ev: Any, routing: Any) -> Tuple[bool, str]:
    """The ``routing_*`` hard check: the SelectionRecord's PER-PATH counters
    (``selection_evidence``) agree with the case's routing class. A record
    without per-path counters (the base commit's) or with unattributed
    applications fails — never read as either path."""
    if not isinstance(ev, Mapping) or ev.get("paths_recorded") is not True:
        return False, ("the worker's SelectionRecord carries no per-path counters "
                       "(n_applications_per_slot / n_applications_grouped)")
    un = int(ev.get("n_applications_unattributed") or 0)
    if un:
        return False, f"{un} application(s) named no apply path"
    ps, gr = int(ev.get("n_modifying_per_slot") or 0), int(ev.get("n_modifying_grouped") or 0)
    if routing == ROUTING_PER_SLOT:
        return (ps > 0 and gr == 0), (f"per-slot routing: {ps} modifying per-slot application(s) "
                                      f"(must be > 0), {gr} grouped (must be 0)")
    if routing == ROUTING_VECTORIZED:
        return (gr > 0), (f"vectorized routing: {gr} modifying grouped application(s) (must be "
                          f"> 0); {ps} per-slot (prefill slots)")
    return False, f"unknown routing class {routing!r}"


def subset_slug(cases: Sequence[str]) -> str:
    """The results sub-directory of a ``--cases`` SUBSET run: the case ids
    with ``:`` -> ``_`` joined by ``+`` (the sbatch mirrors it byte for byte
    with ``tr ':,' '_+'`` over GATE_CASES). A subset run writes under
    ``<model>/subset/<slug>/`` so it can never overwrite the family's FULL
    report; it is complete=False by construction (every case not run stays
    uncovered) and exits 3 whatever passed — never a certification."""
    return "+".join(str(c).replace(":", "_") for c in cases)


def gate_batches(spec: Any) -> Tuple[int, ...]:
    """The batch ladder from ``--batches`` / ``GATE_BATCHES``: B=1 ALWAYS
    FIRST — the hard cross-server verdict (``pure_B1``) runs outside the
    configurable ladder and cannot be configured away — then the requested
    B>1 evidence sizes, deduplicated, ascending. Refuses (GateError) any entry
    that is not a positive integer; an empty list means B=1 only."""
    return (1,) + tuple(b for b in parse_int_list(spec, "--batches") if b != 1)


def parse_int_list(spec: Any, what: str) -> Tuple[int, ...]:
    """A comma list (or sequence) of POSITIVE integers, deduplicated and
    sorted; anything else refuses loudly naming the entry."""
    parts = ([x.strip() for x in spec.split(",") if x.strip()] if isinstance(spec, str)
             else list(spec or ()))
    out = set()
    for x in parts:
        if isinstance(x, bool) or not (isinstance(x, int) or (isinstance(x, str) and x.isdigit())):
            raise GateError(f"{what}: entry {x!r} is not a positive integer")
        b = int(x)
        if b < 1:
            raise GateError(f"{what}: entry {x!r} is not a positive integer")
        out.add(b)
    return tuple(sorted(out))


def shadow_multigroup_check(snapshot: Any, *, reduction_mode: str = DEFAULT_REDUCTION_MODE
                            ) -> Tuple[Optional[bool], str]:
    """The multigroup shadow requirement: the record inside THE REDUCTION
    CONTRACT under ``reduction_mode`` (``reduction_tolerance``; exact for
    elementwise ops and under the per-row mode, the bounded tolerance for
    reduction ops under the stacked mode), no shadow / record error, and at
    least one layer call that compared groups of >= 2 distinct cells."""
    if not isinstance(snapshot, Mapping) or snapshot.get("rpc") == "absent":
        return False, "no sisteer_shadow_compare record from the shadow standup"
    if not snapshot.get("enabled"):
        return False, "the shadow standup did not enable the shadow compare"
    tol = reduction_tolerance(snapshot, reduction_mode=reduction_mode)
    if tol["ok"] is not True:
        return False, tol["detail"]
    if int(snapshot.get("n_shadow_errors") or 0) > 0 or int(snapshot.get("n_record_errors") or 0) > 0:
        return False, f"shadow/record errors: {snapshot.get('first_shadow_error')} {snapshot.get('first_record_error')}"
    if int(snapshot.get("n_layer_calls_with_ge2_cells_compared") or 0) < 1:
        return None, (f"no layer call compared >= 2 distinct cells (max "
                      f"{snapshot.get('max_distinct_cells_compared_in_one_layer_call')}; cells "
                      f"{snapshot.get('cells_compared')})")
    return True, (f"{snapshot.get('n_groups_compared')} groups compared, {tol['detail']}; "
                  f"{snapshot.get('n_layer_calls_with_ge2_cells_compared')} layer call(s) with >= 2 "
                  f"cells ({snapshot.get('cells_compared')})")


def bi_attestation_ok(attested: Any, intent: int, *, allow_absent: bool) -> Tuple[bool, str]:
    """The worker-side resolved batch-invariance state (lens_worker.
    batch_invariant_report, carried per worker in the attestation file)
    against the launcher's intent. Every worker must attest ``env`` AND
    ``active`` as real booleans EQUAL to the intent (1 -> True, 0 -> False)
    with NO ``batch_invariant_error``: an unreadable state (the pinned vLLM's
    surface changed, the module could not be imported), a None / missing
    field, or a helper-side read error is a refusal under BOTH intents —
    never "not active". The ONE relaxation: a record set carrying none of the
    fields at all (the base commit's worker, whose report predates them) is
    accepted where ``allow_absent`` (the reference at intent 0) — a mixed set
    (some workers attest, some not) is not."""
    workers = list(attested or [])
    if not workers or not all(isinstance(w, Mapping) for w in workers):
        return False, f"no worker attestation records (got {str(attested)[:120]!r})"
    for w in workers:
        if "error" in w and not any(k in w for k in BI_FIELDS):
            return False, f"the attestation could not be read: {w['error']}"
    with_fields = [w for w in workers if any(k in w for k in BI_FIELDS)]
    if not with_fields:
        ok = bool(allow_absent) and int(intent) == 0   # an absent attestation never satisfies BI=1
        return (ok, "no worker attests its batch-invariance state"
                + (" (accepted: base-commit reference at BI=0)" if ok else ""))
    if len(with_fields) != len(workers):
        return False, (f"{len(with_fields)} of {len(workers)} workers attest their batch-invariance "
                       "state; the rest carry no fields")
    want = int(intent) == 1
    surfaces = []
    for w in with_fields:
        err = w.get("batch_invariant_error")
        if err:
            return False, f"a worker's batch-invariance state could not be read: {err}"
        env, active = w.get("batch_invariant_env"), w.get("batch_invariant_active")
        if not isinstance(env, bool) or not isinstance(active, bool):
            return False, (f"a worker attests non-boolean batch-invariance state env={env!r} "
                           f"active={active!r} (missing or None is not a state)")
        if env is not want or active is not want:
            return False, f"intent BI={int(intent)} but a worker attests env={env!r} active={active!r}"
        surfaces.append(str(w.get("batch_invariant_surface")))
    return True, (f"{len(with_fields)} worker(s) attest BI={int(intent)} (env and active both "
                  f"{want}) via {sorted(set(surfaces))}")


def mixed_roles(B: int) -> List[str]:
    if B == 1:
        return ["steered"]
    return ["passthrough" if i % 4 == 0 else "scale0" if i % 4 == 1 else "steered"
            for i in range(B)]


def prompt_targets(B: int, targets: Sequence[int] = PROMPT_TOKEN_TARGETS) -> List[int]:
    return [int(targets[i % len(targets)]) for i in range(B)]


# --------------------------------------------------------------------------- #
# The hard checks (pure)                                                       #
# --------------------------------------------------------------------------- #


def apply_path_ok_reference(path: Any, *, reference_sha: str, base_sha: str) -> bool:
    """The reference's attested path: ``"per_row"`` when the tree has the RPC;
    otherwise the RPC must be ABSENT (an error record) AND the tree must be
    the base commit, which predates ``sisteer_apply_path``."""
    if path == "per_row":
        return True
    if isinstance(path, Mapping) and path.get("rpc") == "absent" and reference_sha == base_sha:
        return True
    return False


def case_checks(case: Mapping[str, Any], *, reference_sha: str, base_sha: str,
                have_oracle: bool, reduction_mode: str = DEFAULT_REDUCTION_MODE
                ) -> Dict[str, Dict[str, Any]]:
    """The module docstring's hard checks for one single-op case, from its
    recorded facts: the three attested apply paths (the candidate's and the
    shadow standup's name ``reduction_mode``), the worker's SelectionRecord
    evidence (reference + candidate) and its attested reduction mode
    (candidate + shadow), the B=1 singleton controls (passthrough and scale-0
    equal the oracle; the steered request differs), full response evidence,
    the batch-invariance attestations, and the in-worker shadow compare held
    to THE REDUCTION CONTRACT under ``reduction_mode`` (True / None / False
    per :func:`shadow_check`). Every entry is ``{"ok": bool | None, "detail":
    str}``; a missing fact is a failed check."""
    mode = check_reduction_mode(reduction_mode)
    out: Dict[str, Dict[str, Any]] = {}
    ap = case.get("apply_path") or {}
    out["reference_apply_path"] = {
        "ok": apply_path_ok_reference(ap.get(REFERENCE), reference_sha=reference_sha,
                                      base_sha=base_sha),
        "detail": f"reference attested {ap.get(REFERENCE)!r} (tree {reference_sha[:12]})"}
    want_c, want_s = expected_apply_path(mode), expected_apply_path(mode, shadow=True)
    out["candidate_apply_path"] = {
        "ok": ap.get(CANDIDATE) == want_c,
        "detail": f"candidate attested {ap.get(CANDIDATE)!r}; the {mode} mode expects {want_c!r}"}
    out["shadow_apply_path"] = {
        "ok": ap.get(SHADOW) == want_s,
        "detail": f"shadow standup attested {ap.get(SHADOW)!r}; the {mode} mode expects {want_s!r}"}
    if LEGACY in ap:
        out["legacy_ab_apply_path"] = {
            "ok": ap.get(LEGACY) == APPLY_PATH_PER_ROW,
            "detail": f"legacy A/B standup attested {ap.get(LEGACY)!r}"}
    ev = case.get("steering_evidence") or {}
    for side in (REFERENCE, CANDIDATE):
        e = ev.get(side) or {}
        n_mod = int(e.get("n_modifying") or 0)
        matched = bool(e.get("all_observations_matched", False))
        out[f"evidence_{side}"] = {
            "ok": matched and n_mod > 0,
            "detail": f"{side}: {n_mod} modifying applications over "
                      f"{e.get('n_observations')} observations, matched={matched}"}
    for side in (REFERENCE, CANDIDATE):
        c1 = (case.get("controls_B1") or {}).get(side) or {}
        out[f"controls_B1_{side}"] = {
            "ok": have_oracle and c1.get("controls_match") is True,
            "detail": c1.get("controls_detail") or (
                f"{side}: B=1 passthrough and scale-0 requests equal the oracle" if c1.get("controls_match")
                else f"{side}: no B=1 control comparison recorded")}
        out[f"steering_bites_B1_{side}"] = {
            "ok": have_oracle and c1.get("steered_differs") is True,
            "detail": (f"{side}: the B=1 steered request differs from the oracle" if c1.get("steered_differs")
                       else f"{side}: the B=1 steered request is bit-identical to passthrough — nothing "
                            "was applied" if c1.get("steered_differs") is False
                       else f"{side}: no B=1 steered comparison recorded")}
    bi = case.get("bi_attested") or {}
    for side in (REFERENCE, CANDIDATE, SHADOW):
        b = bi.get(side)
        out[f"batch_invariant_{side}"] = {
            "ok": bool(b.get("ok")) if isinstance(b, Mapping) else False,
            "detail": (b.get("detail") if isinstance(b, Mapping) else f"{side}: no attestation recorded")}
    sig_err = case.get("signature_errors") or []
    out["signatures"] = {"ok": not sig_err,
                         "detail": ("every response carried full evidence" if not sig_err
                                    else f"{len(sig_err)} response(s) lacked evidence: {sig_err[0]}")}
    # the ROUTING CLASS (module docstring) from the case id, the worker's rule
    routing: Any = None
    try:
        routing = case_routing(str(case.get("case_id")))
    except GateError as e:
        out["routing_class"] = {"ok": False, "detail": str(e)}
    for side in (CANDIDATE, SHADOW):
        rok, rdetail = routing_evidence_ok(ev.get(side), routing)
        out[f"routing_{side}"] = {"ok": rok, "detail": f"{side}: {rdetail}"}
    for side in (CANDIDATE, SHADOW):
        mok, mdetail = reduction_mode_evidence_ok(ev.get(side), mode)
        out[f"reduction_mode_{side}"] = {"ok": mok, "detail": f"{side}: {mdetail}"}
    b1 = (case.get("verdicts") or {}).get("pure_B1")
    filed = isinstance(b1, Mapping) and not b1.get("evidence") and b1.get("verdict") != EVIDENCE \
        and b1.get("verdict") in (PASS, FAIL, UNSTABLE_REF, UNSTABLE_VEC)
    out["pure_B1_present"] = {
        "ok": bool(filed),
        "detail": (f"the B=1 reference-vs-candidate verdict was filed: {b1.get('verdict')}" if filed
                   else f"no hard B=1 reference-vs-candidate verdict was filed (got {b1!r}); B=1 is "
                        "not configurable away")}
    ok, detail = shadow_check(case.get("shadow"), case.get("batches") or (), routing=routing,
                              waves=case.get("shadow_waves"), reduction_mode=mode)
    out["shadow_compare"] = {"ok": ok, "detail": detail}
    r_ok, r_detail = resumed_prefill_check(case.get("resumed_prefill"))
    out["resumed_prefill"] = {"ok": r_ok, "detail": r_detail}
    return out


# --------------------------------------------------------------------------- #
# THE SHARED-PREFILL CASE (pure parts)                                         #
# --------------------------------------------------------------------------- #


def resumed_policy_from_case_id(case_id: str) -> str:
    """The resumed-prefill POLICY read off a case id's op / positions / phase
    (:data:`RESUMED_POLICIES`): ``always`` for a per-position op at ``all``
    in both phases, ``conditional`` for one at ``response`` in both phases,
    ``never`` otherwise — the cross-check against ``lens_adapters.
    resumed_prefill_policy`` over the served object's adapted ops
    (``case_resumed_policy``); a disagreement is a GateError, never a
    default."""
    from sisteer.capeval.serving import lens_adapters as la
    c = parse_case_id(case_id)
    per_pos = bool(la.op_name_per_position(c["op"])) and c["phase"] == "both"
    if per_pos and c["positions"] == "all":
        return RESUMED_ALWAYS
    if per_pos and c["positions"] == "response":
        return RESUMED_CONDITIONAL
    return RESUMED_NEVER


def resumed_rule_from_case_id(case_id: str) -> bool:
    """``resumed_policy_from_case_id(case_id) == "always"`` — the bool form."""
    return resumed_policy_from_case_id(case_id) == RESUMED_ALWAYS


def case_resumed_policy(obj: Mapping[str, Any], *, header_len: int) -> str:
    """The case's SERVED object's resumed-prefill policy: ``lens_adapters.
    resumed_prefill_policy`` over its REAL adapted ops (the worker's own
    predicate, on the object the standup serves), cross-checked against the
    case id's rule."""
    from sisteer.capeval.serving import lens_adapters as la
    _kind, ops = adapt_object_dir(Path(obj["source"]).parent, arm=obj["arm"], header_len=header_len)
    policy = la.resumed_prefill_policy(ops)
    want = resumed_policy_from_case_id(str(obj["case_id"]))
    if policy != want:
        raise GateError(f"case {obj['case_id']}: resumed_prefill_policy over the served object's ops is "
                        f"{policy!r} but the case id's op/positions/phase say {want!r} — the predicate and "
                        "the case schema disagree; refusing to drive the shared-prefill case on a misfiled object")
    return policy


def case_resumable(obj: Mapping[str, Any], *, header_len: int) -> bool:
    """``case_resumed_policy(obj) == "always"`` — the bool form."""
    return case_resumed_policy(obj, header_len=header_len) == RESUMED_ALWAYS


def proxy_salt_decision(endpoint: Mapping[str, Any], *, harness_cell: str,
                        intervention_hash: str, virtual_model: str) -> Dict[str, Any]:
    """WHAT THE PRODUCTION PROXY WOULD MINT for this cell — computed by calling
    ``proxy.RoutingProxy``'s own minting, against the per-cell policy THIS
    standup's workers attested (the standup published it in the endpoint after
    reading the steering attestation with ``lens_worker.
    resumed_prefill_policy_cells_attested``). Not a mirror: the salt and the
    regime come out of the shipped proxy code, so the shared-prefill case can
    drive the exact namespace production would and the gate can say whether a
    cell is routed onto a shared prefix at all.

    Records the attested policy (or the reason there is none), the regime, the
    salt, and — as the demonstration that a request-unique salt really is a
    namespace of one — two successive mints of it."""
    from sisteer.capeval.serving import proxy as px
    cells = endpoint.get("resumed_prefill_policy_cells")
    attested = cells.get(harness_cell) if isinstance(cells, Mapping) else None
    entry = px.RouteEntry(virtual_model=virtual_model, cell_id=harness_cell,
                          cell_class="gate", intervention_hash=intervention_hash,
                          serving_identity="proxy_vllm+lens+interleaved")
    regime = px.RoutingProxy.salt_regime_for(lens=True, resumed_prefill_policy=attested)
    mints = [px.RoutingProxy.cache_salt_for(entry, lens=True, resumed_prefill_policy=attested)
             for _ in range(2)]
    return {"attested_policy": attested,
            "attest_error": endpoint.get("resumed_prefill_policy_error"),
            "policy_cells_present": isinstance(cells, Mapping),
            "cache_salt_regime": regime,
            "salt": mints[0], "second_mint": mints[1],
            "salt_is_stable_across_mints": mints[0] == mints[1],
            # the unhooked namespace of the SAME (cell, intervention): a lens
            # salt must never equal it, or a steered and an unsteered leg would
            # share one KV namespace
            "unhooked_salt": px.RoutingProxy.cache_salt_for(entry, lens=False),
            "regime_vocabulary": list(px.SALT_REGIMES),
            "cell_stable": px.SALT_REGIME_CELL_STABLE,
            "request_unique": px.SALT_REGIME_REQUEST_UNIQUE}


def proxy_regime_probe(su: "Standup", vmodel: str, *, timeout_s: float = 300.0
                       ) -> Dict[str, Any]:
    """ONE short chat THROUGH the standup's routing proxy on the case's steered
    vmodel, read for the decision the LIVE proxy made for it: the response's
    ``petri`` context block echoes the request's ``cache_salt_regime`` and the
    ``resumed_prefill_policy`` the proxy read. This is the only leg of the
    shared-prefill case that crosses the proxy (the rest drives vLLM directly,
    module docstring), and it is sent FIRST — before the forced-salt legs,
    whose refusal on a ``never`` cell is engine-fatal by design.

    Deliberately a SHORT, distinct prompt: it must not share a prefix with the
    case's A/B pair, or it would seed the very cache the case is measuring."""
    import urllib.error
    import urllib.request
    body = json.dumps({"model": vmodel, "messages": PROXY_PROBE_MESSAGES,
                       "max_tokens": 1, "temperature": 0.0, "stream": False}).encode()
    headers = {"Content-Type": "application/json"}
    if su.api_key:
        headers["Authorization"] = f"Bearer {su.api_key}"
    req = urllib.request.Request(f"{su.proxy_url.rstrip('/')}/chat/completions",
                                 data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            status, parsed = int(r.status), json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001 — recorded, judged by the check
        return {"status": None, "error": f"{type(e).__name__}: {e}"}
    petri = parsed.get("petri") if isinstance(parsed, Mapping) else None
    petri = petri if isinstance(petri, Mapping) else {}
    return {"status": status,
            "cache_salt_regime": petri.get("cache_salt_regime"),
            "resumed_prefill_policy": petri.get("resumed_prefill_policy"),
            "cell_id": petri.get("cell_id")}


def resumed_case_applicable(ms: ModelServe) -> Tuple[bool, str]:
    """Whether the shared-prefill case can run on ``ms``: prefix caching must be
    able to resolve ON — structurally impossible on a hybrid-recurrent target
    (``vllm_server.hybrid_recurrent_model``: Qwen3.5's GDN stack, NemotronH's
    Mamba2 stack), where the case is NOT APPLICABLE and recorded so. An
    unreadable model config does not skip: the standup is attempted and
    :func:`prefix_caching_on_ok` judges the resolved state."""
    try:
        from sisteer.capeval.serving import vllm_server as vs
        hybrid = bool(vs.hybrid_recurrent_model(ms.hf, ms.rev))
    except Exception as e:  # noqa: BLE001 — recorded; the standup's attestation decides
        return True, f"could not resolve the target's recurrence ({type(e).__name__}: {e}); standing up"
    if hybrid:
        return False, (f"{ms.hf} is a hybrid-recurrent target: prefix caching is structurally OFF on this "
                       "stack (vllm_server.hybrid_recurrent_model), so no request can resume a prefill "
                       "and the worker's resumed-prefill acceptance is never exercised here")
    return True, "dense / sliding-attention target: prefix caching resolves ON"


def resumed_prompt_pair(prompts: Sequence[Tuple[Any, int]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Two single-turn prompts sharing the LONGEST gate passage as a prefix
    and differing in a short question tail — the shared blocks the warm
    request resumes are the passage's; the tails keep the two prompts
    distinct so the warm request is a genuine partial resume, not the
    identical-prompt repeat (which the drive sends separately)."""
    base = str(prompts[-1][0][0]["content"])
    a = [{"role": "user", "content": base + " First question: what did the miller count?"}]
    b = [{"role": "user", "content": base + " Second question: where did the afternoon light move?"}]
    return a, b


#: the assistant text of the multi-turn pair's first turn
MULTITURN_ANSWER = "The miller counted the sacks of grain one by one."


def resumed_multiturn_pair(prompts: Sequence[Tuple[Any, int]]
                           ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Two TWO-TURN prompts — user (the longest passage + a question),
    assistant (a fixed answer), user (the second-longest passage + a
    distinct question) — sharing everything through the second passage, so
    the shared prefix carries the FIRST turn's assistant header deep inside
    it (several KV blocks before the divergence). A prior writer's response
    boundary sits in the cached region: the multi-turn shape — served on an
    ``always`` cell (the turn-N-from-turn-N-1 payoff), REFUSED and contained
    on a ``conditional`` one (a header inside the cached region), whatever
    the block alignment."""
    long_ = str(prompts[-1][0][0]["content"])
    second = str(prompts[-2][0][0]["content"]) if len(prompts) >= 2 else long_
    first = [{"role": "user", "content": long_ + " First question: what did the miller count?"},
             {"role": "assistant", "content": MULTITURN_ANSWER}]
    a = first + [{"role": "user", "content": second + " Second question: where did the afternoon light move?"}]
    b = first + [{"role": "user", "content": second + " Third question: what did the mill stand beside?"}]
    return a, b


def signature_closeness(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """How close two response signatures are — EVIDENCE for the warm-vs-cold
    pairs (module docstring, THE SHARED-PREFILL CASE): exact equality, the
    first position whose sampled token differs (None when none does), the
    largest |delta logprob| of the sampled tokens before that divergence, and
    at POSITION 0 (the prefill's own readout) the largest |delta logprob|
    over the top tokens both sides list."""
    ids_a, ids_b = list(a["token_ids"]), list(b["token_ids"])
    n = min(len(ids_a), len(ids_b))
    div = next((i for i in range(n) if ids_a[i] != ids_b[i]), None)
    if div is None and len(ids_a) != len(ids_b):
        div = n
    upto = n if div is None else div
    lp = [abs(float(x) - float(y)) for x, y in zip(a["logprobs"][:upto], b["logprobs"][:upto])]
    top_a = dict(a["top"][0]) if a.get("top") else {}
    top_b = dict(b["top"][0]) if b.get("top") else {}
    common = sorted(set(top_a) & set(top_b))
    return {"equal": (ids_a == ids_b and list(a["logprobs"]) == list(b["logprobs"])
                      and list(a["top"]) == list(b["top"])),
            "first_token_divergence": div, "n_tokens": [len(ids_a), len(ids_b)],
            "max_abs_dlogprob_before_divergence": (max(lp) if lp else None),
            "pos0_max_abs_dlogprob_top": (max(abs(top_a[t] - top_b[t]) for t in common) if common else None),
            "pos0_top_tokens_compared": len(common)}


def one_request_attestation(requests: Mapping[str, Any]) -> Dict[str, Any]:
    """The SelectionRecord's ``requests`` map read after ONE request on a
    record reset just before it: exactly one entry, whose resumed fields
    (:data:`RESUMED_ATTEST_FIELDS`) are the attestation. Zero or several
    entries are recorded as such — never the first one taken."""
    if not isinstance(requests, Mapping) or len(requests) != 1:
        return {"n_entries": (len(requests) if isinstance(requests, Mapping) else None),
                "error": "expected exactly one request entry in the reset SelectionRecord"}
    (rid, entry), = requests.items()
    out: Dict[str, Any] = {"n_entries": 1, "req_id": str(rid)}
    for k in RESUMED_ATTEST_FIELDS:
        out[k] = entry.get(k) if isinstance(entry, Mapping) else None
    return out


def _attest_resumed(att: Any, name: str, problems: List[str]) -> None:
    if not isinstance(att, Mapping) or att.get("n_entries") != 1:
        problems.append(f"{name}: no single SelectionRecord entry to attest from ({att!r})")
        return
    if att.get("prefill_resumed") is not True:
        problems.append(f"{name}: the worker did not attest a RESUMED prefill "
                        f"(prefill_resumed={att.get('prefill_resumed')!r})")
    c, k = att.get("prefill_cached_positions"), att.get("prefill_computed_positions")
    if not (isinstance(c, int) and c >= 1):
        problems.append(f"{name}: prefill_cached_positions={c!r} (expected >= 1)")
    if not (isinstance(k, int) and k >= 1):
        problems.append(f"{name}: prefill_computed_positions={k!r} (expected >= 1)")
    m = att.get("n_applications_modifying_resumed")
    if not (isinstance(m, int) and m >= 1):
        problems.append(f"{name}: n_applications_modifying_resumed={m!r} — no op modified the resumed rows")


def proxy_decision_problems(pdec: Any, *, policy: Any, harness_cell: Any,
                            drive_salt: Any) -> List[str]:
    """THE PROXY HALF's requirements inside the ``resumed_prefill`` hard check.
    The lens half decides what a resumed prefill DOES; the proxy half decides
    whether a cell is offered one at all, and these are the facts that make the
    second checkable on a live standup:

    - the standup ATTESTED a per-cell policy and the proxy read the SAME policy
      the case is driving (a case whose served ops say ``always`` while the
      workers attested something else is a wiring split, not a pass);
    - the regime is CELL-STABLE for ``always`` and REQUEST-UNIQUE for
      ``conditional`` / ``never`` — i.e. a broadcast-kind cell is not routed
      onto a shared prefix BY THE PROXY, so the worker's engine-fatal refusal
      is unreachable in production rather than merely survivable;
    - a request-unique salt really is a namespace of one (two mints differ) and
      a cell-stable one really is stable (two mints agree), and neither ever
      equals the UNHOOKED namespace of the same (cell, intervention);
    - for ``always`` the case DRIVES the proxy's own salt, so the resume proved
      here is production's;
    - the LIVE proxy agreed: one request through it came back 200 with the same
      regime and policy echoed on its ``petri`` block (its audit row carries
      the same two fields).

    Missing facts are failures, never passes by silence — a record written
    before this landing fails too, with a message that says which fact is
    absent rather than reading its silence as a pass."""
    if pdec is None:
        return ["the record carries no proxy_decision — a shared-prefill drive from before "
                "the proxy half; the salt regime it drove under is unknown"]
    if not isinstance(pdec, Mapping):
        return [f"unreadable proxy_decision ({type(pdec).__name__})"]
    out: List[str] = []
    stable, unique = pdec.get("cell_stable"), pdec.get("request_unique")
    attested, regime = pdec.get("attested_policy"), pdec.get("cache_salt_regime")
    if not pdec.get("policy_cells_present"):
        out.append("the standup published no attested resumed_prefill_policy_cells "
                   f"({pdec.get('attest_error')!r}) — the proxy was handed nothing and can "
                   "share no steered prefill at all")
    elif attested != policy:
        out.append(f"the workers attested resumed_prefill_policy {attested!r} for cell "
                   f"{harness_cell!r} but the served object's ops say {policy!r} — the proxy and "
                   "the worker would disagree about this cell")
    want = stable if policy == RESUMED_ALWAYS else unique
    if regime != want:
        out.append(f"the proxy mints a {regime!r} cache_salt for a {policy!r} cell (expected "
                   f"{want!r}) — an always cell must share one namespace, and a conditional or "
                   "never cell must never be routed onto a shared prefix")
    if regime == stable and not pdec.get("salt_is_stable_across_mints"):
        out.append("the cell-stable salt is not stable across mints — nothing would resume")
    if regime == unique and pdec.get("salt_is_stable_across_mints"):
        out.append("the request-unique salt repeated across mints — it is not a namespace of one")
    if pdec.get("salt") == pdec.get("unhooked_salt"):
        out.append("the lens salt equals the UNHOOKED salt of the same (cell, intervention) — a "
                   "steered and an unsteered leg would share one KV namespace")
    if policy == RESUMED_ALWAYS:
        if pdec.get("forced_stable_salt"):
            out.append("the always case drove a FORCED gate salt, not the proxy's — the resume it "
                       "proves is not the one production would get")
        if drive_salt != pdec.get("salt"):
            out.append(f"the always case drove salt {drive_salt!r}, the proxy mints "
                       f"{pdec.get('salt')!r}")
    elif not pdec.get("forced_stable_salt"):
        out.append(f"a {policy!r} case drove the proxy's own salt — the proxy declines to mint a "
                   "shared namespace for it, so the case must force one to reach the worker")
    probe = pdec.get("live_probe")
    if not isinstance(probe, Mapping) or probe.get("status") != 200:
        out.append(f"the live proxy probe did not serve ({probe!r}) — the decision this proxy "
                   "actually makes per request was never read")
    else:
        if probe.get("cache_salt_regime") != regime:
            out.append(f"the live proxy served a {probe.get('cache_salt_regime')!r} request while "
                       f"the same code mints {regime!r} out of process")
        if pdec.get("policy_cells_present") and probe.get("resumed_prefill_policy") != attested:
            out.append(f"the live proxy read policy {probe.get('resumed_prefill_policy')!r}, the "
                       f"standup attested {attested!r}")
    return out


def resumed_prefill_check(rec: Any) -> Tuple[Optional[bool], str]:
    """THE ``resumed_prefill`` HARD CHECK (module docstring, THE SHARED-PREFILL
    CASE), from the drive's record: ``None`` when the case did not run,
    ``True`` with "not applicable" when it was skipped on a hybrid family,
    else the resumable / non-resumable protocol's requirements — every
    missing fact is a failure, never a pass by silence."""
    if rec is None:
        return None, "the shared-prefill case did not run (--no-resumed-prefill): inconclusive"
    if not isinstance(rec, Mapping):
        return False, f"unreadable shared-prefill record ({type(rec).__name__})"
    if rec.get("skipped"):
        return True, f"not applicable: {rec['skipped']}"
    if rec.get("error"):
        return False, f"the shared-prefill drive errored: {rec['error']}"
    reqs = rec.get("requests") or {}
    att = rec.get("attest") or {}
    problems: List[str] = []

    def served(name: str) -> bool:
        """SERVED under the PRODUCTION CONTRACT: HTTP 200 with a full signature
        AND — for a steered request — the worker's per-request failure ledger
        CLAIMED for it (``claimed_failure`` present on the entry) and EMPTY. A
        worker-contained failure comes back 200 from vLLM, and the proxy
        (``_refuse_poisoned_lens_response``) turns the claimed failure into a
        500 ``lens_request_failed``; an unclaimed steered entry is missing
        evidence, never a pass."""
        e = reqs.get(name)
        if not (isinstance(e, Mapping) and e.get("status") == 200 and isinstance(e.get("signature"), Mapping)):
            return False
        if name.startswith("steered_"):
            return "claimed_failure" in e and e["claimed_failure"] is None
        return True

    def why_not_served(name: str) -> str:
        e = reqs.get(name) or {}
        if e.get("status") != 200:
            return f"status {e.get('status')!r}"
        if not isinstance(e.get("signature"), Mapping):
            return f"no full signature ({e.get('signature_error') or 'missing'})"
        if name.startswith("steered_") and "claimed_failure" not in e:
            return "the worker's failure ledger was never claimed for it (missing evidence)"
        if name.startswith("steered_") and e.get("claimed_failure") is not None:
            return (f"the worker CONTAINED a failure on it ({_claim_type(e['claimed_failure'])!r}: "
                    f"{str((e['claimed_failure'] or {}).get('error', ''))[:160]}) — production would have 500'd "
                    "it (lens_request_failed), so it is not a served request")
        return "served"

    for name in ("passthrough_cold_b", "passthrough_writer_a", "passthrough_warm_b",
                 "steered_cold_b", "steered_writer_a"):
        if not served(name):
            problems.append(f"{name}: not served — {why_not_served(name)}")
    cold = att.get("steered_cold_b")
    if not isinstance(cold, Mapping) or cold.get("n_entries") != 1:
        problems.append(f"steered_cold_b: no single SelectionRecord entry ({cold!r})")
    elif cold.get("prefill_resumed") is not False:
        problems.append(f"steered_cold_b: attested prefill_resumed={cold.get('prefill_resumed')!r}, "
                        "expected False (a unique salt never resumes)")
    policy = rec.get("policy")
    if policy is None and "resumable" in rec:        # a pre-policy record
        policy = RESUMED_ALWAYS if rec.get("resumable") else RESUMED_NEVER
    problems += proxy_decision_problems(rec.get("proxy_decision"), policy=policy,
                                        harness_cell=rec.get("harness_cell"),
                                        drive_salt=(rec.get("salts") or {}).get("cell"))
    if policy in (RESUMED_ALWAYS, RESUMED_CONDITIONAL):
        # the SAFE shape (two single-turn prompts diverging before the header): served + attested
        if not served("steered_warm_b"):
            problems.append(f"steered_warm_b: the {policy} cell's warm request (the safe shape) was not served "
                            f"— {why_not_served('steered_warm_b')}")
        _attest_resumed(att.get("steered_warm_b"), "steered_warm_b", problems)
        if served("steered_warm_b") and served("passthrough_warm_b") and not signatures_differ_somewhere(
                [reqs["steered_warm_b"]["signature"]], [reqs["passthrough_warm_b"]["signature"]]):
            problems.append("steered_warm_b is bit-identical to passthrough_warm_b — nothing was applied "
                            "on the resumed prefill")
        # the multi-turn WRITER seeds the cache the multi-turn reader resumes: it must itself have
        # been served under the production contract, or the reuse it enables is of a prefix
        # production would never have accepted
        if not served("steered_multiturn_writer"):
            problems.append(f"steered_multiturn_writer: not served — {why_not_served('steered_multiturn_writer')}")
        if policy == RESUMED_ALWAYS:
            if not served("steered_warm_b_repeat"):
                problems.append("steered_warm_b_repeat: the identical-prompt repeat (the avg@k shape) was not "
                                f"served — {why_not_served('steered_warm_b_repeat')}")
            _attest_resumed(att.get("steered_warm_b_repeat"), "steered_warm_b_repeat", problems)
            # the multi-turn shape (a prior turn's header inside the cached region): served on always
            if not served("steered_multiturn_warm"):
                problems.append("steered_multiturn_warm: the always cell's multi-turn resume (a prior header in "
                                f"the cached region) was not served — {why_not_served('steered_multiturn_warm')}")
            _attest_resumed(att.get("steered_multiturn_warm"), "steered_multiturn_warm", problems)
        else:
            # the UNSAFE shape on a conditional cell — THE ONE request required to carry a claimed
            # failure: served by the engine (200), FAILED per request with the contained
            # ResumedPrefillRefusal claimed exactly as the proxy claims it, and the engine still
            # serving afterwards
            mt = reqs.get("steered_multiturn_warm") or {}
            if mt.get("status") != 200:
                problems.append("steered_multiturn_warm: the conditional cell's unsafe resume must be CONTAINED "
                                f"(engine reply 200 with a claimed refusal), got status {mt.get('status')!r} — "
                                "an engine-fatal refusal is the wrong class here")
            claim = mt.get("claimed_failure") if "claimed_failure" in mt else rec.get("multiturn_claimed_failure")
            if not isinstance(claim, Mapping) or claim.get("error_type") != RESUMED_CONTAINED_ERROR_TYPE:
                problems.append("steered_multiturn_warm: no contained refusal was claimed for the unsafe resume "
                                f"(sisteer_take_request_failure returned {claim!r}; expected error_type "
                                f"{RESUMED_CONTAINED_ERROR_TYPE!r}) — the unsafe hit was served STEERED, or "
                                "the classifier did not decide per request")
            if not served("steered_after_refusal"):
                problems.append("steered_after_refusal: the engine did not keep serving after the contained "
                                f"refusal, or the containment leaked onto the fresh request — "
                                f"{why_not_served('steered_after_refusal')}")
            # the identical-prompt repeat is EVIDENCE on a conditional cell: whether it resumes safely
            # depends on the block alignment of the header (recorded, never ruled) — but a failure of
            # any OTHER class on it, or a dead engine, is a defect
            rp = reqs.get("steered_warm_b_repeat") or {}
            rp_claim = rp.get("claimed_failure") if "claimed_failure" in rp else rec.get("repeat_claimed_failure")
            if rp.get("status") != 200 or not isinstance(rp.get("signature"), Mapping):
                problems.append(f"steered_warm_b_repeat: the identical-prompt repeat was not answered (status "
                                f"{rp.get('status')!r})")
            elif "claimed_failure" not in rp and "repeat_claimed_failure" not in rec:
                problems.append("steered_warm_b_repeat: the worker's failure ledger was never claimed for it "
                                "(missing evidence)")
            elif rp_claim is not None and _claim_type(rp_claim) != RESUMED_CONTAINED_ERROR_TYPE:
                problems.append(f"steered_warm_b_repeat: a failure of another class was claimed on it "
                                f"({_claim_type(rp_claim)!r}) — not the block-alignment refusal this evidence "
                                "tolerates")
    elif policy == RESUMED_NEVER:
        w = reqs.get("steered_warm_b") or {}
        if w.get("status") == 200:
            problems.append("steered_warm_b was SERVED on a cell whose ops can never resume — the worker's "
                            "salt-discipline backstop did not fire")
        if rec.get("refusal_marker_in_log") is not True:
            problems.append(f"the worker's typed refusal ({RESUMED_REFUSAL_MARKER!r}) was not found in the "
                            "standup's vllm_server.log")
    else:
        problems.append(f"the record does not carry a resumed-prefill policy ({policy!r})")
    if rec.get("signature_errors"):
        problems.append(f"{len(rec['signature_errors'])} response(s) lacked evidence: {rec['signature_errors'][0]}")
    cl = rec.get("closeness") or {}
    ev = ("; evidence — steered warm vs cold: " + _closeness_brief(cl.get("steered"))
          + "; passthrough warm vs cold (the caching floor): " + _closeness_brief(cl.get("passthrough"))
          if cl else "")
    if problems:
        return False, f"policy={policy}: " + "; ".join(problems) + ev
    w = att.get("steered_warm_b") or {}
    if policy == RESUMED_ALWAYS:
        r = att.get("steered_warm_b_repeat") or {}
        m = att.get("steered_multiturn_warm") or {}
        return True, (f"always: the warm steered request was served and attested RESUMED (cached "
                      f"{w.get('prefill_cached_positions')}, computed {w.get('prefill_computed_positions')}, "
                      f"{w.get('n_applications_modifying_resumed')} modifying applications on the resumed rows), "
                      f"the cold one not; steering bit on the resumed prefill; the identical-prompt repeat "
                      f"resumed too (cached {r.get('prefill_cached_positions')}, computed "
                      f"{r.get('prefill_computed_positions')}); the multi-turn resume served and attested "
                      f"(cached {m.get('prefill_cached_positions')}, computed {m.get('prefill_computed_positions')})"
                      + ev)
    if policy == RESUMED_CONDITIONAL:
        rep_ = reqs.get("steered_warm_b_repeat") or {}
        rep_att = att.get("steered_warm_b_repeat") or {}
        rep_claim = rec.get("repeat_claimed_failure")
        return True, (f"conditional: the safe single-turn warm request was served and attested RESUMED (cached "
                      f"{w.get('prefill_cached_positions')}, computed {w.get('prefill_computed_positions')}, "
                      f"{w.get('n_applications_modifying_resumed')} modifying applications on the resumed rows), "
                      "the cold one not; steering bit on the resumed prefill; the unsafe multi-turn resume was "
                      "CONTAINED (engine 200, ResumedPrefillRefusal claimed) and the engine kept serving; "
                      f"evidence — identical-prompt repeat: status {rep_.get('status')!r}, attested resumed "
                      f"{rep_att.get('prefill_resumed')!r} (cached {rep_att.get('prefill_cached_positions')}, "
                      f"computed {rep_att.get('prefill_computed_positions')}), claimed refusal "
                      f"{(rep_claim or {}).get('error_type') if isinstance(rep_claim, Mapping) else rep_claim!r}"
                      + ev)
    return True, ("never: the warm steered request was refused (status "
                  f"{(reqs.get('steered_warm_b') or {}).get('status')!r}) and the worker's typed refusal is in "
                  "the server log; the cold request served" + ev)


def _claim_type(claim: Any) -> Optional[str]:
    """The ``error_type`` of a claimed worker failure record (``None`` for no
    claim; a non-record reads as its own repr so the detail names it)."""
    if claim is None:
        return None
    if isinstance(claim, Mapping):
        return str(claim.get("error_type"))
    return repr(claim)


def _closeness_brief(c: Any) -> str:
    if not isinstance(c, Mapping):
        return "n/a"
    return (f"equal={c.get('equal')}, first token divergence {c.get('first_token_divergence')}, "
            f"pos0 max |dlogprob| over {c.get('pos0_top_tokens_compared')} top tokens "
            f"{_fmt_g(c.get('pos0_max_abs_dlogprob_top'))}")


# --------------------------------------------------------------------------- #
# THE DECLARED-RENDER CASES (pure parts)                                      #
# --------------------------------------------------------------------------- #

#: the three named cases (module docstring, THE DECLARED-RENDER CASES);
#: selectable by name through ``--cases``, never part of the derived inventory
DECLARED_THINKING = "declared_thinking"
ALL_POSITIONS_CARVEOUT = "all_positions_carveout"
ALL_TURNS_RENDER_CHECK = "all_turns_render_check"
DECLARED_RENDER_CASES: Tuple[str, ...] = (DECLARED_THINKING, ALL_POSITIONS_CARVEOUT,
                                          ALL_TURNS_RENDER_CHECK)
#: case -> the serve VARIANT its steered cell wears (``serve_declaration.
#: SERVE_VARIANTS``; the cell is spelled ``<base>@<variant>``)
DECLARED_RENDER_VARIANT: Dict[str, str] = {DECLARED_THINKING: "thinking",
                                           ALL_POSITIONS_CARVEOUT: "all_positions",
                                           ALL_TURNS_RENDER_CHECK: "all_turns"}
#: case -> the families it runs on. Thinking moves the header on Qwen3.5 only
#: (gemma-4's switch leaves it in place: a ``@thinking`` cell there is admitted
#: with an EQUAL second header since 2026-09-10, but this gate's declared-thinking
#: case certifies the MOVING header and stays on Qwen); the all-turns certification refuses Qwen (its
#: successor-turn probe re-tokenizes) and the all-positions corner belongs to
#: experiment 2's gemma target. A case named for another family REFUSES at
#: startup rather than standing up something the plan did not ask for.
DECLARED_RENDER_FAMILIES: Dict[str, Tuple[str, ...]] = {DECLARED_THINKING: ("qwen35_4b",),
                                                        ALL_POSITIONS_CARVEOUT: ("gemma4e4b",),
                                                        ALL_TURNS_RENDER_CHECK: ("gemma4e4b",)}
#: the REAL response-fit object every declared-render case serves: the
#: scale-12 bank's ``safety_refusal__psr_kl_all`` (the bank's spelling; harness
#: ``si_safety__psr_kl_all``, derived through ``harness_cell_id`` at run time,
#: never spelled). No synthetic fallback: the cases are about serving THIS
#: object out of its fit support, so a bank without it refuses.
DECLARED_RENDER_SI = "safety_refusal"
DECLARED_RENDER_ARM = "psr_kl_all"
DECLARED_RENDER_CELL = f"{DECLARED_RENDER_SI}__{DECLARED_RENDER_ARM}"
#: the standup side tag of a declared-render standup (the candidate tree; no
#: reference standup exists for these cases — the base commit cannot spell a
#: variant cell)
DECLARED = "candidate_declared"
DECLARED_FORCED = "candidate_declared_forced"

#: PROXY LITERALS the cases assert against (``proxy.py``; pinned equal by
#: tests/test_lens_declared_render_gate.py — literals here because the CPU
#: tests import this module without the serving stack)
THINKING_ON_KEY = "petri_thinking_on"
CHAT_TEMPLATE_KWARGS_KEY = "chat_template_kwargs"
ENABLE_THINKING_KEY = "enable_thinking"
THINKING_KWARGS_INVALID = "thinking_kwargs_invalid"
THINKING_ON_UNDECLARED = "thinking_on_undeclared"
THINKING_ON_MISMATCH = "thinking_on_mismatch"
THINKING_ON_NOT_LENS = "thinking_on_not_lens"
THINKING_ON_CELL_NOT_DECLARED = "thinking_on_cell_not_declared"
ALL_TURNS_RENDER_REFUSED = "all_turns_render_refused"
ALL_TURNS_EXPECTED_UNAVAILABLE = "all_turns_expected_unavailable"
LEDGER_KEY_RENDER_REGIME = "render_regime"
LEDGER_KEY_ALL_TURNS_CHECK = "all_turns_check"
ALL_TURNS_CHECK_OK = "ok"
RENDER_REGIME_THINKING_ON = "thinking_on"
RENDER_REGIME_THINKING_OFF = "thinking_off"
LEDGER_KEY_SALT_REGIME = "cache_salt_regime"
LEDGER_KEY_RESUMED_POLICY = "resumed_prefill_policy"
SALT_REGIME_CELL_STABLE = "cell-stable"
SALT_REGIME_REQUEST_UNIQUE = "request-unique"
POLICY_DECLARED_SHARES_NOTHING = "declared-request-unique"
POLICY_FORCED_REQUEST_UNIQUE = "forced-request-unique"
POLICY_NOT_CONSULTED = "not-consulted"
LENS_SALT_POLICY_ATTESTED = "attested"
LENS_SALT_POLICY_REQUEST_UNIQUE_ALL = "request_unique_all"
LENS_SALT_POLICY_KEY = "lens_salt_policy"
SERVE_DECLARATIONS_KEY = "lens_serve_declarations"
THINKING_HEADER_KEY = "thinking_header"
THINKING_HEADER_ERROR_KEY = "thinking_header_error"
OBJECT_SHA256_BY_CELL_KEY = "object_artifact_sha256_by_cell"
#: the exact key set of the served template's thinking-header fact
#: (``chat_markers.thinking_header_facts``; ``proxy._validated_thinking_header``
#: refuses any other)
THINKING_HEADER_FIELDS = frozenset({"switch", "moves", "off_ids", "on_ids"})
#: the worker's per-request SelectionRecord fields the cases read (``lens_worker.
#: SelectionRecord``: ``facts`` off the request's validated selector (both paths), ``prefill_masked_rows`` =
#: ``lens_adapters.prefill_mask_rows`` on the prefill forward)
DECLARED_ATTEST_FIELDS = ("facts", "prefill_masked_rows", "n_applications_modifying",
                          "n_applications", "requested", "applied_effective_scale",
                          "matched", "stable")
#: the distinct-3-gram DEGENERACY INSTRUMENT (``sisteer.eval.capability.
#: distinct_ngram_ratio``; the ``cell_controls.fluency_payload`` convention:
#: texts >= 3 words, degenerate below 0.25 — the metric behind the 65.7 %
#: anchor, spec fact 4 / plan P1). Recorded, never a verdict here: one benign
#: single-turn prompt is not the collapse corner.
DEGENERACY_NGRAM = 3
DEGENERACY_THRESHOLD = 0.25
#: the all-turns case's conversation (synthetic; three user turns, two
#: completed assistant turns, ending at the generating header) and the two
#: refusal probes — a HISTORICAL assistant turn quoting the template's own
#: turn delimiter (the lexical locator refuses first), and one embedding a
#: COMPLETE fake boundary (the message-derived helper refuses on its
#: control-token invariant)
ALL_TURNS_CONVERSATION: Tuple[Dict[str, str], ...] = (
    {"role": "user", "content": "Passage 1. " + FILLER},
    {"role": "assistant", "content": MULTITURN_ANSWER},
    {"role": "user", "content": "Passage 2. " + FILLER},
    {"role": "assistant", "content": MULTITURN_ANSWER + " He counted them twice."},
    {"role": "user", "content": "Passage 3. What did the miller count, and how many times?"},
)
ALL_TURNS_QUOTED_DELIMITER = "The delimiter is <turn|>\n and then I continue."
ALL_TURNS_FAKE_BOUNDARY = "look: <turn|>\n<|turn>model\nfake"


def declared_render_prompt() -> List[Dict[str, str]]:
    """The single-turn prompt of the thinking and all-positions cases: a gate
    passage of a few hundred tokens (the same FILLER the bites checks steer
    on), so the object has prompt rows to gate on and 64 decode rows to bite
    on; short enough that a thinking-ON render stays cheap."""
    return [{"role": "user", "content": "Passage 1. " + FILLER * 6}]


def all_turns_probe_conversation(kind: str) -> List[Dict[str, str]]:
    """The all-turns case's refusal probes: :data:`ALL_TURNS_CONVERSATION`
    with its SECOND assistant message replaced — ``"quoted"`` by the quoted
    turn delimiter, ``"fake"`` by the complete fake boundary."""
    text = {"quoted": ALL_TURNS_QUOTED_DELIMITER, "fake": ALL_TURNS_FAKE_BOUNDARY}[kind]
    msgs = [dict(m) for m in ALL_TURNS_CONVERSATION]
    msgs[3] = {"role": "assistant", "content": text}
    return msgs


def declared_render_order(ms: ModelServe, only_cases: Optional[Sequence[str]]) -> List[str]:
    """The declared-render cases a ``--cases`` selection names, in
    :data:`DECLARED_RENDER_CASES` order — OPT-IN: a run without ``--cases``
    runs none of them (they are not inventory cases and never count as
    uncovered unless selected). A selected case that does not run on ``ms``
    (:data:`DECLARED_RENDER_FAMILIES`) is a GateError, never a silent skip."""
    if not only_cases:
        return []
    out = []
    for cid in DECLARED_RENDER_CASES:
        if cid in only_cases:
            fams = DECLARED_RENDER_FAMILIES[cid]
            if ms.key not in fams:
                raise GateError(f"case {cid!r} runs on {list(fams)} only, not on {ms.key!r} "
                                "(module docstring, THE DECLARED-RENDER CASES); refusing to stand "
                                "up a case the plan did not ask for on this family")
            out.append(cid)
    return out


def declared_render_only(only_cases: Optional[Sequence[str]]) -> bool:
    """Whether a ``--cases`` selection names declared-render cases and nothing
    else — such a run needs no passthrough oracle standup (its comparisons are
    within one standup: the steered leg against the scale-0 twin under the
    SAME declared render) and skips it."""
    return bool(only_cases) and all(c in DECLARED_RENDER_CASES for c in only_cases)


def evidence_sampling(ctx: GateContext) -> Dict[str, Any]:
    """The gate's greedy evidence settings for one request (the same keys
    ``_submit_batch`` / ``upstream_payload`` use): temperature 0, seed 0,
    logprobs + top-logprobs + token ids on, the gate's max_tokens / ignore_eos."""
    return {"max_tokens": int(ctx.max_tokens), "temperature": 0.0, "seed": 0, "logprobs": True,
            "top_logprobs": int(ctx.top_logprobs), "return_token_ids": True,
            "ignore_eos": bool(ctx.ignore_eos), "stream": False}


def thinking_request(vmodel: str, messages: Sequence[Mapping[str, Any]], sampling: Mapping[str, Any],
                     *, enable_thinking: Optional[bool], declare: bool,
                     extra_kwargs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """ONE chat-completions body of the thinking matrix (byte for
    byte in its keys): ``chat_template_kwargs: {"enable_thinking": <bool>}``
    when ``enable_thinking`` is not None (``extra_kwargs`` adds foreign keys
    for the shape-refusal probe), and ``petri_thinking_on: true`` when
    ``declare`` — the declared ON shape is kwargs true + declaration on the
    ``@thinking`` cell's vmodel AND on the ``none__lens0`` twin's."""
    body: Dict[str, Any] = {"model": vmodel, "messages": [dict(m) for m in messages]}
    if enable_thinking is not None:
        kw: Dict[str, Any] = {ENABLE_THINKING_KEY: bool(enable_thinking)}
        kw.update(extra_kwargs or {})
        body[CHAT_TEMPLATE_KWARGS_KEY] = kw
    if declare:
        body[THINKING_ON_KEY] = True
    body.update(sampling)
    return body


def plain_request(vmodel: str, messages: Sequence[Mapping[str, Any]], sampling: Mapping[str, Any]
                  ) -> Dict[str, Any]:
    """A chat-completions body with no render keys at all (the gemma cases:
    the kwarg is absent, so the stamped ``render_regime`` is ``unspecified``)."""
    body: Dict[str, Any] = {"model": vmodel, "messages": [dict(m) for m in messages]}
    body.update(sampling)
    return body


def expected_declarations(variant: str) -> Dict[str, Any]:
    """The declaration a ``@<variant>`` cell is attested under, DERIVED from
    the closed vocabulary (``serve_declaration.variant_declarations``) — never
    spelled here."""
    from sisteer.capeval.serving import serve_declaration as sd
    return sd.variant_declarations(variant)


def variant_cell(base_tt: str, variant: str) -> str:
    """``<base>@<variant>`` through ``serve_declaration.join_variant`` (an
    unknown variant refuses there)."""
    from sisteer.capeval.serving import serve_declaration as sd
    return sd.join_variant(base_tt, variant)


def response_rows_expected(header_ids: Any) -> Optional[int]:
    """How many PREFILL rows a ``positions="response"`` op selects on a
    single-turn render that ends at the header: exactly the header's width
    (the mask runs from the located header to the prompt end). Under a
    declared thinking-ON render it is the ON header's width (5 on Qwen3.5),
    under OFF the OFF header's (7) — the plan's 7-vs-5 evidence, DERIVED from
    the served template's ids, never a literal."""
    if not isinstance(header_ids, (list, tuple)) or not header_ids:
        return None
    return len(header_ids)


def all_positions_rows_expected(prompt_len: Any, *, skip_bos: bool) -> Optional[int]:
    """How many PREFILL rows an ``all``-positions op selects: every prompt
    row, minus the BOS row when the object skips it."""
    if not isinstance(prompt_len, int) or prompt_len <= 0:
        return None
    return prompt_len - (1 if skip_bos else 0)


def all_turns_spans_of(ids: Sequence[int], header_ids: Sequence[int], turn_end_ids: Sequence[int]
                       ) -> Tuple[Tuple[int, int], ...]:
    """THE GATE'S OWN all-turns span computation over a rendered id sequence
    (``/v1/tokenize``'s ids): the worker's arithmetic (``span_locator``) on
    the same ids — ``locate_final_header`` then ``assistant_turn_spans``."""
    from sisteer.capeval.serving import span_locator as sl
    final = sl.locate_final_header(list(ids), tuple(header_ids), declared=False)
    return tuple(sl.assistant_turn_spans(list(ids), tuple(header_ids), tuple(turn_end_ids), final))


def all_turns_rows_expected(spans: Sequence[Tuple[int, int]]) -> int:
    """The all-turns mask's PREFILL row count: the sum of its spans (each
    historical span header + content + scaffold; the generating span the
    header to the prompt end)."""
    return int(sum(int(hi) - int(lo) for lo, hi in spans))


def degeneracy_instrument(text: Any) -> Dict[str, Any]:
    """The distinct-3-gram instrument over ONE served text (:data:`DEGENERACY_
    NGRAM` / :data:`DEGENERACY_THRESHOLD`): ``{ratio, n_words, degenerate,
    ran}`` — ``degenerate`` is judged only on texts of >= 3 words (the
    convention), ``None`` otherwise; ``ran`` says the instrument produced a
    number. Recorded as EVIDENCE."""
    from sisteer.eval.capability import distinct_ngram_ratio
    if not isinstance(text, str):
        return {"ran": False, "error": f"no text to measure ({type(text).__name__})"}
    words = text.split()
    ratio = float(distinct_ngram_ratio([text], n=DEGENERACY_NGRAM))
    return {"ran": True, "ratio": ratio, "n_words": len(words), "ngram": DEGENERACY_NGRAM,
            "threshold": DEGENERACY_THRESHOLD,
            "degenerate": (ratio < DEGENERACY_THRESHOLD) if len(words) >= DEGENERACY_NGRAM else None}


def _served(entry: Any) -> Tuple[bool, str]:
    """The gate's PRODUCTION CONTRACT for one lens request (module docstring,
    THE SHARED-PREFILL CASE): HTTP 200, a full signature, and — for a steered
    leg — an EMPTY claimed failure. A missing entry is not served."""
    if not isinstance(entry, Mapping):
        return False, "no request recorded"
    if entry.get("status") != 200:
        return False, f"status {entry.get('status')!r} (code {entry.get('code')!r}): {str(entry.get('error'))[:200]}"
    if "signature" not in entry:
        return False, f"200 without a full signature ({entry.get('signature_error')!r})"
    if entry.get("lens_route") and entry.get("claimed_failure") is not None:
        return False, f"200 with a claimed worker failure {_claim_type(entry.get('claimed_failure'))!r}"
    return True, f"200, {len(entry['signature'].get('token_ids') or [])} tokens"


def _attest(entry: Any) -> Dict[str, Any]:
    a = entry.get("attest") if isinstance(entry, Mapping) else None
    return a if isinstance(a, Mapping) else {}


def _petri(entry: Any) -> Dict[str, Any]:
    p = entry.get("petri") if isinstance(entry, Mapping) else None
    return p if isinstance(p, Mapping) else {}


#: THE ENGINE's per-request id suffix at the pinned vLLM (0.25.1):
#: ``InputProcessor.assign_request_id`` moves the client-visible id to
#: ``external_req_id`` and sets ``request_id = f"{external_req_id}-{random_uuid():.8}"``
#: — ``-`` plus 8 LOWERCASE hex (``random_uuid()`` is ``:016x``) — exactly ONCE
#: per ``AsyncLLM.add_request``, before any n>1 fan-out (children are then
#: ``<i>_<internal id>``). The chat completion's ``id`` is the EXTERNAL id
#: (``chatcmpl-<16 hex>``, no header from the proxy), while the runner's
#: ``input_batch.req_ids`` — the worker's SelectionRecord key — carries the
#: INTERNAL one. vLLM's own ``nixl.utils.get_base_request_id`` strips the same
#: ``-[0-9a-f]{8}$``. Every attested ``req_id`` in the shared-prefill records
#: to date wears it (the case never compared); the declared-render legs
#: (2026-09-07, qwen35_4b) were the first to compare and refused it.
ENGINE_REQUEST_ID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}\Z")


def is_engine_id_of(req_id: Any, response_id: Any) -> bool:
    """Whether SelectionRecord key ``req_id`` is THIS response's request: the
    completion ``id`` itself (``VLLM_DISABLE_REQUEST_ID_RANDOMIZATION``, an
    offline record), or that id plus exactly ONE engine suffix
    (:data:`ENGINE_REQUEST_ID_SUFFIX_RE` — the delimiter, the width and the
    alphabet as the runtime mints them; nothing before, nothing after). A
    different request's id — one merely sharing a prefix included — another
    delimiter, a second suffix or a parallel-sampling child (``<i>_<id>-…``)
    is not. An empty or missing id on either side matches nothing."""
    if req_id is None or response_id is None:
        return False
    r, x = str(req_id), str(response_id)
    if not r or not x:
        return False
    if r == x:
        return True
    return r.startswith(x) and ENGINE_REQUEST_ID_SUFFIX_RE.fullmatch(r, len(x)) is not None


def _one_entry(entry: Any) -> Tuple[bool, str]:
    """Exactly one SelectionRecord entry after the reset, and it is THIS
    request's — its key the completion id or the id + the engine's one
    ``-<8 hex>`` internal suffix (:func:`is_engine_id_of`). Two entries, or one
    for another request, fail; an id the response never carried is not judged."""
    a = _attest(entry)
    if a.get("n_entries") != 1:
        return False, f"SelectionRecord held {a.get('n_entries')!r} entries after the request (expected exactly 1)"
    if a.get("req_id") is not None and entry.get("id") is not None and not is_engine_id_of(a["req_id"], entry["id"]):
        return False, (f"the one SelectionRecord entry is {a['req_id']!r}, not this request's id {entry['id']!r} "
                       f"(nor that id + the engine's one '-<8 hex>' internal suffix)")
    return True, f"one SelectionRecord entry ({a.get('req_id')!r})"


def _refused(entry: Any, code: str) -> Tuple[bool, str]:
    """A proxy REFUSAL: HTTP 400 with the named ``error.code`` and NOTHING
    reaching the worker (an empty SelectionRecord after the request)."""
    if not isinstance(entry, Mapping):
        return False, "no request recorded"
    if entry.get("status") != 400 or entry.get("code") != code:
        return False, (f"expected 400 {code!r}, got status {entry.get('status')!r} code "
                       f"{entry.get('code')!r}: {str(entry.get('error'))[:200]}")
    n = _attest(entry).get("n_entries")
    if n not in (0, None):
        return False, f"400 {code!r} but the worker recorded {n} request entr(y/ies) — the refusal was not at the proxy"
    return True, f"400 {code!r}, nothing reached the worker"


def _stamp_ok(entry: Any, *, regime: Optional[str], policy: Optional[str],
              render: Optional[str]) -> Tuple[bool, str]:
    p = _petri(entry)
    got = (p.get(LEDGER_KEY_SALT_REGIME), p.get(LEDGER_KEY_RESUMED_POLICY), p.get(LEDGER_KEY_RENDER_REGIME))
    want = (regime, policy, render)
    problems = [f"{k}={g!r} (expected {w!r})" for k, g, w in zip(
        (LEDGER_KEY_SALT_REGIME, LEDGER_KEY_RESUMED_POLICY, LEDGER_KEY_RENDER_REGIME), got, want)
        if w is not None and g != w]
    if problems:
        return False, "petri stamp: " + ", ".join(problems)
    return True, (f"petri stamp {LEDGER_KEY_SALT_REGIME}={got[0]!r} {LEDGER_KEY_RESUMED_POLICY}={got[1]!r} "
                  f"{LEDGER_KEY_RENDER_REGIME}={got[2]!r}")


def _rows_ok(entry: Any, expected: Any, what: str) -> Tuple[bool, str]:
    got = _attest(entry).get("prefill_masked_rows")
    if not isinstance(expected, int):
        return False, f"{what}: no expected row count could be derived ({expected!r})"
    if got != expected:
        return False, f"{what}: the worker filed prefill_masked_rows={got!r}, expected {expected}"
    return True, f"{what}: prefill_masked_rows={got} == {expected}"


def _modifying_ok(entry: Any, *, want_zero: bool) -> Tuple[bool, str]:
    n = _attest(entry).get("n_applications_modifying")
    if not isinstance(n, int):
        return False, f"no n_applications_modifying filed ({n!r})"
    if want_zero:
        return (n == 0), f"n_applications_modifying={n} (the scale-0 twin must modify nothing)"
    return (n >= 1), f"n_applications_modifying={n} (steering must have modified >= 1 application)"


def _selection_ok(entry: Any, *, cell: Any) -> Tuple[bool, str]:
    """The worker's SelectionRecord entry names THIS cell and applied it: ``requested``
    == the harness cell the request was routed to, ``matched`` and ``stable`` True
    (the bound object was the requested one on every layer), ``n_applications``
    >= 1 (S-G1's positive evidence — an unsteered request files no entry, a
    mis-routed one files another cell's)."""
    a = _attest(entry)
    if not cell:
        return False, "no expected harness cell recorded"
    problems = []
    if a.get("requested") != cell:
        problems.append(f"requested={a.get('requested')!r} (expected {cell!r})")
    if a.get("matched") is not True:
        problems.append(f"matched={a.get('matched')!r} (expected True)")
    if a.get("stable") is not True:
        problems.append(f"stable={a.get('stable')!r} (expected True)")
    n = a.get("n_applications")
    if not isinstance(n, int) or n < 1:
        problems.append(f"n_applications={n!r} (expected >= 1)")
    if problems:
        return False, "SelectionRecord entry: " + ", ".join(problems)
    return True, f"SelectionRecord entry: requested={cell!r} matched stable n_applications={n}"


def _scale_zero_ok(entry: Any) -> Tuple[bool, str]:
    """The scale-0 twin's SelectionRecord entry says so: ``applied_effective_scale``
    (the worker files ``float(applied.effective_scale)``) is exactly 0 — a twin
    that applied its donor's tensors at any other scale is a steered control,
    and a bites comparison against it proves nothing (AG-1)."""
    sc = _attest(entry).get("applied_effective_scale")
    if isinstance(sc, bool) or not isinstance(sc, (int, float)):
        return False, f"no applied_effective_scale filed ({sc!r})"
    return (sc == 0), f"applied_effective_scale={sc!r} (the scale-0 twin must apply at 0)"


#: THE SUCCESSFUL-LENS-LEG VALIDATOR's sub-checks, filed as ``<leg prefix>_<name>``
LENS_LEG_CHECK_NAMES: Tuple[str, ...] = ("served", "one_entry", "selection", "masked_rows", "modifying")
#: ...and the twin's extra one
LENS_TWIN_CHECK_NAMES: Tuple[str, ...] = LENS_LEG_CHECK_NAMES + ("scale",)


def lens_leg_checks(entry: Any, *, cell: Any, rows: Any, what: str, twin: bool
                    ) -> Dict[str, Tuple[bool, str]]:
    """THE SUCCESSFUL-LENS-LEG VALIDATOR (code review S-G1 / AG-1): the evidence
    EVERY lens leg whose SUCCESS is evidence must carry, in one place so no
    case can hold a leg to less — ``served`` (HTTP 200, a full signature, an
    empty claimed failure ledger), ``one_entry`` (exactly one SelectionRecord
    entry, keyed by this request's engine id — :func:`is_engine_id_of`),
    ``selection`` (``requested`` == ``cell``,
    ``matched`` and ``stable`` True, >= 1 application), ``masked_rows`` (the
    derived geometry ``rows``, named by ``what``), ``modifying`` (>= 1 on a
    steered cell; 0 on the scale-0 twin) and, for a ``twin``, ``scale``
    (``applied_effective_scale == 0``). An unsteered request (no entry), a
    misrouted one (another cell's entry) or a steered "control" (modifying,
    or at a non-zero scale) each fail here, whatever the proxy stamped."""
    out = {"served": _served(entry), "one_entry": _one_entry(entry),
           "selection": _selection_ok(entry, cell=cell),
           "masked_rows": _rows_ok(entry, rows, what),
           "modifying": _modifying_ok(entry, want_zero=twin)}
    if twin:
        out["scale"] = _scale_zero_ok(entry)
    return out


def file_lens_leg(out: Dict[str, Dict[str, Any]], prefix: str, entry: Any, *, cell: Any, rows: Any,
                  what: str, twin: bool) -> None:
    """File :func:`lens_leg_checks` on ``out`` under ``<prefix>_<name>``."""
    for name, (ok, detail) in lens_leg_checks(entry, cell=cell, rows=rows, what=what, twin=twin).items():
        out[f"{prefix}_{name}"] = {"ok": ok, "detail": detail}


#: case -> {request name: (check-key prefix, cell role)} for EVERY positive lens
#: leg the drive files — every served lens request is here (the test suite
#: pins the map complete against the green records), each validated by
#: :func:`lens_leg_checks`; role ``twin`` adds the scale check
POSITIVE_LENS_LEGS: Dict[str, Dict[str, Tuple[str, str]]] = {
    DECLARED_THINKING: {"on_steered": ("thinking_on", "variant"), "on_twin": ("thinking_twin", "twin"),
                        "off_base": ("thinking_off", "base")},
    ALL_POSITIONS_CARVEOUT: {"variant_single": ("carveout_variant", "variant"),
                             "base_single": ("carveout_base", "base"), "twin_single": ("carveout_twin", "twin")},
    ALL_TURNS_RENDER_CHECK: {"convo_variant": ("all_turns_convo", "variant"), "convo_base": ("all_turns_base", "base"),
                             "convo_twin": ("all_turns_twin", "twin"),
                             "quoted_base": ("all_turns_quoted_base", "base"),
                             "fake_base": ("all_turns_fake_base", "base")},
}
#: the carve-out's SECOND standup (``request_unique_all``): the same three legs
FORCED_LENS_LEGS: Dict[str, Tuple[str, str]] = {"variant_single": ("carveout_forced_variant", "variant"),
                                                "base_single": ("carveout_forced_base", "base"),
                                                "twin_single": ("carveout_forced_twin", "twin")}


def _facts_ok(entry: Any, *, thinking_on: bool) -> Tuple[bool, str]:
    facts = _attest(entry).get("facts")
    if not isinstance(facts, Mapping) or "thinking_on" not in facts:
        return False, f"no facts.thinking_on filed ({facts!r})"
    ok = bool(facts.get("thinking_on")) is thinking_on
    return ok, f"facts={dict(facts)} (thinking_on expected {thinking_on})"


def _map_ok(rec: Mapping[str, Any], expected_map: Mapping[str, Any]) -> Tuple[bool, str]:
    """The attested ``lens_serve_declarations`` map — in ``endpoint.json`` AND
    echoed by ``/v1/models`` — equals ``expected_map`` EXACTLY (every expected
    cell with its literal; no other cell — the base cell absent)."""
    ep = (rec.get("endpoint") or {})
    if SERVE_DECLARATIONS_KEY not in ep:
        return False, (f"endpoint.json carries no {SERVE_DECLARATIONS_KEY} "
                       f"({ep.get(SERVE_DECLARATIONS_KEY + '_error')!r})")
    got_ep = ep.get(SERVE_DECLARATIONS_KEY)
    got_models = (rec.get("models") or {}).get(SERVE_DECLARATIONS_KEY)
    if got_ep != dict(expected_map):
        return False, f"endpoint {SERVE_DECLARATIONS_KEY}={got_ep!r}, expected exactly {dict(expected_map)!r}"
    if got_models != dict(expected_map):
        return False, f"/v1/models {SERVE_DECLARATIONS_KEY}={got_models!r}, expected exactly {dict(expected_map)!r}"
    return True, f"{SERVE_DECLARATIONS_KEY} == {dict(expected_map)!r} in endpoint.json and /v1/models"


def _differs(a: Any, b: Any) -> Tuple[Optional[bool], str]:
    sa = a.get("signature") if isinstance(a, Mapping) else None
    sb = b.get("signature") if isinstance(b, Mapping) else None
    if not isinstance(sa, Mapping) or not isinstance(sb, Mapping):
        return None, "one side carries no signature"
    if not signatures_differ_somewhere([sa], [sb]):
        return False, "identical token ids and sampled-token logprobs"
    _eq, why = signatures_equal([sa], [sb])
    return True, (why or "differs")


def _claims_ok(rec: Mapping[str, Any]) -> Tuple[bool, str]:
    reqs = rec.get("requests") or {}
    bad = [n for n, e in reqs.items() if isinstance(e, Mapping) and e.get("lens_route")
           and e.get("status") == 200 and e.get("claimed_failure") is not None]
    missing = [n for n, e in reqs.items() if isinstance(e, Mapping) and e.get("lens_route")
               and e.get("status") == 200 and "claimed_failure" not in e]
    if bad:
        return False, f"claimed worker failures on served lens requests {bad}"
    if missing:
        return False, f"served lens requests whose failure ledger was never claimed: {missing}"
    n = sum(1 for e in reqs.values() if isinstance(e, Mapping) and e.get("lens_route") and e.get("status") == 200)
    if n == 0:
        return False, "no served lens request to claim a failure ledger for — missing evidence"
    return True, f"every served lens request ({n}) claimed an EMPTY failure ledger"


def _sig_ok(rec: Mapping[str, Any]) -> Tuple[bool, str]:
    errs = rec.get("signature_errors")
    if not isinstance(errs, list):
        return False, "no signature ledger recorded — missing evidence"
    return (not errs), ("every 200 carried a full signature" if not errs
                        else f"{len(errs)} response(s) lacked evidence: {errs[0]}")


def _twin_donor_ok(rec: Mapping[str, Any], want: Any) -> Tuple[bool, str]:
    got = rec.get("twin_donor")
    if got is None:
        return False, "the registry index names no donor for none__lens0"
    return (got == want), f"none__lens0 donor {got!r} (expected {want!r})"


def _artifact_ok(rec: Mapping[str, Any], facts: Optional[Mapping[str, Any]] = None) -> Tuple[bool, str]:
    """Q8 (code review S-G2): ``endpoint.json.object_artifact_sha256`` equals
    the bank artifact's sha256 (the gate hashed the same file), the per-cell
    map names the base and the variant with that one digest — the EXACT map
    in ``endpoint.json`` AND in ``/v1/models`` (the proxy publishes only what
    it was handed, so a map missing there means the served proxy never
    received the identity) — and the export was BOUND to it. ``facts`` is the
    record whose ``endpoint`` / ``models`` are read (the carve-out case's
    forced standup); the artifact and cells come from ``rec``."""
    f = facts if facts is not None else rec
    ep = f.get("endpoint") or {}
    models = f.get("models") or {}
    want = (rec.get("object") or {}).get("sha256")
    cells = rec.get("cells") or {}
    if not want:
        return False, "the gate recorded no artifact sha256"
    if ep.get("object_artifact_sha256") != want:
        return False, (f"endpoint object_artifact_sha256={ep.get('object_artifact_sha256')!r} != bank artifact {want}"
                       + (f" ({ep.get('object_artifact_sha256_error')})" if ep.get("object_artifact_sha256_error") else ""))
    by = ep.get(OBJECT_SHA256_BY_CELL_KEY)
    exp = {cells.get("base"): want, cells.get("variant"): want}
    if by != exp:
        return False, (f"endpoint {OBJECT_SHA256_BY_CELL_KEY}={by!r}, expected {exp!r}"
                       + (f" ({ep.get('object_artifact_sha256_error')})" if ep.get("object_artifact_sha256_error") else ""))
    if ep.get("object_artifact_sha256_bound") is not True:
        return False, ("the export was not bound to the digest / the proxy was not handed it "
                       f"(object_artifact_sha256_bound={ep.get('object_artifact_sha256_bound')!r}"
                       + (f"; {ep.get('object_artifact_sha256_error')}" if ep.get("object_artifact_sha256_error") else "")
                       + ")")
    if OBJECT_SHA256_BY_CELL_KEY not in models:
        return False, f"/v1/models publishes no {OBJECT_SHA256_BY_CELL_KEY} — the served proxy holds no artifact identity"
    if models.get(OBJECT_SHA256_BY_CELL_KEY) != exp:
        return False, f"/v1/models {OBJECT_SHA256_BY_CELL_KEY}={models.get(OBJECT_SHA256_BY_CELL_KEY)!r}, expected {exp!r}"
    return True, (f"object_artifact_sha256 == {want[:16]}… on the base and the variant in endpoint.json and "
                  "/v1/models, bound at export")


def _header_attested_ok(rec: Mapping[str, Any]) -> Tuple[bool, str]:
    """Code review S-G3: the served template's thinking-header fact and the
    reason beside it are PUBLISHED CONSISTENTLY — the exact four-key fact
    (:data:`THINKING_HEADER_FIELDS`) present in ``endpoint.json`` and equal in
    ``/v1/models``, ``thinking_header_error`` present on both and equal, and
    the two PAIRED: a partial fact (``switch`` True, ``on_ids`` None — the ON
    certification unavailable) carries a non-empty reason, a complete one
    None. A partial-but-explained pair is consistent (the gemma cases never
    need the ON header); certification is the thinking case's own check."""
    ep = rec.get("endpoint") or {}
    models = rec.get("models") or {}
    if THINKING_HEADER_KEY not in ep or THINKING_HEADER_ERROR_KEY not in ep:
        return False, f"endpoint.json carries no {THINKING_HEADER_KEY} / {THINKING_HEADER_ERROR_KEY} pair"
    th, err = ep.get(THINKING_HEADER_KEY), ep.get(THINKING_HEADER_ERROR_KEY)
    if not isinstance(th, Mapping):
        return False, f"endpoint {THINKING_HEADER_KEY}={th!r} — the standup derived no fact ({err!r})"
    if set(th) != THINKING_HEADER_FIELDS:
        return False, f"endpoint {THINKING_HEADER_KEY} keys {sorted(th)} != {sorted(THINKING_HEADER_FIELDS)}"
    if models.get(THINKING_HEADER_KEY) != dict(th):
        return False, f"/v1/models {THINKING_HEADER_KEY}={models.get(THINKING_HEADER_KEY)!r} != endpoint {dict(th)!r}"
    if THINKING_HEADER_ERROR_KEY not in models:
        return False, f"/v1/models publishes no {THINKING_HEADER_ERROR_KEY} beside the fact"
    if models.get(THINKING_HEADER_ERROR_KEY) != err:
        return False, (f"/v1/models {THINKING_HEADER_ERROR_KEY}={models.get(THINKING_HEADER_ERROR_KEY)!r} != endpoint "
                       f"{err!r}")
    partial = th.get("switch") is True and th.get("on_ids") is None
    explained = isinstance(err, str) and bool(err)
    if partial and not explained:
        return False, f"a PARTIAL fact (switch, on_ids None) with no reason beside it ({THINKING_HEADER_ERROR_KEY}={err!r})"
    if explained and not partial:
        return False, f"a reason ({err!r}) beside a fact whose ON certification is not unavailable ({dict(th)!r})"
    return True, (f"{THINKING_HEADER_KEY} switch={th.get('switch')} moves={th.get('moves')} on_ids="
                  f"{'None' if th.get('on_ids') is None else len(th['on_ids'])} ids; {THINKING_HEADER_ERROR_KEY}="
                  f"{err!r}; equal in endpoint.json and /v1/models")


def declared_thinking_checks(rec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The ``declared_thinking`` case's hard checks (module docstring, THE
    DECLARED-RENDER CASES) over its drive record — every missing fact a
    FAIL."""
    out: Dict[str, Dict[str, Any]] = {}
    reqs = rec.get("requests") or {}
    th = (rec.get("endpoint") or {}).get(THINKING_HEADER_KEY) or {}
    on_rows = response_rows_expected(th.get("on_ids"))
    off_rows = response_rows_expected(th.get("off_ids"))

    def put(name, pair):
        out[name] = {"ok": pair[0], "detail": pair[1]}

    cells = rec.get("cells") or {}
    on, twin, off = reqs.get("on_steered"), reqs.get("on_twin"), reqs.get("off_base")
    # every positive lens leg through THE validator (POSITIVE_LENS_LEGS; AG-1),
    # then the case's own facts and stamps
    rows = {"on_steered": (on_rows, "ON steered (the ON header's width)"),
            "on_twin": (on_rows, "ON twin (the ON header's width)"),
            "off_base": (off_rows, "OFF base (the OFF header's width)")}
    for name, (prefix, role) in POSITIVE_LENS_LEGS[DECLARED_THINKING].items():
        file_lens_leg(out, prefix, reqs.get(name), cell=cells.get(role), rows=rows[name][0], what=rows[name][1],
                      twin=(role == "twin"))
    put("thinking_on_facts", _facts_ok(on, thinking_on=True))
    put("thinking_on_stamp", _stamp_ok(on, regime=SALT_REGIME_REQUEST_UNIQUE,
                                       policy=POLICY_DECLARED_SHARES_NOTHING,
                                       render=RENDER_REGIME_THINKING_ON))
    put("thinking_twin_facts", _facts_ok(twin, thinking_on=True))
    put("thinking_twin_stamp", _stamp_ok(twin, regime=SALT_REGIME_REQUEST_UNIQUE,
                                         policy=POLICY_DECLARED_SHARES_NOTHING,
                                         render=RENDER_REGIME_THINKING_ON))
    put("thinking_off_facts", _facts_ok(off, thinking_on=False))
    put("thinking_off_stamp", _stamp_ok(off, regime=SALT_REGIME_REQUEST_UNIQUE, policy=RESUMED_CONDITIONAL,
                                        render=RENDER_REGIME_THINKING_OFF))
    d, why = _differs(on, twin)
    out["thinking_bites_on"] = {"ok": d is True,
                                "detail": ("the declared ON steered output differs from the twin's under the same "
                                           f"render: {why}" if d else
                                           "the declared ON steered output is bit-identical to the scale-0 twin's "
                                           f"— nothing was applied under the ON render ({why})")}
    for name, code in (("thinking_refuse_undeclared", THINKING_ON_UNDECLARED),
                       ("thinking_refuse_mismatch", THINKING_ON_MISMATCH),
                       ("thinking_refuse_kwargs_invalid", THINKING_KWARGS_INVALID),
                       ("thinking_refuse_cell_not_declared", THINKING_ON_CELL_NOT_DECLARED),
                       ("thinking_refuse_not_lens", THINKING_ON_NOT_LENS)):
        put(name, _refused(reqs.get(name.replace("thinking_refuse_", "refuse_")), code))
    pt = reqs.get("passthrough_on")
    put("thinking_passthrough_on_served", _served(pt))
    put("thinking_passthrough_on_stamp", _stamp_ok(pt, regime=SALT_REGIME_CELL_STABLE, policy=POLICY_NOT_CONSULTED,
                                                   render=RENDER_REGIME_THINKING_ON))
    decl = rec.get("expected_declaration")
    if isinstance(decl, Mapping) and cells.get("variant") and cells.get("twin"):
        put("thinking_attestation_map", _map_ok(rec, {cells["variant"]: dict(decl), cells["twin"]: dict(decl)}))
    else:
        out["thinking_attestation_map"] = {"ok": False, "detail": "no expected declaration / cell ids recorded"}
    put("thinking_artifact_sha256", _artifact_ok(rec))
    tok_th = rec.get("tokenizer_thinking_header")
    tok_err = rec.get("tokenizer_thinking_header_error")
    models_th = (rec.get("models") or {}).get(THINKING_HEADER_KEY)
    if not isinstance(tok_th, Mapping) or not th:
        out["thinking_header_matches_tokenizer"] = {
            "ok": False, "detail": f"endpoint thinking_header={th!r}, tokenizer facts={tok_th!r} ({tok_err!r})"}
    else:
        # the gate's own derivation must have CERTIFIED the ON header (no reason
        # beside it): a partial derivation cannot vouch for the served fact
        ok = (tok_err is None and th.get("on_ids") is not None
              and th.get("on_ids") == tok_th.get("on_ids") and th.get("off_ids") == tok_th.get("off_ids")
              and th.get("moves") is True and models_th == th)
        out["thinking_header_matches_tokenizer"] = {
            "ok": ok, "detail": (f"thinking_header on_ids={th.get('on_ids')} off_ids={th.get('off_ids')} "
                                 f"moves={th.get('moves')}; tokenizer on={tok_th.get('on_ids')} "
                                 f"off={tok_th.get('off_ids')} (reason {tok_err!r}); /v1/models equal={models_th == th}")}
    put("thinking_header_attested", _header_attested_ok(rec))
    put("thinking_twin_donor", _twin_donor_ok(rec, cells.get("variant")))
    put("thinking_claims_empty", _claims_ok(rec))
    put("thinking_signatures", _sig_ok(rec))
    return out


def all_positions_carveout_checks(rec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The ``all_positions_carveout`` case's hard checks over its drive record
    (both standups: the default ``attested`` policy, then
    ``request_unique_all``) — every missing fact a FAIL."""
    out: Dict[str, Dict[str, Any]] = {}
    reqs = rec.get("requests") or {}
    forced = rec.get("forced") or {}
    freqs = forced.get("requests") or {}
    cells = rec.get("cells") or {}
    header_rows = response_rows_expected(((rec.get("endpoint") or {}).get("registry_markers") or {})
                                         .get("assistant_header_ids"))

    def put(name, pair):
        out[name] = {"ok": pair[0], "detail": pair[1]}

    guard = rec.get("guard") or {}
    dec, und = guard.get("declared") or {}, guard.get("undeclared") or {}
    out["carveout_guard_admits_declared"] = {
        "ok": dec.get("adapted") is True and dec.get("served_positions") == ["all"],
        "detail": (f"the real artifact adapted under serve_experiment={dec.get('experiment')!r} to "
                   f"positions {dec.get('served_positions')!r}" if dec.get("adapted")
                   else f"the declared adapt did not pass: {dec.get('error')!r}")}
    out["carveout_guard_refuses_undeclared"] = {
        "ok": und.get("refused") is True and "fit_response_serve_all_positions" in str(und.get("error")),
        "detail": (f"the UNDECLARED fit-response/serve-all adapt was refused, naming the carve-out: "
                   f"{str(und.get('error'))[:300]}" if und.get("refused")
                   else f"the undeclared adapt was NOT refused ({und!r})")}
    var, twin = reqs.get("variant_single"), reqs.get("twin_single")
    # every positive lens leg through THE validator (POSITIVE_LENS_LEGS; AG-1),
    # then the stamps: the variant's rows are P (minus BOS iff skipped), the
    # base's and the twin's the header's width (the twin's donor is the base)
    rows = {"variant_single": (rec.get("expected_all_rows"),
                               f"@all_positions (P={rec.get('prompt_len')}, skip_bos="
                               f"{(rec.get('object') or {}).get('skip_bos')})"),
            "base_single": (header_rows, "base response cell (the header's width)"),
            "twin_single": (header_rows, "twin (its donor is the base: the header's width)")}
    for name, (prefix, role) in POSITIVE_LENS_LEGS[ALL_POSITIONS_CARVEOUT].items():
        file_lens_leg(out, prefix, reqs.get(name), cell=cells.get(role), rows=rows[name][0], what=rows[name][1],
                      twin=(role == "twin"))
    put("carveout_variant_stamp", _stamp_ok(var, regime=SALT_REGIME_REQUEST_UNIQUE,
                                            policy=POLICY_DECLARED_SHARES_NOTHING, render=None))
    put("carveout_base_stamp", _stamp_ok(reqs.get("base_single"), regime=SALT_REGIME_REQUEST_UNIQUE,
                                         policy=RESUMED_CONDITIONAL, render=None))
    put("carveout_twin_stamp", _stamp_ok(twin, regime=SALT_REGIME_REQUEST_UNIQUE, policy=RESUMED_CONDITIONAL,
                                         render=None))
    d, why = _differs(var, twin)
    out["carveout_bites"] = {"ok": d is True,
                             "detail": (f"the @all_positions output differs from the scale-0 twin's: {why}" if d
                                        else f"the @all_positions output equals the twin's — nothing applied ({why})")}
    decl = rec.get("expected_declaration")
    if isinstance(decl, Mapping) and cells.get("variant"):
        put("carveout_attestation_map", _map_ok(rec, {cells["variant"]: dict(decl)}))
    else:
        out["carveout_attestation_map"] = {"ok": False, "detail": "no expected declaration / cell ids recorded"}
    ep = rec.get("endpoint") or {}
    pol = ep.get(LENS_SALT_POLICY_KEY), (rec.get("models") or {}).get(LENS_SALT_POLICY_KEY)
    out["carveout_attested_policy_endpoint"] = {
        "ok": pol == (LENS_SALT_POLICY_ATTESTED, LENS_SALT_POLICY_ATTESTED),
        "detail": f"endpoint / models {LENS_SALT_POLICY_KEY} = {pol!r} (expected {LENS_SALT_POLICY_ATTESTED!r} twice)"}
    put("carveout_artifact_sha256", _artifact_ok(rec))
    put("carveout_thinking_header_attested", _header_attested_ok(rec))
    deg = rec.get("degeneracy") or {}
    dv = deg.get("variant_single") or {}
    out["carveout_degeneracy_instrument"] = {
        "ok": dv.get("ran") is True and _finite_number(dv.get("ratio")),
        "detail": (f"distinct-{DEGENERACY_NGRAM}-gram ratio {dv.get('ratio')!r} over {dv.get('n_words')} words "
                   f"(degenerate<{DEGENERACY_THRESHOLD}: {dv.get('degenerate')!r}); base "
                   f"{(deg.get('base_single') or {}).get('ratio')!r} — evidence, never a verdict"
                   if dv.get("ran") else f"the instrument did not run: {dv!r}")}
    # the second standup: request_unique_all
    fep = forced.get("endpoint") or {}
    fpol = fep.get(LENS_SALT_POLICY_KEY), (forced.get("models") or {}).get(LENS_SALT_POLICY_KEY)
    out["carveout_forced_policy_endpoint"] = {
        "ok": fpol == (LENS_SALT_POLICY_REQUEST_UNIQUE_ALL, LENS_SALT_POLICY_REQUEST_UNIQUE_ALL),
        "detail": (f"re-stood endpoint / models {LENS_SALT_POLICY_KEY} = {fpol!r} (expected "
                   f"{LENS_SALT_POLICY_REQUEST_UNIQUE_ALL!r} twice)")}
    # S-G1: the forced legs carry the SAME worker evidence as the default-policy
    # legs (THE validator, FORCED_LENS_LEGS) — a fresh standup, so nothing from
    # the first one vouches for these requests having reached the lens worker
    # at all. The geometry is the FORCED standup's own (its /v1/tokenize P, its
    # registry header); the stamp is the forced regime's.
    forced_header_rows = response_rows_expected((fep.get("registry_markers") or {}).get("assistant_header_ids"))
    forced_rows = {"variant_single": (forced.get("expected_all_rows"),
                                      f"forced @all_positions (P={forced.get('prompt_len')}, skip_bos="
                                      f"{((forced.get('object') or rec.get('object')) or {}).get('skip_bos')})"),
                   "base_single": (forced_header_rows, "forced base response cell (the header's width)"),
                   "twin_single": (forced_header_rows, "forced twin (its donor is the base: the header's width)")}
    for name, (prefix, role) in FORCED_LENS_LEGS.items():
        e = freqs.get(name)
        file_lens_leg(out, prefix, e, cell=cells.get(role), rows=forced_rows[name][0], what=forced_rows[name][1],
                      twin=(role == "twin"))
        put(f"{prefix}_stamp", _stamp_ok(e, regime=SALT_REGIME_REQUEST_UNIQUE, policy=POLICY_FORCED_REQUEST_UNIQUE,
                                         render=None))
    if isinstance(decl, Mapping) and cells.get("variant"):
        put("carveout_forced_attestation_map", _map_ok(forced, {cells["variant"]: dict(decl)}))
    else:
        out["carveout_forced_attestation_map"] = {"ok": False, "detail": "no expected declaration / cell ids recorded"}
    put("carveout_forced_artifact_sha256", _artifact_ok(rec, forced))
    put("carveout_forced_thinking_header_attested", _header_attested_ok(forced))
    put("carveout_twin_donor", _twin_donor_ok(rec, cells.get("base")))
    ok_a, d_a = _claims_ok(rec)
    ok_b, d_b = _claims_ok(forced)
    out["carveout_claims_empty"] = {"ok": ok_a and ok_b, "detail": f"attested: {d_a}; forced: {d_b}"}
    ok_a, d_a = _sig_ok(rec)
    ok_b, d_b = _sig_ok(forced)
    out["carveout_signatures"] = {"ok": ok_a and ok_b, "detail": f"attested: {d_a}; forced: {d_b}"}
    return out


def all_turns_render_checks(rec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The ``all_turns_render_check`` case's hard checks over its drive record
    — every missing fact a FAIL."""
    out: Dict[str, Dict[str, Any]] = {}
    reqs = rec.get("requests") or {}
    cells = rec.get("cells") or {}
    markers = (rec.get("endpoint") or {}).get("registry_markers") or {}
    header_rows = response_rows_expected(markers.get("assistant_header_ids"))

    def put(name, pair):
        out[name] = {"ok": pair[0], "detail": pair[1]}

    convo, base = reqs.get("convo_variant"), reqs.get("convo_base")
    spans = rec.get("located_spans")
    # every positive lens leg through THE validator (POSITIVE_LENS_LEGS; AG-1:
    # the twin the bites comparison rests on and the two refusal probes' base
    # CONTROLS were once held to _served alone): the variant's rows are the
    # gate's own span sum, every base-object leg's and the twin's the header's
    # width (a response cell masks the final header whatever the history; the
    # twin's donor is the base)
    hdr = "the header's width"
    rows = {"convo_variant": (rec.get("expected_all_turns_rows"),
                              f"@all_turns over the gate's own spans {spans} of P={rec.get('prompt_len')}"),
            "convo_base": (header_rows, f"base response cell on the 3-turn render ({hdr})"),
            "convo_twin": (header_rows, f"twin on the 3-turn render (its donor is the base: {hdr})"),
            "quoted_base": (header_rows, f"base control on the quoted-delimiter probe ({hdr})"),
            "fake_base": (header_rows, f"base control on the fake-boundary probe ({hdr})")}
    for name, (prefix, role) in POSITIVE_LENS_LEGS[ALL_TURNS_RENDER_CHECK].items():
        file_lens_leg(out, prefix, reqs.get(name), cell=cells.get(role), rows=rows[name][0], what=rows[name][1],
                      twin=(role == "twin"))
    label = _petri(convo).get(LEDGER_KEY_ALL_TURNS_CHECK)
    out["all_turns_check_label_ok"] = {"ok": label == ALL_TURNS_CHECK_OK,
                                       "detail": f"petri.{LEDGER_KEY_ALL_TURNS_CHECK}={label!r} (expected {ALL_TURNS_CHECK_OK!r})"}
    put("all_turns_stamp", _stamp_ok(convo, regime=SALT_REGIME_REQUEST_UNIQUE,
                                     policy=POLICY_DECLARED_SHARES_NOTHING, render=None))
    base_petri = _petri(base)
    unlabelled = bool(base_petri) and LEDGER_KEY_ALL_TURNS_CHECK not in base_petri
    out["all_turns_base_unlabelled"] = {"ok": unlabelled,
                                        "detail": ("no all_turns_check label on the base route" if unlabelled
                                                   else ("the base route carries an all_turns_check label" if base_petri
                                                         else "no petri stamp on the base leg — nothing to read"))}
    for name in ("convo_twin", "quoted_base", "fake_base"):
        put(f"{POSITIVE_LENS_LEGS[ALL_TURNS_RENDER_CHECK][name][0]}_stamp",
            _stamp_ok(reqs.get(name), regime=SALT_REGIME_REQUEST_UNIQUE, policy=RESUMED_CONDITIONAL, render=None))
    for kind, code in (("quoted", ALL_TURNS_RENDER_REFUSED), ("fake", ALL_TURNS_EXPECTED_UNAVAILABLE)):
        # the VARIANT's refusal; the base control's success is the validator's business above
        ok1, d1 = _refused(reqs.get(f"{kind}_variant"), code)
        extra_ok = True
        if kind == "fake":
            msg = str((reqs.get("fake_variant") or {}).get("error") or "")
            extra_ok = "ChatMarkerError" in msg and "control tokens survived" in msg
            d1 += ("; the body names the ChatMarkerError (control tokens survived inside the content span)"
                   if extra_ok else f"; the body does NOT name the ChatMarkerError: {msg[:200]!r}")
        out[f"all_turns_refuse_{kind}"] = {"ok": ok1 and extra_ok, "detail": f"variant: {d1}"}
    decl = rec.get("expected_declaration")
    if isinstance(decl, Mapping) and cells.get("variant"):
        put("all_turns_attestation_map", _map_ok(rec, {cells["variant"]: dict(decl)}))
    else:
        out["all_turns_attestation_map"] = {"ok": False, "detail": "no expected declaration / cell ids recorded"}
    tok_te = rec.get("tokenizer_turn_end_ids")
    got_te = markers.get("assistant_turn_end_ids")
    out["all_turns_scaffold"] = {
        "ok": bool(got_te) and list(got_te) == list(tok_te or []),
        "detail": f"registry assistant_turn_end_ids={got_te!r}; the gate's own derivation {tok_te!r}"}
    d, why = _differs(convo, reqs.get("convo_twin"))
    out["all_turns_bites"] = {"ok": d is True,
                              "detail": (f"the @all_turns output differs from the scale-0 twin's on the same "
                                         f"conversation: {why}" if d else f"identical to the twin — nothing applied ({why})")}
    put("all_turns_twin_donor", _twin_donor_ok(rec, cells.get("base")))
    put("all_turns_artifact_sha256", _artifact_ok(rec))
    put("all_turns_thinking_header_attested", _header_attested_ok(rec))
    put("all_turns_claims_empty", _claims_ok(rec))
    put("all_turns_signatures", _sig_ok(rec))
    return out


DECLARED_RENDER_CHECKERS: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Dict[str, Any]]]] = {
    DECLARED_THINKING: declared_thinking_checks,
    ALL_POSITIONS_CARVEOUT: all_positions_carveout_checks,
    ALL_TURNS_RENDER_CHECK: all_turns_render_checks,
}


def declared_render_checks(case_id: str, rec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Dispatch to the case's checker; an unknown case id is a GateError."""
    if case_id not in DECLARED_RENDER_CHECKERS:
        raise GateError(f"{case_id!r} is not a declared-render case ({DECLARED_RENDER_CASES})")
    return DECLARED_RENDER_CHECKERS[case_id](rec)


def declared_render_report_lines(doc: Mapping[str, Any]) -> List[str]:
    """The report's DECLARED-RENDER section: per case the served ids, the
    attested map, the twin's donor, every check with its observed literal, and
    each request's status / code / stamp / masked rows — the positive artifact
    a reader can hold the run to."""
    lines: List[str] = []
    for kr in doc.get("cases") or []:
        cid = kr.get("case_id")
        if cid not in DECLARED_RENDER_CASES:
            continue
        lines.append(f"- DECLARED-RENDER case {cid}: cells {kr.get('cells_tt')} -> harness "
                     f"{(kr.get('cells') or {}).get('base')!r} / {(kr.get('cells') or {}).get('variant')!r}, twin "
                     f"{(kr.get('cells') or {}).get('twin')!r} (donor {kr.get('twin_donor')!r}); object "
                     f"{kr.get('objects')} {kr.get('source')} sha256 {str((kr.get('object') or {}).get('sha256'))[:16]}")
        if kr.get("error"):
            lines.append(f"  ERROR: {kr['error'][:300]}")
            continue
        ep = kr.get("endpoint") or {}
        lines.append(f"  attested {SERVE_DECLARATIONS_KEY}: {ep.get(SERVE_DECLARATIONS_KEY)!r}; "
                     f"{LENS_SALT_POLICY_KEY}: {ep.get(LENS_SALT_POLICY_KEY)!r}; object_artifact_sha256: "
                     f"{str(ep.get('object_artifact_sha256'))[:16]} (bound {ep.get('object_artifact_sha256_bound')!r}); "
                     f"{THINKING_HEADER_KEY}: {ep.get(THINKING_HEADER_KEY)!r}; registry_markers: "
                     f"{ep.get('registry_markers')!r}")
        for label, reqs in (("", kr.get("requests") or {}),
                            ("forced ", (kr.get("forced") or {}).get("requests") or {})):
            for name, e in reqs.items():
                if not isinstance(e, Mapping):
                    continue
                a, p = e.get("attest") or {}, e.get("petri") or {}
                lines.append(f"  {label}{name} [{e.get('vmodel')}]: status {e.get('status')}"
                             + (f" code {e.get('code')!r}" if e.get("status") != 200 else "")
                             + (f", rows {a.get('prefill_masked_rows')!r}, facts {a.get('facts')!r}, "
                                f"n_mod {a.get('n_applications_modifying')!r}" if a.get("n_entries") == 1
                                else f", record entries {a.get('n_entries')!r}")
                             + (f", stamp {p.get(LEDGER_KEY_SALT_REGIME)!r}/{p.get(LEDGER_KEY_RESUMED_POLICY)!r}/"
                                f"{p.get(LEDGER_KEY_RENDER_REGIME)!r}"
                                + (f"/{LEDGER_KEY_ALL_TURNS_CHECK}={p.get(LEDGER_KEY_ALL_TURNS_CHECK)!r}"
                                   if LEDGER_KEY_ALL_TURNS_CHECK in p else "") if p else ""))
        if kr.get("guard"):
            lines.append(f"  guard probe: {kr['guard']!r}")
        if kr.get("degeneracy"):
            lines.append(f"  degeneracy instrument (evidence): {kr['degeneracy']!r}")
        if kr.get("located_spans") is not None:
            lines.append(f"  all-turns spans (the gate's own, over /v1/tokenize ids; P={kr.get('prompt_len')}): "
                         f"{kr['located_spans']} -> {kr.get('expected_all_turns_rows')} rows")
        for name, c in (kr.get("checks") or {}).items():
            lines.append(f"  check {name}: {'ok' if c.get('ok') else ('INCONCLUSIVE' if c.get('ok') is None else 'FAIL')}"
                         f" — {c.get('detail')}")
    return lines


def overall_exit_code(doc: Mapping[str, Any]) -> int:
    """0 = every case PASS with every check ok and ``complete`` true;
    2 = any FAIL / UNSTABLE_VECTORIZED / failed hard check (``ok`` False) /
    signature error / tree refusal — and, under ``batch_invariant`` 1, an
    UNSTABLE_REFERENCE too (composition-independent kernels make a
    non-repeating reference a real bug); 3 = no failure but an inconclusive
    verdict (UNSTABLE_REFERENCE at BI=0, INCONCLUSIVE_COSCHEDULE) or check
    (``ok`` None), an errored case, an uncovered inventory case, or
    ``complete`` false for any reason (the field is authoritative: a doc
    without it is incomplete)."""
    if doc.get("tree_error"):
        return 2
    bi1 = str(doc.get("batch_invariant")) == "1"
    worst = 0
    for kr in doc.get("cases") or []:
        if kr.get("error"):
            worst = max(worst, 3)
            continue
        for v in (kr.get("verdicts") or {}).values():
            if v.get("evidence") or v.get("verdict") == EVIDENCE:
                continue   # B>1 cross-server comparisons: recorded, never a verdict
            if v.get("verdict") in (FAIL, UNSTABLE_VEC):
                return 2
            if v.get("verdict") == UNSTABLE_REF and bi1:
                return 2   # composition-independent kernels: a real bug, not jitter
            if v.get("verdict") in INCONCLUSIVE_VERDICTS:
                worst = max(worst, 3)
        for c in (kr.get("checks") or {}).values():
            if c.get("ok") is None:
                worst = max(worst, 3)      # inconclusive, never a pass
            elif not c.get("ok"):
                return 2
    if doc.get("uncovered_cases"):
        worst = max(worst, 3)
    if doc.get("passthrough_error"):
        worst = max(worst, 3)
    if not doc.get("complete", False):
        worst = max(worst, 3)   # the field is authoritative, however it was derived
    return worst


# --------------------------------------------------------------------------- #
# The served-op inventory (derived, not a hand list)                           #
# --------------------------------------------------------------------------- #


def op_shape_class(op: Any) -> str:
    """The kernel-shape class of a served op: rank-1 vs rank-k LoRA are two
    GEMM shapes (a different kernel-selection risk each); the gated vector
    is one gemv; the affine subspace one rank class; the add and the
    last-token matmul one each."""
    from sisteer.capeval.serving import lens_adapters as la
    if op.op == la.OP_LORA_OPROJ:
        return "rank1" if op.lowrank_a.shape[0] == 1 else "rankK"
    if op.op in (la.OP_LOWRANK_RELU_GATED, la.OP_LOWRANK_LINEAR_GATED):
        return "rank1" if op.lowrank_a.shape[0] == 1 else "rankK"
    if op.op == la.OP_AFFINE_SUBSPACE:
        return "rank1" if op.rotate.shape[0] == 1 else "rankK"
    if op.op in (la.OP_RELU_GATED_VECTOR, la.OP_LINEAR_GATED_VECTOR, la.OP_RELU_GATED_VECTOR_PAIR):
        return "gemv"      # the paired write: two same-shape gemvs per row
    if op.op == la.OP_ADD_VECTOR:
        return "vec"
    if op.op == la.OP_PREFILL_LAST_MATMUL:
        return "dxd"
    return "other"


def case_id_of(kind: str, ops: Sequence[Any]) -> str:
    """``kind:op:shape:positions:skipbos{0,1}:phase`` of an adapted object;
    every op of one object must agree on the five (a mixed object refuses —
    no such kind exists in the vocabulary)."""
    sigs = sorted({(op.op, op_shape_class(op), op.positions, bool(op.skip_bos), op.phase)
                   for op in ops})
    if len(sigs) != 1:
        raise GateError(f"kind {kind!r}: its ops span several signatures {sigs}; the "
                        "inventory cannot file it under one case")
    op, shape, positions, skip_bos, phase = sigs[0]
    return f"{kind}:{op}:{shape}:{positions}:skipbos{int(skip_bos)}:{phase}"


def parse_case_id(case_id: str) -> Dict[str, Any]:
    kind, op, shape, positions, sb, phase = case_id.split(":")
    return {"kind": kind, "op": op, "shape": shape, "positions": positions,
            "skip_bos": sb == "skipbos1", "phase": phase}


def timing_for_case(case_id: str) -> bool:
    if case_id == MULTIGROUP:
        return True
    c = parse_case_id(case_id)
    return (c["kind"] in TIMING_KINDS and c["positions"] == "response"
            and not c["skip_bos"])


def servable_arms() -> Dict[str, str]:
    """arm -> object kind for every arm of the production Petri vocabulary
    that is lens-served: a registered lens adapter and not an unservable
    kind. (The serving path is the cell source's serving_path_policy —
    registry_adapter.SERVING_PATH_POLICIES; the lens standups' default,
    "lens", puts every object cell on the lens.)"""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    from sisteer.capeval.serving import lens_adapters as la
    kinds = set(la.registered_kinds())
    out: Dict[str, str] = {}
    for arm in tuple(dc.CLASS_A_METHODS) + tuple(dc.CLASS_B_IDS):
        kind = ARM_OBJECT_KINDS.get(arm)
        if kind is None or kind in la.UNSERVABLE_KIND_REASONS or kind not in kinds:
            continue
        out[arm] = kind
    return out


def resolved_spec_for(blob_or_sidecar: Mapping[str, Any], *, kind: str, arm: str,
                      safetensors: bool) -> Dict[str, Any]:
    """The resolved spec the registry adapter would build for this artifact
    (registry_adapter._resolve_torch_blob / _resolve_safetensors rules:
    the blob's JSON-safe non-tensor fields, or the sidecar's fields)."""
    from sisteer.capeval.petri.registry_adapter import _TENSOR, _json_safe
    spec: Dict[str, Any] = {}
    for key, value in blob_or_sidecar.items():
        if key == "kind" and not safetensors:
            continue
        safe = _json_safe(value, f"blob[{key!r}]")
        if safe is _TENSOR:
            continue
        spec[key] = safe
    spec.update(kind=kind, arm=arm)
    return spec


def adapt_object_dir(cell_dir: Path, *, arm: str, header_len: int) -> Tuple[str, Tuple[Any, ...]]:
    """Adapt the object in ``cell_dir`` through the REAL lens adapter of its
    arm's kind (the worker registry's path) and return ``(kind, ops)``; the
    blob's stored kind must equal ARM_OBJECT_KINDS[arm] (the rank-4 lora_all
    objects staged under the legacy ``lora_r1_all`` spelling are NOT rank-1
    ``lora`` objects — the kind check keeps the cases honest)."""
    import torch
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.methods import VECTOR_SAFETENSOR_ARMS
    kind = ARM_OBJECT_KINDS[arm]
    header = tuple(range(1000, 1000 + int(header_len)))
    if arm in VECTOR_SAFETENSOR_ARMS:
        art = cell_dir / "object.safetensors"
        side = cell_dir / "object.json"
        sidecar = json.loads(side.read_text())
        if sidecar.get("method") != kind:
            raise GateError(f"{cell_dir}: sidecar method {sidecar.get('method')!r} != {kind!r}")
        spec = resolved_spec_for(sidecar, kind=kind, arm=arm, safetensors=True)
        ops = la.get_adapter(kind)(spec, art.read_bytes(), side.read_bytes(), header_ids=header)
    else:
        art = cell_dir / "object.pt"
        raw = art.read_bytes()
        blob = torch.load(art, map_location="cpu", weights_only=True)
        if not isinstance(blob, Mapping) or blob.get("kind") != kind:
            raise GateError(f"{cell_dir}: blob kind {blob.get('kind') if isinstance(blob, Mapping) else '?'!r} "
                            f"!= {kind!r} (arm {arm})")
        spec = resolved_spec_for(blob, kind=kind, arm=arm, safetensors=False)
        ops = la.get_adapter(kind)(spec, raw, b"", header_ids=header)
    return kind, tuple(ops)


def format_variant_axes(kind: str) -> Dict[str, Tuple[Any, ...]]:
    """The value DOMAIN of every field a kind's save format carries a choice
    for — read from the classes' own vocabularies, not from what the gate
    thinks is supported: PositionMaskedFit's fit leg (psr / lora / lora_all /
    vector_all ``positions``) ranges over ``sisteer.hooks.POSITION_MODES``;
    MultiLayerVectorObject additionally stores ``skip_bos``; LoReFTObject
    persists its intrinsic policy as ``prompt`` or ``response``. The ADAPTER
    then decides which values are servable (``probe_variants``): a value it
    refuses is not a case. Kinds whose classes fix every field (antidote,
    alphasteer) have no axes."""
    from sisteer.hooks import POSITION_MODES
    modes = tuple(POSITION_MODES)
    # lora_subtraction persists the ONE fit leg its two parents share
    # (LoRASubtractionObject refuses parents whose legs differ), in the
    # same PositionMaskedFit vocabulary as the lora kinds it composes.
    if kind in ("psr", "lora", "lora_all", "lora_subtraction"):
        return {"positions": modes}
    if kind == "vector_all":
        return {"positions": modes, "skip_bos": (True, False)}
    if kind == "loreft":
        return {"positions": ("prompt", "response")}
    return {}


def probe_variants(arm: str, kind: str, header_len: int, root: Path
                   ) -> List[Tuple[str, Dict[str, Any]]]:
    """Every (case_id, variant) the real adapter ACCEPTS for ``arm`` over
    the cartesian product of its format's axes (a refused variant is
    recorded nowhere — the adapter is the authority)."""
    import itertools
    from sisteer.capeval.serving import lens_adapters as la
    axes = format_variant_axes(kind)
    names = sorted(axes)
    out: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()
    for values in itertools.product(*(axes[n] for n in names)) if names else [()]:
        variant = dict(zip(names, values))
        d = root / arm / "_".join(f"{n}-{v}" for n, v in variant.items()) if variant else root / arm
        try:
            cell, ddir = write_synthetic_object(arm, INVENTORY_DIMS, d, header_len, **variant)
            k2, ops = adapt_object_dir(ddir / cell, arm=arm, header_len=header_len)
            cid = case_id_of(k2, ops)
        except (la.AdapterError, GateError):
            continue  # not servable in that regime: not a case
        if cid in seen:
            continue
        seen.add(cid)
        out.append((cid, variant))
    return out


def served_inventory(header_len: int = 4) -> Dict[str, Dict[str, Any]]:
    """case_id -> {kind, arms, variants, sources}: the cases the gate MUST
    cover, derived by probing every servable arm's adapter over its save
    format's value domain (:func:`probe_variants`)."""
    inv: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as td:
        for arm, kind in servable_arms().items():
            for cid, variant in probe_variants(arm, kind, header_len, Path(td)):
                entry = inv.setdefault(cid, {"kind": kind, "arms": [], "variants": {},
                                             "sources": ["adapter-probe"]})
                if arm not in entry["arms"]:
                    entry["arms"].append(arm)
                entry["variants"][arm] = variant
    return inv


#: registered lens kinds that are internal machinery, never an object kind a
#: cell names: the inventory ignores them
INTERNAL_KINDS: Tuple[str, ...] = ("synthetic_vector", "__zero__", "stacked")

#: registered lens kinds with NO Petri arm (no cell id names them —
#: defense_cells.CLASS_A_METHODS / CLASS_B_IDS -> registry_adapter.
#: ARM_OBJECT_KINDS), so no standup can serve them: each is minted
#: synthetically (write_synthetic_kind) to derive its op class and
#: decode-capability, and reported as an UNSEEN class (module docstring,
#: KIND COVERAGE). A registered kind absent from BOTH the Petri vocabulary
#: and this table refuses in kind_coverage.
NON_PETRI_KINDS: Dict[str, str] = {
    "loreft_both": ("the earlier development branch's lineage (its jailbreak object "
                    "directories; a both-phase LoReFT object): the LoReFT affine subspace in BOTH "
                    "phases — its decode rows are grouped, the round-8 review finding"),
    "lowrank_gated": ("the earlier development branch's lineage (its low-rank gated object): "
                      "low-rank gated block edit in both phases — its decode rows are grouped"),
    "alphakl": ("the TrainTest-only alphakl ladder (no harness id — registry_adapter."
                "harness_cell_id refuses it): a linear/relu-gated psr variant in both phases"),
    "vector": ("the SteeringVector container arms (methods.VECTOR_SAFETENSOR_ARMS: vector, "
               "bidpo, did — `vector` is a TrainTest arm but not a harness Class A method): "
               "an elementwise add in both phases"),
    "static_r": ("the earlier development branch's StaticRObject (extraction/baseline_arms; no "
                 "harness arm maps to this kind — the two-SI CAA floor is the SEPARATE "
                 "caa_static_r kind, which since E45 H2 HAS an arm and is covered through it): "
                 "a fixed vector — prefill-only in its native modes, phase=both (grouped) when "
                 "served under the response-mode override"),
    "cosine_gated_r": ("the cosine-gated static_r sibling: prefill-only (positions=last), "
                       "never grouped"),
}


#: the variant axes that are RESOLVED-SPEC overrides rather than artifact
#: fields (``_served_positions`` reads ``resolved_spec["positions"]`` — the
#: cross-phase cell's serve-side mask, never stored in the blob)
KIND_SPEC_AXES: Tuple[str, ...] = ("positions",)


def kind_variant_axes(kind: str) -> Dict[str, Tuple[Any, ...]]:
    """The value DOMAIN of every choice a non-Petri kind's adapter reads —
    from the adapters' own code, the twin of :func:`format_variant_axes`
    (``None`` = the field omitted, the adapter's default): static_r's
    ``steer_positions`` (``prompt_last`` / ``prefill_all``, the artifact's fit
    mode) x the ``positions="response"`` spec override that serves it
    cross-phase (phase both); lowrank_gated's and alphakl's ``clamp_gate``
    (ReLU- vs linear-gated); the low-rank kinds' ``rank`` — their adapters
    accept every positive rank and the case schema (``op_shape_class``)
    files rank-1 and rank-k as DISTINCT kernel-shape classes, so both are
    probed; the SteeringVector container arm's
    ``positions`` (hooks.POSITION_MODES; non-served modes refuse) x
    ``skip_bos``; cosine_gated_r's response override (refused: derivation
    locality). The ADAPTER decides what serves (:func:`probe_kind_variants`)."""
    from sisteer.hooks import POSITION_MODES
    if kind == "static_r":
        return {"steer_positions": ("prompt_last", "prefill_all"), "positions": (None, "response")}
    if kind == "lowrank_gated":
        return {"clamp_gate": (True, False), "rank": (1, 4)}
    if kind == "loreft_both":
        return {"rank": (1, 4)}
    if kind == "alphakl":
        return {"clamp_gate": (True, False)}
    if kind == "vector":
        return {"positions": (None,) + tuple(POSITION_MODES), "skip_bos": (True, False)}
    if kind == "cosine_gated_r":
        return {"positions": (None, "response")}
    return {}


def write_synthetic_kind(kind: str, dims: Mapping[str, Any], out_dir: Path, header_len: int,
                         *, seed: int = 0, **variant: Any) -> Path:
    """A synthetic object for a registered kind WITHOUT a Petri arm, in the
    kind's own save format (the keys its adapter reads), with the model's
    dims — the kind-level twin of :func:`write_synthetic_object`. ``variant``
    sets the ARTIFACT-level fields of :func:`kind_variant_axes` (``None`` =
    omit; spec-level axes are :func:`adapt_kind_dir`'s ``spec_overrides``).
    Returns the object dir (``object.pt``, or the SteeringVector
    ``object.safetensors`` + ``object.json`` pair)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    d = int(dims["hidden_size"])
    L = int(dims["num_hidden_layers"])
    mid = L // 2
    layers = sorted({max(0, L // 3), mid})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "loreft_both":
        rank = int(variant.get("rank", 4))
        blob: Dict[str, Any] = {
            "kind": kind, "layers": layers, "rank": rank, "base_scale": 1.0,
            "Rs": {str(l): torch.linalg.qr(torch.randn(d, rank, generator=g))[0].t().contiguous()
                   for l in layers},                                   # [rank, d], orthonormal rows
            "Ws": {str(l): torch.randn(rank, d, generator=g) * 0.01 for l in layers},
            "bs": {str(l): torch.randn(rank, generator=g) * 0.01 for l in layers}}
    elif kind == "lowrank_gated":
        rank = int(variant.get("rank", 4))
        blob = {"kind": kind, "layers": layers, "rank": rank, "base_scale": 1.0,
                "clamp_gate": variant.get("clamp_gate", True),
                "As": {str(l): torch.randn(rank, d, generator=g) * (1.0 / math.sqrt(d)) for l in layers},
                "Bs": {str(l): torch.randn(d, rank, generator=g) * 0.01 for l in layers}}
    elif kind == "alphakl":
        blob = {"kind": kind, "layers": layers, "base_scale": 1.0,
                "clamp_gate": variant.get("clamp_gate", True),
                "ws": {str(l): torch.randn(d, generator=g) * (1.0 / math.sqrt(d)) for l in layers},
                "rs": {str(l): torch.randn(d, generator=g) * (0.02 / math.sqrt(d)) for l in layers}}
    elif kind == "static_r":
        blob = {"kind": kind, "base_scale": 1.0,
                "steer_positions": variant.get("steer_positions") or "prompt_last", "site": "post",
                "layers": layers,
                "rs": {str(l): torch.randn(d, generator=g) * (0.02 / math.sqrt(d)) for l in layers}}
    elif kind == "cosine_gated_r":
        r = torch.randn(d, generator=g)
        cond = torch.randn(d, generator=g)
        blob = {"kind": kind, "layer": mid, "r": (r / r.norm() * 0.5).contiguous(), "theta": 0.5,
                "d_cond": (cond / cond.norm()).contiguous(), "base_scale": 1.0}
    elif kind == "vector":
        from safetensors.torch import save_file
        vec = torch.randn(d, generator=g)
        vec = vec / vec.norm() * 0.5
        save_file({f"layer_{mid}": vec.contiguous()}, str(out_dir / "object.safetensors"))
        side: Dict[str, Any] = {"method": "diff_in_means", "layers": [mid],
                                "skip_bos": variant.get("skip_bos", True),
                                "metadata": {"synthetic": True}}
        if variant.get("positions") is not None:
            side["positions"] = variant["positions"]     # the sidecar IS the resolved spec here
        (out_dir / "object.json").write_text(json.dumps(side, indent=2))
        return out_dir
    else:
        raise GateError(f"no synthetic writer for kind {kind!r}")
    torch.save(blob, out_dir / "object.pt")
    return out_dir


def adapt_kind_dir(obj_dir: Path, *, kind: str, header_len: int,
                   spec_overrides: Optional[Mapping[str, Any]] = None) -> Tuple[Any, ...]:
    """Adapt a kind-level synthetic object (:func:`write_synthetic_kind`)
    through the REAL lens adapter of ``kind`` — :func:`adapt_object_dir`
    without an arm. ``spec_overrides`` are RESOLVED-SPEC keys
    (:data:`KIND_SPEC_AXES`: the cell's serve-side ``positions`` override,
    which the registry adapter carries in the resolved spec, never in the
    artifact)."""
    import torch
    from sisteer.capeval.serving import lens_adapters as la
    header = tuple(range(1000, 1000 + int(header_len)))
    st = Path(obj_dir) / "object.safetensors"
    if st.is_file():
        side = Path(obj_dir) / "object.json"
        spec = resolved_spec_for(json.loads(side.read_text()), kind=kind, arm=f"kind:{kind}",
                                 safetensors=True)
        spec.update({k: v for k, v in (spec_overrides or {}).items() if v is not None})
        return tuple(la.get_adapter(kind)(spec, st.read_bytes(), side.read_bytes(), header_ids=header))
    art = Path(obj_dir) / "object.pt"
    raw = art.read_bytes()
    blob = torch.load(art, map_location="cpu", weights_only=True)
    if not isinstance(blob, Mapping) or blob.get("kind") != kind:
        raise GateError(f"{obj_dir}: blob kind != {kind!r}")
    spec = resolved_spec_for(blob, kind=kind, arm=f"kind:{kind}", safetensors=False)
    spec.update({k: v for k, v in (spec_overrides or {}).items() if v is not None})
    return tuple(la.get_adapter(kind)(spec, raw, b"", header_ids=header))


def probe_kind_variants(kind: str, header_len: int, root: Path
                        ) -> List[Tuple[str, Dict[str, Any], Tuple[Any, ...]]]:
    """Every ``(case_id, variant, ops)`` the real adapter ACCEPTS for a
    non-Petri kind over the cartesian product of :func:`kind_variant_axes`
    (a refused variant is recorded nowhere — the adapter is the authority;
    one entry per distinct case signature) — :func:`probe_variants` for
    kinds without an arm. Artifact-level axes go into the mint, spec-level
    axes (:data:`KIND_SPEC_AXES`) into the resolved spec."""
    import itertools
    from sisteer.capeval.serving import lens_adapters as la
    axes = kind_variant_axes(kind)
    names = sorted(axes)
    out: List[Tuple[str, Dict[str, Any], Tuple[Any, ...]]] = []
    seen = set()
    for values in itertools.product(*(axes[n] for n in names)) if names else [()]:
        variant = dict(zip(names, values))
        blob_fields = {k: v for k, v in variant.items() if k not in KIND_SPEC_AXES}
        spec_fields = {k: v for k, v in variant.items() if k in KIND_SPEC_AXES}
        d = root / kind / ("_".join(f"{n}-{v}" for n, v in variant.items()) or "native")
        try:
            obj_dir = write_synthetic_kind(kind, INVENTORY_DIMS, d, header_len, **blob_fields)
            ops = adapt_kind_dir(obj_dir, kind=kind, header_len=header_len, spec_overrides=spec_fields)
            cid = case_id_of(kind, ops)
        except (la.AdapterError, GateError):
            continue  # not servable in that regime: not a case
        if cid in seen:
            continue
        seen.add(cid)
        out.append((cid, variant, ops))
    return out


def kind_coverage(header_len: int = 4) -> Dict[str, Dict[str, Any]]:
    """kind -> {servable, petri_arms, cases, variants, routing_by_case,
    decode_capable_cases, decode_capable, routing, reason} for EVERY
    registered lens kind that is neither internal nor documented-unservable
    (module docstring, KIND COVERAGE): a kind with a Petri arm is covered by
    the served inventory's cases of that kind; a kind without one must be in
    :data:`NON_PETRI_KINDS` and is probed over its whole adapter-supported
    variant domain (:func:`probe_kind_variants`) — EVERY distinct case
    signature is kept, and decode-capability / routing are computed from all
    of them (``lens_worker.ops_vectorize_in_decode``, the worker's own rule,
    per case). REFUSES a registered kind with neither an arm nor an entry,
    and any kind whose probe yields zero cases (a Petri-armed kind the served
    inventory does not file; a non-Petri mint the adapter refuses in every
    variant) — a kind without a case is a claim of exhaustiveness the gate
    cannot make."""
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lens_worker as lw
    arms_by_kind: Dict[str, List[str]] = {}
    for arm, kind in servable_arms().items():
        arms_by_kind.setdefault(kind, []).append(arm)
    inv = served_inventory(header_len)
    out: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as td:
        for kind in la.registered_kinds():
            if kind in INTERNAL_KINDS:
                continue
            if kind in la.UNSERVABLE_KIND_REASONS:
                out[kind] = {"servable": False, "petri_arms": [], "cases": [], "decode_capable": None,
                             "routing": None, "reason": "unservable: " + la.UNSERVABLE_KIND_REASONS[kind]}
                continue
            arms = arms_by_kind.get(kind, [])
            if arms:
                cases = sorted(cid for cid, e in inv.items() if e["kind"] == kind)
                if not cases:
                    raise GateError(
                        f"Petri-armed lens kind {kind!r} (arms {arms}) yields ZERO inventory cases — "
                        "its adapter refused every probed variant, so the gate cannot claim to have "
                        "filed its served signatures; refusing rather than reporting it covered")
                variants = {c: dict(inv[c].get("variants") or {}) for c in cases}
                rbc = {c: case_routing(c) for c in cases}
                out[kind] = {"servable": True, "petri_arms": arms, "cases": cases, "variants": variants,
                             "routing_by_case": rbc,
                             "decode_capable_cases": [c for c in cases if rbc[c] == ROUTING_VECTORIZED],
                             "decode_capable": any(r == ROUTING_VECTORIZED for r in rbc.values()),
                             "routing": sorted(set(rbc.values())), "reason": None}
                continue
            if kind not in NON_PETRI_KINDS:
                raise GateError(
                    f"registered lens kind {kind!r} has neither a Petri arm (registry_adapter."
                    "ARM_OBJECT_KINDS over defense_cells.CLASS_A_METHODS / CLASS_B_IDS) nor a "
                    "NON_PETRI_KINDS entry — a grouped op class the gate would not know it cannot "
                    "see; file it deliberately (a Petri arm, or an entry + write_synthetic_kind)")
            probed = probe_kind_variants(kind, header_len, Path(td))
            if not probed:
                raise GateError(
                    f"non-Petri lens kind {kind!r}: the adapter refused EVERY probed variant "
                    f"({kind_variant_axes(kind)}) — the synthetic mint does not produce this kind's "
                    "save format, or the adapter serves nothing; refusing to file it")
            cases = sorted(cid for cid, _, _ in probed)
            ops_by_case = {cid: ops for cid, _, ops in probed}
            rbc = {c: case_routing(c) for c in cases}
            for c in cases:  # the id's phase and the worker's rule must agree, per case
                if (rbc[c] == ROUTING_VECTORIZED) != bool(lw.ops_vectorize_in_decode(ops_by_case[c])):
                    raise GateError(f"kind {kind!r} case {c}: routing class {rbc[c]} disagrees with "
                                    "lens_worker.ops_vectorize_in_decode on its adapted ops")
            out[kind] = {"servable": False, "petri_arms": [], "cases": cases,
                         "variants": {cid: variant for cid, variant, _ in probed},
                         "routing_by_case": rbc,
                         "decode_capable_cases": [c for c in cases if rbc[c] == ROUTING_VECTORIZED],
                         "decode_capable": any(r == ROUTING_VECTORIZED for r in rbc.values()),
                         "routing": sorted(set(rbc.values())), "reason": NON_PETRI_KINDS[kind]}
    return out


def real_artifact_cases(ms: ModelServe, *, bank_dir: Optional[Path], extra_dirs: Sequence[Path],
                        candsi_dir: Optional[Path], header_len: int,
                        extra_origins: Optional[Mapping[Path, str]] = None
                        ) -> Dict[str, Dict[str, Any]]:
    """case_id -> {kind, arms, sources}: the adapted signatures of every REAL
    production artifact present FOR THIS MODEL (bank cells of servable arms
    under any SI, the extra dirs' objects whose fit-time record — or
    configured origin — names this family, the candsi_refit fits), so a
    served regime the probe did not produce still becomes a required case.
    An extra-dir object fit on another family is a ``__foreign__`` note,
    never a source (``extra_object_provenance``)."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.eval.defense_cell import object_path_for
    out: Dict[str, Dict[str, Any]] = {}
    arms = servable_arms()
    origins = dict(extra_origins or {})
    dirs = [(Path(bank_dir), False)] if bank_dir and Path(bank_dir).is_dir() else []
    dirs += [(Path(xd), True) for xd in extra_dirs if xd and Path(xd).is_dir()]
    for d, is_extra in dirs:
        for arm, kind in arms.items():
            cells = ([arm] if arm in dc.CLASS_B_IDS
                     else [f"{si}__{arm}" for si in BANK_SIS])
            for cell in cells:
                obj_path = object_path_for(d, cell)
                if obj_path is None or not Path(obj_path).is_file():
                    continue
                if is_extra:
                    label, reason, _prov = extra_object_provenance(
                        Path(obj_path).parent, Path(obj_path), d, ms, origins)
                    if label is None:
                        out.setdefault("__foreign__", {"kind": None, "arms": [], "sources": []})[
                            "sources"].append(f"{obj_path}: {reason}")
                        continue
                try:
                    k2, ops = adapt_object_dir(Path(obj_path).parent, arm=arm, header_len=header_len)
                    cid = case_id_of(k2, ops)
                except Exception as e:  # noqa: BLE001 — a real artifact that does not adapt is recorded
                    out.setdefault("__unadaptable__", {"kind": None, "arms": [], "sources": []})[
                        "sources"].append(f"{obj_path}: {type(e).__name__}: {e}")
                    continue
                entry = out.setdefault(cid, {"kind": kind, "arms": [], "sources": []})
                if arm not in entry["arms"]:
                    entry["arms"].append(arm)
                entry["sources"].append(str(obj_path))
    if candsi_dir and Path(candsi_dir).is_dir():
        for arm, kind in arms.items():
            fit = find_candsi_cell(candsi_dir, arm)
            if fit is None:
                continue
            try:
                check_fit_provenance(fit, ms)
                k2, ops = adapt_object_dir(fit, arm=arm, header_len=header_len)
                cid = case_id_of(k2, ops)
            except Exception as e:  # noqa: BLE001
                out.setdefault("__unadaptable__", {"kind": None, "arms": [], "sources": []})[
                    "sources"].append(f"{fit}: {type(e).__name__}: {e}")
                continue
            entry = out.setdefault(cid, {"kind": kind, "arms": [], "sources": []})
            if arm not in entry["arms"]:
                entry["arms"].append(arm)
            entry["sources"].append(str(fit / "object.pt"))
    return out


def plan_inventory(ms: ModelServe, *, bank_dir: Optional[Path], extra_dirs: Sequence[Path],
                   candsi_dir: Optional[Path], header_len: int,
                   extra_origins: Optional[Mapping[Path, str]] = None) -> Dict[str, Dict[str, Any]]:
    """The probe-derived inventory UNIONED with the real artifacts' adapted
    signatures for this model; :data:`INVENTORY_NOTES` (``__unadaptable__``:
    real artifacts that do not adapt; ``__foreign__``: another family's
    objects under an extra dir) are carried as notes, never as cases."""
    inv = served_inventory(header_len)
    real = real_artifact_cases(ms, bank_dir=bank_dir, extra_dirs=extra_dirs,
                               candsi_dir=candsi_dir, header_len=header_len,
                               extra_origins=extra_origins)
    notes = {k: real.pop(k) for k in INVENTORY_NOTES if k in real}
    for cid, entry in real.items():
        e = inv.setdefault(cid, {"kind": entry["kind"], "arms": [], "variants": {}, "sources": []})
        for arm in entry["arms"]:
            if arm not in e["arms"]:
                e["arms"].append(arm)
        e["sources"] = list(e.get("sources", [])) + entry["sources"]
    inv.update(notes)
    return inv


# --------------------------------------------------------------------------- #
# Objects: real bank -> extra dirs -> staged candsi -> synthetic               #
# --------------------------------------------------------------------------- #


def _cell_dir_has_object(d: Path) -> bool:
    return d.is_dir() and any(p.is_file() for p in d.glob("object.*"))


def find_bank_cell(bank_dir: Optional[Path], arm: str) -> Optional[Tuple[str, Path]]:
    """``(cell_id, bank)`` of the first real bank cell of ``arm``: Class A
    ``<si>__<arm>/`` (lora_r4 also under the legacy ``lora_r1_all``
    spelling), Class B bare ``<arm>/``."""
    if bank_dir is None or not Path(bank_dir).is_dir():
        return None
    from sisteer.capeval.petri import defense_cells as dc
    if arm in dc.CLASS_B_IDS:
        d = Path(bank_dir) / arm
        return (arm, Path(bank_dir)) if _cell_dir_has_object(d) else None
    names = [arm] + (["lora_r1_all"] if arm == "lora_r4" else [])
    for si in BANK_SIS:
        for name in names:
            if _cell_dir_has_object(Path(bank_dir) / f"{si}__{name}"):
                return f"{si}__{arm}", Path(bank_dir)
    return None


def find_candsi_cell(candsi_dir: Optional[Path], arm: str) -> Optional[Path]:
    if candsi_dir is None or not Path(candsi_dir).is_dir():
        return None
    for cand in CANDSI_IDS:
        d = Path(candsi_dir) / f"{cand}__{arm}"
        if (d / "object.pt").is_file() and (d / "COMPLETE").is_file():
            return d
    return None


def check_fit_provenance(fit_dir: Path, ms: ModelServe) -> Dict[str, Any]:
    prov_path = fit_dir / "fit_provenance.json"
    if not prov_path.is_file():
        raise GateError(f"{fit_dir}: no fit_provenance.json — cannot verify the object was "
                        "fit on this family's pinned model")
    prov = json.loads(prov_path.read_text())
    got_model, got_rev = prov.get("resolved_model_id"), prov.get("resolved_revision")
    if got_model != ms.hf or got_rev != ms.rev:
        raise GateError(f"{fit_dir}: fit_provenance pins {got_model}@{got_rev}, but the gate "
                        f"serves {ms.hf}@{ms.rev} — refusing to stage another model's object")
    return {"resolved_model_id": got_model, "resolved_revision": got_rev,
            "positions_at_fit": prov.get("positions_at_fit"), "git_sha": prov.get("git_sha"),
            "si_id": prov.get("si_id")}


def parse_extra_objects(specs: Optional[Sequence[str]], user: str
                        ) -> Tuple[List[Path], Dict[Path, str]]:
    """``--extra-objects-dir PATH[=<gate family key>]`` (default
    :data:`DEFAULT_EXTRA_OBJECTS`) -> ``(dirs in order, {dir: configured
    origin})``. A suffix that is not a gate family refuses at startup — a
    typo would otherwise become a nonexistent path, searched and skipped in
    silence."""
    items = list(specs) if specs else [f"{p}={k}" for p, k in DEFAULT_EXTRA_OBJECTS.items()]
    dirs: List[Path] = []
    origins: Dict[Path, str] = {}
    for spec in items:
        path, key = spec, None
        if "=" in spec:
            path, key = spec.rsplit("=", 1)
            if key not in MODELS:
                raise GateError(f"--extra-objects-dir {spec!r}: origin {key!r} is not a gate "
                                f"family {list(MODELS)}")
        d = Path(root_path(path, user, strict=bool(specs)))   # a default dir only searched: absent, it is skipped as before
        dirs.append(d)
        if key:
            origins[d] = key
    return dirs, origins


def extra_object_provenance(cell_dir: Path, obj_path: Path, extra_dir: Path, ms: ModelServe,
                            origins: Mapping[Path, str]
                            ) -> Tuple[Optional[str], str, Optional[Dict[str, Any]]]:
    """May the real object in ``cell_dir`` (under the extra bank dir
    ``extra_dir``) serve the family ``ms`` AT ALL — decided from the object's
    OWN fit-time record, never from whether its shapes happen to fit the
    model. Returns ``(label, reason, provenance)``: ``label`` is the
    ``objects`` report label when the object is this family's
    (``real(provenance: <key> via <record>)``), else None with ``reason``
    naming whose it is.

    The record is read through the sweep's authoritative reader
    (``sisteer.eval.sweep.read_stage3_fit_provenance``: fit_provenance.json,
    the training run's train_budget_meta.json, or the save sidecar's
    ``model_name``). A dir may also carry a CONFIGURED origin
    (:func:`parse_extra_objects`, :data:`DEFAULT_EXTRA_OBJECTS`): when both
    exist they must agree (a contradiction is a GateError — the mapping is
    wrong, not the object); when the object carries no record the origin is
    the only admissible evidence, and the label says so. An object with
    NEITHER is unattributable and refuses loudly, at resolution time: it is
    never served to any family. The lora blobs themselves carry no model
    name (kind/A/B/layer/positions only), so this is the whole check — the
    re-gate at aa2b8a1 (array 384162) searched the extra dirs unkeyed and
    served stage3-wave lane A's Llama-3.1-8B rank-1 LoRA (o_proj 4096x4096,
    layer 15) to the other four families: gemma4e4b / qwen35_4b AdapterError
    500s (read basis 4096 != o_proj in 2048; write basis 4096 != o_proj out
    2560), nemotron3nano4b's reference dead before readiness (layer 15 is a
    Mamba block: no o_proj), and on glm46vflash — whose o_proj IS 4096x4096 —
    a silently served PASS on a foreign object."""
    from sisteer.eval.sweep import read_stage3_fit_provenance
    extra_dir = Path(extra_dir)
    origin = origins.get(extra_dir)
    if origin is not None and origin not in MODELS:
        raise GateError(f"{extra_dir}: configured origin {origin!r} is not a gate family {list(MODELS)}")
    rec = read_stage3_fit_provenance(cell_dir, obj_path)
    if rec is None:
        if origin is None:
            raise GateError(
                f"{obj_path}: no fit-time record names its model (no fit_provenance.json / "
                f"train_budget_meta.json / save-sidecar model_name beside it) and {extra_dir} has no "
                f"configured origin — an unattributable object is never served to any family; pass "
                f"--extra-objects-dir {extra_dir}=<gate family key> if its origin is known")
        if origin != ms.key:
            return None, f"configured origin {origin}, not {ms.key} (the object carries no fit-time record)", None
        return (f"real(provenance: {origin}, configured origin; the object carries no fit-time record)",
                "", {"resolved_model_id": MODELS[origin].hf, "resolved_revision": None,
                     "chat_template_hash": None, "source": "configured_origin", "status": "unverified",
                     "positions_at_fit": None, "origin": origin})
    model_id, rev = rec["resolved_model_id"], rec.get("resolved_revision")
    fit_key = next((k for k, m in MODELS.items() if m.hf == model_id), None)
    if origin is not None and fit_key != origin:
        raise GateError(
            f"{obj_path}: its {rec['source']} record names {model_id}@{rev} "
            f"({fit_key or 'no gate family'}), but {extra_dir} is configured as {origin}'s objects — "
            f"the origin mapping contradicts the object's own record; fix the mapping")
    if model_id != ms.hf:
        return None, (f"fit on {model_id}@{rev} ({fit_key or 'no gate family'}) per {rec['source']}, "
                      f"not {ms.key}"), None
    if rev is not None and rev != ms.rev:
        return None, f"fit on {model_id}@{rev} per {rec['source']}, but the gate serves @{ms.rev}", None
    note = "" if rev is not None else ", revision unrecorded"
    return f"real(provenance: {ms.key} via {rec['source']}{note})", "", dict(rec, origin=ms.key)


def stage_cell(fit_dir: Path, arm: str, stage_root: Path) -> Tuple[str, Path]:
    cell = f"{STAGE_SI}__{arm}"
    d = stage_root / cell
    d.mkdir(parents=True, exist_ok=True)
    link = d / "object.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((fit_dir / "object.pt").resolve())
    return cell, stage_root


def oproj_shapes_from_skeleton(skeleton: Any) -> Dict[int, Tuple[int, int]]:
    """``{decoder layer index: (o_proj in_features, out_features)}`` read
    off the model's OWN module tree (a weight-free meta-device skeleton),
    for every block that carries an attention o_proj (``models.
    attention_oproj`` — the same probe the LoRA export uses; Mamba2 / MLP /
    linear-attention blocks contribute nothing). PER LAYER on purpose:
    gemma-4 E4B's o_proj is ``[2560, 2048]`` on its sliding-attention
    layers (8 heads x head_dim 256) and ``[2560, 4096]`` on its global
    layers (8 heads x global_head_dim 512) — read from the pinned
    snapshot's safetensors headers — so the config-derived
    ``num_attention_heads * head_dim`` (2048) that the synthetic mint used
    to apply to EVERY layer was wrong on every global layer, and job array
    382416's gemma reference standup refused the synthetic lora_all object
    at layer 5: "read basis dim 2048 != o_proj input dim 4096 — the object
    was fit on a different architecture". The module's ``in_features`` /
    ``out_features`` are authoritative (an HF Linear); a module exposing
    neither, and no 2-D ``weight``, refuses rather than guessing."""
    from sisteer.models import attention_oproj, get_decoder_layers
    out: Dict[int, Tuple[int, int]] = {}
    for i, block in enumerate(get_decoder_layers(skeleton)):
        mod = attention_oproj(block)
        if mod is None:
            continue
        in_f, out_f = getattr(mod, "in_features", None), getattr(mod, "out_features", None)
        w = getattr(mod, "weight", None)
        if (in_f is None or out_f is None) and w is not None and getattr(w, "dim", lambda: 0)() == 2:
            out_f, in_f = int(w.shape[0]), int(w.shape[1])      # nn.Linear layout [out, in]
        if not in_f or not out_f:
            raise GateError(f"decoder layer {i}: attention o_proj {type(mod).__name__} exposes neither "
                            "in_features/out_features nor a 2-D weight; refusing to guess its width")
        out[int(i)] = (int(in_f), int(out_f))
    return out


def _oproj_shapes(dims: Mapping[str, Any]) -> Dict[int, Tuple[int, int]]:
    """``dims["oproj_shapes"]`` normalized to ``{int layer: (int, int)}``
    (JSON round trips turn the keys into strings); ``{}`` when absent."""
    raw = dims.get("oproj_shapes") or {}
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def model_dims(hf: str, rev: str) -> Dict[str, Any]:
    """The served model's dims for the synthetic mint: hidden size and
    layer count from the (text) config, and the PER-LAYER attention o_proj
    shapes from a meta-device skeleton of the model's own class
    (:func:`oproj_shapes_from_skeleton`; the LoRA export's discipline —
    never a heads x head_dim formula, which gemma-4 breaks per layer)."""
    import torch
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf, revision=rev, local_files_only=True)
    text = getattr(cfg, "text_config", None) or cfg
    out: Dict[str, Any] = {"hidden_size": int(text.hidden_size),
                           "num_hidden_layers": int(text.num_hidden_layers)}
    try:
        from sisteer.models import auto_model_class
        cls = auto_model_class(hf, revision=rev, config=cfg)
        with torch.device("meta"):
            skeleton = cls.from_config(cfg)
        shapes = oproj_shapes_from_skeleton(skeleton)
        out["oproj_shapes"] = shapes
        out["oproj_layers"] = sorted(shapes)
    except Exception as e:  # noqa: BLE001 — recorded; lora synthetics refuse
        out["oproj_shapes"] = None
        out["oproj_layers"] = None
        out["oproj_layers_error"] = f"{type(e).__name__}: {e}"
    return out


def write_synthetic_object(arm: str, dims: Mapping[str, Any], out_dir: Path,
                           header_len: int, *, seed: int = 0,
                           positions: Optional[str] = None,
                           skip_bos: Optional[bool] = None,
                           si: str = STAGE_SI) -> Tuple[str, Path]:
    """A synthetic object for ``arm`` in its kind's OWN save format (the keys
    PSRObject / MultiLayerVectorObject / MultiLayerLoRAObject /
    RankOneLoRAObject / LoReFTObject / AlphaSteerObject .save write and the
    SteeringVector container antidote uses), with the model's real dims,
    small in magnitude. ``positions`` / ``skip_bos`` set the format's own
    fit-leg fields (defaults = each class's default: psr / LoRA "response"
    as the production fits, vector_all "all" + skip_bos True as
    MultiLayerVectorObject, loreft "prompt"); the ADAPTER decides whether
    the variant serves. Returns (cell_id, defenses_dir)."""
    import torch
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    kind = ARM_OBJECT_KINDS.get(arm)
    if kind is None:
        raise GateError(f"no object kind for arm {arm!r}")
    g = torch.Generator().manual_seed(seed)
    d = int(dims["hidden_size"])
    L = int(dims["num_hidden_layers"])
    cell = arm if arm in dc.CLASS_B_IDS else f"{si}__{arm}"
    cdir = out_dir / cell
    cdir.mkdir(parents=True, exist_ok=True)
    H = int(header_len)
    mid = L // 2

    def _fit_leg(default: str) -> Tuple[str, Optional[int]]:
        pos = default if positions is None else str(positions)
        return pos, (H if pos == "response" else None)

    if kind == "psr":
        layers = list(range(L))
        pos, rh = _fit_leg("response")
        blob = {"kind": "psr", "layers": layers, "train_direction": False,
                "positions": pos, "response_header_len": rh,
                "directions": {str(l): torch.randn(d, generator=g) * (0.02 / math.sqrt(d))
                               for l in layers},
                "gate_w": {str(l): torch.randn(d, generator=g) * (1.0 / math.sqrt(d))
                           for l in layers},
                "gate_b": {str(l): torch.tensor([0.5]) for l in layers}}
    elif kind == "vector_all":
        layers = list(range(L))
        pos, rh = _fit_leg("all")
        sb = True if skip_bos is None else bool(skip_bos)
        blob = {"kind": "vector_all",
                "vectors": {str(l): torch.randn(d, generator=g) * (0.02 / math.sqrt(d))
                            for l in layers},
                "positions": pos, "skip_bos": sb,
                "fit_positions": pos, "fit_response_header_len": rh}
    elif kind in ("lora_all", "lora"):
        # PER-LAYER o_proj shapes (oproj_shapes_from_skeleton): the read basis
        # A[l] spans THAT layer's o_proj input width and the write basis B[l]
        # its output width — gemma-4's widths differ between its sliding and
        # global attention layers, which is what broke the uniform mint (382416)
        shapes = _oproj_shapes(dims)
        if not shapes:
            raise GateError(f"synthetic {arm}: the attention o_proj shapes of this model could not "
                            f"be resolved ({dims.get('oproj_layers_error')}); refusing to target a "
                            "block without an attention o_proj")
        layers = sorted(shapes)
        pos, rh = _fit_leg("response")
        if kind == "lora_all":
            rank = 4
            blob = {"kind": "lora_all", "layers": layers, "rank": rank,
                    "A": {str(l): torch.randn(shapes[l][0], rank, generator=g)
                          * (1.0 / math.sqrt(shapes[l][0])) for l in layers},
                    "B": {str(l): torch.randn(shapes[l][1], rank, generator=g) * 0.01
                          for l in layers},
                    "positions": pos, "response_header_len": rh}
        else:
            l0 = layers[len(layers) // 2]
            d_in, d_out = shapes[l0]
            blob = {"kind": "lora", "layer": l0,
                    "A": torch.randn(d_in, generator=g) * (1.0 / math.sqrt(d_in)),
                    "B": torch.randn(d_out, generator=g) * 0.01,
                    "positions": pos, "response_header_len": rh}
    elif kind == "psr_paired_write":
        # the DERIVED paired write (extraction/paired_write.PSRPairedWriteObject): BOTH psr
        # parents' (w, b, d) stored bit-exact under `pos` / `neg`, all-position by construction
        # (the class refuses masked parents — the adapter refuses a `response` variant, so the
        # probe files the `all` case alone), with the builder's provenance naming both parents
        layers = list(range(L))
        pos, rh = _fit_leg("all")

        def _psr_side(dir_scale: float) -> Dict[str, Any]:
            return {"directions": {str(l): torch.randn(d, generator=g) * (dir_scale / math.sqrt(d))
                                   for l in layers},
                    "gate_w": {str(l): torch.randn(d, generator=g) * (1.0 / math.sqrt(d)) for l in layers},
                    "gate_b": {str(l): torch.tensor([0.5]) for l in layers}}
        blob = {"kind": "psr_paired_write", "layers": layers,
                "application_order": "shared_pre_write_state",
                "pos": _psr_side(0.02), "neg": _psr_side(0.015),
                "positions": pos, "response_header_len": rh,
                "provenance": {"kind": "psr_paired_write", "application_order": "shared_pre_write_state",
                               "parent_pos": {"object_sha256": "a" * 64},
                               "parent_neg": {"object_sha256": "b" * 64}}}
    elif kind == "lora_subtraction":
        # the DERIVED difference object (multilayer.LoRASubtractionObject):
        # the same per-layer o_proj shapes as lora_all, TWICE — both
        # parents' factors, which the artifact stores bit-exact. Same-shape
        # parents by construction here, as the class requires.
        shapes = _oproj_shapes(dims)
        if not shapes:
            raise GateError(f"synthetic {arm}: the attention o_proj shapes of this model could not "
                            f"be resolved ({dims.get('oproj_layers_error')}); refusing to target a "
                            "block without an attention o_proj")
        layers = sorted(shapes)
        pos, rh = _fit_leg("all")   # the sdfbase base-lane fits are all-position
        rank = 4
        blob = {"kind": "lora_subtraction", "layers": layers, "rank": rank,
                "positions": pos, "response_header_len": rh}
        for side in ("pos", "neg"):
            blob[f"A_{side}"] = {
                str(l): torch.randn(shapes[l][0], rank, generator=g)
                * (1.0 / math.sqrt(shapes[l][0])) for l in layers}
            blob[f"B_{side}"] = {
                str(l): torch.randn(shapes[l][1], rank, generator=g) * 0.01
                for l in layers}
    elif kind == "loreft":
        from torch import nn
        from torch.nn.utils import parametrizations
        layers = sorted({max(0, L // 3), mid})
        rank = 4
        rotate, learned = {}, {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for l in layers:
                rot = nn.Linear(d, rank, bias=False)
                parametrizations.orthogonal(rot)
                lin = nn.Linear(d, rank, bias=True)
                with torch.no_grad():
                    lin.weight.mul_(0.01)
                    lin.bias.mul_(0.01)
                rotate[str(l)] = {k: v.detach().clone() for k, v in rot.state_dict().items()}
                learned[str(l)] = {k: v.detach().clone() for k, v in lin.state_dict().items()}
        pos = "prompt" if positions is None else str(positions)
        blob = {"kind": "loreft", "layers": layers, "rank": rank, "positions": pos,
                "response_header_len": (H if pos == "response" else None),
                "rotate": rotate, "learned": learned}
    elif kind == "alphasteer":
        M = torch.randn(d, d, generator=g) * (0.01 / math.sqrt(d))
        blob = {"kind": "alphasteer", "Ms": {str(mid): M}, "layers": [mid],
                "base_scale": 1.0, "site": "post", "steer_positions": "prefill",
                "metadata": {"synthetic": True}}
    elif kind == "caa_static_r":
        # CaaStaticRObject.save's EXACT shape: {kind, rs, base_scale,
        # metadata} and nothing else — the class records neither
        # steer_positions nor site, because it has ONE deployment (prefill_all
        # at resid_pre) which its adapter pins. A writer that added those keys
        # would mint an artifact the fit path never produces.
        layers = sorted({max(0, L // 3), mid})
        blob = {"kind": "caa_static_r", "base_scale": 0.05,
                "rs": {str(l): torch.randn(d, generator=g) * (0.02 / math.sqrt(d))
                       for l in layers},
                "metadata": {"synthetic": True}}
    elif kind == "antidote":
        from safetensors.torch import save_file
        vec = torch.randn(d, generator=g)
        vec = vec / vec.norm() * 0.5
        save_file({f"layer_{mid}": vec.contiguous()}, str(cdir / "object.safetensors"))
        (cdir / "object.json").write_text(json.dumps(
            {"method": "antidote", "model_name": "synthetic", "instruction_id": "synthetic",
             "layers": [mid], "norms": {str(mid): float(vec.norm())},
             "metadata": {"base_alpha": 1.0, "synthetic": True}}, indent=2))
        return cell, out_dir
    else:
        raise GateError(f"no synthetic writer for kind {kind!r} (arm {arm})")
    torch.save(blob, cdir / "object.pt")
    return cell, out_dir


def header_len_for(hf: str, rev: str) -> int:
    from transformers import AutoTokenizer
    from sisteer.chat_markers import generation_header_ids
    from sisteer.models import pin_reasoning_template_off
    tok = AutoTokenizer.from_pretrained(hf, revision=rev)
    pin_reasoning_template_off(tok)
    return len(generation_header_ids(tok))


def resolve_case_object(ms: ModelServe, case_id: str, arms: Sequence[str], *,
                        bank_dir: Optional[Path], extra_dirs: Sequence[Path],
                        candsi_dir: Optional[Path], stage_root: Path, header_len: int,
                        dims_fn: Callable[[], Mapping[str, Any]],
                        allow_synthetic: bool,
                        extra_origins: Optional[Mapping[Path, str]] = None) -> Dict[str, Any]:
    """The object that covers ``case_id`` (module docstring ORDER). A real
    object is used only when it is THIS family's (the bank and candsi dirs
    are model-keyed; an extra dir's object must carry a fit-time record — or
    a configured origin — naming this family: ``extra_object_provenance``)
    AND its ADAPTED case id equals the required one; otherwise the case falls
    to a synthetic object of the case's first arm, and every rejected real
    candidate is recorded by name (``real_candidates_rejected``)."""
    from sisteer.eval.defense_cell import object_path_for
    origins = dict(extra_origins or {})
    seen: List[str] = []
    for arm in arms:
        candidates: List[Tuple[str, Path, str, Optional[Dict[str, Any]]]] = []
        found = find_bank_cell(bank_dir, arm)
        if found:
            candidates.append((found[0], found[1], "real", None))
        for xd in extra_dirs:
            f2 = find_bank_cell(xd, arm)
            if f2:
                obj2 = Path(object_path_for(f2[1], f2[0]))
                label, reason, prov2 = extra_object_provenance(obj2.parent, obj2, Path(xd), ms, origins)
                if label is None:
                    seen.append(f"real({Path(xd).name}) {f2[0]}: {reason}")
                    continue
                candidates.append((f2[0], f2[1], label, prov2))
        fit = find_candsi_cell(candsi_dir, arm)
        if fit is not None:
            prov = check_fit_provenance(fit, ms)
            cell, ddir = stage_cell(fit, arm, stage_root / "staged")
            candidates.append((cell, ddir, f"real(staged:{fit.parent.name}/{fit.name})", prov))
        for cell, ddir, label, prov in candidates:
            try:
                obj_path = object_path_for(ddir, cell)
                kind, ops = adapt_object_dir(Path(obj_path).parent, arm=arm, header_len=header_len)
                cid = case_id_of(kind, ops)
            except Exception as e:  # noqa: BLE001 — recorded, next candidate
                seen.append(f"{label} {cell}: {type(e).__name__}: {e}")
                continue
            if cid != case_id:
                seen.append(f"{label} {cell}: adapts to case {cid}, not {case_id}")
                continue
            out = {"arm": arm, "cell": cell, "defenses_dir": str(ddir), "objects": label,
                   "source": str(obj_path), "case_id": cid, "n_ops": len(ops)}
            if prov:
                out["fit_provenance"] = prov
            return out
    if not allow_synthetic:
        raise GateError(f"{ms.key}/{case_id}: no real object adapts to this case "
                        f"({'; '.join(seen) or 'none found'}) and synthetic objects are disabled")
    arm = arms[0]
    c = parse_case_id(case_id)
    kind0 = c["kind"]
    variant: Dict[str, Any] = {}
    axes = format_variant_axes(kind0)
    if "positions" in axes:
        # the CLASS field value that adapts to the case's served mode: the
        # fit-leg modes are the served modes for psr/lora/vector; loreft's
        # served "all"/prefill regime is its stored "prompt"
        variant["positions"] = "prompt" if kind0 == "loreft" else c["positions"]
    if "skip_bos" in axes:
        variant["skip_bos"] = c["skip_bos"]
    cell, ddir = write_synthetic_object(arm, dims_fn(), stage_root / "synthetic" / case_id.replace(":", "_"),
                                        header_len, **variant)
    from sisteer.eval.defense_cell import object_path_for
    obj_path = object_path_for(ddir, cell)
    kind, ops = adapt_object_dir(Path(obj_path).parent, arm=arm, header_len=header_len)
    cid = case_id_of(kind, ops)
    if cid != case_id:
        raise GateError(f"synthetic {arm} ({variant}) adapts to case {cid}, not the required {case_id}")
    return {"arm": arm, "cell": cell, "defenses_dir": str(ddir), "objects": "synthetic",
            "source": str(obj_path), "case_id": cid, "n_ops": len(ops), "variant": variant,
            "real_candidates_rejected": seen}


# --------------------------------------------------------------------------- #
# The multigroup case (pure parts)                                             #
# --------------------------------------------------------------------------- #


def unwrap_rpc(payload: Any) -> List[Any]:
    """The per-worker result list of a server-mode ``/collective_rpc``
    reply. The pinned 0.25.1 route wraps it as ``{"results": [<worker 0>,
    ...]}`` (the shape vllm_server's own readiness reads —
    ``payload.get("results")``); a bare list is taken as the worker list
    and a bare dict as ONE worker's result. Anything else refuses naming
    the shape. EVERY worker RPC read in this gate goes through here: the
    2026-09-04 run read the SelectionRecord / apply path / co-scheduling
    fields off the wrapper itself and saw None everywhere — "not
    attested" with 63 co-scheduled forwards in the log."""
    if isinstance(payload, Mapping) and "results" in payload:
        res = payload["results"]
        if not isinstance(res, list):
            raise GateError(f"collective_rpc payload 'results' is a {type(res).__name__}, not a "
                            f"per-worker list: {str(payload)[:300]}")
        return list(res)
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, Mapping):
        return [dict(payload)]
    raise GateError(f"unrecognized collective_rpc payload shape {type(payload).__name__}: "
                    f"{str(payload)[:300]}")


def single_worker(results: Sequence[Any], what: str) -> Any:
    """The one worker's result (TP=1 launches); several workers must agree."""
    if not results:
        raise GateError(f"{what}: the RPC returned no worker results")
    first = results[0]
    for r in results[1:]:
        if r != first:
            raise GateError(f"{what}: the workers disagree: {str(results)[:500]}")
    return first


def coschedule_attested(rec: Any) -> Optional[bool]:
    """Whether a CoScheduleRecord snapshot attests a SAME-RENDER pair of
    distinct nonzero cells in one forward; ``None`` when the record is
    absent / unreadable (the RPC does not exist on that tree)."""
    if not isinstance(rec, Mapping) or rec.get("rpc") == "absent":
        return None
    try:
        return (int(rec.get("n_forwards_with_same_render_pair") or 0) >= 1
                and int(rec.get("max_distinct_nonzero_groups_in_one_forward") or 0) >= 2)
    except (TypeError, ValueError):
        return None


def aggregate_coschedule(waves: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """ONE snapshot over every wave the multigroup case drove (accepted or
    re-driven): ``n_*`` counters summed, ``max_*`` counters maxed, plus which
    waves attested — so a single early attestation is never lost to a later
    wave that vLLM's admission happened to spread across steps. ``rpc:
    absent`` when no wave answered (the reference). Per-wave snapshots stay
    in ``coschedule_waves`` for provenance."""
    recs = [w.get("record") for w in waves if isinstance(w.get("record"), Mapping)
            and w["record"].get("rpc") != "absent"]
    if not recs:
        return {"rpc": "absent", "n_waves": len(waves)}
    out: Dict[str, Any] = {"n_waves": len(waves),
                           "attested_waves": [int(w.get("wave", i)) for i, w in enumerate(waves)
                                              if w.get("attested")]}
    for r in recs:
        for k, v in r.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if k.startswith("max_"):
                out[k] = max(int(out.get(k, 0)), int(v))
            elif k.startswith("n_"):
                out[k] = int(out.get(k, 0)) + int(v)
    out["first_record_error"] = next((r.get("first_record_error") for r in recs
                                      if r.get("first_record_error")), None)
    return out


def harness_map_of(endpoint: Mapping[str, Any], cells: Sequence[str]) -> Dict[str, str]:
    """TrainTest cell id -> harness id for ``cells``, from the standup
    helper's explicit ``cell_map``; a cell the standup did not serve is a
    GateError (never an identity guess — ``safety_refusal__psr_kl_all``
    serves as ``si_safety__psr_kl_all``)."""
    cmap = endpoint.get("cell_map") or {}
    out = {}
    for c in cells:
        if c == "none":
            out[c] = "none"
            continue
        if c not in cmap:
            raise GateError(f"cell {c!r} is not in the standup's cell_map {cmap} — the standup "
                            "did not serve it; refusing to guess its harness id")
        out[c] = str(cmap[c])
    return out


def gate_case_order(cases: Sequence[str], only_cases: Optional[Sequence[str]]) -> List[str]:
    """The run order: the multigroup case FIRST (a routing / config fault
    surfaces in minutes, not after thirteen single-op cases), then the
    inventory cases in inventory order; ``only_cases`` filters both."""
    order = [MULTIGROUP] + [c for c in cases if c != MULTIGROUP]
    if only_cases:
        order = [c for c in order if c in only_cases]
    return order


def multigroup_spec(B: int, n_prompts: Optional[int] = None) -> List[Dict[str, Any]]:
    """The interleaved batch: row i takes prompt ``i % n_prompts`` (default
    B // 2, so every prompt appears twice); a prompt's first row takes role
    ``MG_ROLES[j % 8]`` (j = the prompt index) and its second row the TWIN
    role (:data:`MG_TWIN_ROLE`) — a different role, and for every nonzero
    cell a different nonzero cell under the same render — so the batch
    interleaves passthrough, scale-0, plain and seeded rows of every cell,
    and every prompt yields a same-prompt pair the cross-cell distinctness
    check can compare. ``seeded`` rows are DECLARED prefill-seeded renders
    (module docstring)."""
    n = int(n_prompts or max(1, B // 2))
    out = []
    for i in range(int(B)):
        j, k = i % n, i // n
        role = MG_ROLES[j % len(MG_ROLES)]
        if k % 2 == 1:
            role = MG_TWIN_ROLE[role]
        base, _, seeded = role.partition("+")
        out.append({"i": i, "role": base, "seeded": seeded == "seeded", "prompt": j})
    return out


def resolve_mg_role(role: str, cells: Sequence[str]) -> str:
    """cellA/cellB/cellC -> a cell id (cellC falls back to cellA with two
    cells); passthrough / scale0 unchanged."""
    if role in ("passthrough", "scale0"):
        return role
    idx = {"cellA": 0, "cellB": 1, "cellC": 2}[role]
    if idx >= len(cells):
        idx = 0
    return cells[idx]


def multigroup_checks(case: Mapping[str, Any], spec: Sequence[Mapping[str, Any]],
                      cells: Sequence[str], *, reference_sha: str, base_sha: str,
                      have_oracle: bool, reduction_mode: str = DEFAULT_REDUCTION_MODE
                      ) -> Dict[str, Dict[str, Any]]:
    """The hard checks of the multigroup case: the attested apply paths (the
    candidate's and the shadow standup's name ``reduction_mode``), the
    attested reduction mode in the candidate's and the shadow standup's
    SelectionRecord, signatures, the worker's per-(cell, scale) evidence on
    both sides (every nonzero cell n_modifying > 0, the scale-0 twin 0, all
    matched), the CANDIDATE's co-scheduling attestation (any wave), the
    SHADOW standup's record (inside THE REDUCTION CONTRACT under
    ``reduction_mode``, a layer call that compared >= 2 distinct cells), the
    batch-invariance attestations, and the B=1 singleton controls
    (passthrough and scale-0 equal the oracle; every cell differs from the
    oracle and the cells differ pairwise). The B=32 row facts (control rows
    vs the oracle, per-cell bites, cross-cell distinctness) are EVIDENCE,
    recorded on the case, not checked here. ``cells`` are HARNESS ids."""
    mode = check_reduction_mode(reduction_mode)
    out: Dict[str, Dict[str, Any]] = {}
    ap = case.get("apply_path") or {}
    out["reference_apply_path"] = {
        "ok": apply_path_ok_reference(ap.get(REFERENCE), reference_sha=reference_sha,
                                      base_sha=base_sha),
        "detail": f"reference attested {ap.get(REFERENCE)!r}"}
    want_c, want_s = expected_apply_path(mode), expected_apply_path(mode, shadow=True)
    out["candidate_apply_path"] = {"ok": ap.get(CANDIDATE) == want_c,
                                   "detail": f"candidate attested {ap.get(CANDIDATE)!r}; the {mode} mode "
                                             f"expects {want_c!r}"}
    out["shadow_apply_path"] = {"ok": ap.get(SHADOW) == want_s,
                                "detail": f"shadow standup attested {ap.get(SHADOW)!r}; the {mode} mode "
                                          f"expects {want_s!r}"}
    for side in (CANDIDATE, SHADOW):
        mok, mdetail = reduction_mode_evidence_ok((case.get("steering_evidence") or {}).get(side), mode)
        out[f"reduction_mode_{side}"] = {"ok": mok, "detail": f"{side}: {mdetail}"}
    if LEGACY in ap:
        out["legacy_ab_apply_path"] = {"ok": ap.get(LEGACY) == APPLY_PATH_PER_ROW,
                                       "detail": f"legacy A/B attested {ap.get(LEGACY)!r}"}
    sig_err = case.get("signature_errors") or []
    out["signatures"] = {"ok": not sig_err,
                         "detail": "full evidence" if not sig_err else f"{len(sig_err)}: {sig_err[0]}"}
    nonzero = list(cells)
    for side in (REFERENCE, CANDIDATE):
        ev = ((case.get("steering_evidence") or {}).get(side) or {})
        by_cell = ev.get("by_cell") or {}
        ok_ev = bool(ev.get("all_observations_matched", False)) and bool(nonzero)
        detail = []
        for c in nonzero:
            n = int((by_cell.get(c) or {}).get("n_modifying") or 0)
            ok_ev = ok_ev and n > 0
            detail.append(f"{c}: {n}")
        z = by_cell.get("none__lens0") or {}
        if z:
            ok_ev = ok_ev and int(z.get("n_modifying") or 0) == 0
            detail.append(f"none__lens0: {z.get('n_modifying')} (must be 0)")
        else:
            ok_ev = False
            detail.append("none__lens0: no observations")
        out[f"evidence_{side}"] = {"ok": ok_ev,
                                   "detail": f"{side} matched={ev.get('all_observations_matched')}; "
                                             + ", ".join(detail)}
        c1 = (case.get("controls_B1") or {}).get(side) or {}
        out[f"controls_B1_{side}"] = {
            "ok": have_oracle and c1.get("controls_match") is True,
            "detail": c1.get("controls_detail") or (f"{side}: B=1 passthrough and scale-0 equal the oracle"
                                                    if c1.get("controls_match") else
                                                    f"{side}: no B=1 control comparison recorded")}
        out[f"cells_B1_{side}"] = {
            "ok": have_oracle and c1.get("cells_distinct") is True,
            "detail": c1.get("cells_detail") or (f"{side}: every cell's B=1 request differs from the "
                                                 "oracle and the cells differ pairwise"
                                                 if c1.get("cells_distinct") else
                                                 f"{side}: no B=1 per-cell comparison recorded")}
    cs = (case.get("coschedule") or {}).get(CANDIDATE)   # the AGGREGATE over every wave
    waves = (case.get("coschedule_waves") or {}).get(CANDIDATE) or []
    attested_waves = [w.get("wave", i) for i, w in enumerate(waves) if w.get("attested")]
    if case.get("coschedule_attested") is True or coschedule_attested(cs) is True:
        out["coschedule_candidate"] = {
            "ok": True, "detail": f"attested on wave(s) {attested_waves or '?'}: {cs}"}
    elif isinstance(cs, Mapping) and cs.get("rpc") == "absent":
        out["coschedule_candidate"] = {"ok": False, "detail": "the candidate worker has no "
                                                             "sisteer_coschedule_record RPC"}
    else:
        out["coschedule_candidate"] = {"ok": None, "detail": f"not attested on any of "
                                                            f"{case.get('coschedule_attempts')} "
                                                            f"wave attempt(s): {cs}"}
    ok, detail = shadow_multigroup_check(case.get("shadow"), reduction_mode=mode)
    out["shadow_compare"] = {"ok": ok, "detail": detail}
    bi = case.get("bi_attested") or {}
    for side in (REFERENCE, CANDIDATE, SHADOW):
        b = bi.get(side)
        out[f"batch_invariant_{side}"] = {
            "ok": bool(b.get("ok")) if isinstance(b, Mapping) else False,
            "detail": (b.get("detail") if isinstance(b, Mapping) else f"{side}: no attestation recorded")}
    return out


def multigroup_row_facts(spec: Sequence[Mapping[str, Any]], cells: Sequence[str],
                         mixed: Sequence[Mapping[str, Any]], oracle: Sequence[Mapping[str, Any]]
                         ) -> Dict[str, Any]:
    """Per side: control rows vs the oracle, steering-bites per cell, and
    same-prompt cross-cell distinctness — from the recorded signatures."""
    if len(mixed) != len(spec) or len(oracle) != len(spec):
        return {"control_rows_match": None, "control_rows_detail":
                f"batch sizes differ: spec {len(spec)}, batch {len(mixed)}, oracle {len(oracle)}"}
    ctrl = [r["i"] for r in spec if r["role"] in ("passthrough", "scale0")]
    ok, why = signatures_equal([mixed[i] for i in ctrl], [oracle[i] for i in ctrl])
    facts: Dict[str, Any] = {"control_rows_match": ok}
    if not ok:
        facts["control_rows_detail"] = why
    bites: Dict[str, bool] = {}
    for c in cells:
        idx = [r["i"] for r in spec if resolve_mg_role(r["role"], cells) == c]
        bites[c] = signatures_differ_somewhere([mixed[i] for i in idx], [oracle[i] for i in idx]) \
            if idx else False
    facts["bites_by_cell"] = bites
    distinct = True
    detail = []
    by_key: Dict[Tuple[int, bool], List[Tuple[str, int]]] = {}
    for r in spec:
        cell = resolve_mg_role(r["role"], cells)
        if cell in ("passthrough", "scale0"):
            continue
        by_key.setdefault((r["prompt"], r["seeded"]), []).append((cell, r["i"]))
    n_pairs = 0
    for key, rows in by_key.items():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                (ca, ia), (cb, ib) = rows[a], rows[b]
                if ca == cb:
                    continue
                n_pairs += 1
                if mixed[ia].get("logprobs") == mixed[ib].get("logprobs") \
                        and mixed[ia].get("token_ids") == mixed[ib].get("token_ids"):
                    distinct = False
                    detail.append(f"prompt {key[0]} seeded={key[1]}: rows {ia} ({ca}) and {ib} ({cb}) identical")
    facts["cross_cell_distinct"] = distinct if n_pairs else None
    facts["cross_cell_pairs"] = n_pairs
    if detail:
        facts["cross_cell_detail"] = "; ".join(detail[:3])
    elif not n_pairs:
        facts["cross_cell_detail"] = "no same-prompt pair under two different cells in the spec"
    return facts


# --------------------------------------------------------------------------- #
# Trees                                                                        #
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def tree_facts(repo: Path) -> Dict[str, Any]:
    repo = Path(repo)
    head = _git(repo, "rev-parse", "HEAD")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=str(repo),
                           capture_output=True).returncode != 0 if head else True
    return {"path": str(repo), "sha": head or None, "dirty": bool(dirty)}


def check_trees(candidate: Mapping[str, Any], reference: Mapping[str, Any], *,
                reference_sha: str) -> Optional[str]:
    """None when the two trees are usable as (candidate, reference); else the
    refusal: reference HEAD must equal ``reference_sha``, the candidate must
    differ from it, both must be clean checkouts."""
    if not reference.get("sha"):
        return f"reference tree {reference.get('path')} is not a git checkout"
    if not candidate.get("sha"):
        return f"candidate tree {candidate.get('path')} is not a git checkout"
    if reference["sha"] != reference_sha:
        return (f"reference tree is at {reference['sha']}, not the base commit "
                f"{reference_sha}")
    if candidate["sha"] == reference["sha"]:
        return f"candidate and reference are the SAME commit {candidate['sha']} — nothing to gate"
    if reference.get("dirty"):
        return f"reference tree {reference['path']} has modified tracked files"
    if candidate.get("dirty"):
        return f"candidate tree {candidate['path']} has modified tracked files"
    return None


# --------------------------------------------------------------------------- #
# Reporting (pure)                                                             #
# --------------------------------------------------------------------------- #


def _fmt_frac(x: Any) -> str:
    """A fraction for the report: ``1.5e-06`` style, ``n/a`` when unknown."""
    v = _opt_num(x)
    return "n/a" if v is None else format(v, ".1e")


def _fmt_g(x: Any) -> str:
    v = _opt_num(x)
    return "n/a" if v is None else format(v, "g")


def implied_step_ms(running_mean: Optional[float], gen_tok_s: Optional[float]) -> Optional[float]:
    if not running_mean or not gen_tok_s or gen_tok_s <= 0:
        return None
    return 1000.0 * float(running_mean) / float(gen_tok_s)


def shell_env_lines(ms: ModelServe, repo: Path) -> List[str]:
    rel = f"hub/models--{ms.hf.replace('/', '--')}/snapshots/{ms.rev}"
    return [f"GATE_MODEL_KEY={shlex.quote(ms.key)}", f"GATE_HF_ID={shlex.quote(ms.hf)}",
            f"GATE_HF_REV={shlex.quote(ms.rev)}",
            f"GATE_CONFIG={shlex.quote(str(Path(repo) / ms.cfg))}",
            f"GATE_BANK={shlex.quote(ms.bank)}", f"GATE_HF_SNAPSHOT_REL={shlex.quote(rel)}"]


def render_report(doc: Mapping[str, Any]) -> str:
    ms = doc["model"]
    t = doc.get("trees") or {}
    lines = [f"# lens_vectorize_gate — {ms['key']}  ({ms['hf']} @ {ms['rev'][:12]})", "",
             f"branch exercised: {ms['branch']}", "",
             f"reference tree: {t.get('reference', {}).get('path')} @ "
             f"{(t.get('reference', {}).get('sha') or '?')[:12]}   candidate tree: "
             f"{t.get('candidate', {}).get('path')} @ {(t.get('candidate', {}).get('sha') or '?')[:12]}",
             f"geometry: max_model_len {doc['geometry']['max_model_len']} x max_num_seqs "
             f"{doc['geometry']['max_num_seqs']}; node {doc.get('node')}; job {doc.get('job_id')}", ""]
    if doc.get("only_cases"):
        lines += [f"SUBSET RUN — cases {doc['only_cases']} only: complete=False by construction (every "
                  "case not run stays uncovered) and exit 3 whatever passed. NOT a certification of this "
                  "family; the full report stays at the family's root results dir.", ""]
    lines.append(f"batch invariance: VLLM_BATCH_INVARIANT={doc.get('batch_invariant')} on every standup "
                 "(production regime; attested by each worker's resolved state). The authoritative "
                 "identity check is the in-worker shadow compare (candidate_shadow standup); B=1 "
                 "cross-server verdicts are hard; every B>1 token comparison is evidence.")
    lines.append("prefix caching: OFF on every fidelity standup (the helper composes prefix_caching=False; "
                 "start_standup refuses otherwise), so the B=1 passthrough, scale-0 and oracle "
                 "requests are all cold prefills. The scale-0 `none__lens0` request is a lens route "
                 "and carries the proxy's request-unique cache_salt (cold in production too); against "
                 "a warm passthrough / oracle it read as 'scale0 differs' on the dense families in "
                 "job 382416. The ONE caching-ON standup per case is the shared-prefill case's "
                 f"({RESUMED}: resumed prefills for per-position ops — by the cell's policy an `always` "
                 "cell's warm steered request must be served and attested RESUMED by the worker, a "
                 "`conditional` (response-mode) cell's safe single-turn resume served and its unsafe "
                 "multi-turn resume CONTAINED per request, a `never` cell's refused; warm-vs-cold "
                 "closeness is evidence next to the passthrough floor, never a verdict — the "
                 "`resumed_prefill` check).")
    mode = doc.get("reduction_mode") or DEFAULT_REDUCTION_MODE
    lines.append(f"reduction mode: {mode} ({STACKED_ENV}="
                 f"{'1' if mode == REDUCTION_MODE_STACKED else '0'} set explicitly on every candidate-tree "
                 "standup, scrubbed on the base-commit reference; attested by each worker's apply path "
                 f"and {LENS_REDUCTION_MODE_KEY}). THE REDUCTION CONTRACT the shadow compare is held to "
                 f"— {REDUCTION_CONTRACT}. Every B=1 check (pure_B1, controls_B1, steering_bites_B1) "
                 "stays EXACT in either mode: a one-row group is one call.")
    lines.append("")
    if doc.get("tree_error"):
        lines += [f"TREE REFUSAL: {doc['tree_error']}", ""]
    pt = doc.get("passthrough_timing") or {}
    if pt:
        lines.append("passthrough floor (ms/step): " + ", ".join(
            f"B={b}: {v.get('step_ms'):.1f}" if v.get("step_ms") else f"B={b}: n/a"
            for b, v in sorted(pt.items(), key=lambda kv: int(kv[0]))))
        lines.append("")
    tb = doc.get("timing_batches") or []
    head = ("| case | arm | objects | shadow (compared/unequal[ frac row group ulps_edit"
            "[ gate_rel_sum flip_rel_sum relu_flips gate_cons exact_ne recon_ne]]) | " + " | ".join(
        f"B={b}" + ("" if int(b) == 1 else " (evidence)") for b in doc["batches"])
            + " | mixed (evidence) | checks | ref ms/step @" + "/".join(map(str, tb))
            + " | cand ms/step @" + "/".join(map(str, tb))
            + " | cand+shadow ms/step @" + "/".join(map(str, tb)) + " | modifying (ref/cand) |")
    lines += [head, "|" + "---|" * (10 + len(doc["batches"]))]
    for kr in doc.get("cases") or []:
        if kr.get("error"):
            lines.append(f"| {kr['case_id']} | {kr.get('arm', '?')} | {kr.get('objects', '?')} | "
                         f"ERROR: {kr['error'][:120]} |")
            continue
        vd = kr.get("verdicts") or {}

        def cell_of(v):
            if not v:
                return "-"
            if v.get("evidence"):
                return "equal" if v.get("equal") else f"differs({v.get('underlying')})"
            return v.get("verdict", "-")
        cells = [cell_of(vd.get(f"pure_B{b}")) for b in doc["batches"]]
        mb = kr.get("mixed_B") or max(doc["batches"])
        mixed = cell_of(vd.get(f"mixed_B{mb}") or vd.get(f"multigroup_B{mb}"))
        checks = kr.get("checks") or {}
        n_ok = sum(1 for c in checks.values() if c.get("ok"))
        n_inc = sum(1 for c in checks.values() if c.get("ok") is None)
        chk = f"{n_ok}/{len(checks)} ok" + (f", {n_inc} inconclusive" if n_inc else "")
        sh = kr.get("shadow") if isinstance(kr.get("shadow"), Mapping) else {}
        shadow_col = "-"
        if sh and sh.get("rpc") != "absent":
            shadow_col = f"{sh.get('n_groups_compared')}/{sh.get('n_unequal')}"
            if int(sh.get("n_unequal") or 0) > 0:
                tol = kr.get("reduction_tolerance") or reduction_tolerance(sh, reduction_mode=mode)
                shadow_col += (f" frac={_fmt_frac(tol.get('frac_unequal_elems'))} "
                               f"row={_fmt_frac(tol.get('max_frac_row'))} "
                               f"group={_fmt_frac(tol.get('max_frac_group'))} "
                               f"ulps_edit={_fmt_g(tol.get('max_ulps_edit'))}")
                if tol.get("gated_keys_unequal"):
                    # the gated rule's numbers, in the contract's unit (n/a = missing evidence),
                    # then the evidence read (relative deviation, smallest gate that differed)
                    shadow_col += (f" gate_rel_sum={fmt_gate_units(tol.get('max_gate_diff_rel_sum'))} "
                                   f"flip_rel_sum={fmt_gate_units(tol.get('max_gate_flip_rel_sum'))} "
                                   f"relu_flips={tol.get('n_gate_relu_flips')} "
                                   f"gate_cons={_fmt_g(tol.get('max_gate_edit_consistency_ulps'))} "
                                   f"exact_ne={tol.get('gate_exact_rows_unequal_elems')} "
                                   f"recon_ne={tol.get('gate_write_recon_mismatch_elems')}")
                    if tol.get("gate"):
                        shadow_col += (f" gate_rel={_fmt_g(tol['gate'].get('gate_max_rel_diff'))} "
                                       f"gate_min={_fmt_g(tol['gate'].get('gate_min_abs_at_diff'))}")

        def fmt(side):
            tt = (kr.get("timing") or {}).get(side, {})
            return "/".join(f"{tt.get(str(b), {}).get('step_ms'):.1f}"
                            if tt.get(str(b), {}).get("step_ms") else "n/a" for b in tb)
        ev = kr.get("steering_evidence") or {}
        lines.append(f"| {kr['case_id']} | {kr.get('arm')} | {kr.get('objects')} | {shadow_col} | "
                     + " | ".join(cells) + f" | {mixed} | {chk} | {fmt(REFERENCE)} | "
                     f"{fmt(CANDIDATE)} | {fmt(SHADOW)} | {ev.get(REFERENCE, {}).get('n_modifying')}/"
                     f"{ev.get(CANDIDATE, {}).get('n_modifying')} |")
    lines.append("")
    inv = {k: v for k, v in (doc.get("inventory") or {}).items() if k not in INVENTORY_NOTES}
    unc = doc.get("uncovered_cases") or []
    lines.append(f"inventory: {len(inv)} served-op cases + {MULTIGROUP}; uncovered {len(unc)}; "
                 f"complete={bool(doc.get('complete', False))}")
    for cid, e in inv.items():
        lines.append(f"- {cid}: arms {e.get('arms')}; sources {e.get('sources')}; routing "
                     f"{e.get('routing')}")
    notes = doc.get("inventory_notes") or {}
    for src in (notes.get("__foreign__") or {}).get("sources") or []:
        lines.append(f"- FOREIGN real artifact (another family's object; never served here): {src}")
    for src in (notes.get("__unadaptable__") or {}).get("sources") or []:
        lines.append(f"- UNADAPTABLE real artifact (present, does not adapt): {src}")
    kc = doc.get("kind_coverage") or {}
    if kc:
        unseen_sigs = [c for k, v in kc.items() if v.get("servable") is False
                       for c in (v.get("decode_capable_cases") or [])]
        unseen_kinds = sorted(k for k, v in kc.items() if v.get("servable") is False and v.get("decode_capable"))
        lines.append(f"kind coverage: {len(kc)} registered lens kinds; {len(unseen_sigs)} decode-capable "
                     f"signature(s) of {len(unseen_kinds)} kind(s) {unseen_kinds} have NO Petri-servable "
                     "cell — grouped op classes this GPU gate cannot see: their call shapes and exact "
                     "integer arithmetic are certified by the CPU call-shape / torch.equal tests only; "
                     "stacked reductions are the served default, so the reduction-op tolerance is "
                     "certified on hardware only for the cases this gate sees, and an unseen "
                     "signature's bf16 deviation is unmeasured:")
        for sig in unseen_sigs:
            lines.append(f"- UNSEEN grouped signature: {sig}")
        for k, v in kc.items():
            if v.get("servable") is False:
                lines.append(f"- UNSEEN kind {k}: cases {v.get('cases')}; decode-capable cases "
                             f"{v.get('decode_capable_cases')}; routing {v.get('routing')}; {v.get('reason')}")
    for kr in doc.get("cases") or []:
        sh = kr.get("shadow") if isinstance(kr.get("shadow"), Mapping) else None
        if sh and sh.get("rpc") != "absent" and kr.get("case_id") != MULTIGROUP:
            tol = kr.get("reduction_tolerance") or reduction_tolerance(sh, reduction_mode=mode)
            lines.append(f"- {kr['case_id']} shadow ({kr.get('routing')}): {sh.get('n_groups_compared')} "
                         f"compared / {sh.get('n_unequal')} unequal / {sh.get('n_unmodified_groups')} "
                         f"unmodified; counted per submitted wave size "
                         f"{compared_per_wave_size(kr.get('shadow_waves'), doc['batches'])}; group rows "
                         f"compared (kernel-shape evidence, never coverage): {rows_compared(sh)}; "
                         f"reduction-op elements unequal {tol.get('n_unequal_elems')}/"
                         f"{tol.get('n_elems_compared')} (frac {_fmt_frac(tol.get('frac_unequal_elems'))} "
                         f"<= {REDUCTION_FRAC_TOL:g}; max in one row {_fmt_frac(tol.get('max_frac_row'))} "
                         f"<= {REDUCTION_ROW_FRAC_TOL:g}, in one group {_fmt_frac(tol.get('max_frac_group'))} "
                         f"<= {REDUCTION_GROUP_FRAC_TOL:g}), max_ulps_edit {_fmt_g(tol.get('max_ulps_edit'))} "
                         f"(<= {REDUCTION_ULPS_EDIT_TOL:g} for linear ops, informational for gated ops)"
                         + (f"; gated ops: |delta gate|/sum|hw| {fmt_gate_units(tol.get('max_gate_diff_rel_sum'))} "
                            f"(<= {REDUCTION_GATE_C:g} x 2^-24), ReLU flips {tol.get('n_gate_relu_flips')} "
                            f"(fired side {fmt_gate_units(tol.get('max_gate_flip_rel_sum'))}), gate-edit identity "
                            f"{_fmt_g(tol.get('max_gate_edit_consistency_ulps'))} ulps "
                            f"(<= {REDUCTION_GATE_CONSISTENCY_ULPS_TOL:g}), exact-row unequal elements "
                            f"{tol.get('gate_exact_rows_unequal_elems')} (must be 0), write reconstruction mismatches "
                            f"{tol.get('gate_write_recon_mismatch_elems')} (must be 0; {tol.get('n_gate_groups_recon')} "
                            f"group(s) rebuilt), observed on {tol.get('n_gate_groups_observed')} group(s)"
                            if tol.get("gated_keys_unequal") else "")
                         + f"; flag disagreements "
                         f"{tol.get('n_flag_disagreements')}; contract: "
                         f"{'ok' if tol.get('ok') else ('n/a' if tol.get('ok') is None else 'FAIL')}"
                         + (f"; {gate_hist_brief(tol.get('gate'))} (informational)"
                            if tol.get("gated_keys_unequal") and gate_hist_brief(tol.get("gate")) else ""))
    for kr in doc.get("cases") or []:
        rp_ = kr.get("resumed_prefill")
        if rp_ is not None and kr.get("case_id") != MULTIGROUP:
            r_ok, r_detail = resumed_prefill_check(rp_)
            lines.append(f"- {kr['case_id']} shared prefill ({RESUMED}): "
                         f"{'ok' if r_ok else ('INCONCLUSIVE' if r_ok is None else 'FAIL')} — {r_detail}")
    for cid in unc:
        lines.append(f"- UNCOVERED: {cid} (arms {inv.get(cid, {}).get('arms')})")
    lines += declared_render_report_lines(doc)
    for kr in doc.get("cases") or []:
        if kr.get("case_id") == MULTIGROUP and kr.get("members"):
            lines.append(f"- {MULTIGROUP} cells: " + "; ".join(
                f"{m['cell']} ({m['objects']}, {m['case_id']})" for m in kr["members"])
                + f"; served as {kr.get('harness_cells')}")
            cs = (kr.get("coschedule") or {}).get(CANDIDATE)
            lines.append(f"- {MULTIGROUP} co-scheduling (candidate, {kr.get('coschedule_attempts')} "
                         f"wave attempt(s)): {cs}")
            if kr.get("bi0_probe"):
                lines.append(f"- {MULTIGROUP} BI=0 probe (evidence only): {kr['bi0_probe']}")
    lines.append("")
    lines.append(f"exit code: {doc.get('exit_code')}  (0 every case PASS + every hard check ok + "
                 "complete; 2 a FAIL / unstable candidate, a failed hard check — missing evidence, an "
                 "unreadable batch-invariance state, a routing disagreement, a missing pure_B1, an "
                 "unattested reduction mode, a shadow record outside the reduction contract — a "
                 "signature error or a tree refusal; 3 inconclusive / incomplete). Shadow-compare "
                 "contract — reduction ops: bounded bf16-kernel-variance tolerance (fraction of unequal "
                 f"elements <= {REDUCTION_FRAC_TOL:g} over the case, <= {REDUCTION_GROUP_FRAC_TOL:g} in any "
                 f"group, <= {REDUCTION_ROW_FRAC_TOL:g} in any row; linear ops: every unequal element <= "
                 f"{REDUCTION_ULPS_EDIT_TOL:g} bf16 ulps of the per-row edit beyond one output ulp; gated ops: "
                 f"|g_stacked - g_per_row| <= {REDUCTION_GATE_C:g} x 2^-24 x sum|h_i w_i| per gate element, a "
                 "ReLU flip only when the fired side is within that bound, gates observed on both applies and "
                 f"applied - ref = eff x (g_stacked - g_per_row) x v within {REDUCTION_GATE_CONSISTENCY_ULPS_TOL:g} "
                 "ulps, writes exact where the gates agree and each gated-vector side the bit-exact reconstruction "
                 "from its observed gate); elementwise + per-row mode: exact.")
    for kr in doc.get("cases") or []:
        for name, v in (kr.get("verdicts") or {}).items():
            if v.get("evidence"):
                if not v.get("equal"):
                    lines.append(f"- {kr['case_id']} {name}: evidence — {v.get('underlying')}: {v.get('detail')}")
            elif v.get("verdict") != PASS:
                lines.append(f"- {kr['case_id']} {name}: {v.get('verdict')} — {v.get('detail')}")
        for name, c in (kr.get("checks") or {}).items():
            if not c.get("ok"):
                lines.append(f"- {kr['case_id']} check {name}: "
                             f"{'INCONCLUSIVE' if c.get('ok') is None else 'FAIL'} — {c.get('detail')}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Runtime (GPU / cluster only; every import lazy)                              #
# --------------------------------------------------------------------------- #


def _say(msg: str) -> None:
    print(f"[gate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))
    tmp.replace(path)


def _nvidia_smi(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                              timeout=30).stdout
    except Exception:
        return None


def gpu_selector() -> Optional[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return (v.split(",")[0].strip() or None) if v else None


def wait_gpu_free(threshold_mib: int = 4096, timeout_s: float = 180.0) -> Optional[int]:
    sel = gpu_selector()
    args = ["--query-gpu=memory.used", "--format=csv,noheader,nounits"] + (["-i", sel] if sel else [])
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout_s:
        out = _nvidia_smi(*args)
        if out is None:
            return None
        try:
            last = int(float(out.strip().splitlines()[0]))
        except (ValueError, IndexError):
            return None
        if last < threshold_mib:
            return last
        time.sleep(3.0)
    _say(f"WARNING: GPU still holds {last} MiB after {timeout_s:.0f}s; continuing")
    return last


def parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    out: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)', ln)
        if not m:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        labels: Dict[str, str] = {}
        if m.group(2):
            for lm in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', m.group(2)):
                labels[lm.group(1)] = lm.group(2)
        out.setdefault(m.group(1), []).append((labels, val))
    return out


def metric_sum(parsed: Mapping[str, Sequence[Tuple[Mapping[str, str], float]]],
               *names: str) -> Optional[float]:
    for n in names:
        if n in parsed:
            return float(sum(v for _, v in parsed[n]))
    return None


class MetricsSampler:
    def __init__(self, host: str, port: int, api_key: Optional[str], period_s: float = 2.0):
        self.url = f"http://{host}:{port}/metrics"
        self.api_key = api_key
        self.period_s = period_s
        self.samples: List[Tuple[float, Dict[str, Any]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="metrics", daemon=True)

    def fetch(self) -> Optional[Dict[str, Any]]:
        import urllib.request
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        req = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:
            return None
        p = parse_prometheus(text)
        return {"running": metric_sum(p, "vllm:num_requests_running"),
                "gen_tokens": metric_sum(p, "vllm:generation_tokens_total"),
                "preemptions": metric_sum(p, "vllm:num_preemptions_total")}

    def _run(self) -> None:
        while not self._stop.is_set():
            s = self.fetch()
            if s is not None:
                self.samples.append((time.monotonic(), s))
            self._stop.wait(self.period_s)

    def start(self) -> "MetricsSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def window(self, w0: float, w1: float) -> Dict[str, Any]:
        win = [(t, s) for t, s in self.samples if w0 <= t <= w1]
        if len(win) < 2:
            return {"metrics_samples": len(win)}
        (t0, s0), (t1, s1) = win[0], win[-1]
        dt = max(1e-9, t1 - t0)
        running = [s["running"] for _, s in win if s.get("running") is not None]
        gen = (s1["gen_tokens"] - s0["gen_tokens"]
               if s0.get("gen_tokens") is not None and s1.get("gen_tokens") is not None else None)
        pre = (s1["preemptions"] - s0["preemptions"]
               if s0.get("preemptions") is not None and s1.get("preemptions") is not None else None)
        rm = (sum(running) / len(running)) if running else None
        tok_s = (gen / dt) if gen is not None else None
        return {"metrics_samples": len(win), "running_mean": rm, "engine_gen_tok_s": tok_s,
                "preemptions_delta": pre, "step_ms": implied_step_ms(rm, tok_s)}


# --- standups: one subprocess per tree ------------------------------------------


@dataclasses.dataclass
class Standup:
    tag: str
    side: str
    tree: Path
    out: Path
    proc: Any
    stop_file: Path
    endpoint: Dict[str, Any]
    api_key: Optional[str]

    @property
    def proxy_url(self) -> str:
        return self.endpoint["proxy_url"]

    @property
    def server_base_url(self) -> str:
        return self.endpoint["server_base_url"]

    @property
    def commit(self) -> Optional[str]:
        return self.endpoint.get("commit")


@dataclasses.dataclass
class GateContext:
    ms: ModelServe
    candidate: Path
    reference: Path
    reference_sha: str
    results_dir: Path
    max_model_len: int
    max_num_seqs: int
    gpu_mem_util: str
    ready_timeout_s: int
    batches: Tuple[int, ...]
    timing_batches: Tuple[int, ...]
    max_tokens: int
    repeats: int
    top_logprobs: int
    ignore_eos: bool
    warmup_s: float
    measure_s: float
    manifest: str
    legacy_ab: bool
    batch_invariant: int = 1
    bi0_probe: bool = False
    #: the REDUCTION MODE every candidate-tree standup is set to and judged
    #: under (module docstring, THE REDUCTION CONTRACT)
    reduction_mode: str = DEFAULT_REDUCTION_MODE
    #: whether every single-op case also runs THE SHARED-PREFILL CASE (module
    #: docstring) on a caching-ON candidate standup; ``--no-resumed-prefill``
    #: turns it off and leaves the ``resumed_prefill`` check INCONCLUSIVE
    resumed_prefill: bool = True


def standup_env(base: Mapping[str, str], tree: Path, *, batch_invariant: int,
                env_extra: Optional[Mapping[str, str]] = None,
                reduction_mode: Optional[str] = None) -> Dict[str, str]:
    """The helper's environment: PYTHONPATH naming ONLY ``tree`` (never the
    inherited one — it names the candidate), the per-row, shadow and stacked
    switches removed unless set below or by ``env_extra`` (the shadow standup
    sets ``SISTEER_LENS_SHADOW_COMPARE=1``, the legacy A/B ``SISTEER_LENS_PER_
    ROW_APPLY=1``), ``VLLM_BATCH_INVARIANT`` set to ``batch_invariant`` (0 =
    removed, the production default of setting nothing — a "0" would be an
    explicit value the launcher never composes; 1 = the opt-in experiment),
    and — when ``reduction_mode`` is given — :data:`STACKED_ENV` set
    EXPLICITLY to ``"1"`` (stacked) or ``"0"`` (per_row), never inherited, so
    the standup's arithmetic is the one the gate judges it under. ``None``
    (the reference standup on the base commit, which does not read the
    switch) leaves it scrubbed. ``env_extra`` is applied last."""
    env = dict(base)
    env.pop(PER_ROW_ENV, None)
    env.pop(SHADOW_ENV, None)
    env.pop(STACKED_ENV, None)
    env.pop(BATCH_INVARIANT_ENV, None)
    env["PYTHONPATH"] = f"{Path(tree) / 'src'}{os.pathsep}{Path(tree) / 'scripts'}"
    if int(batch_invariant) == 1:
        env[BATCH_INVARIANT_ENV] = "1"
    if reduction_mode is not None:
        env[STACKED_ENV] = "1" if check_reduction_mode(reduction_mode) == REDUCTION_MODE_STACKED else "0"
    env.update(env_extra or {})
    return env


def batch_invariant_ok(reported: Any, requested: int) -> bool:
    """Whether the helper's recorded ``VLLM_BATCH_INVARIANT`` matches the
    requested regime: "1" for 1; unset or "0" for 0."""
    if int(requested) == 1:
        return str(reported) == "1"
    return reported in (None, "", "0", 0)


def prefix_caching_ok(endpoint: Any) -> Tuple[bool, str]:
    """Whether a standup runs with prefix caching OFF (module docstring,
    PREFIX CACHING): the helper's COMPOSED state must be False AND the
    launcher's RESOLVED state (``prefix_caching_attested``, read by the
    helper from the attestation's ``scheduler_geometry.expected`` and every
    worker report) must be the boolean False with no read error
    (``prefix_caching_attest_error``). None, a string, a missing field or
    an error is a refusal — never a pass by silence (the helper used to
    swallow every read error into None and this accepted it). A composed
    True / None is the 382416 regime: the B=1 controls would compare a
    warm passthrough with the cold, salted scale-0 request."""
    if not isinstance(endpoint, Mapping):
        return False, "no endpoint record to read the cache state from"
    composed = endpoint.get("prefix_caching")
    attested = endpoint.get("prefix_caching_attested")
    err = endpoint.get("prefix_caching_attest_error")
    if composed is not False:
        return False, (f"the standup composed prefix_caching={composed!r}; the gate requires OFF so "
                       "the B=1 passthrough / scale-0 / oracle requests are all cold prefills (382416: "
                       "a warm passthrough vs the salted, cold scale-0 request read as 'scale0 differs')")
    if err:
        return False, (f"the launcher's resolved cache state could not be read from the attestation "
                       f"({err}) — a standup whose cache state is unverifiable does not run")
    if attested is not False:
        return False, (f"the launcher attests enable_prefix_caching={attested!r}, not the boolean "
                       "False the composed OFF launch must resolve to on the launch AND every worker "
                       "report" + (" (no attested value at all)" if attested is None else
                                   " — the resolved cache state is not the composed one"))
    return True, "prefix caching OFF: composed False, attested False"


def prefix_caching_on_ok(endpoint: Any) -> Tuple[bool, str]:
    """The shared-prefill standup's twin of :func:`prefix_caching_ok`: prefix
    caching must be ON — the helper composed the per-model default
    (``prefix_caching_requested == "default"``) which RESOLVED to True, and
    the launcher attests ``enable_prefix_caching`` True with no read error.
    A resolved False (a hybrid-recurrent family, where caching is
    structurally OFF) is a refusal here — the gate skips the case on those
    families BEFORE standing up (``resumed_case_applicable``), so reaching
    this means the applicability read and the launcher disagree."""
    if not isinstance(endpoint, Mapping):
        return False, "no endpoint record to read the cache state from"
    requested = endpoint.get("prefix_caching_requested")
    composed = endpoint.get("prefix_caching")
    attested = endpoint.get("prefix_caching_attested")
    err = endpoint.get("prefix_caching_attest_error")
    if requested != PREFIX_CACHING_DEFAULT:
        return False, (f"the standup was asked for prefix_caching={requested!r}; the shared-prefill case "
                       f"needs the helper's {PREFIX_CACHING_DEFAULT!r} regime (caching ON, resolved per model)")
    if composed is not True:
        return False, (f"the standup composed prefix_caching={composed!r}; the shared-prefill case needs "
                       "caching ON — on a hybrid-recurrent family the per-model default resolves OFF and "
                       "the case is not applicable (skipped before standing up), so a False here is a "
                       "disagreement between the applicability read and the launcher")
    if err:
        return False, (f"the launcher's resolved cache state could not be read from the attestation "
                       f"({err}) — a standup whose cache state is unverifiable does not run")
    if attested is not True:
        return False, (f"the launcher attests enable_prefix_caching={attested!r}, not the boolean True "
                       "the composed ON launch must resolve to on the launch AND every worker report")
    return True, "prefix caching ON: requested default, composed True, attested True"


def start_standup(ctx: GateContext, *, side: str, tree: Path, expect_sha: str, cell: Any,
                  defenses_dir: Path, tag: str, env_extra: Optional[Mapping[str, str]] = None,
                  batch_invariant: Optional[int] = None,
                  reduction_mode: Optional[str] = None,
                  prefix_caching: str = PREFIX_CACHING_OFF,
                  lens_salt_policy: Optional[str] = None,
                  declared_render: bool = False) -> Standup:
    """Spawn scripts/lens_gate_standup.py (from the CANDIDATE tree — it imports
    nothing from its own directory) with PYTHONPATH naming ONLY ``tree``, so
    the server, worker extension, adapters and registry all come from that
    tree; wait for endpoint.json + server_auth.json. ``reduction_mode`` sets
    :data:`STACKED_ENV` explicitly (``standup_env``); None scrubs it (the
    base-commit reference, or a caller that sets it through ``env_extra``).
    ``prefix_caching`` is the helper's regime: OFF (every fidelity standup,
    held by :func:`prefix_caching_ok`) or the per-model DEFAULT (the
    shared-prefill standup, held to ON by :func:`prefix_caching_on_ok`).
    ``declared_render`` (the declared-render cases' standups only) tells the
    helper the artifact digest binding and its publication by the proxy are
    REQUIRED, so a served tree that drops either refuses before any launch
    (S-G2); every other standup's argv is byte-identical to before."""
    if prefix_caching not in (PREFIX_CACHING_OFF, PREFIX_CACHING_DEFAULT):
        raise GateError(f"unknown prefix-caching regime {prefix_caching!r} (expected "
                        f"{PREFIX_CACHING_OFF!r} or {PREFIX_CACHING_DEFAULT!r})")
    out = ctx.results_dir / "standups" / f"{tag}__job{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    bi = ctx.batch_invariant if batch_invariant is None else int(batch_invariant)
    env = standup_env(os.environ, tree, batch_invariant=bi, env_extra=env_extra,
                      reduction_mode=reduction_mode)
    helper = REPO / "scripts" / "lens_gate_standup.py"
    cells = [cell] if isinstance(cell, str) else list(cell)
    cmd = [sys.executable, str(helper), "--repo-dir", str(tree), "--expect-sha", expect_sha,
           "--config", ctx.ms.cfg]
    for c in cells:
        cmd += ["--cell", c]
    cmd += ["--defenses-dir", str(defenses_dir),
           "--out", str(out), "--stop-file", str(stop_file),
           "--max-model-len", str(ctx.max_model_len), "--max-num-seqs", str(ctx.max_num_seqs),
           "--gpu-mem-util", str(ctx.gpu_mem_util), "--ready-timeout-s", str(ctx.ready_timeout_s),
           "--manifest", ctx.manifest, "--prefix-caching", prefix_caching]
    if lens_salt_policy is not None:
        # THE DECLARED-RENDER CASES' knob: passed only when a
        # case asks, so every other standup's argv is byte-identical to before
        if lens_salt_policy not in (LENS_SALT_POLICY_ATTESTED, LENS_SALT_POLICY_REQUEST_UNIQUE_ALL):
            raise GateError(f"unknown lens salt policy {lens_salt_policy!r} (expected "
                            f"{LENS_SALT_POLICY_ATTESTED!r} or {LENS_SALT_POLICY_REQUEST_UNIQUE_ALL!r})")
        cmd += ["--lens-salt-policy", lens_salt_policy]
    if declared_render:
        cmd += ["--declared-render"]
    log = open(out / "standup.log", "ab")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(tree))
    ep_path, auth_path = out / "endpoint.json", out / "server_auth.json"
    deadline = t0 + ctx.ready_timeout_s + 600
    while time.monotonic() < deadline:
        if ep_path.is_file() and auth_path.is_file():
            break
        if proc.poll() is not None:
            tail = (out / "standup.log").read_text(errors="replace")[-3000:]
            raise GateError(f"standup {tag} ({side}) exited {proc.returncode} before readiness:\n{tail}")
        time.sleep(2)
    else:
        proc.terminate()
        raise GateError(f"standup {tag} ({side}) did not become ready in time")
    endpoint = json.loads(ep_path.read_text())
    api_key = json.loads(auth_path.read_text()).get("api_key")
    if endpoint.get("commit") != expect_sha:
        proc.terminate()
        raise GateError(f"standup {tag} ({side}) reports commit {endpoint.get('commit')}, "
                        f"expected {expect_sha}")
    served = endpoint.get("served_from", {}).get("sisteer", "")
    if str(tree.resolve() / "src") not in served:
        proc.terminate()
        raise GateError(f"standup {tag} ({side}) served sisteer from {served}, not {tree}/src")
    if not batch_invariant_ok(endpoint.get("batch_invariant"), bi):
        proc.terminate()
        raise GateError(f"standup {tag} ({side}) reports VLLM_BATCH_INVARIANT="
                        f"{endpoint.get('batch_invariant')!r}, expected regime {bi}")
    att_ok, att_detail = bi_attestation_ok(endpoint.get("batch_invariant_attested"), bi,
                                           allow_absent=(side == REFERENCE and bi == 0))
    if not att_ok:
        proc.terminate()
        raise GateError(f"standup {tag} ({side}): the worker's resolved batch-invariance state "
                        f"disagrees with the launcher's intent (BI={bi}): {att_detail}")
    endpoint["bi_attested"] = {"ok": att_ok, "detail": att_detail, "intent": bi,
                               "attested": endpoint.get("batch_invariant_attested")}
    pc_check = prefix_caching_on_ok if prefix_caching == PREFIX_CACHING_DEFAULT else prefix_caching_ok
    pc_ok, pc_detail = pc_check(endpoint)
    if not pc_ok:
        proc.terminate()
        raise GateError(f"standup {tag} ({side}): {pc_detail}")
    endpoint["prefix_caching_check"] = {"ok": pc_ok, "detail": pc_detail}
    rm_ok, rm_detail = reduction_mode_endpoint_ok(endpoint, reduction_mode)
    if not rm_ok:
        proc.terminate()
        raise GateError(f"standup {tag} ({side}): {rm_detail}")
    endpoint["reduction_mode_check"] = {"ok": rm_ok, "detail": rm_detail, "gate_mode": reduction_mode,
                                        "attested": endpoint.get(LENS_REDUCTION_MODE_KEY)}
    if lens_salt_policy is not None and endpoint.get(LENS_SALT_POLICY_KEY) != lens_salt_policy:
        # the helper refuses a non-default regime the served proxy cannot take;
        # here the endpoint must say the regime the gate asked for, or the case
        # would judge stamps minted under another
        proc.terminate()
        raise GateError(f"standup {tag} ({side}) was asked for {LENS_SALT_POLICY_KEY}={lens_salt_policy!r} "
                        f"but its endpoint records {endpoint.get(LENS_SALT_POLICY_KEY)!r} "
                        f"({endpoint.get(LENS_SALT_POLICY_KEY + '_error')!r}); refusing to judge stamps "
                        "minted under another regime")
    _say(f"standup {tag} ({side}) ready in {time.monotonic() - t0:.0f}s from {tree} @ "
         f"{expect_sha[:12]} (VLLM_BATCH_INVARIANT={endpoint.get('batch_invariant')!r}, "
         f"{STACKED_ENV}={env.get(STACKED_ENV)!r}): proxy "
         f"{endpoint['proxy_url']}, server {endpoint['server_base_url']}")
    return Standup(tag=tag, side=side, tree=tree, out=out, proc=proc, stop_file=stop_file,
                   endpoint=endpoint, api_key=api_key)


def stop_standup(su: Optional[Standup]) -> None:
    if su is None:
        return
    try:
        su.stop_file.write_text("stop\n")
        su.proc.wait(timeout=600)
    except Exception:
        try:
            su.proc.terminate()
            su.proc.wait(timeout=60)
        except Exception:
            try:
                su.proc.kill()
            except Exception:
                pass
    wait_gpu_free()


class RpcError(GateError):
    def __init__(self, status: Any, detail: str):
        super().__init__(f"HTTP {status}: {detail[:500]}")
        self.status = status
        self.detail = detail


def rpc(su: Standup, method: str, args: Sequence[Any] = ()) -> Any:
    """POST one server-mode collective_rpc (the pinned 0.25.1 dev route the
    launcher uses for readiness), with this launch's token."""
    import urllib.error
    import urllib.request
    body = json.dumps({"method": method, "args": list(args)}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if su.api_key:
        headers["Authorization"] = f"Bearer {su.api_key}"
    req = urllib.request.Request(f"{su.server_base_url}/collective_rpc", data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        raise RpcError(e.code, detail) from e
    return json.loads(raw.decode("utf-8")) if raw else None


def apply_path_of(su: Standup) -> Any:
    """The standup's attested apply path (``sisteer_apply_path``), or the
    record ``{"rpc": "absent", ...}`` when the worker has no such method
    (the base commit) — recorded, and judged by ``case_checks``. Read
    through :func:`unwrap_rpc` (the per-worker ``results`` list)."""
    try:
        res = rpc(su, "sisteer_apply_path", [])
    except RpcError as e:
        return {"rpc": "absent", "status": e.status, "detail": e.detail[:300]}
    return single_worker(unwrap_rpc(res), "sisteer_apply_path")


def selection_evidence(su: Standup) -> Dict[str, Any]:
    """The worker's SelectionRecord since the last reset, totals and PER
    (requested cell, applied effective scale): modifying applications,
    applications, request count — the per-group evidence the multigroup
    case requires — plus the reduction mode the worker attests
    (``lens_reduction_mode``: the value when EVERY worker record carries one
    and they agree, else None, with the distinct values seen under
    ``lens_reduction_modes_seen``; the ``reduction_mode_*`` hard check reads
    it and a None fails it)."""
    recs = unwrap_rpc(rpc(su, "sisteer_selection_record", [True]))
    n_mod = n_obs = 0
    matched = True
    by_cell: Dict[str, Dict[str, Any]] = {}
    paths = {k: 0 for k in PATH_COUNTERS}
    paths_recorded = bool(recs)
    modes_recorded = bool(recs)
    modes: List[str] = []
    for r in recs:
        if not isinstance(r, Mapping):
            raise GateError(f"sisteer_selection_record: a worker returned a "
                            f"{type(r).__name__}, not a record: {str(r)[:200]}")
        n_obs += int(r.get("n_observations") or 0)
        matched = matched and bool(r.get("all_observations_matched", False))
        if all(k in r for k in PATH_COUNTERS):
            for k in PATH_COUNTERS:
                paths[k] += int(r.get(k) or 0)
        else:
            paths_recorded = False   # the base commit's record: no per-path counters
        if r.get(LENS_REDUCTION_MODE_KEY) is not None:
            if str(r[LENS_REDUCTION_MODE_KEY]) not in modes:
                modes.append(str(r[LENS_REDUCTION_MODE_KEY]))
        else:
            modes_recorded = False   # a record that predates the attestation
        for e in (r.get("requests") or {}).values():
            m = int(e.get("n_applications_modifying") or 0)
            n_mod += m
            cell = str(e.get("requested"))
            b = by_cell.setdefault(cell, {"n_modifying": 0, "n_applications": 0, "n_requests": 0,
                                          "scales": []})
            b["n_modifying"] += m
            b["n_applications"] += int(e.get("n_applications") or 0)
            b["n_requests"] += 1
            sc = e.get("applied_effective_scale")
            if sc not in b["scales"]:
                b["scales"].append(sc)
    out = {"n_observations": n_obs, "n_modifying": n_mod, "all_observations_matched": matched,
           "by_cell": by_cell, "paths_recorded": paths_recorded,
           LENS_REDUCTION_MODE_KEY: (modes[0] if modes_recorded and len(modes) == 1 else None),
           "lens_reduction_modes_seen": sorted(modes)}
    out.update(paths if paths_recorded else {k: None for k in PATH_COUNTERS})
    return out



def vmodels_of(su: Standup) -> Dict[str, str]:
    doc = json.loads(Path(su.endpoint["routing_table"]).read_text())
    return {str(r["cell_id"]): str(r["virtual_model"]) for r in doc["routes"]}


# --- requests -------------------------------------------------------------------


def tokenize_count(proxy_url: str, vmodel: str, messages, timeout_s: float = 120.0) -> int:
    import urllib.request
    body = json.dumps({"model": vmodel, "messages": messages,
                       "add_generation_prompt": True}).encode("utf-8")
    req = urllib.request.Request(f"{proxy_url.rstrip('/')}/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return int(json.loads(r.read().decode("utf-8", "replace"))["count"])


def fit_repeats(count_fn: Callable[[int], int], target_tokens: int, max_iter: int = 12
                ) -> Tuple[int, int]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    k = max(1, target_tokens // 30)
    c = count_fn(k)
    if c <= 0:
        raise RuntimeError("count_fn returned a non-positive token count")
    for _ in range(max_iter):
        if c <= target_tokens and count_fn(k + 1) > target_tokens:
            return k, c
        per = max(1.0, (c - count_fn(1)) / k) if k > 1 else max(1.0, float(c))
        k_new = max(1, int(round(k + (target_tokens - c) / per)))
        if k_new == k:
            k_new = k + (1 if c < target_tokens else -1)
        k = max(1, k_new)
        c = count_fn(k)
    while c > target_tokens and k > 1:
        k -= 1
        c = count_fn(k)
    return k, c


def build_prompts(proxy_url: str, vmodel: str, targets: Sequence[int]
                  ) -> List[Tuple[List[Dict[str, str]], int]]:
    cache: Dict[int, int] = {}

    def count_fn(k: int) -> int:
        if k not in cache:
            cache[k] = tokenize_count(proxy_url, vmodel, [{"role": "user", "content": FILLER * k}])
        return cache[k]
    out = []
    for i, t in enumerate(targets):
        k, c = fit_repeats(count_fn, int(t))
        out.append(([{"role": "user", "content": f"Passage {i + 1}. " + FILLER * k}], c))
    return out


async def _submit_batch(proxy_url: str, requests: Sequence[Tuple[str, Any, Mapping[str, Any]]], *,
                        max_tokens: int, ignore_eos: bool, top_logprobs: int) -> List[Dict[str, Any]]:
    import aiohttp
    base = proxy_url.rstrip("/")

    async def one(session, vmodel, messages, extras):
        payload = {"model": vmodel, "messages": messages, "max_tokens": int(max_tokens),
                   "temperature": 0.0, "seed": 0, "logprobs": True,
                   "top_logprobs": int(top_logprobs), "return_token_ids": True,
                   "ignore_eos": bool(ignore_eos), "stream": False}
        payload.update(extras or {})
        async with session.post(f"{base}/chat/completions", json=payload) as r:
            body = await r.json(content_type=None)
            if r.status != 200:
                raise GateError(f"HTTP {r.status} from the proxy: {str(body)[:500]}")
            return body
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        return list(await asyncio.gather(*(one(session, vm, msgs, ex) for vm, msgs, ex in requests)))


def seeded_request(messages: Sequence[Mapping[str, Any]], *, lens_route: bool
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """A DECLARED prefill-seeded render (proxy._validate_response_prefill's
    contract): the final message is the assistant seed being continued,
    ``continue_final_message`` true, ``add_generation_prompt`` false, and
    on a LENS route the ``petri_response_prefill`` declaration (a JSON
    bool); a passthrough route takes the same render WITHOUT the key (the
    proxy refuses the declaration there)."""
    msgs = list(messages) + [{"role": "assistant", "content": SEED_TEXT}]
    extras: Dict[str, Any] = {"continue_final_message": True, "add_generation_prompt": False}
    if lens_route:
        extras["petri_response_prefill"] = True
    return msgs, extras


def spec_requests(spec: Sequence[Mapping[str, Any]], prompts, cells: Sequence[str],
                  vm: Mapping[str, Optional[str]], *, all_passthrough: bool = False
                  ) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """The multigroup spec as (vmodel, messages, extras) rows — the same
    rows with every role passthrough for the oracle (seeded rows stay
    seeded, undeclared)."""
    out = []
    for r in spec:
        role = "passthrough" if all_passthrough else resolve_mg_role(r["role"], cells)
        v = vm.get(role)
        if not v:
            raise GateError(f"no virtual model for role {role!r} (routes: {vm})")
        msgs = prompts[r["prompt"]][0]
        extras: Dict[str, Any] = {}
        if r["seeded"]:
            msgs, extras = seeded_request(msgs, lens_route=(role != "passthrough"))
        out.append((v, msgs, extras))
    return out


def run_batch(proxy_url: str, requests, **kw) -> List[Dict[str, Any]]:
    bodies = asyncio.run(_submit_batch(proxy_url, requests, **kw))
    return [signature(b["choices"][0]) for b in bodies]


async def _closed_loop(proxy_url: str, vmodel: str, messages, *, B: int, max_tokens: int,
                       warmup_s: float, measure_s: float) -> Tuple[int, int, Tuple[float, float]]:
    import aiohttp
    base = proxy_url.rstrip("/")
    payload = {"model": vmodel, "messages": messages, "max_tokens": int(max_tokens),
               "ignore_eos": True, "temperature": 0.0, "stream": False}
    records: List[Tuple[float, bool]] = []
    stop = asyncio.Event()
    t0 = time.monotonic()
    w0, w1 = t0 + warmup_s, t0 + warmup_s + measure_s

    async def worker(session):
        while not stop.is_set():
            ok = False
            try:
                async with session.post(f"{base}/chat/completions", json=payload) as r:
                    await r.json(content_type=None)
                    ok = r.status == 200
            except Exception:  # noqa: BLE001
                ok = False
            te = time.monotonic()
            records.append((te, ok))
            if te >= w1:
                break
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(B)]
        await asyncio.sleep(warmup_s + measure_s)
        stop.set()
        _done, pending = await asyncio.wait(tasks, timeout=300)
        for t in pending:
            t.cancel()
    inwin = [ok for te, ok in records if w0 <= te <= w1]
    return sum(1 for ok in inwin if ok), sum(1 for ok in inwin if not ok), (w0, w1)


def measure_timing(ctx: GateContext, su: Standup, vmodel: str, messages) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    sampler = MetricsSampler(su.endpoint["server_host"], su.endpoint["server_port"], su.api_key).start()
    try:
        for B in ctx.timing_batches:
            if B > ctx.max_num_seqs:
                out[str(B)] = {"skipped": f"B={B} > max_num_seqs={ctx.max_num_seqs}"}
                continue
            n_ok, n_err, (w0, w1) = asyncio.run(_closed_loop(
                su.proxy_url, vmodel, messages, B=B, max_tokens=128,
                warmup_s=ctx.warmup_s, measure_s=ctx.measure_s))
            time.sleep(2.5)
            rec = sampler.window(w0, w1)
            rec.update({"B": B, "n_completed": n_ok, "n_errors": n_err})
            out[str(B)] = rec
            _say(f"    timing {su.side} {vmodel} B={B}: step {rec.get('step_ms')} ms, "
                 f"{rec.get('engine_gen_tok_s')} tok/s, running {rec.get('running_mean')}")
    finally:
        sampler.stop()
    return out


def _batch_requests(prompts, roles: Sequence[str], vm: Mapping[str, Optional[str]]
                    ) -> List[Tuple[str, Any, Dict[str, Any]]]:
    out = []
    for i, role in enumerate(roles):
        v = vm.get(role)
        if not v:
            raise GateError(f"no virtual model for role {role!r} (routes: {vm})")
        out.append((v, prompts[i][0], {}))
    return out


def drive_standup(ctx: GateContext, su: Standup, *, cell: str, prompts,
                  passthrough_only: bool = False, mg_spec: Optional[Sequence[Mapping[str, Any]]] = None,
                  timing: bool = True) -> Dict[str, Any]:
    vms = vmodels_of(su)
    hcell = harness_map_of(su.endpoint, [cell])[cell] if not passthrough_only else "none"
    vm = {"passthrough": vms.get("none"), "scale0": vms.get("none__lens0"),
          "steered": vms.get(hcell)}
    rec: Dict[str, Any] = {"apply_path": apply_path_of(su), "vmodels": vm, "batches": {},
                           "signature_errors": [], "commit": su.commit,
                           "served_from": su.endpoint.get("served_from"),
                           "harness_cell": hcell, "bi_attested": su.endpoint.get("bi_attested"),
                           "prefix_caching": su.endpoint.get("prefix_caching_check"),
                           "reduction_mode_endpoint": su.endpoint.get("reduction_mode_check")}
    B_max = max(ctx.batches)
    timing_prompt = prompts[min(4, len(prompts) - 1)][0]

    def _runs(roles):
        runs = []
        for _ in range(ctx.repeats):
            try:
                runs.append(run_batch(su.proxy_url, _batch_requests(prompts, roles, vm),
                                      max_tokens=ctx.max_tokens, ignore_eos=ctx.ignore_eos,
                                      top_logprobs=ctx.top_logprobs))
            except SignatureError as e:
                rec["signature_errors"].append(str(e))
                runs.append([])
        return runs

    if passthrough_only:
        rec["batches"][f"passthrough_B{B_max}"] = _runs(["passthrough"] * B_max)
        rec["batches"]["passthrough_B1"] = _runs(["passthrough"])       # the B=1 oracle (prompt 0 alone)
        if mg_spec is not None:
            runs = []
            for _ in range(ctx.repeats):
                try:
                    runs.append(run_batch(su.proxy_url,
                                          spec_requests(mg_spec, prompts, (), vm, all_passthrough=True),
                                          max_tokens=ctx.max_tokens, ignore_eos=ctx.ignore_eos,
                                          top_logprobs=ctx.top_logprobs))
                except SignatureError as e:
                    rec["signature_errors"].append(str(e))
                    runs.append([])
            rec["batches"]["multigroup_oracle"] = runs
        rec["timing"] = measure_timing(ctx, su, vm["passthrough"], timing_prompt)
        return rec
    n_mod = n_obs = 0
    matched = True
    try:
        rpc(su, "sisteer_selection_record", [True])   # reset
    except RpcError as e:
        rec["evidence_error"] = str(e)
    for B in ctx.batches:
        runs = _runs(["steered"] * B)
        rec["batches"][f"pure_B{B}"] = runs
        _say(f"    {su.side} pure B={B}: stable={stable(runs)[0]}")
    roles = mixed_roles(B_max)
    rec["batches"][f"mixed_B{B_max}"] = _runs(roles)
    rec["mixed_roles"] = roles
    rec["controls_B1"] = controls_B1(ctx, su, prompts, vm, ("passthrough", "scale0", "steered"))
    try:
        # the WHOLE selection_evidence record -- totals, by_cell AND the per-path
        # counters (paths_recorded / n_applications_* / n_modifying_*) the
        # routing_* hard check reads. Job 382416 summarized it down to three
        # totals here, so routing_candidate read "no per-path counters" on every
        # family while the shadow standup (drive_shadow, full record) passed.
        ev = selection_evidence(su)
    except RpcError as e:
        rec["evidence_error"] = str(e)
        ev = {"n_modifying": 0, "n_observations": 0, "all_observations_matched": False,
              "by_cell": {}, "paths_recorded": False, **{k: None for k in PATH_COUNTERS}}
    rec["evidence"] = ev
    n_mod, n_obs = ev["n_modifying"], ev["n_observations"]
    _say(f"    {su.side} mixed B={B_max}: stable={stable(rec['batches'][f'mixed_B{B_max}'])[0]}; "
         f"worker evidence: {n_mod} modifying applications over {n_obs} observations"
         + (f"; paths per_slot={ev.get('n_modifying_per_slot')} grouped={ev.get('n_modifying_grouped')}"
            if ev.get("paths_recorded") else "; no per-path counters (the base commit's record)"))
    rec["timing"] = (measure_timing(ctx, su, vm["steered"], timing_prompt) if timing
                     else {"skipped": "timing is measured on the production regime of each kind"})
    return rec


def _upstream_chat(su: Standup, payload: Mapping[str, Any], timeout_s: float = 1800.0
                   ) -> Tuple[Optional[int], Any]:
    """ONE chat completion DIRECTLY at the standup's vLLM server (bypassing
    the proxy — the shared-prefill case mints its own salts; module
    docstring), authenticated with the launch token. Returns ``(status,
    body)``; a non-2xx is returned, not raised, and a dead / unreachable
    server reads as ``(None, {"error": ...})`` — the negative protocol
    expects exactly that."""
    import urllib.error
    import urllib.request
    body = json.dumps(dict(payload)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if su.api_key:
        headers["Authorization"] = f"Bearer {su.api_key}"
    req = urllib.request.Request(f"{su.server_base_url}/v1/chat/completions", data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return int(r.status), json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": {"message": raw[:2000]}}
        return int(e.code), parsed
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return None, {"error": {"message": f"{type(e).__name__}: {e}"}}


def upstream_payload(ctx: GateContext, model: str, messages: Any, *, cache_salt: str,
                     selector: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The upstream request the proxy would have built (``proxy._rewrite``):
    the gate's greedy evidence settings, the base model, the selector in
    :data:`UPSTREAM_EXTRA_ARGS_KEY` when steered, the KV namespace in
    ``cache_salt``."""
    payload: Dict[str, Any] = {"model": model, "messages": list(messages), "max_tokens": int(ctx.max_tokens),
                               "temperature": 0.0, "seed": 0, "logprobs": True,
                               "top_logprobs": int(ctx.top_logprobs), "return_token_ids": True,
                               "ignore_eos": bool(ctx.ignore_eos), "stream": False,
                               "cache_salt": str(cache_salt)}
    if selector is not None:
        payload[UPSTREAM_EXTRA_ARGS_KEY] = dict(selector)
    return payload


def selection_requests(su: Standup) -> Dict[str, Any]:
    """The per-request entries of every worker's SelectionRecord (no reset),
    keyed by request id — union over workers (TP=1 here: one)."""
    out: Dict[str, Any] = {}
    for r in unwrap_rpc(rpc(su, "sisteer_selection_record", [False])):
        if isinstance(r, Mapping):
            for k, v in (r.get("requests") or {}).items():
                out.setdefault(str(k), v)
    return out


def claim_request_failure(su: Standup, request_id: Any) -> Any:
    """``sisteer_take_request_failure`` for one chat completion id — the
    worker's contained per-request failure record (``None`` when none), read
    across workers exactly as the proxy claims it."""
    if request_id is None:
        return {"error": "the response carried no id to claim a failure for"}
    for r in unwrap_rpc(rpc(su, "sisteer_take_request_failure", [str(request_id)])):
        if isinstance(r, Mapping):
            return dict(r)
    return None


def drive_resumed_prefill(ctx: GateContext, su: Standup, *, cell: str, prompts,
                          policy: str) -> Dict[str, Any]:
    """THE SHARED-PREFILL CASE's drive on the caching-ON candidate standup
    (module docstring): the passthrough floor triple, the steered cold
    request, the steered writer, the steered warm request (the SelectionRecord
    reset before it and read after, so its one entry IS the attestation), then
    by the cell's POLICY — ``always``: the identical-prompt repeat and the
    multi-turn pair (a prior turn's header inside the cached region), both
    served and attested; ``conditional``: the multi-turn pair's warm request
    must be CONTAINED (the engine replies 200, the worker's
    ResumedPrefillRefusal is claimed over ``sisteer_take_request_failure``)
    and a fresh request afterwards proves the engine alive, the repeat
    recorded as evidence; ``never``: the warm request is the LAST thing sent
    (its refusal is engine-fatal by design) and the server log is read for the
    typed refusal."""
    import uuid
    if policy not in RESUMED_POLICIES:
        raise GateError(f"unknown resumed-prefill policy {policy!r} (expected one of {RESUMED_POLICIES})")
    hcell = harness_map_of(su.endpoint, [cell])[cell]
    routes = json.loads(Path(su.endpoint["routing_table"]).read_text()).get("routes") or []
    route = next((r for r in routes if str(r.get("cell_id")) == hcell), None)
    if route is None:
        raise GateError(f"the standup's routing table names no cell {hcell!r} (routes: "
                        f"{[r.get('cell_id') for r in routes]})")
    selector = {"petri_object": hcell, "expect_hash": str(route["intervention_hash"])}
    model = str(su.endpoint["model"])
    vms = vmodels_of(su)
    msgs_a, msgs_b = resumed_prompt_pair(prompts)
    uid = uuid.uuid4().hex
    salt_pass = f"gate|passthrough|{uid}"
    # WHAT THE PRODUCTION PROXY WOULD MINT for this cell, from the shipped
    # proxy code against the policy THIS standup's workers attested — and, for
    # an `always` cell, the salt the case actually drives, so the resume the
    # gate proves is the one production gets rather than a look-alike. A
    # `conditional` / `never` case keeps a FORCED gate salt: production never
    # routes those onto a shared prefix (the decision above says so), and
    # reaching the worker's per-request containment / typed refusal at all
    # requires forcing the namespace the proxy declines to mint.
    pdec = proxy_salt_decision(su.endpoint, harness_cell=hcell,
                               intervention_hash=str(route["intervention_hash"]),
                               virtual_model=str(vms[hcell]))
    forced = pdec["cache_salt_regime"] != pdec["cell_stable"]
    salt_cell = (f"gate|{hcell}|{route['intervention_hash']}|{uid}" if forced
                 else str(pdec["salt"]))
    pdec["forced_stable_salt"] = bool(forced)
    pdec["drive_salt"] = salt_cell
    # the live proxy's own decision for a request of this cell, read off the
    # response's petri block (its audit row carries the same two fields).
    # FIRST, before anything that can kill the engine on a `never` cell.
    pdec["live_probe"] = proxy_regime_probe(su, str(vms[hcell]))
    _say(f"    {su.side} proxy decision: attested policy "
         f"{pdec['attested_policy']!r}, regime {pdec['cache_salt_regime']!r}, "
         f"live probe {pdec['live_probe'].get('cache_salt_regime')!r} "
         f"(status {pdec['live_probe'].get('status')!r})")
    rec: Dict[str, Any] = {"apply_path": apply_path_of(su), "harness_cell": hcell, "policy": policy,
                           "resumable": policy == RESUMED_ALWAYS,
                           "proxy_decision": pdec,
                           "commit": su.commit, "served_from": su.endpoint.get("served_from"),
                           "prefix_caching": su.endpoint.get("prefix_caching_check"),
                           "bi_attested": su.endpoint.get("bi_attested"),
                           "reduction_mode_endpoint": su.endpoint.get("reduction_mode_check"),
                           "n_prompt_tokens": {"a": tokenize_count(su.proxy_url, vms["none"], msgs_a),
                                               "b": tokenize_count(su.proxy_url, vms["none"], msgs_b)},
                           "salts": {"passthrough": salt_pass, "cell": salt_cell},
                           "requests": {}, "attest": {}, "signature_errors": [], "closeness": {}}

    def one(name: str, messages: Any, *, salt: str, steered: bool) -> Dict[str, Any]:
        status, body = _upstream_chat(su, upstream_payload(ctx, model, messages, cache_salt=salt,
                                                           selector=(selector if steered else None)))
        entry: Dict[str, Any] = {"status": status, "steered": steered, "salt": salt,
                                 "id": (body.get("id") if isinstance(body, Mapping) else None)}
        if status == 200:
            try:
                entry["signature"] = signature(body["choices"][0])
            except (SignatureError, KeyError, IndexError, TypeError) as e:
                rec["signature_errors"].append(f"{name}: {e}")
                entry["signature_error"] = str(e)
        else:
            entry["error"] = str(body)[:2000]
        if steered:
            # THE PRODUCTION CONTRACT (proxy._refuse_poisoned_lens_response): a
            # worker-CONTAINED hook failure still comes back 200 from vLLM; the
            # proxy claims the per-request failure ledger for EVERY lens response
            # and turns a claimed failure into a 500. The gate drives vLLM
            # directly, so it makes the same claim for EVERY steered response,
            # here, and "served" below means 200 + a signature + NO claimed
            # failure — a failed writer must not seed a cache the gate then
            # validates reuse of.
            entry["claimed_failure"] = claim_request_failure(su, entry.get("id"))
        rec["requests"][name] = entry
        _say(f"    {su.side} {name}: status {status}"
             + (f", {len(entry['signature']['token_ids'])} tokens" if "signature" in entry else "")
             + (f", claimed failure {_claim_type(entry['claimed_failure'])!r}"
                if steered and entry["claimed_failure"] is not None else ""))
        return entry

    # the passthrough floor: cold B under its own salt; writer A and warm B under ONE salt
    one("passthrough_cold_b", msgs_b, salt=f"gate|passthrough-cold|{uid}", steered=False)
    one("passthrough_writer_a", msgs_a, salt=salt_pass, steered=False)
    one("passthrough_warm_b", msgs_b, salt=salt_pass, steered=False)
    # the steered cold request: a unique salt, attested NOT resumed
    rpc(su, "sisteer_selection_record", [True])
    one("steered_cold_b", msgs_b, salt=f"gate|{hcell}-cold|{uid}", steered=True)
    rec["attest"]["steered_cold_b"] = one_request_attestation(selection_requests(su))
    # the writer under the cell-stable salt, then the warm request that resumes it
    one("steered_writer_a", msgs_a, salt=salt_cell, steered=True)
    rpc(su, "sisteer_selection_record", [True])
    warm = one("steered_warm_b", msgs_b, salt=salt_cell, steered=True)
    if policy in (RESUMED_ALWAYS, RESUMED_CONDITIONAL):
        rec["attest"]["steered_warm_b"] = one_request_attestation(selection_requests(su))
        rec["warm_claimed_failure"] = warm.get("claimed_failure")          # aliases of the entries' claims, for the report
        rpc(su, "sisteer_selection_record", [True])
        rep_ = one("steered_warm_b_repeat", msgs_b, salt=salt_cell, steered=True)   # the avg@k shape
        rec["attest"]["steered_warm_b_repeat"] = one_request_attestation(selection_requests(su))
        rec["repeat_claimed_failure"] = rep_.get("claimed_failure")
        # THE MULTI-TURN SHAPE: writer + warm reader sharing a prefix that carries the
        # first turn's assistant header deep inside the cached region
        mt_a, mt_b = resumed_multiturn_pair(prompts)
        rec["n_prompt_tokens"]["multiturn_a"] = tokenize_count(su.proxy_url, vms["none"], mt_a)
        rec["n_prompt_tokens"]["multiturn_b"] = tokenize_count(su.proxy_url, vms["none"], mt_b)
        salt_mt = f"gate|{hcell}|{route['intervention_hash']}|multiturn|{uid}"
        one("steered_multiturn_writer", mt_a, salt=salt_mt, steered=True)
        rpc(su, "sisteer_selection_record", [True])
        mt = one("steered_multiturn_warm", mt_b, salt=salt_mt, steered=True)
        rec["attest"]["steered_multiturn_warm"] = one_request_attestation(selection_requests(su))
        rec["multiturn_claimed_failure"] = mt.get("claimed_failure")
        if policy == RESUMED_CONDITIONAL:
            # the engine must still be serving after the CONTAINED refusal: a fresh cold
            # steered request, with no failure record of its own
            after = one("steered_after_refusal", msgs_b, salt=f"gate|{hcell}-after|{uid}", steered=True)
            rec["after_refusal_claimed_failure"] = after.get("claimed_failure")
        _say(f"    {su.side} {policy} cell: multi-turn warm status {mt.get('status')!r}, attested "
             f"{rec['attest']['steered_multiturn_warm'].get('prefill_resumed')!r}, claimed failure "
             f"{(rec['multiturn_claimed_failure'] or {}).get('error_type') if isinstance(rec['multiturn_claimed_failure'], Mapping) else rec['multiturn_claimed_failure']!r}")
    else:
        # engine-fatal by design (module docstring): nothing else touches the
        # server; the typed refusal is read out of its log
        rec["server_died"] = warm.get("status") != 200
        try:
            log_text = (su.out / "vllm_server.log").read_text(errors="replace")
        except OSError as e:
            log_text = ""
            rec["log_error"] = f"{type(e).__name__}: {e}"
        rec["refusal_marker_in_log"] = RESUMED_REFUSAL_MARKER in log_text
        _say(f"    {su.side} non-resumable cell: warm request status {warm.get('status')!r}; typed refusal in "
             f"the server log: {rec['refusal_marker_in_log']}")
    reqs = rec["requests"]
    for label, warm_name, cold_name in (("steered", "steered_warm_b", "steered_cold_b"),
                                        ("passthrough", "passthrough_warm_b", "passthrough_cold_b")):
        w, c = reqs.get(warm_name) or {}, reqs.get(cold_name) or {}
        if "signature" in w and "signature" in c:
            rec["closeness"][label] = signature_closeness(w["signature"], c["signature"])
    ok, detail = resumed_prefill_check(rec)
    _say(f"    {su.side} shared-prefill case: {'ok' if ok else 'FAIL'} — {detail[:400]}")
    return rec


# --------------------------------------------------------------------------- #
# THE DECLARED-RENDER CASES (runtime)                                         #
# --------------------------------------------------------------------------- #


def proxy_chat(su: Standup, payload: Mapping[str, Any], timeout_s: float = 1800.0
               ) -> Tuple[Optional[int], Any]:
    """ONE chat completion THROUGH the standup's routing proxy — the
    production door every declared-render request must pass (the proxy's
    validation IS what the cases prove). ``(status, body)``; a 4xx/5xx is
    returned, not raised, with vLLM's / the proxy's error body parsed."""
    import urllib.error
    import urllib.request
    body = json.dumps(dict(payload)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if su.api_key:
        headers["Authorization"] = f"Bearer {su.api_key}"
    req = urllib.request.Request(f"{su.proxy_url.rstrip('/')}/chat/completions", data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return int(r.status), json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": {"message": raw[:2000]}}
        return int(e.code), parsed
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return None, {"error": {"message": f"{type(e).__name__}: {e}"}}


def proxy_models(su: Standup, timeout_s: float = 120.0) -> Dict[str, Any]:
    """The proxy's ``/v1/models`` identity block (``petri_proxy``)."""
    import urllib.request
    headers = {}
    if su.api_key:
        headers["Authorization"] = f"Bearer {su.api_key}"
    req = urllib.request.Request(f"{su.proxy_url.rstrip('/')}/models", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        doc = json.loads(r.read().decode("utf-8", "replace"))
    block = doc.get("petri_proxy") if isinstance(doc, Mapping) else None
    if not isinstance(block, Mapping):
        raise GateError(f"/v1/models carries no petri_proxy identity block: {str(doc)[:300]}")
    return dict(block)


def proxy_tokenize(su: Standup, vmodel: str, messages: Any, timeout_s: float = 120.0
                   ) -> Dict[str, Any]:
    """The proxy's ``/v1/tokenize`` for ``messages`` under ``vmodel``: vLLM's
    ``{count, tokens, ...}`` — the rendered prompt ids the gate runs its own
    span arithmetic on."""
    import urllib.request
    body = json.dumps({"model": vmodel, "messages": list(messages),
                       "add_generation_prompt": True}).encode("utf-8")
    req = urllib.request.Request(f"{su.proxy_url.rstrip('/')}/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def one_declared_attestation(requests: Mapping[str, Any]) -> Dict[str, Any]:
    """The SelectionRecord's ``requests`` map read after ONE lens request on a
    record reset just before it (the shared-prefill case's discipline): the
    entry count, and — for exactly one entry — its :data:`DECLARED_ATTEST_
    FIELDS`. Zero or several entries are recorded as such, never the first
    one taken."""
    n = len(requests) if isinstance(requests, Mapping) else None
    if n != 1:
        return {"n_entries": n, "error": "expected exactly one request entry in the reset SelectionRecord"}
    (rid, entry), = requests.items()
    out: Dict[str, Any] = {"n_entries": 1, "req_id": str(rid)}
    for k in DECLARED_ATTEST_FIELDS:
        out[k] = entry.get(k) if isinstance(entry, Mapping) else None
    return out


def twin_donor_of(su: Standup) -> Any:
    """The ``none__lens0`` entry's ``donor_cell`` from the standup's exported
    registry index — the P12 donor rule's positive artifact (the ``@thinking``
    cell on a thinking standup; the base cell where the only variant is a
    position variant)."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import lens_steer as ls
    idx = Path(su.out) / "lens_registry" / ls.REGISTRY_INDEX
    if not idx.is_file():
        return None
    for e in (json.loads(idx.read_text()).get("entries") or []):
        if isinstance(e, Mapping) and e.get("cell_id") == dc.NOOP_LENS_CELL_ID:
            return e.get("donor_cell")
    return None


def declared_render_object(ms: ModelServe, *, bank_dir: Optional[Path], header_len: int
                           ) -> Dict[str, Any]:
    """THE REAL response-fit object the declared-render cases serve —
    :data:`DECLARED_RENDER_CELL` from this family's scale-12 bank: its path,
    its sha256 (the digest ``endpoint.json.object_artifact_sha256`` must
    equal, Q8), and the facts its REAL adapt yields (kind, the ops' served
    positions — which must be ``response`` on every op, skip_bos). No
    synthetic fallback and no other SI: a bank without the cell refuses."""
    from sisteer.eval.defense_cell import object_path_for
    if bank_dir is None or not Path(bank_dir).is_dir():
        raise GateError(f"{ms.key}: no scale-12 bank at {bank_dir} — the declared-render cases serve the "
                        f"REAL {DECLARED_RENDER_CELL} object and take no synthetic stand-in")
    obj_path = object_path_for(Path(bank_dir), DECLARED_RENDER_CELL)
    if obj_path is None or not Path(obj_path).is_file():
        raise GateError(f"{ms.key}: the bank {bank_dir} holds no {DECLARED_RENDER_CELL} artifact "
                        f"({obj_path}) — the declared-render cases serve that object and no other")
    raw = Path(obj_path).read_bytes()
    kind, ops = adapt_object_dir(Path(obj_path).parent, arm=DECLARED_RENDER_ARM, header_len=header_len)
    served = sorted({op.positions for op in ops})
    if served != ["response"]:
        raise GateError(f"{obj_path}: the real {DECLARED_RENDER_CELL} adapts to positions {served}, not "
                        "['response'] — the declared-render cases are about a RESPONSE-fit object")
    return {"cell": DECLARED_RENDER_CELL, "defenses_dir": str(Path(bank_dir)), "objects": "real",
            "source": str(obj_path), "sha256": hashlib.sha256(raw).hexdigest(),
            "kind": kind, "n_ops": len(ops), "served_positions": served,
            "skip_bos": bool(any(getattr(op, "skip_bos", False) for op in ops)),
            "arm": DECLARED_RENDER_ARM}


def guard_probe(obj: Mapping[str, Any], *, header_len: int) -> Dict[str, Any]:
    """THE FIT/SERVE GUARD, exercised in-process on the REAL artifact's bytes
    through the REAL adapter (``lens_adapters.get_adapter`` — the same
    guarded call the worker makes at load): the base spec overlaid with
    ``@all_positions`` ADAPTS (served positions ``["all"]``, admitted by
    name), and the same spec set to ``positions="all"`` WITHOUT the name is
    REFUSED, the refusal naming the carve-out. Recorded, judged by
    ``carveout_guard_*``. CPU, in the gate container — the negative half
    needs no second failing GPU standup."""
    import torch
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import serve_declaration as sd
    art = Path(obj["source"])
    raw = art.read_bytes()
    blob = torch.load(art, map_location="cpu", weights_only=True)
    base_spec = resolved_spec_for(blob, kind=obj["kind"], arm=obj["arm"], safetensors=False)
    header = tuple(range(1000, 1000 + int(header_len)))
    out: Dict[str, Any] = {"base_positions": base_spec.get("positions")}
    declared_spec = sd.apply_variant(base_spec, "all_positions")
    try:
        ops = la.get_adapter(obj["kind"])(declared_spec, raw, b"", header_ids=header)
        out["declared"] = {"adapted": True, "served_positions": sorted({op.positions for op in ops}),
                           "experiment": declared_spec.get(sd.SERVE_EXPERIMENT_KEY)}
    except Exception as e:  # noqa: BLE001 — recorded, judged
        out["declared"] = {"adapted": False, "error": f"{type(e).__name__}: {e}"[:1500]}
    undeclared_spec = dict(base_spec)
    undeclared_spec["positions"] = "all"
    try:
        la.get_adapter(obj["kind"])(undeclared_spec, raw, b"", header_ids=header)
        out["undeclared"] = {"refused": False}
    except Exception as e:  # noqa: BLE001 — the refusal IS the evidence
        out["undeclared"] = {"refused": True, "error": f"{type(e).__name__}: {e}"[:1500]}
    return out


def thinking_header_for(hf: str, rev: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """The gate's OWN derivation of the served template's thinking-header fact
    on the pinned tokenizer — ``(facts, reason)`` through the NON-FATAL
    ``chat_markers.thinking_header_facts_or_error`` (code review S-G3 / X1):
    a template whose ON header cannot certify (Nemotron-3) yields the partial
    fact and the ``ChatMarkerError`` reason instead of killing the gate run;
    ``thinking_header_matches_tokenizer`` then fails naming it. What the
    endpoint's / the proxy's fact must equal."""
    from transformers import AutoTokenizer
    from sisteer.chat_markers import thinking_header_facts_or_error
    from sisteer.models import pin_reasoning_template_off
    tok = AutoTokenizer.from_pretrained(hf, revision=rev)
    pin_reasoning_template_off(tok)
    facts, reason = thinking_header_facts_or_error(tok)
    return dict(facts), reason


def turn_end_ids_for(hf: str, rev: str) -> Optional[List[int]]:
    """The gate's OWN derivation of the served template's turn-end scaffold
    (``chat_markers.assistant_turn_end_ids``), ``None`` when the template has
    none."""
    from transformers import AutoTokenizer
    from sisteer.chat_markers import assistant_turn_end_ids
    from sisteer.models import pin_reasoning_template_off
    tok = AutoTokenizer.from_pretrained(hf, revision=rev)
    pin_reasoning_template_off(tok)
    got = assistant_turn_end_ids(tok)
    return None if got is None else [int(t) for t in got]


def _endpoint_facts(su: Standup) -> Dict[str, Any]:
    ep = su.endpoint
    keys = (SERVE_DECLARATIONS_KEY, SERVE_DECLARATIONS_KEY + "_error", THINKING_HEADER_KEY,
            THINKING_HEADER_ERROR_KEY, LENS_SALT_POLICY_KEY, LENS_SALT_POLICY_KEY + "_error",
            OBJECT_SHA256_BY_CELL_KEY, "object_artifact_sha256", "object_artifact_sha256_bound",
            "object_artifact_sha256_error", "declared_render", "registry_markers",
            "declared_render_needs", "cell_map", "harness_cells", "resumed_prefill_policy_cells")
    return {k: ep.get(k) for k in keys if k in ep}


def _models_facts(su: Standup) -> Dict[str, Any]:
    block = proxy_models(su)
    keys = (SERVE_DECLARATIONS_KEY, SERVE_DECLARATIONS_KEY + "_error", THINKING_HEADER_KEY,
            THINKING_HEADER_ERROR_KEY, LENS_SALT_POLICY_KEY, OBJECT_SHA256_BY_CELL_KEY,
            "resumed_prefill_policy_cells", LENS_REDUCTION_MODE_KEY)
    return {k: block.get(k) for k in keys if k in block}


def _declared_cells(su: Standup, base_tt: str, variant_tt: str) -> Dict[str, str]:
    """The served ids, DERIVED from the standup's explicit cell map and its
    routing table (never spelled): the base and variant harness ids, the twin
    ``none__lens0``, and every vmodel."""
    from sisteer.capeval.petri import defense_cells as dc
    hmap = harness_map_of(su.endpoint, [base_tt, variant_tt])
    vms = vmodels_of(su)
    out = {"base_tt": base_tt, "variant_tt": variant_tt, "base": hmap[base_tt],
           "variant": hmap[variant_tt], "twin": dc.NOOP_LENS_CELL_ID}
    for role in ("base", "variant", "twin"):
        if out[role] not in vms:
            raise GateError(f"the standup's routing table names no cell {out[role]!r} (routes: {sorted(vms)})")
    out["vm_base"], out["vm_variant"], out["vm_twin"] = vms[out["base"]], vms[out["variant"]], vms[out["twin"]]
    if "none" not in vms:
        raise GateError(f"the standup's routing table names no base route (routes: {sorted(vms)})")
    out["vm_passthrough"] = vms["none"]
    return out


class _DeclaredDriver:
    """One standup's request driver: sends each leg THROUGH the proxy, resets
    the SelectionRecord before a lens leg and reads it after (exactly one
    entry for a served lens request; NONE for a refused one), claims the
    failure ledger for every served lens response, and files everything on
    ``rec["requests"][name]``."""

    def __init__(self, su: Standup, rec: Dict[str, Any]):
        self.su, self.rec = su, rec

    def send(self, name: str, payload: Mapping[str, Any], *, lens_route: bool) -> Dict[str, Any]:
        su = self.su
        if lens_route:
            rpc(su, "sisteer_selection_record", [True])   # reset: the next read is this request's
        status, body = proxy_chat(su, payload)
        entry: Dict[str, Any] = {"status": status, "vmodel": payload.get("model"), "lens_route": bool(lens_route),
                                 "request_keys": sorted(k for k in payload if k not in ("model", "messages")),
                                 "id": (body.get("id") if isinstance(body, Mapping) else None)}
        if status == 200:
            try:
                entry["signature"] = signature(body["choices"][0])
            except (SignatureError, KeyError, IndexError, TypeError) as e:
                self.rec["signature_errors"].append(f"{name}: {e}")
                entry["signature_error"] = str(e)
            try:
                entry["text"] = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                entry["text"] = None
            petri = body.get("petri") if isinstance(body, Mapping) else None
            entry["petri"] = dict(petri) if isinstance(petri, Mapping) else None
        else:
            err = body.get("error") if isinstance(body, Mapping) else None
            entry["code"] = err.get("code") if isinstance(err, Mapping) else None
            entry["error"] = (err.get("message") if isinstance(err, Mapping) else str(body))
            entry["error"] = str(entry["error"])[:2000]
        if lens_route:
            entry["attest"] = one_declared_attestation(selection_requests(su))
            if status == 200:
                # THE PRODUCTION CONTRACT: the proxy claimed this response's
                # ledger before forwarding it (a claimed failure is a 500, never
                # a 200); the gate claims again and records the EMPTY ledger
                entry["claimed_failure"] = claim_request_failure(su, entry.get("id"))
        self.rec["requests"][name] = entry
        _say(f"    {su.side} {name} [{payload.get('model')}]: status {status}"
             + (f" code {entry.get('code')!r}" if status != 200 else "")
             + (f", {len(entry['signature']['token_ids'])} tokens" if "signature" in entry else "")
             + (f", rows {entry['attest'].get('prefill_masked_rows')!r} facts "
                f"{entry['attest'].get('facts')!r} n_mod {entry['attest'].get('n_applications_modifying')!r}"
                if lens_route and entry.get("attest", {}).get("n_entries") == 1 else "")
             + (f", stamp {entry['petri'].get(LEDGER_KEY_RESUMED_POLICY)!r}/"
                f"{entry['petri'].get(LEDGER_KEY_RENDER_REGIME)!r}" if entry.get("petri") else ""))
        return entry


def _base_record(su: Standup, cells: Mapping[str, str], obj: Mapping[str, Any], variant: str
                 ) -> Dict[str, Any]:
    return {"apply_path": apply_path_of(su), "commit": su.commit, "served_from": su.endpoint.get("served_from"),
            "prefix_caching": su.endpoint.get("prefix_caching_check"),
            "bi_attested": su.endpoint.get("bi_attested"),
            "reduction_mode_endpoint": su.endpoint.get("reduction_mode_check"),
            "cells": dict(cells), "object": dict(obj), "variant": variant,
            "expected_declaration": expected_declarations(variant),
            "endpoint": _endpoint_facts(su), "models": _models_facts(su),
            "twin_donor": twin_donor_of(su),
            "requests": {}, "signature_errors": []}


def drive_declared_thinking(ctx: GateContext, su: Standup, *, base_tt: str, variant_tt: str,
                            obj: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``declared_thinking`` drive (module docstring, THE DECLARED-RENDER
    CASES): the declared ON request on the ``@thinking`` cell and on the twin,
    the OFF request on the base cell, the five refusals, the undeclared ON
    passthrough, then the standup's attestation facts and the gate's own
    tokenizer-derived header fact."""
    cells = _declared_cells(su, base_tt, variant_tt)
    rec = _base_record(su, cells, obj, DECLARED_RENDER_VARIANT[DECLARED_THINKING])
    rec["tokenizer_thinking_header"], rec["tokenizer_thinking_header_error"] = \
        thinking_header_for(ctx.ms.hf, ctx.ms.rev)
    sampling = evidence_sampling(ctx)
    msgs = declared_render_prompt()
    d = _DeclaredDriver(su, rec)
    # the served legs (byte for byte in their keys)
    d.send("on_steered", thinking_request(cells["vm_variant"], msgs, sampling, enable_thinking=True, declare=True),
           lens_route=True)
    d.send("on_twin", thinking_request(cells["vm_twin"], msgs, sampling, enable_thinking=True, declare=True),
           lens_route=True)
    d.send("off_base", thinking_request(cells["vm_base"], msgs, sampling, enable_thinking=False, declare=False),
           lens_route=True)
    d.send("passthrough_on", thinking_request(cells["vm_passthrough"], msgs, sampling, enable_thinking=True,
                                              declare=False), lens_route=False)
    # the refusal matrix (§2.5.6), each on a reset record: nothing may reach the worker
    d.send("refuse_undeclared", thinking_request(cells["vm_variant"], msgs, sampling, enable_thinking=True,
                                                 declare=False), lens_route=True)
    d.send("refuse_mismatch", thinking_request(cells["vm_variant"], msgs, sampling, enable_thinking=None,
                                               declare=True), lens_route=True)
    d.send("refuse_kwargs_invalid", thinking_request(cells["vm_variant"], msgs, sampling, enable_thinking=True,
                                                     declare=True, extra_kwargs={"foreign_kwarg": 1}),
           lens_route=True)
    d.send("refuse_cell_not_declared", thinking_request(cells["vm_base"], msgs, sampling, enable_thinking=True,
                                                        declare=True), lens_route=True)
    d.send("refuse_not_lens", thinking_request(cells["vm_passthrough"], msgs, sampling, enable_thinking=True,
                                               declare=True), lens_route=False)
    rec["requests"]["refuse_not_lens"]["attest"] = {"n_entries": 0}   # a passthrough route has no lens entry to reset/read
    return rec


def drive_all_positions_carveout(ctx: GateContext, su: Standup, *, base_tt: str, variant_tt: str,
                                 obj: Mapping[str, Any], header_len: int) -> Dict[str, Any]:
    """The ``all_positions_carveout`` drive on the default-policy standup: the
    in-process guard probe on the real artifact, the single-turn request on
    the ``@all_positions`` cell (masked rows == P), the base cell (== H) and
    the twin, the degeneracy instrument over the served texts, the
    attestation facts."""
    cells = _declared_cells(su, base_tt, variant_tt)
    rec = _base_record(su, cells, obj, DECLARED_RENDER_VARIANT[ALL_POSITIONS_CARVEOUT])
    rec["guard"] = guard_probe(obj, header_len=header_len)
    _say(f"    {su.side} guard probe: declared {rec['guard'].get('declared')}, undeclared refused "
         f"{(rec['guard'].get('undeclared') or {}).get('refused')!r}")
    sampling = evidence_sampling(ctx)
    msgs = declared_render_prompt()
    tk = proxy_tokenize(su, cells["vm_variant"], msgs)
    rec["prompt_len"] = int(tk.get("count")) if isinstance(tk.get("count"), int) else None
    rec["expected_all_rows"] = all_positions_rows_expected(rec["prompt_len"], skip_bos=bool(obj.get("skip_bos")))
    d = _DeclaredDriver(su, rec)
    d.send("variant_single", plain_request(cells["vm_variant"], msgs, sampling), lens_route=True)
    d.send("base_single", plain_request(cells["vm_base"], msgs, sampling), lens_route=True)
    d.send("twin_single", plain_request(cells["vm_twin"], msgs, sampling), lens_route=True)
    rec["degeneracy"] = {name: degeneracy_instrument((rec["requests"].get(name) or {}).get("text"))
                         for name in ("variant_single", "base_single", "twin_single")}
    return rec


def drive_forced_policy(ctx: GateContext, su: Standup, *, base_tt: str, variant_tt: str,
                        obj: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``all_positions_carveout`` case's SECOND standup
    (``--lens-salt-policy request_unique_all``): the same three legs, read
    for the forced stamp AND for the same worker evidence as the first
    standup's (S-G1: one entry, requested cell, matched/stable, applications,
    the masked-row geometry — derived from THIS standup's /v1/tokenize
    render, never carried over); the attestation facts again (the base cell
    must still be absent from the map)."""
    cells = _declared_cells(su, base_tt, variant_tt)
    rec = _base_record(su, cells, obj, DECLARED_RENDER_VARIANT[ALL_POSITIONS_CARVEOUT])
    sampling = evidence_sampling(ctx)
    msgs = declared_render_prompt()
    tk = proxy_tokenize(su, cells["vm_variant"], msgs)
    rec["prompt_len"] = int(tk.get("count")) if isinstance(tk.get("count"), int) else None
    rec["expected_all_rows"] = all_positions_rows_expected(rec["prompt_len"], skip_bos=bool(obj.get("skip_bos")))
    d = _DeclaredDriver(su, rec)
    d.send("variant_single", plain_request(cells["vm_variant"], msgs, sampling), lens_route=True)
    d.send("base_single", plain_request(cells["vm_base"], msgs, sampling), lens_route=True)
    d.send("twin_single", plain_request(cells["vm_twin"], msgs, sampling), lens_route=True)
    return rec


def drive_all_turns_render_check(ctx: GateContext, su: Standup, *, base_tt: str, variant_tt: str,
                                 obj: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``all_turns_render_check`` drive: the 3-turn conversation on the
    ``@all_turns`` cell (label ``ok``; masked rows == the gate's own span sum
    over ``/v1/tokenize``'s ids), on the base cell (== H) and on the twin;
    the quoted-delimiter and complete-fake-boundary probes (400 on the
    variant, 200 on the base); the attestation facts; the gate's own
    scaffold derivation."""
    cells = _declared_cells(su, base_tt, variant_tt)
    rec = _base_record(su, cells, obj, DECLARED_RENDER_VARIANT[ALL_TURNS_RENDER_CHECK])
    rec["tokenizer_turn_end_ids"] = turn_end_ids_for(ctx.ms.hf, ctx.ms.rev)
    markers = (su.endpoint.get("registry_markers") or {})
    sampling = evidence_sampling(ctx)
    convo = [dict(m) for m in ALL_TURNS_CONVERSATION]
    tk = proxy_tokenize(su, cells["vm_variant"], convo)
    ids = tk.get("tokens")
    rec["prompt_len"] = int(tk.get("count")) if isinstance(tk.get("count"), int) else None
    rec["located_spans"] = None
    rec["expected_all_turns_rows"] = None
    try:
        if not isinstance(ids, list) or not ids:
            raise GateError(f"/v1/tokenize returned no token ids ({str(tk)[:200]})")
        spans = all_turns_spans_of([int(t) for t in ids], markers.get("assistant_header_ids") or (),
                                   markers.get("assistant_turn_end_ids") or ())
        rec["located_spans"] = [list(s) for s in spans]
        rec["expected_all_turns_rows"] = all_turns_rows_expected(spans)
    except Exception as e:  # noqa: BLE001 — recorded; the masked-rows check then fails naming it
        rec["span_error"] = f"{type(e).__name__}: {e}"
    _say(f"    {su.side} 3-turn render: P={rec['prompt_len']} spans {rec['located_spans']} -> "
         f"expected rows {rec['expected_all_turns_rows']}" + (f" ({rec.get('span_error')})" if rec.get("span_error") else ""))
    d = _DeclaredDriver(su, rec)
    d.send("convo_variant", plain_request(cells["vm_variant"], convo, sampling), lens_route=True)
    d.send("convo_base", plain_request(cells["vm_base"], convo, sampling), lens_route=True)
    d.send("convo_twin", plain_request(cells["vm_twin"], convo, sampling), lens_route=True)
    for kind in ("quoted", "fake"):
        probe = all_turns_probe_conversation(kind)
        d.send(f"{kind}_variant", plain_request(cells["vm_variant"], probe, sampling), lens_route=True)
        d.send(f"{kind}_base", plain_request(cells["vm_base"], probe, sampling), lens_route=True)
    return rec


def run_declared_render_case(ctx: GateContext, case_id: str, *, cand_sha: str, obj: Mapping[str, Any],
                             header_len: int) -> Dict[str, Any]:
    """Stand the CANDIDATE up with ``none`` + the base cell + the case's
    variant cell and drive the case; the carve-out case stands up a second
    time under ``--lens-salt-policy request_unique_all``. Returns the case
    record (checks filed by :func:`declared_render_checks`)."""
    variant = DECLARED_RENDER_VARIANT[case_id]
    base_tt = str(obj["cell"])
    variant_tt = variant_cell(base_tt, variant)
    cells = [base_tt, variant_tt]
    kr: Dict[str, Any] = {"case_id": case_id, "kind": "declared_render", "arms": [obj["arm"]],
                          "timing_measured": False, "cells_tt": cells}
    kr.update({k: obj[k] for k in ("cell", "defenses_dir", "objects", "source")})
    kr["arm"] = obj["arm"]
    su = None
    try:
        su = start_standup(ctx, side=DECLARED, tree=ctx.candidate, expect_sha=cand_sha, cell=cells,
                           defenses_dir=Path(obj["defenses_dir"]), tag=f"{case_id}__{DECLARED}",
                           reduction_mode=ctx.reduction_mode, declared_render=True)
        if case_id == DECLARED_THINKING:
            rec = drive_declared_thinking(ctx, su, base_tt=base_tt, variant_tt=variant_tt, obj=obj)
        elif case_id == ALL_POSITIONS_CARVEOUT:
            rec = drive_all_positions_carveout(ctx, su, base_tt=base_tt, variant_tt=variant_tt, obj=obj,
                                               header_len=header_len)
        else:
            rec = drive_all_turns_render_check(ctx, su, base_tt=base_tt, variant_tt=variant_tt, obj=obj)
    finally:
        stop_standup(su)
    if case_id == ALL_POSITIONS_CARVEOUT:
        su = None
        try:
            su = start_standup(ctx, side=DECLARED_FORCED, tree=ctx.candidate, expect_sha=cand_sha, cell=cells,
                               defenses_dir=Path(obj["defenses_dir"]), tag=f"{case_id}__{DECLARED_FORCED}",
                               reduction_mode=ctx.reduction_mode,
                               lens_salt_policy=LENS_SALT_POLICY_REQUEST_UNIQUE_ALL, declared_render=True)
            rec["forced"] = drive_forced_policy(ctx, su, base_tt=base_tt, variant_tt=variant_tt, obj=obj)
        finally:
            stop_standup(su)
    kr.update(rec)
    kr["checks"] = declared_render_checks(case_id, rec)
    return kr


def _declared_render_case(ctx: GateContext, case_id: str, *, cand_sha: str, bank_dir: Optional[Path],
                          header_len: int, uncovered: List[str]) -> Dict[str, Any]:
    """One declared-render case, recorded like every other case: resolve the
    REAL object, run it, file its checks; an exception is the case's
    ``error`` (exit 3 through ``overall_exit_code``), never a crash of the
    run. The case leaves ``uncovered`` only when it filed checks."""
    kr: Dict[str, Any] = {"case_id": case_id, "kind": "declared_render", "arms": [DECLARED_RENDER_ARM],
                          "timing_measured": False}
    try:
        obj = declared_render_object(ctx.ms, bank_dir=bank_dir, header_len=header_len)
        _say(f"case {case_id}: cell {obj['cell']} objects={obj['objects']} sha256 {obj['sha256'][:16]}… "
             f"from {obj['source']} (adapts to positions {obj['served_positions']}, skip_bos {obj['skip_bos']})")
        kr = run_declared_render_case(ctx, case_id, cand_sha=cand_sha, obj=obj, header_len=header_len)
        if case_id in uncovered:
            uncovered.remove(case_id)
        _say(f"case {case_id}: checks " + ", ".join(
            f"{k}={'ok' if c['ok'] else ('inconclusive' if c['ok'] is None else 'FAIL')}"
            for k, c in kr["checks"].items()))
    except Exception as e:  # noqa: BLE001
        kr["error"] = f"{type(e).__name__}: {e}"
        kr["traceback"] = traceback.format_exc()[-3000:]
        _say(f"case {case_id} FAILED: {type(e).__name__}: {str(e)[:300]}")
    return kr


def coschedule_snapshot(su: Standup, *, reset: bool) -> Any:
    """The worker's CoScheduleRecord snapshot (``sisteer_coschedule_record``),
    read through :func:`unwrap_rpc`; ``{"rpc": "absent"}`` on the base
    commit, which has no such method."""
    try:
        res = rpc(su, "sisteer_coschedule_record", [bool(reset)])
    except RpcError as e:
        return {"rpc": "absent", "status": e.status, "detail": e.detail[:300]}
    return single_worker(unwrap_rpc(res), "sisteer_coschedule_record")


def shadow_snapshot(su: Standup, *, reset: bool) -> Any:
    """The worker's ShadowCompareRecord (``sisteer_shadow_compare``), read
    through :func:`unwrap_rpc`; ``{"rpc": "absent"}`` where the method does
    not exist (the base commit)."""
    try:
        res = rpc(su, "sisteer_shadow_compare", [bool(reset)])
    except RpcError as e:
        return {"rpc": "absent", "status": e.status, "detail": e.detail[:300]}
    return single_worker(unwrap_rpc(res), "sisteer_shadow_compare")


def controls_B1(ctx: GateContext, su: Standup, prompts, vm: Mapping[str, Optional[str]],
                roles: Sequence[str]) -> Dict[str, Any]:
    """The B=1 singleton controls: each role's request for prompt 0 sent ALONE
    (one request per step — no composition confound), in sequence."""
    out: Dict[str, Any] = {}
    for role in roles:
        v = vm.get(role)
        if not v:
            raise GateError(f"no virtual model for role {role!r} (routes: {vm})")
        try:
            out[role] = run_batch(su.proxy_url, [(v, prompts[0][0], {})], max_tokens=ctx.max_tokens,
                                  ignore_eos=ctx.ignore_eos, top_logprobs=ctx.top_logprobs)[0]
        except SignatureError as e:
            out[role] = {"signature_error": str(e)}
    return out


def controls_B1_facts(controls: Mapping[str, Any], oracle_b1: Optional[Mapping[str, Any]],
                      steered_roles: Sequence[str]) -> Dict[str, Any]:
    """Compare the B=1 singletons: passthrough and scale-0 must equal the
    oracle's B=1 passthrough; every steered role must differ from it; the
    steered roles must differ pairwise (the multigroup cells)."""
    facts: Dict[str, Any] = {}
    if oracle_b1 is None:
        return {"controls_match": None, "controls_detail": "no B=1 oracle", "steered_differs": None,
                "cells_distinct": None}
    bad = []
    for role in ("passthrough", "scale0"):
        sig = controls.get(role)
        if not isinstance(sig, Mapping) or "signature_error" in sig:
            bad.append(f"{role}: {sig}")
            continue
        ok, why = signatures_equal([sig], [oracle_b1])
        if not ok:
            bad.append(f"{role}: {why}")
    facts["controls_match"] = not bad
    if bad:
        facts["controls_detail"] = "; ".join(bad)
    differs = []
    for role in steered_roles:
        sig = controls.get(role)
        differs.append(isinstance(sig, Mapping) and "signature_error" not in sig
                       and signatures_differ_somewhere([sig], [oracle_b1]))
    facts["steered_differs"] = bool(steered_roles) and all(differs)
    distinct = True
    detail = []
    for i in range(len(steered_roles)):
        for j in range(i + 1, len(steered_roles)):
            a, b = controls.get(steered_roles[i]), controls.get(steered_roles[j])
            if isinstance(a, Mapping) and isinstance(b, Mapping) and not signatures_differ_somewhere([a], [b]):
                distinct = False
                detail.append(f"{steered_roles[i]} == {steered_roles[j]}")
    facts["cells_distinct"] = (facts["steered_differs"] and distinct) if steered_roles else None
    if detail:
        facts["cells_detail"] = "identical B=1 outputs: " + "; ".join(detail)
    elif not facts["steered_differs"]:
        facts["cells_detail"] = "a steered B=1 request is bit-identical to passthrough"
    return facts


def drive_shadow(ctx: GateContext, su: Standup, *, cell: str, prompts, timing: bool,
                 routing: str) -> Dict[str, Any]:
    """The candidate-with-shadow standup of a single-op case: the pure batches
    at every ladder size. Each wave's comparisons are ATTRIBUTED TO THAT WAVE
    by snapshotting the ShadowCompareRecord before and after it
    (``shadow_wave_delta``); a VECTORIZED case re-drives a size up to
    SHADOW_ATTEMPTS times until some wave at that size carries a counted
    comparison (a wave whose requests all failed carries none), a PER-SLOT
    case drives each size once (nothing to wait for — its check is the
    SelectionRecord's per-path counters, read here too). The cumulative record
    is the case's shadow evidence, the actual group row counts its
    kernel-shape evidence; timing when the case measures it."""
    vms = vmodels_of(su)
    hcell = harness_map_of(su.endpoint, [cell])[cell]
    vm = {"passthrough": vms.get("none"), "scale0": vms.get("none__lens0"), "steered": vms.get(hcell)}
    rec: Dict[str, Any] = {"apply_path": apply_path_of(su), "vmodels": vm, "signature_errors": [],
                           "commit": su.commit, "served_from": su.endpoint.get("served_from"),
                           "harness_cell": hcell, "bi_attested": su.endpoint.get("bi_attested"),
                           "prefix_caching": su.endpoint.get("prefix_caching_check"),
                           "reduction_mode_endpoint": su.endpoint.get("reduction_mode_check"),
                           "routing": routing, "shadow_waves": []}
    shadow_snapshot(su, reset=True)
    try:
        rpc(su, "sisteer_selection_record", [True])   # reset: scope the per-path counters to this drive
    except RpcError as e:
        rec["evidence_error"] = str(e)
    attempts = SHADOW_ATTEMPTS if routing == ROUTING_VECTORIZED else 1
    for B in ctx.batches:
        for attempt in range(attempts):
            before = shadow_snapshot(su, reset=False)
            try:
                run_batch(su.proxy_url, _batch_requests(prompts, ["steered"] * B, vm),
                          max_tokens=ctx.max_tokens, ignore_eos=ctx.ignore_eos,
                          top_logprobs=ctx.top_logprobs)
            except SignatureError as e:
                rec["signature_errors"].append(str(e))
            after = shadow_snapshot(su, reset=False)
            delta = shadow_wave_delta(before, after)
            got = delta["compared"] > 0
            accepted = routing != ROUTING_VECTORIZED or got or attempt == attempts - 1
            rec["shadow_waves"].append({
                "B": B, "attempt": attempt + 1, "compared_in_wave": delta["compared"],
                "unequal_in_wave": delta["unequal"], "unmodified_in_wave": delta["unmodified"],
                "shadow_errors_in_wave": delta["shadow_errors"],
                "rows_compared_in_wave": delta["rows_compared"], "accepted": accepted,
                "n_unequal_so_far": (after or {}).get("n_unequal") if isinstance(after, Mapping) else None})
            if accepted:
                break
            _say(f"    {su.side} shadow: no counted comparison in the B={B} wave (attempt "
                 f"{attempt + 1}/{attempts}); re-driving")
    rec["shadow"] = shadow_snapshot(su, reset=False)
    try:
        rec["evidence"] = selection_evidence(su)
    except RpcError as e:
        rec["evidence_error"] = str(e)
        rec["evidence"] = {"n_modifying": 0, "n_observations": 0, "all_observations_matched": False,
                           "by_cell": {}, "paths_recorded": False}
    sn = rec["shadow"] if isinstance(rec["shadow"], Mapping) else {}
    mode = getattr(ctx, "reduction_mode", DEFAULT_REDUCTION_MODE)
    tol = reduction_tolerance(sn, reduction_mode=mode)
    rec["reduction_tolerance"] = tol
    _say(f"    {su.side} shadow ({routing}, {mode} mode): {sn.get('n_groups_compared')} compared, "
         f"{sn.get('n_unequal')} unequal, max |diff| {sn.get('max_abs_diff')}, reduction-op elements "
         f"unequal {tol.get('n_unequal_elems')}/{tol.get('n_elems_compared')} (frac "
         f"{_fmt_frac(tol.get('frac_unequal_elems'))}; row {_fmt_frac(tol.get('max_frac_row'))}, group "
         f"{_fmt_frac(tol.get('max_frac_group'))}), max_ulps_edit {_fmt_g(tol.get('max_ulps_edit'))}"
         + (f", gated |delta gate|/sum|hw| {fmt_gate_units(tol.get('max_gate_diff_rel_sum'))} "
            f"(flips {tol.get('n_gate_relu_flips')}, fired side {fmt_gate_units(tol.get('max_gate_flip_rel_sum'))}; "
            f"gate-edit identity {_fmt_g(tol.get('max_gate_edit_consistency_ulps'))} ulps)"
            if tol.get("gated_keys_unequal") else "")
         + f", contract {'ok' if tol.get('ok') else 'FAIL'}; counted per wave size "
         f"{compared_per_wave_size(rec['shadow_waves'], ctx.batches)}; group rows compared "
         f"{rows_compared(sn)}; per-path modifying: per_slot={rec['evidence'].get('n_modifying_per_slot')} "
         f"grouped={rec['evidence'].get('n_modifying_grouped')}; attested "
         f"{LENS_REDUCTION_MODE_KEY}={rec['evidence'].get(LENS_REDUCTION_MODE_KEY)!r}")
    rec["timing"] = (measure_timing(ctx, su, vm["steered"], prompts[min(4, len(prompts) - 1)][0])
                     if timing else {"skipped": "timing is measured on the production regime of each kind"})
    return rec


def drive_multigroup(ctx: GateContext, su: Standup, *, cells: Sequence[str], prompts,
                     spec: Sequence[Mapping[str, Any]], require_attestation: bool,
                     shadow: bool = False) -> Dict[str, Any]:
    """The multigroup standup: ``cells`` are TrainTest ids, translated through
    the standup's cell_map to the HARNESS ids the routes, the requests, the
    row facts and the SelectionRecord use. Each of ``ctx.repeats`` waves is
    one simultaneous submission of the spec; on the CANDIDATE
    (``require_attestation``) the CoScheduleRecord is reset before and read
    after every wave and an unattested wave is re-driven up to
    COSCHEDULE_ATTEMPTS times — the last attempt's outputs are kept either
    way and the attestation recorded, so an unattested case reads
    INCONCLUSIVE rather than PASS. The reference cannot attest (no RPC on
    the base commit) and is never asked to."""
    vms = vmodels_of(su)
    hmap = harness_map_of(su.endpoint, cells)
    hcells = [hmap[c] for c in cells]
    vm: Dict[str, Optional[str]] = {"passthrough": vms.get("none"), "scale0": vms.get("none__lens0")}
    for h in hcells:
        vm[h] = vms.get(h)
    rec: Dict[str, Any] = {"apply_path": apply_path_of(su), "vmodels": vm, "batches": {},
                           "signature_errors": [], "commit": su.commit,
                           "served_from": su.endpoint.get("served_from"), "cells": list(cells),
                           "harness_cells": hcells, "cell_map": hmap, "coschedule_waves": [],
                           "bi_attested": su.endpoint.get("bi_attested"),
                           "prefix_caching": su.endpoint.get("prefix_caching_check"),
                           "reduction_mode_endpoint": su.endpoint.get("reduction_mode_check")}
    if shadow:
        shadow_snapshot(su, reset=True)
    try:
        rpc(su, "sisteer_selection_record", [True])
    except RpcError as e:
        rec["evidence_error"] = str(e)
    agg: Dict[str, Any] = {"n_modifying": 0, "n_observations": 0, "all_observations_matched": True,
                           "by_cell": {}, "paths_recorded": None, **{k: 0 for k in PATH_COUNTERS},
                           LENS_REDUCTION_MODE_KEY: None, "lens_reduction_modes_seen": []}
    modes_ok = True    # every accepted wave attested one mode, and the same one
    runs = []
    n_attempts = 0
    for repeat in range(ctx.repeats):
        attempts = COSCHEDULE_ATTEMPTS if require_attestation else 1
        for attempt in range(attempts):
            n_attempts += 1
            coschedule_snapshot(su, reset=True)          # reset (absent on the reference: no-op)
            try:
                rpc(su, "sisteer_selection_record", [True])
            except RpcError:
                pass
            try:
                run = run_batch(su.proxy_url, spec_requests(spec, prompts, hcells, vm),
                                max_tokens=ctx.max_tokens, ignore_eos=ctx.ignore_eos,
                                top_logprobs=ctx.top_logprobs)
            except SignatureError as e:
                rec["signature_errors"].append(str(e))
                run = []
            cs = coschedule_snapshot(su, reset=False)
            att = coschedule_attested(cs)
            if shadow:
                # the SHADOW standup's wave is accepted only once the record shows
                # a layer call that compared >= 2 distinct cells (cumulative)
                snap = shadow_snapshot(su, reset=False)
                sh_ok, _ = shadow_multigroup_check(
                    snap, reduction_mode=getattr(ctx, "reduction_mode", DEFAULT_REDUCTION_MODE))
                att = bool(att) and sh_ok is not None
            rec["coschedule_waves"].append({"wave": n_attempts, "repeat": repeat + 1,
                                            "attempt": attempt + 1, "attested": att,
                                            "accepted": bool(not require_attestation or att
                                                             or attempt == attempts - 1),
                                            "record": cs})
            try:
                ev = selection_evidence(su)
            except RpcError as e:
                rec["evidence_error"] = str(e)
                ev = None
            if not require_attestation or att or attempt == attempts - 1:
                runs.append(run)
                if ev is not None:
                    agg["n_modifying"] += ev["n_modifying"]
                    agg["n_observations"] += ev["n_observations"]
                    agg["all_observations_matched"] = (agg["all_observations_matched"]
                                                       and ev["all_observations_matched"])
                    # the per-path counters, summed over accepted waves; a wave
                    # without them (the base commit) makes the aggregate unrecorded
                    if ev.get("paths_recorded") is True and agg["paths_recorded"] is not False:
                        agg["paths_recorded"] = True
                        for k in PATH_COUNTERS:
                            agg[k] = int(agg[k] or 0) + int(ev.get(k) or 0)
                    else:
                        agg["paths_recorded"] = False
                        agg.update({k: None for k in PATH_COUNTERS})
                    for c, b in ev["by_cell"].items():
                        a = agg["by_cell"].setdefault(c, {"n_modifying": 0, "n_applications": 0,
                                                          "n_requests": 0, "scales": []})
                        a["n_modifying"] += b["n_modifying"]
                        a["n_applications"] += b["n_applications"]
                        a["n_requests"] += b["n_requests"]
                        for sc in b["scales"]:
                            if sc not in a["scales"]:
                                a["scales"].append(sc)
                    # the attested reduction mode: every accepted wave must name
                    # one, and the same one (an unattested wave makes the
                    # aggregate None -> the reduction_mode_* check fails)
                    m = ev.get(LENS_REDUCTION_MODE_KEY)
                    for seen in list(ev.get("lens_reduction_modes_seen") or ([] if m is None else [m])):
                        if seen not in agg["lens_reduction_modes_seen"]:
                            agg["lens_reduction_modes_seen"].append(seen)
                    modes_ok = modes_ok and m is not None
                else:
                    agg["all_observations_matched"] = False
                    modes_ok = False
                break
            _say(f"    {su.side} multigroup wave not co-scheduled (attempt {attempt + 1}/{attempts}): "
                 f"{cs}; re-driving")
    rec["batches"][f"multigroup_B{len(spec)}"] = runs
    agg["lens_reduction_modes_seen"].sort()
    agg[LENS_REDUCTION_MODE_KEY] = (agg["lens_reduction_modes_seen"][0]
                                    if modes_ok and len(agg["lens_reduction_modes_seen"]) == 1 else None)
    rec["evidence"] = agg
    # the AGGREGATE over every wave (any attestation counts; per-wave
    # snapshots kept above for provenance)
    rec["coschedule"] = aggregate_coschedule(rec["coschedule_waves"])
    rec["coschedule_attested"] = (any(bool(w["attested"]) for w in rec["coschedule_waves"])
                                  if require_attestation else None)
    rec["coschedule_attempts"] = n_attempts
    if shadow:
        rec["shadow"] = shadow_snapshot(su, reset=False)
        rec["reduction_tolerance"] = reduction_tolerance(
            rec["shadow"], reduction_mode=getattr(ctx, "reduction_mode", DEFAULT_REDUCTION_MODE))
    rec["controls_B1"] = controls_B1(ctx, su, prompts, vm, ["passthrough", "scale0"] + list(hcells))
    _say(f"    {su.side} multigroup B={len(spec)} cells={hcells}: stable={stable(runs)[0]}; "
         f"attested={rec['coschedule_attested']}; evidence by cell: "
         f"{ {c: b['n_modifying'] for c, b in agg['by_cell'].items()} }; attested "
         f"{LENS_REDUCTION_MODE_KEY}={agg.get(LENS_REDUCTION_MODE_KEY)!r}"
         + (f"; shadow: {shadow_multigroup_check(rec['shadow'], reduction_mode=getattr(ctx, 'reduction_mode', DEFAULT_REDUCTION_MODE))[1]}"
            if shadow else ""))
    rec["timing"] = measure_timing(ctx, su, vm[hcells[0]], prompts[min(4, len(prompts) - 1)][0])
    return rec


def build_multigroup_members(ms: ModelServe, resolved: Mapping[str, Mapping[str, Any]], *,
                             bank_dir: Optional[Path], stage_root: Path, header_len: int,
                             dims_fn: Callable[[], Mapping[str, Any]], allow_synthetic: bool
                             ) -> Dict[str, Any]:
    """The multigroup cells and ONE defenses dir holding all of them (symlinks
    to the resolved objects' files): the psr response object, the lora_r4
    response object (else the vector_all response object), and a second psr
    object of ANOTHER SI — a real bank cell whose adapted case is the psr
    response case when one exists, else a seed-1 synthetic under
    safety_refusal — so at least two distinct nonzero objects serve at once."""
    from sisteer.eval.defense_cell import object_path_for
    from sisteer.capeval.petri import defense_cells as dc

    def pick(prefix: str) -> Optional[Mapping[str, Any]]:
        for cid, obj in resolved.items():
            if cid.startswith(prefix) and obj and not obj.get("error"):
                return obj
        return None
    psr = pick("psr:") and pick("psr:relu_gated_vector:gemv:response:skipbos0:")
    lora = pick("lora_all:lora_oproj:rankK:response:skipbos0:") or \
        pick("vector_all:add_vector:vec:response:skipbos0:") or pick("lora:lora_oproj:rank1:response:skipbos0:")
    members = [m for m in (psr, lora) if m]
    if psr:
        psr_case = psr["case_id"]
        primary_si = psr["cell"].split("__")[0]
        second = None
        for si in BANK_SIS:
            if si == primary_si:
                continue
            found = find_bank_cell(bank_dir, "psr_kl_all")
            d = Path(bank_dir) / f"{si}__psr_kl_all" if bank_dir else None
            if d is not None and _cell_dir_has_object(d):
                try:
                    k, ops = adapt_object_dir(d, arm="psr_kl_all", header_len=header_len)
                    if case_id_of(k, ops) == psr_case:
                        second = {"arm": "psr_kl_all", "cell": f"{si}__psr_kl_all",
                                  "defenses_dir": str(bank_dir), "objects": "real",
                                  "source": str(d / "object.pt"), "case_id": psr_case}
                        break
                except Exception:  # noqa: BLE001
                    continue
        if second is None and allow_synthetic:
            si = "safety_refusal" if primary_si != "safety_refusal" else "si_irrelevant"
            cell, ddir = write_synthetic_object("psr_kl_all", dims_fn(), stage_root / "synthetic" / "mg_psr2",
                                                header_len, seed=1, positions="response", si=si)
            second = {"arm": "psr_kl_all", "cell": cell, "defenses_dir": str(ddir),
                      "objects": "synthetic(seed 1)", "source": str(object_path_for(ddir, cell)),
                      "case_id": psr_case}
        if second:
            members.append(second)
    cells = []
    for m in members:
        if m["cell"] not in cells:
            cells.append(m["cell"])
    if len(cells) < 2:
        raise GateError("multigroup needs at least two distinct nonzero cells; resolved: "
                        f"{[m['cell'] for m in members]}")
    mg_dir = stage_root / "multigroup"
    for m in members:
        src_dir = Path(m["source"]).parent
        dst = mg_dir / m["cell"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob("object.*"):
            link = dst / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f.resolve())
    return {"cells": cells, "defenses_dir": str(mg_dir), "members": members}


def compare_case(ctx: GateContext, ref: Mapping[str, Any], cand: Mapping[str, Any],
                 pt: Optional[Mapping[str, Any]], legacy: Optional[Mapping[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Verdicts: ``pure_B1`` is HARD (reference vs candidate, one request per
    step); every B > 1 comparison (pure_B<n>, mixed, control rows vs the
    oracle, bites) is EVIDENCE. The B=1 singleton controls are recorded as
    facts for :func:`case_checks`."""
    verdicts: Dict[str, Any] = {}
    if 1 not in [int(b) for b in ctx.batches]:
        raise GateError(f"the batch ladder {tuple(ctx.batches)} omits B=1 — the hard reference-vs-"
                        "candidate comparison is not configurable away (gate_batches always "
                        "prepends it); refusing to compare without it")
    for B in ctx.batches:
        key = f"pure_B{B}"
        v = (verdict if int(B) == 1 else evidence_verdict)(ref["batches"][key], cand["batches"][key])
        verdicts[key] = v
    B = max(ctx.batches)
    mkey = f"mixed_B{B}"
    verdicts[mkey] = (verdict if B == 1 else evidence_verdict)(ref["batches"][mkey], cand["batches"][mkey])
    if legacy is not None:
        for key in list(verdicts):
            fn = verdict if key == "pure_B1" else evidence_verdict
            verdicts[f"legacy_ab_{key}"] = fn(legacy["batches"][key], cand["batches"][key])
    out: Dict[str, Any] = {"verdicts": verdicts, "mixed_B": B, "evidence_B_gt_1": {}}
    roles = ref.get("mixed_roles") or mixed_roles(B)
    pt_runs = (pt or {}).get("batches", {}).get(f"passthrough_B{B}") or []
    oracle = pt_runs[0] if pt_runs and pt_runs[0] else None
    pt_b1_runs = (pt or {}).get("batches", {}).get("passthrough_B1") or []
    oracle_b1 = pt_b1_runs[0][0] if pt_b1_runs and pt_b1_runs[0] else None
    ctrl_idx = [i for i, r in enumerate(roles) if r in ("passthrough", "scale0")]
    steer_idx = [i for i, r in enumerate(roles) if r == "steered"]
    out["controls_B1"] = {}
    for side, rec in ((REFERENCE, ref), (CANDIDATE, cand)):
        mixed_runs = rec["batches"].get(mkey) or []
        mixed = mixed_runs[0] if mixed_runs and mixed_runs[0] else None
        ev: Dict[str, Any] = {"note": "B>1 vs the oracle: composition-confounded evidence"}
        if oracle is not None and mixed is not None and len(mixed) == len(oracle):
            ok, why = signatures_equal([mixed[i] for i in ctrl_idx], [oracle[i] for i in ctrl_idx])
            ev["control_rows_match"] = ok
            if not ok:
                ev["control_rows_detail"] = why
            ev["steering_bites"] = signatures_differ_somewhere(
                [mixed[i] for i in steer_idx], [oracle[i] for i in steer_idx])
        out["evidence_B_gt_1"][side] = ev
        out["controls_B1"][side] = controls_B1_facts(rec.get("controls_B1") or {}, oracle_b1, ["steered"])
    if pt_b1_runs and len(pt_b1_runs) >= 2:
        st_ok, st_why = stable(pt_b1_runs)
        out["oracle_B1_stable"] = {"stable": st_ok, "detail": st_why}
    return out


def run_gate(ctx: GateContext, *, cases: Mapping[str, Mapping[str, Any]],
             only_cases: Optional[Sequence[str]], bank_dir: Optional[Path],
             extra_dirs: Sequence[Path], candsi_dir: Optional[Path],
             allow_synthetic: bool,
             extra_origins: Optional[Mapping[Path, str]] = None) -> Dict[str, Any]:
    ms = ctx.ms
    notes = {k: v for k, v in cases.items() if k in INVENTORY_NOTES}
    cases = {k: v for k, v in cases.items() if k not in INVENTORY_NOTES}
    trees = {"candidate": tree_facts(ctx.candidate), "reference": tree_facts(ctx.reference)}
    doc: Dict[str, Any] = {
        "model": dataclasses.asdict(ms),
        "geometry": {"max_model_len": ctx.max_model_len, "max_num_seqs": ctx.max_num_seqs},
        "trees": trees, "reference_sha": ctx.reference_sha,
        "batches": list(ctx.batches), "timing_batches": list(ctx.timing_batches),
        "max_tokens": ctx.max_tokens, "repeats": ctx.repeats, "ignore_eos": ctx.ignore_eos,
        "legacy_ab": ctx.legacy_ab, "batch_invariant": int(ctx.batch_invariant),
        "bi0_probe": bool(ctx.bi0_probe), "shadow_env": SHADOW_ENV,
        "reduction_mode": check_reduction_mode(ctx.reduction_mode), "stacked_env": STACKED_ENV,
        "reduction_contract": {"text": REDUCTION_CONTRACT, "frac_tol": REDUCTION_FRAC_TOL,
                               "group_frac_tol": REDUCTION_GROUP_FRAC_TOL, "row_frac_tol": REDUCTION_ROW_FRAC_TOL,
                               "ulps_edit_tol": REDUCTION_ULPS_EDIT_TOL,
                               "b1_exact": "pure_B1 / controls_B1 / steering_bites_B1 stay exact in "
                                           "either mode: a one-row group is one call"},
        "node": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": (_nvidia_smi("--query-gpu=name,memory.total", "--format=csv,noheader") or "").strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inventory": {k: dict(v, routing=case_routing(k)) for k, v in cases.items()},
        "inventory_notes": notes, "cases": [],
        "only_cases": list(only_cases) if only_cases else None,
        "uncovered_cases": list(cases) + [MULTIGROUP], "complete": False}
    path_json = ctx.results_dir / "lens_vectorize_gate.json"
    _write_json(path_json, doc)
    refusal = check_trees(trees["candidate"], trees["reference"], reference_sha=ctx.reference_sha)
    if refusal:
        doc["tree_error"] = refusal
        doc["exit_code"] = overall_exit_code(doc)
        _write_json(path_json, doc)
        (ctx.results_dir / "report.md").write_text(render_report(doc))
        _say(f"TREE REFUSAL: {refusal}")
        return doc
    cand_sha, ref_sha = trees["candidate"]["sha"], trees["reference"]["sha"]

    stage_root = ctx.results_dir / "objects"
    stage_root.mkdir(parents=True, exist_ok=True)
    empty = stage_root / "no_objects"
    empty.mkdir(exist_ok=True)
    header_len = header_len_for(ms.hf, ms.rev)
    doc["kind_coverage"] = kind_coverage(header_len)
    _write_json(path_json, doc)
    dims_cache: Dict[str, Any] = {}

    def dims_fn():
        if not dims_cache:
            dims_cache.update(model_dims(ms.hf, ms.rev))
        return dims_cache

    # THE DECLARED-RENDER CASES (module docstring): opt-in by name, on their
    # own families only (a mis-targeted selection refused at startup, in main)
    declared_order = declared_render_order(ms, only_cases)
    doc["declared_render_cases"] = list(declared_order)
    doc["uncovered_cases"] += list(declared_order)
    _write_json(path_json, doc)

    # 1. the passthrough-only oracle + floor (the candidate tree; its rows must
    #    equal both sides' control rows — a difference on either side fails).
    #    A run naming declared-render cases ONLY needs no oracle — its
    #    comparisons are within one standup (the steered leg against the
    #    scale-0 twin under the same declared render) — and skips it.
    B_max = max(ctx.batches)
    mg_spec = multigroup_spec(B_max)
    doc["multigroup_spec"] = mg_spec
    pt_rec: Optional[Dict[str, Any]] = None
    if declared_render_only(only_cases):
        doc["passthrough_skipped"] = "declared-render cases only: no passthrough oracle is needed"
        _say(doc["passthrough_skipped"])
        for case_id in declared_order:
            doc["cases"].append(_declared_render_case(ctx, case_id, cand_sha=cand_sha, bank_dir=bank_dir,
                                                      header_len=header_len, uncovered=doc["uncovered_cases"]))
            _write_json(path_json, doc)
        doc["complete"] = not doc["uncovered_cases"]
        doc["exit_code"] = overall_exit_code(doc)
        doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(path_json, doc)
        (ctx.results_dir / "report.md").write_text(render_report(doc))
        return doc
    prompts = None
    su = None
    try:
        su = start_standup(ctx, side=PASSTHROUGH, tree=ctx.candidate, expect_sha=cand_sha,
                           cell="none", defenses_dir=empty, tag="passthrough",
                           reduction_mode=ctx.reduction_mode)
        vms = vmodels_of(su)
        prompts = build_prompts(su.proxy_url, vms["none"], prompt_targets(B_max))
        doc["prompt_tokens"] = [c for _m, c in prompts]
        pt_rec = drive_standup(ctx, su, cell="none", prompts=prompts, passthrough_only=True,
                               mg_spec=mg_spec)
        doc["passthrough_timing"] = pt_rec["timing"]
        if pt_rec["signature_errors"]:
            doc["passthrough_error"] = f"signature errors: {pt_rec['signature_errors'][0]}"
    except Exception as e:  # noqa: BLE001
        doc["passthrough_error"] = f"{type(e).__name__}: {e}"
        doc["passthrough_traceback"] = traceback.format_exc()[-3000:]
        _say(f"passthrough standup FAILED: {type(e).__name__}: {str(e)[:300]}")
    finally:
        stop_standup(su)
    _write_json(path_json, doc)
    if prompts is None:
        doc["exit_code"] = overall_exit_code(doc)
        _write_json(path_json, doc)
        (ctx.results_dir / "report.md").write_text(render_report(doc))
        return doc

    order = gate_case_order(list(cases), only_cases)
    resolved: Dict[str, Mapping[str, Any]] = {}
    # the declared-render cases named beside inventory cases run FIRST, right
    # after the oracle (a plumbing fault surfaces in minutes, as with
    # multigroup); each is its own standup set and nothing here reads the oracle
    for case_id in declared_order:
        doc["cases"].append(_declared_render_case(ctx, case_id, cand_sha=cand_sha, bank_dir=bank_dir,
                                                  header_len=header_len, uncovered=doc["uncovered_cases"]))
        _write_json(path_json, doc)

    def _resolve(case_id: str) -> Mapping[str, Any]:
        if case_id not in resolved:
            resolved[case_id] = resolve_case_object(
                ms, case_id, list(cases[case_id]["arms"]), bank_dir=bank_dir, extra_dirs=extra_dirs,
                candsi_dir=candsi_dir, stage_root=stage_root, header_len=header_len,
                dims_fn=dims_fn, allow_synthetic=allow_synthetic, extra_origins=extra_origins)
        return resolved[case_id]

    for case_id in order:
        if case_id == MULTIGROUP:
            # 2. the multigroup case FIRST (mandatory): distinct nonzero cells
            #    in one batch, co-scheduling attested by the candidate worker
            kr: Dict[str, Any] = {"case_id": MULTIGROUP, "kind": MULTIGROUP, "arms": [],
                                  "timing_measured": True}
            try:
                for cid in cases:
                    if cid.startswith(("psr:relu_gated_vector:gemv:response:skipbos0:",
                                       "lora_all:lora_oproj:rankK:response:skipbos0:",
                                       "vector_all:add_vector:vec:response:skipbos0:",
                                       "lora:lora_oproj:rank1:response:skipbos0:")):
                        try:
                            _resolve(cid)
                        except Exception as e:  # noqa: BLE001 — that case will report it itself
                            _say(f"multigroup member case {cid} did not resolve: {e}")
                mg = build_multigroup_members(ms, resolved, bank_dir=bank_dir, stage_root=stage_root,
                                              header_len=header_len, dims_fn=dims_fn,
                                              allow_synthetic=allow_synthetic)
                kr.update({"cells": mg["cells"], "members": mg["members"],
                           "defenses_dir": mg["defenses_dir"],
                           "objects": ", ".join(f"{m['cell']}={m['objects']}" for m in mg["members"]),
                           "arm": "+".join(m["arm"] for m in mg["members"])})
                _say(f"case {MULTIGROUP}: cells {mg['cells']}")
                recs: Dict[str, Any] = {}
                # (side, tree, sha, env_extra, reduction mode): the base-commit
                # reference does not read the mode switch (scrubbed, None); every
                # candidate-tree standup is SET to the gate's mode explicitly
                plan = [(REFERENCE, ctx.reference, ref_sha, None, None),
                        (CANDIDATE, ctx.candidate, cand_sha, None, ctx.reduction_mode),
                        (SHADOW, ctx.candidate, cand_sha, {SHADOW_ENV: "1"}, ctx.reduction_mode)]
                if ctx.legacy_ab:
                    plan.append((LEGACY, ctx.candidate, cand_sha, {PER_ROW_ENV: "1"}, ctx.reduction_mode))
                for side, tree, sha, env_extra, rmode in plan:
                    su = None
                    try:
                        su = start_standup(ctx, side=side, tree=tree, expect_sha=sha, cell=mg["cells"],
                                           defenses_dir=Path(mg["defenses_dir"]),
                                           tag=f"{MULTIGROUP}__{side}", env_extra=env_extra,
                                           reduction_mode=rmode)
                        recs[side] = drive_multigroup(ctx, su, cells=mg["cells"], prompts=prompts,
                                                      spec=mg_spec,
                                                      require_attestation=(side in (CANDIDATE, SHADOW)),
                                                      shadow=(side == SHADOW))
                    finally:
                        stop_standup(su)
                if ctx.bi0_probe and int(ctx.batch_invariant) == 1:
                    # EVIDENCE ONLY, and only meaningful under the opt-in BI=1
                    # EXPERIMENT: the candidate back at production BI=0 — repeat
                    # stability, co-scheduling and timing; never a verdict. At the
                    # default BI=0 every standup already IS this regime.
                    su = None
                    try:
                        su = start_standup(ctx, side=CANDIDATE, tree=ctx.candidate, expect_sha=cand_sha,
                                           cell=mg["cells"], defenses_dir=Path(mg["defenses_dir"]),
                                           tag=f"{MULTIGROUP}__bi0_probe", batch_invariant=0,
                                           reduction_mode=ctx.reduction_mode)
                        probe = drive_multigroup(ctx, su, cells=mg["cells"], prompts=prompts,
                                                 spec=mg_spec, require_attestation=True)
                        pkey = f"multigroup_B{len(mg_spec)}"
                        st_ok, st_why = stable(probe["batches"][pkey])
                        kr["bi0_probe"] = {"batch_invariant": 0, "repeat_stable": st_ok,
                                           "repeat_detail": st_why,
                                           "coschedule_attested": probe.get("coschedule_attested"),
                                           "coschedule": probe.get("coschedule"),
                                           "timing": probe.get("timing"),
                                           "evidence": probe.get("evidence"),
                                           "note": "evidence only — BI=0 outputs are composition-"
                                                   "dependent; no verdict is derived from this standup"}
                    except Exception as e:  # noqa: BLE001 — evidence only
                        kr["bi0_probe"] = {"error": f"{type(e).__name__}: {e}"}
                    finally:
                        stop_standup(su)
                key = f"multigroup_B{len(mg_spec)}"
                hcells = recs[CANDIDATE]["harness_cells"]
                kr["harness_cells"] = hcells
                kr["cell_map"] = recs[CANDIDATE]["cell_map"]
                kr["apply_path"] = {s_: recs[s_]["apply_path"] for s_ in recs}
                kr["steering_evidence"] = {s_: recs[s_].get("evidence") for s_ in recs}
                kr["coschedule"] = {s_: recs[s_].get("coschedule") for s_ in recs}
                kr["coschedule_waves"] = {s_: recs[s_].get("coschedule_waves") for s_ in recs}
                kr["coschedule_attempts"] = recs[CANDIDATE].get("coschedule_attempts")
                kr["coschedule_attested"] = recs[CANDIDATE].get("coschedule_attested")
                kr["shadow"] = recs[SHADOW].get("shadow")
                kr["reduction_mode"] = ctx.reduction_mode
                kr["reduction_tolerance"] = reduction_tolerance(kr["shadow"], reduction_mode=ctx.reduction_mode)
                kr["shadow_coschedule"] = recs[SHADOW].get("coschedule")
                kr["bi_attested"] = {s_: recs[s_].get("bi_attested") for s_ in recs}
                kr["prefix_caching"] = {s_: recs[s_].get("prefix_caching") for s_ in recs}
                kr["timing"] = {s_: recs[s_]["timing"] for s_ in recs}
                kr["commits"] = {s_: recs[s_]["commit"] for s_ in recs}
                kr["signature_errors"] = [f"{s_}: {e}" for s_ in recs
                                          for e in recs[s_]["signature_errors"]]
                # B=32 cross-server equality: EVIDENCE (module docstring)
                verdicts = {key: evidence_verdict(recs[REFERENCE]["batches"][key],
                                                  recs[CANDIDATE]["batches"][key])}
                if LEGACY in recs:
                    verdicts[f"legacy_ab_{key}"] = evidence_verdict(recs[LEGACY]["batches"][key],
                                                                    recs[CANDIDATE]["batches"][key])
                kr["verdicts"] = verdicts
                kr["mixed_B"] = len(mg_spec)
                oracle_runs = (pt_rec or {}).get("batches", {}).get("multigroup_oracle") or []
                oracle = oracle_runs[0] if oracle_runs and oracle_runs[0] else None
                pt_b1 = (pt_rec or {}).get("batches", {}).get("passthrough_B1") or []
                oracle_b1 = pt_b1[0][0] if pt_b1 and pt_b1[0] else None
                kr["row_facts"] = {}
                kr["controls_B1"] = {}
                for side in (REFERENCE, CANDIDATE):
                    runs = recs[side]["batches"][key]
                    mixed = runs[0] if runs and runs[0] else None
                    facts = (multigroup_row_facts(mg_spec, hcells, mixed, oracle)
                             if mixed is not None and oracle is not None else {"control_rows_match": None})
                    facts["note"] = "B>1 row facts vs the oracle / within the batch: evidence"
                    kr["row_facts"][side] = facts
                    kr["controls_B1"][side] = controls_B1_facts(recs[side].get("controls_B1") or {},
                                                                oracle_b1, list(hcells))
                kr["checks"] = multigroup_checks(kr, mg_spec, hcells, reference_sha=ref_sha,
                                                 base_sha=ctx.reference_sha,
                                                 have_oracle=oracle_b1 is not None,
                                                 reduction_mode=ctx.reduction_mode)
                kr["raw"] = {s_: recs[s_]["batches"] for s_ in recs}
                doc["uncovered_cases"].remove(MULTIGROUP)
                _say(f"case {MULTIGROUP}: " + ", ".join(f"{k}={v['verdict']}" for k, v in verdicts.items())
                     + "; checks " + ", ".join(f"{k}={'ok' if c['ok'] else ('inconclusive' if c['ok'] is None else 'FAIL')}"
                                               for k, c in kr["checks"].items()))
            except Exception as e:  # noqa: BLE001
                kr["error"] = f"{type(e).__name__}: {e}"
                kr["traceback"] = traceback.format_exc()[-3000:]
                _say(f"case {MULTIGROUP} FAILED: {type(e).__name__}: {str(e)[:300]}")
            doc["cases"].append(kr)
            _write_json(path_json, doc)
            continue

        # 3. a single-op case: reference (base tree), candidate, optionally the legacy A/B
        arms = list(cases[case_id]["arms"])
        kr = {"case_id": case_id, "kind": cases[case_id]["kind"], "arms": arms,
              "timing_measured": timing_for_case(case_id), "routing": case_routing(case_id)}
        try:
            obj = _resolve(case_id)
            kr.update(obj)
            _say(f"case {case_id}: arm {obj['arm']} cell {obj['cell']} objects={obj['objects']} "
                 f"from {obj['source']}")
            recs = {}
            plan = [(REFERENCE, ctx.reference, ref_sha, None, None),
                    (CANDIDATE, ctx.candidate, cand_sha, None, ctx.reduction_mode),
                    (SHADOW, ctx.candidate, cand_sha, {SHADOW_ENV: "1"}, ctx.reduction_mode)]
            if ctx.legacy_ab:
                plan.append((LEGACY, ctx.candidate, cand_sha, {PER_ROW_ENV: "1"}, ctx.reduction_mode))
            for side, tree, sha, env_extra, rmode in plan:
                su = None
                try:
                    su = start_standup(ctx, side=side, tree=tree, expect_sha=sha, cell=obj["cell"],
                                       defenses_dir=Path(obj["defenses_dir"]),
                                       tag=f"{case_id.replace(':', '_')}__{side}", env_extra=env_extra,
                                       reduction_mode=rmode)
                    if side == SHADOW:
                        recs[side] = drive_shadow(ctx, su, cell=obj["cell"], prompts=prompts,
                                                  timing=timing_for_case(case_id),
                                                  routing=kr["routing"])
                    else:
                        recs[side] = drive_standup(ctx, su, cell=obj["cell"], prompts=prompts,
                                                   timing=timing_for_case(case_id))
                finally:
                    stop_standup(su)
            # 4. THE SHARED-PREFILL CASE (module docstring): the candidate once
            #    more with prefix caching ON, driven directly at vLLM; skipped
            #    (not applicable) where caching is structurally OFF
            if ctx.resumed_prefill:
                applicable, why = resumed_case_applicable(ms)
                if not applicable:
                    kr["resumed_prefill"] = {"skipped": why}
                    _say(f"case {case_id}: shared-prefill case not applicable — {why}")
                else:
                    policy = case_resumed_policy(obj, header_len=header_len)
                    su = None
                    try:
                        su = start_standup(ctx, side=RESUMED, tree=ctx.candidate, expect_sha=cand_sha,
                                           cell=obj["cell"], defenses_dir=Path(obj["defenses_dir"]),
                                           tag=f"{case_id.replace(':', '_')}__{RESUMED}",
                                           reduction_mode=ctx.reduction_mode,
                                           prefix_caching=PREFIX_CACHING_DEFAULT)
                        kr["resumed_prefill"] = drive_resumed_prefill(ctx, su, cell=obj["cell"], prompts=prompts,
                                                                      policy=policy)
                    except Exception as e:  # noqa: BLE001 — recorded on the case, judged by the check
                        kr["resumed_prefill"] = {"error": f"{type(e).__name__}: {e}", "policy": policy,
                                                 "traceback": traceback.format_exc()[-3000:]}
                        _say(f"case {case_id}: shared-prefill case ERROR: {type(e).__name__}: {str(e)[:300]}")
                    finally:
                        stop_standup(su)
            kr["apply_path"] = {s_: recs[s_]["apply_path"] for s_ in recs}
            kr["harness_cell"] = recs[CANDIDATE].get("harness_cell")
            kr["shadow"] = recs[SHADOW].get("shadow")
            kr["shadow_waves"] = recs[SHADOW].get("shadow_waves")
            kr["reduction_mode"] = ctx.reduction_mode
            kr["reduction_tolerance"] = reduction_tolerance(kr["shadow"], reduction_mode=ctx.reduction_mode)
            kr["batches"] = list(ctx.batches)
            kr["bi_attested"] = {s_: recs[s_].get("bi_attested") for s_ in recs}
            kr["prefix_caching"] = {s_: recs[s_].get("prefix_caching") for s_ in recs}
            kr["steering_evidence"] = {s_: recs[s_].get("evidence") for s_ in recs if "evidence" in recs[s_]}
            kr["timing"] = {s_: recs[s_]["timing"] for s_ in recs}
            kr["commits"] = {s_: recs[s_]["commit"] for s_ in recs}
            kr["served_from"] = {s_: recs[s_]["served_from"] for s_ in recs}
            kr["signature_errors"] = [f"{s_}: {e}" for s_ in recs for e in recs[s_]["signature_errors"]]
            kr.update(compare_case(ctx, recs[REFERENCE], recs[CANDIDATE], pt_rec,
                                   legacy=recs.get(LEGACY)))
            kr["checks"] = case_checks(kr, reference_sha=ref_sha, base_sha=ctx.reference_sha,
                                       have_oracle=pt_rec is not None
                                       and not pt_rec["signature_errors"],
                                       reduction_mode=ctx.reduction_mode)
            kr["raw"] = {s_: recs[s_].get("batches") for s_ in recs}
            doc["uncovered_cases"].remove(case_id)
            _say(f"case {case_id}: " + ", ".join(
                f"{k}={v['verdict']}" + (f"(equal={v.get('equal')})" if v.get("evidence") else "")
                for k, v in kr["verdicts"].items())
                 + "; checks " + ", ".join(
                     f"{k}={'ok' if c['ok'] else ('inconclusive' if c['ok'] is None else 'FAIL')}"
                     for k, c in kr["checks"].items()))
        except Exception as e:  # noqa: BLE001
            kr["error"] = f"{type(e).__name__}: {e}"
            kr["traceback"] = traceback.format_exc()[-3000:]
            _say(f"case {case_id} FAILED: {type(e).__name__}: {str(e)[:300]}")
        doc["cases"].append(kr)
        _write_json(path_json, doc)

    doc["complete"] = not doc["uncovered_cases"]
    doc["exit_code"] = overall_exit_code(doc)
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(path_json, doc)
    (ctx.results_dir / "report.md").write_text(render_report(doc))
    return doc


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--phase", choices=("env", "inventory", "gate"), default="gate")
    ap.add_argument("--repo-dir", default=str(REPO), help="the CANDIDATE checkout")
    ap.add_argument("--reference-repo", default=None,
                    help="a SEPARATE checkout of the base commit (--reference-sha); the "
                         "reference server runs entirely from it. REQUIRED for --phase gate")
    ap.add_argument("--reference-sha", default=DEFAULT_REFERENCE_SHA)
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--objects-root", default=DEFAULT_OBJECTS_ROOT)
    ap.add_argument("--candsi-root", default=DEFAULT_CANDSI_ROOT)
    ap.add_argument("--extra-objects-dir", action="append", default=None, metavar="PATH[=FAMILY]",
                    help="additional bank dirs searched after the scale-12 bank, each optionally "
                         "keyed by the gate family its objects were fit on (default: stage3-wave/A "
                         "and /C =llama8b — the stage-3 wave's Llama-3.1-8B fits). An object serves "
                         "only the family its own fit-time record (fit_provenance.json / "
                         "train_budget_meta.json / save sidecar) names; the key cross-checks that "
                         "record and stands in for it only when the object carries none")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--cases", default=None,
                    help="comma list of inventory case ids (and/or 'multigroup', and/or the "
                         f"declared-render cases {list(DECLARED_RENDER_CASES)} — opt-in, on their own "
                         "families only) to run (default all inventory cases + multigroup); a partial "
                         "run is reported INCOMPLETE (exit 3)")
    ap.add_argument("--batch-invariant", type=int, choices=(0, 1), default=0,
                    help="VLLM_BATCH_INVARIANT for every standup: 0 (default) = the production "
                         "regime the gate certifies (the identity check is the in-worker shadow "
                         "compare); 1 = an opt-in EXPERIMENT (module docstring: BI=1 changes the "
                         "very kernels under test — never the authoritative setting)")
    ap.add_argument("--bi0-probe", action="store_true",
                    help="under the opt-in --batch-invariant 1 experiment only: also stand the "
                         "candidate up at production BI=0 for the multigroup case and record repeat "
                         "stability / co-scheduling / timing as EVIDENCE ONLY (a no-op at the "
                         "default BI=0, where every standup already runs that regime)")
    ap.add_argument("--legacy-ab", action="store_true",
                    help="also stand the candidate up with SISTEER_LENS_PER_ROW_APPLY=1 per case "
                         "(a secondary A/B against the candidate's vectorized path)")
    ap.add_argument("--reduction-mode", choices=list(REDUCTION_MODES), default=None,
                    help="the reduction mode every candidate-tree standup is SET to "
                         f"({STACKED_ENV}=1 / 0, never inherited) and judged under (module docstring, "
                         f"THE REDUCTION CONTRACT): stacked (the served default) or per_row (exact for "
                         f"every op). Default: the {REDUCTION_MODE_ENV} env seam (stacked|per_row|1|0), "
                         f"else {DEFAULT_REDUCTION_MODE}")
    ap.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES),
                    help="the B>1 EVIDENCE sizes of the ladder (comma list of positive ints). B=1 "
                         "— the hard cross-server verdict — ALWAYS runs and cannot be removed here "
                         "(gate_batches); a malformed entry refuses at startup")
    ap.add_argument("--timing-batches", default=",".join(str(b) for b in DEFAULT_TIMING_BATCHES))
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--no-resumed-prefill", action="store_true",
                    help="skip THE SHARED-PREFILL CASE (the caching-ON candidate_resumed standup per case); "
                         "its `resumed_prefill` check then reads INCONCLUSIVE (exit 3), never a pass")
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--measure-s", type=float, default=45.0)
    ap.add_argument("--gpu-mem-util", default="0.90")
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--manifest", default="configs/capeval/manifests/petri_twosi.yaml")
    args = ap.parse_args(argv)
    reduction_mode = (check_reduction_mode(args.reduction_mode) if args.reduction_mode
                      else parse_reduction_mode_spec(os.environ.get(REDUCTION_MODE_ENV)))

    ms = MODELS[args.model]
    repo = Path(args.repo_dir).resolve()
    if args.phase == "env":
        print("\n".join(shell_env_lines(ms, repo)))
        return 0
    user = os.environ.get("USER", "user")
    bank_dir = Path(root_path(args.objects_root, user)) / ms.bank
    candsi_dir = Path(root_path(args.candsi_root, user)) / ms.key
    extra, extra_origins = parse_extra_objects(args.extra_objects_dir, user)
    try:
        header_len = header_len_for(ms.hf, ms.rev)
    except Exception as e:  # noqa: BLE001 — off-cluster: no tokenizer cache; the probe needs only a width
        header_len = 4
        if args.phase == "gate":
            raise
        print(f"# note: tokenizer unavailable ({type(e).__name__}); inventory probed with header width 4",
              file=sys.stderr)
    inventory = plan_inventory(ms, bank_dir=bank_dir, extra_dirs=extra, candsi_dir=candsi_dir,
                               header_len=header_len, extra_origins=extra_origins)
    if args.phase == "inventory":
        listing = {k: v for k, v in inventory.items() if k not in INVENTORY_NOTES}
        print(json.dumps({"model": ms.key, "cases": listing,
                          "kind_coverage": kind_coverage(header_len),
                          "mandatory": list(listing) + [MULTIGROUP],
                          "timing_measured": [c for c in list(listing) + [MULTIGROUP] if timing_for_case(c)],
                          "unadaptable_real_artifacts": inventory.get("__unadaptable__"),
                          "foreign_real_artifacts": inventory.get("__foreign__"),
                          "extra_objects": {str(d): extra_origins.get(d) for d in extra}}, indent=2))
        return 0
    if not args.reference_repo:
        raise SystemExit("--reference-repo is REQUIRED: the reference standup runs from a separate "
                         f"checkout of the base commit {args.reference_sha}")
    only = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    try:
        declared_render_order(ms, only)   # a declared-render case named for another family refuses NOW
    except GateError as e:
        raise SystemExit(str(e))
    results_dir = Path(root_path(args.results_root, user)) / ms.key
    if only:
        results_dir = results_dir / "subset" / subset_slug(only)
    results_dir.mkdir(parents=True, exist_ok=True)
    ctx = GateContext(
        ms=ms, candidate=repo, reference=Path(args.reference_repo).resolve(),
        reference_sha=args.reference_sha, results_dir=results_dir,
        max_model_len=int(args.max_model_len), max_num_seqs=int(args.max_num_seqs),
        gpu_mem_util=str(args.gpu_mem_util), ready_timeout_s=int(args.ready_timeout_s),
        batches=gate_batches(args.batches),
        timing_batches=parse_int_list(args.timing_batches, "--timing-batches"),
        max_tokens=int(args.max_tokens), repeats=max(2, int(args.repeats)),
        top_logprobs=int(args.top_logprobs), ignore_eos=not args.no_ignore_eos,
        warmup_s=float(args.warmup_s), measure_s=float(args.measure_s),
        manifest=args.manifest, legacy_ab=bool(args.legacy_ab),
        batch_invariant=int(args.batch_invariant), bi0_probe=bool(args.bi0_probe),
        reduction_mode=reduction_mode, resumed_prefill=not args.no_resumed_prefill)
    _say(f"reduction mode: {reduction_mode} ({STACKED_ENV}={'1' if reduction_mode == REDUCTION_MODE_STACKED else '0'} "
         f"on every candidate-tree standup; contract: {REDUCTION_CONTRACT})")
    if only:
        _say(f"SUBSET run: cases {only} -> {results_dir} (complete=False by construction; exit 3 "
             "whatever passes — never a certification)")
    _say(f"inventory: {len([k for k in inventory if k not in INVENTORY_NOTES])} served-op cases + "
         f"{MULTIGROUP}: {[k for k in inventory if k not in INVENTORY_NOTES]}")
    for src in (inventory.get("__foreign__") or {}).get("sources") or []:
        _say(f"foreign real artifact (another family's object; never served here): {src}")
    doc = run_gate(ctx, cases=inventory, only_cases=only, bank_dir=bank_dir, extra_dirs=extra,
                   candsi_dir=candsi_dir, allow_synthetic=not args.no_synthetic,
                   extra_origins=extra_origins)
    _say(f"done: {results_dir / 'report.md'} (exit {doc.get('exit_code')})")
    rp = results_dir / "report.md"
    print(rp.read_text() if rp.is_file() else json.dumps(doc, indent=1, default=str)[:4000])
    return int(doc.get("exit_code", 3))


if __name__ == "__main__":
    sys.exit(main())
