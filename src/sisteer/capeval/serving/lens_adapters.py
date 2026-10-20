"""Per-kind lens adapters: the landed object representation ->
vllm-lens hook operations (Phase-2 serving).

An adapter translates ONE object kind's on-disk artifact (the exact bytes
the intervention hash covers) into the immutable :class:`HookOp` sequence
the worker-resident hook applies per generated token. Adapters run at
SERVER START inside each vLLM worker (via ``lens_steer.
WorkerObjectRegistry``) — never at request time, and never in the proxy
process.

THE AUTHORITATIVE STATEMENT OF WHAT EACH ARM DOES is ``SEMANTICS.md``,
published beside the objects themselves (both clusters, under the
jailbreak lane's object directories; also ``docs/OBJECT_SEMANTICS.md``). It is
written from the ``apply`` hooks rather than the field names, and it
leads with the axis every confusion in this family has been on: WHAT IS
THE EDIT COMPUTED FROM — a fixed vector, the last prompt token, or the
running stream at that position. Cite it; do not re-derive an arm's
semantics from a blob field. Three traps have already come from the
latter, and each one is a field that says something true which a
reasonable reader takes to mean something else: a ``kind`` string that
names the family rather than the arm, legacy scalars left PRESENT BUT
NULL beside populated plural ones, and a ``positions`` string whose
value names a MODE rather than a mask. If an adapter here ever
disagrees with that file, the file is the contract and this code is the
bug.

THE ADAPTER CONTRACT (binding on every real-kind adapter):

- signature ``adapt(resolved_spec, artifact_bytes, sidecar_bytes) ->
  tuple[HookOp, ...]`` — the SIDECAR bytes are part of the hashed object
  identity (``petri_intervention_hash`` covers them), so a
  behavior-bearing sidecar must reach the adapter: a kind whose
  ``methods.load_object`` consumes sidecar content consumes it here too
  (antidote's SteeringVector JSON), and a kind that takes none REFUSES
  non-empty sidecar bytes (two objects differing only in sidecars must
  never produce identical ops).
- PURE and DETERMINISTIC: same inputs, same ops — no I/O beyond the given
  bytes, no globals, no randomness (loreft's orthogonal-parametrization
  rebuild runs under a forked RNG so even torch's global generator state
  is untouched).
- validates everything it reads (layer indices, tensor shapes/dtypes,
  spec fields, the sidecar) and REFUSES loudly on anything unexpected —
  an unadapted or malformed kind is ABSENT / a §10 gate failure
  upstream, never a guessed translation.
- bakes the object's own trained parameters (its internal scale, layer,
  positions) INTO the ops (``HookOp.coeff`` carries a calibrated
  base scale — antidote's ``base_alpha``, alphasteer's ``base_scale``);
  the runtime ``scale`` multiplier that ``apply_hook_ops`` applies on top
  is the SELECTOR's scale (1.0 for a served arm, 0.0 for the
  ``none__lens0`` no-op leg) — the two must never be conflated.
- registered via :func:`register_adapter` under the exact kind string
  from ``methods.py`` (each object class's own ``kind`` attribute;
  ``tests/test_petri_lens_adapters.py`` pins the correspondence against
  the real classes).

THE OP VOCABULARY (v2 — the real-kind extension of the v1 single-op
``add_vector``). Every op names its decoder LAYER; the hook site is the
BLOCK OUTPUT (the residual stream after decoder layer ``layer`` — the
repository's block-output convention, DESIGN.md §4.2):

- :data:`OP_ADD_VECTOR` — ``h += eff · v`` (``eff = selector_scale ·
  coeff``), with a per-op position mode (:data:`POSITIONS_ALL` /
  :data:`POSITIONS_LAST`), an independent BOS-skip mask, and a
  prefill/decode phase. Serves ``vector_all`` (all layers, skip-BOS,
  both phases) and ``antidote`` (positions="last", ``coeff =
  base_alpha``), plus the synthetic/zero test kinds.
- :data:`OP_RELU_GATED_VECTOR` — the PSR deployment (extraction/psr.py):
  ``h += eff · relu(h_fp32 · w + b) · d`` per layer, gate computed in
  fp32 on the PRE-steering hidden state, result cast back to the model
  dtype. Prefill AND decode with NO BOS skip; POSITIONS come from the
  fit leg + the cell like every re-maskable arm (default "all" —
  PSRObject's any-position deployment; a "response" fit or cell serves
  the cross-phase mask). The learned gate does NOT substitute for the
  mask: serving a response-fit object unmasked is the measured
  fit=response/serve=all model-destroyer corner of the masking 2x2,
  not a no-op — the gate down-weights BOS/outlier tokens WITHIN the
  served region, which is a different job than choosing the region.
- :data:`OP_LINEAR_GATED_VECTOR` — the ``clamp_gate=False``
  (paper-parity) mode: the gated body with the ReLU removed,
  ``h += eff · (h_fp32 · w + b) · d``. Separate from the ReLU op because
  dropping the clamp is a semantic choice — it keeps the
  negative-projection half of the steer — and serving one as the other
  would be a different intervention, not a tolerance.
- :data:`OP_AFFINE_SUBSPACE` — LoReFT (extraction/reft.py):
  ``h += eff · ((h_fp32 (W−R)ᵀ + b) R)`` in fp32, PREFILL ONLY (prompt
  positions only; KV-cached decode steps pass through untouched — the
  class's own seq_len gating).
- :data:`OP_PREFILL_LAST_MATMUL` — AlphaSteer (extraction/alphasteer.py):
  during the prefill only, ``h += eff · (h_last · M)`` broadcast to every
  prompt position, ``h_last`` the LAST prompt token's hidden state,
  ``coeff = base_scale``; decode steps are untouched (influenced only
  through the steered prompt KV — the paper's prefill-only deployment).

POSITION/PHASE CONTEXT. Position- or phase-dependent ops need a
:class:`HookContext` (phase + the global per-token positions of the rows
in ``hidden``); the GPU hook wiring builds it from vllm-lens's per-request
forward metadata, and MUST refuse at startup if the metadata cannot
express it (never a guessed row). A prefill context carries a CONTIGUOUS
prompt slice at absolute positions ``cached..cached+T-1``: ``cached == 0``
is the complete prompt in one forward (positions 0..P−1 — the only prefill
shape the hooks accepted before 2026-09-06); ``cached > 0`` is a RESUMED
prefill — the request's first ``cached`` prompt positions were served
from the KV cache (a prefix-cache hit) and only the tail is in this
forward. A resumed context is legal ONLY for ops that pass
:func:`resumed_prefill_ok` (below): every other op meets a typed
:class:`ResumedPrefillRefusal` from ``apply_hook_ops``, never a guessed
row. Chunked-prefill continuations stay unrepresentable (the worker's
lattice, ``lens_batch``, tells the two apart by the request's own prompt
length and admits only a slice that ends exactly at the prompt end).

RESUMED PREFILLS — WHICH OPS MAY RESUME (2026-09-06, for shared prefills
on steered cells). vLLM keys KV blocks on token ids + the request
``cache_salt`` and never on the steering selector, and a cached block
permanently carries whatever intervention computed it (vllm_server /
proxy module docstrings). Resuming a steered prompt from a cache written
by the SAME cell under the SAME intervention is therefore exactly right,
position by position, iff the edit at every position is a function of
THAT position's own hidden state and absolute position alone — then the
cached rows carry precisely the edit a cold prefill would have given
them, and the hook only has to apply the same op to the rows that are
actually in the forward. Three things break that:

- a BROADCAST op — ``prefill_last_matmul`` (alphasteer) and
  ``prefill_last_cosine_gated`` (cosine_gated_r) compute ONE edit from the
  LAST prompt token and write it onto (or decide it for) every prompt
  position, so a prefix cached under a different-suffix prompt carries a
  different request's edit;
- ``positions="last"`` on any op (antidote, static_r's prompt_last) — the
  edit's PLACEMENT depends on the prompt length, so a cached position that
  was the writer's last token was edited while it is not the reader's;
- a phase restriction (``phase != both``: loreft, static_r, alphasteer) or
  ``positions="response"``: what the writer did at a cached position
  depends on the writer's OWN prompt boundary (its generated rows were
  decode rows; its response boundary sat at ITS header), which the reader
  cannot recover from the cache — a multi-turn turn N resumed from turn
  N-1's cache would carry steered header/assistant rows a cold prefill of
  turn N leaves untouched under a tail-anchored response mask. (THE
  ALL-TURNS MASK, below, steers those rows too in ITS cold prefill of
  turn N; its ADMISSION of a resume is nonetheless identical to
  ``response`` today — the bullet below says why.)

So the RESUMED-PREFILL POLICY of a cell (:func:`resumed_prefill_policy`) is
THREE-VALUED (the reviewer's ruling, 2026-09-06):

- :data:`RESUMED_ALWAYS` — every op a per-position op
  (:data:`PER_POSITION_OPS` — the fixed vector, the two gated vectors, the
  affine subspace, the two low-rank gates, the o_proj LoRA) served at
  ``positions="all"`` in BOTH phases, ``skip_bos`` either way (position 0
  is the same absolute position for every request sharing a prefix): any
  resume is correct, position by position.
- :data:`RESUMED_NEVER` — any broadcast op, any ``positions="last"`` op,
  any phase-restricted op: no resume is ever safe; the worker keeps the
  engine-fatal salt-discipline refusal for such a cell (behind the proxy's
  request-unique salt a resume there is an invariant breach).
- :data:`RESUMED_CONDITIONAL` — every op per-position in both phases and
  at least one at ``positions="response"``: safe for SOME requests, decided
  PER REQUEST at classification time. The response mask starts at the
  request's own assistant header, so a cached row is correct for the
  reader iff its writer left it unsteered as well — which is certain
  exactly when the cached region ``[0, cached)`` contains NO occurrence of
  the header id sequence, full or partial at the cache boundary: every
  writer's boundary is a header occurrence, so a header-free cached region
  was steered by nobody, and the reader's own boundary lies past it too
  (:func:`cached_region_header_conflict`). That admits the production
  motivation — single-turn requests on a ``response``-mode scale-12 cell
  sharing a long few-shot prefix that diverges BEFORE the final header —
  and refuses a cached earlier header (a prior writer's response boundary:
  the multi-turn turn-N-from-turn-N-1 shape) and a cache boundary inside
  the header. An unsafe hit is an EXPECTED event under a stable salt, so
  its refusal is CONTAINED per request (``lens_worker``: the request is
  failed and served unsteered, its response refused at the proxy, the
  engine kept), never engine-fatal.
- and ``positions="response_all_turns"`` (THE ALL-TURNS MASK below) is
  CONDITIONAL too, under the SAME two clauses as ``response`` — today. The
  cached-region check is factored into its two clauses
  (:data:`CONFLICT_CLAUSES`: a FULL header occurrence anywhere in the
  region, and a cache boundary INSIDE a header) with a per-mode clause
  set (:func:`conflict_clauses_for`), and the all-turns set is the FULL
  set. So a cached region holding a complete earlier header is refused
  exactly as under ``response``: turn N still does not resume turn N-1
  on an all-turns cell. That is deliberate (the 2026-09-07 reviews of the
  first two landings). Dropping the full-occurrence clause for all-turns
  — the narrowing this mode was conceived for, since its own cold prefill
  steers the earlier turn too — was shown unsound as shipped, twice, and
  for a structural reason: (a) the clause also blocked inheriting a
  FAILED writer's rows — a request the locator refuses is served
  unsteered but still cached, and under a cell-stable salt an identical
  retry whose region holds that complete header would have inherited its
  unsteered rows; (b) spans are located by LEXICAL scanning over token
  ids, and content can reproduce the template's own markers (a message
  quoting the delimiter ends a historical span early, or runs the final
  span past it). Three properties come apart here (docs §7): recovering
  the intended message ROLES from token ids is not always possible;
  cache CONSISTENCY across requests is achievable with a conservative
  token-only predicate; and a successful steered writer's PROVENANCE is
  a third thing neither establishes — a cache-consistent lexical mask
  can still be semantically wrong. Two preconditions gate narrowing the
  clause set: (a) a failed writer must not be able to leave reusable
  entries in a shared steering namespace, and (b) span boundaries must
  come from the request's declared message structure rather than from
  lexical scanning of the token stream; the clause-set machinery exists so
  that landing is a one-line change once they hold. Like ``response`` the
  all-turns op needs the cached region's IDS on a resumed context — to
  verify the clauses AND to locate a span that opened before the boundary.

:func:`resumed_prefill_ok` stays as ``policy == RESUMED_ALWAYS`` for the
callers that need a plain bool. The partition is derived from the SAME op
vocabulary the reduction contract partitions (:data:`REDUCTION_OPS` /
:data:`ELEMENTWISE_OPS`): every op in :data:`OPS` is exactly one of
:data:`PER_POSITION_OPS` / :data:`BROADCAST_OPS`, and an unclassified op
name refuses rather than defaulting.

What this module does NOT decide is whether the cache a request resumes
WAS written under the identical intervention: that is the routing proxy's
salt namespace (``proxy.cache_salt_for`` — request-unique for lens legs
today; a stable per-(cell, intervention) salt for ``always`` cells and for
single-turn requests on ``conditional`` cells is the proxy half of this
landing), and the worker can attest only that a resume happened and how
many positions it covered (``lens_worker.SelectionRecord``), never who
wrote the blocks.

THE RESPONSE MASK (``positions="response"`` — the cross-phase
mechanism). Steering "the generated text" must NOT be implemented as an
unsteered prefill followed by steering from the first generated token
onward: token 1 is sampled from the prefill forward, so such a mechanism
cannot shape it by construction, and an unsteered token 1 silently
disables a refusal defense. The served region is therefore the
ASSISTANT-HEADER positions at the tail of the prompt PLUS every generated
position — the final prefill row is inside the mask, so the forward that
emits token 1 is steered.

The boundary is LOCATED, never computed. The op carries the header's
exact token ids (``header_ids``, from
:func:`sisteer.chat_markers.generation_header_ids` — the id-form of the
fit-time width, NOT the terminator-anchored located header) rather than a
width H, so a baked count cannot drift from the served template; the
hook matches those ids against the request's own tokens before masking
and refuses on any mismatch. ``P − H`` arithmetic alone would pass every
width check and still be wrong under assistant prefill, where the last H
prompt tokens are seeded content rather than the header — see
:func:`_response_start`. This is the same principle the marker module
itself encodes: locate template structure in a render, never infer it
from a count.

PREFILL-SEEDED RENDERS (the served-loglikelihood landing, 2026-09-01)
are served ONLY when the request declares them: the proxy's
``petri_response_prefill`` gate threads ``HookContext.response_prefill``,
and :func:`_response_start` then locates the boundary at the LAST header
occurrence (header + seed + generated all masked — the HF reference
scorer ``parity._response_letter_nlls``'s re-pointed-boundary semantics),
so a teacher-forced scoring request's final prefill row — the forward
that emits token 1, i.e. the scored distribution — is inside the mask by
construction. An UNDECLARED seeded render still refuses exactly as
before, and the flag changes only WHERE a response op's boundary is
located within one declared request — never which position MODE is
served, so the fit/serve position-regime guard below is untouched.

PETRI AGENT-LOOP RENDERS (the declared multi-turn landing, 2026-09-02)
travel the same path under their own declaration: the proxy's
``petri_agent_loop`` gate threads ``HookContext.agent_loop``, and
:func:`_response_start` then locates the boundary at the LAST header
occurrence exactly as the seeded arm does. The shapes are the three the
Petri auditor actually sends (a pending assistant tool call; a tool
result awaiting the model's continuation; assistant text + tool call +
tool result) — all ``add_generation_prompt=True``, none ending at the
header, because the canonical gemma-4 template continues the OPEN model
turn after tool traffic (its "generation prompt" after a tool call is a
tool-response opener, not a fresh header). The only shape invariant is
therefore "the header occurs somewhere and the prompt does not end with
it" — never a particular tail token, which differs across all three.
WHAT THE MASK COVERS under this arm, stated so nobody reads the masked
region as ordinary assistant text: the header + the assistant's tool
call + the tool-response SYNTAX AND BODY (external input, not model
output) + every generated position. That is exactly how the completed
gemma-E4B audit ran (the unconditional locator on origin/petri-gemma-e4b),
so the declared path preserves comparability with it; excluding
tool-result spans from the mask is a possible later refinement, not this
arm. As with the seeded arm, the flag changes only WHERE one declared
request's boundary sits — never which position MODE is served — so the
fit/serve position-regime guard is untouched by construction (it reads
``op.positions`` and no context at all).

THE ALL-TURNS MASK (``positions="response_all_turns"`` — a SERVED position
mode, 2026-09-07). Under ``response`` a multi-turn conversation that
steers only the final assistant turn re-encodes every EARLIER assistant
turn UNSTEERED, and turn N cannot resume turn N-1's KV on a
``conditional`` cell (RESUMED PREFILLS above). The all-turns mode removes
the FIRST; the second it does NOT deliver today — its resume admission is
identical to ``response``'s (RESUMED PREFILLS above; the two
preconditions for narrowing it are stated there). The
served region is EVERY assistant-role span in the render — each
assistant header through that turn's END SCAFFOLD — plus every generated
position. Two markers locate it, both verified against the request's own
tokens and neither ever inferred from a width: the header (``header_ids``,
as for ``response``) opens a span, and the turn-end scaffold
(``turn_end_ids``, from :func:`sisteer.chat_markers.assistant_turn_end_ids`
— the terminator plus the glue before the next turn's opener: gemma-4
``[<turn|>, \n]``, Llama-3.1 ``[<|eot_id|>]``) closes every earlier one.
That scaffold is ``AssistantMarkers.closing`` — the authoritative
"content-independent scaffold after assistant content" — as it appears
BETWEEN turns: equal to ``closing`` on every template that appends no
unconditional trailing header (gemma-4, meta-llama; asserted on the real
gemma-4 template), and ``closing``'s head minus that trailing header on a
flag-ignoring template (the NousResearch mirror), where raw ``closing``
would never occur after a historical turn. It is LOCATED in a multi-turn
probe and cross-checked against ``closing`` rather than copied from it.
The next header would NOT do as a span end: on gemma-4 the same control
token ``<|turn>`` opens user and model turns, so "header to header" would
steer the user's turns. The SCAFFOLD rather than the bare terminator
because the final span always runs to the prompt end and a supported
closed-tail render (gemma-4's Petri ``content_and_tool_response`` shape)
ends in ``<turn|>\n`` — steering that glue while the turn is final — so
a historical span that stopped at the terminator would give the SAME ids
a DIFFERENT mask once a user turn followed, and a cached block holding
the glue would be poison; ending the historical span after the scaffold
makes a turn's mask invariant when it stops being final (verified on the
real gemma-4 template, where the scaffold between turns is exactly the
closed-tail render's tail). The generating turn's boundary is located
EXACTLY as ``response`` locates it (:func:`_locate_final_header` — the
tail rule, the two declared-render arms, the same refusals), and an
earlier header not closed by the scaffold before the next header REFUSES
(:func:`assistant_turn_spans`) rather than guessing where the user's turn
begins. On a RESUMED prefill the spans are located over the FULL prompt
— cached region + slice — so a turn that opened before the cache boundary
is masked correctly in the slice, and the cached region is verified
against the same two clauses ``response`` checks; that is why an
all-turns op needs ``cached_ids`` too. A completed gemma-4 agent-loop turn
(tool call + tool response inside one open model turn, one ``<turn|>\n``
at its end) is one span, exactly the region the declared agent-loop arm
steers while that turn is the generating one. One shape FAILS CLOSED
(:func:`assistant_turn_spans`): a generating turn whose content contains
a complete scaffold followed by more content — a message quoting the
template's delimiter — because as a historical turn the same tokens
would end the span at that scaffold, so the mask would depend on whether
the turn is final; the lexical locator cannot tell quoted scaffold from
real, and refusing is the only reading that is not silently wrong. The
mirror rule holds for HISTORICAL turns: the first scaffold after an
earlier header must be followed by the next header or by a COMPLETE role
header of the template's shape (opener, one role token, the assistant
header's tail — certified for the user header by
``chat_markers.successor_role_header_ids``), else the scaffold was quoted
and the render refuses — a scanner that trusted it would leave the rest
of that turn unsteered while the eval records a successful all-turns
serve, and an opener token alone is not proof of a turn.

THE BLIND SPOT, AND THE STATUS IT IMPOSES. Content that reproduces a
COMPLETE turn boundary — scaffold and a role header of the certified
shape — is indistinguishable from a real one by any reading of the token
stream, and no refusal rule closes it: a fake conversation embedded in a
message is masked as if it were real. The mode is therefore EXPERIMENTAL
AND UNSUPPORTED until boundaries are declared authoritatively from the
request's message structure (precondition (b) above). There is no
corpus-shape escape hatch, because we do
not have a sound one: a scan of raw message content cannot see template
transformations (thinking-strip reassembling a boundary across the gap),
concatenation, tool arguments or tool descriptions — three executed false
negatives (§7) — so "verified free of boundary tokens" cannot be
established by scanning content, and the earlier contract that said it
could is withdrawn. Whether collisions concentrate on adversarial corpora
remains a hypothesis. This cannot be detected here: the hook sees token
ids, not messages.

It is a POSITION MODE, not a third declared render bit, on purpose: the
two declared bits (``response_prefill``, ``agent_loop``) describe a
request's RENDER SHAPE and move ONE boundary; this changes WHICH REGIONS
are edited, which is the op's position mode — and being a mode it is
visible in the cell identity (the hashed spec's ``positions``) instead of
riding a per-request flag. It is SERVED-ONLY (not in FIT_POSITION_MODES):
the fit leg of a cell served this way stays ``response``, and the
fit/serve guard admits exactly that pairing as a named carve-out
(:func:`_check_fit_serve_positions`). ``response`` itself is untouched —
existing cells, the Petri agent-loop arm and the completed audit's regime
keep the last-header semantics for comparability; this is an added
alternative, never a replacement, and no default changes.

WHICH ARMS MAY BE RE-MASKED — DERIVATION LOCALITY. The response mask is
a property of the CELL, not of the object: a cross-phase cell IS an
all-position artifact served under the mask, so the resolved spec's
``positions`` overrides the artifact's. That override is only sound for
arms whose edit is COMPUTABLE FROM THE POSITIONS IT IS APPLIED TO.

Two conditions, and the second is the one that is easy to miss:

- POSITIONAL SUBSETTING — the served footprint must be a subset of the
  arm's defined footprint. A restriction is meaningful; a relocation is
  a different method.
- DERIVATION LOCALITY — the edit at a position must be derived from
  THAT position. An edit derived from position X and applied at
  position Y is a RELOCATION even when Y lies inside the defined
  footprint, because the function changed while the footprint merely
  shrank.

Fixed-vector arms (``vector``, ``vector_all``, ``static_r``) read
nothing from the activation and pass both trivially. The per-position
gated arms (``psr``, ``alphakl``) evaluate their gate at each position
from that position's own hidden state, so they pass too. ``alphasteer``
FAILS the second: it computes ONE vector from the last prompt token and
broadcasts it, so masking to header-plus-generated would apply a
prompt-derived vector to generated positions. ``cosine_gated_r`` fails
likewise (a per-request decision taken off the last prompt token). Both
REFUSE a response override rather than serve an undefined function; a
genuine cross-phase variant of either belongs in its own arm with its
own semantics.

(A subset test alone would have waved ``alphasteer`` through on
footprint grounds.)

DECODE-ONLY / PREFILL-ONLY SEMANTICS (shared, verbatim, with the Evals
lane's method-generic ``steer-generated-only`` mode — do NOT "simplify"
this back into a scale-0 application):

  A phase-restricted op is NOT INVOKED outside its phase. Under
  ``phase="decode"`` the prefill runs with the op entirely un-applied —
  ``apply_hook_ops`` returns the prompt-position hidden states as the
  SAME tensor, so the prompt KV stays the BASE model's bytes and steered
  decode attends over UNSTEERED prompt KV ("prompt encoded by base
  weights, generation by adapted weights"). This is deliberately NOT
  "applied at prompt positions with scale 0": the two coincide for pure
  additive ops but NOT for gated ops whose gate reads the hidden state.
  BOS-skip is ORTHOGONAL to phase — it is a position mask WITHIN the
  hooked region, never a phase gate, and the two flags never collapse
  into each other.

PER-KIND COVERAGE (the petri matrix's ARM_OBJECT_KINDS):

  =================  ==========================================
  kind               translation
  =================  ==========================================
  psr                OP_RELU_GATED_VECTOR per stored layer; positions
                     from the fit leg + the cell (serves "all" AND
                     the "response" mask, like the lora kinds)
  vector             OP_ADD_VECTOR per layer over a SteeringVector
                     container — also the served form of the
                     diff-in-means family and BiPO, which are
                     extraction methods rather than kinds
  vector_all         OP_ADD_VECTOR per layer (skip_bos honored;
                     serves "all" AND the "response" mask)
  antidote           OP_ADD_VECTOR positions="last", coeff=base_alpha
  alphasteer         OP_PREFILL_LAST_MATMUL, coeff=base_scale
  loreft             OP_AFFINE_SUBSPACE per layer (prefill-only; a
                     response-fit blob REFUSES — this translation
                     cannot express its recorded mask)
  alphakl            OP_RELU_GATED_VECTOR, or OP_LINEAR_GATED_VECTOR
                     when clamp_gate is False (zero bias either way)
  static_r           OP_ADD_VECTOR, prefill-only; positions "last"
                     or "all" per steer_positions, and a resid_pre
                     fit serves one block earlier (site shift)
  caa_static_r       OP_ADD_VECTOR per stored layer, prefill-only,
                     ALL prompt positions, resid_pre (one-block
                     shift) — the Class B CAA floor's single
                     deployment, PINNED here because its save blob
                     records no mode and static_r's defaults would
                     read it as a one-layer last-token edit
  cosine_gated_r     OP_PREFILL_LAST_COSINE_GATED
  lowrank_gated      OP_LOWRANK_{RELU,LINEAR}_GATED per layer
  loreft_both        OP_AFFINE_SUBSPACE, BOTH phases (the one
                     difference from the loreft kind)
  lora               OP_LORA_OPROJ (ONE layer, rank 1) — served at
                     the o_proj MODULE site (HOOK_SITE_OPROJ), the
                     mid-block site the block-output vocabulary
                     cannot express; serves "all" AND the
                     "response" mask. The genuine PEFT export
                     (serving/lora_export.py) remains the committed
                     path for ALL-POSITION fits; a position-masked
                     fit serves HERE, because native multi-LoRA
                     applies an adapter at every position
  lora_all           OP_LORA_OPROJ per stored layer, rank-general
                     (the scale-12 lora_r4 objects: rank 4, every
                     layer, positions="response")
  lora_subtraction   TWO OP_LORA_OPROJ ops per stored layer — the
                     positive parent's factors at coeff +1, the
                     negative parent's at coeff -1, in that order
                     (the difference object's own registration
                     order). A new blob SCHEMA, not a new op or
                     hook site: both parents' factors ride bit-exact
                     and each serves through lora_all's translation
  circuit_breaker    UNSERVABLE here -> native multi-LoRA export
  lofit              UNSERVABLE on the lens path
  token              UNSERVABLE on the lens path
  cast               UNSERVABLE on the lens path
  fishback           UNSERVABLE on ANY batched engine
  gated_fishback     UNSERVABLE on ANY batched engine
  =================  ==========================================

The UNSERVABLE kinds refuse with the exact reason
(:data:`UNSERVABLE_KIND_REASONS`): they are weight-level, mid-block, or
stateful interventions that neither hook site can express — a faked
residual approximation would silently serve a different intervention
than the cell's hash claims. Their committed paths are the genuine PEFT
adapter export (circuit_breaker — two-SI, ``serving/lora_export.py``)
or remain gate failures until a deliberate extension lands (token /
lofit / cast). Default-deny stands: an unknown kind is ABSENT / a §10
gate failure, never a guessed translation.

THE O_PROJ SITE (HOOK_SITE_OPROJ — landed 2026-08-28 for the lora
kinds). The lora family's edit reads the o_proj INPUT (the
pre-projection attention output, not the residual stream) and shifts
the o_proj OUTPUT mid-block, before the same block's MLP reads it — its
block-output effect is input-dependent through the MLP and is NOT an
additive block-output op, which is why the kind was UNSERVABLE while
the block output was the only hook site. The worker now installs a
second hook directly on each steered layer's ``self_attn.o_proj``
module (lens_worker), where the edit IS exact:
:func:`apply_oproj_ops` mirrors the authoritative hook bodies
(RankOneLoRAObject.apply / MultiLayerLoRAObject._make_hook +
PositionMaskedFit._fit_masked) verbatim — fp32 read of the o_proj input
through A, fp32 write through B, one cast, torch.where for the mask so
masked-out positions stay BIT-identical. Native multi-LoRA
(``--enable-lora``) remains the committed path for ALL-POSITION lora
fits (lora_export.py); it applies an adapter at EVERY position, so a
positions="response" fit — token 1 steered from its own prefill row —
is not expressible there and serves here instead. TP is pinned to 1 for
this site: under tensor parallelism the o_proj input is sharded across
ranks while the hook's delta lands after the all-reduce on every rank,
so the worker preflight refuses TP>1 rather than serving a wrong
partial read.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import os
from dataclasses import dataclass, replace
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import torch
import torch.nn.functional as F

from sisteer.capeval.serving import serve_declaration as sd
from sisteer.capeval.serving import span_locator as sl


class AdapterError(RuntimeError):
    """A lens-adapter refusal (unknown kind, malformed artifact/spec) —
    loud, never a guessed translation."""


#: The one synthetic object kind (tests + gate plumbing).
#: Artifact = ``torch.save({"layer": int, "vector": 1-D float tensor})``.
SYNTHETIC_KIND = "synthetic_vector"
#: Internal kind for a hook-armed exact no-op (the ``none__lens0`` leg
#: when no lens-served donor object exists): adds a zero vector, so the
#: hook machinery runs and the activations are untouched.
ZERO_KIND = "__zero__"

# --- the op vocabulary ---------------------------------------------------------

#: add ``eff * vector`` to the residual stream at ``layer`` (position
#: mode + BOS-skip + phase aware).
OP_ADD_VECTOR = "add_vector"
#: PSR: add ``eff * relu(h_fp32 @ gate_w + gate_b) * vector`` (fp32 gate
#: on the pre-steering hidden state, cast back to the model dtype).
OP_RELU_GATED_VECTOR = "relu_gated_vector"
#: AlphaKL with ``clamp_gate=False``: the SAME arithmetic with the ReLU
#: removed — ``eff * (h_fp32 @ gate_w + gate_b) * vector``. A separate op
#: rather than a flag on the gated op, because the difference is a
#: SEMANTIC one and not an approximation: the linear gate deliberately
#: keeps the negative-projection half that a ReLU would discard (the
#: paper-parity mode), so serving one as the other is a different
#: intervention, not a rounding difference.
OP_LINEAR_GATED_VECTOR = "linear_gated_vector"
#: LoReFT: add ``eff * ((h_fp32 @ (W - R).T + b) @ R)`` in fp32.
OP_AFFINE_SUBSPACE = "affine_subspace"
#: AlphaSteer: prefill-only ``h += eff * (h_last @ matrix)`` broadcast to
#: every prompt position.
OP_PREFILL_LAST_MATMUL = "prefill_last_matmul"
#: Gated low-rank: ``h += eff * gate(h_fp32 @ Aᵀ) @ Bᵀ`` — the rank-k
#: generalization of the gated vector (k=1 with B=d recovers it). Two
#: ops for the two gates, for the same reason the vector pair is two.
OP_LOWRANK_RELU_GATED = "lowrank_relu_gated"
OP_LOWRANK_LINEAR_GATED = "lowrank_linear_gated"
#: cosine-gated R: prefill only, a per-REQUEST hard boolean gate —
#: ``cos(h_last, cond) > threshold`` decides whether the last prompt
#: token gets ``eff * vector`` at all. Unlike every other gate here it is
#: one decision per request, not a per-token coefficient.
OP_PREFILL_LAST_COSINE_GATED = "prefill_last_cosine_gated"
#: LoRA at the attention o_proj MODULE (the one op at HOOK_SITE_OPROJ):
#: ``o_proj_out += eff * ((x_fp32 @ Aᵀ) @ Bᵀ)`` where ``x`` is the o_proj
#: INPUT (the pre-projection attention output — NOT the residual stream)
#: and the delta lands on the o_proj OUTPUT, mid-block, before the same
#: block's MLP reads it. The authoritative bodies are
#: extraction/objects.RankOneLoRAObject.apply (kind 'lora', rank 1, one
#: layer) and extraction/multilayer.MultiLayerLoRAObject._make_hook
#: (kind 'lora_all', rank-general, every layer): fp32 read through A,
#: fp32 write through B, ONE cast at the end, and masked-out positions
#: left BIT-identical via torch.where (PositionMaskedFit._fit_masked).
#: o_proj is a position-wise linear map, so a position mask on the delta
#: is EXACT — which is what makes the response mask servable here when
#: vLLM's native multi-LoRA (which applies an adapter at EVERY position)
#: cannot express it.
OP_LORA_OPROJ = "lora_oproj"
#: THE PAIRED WRITE (relu-differenced-native-qwen; extraction/paired_write.
#: PSRPairedWriteObject, kind ``psr_paired_write``): ``h += eff * (relu(h_fp32 @
#: gate_w + gate_b) * vector - relu(h_fp32 @ gate_w2 + gate_b2) * vector2)`` —
#: BOTH gates read the SAME pre-steering hidden state, the two terms are summed
#: in fp32 and cast ONCE. ONE op and not two ``relu_gated_vector`` ops on the
#: same layer, because a second hook would read the FIRST hook's post-write
#: state (the class's own ``application_order = shared_pre_write_state``; the
#: sequential orders are different functions of the same parents). The native
#: residual slot serves the same arithmetic as ``relu_gated_vector_pair`` — the
#: ONE op name, pinned equal to ``native_residual.OP_RELU_GATED_VECTOR_PAIR``.
OP_RELU_GATED_VECTOR_PAIR = "relu_gated_vector_pair"

OPS = (OP_ADD_VECTOR, OP_RELU_GATED_VECTOR, OP_LINEAR_GATED_VECTOR,
       OP_AFFINE_SUBSPACE, OP_PREFILL_LAST_MATMUL,
       OP_LOWRANK_RELU_GATED, OP_LOWRANK_LINEAR_GATED,
       OP_PREFILL_LAST_COSINE_GATED, OP_LORA_OPROJ,
       OP_RELU_GATED_VECTOR_PAIR)

#: THE HOOK SITE. Every op so far applies at the decoder block's OUTPUT
#: (the residual stream after layer ``layer`` — HOOK_SITE_BLOCK); the
#: o_proj ops apply INSIDE the block, at the attention output projection
#: module of layer ``layer`` (HOOK_SITE_OPROJ). The two sites carry
#: different tensors (residual stream vs o_proj input/output), so an op
#: applied at the wrong site is wrong MATH, not a tolerance: the apply
#: entry points refuse cross-site ops, and callers route by
#: :func:`ops_at_site`.
HOOK_SITE_BLOCK = "block"
HOOK_SITE_OPROJ = "oproj"
#: ops applied at HOOK_SITE_OPROJ; everything else is HOOK_SITE_BLOCK.
OPROJ_SITE_OPS = (OP_LORA_OPROJ,)


def op_site(op: "HookOp") -> str:
    """Which hook site serves ``op`` (HOOK_SITE_BLOCK / HOOK_SITE_OPROJ)."""
    return HOOK_SITE_OPROJ if op.op in OPROJ_SITE_OPS else HOOK_SITE_BLOCK


def ops_at_site(ops: Sequence["HookOp"], site: str) -> Tuple["HookOp", ...]:
    """The subset of ``ops`` served at ``site`` — how a caller routes a
    mixed object's ops to the two hook sites, each op to exactly one."""
    if site not in (HOOK_SITE_BLOCK, HOOK_SITE_OPROJ):
        raise AdapterError(f"unknown hook site {site!r} (sites: "
                           f"{[HOOK_SITE_BLOCK, HOOK_SITE_OPROJ]})")
    return tuple(op for op in ops if op_site(op) == site)
#: the per-token gated-vector ops, which share tensor fields, shape rules
#: and arithmetic and differ only in the gate's nonlinearity.
_GATED_OPS = (OP_RELU_GATED_VECTOR, OP_LINEAR_GATED_VECTOR)
#: likewise for the low-rank pair.
_LOWRANK_OPS = (OP_LOWRANK_RELU_GATED, OP_LOWRANK_LINEAR_GATED)
#: every op whose gate is a ReLU rather than the raw linear projection (both
#: of the paired write's gates are).
_RELU_GATES = (OP_RELU_GATED_VECTOR, OP_LOWRANK_RELU_GATED, OP_RELU_GATED_VECTOR_PAIR)

#: phases (see the module docstring's decode-only/prefill-only contract —
#: shared with the Evals steer-generated-only lane; a phase-restricted op
#: is NOT INVOKED outside its phase, never applied-at-scale-0).
PHASE_BOTH = "both"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASES = (PHASE_BOTH, PHASE_PREFILL, PHASE_DECODE)

#: position modes for OP_ADD_VECTOR ("last" = the final prompt token
#: during the prefill, every incremental forward's single position during
#: decode — hooks.py's Jailbreak-Antidote mode).
POSITIONS_ALL = "all"
POSITIONS_LAST = "last"
#: "response" — the CROSS-PHASE mask (see THE RESPONSE MASK in the module
#: docstring): the assistant-header positions at the tail of the prompt
#: PLUS every generated position. An op in this mode carries the header's
#: TOKEN IDS and verifies them against the request before masking.
POSITIONS_RESPONSE = "response"
#: "response_all_turns" — the ALL-ASSISTANT-TURNS variant of the response
#: mask (THE ALL-TURNS MASK in the module docstring): EVERY assistant-role
#: span in the render — each assistant header through that turn's END
#: SCAFFOLD (terminator + the glue before the next turn's opener) — PLUS
#: every generated position, not only the final turn. A SERVED mode only,
#: never a fit leg (deliberately absent from FIT_POSITION_MODES below: no
#: object is fitted under it; the fit leg of a cell served this way is
#: "response"). An op in this mode carries the header's token ids AND the
#: turn-end scaffold's, and LOCATES every span by matching them against
#: the request — never a width, never the next header (on gemma-4 the
#: same opener control token starts user and model turns alike, so "up to
#: the next header" would steer the user turns).
#:
#: STATUS (2026-09-07): EXPERIMENTAL AND UNSUPPORTED until span boundaries
#: are declared authoritatively from the request's message structure
#: (precondition (b) of the module docstring). Spans are located
#: in the token stream; content that reproduces a complete turn boundary
#: (scaffold + a role header of the certified shape) is indistinguishable
#: from a real one by any reading of that stream and is masked as if real,
#: nothing here can detect it (the hook sees ids, not messages), and there
#: is NO corpus-shape escape hatch: a scan of raw message content cannot
#: see template transformations, concatenation, tool arguments or tool
#: descriptions (three executed false negatives, §7), so no scan certifies
#: a corpus. The mask matches the message-derived expectation on all TESTED
#: production fixtures (the real-template acceptance matrix) — no more than
#: that; what is unsupported is the claim that an arbitrary request's
#: assistant turns are the ones it steers.
POSITIONS_RESPONSE_ALL_TURNS = "response_all_turns"
POSITION_MODES = (POSITIONS_ALL, POSITIONS_LAST, POSITIONS_RESPONSE,
                  POSITIONS_RESPONSE_ALL_TURNS)
#: the HEADER-LOCATED modes: every op in either carries ``header_ids`` and
#: edits assistant-turn spans ONLY (the all-turns mode also carries
#: ``turn_end_ids``). This is the "same KIND of position" the fit/serve
#: guard's response carve-out rests on (:func:`_check_fit_serve_positions`),
#: and both are ``conditional`` for a resumed prefill under the SAME
#: clause set today (:func:`conflict_clauses_for` — the machinery that
#: would let the all-turns set narrow later).
RESPONSE_MASK_MODES = (POSITIONS_RESPONSE, POSITIONS_RESPONSE_ALL_TURNS)

#: the FIT-LEG vocabulary — the position regime an object was TRAINED
#: under, as its own HASHED bytes record it (the universal explicit key
#: ``positions_at_fit``, torch blob or JSON sidecar, plus each kind's own
#: save-schema keys read through :data:`_BLOB_POSITION_SCHEMAS`; the fit
#: driver stamps the same fact per cell). Exactly two legs exist
#: (hooks.PositionMaskedFit): "all" (Class A/B) and "response" (Class C,
#: decode-only — masked at the assistant header + generated positions).
#: This is a DIFFERENT axis from the serving modes above: a "last" or
#: prefill-only DEPLOYMENT (antidote, alphasteer, cosine_gated_r) is an
#: all-position FIT with a narrow serve, not a masked fit.
FIT_POSITION_MODES = (POSITIONS_ALL, POSITIONS_RESPONSE)
#: ``response_all_turns`` is deliberately NOT a fit leg: a cell served under
#: it is a "response"-fit object served over MORE assistant-turn spans (the
#: guard's one named carve-out), never an object trained under a third
#: regime.

#: "prompt" — loreft's kind-INTRINSIC regime (reft.LoReFTObject.save
#: persists it as "the object's INTRINSIC position policy": the method's
#: identity fit applies at prompt positions during the prefill and gates
#: itself out of decode by seq_len). NOT a PositionMaskedFit leg and NOT
#: a served mode — it exists only in that class's blob vocabulary, so it
#: is deliberately a member of neither FIT_POSITION_MODES nor
#: POSITION_MODES.
POSITIONS_PROMPT = "prompt"

#: the two PositionMaskedFit legs, normalized to themselves — the
#: vocabulary of every explicit ``positions_at_fit`` record and of the
#: masked-fit family's blob ``positions`` key.
_MASKED_FIT_VOCAB: Dict[str, str] = {
    POSITIONS_ALL: POSITIONS_ALL,
    POSITIONS_RESPONSE: POSITIONS_RESPONSE,
}

#: PER-KIND SAVE-SCHEMA DECLARATIONS for the torch-blob keys that record
#: a position regime — :func:`recorded_fit_positions` reads blobs ONLY
#: through this table. The generic key name ``positions`` spans THREE
#: vocabularies across the save schemas: the PositionMaskedFit fit leg
#: (PSRObject / RankOneLoRAObject / MultiLayerLoRAObject.save — "The
#: FIT's mask leg is part of the artifact's identity"), vector_all's
#: serve-time DEPLOYMENT attr (MultiLayerVectorObject.save keeps the fit
#: leg under the DISTINCT key ``fit_positions``), and loreft's intrinsic
#: prompt/response policy (LoReFTObject.save — "prompt" is the method's
#: identity; that class has no "all"). Reading every blob as if it spoke
#: the fit-leg vocabulary mis-reads two of the three; it refused every
#: loreft artifact ever saved (caught 2026-09-01).
#:
#: Entry format: ordered ``(key, vocabulary)`` pairs. The FIRST key
#: present in the blob is that artifact's regime record — later keys are
#: the schema's serve-time defaults, superseded by the explicit fit leg
#: when both exist (vector_all: a Class-C blob whose deployment attr
#: says "all" still records a response fit). Each vocabulary maps a
#: stored value to the guard's regime; a stored value outside it
#: refuses rather than guesses.
_BLOB_POSITION_SCHEMAS: Dict[str, Tuple[Tuple[str, Mapping[str, str]],
                                        ...]] = {
    "psr": (("positions", _MASKED_FIT_VOCAB),),
    "lora": (("positions", _MASKED_FIT_VOCAB),),
    "lora_all": (("positions", _MASKED_FIT_VOCAB),),
    # LoRASubtractionObject.save persists the leg its two parents SHARE
    # (the class refuses parents whose legs differ), in lora_all's own
    # vocabulary — the same PositionMaskedFit key, same two values.
    "lora_subtraction": (("positions", _MASKED_FIT_VOCAB),),
    # PSRPairedWriteObject.save writes the leg its two psr parents SHARE — always "all"
    # (the class refuses masked parents) — under psr's own key and vocabulary.
    "psr_paired_write": (("positions", _MASKED_FIT_VOCAB),),
    "vector_all": (("fit_positions", _MASKED_FIT_VOCAB),
                   ("positions", _MASKED_FIT_VOCAB)),
    "loreft": (("positions", {POSITIONS_PROMPT: POSITIONS_PROMPT,
                              POSITIONS_RESPONSE: POSITIONS_RESPONSE}),),
}

#: every blob key ANY schema above uses to record a regime. One of these
#: on a kind (or under a key) the table does not declare is an
#: UNDECLARED vocabulary — reading it as any particular one would be a
#: guess (the loreft mis-read again, in either direction), so it refuses
#: until the schema is declared.
_POSITION_RECORD_KEYS = ("positions", "fit_positions")

#: which tensor fields each op carries, and their materialization policy:
#: "model" fields move to the serving dtype at worker startup; "fp32"
#: fields move device-only and STAY fp32 — their op computes in fp32 and
#: casts the result back (the authoritative hook bodies' arithmetic; a
#: bf16 gate would silently change the PSR/LoReFT math).
_OP_TENSOR_FIELDS: Dict[str, Dict[str, str]] = {
    OP_ADD_VECTOR: {"vector": "model"},
    OP_RELU_GATED_VECTOR: {"vector": "fp32", "gate_w": "fp32",
                           "gate_b": "fp32"},
    OP_LINEAR_GATED_VECTOR: {"vector": "fp32", "gate_w": "fp32",
                             "gate_b": "fp32"},
    OP_AFFINE_SUBSPACE: {"rotate": "fp32", "affine_w": "fp32",
                         "affine_b": "fp32"},
    OP_PREFILL_LAST_MATMUL: {"matrix": "model"},
    OP_LOWRANK_RELU_GATED: {"lowrank_a": "fp32", "lowrank_b": "fp32"},
    OP_LOWRANK_LINEAR_GATED: {"lowrank_a": "fp32", "lowrank_b": "fp32"},
    OP_PREFILL_LAST_COSINE_GATED: {"vector": "fp32", "cond": "fp32"},
    # fp32 on purpose: the authoritative hooks keep A/B in the fp32 the
    # save path wrote and cast x up / the delta down (one cast each way).
    OP_LORA_OPROJ: {"lowrank_a": "fp32", "lowrank_b": "fp32"},
    # the paired write: two gated-vector terms, every field fp32 (both gates
    # and the difference compute in fp32; ONE cast at the end)
    OP_RELU_GATED_VECTOR_PAIR: {"vector": "fp32", "gate_w": "fp32", "gate_b": "fp32",
                                "vector2": "fp32", "gate_w2": "fp32", "gate_b2": "fp32"},
}
_ALL_TENSOR_FIELDS = ("vector", "gate_w", "gate_b", "rotate", "affine_w",
                      "affine_b", "matrix", "lowrank_a", "lowrank_b",
                      "cond", "vector2", "gate_w2", "gate_b2")


@dataclass(frozen=True)
class HookOp:
    """One immutable worker-resident hook operation (vocabulary above).

    ``coeff`` is the object's own baked scale (antidote base_alpha,
    alphasteer base_scale; 1.0 elsewhere) — multiplied with the SELECTOR
    scale at apply time, never conflated with it. ``skip_bos`` and
    ``phase`` are independent flags (module docstring): BOS-skip masks
    positions WITHIN the hooked region; phase decides whether the op is
    invoked at all in a given forward.

    ``header_ids`` belongs to ``positions="response"`` alone: the exact
    token ids of the assistant header, derived at export time by
    :func:`sisteer.chat_markers.generation_header_ids`. The ids rather
    than a width H, so a baked count can never disagree with the served
    template — H is just ``len(header_ids)``, and the hook VERIFIES the
    ids against each request's own tokens before masking.

    ``turn_end_ids`` belongs to ``positions="response_all_turns"`` alone
    (beside ``header_ids``): the exact token ids of the assistant turn's
    END SCAFFOLD as it appears between turns — the terminator plus the
    glue before the next turn's opener
    (:func:`sisteer.chat_markers.assistant_turn_end_ids`; gemma-4
    ``[<turn|>, \\n]``, Llama-3.1 ``[<|eot_id|>]``). An earlier assistant
    turn's span runs from its header THROUGH this scaffold, and that end
    is located by matching these ids — the same discipline as the header,
    for the same reason: a span end inferred from a count, or from the
    next header, would steer user-turn positions on templates whose turn
    opener is role-agnostic. The scaffold rather than the bare terminator
    because a closed-tail render steers the scaffold as part of its FINAL
    span, and a turn's mask must not change when it stops being final
    (THE ALL-TURNS MASK, module docstring).

    ``thinking_header_ids`` is the DECLARED SECOND HEADER of a cell served
    out of its fit support under a thinking-ON render
    (``serve_declaration.SERVE_OUT_OF_SUPPORT_KEY`` carrying
    ``"thinking"``): on Qwen3.5 the thinking switch MOVES the assistant
    header (7 tokens ending in a closed empty think block pinned OFF; 5
    tokens ending in an OPEN ``<think>`` pinned ON —
    ``chat_markers.generation_header_ids_under``), so the OFF header never
    occurs in an ON render and the locator would refuse every such request.
    A request that DECLARES the ON render (the proxy's ``petri_thinking_on``
    gate -> ``HookContext.thinking_on``) is located against THIS header
    instead of ``header_ids``; everything else about the mask is unchanged
    (header + every generated position — under thinking-ON that is the
    think block AND the answer, by design: decode rows are all-or-nothing).
    Legal on the header-located modes only, attached by
    :func:`attach_thinking_header` from the cell's hashed declaration, and
    never equal to ``header_ids`` (a second header that IS the first would
    make the declaration vacuous — refused, never a silent no-op)."""

    op: str
    layer: int
    vector: Optional[torch.Tensor] = None      # [d]
    gate_w: Optional[torch.Tensor] = None      # [d]
    gate_b: Optional[torch.Tensor] = None      # [1]
    rotate: Optional[torch.Tensor] = None      # [r, d] (orthonormal rows)
    affine_w: Optional[torch.Tensor] = None    # [r, d]
    affine_b: Optional[torch.Tensor] = None    # [r]
    matrix: Optional[torch.Tensor] = None      # [d, d]
    lowrank_a: Optional[torch.Tensor] = None   # [k, d] (the read basis)
    lowrank_b: Optional[torch.Tensor] = None   # [d, k] (the write basis)
    cond: Optional[torch.Tensor] = None        # [d] (cosine gate's probe)
    vector2: Optional[torch.Tensor] = None     # [d] the paired write's NEG direction
    gate_w2: Optional[torch.Tensor] = None     # [d] the paired write's NEG gate
    gate_b2: Optional[torch.Tensor] = None     # [1]
    threshold: Optional[float] = None          # the cosine gate's theta
    coeff: float = 1.0
    positions: str = POSITIONS_ALL
    skip_bos: bool = False
    phase: str = PHASE_BOTH
    header_ids: Optional[Tuple[int, ...]] = None
    turn_end_ids: Optional[Tuple[int, ...]] = None
    thinking_header_ids: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.op not in OPS:
            raise AdapterError(f"unknown hook op {self.op!r} (vocabulary: "
                               f"{list(OPS)})")
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) \
                or self.layer < 0:
            raise AdapterError(f"HookOp.layer must be an int >= 0, got "
                               f"{self.layer!r}")
        if self.phase not in PHASES:
            raise AdapterError(f"HookOp.phase must be one of {list(PHASES)},"
                               f" got {self.phase!r}")
        if self.positions not in POSITION_MODES:
            raise AdapterError(
                f"HookOp.positions must be one of {list(POSITION_MODES)}, "
                f"got {self.positions!r}")
        if not isinstance(self.skip_bos, bool):
            raise AdapterError(f"HookOp.skip_bos must be a bool, got "
                               f"{self.skip_bos!r}")
        self._validate_header_ids()
        self._validate_threshold()
        if not (isinstance(self.coeff, (int, float))
                and not isinstance(self.coeff, bool)
                and math.isfinite(float(self.coeff))):
            raise AdapterError(f"HookOp.coeff must be a finite number, got "
                               f"{self.coeff!r}")
        object.__setattr__(self, "coeff", float(self.coeff))
        wanted = _OP_TENSOR_FIELDS[self.op]
        for name in _ALL_TENSOR_FIELDS:
            val = getattr(self, name)
            if name in wanted:
                if not isinstance(val, torch.Tensor):
                    raise AdapterError(
                        f"HookOp(op={self.op!r}) requires tensor field "
                        f"{name!r}, got {type(val).__name__}")
            elif val is not None:
                raise AdapterError(
                    f"HookOp(op={self.op!r}) does not take field {name!r} — "
                    "refusing a tensor the op would silently ignore")
        self._validate_shapes()

    def _validate_header_ids(self) -> None:
        """``header_ids`` belongs to the header-located modes
        (:data:`RESPONSE_MASK_MODES`) and to nothing else, in BOTH
        directions — a response-mask op without them cannot locate its
        own mask, and any other op carrying them would advertise a
        boundary check it never performs. ``turn_end_ids`` likewise
        belongs to ``positions="response_all_turns"`` alone: the all-turns
        mask cannot end an earlier turn's span without them, and no other
        mode reads them."""
        if self.positions in RESPONSE_MASK_MODES:
            if self.header_ids is None:
                raise AdapterError(
                    f"HookOp(positions={self.positions!r}) requires "
                    "header_ids — the assistant header's exact token ids "
                    "(from chat_markers.generation_header_ids). The mask is "
                    "LOCATED by matching those ids against the request's "
                    "own tokens, never computed from a width alone: under "
                    "assistant prefill a width-only mask steers seeded "
                    "content and leaves the header unsteered, which is the "
                    "exact condition the cross-phase ruling exists to "
                    "prevent")
            ids = tuple(self.header_ids)
            if not ids:
                raise AdapterError(
                    f"HookOp(positions={self.positions!r}): header_ids is "
                    "empty (a collapsed assistant-header boundary) — "
                    "refusing, because an H==0 boundary silently disables "
                    "the whole steered region and the arm would score as "
                    "baseline")
            for i in ids:
                if not isinstance(i, int) or isinstance(i, bool) or i < 0:
                    raise AdapterError(
                        f"HookOp.header_ids must be non-negative ints, got "
                        f"{self.header_ids!r}")
            object.__setattr__(self, "header_ids", ids)
        elif self.header_ids is not None:
            raise AdapterError(
                f"HookOp(positions={self.positions!r}) does not take "
                "header_ids — they are the response mask's own boundary "
                "marker; refusing a field the op would silently ignore")
        if self.positions == POSITIONS_RESPONSE_ALL_TURNS:
            if self.turn_end_ids is None:
                raise AdapterError(
                    "HookOp(positions='response_all_turns') requires "
                    "turn_end_ids — the assistant turn's end scaffold "
                    "(terminator + following glue; chat_markers."
                    "assistant_turn_end_ids). An earlier assistant turn's "
                    "span runs from its header THROUGH that scaffold, and "
                    "the end is LOCATED by matching these ids — never "
                    "inferred from the next header (gemma-4 opens user and "
                    "model turns with the same control token, so that "
                    "would steer the user turns) and never from a width. "
                    "Without them the op cannot tell where an assistant "
                    "turn ends; refusing")
            tids = tuple(self.turn_end_ids)
            if not tids:
                raise AdapterError(
                    "HookOp(positions='response_all_turns'): turn_end_ids "
                    "is empty — a turn with no locatable end has no span; "
                    "refusing (a successor-bounded template, GLM-4.x, has "
                    "no terminator and cannot serve this mode)")
            for i in tids:
                if not isinstance(i, int) or isinstance(i, bool) or i < 0:
                    raise AdapterError(
                        f"HookOp.turn_end_ids must be non-negative ints, "
                        f"got {self.turn_end_ids!r}")
            object.__setattr__(self, "turn_end_ids", tids)
        elif self.turn_end_ids is not None:
            raise AdapterError(
                f"HookOp(positions={self.positions!r}) does not take "
                "turn_end_ids — they are the all-turns mask's own turn-end "
                "marker; refusing a field the op would silently ignore")
        if self.thinking_header_ids is not None:
            if self.positions not in RESPONSE_MASK_MODES:
                raise AdapterError(
                    f"HookOp(positions={self.positions!r}) does not take "
                    "thinking_header_ids — the declared thinking-ON header "
                    "is the header-located masks' own second boundary "
                    "marker (fit-support-transfer plan §2.1); refusing a "
                    "field the op would silently ignore")
            on = tuple(self.thinking_header_ids)
            if not on:
                raise AdapterError(
                    f"HookOp(positions={self.positions!r}): thinking_header_ids "
                    "is empty — a collapsed thinking-ON header boundary "
                    "would silently disable the steered region under the "
                    "declared render; refusing")
            for i in on:
                if not isinstance(i, int) or isinstance(i, bool) or i < 0:
                    raise AdapterError(
                        f"HookOp.thinking_header_ids must be non-negative "
                        f"ints, got {self.thinking_header_ids!r}")
            # an ON header EQUAL to header_ids is the header-INVARIANT
            # template's case (gemma-4: the thinking switch leaves the
            # assistant header in place) — attached like any other, the
            # boundary the same under both renders (the rule 2026-09-10;
            # plan P9's "vacuous" refusal withdrawn)
            object.__setattr__(self, "thinking_header_ids", on)

    def _validate_threshold(self) -> None:
        """``threshold`` is the cosine gate's calibrated theta and belongs
        to that op alone — it is a fitted decision boundary, not a knob,
        so a missing one cannot be defaulted and a stray one cannot be
        ignored."""
        if self.op == OP_PREFILL_LAST_COSINE_GATED:
            t = self.threshold
            if not (isinstance(t, (int, float)) and not isinstance(t, bool)
                    and math.isfinite(float(t))):
                raise AdapterError(
                    f"{self.op}: threshold must be a finite number (the "
                    f"calibrated theta), got {t!r} — it is fitted against a "
                    "held-out benign split to a target false-positive rate; "
                    "there is no sensible default")
            object.__setattr__(self, "threshold", float(t))
        elif self.threshold is not None:
            raise AdapterError(
                f"HookOp(op={self.op!r}) does not take a threshold — "
                "refusing a field the op would silently ignore")

    def _validate_shapes(self) -> None:
        if self.op in (OP_ADD_VECTOR,) + _GATED_OPS + (OP_RELU_GATED_VECTOR_PAIR,):
            if self.vector.dim() != 1:
                raise AdapterError(
                    f"HookOp(op={self.op!r}).vector must be 1-D, got shape "
                    f"{tuple(self.vector.shape)}")
        if self.op in _GATED_OPS + (OP_RELU_GATED_VECTOR_PAIR,):
            if self.gate_w.dim() != 1 or self.gate_w.shape != self.vector.shape:
                raise AdapterError(
                    f"{self.op}: gate_w must be 1-D with the "
                    f"vector's dim, got {tuple(self.gate_w.shape)} vs "
                    f"{tuple(self.vector.shape)}")
            if self.gate_b.numel() != 1:
                raise AdapterError(
                    f"{self.op}: gate_b must have exactly one "
                    f"element, got {self.gate_b.numel()}")
        if self.op == OP_RELU_GATED_VECTOR_PAIR:
            # the NEG side: the same shapes as the POS side, layer for layer (the class refuses
            # parents whose direction shapes differ; the two gates read one hidden width)
            if self.vector2.shape != self.vector.shape or self.gate_w2.shape != self.vector.shape:
                raise AdapterError(
                    f"{self.op}: vector2 / gate_w2 must match the pos side's dim "
                    f"{tuple(self.vector.shape)}, got {tuple(self.vector2.shape)} / "
                    f"{tuple(self.gate_w2.shape)}")
            if self.gate_b2.numel() != 1:
                raise AdapterError(
                    f"{self.op}: gate_b2 must have exactly one element, got {self.gate_b2.numel()}")
            if self.positions != POSITIONS_ALL or self.skip_bos:
                raise AdapterError(
                    f"{self.op}: the paired write is defined on all-position parents and served at "
                    "every position (PSRPairedWriteObject refuses a masked parent: a difference of "
                    "two writes has no mask of its own) — positions must be 'all' and skip_bos False")
        if self.op == OP_AFFINE_SUBSPACE:
            if (self.rotate.dim() != 2 or self.affine_w.dim() != 2
                    or self.rotate.shape != self.affine_w.shape):
                raise AdapterError(
                    "affine_subspace: rotate and affine_w must be equal-"
                    f"shape [r, d] matrices, got "
                    f"{tuple(self.rotate.shape)} vs "
                    f"{tuple(self.affine_w.shape)}")
            if (self.affine_b.dim() != 1
                    or self.affine_b.shape[0] != self.rotate.shape[0]):
                raise AdapterError(
                    f"affine_subspace: affine_b must be [r={self.rotate.shape[0]}],"
                    f" got shape {tuple(self.affine_b.shape)}")
        if self.op == OP_PREFILL_LAST_MATMUL:
            if (self.matrix.dim() != 2
                    or self.matrix.shape[0] != self.matrix.shape[1]):
                raise AdapterError(
                    f"prefill_last_matmul: matrix must be square [d, d], "
                    f"got shape {tuple(self.matrix.shape)}")
            if self.phase != PHASE_PREFILL:
                raise AdapterError(
                    "prefill_last_matmul is defined for the prefill only "
                    "(the last-prompt-token read has no decode meaning); "
                    f"got phase {self.phase!r}")
            if self.positions != POSITIONS_ALL or self.skip_bos:
                raise AdapterError(
                    "prefill_last_matmul broadcasts to every prompt "
                    "position (AlphaSteer has no position mask / BOS "
                    "carve-out) — positions must be 'all' and skip_bos "
                    "False")
        if self.op in _LOWRANK_OPS:
            a, b = self.lowrank_a, self.lowrank_b
            if a.dim() != 2 or b.dim() != 2:
                raise AdapterError(
                    f"{self.op}: lowrank_a must be [k, d] and lowrank_b "
                    f"[d, k], got {tuple(a.shape)} and {tuple(b.shape)}")
            if b.shape != (a.shape[1], a.shape[0]):
                raise AdapterError(
                    f"{self.op}: lowrank_b must be the transpose SHAPE of "
                    f"lowrank_a — expected {(a.shape[1], a.shape[0])} for a "
                    f"read basis {tuple(a.shape)}, got {tuple(b.shape)}. "
                    "The read and write bases are independent tensors; a "
                    "shape slip here would silently steer in the wrong "
                    "subspace")
        if self.op == OP_LORA_OPROJ:
            a, b = self.lowrank_a, self.lowrank_b
            if a.dim() != 2 or b.dim() != 2:
                raise AdapterError(
                    f"{self.op}: lowrank_a must be [k, d_in] (reads the "
                    f"o_proj INPUT) and lowrank_b [d_out, k] (spans the "
                    f"o_proj OUTPUT), got {tuple(a.shape)} and "
                    f"{tuple(b.shape)}")
            if b.shape[1] != a.shape[0]:
                raise AdapterError(
                    f"{self.op}: rank disagreement — lowrank_a reads "
                    f"{a.shape[0]} components but lowrank_b writes "
                    f"{b.shape[1]} (a {tuple(a.shape)}, b {tuple(b.shape)})."
                    " Deliberately NOT a transpose-shape rule: o_proj is "
                    "non-square on some served models (gemma-4-E4B "
                    "2048->2560), so d_in and d_out are independent and "
                    "only the rank must agree")
            if self.positions not in (POSITIONS_ALL,) + RESPONSE_MASK_MODES:
                raise AdapterError(
                    f"{self.op}: positions must be 'all' or 'response' (or "
                    "'response_all_turns', the response fit served over "
                    "every assistant turn) — the authoritative fits "
                    "(PositionMaskedFit.FIT_POSITION_MODES) define no other "
                    f"mode for this edit, got {self.positions!r}")
            if self.skip_bos or self.phase != PHASE_BOTH:
                raise AdapterError(
                    f"{self.op}: skip_bos must be False and phase 'both' "
                    f"(got skip_bos={self.skip_bos}, phase={self.phase!r})"
                    " — the o_proj hook bodies carry no BOS carve-out and "
                    "no phase gate; a masked variant is the position "
                    "mask's job, never a phase flag's")
        if self.op == OP_PREFILL_LAST_COSINE_GATED:
            if self.vector.dim() != 1 or self.cond.dim() != 1:
                raise AdapterError(
                    f"{self.op}: vector and cond must both be 1-D, got "
                    f"{tuple(self.vector.shape)} and "
                    f"{tuple(self.cond.shape)}")
            if self.vector.shape != self.cond.shape:
                raise AdapterError(
                    f"{self.op}: cond must share the vector's dim, got "
                    f"{tuple(self.cond.shape)} vs "
                    f"{tuple(self.vector.shape)}")
            if self.phase != PHASE_PREFILL:
                raise AdapterError(
                    f"{self.op} reads the LAST PROMPT token to decide "
                    "whether to fire, which has no decode meaning (the "
                    "edit reaches generation through the KV cache alone); "
                    f"got phase {self.phase!r}")
            if self.positions != POSITIONS_LAST or self.skip_bos:
                raise AdapterError(
                    f"{self.op} edits the last prompt token only — "
                    "positions must be 'last' and skip_bos False")


def op_needs_context(op: HookOp) -> bool:
    """Whether applying ``op`` requires a :class:`HookContext` (any
    phase-, position-, or BOS-dependence)."""
    return (op.phase != PHASE_BOTH or op.positions != POSITIONS_ALL
            or op.skip_bos)


@dataclass(frozen=True)
class HookContext:
    """The per-forward position/phase context (built by the GPU hook
    wiring from vllm-lens's per-request forward metadata; built directly
    by CPU tests).

    ``positions`` are the GLOBAL per-token sequence positions of the rows
    along ``hidden``'s dim -2, for ONE request (vLLM has no padding:
    prompt tokens sit at 0..P−1, decode steps at P, P+1, ...; the BOS is
    position 0). A ``prefill`` context must carry the COMPLETE prompt —
    positions 0..P−1, contiguous — so chunked-prefill continuations and
    prefix-cache hits are unrepresentable rather than silently
    mis-rowed. A ``decode`` context carries exactly ONE position >= 1
    (one generated token per step; the saved-sentinel gate assertion pins
    per-token firing).

    ``token_ids`` are this request's token ids for the same rows, in the
    same order — supplied by the hook wiring where it can slice them out
    of the flattened forward, and ``None`` where it cannot. They exist so
    the ``positions="response"`` mask can be LOCATED against the header
    marker rather than computed from a width; an op that needs them and
    finds ``None`` refuses rather than falling back to arithmetic.

    ``response_prefill`` is the request's DECLARED prefill-seeded-render
    bit (the served-loglikelihood landing): the client declared — through
    the proxy's ``petri_response_prefill`` gate, never by accident — that
    the final assistant turn is SEEDED (assistant prefix forcing), so the
    ``positions="response"`` boundary must be LOCATED by matching the
    header ids inside the prompt (``_response_start``'s seeded arm:
    header + seed + generated all masked, the HF scorer's
    ``parity._response_boundary`` semantics) instead of requiring the
    prompt to END at the header. Default ``False`` — every ordinary
    render keeps the exact tail-anchored behavior and its template-drift
    refusal.

    ``agent_loop`` is the request's DECLARED Petri agent-loop-render bit
    (the multi-turn landing): the Petri-facing entry declared — through
    the proxy's ``petri_agent_loop`` gate, verified there against the
    request's structure AND render — that the final model turn is OPEN
    after tool traffic (a pending assistant tool call, or a tool result
    awaiting the model), so the boundary is again LOCATED at the last
    header occurrence (``_response_start``'s declared arm; the mask then
    covers header + tool call + tool-response syntax and body + generated,
    module docstring) instead of required at the tail, which such renders
    never reach. Mutually exclusive with ``response_prefill``: the two
    declarations describe disjoint request shapes (a continued final
    message vs a generation-prompt render), so a context carrying both
    refuses.

    ``thinking_on`` is the request's DECLARED thinking-ON render bit: the
    proxy verified, through its ``petri_thinking_on`` gate, that the
    request renders under ``chat_template_kwargs.enable_thinking: true``
    on a lens cell whose hashed declaration admits it, so a
    ``positions="response"`` op's boundary is located against its
    ``thinking_header_ids`` (the ON header) instead of ``header_ids``. It
    is NOT a non-tail declaration: the ON render still ENDS at its header,
    so the tail rule holds as written, against the other header. Mutually
    exclusive with both non-tail declarations (an ON render that is seeded
    or an agent-loop turn is not a shape this experiment serves); a
    context claiming ``thinking_on`` beside either refuses."""

    phase: str
    positions: Tuple[int, ...]
    token_ids: Optional[Tuple[int, ...]] = None
    response_prefill: bool = False
    agent_loop: bool = False
    thinking_on: bool = False
    #: RESUMED PREFILL (module docstring): how many of THIS request's
    #: prompt positions were served from the KV cache before this forward
    #: — 0 for a complete prefill (positions 0..P-1) and for every decode
    #: context; ``cached > 0`` makes a prefill context a resumed one whose
    #: rows sit at absolute positions ``cached..cached+T-1``. An op whose
    #: policy is :data:`RESUMED_NEVER` refuses under it.
    cached: int = 0
    #: the token ids of the CACHED region ``[0, cached)`` of a resumed
    #: prefill — required by a ``positions="response"`` op under a resumed
    #: context, whose locator VERIFIES the cached region is header-free
    #: (:func:`cached_region_header_conflict`) before masking, exactly as it
    #: locates the boundary instead of trusting a width. ``None`` on a
    #: complete prefill / decode context and on a resumed context serving
    #: ``always``-policy ops only.
    cached_ids: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.phase not in (PHASE_PREFILL, PHASE_DECODE):
            raise AdapterError(
                f"HookContext.phase must be {PHASE_PREFILL!r} or "
                f"{PHASE_DECODE!r}, got {self.phase!r}")
        if isinstance(self.cached, bool) or not isinstance(self.cached, int) \
                or self.cached < 0:
            raise AdapterError(
                f"HookContext.cached must be an int >= 0 (the request's "
                f"prompt positions already in the KV cache), got "
                f"{self.cached!r}")
        if self.phase == PHASE_DECODE and self.cached != 0:
            raise AdapterError(
                f"HookContext(phase='decode') does not take cached="
                f"{self.cached} — a decode row's cache state is its "
                "position; 'cached' describes a RESUMED PREFILL slice only")
        if self.cached_ids is not None:
            cids = tuple(int(t) for t in self.cached_ids)
            if any(isinstance(t, bool) for t in self.cached_ids):
                raise AdapterError("HookContext.cached_ids must be ints")
            if self.cached == 0:
                raise AdapterError(
                    "HookContext carries cached_ids but cached == 0 — the "
                    "cached region's ids belong to a RESUMED prefill only")
            if len(cids) != self.cached:
                raise AdapterError(
                    f"HookContext carries {len(cids)} cached_ids for a "
                    f"cached region of {self.cached} positions — the ids "
                    "must be exactly the request's prompt ids [0, cached); "
                    "refusing a mis-sized region")
            object.__setattr__(self, "cached_ids", cids)
        if not isinstance(self.response_prefill, bool):
            raise AdapterError(
                f"HookContext.response_prefill must be a bool, got "
                f"{self.response_prefill!r} — the seeded-render bit is a "
                "declared fact, never a truthy coercion")
        if not isinstance(self.agent_loop, bool):
            raise AdapterError(
                f"HookContext.agent_loop must be a bool, got "
                f"{self.agent_loop!r} — the agent-loop bit is a declared "
                "fact, never a truthy coercion")
        if self.response_prefill and self.agent_loop:
            raise AdapterError(
                "HookContext carries BOTH response_prefill and agent_loop — "
                "the two declarations describe disjoint request shapes (a "
                "continued final message vs a generation-prompt render with "
                "an open model turn), so a context claiming both is "
                "malformed; refusing")
        if not isinstance(self.thinking_on, bool):
            raise AdapterError(
                f"HookContext.thinking_on must be a bool, got "
                f"{self.thinking_on!r} — the thinking-ON render bit is a "
                "declared fact, never a truthy coercion")
        if self.thinking_on and (self.response_prefill or self.agent_loop):
            raise AdapterError(
                "HookContext carries thinking_on beside a non-tail "
                "declaration (response_prefill / agent_loop) — a declared "
                "thinking-ON render is an ordinary tail render under the ON "
                "header; a seeded or agent-loop ON render is not a shape "
                "this serving path admits, so a context claiming both is "
                "malformed; refusing (fit-support-transfer plan §2.1)")
        pos = tuple(int(p) for p in self.positions)
        if any(isinstance(p, bool) for p in self.positions):
            raise AdapterError("HookContext.positions must be ints")
        object.__setattr__(self, "positions", pos)
        if not pos:
            raise AdapterError("HookContext.positions must be non-empty")
        if any(p < 0 for p in pos):
            raise AdapterError(
                f"HookContext.positions must be >= 0, got {pos}")
        if self.phase == PHASE_PREFILL:
            want = tuple(range(self.cached, self.cached + len(pos)))
            if pos != want:
                if self.cached == 0:
                    raise AdapterError(
                        f"a prefill HookContext must carry the COMPLETE "
                        f"prompt (contiguous positions 0..P-1), got {pos} "
                        "— a chunked-prefill continuation is "
                        "unrepresentable by design, and a prefix-cache hit "
                        "is a RESUMED prefill, which declares its cached "
                        "prefix (cached=<n>) instead of hiding it")
                raise AdapterError(
                    f"a RESUMED prefill HookContext (cached={self.cached}) "
                    f"must carry the contiguous prompt tail at positions "
                    f"{self.cached}..{self.cached + len(pos) - 1}, got "
                    f"{pos} — the rows in the forward are exactly the "
                    "positions the cache did not serve; refusing a "
                    "mis-aligned slice")
        else:  # decode
            if len(pos) != 1:
                raise AdapterError(
                    f"a decode HookContext carries exactly ONE generated "
                    f"token's position, got {len(pos)}")
            if pos[0] < 1:
                raise AdapterError(
                    f"a decode position must be >= 1 (position 0 is the "
                    f"prompt's BOS), got {pos[0]}")
        if self.token_ids is not None:
            ids = tuple(int(t) for t in self.token_ids)
            if any(isinstance(t, bool) for t in self.token_ids):
                raise AdapterError("HookContext.token_ids must be ints")
            if len(ids) != len(pos):
                raise AdapterError(
                    f"HookContext carries {len(ids)} token ids for "
                    f"{len(pos)} positions — the ids must be THIS request's "
                    "rows, sliced the same way the hidden states are; "
                    "refusing a mis-aligned pair")
            object.__setattr__(self, "token_ids", ids)

    @property
    def is_resumed(self) -> bool:
        """True iff this is a RESUMED prefill context (``cached > 0``): the
        rows are the tail of a prompt whose head the KV cache served."""
        return self.phase == PHASE_PREFILL and self.cached > 0


@dataclass(frozen=True)
class DecodeRows:
    """The context of a VECTORIZED group of DECODE rows — one row per
    request, every request at a generated position (>= 1) — sharing the
    same declared render facts (the lens-vectorize landing, 2026-09-04).

    The hook wiring groups a forward's decode slots by (object, declared
    facts) and applies each object's ops ONCE to the stacked rows instead
    of once per request; :class:`HookContext` cannot describe that group
    (a decode context carries exactly ONE position by contract), so this
    is its group-shaped sibling. It carries no positions and no token
    ids on purpose: nothing an op consults in the decode phase depends on
    either — every decode row sits at a position >= 1, so ``skip_bos``
    never removes a row, ``positions="last"`` names every decode row, and
    the response region contains every generated position
    (:func:`_row_selection`). The two declared facts are kept because
    ``positions="last"`` REFUSES under a declared non-tail render, and
    that refusal must fire for a group exactly as for each of its rows."""

    response_prefill: bool = False
    agent_loop: bool = False

    def __post_init__(self):
        if not isinstance(self.response_prefill, bool) \
                or not isinstance(self.agent_loop, bool):
            raise AdapterError("DecodeRows facts must be bools")
        if self.response_prefill and self.agent_loop:
            raise AdapterError(
                "DecodeRows carries BOTH response_prefill and agent_loop — "
                "the two declarations describe disjoint request shapes; "
                "refusing")

    @property
    def phase(self) -> str:
        return PHASE_DECODE


#: what ``apply_hook_ops`` / ``apply_oproj_ops`` accept as the row
#: context: one request's :class:`HookContext`, or a vectorized decode
#: group's :class:`DecodeRows`.
RowContext = Union[HookContext, DecodeRows]


#: ``adapt(resolved_spec, artifact_bytes, sidecar_bytes, *, header_ids,
#: turn_end_ids)``.
#: ``header_ids`` (and ``turn_end_ids``, the all-turns mask's turn-end
#: marker) are SERVING-side context, deliberately NOT part of the
#: object's identity: the assistant header belongs to the chat template,
#: which the serving-identity stamp already pins. It is handed in by the
#: worker registry rather than read out of the hashed resolved spec —
#: putting it in the spec would make the intervention hash depend on the
#: tokenizer and break the rule that one intervention keeps one hash
#: across serving paths.
AdapterFn = Callable[..., Tuple[HookOp, ...]]

_ADAPTERS: Dict[str, AdapterFn] = {}

#: kinds that CANNOT be expressed as block-output hook ops, each with the
#: exact reason (get_adapter surfaces it verbatim). These refuse loudly —
#: never a lossy residual approximation served under the cell's hash.
UNSERVABLE_KIND_REASONS: Dict[str, str] = {
    "circuit_breaker": (
        "rank-16 LoRA over the seven q/k/v/o/gate/up/down projections of "
        "layers 0..max(target) (extraction/circuit_breaker."
        "CircuitBreakerObject): a weight-level intervention on 7 x ~21 "
        "projection INPUTS — not expressible as block-output residual "
        "ops. The committed serving path is the genuine PEFT adapter "
        "export + vLLM native multi-LoRA (two-SI D13, "
        "serving/lora_export.py)."),
    "token": (
        "soft token (extraction/objects.SoftTokenObject): needs the "
        "sentinel inserted into the RENDERED PROMPT (obj.inject) plus an "
        "input-embedding-layer hook replacing the sentinel's embedding "
        "with the trained vector — the embedding layer is not a decoder "
        "block-output site, and the replaced embedding propagates "
        "through attention from position 1, which no additive "
        "block-output op reproduces. The committed serving path is "
        "vLLM's prompt-embeds input (serving/soft_token.py): the client "
        "computes the injected prompt's input-embedding rows itself, "
        "substituting scale*param at the LOCATED sentinel position "
        "exactly as the class's hook does, and submits them as an "
        "EmbedsPrompt to an engine started with --enable-prompt-embeds "
        "— no hook, no enforce_eager, no prefix-cache disablement, so "
        "the arm serves on the fast lens-free server."),
    "lofit": (
        "per-head additive biases on the selected heads' PRE-o_proj "
        "attention outputs (extraction/lofit.LoFiTObject): the bias is "
        "applied mid-block (before o_proj and before the same block's "
        "MLP), so its block-output effect is input-dependent through the "
        "MLP — not a constant residual add. The o_proj MODULE is now "
        "hookable (HOOK_SITE_OPROJ, landed for the lora kinds), and the "
        "bias's o_proj-output image is a constant vector (W_o times the "
        "per-head biases), so a deliberate o_proj-site translation is "
        "possible — but it has not landed, and it must land WITH its own "
        "equivalence evidence, never as a here-and-now guess."),
    "cast": (
        "CAST (extraction/cast.CastObject) needs a per-request FIRE "
        "LATCH: the condition is evaluated ONCE on the prompt's prefill "
        "(pooled over all prompt positions at the condition layer) and "
        "then consumed by the behavior layers on every subsequent "
        "KV-cached decode forward of that request. The H3 worker hook "
        "contract is stateless and immutable (requests carry only "
        "immutable selectors; ops are pure per-token functions), so the "
        "cross-forward latch cannot be expressed; a request-scoped "
        "stateful hook contract must land deliberately WITH its own gate "
        "assertions (latch isolation under mixed batches), never as a "
        "guessed translation."),
    "fishback": (
        "FishBack (extraction/fishback.FishBackObject) is not a static "
        "intervention at all: ``apply`` refuses until a per-prompt "
        "``resolve(model, tok, prompt)`` has run an inference-time "
        "iterative metric solve (order a GPU-minute per case) for THAT "
        "exact prompt, and the hook then OVERWRITES the last prompt "
        "token's hidden state with the solved activation after "
        "byte-comparing the request's input ids against the resolved "
        "prompt. It is prompt-dependent, not additive, and needs "
        "autograd at inference — so it is unservable on every batched "
        "engine path, not merely on this one, and no export can rescue "
        "it. Moot in any case: the arm was REMOVED from the campaign "
        "(2026-08-20, its spec gate unsatisfiable). The loader survives "
        "so old artifacts re-hydrate; it is not a live arm."),
    "gated_fishback": (
        "the same two-phase resolve-then-inject design as 'fishback' "
        "(extraction/baseline_arms), behind a ReLU-probe gate — "
        "unservable for the identical reason: an inference-time "
        "per-prompt solve that no batched engine can host."),
}


def register_adapter(kind: str, fn: AdapterFn) -> None:
    """Register a per-kind adapter. Re-registering a kind is a refusal —
    two translations for one kind means one of them is wrong."""
    if not isinstance(kind, str) or not kind:
        raise AdapterError("adapter kind must be a non-empty string")
    if kind in _ADAPTERS:
        raise AdapterError(f"adapter for kind {kind!r} already registered — "
                           "one kind, one translation")
    if kind in UNSERVABLE_KIND_REASONS:
        raise AdapterError(
            f"kind {kind!r} is documented UNSERVABLE on the lens path — "
            "registering an adapter for it contradicts "
            "UNSERVABLE_KIND_REASONS; resolve the contradiction "
            "deliberately (remove the reason WITH the change that makes "
            "the kind servable)")
    # every registered adapter is wrapped with the fit/serve position-
    # regime guard (_check_fit_serve_positions) — get_adapter is the one
    # door to every serving path (capeval export, the engine's CPU
    # mirror, the GPU worker's install, the petri shim), so wrapping
    # here is what makes the guard unbypassable.
    _ADAPTERS[kind] = _fit_serve_guarded(kind, fn)


def get_adapter(kind: str) -> AdapterFn:
    """The adapter for ``kind``, or a LOUD refusal. An unknown kind means
    the cell is ABSENT (or a recorded HF exception via the §10 gate) —
    the caller reports it; it is never served by a guess."""
    fn = _ADAPTERS.get(kind)
    if fn is not None:
        return fn
    reason = UNSERVABLE_KIND_REASONS.get(kind)
    if reason is not None:
        raise AdapterError(
            f"no lens adapter registered for object kind {kind!r} — this "
            f"kind is UNSERVABLE on the lens path: {reason}")
    raise AdapterError(
        f"no lens adapter registered for object kind {kind!r} "
        f"(registered: {sorted(_ADAPTERS)}; documented-unservable: "
        f"{sorted(UNSERVABLE_KIND_REASONS)}) — an unknown kind is ABSENT "
        "or a recorded §10 gate failure, never served by a guessed "
        "translation")


def registered_kinds() -> Tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


# --- shared validation helpers -------------------------------------------------


def _refuse_sidecar(kind: str, sidecar_bytes: bytes) -> None:
    if sidecar_bytes:
        raise AdapterError(
            f"kind {kind!r} takes no sidecar bytes, got "
            f"{len(bytes(sidecar_bytes))} — a behavior-bearing sidecar on "
            "this kind means the entry is not what this adapter translates "
            "(the sidecar is hashed into the intervention identity; it must "
            "never be silently dropped)")


def _torch_blob(kind: str, artifact_bytes: bytes) -> Dict[str, Any]:
    """The kind's torch-blob artifact, loaded and kind-verified."""
    if not artifact_bytes:
        raise AdapterError(f"kind {kind!r}: empty artifact bytes")
    try:
        blob = torch.load(io.BytesIO(bytes(artifact_bytes)),
                          map_location="cpu", weights_only=True)
    except Exception as e:  # torch raises several types here
        raise AdapterError(
            f"kind {kind!r}: artifact bytes did not load as a torch "
            f"payload: {e}") from e
    if not isinstance(blob, dict) or "kind" not in blob:
        raise AdapterError(
            f"kind {kind!r}: artifact is not a torch blob with a 'kind' "
            "field (the load_object contract)")
    if blob["kind"] != kind:
        raise AdapterError(
            f"kind {kind!r}: artifact stores kind {blob['kind']!r} — the "
            "registry mapped this cell to the wrong adapter; refusing")
    return blob


def _spec_check(kind: str, spec: Mapping[str, Any], key: str,
                expected: Any) -> None:
    """F9-style cross-check: when the resolved spec carries ``key`` it
    must agree with the artifact's own stored value."""
    if key in spec and spec[key] != expected:
        raise AdapterError(
            f"kind {kind!r}: resolved_spec {key}={spec[key]!r} != artifact "
            f"{key}={expected!r} (F9-style cross-check) — the spec and the "
            "bytes disagree; refusing")


def _f32_1d(kind: str, name: str, t: Any, dim: Optional[int] = None
            ) -> torch.Tensor:
    if not isinstance(t, torch.Tensor) or t.dim() != 1:
        raise AdapterError(f"kind {kind!r}: {name} must be a 1-D tensor, "
                           f"got {type(t).__name__}"
                           + (f" of shape {tuple(t.shape)}"
                              if isinstance(t, torch.Tensor) else ""))
    if dim is not None and t.shape[0] != dim:
        raise AdapterError(f"kind {kind!r}: {name} has dim {t.shape[0]}, "
                           f"expected {dim}")
    if not torch.isfinite(t).all():
        raise AdapterError(f"kind {kind!r}: {name} carries non-finite "
                           "values — refusing to serve a corrupt tensor")
    return t.detach().to(torch.float32)


def _layer_list(kind: str, raw: Any) -> Tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise AdapterError(f"kind {kind!r}: 'layers' must be a non-empty "
                           f"list, got {raw!r}")
    layers = []
    for l in raw:
        li = int(l)
        if li < 0:
            raise AdapterError(f"kind {kind!r}: layer {li} < 0")
        if li in layers:
            raise AdapterError(f"kind {kind!r}: duplicate layer {li}")
        layers.append(li)
    return tuple(layers)


#: what a stored ``positions`` value means on the lens path. Anything
#: absent from this map rides the reference path — never a silent "all"
#: substitution, which would serve a wider intervention than was fit.
_SERVED_POSITION_MODES = {"all": POSITIONS_ALL, "response": POSITIONS_RESPONSE,
                          "response_all_turns": POSITIONS_RESPONSE_ALL_TURNS}


def _served_positions(kind: str, stored: Any,
                      header_ids: Optional[Tuple[int, ...]],
                      resolved_spec: Optional[Mapping[str, Any]] = None,
                      turn_end_ids: Optional[Tuple[int, ...]] = None,
                      ) -> Tuple[str, Optional[Tuple[int, ...]],
                                 Optional[Tuple[int, ...]]]:
    """The position mode to SERVE, plus its header ids and — for the
    all-turns mode — its terminator ids (``None`` for every other mode).

    THE MODE BELONGS TO THE CELL, NOT TO THE OBJECT. A cross-phase cell
    IS an all-position artifact served under the response mask — that is
    what the axis means — so the resolved spec's ``positions`` OVERRIDES
    the artifact's, which is only a default. An earlier version made the
    artifact authoritative and cross-checked the spec against it,
    refusing on disagreement; that made a cross-phase cell unservable by
    construction, which is the exact opposite of the intent.

    Nothing is smuggled by the override: the intervention hash covers
    the resolved spec, so the same bytes served at ``all`` and at
    ``response`` hash differently and are correctly two distinct cells.
    Contrast the LoRA export, which REFUSES to serve a response-fit
    adapter unmasked — there the mask is not expressible at all on the
    target path, so serving it anyway would misreport what ran.

    ``response_all_turns`` (THE ALL-TURNS MASK, module docstring) is
    requested the same way — a CELL's ``positions`` — and needs BOTH
    markers: the header to open each assistant span and the turn-end
    scaffold to close every earlier one. A registry exported without one
    (a successor-bounded template such as GLM-4.x has no terminator)
    cannot serve the mode and refuses HERE, at adapt time, never per
    request.

    Only arms satisfying DERIVATION LOCALITY may be re-masked (see the
    module docstring); the caller decides, and an arm that fails it
    never routes here."""
    requested = stored
    if resolved_spec is not None \
            and resolved_spec.get("positions") is not None:
        requested = resolved_spec["positions"]
    mode = _SERVED_POSITION_MODES.get(requested)
    if mode is None:
        raise AdapterError(
            f"kind {kind!r}: positions={requested!r} is not lens-served "
            f"(served modes: {sorted(_SERVED_POSITION_MODES)}). The "
            "remaining hooks.py modes carry position semantics this op "
            "vocabulary does not express; they ride the reference path "
            "until a deliberate mapping lands, never a silent 'all' "
            "substitution — which would serve a WIDER intervention than "
            "the one that was fit")
    if mode not in RESPONSE_MASK_MODES:
        return mode, None, None
    if not header_ids:
        raise AdapterError(
            f"kind {kind!r} is served at positions={mode!r} (the "
            "cross-phase mask) but the worker registry supplied no "
            "assistant header — "
            "the served region is located by matching the header's token "
            "ids, so without them the cell cannot be served. Derive them "
            "with chat_markers.generation_header_ids against the served "
            "tokenizer and pass them to the registry")
    if mode == POSITIONS_RESPONSE:
        return mode, tuple(header_ids), None
    if not turn_end_ids:
        raise AdapterError(
            f"kind {kind!r} is served at positions='response_all_turns' "
            "(every assistant turn's span) but the worker registry supplied "
            "no turn-end scaffold — an earlier turn's span ends at its "
            "terminator-plus-glue scaffold, LOCATED by matching those token "
            "ids (chat_markers.assistant_turn_end_ids), so without them the "
            "cell cannot be served. Record assistant_turn_end_ids in the "
            "registry; a successor-bounded template (GLM-4.x: assistant "
            "turns end bare, no terminator) has no locatable turn end and "
            "cannot serve this mode at all")
    return mode, tuple(header_ids), tuple(turn_end_ids)


def _normalized_position_record(kind: str, source: str, value: Any,
                                vocab: Mapping[str, str]) -> str:
    """One position-regime record, validated against ITS OWN declared
    vocabulary and normalized to the guard's regime — never read as if
    every record spoke the fit-leg vocabulary (the loreft mis-read)."""
    if not isinstance(value, str):
        raise AdapterError(
            f"kind {kind!r}: {source} must be a string, got {value!r}")
    if value not in vocab:
        raise AdapterError(
            f"kind {kind!r}: {source}={value!r} is not a known fit "
            f"position mode ({sorted(vocab)}) — refusing to "
            "guess what the fit meant")
    return vocab[value]


def recorded_fit_positions(kind: str, artifact_bytes: bytes,
                           sidecar_bytes: bytes) -> Optional[str]:
    """The position regime the artifact's own HASHED bytes record it was
    FIT under — "all" / "response" (the PositionMaskedFit legs) or
    "prompt" (loreft's kind-intrinsic identity fit) — or ``None`` when
    nothing records one.

    Sources, all part of the hashed intervention identity:

    - ``positions_at_fit`` (JSON sidecar or torch blob) — the UNIVERSAL
      explicit fit-leg key (the ``lora_export._check_sidecar``
      vocabulary — one refusal vocabulary, two serving paths), always in
      :data:`FIT_POSITION_MODES`.
    - the kind's own save-schema keys, read ONLY through
      :data:`_BLOB_POSITION_SCHEMAS` — the blob key ``positions`` means
      three different things across the save schemas (fit leg /
      deployment attr / loreft's intrinsic policy; see the table), so
      each kind's record is validated against ITS OWN declared
      vocabulary and normalized. A blob that carries a position-record
      key its kind's schema does not declare refuses: an undeclared
      vocabulary read as any particular one is a guess.

    Every record present must agree (after normalization); a
    disagreement means the fit's own records contradict each other and
    is a refusal, never a pick.

    Deliberately NOT consulted: ``train_budget_meta.json`` (it sits next
    to the artifact where the fit wrote it but is OUTSIDE the hashed
    bytes and absent from every worker registry dir), and static_r /
    alphasteer's ``steer_positions`` (a deployment-shape key in a
    different vocabulary, already cross-checked per adapter)."""
    records = []  # (source, raw value, normalized regime)
    if sidecar_bytes:
        try:
            side = json.loads(bytes(sidecar_bytes).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            side = None  # sidecar syntax is the consuming adapter's call
        if isinstance(side, dict) and "positions_at_fit" in side:
            raw = side["positions_at_fit"]
            records.append(("sidecar positions_at_fit", raw,
                            _normalized_position_record(
                                kind, "sidecar positions_at_fit", raw,
                                _MASKED_FIT_VOCAB)))
    if artifact_bytes:
        try:
            blob = torch.load(io.BytesIO(bytes(artifact_bytes)),
                              map_location="cpu", weights_only=True)
        except Exception:
            blob = None  # not a torch blob (safetensors etc.) — no record
        if isinstance(blob, dict):
            if "positions_at_fit" in blob:
                raw = blob["positions_at_fit"]
                records.append(("blob positions_at_fit", raw,
                                _normalized_position_record(
                                    kind, "blob positions_at_fit", raw,
                                    _MASKED_FIT_VOCAB)))
            schema = _BLOB_POSITION_SCHEMAS.get(kind, ())
            declared = {key for key, _ in schema}
            undeclared = [key for key in _POSITION_RECORD_KEYS
                          if key in blob and key not in declared]
            if undeclared:
                raise AdapterError(
                    f"kind {kind!r}: the artifact carries position-record "
                    f"key(s) {undeclared} that this kind's save schema "
                    "does not declare (_BLOB_POSITION_SCHEMAS) — the same "
                    "key name spans several vocabularies (a "
                    "PositionMaskedFit leg, a serve-time deployment attr, "
                    "loreft's intrinsic policy), so reading it as any one "
                    "of them would be a guess about what the fit meant; "
                    "declare the kind's schema before serving it")
            for key, vocab in schema:
                if key in blob:
                    raw = blob[key]
                    records.append(
                        (f"blob {key}", raw,
                         _normalized_position_record(kind, f"blob {key}",
                                                     raw, vocab)))
                    # the FIRST present schema key IS the record; later
                    # keys are the schema's serve-time defaults
                    # (vector_all's deployment attr under an explicit
                    # fit leg), not independent fit claims
                    break
    if not records:
        return None
    values = {norm for _, _, norm in records}
    if len(values) > 1:
        listed = ", ".join(f"{src}={raw!r}" for src, raw, _ in records)
        raise AdapterError(
            f"kind {kind!r}: {listed} — the fit's own records disagree "
            "about what was trained; refusing to pick one")
    return records[0][2]


def _object_leg_spec(resolved_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """The spec that carries a cell's served-position and serve-declaration
    keys: the spec itself, or — for a STACKED wrapper — its ``object`` leg
    (the wrapper carries none of its own), exactly as the guard and
    ``lens_steer._assert_ops_express_spec_positions`` unwrap it."""
    if isinstance(resolved_spec, Mapping) \
            and resolved_spec.get("kind") == STACKED_KIND:
        obj = resolved_spec.get("object")
        if isinstance(obj, Mapping):
            return obj
    return resolved_spec if isinstance(resolved_spec, Mapping) else {}


def cell_declarations(resolved_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The cell's validated serve declaration (``serve_declaration.
    declarations_of`` over the object leg's spec): ``{}`` for an undeclared
    cell, else exactly one of the two closed shapes. A malformed
    declaration is an :class:`AdapterError` — refused at adapt, never read
    as undeclared."""
    try:
        return sd.declarations_of(_object_leg_spec(resolved_spec))
    except sd.ServeDeclarationError as e:
        raise AdapterError(f"malformed serve declaration in the resolved "
                           f"spec: {e}") from e


def declares_thinking(resolved_spec: Mapping[str, Any]) -> bool:
    """Whether the cell's hashed declaration names the thinking-ON render
    (``serve_out_of_support`` carrying ``"thinking"``) — the fact that
    decides whether a second header is attached to its response ops
    (:func:`attach_thinking_header`), required in the registry index, and
    reported in ``lens_serve_declarations``."""
    decl = cell_declarations(resolved_spec)
    return sd.OUT_OF_SUPPORT_THINKING in decl.get(sd.SERVE_OUT_OF_SUPPORT_KEY, [])


def _declared_experiment(kind: str, resolved_spec: Mapping[str, Any]
                         ) -> Optional[str]:
    """The named transfer experiment a cell declares (a
    ``serve_declaration.SERVE_EXPERIMENTS`` member), or ``None`` — read
    strictly (a malformed declaration refuses), for the guard's named
    carve-out."""
    try:
        return cell_declarations(resolved_spec).get(sd.SERVE_EXPERIMENT_KEY)
    except AdapterError as e:
        raise AdapterError(f"kind {kind!r}: {e}") from e


def _check_fit_serve_positions(kind: str, resolved_spec: Mapping[str, Any],
                               artifact_bytes: bytes, sidecar_bytes: bytes,
                               ops: Sequence[HookOp]) -> Optional[str]:
    """THE FIT/SERVE POSITION-REGIME GUARD — the lens-path mirror of
    ``lora_export``'s ``_gate_positions`` + ``_check_sidecar`` (audit
    2026-08-21 / decode-only steering audit 2026-08-26): the two-SI
    objects come in two INCOMPATIBLE position regimes (all-token vs
    decode-only), and an object fit under one must never be served under
    the other. Runs on EVERY adapt (:func:`register_adapter` wraps each
    adapter), so no lens-served cell — capeval export, the engine's CPU
    mirror, or the GPU worker's install — can bypass it.

    The rules, in terms of the artifact's recorded fit regime
    (:func:`recorded_fit_positions`) and the positions the produced ops
    actually serve:

    - fit ``"response"`` (decode-only): every op must serve a
      HEADER-LOCATED mask — ``response`` itself, or THE ONE DELIBERATE,
      NAMED CARVE-OUT: ``response_all_turns`` (the all-assistant-turns
      mask, module docstring). The carve-out rests on one judgement,
      stated here so it cannot be widened by analogy: every span the
      all-turns mask edits is an ASSISTANT-TURN span (a header through its
      own turn's end scaffold, plus the generated positions) — the same
      KIND of position the object was fitted on — so it is the response
      regime applied to MORE spans of that kind, not a different regime.
      Fit-response / serve-``all`` is the opposite: it edits prompt
      positions of every kind (system, user, tool), a wider-position
      APPROXIMATION of a masked fit, which is exactly what this guard
      exists to forbid and keeps forbidding. Anything else silently runs
      the all-token method under a decode-only artifact's hash — the trap
      PSRObject.save's fit leg exists to prevent; refuse.
    - ``response_all_turns`` served under ANY other record refuses: the
      carve-out above is sound only when the fit is KNOWN to be
      "response". An ``all`` fit already steers every turn at ``all`` and
      has its own hashed cross-phase re-mask spelled ``response``; a third
      deployment of it needs its own carve-out, never this one. An ABSENT
      record is the class-documented all-position fit (below) and refuses
      here for the same reason — a positive "response" record is required.
    - fit ``"all"``: any deployment EXCEPT the response mask passes (a
      "last"/prefill-only deployment is the class's own narrow serve of
      an all-position fit). The response mask passes only when the
      resolved spec REQUESTS it — the deliberate, hashed cross-phase
      re-mask (`_served_positions`: the mode belongs to the CELL), which
      derivation-locality-failing adapters already refuse on their own.
    - fit ``"prompt"`` (loreft's kind-INTRINSIC regime, in that class's
      own blob vocabulary — see :data:`_BLOB_POSITION_SCHEMAS`): every
      op must be prefill-only and none may wear the response mask — the
      class's identity is "prompt positions during the prefill, decode
      untouched", so anything decode-reaching runs a different
      intervention under the fit's hash (the psr bug's shape on another
      kind).
    - THE NAMED CARVE-OUTS: a resolved
      spec carrying ``serve_experiment`` (``serve_declaration.
      SERVE_EXPERIMENT_KEY``, hashed into the cell's identity and visible
      in its name as ``<cell>@all_positions`` / ``<cell>@all_turns``) names
      exactly ONE ``(recorded fit, served positions)`` pair
      (``serve_declaration.EXPERIMENT_FIT_SERVE_PAIRS``). The guard PASSES a
      named cell iff the artifact's recorded fit and the ops' served
      positions are that pair — so ``fit_response_serve_all_positions``
      admits fit-``response`` / serve-``all``, the collapse corner the
      rules above forbid for every UNNAMED cell, as a deliberate transfer
      experiment whose collapse is measured beside its ASR — and REFUSES a
      name whose pair the deployment does not match (a name on an all-fit,
      on an absent record, or on ops that serve something else), and any
      name outside the closed vocabulary. ``fit_response_serve_all_turns``
      names the response/all-turns pair the unnamed rule already admits;
      naming it adds validation and attestation, never behaviour.
    - no record: pass, returning ``None`` for the caller to RECORD (the
      export index stamps ``positions_at_fit`` per cell — a stated
      absence, never a silent omission). NOT a refusal, deliberately:
      zero on-disk artifacts carry ``positions_at_fit`` and only the
      PositionMaskedFit family persists a fit leg at all (surveyed
      2026-08-27: 284 blobs — 164 with ``positions``, 0 with
      ``positions_at_fit``, 0 sidecars carrying either), so refusing
      absence would refuse essentially every legitimate artifact ever
      fit, including whole classes (alphakl and the closed-form arms)
      whose trainers REFUSE masked fits and whose loaders pin an absent
      key to the historical all-token fit by contract. An absent record
      therefore IS the class-documented all-position fit; the un-provable
      residue (a pre-audit response-fit blob saved before the fit leg
      was persisted) is not detectable from bytes that never recorded it.
    """
    if kind == STACKED_KIND:
        # a stacked cell's artifact IS its object leg's blob and its
        # wrapper spec carries no serve keys (_adapt_stacked delegated
        # the translation already) — evaluate the guard in the OBJECT
        # kind's own vocabulary and spec, exactly as the delegated
        # inner adapt did
        obj_spec = resolved_spec.get("object")
        if isinstance(obj_spec, Mapping) \
                and isinstance(obj_spec.get("kind"), str):
            kind = obj_spec["kind"]
            resolved_spec = obj_spec
    recorded = recorded_fit_positions(kind, artifact_bytes, sidecar_bytes)
    served = sorted({op.positions for op in ops})
    experiment = _declared_experiment(kind, resolved_spec)
    if POSITIONS_RESPONSE_ALL_TURNS in served \
            and recorded != POSITIONS_RESPONSE:
        raise AdapterError(
            f"kind {kind!r}: this cell would serve positions="
            f"{POSITIONS_RESPONSE_ALL_TURNS!r} (every assistant turn's "
            f"span) but the artifact records fit regime {recorded!r} — the "
            "all-turns mask is admitted for a KNOWN 'response' fit only: it "
            "is the response regime applied to more assistant-turn spans, "
            "and that argument needs the fit to BE response. An "
            "all-position fit already steers every turn at "
            "positions='all' (and its hashed cross-phase re-mask is "
            "spelled 'response'); an absent record is the class-documented "
            "all-position fit. A third deployment of an all-position fit "
            "needs its own carve-out here, never this one; refusing")
    if experiment is not None:
        # THE NAMED CARVE-OUT (docstring): the name describes exactly one
        # (recorded fit, served positions) pair; the deployment must BE
        # that pair, and then it is admitted BY NAME — the hashed, visible
        # declaration the spec's exit criterion demands, never a silent
        # pass and never a mask override's fallout.
        want_fit, want_served = sd.EXPERIMENT_FIT_SERVE_PAIRS[experiment]
        if recorded != want_fit or served != [want_served]:
            raise AdapterError(
                f"kind {kind!r}: the cell declares serve_experiment="
                f"{experiment!r}, which describes exactly the deployment "
                f"(fit {want_fit!r}, served positions=[{want_served!r}]) — "
                f"but this cell records fit regime {recorded!r} and would "
                f"serve positions={served}. A named transfer experiment must "
                "describe the deployment it names, or the attested name "
                "would mislabel what ran; refusing (fit-support-transfer plan "
                "§2.2)")
        return recorded
    if recorded is None:
        return None
    if recorded == POSITIONS_RESPONSE:
        if not served or any(m not in RESPONSE_MASK_MODES for m in served):
            raise AdapterError(
                f"kind {kind!r}: the artifact records a DECODE-ONLY fit "
                f"(fit regime {recorded!r}) but this cell would serve "
                f"positions={served} — an object fit under one position "
                "regime must never be served under the other (the fit "
                "driver's contract; the PEFT export path refuses the "
                "same mismatch in serving/lora_export.py). Serve this "
                "artifact under the response mask on a path that "
                "expresses it — 'response', or 'response_all_turns' (the "
                "one carve-out: the same response regime over EVERY "
                "assistant-turn span, still assistant-turn positions "
                "only) — or refit it all-position; never a wider-position "
                "approximation of a masked fit. A deliberate "
                "fit-response/serve-all transfer cell is admitted ONLY by "
                "name: declare `serve_experiment: "
                f"{sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS}` on the "
                "CELL (spelled `<cell>@all_positions`), so the collapse "
                "corner runs as a hashed, attested experiment — never a "
                "silent pass (fit-support-transfer plan §2.2)")
    elif recorded == POSITIONS_PROMPT:
        phases = sorted({op.phase for op in ops})
        if phases != [PHASE_PREFILL] \
                or any(m in RESPONSE_MASK_MODES for m in served):
            raise AdapterError(
                f"kind {kind!r}: the artifact records the kind-intrinsic "
                f"PROMPT-POSITIONS fit (fit regime {recorded!r}) but this "
                f"cell would serve positions={served} in phase(s)="
                f"{phases} — the class's identity is prompt positions "
                "during the prefill with decode untouched "
                "(reft.LoReFTObject's seq_len gate, which PHASE_PREFILL "
                "expresses exactly), so a decode-reaching serve runs a "
                "DIFFERENT intervention under the fit's hash. A "
                "decode-covering serve of this arithmetic is its own "
                "deliberate translation (the loreft_both kind), never "
                "a silent widen here")
    elif (POSITIONS_RESPONSE in served
            and resolved_spec.get("positions") != POSITIONS_RESPONSE):
        raise AdapterError(
            f"kind {kind!r}: the artifact records an ALL-POSITION fit "
            f"(fit regime {recorded!r}) but this cell would serve "
            f"positions={served} without the resolved spec requesting "
            "the response mask — the translation disagrees with the "
            "fit's own record; refusing (a cross-phase cell is spelled "
            "positions='response' in the CELL spec, where it is hashed)")
    return recorded


def _fit_serve_guarded(kind: str, fn: AdapterFn) -> AdapterFn:
    """Wrap an adapter so the fit/serve position-regime guard runs after
    every successful adapt — installed by :func:`register_adapter`, so
    the guard rides :func:`get_adapter` on every serving path and a new
    adapter cannot forget it. Pass-through of the ops, bit-untouched:
    this is a guard, never a rewrite."""
    @functools.wraps(fn)
    def guarded(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                sidecar_bytes: bytes = b"", *,
                header_ids: Optional[Tuple[int, ...]] = None,
                turn_end_ids: Optional[Tuple[int, ...]] = None
                ) -> Tuple[HookOp, ...]:
        ops = fn(resolved_spec, artifact_bytes, sidecar_bytes,
                 header_ids=header_ids, turn_end_ids=turn_end_ids)
        _check_fit_serve_positions(kind, resolved_spec, artifact_bytes,
                                   sidecar_bytes, ops)
        return ops
    return guarded


def _block_output_layer(kind: str, layer: int, site: str) -> int:
    """The block-output layer index that carries a ``site`` edit.

    Lens ops apply at a decoder block's OUTPUT, so a resid_post object
    serves at its own index and a resid_pre object serves one block
    EARLIER — resid_pre of layer L is the output of layer L-1. Layer 0 is
    the exception with no fix: its resid_pre is the embedding output, and
    there is no preceding block to hang the edit on."""
    if site == "post":
        return layer
    if site != "pre":
        raise AdapterError(
            f"kind {kind!r}: unknown hook site {site!r} (expected 'pre' or "
            "'post') — the site decides which block's output carries the "
            "edit, so guessing it would silently steer the wrong layer")
    if layer == 0:
        raise AdapterError(
            f"kind {kind!r}: fit at resid_pre of layer 0, which is the "
            "embedding output — there is no preceding decoder block whose "
            "output could carry the edit, so it is unservable on the lens "
            "path. Serving it one block later would be a different "
            "intervention, not a rounding of this one")
    return layer - 1


#: how far a loaded tensor's norm may sit from the norm recorded at
#: fit time for that layer. The check exists to catch a MIS-KEYED
#: lookup, and on the released matched object the per-layer norms are
#: strictly monotone in depth with a smallest adjacent gap of ~0.25 —
#: so this tolerance leaves roughly two orders of magnitude of margin
#: at the tightest point while still admitting fp32 round-trip noise.
_NORM_ATOL = 1e-3


def _recorded_layer_norms(kind: str, blob: Mapping[str, Any], key: str
                          ) -> Dict[int, float]:
    """The fit lane's recorded per-layer tensor norms, if the artifact
    carries them. Absence is fine — older artifacts predate the field
    and the rest of the validation still applies."""
    meta = blob.get("metadata")
    if not isinstance(meta, Mapping):
        return {}
    recorded = meta.get(key)
    if not isinstance(recorded, Mapping):
        return {}
    out: Dict[int, float] = {}
    for k, v in recorded.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError) as e:
            raise AdapterError(
                f"kind {kind!r}: metadata[{key!r}] entry {k!r}={v!r} is not "
                "a layer->number pair; refusing to skip an integrity check "
                "because its own record is malformed") from e
    return out


def _assert_norms_discriminate(kind: str, recorded: Mapping[int, float]
                               ) -> None:
    """Refuse a recorded-norm set that cannot tell two layers apart.

    The cross-check below exists to localize a MIS-KEYED lookup, and it
    can only do that if the layers' norms differ. A set that does not
    discriminate passes for every possible permutation — it would report
    success while checking nothing, which is worse than having no check
    at all, because the false assurance is invisible.

    This is not hypothetical: a sibling arm's directions are all exactly
    unit-norm, and the metadata beside them records precisely those
    norms. Wiring this helper to that record would have "verified" all
    ten layers vacuously. So an artifact that records non-discriminating
    norms for a multi-layer object is a refusal, with the instruction to
    record the norm of whichever tensor actually discriminates."""
    if len(recorded) < 2:
        return  # single-layer: there is no other layer to confuse it with
    vals = sorted(recorded.values())
    gap = min(b - a for a, b in zip(vals, vals[1:]))
    if gap <= _NORM_ATOL * 10:
        raise AdapterError(
            f"kind {kind!r}: the recorded per-layer norms do not "
            f"discriminate between layers (smallest gap {gap:.6g}, needs "
            f"> {_NORM_ATOL * 10:g}) — a cross-check against them would "
            "pass for EVERY permutation of the layers, reporting success "
            "while verifying nothing. Record the per-layer norm of "
            "whichever tensor is actually discriminating for this kind, "
            "and name which; refusing rather than serving under a vacuous "
            "integrity check")


def _check_recorded_norm(kind: str, layer: int, tensor: torch.Tensor,
                         recorded: Mapping[int, float]) -> None:
    """Cross-check one loaded tensor against the norm the fit recorded
    FOR THAT LAYER.

    This is the check that catches a mis-keyed lookup, which nothing
    downstream would. A multi-layer object fit at resid_pre is served
    one block earlier, and the fit and shifted layer sets overlap
    heavily — on the released matched object, seven of ten. So keying
    the tensor lookup by the SHIFTED index instead of shifting only the
    serving location finds a real, valid tensor for most layers and
    steers one block off: fluent output, plausible behaviour, wrong
    everywhere. Comparing against a per-layer norm turns that into a
    startup refusal, because the norms differ between layers."""
    want = recorded.get(layer)
    if want is None:
        return
    got = float(tensor.detach().float().norm())
    if abs(got - want) > _NORM_ATOL:
        raise AdapterError(
            f"kind {kind!r}: the tensor loaded for layer {layer} has norm "
            f"{got:.6f} but the artifact records {want:.6f} for that layer "
            f"(tolerance {_NORM_ATOL}). The tensor and the layer it is "
            "being served under disagree — the usual cause is keying the "
            "lookup by the SHIFTED serving index rather than the fit "
            "layer, which finds a valid tensor at the wrong depth. "
            "Refusing at STARTUP")


def _steering_vector_container(kind: str, artifact_bytes: bytes,
                               sidecar_bytes: bytes
                               ) -> Tuple[Dict[int, Any], Mapping[str, Any]]:
    """A ``SteeringVector`` artifact: the safetensors container keyed
    ``layer_<k>`` plus its REQUIRED JSON sidecar (method + metadata —
    what ``load_object`` dispatches on, and part of the hashed identity,
    so it can never be silently dropped)."""
    if not artifact_bytes:
        raise AdapterError(f"kind {kind!r}: empty artifact bytes")
    if not sidecar_bytes:
        raise AdapterError(
            f"kind {kind!r}: the SteeringVector JSON sidecar is REQUIRED — "
            "it carries the method the loader dispatches on, and two "
            "containers differing only in sidecar are different "
            "interventions")
    try:
        from safetensors.torch import load as _st_load
        raw = _st_load(bytes(artifact_bytes))
    except Exception as e:
        raise AdapterError(
            f"kind {kind!r}: artifact bytes did not load as a safetensors "
            f"container: {e}") from e
    try:
        side = json.loads(bytes(sidecar_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AdapterError(
            f"kind {kind!r}: unparseable JSON sidecar: {e}") from e
    if not isinstance(side, dict):
        raise AdapterError(f"kind {kind!r}: sidecar is not a JSON object")
    tensors: Dict[int, Any] = {}
    for name, t in raw.items():
        if not name.startswith("layer_"):
            raise AdapterError(
                f"kind {kind!r}: unexpected safetensors entry {name!r} — "
                "not a SteeringVector container")
        try:
            l = int(name.removeprefix("layer_"))
        except ValueError as e:
            raise AdapterError(
                f"kind {kind!r}: unparseable layer key {name!r}") from e
        if l < 0:
            raise AdapterError(f"kind {kind!r}: layer {l} < 0")
        tensors[l] = t
    if not tensors:
        raise AdapterError(f"kind {kind!r}: container holds no layer vectors")
    if "layers" in side and sorted(int(l) for l in side["layers"]) != \
            sorted(tensors):
        raise AdapterError(
            f"kind {kind!r}: sidecar layers {side['layers']} != container "
            f"layers {sorted(tensors)} — inconsistent artifact")
    return tensors, side


# --- the synthetic + zero kinds (tests + the no-donor no-op leg) ---------------


def _adapt_synthetic(resolved_spec: Mapping[str, Any],
                     artifact_bytes: bytes,
                     sidecar_bytes: bytes = b"", *,
                     header_ids: Optional[Tuple[int, ...]] = None,
                     turn_end_ids: Optional[Tuple[int, ...]] = None
                     ) -> Tuple[HookOp, ...]:
    _refuse_sidecar(SYNTHETIC_KIND, sidecar_bytes)
    if not artifact_bytes:
        raise AdapterError(f"kind {SYNTHETIC_KIND!r}: empty artifact bytes")
    try:
        payload = torch.load(io.BytesIO(bytes(artifact_bytes)),
                             map_location="cpu", weights_only=True)
    except Exception as e:  # torch raises several types here
        raise AdapterError(
            f"kind {SYNTHETIC_KIND!r}: artifact bytes did not load as a "
            f"torch payload: {e}") from e
    if (not isinstance(payload, dict)
            or set(payload) != {"layer", "vector"}):
        raise AdapterError(
            f"kind {SYNTHETIC_KIND!r}: artifact must be exactly "
            f"{{'layer', 'vector'}}, got "
            f"{sorted(payload) if isinstance(payload, dict) else type(payload).__name__}")
    layer = payload["layer"]
    vector = payload["vector"]
    if isinstance(layer, torch.Tensor):
        layer = int(layer.item())
    if not isinstance(vector, torch.Tensor) or vector.dim() != 1:
        raise AdapterError(f"kind {SYNTHETIC_KIND!r}: 'vector' must be a "
                           "1-D tensor")
    spec_layer = resolved_spec.get("layer")
    if spec_layer is not None and int(spec_layer) != int(layer):
        raise AdapterError(
            f"kind {SYNTHETIC_KIND!r}: resolved_spec layer {spec_layer!r} "
            f"!= artifact layer {layer!r} (F9-style cross-check)")
    return (HookOp(op=OP_ADD_VECTOR, layer=int(layer),
                   vector=vector.to(torch.float32)),)


def _adapt_zero(resolved_spec: Mapping[str, Any],
                artifact_bytes: bytes,
                sidecar_bytes: bytes = b"", *,
                header_ids: Optional[Tuple[int, ...]] = None,
                turn_end_ids: Optional[Tuple[int, ...]] = None
                ) -> Tuple[HookOp, ...]:
    if artifact_bytes:
        raise AdapterError(f"kind {ZERO_KIND!r} takes no artifact bytes")
    _refuse_sidecar(ZERO_KIND, sidecar_bytes)
    hidden = resolved_spec.get("hidden_size")
    layer = resolved_spec.get("layer")
    if not isinstance(hidden, int) or hidden <= 0:
        raise AdapterError(f"kind {ZERO_KIND!r}: resolved_spec.hidden_size "
                           f"must be a positive int, got {hidden!r}")
    if not isinstance(layer, int) or layer < 0:
        raise AdapterError(f"kind {ZERO_KIND!r}: resolved_spec.layer must "
                           f"be a non-negative int, got {layer!r}")
    return (HookOp(op=OP_ADD_VECTOR, layer=layer,
                   vector=torch.zeros(hidden, dtype=torch.float32)),)


# --- the real kinds -------------------------------------------------------------


def _adapt_psr(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
               sidecar_bytes: bytes = b"", *,
               header_ids: Optional[Tuple[int, ...]] = None,
               turn_end_ids: Optional[Tuple[int, ...]] = None
               ) -> Tuple[HookOp, ...]:
    """PSR (extraction/psr.PSRObject): per stored layer l,
    ``h += scale * relu(h_fp32 · w_l + b_l) * d_l`` — the gate computed
    in fp32 on the PRE-steering hidden state, the addition cast back to
    the model dtype. Prefill AND decode, NO BOS skip in any mode (the
    gate is learned; down-weighting BOS/outlier tokens is its job, not
    a mask's). Multi-layer (psr_kl_all / psr_mse_all store all layers).

    POSITIONS ARE THE FIT LEG'S + THE CELL'S, never hardcoded
    (:func:`_fit_leg_positions`, the lora kinds' seam): the artifact's
    recorded fit mask is the default, the resolved spec's ``positions``
    overrides it, and a "response" serve carries the located header ids
    plus the ``response_header_len`` width cross-check. PSR satisfies
    DERIVATION LOCALITY (module docstring) — the gate reads each
    position's own hidden state — so the mask is a pure positional
    subset of the every-position deployment.

    An earlier version served EVERY position unconditionally on the
    theory that "the gate is learned, masking is its job". That belief
    is contradicted by measurement AND by the spec this adapter is
    handed: a response-fit psr object served at all positions is the
    fit=response/serve=all corner of the masking 2x2 — a measured
    model-destroyer (65.7%/26.7% degenerate output against a 0.5%
    baseline) — and it shipped for every fitted psr_kl condition of the
    scale-12 served wave while the registries correctly recorded
    positions="response" + response_header_len the whole time. The
    worker registry now refuses at load when a response spec yields
    non-response ops (lens_steer), and test_petri_lens_adapters pins
    this adapter's routing. Do not reintroduce the hardcode."""
    _refuse_sidecar("psr", sidecar_bytes)
    blob = _torch_blob("psr", artifact_bytes)
    for key in ("layers", "directions", "gate_w", "gate_b"):
        if key not in blob:
            raise AdapterError(f"kind 'psr': artifact lacks {key!r} — not "
                               "a PSRObject.save blob")
    layers = _layer_list("psr", blob["layers"])
    if "layers" in resolved_spec:
        spec_layers = tuple(int(l) for l in resolved_spec["layers"])
        if spec_layers != layers:
            raise AdapterError(
                f"kind 'psr': resolved_spec layers {list(spec_layers)} != "
                f"artifact layers {list(layers)} (F9-style cross-check)")
    mode, header, turn_end = _fit_leg_positions(
        "psr", blob, resolved_spec, header_ids, turn_end_ids)
    ops = []
    d = None
    for l in layers:
        key = str(l)
        for group in ("directions", "gate_w", "gate_b"):
            if not isinstance(blob[group], Mapping) or key not in blob[group]:
                raise AdapterError(
                    f"kind 'psr': artifact {group!r} lacks layer {l} — "
                    "inconsistent with its own 'layers' list")
        direction = _f32_1d("psr", f"directions[{l}]", blob["directions"][key],
                            dim=d)
        d = direction.shape[0]
        gate_w = _f32_1d("psr", f"gate_w[{l}]", blob["gate_w"][key], dim=d)
        gate_b = blob["gate_b"][key]
        if not isinstance(gate_b, torch.Tensor) or gate_b.numel() != 1:
            raise AdapterError(
                f"kind 'psr': gate_b[{l}] must be a single-element tensor")
        if not torch.isfinite(gate_b).all():
            raise AdapterError(f"kind 'psr': gate_b[{l}] is non-finite")
        ops.append(HookOp(
            op=OP_RELU_GATED_VECTOR, layer=l, vector=direction,
            gate_w=gate_w, gate_b=gate_b.detach().reshape(1).to(torch.float32),
            positions=mode, skip_bos=False, phase=PHASE_BOTH,
            header_ids=header, turn_end_ids=turn_end))
    return tuple(ops)


#: ``extraction.paired_write.APPLICATION_ORDER`` spelled here (pinned equal by test; this
#: module imports no extraction code): the ONE composition the paired op serves
_PAIRED_APPLICATION_ORDER = "shared_pre_write_state"


def _adapt_psr_paired_write(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                            sidecar_bytes: bytes = b"", *,
                            header_ids: Optional[Tuple[int, ...]] = None,
                            turn_end_ids: Optional[Tuple[int, ...]] = None
                            ) -> Tuple[HookOp, ...]:
    """THE PAIRED WRITE (extraction/paired_write.PSRPairedWriteObject, kind
    ``psr_paired_write`` — the corpus PSR fit MINUS its register-matched control
    fit as the coherent virtual teacher): per stored layer l, ONE
    :data:`OP_RELU_GATED_VECTOR_PAIR` op — ``h += scale * (relu(h_fp32 . w_pos + b_pos)
    d_pos - relu(h_fp32 . w_neg + b_neg) d_neg)`` with BOTH gates on the SAME
    pre-steering hidden state, the difference cast back once. Prefill AND decode,
    every position, no BOS skip (the parents' gates are learned). The artifact is
    ``PSRPairedWriteObject.save`` / ``compose_blob``'s: ``layers``,
    ``application_order``, the ``pos`` and ``neg`` sides (each ``directions`` /
    ``gate_w`` / ``gate_b`` per layer), ``positions`` (always ``"all"``), a
    ``provenance`` block.

    Refuses: another ``application_order`` (the two sequential two-hook orders are
    DIFFERENT functions of the same parents — the class's own contract); a layer set
    that differs between the sides or from ``layers``; a side of another width; a
    masked position leg, fit or served (the class refuses masked parents, and a
    difference of two writes has no mask of its own — so no ``response`` override
    here, unlike psr); a spec/blob layer disagreement. The native residual slot
    serves the same artifact through the same-named op (``lora_export.
    _residual_psr_paired_write``), so a cell reads the same on either path."""
    kind = "psr_paired_write"
    _refuse_sidecar(kind, sidecar_bytes)
    blob = _torch_blob(kind, artifact_bytes)
    for key in ("layers", "pos", "neg", "application_order"):
        if key not in blob:
            raise AdapterError(f"kind {kind!r}: artifact lacks {key!r} — not a "
                               "PSRPairedWriteObject.save blob")
    order = blob["application_order"]
    if order != _PAIRED_APPLICATION_ORDER:
        raise AdapterError(
            f"kind {kind!r}: application_order={order!r}; the lens serves ONLY "
            f"{_PAIRED_APPLICATION_ORDER!r} (both gates read the shared pre-write state, one "
            "combined add) — a sequential two-hook composition is a different function of the "
            "same parents; refusing")
    layers = _layer_list(kind, blob["layers"])
    if "layers" in resolved_spec:
        spec_layers = tuple(int(l) for l in resolved_spec["layers"])
        if spec_layers != layers:
            raise AdapterError(
                f"kind {kind!r}: resolved_spec layers {list(spec_layers)} != "
                f"artifact layers {list(layers)} (F9-style cross-check)")
    mode, header, turn_end = _fit_leg_positions(kind, blob, resolved_spec, header_ids, turn_end_ids)
    if mode != POSITIONS_ALL or blob.get("positions", POSITIONS_ALL) != POSITIONS_ALL:
        raise AdapterError(
            f"kind {kind!r}: positions fit={blob.get('positions', 'all')!r} served={mode!r} — the "
            "paired write is defined on all-position parents only (PSRPairedWriteObject refuses "
            "a masked parent: a difference of two writes has no mask of its own) and is served at "
            "every position; refusing")
    sides: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    d: Optional[int] = None
    for tag in ("pos", "neg"):
        side = blob[tag]
        if not isinstance(side, Mapping):
            raise AdapterError(f"kind {kind!r}: artifact side {tag!r} is not a mapping")
        for group in ("directions", "gate_w", "gate_b"):
            if group not in side or not isinstance(side[group], Mapping):
                raise AdapterError(f"kind {kind!r}: side {tag!r} lacks the {group!r} mapping — not a "
                                   "PSRPairedWriteObject.save blob")
            extra = set(str(k) for k in side[group]) - {str(l) for l in layers}
            if extra:
                raise AdapterError(
                    f"kind {kind!r}: the {tag} side's {group!r} carries layers {sorted(extra)} the "
                    f"artifact does not declare ({list(layers)}) — the two sides' layer sets differ "
                    "from the declared set; refusing")
        per_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for l in layers:
            key = str(l)
            for group in ("directions", "gate_w", "gate_b"):
                if key not in side[group]:
                    raise AdapterError(
                        f"kind {kind!r}: the {tag} side lacks layer {l} in {group!r} — the two sides' "
                        "layer sets differ (the paired write is defined layerwise on ONE layer set); "
                        "refusing")
            direction = _f32_1d(kind, f"{tag}.directions[{l}]", side["directions"][key], dim=d)
            d = direction.shape[0]
            gate_w = _f32_1d(kind, f"{tag}.gate_w[{l}]", side["gate_w"][key], dim=d)
            gate_b = side["gate_b"][key]
            if not isinstance(gate_b, torch.Tensor) or gate_b.numel() != 1:
                raise AdapterError(f"kind {kind!r}: {tag}.gate_b[{l}] must be a single-element tensor")
            if not torch.isfinite(gate_b).all():
                raise AdapterError(f"kind {kind!r}: {tag}.gate_b[{l}] is non-finite")
            per_layer[l] = (direction, gate_w, gate_b.detach().reshape(1).to(torch.float32))
        sides[tag] = per_layer
    ops = []
    for l in layers:
        d_pos, w_pos, b_pos = sides["pos"][l]
        d_neg, w_neg, b_neg = sides["neg"][l]
        ops.append(HookOp(
            op=OP_RELU_GATED_VECTOR_PAIR, layer=l, vector=d_pos, gate_w=w_pos, gate_b=b_pos,
            vector2=d_neg, gate_w2=w_neg, gate_b2=b_neg,
            positions=POSITIONS_ALL, skip_bos=False, phase=PHASE_BOTH,
            header_ids=header, turn_end_ids=turn_end))
    return tuple(ops)


def _adapt_vector_all(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                      sidecar_bytes: bytes = b"", *,
                      header_ids: Optional[Tuple[int, ...]] = None,
                      turn_end_ids: Optional[Tuple[int, ...]] = None
                      ) -> Tuple[HookOp, ...]:
    """vector_all (extraction/multilayer.MultiLayerVectorObject): one
    additive residual vector per layer through hooks.steering — BOS
    excluded per the stored ``skip_bos`` (default True), both phases, one
    shared selector scale.

    Serves BOTH stored position modes: ``"all"``, and ``"response"`` —
    the cross-phase mask, whose served region is the assistant header
    plus every generated position (see THE RESPONSE MASK in the module
    docstring). Response-mode vector arms were HF-only until that mask
    landed."""
    _refuse_sidecar("vector_all", sidecar_bytes)
    blob = _torch_blob("vector_all", artifact_bytes)
    if "vectors" not in blob or not isinstance(blob["vectors"], Mapping) \
            or not blob["vectors"]:
        raise AdapterError("kind 'vector_all': artifact lacks a non-empty "
                           "'vectors' mapping — not a "
                           "MultiLayerVectorObject.save blob")
    positions = blob.get("positions", "all")
    skip_bos = blob.get("skip_bos", True)
    # the SPEC decides the served mask; the artifact's value is the
    # default. A cross-phase cell is this artifact (fit "all") served
    # "response", so the two legitimately disagree and must not be
    # cross-checked against each other.
    mode, header, turn_end = _served_positions(
        "vector_all", positions, header_ids, resolved_spec, turn_end_ids)
    if not isinstance(skip_bos, bool):
        raise AdapterError(f"kind 'vector_all': stored skip_bos must be a "
                           f"bool, got {skip_bos!r}")
    _spec_check("vector_all", resolved_spec, "skip_bos", skip_bos)
    layers = sorted(int(k) for k in blob["vectors"])  # load_blob's order
    ops = []
    d = None
    for l in layers:
        if l < 0:
            raise AdapterError(f"kind 'vector_all': layer {l} < 0")
        vec = _f32_1d("vector_all", f"vectors[{l}]", blob["vectors"][str(l)],
                      dim=d)
        d = vec.shape[0]
        ops.append(HookOp(op=OP_ADD_VECTOR, layer=l, vector=vec,
                          positions=mode, skip_bos=skip_bos,
                          phase=PHASE_BOTH, header_ids=header,
                          turn_end_ids=turn_end))
    return tuple(ops)


def _adapt_antidote(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                    sidecar_bytes: bytes = b"", *,
                    header_ids: Optional[Tuple[int, ...]] = None,
                    turn_end_ids: Optional[Tuple[int, ...]] = None
                    ) -> Tuple[HookOp, ...]:
    """antidote (extraction/antidote.AntidoteObject): the masked unit-PC
    steered at positions="last" (the final real prompt token during the
    prefill, every decode forward's single position), skip_bos off, with
    the ABSOLUTE calibrated ``base_alpha`` baked as the op coeff — the
    selector scale multiplies it, exactly as ``AntidoteObject.apply``
    passes ``base_alpha * scale`` to hooks.steering. Artifact = the
    SteeringVector safetensors container; the JSON sidecar (method +
    metadata.base_alpha) is REQUIRED — it is what load_object dispatches
    on, and an uncalibrated object cannot be applied."""
    if not sidecar_bytes:
        raise AdapterError(
            "kind 'antidote': the SteeringVector JSON sidecar is REQUIRED "
            "(it carries method + metadata.base_alpha — the loader's "
            "dispatch and the calibration); a sidecar-less antidote entry "
            "is unservable")
    if not artifact_bytes:
        raise AdapterError("kind 'antidote': empty artifact bytes")
    try:
        from safetensors.torch import load as _st_load
        tensors = _st_load(bytes(artifact_bytes))
    except Exception as e:
        raise AdapterError(
            f"kind 'antidote': artifact bytes did not load as a "
            f"safetensors container: {e}") from e
    try:
        side = json.loads(bytes(sidecar_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AdapterError(
            f"kind 'antidote': unparseable JSON sidecar: {e}") from e
    if not isinstance(side, dict):
        raise AdapterError("kind 'antidote': sidecar is not a JSON object")
    if side.get("method") != "antidote":
        raise AdapterError(
            f"kind 'antidote': sidecar method {side.get('method')!r} != "
            "'antidote' — load_object would hydrate a different "
            "intervention; refusing (mirrors AntidoteObject's own check)")
    meta = side.get("metadata")
    if not isinstance(meta, Mapping) or "base_alpha" not in meta:
        raise AdapterError(
            "kind 'antidote': no metadata.base_alpha in the sidecar — an "
            "uncalibrated object cannot be applied (the factor grid "
            "multiplies base_alpha; retrain the cell — mirrors "
            "AntidoteObject's own refusal)")
    base_alpha = meta["base_alpha"]
    if not isinstance(base_alpha, (int, float)) or isinstance(base_alpha, bool) \
            or not math.isfinite(float(base_alpha)):
        raise AdapterError(
            f"kind 'antidote': metadata.base_alpha must be a finite "
            f"number, got {base_alpha!r}")
    _spec_check("antidote", resolved_spec, "method", "antidote")
    layers = {}
    for name, t in tensors.items():
        if not name.startswith("layer_"):
            raise AdapterError(
                f"kind 'antidote': unexpected safetensors entry {name!r} — "
                "not a SteeringVector container")
        try:
            l = int(name.removeprefix("layer_"))
        except ValueError as e:
            raise AdapterError(
                f"kind 'antidote': unparseable layer key {name!r}") from e
        layers[l] = t
    if not layers:
        raise AdapterError("kind 'antidote': container holds no layer "
                           "vectors")
    if "layers" in side and sorted(int(l) for l in side["layers"]) != \
            sorted(layers):
        raise AdapterError(
            f"kind 'antidote': sidecar layers {side['layers']} != container "
            f"layers {sorted(layers)} — inconsistent artifact")
    ops = []
    d = None
    for l in sorted(layers):
        if l < 0:
            raise AdapterError(f"kind 'antidote': layer {l} < 0")
        vec = _f32_1d("antidote", f"layer_{l}", layers[l], dim=d)
        d = vec.shape[0]
        ops.append(HookOp(op=OP_ADD_VECTOR, layer=l, vector=vec,
                          coeff=float(base_alpha), positions=POSITIONS_LAST,
                          skip_bos=False, phase=PHASE_BOTH))
    return tuple(ops)


def _adapt_alphasteer(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                      sidecar_bytes: bytes = b"", *,
                      header_ids: Optional[Tuple[int, ...]] = None,
                      turn_end_ids: Optional[Tuple[int, ...]] = None
                      ) -> Tuple[HookOp, ...]:
    """alphasteer (extraction/alphasteer.AlphaSteerObject): during the
    PREFILL only, ``h += base_scale · scale · (h_last @ M)`` broadcast to
    every prompt position (``h_last`` = the last prompt token's hidden
    state; the matmul runs in the model dtype, the class's own
    arithmetic); decode steps are untouched — influenced only through the
    steered prompt KV (the paper's prefill-only deployment).

    MULTI-LAYER, and the plural fields win. A released object stores its
    matrices in ``Ms``/``layers`` and leaves the LEGACY singular
    ``layer``/``M``/``refusal`` present but NULL — so a reader that takes
    the singular fields gets ``None`` rather than a missing key, and a
    presence check passes right before the value fails. Same trap shape
    as ``static_r``'s kind string; read the plural.

    Released objects are fit at resid_pre, so the served layer is one
    block EARLIER (:func:`_block_output_layer`). The fit layer stays the
    lookup key into ``Ms`` — only the SERVING LOCATION shifts. Getting
    that backwards finds a valid matrix at most shifted indices (the fit
    and shifted layer sets overlap heavily) and steers one block off,
    which produces fluent text and a wrong intervention."""
    if resolved_spec.get("positions") == "response":
        raise AdapterError(
            "kind 'alphasteer': refusing to serve this arm under the response "
            "mask. It fails DERIVATION LOCALITY (module docstring): its "
            "edit is computed ONCE from the LAST PROMPT TOKEN and then "
            "applied elsewhere, so restricting the applied region to "
            "header-plus-generated does not give the same function on "
            "fewer positions — it applies a prompt-derived vector to "
            "generated positions, which is a different intervention "
            "nobody has defined. A cross-phase variant of this arm "
            "should be built deliberately with its own semantics (the "
            "class already has a 'prompt_hold' mode that reuses the "
            "prefill vector during decode, which is probably the thing "
            "meant), never fall out of a mask override")
    _refuse_sidecar("alphasteer", sidecar_bytes)
    blob = _torch_blob("alphasteer", artifact_bytes)
    if "base_scale" not in blob:
        raise AdapterError("kind 'alphasteer': artifact lacks 'base_scale' "
                           "— not an AlphaSteerObject.save blob")
    base_scale = _finite_float("alphasteer", "base_scale", blob["base_scale"])
    site = blob.get("site") or "post"
    steer_positions = blob.get("steer_positions") or "prefill"
    if steer_positions != "prefill":
        raise AdapterError(
            f"kind 'alphasteer': stored steer_positions "
            f"{steer_positions!r} is not the prefill-only broadcast this op "
            "expresses — the mask is the intervention; refusing to guess it")
    _spec_check("alphasteer", resolved_spec, "site", site)
    _spec_check("alphasteer", resolved_spec, "steer_positions",
                steer_positions)

    Ms = blob.get("Ms")
    if isinstance(Ms, Mapping) and Ms:
        layers = _layer_list("alphasteer", sorted(int(k) for k in Ms))
        if blob.get("layers"):
            declared = tuple(sorted(int(l) for l in blob["layers"]))
            if declared != layers:
                raise AdapterError(
                    f"kind 'alphasteer': blob declares layers "
                    f"{list(declared)} but 'Ms' carries {list(layers)} — "
                    "inconsistent artifact; refusing")
        mats = {l: _layer_entry("alphasteer", "Ms", Ms, l) for l in layers}
    else:  # legacy single-layer blob
        if blob.get("layer") is None or blob.get("M") is None:
            raise AdapterError(
                "kind 'alphasteer': artifact has neither a non-empty 'Ms' "
                "mapping nor a legacy 'layer'/'M' pair — refusing an "
                "artifact with no matrix to serve")
        layers = (int(blob["layer"]),)
        mats = {layers[0]: blob["M"]}

    ops = []
    d = None
    for l in layers:
        M = mats[l]
        if not isinstance(M, torch.Tensor) or M.dim() != 2 \
                or M.shape[0] != M.shape[1]:
            raise AdapterError(
                f"kind 'alphasteer': Ms[{l}] must be a square [d, d] "
                "tensor, got "
                + (f"shape {tuple(M.shape)}" if isinstance(M, torch.Tensor)
                   else type(M).__name__))
        if d is None:
            d = int(M.shape[0])
        M = _f32_2d("alphasteer", f"Ms[{l}]", M, (d, d))
        ops.append(HookOp(
            op=OP_PREFILL_LAST_MATMUL,
            layer=_block_output_layer("alphasteer", l, site),
            matrix=M, coeff=base_scale, phase=PHASE_PREFILL))
    return tuple(ops)


def _materialize_orthogonal_R(kind: str, layer: int, rank: int,
                              d: int, state_dict: Mapping[str, Any]
                              ) -> torch.Tensor:
    """The materialized orthonormal subspace matrix R [r, d] from a saved
    LoReFT ``rotate`` state_dict — rebuilt with EXACTLY the construction
    ``LoReFTObject.__init__``/``load_blob`` use (``nn.Linear(d, r,
    bias=False)`` under torch's orthogonal parametrization), so the
    materialized weight is bit-identical to the object's own
    ``rot.weight``. The module construction draws torch RNG for its
    (immediately overwritten) init, so it runs under a forked RNG — the
    adapter stays side-effect free."""
    from torch import nn
    from torch.nn.utils import parametrizations

    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)  # overwritten by load_state_dict below
            rot = nn.Linear(d, rank, bias=False)
            parametrizations.orthogonal(rot)
            rot.load_state_dict(dict(state_dict), strict=True)
            with torch.no_grad():
                R = rot.weight.detach().clone()
    except AdapterError:
        raise
    except Exception as e:
        raise AdapterError(
            f"kind {kind!r}: rotate state_dict for layer {layer} did not "
            f"load into the orthogonal-parametrized Linear(d={d}, "
            f"rank={rank}): {e}") from e
    if R.shape != (rank, d):
        raise AdapterError(
            f"kind {kind!r}: materialized R has shape {tuple(R.shape)}, "
            f"expected ({rank}, {d})")
    if not torch.isfinite(R).all():
        raise AdapterError(f"kind {kind!r}: materialized R is non-finite")
    return R.to(torch.float32)


def _adapt_loreft(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                  sidecar_bytes: bytes = b"", *,
                  header_ids: Optional[Tuple[int, ...]] = None,
                  turn_end_ids: Optional[Tuple[int, ...]] = None
                  ) -> Tuple[HookOp, ...]:
    """loreft (extraction/reft.LoReFTObject): the low-rank affine
    subspace edit ``h += scale · ((h_fp32 (W−R)ᵀ + b) R)`` in fp32 at the
    stored layer(s), PROMPT POSITIONS ONLY — the class gates on
    prefill-vs-decode by seq_len (a decode step passes through
    untouched), which the phase flag expresses exactly. R is materialized
    from the saved orthogonal parametrization state via the class's own
    construction."""
    _refuse_sidecar("loreft", sidecar_bytes)
    blob = _torch_blob("loreft", artifact_bytes)
    for key in ("layers", "rank", "rotate", "learned"):
        if key not in blob:
            raise AdapterError(f"kind 'loreft': artifact lacks {key!r} — "
                               "not a LoReFTObject.save blob")
    layers = _layer_list("loreft", blob["layers"])
    rank = int(blob["rank"])
    if rank <= 0:
        raise AdapterError(f"kind 'loreft': rank must be > 0, got {rank}")
    fit_positions = blob.get("positions", "prompt")
    if fit_positions == "response" \
            or resolved_spec.get("positions") == "response":
        raise AdapterError(
            "kind 'loreft': the artifact records a response-mask fit leg "
            "(or the cell requests a response serve), but this translation "
            "is the class's prompt-positions deployment (prefill-only) — "
            "serving it would silently DROP the recorded mask and run the "
            "prompt-only method under a decode-only label, the psr bug's "
            "shape on another kind. A response-mode loreft needs its own "
            "deliberate translation (loreft_both is the both-phases "
            "arithmetic); refusing")
    _spec_check("loreft", resolved_spec, "positions", fit_positions)
    if "layers" in resolved_spec:
        spec_layers = tuple(int(l) for l in resolved_spec["layers"])
        if spec_layers != layers:
            raise AdapterError(
                f"kind 'loreft': resolved_spec layers {list(spec_layers)} "
                f"!= artifact layers {list(layers)} (F9-style cross-check)")
    _spec_check("loreft", resolved_spec, "rank", rank)
    ops = []
    d = None
    for l in layers:
        key = str(l)
        for group in ("rotate", "learned"):
            if not isinstance(blob[group], Mapping) or key not in blob[group]:
                raise AdapterError(
                    f"kind 'loreft': artifact {group!r} lacks layer {l} — "
                    "inconsistent with its own 'layers' list")
        lin_sd = blob["learned"][key]
        if not isinstance(lin_sd, Mapping) or "weight" not in lin_sd \
                or "bias" not in lin_sd:
            raise AdapterError(
                f"kind 'loreft': learned[{l}] is not a Linear state_dict "
                "with weight + bias")
        W = lin_sd["weight"]
        b = lin_sd["bias"]
        if not isinstance(W, torch.Tensor) or W.dim() != 2 \
                or W.shape[0] != rank:
            raise AdapterError(
                f"kind 'loreft': learned[{l}].weight must be [rank={rank},"
                f" d], got "
                + (f"shape {tuple(W.shape)}" if isinstance(W, torch.Tensor)
                   else type(W).__name__))
        if d is None:
            d = W.shape[1]
        elif W.shape[1] != d:
            raise AdapterError(
                f"kind 'loreft': learned[{l}].weight dim {W.shape[1]} != "
                f"{d} of earlier layers")
        b = _f32_1d("loreft", f"learned[{l}].bias", b, dim=rank)
        if not torch.isfinite(W).all():
            raise AdapterError(f"kind 'loreft': learned[{l}].weight is "
                               "non-finite")
        R = _materialize_orthogonal_R("loreft", l, rank, d,
                                      blob["rotate"][key])
        ops.append(HookOp(op=OP_AFFINE_SUBSPACE, layer=l, rotate=R,
                          affine_w=W.detach().to(torch.float32), affine_b=b,
                          phase=PHASE_PREFILL))
    return tuple(ops)


def _layer_entry(kind: str, group_name: str, group: Any, l: int) -> Any:
    """One layer's tensor out of a ``{layer: tensor}`` blob group. The
    fit lane's ``torch.save`` keeps int keys while some loaders write str
    ones, so both are accepted — and a layer the group is missing is a
    refusal, never a skipped layer."""
    if isinstance(group, Mapping):
        for key in (l, str(l)):
            if key in group:
                return group[key]
    raise AdapterError(
        f"kind {kind!r}: artifact {group_name!r} lacks layer {l} — "
        "inconsistent with its own 'layers' list")


def _finite_float(kind: str, name: str, v: Any) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) \
            or not math.isfinite(float(v)):
        raise AdapterError(f"kind {kind!r}: {name} must be a finite number, "
                           f"got {v!r}")
    return float(v)


def _f32_2d(kind: str, name: str, t: Any,
            shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise AdapterError(
            f"kind {kind!r}: {name} must be a 2-D tensor, got "
            f"{type(t).__name__}"
            + (f" of shape {tuple(t.shape)}" if isinstance(t, torch.Tensor)
               else ""))
    if shape is not None and tuple(t.shape) != shape:
        raise AdapterError(f"kind {kind!r}: {name} has shape "
                           f"{tuple(t.shape)}, expected {shape}")
    if not torch.isfinite(t).all():
        raise AdapterError(f"kind {kind!r}: {name} carries non-finite values "
                           "— refusing to serve a corrupt tensor")
    return t.detach().to(torch.float32)


def _adapt_alphakl(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                   sidecar_bytes: bytes = b"", *,
                   header_ids: Optional[Tuple[int, ...]] = None,
                   turn_end_ids: Optional[Tuple[int, ...]] = None
                   ) -> Tuple[HookOp, ...]:
    """alphakl (extraction/alphakl.AlphaKLObject): per stored layer l,
    ``h += base_scale · scale · gate(h_fp32 · w_l) · r_l``, the gate
    computed in fp32 on the pre-steering block output and the whole
    addition cast back once. Every position, prefill AND decode, no BOS
    skip — unlike the prefill-only baselines, this arm steers generated
    tokens too.

    ``clamp_gate`` selects the op: True (the default on load) is a ReLU
    with ZERO bias; False is the paper-parity RAW LINEAR gate, which
    deliberately keeps the negative-projection half. Serving one as the
    other would silently change the intervention, so the flag is READ,
    never assumed.

    ``w`` is NOT unit-norm — the gate's magnitude lives in it, so it is
    baked verbatim and never folded into the scale or renormalized."""
    _refuse_sidecar("alphakl", sidecar_bytes)
    blob = _torch_blob("alphakl", artifact_bytes)
    for key in ("layers", "ws", "rs", "base_scale"):
        if key not in blob:
            raise AdapterError(f"kind 'alphakl': artifact lacks {key!r} — "
                               "not an AlphaKLObject.save blob")
    layers = _layer_list("alphakl", blob["layers"])
    if "layers" in resolved_spec:
        spec_layers = tuple(int(l) for l in resolved_spec["layers"])
        if spec_layers != layers:
            raise AdapterError(
                f"kind 'alphakl': resolved_spec layers {list(spec_layers)} != "
                f"artifact layers {list(layers)} (F9-style cross-check)")
    base_scale = _finite_float("alphakl", "base_scale", blob["base_scale"])
    clamp_gate = blob.get("clamp_gate", True)  # AlphaKLObject.load's default
    if not isinstance(clamp_gate, bool):
        raise AdapterError(f"kind 'alphakl': clamp_gate must be a bool, got "
                           f"{clamp_gate!r}")
    _spec_check("alphakl", resolved_spec, "clamp_gate", clamp_gate)
    _spec_check("alphakl", resolved_spec, "base_scale", base_scale)
    op = OP_RELU_GATED_VECTOR if clamp_gate else OP_LINEAR_GATED_VECTOR
    ops = []
    d = None
    for l in layers:
        w = _f32_1d("alphakl", f"ws[{l}]",
                    _layer_entry("alphakl", "ws", blob["ws"], l), dim=d)
        d = w.shape[0]
        r = _f32_1d("alphakl", f"rs[{l}]",
                    _layer_entry("alphakl", "rs", blob["rs"], l), dim=d)
        ops.append(HookOp(
            op=op, layer=l, vector=r, gate_w=w,
            # AlphaKL's gate has no bias term at all; the zero keeps one
            # shared op body with PSR rather than a near-duplicate.
            gate_b=torch.zeros(1, dtype=torch.float32),
            coeff=base_scale, positions=POSITIONS_ALL, skip_bos=False,
            phase=PHASE_BOTH))
    return tuple(ops)


def _adapt_static_r(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                    sidecar_bytes: bytes = b"", *,
                    header_ids: Optional[Tuple[int, ...]] = None,
                    turn_end_ids: Optional[Tuple[int, ...]] = None
                    ) -> Tuple[HookOp, ...]:
    """static_r (extraction/baseline_arms.StaticRObject), the CAA floor —
    one stored direction added during the PREFILL only; decode steps are
    never edited and inherit the effect through the KV cache.

    Two deployment shapes share the kind, and they differ in both mask
    and SITE, so neither may be inferred from the other:

    - ``prompt_last`` at site ``post`` (arm static_r): the last prompt
      token of the block OUTPUT, one layer.
    - ``prefill_all`` at site ``pre`` (arm static_r_matched): every
      prompt position at the block INPUT, ten layers. Serving a
      resid_pre edit at a block output means a deliberate one-block
      index shift (:func:`_block_output_layer`).

    Legacy single-layer blobs carry ``layer``/``r`` with no
    ``steer_positions`` or ``site``; the class's own load defaults
    (``prompt_last``/``post``) are mirrored here."""
    _refuse_sidecar("static_r", sidecar_bytes)
    blob = _torch_blob("static_r", artifact_bytes)
    if "base_scale" not in blob:
        raise AdapterError("kind 'static_r': artifact lacks 'base_scale' — "
                           "not a StaticRObject.save blob")
    base_scale = _finite_float("static_r", "base_scale", blob["base_scale"])
    steer_positions = blob.get("steer_positions") or "prompt_last"
    site = blob.get("site") or "post"
    if steer_positions not in ("prompt_last", "prefill_all"):
        raise AdapterError(
            f"kind 'static_r': unknown steer_positions "
            f"{steer_positions!r} (expected 'prompt_last' or "
            "'prefill_all') — the mask is the intervention; refusing to "
            "guess it")
    # 'steer_positions' is the artifact's own FIT mode and is still
    # cross-checked: the serve-side override rides on 'positions', a
    # different key, and only that one may legitimately disagree.
    _spec_check("static_r", resolved_spec, "steer_positions", steer_positions)
    _spec_check("static_r", resolved_spec, "site", site)
    # THE CROSS-PHASE CELL: this artifact, fit at all prompt positions,
    # served under the response mask. Legitimate here because the edit
    # is a FIXED vector that reads nothing from the activation, so
    # re-masking it is the same function on a different position set
    # (derivation locality — module docstring). Serving it response-mode
    # also flips the phase: the region spans the header at prefill AND
    # every generated position, where the object's own deployment is
    # prefill-only.
    served_response = resolved_spec.get("positions") == "response"

    # DISPATCH ON THE FIELDS, NOT THE KIND. Both deployment shapes save
    # kind='static_r' — the multi-layer resid_pre arm included — so a
    # kind-keyed branch would route it to the single-layer path and read
    # its legacy layer=None/r=None as a one-layer, one-position edit:
    # either a crash or, worse, a plausible wrong intervention. The
    # shapes are told apart by whether 'rs' carries anything.
    rs = blob.get("rs")
    if isinstance(rs, Mapping) and rs:
        layers = _layer_list("static_r", sorted(int(k) for k in rs))
        if blob.get("layers"):
            declared = tuple(sorted(int(l) for l in blob["layers"]))
            if declared != layers:
                raise AdapterError(
                    f"kind 'static_r': blob declares layers "
                    f"{list(declared)} but 'rs' carries {list(layers)} — "
                    "inconsistent artifact; refusing rather than serving "
                    "whichever one happens to be read first")
        vectors = {l: _layer_entry("static_r", "rs", rs, l) for l in layers}
    else:  # legacy single-layer blob
        if blob.get("layer") is None or blob.get("r") is None:
            raise AdapterError(
                "kind 'static_r': artifact has neither a non-empty 'rs' "
                "mapping nor a legacy 'layer'/'r' pair — refusing an "
                "artifact with no direction to serve")
        layers = (int(blob["layer"]),)
        vectors = {layers[0]: blob["r"]}

    if served_response:
        positions, header, turn_end = _served_positions(
            "static_r", "response", header_ids, resolved_spec,
            turn_end_ids)
        phase = PHASE_BOTH
    else:
        positions = (POSITIONS_LAST if steer_positions == "prompt_last"
                     else POSITIONS_ALL)
        header = None
        turn_end = None
        # prefill ONLY: the class returns early on any seq_len<=1
        # forward, so a decode step must not be invoked at all.
        phase = PHASE_PREFILL
    recorded_norms = _recorded_layer_norms("static_r", blob,
                                           "per_layer_r_norm")
    _assert_norms_discriminate("static_r", recorded_norms)
    ops = []
    d = None
    for l in layers:
        vec = _f32_1d("static_r", f"rs[{l}]", vectors[l], dim=d)
        d = vec.shape[0]
        _check_recorded_norm("static_r", l, vec, recorded_norms)
        ops.append(HookOp(
            op=OP_ADD_VECTOR, layer=_block_output_layer("static_r", l, site),
            vector=vec, coeff=base_scale, positions=positions,
            skip_bos=False, phase=phase, header_ids=header,
            turn_end_ids=turn_end))
    return tuple(ops)


#: the deployment ``CaaStaticRObject.apply`` implements, which its ``save``
#: does NOT record: every prompt position (``prefill_all``) at the block INPUT
#: (``pre`` — resid_pre, the paper's edit site). The class has exactly one
#: mode, so these are its identity rather than defaults.
_CAA_STATIC_R_POSITIONS = "prefill_all"
_CAA_STATIC_R_SITE = "pre"


def _adapt_caa_static_r(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                        sidecar_bytes: bytes = b"", *,
                        header_ids: Optional[Tuple[int, ...]] = None,
                        turn_end_ids: Optional[Tuple[int, ...]] = None
                        ) -> Tuple[HookOp, ...]:
    """caa_static_r (extraction/caa_static_r.CaaStaticRObject), the Class B
    CAA floor DEPLOYMENT-MATCHED to AlphaSteer (E45 H2).

    ``apply()`` registers a forward PRE-hook on each steered decoder layer
    (resid_pre — the block input, before input_layernorm) that, DURING PREFILL
    ONLY, adds ``base_scale * r_l`` to EVERY prompt position; decode steps see
    the edit only through the steered prompt KV. Served here as
    :data:`OP_ADD_VECTOR` per stored layer at :data:`POSITIONS_ALL`,
    :data:`PHASE_PREFILL`, with the one-block index shift
    (:func:`_block_output_layer`) a resid_pre edit needs at a block-output
    hook site — the same translation ``static_r``'s ``prefill_all`` / ``pre``
    branch performs.

    IT IS ITS OWN KIND ON PURPOSE. ``CaaStaticRObject.save`` writes
    ``{kind, rs, base_scale, metadata}`` and records NEITHER ``steer_positions``
    NOR ``site``; ``static_r``'s adapter defaults an absent pair to
    ``prompt_last`` / ``post`` (its legacy single-layer blob's shape). Routing
    these bytes there would serve a one-layer, last-prompt-token, block-OUTPUT
    edit — a different intervention with entirely plausible numbers — so the
    mode is PINNED here instead, and a blob that carries a contradicting mode
    is refused rather than followed.

    ``turn_end_ids`` is accepted and UNUSED, exactly as in the
    deployment-matched twin :func:`_adapt_alphasteer`: every op this adapter
    emits is :data:`POSITIONS_ALL` in :data:`PHASE_PREFILL` with
    ``header_ids=None``, so it can never wear a response mask and has no
    assistant-turn span to bound. The parameter is in the signature because
    :func:`register_adapter`'s guard passes it to EVERY adapter."""
    _refuse_sidecar("caa_static_r", sidecar_bytes)
    blob = _torch_blob("caa_static_r", artifact_bytes)
    if "base_scale" not in blob:
        raise AdapterError("kind 'caa_static_r': artifact lacks 'base_scale' — "
                           "not a CaaStaticRObject.save blob")
    base_scale = _finite_float("caa_static_r", "base_scale", blob["base_scale"])
    for key, pinned in (("steer_positions", _CAA_STATIC_R_POSITIONS),
                        ("site", _CAA_STATIC_R_SITE)):
        stored = blob.get(key)
        if stored is not None and stored != pinned:
            raise AdapterError(
                f"kind 'caa_static_r': artifact records {key}={stored!r} but the class has "
                f"one deployment, {pinned!r} (extraction/caa_static_r.CaaStaticRObject.apply) "
                "— refusing rather than serving a mode this translation does not express")
        _spec_check("caa_static_r", resolved_spec, key, pinned)
    rs = blob.get("rs")
    if not isinstance(rs, Mapping) or not rs:
        raise AdapterError(
            "kind 'caa_static_r': artifact has no non-empty 'rs' mapping — refusing an "
            "artifact with no direction to serve")
    layers = _layer_list("caa_static_r", sorted(int(k) for k in rs))
    recorded_norms = _recorded_layer_norms("caa_static_r", blob, "per_layer_r_norm")
    _assert_norms_discriminate("caa_static_r", recorded_norms)
    ops = []
    d = None
    for l in layers:
        vec = _f32_1d("caa_static_r", f"rs[{l}]", _layer_entry("caa_static_r", "rs", rs, l),
                      dim=d)
        d = vec.shape[0]
        _check_recorded_norm("caa_static_r", l, vec, recorded_norms)
        ops.append(HookOp(
            op=OP_ADD_VECTOR,
            layer=_block_output_layer("caa_static_r", l, _CAA_STATIC_R_SITE),
            vector=vec, coeff=base_scale, positions=POSITIONS_ALL,
            skip_bos=False, phase=PHASE_PREFILL, header_ids=None))
    return tuple(ops)


def _adapt_cosine_gated_r(resolved_spec: Mapping[str, Any],
                          artifact_bytes: bytes, sidecar_bytes: bytes = b"",
                          *, header_ids: Optional[Tuple[int, ...]] = None,
                          turn_end_ids: Optional[Tuple[int, ...]] = None
                          ) -> Tuple[HookOp, ...]:
    """cosine_gated_r (extraction/baseline_arms.CosineGatedRObject): a
    conditional CAA floor. On the prefill, ``cos(h_last, d_cond)`` is
    compared against the calibrated ``theta`` — STRICTLY greater fires —
    and only a firing request gets ``base_scale · scale · r`` on its last
    prompt token. A non-firing request is bit-identical to unsteered.

    The gate is one decision PER REQUEST rather than a per-token
    coefficient, which is why it has its own op rather than reusing the
    gated-vector body."""
    if resolved_spec.get("positions") == "response":
        raise AdapterError(
            "kind 'cosine_gated_r': refusing to serve this arm under the response "
            "mask. It fails DERIVATION LOCALITY (module docstring): its "
            "edit is computed ONCE from the LAST PROMPT TOKEN and then "
            "applied elsewhere, so restricting the applied region to "
            "header-plus-generated does not give the same function on "
            "fewer positions — it applies a prompt-derived vector to "
            "generated positions, which is a different intervention "
            "nobody has defined. A cross-phase variant of this arm "
            "should be built deliberately with its own semantics (the "
            "class already has a 'prompt_hold' mode that reuses the "
            "prefill vector during decode, which is probably the thing "
            "meant), never fall out of a mask override")
    _refuse_sidecar("cosine_gated_r", sidecar_bytes)
    blob = _torch_blob("cosine_gated_r", artifact_bytes)
    for key in ("layer", "r", "theta", "d_cond", "base_scale"):
        if key not in blob:
            raise AdapterError(
                f"kind 'cosine_gated_r': artifact lacks {key!r} — not a "
                "CosineGatedRObject.save blob")
    layer = int(blob["layer"])
    if layer < 0:
        raise AdapterError(f"kind 'cosine_gated_r': layer {layer} < 0")
    r = _f32_1d("cosine_gated_r", "r", blob["r"])
    cond = _f32_1d("cosine_gated_r", "d_cond", blob["d_cond"],
                   dim=r.shape[0])
    if float(cond.norm()) == 0.0:
        raise AdapterError(
            "kind 'cosine_gated_r': d_cond has zero norm — every cosine "
            "would collapse to 0 and the gate would fire on the sign of "
            "theta alone; refusing")
    theta = _finite_float("cosine_gated_r", "theta", blob["theta"])
    base_scale = _finite_float("cosine_gated_r", "base_scale",
                               blob["base_scale"])
    _spec_check("cosine_gated_r", resolved_spec, "layer", layer)
    _spec_check("cosine_gated_r", resolved_spec, "theta", theta)
    return (HookOp(op=OP_PREFILL_LAST_COSINE_GATED, layer=layer, vector=r,
                   cond=cond, threshold=theta, coeff=base_scale,
                   positions=POSITIONS_LAST, phase=PHASE_PREFILL),)


def _adapt_lowrank_gated(resolved_spec: Mapping[str, Any],
                         artifact_bytes: bytes, sidecar_bytes: bytes = b"", *,
                         header_ids: Optional[Tuple[int, ...]] = None,
                         turn_end_ids: Optional[Tuple[int, ...]] = None
                         ) -> Tuple[HookOp, ...]:
    """lowrank_gated (the earlier development branch's low-rank gated
    object): the rank-k generalization of the gated vector,
    ``h += base_scale · scale · gate(h_fp32 Aᵀ) Bᵀ`` at every position,
    prefill AND decode.

    Despite the name this is a RESIDUAL-STREAM object, not the wave's
    ``lora`` kind — that one edits attention o_proj mid-block and is
    unservable here. ``B`` is a fixed write basis; only ``A`` is fit."""
    _refuse_sidecar("lowrank_gated", sidecar_bytes)
    blob = _torch_blob("lowrank_gated", artifact_bytes)
    for key in ("layers", "rank", "As", "Bs", "base_scale"):
        if key not in blob:
            raise AdapterError(f"kind 'lowrank_gated': artifact lacks {key!r} "
                               "— not a low-rank gated object's save blob")
    layers = _layer_list("lowrank_gated", blob["layers"])
    rank = int(blob["rank"])
    if rank <= 0:
        raise AdapterError(f"kind 'lowrank_gated': rank must be > 0, got "
                           f"{rank}")
    _spec_check("lowrank_gated", resolved_spec, "rank", rank)
    base_scale = _finite_float("lowrank_gated", "base_scale",
                               blob["base_scale"])
    clamp_gate = blob.get("clamp_gate", True)
    if not isinstance(clamp_gate, bool):
        raise AdapterError(f"kind 'lowrank_gated': clamp_gate must be a bool, "
                           f"got {clamp_gate!r}")
    op = OP_LOWRANK_RELU_GATED if clamp_gate else OP_LOWRANK_LINEAR_GATED
    ops = []
    d = None
    for l in layers:
        a = _layer_entry("lowrank_gated", "As", blob["As"], l)
        if not isinstance(a, torch.Tensor) or a.dim() != 2 \
                or a.shape[0] != rank:
            raise AdapterError(
                f"kind 'lowrank_gated': As[{l}] must be [rank={rank}, d], got "
                + (f"shape {tuple(a.shape)}" if isinstance(a, torch.Tensor)
                   else type(a).__name__))
        if d is None:
            d = int(a.shape[1])
        a = _f32_2d("lowrank_gated", f"As[{l}]", a, (rank, d))
        b = _f32_2d("lowrank_gated", f"Bs[{l}]",
                    _layer_entry("lowrank_gated", "Bs", blob["Bs"], l),
                    (d, rank))
        ops.append(HookOp(op=op, layer=l, lowrank_a=a, lowrank_b=b,
                          coeff=base_scale, positions=POSITIONS_ALL,
                          skip_bos=False, phase=PHASE_BOTH))
    return tuple(ops)


def _adapt_loreft_both(resolved_spec: Mapping[str, Any],
                       artifact_bytes: bytes, sidecar_bytes: bytes = b"",
                       *, header_ids: Optional[Tuple[int, ...]] = None,
                       turn_end_ids: Optional[Tuple[int, ...]] = None
                       ) -> Tuple[HookOp, ...]:
    """loreft_both (the earlier development branch's both-phase LoReFT
    object): the LoReFT affine subspace edit ``h += base_scale · scale ·
    ((h_fp32 (W−R)ᵀ + b) R)``, applied at EVERY position in BOTH phases.

    Same arithmetic as the ``loreft`` kind, deliberately different phase:
    that class gates on prefill-vs-decode by seq_len, while this one's
    benign penalty covers generation positions so the edit must apply
    there too. R is stored materialized here (no orthogonal
    parametrization to rebuild) and is a FIXED basis."""
    _refuse_sidecar("loreft_both", sidecar_bytes)
    blob = _torch_blob("loreft_both", artifact_bytes)
    for key in ("layers", "rank", "Rs", "Ws", "bs", "base_scale"):
        if key not in blob:
            raise AdapterError(f"kind 'loreft_both': artifact lacks "
                               f"{key!r} — not a both-phase LoReFT object's "
                               "save blob")
    layers = _layer_list("loreft_both", blob["layers"])
    rank = int(blob["rank"])
    if rank <= 0:
        raise AdapterError(f"kind 'loreft_both': rank must be > 0, got "
                           f"{rank}")
    _spec_check("loreft_both", resolved_spec, "rank", rank)
    base_scale = _finite_float("loreft_both", "base_scale",
                               blob["base_scale"])
    ops = []
    d = None
    for l in layers:
        R = _layer_entry("loreft_both", "Rs", blob["Rs"], l)
        if not isinstance(R, torch.Tensor) or R.dim() != 2 \
                or R.shape[0] != rank:
            raise AdapterError(
                f"kind 'loreft_both': Rs[{l}] must be [rank={rank}, d], "
                "got " + (f"shape {tuple(R.shape)}"
                          if isinstance(R, torch.Tensor)
                          else type(R).__name__))
        if d is None:
            d = int(R.shape[1])
        R = _f32_2d("loreft_both", f"Rs[{l}]", R, (rank, d))
        W = _f32_2d("loreft_both", f"Ws[{l}]",
                    _layer_entry("loreft_both", "Ws", blob["Ws"], l),
                    (rank, d))
        b = _f32_1d("loreft_both", f"bs[{l}]",
                    _layer_entry("loreft_both", "bs", blob["bs"], l),
                    dim=rank)
        ops.append(HookOp(op=OP_AFFINE_SUBSPACE, layer=l, rotate=R,
                          affine_w=W, affine_b=b, coeff=base_scale,
                          phase=PHASE_BOTH))
    return tuple(ops)


def _adapt_vector(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                  sidecar_bytes: bytes = b"", *,
                  header_ids: Optional[Tuple[int, ...]] = None,
                  turn_end_ids: Optional[Tuple[int, ...]] = None
                  ) -> Tuple[HookOp, ...]:
    """vector (methods.VectorObject over a SteeringVector container) —
    the plain additive arm, and the served form of every trainer that
    emits a SteeringVector: the diff-in-means family (CAA, instruction
    diff) and BiPO alike, which are extraction methods rather than object
    kinds of their own.

    Deployment is ``hooks.steering`` with the object's ``positions`` and
    ``skip_bos`` (defaults "all"/True), both phases, and NO baked scale —
    the magnitude lives entirely in the vector's norm, so ``coeff``
    stays 1.0 and the selector scale multiplies the stored vector
    directly. The add happens in the MODEL dtype (hooks.py casts the
    vector before multiplying), which the "model" materialization policy
    reproduces.

    Two VectorObject options are NOT this op and refuse rather than
    approximate: ``norm_relative`` rescales the addition by each
    position's own hidden norm, and a container carrying
    ``projection_targets`` deploys the Stolfo projection edit, which
    MOVES each position's projection onto the unit direction to a target
    value instead of adding to it."""
    tensors, side = _steering_vector_container("vector", artifact_bytes,
                                               sidecar_bytes)
    method = side.get("method")
    if method == "antidote":
        raise AdapterError(
            "kind 'vector': the sidecar declares method='antidote', which "
            "load_object hydrates as AntidoteObject (positions='last', a "
            "calibrated base_alpha) — a different intervention with its "
            "own adapter; refusing to serve it as a plain vector")
    meta = side.get("metadata")
    if isinstance(meta, Mapping) and meta.get("projection_targets"):
        raise AdapterError(
            "kind 'vector': the container carries 'projection_targets', so "
            "it deploys the projection edit (each position's projection "
            "onto the unit direction is MOVED to a target value, not added "
            "to) — that is not an additive residual op and has no "
            "translation here")
    if resolved_spec.get("norm_relative"):
        raise AdapterError(
            "kind 'vector': norm_relative=True scales the addition by each "
            "position's own hidden-state norm, so the added vector is not "
            "constant across positions — not expressible as a fixed "
            "additive op; refusing rather than serving the unscaled add")
    positions = resolved_spec.get("positions", "all")
    skip_bos = resolved_spec.get("skip_bos", True)
    if not isinstance(skip_bos, bool):
        raise AdapterError(f"kind 'vector': skip_bos must be a bool, got "
                           f"{skip_bos!r}")
    mode, header, turn_end = _served_positions(
        "vector", positions, header_ids, resolved_spec, turn_end_ids)
    ops = []
    d = None
    for l in sorted(tensors):
        vec = _f32_1d("vector", f"layer_{l}", tensors[l], dim=d)
        d = vec.shape[0]
        ops.append(HookOp(op=OP_ADD_VECTOR, layer=l, vector=vec,
                          positions=mode, skip_bos=skip_bos,
                          phase=PHASE_BOTH, header_ids=header,
                          turn_end_ids=turn_end))
    return tuple(ops)


def _lora_pair_ops(kind: str, layer_pairs: Sequence[Tuple[int, Any, Any]],
                   mode: str, header: Optional[Tuple[int, ...]],
                   turn_end: Optional[Tuple[int, ...]] = None
                   ) -> Tuple[HookOp, ...]:
    """One OP_LORA_OPROJ per (layer, A, B) pair, with the shared shape
    discipline of the two lora kinds: A [d_in] / B [d_out] rank-1
    vectors (RankOneLoRAObject and this repo's own lora_all trainer) or
    [d_in, k] / [d_out, k] rank-k matrices, the
    pair agreeing on its own rank. The op stores the READ basis
    transposed ([k, d_in], the existing lowrank_a convention) so
    ``(x @ lowrank_aᵀ) @ lowrank_bᵀ`` is exactly the hook body's
    ``(x @ A) @ Bᵀ``."""
    ops = []
    for layer, A, B in layer_pairs:
        for name, t in (("A", A), ("B", B)):
            if not isinstance(t, torch.Tensor):
                raise AdapterError(
                    f"kind {kind!r}: {name}[{layer}] must be a tensor, got "
                    f"{type(t).__name__}")
            if not torch.isfinite(t).all():
                raise AdapterError(
                    f"kind {kind!r}: {name}[{layer}] carries non-finite "
                    "values — refusing to serve a corrupt tensor")
        if A.dim() != B.dim() or A.dim() not in (1, 2):
            raise AdapterError(
                f"kind {kind!r} layer {layer}: A is {tuple(A.shape)} and B "
                f"is {tuple(B.shape)} — the pair must both be 1-D (rank-1 "
                "vectors) or both [d, k] matrices (the load_blob rule); "
                "refusing")
        rank = 1 if A.dim() == 1 else int(A.shape[1])
        if A.dim() == 2 and int(B.shape[1]) != rank:
            raise AdapterError(
                f"kind {kind!r} layer {layer}: A rank {rank} != B rank "
                f"{int(B.shape[1])} (A {tuple(A.shape)}, B "
                f"{tuple(B.shape)}) — refusing")
        a2 = A.detach().to(torch.float32).reshape(-1, rank) \
            if A.dim() == 1 else A.detach().to(torch.float32)
        b2 = B.detach().to(torch.float32).reshape(-1, rank) \
            if B.dim() == 1 else B.detach().to(torch.float32)
        ops.append(HookOp(
            op=OP_LORA_OPROJ, layer=int(layer),
            lowrank_a=a2.t().contiguous(),      # [k, d_in]
            lowrank_b=b2.contiguous(),          # [d_out, k]
            positions=mode, skip_bos=False, phase=PHASE_BOTH,
            header_ids=header, turn_end_ids=turn_end))
    return tuple(ops)


def _fit_leg_positions(kind: str, blob: Mapping[str, Any],
                       resolved_spec: Mapping[str, Any],
                       header_ids: Optional[Tuple[int, ...]],
                       turn_end_ids: Optional[Tuple[int, ...]] = None
                       ) -> Tuple[str, Optional[Tuple[int, ...]],
                                  Optional[Tuple[int, ...]]]:
    """The served position mode + header ids (+ the all-turns mode's
    terminator ids) for a blob that records
    its own fit leg (``positions`` [+ ``response_header_len``] — the
    lora kinds and psr), with the fit leg's records cross-checked.

    The mode belongs to the CELL (:func:`_served_positions` — the spec
    overrides the artifact's fit default; sound for every caller
    because their edits are derived per position from that position's
    own input — the o_proj edit from its o_proj input, the psr gate
    from its hidden state — so they satisfy DERIVATION LOCALITY, and a
    response serve is a pure
    positional subset of the all-position footprint). When BOTH the fit
    recorded a header width (``response_header_len``) and the serve is
    response-masked, the two must agree EXACTLY — like with like: the
    fit masked the last H_fit prompt positions by WIDTH
    (hooks.PositionTracker), and ``header_ids`` are the id-form of the
    SAME width convention (``chat_markers.generation_header_ids`` — the
    tokens add_generation_prompt appends), so on the template the
    object was fit against the two are the same number (gemma-4: 3;
    Llama-3.1: 4). A disagreement is a template drift and refuses.
    (The registry must NOT carry the terminator-anchored
    ``derive_assistant_markers().header`` here: that locator absorbs
    prompt-side scaffold — gemma-4's ``\n`` after ``<turn|>`` — and
    serving it would steer one token OUTSIDE the fit support while this
    check spuriously refused 3-vs-4; verify_assistant_header holds the
    registry to the generation-prompt convention at worker startup.)"""
    positions = blob.get("positions", "all")
    mode, header, turn_end = _served_positions(
        kind, positions, header_ids, resolved_spec, turn_end_ids)
    fit_h = blob.get("response_header_len")
    if fit_h is not None:
        if isinstance(fit_h, bool) or not isinstance(fit_h, int) \
                or fit_h <= 0:
            raise AdapterError(
                f"kind {kind!r}: response_header_len must be a positive "
                f"int when present, got {fit_h!r} — an H <= 0 boundary "
                "silently disables the whole steered region")
        if mode in RESPONSE_MASK_MODES and fit_h != len(header):
            raise AdapterError(
                f"kind {kind!r}: the fit recorded a {fit_h}-token "
                f"assistant header but the served generation-prompt "
                f"header is {len(header)} tokens ({list(header)}) — the "
                "fit's width mask and the serve's id-located mask would "
                "cover DIFFERENT regions; the object was fit against a "
                "different chat template (or the registry was exported "
                "with the terminator-anchored located header instead of "
                "chat_markers.generation_header_ids). Refusing")
    _spec_check(kind, resolved_spec, "response_header_len", fit_h)
    return mode, header, turn_end


def _adapt_lora(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                sidecar_bytes: bytes = b"", *,
                header_ids: Optional[Tuple[int, ...]] = None,
                turn_end_ids: Optional[Tuple[int, ...]] = None
                ) -> Tuple[HookOp, ...]:
    """lora (extraction/objects.RankOneLoRAObject): rank-1 LoRA on ONE
    layer's attention o_proj — ``coef = x @ A; out += scale *
    coef.unsqueeze(-1) * B`` with x the o_proj INPUT. Served at
    HOOK_SITE_OPROJ as the k=1 case of OP_LORA_OPROJ (the same
    reduction over d_in then outer write over d_out). The genuine PEFT
    export (lora_export.py) remains the committed path for all-position
    fits; the lens site is what serves the position-masked ones."""
    _refuse_sidecar("lora", sidecar_bytes)
    blob = _torch_blob("lora", artifact_bytes)
    for key in ("A", "B", "layer"):
        if key not in blob:
            raise AdapterError(f"kind 'lora': artifact lacks {key!r} — not "
                               "a RankOneLoRAObject.save blob")
    layer = blob["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise AdapterError(f"kind 'lora': layer must be an int >= 0, got "
                           f"{layer!r}")
    _spec_check("lora", resolved_spec, "layer", layer)
    A, B = blob["A"], blob["B"]
    if not (isinstance(A, torch.Tensor) and A.dim() == 1
            and isinstance(B, torch.Tensor) and B.dim() == 1):
        raise AdapterError(
            "kind 'lora': A and B must be 1-D tensors (RankOneLoRAObject "
            "is rank-1 by construction; the rank-general kind is "
            f"'lora_all'), got "
            f"{tuple(A.shape) if isinstance(A, torch.Tensor) else type(A).__name__}"
            " and "
            f"{tuple(B.shape) if isinstance(B, torch.Tensor) else type(B).__name__}")
    mode, header, turn_end = _fit_leg_positions(
        "lora", blob, resolved_spec, header_ids, turn_end_ids)
    return _lora_pair_ops("lora", [(layer, A, B)], mode, header, turn_end)


def _adapt_lora_all(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                    sidecar_bytes: bytes = b"", *,
                    header_ids: Optional[Tuple[int, ...]] = None,
                    turn_end_ids: Optional[Tuple[int, ...]] = None
                    ) -> Tuple[HookOp, ...]:
    """lora_all (extraction/multilayer.MultiLayerLoRAObject): low-rank
    LoRA on EVERY stored layer's attention o_proj, rank-general —
    ``out += scale * (x @ A_l) @ B_lᵀ`` per layer, x the o_proj INPUT
    (the class docstring's two blob shapes: this repo's 1-D rank-1
    pairs, the external [d, k] rank-k pairs — the scale-12 lora_r4
    objects are rank 4 over every layer, positions='response'). Served
    at HOOK_SITE_OPROJ. Every self-consistency rule of
    MultiLayerLoRAObject.load_blob is enforced here too: A/B dim and
    rank agreement, a `rank` key matching the tensors, `layers`
    metadata matching the tensors' own coverage."""
    _refuse_sidecar("lora_all", sidecar_bytes)
    blob = _torch_blob("lora_all", artifact_bytes)
    for key in ("A", "B"):
        if key not in blob or not isinstance(blob[key], Mapping) \
                or not blob[key]:
            raise AdapterError(
                f"kind 'lora_all': artifact lacks a non-empty {key!r} "
                "mapping — not a MultiLayerLoRAObject.save blob")
    layers = sorted(int(k) for k in blob["A"])
    if sorted(int(k) for k in blob["B"]) != layers:
        raise AdapterError(
            "kind 'lora_all': A and B cover different layers "
            f"({layers} vs {sorted(int(k) for k in blob['B'])}) — "
            "refusing a self-inconsistent artifact")
    if "layers" in blob \
            and sorted(int(x) for x in blob["layers"]) != layers:
        raise AdapterError(
            f"kind 'lora_all': the blob's `layers` metadata "
            f"{sorted(int(x) for x in blob['layers'])} disagrees with the "
            f"layers its A tensors actually cover {layers} — refusing to "
            "serve a self-inconsistent artifact (the load_blob rule)")
    if "layers" in resolved_spec:
        spec_layers = sorted(int(l) for l in resolved_spec["layers"])
        if spec_layers != layers:
            raise AdapterError(
                f"kind 'lora_all': resolved_spec layers {spec_layers} != "
                f"artifact layers {layers} (F9-style cross-check)")
    mode, header, turn_end = _fit_leg_positions(
        "lora_all", blob, resolved_spec, header_ids, turn_end_ids)
    pairs = [(l, blob["A"][str(l)], blob["B"][str(l)]) for l in layers]
    ops = _lora_pair_ops("lora_all", pairs, mode, header, turn_end)
    ranks = {op.lowrank_a.shape[0] for op in ops}
    if len(ranks) != 1:
        raise AdapterError(
            f"kind 'lora_all': layers disagree on rank ({sorted(ranks)}) "
            "— one object, one rank; refusing")
    rank = ranks.pop()
    if "rank" in blob and int(blob["rank"]) != rank:
        raise AdapterError(
            f"kind 'lora_all': tensors are rank {rank} but the blob's "
            f"`rank` metadata says {int(blob['rank'])} — the artifact's "
            "own records disagree; refusing (the load_blob rule)")
    _spec_check("lora_all", resolved_spec, "rank", rank)
    return ops


def _adapt_lora_subtraction(resolved_spec: Mapping[str, Any],
                            artifact_bytes: bytes, sidecar_bytes: bytes = b"",
                            *, header_ids: Optional[Tuple[int, ...]] = None,
                            turn_end_ids: Optional[Tuple[int, ...]] = None
                            ) -> Tuple[HookOp, ...]:
    """lora_subtraction (extraction/multilayer.LoRASubtractionObject): the
    task-vector DIFFERENCE of two same-shape ``lora_all`` fits,
    ``ΔW(pos) − ΔW(neg)`` per layer (``ΔW = B @ Aᵀ`` in the factor layout
    ``A[l] [d_in, r]``, ``B[l] [d_out, r]``) — the sdf-base-steering
    round-2 corpus-minus-control decomposition as a servable object,
    ported onto main's adapter API 2026-09-10 (bcb-bfcl-200 lane).

    NOT A NEW SERVED SHAPE, and the translation says so structurally. The
    artifact stores BOTH parents' (A, B) factors bit-exact and the class's
    ``apply`` registers each parent's OWN ``MultiLayerLoRAObject`` hook on
    the same o_proj — the positive at ``+scale``, the negative at
    ``-scale`` — so the served edit is exactly two OP_LORA_OPROJ ops per
    layer at the existing HOOK_SITE_OPROJ: the pos op at ``coeff=+1`` and
    the neg op at ``coeff=-1``, emitted in that per-layer order because
    forward hooks compose additively in REGISTRATION order and
    :func:`apply_oproj_ops` applies a layer's ops in list order. No new
    op, no new hook site, no new reduction class.

    THE COEFFICIENT CARRIES THE SIGN, not a negated write basis. Both
    reach the same value in IEEE-754, but ``coeff=-1`` is what the class
    itself does (``neg.register_hooks(-scale, …)`` — ``eff = scale *
    coeff`` is that call's ``scale`` argument, term for term), and it
    keeps ``op.lowrank_b`` the parent's OWN bytes, so the "both parents'
    factors ride bit-exact" property the algebra pins depend on is
    visible in the served ops rather than folded away.

    Every same-shape precondition ``LoRASubtractionObject._refuse_mismatch``
    enforces on the blobs is enforced here too (equal layer sets, equal
    rank, equal per-layer (A, B) shapes across the two parents), plus
    ``lora_all``'s own self-consistency rules per side. Positions are the
    recorded fit leg + the cell's override, exactly as for the lora kinds
    (the shared :func:`_fit_leg_positions`, so the all-turns mask's
    ``turn_end_ids`` ride through to every op): the edit at a position is
    derived from THAT position's o_proj input, so DERIVATION LOCALITY
    holds and a response serve is a pure positional subset (the parents
    are required to share one fit leg, so there is a single leg to
    serve)."""
    _refuse_sidecar("lora_subtraction", sidecar_bytes)
    blob = _torch_blob("lora_subtraction", artifact_bytes)
    factor_keys = ("A_pos", "B_pos", "A_neg", "B_neg")
    for key in factor_keys:
        if key not in blob or not isinstance(blob[key], Mapping) \
                or not blob[key]:
            raise AdapterError(
                f"kind 'lora_subtraction': artifact lacks a non-empty "
                f"{key!r} mapping — not a LoRASubtractionObject.save "
                "blob (its parents' factors ride under A_pos/B_pos and "
                "A_neg/B_neg)")
    layers = sorted(int(k) for k in blob["A_pos"])
    for key in factor_keys:
        if sorted(int(k) for k in blob[key]) != layers:
            raise AdapterError(
                f"kind 'lora_subtraction': {key!r} covers layers "
                f"{sorted(int(k) for k in blob[key])} but A_pos covers "
                f"{layers} — the difference is only defined layerwise "
                "on the SAME set; refusing a self-inconsistent artifact")
    if "layers" in blob \
            and sorted(int(x) for x in blob["layers"]) != layers:
        raise AdapterError(
            f"kind 'lora_subtraction': the blob's `layers` metadata "
            f"{sorted(int(x) for x in blob['layers'])} disagrees with the "
            f"layers its factors actually cover {layers} — refusing to "
            "serve a self-inconsistent artifact (the load_blob rule)")
    if "layers" in resolved_spec:
        spec_layers = sorted(int(l) for l in resolved_spec["layers"])
        if spec_layers != layers:
            raise AdapterError(
                f"kind 'lora_subtraction': resolved_spec layers "
                f"{spec_layers} != artifact layers {layers} (F9-style "
                "cross-check)")
    # the parents must be SAME-SHAPE per layer, or the difference is not a
    # task vector (LoRASubtractionObject._refuse_mismatch, on the blobs)
    for l in layers:
        for a_key, b_key in (("A_pos", "A_neg"), ("B_pos", "B_neg")):
            ta, tb = blob[a_key][str(l)], blob[b_key][str(l)]
            if not (isinstance(ta, torch.Tensor)
                    and isinstance(tb, torch.Tensor)):
                raise AdapterError(
                    f"kind 'lora_subtraction': layer {l} {a_key}/{b_key} "
                    "must both be tensors")
            if tuple(ta.shape) != tuple(tb.shape):
                raise AdapterError(
                    f"kind 'lora_subtraction': layer {l} {a_key} is "
                    f"{tuple(ta.shape)} but {b_key} is {tuple(tb.shape)} — "
                    "the parents are not same-shape, so their difference is "
                    "not a defined task vector; refusing")
    mode, header, turn_end = _fit_leg_positions(
        "lora_subtraction", blob, resolved_spec, header_ids, turn_end_ids)
    ops: List[HookOp] = []
    ranks = set()
    for l in layers:
        # per layer: the POSITIVE parent's op then the NEGATIVE parent's —
        # the class's registration order, which is its composition order
        pos_op, = _lora_pair_ops("lora_subtraction",
                                 [(l, blob["A_pos"][str(l)],
                                   blob["B_pos"][str(l)])],
                                 mode, header, turn_end)
        neg_op, = _lora_pair_ops("lora_subtraction",
                                 [(l, blob["A_neg"][str(l)],
                                   blob["B_neg"][str(l)])],
                                 mode, header, turn_end)
        ranks.add(pos_op.lowrank_a.shape[0])
        ranks.add(neg_op.lowrank_a.shape[0])
        ops.append(pos_op)
        ops.append(replace(neg_op, coeff=-1.0))
    if len(ranks) != 1:
        raise AdapterError(
            f"kind 'lora_subtraction': layers/parents disagree on rank "
            f"({sorted(ranks)}) — one object, one rank; refusing")
    rank = ranks.pop()
    if "rank" in blob and int(blob["rank"]) != rank:
        raise AdapterError(
            f"kind 'lora_subtraction': tensors are rank {rank} but the "
            f"blob's `rank` metadata says {int(blob['rank'])} — the "
            "artifact's own records disagree; refusing (the load_blob rule)")
    _spec_check("lora_subtraction", resolved_spec, "rank", rank)
    return tuple(ops)


#: the stacked prompt+object WRAPPER kind (mirrors
#: ``manifest.PETRI_STACKED_KIND`` — pinned equal by test; this module
#: deliberately imports no manifest machinery).
STACKED_KIND = "stacked"


def _adapt_stacked(resolved_spec: Mapping[str, Any], artifact_bytes: bytes,
                   sidecar_bytes: bytes = b"", *,
                   header_ids: Optional[Tuple[int, ...]] = None,
                   turn_end_ids: Optional[Tuple[int, ...]] = None
                   ) -> Tuple[HookOp, ...]:
    """kind 'stacked' — a verified SI system prefix AND a steered object
    served simultaneously on one cell (stacked routing).

    The worker serves ONLY the object leg: the SI prefix rides in-band
    as prompt text the proxy hash-verifies per request (H8), so the hook
    steers exactly as it would for the object alone — there is no
    physical stacking mechanism to add, only the wrapper identity to
    validate. This adapter checks the wrapper spec's structure (the same
    shape ``defense_cells.stacked_resolved_spec`` builds and the
    intervention hash covers) and DELEGATES to the object kind's own
    adapter, so the resulting ops are bit-identical to the object-only
    single's — any difference between the stack and its singles at serve
    time is then attributable to the prompt leg and the object leg
    themselves, never to a divergent translation."""
    spec = dict(resolved_spec)
    allowed = {"kind", "si_id", "si_prefix_sha16", "object"}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise AdapterError(
            f"kind 'stacked': wrapper spec carries unknown key(s) "
            f"{unknown} (allowed: {sorted(allowed)})")
    si = spec.get("si_id")
    if not isinstance(si, str) or not si:
        raise AdapterError(
            f"kind 'stacked': wrapper spec si_id must be a non-empty "
            f"string, got {si!r}")
    sha = spec.get("si_prefix_sha16")
    if (not isinstance(sha, str) or len(sha) != 16
            or any(c not in "0123456789abcdef" for c in sha)):
        raise AdapterError(
            f"kind 'stacked': wrapper spec si_prefix_sha16 must be a "
            f"16-hex hash of the exact SI prefix bytes, got {sha!r} — a "
            "stack whose prompt leg is unbound is half an identity")
    obj = spec.get("object")
    if not isinstance(obj, Mapping) or not obj:
        raise AdapterError(
            "kind 'stacked': wrapper spec needs a non-empty 'object' "
            "mapping (the object leg's own resolved spec)")
    okind = obj.get("kind")
    if not isinstance(okind, str) or not okind:
        raise AdapterError(
            f"kind 'stacked': object leg carries no kind, got {okind!r}")
    if okind == STACKED_KIND:
        raise AdapterError(
            "kind 'stacked': nested stacks are refused — one prompt leg, "
            "one object leg, one cell")
    return get_adapter(okind)(obj, artifact_bytes, sidecar_bytes,
                              header_ids=header_ids,
                              turn_end_ids=turn_end_ids)


register_adapter(SYNTHETIC_KIND, _adapt_synthetic)
register_adapter(ZERO_KIND, _adapt_zero)
register_adapter("psr", _adapt_psr)
register_adapter("psr_paired_write", _adapt_psr_paired_write)
register_adapter("vector_all", _adapt_vector_all)
register_adapter("antidote", _adapt_antidote)
register_adapter("alphasteer", _adapt_alphasteer)
register_adapter("loreft", _adapt_loreft)
register_adapter("alphakl", _adapt_alphakl)
register_adapter("static_r", _adapt_static_r)
register_adapter("caa_static_r", _adapt_caa_static_r)
register_adapter("cosine_gated_r", _adapt_cosine_gated_r)
register_adapter("lowrank_gated", _adapt_lowrank_gated)
register_adapter("loreft_both", _adapt_loreft_both)
register_adapter("vector", _adapt_vector)
register_adapter("lora", _adapt_lora)
register_adapter("lora_all", _adapt_lora_all)
register_adapter("lora_subtraction", _adapt_lora_subtraction)
register_adapter(STACKED_KIND, _adapt_stacked)


# --- applying ops (the worker-side hook body) --------------------------------


def materialize_ops(ops: Sequence[HookOp], *, device: Any,
                    dtype: Any) -> Tuple[HookOp, ...]:
    """Materialize immutable device/dtype-specific hook ops ONCE — at
    worker startup, after the model device and serving dtype are known
    (``WorkerObjectRegistry.materialize`` calls this). The request-time
    hook (:func:`apply_hook_ops`) then performs only the tensor op:
    a per-generated-token host-to-device copy inside the hook loop was
    repeated allocator/PCIe pressure for the whole serving run.

    Per-field dtype policy (:data:`_OP_TENSOR_FIELDS`): "model" fields
    take the serving ``dtype``; "fp32" fields move device-only and STAY
    float32 — the gated/affine ops compute in fp32 and cast the result
    back, exactly like the authoritative HF hook bodies."""
    out = []
    for op in ops:
        changes: Dict[str, torch.Tensor] = {}
        for name, policy in _OP_TENSOR_FIELDS[op.op].items():
            t = getattr(op, name)
            target_dtype = dtype if policy == "model" else torch.float32
            moved = t.to(device=device, dtype=target_dtype)
            if moved is not t:
                changes[name] = moved
        out.append(replace(op, **changes) if changes else op)
    return tuple(out)


def _guard_tensor(op: HookOp, name: str, t: torch.Tensor,
                  hidden: torch.Tensor, *, layer: int,
                  want_model_dtype: bool) -> None:
    want_dtype = hidden.dtype if want_model_dtype else torch.float32
    if t.device != hidden.device or t.dtype != want_dtype:
        raise AdapterError(
            f"hook op {op.op!r} at layer {layer}: {name} is {t.dtype} on "
            f"{t.device} but the op needs {want_dtype} on {hidden.device} "
            "— materialize the registry ONCE at worker startup "
            "(WorkerObjectRegistry.materialize / materialize_ops), never a "
            "per-token conversion inside the hook loop")


def _response_start(op: HookOp, ctx: HookContext, T: int) -> int:
    """The first steered PREFILL row of a ``positions="response"`` op:
    the start of the assistant header, LOCATED by matching the header's
    token ids against this request's own tokens.

    Deliberately not ``P - H`` arithmetic alone. ``P - H`` assumes the
    header is the last thing in the prompt, which every ordinary render
    satisfies but assistant prefill (prefix forcing — seeding the start
    of the assistant turn) does not: there the last H tokens are SEEDED
    CONTENT, so an arithmetic mask steers the seed and leaves the header
    unsteered, producing exactly the unsteered-token-1 condition the
    cross-phase ruling forbids — while looking correct and passing every
    width check. So the ids are verified and a mismatch REFUSES.

    THE SEEDED ARM (``ctx.response_prefill`` — the served-loglikelihood
    landing, 2026-09-01). A prefill-seeded render IS served, but only
    when the request DECLARED it (the proxy's ``petri_response_prefill``
    gate; a seeded render arriving undeclared still refuses below): the
    boundary is then the LAST occurrence of the header ids anywhere in
    the prompt — ``parity._response_letter_nlls``'s ``_rfind_subseq``
    semantics, so the served mask (header + seed + every generated
    position) matches the HF reference scorer's re-pointed boundary
    exactly, and the final prefill row (the forward that emits token 1,
    i.e. the scored distribution of a teacher-forced letter readout) is
    inside the mask BY CONSTRUCTION whenever the request is served at
    all. A declared-seeded prompt in which the header ids do not occur
    refuses — boundary retokenization across the header/seed boundary,
    or a foreign template; never a width-arithmetic guess.

    THE AGENT-LOOP ARM (``ctx.agent_loop`` — the declared multi-turn
    landing, 2026-09-02) is the SAME locator under the proxy's
    ``petri_agent_loop`` declaration: a Petri agent-loop render (an open
    model turn after tool traffic) never ends at the header — the three
    real shapes end in a tool-response opener, a closed tool response,
    and a closed turn respectively — so the boundary is the last header
    occurrence, and the mask from there covers the assistant's tool call
    and the tool-response syntax and body along with every generated
    position (module docstring: the completed audit's regime, kept for
    comparability). Undeclared, the same render refuses below exactly as
    before."""
    if ctx.token_ids is None:
        raise AdapterError(
            f"hook op {op.op!r} at layer {op.layer} steers the response "
            "region, which is located by matching the assistant header's "
            "token ids — but the forward context carries no token ids. The "
            "hook wiring must slice this request's ids out of the "
            "flattened forward alongside its hidden states; refusing "
            "rather than falling back to a width-only mask (which is "
            "wrong under assistant prefill)")
    if ctx.is_resumed:
        # THE CONDITIONAL RESUME (module docstring, RESUMED PREFILLS): a
        # response op serves a resumed prefill only when the cached region
        # is header-free — VERIFIED here from the cached ids the wiring
        # supplies, never trusted from a flag, exactly as the boundary is
        # located rather than computed from a width. With it verified, the
        # boundary located in the slice below is the full-prompt boundary
        # minus ``cached``: the header is entirely inside the slice.
        if ctx.cached_ids is None:
            raise ResumedPrefillRefusal(
                f"hook op {op.op!r} at layer {op.layer} steers the response "
                f"region on a RESUMED prefill (cached={ctx.cached}) but the "
                "context carries no cached_ids — the cached region cannot be "
                "verified header-free; refusing (the wiring must supply the "
                "request's prompt ids [0, cached))")
        # an op carrying a declared SECOND header is checked against BOTH
        # (conservative: a cached region holding either header is a prior
        # writer's boundary; unreachable in practice — declared cells are
        # minted request-unique salts by the proxy)
        for hdr in _op_headers(op):
            conflict = cached_region_header_conflict(hdr, ctx.cached_ids)
            if conflict is not None:
                raise ResumedPrefillRefusal(
                    f"hook op {op.op!r} at layer {op.layer}: RESUMED prefill "
                    f"(cached={ctx.cached}) on a response-mode op — {conflict}; "
                    "resumed_prefill_policy is conditional and this request's "
                    "cached region is not header-free, refusing (contained per "
                    "request)")
    return _locate_final_header(op, ctx, ctx.token_ids, T)


def _op_headers(op: HookOp) -> Tuple[Tuple[int, ...], ...]:
    """The header id sequences a header-located op carries: its OFF header
    and, when the cell declared the thinking-ON render, its ON header."""
    out = [tuple(op.header_ids)]
    if op.thinking_header_ids is not None:
        out.append(tuple(op.thinking_header_ids))
    return tuple(out)


def _locator_header(op: HookOp, ctx: HookContext) -> Tuple[int, ...]:
    """WHICH header this request's boundary is located against: the
    declared thinking-ON header under ``ctx.thinking_on`` (the request
    renders under the ON regime, whose header differs), else the
    op's OFF header. A thinking-ON declaration reaching an op that carries
    no second header refuses, naming the missing CELL declaration: the
    proxy admits ``petri_thinking_on`` only on a cell whose attested
    declaration carries it, so reaching this is a registry/proxy
    disagreement, contained per request."""
    if not ctx.thinking_on:
        return tuple(op.header_ids)
    if op.thinking_header_ids is None:
        raise AdapterError(
            f"hook op {op.op!r} at layer {op.layer}: the request declares a "
            "thinking-ON render (thinking_on) but this op carries no "
            "thinking-ON header — the cell's hashed spec does not declare "
            f"`{sd.SERVE_OUT_OF_SUPPORT_KEY}: [\"{sd.OUT_OF_SUPPORT_THINKING}\"]` "
            "(spelled `<cell>@thinking`), so no second header was attached at "
            "load (attach_thinking_header). Serve the declared render on the "
            "declared cell; refusing rather than locating the OFF header in "
            "an ON render (which never contains it)")
    return tuple(op.thinking_header_ids)


def _locate_final_header(op: HookOp, ctx: HookContext, ids: Sequence[int],
                         T: int) -> int:
    """The start of the GENERATING turn's assistant header in ``ids`` —
    :func:`_response_start`'s boundary rules, verbatim and in the same
    order, over an explicit id sequence of ``T`` rows: ``H > T`` refuses;
    a DECLARED non-tail render (seeded / agent-loop) takes the LAST
    header occurrence and refuses when none exists; an ordinary render
    must END at the header and refuses otherwise. Factored out so THE
    ALL-TURNS MASK can locate the same boundary over the FULL prompt of
    a resumed request (cached region + slice) while ``response`` keeps
    calling it on its slice exactly as before.

    The arithmetic is ``span_locator.locate_final_header`` — the ONE
    implementation the proxy's per-request render check runs on the same
    ids — over the header this request is
    located against (:func:`_locator_header`: the declared thinking-ON
    header under ``ctx.thinking_on``, else the OFF header); its refusals are
    re-raised here as this module's :class:`AdapterError`, message for
    message."""
    header = _locator_header(op, ctx)
    H = len(header)
    declared = _declared_locator(ctx)
    try:
        return sl.locate_final_header(ids, header, declared=declared is not None)
    except sl.SpanLocatorError as e:
        if e.reason == sl.REASON_PROMPT_SHORTER_THAN_HEADER:
            raise AdapterError(
                f"hook op {op.op!r} at layer {op.layer}: assistant header is "
                f"{H} tokens but the prompt is only {T} — a prompt shorter "
                "than its own header means the render and the served chat "
                "template disagree; refusing") from None
        if e.reason == sl.REASON_HEADER_NOWHERE:
            raise AdapterError(
                f"hook op {op.op!r} at layer {op.layer}: the request "
                f"declares {declared} but the assistant header ids "
                f"{list(header)} occur NOWHERE in its prompt tokens "
                "— either the render used a different chat template than "
                "the one this object was exported against, or the seed "
                "retokenized across the header boundary. The boundary "
                "must be located in the request's own tokens, never "
                "assumed; refusing") from None
        if e.reason == sl.REASON_NOT_AT_TAIL:
            got = e.detail.get("got", ())
            raise AdapterError(
                f"hook op {op.op!r} at layer {op.layer}: the last {H} prompt "
                f"tokens are {list(got)}, not the assistant header "
                f"{list(header)} — this prompt does not end at the "
                "header. Either the render used a different chat template "
                "than the one this object was exported against, or the turn "
                "is prefill-seeded (assistant prefix forcing) or is a Petri "
                "agent-loop turn (an open model turn after tool traffic) — "
                "both served only when the request DECLARES it (the proxy's "
                "petri_response_prefill gate -> ctx.response_prefill, or its "
                "petri_agent_loop gate -> ctx.agent_loop) — or a thinking-ON "
                "render of a template whose thinking switch moves the header, "
                "served only when the request DECLARES it (the proxy's "
                "petri_thinking_on gate -> ctx.thinking_on) on a cell whose "
                "hashed spec declares the render (`<cell>@thinking`). "
                "Refusing: masking by width here would steer the seeded "
                "content and leave the header — and therefore the forward "
                "that emits the first generated token — unsteered") from None
        raise AdapterError(f"hook op {op.op!r} at layer {op.layer}: {e}") from None


def _declared_locator(ctx: HookContext) -> Optional[str]:
    """Which DECLARED non-tail render this context carries, as the noun
    the refusal messages use — or ``None`` for an ordinary render. The
    two declarations are mutually exclusive (``HookContext`` refuses
    both), so at most one names the located-boundary arm."""
    if ctx.response_prefill:
        return "a prefill-seeded render (response_prefill)"
    if ctx.agent_loop:
        return "a Petri agent-loop render (agent_loop)"
    return None


#: the pure span helpers ARE ``span_locator``'s (one implementation, shared
#: with the proxy's per-request render check); the historical
#: names stay for every caller and test.
_rfind_header = sl.rfind
_find_all = sl.find_all
_find_first = sl.find_first


def assistant_turn_spans(ids: Sequence[int], header_ids: Sequence[int],
                         turn_end_ids: Sequence[int], final_start: int
                         ) -> Tuple[Tuple[int, int], ...]:
    """``span_locator.assistant_turn_spans`` raising this module's
    :class:`AdapterError` (the message verbatim) — the all-turns locator the
    mask builder and the proxy's render check share. Every assistant-turn span ``(lo, hi)`` (``hi`` exclusive) of a FULL
    prompt ``ids`` under THE ALL-TURNS MASK (module docstring): for each
    assistant-header occurrence before ``final_start`` (the generating
    turn's header, located by :func:`_locate_final_header`) the span runs
    from the header THROUGH the first occurrence of the turn-END SCAFFOLD
    after it — the turn's own terminator plus the glue the template puts
    after it (gemma-4 ``<turn|>\\n``; :func:`sisteer.chat_markers.
    assistant_turn_end_ids`) — and the generating turn's span runs from
    ``final_start`` to the end of the prompt (open: every generated
    position is masked). An earlier header that is not closed by the
    scaffold before the next header REFUSES: the render and the served
    template disagree about where assistant turns end, and a guessed end
    would steer the user's turn.

    WHY THE SCAFFOLD AND NOT THE BARE TERMINATOR (the 2026-09-07 review's
    P1-2): the final span always runs to the prompt end, and a supported
    closed-tail render (gemma-4's Petri ``content_and_tool_response``
    shape) ends in ``<turn|>\\n`` — so while that turn is final its glue
    ``\\n`` is steered. Once the same turn is historical in a longer
    conversation whose prefix is token-identical (a user turn following
    the closed tool turn), a span that stopped at the terminator would
    leave that glue unsteered: same ids, different mask — a cached block
    holding the glue would be poison. Ending the historical span after the
    scaffold makes a turn's mask INVARIANT when it stops being final,
    verified on the real gemma-4 template (the scaffold between turns is
    exactly ``[<turn|>, \\n]``, what the closed-tail render ends with). The
    cost: those glue positions are steered historically, where the
    single-turn regime never encodes them as prompt rows at all (it
    generates the terminator and stops) — nothing the existing regime
    steers is contradicted, and user content is never inside a span.

    Template glue is therefore INSIDE the span; the user / system turns'
    own terminators and glue lie outside every span by construction: the
    search for a turn's end starts at its header, so a scaffold that
    closes another role's turn is never the one it finds.

    QUOTED DELIMITERS FAIL CLOSED, in both positions. The generating
    turn: a complete scaffold inside its content followed by more content
    refuses — as a historical turn the same tokens would end the span at
    that scaffold, so the two readings of one token sequence disagree.
    A historical turn: the first scaffold after its header must be
    followed by the next header itself or by a COMPLETE role header of
    the template's one shape — the opener, exactly one role token, the
    assistant header's tail (``header_ids[0]`` / ``header_ids[2:]``),
    certified for the template's USER header by
    :func:`sisteer.chat_markers.successor_role_header_ids` at registry
    build and at worker startup (only for a registry that actually serves
    all-turns ops); anything else there is a successor header this
    checker cannot certify — EITHER content quoting the template's own
    delimiter (the scaffold did not end the turn; reading it as the end
    would leave the rest of that turn unsteered: a cold counterexample of
    146 rows, and an opener-only bypass of 108 rows) OR a genuine turn of
    a role whose header has another shape (a multi-token role such as
    Llama-3.1's ``ipython``). The stream cannot tell the two apart, and
    the certification probes only the user header, so a tool-role turn of
    another shape is refused at APPLY, the first point the information
    exists — a template whose USER header already differs in shape
    (Nemotron's think-suffixed assistant header) refuses earlier, at build
    and startup. Every production shape — the three Petri agent-loop
    renders including the closed tail, ordinary single- and multi-turn
    renders, the declared seeded render, a bare terminator without its
    glue in either position — still serves
    (``tests/test_lens_all_turns_mask.py``, the real gemma-4 matrix).

    THE RESIDUAL LIMIT, stated plainly: content that reproduces an ENTIRE
    turn boundary — scaffold AND opener AND role — is indistinguishable
    from a real one by any reading of the token stream; that is
    precondition (b) of the module docstring (boundaries from the
    request's declared message structure), not something a
    refusal rule can close. Scaffold ids from another template find NO
    occurrence and refuse here; a registry exported against another
    template refuses at startup (``lens_steer.verify_assistant_turn_end``)."""
    try:
        return sl.assistant_turn_spans(ids, header_ids, turn_end_ids, final_start)
    except sl.SpanLocatorError as e:
        raise AdapterError(str(e)) from None


#: one implementation: the wrapper above IS ``span_locator.assistant_turn_spans``
#: with this module's error class (pinned by tests/test_all_turns_render_check.py)
assistant_turn_spans.__wrapped__ = sl.assistant_turn_spans  # type: ignore[attr-defined]


def _all_turns_prefill_mask(op: HookOp, ctx: HookContext, T: int
                            ) -> torch.Tensor:
    """THE ALL-TURNS MASK over one prefill forward's ``T`` rows, as a CPU
    bool tensor (the caller moves it): every assistant-turn span of the
    FULL prompt (:func:`assistant_turn_spans`), sliced to the rows in
    this forward. The full prompt is the request's own token ids — for a
    RESUMED prefill the cached region's ids (``ctx.cached_ids``) followed
    by the slice, because a turn that opened before the cache boundary
    (turn N-1's response, partly or wholly cached) can only be located
    from the cached ids. The generating turn's boundary is the one
    ``response`` locates (:func:`_locate_final_header`), with the same
    refusals — no token ids, a prompt shorter than its header, an
    undeclared render that does not end at the header, a declared one in
    which the header occurs nowhere — so no request shape is admitted
    here that ``response`` would refuse.

    THE CONDITIONAL RESUME (module docstring, RESUMED PREFILLS): a resumed
    prefill is served only when the cached region passes the mode's
    clauses — VERIFIED here from the cached ids
    (:func:`cached_region_header_conflict` with
    :data:`ALL_TURNS_CONFLICT_CLAUSES`), never trusted from the worker's
    decision alone, exactly as ``response`` re-verifies its own. Today
    those are the SAME two clauses as ``response``: a complete earlier
    header in the region refuses, a boundary inside a header refuses."""
    if ctx.token_ids is None:
        raise AdapterError(
            f"hook op {op.op!r} at layer {op.layer} steers every assistant "
            "turn's span, which is located by matching the assistant "
            "header's and turn-end scaffold's token ids — but the forward "
            "context carries no token ids. The hook wiring must slice this "
            "request's ids out of the flattened forward alongside its "
            "hidden states; refusing rather than falling back to a "
            "width-only mask")
    if len(ctx.token_ids) != T:
        raise AdapterError(
            f"hook op {op.op!r} at layer {op.layer}: the context carries "
            f"{len(ctx.token_ids)} token ids for a forward of {T} rows — "
            "mis-aligned forward metadata; refusing to guess rows")
    if ctx.is_resumed:
        if ctx.cached_ids is None:
            raise ResumedPrefillRefusal(
                f"hook op {op.op!r} at layer {op.layer} steers every "
                "assistant turn's span on a RESUMED prefill (cached="
                f"{ctx.cached}) but the context carries no cached_ids — the "
                "cache boundary cannot be verified to lie outside an "
                "assistant header, and an assistant turn that opened before "
                "the boundary can only be located from the cached region's "
                "ids; refusing (the wiring must supply the request's prompt "
                "ids [0, cached))")
        conflict = cached_region_header_conflict(
            op.header_ids, ctx.cached_ids, ALL_TURNS_CONFLICT_CLAUSES)
        if conflict is not None:
            raise ResumedPrefillRefusal(
                f"hook op {op.op!r} at layer {op.layer}: RESUMED prefill "
                f"(cached={ctx.cached}) on an all-turns op — {conflict}; "
                "resumed_prefill_policy is conditional and this request's "
                "cached region conflicts under the same clauses response "
                "checks, refusing (contained per request)")
        full: Tuple[int, ...] = tuple(ctx.cached_ids) + tuple(ctx.token_ids)
    else:
        full = tuple(ctx.token_ids)
    P = len(full)
    final = _locate_final_header(op, ctx, full, P)
    mask = [False] * P
    for lo, hi in assistant_turn_spans(full, op.header_ids,
                                       op.turn_end_ids, final):
        for i in range(lo, hi):
            mask[i] = True
    return torch.tensor(mask[ctx.cached:], dtype=torch.bool)


def attach_thinking_header(ops: Sequence[HookOp],
                           resolved_spec: Mapping[str, Any],
                           thinking_header_ids: Optional[Sequence[int]]
                           ) -> Tuple[HookOp, ...]:
    """THE ONE PLACE the declared thinking-ON header meets a cell's ops:
    a NEW op tuple, the adapters' output
    untouched otherwise, so adapter signatures stay put and the fit/serve
    guard stays a guard.

    Rules, every one a refusal rather than a default:

    1. the spec declares ``thinking`` (:func:`declares_thinking`) ⇒
       ``thinking_header_ids`` is REQUIRED (``None`` refuses: the registry
       exported no ON header for a cell that needs one);
    2. ... and is attached to EVERY ``positions="response"`` op;
    3. an op set with NO header-located op refuses — a thinking declaration
       on an ``all`` / ``last`` cell is vacuous (the mask does not move);
    4. a ``response_all_turns`` op refuses — never combined: the all-turns
       certification refuses Qwen (its historical turns carry no think
       block while its final turn does) and the two experiments run on
       different models;
    5. the spec does NOT declare ⇒ ``thinking_header_ids`` must be ``None``
       (a second header handed to an undeclared cell is a caller error, not
       something to attach quietly);
    6. a second header EQUAL to the op's header is the header-INVARIANT
       template's case (gemma-4: the thinking switch leaves the assistant
       header in place) — attached like any other, the boundary the same
       under both renders (plan P9's refusal withdrawn);
       a malformed declaration refuses in :func:`cell_declarations`."""
    ops = tuple(ops)
    declared = declares_thinking(resolved_spec)
    ids = (None if thinking_header_ids is None
           else tuple(int(t) for t in thinking_header_ids))
    if not declared:
        if ids is not None:
            raise AdapterError(
                "attach_thinking_header: a thinking-ON header was given but "
                "the cell's resolved spec does not declare the thinking-ON "
                f"render (`{sd.SERVE_OUT_OF_SUPPORT_KEY}: "
                f"[\"{sd.OUT_OF_SUPPORT_THINKING}\"]`, spelled "
                "`<cell>@thinking`) — a second header on an undeclared cell "
                "is refused, never attached quietly")
        return ops
    if ids is None:
        raise AdapterError(
            "attach_thinking_header: the cell declares the thinking-ON "
            f"render (`{sd.SERVE_OUT_OF_SUPPORT_KEY}: "
            f"[\"{sd.OUT_OF_SUPPORT_THINKING}\"]`) but no thinking-ON header "
            "was supplied — the registry must record "
            "assistant_header_ids_thinking_on (chat_markers."
            "generation_header_ids_under(tok, enable_thinking=True)) for a "
            "registry serving a declared cell; refusing")
    if any(op.positions == POSITIONS_RESPONSE_ALL_TURNS for op in ops):
        raise AdapterError(
            "attach_thinking_header: the cell declares the thinking-ON render "
            "but serves positions='response_all_turns' — the two are never "
            "combined (the all-turns certification refuses the one template "
            "whose thinking switch moves the header, and the two transfer "
            "experiments run on different models); refusing")
    if not any(op.positions in RESPONSE_MASK_MODES for op in ops):
        raise AdapterError(
            "attach_thinking_header: the cell declares the thinking-ON render "
            f"but none of its ops is header-located (positions={sorted({op.positions for op in ops})}) "
            "— the declaration is VACUOUS on an all / last cell, whose mask "
            "does not move with the render; refusing rather than attesting "
            "an out-of-support render that changes nothing")
    return tuple(replace(op, thinking_header_ids=ids)
                 if op.positions in RESPONSE_MASK_MODES else op
                 for op in ops)


def prefill_mask_rows(ops: Sequence[HookOp], ctx: Optional[RowContext]
                      ) -> Optional[int]:
    """AN INSTRUMENT: how
    many rows of ONE PREFILL forward the cell's ops select — the union over
    every op that applies in the prefill phase of the rows
    :func:`_row_selection` would touch (``None`` = every row; ``NO_ROWS`` =
    none). ``None`` for anything that is not a prefill :class:`HookContext`
    and for an op set with no prefill-phase op. On a Qwen3.5 response cell
    this reads 7 under the OFF render and 5 under a declared ON render; on a
    gemma-4 all-turns cell the generating header plus every historical span;
    on an ``all`` cell the whole prompt. Refusals are the mask builders' own
    (an unlocatable boundary raises) — the recorder that files this value
    contains them, never the apply path."""
    if not isinstance(ctx, HookContext) or ctx.phase != PHASE_PREFILL:
        return None
    T = len(ctx.positions)
    probe = torch.empty((T, 0))
    union = torch.zeros(T, dtype=torch.bool)
    counted = False
    for op in ops:
        if op.phase == PHASE_DECODE:
            continue  # not invoked in this phase
        counted = True
        sel = _row_selection(op, ctx, probe)
        if sel is None:
            union[:] = True
        elif sel is NO_ROWS:
            continue
        else:
            union |= sel.to("cpu")
    return int(union.sum().item()) if counted else None


def _row_mask(op: HookOp, ctx: Optional[RowContext],
              hidden: torch.Tensor) -> Optional[torch.Tensor]:
    """[T] bool mask of rows this op touches, or None for all rows —
    the DENSE, device-resident mask (BOS-skip applied WITHIN the
    selected region, orthogonal to phase — module docstring).

    THE REFERENCE BUILDER, kept byte-for-byte as it was before the
    vectorized apply landed (2026-09-04): it is what the per-row
    reference path (``SISTEER_LENS_PER_ROW_APPLY=1``) runs, so the GPU
    bit-identity gate compares the new path against the exact old
    arithmetic, and what the pre-existing tests pin. The serving hot
    path uses :func:`_row_selection` instead — same region logic, no
    device-side construction, no all-ones multiply. Two syncs live here
    on purpose (the blocking ``torch.tensor(..., device=cuda)`` and the
    caller's ``mask.any()``); they are the cost this function's sibling
    removes."""
    if op.positions == POSITIONS_ALL and not op.skip_bos:
        return None
    assert ctx is not None  # op_needs_context guaranteed by the caller
    _refuse_positioned_under_resumed(op, ctx)
    T = hidden.shape[-2]
    pos = torch.tensor(_ctx_positions_for_mask(ctx, T), dtype=torch.long,
                       device=hidden.device)
    if op.positions == POSITIONS_LAST:
        _refuse_last_under_declared(op, ctx)
        if ctx.phase == PHASE_PREFILL:
            mask = torch.zeros(T, dtype=torch.bool, device=hidden.device)
            mask[-1] = True  # the last prompt token (ctx pins contiguity)
        else:  # decode: every incremental forward's single position
            mask = torch.ones(T, dtype=torch.bool, device=hidden.device)
    elif op.positions == POSITIONS_RESPONSE:
        if ctx.phase == PHASE_PREFILL:
            mask = torch.zeros(T, dtype=torch.bool, device=hidden.device)
            mask[_response_start(op, ctx, T):] = True
        else:  # decode: every generated position is inside the region
            mask = torch.ones(T, dtype=torch.bool, device=hidden.device)
    elif op.positions == POSITIONS_RESPONSE_ALL_TURNS:
        if ctx.phase == PHASE_PREFILL:
            mask = _all_turns_prefill_mask(op, ctx, T).to(hidden.device)
        else:  # decode: every generated position is inside the region
            mask = torch.ones(T, dtype=torch.bool, device=hidden.device)
    else:
        mask = torch.ones(T, dtype=torch.bool, device=hidden.device)
    if op.skip_bos:
        mask = mask & (pos != 0)
    return mask


def _ctx_positions_for_mask(ctx: RowContext, T: int) -> Tuple[int, ...]:
    """The per-row positions a mask is built against: the request's own
    for a :class:`HookContext`; for a :class:`DecodeRows` group every
    row is a generated position, and only ``!= 0`` is ever asked of it,
    so any value >= 1 stands in (1 here) — decode rows are >= 1 by the
    HookContext contract the group was built from."""
    if isinstance(ctx, HookContext):
        return ctx.positions
    return (1,) * T


def _refuse_last_under_declared(op: HookOp, ctx: RowContext) -> None:
    declared = _declared_locator(ctx)
    if declared is not None:
        raise AdapterError(
            f"hook op {op.op!r} at layer {op.layer}: the request "
            f"declares {declared} but this op serves positions='last' "
            "— under such a render the last prompt token is SEED "
            "CONTENT or tool-traffic scaffold, not the position the "
            "arm's semantics name, and an edit derived from it would "
            "be a relocation, not a restriction (derivation locality, "
            "module docstring). Declared non-tail renders are defined "
            "for 'response' and 'all' deployments only; refusing")


def _refuse_positioned_under_resumed(op: HookOp, ctx: RowContext) -> None:
    """A ``positions="last"`` op under a RESUMED prefill context refuses
    (module docstring, RESUMED PREFILLS): the last prompt token of the
    slice IS the request's last token, but the writer of the cached rows
    had its own. (A ``response`` op is CONDITIONAL and is verified against
    the cached ids in :func:`_response_start` instead.) The apply loop's
    policy check fires first; this is the mask builders' own guard for
    direct callers."""
    if isinstance(ctx, HookContext) and ctx.is_resumed \
            and op.positions == POSITIONS_LAST:
        raise ResumedPrefillRefusal(
            f"hook op {op.op!r} at layer {op.layer} serves positions="
            f"{op.positions!r} but the forward is a RESUMED prefill "
            f"(cached={ctx.cached}): that mode places its edit by the "
            "request's prompt length, which the cached rows' writer placed "
            "by ITS OWN — the cache cannot be trusted to carry this op's "
            "edit; refusing (resumed_prefill_policy is never)")


#: the row-selection sentinel: this op touches NO row of this forward
#: (a fully-masked prefill — e.g. a 1-token BOS-only prompt under
#: skip_bos). Identity-compared (``sel is NO_ROWS``), never a tensor
#: the caller has to inspect.
NO_ROWS = object()


def _row_selection(op: HookOp, ctx: Optional[RowContext],
                   hidden: torch.Tensor) -> Any:
    """Which rows of ``hidden`` this op touches, WITHOUT a device sync:
    ``None`` = every row (the caller applies the op unmasked),
    :data:`NO_ROWS` = none (the caller leaves the tensor untouched),
    else a ``[T]`` bool device mask with at least one True row, built
    from CPU ints and moved once.

    EXACTNESS. The dense builder :func:`_row_mask` multiplied the edit
    by an all-ones mask on every decode row (``add * 1.0``, then the
    same cast) and asked the device ``mask.any()`` — two syncs and
    ~4 launches per (request, layer) that changed no bit: IEEE-754
    multiplication by 1.0 returns its operand exactly (finite,
    infinite, signed zero alike), so ``x + d`` and ``x + d * 1.0``
    are the same bits. This function returns ``None`` wherever the
    dense mask would have been all-ones BY CONSTRUCTION:

    - ``positions="all"`` without ``skip_bos`` (as before);
    - EVERY decode context: a decode row sits at a position >= 1
      (HookContext refuses position 0 in decode; DecodeRows groups are
      built from such contexts), so ``skip_bos`` removes nothing,
      ``positions="last"`` names the row, and the response region holds
      every generated position.

    A PREFILL context still gets its dense mask (the response boundary
    is per request), built on the CPU from the context's ints — the
    "fully masked?" decision is made there too — and moved to the
    device in ONE copy (pinned + non_blocking on CUDA). A prefill mask
    that happens to be all-True is still returned as a tensor: the
    prefill is not the hot path, and callers such as the cosine gate
    give ``None`` a different meaning.

    Refusals are the dense builder's, verbatim, in the same order:
    ``positions="last"`` under a declared non-tail render refuses
    before any phase branch."""
    if op.positions == POSITIONS_ALL and not op.skip_bos:
        return None
    assert ctx is not None  # op_needs_context guaranteed by the caller
    _refuse_positioned_under_resumed(op, ctx)
    if op.positions == POSITIONS_LAST:
        _refuse_last_under_declared(op, ctx)
    if ctx.phase == PHASE_DECODE:
        return None  # all-ones by construction — see the docstring
    assert isinstance(ctx, HookContext)  # DecodeRows are decode-only
    T = hidden.shape[-2]
    if op.positions == POSITIONS_LAST:
        cpu = torch.zeros(T, dtype=torch.bool)
        cpu[-1] = True
    elif op.positions == POSITIONS_RESPONSE:
        cpu = torch.zeros(T, dtype=torch.bool)
        cpu[_response_start(op, ctx, T):] = True
    elif op.positions == POSITIONS_RESPONSE_ALL_TURNS:
        cpu = _all_turns_prefill_mask(op, ctx, T)
    else:
        cpu = torch.ones(T, dtype=torch.bool)
    if op.skip_bos:
        cpu &= torch.tensor([p != 0 for p in ctx.positions],
                            dtype=torch.bool)
    if not bool(cpu.any()):  # CPU: no device sync
        return NO_ROWS
    return _mask_to_device(cpu, hidden.device)


def _mask_to_device(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """One host-to-device copy of a CPU-built mask. On CUDA the source is
    staged through PyTorch's pinned caching allocator so the copy is
    asynchronous (``torch.tensor(..., device=cuda)`` is a BLOCKING
    pageable copy — the sync this replaces); the allocator tracks the
    copy stream before reusing the pinned block, so dropping ``cpu``
    right after is safe. Elsewhere (CPU tests) a plain move."""
    if device.type != "cuda":
        return cpu.to(device)
    pinned = cpu.pin_memory()
    return pinned.to(device, non_blocking=True)


def _select_rows(op: HookOp, ctx: Optional[RowContext], hidden: torch.Tensor,
                 legacy_masks: bool) -> Any:
    """The row selection an apply site consumes: :func:`_row_selection`
    on the serving path; under ``legacy_masks`` the dense
    :func:`_row_mask` plus the device ``mask.any()`` read — the
    pre-vectorize control flow verbatim (the reference path)."""
    if not legacy_masks:
        return _row_selection(op, ctx, hidden)
    mask = _row_mask(op, ctx, hidden)
    if mask is None:
        return None
    if not bool(mask.any()):
        return NO_ROWS
    return mask


def _resumed_refusal(op: HookOp, layer: int, ctx: HookContext) -> AdapterError:
    return ResumedPrefillRefusal(
        f"hook op {op.op!r} at layer {layer} (positions={op.positions!r}, "
        f"phase={op.phase!r}) cannot be applied on a RESUMED prefill "
        f"(cached={ctx.cached} of this request's prompt positions came from "
        "the KV cache): its edit depends on positions outside the current "
        "row or on the writer's prompt boundary, so the cached rows' writer "
        "may have given them a different edit (module docstring, RESUMED "
        "PREFILLS; resumed_prefill_policy is never for this op — "
        "resumed_prefill_ok is False). The worker's lattice refuses such a "
        "resume before any apply; reaching this is a wiring defect, "
        "contained per request")


def _needs_ctx_refusal(op: HookOp, layer: int) -> AdapterError:
    return AdapterError(
        f"hook op {op.op!r} at layer {layer} is position/phase-dependent "
        f"(phase={op.phase!r}, positions={op.positions!r}, "
        f"skip_bos={op.skip_bos}) and needs a HookContext — the hook "
        "wiring must pass the per-request forward context (phase + global "
        "token positions); applying without one would guess rows, never "
        "allowed")


#: the matmul every row-wise reduction goes through — a module attribute
#: so a test can observe the CALL SHAPES (the property that matters:
#: which rows each cuBLAS call sees), not just the values.
_matmul = torch.matmul
#: its affine sibling for the LoReFT subspace projections (``F.linear``)
_linear = F.linear

#: ``SISTEER_LENS_STACKED_REDUCTIONS`` — the REDUCTION MODE inside a vectorized
#: group (:func:`_row_reduction`). DEFAULT ``stacked``
#: (docs/lens_stacked_reductions_scoping.md): every reduction
#: over the hidden dim inside a group — the PSR gate gemv, the LoRA rank-1 /
#: rank-K GEMM pair, the LoReFT affine ``F.linear`` pair and back-projection
#: — is ONE ``[B, K]`` call. ``0`` / ``per_row`` opts back to ``B`` single-row
#: calls, the bit-identical reference of the pre-landing per-row path (the
#: gate's shadow compare and any exact reproduction of an older run need it).
#: A stacked GEMM reduces in an (M, N, K)-dependent order, so the two modes
#: are NOT bit-identical at B > 1: the difference is bf16-kernel variance —
#: a few elements per 10^5 moved by one bf16 rounding of the edit or of the
#: sum (microbench 388414: fraction unequal <= 3.9e-5, |diff| <= one bf16 ulp
#: of max(|output|, |edit|) — the gate's contract reads that per op CLASS since
#: 2026-09-06: in ulps of the EDIT for :data:`LINEAR_REDUCTION_OPS`, at the
#: GATE for :data:`GATED_REDUCTION_OPS`); a one-row group is one call in
#: either mode, so B=1 is exact by construction. The value is read ONCE at import into
#: :data:`_STACKED_REDUCTIONS` (the vLLM worker is spawned with its env in
#: place); the worker attests the mode in its apply path (``vectorized+
#: stacked`` / ``vectorized+per_row``) and as ``lens_reduction_mode`` in its
#: scheduler-config report. An unrecognised value REFUSES at import: a typo
#: must never pick a mode silently.
STACKED_REDUCTIONS_ENV = "SISTEER_LENS_STACKED_REDUCTIONS"
REDUCTION_MODE_STACKED = "stacked"
REDUCTION_MODE_PER_ROW = "per_row"
REDUCTION_MODES = (REDUCTION_MODE_STACKED, REDUCTION_MODE_PER_ROW)
#: the env vocabulary: unset / empty = the default (stacked)
_STACKED_ON = ("1", "true", "on", "yes", "all", REDUCTION_MODE_STACKED)
_STACKED_OFF = ("0", "false", "off", "no", REDUCTION_MODE_PER_ROW)


def parse_reduction_mode(value: Optional[str]) -> str:
    """The env's vocabulary -> a :data:`REDUCTION_MODES` member. Unset or
    empty = the default ``stacked``; ``1``/``true``/``on``/``yes``/``all``/
    ``stacked`` = stacked; ``0``/``false``/``off``/``no``/``per_row`` =
    per-row (case-insensitive, whitespace-tolerant). Anything else is an
    :class:`AdapterError`: the mode is served arithmetic, and a value nobody
    recognises must fail the worker's import, not pick a default."""
    if value is None or not value.strip():
        return REDUCTION_MODE_STACKED
    v = value.strip().lower()
    if v in _STACKED_ON:
        return REDUCTION_MODE_STACKED
    if v in _STACKED_OFF:
        return REDUCTION_MODE_PER_ROW
    raise AdapterError(
        f"{STACKED_REDUCTIONS_ENV}={value!r} is not a reduction mode: use one of "
        f"{_STACKED_ON} (stacked, the default) or {_STACKED_OFF} (per-row)")


def parse_stacked_reductions(value: Optional[str]) -> bool:
    """``parse_reduction_mode(value) == "stacked"`` (the boolean form the
    microbench and A/B driver read)."""
    return parse_reduction_mode(value) == REDUCTION_MODE_STACKED


#: read once at import — the vLLM worker process is spawned with its env in
#: place; a test flips this attribute directly (never the env at call time)
_STACKED_REDUCTIONS = parse_stacked_reductions(os.environ.get(STACKED_REDUCTIONS_ENV))


def stacked_reductions_enabled() -> bool:
    """Whether :func:`_row_reduction` issues stacked reductions (the default;
    False under the ``per_row`` opt-out)."""
    return bool(_STACKED_REDUCTIONS)


def reduction_mode() -> str:
    """The reduction mode in effect in THIS process (``stacked`` /
    ``per_row``) — what the worker attests as ``lens_reduction_mode`` and
    what the apply path names after its ``vectorized`` token."""
    return REDUCTION_MODE_STACKED if _STACKED_REDUCTIONS else REDUCTION_MODE_PER_ROW


#: The HookOp ops whose apply REDUCES over the hidden dim through
#: :func:`_row_reduction` inside a vectorized group — the ops the reduction
#: mode governs and the ONLY ops whose stacked result may differ from the
#: per-row one (bf16-kernel variance, bounded; the gate's tolerance class).
#: Every other op is elementwise at any row count (``add_vector``) or
#: prefill-only and per request by construction (the two ``prefill_last_*``
#: ops never form a group): exact in either mode. ``tests/
#: test_lens_stacked_reductions.py`` pins this set against the calls the
#: adapter actually makes, so it cannot drift from the code.
REDUCTION_OPS = frozenset((OP_RELU_GATED_VECTOR, OP_LINEAR_GATED_VECTOR, OP_AFFINE_SUBSPACE,
                           OP_LOWRANK_RELU_GATED, OP_LOWRANK_LINEAR_GATED, OP_LORA_OPROJ,
                           OP_RELU_GATED_VECTOR_PAIR))
ELEMENTWISE_OPS = frozenset((OP_ADD_VECTOR, OP_PREFILL_LAST_MATMUL, OP_PREFILL_LAST_COSINE_GATED))
#: THE TWO REDUCTION CLASSES (contract change, 2026-09-06, on the
#: ff35913 evidence; docs/lens_stacked_reductions_scoping.md
#: section 9). A GATED reduction op reduces the hidden state to a per-row
#: (per-latent) GATE SCALAR — ``h @ gate_w`` for the gated-vector pair,
#: ``h @ Aᵀ`` for the low-rank pair — that then SCALES a fixed direction.
#: Stacked-vs-per-row kernel variance perturbs that scalar by a few fp32
#: ulps of its PARTIAL SUMS ``sum_i |h_i w_i|``, independent of the gate's
#: own size: on a near-threshold ReLU gate (1e-5 .. 1e-4) that is a 2-11 %
#: change of the whole row's edit, which the per-element edit-ulp unit reads
#: as 5-23 "ulps" wherever the activation is exactly zero — so for these ops
#: the contract bounds the deviation AT THE GATE (|g_stacked - g_per_row| <=
#: C * 2^-24 * sum_i |h_i w_i|, C = 16; the worker's ``_gate_stats``) and
#: the edit-ulp unit is recorded but is not a rule. A LINEAR reduction op
#: (the LoRA GEMM pair, the LoReFT affine pair) writes the reduction's
#: output INTO the edit, so kernel variance there IS a few fp32 ulps of the
#: edit, and the per-element edit-ulp bound (<= 4) is the right unit and
#: stays. This is the ONE definition of the split; scripts/lens_vectorize_
#: gate.py mirrors the literals and tests/test_lens_stacked_reductions.py
#: pins the mirror equal.
GATED_REDUCTION_OPS = frozenset((OP_RELU_GATED_VECTOR, OP_LINEAR_GATED_VECTOR,
                                 OP_LOWRANK_RELU_GATED, OP_LOWRANK_LINEAR_GATED,
                                 # the paired write: TWO gated-vector gates per row, each a
                                 # reduction scaling a fixed direction (the same class twice)
                                 OP_RELU_GATED_VECTOR_PAIR))
LINEAR_REDUCTION_OPS = REDUCTION_OPS - GATED_REDUCTION_OPS
#: the three op classes a shadow key's op resolves to (``reduction_op_class``)
OP_CLASS_REDUCTION_LINEAR = "reduction_linear"
OP_CLASS_REDUCTION_GATED = "reduction_gated"
OP_CLASS_ELEMENTWISE = "elementwise"
OP_CLASSES = (OP_CLASS_REDUCTION_LINEAR, OP_CLASS_REDUCTION_GATED, OP_CLASS_ELEMENTWISE)


def op_reduces_in_group(op_name: str) -> bool:
    """Whether ``op_name`` (a HookOp ``op`` value) is a :data:`REDUCTION_OPS`
    member. An op name that is neither a reduction nor an elementwise op is
    an :class:`AdapterError` — a new op must be classified before anything
    reads the reduction contract for it."""
    if op_name in REDUCTION_OPS:
        return True
    if op_name in ELEMENTWISE_OPS:
        return False
    raise AdapterError(f"hook op {op_name!r} is classified neither as a reduction op "
                       f"{sorted(REDUCTION_OPS)} nor as an elementwise op {sorted(ELEMENTWISE_OPS)}")


def reduction_op_class(op_name: str) -> str:
    """The reduction-contract CLASS of ``op_name``: :data:`OP_CLASS_REDUCTION_
    GATED` for a :data:`GATED_REDUCTION_OPS` member, :data:`OP_CLASS_REDUCTION_
    LINEAR` for every other :data:`REDUCTION_OPS` member, :data:`OP_CLASS_
    ELEMENTWISE` for an :data:`ELEMENTWISE_OPS` member; any other name is an
    :class:`AdapterError` (never a default class)."""
    if op_name in GATED_REDUCTION_OPS:
        return OP_CLASS_REDUCTION_GATED
    if op_name in LINEAR_REDUCTION_OPS:
        return OP_CLASS_REDUCTION_LINEAR
    if op_name in ELEMENTWISE_OPS:
        return OP_CLASS_ELEMENTWISE
    raise AdapterError(f"hook op {op_name!r} is classified neither as a gated reduction op "
                       f"{sorted(GATED_REDUCTION_OPS)}, a linear reduction op {sorted(LINEAR_REDUCTION_OPS)}, "
                       f"nor an elementwise op {sorted(ELEMENTWISE_OPS)}")


#: RESUMED-PREFILL SAFETY — the op-NAME partition (module docstring,
#: RESUMED PREFILLS). PER_POSITION_OPS compute each row's edit from that
#: row's own hidden state (or from nothing, for the fixed vector) and write
#: it to that row alone; BROADCAST_OPS read the LAST prompt token and write
#: (or decide) for every prompt position. Every member of :data:`OPS` is in
#: exactly one of the two — ``tests/test_lens_resumed_prefill.py`` pins the
#: partition against OPS, so a new op cannot land unclassified. The op NAME is
#: one axis of the three-valued policy below; positions and phase are the other.
PER_POSITION_OPS = frozenset((OP_ADD_VECTOR, OP_RELU_GATED_VECTOR, OP_LINEAR_GATED_VECTOR,
                              OP_AFFINE_SUBSPACE, OP_LOWRANK_RELU_GATED,
                              OP_LOWRANK_LINEAR_GATED, OP_LORA_OPROJ,
                              OP_RELU_GATED_VECTOR_PAIR))
BROADCAST_OPS = frozenset((OP_PREFILL_LAST_MATMUL, OP_PREFILL_LAST_COSINE_GATED))


#: THE RESUMED-PREFILL POLICY of a cell (module docstring, RESUMED PREFILLS):
#: ``never`` (a broadcast, last-position or phase-restricted op), ``always``
#: (per-position ops at all positions in both phases), ``conditional``
#: (per-position ops in both phases with a ``response`` mask — decided per
#: request by :func:`cached_region_header_conflict`).
RESUMED_NEVER = "never"
RESUMED_ALWAYS = "always"
RESUMED_CONDITIONAL = "conditional"
RESUMED_POLICIES = (RESUMED_NEVER, RESUMED_ALWAYS, RESUMED_CONDITIONAL)


class ResumedPrefillRefusal(AdapterError):
    """A RESUMED prefill that cannot be served as steered (module docstring,
    RESUMED PREFILLS): a ``never``-policy op met a resumed
    :class:`HookContext` (a wiring defect — the worker's lattice refuses
    such a resume engine-fatally before any apply), or a
    ``conditional``-policy request's cached region carries an assistant
    header (an EXPECTED event under a stable salt). An
    :class:`AdapterError`, so the worker CONTAINS it per request — the
    request is failed, served unsteered from that forward on, and its
    response refused at the proxy — and the engine keeps serving."""


def op_name_per_position(op_name: str) -> bool:
    """The NAME-level half of the resumed-prefill partition: True for a
    :data:`PER_POSITION_OPS` member, False for a :data:`BROADCAST_OPS`
    member, an :class:`AdapterError` for any other name (never a default).
    Necessary, NOT sufficient — the op's position mode and phase decide
    too (:func:`op_resumed_prefill_ok`); this exists so the partition can
    be pinned against :data:`OPS` name by name."""
    if op_name in PER_POSITION_OPS:
        return True
    if op_name in BROADCAST_OPS:
        return False
    raise AdapterError(
        f"hook op {op_name!r} is classified neither as a per-position op "
        f"{sorted(PER_POSITION_OPS)} nor as a broadcast op {sorted(BROADCAST_OPS)} — "
        "classify it before anything reads the resumed-prefill contract for it")


def op_resumed_prefill_policy(op: "HookOp") -> str:
    """ONE op's resumed-prefill policy (module docstring, RESUMED PREFILLS):
    :data:`RESUMED_NEVER` for a broadcast op name, a ``positions="last"``
    op (the edit is placed by the prompt length) or a phase-restricted op
    (the writer's generated rows were not edited; the reader's prompt rows
    would be); :data:`RESUMED_ALWAYS` for a per-position op at
    ``positions="all"`` in both phases (``skip_bos`` either way — position 0
    is position 0 for every request sharing a prefix);
    :data:`RESUMED_CONDITIONAL` for a per-position op at
    ``positions="response"`` in both phases (safe iff the cached region is
    header-free — decided per request) and likewise for
    ``positions="response_all_turns"``, under the SAME clause set today
    (:data:`ALL_TURNS_CONFLICT_CLAUSES` — the machinery admits a narrower
    one later, once the two preconditions in the module docstring hold).
    An unclassified op name
    refuses (never a default)."""
    if not isinstance(op, HookOp):
        raise AdapterError(f"op_resumed_prefill_policy takes a HookOp, got {type(op).__name__}")
    if not op_name_per_position(op.op) or op.phase != PHASE_BOTH:
        return RESUMED_NEVER
    if op.positions == POSITIONS_ALL:
        return RESUMED_ALWAYS
    if op.positions == POSITIONS_RESPONSE:
        return RESUMED_CONDITIONAL
    if op.positions == POSITIONS_RESPONSE_ALL_TURNS:
        # CONDITIONAL under the same two clauses as "response" today
        # (ALL_TURNS_CONFLICT_CLAUSES): the mode ships as a MASK, with no
        # new cache-sharing claim. Narrowing its clause set is the future
        # lane gated on the two preconditions in the module docstring.
        return RESUMED_CONDITIONAL
    return RESUMED_NEVER


def resumed_prefill_policy(ops: Union["HookOp", Sequence["HookOp"]]) -> str:
    """THE cell-level policy the worker (serve / decide per request / refuse
    a resumed prefill) and the proxy half (which cache salt to mint per
    cell) consult, over the cell's ops: :data:`RESUMED_NEVER` if ANY op is
    never-policy (the cache would carry that op's request-dependent rows) or
    the sequence is EMPTY (nothing to vouch for — a cell with no ops is not a
    served cell); else :data:`RESUMED_CONDITIONAL` if any op is conditional;
    else :data:`RESUMED_ALWAYS`. A single HookOp is accepted as a one-op
    sequence; a non-HookOp element refuses."""
    if isinstance(ops, HookOp):
        ops = (ops,)
    ops = tuple(ops)
    if not ops:
        return RESUMED_NEVER
    policies = {op_resumed_prefill_policy(op) for op in ops}
    if RESUMED_NEVER in policies:
        return RESUMED_NEVER
    if RESUMED_CONDITIONAL in policies:
        return RESUMED_CONDITIONAL
    return RESUMED_ALWAYS


def op_resumed_prefill_ok(op: "HookOp") -> bool:
    """``op_resumed_prefill_policy(op) == RESUMED_ALWAYS`` — the plain-bool
    form (unconditionally resumable)."""
    return op_resumed_prefill_policy(op) == RESUMED_ALWAYS


def resumed_prefill_ok(ops: Union["HookOp", Sequence["HookOp"]]) -> bool:
    """``resumed_prefill_policy(ops) == RESUMED_ALWAYS`` — the plain-bool
    form kept for callers that need "resumable without a per-request
    decision"; a ``conditional`` cell reads False here and is decided per
    request by :func:`cached_region_header_conflict`."""
    return resumed_prefill_policy(ops) == RESUMED_ALWAYS


#: THE TWO CLAUSES of the cached-region check, so each header-located mode
#: names the ones it needs (:func:`conflict_clauses_for`):
#: - FULL OCCURRENCE — the cached region holds a complete assistant header
#:   anywhere: a prior writer's boundary. Under ``response`` the rows from
#:   there on were steered by that writer while this reader's tail-anchored
#:   mask leaves them alone — refuse. Under ``response_all_turns`` the
#:   reader steers that turn too, so this clause LOOKS droppable; it is
#:   NOT dropped today: it also blocks inheriting a FAILED writer's
#:   unsteered rows (a refused request is served unsteered but still
#:   cached), and the lexical span locator cannot promise the two requests
#:   masked the shared prefix identically.
#: - BOUNDARY INSIDE HEADER — the cached region ends with a proper prefix
#:   of the header. Kept by BOTH modes: the header's opening token (gemma-4
#:   ``<|turn>``) also opens user turns, so whether the writer steered it
#:   depends on the tokens that FOLLOWED it in the writer's prompt, which
#:   the reader may not share. A coincidental prefix match refuses too —
#:   conservative on purpose.
CONFLICT_FULL_OCCURRENCE = "full_occurrence"
CONFLICT_BOUNDARY_INSIDE_HEADER = "boundary_inside_header"
CONFLICT_CLAUSES = (CONFLICT_FULL_OCCURRENCE, CONFLICT_BOUNDARY_INSIDE_HEADER)
RESPONSE_CONFLICT_CLAUSES = frozenset(CONFLICT_CLAUSES)
#: the all-turns set is the FULL set TODAY — identical admission to
#: ``response``, no new cache-sharing claim. Narrowing it to
#: ``{CONFLICT_BOUNDARY_INSIDE_HEADER}`` is future work, gated on the
#: two preconditions in the module docstring; this constant is where
#: that lands.
ALL_TURNS_CONFLICT_CLAUSES = RESPONSE_CONFLICT_CLAUSES


def conflict_clauses_for(positions: str) -> FrozenSet[str]:
    """The cached-region clauses a header-located position mode checks on
    a resumed prefill: both for ``response``, and — today — both for
    ``response_all_turns`` as well (:data:`ALL_TURNS_CONFLICT_CLAUSES`;
    the per-mode indirection is what lets that set narrow later). Any
    other mode refuses — it has no header to check against."""
    if positions == POSITIONS_RESPONSE:
        return RESPONSE_CONFLICT_CLAUSES
    if positions == POSITIONS_RESPONSE_ALL_TURNS:
        return ALL_TURNS_CONFLICT_CLAUSES
    raise AdapterError(f"positions={positions!r} is not a header-located mode; it has no "
                       "cached-region clauses")


def cached_region_header_conflict(header_ids: Sequence[int],
                                  cached_ids: Sequence[int],
                                  clauses: FrozenSet[str] = RESPONSE_CONFLICT_CLAUSES,
                                  ) -> Optional[str]:
    """THE PER-REQUEST DECISION for a ``conditional`` cell (module docstring,
    RESUMED PREFILLS): ``None`` when the cached region ``cached_ids`` (the
    request's prompt ids ``[0, cached)``) violates none of the requested
    ``clauses`` (:data:`CONFLICT_CLAUSES`), else the reason, for the
    refusal:

    - :data:`CONFLICT_FULL_OCCURRENCE` — a FULL occurrence of the assistant
      header ``header_ids`` anywhere in the region (a prior writer's
      response boundary: rows from there on were steered by that writer
      and must not be inherited by a tail-anchored ``response`` mask; the
      multi-turn shape). Checked by BOTH modes today: ``response_all_turns``
      steers that turn too, but the clause also blocks inheriting a FAILED
      writer's unsteered rows, and lexical span location cannot promise
      two requests masked the shared prefix identically (docs §7).
    - :data:`CONFLICT_BOUNDARY_INSIDE_HEADER` — a PARTIAL occurrence at the
      cache boundary (``cached_ids[-k:] == header[:k]`` for some
      ``1 <= k < H``: a cache boundary INSIDE the header, whose cached half
      the writer steered while the reader's mask may not start at it, or
      vice versa — the opener is shared with user turns; a coincidental
      prefix match is refused too — conservative on purpose). Checked by
      BOTH modes.

    ``header_ids`` must be non-empty; ``clauses`` must be a non-empty subset
    of :data:`CONFLICT_CLAUSES`; an empty cached region is trivially safe.

    WHY THERE IS NO PARTIAL clause for the all-turns TURN-END SCAFFOLD
    (``turn_end_ids``, multi-token on gemma-4: ``[<turn|>, \\n]``), although
    it has the header's "shared prefix completed two ways" structure: the
    header is a span START — a completion that is not the header leaves
    the cached opener UNMASKED for one request and masked for the other —
    while the scaffold is a span END: a completion that is not the
    scaffold does not end the span, it EXTENDS it (:func:`assistant_turn_
    spans` keeps searching, or refuses), so a cached terminator that
    follows an assistant header is masked under EVERY completion. The
    mask at a position inside a partially cached scaffold therefore does
    not depend on what follows, and a boundary there is sound
    (``tests/test_lens_all_turns_mask.py`` pins it in the writer/reader
    shape). Add the clause if a span end ever becomes a
    match-or-unmask decision."""
    want = tuple(int(t) for t in header_ids)
    if not want:
        raise AdapterError("cached_region_header_conflict needs a non-empty header id sequence")
    if not clauses or any(c not in CONFLICT_CLAUSES for c in clauses):
        raise AdapterError(f"cached_region_header_conflict: clauses {sorted(clauses)} must be a "
                           f"non-empty subset of {list(CONFLICT_CLAUSES)}")
    ids = tuple(int(t) for t in cached_ids)
    H, L = len(want), len(ids)
    if CONFLICT_FULL_OCCURRENCE in clauses:
        for start in range(0, L - H + 1):
            if ids[start:start + H] == want:
                return (f"the cached region [0, {L}) contains the assistant header {list(want)} at "
                        f"position {start} — a prior writer's response boundary; its cached rows from "
                        "there on were steered under THAT boundary and cannot be inherited")
    if CONFLICT_BOUNDARY_INSIDE_HEADER in clauses:
        for k in range(min(H - 1, L), 0, -1):
            if ids[L - k:] == want[:k]:
                return (f"the cache boundary at {L} falls INSIDE the assistant header: the cached "
                        f"region ends with header ids {list(want[:k])} ({k} of {H}) — the header's "
                        "cached half was steered by its writer while this request's mask would start "
                        "at it")
    return None


def response_header_ids(ops: Sequence["HookOp"]) -> Optional[Tuple[int, ...]]:
    """The ONE header id sequence the HEADER-LOCATED ops of a cell carry
    (``positions`` in :data:`RESPONSE_MASK_MODES`; ``None`` when the cell
    has none); such ops disagreeing on their header refuse — one registry,
    one template, one header."""
    headers = {tuple(op.header_ids) for op in ops
               if isinstance(op, HookOp) and op.positions in RESPONSE_MASK_MODES}
    if not headers:
        return None
    if len(headers) != 1:
        raise AdapterError(f"the cell's response ops disagree on their assistant header: "
                           f"{sorted(headers)} — one registry serves one template; refusing")
    return headers.pop()


def resumed_conflict_clauses(ops: Sequence["HookOp"]) -> FrozenSet[str]:
    """The UNION of the cached-region clauses the cell's header-located ops
    check (:func:`conflict_clauses_for`): a cell mixing ``response`` and
    ``response_all_turns`` ops checks both clauses, because its response
    ops need the full-occurrence one. Empty for a cell with no
    header-located op."""
    out: FrozenSet[str] = frozenset()
    for op in ops:
        if isinstance(op, HookOp) and op.positions in RESPONSE_MASK_MODES:
            out = out | conflict_clauses_for(op.positions)
    return out


def resumed_header_conflict(ops: Sequence["HookOp"], cached_ids: Sequence[int]) -> Optional[str]:
    """:func:`cached_region_header_conflict` for a CELL, under the clause
    set its header-located ops need (:func:`resumed_conflict_clauses`): the
    reason the cached region conflicts, or ``None`` (also ``None`` for a
    cell with no header-located op — nothing to conflict)."""
    header = response_header_ids(ops)
    if header is None:
        return None
    return cached_region_header_conflict(header, cached_ids, resumed_conflict_clauses(ops))


def ulp_spacing(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The unit in the last place of ``dtype`` at each element's magnitude,
    as fp32: ``eps(dtype) * 2**floor(log2 |x|)`` for a normal ``|x|``, the
    smallest-normal spacing below that (``x`` may be any float tensor; it is
    read in fp32)."""
    fi = torch.finfo(dtype)
    mag = x.detach().float().abs().clamp_min(fi.tiny)
    return torch.ldexp(torch.full_like(mag, fi.eps), torch.floor(torch.log2(mag)))


def ulp_distance(a: torch.Tensor, b: torch.Tensor, at: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
    """``|a - b|`` in units of the last place of ``b``'s OWN dtype at the
    magnitude of ``at`` (default ``b`` itself), elementwise (fp32 arithmetic;
    bf16 and fp32 values are exact in fp32; :func:`ulp_spacing`) — so two
    adjacent representable values read as 1.0, and a stacked-vs-per-row
    difference can be stated in the activation's own resolution instead of an
    absolute number that depends on where in the residual stream it landed.
    Read at the output's own magnitude this INFLATES where the edit nearly
    cancels the activation (the microbench's 128-ulp tail: one rounding of a
    0.75 edit landing on a 0.004 output); the shadow compare's contract unit
    is :func:`edit_excess_ulps` instead. All inputs must share a dtype and
    shape."""
    if a.dtype != b.dtype or a.shape != b.shape:
        raise AdapterError(
            f"ulp_distance needs two tensors of one dtype and shape, got "
            f"{a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)}")
    if at is None:
        at = b
    elif at.dtype != b.dtype or at.shape != b.shape:
        raise AdapterError(
            f"ulp_distance: the magnitude tensor must match, got {at.dtype} {tuple(at.shape)} "
            f"vs {b.dtype} {tuple(b.shape)}")
    return (a.detach().float() - b.detach().float()).abs() / ulp_spacing(at, b.dtype)


#: the cap on :func:`edit_excess_ulps` (an edit stacked made where per-row made
#: none divides by the smallest-normal spacing): finite, so a record stores it;
#: a power of two, so it survives the fp32 arithmetic exactly
EDIT_ULPS_CAP = 2.0 ** 100


def edit_excess_ulps(applied: torch.Tensor, ref: torch.Tensor, base: torch.Tensor
                     ) -> torch.Tensor:
    """THE CONTRACT'S PER-ELEMENT UNIT: how far
    the STACKED edit is from the PER-ROW edit, beyond one rounding of the sum,
    in bf16 ulps of the per-row edit — elementwise,
    ``max(0, |applied - ref| - ulp(|ref|)) / ulp(|ref - base|)`` with
    ``applied`` the stacked output rows, ``ref`` the per-row output rows and
    ``base`` the UNSTEERED rows both were computed from (so ``ref - base`` is
    the per-row edit actually added to the stream and ``applied - ref`` is the
    whole difference between the two edits). Kernel variance perturbs the fp32
    edit by a few fp32 ulps, which after the two bf16 roundings moves the
    output by at most one ulp of the EDIT plus one ulp of the SUM — so this
    reads <= 1 (the mechanism's prediction; the gate allows 4). A dropped
    edit (stacked added nothing: ``applied == base``) reads ``(|edit| -
    ulp(|out|)) / ulp(|edit|)`` — >= 60 for any edit larger than two output
    ulps — a halved edit about half that, and an edit written to the wrong row
    the whole misplaced edit; none of them is a rounding, and the unit says
    so wherever the edit exceeds a couple of ulps of the output. Where the
    edit is at most ONE ulp of the output a dropped edit is indistinguishable
    per element from a flipped sum rounding (both are |out| moved by one ulp);
    the concentration bounds on unequal elements per ROW and per GROUP (the
    shadow record's ``max_frac_row`` / ``frac_group``) catch that case: a row
    whose edit was dropped differs on nearly every element, kernel variance on
    a few per 10^5. A ``ref - base`` of exactly zero (no per-row edit on that
    element) reads at the smallest-normal spacing: any excess there is huge
    (capped at :data:`EDIT_ULPS_CAP`), correctly — stacked added something
    per-row did not. All three tensors must share a dtype and shape."""
    if not (applied.dtype == ref.dtype == base.dtype) or not (applied.shape == ref.shape == base.shape):
        raise AdapterError(
            f"edit_excess_ulps needs the stacked output, the per-row output and the unsteered rows in "
            f"one dtype and shape, got {applied.dtype} {tuple(applied.shape)}, {ref.dtype} "
            f"{tuple(ref.shape)}, {base.dtype} {tuple(base.shape)}")
    a, r, b = applied.detach().float(), ref.detach().float(), base.detach().float()
    excess = ((a - r).abs() - ulp_spacing(r, ref.dtype)).clamp_min(0.0)
    # a zero per-row edit puts the smallest-normal spacing in the denominator;
    # the ratio is kept FINITE (capped) so a record stores a number, not None
    return (excess / ulp_spacing(r - b, ref.dtype)).clamp_max(EDIT_ULPS_CAP)


class GateCapture:
    """THE GATES A GATED REDUCTION OP ACTUALLY USED (2026-09-06, review P1 of
    the gated contract): while a capture is armed (:func:`gate_capture`) the
    gated branches of :func:`_apply_one` append one entry per gated op they
    apply, in op order — ``op``; ``kind`` (``'vector'`` for the gated-vector
    pair, ``'lowrank'`` for the low-rank pair); ``gate``: the gate tensor the
    edit was built from, fp32, POST-ReLU, ``[rows]`` or ``[rows, k]``;
    ``sum_abs``: ``sum_i |h_i w_i|`` per gate element, from the same
    ``hidden.float()`` the gate read; ``eff``; ``mask``: the row selection
    multiplied into the edit (None = every row); and the direction the gate
    scales — ``vector`` ``[d]`` or ``write`` ``[d, k]``. The shadow compare
    arms one capture around the GROUPED apply and one around each PER-ROW
    reference apply and compares the two OBSERVATIONS
    (lens_worker._gate_stats_observed) — the gated contract's evidence is the
    gate the candidate used, never a recomputation of it (a grouped-only
    defect that hands row 0 row 1's gate is invisible to a recomputation and
    to every concentration bound when the direction is sparse). Unarmed cost
    (the production path), per op class — no allocation and no context manager
    in either: a gated-VECTOR op does one module-global lookup and one branch;
    a LOW-RANK op does one module-global lookup and three branches in the
    stacked mode (the list guard, the latent guard inside the reduction
    closure, the record guard) and B + 2 in the per-row mode, where the closure
    runs once per row. Not zero; tests/test_lens_stacked_reductions.py counts
    the guards executably by tracing them."""
    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []


#: the armed capture, None outside :func:`gate_capture` (production: always None)
_GATE_CAPTURE: Optional[GateCapture] = None


@contextlib.contextmanager
def gate_capture(sink: GateCapture):
    """Arm ``sink`` as the gate capture for the block, restoring the previously
    armed sink afterwards — exception-safe. Entering it is the shadow compare's
    act alone: a caller that is not observing must not enter (the production
    apply path never does — lens_worker enters only under
    SISTEER_LENS_SHADOW_COMPARE=1), so anything but a :class:`GateCapture` is
    refused rather than treated as a no-op."""
    global _GATE_CAPTURE
    if not isinstance(sink, GateCapture):
        raise AdapterError("gate_capture needs a GateCapture to arm — the seam is entered only by the "
                           f"shadow compare, never as a no-op (got {type(sink).__name__})")
    prev = _GATE_CAPTURE
    _GATE_CAPTURE = sink
    try:
        yield sink
    finally:
        _GATE_CAPTURE = prev


def gated_vector_write(out: torch.Tensor, g: torch.Tensor, eff: float, vector: torch.Tensor,
                       sel: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """THE WRITE of a gated-vector op, given its gate: ``out + ((eff * g)
    .unsqueeze(-1) * vector [* sel]).to(dtype)`` — fp32 addition, the row
    selection multiplied in as the HF hook bodies do, ONE cast at the end.
    :func:`_apply_one` calls this for the gated-vector pair, and so does the
    shadow compare's write reconstruction (lens_worker._gate_stats_observed:
    from the OBSERVED gate of each side it rebuilds that side's expected output
    and requires it bit-for-bit) — one code path, so the reconstruction cannot
    drift from the apply. Everything here is elementwise: for a given gate the
    result is exact at any row count."""
    add = (eff * g).unsqueeze(-1) * vector                       # fp32
    if sel is not None:
        add = add * sel.to(add.dtype).unsqueeze(-1)
    return out + add.to(dtype)


def paired_gated_vector_write(out: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor, eff: float,
                              vector: torch.Tensor, vector2: torch.Tensor,
                              sel: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """THE WRITE of the paired op (:data:`OP_RELU_GATED_VECTOR_PAIR`), given its two
    gates — ``SharedStateWriteObject.paired_write`` + ``_make_hook`` term by term:
    ``add = (+1 g1)[..., None] vector + (-1 g2)[..., None] vector2`` in fp32 (``1.0 * g``
    is ``g`` and ``-1.0 * g`` is ``-g`` bit for bit, so the class's ``coef * g`` products
    are spelled as the sign they are), then ``out + (eff * add [* sel]).to(dtype)`` — the
    hook's ``hidden + (scale * add).to(hidden.dtype)``. :func:`_apply_one` calls this and
    so does the shadow compare's write reconstruction (``lens_worker._gate_stats_observed``)
    — one code path, so the reconstruction cannot drift from the apply."""
    add = g1.unsqueeze(-1) * vector + (-g2).unsqueeze(-1) * vector2   # fp32
    add = eff * add
    if sel is not None:
        add = add * sel.to(add.dtype).unsqueeze(-1)
    return out + add.to(dtype)


def _row_reduction(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor],
                   rowwise: bool) -> torch.Tensor:
    """``fn(x)`` — the op's REDUCTION over the hidden dim (a gate gemv,
    a LoRA read/write GEMM, the LoReFT affine-subspace ``F.linear``
    pair and its back-projection) — issued as ONE call over every row of ``x``
    when the rows belong to one request (the per-request path's own
    call), and, when ``rowwise`` (the rows are a vectorized group: one
    decode row of each of several requests), either as ONE stacked call
    over the group (the ``stacked`` mode, the default) or as one call PER
    ROW, concatenated (the ``per_row`` mode, :data:`STACKED_REDUCTIONS_ENV`
    ``=0``).

    THE TWO MODES. A stacked ``[B, K] @ [K]`` fp32 gemv is NOT bit-identical
    to the ``B`` single-row calls the per-request path makes: cuBLAS picks
    the reduction schedule from (M, N, K), so the fp32 sum comes out a few
    fp32 ulps apart, and wherever that lands on a bf16 rounding boundary of
    the edit or of the sum one output element moves by one bf16 ulp. The
    per-row mode issues the single-row call itself (same shapes, same
    strides, a fresh cast per row exactly as ``hidden.float()`` on a one-row
    slice) and is therefore bit-identical to the per-request path by
    construction — the reference the gate's shadow compare measures the
    stacked mode against. Everything around the reduction — the ReLU, the
    outer write, the scale, the cast back — is elementwise and exact at
    any row count in either mode.

    WHY STACKED IS THE DEFAULT (the numbers in
    docs/lens_stacked_reductions_scoping.md). The per-row loop costs ~26 us
    per row per reduction per layer — 1.7 ms/step at B=64 and 6.4 ms at
    B=256 for a psr gemv over 32 layers; the stacked call is one launch,
    ~50-100 us per layer at any B — so a steered decode step at B=32-64 is
    2-3.5x faster per step (hook work 3-9x lower). The deviation is bounded
    bf16-kernel variance: on the microbench (job 388414, every family's
    shapes, B in 8..256) the fraction of output elements that differ is
    0 to 3.9e-5 and every difference is one bf16 rounding of the edit or of
    the sum; the same class of perturbation that batch composition already
    injects at ``VLLM_BATCH_INVARIANT=0``. The gate holds served stacked
    arithmetic to that class (scripts/lens_vectorize_gate.py, THE REDUCTION
    CONTRACT): over a case the fraction of unequal elements is <= 1e-4, in
    any one group <= 2e-3 and in any one row <= 2e-2; for a LINEAR reduction
    op (:data:`LINEAR_REDUCTION_OPS` — the LoRA GEMM pair, the LoReFT affine
    pair) every unequal element's stacked edit is within 4 bf16 ulps of the
    per-row edit beyond one bf16 ulp of the per-row output
    (:func:`edit_excess_ulps`); for a GATED reduction op
    (:data:`GATED_REDUCTION_OPS` — the gated-vector and low-rank pairs) the
    bound is AT THE GATE, per row: |g_stacked - g_per_row| <= 16 * 2^-24 *
    sum_i |h_i w_i|, a ReLU flip allowed only when the side that fired is
    within that same bound (the edit-ulp unit is recorded there but is not a
    rule — the 2026-09-06 contract change; the gate scalar's rounding is a
    few fp32 ulps of its PARTIAL SUMS whatever the gate's size, and a
    near-threshold ReLU gate turns that into many "edit ulps" wherever the
    activation is exactly zero); exact for elementwise ops and for the
    per-row mode. B=1 stays exact for every kind: a single row is one call
    either way (identical by definition)."""
    if not rowwise or x.dim() != 2 or x.shape[0] <= 1:
        return fn(x)
    if _STACKED_REDUCTIONS:
        return fn(x)
    return torch.cat([fn(x[i:i + 1]) for i in range(x.shape[0])], dim=0)


def apply_hook_ops(hidden: torch.Tensor, ops: Sequence[HookOp],
                   scale: float, *, layer: int,
                   ctx: Optional[RowContext] = None,
                   legacy_masks: bool = False,
                   independent_rows: bool = False) -> torch.Tensor:
    """Apply every op registered at ``layer`` to ``hidden`` (the residual
    stream, ``[..., T, hidden_size]`` — ``T`` the current forward's rows
    for ONE request), multiplied by the SELECTOR scale.

    ``scale == 0.0`` returns ``hidden`` UNTOUCHED (the same tensor) — the
    ``none__lens0`` no-op leg and Phase-0 assertion 8 need exact,
    bit-identical no-op behavior, not float arithmetic that happens to
    round to zero. The same bit-identity holds when every op at this
    layer is phase-skipped or fully masked: an op outside its phase is
    NOT INVOKED (the Evals-shared steer-generated-only semantics — a
    decode-only op leaves the prefill entirely unhooked, so the prompt KV
    stays the base model's bytes; see the module docstring, and do not
    rewrite this as a scale-0 application).

    ``ctx`` carries the per-forward phase + global positions; ops with
    any phase/position/BOS dependence REFUSE loudly without one.

    Request-time contract: the ops must ALREADY live on ``hidden``'s
    device in their policy dtype (:func:`materialize_ops` at worker
    startup) — a mismatch refuses loudly rather than quietly
    re-converting per token.

    ``ctx`` may also be a :class:`DecodeRows` — the hook wiring's
    vectorized group of decode rows (one per request, same declared
    facts); it carries no positions, so the row-count cross-check
    below applies to a :class:`HookContext` only. ``legacy_masks``
    selects the dense reference mask path (:func:`_select_rows`).

    ``independent_rows`` declares that EVERY row of ``hidden`` is a
    different request's single row (a vectorized group whose ops need
    no context, so ``ctx`` is None): every reduction over the hidden
    dim then goes through :func:`_row_reduction` under the reduction
    mode (one stacked call by default; the per-request single-row call
    per row under the per-row mode) — and a :class:`DecodeRows`
    context implies it (one row per request by contract)."""
    for op in ops:
        if op_site(op) != HOOK_SITE_BLOCK:
            raise AdapterError(
                f"hook op {op.op!r} (layer {op.layer}) is an "
                f"{HOOK_SITE_OPROJ}-site op and cannot be applied at the "
                "block output — the two sites carry different tensors "
                "(o_proj input/output vs the residual stream), so this "
                "would be wrong MATH, not a tolerance. Route by "
                "ops_at_site and apply it via apply_oproj_ops from the "
                "o_proj module hook")
    if scale == 0.0:
        return hidden
    if ctx is not None:
        if hidden.dim() < 2:
            raise AdapterError(
                f"apply_hook_ops with a HookContext needs hidden of shape "
                f"[..., T, d], got {tuple(hidden.shape)}")
        if isinstance(ctx, HookContext) \
                and hidden.shape[-2] != len(ctx.positions):
            raise AdapterError(
                f"hidden carries {hidden.shape[-2]} rows but the "
                f"HookContext names {len(ctx.positions)} positions — "
                "mis-aligned forward metadata; refusing to guess rows")
    rowwise = bool(independent_rows) or isinstance(ctx, DecodeRows)
    resumed = isinstance(ctx, HookContext) and ctx.is_resumed
    out = hidden
    for op in ops:
        if op.layer != layer:
            continue
        if op_needs_context(op) and ctx is None:
            raise _needs_ctx_refusal(op, layer)
        if resumed and op_resumed_prefill_policy(op) == RESUMED_NEVER:
            raise _resumed_refusal(op, layer, ctx)
        if op.phase != PHASE_BOTH and ctx is not None \
                and ctx.phase != op.phase:
            continue  # NOT INVOKED outside its phase (never scale-0)
        out = _apply_one(op, out, hidden, float(scale), layer, ctx,
                         legacy_masks, rowwise)
    return out


def _apply_one(op: HookOp, out: torch.Tensor, hidden: torch.Tensor,
               scale: float, layer: int, ctx: Optional[RowContext],
               legacy_masks: bool = False, rowwise: bool = False
               ) -> torch.Tensor:
    """One op's contribution, mirroring its authoritative HF hook body's
    arithmetic exactly (op order and dtypes).

    Rows come from :func:`_select_rows`: ``None`` = every row (the edit
    is added unmasked — exact, :func:`_row_selection`), :data:`NO_ROWS`
    = nothing to do (``out`` returned as the SAME tensor), else a device
    bool mask multiplied in exactly as the HF hook bodies do. Every
    reduction over the hidden dim goes through :func:`_row_reduction`
    (one stacked call, or per row under the per-row mode, when ``rowwise``
    — a vectorized group)."""
    eff = scale * op.coeff
    d_hidden = hidden.shape[-1]

    if op.op == OP_ADD_VECTOR:
        vec = op.vector
        _guard_tensor(op, "vector", vec, hidden, layer=layer,
                      want_model_dtype=True)
        if vec.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: vector dim {vec.shape[-1]} != "
                f"hidden dim {d_hidden}")
        sel = _select_rows(op, ctx, hidden, legacy_masks)
        if sel is None:
            return out + eff * vec
        if sel is NO_ROWS:
            return out  # fully masked: untouched (bit-identical)
        # hooks.SteeringHandle._make_hook: addition = v * scale, then
        # hidden + addition * mask (float mask in the hidden dtype).
        addition = vec * eff
        return out + addition * sel.to(hidden.dtype).unsqueeze(-1)

    if op.op in _GATED_OPS:
        for name in ("vector", "gate_w", "gate_b"):
            _guard_tensor(op, name, getattr(op, name), hidden, layer=layer,
                          want_model_dtype=False)
        if op.vector.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: vector dim "
                f"{op.vector.shape[-1]} != hidden dim {d_hidden}")
        sel = _select_rows(op, ctx, hidden, legacy_masks)
        if sel is NO_ROWS:
            return out
        # PSRObject.gate + _make_hook, verbatim: fp32 gate on the
        # pre-steering hidden, fp32 addition, cast back once. AlphaKL's
        # clamp_gate=False mode is the same body with the ReLU removed,
        # keeping the negative-projection half on purpose.
        g = _row_reduction(hidden, lambda r: _matmul(r.float(), op.gate_w),
                           rowwise) + op.gate_b                  # [..., T]
        if op.op == OP_RELU_GATED_VECTOR:
            g = F.relu(g)
        cap = _GATE_CAPTURE   # the shadow compare's seam: one module-global lookup + one branch (vector op)
        if cap is not None:
            cap.entries.append({
                "op": op.op, "kind": "vector", "gate": g.detach().reshape(-1),
                "sum_abs": _matmul(hidden.detach().float().abs(), op.gate_w.abs()).reshape(-1),
                "eff": float(eff), "mask": (None if sel is None else sel.detach().reshape(-1)),
                "vector": op.vector})
        return gated_vector_write(out, g, eff, op.vector, sel, hidden.dtype)

    if op.op == OP_RELU_GATED_VECTOR_PAIR:
        for name in ("vector", "gate_w", "gate_b", "vector2", "gate_w2", "gate_b2"):
            _guard_tensor(op, name, getattr(op, name), hidden, layer=layer,
                          want_model_dtype=False)
        if op.vector.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: vector dim "
                f"{op.vector.shape[-1]} != hidden dim {d_hidden}")
        sel = _select_rows(op, ctx, hidden, legacy_masks)
        if sel is NO_ROWS:
            return out
        # PSRPairedWriteObject.paired_write + SharedStateWriteObject._make_hook, verbatim:
        # BOTH gates in fp32 on the SAME pre-steering hidden (never on the other term's
        # post-write state), the two terms summed in fp32 (+1 pos, -1 neg), ONE cast at
        # the end. Each gate is its own reduction under the reduction mode.
        g1 = F.relu(_row_reduction(hidden, lambda r: _matmul(r.float(), op.gate_w), rowwise)
                    + op.gate_b)                                  # [..., T]
        g2 = F.relu(_row_reduction(hidden, lambda r: _matmul(r.float(), op.gate_w2), rowwise)
                    + op.gate_b2)                                 # [..., T]
        cap = _GATE_CAPTURE   # the shadow compare's seam: ONE entry carrying BOTH gates
        if cap is not None:
            habs = hidden.detach().float().abs()
            cap.entries.append({
                "op": op.op, "kind": "pair",
                "gate": torch.stack([g1.detach().reshape(-1), g2.detach().reshape(-1)], dim=-1),   # [T, 2]
                "sum_abs": torch.stack([_matmul(habs, op.gate_w.abs()).reshape(-1),
                                        _matmul(habs, op.gate_w2.abs()).reshape(-1)], dim=-1),  # [T, 2]
                "eff": float(eff), "mask": (None if sel is None else sel.detach().reshape(-1)),
                "vector": torch.stack([op.vector, op.vector2], dim=0)})                           # [2, d]
        return paired_gated_vector_write(out, g1, g2, eff, op.vector, op.vector2, sel, hidden.dtype)

    if op.op == OP_AFFINE_SUBSPACE:
        for name in ("rotate", "affine_w", "affine_b"):
            _guard_tensor(op, name, getattr(op, name), hidden, layer=layer,
                          want_model_dtype=False)
        if op.rotate.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: subspace dim "
                f"{op.rotate.shape[-1]} != hidden dim {d_hidden}")
        sel = _select_rows(op, ctx, hidden, legacy_masks)
        if sel is NO_ROWS:
            return out
        # LoReFTObject._make_hook, verbatim: fp32 affine-minus-rotation
        # into the subspace, back out through R, one cast at the end.
        # All three reductions (the two F.linear projections over d and
        # the back-projection over rank) run inside _row_reduction — the
        # reduction mode governs a vectorized group (the stacked [B, d]
        # F.linear / [B, rank] @ R calls of a loreft_both group —
        # OP_AFFINE_SUBSPACE in BOTH phases — reduce in an M-dependent
        # order, 13 unequal bf16 elements at B=32, d=2560, rank=4: the
        # bounded class the gate tolerates); one call for one request's
        # rows; the subtraction and the cast are elementwise either way.
        def _affine(r: torch.Tensor) -> torch.Tensor:
            r32 = r.float()
            return _matmul(_linear(r32, op.affine_w, op.affine_b)
                           - _linear(r32, op.rotate), op.rotate)  # [..., T, d]
        delta = _row_reduction(hidden, _affine, rowwise)
        if sel is not None:
            delta = delta * sel.to(delta.dtype).unsqueeze(-1)
        return out + (eff * delta).to(hidden.dtype)

    if op.op in _LOWRANK_OPS:
        for name in ("lowrank_a", "lowrank_b"):
            _guard_tensor(op, name, getattr(op, name), hidden, layer=layer,
                          want_model_dtype=False)
        if op.lowrank_a.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: read basis dim "
                f"{op.lowrank_a.shape[-1]} != hidden dim {d_hidden}")
        sel = _select_rows(op, ctx, hidden, legacy_masks)
        if sel is NO_ROWS:
            return out
        # the low-rank gated object's latents + its hook, verbatim: fp32 read through
        # A, optional clamp, fp32 write through B, one cast at the end.
        # the shadow compare's seam, low-rank op: one module-global lookup + three branches in the stacked
        # mode (this list guard, the latent guard inside _lowrank once per reduction call, the record guard
        # below) — B + 2 in the per-row mode, where _lowrank runs once per row
        cap = _GATE_CAPTURE
        zs: Optional[List[torch.Tensor]] = [] if cap is not None else None   # the list exists only when armed

        def _lowrank(r: torch.Tensor) -> torch.Tensor:
            z = _matmul(r.float(), op.lowrank_a.t())           # [..., T, k]
            if op.op == OP_LOWRANK_RELU_GATED:
                z = F.relu(z)
            if zs is not None:
                zs.append(z.detach())                          # the latents the write used
            return _matmul(z, op.lowrank_b.t())
        add = eff * _row_reduction(hidden, _lowrank, rowwise)  # fp32
        if sel is not None:
            add = add * sel.to(add.dtype).unsqueeze(-1)
        if cap is not None:   # the shadow compare's gate observation (GateCapture)
            k = int(op.lowrank_a.shape[0])
            cap.entries.append({
                "op": op.op, "kind": "lowrank",
                "gate": (zs[0] if len(zs) == 1 else torch.cat(zs, dim=-2)).reshape(-1, k),
                "sum_abs": _matmul(hidden.detach().float().abs(), op.lowrank_a.abs().t()).reshape(-1, k),
                "eff": float(eff), "mask": (None if sel is None else sel.detach().reshape(-1)),
                "write": op.lowrank_b})
        return out + add.to(hidden.dtype)

    if op.op == OP_PREFILL_LAST_COSINE_GATED:
        for name in ("vector", "cond"):
            _guard_tensor(op, name, getattr(op, name), hidden, layer=layer,
                          want_model_dtype=False)
        if op.vector.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: vector dim "
                f"{op.vector.shape[-1]} != hidden dim {d_hidden}")
        # CosineGatedRObject.gate_mask + its hook, verbatim: fp32 cosine
        # against the LAST PROMPT token with the 1e-12 denominator floor,
        # STRICT '>' (a score exactly at the threshold does not fire), and
        # a non-firing request left bit-identical to unsteered. Prefill-
        # only and per request by construction (HookOp validation), so
        # the one D2H read of ``fire`` here is never on the decode path.
        h_last = hidden[..., -1, :].float()
        cosine = ((h_last @ op.cond)
                  / (h_last.norm(dim=-1) * op.cond.norm() + 1e-12))
        fire = cosine > op.threshold
        if not bool(fire.any()):
            return out
        sel = _select_rows(op, ctx, hidden, legacy_masks)
        if sel is None or sel is NO_ROWS:
            # positions is pinned to 'last' for this op, so a None
            # (all-rows) selection never arises; kept as the dense
            # builder's defensive branch, unchanged.
            return out
        delta = (eff * op.vector) * sel.to(torch.float32).unsqueeze(-1)
        fire_f = fire.to(torch.float32).unsqueeze(-1).unsqueeze(-1)
        return out + (delta * fire_f).to(hidden.dtype)

    if op.op == OP_PREFILL_LAST_MATMUL:
        M = op.matrix
        _guard_tensor(op, "matrix", M, hidden, layer=layer,
                      want_model_dtype=True)
        if M.shape[-1] != d_hidden:
            raise AdapterError(
                f"hook op at layer {layer}: matrix dim {M.shape[-1]} != "
                f"hidden dim {d_hidden}")
        # AlphaSteerObject.apply's hook, verbatim: model-dtype matmul on
        # the LAST prompt token, broadcast to every prompt position (the
        # ctx pins the complete prompt, so row -1 IS the last token).
        h_last = hidden[..., -1, :]
        steer = h_last @ M
        return out + (eff * steer).unsqueeze(-2).to(hidden.dtype)

    raise AdapterError(f"unhandled hook op {op.op!r}")  # pragma: no cover


def apply_oproj_ops(oproj_in: torch.Tensor, oproj_out: torch.Tensor,
                    ops: Sequence[HookOp], scale: float, *, layer: int,
                    ctx: Optional[RowContext] = None,
                    legacy_masks: bool = False,
                    independent_rows: bool = False) -> torch.Tensor:
    """Apply every OPROJ-SITE op registered at ``layer`` — the o_proj
    module hook's body, the sibling of :func:`apply_hook_ops` for
    HOOK_SITE_OPROJ.

    ``oproj_in`` is the o_proj module's INPUT rows (``[T, d_in]`` — the
    pre-projection attention output for ONE request's rows of the
    flattened forward) and ``oproj_out`` its OUTPUT rows (``[T,
    d_out]``); the edit is computed from the input and added to the
    output, exactly the authoritative hook bodies
    (RankOneLoRAObject.apply / MultiLayerLoRAObject._make_hook): fp32
    read through A, fp32 write through B, ONE cast at the end, and the
    position mask applied via ``torch.where`` so masked-out rows stay
    BIT-identical to the unsteered output
    (PositionMaskedFit._fit_masked).

    The contracts of :func:`apply_hook_ops` hold verbatim: ``scale ==
    0.0`` (the no-op leg) returns ``oproj_out`` as the SAME tensor;
    so does a fully-masked forward; position-dependent ops REFUSE
    without a ctx; tensors must be pre-materialized on ``oproj_out``'s
    device. A block-site op in ``ops`` refuses — wrong site, wrong
    math (see :func:`ops_at_site`)."""
    for op in ops:
        if op_site(op) != HOOK_SITE_OPROJ:
            raise AdapterError(
                f"hook op {op.op!r} (layer {op.layer}) is a block-output "
                "op and cannot be applied at the o_proj module — the two "
                "sites carry different tensors (the residual stream vs "
                "o_proj input/output), so this would be wrong MATH, not "
                "a tolerance. Route by ops_at_site and apply it via "
                "apply_hook_ops from the decoder-layer hook")
    if scale == 0.0:
        return oproj_out
    if oproj_in.dim() != 2 or oproj_out.dim() != 2 \
            or oproj_in.shape[0] != oproj_out.shape[0]:
        raise AdapterError(
            f"apply_oproj_ops needs [T, d_in] input and [T, d_out] output "
            f"rows of the same T, got {tuple(oproj_in.shape)} and "
            f"{tuple(oproj_out.shape)} — mis-paired module tensors; "
            "refusing to guess rows")
    if isinstance(ctx, HookContext) \
            and oproj_out.shape[0] != len(ctx.positions):
        raise AdapterError(
            f"o_proj output carries {oproj_out.shape[0]} rows but the "
            f"HookContext names {len(ctx.positions)} positions — "
            "mis-aligned forward metadata; refusing to guess rows")
    rowwise = bool(independent_rows) or isinstance(ctx, DecodeRows)
    resumed = isinstance(ctx, HookContext) and ctx.is_resumed
    out = oproj_out
    for op in ops:
        if op.layer != layer:
            continue
        if op_needs_context(op) and ctx is None:
            raise _needs_ctx_refusal(op, layer)
        if resumed and op_resumed_prefill_policy(op) == RESUMED_NEVER:
            raise _resumed_refusal(op, layer, ctx)
        # phase is pinned to "both" for every oproj op (HookOp validation)
        # — no phase skip can apply here; the mask is the only gate.
        for name in ("lowrank_a", "lowrank_b"):
            _guard_tensor(op, name, getattr(op, name), oproj_out,
                          layer=layer, want_model_dtype=False)
        if op.lowrank_a.shape[-1] != oproj_in.shape[-1]:
            raise AdapterError(
                f"hook op at layer {layer}: read basis dim "
                f"{op.lowrank_a.shape[-1]} != o_proj input dim "
                f"{oproj_in.shape[-1]} — the object was fit on a "
                "different architecture; refusing")
        if op.lowrank_b.shape[0] != oproj_out.shape[-1]:
            raise AdapterError(
                f"hook op at layer {layer}: write basis dim "
                f"{op.lowrank_b.shape[0]} != o_proj output dim "
                f"{oproj_out.shape[-1]} — the object was fit on a "
                "different architecture; refusing")
        eff = float(scale) * op.coeff
        # MultiLayerLoRAObject._make_hook, verbatim (rank-general path;
        # the rank-1 vectors were canonicalized to [k=1] at adapt time —
        # same reduction over d_in, same outer write over d_out):
        #   coef  = x.to(A.dtype) @ A          # fp32
        #   delta = coef @ B.T                 # fp32
        #   steered = output + (scale * delta).to(output.dtype)
        # both GEMMs through _row_reduction inside a vectorized group
        # (stacked by default; per row under the per-row mode):
        #   coef = x.to(A.dtype) @ A.t();  delta = coef @ B.t()
        a_t, b_t, a_dtype = op.lowrank_a.t(), op.lowrank_b.t(), op.lowrank_a.dtype
        delta = _row_reduction(
            oproj_in, lambda r: _matmul(_matmul(r.to(a_dtype), a_t), b_t),
            rowwise)                                           # [T, d_out]
        sel = _select_rows(op, ctx, oproj_out, legacy_masks)
        if sel is NO_ROWS:
            continue  # fully masked: untouched (bit-identical)
        steered = out + (eff * delta).to(oproj_out.dtype)
        if sel is not None:
            # PositionMaskedFit._fit_masked, verbatim: torch.where, so
            # masked-out rows are BIT-identical to the unsteered output.
            # A None selection (every row — every decode row) IS the
            # where(all-True) result, so nothing is selected against.
            steered = torch.where(sel.unsqueeze(-1), steered, out)
        out = steered
    return out
