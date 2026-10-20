"""Per-eval SERVED vLLM geometry (``max_model_len`` x ``max_num_seqs``) — the
ONE table every served launcher resolves its window from.

WHY A TABLE (the user's ruling, 2026-09-03). Every served eval had been
inheriting one geometry — Petri's 16384 x 24 — because that is what the
composer defaults to and what the certified smoke gate pins. Petri needs
that window (its measured p99 of prompt+completion is ABOVE 16k, see the
``petri`` entry); nothing else does. The ruling: each eval gets its own
default, measured to the p99 of prompt+completion on that eval's REAL
records, and that window is what buys the speed-up.

WHY A SMALLER WINDOW IS A FREE BIGGER BATCH. The composer pins the
no-chunking identity ``max_num_batched_tokens = max_num_seqs *
max_model_len`` (``vllm_server.py`` module constants: the plan lattice
refuses chunked prefills, so the scheduler's token budget must never bind
mid-prompt). Under that identity the MEMORY footprint of a launch is set by
the PRODUCT, not the factors: vLLM's profiling forward runs over the whole
budget (the 14 GiB MLP intermediate in the 32 x 16384 negative result), and
full-coverage KV is budget x bytes/token. So at the SAME batched-token
budget — :data:`BATCHED_TOKEN_BUDGET` = 24 x 16384 = 393,216, the certified
Petri footprint — a window of L tokens buys ``393,216 // L`` concurrent
sequences with an identical profiling forward and an identical KV
reservation: :func:`identity_seqs`. (The certified footprint is per H200
with the Llama-8B-class KV of the Petri gate; a launcher's readiness check
still attests the RESOLVED coverage live — this table sizes the request, it
does not replace that gate.)

THE MEASUREMENT (2026-09-03, the grading host; ``docs/served_geometry_p99.md`` is the
condensed table and the cited basis of every entry). Every eval's real
records — the served runs' own timing / usage files and the stored
generations — were tokenized with the EXACT tokenizer of each model
(multi-turn and Petri from vLLM's own usage counters), per eval x arm x
model family: prompt tokens, completion tokens, their sum's p50/p90/p99/max,
the completion cap the driver requested, and two tail fractions at a
candidate window L — ``frac_over_L`` (prompt+completion > L: the calls a
window of L would have truncated or rejected) and the CLAMP RATE
(prompt + requested max_tokens > L: the calls vLLM 400s at ``max_model_len =
L`` with the cap sent unclamped, i.e. how often the client's once-only
clamp, :mod:`served_window`, would fire).

THE SIZING RULE (:func:`sizing_problem`, asserted on construction and by
``tests/test_served_geometry.py``)::

    L = next block multiple >= p99(prompt + completion)
      UNLESS the clamp rate at that L exceeds CLAMP_RATE_CEILING (5%):
    L = next block multiple >= p99(prompt) + requested max_tokens

The clamp exists for a TAIL. If it fires on most of an arm it silently
re-parameterizes a frozen completion cap — the single-shot 4,096 cap was
itself a truncation-bias fix, and the ``none`` arms of those pools finish
near the cap on purpose, so sizing them to the total p99 would have clamped
every one of their calls. A window may round up above its floor for
headroom (the measurement rounded to 256), but no more than
:data:`SIZING_HEADROOM` tokens: past that it is no longer a p99-sized
window. ``max_num_seqs = identity_seqs(L)`` everywhere; the only S that is
not derived is Petri's certified 24, which IS ``identity_seqs(16384)``.

ENTRIES. Top-level evals (``geometry_for("capability")``) and per-task
SUB-ENTRIES (``geometry_for("capability", task="short")`` or the
``"capability:short"`` key a launcher's ``--eval`` accepts): a launcher that
serves ONE attack or only short (none / fitted) arms may take the tighter
window. The ``petri:<model>`` sub-entries are INFORMATIONAL — flagged
``requires_recertification`` and refused by :func:`resolve_geometry` —
because Petri's window is certified by the fidelity smoke (petri-gate16k)
and pinned by ``test_petri_geometry_cert``; adopting a per-model window
means moving those together, and for the qwen / long-SI cells the window
has to go UP, not down.

THE TINYCAP ENTRIES (registered provisional; the measurement was accepted
2026-09-04 with the geometry verdict). The six served
``capability:{tiny, tiny_prompted, gen_long, gen_long_thinking, agent,
agent_thinking}`` sub-entries (:data:`TINYCAP_KEYS`) are MEASURED from the
pass-1 usage LEDGERS, not from vLLM's counters: a loopback ledger
pre-tokenized every attempted /v1/chat/completions call, so a 400-rejected
call is in the basis at its true prompt length (``n_rejected``, its own
field) and every entry carries its OWN ``measured_on``. Five are measured
(``scripts/tinycap_geometry_measure.py`` pooled the six pass-1 ledgers and
applied :func:`sizing_problem`'s rule; Evals' two conditions are encoded:
a CAP-LESS call — no ``max_tokens`` — cannot clamp and is OUTSIDE the clamp
basis (``n_capless``), and the model-grader traffic on the unsteered base
route is its own INFORMATIONAL key ``capability:grader`` (``informational``,
refused by :func:`resolve_geometry` like the ``petri:<model>`` rows), never
pooled into ``gen_long``). ``capability:tiny_prompted`` stays PROVISIONAL
with an ANALYTIC window (``GeometryBasis.analytic``): it was measured from
the ~90-token safety SI only, and safety is the SHORT SI — the window is
p99_prompt(tiny) + the longest SI's gemma-4 tokens + the chat-template
delta, and it is served for measurement only. :func:`resolve_geometry`
REFUSES a provisional entry (:class:`GeometryProvisional`); the measurement
pass runs at the certified Petri footprint through :func:`pass1_geometry` —
the composer's ``PETRI_MAX_MODEL_LEN x DEFAULT_MAX_NUM_SEQS`` read from the
modules, never a literal window (``test_served_geometry`` greps every
launcher for one) — which also serves a MEASURED tinycap entry at the
footprint (``window_source`` :data:`MEASUREMENT_SOURCE`) when a pass 1 is
re-run to extend its ledger; pass 2 serves the measured window through
:func:`resolve_geometry`.

PER-MODEL BUDGETS (the user's ruling, 2026-09-03: make the per-model budget,
then measure each model's throughput on the cluster). 393,216 is
Llama-8B-CERTIFIED and UNDERSELLS smaller-KV models: live engines at
16384 x 24 report a GPU KV cache of 2.11M tokens on qwen35_4b (128 full
16k sequences), 1.16M on gemma4e4b (70x), 0.60M on llama8b (36x). So the
budget is a per-model number — :data:`BUDGET_BY_MODEL` /
:func:`budget_for` — MEASURED by ``scripts/served_decode_sweep.py --phase
budget`` (the largest S x L whose launch passes ``kv_coverage_problem`` on
one H200 at util 0.90) and folded in through the marked block below. ALL
EIGHT canonical models are measured (2026-09-04; cluster A jobs 374139 /
375170 / 375192 / 375242 / 373464 and cluster B 65057 / 65668 / 65727, the
budget probes' own ids in each basis): llama8b 458,752 (56 x 8192; the
measured, uncertified number supersedes the certified 393,216 for
``budget_for("llama8b")`` — the certified footprint stays the NO-MODEL default
:data:`BATCHED_TOKEN_BUDGET` and the per-eval table's sizing), llama3b 655,360,
gemma4e4b 589,824, gemma4e2b 786,432, qwen35_9b 589,824, qwen35_4b 720,896 (RE-MEASURED
2026-09-13 AT the MATH-500 window on the lens stack: 741,888 = 42 x 17,664, cluster B
job 100196 — the row in the paste block; the sweep's 720,896 is its history),
glm46vflash 786,432, nemotron3nano4b 786,432 (``scripts/geom_sweep_analyze.py --doc``
regenerates the per-window throughput table from the sweep results). ``budget_for`` returns them as
``source="measured"`` with no warning; a key that is not in :data:`MODEL_KEYS`
is refused, and only a future key added there without a measurement falls to the
Llama-certified default WITH a warning.

FOUR OF THOSE ROWS ARE NOW PROVISIONAL CEILINGS, NOT MEASUREMENTS (2026-09-06; ported
onto this lineage from main @ 31b869f7 on 2026-09-10 for the sdfbase-llama8b wave).
The 2026-09-04 sweep measured every budget at L=8192 on a bare engine; the
tiny-capability standups run the LENS stack at L=16384, and the two 8B/9B
dense-attention models died in engine init at their swept budgets on one H200 —
llama8b OOM'd at 458,752 (28 x 16384: 92.47 GiB of activations held after the
profile forward, the 24.50 GiB KV allocation refused) and qwen35_9b raised
``CUBLAS_STATUS_EXECUTION_FAILED`` in a bf16 GEMM at 589,824 (36 x 16384), the
same exhaustion by another symptom. gemma4e4b passes at that same 589,824, so no
Gemma row moved. :data:`_PROVISIONAL_BUDGETS` therefore SHADOWED four rows with
conservative ceilings at the certified 16384 window — llama8b 229,376 (S=14;
REPLACED 2026-09-10 by its lens-stack measurement in the paste block, 29 x 16384
= 475,136, the sdfbase-llama8b serving gate), qwen35_9b 196,608 (S=12), and,
unmeasured but at the same risk, llama3b 327,680 (S=20) and qwen35_4b 360,448
(S=22) halved — two rows since 2026-09-11: qwen35_4b's halved ceiling was LIFTED
once the AlphaSteer re-bake ladder exercised the swept 720,896 on the lens stack
(six standups at 148 x 4864, KV attested 993,378 tokens), and on 2026-09-13 the row
itself was RE-MEASURED at 17,664 on the lens stack (741,888 = 42 x 17,664; the row's
``note`` carries the whole record) — carrying
``source=``:data:`PROVISIONAL_BUDGET_SOURCE`. They size exactly like measured
rows, but ``budget_for`` warns, ``describe()``/``describe_budgets`` label them,
and every record of a launch says ``provisional-conservative``, so replacing one
with a real ``--phase budget`` measurement on this stack is a deliberate act.
llama8b's 2026-09-10 row (475,136 = 29 x 16,384, ``source="measured"``) sits in the
:data:`_MEASURED_BUDGETS` paste block with the gate's ladder in its ``note`` — 14 / 23 /
27 / 29 / 30 x 16384 pass, 31 refused (477,856 < 507,904 KV tokens), 30 excluded by choice
for the cluster's 139.80-GiB cards — and its ceiling row is gone.
:attr:`ResolvedBudget.per_model` — not ``measured`` — is what sizing keys on. ``identity_seqs(L, model_key=...)`` is
the raw quotient; ``resolve_geometry(eval, model_key=...)`` sizes S through
``ModelBudget.seqs_at`` (the hybrid cap applied), and every per-eval entry's S
stays the Llama default when no model is named (backwards compatible).

STATED LIMITATION — the budget is a TOKEN budget measured at the probe's S.
It is not purely ``S x L`` for hybrid-recurrent families (Qwen3.5's
gated-deltanet layers, Nemotron-H's Mamba2 blocks): those layers pre-allocate
a fixed recurrent state PER SLOT (~24 MiB per sequence at 9B), so
``max_num_seqs`` itself costs memory and the feasible product SHRINKS as S
grows. Measured (canary 373464, 2026-09-03): qwen35_4b's 720,896 launched as
88 x 8192 and was REFUSED as 469 x 1536 — the same product. So
``identity_seqs(L, model_key=...)`` at a much larger S than the probe's is an
upper bound for a hybrid model, not a launch guarantee; the sweep caps a
hybrid model's per-window S at 256 (``served_decode_sweep.S_MAX_HYBRID``, the
same ``layer_types`` / ``linear_attention`` detection as
``vllm_server.hybrid_recurrent_model``) and its per-window standup + retry is
the real check. Every launcher still attests the RESOLVED coverage live.

GEOMETRY CERTIFICATION AND ADAPTER PARITY ARE SEPARATE FACTS (the
coordinator's ruling D145, 2026-09-30). The serving probe's overlay record
says BOTH: per module set, the geometry MEASURED over every pair whose engines
came up and served adapters of the pair's class at every window
(``sets[<set>]``: the widest executing dim, the largest S x L), and, beside it,
whether each folded pair's adapters were shown to compute what the HF+PEFT
reference computes (``adapter_parity``: ``verified`` / ``mismatch`` /
``unverified``, per set and per pair — :data:`ADAPTER_PARITY_KEY`). The LoRA
MENU (``allowed_pairs.json``) still requires ``verified`` parity, unchanged.
Consumers split on what they SERVE: an UNSTEERED standup (no adapter, or the
zero adapter alone — the u0 views) may size from a record whose parity is
``mismatch`` or ``unverified``, because the geometry was measured on engines
that served adapters of that class whatever those adapters computed; a STEERED
standup — one that serves a trained adapter of (rank, set): the padded-with-
object launcher, a sampled view with staged objects — REFUSES that record BY
NAME (:func:`require_verified_adapter_parity`), since its arms would run
adapters whose numerics were never shown to be the fitted ones. The
record-level block aggregates the pairs (any ``mismatch`` -> ``mismatch``;
else any ``unverified`` -> ``unverified``; else ``verified``); a file without
the block is refused, and a record written without one reads ``unverified``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import serving_attestation as _sa  # stdlib-only leaf: the LoRA kernel ceiling
from . import vllm_server as vs

log = logging.getLogger(__name__)

#: vLLM's KV block size: every window must be a whole number of blocks.
VLLM_BLOCK_SIZE = 16

#: The certified Petri window — read from the composer's own default so the
#: budget below can never drift from what ``build_server_launch`` serves
#: (``test_petri_geometry_cert`` pins the smoke gate to the same default).
PETRI_MAX_MODEL_LEN: int = int(
    inspect.signature(vs.build_server_launch).parameters["max_model_len"].default)

#: THE batched-token budget every identity geometry is sized to:
#: ``DEFAULT_MAX_NUM_SEQS x 16384`` = 393,216 — the certified Petri footprint
#: (vllm_server.py's "24 x 16384 = 393,216 is the widest audit geometry that
#: fits with real slack"). Derived, not typed, so it stays coupled to the
#: composer; a test asserts the literal.
BATCHED_TOKEN_BUDGET: int = vs.DEFAULT_MAX_NUM_SEQS * PETRI_MAX_MODEL_LEN

#: The sizing rule's clamp-rate ceiling: above this share of calls the
#: once-only clamp is no longer handling a tail, so the window is sized to
#: p99(prompt) + requested max_tokens instead of p99(prompt + completion).
CLAMP_RATE_CEILING = 0.05

#: The most a window may round up above its sizing floor (tokens). The
#: measurement rounded to 256; anything further is not a p99-sized window.
SIZING_HEADROOM = 512

#: The condensed measurement table every basis cites (in-repo).
MEASUREMENT_DOC = "docs/served_geometry_p99.md"
#: The date of the ORIGINAL measurement pass (every pre-tinycap entry);
#: entries carry their own ``measured_on`` — this is not a table-wide pin.
MEASURED_ON = "2026-09-03"
#: A provisional entry's ``measured_on``: no ledger has been pooled yet.
UNMEASURED = "unmeasured"

SIZED_BY_TOTAL = "p99_total"
SIZED_BY_PROMPT_PLUS_MAX = "p99_prompt_plus_max_tokens"
#: A basis with neither p99 measured (a provisional entry) sizes nothing.
SIZED_BY_UNMEASURED = "unmeasured"
#: A PROVISIONAL entry sized by DERIVATION, not from pooled calls
#: (``GeometryBasis.analytic``): L = next block >= the derived prompt bound
#: (carried as ``p99_prompt``) + requested max_tokens — the prompt+cap
#: branch, because nothing about completions or clamps was measured.
SIZED_BY_ANALYTIC = "analytic_p99_prompt_plus_max_tokens"
#: A MEASURED entry sized to hold EVERY measured prompt at the requested cap —
#: L = next block >= ``max_prompt`` + requested max_tokens — instead of the
#: p99 (``GeometryBasis.sized_to_max_prompt``): the branch of a workload
#: whose driver REFUSES a clamped cap-bound sample by name
#: (``inspect_served.assert_pass_fits_window``, MATH-500's one pass), so the
#: once-only clamp the p99 branches leave the prompt tail to is a refusal
#: there, not a tail handler. Its clamp rate at L is 0 by construction.
SIZED_BY_MAX_PROMPT_PLUS_MAX = "max_prompt_plus_max_tokens"
#: The ``window_source`` / ``seqs_source`` of a :func:`pass1_geometry`
#: resolution of a PROVISIONAL entry: the certified Petri footprint,
#: serving the entry for its MEASUREMENT pass only.
PROVISIONAL_SOURCE = "provisional_footprint"
#: The ``window_source`` / ``seqs_source`` of a :func:`pass1_geometry`
#: resolution of a MEASURED tinycap entry (:data:`TINYCAP_KEYS`): the same
#: certified footprint, served to EXTEND the entry's ledger (a pass 1 re-run
#: — e.g. the fitted cells' generation tasks, which the accepted pools did
#: not yet contain), never the entry's own measured window.
MEASUREMENT_SOURCE = "measurement_footprint"


def next_block(n: int, block: int = VLLM_BLOCK_SIZE) -> int:
    """The smallest multiple of ``block`` that is >= ``n``."""
    return -(-int(n) // block) * block


#: the grid the table's windows sit on (every TABLE entry's max_model_len is a multiple of it); a DERIVED window is
#: aligned up to it so it reads like a table row (a multiple of the vLLM block by construction)
WINDOW_ALIGN = 128


def align_window(n: int, align: int = WINDOW_ALIGN) -> int:
    """The smallest multiple of ``align`` that is >= ``n`` (``align`` itself a multiple of the vLLM block)."""
    if align < VLLM_BLOCK_SIZE or align % VLLM_BLOCK_SIZE:
        raise ValueError(f"align {align} is not a positive multiple of the vLLM block size {VLLM_BLOCK_SIZE}")
    return -(-int(n) // align) * align


def derived_window(eval_name: str, cap: int, *, from_field: str = "p99_prompt",
                   align: int = WINDOW_ALIGN) -> int:
    """A serving window DERIVED from an eval entry's measured basis and a RUN'S completion cap (2026-09-10: the run's
    caps moved the multi-turn target reply cap from 512 to 4,096, so the table's
    8,704 ``multiturn_attacks`` window — sized to the measured p99 TOTAL under a 2,048 request cap — can no longer
    carry the leg's p99 prompt plus its cap): ``align_window(next_block(basis.<from_field> + cap))`` — the table's
    own prompt-plus-max sizing rule (``GeometryBasis.floor``'s second branch) at the run's cap, on the table's grid.
    The concurrency follows from the model's budget at the window (``resolve_geometry`` / ``identity_seqs``), so a new
    cap or a re-measured basis moves everything from here. ``from_field`` is ``p99_prompt`` (a leg's first pass) or
    ``max_total`` (the over-window RE-RUN pass: the measured maximum call, prompt + completion, bounds the largest
    prompt the first pass refused). Never a literal in a launcher."""
    b = geometry_for(eval_name).basis
    base = getattr(b, from_field, None)
    if base is None:
        raise ValueError(f"{eval_name}: basis has no {from_field} to derive a window from")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"cap must be a positive int, got {cap!r}")
    return align_window(next_block(int(base) + cap), align)


def pin_window(pin: Optional[int], derived: int, *, what: str, align: int = WINDOW_ALIGN) -> int:
    """A serving window PINNED by the run over the DERIVED one (2026-09-10: gemma4e4b / Crescendo was MEASURED at a
    4,096 target cap and the realized full-conversation context reached p99 9,950 / max 16,428 and was still rising
    over 1,784 calls — past the table's p99 rendered prompt
    7,496 + cap the derivation sizes to; Crescendo's median is 5 target forwards but its tail runs to 20, and
    multi-turn is FLAT, so an undersized window costs re-runs and terminal typed ``PromptOverWindow`` errors with no
    rung to recover on). ``None`` -> ``derived`` unchanged (every caller without a pin). A pin must be a positive int
    on the table's grid and NEVER below the derived minimum: a pin may only WIDEN (the derived value is the measured
    basis + the run's cap, the least that carries the leg's measured prompts with their whole cap). S follows from
    the model's budget at the pinned L exactly as at a derived one (``resolve_geometry`` / ``identity_seqs``)."""
    if pin is None:
        return derived
    if isinstance(pin, bool) or not isinstance(pin, int) or pin < 1:
        raise ValueError(f"{what}: pinned window must be a positive int, got {pin!r}")
    if pin % align:
        raise ValueError(f"{what}: pinned window {pin} is not on the table's {align}-token grid")
    if pin < derived:
        raise ValueError(f"{what}: pinned window {pin} is below the derived minimum {derived} (the measured basis + the "
                         "run's cap) — a pin may only widen the window, never shrink it under the leg's measured prompts")
    return pin


def _multiturn_window_for_cap(cap: int, pin: Optional[int] = None) -> int:
    """The multi-turn leg's serving window for a run whose target-reply cap is ``cap``
    (the cap face of :func:`multiturn_window`): the table's p99 rendered
    prompt (7,496 measured calls, chat template included) + cap, on the grid — or ``pin`` when the run PINS a wider
    one (:func:`pin_window`: refused below that derived minimum). S = the model's budget // L either way."""
    return pin_window(pin, derived_window("multiturn_attacks", cap, from_field="p99_prompt"), what="multiturn_window")


def multiturn_rerun_window(cap: int, pin: Optional[int] = None, *, window: Optional[int] = None) -> int:
    """The over-window RE-RUN pass's window: the table's measured MAXIMUM multi-turn
    call (12,657, prompt + completion) + cap, so every goal the first pass refused fits with its whole cap; a goal
    still over it is terminal ``error:PromptOverWindow``. At cap 4,096 this is above 16,384 (12,657 + 4,096 = 16,753).
    ``pin`` widens it (:func:`pin_window`); with ``window`` (the first pass's L) a re-run window that is not WIDER
    than the pass it re-runs is refused — it could carry no goal that pass refused."""
    Lr = pin_window(pin, derived_window("multiturn_attacks", cap, from_field="max_total"), what="multiturn_rerun_window")
    if window is not None and Lr <= int(window):
        raise ValueError(f"multiturn_rerun_window: the re-run window {Lr} is not wider than the first pass's {window} — it "
                         "could carry no goal that pass refused (PromptOverWindow); pin it wider")
    return Lr


def multiturn_geometry(cap: int, *, model_key: str, window: Optional[int] = None,
                       rerun_window: Optional[int] = None) -> dict:
    """The multi-turn leg's whole serving record for ``model_key`` at ``cap``: the first-pass window and the re-run
    window (each PINNED when given, else DERIVED — :func:`multiturn_window` / :func:`multiturn_rerun_window`, the
    same calls the attacks sbatch and the re-run pass make), S at each through the model's measured budget
    (``budget_for(model_key).entry.seqs_at``), and the derived minimum beside each so the record says what a pin
    bought. Refuses a window the budget cannot hold even once (``budget // L < 1``: ``seqs_at`` floors to 1 and would
    hide it). Keys: cap, model_key, budget, budget_source, window, S, window_source, derived_window, rerun_window,
    rerun_S, rerun_window_source, derived_rerun_window."""
    rb = budget_for(model_key)
    dL = derived_window("multiturn_attacks", cap, from_field="p99_prompt")
    dLr = derived_window("multiturn_attacks", cap, from_field="max_total")
    L = multiturn_window(cap, window)
    Lr = multiturn_rerun_window(cap, rerun_window, window=L)
    out = {"cap": int(cap), "model_key": model_key, "budget": rb.batched_token_budget, "budget_source": rb.source}
    for name, s_key, Lx, dLx, pinned in (("window", "S", L, dL, window is not None),
                                         ("rerun_window", "rerun_S", Lr, dLr, rerun_window is not None)):
        if rb.batched_token_budget // Lx < 1:
            raise ValueError(f"multiturn_geometry: {name} {Lx} exceeds the {model_key} batched-token budget "
                             f"{rb.batched_token_budget:,} — S = budget // L would be 0; not a servable window")
        out[name] = Lx
        out[s_key] = rb.entry.seqs_at(Lx)
        out[name + "_source"] = "pinned" if pinned else "derived"
        out["derived_" + name] = dLx
    return out


def identity_seqs(max_model_len: int, budget: Optional[int] = None, *,
                  model_key: Optional[str] = None) -> int:
    """The concurrency a window buys at the SAME batched-token budget under the
    no-chunking identity: ``budget // max_model_len``. This is the "free
    bigger batch" of the ruling — same profiling forward, same KV reservation,
    more sequences in flight.

    The budget is, in order: an explicit ``budget``; else the MODEL's
    (:func:`budget_for` — measured, or the Llama-certified default with a
    warning when ``model_key`` names an unmeasured model); with neither, the
    Llama-certified :data:`BATCHED_TOKEN_BUDGET` — so every existing call
    (``identity_seqs(L)``) is unchanged."""
    if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) \
            or max_model_len < 1:
        raise ValueError(f"max_model_len must be a positive int, got {max_model_len!r}")
    if budget is None:
        budget = budget_for(model_key).batched_token_budget
    return max(1, int(budget) // int(max_model_len))


# --------------------------------------------------------------------------- #
# Per-model batched-token budgets                                              #
# --------------------------------------------------------------------------- #

#: The repo's canonical model keys — the fitted-object bank layout
#: (``$SHARED/exp/objects/scale-12/<key>``), the served capability lane's
#: ``MODELS`` (scripts/modal/capctrl_served.py) and the per-model
#: ``configs/two_si_*.yaml``. ``budget_for`` refuses any other key (a typo
#: must never resolve to a silent default).
MODEL_KEYS: Tuple[str, ...] = (
    "llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b",
    "glm46vflash", "nemotron3nano4b",
)
#: The paper-companion-repo ONBOARDED models (2026-09-29): the front door's
#: models beyond the sweep's eight — today the worked example qwen35_4b_demo
#: (configured, not run; it serves the sweep's Qwen3.5-4B checkpoint under its
#: own key) — keyed by their SERVED checkpoints' registry model_ids
#: (reproduce/models.yaml). Deliberately NOT appended to :data:`MODEL_KEYS` —
#: that tuple is the 2026-09-04 sweep's table (its array index, the tinycap
#: contracts' per-key rows and the served-decode sweep's MODELS all pin it
#: exactly) — so an onboarded key lives in its OWN tables
#: (:data:`ONBOARDED_MODEL_HF_IDS`, :data:`ONBOARDED_HYBRID_MODEL_KEYS`,
#: :data:`ONBOARDED_BUDGETS`) and :func:`rows_for` / :func:`budget_for` /
#: :func:`model_key_for` serve BOTH kinds from one lookup.
ONBOARDED_MODEL_KEYS: Tuple[str, ...] = ("qwen35_4b_demo",)
#: Every key a launch may size from: the sweep's eight + the onboarded ones.
KNOWN_MODEL_KEYS: Tuple[str, ...] = MODEL_KEYS + ONBOARDED_MODEL_KEYS

#: ``ResolvedBudget.source`` values.
MEASURED_BUDGET_SOURCE = "measured"
DEFAULT_BUDGET_SOURCE = "default_llama_certified"
#: A row whose number was NOT measured: a conservative CEILING chosen from an
#: OBSERVED failure at the larger number (the 2026-09-06 standup OOMs — see the
#: ``_PROVISIONAL_BUDGETS`` block and ``docs/served_geometry_p99.md``). It sizes
#: exactly like a measured row (the model's own entry, through its ``seqs_at``),
#: and every place that names a source says plainly that it is provisional, so a
#: later ``--phase budget`` measurement replaces it deliberately instead of a
#: guess quietly hardening into the record.
PROVISIONAL_BUDGET_SOURCE = "provisional-conservative"
#: The sources that mean "the MODEL's OWN row sized this", as opposed to the
#: Llama-certified default. Every sizing site keys on this
#: (:attr:`ResolvedBudget.per_model`), never on ``measured`` alone: a
#: provisional row must size from ITSELF, never fall back to a default that is
#: LARGER than the ceiling it was chosen to respect.
PER_MODEL_BUDGET_SOURCES: Tuple[str, ...] = (MEASURED_BUDGET_SOURCE,
                                             PROVISIONAL_BUDGET_SOURCE)

#: HYBRID-RECURRENT families (Qwen3.5 gated-deltanet, Nemotron-H Mamba2 —
#: ``vllm_server.hybrid_recurrent_model``: ``layer_types`` containing
#: ``linear_attention``). Their layers pre-allocate a fixed recurrent state PER
#: SLOT, so the token budget measured at the probe's S does not buy ``budget //
#: L`` slots at a short window: qwen35_4b's 720,896 launched as 88 x 8192 and
#: was REFUSED as 469 x 1536 (canary 373464). :meth:`ModelBudget.seqs_at` caps
#: these at :data:`HYBRID_SEQS_CAP`; ``served_decode_sweep.S_MAX_HYBRID`` is the
#: same number (a test pins them equal) and the sweep's per-window standup +
#: retry is the live check.
HYBRID_MODEL_KEYS: Tuple[str, ...] = ("qwen35_9b", "qwen35_4b", "nemotron3nano4b")
#: The onboarded keys that are hybrid-recurrent too (a per-slot recurrent
#: state like the sweep's Qwen3.5 / Nemotron-H hybrids, so the same cap).
#: qwen35_4b_demo serves the Qwen3.5-4B hybrid. An onboarded hybrid is listed HERE, never
#: appended to :data:`HYBRID_MODEL_KEYS` (the sweep's pinned tuple).
ONBOARDED_HYBRID_MODEL_KEYS: Tuple[str, ...] = ("qwen35_4b_demo",)
HYBRID_SEQS_CAP = 256


def _is_hybrid(model_key: str) -> bool:
    """Hybrid-recurrent, whichever table the key lives in (the ONE place the
    two hybrid tuples are joined; :attr:`ModelBudget.hybrid` and
    :func:`rows_for` both read it)."""
    return model_key in HYBRID_MODEL_KEYS or model_key in ONBOARDED_HYBRID_MODEL_KEYS


#: the public spelling of :func:`_is_hybrid` for the wrappers (common.sh f22_model_facts once read HYBRID_MODEL_KEYS alone and
#: reported `"hybrid": false` for an onboarded hybrid, 2026-10-02; it reads this instead)
is_hybrid = _is_hybrid


@dataclass(frozen=True)
class WindowSeqsCeiling:
    """A MEASURED per-window ceiling on S for a hybrid-recurrent model — the
    allocator's own max-concurrency verdict at readiness for ONE (model, L),
    from a launch that was REFUSED by ``vllm_server.kv_coverage_problem`` at the
    budget's quotient (:data:`HYBRID_WINDOW_SEQS_CEILINGS`).

    WHY a per-window number: a hybrid cache's token capacity is NOT ``S x L``.
    vLLM equalises page sizes (attention block 976 on Nemotron-3-Nano so a page
    holds one Mamba state), pads the layer groups to a uniform size (3 padding
    layers there), and charges a request ``n_mamba_groups`` state blocks PLUS
    ``ceil(L / block)`` attention blocks. With a fixed block pool the per-request
    cost is ``6 + ceil(L / 976)`` blocks on nemotron3nano4b, so the token
    capacity ``blocks x L / (6 + ceil(L / 976))`` GROWS with L and the budget
    measured at 8192 (786,432 = 96 x 8192; the engine reported 993,416 tokens =
    121.27x there) over-buys at a SHORTER window: at 4608 the quotient 170
    (783,360 tokens) exceeded the engine's 167.55x (772,049 tokens) and the
    no-chunking invariant refused readiness (gsm8k-paired200-paper-objects
    overlap leg, job 483254_2). The three engine reports agree on the pool:
    4608 -> 167.55 x 11 = 1,843 blocks; 8192 -> 121.27 x 15 = 1,819; 10752 ->
    101.72 x 18 = 1,831.

    The ceiling is applied INSIDE :meth:`ModelBudget.seqs_at` (after the
    quotient and :data:`HYBRID_SEQS_CAP`), so every sizing site — the standup,
    the dry run, the composed identity — derives the same S and readiness still
    holds the resolved state to it exactly: this is a smaller composed
    geometry, never a runtime clamp. It lowers ONE window only; the budget row
    (and every other window's S, e.g. 73 at 10752 already served) is untouched."""

    model_key: str
    max_model_len: int
    max_num_seqs: int
    #: the allocator's verdict the ceiling is read from (``kv_max_concurrency``
    #: in the worker report; "Maximum concurrency for L tokens per request")
    kv_max_concurrency: float
    kv_tokens_reported: int
    gpu_memory_utilization: float
    probe_job_id: str
    measured_on: str
    note: str
    #: the serving configuration the verdict was taken under: a ceiling is comparable only to a budget row at the same
    #: gpu_memory_utilization (checked at import against the TABLE row), on one GPU — this campaign serves every hybrid TP = 1, and a
    #: TP > 1 verdict is refused by name rather than clamping silently; the path is recorded, never compared (the KV allocator is the same)
    tensor_parallel: int = 1
    serving_path: str = "lens"
    #: the batched-token budget (the S x L no-chunking PROFILE) the launch was composed from: the allocator's verdict is a function of
    #: that profile (what the profiling batch left for KV), so the ceiling shadows ONLY a budget row sized from the same budget — the table
    #: row it was checked against — never a measured overlay / fold row at another budget
    batched_token_budget: int = 0
    #: the engine the verdict came from (recorded for the row's provenance; a re-probe under another engine refreshes the SITE row, which
    #: this ceiling never shadows)
    engine_version: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.batched_token_budget, bool) or not isinstance(self.batched_token_budget, int) or self.batched_token_budget <= 0:
            raise ValueError(f"WindowSeqsCeiling {self.model_key}@{self.max_model_len}: name the batched_token_budget the refused launch was "
                             "composed from (the verdict is a function of that profile)")
        if self.tensor_parallel != 1:
            raise ValueError(f"WindowSeqsCeiling {self.model_key}@{self.max_model_len}: tensor_parallel={self.tensor_parallel} — this campaign "
                             "serves every hybrid TP = 1 and a verdict under another TP is not comparable; record it as its own row kind, never here")
        if not _is_hybrid(self.model_key):                 # BOTH hybrid tuples (an onboarded hybrid included; X7)
            raise ValueError(f"WindowSeqsCeiling: {self.model_key!r} is not a hybrid model key "
                             f"({list(HYBRID_MODEL_KEYS) + list(ONBOARDED_HYBRID_MODEL_KEYS)}) — a dense cache's capacity IS S x L")
        if self.max_model_len % VLLM_BLOCK_SIZE:
            raise ValueError(f"{self.model_key}@{self.max_model_len}: the window is not a multiple of "
                             f"the vLLM block size {VLLM_BLOCK_SIZE}")
        if isinstance(self.max_num_seqs, bool) or not isinstance(self.max_num_seqs, int) or self.max_num_seqs < 1:
            raise ValueError(f"{self.model_key}@{self.max_model_len}: max_num_seqs must be a positive int")
        if not (self.max_num_seqs <= self.kv_max_concurrency < self.max_num_seqs + 1):
            raise ValueError(
                f"{self.model_key}@{self.max_model_len}: the ceiling {self.max_num_seqs} is not "
                f"floor({self.kv_max_concurrency}) — paste the allocator's verdict and its floor from the same "
                "refusal")
        if self.max_num_seqs * self.max_model_len > self.kv_tokens_reported:
            raise ValueError(
                f"{self.model_key}@{self.max_model_len}: {self.max_num_seqs} x {self.max_model_len} = "
                f"{self.max_num_seqs * self.max_model_len:,} exceeds the KV the engine reported "
                f"({self.kv_tokens_reported:,} tokens)")


#: (model_key, max_model_len) -> the measured ceiling. ONE entry today:
#: nemotron3nano4b at the GSM8K overlap leg's ``capability:gen_long`` window
#: 4608, from the refused standup's own numbers (util 0.90, one H200, the lens
#: stack, vLLM 0.25.1; "Available KV cache memory: 27.46 GiB", "GPU KV cache
#: size: 772,049 tokens", "Maximum concurrency for 4,608 tokens per request:
#: 167.55x"). 167 x 4608 = 769,536 <= 772,049. The other GSM8K windows need
#: none: at 10752 the quotient 73 sits under the engine's 101.72x (rows 27 / 28
#: passed readiness), and at 18944 the quotient 41 sits under a predicted ~70x
#: (26 blocks per request). A ceiling must be BELOW the quotient it shadows
#: (:func:`_check_window_ceilings` refuses at import otherwise): once a budget
#: re-measurement makes it moot, drop the row.
#: the ``basis.source`` spelling of a budget row built from the site geometry overlay (``site_record:<generation>``; rows_for below)
SITE_RECORD_SOURCE_PREFIX = "site_record:"
HYBRID_WINDOW_SEQS_CEILINGS: Dict[Tuple[str, int], WindowSeqsCeiling] = {
    ("nemotron3nano4b", 4608): WindowSeqsCeiling(
        model_key="nemotron3nano4b", max_model_len=4608, max_num_seqs=167,
        kv_max_concurrency=167.55, kv_tokens_reported=772_049, gpu_memory_utilization=0.9,
        probe_job_id="483254_2", measured_on="2026-09-11", tensor_parallel=1, serving_path="lens", batched_token_budget=786_432, engine_version="vllm 0.25.1",
        note=("gsm8k-paired200-paper-objects overlap_off leg (capability:gen_long, thinking OFF, cap 512) refused at "
              "readiness: hybrid KV cache (6 Mamba groups, block 4608 + 4 x FullAttentionSpec block 976; 3 padding "
              "layers) supports only 167.55-way concurrency at max_model_len=4608 (~772,049 tokens) < the "
              "budget quotient 170; node-25, cluster A"),
    ),
}

#: canonical key -> the HF id the two-SI configs pin (``configs/two_si_*.yaml``
#: ``model:``; the sweep's ``served_decode_sweep.MODELS`` table pins the same
#: ids). :func:`model_key_for` is how a launcher that knows only its config's
#: model id (the families waves' ``WAVE_MODEL``, a standup's config) finds the
#: budget to size from.
MODEL_HF_IDS: Dict[str, str] = {
    "llama8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "gemma4e4b": "google/gemma-4-E4B-it",
    "gemma4e2b": "google/gemma-4-E2B-it",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "glm46vflash": "zai-org/GLM-4.6V-Flash",
    "nemotron3nano4b": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
}
#: onboarded key -> the SERVED checkpoint's HF id (reproduce/models.yaml
#: serve.checkpoint.hf_repo); :func:`model_key_for` consults both maps.
ONBOARDED_MODEL_HF_IDS: Dict[str, str] = {
    "qwen35_4b_demo": "Qwen/Qwen3.5-4B",
}


def model_key_for(model_id: Optional[str]) -> Optional[str]:
    """The canonical model key for an HF id (``org/name``, case-insensitive) or
    a staged snapshot directory whose last component is the HF-cache mangled
    name (``.../google--gemma-4-E2B-it``). ``None`` — with a WARNING — for an id
    outside :data:`MODEL_HF_IDS`: the launcher then sizes from the
    Llama-certified default, which undersells every measured model."""
    if not model_id:
        return None
    wanted = str(model_id).strip().rstrip("/")
    tail = wanted.rsplit("/", 1)[-1]
    # the sweep's eight first, then the onboarded models (the ONE registry, _ROWS)
    for key, row in _ROWS.items():
        hf = row.hf_id
        if wanted.lower() == hf.lower() or tail.lower() == hf.replace("/", "--").lower():
            return key
    log.warning("[geometry] model id %r matches no canonical model key (%s) — no measured "
                "budget; sizing falls to the Llama-certified default %s",
                model_id, ", ".join(KNOWN_MODEL_KEYS), f"{BATCHED_TOKEN_BUDGET:,}")
    return None


class UnknownModelKey(KeyError):
    """``budget_for`` / ``rows_for`` was asked for a model key outside
    :data:`KNOWN_MODEL_KEYS` (the sweep's eight + the onboarded ones)."""


@dataclass(frozen=True)
class BudgetBasis:
    """How a model's budget was MEASURED: the largest ``S x L`` whose launch
    passed ``vllm_server.kv_coverage_problem`` at readiness on ONE H200 at the
    stated ``gpu_memory_utilization`` — the composer's no-chunking identity, so
    the profiling forward ran over that whole product and the worker's attested
    KV facts covered it. ``kv_tokens_reported`` is the attested KV capacity at
    that launch (uniform cache: ``num_gpu_blocks x block_size``; hybrid: the
    allocator's max-concurrency x L). ``first_refused`` is the smallest probed
    ``S x L`` that did NOT launch — the wall — when one was probed."""

    method: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    kv_tokens_reported: Optional[int]
    probe_job_id: Optional[str]
    measured_on: str
    source: str
    first_refused: Optional[int] = None
    note: str = ""


@dataclass(frozen=True)
class ModelBudget:
    """One model's batched-token budget (``max_num_seqs x max_model_len`` under
    the no-chunking identity) and the measurement behind it. ``certified``: the
    number a serving gate certifies (Llama-8B's Petri footprint); a measured
    but uncertified entry is the sweep's bracket, not a gate's verdict.

    STATED LIMITATION: the token budget was measured at the PROBE's S
    (``basis.max_num_seqs`` at ``basis.max_model_len``). For a hybrid-recurrent
    family (Qwen3.5 gated-deltanet, Nemotron-H Mamba2) the per-slot recurrent
    state dominates at much larger S, so the same product does NOT launch as
    ``budget // L`` sequences at a short window (qwen35_4b: 88 x 8192 launched,
    469 x 1536 refused, canary 373464). ``budget // L`` is therefore only an
    upper bound there, so hybrid families (:data:`HYBRID_MODEL_KEYS`: qwen35_9b,
    qwen35_4b, nemotron3nano4b) ADDITIONALLY cap S at :data:`HYBRID_SEQS_CAP` =
    256 — ``seqs_at(L)`` applies it (and ``resolve_geometry(model_key=...)``
    sizes through ``seqs_at``); the sweep's per-window standup (the same cap as
    ``served_decode_sweep.S_MAX_HYBRID``, retried at smaller S) is the live
    check, and a launcher's readiness gate attests the resolved coverage.

    ``source`` is :data:`MEASURED_BUDGET_SOURCE` for a swept row and
    :data:`PROVISIONAL_BUDGET_SOURCE` for a conservative ceiling chosen from an
    observed FAILURE rather than a passing probe — for a provisional row
    ``basis.max_num_seqs x basis.max_model_len`` is the geometry the ceiling
    ASSERTS, not a launch anyone watched succeed, and ``basis.first_refused``
    is the product that actually failed."""

    model_key: str
    batched_token_budget: int
    basis: BudgetBasis
    certified: bool = False
    source: str = MEASURED_BUDGET_SOURCE
    #: an INSTANCE-level per-window minimum on S (window -> S), applied inside :meth:`seqs_at` after the quotient, the hybrid cap and
    #: the table ceiling: the unsteered site default carries the least admitted S over the record's certified sets at each PROBED
    #: window (2026-09-30); None for every table row
    window_seqs: Optional[Mapping[int, int]] = None
    #: SITE-OVERLAY PROVENANCE: the result generation of the site record this row was PARSED from — stamped by the
    #: overlay loader on every budget it reads (a named set's row keeps the probe's own `basis.source`; only the unsteered default spells
    #: `site_record:<generation>` there, so a source-string prefix identified half the rows). None for every table row and every row a
    #: producer constructs; :meth:`window_ceiling` exempts a stamped row (the probe's own measurement on this stack), and :meth:`as_dict`
    #: drops it so a written record's format is unchanged.
    site_record_generation: Optional[str] = None
    #: how `window_seqs` was derived (2026-10-02): ``record-wide`` — the least admitted S over EVERY launch at the window (the pre-class rule,
    #: and the rule for a record whose launches carry no class), or ``launch_class:<rank cap>x<adapters>`` — the least admitted S over the
    #: launches whose adapter class DOMINATES the served one, at the tightest dominating footprint. `window_seqs_launches` names the launches
    #: that set each window's S. Read-side provenance like site_record_generation (dropped by :meth:`as_dict`).
    window_seqs_source: Optional[str] = None
    window_seqs_launches: Optional[Mapping[int, Tuple[str, ...]]] = None

    def __post_init__(self) -> None:
        if self.model_key not in KNOWN_MODEL_KEYS:
            raise ValueError(f"ModelBudget: unknown model key {self.model_key!r}; "
                             f"known: {list(KNOWN_MODEL_KEYS)}")
        if self.source not in PER_MODEL_BUDGET_SOURCES:
            raise ValueError(f"{self.model_key}: source must be one of "
                             f"{list(PER_MODEL_BUDGET_SOURCES)}, got {self.source!r} "
                             f"({DEFAULT_BUDGET_SOURCE} is not an entry — it is the "
                             "absence of one)")
        if self.source == PROVISIONAL_BUDGET_SOURCE and self.certified:
            raise ValueError(f"{self.model_key}: a provisional ceiling cannot be certified")
        b = self.batched_token_budget
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"{self.model_key}: batched_token_budget must be a positive "
                             f"int, got {b!r}")
        probed = self.basis.max_num_seqs * self.basis.max_model_len
        if probed != b:
            raise ValueError(
                f"{self.model_key}: batched_token_budget {b:,} != the basis probe "
                f"{self.basis.max_num_seqs} x {self.basis.max_model_len} = {probed:,} — "
                "the budget IS the passing probe's product; paste both from the same row")
        if self.basis.max_model_len % VLLM_BLOCK_SIZE:
            raise ValueError(f"{self.model_key}: probed max_model_len {self.basis.max_model_len} "
                             f"is not a multiple of the vLLM block size {VLLM_BLOCK_SIZE}")

    @property
    def hybrid(self) -> bool:
        return _is_hybrid(self.model_key)

    @property
    def provisional(self) -> bool:
        """This row is a conservative ceiling awaiting a measurement, not a
        swept number (:data:`PROVISIONAL_BUDGET_SOURCE`)."""
        return self.source == PROVISIONAL_BUDGET_SOURCE

    @property
    def seqs_cap(self) -> Optional[int]:
        """The per-slot-state cap on S: :data:`HYBRID_SEQS_CAP` for a hybrid
        family, ``None`` (no cap beyond the budget) otherwise."""
        return HYBRID_SEQS_CAP if self.hybrid else None

    def window_ceiling(self, max_model_len: int, *, budget: Optional[int] = None) -> Optional[WindowSeqsCeiling]:
        """The measured per-window ceiling on S at ``max_model_len`` for a
        hybrid model (:data:`HYBRID_WINDOW_SEQS_CEILINGS`), or ``None``. A
        ceiling shadows ONLY a row sized from the budget (the S x L
        no-chunking profile) and util the refused launch was composed from —
        the table row it was checked against (a provisional row, or one
        measured at another window: nemotron3nano4b's 8192 row at 4608). A
        measured overlay or fold row at another budget (the FD 24 probe's own
        measurement of this model on this stack, under its own profile) is not
        shadowed: the allocator's verdict is a function of the profile and does
        not transfer (the reds measured at 49,152).
        Two more exemptions: an independently
        MEASURED site row — any row the overlay loader parsed from a site
        record, named set or unsteered default, stamped ``site_record_generation``
        (the FD 24 probe's own verdict, which a re-probe after an engine fix
        refreshes) — is never shadowed even at the same budget and util: its
        observed window S is the measurement; and ``budget`` (the EFFECTIVE budget :meth:`seqs_at`
        is sizing from — a served record's own recorded budget) is what the
        ceiling's profile is compared with, never this row's nominal budget."""
        if not self.hybrid or self.site_record_generation is not None:      # a site-overlay row, by the loader's stamp, never a string prefix
            return None
        effective = int(self.batched_token_budget if budget is None else budget)
        c = HYBRID_WINDOW_SEQS_CEILINGS.get((self.model_key, int(max_model_len)))
        if c is None or int(c.batched_token_budget) != effective or float(c.gpu_memory_utilization) != float(self.basis.gpu_memory_utilization):
            return None                                   # another profile (a measured overlay / fold row, a recorded budget, another util): no transfer
        return c

    def seqs_at(self, max_model_len: int, *, budget: Optional[int] = None) -> int:
        """The S this budget buys at window L: ``budget // L`` — capped at
        :data:`HYBRID_SEQS_CAP` for a hybrid-recurrent family, because the
        uncapped quotient is only an upper bound there (the class docstring's
        stated limitation; qwen35_4b: 469 -> 256 at L=1536), and then at the
        window's MEASURED ceiling when one is recorded
        (:meth:`window_ceiling`: nemotron3nano4b 170 -> 167 at L=4608, the
        allocator's verdict on the refused standup). ``budget`` (keyword-only)
        applies the same rule at ANOTHER budget than this row's — a served
        record's OWN recorded budget, what its §1.8 kernel step was applied to
        (``served_capability_run._native_identity_seqs``); the default is this
        row's ``batched_token_budget``, so every existing call is unchanged."""
        s = identity_seqs(max_model_len, budget=self.batched_token_budget if budget is None else budget)
        if self.hybrid:
            s = min(s, HYBRID_SEQS_CAP)
            ceiling = self.window_ceiling(max_model_len, budget=budget)     # the EFFECTIVE budget
            if ceiling is not None:
                s = min(s, ceiling.max_num_seqs)
        # the instance-level per-window minimum (the unsteered site default: the least admitted S over the record's certified sets at a
        # PROBED window, when smaller than the quotient — a measurement, applied whatever the family)
        w = (self.window_seqs or {}).get(int(max_model_len))
        return min(s, int(w)) if w is not None else s

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("site_record_generation", None)            # read-side provenance: a written record's format is unchanged
        d.pop("window_seqs_source", None); d.pop("window_seqs_launches", None)
        return d


#: THE one certified entry: Llama-3.1-8B's Petri footprint, 24 x 16384 =
#: 393,216 (vllm_server.py's DEFAULT_MAX_NUM_SEQS negative-result note).
_LLAMA8B_CERTIFIED = ModelBudget(
    model_key="llama8b", batched_token_budget=BATCHED_TOKEN_BUDGET, certified=True,
    basis=BudgetBasis(
        method=("the certified Petri footprint (24 x 16384 launches with ~25 GiB of slack) "
                "bracketed against the NEGATIVE RESULT: 32 x 16384 = 524,288 does NOT fit "
                "one H200 under the no-chunking identity at ANY utilization — util 0.95 "
                "under-covers KV by 2,768 tokens (521,520 = 31.83x, the coverage gate "
                "refused), util 0.97 covers KV and then OOMs by 0.48 GiB on the 14.0 GiB "
                "warmup MLP intermediate; re-tested at BI=0 with the same wall (structural)"),
        max_model_len=PETRI_MAX_MODEL_LEN, max_num_seqs=vs.DEFAULT_MAX_NUM_SEQS,
        gpu_memory_utilization=0.90,
        kv_tokens_reported=574064,       # job 230089 (BI=0, util 0.90): 35.04x at 16384
        probe_job_id="229194+229390 (the wall, two nodes); 229597/230089 (the certified footprint)",
        measured_on="2026-08-28",
        source=("src/sisteer/capeval/serving/vllm_server.py — DEFAULT_MAX_NUM_SEQS "
                "'NEGATIVE RESULT' + 'RE-TESTED AT BI=0' notes; the petri_steered_smoke "
                "PETRI_* gate (test_petri_geometry_cert) certifies the footprint live"),
        first_refused=524288,
        note=("Weights ~15 GiB + full-coverage KV 24 x 16384 x 128 KiB/token = 48.0 GiB + "
              "the 10.5 GiB profiling MLP intermediate on a 139.81 GiB card."),
    ))

# ---------------------------------------------------------------------------
# MEASURED BY scripts/served_decode_sweep.py --phase budget  (BEGIN)
#
# (pasted by scripts/geom_sweep_analyze.py; <SHARED> in a source = the cluster's per-user shared root, hf_env.sh's $SHARED)
_MEASURED_BUDGETS: Tuple[ModelBudget, ...] = (
    ModelBudget(
        model_key='llama8b', batched_token_budget=475_136, certified=False,
        basis=BudgetBasis(
            method="largest S x L that STOOD UP through the PRODUCTION STACKED-LENS STANDUP (scripts/sdfbase/mt_serving_up.py --reduction-mode stacked --require-attested-mode stacked, the full22 wave config + manifest + bank view, --allow-over-budget) — readiness passed vllm_server.kv_coverage_problem and the workers attested (S, L) and lens_reduction_mode=stacked — AND carried S concurrent FULL-WINDOW requests (gate_probe.py: S distinct L - cap - 64 token prompts at the wave's cap, ignore_eos, every finish_reason length, server alive) on one H200 at gpu_memory_utilization 0.90; bisection from the provisional S to the certified-negative 524,288 wall (the sdfbase-llama8b serving gate, 2026-09-10)",
            max_model_len=16_384, max_num_seqs=29,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=505_504,
            probe_job_id='476440', measured_on='2026-09-10',
            source='<SHARED>/exp/sdfbase_full22/serving_gate/ladder_16384_476440/ladder.json (node node-1; also ladder_8704_476440, ladder_45056_476439 and serving/static_llama8b_fullconst_gate_476439/static = Part A)',
            first_refused=507_904,
            note="kv facts: uniform_blocks_x_block_size; log max-concurrency 30.85x at 16384; probes: 6 at 16384 (14/23/27/29/30 pass, 31 kv_under_coverage 477,856 < 507,904), 5 at 8704 (26/43/52/56 pass, 57 + 58 kv_under_coverage), 4 at 45056 with the full-constitution prefix at cap 8192 on the prompted route (5/8/10 pass, 11 kv_under_coverage); wall reached: the KV cache shrinks ~13.8k tokens per 16,384 batched tokens of profile activations (712,896 at 229,376 -> 491,680 at 491,520, window-independent) and crosses S x L near 491.7k. 30 x 16384 = 491,520 ALSO passed, by 160 KV tokens on this 143,771-MiB card, and is EXCLUDED by choice: the cluster's 139.80-GiB H200s (the 2026-09-06 smoke's) hold ~4,400 fewer KV tokens at util 0.9 and would refuse it at readiness, while 29 keeps 30,368. The 2026-09-06 profile-forward OOM at 28 x 16384 (tree d844f0ae) did NOT reproduce on this tree: 29, 30 and the profile forward at 31 all passed. Supersedes the 2026-09-04 bare-engine 458,752 (56 x 8192, job 374299) and the 229,376 provisional ceiling. At the wave's windows: 8704 -> 54 (56 measured), 4864 -> 97, 7424 -> 64, 2048 -> 232, 45056 -> 10 (10 measured; Part A 5 x 45056 passed natural + full-cap). The 2026-09-06 failure was a torch.OutOfMemoryError (92.47 GiB held after the profile forward, the 24.50 GiB KV allocation refused); the ladders ran on tree fa9eb1c4 (decision D103), and the 45056 refusal at 11 was 488,224 < 495,616 KV tokens. This is the STARTING S only: on Modal the standup's KV-derived shrink (kv_shrink, D110) lowers S at readiness to whatever KV capacity that container reports",
        )),
    ModelBudget(
        model_key='llama3b', batched_token_budget=655_360, certified=False,
        basis=BudgetBasis(
            method='largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked upward to the first refusal, then one bisection)',
            max_model_len=8192, max_num_seqs=80,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=733_712,
            probe_job_id='375170', measured_on='2026-09-04',
            source='<SHARED>/exp/geom-sweep/llama3b/budget.json (node node-0)',
            first_refused=720_896,
            note='kv facts: uniform_blocks_x_block_size; log max-concurrency 89.56x at 8192; probes: 5; wall reached',
        )),
    ModelBudget(
        model_key='gemma4e4b', batched_token_budget=589_824, certified=False,
        basis=BudgetBasis(
            method='largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked upward to the first refusal, then one bisection)',
            max_model_len=8192, max_num_seqs=72,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=625_473,
            probe_job_id='65674', measured_on='2026-09-04',
            source='<SHARED>/exp/geom-sweep/gemma4e4b/budget.json (node node-10)',
            first_refused=655_360,
            note='kv facts: hybrid_max_concurrency_x_L; log max-concurrency 76.35x at 8192; probes: 5; wall reached',
        )),
    ModelBudget(
        model_key='gemma4e2b', batched_token_budget=786_432, certified=False,
        basis=BudgetBasis(
            method='largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked upward to the first refusal, then one bisection)',
            max_model_len=8192, max_num_seqs=96,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=960_543,
            probe_job_id='65677', measured_on='2026-09-04',
            source='<SHARED>/exp/geom-sweep/gemma4e2b/budget.json (node node-1)',
            first_refused=917_504,
            note='kv facts: hybrid_max_concurrency_x_L; log max-concurrency 117.25x at 8192; probes: 7; wall reached',
        )),
    ModelBudget(
        model_key='qwen35_9b', batched_token_budget=589_824, certified=False,
        basis=BudgetBasis(
            method='largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked upward to the first refusal, then one bisection)',
            max_model_len=8192, max_num_seqs=72,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=720_033,
            probe_job_id='65727', measured_on='2026-09-04',
            source='<SHARED>/exp/geom-sweep/qwen35_9b/budget.json (node node-6)',
            first_refused=655_360,
            note='kv facts: hybrid_max_concurrency_x_L; log max-concurrency 87.89x at 8192; probes: 4; wall reached',
        )),
    ModelBudget(
        model_key='qwen35_4b', batched_token_budget=741_888, certified=False,
        basis=BudgetBasis(
            method=('largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at '
                    'gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked '
                    'upward to the first refusal, then one bisection) — run AT THE MATH-500 WINDOW on the LENS STACK '
                    '(scripts/sbatch/geom_budget_window.sbatch: --budget-len 17664 = capability:gen_long_math, the '
                    'measured MATH prompt max + the 16,384 cap; the steered si_constitution__psr_kl_all cell from the '
                    'scale-12 bank loaded, the same cell the 2026-09-04 probes served; vllm 0.25.1+cu129, transformers '
                    "5.14.1, torch 2.11.0+cu129), the user's ruling of 2026-09-13 (math500-thinking-on-full22 D208 item "
                    '5): widen back to a MEASURED value honouring the 16,384 cap, not a hand revert'),
            max_model_len=17_664, max_num_seqs=42,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=850_259,
            probe_job_id='100196', measured_on='2026-09-13',
            source='<SHARED>/exp/geom-sweep-L17664/qwen35_4b/budget.json (node node-6; job 100196 on the overflow cluster; code sha 44111f3f = main b91d3921 + scripts/sbatch/geom_budget_window.sbatch; job log exp/logs/geom-budget-qwen35_4b-L17664_100196.out)',
            first_refused=777_216,
            note=('kv facts: hybrid_max_concurrency_x_L; log max-concurrency 48.14x at 17664; probes: 5 (20 / 32 / 40 x '
                  '17664 pass — KV 2,221,367 / 1,473,273 / 974,861 tokens, 125.76x / 83.41x / 55.19x; 44 x 17664 = '
                  "777,216 kv_under_coverage, the allocator's 41.05x < 44 at readiness; bisection 42 pass, KV 850,259); "
                  'wall reached. Supersedes the 2026-09-04 bare-engine sweep row 720,896 (88 x 8192, job 375242, first '
                  'refused 786,432) and the 2026-09-11 lift of its halved provisional shadow (360,448; PRECAUTION LIFTED '
                  "2026-09-11 on the AlphaSteer re-bake ladder's six lens-stack standups at 148 x 4864 = 719,872, "
                  'cluster B job 90773, KV attested 993,378) — the first measurement of this model AT a production '
                  'window on the lens stack. At the wave windows (seqs_at): 17,664 -> 42 (the MATH-500 wave ran at the '
                  "provisional 20; the swept row gave 40), 24,064 -> 30, 18,944 -> 39, 16,384 -> 45, 10,752 -> 69, "
                  '7,424 -> 99, 4,864 -> 152, 2,048 -> the hybrid cap 256. The hybrid per-slot cap still applies on top.'),
        )),
    ModelBudget(
        model_key='glm46vflash', batched_token_budget=786_432, certified=False,
        basis=BudgetBasis(
            method='largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked upward to the first refusal, then one bisection)',
            max_model_len=8192, max_num_seqs=96,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=867_584,
            probe_job_id='375192', measured_on='2026-09-04',
            source='<SHARED>/exp/geom-sweep/glm46vflash/budget.json (node node-0)',
            first_refused=917_504,
            note='kv facts: uniform_blocks_x_block_size; log max-concurrency 105.91x at 8192; probes: 7; wall reached',
        )),
    ModelBudget(
        model_key='nemotron3nano4b', batched_token_budget=786_432, certified=False,
        basis=BudgetBasis(
            method='largest S x L whose launch passed vllm_server.kv_coverage_problem at readiness on one H200 at gpu_memory_utilization 0.90 (scripts/served_decode_sweep.py --phase budget: the ladder walked upward to the first refusal, then one bisection)',
            max_model_len=8192, max_num_seqs=96,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=993_416,
            probe_job_id='65668', measured_on='2026-09-04',
            source='<SHARED>/exp/geom-sweep/nemotron3nano4b/budget.json (node node-0)',
            first_refused=917_504,
            note='kv facts: hybrid_max_concurrency_x_L; log max-concurrency 121.27x at 8192; probes: 6; wall reached',
        )),
)
# MEASURED BY scripts/served_decode_sweep.py --phase budget  (END)
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------- #
# PROVISIONAL CEILINGS — hand-written, and deliberately OUTSIDE the generated   #
# block above so a future analyzer paste cannot silently revert them (nor they  #
# it: _build_budget_table refuses a ceiling that is not BELOW the measurement   #
# it shadows, so re-running the sweep and pasting a smaller number is loud).    #
# Ported from main @ 31b869f7 (2026-09-06) onto the wave lineage, 2026-09-10.  #
# --------------------------------------------------------------------------- #
#
# WHY (the tiny-capability standup smokes, 2026-09-06, tree d844f0ae, ONE H200
# of 139.80 GiB). The composer pins the no-chunking identity
# ``max_num_batched_tokens = max_num_seqs x max_model_len`` exactly (the lens
# stack forbids chunked prefill), so vLLM's PROFILE forward runs the whole
# batched-token count and its activation peak must fit beside the KV cache. The
# 2026-09-04 sweep measured every budget at L=8192 on a bare engine; the lens
# stack at L=16384 does not have that headroom on the 8B/9B dense-attention
# models, and both died in engine init:
#
#   llama8b   458,752 = 28 x 16384 -> torch.OutOfMemoryError. After the profile
#             forward PyTorch held 92.47 GiB, 124.37 GiB of the card was in use
#             and only 15.42 GiB free, so the 24.50 GiB KV-cache allocation
#             failed ("Available KV cache memory: 63.37 GiB" had been computed
#             BEFORE the peak).
#   qwen35_9b 589,824 = 36 x 16384 -> CUBLAS_STATUS_EXECUTION_FAILED in a bf16
#             ``cublasGemmEx`` about ten minutes into init: cuBLAS could not
#             obtain workspace at a ~590k-row GEMM. Same exhaustion, different
#             symptom.
#
# gemma4e4b passed at the SAME 589,824 (E4B's hidden/intermediate sizes are far
# smaller), so no Gemma row is touched. The two ceilings below are chosen so the
# profile forward is roughly HALF the peak that OOM'd, at the certified 16384
# window; llama3b and qwen35_4b were not smoked, and were halved as the
# same-stack precaution rather than left at a number nothing had exercised here.
# Each is REPLACED by re-running ``scripts/served_decode_sweep.py --phase
# budget`` ON THIS STACK — through the production lens standup, at the windows
# the wave serves (the sdfbase-llama8b lane) — or by a completed launch
# at the swept budget on the lens stack — and deleting its row from this tuple;
# every S the wave derives then follows from the one table.
#
# llama8b's ceiling WAS replaced on 2026-09-10 (the sdfbase-llama8b serving gate,
# cluster A jobs 476439 / 476440, a compute node, a 143,771-MiB H200): the production
# stacked-lens standup (mt_serving_up.py, --allow-over-budget) passed 14 / 23 /
# 27 / 29 / 30 x 16384 at readiness AND under a full-window load probe; 31 x
# 16384 was refused by KV coverage (477,856 < 507,904) — the 2026-09-06
# profile-forward OOM at 28 x 16384 did not reproduce. Its measured row sits in
# the paste block above at 29 x 16384 (one card-size of KV margin below the 30
# that also passed). qwen35_4b's halved row (360,448) was deleted 2026-09-11 on
# the AlphaSteer re-bake ladder's six standups at 148 x 4864, and its row was then
# RE-MEASURED at the MATH-500 window (17,664) on the lens stack on 2026-09-13
# (741,888 = 42 x 17,664, cluster B job 100196; see the paste block).
# The two ceilings below still await their own measurement.
_PROVISIONAL_BUDGETS: Tuple[ModelBudget, ...] = (
    ModelBudget(
        model_key="qwen35_9b", batched_token_budget=196_608, certified=False,
        source=PROVISIONAL_BUDGET_SOURCE,
        basis=BudgetBasis(
            method=("CONSERVATIVE CEILING, not a measurement: a third of the batched-token "
                    "count that failed, expressed at the certified 16384 window (S=12). Sized "
                    "below llama8b's ceiling because Qwen3.5-9B's per-token activation cost is "
                    "above Llama-8B's and its failure came earlier, inside a GEMM"),
            max_model_len=PETRI_MAX_MODEL_LEN, max_num_seqs=12,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=None,
            probe_job_id=None, measured_on=UNMEASURED,
            source=("tiny-capability-lmeval standup smoke 2026-09-06 (tree d844f0ae, cell "
                    "tinycap-qwen9b-psrkl, thinking off): the cell's vllm_server.log — engine "
                    "init died at max_num_batched_tokens=589824, max_num_seqs=36"),
            first_refused=589_824,
            note=("OBSERVED FAILURE at 36 x 16384 on one H200: CUBLAS_STATUS_EXECUTION_FAILED "
                  "in a bf16 cublasGemmEx about ten minutes into init — cuBLAS could not "
                  "obtain workspace at a ~590k-row GEMM, the same exhaustion llama8b reported "
                  "as an OOM. Supersedes the 2026-09-04 sweep's 589,824 (72 x 8192, job 65727)"),
        )),
    ModelBudget(
        model_key="llama3b", batched_token_budget=327_680, certified=False,
        source=PROVISIONAL_BUDGET_SOURCE,
        basis=BudgetBasis(
            method=("CONSERVATIVE CEILING, not a measurement and NOT a failure of this model: "
                    "half the sweep's 655,360, expressed at the certified 16384 window (S=20), "
                    "because its Llama sibling OOM'd on this stack at a budget the same sweep "
                    "had passed"),
            max_model_len=PETRI_MAX_MODEL_LEN, max_num_seqs=20,
            gpu_memory_utilization=0.9,
            kv_tokens_reported=None,
            probe_job_id=None, measured_on=UNMEASURED,
            source=("precaution from the tiny-capability-lmeval standup smokes 2026-09-06 "
                    "(tree d844f0ae): llama8b and qwen35_9b failed on this stack at their "
                    "swept budgets; llama3b was not smoked"),
            first_refused=None,
            note=("NOT observed to fail — nothing has exercised 655,360 (80 x 8192, job "
                  "375170) on the lens stack at L=16384. Halved as the same-stack precaution; "
                  "re-measure before treating 327,680 as this model's wall"),
        )),
)


def _build_budget_table() -> Dict[str, ModelBudget]:
    """:data:`BUDGET_BY_MODEL`: the certified entry, then the swept block, then
    the provisional ceilings ON TOP. A ceiling may only LOWER the number it
    shadows — if a re-run sweep ever pastes a budget at or below one, the import
    fails rather than let the stale ceiling win silently."""
    table: Dict[str, ModelBudget] = {
        b.model_key: b for b in (_LLAMA8B_CERTIFIED,) + _MEASURED_BUDGETS}
    for prov in _PROVISIONAL_BUDGETS:
        shadowed = table.get(prov.model_key)
        if shadowed is not None and prov.batched_token_budget >= shadowed.batched_token_budget:
            raise ValueError(
                f"{prov.model_key}: the provisional ceiling {prov.batched_token_budget:,} does "
                f"not LOWER the {shadowed.source} {shadowed.batched_token_budget:,} it shadows "
                "— a ceiling exists to shrink a number that failed; drop the row from "
                "_PROVISIONAL_BUDGETS if the measurement has caught up")
        table[prov.model_key] = prov
    return table


#: model key -> its budget: a MEASURED (or certified) entry, or a PROVISIONAL
#: ceiling shadowing one (:data:`_PROVISIONAL_BUDGETS`). An absent key means
#: "not measured yet" and :func:`budget_for` says so.
BUDGET_BY_MODEL: Dict[str, ModelBudget] = _build_budget_table()

# --------------------------------------------------------------------------- #
# The ONBOARDED models' rows (paper-companion-repo, 2026-09-29)                #
# --------------------------------------------------------------------------- #
# An onboarded model is NOT in MODEL_KEYS / BUDGET_BY_MODEL (the sweep's pinned
# eight-row tables): its built-in row, when it has one, lives here and is served
# by :func:`rows_for`. A row here is a PROVISIONAL starting number (a stated
# ceiling adopted before anything has served the model on this stack — never a
# measurement), so a standup sizes from a stated number instead of a silent
# default until the FD 24 serving probe (stage 03) measures the admitted S per
# window. EMPTY today: qwen35_4b_demo (configured, not run) has NO row and sizes from
# the Llama-certified default WITH the warning until a probe measures it. The
# probe's measurement does NOT land in this tuple by hand: its verdict fold
# writes a measured row (certified, source=MEASURED_BUDGET_SOURCE) to the SITE
# GEOMETRY OVERLAY below, which :func:`rows_for` consults BEFORE these tables.
ONBOARDED_BUDGETS: Tuple[ModelBudget, ...] = ()
ONBOARDED_BUDGET_BY_MODEL: Dict[str, ModelBudget] = {b.model_key: b for b in ONBOARDED_BUDGETS}
for _b in ONBOARDED_BUDGETS:
    if _b.model_key not in ONBOARDED_MODEL_KEYS:
        raise ValueError(f"ONBOARDED_BUDGETS names {_b.model_key!r}, not an onboarded key")

# --------------------------------------------------------------------------- #
# The SITE geometry overlay                                                   #
# --------------------------------------------------------------------------- #
# The FD 24 serving probe (scripts/sdfbase/serving_probe.py) is the ONE place a
# test model's serving geometry is MEASURED: per allowed MODULE SET, the widest
# EXECUTING LoRA module stride the worker reported (the D of the kernel index
# ceiling) and the largest S x L its engine came up at with usable KV. Those
# numbers had no way into this module short of a hand-typed edit of the tables
# above, so a successful probe left :func:`kernel_ceiling_adjustment` refusing
# the model by name and :func:`budget_for` on the provisional row. The probe's
# verdict fold writes ONE per-model file, KEYED BY MODULE SET, with provenance,
# to a SITE overlay directory; :func:`rows_for` / :func:`budget_for` /
# :func:`kernel_ceiling_adjustment` resolve the record of the SELECTED set
# (``module_set=``) BEFORE the built-in tables. The overlay is SITE state,
# never a literal typed into this file.
#
# WHY PER SET: a native standup sizes its kernel ceiling from the adapters it
# actually registered (``native_standup.lora_kernel_ceiling_geometry`` over
# ``sa.adapter_module_dims``), so a wo design (D = o_proj's 4,096) attests a
# different S than a qkvo design (the packed 9,216 stride) at a ceiling-binding
# window. A model-wide maximum over the allowed sets would make the padded
# preflight and the post-standup pins expect the WRONG S for every design
# narrower than the widest allowed set — every steered padded unit refusing
# after standup despite a successful probe. So each set carries its own
# certified D and budget; a MODEL-WIDE summary is recorded for reading only and
# never sizes anything (``summary.note``). A caller that names no set gets the
# built-in tables (the provisional / default row, which steered padding refuses
# by name); a caller naming a set the file does not certify is REFUSED by name.
#
# Where (``site_geometry_dir``): ``$SISTEER_GEOMETRY_DIR/<model_key>.json``
# when that variable is set; else ``$SISTEER_SITE_DIR/geometry/`` (a front-door
# process: reproduce/lib/common.sh exports SISTEER_SITE_DIR after loading the
# site config, shipped as reproduce/site.example/; reproduce/03_hpo.sh copies
# the probe's record there); else ``$SISTEER_SHARED_ROOT/exp/reproduce/geometry/``
# (a cluster job under the front door's job env — reproduce/lib/slurm.sh
# slurm_job_env; the probe sbatch writes there, and slurm.sh's mirror step
# stages the site's records to EVERY target cluster with a sha check, so a
# twin on the other cluster reads the same file). With none set there is NO
# overlay and the built-in tables stand alone — every pre-existing caller and
# test. Never ``$HOME``. A present file that does not parse, names another
# model, or holds an inconsistent record is a :class:`SiteGeometryError`,
# never silently ignored.
SITE_GEOMETRY_DIR_ENV = "SISTEER_GEOMETRY_DIR"
SITE_DIR_ENV = "SISTEER_SITE_DIR"
SHARED_ROOT_ENV = "SISTEER_SHARED_ROOT"
SITE_GEOMETRY_SUBDIR = "geometry"
SITE_GEOMETRY_SHARED_SUBDIR = ("exp", "reproduce", "geometry")
#: version 2 = keyed by module set (version 1, one model-wide record, is REFUSED: that shape is retired)
SITE_GEOMETRY_VERSION = 2
#: ``BudgetBasis.method`` prefix of a probe-folded row
SITE_GEOMETRY_METHOD = "serving_probe"
#: the provenance keys every overlay record must carry (the reviewer's list: probe id, attestation sha, plus where it came from)
SITE_GEOMETRY_PROVENANCE_REQUIRED: Tuple[str, ...] = ("probe_job_id", "probe_record", "attestation_sha256")
#: the per-set keys every set record must carry
SITE_GEOMETRY_SET_REQUIRED: Tuple[str, ...] = ("lora_modules", "lora_max_module_dim", "budget", "allowed_pairs",
                                                "certified_by", "module_dim_launches", "budget_launches", "adapter_parity")
SITE_GEOMETRY_SUMMARY_NOTE = ("model-wide summary for READING only — it never sizes a standup: a native standup's ceiling is "
                              "the SELECTED set's D and its budget is that set's (D69)")
#: THE UNSTEERED DEFAULT (2026-09-30, the second failure of this kind): a caller that names NO module set — every production standup's
#: `budget_for(model_key)`, the unsteered padded preflight and pin — resolves, when the model HAS a site record, the set of the ZERO
#: adapter an unsteered native standup registers: `none__lora0` keeps the o_proj class when no genuine adapter selects another
#: (native_standup's zero-class rule: `zero_class = distinct[0] if distinct else ("o_proj",)`), i.e. the `wo` set. Before, no set
#: meant the built-in row even with a certified record present — an onboarded model's unsteered standups sized from their PROVISIONAL
#: built-in row while the staged overlay certified a smaller budget for every set, and died on the KV admission (2026-09-30). Now a
#: record that does not certify this set REFUSES by name, and a present-but-unusable record refuses as it always did: the provisional
#: / default row never stands in silently for a record that exists. A caller naming a set is unchanged (the SELECTED set's row).
UNSTEERED_MODULE_SET = "wo"
MODULE_SET_SOURCE_NAMED, MODULE_SET_SOURCE_UNSTEERED = "named", "unsteered_default"
#: THE ADAPTER-PARITY BLOCK: geometry certification and adapter parity are SEPARATE facts and the record says
#: both. Per set and at the record's top level: ``{"status": verified|mismatch|unverified, "basis": "hf_peft"|None, "per_pair":
#: [{"rank", "set", "status", "basis", "mismatched_adapters"}, ...], "rule": ADAPTER_PARITY_RULE}``; ``status`` is the aggregate
#: of ``per_pair`` (:func:`adapter_parity_aggregate`), the record's block the aggregate of its sets'. A STEERED consumer requires
#: ``verified`` for the pair it serves (:func:`require_verified_adapter_parity`); an UNSTEERED one reads the geometry whatever the
#: status says. A set's ``allowed_pairs`` are the pairs the fold read (their SERVING certified) — the LoRA menu is the subset
#: ``allowed_pairs.json`` lists, which requires verified parity.
ADAPTER_PARITY_KEY = "adapter_parity"
PARITY_VERIFIED, PARITY_MISMATCH, PARITY_UNVERIFIED = "verified", "mismatch", "unverified"
PARITY_STATUSES: Tuple[str, ...] = (PARITY_VERIFIED, PARITY_MISMATCH, PARITY_UNVERIFIED)
ADAPTER_PARITY_REQUIRED: Tuple[str, ...] = ("status", "basis", "per_pair")
ADAPTER_PARITY_PAIR_REQUIRED: Tuple[str, ...] = ("rank", "set", "status")
ADAPTER_PARITY_RULE = ("D145: geometry certification and adapter parity are separate facts. The geometry of each set was MEASURED on "
                       "engines that came up and served adapters of the folded pairs' class; whether those adapters compute what HF+PEFT "
                       "computes is adapter_parity. An UNSTEERED standup (no adapter, or the zero adapter alone) may size from this record "
                       "whatever the status; a STEERED standup (a trained adapter of (rank, set) served) requires status 'verified' for "
                       "that pair and refuses by name otherwise; the LoRA menu (allowed_pairs.json) lists verified pairs only.")
_BASIS_REQUIRED: Tuple[str, ...] = ("method", "max_model_len", "max_num_seqs", "gpu_memory_utilization", "measured_on", "source")
_BASIS_OPTIONAL: Tuple[str, ...] = ("kv_tokens_reported", "probe_job_id", "first_refused", "note")


class SiteGeometryError(ValueError):
    """A site geometry overlay file is present but unusable (malformed, another
    model's, an inconsistent record), the SELECTED module set has no certified
    record in it, or a write was asked for with no overlay directory
    configured. Loud by design: never a fallback to the built-in row."""


@dataclass(frozen=True)
class SiteGeometrySet:
    """ONE (model, module set)'s probe-certified geometry: the widest executing
    LoRA module dim ``D`` of that set's adapters (the kernel index ceiling's
    divisor), the measured :class:`ModelBudget` of that set's launches
    (certified, ``source=measured``), the allowed (rank, set) pairs and the
    launches that certified them, and the record's (shared) provenance."""

    model_key: str
    module_set: str
    lora_modules: Tuple[str, ...]
    lora_max_module_dim: int
    budget: ModelBudget
    allowed_pairs: Tuple[Tuple[int, str], ...]
    certified_by: Tuple[str, ...]
    module_dim_launches: Mapping[str, int]
    budget_launches: Mapping[str, Mapping[str, int]]
    provenance: Mapping[str, Any]
    #: this set's adapter-parity block ({status, basis, per_pair, rule}); a steered consumer requires 'verified' for its pair
    adapter_parity: Mapping[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    def as_set_record(self) -> Dict[str, Any]:
        return {"lora_modules": list(self.lora_modules), "lora_max_module_dim": int(self.lora_max_module_dim),
                ADAPTER_PARITY_KEY: json.loads(json.dumps(dict(self.adapter_parity))),
                "budget": {"batched_token_budget": self.budget.batched_token_budget, "certified": self.budget.certified,
                           "source": self.budget.source, "basis": asdict(self.budget.basis)},
                "allowed_pairs": [[int(r), str(s)] for r, s in self.allowed_pairs],
                "certified_by": list(self.certified_by),
                "module_dim_launches": {str(k): int(v) for k, v in dict(self.module_dim_launches).items()},
                "budget_launches": {str(k): {str(kk): int(vv) for kk, vv in dict(v).items()}
                                    for k, v in dict(self.budget_launches).items()}}


@dataclass(frozen=True)
class SiteGeometry:
    """ONE model's overlay FILE: its per-set records (:class:`SiteGeometrySet`,
    keyed by module set) plus the reading-only summary and the shared
    provenance (probe job id, record dir, the launch attestations' sha256s,
    tree sha, host)."""

    model_key: str
    hf_id: str
    revision: str
    written_utc: str
    provenance: Mapping[str, Any]
    sets: Mapping[str, SiteGeometrySet]
    summary: Mapping[str, Any]
    #: the record-level adapter-parity block, DERIVED from the sets' (any mismatch -> mismatch; else any unverified -> unverified)
    adapter_parity: Mapping[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    #: the staging-time verification block a staged copy carries (None on the probe cluster's own overlay and on the grading host
    #: site copy) — see :data:`STAGING_VERIFICATION_KEY`
    staging_verification: Optional[Mapping[str, Any]] = None

    def as_record(self) -> Dict[str, Any]:
        rec = site_geometry_record(model_key=self.model_key, hf_id=self.hf_id, revision=self.revision,
                                   sets={k: v.as_set_record() for k, v in self.sets.items()},
                                   provenance=self.provenance, written_utc=self.written_utc,
                                   adapter_parity=(dict(self.adapter_parity) or None))
        if self.staging_verification is not None:
            rec[STAGING_VERIFICATION_KEY] = dict(self.staging_verification)   # a staged copy round-trips WITH its verification
        return rec


def site_geometry_dir(directory: Optional[object] = None) -> Optional[Path]:
    """The overlay directory: ``directory`` when given, else the env rule in
    the block comment above; ``None`` when nothing configures one."""
    if directory is not None and str(directory):
        return Path(str(directory))
    v = os.environ.get(SITE_GEOMETRY_DIR_ENV)
    if v:
        return Path(v)
    v = os.environ.get(SITE_DIR_ENV)
    if v:
        return Path(v) / SITE_GEOMETRY_SUBDIR
    v = os.environ.get(SHARED_ROOT_ENV)
    if v:
        return Path(v).joinpath(*SITE_GEOMETRY_SHARED_SUBDIR)
    return None


def site_geometry_path(model_key: str, directory: Optional[object] = None) -> Optional[Path]:
    d = site_geometry_dir(directory)
    return None if d is None else d / f"{model_key}.json"


def _site_overlay_where(model_key: str, module_set: Optional[str] = None) -> str:
    """For refusal messages: where the overlay for ``model_key`` was looked for and what it holds."""
    p = site_geometry_path(model_key)
    if p is None:
        return (f"not configured ({SITE_GEOMETRY_DIR_ENV} / {SITE_DIR_ENV} / {SHARED_ROOT_ENV} all unset)")
    if not p.is_file():
        marker = read_revoked_marker(p)
        if marker is not None:
            return (f"REVOKED — absent at {p} with {revoked_marker_path(p).name} beside it (revoked_utc={marker.get('revoked_utc')}, "
                    f"reason={marker.get('reason')!r}): the record was reopened; nothing sizes this model until it is re-staged")
        return f"absent at {p}"
    try:
        site = load_site_geometry(model_key)
    except SiteGeometryError as e:
        return f"present at {p} but UNUSABLE ({e})"
    certified = sorted(site.sets) if site is not None else []
    if module_set is None:
        return (f"present at {p} with certified module sets {certified}; a caller naming no set takes the {UNSTEERED_MODULE_SET!r} row "
                "(the zero adapter's class an unsteered standup registers)"
                + ("" if UNSTEERED_MODULE_SET in certified else f" — which this record does NOT certify") + "; a steered caller names its set (module_set=)")
    return f"present at {p}; certified module sets {certified}"


def _pos_int(v: object, what: str, path: Optional[str]) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise SiteGeometryError(f"site geometry {path or '<record>'}: {what} must be a positive int, got {v!r}")
    return v


def _summary_of(sets: Mapping[str, SiteGeometrySet]) -> Dict[str, Any]:
    dims = {k: int(v.lora_max_module_dim) for k, v in sets.items()}
    budgets = {k: int(v.budget.batched_token_budget) for k, v in sets.items()}
    return {"note": SITE_GEOMETRY_SUMMARY_NOTE, "sets": sorted(sets),
            "lora_max_module_dim_by_set": dict(sorted(dims.items())), "lora_max_module_dim_max": max(dims.values()),
            "batched_token_budget_by_set": dict(sorted(budgets.items())),
            "batched_token_budget_max": max(budgets.values()), "batched_token_budget_min": min(budgets.values())}


def adapter_parity_aggregate(statuses: Sequence[str]) -> str:
    """The set- or record-level parity status over its pairs' statuses: any ``mismatch`` -> ``mismatch``; else any
    ``unverified`` (or no pair at all — nothing was shown) -> ``unverified``; else ``verified``."""
    if any(s == PARITY_MISMATCH for s in statuses):
        return PARITY_MISMATCH
    if not statuses or any(s != PARITY_VERIFIED for s in statuses):
        return PARITY_UNVERIFIED
    return PARITY_VERIFIED


def adapter_parity_block(per_pair: Sequence[Mapping[str, Any]], *, basis: Optional[str] = None,
                         note: Optional[str] = None) -> Dict[str, Any]:
    """ONE parity block over ``per_pair`` entries (``{rank, set, status, basis?, mismatched_adapters?}``), sorted by (set,
    rank), its ``status`` the aggregate and its ``basis`` the pairs' one basis (``basis`` when given; None when they differ or
    none has one)."""
    rows: List[Dict[str, Any]] = []
    for e in per_pair:
        row = {"rank": int(e["rank"]), "set": str(e["set"]), "status": str(e.get("status") or PARITY_UNVERIFIED),
               "basis": e.get("basis"), "mismatched_adapters": sorted(str(x) for x in (e.get("mismatched_adapters") or []))}
        # the venv the pair's parity was measured in and the certificate version ride on the row natively (the probe's fold
        # writes them; None = unproven, which the admission gate refuses by name) — never dropped by the aggregate
        for key in (CERTIFICATE_VENV_KEY, "certificate_version"):
            if key in e:
                row[key] = (dict(e[key]) if isinstance(e[key], Mapping) else e[key])
        rows.append(row)
    rows.sort(key=lambda r: (r["set"], r["rank"]))
    bases = sorted({r["basis"] for r in rows if r["basis"]})
    blk: Dict[str, Any] = {"status": adapter_parity_aggregate([r["status"] for r in rows]),
                           "basis": (basis if basis is not None else (bases[0] if len(bases) == 1 else None)),
                           "per_pair": rows, "rule": ADAPTER_PARITY_RULE}
    if note:
        blk["note"] = note
    return blk


def _parse_parity(doc: object, where: str, *, pairs: Optional[Sequence[Tuple[int, str]]] = None,
                  module_set: Optional[str] = None) -> Dict[str, Any]:
    """Validate ONE parity block: the required keys, a known status, ``per_pair`` entries with rank / set / status (exactly the
    ``pairs`` when given; every entry's set == ``module_set`` when given) and ``status`` equal to the aggregate of ``per_pair``."""
    k = ADAPTER_PARITY_KEY
    if not isinstance(doc, Mapping):
        raise SiteGeometryError(f"{where}.{k} must be an object (D145: geometry certification and adapter parity are separate facts "
                                "and the record says both)")
    missing = [f for f in ADAPTER_PARITY_REQUIRED if f not in doc]
    if missing:
        raise SiteGeometryError(f"{where}.{k} lacks {missing}")
    if doc["status"] not in PARITY_STATUSES:
        raise SiteGeometryError(f"{where}.{k}.status {doc['status']!r} is not one of {list(PARITY_STATUSES)}")
    if doc["basis"] is not None and (not isinstance(doc["basis"], str) or not doc["basis"]):
        raise SiteGeometryError(f"{where}.{k}.basis must be a non-empty string or null")
    pp = doc["per_pair"]
    if not isinstance(pp, (list, tuple)):
        raise SiteGeometryError(f"{where}.{k}.per_pair must be a list")
    seen: List[Tuple[int, str]] = []
    rows: List[Dict[str, Any]] = []
    for e in pp:
        if not isinstance(e, Mapping) or any(f not in e for f in ADAPTER_PARITY_PAIR_REQUIRED):
            raise SiteGeometryError(f"{where}.{k}.per_pair entry {e!r} lacks {list(ADAPTER_PARITY_PAIR_REQUIRED)}")
        if isinstance(e["rank"], bool) or not isinstance(e["rank"], int) or not isinstance(e["set"], str) or not e["set"]:
            raise SiteGeometryError(f"{where}.{k}.per_pair entry {e!r}: rank must be an int and set a module set id")
        if e["status"] not in PARITY_STATUSES:
            raise SiteGeometryError(f"{where}.{k}.per_pair [{e['rank']}, {e['set']!r}] status {e['status']!r} is not one of {list(PARITY_STATUSES)}")
        if module_set is not None and e["set"] != module_set:
            raise SiteGeometryError(f"{where}.{k}.per_pair entry names set {e['set']!r}, not {module_set!r}")
        seen.append((int(e["rank"]), str(e["set"])))
        rows.append(dict(e))
    if len(set(seen)) != len(seen):
        raise SiteGeometryError(f"{where}.{k}.per_pair lists a pair twice ({seen})")
    if pairs is not None:
        want = sorted({(int(r), str(s_)) for r, s_ in pairs})
        if sorted(set(seen)) != want:
            raise SiteGeometryError(f"{where}.{k}.per_pair {sorted(set(seen))} is not the set's folded pairs {want}")
    expect = adapter_parity_aggregate([r["status"] for r in rows])
    if doc["status"] != expect:
        raise SiteGeometryError(f"{where}.{k}.status {doc['status']!r} disagrees with its per_pair statuses (aggregate {expect!r})")
    return dict(doc)


def _parity_of_sets(sets: Mapping[str, SiteGeometrySet]) -> Dict[str, Any]:
    """The record-level parity block DERIVED from the sets' blocks (never hand-written): every set's per_pair, aggregated."""
    rows = [dict(e) for s_ in sets.values() for e in (s_.adapter_parity or {}).get("per_pair") or []]
    bases = sorted({(s_.adapter_parity or {}).get("basis") for s_ in sets.values()} - {None})
    return adapter_parity_block(rows, basis=(bases[0] if len(bases) == 1 else None))


def _parity_rows(block: Mapping[str, Any]) -> List[Tuple[int, str, str]]:
    return sorted((int(r["rank"]), str(r["set"]), str(r["status"])) for r in (block.get("per_pair") or []))


def _parse_set(key: str, sid: object, doc: object, prov: Mapping[str, Any], p: Optional[str]) -> SiteGeometrySet:
    where = f"site geometry {p or '<record>'}: sets[{sid!r}]"
    if not isinstance(sid, str) or not sid or not sid.replace("_", "").replace("+", "").isalnum():
        raise SiteGeometryError(f"site geometry {p or '<record>'}: module set id {sid!r} is not an identifier")
    if not isinstance(doc, Mapping):
        raise SiteGeometryError(f"{where} must be an object")
    missing = [f for f in SITE_GEOMETRY_SET_REQUIRED if f not in doc]
    if missing:
        raise SiteGeometryError(f"{where} lacks {missing}")
    mods = doc["lora_modules"]
    if not isinstance(mods, (list, tuple)) or not mods or not all(isinstance(m, str) and m for m in mods):
        raise SiteGeometryError(f"{where}.lora_modules must be a non-empty list of module names")
    D = _pos_int(doc["lora_max_module_dim"], f"sets[{sid!r}].lora_max_module_dim", p)
    mdl = doc["module_dim_launches"]
    if not isinstance(mdl, Mapping) or not mdl:
        raise SiteGeometryError(f"{where}.module_dim_launches must be a non-empty object (launch id -> measured D)")
    mdl_out = {str(k): _pos_int(v, f"sets[{sid!r}].module_dim_launches[{k!r}]", p) for k, v in mdl.items()}
    if max(mdl_out.values()) != D:
        raise SiteGeometryError(f"{where}.lora_max_module_dim {D} != the widest measured launch dim {max(mdl_out.values())} ({mdl_out})")
    bl = doc["budget_launches"]
    if not isinstance(bl, Mapping) or not bl:
        raise SiteGeometryError(f"{where}.budget_launches must be a non-empty object")
    bl_out: Dict[str, Dict[str, int]] = {}
    for k, v in bl.items():
        if not isinstance(v, Mapping):
            raise SiteGeometryError(f"{where}.budget_launches[{k!r}] must be an object")
        bl_out[str(k)] = {str(kk): _pos_int(vv, f"sets[{sid!r}].budget_launches[{k!r}][{kk!r}]", p) for kk, vv in v.items()}
    ap = doc["allowed_pairs"]
    if not isinstance(ap, (list, tuple)) or not ap:
        raise SiteGeometryError(f"{where}.allowed_pairs must be a non-empty list of [rank, set]")
    pairs: List[Tuple[int, str]] = []
    for item in ap:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or isinstance(item[0], bool) or not isinstance(item[0], int) \
                or item[1] != sid:
            raise SiteGeometryError(f"{where}.allowed_pairs entry {item!r} is not [rank, {sid!r}]")
        pairs.append((int(item[0]), str(item[1])))
    cb = doc["certified_by"]
    if not isinstance(cb, (list, tuple)) or not cb or not all(isinstance(x, str) and x for x in cb):
        raise SiteGeometryError(f"{where}.certified_by must be a non-empty list of launch ids")
    parity = _parse_parity(doc[ADAPTER_PARITY_KEY], where, pairs=pairs, module_set=str(sid))     # The set says both facts
    b = doc["budget"]
    if not isinstance(b, Mapping) or not isinstance(b.get("basis"), Mapping):
        raise SiteGeometryError(f"{where}.budget must be an object with a basis object")
    basis_doc = b["basis"]
    missing = [f for f in _BASIS_REQUIRED if f not in basis_doc]
    if missing:
        raise SiteGeometryError(f"{where}.budget.basis lacks {missing}")
    unknown = sorted(set(basis_doc) - set(_BASIS_REQUIRED) - set(_BASIS_OPTIONAL))
    if unknown:
        raise SiteGeometryError(f"{where}.budget.basis has unknown keys {unknown}")
    try:
        basis = BudgetBasis(**{k: basis_doc[k] for k in basis_doc})
        budget = ModelBudget(model_key=key, batched_token_budget=b.get("batched_token_budget"), basis=basis,
                             certified=bool(b.get("certified", False)), source=b.get("source", MEASURED_BUDGET_SOURCE),
                             site_record_generation=str((prov or {}).get(RESULT_GENERATION_KEY) or "unstamped"))   # the loader's stamp
    except (TypeError, ValueError) as e:
        raise SiteGeometryError(f"{where}.budget rejected — {e}") from None
    if budget.batched_token_budget not in {v.get("batched_tokens") for v in bl_out.values()}:
        raise SiteGeometryError(f"{where}.budget.batched_token_budget {budget.batched_token_budget:,} is not one of its "
                                f"launches' products ({[v.get('batched_tokens') for v in bl_out.values()]})")
    return SiteGeometrySet(model_key=key, module_set=sid, lora_modules=tuple(str(m) for m in mods), lora_max_module_dim=D,
                           budget=budget, allowed_pairs=tuple(pairs), certified_by=tuple(str(x) for x in cb),
                           module_dim_launches=mdl_out, budget_launches=bl_out, provenance=prov, adapter_parity=parity, path=p)


def parse_site_geometry(doc: object, *, path: Optional[object] = None,
                        model_key: Optional[str] = None) -> SiteGeometry:
    """Validate ONE overlay file. Every field the resolvers read is checked
    here; each set's budget goes through :class:`ModelBudget` itself (its
    product / block-size / source rules); the stored summary must agree with
    the sets. ``model_key``, when given, must be the record's (a file under one
    name holding another model's row is refused). A version-1 (model-wide)
    record is refused: it is a retired shape."""
    p = None if path is None else str(path)
    if not isinstance(doc, Mapping):
        raise SiteGeometryError(f"site geometry {p or '<record>'}: not a JSON object ({type(doc).__name__})")
    if doc.get("version") != SITE_GEOMETRY_VERSION:
        raise SiteGeometryError(f"site geometry {p or '<record>'}: version {doc.get('version')!r} != {SITE_GEOMETRY_VERSION} "
                                "(a version-1 model-wide record is refused: the geometry is certified PER MODULE SET, D69 — re-run the "
                                "probe's `verdict` to re-fold)")
    key = doc.get("model_key")
    if not isinstance(key, str) or key not in KNOWN_MODEL_KEYS:
        raise SiteGeometryError(f"site geometry {p or '<record>'}: model_key {key!r} is not a known model key "
                                f"({list(KNOWN_MODEL_KEYS)})")
    if model_key is not None and key != model_key:
        raise SiteGeometryError(f"site geometry {p or '<record>'}: names model {key!r}, asked for {model_key!r}")
    for f in ("hf_id", "revision", "written_utc"):
        if not isinstance(doc.get(f), str) or not doc[f]:
            raise SiteGeometryError(f"site geometry {p or '<record>'}: {f} must be a non-empty string")
    prov = doc.get("provenance")
    if not isinstance(prov, Mapping):
        raise SiteGeometryError(f"site geometry {p or '<record>'}: provenance must be an object")
    for f in SITE_GEOMETRY_PROVENANCE_REQUIRED:
        if f not in prov or prov[f] in (None, "", {}, []):
            raise SiteGeometryError(f"site geometry {p or '<record>'}: provenance.{f} is missing or empty "
                                    f"(required: {list(SITE_GEOMETRY_PROVENANCE_REQUIRED)})")
    sets_doc = doc.get("sets")
    if not isinstance(sets_doc, Mapping) or not sets_doc:
        raise SiteGeometryError(f"site geometry {p or '<record>'}: sets must be a non-empty object keyed by module set")
    sets = {str(sid): _parse_set(key, sid, sdoc, dict(prov), p) for sid, sdoc in sets_doc.items()}
    summary = doc.get("summary")
    if not isinstance(summary, Mapping):
        raise SiteGeometryError(f"site geometry {p or '<record>'}: summary must be an object")
    expect = _summary_of(sets)
    for f in ("lora_max_module_dim_max", "batched_token_budget_max", "batched_token_budget_min", "sets"):
        if summary.get(f) != expect[f]:
            raise SiteGeometryError(f"site geometry {p or '<record>'}: summary.{f} {summary.get(f)!r} disagrees with the sets ({expect[f]!r})")
    # the record says BOTH facts — the record-level parity block is required and must be the aggregate of the sets' blocks
    ap_doc = doc.get(ADAPTER_PARITY_KEY)
    if ap_doc is None:
        raise SiteGeometryError(f"site geometry {p or '<record>'}: no {ADAPTER_PARITY_KEY} block — D145: a geometry record must say both "
                                "facts (geometry certification AND adapter parity); a record without it predates the rule — re-fold it "
                                "(`serving_probe.py reverdict`)")
    parity = _parse_parity(ap_doc, f"site geometry {p or '<record>'}")
    expect_p = _parity_of_sets(sets)
    if parity["status"] != expect_p["status"] or _parity_rows(parity) != _parity_rows(expect_p):
        raise SiteGeometryError(f"site geometry {p or '<record>'}: {ADAPTER_PARITY_KEY} ({parity['status']}, {_parity_rows(parity)}) disagrees "
                                f"with the sets' blocks ({expect_p['status']}, {_parity_rows(expect_p)})")
    sv = doc.get(STAGING_VERIFICATION_KEY)
    if sv is not None:
        if not isinstance(sv, Mapping):
            raise SiteGeometryError(f"site geometry {p or '<record>'}: {STAGING_VERIFICATION_KEY} must be an object")
        for f in STAGING_VERIFICATION_REQUIRED:
            if not isinstance(sv.get(f), str) or not sv.get(f):
                raise SiteGeometryError(f"site geometry {p or '<record>'}: {STAGING_VERIFICATION_KEY}.{f} is missing or empty "
                                        f"(required: {list(STAGING_VERIFICATION_REQUIRED)})")
    return SiteGeometry(model_key=key, hf_id=doc["hf_id"], revision=doc["revision"], written_utc=doc["written_utc"],
                        provenance=dict(prov), sets=sets, summary=dict(summary), adapter_parity=parity, path=p,
                        staging_verification=(dict(sv) if sv is not None else None))


def site_geometry_record(*, model_key: str, hf_id: str, revision: str, sets: Mapping[str, Mapping[str, Any]],
                         provenance: Mapping[str, Any], written_utc: str,
                         adapter_parity: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The overlay's JSON body for ONE model: ``sets`` = module set -> the
    per-set record (``lora_modules``, ``lora_max_module_dim``, ``budget`` as a
    :class:`ModelBudget` OR its dict, ``allowed_pairs``, ``certified_by``,
    ``module_dim_launches``, ``budget_launches``); the summary is derived here.
    Validated by parsing it back, so a record this function returns is a
    record :func:`load_site_geometry` accepts. A set record without an
    ``adapter_parity`` block gets one reading ``unverified`` for every pair
    (stated, never inferred as verified); the record-level block is derived
    from the sets' (``adapter_parity`` given: it must agree)."""
    sets_doc: Dict[str, Any] = {}
    for sid, rec in sets.items():
        rec = dict(rec)
        b = rec.get("budget")
        if isinstance(b, ModelBudget):
            rec["budget"] = {"batched_token_budget": b.batched_token_budget, "certified": b.certified, "source": b.source,
                             "basis": asdict(b.basis)}
        if ADAPTER_PARITY_KEY not in rec:
            # no block recorded (an older caller, a fixture): the pairs' parity was NOT shown -> `unverified`, stated — a steered
            # standup refuses it by name, an unsteered one reads the geometry; never inferred as verified
            rec[ADAPTER_PARITY_KEY] = adapter_parity_block(
                [{"rank": r_, "set": str(sid), "status": PARITY_UNVERIFIED, "basis": None} for r_, _s in (rec.get("allowed_pairs") or [])],
                note="no adapter-parity block was recorded for this set: every pair reads `unverified` (D145)")
        sets_doc[str(sid)] = rec
    doc = {"version": SITE_GEOMETRY_VERSION, "model_key": model_key, "hf_id": hf_id, "revision": revision,
           "written_utc": written_utc, "provenance": dict(provenance), "sets": sets_doc, "summary": {}}
    # the summary is DERIVED from the sets (never hand-written): parse once without it to build it, then attach
    parsed_sets = {str(sid): _parse_set(model_key, str(sid), sdoc, dict(provenance), None) for sid, sdoc in sets_doc.items()}
    doc["summary"] = _summary_of(parsed_sets)
    # the record-level parity block is DERIVED from the sets' too; a caller-given one must agree (the parse below checks)
    doc[ADAPTER_PARITY_KEY] = dict(adapter_parity) if adapter_parity is not None else _parity_of_sets(parsed_sets)
    parse_site_geometry(doc, model_key=model_key)
    return doc


#: THE RESULT GENERATION of a serving-probe record: a content hash over the record's
#: current result set — every ``launches/*.json`` file name with the sha256 of its bytes, sorted — stamped by the probe
#: into its checkpoint, its verdict, its menu (``allowed_pairs.json``) and this overlay's provenance. A consumer that can
#: reach the record verifies the stamp against the record as it stands NOW, so a menu or an overlay written before a
#: result was reopened (``run --resume --replan``, a deleted result, a re-run) is never read as current.
RESULT_GENERATION_KEY = "result_generation"
RECORD_LAUNCHES_DIR = "launches"
RECORD_PROGRESS_FILE = "probe_progress.json"
#: THE STAGING VERIFICATION: a copy of an overlay that leaves the probe cluster (the grading host site
#: copy staged to another cluster by ``reproduce/lib/slurm.sh slurm_stage_geometry``) is verified at STAGING time against the
#: authoritative record — the verifying status (``hpo_probe_status.py --record``) run over ssh on the probe cluster must say
#: READY with the record's generation equal to the overlay's own ``result_generation`` — and the staged copy is stamped with
#: this top-level block ``{verified_utc, record_generation, verifier_host}``. A consumer that cannot reach the record accepts
#: the overlay ONLY when the block is present and its ``record_generation`` equals the overlay's stamp; the grading host site copy
#: itself never carries the block, so nothing on the grading host reads as valid without staging. The block is outside the record's
#: canonical content (the staging compare ignores it).
STAGING_VERIFICATION_KEY = "staging_verification"
STAGING_VERIFICATION_REQUIRED: Tuple[str, ...] = ("verified_utc", "record_generation", "verifier_host")
#: THE REVOKED MARKER: clusters cannot read each other or the grading host (no keys), so a staged copy's
#: two stamps (its own ``result_generation`` and the ``staging_verification`` block) never change when the SOURCE record is
#: reopened — a job on the other cluster could accept revoked geometry. The invariant is revoke-before-reopen: before the
#: front door reopens a record whose checkpoint is complete, it archives the site copy and EVERY configured cluster's staged
#: overlay and writes ``<model_key>.REVOKED.json`` beside where the overlay stood ({model_key, generation, revoked_utc,
#: reason, by, and ``epoch`` — sha256 of the revocation receipt's nonce — when the revocation named its record}); a consumer
#: refuses whenever that marker exists, reachable record or not — for a record this host cannot reach the MARKER, not the
#: stamp, is the freshness signal. The next verified staging moves the marker aside: for a copy of the SAME generation, only
#: the copy certified through that revocation's receipt (its ``provenance.reopen_epoch`` is the marker's ``epoch``; B11) —
#: never one ordered against the marker by the clock alone.
REVOKED_MARKER_SUFFIX = ".REVOKED.json"


def record_result_generation(record_dir: object) -> Optional[Dict[str, Any]]:
    """``{generation, n_results, files}`` over ``<record>/launches/*.json`` (the per-launch result files; the
    ``superseded/`` history and the per-launch dirs are not part of it), or None when the record has no launches
    directory. The SAME algorithm as ``scripts/sdfbase/hpo_probe_status.record_generation`` (stdlib-only for the cluster
    login host; a test pins them equal): ``sha256("\n".join(f"{name}:{sha256(bytes)}" for sorted files))``."""
    d = Path(str(record_dir)) / RECORD_LAUNCHES_DIR
    if not d.is_dir():
        return None
    lines: List[str] = []
    for f in sorted(d.glob("*.json")):
        if f.is_file():
            lines.append(f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}")
    return {"generation": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), "n_results": len(lines),
            "files": [ln.split(":", 1)[0] for ln in lines]}


def revoked_marker_path(overlay_path: object) -> Path:
    """``<dir>/<model_key>.REVOKED.json`` for the overlay file ``<dir>/<model_key>.json``."""
    p = Path(str(overlay_path))
    stem = p.name[:-5] if p.name.endswith(".json") else p.name
    return p.with_name(stem + REVOKED_MARKER_SUFFIX)


def read_revoked_marker(overlay_path: object) -> Optional[Dict[str, Any]]:
    """The marker beside the overlay, parsed (``{"unreadable": True}`` for a torn one), or None when none stands."""
    m = revoked_marker_path(overlay_path)
    if not m.is_file():
        return None
    try:
        doc = json.loads(m.read_text(encoding="utf-8"))
        return dict(doc) if isinstance(doc, Mapping) else {"unreadable": True}
    except (OSError, ValueError):
        return {"unreadable": True}


def check_overlay_generation(site: "SiteGeometry") -> Optional[str]:
    """Refuse an overlay OLDER than its probe record. When the record dir the provenance names is
    reachable from here (the probe cluster; a fetched copy), the record must be CLOSED — its ``probe_progress.json``
    ``complete`` — and, when the overlay carries a ``result_generation`` stamp, the stamp must equal the generation of the
    record's result set as it stands now; either failure is a :class:`SiteGeometryError` naming the record. A record
    that is NOT reachable (another cluster, the grading host site copy) is accepted ONLY when the overlay carries a
    :data:`STAGING_VERIFICATION_KEY` block whose ``record_generation`` equals the overlay's own ``result_generation``
    stamp — the proof that ``slurm_stage_geometry`` verified it against the authoritative record at staging time; an
    unreachable record without that block, or with a block for another generation, or an unstamped overlay, REFUSES by
    name (an authoritative source that cannot be read is an error, never a silent accept). Before either branch, a
    ``<model_key>.REVOKED.json`` marker beside the overlay refuses outright (re-check 2): for a record this host cannot reach
    the marker — written by the front door before it reopens the record — is the freshness signal, not the stamp, whose two
    fields never change when the source is reopened. Returns the check's outcome (``"verified"`` / ``"verified:unstamped"``
    / ``"verified:staged"``)."""
    prov = site.provenance or {}
    rec_dir = prov.get("probe_record")
    where = f"site geometry {site.path or '<record>'}"
    # the REVOKED marker beside the overlay comes first: it is written by the front door BEFORE the record is reopened and is the
    # only signal a host that cannot reach the record has (re-check 2); its presence refuses whatever the stamps say
    if site.path:
        marker = read_revoked_marker(site.path)
        if marker is not None:
            raise SiteGeometryError(
                f"{where}: REVOKED — {revoked_marker_path(site.path).name} stands beside it (revoked_utc={marker.get('revoked_utc')}, "
                f"generation={str(marker.get('generation'))[:16]}…, reason={marker.get('reason')!r}): the front door reopened this "
                "record; this copy's geometry is withdrawn until the re-probe's verified staging replaces it and clears the marker")
    if not isinstance(rec_dir, str) or not rec_dir or not Path(rec_dir).is_dir():
        stamp = prov.get(RESULT_GENERATION_KEY)
        sv = site.staging_verification
        if not stamp:
            raise SiteGeometryError(f"{where}: its probe record {rec_dir!r} is not reachable from here and the overlay carries no "
                                    f"{RESULT_GENERATION_KEY} stamp — it cannot be verified against its record; refusing (re-fold the record "
                                    "on the probe cluster and stage it through the front door)")
        if not sv:
            raise SiteGeometryError(f"{where}: its probe record {rec_dir!r} is not reachable from here and the overlay carries no "
                                    f"{STAGING_VERIFICATION_KEY} block — it was never staged through the verifying step "
                                    "(reproduce/lib/slurm.sh slurm_stage_geometry); refusing to size from an unverified copy")
        if str(sv.get("record_generation")) != str(stamp):
            raise SiteGeometryError(f"{where}: its {STAGING_VERIFICATION_KEY}.record_generation {str(sv.get('record_generation'))[:16]}… "
                                    f"is not the overlay's own {RESULT_GENERATION_KEY} {str(stamp)[:16]}… — the staging verified another "
                                    "result set; refusing")
        return "verified:staged"
    prog_p = Path(rec_dir) / RECORD_PROGRESS_FILE
    if prog_p.is_file():
        try:
            prog = json.loads(prog_p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SiteGeometryError(f"{where}: its probe record {rec_dir} has an unreadable {RECORD_PROGRESS_FILE} "
                                    f"({type(e).__name__}: {e}) — refusing to size from an overlay whose record cannot be read") from None
        if not (isinstance(prog, dict) and prog.get("complete") is True):
            raise SiteGeometryError(
                f"{where}: its probe record {rec_dir} is REOPENED ({RECORD_PROGRESS_FILE} complete={prog.get('complete') if isinstance(prog, dict) else prog!r}"
                f"{', replan_required=' + str(prog.get('replan_required')) if isinstance(prog, dict) and prog.get('replan_required') else ''})"
                " — this overlay predates the re-probe; refusing until the record's verdict re-certifies it")
    stamped = prov.get(RESULT_GENERATION_KEY)
    gen = record_result_generation(rec_dir)
    if stamped and gen is not None and gen["generation"] != stamped:
        raise SiteGeometryError(
            f"{where}: its {RESULT_GENERATION_KEY} {str(stamped)[:16]}… is OLDER than the probe record's result set "
            f"({gen['generation'][:16]}…, {gen['n_results']} result files under {rec_dir}/{RECORD_LAUNCHES_DIR}) — a result was "
            "reopened or re-run after this overlay was folded; refusing until the verdict re-folds it")
    return "verified" if stamped else "verified:unstamped"


def load_site_geometry(model_key: str, directory: Optional[object] = None) -> Optional[SiteGeometry]:
    """The overlay FILE for ``model_key``: ``None`` when no overlay directory
    is configured, or when NEITHER the file NOR a ``<model_key>.REVOKED.json``
    marker exists (the built-in tables stand); a :class:`SiteGeometryError`
    when the file is present and unusable — or OLDER than the probe record it
    names (:func:`check_overlay_generation`) — or when the file is ABSENT but
    the marker stands where it stood: a receipted reopen
    archives every copy and leaves exactly that marker, and a revoked record is
    never "no record" — a queued unpadded standup during a re-probe must refuse,
    not fall to the provisional row."""
    p = site_geometry_path(model_key, directory)
    if p is None:
        return None
    if not p.is_file():
        marker = read_revoked_marker(p)
        if marker is not None:
            raise SiteGeometryError(
                f"site geometry {p}: the overlay is ABSENT and {revoked_marker_path(p).name} stands where it stood "
                f"(revoked_utc={marker.get('revoked_utc')}, generation={str(marker.get('generation'))[:16]}…, reason={marker.get('reason')!r}): "
                f"the front door reopened this record and archived every copy — nothing sizes {model_key!r} until the re-probe's verified "
                "staging replaces it and clears the marker; refusing (the built-in tables never stand in for a revoked record)")
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SiteGeometryError(f"site geometry {p}: unreadable — {type(e).__name__}: {e}") from None
    site = parse_site_geometry(doc, path=p, model_key=model_key)
    check_overlay_generation(site)
    return site


def site_geometry_for(model_key: str, module_set: str, directory: Optional[object] = None) -> Optional[SiteGeometrySet]:
    """The certified record of (``model_key``, ``module_set``): ``None`` when
    no overlay file exists for the model (the built-in tables stand); the
    set's record when the file certifies it; a :class:`SiteGeometryError`
    NAMING the set and the certified ones when the file exists but does not
    certify the selected set (never a neighbour's record, never the
    summary)."""
    if not isinstance(module_set, str) or not module_set:
        raise SiteGeometryError(f"{model_key}: module_set must be a non-empty module set id, got {module_set!r}")
    site = load_site_geometry(model_key, directory)
    if site is None:
        return None
    rec = site.sets.get(module_set)
    if rec is None:
        raise SiteGeometryError(
            f"{model_key}: the selected module set {module_set!r} has NO certified geometry record in {site.path} "
            f"(certified module sets: {sorted(site.sets)}) — the serving probe did not certify that set (or the HPO pick "
            "names a set outside the probe's menu); refusing rather than sizing from another set's D or the summary")
    return rec


#: the per-launch adapter-class keys the probe fold writes beside S / L / batched_tokens (2026-10-02)
LAUNCH_CLASS_KEYS: Tuple[str, str] = ("max_lora_rank", "max_loras")
WINDOW_SEQS_RECORD_WIDE = "record-wide"


def launch_class(bl: Mapping[str, object]) -> Optional[Tuple[int, int]]:
    """``(max_lora_rank, max_loras)`` of a folded budget launch, or None when the record predates launch classes (either key absent or
    not a positive int — never guessed from the launch id)."""
    vals = []
    for k in LAUNCH_CLASS_KEYS:
        v = bl.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            return None
        vals.append(int(v))
    return vals[0], vals[1]


def least_admitted_by_window(model_key: str, launches: Mapping[str, Mapping[str, object]],
                             served_lora: Optional[Tuple[int, int]] = None) -> Tuple[Dict[int, int], str, Dict[int, Tuple[str, ...]]]:
    """``(window -> S, source, window -> the launches that set it)`` over a record's budget launches (every set's).

    With NO served class, or a record whose launches carry no class: the least admitted S over every launch at each window
    (source ``record-wide``, with a note when classes are unrecorded). With a served class ``(rank cap, adapters)`` on a classed record:
    at each window the launches whose class DOMINATES the served one (rank cap >= and adapters >=), narrowed to the TIGHTEST dominating
    footprint (the least rank x adapters), and the least admitted S over those — a rank-64 x 128-adapter launch admitted at S=1 no longer
    caps a rank-1 x 3-adapter standup (the s2r pair target 831867, 2026-10-02) while the r4 / cap8 launches at the same window admitted 14.
    Refused by name: a window none of whose launches is classed while others are (a half-classed record), and a window where no probed
    launch dominates the served class (nothing certifies it there)."""
    by_L: Dict[int, List[Tuple[str, Mapping[str, object]]]] = {}
    for lid, bl in launches.items():
        by_L.setdefault(int(bl["L"]), []).append((str(lid), bl))
    classed_any = any(launch_class(bl) is not None for bl in launches.values())
    if served_lora is None or not classed_any:
        ws = {L: min(int(bl["S"]) for _, bl in ls) for L, ls in by_L.items()}
        src = WINDOW_SEQS_RECORD_WIDE if served_lora is None else f"{WINDOW_SEQS_RECORD_WIDE} (launch classes unrecorded — re-fold the record)"
        chosen = {L: tuple(sorted(lid for lid, bl in ls if int(bl["S"]) == ws[L])) for L, ls in by_L.items()}
        return dict(sorted(ws.items())), src, dict(sorted(chosen.items()))
    rank_s, n_s = int(served_lora[0]), int(served_lora[1])
    if rank_s < 1 or n_s < 1:
        raise ValueError(f"served_lora must be (rank cap >= 1, adapters >= 1), got {served_lora!r}")
    ws: Dict[int, int] = {}; chosen: Dict[int, Tuple[str, ...]] = {}
    for L, ls in sorted(by_L.items()):
        classed = [(lid, bl, launch_class(bl)) for lid, bl in ls if launch_class(bl) is not None]
        if not classed:
            raise SiteGeometryError(f"{model_key}: no launch at window {L} carries an adapter class ({'/'.join(LAUNCH_CLASS_KEYS)}) while other "
                                    f"windows do — a half-classed record sizes nothing at {L}; re-fold it")
        dom = [(lid, bl, c) for lid, bl, c in classed if c is not None and c[0] >= rank_s and c[1] >= n_s]
        if not dom:
            raise SiteGeometryError(f"{model_key}: no probed launch at window {L} dominates the served adapter class (rank cap {rank_s}, {n_s} "
                                    f"adapters): probed classes {sorted({c for _, _, c in classed if c is not None})} — nothing certifies that class at {L}")
        tight = min(c[0] * c[1] for _, _, c in dom if c is not None)
        pick = [(lid, bl) for lid, bl, c in dom if c is not None and c[0] * c[1] == tight]
        ws[L] = min(int(bl["S"]) for _, bl in pick)
        chosen[L] = tuple(sorted(lid for lid, bl in pick if int(bl["S"]) == ws[L]))
    return ws, f"launch_class:{rank_s}x{n_s}", chosen


def unsteered_site_set(model_key: str, *, served_lora: Optional[Tuple[int, int]] = None) -> Optional[SiteGeometrySet]:
    """The row an UNSTEERED caller — one naming no module set — sizes from, when the model has a site record
    (2026-09-30): CONSERVATIVE across the record's certified sets — the MINIMUM batched-token budget over the sets and, at each
    PROBED window, the least admitted S over the sets (inside ``seqs_at`` when smaller than ``budget // L``; an unprobed window
    is the quotient) — with the D of the zero adapter's class (:data:`UNSTEERED_MODULE_SET`, o_proj: what an unsteered native
    standup registers and attests), the record-level adapter parity, and ``basis.source = site_record:<generation>``. ``None``
    when the model has no overlay at all (the built-in tables stand — every pre-existing caller); a :class:`SiteGeometryError`
    NAMING the certified sets when the overlay exists but does not certify the zero adapter's class, and whatever
    :func:`load_site_geometry` refuses for a present-but-unusable / stale / REVOKED file: a record that exists is never silently
    replaced by the provisional / default row. The row inherits the sets' standing (a provisional set row makes it provisional).
    Callers apply it where the built-in tables hold a guess or nothing (``rows_for``: a provisional / absent row;
    ``lora_max_module_dim_for``: no pinned D) — a MEASURED sweep row stands for a no-set caller. The resolution itself is
    :func:`unsteered_row_of` over the loaded record; this function only loads it."""
    site = load_site_geometry(model_key) if isinstance(model_key, str) else None
    if site is None:
        return None
    return unsteered_row_of(site, served_lora=served_lora)


def unsteered_row_of(site: SiteGeometry, *, served_lora: Optional[Tuple[int, int]] = None) -> SiteGeometrySet:
    """:func:`unsteered_site_set`'s resolution over an already-parsed record — the pure half, so the serving probe's verdict can
    run the SAME resolution over the record it just folded: a steered standup's S
    is ``native_standup.launch_class_geometry`` -> ``resolve_geometry(model_key, served_lora)`` -> ``budget_for`` with NO module
    set -> this row's ``budget.seqs_at(L)`` — every certified set's launches merged, the least admitted S among the TIGHTEST launches
    dominating the served class at each window (:func:`least_admitted_by_window`), inside the record's MINIMUM budget ``// L`` — so
    another set's launch of the same class caps a pair's standup below its own union's S, and whatever prices a steered standup must
    read this row, never one set's. Raises :class:`SiteGeometryError` when the record does not certify
    :data:`UNSTEERED_MODULE_SET` or nothing probed dominates the served class at a window (the standup refuses by the same name)."""
    model_key = site.model_key
    zero = site.sets.get(UNSTEERED_MODULE_SET)
    if zero is None:
        raise SiteGeometryError(
            f"{model_key}: a site geometry record exists ({site.path}; certified module sets {sorted(site.sets)}) but the caller named no "
            f"module set and the record does not certify {UNSTEERED_MODULE_SET!r} — the class of the zero adapter (o_proj) an unsteered "
            "native standup registers, whose D the standup attests — so nothing in it sizes an unsteered standup; refusing rather than "
            f"sizing from the built-in row while a record exists (a steered standup names the set it serves; an unsteered one needs the "
            f"probe to certify {UNSTEERED_MODULE_SET!r})")
    # CONSERVATIVE across the record's certified sets: the MINIMUM batched-token budget over the sets, and at every PROBED window
    # the least admitted S over the sets (applied inside seqs_at when smaller than budget // L); the D is the zero adapter's class's
    # (what the standup attests). The row says where it came from: basis.source = site_record:<generation>.
    sets = sorted(site.sets.values(), key=lambda r: (r.budget.batched_token_budget, r.module_set))
    least = sets[0]
    launches: Dict[str, Mapping[str, int]] = {str(lid): dict(bl) for r in site.sets.values() for lid, bl in dict(r.budget_launches).items()}
    # the per-window S: record-wide with no served class; else over the launches whose class dominates the served one
    window_seqs, ws_source, ws_launches = least_admitted_by_window(model_key, launches, served_lora)
    gen = str((site.provenance or {}).get(RESULT_GENERATION_KEY) or "unstamped")
    basis = replace(least.budget.basis,
                    source=f"site_record:{gen}",
                    method=(f"{SITE_GEOMETRY_METHOD}: unsteered default — no module set named; the MINIMUM batched-token budget over the "
                            f"record's certified sets {sorted(site.sets)} ({least.module_set!r}: {least.budget.basis.method}); at a probed "
                            f"window the least admitted S ({ws_source}) when smaller than budget // L: {dict(sorted(window_seqs.items()))}; "
                            f"D = the zero adapter's class {UNSTEERED_MODULE_SET!r} ({zero.lora_max_module_dim})"),
                    note=(f"unsteered default from {site.path}; per-set budgets "
                          f"{ {k: v.budget.batched_token_budget for k, v in sorted(site.sets.items())} }; {least.budget.basis.note}"))
    # the row inherits the sets' own standing: a provisional row among the certified sets makes the default provisional (and never
    # certified) — a padded standup then refuses it by name exactly as it refuses a provisional named row
    provisional = any(r.budget.provisional for r in site.sets.values())
    budget = ModelBudget(model_key=model_key, batched_token_budget=int(least.budget.batched_token_budget), basis=basis,
                         certified=(not provisional and all(r.budget.certified for r in site.sets.values())),
                         source=(PROVISIONAL_BUDGET_SOURCE if provisional else MEASURED_BUDGET_SOURCE),
                         window_seqs=dict(sorted(window_seqs.items())), site_record_generation=gen,
                         window_seqs_source=ws_source, window_seqs_launches=dict(sorted(ws_launches.items())))
    log.info("[geometry] model %r: no module set named — sizing from the site record (source=site_record:%s): the minimum batched-token "
             "budget over its certified sets %s = %s (%r), the least admitted S per probed window %s, D %s (the zero adapter's class %r); %s",
             model_key, gen[:16], sorted(site.sets), f"{budget.batched_token_budget:,}", least.module_set, dict(sorted(window_seqs.items())),
             zero.lora_max_module_dim, UNSTEERED_MODULE_SET, site.path)
    return SiteGeometrySet(model_key=site.model_key, module_set=UNSTEERED_MODULE_SET, lora_modules=zero.lora_modules,
                           lora_max_module_dim=zero.lora_max_module_dim, budget=budget, allowed_pairs=zero.allowed_pairs,
                           certified_by=tuple(sorted({l for r in site.sets.values() for l in r.certified_by})),
                           module_dim_launches=zero.module_dim_launches, budget_launches=launches, provenance=site.provenance,
                           adapter_parity=site.adapter_parity, path=site.path)


def adapter_parity_for(rec: SiteGeometrySet, rank: Optional[int] = None) -> Dict[str, Any]:
    """The parity of (``rank``, this set) from the set's block — ``rank`` None: the set's aggregate — as ``{rank, set, status,
    basis, mismatched_adapters}``; a rank the set did not fold refuses by name (its parity is unknown)."""
    blk = dict(rec.adapter_parity or {})
    if rank is None:
        return {"rank": None, "set": rec.module_set, "status": blk.get("status", PARITY_UNVERIFIED), "basis": blk.get("basis"),
                "mismatched_adapters": sorted({a for e in blk.get("per_pair") or [] for a in (e.get("mismatched_adapters") or [])})}
    for e in blk.get("per_pair") or []:
        if int(e["rank"]) == int(rank):
            return {"rank": int(rank), "set": rec.module_set, "status": e["status"], "basis": e.get("basis"),
                    "mismatched_adapters": list(e.get("mismatched_adapters") or []),
                    # the venv the probe measured this pair's parity in (bound by the probe's fold; None on an older certificate)
                    CERTIFICATE_VENV_KEY: (dict(e[CERTIFICATE_VENV_KEY]) if isinstance(e.get(CERTIFICATE_VENV_KEY), Mapping) else None),
                    "certificate_version": e.get("certificate_version")}
    raise SiteGeometryError(f"{rec.model_key}: the pair (rank {rank}, set {rec.module_set!r}) is not one the serving probe folded for this "
                            f"set (folded ranks: {sorted(int(e['rank']) for e in blk.get('per_pair') or [])}; record {rec.path or '<record>'}) — "
                            "its adapter parity is unknown; a steered standup of that pair refuses by name")


#: THE CERTIFICATE VENV: a pair's parity was measured by the serving probe in ONE venv — the base
#: (pure-torch GDN reference) or a kernel variant (fused) — and the numerics differ; the probe's fold binds that venv into each
#: ``per_pair`` entry as ``serving_venv: {requirements_sha256, variant, name}`` (the standup-side facts of the probe's own
#: process: serving_attestation.serving_venv_facts), and a production standup may serve the pair only in THAT venv. The digest
#: is the authoritative key; the campaign venv name (its variant + hash8) is the fallback; two dev venvs match on an equal prefix.
CERTIFICATE_VENV_KEY = "serving_venv"
#: THE DIFFERENCE ADAPTER: the kind lora_export._export_lora_subtraction stamps on a LoRA subtraction's exported
#: adapter (``petri_source_kind``; native_standup.genuine_adapter_pairs forwards it as ``kind``). The probe folds ONE pair per FIT rank
#: and certifies it only when "its single-rank AND its rank-2r difference adapter passed" (serving_probe.py: the difference is a rank-2r
#: adapter of the same shape, natural_cap = rank_cap(2r), the S1d server, the role-keyed rule), so the certificate of (r, set) IS the
#: certificate of the rank-2r subtraction of two rank-r fits on that set. The subtraction exports as the rank-stacked pair (r = 2r), and
#: the gate used to look that served rank up as a pair of its own — (32, 'mlp') on a front door whose overlay folds 4 / 16 / 64, never
#: folded — refusing every standup that served the difference. A subtraction is held to (rank / 2, set) and the record says so.
DIFFERENCE_ADAPTER_KIND = "lora_subtraction"
CERTIFICATE_VENV_FIELDS: Tuple[str, ...] = ("requirements_sha256", "variant", "name", "prefix")
#: THE TWO SIDES OF A CERTIFICATE: the engine identity's VALUES are not enough — WHERE it came from
#: decides. ``serving_venv.serving`` names the identity of the engine that produced the SERVING outputs the parity was judged on, with
#: its ``engine_identity_source``: ``standup_process`` (the probe's standup process recorded its own facts) or ``launch_provenance``
#: (derived per launch from the launch's vllm_server.log version lines + the carrier job's recorded venv, derivation recorded on the
#: block); ``serving_venv.reference`` is the HF+PEFT reference process's identity (``reference_process``) — it attests the HF side
#: only and never admits. A serving side sourced from the reference process alone, an unknown source, or no serving side refuses.
CERTIFICATE_SERVING_SIDE_KEY, CERTIFICATE_REFERENCE_SIDE_KEY = "serving", "reference"
ENGINE_IDENTITY_SOURCE_KEY = "engine_identity_source"
ENGINE_SOURCES_ADMITTED: Tuple[str, ...] = ("standup_process", "launch_provenance")
ENGINE_SOURCE_REFERENCE = "reference_process"


def certificate_venv_problem(pair: Mapping[str, Any], serving: Mapping[str, Any]) -> Optional[str]:
    """Why ``serving`` (this standup's :func:`serving_attestation.serving_venv_facts`) may NOT serve the pair whose certificate
    entry is ``pair`` (:func:`adapter_parity_for`'s return); ``None`` when the ENGINE identities agree (the engine is
    venv-independent — vLLM vendors its GDN / causal-conv1d kernels and the overlays reach only the HF+PEFT reference — so the gate
    compares ``engine_identity`` {vllm_version, base_requirements_sha256, flashinfer_version} of the certificate's SERVING side,
    whose ``engine_identity_source`` must be ``standup_process`` or ``launch_provenance`` — the reference process
    attests the HF side only; the venv's variant and full digest are the HF-reference PROVENANCE, recorded, never gated on).
    Refusals, each distinct: (a) the certificate carries no venv block —
    UNPROVEN (an older certificate is not exempt: the 25898f8d7 lesson); (b) its block carries no complete engine identity — UNPROVEN
    engine; (c) the standup's own identity is not its process's (the exported venv is not sys.prefix; an exported digest its campaign
    name does not agree with; a non-campaign prefix with a campaign digest) — refused before anything is compared; (d) a certificate
    whose provenance disagrees with itself (a digest without the campaign name that earned it, or a name whose hash8 / variant disagree
    with its digest) certifies nothing; (e) the engine identities differ — MISMATCH naming both; (f) a side cannot state its engine."""
    cert = pair.get(CERTIFICATE_VENV_KEY)
    if not isinstance(cert, Mapping) or not any(cert.get(k) for k in CERTIFICATE_VENV_FIELDS + (_sa.ENGINE_IDENTITY_KEY, CERTIFICATE_SERVING_SIDE_KEY, CERTIFICATE_REFERENCE_SIDE_KEY)):
        return ("its parity certificate names NO serving venv (no `serving_venv` on the pair's entry) — an older certificate whose "
                "reference may have run on another numerics path is unproven, not exempt; re-fold the probe's verdict on the venv "
                "that serves (serving_probe.py reverdict after the venv-bound reference phase)")
    exported = dict(serving.get("exported") or {})
    s_digest = exported.get("requirements_sha256") or None
    s_prefix = serving.get("prefix")
    c_digest, c_name, c_variant = cert.get("requirements_sha256") or None, cert.get("name") or None, cert.get("variant")
    s_variant = exported.get("variant") if exported.get("variant") is not None else serving.get("variant")
    def _label(digest, name, variant, prefix=None):
        return f"variant {variant or 'base'}, {('requirements ' + str(digest)[:12] + '…') if digest else ('venv ' + str(name or prefix))}"
    # (c) THE SERVING IDENTITY MUST BE THE PROCESS'S: the exported identity is what the SHELL selected; the standup
    # that will serve runs in sys.prefix. A process outside the exported venv (the driver python after the shell picked the gdn
    # venv) would carry one venv's identity into another's numerics — refused by name before anything is compared.
    if serving.get("exported_matches_prefix") is False:
        return (f"this standup's process runs in {s_prefix!r} but the shell selected the serving venv {exported.get('venv')!r} "
                f"({_label(s_digest, exported.get('venv'), s_variant)}) — the exported identity is not the process's, so nothing it "
                "carries certifies this process; run the standup with the selected venv's python (common.sh: $SPY, never $MPY)")
    if s_digest:
        # a digest is believed only when the process's own venv name agrees with it: the campaign venv dir is named
        # sisteer-petri-vllm[-<variant>]-<sha256(requirements [+ overlay])[:8]>, so hash8 == digest[:8] and the variants agree
        if not serving.get("campaign_venv"):
            return (f"this standup's process runs in {s_prefix!r}, which is not a campaign serving venv, yet carries the exported requirements "
                    f"digest {str(s_digest)[:12]}… — an identity the process cannot have earned; refusing")
        if str(serving.get("hash8") or "") != str(s_digest)[:8] or (exported.get("variant") or None) != (serving.get("variant") or None):
            return (f"this standup's exported identity ({_label(s_digest, exported.get('venv'), exported.get('variant'))}) is inconsistent with "
                    f"its process venv {serving.get('name')!r} (variant {serving.get('variant') or 'base'}, hash8 {serving.get('hash8')!r} != "
                    f"digest[:8] {str(s_digest)[:8]!r}); refusing")
    # (d) THE CERTIFICATE'S PROVENANCE MUST AGREE WITH ITSELF: a probe run from a developer or driver venv that merely
    # inherited SISTEER_SERVING_REQS_SHA256 records {name: '.venv', requirements_sha256: <gdn>} — a digest its process cannot have earned
    if c_digest:
        c_parsed = _sa.parse_serving_venv_name(c_name or "")
        if not c_parsed["campaign_venv"]:
            return (f"its parity certificate asserts the requirements digest {str(c_digest)[:12]}… but names {c_name!r}, which is not a "
                    "campaign serving venv (sisteer-petri-vllm[-<variant>]-<hash8>) — a digest the certifying process cannot have earned "
                    "(an inherited SISTEER_SERVING_REQS_SHA256); re-fold the certificate from a probe stood up through common.sh "
                    "f22_serving_venv, in the venv that serves")
        if c_parsed["hash8"] != str(c_digest)[:8] or (c_parsed["variant"] or None) != (c_variant or None):
            return (f"its parity certificate's venv identity is inconsistent (name {c_name!r} vs requirements {str(c_digest)[:12]}…, variant "
                    f"{c_variant or 'base'}) — a certificate that disagrees with itself certifies nothing; re-fold the probe's verdict")
    # (b) / (e) / (f) THE ENGINE IDENTITY OF THE SERVING SIDE, FROM AN ADMITTED SOURCE: the one comparison that admits. The identity
    # must be the one the SERVING outputs were produced under — the probe's standup process or the launch's own provenance — never
    # the reference process's (it attests the HF side only); an unknown or missing source refuses by name.
    side = cert.get(CERTIFICATE_SERVING_SIDE_KEY)
    if not isinstance(side, Mapping) or not side:
        return (f"its parity certificate names no SERVING-side identity (`{CERTIFICATE_VENV_KEY}.{CERTIFICATE_SERVING_SIDE_KEY}` absent) — the "
                "engine that produced the serving outputs its parity was judged on is unproven, not exempt (D654); a top-level engine identity "
                "or a reference-side block does not stand in for it; re-fold the certificate from launch provenance or a standup that records "
                "its own engine")
    source = side.get(ENGINE_IDENTITY_SOURCE_KEY)
    if source == ENGINE_SOURCE_REFERENCE:
        return (f"its parity certificate's serving-side engine identity comes from the reference process alone "
                f"({ENGINE_IDENTITY_SOURCE_KEY}={source!r}) — the reference process attests the HF side only; the launches that produced the "
                "serving outputs were never measured, so the pair is unproven (D654); bind the serving side from launch provenance "
                "(vllm_server.log version lines + the carrier job's venv) or re-judge under a standup that records its own engine")
    if source not in ENGINE_SOURCES_ADMITTED:
        return (f"its parity certificate's serving-side engine identity carries an unknown or missing source "
                f"({ENGINE_IDENTITY_SOURCE_KEY}={source!r}; admitted: {list(ENGINE_SOURCES_ADMITTED)}) — refusing by name")
    c_engine = side.get(_sa.ENGINE_IDENTITY_KEY)
    if not isinstance(c_engine, Mapping) or any(not c_engine.get(k) for k in _sa.ENGINE_IDENTITY_FIELDS):
        return (f"its parity certificate's serving side ({ENGINE_IDENTITY_SOURCE_KEY}={source!r}) carries no complete engine identity "
                f"(`{_sa.ENGINE_IDENTITY_KEY}` with {list(_sa.ENGINE_IDENTITY_FIELDS)}; got "
                f"{dict(c_engine) if isinstance(c_engine, Mapping) else c_engine!r}) — the engine that measured its parity is unproven, not "
                "exempt (D548); re-fold the probe's verdict from a probe whose standup records its engine")
    s_engine = serving.get(_sa.ENGINE_IDENTITY_KEY)
    if not isinstance(s_engine, Mapping) or any(not s_engine.get(k) for k in _sa.ENGINE_IDENTITY_FIELDS):
        return (f"this standup cannot state its engine identity ({list(_sa.ENGINE_IDENTITY_FIELDS)}; got "
                f"{dict(s_engine) if isinstance(s_engine, Mapping) else s_engine!r}) — stand the engine up through common.sh f22_serving_venv "
                "(which exports the base requirements digest) in a venv that installs vllm and flashinfer; refusing")
    diff = {k: (c_engine.get(k), s_engine.get(k)) for k in _sa.ENGINE_IDENTITY_FIELDS if str(c_engine.get(k)) != str(s_engine.get(k))}
    if diff:
        return (f"its parity certificate was measured on another ENGINE than the one about to serve it (serving side from {source}) — "
                + "; ".join(f"{k}: certificate {c!r} vs this standup {s!r}" for k, (c, s) in diff.items())
                + f" (certificate HF-reference venv: {_label(c_digest, c_name, c_variant)}; this standup: "
                f"{_label(s_digest, serving.get('name'), s_variant, s_prefix)}) — parity under one engine does not certify another; serve the "
                "pair from the certificate's engine or re-fold the certificate on this one")
    return None


def require_verified_adapter_parity(rec: SiteGeometrySet, rank: Optional[int] = None, *,
                                    what: str = "a steered standup") -> Dict[str, Any]:
    """The STEERED gate: ``what`` serves a TRAINED adapter of (``rank``, this set) and may size from this record only when
    that pair's adapter parity is ``verified``. Returns the pair's parity entry; refuses (:class:`SiteGeometryError`) a
    ``mismatch`` or ``unverified`` pair — or a rank the set did not fold — NAMING the pair, the status, the adapters that
    disagreed and the way back (the probe's reference phase, then ``serving_probe.py reverdict``). An UNSTEERED standup (no
    adapter, or the zero adapter alone) never calls this: it reads the geometry whatever the block says."""
    par = adapter_parity_for(rec, rank)
    if par["status"] == PARITY_VERIFIED:
        return par
    pair = f"(rank {rank if rank is not None else 'any'}, set {rec.module_set!r})"
    detail = (f"; adapters that disagreed with the HF+PEFT reference: {par['mismatched_adapters']}" if par["mismatched_adapters"] else "")
    raise SiteGeometryError(
        f"{rec.model_key}: {what} serves a trained adapter of {pair}, whose adapter parity in {rec.path or '<record>'} is "
        f"{par['status']!r} (basis {par['basis']!r}){detail} — D145: this set's geometry is certified (an unsteered standup may size "
        "from it), but a steered standup requires 'verified' parity for the pair it serves; refusing by name until the probe's "
        "reference phase verifies the pair and its verdict is re-folded (serving_probe.py reverdict)")


#: hpo_probe_status's READY (the front door's only fetch state) — spelled here so the standup gate can hold a reachable record to it
#: without importing the script (the caller hands the status reader in; the string is the contract, pinned by the tests)
PROBE_RECORD_READY = "READY"


def standup_adapter_parity_gate(model_key: str, adapters: Mapping[str, Mapping[str, Any]], *,
                                record_check: Optional[Callable[[Path], Tuple[str, str, Mapping[str, Any]]]] = None,
                                what: str = "this steered standup", stamp_not_before: Optional[str] = None,
                                serving_venv: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """THE RUNTIME STANDUP GATE (the steered rule applied where the engine actually comes up): ``adapters``
    maps every GENUINE adapter the standup banks (never the zero adapter) to ``{rank, module_set}`` — the pair it serves, read
    off its own files; the standup may launch only when the site geometry overlay staged on THIS cluster says every such pair's
    adapter parity is ``verified``. Returns the gate's record for the endpoint:

    * no genuine adapter (the zero adapter alone, a prompted-only unsteered standup): ``{"status": "unsteered"}`` — nothing to gate;
    * no overlay for a SWEEP model (the pinned tables stand, as before): ``{"status": "no_overlay"}``;
    * no overlay for an ONBOARDED model (:data:`ONBOARDED_MODEL_KEYS`): REFUSED — a steered standup of an onboarded model sizes and
      certifies only from the probe's staged, parity-verified record (03_hpo.sh --phase probe stages it; lib/slurm.sh verifies it);
    * an overlay: every pair through :func:`require_verified_adapter_parity` (a set the overlay does not certify, a rank it did not
      fold, a ``mismatch`` / ``unverified`` pair each refuse by name); a REVOKED marker beside / instead of the overlay refuses through
      :func:`load_site_geometry`; then, when the overlay's probe record is REACHABLE from here (the probe cluster) and the caller
      handed ``record_check`` (``hpo_probe_status.probe_state_record`` — the SAME predicate the front door reads: READY iff complete,
      menu + verdict of the current generation AND the parity record it read is current; imported, never re-derived), the record
      must read :data:`PROBE_RECORD_READY` or the standup refuses with its state and detail: ``{"status": "verified", "pairs": [...],
      "record": {...} | None}``.

    Every refusal is a :class:`SiteGeometryError` naming the pair, the adapter, the status and the way back."""
    # the RESIDUAL-SLOT CARVE-OUT (the parity rule on the residual path, 2026-09-30): a residual carrier
    # (``residual: True`` — a rank-1 zero o_proj pair that only carries a steering payload: psr / vector / vector_all /
    # diffmean / static_r / psr_paired_write) is no LoRA the parity record could have folded — gated as (1, 'wo') it was
    # refused by name ("not one the serving probe folded") and a steered standup of this model could never launch. Its
    # certificate is the residual op's OWN: the attestation's ``residual_slot.armed`` with every served carrier listed,
    # the noop certificate and the positive controls that bring_up_native runs after launch and refuses on — stated here,
    # enforced there (native_standup.residual_slot_facts / run_positive_controls)
    residual = {n: dict(a) for n, a in adapters.items() if not a.get("zero") and a.get("residual")}
    residual_block = {"carriers": [{"adapter": n, "kind": a.get("kind"), "rank": int(a["rank"]), "set": str(a["module_set"])} for n, a in sorted(residual.items())],
                      "certificate": ("the residual op's own: attestation residual_slot.armed with every served carrier, the noop certificate "
                                      "and the positive controls after launch (bring_up_native refuses when any fails) — not LoRA parity")}
    steered = {n: dict(a) for n, a in adapters.items() if not a.get("zero") and not a.get("residual")}
    if not steered:
        if residual:
            return {"status": "residual_only", "pairs": [], "record": None, "residual": residual_block,
                    "note": "every served adapter is a residual-slot carrier: no LoRA pair to hold the parity record to; the residual op's "
                            "certificate (noop + positive controls after launch) is the gate"}
        return {"status": "unsteered", "pairs": [], "record": None, "residual": residual_block,
                "note": "no genuine adapter is served (the zero adapter alone / prompted-only): adapter parity gates nothing"}
    site = load_site_geometry(model_key)                       # a REVOKED marker (with or without the file) refuses here, by name
    if site is None:
        if model_key in ONBOARDED_MODEL_KEYS:
            pairs = sorted({(int(a["rank"]), str(a["module_set"])) for a in steered.values()})
            raise SiteGeometryError(
                f"{model_key}: {what} serves trained adapter(s) of {pairs} ({sorted(steered)}) but NO site geometry overlay is staged "
                f"for this onboarded model ({_site_overlay_where(model_key, None)}) — a steered standup of an onboarded model launches only "
                "from the serving probe's staged, parity-verified record (03_hpo.sh --phase probe fetches it; lib/slurm.sh "
                "slurm_stage_geometry verifies and stages it); refusing rather than serving adapters whose parity was never certified")
        return {"status": "no_overlay", "record": None,
                "pairs": [{"adapter": n, "rank": int(a["rank"]), "set": str(a["module_set"]), "status": None, "basis": None} for n, a in sorted(steered.items())],
                "note": "a sweep model with no site overlay: the pinned tables stand (the steered-parity gate applies to overlays)"}
    out_pairs: List[Dict[str, Any]] = []
    serving: Optional[Mapping[str, Any]] = dict(serving_venv) if serving_venv is not None else None
    for name, a in sorted(steered.items()):
        rank, module_set = int(a["rank"]), str(a["module_set"])
        rec = site.sets.get(module_set)
        if rec is None:
            raise SiteGeometryError(
                f"{model_key}: {what} serves adapter {name!r} of (rank {rank}, set {module_set!r}) but the site geometry overlay {site.path} "
                f"certifies no such set (certified: {sorted(site.sets)}) — its adapter parity is unknown; refusing by name")
        # THE DIFFERENCE ADAPTER: a lora_subtraction of two rank-r parents serves as rank 2r; its certificate is the fit-rank
        # pair's (see DIFFERENCE_ADAPTER_KIND). An odd served rank is not a stacked pair of two equal-rank parents: refused by name.
        pair_rank: int = rank
        certified_as: Optional[Dict[str, Any]] = None
        if str(a.get("kind") or "") == DIFFERENCE_ADAPTER_KIND:
            if rank % 2:
                raise SiteGeometryError(
                    f"{model_key}: {what} serves adapter {name!r} of kind {DIFFERENCE_ADAPTER_KIND!r} with an odd rank {rank} on set "
                    f"{module_set!r} — a LoRA subtraction exports as the rank-stacked pair of two equal-rank parents (rank 2r), so no "
                    "fit-rank pair (r, set) certifies it; refusing by name")
            pair_rank = rank // 2
            certified_as = {"pair_rank": pair_rank, "role": "difference",
                            "why": f"the probe certifies the pair (rank {pair_rank}, set {module_set!r}) only with its rank-{rank} difference adapter passed"}
        try:
            par = require_verified_adapter_parity(rec, pair_rank, what=f"{what} (adapter {name!r})")
        except SiteGeometryError as e:
            if certified_as is None:
                raise
            raise SiteGeometryError(
                f"{model_key}: {what} serves adapter {name!r}, the rank-{rank} {DIFFERENCE_ADAPTER_KIND} of two rank-{pair_rank} parents on "
                f"set {module_set!r}, held to the fit-rank pair (rank {pair_rank}, set {module_set!r}) whose certificate includes its "
                f"rank-{rank} difference — {e}") from e
        # THE CERTIFICATE VENV: the pair may be served only in the venv its parity was measured in
        if serving is None:
            serving = _sa.serving_venv_facts()
        venv_problem = certificate_venv_problem(par, serving)
        if venv_problem is not None:
            raise SiteGeometryError(
                f"{model_key}: {what} serves adapter {name!r} of (rank {rank}, set {module_set!r}) — {venv_problem} (overlay {site.path})")
        out_pairs.append({"adapter": name, "rank": rank, "set": module_set, "status": par["status"], "basis": par.get("basis"),
                          CERTIFICATE_VENV_KEY: par.get(CERTIFICATE_VENV_KEY), "certified_as": certified_as})
    record: Optional[Dict[str, Any]] = None
    rec_dir = (site.provenance or {}).get("probe_record")
    reachable = isinstance(rec_dir, str) and bool(rec_dir) and Path(rec_dir).is_dir()
    pairs_s = [(p["rank"], p["set"]) for p in out_pairs]
    if reachable and record_check is not None:
        # (a) the record is HERE (the probe cluster): the live predicate, the front door's own
        state, detail, info = record_check(Path(rec_dir))
        record = {"dir": rec_dir, "state": str(state), "parity_check": (info or {}).get("parity_check"),
                  "generation": (info or {}).get("generation"), "predicate": "live"}
        if str(state) != PROBE_RECORD_READY:
            raise SiteGeometryError(
                f"{model_key}: {what} serves {pairs_s} whose overlay {site.path} reads verified, but the probe "
                f"record it was folded from is reachable here and reads {state} (parity_check {record['parity_check']!r}) — {detail} — the "
                "overlay is not the record's current certification; refusing until the front door re-fetches and re-stages a READY record")
    elif reachable and record_check is None:
        record = {"dir": rec_dir, "state": None, "parity_check": None, "generation": None, "predicate": "opted_out"}
    else:
        # (b) the record is NOT reachable (the normal cross-cluster case: slurm_stage_geometry staged the overlay here, the record stays
        # on its home cluster): the ONLY admissible proof is the staging stamp the front door wrote AFTER running the
        # live predicate at staging time — it must carry the record's state (READY), the parity_check (none | current) and the
        # generation, and it must not predate the unit's submission when the caller knows it (a stamp older than a later revocation
        # window is what the REVOKED marker beside the overlay covers — load_site_geometry refused it above); (c) anything less refuses
        sv = dict(site.staging_verification or {})
        missing = [f for f in STAGING_PREDICATE_FIELDS if not sv.get(f)]
        if not sv:
            raise SiteGeometryError(
                f"{model_key}: {what} serves {pairs_s}; the overlay {site.path}'s probe record {rec_dir!r} is not reachable from here and the "
                f"overlay carries no {STAGING_VERIFICATION_KEY} block — no live predicate ran and no fenced proof stands in for it; refusing "
                "(lib/slurm.sh slurm_stage_geometry stages a verified copy)")
        if missing:
            raise SiteGeometryError(
                f"{model_key}: {what} serves {pairs_s}; the overlay {site.path}'s probe record {rec_dir!r} is not reachable from here and its "
                f"{STAGING_VERIFICATION_KEY} block lacks {missing} — a stamp that does not carry what the live predicate checked at staging "
                "time (the record's READY state, its parity_check) is no proof; refusing until the front door re-stages the overlay "
                "(slurm_stage_geometry re-stamps a copy whose block predates these fields)")
        if str(sv.get("record_state")) != PROBE_RECORD_READY or str(sv.get("parity_check")) not in ("none", "current"):
            raise SiteGeometryError(
                f"{model_key}: {what} serves {pairs_s}; the overlay {site.path} was staged while its record read {sv.get('record_state')!r} with "
                f"parity_check {sv.get('parity_check')!r} — not a READY record with a current parity basis; refusing")
        if str(sv.get("record_generation")) != str((site.provenance or {}).get(RESULT_GENERATION_KEY)):
            raise SiteGeometryError(
                f"{model_key}: {what}: the overlay {site.path}'s staging stamp verified generation {str(sv.get('record_generation'))[:16]}… but "
                f"the overlay is of {str((site.provenance or {}).get(RESULT_GENERATION_KEY))[:16]}… — the stamp is another copy's; refusing")
        # the fence is REQUIRED: a stamp is a proof only relative to a time this unit was submitted; a job that carries
        # no submission time cannot fence it, and an unfenced stamp accepts a copy staged before a later revocation that never reached
        # this cluster — refused by name, never silently unfenced
        if not stamp_not_before:
            raise SiteGeometryError(
                f"{model_key}: {what} serves {pairs_s}; the overlay {site.path}'s probe record {rec_dir!r} is not reachable from here, so only its "
                f"staging stamp ({sv.get('verified_utc')}) could stand in for the live predicate — but this job carries no submission time, so "
                "the stamp cannot be fenced against a later revocation; the front door must export SISTEER_UNIT_SUBMITTED_UTC (ISO-8601 Z, at "
                "submission) into the job's environment; refusing")
        if str(sv.get("verified_utc")) < str(stamp_not_before):
            raise SiteGeometryError(
                f"{model_key}: {what} serves {pairs_s}; the overlay {site.path}'s staging stamp ({sv.get('verified_utc')}) predates this unit's "
                f"submission ({stamp_not_before}) — the record may have been revoked and re-certified since; refusing until the front door "
                "re-stages the overlay")
        record = {"dir": rec_dir, "state": str(sv.get("record_state")), "parity_check": str(sv.get("parity_check")),
                  "generation": str(sv.get("record_generation")), "predicate": f"staging_stamp:{sv.get('verified_utc')}",
                  "stamp_not_before": stamp_not_before, "parity_record": sv.get("parity_record")}
    return {"status": "verified", "pairs": out_pairs, "record": record, "overlay": site.path, "residual": residual_block,
            "serving_venv": ({k: serving.get(k) for k in ("name", "variant", "campaign_venv", "prefix")} | {"exported": dict(serving.get("exported") or {})}
                             if serving is not None else None)}


def site_geometry_bytes(record: Mapping[str, Any]) -> bytes:
    """THE ONE serialization of an overlay record — sorted keys, two-space
    indent, ONE trailing newline, UTF-8. Every copy of a record (the probe's
    ``geometry_record.json``, the probe cluster's overlay, the site's copy, the
    staged copies on every other cluster) is written through this function, so
    the copies are byte-identical and a byte compare between them is meaningful
    (the record used to be written without the newline the overlay carried, and
    the staging compare aborted every first wave on that one byte)."""
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


#: the canonical CONTENT form a staging compare hashes: the parsed document re-dumped with sorted keys and no whitespace
#: (``separators=(",", ":")``), ASCII-escaped — identical for identical content whatever the file's formatting
SITE_GEOMETRY_CANONICAL_SEPARATORS = (",", ":")


def canonical_site_geometry_sha256(source: object) -> str:
    """The sha256 of a record's CANONICAL content: the JSON document (a path,
    JSON text/bytes, or a mapping) parsed and re-dumped with sorted keys and
    :data:`SITE_GEOMETRY_CANONICAL_SEPARATORS`. Two files with the same content
    and different formatting (indent, a trailing newline, key order) hash the
    same; any content difference hashes differently. ``reproduce/lib/slurm.sh``
    ``slurm_stage_geometry`` computes the SAME digest with a stdlib python3
    one-liner on both sides of its compare (a second belt), so formatting
    can never masquerade as tampering there."""
    if isinstance(source, Mapping):
        doc: Any = dict(source)
    else:
        if isinstance(source, (bytes, bytearray)):
            text = bytes(source).decode("utf-8")
        elif isinstance(source, str) and source.lstrip().startswith("{"):
            text = source
        else:
            text = Path(str(source)).read_text(encoding="utf-8")
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise SiteGeometryError(f"site geometry {source!s}: not JSON — {e}") from None
    if isinstance(doc, dict):
        doc.pop(STAGING_VERIFICATION_KEY, None)     # the staging stamp is outside the record's content (a staged copy compares equal to its site copy)
    canon = json.dumps(doc, sort_keys=True, separators=SITE_GEOMETRY_CANONICAL_SEPARATORS, ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def write_site_geometry_file(record: Mapping[str, Any], path: object) -> Path:
    """Validate ``record`` and write it ATOMICALLY (tmp + replace) to ``path``
    as :func:`site_geometry_bytes` — the probe's ``geometry_record.json``, the
    overlay copy, the site's copy all go through here, so they are byte-
    identical."""
    parse_site_geometry(record)
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(site_geometry_bytes(record))
    os.replace(tmp, path)
    return path


#: the staging-stamp fields a standup on a cluster that CANNOT reach the probe record accepts as the fenced proof of the live
#: predicate: the record's state at staging (READY), its parity_check (none | current), the parity record it
#: read (dir + generation), beside the three the block always carried
STAGING_PREDICATE_FIELDS: Tuple[str, ...] = ("record_state", "parity_check")


def staged_site_geometry_bytes(source: object, *, record_generation: str, verifier_host: str,
                               verified_utc: Optional[str] = None, record_state: Optional[str] = None,
                               parity_check: Optional[str] = None, parity_record: Optional[Mapping[str, Any]] = None) -> bytes:
    """The bytes ``slurm_stage_geometry`` puts on a target cluster: the site copy (a path or a mapping), validated, with
    the :data:`STAGING_VERIFICATION_KEY` block ``{verified_utc, record_generation, verifier_host}`` plus
    what the verifying status (``hpo_probe_status.py --record``, the live predicate) said at staging time: ``record_state``
    (READY), ``parity_check`` (none | current) and ``parity_record`` ({dir, generation} of the parity-pass record it read, or
    None), so a standup on a cluster that cannot reach the record has a FENCED proof of the predicate rather than a bare
    generation. Through the ONE serializer. Refuses a copy whose own ``result_generation`` stamp is not ``record_generation``
    (the verifying status said READY for another result set than this copy was folded over) or that carries no stamp at all."""
    import time as _time
    doc = json.loads(Path(str(source)).read_text(encoding="utf-8")) if not isinstance(source, Mapping) else dict(source)
    site = parse_site_geometry(doc, path=(None if isinstance(source, Mapping) else str(source)))
    stamp = (site.provenance or {}).get(RESULT_GENERATION_KEY)
    if not stamp:
        raise SiteGeometryError(f"site geometry {site.path or '<record>'}: no {RESULT_GENERATION_KEY} stamp — an unstamped copy cannot be "
                                "verified against its record and is never staged")
    if str(stamp) != str(record_generation):
        raise SiteGeometryError(f"site geometry {site.path or '<record>'}: its {RESULT_GENERATION_KEY} {str(stamp)[:16]}… is not the record's "
                                f"current generation {str(record_generation)[:16]}… — a stale site copy is never staged")
    out = dict(doc)
    block: Dict[str, Any] = {"verified_utc": verified_utc or _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                             "record_generation": str(record_generation), "verifier_host": str(verifier_host)}
    if record_state is not None:
        block["record_state"] = str(record_state)
    if parity_check is not None:
        block["parity_check"] = str(parity_check)
    if record_state is not None or parity_check is not None:
        # the predicate's reading travels whole: the parity record it read, or None when the verdict read none
        block["parity_record"] = ({"dir": parity_record.get("dir"), "generation": parity_record.get("generation")}
                                  if isinstance(parity_record, Mapping) else None)
    out[STAGING_VERIFICATION_KEY] = block
    return site_geometry_bytes(out)


def write_site_geometry(record: Mapping[str, Any], directory: Optional[object] = None) -> Path:
    """Write ONE validated overlay record atomically under the overlay
    directory as ``<model_key>.json`` (:func:`write_site_geometry_file` — the
    one serializer) and read it back. Refuses (:class:`SiteGeometryError`)
    when no directory is configured — a fold with nowhere to land is a
    problem to see, not a silent no-op."""
    site = parse_site_geometry(record)
    d = site_geometry_dir(directory)
    if d is None:
        raise SiteGeometryError(f"no site geometry directory: set {SITE_GEOMETRY_DIR_ENV} (or run under the front "
                                f"door's {SITE_DIR_ENV} / {SHARED_ROOT_ENV} job env)")
    path = write_site_geometry_file(record, d / f"{site.model_key}.json")
    # the read-back validates SHAPE and byte-equality only (parse, not load): the writer on the probe cluster must not trip on
    # the freshness checks a consumer runs — a REVOKED marker the front door left beside the old copy (re-check 2), or the
    # record's checkpoint — the re-fold is exactly what makes the copy current again; the next verified staging clears the marker
    back = parse_site_geometry(json.loads(path.read_text(encoding="utf-8")), path=path, model_key=site.model_key)
    if back is None or sorted(back.sets) != sorted(site.sets) or any(
            back.sets[k].lora_max_module_dim != site.sets[k].lora_max_module_dim
            or back.sets[k].budget.batched_token_budget != site.sets[k].budget.batched_token_budget for k in site.sets):
        raise SiteGeometryError(f"site geometry {path}: the record read back does not match what was written")
    return path


def lora_max_module_dim_for(model_key: str, module_set: Optional[str] = None) -> Optional[int]:
    """The widest ADAPTED / EXECUTING module dim ``D`` for ``model_key``: with a
    ``module_set``, the site overlay's record of THAT set when the model has an
    overlay (a set the file does not certify REFUSES by name); with NO set and
    an overlay, the :data:`UNSTEERED_MODULE_SET` row's (the zero adapter's
    class; refused by name when the record lacks it); else the pinned
    :data:`LORA_MAX_MODULE_DIM_BY_MODEL` row; else ``None`` (the model has no
    re-derivable kernel step — the callers refuse by name)."""
    if isinstance(model_key, str):
        # no set: the unsteered default fills only where the pinned table has nothing (a sweep model's pinned D stands)
        rec = (site_geometry_for(model_key, module_set) if module_set is not None
               else (unsteered_site_set(model_key) if LORA_MAX_MODULE_DIM_BY_MODEL.get(model_key) is None else None))
        if rec is not None:
            return int(rec.lora_max_module_dim)
    v = LORA_MAX_MODULE_DIM_BY_MODEL.get(model_key) if isinstance(model_key, str) else None
    return int(v) if v is not None else None


def certified_module_dims(model_key: str) -> Dict[str, int]:
    """Every certified module set's ``D`` for ``model_key`` from the overlay
    (``{}`` when the model has none) — what a record's kernel step may name."""
    site = load_site_geometry(model_key) if isinstance(model_key, str) else None
    return {} if site is None else {k: int(v.lora_max_module_dim) for k, v in site.sets.items()}


@dataclass(frozen=True)
class ModelRows:
    """Everything the sizing sites read about ONE model key, from whichever
    table holds it: the sweep's eight (``onboarded=False``) or the
    paper-companion onboarded models (``onboarded=True``). ``budget`` is
    ``None`` for a key with no row (a known but unmeasured model — the
    Llama-certified default applies, with the warning)."""

    model_key: str
    hf_id: str
    hybrid: bool
    onboarded: bool
    budget: Optional[ModelBudget]
    #: the site overlay's (model, module set) record that supplied ``budget`` (None: the built-in tables did)
    site: Optional[SiteGeometrySet] = None
    #: how that record was selected: :data:`MODULE_SET_SOURCE_NAMED` (the caller named the set) or
    #: :data:`MODULE_SET_SOURCE_UNSTEERED` (no set named: the zero adapter's class, :data:`UNSTEERED_MODULE_SET`)
    site_module_set_source: Optional[str] = None

    @property
    def provisional(self) -> bool:
        return self.budget is not None and self.budget.provisional


#: THE registry the sizing sites read: every known key -> its rows, the sweep's
#: eight (from the pinned tables) then the onboarded ones (from theirs) —
#: :data:`KNOWN_MODEL_KEYS` order. The two halves are joined HERE and nowhere
#: else (:func:`rows_for`, :func:`model_key_for`, :func:`budget_for` all read it).
_ROWS: Dict[str, ModelRows] = {
    **{k: ModelRows(model_key=k, hf_id=MODEL_HF_IDS[k], hybrid=_is_hybrid(k), onboarded=False,
                    budget=BUDGET_BY_MODEL.get(k)) for k in MODEL_KEYS},
    **{k: ModelRows(model_key=k, hf_id=ONBOARDED_MODEL_HF_IDS[k], hybrid=_is_hybrid(k),
                    onboarded=True, budget=ONBOARDED_BUDGET_BY_MODEL.get(k))
       for k in ONBOARDED_MODEL_KEYS},
}
assert tuple(_ROWS) == KNOWN_MODEL_KEYS


def rows_for(model_key: str, module_set: Optional[str] = None, *, served_lora: Optional[Tuple[int, int]] = None) -> ModelRows:
    """The ONE lookup over both kinds of key (:data:`_ROWS`): a paper key
    (:data:`MODEL_KEYS`) from the existing tables, an onboarded key
    (:data:`ONBOARDED_MODEL_KEYS`) from the onboarded ones; anything else is
    :class:`UnknownModelKey` (loud, never a silent default for a typo).
    ``module_set`` names the SELECTED LoRA module set: when the model has a
    site geometry overlay, THAT set's certified budget replaces the table's
    (``ModelRows.site`` carries the record); a set the overlay does not
    certify refuses by name (:class:`SiteGeometryError`). With NO set (an
    UNSTEERED caller) and a site record for the model, the record's
    :data:`UNSTEERED_MODULE_SET` row — the class of the zero adapter an
    unsteered native standup registers — sizes it, and a record that does not
    certify that set refuses by name (:func:`unsteered_site_set`); only a model
    with no record at all falls to the built-in tables (for an onboarded model
    the provisional / default row, which every padded standup refuses)."""
    row = _ROWS.get(model_key) if isinstance(model_key, str) else None
    if row is None:
        raise UnknownModelKey(
            f"unknown model key {model_key!r}; known model keys: {list(KNOWN_MODEL_KEYS)}")
    # the budget is read from its TABLE at call time (BUDGET_BY_MODEL for the sweep's keys, ONBOARDED_BUDGET_BY_MODEL for
    # the onboarded ones), never from the import-time snapshot in _ROWS: the tables are the documented seam
    # (test_model_budget monkeypatches BUDGET_BY_MODEL), and a row frozen at import would also freeze out the overlay below
    table = ONBOARDED_BUDGET_BY_MODEL if row.onboarded else BUDGET_BY_MODEL
    row = replace(row, budget=table.get(model_key))
    # the SITE geometry overlay wins over the built-in row when the serving probe has folded a measured row for this
    # model AND the caller names the selected module set — read at CALL time, never cached at import: the probe
    # writes it while other processes are already up
    if module_set is not None:
        site_set = site_geometry_for(model_key, module_set)
        if site_set is not None:
            row = replace(row, budget=site_set.budget, site=site_set, site_module_set_source=MODULE_SET_SOURCE_NAMED)
    elif row.budget is None or row.budget.provisional:
        # NO set named — an UNSTEERED caller (every production standup's budget_for(model_key)) — on a model whose built-in row is a
        # GUESS (a provisional ceiling) or absent: the record's conservative row when the model has a record; refused by name
        # when the record does not certify the zero adapter's class; the built-in guess only when there is no record at all (2026-09-30:
        # the u0 wave sized from the provisional row beside a staged record). A MEASURED built-in row (the sweep's models) stands for a
        # no-set caller as it always has — the records those pins were built on — while a named set still takes the overlay's row.
        site_set = unsteered_site_set(model_key, served_lora=served_lora)
        if site_set is not None:
            row = replace(row, budget=site_set.budget, site=site_set, site_module_set_source=MODULE_SET_SOURCE_UNSTEERED)
    return row


def _check_window_ceilings(table: Dict[str, ModelBudget]) -> None:
    """Every :data:`HYBRID_WINDOW_SEQS_CEILINGS` row must LOWER the S its model's
    budget row buys at that window (the quotient under the hybrid cap) — a
    ceiling exists to shrink a number the engine refused; one at or above the
    quotient is stale (the budget was re-measured underneath it) and the import
    refuses rather than let it read as live evidence."""
    for (key, L), c in HYBRID_WINDOW_SEQS_CEILINGS.items():
        if (key, L) != (c.model_key, c.max_model_len):
            raise ValueError(f"HYBRID_WINDOW_SEQS_CEILINGS[{(key, L)!r}] names {(c.model_key, c.max_model_len)!r}")
        entry = table.get(key)
        if entry is None:
            raise ValueError(f"{key}@{L}: a window ceiling for a model with no budget row")
        if float(entry.basis.gpu_memory_utilization) != float(c.gpu_memory_utilization):
            raise ValueError(f"{key}@{L}: the window ceiling was measured at gpu_memory_utilization {c.gpu_memory_utilization}, the budget row "
                             f"states {entry.basis.gpu_memory_utilization} — not comparable; re-measure the ceiling under the row's configuration")
        if int(c.batched_token_budget) != int(entry.batched_token_budget):
            raise ValueError(f"{key}@{L}: the window ceiling was measured under the {c.batched_token_budget:,}-token profile, the budget row is "
                             f"{entry.batched_token_budget:,} — the budget moved underneath it; drop or re-measure the row")
        quotient = min(identity_seqs(L, budget=entry.batched_token_budget), HYBRID_SEQS_CAP)
        if c.max_num_seqs >= quotient:
            raise ValueError(
                f"{key}@{L}: the window ceiling {c.max_num_seqs} does not LOWER the budget's {quotient} "
                f"({entry.batched_token_budget:,} // {L}, hybrid cap {HYBRID_SEQS_CAP}) — drop the row from "
                "HYBRID_WINDOW_SEQS_CEILINGS if the budget has caught up")


_check_window_ceilings({**BUDGET_BY_MODEL, **ONBOARDED_BUDGET_BY_MODEL})   # an onboarded key's budget row lives in ONBOARDED_BUDGETS (X7)


@dataclass(frozen=True)
class ResolvedBudget:
    """What ``budget_for`` handed back and WHERE it came from: ``source`` is the
    table entry's own (:data:`MEASURED_BUDGET_SOURCE`, or
    :data:`PROVISIONAL_BUDGET_SOURCE` for a conservative ceiling awaiting a
    measurement — ``entry`` is that model's) or :data:`DEFAULT_BUDGET_SOURCE`
    for the Llama-certified fallback (``entry`` is llama8b's, whatever
    ``model_key`` says).

    :attr:`per_model` — NOT :attr:`measured` — is the predicate for "size from
    this entry": a provisional ceiling is not measured, but sizing it from the
    Llama-certified default would hand back an S ABOVE the ceiling."""

    model_key: Optional[str]
    batched_token_budget: int
    source: str
    entry: ModelBudget
    #: the site overlay's module set that supplied ``entry`` (None: the built-in tables) and how it was selected
    #: (:data:`MODULE_SET_SOURCE_NAMED` / :data:`MODULE_SET_SOURCE_UNSTEERED`) — so every record of a launch says which row sized it
    module_set: Optional[str] = None
    module_set_source: Optional[str] = None

    @property
    def measured(self) -> bool:
        return self.source == MEASURED_BUDGET_SOURCE

    @property
    def provisional(self) -> bool:
        """A conservative ceiling chosen from an observed failure, not a swept
        number (:data:`PROVISIONAL_BUDGET_SOURCE`)."""
        return self.source == PROVISIONAL_BUDGET_SOURCE

    @property
    def per_model(self) -> bool:
        """The MODEL's own row supplied this budget (measured or provisional),
        rather than the Llama-certified default."""
        return self.source in PER_MODEL_BUDGET_SOURCES

    def as_dict(self) -> dict:
        return {"model_key": self.model_key,
                "batched_token_budget": self.batched_token_budget,
                "source": self.source, "certified": self.entry.certified,
                "provisional": self.provisional,
                "entry_model_key": self.entry.model_key,
                "module_set": self.module_set, "module_set_source": self.module_set_source}


def budget_for(model_key: Optional[str] = None, module_set: Optional[str] = None, *, served_lora: Optional[Tuple[int, int]] = None) -> ResolvedBudget:
    """The batched-token budget to size a launch for ``model_key`` — with
    ``module_set``, the SELECTED set's certified row from the site geometry
    overlay when the model has one (:func:`rows_for`; a set the overlay does
    not certify refuses by name); with NO set and an overlay, the
    :data:`UNSTEERED_MODULE_SET` row (the zero adapter's class an unsteered
    standup registers; a record without it refuses by name — never the
    provisional row beside a record, 2026-09-30). ``module_set`` /
    ``module_set_source`` on the result say which row sized the launch.

    - a MEASURED model (in :data:`BUDGET_BY_MODEL` — the paste block: the 2026-09-04
      sweep, with llama8b's row re-measured on the lens stack by the
      serving gate, 475,136, 2026-09-10, and qwen35_4b's re-measured at the MATH-500 window
      on the lens stack, 741,888 = 42 x 17,664, 2026-09-13): its entry, ``source="measured"``, no warning;
    - a model whose row is a PROVISIONAL ceiling (:data:`_PROVISIONAL_BUDGETS`,
      2026-09-06: today llama3b and qwen35_9b — llama8b's ceiling was replaced by its
      2026-09-10 measurement, qwen35_4b's lifted 2026-09-11 and re-measured 2026-09-13): its entry,
      ``source="provisional-conservative"``, and a WARNING naming the ceiling
      and the failure it was chosen from — the number is a floor on nobody's
      throughput and is meant to be re-measured, so it never passes silently;
    - a known but UNMEASURED model (a key added to :data:`MODEL_KEYS` before its
      sweep): the Llama-certified :data:`BATCHED_TOKEN_BUDGET` with
      ``source="default_llama_certified"`` and a WARNING naming the model — the
      default undersells smaller-KV models, so a launcher running on it is
      leaving concurrency on the table until the sweep measures that model;
    - ``None`` (no model named): the same default, silently — the pre-per-model
      behaviour of every existing caller;
    - a key outside :data:`MODEL_KEYS`: :class:`UnknownModelKey` (loud, never a
      silent default for a typo)."""
    if model_key is None:
        return ResolvedBudget(None, BATCHED_TOKEN_BUDGET, DEFAULT_BUDGET_SOURCE,
                              _LLAMA8B_CERTIFIED)
    # ONE lookup over the sweep's keys and the onboarded ones (rows_for refuses
    # an unknown key); an onboarded key's row comes from ONBOARDED_BUDGETS
    rows = rows_for(model_key, module_set, served_lora=served_lora)
    entry = rows.budget
    if entry is None:
        log.warning(
            "[geometry] model %r has NO measured batched-token budget — using the "
            "Llama-8B-certified default %s (source=%s), which undersells smaller-KV "
            "models; measure it with scripts/served_decode_sweep.py --phase budget "
            "and paste the analyzer's block into geometry.py (an ONBOARDED model: the FD 24 "
            "serving probe's verdict fold writes the per-set rows to the site geometry overlay, %s)",
            model_key, f"{BATCHED_TOKEN_BUDGET:,}", DEFAULT_BUDGET_SOURCE, _site_overlay_where(model_key, module_set))
        return ResolvedBudget(model_key, BATCHED_TOKEN_BUDGET, DEFAULT_BUDGET_SOURCE,
                              _LLAMA8B_CERTIFIED)
    if entry.provisional:
        log.warning(
            "[geometry] model %r is on a PROVISIONAL batched-token budget %s "
            "(source=%s): a conservative CEILING chosen from an observed engine-init "
            "failure, NOT a measurement — %s. Re-measure it with "
            "scripts/served_decode_sweep.py --phase budget ON THIS STACK and drop the "
            "row from geometry._PROVISIONAL_BUDGETS (an ONBOARDED model: the FD 24 serving "
            "probe's verdict fold writes the measured row to the site geometry overlay, "
            "%s)",
            model_key, f"{entry.batched_token_budget:,}", PROVISIONAL_BUDGET_SOURCE,
            entry.basis.source, _site_overlay_where(model_key, module_set))
    return ResolvedBudget(model_key, entry.batched_token_budget, entry.source, entry,
                          module_set=(rows.site.module_set if rows.site is not None else None), module_set_source=rows.site_module_set_source)


def describe_budgets() -> str:
    """The per-model budget table as markdown: every canonical model key, its
    budget (measured or the default), the S it buys at the Petri and budget-probe
    windows, and the basis columns."""
    cols = ("model_key", "budget", "source", "certified", "hybrid_cap", "S@16384", "S@8192",
            "S@1536", "probe LxS", "util", "kv_tokens", "first_refused", "job", "measured_on")
    rows = []
    for key in MODEL_KEYS:
        entry = BUDGET_BY_MODEL.get(key)
        budget = entry.batched_token_budget if entry is not None else BATCHED_TOKEN_BUDGET
        source = entry.source if entry is not None else DEFAULT_BUDGET_SOURCE
        b = entry.basis if entry is not None else None
        hybrid = key in HYBRID_MODEL_KEYS

        def seqs(L: int) -> str:
            raw = identity_seqs(L, budget=budget)
            return str(min(raw, HYBRID_SEQS_CAP) if hybrid else raw)

        rows.append((
            key, f"{budget:,}", source,
            ("yes" if entry is not None and entry.certified else "no"),
            (str(HYBRID_SEQS_CAP) if hybrid else "-"),
            seqs(PETRI_MAX_MODEL_LEN), seqs(8192), seqs(1536),
            (f"{b.max_model_len} x {b.max_num_seqs}" if b else "-"),
            (f"{b.gpu_memory_utilization:.2f}" if b else "-"),
            (_cell(b.kv_tokens_reported) if b else "-"),
            (_cell(b.first_refused) if b else "-"),
            (_cell(b.probe_job_id) if b else "-"),
            (b.measured_on if b else "-"),
        ))
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    out.append("")
    out.append(f"default = the Llama-8B-certified {BATCHED_TOKEN_BUDGET:,} "
               f"(source={DEFAULT_BUDGET_SOURCE}; the no-model default and the per-eval "
               "table's sizing); measured entries come from scripts/served_decode_sweep.py "
               "--phase budget via scripts/geom_sweep_analyze.py (2026-09-04, all eight; llama8b's row "
               "re-measured 2026-09-10 by the sdfbase-llama8b serving gate's readiness ladders on the "
               "lens stack, jobs 476439 / 476440: 475,136 = 29 x 16384, in the same paste block); "
               f"a {PROVISIONAL_BUDGET_SOURCE} row is a CEILING chosen from an observed "
               "engine-init failure on the lens stack (2026-09-06), not a measurement — see "
               "geometry._PROVISIONAL_BUDGETS and docs/served_geometry_p99.md. "
               f"hybrid_cap = the per-slot-state cap {HYBRID_SEQS_CAP} on S "
               f"(ModelBudget.seqs_at) for {', '.join(HYBRID_MODEL_KEYS)}: S@L columns show "
               "it applied. Throughput per window: scripts/geom_sweep_analyze.py --doc regenerates the table from the sweep results.")
    if HYBRID_WINDOW_SEQS_CEILINGS:
        out.append("")
        out.append("Measured per-window ceilings on S (HYBRID_WINDOW_SEQS_CEILINGS, applied by ModelBudget.seqs_at "
                   "after the quotient and the hybrid cap — a hybrid cache's token capacity is not S x L):")
        for (key, L), c in sorted(HYBRID_WINDOW_SEQS_CEILINGS.items()):
            out.append(f"- {key} @ {L}: S {c.max_num_seqs} (the allocator's {c.kv_max_concurrency:.2f}x = "
                       f"{c.kv_tokens_reported:,} KV tokens at util {c.gpu_memory_utilization:.2f}; job "
                       f"{c.probe_job_id}, {c.measured_on}): {c.note}")
    return "\n".join(out)


class UnknownEvalGeometry(KeyError):
    """``geometry_for`` was asked for an eval (or eval:task) the table does not
    know; the message lists the known keys (loud, never a silent default)."""


class GeometryOverBudget(ValueError):
    """An OVERRIDDEN geometry's ``max_num_seqs x max_model_len`` exceeds
    :data:`BATCHED_TOKEN_BUDGET` and the caller did not opt in
    (``allow_over_budget``). Above the budget the launch has a bigger
    profiling forward and a bigger KV reservation than the certified
    footprint; the 32 x 16384 negative result is what that looks like."""


class GeometryIdentityBroken(ValueError):
    """An OVERRIDE left ``max_num_seqs != identity_seqs(max_model_len)`` on a
    table-resolved launch (``strict_identity``): a non-identity pair is either
    over the budget or a silently narrower batch than the window buys; a
    launcher that wants it says so (``allow_over_budget`` waives this too)."""


class GeometryRequiresRecertification(ValueError):
    """A launcher asked to serve an INFORMATIONAL entry (``petri:<model>``):
    Petri's window is certified by the fidelity smoke and pinned by
    ``test_petri_geometry_cert``; those move together with any adoption."""


class GeometryCapExceedsBasis(ValueError):
    """A TABLE window was asked to serve a driver whose completion cap exceeds the cap the
    entry was MEASURED under (``GeometryBasis.requested_max_tokens``). The window rule for a
    multi-turn workload is a measured maximum on a COMPLETE store at the cap being served;
    a p99 window measured under a smaller cap is not that measurement (the 8,704
    ``multiturn_attacks`` entry at a 4,096 cap: 129 distinct clamped ``max_tokens`` on one
    PAIR cell, the smallest 69, 3 goals killed by PromptOverWindow)."""


class GeometryProvisional(ValueError):
    """A launcher asked :func:`resolve_geometry` for a PROVISIONAL entry (a
    tinycap ``capability:*`` sub-entry whose pass-1 ledger has not been
    pooled yet — or, ``capability:tiny_prompted``, whose window is only an
    ANALYTIC derivation): there is no measured window to serve. The
    measurement pass runs at the certified footprint via
    :func:`pass1_geometry`; the measured entry lands (``provisional``
    dropped) once Evals accepts the proposal
    ``scripts/tinycap_geometry_measure.py`` emits."""


class GeometryInformational(ValueError):
    """A launcher asked :func:`resolve_geometry` for an INFORMATIONAL entry
    (``capability:grader``): it is measured for the record — the
    model-grader traffic's own p99, kept OUT of the target pools it had
    diluted — and no launcher serves it on its own; the grader rides the
    target's server at the target's window."""


@dataclass(frozen=True)
class GeometryBasis:
    """Why this window: the measurement behind it (``docs/served_geometry_p99.md``).

    ``p99_total`` / ``max_total`` are prompt+completion tokens; ``p99_prompt``
    prompt tokens alone; ``requested_max_tokens`` the completion cap the
    driver sends. ``clamp_rate_at_L`` is the share of measured calls with
    prompt + requested max_tokens > the entry's L (the calls the client's
    once-only clamp handles at that window); ``frac_over_L`` the share with
    prompt + completion > L. ``clamp_rate_at_total_sizing`` is the clamp rate
    at the total-p99 window — the number the sizing rule branches on (None
    when it was not measured separately; then ``clamp_rate_at_L`` stands in,
    which for a total-sized entry is the same window). ``families_measured``
    are the model families pooled; ``source`` names the records and
    ``measured_on`` the ISO date (per entry — or :data:`UNMEASURED` on a
    provisional one). ``n_rejected`` (optional; the tinycap ledgers) is the
    number of the ``n_calls`` that vLLM 400-rejected at the measurement
    window: those calls carry no ``usage``, so their prompt length comes
    from the ledger's pre-tokenized count and they enter the clamp rate and
    ``frac_over_L`` like every other attempted call — kept as its own field
    so a basis can never hide a rejection-heavy tail inside its p99.
    ``n_capless`` (optional; the tinycap ledgers, Evals' verdict 2026-09-04)
    is the number of the ``n_calls`` sent with NO ``max_tokens``: a cap-less
    call cannot clamp (vLLM sizes its completion to the room left), so it
    is in the prompt / total p99 pools but OUTSIDE the clamp basis — the
    clamp rates are over the ``n_calls - n_capless`` capped calls, and a
    pool with no capped call has a clamp rate of 0 by construction.
    ``analytic`` (optional) marks a PROVISIONAL entry sized by DERIVATION
    rather than from pooled calls: the string is the derivation, in
    words + numbers; ``p99_prompt`` then carries the derived prompt bound,
    everything measured from calls stays None / 0, ``measured_on`` stays
    :data:`UNMEASURED`, and the window is held to the prompt+cap floor of
    that bound (:data:`SIZED_BY_ANALYTIC`). A measured entry never carries
    it (:func:`sizing_problem` refuses the combination).
    ``max_prompt`` (optional) is the LONGEST measured prompt (tokens);
    ``sized_to_max_prompt`` marks a MEASURED entry whose window is held to
    ``max_prompt + requested_max_tokens`` rather than the p99
    (:data:`SIZED_BY_MAX_PROMPT_PLUS_MAX`) — for a workload whose driver
    refuses a clamped cap-bound sample, so every prompt must fit at the cap;
    such a basis measures a clamp rate of exactly 0 at L, and a provisional
    entry may carry neither (both are measured quantities)."""

    n_calls: int
    p99_total: Optional[int]
    max_total: Optional[int]
    p99_prompt: Optional[int]
    requested_max_tokens: int
    clamp_rate_at_L: Optional[float]
    frac_over_L: Optional[float]
    families_measured: Tuple[str, ...]
    measured_on: str
    source: str
    note: str = ""
    clamp_rate_at_total_sizing: Optional[float] = None
    n_rejected: Optional[int] = None
    n_capless: Optional[int] = None
    analytic: Optional[str] = None
    max_prompt: Optional[int] = None
    sized_to_max_prompt: bool = False

    def __post_init__(self) -> None:
        if self.max_prompt is not None:
            if (isinstance(self.max_prompt, bool) or not isinstance(self.max_prompt, int)
                    or self.max_prompt < 1):
                raise ValueError(f"max_prompt must be a positive int, got {self.max_prompt!r}")
            if self.p99_prompt is not None and self.max_prompt < self.p99_prompt:
                raise ValueError(
                    f"max_prompt {self.max_prompt} is below p99_prompt {self.p99_prompt}: the "
                    "longest prompt cannot be shorter than the 99th percentile")
        if self.sized_to_max_prompt and self.max_prompt is None:
            raise ValueError("sized_to_max_prompt needs max_prompt (the longest measured prompt)")
        if self.sized_to_max_prompt and self.analytic is not None:
            raise ValueError("sized_to_max_prompt is a MEASURED branch; an analytic basis "
                             "derives a bound instead")
        for name in ("n_rejected", "n_capless"):
            v = getattr(self, name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative int, got {v!r}")
            if v > self.n_calls:
                raise ValueError(
                    f"{name} {v} exceeds n_calls {self.n_calls}: "
                    f"{'rejected' if name == 'n_rejected' else 'cap-less'} calls are a SUBSET "
                    "of the attempted calls in the basis")
        if self.analytic is not None and (not isinstance(self.analytic, str)
                                          or not self.analytic.strip()):
            raise ValueError("analytic must be the derivation (a non-empty string) or None")

    @property
    def measured(self) -> bool:
        """Whether any p99 or the prompt max was MEASURED from calls (False on an unmeasured
        or an analytic provisional basis — an analytic ``p99_prompt`` is a
        derived bound, not a measurement)."""
        if self.analytic is not None:
            return False
        return (self.p99_total is not None or self.p99_prompt is not None
                or self.max_prompt is not None)

    @property
    def n_capped(self) -> Optional[int]:
        """``n_calls - n_capless`` — the clamp basis — when ``n_capless`` is
        known, else None (every call of a pre-tinycap basis carried a cap)."""
        if self.n_capless is None:
            return None
        return self.n_calls - self.n_capless

    @property
    def rejection_rate(self) -> Optional[float]:
        """``n_rejected / n_calls`` when both are known, else None."""
        if self.n_rejected is None or not self.n_calls:
            return None
        return self.n_rejected / self.n_calls

    @property
    def sized_by(self) -> str:
        """Which branch of the sizing rule applies to this basis."""
        if self.analytic is not None:
            return SIZED_BY_ANALYTIC
        if not self.measured:
            return SIZED_BY_UNMEASURED
        if self.sized_to_max_prompt:
            return SIZED_BY_MAX_PROMPT_PLUS_MAX
        rate = (self.clamp_rate_at_total_sizing
                if self.clamp_rate_at_total_sizing is not None
                else self.clamp_rate_at_L)
        if self.p99_total is None or (rate is not None and rate > CLAMP_RATE_CEILING):
            return SIZED_BY_PROMPT_PLUS_MAX
        return SIZED_BY_TOTAL

    @property
    def floor(self) -> Optional[int]:
        """The sizing rule's floor for this basis (a block multiple), or None
        when the branch's inputs are missing."""
        branch = self.sized_by
        if branch == SIZED_BY_UNMEASURED:
            return None
        if branch == SIZED_BY_TOTAL:
            return next_block(self.p99_total)
        if branch == SIZED_BY_MAX_PROMPT_PLUS_MAX:
            return next_block(self.max_prompt + self.requested_max_tokens)
        # the prompt+cap branch, measured or analytic
        if self.p99_prompt is None:
            return None
        return next_block(self.p99_prompt + self.requested_max_tokens)


@dataclass(frozen=True)
class ServedGeometry:
    """One eval's (or eval:task's) served geometry + its basis.

    ``certified``: the composer's certified window (Petri) — held to the
    budget and the clamp ceiling but exempt from the p99 floor, because it is
    deliberately BELOW its p99 (the failure ledger + the 32k re-run proxies
    own the tail). ``requires_recertification``: informational only; never
    resolved by a launcher. ``provisional``: registered but UNMEASURED (the
    tinycap sub-entries before their pass-1 ledgers are pooled) — its L x S
    is the certified footprint the measurement pass runs at, its basis is
    empty by construction, and :func:`resolve_geometry` refuses it
    (:class:`GeometryProvisional`); :func:`pass1_geometry` is the only way
    to serve it — a provisional entry with an ANALYTIC basis carries a
    derived window instead of the footprint (see ``GeometryBasis.analytic``)
    but is still refused and still measured at the footprint.
    ``informational``: measured for the RECORD and consumed by no launcher
    (``capability:grader`` — the model-grader traffic, a workload nobody
    stands a server up for on its own); :func:`resolve_geometry` refuses it
    (:class:`GeometryInformational`). ``seqs_override`` is a stated reason S
    is not ``identity_seqs(L)`` — unused by every current entry."""

    eval: str
    max_model_len: int
    max_num_seqs: int
    basis: GeometryBasis
    task: Optional[str] = None
    certified: bool = False
    requires_recertification: bool = False
    seqs_override: Optional[str] = None
    provisional: bool = False
    informational: bool = False

    def __post_init__(self) -> None:
        L, S = self.max_model_len, self.max_num_seqs
        for name, v in (("max_model_len", L), ("max_num_seqs", S)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{self.key}: {name} must be a positive int, got {v!r}")
        if self.seqs_override is None and S != identity_seqs(L):
            raise ValueError(
                f"{self.key}: max_num_seqs {S} != identity_seqs({L}) = "
                f"{identity_seqs(L)} and no seqs_override reason is given")
        if self.over_budget and not self.seqs_override:
            raise ValueError(
                f"{self.key}: {S} x {L} = {self.batched_tokens} exceeds the "
                f"budget {BATCHED_TOKEN_BUDGET} with no seqs_override reason")
        problem = sizing_problem(self)
        if problem is not None:
            raise ValueError(f"{self.key}: {problem}")

    @property
    def key(self) -> str:
        return self.eval if self.task is None else f"{self.eval}:{self.task}"

    @property
    def batched_tokens(self) -> int:
        return self.max_num_seqs * self.max_model_len

    @property
    def over_budget(self) -> bool:
        return self.batched_tokens > BATCHED_TOKEN_BUDGET

    @property
    def is_identity(self) -> bool:
        return self.max_num_seqs == identity_seqs(self.max_model_len)

    @property
    def sized_by(self) -> str:
        return self.basis.sized_by

    def as_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        d["batched_tokens"] = self.batched_tokens
        d["budget"] = BATCHED_TOKEN_BUDGET
        d["over_budget"] = self.over_budget
        d["sized_by"] = self.sized_by
        d["sizing_floor"] = self.basis.floor
        return d


def sizing_problem(g: ServedGeometry) -> Optional[str]:
    """The sizing rule as a verdict: ``None`` when ``g``'s window satisfies it,
    else the violation, named. Checks, in order: L is a block multiple; the
    clamp rate at L is at most :data:`CLAMP_RATE_CEILING` (a window whose
    clamp fires on most calls is refused — total-p99 sizing of an arm that
    runs to its cap is exactly that); L is at least the rule's floor for the
    branch the basis lands in; and L is within :data:`SIZING_HEADROOM` of
    that floor. A ``certified`` entry (Petri) is exempt from the floor and
    headroom checks — its window is deliberately below its p99 — and an
    informational ``requires_recertification`` entry may leave the clamp rate
    unmeasured. A ``provisional`` entry must carry NOTHING measured from
    calls (no calls, no clamp rate, no tail fraction, ``measured_on`` =
    :data:`UNMEASURED`) and then EITHER an empty basis at the certified
    footprint (unmeasured: what its measurement pass runs at) OR an
    ``analytic`` basis — the derivation stated, the derived prompt bound in
    ``p99_prompt``, no ``p99_total`` / ``max_total`` (completions were not
    derived), and a window on the prompt+cap floor of that bound within the
    headroom: a provisional entry with measured numbers in it is a measured
    entry wearing the wrong flag, and a non-provisional entry with an
    ``analytic`` basis is a guess wearing a measurement's flag — both
    refused. An ``informational`` entry is measured (never provisional). A
    ``sized_to_max_prompt`` entry (:data:`SIZED_BY_MAX_PROMPT_PLUS_MAX`) is
    measured, holds every prompt at the cap by construction, and must
    therefore carry a clamp rate of EXACTLY 0 at L — a non-zero rate says
    the ``max_prompt`` it was sized from is not the longest prompt."""
    b = g.basis
    L = g.max_model_len
    if L % VLLM_BLOCK_SIZE:
        return (f"max_model_len {L} is not a multiple of the vLLM block size "
                f"{VLLM_BLOCK_SIZE}")
    if b.analytic is not None and not g.provisional:
        return ("an analytic basis (a derived window, no pooled calls) is provisional by "
                "definition — flag the entry provisional, or measure it")
    if g.provisional:
        if g.informational:
            return "an informational entry is measured for the record; it cannot be provisional"
        if (b.n_calls or b.clamp_rate_at_L is not None or b.frac_over_L is not None
                or b.n_rejected or b.n_capless or b.clamp_rate_at_total_sizing is not None
                or b.max_prompt is not None):
            return ("a provisional entry carries measured basis fields — register it as "
                    "MEASURED (drop provisional) once its ledger is pooled, never both")
        if b.measured_on != UNMEASURED:
            return f"a provisional entry's measured_on must be {UNMEASURED!r}, got {b.measured_on!r}"
        if b.analytic is None:
            if b.measured:
                return ("a provisional entry carries measured basis fields — register it as "
                        "MEASURED (drop provisional) once its ledger is pooled, never both")
            if (L, g.max_num_seqs) != (PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS):
                return (f"a provisional entry's window must be the certified footprint "
                        f"{PETRI_MAX_MODEL_LEN} x {vs.DEFAULT_MAX_NUM_SEQS} (what its "
                        f"measurement pass runs at), got {L} x {g.max_num_seqs}")
            return None
        # ANALYTIC provisional: a derived prompt bound, the prompt+cap floor
        if b.p99_prompt is None:
            return ("an analytic provisional entry must carry its derived prompt bound as "
                    "p99_prompt (the derivation in `analytic` names how it was composed)")
        if b.p99_total is not None or b.max_total is not None:
            return ("an analytic basis derives a PROMPT bound only; p99_total / max_total are "
                    "measured quantities and must stay None")
        floor = b.floor
        if L < floor:
            return (f"max_model_len {L} is below the analytic floor {floor} "
                    f"({b.sized_by}: {b.p99_prompt} + {b.requested_max_tokens})")
        if L - floor > SIZING_HEADROOM:
            return (f"max_model_len {L} is {L - floor} tokens above the analytic floor "
                    f"{floor} — more than the {SIZING_HEADROOM}-token headroom allowance")
        return None
    if b.clamp_rate_at_L is None:
        if not g.requires_recertification:
            return "clamp_rate_at_L is unmeasured (only an informational entry may omit it)"
    elif b.sized_to_max_prompt and b.clamp_rate_at_L != 0:
        return (f"clamp rate {b.clamp_rate_at_L:.4f} at max_model_len {L} on a window sized to "
                f"hold EVERY prompt at the cap ({b.sized_by}: max_prompt {b.max_prompt} + "
                f"{b.requested_max_tokens}) — a clamped call means max_prompt is not the "
                "longest prompt; re-measure or size by the p99 branches")
    elif b.clamp_rate_at_L > CLAMP_RATE_CEILING:
        return (f"clamp rate {b.clamp_rate_at_L:.4f} at max_model_len {L} exceeds the "
                f"{CLAMP_RATE_CEILING:.0%} ceiling — the clamp would fire on more than a "
                f"tail; size to p99(prompt) + max_tokens "
                f"({b.p99_prompt} + {b.requested_max_tokens}) instead")
    if g.certified:
        return None
    floor = b.floor
    if floor is None:
        return (f"cannot size: the {b.sized_by} branch needs "
                f"{'p99_total' if b.sized_by == SIZED_BY_TOTAL else 'p99_prompt'}")
    if L < floor:
        return (f"max_model_len {L} is below the sizing floor {floor} "
                f"({b.sized_by} branch)")
    if L - floor > SIZING_HEADROOM:
        return (f"max_model_len {L} is {L - floor} tokens above the sizing floor "
                f"{floor} ({b.sized_by} branch) — more than the {SIZING_HEADROOM}-token "
                "headroom allowance; this is not a p99-sized window")
    return None


# --------------------------------------------------------------------------- #
# The 4,096-cap multi-turn windows: explicit, per attack, from measured maxima  #
# --------------------------------------------------------------------------- #

#: THE MULTI-TURN SERVING WINDOWS OF THE CANONICAL 4,096-CAP INSTRUMENT, passed
#: EXPLICITLY by scripts/sdfbase/full22/target_attacks*.sbatch and never left to the
#: ``multiturn_attacks`` table entries (sized under a 2,048-token cap on 512-cap-era
#: prompts; :class:`GeometryCapExceedsBasis` refuses them for a 4,096 driver).
#:
#: THE WINDOW RULE: a multi-turn window is sized from a MEASURED MAXIMUM on a COMPLETE
#: store at the cap being served -- never a percentile, never a partial store -- because
#: long conversations finish last, so every early reading is a floor. The same quantity
#: read 8,731 -> 16,428 -> 34,361 -> 41,433 as its store filled. Crescendo's measured
#: maximum context (prompt + reply) over the four complete 259-goal stores is 41,433
#: (glm46vflash; gemma 16,428, nemotron 15,462, qwen 8,597 sampled / 41,311 greedy),
#: which is why 49,152 and not 32,768 (0 clamped calls on every crescendo cell). PAIR's
#: measured maximum at a clean window is 12,825 (glm46vflash at 16,384; the earlier
#: 8,704 run clamped 9 qwen calls and killed 3 GLM goals), hence 16,384. ActorAttack
#: is deliberately ABSENT: nothing has been measured at 4,096 (23-26 turns per goal;
#: ~1,030 of the ~1,600 GPU-hours first projected, and the run was cut) -- it stays
#: available through an explicit window and is never selected by default.
MULTITURN_CAP4096_CAP = 4096
MULTITURN_CAP4096_WINDOWS: Dict[str, int] = {"crescendo": 49152, "pair": 16384}
#: the measured maxima behind each window (tokens of prompt + reply, complete stores,
#: 2026-09-11), kept beside the windows so the next re-sizing starts from evidence
MULTITURN_CAP4096_MEASURED_MAX_CONTEXT: Dict[str, int] = {"crescendo": 41433, "pair": 12825}
MULTITURN_CAP4096_SOURCE = ("R2-D253 (a-x): the 4,096-cap "
                            "re-run of constv2-mix-full__lora_r4_ce_minus_ctl on gemma4e4b / qwen35_4b / "
                            "nemotron3nano4b / glm46vflash, cap_4096_timing.context_max of the complete stores")


def _multiturn_window_for_attacks(attacks) -> int:
    """The explicit serving window for a job that runs ``attacks`` (the attacks face of :func:`multiturn_window`) (an iterable of attack
    names, or a comma list) against a 4,096-cap target: the LARGEST of the attacks' windows,
    since one standup serves every attack of the job. REFUSES an attack with no measured
    4,096-cap window (ActorAttack) rather than falling back to the table -- pass the window
    yourself, deliberately, and record it."""
    names = [a.strip() for a in (attacks.split(",") if isinstance(attacks, str) else attacks) if a and a.strip()]
    if not names:
        raise ValueError("multiturn_window needs at least one attack name")
    missing = sorted(a for a in names if a not in MULTITURN_CAP4096_WINDOWS)
    if missing:
        raise KeyError(
            f"no measured {MULTITURN_CAP4096_CAP}-cap serving window for {missing} (known: "
            f"{sorted(MULTITURN_CAP4096_WINDOWS)}); the window rule needs a measured maximum "
            "on a COMPLETE store at this cap -- pass max_model_len explicitly and record it")
    return max(MULTITURN_CAP4096_WINDOWS[a] for a in names)


def multiturn_window(cap_or_attacks, pin: Optional[int] = None) -> int:
    """The multi-turn leg's serving window — ONE name, two faces, told apart by the argument's type
    (the 2026-09-11 reconciliation of the sdfbase-llama8b wave branch with main, which had grown the
    two independently; both call shapes are live in the tree and its tests):

    * ``multiturn_window(cap: int, pin=None)`` — the CAP face: the window DERIVED from a
      target-reply cap (the table's p99 rendered prompt + cap, on the grid), or ``pin`` when the
      run PINS a wider one (:func:`pin_window` refuses a pin below the derived minimum). The
      roster's per-family pins (``attacks.multiturn_max_model_len``) ride this face —
      ActorAttack's 49,152 x 9 included, which the attacks face has no measured row for.
    * ``multiturn_window(attacks: str | iterable)`` — the ATTACKS face (the canonical 4,096-cap
      instrument): the LARGEST of the attacks' MEASURED windows
      (:data:`MULTITURN_CAP4096_WINDOWS`); an attack with no measured window (ActorAttack) raises
      ``KeyError`` rather than falling back to the table; an empty list raises ``ValueError``.

    A ``bool`` is neither face; a ``pin`` beside an attacks argument is refused (``TypeError``)."""
    if isinstance(cap_or_attacks, bool):
        raise TypeError("multiturn_window: a bool is neither a cap nor an attack list")
    if isinstance(cap_or_attacks, int):
        return _multiturn_window_for_cap(cap_or_attacks, pin)
    if pin is not None:
        raise TypeError("multiturn_window: a pin rides the cap face only (multiturn_window(cap, pin)); the attacks "
                        "face reads the measured table")
    return _multiturn_window_for_attacks(cap_or_attacks)


# --------------------------------------------------------------------------- #
# The entries                                                                  #
# --------------------------------------------------------------------------- #

_MT_SOURCE = (f"{MEASUREMENT_DOC} section 1 — the served target's per-call timing rows "
              "(vLLM usage) of run grokmt-0902T2138, 36 shard files")
_CAP_SOURCE = (f"{MEASUREMENT_DOC} section 2 — capability corpora re-rendered with the "
               "harness's chat render, completions re-tokenized from the stored generations")
_SS_SOURCE = (f"{MEASUREMENT_DOC} section 3 — the single-shot pools (pasttense / codeattack / "
              "renellm / grok420) at the 4,096 completion cap, exact tokenizers")
_BON_SOURCE = (f"{MEASUREMENT_DOC} section 4 — BoN N=64 responses at the 512 cap, exact tokenizers")
_PETRI_SOURCE = (f"{MEASUREMENT_DOC} section 5 — every served target call in the inspect "
                 "audit logs, vLLM usage counters (target-role model events only)")
_TINYCAP_MEASURED_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the six tinycap pass-1 usage ledgers (usage_ledger.jsonl "
    "beside each configuration's cells/ on the Modal volume: every attempted "
    "/v1/chat/completions call, prompt pre-tokenized via /v1/tokenize BEFORE forwarding, "
    "rejected calls included at prompt + requested cap; 36,274 calls, 0 rejected / torn / "
    "untagged / unmeasured): the four f6cf462 tiny_ll runs — cells prompted__safety_refusal and "
    "safety_refusal__psr_kl_all x thinking off / on at launcher sha f6cf462 (2,210 / 2,210 / "
    "2,621 / 2,621 rows) — and the two 618f9dd prompted generation-suite runs — "
    "prompted__safety_refusal x thinking off / on at launcher sha 618f9dd, out roots "
    "<volume>/tinycap/gen/gemma4e4b/prompted__safety_refusal/thinking_{off,on}/618f9dd "
    "(12,930 / 13,682 rows) — all served at the certified footprint 16384 x 36 (gemma4e4b's "
    "budget). Pooled 2026-09-04 by scripts/tinycap_geometry_measure.py (v1 at 9e23be3 = the "
    "proposal Evals reviewed; the re-emit at PW11 — cap-less calls outside the clamp basis, "
    "the grader route its own key — is the commit landing these entries; outputs "
    "proposal_v2/ beside the pulled 618f9dd ledgers under the grading root)")
_TINYCAP_ANALYTIC_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — ANALYTIC, no pooled cell carries the long SIs: "
    "p99_prompt of capability:tiny from the pass-1 ledgers above + the longest SI's gemma-4 "
    "token count + the chat-template delta of a system turn, both from the offline "
    "gemma-4-E4B-it tokenizer + chat_template.jinja (the doc header's SI counts), the delta "
    "cross-checked against the per-sample prompted-minus-fitted prompt lengths of the pass-1 "
    "tiny ledgers (2,210 pairs: +92 thinking OFF = 87 safety-SI tokens + 5, +87 ON = 87 + 0, "
    "every pair); derivation script tiny_prompted_analytic.py beside proposal_v2/")
_MATH500_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the MATH-500 PROMPT BUDGET (measured 2026-09-09, not a "
    "pooled ledger): scripts/tinycap_oos_prompt_budget.py --task math500 --arm on --offline "
    "over the 500 MATH-500 problems (inspect_evals' MATH solver chain, 0-shot) for each of the "
    "four full22 models' battery configs (configs/sdfbase/math500/"
    "sdfbase_full22_<key>.math500.yaml) x {no system turn (none / object cells), the "
    "prompted__safety_refusal system turn} — 8 artifacts, 4,000 exact renders under each "
    "pinned tokenizer's thinking-ON chat render, the same counter the launcher's "
    "prompt-budget gate reads; no completion generated (the branch sizes on prompts + the cap)")

_MATH500_LONGSI_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the MATH-500 PROMPT BUDGET of the two LONG-instruction prompt "
    "classes (measured 2026-09-10, not a pooled ledger): scripts/tinycap_oos_prompt_budget.py "
    "--task math500 --arm on --offline --system-prompt-cell {prompted__si_constitution, "
    "prompted__si_irrelevant} over the 500 MATH-500 problems (inspect_evals' MATH solver chain, "
    "0-shot) for each of the four full22 models' battery configs (configs/sdfbase/math500/"
    "sdfbase_full22_<key>.math500.yaml) — 8 artifacts, 4,000 exact renders under each pinned "
    "tokenizer's thinking-ON chat render, plus the thinking-OFF render of the same 4,000 prompts "
    "with the same counter (8,000 renders pooled: the longest is an OFF render); no completion "
    "generated (the branch sizes on prompts + the cap)")

_GSM8K_RUNG_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the GSM8K PAIRED-200 PROMPT BUDGET (measured 2026-09-10, not a pooled ledger): "
    "scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm on --cap 16384 --config <the model's full22 "
    "battery config> --model-key <key> [--system-prompt-cell prompted__safety_refusal] over the 200 paired items of "
    "inspect_evals' stock gsm8k task (its 10-shot system turn + the templated user turn, rendered through the task's own "
    "solver chain; the prompted cell's SI folded into the leading system turn as the provider collapses it) for each of "
    "the four full22 models x {no system turn (the none / object cells), the prompted__safety_refusal system turn} — 8 "
    "artifacts, 1,600 exact renders under each pinned tokenizer's thinking-ON chat render, the same counter the "
    "launcher's prompt-budget gate reads; no completion generated (the branch sizes on prompts + the cap)")

_IFEVAL_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the IFEval PROMPT BUDGET (measured 2026-09-11, not a pooled ledger): "
    "configs/sdfbase/ifeval/geometry_measurement.json (schema ifeval_geometry_measurement/1; the lengths measured at "
    "tree 9046696c, the committed file re-rendered at 0e4a5c09 — the origin/main merge — after the budget table moved: "
    "its per-entry starting_seqs_by_model follows qwen35_4b's lifted 720,896 and llama8b's serving-gate 475,136, every "
    "length leaf byte-identical; re-rendered again at 5b01824e — the follow-up's origin/main merge — when the "
    "constitution prompt class joined the measurement (entries_longsi), every short-class length leaf byte-identical; "
    "re-rendered again at c758385d when qwen35_4b's row moved to its 2026-09-13 lens-stack measurement at the MATH-500 "
    "window, 741,888 = 42 x 17,664 — only the five qwen35_4b starting_seqs_by_model leaves and tree_sha changed, every "
    "length leaf byte-identical), "
    "produced by scripts/sdfbase/ifeval/measure_ifeval_geometry.py — the 541 google/IFEval "
    "items (revision "
    "966cd895) rendered through the task's own solver chain (sisteer.capeval.tiers.ifeval_messages: inspect_evals' "
    "ifeval with generate() a no-op) and tokenized under each pinned tokenizer's thinking-OFF chat render "
    "(enable_thinking false, add_generation_prompt true) for each of the four battery models x {no system turn (the "
    "none / object cells), the prompted__safety_refusal system turn folded into the leading system message as the "
    "provider collapses it} — 4,328 exact renders, the same counter the launcher's prompt-budget gate reads; no "
    "completion generated (the branch sizes on prompts + the cap)")

#: the measurement half of the three ``capability:ifeval_<cap>`` notes — one
#: prompt budget, three caps (contracts.IFEVAL_CAP_LADDER), so the facts are
#: stated once and each entry adds only its own arithmetic and S
_IFEVAL_NOTE_MEASUREMENT = (
    "MEASURED from the IFEval PROMPT BUDGET, not from a pooled ledger: the 541 google/IFEval items (revision "
    "966cd895) rendered through the task's own solver chain (inspect_evals' ifeval, generate() a no-op) and "
    "tokenized under each model's PINNED tokenizer and thinking-OFF chat render "
    "(scripts/sdfbase/ifeval/measure_ifeval_geometry.py -> configs/sdfbase/ifeval/geometry_measurement.json, the lengths "
    "at tree 9046696c, the file re-rendered at 0e4a5c09 for its starting-S block) — 4 models x 2 prompt classes (no "
    "system turn = the none / object cells; the prompted__safety_refusal "
    "system turn, folded into the leading system message) = 4,328 renders. Prompt max, bare / prompted: gemma4e4b "
    "353 / 445, qwen35_4b 375 / 468, nemotron3nano4b 375 / 463, llama8b 370 / 463; pooled p99 201, p50 119, max 468 "
    "(qwen35_4b, prompted); the safety system turn adds a CONSTANT 92 / 93 / 88 / 93 tokens (gemma / qwen / "
    "nemotron / llama) to every item. IFEval is ONE thinking-OFF arm (the user's ruling; the row task's "
    "THINKING_BY_TASK entry is off), so ONE key per cap and no _thinking twin. Sized by the MAX-PROMPT branch like "
    "gen_long_math: the driver's assert_pass_fits_window refuses a cap-bound draw whose prompt + cap does not fit "
    "the window, so the once-only clamp the p99 branches leave the prompt tail to is a REFUSAL here and the window "
    "must hold EVERY prompt at the cap. ")

#: the follow-up: the constitution-prompted cells' OWN entries. The SHORT
#: constitution SI (``prompted__si_constitution``, sisteer.data.instructions
#: si_constitution, 30,952 chars, sha256 21e06f48…) adds a CONSTANT ~5.9-6.0k
#: tokens per item, so a prompt of that class is ~14x the short classes' max
#: and cannot share their window; one entry per cap, sized by the same
#: max-prompt rule over the constitution class alone.
_IFEVAL_LONGSI_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the IFEval PROMPT BUDGET, the CONSTITUTION prompt class (measured 2026-09-12, "
    "not a pooled ledger): configs/sdfbase/ifeval/geometry_measurement.json (schema ifeval_geometry_measurement/1; "
    "the constitution class measured at tree 5b01824e — the follow-up's origin/main merge — beside the short classes "
    "of 9046696c, whose length leaves are byte-identical; the entries_longsi block), produced by "
    "scripts/sdfbase/ifeval/measure_ifeval_geometry.py — the 541 google/IFEval items (revision 966cd895) rendered "
    "through the task's own solver chain (sisteer.capeval.tiers.ifeval_messages: inspect_evals' ifeval with generate() "
    "a no-op) with the prompted__si_constitution system turn (the SHORT constitution SI, si_constitution, 30,952 chars, "
    "sha256 21e06f48…) folded into the leading system message as the provider collapses it, tokenized under each "
    "pinned tokenizer's thinking-OFF chat render (enable_thinking false, add_generation_prompt true) for each of the "
    "four battery models — 2,164 exact renders, the same counter the launcher's prompt-budget gate reads; no completion "
    "generated (the branch sizes on prompts + the cap)")
_IFEVAL_LONGSI_NOTE_MEASUREMENT = (
    "MEASURED from the IFEval PROMPT BUDGET's CONSTITUTION class, not from a pooled ledger: the 541 google/IFEval items "
    "(revision 966cd895) rendered through the task's own solver chain (inspect_evals' ifeval, generate() a no-op) with the "
    "prompted__si_constitution system turn folded into the leading system message, tokenized under each model's PINNED "
    "tokenizer and thinking-OFF chat render (scripts/sdfbase/ifeval/measure_ifeval_geometry.py -> "
    "configs/sdfbase/ifeval/geometry_measurement.json entries_longsi, at tree 5b01824e) — 4 models x 1 prompt class = "
    "2,164 renders. Prompt max, constitution: gemma4e4b 6,385, qwen35_4b 6,338, nemotron3nano4b 6,327, llama8b 6,247; "
    "pooled p99 6,132, max 6,385 (gemma4e4b); the constitution system turn adds a CONSTANT 6,032 / 5,963 / 5,952 / 5,877 "
    "tokens (gemma / qwen / nemotron / llama) to every item — ~14x the short classes' 468 max, so a constitution cell "
    "resolves THIS entry and never the short cells' (contracts.ifeval_geometry_entry by prompt class; the MATH-500 lane's "
    "gen_long_math_longsi precedent). IFEval is ONE thinking-OFF arm, so ONE key per cap and no _thinking twin. Sized by "
    "the MAX-PROMPT branch like gen_long_math: the driver's assert_pass_fits_window refuses a cap-bound draw whose prompt + "
    "cap does not fit the window, so the window must hold EVERY prompt at the cap. ")

#: The eight SERVED tinycap sub-entries (== contracts.GEOMETRY_SUBENTRY_KEYS,
#: pinned equal by test): the keys a configuration's traffic pools into, the
#: keys pass 1 serves at the certified footprint (:func:`pass1_geometry`) and
#: pass 2 at their measured windows. ``capability:grader`` is NOT one: it is
#: informational (the grader rides the target's server).
TINYCAP_KEYS: Tuple[str, ...] = (
    "capability:tiny", "capability:tiny_prompted",
    "capability:gen_long", "capability:gen_long_thinking",
    "capability:agent", "capability:agent_thinking",
    "capability:gen_long_math", "capability:gen_long_math_longsi",
)
_GSM8K_LONGSI_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the GSM8K PAIRED-200 PROMPT BUDGET under the CONSTITUTION system turn (measured "
    "2026-09-12 on the grading host, not a pooled ledger): scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm "
    "not_available --cap 8192|16384 --config configs/sdfbase/gsm8k/sdfbase_full22_llama8b.gsm8k.yaml --model-key llama8b "
    "--system-prompt-cell prompted__si_constitution --offline over the 200 paired items of inspect_evals' stock gsm8k task "
    "(its 10-shot system turn + the templated user turn, rendered through the task's own solver chain; the si_constitution "
    "SI — sha256 21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f, 5,872 Llama tokens — folded into the "
    "leading system turn as the provider collapses it) under NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79's tokenizer "
    "and no-switch render: 200 exact renders, the same counter the launcher's prompt-budget gate reads; no completion "
    "generated (the branch sizes on prompts + the cap)")
_GSM8K_LONGSI_F22_PROMPT_BUDGET_SOURCE = (
    f"{MEASUREMENT_DOC} section 6 — the GSM8K PAIRED-200 PROMPT BUDGET under the CONSTITUTION system turn on the full22 "
    "panel's three SWITCH families (math-eight-arms-extension E2, plan P3: the long-SI class is scoped by family — the Llama "
    "entries hold the Llama render, these hold the four full22 families'; measured 2026-09-13 on the grading host, not a pooled "
    "ledger): scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm on --cap 8192 --config "
    "configs/sdfbase_full22_<key>.yaml --model-key <key> --system-prompt-cell prompted__si_constitution --offline for "
    "gemma4e4b / qwen35_4b / nemotron3nano4b over the 200 paired items of inspect_evals' stock gsm8k task (its 10-shot system "
    "turn + the templated user turn, rendered through the task's own solver chain; the si_constitution SI — sha256 "
    "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f — folded into the leading system turn as the provider "
    "collapses it) under each model's PINNED tokenizer and thinking-ON render: 600 exact renders (prompt max 8,338 / 8,332 / "
    "8,288; p99 8,314 / 8,304 / 8,259), the same counter the launcher's prompt-budget gate reads; no completion generated "
    "(the branch sizes on prompts + the cap)")

#: The eight SERVED tinycap sub-entries (== contracts.GEOMETRY_SUBENTRY_KEYS,
#: pinned equal by test): the keys a configuration's traffic pools into, the
#: keys pass 1 serves at the certified footprint (:func:`pass1_geometry`) and
#: pass 2 at their measured windows. ``capability:grader`` is NOT one: it is
#: informational (the grader rides the target's server).
TINYCAP_KEYS: Tuple[str, ...] = (
    "capability:tiny", "capability:tiny_prompted",
    "capability:gen_long", "capability:gen_long_thinking",
    "capability:agent", "capability:agent_thinking",
    "capability:gen_long_math", "capability:gen_long_math_longsi",
)

_PETRI_MODELS = (
    # model, L, n served calls, p99_total, max_total, p99_prompt, clamp@L, frac_over_L, n rejected 400
    ("llama8b", 11776, 13814, 11575, 15566, 11261, 0.0142, 0.0085, 1),
    ("gemma4e4b", 15872, 18832, 15758, 20647, 15024, 0.0129, 0.0085, 113),
    ("gemma4e2b", 15616, 7703, 15443, 19896, 14871, 0.0122, 0.0086, 37),
    ("qwen35_4b", 17408, 8242, 17395, 21695, 16562, 0.0110, 0.0101, 62),
    ("qwen35_9b", 18432, 7190, 18385, 23209, 17714, 0.0104, 0.0097, 71),
)


def _entries() -> Tuple[ServedGeometry, ...]:
    petri_L = PETRI_MAX_MODEL_LEN
    out = [
        # ---------------------------------------------------------------- petri
        ServedGeometry(
            eval="petri", certified=True,
            max_model_len=petri_L,
            max_num_seqs=vs.DEFAULT_MAX_NUM_SEQS,     # == identity_seqs(16384) = 24
            basis=GeometryBasis(
                n_calls=55781, p99_total=18385, max_total=23209, p99_prompt=17714,
                requested_max_tokens=1024,
                # the 284 genuine context-length 400s over 56,065 target calls
                clamp_rate_at_L=0.0051, frac_over_L=None,
                families_measured=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_4b", "qwen35_9b"),
                measured_on=MEASURED_ON, source=_PETRI_SOURCE,
                note=("KEEPS the certified 16384 x 24 (the composer default and the smoke "
                      "gate's PETRI_* pins — never changed here). Measured per-model p99 of "
                      "prompt+completion: llama8b 11,575 / gemma4e2b 15,443 / gemma4e4b "
                      "15,758 / qwen35_4b 17,395 / qwen35_9b 18,385, max 23,209 — ABOVE the "
                      "window for every family but llama; the p99_total / p99_prompt here "
                      "are the worst family's. Every audit's target config requests "
                      "max_tokens=1024, so vLLM 400-rejects any turn whose prompt exceeds "
                      "15,360 tokens: 284 such rejections (0.5-1% of target calls: "
                      "gemma4e4b 113, qwen35_9b 71, qwen35_4b 62, gemma4e2b 37, llama8b "
                      "1), passed through to the auditor, so those audits were judged on "
                      "truncated transcripts. frac_over_L is not stated: the served calls "
                      "above 16,384 ran on the 32k resume endpoint. The tail is owned by the "
                      "runner's failure ledger + the 32k re-run proxies "
                      "(petri_runner.rerun_serving_geometry). Per-model sized windows: the "
                      "petri:<model> sub-entries (informational)."),
            )),
        # ----------------------------------------------------- multiturn_attacks
        ServedGeometry(
            eval="multiturn_attacks",
            max_model_len=8704,                       # >= p99 8,487; identity 45
            max_num_seqs=identity_seqs(8704),
            basis=GeometryBasis(
                n_calls=4598, p99_total=8487, max_total=12657, p99_prompt=7496,
                requested_max_tokens=2048, clamp_rate_at_L=0.0137, frac_over_L=0.0091,
                families_measured=("gemma4e4b",),
                measured_on=MEASURED_ON, source=_MT_SOURCE,
                note=("PAIR + Crescendo + ActorAttack pooled against the served gemma4e4b "
                      "target (cells none + si_constitution__psr_kl_all; attackers grok-4.6 "
                      "and abliterated Qwen3.6-27B) at a 2,048 completion cap. The pooled entry "
                      "is what scripts/modal/grokmt_served.py serves (all three attacks on ONE "
                      "server); 42/4,598 calls (41 Crescendo) exceed the window and are the "
                      "client clamp's / PromptOverWindow's tail. Per-attack sub-entries: "
                      "multiturn_attacks:{pair,crescendo,actorattack}. NOT the campaign's "
                      "4,096-cap multi-turn instrument (2026-09-11): sized by a p99 rule on "
                      "prompts collected under the old caps, it clamped 129 distinct max_tokens "
                      "(smallest 69) and killed 3 goals on one PAIR cell at 4,096; those jobs pass "
                      "MULTITURN_CAP4096_WINDOWS explicitly (crescendo 49,152 / pair 16,384, from "
                      "measured maxima on complete stores) and GeometryCapExceedsBasis refuses "
                      "this entry for a driver capping above 2,048."),
            )),
        ServedGeometry(
            eval="multiturn_attacks", task="pair",
            max_model_len=3456, max_num_seqs=identity_seqs(3456),      # 113
            basis=GeometryBasis(
                n_calls=1217, p99_total=1864, max_total=3229, p99_prompt=1343,
                requested_max_tokens=2048, clamp_rate_at_L=0.0082, frac_over_L=0.0,
                clamp_rate_at_total_sizing=1.0,
                families_measured=("gemma4e4b",),
                measured_on=MEASURED_ON, source=_MT_SOURCE,
                note=("Sized by the rule's second branch: at the total-p99 window (1,872) "
                      "EVERY PAIR call requests 2,048 > the room left (clamp rate 100%), so "
                      "L = next block >= p99(prompt) 1,343 + 2,048 = 3,391 -> 3,456. For a "
                      "launcher that serves PAIR alone (capctrl_serving_up --eval "
                      "multiturn_attacks:pair) at a 2,048 cap. The 4,096-cap instrument serves "
                      "PAIR at MULTITURN_CAP4096_WINDOWS['pair'] = 16,384 (measured max context "
                      "12,825 on a complete store; PAIR's prompt DOES accumulate replies -- the "
                      "filed 1.5k prompts were short because the replies were capped, R2-D253x)."),
            )),
        ServedGeometry(
            eval="multiturn_attacks", task="crescendo",
            max_model_len=9984, max_num_seqs=identity_seqs(9984),      # 39
            basis=GeometryBasis(
                n_calls=1343, p99_total=9788, max_total=12657, p99_prompt=9443,
                requested_max_tokens=2048, clamp_rate_at_L=0.0298, frac_over_L=0.0074,
                families_measured=("gemma4e4b",),
                measured_on=MEASURED_ON, source=_MT_SOURCE,
                note=("Crescendo alone at a 2,048 cap: the longest of the three (10-turn "
                      "escalation). The 4,096-cap instrument serves it at "
                      "MULTITURN_CAP4096_WINDOWS['crescendo'] = 49,152 (measured max context "
                      "41,433 on a complete store, 0 clamps on every crescendo cell)."),
            )),
        ServedGeometry(
            eval="multiturn_attacks", task="actorattack",
            max_model_len=6656, max_num_seqs=identity_seqs(6656),      # 59
            basis=GeometryBasis(
                n_calls=2038, p99_total=6579, max_total=9519, p99_prompt=5558,
                requested_max_tokens=2048, clamp_rate_at_L=0.0334, frac_over_L=0.0098,
                families_measured=("gemma4e4b",),
                measured_on=MEASURED_ON, source=_MT_SOURCE,
                note=("ActorAttack alone at a 2,048 cap. UNMEASURED at 4,096 (23-26 turns per "
                      "goal; cut from the 4,096-cap instrument by the user, 2026-09-11): a "
                      "4,096-cap ActorAttack job must pass its window explicitly."),
            )),
        # ------------------------------------------------------------ capability
        ServedGeometry(
            eval="capability",
            max_model_len=7424, max_num_seqs=identity_seqs(7424),      # 52
            basis=GeometryBasis(
                n_calls=74480, p99_total=7332, max_total=7701, p99_prompt=6946,
                requested_max_tokens=512, clamp_rate_at_L=0.015, frac_over_L=0.009,
                families_measured=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                measured_on=MEASURED_ON, source=_CAP_SOURCE,
                note=("Served GSM8K / MMLU / XSTest / OR-Bench / alpaca controls "
                      "(scripts/served_capability_run.py via scripts/modal/capctrl_served.py), "
                      "all cells pooled per family: the largest task p99 is qwen gsm8k_0shot "
                      "7,332 (its completion column is the 512 cap, an upper bound — gsm8k "
                      "stores no generations); p99_prompt 6,946 is mmlu_letter (max_tokens 1 "
                      "there, 512 elsewhere); the tail is the 6-7k constitution / irrelevant "
                      "SI prefix of the PROMPTED cells, and frac_over / clamp are alpaca's "
                      "(0.9% / 1.5%). none / fitted cells are <= 1.1k: capability:short."),
            )),
        ServedGeometry(
            eval="capability", task="short",
            max_model_len=1536, max_num_seqs=identity_seqs(1536),      # 256
            basis=GeometryBasis(
                n_calls=1500, p99_total=1076, max_total=1076, p99_prompt=575,
                requested_max_tokens=512, clamp_rate_at_L=0.0, frac_over_L=0.0,
                families_measured=("llama3b",),
                measured_on=MEASURED_ON, source=_CAP_SOURCE,
                note=("none / fitted-only capability servers (no long-SI prompted cell). "
                      "UPPER BOUND: the capability rows pool every cell per family, so the "
                      "only short-arm-dominated measurement is llama3b (cells none, prompted "
                      "safety, fitted safety psr): its alpaca MAX 1,076 is taken as the "
                      "p99 bound (the llama3b alpaca p99 is 895; p99_prompt 575 is mmlu)."),
            )),
        # ------------------------------------------------------- families_served
        ServedGeometry(
            eval="families_served",
            max_model_len=11520, max_num_seqs=identity_seqs(11520),    # 34
            basis=GeometryBasis(
                n_calls=129500, p99_total=8317, max_total=11466, p99_prompt=7352,
                requested_max_tokens=4096, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=0.20,
                families_measured=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                measured_on=MEASURED_ON, source=_SS_SOURCE,
                note=("unified_sweep_run's static + held-out single-shot families at the 4,096 "
                      "completion cap (the launcher scripts/petri_families_served_integration.py; "
                      "the BoN served leg rides the same server). Sized by the rule's second "
                      "branch: the per-family total p99 is 7,659-8,531 (pooled 8,317) but at "
                      "that window the clamp fires on 20% of all calls — 100% of the prompted "
                      "constitution / irrelevant arms (prompt p99 6,632-7,352 + 4,096) — so "
                      "L = next block >= 7,352 + 4,096 = 11,448 -> 11,520; the observed max "
                      "11,466 (qwen irrelevant) is covered; frac_over / clamp at 11,520 are 0. "
                      "n = 25,900 per family x 5. none / fitted arms: families_served:short."),
            )),
        ServedGeometry(
            eval="families_served", task="short",
            max_model_len=4864, max_num_seqs=identity_seqs(4864),      # 80
            basis=GeometryBasis(
                n_calls=90650, p99_total=2695, max_total=4792, p99_prompt=762,
                requested_max_tokens=4096, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=1.0,
                families_measured=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                measured_on=MEASURED_ON, source=_SS_SOURCE,
                note=("none / fitted arms only (no SI prefix). Second branch: the none arms "
                      "run to the 4,096 cap, so at the total-p99 window (2,704) the clamp "
                      "rate is 100%; L = next block >= p99(prompt) 762 (the worst "
                      "per-attack-family row, grok420 templates; attack-pooled 663) + 4,096 = "
                      "4,858 -> 4,864; the llama8b fitted row is 0% over."),
            )),
        # ----------------------------------------------------- families_prompted
        ServedGeometry(
            eval="families_prompted",
            max_model_len=7424, max_num_seqs=identity_seqs(7424),      # 52
            basis=GeometryBasis(
                n_calls=265216, p99_total=7284, max_total=7305, p99_prompt=6777,
                requested_max_tokens=512, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=0.021,
                families_measured=("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                measured_on=MEASURED_ON, source=_BON_SOURCE,
                note=("scripts/petri_families_prompted.py — the engine A/B on the prompted "
                      "long-SI arms at 512 completions. Basis: the BoN prompted-arm rows "
                      "(identical prompts + the same 512 cap): per-family p99 <= 7,284 "
                      "(qwen irrelevant), max 7,305, p99_prompt <= 6,777; clamp at the "
                      "total-p99 window 2.1%, at 7,424 zero."),
            )),
        # ------------------------------------------------------------------- bon
        ServedGeometry(
            eval="bon",
            max_model_len=7424, max_num_seqs=identity_seqs(7424),      # 52
            basis=GeometryBasis(
                n_calls=994560, p99_total=7261, max_total=7305, p99_prompt=6777,
                requested_max_tokens=512, clamp_rate_at_L=0.0007, frac_over_L=0.0004,
                clamp_rate_at_total_sizing=0.017,
                families_measured=("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                measured_on=MEASURED_ON, source=_BON_SOURCE,
                note=("BoN N=64 at the 512 cap, six families pooled (215,488 calls llama8b "
                      "... 66,304 llama3b): per-family p99 6,940-7,261, max 7,305, "
                      "p99_prompt <= 6,777; frac_over <= 0.04%, clamp <= 0.07% (the "
                      "per-family maxima at the measurement's candidate window). This "
                      "entry is for BoN-ONLY standups; production BoN runs on the families "
                      "server, where families_served's window governs. The client fan-out "
                      "default (bon_served.DEFAULT_MAX_CONCURRENCY) is this S."),
            )),
        ServedGeometry(
            eval="bon", task="short",
            max_model_len=768, max_num_seqs=identity_seqs(768),        # 512
            basis=GeometryBasis(
                n_calls=729344, p99_total=683, max_total=734, p99_prompt=118,
                requested_max_tokens=512, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=0.0142,
                families_measured=("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                measured_on=MEASURED_ON, source=_BON_SOURCE,
                note=("none / fitted arms only: p99 <= 683, max 734 (the llama3b rows, which "
                      "include its short-SI prompted safety cell; the attack-pooled none / "
                      "fitted arrays read p99 593, max 643, p99_prompt 100); clamp 0."),
            )),
    ]
    # ------------------------- capability:<tinycap> (MEASURED 2026-09-04, PW11)
    # The literals are the measurer's proposal_v2 output, pasted verbatim
    # (tests/test_served_geometry.PINNED carries the same numbers); the grader
    # row is informational and tiny_prompted is the one analytic provisional.
    out += [
        ServedGeometry(
            eval='capability', task='tiny',
            max_model_len=1792, max_num_seqs=identity_seqs(1792),      # 219
            basis=GeometryBasis(
                n_calls=5242, p99_total=1665, max_total=3039, p99_prompt=1664,
                requested_max_tokens=1, clamp_rate_at_L=0.0038, frac_over_L=0.0038,
                clamp_rate_at_total_sizing=None,
                families_measured=('gemma4e4b',),
                measured_on='2026-09-04', source=_TINYCAP_MEASURED_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the pass-1 usage ledgers: 5242 attempted calls, every call "
                      "capped (clamp basis = all 5242); 0 rejected with the context-length 400 at "
                      "the certified footprint — the ledger's error_class context_length, never a "
                      "raw 400 — entering the basis at prompt + requested cap; every call "
                      "pre-tokenized and counted — a pool with any unmeasured, tokenize-mismatch "
                      "or otherwise-rejected call is refused, never sized. Tasks pooled: "
                      "served_ll/tinyArc=994, served_ll/tinyHellaswag=992, served_ll/tinyMMLU=1048, "
                      "served_ll/tinyTruthfulQA=1712, served_ll/tinyWinogrande=496 (the fitted "
                      "safety_refusal__psr_kl_all cell + its none__lens0 control legs, thinking "
                      "off + on). Requested caps: [1]. Sized by the total-p99 branch: L = "
                      "round-up(1665) = 1792; the clamp rate there is 0.38% of the capped calls "
                      "(tinyMMLU's longest few-shot items, max prompt 3,038). Accepted by Evals "
                      "(E2 / the 2026-09-04 verdict); replaces the provisional entry."),
            )),
        ServedGeometry(
            eval='capability', task='tiny_prompted', provisional=True,
            max_model_len=8448, max_num_seqs=identity_seqs(8448),      # 46
            basis=GeometryBasis(
                n_calls=0, p99_total=None, max_total=None, p99_prompt=8229,
                requested_max_tokens=1, clamp_rate_at_L=None, frac_over_L=None,
                clamp_rate_at_total_sizing=None,
                families_measured=('gemma4e4b',),
                measured_on=UNMEASURED, source=_TINYCAP_ANALYTIC_SOURCE,
                n_rejected=None, n_capless=None,
                analytic=("p99_prompt(capability:tiny) 1,664 + si_irrelevant 6,560 gemma-4 tokens "
                          "(the LONGEST SI; si_constitution 6,027, safety_refusal 87) + 5 (the "
                          "chat-template delta of a system turn under thinking OFF; 0 under ON) "
                          "= 8,229 analytic p99 prompt; + max_tokens 1 = 8,230 -> block floor "
                          "8,240 -> L = round-up-256 = 8,448 (S 46 at the certified budget, 69 at "
                          "gemma4e4b's 589,824)"),
                note=("PROVISIONAL, HELD by Evals' 2026-09-04 verdict — refused by "
                      "resolve_geometry; pass 1 serves it at the certified footprint via "
                      "pass1_geometry(). The pass-1 measurement (4,420 prompted tiny calls, "
                      "p99 prompt 1,760, would-be window 1,792 x 219 — docs section 6 keeps the "
                      "row) was taken under the ~90-token safety SI ONLY, and safety is the SHORT "
                      "SI: the prompt is SI + item, so the window is sized from the longest SI "
                      "ANALYTICALLY (the `analytic` field): 1,664 + 6,560 + 5 = 8,229 -> 8,448. "
                      "Cross-checked: every one of the 2,210 per-sample prompted-minus-fitted "
                      "prompt deltas in the pass-1 tiny ledgers is exactly 87 + 5 = 92 (thinking "
                      "OFF) / 87 + 0 = 87 (ON), so the SI-token + template-delta composition is "
                      "exact on the served render; the doc header's 6,032 / 6,565 system-turn "
                      "counts are the raw 6,027 / 6,560 + that 5. Tail at 8,448 under "
                      "si_irrelevant: 20 / 5,242 tiny prompts (0.38%, tinyMMLU's longest few-shot "
                      "items, max 3,038 + 6,565 = 9,603) exceed the window and would be "
                      "400-rejected at max_tokens 1 (the prompt alone overflows); under "
                      "si_constitution 10 / 5,242 (0.19%; analytic p99 7,696). What lands it as "
                      "MEASURED: pooled prompted cells under the constitution / irrelevant SIs, "
                      "or the P17 offline render fed to the measurer as --extra-rows."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long',
            max_model_len=4608, max_num_seqs=identity_seqs(4608),      # 85
            basis=GeometryBasis(
                n_calls=5880, p99_total=2789, max_total=2945, p99_prompt=2349,
                requested_max_tokens=2048, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=0.0867,
                families_measured=('gemma4e4b',),
                measured_on='2026-09-04', source=_TINYCAP_MEASURED_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the pass-1 usage ledgers: 5880 attempted calls, every call "
                      "capped (clamp basis = all 5880); 0 rejected with the context-length 400 at "
                      "the certified footprint — the ledger's error_class context_length, never a "
                      "raw 400 — entering the basis at prompt + requested cap; every call "
                      "pre-tokenized and counted — a pool with any unmeasured, tokenize-mismatch "
                      "or otherwise-rejected call is refused, never sized. Tasks pooled: "
                      "evalplus/humaneval_plus=164, evalplus/mbpp_plus=378, inspect/gsm8k=1319, "
                      "inspect/gsm8k_paired200=200, inspect/hellaswag=100, inspect/math=2000, "
                      "inspect/mmlu_0_shot=200, inspect/mmlu_letter_valid=200, "
                      "inspect/or_bench_hard_1k=1319. Requested caps: [16, 256, 512, 768, 2048]. "
                      "Sized by the rule's second branch: at the total-p99 window (2816) the clamp "
                      "fires on 8.7% of the capped calls, so L = round-up(2349 + 2048) = 4608. "
                      "EXCLUDED from this basis (Evals' verdict 2026-09-04): the 6,348 CAP-LESS "
                      "model-grader calls on the base route (inspect/math 3,710 + "
                      "inspect/or_bench_hard_1k 2,638, no max_tokens) that the v1 proposal had "
                      "pooled here — a cap-less call cannot clamp, so in the denominator they had "
                      "read the clamp at 2,816 as 4.17% (under the 5% ceiling, total-p99 branch, "
                      "2,816 x 139) while 428 / 1,319 GSM8K calls (prompt p50 2,295 + cap 512) "
                      "would have been clamped to as little as 389 completion tokens; they are "
                      "their own key, capability:grader. Accepted by Evals (E2 / the 2026-09-04 "
                      "verdict); replaces the provisional entry."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_thinking',
            max_model_len=10752, max_num_seqs=identity_seqs(10752),      # 36
            basis=GeometryBasis(
                n_calls=6422, p99_total=7342, max_total=8715, p99_prompt=2348,
                requested_max_tokens=8192, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=0.3114,
                families_measured=('gemma4e4b',),
                measured_on='2026-09-04', source=_TINYCAP_MEASURED_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the pass-1 usage ledgers: 6422 attempted calls, every call "
                      "capped (clamp basis = all 6422); 0 rejected with the context-length 400 at "
                      "the certified footprint — the ledger's error_class context_length, never a "
                      "raw 400 — entering the basis at prompt + requested cap; every call "
                      "pre-tokenized and counted — a pool with any unmeasured, tokenize-mismatch "
                      "or otherwise-rejected call is refused, never sized. Tasks pooled: "
                      "evalplus/humaneval_plus=164, evalplus/humaneval_plus:open=164, "
                      "evalplus/mbpp_plus=378, evalplus/mbpp_plus:open=378, inspect/gsm8k=1319, "
                      "inspect/gsm8k_paired200=200, inspect/hellaswag=100, inspect/math=2000, "
                      "inspect/mmlu_0_shot=200, inspect/mmlu_letter_valid=200, "
                      "inspect/or_bench_hard_1k=1319. Requested caps: [768, 2048, 4096, 8192]. "
                      "Sized by the rule's second branch: at the total-p99 window (7424) the clamp "
                      "fires on 31.1% of the capped calls, so L = round-up(2348 + 8192) = 10752. "
                      "POOLED (Evals, recorded follow-up, not a blocker): the pool-wide p99 prompt "
                      "is GSM8K's 8-shot prompt and the 8,192 cap is MATH-500's (its own p99 "
                      "prompt 560), so per-cap-class sub-entries would size ~8,960 and buy ~20% "
                      "more S; the grader traffic of the thinking-ON containers runs with "
                      "thinking false and is in capability:grader, not here. Accepted by Evals "
                      "(E2 / the 2026-09-04 verdict); replaces the provisional entry."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_math',
            max_model_len=17664, max_num_seqs=identity_seqs(17664),      # 22
            basis=GeometryBasis(
                n_calls=4000, p99_total=None, max_total=None, p99_prompt=540, max_prompt=1041,
                requested_max_tokens=16384, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'glm46vflash', 'nemotron3nano4b'),
                measured_on='2026-09-09', source=_MATH500_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the MATH-500 PROMPT BUDGET, not from a pooled ledger: the 500 "
                      "MATH-500 problems rendered through inspect_evals' MATH solver chain (0-shot) "
                      "and tokenized under each model's PINNED tokenizer and thinking-ON chat render "
                      "(scripts/tinycap_oos_prompt_budget.py --task math500) — 4 models x 2 prompt "
                      "classes (no system turn = the none / object cells; the "
                      "prompted__safety_refusal system turn) = 4,000 renders. Prompt max, bare / "
                      "prompted: gemma4e4b 954 / 1,041, qwen35_4b 937 / 1,030, glm46vflash 856 / "
                      "946, nemotron3nano4b 935 / 1,023; pooled p99 540, max 1,041; the thinking-OFF "
                      "render differs by a constant -7..+4 tokens per item (OFF max 1,039). Requested "
                      "cap: [16384] — MATH-500's ONE pass in both arms (contracts."
                      "MATH500_MAX_TOKENS_DEFAULT; the user's ruling 2026-09-09: the standing "
                      "truncation rule's ceiling, no rung above). Sized by the MAX-PROMPT branch: the "
                      "driver's assert_pass_fits_window refuses a MATH task by name when any "
                      "cap-bound sample has prompt + cap > L, so the once-only clamp the p99 branches "
                      "leave the prompt tail to is a REFUSAL here and the window must hold EVERY "
                      "prompt at the cap — 1,041 + 16,384 = 17,425 -> block floor 17,440 -> L = "
                      "round-up-256 = 17,664 (headroom 224 <= 512; 239 tokens over the longest "
                      "prompt; the p99 branch would have refused it: 540 + 16,384 -> 16,928, 736 "
                      "below). Clamp at L: 0 of 4,000 (exact); frac over L: 0 by construction "
                      "(completion <= cap). NOT measured: completions at 16,384 (p99_total / "
                      "max_total None — the branch needs neither; gen_long_thinking's 8,192 pass "
                      "clamped 31% of its calls at total sizing, which is why the prompt+cap family "
                      "of branches exists). Both arms resolve here (class gen_long_math; ONE key, no "
                      "_thinking twin). S: 22 at the certified budget; 33 / 20 / 44 / 44 at the four "
                      "models' budgets (vs 54 / 33 / 73 / 73 at gen_long_thinking's 10,752). "
                      "Replaces MATH-500's resolution to capability:gen_long_thinking (the one-pass "
                      "8,192 default), whose basis keeps its historical inspect/math rows as a "
                      "record of that measurement."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_math_longsi',
            max_model_len=24064, max_num_seqs=identity_seqs(24064),      # 16
            basis=GeometryBasis(
                n_calls=8000, p99_total=None, max_total=None, p99_prompt=7024, max_prompt=7622,
                requested_max_tokens=16384, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'glm46vflash', 'nemotron3nano4b'),
                measured_on='2026-09-10', source=_MATH500_LONGSI_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the MATH-500 PROMPT BUDGET of the two LONG-instruction prompt "
                      "classes, not from a pooled ledger (D80: the sibling capability:gen_long_math "
                      "was sized from the bare and prompted__safety_refusal renders, max 1,041, and "
                      "the launcher assigned it to EVERY prompted MATH configuration — the "
                      "si_constitution / si_irrelevant system turns alone are ~5.9-6.7k tokens, so "
                      "their requests would have been window-clamped and the driver would have "
                      "refused the task AFTER generation): the 500 MATH-500 problems rendered through "
                      "inspect_evals' MATH solver chain (0-shot) with the cell's system turn, "
                      "tokenized under each model's PINNED tokenizer in BOTH chat renders "
                      "(scripts/tinycap_oos_prompt_budget.py --task math500 --system-prompt-cell "
                      "<cell> for ON; the OFF render beside it with the same counter) — 4 models x 2 "
                      "classes x 2 arms = 8,000 renders. Prompt max, si_constitution / si_irrelevant, "
                      "ON: gemma4e4b 6,981 / 7,514, qwen35_4b 6,900 / 7,620, glm46vflash 6,729 / "
                      "7,265, nemotron3nano4b 6,887 / 7,584; OFF: 6,979 / 7,512, 6,902 / 7,622, 6,733 "
                      "/ 7,269, 6,887 / 7,584 (the OFF render differs by a constant -4..+5 tokens per "
                      "item; the pooled max 7,622 IS an OFF render, so both arms are in the basis). "
                      "Pooled p99 7,024, max 7,622. A system turn adds a CONSTANT to every item's "
                      "render (min == max prompted-minus-bare delta over the 500 problems, every model, "
                      "both arms: gemma 6,027 / 6,560 ON), so a class max is the bare max + the SI's "
                      "tokens. Requested cap: [16384] — MATH-500's ONE pass in both arms (contracts."
                      "MATH500_MAX_TOKENS_DEFAULT). Sized by the MAX-PROMPT branch like its sibling: "
                      "the driver's assert_pass_fits_window refuses a MATH task by name when any "
                      "cap-bound sample has prompt + cap > L, so the window must hold EVERY prompt of "
                      "the class at the cap — 7,622 + 16,384 = 24,006 -> block floor 24,016 -> L = "
                      "round-up-256 = 24,064 (headroom 48 <= 512; 58 tokens over the longest prompt; "
                      "the p99 branch would have refused it: 7,024 + 16,384 -> 23,408, 656 below). "
                      "Clamp at L: 0 of 8,000 (exact); frac over L: 0 by construction (completion <= "
                      "cap). NOT measured: completions at 16,384 (p99_total / max_total None — the "
                      "branch needs neither). Selected by the configuration's ACTUAL prompt class "
                      "(contracts.MATH500_PROMPT_CLASS_OF_SI: long_si = {si_constitution, "
                      "si_irrelevant} -> here; short = no system turn or safety_refusal -> "
                      "capability:gen_long_math); a prompted SI in neither class is refused by name "
                      "before allocation. One key for both arms (no _thinking twin). S: 16 at the "
                      "certified budget; 24 / 14 / 32 / 32 at the four models' budgets (vs 33 / 20 / "
                      "44 / 44 at gen_long_math's 17,664: x(17,664 / 24,064) = 0.734)."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_rung',
            max_model_len=18944, max_num_seqs=identity_seqs(18944),      # 20
            basis=GeometryBasis(
                n_calls=1600, p99_total=None, max_total=None, p99_prompt=2400, max_prompt=2462,
                requested_max_tokens=16384, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'glm46vflash', 'nemotron3nano4b'),
                measured_on='2026-09-10', source=_GSM8K_RUNG_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the GSM8K PAIRED-200 PROMPT BUDGET, not from a pooled ledger: the 200 paired items "
                      "of inspect_evals' stock gsm8k task (10-shot — the ten TRAIN-split exemplars as the system turn + "
                      "the templated user turn) rendered through the task's own solver chain and tokenized under each "
                      "model's PINNED tokenizer and thinking-ON chat render (scripts/tinycap_oos_prompt_budget.py --task "
                      "gsm8k_paired200) — 4 models x 2 prompt classes (no system turn = the none / object cells; the "
                      "prompted__safety_refusal system turn, folded into the leading system message) = 1,600 renders. "
                      "Prompt max, bare / prompted: gemma4e4b 2,310 / 2,398, qwen35_4b 2,373 / 2,462, glm46vflash 2,128 "
                      "/ 2,216, nemotron3nano4b 2,336 / 2,424; pooled p99 2,400, p50 2,255, max 2,462; the safety system "
                      "turn adds a CONSTANT 88 (qwen 89) tokens to every item; the thinking-OFF render differs by a "
                      "constant -2..+4 tokens per item (OFF max 2,375). Requested cap: [16384] — the standing truncation "
                      "ladder's ONE rung above the experiment's 8,192 pass (gsm8k-paired200-paper-objects: a cell over "
                      "3 % at cap at 8,192 re-draws its eligible truncating items at 16,384, the ceiling; no rung above). "
                      "Sized by the MAX-PROMPT branch like gen_long_math: every redrawn item is served AT the cap and a "
                      "clamped redraw is not a 16,384 draw, so the window holds EVERY prompt at the cap — 2,462 + 16,384 "
                      "= 18,846 -> block floor 18,848 -> L = round-up-256 = 18,944 (headroom 96 <= 512; 98 tokens over "
                      "the longest prompt; the p99 branch, 2,400 + 16,384 = 18,784, lands on the same 18,944 here — the "
                      "max branch is chosen for the no-clamp rule, not because the p99 one refused). Clamp at L: 0 of "
                      "1,600 (exact); frac over L: 0 by construction. NOT measured: completions at 16,384 (p99_total / "
                      "max_total None — the branch needs neither). The experiment's 8,192 PASS needs NO new entry: 2,462 "
                      "+ 8,192 = 10,654 fits capability:gen_long_thinking (10,752) with margin 98 — the entry the "
                      "contracts' class map already resolves for a thinking-ON GSM8K control (GEOMETRY_CLASS_OF_CONTROL "
                      "gsm8k -> gen_long, + _thinking) — so the pass serves there (S 54 / 33 / 73 / 73) and only the rung "
                      "here: S 20 at the certified budget; 31 / 19 / 41 / 41 at the four models' budgets. NOT a tinycap "
                      "sub-entry (TINYCAP_KEYS unchanged: no configuration's traffic pools into it; the launcher names it "
                      "explicitly and it resolves through resolve_geometry like capability:short)."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_longsi',
            max_model_len=16384, max_num_seqs=identity_seqs(16384),      # 24
            basis=GeometryBasis(
                n_calls=200, p99_total=None, max_total=None, p99_prompt=7975, max_prompt=8007,
                requested_max_tokens=8192, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('llama8b',),
                measured_on='2026-09-12', source=_GSM8K_LONGSI_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the GSM8K PAIRED-200 PROMPT BUDGET under the CONSTITUTION system turn (llama-paper-rows, the "
                      "six-arm design of 2026-09-12: prompted__si_constitution joins prompted__safety_refusal as a served baseline). "
                      "The short constitution SI is 5,872 Llama tokens, so the prompted prompt max is 8,007 (bare 2,135, "
                      "safety-prompted 2,223; the SI adds a constant) and the 8,192 PASS no longer fits capability:gen_long_thinking "
                      "(8,007 + 8,192 = 16,199 > 10,752). Sized by the MAX-PROMPT branch like gen_long_math: 16,199 -> block floor "
                      "16,208 -> L = round-up-256 = 16,384 (headroom 176 <= 512; 185 tokens over the longest prompt) — which is "
                      "exactly the certified standup geometry (29 x 16,384 = the model's measured budget). Clamp at L: 0 of 200 "
                      "(exact); frac over L: 0 by construction. Measured on the Llama family alone: another family's constitution "
                      "cell measures its own class before it is launched (the fit sidecar refuses otherwise). NOT a tinycap "
                      "sub-entry (TINYCAP_KEYS unchanged): the launcher names it explicitly (g8k_cells.geometry_for_cell) and it "
                      "resolves through resolve_geometry like capability:gen_long_gsm8k_rung. S: 24 at the certified budget; 29 "
                      "at llama8b's."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_rung_longsi',
            max_model_len=24576, max_num_seqs=identity_seqs(24576),      # 16
            basis=GeometryBasis(
                n_calls=200, p99_total=None, max_total=None, p99_prompt=7975, max_prompt=8007,
                requested_max_tokens=16384, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('llama8b',),
                measured_on='2026-09-12', source=_GSM8K_LONGSI_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("The 16,384 RUNG of the constitution-prompted GSM8K cells (the same 200 renders as gen_long_gsm8k_longsi, "
                      "prompt max 8,007): every redrawn item is served AT the cap and a clamped redraw is not a 16,384 draw, so the "
                      "window holds EVERY prompt at the cap — 8,007 + 16,384 = 24,391 -> block floor 24,400 -> L = round-up-256 = "
                      "24,576 (headroom 176 <= 512). MATH-500's longsi entry (24,064: its class max 7,622) does NOT hold this "
                      "prompt (the 10-shot exemplars sit under the SI). Clamp at L: 0 of 200 (exact); frac over L: 0 by "
                      "construction. Measured on the Llama family alone. NOT a tinycap sub-entry; the launcher names it "
                      "explicitly for a constitution row's rung. S: 16 at the certified budget; 19 at llama8b's."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_longsi_f22',
            max_model_len=16640, max_num_seqs=identity_seqs(16640),      # 23
            basis=GeometryBasis(
                n_calls=600, p99_total=None, max_total=None, p99_prompt=8305, max_prompt=8338,
                requested_max_tokens=8192, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b'),
                measured_on='2026-09-13', source=_GSM8K_LONGSI_F22_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the GSM8K PAIRED-200 PROMPT BUDGET under the CONSTITUTION system turn on the full22 panel's "
                      "three SWITCH families (math-eight-arms-extension E2, plan P3: the eight-arm panel adds prompted__si_constitution "
                      "on gemma4e4b / qwen35_4b / nemotron3nano4b `main`). The long-SI class is SCOPED BY FAMILY: the Llama entries "
                      "(gen_long_gsm8k_longsi, 16,384) hold the Llama tokenizer's 8,007 and NOT these renders — under the four full22 "
                      "families' tokenizers and thinking-ON render the constitution-prompted prompt max is 8,338 (gemma4e4b; qwen35_4b "
                      "8,332, nemotron3nano4b 8,288; pooled p99 8,305, p50 8,222 over 600 renders; bare / safety-prompted maxima on "
                      "the same tokenizers 2,310-2,462, so the SI adds ~5,900-6,000 tokens per family — instruction length is a "
                      "property of the rendered prompt in a template, never of the text; math-eight-arms D46). 8,338 + 8,192 = 16,530 "
                      "> 16,384, so PR-A's Llama-scoped entry does not hold the pass either. Sized by the MAX-PROMPT branch like "
                      "gen_long_math: 16,530 -> block floor 16,544 -> L = round-up-256 = 16,640 (headroom 96 <= 512; 110 tokens over "
                      "the longest prompt). Clamp at L: 0 of 600 (exact); frac over L: 0 by construction. NOT a tinycap sub-entry: "
                      "the launcher names it explicitly by (prompt class, family scope) — g8k_cells.geometry_for_cell — and it "
                      "resolves through resolve_geometry like capability:gen_long_gsm8k_rung. S: 23 at the certified budget; 35 / "
                      "43 / 47 at the three models' budgets (x0.98 of the Llama-scoped entry's 36 / 44 / 48)."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_rung_longsi_f22',
            max_model_len=24832, max_num_seqs=identity_seqs(24832),      # 15
            basis=GeometryBasis(
                n_calls=600, p99_total=None, max_total=None, p99_prompt=8305, max_prompt=8338,
                requested_max_tokens=16384, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b'),
                measured_on='2026-09-13', source=_GSM8K_LONGSI_F22_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("The 16,384 RUNG of the constitution-prompted GSM8K cells on the full22 panel's three switch families (the "
                      "same 600 renders as gen_long_gsm8k_longsi_f22, prompt max 8,338): every redrawn item is served AT the cap "
                      "and a clamped redraw is not a 16,384 draw, so the window holds EVERY prompt at the cap — 8,338 + 16,384 = "
                      "24,722 -> block floor 24,736 -> L = round-up-256 = 24,832 (headroom 96 <= 512). PR-A's Llama-scoped rung "
                      "entry (24,576) does not hold it (24,722 > 24,576), nor does MATH-500's longsi (24,064). Clamp at L: 0 of "
                      "600 (exact); frac over L: 0 by construction. NOT a tinycap sub-entry; the launcher names it explicitly for "
                      "a full22 family's constitution row's rung. S: 15 at the certified budget; 23 / 29 / 31 at the three "
                      "models' budgets."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_common_f22',
            max_model_len=16640, max_num_seqs=identity_seqs(16640),      # 23
            basis=GeometryBasis(
                n_calls=600, p99_total=None, max_total=None, p99_prompt=8305, max_prompt=8338,
                requested_max_tokens=8192, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b'),
                measured_on='2026-09-13', source=_GSM8K_LONGSI_F22_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("THE COMMON GSM8K PAIRED-200 CLASS of the qwen-seedrep-native-battery wave (plan §2.2 rev 3; spec criterion 4: S x L "
                      "is an identity field shared by EVERY arm of a leg, so the constitution-prompted arm's window is the window of the "
                      "whole leg). A COPY of capability:gen_long_gsm8k_longsi_f22's MEASURED values — the same 600 renders (prompt max "
                      "8,338 / p99 8,305 under the full22 switch families' tokenizers and thinking-ON render; 8,338 + 8,192 = 16,530 -> "
                      "block floor 16,544 -> L = 16,640, headroom 96) — registered under a name every prompt class of the wave resolves "
                      "to: the long-SI constitution class it was measured on, and a fortiori every SHORT prompt (bare / object / "
                      "safety-prompted max 2,462, margin 5,986). NO new measurement. Selected by the MATRIX (views_qwen_seedrep.yaml "
                      "`geometry_class: common_f22`, carried on every row; g8k_cells.geometry_for_cell(geometry_class=)) — never by "
                      "prompt class, which stays the default for every other matrix. Kernel S: 524,287 // 16,640 = 31 under the LoRA "
                      "kernel index ceiling (D 4096); identity S at qwen35_4b's 741,888 budget: 44. NOT a tinycap sub-entry (TINYCAP_KEYS "
                      "unchanged): the launcher names it explicitly and it resolves through resolve_geometry like "
                      "capability:gen_long_gsm8k_rung. S: 23 at the certified budget."),
            )),
        ServedGeometry(
            eval='capability', task='gen_long_gsm8k_rung_common_f22',
            max_model_len=24832, max_num_seqs=identity_seqs(24832),      # 15
            basis=GeometryBasis(
                n_calls=600, p99_total=None, max_total=None, p99_prompt=8305, max_prompt=8338,
                requested_max_tokens=16384, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b'),
                measured_on='2026-09-13', source=_GSM8K_LONGSI_F22_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=("The 16,384 RUNG of the qwen-seedrep-native-battery wave's COMMON GSM8K class (plan §2.2 rev 3): a COPY of "
                      "capability:gen_long_gsm8k_rung_longsi_f22's MEASURED values (the same 600 renders, prompt max 8,338; 8,338 + "
                      "16,384 = 24,722 -> block floor 24,736 -> L = 24,832, headroom 96) under the common name every re-drawn item of "
                      "the wave is served at, short prompts a fortiori. NO new measurement. Selected by the matrix's `geometry_class`, "
                      "never by prompt class. Kernel S: 524,287 // 24,832 = 21 under the LoRA kernel index ceiling (D 4096); identity S at "
                      "qwen35_4b's budget: 29. NOT a tinycap sub-entry; the launcher names it explicitly for a common-class row's rung. "
                      "S: 15 at the certified budget."),
            )),
        ServedGeometry(
            eval='capability', task='ifeval_2048',
            max_model_len=2528, max_num_seqs=identity_seqs(2528),      # 155
            basis=GeometryBasis(
                n_calls=4328, p99_total=None, max_total=None, p99_prompt=201, max_prompt=468,
                requested_max_tokens=2048, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b', 'llama8b'),
                measured_on='2026-09-11', source=_IFEVAL_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=(_IFEVAL_NOTE_MEASUREMENT +
                      "Requested cap: [2048] — the IFEval battery's PASS-1 cap, the first rung of "
                      "contracts.IFEVAL_CAP_LADDER (2048 / 4096 / 8192: the standing truncation ladder re-draws a "
                      "cell's eligible truncating items at the next rung, capability:ifeval_4096, then the ceiling, "
                      "capability:ifeval_8192). Arithmetic: 468 + 2,048 = 2,516 -> 16-block floor 2,528 = L "
                      "(headroom 0; 12 tokens over the longest prompt at the cap; the p99 branch, 201 + 2,048 = "
                      "2,249 -> 2,256, would also have accepted 2,528 — 272 above, inside the headroom — so the max "
                      "branch is chosen for the no-clamp rule, not because the p99 one refused). Clamp at L: 0 of "
                      "4,328 (exact); frac over L: 0 by construction (completion <= cap). NOT measured: completions "
                      "at 2,048 (p99_total / max_total None — the branch needs neither). S: 155 at the certified "
                      "budget; 233 / 256 / 256 / 187 at the four models' budgets (gemma4e4b 589,824; qwen35_4b "
                      "741,888 (the 2026-09-13 lens-stack row at 17,664; its 2026-09-06 halved ceiling 360,448 lifted "
                      "2026-09-11, the swept 720,896 gave 285), whose quotient 293, and "
                      "nemotron3nano4b 786,432, whose quotient 311, are held to the hybrid "
                      "per-slot cap HYBRID_SEQS_CAP 256; llama8b 475,136, the serving gate's measured row, D103) — "
                      "the STARTING S: on Modal the standup's KV-derived shrink (kv_shrink, D110) may lower S at "
                      "readiness to what that container's KV capacity actually backs. NOT a tinycap sub-entry "
                      "(TINYCAP_KEYS unchanged): the launcher names it through "
                      "contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP[2048] and it resolves through resolve_geometry like "
                      "capability:short; pass1_geometry refuses it."),
            )),
        ServedGeometry(
            eval='capability', task='ifeval_4096',
            max_model_len=4576, max_num_seqs=identity_seqs(4576),      # 85
            basis=GeometryBasis(
                n_calls=4328, p99_total=None, max_total=None, p99_prompt=201, max_prompt=468,
                requested_max_tokens=4096, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b', 'llama8b'),
                measured_on='2026-09-11', source=_IFEVAL_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=(_IFEVAL_NOTE_MEASUREMENT +
                      "Requested cap: [4096] — the IFEval ladder's SECOND rung (contracts.IFEVAL_CAP_LADDER 2048 / "
                      "4096 / 8192): a cell over the truncation threshold at the 2,048 pass re-draws its eligible "
                      "truncating items here, and one still over it at 4,096 goes on to capability:ifeval_8192, the "
                      "ceiling. Same prompt budget as capability:ifeval_2048 (the prompts do not change with the "
                      "cap). Arithmetic: 468 + 4,096 = 4,564 -> 16-block floor 4,576 = L (headroom 0; 12 tokens "
                      "over the longest prompt at the cap; the p99 branch, 201 + 4,096 = 4,297 -> 4,304, would also "
                      "have accepted 4,576 — 272 above, inside the headroom — so the max branch is chosen for the "
                      "no-clamp rule, not because the p99 one refused). Clamp at L: 0 of 4,328 (exact); frac over "
                      "L: 0 by construction. NOT measured: completions at 4,096 (p99_total / max_total None). S: 85 "
                      "at the certified budget; 128 / 162 / 171 / 103 at the four models' budgets (gemma4e4b "
                      "589,824; qwen35_4b 741,888 (42 x 17,664, the 2026-09-13 lens-stack row at the MATH-500 window; the "
                      "halved 360,448 lifted 2026-09-11, the swept 720,896 gave 157); nemotron3nano4b 786,432, "
                      "under the hybrid per-slot "
                      "cap here; llama8b 475,136, the serving gate's measured row, D103) — the STARTING S: on Modal "
                      "the standup's KV-derived shrink (kv_shrink, D110) may lower S at readiness. NOT a tinycap "
                      "sub-entry (TINYCAP_KEYS unchanged): named through contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP[4096], "
                      "resolved through resolve_geometry; pass1_geometry refuses it."),
            )),
        ServedGeometry(
            eval='capability', task='ifeval_8192',
            max_model_len=8672, max_num_seqs=identity_seqs(8672),      # 45
            basis=GeometryBasis(
                n_calls=4328, p99_total=None, max_total=None, p99_prompt=201, max_prompt=468,
                requested_max_tokens=8192, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b', 'llama8b'),
                measured_on='2026-09-11', source=_IFEVAL_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=(_IFEVAL_NOTE_MEASUREMENT +
                      "Requested cap: [8192] — the IFEval ladder's CEILING (contracts.IFEVAL_CAP_CEILING; no rung "
                      "above): the re-draw of a cell still over the truncation threshold at 4,096. Same prompt "
                      "budget as capability:ifeval_2048 (the prompts do not change with the cap). Arithmetic: 468 + "
                      "8,192 = 8,660 -> 16-block floor 8,672 = L (headroom 0; 12 tokens over the longest prompt at "
                      "the cap; the p99 branch, 201 + 8,192 = 8,393 -> 8,400, would also have accepted 8,672 — 272 "
                      "above, inside the headroom — so the max branch is chosen for the no-clamp rule, not because "
                      "the p99 one refused). Clamp at L: 0 of 4,328 (exact); frac over L: 0 by construction. NOT "
                      "measured: completions at 8,192 (p99_total / max_total None). S: 45 at the certified budget; "
                      "68 / 85 / 90 / 54 at the four models' budgets (gemma4e4b 589,824; qwen35_4b 741,888 (the "
                      "2026-09-13 lens-stack row at 17,664; the halved 360,448 lifted 2026-09-11, the swept 720,896 gave 83); "
                      "nemotron3nano4b 786,432, under the hybrid per-slot cap here; "
                      "llama8b 475,136, the "
                      "serving gate's measured row, D103) — the STARTING S: on Modal the standup's KV-derived shrink "
                      "(kv_shrink, D110) may lower S at readiness. NOT a tinycap sub-entry (TINYCAP_KEYS unchanged): "
                      "named through contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP[8192], resolved through resolve_geometry; "
                      "pass1_geometry refuses it."),
            )),
        ServedGeometry(
            eval='capability', task='ifeval_longsi_2048',
            max_model_len=8448, max_num_seqs=identity_seqs(8448),      # 46
            basis=GeometryBasis(
                n_calls=2164, p99_total=None, max_total=None, p99_prompt=6132, max_prompt=6385,
                requested_max_tokens=2048, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b', 'llama8b'),
                measured_on='2026-09-12', source=_IFEVAL_LONGSI_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=(_IFEVAL_LONGSI_NOTE_MEASUREMENT +
                      f"Requested cap: [2048] — the constitution cells' PASS-1 window (the first rung of contracts.IFEVAL_CAP_LADDER; the ladder re-draws a cell's eligible truncating items at capability:ifeval_longsi_4096, then capability:ifeval_longsi_8192). Arithmetic: 6,385 + 2,048 = 8,433 -> 16-block ceiling 8,448 = L (headroom 15; the p99 branch, 6,132 + 2,048 = 8,180 -> 8,192, would NOT hold the longest prompt at the cap — the max branch is the rule, not a rescue; the window equals capability:tiny_prompted's provisional 8,448 by coincidence of arithmetic). Clamp at L: 0 of 2,164 (exact); frac over L: 0 by "
                      f"construction (completion <= cap). NOT measured: completions at 2,048 (p99_total / max_total None — the "
                      f"branch needs neither). S: 46 at the certified budget; 69 / 87 / 93 / 56 at the four models' budgets (gemma4e4b "
                      "589,824; qwen35_4b 741,888 (the 2026-09-13 lens-stack row at 17,664; the swept 720,896 gave 85); "
                      "nemotron3nano4b 786,432; llama8b 475,136, the serving gate's measured row, "
                      "D103) — the STARTING S, about 0.3x the short cells' at the same cap (the window is ~3.3x wider): on "
                      "Modal the standup's KV-derived shrink (kv_shrink, D110) may lower S at readiness. NOT a tinycap "
                      f"sub-entry (TINYCAP_KEYS unchanged): named through contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI[2048] "
                      "(contracts.ifeval_geometry_entry for a long_si cell), resolved through resolve_geometry; pass1_geometry "
                      "refuses it."),
            )),
        ServedGeometry(
            eval='capability', task='ifeval_longsi_4096',
            max_model_len=10496, max_num_seqs=identity_seqs(10496),      # 37
            basis=GeometryBasis(
                n_calls=2164, p99_total=None, max_total=None, p99_prompt=6132, max_prompt=6385,
                requested_max_tokens=4096, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b', 'llama8b'),
                measured_on='2026-09-12', source=_IFEVAL_LONGSI_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=(_IFEVAL_LONGSI_NOTE_MEASUREMENT +
                      f"Requested cap: [4096] — the constitution cells' SECOND rung; one still over the truncation threshold at 4,096 goes on to capability:ifeval_longsi_8192, the ceiling. Same prompt budget as capability:ifeval_longsi_2048 (the prompts do not change with the cap). Arithmetic: 6,385 + 4,096 = 10,481 -> 16-block ceiling 10,496 = L (headroom 15; the p99 branch, 6,132 + 4,096 = 10,228 -> 10,240, would not hold the longest prompt at the cap). Clamp at L: 0 of 2,164 (exact); frac over L: 0 by "
                      f"construction (completion <= cap). NOT measured: completions at 4,096 (p99_total / max_total None — the "
                      f"branch needs neither). S: 37 at the certified budget; 56 / 70 / 74 / 45 at the four models' budgets (gemma4e4b "
                      "589,824; qwen35_4b 741,888 (the 2026-09-13 lens-stack row at 17,664; the swept 720,896 gave 68); "
                      "nemotron3nano4b 786,432; llama8b 475,136, the serving gate's measured row, "
                      "D103) — the STARTING S, about 0.3x the short cells' at the same cap (the window is ~3.3x wider): on "
                      "Modal the standup's KV-derived shrink (kv_shrink, D110) may lower S at readiness. NOT a tinycap "
                      f"sub-entry (TINYCAP_KEYS unchanged): named through contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI[4096] "
                      "(contracts.ifeval_geometry_entry for a long_si cell), resolved through resolve_geometry; pass1_geometry "
                      "refuses it."),
            )),
        ServedGeometry(
            eval='capability', task='ifeval_longsi_8192',
            max_model_len=14592, max_num_seqs=identity_seqs(14592),      # 26
            basis=GeometryBasis(
                n_calls=2164, p99_total=None, max_total=None, p99_prompt=6132, max_prompt=6385,
                requested_max_tokens=8192, clamp_rate_at_L=0.0, frac_over_L=0.0,
                clamp_rate_at_total_sizing=None, sized_to_max_prompt=True,
                families_measured=('gemma4e4b', 'qwen35_4b', 'nemotron3nano4b', 'llama8b'),
                measured_on='2026-09-12', source=_IFEVAL_LONGSI_PROMPT_BUDGET_SOURCE,
                n_rejected=0, n_capless=0,
                note=(_IFEVAL_LONGSI_NOTE_MEASUREMENT +
                      f"Requested cap: [8192] — the constitution cells' CEILING (contracts.IFEVAL_CAP_CEILING; no rung above). Same prompt budget as capability:ifeval_longsi_2048. Arithmetic: 6,385 + 8,192 = 14,577 -> 16-block ceiling 14,592 = L (headroom 15; the p99 branch, 6,132 + 8,192 = 14,324 -> 14,336, would not hold the longest prompt at the cap). Clamp at L: 0 of 2,164 (exact); frac over L: 0 by "
                      f"construction (completion <= cap). NOT measured: completions at 8,192 (p99_total / max_total None — the "
                      f"branch needs neither). S: 26 at the certified budget; 40 / 50 / 53 / 32 at the four models' budgets (gemma4e4b "
                      "589,824; qwen35_4b 741,888 (the 2026-09-13 lens-stack row at 17,664; the swept 720,896 gave 49); "
                      "nemotron3nano4b 786,432; llama8b 475,136, the serving gate's measured row, "
                      "D103) — the STARTING S, about 0.3x the short cells' at the same cap (the window is ~3.3x wider): on "
                      "Modal the standup's KV-derived shrink (kv_shrink, D110) may lower S at readiness. NOT a tinycap "
                      f"sub-entry (TINYCAP_KEYS unchanged): named through contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI[8192] "
                      "(contracts.ifeval_geometry_entry for a long_si cell), resolved through resolve_geometry; pass1_geometry "
                      "refuses it."),
            )),
        ServedGeometry(
            eval='capability', task='agent',
            max_model_len=3328, max_num_seqs=identity_seqs(3328),      # 118
            basis=GeometryBasis(
                n_calls=3981, p99_total=2297, max_total=6050, p99_prompt=2173,
                requested_max_tokens=1024, clamp_rate_at_L=0.0093, frac_over_L=0.0008,
                clamp_rate_at_total_sizing=0.0907,
                families_measured=('gemma4e4b',),
                measured_on='2026-09-04', source=_TINYCAP_MEASURED_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the pass-1 usage ledgers: 3981 attempted calls, every call "
                      "capped (clamp basis = all 3981); 0 rejected with the context-length 400 at "
                      "the certified footprint — the ledger's error_class context_length, never a "
                      "raw 400 — entering the basis at prompt + requested cap; every call "
                      "pre-tokenized and counted — a pool with any unmeasured, tokenize-mismatch "
                      "or otherwise-rejected call is refused, never sized. Tasks pooled: "
                      "inspect/bfcl=3981 (V1 u V2 minus REST, tools in the prompt). Requested "
                      "caps: [1024] — the cap inspect_evals' bfcl actually requests, not the "
                      "planned 2,048 (a plan discrepancy on our side; the entry reads the ledger). "
                      "Sized by the rule's second branch: at the total-p99 window (2304) the clamp "
                      "fires on 9.1% of the capped calls, so L = round-up(2173 + 1024) = 3328; the "
                      "37 calls above it (max total 6,050) are the once-only clamp's / 400 tail "
                      "(0.93%). Accepted by Evals (E2 / the 2026-09-04 verdict); replaces the "
                      "provisional entry."),
            )),
        ServedGeometry(
            eval='capability', task='agent_thinking',
            max_model_len=6400, max_num_seqs=identity_seqs(6400),      # 61
            basis=GeometryBasis(
                n_calls=3981, p99_total=2832, max_total=6703, p99_prompt=2175,
                requested_max_tokens=4096, clamp_rate_at_L=0.0093, frac_over_L=0.0003,
                clamp_rate_at_total_sizing=1.0,
                families_measured=('gemma4e4b',),
                measured_on='2026-09-04', source=_TINYCAP_MEASURED_SOURCE,
                n_rejected=0, n_capless=0,
                note=("MEASURED from the pass-1 usage ledgers: 3981 attempted calls, every call "
                      "capped (clamp basis = all 3981); 0 rejected with the context-length 400 at "
                      "the certified footprint — the ledger's error_class context_length, never a "
                      "raw 400 — entering the basis at prompt + requested cap; every call "
                      "pre-tokenized and counted — a pool with any unmeasured, tokenize-mismatch "
                      "or otherwise-rejected call is refused, never sized. Tasks pooled: "
                      "inspect/bfcl=3981. Requested caps: [4096] — the thinking-ON floor the "
                      "provider raises BFCL to (PB5), not the planned 8,192. Sized by the rule's "
                      "second branch: at the total-p99 window (3072) the clamp fires on 100.0% of "
                      "the capped calls, so L = round-up(2175 + 4096) = 6400. Accepted by Evals "
                      "(E2 / the 2026-09-04 verdict); replaces the provisional entry."),
            )),
        ServedGeometry(
            eval='capability', task='grader', informational=True,
            max_model_len=2304, max_num_seqs=identity_seqs(2304),      # 170
            basis=GeometryBasis(
                n_calls=6348, p99_total=2150, max_total=2436, p99_prompt=1722,
                requested_max_tokens=0, clamp_rate_at_L=0.0, frac_over_L=0.0019,
                clamp_rate_at_total_sizing=None,
                families_measured=('gemma4e4b',),
                measured_on='2026-09-04', source=_TINYCAP_MEASURED_SOURCE,
                n_rejected=0, n_capless=6348,
                note=("INFORMATIONAL — the MODEL-GRADER traffic on the unsteered base route (the "
                      "routing table's 'none' entry, the store manifests' grader_virtual_model): "
                      "a different workload from the target's, measured for the record with its "
                      "own p99 and served by no launcher (the grader rides the target's server at "
                      "the target's window; resolve_geometry refuses this key). Kept OUT of "
                      "capability:gen_long, whose clamp denominator it had diluted (Evals' "
                      "verdict 2026-09-04). 6348 attempted calls, 6348 of them CAP-LESS (no "
                      "max_tokens sent: vLLM sizes the completion to the room left, so they "
                      "cannot clamp — in the prompt / total p99s, outside the clamp basis of 0 "
                      "capped calls); 0 rejected with the context-length 400 at the certified "
                      "footprint — the ledger's error_class context_length, never a raw 400 — "
                      "entering the basis at prompt + requested cap; every call pre-tokenized and "
                      "counted — a pool with any unmeasured, tokenize-mismatch or "
                      "otherwise-rejected call is refused, never sized. Tasks pooled: "
                      "inspect/math=3710 (the equivalence grader), inspect/or_bench_hard_1k=2638 "
                      "(the refusal grader); both 618f9dd containers, thinking false on the grader "
                      "route in both arms (PB4). Requested caps: none (cap-less); max completion "
                      "858. Sized by the total-p99 branch: L = round-up(2150) = 2304; the clamp "
                      "rate there is 0.00% by construction (no capped call). Recorded for Evals "
                      "(E2); a different workload, never pooled into gen_long. "
                      "STALE IN ONE RESPECT SINCE E56 (3) / PR19-C, deliberately not re-measured: "
                      "the grader is no longer cap-less. inspect_served.GRADER_MAX_TOKENS (4,096) "
                      "now rides on the grader MODEL's own GenerateConfig, so a post-cap ledger "
                      "records max_tokens_requested 4096 where this basis records 0 cap-less "
                      "calls. Nothing here is consumed: the key is INFORMATIONAL and "
                      "resolve_geometry refuses it (the grader rides the target's server at the "
                      "target's window), and the cap is BELOW this row's completion max, so the "
                      "sized window can only shrink. Re-measure the row from a post-cap wave "
                      "before ever making this key non-informational. The cap's 3-of-13,601 "
                      "truncation rate bounds EXPOSURE, not direction: a truncated grader files "
                      "the unscored sentinel and its item leaves the mean, so a capped store is a "
                      "CHANGED INSTRUMENT with potentially non-random exclusions (E58 note (b))."),
            )),
    ]
    # ------------------------------------------ petri:<model> (informational)
    for model, L, n, p99, mx, p99p, clamp, over, rej in _PETRI_MODELS:
        out.append(ServedGeometry(
            eval="petri", task=model, requires_recertification=True,
            max_model_len=L, max_num_seqs=identity_seqs(L),
            basis=GeometryBasis(
                n_calls=n, p99_total=p99, max_total=mx, p99_prompt=p99p,
                requested_max_tokens=1024, clamp_rate_at_L=clamp, frac_over_L=over,
                families_measured=(model,), measured_on=MEASURED_ON, source=_PETRI_SOURCE,
                note=(f"INFORMATIONAL — requires re-certification, consumed by no launcher: the "
                      f"p99-sized Petri window for {model} ({rej} context-length 400s at the "
                      "canonical 16384 x 24 with max_tokens 1024). Adopting it means moving "
                      "the Petri fidelity smoke (petri-gate16k, PETRI_* pins) and "
                      "test_petri_geometry_cert together with the composer default; for the "
                      "qwen / long-SI cells the window goes UP, not down."),
            )))
    return tuple(out)


#: The table: key (``eval`` or ``eval:task``) -> geometry. Built once; frozen.
TABLE: Dict[str, ServedGeometry] = {g.key: g for g in _entries()}

#: The top-level evals a launcher may name, in table order.
KNOWN_EVALS: Tuple[str, ...] = tuple(dict.fromkeys(g.eval for g in TABLE.values()))
#: Every key, including sub-entries (``--eval`` accepts any of these).
KNOWN_KEYS: Tuple[str, ...] = tuple(TABLE)


def parse_key(eval_name: str, task: Optional[str] = None) -> str:
    """``("capability", "short")`` and ``"capability:short"`` name the same
    entry; a task given both ways must agree."""
    if not isinstance(eval_name, str) or not eval_name:
        raise UnknownEvalGeometry(f"eval name must be a non-empty string, got {eval_name!r}")
    base, sep, inline = eval_name.partition(":")
    if sep and task is not None and inline != task:
        raise UnknownEvalGeometry(
            f"eval {eval_name!r} names task {inline!r} but task={task!r} was also given")
    t = inline if sep else task
    return base if not t else f"{base}:{t}"


def geometry_for(eval_name: str, task: Optional[str] = None) -> ServedGeometry:
    """The table entry for ``eval_name`` (``"eval"`` or ``"eval:task"``, or
    ``task=`` separately); :class:`UnknownEvalGeometry` (a ``KeyError``)
    listing the known keys otherwise. Informational entries ARE returned here
    (inspection); :func:`resolve_geometry` is what refuses to launch them."""
    key = parse_key(eval_name, task)
    try:
        return TABLE[key]
    except KeyError:
        raise UnknownEvalGeometry(
            f"no served geometry for {key!r}; known evals: {list(KNOWN_EVALS)}; "
            f"sub-entries: {[k for k in KNOWN_KEYS if ':' in k]}") from None


# --------------------------------------------------------------------------- #
# Resolution: table default < explicit override, with the budget guard.       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolvedGeometry:
    """What a launcher ends up serving, with WHERE each factor came from
    (``"table"`` / ``"override"`` / ``"fallback"``) so the startup log and the
    run's provenance say whether the identity geometry is in effect."""

    eval: Optional[str]
    max_model_len: int
    max_num_seqs: int
    window_source: str
    seqs_source: str
    entry: Optional[ServedGeometry]
    #: the model the launch is sized for and the budget that sizing used
    #: (``None`` = no model named: the Llama-certified default, as before)
    model_key: Optional[str] = None
    budget: Optional[ResolvedBudget] = None

    @property
    def batched_tokens(self) -> int:
        return self.max_num_seqs * self.max_model_len

    @property
    def budget_tokens(self) -> int:
        """The batched-token budget this resolution is held to: the model's
        (measured or default) when a model was named, else the Llama-certified
        :data:`BATCHED_TOKEN_BUDGET`."""
        return (self.budget.batched_token_budget if self.budget is not None
                else BATCHED_TOKEN_BUDGET)

    @property
    def budget_source(self) -> str:
        return self.budget.source if self.budget is not None else DEFAULT_BUDGET_SOURCE

    @property
    def over_budget(self) -> bool:
        return self.batched_tokens > self.budget_tokens

    @property
    def identity_S(self) -> int:
        """The S the budget buys at this window: the model entry's ``seqs_at``
        (the hybrid per-slot cap applied) when a MEASURED model budget was
        resolved, else the raw ``identity_seqs`` at the effective budget."""
        if self.budget is not None and self.budget.per_model:
            return self.budget.entry.seqs_at(self.max_model_len)
        return identity_seqs(self.max_model_len, budget=self.budget_tokens)

    @property
    def is_identity(self) -> bool:
        return self.max_num_seqs == self.identity_S

    @property
    def overridden(self) -> bool:
        return "override" in (self.window_source, self.seqs_source)

    @property
    def provisional(self) -> bool:
        """Resolved by :func:`pass1_geometry` for a PROVISIONAL entry: the
        certified footprint serving it for its measurement pass."""
        return PROVISIONAL_SOURCE in (self.window_source, self.seqs_source)

    @property
    def measurement_pass(self) -> bool:
        """Resolved by :func:`pass1_geometry` — the certified footprint served
        for a MEASUREMENT pass, whether the entry is still provisional
        (:data:`PROVISIONAL_SOURCE`) or already measured and being extended
        (:data:`MEASUREMENT_SOURCE`)."""
        return (self.provisional
                or MEASUREMENT_SOURCE in (self.window_source, self.seqs_source))

    def as_dict(self) -> dict:
        return {
            "eval": self.eval,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "batched_tokens": self.batched_tokens,
            "budget": self.budget_tokens,
            "budget_source": self.budget_source,
            "model_key": self.model_key,
            "over_budget": self.over_budget,
            "identity": self.is_identity,
            "window_source": self.window_source,
            "seqs_source": self.seqs_source,
            "provisional": self.provisional,
            "measurement_pass": self.measurement_pass,
            "basis": (asdict(self.entry.basis) if self.entry is not None else None),
            "sized_by": (self.entry.sized_by if self.entry is not None else None),
        }

    @property
    def legacy(self) -> bool:
        """Resolved from a launcher's pre-table fallback (no eval, no table)."""
        return "fallback" in (self.window_source, self.seqs_source)

    def describe(self) -> str:
        """The startup log lines: geometry, budget position, and the basis."""
        ident = ("identity" if self.is_identity
                 else f"NOT identity (identity_seqs({self.max_model_len}) = "
                      f"{self.identity_S})")
        over = ""
        if self.over_budget:
            over = (" — LEGACY OVER BUDGET (no --eval / no table entry: the pre-table "
                    "default, accepted as-is)" if self.legacy
                    else " — OVER BUDGET: bigger profiling forward + KV reservation "
                         "than the certified footprint")
        cap = ""
        if (self.budget is not None and self.budget.per_model
                and self.budget.entry.hybrid
                and identity_seqs(self.max_model_len, budget=self.budget_tokens)
                > HYBRID_SEQS_CAP):
            cap = (f"; hybrid per-slot cap: S {HYBRID_SEQS_CAP} not "
                   f"{identity_seqs(self.max_model_len, budget=self.budget_tokens)}")
        ceiling = (self.budget.entry.window_ceiling(self.max_model_len)
                   if self.budget is not None and self.budget.per_model else None)
        if ceiling is not None:
            quotient = min(identity_seqs(self.max_model_len, budget=self.budget_tokens), HYBRID_SEQS_CAP)
            cap += (f"; hybrid window ceiling: S {ceiling.max_num_seqs} not {quotient} at L={self.max_model_len} "
                    f"(the allocator's {ceiling.kv_max_concurrency:.2f}x = {ceiling.kv_tokens_reported:,} KV tokens, "
                    f"job {ceiling.probe_job_id}, {ceiling.measured_on})")
        if self.budget is None or self.budget.measured:
            why = ""
        elif self.budget.provisional:
            why = (" — PROVISIONAL, a conservative ceiling chosen from an observed "
                   "engine-init failure and awaiting a measurement")
        else:
            why = " — UNMEASURED, the Llama-certified default undersells smaller-KV models"
        whose = ("" if self.budget is None else
                 f" [model {self.model_key}: {self.budget_source}" + why + cap + "]")
        budget = (f"{self.batched_tokens:,} of the {self.budget_tokens:,} "
                  f"batched-token budget{whose}" + over)
        head = (f"[geometry] eval={self.eval or '-'} max_model_len="
                f"{self.max_model_len} ({self.window_source}) max_num_seqs="
                f"{self.max_num_seqs} ({self.seqs_source}); {ident}; {budget}")
        if self.entry is None:
            return head
        b = self.entry.basis
        basis = (f"[geometry] basis ({self.entry.sized_by}): n_calls={b.n_calls} "
                 f"p99_total={b.p99_total} max_total={b.max_total} p99_prompt={b.p99_prompt} "
                 f"requested_max_tokens={b.requested_max_tokens} clamp_rate_at_L="
                 f"{b.clamp_rate_at_L} frac_over_L={b.frac_over_L} families="
                 f"{','.join(b.families_measured)} measured_on={b.measured_on} "
                 f"source={b.source}")
        if b.n_rejected is not None:
            basis += f" n_rejected={b.n_rejected}"
        lines = [head, basis]
        if self.entry.certified:
            lines.append("[geometry] certified window (Petri): below its p99 by design; the "
                         "failure ledger + 32k re-run proxies own the tail")
        if self.provisional:
            lines.append(f"[geometry] PROVISIONAL entry {self.entry.key}: served at the certified "
                         "Petri footprint for its MEASUREMENT pass (pass 1); the usage ledger "
                         "is the basis of the entry that replaces it"
                         + (f" (its own window {self.entry.max_model_len} is an ANALYTIC "
                            "derivation, not served)" if self.entry.basis.analytic else ""))
        elif self.measurement_pass:
            lines.append(f"[geometry] MEASUREMENT PASS for measured entry {self.entry.key}: served "
                         "at the certified Petri footprint (pass 1) to extend its ledger, not at "
                         f"its measured window {self.entry.max_model_len}; pass 2 serves the "
                         "measured window through resolve_geometry")
        if self.entry.seqs_override:
            lines.append(f"[geometry] seqs_override: {self.entry.seqs_override}")
        return "\n".join(lines)


def resolve_geometry(eval_name: Optional[str] = None, *,
                     max_model_len: Optional[int] = None,
                     max_num_seqs: Optional[int] = None,
                     fallback: Optional[Tuple[int, int]] = None,
                     allow_over_budget: bool = False,
                     strict_identity: bool = False,
                     model_key: Optional[str] = None,
                     requested_max_tokens: Optional[int] = None,
                     served_lora: Optional[Tuple[int, int]] = None) -> ResolvedGeometry:
    """Resolve a launch geometry: an explicit factor wins; an unset factor
    comes from the eval's table entry (``"eval"`` or ``"eval:task"``); with no
    eval, from ``fallback`` (a launcher's pre-table legacy default, so its
    behaviour is unchanged when nobody names an eval).

    THE PER-MODEL BUDGET: with ``model_key``, an unset ``max_num_seqs`` is
    DERIVED from that model's budget at the resolved window —
    ``identity_seqs(L, budget=budget_for(model_key))``, ``seqs_source =
    "model_budget"`` — instead of the entry's Llama-sized S, and the budget
    guard below is held to the model's budget (measured, or the Llama-certified
    default with a warning when the model is unmeasured). Without a model
    every entry's S is the Llama default, exactly as before.

    THE BUDGET GUARD: when an override pushes ``S x L`` above
    :data:`BATCHED_TOKEN_BUDGET`, refuse (:class:`GeometryOverBudget`) unless
    ``allow_over_budget`` — an operator who wants the bigger footprint says
    so on the command line. With ``strict_identity`` (the standup and the
    families wave when resolving from the table) an override that leaves the
    pair NON-identity is refused as well (:class:`GeometryIdentityBroken`),
    unless ``allow_over_budget`` — the second adversarial review found
    ``--eval capability --max-model-len 16384`` accepted as a silent
    851,968-token launch. An informational entry is refused outright
    (``requires_recertification``: :class:`GeometryRequiresRecertification`;
    ``informational``: :class:`GeometryInformational`), and so is a
    provisional one (:class:`GeometryProvisional`)."""
    entry = geometry_for(eval_name) if eval_name is not None else None
    if entry is not None and entry.requires_recertification:
        raise GeometryRequiresRecertification(
            f"{entry.key} is informational: Petri's window is certified by the fidelity "
            "smoke (petri-gate16k) and pinned by test_petri_geometry_cert; adopt a "
            "per-model window by moving those together with the composer default, not "
            "by launching this entry")
    if entry is not None and entry.informational:
        raise GeometryInformational(
            f"{entry.key} is informational: measured for the record (the model-grader "
            "traffic's own p99), consumed by no launcher — the grader rides the target's "
            "server at the target's window; there is nothing to stand up here")
    if entry is not None and entry.provisional:
        raise GeometryProvisional(
            f"{entry.key} is provisional: "
            + ("its window is an ANALYTIC derivation, not a measurement — no pooled cell "
               "carries the long SIs yet" if entry.basis.analytic is not None else
               "no pass-1 ledger has been pooled, so there is no measured window")
            + ". Run the measurement pass at the certified footprint with pass1_geometry(), "
              "pool its usage ledgers with scripts/tinycap_geometry_measure.py, and land the "
              "measured entry once Evals accepts the proposal")
    if entry is None and fallback is None and (max_model_len is None or max_num_seqs is None):
        raise ValueError(
            "resolve_geometry needs an eval name, or a fallback, or BOTH "
            "max_model_len and max_num_seqs — nothing is guessed")

    rb = budget_for(model_key, served_lora=served_lora) if model_key is not None else None   # the served class sizes an unsteered S (2026-10-02)

    def pick(explicit, attr, idx):
        if explicit is not None:
            if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
                raise ValueError(f"{attr} override must be a positive int, got {explicit!r}")
            return int(explicit), "override"
        if entry is not None:
            return getattr(entry, attr), "table"
        return int(fallback[idx]), "fallback"

    L, ws = pick(max_model_len, "max_model_len", 0)
    if (requested_max_tokens is not None and ws == "table" and entry is not None
            and entry.basis.requested_max_tokens is not None
            and int(requested_max_tokens) > int(entry.basis.requested_max_tokens)):
        # THE CAP GUARD (2026-09-11): a table window is a measurement under ONE completion cap;
        # serving a driver that caps higher from it re-creates the clamping the 4,096-cap
        # multi-turn re-run hit at the table's 8,704. Pass the window explicitly
        # (MULTITURN_CAP4096_WINDOWS for the multi-turn attacks) or re-measure the entry.
        raise GeometryCapExceedsBasis(
            f"{entry.key}'s table window {L} was measured under a {entry.basis.requested_max_tokens}-token "
            f"completion cap; this driver caps replies at {int(requested_max_tokens)}. A window is sized "
            "from a MEASURED MAXIMUM on a COMPLETE store at the cap being served, never a percentile "
            "under a smaller cap -- pass max_model_len explicitly (the multi-turn attacks: "
            f"{MULTITURN_CAP4096_WINDOWS}) or re-measure the entry at this cap")
    if max_num_seqs is None and rb is not None and (entry is not None or fallback is not None):
        # the per-model budget sizes S at the RESOLVED window — through the
        # entry's seqs_at, so a hybrid family's per-slot cap applies
        S = (rb.entry.seqs_at(L) if rb.per_model
             else identity_seqs(L, budget=rb.batched_token_budget))
        ss = "model_budget"
    else:
        S, ss = pick(max_num_seqs, "max_num_seqs", 1)
    res = ResolvedGeometry(eval=entry.key if entry is not None else None,
                           max_model_len=L, max_num_seqs=S,
                           window_source=ws, seqs_source=ss, entry=entry,
                           model_key=model_key, budget=rb)
    budget_tokens = res.budget_tokens
    whose = ("certified" if rb is None else
             f"{rb.source if rb.per_model else 'default (Llama-certified)'} "
             f"{model_key}")
    if res.over_budget and res.overridden and not allow_over_budget:
        raise GeometryOverBudget(
            f"{S} x {L} = {res.batched_tokens:,} exceeds the {whose} "
            f"batched-token budget {budget_tokens:,} (identity_seqs({L}) "
            f"= {res.identity_S}); the no-chunking identity is "
            "broken and the launch would profile + reserve KV over a bigger footprint "
            "than the certified one — pass allow_over_budget (the launcher's "
            "--allow-over-budget) to do it on purpose")
    if strict_identity and res.overridden and not res.is_identity and not allow_over_budget:
        raise GeometryIdentityBroken(
            f"{S} x {L} is not the identity geometry (identity_seqs({L}) = "
            f"{res.identity_S}): the override breaks the "
            "no-chunking identity the table sizes to — pass both factors as an identity "
            "pair, or allow_over_budget (the launcher's --allow-over-budget) to launch it "
            "on purpose")
    return res


def pass1_geometry(eval_name: str, task: Optional[str] = None, *,
                   model_key: Optional[str] = None, window: Optional[int] = None) -> ResolvedGeometry:
    """The MEASUREMENT-pass geometry of a PROVISIONAL entry: the certified
    Petri WINDOW — ``PETRI_MAX_MODEL_LEN``, read from the composer's own
    constant — labelled with the entry it measures (``eval`` = the entry's
    key, ``window_source`` :data:`PROVISIONAL_SOURCE`), so the standup argv,
    the run's manifest and the ledger's basis all name the sub-entry being
    sized while serving the one window that is certified.

    THE PER-MODEL BUDGET (E7 / the per-model budgets, 2026-09-04): with
    ``model_key`` the S is sized from THAT model's budget at the certified
    window — :func:`budget_for`, through the entry's ``seqs_at`` when measured
    (the hybrid per-slot cap applied), the raw quotient of the Llama-certified
    default otherwise (with ``budget_for``'s warning) — ``seqs_source =
    "model_budget"``, the same derivation :func:`resolve_geometry` performs
    for a measured entry; the geometry still reads as provisional (its
    window source). Without a key: the certified footprint's own S
    (``DEFAULT_MAX_NUM_SEQS``, ``seqs_source`` :data:`PROVISIONAL_SOURCE`),
    exactly as before the budgets.

    Serves a PROVISIONAL entry (``window_source`` :data:`PROVISIONAL_SOURCE`)
    or a MEASURED tinycap entry (:data:`TINYCAP_KEYS`; ``window_source``
    :data:`MEASUREMENT_SOURCE` — a pass 1 re-run extends the ledger the
    entry was measured from, e.g. the fitted cells' generation tasks the
    accepted pools did not contain; pass 2 serves the measured window
    through :func:`resolve_geometry`). Refuses every other entry — a
    pre-tinycap eval has no ledger-based measurement pass, and an
    informational one is served by no launcher."""
    entry = geometry_for(eval_name, task)
    if entry.informational:
        raise ValueError(
            f"{entry.key} is informational — measured for the record, served by no launcher "
            "(the grader rides the target's server); pass1_geometry serves the tinycap "
            "sub-entries only")
    if not entry.provisional and entry.key not in TINYCAP_KEYS:
        raise ValueError(
            f"{entry.key} is not provisional and not a tinycap sub-entry — resolve it with "
            "resolve_geometry(); pass1_geometry serves only entries measured (or awaiting "
            "measurement) from the usage ledger at the certified footprint")
    source = PROVISIONAL_SOURCE if entry.provisional else MEASUREMENT_SOURCE
    # a caller may name a pass-1 window of its own (`window`; the Modal
    # launcher's per-key window table that once passed it left with the
    # onboarded-row machinery, 2026-10-08) — an UNPROBED window of its site overlay, where no record-wide
    # floor binds a no-set reader and S is the plain quotient; the certified
    # footprint for every other caller, byte for byte. A window without a model
    # key is refused: the footprint's own S means nothing at another window.
    if window is not None:
        if model_key is None:
            raise ValueError("pass1_geometry: a window is sized from a MODEL's budget — pass model_key with it")
        if int(window) <= 0:
            raise ValueError(f"pass1_geometry: window must be a positive token count, got {window!r}")
    L = int(window) if window is not None else PETRI_MAX_MODEL_LEN
    if model_key is None:
        return ResolvedGeometry(eval=entry.key, max_model_len=L,
                                max_num_seqs=vs.DEFAULT_MAX_NUM_SEQS,
                                window_source=source, seqs_source=source,
                                entry=entry)
    rb = budget_for(model_key)
    # the model's budget at the certified window — resolve_geometry's own
    # derivation for an unset S (seqs_at: the hybrid per-slot cap applies)
    S = (rb.entry.seqs_at(L) if rb.per_model
         else identity_seqs(L, budget=rb.batched_token_budget))
    return ResolvedGeometry(eval=entry.key, max_model_len=L, max_num_seqs=S,
                            window_source=source, seqs_source="model_budget",
                            entry=entry, model_key=model_key, budget=rb)


def provisional_keys() -> Tuple[str, ...]:
    """Every provisional key in table order (the entries still awaiting a
    measured window — served for measurement only)."""
    return tuple(k for k, g in TABLE.items() if g.provisional)


def informational_keys() -> Tuple[str, ...]:
    """Every key no launcher may serve (``requires_recertification`` or
    ``informational``), in table order."""
    return tuple(k for k, g in TABLE.items() if g.requires_recertification or g.informational)


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# THE KV-DERIVED S. The per-model budgets above were MEASURED on the          #
# cluster's H200; a Modal H200 container reported 782,688 KV tokens at        #
# L=16,384 for glm46vflash against the 786,432-token budget (48 x 16,384) —   #
# 47.77-way, so the configured S=48 was not KV-backed there and readiness     #
# refused (correctly: a width in argv but not in memory). The window L stays  #
# (the per-eval p99 rule); S is derived from the KV capacity the container    #
# ACTUALLY REPORTS — S = floor(kv_backed_concurrency) — by the standup        #
# (capctrl_serving_up --kv-derived-seqs on) on the ONE refusal that names     #
# nothing but the shortfall (vllm_server.KvCoverageShortfall: every pin       #
# held), re-launched once, and recorded as ONE record the reader (the         #
# launcher, the store-trust check) accepts only through kv_shrink_problem.    #
# Never a hand-typed Modal table.                                             #
# --------------------------------------------------------------------------- #

#: ``seqs_source`` of a geometry whose S the STANDUP derived from the KV capacity
#: the serving container reported at readiness (the configured S — a budget
#: measured on another machine — was not KV-backed there).
KV_DERIVED_SOURCE = "kv_derived"
#: the record's key in the standup's ``endpoint.json`` geometry block, the driver's
#: manifest ``geometry`` block (copied verbatim) and the launcher's REC / served_geometry.json
KV_SHRINK_KEY = "kv_shrink"
KV_SHRINK_RULE = "S = floor(kv_backed_concurrency); kv_backed_concurrency = KV capacity at L / L"
KV_SHRINK_FIELDS: Tuple[str, ...] = ("configured_max_num_seqs", "served_max_num_seqs", "max_model_len",
                                     "kv_backed_concurrency", "kv_capacity_tokens", "basis", "rule")


def kv_derived_seqs(kv_backed_concurrency: float) -> int:
    """The S the container's KV cache backs at full length: the FLOOR of the
    concurrency the resolved scheduler report states (uniform cache:
    ``num_gpu_blocks x block_size / L``; hybrid: the allocator's own
    ``kv_max_concurrency``). Refuses a non-finite or non-positive number."""
    c = float(kv_backed_concurrency)
    if not math.isfinite(c) or c <= 0:
        raise ValueError(f"kv_backed_concurrency {kv_backed_concurrency!r} is not a positive finite number")
    return int(math.floor(c))


def kv_shrink_record(*, configured_max_num_seqs: int, max_model_len: int, kv_backed_concurrency: float,
                     kv_capacity_tokens: int, basis: str, first_launch_log: Optional[str] = None) -> dict:
    """The KV-derived shrink as ONE record: what was configured, what the container
    reported (the concurrency, the capacity in tokens, which formula — the
    basis), what is served and by which rule; validated by
    :func:`kv_shrink_problem` before it is returned (``ValueError`` when the
    numbers do not shrink: a concurrency at or above the configured S, or one
    that backs no full-length sequence)."""
    served = kv_derived_seqs(kv_backed_concurrency)
    rec = {"configured_max_num_seqs": int(configured_max_num_seqs), "served_max_num_seqs": served,
           "max_model_len": int(max_model_len), "kv_backed_concurrency": float(kv_backed_concurrency),
           "kv_capacity_tokens": int(kv_capacity_tokens), "basis": str(basis), "rule": KV_SHRINK_RULE,
           "reported_by": ("the serving container's resolved scheduler report (vllm_server readiness "
                           "half 3) at the configured S — never a table"),
           "first_launch_log": first_launch_log}
    problem = kv_shrink_problem(rec, configured_max_num_seqs=configured_max_num_seqs,
                                max_model_len=max_model_len, served_max_num_seqs=served)
    if problem:
        raise ValueError(problem)
    return rec


def kv_shrink_problem(shrink: object, *, configured_max_num_seqs: int, max_model_len: int,
                      served_max_num_seqs: object) -> Optional[str]:
    """``None`` when ``shrink`` is a well-formed record of THIS rule for these
    numbers — the S it names IS ``floor`` of the concurrency it reports, the
    capacity it reports divides to that S at this window, it is strictly below
    the configured S it names, that configured S and window are THIS run's, and
    it equals the S the reader sees served (``served_max_num_seqs``); else why
    not. Fail-closed: a reader (the launcher after the standup, the store-trust
    check before it) accepts a served S below the configured one ONLY through
    this function."""
    if not isinstance(shrink, Mapping):
        return f"{KV_SHRINK_KEY} is not a mapping ({type(shrink).__name__})"
    missing = [k for k in KV_SHRINK_FIELDS if k not in shrink]
    if missing:
        return f"{KV_SHRINK_KEY} lacks {missing}"

    def _pos_int(k):
        v = shrink[k]
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None
    conf, served, L, cap = (_pos_int(k) for k in ("configured_max_num_seqs", "served_max_num_seqs",
                                                   "max_model_len", "kv_capacity_tokens"))
    if None in (conf, served, L, cap):
        return (f"{KV_SHRINK_KEY}: configured_max_num_seqs / served_max_num_seqs / max_model_len / "
                "kv_capacity_tokens must be positive integers")
    conc = shrink["kv_backed_concurrency"]
    if isinstance(conc, bool) or not isinstance(conc, (int, float)) or not math.isfinite(conc) or conc <= 0:
        return f"{KV_SHRINK_KEY}: kv_backed_concurrency {conc!r} is not a positive finite number"
    if shrink["rule"] != KV_SHRINK_RULE:
        return f"{KV_SHRINK_KEY}: rule {shrink['rule']!r} is not {KV_SHRINK_RULE!r}"
    if conf != int(configured_max_num_seqs):
        return f"{KV_SHRINK_KEY}: configured S {conf} is not this run's configured {int(configured_max_num_seqs)}"
    if L != int(max_model_len):
        return f"{KV_SHRINK_KEY}: max_model_len {L} is not this run's {int(max_model_len)}"
    if served != kv_derived_seqs(conc):
        return f"{KV_SHRINK_KEY}: served S {served} != floor({conc}) = {kv_derived_seqs(conc)}"
    if cap // L != served:
        return f"{KV_SHRINK_KEY}: kv_capacity_tokens {cap} // {L} = {cap // L} != served S {served}"
    if served >= conf:
        return f"{KV_SHRINK_KEY}: served S {served} is not below the configured {conf} — not a shrink"
    if served_max_num_seqs is None or isinstance(served_max_num_seqs, bool) or int(served_max_num_seqs) != served:
        return f"{KV_SHRINK_KEY}: the record's served S {served} != the S actually served {served_max_num_seqs!r}"
    return None


# --------------------------------------------------------------------------- #
# ONE geometry-adjustment contract                                            #
# --------------------------------------------------------------------------- #
#
# A native multi-LoRA standup lowers S under the LoRA kernel's int32 index
# ceiling (``lora_serving_up.lora_kernel_ceiling_geometry``: gemma 121 -> 107 and
# nemotron 161 -> 86 at L 4864) and a standup under ``--kv-derived-seqs`` lowers
# it once more to the S its KV cache backs. Both are the same thing to a reader:
# a served S below the budget's, lowered for a NAMED, RE-DERIVABLE reason. Before
# this contract the two were read by two rules — a ``kv_shrink`` record held to
# ``kv_shrink_problem`` (accepted) and everything else refused as
# ``SERVED_GEOMETRY_MISMATCH`` — so a gemma / nemotron native row was
# un-servable to every tinycap / math reader, and admitting "any smaller S" was
# the wrong fix. ``geometry.adjustments`` is the ONE record, ORDERED AS APPLIED:
# ``[]`` (identity), ``[kernel]``, ``[kv]`` or ``[kernel, kv]`` (the KV shrink is
# computed from the kernel-lowered S); :func:`geometry_adjustment_problem`
# re-derives every step and requires the served S to be the last step's — a
# step out of order, an unknown kind, a kernel step whose numbers are not the
# formula's, or a bare smaller S with no record is a NAMED problem.
# ``identity`` stays True iff ``adjustments == []``; the existing
# ``geometry_lowered`` / ``lora_kernel_index_ceiling`` / ``kv_shrink`` keys stay
# beside it (today's pins hold); ``geometry_override_reason`` keeps the FIRST
# adjustment's kind.

ADJUSTMENT_KERNEL = _sa.LORA_KERNEL_CEILING_REASON      # "lora_kernel_index_ceiling"
ADJUSTMENT_KV = KV_SHRINK_KEY                             # "kv_shrink"
#: the §1.8 adjustment a lora-native standup records when the site record's per-window S, re-derived over the launches whose adapter class
#: DOMINATES the class it serves, differs from the record-wide S the caller resolved (2026-10-02; the s2r pair target 831867). It is the one
#: adjustment that may RAISE S: the record-wide floor is a rank-64 x 128-adapter ceiling, not this standup's.
ADJUSTMENT_LAUNCH_CLASS = "launch_class"


def launch_class_adjustment(*, from_S: int, to_S: int, served_lora: Tuple[int, int], window: int, window_seqs_source: Optional[str],
                            launches: Sequence[str]) -> Dict[str, Any]:
    """The recorded ``launch_class`` step: ``{kind, from_S, to_S, max_lora_rank, max_loras, L, window_seqs_source, launches}``."""
    return {"kind": ADJUSTMENT_LAUNCH_CLASS, "from_S": int(from_S), "to_S": int(to_S), "max_lora_rank": int(served_lora[0]),
            "max_loras": int(served_lora[1]), "L": int(window), "window_seqs_source": window_seqs_source, "launches": list(launches)}
ADJUSTMENT_KINDS: Tuple[str, ...] = (ADJUSTMENT_LAUNCH_CLASS, ADJUSTMENT_KERNEL, ADJUSTMENT_KV)
#: the `launch_class` step's exact keys (geometry.launch_class_adjustment; 2026-10-02): the record-wide S it started from, the class-specific
#: S it reached, the served class (rank cap, adapters), the window, the derivation's source and the launch ids that set the window's S
LAUNCH_CLASS_ADJUSTMENT_FIELDS: Tuple[str, ...] = ("kind", "from_S", "to_S", "max_lora_rank", "max_loras", "L", "window_seqs_source", "launches")
#: the key of the ordered adjustment list inside a ``geometry`` block
ADJUSTMENTS_KEY = "adjustments"
#: the keys of one KERNEL step, exactly: ``{kind, from_S, to_S, D, ceiling}``
KERNEL_ADJUSTMENT_FIELDS: Tuple[str, ...] = ("kind", "from_S", "to_S", "D", "ceiling")
#: the widest ADAPTED module per model — the ``o_proj`` input width D the LoRA
#: kernel ceiling ``(2^31 - 1) // D`` is computed from. Pinned by test against
#: the ``lora_max_module_dim`` the pulled certification attestations recorded
#: (``$SISTEER_STATE_DIR/multilora-cert/<target>_<job>/unit/lora_native/
#: serving_attestation.json``: gemma4e4b 4096, qwen35_4b 4096, nemotron3nano4b
#: 5120); ``llama8b`` is the architecture's width (Llama-3.1-8B: 32 heads x 128
#: head_dim = 4096 = hidden_size, the certification target of this spec) — its
#: attestation lands with the llama8b units. A model absent here has no
#: re-derivable kernel step: :func:`kernel_ceiling_adjustment` refuses it by
#: name rather than guessing a width; add a row WITH its attested dim. The
#: ONBOARDED models are deliberately NOT here: their width is what the FD 24
#: serving probe MEASURED (the widest executing wrapper stride the worker
#: reported), folded into the SITE geometry overlay by the probe's verdict and
#: read through :func:`lora_max_module_dim_for` BEFORE this table.
LORA_MAX_MODULE_DIM_BY_MODEL: Mapping[str, int] = {
    "gemma4e4b": 4096,
    "qwen35_4b": 4096,
    "nemotron3nano4b": 5120,
    "llama8b": 4096,
}


def _pos_int_or_none(v: object) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


def kernel_ceiling_adjustment(model_key: str, L: int, S_identity: int, *, module_set: Optional[str] = None) -> Optional[dict]:
    """The LoRA-kernel step for ``model_key`` at window ``L`` from the identity
    S — PURE (no adapter dirs, no standup): ``{kind, from_S, to_S, D, ceiling}``
    with ``D = lora_max_module_dim_for(model_key, module_set)`` (the site overlay's
    probe-certified width of the SELECTED set first, else
    :data:`LORA_MAX_MODULE_DIM_BY_MODEL`), ``ceiling = (2^31-1) //
    D`` (``serving_attestation.lora_kernel_index_ceiling``) and ``to_S =
    ceiling // L``; ``None`` when the ceiling does not lower S at this L
    (``ceiling // L >= S_identity``) — the step is PRESENT IFF IT LOWERS. So the
    expected served S of a native standup is known BEFORE the standup
    (``expected_identity`` pre-computes it and the resume decision lands on the
    right store). Refuses (:class:`UnknownModelKey`) a model without a pinned
    width, and (``ValueError``) a window the ceiling admits no row at (the
    standup refuses the same)."""
    D_pin = lora_max_module_dim_for(model_key, module_set)
    if D_pin is None:
        raise UnknownModelKey(
            f"model key {model_key!r} has no pinned LoRA max module dim"
            f"{'' if module_set is None else f' for module set {module_set!r}'} "
            f"(geometry.LORA_MAX_MODULE_DIM_BY_MODEL knows {sorted(LORA_MAX_MODULE_DIM_BY_MODEL)}; site geometry "
            f"overlay {_site_overlay_where(model_key, module_set)}); "
            "the kernel-ceiling step cannot be re-derived for it — pin its attested "
            "lora_max_module_dim (the FD 24 serving probe's verdict fold writes it to the site overlay), never a guess")
    L_i, S_i = _pos_int_or_none(L), _pos_int_or_none(S_identity)
    if L_i is None or S_i is None:
        raise ValueError(f"kernel_ceiling_adjustment: L={L!r} and S_identity={S_identity!r} must be positive ints")
    D = int(D_pin)
    ceiling = _sa.lora_kernel_index_ceiling(D)
    to_S = ceiling // L_i
    if to_S < 1:
        raise ValueError(
            f"max_model_len={L_i} alone exceeds the LoRA kernel index ceiling {ceiling:,} tokens "
            f"(D={D} for {model_key!r}): no max_num_seqs can serve this window natively")
    if to_S >= S_i:
        return None
    return {"kind": ADJUSTMENT_KERNEL, "from_S": S_i, "to_S": to_S, "D": D, "ceiling": ceiling}


def adjusted_seqs(model_key: str, L: int, S_identity: int, *, module_set: Optional[str] = None) -> int:
    """The S a native standup SERVES at ``L`` before any KV step: the kernel
    step's ``to_S`` when it fires, else the identity S. What a launcher's
    expected identity and its pre-standup resume decision are computed from.
    ``module_set`` selects the certified per-set D of an onboarded model."""
    step = kernel_ceiling_adjustment(model_key, L, S_identity, module_set=module_set)
    return int(step["to_S"]) if step is not None else int(S_identity)


def kv_shrink_adjustment(shrink: Mapping) -> dict:
    """The KV step as an ``adjustments`` entry: ``{kind: "kv_shrink", from_S,
    to_S}`` plus EVERY field of the :func:`kv_shrink_record` it is made from
    (:data:`KV_SHRINK_FIELDS` and the record's provenance keys), inline, so
    :func:`kv_shrink_problem` validates the entry itself against the PREVIOUS
    step's S. ``from_S`` / ``to_S`` are the record's configured / served S."""
    if not isinstance(shrink, Mapping):
        raise ValueError(f"kv_shrink_adjustment: the shrink record is not a mapping ({type(shrink).__name__})")
    missing = [k for k in KV_SHRINK_FIELDS if k not in shrink]
    if missing:
        raise ValueError(f"kv_shrink_adjustment: the shrink record lacks {missing}")
    return {"kind": ADJUSTMENT_KV, "from_S": int(shrink["configured_max_num_seqs"]),
            "to_S": int(shrink["served_max_num_seqs"]), **{k: shrink[k] for k in shrink if k != "kind"}}


# --------------------------------------------------------------------------- #
# the truncation ladder's DELIBERATE reduction — beside the kernel and KV     #
# steps, the third named reason a reader accepts a served S below the         #
# budget's                                                                    #
# --------------------------------------------------------------------------- #
#
# A truncation-ladder rung (tinycap's bcb-bfcl-200 §S2b) serves the rung-2 window at an S the
# ladder REDUCED from the budget's identity S — deliberately: accepted after a KV refusal, or
# named by the operator — and the standup is ASKED that reduced S. The standup's own record
# (``adjustments``) therefore starts from what it was asked, not from the identity S a reader
# re-derives from the block's ``budget``; held to the §1.8 contract alone, a native ladder rung
# read as "a smaller S with no step" and the scoring driver refused every one, while the
# launcher — reading the same record against the S it had asked — certified it (two
# interpretations of one record). ``geometry.ladder_reduction`` is the ONE record of the
# reduction, beside ``adjustments`` the way ``kv_shrink`` sits beside them: written by the
# LAUNCHER (the standup knows no ladder) into the ``served_geometry.json`` it files beside the
# cells and composed into the block a reader validates. :func:`geometry_adjustment_problem`
# applies it FIRST — the identity S -> the requested rung — and every step after it re-derives
# from THAT S (the kernel step present iff it lowers the requested rung, the KV step from the
# previous S, the served S the last step's), so the probe that certifies a rung and the driver
# that scores it read one chain from the budget's identity to the served S. Absent, every
# reading is byte-identical to before. It is NOT an :data:`ADJUSTMENT_KINDS` member: the list is
# the standup's own record and a standup never writes this step. The identity rung (the
# requested S IS the identity S) has no record — present iff it reduces, like the kernel step.

#: the record's key inside a ``geometry`` block (the launcher's record; composed into the standup's
#: block by every reader that validates a ladder rung)
LADDER_REDUCTION_KEY = "ladder_reduction"
LADDER_REDUCTION_RULE = ("the standup is asked to_S, a rung of ladder_seqs below the identity S from_S at "
                         "max_model_len; its own adjustments follow from to_S")
LADDER_REDUCTION_FIELDS: Tuple[str, ...] = ("from_S", "to_S", "max_model_len", "ladder_seqs", "rung_source", "rule")
#: WHY the rung is below the identity S — the ladder's ``probe`` (accepted after a KV refusal) or
#: ``operator`` (``--ladder-rung``), the two below-identity words of ``rung_probe.RUNG_SOURCES``,
#: restated here (this module imports nothing of the ladder; ``rung_probe`` asserts the spellings
#: agree at import). ``identity`` never reduces, so it never appears in a record.
LADDER_REDUCTION_SOURCES: Tuple[str, ...] = ("probe", "operator")


def ladder_reduction_record(*, identity_S: int, requested_S: int, max_model_len: int,
                            ladder_seqs: Sequence[int], rung_source: str) -> dict:
    """The ladder's reduction as ONE record — the identity S at the window (``from_S``), the
    requested rung the standup is asked (``to_S``), the window, the REQUESTED ladder the rung is
    on (the budget's rungs at the window, descending from the identity) and why — validated by
    :func:`ladder_reduction_problem` before it is returned (``ValueError`` when it does not
    reduce, names a rung off the ladder, or an unknown source)."""
    rec = {"from_S": int(identity_S), "to_S": int(requested_S), "max_model_len": int(max_model_len),
           "ladder_seqs": [int(s) for s in ladder_seqs], "rung_source": str(rung_source),
           "rule": LADDER_REDUCTION_RULE}
    problem = ladder_reduction_problem(rec, L=int(max_model_len), S_identity=int(identity_S))
    if problem:
        raise ValueError(problem)
    return rec


def ladder_reduction_problem(reduction: object, *, L: int, S_identity: int) -> Optional[str]:
    """``None`` when ``reduction`` is a well-formed record of THIS rule for these numbers — exactly
    :data:`LADDER_REDUCTION_FIELDS`, ``from_S`` the identity S the reader re-derived, the window
    this run's, ``ladder_seqs`` a strictly descending ladder headed by that identity S, ``to_S``
    one of its rungs and strictly below ``from_S``, a below-identity source; else why not.
    Fail-closed: a reader accepts a ladder rung's reduction ONLY through this function."""
    if not isinstance(reduction, Mapping):
        return f"{LADDER_REDUCTION_KEY} is not a mapping ({type(reduction).__name__})"
    if set(reduction) != set(LADDER_REDUCTION_FIELDS):
        missing = sorted(set(LADDER_REDUCTION_FIELDS) - set(reduction))
        extra = sorted(set(reduction) - set(LADDER_REDUCTION_FIELDS))
        return (f"{LADDER_REDUCTION_KEY} keys != {list(LADDER_REDUCTION_FIELDS)}"
                + (f" (missing {missing})" if missing else "") + (f" (extra {extra})" if extra else ""))
    frm, to, L_rec = (_pos_int_or_none(reduction[k]) for k in ("from_S", "to_S", "max_model_len"))
    if None in (frm, to, L_rec):
        return f"{LADDER_REDUCTION_KEY}: from_S / to_S / max_model_len must be positive ints"
    if reduction["rule"] != LADDER_REDUCTION_RULE:
        return f"{LADDER_REDUCTION_KEY}: rule {reduction['rule']!r} is not {LADDER_REDUCTION_RULE!r}"
    L_i, S_id = _pos_int_or_none(L), _pos_int_or_none(S_identity)
    if L_i is None or S_id is None:
        return f"{LADDER_REDUCTION_KEY}: L={L!r} / S_identity={S_identity!r} are not positive ints — nothing to re-derive against"
    if L_rec != L_i:
        return f"{LADDER_REDUCTION_KEY}: max_model_len {L_rec} is not this run's {L_i}"
    if frm != S_id:
        return f"{LADDER_REDUCTION_KEY}: from_S {frm} is not the identity S {S_id} the budget buys at L={L_i}"
    seqs = reduction["ladder_seqs"]
    if (not isinstance(seqs, (list, tuple)) or not seqs
            or any(_pos_int_or_none(s) is None for s in seqs)):
        return f"{LADDER_REDUCTION_KEY}: ladder_seqs {seqs!r} is not a non-empty list of positive ints"
    seqs = [int(s) for s in seqs]
    if any(a <= b for a, b in zip(seqs, seqs[1:])):
        return f"{LADDER_REDUCTION_KEY}: ladder_seqs {seqs} is not strictly descending"
    if seqs[0] != frm:
        return f"{LADDER_REDUCTION_KEY}: ladder_seqs {seqs} is not headed by the identity S {frm} (the ladder walks down from it)"
    if to not in seqs:
        return f"{LADDER_REDUCTION_KEY}: to_S {to} is not a rung of the ladder {seqs}"
    if to >= frm:
        return f"{LADDER_REDUCTION_KEY}: to_S {to} is not below the identity S {frm} — not a reduction (the identity rung records none)"
    if reduction["rung_source"] not in LADDER_REDUCTION_SOURCES:
        return (f"{LADDER_REDUCTION_KEY}: rung_source {reduction['rung_source']!r} is not one of "
                f"{list(LADDER_REDUCTION_SOURCES)}")
    return None


def recorded_identity_seqs(geometry_block: Mapping) -> int:
    """The IDENTITY S a ``geometry`` block's OWN recorded ``budget`` buys at its window — what the
    §1.8 steps (and a ladder reduction) were applied TO: ``ModelBudget.seqs_at(L, budget=<the
    block's>)`` through the row ``block["model_key"]`` names (the hybrid per-slot cap and the
    measured window ceiling on top), the bare quotient (:func:`identity_seqs`) when it names none.
    Read off the block, never the live table: a standup under a provisional budget contracts
    THAT budget's S. The ONE derivation the scoring driver (``served_capability_run``), the tinycap
    launcher's production reading of a ladder rung and the validators share, so they cannot
    disagree on what "the identity S" of a record is. ``ValueError`` on a block naming no
    positive-int budget / window; :class:`UnknownModelKey` on a model the table does not know."""
    if not isinstance(geometry_block, Mapping):
        raise ValueError(f"the geometry block is not a mapping ({type(geometry_block).__name__})")
    L, budget = geometry_block.get("max_model_len"), geometry_block.get("budget")
    for name, v in (("max_model_len", L), ("budget", budget)):
        if _pos_int_or_none(v) is None:
            raise ValueError(f"geometry.{name} is {v!r}, not a positive int")
    model_key = geometry_block.get("model_key")
    if model_key is None:
        return int(identity_seqs(int(L), budget=int(budget)))
    entry = budget_for(str(model_key)).entry            # UnknownModelKey: the caller names the refusal
    return int(entry.seqs_at(int(L), budget=int(budget)))


def geometry_adjustments(geometry_block: Mapping) -> List[dict]:
    """The ordered adjustment list a ``geometry`` block records —
    ``geometry.adjustments`` verbatim when the key is present (absent reads
    ``[]``: a lens block, or a record from before the contract); when it is
    ABSENT and the block carries a ``kv_shrink`` record (today's shape,
    written before ``adjustments`` existed), the ONE derived entry
    ``[kv_shrink_adjustment(kv_shrink)]`` so a pre-contract KV-shrunk store
    reads through the one function byte-identically. Shape problems are left
    to :func:`geometry_adjustment_problem` (this returns the raw list)."""
    if not isinstance(geometry_block, Mapping):
        return []
    if ADJUSTMENTS_KEY in geometry_block:
        adj = geometry_block[ADJUSTMENTS_KEY]
        return list(adj) if isinstance(adj, (list, tuple)) else [adj]
    shrink = geometry_block.get(KV_SHRINK_KEY)
    if isinstance(shrink, Mapping) and not [k for k in KV_SHRINK_FIELDS if k not in shrink]:
        return [kv_shrink_adjustment(shrink)]
    return []


def geometry_adjustment_problem(geometry_block: Mapping, *, model_key: Optional[str], L: int,
                                S_identity: int, S_served: object,
                                native: Optional[bool] = None,
                                served_lora: Optional[Tuple[int, int]] = None) -> Optional[str]:
    """``None`` when the ``geometry`` block's ``adjustments`` (``[]`` / ``[kernel]``
    / ``[kv]`` / ``[kernel, kv]``, in that order) re-derive from ``model_key``,
    ``L`` and the identity S and end at ``S_served``; else the FIRST problem,
    named. The ONE acceptance rule for a served S below the budget's, so every
    reader (the launcher's endpoint read and resume decision, the driver's
    manifest ``geometry`` copy, the math / GSM8K / IFEval validators, the wave's
    post-standup compare) accepts exactly the same records:

    - ``[]`` -> ``S_served == S_identity`` (identity; a lens row, or a native
      row whose model's D does not lower at this L — llama8b at 4864);
    - a KERNEL step must come first, once: exactly
      :data:`KERNEL_ADJUSTMENT_FIELDS`, ``from_S == S_identity``, ``D ==
      LORA_MAX_MODULE_DIM_BY_MODEL[model_key]``, ``ceiling == (2^31-1) // D``,
      ``to_S == ceiling // L`` and ``to_S < from_S`` (a step on a model whose D
      does NOT lower at this L is a problem, not a no-op);
    - a KV step must come last, once, with ``from_S`` == the previous step's S
      (the kernel's ``to_S``, else the identity S — a ``kv_shrink`` from the
      identity S when a kernel step precedes it is a problem) and hold
      :func:`kv_shrink_problem` against that S at this L with its own ``to_S``
      as the served S (the record's fields ride inline —
      :func:`kv_shrink_adjustment`); a top-level ``kv_shrink`` record beside
      the list must be the SAME record;
    - ``S_served`` == the last step's ``to_S`` (a bare smaller S with no record
      refuses by name); the block's own ``max_num_seqs`` / ``max_model_len`` /
      ``identity``, when present, must agree (``identity`` True iff no step).

    ``native`` (keyword-only, default ``None``) is the engine when the caller
    knows it: ``True`` additionally requires the kernel step to be PRESENT
    whenever the formula lowers (a native record that omitted it is a problem);
    ``False`` refuses ANY kernel step (a lens engine has no LoRA kernel);
    ``None`` checks self-consistency only — which already refuses every record
    the plan's negatives name. ``model_key`` None with a kernel step present is
    a problem (the step cannot be re-derived); with no kernel step it is fine
    (the lens path never needs the width). ``seqs_source`` is NOT judged here —
    each reader has its own rule for it (``model_budget`` / ``kv_derived``) and
    keeps it.

    A :data:`LADDER_REDUCTION_KEY` record beside the list (the truncation
    ladder's deliberate reduction, the launcher's record) is applied FIRST, held
    to :func:`ladder_reduction_problem` against ``S_identity`` and ``L``: the
    chain then starts at its ``to_S`` — the requested rung the standup was asked
    — so the kernel step is present iff it lowers THAT S, the KV step follows as
    before, the served S is the last step's (``to_S`` itself when the list is
    empty) and ``identity`` is True iff no adjustment AND no reduction. Absent,
    every verdict is byte-identical."""
    if not isinstance(geometry_block, Mapping):
        return f"the geometry block is not a mapping ({type(geometry_block).__name__})"
    L_i, S_id = _pos_int_or_none(L), _pos_int_or_none(S_identity)
    if L_i is None or S_id is None:
        return f"L={L!r} / S_identity={S_identity!r} are not positive ints — nothing to re-derive against"
    S_srv = _pos_int_or_none(S_served)
    if S_srv is None:
        return f"the served S {S_served!r} is not a positive int"
    if "max_model_len" in geometry_block and geometry_block["max_model_len"] != L_i:
        return (f"the geometry block records max_model_len={geometry_block['max_model_len']!r}, not this "
                f"run's {L_i}")
    if "max_num_seqs" in geometry_block and geometry_block["max_num_seqs"] != S_srv:
        return (f"the geometry block records max_num_seqs={geometry_block['max_num_seqs']!r} but the S "
                f"served is {S_srv}")
    # the truncation ladder's DELIBERATE reduction, when the block records one, is applied FIRST — the
    # identity S -> the requested rung the standup was asked — and every step below re-derives from THAT S
    # (the standup's own record starts from what it was asked, not from the identity)
    reduction = geometry_block.get(LADDER_REDUCTION_KEY)
    S_start = S_id
    if reduction is not None:
        problem = ladder_reduction_problem(reduction, L=L_i, S_identity=S_id)
        if problem:
            return f"the {LADDER_REDUCTION_KEY} record does not hold: {problem}"
        S_start = int(reduction["to_S"])
    start_label = f"identity S={S_id}" if reduction is None else f"the {LADDER_REDUCTION_KEY}'s to_S={S_start} (identity S {S_id})"
    adjustments = geometry_adjustments(geometry_block)
    if ADJUSTMENTS_KEY in geometry_block and not isinstance(geometry_block[ADJUSTMENTS_KEY], (list, tuple)):
        return f"geometry.{ADJUSTMENTS_KEY} is not a list ({type(geometry_block[ADJUSTMENTS_KEY]).__name__})"
    if ADJUSTMENTS_KEY not in geometry_block:
        # the pre-contract shape: the ONE step is the kv_shrink record itself, so a record
        # that is present but not a complete shrink record derives nothing — named, never
        # read as "no adjustment" (a null kv_shrink is the lens writers' spelling of none)
        shrink = geometry_block.get(KV_SHRINK_KEY)
        if shrink is not None:
            if not isinstance(shrink, Mapping):
                return f"geometry.{KV_SHRINK_KEY} is not a mapping ({type(shrink).__name__}) — not a shrink record"
            missing = [k for k in KV_SHRINK_FIELDS if k not in shrink]
            if missing:
                return (f"geometry.{KV_SHRINK_KEY} lacks {missing} — not a complete shrink record (the fields are "
                        f"{list(KV_SHRINK_FIELDS)}), so no {ADJUSTMENT_KV!r} step can be re-derived from it")
    kinds = []
    for i, adj in enumerate(adjustments):
        if not isinstance(adj, Mapping):
            return f"geometry.{ADJUSTMENTS_KEY}[{i}] is not a mapping ({type(adj).__name__})"
        kind = adj.get("kind")
        if kind not in ADJUSTMENT_KINDS:
            return f"geometry.{ADJUSTMENTS_KEY}[{i}].kind={kind!r} is not one of {list(ADJUSTMENT_KINDS)}"
        kinds.append(kind)
    if len(kinds) != len(set(kinds)):
        return f"geometry.{ADJUSTMENTS_KEY} names a kind twice ({kinds}) — each adjustment is applied at most once"
    if kinds != [k for k in ADJUSTMENT_KINDS if k in kinds]:
        return (f"geometry.{ADJUSTMENTS_KEY} is ordered {kinds}; the contract's order is "
                f"[{ADJUSTMENT_LAUNCH_CLASS!r}, {ADJUSTMENT_KERNEL!r}, {ADJUSTMENT_KV!r}] (the kernel ceiling is computed from the "
                "class-adjusted S and the KV shrink from the kernel-lowered S, never the reverse)")
    # THE LAUNCH-CLASS STEP: re-derived from the PINNED site record, never waved through — the
    # record-wide S the identity names is where it starts, the class-specific S for the step's own (rank cap, adapters) at this window is
    # where it ends, and the launch ids / source are the record's; the caller's attested class, when known, must be the step's
    S_after_class = S_start
    class_label = start_label
    if ADJUSTMENT_LAUNCH_CLASS in kinds:
        step = adjustments[0]
        if model_key is None:
            return f"a {ADJUSTMENT_LAUNCH_CLASS!r} step with no model key — the class-specific S cannot be re-derived"
        if reduction is not None:
            return (f"a {ADJUSTMENT_LAUNCH_CLASS!r} step beside a {LADDER_REDUCTION_KEY} record — the class step re-sizes a budget-DERIVED S, "
                    "and a ladder rung's requested S is not derived")
        if set(step) != set(LAUNCH_CLASS_ADJUSTMENT_FIELDS):
            return (f"the {ADJUSTMENT_LAUNCH_CLASS!r} step's keys are {sorted(step)}, not exactly {sorted(LAUNCH_CLASS_ADJUSTMENT_FIELDS)}")
        rank_s, n_s = _pos_int_or_none(step.get("max_lora_rank")), _pos_int_or_none(step.get("max_loras"))
        if rank_s is None or n_s is None:
            return (f"the {ADJUSTMENT_LAUNCH_CLASS!r} step's class max_lora_rank={step.get('max_lora_rank')!r} / max_loras={step.get('max_loras')!r} "
                    "is not a pair of positive ints")
        if served_lora is not None:
            want = (_pos_int_or_none(served_lora[0]), _pos_int_or_none(served_lora[1]))
            if (rank_s, n_s) != want:
                return (f"the {ADJUSTMENT_LAUNCH_CLASS!r} step records class rank cap {rank_s} x {n_s} adapters but the standup attests "
                        f"{want[0]} x {want[1]} (its lora_config) — the step is not this standup's")
        if _pos_int_or_none(step.get("L")) != L_i:
            return f"the {ADJUSTMENT_LAUNCH_CLASS!r} step records L={step.get('L')!r}, not this run's {L_i}"
        if step.get("from_S") != S_start:
            return f"the {ADJUSTMENT_LAUNCH_CLASS!r} step's from_S={step.get('from_S')!r} is not the {start_label}"
        try:
            rb_wide = budget_for(model_key)
            rb_cls = budget_for(model_key, served_lora=(rank_s, n_s))
        except SiteGeometryError as e:
            return f"the site geometry overlay for {model_key!r} refuses the launch-class re-derivation — {e}"
        except UnknownModelKey as e:
            return str(e)
        exp_from, exp_to = int(rb_wide.entry.seqs_at(L_i)), int(rb_cls.entry.seqs_at(L_i))
        if exp_from != S_start:
            return (f"the pinned site record's record-wide S at L={L_i} is {exp_from}, not the {start_label} the {ADJUSTMENT_LAUNCH_CLASS!r} "
                    "step starts from — the record moved under this standup (re-derive against the generation it was sized from)")
        if exp_to == exp_from:
            return (f"a {ADJUSTMENT_LAUNCH_CLASS!r} step on {model_key!r} at L={L_i} where class rank cap {rank_s} x {n_s} adapters re-derives the same "
                    f"S {exp_from} — the step is present iff it changes S")
        if _pos_int_or_none(step.get("to_S")) != exp_to:
            return (f"the {ADJUSTMENT_LAUNCH_CLASS!r} step records to_S={step.get('to_S')!r}; the pinned record re-derives {exp_to} for class rank cap "
                    f"{rank_s} x {n_s} adapters at L={L_i} ({rb_cls.entry.window_seqs_source})")
        exp_launches = list((rb_cls.entry.window_seqs_launches or {}).get(L_i) or ())
        if list(step.get("launches") or []) != exp_launches:
            return (f"the {ADJUSTMENT_LAUNCH_CLASS!r} step names launches {list(step.get('launches') or [])}; the pinned record's launches that set "
                    f"S={exp_to} at L={L_i} for that class are {exp_launches}")
        if step.get("window_seqs_source") != rb_cls.entry.window_seqs_source:
            return (f"the {ADJUSTMENT_LAUNCH_CLASS!r} step records window_seqs_source={step.get('window_seqs_source')!r}; the pinned record derives "
                    f"{rb_cls.entry.window_seqs_source!r}")
        S_after_class = exp_to
        class_label = f"the {ADJUSTMENT_LAUNCH_CLASS!r} step's to_S={exp_to} ({start_label})"
    if native is False and ADJUSTMENT_KERNEL in kinds:
        return (f"a {ADJUSTMENT_KERNEL!r} step on a LENS engine — the lens has no LoRA kernel, so "
                "this record cannot be a lens standup's")
    # the kernel step: PRESENT IFF IT LOWERS (the "iff" needs the engine; the "if" does not)
    expected_kernel = None
    # the PINNED table's D (the sweep's models); a model whose geometry is certified PER MODULE SET (the site overlay) has no
    # single pinned D — read the table directly, never lora_max_module_dim_for(model_key), whose no-set answer is the unsteered
    # default's D (2026-09-30) and would re-derive a qkvo standup's step from wo's width
    D_pin = LORA_MAX_MODULE_DIM_BY_MODEL.get(model_key) if isinstance(model_key, str) else None
    # a per-set model: the record's own kernel step names the D its standup sized from, which must be one of the certified sets'
    # widths — the arithmetic is re-derived from THAT D (the pins, which know the SELECTED set, hold the record to that set's D
    # exactly); a record with NO kernel step is judged as an UNSTEERED standup's (the zero adapter's class)
    try:
        per_set = certified_module_dims(model_key) if isinstance(model_key, str) else {}
    except SiteGeometryError as e:
        return f"the site geometry overlay for {model_key!r} refuses — {e}"       # a revoked / unusable record is a problem, named, not a raise
    kernel_at = kinds.index(ADJUSTMENT_KERNEL) if ADJUSTMENT_KERNEL in kinds else None       # the kernel step's OWN index (after a class step: 1)
    if D_pin is None and per_set:
        if kernel_at is not None and isinstance(adjustments[kernel_at], Mapping):
            rec_D = _pos_int_or_none(adjustments[kernel_at].get("D"))
            if rec_D is None or rec_D not in per_set.values():
                return (f"a {ADJUSTMENT_KERNEL!r} step on {model_key!r} records D={adjustments[kernel_at].get('D')!r}, which is not a "
                        f"certified module set's width for this model (certified: {per_set}) — the step cannot be re-derived")
            D_pin = rec_D
        else:
            D_pin = per_set.get(UNSTEERED_MODULE_SET)
    if D_pin is not None and native is not False:
        try:
            D_set = next((s_ for s_, d_ in per_set.items() if d_ == D_pin), None) if per_set and model_key not in LORA_MAX_MODULE_DIM_BY_MODEL else None
            expected_kernel = kernel_ceiling_adjustment(model_key, L_i, S_after_class, module_set=D_set)   # from the CLASS-ADJUSTED S
        except ValueError as e:
            return str(e)
    prev_S = S_after_class
    if ADJUSTMENT_KERNEL in kinds:
        step = adjustments[kernel_at]
        if model_key is None:
            return (f"a {ADJUSTMENT_KERNEL!r} step with no model key — its D cannot be re-derived")
        if D_pin is None:
            return (f"a {ADJUSTMENT_KERNEL!r} step on model {model_key!r}, which has no pinned LoRA max "
                    f"module dim (geometry.LORA_MAX_MODULE_DIM_BY_MODEL knows "
                    f"{sorted(LORA_MAX_MODULE_DIM_BY_MODEL)}; site geometry overlay {_site_overlay_where(model_key)}) "
                    "— the step cannot be re-derived")
        if set(step) != set(KERNEL_ADJUSTMENT_FIELDS):
            return (f"the {ADJUSTMENT_KERNEL!r} step's keys are {sorted(step)}, not exactly "
                    f"{sorted(KERNEL_ADJUSTMENT_FIELDS)}")
        if expected_kernel is None:
            return (f"a {ADJUSTMENT_KERNEL!r} step on {model_key!r} at L={L_i}, where D="
                    f"{D_pin} does not lower the "
                    f"{'identity S' if (reduction is None and S_after_class == S_start) else ('requested rung S' if reduction is not None else 'class-adjusted S')} {S_after_class} "
                    f"(ceiling // L = {_sa.lora_kernel_index_ceiling(int(D_pin)) // L_i}) "
                    "— the step is present iff it lowers")
        for k in KERNEL_ADJUSTMENT_FIELDS:
            if step[k] != expected_kernel[k]:
                return (f"the {ADJUSTMENT_KERNEL!r} step records {k}={step[k]!r}; the formula gives "
                        f"{expected_kernel[k]!r} (D={expected_kernel['D']}, ceiling={expected_kernel['ceiling']:,}, "
                        f"L={L_i}, {class_label})")
        prev_S = int(step["to_S"])
    elif native is True and expected_kernel is not None:
        return (f"a native standup on {model_key!r} at L={L_i} must record the {ADJUSTMENT_KERNEL!r} step "
                f"({S_after_class} -> {expected_kernel['to_S']}; D={expected_kernel['D']}, ceiling="
                f"{expected_kernel['ceiling']:,}) and this record names none")
    top_shrink = geometry_block.get(KV_SHRINK_KEY)
    if ADJUSTMENT_KV in kinds:
        step = adjustments[-1]
        to_S = _pos_int_or_none(step.get("to_S"))
        if to_S is None:
            return f"the {ADJUSTMENT_KV!r} step's to_S={step.get('to_S')!r} is not a positive int"
        # today's rule FIRST, against the previous step's S (its wording for its failures —
        # "configured S 97 is not this run's configured 98" — is what the readers quote)
        problem = kv_shrink_problem(step, configured_max_num_seqs=prev_S, max_model_len=L_i,
                                    served_max_num_seqs=to_S)
        if problem:
            return f"the {ADJUSTMENT_KV!r} step does not hold: {problem}"
        from_S = step.get("from_S")
        if from_S != prev_S:
            return (f"the {ADJUSTMENT_KV!r} step's from_S={from_S!r} is not the previous step's S {prev_S}"
                    + (f" (the {ADJUSTMENT_KERNEL!r} step's to_S)" if ADJUSTMENT_KERNEL in kinds
                       else (f" (the {ADJUSTMENT_LAUNCH_CLASS!r} step's to_S)" if ADJUSTMENT_LAUNCH_CLASS in kinds else " (the identity S)")))
        if top_shrink is not None:
            if not isinstance(top_shrink, Mapping) or any(
                    top_shrink.get(k) != step.get(k) for k in KV_SHRINK_FIELDS):
                return (f"geometry.{KV_SHRINK_KEY} beside the {ADJUSTMENT_KV!r} step is not the same record "
                        "(the two spellings of one shrink must agree)")
        prev_S = to_S
    elif top_shrink is not None and ADJUSTMENTS_KEY in geometry_block:
        return (f"geometry.{KV_SHRINK_KEY} is present but geometry.{ADJUSTMENTS_KEY} names no "
                f"{ADJUSTMENT_KV!r} step — a torn record")
    if S_srv != prev_S:
        if not kinds:
            return (f"the served S {S_srv} is not {start_label} and the record names no adjustment "
                    "— a smaller S is accepted only through a named, re-derivable step")
        return f"the served S {S_srv} is not the last adjustment's to_S {prev_S}"
    if "identity" in geometry_block:
        want = not kinds and reduction is None
        if bool(geometry_block["identity"]) is not want:
            return (f"the geometry block records identity={geometry_block['identity']!r} with "
                    f"{len(kinds)} adjustment(s)"
                    + (f" and a {LADDER_REDUCTION_KEY} — identity is True iff no adjustment and no reduction applied"
                       if reduction is not None else " — identity is True iff no adjustment applied"))
    return None


def kv_shrunk_geometry(geom: ResolvedGeometry, shrink: Mapping) -> ResolvedGeometry:
    """``geom`` re-stated at the KV-derived S the record names (``seqs_source``
    :data:`KV_DERIVED_SOURCE`; ``is_identity`` false, honestly — the identity S
    is the budget's); the record is validated against ``geom``'s configured
    numbers first (``ValueError`` on a problem)."""
    problem = kv_shrink_problem(shrink, configured_max_num_seqs=geom.max_num_seqs,
                                max_model_len=geom.max_model_len,
                                served_max_num_seqs=shrink.get("served_max_num_seqs")
                                if isinstance(shrink, Mapping) else None)
    if problem:
        raise ValueError(problem)
    return replace(geom, max_num_seqs=int(shrink["served_max_num_seqs"]), seqs_source=KV_DERIVED_SOURCE)


def describe(entries: Optional[Mapping[str, ServedGeometry]] = None) -> str:
    """The table as markdown (for logs and manifests): one row per key with
    the geometry, its sizing branch, and the basis columns."""
    table = TABLE if entries is None else entries
    cols = ("key", "L", "S", "SxL", "identity", "sized_by", "n_calls", "p99_total",
            "max_total", "p99_prompt", "max_prompt", "max_tokens", "clamp@L", "frac>L",
            "n_rejected", "families", "measured_on", "flags")
    rows = []
    for g in table.values():
        b = g.basis
        flags = ",".join(f for f, on in (("certified", g.certified),
                                         ("requires_recertification", g.requires_recertification),
                                         ("provisional", g.provisional),
                                         ("analytic", g.basis.analytic is not None),
                                         ("informational", g.informational),
                                         ("over_budget", g.over_budget)) if on) or "-"
        rows.append((
            g.key, str(g.max_model_len), str(g.max_num_seqs), f"{g.batched_tokens:,}",
            "yes" if g.is_identity else f"no ({identity_seqs(g.max_model_len)})",
            g.sized_by, _cell(b.n_calls), _cell(b.p99_total), _cell(b.max_total),
            _cell(b.p99_prompt), _cell(b.max_prompt), _cell(b.requested_max_tokens),
            _cell(b.clamp_rate_at_L),
            _cell(b.frac_over_L), _cell(b.n_rejected), ",".join(b.families_measured),
            b.measured_on, flags,
        ))
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    out.append("")
    out.append(f"budget = DEFAULT_MAX_NUM_SEQS x {PETRI_MAX_MODEL_LEN} = "
               f"{BATCHED_TOKEN_BUDGET:,} batched tokens (the certified Petri "
               f"footprint); S = identity_seqs(L) = budget // L. Sizing rule: L = next "
               f"block >= p99_total unless the clamp rate there exceeds "
               f"{CLAMP_RATE_CEILING:.0%}, then L = next block >= p99_prompt + max_tokens "
               f"(headroom <= {SIZING_HEADROOM}); an entry whose driver refuses a clamped "
               f"cap-bound sample is held to max_prompt + max_tokens instead "
               f"({SIZED_BY_MAX_PROMPT_PLUS_MAX}, clamp 0 by construction). Basis: {MEASUREMENT_DOC} "
               f"(the original pass measured {MEASURED_ON}; entries carry their own "
               f"measured_on; provisional entries are {UNMEASURED} and serve only through "
               f"pass1_geometry at the certified footprint — an analytic one carries a "
               f"derived window ({SIZED_BY_ANALYTIC}) it is not served at; informational "
               f"entries are served by no launcher; a basis with n_capless counts its "
               f"cap-less calls, which are outside the clamp basis).")
    return "\n".join(out)


def _cell(v) -> str:
    return "-" if v is None else str(v)


__all__ = [
    "BATCHED_TOKEN_BUDGET", "PETRI_MAX_MODEL_LEN", "VLLM_BLOCK_SIZE",
    "MODEL_KEYS", "ONBOARDED_MODEL_KEYS", "KNOWN_MODEL_KEYS", "ONBOARDED_MODEL_HF_IDS",
    "ONBOARDED_HYBRID_MODEL_KEYS", "ONBOARDED_BUDGETS", "ModelRows", "rows_for",
    "MEASURED_BUDGET_SOURCE", "DEFAULT_BUDGET_SOURCE",
    "PROVISIONAL_BUDGET_SOURCE", "PER_MODEL_BUDGET_SOURCES", "UNMEASURED",
    "HYBRID_MODEL_KEYS", "HYBRID_SEQS_CAP", "HYBRID_WINDOW_SEQS_CEILINGS", "WindowSeqsCeiling", "MODEL_HF_IDS", "model_key_for",
    "is_hybrid", "launch_class", "least_admitted_by_window", "LAUNCH_CLASS_KEYS", "WINDOW_SEQS_RECORD_WIDE", "ADJUSTMENT_LAUNCH_CLASS", "launch_class_adjustment",
    "MULTITURN_CAP4096_CAP", "MULTITURN_CAP4096_WINDOWS", "MULTITURN_CAP4096_MEASURED_MAX_CONTEXT",
    "MULTITURN_CAP4096_SOURCE", "GeometryCapExceedsBasis",
    "BudgetBasis", "ModelBudget", "ResolvedBudget", "BUDGET_BY_MODEL",
    "UnknownModelKey", "budget_for", "describe_budgets",
    "SITE_GEOMETRY_DIR_ENV", "SITE_DIR_ENV", "SHARED_ROOT_ENV", "SITE_GEOMETRY_SUBDIR", "SITE_GEOMETRY_SHARED_SUBDIR",
    "SITE_GEOMETRY_VERSION", "SITE_GEOMETRY_METHOD", "SITE_GEOMETRY_PROVENANCE_REQUIRED", "SITE_GEOMETRY_SET_REQUIRED",
    "SITE_GEOMETRY_SUMMARY_NOTE",
    "SiteGeometryError", "SiteGeometry", "SiteGeometrySet", "site_geometry_dir", "site_geometry_path", "parse_site_geometry",
    "site_geometry_record", "load_site_geometry", "site_geometry_for", "write_site_geometry", "lora_max_module_dim_for",
    "certified_module_dims", "standup_adapter_parity_gate", "PROBE_RECORD_READY", "STAGING_PREDICATE_FIELDS", "site_geometry_bytes", "write_site_geometry_file", "canonical_site_geometry_sha256",
    "SITE_GEOMETRY_CANONICAL_SEPARATORS",
    "CLAMP_RATE_CEILING", "SIZING_HEADROOM", "MEASUREMENT_DOC", "MEASURED_ON",
    "UNMEASURED", "PROVISIONAL_SOURCE", "MEASUREMENT_SOURCE", "TINYCAP_KEYS",
    "SIZED_BY_TOTAL", "SIZED_BY_PROMPT_PLUS_MAX", "SIZED_BY_UNMEASURED", "SIZED_BY_ANALYTIC",
    "SIZED_BY_MAX_PROMPT_PLUS_MAX",
    "GeometryBasis", "ServedGeometry", "ResolvedGeometry",
    "KV_DERIVED_SOURCE", "KV_SHRINK_KEY", "KV_SHRINK_RULE", "KV_SHRINK_FIELDS",
    "kv_derived_seqs", "kv_shrink_record", "kv_shrink_problem", "kv_shrunk_geometry",
    "ADJUSTMENT_KINDS", "ADJUSTMENT_KERNEL", "ADJUSTMENT_KV", "ADJUSTMENTS_KEY", "LAUNCH_CLASS_ADJUSTMENT_FIELDS",
    "KERNEL_ADJUSTMENT_FIELDS", "LORA_MAX_MODULE_DIM_BY_MODEL",
    "kernel_ceiling_adjustment", "adjusted_seqs", "kv_shrink_adjustment",
    "geometry_adjustments", "geometry_adjustment_problem",
    "LADDER_REDUCTION_KEY", "LADDER_REDUCTION_RULE", "LADDER_REDUCTION_FIELDS", "LADDER_REDUCTION_SOURCES",
    "ladder_reduction_record", "ladder_reduction_problem", "recorded_identity_seqs",
    "UnknownEvalGeometry", "GeometryOverBudget", "GeometryIdentityBroken",
    "GeometryRequiresRecertification", "GeometryProvisional", "GeometryInformational",
    "TABLE", "KNOWN_EVALS", "KNOWN_KEYS", "next_block", "identity_seqs",
    "WINDOW_ALIGN", "align_window", "derived_window", "multiturn_window", "multiturn_rerun_window",
    "pin_window", "multiturn_geometry",
    "sizing_problem", "parse_key", "geometry_for", "resolve_geometry",
    "pass1_geometry", "provisional_keys", "informational_keys", "describe",
]
