"""LLMHFSteered — the defense wrapper for the jailbreak-robustness eval.

One HF forward path so the attacker and the eval hit the SAME object under
every defense (spec). The wrapper is deliberately thin: it reuses the
extraction objects and ``generate.generate_with_object`` verbatim (the
per-chunk ``obj.apply`` fix lives there) and never re-opens a steering
hook of its own.

The defense axis is the FULL object axis: ``none`` + ``prompted`` + every
compiled arm in ``methods.ARMS`` as a SINGLE-object safety defense (24
defenses at the current wiring: 22 wired arms of the 27-arm registry).
Each compiled arm is one object distilled from the same held-in
rejection-sampled safety pairs via ``methods.ARMS[arm].train`` (its native
loss), applied uniformly to every eval goal:

- ``none``     — no intervention, no system prompt (``obj=None``).
- ``prompted`` — the safety SI as a system message (``build_chat_prompt``),
  no object; the authoritative-safety-SI arm (kept authoritative multi-turn).
- one mode PER compiled arm (registry order, ``vector`` through the
  ``did``/``adasteer`` arms wired 2026-08-19) — the compiled object
  re-hydrated by ``methods.load_object`` (dispatch on stored kind).
  ``obj.inject`` is identity for non-token kinds and sentinel insertion for
  the two token kinds; steering is applied by ``obj.apply`` (per-chunk, in
  ``generate_with_object``), so both ``generate`` and multi-turn ``chat`` hit
  the same object and scale-0 is a per-kind no-op.

**Query accounting (JB7).** ``n_target_forwards`` is incremented at THIS
boundary — ``+len(goals)`` per single-shot ``generate`` call and ``+1`` per
``chat`` turn — so it is the authoritative query metric that already counts
backtracks/retries/summary calls that don't advance an attack's own "turn"
counter. Greedy decoding by default (HarmBench's standardised 512-token
greedy eval); BoN passes ``do_sample=True, temperature=1.0`` with a per-
variant ``sampling_seed`` so the exact same decode is reproducible and can
be presented identically to every defense (JB6).
"""

from __future__ import annotations

from ..generate import encode_kwargs, generate_with_object
from ..methods import ARMS, load_object
from ..models import apply_chat_template_folding, build_chat_prompt

# The compiled object-arm defense modes (== the main object axis, in
# registry order: the 12 plan arms, the 3 hard-LL/MSE-all PSR arms, the 2
# divergence arms, the 3 brought-in defense arms antidote/alphasteer/
# lofit, the 3 alphakl rungs, the diffmean floor, bidpo_lora and
# caa_static_r, then the later evals arms appended last
# (did/adasteer and the cheap-estimator caa/actadd/actadd_front) — cast/
# circuit_breaker stay OUT of the registry per the 2026-08-20 removal
# (load_object still hydrates their on-disk artifacts). A compiled
# defense loads ONE object of this arm and steers. Sized from ARMS,
# never a literal, so the count tracks the registry — deliberately
# tuple(ARMS), not a suite-filtered subset (the two-SI cell path,
# mode=cell.arm, needs its arms IN here). Which arms are WIRED into the
# single-SI jailbreak config is a separate, narrower list (the config's
# defenses: + test_jailbreak_run's exclusion tuple).
COMPILED = tuple(ARMS)
# The STACKED mode: a compiled object AND the SI as the authoritative system
# message, applied TOGETHER (the complementarity test — activation steering and
# prompting fail on DIFFERENT goal sets, so stacking should fix their union
# unless the two interventions interfere or compound over-refusal). It is a
# MODE, not an arm: WHICH object it stacks is carried by the cell
# (defense_cell.arm), the SI by the cell (defense_cell.si_id). Kept OUT of
# COMPILED (that tuple is the object-arm registry) so no arm-major array index
# or per-arm loop changes.
STACKED = "stacked"
# The defense axis: none + prompted + every compiled arm, plus the stacked
# mode. ``stacked`` is appended LAST so every position-indexed use of
# DEFENSES (and the arm-major two-SI array decode) is byte-for-byte unchanged.
DEFENSES = ("none", "prompted", *COMPILED, STACKED)

# Position modes the jailbreak defense exposes as a per-run option.
POSITIONS = ("all", "response")


def safety_si():
    """The benign safety system-instruction (an Instruction object — the
    thing the vector/token distills and the prompted arm sends)."""
    from ..data.instructions import SAFETY_SI

    return SAFETY_SI


def resolve_instruction(si_id: str):
    """The ``Instruction`` for a system-instruction id — used to give each of the
    two-SI matrix's TWO prompted baselines its own system text. Checks the
    ``SAFETY_SI`` module constant (id ``safety_refusal``, deliberately OUT of the
    static ``instructions.REGISTRY``) first, then the registered SIs in
    ``instructions.BY_ID``. Raises loudly on an unknown id rather than silently
    prompting with the wrong SI."""
    from ..data.instructions import BY_ID, SAFETY_SI

    if si_id == SAFETY_SI.id:
        return SAFETY_SI
    if si_id in BY_ID:
        return BY_ID[si_id]
    raise KeyError(f"unknown system-instruction id {si_id!r}")


def safety_text() -> str:
    """The benign safety system-instruction text."""
    return safety_si().text


def _render_chat(tok, messages: list[dict]) -> str:
    """Multi-turn render with the chat-template system-fold branch shared with
    ``models.build_chat_prompt``; keeps its OWN ``role: content`` plain-string
    fallback for templateless models."""
    if getattr(tok, "chat_template", None):
        return apply_chat_template_folding(tok, messages)
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    return "\n".join(lines) + "\nassistant:"


def _truncate_to_last_user(messages: list[dict]) -> list[dict]:
    """Messages up to and INCLUDING the last user turn (the response boundary
    is the end of the user's content; anything after is scaffolding + reply)."""
    msgs = [dict(m) for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            return msgs[: i + 1]
    return msgs


def response_prompt_len(tok, messages: list[dict]) -> int:
    """Token index where the assistant's turn begins: the length of the chat
    prompt rendered through the LAST user turn with ``add_generation_prompt=
    False``. Positions ``>=`` this index (the ``<|eot_id|><|start_header_id|>
    assistant…`` scaffolding plus every generated token) are the "response";
    the system + user content sits strictly before it.

    Computed from the SAME tokenizer/template a run uses (``build_chat_prompt``
    / ``_render_chat``) and encoded with the SAME ``encode_kwargs`` as
    generation, so the count lands in the generation prompt's own token
    coordinates. Multi-turn (Crescendo / ActorAttack) is handled: the boundary
    is the end of the last user message.

    OFF-DISTRIBUTION by construction: training applied the intervention at ALL
    positions (prompt + scaffolding + completion) and only restricted the LOSS
    to completion tokens, so the objects were fit assuming the prompt is steered
    too. positions="response" is therefore a deliberate train/test mismatch."""
    msgs = _truncate_to_last_user(messages)
    if getattr(tok, "chat_template", None):
        head = apply_chat_template_folding(tok, msgs, add_generation_prompt=False)
    else:
        # Templateless (gpt2 smoke): no assistant-header tokens exist, so the
        # boundary is the whole plain-concat prompt — response mode then steers
        # only truly-generated tokens. Exact only for templated models.
        head = _render_chat(tok, msgs)
    return len(tok(head, **encode_kwargs(tok, head))["input_ids"])


# H — the trailing assistant-header token count; the response boundary is
# prompt_width - H (the header IS steered, so the forward that produces the
# FIRST generated token — for a refusal defense the decisive one — runs
# steered). ONE home for the derivation (rendered-diff first, header-marker
# fallback for flag-ignoring templates like the NousResearch mirror):
# sisteer.hooks.response_header_len — the SAME H the fit-time mask leg uses,
# so train and test can never derive different boundaries. Re-exported here
# because this module is the jailbreak positions API (its tests monkeypatch
# the name on THIS module).
from ..hooks import response_header_len  # noqa: E402  (single mask home)


def _check_recorded_header(obj, *, width: int) -> None:
    """A loaded object's RECORDED fit boundary must equal this
    tokenizer's own width derivation before a response serve arms it —
    like with like: ``hooks.response_header_len`` is the deriver the
    FIT-TIME mask used, so on the template the object was fit against,
    the record and the serve-time width are the same number (gemma-4
    artifacts record 3; this tokenizer derives 3). A different number
    means the object was fit against a DIFFERENT chat template, so a
    mask armed here would cover a region the fit never saw — refuse,
    never guess. ``None`` (all-position fits, fresh objects) is not a
    record and passes. (Deliberately NOT compared against
    ``derive_assistant_markers``'s terminator-anchored header, whose
    definition legitimately disagrees with the width on templates with
    post-terminator scaffold — see chat_markers.generation_header_ids.)"""
    for attr in ("fit_response_header_len", "response_header_len"):
        rec = getattr(obj, attr, None)
        if rec is None or isinstance(rec, bool) or not isinstance(rec, int):
            continue
        if rec != width:
            raise ValueError(
                f"positions='response': the loaded object records "
                f"{attr}={rec}, but this tokenizer's assistant header is "
                f"H={width} (hooks.response_header_len — the SAME deriver "
                "the fit-time mask used) — the object was fit against a "
                "different chat template, so a response mask armed here "
                "would cover a region the fit never saw; refusing rather "
                "than serving it")


def _set_response_positions(obj, positions: str, tok) -> None:
    """Set the SERVE-phase position mode on a loaded compiled object, or
    raise a clear error for an arm whose intervention cannot honour a mask.

    This is the serve half of the fit x serve 2x2 (2026-08-21): any
    maskable object can be served all-position OR response-only regardless
    of which leg it was FITTED on, so all four cells (matched all/all and
    response/response, plus both cross-phase transfers) are reachable from
    this one seam. Consequences, both deliberate:

    - Maskable = any object exposing ``set_fit_positions``
      (hooks.PositionMaskedFit — psr*/lora*/lofit/loreft/diffmean/
      vector_all/...) or carrying the vector deployment attrs
      (``positions``/``response_header_len`` — VectorObject: the vector and
      bidpo artifacts). The earlier residual-addition-only allowlist
      predated the PositionMaskedFit extension and was stale.
    - ``positions`` is ALWAYS applied, "all" included: an explicit "all"
      ACTIVELY resets a persisted response-fit object (the
      fit=response/serve=all cell). The old early-return silently left such
      an object in response mode — the wrong leg, and that cell was
      unreachable. Objects that expose NEITHER seam (the soft-token arms,
      antidote's intrinsic last-token deployment, the closed-form Class B
      objects) keep the historical no-op under "all" — nothing about plain
      Class A/B serving changes — and refuse "response" with their real
      reason (FIT_MASK_REFUSAL where the class declares one).

    The FIT leg each artifact ran is recorded per object
    (train_budget_meta.json ``positions_at_fit``; the torch-blob objects
    also restore it at load) — the eval path reads that record to assert
    or deliberately cross the phases; this helper only sets the serve leg.
    """
    if positions not in POSITIONS:
        raise ValueError(f"positions must be one of {POSITIONS}, got {positions!r}")
    # GUARD (2026-08-20): a zero header length silently neuters the whole
    # intervention and every test still passes. The boundary would collapse to
    # prompt_width, putting the assistant header on the PROMPT side, so the
    # forward that produces the FIRST generated token is unsteered -- and for a
    # refusal defense the first token is the decisive one. This was live on a
    # chat template that ignores add_generation_prompt. Never silent again:
    # response mode with H == 0 is a hard error, on ANY tokenizer.
    header = None
    if positions == "response":
        _h = response_header_len(tok)
        if _h <= 0:
            raise ValueError(
                "positions='response' requires a non-zero assistant-header "
                f"length, got H={_h}. The chat template exposes no header "
                "suffix, so the steering boundary would collapse to the full "
                "prompt width and the first generated token would be produced "
                "from an UNSTEERED position -- silently disabling the defense. "
                "Fix the template/tokenizer or use positions='all'."
            )
        # The WIDTH is the serve boundary: response_header_len is the
        # deriver the FIT-TIME mask used, so arming anything else would
        # steer outside the fit support (the terminator-anchored located
        # header absorbs prompt-side scaffold on gemma-4 — see
        # chat_markers.generation_header_ids). The loaded object's own
        # recorded boundary is checked like-with-like first.
        _check_recorded_header(obj, width=_h)
        header = _h

    # An arm may be mechanically maskable and still have no meaningful
    # decode-only cell. SERVE_MASK_REFUSAL is that opt-out, checked BEFORE
    # the seams: unlike FIT_MASK_REFUSAL ("cannot express a mask") it means
    # "can, but the cell would run a different method under this arm's name"
    # -- LoReFT, whose prompt-only support IS its identity. Refuse loudly
    # rather than serve a number nobody should compare.
    _serve_refusal = getattr(obj, "SERVE_MASK_REFUSAL", None)
    if positions != "all" and _serve_refusal:
        raise ValueError(
            f"--positions {positions!r}: {type(obj).__name__} is excluded "
            f"from decode-only serving — {_serve_refusal}. Serve it with "
            "--positions all, or pick a different defense."
        )

    set_fit = getattr(obj, "set_fit_positions", None)
    # The vector deployment attrs are a REAL seam only where the class
    # defines them (VectorObject / MultiLayerVectorObject / PSRObject);
    # hasattr keeps intrinsic-deployment objects (antidote's "last") and
    # everything else out of this branch rather than growing stray attrs.
    has_deploy = (hasattr(obj, "positions")
                  and hasattr(obj, "response_header_len"))
    if set_fit is None and not has_deploy:
        if positions == "all":
            return  # never-maskable object serving as itself — the no-op
        reason = getattr(obj, "FIT_MASK_REFUSAL", None) or (
            "it exposes neither a fit-position mask (set_fit_positions) nor "
            "the vector deployment attributes")
        raise ValueError(
            f"--positions {positions!r}: the loaded object "
            f"{type(obj).__name__} cannot honour a position mask — {reason}. "
            "Re-run with --positions all, or pick a maskable defense."
        )
    if set_fit is not None:
        # One mask home: the same PositionMaskedFit leg the masked FIT used
        # (SteeringHandle._position_mask underneath), never a serve-side
        # reimplementation.
        set_fit(positions, header)
    if has_deploy:
        obj.positions = positions
        obj.response_header_len = header


# The composer lives in the stdlib-only `system_prefix` module so the seedrep ladder can import it into a
# torch-less app process; re-exported here under its original name -- every caller and every patch of this name still works.
from .system_prefix import install_authoritative_system  # noqa: E402,F401


class LLMHFSteered:
    """A single steered/prompted/plain target with an authoritative
    query counter. ``generate_fn`` is a test seam (rendered prompts ->
    completions) that bypasses the model; the counter still increments at
    this boundary so query accounting is tested without a GPU."""

    def __init__(
        self,
        model,
        tok,
        *,
        mode: str,
        obj=None,
        scale: float = 1.0,
        system: str | None = None,
        batch_size: int = 8,
        generate_fn=None,
    ):
        if mode not in DEFENSES:
            raise ValueError(f"mode must be one of {DEFENSES}, got {mode!r}")
        if mode in COMPILED and obj is None:
            raise ValueError(f"defense {mode!r} needs a compiled object")
        if mode == "prompted" and not system:
            raise ValueError("defense 'prompted' needs the safety system text")
        if mode in ("none", "prompted") and obj is not None:
            raise ValueError(f"defense {mode!r} takes no object")
        # STACKED needs BOTH halves: the compiled object AND the SI system text.
        if mode == STACKED and obj is None:
            raise ValueError("defense 'stacked' needs a compiled object")
        if mode == STACKED and not system:
            raise ValueError("defense 'stacked' needs the safety system text")
        self.model = model
        self.tok = tok
        self.mode = mode
        self.obj = obj
        self.scale = float(scale)
        self.system = system
        self.batch_size = batch_size
        self._generate_fn = generate_fn
        # JB7: the authoritative query metric, at THIS boundary.
        self.n_target_forwards = 0

    @property
    def _obj(self):
        """The intervention object for compiled arms AND the stacked mode,
        else None (stacked applies the same per-chunk ``obj.apply`` as a
        compiled arm)."""
        return self.obj if self.mode in COMPILED or self.mode == STACKED else None

    def _system_for(self) -> str | None:
        # prompted AND stacked both send the SI as the single-shot system
        # message (build_chat_prompt); stacked additionally steers via _obj.
        return self.system if self.mode in ("prompted", STACKED) else None

    def generate(
        self,
        goals: list[str],
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_seed: int | None = None,
    ) -> list[str]:
        """Single-shot generation for a batch of raw goal strings.

        Counts ``len(goals)`` target forwards (JB7). ``sampling_seed`` makes
        ``do_sample`` decoding reproducible (BoN target-sampling RNG, JB6)."""
        self.n_target_forwards += len(goals)
        rendered = [
            build_chat_prompt(self.tok, g, system=self._system_for()) for g in goals
        ]
        if self._generate_fn is not None:
            return self._generate_fn(
                rendered, do_sample=do_sample, temperature=temperature,
                sampling_seed=sampling_seed, max_new_tokens=max_new_tokens,
            )
        if do_sample and sampling_seed is not None:
            import torch

            torch.manual_seed(sampling_seed)
        return generate_with_object(
            self.model, self.tok, rendered, self._obj, self.scale,
            batch_size=self.batch_size, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        sampling_seed: int | None = None,
    ) -> str:
        """Multi-turn generation (Crescendo / ActorAttack path). One code
        path with ``generate`` so the defense applies identically single-
        and multi-turn; counts ONE target forward (JB7 — backtracks and
        retries call this again and are counted, even when they don't
        advance an attack's turn counter)."""
        self.n_target_forwards += 1
        msgs = [dict(m) for m in messages]
        if self.mode in ("prompted", STACKED) and self.system:
            # HIGH: always install the safety SI as the AUTHORITATIVE system
            # message, folding any attack-supplied system content into user
            # text so it cannot REPLACE the safety SI in multi-turn.
            msgs = install_authoritative_system(msgs, self.system)
        rendered = _render_chat(self.tok, msgs)
        if self._generate_fn is not None:
            return self._generate_fn(
                [rendered], do_sample=do_sample, temperature=temperature,
                sampling_seed=sampling_seed, max_new_tokens=max_new_tokens,
            )[0]
        if do_sample and sampling_seed is not None:
            import torch

            torch.manual_seed(sampling_seed)
        # generate_with_object handles token-sentinel injection and the
        # per-chunk obj.apply for the compiled arms; obj=None is plain.
        out = generate_with_object(
            self.model, self.tok, [rendered], self._obj, self.scale,
            batch_size=1, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature,
        )
        return out[0]


#: Sentinel for ``positions`` — there is NO default. ``build_defense`` /
#: ``build_defense_cell`` REQUIRE an explicit ``positions`` because "all" is no
#: longer a no-op: ``_set_response_positions`` now ACTIVELY resets a
#: masked-hydrated object to all-position, so an OMITTED positions would silently
#: re-serve a matched decode-only (Class C) fit as the reverse cross-phase cell —
#: correctly labelled by the three-way probe, but answering the WRONG question.
#: Every sweep construction path must state the leg it means. Kept a sentinel
#: (not a bare required kwarg) so the error names the footgun rather than a
#: generic TypeError.
_POSITIONS_REQUIRED = object()


def build_defense(
    model,
    tok,
    *,
    mode: str,
    object_path=None,
    scale: float = 1.0,
    system: str | None = None,
    batch_size: int = 8,
    positions=_POSITIONS_REQUIRED,
    generate_fn=None,
) -> LLMHFSteered:
    """Factory for the 14 defenses. Any of the 12 compiled arms re-hydrates
    its single ``object_path`` artifact through ``methods.load_object``
    (dispatch on stored kind) — one code path for every kind, so
    ``vector`` and ``bidpo`` (safetensors), the two token kinds (sentinel
    injection), PSR/LoReFT/ReFT/all-layer blobs all build identically.
    ``prompted`` defaults to the registered safety SI text; ``none`` takes
    no object.

    ``positions`` is MANDATORY-EXPLICIT — there is no default (see
    :data:`_POSITIONS_REQUIRED`). ``"all"`` steers every position (and now
    ACTIVELY resets a persisted response-fit object to all-position, so it is a
    real choice, not a no-op); ``"response"`` steers ONLY the model's own
    generated tokens plus the assistant-header scaffolding after the user's turn —
    a deliberate train/test mismatch. ``"response"`` is rejected for
    ``none``/``prompted`` (no object) and for arms whose intervention cannot
    honour a position mask (loreft's ``SERVE_MASK_REFUSAL``, the soft tokens'
    ``FIT_MASK_REFUSAL``)."""
    if positions is _POSITIONS_REQUIRED:
        raise TypeError(
            "build_defense requires an explicit positions=... ('all' or "
            "'response'). The 'all' default was REMOVED because 'all' now "
            "ACTIVELY RESETS a masked-hydrated object rather than being a no-op — "
            "an omitted positions would silently serve the reverse cross-phase "
            "cell (a matched decode-only/Class-C fit re-served all-position), "
            "correctly labelled by the probe but answering the wrong question. "
            "State the leg on every sweep construction path.")
    if mode == "none":
        if positions != "all":
            raise ValueError(
                f"--positions {positions!r} only applies to a compiled steering "
                f"arm; defense 'none' has no intervention to mask")
        return LLMHFSteered(model, tok, mode="none", batch_size=batch_size,
                            generate_fn=generate_fn)
    if mode == "prompted":
        if positions != "all":
            raise ValueError(
                f"--positions {positions!r} only applies to a compiled steering "
                f"arm; defense 'prompted' steers via a system message, not a "
                f"position-maskable intervention")
        return LLMHFSteered(model, tok, mode="prompted",
                            system=system or safety_text(),
                            batch_size=batch_size, generate_fn=generate_fn)
    if mode in COMPILED:
        if object_path is None:
            raise ValueError(
                f"defense {mode!r} needs object_path (the compiled {mode} artifact)")
        obj = load_object(object_path, model, tok)
        _set_response_positions(obj, positions, tok)
        return LLMHFSteered(model, tok, mode=mode, obj=obj, scale=scale,
                            batch_size=batch_size, generate_fn=generate_fn)
    if mode == STACKED:
        # The stacked defense = a compiled object (loaded EXACTLY as a compiled
        # arm — same load_object dispatch, same position seam) served WITH the
        # SI as the authoritative system message (as ``prompted`` does). The two
        # halves compose the way the existing arms compose ``apply()`` with the
        # system prompt: the object edits the residual stream the system-prompted
        # context produces. ``positions`` masks the OBJECT half exactly as for a
        # compiled arm; the SI half is unmasked (it is prompt text, not a
        # position-maskable intervention). ``system`` defaults to the registered
        # safety SI, but the cell path passes the resolved si_id text so other
        # SIs can be stacked.
        if object_path is None:
            raise ValueError(
                "defense 'stacked' needs object_path (the compiled arm artifact)")
        obj = load_object(object_path, model, tok)
        _set_response_positions(obj, positions, tok)
        return LLMHFSteered(model, tok, mode=STACKED, obj=obj, scale=scale,
                            system=system or safety_text(),
                            batch_size=batch_size, generate_fn=generate_fn)
    raise ValueError(f"unknown defense mode {mode!r}; known: {DEFENSES}")


#: The fit driver's per-cell provenance sidecar (train_two_si_objects.py
#: writes it next to every published artifact) — the authoritative record of
#: the positions leg the object was FITTED under.
TRAIN_BUDGET_META = "train_budget_meta.json"


def cell_positions_at_fit(cell) -> str:
    """The positions leg ``cell``'s artifact was FITTED under — read from the
    fit driver's ``train_budget_meta.json`` sidecar next to the object, the
    per-cell provenance record stamped at publish time (its
    ``positions_at_fit`` key). This is the AUTHORITATIVE source: some artifact
    formats (the SteeringVector safetensors path) deliberately drop the fit
    leg on load, so the loaded object cannot answer this question itself.

    Missing sidecar:

    - a Class B cell resolves to ``"all"`` — ``fit_class_b.py`` publishes no
      ``train_budget_meta.json`` by design, and every Class B fit (antidote /
      alphasteer / caa_static_r) is a closed-form all-position object (or, for
      antidote, an intrinsic last-token deployment that ignores the serve
      mask). A future masked Class B fit must ship the sidecar.
    - anything else RAISES: the two-SI publish contract includes the sidecar
      (the driver's resume verification refuses a cell without it), so its
      absence means an incomplete artifact set — never a leg to guess.
    """
    import json

    if cell.object_path is None:
        raise ValueError(
            f"cell {cell.id!r} has no object_path — a baseline has no fit "
            "leg to resolve")
    sidecar = cell.object_path.parent / TRAIN_BUDGET_META
    if not sidecar.exists():
        if cell.class_ == "B":
            return "all"
        raise ValueError(
            f"cell {cell.id!r}: no {TRAIN_BUDGET_META} sidecar next to "
            f"{cell.object_path} — the fit-phase record is part of the "
            "publish contract, and without it the serve leg cannot be "
            "matched to the fit leg. Re-publish the cell through the fit "
            "driver rather than serving an artifact of unknown leg.")
    try:
        meta = json.loads(sidecar.read_text())
    except ValueError as e:
        raise ValueError(
            f"cell {cell.id!r}: unreadable {sidecar}: {e} — the fit-phase "
            "record is authoritative; fix it rather than serving around it.")
    at_fit = meta.get("positions_at_fit")
    if at_fit not in POSITIONS:
        raise ValueError(
            f"cell {cell.id!r}: {sidecar} records positions_at_fit="
            f"{at_fit!r}, not one of {POSITIONS} — refusing to serve an "
            "artifact whose fit leg cannot be established.")
    return str(at_fit)


def build_defense_cell(
    model,
    tok,
    cell,
    *,
    scale: float = 1.0,
    batch_size: int = 8,
    positions: str = "fit",
    generate_fn=None,
) -> LLMHFSteered:
    """Build the target for a two-SI :class:`~sisteer.eval.defense_cell.DefenseCell`,
    reusing ``build_defense``/``LLMHFSteered`` UNCHANGED. This is the
    cell-id-keyed serving surface; object CONSTRUCTION keys on ``cell.arm`` and
    the SI on ``cell.si_id``, everything else on ``cell.id`` (via the
    ``cell.object_path`` the registry resolved from the cell id):

    - ``none``            -> ``mode='none'`` (no object, no SI).
    - ``prompted__<si>``  -> ``mode='prompted'`` with ``system`` = the ``si_id``'s
      instruction text, so the two prompted baselines each send THEIR OWN SI —
      a single ``prompted`` identity cannot.
    - a compiled cell (Class A ``<si>__<arm>`` or Class B ``<arm>``) -> the
      EXISTING compiled path with ``mode=cell.arm`` (a real ``COMPILED`` arm, so
      ``LLMHFSteered`` validates unchanged) and the object loaded from
      ``cell.object_path`` — which is keyed on ``cell.id``, so the two SIs'
      objects for one arm are distinct artifacts, never one overwriting the
      other.

    ``positions`` (default ``"fit"``): the serve leg for a compiled cell.
    ``"fit"`` reads the artifact's persisted ``positions_at_fit``
    (:func:`cell_positions_at_fit`) and serves at THAT leg, so serve
    auto-matches fit — a decode-only (Class C) object keeps its mask, an
    all-token object serves all-position, with no per-arm knowledge here. An
    EXPLICIT leg is asserted against the record and a mismatch RAISES: this is
    the fit==serve guard the fit driver's provenance comment promises
    (train_two_si_objects.py, ``positions_at_fit``). Before it, this surface
    defaulted to ``"all"``, and ``_set_response_positions`` — whose explicit
    "all" ACTIVELY resets a response-fit object (the fit x serve 2x2's
    deliberate seam, pinned by tests/test_class_c_arms.py) — silently unmasked
    every decode-only object served through the two-SI attack path. A
    deliberate cross-phase serve remains reachable through ``build_defense``
    on the artifact path directly (the single-SI evals lane's 2x2); this
    cell-keyed surface serves each artifact as what it is."""
    if cell.arm is None:  # a baseline (none / prompted__<si>)
        # A baseline has no artifact and therefore no fit leg: "fit" resolves
        # to "all"; any OTHER explicit value is forwarded so it hits
        # build_defense's loud refusal (a baseline has no maskable
        # intervention) instead of being silently dropped here.
        positions = "all" if positions == "fit" else positions
        if cell.si_id is None:  # the undefended base
            d = build_defense(model, tok, mode="none", batch_size=batch_size,
                              positions=positions, generate_fn=generate_fn)
        else:
            d = build_defense(
                model, tok, mode="prompted",
                system=resolve_instruction(cell.si_id).text,
                batch_size=batch_size, positions=positions,
                generate_fn=generate_fn)
        d.positions_served = positions
        return d
    if cell.class_ == "stacked":
        # A stacked cell serves mode='stacked': the SI (cell.si_id, resolved to
        # its text) as the authoritative system message AND the compiled object
        # (cell.object_path, which object_path_for keyed on the UNDERLYING
        # <si>__<arm> so the SAME object bytes are reused — no new artifact).
        # The object still has a fit leg, so the SAME fit==serve guard and
        # provenance stamps as the compiled path below apply here.
        if cell.object_path is None:
            raise ValueError(
                f"stacked cell {cell.id!r} has no object_path (registry built "
                "without a defenses_dir?)")
        at_fit = cell_positions_at_fit(cell)
        if positions == "fit":
            positions = at_fit
        elif positions != at_fit:
            raise ValueError(
                f"cell {cell.id!r}: serve positions {positions!r} != "
                f"positions_at_fit {at_fit!r} (recorded in "
                f"{cell.object_path.parent / TRAIN_BUDGET_META}). Serving an "
                "object at a leg it was not fitted on mislabels the defense; "
                "omit positions (or pass 'fit') to serve at the fit leg, or "
                "run a deliberate cross-phase cell through build_defense on "
                "the artifact path directly.")
        d = build_defense(
            model, tok, mode="stacked", object_path=cell.object_path,
            system=resolve_instruction(cell.si_id).text, scale=scale,
            batch_size=batch_size, positions=positions,
            generate_fn=generate_fn)
        d.positions_served = positions
        d.positions_at_fit = at_fit
        return d
    # DEFAULT-DENY (2026-08-25): a stacked cell must have been handled above
    # (class_ == "stacked"). If one reaches here its stacked dispatch is
    # missing/reordered — REFUSE loudly rather than silently serving
    # object-only (the bug that shipped an object-only JB leg). Same
    # principle as the provenance gate's default-deny: an unrecognised cell
    # class is refused, not approximated. ``.stacked`` is the redundant
    # marker kept precisely so this guard holds even if ``class_`` dispatch
    # regresses.
    if getattr(cell, "stacked", False):
        raise ValueError(
            f"stacked cell {cell.id!r} reached the compiled path — its stacked "
            "branch is missing; refusing to serve object-only")
    # a compiled cell: the ARM drives the serving behaviour, the cell-id-keyed
    # path drives WHICH object is served.
    if cell.object_path is None:
        raise ValueError(
            f"compiled cell {cell.id!r} has no object_path (registry built "
            "without a defenses_dir?)")
    at_fit = cell_positions_at_fit(cell)
    if positions == "fit":
        positions = at_fit
    elif positions != at_fit:
        # The fit==serve guard: a mismatch RAISES, never warns — an
        # all-position serve of a response-fit object produces perfectly
        # well-formed output under a decode-only label (the invisible
        # failure test_class_c_arms pins), so nothing downstream would
        # catch it.
        raise ValueError(
            f"cell {cell.id!r}: serve positions {positions!r} != "
            f"positions_at_fit {at_fit!r} (recorded in "
            f"{cell.object_path.parent / TRAIN_BUDGET_META}). Serving an "
            "object at a leg it was not fitted on mislabels the defense; "
            "omit positions (or pass 'fit') to serve at the fit leg, or run "
            "a deliberate cross-phase cell through build_defense on the "
            "artifact path directly.")
    d = build_defense(model, tok, mode=cell.arm, object_path=cell.object_path,
                      scale=scale, batch_size=batch_size,
                      positions=positions, generate_fn=generate_fn)
    # The serve-leg record, stamped on the built target so downstream
    # provenance (two_si_run's cell_meta sidecars) reads what was ACTUALLY
    # served off the defense itself — the designed source — instead of
    # re-deriving it.
    d.positions_served = positions
    d.positions_at_fit = at_fit
    return d
