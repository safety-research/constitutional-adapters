"""HF engine — every v0 scored path.

Reuses the sisteer core end-to-end: ``models.load_model_and_tokenizer``
(revision-pinned load), ``methods.load_object`` (content-sniffing
artifact hydration — safetensors → VectorObject with the S2-exact
``skip_bos=True`` semantics), ``hooks.steering`` (through the objects'
own ``apply``). Control arms (``control:*``) are built by
``stats/controls.py`` (§8).

Contract highlights:

* ``ArmHandle.ctx()`` yields ``obj.apply(model, scale=coeff)`` FRESH per
  call: callers wrap each forward/generate call individually (F10); a
  long-lived context is structurally impossible. The handle counts its
  entries (``ctx_entries``) so the hygiene tests can pin the per-call
  discipline.
* Weight-mutating control arms build their perturbations ONCE at arm
  construction and cache them (plus original-weight copies) on the
  parameter's own device, so the fresh-ctx-per-forward discipline never
  regenerates or re-transfers full-model tensors; every context exit
  restores bit-exactly from the cached originals; the driver
  additionally isolates them one-arm-per-process (§11.1).
* Kind validation: ``kind: steering`` asserts the artifact hydrates to a
  safetensors SteeringVector (VectorObject); ``kind: sisteer_object``
  asserts a torch-blob object — one load path, two validations. The
  manifest ``layer`` is validated against the artifact's JSON sidecar
  (F9, via manifest.validate_layer_against_sidecar); a blob with no
  sidecar file is validated against the object's own layer attributes
  when it has them.
* The engine-matched pairing rule lives in the fingerprint (it
  refuses to pair rows whose ``engine`` fields differ); this engine's
  contribution is stamping ``name = "hf"`` so every row carries the
  engine correctly.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.manifest import AblationSpec
from sisteer.capeval.serving import (
    GENERATED_POSITIONS,
    RESPONSE_POSITIONS,
    ArmHandle,
    GenOut,
    PositionScore,
    arm_position_policy,
)
from sisteer.capeval.serving.resolved_spec import resolve_and_hash
from sisteer.capeval.tiers.tier0 import AtomScores
from sisteer.generate import encode_kwargs
from sisteer.methods import load_object
from sisteer.models import build_chat_prompt, load_model_and_tokenizer


class EngineError(RuntimeError):
    """An engine refusal (kind/layer/coeff validation, load misuse)."""


#: The canonical §2.4 gen_cfg vocabulary; anything else is a refusal.
#: Semantics: ``greedy: bool`` -> do_sample = not greedy (a conflicting
#: explicit do_sample is a refusal); ``chat: bool`` -> render each
#: prompt through the chat template before tokenizing; ``until:
#: list[str]`` -> stop strings applied POST-truncation to the decoded
#: text (token ids/logprobs stay raw — the canary needs them);
#: ``batch_size: int`` -> generation chunk size, ONE fresh arm ctx per
#: chunk.
_GEN_CFG_KEYS = frozenset({"greedy", "chat", "until", "batch_size",
                           "max_new_tokens", "do_sample", "temperature",
                           "top_p"})
#: positions chunked through the float64 log-softmax to bound memory
#: (128 positions x |V| float64 ≈ 128MB on a 128k vocab).
_POSITION_CHUNK = 128


class HFArmHandle(ArmHandle):
    """One (arm, coeff) binding over a loaded intervention object."""

    def __init__(self, *, name: str, kind: str, coeff: float,
                 intervention_hash: str, obj: Any, model: Any,
                 resolved_spec: Mapping[str, Any]):
        super().__init__(name=name, kind=kind, coeff=coeff,
                         intervention_hash=intervention_hash)
        self._obj = obj
        self._model = model
        self.resolved_spec = dict(resolved_spec)
        #: per-call entry counter — the hygiene guard reads this.
        self.ctx_entries = 0

    def ctx(self):
        """A FRESH context manager per call: ``obj.apply``
        constructs a new hooks.steering(...) context (or a fresh
        mutate/restore context for weight-mutating controls) on every
        invocation."""
        self.ctx_entries += 1
        return self._obj.apply(self._model, scale=self.coeff)

    def transform_prompt(self, s: str) -> str:
        """obj.inject — the sisteer downstream contract (identity for
        all non-token kinds, sentinel insertion for token kinds)."""
        return self._obj.inject(s)


class HFEngine:
    """The §2.4 Engine protocol on transformers (§6.1)."""

    name = "hf"

    def __init__(self, model_id: str, revision: Optional[str] = None,
                 dtype: Any = None, device: Optional[str] = None,
                 preloaded: Optional[tuple] = None):
        """``preloaded=(model, tok)`` skips the load (tests / a caller
        that already holds the pair); model_id/revision still stamp the
        engine's identity fields."""
        self.model_id = model_id
        self.revision = revision
        self._dtype = dtype
        self._device = device
        self._model = None
        self._tok = None
        if preloaded is not None:
            self._model, self._tok = preloaded

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        """One model load per process (models.load_model_and_tokenizer,
        revision-pinned). Idempotent for a preloaded engine."""
        if self._model is None:
            self._model, self._tok = load_model_and_tokenizer(
                self.model_id, device=self._device, dtype=self._dtype,
                revision=self.revision)

    @property
    def model(self):
        if self._model is None:
            raise EngineError("engine.load() has not run — no model")
        return self._model

    @property
    def tokenizer(self):
        if self._tok is None:
            raise EngineError("engine.load() has not run — no tokenizer")
        return self._tok

    # -- arms ---------------------------------------------------------------

    def arm(self, spec: AblationSpec, coeff: float) -> HFArmHandle:
        """Resolve one manifested arm at one coefficient (§6.1).

        steering/sisteer_object route through methods.load_object;
        control:* kinds through stats/controls.build_control. kind
        "none" is refused — the base model is ``arm=None`` everywhere in
        the protocol, and a handle pretending otherwise would let a
        no-op slip into steered bookkeeping."""
        if spec.kind == cm.NONE_KIND:
            raise EngineError(
                "arm(kind='none') is refused: pass arm=None for base-model "
                "scoring (§2.4) — a no-op handle would mislabel rows")

        if spec.kind in cm.STEERING_KINDS:
            return self._steering_arm(spec, coeff)
        if spec.kind.startswith(cm.CONTROL_KIND_PREFIX):
            return self._control_arm(spec, coeff)
        raise EngineError(f"arm {spec.name!r}: unknown kind {spec.kind!r}")

    def _steering_arm(self, spec: AblationSpec, coeff: float) -> HFArmHandle:
        if not spec.artifact:
            raise EngineError(f"arm {spec.name!r}: kind {spec.kind!r} "
                              "requires an artifact path")
        artifact = Path(spec.artifact)
        if not artifact.is_file():
            raise EngineError(f"arm {spec.name!r}: artifact does not exist: "
                              f"{artifact}")
        obj = load_object(artifact, self.model, self.tokenizer)

        # kind validation -> sidecar/F9 layer validation -> engine-phase
        # position policy (generated / response mask) -> manifest
        # overrides -> resolved spec -> F9 hash: the SHARED sequence
        # (serving/resolved_spec.py) — every engine and verifier runs the
        # same code over the same hydration, so the post-load
        # arm_intervention_hash cannot drift between engines. The
        # response-mask arming stays engine-held (_arm_response_mask
        # needs the served tokenizer + the JB mask seam) and rides in as
        # the callback.
        resolved, ih = resolve_and_hash(
            spec, obj, coeff, EngineError,
            arm_response_mask=self._arm_response_mask)
        return HFArmHandle(name=spec.name, kind=spec.kind, coeff=coeff,
                           intervention_hash=ih, obj=obj, model=self.model,
                           resolved_spec=resolved)

    def _arm_response_mask(self, name: str, obj: Any) -> int:
        """Arm the RESPONSE MASK on a loaded object and return the armed
        header length H — the capabilities-leg mirror of the JB defense
        build, through the SAME seam (``jailbreak._set_response_positions``,
        never a reimplementation), so the two legs can never derive
        different boundaries. The seam carries the hard guards this mode
        requires: H <= 0 is a loud error on ANY tokenizer (a collapsed
        boundary silently serves token 1 unsteered — the decisive token),
        an arm whose intervention cannot honour a mask refuses
        (FIT_MASK_REFUSAL / SERVE_MASK_REFUSAL: soft tokens, loreft,
        alphasteer-style derivation-locality failures), and a maskable
        object is armed through its real seam (set_fit_positions or the
        vector deployment attrs).

        The boundary is then verified LIKE WITH LIKE: the armed H is the
        width the fit-time mask home derives (``hooks.
        response_header_len`` — the loaded object's recorded fit
        boundary is checked against it inside the seam,
        ``jailbreak._check_recorded_header``), and this guard holds that
        H to the ID-form of the SAME convention,
        ``chat_markers.generation_header_ids`` — the tokens
        ``add_generation_prompt`` actually appends, cross-checked
        against located template structure inside the helper (the
        appended text must be a prefix of the scaffold a completed
        assistant turn renders; seeded content, a divergent generation
        prompt, and an unpinned reasoning template refuse there).
        Deliberately NOT compared
        against ``derive_assistant_markers``'s terminator-anchored
        header: that locator absorbs prompt-side scaffold (gemma-4's
        user turn closes ``<turn|>\n``, so it measures 4 where the fit
        support is 3), and refusing on ITS disagreement with the width
        rejected the agreement of two different definitions — the
        scale-12 gemma fitted-cell blocker. A mask armed at that wider
        header would steer one prompt-side token OUTSIDE the fit
        support: a train/serve mismatch, not a margin. (Prompts on this leg are chat renders that
        END with the header, where width == located boundary once this
        agreement holds; the one instrument that appends content after
        the header — mmlu's "Answer:" prefill — locates the header
        per-request in tiers/parity.py.)"""
        from sisteer.chat_markers import ChatMarkerError, generation_header_ids
        from sisteer.eval.jailbreak import _set_response_positions

        try:
            _set_response_positions(obj, RESPONSE_POSITIONS, self.tokenizer)
        except ValueError as e:
            raise EngineError(f"arm {name!r}: {e}") from e
        h = getattr(obj, "response_header_len", None)
        if not h:
            h = getattr(obj, "fit_response_header_len", None)
        if not h or int(h) <= 0:
            raise EngineError(
                f"arm {name!r}: positions='response' armed but the object "
                f"reports header length {h!r} — the mask seam did not take; "
                "refusing rather than serving a collapsed boundary")
        try:
            gen_ids = generation_header_ids(self.tokenizer)
        except ChatMarkerError as e:
            raise EngineError(
                f"arm {name!r}: positions='response' requires the "
                "generation-prompt assistant header to be derivable from "
                "the served template (the id-form of the width the mask "
                f"arms, backed by located structure): {e}") from e
        if len(gen_ids) != int(h):
            raise EngineError(
                f"arm {name!r}: armed header length H={int(h)} disagrees "
                f"with the generation-prompt header ids {list(gen_ids)} "
                f"(len {len(gen_ids)}) — the COUNT derivation "
                "(hooks.response_header_len, the fit-time mask home) and "
                "the ID derivation (chat_markers.generation_header_ids, "
                "the serve-time boundary) must agree on the same "
                "template; a disagreement means one of them mis-read the "
                "template structure, or the object's mask seam ignored "
                "the armed value — refusing to serve the mask")
        return int(h)

    def _control_arm(self, spec: AblationSpec, coeff: float) -> HFArmHandle:
        from sisteer.capeval.stats.controls import build_control  # cycle-free

        if spec.kind == "control:zero_coeff" and coeff != 0.0:
            raise EngineError(
                f"arm {spec.name!r}: control:zero_coeff must run at coeff "
                f"0.0 (it tests the apply/ctx machinery is a no-op at zero, "
                f"§8), got {coeff}")
        if spec.positions == RESPONSE_POSITIONS:
            raise EngineError(
                f"arm {spec.name!r}: control arms do not serve the response "
                "mask — the mask is armed on a loaded intervention object "
                "through its own seams (set_fit_positions / the vector "
                "deployment attrs), which control perturbations do not "
                "carry; refusing rather than serving an unmasked control "
                "under a response label")
        obj = build_control(spec, self.model, self.tokenizer)
        resolved = {"kind": spec.kind, "seed": spec.seed,
                    "params": dict(spec.params)}
        if spec.positions == GENERATED_POSITIONS:
            # same engine-phase policy as steering arms — a control under
            # generated-only serving is a DIFFERENT intervention identity,
            # so the key enters the hash; absent for the default, keeping
            # every existing control hash byte-identical.
            resolved["steer_positions"] = GENERATED_POSITIONS
        ih = cm.intervention_hash(resolved)
        return HFArmHandle(name=spec.name, kind=spec.kind, coeff=coeff,
                           intervention_hash=ih, obj=obj, model=self.model,
                           resolved_spec={**resolved, "coeff": coeff})

    # -- Tier 0: teacher-forced scoring --------------------------------------

    @torch.no_grad()
    def score_corpus(
        self,
        corpus: Any,
        arm: Optional[HFArmHandle] = None,
        atom_ids: Optional[Any] = None,
    ) -> Iterator[PositionScore]:
        """Per-position teacher-forced scoring over a WindowedCorpus
        (tier0's §2.4 corpus type), in the corpus's item order.

        ``atom_ids`` selects the §4.3 mode:

        * ``None`` — NLL only (atom_probs is None);
        * an ``int`` k — BASE pass: per-position top-k atoms by
          probability (ties broken by ascending token id via a stable
          descending sort) + float64 masked-logsumexp tail;
        * a ``[N, k]`` int tensor (the cache's ids, aligned to the
          corpus's scored positions in order) — ABLATION pass: exact
          logq at those ids + logq_tail by the same masked logsumexp.

        The steering ctx is entered FRESH per item forward (F10);
        ``arm=None`` never enters a ctx — that is the hooks-disabled
        base pass §4.5 relies on."""
        policy = arm_position_policy(arm)
        if policy != "all":
            raise EngineError(
                f"score_corpus: arm {arm.name!r} resolves position policy "
                f"{policy!r} — teacher-forced scoring forwards the complete "
                "scored sequence, so any policy but 'all' steers the wrong "
                "positions (e.g. the appended target token instead of its "
                "predictor); the selector-aware target-excluded-prefix "
                "scorer is a v1 item (r1 finding 1) — refusing")
        fixed_ids = None
        top_k = None
        if isinstance(atom_ids, int):
            top_k = atom_ids
        elif atom_ids is not None:
            fixed_ids = torch.as_tensor(atom_ids)
            if fixed_ids.dim() != 2:
                raise EngineError("score_corpus: atom_ids tensor must be "
                                  "[N, k] aligned to the scored positions")

        device = next(self.model.parameters()).device
        offset = 0
        for item in corpus.items:
            ids = torch.tensor([list(item.input_ids)], dtype=torch.long,
                               device=device)
            ctx = arm.ctx() if arm is not None else contextlib.nullcontext()
            with ctx:
                logits = self.model(input_ids=ids).logits[0]
            positions = list(item.score_positions)
            n = len(positions)
            item_fixed = None
            if fixed_ids is not None:
                if offset + n > fixed_ids.shape[0]:
                    raise EngineError(
                        "score_corpus: atom_ids has fewer rows than the "
                        "corpus's scored positions — cache/corpus mismatch")
                item_fixed = fixed_ids[offset:offset + n]
            offset += n

            for c0 in range(0, n, _POSITION_CHUNK):
                chunk = positions[c0:c0 + _POSITION_CHUNK]
                # logits at i-1 predict the token at i; float64 end-to-end
                # from here (F4/r2-F4)
                lp = torch.log_softmax(
                    logits[[p - 1 for p in chunk]].double(), dim=-1)
                if torch.isnan(lp).any():
                    raise EngineError(
                        f"score_corpus: NaN log-probabilities at item "
                        f"{item.item_id} — the forward produced non-finite "
                        "logits (hard numerical failure, r1 finding 10); "
                        "refusing the cell rather than letting NaN evade "
                        "the KL red-flag path")
                targets = torch.tensor(
                    [item.input_ids[p] for p in chunk], device=lp.device)
                nll = -lp.gather(1, targets[:, None])[:, 0]

                if top_k is not None:
                    vals, order = torch.sort(lp, dim=-1, descending=True,
                                             stable=True)
                    a_ids = order[:, :top_k]
                    a_logp = vals[:, :top_k]
                elif item_fixed is not None:
                    a_ids = item_fixed[c0:c0 + len(chunk)].to(lp.device).long()
                    a_logp = lp.gather(1, a_ids)
                else:
                    for j in range(len(chunk)):
                        yield PositionScore(float(nll[j]), None)
                    continue

                masked = lp.scatter(1, a_ids, float("-inf"))
                tail = torch.logsumexp(masked, dim=-1)
                for j in range(len(chunk)):
                    yield PositionScore(
                        float(nll[j]),
                        AtomScores(ids=a_ids[j].to(torch.int32).cpu(),
                                   logp=a_logp[j].double().cpu(),
                                   logp_tail=float(tail[j])))
        if fixed_ids is not None and offset != fixed_ids.shape[0]:
            raise EngineError(
                f"score_corpus: atom_ids has {fixed_ids.shape[0]} rows but "
                f"the corpus scored {offset} positions — cache/corpus "
                "mismatch")

    # -- generation (canary §4.4, target plugins §9) --------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        arm: Optional[HFArmHandle] = None,
        gen_cfg: Optional[Mapping[str, Any]] = None,
    ) -> List[GenOut]:
        """Chunked generation with scores (the §2.4 gen_cfg vocabulary —
        see :data:`_GEN_CFG_KEYS`). ``batch_size`` chunks the prompt list
        and ONE fresh ctx wraps each chunk's generate call (F10 — never
        per decode step, never one giant batch); no ``batch_size`` means
        a single chunk. Without ``chat``, prompts arrive already rendered
        (and already arm.transform_prompt-ed by the caller);
        encode_kwargs prevents the double-BOS on chat-templated strings."""
        cfg = dict(gen_cfg or {})
        unknown = sorted(set(cfg) - _GEN_CFG_KEYS)
        if unknown:
            raise EngineError(f"generate: unknown gen_cfg key(s) {unknown} "
                              f"(allowed: {sorted(_GEN_CFG_KEYS)})")
        max_new = int(cfg.get("max_new_tokens", 256))
        if "greedy" in cfg:
            greedy = bool(cfg["greedy"])
            if "do_sample" in cfg and bool(cfg["do_sample"]) != (not greedy):
                raise EngineError(
                    f"generate: greedy={greedy} conflicts with "
                    f"do_sample={cfg['do_sample']} — pass one or the other")
            do_sample = not greedy
        else:
            do_sample = bool(cfg.get("do_sample", False))
        until = cfg.get("until")
        if until is not None and (
                not isinstance(until, (list, tuple))
                or not all(isinstance(s, str) and s for s in until)):
            raise EngineError("generate: until must be a list of non-empty "
                              f"stop strings, got {until!r}")
        gen_kwargs: Dict[str, Any] = {}
        if do_sample:
            for k in ("temperature", "top_p"):
                if k in cfg:
                    gen_kwargs[k] = cfg[k]

        rendered = list(prompts)
        if bool(cfg.get("chat", False)):
            rendered = [build_chat_prompt(self.tokenizer, p) for p in rendered]

        batch_size = cfg.get("batch_size")
        if batch_size is None:
            batch_size = max(len(rendered), 1)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise EngineError(f"generate: batch_size must be >= 1, got "
                              f"{batch_size}")

        results: List[GenOut] = []
        for c0 in range(0, len(rendered), batch_size):
            results.extend(self._generate_chunk(
                rendered[c0:c0 + batch_size], arm, max_new=max_new,
                do_sample=do_sample, gen_kwargs=gen_kwargs,
                until=list(until) if until else None))
        return results

    def _generate_chunk(
        self,
        prompts: Sequence[str],
        arm: Optional[HFArmHandle],
        *,
        max_new: int,
        do_sample: bool,
        gen_kwargs: Mapping[str, Any],
        until: Optional[List[str]],
    ) -> List[GenOut]:
        """One batched generate call under ONE fresh arm ctx (F10)."""
        tok, model = self.tokenizer, self.model
        device = next(model.parameters()).device
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_ids = _eos_ids(tok, model)

        encoded = [tok(p, **encode_kwargs(tok, p))["input_ids"] for p in prompts]
        in_len = max(len(e) for e in encoded)
        input_ids = torch.full((len(encoded), in_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(encoded), in_len), dtype=torch.long)
        for i, e in enumerate(encoded):  # left padding for generation
            input_ids[i, in_len - len(e):] = torch.tensor(e, dtype=torch.long)
            attn[i, in_len - len(e):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        if arm is not None and _steer_positions_of(arm) == GENERATED_POSITIONS:
            # Generated-only serving (2026-08-20), two-phase: the PREFILL
            # forward runs ENTIRELY OUTSIDE the arm ctx — the arm
            # machinery is NOT INVOKED at any prompt position (never
            # merely invoked-with-zero-coefficient), the base model
            # encodes the prompt, so steered decode attends over
            # unsteered base-computed prompt KV, and the FIRST generated
            # token is picked from the unsteered prompt logits. Every
            # DECODE forward then runs INSIDE one fresh ctx (still ONE
            # ctx per chunk — F10), so every generated position is served
            # under the intervention. Gating at ctx enter/exit is what
            # keeps this method-generic: hook arms get hooks on decode
            # forwards only; weight-edit arms encode the prompt with base
            # weights and generate with adapted weights. BOS-skip remains
            # an INDEPENDENT within-hooked-region position mask, not
            # collapsed into this phase flag. The semantic is shared with
            # the vLLM proxy lane (branch integrate-main) by cross-path
            # agreement, 2026-08-20. The stitch (prefill+1 token, then
            # continuation from the returned KV cache) is bit-identical
            # to the single call at zero effect — pinned by
            # tests/test_steer_generated_only.py.
            first = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=1, do_sample=do_sample,
                pad_token_id=pad_id, return_dict_in_generate=True,
                output_scores=True, **gen_kwargs)
            if first.past_key_values is None:
                raise EngineError(
                    "generate: the prefill pass returned no KV cache — "
                    "generated-only serving cannot hand the unsteered "
                    "prompt cache to the decode phase; refusing rather "
                    "than silently re-encoding the prompt under the arm")
            sequences, scores = first.sequences, list(first.scores)
            if max_new > 1:
                attn_cont = torch.cat(
                    [attn, torch.ones_like(attn[:, :1])], dim=1)
                with arm.ctx():
                    cont = model.generate(
                        input_ids=sequences, attention_mask=attn_cont,
                        past_key_values=first.past_key_values,
                        max_new_tokens=max_new - 1, do_sample=do_sample,
                        pad_token_id=pad_id, return_dict_in_generate=True,
                        output_scores=True, **gen_kwargs)
                sequences = cont.sequences
                scores = scores + list(cont.scores)
        else:
            ctx = arm.ctx() if arm is not None else contextlib.nullcontext()
            with ctx:
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=max_new, do_sample=do_sample,
                    pad_token_id=pad_id, return_dict_in_generate=True,
                    output_scores=True, **gen_kwargs)
            sequences, scores = out.sequences, list(out.scores)

        results: List[GenOut] = []
        gen_tokens = sequences[:, in_len:]
        for b in range(len(prompts)):
            toks: List[int] = []
            logprobs: List[float] = []
            ended = False
            for step in range(gen_tokens.shape[1]):
                t = int(gen_tokens[b, step])
                lp = torch.log_softmax(scores[step][b].double(), dim=-1)
                toks.append(t)
                logprobs.append(float(lp[t]))
                if t in eos_ids:
                    ended = True
                    break
            text_tokens = toks[:-1] if ended else toks
            text = tok.decode(text_tokens, skip_special_tokens=True)
            if until:  # stop strings, post-truncation (text only — the
                # raw ids/logprobs stay full for the canary)
                cut = len(text)
                for stop in until:
                    i = text.find(stop)
                    if i != -1:
                        cut = min(cut, i)
                text = text[:cut]
            results.append(GenOut(text=text, gen_token_logprobs=logprobs,
                                  ended_with_eos=ended, gen_token_ids=toks))
        return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# The F9 resolved-spec builders (_object_layers / _resolved_spec) moved
# to serving/resolved_spec.py (object_layers / resolved_spec_from_object),
# where every engine and the verifiers import them — one definition, so
# the post-load arm_intervention_hash cannot drift between engines again.


def _steer_positions_of(arm: Optional[ArmHandle]) -> str:
    """The engine-phase steering policy an arm resolved to ("all" for the
    base model and for handles carrying no resolved spec)."""
    if arm is None:
        return "all"
    spec = getattr(arm, "resolved_spec", None) or {}
    return str(spec.get("steer_positions", "all"))


def _eos_ids(tok, model) -> set:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(e) for e in eos)
    return ids
