"""vLLM-backed attacker adapter for the jailbreak battery (opt-in).

Only the UNSTEERED adversary moves here. The steered TARGET stays on HF
transformers because it is a residual-stream forward hook and vLLM cannot host
it. The attacker (``huihui-ai/Huihui-Qwen3.6-27B-abliterated``) needs no hooks
-- it is never steered -- so a vanilla vLLM process serving an OpenAI-compatible
endpoint can generate its turns far faster than HF ``model.generate``. The
in-loop 1-10 local scorer shares the attacker weights, so it hits the SAME
endpoint.

Wiring: ``scripts/jailbreak_run.py`` builds the attacker/scorer seams from this
module when ``--attacker-engine vllm`` is passed (default is ``hf``, which
changes nothing). The launcher that starts the endpoint on its own GPU is
``scripts/vllm_attacker_serve.sh``.

IMPORTANT -- results are NOT cross-engine comparable. vLLM and HF do not produce
identical attacker text (different kernels, different sampling implementations;
a peer's equivalence gate failed over a single trailing newline). Different
attacker text means different attack prompts, so ASR from a ``--attacker-engine
vllm`` run is NOT comparable to an HF-engine battery. The engine is therefore
recorded in the per-run manifest (``attacker_engine.json`` beside the
transcripts) so every artifact is self-identifying. Do NOT try to make the two
engines bit-identical; that is not achievable and pretending otherwise is worse
than documenting it.
"""

from __future__ import annotations

import re
import time

import requests

from .attacker_serving import ATTACKER_ENABLE_THINKING

# The attacker is a reasoning model; the shared serving identity pins its
# chat-template thinking mode off. A flip in attacker_serving is a scientific
# change and automatically reaches both requests and manifests.


def _strip_reasoning(text: str | None) -> str | None:
    """Return only the final answer from a reasoning model's chat output.

    The abliterated-Qwen attacker is a reasoning model: over the vLLM endpoint
    its chat output can include a ``<think>...</think>`` chain-of-thought (the HF
    path renders the template locally and does not emit one). The PyRIT strategy
    parsers (ActorAttack actor-discovery, Crescendo strategy) need ONLY the final
    answer -- the raw reasoning breaks them (ActorDiscoveryError /
    _StrategyRuntimeError, 100% of goals). Drop everything up to and including
    the last ``</think>``. Belt-and-suspenders with ``enable_thinking=False`` in
    the request payload, which stops the reasoning being generated at all (so it
    also cannot eat the max_tokens budget and truncate the answer)."""
    if text is None:
        return text
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # also drop any stray/unclosed opening tag left behind
    text = re.sub(r"<think>", "", text)
    return text.strip()


class VLLMEndpointUnreachable(RuntimeError):
    """Raised when the vLLM attacker endpoint does not answer a health check.

    We fail loudly rather than silently falling back to HF: a silent fallback
    would make a "vLLM run" secretly an HF run, which the manifest would then
    mislabel -- and since vLLM is now the DEFAULT engine, that mislabelling
    would be the common case rather than an edge case.

    Because it is the default, this is also the error a caller hits when they
    simply have not started a server, so the message must say exactly what to
    do rather than just what went wrong."""

    @classmethod
    def for_endpoint(cls, endpoint: str, detail: str = "") -> "VLLMEndpointUnreachable":
        return cls(
            f"vLLM attacker endpoint {endpoint!r} did not answer a health check"
            f"{': ' + detail if detail else ''}.\n"
            "vLLM is the DEFAULT attacker engine. Either:\n"
            "  1. start a server first:  scripts/vllm_attacker_serve.sh\n"
            "     (needs its own GPU; the pinned env is vllm 0.25.1 / "
            "torch 2.11.0+cu129, which is NOT the main lane's cu128 venv)\n"
            "  2. or run on transformers: --attacker-engine hf\n"
            "Not falling back automatically, because a vLLM run and an HF run "
            "produce DIFFERENT attack prompts and must never be conflated."
        )


class VLLMEndpointUnauthorized(VLLMEndpointUnreachable):
    """The endpoint answered 401/403: it is bearer-guarded (a shared attacker
    pool) and the request carried no token or the wrong one. A subclass of the
    unreachable error so every existing ``except`` still fails loudly; its own
    type so a caller can name the remedy (the pool's key file / env var)."""

    @classmethod
    def for_endpoint(cls, endpoint: str, status: int, *,
                     api_key_set: bool) -> "VLLMEndpointUnauthorized":
        return cls(
            f"vLLM attacker endpoint {endpoint!r} rejected the request with "
            f"HTTP {status} ({'the bearer token sent was refused' if api_key_set else 'no bearer token was sent'}). "
            "A shared attacker pool requires its per-launch token: pass the pool "
            "(--attacker-pool <POOL_DIR>) so the driver reads it from the pool's "
            "0600 key file, or export the variable endpoint.json names "
            "(api_key_env_name); a seedrep driver container carries the lane's "
            f"pool bearer as ${_pool_bearer_env_name()} (D99). NOT falling back "
            "to HF or to a local attacker."
        )


def _pool_bearer_env_name() -> str:
    from .attacker_pool import POOL_BEARER_ENV   # stdlib-only contract module; imported lazily to keep this module's import light

    return POOL_BEARER_ENV


def attacker_unauthorized_in(exc: BaseException) -> "VLLMEndpointUnauthorized | None":
    """The :class:`VLLMEndpointUnauthorized` behind ``exc``, walking ``__cause__`` / ``__context__``
    (PyRIT's normalizer re-raises an attacker failure as a bare ``Exception(...) from ex``; the
    ActorAttack loop wraps nothing but retries), else None. The FIRST such error aborts a run: a
    pool that rejects the bearer rejects every call, so a per-goal record-and-continue would burn
    the whole shard on 401s and exit 0 with zero target forwards."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, VLLMEndpointUnauthorized):
            return e
        e = e.__cause__ or e.__context__
    return None


def root_cause_line(exc: BaseException) -> str:
    """``TypeName: message`` of the DEEPEST exception behind ``exc`` -- what a driver's child log
    names when a goal errors, so a swallowed cause (PyRIT's ``Error sending prompt with conversation
    ID: ...`` wrapper) still shows the ``HTTP 401`` and the endpoint it came from. The client's own
    messages carry the endpoint and the status, never the bearer."""
    seen: set[int] = set()
    e: BaseException = exc
    while True:
        seen.add(id(e))
        nxt = e.__cause__ or e.__context__
        if nxt is None or id(nxt) in seen:
            break
        e = nxt
    msg = str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0] if msg else ''}".rstrip(": ")


class AttackerPromptOverWindow(VLLMEndpointUnreachable):
    """A context-length 400 the once-only clamp could not absorb: the PROMPT
    (the attacker's growing transcript) leaves fewer than
    ``served_window.MIN_CLAMPED_COMPLETION`` tokens of the server's window.
    The twin of the served target's PromptOverWindow, for the attacker side:
    a p99-sized attacker window (attacker_pool.DEFAULT_MAX_MODEL_LEN) has a
    tail by construction, and this is how that tail surfaces — as a typed goal
    error, never as a silent truncation or a retry storm."""

    def __init__(self, endpoint: str, plan) -> None:
        self.plan = plan
        super().__init__(
            f"vLLM attacker endpoint {endpoint!r} rejected the request: {plan.reason} "
            f"(window {plan.effective_window}, prompt {plan.prompt_tokens}); the "
            "conversation no longer fits the attacker's window. NOT falling back to HF.")


class VLLMAttackerClient:
    """Thin OpenAI-compatible chat client for the vLLM attacker endpoint.

    ``endpoint`` is the server root (e.g. ``http://127.0.0.1:8000``); requests
    go to ``<endpoint>/v1/chat/completions`` and health to ``<endpoint>/health``
    (both are what ``vllm serve`` exposes). ``model`` is the served model id
    (the attacker repo id or local snapshot path passed to ``vllm serve``).

    The ``session`` seam lets tests inject a fake transport (any object with
    ``.get`` / ``.post`` returning a requests-like response) so the adapter is
    exercised with no server running.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 600.0,
        health_timeout: float = 5.0,
        max_retries: int = 2,
        backoff: float = 1.0,
        session=None,
        api_key: str | None = None,
        failover: "list[str] | tuple[str, ...] | None" = None,
    ) -> None:
        if not endpoint:
            raise ValueError("VLLMAttackerClient needs a non-empty endpoint URL")
        self.endpoint = endpoint.rstrip("/")
        # 31d: the OTHER live members of a shared pool, in order. When a request exhausts its transport retries against the current
        # member (connection refused, timeout, 5xx — a member that retired or died), the client rotates to the next member that answers
        # /health and retries there; a 401/403 or a prompt-over-window never rotates (they are not the member's death). Every rotation
        # is recorded (`failover_log`) and the member each request hit rides `last_meta["endpoint"]` into the per-request ledger row.
        self.failover: list[str] = [e.rstrip("/") for e in (failover or []) if e and e.rstrip("/") != self.endpoint]
        self.failover_log: list[dict] = []
        self.failover_tried: list[str] = []          # every candidate asked for /health, answered or not (named in the final refusal)
        self.n_failovers = 0
        self.model = model
        self.timeout = timeout
        self.health_timeout = health_timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._session = session if session is not None else requests.Session()
        # A bearer-guarded endpoint (the shared attacker pool serves behind
        # vLLM's stock VLLM_API_KEY guard on a shared-cluster port) takes the
        # token on the session headers -- never on argv, never logged. A
        # loopback per-job server passes None and sends no header, as before.
        # the once-only max_tokens clamp (served_window policy, attacker side):
        # a context-length 400 whose prompt still leaves room is retried ONCE
        # with max_tokens = window - prompt; a prompt that does not fit raises
        # AttackerPromptOverWindow. Counted here; the last call's clamp facts
        # ride ``last_meta`` into the RecordingAttackerClient ledger row.
        self.n_clamped = 0
        self.n_prompt_over_window = 0
        self.last_meta: dict = {}
        self.api_key_set = bool(api_key)
        if api_key:
            headers = getattr(self._session, "headers", None)
            if headers is None:
                self._session.headers = {}
                headers = self._session.headers
            headers.update({"Authorization": f"Bearer {api_key}"})

    # -- health ------------------------------------------------------------- #
    def check_health(self) -> None:
        """GET ``/health`` and raise :class:`VLLMEndpointUnreachable` with a
        clear, actionable message if the endpoint does not answer 2xx. Called
        ONCE when the vllm seams are built, so a misconfigured endpoint fails at
        startup instead of mid-sweep."""
        url = f"{self.endpoint}/health"
        try:
            resp = self._session.get(url, timeout=self.health_timeout)
        except Exception as exc:  # noqa: BLE001 - any transport failure is "unreachable"
            if self._rotate(f"health check of {self.endpoint} raised {type(exc).__name__}"):
                return          # 31d: a surviving member answered its health check and is now the endpoint
            raise VLLMEndpointUnreachable(
                f"vLLM attacker endpoint {self.endpoint!r} is unreachable "
                f"(GET {url} raised {type(exc).__name__}: {exc}). "
                "Start it with scripts/vllm_attacker_serve.sh and confirm the "
                "--attacker-endpoint host/port. NOT falling back to HF: a silent "
                "fallback would mislabel an HF run as a vLLM run."
            ) from exc
        status = getattr(resp, "status_code", None)
        if status is None or not (200 <= status < 300):
            raise VLLMEndpointUnreachable(
                f"vLLM attacker endpoint {self.endpoint!r} answered health check "
                f"with status {status} (expected 2xx). The server may still be "
                "loading the 27B weights, or it crashed at init (check the serve "
                "log for a CUDA/driver error). NOT falling back to HF."
            )

    # -- chat --------------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        """POST one chat completion and return the assistant text. vLLM applies
        the served model's OWN chat template, so ``messages`` is passed through
        as-is (the HF path renders the template locally; this is the intended,
        non-bit-identical, equivalent). ``max_new_tokens`` maps to OpenAI
        ``max_tokens``. Transient transport / 5xx failures are retried
        ``max_retries`` times -- but an unreachable endpoint still raises rather
        than degrading to HF."""
        url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            # abliterated-Qwen is a reasoning model; without this it emits a
            # <think> chain that (a) breaks the strategy parsers and (b) eats the
            # max_tokens budget so the real answer truncates. Qwen chat templates
            # honor this kwarg; _strip_reasoning() is the safety net if not.
            "chat_template_kwargs": {"enable_thinking": ATTACKER_ENABLE_THINKING},
        }
        self.last_meta = {}
        attempt = 0          # transient failures only; a clamp retry is not an attempt
        clamped = False
        while True:
            url = f"{self.endpoint}/v1/chat/completions"      # the CURRENT member's (a rotation moves it, 31d)
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                status = getattr(resp, "status_code", 200)
                if status >= 500:
                    raise VLLMEndpointUnreachable(
                        f"vLLM attacker endpoint returned {status} for "
                        f"POST {url}"
                    )
                if status in (401, 403):
                    # the endpoint is bearer-guarded and rejected ours (or we
                    # sent none): not transient, never retried, never HF.
                    raise VLLMEndpointUnauthorized.for_endpoint(
                        self.endpoint, status, api_key_set=self.api_key_set)
                if status == 400 and not clamped:
                    plan = self._overflow_plan(resp, payload)
                    if plan is not None:
                        from ...capeval.serving import served_window as sw

                        if plan.action == sw.RETRY:
                            payload["max_tokens"] = int(plan.max_tokens)
                            clamped = True
                            self.n_clamped += 1
                            self.last_meta = {"clamped_max_tokens": int(plan.max_tokens),
                                              "prompt_tokens": plan.prompt_tokens,
                                              "window": plan.effective_window,
                                              "prompt_tokens_source": plan.prompt_tokens_source}
                            continue
                        self.n_prompt_over_window += 1
                        self.last_meta = {"prompt_over_window": True,
                                          "prompt_tokens": plan.prompt_tokens,
                                          "window": plan.effective_window,
                                          "reason": plan.reason}
                        raise AttackerPromptOverWindow(self.endpoint, plan)
                resp.raise_for_status()
                data = resp.json()
                self.last_meta["endpoint"] = self.endpoint                 # the member this request hit (31d)
                if self.failover_log:
                    self.last_meta["failover"] = dict(self.failover_log[-1])
                return _strip_reasoning(data["choices"][0]["message"]["content"])
            except (VLLMEndpointUnauthorized, AttackerPromptOverWindow):
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient, then fail loud
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self.backoff * attempt)
                    continue
                if self._rotate(f"{type(exc).__name__} after {attempt + 1} attempt(s) at {url}"):
                    attempt = 0; clamped = False
                    continue      # 31d: re-bound to a surviving member; the request is retried there from scratch
                raise VLLMEndpointUnreachable(
                    f"vLLM attacker request to {url} failed after "
                    f"{self.max_retries + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}. "
                    + (f"No surviving pool member answered either (tried {self.failover_tried}). " if self.failover_tried else "")
                    + "NOT falling back to HF."
                ) from exc

    def _rotate(self, reason: str) -> bool:
        """Move to the next failover member that answers /health; False when none is left. Records the rotation (from, to, reason, utc)
        in ``failover_log`` — the driver's per-request ledger row then names the member each call hit (31d)."""
        while self.failover:
            cand = self.failover.pop(0)
            self.failover_tried.append(cand)
            try:
                resp = self._session.get(f"{cand}/health", timeout=self.health_timeout)
                status = getattr(resp, "status_code", None)
            except Exception:  # noqa: BLE001 — a dead candidate is skipped, never a rotation target
                continue
            if status is None or not (200 <= status < 300):
                continue
            prev = self.endpoint
            self.endpoint = cand
            self.n_failovers += 1
            self.failover_log.append({"from": prev, "to": cand, "reason": reason, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"[vllm-attacker] FAILOVER {prev} -> {cand}: {reason}", flush=True)
            return True
        return False

    def _overflow_plan(self, resp, payload: dict):
        """The served_window clamp plan for a 400, or None when the 400 is not
        a context-length rejection (then the generic path raises). A floor-only
        rejection ("prompt contains at least N input tokens") is resolved
        through vLLM's ``/tokenize`` on the SAME messages, exactly as the
        served target does — the floor is never used as the length."""
        from ...capeval.serving import served_window as sw

        try:
            body = resp.text if isinstance(getattr(resp, "text", None), str) else str(resp.json())
        except Exception:  # noqa: BLE001
            body = ""
        ov = sw.parse_context_overflow(f"vLLM attacker returned 400: {body}", http_status=400)
        if ov is None:
            return None
        plan = sw.clamp_plan(ov, requested_max_tokens=int(payload["max_tokens"]))
        if plan.prompt_tokens is None and ov.server_window is not None:
            # a floor-only rejection (RESOLVE) or one with no parseable prompt count:
            # the exact length comes from /tokenize, never from the floor or a guess
            try:
                tr = self._session.post(f"{self.endpoint}/tokenize",
                                        json=sw.tokenize_payload(payload), timeout=self.timeout)
                exact = sw.exact_prompt_tokens(tr.json())
                plan = sw.clamp_plan(ov, requested_max_tokens=int(payload["max_tokens"]),
                                     prompt_tokens=exact, prompt_tokens_source=sw.SOURCE_TOKENIZE)
            except Exception as e:  # noqa: BLE001 — unresolvable floor: fail, typed
                plan = plan.unresolvable(f"{type(e).__name__}: {e}")
        return plan


# --------------------------------------------------------------------------- #
# Seam builders -- mirror the HF _make_* helpers in scripts/jailbreak_run.py,   #
# same callable signatures so PyRIT and the hand-rolled multiturn loop are       #
# unchanged.                                                                     #
# --------------------------------------------------------------------------- #


def make_attacker_fn_vllm(
    client: VLLMAttackerClient,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    """``attacker_fn([{role, content}, ...]) -> str`` for the hand-rolled
    ActorAttack loop -- the vLLM twin of ``_make_attacker_fn``. Threads the
    attack's pinned sampling params (from ``_attacker_sampling``) into every
    request."""

    def attacker_fn(chat: list[dict]) -> str:
        return client.chat(
            chat, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    return attacker_fn


def make_scorer_fn_vllm(client: VLLMAttackerClient, *, max_new_tokens: int = 8):
    """``scorer_fn(prompt: str) -> str`` for LocalAttackScorers -- the vLLM twin
    of ``_make_scorer_fn``. GREEDY (``temperature=0``) and short, matching the HF
    scorer's ``do_sample=False`` / few-token decode; wraps the raw rubric prompt
    as a single user turn so vLLM's chat template renders it the same way the HF
    path's ``_render_chat`` does."""

    def scorer_fn(prompt: str) -> str:
        return client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0, top_p=1.0, max_new_tokens=max_new_tokens,
        )

    return scorer_fn


def make_attacker_factory_vllm(
    client: VLLMAttackerClient,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    """``attacker_factory(gpu_lock) -> AttackerChatTarget`` for the PyRIT
    attacks -- the vLLM twin of ``_make_attacker_factory``. Uses
    ``AttackerChatTarget``'s ``generate_fn`` seam so the PyRIT target is
    identical apart from where the tokens come from.

    The shared ``gpu_lock`` is still passed through: the attacker runs on a
    SEPARATE process/GPU so it needs no local GPU, but keeping the lock
    preserves the exact one-forward-at-a-time serialization the HF path had (the
    PyRIT conversation is sequential anyway). It could be dropped to let a future
    concurrent runner overlap attacker HTTP with target generation."""
    from .pyrit_adapters import AttackerChatTarget

    def generate_fn(chat: list[dict]) -> str:
        return client.chat(
            chat, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    def factory(gpu_lock):
        return AttackerChatTarget(generate_fn=generate_fn, gpu_lock=gpu_lock)

    return factory
