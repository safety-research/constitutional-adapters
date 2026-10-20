"""OpenRouter-hosted ATTACKER for the gated battery (opt-in), plus a recording
wrapper that logs every adversarial-chat call.

``scripts/jailbreak_run.py generate --attacker-engine openrouter --attacker-model
x-ai/grok-4.6`` drives the adversarial chat through an OpenAI-compatible
hosted endpoint instead of the campaign's vLLM-served abliterated Qwen. The
attack code is untouched: :class:`OpenRouterAttackerClient` is a
:class:`~sisteer.eval.attacks.vllm_attacker.VLLMAttackerClient` whose
transport speaks OpenRouter:

- the bearer key rides the session headers (read from an env var by the
  driver — never a file in the repo, never printed);
- ``check_health`` uses ``/v1/key`` (auth) + ``/v1/models`` (the model id is
  actually offered) — OpenRouter has no ``/health``;
- the payload is the plain OpenAI subset ``model / messages / temperature /
  top_p / max_tokens`` (+ an optional ``reasoning`` block). The vLLM client's
  ``chat_template_kwargs`` is NOT sent, and nothing Anthropic-only
  (``thinking``, ``temperature=None``) ever is — those 400 on OpenRouter (a
  previous grok run in this repo hit exactly that);
- a 4xx other than 429 is FINAL (no retry — a bad param must surface at once,
  which is why the driver's preflight sends one real request first).

Provider facts measured 2026-09-02 for ``x-ai/grok-4.6`` (recorded so the
scorer contract is understood): reasoning is MANDATORY and cannot be disabled
(``reasoning.enabled=false`` -> 400); ``max_tokens`` caps the VISIBLE
completion only (a ``max_tokens=8`` rubric call still returns a full integer
answer while reasoning ~200-400 tokens); the reasoning text arrives in
``message.reasoning``, never inline, so ``content`` needs no stripping.

The attacker is ALSO a safety-aligned model here, which the campaign's design
note says distorts ASR — that is the QUESTION this client exists to measure
(does a frontier attacker refuse to red-team an open-weight target?), not a
default. The default engine is unchanged.

:class:`RecordingAttackerClient` wraps ANY attacker client and appends one
JSONL row per call (the full chat sent, the reply, provider metadata, latency,
or the exception) so refusals and parse failures can be counted and quoted
post hoc — the attack drivers otherwise keep only the target's replies.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import requests

from ..jailbreak_artifacts import append_jsonl
from .vllm_attacker import VLLMAttackerClient, VLLMEndpointUnreachable, _strip_reasoning

DEFAULT_ENDPOINT = "https://openrouter.ai/api"


class OpenRouterError(VLLMEndpointUnreachable):
    """A FINAL OpenRouter failure (bad key, bad param, unknown model)."""


class OpenRouterAttackerClient(VLLMAttackerClient):
    """OpenAI-compatible chat client for an OpenRouter-hosted attacker."""

    def __init__(self, model: str, *, api_key: str,
                 endpoint: str = DEFAULT_ENDPOINT,
                 reasoning: Optional[Mapping[str, Any]] = None,
                 timeout: float = 600.0, health_timeout: float = 20.0,
                 max_retries: int = 3, backoff: float = 2.0,
                 session=None) -> None:
        if not api_key:
            raise ValueError("OpenRouterAttackerClient needs a non-empty api_key")
        if not model:
            raise ValueError("OpenRouterAttackerClient needs a model id")
        super().__init__(endpoint or DEFAULT_ENDPOINT, model, timeout=timeout,
                         health_timeout=health_timeout, max_retries=max_retries,
                         backoff=backoff, session=session)
        self._session.headers.update({"Authorization": f"Bearer {api_key}",
                                      "Content-Type": "application/json"})
        self.reasoning = dict(reasoning) if reasoning else None
        #: provider metadata of the LAST call (finish reasons, usage,
        #: reasoning length, latency) for the recording wrapper / preflight.
        self.last_meta: dict = {}

    # -- health ------------------------------------------------------------- #
    def check_health(self) -> None:
        for path in ("/v1/key", "/v1/models"):
            url = f"{self.endpoint}{path}"
            try:
                resp = self._session.get(url, timeout=self.health_timeout)
            except Exception as exc:  # noqa: BLE001
                raise OpenRouterError(
                    f"OpenRouter {url} unreachable ({type(exc).__name__}: "
                    f"{exc})") from exc
            status = getattr(resp, "status_code", None)
            if status is None or not (200 <= status < 300):
                raise OpenRouterError(
                    f"OpenRouter {url} answered {status} — "
                    + ("bad/expired key" if path == "/v1/key"
                       else "endpoint is not OpenRouter"))
            if path == "/v1/models":
                ids = {m.get("id") for m in (resp.json().get("data") or [])
                       if isinstance(m, Mapping)}
                if self.model not in ids:
                    raise OpenRouterError(
                        f"OpenRouter does not list model {self.model!r} "
                        f"({len(ids)} models offered)")

    # -- chat --------------------------------------------------------------- #
    def build_payload(self, messages: list[dict], *, temperature: float,
                      top_p: float, max_new_tokens: int) -> dict:
        payload = {"model": self.model, "messages": messages,
                   "temperature": float(temperature), "top_p": float(top_p),
                   "max_tokens": int(max_new_tokens)}
        if self.reasoning:
            payload["reasoning"] = dict(self.reasoning)
        return payload

    def chat(self, messages: list[dict], *, temperature: float, top_p: float,
             max_new_tokens: int) -> str:
        url = f"{self.endpoint}/v1/chat/completions"
        payload = self.build_payload(messages, temperature=temperature,
                                     top_p=top_p, max_new_tokens=max_new_tokens)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                status = getattr(resp, "status_code", 200)
                if status == 429 or status >= 500:
                    raise VLLMEndpointUnreachable(
                        f"OpenRouter returned {status} for POST {url}: "
                        f"{_body(resp)[:300]}")
                if status >= 400:  # FINAL: a param/key/model problem
                    raise OpenRouterError(
                        f"OpenRouter returned {status} for POST {url}: "
                        f"{_body(resp)[:600]} — payload keys "
                        f"{sorted(payload)}")
                data = resp.json()
                if "error" in data and not data.get("choices"):
                    raise VLLMEndpointUnreachable(
                        f"OpenRouter error body: {str(data['error'])[:300]}")
                choice = data["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content")
                usage = data.get("usage") or {}
                self.last_meta = {
                    "latency_s": round(time.time() - t0, 3),
                    "finish_reason": choice.get("finish_reason"),
                    "native_finish_reason": choice.get("native_finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": ((usage.get("completion_tokens_details")
                                          or {}).get("reasoning_tokens")),
                    "reasoning_chars": len(msg.get("reasoning") or ""),
                    "refusal_field": msg.get("refusal"),
                    "provider": data.get("provider"),
                    "attempt": attempt,
                }
                return _strip_reasoning(content) if content is not None else ""
            except OpenRouterError:
                raise
            except Exception as exc:  # noqa: BLE001 — transient: retry, then fail loud
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise VLLMEndpointUnreachable(
                    f"OpenRouter request to {url} failed after "
                    f"{self.max_retries + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}") from exc
        raise last_exc  # unreachable


def _body(resp) -> str:
    try:
        return resp.text
    except Exception:  # noqa: BLE001
        return ""


class RecordingAttackerClient:
    """Wrap an attacker/scorer client and append one JSONL row per ``chat``
    call: the full chat sent, the reply (or the exception), the provider
    metadata the client exposes (``last_meta``) and the latency. ``role``
    labels the stream (``attacker`` / ``scorer``); ``max_reply_chars`` caps
    a stored reply (None = full)."""

    def __init__(self, inner, log_path: str | Path, *, role: str = "attacker",
                 max_reply_chars: Optional[int] = None) -> None:
        self._inner = inner
        self.log_path = Path(log_path)
        self.role = role
        self.max_reply_chars = max_reply_chars
        self.n_calls = 0
        self.n_errors = 0

    def __getattr__(self, name):  # health checks, .model, .endpoint, ...
        return getattr(self._inner, name)

    def chat(self, messages: list[dict], **kw) -> str:
        t0 = time.time()
        self.n_calls += 1
        row: dict = {"role": self.role, "seq": self.n_calls, "t_start": t0,
                     "n_messages": len(messages), "messages": messages,
                     "params": {k: v for k, v in kw.items()}}
        try:
            out = self._inner.chat(messages, **kw)
        except Exception as exc:  # noqa: BLE001 — record, then re-raise unchanged
            self.n_errors += 1
            row.update(latency_s=round(time.time() - t0, 3), error=repr(exc)[:2000])
            append_jsonl(self.log_path, row)
            raise
        row.update(latency_s=round(time.time() - t0, 3),
                   reply=(out if self.max_reply_chars is None
                          else (out or "")[: self.max_reply_chars]),
                   reply_chars=len(out or ""),
                   meta=dict(getattr(self._inner, "last_meta", None) or {}))
        append_jsonl(self.log_path, row)
        return out
