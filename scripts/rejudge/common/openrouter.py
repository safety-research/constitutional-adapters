"""OpenRouter Batch API client (stdlib only).

Endpoints (https://openrouter.ai/docs/guides/features/batch-api, re-read 2026-11-09):
  POST   /api/v1/batches          {endpoint, model, [provider], [completion_window], requests: [{custom_id, body}]} -> 202 + batch
  GET    /api/v1/batches/{id}     the batch; when status == completed, ``results`` holds one entry per request (not in input order)
  GET    /api/v1/batches?status=  list (metadata only; ``results`` always null), newest first, cursor ``after``
  DELETE /api/v1/batches/{id}     only a finished batch (409 while running); there is no cancel
  GET    /api/v1/key              the key's limit / usage / remaining (no secret in the reply)

THE KEY. Read from the environment variable named at construction (default ``OPENROUTER_API_KEY``) and kept in the instance only:
never printed, logged, written, or put on a command line. :meth:`BatchClient.__repr__` and every error message omit it, and
:func:`redact` scrubs it from any text this module returns, in case a server ever echoes the header.

Errors are classified by :func:`classify_error` into the kinds the submitter acts on:
  too_large      413: more requests than the per-batch cap, or a body over 200 MB (the message states the limit; :func:`limit_from_message`)
  rate_limited   429: too many submits, batches in progress, or requests in progress
  balance        402 (the estimated cost exceeds the balance), or a 403 whose message names a spending limit / credits / budget
  moderation     403 naming moderation / flagged content (OpenRouter moderates OpenAI models)
  auth           401, a 403 naming the account or credential (:func:`is_account_failure`, tested FIRST), or another 403
  bad_request    400 / 404 / 422
  not_found      404 on a batch id; gone: 410
  conflict       409
  transient      408 / 5xx / connection errors / timeouts

ONE REQUEST'S error inside a finished batch (a ``results[]`` entry's ``error``, a non-200 ``response``, or a 200 body carrying ``error``)
is classified by :func:`classify_result_error`, which every track's parser calls, in this order:
  account_error  the provider refused the CREDENTIAL or the ACCOUNT, not this input (a dead own-provider key: ``HTTP 401: Incorrect API
                 key provided``; an account block: ``Policy Violation: this user has been blocked for a previous policy violation``; any
                 401; ``error_type`` authentication / permission_denied; a region block). Tested BEFORE anything else: such an error
                 says nothing about the item, so the harvester never makes it terminal and PAUSES the run.
  balance_error  the key's or account's money ran out (402, ``error_type`` payment_required, ``metadata.limit_source``, a message naming a
                 spending / key / budget / credit limit). Never terminal; the harvester re-asks it after a backoff.
  moderation     ONLY on positive evidence of a content decision about THIS input: OpenRouter's moderation metadata (``reasons`` /
                 ``flagged_input``), ``error_type`` content_policy_violation / refusal (a provider policy block such as OpenAI's
                 ``bio_policy`` / ``cyber_policy``), a ``*_policy`` / content-filter provider code, or a message naming moderation, a
                 flag, prohibited content, a content filter / policy or a safety system. NEVER a bare 403 or a word like "violation".
  provider_error everything else (re-askable; a flood of identical ones pauses the run, :mod:`harvester`).
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

BASE_URL = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"
USER_AGENT = "rejudge-batch/1 (python-urllib)"
FINAL_STATUSES = ("completed", "failed", "expired", "cancelled")

_LIMIT_WORDS = re.compile(r"\b(limit|credit|credits|balance|budget|spend|spending|quota|insufficient|afford|payment)\b", re.I)
_MODERATION_WORDS = re.compile(r"\b(moderat\w*|flagged|prohibited content|content policy|requires moderation)\b", re.I)

#: the provider refusing the CREDENTIAL or the ACCOUNT -- never a property of one input. Kept specific on purpose: a match pauses the run
#: (the safe direction), and a bare 403 or the word "violation" is NOT here (moderation and policy blocks carry both)
ACCOUNT_FAILURE = re.compile(
    r"\bHTTP 401\b|incorrect api key|invalid api key|invalid_api_key|api key (is )?(invalid|expired|revoked|disabled)|\bunauthori[sz]ed\b|"
    r"blocked for a previous policy violation|"
    r"\b(user|account|organi[sz]ation|project|workspace|key) (has been|is|was) (blocked|banned|suspended|deactivated|disabled|revoked)\b|"
    r"exceeded your current quota|insufficient_quota|billing (is )?(not active|hard limit)|"
    r"not supported in your (country|region|territory)|country, region, or territory not supported|unsupported_country|no eligible key", re.I)
#: a whole-BATCH failure message (``error.message`` of a failed batch) adds the bare HTTP 401 / 403 status: a batch-level 403 is never
#: one item's moderation (OpenRouter names the request when screening rejects one, :mod:`harvester`)
BATCH_ACCOUNT_FAILURE = re.compile(ACCOUNT_FAILURE.pattern + r"|\bHTTP 40[13]\b", re.I)
ACCOUNT_ERROR_TYPES = ("authentication", "authentication_error", "permission_denied")
#: the money, not the input (a per-request 402 / budget 403: OpenRouter's "Budget errors" contract names ``limit_source``)
_BALANCE_TEXT = re.compile(r"\b(spending|budget|key|credit|weekly|daily|monthly)\s+limit\b|insufficient (credits|balance|funds)|"
                           r"exceeds your (balance|credits)|\$[\d.,]+ of \$[\d.,]+ used|out of credits|add (more )?credits", re.I)
BALANCE_ERROR_TYPES = ("payment_required",)
#: positive evidence that THIS input was refused on its content
_MODERATION_TEXT = re.compile(r"moderat\w*|\bflagged\b|prohibited content|content[ _-]?(filter|policy|management)|safety system|"
                              r"usage polic(y|ies)|potentially violating|bio(logical)? risk|cyber(security)? risk", re.I)
MODERATION_ERROR_TYPES = ("content_policy_violation", "refusal")
MODERATION_METADATA_KEYS = ("reasons", "flagged_input", "flagged_input_chars", "categories")


def is_account_failure(text: str, batch_level: bool = False) -> bool:
    """True when ``text`` (an error message, or the JSON of an error object) names the provider refusing the credential or the account.
    ``batch_level`` also counts a bare ``HTTP 401`` / ``HTTP 403`` (a whole batch's failure message)."""
    return bool((BATCH_ACCOUNT_FAILURE if batch_level else ACCOUNT_FAILURE).search(text or ""))


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def classify_result_error(status: Optional[int], err: Any) -> str:
    """One request's error inside a finished batch -> ``account_error`` | ``balance_error`` | ``moderation`` | ``provider_error`` (module
    docstring, in that order). ``status`` is the request's HTTP status (the error's ``code`` or the response's ``status_code``), ``err``
    the error object (``{code, message, type, error_type, metadata: {error_type, provider_code, reasons, flagged_input, limit_source,
    raw, ...}}``) or its text."""
    # only the MESSAGE and the typed fields are read, never the whole object: ``metadata.raw`` / ``flagged_input`` can quote the judged
    # transcript, and a jailbreak response that says "your account has been suspended" must not pause the run
    message = str(_get(err, "message") or "") if isinstance(err, Mapping) else str(err or "")
    etypes = {str(x).lower() for x in (_get(err, "error_type"), _get(err, "metadata", "error_type"), _get(err, "type")) if x}
    pcode = str(_get(err, "metadata", "provider_code") or _get(err, "provider_code") or "").lower()
    scode = _get(err, "code") if isinstance(_get(err, "code"), str) else ""      # OpenAI's string codes: invalid_api_key, insufficient_quota
    md = _get(err, "metadata")
    try:
        st = int(status) if status is not None else None
    except (TypeError, ValueError):
        st = None
    # 1. the account or the credential -- before anything else
    if st == 401 or etypes & set(ACCOUNT_ERROR_TYPES) or is_account_failure(message) or is_account_failure(pcode) \
            or is_account_failure(scode):
        return "account_error"
    # 2. the money
    if st == 402 or etypes & set(BALANCE_ERROR_TYPES) or _get(err, "metadata", "limit_source") or _BALANCE_TEXT.search(message):
        return "balance_error"
    # 3. a content decision about THIS input, on positive evidence only
    if (isinstance(md, Mapping) and any(k in md for k in MODERATION_METADATA_KEYS)) or etypes & set(MODERATION_ERROR_TYPES) \
            or pcode.endswith("_policy") or "content_filter" in pcode or "moderation" in pcode or _MODERATION_TEXT.search(message):
        return "moderation"
    return "provider_error"


class OpenRouterError(Exception):
    """An HTTP or transport failure, classified. ``kind`` is one of the kinds in the module docstring; ``status`` the HTTP status
    (0 for a transport error); ``message`` the server's ``error.message`` (redacted); ``body`` the parsed JSON body when there was one."""

    def __init__(self, kind: str, status: int, message: str, body: Optional[Any] = None, retry_after_s: Optional[float] = None):
        super().__init__(f"{kind} (HTTP {status}): {message}")
        self.kind = kind
        self.status = status
        self.message = message
        self.body = body
        self.retry_after_s = retry_after_s


def redact(text: str, key: Optional[str]) -> str:
    """``text`` with any occurrence of the key (and of a ``Bearer`` header value) replaced by a placeholder."""
    if not isinstance(text, str):
        return text
    if key:
        text = text.replace(key, "<redacted-key>")
    return re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", r"\1<redacted>", text)


def error_message(body: Any) -> str:
    """The human message of an OpenRouter error body (``{"error": {"message": ...}}``), else the body as text."""
    if isinstance(body, Mapping):
        err = body.get("error")
        if isinstance(err, Mapping) and err.get("message") is not None:
            return str(err.get("message"))
        if isinstance(err, str):
            return err
        if body.get("message") is not None:
            return str(body.get("message"))
    return "" if body is None else (body if isinstance(body, str) else json.dumps(body)[:2000])


def classify_error(status: int, message: str) -> str:
    """The error kind for an HTTP status and message (module docstring)."""
    msg = message or ""
    if status == 413:
        return "too_large"
    if status == 429:
        return "rate_limited"
    if status == 402:
        return "balance"
    if status == 403:
        if is_account_failure(msg):
            return "auth"
        if _MODERATION_WORDS.search(msg):
            return "moderation"
        if _LIMIT_WORDS.search(msg):
            return "balance"
        return "auth"
    if status == 401:
        return "auth"
    if status == 404:
        return "not_found"
    if status == 410:
        return "gone"
    if status == 409:
        return "conflict"
    if status in (400, 422):
        return "bad_request"
    if status == 408 or status >= 500 or status == 0:
        return "transient"
    return "bad_request" if 400 <= status < 500 else "transient"


def limit_from_message(message: str) -> Optional[int]:
    """The request-count limit a 413 states (``... more than 50,000 requests ...``), else None."""
    m = re.search(r"(\d[\d,]*)\s*(?:requests|request limit)", message or "", re.I) or re.search(r"limit(?: of| is)?\s*(\d[\d,]*)", message or "", re.I)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


class BatchClient:
    """The Batch API over urllib. ``timeout_s`` bounds every socket operation; ``max_get_retries`` retries a GET (never a POST: a
    submit that timed out may have been accepted, and the caller decides -- :mod:`submitter` lists the account's batches to find it)."""

    def __init__(self, key_env: str = KEY_ENV, base_url: str = BASE_URL, timeout_s: float = 600.0, max_get_retries: int = 4):
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"the OpenRouter key is not in the environment variable {key_env} (load it from the secret store; never write it to a file)")
        self._key = key
        self.key_env = key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_get_retries = max_get_retries

    def __repr__(self) -> str:
        return f"BatchClient(base_url={self.base_url!r}, key_env={self.key_env!r})"

    # -- transport ------------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[bytes] = None, params: Optional[List[Tuple[str, str]]] = None) -> Tuple[int, Any, Dict[str, str]]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                status = resp.status
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            status = e.code
            hdrs = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            raise OpenRouterError("transient", 0, redact(f"{type(e).__name__}: {e}", self._key))
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            parsed = redact(raw[:4000].decode("utf-8", "replace"), self._key)
        return status, parsed, hdrs

    def _raise_for(self, status: int, parsed: Any, hdrs: Mapping[str, str]) -> None:
        msg = redact(error_message(parsed), self._key)
        retry_after = None
        if hdrs.get("retry-after"):
            try:
                retry_after = float(hdrs["retry-after"])
            except ValueError:
                retry_after = None
        raise OpenRouterError(classify_error(status, msg), status, msg, parsed if isinstance(parsed, (dict, list)) else None, retry_after)

    def _get(self, path: str, params: Optional[List[Tuple[str, str]]] = None) -> Any:
        last: Optional[OpenRouterError] = None
        for attempt in range(self.max_get_retries + 1):
            try:
                status, parsed, hdrs = self._request("GET", path, params=params)
            except OpenRouterError as e:
                last = e
            else:
                if 200 <= status < 300:
                    return parsed
                try:
                    self._raise_for(status, parsed, hdrs)
                except OpenRouterError as e:
                    if e.kind not in ("transient", "rate_limited"):
                        raise
                    last = e
            if attempt < self.max_get_retries:
                time.sleep(min(120.0, (last.retry_after_s if last and last.retry_after_s else 0) or 5.0 * (2 ** attempt)))
        assert last is not None
        raise last

    # -- the API ------------------------------------------------------------------------------
    def submit(self, body: bytes) -> Dict[str, Any]:
        """POST one batch body (bytes, already serialized with ``requests`` LAST: :func:`batchfile.serialize_batch`). Returns the
        batch object on 202 (or any 2xx); raises :class:`OpenRouterError` otherwise. Never retried here."""
        status, parsed, hdrs = self._request("POST", "/batches", body=body)
        if 200 <= status < 300 and isinstance(parsed, dict):
            return parsed
        self._raise_for(status, parsed, hdrs)
        raise AssertionError("unreachable")

    def get(self, batch_id: str) -> Dict[str, Any]:
        return self._get(f"/batches/{urllib.parse.quote(batch_id)}")

    def list(self, statuses: Optional[List[str]] = None, limit: int = 100, after: Optional[str] = None,
             created_after: Optional[str] = None) -> Dict[str, Any]:
        params: List[Tuple[str, str]] = [("limit", str(limit))]
        for s in statuses or []:
            params.append(("status", s))
        if after:
            params.append(("after", after))
        if created_after:
            params.append(("created_after", created_after))
        return self._get("/batches", params)

    def iter_batches(self, statuses: Optional[List[str]] = None, created_after: Optional[str] = None, max_pages: int = 1000) -> Iterator[Dict[str, Any]]:
        after = None
        for _ in range(max_pages):
            page = self.list(statuses, 100, after, created_after)
            for b in page.get("data") or []:
                yield b
            if not page.get("has_more"):
                return
            after = page.get("last_id")
            if not after:
                return

    def delete(self, batch_id: str) -> Dict[str, Any]:
        status, parsed, hdrs = self._request("DELETE", f"/batches/{urllib.parse.quote(batch_id)}")
        if 200 <= status < 300:
            return parsed if isinstance(parsed, dict) else {}
        self._raise_for(status, parsed, hdrs)
        raise AssertionError("unreachable")

    def key_info(self) -> Dict[str, Any]:
        """The key's spending limit and usage (``limit``, ``limit_remaining``, ``usage``, ``usage_daily`` ...), label dropped."""
        d = self._get("/key")
        d = d.get("data", d) if isinstance(d, dict) else {}
        keep = ("limit", "limit_remaining", "limit_reset", "usage", "usage_daily", "usage_weekly", "usage_monthly", "is_free_tier")
        return {k: d.get(k) for k in keep if k in d}
