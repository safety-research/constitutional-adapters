"""What one OpenRouter Batch API result entry IS, before a family reads its verdict: an error entry classified as an ACCOUNT failure,
a MODERATION rejection or a provider error, with a summary that keeps codes, types, the message and the moderation reasons but never
the flagged input text (only its length).

THE ORDER, and why moderation needs POSITIVE evidence. An error entry is classified (:func:`classify_error`) as:
  1. ``account_error`` -- the provider refused the CREDENTIAL or the ACCOUNT, not this input: any 401, an ``error_type``
     authentication / permission_denied, or account wording (OpenAI's ``Policy Violation: this user has been blocked for a previous
     policy violation``, as OpenAI models answer a blocked account; a dead own-provider key's ``Incorrect API key``).
     Tested FIRST: that block says "policy violation" and carries ``error_type: refusal``, exactly like a content block.
  2. ``provider_error`` -- the money (402, or a message naming a spending / key / credit limit): re-asked, never filed as a row.
  3. ``moderation`` -- ONLY on positive evidence of a content decision about THIS input: OpenRouter's moderation metadata
     (``reasons`` / ``flagged_input`` / ``categories``), ``error_type`` content_policy_violation, a policy / content-filter provider code
     (OpenAI's ``bio_policy`` / ``cyber_policy`` blocks, which arrive as a 403 ``refusal``), or content wording in the MESSAGE
     (moderation, flagged, prohibited content, content filter / policy / management, safety system, potentially violating,
     biological or cybersecurity risk). Never a bare 403, never the words "violation" / "violates", and never ``error_type: refusal`` alone: the
     account block carries all three.
  4. any other 403 -- ``account_error``: a refusal that names no content is the gateway or the provider refusing THIS ACCOUNT or this
     route (a reworded block, a region or model-access 403, a workspace guardrail). The run pauses (:mod:`rejudge.terra.harvest`) and
     nothing is recorded against the items, so a wording nobody foresaw costs a person's look, never a run of false moderation rows.
  5. anything else -- ``provider_error`` (re-asked).
Only the message and the typed fields are read, never the whole object: ``metadata.raw`` / ``flagged_input`` can quote the judged text.
``account_error`` is never terminal and never counted against an item's re-asks (:mod:`rejudge.terra.harvest`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

#: the provider refusing the CREDENTIAL or the ACCOUNT (never a property of one input); kept specific, since a bare 403 is handled below
ACCOUNT_FAILURE = re.compile(
    r"\bHTTP 401\b|incorrect api key|invalid api key|invalid_api_key|api key (is )?(invalid|expired|revoked|disabled)|\bunauthori[sz]ed\b|"
    r"no auth credentials|blocked for a previous policy violation|"
    r"\b(user|account|organi[sz]ation|project|workspace|key) (has been|is|was) (blocked|banned|suspended|deactivated|disabled|revoked|restricted|flagged)\b|"
    r"exceeded your current quota|insufficient_quota|billing (is )?(not active|hard limit)|"
    r"not supported in your (country|region|territory)|country, region, or territory not supported|unsupported_country|no eligible key", re.I)
ACCOUNT_ERROR_TYPES = ("authentication", "authentication_error", "permission_denied")
#: the money: the key's limit or the account's balance, never the item's
LIMIT_TEXT = re.compile(r"\b(spending|budget|key|credit|weekly|daily|monthly)\s+limit\b|insufficient (credits|balance|funds)|"
                        r"exceeds your (balance|credits)|out of credits|add (more )?credits|\bpayment required\b", re.I)
#: positive evidence, in the MESSAGE, that THIS input was refused on its content
MODERATION_TEXT = re.compile(r"moderat\w*|\bflagged\b|prohibited content|content[ _-]?(filter|policy|management)|safety system|"
                             r"potentially violating|bio(logical)? risk|cyber(security)? risk", re.I)
MODERATION_ERROR_TYPES = ("content_policy_violation",)
MODERATION_METADATA_KEYS = ("reasons", "flagged_input", "flagged_input_chars", "categories")


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def err_summary(err: Any, status: Optional[int]) -> Dict[str, Any]:
    """An error object reduced to what is safe and useful to keep: code, type, message (cut), the metadata's reasons / categories /
    provider / model and the LENGTH of any flagged input -- never the flagged text itself."""
    if not isinstance(err, Mapping):
        return {"status_code": status, "message": str(err)[:1000]}
    out: Dict[str, Any] = {"status_code": status}
    for k in ("code", "type", "error_type", "param"):
        if err.get(k) is not None:
            out[k] = err.get(k)
    out["message"] = str(err.get("message") or "")[:1000]
    md = err.get("metadata")
    if isinstance(md, Mapping):
        keep: Dict[str, Any] = {}
        for k, v in md.items():
            if k == "flagged_input":
                keep["flagged_input_chars"] = len(str(v))
            elif k in ("reasons", "categories", "provider_name", "model_slug", "error_type", "provider_code", "limit_source"):
                keep[k] = v
            elif k == "raw":
                keep["raw"] = str(v)[:500]
        out["metadata"] = keep
    return out


def classify_error(status: Optional[int], summary: Mapping[str, Any]) -> str:
    """``account_error``, ``provider_error`` or ``moderation`` for one error entry's summary (:func:`err_summary`), in the module
    docstring's order (the evidence behind a ``moderation`` is named by :func:`moderation_evidence`)."""
    message = str(_get(summary, "message") or "")
    etypes = {str(x).lower() for x in (_get(summary, "error_type"), _get(summary, "metadata", "error_type"), _get(summary, "type")) if x}
    pcode = str(_get(summary, "metadata", "provider_code") or _get(summary, "provider_code") or "").lower()
    scode = str(_get(summary, "code")) if isinstance(_get(summary, "code"), str) else ""
    try:
        st = int(status) if status is not None else None
    except (TypeError, ValueError):
        st = None
    if st == 401 or etypes & set(ACCOUNT_ERROR_TYPES) or any(ACCOUNT_FAILURE.search(t) for t in (message, pcode, scode) if t):
        return "account_error"
    if st == 402 or LIMIT_TEXT.search(message) or _get(summary, "metadata", "limit_source"):
        return "provider_error"
    if moderation_evidence(summary):
        return "moderation"
    if st == 403:
        return "account_error"
    return "provider_error"


def moderation_evidence(summary: Mapping[str, Any]) -> Optional[str]:
    """The positive evidence that an error entry is a content decision about THIS input, named (``openrouter_moderation``,
    ``content_policy_violation``, ``provider_policy:<code>``, ``message_wording``), else None."""
    md = _get(summary, "metadata")
    if isinstance(md, Mapping) and any(md.get(k) for k in MODERATION_METADATA_KEYS):
        return "openrouter_moderation"
    etypes = {str(x).lower() for x in (_get(summary, "error_type"), _get(summary, "metadata", "error_type"), _get(summary, "type")) if x}
    if etypes & set(MODERATION_ERROR_TYPES):
        return "content_policy_violation"
    pcode = str(_get(summary, "metadata", "provider_code") or _get(summary, "provider_code") or "").lower()
    if pcode.endswith("_policy") or "content_filter" in pcode or "moderation" in pcode:
        return f"provider_policy:{pcode}"
    if MODERATION_TEXT.search(str(_get(summary, "message") or "")):
        return "message_wording"
    return None


def _outcome(summary: Dict[str, Any], status: Optional[int]) -> Dict[str, Any]:
    o = classify_error(status, summary)
    err = dict(summary, kind=o)
    if o == "moderation":
        err["moderation_source"] = moderation_evidence(summary)
    return {"outcome": o, "error": err, "status_code": status}


def error_outcome(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The outcome of an entry that carries an error (top-level ``error``, a non-200 ``response.status_code``, or an ``error`` inside
    a 200 body -- OpenRouter's in-band provider errors), else None (the entry has a completion to read)."""
    resp = entry.get("response")
    err = entry.get("error")
    if err:
        status = None
        if isinstance(err, Mapping):
            status = err.get("code") if isinstance(err.get("code"), int) else err.get("status_code")
        return _outcome(err_summary(err, status), status)
    if not isinstance(resp, Mapping):
        return {"outcome": "provider_error", "error": {"kind": "no_response", "message": "result entry has neither response nor error"}}
    status = resp.get("status_code")
    body = resp.get("body") or {}
    if status is not None and int(status) != 200:
        inner = body.get("error") if isinstance(body, Mapping) and body.get("error") else body
        return _outcome(err_summary(inner, status), int(status))
    if isinstance(body, Mapping) and body.get("error"):
        inner = body["error"]
        code = inner.get("code") if isinstance(inner, Mapping) and isinstance(inner.get("code"), int) else status
        return _outcome(err_summary(inner, code), code)
    return None


def completion(entry: Mapping[str, Any]) -> Tuple[Dict[str, Any], Mapping[str, Any], Optional[str]]:
    """``(base, message, finish_reason)`` of an entry with a completion: ``base`` carries status, usage, provider and model."""
    resp = entry.get("response") or {}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    ch = (choices[0] or {}) if choices else {}
    finish = ch.get("finish_reason") or ch.get("native_finish_reason")
    base = {"status_code": resp.get("status_code"), "usage": body.get("usage"), "provider": body.get("provider"),
            "model": body.get("model"), "finish_reason": finish}
    if not choices:
        base["_no_choices"] = True
    return base, (ch.get("message") or {}), finish
