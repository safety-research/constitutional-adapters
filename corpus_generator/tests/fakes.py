"""FakeBatchClient: implements the client surface the engine touches
(messages.batches.create/retrieve/results/list, messages.create,
with_options) from canned responders, with fault-injection at the create
boundary. No network, no keys."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from types import SimpleNamespace as NS

import anthropic
import httpx2


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages/batches")


def transport_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_req())


def bad_request(message: str = "invalid request") -> anthropic.BadRequestError:
    resp = httpx2.Response(400, request=_req(), json={"error": {"message": message}})
    return anthropic.BadRequestError(message, response=resp, body=None)


def rate_limited() -> anthropic.RateLimitError:
    resp = httpx2.Response(429, request=_req(), json={"error": {"message": "rate limited"}})
    return anthropic.RateLimitError("rate limited", response=resp, body=None)


def text_result(custom_id: str, text: str, stop_reason: str = "end_turn") -> NS:
    return NS(
        custom_id=custom_id,
        result=NS(
            type="succeeded",
            message=NS(content=[NS(type="text", text=text)], stop_reason=stop_reason),
        ),
    )


def error_result(custom_id: str, error_type: str = "api_error", message: str = "boom") -> NS:
    return NS(
        custom_id=custom_id,
        result=NS(type="errored", error=NS(type="error", error=NS(type=error_type, message=message))),
    )


def expired_result(custom_id: str) -> NS:
    return NS(custom_id=custom_id, result=NS(type="expired"))


def canceled_result(custom_id: str) -> NS:
    return NS(custom_id=custom_id, result=NS(type="canceled"))


class FakeBatchClient:
    """responder(custom_id, params, attempt) -> result NS (see helpers above).

    create_script: per-create behaviors, popped in order; default 'ok'.
      'ok' | 'transport_no_create' | 'transport_after_create' | '429' | 'bad_request'
    metadata_supported: False -> create with a metadata extra_body raises 400.
    """

    def __init__(self, responder, *, metadata_supported: bool = True):
        self.responder = responder
        self.metadata_supported = metadata_supported
        self.create_script: list[str] = []
        self.retrieve_script: list[str] = []
        self.calls: list[tuple] = []  # (op, detail)
        self.batches: dict[str, dict] = {}
        self._n = 0
        self._attempts: dict[str, int] = {}
        self._with_options: dict = {}
        self.last_create_max_retries: int | None = None
        self.messages = NS(
            batches=NS(
                create=self._create,
                retrieve=self._retrieve,
                results=self._results,
                list=self._list,
            ),
            create=self._messages_create,
        )

    # -- client surface ------------------------------------------------------

    def with_options(self, **kw):
        self._with_options = kw
        return self

    def _messages_create(self, **params):
        self.calls.append(("messages.create", params.get("model")))
        raise AssertionError("offline tests must not hit live messages.create")

    def register_foreign_batch(self, n_requests: int, custom_ids: list[str] | None = None):
        """Plant a batch in the window that no ledger knows about (decoys for
        no-nonce reconciliation tests)."""
        self._n += 1
        bid = f"batch_foreign_{self._n}"
        ids = custom_ids or [f"foreign-{i}" for i in range(n_requests)]
        self.batches[bid] = {
            "requests": [{"custom_id": c, "params": {}} for c in ids],
            "metadata": None,
            "created_at": dt.datetime.now(dt.timezone.utc),
            "status": "ended",
        }
        return bid

    def _create(self, requests, extra_body=None, **kw):
        self.last_create_max_retries = self._with_options.get("max_retries")
        behavior = self.create_script.pop(0) if self.create_script else "ok"
        meta = (extra_body or {}).get("metadata")
        self.calls.append(("create", {"n": len(requests), "metadata": meta, "behavior": behavior}))
        if behavior == "429":
            raise rate_limited()
        if behavior == "bad_request":
            raise bad_request()
        if meta is not None and not self.metadata_supported:
            raise bad_request("metadata: Extra inputs are not permitted")
        if behavior == "transport_no_create":
            raise transport_error()
        self._n += 1
        bid = f"batch_{self._n:04d}"
        self.batches[bid] = {
            "requests": [dict(r) for r in requests],
            "metadata": meta if self.metadata_supported else None,
            "created_at": dt.datetime.now(dt.timezone.utc),
            "status": "ended",
        }
        if behavior == "transport_after_create":
            raise transport_error()
        return self._batch_obj(bid)

    def _batch_obj(self, bid: str):
        b = self.batches[bid]
        return NS(
            id=bid,
            processing_status=b["status"],
            request_counts=NS(total=len(b["requests"])),
            created_at=b["created_at"],
            metadata=b["metadata"],
            model_extra={"metadata": b["metadata"]} if b["metadata"] else {},
        )

    def _retrieve(self, bid: str):
        if self.retrieve_script:
            behavior = self.retrieve_script.pop(0)
            if behavior == "raise":
                raise transport_error()
        self.calls.append(("retrieve", bid))
        return self._batch_obj(bid)

    def _results(self, bid: str):
        self.calls.append(("results", bid))
        for r in self.batches[bid]["requests"]:
            cid = r["custom_id"]
            self._attempts[cid] = self._attempts.get(cid, 0) + 1
            yield self.responder(cid, r.get("params", {}), self._attempts[cid])

    def _list(self, limit: int = 20, **kw):
        self.calls.append(("list", limit))
        ordered = sorted(
            self.batches, key=lambda b: self.batches[b]["created_at"], reverse=True
        )
        return [self._batch_obj(b) for b in ordered]

    # -- test observability ---------------------------------------------------

    def create_count(self) -> int:
        return sum(
            1 for op, d in self.calls
            if op == "create" and d["behavior"] in ("ok", "transport_after_create")
        )

    def op_sequence(self) -> list[str]:
        return [op for op, _ in self.calls]


def overloaded() -> anthropic.APIStatusError:
    resp = httpx2.Response(529, request=_req(), json={"error": {"type": "overloaded_error", "message": "Overloaded"}})
    return anthropic.APIStatusError("Overloaded", response=resp, body={"error": {"type": "overloaded_error", "message": "Overloaded"}})


def live_bad_request(message: str, error_type: str = "invalid_request_error") -> anthropic.BadRequestError:
    body = {"error": {"type": error_type, "message": message}}
    resp = httpx2.Response(400, request=_req(), json=body)
    return anthropic.BadRequestError(message, response=resp, body=body)


def live_message(text: str, stop_reason: str = "end_turn") -> NS:
    return NS(content=[NS(type="text", text=text)], stop_reason=stop_reason)


def live_unauthorized(status: int = 401, api_type: str | None = None) -> anthropic.APIStatusError:
    """The API's credential rejection on a request: 401 ``authentication_error`` (``AuthenticationError``) or 403
    ``permission_error`` (``PermissionDeniedError``), the body shape the API returns -- no key anywhere."""
    api_type = api_type or ("authentication_error" if status == 401 else "permission_error")
    body = {"type": "error", "error": {"type": api_type, "message": "synthetic"}}
    resp = httpx2.Response(status, request=_req(), json=body)
    cls = anthropic.AuthenticationError if status == 401 else anthropic.PermissionDeniedError
    return cls(f"Error code: {status} - {body}", response=resp, body=body)


def federation_rejected(status: int = 401, body=None) -> anthropic.WorkloadIdentityError:
    """The SDK's FEDERATED rejection: under wif the credential is the identity, and the API refuses it at the token exchange
    (``POST /v1/oauth/token``) -- ``WorkloadIdentityError(status_code=401 | 403)`` with a REDACTED body (an OAuth error string, or a
    non-JSON text), an ``AnthropicError`` and NOT an ``APIStatusError``. ``status`` None models an unreachable token endpoint."""
    if status is None:
        return anthropic.WorkloadIdentityError("Failed to reach token endpoint: synthetic")
    body = {"error": "invalid_grant", "error_description": "synthetic"} if body is None else body
    return anthropic.WorkloadIdentityError(f"Token exchange failed (HTTP {status}): {body}", status_code=status, body=body)


def exchange_rejecting_async_client(status: int, body=None):
    """A REAL ``anthropic.AsyncAnthropic`` whose federation token exchange answers ``status`` -- OFFLINE: the credentials' own http
    client is a mock transport for ``/v1/oauth/token`` (``exchange_hits`` counts the exchanges); no bearer is ever minted, so the API
    client's transport is never reached. Placeholder ids; no key anywhere."""
    hits = {"n": 0}
    body = {"error": "invalid_grant", "error_description": "synthetic"} if body is None else body

    def token_endpoint(request):
        assert request.url.path.endswith("/v1/oauth/token"), request.url
        hits["n"] += 1
        if isinstance(body, str):
            return httpx2.Response(status, text=body, request=request)
        return httpx2.Response(status, json=body, request=request)

    creds = anthropic.WorkloadIdentityCredentials(
        identity_token_provider=lambda: "header.payload.signature", federation_rule_id="fr_11111111",
        organization_id="org_22222222", service_account_id="sa_33333333", workspace_id="wrkspc_44444444",
        http_client=httpx2.Client(transport=httpx2.MockTransport(token_endpoint)))
    client = anthropic.AsyncAnthropic(credentials=creds, base_url="https://api.anthropic.com", max_retries=0)
    client.exchange_hits = hits   # type: ignore[attr-defined]
    # the REQUESTS the engine issued (``messages.stream`` calls): the SDK retries a failed exchange once, so exchanges run at
    # up to two per request -- the engine's first-rejection bound is on requests
    requests = {"n": 0}
    stream = client.messages.stream

    def counting_stream(**params):
        requests["n"] += 1
        return stream(**params)

    client.messages.stream = counting_stream    # type: ignore[method-assign]
    client.request_hits = requests              # type: ignore[attr-defined]
    return client


class _FakeLiveStream:
    def __init__(self, client: "FakeLiveClient", params: dict):
        self.client = client
        self.params = params

    async def __aenter__(self):
        key = json.dumps(self.params, sort_keys=True, default=str)
        self.client._attempts[key] = self.client._attempts.get(key, 0) + 1
        self.client.calls += 1
        self.client.in_flight += 1
        self.client.max_in_flight = max(self.client.max_in_flight, self.client.in_flight)
        try:
            self._msg = self.client.responder(self.params, self.client._attempts[key])
        except BaseException:
            self.client.in_flight -= 1
            raise
        return self

    async def __aexit__(self, *exc):
        self.client.in_flight -= 1
        return False

    async def get_final_message(self):
        return self._msg


class FakeLiveClient:
    """The async client surface LiveEngine touches: ``messages.stream(**params)``
    as an async context manager with ``get_final_message()``.

    responder(params, attempt) -> live_message(...) or raises an SDK error
    (rate_limited() / overloaded() / live_bad_request() / transport_error());
    ``attempt`` counts per unique params, so scripts can fail-then-succeed."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._attempts: dict[str, int] = {}
        self.closed = False
        self.messages = NS(stream=lambda **params: _FakeLiveStream(self, params))

    async def close(self):
        self.closed = True


def words_for(key: str, n: int) -> str:
    """Deterministic, key-unique word salad (defeats near-dedup between keys)."""
    h = hashlib.sha256(key.encode()).hexdigest()
    out = []
    i = 0
    while len(out) < n:
        out.append(f"w{h[(i * 3) % 56:(i * 3) % 56 + 3]}{i}")
        i += 1
    return " ".join(out)


class PipelineResponder:
    """Synthesizes VALID stage outputs for any request either pipeline builds,
    keyed off custom_id + the rendered prompt (slot ids are recovered from the
    template's explicit slot-id lines)."""

    SLOT_LINE = re.compile(
        r"slot ids (?:must be )?exactly ([a-z0-9, ]+) in order", re.IGNORECASE
    )
    MISSING_LINE = re.compile(r"missing slots: ([a-z0-9, ]+)\.")

    def __init__(self, sections):
        self.sections = sections

    def _slots(self, prompt: str) -> list[str]:
        m = self.MISSING_LINE.search(prompt)
        if m:
            return [s.strip() for s in m.group(1).split(",") if s.strip()]
        m = self.SLOT_LINE.search(prompt)
        if not m:
            raise AssertionError("responder could not find slot ids in prompt")
        return [s.strip() for s in m.group(1).split(",") if s.strip()]

    @staticmethod
    def _prompt_text(params: dict) -> str:
        content = params["messages"][0]["content"]
        if isinstance(content, str):
            return content
        return "\n".join(b["text"] for b in content)

    def __call__(self, cid: str, params: dict, attempt: int) -> NS:
        # the engine submits run-tagged wire ids ("<8-hex>-<logical>"); the
        # response content is keyed off the LOGICAL id, but the result
        # envelope echoes the wire id back exactly like the real API
        wire_id = cid
        m = re.match(r"^[0-9a-f]{8}-(adv-.*|sto-.*)$", cid)
        if m:
            cid = m.group(1)
        result = self._respond(cid, params, attempt)
        result.custom_id = wire_id
        return result

    def _respond(self, cid: str, params: dict, attempt: int) -> NS:
        prompt = self._prompt_text(params)
        stage = cid.split("-")[1] if cid.startswith(("adv-", "sto-")) else cid
        stage = re.sub(r"(x\d+|r\d+)$", "", stage)

        if stage == "a1":
            items = [
                {
                    "slot": s,
                    "section_id": self.sections[i % len(self.sections)].id,
                    "principle": f"principle {cid} {s} " + words_for(f"{cid}{s}", 12),
                    "tension": f"tension for {s}",
                }
                for i, s in enumerate(self._slots(prompt))
            ]
            return text_result(cid, json.dumps(items))
        if stage == "a2":
            items = [
                {"slot": s, "prompt_type": f"type {cid} {s} " + words_for(f"{cid}{s}", 10)}
                for s in self._slots(prompt)
            ]
            return text_result(cid, json.dumps(items))
        if stage == "a3":
            items = [
                {"slot": s, "scenario": f"scenario {cid} {s}. " + words_for(f"{cid}{s}", 40)}
                for s in self._slots(prompt)
            ]
            return text_result(cid, json.dumps(items))
        if stage == "b1" or stage == "b3":
            no_system = (
                "NO system prompt" in prompt or "must remain user-message-only" in prompt
            )
            parts = []
            if stage == "b3":
                parts.append("<critique>\nfixed the draft\n</critique>")
            if not no_system:
                parts.append(
                    f"<system_prompt>\nsys {cid} {words_for('sys' + cid, 20)}\n</system_prompt>"
                )
            parts.append(
                f"<user_message>\nuser {cid} {words_for('usr' + cid, 60)}\n</user_message>"
            )
            return text_result(cid, "\n".join(parts))
        if stage == "b2":
            return text_result(cid, f"diagnostic {cid} " + words_for("diag" + cid, 50))
        if stage == "c":
            return text_result(cid, f"initial {cid} " + words_for("init" + cid, 120))
        if stage == "d":
            return text_result(
                cid,
                "problems: minor tone issues\n<improved_response>\n"
                f"advice {cid} " + words_for("final" + cid, 150)
                + "\n</improved_response>",
            )
        if stage == "e":
            return text_result(
                cid,
                json.dumps(
                    {"alignment": 8, "realism": 8, "refusal": False,
                     "ai_named": False, "rationale": f"solid ({cid})"},
                ),
            )
        if stage == "s1":
            slots = self._slots(prompt)
            items = [
                {
                    "slot": s,
                    "category": f"category {s} " + words_for("cat" + s, 4),
                    "description": f"desc {s} " + words_for("catd" + s, 15),
                    "register": "literary",
                    "ordinary": i % 2 == 0,
                }
                for i, s in enumerate(slots)
            ]
            return text_result(cid, json.dumps(items))
        if stage == "s2":
            items = [
                {
                    "slot": s,
                    "subcategory": f"subcat {cid} {s} " + words_for(cid + s, 4),
                    "description": f"sdesc {cid} {s} " + words_for("sd" + cid + s, 12),
                }
                for s in self._slots(prompt)
            ]
            return text_result(cid, json.dumps(items))
        if stage == "s3":
            n = max(len(re.findall(r"Story \d+: target length", prompt)), 1)
            parts = [
                f"<story_{i + 1}>\nstory {cid} y{i}. "
                + words_for(f"story{cid}y{i}", 80)
                + "\n</story_{}>".format(i + 1)
                for i in range(n)
            ]
            return text_result(cid, "\n\n".join(parts))
        if stage == "s4":
            m = re.search(r"<story>\n(.*?)\n</story>", prompt, re.DOTALL)
            body = m.group(1) if m else words_for("s4" + cid, 80)
            return text_result(
                cid,
                "problems: none serious\n<improved_document>\n"
                + body + " improved\n</improved_document>",
            )
        if stage == "s5":
            return text_result(
                cid,
                json.dumps(
                    {"constitution_consistency": 8, "quality": 8, "ai_named": False,
                     "factual_claims": False, "rationale": f"good ({cid})"},
                ),
            )
        raise AssertionError(f"responder: unrecognized custom_id {cid!r}")
