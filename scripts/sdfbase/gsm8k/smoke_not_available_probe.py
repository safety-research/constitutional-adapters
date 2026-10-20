#!/usr/bin/env python3
"""The NO-SWITCH DIAGNOSTIC smoke's CPU-side probe (llama-paper-rows): the
certification of a switch-less family's render fact, on the MATH-500 smoke's shape
(``scripts/sdfbase/math500/smoke_thinking_probe.py``) for a family whose chat template carries NO thinking switch.
Runs from the client venv against a live BASE-ROUTE standup THROUGH the usage ledger, and writes records that CANNOT
be read as production results — every one carries ``diagnostic: true``, the out root is
``<wave root>/diagnostic/<family>/smoke_<job>/``, no tinycap store is written.

Why a probe and not ``served_capability_run.py --diagnostic``: the driver's contracts refuse the smoke's shape by
design — ``contracts.ON_CAP_BY_TASK_ALLOWED`` admits 8,192 / 16,384 only (cap 1,536 is not a plan value) and
``--item-subset`` is the 16,384 RUNG's mechanism (a source store's identity hash, ``source_cap < cap``, a transport
verdict). The probe sends what the driver would send on the base route, minus the store.

What it certifies — through the contracts' OWN seam, ``contracts.no_switch_render_certified`` (predicate (A″), the
one the driver and the analysis' reader share), never a re-derivation:
  (i)   the assistant header measured ONCE through live ``/v1/tokenize`` with NO ``chat_template_kwargs`` by the
        construction the template's shape admits (:func:`served_no_switch_header` — the driver's
        ``_served_no_switch_header`` mirrored: the generation-prompt difference on a gated template; the unconditional
        trailing header, located by content substitution, on a template that never consults ``add_generation_prompt``)
        EQUALS ``FAMILY_RENDER_FACTS[family]``'s single render by TUPLE (the fact records it under both id fields;
        ``header_moves`` False by construction);
  (ii)  N GSM8K paired-200 items (the first N in the served task's own plan order, rendered through its solver —
        ``inspect_served.task_messages``) on the base route at the smoke cap, under the model's ``card`` sampling row
        and the seed rule (``contracts.SEED_RULE_INSPECT``, identity version 2's — ``inspect_served.sample_seed``),
        with BOUNDED concurrency through the ledger; every request body is FREE of ``chat_template_kwargs``
        (asserted at construction, recorded per row as ``body_has_chat_template_kwargs: false`` beside the sha256 of
        the exact bytes sent); per row: the proxy's stamps (``render_regime`` must read ``unspecified`` — the proxy's
        literal for an absent switch; ``cache_salt_regime`` / ``resumed_prefill_policy`` the row's expected pair),
        ``finish_reason``, usage, the raw completion, ``reasoning_content`` (must be empty: there is no channel),
        the ``none``-split reading (``think_split.split_think``: the content IS the answer, no block, a tag literal is
        a STRAY) and the per-completion certification record;
  (iii) the ledger read back: rows == requests, ``error_code`` null and ``error_class none`` on every row, every row's
        ``thinking`` NULL (P17: the switch was ABSENT, never ``false``), every row's ``body_sha256`` EQUAL to the
        sha256 of the bytes this probe sent under that ``request_id`` — so what the ledger FORWARDED is what was
        built, free of the switch — and the attempt id on every row when one was given.

Exit 0 = PASS (every check), 2 = FAIL (the report names each failing check). The report, the completions and the
ledger are what the re-reading test consumes; the runner commits them under
``contracts.NO_SWITCH_SMOKE_COMPLETIONS_DIR/<family>/`` and the fact then flips to ``certified`` (a NEW pin).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

DIAGNOSTIC = True
#: the arm this probe certifies — the ONE literal `--thinking` admits
ARM = C.THINKING_NOT_AVAILABLE
#: the split mode of every switch-less family (contracts.THINK_SPLIT_NONE): the content IS the answer
SPLIT_MODE = C.THINK_SPLIT_NONE
HEADER_PROBE_MESSAGES = ({"role": "user", "content": "Probe."},)
#: requests in flight at once through the ledger (the sbatch's --time formula is written for THIS number; the standup's
#: S at the smoke geometry is 44 for llama8b, so 8 draws decode together rather than queue)
DEFAULT_CONCURRENCY = 8
#: the identity version whose seed rule the smoke draws under (contracts.TINYCAP_IDENTITY_VERSION_2); the ONE value
#: `--tinycap-identity-version` admits — recorded, never hashed (no store)
IDENTITY_VERSION = 2


def item_seed(item_id: str, sample_idx: int = 0) -> int:
    """``contracts.SEED_RULE_INSPECT`` through the driver's ONE implementation (``inspect_served.sample_seed``)."""
    from sisteer.capeval.tiers import inspect_served as isv
    return isv.sample_seed(item_id, sample_idx)


def body_bytes(body: dict[str, Any]) -> bytes:
    """The EXACT bytes a request goes out as — hashed here and by the ledger (``usage_ledger.wire_body_sha256``)."""
    return json.dumps(body).encode("utf-8")


def _post_bytes(url: str, path: str, raw: bytes, *, token: str | None, headers: dict[str, str] | None = None,
                timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(f"{url.rstrip('/')}/{path.lstrip('/')}", data=raw,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        return {"error": {"status": e.code, "body": detail}}


def _post(url: str, path: str, body: dict[str, Any], *, token: str | None, headers: dict[str, str] | None = None,
          timeout: float) -> dict[str, Any]:
    return _post_bytes(url, path, body_bytes(body), token=token, headers=headers, timeout=timeout)


#: how the header was read (mirrors served_capability_run.HEADER_CONSTRUCTION_*, pinned equal by test)
HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF = "generation_prompt_diff"
HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING = "unconditional_trailing_header"


def served_no_switch_header(tokenize, vmodel: str) -> tuple[list[int], dict[str, Any]]:
    """The served generation header under the family's ONE render, by the construction the template's shape admits —
    the driver's ``_served_no_switch_header`` MIRRORED (pinned equal by test on the same fakes), which is
    ``chat_markers.derive_assistant_markers``' rule served through live ``/v1/tokenize`` with NO ``chat_template_kwargs``:
    the generation-prompt difference (agp True minus False) on a template that GATES its header on the flag; on a template
    that never consults the flag (True == False — the diff is EMPTY where the header is always present, chat_markers'
    FORBIDDEN comparison), the content-substitution probe: two assistant sentinels, the first differing token is the
    content start, the common suffix the closing scaffold; the header is unconditional iff the generation prompt ends
    EXACTLY at the content start and ENDS with the scaffold's trailing header (after its terminator), by tuple.
    ``tokenize(body) -> response`` is the transport; returns ``(header_ids, facts)``."""
    from sisteer import chat_markers as cm

    def ids(messages, agp: bool) -> list[int]:
        body = {"model": vmodel, "messages": [dict(m) for m in messages], "add_generation_prompt": bool(agp),
                **C.chat_template_kwargs(ARM)}
        assert C.CHAT_TEMPLATE_KWARGS_KEY not in body
        resp = tokenize(body)
        toks = resp.get("tokens") if isinstance(resp, dict) else None
        if not isinstance(toks, list) or not toks or not all(isinstance(t, int) and not isinstance(t, bool) for t in toks):
            raise RuntimeError(f"/v1/tokenize answered agp={agp} of {len(messages)} message(s) (no chat_template_kwargs) with no tokens: {resp!r}")
        return [int(t) for t in toks]

    r_true, r_false = ids(HEADER_PROBE_MESSAGES, True), ids(HEADER_PROBE_MESSAGES, False)
    facts: dict[str, Any] = {"render_len_by_agp": {"true": len(r_true), "false": len(r_false)}}
    if r_true[:len(r_false)] != r_false:
        raise RuntimeError("the add_generation_prompt=False render is not a token prefix of the True render — the served "
                           f"header cannot be read (agp {r_true[:12]}… vs {r_false[:12]}…)")
    if r_true != r_false:
        return r_true[len(r_false):], {**facts, "construction": HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF, "agp_invariant": False}
    probe = list(HEADER_PROBE_MESSAGES)
    fa = ids(probe + [{"role": "assistant", "content": cm._SENTINEL_A}], False)
    fb = ids(probe + [{"role": "assistant", "content": cm._SENTINEL_B}], False)
    if fa == fb:
        raise RuntimeError("the template ignores add_generation_prompt and substituting the assistant content changed nothing — "
                           "no assistant turn can be located")
    s = cm.divergence(fa, fb)
    q = 0
    while q < min(len(fa), len(fb)) - s and fa[len(fa) - 1 - q] == fb[len(fb) - 1 - q]:
        q += 1
    closing = fa[len(fa) - q:]
    facts.update(content_start=s, closing_scaffold=list(closing))
    if s != len(r_true) or fa[:s] != r_true:
        raise RuntimeError(f"the template ignores add_generation_prompt but its generation prompt ({len(r_true)} tokens) does not "
                           f"end where a completed assistant turn's content begins (token {s}) — a template that ignores the flag "
                           "WITHOUT appending the header unconditionally; the header cannot be read")
    if len(closing) < 2:
        raise RuntimeError(f"the closing scaffold {closing!r} carries nothing after its terminator — no unconditional trailing header")
    header = closing[1:]
    if r_true[len(r_true) - len(header):] != header:
        raise RuntimeError(f"the generation prompt does not END with the closing scaffold's trailing header {header!r} "
                           f"(it ends {r_true[-len(header):]!r})")
    return header, {**facts, "construction": HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING, "agp_invariant": True}


def header_ids_no_switch(proxy_url: str, vmodel: str, *, token: str | None) -> tuple[list[int], dict[str, Any]]:
    """:func:`served_no_switch_header` over the live proxy's ``/v1/tokenize`` (bearer ``token``)."""
    return served_no_switch_header(lambda body: _post(proxy_url, "tokenize", body, token=token, timeout=60), vmodel)


def shape_check(content: str, reasoning: str | None, *, finish_reason: str) -> dict[str, Any]:
    """The emitted shape under the ``none`` split (P19): the content IS the answer byte for byte, no block is ever
    present or unterminated, a think-tag literal is a STRAY (recorded, and a FAILURE here — the family has no
    channel), a reasoning channel is a stray too."""
    from sisteer.capeval.tiers.think_split import THINK_CLOSE, THINK_OPEN, split_think

    rec: dict[str, Any] = {"mode": SPLIT_MODE, "arm": ARM, "has_open": THINK_OPEN in content, "has_close": THINK_CLOSE in content,
                           "reasoning_content_present": bool(reasoning), "finish_reason": finish_reason}
    try:
        sp = split_think(content, mode=SPLIT_MODE)
        rec.update(present=sp.present, unterminated=sp.unterminated, stray_open_tag=sp.stray_open_tag,
                   stray_close_tag=sp.stray_close_tag, answer_chars=len(sp.answer_text),
                   answer_is_content=(sp.answer_text == content))
    except Exception as e:  # noqa: BLE001 — recorded as the row's failure, never raised past the row
        rec["split_error"] = f"{type(e).__name__}: {e}"
    rec["ok"] = ("split_error" not in rec and not rec["has_open"] and not rec["has_close"]
                 and not rec["reasoning_content_present"] and rec.get("present") is False
                 and rec.get("unterminated") is False and rec.get("answer_is_content") is True
                 and finish_reason in ("stop", "length"))
    rec["why"] = None if rec["ok"] else (rec.get("split_error")
                                        or ("a think-tag literal on a family with NO thinking channel (a stray)"
                                            if rec["has_open"] or rec["has_close"] else
                                            "a non-empty reasoning channel on a family with NO thinking channel"
                                            if rec["reasoning_content_present"] else
                                            f"finish_reason {finish_reason!r} is neither stop nor length"
                                            if finish_reason not in ("stop", "length") else
                                            "the none split did not return the content as the answer"))
    return rec


def certify_completion(family: str, header_ids: list[int], *, regime: object, content: str, reasoning: str | None,
                       finish_reason: str | None) -> dict[str, Any]:
    """PREDICATE (A″) on ONE completion through the contracts' seam (``no_switch_render_certified``): the header by
    tuple, the ``unspecified`` stamp, the tag-free channel-free completion. Returns the certification record — the
    seam's evidence with ``render_certified`` True, or its refusal's evidence with every problem named and
    ``error`` set (the row is written before any verdict)."""
    try:
        rec = C.no_switch_render_certified(family=family, header_ids=header_ids, measured_live=True, probe_regime=regime,
                                           probe_completion={"content": content, "reasoning": reasoning,
                                                             "finish_reason": finish_reason})
        return {**rec, "error": None}
    except C.RenderCertificationError as e:
        return {**dict(e.evidence), "error": str(e)}


def run_completions(url: str, plan: list[tuple[str, list[dict[str, str]]]], *, vmodel: str, cap: int,
                    params: dict[str, Any], token: str | None, model_key: str, family: str, sampling_mode: str,
                    task_label: str, concurrency: int, timeout_s: float, header_ids: list[int] | None = None
                    ) -> list[dict[str, Any]]:
    """(ii): the plan's requests through the ledger with at most ``concurrency`` in flight (a thread pool; every
    request keeps its own ``timeout_s`` — the 7,200 s upstream timeout — so one slow cap-bound draw holds one slot,
    not the wall). Every body is FREE of ``chat_template_kwargs`` (asserted before it is sent) and carries a minted
    ``X-Tinycap-Request-Id`` the ledger writes back beside the sha256 of the exact bytes it forwarded. Returned in
    ONE deterministic order — item id (a total order: one arm, one draw per item) — whatever order the server
    answered in. A transport error propagates as it did serially; the requests not yet started are cancelled."""
    if concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {concurrency}")
    t_batch = time.time()

    def one(item_id: str, msgs: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": vmodel, "messages": msgs, "max_tokens": int(cap), "seed": item_seed(item_id, 0),
                **C.chat_template_kwargs(ARM), **params}
        assert C.CHAT_TEMPLATE_KWARGS_KEY not in body and C.ENABLE_THINKING_KEY not in body
        raw = body_bytes(body)
        request_id = f"smoke-{uuid.uuid4().hex[:16]}"
        hdrs = {C.HEADER_TASK: task_label, C.HEADER_SAMPLE: f"{item_id}/0", C.HEADER_REQUEST_ID: request_id}
        t0 = time.time()
        resp = _post_bytes(url, "chat/completions", raw, token=token, headers=hdrs, timeout=timeout_s)
        t1 = time.time()
        choice = ((resp.get("choices") or [{}])[0]) if isinstance(resp, dict) and "error" not in resp else {}
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        stamps = resp.get("petri") if isinstance(resp, dict) else None
        row = {"diagnostic": DIAGNOSTIC, "item_id": item_id, "arm": ARM, "model_key": model_key, "family": family,
               "split_mode": SPLIT_MODE, "cert_class": C.CERT_CLASS_NO_SWITCH, "cap": int(cap), "seed": body["seed"],
               "sampling_mode": sampling_mode, "sampling_params": params,
               "request_id": request_id, "body_sha256": hashlib.sha256(raw).hexdigest(),
               "body_has_chat_template_kwargs": C.CHAT_TEMPLATE_KWARGS_KEY in body,
               "finish_reason": choice.get("finish_reason"), "usage": resp.get("usage") if isinstance(resp, dict) else None,
               "stamps": stamps, "content": content, "reasoning_content": reasoning, "latency_s": round(t1 - t0, 2),
               "t_start_s": round(t0 - t_batch, 2), "t_end_s": round(t1 - t_batch, 2),
               "error": resp.get("error") if isinstance(resp, dict) else {"raw": str(resp)[:300]}}
        row["shape"] = (shape_check(content, reasoning, finish_reason=str(choice.get("finish_reason")))
                        if not row["error"] else None)
        row["certification"] = (certify_completion(family, list(header_ids), regime=(stamps or {}).get("render_regime"),
                                                   content=content, reasoning=reasoning, finish_reason=choice.get("finish_reason"))
                                if not row["error"] and header_ids is not None else None)
        return row

    ex = cf.ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(plan))))
    try:
        rows = list(ex.map(lambda p: one(*p), plan))
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return sorted(rows, key=lambda r: r["item_id"])


def dump_completions(path: Path, rows: list[dict[str, Any]]) -> None:
    """``smoke_completions.jsonl``, one row per line in the order ``run_completions`` returned them."""
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    rows.append({"_unparsable": line[:100]})
    return rows


def ledger_reconciliation(lrows: list[dict[str, Any]], rows: list[dict[str, Any]], *, attempt_id: str | None
                          ) -> dict[str, Any]:
    """(iii): the ledger against the probe's own records — by ``request_id``: every sent request has exactly one row,
    its ``body_sha256`` equals the sha256 of the bytes the probe sent, ``thinking`` is NULL on every row, no error, and
    the attempt id (when given) rides every row."""
    by_rid = {r.get("request_id"): r for r in lrows if r.get("request_id")}
    sent = {r["request_id"]: r for r in rows}
    missing = sorted(rid for rid in sent if rid not in by_rid)
    hash_mismatch = sorted(rid for rid, r in sent.items() if rid in by_rid and by_rid[rid].get("body_sha256") != r["body_sha256"])
    thinking_values = sorted({str(r.get("thinking")) for r in lrows})
    errors = [r for r in lrows if r.get("error_code") is not None or r.get("error_class") not in (None, "none")]
    attempt_ok = attempt_id is None or all(r.get(C.LEDGER_ATTEMPT_ID_FIELD) == attempt_id for r in lrows)
    ok = (len(lrows) == len(rows) and not missing and not hash_mismatch and not errors
          and thinking_values == ["None"] and attempt_ok and len(by_rid) == len(lrows))
    return {"ok": ok, "n_rows": len(lrows), "n_sent": len(rows), "n_missing_request_ids": len(missing),
            "first_missing": missing[:3], "n_body_sha256_mismatch": len(hash_mismatch), "first_mismatch": hash_mismatch[:3],
            "n_errors": len(errors), "error_codes": sorted({str(r.get("error_code")) for r in lrows}),
            "error_classes": sorted({str(r.get("error_class")) for r in lrows}),
            "thinking_values": thinking_values, "attempt_id": attempt_id, "attempt_id_on_every_row": attempt_ok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ledger-url", required=True, help="the usage ledger's /v1 base (the requests go HERE)")
    ap.add_argument("--proxy-url", required=True, help="the standup's proxy /v1 base (/v1/tokenize; the ledger's upstream)")
    ap.add_argument("--proxy-token", default=None)
    ap.add_argument("--routing-table", required=True)
    ap.add_argument("--endpoint-json", required=True)
    ap.add_argument("--ledger-path", required=True, help="the ledger's usage_ledger.jsonl (read back)")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--family", required=True, choices=sorted(C.NO_THINKING_CHANNEL_FAMILIES),
                    help="the switch-less family (contracts.NO_THINKING_CHANNEL_FAMILIES) — a switched family is refused")
    ap.add_argument(C.FLAG_THINKING, dest="thinking", required=True, choices=[ARM],
                    help="the arm: the no-switch literal alone (the ON / OFF arms are the MATH-500 smoke's)")
    ap.add_argument(C.FLAG_DIAGNOSTIC, dest="diagnostic", action="store_true", required=True,
                    help="REQUIRED: the argv's own statement that this run is diagnostic (every record says so)")
    ap.add_argument("--sampling-mode", default=C.SAMPLING_CARD, choices=[C.SAMPLING_CARD],
                    help="the model's frozen `card` row (P6) — the one mode a no-switch row runs")
    ap.add_argument("--sampling-params", default="{}", help="JSON: the row's request parameters (g8k_cells.py)")
    ap.add_argument("--expect-stamps", default=None,
                    help="the row's expected stamp triple `cache_salt_regime,resumed_prefill_policy,render_regime` (g8k_cells.py)")
    ap.add_argument("--n", type=int, default=24, help="paired items (the first N in the served task's plan order)")
    ap.add_argument("--cap", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="completions in flight at once through the ledger (>= 1; the sbatch's --time formula is written for the default)")
    ap.add_argument(C.FLAG_ATTEMPT_ID, dest="attempt_id", default=None, help="the attempt the ledger stamps (read back on every row)")
    ap.add_argument(C.FLAG_TINYCAP_IDENTITY_VERSION, dest="identity_version", type=int, default=IDENTITY_VERSION, choices=[IDENTITY_VERSION],
                    help="the identity version whose seed rule the draws use (recorded; no store is written)")
    ap.add_argument("--task-label", default="smoke/gsm8k", help="the ledger's X-Tinycap-Task on every smoke request")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=str(REPO))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, **detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", **detail})
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:400]}", flush=True)

    fact = C.render_fact_for(a.family)
    assert fact is not None and fact.thinking_switch is False and fact.split_mode == SPLIT_MODE, a.family
    from sisteer.capeval.serving.proxy import load_routing_table
    table = load_routing_table(Path(a.routing_table))
    base = [r for r in table["routes"] if r.cell_id == "none"]
    if not base:
        check("base_route", False, why="no base route in the routing table")
    vmodel = base[0].virtual_model if base else "petri__base"
    ep = json.loads(Path(a.endpoint_json).read_text())
    check("upstream_timeout", ep.get(C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S) == C.PROXY_UPSTREAM_TIMEOUT_S,
          recorded=ep.get(C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S), want=C.PROXY_UPSTREAM_TIMEOUT_S)
    expect_stamps = None
    if a.expect_stamps:
        parts = a.expect_stamps.split(",")
        if len(parts) != 3:
            raise SystemExit(f"--expect-stamps wants `salt,policy,regime`, got {a.expect_stamps!r}")
        expect_stamps = {"cache_salt_regime": parts[0], "resumed_prefill_policy": parts[1], "render_regime": parts[2]}
        check("expected_regime_is_the_arms", expect_stamps["render_regime"] == C.RENDER_REGIME_BY_ARM[ARM],
              expected=expect_stamps["render_regime"], arms=C.RENDER_REGIME_BY_ARM[ARM])

    # (i) predicate (A″) rule 1: the ONE render's header, live, no kwargs, by TUPLE against the fact
    header: dict[str, Any] = {"want": list(fact.off_ids), "fact_ids_invariant": tuple(fact.off_ids) == tuple(fact.on_ids)}
    try:
        got, hfacts = header_ids_no_switch(a.proxy_url, vmodel, token=a.proxy_token)
        header.update(got=got, error=None, **hfacts)
    except Exception as e:  # noqa: BLE001
        got = None
        header.update(got=None, error=f"{type(e).__name__}: {e}", construction=None, agp_invariant=None)
    header["certification"] = (certify_completion(a.family, got, regime=None, content="", reasoning=None, finish_reason=None)
                               if got is not None else None)
    header_ok = got is not None and header["certification"]["error"] is None and header["fact_ids_invariant"]
    check("header_no_switch", header_ok, got=got, want=header["want"], error=header["error"],
          construction=header.get("construction"), agp_invariant=header.get("agp_invariant"),
          render_len_by_agp=header.get("render_len_by_agp"), content_start=header.get("content_start"),
          problems=(header["certification"] or {}).get("certification_problems"))

    # (ii) the items, through the ledger: the served task's own ids and renders
    from sisteer.capeval.tiers import inspect_served as isv
    plan_off = isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, repo_root=Path(a.repo_root))
    ids = list(plan_off.sample_ids or [])
    messages = isv.task_messages(C.GSM8K_PAIRED_TASK_NAME, repo_root=Path(a.repo_root))
    check("items_planned", len(ids) == C.GSM8K_PAIRED_N == len(messages), n_ids=len(ids), n_messages=len(messages),
          want=C.GSM8K_PAIRED_N)
    params = json.loads(a.sampling_params or "{}")
    n = min(a.n, len(ids))
    plan = [(ids[i], messages[i]) for i in range(n)]
    rows = run_completions(a.ledger_url, plan, vmodel=vmodel, cap=int(a.cap), params=params, token=a.proxy_token,
                           model_key=a.model_key, family=a.family, sampling_mode=a.sampling_mode, task_label=a.task_label,
                           concurrency=a.concurrency, timeout_s=float(C.PROXY_UPSTREAM_TIMEOUT_S), header_ids=got)
    dump_completions(out / "smoke_completions.jsonl", rows)
    errors = [r for r in rows if r["error"]]
    n_at_cap = sum(1 for r in rows if r["finish_reason"] == "length")
    check("requests_served", len(rows) == n and not errors, n=len(rows), want=n, n_errors=len(errors),
          first_error=(errors[0]["error"] if errors else None))
    check("wire_bodies_free_of_switch", bool(rows) and not any(r["body_has_chat_template_kwargs"] for r in rows),
          n_bodies=len(rows), n_with_chat_template_kwargs=sum(1 for r in rows if r["body_has_chat_template_kwargs"]))
    regimes = sorted({str((r["stamps"] or {}).get("render_regime")) for r in rows})
    check("render_regime_stamps", regimes == [C.RENDER_REGIME_BY_ARM[ARM]], regimes=regimes, want=C.RENDER_REGIME_BY_ARM[ARM])
    if expect_stamps is not None:
        for field in ("cache_salt_regime", "resumed_prefill_policy"):
            seen = sorted({str((r["stamps"] or {}).get(field)) for r in rows})
            check(f"stamp_{field}", seen == [expect_stamps[field]], seen=seen, want=expect_stamps[field])
    cells = sorted({str((r["stamps"] or {}).get("cell_id")) for r in rows})
    check("base_route_cell", cells == ["none"], cells=cells)
    bad = [r for r in rows if r["shape"] is None or not r["shape"]["ok"]]
    check("emitted_shape_none_split", not bad, n_bad=len(bad),
          first_bad=({k: bad[0]["shape"].get(k) for k in ("why", "has_open", "has_close", "reasoning_content_present", "finish_reason")}
                     if bad and bad[0]["shape"] else (bad[0]["error"] if bad else None)), n_at_cap=n_at_cap)
    uncert = [r for r in rows if r["certification"] is None or r["certification"]["error"] is not None]
    check("per_completion_render_certified", bool(rows) and not uncert, n_certified=len(rows) - len(uncert), n=len(rows),
          first_problems=((uncert[0]["certification"] or {}).get("certification_problems") if uncert else None))
    # (iii) the ledger read back, reconciled by request id
    lrows = read_ledger(Path(a.ledger_path))
    rec = ledger_reconciliation(lrows, rows, attempt_id=a.attempt_id)
    check("ledger_rows", rec["ok"], **{k: v for k, v in rec.items() if k != "ok"})
    report = {"diagnostic": DIAGNOSTIC, "model_key": a.model_key, "family": a.family, "thinking": ARM, "split_mode": SPLIT_MODE,
              "cert_class": C.CERT_CLASS_NO_SWITCH, "vmodel": vmodel, "cap": int(a.cap), "n": n, "concurrency": a.concurrency,
              "sampling_mode": a.sampling_mode, "sampling_params": params,
              "tinycap_identity_version": a.identity_version, "seed_rule": C.SEED_RULE_INSPECT, "attempt_id": a.attempt_id,
              "task_label": a.task_label, "expected_stamps": expect_stamps,
              "chat_template_hash": ep.get("chat_template_hash"), "header": header, "checks": checks,
              "n_at_cap": n_at_cap,
              "completion_tokens": [((r["usage"] or {}).get("completion_tokens")) for r in rows],
              "ledger": rec,
              "verdict": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
              "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[smoke] {report['verdict']} — {sum(c['status'] == 'FAIL' for c in checks)} failing check(s); "
          f"report {out / 'smoke_report.json'}", flush=True)
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
