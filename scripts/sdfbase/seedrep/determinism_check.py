#!/usr/bin/env python3
"""The Qwen seed-replicate battery — the determinism spot-check. Runs IN-CONTAINER
before teardown, after the leg's traffic has drained.

    determinism_check.py --root <attempt root> --leg <table leg> [--arm none] [--extra-arm <cell>] [--n 20] [--item-budget <N>]
        [--proxy-url <RoutingProxy /v1 base>] [--metrics-url <the proxy's /metrics pass-through>]   ($VLLM_API_KEY = the proxy bearer, env only)
        [--tokenize-url <proxy /v1/tokenize>] --out <root>/determinism.json [--copy-to audits/determinism/<leg>.json]
        [--allow-busy] [--petri-behaviours b1,b2]

Per arm (`none` by default; `--extra-arm` adds the mix PSR cell on static): the 20 lowest-`sha256(item_key)` filed
items of the leg's store are re-issued with the BYTE-IDENTICAL request bodies their filed draws were sent with (read
from the request-evidence RECORD's additive `body_b64` — the exact bytes the client put on the wire, beside the
`body_sha256` / `body_bytes` (a length) the recorder writes — and held to that digest and length before anything is
sent; a record without the bytes REFUSES the replay of its draw: a body is never reconstructed and no side file is read), against the SAME standup (`endpoint.json.proxy_url`, the RoutingProxy — the
replay bypasses the usage ledger by design, so it adds no attempt the audit would have to reconcile). Static / BoN /
controls / GSM8K / misalign replay EVERY draw of the selected items; multi-turn / Petri replay the FIRST target turn
(`sample_idx 0`) of 20 goals / scenarios. Texts are compared byte-for-byte with the ORIGINAL — the stored row's text
where the store holds it, else the `text_sha256` `response_evidence.jsonl` recorded for the filed attempt (Inspect
legs) — and every miss records its first divergence (character index, and token index through the proxy's
`/v1/tokenize`; when no tokenize URL is given the token basis is WHITESPACE and says so — never a silent null).

Refuses (exit 2) when: the traffic has not drained (`num_requests_running > 0` unless `--allow-busy`, which is
recorded), a selected draw's body bytes or original text are unavailable, a body's sha256 is not its record's, or
the store / evidence / endpoint is missing. The record: `{leg, standup_id, view, stage, shard, smoke_id (null on a production row),
arms, n_items, n_planned, n_selected, budget_limited, item_budget, n_requests, identical, misses: [...], engine_batch_state, replay_target,
token_basis, W, attempt_id, S, L}`. The attempt's
standup id is read through `audit_common.split_standup_id`, the ONE parser of both the production and the `smoke-<S>__…` id forms (F-U). A non-reproducing engine is a RESULT
(reported in the results doc), not a refusal — exit 0 with `identical < n_requests`.

The selection count: the plan selects `N_PLANNED` (20) items per arm and REFUSES a store that files fewer — on a
PRODUCTION row that refusal is a real defect signal (a short store) and stays. A smoke's BoN / misalign row runs under
the smoke-only ITEM budget (`--limit-goals 10` / `--limit-items 10`), so its store files 10 items by construction and a
20-item selection could never pass; for such a BUDGET-LIMITED row the served app derives `--n min(N_PLANNED, <the row's item
budget>)` from the row's `smoke.budget` and names the budget with `--item-budget <N>` — the check NEVER reads the store's size
to shrink its own selection (that would hide the production defect). The record says which happened: `n_planned` (always 20),
`n_selected` (the `--n` actually selected), `budget_limited` (true iff `--item-budget` was given) and `item_budget`.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_common as ac  # noqa: E402
from audit_common import C, AttemptRoot, StoreRead, Draw, refuse  # noqa: E402
from sisteer.capeval.serving.vllm_auth import TOKEN_ENV as PROXY_BEARER_ENV  # noqa: E402  (VLLM_API_KEY: the serving stack's one bearer name)

CHAT_PATH = "/chat/completions"
#: injection points for the tests (a fake POST / metrics / tokenize); production leaves them None
POST_FN: Optional[Callable[[bytes, Mapping[str, str]], Dict[str, Any]]] = None
METRICS_FN: Optional[Callable[[], Dict[str, Any]]] = None
TOKENIZE_FN: Optional[Callable[[str], List[int]]] = None
#: legs whose replay is the first target turn only (Petri and multi-turn)
FIRST_TURN_LEGS: Tuple[str, ...] = (*ac.MULTITURN_ATTACKS, "petri")
#: the items the plan selects per arm (§9.2: "the 20 lowest-sha256 items") -- `--n`'s default and the record's `n_planned`; MIRRORED by
#: `seedrep_launch_plan.DETERMINISM_N_PLANNED` (the audit imports no plan module; a test pins the two equal)
N_PLANNED = 20


def item_rank_key(item_id: str) -> str:
    """The selection key: sha256 of the item key, lowest first."""
    return hashlib.sha256(str(item_id).encode("utf-8")).hexdigest()


def select_items(store: StoreRead, n: int) -> List[str]:
    items = sorted({i for _, i in store.filed}, key=item_rank_key)
    if len(items) < n:
        refuse(f"{store.store_dir}: the store files {len(items)} item(s), fewer than the {n} the spot-check selects")
    return items[:n]


def select_draws(store: StoreRead, items: Sequence[str]) -> List[Draw]:
    want = set(items)
    draws = [d for d in store.draws if d.item_id in want]
    if store.leg in FIRST_TURN_LEGS:
        # the first target turn of each item: the LOWEST turn index it has (Petri turns are 1-based per the recorder's
        # `target_turns`, multi-turn `order`s 0-based — the index convention is the writer's, never assumed to be 0)
        first: Dict[str, int] = {}
        for d in draws:
            first[d.item_id] = min(first.get(d.item_id, d.sample_idx), d.sample_idx)
        draws = [d for d in draws if d.sample_idx == first[d.item_id]]
    if not draws:
        refuse(f"{store.store_dir}: no draw of the selected items (leg {store.leg})")
    return sorted(draws, key=lambda d: (item_rank_key(d.item_id), d.sample_idx))


def bodies_from_records(records: Sequence[Mapping[str, Any]], *, where: Path) -> Dict[str, bytes]:
    """request_id → the exact wire bytes a request-evidence record retains under `body_b64` (:data:`audit_common.
    REQUEST_BODY_B64_FIELD` — the ADDITIVE field beside the recorder's `body_sha256` / `body_bytes`). A retained body must
    hash to the record's own digest and have its recorded length; a record without the field is simply absent here and the
    replay of its draw refuses by name (consume the producer's record, never a side file nobody writes)."""
    out: Dict[str, bytes] = {}
    for rec in records:
        rid, b64 = rec.get("request_id"), rec.get(ac.REQUEST_BODY_B64_FIELD)
        if not rid or b64 is None:
            continue
        if not isinstance(b64, str):
            refuse(f"{where}: record {rid!r}: {ac.REQUEST_BODY_B64_FIELD} is {type(b64).__name__}, not a base64 string")
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            refuse(f"{where}: record {rid!r}: {ac.REQUEST_BODY_B64_FIELD} is not base64 ({e})")
        if C.wire_body_sha256(raw) != rec.get("body_sha256"):
            refuse(f"{where}: retained body of {rid!r} hashes to {C.wire_body_sha256(raw)} but the record says "
                   f"{rec.get('body_sha256')!r} — not the wire bytes")
        n = rec.get("body_bytes")
        if isinstance(n, int) and not isinstance(n, bool) and n != len(raw):
            refuse(f"{where}: retained body of {rid!r} is {len(raw)} bytes but the record says body_bytes {n} — not the wire bytes")
        out[str(rid)] = raw
    return out


def _http_json(url: str, *, body: Optional[bytes] = None, headers: Optional[Mapping[str, str]] = None,
               timeout_s: float = 7200.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=body, headers=dict(headers or {}), method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        refuse(f"{url}: HTTP {e.code} {e.read()[:200]!r}")
    except urllib.error.URLError as e:
        refuse(f"{url}: {e}")
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        refuse(f"{url}: response is not JSON ({raw[:120]!r})")


def _parse_prometheus(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        base = name.split("{", 1)[0]
        try:
            out[base] = out.get(base, 0.0) + float(rest.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def engine_state(metrics_url: Optional[str], api_key: Optional[str] = None) -> Dict[str, Any]:
    """The engine's request gauges off `--metrics-url`. F-O (2026-09-18): that URL is the ROUTING PROXY's `/metrics`
    pass-through (`proxy.RoutingProxy.handle_metrics`), never the engine's own page -- the engine guards `/metrics` with
    its per-launch bearer (`vllm_auth.SisteerTokenAuthMiddleware`), which this check never holds; the proxy forwards with
    it and answers the body verbatim. The proxy bearer (`$VLLM_API_KEY`, when the standup names one) rides along the way it
    does on the replay, so a token-guarded proxy answers this read too."""
    if METRICS_FN is not None:
        return dict(METRICS_FN())
    if metrics_url is None:
        refuse("--metrics-url is required (the proxy's /metrics pass-through) unless a metrics hook is injected — the drained state must be READ, not assumed")
    req = urllib.request.Request(metrics_url, headers=({"Authorization": f"Bearer {api_key}"} if api_key else {}))
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        refuse(f"{metrics_url}: {e}")
    m = _parse_prometheus(text)
    if "vllm:num_requests_running" not in m:
        refuse(f"{metrics_url}: no vllm:num_requests_running gauge")
    return {"num_requests_running": m.get("vllm:num_requests_running"),
            "num_requests_waiting": m.get("vllm:num_requests_waiting"),
            "generation_tokens_total": m.get("vllm:generation_tokens_total")}


def replay(raw: bytes, *, proxy_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if POST_FN is not None:
        return dict(POST_FN(raw, headers))
    return _http_json(proxy_url.rstrip("/") + CHAT_PATH, body=raw, headers=headers)


def completion_text(resp: Mapping[str, Any]) -> str:
    """`choices[0].message.content` (the text the stores persist and `response_evidence.jsonl` hashes)."""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        refuse(f"replay response has no choices: {str(resp)[:200]}")
    msg = choices[0].get("message")
    content = msg.get("content") if isinstance(msg, Mapping) else None
    if content is None:
        content = ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, Mapping) and p.get("type") == "text")
    if not isinstance(content, str):
        refuse(f"replay response content is {type(content).__name__}, not text")
    return content


def tokens_of(text: str, tokenize_url: Optional[str], api_key: Optional[str]) -> Tuple[List[Any], str]:
    if TOKENIZE_FN is not None:
        return list(TOKENIZE_FN(text)), "served_tokenizer"
    if tokenize_url is None:
        return text.split(), "whitespace"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps({"prompt": text}).encode("utf-8")
    out = _http_json(tokenize_url, body=body, headers=headers, timeout_s=120.0)
    toks = out.get("tokens")
    if not isinstance(toks, list):
        refuse(f"{tokenize_url}: response carries no tokens list")
    return toks, "served_tokenizer"


def first_divergence(a: Sequence[Any], b: Sequence[Any]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def run_check(root: AttemptRoot, *, leg: str, arms: Sequence[str], n: int, proxy_url: Optional[str], api_key: Optional[str],
              metrics_url: Optional[str], tokenize_url: Optional[str], allow_busy: bool,
              petri_behaviours: Sequence[str], item_budget: Optional[int] = None) -> Dict[str, Any]:
    """``item_budget``: the row's smoke-only ITEM budget when the row is budget-limited (the served app derives both it and ``n`` =
    min(N_PLANNED, budget) from the row's `smoke.budget`; None on a production row). It is RECORDED (`budget_limited` / `item_budget`), never
    used to shrink the selection here: ``n`` is what is selected, and a store filing fewer than ``n`` refuses exactly as before."""
    if item_budget is not None and n > item_budget:
        refuse(f"--n {n} exceeds the row's item budget {item_budget} (D122: a budget-limited row selects min({N_PLANNED}, budget) -- the served app "
               "derives --n from the row's smoke.budget; the check never shrinks its own selection)")
    standup = ac.read_standup(root)
    target = proxy_url or standup.endpoint.get("proxy_url")
    if not target and POST_FN is None:
        refuse(f"{standup.endpoint_path}: no proxy_url and no --proxy-url — nothing to replay against")
    records = ac.read_request_evidence(root.path)
    by_rid = {str(r.get("request_id")): r for r in records if r.get("request_id")}
    bodies = bodies_from_records(records, where=root.path / ac.EVIDENCE_FILE)
    resp_shas = ac.response_text_shas_by_rid(root.path)
    state_before = engine_state(metrics_url, api_key)
    running = state_before.get("num_requests_running")
    if running is None:
        refuse("engine state carries no num_requests_running")
    if float(running) > 0 and not allow_busy:
        refuse(f"the engine still has {running} request(s) running — the spot-check replays against a DRAINED engine "
               "(pass --allow-busy to record the replay under load instead)")
    stores = ac.discover_stores(root, leg, petri_behaviours=petri_behaviours)
    by_arm = {s.arm: s for s in stores}
    per_arm: List[Dict[str, Any]] = []
    misses: List[Dict[str, Any]] = []
    n_requests = 0
    n_identical = 0
    token_basis: Optional[str] = None
    for arm in arms:
        store = by_arm.get(arm)
        if store is None:
            refuse(f"{root.label}: no {leg} store for arm {arm!r} (arms present: {sorted(by_arm)})")
        items = select_items(store, n)
        draws = select_draws(store, items)
        plan: List[Tuple[Draw, bytes, Optional[str], Optional[str]]] = []
        for d in draws:
            if not d.request_id:
                refuse(f"{store.store_dir}: draw {d.task} {d.item_id}/{d.sample_idx} carries no request_id — its body cannot be found")
            rec = by_rid.get(d.request_id)
            if rec is None:
                refuse(f"{store.store_dir}: request_id {d.request_id!r} has no request-evidence record")
            raw = bodies.get(d.request_id)
            if raw is None:
                refuse(f"{root.path}: the request-evidence record of {d.request_id!r} retains no wire bytes ({ac.REQUEST_BODY_B64_FIELD}; it "
                       "records body_sha256 and body_bytes, a digest and a length) — criterion 3's byte-identical replay needs the bytes "
                       "the client put on the wire, so the replay REFUSES rather than reconstructing a body")
            orig_text = d.text
            orig_sha = d.text_sha256 or resp_shas.get(d.request_id)
            if orig_text is None and orig_sha is None:
                refuse(f"{store.store_dir}: no original text (store row) and no {ac.RESPONSE_TEXT_SHA_FIELD} on the response record of "
                       f"attempt {d.request_id!r} ({ac.RESPONSE_EVIDENCE_FILE}; WS1-D′'s additive response hash) for {d.task} "
                       f"{d.item_id}/{d.sample_idx} — nothing to compare the replay against; never invented")
            plan.append((d, raw, orig_text, orig_sha))
        arm_identical = 0
        for d, raw, orig_text, orig_sha in plan:
            resp = replay(raw, proxy_url=str(target), api_key=api_key)
            text = completion_text(resp)
            sha = ac.text_sha256(text)
            n_requests += 1
            if sha == orig_sha or (orig_text is not None and text == orig_text):
                arm_identical += 1
                n_identical += 1
                continue
            miss: Dict[str, Any] = {"arm": arm, "task": d.task, "item_id": d.item_id, "sample_idx": d.sample_idx,
                                    "request_id": d.request_id, "original_sha256": orig_sha, "replay_sha256": sha,
                                    "replay_completion_id": resp.get("id"),
                                    "replay_finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason")}
            if orig_text is not None:
                miss["first_divergence_char"] = first_divergence(orig_text, text)
                ta, basis = tokens_of(orig_text, tokenize_url, api_key)
                tb, _ = tokens_of(text, tokenize_url, api_key)
                miss["first_divergence_token"] = first_divergence(ta, tb)
                miss["token_basis"] = basis
                token_basis = basis
            else:
                miss["first_divergence_char"] = None
                miss["first_divergence_token"] = None
                miss["token_basis"] = "unavailable: the original text is known by sha256 only (Inspect leg)"
            misses.append(miss)
        per_arm.append({"arm": arm, "vmodel": standup.vmodel_of(arm), "items": items, "n_requests": len(plan),
                        "identical": arm_identical, "store_dir": str(store.store_dir)})
    state_after = engine_state(metrics_url, api_key)
    # F-U: the row this record is OF, spelled the plan's way (`smoke-<S>__…` for a smoke row, whose `smoke_id` is recorded; null for
    # production) beside its axes — a smoke row's record used to be refused at the id parse and never written
    return {"leg": leg, "standup_id": root.standup_id, "view": root.view, "stage": root.stage, "shard": root.shard, "smoke_id": root.smoke_id,
            "arms": per_arm, "n_items": n, "n_requests": n_requests, "n": n_requests, "identical": n_identical,
            # the selection the plan calls for vs the one made -- `budget_limited` is true ONLY when the served app named the row's
            # smoke item budget (`--item-budget`), never inferred from the store's size or from n < N_PLANNED
            "n_planned": N_PLANNED, "n_selected": n, "budget_limited": item_budget is not None, "item_budget": item_budget,
            "reproduces": n_identical == n_requests, "misses": misses,
            "engine_batch_state": {"before": state_before, "after": state_after, "drained": float(running) == 0,
                                   "allow_busy": allow_busy},
            "replay_target": str(target), "replay_bypasses_ledger": True,
            "token_basis": token_basis or ("served_tokenizer" if (tokenize_url or TOKENIZE_FN) else "whitespace"),
            "first_turn_only": leg in FIRST_TURN_LEGS, "W": root.W, "attempt_id": root.attempt_id,
            "S": standup.S, "L": standup.L, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # the standup record's provenance (F-AA / F-AE): which endpoint file S / L / vmodels were read from, and whether the
            # standup wrote `standup_jobs.jsonl` or the standup was derived from the attested records
            "standup_record": standup.record_source, "endpoint_file": standup.endpoint_path.name, "standup_provenance": dict(standup.provenance)}


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--leg", required=True, choices=ac.LEGS)
    ap.add_argument("--arm", default="none")
    ap.add_argument("--extra-arm", default=None, help="a second arm to spot-check (the mix PSR cell on static)")
    ap.add_argument("--n", type=int, default=N_PLANNED, help=f"items selected per arm (default {N_PLANNED}, the plan's; a budget-limited smoke row's "
                    "served app passes min(20, the row's item budget) -- D122)")
    ap.add_argument("--item-budget", type=int, default=None,
                    help="D122: the row's smoke-only ITEM budget (D107 `--limit-goals` / `--limit-items`) when the row is budget-limited; recorded as "
                         "`budget_limited: true` + `item_budget`. Never given on a production row (its short store must still refuse).")
    ap.add_argument("--proxy-url", default=None, help="RoutingProxy /v1 base (default: endpoint.json.proxy_url)")
    # the proxy bearer is READ FROM THE ENVIRONMENT (``$VLLM_API_KEY``, the launch's per-server token the serving stack
    # names once, vllm_auth.TOKEN_ENV — exactly how seedrep_petri_drive.py reads a pool bearer); never a key FILE flag:
    # the repo's vault rule (tests/test_no_credential_files.py) forbids a flag that takes a credential file
    ap.add_argument("--metrics-url", default=None, help="the proxy's /metrics pass-through of the engine's page (drained-state read)")
    ap.add_argument("--tokenize-url", default=None, help="the proxy's /v1/tokenize (token divergence index)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy-to", type=Path, default=None, help="audits/determinism/<leg>.json")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--petri-behaviours", default="")
    args = ap.parse_args(argv)
    if args.n < 1:
        refuse("--n must be ≥ 1")
    if args.item_budget is not None and args.item_budget < 1:
        refuse("--item-budget must be ≥ 1 (the row's smoke-only item budget; omit it on a production row)")
    root = ac.read_attempt_root(args.root)
    api_key = os.environ.get(PROXY_BEARER_ENV, "").strip() or None
    arms = [args.arm] + ([args.extra_arm] if args.extra_arm else [])
    petri_behaviours = [b.strip() for b in args.petri_behaviours.split(",") if b.strip()]
    rec = run_check(root, leg=args.leg, arms=arms, n=args.n, proxy_url=args.proxy_url, api_key=api_key,
                    metrics_url=args.metrics_url, tokenize_url=args.tokenize_url, allow_busy=args.allow_busy,
                    petri_behaviours=petri_behaviours, item_budget=args.item_budget)
    ac.write_json(args.out, rec)
    if args.copy_to is not None:
        ac.write_json(args.copy_to, rec)
    budget_note = f"; budget-limited: {rec['n_selected']} of {rec['n_planned']} planned items (item budget {rec['item_budget']}, D122)" if rec["budget_limited"] else ""
    print(f"[determinism] {args.leg}: {rec['identical']}/{rec['n_requests']} byte-identical "
          f"({'reproduces' if rec['reproduces'] else 'DOES NOT reproduce'}{budget_note}) → {args.out}")
    return ac.EXIT_OK


if __name__ == "__main__":
    sys.exit(ac.run_main(main))
