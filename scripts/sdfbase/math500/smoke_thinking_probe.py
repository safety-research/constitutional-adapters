#!/usr/bin/env python3
"""The DIAGNOSTIC smoke's CPU-side probe for math500-thinking-on-full22: runs from
the client venv against a live base-route standup THROUGH the usage ledger, and writes records that CANNOT be
read as production results — every one carries ``diagnostic: true``, the out root is ``diagnostic/<family>/``,
no tinycap store is written.

  (i)   predicate (A) on the base route: the served header ids through live ``/v1/tokenize`` under
        ``enable_thinking`` true AND false (explicit JSON booleans, never omitted — "assume ON" is the failure
        mode) must equal the family's render fact by TUPLE (``contracts.FAMILY_RENDER_FACTS[family]`` — the
        table; ``--expect-off-ids`` / ``--expect-on-ids`` name the tuples explicitly when the table is not on the
        tree; a probe with NO expectation refuses — it must never pass vacuously);
  (ii)  N_ON MATH-500 items (the first ids in dataset order, rendered through the task's own solver —
        ``tinycap_oos_prompt_budget.math500_messages``) on the base route with ``enable_thinking: true`` and N_OFF
        with ``false``, at the smoke cap, under the model's sampling row and the seed rule
        (``contracts.SEED_RULE_INSPECT``); per row: the proxy's stamps (``render_regime`` / ``cache_salt_regime`` /
        ``resumed_prefill_policy`` from the response's ``petri`` block), ``finish_reason``, usage, the raw
        completion (+ ``reasoning_content`` where the server parsed) -> ``smoke_completions.jsonl``. The requests go
        out with BOUNDED concurrency (``--concurrency``, default ``DEFAULT_CONCURRENCY`` in flight; every request
        keeps the 7,200 s upstream timeout of its own): serially, 28 cap-bound thinking draws did not reliably
        fit the smoke's wall, and the worst case was 28 x 7,200 s inside it. The dump is written once every request
        has returned, in ONE deterministic order — (item id, arm) — whatever order the server answered in. What
        stays sequential: the four ``/v1/tokenize`` header probes of (i) (60 s each; ``header_ids_under`` compares
        its two renders) and the read-back of (iii), which by construction runs after the last completion;
  (iii) the ledger read back: rows == requests, ``error_code`` null and ``error_class == none`` on every
        forwarded row, every row's ``thinking`` flag as sent;
  (iv)  the emitted shape: OFF completions carry NO think tag; ON completions carry the family's shape
        (``tiers.think_split.split_think`` under the candidate mode: implicit_open -> a closing tag or at-cap
        unterminated, never a stray opening tag; explicit_tags -> an opening tag (or a tag-free answer, GLM's
        legal third case); record_only (gemma + the server parser) -> a non-empty ``reasoning_content``).

Exit 0 = PASS (every check), 2 = FAIL (the report names each failing check). The report and the completions are
what G-PARSER (the real-completion certification) reads; the runner commits them under
``data/tinycap/thinking_smoke/<family>/``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "math500"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cells import SPLIT_MODE_CANDIDATE  # the candidate split mode per family, the matrix's ONE copy
from sisteer.capeval.tinycap import contracts as C

DIAGNOSTIC = True
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
#: record_only (a server reasoning parser owns the split): the share of ON rows whose reasoning channel the parser
#: populated must reach this floor, else the channel is dead — a model may decline to think on a sampled draw
#: (temperature 1.0: 1 of 24 on the first gemma smoke), a parser cannot decline on half of them
MIN_ON_REASONING_SHARE = 0.5
HEADER_PROBE_MESSAGES = ({"role": "user", "content": "Probe."},)
# requests in flight at once through the ledger. The sbatch's --time formula is written for THIS number: the
# standup's S is >= 20 for every model at the smoke geometry (MATH's own 17,664 window), so 8 draws decode together rather than queue.
DEFAULT_CONCURRENCY = 8


def item_seed(item_id: str, sample_idx: int = 0) -> int:
    """``contracts.SEED_RULE_INSPECT`` through the driver's ONE implementation (``inspect_served.sample_seed``)."""
    from sisteer.capeval.tiers import inspect_served as isv
    return isv.sample_seed(item_id, sample_idx)


def _post(url: str, path: str, body: dict[str, Any], *, token: str | None, headers: dict[str, str] | None = None,
          timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(f"{url.rstrip('/')}/{path.lstrip('/')}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {}), **(headers or {})},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        return {"error": {"status": e.code, "body": detail}}


def header_ids_under(proxy_url: str, vmodel: str, *, enable_thinking: bool, token: str | None) -> list[int]:
    """The served generation header under the arm: the ``add_generation_prompt`` render minus the completed-turn
    render (the driver's ``_served_header_ids_under_arm`` construction), both through live ``/v1/tokenize`` with
    the kwarg as an explicit boolean."""
    def ids(agp: bool) -> list[int]:
        body = {"model": vmodel, "messages": [dict(m) for m in HEADER_PROBE_MESSAGES], "add_generation_prompt": agp,
                C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: bool(enable_thinking)}}
        resp = _post(proxy_url, "tokenize", body, token=token, timeout=60)
        toks = resp.get("tokens") if isinstance(resp, dict) else None
        if not isinstance(toks, list) or not toks:
            raise RuntimeError(f"/v1/tokenize answered agp={agp} enable_thinking={enable_thinking} with no tokens: {resp!r}")
        return [int(t) for t in toks]
    with_agp, without = ids(True), ids(False)
    if with_agp[:len(without)] != without:
        raise RuntimeError("the add_generation_prompt render is not an extension of the completed-turn render — "
                           f"the served header cannot be read (agp {with_agp[:12]}… vs {without[:12]}…)")
    return with_agp[len(without):]


def expected_header(family: str, arm: str, override: list[int] | None) -> list[int] | None:
    if override is not None:
        return list(override)
    facts = getattr(C, "FAMILY_RENDER_FACTS", None)
    if isinstance(facts, dict) and family in facts:
        f = facts[family]
        v = getattr(f, f"{arm}_ids", None) if not isinstance(f, dict) else f.get(f"{arm}_ids")
        return list(v) if v is not None else None
    return None


def shape_check(content: str, reasoning: str | None, *, finish_reason: str, mode: str, arm: str) -> dict[str, Any]:
    """The emitted shape under the candidate split mode (iv)."""
    rec: dict[str, Any] = {"mode": mode, "arm": arm, "has_open": THINK_OPEN in content, "has_close": THINK_CLOSE in content,
                           "reasoning_content_present": bool(reasoning), "finish_reason": finish_reason}
    if arm == "off":
        rec["ok"] = not rec["has_open"] and not rec["has_close"] and not (mode == "record_only" and rec["reasoning_content_present"])
        rec["why"] = None if rec["ok"] else "an OFF completion carries a think tag / reasoning channel"
        return rec
    try:
        from sisteer.capeval.tiers.think_split import split_think
        sp = split_think(content, mode=mode)
        rec.update(present=sp.present, unterminated=sp.unterminated, stray_open_tag=sp.stray_open_tag,
                   answer_chars=len(sp.answer_text))
    except Exception as e:
        rec["split_error"] = f"{type(e).__name__}: {e}"
    at_cap = finish_reason == "length"
    if mode == "record_only":
        # the server parser owns the split, so the only parser failure legible in the emitted shape is a think tag
        # LEAKED into content (a block the parser missed). A finished ON row with neither tag nor reasoning is the
        # MODEL declining to think on that draw: recorded as no_thinking_emitted and counted by the aggregate
        # on_reasoning_share check — never a per-row failure (gemma smoke 88959 failed on exactly one such row)
        leaked = rec["has_open"] or rec["has_close"]
        rec["no_thinking_emitted"] = not rec["reasoning_content_present"] and not at_cap and not leaked
        rec["ok"] = not leaked and "split_error" not in rec
        rec["why"] = None if rec["ok"] else (rec.get("split_error")
                                            or "gemma + server parser: a think tag leaked into content — the parser missed the block")
    elif mode == "implicit_open":
        rec["ok"] = (rec["has_close"] or at_cap) and not rec["has_open"]
        rec["why"] = None if rec["ok"] else ("a stray opening tag inside a prompt-opened block" if rec["has_open"]
                                            else "no closing tag on a finished completion (prompt-opened: entirely reasoning)")
    elif mode == "explicit_tags":
        rec["ok"] = (rec["has_open"] and (rec["has_close"] or at_cap)) or (not rec["has_open"] and not rec["has_close"])
        rec["why"] = None if rec["ok"] else "an explicit-tags completion opened a block it never closed before finishing"
    else:
        rec["ok"] = False
        rec["why"] = f"unknown split mode {mode!r}"
    return rec


def run_completions(url: str, plan: list[tuple[str, list[dict[str, str]], bool]], *, vmodel: str, cap: int,
                    params: dict[str, Any], token: str | None, mode: str, model_key: str, family: str,
                    sampling_mode: str, concurrency: int, timeout_s: float) -> list[dict[str, Any]]:
    """(ii): the plan's requests through the ledger with at most ``concurrency`` in flight (a thread pool; every
    request keeps its own ``timeout_s`` — the 7,200 s upstream timeout — so one slow cap-bound draw holds one slot,
    not the wall). Each row is recorded exactly as the serial probe recorded it, plus the request's start/end offsets
    from the batch's t0 (``t_start_s`` / ``t_end_s``: the overlap is legible from the dump itself). Returned in ONE
    deterministic order — (item id, arm), a total order since the OFF ids repeat the first ON ids — whatever order
    the server answered in. A transport error (not an HTTP status, which ``_post`` records on the row) propagates
    as it did serially; the requests not yet started are cancelled, the in-flight ones drain."""
    if concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {concurrency}")
    t_batch = time.time()

    def one(item_id: str, msgs: list[dict[str, str]], on: bool) -> dict[str, Any]:
        body = {"model": vmodel, "messages": msgs, "max_tokens": int(cap), "seed": item_seed(item_id, 0),
                C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: bool(on)}, **params}
        hdrs = {C.HEADER_TASK: "smoke/math", C.HEADER_SAMPLE: item_id}
        t0 = time.time()
        resp = _post(url, "chat/completions", body, token=token, headers=hdrs, timeout=timeout_s)
        t1 = time.time()
        choice = ((resp.get("choices") or [{}])[0]) if isinstance(resp, dict) and "error" not in resp else {}
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        stamps = resp.get("petri") if isinstance(resp, dict) else None
        row = {"diagnostic": DIAGNOSTIC, "item_id": item_id, "arm": "on" if on else "off", "model_key": model_key,
               "family": family, "split_mode_candidate": mode, "cap": int(cap), "seed": body["seed"],
               "sampling_mode": sampling_mode, "sampling_params": params,
               "finish_reason": choice.get("finish_reason"), "usage": resp.get("usage") if isinstance(resp, dict) else None,
               "stamps": stamps, "content": content, "reasoning_content": reasoning, "latency_s": round(t1 - t0, 2),
               "t_start_s": round(t0 - t_batch, 2), "t_end_s": round(t1 - t_batch, 2),
               "error": resp.get("error") if isinstance(resp, dict) else {"raw": str(resp)[:300]}}
        row["shape"] = shape_check(content, reasoning, finish_reason=str(choice.get("finish_reason")), mode=mode,
                                   arm=row["arm"]) if not row["error"] else None
        return row

    ex = cf.ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(plan))))
    try:
        rows = list(ex.map(lambda p: one(*p), plan))
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return sorted(rows, key=lambda r: (r["item_id"], r["arm"]))


def dump_completions(path: Path, rows: list[dict[str, Any]]) -> None:
    """``smoke_completions.jsonl``, one row per line in the order ``run_completions`` returned them."""
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ledger-url", required=True, help="the usage ledger's /v1 base (the requests go HERE)")
    ap.add_argument("--proxy-url", required=True, help="the standup's proxy /v1 base (/v1/tokenize; the ledger's upstream)")
    ap.add_argument("--proxy-token", default=None)
    ap.add_argument("--routing-table", required=True)
    ap.add_argument("--endpoint-json", required=True)
    ap.add_argument("--ledger-path", required=True, help="the ledger's usage_ledger.jsonl (read back)")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--family", required=True, choices=sorted(SPLIT_MODE_CANDIDATE))
    ap.add_argument("--sampling-mode", default="thinking_card")
    ap.add_argument("--sampling-params", default="{}", help="JSON: the row's request parameters (cells.py)")
    ap.add_argument("--n-on", type=int, default=24)
    ap.add_argument("--n-off", type=int, default=4)
    ap.add_argument("--cap", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="completions in flight at once through the ledger (>= 1; the sbatch's --time formula is written for the default)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-off-ids", default=None, help="JSON list: the family's OFF header ids (else FAMILY_RENDER_FACTS)")
    ap.add_argument("--expect-on-ids", default=None, help="JSON list: the family's ON header ids (else FAMILY_RENDER_FACTS)")
    ap.add_argument("--split-mode", default=None, help="override the candidate split mode")
    ap.add_argument("--repo-root", default=str(REPO))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, **detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", **detail})
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:400]}", flush=True)

    from sisteer.capeval.serving.proxy import load_routing_table
    table = load_routing_table(Path(a.routing_table))
    base = [r for r in table["routes"] if r.cell_id == "none"]
    if not base:
        check("base_route", False, why="no base route in the routing table")
    vmodel = base[0].virtual_model if base else "petri__base"
    ep = json.loads(Path(a.endpoint_json).read_text())
    check("upstream_timeout", ep.get(C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S) == C.PROXY_UPSTREAM_TIMEOUT_S,
          recorded=ep.get(C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S), want=C.PROXY_UPSTREAM_TIMEOUT_S)
    mode = a.split_mode or SPLIT_MODE_CANDIDATE[a.family]

    # (i) predicate (A): the served header under both arms, by TUPLE, against an expectation that must exist
    measured: dict[str, Any] = {}
    for arm, flag, override in (("off", False, a.expect_off_ids), ("on", True, a.expect_on_ids)):
        want = expected_header(a.family, arm, json.loads(override) if override else None)
        try:
            got = header_ids_under(a.proxy_url, vmodel, enable_thinking=flag, token=a.proxy_token)
        except Exception as e:
            got, err = None, f"{type(e).__name__}: {e}"
        else:
            err = None
        measured[arm] = {"got": got, "want": want, "error": err}
        if want is None:
            check(f"header_{arm}", False, got=got, why="no expected header ids: neither FAMILY_RENDER_FACTS nor --expect-*-ids "
                                                        "names the tuple — the probe must not pass vacuously")
        else:
            check(f"header_{arm}", err is None and list(got or []) == list(want), got=got, want=want, error=err)
    if measured["off"]["got"] is not None and measured["on"]["got"] is not None:
        check("header_arms_differ_or_invariant_family", (measured["off"]["got"] != measured["on"]["got"]) or a.family == "gemma4",
              off=measured["off"]["got"], on=measured["on"]["got"])

    # (ii) the items, through the ledger
    sys.path.insert(0, str(Path(a.repo_root) / "scripts"))
    spec = importlib.util.spec_from_file_location("_m5t_budget", Path(a.repo_root) / "scripts/tinycap_oos_prompt_budget.py")
    bud = importlib.util.module_from_spec(spec)
    sys.modules["_m5t_budget"] = bud
    spec.loader.exec_module(bud)
    from sisteer.capeval.tiers import inspect_served as isv
    ids = isv.math500_sample_ids()
    messages = bud.math500_messages(Path(a.repo_root))
    params = json.loads(a.sampling_params or "{}")
    plan = [(ids[i], messages[i], True) for i in range(min(a.n_on, len(ids)))] + \
           [(ids[i], messages[i], False) for i in range(min(a.n_off, len(ids)))]
    rows = run_completions(a.ledger_url, plan, vmodel=vmodel, cap=int(a.cap), params=params, token=a.proxy_token, mode=mode,
                           model_key=a.model_key, family=a.family, sampling_mode=a.sampling_mode,
                           concurrency=a.concurrency, timeout_s=float(C.PROXY_UPSTREAM_TIMEOUT_S))
    dump_completions(out / "smoke_completions.jsonl", rows)
    n_sent = len(rows)
    errors = [r for r in rows if r["error"]]
    n_at_cap_on = sum(1 for r in rows if r["arm"] == "on" and r["finish_reason"] == "length")
    check("requests_served", not errors, n=len(rows), n_errors=len(errors), first_error=(errors[0]["error"] if errors else None))
    on_regimes = sorted({str((r["stamps"] or {}).get("render_regime")) for r in rows if r["arm"] == "on"})
    off_regimes = sorted({str((r["stamps"] or {}).get("render_regime")) for r in rows if r["arm"] == "off"})
    check("render_regime_stamps", on_regimes == ["thinking_on"] and off_regimes == ["thinking_off"], on=on_regimes, off=off_regimes)
    salt = sorted({str((r["stamps"] or {}).get("cache_salt_regime")) for r in rows})
    from sisteer.capeval.serving.proxy import SALT_REGIME_CELL_STABLE
    check("base_route_salt_regime", salt == [SALT_REGIME_CELL_STABLE], regimes=salt)
    # (iv) the emitted shape
    bad = [r for r in rows if r["shape"] is not None and not r["shape"]["ok"]]
    check("emitted_shape", not bad and not errors, n_bad=len(bad),
          first_bad=({k: bad[0]["shape"].get(k) for k in ("arm", "why", "has_open", "has_close", "finish_reason")} if bad else None),
          at_cap_on=n_at_cap_on)
    # (iv-b) record_only: the reasoning channel is ALIVE — the parser populated it on at least MIN_ON_REASONING_SHARE of
    # the ON rows; the rows where the model declined to think are counted beside it
    if mode == "record_only":
        on_rows = [r for r in rows if r["arm"] == "on" and r["shape"] is not None]
        n_reasoning_on = sum(1 for r in on_rows if r["shape"].get("reasoning_content_present"))
        n_no_think_on = sum(1 for r in on_rows if r["shape"].get("no_thinking_emitted"))
        share = (n_reasoning_on / len(on_rows)) if on_rows else None
        check("on_reasoning_share", share is not None and share >= MIN_ON_REASONING_SHARE, n_on=len(on_rows),
              n_reasoning_on=n_reasoning_on, n_no_think_on=n_no_think_on, share=(round(share, 4) if share is not None else None),
              floor=MIN_ON_REASONING_SHARE)
    # (iii) the ledger read back
    lrows = []
    lp = Path(a.ledger_path)
    if lp.is_file():
        for line in lp.read_text().splitlines():
            if line.strip():
                try:
                    lrows.append(json.loads(line))
                except ValueError:
                    lrows.append({"_unparsable": line[:100]})
    fwd = [r for r in lrows if r.get("error_code") is None and r.get("error_class") in (None, "none")]
    check("ledger_rows", len(lrows) == n_sent and len(fwd) == n_sent and not [r for r in lrows if r.get("error_code")],
          n_rows=len(lrows), n_sent=n_sent, error_codes=sorted({str(r.get("error_code")) for r in lrows}),
          thinking_flags=sorted({str(r.get("thinking")) for r in lrows}))
    report = {"diagnostic": DIAGNOSTIC, "model_key": a.model_key, "family": a.family, "split_mode_candidate": mode,
              "vmodel": vmodel, "cap": int(a.cap), "n_on": a.n_on, "n_off": a.n_off, "concurrency": a.concurrency,
              "sampling_mode": a.sampling_mode,
              "sampling_params": params, "header": measured, "checks": checks,
              "n_at_cap_on": n_at_cap_on,
              "completion_tokens_on": [((r["usage"] or {}).get("completion_tokens")) for r in rows if r["arm"] == "on"],
              "verdict": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
              "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[smoke] {report['verdict']} — {sum(c['status'] == 'FAIL' for c in checks)} failing check(s); "
          f"report {out / 'smoke_report.json'}", flush=True)
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
