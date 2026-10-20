"""THE TOOL-PARSER PROBE'S TWO HALVES as plain functions (2026-10-03).

Split out of ``scripts/modal/tinycap_parser_probe.py`` (the Modal plumbing, which builds a ``modal.App`` at import) so the
mechanics import without modal: the standalone probe stands a row's ``none`` up itself, then runs the offline half and the live
half against it through these functions.

Pure rules (fixtures, pins, judging, the raw-token replay's comparison) stay in ``scripts/bfcl_parser_probe.py`` — READ FROM THE STAGED
TREE by the caller (``P`` below) — so a tree's pins and its probe travel together. This module holds only the mechanics: the real
vLLM parser instance, the chat request shell, the concurrent POST with raw-token capture, the CLIENT-venv item dump, the tokenizer at
the row's pin. Nothing here touches the volume; callers write.
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.tinycap import contracts as C

REQUEST_TIMEOUT_S = 900.0
#: the live probe's request fan-out (the pass-1 agent geometry serves S >= 14 seats) and the client-venv item dump's wall bound
REQUEST_CONCURRENCY = 8
ITEMS_DUMP_TIMEOUT_S = 1800
#: the CLIENT venv's python (inspect_evals lives there, never in the serving interpreter)
CLIENT_PY = f"{C.CLIENT_VENV}/bin/python"
#: The snippet that dumps the probe items under the CLIENT venv: the pinned ids' messages and their tools as NATIVE OpenAI tools
#: through inspect_evals' own ToolInfo conversion.
ITEMS_SNIPPET = r"""
import json, sys
tree, ids_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{tree}/src")
from inspect_evals.bfcl.bfcl import bfcl
from inspect_evals.bfcl.utils import create_tool_info_from_dict
from sisteer.capeval.tiers import inspect_served as I
want = json.loads(ids_json)
task = bfcl(categories=I.bfcl_categories())
by_id = {str(s.id): s for s in task.dataset}
missing = [i for i in want if i not in by_id]
if missing:
    raise SystemExit(f"probe ids not in the BFCL dataset: {missing}")
out = []
for i in want:
    s = by_id[i]
    msgs = [{"role": m["role"], "content": m["content"]} for m in s.metadata["raw_question"]]
    tools = []
    for spec in s.metadata["tools"]:
        ti = create_tool_info_from_dict(spec)
        tools.append({"type": "function", "function": {"name": ti.name, "description": ti.description,
                      "parameters": ti.parameters.model_dump(exclude_none=True)}})
    out.append({"id": i, "category": str(s.metadata["category_name"]), "messages": msgs, "tools": tools})
print(json.dumps(out))
"""


def vllm_tool_parser(tokenizer, name: str = "qwen3_xml"):
    """The REAL parser instance from the pinned wheel (0.25.1: ``vllm.tool_parsers``; the older
    ``vllm.entrypoints.openai.tool_parsers`` path as the fallback) — a fresh instance per call."""
    try:
        from vllm.tool_parsers import ToolParserManager  # type: ignore
    except ImportError:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager  # type: ignore
    return ToolParserManager.get_tool_parser(name)(tokenizer)


def chat_request(tools: Sequence[Mapping[str, Any]], model: str = "probe"):
    """A ``ChatCompletionRequest`` carrying the tools (the parser types parameters off ``request.tools``)."""
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # type: ignore
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "probe"}], tools=list(tools))


def parser_reading(parser, text: str, tools: Sequence[Mapping[str, Any]], *, request: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """``extract_tool_calls`` -> the probe's reading shape (``tools_called``, ``content``, ``tool_calls``). ``request`` builds the
    ``ChatCompletionRequest`` the parser types parameters off (default: this module's `chat_request`; the Modal plumbing passes
    its own seam so a test may replace it without vLLM installed)."""
    info = parser.extract_tool_calls(text, request=(request or chat_request)(tools))
    calls = []
    for tc in info.tool_calls or []:
        fn = getattr(tc, "function", tc)
        calls.append({"name": getattr(fn, "name", None), "arguments": getattr(fn, "arguments", None)})
    return {"tools_called": bool(info.tools_called), "content": info.content, "tool_calls": calls}


def pinned_tokenizer(row: C.ModelRow):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(row.hf, revision=row.rev, local_files_only=True)


def post_json(url: str, body: Mapping[str, Any], token: str, timeout_s: float = REQUEST_TIMEOUT_S) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body_text = ""
        return int(e.code), {"error": body_text[-2000:]}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def base_route_vmodel(endpoint: Mapping[str, Any]) -> str:
    """The served-model name of the BASE route (the `none` cell's vmodel) — the probe posts to it."""
    vmodels = endpoint.get("vmodels") or {}
    v = vmodels.get(C.ROUTE_NONE_CELL) if isinstance(vmodels, Mapping) else None
    if not v:
        raise KeyError(f"endpoint.json names no vmodel for the base route {C.ROUTE_NONE_CELL!r} (vmodels: {vmodels!r})")
    return str(v)


def offline_check_for(P, row: C.ModelRow, tree: str, sha: str, family: str, *, tokenizer: Optional[Any] = None,
                      rec: Optional[Dict[str, Any]] = None, tool_parser: Optional[Callable[..., Any]] = None,
                      reading: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """THE OFFLINE HALF for ``row``: the staged tree's fixtures through the REAL parser over the row's pinned tokenizer (the fixtures
    are keyed by the RENDER family — a certified served variant shares its family's template — the tokenizer is the ROW's); returns
    the judged record (``status OK`` with ``verdict``, or ``FIXTURES_INVALID`` / ``TOKENIZER_UNAVAILABLE``). The caller writes it."""
    rec = dict(rec or {"name": "parser-check-offline", "family": family, "sha": sha[:7], "model": row.slug, "model_key": row.model_key})
    tool_parser = tool_parser or vllm_tool_parser          # the seams a caller (the Modal plumbing's tests) may bind
    reading = reading or parser_reading
    try:
        fixtures = P.load_fixtures(tree)
    except P.ParserProbeError as e:
        return {**rec, "status": "FIXTURES_INVALID", "error": str(e)}
    try:
        tok = tokenizer if tokenizer is not None else pinned_tokenizer(row)
    except Exception as e:  # noqa: BLE001
        return {**rec, "status": "TOKENIZER_UNAVAILABLE", "error": f"{type(e).__name__}: {e} — stage the model into the volume HF cache first"}
    spelling = P.template_spelling_problems(fixtures, {family: tok})
    results: List[Dict[str, Any]] = []
    for fx in fixtures["fixtures"] + [{"id": f"family:{family}", "text": fixtures["families"][family]["rendered"], "tools": fixtures["tools"],
                                       "expected": {"tools_called": True, "content": None, "tool_calls": P.CANONICAL_CALLS}}]:
        try:
            got = reading(tool_parser(tok, P.PARSER), fx["text"], fx["tools"])
            results.append({"id": fx["id"], "parsed": got, "differences": P.parse_matches(got, fx["expected"]), "error": None})
        except Exception as e:  # noqa: BLE001 — a parser that raises on a fixture is a failing fixture
            results.append({"id": fx["id"], "parsed": None, "differences": [f"{type(e).__name__}: {e}"], "error": f"{type(e).__name__}: {e}"})

    def any_of(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Each any-of fixture (an edge, or an either-of) through a FRESH parser: the index of the first admitted reading the
        real parse equals, None when it equals none (or the parser raised)."""
        out: List[Dict[str, Any]] = []
        for fx in rows:
            try:
                got = reading(tool_parser(tok, P.PARSER), fx["text"], fx["tools"])
                idx = next((k for k, o in enumerate(fx["expected_any_of"]) if not P.parse_matches(got, o)), None)
                out.append({"id": fx["id"], "parsed": got, "outcome_index": idx})
            except Exception as e:  # noqa: BLE001
                out.append({"id": fx["id"], "parsed": None, "outcome_index": None, "error": f"{type(e).__name__}: {e}"})
        return out

    edges = any_of(fixtures["edges"])
    # the either-of class is ASSERTED — `judge_offline` FAILS a fixture whose real parse is outside
    # every admitted reading; the staged tree's module and its fixtures file travel together
    either_of = any_of(fixtures.get("either_of") or ())
    verdict = P.judge_offline(results, edges, either_of) if either_of else P.judge_offline(results, edges)
    if spelling:
        verdict = {**verdict, "verdict": P.VERDICT_FAIL, "reasons": list(verdict["reasons"]) + [f"template spelling: {spelling}"]}
    return {**rec, "status": "OK", "parser": P.PARSER, "chat_template_hash": P.template_hash(tok), "results": results,
            "either_of": either_of, "edges": edges, "template_spelling_problems": spelling, "verdict": verdict}


def live_probe_against(P, row: C.ModelRow, tree: str, endpoint: Mapping[str, Any], env: Mapping[str, str], pin: Mapping[str, Any], *,
                       endpoint_refusal: Optional[Callable[[C.ModelRow, Mapping[str, Any]], Optional[Mapping[str, Any]]]] = None,
                       tokenizer: Optional[Any] = None, post: Callable[..., Tuple[int, Dict[str, Any]]] = post_json,
                       run: Callable[..., subprocess.CompletedProcess] = subprocess.run, client_py: str = CLIENT_PY,
                       rec: Optional[Dict[str, Any]] = None, log: Callable[[str], None] = print,
                       tool_parser: Optional[Callable[..., Any]] = None, reading: Optional[Callable[..., Dict[str, Any]]] = None,
                       vmodel_of: Optional[Callable[[Mapping[str, Any]], str]] = None) -> Dict[str, Any]:
    """THE LIVE HALF against an EXISTING standup: the endpoint must carry the agent modality (and pass ``endpoint_refusal`` when the
    caller hands one — the launcher's full22 gates); the pinned items are dumped under the CLIENT venv, posted concurrently with the
    raw-token capture, detokenized and replayed through a FRESH parser; the judged record is returned — ``status OK`` with ``items`` +
    ``verdict``, or a named refusal (``ENDPOINT_REFUSED`` / ``NO_TOOL_CALLS`` / ``ITEMS_FAILED``). The caller writes it."""
    rec = dict(rec or {"name": "parser-probe-live", "family": C.family_of_model_key(row.model_key), "model": row.slug, "model_key": row.model_key})
    rec["endpoint"] = {k: endpoint.get(k) for k in ("tool_calls", "chat_template_hash", "max_model_len", "max_num_seqs", "model")}
    if endpoint_refusal is not None:
        refusal = endpoint_refusal(row, endpoint)
        if refusal is not None:
            status = refusal.get("status", "ENDPOINT_REFUSED") if isinstance(refusal, Mapping) else refusal[0]
            error = refusal.get("error") if isinstance(refusal, Mapping) else refusal[1]
            rec.update(status=status, error=error)
            return rec
    if not endpoint.get("tool_calls"):
        rec.update(status="NO_TOOL_CALLS", error="the standup came up WITHOUT the agent modality (--tool-calls)")
        return rec
    tool_parser = tool_parser or vllm_tool_parser
    reading = reading or parser_reading
    vmodel = (vmodel_of or base_route_vmodel)(endpoint)
    token = str(endpoint.get("proxy_token") or "petri")
    url = str(endpoint["proxy_url"]).rstrip("/")
    url = url if url.endswith("/v1") else url + "/v1"
    try:
        dump = run([client_py, "-c", ITEMS_SNIPPET, tree, json.dumps(list(pin["ids"]))],
                   env=dict(env, INSPECT_EVALS_CACHE_DIR=C.VOL_INSPECT_EVALS_CACHE), text=True, capture_output=True, timeout=ITEMS_DUMP_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:          # code review: a traceback here lost the standup's record
        rec.update(status="ITEMS_FAILED", error=f"the item dump did not finish within {ITEMS_DUMP_TIMEOUT_S} s "
                                                 f"({type(e).__name__}); stderr tail: {str(e.stderr or '')[-2000:]}")
        return rec
    if dump.returncode != 0:
        rec.update(status="ITEMS_FAILED", error=str(dump.stderr)[-3000:])
        return rec
    probe_items = json.loads(dump.stdout.strip().splitlines()[-1])
    tok = tokenizer if tokenizer is not None else pinned_tokenizer(row)
    eos = tok.eos_token
    items: List[Dict[str, Any]] = []
    # the live requests go out CONCURRENTLY (the served geometry batches them) and are read back in pin order
    bodies = [P.probe_body(it, model=vmodel) for it in probe_items]
    with ThreadPoolExecutor(max_workers=min(REQUEST_CONCURRENCY, max(1, len(bodies)))) as pool:
        responses = list(pool.map(lambda body: post(f"{url}/chat/completions", body, token), bodies))
    for it, (status, resp) in zip(probe_items, responses):
        if status != 200:
            items.append(P.item_record(it["id"], it["category"], http_status=status, served=None, raw_text=None, replayed=None,
                                       error=str(resp.get("error"))[:2000]))
            continue
        served = P.served_reading(resp)
        ids = served.get(P.RAW_TOKEN_IDS_KEY)
        raw_text = None
        replayed = None
        if ids:
            raw_text = P.strip_trailing_eos(tok.decode(list(ids), skip_special_tokens=False), eos)
            replayed = reading(tool_parser(tok, P.PARSER), raw_text, it["tools"])
            replayed["tool_calls"] = P.normalize_calls(replayed["tool_calls"])
        items.append(P.item_record(it["id"], it["category"], http_status=status, served=served, raw_text=raw_text, replayed=replayed))
        log(f"[parser-probe] {it['id']}: served {len(served['tool_calls'])} call(s), raw {'captured' if ids else 'MISSING'}, "
            f"replay {'match' if items[-1]['replay_matches'] else items[-1]['replay_matches']}")
    verdict = P.judge_live(items, n_expected=int(pin["n"]))
    rec.update(status="OK", items=items, verdict=verdict,
               request_shape={k: v for k, v in P.probe_body({"messages": [], "tools": []}, model=vmodel).items() if k not in ("messages", "tools")})
    return rec


def combined_doc(P, row: C.ModelRow, sha: str, family: str, offline: Optional[Mapping[str, Any]], live: Optional[Mapping[str, Any]], *,
                 at: str) -> Dict[str, Any]:
    """The probe's ``verdict.json`` body (the shape ``contracts.tool_parser_live_gate_problem`` opens): PASS only when BOTH halves read
    ``status OK`` + ``verdict PASS`` (``P.combined_verdict``); ``model_key`` is the ROW's (the family's probe row)."""
    ov = (offline or {}).get("verdict") if offline and offline.get("status") == "OK" else None
    lv = (live or {}).get("verdict") if live and live.get("status") == "OK" else None
    verdict = P.combined_verdict(ov, lv)
    return {"family": family, "model": row.slug, "model_key": row.model_key, "sha": sha[:7], "parser": P.PARSER, "verdict": verdict,
            "offline": ({"status": (offline or {}).get("status"), "verdict": ov} if offline is not None else None),
            "live": ({"status": (live or {}).get("status"), "verdict": lv} if live is not None else None),
            "on_pass": (P.on_pass_actions(family) if verdict == P.VERDICT_PASS and hasattr(P, "on_pass_actions") else []), "at": at}
