"""bcb-bfcl-200-widen S5: the tool-call parser probe's pure rules (`scripts/bfcl_parser_probe.py`)
and its Modal plumbing (`scripts/modal/tinycap_parser_probe.py`, loaded under the modal stub) —
the 24-item pin regenerates from the bfcl_200 population and refuses drift; the grammar reference
reads the fixtures; the request carries the same-request raw capture; the verdicts FAIL on every
silent failure and are INCONCLUSIVE when the parser was never exercised; the RoutingProxy relays
``return_token_ids`` / ``token_ids`` on a synthetic upstream (so no passthrough flag is needed)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import proxy as px
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bfcl_parser_probe as P  # noqa: E402
import test_petri_proxy as TP  # noqa: E402  (the stub backend + the hermetic proxy builder)
import test_tinycap_launcher as TL  # noqa: E402  (the modal stub)

MODAL_PROBE = REPO / "scripts" / "modal" / "tinycap_parser_probe.py"


# --------------------------------------------------------------------------- #
# the 24-item pin                                                              #
# --------------------------------------------------------------------------- #


def test_probe_pin_regenerates_from_the_bfcl_200_population_and_covers_every_category():
    pin = P.load_probe_pin(REPO)
    bfcl = C.load_bfcl_pin(REPO)
    assert pin["n"] == P.PROBE_N == len(pin["ids"]) == len(set(pin["ids"])) == 24
    assert set(pin["per_category"]) == set(bfcl["population"]) and min(pin["per_category"].values()) >= 1
    assert sum(pin["per_category"].values()) == 24
    # the probe's ids are the leg's: the same hash order per category, a prefix of the leg's seats
    assert set(pin["ids"]) <= set(bfcl["ids"]) and pin["in_bfcl_200"] == sorted(pin["ids"])
    for cat, ids in bfcl["population"].items():
        want = C.hash_order(ids, seed=C.SUBSET_SEED)[:pin["per_category"][cat]]
        assert [i for i in pin["ids"] if i in set(ids)] == want, cat
    # the remaining 6 seats went by largest remainder over the sizes (floor 0 after the 18 floors)
    extra = C.hamilton_allocation({c: len(v) for c, v in bfcl["population"].items()}, 6, floor=0)
    assert pin["per_category"] == {c: 1 + extra[c] for c in extra}


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["ids"].__setitem__(0, d["ids"][1]) or d["ids"].__setitem__(1, "exec_multiple_6"), "ids do not regenerate"),
    (lambda d: d["per_category"].__setitem__("sql", 2), "per_category"),
    (lambda d: d.__setitem__("n", 25), "keys|n "),
    (lambda d: d["source_sha256"].__setitem__("bfcl_200_population", "0" * 64), "source_sha256"),
])
def test_probe_pin_refuses_drift(tmp_path, mutate, needle):
    root = tmp_path / "repo"
    for rel in (C.BFCL_PIN_PATH, P.PROBE_PIN_PATH):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text((REPO / rel).read_text())
    doc = json.loads((root / P.PROBE_PIN_PATH).read_text())
    mutate(doc)
    (root / P.PROBE_PIN_PATH).write_text(json.dumps(doc))
    with pytest.raises((P.ParserProbeError, C.SubsetPinError), match=needle):
        P.load_probe_pin(root)


def test_the_plans_literal_hamilton_floor_1_at_24_is_infeasible_so_the_floor_comes_first():
    """The recorded decision: `hamilton_allocation` at floor 1 needs 28 seats over the 18 categories."""
    bfcl = C.load_bfcl_pin(REPO)
    with pytest.raises(ValueError, match="floors alone allocate 28 > 24"):
        C.bfcl_select(bfcl["population"], n=24, seed=C.SUBSET_SEED)


# --------------------------------------------------------------------------- #
# the grammar reference and the fixtures                                       #
# --------------------------------------------------------------------------- #


def test_fixtures_load_and_every_definite_fixture_parses_under_the_grammar():
    doc = P.load_fixtures(REPO)
    assert doc["parser"] == "qwen3_xml" and set(doc["families"]) == {"qwen3_5", "nemotron_h"}
    # `null_value` / `undeclared_param` MOVED from the definite fixtures to the ASSERTED either-of class
    assert {f["id"] for f in doc["fixtures"]} == {"single_string", "typed_params", "parallel_two", "multiline_value",
                                                  "content_then_call", "no_call_plain", "irrelevance_answer", "unclosed_trailing_block"}
    assert {e["id"] for e in doc["either_of"]} == {"null_value", "undeclared_param"}
    assert {e["id"] for e in doc["edges"]} == {"bare_function_no_wrapper", "empty_block"}
    assert len(doc["fixtures"]) == 8 and len(doc["either_of"]) == 2 and len(doc["edges"]) == 2
    assert P.fixture_problems(doc) == []
    # each either-of lists the REFERENCE reading first (the lane grammar's: JSON null / typed 7) and the schema-faithful
    # qwen3_xml reading second (the "null" / "7" strings) — two distinct outcomes, the reference's among them
    by_id = {e["id"]: e for e in doc["either_of"]}
    assert by_id["null_value"]["expected_any_of"][0]["tool_calls"][0]["arguments"] == {"city": "Paris", "note": None}
    assert by_id["null_value"]["expected_any_of"][1]["tool_calls"][0]["arguments"] == {"city": "Paris", "note": "null"}
    assert by_id["undeclared_param"]["expected_any_of"][0]["tool_calls"][0]["arguments"] == {"city": "Paris", "extra": 7}
    assert by_id["undeclared_param"]["expected_any_of"][1]["tool_calls"][0]["arguments"] == {"city": "Paris", "extra": "7"}
    for e in doc["either_of"]:
        got = P.parse_tool_calls_xml(e["text"], e["tools"])
        assert P.parse_matches(got, e["expected_any_of"][0]) == [] and P.parse_matches(got, e["expected_any_of"][1])
    assert P.either_of_fixtures() == doc["either_of"] and P.grammar_fixtures() == doc["fixtures"] and P.edge_fixtures() == doc["edges"]


def test_grammar_types_by_schema_and_reads_parallel_unclosed_and_content():
    tools = P.FIXTURE_TOOLS
    got = P.parse_tool_calls_xml(
        "thinking aloud\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
        "<parameter=zip>\n00075\n</parameter>\n<parameter=hourly>\nfalse\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=calc.add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2.5\n</parameter>", tools)
    assert got["tools_called"] is True and got["content"] == "thinking aloud\n"
    assert got["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3, "zip": "00075", "hourly": False}},
                                 {"name": "calc.add", "arguments": {"a": 1, "b": 2.5}}]
    assert P.parse_tool_calls_xml("no call here", tools) == {"tools_called": False, "content": "no call here", "tool_calls": []}
    # typed inequality is a difference: the grammar types by schema
    assert P.compare_calls([{"name": "f", "arguments": {"days": 3}}], [{"name": "f", "arguments": {"days": "3"}}])
    assert P.compare_calls([{"name": "f", "arguments": {"a": 1}}], [{"name": "f", "arguments": {"a": 1}}]) == []
    assert P.compare_calls([], [{"name": "f", "arguments": {}}]) == ["count 0 != 1"]
    # OpenAI tool_calls (arguments as a JSON string) normalise to values
    served = [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]
    assert P.normalize_calls(served) == [{"name": "f", "arguments": {"a": 1}}]


def test_fixture_problems_catch_a_wrong_expectation_and_a_family_pin_drift():
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc["fixtures"][0]["expected"]["tool_calls"][0]["arguments"]["city"] = "Rome"
    assert any("does not parse to its expectation" in p for p in P.fixture_problems(doc))
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc["families"]["qwen3_5"]["rev"] = "0" * 40
    assert any("rev" in p for p in P.fixture_problems(doc))
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc["edges"][0]["expected_any_of"] = doc["edges"][0]["expected_any_of"][:1]
    assert any(">= 2 distinct outcomes" in p for p in P.fixture_problems(doc))
    # the either-of class is held to the same self-consistency rules as the edges — and to its own key set
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc["either_of"][0]["expected_any_of"] = doc["either_of"][0]["expected_any_of"][:1]
    assert any(p.startswith("either-of 'null_value'") and ">= 2 distinct outcomes" in p for p in P.fixture_problems(doc))
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    for o in doc["either_of"][1]["expected_any_of"]:
        o["tool_calls"][0]["arguments"]["extra"] = "seven"
    assert any(p.startswith("either-of 'undeclared_param'") and "reference's reading is none" in p for p in P.fixture_problems(doc))
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc["either_of"][0]["expected"] = doc["either_of"][0].pop("expected_any_of")
    assert any(p.startswith("either-of 'null_value'") and "keys" in p for p in P.fixture_problems(doc))
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc.pop("either_of")
    assert P.fixture_problems(doc) and "keys" in P.fixture_problems(doc)[0]                       # schema 2 REQUIRES the key
    doc = json.loads(json.dumps(P.load_fixtures(REPO)))
    doc["fixtures"].append(dict(doc["either_of"][0], expected=doc["either_of"][0]["expected_any_of"][0]))
    del doc["fixtures"][-1]["expected_any_of"]
    assert any("duplicate fixture id 'null_value'" in p for p in P.fixture_problems(doc))


# --------------------------------------------------------------------------- #
# the live request, the readings and the verdicts                             #
# --------------------------------------------------------------------------- #

ITEM = {"id": "simple_python_42", "category": "simple_python",
        "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [P.WEATHER_TOOL]}


def test_probe_body_is_the_providers_request_plus_the_raw_capture():
    body = P.probe_body(ITEM, model="tinycap-none")
    assert body["model"] == "tinycap-none" and body["tool_choice"] == "auto" and body[P.RAW_CAPTURE_KEY] is True
    assert body["messages"] == ITEM["messages"] and body["tools"] == ITEM["tools"]
    assert body["temperature"] == 0.0 and body["max_tokens"] == P.PROBE_MAX_TOKENS
    assert not any(m["role"] == "system" for m in body["messages"])        # inspect_evals' single-turn solver adds none
    assert P.RAW_CAPTURE_KEY == "return_token_ids"                            # vLLM 0.25.1's extra-body key


def _response(calls, *, content=None, token_ids=(1, 2, 3)):
    return {"choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                         "message": {"role": "assistant", "content": content,
                                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                                    for i, c in enumerate(calls)]},
                         "token_ids": list(token_ids) if token_ids is not None else None}],
            "prompt_token_ids": [9, 9]}


def _item(i, *, calls=({"name": "get_weather", "arguments": {"city": "Paris"}},), http=200, raw=True, replay_calls=None, raw_text=None):
    text = raw_text if raw_text is not None else ("<tool_call>\n<function=get_weather>…" if calls else "plain answer")
    served = (P.served_reading(_response(list(calls), content=(None if calls else text), token_ids=(1, 2) if raw else None))
              if http == 200 else None)
    replayed = None
    if served is not None and raw:
        replayed = {"tools_called": bool(calls), "content": None if calls else text,
                    "tool_calls": list(calls) if replay_calls is None else list(replay_calls)}
    return P.item_record(f"id_{i}", f"cat_{i % 18}", http_status=http, served=served, raw_text=(text if raw else None), replayed=replayed)


def test_served_reading_returns_one_key_set_with_or_without_a_choice():
    """Code review: the no-choices branch omitted `prompt_token_ids`, so `item_record` (which
    copies `served` wholesale) filed two different shapes."""
    full = P.served_reading({"choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop",
                                          P.RAW_TOKEN_IDS_KEY: [1, 2]}], P.RAW_PROMPT_IDS_KEY: [7, 8]})
    empty = P.served_reading({"choices": [], P.RAW_PROMPT_IDS_KEY: [7, 8]})
    assert set(full) == set(empty) == {"content", "tool_calls", "finish_reason", P.RAW_TOKEN_IDS_KEY, P.RAW_PROMPT_IDS_KEY}
    assert empty[P.RAW_PROMPT_IDS_KEY] == [7, 8] and empty["tool_calls"] == [] and empty[P.RAW_TOKEN_IDS_KEY] is None
    assert P.served_reading({})[P.RAW_PROMPT_IDS_KEY] is None


def test_judge_live_passes_only_a_full_clean_probe_with_at_least_one_call():
    items = [_item(i) for i in range(24)]
    v = P.judge_live(items)
    assert v["verdict"] == P.VERDICT_PASS and v["reasons"] == [] and v["n_called"] == 24
    assert P.judge_live(items[:23])["verdict"] == P.VERDICT_FAIL                                  # not the pinned count
    assert P.judge_live(items[:23] + [_item(23, http=500)])["verdict"] == P.VERDICT_FAIL         # a 500 = the job-232411 failure
    v = P.judge_live(items[:23] + [_item(23, raw=False)])
    assert v["verdict"] == P.VERDICT_FAIL and any("return_token_ids" in r for r in v["reasons"])   # the capture was not honoured
    mismatch = _item(23, replay_calls=[{"name": "get_weather", "arguments": {"city": "Rome"}}])
    assert mismatch["replay_matches"] is False and P.judge_live(items[:23] + [mismatch])["verdict"] == P.VERDICT_FAIL
    # the SILENT failure: the raw text carries the grammar but the server parsed no call
    silent = _item(23, calls=(), raw_text="<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
    silent["replayed"] = {"tools_called": False, "content": silent["raw_text"], "tool_calls": []}
    silent = P.item_record("id_23", "cat_5", http_status=200, served=silent["served"], raw_text=silent["raw_text"], replayed=silent["replayed"])
    v = P.judge_live(items[:23] + [silent])
    assert v["verdict"] == P.VERDICT_FAIL and any("parsed no call" in r for r in v["reasons"])
    # no call anywhere: the parser was never exercised
    v = P.judge_live([_item(i, calls=()) for i in range(24)])
    assert v["verdict"] == P.VERDICT_INCONCLUSIVE and v["n_called"] == 0


def test_offline_and_combined_verdicts():
    ok = [{"id": "a", "differences": [], "error": None}, {"id": "b", "differences": [], "error": None}]
    assert P.judge_offline(ok, [{"id": "e", "outcome_index": 1, "parsed": {}}])["verdict"] == P.VERDICT_PASS
    assert P.judge_offline(ok + [{"id": "c", "differences": ["count 1 != 0"], "error": None}], [])["verdict"] == P.VERDICT_FAIL
    assert P.judge_offline(ok + [{"id": "c", "differences": ["boom"], "error": "boom"}], [])["n_failed"] == 1
    # the either-of class is ASSERTED — a fixture parsed to its second admitted reading PASSES; one parsed to NEITHER
    # (outcome_index None) or one the parser raised on FAILS by name; an EDGE outside its outcomes still only records
    v = P.judge_offline(ok, [{"id": "e", "outcome_index": None, "parsed": {}}], [{"id": "null_value", "outcome_index": 1, "parsed": {}}])
    assert v["verdict"] == P.VERDICT_PASS and v["n_either_of"] == 1 and v["n_either_of_failed"] == 0 and v["either_of"][0]["outcome_index"] == 1
    v = P.judge_offline(ok, [], [{"id": "null_value", "outcome_index": 1, "parsed": {}}, {"id": "undeclared_param", "outcome_index": None, "parsed": {"x": 1}}])
    assert v["verdict"] == P.VERDICT_FAIL and v["n_either_of_failed"] == 1
    assert any("either-of fixtures outside every admitted reading: ['undeclared_param']" in r for r in v["reasons"])
    v = P.judge_offline(ok, [], [{"id": "null_value", "outcome_index": None, "parsed": None, "error": "boom"}])
    assert v["verdict"] == P.VERDICT_FAIL and v["either_of"][0]["error"] == "boom"
    assert P.judge_offline(ok, [])["n_either_of"] == 0 and P.judge_offline(ok, [])["verdict"] == P.VERDICT_PASS       # no either-of: the old shape
    p, f = {"verdict": P.VERDICT_PASS}, {"verdict": P.VERDICT_FAIL}
    assert P.combined_verdict(p, p) == P.VERDICT_PASS
    assert P.combined_verdict(p, None) == P.VERDICT_INCONCLUSIVE == P.combined_verdict(None, p)
    assert P.combined_verdict(f, p) == P.VERDICT_FAIL == P.combined_verdict(p, f) == P.combined_verdict(None, f)
    acts = P.on_pass_actions("nemotron_h")
    assert any("TOOL_PARSER_BY_FAMILY" in a and "nemotron_h" in a for a in acts)
    assert any("test_nemotron_serving_registration" in a for a in acts) and any("petri_manifest" in a for a in acts)
    assert not any("test_nemotron_serving_registration" in a for a in P.on_pass_actions("qwen3_5"))
    cmds = P.modal_commands("f" * 40)
    assert len(cmds) == 2 and all(C.MODAL_APP_NAME_ENV in c and "tinycap_parser_probe.py" in c for c in cmds)
    assert any("--family nemotron_h" in c for c in cmds) and any("--family qwen3_5" in c for c in cmds)


# --------------------------------------------------------------------------- #
# the RoutingProxy relays the raw capture: request key forwarded, token ids    #
# relayed — verified on a synthetic upstream (no passthrough flag needed)      #
# --------------------------------------------------------------------------- #


def test_proxy_forwards_return_token_ids_and_relays_token_ids_and_prompt_token_ids(tmp_path):
    upstream = {"id": "cmpl-x", "object": "chat.completion", "created": 0, "model": TP.BASE_MODEL,
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None,
                                         "tool_calls": [{"id": "c0", "type": "function",
                                                         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                             "token_ids": [11, 22, 33]}],
                "prompt_token_ids": [1, 2, 3, 4], "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}
    backend = TP.StubBackend(responses=[json.loads(json.dumps(upstream))])
    proxy, backend, cells = TP._proxy(tmp_path, backend=backend)
    base = next(r for r in px.routes_from_cells(cells) if r.cell_id == "none")
    body = P.probe_body(ITEM, model=base.virtual_model)
    out = asyncio.run(proxy.handle_chat_completions(body))
    sent = backend.payloads[-1]
    assert sent[P.RAW_CAPTURE_KEY] is True and sent["tools"] == ITEM["tools"] and sent["tool_choice"] == "auto"
    assert sent["model"] == TP.BASE_MODEL                                     # the passthrough route: base weights
    assert out["choices"][0][P.RAW_TOKEN_IDS_KEY] == [11, 22, 33] and out[P.RAW_PROMPT_IDS_KEY] == [1, 2, 3, 4]
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    served = P.served_reading(out)
    assert served[P.RAW_TOKEN_IDS_KEY] == [11, 22, 33] and served["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    # the declaration keys are the ONLY ones the proxy pops; the reserved selector keys refuse
    assert P.RAW_CAPTURE_KEY not in px.FORBIDDEN_CLIENT_KEYS


# --------------------------------------------------------------------------- #
# the Modal plumbing under the stub                                            #
# --------------------------------------------------------------------------- #


def _load_modal_probe():
    # The probe module does `import tinycap_served as T`, and tinycap_served captures TINYCAP_APP_NAME at
    # import (`APP_NAME = os.environ.get(...)`) — so the sibling module imported HERE, under the test's
    # app name, must not stay in sys.modules: tinycap_tb reads `TS.APP_NAME` from whatever tinycap_served
    # is cached, and the terminal-bench launcher tests then saw "tinycap-bcbw-parser-test" as the running
    # app (the Modal CPU suite on the bcb-bfcl-200-widen merge). Restore sys.modules and sys.path too.
    saved, saved_env = sys.modules.get("modal"), os.environ.get(C.MODAL_APP_NAME_ENV)
    saved_siblings = {name: sys.modules.get(name) for name in ("tinycap_served", "tinycap_images")}
    modal_dir = str(REPO / "scripts" / "modal")
    sys.modules["modal"] = TL._stub_modal()
    os.environ[C.MODAL_APP_NAME_ENV] = "tinycap-bcbw-parser-test"
    sys.path.insert(0, modal_dir)
    try:
        spec = importlib.util.spec_from_file_location("tinycap_parser_probe_under_test", MODAL_PROBE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["modal"] = saved
        else:
            sys.modules.pop("modal", None)
        for name, prior in saved_siblings.items():
            if prior is not None:
                sys.modules[name] = prior
            else:
                sys.modules.pop(name, None)
        if sys.path and sys.path[0] == modal_dir:
            sys.path.pop(0)
        if saved_env is None:
            os.environ.pop(C.MODAL_APP_NAME_ENV, None)
        else:
            os.environ[C.MODAL_APP_NAME_ENV] = saved_env
    return mod


def test_modal_probe_module_binds_both_functions_and_routes_nemotron_to_its_image():
    M = _load_modal_probe()
    for name in ("check_tool_parser_offline", "probe_tool_parser", "probe_tool_parser_nemotron", "run_probe", "main"):
        assert callable(getattr(M, name)), name
    assert M.live_function("nemotron_h") is M.probe_tool_parser_nemotron and M.live_function("qwen3_5") is M.probe_tool_parser
    assert M.BFCL_GEOMETRY_KEY == "capability:agent"
    compile(M.ITEMS_SNIPPET, "<items>", "exec")                                # the CLIENT-venv dumper is valid Python
    assert "create_tool_info_from_dict" in M.ITEMS_SNIPPET and "raw_question" in M.ITEMS_SNIPPET
    # a parser double: the reading shape the live probe and the offline check share
    class _Info(SimpleNamespace):
        pass
    fake = SimpleNamespace(extract_tool_calls=lambda text, request: _Info(
        tools_called=True, content=None,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name="get_weather", arguments="{\"city\": \"Paris\"}"))]))
    M.chat_request = lambda tools, model="probe": None                          # no vllm here
    got = M.parser_reading(fake, "<tool_call>…", [P.WEATHER_TOOL])
    assert got == {"tools_called": True, "content": None, "tool_calls": [{"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}]}
    assert P.compare_calls(got["tool_calls"], [{"name": "get_weather", "arguments": {"city": "Paris"}}]) == []
    # NO_TREE: a missing staged tree is a status, never a traceback
    M.T.resolve_staged_tree = lambda sha, repos_root=None: (_ for _ in ()).throw(M.T.NoStagedTree("none"))
    assert M._check_offline("f" * 40, "qwen3_5")["status"] == "NO_TREE"
    assert M._probe_live("f" * 40, "qwen3_5")["status"] == "NO_TREE"


def _exec_in_container_root(script: Path, inventory, name: str):
    """Execute a Modal script's source the way its CONTAINER does: the source at
    ``/root/<name>.py`` (a fake root holding the entrypoint + EXACTLY the inventory of files the
    image places beside it), under the modal stub, with the checkout's ``scripts/modal`` OFF
    ``sys.path`` and no launcher module cached — so a sibling import that only resolves through
    the checkout fails here as it failed live. ``sisteer`` stays importable (the image's
    ``add_local_python_source``)."""
    import shutil
    import tempfile
    import types
    root = Path(tempfile.mkdtemp(prefix="modal_root_")) / "root"
    root.mkdir()
    shutil.copy(script, root / script.name)
    for remote, local in inventory.items():
        assert remote.startswith("/root/") and "/" not in remote[len("/root/"):], remote
        shutil.copy(local, root / remote[len("/root/"):])
    saved_path, saved_env = list(sys.path), os.environ.get(C.MODAL_APP_NAME_ENV)
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k in ("modal", "tinycap_served", "tinycap_images")}
    sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("scripts/modal")]
    sys.modules["modal"] = TL._stub_modal()
    os.environ[C.MODAL_APP_NAME_ENV] = "tinycap-bcbw-container-import-test"
    mod = types.ModuleType(name)
    mod.__file__ = str(root / script.name)
    try:
        exec(compile((root / script.name).read_text(), mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = saved_path
        for k in ("modal", "tinycap_served", "tinycap_images"):
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
        if saved_env is None:
            os.environ.pop(C.MODAL_APP_NAME_ENV, None)
        else:
            os.environ[C.MODAL_APP_NAME_ENV] = saved_env
    return mod, root
def test_the_modal_probe_module_imports_from_the_containers_flat_mount_with_only_the_images_files():
    """The container root holds the entrypoint, `tinycap_images.py` and the
    launcher the script now mounts beside itself — nothing from the checkout's `scripts/modal`."""
    M = _load_modal_probe()
    inv = M.container_inventory()
    # tinycap-shared-standup §1.5 (module D): the compile-cache sibling is a third beside-the-entrypoint file
    assert set(inv) == {"/root/tinycap_served.py", "/root/tinycap_images.py", "/root/tinycap_compile_cache.py"}
    mod, root = _exec_in_container_root(MODAL_PROBE, inv, "tinycap_parser_probe_in_container")
    assert mod.HERE == root and mod.REPO_LOCAL == root and callable(mod.run_probe)
    assert Path(mod.T.__file__).resolve() == (root / "tinycap_served.py").resolve()
    with pytest.raises(ModuleNotFoundError, match="tinycap_served"):
        _exec_in_container_root(MODAL_PROBE, dict(M.T.IMAGE_LOCAL_FILES), "tinycap_parser_probe_no_launcher")
    src = MODAL_PROBE.read_text()
    assert "image=T.image" not in src and src.count("image=IMAGE,") == 3 and "image=IMAGE_NEMOTRON," in src
    assert M.live_function("nemotron_h") is M.probe_tool_parser_nemotron           # the Nemotron function keeps its own image
    assert _load_modal_probe().REPO_LOCAL == REPO                                  # the grading host import finds the checkout


def test_every_live_outcome_is_filed_and_a_missing_tree_is_a_verdict(tmp_path, monkeypatch):
    """Code review: a non-OK live probe returned WITHOUT writing `parser_probe.json` (only
    verdict.json's status survived); `run_probe` tracebacked on a missing tree; the item dump's
    TimeoutExpired was uncaught; a KV refusal was read off a standup still alive."""
    M = _load_modal_probe()
    # run_probe: NO_TREE is a verdict document, filed nowhere (no row to file under), never a traceback
    monkeypatch.setattr(M.T, "resolve_staged_tree", lambda sha, repos_root=None: (_ for _ in ()).throw(M.T.NoStagedTree("none")))
    doc = M.run_probe("f" * 40, "qwen3_5")
    assert doc["status"] == doc["verdict"] == "NO_TREE" and doc["on_pass"] == []
    # the live probe from the standup on: a fake tree carrying the probe module and the pin
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "scripts" / "bfcl_parser_probe.py").write_text((REPO / "scripts" / "bfcl_parser_probe.py").read_text())
    for rel in (P.PROBE_PIN_PATH, P.FIXTURES_PATH, C.BFCL_PIN_PATH):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(M.T, "resolve_staged_tree", lambda sha, repos_root=None: (tree, "f" * 40))
    monkeypatch.setattr(M, "_out_dir", lambda Pm, row, sha: out_dir)
    monkeypatch.setattr(M, "_write", lambda path, doc: Path(path).write_text(json.dumps(doc, default=str)))
    monkeypatch.setattr(M.T, "overlaid_config", lambda row, tree_, out: (_ for _ in ()).throw(ValueError("no overlay")))
    rec = M._probe_live("f" * 40, "qwen3_5")
    assert rec["status"] == "CONFIG_OVERLAY_FAILED"
    assert json.loads((out_dir / "parser_probe.json").read_text())["status"] == "CONFIG_OVERLAY_FAILED"   # FILED
    # a standup that never came up and is STILL ALIVE at the deadline: no KV refusal is read off it
    monkeypatch.setattr(M.T, "overlaid_config", lambda row, tree_, out: (str(tree / "cfg.yaml"), {}))
    monkeypatch.setattr(M.T, "_geometry_module", lambda tree_: SimpleNamespace(
        pass1_geometry=lambda key, model_key: SimpleNamespace(max_model_len=16384, max_num_seqs=16)))
    (tree / P.PROBE_MANIFESTS["qwen3_5"]).parent.mkdir(parents=True, exist_ok=True)
    (tree / P.PROBE_MANIFESTS["qwen3_5"]).write_text("petri: {}\n")
    monkeypatch.setattr(M.T, "_standup_cmd", lambda *a, **k: ["fake-standup"])
    monkeypatch.setattr(M.T, "_wait_for", lambda path, proc, timeout_s: False)
    monkeypatch.setattr(M.T, "_tail", lambda path, n=3000: "KV cache holds 1000 tokens < the 2000-token budget (= 1 x 2000)")
    monkeypatch.setattr(M.T, "engine_core_tail", lambda text: None)
    monkeypatch.setattr(M.T, "_tear_down_standup", lambda stop, proc: None)

    class _Alive:
        def poll(self):
            return None

    class _Dead(_Alive):
        def poll(self):
            return 1
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Alive())
    monkeypatch.setattr(M, "STANDUP_WAIT_S", 0)
    rec = M._probe_live("f" * 40, "qwen3_5")
    assert rec["status"] == "STANDUP_FAILED" and rec["kv_refusal"] is None and rec["standup_exited"] is False
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Dead())
    rec = M._probe_live("f" * 40, "qwen3_5")
    assert rec["status"] == "STANDUP_FAILED" and rec["standup_exited"] is True and "KV cache holds" in rec["kv_refusal"]
    assert json.loads((out_dir / "parser_probe.json").read_text())["status"] == "STANDUP_FAILED"
    # the item dump's timeout is ITEMS_FAILED, filed
    monkeypatch.setattr(M.T, "_wait_for", lambda path, proc, timeout_s: True)
    (tree / "endpoint.json").write_text(json.dumps({"tool_calls": True, "proxy_url": "http://x/v1", "proxy_token": "t"}))
    monkeypatch.setattr(M.T, "full22_endpoint_refusal", lambda row, endpoint, tree=None: None)
    monkeypatch.setattr(M, "_base_route_vmodel", lambda endpoint: "vm")
    import shutil as _sh
    srv = Path(f"/tmp/parser_probe_{C.model_row(P.PROBE_ROWS['qwen3_5']).slug}")
    monkeypatch.setattr(M.T, "_wait_for", lambda path, proc, timeout_s: (Path(path).write_text(json.dumps(
        {"tool_calls": True, "proxy_url": "http://x/v1", "proxy_token": "t"})) or True))
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a[0], k.get("timeout"))))
    rec = M._probe_live("f" * 40, "qwen3_5")
    assert rec["status"] == "ITEMS_FAILED" and "did not finish within" in rec["error"]
    assert json.loads((out_dir / "parser_probe.json").read_text())["status"] == "ITEMS_FAILED"
    _sh.rmtree(srv, ignore_errors=True)


def test_the_offline_check_asserts_the_either_of_class_through_the_real_parser_hook(tmp_path, monkeypatch):
    """End to end on the Modal plumbing (no vLLM here — the parser is a double over the lane's reference grammar
    with a SWITCHABLE reading of the two admitted cases): the schema-faithful reading of `null_value` / `undeclared_param`
    PASSES as outcome 1 (the admitted qwen3_xml behaviour), the reference reading PASSES as outcome 0, and a
    THIRD reading FAILS naming the fixture — the record carries the either_of block beside results / edges."""
    M = _load_modal_probe()
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "scripts" / "bfcl_parser_probe.py").write_text((REPO / "scripts" / "bfcl_parser_probe.py").read_text())
    for rel in (P.PROBE_PIN_PATH, P.FIXTURES_PATH, C.BFCL_PIN_PATH):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(M.T, "resolve_staged_tree", lambda sha, repos_root=None: (tree, "f" * 40))
    monkeypatch.setattr(M, "_out_dir", lambda Pm, row, sha: out_dir)
    monkeypatch.setattr(M, "_write", lambda path, doc: Path(path).write_text(json.dumps(doc, default=str)))
    monkeypatch.setattr(M, "_tokenizer", lambda row: SimpleNamespace(chat_template="x"))
    monkeypatch.setattr(M, "chat_request", lambda tools, model="probe": None)
    monkeypatch.setattr(M.T, "_load_tree_module", lambda tree_, rel, name: P)
    monkeypatch.setattr(P, "template_spelling_problems", lambda doc, toks: [])
    monkeypatch.setattr(P, "template_hash", lambda tok: P.TEMPLATE_HASHES["qwen3_5"])

    def parser_with(reading):
        def extract(text, request):
            got = P.parse_tool_calls_xml(text, P.FIXTURE_TOOLS)                   # the reference grammar ...
            for c in got["tool_calls"]:                                            # ... with the ruled cases re-read
                a = c["arguments"]
                if "note" in a and a["note"] is None:
                    a["note"] = {"reference": None, "schema_faithful": "null", "third": "NULL!"}[reading]
                if "extra" in a and a["extra"] == 7:
                    a["extra"] = {"reference": 7, "schema_faithful": "7", "third": 7.5}[reading]
            calls = [SimpleNamespace(function=SimpleNamespace(name=c["name"], arguments=json.dumps(c["arguments"]))) for c in got["tool_calls"]]
            return SimpleNamespace(tools_called=got["tools_called"], content=got["content"], tool_calls=calls)
        return SimpleNamespace(extract_tool_calls=extract)

    for reading, want, idx in (("schema_faithful", P.VERDICT_PASS, 1), ("reference", P.VERDICT_PASS, 0), ("third", P.VERDICT_FAIL, None)):
        monkeypatch.setattr(M, "vllm_tool_parser", lambda tok, name="qwen3_xml", _r=reading: parser_with(_r))
        rec = M._check_offline("f" * 40, "qwen3_5")
        assert rec["status"] == "OK", rec.get("error")
        v = rec["verdict"]
        assert v["verdict"] == want, (reading, v["reasons"])
        assert v["n_fixtures"] == 9 and v["n_failed"] == 0                         # 8 definite + the family spelling, all exact
        assert v["n_either_of"] == 2 and [e["outcome_index"] for e in rec["either_of"]] == [idx, idx]
        assert v["n_either_of_failed"] == (2 if idx is None else 0)
        if idx is None:
            assert any("either-of fixtures outside every admitted reading: ['null_value', 'undeclared_param']" in r for r in v["reasons"])
        assert [e["id"] for e in rec["edges"]] == ["bare_function_no_wrapper", "empty_block"]
        filed = json.loads((out_dir / "parser_check.json").read_text())
        assert filed["verdict"]["verdict"] == want and [e["id"] for e in filed["either_of"]] == ["null_value", "undeclared_param"]

