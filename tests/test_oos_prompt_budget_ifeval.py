"""ifeval-canonical-battery — the ``--task ifeval`` branch of the prompt-budget CLI and the G-LENGTHS measurement.

Pinned on a SYNTHETIC tokenizer + renderer (no model download): the branch on all four IFEval model keys, bare and
prompted (the SI folded into ONE leading system turn), the lengths-only artifact (contract-exact, ``n 541``, ``max``,
``p99``, ``sum`` of the per-item lengths, ``cap_task inspect/ifeval``, the OFF render), NO sidecar without
``--geometry-eval``, the ``.fit.json`` sidecar with exit 2 on a non-fit, the by-name refusals (``--arm on``; a non-IFEval
model key; a cap off the ladder; an IFEval request routed to the GSM8K renderer; a --config whose pin disagrees), and
the measurement document (entries ``next_block(max + cap)`` per PROMPT CLASS — the short classes' max sizes
``capability:ifeval_<cap>``, the constitution class's ``capability:ifeval_longsi_<cap>`` — byte-identical
rendering, ``--check``). On the REAL pinned tokenizers (offline from the local HF cache; skipped otherwise): the committed
``configs/sdfbase/ifeval/geometry_measurement.json`` regenerates byte-identically at its recorded tree sha.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / C.OOS_PROMPT_BUDGET_SCRIPT
MEASURE = REPO / "scripts" / "sdfbase" / "ifeval" / "measure_ifeval_geometry.py"
SI = "SAFETY SI: refuse harm, help fully otherwise."


def _head_sha() -> str:
    """This checkout's HEAD: the only --tree-sha the producer accepts (it is checked, never recorded on the caller's word)."""
    import subprocess
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


@pytest.fixture(scope="module")
def B():
    return _load("tinycap_oos_prompt_budget_ifeval_under_test", SCRIPT)


@pytest.fixture(scope="module")
def G():
    return _load("measure_ifeval_geometry_under_test", MEASURE)


class _FakeTok:
    """chars // 4 + a header of 7 under the OFF render (5 under ON), like the GSM8K fixture; the template text per model so
    the four hashes differ."""

    def __init__(self, tag: str):
        self.chat_template = f"{{% for m in messages %}}<{tag}>{{{{ m.role }}}}{{% endfor %}}"
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], "kwargs": dict(kw), "add_generation_prompt": add_generation_prompt})
        # a message's content is a string OR the OpenAI text-parts list a multimodal template is handed
        # (contracts.PROMPT_BUDGET_CONTENT_FORMAT_OPENAI); this synthetic tokenizer has no gemma quirk, so both count alike
        body = sum(len(m["content"]) if isinstance(m["content"], str) else sum(len(p["text"]) for p in m["content"])
                   for m in messages) // 4
        return list(range(body + (5 if kw.get("enable_thinking") else 7)))


def _ifeval_msgs(system=None, n=IC.IFEVAL_N):
    base = [[{"role": "user", "content": f"Write {i} sentences about item {i}. " + "x" * (i % 97)}] for i in range(n)]
    if system:
        return [[{"role": "system", "content": system}, *m] for m in base]
    return base


def _fake_loader(B, monkeypatch, toks):
    def loader(key, config_path):
        hf, rev = IC.IFEVAL_MODEL_PINS[key]
        return hf, rev, key, toks[key]
    monkeypatch.setattr(B, "_load_ifeval_tokenizer", loader)


def _fake_renderer(B, monkeypatch):
    monkeypatch.setattr(B, "ifeval_class_messages", lambda root, system_prompt=None: _ifeval_msgs(system_prompt))
    monkeypatch.setattr(B, "system_prompt_for_cell", lambda cell: SI if cell == IC.CELL_PROMPTED else None)


# --------------------------------------------------------------------------- the branch on a synthetic tokenizer
@pytest.mark.parametrize("key", IC.IFEVAL_MODELS)
@pytest.mark.parametrize("prompted", [False, True])
def test_the_ifeval_branch_writes_a_contract_exact_lengths_artifact_for_every_model_bare_and_prompted(B, tmp_path, monkeypatch, capsys, key, prompted):
    toks = {k: _FakeTok(k) for k in IC.IFEVAL_MODELS}
    _fake_loader(B, monkeypatch, toks)
    _fake_renderer(B, monkeypatch)
    out = tmp_path / "b.json"
    argv = ["--task", "ifeval", "--arm", "off", "--model-key", key, "--cap", "2048", "--out", str(out)]
    if prompted:
        argv += ["--system-prompt-cell", IC.CELL_PROMPTED]
    assert B.main(argv) == 0
    doc = json.loads(out.read_text())
    assert set(doc) == set(C.OOS_PROMPT_BUDGET_FIELDS)
    hf, rev = IC.IFEVAL_MODEL_PINS[key]
    assert (doc["n"], doc["model"], doc["revision"], doc["model_slug"], doc["arm"], doc["cap"], doc["cap_task"]) == (541, hf, rev, key, "off", 2048, "inspect/ifeval")
    want = [sum(len(m["content"]) for m in ml) // 4 + 7 for ml in _ifeval_msgs(SI if prompted else None)]
    assert doc["lengths"] == want and doc["max"] == max(want) and doc["max_plus_cap"] == max(want) + 2048
    assert doc["p99"] == C.nearest_rank_percentile(want, 0.99) and sum(doc["lengths"]) == sum(want)
    # the IFEval artifact RECORDS the chat-content form it rendered (the served path's form for the family — the wave
    # shakedown, 2026-09-13): parts for the multimodal templates (gemma4, qwen3_5), strings for the text templates
    assert doc["render"] == {"add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
                             "content_format": C.prompt_budget_content_format(key)}
    assert doc["render"]["content_format"] == ("openai" if key in ("gemma4e4b", "qwen35_4b") else "string")
    assert doc["chat_template_hash"] == __import__("sisteer.models", fromlist=["chat_template_hash"]).chat_template_hash(toks[key])
    from sisteer.capeval.tiers import ifeval_messages as IFM
    assert doc["messages_source"] == IFM.messages_source(prompted=prompted)
    # the render the tokenizer saw: OFF, the SI folded into ONE leading system turn for the prompted class
    first = toks[key].calls[0]
    assert first["kwargs"] == {"enable_thinking": False} and first["add_generation_prompt"] is True
    assert [m["role"] for m in first["messages"]] == (["system", "user"] if prompted else ["user"])
    if prompted:
            # ... in the family's chat-content form: the OpenAI text parts a multimodal template is handed, else the string
        want_si = [{"type": "text", "text": SI}] if C.prompt_budget_content_format(key) == "openai" else SI
        assert first["messages"][0]["content"] == want_si
    assert not (tmp_path / "b.json.fit.json").exists()
    assert "NO fit sidecar" in capsys.readouterr().out


def test_the_ifeval_branch_measures_the_no_switch_arm_for_llama_sending_no_switch(B, tmp_path, monkeypatch):
    """A llama IFEval configuration serves `not_available` (the certified render
    fact — no `enable_thinking` switch is sent), and the driver's live guard refuses an artifact measured under `off`.
    The branch admits `--arm not_available` (it refuses ON only), records the arm, renders with NO switch kwarg, and
    keeps the family's string chat-content form; the explicit `--cap` is the rung (a no-switch family has no cap table)."""
    toks = {k: _FakeTok(k) for k in IC.IFEVAL_MODELS}
    _fake_loader(B, monkeypatch, toks)
    _fake_renderer(B, monkeypatch)
    out = tmp_path / "ll.json"
    argv = ["--task", "ifeval", "--arm", C.THINKING_NOT_AVAILABLE, "--model-key", "llama8b", "--cap", "2048", "--out", str(out)]
    assert B.main(argv) == 0
    doc = json.loads(out.read_text())
    assert (doc["arm"], doc["cap"], doc["cap_task"], doc["model_slug"]) == (C.THINKING_NOT_AVAILABLE, 2048, "inspect/ifeval", "llama8b")
    assert doc["render"] == {"add_generation_prompt": True, "chat_template_kwargs": {}, "content_format": "string"}
    first = toks["llama8b"].calls[0]
    assert first["kwargs"] == {} and first["add_generation_prompt"] is True          # no switch sent, as the served path sends none
    want = [sum(len(m["content"]) for m in ml) // 4 + 7 for ml in _ifeval_msgs(None)]
    assert doc["lengths"] == want and doc["max_plus_cap"] == max(want) + 2048
    # the launcher's guard on the served side accepts exactly this artifact for the no-switch arm, and refuses OFF's
    C.prompt_budget_matches_run(doc, arm=C.THINKING_NOT_AVAILABLE, cap=2048)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="measure the arm that is served"):
        C.prompt_budget_matches_run(doc, arm=C.THINKING_OFF, cap=2048)


def test_the_fit_sidecar_is_written_only_with_geometry_eval_and_a_non_fit_exits_2(B, tmp_path, monkeypatch):
    toks = {k: _FakeTok(k) for k in IC.IFEVAL_MODELS}
    _fake_loader(B, monkeypatch, toks)
    _fake_renderer(B, monkeypatch)
    windows = {"capability:ifeval_2048": 2528, "small": 2050}
    monkeypatch.setattr(B, "geometry_window", lambda name: windows[name])
    out = tmp_path / "b.json"
    # --tree-sha is checked against the producer's checkout: the pin is THIS checkout's HEAD, and a fake one refuses by name
    head = _head_sha()
    with pytest.raises(SystemExit, match="names a tree this checkout is not at"):
        B.main(["--task", "ifeval", "--arm", "off", "--model-key", "qwen35_4b", "--cap", "2048", "--out", str(out),
                "--geometry-eval", "capability:ifeval_2048", "--tree-sha", "a" * 40])
    rc = B.main(["--task", "ifeval", "--arm", "off", "--model-key", "qwen35_4b", "--cap", "2048", "--out", str(out),
                 "--geometry-eval", "capability:ifeval_2048", "--tree-sha", head])
    side = json.loads((tmp_path / "b.json.fit.json").read_text())
    assert rc == 0 and side["fits"] is True and side["task"] == "ifeval" and side["control"] == "ifeval"
    assert side["window"] == 2528 and side["geometry_eval"] == "capability:ifeval_2048" and side["tree_sha"] == head
    assert side["system_prompt_cell"] is None and side["system_prompt_sha16"] is None
    rc = B.main(["--task", "ifeval", "--arm", "off", "--model-key", "qwen35_4b", "--cap", "2048", "--out", str(tmp_path / "c.json"),
                 "--geometry-eval", "small", "--system-prompt-cell", IC.CELL_PROMPTED])
    side = json.loads((tmp_path / "c.json.fit.json").read_text())
    assert rc == 2 and side["fits"] is False and "small" in side["geometry_eval"] and side["system_prompt_cell"] == IC.CELL_PROMPTED
    assert side["system_prompt_sha16"] == __import__("hashlib").sha256(SI.encode()).hexdigest()[:16]


def test_the_branch_refuses_by_name_the_on_arm_a_foreign_key_a_cap_off_the_ladder_and_a_disagreeing_config(B, tmp_path, monkeypatch):
    toks = {k: _FakeTok(k) for k in IC.IFEVAL_MODELS}
    _fake_renderer(B, monkeypatch)
    with pytest.raises(SystemExit, match="thinking-OFF instrument"):
        B.main(["--task", "ifeval", "--arm", "on", "--model-key", "qwen35_4b", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="needs --model-key"):
        B.main(["--task", "ifeval", "--arm", "off", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="not a rung of the IFEval ladder"):
        B.main(["--task", "ifeval", "--arm", "off", "--model-key", "qwen35_4b", "--cap", "3000", "--out", str(tmp_path / "x")])
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="not an IFEval model"):
        B._load_ifeval_tokenizer("glm46vflash", None)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: Qwen/Qwen3.5-4B\nmodel_revision: 0000000000000000000000000000000000000000\n")
    monkeypatch.setattr("sisteer.models.load_tokenizer", lambda hf, revision: toks["qwen35_4b"])
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="pins disagree"):
        B._load_ifeval_tokenizer("qwen35_4b", cfg)
    good = tmp_path / "good.yaml"
    good.write_text("model: Qwen/Qwen3.5-4B\nmodel_revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a\n")
    assert B._load_ifeval_tokenizer("qwen35_4b", good)[:3] == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "qwen35_4b")


def test_an_ifeval_request_routed_to_the_gsm8k_renderer_refuses_by_name_and_the_dispatch_never_routes_it_there(B, monkeypatch):
    a = B.build_parser().parse_args(["--task", "ifeval", "--config", "x.yaml", "--model-key", "qwen35_4b"])
    with pytest.raises(SystemExit, match="routed to the GSM8K renderer"):
        B.main_gsm8k_general(a)
    called = {}

    def _mark(name):
        def run(a):
            called[name] = True
            return 0
        return run
    monkeypatch.setattr(B, "main_ifeval", _mark("ifeval"))
    monkeypatch.setattr(B, "main_gsm8k_general", _mark("gsm8k"))
    assert B.main(["--task", "ifeval", "--arm", "off", "--config", "x.yaml", "--model-key", "qwen35_4b"]) == 0
    assert called == {"ifeval": True}
    assert B.TASKS == ("gsm8k_paired200", "math500", "ifeval") and B.TASK_IFEVAL == IC.IFEVAL_TASK_NAME
    # the default task is unchanged
    assert B.build_parser().parse_args([]).task == "gsm8k_paired200"


# --------------------------------------------------------------------------- the measurement document
def test_the_measurement_sizes_every_entry_by_next_block_of_the_max_over_models_and_classes_and_renders_byte_identically(G, monkeypatch):
    toks = {k: _FakeTok(k) for k in IC.IFEVAL_MODELS}

    def loader(key):
        hf, rev = IC.IFEVAL_MODEL_PINS[key]
        return hf, rev, toks[key]

    def renderer(system_prompt):
        return _ifeval_msgs(system_prompt)

    import tinycap_oos_prompt_budget_ifeval_under_test as B  # the same module object the measurement loads by path
    monkeypatch.setattr(G, "_budget_module", lambda: B)
    # the constitution class folds the REAL si_constitution text (the measurement asserts its pinned sha256); the
    # tokenizers and the renderer stay synthetic
    from sisteer.eval.jailbreak import resolve_instruction
    CONST_SI = str(resolve_instruction(IC.IFEVAL_CONSTITUTION_SI_ID).text)
    monkeypatch.setattr(B, "system_prompt_for_cell",
                        lambda cell: SI if cell == IC.CELL_PROMPTED else CONST_SI if cell == IC.CELL_PROMPTED_CONSTITUTION else None)
    doc = G.measure(tree="b" * 40, loader=loader, renderer=renderer)
    doc2 = G.measure(tree="b" * 40, loader=loader, renderer=renderer)
    assert G.render_measurement(doc) == G.render_measurement(doc2)
    assert doc["schema"] == "ifeval_geometry_measurement/1" and doc["tree_sha"] == "b" * 40 and doc["n"] == 541 and doc["arm"] == "off"
    assert set(doc["by_model"]) == set(IC.IFEVAL_MODELS) and set(doc["by_slug"]) == set(IC.IFEVAL_SLUGS.values())
    bare = [sum(len(m["content"]) for m in ml) // 4 + 7 for ml in _ifeval_msgs()]
    prompted = [sum(len(m["content"]) for m in ml) // 4 + 7 for ml in _ifeval_msgs(SI)]
    for key, e in doc["by_model"].items():
        assert e["classes"]["bare"]["lengths"] == bare and e["classes"]["prompted"]["lengths"] == prompted
        assert e["classes"]["bare"]["max"] == max(bare) and e["classes"]["prompted"]["sum"] == sum(prompted)
        assert e["si_delta"] == {"min": len(SI) // 4 - 0, "max": len(SI) // 4} or e["si_delta"]["min"] <= e["si_delta"]["max"]
    assert doc["max_prompt_tokens"] == max(prompted) and doc["max_prompt_argmax"]["class"] == "prompted"
    for cap in IC.IFEVAL_CAP_LADDER:
        en = doc["entries"][str(cap)]
        assert en["name"] == f"capability:ifeval_{cap}" and en["max_model_len"] == sg.next_block(max(prompted) + cap)
        assert en["sized_by"] == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX and 0 <= en["headroom"] < sg.VLLM_BLOCK_SIZE
        assert en["max_model_len"] % sg.VLLM_BLOCK_SIZE == 0 and en["requested_max_tokens"] == cap
        assert en["classes"] == ["bare", "prompted"]
    assert doc["system_prompt_sha16"] == __import__("hashlib").sha256(SI.encode()).hexdigest()[:16]
    # the follow-up: the constitution class is its OWN entry family, sized over that class alone
    constitution = [sum(len(m["content"]) for m in ml) // 4 + 7 for ml in _ifeval_msgs(CONST_SI)]
    assert doc["prompt_classes"] == {"short": ["bare", "prompted"], "long_si": ["constitution"]}
    assert doc["max_prompt_tokens_longsi"] == max(constitution) > 4 * doc["max_prompt_tokens"]
    assert doc["max_prompt_argmax_longsi"]["class"] == "constitution"
    assert doc["system_prompt_sha256_by_class"]["constitution"] == IC.IFEVAL_CONSTITUTION_SI_SHA256
    assert doc["system_prompt_chars_by_class"] == {"prompted": len(SI), "constitution": len(CONST_SI)}
    for key, e in doc["by_model"].items():
        assert e["classes"]["constitution"]["lengths"] == constitution
        assert e["constitution_si_delta"] == {"min": len(CONST_SI) // 4, "max": len(CONST_SI) // 4}
    for cap in IC.IFEVAL_CAP_LADDER:
        en = doc["entries_longsi"][str(cap)]
        assert en["name"] == f"capability:ifeval_longsi_{cap}" and en["max_model_len"] == sg.next_block(max(constitution) + cap)
        assert en["classes"] == ["constitution"] and en["max_prompt"] == max(constitution)
        assert en["max_model_len"] % sg.VLLM_BLOCK_SIZE == 0 and 0 <= en["headroom"] < sg.VLLM_BLOCK_SIZE
        assert set(en["starting_seqs_by_model"]) == set(IC.IFEVAL_MODELS)
    # a moved constitution text refuses the measurement (the pinned sha256 is asserted, never re-derived)
    monkeypatch.setattr(B, "system_prompt_for_cell",
                        lambda cell: SI if cell == IC.CELL_PROMPTED else CONST_SI + " " if cell == IC.CELL_PROMPTED_CONSTITUTION else None)
    with pytest.raises(SystemExit, match="sha256 is not the pinned"):
        G.measure(tree="b" * 40, loader=loader, renderer=renderer)


# --------------------------------------------------------------------------- the REAL pinned tokenizers (offline; skipped otherwise)
@contextmanager
def _offline():
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    for k in saved:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cached(hf: str, rev: str) -> bool:
    root = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    return (root / f"models--{hf.replace('/', '--')}" / "snapshots" / rev / "tokenizer_config.json").is_file()


@pytest.mark.skipif(not all(_cached(*IC.IFEVAL_MODEL_PINS[k]) for k in IC.IFEVAL_MODELS)
                    or not (Path.home() / ".cache/huggingface/hub/datasets--google--IFEval/snapshots" / IC.IFEVAL_DATASET_REVISION).is_dir(),
                    reason="the four pinned tokenizers + the IFEval dataset are not all in the local HF cache (offline)")
def test_the_committed_measurement_regenerates_byte_identically_on_the_real_pinned_tokenizers(G):
    """G-LENGTHS is REPRODUCIBLE: the committed configs/sdfbase/ifeval/geometry_measurement.json equals a fresh
    measurement at its recorded tree sha, byte for byte, and its three entries are next_block(max + cap) of it."""
    committed_path = REPO / IC.IFEVAL_GEOMETRY_MEASUREMENT_REL
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    with _offline():
        doc = G.measure(tree=committed["tree_sha"])
    assert G.render_measurement(doc) == committed_path.read_text(encoding="utf-8")
    for cap in IC.IFEVAL_CAP_LADDER:
        assert committed["entries"][str(cap)]["max_model_len"] == sg.next_block(committed["max_prompt_tokens"] + cap)
        assert committed["entries_longsi"][str(cap)]["max_model_len"] == sg.next_block(committed["max_prompt_tokens_longsi"] + cap)
        # the registered entries ARE the measurement's (both classes)
        assert sg.geometry_for(f"capability:ifeval_{cap}").max_model_len == committed["entries"][str(cap)]["max_model_len"]
        assert sg.geometry_for(f"capability:ifeval_longsi_{cap}").max_model_len == committed["entries_longsi"][str(cap)]["max_model_len"]
    # 6386 = the constitution prompt under gemma-4's template with the system message in the OpenAI parts form (the form
    # the served /v1/tokenize sees; the driver's live pass measured exactly this) — one more than the string form's 6385
    assert committed["max_prompt_tokens"] == 468 and committed["max_prompt_tokens_longsi"] == 6386
    assert committed["system_prompt_sha256_by_class"]["constitution"] == IC.IFEVAL_CONSTITUTION_SI_SHA256
    assert committed["system_prompt_chars_by_class"]["constitution"] == 30952
    assert committed["n"] == 541 and committed["dataset_revision"] == IC.IFEVAL_DATASET_REVISION
    assert {e["chat_template_hash"] for e in committed["by_model"].values()} == {"0a2c8073c878ab1d", "a4aee8afcf2e0711", "ab7813c3abdd9cb6", "b48c47f644389271"}



def test_the_artifact_renders_chat_content_in_the_served_paths_format_per_family():
    """The wave shakedown (2026-09-13): the driver's live /v1/tokenize pass over the served render was exactly +1 token
    on every gemma PROMPTED item vs the CPU artifact, because vLLM hands a multimodal template its messages in the
    OpenAI text-parts form and gemma-4's template renders a leading system message one token longer that way (Qwen's is
    invariant; the text templates — llama, nemotron_h — are rendered as strings by vLLM). The artifact now renders
    parts for the multimodal families, strings otherwise, by model key OR registry slug, and records the choice."""
    from sisteer.capeval.tinycap import contracts as C
    import importlib.util
    spec = importlib.util.spec_from_file_location("tinycap_oos_prompt_budget_fmt", REPO / "scripts" / "tinycap_oos_prompt_budget.py")
    B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    assert C.prompt_budget_content_format("gemma4e4b") == C.prompt_budget_content_format("qwen35_4b") == "openai"
    assert C.prompt_budget_content_format("llama8b") == C.prompt_budget_content_format("nemotron3nano4b") == "string"
    assert C.prompt_budget_content_format("gemma4e4b-sdfbase-it") == "openai"          # a registry slug resolves too
    assert C.prompt_budget_content_format("llama8b-f22") == "string"
    with pytest.raises(KeyError):
        C.prompt_budget_content_format("nope")
    msgs = [{"role": "system", "content": "SI"}, {"role": "user", "content": "u"}]
    assert B.content_in_format(msgs, "string") == msgs
    parts = B.content_in_format(msgs, "openai")
    assert parts == [{"role": "system", "content": [{"type": "text", "text": "SI"}]}, {"role": "user", "content": [{"type": "text", "text": "u"}]}]
    assert B.content_in_format(parts, "openai") == parts                                # already parts: untouched
    with pytest.raises(ValueError):
        B.content_in_format(msgs, "weird")

    class _Tok:
        seen = []
        def apply_chat_template(self, messages, **kw):
            _Tok.seen.append(messages)
            return [0] * (sum(1 for m in messages if isinstance(m.get("content"), list)) + 10)
    tok = _Tok()
    n_str = B.prompt_token_count(tok, msgs, C.THINKING_OFF)
    n_parts = B.prompt_token_count(tok, msgs, C.THINKING_OFF, content_format="openai")
    assert n_str == 10 and n_parts == 12 and isinstance(_Tok.seen[0][0]["content"], str) and isinstance(_Tok.seen[1][0]["content"], list)
