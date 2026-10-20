"""The thinking-ON prompt-budget artifact (``scripts/tinycap_oos_prompt_budget.py``
+ the ``contracts`` additions) — fit-support transfer, experiment 1. CPU only.

Covered on a SYNTHETIC tokenizer (no model download): the artifact is EXACTLY
``contracts.OOS_PROMPT_BUDGET_FIELDS`` with ``chat_template_hash ==
sisteer.models.chat_template_hash(tok)`` (the 16 hex ``capctrl_serving_up``
writes into ``endpoint.json``), the render is the DECLARED arm's
(``apply_chat_template(..., add_generation_prompt=True, **chat_template_kwargs(arm))``
— ``object_identity.header_ids_under_arm``'s call, every message list once),
the statistics are nearest-rank (the geometry measurement's convention); the
CLI writes a validated file from the ONE shared messages source and the pinned
loader; the reader refuses a torn or foreign artifact by name; the shared fit
check passes at the resolved window and fails at 10,000; the shared template
equality refuses a differing ``endpoint.json`` digest (and a missing one) with
the named REC. The REAL Qwen3.5-4B tokenizer runs offline from the local HF
cache when present (skipped with the reason otherwise): the render fact behind
``prompt_opens_think_on`` (ON header == ``QWEN35_HEADER_IDS_ON``) and the
budget's digest. The full 200-prompt render is opt-in (``TINYCAP_OOS_REAL=1``:
it loads two public datasets, minutes) — that number is the plan's C2 pre-check.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import types
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
# `--tree-sha` is the fit sidecar's PIN and the producer asserts it against its own checkout's HEAD: the
# tests pass the real HEAD, never a placeholder
HEAD_SHA = __import__("subprocess").run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
SCRIPT = REPO / C.OOS_PROMPT_BUDGET_SCRIPT
ROW = C.model_row(C.AUTHORISED_OOS_THINKING.model)
GSM8K = C.GSM8K_STANDARD_TASK


def _load(name: str, path: Path):
    """A script module by path, REGISTERED in ``sys.modules`` before it runs:
    a dataclass under ``from __future__ import annotations`` resolves its string
    annotations through ``sys.modules[cls.__module__]`` (the geometry-measure
    script carries several)."""
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
    return _load("tinycap_oos_prompt_budget_under_test", SCRIPT)


class _FakeTok:
    """``apply_chat_template(messages, add_generation_prompt, tokenize, **kw)``
    -> ids; records every call. Length = the messages' characters // 4 plus a
    header of 5 (``enable_thinking`` true — Qwen's open ``<think>\\n``) or 7
    (false — the closed empty block), so ON and OFF differ by exactly the two
    tokens the real template moves."""

    def __init__(self, template="{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}",
                 batch_encoding=False):
        self.chat_template = template
        self.calls = []
        self.batch_encoding = batch_encoding

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "add_generation_prompt": add_generation_prompt,
                           "tokenize": tokenize, "kwargs": dict(kw)})
        body = sum(len(m["content"]) for m in messages) // 4
        header = 5 if kw.get("enable_thinking") else 7
        ids = list(range(body + header))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)} if self.batch_encoding else ids


def _messages(n=C.GSM8K_PAIRED_N):
    return [[{"role": "system", "content": "FEWSHOT " * 40},
             {"role": "user", "content": f"Question: what is {i} + {i}?\nAnswer:" + "x" * i}]
            for i in range(n)]


def _expected_lengths(msgs, *, on=True):
    return [sum(len(m["content"]) for m in ml) // 4 + (5 if on else 7) for ml in msgs]


def _build(B, msgs=None, tok=None, **kw):
    msgs = _messages() if msgs is None else msgs
    tok = _FakeTok() if tok is None else tok
    args = {"arm": C.THINKING_ON, "model": ROW.hf, "model_slug": ROW.slug, "revision": ROW.rev,
            "cap": C.OOS_THINKING_MAX_TOKENS, "cap_task": GSM8K, "tree_sha": None}
    args.update(kw)
    return B.build_prompt_budget(msgs, tok, **args)


# --------------------------------------------------------------------------- #
# the artifact                                                                 #
# --------------------------------------------------------------------------- #


def test_the_artifact_is_the_contract_s_fields_under_the_declared_render_with_the_template_digest(B):
    from sisteer.models import chat_template_hash

    msgs, tok = _messages(), _FakeTok()
    doc = _build(B, msgs, tok)
    assert set(doc) == set(C.OOS_PROMPT_BUDGET_FIELDS)
    lengths = _expected_lengths(msgs, on=True)
    assert doc["lengths"] == lengths and doc["n"] == C.GSM8K_PAIRED_N == len(lengths)
    s = sorted(lengths)
    assert doc["max"] == max(lengths) and doc["p99"] == s[197] and doc["p50"] == s[99]
    assert doc["mean"] == round(sum(lengths) / len(lengths), 2)
    assert doc["cap"] == C.OOS_THINKING_MAX_TOKENS == C.oos_max_tokens_on("gsm8k_paired200") == 8192
    assert doc["cap_task"] == GSM8K == "inspect/gsm8k_paired200"
    assert doc["max_plus_cap"] == doc["max"] + doc["cap"]
    # the digest is sisteer.models.chat_template_hash — the first 16 hex of
    # sha256 over the template text, the bytes capctrl_serving_up hashes into endpoint.json
    assert doc["chat_template_hash"] == chat_template_hash(tok) \
        == hashlib.sha256(tok.chat_template.encode()).hexdigest()[:16]
    assert doc["arm"] == C.THINKING_ON and doc["model"] == ROW.hf and doc["revision"] == ROW.rev
    assert doc["model_slug"] == ROW.slug == "qwen35_4b" and doc["tree_sha"] is None
    assert doc["render"] == {"add_generation_prompt": True,
                             C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: True}}
    assert doc["render"][C.CHAT_TEMPLATE_KWARGS_KEY] == C.chat_template_kwargs(C.THINKING_ON)[C.CHAT_TEMPLATE_KWARGS_KEY]
    assert doc["messages_source"] == C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE
    assert doc["produced_by"]["script"] == C.OOS_PROMPT_BUDGET_SCRIPT
    assert doc["produced_by"]["tokenizer_loader"] == "sisteer.models.load_tokenizer"
    # the render: every message list ONCE, the full generation prompt, the arm's kwargs
    # passed explicitly (header_ids_under_arm's call shape) — never the pin, never tokenize=False
    assert len(tok.calls) == C.GSM8K_PAIRED_N
    assert [c["messages"] for c in tok.calls] == msgs
    assert all(c["add_generation_prompt"] is True and c["tokenize"] is True for c in tok.calls)
    assert all(c["kwargs"] == {C.ENABLE_THINKING_KEY: True} for c in tok.calls)
    # the artifact validates through the reader's own rule
    assert C.validate_oos_prompt_budget(doc) == doc
    # transformers 5.x answers tokenize=True with a BatchEncoding: same count
    assert _build(B, msgs, _FakeTok(batch_encoding=True))["lengths"] == lengths


def test_the_off_arm_renders_the_closed_block_and_a_template_less_tokenizer_refuses(B):
    msgs = _messages()
    off = _build(B, msgs, _FakeTok(), arm=C.THINKING_OFF, cap=512)
    assert off["lengths"] == [n + 2 for n in _expected_lengths(msgs, on=True)]      # H=7 vs H=5
    assert off["render"][C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    assert off["arm"] == C.THINKING_OFF and off["max_plus_cap"] == off["max"] + 512
    with pytest.raises(ValueError, match="thinking mode must be one of"):
        _build(B, msgs, _FakeTok(), arm=C.THINKING_PER_TASK)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="not the 16-hex digest"):
        _build(B, msgs, _FakeTok(template=None))
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="no messages to budget"):
        _build(B, [], _FakeTok())


def test_percentile_is_the_geometry_measurement_s_nearest_rank(B):
    G = _load("tinycap_geometry_measure_under_test", REPO / "scripts" / "tinycap_geometry_measure.py")
    for pool in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], list(range(1, 201)), [7], [3, 3, 9, 1]):
        for q in (0.5, 0.9, 0.99, 1.0):
            assert B.percentile(pool, q) == G.percentile(pool, q)
    assert B.percentile(list(range(1, 201)), 0.99) == 198 and B.percentile(list(range(1, 201)), 0.5) == 100
    with pytest.raises(ValueError):
        B.percentile([], 0.5)


# --------------------------------------------------------------------------- #
# the CLI                                                                      #
# --------------------------------------------------------------------------- #


def test_the_cli_writes_a_validated_artifact_from_the_shared_messages_and_the_pinned_loader(
        B, tmp_path, monkeypatch, capsys):
    """One messages source (``inspect_served.gsm8k_paired200_messages`` via the
    module's seam), one tokenizer loader (``sisteer.models.load_tokenizer`` at
    the registry row's pinned revision), the tuple's model by default, an atomic
    write, and the ON cap from the tuple / the OFF cap from the table."""
    msgs = _messages()
    seen = {}

    def fake_messages(repo_root):
        seen["repo_root"] = Path(repo_root)
        return msgs

    def fake_loader(slug):
        seen["slug"] = slug
        return C.model_row(slug), _FakeTok()

    monkeypatch.setattr(B, "_messages", fake_messages)
    monkeypatch.setattr(B, "_load_pinned_tokenizer", fake_loader)
    out = tmp_path / "budget" / C.OOS_PROMPT_BUDGET_FILENAME
    assert B.main(["--out", str(out), "--tree-sha", HEAD_SHA]) == 0
    assert seen["slug"] == C.AUTHORISED_OOS_THINKING.model == "qwen35_4b"       # the tuple's model by default
    assert seen["repo_root"] == B.REPO == REPO
    doc = C.read_oos_prompt_budget(out)
    assert doc["model"] == ROW.hf == "Qwen/Qwen3.5-4B" and doc["revision"] == ROW.rev
    assert doc["cap"] == 8192 and doc["cap_task"] == GSM8K and doc["arm"] == C.THINKING_ON
    assert doc["tree_sha"] == HEAD_SHA and doc["lengths"] == _expected_lengths(msgs)
    assert not out.with_name(out.name + ".tmp").exists()                        # written atomically
    printed = capsys.readouterr().out
    assert "lengths" not in printed.splitlines()[0] and f"wrote {out}" in printed
    # the OFF arm: the table's cap, the closed-block render
    from sisteer.capeval.tiers import inspect_served as isv
    off = tmp_path / "off.json"
    assert B.main(["--model", ROW.slug, "--arm", C.THINKING_OFF, "--out", str(off)]) == 0
    d = C.read_oos_prompt_budget(off)
    assert d["cap"] == isv.MAX_TOKENS_OFF["gsm8k_paired200"] == 512 and d["tree_sha"] is None
    assert d["render"][C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    # --offline pins both caches to local files only (the preflight's mode) FOR THE CALL —
    # the loader sees the flags; the process does not keep them (the dedicated test below)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    seen_env = {}

    def fake_loader_recording(model):
        seen_env.update({k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
        return fake_loader(model)

    monkeypatch.setattr(B, "_load_pinned_tokenizer", fake_loader_recording)
    assert B.main(["--out", str(tmp_path / "o.json"), "--offline"]) == 0
    assert seen_env == {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    assert "HF_HUB_OFFLINE" not in os.environ and "HF_DATASETS_OFFLINE" not in os.environ
    # an unknown slug refuses through the registry, never a silent default
    with pytest.raises(C.UnknownModel):
        B.main(["--model", "nope", "--out", str(tmp_path / "x.json")])
    # the cap seam: ON is the tuple's raised cap, OFF the table's
    assert B.cap_for(C.THINKING_ON) == (C.OOS_THINKING_MAX_TOKENS, GSM8K)
    assert B.cap_for(C.THINKING_OFF) == (512, GSM8K)


def test_the_reader_refuses_a_torn_or_foreign_artifact_by_name(B, tmp_path):
    good = _build(B)

    def refused(mutate, match):
        doc = json.loads(json.dumps(good))
        mutate(doc)
        p = tmp_path / "a.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(C.OutOfSupportPromptBudgetError, match=match):
            C.read_oos_prompt_budget(p)

    refused(lambda d: d.pop("p99"), r"missing \['p99'\]")
    refused(lambda d: d.update(extra=1), r"unknown \['extra'\]")
    refused(lambda d: d.update(max=d["max"] + 1), r"max .* != max\(lengths\)")
    refused(lambda d: d.update(n=199), r"n 199 != len\(lengths\) 200")
    refused(lambda d: d.update(max_plus_cap=1), r"max_plus_cap 1 != max")
    refused(lambda d: d.update(chat_template_hash="xyz"), "not the 16-hex digest")
    refused(lambda d: d.update(chat_template_hash=None), "not the 16-hex digest")
    refused(lambda d: d.update(arm=C.THINKING_PER_TASK), "arm 'per-task' is not one of")
    refused(lambda d: d.update(cap=True), "cap must be a non-negative integer")
    refused(lambda d: d.update(lengths=[]), "non-empty list")
    refused(lambda d: d.update(model=""), "model must be a non-empty string")
    p = tmp_path / "torn.json"
    p.write_text("{torn")
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="unreadable"):
        C.read_oos_prompt_budget(p)
    p.write_text("[1, 2]")
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="expected a JSON object"):
        C.read_oos_prompt_budget(p)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="unreadable"):
        C.read_oos_prompt_budget(tmp_path / "absent.json")
    assert C.read_oos_prompt_budget(_write(tmp_path, good)) == good


def _write(tmp_path, doc, name="ok.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


# --------------------------------------------------------------------------- #
# the two shared checks (preflight, worker S4.3, driver S3.2)                  #
# --------------------------------------------------------------------------- #


def test_the_fit_check_passes_at_the_resolved_window_and_fails_at_10000(B):
    L = sg.PETRI_MAX_MODEL_LEN                     # what --pass1 resolves (pass1_geometry), never typed
    doc = _build(B, [[{"role": "user", "content": "x" * 8000}]] + _messages(199))   # max = 2005
    prompt_max = doc["max"]
    rec = C.prompt_budget_fit(doc, arm=C.THINKING_ON, cap=8192, max_model_len=L)
    assert rec == {"prompt_max": prompt_max, "cap": 8192, "max_model_len": L,
                   "total": prompt_max + 8192, "margin": L - prompt_max - 8192,
                   "p99": doc["p99"], "chat_template_hash": doc["chat_template_hash"]}
    assert rec["margin"] > 0
    # the boundary fits with margin 0; one over refuses naming the three numbers
    assert C.prompt_budget_fit(doc, arm="on", cap=8192, max_model_len=prompt_max + 8192)["margin"] == 0
    with pytest.raises(C.OutOfSupportPromptBudgetError,
                       match=rf"prompt max {prompt_max} \+ cap 8192 = {prompt_max + 8192} > the resolved "
                             rf"max_model_len {prompt_max + 8191} \(over by 1\)"):
        C.prompt_budget_fit(doc, arm="on", cap=8192, max_model_len=prompt_max + 8191)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="> the resolved max_model_len 10000"):
        C.prompt_budget_fit(doc, arm="on", cap=8192, max_model_len=10000)
    # the artifact's OWN cap must be the run's (one 8192 — never three different ones)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="written for cap 8192 but the run's cap is 2048"):
        C.prompt_budget_fit(doc, arm="on", cap=2048, max_model_len=L)
    # and the arm the run serves
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="tokenized the 'on' render but the run serves the 'off'"):
        C.prompt_budget_fit(doc, arm=C.THINKING_OFF, cap=8192, max_model_len=L)


def test_the_template_equality_refuses_a_differing_or_missing_served_digest_with_the_named_rec(B, tmp_path):
    """S3.2 / S4.3 / S4.6 share ONE equality: the artifact's digest against the
    served one (``endpoint.json.chat_template_hash`` after standup; the
    object-identity report's at preflight). A synthetic endpoint.json whose
    digest differs is refused naming both; a null one is refused too (E50 #6:
    an unmade comparison never passes); the caller files the REC."""
    doc = _build(B)
    mine = doc["chat_template_hash"]
    ep = tmp_path / "endpoint.json"
    ep.write_text(json.dumps({"proxy_url": "http://127.0.0.1:1/v1", "chat_template_hash": mine}))
    served = json.loads(ep.read_text())["chat_template_hash"]
    assert C.prompt_budget_template_mismatch(doc, served, served_source="endpoint.json") is None
    other = "0" * 16
    ep.write_text(json.dumps({"chat_template_hash": other}))
    why = C.prompt_budget_template_mismatch(doc, json.loads(ep.read_text())["chat_template_hash"],
                                            served_source="the standup's endpoint.json")
    assert why and mine in why and other in why and "nobody is serving" in why
    assert C.OOS_PROMPT_BUDGET_SCRIPT in why
    # a standup that recorded no digest: refused, naming the side that was unavailable
    ep.write_text(json.dumps({"proxy_url": "x"}))
    why = C.prompt_budget_template_mismatch(doc, json.loads(ep.read_text()).get("chat_template_hash"),
                                            served_source="the standup's endpoint.json")
    assert why and "the standup's endpoint.json carries no 16-hex chat_template_hash (None)" in why
    assert "never ran" in why
    # an artifact without a digest cannot bind to anything
    assert "carries no 16-hex chat_template_hash" in C.prompt_budget_template_mismatch(
        {**doc, "chat_template_hash": "short"}, mine, served_source="x")
    assert C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH == "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH"
    assert C.OOS_PROMPT_BUDGET_FILENAME == "oos_prompt_budget.json"
    assert SCRIPT.is_file() and C.OOS_PROMPT_BUDGET_SCRIPT == "scripts/tinycap_oos_prompt_budget.py"


# --------------------------------------------------------------------------- #
# the REAL pinned tokenizer (offline from the local HF cache; skipped otherwise) #
# --------------------------------------------------------------------------- #


@contextmanager
def _offline():
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def qwen_tok():
    """The pinned Qwen3.5-4B tokenizer through the loader the derivation uses,
    OFFLINE — present in the local cache or the test is skipped with the reason
    (never a download here)."""
    from sisteer.models import load_tokenizer

    with _offline():
        try:
            return load_tokenizer(ROW.hf, revision=ROW.rev)
        except Exception as e:                                          # noqa: BLE001
            pytest.skip(f"{ROW.hf}@{ROW.rev[:12]} tokenizer not in the local HF cache "
                        f"(offline): {type(e).__name__}: {str(e)[:200]}")


def test_real_qwen35_4b_render_fact_and_budget_digest(B, qwen_tok):
    """On the pinned tokenizer: the ON header is the 5-id open-think sequence
    the contract pins (the render fact ``prompt_opens_think_on`` rests on), the
    OFF header the 7-id closed block; the budget's render is the ON one (two
    tokens shorter than OFF on the same messages, ending in the ON header);
    the artifact's digest is the loader's template digest, 16 hex."""
    from sisteer.capeval.tinycap.object_identity import header_ids_under_arm
    from sisteer.models import chat_template_hash

    assert header_ids_under_arm(qwen_tok, C.THINKING_ON) == C.QWEN35_HEADER_IDS_ON
    assert header_ids_under_arm(qwen_tok, C.THINKING_OFF) == C.QWEN35_HEADER_IDS_OFF
    assert C.QWEN35_HEADER_IDS_ON[-2:] == (C.QWEN35_THINK_OPEN_ID, 198)
    msgs = [[{"role": "system", "content": "Solve the problem."},
             {"role": "user", "content": f"Question: what is {i} + {i}?\nAnswer:"}] for i in range(3)]
    on = _build(B, msgs, qwen_tok)
    off = _build(B, msgs, qwen_tok, arm=C.THINKING_OFF, cap=512)
    assert [a + 2 for a in on["lengths"]] == off["lengths"]                     # H=5 vs H=7
    kw = C.chat_template_kwargs(C.THINKING_ON)[C.CHAT_TEMPLATE_KWARGS_KEY]
    ids = qwen_tok.apply_chat_template(msgs[0], add_generation_prompt=True, tokenize=True, **kw)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    assert len(ids) == on["lengths"][0] and tuple(ids[-5:]) == C.QWEN35_HEADER_IDS_ON
    assert C.QWEN35_THINK_CLOSE_ID not in ids[-5:]
    digest = chat_template_hash(qwen_tok)
    assert on["chat_template_hash"] == digest and len(digest) == 16
    assert C.prompt_budget_template_mismatch(on, digest, served_source="x") is None


@pytest.fixture(scope="module")
def real_messages():
    """The 200 exact message lists — opt-in (``TINYCAP_OOS_REAL=1``): rendering
    them loads openai/gsm8k and gsm8k-platinum (public; minutes)."""
    if os.environ.get("TINYCAP_OOS_REAL") != "1":
        pytest.skip("set TINYCAP_OOS_REAL=1 to render the 200 real GSM8K prompts (loads two "
                    "public datasets from the HF cache / hub; minutes)")
    from sisteer.capeval.tiers import inspect_served as isv

    return isv.gsm8k_paired200_messages(REPO)


def test_real_prompt_budget_fits_the_resolved_window_with_the_8192_cap(B, qwen_tok, real_messages):
    """The plan's C2 pre-check number: on the REAL 200 prompts under the REAL
    pinned tokenizer, ``max + 8192 <= PETRI_MAX_MODEL_LEN`` (what ``--pass1``
    resolves). Also the messages' shape: one 8-shot system turn shared by every
    item, one templated user turn per item."""
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE

    assert len(real_messages) == C.GSM8K_PAIRED_N
    assert all([m["role"] for m in ml] == ["system", "user"] for ml in real_messages)
    assert len({ml[0]["content"] for ml in real_messages}) == 1                  # one few-shot block
    assert all(ml[1]["content"].endswith(MATH_PROMPT_TEMPLATE.format(prompt="")[-30:])
               for ml in real_messages)
    doc = _build(B, real_messages, qwen_tok)
    rec = C.prompt_budget_fit(doc, arm=C.THINKING_ON, cap=C.OOS_THINKING_MAX_TOKENS,
                              max_model_len=sg.PETRI_MAX_MODEL_LEN)
    print(f"[C2 pre-check] n {doc['n']} max {doc['max']} p99 {doc['p99']} p50 {doc['p50']} "
          f"mean {doc['mean']} template {doc['chat_template_hash']} total {rec['total']} "
          f"margin {rec['margin']} at L {sg.PETRI_MAX_MODEL_LEN}")
    assert rec["margin"] >= 0


# --------------------------------------------------------------------------- #
# offcap: the comparator's OFF artifact at the OVERRIDE cap                          #
# (`--off-cap-override`) — the cap trap closed at the SOURCE                         #
# --------------------------------------------------------------------------- #


def test_the_off_arm_with_the_override_writes_the_override_cap_and_refuses_it_under_on(B, tmp_path, monkeypatch):
    """offcap §5.2: `cap_for("off", off_cap_override=True)` is the comparator's
    raised cap (contracts.off_cap_override_max_tokens: 8192) with the row task;
    without it the table's 512 (unchanged); under `on` the flag refuses (an OFF
    fact). The CLI flag `--off-cap-override` (contracts.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE)
    threads to `cap_for`, so the artifact reads `arm off, cap 8192, max_plus_cap
    max + 8192` and `contracts.prompt_budget_fit(artifact, arm="off", cap=8192,
    max_model_len=16384)` PASSES — a stale 512 artifact refuses by name."""
    assert C.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE == "--off-cap-override"
    assert B.cap_for(C.THINKING_OFF, off_cap_override=True) == (C.OFF_CAP_OVERRIDE_MAX_TOKENS, GSM8K) == (8192, GSM8K)
    assert B.cap_for(C.THINKING_OFF) == (512, GSM8K) == B.cap_for(C.THINKING_OFF, off_cap_override=False)
    assert B.cap_for(C.THINKING_ON) == (C.OOS_THINKING_MAX_TOKENS, GSM8K)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="OFF"):
        B.cap_for(C.THINKING_ON, off_cap_override=True)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="names no cap"):
        B.cap_for(C.THINKING_OFF, "math", off_cap_override=True)
    msgs = _messages()
    monkeypatch.setattr(B, "_messages", lambda repo_root: msgs)
    monkeypatch.setattr(B, "_load_pinned_tokenizer", lambda slug: (C.model_row(slug), _FakeTok()))
    out = tmp_path / "off_override.json"
    assert B.main(["--model", ROW.slug, "--arm", C.THINKING_OFF, "--off-cap-override", "--out", str(out),
                   "--tree-sha", HEAD_SHA]) == 0
    doc = C.read_oos_prompt_budget(out)
    assert doc["arm"] == C.THINKING_OFF and doc["cap"] == 8192 and doc["cap_task"] == GSM8K
    assert doc["max_plus_cap"] == doc["max"] + 8192 and doc["lengths"] == _expected_lengths(msgs, on=False)
    assert doc["render"][C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    rec = C.prompt_budget_fit(doc, arm=C.THINKING_OFF, cap=8192, max_model_len=sg.PETRI_MAX_MODEL_LEN)
    assert rec["margin"] == sg.PETRI_MAX_MODEL_LEN - doc["max"] - 8192 > 0
    # the same run WITHOUT the flag: the table's cap, and the fit against 8192 refuses by name
    stale = tmp_path / "off_table.json"
    assert B.main(["--model", ROW.slug, "--arm", C.THINKING_OFF, "--out", str(stale)]) == 0
    d = C.read_oos_prompt_budget(stale)
    assert d["cap"] == 512 and d["lengths"] == doc["lengths"]
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="written for cap 512 but the run's cap is 8192"):
        C.prompt_budget_fit(d, arm=C.THINKING_OFF, cap=8192, max_model_len=sg.PETRI_MAX_MODEL_LEN)
    # the flag under --arm on refuses at the CLI
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="OFF"):
        B.main(["--model", ROW.slug, "--arm", C.THINKING_ON, "--off-cap-override", "--out", str(tmp_path / "x.json")])


def test_real_off_render_budget_fits_the_resolved_window_with_the_8192_cap(B, qwen_tok, real_messages):
    """On the REAL 200 prompts under the REAL pinned tokenizer's OFF render,
    n 200, max 2375, p99 2347, template digest a4aee8afcf2e0711 (the same
    template as the ON artifact's — the digest hashes the template, not the
    render), `enable_thinking: false`; 2375 + 8192 = 10567 <= 16384, margin
    5817."""
    doc = _build(B, real_messages, qwen_tok, arm=C.THINKING_OFF, cap=C.OFF_CAP_OVERRIDE_MAX_TOKENS)
    assert (doc["n"], doc["max"], doc["p99"]) == (200, 2375, 2347)
    assert doc["chat_template_hash"] == "a4aee8afcf2e0711"
    assert doc["render"][C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    rec = C.prompt_budget_fit(doc, arm=C.THINKING_OFF, cap=C.OFF_CAP_OVERRIDE_MAX_TOKENS,
                              max_model_len=sg.PETRI_MAX_MODEL_LEN)
    assert (rec["total"], rec["margin"]) == (10567, sg.PETRI_MAX_MODEL_LEN - 10567)
    print(f"[offcap D2] n {doc['n']} max {doc['max']} p99 {doc['p99']} p50 {doc['p50']} mean {doc['mean']} "
          f"template {doc['chat_template_hash']} total {rec['total']} margin {rec['margin']}")


# --------------------------------------------------------------------------- #
# The MATH-500 thinking-ON generalisation: the artifact stays contract-exact,  #
# the fit rides a sidecar                                                      #
# --------------------------------------------------------------------------- #


def _math_msgs(n=500, system=None):
    out = []
    for i in range(n):
        m = [{"role": "user", "content": f"PROBLEM: what is {i} + {i}? " + "x" * (i % 37)}]
        if system:
            m = [{"role": "system", "content": system}] + m
        out.append(m)
    return out


def test_math500_artifact_is_contract_exact_with_the_math_cap_and_the_fit_rides_a_sidecar(B):
    msgs, tok = _math_msgs(), _FakeTok()
    doc = B.build_prompt_budget(msgs, tok, arm=C.THINKING_ON, model="Qwen/Qwen3.5-4B", model_slug="qwen35_4b",
                                revision="8" * 40, cap=B.MATH500_THINKING_CAP, cap_task=C.inspect_task("math"),
                                messages_source=B.MATH500_MESSAGES_SOURCE)
    assert set(doc) == set(C.OOS_PROMPT_BUDGET_FIELDS)            # NO extra key: both consumers read it unchanged
    assert doc["cap"] == 16384 == C.MATH500_MAX_TOKENS_DEFAULT == B.MATH500_THINKING_CAP
    assert doc["cap_task"] == "inspect/math" and doc["n"] == 500
    assert doc["messages_source"] == B.MATH500_MESSAGES_SOURCE and "math500_messages" in doc["messages_source"]
    assert C.validate_oos_prompt_budget(doc) == doc
    side = B.math500_fit_sidecar(doc, geometry_eval="capability:gen_long_math", max_model_len=17664,
                                 system_prompt_cell=None, system_prompt=None)
    assert side["fits"] is True and side["why_not"] is None and side["window"] == 17664 and side["cap"] == 16384
    assert side["margin"] == 17664 - doc["max"] - 16384 and side["task"] == "math" and side["control"] == "math500"
    assert side["system_prompt_cell"] is None and side["system_prompt_sha16"] is None
    # a window too small for max + cap: fits False with the contract's own reason, never an exception here
    tight = B.math500_fit_sidecar(doc, geometry_eval="x", max_model_len=16384 + doc["max"] - 1,
                                  system_prompt_cell="prompted__safety_refusal", system_prompt="SI text")
    assert tight["fits"] is False and "over by 1" in tight["why_not"] and tight["margin"] == -1
    assert tight["system_prompt_sha16"] == hashlib.sha256(b"SI text").hexdigest()[:16]


def test_math500_geometry_window_is_the_registered_entry_and_the_system_prompt_is_the_cells():
    B = _load("tinycap_oos_prompt_budget_math_under_test", SCRIPT)
    assert B.geometry_window("capability:gen_long_thinking") == 10752
    # the default entry is MATH-500's OWN (17,664), derived from the contracts' class map, never typed
    assert B.MATH500_GEOMETRY_EVAL == "capability:gen_long_math" and B.geometry_window(B.MATH500_GEOMETRY_EVAL) == 17664
    with pytest.raises(C.OutOfSupportPromptBudgetError):
        B.geometry_window("capability:no_such_entry")
    from sisteer.eval.jailbreak import resolve_instruction
    assert B.system_prompt_for_cell("prompted__safety_refusal") == resolve_instruction("safety_refusal").text
    assert B.system_prompt_for_cell("none") is None
    assert B.system_prompt_for_cell("constv2-mix-full__lora_r4_ce_minus_ctl") is None


def test_math500_messages_render_through_the_tasks_own_solver_with_the_system_turn_prepended(monkeypatch):
    """The renderer is gsm8k_paired200_messages' construction on the `math` plan: sample messages through the
    task's solver chain with generate a no-op; the prompted cell's SI is the ONE system turn."""
    pytest.importorskip("inspect_ai")  # the Modal CPU-suite image has no inspect_ai; the lane venvs do
    B = _load("tinycap_oos_prompt_budget_math_under_test2", SCRIPT)
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import solver
    from sisteer.capeval.tiers import inspect_served as isv

    @solver
    def wrap():
        async def solve(state, generate):
            state.messages[-1].content = "TEMPLATED: " + state.messages[-1].content
            return state
        return solve

    class _Task:
        def __init__(self, samples):
            self.dataset = samples
            self.solver = wrap()

    samples = [Sample(id=f"math_{i:08d}", input=f"problem {i}", target=str(i)) for i in range(500)]
    plan = types.SimpleNamespace(task=_Task(samples), sample_ids=[s.id for s in samples])
    seen = {}

    def fake_plan_task(name, **kw):
        seen.update(name=name, **kw)
        return plan
    monkeypatch.setattr(isv, "plan_task", fake_plan_task)
    msgs = B.math500_messages(REPO, system_prompt=None)
    assert seen["name"] == "math" and seen["thinking"] == C.THINKING_OFF and seen["grader_model"] is None
    assert len(msgs) == 500 and msgs[7] == [{"role": "user", "content": "TEMPLATED: problem 7"}]
    with_si = B.math500_messages(REPO, system_prompt="SI")
    assert with_si[0] == [{"role": "system", "content": "SI"}, {"role": "user", "content": "TEMPLATED: problem 0"}]
    # a short plan refuses (a "MATH-500" budget over 486 items is not MATH-500)
    plan.sample_ids = plan.sample_ids[:486]
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="not 500"):
        B.math500_messages(REPO)


def test_math500_cli_writes_the_artifact_and_the_sidecar_and_refuses_a_non_fit(tmp_path, monkeypatch, capsys):
    B = _load("tinycap_oos_prompt_budget_math_under_test3", SCRIPT)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: Qwen/Qwen3.5-4B\nmodel_revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a\nexplicit_cells: [none]\n")
    tok = _FakeTok()
    monkeypatch.setattr(B, "_load_config_tokenizer", lambda path, key: ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", key, tok))
    monkeypatch.setattr(B, "math500_messages", lambda root, system_prompt=None: _math_msgs(system=system_prompt))
    out = tmp_path / "budgets" / "qwen35_4b" / "none" / C.OOS_PROMPT_BUDGET_FILENAME
    rc = B.main(["--task", "math500", "--arm", "on", "--config", str(cfg), "--model-key", "qwen35_4b",
                 "--out", str(out), "--tree-sha", HEAD_SHA])
    assert rc == 0, capsys.readouterr().out
    doc = C.read_oos_prompt_budget(out)                           # the reader's own rule accepts it
    assert doc["cap"] == 16384 and doc["model_slug"] == "qwen35_4b" and doc["cap_task"] == "inspect/math"
    side = json.loads((tmp_path / "budgets" / "qwen35_4b" / "none" / "oos_prompt_budget.json.fit.json").read_text())
    assert side["fits"] is True and side["window"] == 17664 and side["geometry_eval"] == "capability:gen_long_math"
    # the prompted class carries the SI as the system turn — recorded by cell and sha
    out2 = tmp_path / "budgets" / "qwen35_4b" / "prompted__safety_refusal" / C.OOS_PROMPT_BUDGET_FILENAME
    rc = B.main(["--task", "math500", "--arm", "on", "--config", str(cfg), "--model-key", "qwen35_4b",
                 "--system-prompt-cell", "prompted__safety_refusal", "--out", str(out2)])
    assert rc == 0
    side2 = json.loads((out2.parent / "oos_prompt_budget.json.fit.json").read_text())
    assert side2["system_prompt_cell"] == "prompted__safety_refusal" and side2["system_prompt_sha16"]
    assert C.read_oos_prompt_budget(out2)["max"] > doc["max"]     # the system turn costs tokens
    # a cap the window cannot hold: exit 2, the sidecar says why
    rc = B.main(["--task", "math500", "--arm", "on", "--config", str(cfg), "--model-key", "qwen35_4b",
                 "--out", str(tmp_path / "big.json"), "--cap", "17664"])
    assert rc == 2 and json.loads((tmp_path / "big.json.fit.json").read_text())["fits"] is False
    # the OFF arm and the comparator's override are refused with --task math500
    with pytest.raises(SystemExit):
        B.main(["--task", "math500", "--arm", "off", "--config", str(cfg), "--model-key", "qwen35_4b", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        B.main(["--task", "math500", "--arm", "on", "--config", str(cfg), "--model-key", "qwen35_4b", "--off-cap-override",
                "--out", str(tmp_path / "y.json")])
    with pytest.raises(SystemExit):
        B.main(["--task", "math500", "--arm", "on", "--out", str(tmp_path / "z.json")])          # no config / key
    # a NO-SWITCH family's MATH-500 rows are `not_available` — admitted with an EXPLICIT --cap
    # (the GSM8K general shape's rule), rendered with no chat_template_kwargs, the sidecar's `arm` the launcher's check reads;
    # without --cap refused by name (no cap table for a no-switch family); --arm off still refused
    out3 = tmp_path / "budgets" / "llama8b" / "none" / C.OOS_PROMPT_BUDGET_FILENAME
    rc = B.main(["--task", "math500", "--arm", C.THINKING_NOT_AVAILABLE, "--cap", "16384", "--config", str(cfg), "--model-key", "qwen35_4b",
                 "--out", str(out3), "--tree-sha", HEAD_SHA, "--geometry-eval", "capability:gen_long_math"])
    assert rc == 0, capsys.readouterr().out
    doc3 = C.read_oos_prompt_budget(out3)
    assert doc3["arm"] == C.THINKING_NOT_AVAILABLE and doc3["cap"] == 16384 and doc3["cap_task"] == "inspect/math" and doc3["tree_sha"] == HEAD_SHA
    side3 = json.loads((out3.parent / "oos_prompt_budget.json.fit.json").read_text())
    assert side3["fits"] is True and side3["arm"] == C.THINKING_NOT_AVAILABLE and side3["cap"] == 16384 and side3["window"] == 17664
    assert side3["geometry_eval"] == "capability:gen_long_math" and side3["system_prompt_cell"] is None and side3["system_prompt_sha16"] is None
    assert B.render_kwargs(C.THINKING_NOT_AVAILABLE) == {}                     # no switch is sent to a no-switch family
    with pytest.raises(SystemExit, match="pass --cap explicitly"):
        B.main(["--task", "math500", "--arm", C.THINKING_NOT_AVAILABLE, "--config", str(cfg), "--model-key", "qwen35_4b", "--out", str(tmp_path / "w.json")])
    with pytest.raises(SystemExit, match="pass --arm on"):
        B.main(["--task", "math500", "--arm", "off", "--cap", "16384", "--config", str(cfg), "--model-key", "qwen35_4b", "--out", str(tmp_path / "v.json")])


# --------------------------------------------------------------------------- #
# The GENERAL gsm8k_paired200 shape — both arms, the cap
# explicit, the prompted cell's SI folded; NO sidecar without --geometry-eval (G-LENGTHS), the sidecar with it (G-FIT)
# --------------------------------------------------------------------------- #


def _g8k_msgs(n=200):
    return [[{"role": "system", "content": "FEWSHOT " * 300},
             {"role": "user", "content": f"Question: what is {i} + {i}?\nAnswer:" + "x" * (i % 41)}] for i in range(n)]


def test_gsm8k_general_shape_lengths_only_writes_no_sidecar_and_the_fit_pass_refuses_a_small_entry_by_name(tmp_path, monkeypatch, capsys):
    B = _load("tinycap_oos_prompt_budget_g8k_under_test", SCRIPT)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: Qwen/Qwen3.5-4B\nmodel_revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a\nexplicit_cells: [none]\n")
    tok = _FakeTok()
    monkeypatch.setattr(B, "_load_config_tokenizer", lambda path, key: ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", key, tok))
    seen = {}

    def fake_class_messages(root, *, system_prompt=None):
        seen["system_prompt"] = system_prompt
        msgs = _g8k_msgs()
        if system_prompt:
            from sisteer.capeval.tiers.inspect_served import collapse_system_messages_dicts
            msgs = [collapse_system_messages_dicts(m, system_prompt) for m in msgs]
        return msgs
    monkeypatch.setattr(B, "gsm8k_paired200_class_messages", fake_class_messages)
    monkeypatch.setattr(B, "system_prompt_for_cell", lambda cell: "SAFETY SI" if cell == "prompted__safety_refusal" else None)
    base = ["--task", "gsm8k_paired200", "--config", str(cfg), "--model-key", "qwen35_4b"]
    # (1) G-LENGTHS: the ON arm at the explicit rung cap, NO --geometry-eval -> the artifact, no sidecar, exit 0
    out = tmp_path / "lengths" / "none" / "on_16384" / C.OOS_PROMPT_BUDGET_FILENAME
    assert B.main([*base, "--arm", "on", "--cap", "16384", "--out", str(out), "--tree-sha", HEAD_SHA]) == 0
    doc = C.read_oos_prompt_budget(out)
    assert doc["cap"] == 16384 and doc["cap_task"] == "inspect/gsm8k_paired200" and doc["arm"] == "on" and doc["n"] == 200
    assert doc["messages_source"] == C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE and doc["tree_sha"] == HEAD_SHA
    assert not Path(str(out) + ".fit.json").exists()
    assert "NO fit sidecar" in capsys.readouterr().out
    # the cap defaults to the arm's table / tuple cap: 8192 ON (the parent tuple's), 512 OFF (the table's)
    out_on = tmp_path / "on_default.json"
    assert B.main([*base, "--arm", "on", "--out", str(out_on)]) == 0 and C.read_oos_prompt_budget(out_on)["cap"] == 8192
    out_off = tmp_path / "off_default.json"
    assert B.main([*base, "--arm", "off", "--out", str(out_off)]) == 0
    off = C.read_oos_prompt_budget(out_off)
    assert off["cap"] == 512 and off["arm"] == "off" and off["max"] == doc["max"] + 2       # the fake's OFF header is 2 tokens longer
    # (2) the prompted class folds the SI into the leading system turn (ONE system message; the render the route serves)
    out_p = tmp_path / "prompted.json"
    assert B.main([*base, "--arm", "on", "--cap", "8192", "--system-prompt-cell", "prompted__safety_refusal", "--out", str(out_p)]) == 0
    assert seen["system_prompt"] == "SAFETY SI"
    folded = tok.calls[-1]["messages"]
    assert [m["role"] for m in folded] == ["system", "user"] and folded[0]["content"].startswith("SAFETY SI\n\nFEWSHOT")
    pdoc = C.read_oos_prompt_budget(out_p)
    assert pdoc["max"] > C.read_oos_prompt_budget(out_on)["max"] and "folded" in pdoc["messages_source"]
    # (3) G-FIT: with --geometry-eval the sidecar rides beside the artifact, names the entry, the pin and fits
    out_fit = tmp_path / "fit" / C.OOS_PROMPT_BUDGET_FILENAME
    assert B.main([*base, "--arm", "on", "--cap", "8192", "--geometry-eval", "capability:gen_long_thinking",
                   "--out", str(out_fit), "--tree-sha", HEAD_SHA]) == 0
    side = json.loads((tmp_path / "fit" / "oos_prompt_budget.json.fit.json").read_text())
    assert side["fits"] is True and side["window"] == 10752 and side["cap"] == 8192 and side["geometry_eval"] == "capability:gen_long_thinking"
    assert side["task"] == "gsm8k_paired200" and side["control"] == "gsm8k" and side["tree_sha"] == HEAD_SHA and side["arm"] == "on"
    assert side["margin"] == 10752 - side["prompt_max"] - 8192
    # the rung against its own entry; the OFF overlap leg against the OFF class's entry
    out_r = tmp_path / "rung" / C.OOS_PROMPT_BUDGET_FILENAME
    assert B.main([*base, "--arm", "on", "--cap", "16384", "--geometry-eval", "capability:gen_long_gsm8k_rung", "--out", str(out_r)]) == 0
    assert json.loads((out_r.parent / "oos_prompt_budget.json.fit.json").read_text())["window"] == 18944
    out_o = tmp_path / "overlap" / C.OOS_PROMPT_BUDGET_FILENAME
    assert B.main([*base, "--arm", "off", "--cap", "512", "--geometry-eval", "capability:gen_long", "--out", str(out_o)]) == 0
    assert json.loads((out_o.parent / "oos_prompt_budget.json.fit.json").read_text())["window"] == 4608
    # (4) a non-fitting entry REFUSES by name (exit 2), the sidecar says why; an unknown entry refuses before writing one
    out_bad = tmp_path / "bad" / C.OOS_PROMPT_BUDGET_FILENAME
    rc = B.main([*base, "--arm", "on", "--cap", "8192", "--geometry-eval", "capability:short", "--out", str(out_bad)])
    bad = json.loads((out_bad.parent / "oos_prompt_budget.json.fit.json").read_text())
    assert rc == 2 and bad["fits"] is False and bad["window"] == 1536 and "over by" in bad["why_not"] and bad["geometry_eval"] == "capability:short"
    assert "REFUSED" in capsys.readouterr().out
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="no geometry entry named"):
        B.main([*base, "--arm", "on", "--cap", "8192", "--geometry-eval", "capability:no_such", "--out", str(tmp_path / "x.json")])
    # the general shape needs BOTH --config and --model-key; the comparator's override is not its fact
    with pytest.raises(SystemExit, match="needs --config"):
        B.main(["--task", "gsm8k_paired200", "--config", str(cfg), "--out", str(tmp_path / "y.json")])
    with pytest.raises(SystemExit, match="off-cap-override"):
        B.main([*base, "--arm", "off", "--off-cap-override", "--out", str(tmp_path / "z.json")])
    # the sidecar's shared builder: math500's keeps its task / control and gains tree_sha
    mside = B.math500_fit_sidecar(doc, geometry_eval="capability:gen_long_math", max_model_len=17664, system_prompt_cell=None, system_prompt=None)
    assert mside["task"] == "math" and mside["control"] == "math500" and mside["tree_sha"] == HEAD_SHA


def test_gsm8k_class_messages_are_the_served_tasks_own_renders_with_the_si_folded(monkeypatch):
    B = _load("tinycap_oos_prompt_budget_g8k_msgs_under_test", SCRIPT)
    from sisteer.capeval.tiers import inspect_served as isv
    base = [[{"role": "system", "content": "FEWSHOT"}, {"role": "user", "content": f"Q{i}"}] for i in range(3)]
    seen: list = []

    def fake_renderer(root, *, system=None):
        seen.append(system)
        rendered = [list(map(dict, m)) for m in base]
        return [isv.collapse_system_messages_dicts(m, system) for m in rendered] if system else rendered
    monkeypatch.setattr(isv, "gsm8k_paired200_messages", fake_renderer)
    assert B.gsm8k_paired200_class_messages(REPO) == base
    with_si = B.gsm8k_paired200_class_messages(REPO, system_prompt="SI")
    assert seen == [None, "SI"]   # the SI reaches the renderer's own fold; the wrapper folds nothing itself
    assert with_si == [isv.collapse_system_messages_dicts(m, "SI") for m in base]
    assert all(len(m) == 2 and m[0]["role"] == "system" and m[0]["content"] == "SI\n\nFEWSHOT" for m in with_si)


def test_the_default_task_is_gsm8k_and_its_artifact_is_unchanged(B):
    """`--task` defaults to the GSM8K instrument: the default argv is byte-identical to before this change."""
    assert B.build_parser().parse_args([]).task == B.TASK_GSM8K == "gsm8k_paired200"
    doc = _build(B)
    assert doc["messages_source"] == C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE and doc["cap_task"] == GSM8K


def test_main_offline_leaves_the_process_environment_unchanged(monkeypatch, tmp_path):
    """--offline must hold for the call only: a process that ran main(--offline) in-process
    (the tests do) must not carry HF_HUB_OFFLINE / HF_DATASETS_OFFLINE into every later
    dataset load — that leak made two unrelated GSM8K loads fail in the single-process
    suite once (the cache fingerprint under offline differs)."""
    import os
    B = _load("tinycap_oos_prompt_budget_env_under_test", SCRIPT)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    seen = {}

    def fake_main(a):
        seen.update({k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
        return 0

    monkeypatch.setattr(B, "_main", fake_main)
    assert B.main(["--arm", "on", "--out", str(tmp_path / "b.json"), "--offline"]) == 0
    assert seen == {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    assert "HF_HUB_OFFLINE" not in os.environ and "HF_DATASETS_OFFLINE" not in os.environ
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert B.main(["--arm", "on", "--out", str(tmp_path / "c.json"), "--offline"]) == 0
    assert os.environ["HF_HUB_OFFLINE"] == "0" and "HF_DATASETS_OFFLINE" not in os.environ
