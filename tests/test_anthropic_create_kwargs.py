"""Every Messages-API request this tree renders uses only keyword arguments the INSTALLED SDK's ``Messages.create`` accepts.
anthropic 1.x dropped ``temperature`` / ``top_p`` / ``top_k`` from the signature (a sampling value rides ``extra_body``), and a
stale kwarg is a TypeError on every call — the probe-set builder could neither generate nor verify an item. Static for the
builder's two ``_call_api`` sites (their kwargs are forwarded verbatim to ``client.messages.create``), rendered for the judge's
request builder."""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import anthropic.resources.messages as _m

from sisteer.capeval import judging as j
from sisteer.capeval.judging import ORBENCH_RUBRIC_V1, JudgeRequest, render_request

REPO = Path(__file__).resolve().parents[1]
CREATE_PARAMS = set(inspect.signature(_m.Messages.create).parameters) - {"self"}


def _call_api_kwargs(src: str) -> list[set[str]]:
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_call_api":
            out.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return out


def test_the_probe_set_builders_create_kwargs_are_the_installed_sdks():
    sites = _call_api_kwargs((REPO / "scripts" / "sdfbase" / "build_probe_set.py").read_text(encoding="utf-8"))
    assert len(sites) == 2, sites                                  # the generation call and the verification call
    for kw in sites:
        assert kw <= CREATE_PARAMS, (kw - CREATE_PARAMS, sorted(CREATE_PARAMS))
        assert not ({"temperature", "top_p", "top_k"} & kw)


def test_the_judges_rendered_request_uses_only_the_installed_sdks_kwargs():
    req = JudgeRequest(prompt="p", candidate_a="r", candidate_b=None, roles=None, rubric_sha=ORBENCH_RUBRIC_V1.sha,
                       judge_model=j.JUDGE_MODEL, protocol_sha="a" * 16)
    for t in (None, 0.0, 0.3):
        params = render_request(ORBENCH_RUBRIC_V1, req, temperature=t)
        assert set(params) <= CREATE_PARAMS, set(params) - CREATE_PARAMS
        assert "temperature" not in params
        if t is not None:
            assert params["extra_body"] == {"temperature": t}
