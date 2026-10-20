"""qwen-seedrep-native-battery — F-S on the MISALIGN leg (found while verifying every sender for the BoN defect).

`runner.run_family` wraps its transport with `rev.maybe_wrap_post_json` (every `/chat/completions` POST a recorded
attempt whose headers come from `headers_for`), and the `request_evidence=` seam is what labels the draws — but the
CLI (`runner.main`) never passed it: `run_family(...)` was called without `request_evidence`, so under a hook that is ON
the wrapped transport refused the FIRST draw as an unlabelled target request (the BoN child's defect, on this leg).
Fix: `main` passes `request_evidence=rev.maybe_misalign_request_evidence()` (None when off — bytes as before). Proved on
the real `run_family` + the real `MisalignRequestEvidence` over the real recorder, a stub transport that accepts headers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

from sisteer.capeval.misalign import runner as R  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.eval import request_evidence_client as rev  # noqa: E402
from test_misalign_runner import CELLS, ITEMS, QWEN_IDENTITY, _rows  # noqa: E402
from test_seedrep_ws1c_misalign_seed_policy import EvidenceHttp, _seeded  # noqa: E402

S = 1
TASK_OF = {b: rev.store_task_name(rev.LEG_MISALIGN, b, family="agentic") for b in rev.MISALIGN_BEHAVIORS}


class _UniqueIdHttp(EvidenceHttp):
    """The stub transport, answering each attempt with its OWN completion id (the engine's shape; one id per
    attempt is what the recorder binds rows by)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n = 0

    def post_json(self, url, payload, headers=None):
        resp = super().post_json(url, payload, headers=headers)
        self.n += 1
        return {**resp, "id": f"cmpl-{self.n}"}


@pytest.fixture(autouse=True)
def _hook_off(monkeypatch):
    monkeypatch.delenv(rev.ENV_DIR, raising=False)
    monkeypatch.delenv(rev.ENV_EXTRA, raising=False)
    monkeypatch.delenv(rev.ENV_SEED_BASE, raising=False)
    rev.reset_env_cache()
    yield
    rev.reset_env_cache()


def _hook_on(monkeypatch, out_dir: Path) -> None:
    monkeypatch.setenv(rev.ENV_DIR, str(out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    monkeypatch.setenv(rev.ENV_EXTRA, json.dumps({"leg": "misalign", "arm": "none", "s": S, "attempt_id": "ta-fs-misalign"}))
    rev.reset_env_cache()


def _jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_cli_passes_the_misalign_request_evidence_seam_bound_to_the_envs_recorder(tmp_path, monkeypatch):
    """`main` -> `run_family(request_evidence=maybe_misalign_request_evidence())`: the env's recorder when the hook is ON,
    None when off. At 79185d481 `main` passed nothing (the kwarg is absent from the captured call)."""
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(CELLS))
    seen = {}

    def fake_run_family(*a, **kw):
        seen.update(kw)
        return {c["cell_id"]: {"requested": 0, "skipped": 0, "generated": 0, "excluded_items": 0} for c in CELLS}
    monkeypatch.setattr(R, "run_family", fake_run_family)
    argv = ["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells), "--family", "agentic", "--n", "1",
            "--out", str(tmp_path / "o"), "--sampler", "vendor", "--seed-policy", "per_response", "--seed-base", str(S)]
    assert R.main(argv) == 0 and "request_evidence" in seen and seen["request_evidence"] is None      # off: inert
    seen.clear()
    _hook_on(monkeypatch, tmp_path / "ev")
    assert R.main(argv) == 0
    ev = seen["request_evidence"]
    assert isinstance(ev, rev.MisalignRequestEvidence) and ev.client is rev.from_env()
    assert ev.headers_for(family="agentic", behavior="leaking", item_id="i1", sample_idx=0)[C.HEADER_TASK] == "agentic/leaking"


def test_real_run_family_under_the_hook_refuses_unlabelled_without_the_seam_and_labels_every_draw_with_it(tmp_path, monkeypatch):
    """The real `run_family` with the hook ON over a stub transport: WITHOUT `request_evidence` (main at 79185d481) the
    wrapped transport refuses the first draw as unlabelled — nothing reaches the wire; WITH the env's
    `MisalignRequestEvidence` every draw carries `agentic/<behavior>` + `<item_id>/<sample_idx>` and its own id, the
    recorder files one task record per attempt, and every row binds to its attempt (`request_id` / `completion_id`)."""
    _hook_on(monkeypatch, tmp_path / "ev")
    http0 = _UniqueIdHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    with pytest.raises(rev.RequestEvidenceError, match="unlabelled target request"):
        _seeded(tmp_path, http0, S, out="bad")
    assert not [u for u, _p in http0.posts if str(u).endswith("/chat/completions")]

    http = _UniqueIdHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    ev = rev.maybe_misalign_request_evidence()
    assert isinstance(ev, rev.MisalignRequestEvidence)
    _seeded(tmp_path, http, S, out="ev", request_evidence=ev)
    posts = [(p, h) for (u, p), h in zip(http.posts, http.headers) if str(u).endswith("/chat/completions")]
    assert posts and all(h is not None for _p, h in posts)
    items = {i["item_id"]: i for i in ITEMS}
    for payload, h in posts:
        item_id, idx = h[C.HEADER_SAMPLE].split("/")
        assert h[C.HEADER_TASK] == TASK_OF[items[item_id]["behavior"]] and len(h[C.HEADER_REQUEST_ID]) == 32
    assert len({h[C.HEADER_REQUEST_ID] for _p, h in posts}) == len(posts)
    records = _jsonl(tmp_path / "ev" / C.REQUEST_EVIDENCE_FILENAME)
    responses = _jsonl(tmp_path / "ev" / rev.RESPONSE_EVIDENCE_FILENAME)
    assert len(records) == len(responses) == len(posts) and all(r["kind"] == "task" for r in records)
    by_rid = {r["request_id"]: r for r in records}
    cid_by_rid = {x["request_id"]: x["completion_id"] for x in responses}
    rows = [r for cell in ("none", "prompted__safety") for r in _rows(tmp_path / "ev" / cell / "responses.jsonl")]
    assert len(rows) == len(posts)
    for row in rows:
        rec = by_rid[row["request_id"]]
        assert (rec["task"], rec["item_id"], rec["sample_idx"]) == (TASK_OF[row["behavior"]], row["item_id"], row["sample_idx"])
        assert row["completion_id"] == cid_by_rid[row["request_id"]] == rec_cid(rec, responses)
    assert len({r["request_id"] for r in rows}) == len(rows) == len({r["completion_id"] for r in rows})


def rec_cid(rec, responses):
    return next(x["completion_id"] for x in responses if x["request_id"] == rec["request_id"])
