"""Shared helpers of the tests for the Qwen seed-replicate native battery:
a plain module the three test files import by name from the tests dir (the `test_petri_store_shape` precedent).
Synthetic throughout: invented ids, a loopback transport, no real session data."""
from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import request_evidence_client as rev

REPO = Path(__file__).resolve().parents[1]
S = 2                                   # the seed base under test (a literal training seed)
VMODEL = "petri__base"
ATTEMPT = "ta-11111111-2222"
GSM8K_TASK = C.inspect_task(C.GSM8K_PAIRED_TASK_NAME)
#: the six vendor sampler terms AS SENT (Qwen instruct; presence is not server-defaultable and rides every request)
VENDOR_SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}


@pytest.fixture(autouse=True)
def hook_off(monkeypatch):
    """Every test starts with the lane switch OFF and the per-env cache empty."""
    for k in (rev.ENV_DIR, rev.ENV_EXTRA, rev.ENV_SEED_BASE):
        monkeypatch.delenv(k, raising=False)
    rev.reset_env_cache()
    yield
    rev.reset_env_cache()


def lane_on(monkeypatch, root: Path, *, extra: Optional[Mapping[str, Any]] = None, seed: int = S) -> rev.RequestEvidenceClient:
    """Flip the lane switch ON for `root` (the attempt root) and return the process client."""
    monkeypatch.setenv(rev.ENV_DIR, str(root))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(seed))
    if extra is not None:
        monkeypatch.setenv(rev.ENV_EXTRA, json.dumps(dict(extra)))
    else:
        monkeypatch.delenv(rev.ENV_EXTRA, raising=False)
    rev.reset_env_cache()
    client = rev.from_env()
    assert client is not None
    return client


def ws3g_modules():
    """`audit_common` + `request_evidence_audit` (read-only): from this tree once landed, else the sibling
    worktree on the grading host; loaded by file so their sys.path insertion never shadows this tree. Skips when absent."""
    candidates = [REPO / "scripts" / "sdfbase" / "seedrep", REPO.parent / "seedrep-ws3g" / "scripts" / "sdfbase" / "seedrep"]
    here = next((d for d in candidates if (d / "request_evidence_audit.py").is_file() and (d / "audit_common.py").is_file()), None)
    if here is None:
        pytest.skip("WS3-G's request_evidence_audit.py is not on this tree (nor in the sibling worktree); the merged reseed runs it")
    before = list(sys.path)
    # registered ONLY for the duration of the load (`request_evidence_audit` binds `import audit_common` to the copy loaded here),
    # then the PRIOR module objects are put back: leaving a second `audit_common` in sys.modules handed every later importer a
    # second `AuditRefusal` class while tools imported earlier (`determinism_check`) kept the first, so a later test module's
    # `pytest.raises(ac.AuditRefusal)` could not catch the refusal the tool raised (order-dependent)
    prior = {name: sys.modules.get(name) for name in ("audit_common", "request_evidence_audit")}
    mods = {}
    try:
        for name in ("audit_common", "request_evidence_audit"):
            spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mods[name] = mod
    finally:
        for name, m in prior.items():
            if m is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = m
        sys.path[:] = before
    return mods["audit_common"], mods["request_evidence_audit"]


def load_served_capability_run():
    """PRODUCTION's driver module (`scripts/served_capability_run.py`), imported from this tree the way
    `gsm8k_synthetic_store.native_lens_position_record` does — so the test drives the REAL `_RequestEvidence` writer,
    never a stand-in spelled by hand."""
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    mod = sys.modules.get("served_capability_run")
    if mod is None:
        spec = importlib.util.spec_from_file_location("served_capability_run", scripts / "served_capability_run.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["served_capability_run"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def body(*, turn_messages: int = 1, seed: int = 1, model: str = VMODEL) -> Dict[str, Any]:
    """One TARGET request body: the six vendor sampler keys AS SENT (presence included), a seed, a transcript with
    `turn_messages - 1` assistant turns (so `petri_turn_index` reads `turn_messages`)."""
    msgs: List[Dict[str, str]] = [{"role": "user", "content": "hello"}]
    for i in range(turn_messages - 1):
        msgs.append({"role": "assistant", "content": f"reply {i}"})
        msgs.append({"role": "user", "content": f"follow-up {i}"})
    return {"model": model, "messages": msgs, "max_tokens": 1024, "seed": seed,
            **VENDOR_SIX,
            "chat_template_kwargs": {"enable_thinking": False}}


def wire(task: str, item_id: str, sample_idx: int, *, payload: Optional[Mapping[str, Any]] = None,
         request_id: Optional[str] = None) -> Tuple[bytes, Dict[str, str]]:
    """The exact bytes + headers of one attempt as the Inspect provider's request hook hands them to its sink."""
    raw = json.dumps(dict(payload if payload is not None else body()), separators=(",", ":")).encode("utf-8")
    headers = {**rev.label_headers(task, item_id, sample_idx), C.HEADER_REQUEST_ID: request_id or uuid.uuid4().hex,
               "content-type": "application/json", "content-length": str(len(raw))}
    return raw, headers


def engine_id(completion_id: str) -> str:
    """The engine log's INTERNAL id: the external completion id + exactly one `-<8 hex>` suffix (is_engine_id_of)."""
    return f"{completion_id}-0123abcd"


def ledger_row(request_id: str, *, status: int, completion_id: Optional[str]) -> Dict[str, Any]:
    """The usage ledger's row for one attempt as the binding reads it (`_ok_status`: not rejected, 2xx; the
    additive `completion_id` column)."""
    return {C.LEDGER_REQUEST_ID_FIELD: request_id, "status": int(status), "rejected": not (200 <= status < 300),
            "completion_id": completion_id}


def _build(dc_cls, known: Mapping[str, Any], what: str):
    """Instantiate one of the audit modules' dataclasses BY FIELD NAME from what this helper knows (their readers are
    a moving target — `Evidence` gained `engine_by_external` / `standups` and lost `vmodels` between their tips
    83375344 and 71d4bae2 while these tests were being written); a REQUIRED field the helper does not know fails the
    test by name instead of guessing."""
    from dataclasses import MISSING, fields
    names = [f.name for f in fields(dc_cls)]
    required = [f.name for f in fields(dc_cls) if f.default is MISSING and f.default_factory is MISSING]
    unknown = [n for n in required if n not in known or known[n] is _ABSENT]
    if unknown:
        pytest.fail(f"WS3-G's {what} requires field(s) {unknown} this helper does not know — mirror their constructor")
    return dc_cls(**{n: known[n] for n in names if n in known and known[n] is not _ABSENT})


_ABSENT = object()


def evidence(rea, *, records: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]],
             response_rows: Sequence[Mapping[str, Any]] = (), vmodels: Optional[Mapping[str, str]] = None,
             label: str = "gsm8k/arditi/base/shard00"):
    """The `Evidence` of one attempt root, built as `load_evidence` builds it from the same inventories
    (records, ledger rows, the response rows' authoritative completion per attempt, the engine log's ids)."""
    completion_by_rid: Dict[str, str] = {}
    for r in ledger_rows:
        if r.get(C.LEDGER_REQUEST_ID_FIELD) and r.get("completion_id"):
            completion_by_rid[str(r[C.LEDGER_REQUEST_ID_FIELD])] = str(r["completion_id"])
    for r in response_rows:
        if r.get("request_id") and r.get("completion_id"):
            completion_by_rid[str(r["request_id"])] = str(r["completion_id"])
    engine_ids = [engine_id(c) for c in sorted(set(completion_by_rid.values()))]
    index = getattr(rea, "engine_id_index", None)
    # their `ResponseIndex` (audit_common.response_evidence_index): (task, sample, vmodel) of the REQUEST record → the
    # completed attempts the response rows recorded for it
    by_rid = {str(r["request_id"]): (str(r["task"]), str(r["sample"]), str(r["vmodel"]))
              for r in records if r.get("request_id") and r.get("task") and r.get("sample") and r.get("vmodel")}
    completed_by_sample: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
    for r in response_rows:
        k = by_rid.get(str(r.get("request_id")))
        if k is not None and r.get("completion_id"):
            completed_by_sample.setdefault(k, []).append((str(r["request_id"]), str(r["completion_id"])))
    known: Dict[str, Any] = {
        "label": label, "records": list(records), "ledger_rows": list(ledger_rows),
        "records_by_id": {str(r["request_id"]): r for r in records if r.get("request_id")},
        "ledger_by_id": {str(r[C.LEDGER_REQUEST_ID_FIELD]): r for r in ledger_rows},
        "engine_ids": engine_ids,
        "engine_by_external": index(engine_ids) if callable(index) else _ABSENT,
        "dup_engine_ids": [], "completion_by_rid": completion_by_rid, "completed_by_sample": completed_by_sample,
        # their determinism-check adoption of the response record's text_sha256: request id → the response text's sha
        "response_sha_by_rid": {str(r["request_id"]): str(r["text_sha256"]) for r in response_rows
                                if r.get("request_id") and isinstance(r.get("text_sha256"), str)},
        "standups": [], "attempt_ids": [ATTEMPT],
        "engine_logs": [],
        # the pre-71d4bae2 shape's arm → vmodel map
        "vmodels": [dict(vmodels or {"none": VMODEL})],
    }
    return _build(rea.Evidence, known, "Evidence")


def store_read(ac, *, draws: Sequence[Any], arm: str, leg: str, tasks: Sequence[str], store_dir: Optional[Path] = None):
    """The `StoreRead` over `draws` (`bind_rows` reads `store.draws`, `store.copied_cap(d)`, `store.arm`) —
    a real instance, not a namespace, so a method the reader grows is present."""
    known: Dict[str, Any] = {
        "leg": leg, "arm": arm, "s": S, "trial": None, "seed_base": S, "tasks": tuple(tasks),
        "filed": {(d.task, d.item_id) for d in draws}, "draws": list(draws), "vmodel": None, "expected_seed": None,
        "seed_rule_note": "test fixture", "store_dir": store_dir or Path("/nonexistent/store"), "complete": True,
        "completeness_note": "test fixture", "n_rows": len(draws), "notes": [], "cap": None, "filed_here": None,
    }
    return _build(ac.StoreRead, known, "StoreRead")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
