"""Generic stage runner: structural-vs-leaf handling, bounded repair, stamps.

A *stage* wraps one engine wave plus up to ``repair_rounds`` repair waves
(subdirs ``repair1/``, ``repair2/``) and writes parsed ``outputs.jsonl`` +
``STAGE.json`` at the stage level. Completeness policy:

- **root** structural stages (A1, S1) never stamp below their configured
  fanout — block loudly after the repair budget, never a skinny corpus;
- **structural** stages block below ``min_slots_frac`` of expected units;
- **leaf** stages never block: refusals / truncations / content-filtered
  outputs / residual errors are per-record failures, counted in the stamp and
  absorbed by overshoot. Parse failures get one repair round (same request
  resubmitted).

Downstream requests are always built from ACTUAL upstream units, and the
stage stamp records achieved-vs-configured cardinality.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .batch import WaveEngine, WaveRequest, outcome_status, request_set_hash
from .config import RunConfig
from .constitution import Section
from .io import atomic_write, iter_jsonl, read_json, write_jsonl_atomic
from .templates import TemplateStore


class ParseError(Exception):
    pass


def normalize_content(value: str) -> str:
    """The ONE content normalization behind every uniqueness check: the
    in-response salvage dedup (parse_slot_array) and the cross-response
    repair-merge dedup (run_stage) must judge 'duplicate' identically."""
    return re.sub(r"\s+", " ", value.casefold()).strip()


class StageBlocked(RuntimeError):
    """A structural stage could not reach its required cardinality within the
    repair budget. The pipeline stops loudly rather than shrinking."""


@dataclass
class OutputRow:
    key: str  # unique unit key within the stage (slot path / leaf custom_id)
    custom_id: str  # producing request
    status: str  # ok | refusal | max_tokens | content_filtered | request_rejected | errored | expired | parse_failed
    # (canceled never reaches a stage: the engine halts the wave — operator action)
    data: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "custom_id": self.custom_id,
            "status": self.status,
            "data": self.data,
            "lineage": self.lineage,
        }

    @classmethod
    def from_json(cls, d: dict) -> "OutputRow":
        return cls(**d)


@dataclass
class StageCtx:
    cfg: RunConfig
    constitution_md: str
    sections: list[Section]
    templates: TemplateStore
    corpus_dir: Path
    upstream: dict[str, list[OutputRow]] = field(default_factory=dict)

    def ok_rows(self, stage: str, default: list | None = None) -> list[OutputRow]:
        rows = self.upstream[stage] if default is None else self.upstream.get(stage, default)
        return [r for r in rows if r.status == "ok"]


@dataclass
class StageSpec:
    name: str  # directory / wave name, e.g. "a1", "a3-topup-001"
    corpus: str
    kind: str  # "root" | "structural" | "leaf"
    templates: list[str]
    build_requests: Callable[[StageCtx], list[WaveRequest]]
    expected_units: Callable[[WaveRequest], list[str]]
    parse: Callable[[StageCtx, WaveRequest, dict], list[OutputRow]]
    build_repair: (
        Callable[[StageCtx, WaveRequest, list[OutputRow], list[str], int], WaveRequest]
        | None
    ) = None
    # statuses whose text still counts as usable data for this stage
    # (B2's diagnostic job is exactly to expose refusals/truncation)
    tolerate_statuses: tuple[str, ...] = ()
    # data field whose case-folded content must stay unique within one
    # request's unit set; repair rows are re-checked against already-accepted
    # rows so a repair can never refill a slot with duplicate content
    content_key: str | None = None

    @property
    def logical(self) -> str:
        """Base stage name with any top-up suffix stripped ("a3-topup-001" -> "a3")."""
        return self.name.split("-")[0]


@dataclass
class TopupExt:
    """Base of a top-up extension spec: ordinal + derived wave naming."""

    n: int  # topup ordinal, 1-based

    @property
    def tag(self) -> str:
        return f"x{self.n:03d}"

    @property
    def suffix(self) -> str:
        return f"-topup-{self.n:03d}"


@dataclass(frozen=True)
class CorpusDef:
    """Everything corpus-specific the orchestrator needs, in one record —
    pipeline.py stays corpus-agnostic."""

    name: str
    stages: Callable[[Any], list[StageSpec]]  # ext or None -> stage list
    ext_cls: type  # TopupExt subclass hydrated from TOPUP-*.json
    ext_upstream: tuple[str, ...]  # base stages an extension sub-DAG consumes
    materialize: Callable[[StageCtx], Path] | Any  # (ctx, suffix="") -> records path


# ------------------------------------------------------- shared stage helpers


def slot_ids(prefix: str, n: int, start: int = 0) -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(start, start + n)]


def stable_mod(key: str, mod: int) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % mod


@lru_cache(maxsize=8)
def _const_block(constitution_md: str) -> str:
    # one shared block string per constitution — thousands of requests per
    # wave reference it, so never rebuild the ~full-document string per call
    return f"<constitution>\n{constitution_md}\n</constitution>"


def const_user_content(constitution_md: str, rendered: str) -> list[dict]:
    """Full-constitution waves: constitution as the first user block with a
    cache breakpoint (identical prefix within a sub-batch), template after."""
    return [
        {
            "type": "text",
            "text": _const_block(constitution_md),
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": rendered},
    ]


def wave_params(ctx: StageCtx, stage: str, messages: list, system=None, *, corpus: str) -> dict:
    p = {
        "model": ctx.cfg.engine.model,
        "max_tokens": getattr(ctx.cfg, corpus).max_tokens[stage],
        "messages": messages,
    }
    if system is not None:
        p["system"] = system
    return p


def structured_format(schema: dict) -> dict:
    """The output_config body for a structured-JSON score request."""
    return {"format": {"type": "json_schema", "schema": schema}}


def repair_request(ctx, req: WaveRequest, existing, missing, round_no: int) -> WaveRequest:
    """Leaf parse failures: resubmit the SAME request under a fresh id."""
    return WaveRequest(
        custom_id=f"{req.custom_id}-r{round_no}",
        params=req.params,
        lineage=req.lineage,
    )


def load_outputs(stage_dir: Path) -> list[OutputRow]:
    return [OutputRow.from_json(d) for d in iter_jsonl(stage_dir / "outputs.jsonl")]


def run_stage(
    engine: WaveEngine,
    ctx: StageCtx,
    spec: StageSpec,
    *,
    repair_rounds: int = 2,
    min_slots_frac: float = 0.95,
    on_event: Callable[[str], None] | None = None,
) -> list[OutputRow]:
    on_event = on_event or (lambda m: None)
    stage_dir = ctx.corpus_dir / spec.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    requests = spec.build_requests(ctx)
    if not requests:
        raise StageBlocked(f"stage {spec.name}: zero requests built (upstream empty?)")
    template_hashes = ctx.templates.hashes(spec.templates)
    stage_hash = request_set_hash(requests, template_hashes)

    stage_stamp_path = stage_dir / "STAGE.json"
    if stage_stamp_path.exists():
        stamp = read_json(stage_stamp_path)
        if stamp["stage_hash"] != stage_hash:
            raise RuntimeError(
                f"stage {spec.name}: completed stage hash mismatch "
                f"({stamp['stage_hash'][:12]} != {stage_hash[:12]}) — completed "
                f"stages are immutable within a run"
            )
        return load_outputs(stage_dir)

    expected: dict[str, list[str]] = {
        r.custom_id: list(spec.expected_units(r)) for r in requests
    }
    all_expected_units = [u for units in expected.values() for u in units]
    if len(set(all_expected_units)) != len(all_expected_units):
        raise RuntimeError(f"stage {spec.name}: duplicate expected unit keys")

    outcomes = engine.run_wave(
        stage_dir,
        requests,
        wave_name=spec.name,
        template_hashes=template_hashes,
        description=f"{ctx.cfg.run_id} {spec.corpus} {spec.name}",
        rs_hash=stage_hash,
    )

    rows: dict[str, OutputRow] = {}
    failures: dict[str, OutputRow] = {}  # custom_id -> terminal failure row

    def _try_parse(req: WaveRequest, outcome: dict) -> None:
        status = outcome_status(outcome)
        usable = status == "succeeded" or status in spec.tolerate_statuses
        if usable:
            try:
                parsed = spec.parse(ctx, req, outcome)
            except ParseError as e:
                failures[req.custom_id] = OutputRow(
                    key=req.custom_id, custom_id=req.custom_id,
                    status="parse_failed", data={"error": str(e)},
                    lineage=req.lineage,
                )
                return
            exp = set(expected[req.custom_id])
            for row in parsed:
                if row.key not in exp:
                    failures[req.custom_id] = OutputRow(
                        key=req.custom_id, custom_id=req.custom_id,
                        status="parse_failed",
                        data={"error": f"unexpected unit key {row.key!r}"},
                        lineage=req.lineage,
                    )
                    return
            for row in parsed:
                if row.key not in rows:  # first producer wins (base before repair)
                    row.lineage = {
                        **req.lineage,
                        **row.lineage,
                        "batch_id": outcome.get("batch_id"),
                    }
                    rows[row.key] = row
            failures.pop(req.custom_id, None)
        else:
            failures[req.custom_id] = OutputRow(
                key=req.custom_id, custom_id=req.custom_id, status=status,
                data={"text": outcome.get("text", ""),
                      "error": outcome.get("error_message", "")},
                lineage=req.lineage,
            )

    for req in requests:
        _try_parse(req, outcomes[req.custom_id])

    # ---- repair rounds ----------------------------------------------------
    rounds_used = 0
    if spec.build_repair is not None:
        for round_no in range(1, repair_rounds + 1):
            repair_reqs: list[WaveRequest] = []
            repair_parent: dict[str, WaveRequest] = {}
            for req in requests:
                missing = [u for u in expected[req.custom_id] if u not in rows]
                if not missing:
                    continue
                # leaf terminal failures (refusal / content_filtered / errored
                # outside the tolerate set) are NOT repairable — attrition by
                # design. Parse failures are, and so is request_rejected: a
                # content-dependent server rejection may be transient, and the
                # fresh custom_id resubmit is its second chance.
                f_row = failures.get(req.custom_id)
                if (
                    spec.kind == "leaf"
                    and f_row is not None
                    and f_row.status not in ("parse_failed", "request_rejected")
                ):
                    continue
                existing = [rows[u] for u in expected[req.custom_id] if u in rows]
                rreq = spec.build_repair(ctx, req, existing, missing, round_no)
                repair_reqs.append(rreq)
                repair_parent[rreq.custom_id] = req
            if not repair_reqs:
                break
            rounds_used = round_no
            on_event(
                f"stage {spec.name}: repair round {round_no} — "
                f"{len(repair_reqs)} requests"
            )
            repair_outcomes = engine.run_wave(
                stage_dir / f"repair{round_no}",
                repair_reqs,
                wave_name=f"{spec.name}-repair{round_no}",
                template_hashes=template_hashes,
                description=f"{ctx.cfg.run_id} {spec.corpus} {spec.name} repair{round_no}",
                # a repair wave is ADVERSELY SELECTED — it holds only requests
                # that already failed once — so the systematic-rejection cap
                # (a fraction of the wave) is meaningless here; the BASE wave's
                # cap already proved the rejections aren't systematic, and a
                # persistently rejected repair just leaves its slot missing,
                # bounded by the completeness policy.
                max_rejected_frac=1.0,
            )
            for rreq in repair_reqs:
                parent = repair_parent[rreq.custom_id]
                outcome = repair_outcomes[rreq.custom_id]
                status = outcome_status(outcome)
                usable = status == "succeeded" or status in spec.tolerate_statuses
                if not usable:
                    continue
                try:
                    parsed = spec.parse(ctx, parent, outcome)
                except ParseError:
                    continue
                exp = set(expected[parent.custom_id])
                accepted_content: set[str] = set()
                if spec.content_key is not None:
                    accepted_content = {
                        normalize_content(rows[u].data[spec.content_key])
                        for u in expected[parent.custom_id]
                        if u in rows
                    }
                for row in parsed:
                    if row.key in exp and row.key not in rows:
                        if spec.content_key is not None:
                            folded = normalize_content(row.data[spec.content_key])
                            if folded in accepted_content:
                                # duplicate of an already-accepted row (base or
                                # earlier repair) — the slot stays missing
                                continue
                            accepted_content.add(folded)
                        row.lineage = {
                            **parent.lineage,
                            **row.lineage,
                            "batch_id": outcome.get("batch_id"),
                            "repair_round": round_no,
                        }
                        rows[row.key] = row
                if all(u in rows for u in expected[parent.custom_id]):
                    failures.pop(parent.custom_id, None)

    # ---- completeness policy ----------------------------------------------
    n_expected = len(all_expected_units)
    n_achieved = len(rows)
    if spec.kind == "root" and n_achieved < n_expected:
        raise StageBlocked(
            f"ROOT stage {spec.name}: {n_achieved}/{n_expected} units after "
            f"{rounds_used} repair rounds — a root wave never stamps below its "
            f"configured fanout; revise the template/config and re-run"
        )
    if spec.kind == "structural" and n_achieved < min_slots_frac * n_expected:
        raise StageBlocked(
            f"stage {spec.name}: {n_achieved}/{n_expected} units "
            f"(< min_slots_frac={min_slots_frac}) after {rounds_used} repair rounds"
        )

    # ---- materialize -------------------------------------------------------
    out_rows = [rows[k] for k in sorted(rows)] + [
        failures[c] for c in sorted(failures) if failures[c].key not in rows
    ]
    write_jsonl_atomic(
        stage_dir / "outputs.jsonl",
        (json.dumps(r.to_json(), ensure_ascii=False) for r in out_rows),
    )
    status_counts: dict[str, int] = {}
    for r in out_rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    atomic_write(
        stage_stamp_path,
        json.dumps(
            {
                "stage": spec.name,
                "corpus": spec.corpus,
                "kind": spec.kind,
                "stage_hash": stage_hash,
                "n_requests": len(requests),
                "expected_units": n_expected,
                "achieved_units": n_achieved,
                "repair_rounds_used": rounds_used,
                "status_counts": status_counts,
                "template_hashes": template_hashes,
            },
            indent=1,
        ),
    )
    return out_rows
