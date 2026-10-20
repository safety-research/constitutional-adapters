"""SYNTHETIC stores for the GSM8K paired-200 thinking-ON analysis tests
(``scripts/tinycap_gsm8k_paired200.py`` / ``scripts/tinycap_gsm8k_validators.py``
/ ``scripts/gsm8k_overlap_check.py``).

Placeholder shas, ``TESTHOST``, invented item ids and question texts — never
real recorded data. Modelled on ``tests/math500_synthetic_store.py`` (imported
for its PURE pieces — the attempt / probe / certification record builders —
and never edited): every store is written through the REAL writers and record
builders of this tree, so the reader and the producer cannot drift apart
without a red test — ``inspect_served.outcome_record`` over ``ItemResult``
objects, ``completion.finish_task`` (summary, outcomes, done marker),
``contracts.render_certified`` / ``lens_position_attested`` /
``lens_on_bite_verdict`` / ``transfer_provenance_block``,
``contracts.sampling_identity`` / ``sampling_ref``,
``contracts.request_evidence_record`` over the wire bytes, ``usage_ledger.ledger_row``.
THE LADDER RECORD AND THE RUNG SUBSET ARE THE ADAPTER'S OWN: ``inspect_rung.
classify_store`` (the record's AUTHOR) is run over the finished synthetic store
with the fixture's draw texts, its record is filed where lane B's
``ladder_plan.py`` (the WRITER) files it — ``<out_root>/cap_ladder.json``, its
serialization — and its ``rung_subset/1`` body is written VERBATIM (validated
by ``contracts.validate_item_subset``); a rung store's ``cap_rung`` identity and
``item_subset`` manifest block are ``contracts.cap_rung_identity`` /
``item_subset_block`` of that subset. Nothing here spells a producer literal
the producers export. The two fields lane H's driver ADDS (``attempt_id`` on
every ledger row and request-evidence record, ``accepted_generation_attempt``
+ per-entry ``attempt_id`` / ``kind`` on the manifest) are written beside the
producer's records — through the producer's own field list whenever this
tree's contracts already carry them.

THE DECLARED CLASS IS WRITTEN IN THE PRODUCER'S REAL SHAPE (served-positions-null-diagnosis.md §4.2 "Fixture"): the
standup record keys ``lens_served_positions`` by the declared ``<harness>@thinking`` spelling ONLY; the evidence carries
predicate (B)'s ``thinking_header_by_arm`` record and a thinking-ON steering-bites probe stamped with the declared
triple, and NO (B′)/(C) block (the driver never records them on a header-located cell); the manifest's three
served-positions fields are the producer's NULL (the bare-key lookup gap at 8f5f1495) unless
``producer_records_served_positions`` simulates the fixed producer; and the standup record the analysis resolves the
fact from (``routing_table.json``, ``standup_jobs.jsonl``, ``steering_attestation.json``,
``lens_registry/registry.json``) is written under ``<gen_root>/serving/<jobdir>/gsm8k/`` beside the stores, the way
the pull relocates the cluster's — the manifest's ``routing_table`` field is the synthetic ABSOLUTE cluster path.

The SYNTHETIC MATRIX below is the 42-row design with four held
rows: gemma4e4b / qwen35_4b in the ``it`` view (none, prompted, two
constitution objects HELD) and the ``main`` view (none + eight), nemotron3nano4b /
glm46vflash in ``main`` (none + seven). Under the header-invariant rule every
launch row is servable: NO row carries an expected refusal; the reader's
expected-refusal path is exercised through ``--expected-refusals``
(:func:`expected_refusals_doc`).

THE NO-SWITCH ARM (llama-paper-rows P7 / P17 / P18 / P21): :func:`matrix_rows_llama` is the 11-row Llama matrix in
``configs/sdfbase/gsm8k/matrix_llama.json``'s shape (schema 2: every row ``thinking_mode: not_available``, the ``card``
sampling row, ``tuple_name`` / ``tuple_names`` null even when fitted, the ``unspecified`` stamp triples, ``declared``
false everywhere; placeholder pins), and :func:`build_store` on such a row writes the store THROUGH THE REAL NO-SWITCH
PRODUCERS: ``contracts.no_switch_render_certified`` / ``lens_position_attested(arm=not_available)`` /
``no_switch_lens_on_bite_record``; ``inspect_served.outcome_record`` over a plan whose split mode is ``none`` (so the
items carry ``stray_close_tag``, the flags derived from the content by ``think_split.split_think(mode="none")``);
ledger rows through ``usage_ledger.body_facts`` over bodies that carry NO ``chat_template_kwargs`` (so ``thinking`` is
null, never typed); request-evidence records with no ``enable_thinking`` key; the manifest's ``thinking.enable_thinking``
None and ``render`` native (I11); ``thinking.mode`` / ``thinking_by_task`` ``not_available`` in the hashed identity; NO
``out_of_support_thinking`` / ``thinking_header_ids``; the row ``thinking`` key ``not_available``; the four row-provenance
fields (``contracts.row_provenance``) when ``bank_view`` is given. The switch families' stores are byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math500_synthetic_store as S

from sisteer.capeval.manifest import sha256_hex16
from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tiers import think_split as TS
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]

FULL_SHA = "7777777788888888999999990000000011111111"
SHA7 = FULL_SHA[:7]
TASK = C.GSM8K_PAIRED_TASK
TASK_NAME = C.GSM8K_PAIRED_TASK_NAME
N = C.GSM8K_PAIRED_N
CAP_BASE, CAP_RUNG = 8192, CL.CAP_CEILING
MODE_PRIMARY = C.SAMPLING_THINKING_CARD
#: the synthetic Platinum selection: question texts, the retired instrument's ids (``parity._content_id``'s rule) and
#: the stock task's inspect ids (``contracts.gsm8k_sample_id``), all from the SAME text
QUESTIONS: list[str] = [f"Synthetic question {i}: a farm has {i + 3} hens laying {2 * i + 1} eggs each; how many eggs?" for i in range(N)]
RETIRED_IDS: list[str] = [f"g{i:04d}-{sha256_hex16(q.encode('utf-8'))[:8]}" for i, q in enumerate(QUESTIONS)]
IDS: list[str] = [C.gsm8k_sample_id(q) for q in QUESTIONS]
ANSWERS: list[str] = [str((i + 3) * (2 * i + 1)) for i in range(N)]
PAIRED_BY_INSPECT_ID: dict[str, str] = dict(zip(IDS, RETIRED_IDS))
#: the GSM8K wrap's registered scorer name (the log's score key) — the runner names no primary scorer for GSM8K
SCORER_NAME = TS.WRAPPER_NAME
#: the ladder literals are the ADAPTER's (inspect_rung), by import — the fixture spells none of them
TRIGGER_RULE: str = IR.GSM8K_TRIGGER_RULE
TRIGGER_THRESHOLD: float = IR.GSM8K_TRIGGER_THRESHOLD
#: the Llama matrix's trigger (the harmonised 1 % RAW at-cap), on the document and every row
TRIGGER_THRESHOLD_LLAMA: float = IR.GSM8K_TRIGGER_THRESHOLD
REPORT_LINE: float = IR.GSM8K_REPORT_LINE
EXIT_TRIGGERED_NO_ELIGIBLE: str = IR.EXIT_TRIGGERED_NO_ELIGIBLE
#: the attempt-scoped fields (pinned equal to the validators module by test)
LEDGER_ATTEMPT_ID_FIELD = "attempt_id"
MANIFEST_KEY_ACCEPTED_ATTEMPT = "accepted_generation_attempt"
#: the fixture's own sidecar of every draw's channels (never read by the analysis; the .eval writer and the adapter's
#: text feed read it — the real log carries the real text)
CHANNELS_NAME = "synthetic_channels.json"
#: the no-switch stores' second sidecar: the provider's lift record per draw (P21), read by the .eval writer only
LIFT_NAME = "synthetic_lift.json"
#: the wrapper's RESULT line beside the store (f22_result's shape) and the driver's gen_stats copy of the summary
RESULT_NAME = "RESULT"
#: a refusal code the --expected-refusals path is tested with (no row of the matrix expects one any more)
EXPECTED_REFUSAL_CODE = "thinking_on_declaration_vacuous"
PROXY_URL, LEDGER_URL = "http://127.0.0.1:19000", "http://127.0.0.1:19001"
UPSTREAM_TIMEOUT_S = 7200.0
#: the proxy upstream timeout that makes cap_ladder.transport_check REFUSE the 16,384 rung (the censoring default)
UPSTREAM_TIMEOUT_REFUSING_S = 600.0
PROMPT_BASE = 2100
TS0 = "2026-01-01T00:00:00Z"
#: the served window: a synthetic gsm8k geometry entry (10-shot prompt + the cap)
MAX_MODEL_LEN_BASE, MAX_MODEL_LEN_RUNG = 11008, 19200
#: the table entries the geometry block names per leg class (the published pull's: the thinking-ON base window, the
#: thinking-OFF overlap window, the 16,384 rung) — the standup resolves S from the model budget at each
GEOMETRY_EVAL_THINKING, GEOMETRY_EVAL_OFF, GEOMETRY_EVAL_RUNG = "capability:gen_long_thinking", "capability:gen_long", "capability:gen_long_gsm8k_rung"
#: the standup record beside every served job (served-positions-null-diagnosis.md §2): a synthetic ABSOLUTE cluster
#: wave root the manifest's `routing_table` names; the reader resolves its trailing serving/<jobdir>/gsm8k/ under the
#: gen root, the way a pull relocates it (no host, no home path — placeholders only)
CLUSTER_WAVE_ROOT = "/cluster-shared/exp/gsm8k_paired200_synthetic"
STANDUP_DIR_PARENT, STANDUP_TASK_SUBDIR = "serving", "gsm8k"
STANDUP_ROUTING_TABLE, STANDUP_JOBS, STANDUP_ATTESTATION, STANDUP_REGISTRY = ("routing_table.json", "standup_jobs.jsonl",
                                                                              "steering_attestation.json", "lens_registry/registry.json")
STANDUP_JOB_BASE = 480000
#: the driver's evidence key of predicate (B)'s record and its header source label (pinned equal to the reader's by test)
EVIDENCE_THINKING_HEADER_BY_ARM = "thinking_header_by_arm"
HEADER_SOURCE_SERVED = "served_tokenize"
#: the routing table's intervention hash of the `none` route and its scale-0 twin (the retired store's, already here)
BASE_INTERVENTION_HASH = "2ff9cdff091aef6f"

MODEL_KEYS: tuple[str, ...] = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")
#: llama-paper-rows: the NO-SWITCH family's ONE served target (P1) — its OWN matrix (`matrix_rows_llama`), never a fifth
#: member of the published design above (the four-family fixtures stay byte-identical)
LLAMA_KEY = "llama8b"
LLAMA_FAMILY = "llama"
NA: str = C.THINKING_NOT_AVAILABLE
MODE_CARD: str = C.SAMPLING_CARD
FAMILY_BY_KEY: Mapping[str, str] = {"gemma4e4b": "gemma4", "qwen35_4b": "qwen3_5", "nemotron3nano4b": "nemotron_h",
                                    "glm46vflash": "glm4v", LLAMA_KEY: LLAMA_FAMILY}
CELL_NONE, CELL_PROMPTED = C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY
CELLS_MAIN_GQ: tuple[str, ...] = (CELL_NONE, "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
                                  "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce",
                                  "safety_refusal__psr_kl_all", "safety_refusal__lora_r4",
                                  "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__lora_r4_ce_minus_ctl")
CELLS_IT_GQ: tuple[str, ...] = (CELL_NONE, CELL_PROMPTED, "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce")
CELLS_MAIN_NG: tuple[str, ...] = (CELL_NONE, CELL_PROMPTED, "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
                                  "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce",
                                  "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2-mix-full__lora_r4_ce_minus_ctl")
HELD_CELLS: tuple[str, ...] = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce")
HELD_REASON = "D8: the it-view constitution objects are being re-fitted at effective batch 64; run in the refit wave"
DECLARED_CELLS: tuple[str, ...] = ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4")
VIEWS_BY_KEY: Mapping[str, tuple[str, ...]] = {"gemma4e4b": ("it", "main"), "qwen35_4b": ("it", "main"),
                                               "nemotron3nano4b": ("main",), "glm46vflash": ("main",)}
SAMPLING_ROWS: Mapping[str, Mapping[str, Mapping[str, Any]]] = S.load_sampling_rows_yaml()
#: the four thinking families' facts (the published design's `--render-facts` table, unchanged) and, apart, the table
#: that ALSO carries the no-switch family's (the Llama fixtures'; `thinking_switch` False rides through `_asdict`)
RENDER_FACTS: Mapping[str, Mapping[str, Any]] = {fam: S._fact_row(fam) for fam in sorted({FAMILY_BY_KEY[k] for k in MODEL_KEYS})}
RENDER_FACTS_ALL: Mapping[str, Mapping[str, Any]] = {fam: S._fact_row(fam) for fam in sorted(set(FAMILY_BY_KEY.values()))}
#: the Llama matrix's cells — the SIX-ARM design (PSR out by default): none, the prompted safety baseline, the prompted
#: CONSTITUTION baseline (the short constitution SI as the system prompt), the two LoRA fits, the LoRA subtraction — in the
#: matrix's row order per view (g8k_views.cells_of: `none` first, the prompted safety baseline SECOND on an `it` view, the
#: roster's order otherwise); the row count is DERIVED from it, never pinned
CELL_PROMPTED_CONST = "prompted__si_constitution"
CELLS_LLAMA_BY_VIEW: Mapping[str, tuple[str, ...]] = {
    "main": (CELL_NONE, "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl",
             CELL_PROMPTED, CELL_PROMPTED_CONST),
    "it": (CELL_NONE, CELL_PROMPTED, "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl",
           CELL_PROMPTED_CONST),
}
CELLS_LLAMA: tuple[str, ...] = CELLS_LLAMA_BY_VIEW["main"]
LLAMA_VIEWS: tuple[str, ...] = ("main", "it")
LLAMA_WAVE_ROOT_REL = "exp/gsm8k_paired200_paper_objects_llama"


def cell_slug(cell: str) -> str:
    return cell.replace("__", "--")


def _hash(s: str, n: int = 64) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def tag_of(key: str, view: str) -> str:
    return key if view == "main" else f"{key}_{view}"


def staged_id_of(cell: str, view: str) -> str:
    if view == "it":
        return f"itdoc__{cell}"
    if cell.startswith("safety_refusal__"):
        return f"si12__{cell}"
    if cell.endswith(("_minus_ctl", "_distilled")):
        return f"derived__{cell}"
    return f"base__{cell}"


def object_pin(key: str, view: str, cell: str) -> str:
    """The synthetic (view, cell) pin — one per staged id."""
    return _hash(f"pin|{key}|{staged_id_of(cell, view)}")


def pins_doc(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """A synthetic ``configs/sdfbase_full22_pins.json`` carrying every fitted row's staged id."""
    out: dict[str, Any] = {k: {} for k in MODEL_KEYS}
    for r in (rows if rows is not None else matrix_rows()):
        if r["fitted"]:
            out.setdefault(str(r["model_key"]), {})[r["staged_id"]] = {"object_sha256": r["object_sha256"], "source": "synthetic"}
    return out


def tuple_name_of(key: str, *, declared: bool, cap: int = CAP_BASE) -> str:
    return f"gsm8k_paired200_{key}{'_declared' if declared else ''}_{cap}"


def _cells_of(key: str, view: str) -> tuple[str, ...]:
    if key in ("gemma4e4b", "qwen35_4b"):
        return CELLS_IT_GQ if view == "it" else CELLS_MAIN_GQ
    return CELLS_MAIN_NG


def matrix_rows() -> list[dict[str, Any]]:
    """The 42-row synthetic matrix."""
    rows: list[dict[str, Any]] = []
    for key in MODEL_KEYS:
        for view in VIEWS_BY_KEY[key]:
            tag = tag_of(key, view)
            for cell in _cells_of(key, view):
                fitted = C.is_lens_cell(cell)
                declared = fitted and cell in DECLARED_CELLS
                held = view == "it" and cell in HELD_CELLS
                short = {CELL_NONE: "none", CELL_PROMPTED: "prompted"}.get(cell, cell.replace("__", "-"))
                row: dict[str, Any] = {
                    "index": len(rows), "job_id": f"g8k-{key}-{view}-{short}", "model_key": key, "view": view, "tag": tag,
                    "cell": cell, "cell_slug": cell_slug(cell), "harness_cell": C.cell_spellings(cell, declared=False).harness,
                    "staged_id": staged_id_of(cell, view) if fitted else None,
                    "object_sha256": object_pin(key, view, cell) if fitted else None,
                    "fitted": fitted, "route_kind": "lens" if fitted else ("prompted" if cell == CELL_PROMPTED else "base"),
                    "served_positions_expected": ("response" if declared else "all") if fitted else None,
                    "declared": declared, "tuple_name": tuple_name_of(key, declared=declared) if fitted else None,
                    "sampling_mode": MODE_PRIMARY, "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]),
                    "out_root": f"controls/{tag}/{cell_slug(cell)}/{MODE_PRIMARY}",
                    "rung_out_root": f"controls/{tag}/{cell_slug(cell)}/{MODE_PRIMARY}__rung16384",
                    "time_limit": "0-01:30:00" if fitted else "0-01:00:00", "family": FAMILY_BY_KEY[key],
                    "reasoning_parser": "gemma4" if key == "gemma4e4b" else None,
                    "cluster_constraint": "primary" if key == "nemotron3nano4b" else "any",
                    "status": "held" if held else "launch", "held_reason": HELD_REASON if held else None,
                    "expected_refusal": False, "expected_refusal_reason": None,
                }
                rows.append(row)
    return rows


def staged_id_of_llama(cell: str, view: str) -> str:
    """The Llama staged ids (``80_stage_wave.sh`` / the pins block): the
    base objects TRANSFERRED onto the instruct target under ``main``
    (``base__<cell>``, derived ``derived__<cell>``), the instruct-direct objects
    under ``it`` (``itdoc__<cell>``, derived ``derived_itdoc__<cell>``)."""
    derived = cell.endswith(("_minus_ctl", "_distilled"))
    if view == "it":
        return f"{'derived_itdoc' if derived else 'itdoc'}__{cell}"
    return f"{'derived' if derived else 'base'}__{cell}"


def object_pin_llama(view: str, cell: str) -> str:
    return _hash(f"pin|{LLAMA_KEY}|{staged_id_of_llama(cell, view)}")


def matrix_rows_llama() -> list[dict[str, Any]]:
    """The synthetic Llama matrix in ``configs/sdfbase/gsm8k/matrix_llama.json``'s
    shape (schema 2; llama-paper-rows P6 / P7 / P11): ``llama8b`` in the ``main``
    view (the base-lane objects transferred: staged ids ``base__*`` /
    ``derived__*``) and the ``it`` view (instruct-direct: ``itdoc__*`` /
    ``derived_itdoc__*``), :data:`CELLS_LLAMA` in each; every row
    ``thinking_mode: not_available``, the ``card`` sampling row, NO tuple even
    when fitted, never declared, the ``unspecified`` stamp triples, the out roots
    under ``controls/<TAG>/<slug>/card``. Placeholder pins (one per staged id).
    The row count is the cell list's — never pinned."""
    rows: list[dict[str, Any]] = []
    key = LLAMA_KEY
    for view in LLAMA_VIEWS:
        tag = tag_of(key, view)
        for cell in CELLS_LLAMA_BY_VIEW[view]:
            fitted = C.is_lens_cell(cell)
            short = {CELL_NONE: "none", CELL_PROMPTED: "prompted", CELL_PROMPTED_CONST: "prompted-const"}.get(cell, cell.replace("constv2-mix-full__", "mix-").replace(
                "constv2ctl2-full__", "ctl-").replace("psr_ce_all", "psr").replace("lora_r4_ce", "lora"))
            rows.append({
                "index": len(rows), "job_id": f"g8k-{key}-{view}-{short}", "model_key": key, "view": view, "tag": tag,
                "cell": cell, "cell_slug": cell_slug(cell), "harness_cell": C.cell_spellings(cell, declared=False).harness,
                "staged_id": staged_id_of_llama(cell, view) if fitted else None,
                "object_sha256": object_pin_llama(view, cell) if fitted else None,
                "fitted": fitted, "route_kind": C.route_kind_of(cell),
                "served_positions_expected": "all" if fitted else None,
                "declared": False, "thinking_mode": NA, "trigger_threshold": TRIGGER_THRESHOLD_LLAMA, "tuple_name": None, "tuple_names": None,
                "sampling_mode": MODE_CARD, "sampling": dict(SAMPLING_ROWS[MODE_CARD][key]),
                "expected_stamps": list(C.g8k_expected_stamps(fitted=fitted, declared=False, arm=NA)),
                "out_root": f"controls/{tag}/{cell_slug(cell)}/{MODE_CARD}",
                "rung_out_root": f"controls/{tag}/{cell_slug(cell)}/{MODE_CARD}__rung16384",
                "time_limit": "0-01:00:00" if (fitted or cell == CELL_PROMPTED_CONST) else "0-00:45:00", "family": LLAMA_FAMILY,
                "split_mode_candidate": C.THINK_SPLIT_NONE, "reasoning_parser": None, "cluster_constraint": "any",
                "status": "launch", "held_reason": None, "expected_refusal": False, "expected_refusal_reason": None,
            })
    return rows


def matrix_doc_llama(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The schema-2 DOCUMENT around :func:`matrix_rows_llama` (lane B's
    ``make_gsm8k_matrix.py`` shape: ``schema``, ``wave_root_rel``, ``rows``)."""
    return {"schema": "g8k_matrix/2", "wave_root_rel": LLAMA_WAVE_ROOT_REL, "trigger_threshold": TRIGGER_THRESHOLD_LLAMA,
            "views": {LLAMA_KEY: list(LLAMA_VIEWS)}, "rows": list(rows if rows is not None else matrix_rows_llama())}


def expected_refusals_doc(rows: Sequence[Mapping[str, Any]] | None = None, *, model_key: str = "gemma4e4b",
                          code: str = EXPECTED_REFUSAL_CODE) -> dict[str, str]:
    """A ``--expected-refusals`` document naming one model's DECLARED main rows
    (the two header-located si12 cells) — the earlier situation, kept as a
    tested capability of the reader: ``{8,192 leg key: refusal code}``."""
    return {f"{r['model_key']}/{r['view']}/{r['cell']}@{CAP_BASE}": code
            for r in (rows if rows is not None else matrix_rows()) if r["model_key"] == model_key and r["declared"] and r["status"] == "launch"}


def expected_refusal_triples(rows: Sequence[Mapping[str, Any]] | None = None, *, model_key: str = "gemma4e4b") -> list[tuple[str, str, str]]:
    return [(r["model_key"], r["view"], r["cell"]) for r in (rows if rows is not None else matrix_rows())
            if r["model_key"] == model_key and r["declared"] and r["status"] == "launch"]


def write_matrix(path: Path, rows: Sequence[Mapping[str, Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows if rows is not None else matrix_rows()), indent=1), encoding="utf-8")
    return path


def row_for(rows: Sequence[Mapping[str, Any]], key: str, view: str, cell: str) -> dict[str, Any]:
    for r in rows:
        if r["model_key"] == key and r["view"] == view and r["cell"] == cell:
            return dict(r)
    raise KeyError(f"{key}/{view}/{cell} is not in the matrix")


def launch_rows(rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [dict(r) for r in (rows if rows is not None else matrix_rows()) if r["status"] == "launch"]


@dataclass
class AttemptSpec:
    """One recorded run of a store: its id, kind, whether it
    completed, and — for a failed GENERATION attempt — how many items it
    issued before dying and which of them the proxy rejected."""

    attempt_id: str
    kind: str = "generation"
    completed: bool = True
    accepted: bool = False
    n_items: int | None = None
    rejected_items: Sequence[int] = field(default_factory=tuple)


def default_attempts() -> list[AttemptSpec]:
    return [AttemptSpec("a1", accepted=True)]


def _arm(thinking_on: bool, arm: str | None) -> str:
    return str(arm) if arm is not None else (C.THINKING_ON if thinking_on else C.THINKING_OFF)


def stamps_for(fitted: bool, declared: bool, *, thinking_on: bool = True, arm: str | None = None, native: bool = False) -> tuple[str, ...]:
    """The stamp triple the proxy puts on a leg's responses: the non-lens ``not-consulted`` triple on an unsteered route
    AND on a NATIVELY served fitted cell (the engine's own triple — no lens selector on a lora route; native-serving-
    default), the declared row on a declared lens cell, the lens ``always`` row on an undeclared lens cell."""
    arm = _arm(thinking_on, arm)
    if not fitted or native:
        return tuple(C.expected_stamp_regimes(lens=False, arm=arm, policy=C.POLICY_NOT_CONSULTED))
    if declared:
        return tuple(C.expected_stamp_regimes(lens=True, arm=arm, policy=C.DECLARED_SALT_POLICY_LABEL))
    return tuple(C.expected_stamp_regimes(lens=True, arm=arm, policy="always"))


def _sampling_row(mode: str, key: str) -> C.SamplingRow:
    if mode == C.SAMPLING_GREEDY:
        return C.GREEDY_SAMPLING
    return S.sampling_row_of(SAMPLING_ROWS[mode][key])


def sampling_block(mode: str, key: str) -> dict[str, Any]:
    srow = _sampling_row(mode, key)
    return {"mode": mode, "model_key": key, **C.sampling_identity(srow, {TASK: 1}), "source": srow.source}


def task_plan(mode: str, key: str, ids: Sequence[str], fact: Mapping[str, Any], *, cap: int, thinking_on: bool,
              arm: str | None = None) -> isv.TaskPlan:
    """The runner's plan for the leg: the arm, the cap, the row; the split mode
    the FACT names under thinking-ON and under the no-switch arm (``none`` —
    the plan's mode is what makes ``outcome_record`` file ``stray_close_tag``),
    ``record_only`` under OFF."""
    arm = _arm(thinking_on, arm)
    srow = _sampling_row(mode, key)
    split = fact["split_mode"] if arm != C.THINKING_OFF else C.THINK_SPLIT_RECORD_ONLY
    return isv.TaskPlan(row_task=TASK, name=TASK_NAME, task=None, sample_ids=list(ids), samples_per_item=1, limit=None,
                        max_tokens=cap, primary_scorer=None, expected_n=len(ids), n_run=len(ids),
                        temperature=srow.temperature, top_p=srow.top_p, top_k=srow.top_k, item_set=C.GSM8K_ITEM_SET_PAIRED,
                        paired_ids=dict(PAIRED_BY_INSPECT_ID),
                        thinking=arm, retry_max_tokens=0, grader_max_tokens=None,
                        think_split_mode=split, family=fact["family"],
                        render_evidence_status=fact["evidence_status"],
                        **{f: getattr(srow, f) for f in C.SAMPLING_OPTIONAL_ROW_FIELDS})


def standup_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    """The standup record as the exporter and ``capctrl_serving_up`` write it:
    the served-positions map under the ONE spelling the standup serves the
    cell as (the bare harness id undeclared, ``<harness>@thinking`` declared —
    never both; ``proxy.lens_served_positions`` keys by the leg's ``cell_id``),
    the declarations map naming the declared cell's harness spelling, the
    digest twice, the policy."""
    declared = bool(row["declared"])
    spell = C.cell_spellings(str(row["cell"]), declared=declared)
    pin = str(row["object_sha256"])
    positions = [str(row["served_positions_expected"])]
    decl = {spell.harness: {C.SERVE_OUT_OF_SUPPORT_KEY: list(C.OOS_CELL_DECLARATION_EXPECTED)}} if declared else {}
    policy = C.DECLARED_SALT_POLICY_LABEL if declared else "always"
    return {C.PROXY_IDENTITY_SERVE_DECLARATIONS: decl, C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
            C.LENS_SERVED_POSITIONS_KEY: {spell.harness: positions},
            C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: pin, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {spell.harness: pin},
            C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {spell.harness: policy}}


def lens_position_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Predicate (B′)'s record THROUGH ``contracts.lens_position_attested`` —
    the UNDECLARED all-position cell's predicate. A DECLARED header-located
    cell has none: the driver runs (B′)/(C) only where ``not header_located``
    (``served_capability_run.py`` 4726), so the fixture writes none there
    (it used to hand-write the helper's success shape with the declared class's
    facts — a record the producer never emits)."""
    assert not bool(row["declared"]), "predicate (B′) is the undeclared cell's record — a declared cell carries thinking_header_by_arm"
    return C.lens_position_attested(standup_endpoint(row), cell=str(row["cell"]), expected_sha256=str(row["object_sha256"]),
                                    model_key=str(row["model_key"]), arm=_arm(True, row.get("thinking_mode")))


def thinking_header_by_arm_record(fact: Mapping[str, Any]) -> dict[str, Any]:
    """Predicate (B)'s record on a DECLARED header-located cell: the driver's
    ``thinking_header_by_arm`` block (``measure_thinking_header_by_arm`` +
    ``_certify_header`` at 8f5f1495) — a lens measurement through the served
    ``/v1/tokenize``, both arms, the certification flags true, the attested
    proxy header equal to the fact's. ``verified`` is the header-INVARIANCE
    verdict: true on an invariant family (gemma), FALSE on a moving-header
    family (qwen) with the lifted mixed-arm refusal recorded beside it — the
    shape the out-of-support override writes, which the reader holds to the
    family fact rather than reading as a pass."""
    off, on = [int(i) for i in fact["off_ids"]], [int(i) for i in fact["on_ids"]]
    moves = off != on
    rec: dict[str, Any] = {
        "lens": True, "source": HEADER_SOURCE_SERVED, "fallback_reason": None, "served_chat_template_hash": "0a" * 8,
        "local_chat_template_hash": None, "header_len_by_arm": {"off": len(off), "on": len(on)}, "header_ids_by_arm": {"off": off, "on": on},
        "invariant": not moves, "header_moves": moves, "run_arms": [C.THINKING_ON], "measured_arms": [C.THINKING_OFF, C.THINKING_ON],
        "template_comparison": {"required": False, "ran": False, "equal": None, "unavailable": ["local_chat_template_hash"],
                                "reason": "local_chat_template_hash is null — the local tokenizer's chat template could not be hashed"},
        "out_of_support_thinking": True, "out_of_support_thinking_note": "synthetic: the override's note",
    }
    if moves:
        rec["overridden_refusal"] = ("synthetic: this run mixes thinking arms on a LENS route, but the served template's assistant header "
                                     f"MOVES between them: {rec['header_len_by_arm']} — lifted by the out-of-support override, recorded")
    rec.update({"verified": not moves, "prompt_opened_think_declared": bool(fact.get("prompt_opens_think_on")), "prompt_opened_think_certified": True,
                "header_certified": True, "header_attested_agrees": True, "attested_switch": True, "attested_moves": moves,
                "attested_thinking_header": {"switch": True, "moves": moves, "off_ids": off, "on_ids": on}, "thinking_header_error": None,
                "expected_header_ids_by_arm": {"off": off, "on": on}, "certification_problems": []})
    return rec


def steering_bites_record_declared(cell: str, *, n_differing: int = 5) -> dict[str, Any]:
    """The driver's ``steering_bites`` block on a DECLARED cell: the
    probe rendered thinking-ON with the declaration on both legs, every reply
    stamped with the declared triple."""
    rec = S.steering_bites_record(cell, n_differing=n_differing)
    stamps = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, stamps_for(True, True)))
    rec[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = C.THINKING_ON
    rec["probe_extra_body"] = C.render_fragment(lens=True, arm=C.THINKING_ON, header_located=True)
    rec["stamps_by_leg"] = {"steered": [dict(stamps) for _ in range(S.N_BITES_PROMPTS)], "scale0": [dict(stamps) for _ in range(S.N_BITES_PROMPTS)]}
    return rec


def steering_bites_record_no_switch(cell: str, *, n_differing: int = 4) -> dict[str, Any]:
    """The driver's ``steering_bites`` block on a fitted NO-SWITCH cell (P18, (C″)):
    the ordinary positive control under the family's ONE render —
    ``probe_render_arm not_available``, an EMPTY probe body fragment
    (``contracts.render_fragment`` under the no-switch arm), every reply on both
    legs stamped ``render_regime unspecified``."""
    rec = S.steering_bites_record(cell, n_differing=n_differing)
    stamps = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, stamps_for(True, False, arm=NA)))
    rec[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = NA
    rec["probe_extra_body"] = C.render_fragment(lens=True, arm=NA, header_located=False)
    rec["stamps_by_leg"] = {"steered": [dict(stamps) for _ in range(S.N_BITES_PROMPTS)], "scale0": [dict(stamps) for _ in range(S.N_BITES_PROMPTS)]}
    return rec


NO_SWITCH_PROBE_COMPLETION: Mapping[str, Any] = {"content": "Blue.", "reasoning": None, "finish_reason": "stop"}


def render_certified_record_no_switch(family: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    """Predicate (A″)'s record THROUGH ``contracts.no_switch_render_certified``: the
    header measured ONCE live (``measured_live``), the one probe stamped
    ``unspecified``, its completion free of think tags and of a reasoning channel."""
    return C.no_switch_render_certified(family=family, header_ids=list(fact["off_ids"]), measured_live=True,
                                        probe_regime=C.RENDER_REGIME_UNSPECIFIED, probe_completion=dict(NO_SWITCH_PROBE_COMPLETION))


def render_probe_record_no_switch() -> dict[str, Any]:
    """The driver's ``render_probe`` block under the no-switch class
    (``_no_switch_render_probe``): the ONE probe under the one arm, no ON draws."""
    return {NA: {**NO_SWITCH_PROBE_COMPLETION, "render_regime": C.RENDER_REGIME_UNSPECIFIED}, "on_attempts": []}


def no_switch_header_record(fact: Mapping[str, Any], *, lens: bool) -> dict[str, Any]:
    """The driver's ``thinking_header_by_arm`` block under the no-switch class
    (``measure_no_switch_header`` at this tree): the header measured ONCE through
    the served ``/v1/tokenize`` under the one arm, ``cert_class no_switch``,
    ``header_moves`` False, ``invariant`` and ``verified`` True."""
    ids = [int(i) for i in fact["off_ids"]]
    return {"lens": bool(lens), "source": HEADER_SOURCE_SERVED, "fallback_reason": None, "served_chat_template_hash": "0a" * 8,
            "local_chat_template_hash": None, "header_len_by_arm": {NA: len(ids)}, "header_ids_by_arm": {NA: ids},
            "invariant": True, "header_moves": False, "run_arms": [NA], "measured_arms": [NA], "cert_class": C.CERT_CLASS_NO_SWITCH,
            "template_comparison": {"required": False, "ran": False, "equal": None, "unavailable": ["local_chat_template_hash"],
                                    "reason": "local_chat_template_hash is null — the local tokenizer's chat template could not be hashed"},
            "verified": True}


def _declared_probe_attempt(vmodel: str, *, kind: str, n: int, arm: str, prompt: str, max_tokens: int, ts: str, salt: str) -> S.Attempt:
    """A labelled probe attempt on a DECLARED lens route: the arm's render
    fragment WITH the ``petri_thinking_on`` declaration under ON (the driver's
    ``_probe_extra_body(header_located=True)``), the MATH builder's labels."""
    body = {"model": vmodel, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_tokens, "seed": 0,
            **C.render_fragment(lens=True, arm=arm, header_located=True)}
    label = C.probe_task_label(kind)
    return S.attempt(body, task=label, sample=f"{vmodel}/{n}", ts=ts, request_id=S._request_id(salt, ts, label, vmodel, n),
                     prompt_tokens=30, completion_tokens=8)


def standup_job_id(row: Mapping[str, Any], cap: int) -> str:
    """The synthetic Slurm job id of a leg's standup (the rung's differs from the base's)."""
    return str(STANDUP_JOB_BASE + int(row.get("index", 0)) + (1000 if int(cap) == CAP_RUNG else 0))


def standup_jobdir(row: Mapping[str, Any], cap: int) -> str:
    """``g8k_<TAG>_<cell slug>_<cap>_<job>`` — the launcher's standup directory name."""
    return f"g8k_{row['tag']}_{cell_slug(str(row['cell']))}_{int(cap)}_{standup_job_id(row, cap)}"


def cluster_routing_table_path(row: Mapping[str, Any], cap: int) -> str:
    """The ABSOLUTE cluster path the manifest's ``routing_table`` / ``binding.routing_table`` name."""
    return f"{CLUSTER_WAVE_ROOT}/{STANDUP_DIR_PARENT}/{standup_jobdir(row, cap)}/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}"


def standup_record_docs(row: Mapping[str, Any], fact: Mapping[str, Any], *, cap: int, tree_sha: str, intervention_hash: str,
                        max_model_len: int) -> dict[str, Any]:
    """The four standup-record documents of a DECLARED leg's job, keyed by their
    file name under ``serving/<jobdir>/gsm8k/`` (the cluster's shape at 8f5f1495):
    the exporter's ``routing_table.json`` (``lens_served_positions`` under the
    declared key ONLY, the three routes with their intervention hashes, the
    thinking header), the lens registry (the cell and its scale-0 twin with
    ``resolved_spec.positions`` the declared mode, the declaration, the
    artifact digest's first 16 hex), the workers' attestation bound to the
    Slurm job id, and ``standup_jobs.jsonl`` binding the job to the tree."""
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    spell = C.cell_spellings(cell, declared=True)
    hf, rev = S.model_pin(str(row["tag"]))
    off, on = [int(i) for i in fact["off_ids"]], [int(i) for i in fact["on_ids"]]
    job, jobdir = standup_job_id(row, cap), standup_jobdir(row, cap)
    pin, mode = str(row["object_sha256"]), str(row["served_positions_expected"])
    arm = cell.split("__")[-1]
    kind = "lora_all" if "lora" in arm else "psr"
    spec = {"arm": arm, "kind": kind, "layers": [0, 1, 2, 3], "positions": mode, "response_header_len": len(off),
            C.SERVE_OUT_OF_SUPPORT_KEY: list(C.OOS_CELL_DECLARATION_EXPECTED), "si_id": "safety"}

    def entry(cell_id: str, *, donor: str | None, expect_hash: str, forced_scale: float | None) -> dict[str, Any]:
        return {"artifact": f"{spell.harness}/artifact.bin", "artifact_intervention_hash": intervention_hash, "artifact_sha16": pin[:16],
                "cell_id": cell_id, "donor_cell": donor, "expect_hash": expect_hash, "forced_scale": forced_scale, "kind": kind,
                "positions_at_fit": None, "resolved_spec": dict(spec), "sidecar": None, "validated_by": None}

    registry = {"assistant_header_ids": off, "assistant_header_ids_thinking_on": on, "assistant_turn_end_ids": None,
                "entries": [entry(spell.harness, donor=None, expect_hash=intervention_hash, forced_scale=None),
                            entry(C.TWIN_CELL_ID, donor=spell.harness, expect_hash=BASE_INTERVENTION_HASH, forced_scale=0.0)], "version": 1}
    routes = [{"cell_class": "base", "cell_id": CELL_NONE, "control_of": None, "intervention_hash": BASE_INTERVENTION_HASH,
               "serving_identity": "proxy_vllm+passthrough+interleaved", "si_prefix_hash": None,
               "virtual_model": C.cell_spellings(CELL_NONE, declared=False).vmodel},
              {"cell_class": "class_a", "cell_id": spell.harness, "control_of": None, "intervention_hash": intervention_hash,
               "serving_identity": "proxy_vllm+lens+interleaved", "si_prefix_hash": None, "virtual_model": spell.vmodel},
              {"cell_class": "control", "cell_id": C.TWIN_CELL_ID, "control_of": CELL_NONE, "intervention_hash": BASE_INTERVENTION_HASH,
               "serving_identity": "proxy_vllm+lens+interleaved", "si_prefix_hash": None, "virtual_model": C.twin_vmodel()}]
    table = {"base_model": hf, "base_revision": rev, "exact_replay_strategy": None, C.LENS_SERVED_POSITIONS_KEY: {spell.harness: [mode]},
             "prompted_prefixes": [], "routes": routes, "serving_mode": "interleaved",
             "thinking_header": {"moves": off != on, "off_ids": off, "on_ids": on, "switch": True}, "thinking_header_error": None, "version": 1}
    decl = {C.TWIN_CELL_ID: {C.SERVE_OUT_OF_SUPPORT_KEY: list(C.OOS_CELL_DECLARATION_EXPECTED)},
            spell.harness: {C.SERVE_OUT_OF_SUPPORT_KEY: list(C.OOS_CELL_DECLARATION_EXPECTED)}}
    attestation = {"slurm_job_id": job, "hostname": "TESTHOST", "host": "127.0.0.1", "port": S.launch_port(job), "model": hf, "n_workers": 1,
                   "attested_at": 0.0, "argv_sha256": sha256_hex16(f"synthetic argv {jobdir}".encode()), "nonce": _hash(f"nonce|{jobdir}", 32),
                   "expected": {"cells": [C.TWIN_CELL_ID, spell.harness], "ledger_cells": [spell.harness], "registration_count": 1},
                   "registry_dir": f"{CLUSTER_WAVE_ROOT}/{STANDUP_DIR_PARENT}/{jobdir}/{STANDUP_TASK_SUBDIR}/lens_registry",
                   "registry_index_sha16": None,        # sealed over the registry bytes by seal_standup_record
                   "worker_reports": [{"cells": [C.TWIN_CELL_ID, spell.harness], "ledger_cells": [spell.harness], "lens_serve_declarations": decl,
                                       "lens_reduction_mode": "stacked"}]}
    jobs = [{"job_id": job, "node": "TESTHOST", "key": key, "view": view, "mode": "attacks", "eval": "capability:gen_long_thinking", "sha": tree_sha,
             "lens_reduction_mode": "stacked", "max_num_seqs": S.served_seqs(key, max_model_len), "max_model_len": int(max_model_len),
             "cluster": "synthetic", "written_at": TS0}]
    return seal_standup_record({STANDUP_ROUTING_TABLE: table, STANDUP_REGISTRY: registry, STANDUP_ATTESTATION: attestation, STANDUP_JOBS: jobs})


def standup_doc_bytes(name: str, doc: Any) -> bytes:
    """ONE serialization for every standup document — the bytes
    :func:`write_standup_record` writes and :func:`seal_standup_record` hashes."""
    if name.endswith(".jsonl"):
        return "".join(json.dumps(r) + "\n" for r in doc).encode("utf-8")
    return json.dumps(doc, indent=1).encode("utf-8")


def seal_standup_record(docs: dict[str, Any]) -> dict[str, Any]:
    """Bind the attestation to the registry it is written beside, the way the
    producer does at launch (``vllm_server._attestation_bindings``:
    ``registry_index_sha16 = sha256_hex16(registry.json bytes)``) — over the
    exact bytes :func:`write_standup_record` will write. Re-applied after a
    ``standup_record_edit``, so an edited registry stays a CONSISTENT record
    (refused on its field, not on the seal); a test that wants a STALE
    attestation rewrites the registry file after the store is built."""
    docs[STANDUP_ATTESTATION]["registry_index_sha16"] = sha256_hex16(standup_doc_bytes(STANDUP_REGISTRY, docs[STANDUP_REGISTRY]))
    return docs


def write_standup_record(gen_root: Path, jobdir: str, docs: Mapping[str, Any]) -> Path:
    """The standup record under ``<gen_root>/serving/<jobdir>/gsm8k/`` — where a
    read-only pull of the cluster's ``serving/`` puts it beside the stores."""
    d = Path(gen_root) / STANDUP_DIR_PARENT / jobdir / STANDUP_TASK_SUBDIR
    for name, doc in docs.items():
        path = d / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(standup_doc_bytes(name, doc))
    return d


#: the NATIVE standup's record (native-positions-resolve): the same ``serving/<jobdir>/gsm8k/`` layout, but the engine's
#: own documents — no lens registry, the SERVING attestation instead of the steering one (pinned equal to the reader's)
STANDUP_SERVING_ATTESTATION = "serving_attestation.json"
#: the native engine's route identity and the engine's zero adapter (the control route), as the native standup stamps them
NATIVE_SERVING_IDENTITY = "proxy_vllm+lora+interleaved"
NATIVE_STEER_POSITIONS = "all"


def native_adapter_specs(row: Mapping[str, Any], *, intervention_hash: str) -> dict[str, dict[str, Any]]:
    """The two adapters a synthetic native standup of ``row`` serves, in ``--lora-modules`` order (the engine's zero
    adapter at id 1, the served cell's harness name at id 2), under the cluster wave root's ``adapters/<model>/`` — ONE
    spelling for the strict attestation (:func:`S.native_serving_attestation_doc`), the routing table's ``adapters`` block
    and the native launch echo's ``lora_modules`` (:func:`S.native_lora_modules`)."""
    key, cell = str(row["model_key"]), str(row["cell"])
    spell = C.cell_spellings(cell, declared=False)
    specs = S.native_adapter_specs(key, spell.harness, intervention_hash=intervention_hash, source_kind=cell.split("__")[-1].split("_")[0],
                                   zero_name=C.TWIN_CELL_ID_LORA)
    root = f"{CLUSTER_WAVE_ROOT}/adapters/{key}"                 # the routing table's adapter_root spelling (below)
    for name, spec in specs.items():
        spec["dir"] = f"{root}/{name}"
    return specs


def native_standup_record_docs(row: Mapping[str, Any], fact: Mapping[str, Any], *, cap: int, tree_sha: str, intervention_hash: str,
                               max_model_len: int, position_mode: str | None = "all",
                               served_positions_map: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The NATIVE standup's record documents beside a natively served UNDECLARED
    fitted leg's job, keyed by file name under ``serving/<jobdir>/gsm8k/`` — key
    for key the shape a real native card's record carries on the cluster, with
    synthetic values: the exporter's ``routing_table.json`` (version 2; the
    ``adapters`` block; ``lens_served_positions: {}`` — a native standup states
    NO lens mode; the three routes, the served cell's a ``+lora+`` identity
    carrying the attested ``petri_position_mode`` — ``position_mode`` None
    OMITS the key, an unattested route; the family's thinking header),
    ``serving_attestation.json`` (kind ``lora_native``, the Slurm job id, the
    adapter id map) and ``standup_jobs.jsonl`` (engine ``lora_native``, the
    tree). ``served_positions_map`` writes the ENGINE-NEUTRAL ``served_positions``
    map the native-route producer fix stamps (``{harness key: mode}``); absent
    by default — the existing native cards' shape."""
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    spell = C.cell_spellings(cell, declared=False)
    hf, rev = S.model_pin(str(row["tag"]))
    off = [int(i) for i in fact["off_ids"]]
    switch = bool(fact.get("thinking_switch"))
    on = [int(i) for i in fact["on_ids"]] if switch else None
    job, jobdir = standup_job_id(row, cap), standup_jobdir(row, cap)
    twin = C.TWIN_CELL_ID_LORA
    adapter_root = f"{CLUSTER_WAVE_ROOT}/adapters/{key}"

    def adapter(name: str, *, ihash: str, zero: bool) -> dict[str, Any]:
        return {"config_sha256": _hash(f"config|{name}|{jobdir}"), "dir": f"{adapter_root}/{name}", "name": name, "petri_intervention_hash": ihash,
                "petri_source_kind": (None if zero else cell.split("__")[-1].split("_")[0]), "petri_zero_adapter": zero, "r": 1,
                "weights_sha256": _hash(f"weights|{name}|{jobdir}")}

    served_route: dict[str, Any] = {"cell_class": "class_a", "cell_id": spell.harness, "control_of": None, "intervention_hash": intervention_hash,
                                    "serving_identity": NATIVE_SERVING_IDENTITY, "si_prefix_hash": None, "virtual_model": spell.vmodel}
    if position_mode is not None:
        served_route["petri_position_mode"] = str(position_mode)           # the exporter writes the key only when a mode is attested
    routes = [{"cell_class": "base", "cell_id": CELL_NONE, "control_of": None, "intervention_hash": BASE_INTERVENTION_HASH,
               "serving_identity": "proxy_vllm+passthrough+interleaved", "si_prefix_hash": None,
               "virtual_model": C.cell_spellings(CELL_NONE, declared=False).vmodel},
              served_route,
              {"cell_class": "control", "cell_id": twin, "control_of": CELL_NONE, "intervention_hash": BASE_INTERVENTION_HASH,
               "serving_identity": NATIVE_SERVING_IDENTITY, "si_prefix_hash": None, "virtual_model": f"petri__{twin}"}]
    table: dict[str, Any] = {
        "adapters": {twin: adapter(twin, ihash=BASE_INTERVENTION_HASH, zero=True), spell.harness: adapter(spell.harness, ihash=intervention_hash, zero=False)},
        "base_model": hf, "base_revision": rev, "exact_replay_strategy": None, C.LENS_SERVED_POSITIONS_KEY: {}, "prompted_prefixes": [],
        "routes": routes, "serving_mode": "interleaved",
        "thinking_header": {"moves": (off != on) if switch else None, "off_ids": off, "on_ids": on, "switch": switch},
        "thinking_header_error": None, "version": 2}
    if served_positions_map is not None:
        table["served_positions"] = dict(served_positions_map)
    served_s = S.served_seqs(key, max_model_len)
    # the STRICT-VALID record (launch-attestation-native): exactly the strict reader's schema-2 key set, the same two adapters
    # (the zero carrier at id 1, the served cell at id 2) and dirs the routing table names, the binding fields the launch
    # attestation reads (`slurm_job_id`, `port`, `argv_sha256`) at the top level as in the real record
    attestation = S.native_serving_attestation_doc(job=job, port=S.launch_port(job), model=hf, revision=rev,
                                                   argv_sha256=_hash(f"synthetic argv {jobdir}"), nonce=_hash(f"nonce|{jobdir}", 32),
                                                   adapters=native_adapter_specs(row, intervention_hash=intervention_hash),
                                                   max_model_len=int(max_model_len), max_num_seqs=served_s)
    jobs = [{"job_id": job, "node": "TESTHOST", "key": key, "view": view, "mode": "attacks", "eval": "capability:gen_long_thinking", "sha": tree_sha,
             "engine": "lora_native", "serving_path": "lora_native", "lens_reduction_mode": None, "max_num_seqs": served_s,
             "max_model_len": int(max_model_len), "cluster": "synthetic", "thinking_mode": str(row.get("thinking_mode") or "on"), "written_at": TS0}]
    return {STANDUP_ROUTING_TABLE: table, STANDUP_SERVING_ATTESTATION: attestation, STANDUP_JOBS: jobs}


def native_lens_position_record() -> dict[str, Any]:
    """Predicate (B′)'s record on a NATIVE store — the DRIVER's own inapplicability record
    (``served_capability_run._native_lens_position_record``), imported from the tree's driver so the fixture
    cannot spell a literal the producer does not write."""
    import importlib.util
    import sys as _sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in _sys.path:
        _sys.path.insert(0, str(scripts))
    mod = _sys.modules.get("served_capability_run")
    if mod is None:
        spec = importlib.util.spec_from_file_location("served_capability_run", scripts / "served_capability_run.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["served_capability_run"] = mod
        spec.loader.exec_module(mod)
    return dict(mod._native_lens_position_record())


def native_bites_record(block: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    """A lens fixture's steering-bites block re-stamped as the NATIVE driver files it: the ENGINE's zero cell
    (``none__lora0``) as the twin, the non-lens probe body (``render_fragment(lens=False, ...)`` — no declaration
    rides an undeclared probe), every reply on both legs stamped with the proxy's non-lens ``not-consulted`` policy
    under the probe's own render regime."""
    stamps = {**dict(block["stamps_by_leg"]["steered"][0]), "resumed_policy": C.POLICY_NOT_CONSULTED}
    return {**dict(block), C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID_LORA,
            "probe_extra_body": C.render_fragment(lens=False, arm=arm, header_located=False),
            "stamps_by_leg": {"steered": [dict(stamps) for _ in range(S.N_BITES_PROMPTS)],
                              "scale0": [dict(stamps) for _ in range(S.N_BITES_PROMPTS)]}}


def nativize_manifest(manifest: dict[str, Any], row: Mapping[str, Any], fact: Mapping[str, Any], *, arm: str,
                      served_positions: Sequence[str] | None = None) -> dict[str, Any]:
    """Re-stamp a fitted lens manifest :func:`build_store` wrote as the NATIVE
    driver files a natively served undeclared fitted cell: the ``+lora+`` route
    identity with ``serving_engine: lora_native`` and the ``not_applicable``
    reduction mode, NO declaration map and a NULL served-positions map in the
    route block and at the top level (the engine's facts), ``steer_positions``
    every position, the top-level ``header_located`` the ENGINE's False, and
    the transfer-provenance block THROUGH ``contracts.transfer_provenance_block``
    with ``served_positions`` = what the driver's lookup returned — None by
    default, the producer's record gap (the helper then derives
    ``header_located`` / ``header_check_class`` fail-closed from it, exactly as
    the real record does); a list simulates the fixed producer."""
    route = manifest[C.MANIFEST_KEY_ROUTE]
    route.update({"serving_identity": NATIVE_SERVING_IDENTITY, C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE,
                  C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_NOT_APPLICABLE, "reduction_mode_expectation": "not_consulted",
                  C.ROUTE_KEY_SERVE_DECLARATIONS: None, C.LENS_SERVED_POSITIONS_KEY: None})
    manifest.update({"serving_identity": NATIVE_SERVING_IDENTITY, "serving_path": "lora", C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE,
                     "steer_positions": NATIVE_STEER_POSITIONS, "header_located": False, C.LENS_SERVED_POSITIONS_KEY: None,
                     C.MANIFEST_KEY_TRANSFER_PROVENANCE: transfer_provenance_record(row, fact, served_positions=served_positions, arm=arm)})
    return manifest


def lens_on_bite_record(row: Mapping[str, Any], fact: Mapping[str, Any]) -> dict[str, Any]:
    lens = {**S.think_shaped_completion(fact, "Rayleigh scattering."), "render_regime": C.RENDER_REGIME_THINKING_ON}
    base = {**S.think_shaped_completion(fact, "Because of scattering."), "render_regime": C.RENDER_REGIME_THINKING_ON}
    rec = C.lens_on_bite_verdict(lens_reply=lens, base_reply=base, family=fact["family"])
    rec["lens_vmodel"] = C.cell_spellings(str(row["cell"]), declared=bool(row["declared"])).vmodel
    rec["base_vmodel"] = C.cell_spellings(CELL_NONE, declared=False).vmodel
    return rec


def transfer_provenance_record(row: Mapping[str, Any], fact: Mapping[str, Any], *, served_positions: Sequence[str] | None,
                               arm: str = C.THINKING_ON) -> dict[str, Any]:
    """The provenance block THROUGH ``contracts.transfer_provenance_block`` with the
    driver's arguments — ``served_positions`` is the value the driver's lookup
    returned (the producer's NULL on a declared cell at 8f5f1495; the helper
    then reads ``header_located(None)`` fail-closed, exactly as the real
    record does). Under the no-switch arm the served render is the arm's
    (``unspecified``) and the header ids the ONE render under the one arm — the
    driver's ``early_header`` block (``measure_no_switch_header``)."""
    ids = ({NA: list(fact["off_ids"])} if arm == NA else {"off": list(fact["off_ids"]), "on": list(fact["on_ids"])})
    return C.transfer_provenance_block(fit_kind="document_fit", fit_template_hash=None, fit_response_header_len=None,
                                       served_positions=(list(served_positions) if served_positions is not None else None),
                                       served_render_arm=arm, served_header_ids_by_arm=ids, family=fact["family"])


def _with_attempt(record: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    """Stamp lane H's ``attempt_id`` beside the producer's record (the column
    is appended after the producer's own schema, never inside it)."""
    out = dict(record)
    out[LEDGER_ATTEMPT_ID_FIELD] = attempt_id
    return out


def native_twin_vmodel() -> str:
    """The NATIVE engine's zero cell's virtual model (``none__lora0`` — the zero adapter the bites control compares
    against on a lora route), through the contract's own engine-keyed rule (``contracts.twin_vmodel_for``)."""
    return C.twin_vmodel_for(NATIVE_SERVING_IDENTITY)


def probe_attempts(row: Mapping[str, Any], *, fitted: bool, declared: bool, ts: str, salt: str,
                   arm: str = C.THINKING_ON, native: bool = False) -> list[S.Attempt]:
    """Every probe attempt ONE run of the driver sends (the MATH builder's
    inventory: bites on both legs OFF, render OFF + ON on the route, the lens-ON
    bite on the lens and base routes), under the cell's spelling. Under the
    NO-SWITCH arm (P18) the class's EXACT inventory: ONE render probe under
    ``not_available`` (a body with no kwargs — ``contracts.render_fragment`` is
    empty there), on a fitted cell the bites probe on the route and the twin
    under it, NO OFF / ON render probe and NO lens-ON bite. ``native`` (a
    natively served fitted cell): the bites twin is the ENGINE's zero cell
    (``none__lora0``); every body is the same bytes (no declaration rides an
    undeclared probe on either engine) and the (C) bite is engine-neutral."""
    cell = str(row["cell"])
    vm = C.cell_spellings(cell, declared=declared).vmodel
    twin = native_twin_vmodel() if native else C.twin_vmodel()
    out: list[S.Attempt] = []
    if arm == NA:
        assert not declared, "a declared no-switch cell is a matrix defect"
        if fitted:
            for target in (vm, twin):
                for i in range(S.N_BITES_PROMPTS):
                    out.append(S.probe_attempt(target, kind=C.PROBE_KIND_BITES, n=i, arm=NA, lens=True,
                                               prompt=f"probe prompt {i}", max_tokens=S.BITES_PROBE_MAX_TOKENS, ts=ts, salt=salt))
        out.append(S.probe_attempt(vm, kind=C.PROBE_KIND_RENDER, n=0, arm=NA, lens=fitted, ts=ts, salt=salt))
        return out
    if fitted and declared:
        # the DECLARED cell's run: the bites probe thinking-ON on both legs with the declaration, the two
        # render probes (the ON one declared), and NO lens-ON bite (predicate (C) is the undeclared cell's) — the
        # inventory the four gsm8k-paired200 declared cells record at 8f5f1495
        for target in (vm, C.twin_vmodel()):
            for i in range(S.N_BITES_PROMPTS):
                out.append(_declared_probe_attempt(target, kind=C.PROBE_KIND_BITES, n=i, arm=C.THINKING_ON, prompt=f"probe prompt {i}",
                                                   max_tokens=S.BITES_PROBE_MAX_TOKENS, ts=ts, salt=salt))
        for i, arm in enumerate((C.THINKING_OFF, C.THINKING_ON)):
            out.append(_declared_probe_attempt(vm, kind=C.PROBE_KIND_RENDER, n=i, arm=arm, prompt="Reply with one word.",
                                               max_tokens=S.RENDER_PROBE_MAX_TOKENS, ts=ts, salt=salt))
        return out
    if fitted:
        for target in (vm, twin):
            for i in range(S.N_BITES_PROMPTS):
                out.append(S.probe_attempt(target, kind=C.PROBE_KIND_BITES, n=i, arm=C.THINKING_OFF, lens=True,
                                           prompt=f"probe prompt {i}", max_tokens=S.BITES_PROBE_MAX_TOKENS, ts=ts, salt=salt))
    for i, arm in enumerate((C.THINKING_OFF, C.THINKING_ON)):
        out.append(S.probe_attempt(vm, kind=C.PROBE_KIND_RENDER, n=i, arm=arm, lens=fitted, ts=ts, salt=salt))
    if fitted:
        for target in (vm, C.cell_spellings(CELL_NONE, declared=False).vmodel):
            out.append(S.probe_attempt(target, kind=C.PROBE_KIND_LENS_ON_BITE, n=0, arm=C.THINKING_ON, lens=True,
                                       prompt="In one short paragraph, why is the sky blue?", max_tokens=S.LENS_ON_BITE_MAX_TOKENS,
                                       ts=ts, salt=salt))
    return out


def task_body(row: Mapping[str, Any], srow: C.SamplingRow, *, item_id: str, seed: int, cap: int, thinking_on: bool,
              declared: bool, arm: str | None = None) -> dict[str, Any]:
    fitted = bool(row["fitted"])
    arm = _arm(thinking_on, arm)
    body: dict[str, Any] = {"model": C.cell_spellings(str(row["cell"]), declared=declared).vmodel,
                            "messages": [{"role": "user", "content": QUESTIONS[IDS.index(item_id)]}],
                            "max_tokens": cap, "temperature": float(srow.temperature), "seed": seed}
    if srow.top_p is not None:
        body["top_p"] = float(srow.top_p)
    if srow.top_k is not None:
        body["top_k"] = int(srow.top_k)
    for f in C.SAMPLING_OPTIONAL_ROW_FIELDS:
        if getattr(srow, f) is not None:
            body[f] = float(getattr(srow, f))
    body.update(C.render_fragment(lens=fitted, arm=arm, header_located=declared))
    return body


def task_attempt(row: Mapping[str, Any], srow: C.SamplingRow, item: Mapping[str, Any], *, cap: int, thinking_on: bool,
                 declared: bool, ts: str, salt: str, rejected: bool = False, arm: str | None = None) -> S.Attempt:
    sample = C.sample_header_value(item["item_id"], item["sample_idx"])
    a = S.attempt(task_body(row, srow, item_id=str(item["item_id"]), seed=int(item["seed"]), cap=cap, thinking_on=thinking_on,
                            declared=declared, arm=arm),
                  task=TASK, sample=sample, ts=ts, request_id=S._request_id(salt, ts, TASK, sample),
                  prompt_tokens=int(item["prompt_tokens"]), completion_tokens=int(item["completion_tokens"]),
                  finish_reason="length" if item["at_cap"] else "stop")
    if rejected:
        a.row.update(rejected=True, status=500, error_class="server_error", finish_reason=None, usage_completion_tokens=0)
    return a


def build_store(gen_root: Path, row: Mapping[str, Any], *, correct: Iterable[int], unscored: Iterable[int] = (),
                not_at_cap: Iterable[int] = (), stray: Iterable[int] = (), capped_scored: Iterable[int] = (),
                no_think: Iterable[int] = (), loops: Iterable[int] = (), cap: int = CAP_BASE, ids: Sequence[str] | None = None,
                rung_source: Mapping[str, Any] | None = None, subset_sha16: str | None = None,
                subset_doc: Mapping[str, Any] | None = None, subset_path: Path | str | None = None,
                source_out_root: Path | str | None = None,
                sha: str = SHA7, tree_sha: str = FULL_SHA, outcomes_tree_sha: str | None = None,
                thinking_on: bool = True, sampling_mode: str = MODE_PRIMARY,
                attempts: Sequence[AttemptSpec] | None = None, write_rows: bool = True, ledger: bool = True,
                request_evidence: bool = True, evidence: bool = True, ladder: bool = True, transport_ok: bool = True,
                summary_edit: Callable[[dict[str, Any]], None] | None = None,
                header_edit: Callable[[dict[str, Any]], None] | None = None,
                manifest_edit: Callable[[dict[str, Any]], None] | None = None,
                evidence_edit: Callable[[dict[str, Any]], None] | None = None,
                ledger_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                request_evidence_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                rows_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                ladder_edit: Callable[[dict[str, Any]], None] | None = None,
                completion_tokens: Mapping[int, int] | None = None,
                texts: Mapping[int, tuple[str | None, str | None]] | None = None,
                out_root: Path | None = None, standup_record: bool = True,
                standup_record_edit: Callable[[dict[str, Any]], None] | None = None,
                producer_records_served_positions: bool = False,
                arm: str | None = None, stray_close: Iterable[int] = (), lifted: Iterable[int] = (),
                bank_view: str | None = None,
                producer_fields: bool = False,
                chat_template_hash: str | None = None, launch_record: bool = True,
                launch_edit: Callable[[dict[str, Any]], None] | None = None, log_requests: bool = False,
                serving_engine: str | None = None) -> Path:
    """One leg's store under ``<gen_root>/<row.out_root>`` (or ``rung_out_root``
    when ``cap`` is the rung's). Item INDICES into ``ids`` (default the 200
    canonical ids; a rung store passes its subset): ``correct``; ``unscored``
    (case (b): the cap inside the think block); ``capped_scored`` (case (c): at
    the cap after the block, scored); ``loops`` (at-cap draws whose text is
    degenerate under Evals' rule — the ladder's non-terminating class);
    ``no_think`` (GLM: no tag emitted); ``stray``. ``attempts``:
    the recorded runs, in order — the ACCEPTED generation attempt issues every
    item; a failed generation attempt issues its first ``n_items`` (and the
    ``rejected_items`` among them are rejected rows); a finalisation attempt
    sends probes only. Every ledger row and request-evidence record carries
    its ``attempt_id``; the manifest names the accepted one and each entry's
    kind. ``ladder`` files THIS instrument's ``cap_ladder.json`` on a base
    store THROUGH THE ADAPTER (``inspect_rung.classify_store`` over the
    fixture's draw texts; the exit follows from the at-cap share, the loops and
    the recorded proxy upstream timeout — ``transport_ok`` False records the
    600 s timeout that makes the transport check refuse the rung) at the
    planner's location. ``rung_source`` (the SOURCE store's manifest) makes
    this a rung store: its ``cap_rung`` identity and ``item_subset`` block are
    the contract's derivations from ``subset_doc`` (the adapter's validated
    subset; default a subset over ``ids`` bound to the source), with
    ``subset_sha16`` overriding the hashed digest for drift tests. On a DECLARED
    row the manifest's three served-positions fields are the producer's NULL
    (the 8f5f1495 record gap) unless ``producer_records_served_positions``
    simulates the fixed producer; ``standup_record`` writes the job's standup
    record under ``<gen_root>/serving/<jobdir>/gsm8k/`` (the pull's layout) for
    the analysis to resolve the fact from, ``standup_record_edit`` mutating its
    documents (keyed by file name) for the refusal tests.

    THE NO-SWITCH ARM (llama-paper-rows): ``arm`` (default the row's
    ``thinking_mode``, else the ``thinking_on`` boolean) — under
    ``not_available`` the store is a Llama store written through the REAL
    no-switch producers (module docstring): no reasoning channel on any draw,
    the content IS the answer, the ``none`` split's stray flags DERIVED from the
    content (``stray`` a literal ``<think>``, ``stray_close`` a literal
    ``</think>``), ``lifted`` the draws whose provider record says the client-side
    lift fired (P21), every task / probe body free of ``chat_template_kwargs``,
    the ``card`` sampling row (``sampling_mode`` defaults to the row's), the
    class's evidence ((A″) / (B′) / (C″) / bites under the single render), the
    four row-provenance fields when ``bank_view`` is given.

    ``producer_fields``
    (the eight-arm extension; default off, so every existing output is
    unchanged) writes the route / serving / SI fields the producer files and the
    per-row ARM CONTRACT reads — ``route.serving_identity`` by the cell's route
    class, ``provenance.si_id``, ``provenance.serving.{chat_template_hash, ...}``,
    ``resolved_config.system_sha`` (the prompted SI digest), ``engine`` /
    ``serving_path``; ``chat_template_hash`` overrides the family's digest.

    ``serving_engine`` (native-positions-resolve): ``contracts.SERVING_ENGINE_LORA_NATIVE``
    writes a fitted cell's store as the NATIVE driver files it — the ``+lora+``
    route with the engine stamp (:func:`nativize_manifest`), the engine's own
    stamp triple, the bites control against the engine's zero cell, predicate
    (B′)'s explicit inapplicability record, and the NATIVE standup record
    beside the job (:func:`native_standup_record_docs`: the routes[]
    attestation ``all``; the engine-neutral ``served_positions`` map too when
    ``producer_records_served_positions``). The manifest's transfer-provenance
    block carries the producer's NULL (the native record gap) unless
    ``producer_records_served_positions`` simulates the fixed producer's
    ``["all"]``. A declared native row is a matrix defect and refuses; None
    (the default) is the lens engine, byte-identical to before."""
    key, cell, view = str(row["model_key"]), str(row["cell"]), str(row["view"])
    fitted, declared = bool(row["fitted"]), bool(row["declared"])
    native = serving_engine == C.SERVING_ENGINE_LORA_NATIVE
    if serving_engine is not None and not native:
        raise ValueError(f"serving_engine {serving_engine!r} is not None or {C.SERVING_ENGINE_LORA_NATIVE!r}")
    if native and declared:
        raise ValueError("a DECLARED natively served row is a matrix defect (the native engine serves undeclared all-position cells only)")
    native = native and fitted                     # an unsteered leg reads the same on both engines here
    arm = _arm(thinking_on, arm if arm is not None else row.get("thinking_mode"))
    thinking_on = arm == C.THINKING_ON
    no_switch = arm == NA
    if no_switch and sampling_mode == MODE_PRIMARY:
        sampling_mode = str(row.get("sampling_mode") or MODE_CARD)
    family = FAMILY_BY_KEY[key]
    fact = RENDER_FACTS_ALL[family]
    srow = _sampling_row(sampling_mode, key)
    ids = list(ids if ids is not None else IDS)
    correct, unscored, not_at_cap = set(correct), set(unscored), set(not_at_cap)
    stray, no_think, loops = set(stray), set(no_think), set(loops)
    stray_close, lifted = set(stray_close), set(lifted)
    capped_scored = set(capped_scored) - unscored
    length_of = dict(completion_tokens or {})
    is_rung = cap == CAP_RUNG and rung_source is not None
    root = Path(out_root) if out_root is not None else Path(gen_root) / str(row["rung_out_root"] if is_rung else row["out_root"])
    spell = C.cell_spellings(cell, declared=declared)
    vmodel = spell.vmodel
    hf, rev = S.model_pin(str(row["tag"]))
    ihash = _hash(f"{key}|{view}|{cell}|ihash", 8)
    chash = _hash(f"{key}|{view}|{cell}|{sampling_mode}|{cap}|{arm if no_switch else thinking_on}|chash", 8)
    cell_key = f"{cell_slug(cell)}__controls__t0__a0__{ihash}__{chash}"
    cell_dir = root / "cells" / cell_key
    cell_dir.mkdir(parents=True, exist_ok=True)
    record_only = fact["split_mode"] == C.THINK_SPLIT_RECORD_ONLY or not thinking_on
    specs = list(attempts) if attempts is not None else default_attempts()
    accepted = [a for a in specs if a.accepted]
    assert len(accepted) == 1, "exactly one accepted generation attempt"
    accepted_id = accepted[0].attempt_id

    block = sampling_block(sampling_mode, key)
    sampling_ref = C.sampling_ref(block)
    results: list[isv.ItemResult] = []
    channels = dict(texts or {})
    for j, item_id in enumerate(ids):
        pos = IDS.index(item_id)
        is_unscored = j in unscored
        at_cap = (is_unscored and j not in not_at_cap) or j in capped_scored
        value = float("nan") if is_unscored else (1.0 if j in correct else 0.0)
        answer_text = f"The answer is {ANSWERS[pos]}." if not is_unscored else ""
        if no_switch:
            # NO reasoning channel: the content IS the answer (an unscored draw here is an at-cap draw whose text carries no
            # parseable answer, never a think block); a stray is a literal tag IN the content, the flags DERIVED from it by
            # the real `none` split (think_split.split_think), never typed
            text = ("step again " * 400).strip() if j in loops else (answer_text if not is_unscored else f"working on item {j} without an answer")
            if j in stray:
                text = f"{TS.THINK_OPEN}{text}"
            if j in stray_close:
                text = f"{text}{TS.THINK_CLOSE}"
            content, reasoning = channels.get(j, (text, None))
            split = TS.split_think(content or "", mode=C.THINK_SPLIT_NONE)
            assert split.present is False and split.unterminated is False
            lift_fired = j in lifted
            results.append(isv.ItemResult(
                item_id=item_id, value=value, sample_idx=0, seed=isv.sample_seed(item_id, 0), answer=None if is_unscored else ANSWERS[pos],
                prompt_tokens=PROMPT_BASE + (pos % 320),
                completion_tokens=cap if at_cap else length_of.get(j, 200 + (pos * 7) % 300),
                paired_id=PAIRED_BY_INSPECT_ID[item_id], scores={SCORER_NAME: value},
                content=content, reasoning=reasoning, content_empty_with_reasoning=False,
                stop_reason=isv.STOP_REASON_AT_CAP if at_cap else "stop",
                unparseable=None if is_unscored else False, attempt=1, max_tokens_used=cap,
                think_block_present=False, think_block_unterminated=False, stray_open_tag=split.stray_open_tag,
                stray_close_tag=split.stray_close_tag, client_lift_fired=lift_fired,
                lifted_reasoning=(f"quoted reasoning lifted on item {j}" if lift_fired else None),
                raw_content_sha256=hashlib.sha256((content or "").encode("utf-8")).hexdigest()))
            continue
        if j in loops:
            loop_text = ("step again " * 400).strip()
            default_channels = ("" if (record_only and is_unscored) else loop_text, loop_text)
        else:
            default_channels = ("" if (record_only and is_unscored) else answer_text, f"reasoning about item {j}" if thinking_on else None)
        content, reasoning = channels.get(j, default_channels)
        results.append(isv.ItemResult(
            item_id=item_id, value=value, sample_idx=0, seed=isv.sample_seed(item_id, 0), answer=None if is_unscored else ANSWERS[pos],
            prompt_tokens=PROMPT_BASE + (pos % 320),
            completion_tokens=cap if at_cap else length_of.get(j, 900 + (pos * 7) % 700),
            paired_id=PAIRED_BY_INSPECT_ID[item_id], scores={SCORER_NAME: value},
            content=content, reasoning=reasoning,
            content_empty_with_reasoning=bool(reasoning) and not content,
            stop_reason=isv.STOP_REASON_AT_CAP if at_cap else "stop",
            unparseable=None if is_unscored else False, attempt=1, max_tokens_used=cap,
            think_block_present=thinking_on and j not in no_think, think_block_unterminated=is_unscored, stray_open_tag=j in stray))
    plan = task_plan(sampling_mode, key, ids, fact, cap=cap, thinking_on=thinking_on, arm=arm)
    counts = isv.attempted_counts(results)
    split_counts = isv.think_split_counts(results, plan=plan)
    n_scored = counts["n_scored"]
    n_correct = sum(1 for r in results if not r.unscored and r.value == 1.0)
    stamps = dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, stamps_for(fitted, declared, thinking_on=thinking_on, arm=arm, native=native)))
    summary: dict[str, Any] = {
        C.METRIC_ACC: n_correct / n_scored if n_scored else None, C.METRIC_N: counts["n"], "n_requests": len(ids),
        "n_rejected": 0, "n_http_rejected_retried": 0, "samples_per_item": 1, "task": TASK, "item_set": C.GSM8K_ITEM_SET_PAIRED,
        "n_paired": len(ids), "framework": C.INSPECT_TASK_VERSION, "engine": "vllm", "route": vmodel,
        "thinking": arm, "render": C.THINKING_RENDER_NATIVE, "wall_clock_s": 900.0,
        "logs_reused": 0, "reasoning_parser": fact["server_parser"] if thinking_on else None,
        "n_content_empty_with_reasoning": sum(1 for r in results if r.content_empty_with_reasoning),
        "n_scored": n_scored, "n_unscored": counts["n_unscored"],
        "n_scored_samples": counts["n_scored_samples"], "n_unscored_samples": counts["n_unscored_samples"],
        **isv.task_quality(results), **split_counts,
        "max_tokens": cap, "temperature": srow.temperature, "top_p": srow.top_p, "top_k": srow.top_k, "grader_max_tokens": None,
    }
    for f in C.EXPECTED_STAMP_TRIPLE_FIELDS:
        summary[f] = stamps[f]
        summary[C.STAMP_REGIME_COUNT_FIELD[f]] = 1
    if summary_edit:
        summary_edit(summary)
    header = isv.outcome_record(results, plan=plan, task=TASK, thinking=summary["thinking"], summary=summary,
                                split_counts={k: summary[k] for k in split_counts},
                                producer_tree_sha=tree_sha if outcomes_tree_sha is None else outcomes_tree_sha,
                                sampling_ref=sampling_ref)
    if header_edit:
        header_edit(header)
    items = header["items"]

    max_model_len = MAX_MODEL_LEN_RUNG if cap == CAP_RUNG else (4608 if (not thinking_on and not no_switch) else MAX_MODEL_LEN_BASE)
    # the served concurrency is the model budget's S at the served window, the client width equals it, and the geometry block
    # is the standup's resolution in the producer's shape (`S.served_seqs` / `S.geometry_block`, the MATH-500 store's)
    served_s = S.served_seqs(key, max_model_len)
    geometry_eval = GEOMETRY_EVAL_RUNG if cap == CAP_RUNG else (GEOMETRY_EVAL_THINKING if thinking_on else GEOMETRY_EVAL_OFF)
    served = {"reasoning_parser": fact["server_parser"] if thinking_on else None, "tool_calls": False, "max_model_len": max_model_len,
              "max_num_seqs": served_s, "concurrency": served_s, C.AGENT_LOOP_ENTRY_KEY: True, C.REDUCTION_MODE_FIELD: "stacked"}
    # `arm` was resolved above (`_arm(thinking_on, ...)`): on / off / the no-switch `not_available`
    tinycap: dict[str, Any] = {
        "tasks": [TASK], "pass": None, "samples": 1, "served": served,
        # the render label (the driver's _render_label): flag_only under ON; native under OFF and the no-switch arm (I11)
        "thinking": {"mode": arm, "by_task": {TASK: arm}, "render": (C.THINKING_RENDER_FLAG_ONLY if thinking_on else C.THINKING_RENDER_NATIVE)},
        C.IDENTITY_KEY_THINKING_BY_TASK: {TASK: arm},
        "versions": {"tinycap": C.TINYCAP_VERSION}, "limits": {}, "frameworks": {"inspect_ai": "0.3.260"},
        C.IDENTITY_KEY_IDENTITY_VERSION: C.TINYCAP_IDENTITY_VERSION_2,
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {TASK: cap},
        C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS: 0, C.IDENTITY_KEY_MATH500_MAX_TOKENS: C.MATH500_MAX_TOKENS_DEFAULT,
        **{k: block[k] for k in C.SAMPLING_IDENTITY_FIELDS},
        **{f: block[f] for f in C.SAMPLING_OPTIONAL_IDENTITY_FIELDS if f in block},
    }
    if not no_switch:
        # "present iff any task is ON" as the fixture has always written it (the OFF overlap leg included); a no-switch
        # store hashes NO header ids (the one render is certified live, predicate (A″))
        tinycap[C.IDENTITY_KEY_THINKING_HEADER_IDS] = {"off": list(fact["off_ids"]), "on": list(fact["on_ids"])}
    if fitted and thinking_on:
        tinycap[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] = True
    if declared:
        tinycap[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] = list(C.OOS_CELL_DECLARATION_EXPECTED)
    item_subset_block: dict[str, Any] | None = None
    if is_rung:
        assert rung_source is not None
        doc = dict(subset_doc) if subset_doc is not None else synthetic_subset_doc(
            ids, source_manifest=rung_source, source_out_root=source_out_root)
        C.validate_item_subset(doc, where="synthetic rung subset")
        cap_rung = C.cap_rung_identity(doc)
        if subset_sha16 is not None:
            cap_rung["subset_sha16"] = subset_sha16
        tinycap[C.IDENTITY_KEY_CAP_RUNG] = cap_rung
        item_subset_block = C.item_subset_block(doc, path=str(subset_path if subset_path is not None
                                                              else Path(gen_root) / f"rung16384/{row['tag']}/{row['cell_slug']}/rung_subset.json"))
    pin = str(row["object_sha256"]) if fitted else None
    # the driver's served-positions lookup as the producer at 8f5f1495 records it: the value under the bare key — the
    # undeclared cell's ["all"]; NULL on a declared cell (the standup keys the map by `<harness>@thinking`; the bare-key
    # read misses; served-positions-null-diagnosis.md §1) unless the fixed producer (§4.3) is being simulated
    served_positions = ([str(row["served_positions_expected"])] if fitted and (not declared or producer_records_served_positions) else None)
    # the route block's `lens_serve_declarations` is the standup's RAW map copied whole (`_attestation_route_block`:
    # `{<key>: {"serve_out_of_support": [...]}}`, the scale-0 twin inheriting its donor's declaration) — the fixture used
    # to write the bare list under the cell's key, a shape the producer never emits
    decl_map = ({C.TWIN_CELL_ID: {C.SERVE_OUT_OF_SUPPORT_KEY: list(C.OOS_CELL_DECLARATION_EXPECTED)},
                 spell.harness: {C.SERVE_OUT_OF_SUPPORT_KEY: list(C.OOS_CELL_DECLARATION_EXPECTED)}} if declared else {})
    routing_table_path = cluster_routing_table_path(row, cap)
    resume: list[dict[str, Any]] = []
    for k, a in enumerate(specs):
        # the attempt's window (the contract's `started` / `ended`): run k generates in [k:00, k:30] on the fixture's clock
        resume.append({"ts": f"2026-01-01T{k:02d}:00:00Z", "tree_sha": tree_sha, "attempt": k + 1, "completed": bool(a.completed),
                       "attempt_id": a.attempt_id, "kind": a.kind, "rc": 0 if a.completed else 1, "terminal": True,
                       "started": f"2026-01-01T{k:02d}:00:00Z", "ended": f"2026-01-01T{k:02d}:30:00Z"})
    manifest: dict[str, Any] = {
        "cell_id": cell, "harness_cell_id": spell.harness, "virtual_model": vmodel,
        "intervention_hash": ihash + "0" * 8, "harness_config_hash": chash + "0" * 8,
        "model_id": hf, "model_revision": rev,
        C.MANIFEST_KEY_PRODUCER: dict(zip(C.PRODUCER_BLOCK_FIELDS, (tree_sha, C.PRODUCER_DRIVER_SERVED))),
        C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION,
        C.MANIFEST_KEY_PROVENANCE: {"cell_id": cell, "class": "A" if fitted else "baseline", "artifact_sha256": pin,
                                    "intervention_hash": ihash + "0" * 8, "model": hf, "model_revision": rev},
        C.MANIFEST_KEY_TINYCAP: tinycap, "resolved_config": {"tinycap": tinycap},
        C.MANIFEST_KEY_ROUTE: {"virtual_model": vmodel, "grader_virtual_model": None, "tool_calls": False, C.AGENT_LOOP_ENTRY_KEY: True,
                               # the producer's spelling (verified against the published stores; apply_producer_fields
                               # below re-derives it): passthrough on a bare / prompted leg, the lens route on a fitted
                               # cell — a bare leg wearing the steered marker would read as a steered path
                               # (`steered_path`) that the producer never files
                               "serving_identity": S.serving_identity_of(fitted),
                               C.REDUCTION_MODE_FIELD: "stacked", C.ROUTE_KEY_SERVE_DECLARATIONS: decl_map,
                               C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: pin,
                               C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {spell.harness: pin} if fitted else {},
                               C.ENDPOINT_KEY_LENS_SALT_POLICY: C.LENS_SALT_POLICY_ATTESTED,
                               C.ROUTE_KEY_RESUMED_PREFILL_POLICY: stamps["resumed_policy"],
                               C.LENS_SERVED_POSITIONS_KEY: served_positions,
                               **({"bank_view": bank_view} if bank_view is not None else {})},
        # `enable_thinking` NULL under the no-switch arm (P17: no switch was SENT — never `false`)
        C.MANIFEST_KEY_THINKING: {"enable_thinking": (None if no_switch else thinking_on), "render": C.THINKING_RENDER_NATIVE,
                                  "reasoning_parser": fact["server_parser"] if thinking_on else None, "mode": arm, "by_task": {TASK: arm}},
        C.MANIFEST_KEY_GEOMETRY: S.geometry_block(key, eval_key=geometry_eval, max_model_len=max_model_len),
        C.MANIFEST_KEY_CONCURRENCY: served_s, C.MANIFEST_KEY_SAMPLING: block,
        C.MANIFEST_KEY_USAGE_LEDGER: str(root / C.LEDGER_FILENAME) if ledger else None, "ledger_url": LEDGER_URL,
        C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: UPSTREAM_TIMEOUT_S if transport_ok else UPSTREAM_TIMEOUT_REFUSING_S,
        "routing_table": routing_table_path, "proxy_url": PROXY_URL,
        "binding": {"proxy_url": PROXY_URL, "routing_table": routing_table_path, "ledger_url": LEDGER_URL,
                    "ledger_upstream": PROXY_URL, "ledger_endpoint": {"ledger_url": LEDGER_URL, "upstream": PROXY_URL}},
        # the driver copies the family fact whole (`render_fact.as_dict()`): `thinking_switch` rides on every store
        C.MANIFEST_KEY_RENDER_FACT_EVIDENCE: {k: fact.get(k) for k in ("family", "off_ids", "on_ids", "split_mode", "server_parser",
                                                                       "evidence_status", "thinking_switch")},
        C.MANIFEST_KEY_AUTHORISED_EXPERIMENT: tuple_name_of(key, declared=declared, cap=cap) if fitted and thinking_on else None,
        "support": C.support_label(fitted=fitted, out_of_support=fitted and thinking_on),
        "header_located": declared if fitted else None, C.LENS_SERVED_POSITIONS_KEY: served_positions,
        C.MANIFEST_KEY_RESUME_ATTEMPTS: resume, MANIFEST_KEY_ACCEPTED_ATTEMPT: accepted_id,
        C.MANIFEST_KEY_REQUEST_EVIDENCE: str(root / C.REQUEST_EVIDENCE_FILENAME), "ledger_thinking_refusals": {}, "host": "TESTHOST",
    }
    if fitted and (thinking_on or no_switch):
        manifest[C.MANIFEST_KEY_TRANSFER_PROVENANCE] = transfer_provenance_record(row, fact, served_positions=served_positions, arm=arm)
    if native:
        # the NATIVE driver's manifest: the route / engine facts, the engine's `header_located: false`, null served-
        # positions maps, and the transfer-provenance block with what the driver's lookup returned — the null (the
        # native record gap) unless the fixed producer is being simulated
        nativize_manifest(manifest, row, fact, arm=arm,
                          served_positions=([str(row["served_positions_expected"])] if producer_records_served_positions else None))
    if no_switch:
        # the setter map beside the hashed executed caps (the cap rides `--on-cap-by-task` under this arm; UNHASHED)
        manifest[C.MANIFEST_KEY_ON_CAP_BY_TASK] = {TASK_NAME: cap}
    if item_subset_block is not None:
        manifest[C.MANIFEST_KEY_ITEM_SUBSET] = item_subset_block
    if producer_fields:
        apply_producer_fields(manifest, row, chat_template_hash=chat_template_hash, native=native)
    if manifest_edit:
        manifest_edit(manifest)
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    # the driver's gen_stats.json: the task's summary under its row task (the real store's shape)
    (cell_dir / "gen_stats.json").write_text(json.dumps({TASK: summary}, indent=2, default=str), encoding="utf-8")
    # every draw's two channels, for the .eval writer (an UNSCORED draw has no row, but the real log carries its text)
    (cell_dir / "private").mkdir(exist_ok=True)
    (cell_dir / "private" / CHANNELS_NAME).write_text(json.dumps({res.item_id: [res.content, res.reasoning] for res in results}), encoding="utf-8")
    if no_switch:
        # the provider's no-switch response-conversion record per draw (P21), for the .eval writer (the real log carries it
        # as ModelOutput.metadata[NO_SWITCH_METADATA_KEY]); never read by the analysis from here
        (cell_dir / "private" / LIFT_NAME).write_text(json.dumps({res.item_id: {f: getattr(res, f) for f in C.ROW_LIFT_FIELDS}
                                                                  for res in results}), encoding="utf-8")
    # the four row-provenance fields, PRESENT iff `bank_view` (the driver's `--bank-view`), through the producer's own builder
    provenance = C.row_provenance(cell=cell, view=bank_view, object_artifact_sha256=(pin if fitted else None), producer_tree_sha=tree_sha)

    if write_rows:
        rows_out = []
        by_item = {res.item_id: res for res in results}
        for r in items:
            if not r["scored"]:
                continue
            res = by_item[r["item_id"]]
            content, reasoning = res.content, res.reasoning
            extra = {"sample_idx": 0, "seed": r["seed"], "answer": ANSWERS[IDS.index(r["item_id"])], "prompt_tokens": r["prompt_tokens"],
                     "completion_tokens": r["completion_tokens"], "paired_id": PAIRED_BY_INSPECT_ID[r["item_id"]],
                     "scores": {SCORER_NAME: r["value"]}, "stop_reason": r["stop_reason"], "unparseable": False, "unscored": False,
                     "attempt": 1, "max_tokens": cap, "max_tokens_used": cap, "content": content, "reasoning": reasoning,
                     "content_empty_with_reasoning": bool(reasoning) and not content, "thinking": arm,
                     "think_block_present": r["think_block_present"], "think_block_unterminated": False, "stray_open_tag": r["stray_open_tag"],
                     C.SAMPLING_REF_FIELD: sampling_ref}
            if no_switch:
                # the no-switch facts ride the row ONLY when recorded (ItemResult.extra: the `none` split's close-tag stray and the
                # provider's lift record) — a switched family's row is byte-identical
                extra["stray_close_tag"] = bool(res.stray_close_tag)
                extra.update({f: getattr(res, f) for f in C.ROW_LIFT_FIELDS})
            extra.update(provenance)
            rows_out.append({"task": TASK, "item_id": r["item_id"], "metric": C.METRIC_ACC, "value": r["value"], "engine": "vllm",
                             "kind": "cell", "domain": "inspect_gen", "extra_json": json.dumps(extra),
                             "harness_config_hash": chash + "0" * 8, "intervention_hash": ihash + "0" * 8})
        if rows_edit:
            rows_edit(rows_out)
        lines = [json.dumps(r) for r in rows_out]
        lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_ACC_AGGREGATE, "value": summary[C.METRIC_ACC],
                                 "engine": "vllm", "kind": "cell", "extra_json": "{}"}))
        lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_N, "value": float(len(ids)), "engine": "vllm",
                                 "kind": "cell", "extra_json": None}))
        (cell_dir / "rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    completion.finish_task(cell_dir, task=TASK, n_items=len(ids), n_expected_full=len(ids), limits=None, n_rows=n_scored,
                           n_requests=len(ids), n_rejected=0, wall_clock_s=900.0, summary=summary, outcomes=header)

    # every attempt's calls, in the driver's order per run: probes, then the task calls (a failed generation attempt's
    # first n_items, its rejected ones as rejected rows; a finalisation attempt probes only); each carries its attempt id
    attempts_out: list[tuple[str, S.Attempt]] = []
    for k, a in enumerate(specs):
        ts = f"2026-01-01T{k:02d}:00:00Z"
        for pa in probe_attempts(row, fitted=fitted, declared=declared, ts=ts, salt=cell_key + a.attempt_id, arm=arm, native=native):
            attempts_out.append((a.attempt_id, pa))
        if a.kind != "generation":
            continue
        issued = items if a.accepted else items[: (a.n_items if a.n_items is not None else len(items))]
        rejected = set(a.rejected_items)
        for j, it in enumerate(issued):
            attempts_out.append((a.attempt_id, task_attempt(row, srow, it, cap=cap, thinking_on=thinking_on, declared=declared, ts=ts,
                                                            salt=cell_key + a.attempt_id, rejected=j in rejected, arm=arm)))
    recs = [_with_attempt(a.record, aid) for aid, a in attempts_out]
    lrows = [_with_attempt(a.row, aid) for aid, a in attempts_out]
    if request_evidence:
        if request_evidence_edit:
            request_evidence_edit(recs)
        (root / C.REQUEST_EVIDENCE_FILENAME).write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n", encoding="utf-8")
    if ledger:
        if ledger_edit:
            ledger_edit(lrows)
        (root / C.LEDGER_FILENAME).write_text("\n".join(json.dumps(r) for r in lrows) + "\n", encoding="utf-8")
    if evidence:
        if no_switch:
            # THE NO-SWITCH CERTIFICATION CLASS (P18) through the real producers: (A″) render + the once-measured header + the
            # one probe; on a fitted cell (B′) verbatim, (C″) the EXPLICIT inapplicability record, the bites control under the
            # single render
            bites_block = (steering_bites_record_no_switch(cell) if fitted
                           else {C.OOS_EVIDENCE_BITES_APPLICABLE: False, "reason": "no steering selector on this route"})
            render_block, probe_block = render_certified_record_no_switch(family, fact), render_probe_record_no_switch()
        else:
            bites_block = ((steering_bites_record_declared(cell) if declared else S.steering_bites_record(cell)) if fitted
                           else {C.OOS_EVIDENCE_BITES_APPLICABLE: False, "reason": "no steering selector on this route"})
            render_block, probe_block = S.render_certified_record(family, fact), S.render_probe_record(fact)
        ev: dict[str, Any] = {
            "driver": "served_capability_run_v2", "cell_id": cell, "harness_cell_id": spell.harness, "virtual_model": vmodel,
            "engine": "vllm", "mask": "generated" if fitted else None,
            C.OOS_EVIDENCE_BITES_BLOCK: bites_block,
            S.EVIDENCE_RENDER_CERTIFIED: render_block, S.EVIDENCE_RENDER_PROBE: probe_block,
            "ledger_thinking_refusals": {}, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(0)),
        }
        if no_switch:
            ev[EVIDENCE_THINKING_HEADER_BY_ARM] = no_switch_header_record(fact, lens=fitted and not native)
            if fitted:
                ev[S.EVIDENCE_LENS_POSITION] = lens_position_record(row)
                ev[S.EVIDENCE_LENS_ON_BITE] = C.no_switch_lens_on_bite_record()
        elif fitted and declared:
            # the DECLARED header-located class: predicate (B)'s record, and NO (B′)/(C) block (the driver runs those
            # only where `not header_located`)
            ev[EVIDENCE_THINKING_HEADER_BY_ARM] = thinking_header_by_arm_record(fact)
        elif fitted:
            ev[S.EVIDENCE_LENS_POSITION] = lens_position_record(row)
            ev[S.EVIDENCE_LENS_ON_BITE] = lens_on_bite_record(row, fact)
        if native:
            # the NATIVE driver's evidence: predicate (B′)'s EXPLICIT inapplicability record (the driver's own
            # `_native_lens_position_record`), the bites control against the ENGINE's zero cell under the non-lens probe
            # body with the proxy's `not-consulted` policy on every reply (the (C) / (C″) record is engine-neutral)
            ev[S.EVIDENCE_LENS_POSITION] = native_lens_position_record()
            ev[C.OOS_EVIDENCE_BITES_BLOCK] = native_bites_record(ev[C.OOS_EVIDENCE_BITES_BLOCK], arm=(NA if no_switch else C.THINKING_OFF))
        if evidence_edit:
            evidence_edit(ev)
        blob = json.dumps(ev, indent=1)
        (cell_dir / "evidence.json").write_text(blob, encoding="utf-8")
        (root / "evidence.json").write_text(blob, encoding="utf-8")
    (root / "endpoint.json").write_text(json.dumps({"upstream_timeout_s": UPSTREAM_TIMEOUT_S if transport_ok else UPSTREAM_TIMEOUT_REFUSING_S,
                                                    "max_model_len": max_model_len}), encoding="utf-8")
    if fitted and declared and standup_record:
        # the standup record the analysis resolves a DECLARED cell's served positions from, pulled beside the stores
        docs = standup_record_docs(row, fact, cap=cap, tree_sha=tree_sha, intervention_hash=manifest["intervention_hash"], max_model_len=max_model_len)
        if standup_record_edit:
            standup_record_edit(docs)
            seal_standup_record(docs)            # an edited registry is still a consistent record: refused on its field, not on the seal
        write_standup_record(Path(gen_root), standup_jobdir(row, cap), docs)
    if native and standup_record:
        # the NATIVE standup record beside the job (the real record's shape): the routes[] attestation of the served
        # cell's mode; the engine-neutral map too once the fixed standup is being simulated
        expect = str(row["served_positions_expected"])
        docs = native_standup_record_docs(row, fact, cap=cap, tree_sha=tree_sha, intervention_hash=manifest["intervention_hash"],
                                          max_model_len=max_model_len, position_mode=expect,
                                          served_positions_map=({spell.harness: [expect]} if producer_records_served_positions else None))
        if standup_record_edit:
            standup_record_edit(docs)
        write_standup_record(Path(gen_root), standup_jobdir(row, cap), docs)
    if launch_record and native:
        # the NATIVE leg's LAUNCH record (launch-attestation-native): the SAME server log, carrying the lora_native engine's echo
        # of the composed launch (`enable_lora`, the LoRA config, `lora_modules` as `LoRAModulePath(...)` reprs, no
        # `enforce_eager`), bound to the job by the STRICT serving attestation's Slurm job id and port (written by the standup
        # record above; written here when `standup_record` is off) — a native standup writes NO steering_attestation.json
        job, jobdir = standup_job_id(row, cap), standup_jobdir(row, cap)
        d = Path(gen_root) / STANDUP_DIR_PARENT / jobdir / STANDUP_TASK_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        port = S.launch_port(job)
        specs = native_adapter_specs(row, intervention_hash=manifest["intervention_hash"])
        if not (d / STANDUP_SERVING_ATTESTATION).is_file():
            attestation = S.native_serving_attestation_doc(job=job, port=port, model=hf, revision=rev, argv_sha256=_hash(f"synthetic argv {jobdir}"),
                                                           nonce=_hash(f"nonce|{jobdir}", 32), adapters=specs,
                                                           max_model_len=int(max_model_len), max_num_seqs=served_s)
            (d / STANDUP_SERVING_ATTESTATION).write_bytes(standup_doc_bytes(STANDUP_SERVING_ATTESTATION, attestation))
        args = S.launch_echo_args(key, revision=rev, port=port, max_model_len=int(max_model_len), max_num_seqs=served_s,
                                  reasoning_parser=fact["server_parser"] if thinking_on else None, log_requests=log_requests, jobdir=jobdir,
                                  native=True, lora_modules=S.native_lora_modules(specs))
        if launch_edit:
            launch_edit(args)
        (d / S.LAUNCH_LOG).write_text(S.launch_log_text(args, generation_config=S.GENERATION_CONFIG_BY_KEY.get(key), log_requests=log_requests),
                                      encoding="utf-8")
    if launch_record and not native:
        # the LAUNCH record beside every job (code review C2): the server log carrying the engine's echo of the composed launch,
        # bound to the job by the attestation's Slurm job id and port (the declared record's attestation, written above, already
        # carries them; a non-declared leg gets the attestation and the jobs file here); `launch_edit` mutates the echoed args
        job, jobdir = standup_job_id(row, cap), standup_jobdir(row, cap)
        d = Path(gen_root) / STANDUP_DIR_PARENT / jobdir / STANDUP_TASK_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        port = S.launch_port(job)
        if not (d / STANDUP_ATTESTATION).is_file():
            attestation = {"slurm_job_id": job, "hostname": "TESTHOST", "host": "127.0.0.1", "port": port, "model": hf, "n_workers": 1,
                           "attested_at": 0.0, "argv_sha256": sha256_hex16(f"synthetic argv {jobdir}".encode()), "nonce": _hash(f"nonce|{jobdir}", 32),
                           "registry_index_sha16": None}
            (d / STANDUP_ATTESTATION).write_bytes(standup_doc_bytes(STANDUP_ATTESTATION, attestation))
        if not (d / STANDUP_JOBS).is_file():
            jobs = [{"job_id": job, "node": "TESTHOST", "key": key, "view": view, "mode": "attacks", "eval": geometry_eval,
                     "sha": tree_sha, "lens_reduction_mode": "stacked", "max_num_seqs": served_s, "max_model_len": int(max_model_len),
                     "cluster": "synthetic", "written_at": TS0}]
            (d / STANDUP_JOBS).write_bytes(standup_doc_bytes(STANDUP_JOBS, jobs))
        args = S.launch_echo_args(key, revision=rev, port=port, max_model_len=int(max_model_len), max_num_seqs=served_s,
                                  reasoning_parser=fact["server_parser"] if thinking_on else None, log_requests=log_requests, jobdir=jobdir)
        if launch_edit:
            launch_edit(args)
        (d / S.LAUNCH_LOG).write_text(S.launch_log_text(args, generation_config=S.GENERATION_CONFIG_BY_KEY.get(key), log_requests=log_requests),
                                      encoding="utf-8")
    # the wrapper's RESULT line beside the store (the planner's store_status reads `result=COMPLETE` off it)
    (root / RESULT_NAME).write_text(f"{TS0} job=0 name={row['job_id']} array={row.get('index', 0)} row={row.get('index', 0)} {row['job_id']} "
                                    f"g8k {key} {cell} {sampling_mode} fitted={int(fitted)} rc=0 driver_rc=0 result=COMPLETE\n", encoding="utf-8")
    if ladder and not is_rung and (thinking_on or no_switch):
        try:
            record, _subset = classify(root, threshold=row_trigger(row))
        except IR.InspectRungError as e:
            # a store the adapter refuses to classify (a defect test tore it: another hashed cap, a diagnostic identity, ...):
            # the planner would abort without filing a record, so none is filed here either; the refusal is kept beside the
            # store for the test to read (a fixture-only sidecar, never read by the analysis)
            (cell_dir / "private" / "synthetic_ladder_refused.txt").write_text(str(e), encoding="utf-8")
        else:
            if ladder_edit:
                ladder_edit(record)
            file_ladder_record(root, record)
    return cell_dir


def store_texts(cell_dir: Path) -> dict[tuple[str, int], str | None]:
    """Every draw's emitted text as the adapter reads it (``inspect_rung.emitted_text``
    over the two channels) — from the fixture's sidecar, standing in for the
    retained ``.eval`` log the adapter reads in the client venv."""
    side = json.loads((Path(cell_dir) / "private" / CHANNELS_NAME).read_text(encoding="utf-8"))
    return {(str(iid), 0): IR.emitted_text(content, reasoning) for iid, (content, reasoning) in side.items()}


def classify(root: Path, *, threshold: float | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``inspect_rung.classify_store`` — the AUTHOR of this instrument's ladder
    record — over the synthetic store under ``root`` with the fixture's texts,
    at the ROW's trigger (the Llama matrix's declared 1 %; a schema-1 row reads the adapter's, also 1 % since 2026-09-12).
    Pure: called again by :func:`build_escalated_cell` for the subset."""
    cell_dir = IR.cell_dir_of(Path(root))
    kw = {"threshold": float(threshold)} if threshold is not None else {}          # the rung Instrument's parameter (2026-09-12)
    return IR.classify_store(Path(root), texts=store_texts(cell_dir), **kw)


def row_trigger(row: Mapping[str, Any]) -> float:
    return float(row.get("trigger_threshold", TRIGGER_THRESHOLD))


def _atomic_json(path: Path, doc: Any) -> None:
    """``ladder_plan.atomic_write_json``'s shape (sort_keys, indent 1, a trailing newline, a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_ladder_record(root: Path, record: Mapping[str, Any]) -> Path:
    """The adapter's record filed where lane B's planner (its one WRITER) files
    it: ``<out_root>/cap_ladder.json``, verbatim."""
    p = Path(root) / CL.RECORD_NAME
    _atomic_json(p, dict(record))
    return p


def synthetic_subset_doc(ids: Sequence[str], *, source_manifest: Mapping[str, Any], source_out_root: Path | str | None = None,
                         n_non_terminating: int = 0, trigger_threshold: float = TRIGGER_THRESHOLD) -> dict[str, Any]:
    """A ``rung_subset/1`` body over ``ids`` bound to ``source_manifest`` in the
    contract's exact shape (for a rung store built WITHOUT the adapter's subset
    — the defect tests); ``source_out_root`` defaults to the source store's own
    root (its manifest names its ledger under it)."""
    if source_out_root is None:
        source_out_root = Path(str(source_manifest[C.MANIFEST_KEY_USAGE_LEDGER])).parent
    tv = CL.transport_check(CAP_RUNG, {CL.HOP_DRIVER: IR.DRIVER_HOP_TIMEOUT_S, CL.HOP_PROXY_UPSTREAM: UPSTREAM_TIMEOUT_S})
    return {"schema": C.ITEM_SUBSET_SCHEMA, "task": TASK, "source_out_root": str(source_out_root),
            "source_identity_hash": str(source_manifest["harness_config_hash"]), "source_cap": CAP_BASE, "cap": CAP_RUNG,
            "rule": IR.gsm8k_trigger_rule(trigger_threshold), "threshold": trigger_threshold, "ids": [str(i) for i in ids],
            "n_at_cap": len(ids) + n_non_terminating, "n_non_terminating": n_non_terminating, "n_boundary": 0,
            "transport_check": tv.as_dict()}


def rung_subset_path(gen_root: Path, row: Mapping[str, Any]) -> Path:
    return Path(gen_root) / f"rung16384/{row['tag']}/{row['cell_slug']}/rung_subset.json"


def write_rung_subset(gen_root: Path, row: Mapping[str, Any], subset: Mapping[str, Any], *,
                      edit: Callable[[dict[str, Any]], None] | None = None) -> Path:
    """The planner's ``rung_subset.json`` under the wave root:
    the ADAPTER's returned subset body, VERBATIM, in the planner's serialization
    (``edit`` applies a defect for the refusal tests, after the contract's
    validator has passed the body)."""
    doc = dict(subset)
    C.validate_item_subset(doc, where="the adapter's subset")
    if edit:
        edit(doc)
    p = rung_subset_path(gen_root, row)
    _atomic_json(p, doc)
    return p


def write_inspect_log(cell_dir: Path, *, name: str, items: Sequence[Mapping[str, Any]] | None = None, vmodel: str | None = None,
                      sample_idx: int = 0, n_samples: int | None = None, cancelled: bool = False,
                      value_edit: Callable[[str, float], float] | None = None, mtime: float | None = None,
                      max_tokens: int | str | None = "store", texts: Mapping[str, tuple[str | None, str | None]] | None = None,
                      inputs: Mapping[str, str] | None = None, created: str | None = None) -> Path:
    """A REAL Inspect ``.eval`` log of the store's GSM8K pass through
    ``inspect_ai.log.write_eval_log`` (the MATH builder's shape, the GSM8K
    scorer's registered name, the question text as each sample's input and the
    answer as its target). ``created`` (``eval.created``) defaults to the
    manifest's accepted attempt's ``started`` — the log of the store AS
    GENERATED, inside the window the reader binds it to; pass another stamp to
    build a log from outside that window. Needs inspect_ai."""
    import os

    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalPlan,
        EvalPlanStep,
        EvalResults,
        EvalSample,
        EvalSpec,
        EvalStats,
        write_eval_log,
    )
    from inspect_ai.log._log import EvalError
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        GenerateConfig,
        ModelOutput,
        ModelUsage,
    )
    from inspect_ai.scorer import Score

    cell_dir = Path(cell_dir)
    manifest = completion.read_store_manifest(cell_dir)
    summary = completion.read_task_summary(cell_dir, TASK)
    if items is None:
        items = completion.read_task_outcomes(cell_dir, TASK)["items"]
    if vmodel is None:
        vmodel = str(manifest[C.MANIFEST_KEY_ROUTE]["virtual_model"])
    model = C.inspect_model_name(vmodel)
    gen: dict[str, Any] = {}
    if max_tokens == "store":
        max_tokens = int(summary["max_tokens"])
    if max_tokens is not None:
        gen["max_tokens"] = int(max_tokens)
    for f, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
        if summary.get(f) is not None:
            gen[f] = cast(summary[f])
    penalty = (manifest.get(C.MANIFEST_KEY_SAMPLING) or {}).get(C.IDENTITY_KEY_PRESENCE_PENALTY)
    if penalty is not None:
        gen["presence_penalty"] = float(penalty)
    plan = EvalPlan(name="plan", steps=[EvalPlanStep(solver="generate", params={})], config=GenerateConfig(**gen))
    kept = list(items)[: (len(items) if n_samples is None else int(n_samples))]
    rows_by_id: dict[str, dict[str, Any]] = {}
    side = cell_dir / "private" / CHANNELS_NAME
    if side.is_file():
        rows_by_id = {k: {"content": v[0], "reasoning": v[1]} for k, v in json.loads(side.read_text(encoding="utf-8")).items()}
    lift_side = cell_dir / "private" / LIFT_NAME
    lift_by_id: dict[str, dict[str, Any]] = json.loads(lift_side.read_text(encoding="utf-8")) if lift_side.is_file() else {}
    samples = []
    for k, it in enumerate(kept):
        last_cancelled = cancelled and k == len(kept) - 1
        value: Any = float("nan")
        if it["scored"]:
            v = float(it["value"]) if value_edit is None else float(value_edit(str(it["item_id"]), float(it["value"])))
            value = "C" if v == 1.0 else "I"
        usage = ModelUsage(input_tokens=int(it["prompt_tokens"]), output_tokens=int(it["completion_tokens"]))
        iid = str(it["item_id"])
        if texts and iid in texts:
            content, reasoning = texts[iid]
        elif iid in rows_by_id:
            content, reasoning = rows_by_id[iid].get("content"), rows_by_id[iid].get("reasoning")
        else:
            content, reasoning = ("" if it["content_empty_with_reasoning"] else "7"), ("reasoning" if it["think_block_present"] else None)
        message_content: Any = (content or "") if reasoning is None else [ContentReasoning(reasoning=reasoning), ContentText(text=content or "")]
        out_kw: dict[str, Any] = {}
        if iid in lift_by_id:
            # the provider's no-switch response-conversion record, as the real log carries it (inspect_served.NO_SWITCH_METADATA_KEY)
            out_kw["metadata"] = {isv.NO_SWITCH_METADATA_KEY: [dict(lift_by_id[iid])]}
        output = ModelOutput(model=model, usage=usage,
                             choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=message_content), stop_reason=it["stop_reason"])],
                             **out_kw)
        meta = {f: bool(it[f]) for f in ("think_block_present", "think_block_unterminated", "stray_open_tag")}
        if "stray_close_tag" in it:
            # the `none` split records its close-tag stray in the score metadata (think_split.SPLIT_METADATA_FIELDS_NONE)
            meta["stray_close_tag"] = bool(it["stray_close_tag"])
        pos = IDS.index(iid) if iid in IDS else None
        question = (inputs or {}).get(iid, QUESTIONS[pos] if pos is not None else f"unknown item {iid}")
        samples.append(EvalSample(id=iid, epoch=1, input=question, target=ANSWERS[pos] if pos is not None else "0", output=output,
                                  scores=None if last_cancelled else {SCORER_NAME: Score(value=value, metadata=meta)},
                                  model_usage={model: usage},
                                  error=EvalError(message="CancelledError()", traceback="", traceback_ansi="") if last_cancelled else None))
    if created is None:
        accepted = {a.get("attempt_id"): a for a in manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS) or []}.get(manifest.get(MANIFEST_KEY_ACCEPTED_ATTEMPT))
        created = str((accepted or {}).get("started") or "2026-01-01T00:00:00Z").replace("Z", "+00:00")
    spec = EvalSpec(created=created, task=TASK_NAME, model=model, config=EvalConfig(),
                    dataset=EvalDataset(name=TASK_NAME, samples=len(items)))
    log = EvalLog(eval=spec, plan=plan, status="cancelled" if cancelled else "success", stats=EvalStats(), samples=samples,
                  results=EvalResults(total_samples=len(items), completed_samples=len(kept) - (1 if cancelled else 0)),
                  error=EvalError(message="eval cancelled", traceback="", traceback_ansi="") if cancelled else None)
    log_dir = C.inspect_log_dir(cell_dir, TASK, sample_idx)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{TASK_NAME}_{_hash(name, 8)}.eval"
    write_eval_log(log, path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# -- the fixture the tests share ------------------------------------------------------------------------------------ #

#: per cell the correct-item set: none 170 / 200; a constitution object 150 (-10 pp, one-way: resolved "lower"); the
#: control objects 168 (a two-way discordance whose CI covers zero); the prompted cell 169; the si12 cells 160; the derived
#: pair 165 — small enough that every contrast has at least a few discordant items
CORRECT_NONE = set(range(170))
CORRECT_BY_CELL: Mapping[str, set[int]] = {
    CELL_NONE: CORRECT_NONE,
    CELL_PROMPTED: (set(range(160)) | set(range(170, 179))),
    CELL_PROMPTED_CONST: (set(range(158)) | set(range(170, 177))),
    "constv2-mix-full__psr_ce_all": set(range(150)), "constv2-mix-full__lora_r4_ce": set(range(152)),
    "constv2ctl2-full__psr_ce_all": (set(range(163)) | set(range(170, 175))), "constv2ctl2-full__lora_r4_ce": (set(range(164)) | set(range(170, 174))),
    "safety_refusal__psr_kl_all": set(range(160)), "safety_refusal__lora_r4": set(range(161)),
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": set(range(165)), "constv2-mix-full__lora_r4_ce_minus_ctl": set(range(166)),
}


def default_kwargs(row: Mapping[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"correct": set(CORRECT_BY_CELL[str(row["cell"])])}
    if FAMILY_BY_KEY[str(row["model_key"])] == "glm4v":
        kw["no_think"] = {40, 41}
    return kw


def build_all(gen_root: Path, *, rows: Sequence[Mapping[str, Any]] | None = None,
              overrides: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None, skip: Iterable[tuple[str, str, str]] = (),
              **common: Any) -> dict[str, Path]:
    """Every LAUNCH leg of the matrix (38 by default; a held row is never built;
    a row carrying ``expected_refusal`` is never built either — the standup
    refused it by name); ``overrides[(key, view, cell)]`` extends that leg's kwargs."""
    out: dict[str, Path] = {}
    skipped = set(skip)
    for r in launch_rows(rows):
        k = (r["model_key"], r["view"], r["cell"])
        if k in skipped or r.get("expected_refusal"):
            continue
        kw = {**default_kwargs(r), **common, **((overrides or {}).get(k, {}))}
        out[f"{k[0]}/{k[1]}/{k[2]}"] = build_store(gen_root, r, **kw)
    return out


def build_escalated_cell(gen_root: Path, row: Mapping[str, Any], *, at_cap: Iterable[int], loops: Iterable[int] = (),
                         correct: Iterable[int] | None = None, capped_scored_correct: Iterable[int] = (),
                         rung_correct: Iterable[int] | None = None, rung_still_at_cap: Iterable[int] = (),
                         rung_boundary: Iterable[int] = (), write_rung: bool = True, **common: Any) -> dict[str, Any]:
    """A cell over the trigger with its rung: the 8,192 store (``at_cap``
    item indices — those in ``capped_scored_correct`` are case (c) and CORRECT
    at the cap, the W/X fixture; the rest case (b) unscored; ``loops`` are
    non-terminating), its ladder record (escalate), the planner's subset file
    and — ``write_rung`` — the 16,384 store over exactly the eligible subset
    (``rung_correct`` indices INTO ``ids`` of the correct redraws;
    ``rung_still_at_cap`` still at the ceiling; ``rung_boundary`` terminated
    below 8,192 tokens). Returns the paths and the subset."""
    at_cap, loops = set(at_cap), set(loops)
    case_c = set(capped_scored_correct)
    correct = set(CORRECT_BY_CELL[str(row["cell"])]) if correct is None else set(correct)
    cd = build_store(gen_root, row, correct=correct | case_c, unscored=at_cap - case_c, capped_scored=case_c, loops=loops, **common)
    root = cd.parents[1]
    manifest = completion.read_store_manifest(cd)
    record = json.loads((root / CL.RECORD_NAME).read_text(encoding="utf-8"))
    _record_again, subset_doc = classify(root, threshold=row_trigger(row))   # the adapter's subset body (pure; the record on disk is this same call's)
    subset = [str(i) for i in record["eligible_ids"]]
    out: dict[str, Any] = {"source_cell_dir": cd, "source_out_root": root, "record": record, "subset": subset, "subset_doc": subset_doc,
                           "rung_cell_dir": None, "subset_path": None}
    if record["exit"] != CL.ESCALATE:
        assert subset_doc is None
        return out
    assert subset_doc is not None and sorted(subset_doc["ids"]) == sorted(subset)
    out["subset_path"] = write_rung_subset(gen_root, row, subset_doc)
    if write_rung:
        idx = {i: j for j, i in enumerate(subset)}
        still = {idx[IDS[j]] for j in rung_still_at_cap if IDS[j] in idx}
        boundary = {idx[IDS[j]] for j in rung_boundary if IDS[j] in idx}
        rc = ({idx[IDS[j]] for j in rung_correct if IDS[j] in idx} if rung_correct is not None
              else {j for j in range(len(subset)) if j % 2 == 0} - still)
        lengths = {j: 4000 for j in boundary}
        lengths.update({j: 9000 for j in range(len(subset)) if j not in boundary and j not in still})
        out["rung_cell_dir"] = build_store(gen_root, row, correct=rc, unscored=still, cap=CAP_RUNG, ids=subset, rung_source=manifest,
                                           subset_doc=subset_doc, subset_path=out["subset_path"], completion_tokens=lengths, ladder=False,
                                           **{k: v for k, v in common.items() if k not in ("ladder",)})
    return out


# -- the overlap check's two legs ----------------------------------------------------------------------------------- #


def build_retired_store(retired_root: Path, view: str, *, correct: Iterable[int], model_key: str | None = None,
                        extraction_failed: Iterable[int] = (), ids: Sequence[str] | None = None,
                        gen: Mapping[str, Any] | None = None, rows_edit: Callable[[list[dict[str, Any]]], None] | None = None) -> Path:
    """The retired instrument's ``none`` store under
    ``<retired_root>/<view>/cells/none__controls__t0__a0__<h>__<h>/``: 200
    ``s2parity/gsm8k_0shot`` ``acc`` rows (``g<idx>-<hash8>`` ids,
    ``extra_json {pred, used_fallback}``), 200 ``extraction_failed`` rows, one
    ``acc_aggregate``; a manifest whose ``resolved_config.gen`` is the OFF greedy
    512 the spec assumes."""
    hf, rev = S.model_pin(view)
    d = Path(retired_root) / view / "cells" / f"none__controls__t0__a0__2ff9cdff__{_hash(view, 8)}"
    d.mkdir(parents=True, exist_ok=True)
    correct, failed = set(correct), set(extraction_failed)
    ids = list(ids if ids is not None else RETIRED_IDS)
    rows: list[dict[str, Any]] = []
    base = {"arm_name": "none", "attempt": 0, "coeff": 1.0, "created_at": 1788828771.0, "domain": "math_s2parity", "engine": "vllm",
            "env_digest": "0000000000000000", "harness_config_hash": _hash(view, 16), "intervention_hash": "2ff9cdff091aef6f", "kind": "cell",
            "model_id": hf, "model_revision": rev, "position_order": 0, "run_id": "served-controls-none", "seed": 0, "slurm_job_id": "0",
            "sweep": "two_si_controls_served", "task": "s2parity/gsm8k_0shot", "task_version": "s2parity_v1", "tier": 1, "trial_id": 0,
            "wall_clock_s": None}
    for j, iid in enumerate(ids):
        rows.append({**base, "item_id": iid, "metric": "acc", "value": 1.0 if j in correct else 0.0,
                     "extra_json": json.dumps({"pred": float(j), "used_fallback": False})})
        rows.append({**base, "item_id": iid, "metric": "extraction_failed", "value": 1.0 if j in failed else 0.0,
                     "extra_json": json.dumps({"pred": None if j in failed else float(j), "used_fallback": False})})
    rows.append({**base, "item_id": "__aggregate__", "metric": "acc_aggregate", "value": len(correct) / len(ids), "position_order": None,
                 "extra_json": json.dumps({"n": len(ids), "finish_reasons": {"stop": len(ids)}})})
    if rows_edit:
        rows_edit(rows)
    (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = {"cell_id": "none", "harness_cell_id": "none", "model_id": hf, "model_revision": rev, "steer_positions": "all",
                "virtual_model": "petri__base", "serving_identity": "proxy_vllm+lens+interleaved", "serving_path": "base",
                "resolved_config": {"cell_id": "none", "gen": dict(gen or {"batch_size": 8, "do_sample": False, "max_new_tokens": 512}),
                                    "controls": ["gsm8k", "mmlu"], "driver": "served_capability_run_v2", "model_id": hf}, "host": "TESTHOST"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return d


def overlap_row(key: str) -> dict[str, Any]:
    """The overlap OFF leg's row: the model's ``main``-view ``none`` cell under the greedy row."""
    r = row_for(matrix_rows(), key, "main", CELL_NONE)
    r["sampling_mode"] = C.SAMPLING_GREEDY
    r["out_root"] = f"overlap/{key}/none/greedy_off512"
    r["rung_out_root"] = r["out_root"] + "__rung16384"
    return r


def build_overlap_stock_store(gen_root: Path, key: str, *, correct: Iterable[int], **kw: Any) -> Path:
    """The stock OFF greedy 512 leg of the overlap check under
    ``<gen_root>/overlap/<key>/none/greedy_off512/``."""
    return build_store(gen_root, overlap_row(key), correct=correct, cap=512, thinking_on=False, sampling_mode=C.SAMPLING_GREEDY,
                       ladder=False, **kw)


def pairing_rows() -> list[dict[str, str]]:
    """The synthetic Platinum selection as ``gsm8k_parity_rows`` shapes it: ``{item_id, question, answer}`` per row."""
    return [{"item_id": RETIRED_IDS[i], "question": QUESTIONS[i], "answer": ANSWERS[i]} for i in range(N)]


def write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


# -- the EIGHT-ARM extension's fixture pieces (math-eight-arms-extension; additive — no existing helper's output moves) -- #

#: the served chat-template digest per family the producer files (``provenance.serving.chat_template_hash``)
CHAT_TEMPLATE_HASH_BY_FAMILY: Mapping[str, str] = dict(S.CHAT_TEMPLATE_HASH_BY_FAMILY)
#: sha256[:16] of each prompted cell's SI text as the driver files it (``resolved_config.system_sha``; the analysis pins the same)
SI_PREFIX_SHA16: Mapping[str, str] = dict(S.SI_PREFIX_SHA16)
#: the extension / stub cell ids the MATH-500 store already carries (one spelling across both stores)
CELL_PROMPTED_CONST, CELL_ALPHASTEER, CELL_ARDITI = S.CELL_PROMPTED_CONST, S.CELL_ALPHASTEER, S.CELL_ARDITI
STEER_CELLS: tuple[str, ...] = tuple(S.STEER_CELLS)
WAVE_EXT, WAVE_STEER = "ext", "steer"
STATUS_LAUNCH, STATUS_COMPLETE, STATUS_HELD, STATUS_DEFERRED = "launch", "complete", "held", "deferred"
DEFERRED_REASON = "deferred: dose unmeasured (D2/D3) — the steering rung is designed, not launched"
#: the extension's main-view prompted rows (no constitution cell on the it view): the constitution SI on
#: gemma / qwen / nemotron, the safety SI on gemma / qwen (whose main views carry none in the published design)
EXT_ROWS_SPEC: tuple[tuple[str, str], ...] = (("gemma4e4b", CELL_PROMPTED_CONST), ("qwen35_4b", CELL_PROMPTED_CONST),
                                              ("nemotron3nano4b", CELL_PROMPTED_CONST), ("gemma4e4b", CELL_PROMPTED), ("qwen35_4b", CELL_PROMPTED))
STEER_MODELS: tuple[str, ...] = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
#: correct-item sets of the extension cells (small effects, a few discordant items each)
CORRECT_BY_CELL_EXT: Mapping[str, set[int]] = {CELL_PROMPTED_CONST: (set(range(157)) | set(range(170, 176)))}
CORRECT_BY_CELL.update(CORRECT_BY_CELL_EXT)         # type: ignore[attr-defined]


#: the SI the driver files as `provenance.si_id` — the MATH-500 store's reader (both stores' bare cell is `none`)
si_of = S.si_of


def apply_producer_fields(manifest: dict[str, Any], row: Mapping[str, Any], *, chat_template_hash: str | None = None,
                          native: bool = False) -> None:
    """Write the producer's route / serving / SI fields into a synthetic manifest the way the driver files them at the
    published pin (verified against the published stores): ``route.serving_identity`` passthrough on the bare and the
    prompted routes, lens on a fitted cell (the ``+lora+`` identity and the ``lora`` path on a NATIVELY served fitted
    cell — ``native``); ``provenance.si_id`` / ``provenance.serving``; ``resolved_config.system_sha`` (the prompted SI
    digest, else null); top-level ``engine`` / ``serving_path``."""
    key, cell = str(row["model_key"]), str(row["cell"])
    fitted = bool(row["fitted"])
    hf, rev = S.model_pin(str(row["tag"]))
    identity = NATIVE_SERVING_IDENTITY if (native and fitted) else S.serving_identity_of(fitted)
    si_id = si_of(cell)
    system_sha = SI_PREFIX_SHA16.get(str(si_id)) if cell.startswith(C.PROMPTED_CELL_PREFIX) else None
    vmodel = str(manifest[C.MANIFEST_KEY_ROUTE]["virtual_model"])
    manifest[C.MANIFEST_KEY_ROUTE]["serving_identity"] = identity
    manifest["serving_identity"], manifest["serving_path"], manifest["engine"] = identity, (("lora" if native else "lens") if fitted else "passthrough"), "vllm"
    prov = dict(manifest[C.MANIFEST_KEY_PROVENANCE])
    prov["si_id"] = si_id
    prov["serving"] = {"chat_template_hash": chat_template_hash or CHAT_TEMPLATE_HASH_BY_FAMILY[FAMILY_BY_KEY[key]], "engine": "vllm",
                       "requested_model": hf, "requested_revision": rev, "resolved_model_id": hf, "resolved_revision": rev,
                       "serving_identity": identity, "virtual_model": vmodel}
    manifest[C.MANIFEST_KEY_PROVENANCE] = prov
    rc = dict(manifest.get("resolved_config") or {})
    rc.update(cell_id=cell, model_id=hf, model_revision=rev, system_sha=system_sha, serving={"serving_identity": identity, "virtual_model": vmodel})
    manifest["resolved_config"] = rc


def ext_row(key: str, cell: str, *, index: int, wave: str = WAVE_EXT, status: str = STATUS_LAUNCH, view: str = "main",
            geometry_L: int | None = None) -> dict[str, Any]:
    """One extension row in the frozen matrix's field shape (plan P2: appended after the published rows), wave-isolated
    under ``controls_<wave>/`` with ``wave`` / ``status`` beside the published fields."""
    tag = tag_of(key, view)
    fitted = C.is_lens_cell(cell)
    # the per-model facts (the served reasoning parser, the placement constraint) are the PUBLISHED row's for this tag —
    # read off the frozen matrix, never decided here by model name
    template = next(r for r in matrix_rows() if r["tag"] == tag)
    row: dict[str, Any] = {
        "index": index, "job_id": f"g8k-{key}-{view}-{cell.replace('__', '-')}-{wave}", "model_key": key, "view": view, "tag": tag,
        "cell": cell, "cell_slug": cell_slug(cell), "harness_cell": C.cell_spellings(cell, declared=False).harness,
        "staged_id": staged_id_of(cell, view) if fitted else None, "object_sha256": object_pin(key, view, cell) if fitted else None,
        "fitted": fitted, "route_kind": "lens" if fitted else ("prompted" if cell.startswith(C.PROMPTED_CELL_PREFIX) else "base"),
        "served_positions_expected": "all" if fitted else None, "declared": False,
        "tuple_name": tuple_name_of(key, declared=False) if fitted else None,
        "sampling_mode": MODE_PRIMARY, "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]),
        "out_root": f"controls_{wave}/{tag}/{cell_slug(cell)}/{MODE_PRIMARY}",
        "rung_out_root": f"controls_{wave}/{tag}/{cell_slug(cell)}/{MODE_PRIMARY}__rung16384",
        "time_limit": "0-01:00:00", "family": FAMILY_BY_KEY[key], "reasoning_parser": template["reasoning_parser"],
        "cluster_constraint": template["cluster_constraint"],
        "status": status, "held_reason": None, "expected_refusal": False, "expected_refusal_reason": None, "wave": wave,
    }
    if geometry_L is not None:
        row["geometry_L"] = int(geometry_L)
        row["geometry_eval"] = "capability:gen_long_thinking"
    return row


def ext_rows(start: int | None = None, *, wave: str = WAVE_EXT, status: str = STATUS_LAUNCH, spec: Sequence[tuple[str, str]] = EXT_ROWS_SPEC,
             geometry_L: int | None = None) -> list[dict[str, Any]]:
    """The extension rows (main view): the prompted constitution cell on gemma / qwen / nemotron and the prompted safety
    cell on gemma / qwen, ``launch`` in wave ``ext`` by default."""
    start = len(matrix_rows()) if start is None else start
    return [ext_row(key, cell, index=start + i, wave=wave, status=status, geometry_L=geometry_L) for i, (key, cell) in enumerate(spec)]


def steer_rows(start: int, *, models: Sequence[str] = STEER_MODELS, status: str = STATUS_DEFERRED) -> list[dict[str, Any]]:
    """The AlphaSteer / Arditi stubs (plan P8): design-only rows (``deferred`` by default), never collected — no object
    pin yet (the admission record is empty), no tuple, the external object source named."""
    out: list[dict[str, Any]] = []
    for key in models:
        for cell in STEER_CELLS:
            tag = tag_of(key, "main")
            out.append({
                "index": start + len(out), "job_id": f"g8k-{key}-main-{cell}", "model_key": key, "view": "main", "tag": tag, "cell": cell,
                "cell_slug": cell, "harness_cell": cell, "staged_id": None, "object_sha256": None, "fitted": True, "route_kind": "lens",
                "served_positions_expected": "all", "declared": False, "tuple_name": None, "sampling_mode": MODE_PRIMARY,
                "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]),
                "out_root": f"controls_{WAVE_STEER}/{tag}/{cell}/{MODE_PRIMARY}", "rung_out_root": f"controls_{WAVE_STEER}/{tag}/{cell}/{MODE_PRIMARY}__rung16384",
                "time_limit": "0-01:30:00", "family": FAMILY_BY_KEY[key], "reasoning_parser": "gemma4" if key == "gemma4e4b" else None,
                "cluster_constraint": "any", "status": status, "held_reason": (None if status != STATUS_HELD else "held: no admission record"),
                "deferred_reason": DEFERRED_REASON if status == STATUS_DEFERRED else None, "object_source": "external",
                "expected_refusal": False, "expected_refusal_reason": None, "wave": WAVE_STEER,
            })
    return out


def eight_arm_matrix(*, with_steer: bool = True, ext_status: str = STATUS_LAUNCH, geometry_L: int | None = None) -> list[dict[str, Any]]:
    """The C4-style fixture's matrix: the 42 published rows (38 launch + 4 held, byte for byte), the five extension rows
    (``launch``, wave ``ext``, ``controls_ext/`` roots) and — by default — the six steer stubs (``deferred``)."""
    rows = matrix_rows() + ext_rows(status=ext_status, geometry_L=geometry_L)
    if with_steer:
        rows += steer_rows(len(rows))
    return rows


def build_rows(gen_root: Path, rows: Sequence[Mapping[str, Any]], *, skip: Iterable[tuple[str, str, str]] = (),
               overrides: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None, **common: Any) -> dict[str, Path]:
    """Every EVIDENCE leg (``launch`` / ``complete``) of ``rows`` under ``gen_root`` — the extension rows' stores land under
    their wave-isolated ``controls_<wave>/`` roots; a held / deferred row is never a store. ``common`` reaches every
    ``build_store`` (``producer_fields=True``, ``tree_sha=...``)."""
    out: dict[str, Path] = {}
    skipped = set(skip)
    for r in rows:
        k = (r["model_key"], r["view"], r["cell"])
        if r["status"] not in (STATUS_LAUNCH, STATUS_COMPLETE) or k in skipped or r.get("expected_refusal"):
            continue
        kw = {**default_kwargs(r), **common, **((overrides or {}).get(k, {}))}
        out[f"{k[0]}/{k[1]}/{k[2]}"] = build_store(gen_root, r, **kw)
    return out


# -- E3: the INSTRUCT-DIRECT ("it-doc") batch-64 RE-FIT rows (math-eight-arms-extension; decisions-e3) — additive -- #
#: the wave, the row's object source (the pin is the it-doc ADMISSION RECORD's, never the full22 pin file's), the lane the pin is
#: keyed under, and the three LoRA arms on the `it` view in the eight-arm order (ctl2 · mix · mix−ctl) — the reader's spellings
#: (tinycap_gsm8k_validators.OBJECT_SOURCE_ITDOC; pinned equal to the generator's by test)
WAVE_ITDOC = "itdoc"
OBJECT_SOURCE_ITDOC = "itdoc"
LANE_ITDOC = "itdoc_eb64"
ITDOC_CELL_CTL2, ITDOC_CELL_MIX, ITDOC_CELL_DIFF = "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"
ITDOC_CELLS: tuple[str, ...] = (ITDOC_CELL_CTL2, ITDOC_CELL_MIX, ITDOC_CELL_DIFF)
ITDOC_MODELS: tuple[str, ...] = ("gemma4e4b", "qwen35_4b")
#: the ruled staged-id spellings per object kind (the lane token names the effective batch that changed) and the sidecar of the kind
ITDOC_STAGED_PREFIX_BY_KIND: Mapping[str, str] = {"fitted": "itdoc64__", "derived": "derived_itdoc64__"}
ITDOC_SIDECAR_BY_KIND: Mapping[str, str] = {"fitted": "train_budget_meta.json", "derived": "derived_meta.json"}
ITDOC_HELD_REASON = "synthetic hold: harvested, not staged; new bank id pending"
#: SYNTHETIC qualification sentences (invented numbers): the reader must carry the ROW's text verbatim, never a retyped one
ITDOC_QUALIFICATION = ("synthetic qualification: the constitution cell is budget-limited, not converged (best_epoch 9 of 9, stopped_early "
                       "False — a box-edge argmin); the control early-stopped at epoch 1 of 4; any difference of the pair inherits this asymmetry")
ITDOC_QUALIFICATION_PATTERN = "synthetic pattern: a trained-to-budget numerator against a converged denominator; state it, do not smooth it"


def itdoc_kind_of(cell: str) -> str:
    return "derived" if cell.endswith("_minus_ctl") else "fitted"


def itdoc_pin(key: str, cell: str) -> str:
    """The synthetic it-doc pin — keyed by the FULL tuple (lane, model key, view, cell): a cell id is not unique across lanes."""
    return _hash(f"pin|{LANE_ITDOC}|{key}|it|{cell}")


def itdoc_row(key: str, cell: str, *, index: int, status: str = STATUS_HELD, pinned: bool = True, job_id: str | None = None,
              held_reason: str | None = ITDOC_HELD_REASON, qualification: str | None = ITDOC_QUALIFICATION,
              qualification_pattern: str | None = ITDOC_QUALIFICATION_PATTERN, superseded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One it-doc row in the regenerated matrix's shape: a fitted, undeclared, all-position LoRA row of the `it` view, wave
    ``itdoc`` under ``controls_itdoc/``, ``object_source: itdoc``, its lane, object kind and sidecar, the ruled staged id, a
    64-hex pin (``pinned``; ``None`` = not yet verified on disk, a held row), the model's ALL-POSITION tuple, the pair's
    qualification (+ the cross-model pattern sentence) verbatim, and — on the row that REPLACES a held direct-fit row in
    place — the superseded object."""
    view = "it"
    tag = tag_of(key, view)
    kind = itdoc_kind_of(cell)
    template = next(r for r in matrix_rows() if r["tag"] == tag)
    return {
        "index": index, "job_id": job_id or f"g8k-{key}-{view}-{cell.replace('__', '-')}-{WAVE_ITDOC}", "model_key": key, "view": view, "tag": tag,
        "cell": cell, "cell_slug": cell_slug(cell), "harness_cell": C.cell_spellings(cell, declared=False).harness,
        "staged_id": ITDOC_STAGED_PREFIX_BY_KIND[kind] + cell, "object_sha256": itdoc_pin(key, cell) if pinned else None,
        "fitted": True, "route_kind": "lens", "served_positions_expected": "all", "declared": False,
        "tuple_name": tuple_name_of(key, declared=False), "sampling_mode": MODE_PRIMARY, "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]),
        "out_root": f"controls_{WAVE_ITDOC}/{tag}/{cell_slug(cell)}/{MODE_PRIMARY}",
        "rung_out_root": f"controls_{WAVE_ITDOC}/{tag}/{cell_slug(cell)}/{MODE_PRIMARY}__rung16384",
        "time_limit": "0-01:30:00", "family": FAMILY_BY_KEY[key], "reasoning_parser": template["reasoning_parser"],
        "cluster_constraint": template["cluster_constraint"],
        "status": status, "held_reason": held_reason if status == STATUS_HELD else None, "expected_refusal": False, "expected_refusal_reason": None,
        "wave": WAVE_ITDOC, "extension": True, "object_source": OBJECT_SOURCE_ITDOC, "lane": LANE_ITDOC, "object_kind": kind,
        "bank_sidecar": ITDOC_SIDECAR_BY_KIND[kind], "bank_view": None, "bank_view_rel": None, "bank_cell_dir": None,
        "parents_expected": ({ITDOC_CELL_MIX: itdoc_pin(key, ITDOC_CELL_MIX), ITDOC_CELL_CTL2: itdoc_pin(key, ITDOC_CELL_CTL2)} if kind == "derived" else None),
        "superseded": dict(superseded) if superseded is not None else None, "stale_objects": {},
        "qualification": qualification, "qualification_pattern": qualification_pattern, "fit_record": None, "admission_notes": [],
    }


def itdoc_matrix(rows: Sequence[Mapping[str, Any]] | None = None, *, launch: Iterable[tuple[str, str]] = (),
                 models: Sequence[str] = ITDOC_MODELS, **row_kw: Any) -> list[dict[str, Any]]:
    """The eight-arm matrix (:func:`eight_arm_matrix` by default) plus the it-doc wave, the way the generator lays it out: the
    `it` view's held ``constv2-mix-full__lora_r4_ce`` row of each model is REPLACED IN PLACE by the re-fit's row (index, tag
    and job id kept; the superseded direct-fit object recorded on the new row), and the control and the difference rows are
    APPENDED. Every it-doc row is ``held`` except the ``(model_key, cell)`` pairs in ``launch`` (EVIDENCE rows); ``row_kw``
    reaches every :func:`itdoc_row`."""
    out = [dict(r) for r in (rows if rows is not None else eight_arm_matrix())]
    launch_set = {tuple(x) for x in launch}
    for key in models:
        i = next(i for i, r in enumerate(out) if (r["model_key"], r["view"], r["cell"]) == (key, "it", ITDOC_CELL_MIX))
        old = out[i]
        superseded = {"lane": "full22 (itdoc__ staged ids)", "staged_id": old["staged_id"], "object_sha256": old["object_sha256"],
                      "where": "synthetic: the D8-held direct-fit row this row replaces in place"}
        status = STATUS_LAUNCH if (key, ITDOC_CELL_MIX) in launch_set else STATUS_HELD
        out[i] = itdoc_row(key, ITDOC_CELL_MIX, index=old["index"], status=status, job_id=old["job_id"], superseded=superseded, **row_kw)
    for key in models:
        for cell in (ITDOC_CELL_CTL2, ITDOC_CELL_DIFF):
            status = STATUS_LAUNCH if (key, cell) in launch_set else STATUS_HELD
            out.append(itdoc_row(key, cell, index=len(out), status=status, **row_kw))
    return out


def build_itdoc_leg(gen_root: Path, row: Mapping[str, Any], *, tree_sha: str, **kw: Any) -> Path:
    """The store of one it-doc EVIDENCE leg under its wave-isolated root, at ``tree_sha``, with the producer fields and the
    four row-provenance fields of a run launched with ``--bank-view it`` (the object digest = the row's pin — identity by pin)."""
    return build_store(gen_root, row, **{**default_kwargs(row), "producer_fields": True, "tree_sha": tree_sha, "bank_view": "it", **kw})
