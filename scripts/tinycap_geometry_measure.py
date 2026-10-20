#!/usr/bin/env python
"""tinycap geometry measurement: pool the pass-1 usage ledgers into PROPOSED
``capability:*`` served-geometry sub-entries.

Input: one or more ``usage_ledger.jsonl`` files written by
``sisteer.capeval.serving.usage_ledger`` (one flat row per ATTEMPTED
/v1/chat/completions call — served or 400-rejected — with the prompt length
pre-tokenized via /v1/tokenize). Every row is classified into a geometry
sub-entry key by (task family -> class, cell kind, thinking):

    inspect/{math,hellaswag,mmlu_*,gsm8k,xstest,or_bench_hard_1k}, evalplus/*
                                      -> gen_long    (+ _thinking under thinking-ON)
    inspect/bfcl                      -> agent       (+ _thinking)

(``contracts.GEOMETRY_CLASS_OF_CONTROL`` / ``geometry_subentry``; P29 pools
HellaSwag-Inspect and the comparison tasks into gen_long). A row on the
GRADER ROUTE — its ``vmodel`` is the ``grader_virtual_model`` the store
manifests beside the ledger record (the routing table's ``none`` entry, the
unsteered base the Inspect model graders are built on; PB3 / P20) — is the
MODEL-GRADER'S traffic, not the target's: it pools into the INFORMATIONAL
key ``capability:grader`` (``contracts.GEOMETRY_GRADER_KEY``) with its own
p99, never into gen_long (Evals' verdict 2026-09-04: 6,348 cap-less grader
calls had sat in gen_long's clamp denominator and hidden an 8.7% clamp on
the capped GSM8K calls). Grader-route traffic is expected only on the
model-graded tasks (``contracts.MODEL_GRADED_INSPECT_TASKS``); on any other
task it is unexplained and REFUSES the measurement, as does a ledger whose
target and grader share one route (a ``none`` cell: not separable by route).
A row tagged with a task OUTSIDE the tinycap families (a legacy control) is
skipped and counted. A row with NO task tag (``task`` null — the ``X-Tinycap-Task``
header absent; the 2026-09-03 smoke's ledger carried 373 such rows because
the tiny posters did not yet send it) is REFUSED: the pool is keyed by
task family and a guess would size a window from traffic of unknown class.
``--allow-untagged`` skips and counts them instead, explicitly, and the
proposal records that it was passed.

Per key, over ALL attempted calls (rejected included — the review finding
the ledger exists for): prompt p50/p90/p99/max from ``prompt_tokens_pre``;
total = prompt + ``usage_completion_tokens`` on a served call, and prompt +
``max_tokens_requested`` on a CONTEXT-LENGTH-rejected one — a row the LEDGER
classified ``error_class == context_length`` (E12 item 5: an upstream 400
whose body is one of the pinned vLLM context-length forms,
``served_window.parse_context_overflow`` — the only recogniser; vLLM 400s a
call whose prompt + cap does not fit, so the cap is the honest stand-in for
its completion — an upper bound, stated in the note); ``clamp rate at L`` =
share of CAPPED calls with prompt + requested cap > L — a call sent with
NO ``max_tokens`` (``max_tokens_requested`` null; vLLM sizes its completion
to the room left) cannot clamp, so it stays in the prompt / total p99
pools but is OUTSIDE the clamp basis, numerator and denominator alike
(``n_capless``, its own basis field; a pool with no capped call has a
clamp rate of 0 by construction); ``frac_over_L`` = share of all calls
with total > L; ``n_rejected`` as its own basis field. Then the sizing
rule EXACTLY as ``geometry.sizing_problem`` encodes it: L = round-up(p99 total) — unless the clamp rate at that L exceeds
``CLAMP_RATE_CEILING`` (5%), then L = round-up(p99 prompt + the largest
requested cap) — with ``S = identity_seqs(L)``. The candidate window rounds
to 256 (the original measurement's convention; ``SIZING_HEADROOM`` allows
it). Each proposal is constructed as a real ``ServedGeometry`` so the
validator asserts the rule on construction; a pool under ``--min-calls`` is
reported as INSUFFICIENT and proposes nothing (a smoke ledger must never
size a window).

A COMPROMISED pool proposes nothing either (R6; status ``refused``, and the
CLI exits non-zero): any row whose prompt was never measured
(``prompt_tokens_pre`` null — the tokenize step failed and the call was
blocked), any served row without an integer ``usage_completion_tokens``,
any served row whose ``tokenize_mismatch`` is true (the server counted the
prompt differently from the pre-tokenize — the very number the window is
sized from is in dispute), and any REJECTED row whose ``error_class`` is
not ``context_length`` — a tool-schema / unsupported-field / malformed
400 (``bad_request_other``), an ``auth`` or ``route_refused`` reply, a
``rate_limited`` 429, a ``server_error`` 5xx, a ``transport`` death, a
``ledger_blocked`` call — or a row with NO ``error_class`` (a ledger
predating the taxonomy): only the recognised context-length form says
"prompt + cap did not fit"; a generic 400 says nothing about the window
and every other failure has no honest total, so none is ever scored as
``prompt + cap`` (before E12 item 5 EVERY non-ledger 400 was). The counts
are reported per class so the operator can see what compromised the
ledger.

Percentiles are nearest-rank (``sorted[ceil(q * n) - 1]``), stated here so
the doc's numbers are reproducible from the ledgers by hand.

Output (``--out DIR``): ``geometry_proposal.json`` (the entries as dicts +
pool stats + skipped counts), ``geometry_proposal.md`` (the section-6 rows
for docs/served_geometry_p99.md), ``geometry_entries.py.txt`` (the
``ServedGeometry(...)`` literals for ``geometry._entries``) and
``geometry_pinned.py.txt`` (the ``test_served_geometry.PINNED`` rows). These
are a PROPOSAL for Evals' acceptance (E2); landing them replaces the
provisional entries.

Cell kind and routes: ``--ledger PATH[:prompted|:fitted]``; the kind (from
``cell_id``), the target's ``route.virtual_model`` and the grader's
``route.grader_virtual_model`` are read from the store manifests next to
the ledger (``<out_root>/cells/*/manifest.json``) and REFUSED if none /
mixed — the kind suffix overrides the kind only; ``--grader-vmodel NAME``
names the grader route for a ledger with no manifests (a hand run). A
ledger whose manifests record no route block (predating it) is refused
unless ``--grader-vmodel`` is given: without the route the grader's calls
would pool into the target's classes silently — the very dilution this
key exists to end. ``--extra-rows PATH:kind`` pools additional
ledger-shaped rows (P17: the tiny prompts rendered offline under the
constitution / irrelevant SIs, so ``tiny_prompted`` is not sized from the
~90-token safety SI alone); such rows carry no grader route.

Run:
  python scripts/tinycap_geometry_measure.py --ledger <cell_out_root>/usage_ledger.jsonl \
      [--ledger ...] [--extra-rows rows.jsonl:prompted] --out <dir> [--measured-on YYYY-MM-DD] \
      [--allow-untagged] [--grader-vmodel NAME]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

#: The measurement's rounding of a candidate window (the original pass used
#: 256; geometry.SIZING_HEADROOM (512) accepts it).
ROUND_TO = 256
#: A pool with fewer attempted calls proposes nothing (a smoke never sizes).
MIN_CALLS = 20
CELL_KINDS = ("prompted", "fitted")
SOURCE = (f"{sg.MEASUREMENT_DOC} section 6 — the tinycap pass-1 usage ledgers "
          "(usage_ledger.jsonl: every attempted /v1/chat/completions call, prompt "
          "pre-tokenized via /v1/tokenize BEFORE forwarding; rejected calls included, "
          "their completion taken as the requested cap; cap-less calls outside the clamp "
          "basis; the grader route's calls filed under capability:grader), pooled by "
          "scripts/tinycap_geometry_measure.py")
#: The informational key the grader route's rows pool into (never served).
GRADER_KEY = C.GEOMETRY_GRADER_KEY
#: The only row tasks grader-route traffic is expected on.
GRADER_TASKS: Tuple[str, ...] = tuple(C.MODEL_GRADED_INSPECT_TASKS)

#: row task -> control atom (reverse of contracts.TASKS_OF_CONTROL)
_ATOM_OF_TASK: Dict[str, str] = {
    t: atom for atom, tasks in C.TASKS_OF_CONTROL.items() for t in tasks}
#: The one rejection class the measurement scores as ``prompt + cap``: the
#: ledger's ``error_class == context_length`` (E12 item 5 — derived in the
#: ledger from the upstream status + body through
#: ``served_window.parse_context_overflow``, never here from the raw
#: status), which implies the upstream HTTP 400. Every other class in
#: ``contracts.LEDGER_ERROR_CLASSES``, and a row without the column,
#: compromises the pool.
CONTEXT_LENGTH_REJECTION_CLASS = C.LEDGER_ERROR_CLASS_CONTEXT_LENGTH
CONTEXT_LENGTH_REJECTION_STATUS = 400
#: The skipped-rows key under which untagged rows are counted when
#: ``--allow-untagged`` is passed (a row's ``task`` is null).
UNTAGGED_KEY = "<untagged>"


class MeasureError(RuntimeError):
    """A pooling / classification refusal (never a silent default)."""


# --------------------------------------------------------------------------- #
# reading + classifying                                                        #
# --------------------------------------------------------------------------- #


def read_ledger(path: Path) -> Tuple[List[dict], int]:
    """The rows of one ledger file; a torn trailing line (a container that
    died mid-write) is dropped and counted, never guessed at."""
    rows: List[dict] = []
    torn = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
        else:
            torn += 1
    return rows, torn


@dataclass(frozen=True)
class LedgerRoute:
    """What the store manifests beside a ledger say about its traffic: the
    cell kind, the TARGET's virtual model and the GRADER's (the routing
    table's ``none`` entry — ``route.grader_virtual_model``, recorded by the
    driver from the table, P20). ``grader_vmodel`` None = no route block
    (the manifests predate it): grader traffic cannot then be told apart."""
    kind: str
    target_vmodel: Optional[str]
    grader_vmodel: Optional[str]
    #: the prompted cell's SI (``prompted__<si>``; None for a fitted cell) — MATH-500
    #: rows pool by the SI's PROMPT CLASS (``contracts.geometry_subentry``'s ``si``)
    si: Optional[str] = None


def _manifests(ledger_path: Path) -> List[dict]:
    out = []
    for m in sorted(Path(ledger_path).parent.glob("cells/*/manifest.json")):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, Mapping):
            out.append(dict(d))
    return out


def read_manifest_route(ledger_path: Path) -> LedgerRoute:
    """The cell kind (``prompted`` / ``fitted`` from ``cell_id``) and the
    target / grader virtual models (the manifest's ``route`` block) from the
    store manifests beside the ledger (``<out_root>/cells/*/manifest.json``);
    refuses when no manifest is found or the manifests disagree on any of
    the three. A manifest without a ``route`` block contributes None for
    both models (the caller decides whether that is acceptable)."""
    kinds, targets, graders, sis = set(), set(), set(), set()
    for d in _manifests(ledger_path):
        cell_id = d.get("cell_id")
        if isinstance(cell_id, str) and cell_id:
            prompted = cell_id.startswith(C.PROMPTED_CELL_PREFIX)
            kinds.add("prompted" if prompted else "fitted")
            sis.add(cell_id[len(C.PROMPTED_CELL_PREFIX):] if prompted else None)
        route = d.get(C.MANIFEST_KEY_ROUTE) if isinstance(d.get(C.MANIFEST_KEY_ROUTE), Mapping) else {}
        targets.add(route.get("virtual_model") if isinstance(route.get("virtual_model"), str) else None)
        graders.add(route.get("grader_virtual_model")
                    if isinstance(route.get("grader_virtual_model"), str) else None)
    if len(kinds) != 1:
        raise MeasureError(
            f"cannot infer the cell kind of {ledger_path}: manifests found {sorted(kinds) or 'none'} "
            f"— pass the kind explicitly as {ledger_path}:prompted or :fitted")
    if len(targets) != 1 or len(graders) != 1:
        raise MeasureError(
            f"the store manifests beside {ledger_path} disagree on the route: target "
            f"{sorted(map(str, targets))}, grader {sorted(map(str, graders))} — one ledger, one route")
    if len(sis) != 1:
        raise MeasureError(
            f"the store manifests beside {ledger_path} disagree on the prompted cell's SI "
            f"{sorted(map(str, sis))} — one ledger, one cell (MATH-500 rows pool by the SI's prompt class)")
    return LedgerRoute(kind=kinds.pop(), target_vmodel=targets.pop(), grader_vmodel=graders.pop(),
                       si=sis.pop())


def infer_cell_kind(ledger_path: Path) -> str:
    """``prompted`` / ``fitted`` from the store manifests beside the ledger
    (:func:`read_manifest_route`'s ``kind``)."""
    return read_manifest_route(ledger_path).kind


def classify(row: Mapping[str, Any], *, prompted: bool,
             grader_vmodel: Optional[str] = None, si: Optional[str] = None) -> Optional[str]:
    """The geometry key a ledger row pools into: :data:`GRADER_KEY` for a
    row on the grader route (``vmodel`` == the ledger's ``grader_vmodel``,
    when known — refusing a grader-route row on a task that has no model
    grader, which is unexplained traffic), else the served sub-entry of its
    task family; None for a row outside the tinycap task families (legacy
    controls, un-tagged calls). ``si`` is the prompted cell's SI (the store
    manifests' ``cell_id``): a MATH-500 row pools by the SI's PROMPT CLASS
    (``capability:gen_long_math`` for the safety SI,
    ``capability:gen_long_math_longsi`` for the constitution / irrelevant
    SIs), so a prompted ledger's MATH-500 rows REFUSE without it rather than
    pooling 7,5xx-token prompts into the entry measured at 1,041."""
    task = row.get("task")
    if grader_vmodel is not None and row.get("vmodel") == grader_vmodel:
        if task not in GRADER_TASKS:
            raise MeasureError(
                f"a call on the grader route {grader_vmodel!r} is tagged {task!r}, which has no "
                f"model grader (expected one of {list(GRADER_TASKS)}): unexplained traffic on "
                "the base route — refusing rather than pooling it into a class")
        return GRADER_KEY
    atom = _ATOM_OF_TASK.get(task) if isinstance(task, str) else None
    if atom is None:
        return None
    cls = C.GEOMETRY_CLASS_OF_CONTROL[C.control_family(atom)]
    try:
        return C.geometry_subentry(cls, prompted=prompted, thinking=bool(row.get("thinking")),
                                   si=si if prompted else None)
    except ValueError as e:                                  # an unresolvable prompt class
        raise MeasureError(
            f"a {task!r} row on a {'prompted' if prompted else 'fitted'} ledger"
            f"{f' (SI {si!r})' if si else ''} pools into no measured sub-entry: {e}") from e


# --------------------------------------------------------------------------- #
# pooling + sizing                                                             #
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile: ``sorted[ceil(q * n) - 1]``."""
    if not values:
        raise MeasureError("percentile of an empty pool")
    s = sorted(int(v) for v in values)
    k = max(1, math.ceil(q * len(s)))
    return s[k - 1]


def round_up(n: int, to: int = ROUND_TO) -> int:
    return -(-int(n) // to) * to


def _int(v: Any) -> Optional[int]:
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def rejection_class(row: Mapping[str, Any]) -> Optional[str]:
    """For a REJECTED row: ``None`` iff the LEDGER classified it
    ``context_length`` (and its status is the 400 that class implies) — the
    one class the measurement scores as prompt + cap; else the label of the
    class that compromises the pool (``"error_class bad_request_other"``,
    ``"error_class auth"``, ``"error_class transport"`` …), or ``"no
    error_class …"`` for a row from a ledger predating the taxonomy. The
    status alone never makes a rejection context-length (E12 item 5): a
    tool-schema, unsupported-field or malformed-prompt 400 is
    ``bad_request_other`` and is never scored."""
    cls = row.get("error_class")
    if not isinstance(cls, str) or not cls:
        return ("no error_class (a ledger row predating the error taxonomy — re-run the pass "
                "under the current ledger; a class is never guessed from the status)")
    if cls != CONTEXT_LENGTH_REJECTION_CLASS:
        return f"error_class {cls}"
    status = _int(row.get("status"))
    if status != CONTEXT_LENGTH_REJECTION_STATUS:
        return f"error_class {cls} with status {status!r}"   # the class implies the 400; a torn row
    return None


@dataclass
class Pool:
    key: str
    prompt: List[int] = field(default_factory=list)          # every measured call
    total: List[int] = field(default_factory=list)           # served: prompt + completion; rejected: prompt + cap
    cap: List[Optional[int]] = field(default_factory=list)   # requested max_tokens per call; None = cap-less
    rejected: List[bool] = field(default_factory=list)
    n_unmeasured: int = 0                                    # prompt never measured / completion missing
    n_mismatch: int = 0                                      # served rows whose tokenize_mismatch is true
    unexpected_rejections: Dict[str, int] = field(default_factory=dict)  # class -> n (R6)
    unmeasured_reasons: Dict[str, int] = field(default_factory=dict)
    tasks: Dict[str, int] = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        return len(self.prompt)

    @property
    def n_rejected(self) -> int:
        return sum(1 for r in self.rejected if r)

    @property
    def n_capless(self) -> int:
        """Calls sent with no ``max_tokens`` — outside the clamp basis."""
        return sum(1 for c in self.cap if c is None)

    @property
    def n_capped(self) -> int:
        return self.n_calls - self.n_capless

    @property
    def caps(self) -> List[int]:
        """The distinct requested caps (cap-less calls contribute none)."""
        return sorted({c for c in self.cap if c is not None})

    @property
    def requested_max_tokens(self) -> int:
        """The largest requested cap; 0 for a pool with no capped call."""
        return max(self.caps) if self.caps else 0

    @property
    def n_unexpected_rejected(self) -> int:
        return sum(self.unexpected_rejections.values())

    def compromised_reasons(self) -> List[str]:
        """Why this pool may NOT propose (R6); empty when it may."""
        out = []
        if self.n_unmeasured:
            out.append(f"{self.n_unmeasured} unmeasured call(s) {dict(sorted(self.unmeasured_reasons.items()))}")
        if self.n_mismatch:
            out.append(f"{self.n_mismatch} served call(s) with tokenize_mismatch")
        if self.n_unexpected_rejected:
            out.append(f"{self.n_unexpected_rejected} rejected call(s) outside the "
                       f"{CONTEXT_LENGTH_REJECTION_CLASS} class "
                       f"{dict(sorted(self.unexpected_rejections.items()))}")
        return out

    def _unmeasured(self, why: str) -> None:
        self.n_unmeasured += 1
        self.unmeasured_reasons[why] = self.unmeasured_reasons.get(why, 0) + 1

    def add(self, row: Mapping[str, Any]) -> None:
        self.tasks[str(row.get("task"))] = self.tasks.get(str(row.get("task")), 0) + 1
        pre = _int(row.get("prompt_tokens_pre"))
        if pre is None:
            self._unmeasured("prompt_tokens_pre null (tokenize failed, call blocked)")
            return
        cap = _int(row.get("max_tokens_requested"))      # None: cap-less (no max_tokens sent)
        rejected = bool(row.get("rejected"))
        if rejected:
            cls = rejection_class(row)
            if cls is not None:
                self.unexpected_rejections[cls] = self.unexpected_rejections.get(cls, 0) + 1
                return                             # no honest total for this call
            comp = cap or 0                        # error_class context_length: prompt + cap (upper bound;
            #                                        a cap-less rejection means the PROMPT alone overflowed)
        else:
            comp = _int(row.get("usage_completion_tokens"))
            if comp is None:
                self._unmeasured("served call without usage_completion_tokens")
                return
            if row.get("tokenize_mismatch") is True:
                self.n_mismatch += 1
        self.prompt.append(pre)
        self.total.append(pre + comp)
        self.cap.append(cap)
        self.rejected.append(rejected)

    def clamp_rate_at(self, L: int) -> float:
        """Share of CAPPED calls whose prompt + requested cap exceeds L — a
        cap-less call cannot clamp and is outside numerator and denominator
        alike; 0.0 for a pool with no capped call."""
        capped = [(p, c) for p, c in zip(self.prompt, self.cap) if c is not None]
        if not capped:
            return 0.0
        return sum(1 for p, c in capped if p + c > L) / len(capped)

    def frac_over(self, L: int) -> float:
        return sum(1 for t in self.total if t > L) / self.n_calls


def pool_rows(rows_by_kind: Iterable[Tuple[Any, ...]], *,
              allow_untagged: bool = False) -> Tuple[Dict[str, Pool], Dict[str, int]]:
    """Pool ``(row, cell_kind[, grader_vmodel[, si]])`` tuples into per-key pools;
    the second value counts what was skipped per task (rows tagged with a
    task outside the tinycap families — legacy controls). With a
    ``grader_vmodel`` the rows on that route pool into :data:`GRADER_KEY`
    (:func:`classify`); without one (a 2-tuple: offline renders, synthetic
    rows) no row is grader traffic. A row with NO task tag REFUSES the
    whole measurement (the pool needs per-task attribution and a guess
    would size a window from traffic of unknown class) unless
    ``allow_untagged`` skips and counts it under :data:`UNTAGGED_KEY`."""
    pools: Dict[str, Pool] = {}
    skipped: Dict[str, int] = {}
    n_untagged = 0
    for row, kind, *rest in rows_by_kind:
        grader_vmodel = rest[0] if rest else None
        si = rest[1] if len(rest) > 1 else None          # the prompted cell's SI
        if kind not in CELL_KINDS:
            raise MeasureError(f"cell kind must be one of {CELL_KINDS}, got {kind!r}")
        task = row.get("task")
        if not isinstance(task, str) or not task:
            n_untagged += 1
            skipped[UNTAGGED_KEY] = skipped.get(UNTAGGED_KEY, 0) + 1
            continue
        key = classify(row, prompted=(kind == "prompted"), grader_vmodel=grader_vmodel, si=si)
        if key is None:
            skipped[task] = skipped.get(task, 0) + 1
            continue
        pools.setdefault(key, Pool(key)).add(row)
    if n_untagged and not allow_untagged:
        raise MeasureError(
            f"{n_untagged} ledger row(s) carry no task attribution (the {C.HEADER_TASK} header "
            "was absent on the request) — the pool is keyed by task family, and guessing a "
            "class for untagged traffic would size a window from calls of unknown kind; "
            "refusing. Re-run the pass with posters that send the header (the tiny posters "
            "gained it after the 2026-09-03 smoke), or pass --allow-untagged to skip and count "
            "these rows explicitly")
    return pools, skipped


def size_pool(pool: Pool, *, measured_on: str, round_to: int = ROUND_TO,
              min_calls: int = MIN_CALLS,
              families: Tuple[str, ...] = ("gemma4e4b",)) -> Dict[str, Any]:
    """The sizing rule applied to one pool -> a proposal record: the stats,
    the branch taken, and (when the pool is large enough) a constructed
    ``ServedGeometry`` whose validator has asserted the rule."""
    ev, task = pool.key.split(":", 1)
    informational = pool.key == GRADER_KEY
    stats: Dict[str, Any] = {
        "key": pool.key, "n_calls": pool.n_calls, "n_rejected": pool.n_rejected,
        "n_capped": pool.n_capped, "n_capless": pool.n_capless, "informational": informational,
        "n_unmeasured": pool.n_unmeasured, "n_tokenize_mismatch": pool.n_mismatch,
        "n_unexpected_rejected": pool.n_unexpected_rejected,
        "unexpected_rejections": dict(sorted(pool.unexpected_rejections.items())),
        "unmeasured_reasons": dict(sorted(pool.unmeasured_reasons.items())),
        "tasks": dict(sorted(pool.tasks.items())),
    }
    compromised = pool.compromised_reasons()
    if compromised:                                # R6: a compromised ledger proposes NOTHING
        stats.update(status="refused",
                     reason="compromised ledger — " + "; ".join(compromised)
                            + ". Every attempted call must be measured (prompt pre-tokenized, "
                              "completion counted, no tokenize mismatch) and every rejection must "
                              f"carry error_class {CONTEXT_LENGTH_REJECTION_CLASS} (the ledger's "
                              "verdict, never the raw status) before a window is sized from this pool")
        return stats
    if pool.n_calls < min_calls:
        stats.update(status="insufficient",
                     reason=f"{pool.n_calls} measured call(s) < min_calls {min_calls}: a pool "
                            "this thin (a smoke) never sizes a window")
        return stats
    p = pool.prompt
    stats.update(
        p50_prompt=percentile(p, 0.50), p90_prompt=percentile(p, 0.90),
        p99_prompt=percentile(p, 0.99), max_prompt=max(p),
        p50_total=percentile(pool.total, 0.50), p90_total=percentile(pool.total, 0.90),
        p99_total=percentile(pool.total, 0.99), max_total=max(pool.total),
        requested_max_tokens=pool.requested_max_tokens, caps=pool.caps,
        rejection_rate=pool.n_rejected / pool.n_calls,
    )
    L_total = round_up(stats["p99_total"], round_to)
    clamp_total = pool.clamp_rate_at(L_total)
    stats["L_at_total_sizing"] = L_total
    stats["clamp_rate_at_total_sizing"] = clamp_total
    if clamp_total > sg.CLAMP_RATE_CEILING:
        L = round_up(stats["p99_prompt"] + stats["requested_max_tokens"], round_to)
        sized_by = sg.SIZED_BY_PROMPT_PLUS_MAX
    else:
        L = L_total
        sized_by = sg.SIZED_BY_TOTAL
    stats.update(L=L, S=sg.identity_seqs(L), sized_by=sized_by,
                 clamp_rate_at_L=pool.clamp_rate_at(L), frac_over_L=pool.frac_over(L))
    capless = (f"{pool.n_capless} of them CAP-LESS (no max_tokens sent: vLLM sizes the completion "
               f"to the room left, so they cannot clamp — in the prompt / total p99s, outside "
               f"the clamp basis of {pool.n_capped} capped calls)"
               if pool.n_capless else
               f"every call capped (clamp basis = all {pool.n_capped})")
    if informational:
        head = (f"INFORMATIONAL — the MODEL-GRADER traffic on the unsteered base route (the "
                f"routing table's {C.ROUTE_NONE_CELL!r} entry): a different workload from the "
                "target's, measured for the record with its own p99 and served by no launcher "
                "(the grader rides the target's server at the target's window; "
                "resolve_geometry refuses this key). Kept OUT of capability:gen_long, whose "
                "clamp denominator it had diluted. ")
    else:
        head = "MEASURED from the pass-1 usage ledgers: "
    note = (head
            + f"{pool.n_calls} attempted calls, {capless}; "
            f"{pool.n_rejected} rejected with the context-length 400 at the certified footprint — "
            f"the ledger's error_class {CONTEXT_LENGTH_REJECTION_CLASS}, never a raw 400 — "
            "entering the basis at prompt + requested cap; every call pre-tokenized and counted — "
            "a pool with any unmeasured, tokenize-mismatch or otherwise-rejected call is refused, "
            "never sized. Tasks pooled: "
            f"{', '.join(f'{t}={n}' for t, n in sorted(pool.tasks.items()))}. Requested caps: "
            f"{stats['caps'] or 'none (cap-less)'}. "
            + (f"Sized by the rule's second branch: at the total-p99 window ({L_total}) the "
               f"clamp fires on {clamp_total:.1%} of the capped calls, so L = round-up("
               f"{stats['p99_prompt']} + {stats['requested_max_tokens']}) = {L}."
               if sized_by == sg.SIZED_BY_PROMPT_PLUS_MAX else
               f"Sized by the total-p99 branch: L = round-up({stats['p99_total']}) = {L}; the "
               f"clamp rate there is {clamp_total:.2%}"
               + (" of the capped calls." if pool.n_capped else
                  " by construction (no capped call)."))
            + (" Recorded for Evals (E2); a different workload, never pooled into gen_long."
               if informational else
               " Accepted by Evals (E2 / the 2026-09-04 verdict); replaces the provisional entry."))
    basis = sg.GeometryBasis(
        n_calls=pool.n_calls, p99_total=stats["p99_total"], max_total=stats["max_total"],
        p99_prompt=stats["p99_prompt"], requested_max_tokens=stats["requested_max_tokens"],
        clamp_rate_at_L=round(stats["clamp_rate_at_L"], 4),
        frac_over_L=round(stats["frac_over_L"], 4),
        clamp_rate_at_total_sizing=(round(clamp_total, 4)
                                    if sized_by == sg.SIZED_BY_PROMPT_PLUS_MAX else None),
        families_measured=tuple(families), measured_on=measured_on, source=SOURCE,
        note=note, n_rejected=pool.n_rejected, n_capless=pool.n_capless)
    entry = sg.ServedGeometry(eval=ev, task=task, max_model_len=L,
                              max_num_seqs=sg.identity_seqs(L), basis=basis,
                              informational=informational)
    assert sg.sizing_problem(entry) is None
    stats.update(status="proposed", entry=entry.as_dict())
    stats["_entry"] = entry
    return stats


# --------------------------------------------------------------------------- #
# rendering                                                                    #
# --------------------------------------------------------------------------- #


def render_entry_py(g: sg.ServedGeometry) -> str:
    """The ``ServedGeometry(...)`` literal for ``geometry._entries``."""
    b = g.basis
    return (
        f"        ServedGeometry(\n"
        f"            eval={g.eval!r}, task={g.task!r},\n"
        f"            max_model_len={g.max_model_len}, max_num_seqs=identity_seqs({g.max_model_len}),"
        f"      # {g.max_num_seqs}\n"
        f"            basis=GeometryBasis(\n"
        f"                n_calls={b.n_calls}, p99_total={b.p99_total}, max_total={b.max_total}, "
        f"p99_prompt={b.p99_prompt},\n"
        f"                requested_max_tokens={b.requested_max_tokens}, "
        f"clamp_rate_at_L={b.clamp_rate_at_L}, frac_over_L={b.frac_over_L},\n"
        f"                clamp_rate_at_total_sizing={b.clamp_rate_at_total_sizing},\n"
        f"                families_measured={b.families_measured!r},\n"
        f"                measured_on={b.measured_on!r}, source=_TINYCAP_MEASURED_SOURCE,\n"
        f"                n_rejected={b.n_rejected}, n_capless={b.n_capless},\n"
        f"                note={b.note!r},\n"
        f"            )" + (",\n            informational=True),\n" if g.informational else "),\n"))


def render_pinned_row(g: sg.ServedGeometry) -> str:
    """The ``test_served_geometry.PINNED`` row."""
    b = g.basis
    return (f"    {g.key!r}: dict(L={g.max_model_len}, S={g.max_num_seqs}, n={b.n_calls}, "
            f"p99_total={b.p99_total}, max_total={b.max_total},\n"
            f"        p99_prompt={b.p99_prompt}, max_tokens={b.requested_max_tokens}, "
            f"clamp={b.clamp_rate_at_L}, over={b.frac_over_L},\n"
            f"        fams={b.families_measured!r}, sized_by={g.sized_by!r}, "
            f"measured_on={b.measured_on!r}, n_rejected={b.n_rejected}, n_capless={b.n_capless}"
            + (", informational=True" if g.informational else "") + "),\n")


def render_doc_rows(proposals: Sequence[Mapping[str, Any]]) -> str:
    """The section-6 markdown rows for docs/served_geometry_p99.md."""
    head = ("| key | n calls | rejected (400) | p50 | p90 | p99 | max | p99 prompt | max_tokens "
            "| L / S | frac > L | clamp (prompt+max_tokens > L) | clamp at total sizing "
            "| cap-less (outside the clamp basis) | sized by |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    lines = []
    for s in proposals:
        if s.get("status") != "proposed":
            lines.append(f"| {s['key']} | {s['n_calls']:,} | {s['n_rejected']} | - | - | - | - | - | - "
                         f"| - | - | - | - | {s['n_capless']:,} | {s['status'].upper()} ({s.get('reason', '')}) |")
            continue
        lines.append(
            f"| {s['key']} | {s['n_calls']:,} | {s['n_rejected']} | {s['p50_total']:,} | "
            f"{s['p90_total']:,} | {s['p99_total']:,} | {s['max_total']:,} | {s['p99_prompt']:,} | "
            f"{s['requested_max_tokens']:,} | {s['L']:,} / {s['S']} | {s['frac_over_L']:.2%} | "
            f"{s['clamp_rate_at_L']:.2%} | {s['clamp_rate_at_total_sizing']:.2%} | {s['n_capless']:,} | "
            f"{s['sized_by']}{' (informational)' if s.get('informational') else ''} |")
    return head + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _split_kind(spec: str, *, infer: bool) -> Tuple[Path, str]:
    """``PATH[:kind]`` -> (path, kind); the kind inferred from the manifests
    beside the ledger when absent and ``infer`` (refused otherwise)."""
    path, _, kind = spec.rpartition(":")
    if kind in CELL_KINDS and path:
        return Path(path), kind
    if not infer:
        raise MeasureError(f"{spec!r}: expected PATH:prompted or PATH:fitted")
    p = Path(spec)
    return p, infer_cell_kind(p)


def ledger_spec(spec: str, *, grader_override: Optional[str] = None) -> Tuple[Path, str, str, Optional[str]]:
    """A ``--ledger PATH[:kind]`` argument -> ``(path, kind, grader_vmodel, si)``
    — ``si`` the prompted cell's SI from the manifests (None for a fitted
    ledger, or when the ``:fitted`` suffix overrides the kind), the class a
    MATH-500 row pools by.
    The grader route comes from the store manifests beside the ledger
    (:func:`read_manifest_route`) unless ``grader_override`` names it; a
    ledger with neither is REFUSED — without the route the grader's calls
    would pool into the target's classes silently. A ledger whose target
    shares the grader's route (a ``none`` cell) is refused too: its grader
    traffic is not separable by route."""
    path, kind = _split_kind(spec, infer=True)
    route = None
    try:
        route = read_manifest_route(path)
    except MeasureError as e:
        if grader_override is None:
            raise MeasureError(
                f"{path}: no grader route — {e}; without the store manifests' route block the "
                "grader's calls cannot be told from the target's. Pass --grader-vmodel NAME (the "
                "routing table's none entry) to name the grader route explicitly") from None
    if route is not None and route.target_vmodel is not None \
            and route.target_vmodel == (grader_override or route.grader_vmodel):
        raise MeasureError(
            f"{path}: the target and the grader share the route {route.target_vmodel!r} (a "
            f"{C.ROUTE_NONE_CELL!r} cell) — grader traffic cannot be separated by route; this "
            "ledger cannot be pooled until the ledger tags the role")
    grader = grader_override if grader_override is not None else (route.grader_vmodel if route else None)
    if grader is None:
        raise MeasureError(
            f"{path}: the store manifests beside it record no route block (grader_virtual_model), "
            "so the grader's calls cannot be told from the target's — pass --grader-vmodel NAME "
            "(the routing table's none entry) to name the grader route explicitly")
    si = route.si if (route is not None and kind == "prompted") else None
    return path, kind, grader, si


def measure(ledgers: Sequence[Tuple[Any, ...]], extra: Sequence[Tuple[Path, str]] = (), *,
            measured_on: str, round_to: int = ROUND_TO, min_calls: int = MIN_CALLS,
            families: Tuple[str, ...] = ("gemma4e4b",),
            allow_untagged: bool = False) -> Dict[str, Any]:
    """Pool every input and size every key present; the proposal record.
    A ledger is ``(path, kind)``, ``(path, kind, grader_vmodel)`` or ``(path,
    kind, grader_vmodel, si)`` — only a grader-bearing one can file
    grader-route rows under :data:`GRADER_KEY`, and only an SI-bearing one
    can pool a prompted cell's MATH-500 rows (the CLI always builds the full
    tuple, :func:`ledger_spec`); ``extra`` rows carry neither. ``refused_keys`` names every pool a compromised ledger kept from
    proposing (R6)."""
    pairs: List[Tuple[Any, ...]] = []
    inputs = []
    torn_total = 0
    for spec in list(ledgers) + [tuple(e) for e in extra]:
        path, kind, *rest = spec
        grader = rest[0] if rest else None
        si = rest[1] if len(rest) > 1 else None
        rows, torn = read_ledger(path)
        torn_total += torn
        n_grader = sum(1 for r in rows if grader is not None and r.get("vmodel") == grader)
        inputs.append({"path": str(path), "cell_kind": kind, "si": si, "n_rows": len(rows), "torn": torn,
                       "grader_vmodel": grader, "n_grader_rows": n_grader})
        pairs.extend((r, kind, grader, si) for r in rows)
    pools, skipped = pool_rows(pairs, allow_untagged=allow_untagged)
    proposals = [size_pool(pools[k], measured_on=measured_on, round_to=round_to,
                           min_calls=min_calls, families=families) for k in sorted(pools)]
    absent = [k for k in C.GEOMETRY_SUBENTRY_KEYS if k not in pools]
    refused = [s["key"] for s in proposals if s.get("status") == "refused"]
    return {"measured_on": measured_on, "inputs": inputs, "torn_lines": torn_total,
            "skipped_rows": skipped, "absent_keys": absent, "refused_keys": refused,
            "proposals": proposals,
            "rule": {"round_to": round_to, "min_calls": min_calls,
                     "clamp_rate_ceiling": sg.CLAMP_RATE_CEILING,
                     "percentile": "nearest-rank sorted[ceil(q*n)-1]",
                     "rejected_total": "prompt_tokens_pre + max_tokens_requested, for rows the ledger "
                                       f"classified error_class == {CONTEXT_LENGTH_REJECTION_CLASS} "
                                       f"(HTTP {CONTEXT_LENGTH_REJECTION_STATUS}) only",
                     "refuse_when": "any unmeasured / tokenize-mismatch call, any rejection whose "
                                    f"error_class is not {CONTEXT_LENGTH_REJECTION_CLASS}, or a row "
                                    "without error_class",
                     "clamp_basis": "capped calls only — a cap-less call (no max_tokens_requested) "
                                    "cannot clamp and is outside the clamp numerator and "
                                    "denominator alike (n_capless); a pool with no capped call "
                                    "has clamp rate 0",
                     "grader_route": f"rows whose vmodel is the manifests' grader_virtual_model "
                                     f"pool into {GRADER_KEY} (informational), expected only on "
                                     f"{list(GRADER_TASKS)}",
                     "allow_untagged": bool(allow_untagged)}}


def write_outputs(result: Mapping[str, Any], out: Path) -> Dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    proposals = result["proposals"]
    entries = [s["_entry"] for s in proposals if s.get("status") == "proposed"]
    clean = {k: v for k, v in result.items() if k != "proposals"}
    clean["proposals"] = [{k: v for k, v in s.items() if k != "_entry"} for s in proposals]
    (out / "geometry_proposal.json").write_text(json.dumps(clean, indent=1, default=str) + "\n")
    md = (f"## 6. tinycap suite (proposed sub-entries; measured {result['measured_on']} from the "
          "pass-1 usage ledgers)\n\n"
          "Every attempted call (rejected 400s included at prompt + requested cap); prompt "
          "pre-tokenized via `/v1/tokenize` before forwarding. Nearest-rank percentiles. Clamp "
          "rates are over the CAPPED calls (a cap-less call cannot clamp); the grader route's "
          "calls are their own informational key.\n\n"
          + render_doc_rows(proposals))
    (out / "geometry_proposal.md").write_text(md)
    (out / "geometry_entries.py.txt").write_text(
        "# paste into geometry._entries() in place of the provisional rows; define\n"
        "# _TINYCAP_MEASURED_SOURCE = the SOURCE string of tinycap_geometry_measure.py\n"
        + "".join(render_entry_py(g) for g in entries))
    (out / "geometry_pinned.py.txt").write_text(
        "# paste into test_served_geometry.PINNED in place of the provisional rows\n"
        + "".join(render_pinned_row(g) for g in entries))
    return {"json": str(out / "geometry_proposal.json"), "md": str(out / "geometry_proposal.md"),
            "entries": str(out / "geometry_entries.py.txt"),
            "pinned": str(out / "geometry_pinned.py.txt")}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", action="append", default=[],
                    help="usage_ledger.jsonl[:prompted|:fitted]; repeatable. Without the "
                         "suffix the kind is inferred from the store manifests beside it")
    ap.add_argument("--extra-rows", action="append", default=[],
                    help="ledger-shaped rows PATH:prompted|fitted to pool in addition "
                         "(P17: offline tiny renders under the long SIs)")
    ap.add_argument("--out", required=True, help="the proposal directory")
    ap.add_argument("--measured-on", default=_dt.date.today().isoformat())
    ap.add_argument("--round", type=int, default=ROUND_TO)
    ap.add_argument("--min-calls", type=int, default=MIN_CALLS)
    ap.add_argument("--families", default="gemma4e4b", help="csv of the families pooled")
    ap.add_argument("--allow-untagged", action="store_true",
                    help="skip and count rows with no task tag instead of refusing the measurement")
    ap.add_argument("--grader-vmodel", default=None,
                    help="the grader route's virtual model for EVERY --ledger (the routing table's "
                         "none entry); by default read from the store manifests beside each ledger")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ledger and not args.extra_rows:
        raise SystemExit("pass at least one --ledger (or --extra-rows)")
    try:
        ledgers = [ledger_spec(s, grader_override=args.grader_vmodel) for s in args.ledger]
        extra = [_split_kind(s, infer=False) for s in args.extra_rows]
        result = measure(ledgers, extra, measured_on=args.measured_on, round_to=args.round,
                         min_calls=args.min_calls,
                         families=tuple(f for f in args.families.split(",") if f),
                         allow_untagged=args.allow_untagged)
    except MeasureError as e:
        raise SystemExit(f"[measure] {e}")
    paths = write_outputs(result, Path(args.out))
    for s in result["proposals"]:
        if s.get("status") == "proposed":
            print(f"[measure] {s['key']}: L={s['L']} S={s['S']} ({s['sized_by']}"
                  f"{', informational' if s.get('informational') else ''}) n={s['n_calls']} "
                  f"capless={s['n_capless']} rejected={s['n_rejected']} p99_total={s['p99_total']} "
                  f"p99_prompt={s['p99_prompt']} clamp@L={s['clamp_rate_at_L']:.4f} "
                  f"frac>L={s['frac_over_L']:.4f}")
        else:
            print(f"[measure] {s['key']}: {s['status'].upper()} — {s.get('reason')}")
    if result["absent_keys"]:
        print(f"[measure] no rows for: {result['absent_keys']}")
    if result["skipped_rows"]:
        print(f"[measure] skipped (legacy / untagged) rows: {result['skipped_rows']}")
    print(f"[measure] wrote {paths}")
    if result["refused_keys"]:
        print(f"[measure] REFUSED (compromised ledger — no entry proposed): {result['refused_keys']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
