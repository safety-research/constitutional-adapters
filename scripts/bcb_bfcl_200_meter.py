#!/usr/bin/env python
"""The H200-hour meter of the bcb-bfcl-200 lanes — PURE: a JSONL ledger,
arithmetic, an exit code. No Modal, no network, no GPU.

The envelope is ONE aggregate ceiling, **25.0 H200-h** for the smokes + every
runnable cell of every wave (the spec's Constraints). There is no per-wave
sub-ceiling and no item count is ever assumed. The truncation ladder's own
allowance — "+ ≤ 1.5 H200-h per leg ONLY if the truncation rule fires" — is NOT
folded into the ceiling automatically: it enters as a RECORDED decision that
names the fired leg(s) and raises the ceiling by at most 1.5 h per newly named
leg (``--decision "ladder fired: <wave>/<leg>" --ceiling-h X``, or the legs as
``--fired-leg <wave>/<leg>`` flags); a raise that names no fired leg, names one
already credited, or exceeds 1.5 h per leg is REFUSED. Lowering or restating the
ceiling needs no leg.

**An explicit ruling is the other door**: the fired-leg rule is the ladder's
allowance, not the only way a ceiling moves. A decision that cites an explicit
ruling — ``--user-ruling "<who, when, what>"``, a non-empty string — is admitted
as a raise WITHOUT a fired leg and without the per-leg cap; the entry carries
``user_ruling`` verbatim (``fired_legs`` null: a ruling and a fired leg are two
decisions, never one line), and the meter prints the ruling on the decision line.
It never credits a leg, never lowers the cap on later fired-leg raises, and never
touches a forecast or an actual: it moves the ceiling in force, nothing else.
Without the flag nothing changes.

**Reservation identity**. A reservation is ``(wave, suffix, unit)``
where ``unit`` is a cell id or a probe group (``probe:<cell>``, ``reprobe:<cell>``,
``budget:<model_key>``, ``smoke``, ``parser-probe``), with the stage KIND
(``smoke``, ``wave``, ``ladder-probe``, ``ladder-reprobe``, ``ladder-score``,
``budget``, ``parser-probe``) a SEPARATE field — so two waves' ladder probes never
replace each other's committed hours. ``--record --actual-h`` replaces the
forecast of THAT identity only. The admission transaction (read the ledger →
check ``Σ committed of the other identities + forecast ≤ ceiling`` → append) runs
under ``fcntl.flock`` on ``<ledger>.lock``, so two simultaneous near-ceiling
admissions admit exactly one. A legacy entry (``stage`` only, from an earlier
ledger format) is still read: its identity is its stage name.

**The ladder forecast** is COMPUTED, never assumed:

    probe_s(cell)     = rungs_walked(cell) × standup_s        # one probe standup per FITTED cell
    reprobe_s(cell)   = standup_s                              # a fitted cell whose accepted rung ≠ the common rung
    waves(task, cell) = ceil(n_selected(task) / S)
    wave_s(cell)      = cap_to / tok_s(cell)
    scoring_s         = Σ_cell [standup_s + Σ_task waves × wave_s]

with S and the rung list from the MODEL's ``LadderSpec`` (``rung_probe.ladder_spec(
model_key)`` — Slice B; ``--model``). **Probe spend and scoring spend are INDEPENDENT
ledger records**: the pre-probe reservation ``probe:<cell>``
budgets the lowest rung, a full walk AND one re-probe standup per fitted cell, and
only ITS recorded actual (the certify step's wall) ever replaces that forecast; the
scoring admission (``--stage-kind ladder-score``) takes the common S from the
per-cell certificates (``--certificate`` repeated) and forecasts the SCORING alone —
it neither carries nor supersedes the probe reservations, so a re-probe that folds
the first walk's timing into ``reprobed_from`` or a scoring actual that replaces the
scoring forecast can never make incurred probe spend vanish.

**An admission is a CLAIM**: the attempt number the meter
prints is the launch's claim token, unique per identity. An identity whose latest
attempt is OPEN — a forecast admitted, no actual recorded, not superseded — is
already claimed, and a second admission under it is REFUSED (exit 2, "CLAIMED"):
a repeated or retried launch never rides an earlier launch's reservation. The
claim closes when its actual is recorded (``--record``) or, when the launch never
ran, when it is RELEASED (``--release [why]`` — an actual of 0.0 h whose note says
so; a released scoring claim hands the probe reservations it superseded back).
Only then does the next admission open attempt n + 1; ``committed_hours`` = every
attempt's actual-or-forecast.

Every dollar line is a RESOURCE-RATE ESTIMATE (H200 ≈ $5/h), not billing.

Usage::

    meter.py --decision "spec envelope" --ceiling-h 25.0                          # the first entry
    meter.py --wave gemma4e4b_it --suffix bcbw-gemma4e4b_it-abc1234 --unit smoke --stage-kind smoke --forecast-h 0.8
    meter.py --wave W --suffix S --unit none --stage-kind wave --forecast-h 0.18
    meter.py --record --wave W --suffix S --unit none --actual-h 0.17
    meter.py --release --wave W --suffix S --unit none                            # the claimed launch never ran
    meter.py --wave W --suffix S --unit probe:<cell> --stage-kind ladder-probe --model gemma4e4b --selection F
    meter.py --wave W --suffix S --unit score --stage-kind ladder-score --model gemma4e4b --selection F \\
             --certificate rung_probe.<cellA>.json --certificate rung_probe.<cellB>.json     # the wave's scoring
    meter.py --wave W --suffix S --unit score:<cellA> --stage-kind ladder-score …          # ONE cell's scoring
    meter.py --decision "ladder fired: gemma4e4b_it/bigcodebench_200" --ceiling-h 26.5
    meter.py --decision "GPU canaries run" --ceiling-h 21.0 \\
             --user-ruling "<who, when, what>: run the GPU canaries"   # a raise by explicit ruling
    meter.py --report

Ledger: ``~/.local/state/bcb-bfcl-200-widen/gpu_ledger.jsonl`` (``--ledger`` to
override); one JSON object per line, ``{at, kind: forecast|actual|decision, stage,
wave, suffix, unit, stage_kind, attempt, hours, ceiling_h, note, inputs,
supersedes, fired_legs}``; a decision recorded by the rule also carries
``user_ruling`` (the cited string, verbatim).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tinycap import rung_probe as RP   # noqa: E402

LEDGER_DEFAULT = "~/.local/state/bcb-bfcl-200-widen/gpu_ledger.jsonl"
#: the spec's Constraints: ≤ 25 H200-h for smokes + all runnable cells
CEILING_H = 25.0
#: "+ ≤ 1.5 H200-h per leg only if the truncation rule fires" — entered as a recorded decision naming the leg
LADDER_LEG_ALLOWANCE_H = 1.5
#: A resource-rate estimate, not billing (H200 ≈ $5/h).
H200_USD_PER_H = 5.0
#: steered ≈ 11 tok/s/seq, unsteered ≈ 34 tok/s/seq (tb-pilot measurements; the forecast inputs)
TOK_S_STEERED = 11.0
TOK_S_UNSTEERED = 34.0
#: One standup ≈ 4 m 51 s – 5 m 45 s (tb-pilot); 5 m 30 s as the forecast input.
STANDUP_S = 330.0
BASE_CELL = "none"
KINDS = ("forecast", "actual", "decision")
STAGE_KINDS = ("smoke", "wave", "ladder-probe", "ladder-reprobe", "ladder-score", "budget", "parser-probe", "other")
LADDER_PROBE_KIND, LADDER_REPROBE_KIND, LADDER_SCORE_KIND = "ladder-probe", "ladder-reprobe", "ladder-score"
#: a ``ladder-score`` reservation's unit: the wave's whole scoring, or ONE cell's (``score:<cell>``,
#: what a per-cell scoring launch claims — two cells' launches are two claims, never one)
SCORE_UNIT = "score"
SCORE_CELL_PREFIX = "score:"
PROBE_UNIT_PREFIX = "probe:"
#: the note prefix of a RELEASED claim (an actual of 0.0 h: the admitted launch never ran)
RELEASED_NOTE_PREFIX = "released:"
RELEASE_DEFAULT_WHY = "the claimed launch did not run"
LEG_ATOMS = ("bigcodebench_200", "bfcl_200", "bigcodebench_full", "bfcl_full")   # capeval-full-tasks: the full siblings
FIRED_LEG_RE = re.compile(r"ladder fired:\s*([^\n]+)", re.IGNORECASE)
REFUSAL_OPTIONS = (
    "(i) raise the ceiling — the user's call, and ONLY for a fired truncation leg (≤ 1.5 h each): record it with the leg named "
    "(`--decision \"ladder fired: <wave>/<leg>\" --ceiling-h <X>`), or by the user's explicit ruling, cited "
    "(`--decision <why> --ceiling-h <X> --user-ruling \"<who, when, what>\"`), then re-run this check;",
    "(ii) cut optional spend — no re-smokes, no `--inspect-limit`-free reruns — record the actuals that free margin "
    "(`--record --wave … --suffix … --unit … --actual-h …`), then re-run this check;",
    "(iii) stop the lane as INCOMPLETE — the un-run cells published by name as not run (never filled), acceptance check 5 stated.",
)


class MeterError(RuntimeError):
    """A refusal of the meter: a malformed ledger, a forecast that cannot be
    computed, or a flag combination that would assume rather than measure."""


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# the ledger                                                                   #
# --------------------------------------------------------------------------- #


def reservation_id(wave: str, suffix: str, unit: str) -> str:
    """The identity string a ledger entry carries as its ``stage``: ``<wave>/<suffix>/<unit>``."""
    for name, v in (("wave", wave), ("suffix", suffix), ("unit", unit)):
        if not isinstance(v, str) or not v.strip() or "/" in v.strip() and name != "unit":
            raise MeterError(f"reservation {name} {v!r} must be a non-empty string without '/'")
    return f"{wave.strip()}/{suffix.strip()}/{unit.strip()}"


def read_ledger(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError as exc:
            raise MeterError(f"{p}:{i}: not JSON: {exc}") from exc
        if not isinstance(e, Mapping) or e.get("kind") not in KINDS:
            raise MeterError(f"{p}:{i}: entry is not a ledger entry (kind ∈ {list(KINDS)})")
        out.append(dict(e))
    return out


def append_entry(path: Union[str, Path], entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Append one entry (the caller holds the lock for an admission). A
    reservation entry names ``wave`` / ``suffix`` / ``unit`` and derives its
    ``stage`` from them; a legacy caller may pass ``stage`` alone."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    e = {"at": _now_iso(), "kind": None, "stage": None, "wave": None, "suffix": None, "unit": None, "stage_kind": None,
         "attempt": None, "hours": None, "ceiling_h": None, "note": None, "inputs": None, "supersedes": None, "fired_legs": None}
    e.update(entry)
    if e["kind"] not in KINDS:
        raise MeterError(f"kind {e['kind']!r} not in {list(KINDS)}")
    if e["kind"] != "decision":
        ident = {k: e[k] for k in ("wave", "suffix", "unit")}
        if any(v is not None for v in ident.values()):
            missing = [k for k, v in ident.items() if v is None]
            if missing:                     # never `str(None)` → a "None/<suffix>/<unit>" stage
                raise MeterError(f"a partially specified reservation identity {ident} — {missing} missing; a reservation "
                                 "names wave, suffix AND unit (or, a legacy entry, a stage name alone)")
            rid = reservation_id(str(e["wave"]), str(e["suffix"]), str(e["unit"]))
            if e["stage"] is not None and e["stage"] != rid:
                raise MeterError(f"stage {e['stage']!r} != the reservation identity {rid!r}")
            e["stage"] = rid
        if not isinstance(e["stage"], str) or not e["stage"]:
            raise MeterError("a forecast / actual entry needs a reservation identity (wave, suffix, unit) or a legacy stage name")
        if e["stage_kind"] is not None and e["stage_kind"] not in STAGE_KINDS:
            raise MeterError(f"stage_kind {e['stage_kind']!r} not in {list(STAGE_KINDS)}")
        if e["attempt"] is None:
            e["attempt"] = 1
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, sort_keys=True, allow_nan=False) + "\n")
    return e


@contextlib.contextmanager
def ledger_lock(path: Union[str, Path]) -> Iterator[None]:
    """``fcntl.flock`` on ``<ledger>.lock`` around an admission transaction
    (read → check → append), so two processes cannot both read a ledger that
    admits them and both append."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_name(p.name + ".lock")
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _attempt_of(e: Mapping[str, Any]) -> int:
    a = e.get("attempt", 1)
    a = 1 if a is None else a
    if not isinstance(a, int) or isinstance(a, bool) or a < 1:
        raise MeterError(f"stage {e.get('stage')!r}: attempt {a!r} is not a positive int")
    return a


def _counted_hours(a: Mapping[str, Any]) -> float:
    if a["actual"] is not None:
        return float(a["actual"])
    return float(a["forecast"] or 0.0) if a.get("superseded_by") is None else 0.0


def _is_release(e: Mapping[str, Any]) -> bool:
    """An ``actual`` entry that RELEASES its claim: 0.0 h with the released note (the launch never ran)."""
    return e.get("kind") == "actual" and str(e.get("note") or "").startswith(RELEASED_NOTE_PREFIX)


def stage_table(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per reservation identity (the entry's ``stage``: ``<wave>/<suffix>/<unit>``,
    or a legacy stage name): every ATTEMPT's latest forecast and latest actual
    (``released`` when the actual is a release), the latest attempt's pair,
    ``counted`` = Σ over attempts of (actual if recorded, else the forecast unless
    SUPERSEDED — a structured ``supersedes`` reference on the superseding forecast
    entry, honoured on every reload; a RELEASED superseding claim supersedes
    nothing, so the reservations it had replaced count again), and the identity's
    ``wave`` / ``suffix`` / ``unit`` / ``stage_kind`` (the first non-null value
    seen; a reservation's kind never changes)."""
    table: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        if e["kind"] == "decision":
            continue
        stage = e.get("stage")
        if not isinstance(stage, str) or not stage:
            raise MeterError(f"a {e['kind']} entry without a reservation identity: {e}")
        h = e.get("hours")
        if not isinstance(h, (int, float)) or isinstance(h, bool) or h < 0:
            raise MeterError(f"reservation {stage!r}: hours {h!r} is not a non-negative number")
        row = table.setdefault(stage, {"attempts": {}, "n_entries": 0, "wave": None, "suffix": None, "unit": None, "stage_kind": None})
        for k in ("wave", "suffix", "unit", "stage_kind"):
            if row[k] is None and e.get(k) is not None:
                row[k] = e[k]
            elif e.get(k) is not None and row[k] != e[k]:
                raise MeterError(f"reservation {stage!r}: {k} moved from {row[k]!r} to {e[k]!r} across entries")
        att = row["attempts"].setdefault(_attempt_of(e), {"forecast": None, "actual": None, "superseded_by": None, "released": False})
        att[e["kind"]] = float(h)
        if e["kind"] == "actual":
            if _is_release(e) and float(h) != 0.0:
                raise MeterError(f"reservation {stage!r}: a released claim records 0.0 h, not {h!r}")
            att["released"] = _is_release(e)
        row["n_entries"] += 1
    for e in entries:
        refs = e.get("supersedes") if e["kind"] == "forecast" else None
        if not refs:
            continue
        if not isinstance(refs, list):
            raise MeterError(f"reservation {e.get('stage')!r}: supersedes must be a list of {{stage, attempt}}, got {refs!r}")
        own = table[e["stage"]]["attempts"][_attempt_of(e)]
        if own.get("released"):          # a released scoring claim hands its probe reservations back
            continue
        for ref in refs:
            if not isinstance(ref, Mapping) or not isinstance(ref.get("stage"), str):
                raise MeterError(f"reservation {e.get('stage')!r}: malformed supersession reference {ref!r}")
            target = table.get(ref["stage"], {}).get("attempts", {}).get(_attempt_of(ref))
            if target is None:
                raise MeterError(f"reservation {e.get('stage')!r} supersedes {ref['stage']!r} attempt {ref.get('attempt')!r}, which the "
                                 "ledger does not hold")
            if target["actual"] is None:            # a recorded actual is spent hours and is never excluded
                target["superseded_by"] = {"stage": e["stage"], "attempt": _attempt_of(e)}
    for stage, row in table.items():
        latest = max(row["attempts"])
        row["latest_attempt"] = latest
        row["forecast"] = row["attempts"][latest]["forecast"]
        row["actual"] = row["attempts"][latest]["actual"]
        row["counted"] = sum(_counted_hours(a) for a in row["attempts"].values())
    return table


def committed_hours(table: Mapping[str, Mapping[str, Any]], *, exclude: Sequence[Tuple[str, int]] = (),
                    wave: Optional[str] = None) -> float:
    """Σ over identities and attempts of (actual if recorded, else forecast);
    ``exclude`` names OPEN reservations whose forecast the check at hand replaces;
    ``wave`` restricts the sum to one wave's reservations (a report grouping)."""
    total = 0.0
    excluded = set(exclude)
    for stage, row in table.items():
        if wave is not None and row.get("wave") != wave:
            continue
        for n, a in row["attempts"].items():
            if a["actual"] is not None:
                total += float(a["actual"])
            elif (stage, n) not in excluded:
                total += _counted_hours(a)
    return total


def hours_by_wave(table: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in table.values():
        w = row.get("wave") or "(legacy / no wave)"
        out[w] = out.get(w, 0.0) + float(row["counted"])
    return dict(sorted(out.items()))


def open_attempts(table: Mapping[str, Mapping[str, Any]], stage: str) -> List[Tuple[str, int]]:
    row = table.get(stage)
    if not row:
        return []
    return [(stage, n) for n, a in row["attempts"].items()
            if a["actual"] is None and a["forecast"] is not None and a.get("superseded_by") is None]


def next_attempt(table: Mapping[str, Mapping[str, Any]], stage: str) -> int:
    row = table.get(stage)
    if not row:
        return 1
    latest = row["latest_attempt"]
    a = row["attempts"][latest]
    closed = a["actual"] is not None or a.get("superseded_by") is not None
    return latest + 1 if closed else latest


# --------------------------------------------------------------------------- #
# the ceiling and the fired-leg decision rule                                  #
# --------------------------------------------------------------------------- #


def ceiling_decision(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The LATEST recorded ceiling decision (the entry that sets the ceiling in force), or None."""
    decisions = [e for e in entries if e["kind"] == "decision" and e.get("ceiling_h") is not None]
    return dict(decisions[-1]) if decisions else None


def ceiling_hours(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The ceiling in force: the LATEST recorded ceiling decision if any, else
    the spec envelope of 25.0 h. The ladder's per-leg allowance is never implied
    — it is a recorded decision (:func:`check_decision`)."""
    d = ceiling_decision(entries)
    if d is not None:
        c = d["ceiling_h"]
        if not isinstance(c, (int, float)) or isinstance(c, bool) or c <= 0:
            raise MeterError(f"decision entry carries ceiling_h {c!r}")
        return {"ceiling_h": float(c), "source": "decision", "note": d.get("note"), "at": d.get("at")}
    return {"ceiling_h": CEILING_H, "source": f"spec envelope ({CEILING_H:.1f} h)", "note": None, "at": None}


def parse_fired_legs(note: Optional[str], flags: Sequence[str] = ()) -> List[str]:
    """``<wave>/<leg>`` tokens from ``--fired-leg`` flags and from a note of the form
    ``ladder fired: <wave>/<leg>[, <wave>/<leg>…]``; each leg atom must be one of the
    lane's two legs (``inspect/<leg>`` is accepted and normalised)."""
    raw = list(flags)
    if note:
        m = FIRED_LEG_RE.search(note)
        if m:
            raw.extend(t.strip() for t in re.split(r"[,;]", m.group(1)) if t.strip())
    out: List[str] = []
    for tok in raw:
        parts = tok.strip().split("/")
        if len(parts) == 3 and parts[1] == "inspect":
            parts = [parts[0], parts[2]]
        if len(parts) != 2 or not parts[0] or parts[1] not in LEG_ATOMS:
            raise MeterError(f"fired leg {tok!r} is not <wave>/<leg> with leg in {list(LEG_ATOMS)}")
        leg = "/".join(parts)
        if leg not in out:
            out.append(leg)
    return out


def credited_legs(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    for e in entries:
        if e["kind"] == "decision":
            for leg in (e.get("fired_legs") or []):
                if leg not in out:
                    out.append(leg)
    return out


def user_ruling_text(user_ruling: Optional[str]) -> Optional[str]:
    """``--user-ruling`` as cited (verbatim), or None when absent; an empty or blank
    citation names no one and no ruling and is REFUSED."""
    if user_ruling is None:
        return None
    if not isinstance(user_ruling, str) or not user_ruling.strip():
        raise MeterError("--user-ruling must cite the ruling — who ruled, when, and what (a non-empty string); an empty citation is no ruling")
    return user_ruling


def check_decision(entries: Sequence[Mapping[str, Any]], *, ceiling_h: float, note: str, fired_legs: Sequence[str] = (),
                   user_ruling: Optional[str] = None) -> Dict[str, Any]:
    """A decision may LOWER or restate the ceiling freely; a RAISE above the
    ceiling in force must name at least one NEWLY fired truncation leg and stay
    within :data:`LADDER_LEG_ALLOWANCE_H` per newly named leg (the spec's clause:
    the ladder is admitted against a RAISED, recorded ceiling, never inside the
    25 h by decree) — OR cite an explicit ruling (``user_ruling``, a non-empty
    string): then the raise is admitted without a leg and without the per-leg
    cap, recorded with the ruling verbatim and no leg credited. A ruling and a
    fired leg never share one decision (two authorities on one line would leave
    a later reader unable to tell which granted the hours): name them in two."""
    if not isinstance(ceiling_h, (int, float)) or isinstance(ceiling_h, bool) or ceiling_h <= 0:
        raise MeterError(f"--ceiling-h {ceiling_h!r} must be positive")
    ruling = user_ruling_text(user_ruling)
    current = ceiling_hours(entries)
    legs = parse_fired_legs(note, fired_legs)
    already = credited_legs(entries)
    new_legs = [leg for leg in legs if leg not in already]
    raise_h = float(ceiling_h) - current["ceiling_h"]
    if ruling is not None and legs:
        raise MeterError(f"a decision cites ONE authority: the user's ruling or the fired leg(s) {legs}, never both on one line — "
                         "record the fired leg's raise and the ruling's raise as two decisions")
    if raise_h > 1e-9 and ruling is None:
        if not legs:
            raise MeterError(f"a ceiling raise ({current['ceiling_h']:.2f} → {ceiling_h:.2f} h) is admitted ONLY for a fired truncation leg, "
                             "none named: write the note as `ladder fired: <wave>/<leg>` (or pass --fired-leg <wave>/<leg>) — "
                             "or, when the user ruled it, cite the ruling with --user-ruling \"<who, when, what>\"")
        if not new_legs:
            raise MeterError(f"every named leg {legs} was already credited by an earlier decision ({already}); a leg's ≤ 1.5 h is granted once")
        allowed = LADDER_LEG_ALLOWANCE_H * len(new_legs)
        if raise_h > allowed + 1e-9:
            raise MeterError(f"the raise {raise_h:.2f} h exceeds {LADDER_LEG_ALLOWANCE_H:.1f} h × {len(new_legs)} newly fired leg(s) = {allowed:.2f} h")
    return {"ceiling_h": float(ceiling_h), "previous_h": current["ceiling_h"], "raise_h": raise_h, "fired_legs": legs, "new_legs": new_legs,
            "already_credited": already, "user_ruling": ruling}


def decision_verb(raise_h: float) -> str:
    """``raised`` / ``lowered`` / ``restated`` — what a decision did to the ceiling in force."""
    return "raised" if raise_h > 1e-9 else "lowered" if raise_h < -1e-9 else "restated"


# --------------------------------------------------------------------------- #
# the ladder forecast — model-aware, per fitted cell                          #
# --------------------------------------------------------------------------- #


def _ladder():
    name = "tinycap_ladder"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    return sys.modules[name]


#: the two serving engines a ladder is walked on (`rung_probe.SERVING_ENGINE_*`; restated for a tree without them)
SERVING_ENGINE_LENS = getattr(RP, "SERVING_ENGINE_LENS", "lens")
SERVING_ENGINE_NATIVE = getattr(RP, "SERVING_ENGINE_NATIVE", "lora_native")
SERVING_ENGINES = (SERVING_ENGINE_LENS, SERVING_ENGINE_NATIVE)


def ladder_spec(model_key: str, serving_engine: Optional[str] = None):
    """Slice B's `rung_probe.ladder_spec(model_key)` → ``LadderSpec(model_key, budget,
    window, seqs, identity_S)``; a tree without it refuses by name (the meter never
    invents a rung list), a provisional-budget model refuses through B's own check.

    ``serving_engine``: ``lora_native`` returns the ENGINE-SERVED ladder
    (`rung_probe.native_ladder_spec` of the table's: each requested rung through
    the LoRA kernel index ceiling, coalesced — gemma (15, 14, 12) at 32,768), the rungs a native
    wave's probes walk and its certificates name, so a native forecast and a native scoring admission
    are sized from the footprint the engine serves; ``lens`` / None is the table's ladder, byte-identical."""
    fn = getattr(RP, "ladder_spec", None)
    if fn is None:
        raise MeterError("rung_probe.ladder_spec is not on this tree (Slice B lands the model-aware LadderSpec); the ladder forecast "
                         "needs the MODEL's rung list and identity S — pass --model once it lands")
    if serving_engine not in (None, *SERVING_ENGINES):
        raise MeterError(f"serving engine {serving_engine!r} is not one of {list(SERVING_ENGINES)}")
    try:
        spec = fn(model_key)
    except Exception as e:      # noqa: BLE001 — B's refusal (unknown / provisional model), named
        raise MeterError(f"ladder_spec({model_key!r}) refused: {type(e).__name__}: {e}") from e
    if serving_engine == SERVING_ENGINE_NATIVE:
        native_fn = getattr(RP, "native_ladder_spec", None)
        if native_fn is None:
            raise MeterError("rung_probe.native_ladder_spec is not on this tree (native-serving-default review r2 lands the "
                             f"engine-served ladder); a {SERVING_ENGINE_NATIVE} ladder cannot be sized here")
        try:
            spec = native_fn(spec)
        except Exception as e:  # noqa: BLE001 — the native sizing's refusal (no pinned width, every rung below the floor), named
            raise MeterError(f"native_ladder_spec({model_key!r}) refused: {type(e).__name__}: {e}") from e
    seqs = tuple(int(s) for s in getattr(spec, "seqs"))
    if not seqs or list(seqs) != sorted(set(seqs), reverse=True):
        raise MeterError(f"ladder_spec({model_key!r}).seqs {seqs!r} is not a strictly descending rung list")
    return spec


def load_selection_counts(path: Union[str, Path]) -> Dict[str, Any]:
    """``n_selected`` and ``cap_to`` per FIRED task from `ladder_selection.json`
    through the contracts' ONE reader (`tinycap_ladder.load_selection`)."""
    Lm = _ladder()
    try:
        sel = Lm.load_selection(path)
    except Lm.LadderError as e:
        raise MeterError(f"{path}: {e}") from e
    tasks = {task: {"n_selected": int(block["n_selected"]), "cap_to": int(block["cap_to"])} for task, block in sel["selection"].items()}
    return {"tasks": tasks, "cells": [str(c) for c in sel["cells"]], "fired_any": bool(tasks),
            "selection_sha256": sel["selection_sha256"], "selection": {t: dict(b) for t, b in sel["selection"].items()}}


def ladder_forecast(counts: Mapping[str, Any], *, seqs: int, fitted_cells: Sequence[str], base_cell: str = BASE_CELL,
                    tok_s_steered: float = TOK_S_STEERED, tok_s_unsteered: float = TOK_S_UNSTEERED, standup_s: float = STANDUP_S,
                    probe_rungs_by_cell: Optional[Mapping[str, int]] = None, probe_seconds_by_cell: Optional[Mapping[str, float]] = None,
                    reprobe_cells: Sequence[str] = (), scoring_cells: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """probe (one walk per FITTED cell: ``probe_rungs × standup_s`` unless a measured
    ``probe_seconds`` is given) + re-probe (one standup per cell in ``reprobe_cells``)
    + scoring (per cell ``standup_s + Σ_task ceil(n_selected / S) × cap_to / tok_s``;
    every cell of the selection, ``scoring_cells`` — ONE cell's scoring launch —, or
    ``[]`` for a probe reservation that scores nothing), in hours with the breakdown.
    Nothing bounds the item count."""
    if not isinstance(seqs, int) or isinstance(seqs, bool) or seqs < 1:
        raise MeterError(f"S must be a positive int (a rung of the ladder), got {seqs!r}")
    if not counts["tasks"]:
        raise MeterError("the selection fired no leg — there is no ladder to forecast")
    scoring = list(counts["cells"]) if scoring_cells is None else list(scoring_cells)     # [] = no scoring (a probe reservation)
    unknown = [c for c in list(fitted_cells) + list(reprobe_cells) + scoring if c not in counts["cells"]]
    if unknown:
        raise MeterError(f"cells {unknown} are not this ladder's cells {list(counts['cells'])}")
    if not scoring and not fitted_cells and not reprobe_cells:
        raise MeterError("an empty ladder forecast: no probe, no re-probe, no scoring cell")
    if base_cell in fitted_cells:
        raise MeterError(f"the base cell {base_cell!r} is object-free and needs no probe")
    probes: Dict[str, Any] = {}
    probe_s = 0.0
    for cell in fitted_cells:
        secs = (probe_seconds_by_cell or {}).get(cell)
        rungs = int((probe_rungs_by_cell or {}).get(cell, 1))
        if secs is None:
            secs = rungs * standup_s
        if secs < 0 or rungs < 0:
            raise MeterError(f"probe of {cell}: negative rungs / seconds")
        probes[cell] = {"rungs": rungs, "seconds": round(float(secs), 1), "measured": (probe_seconds_by_cell or {}).get(cell) is not None}
        probe_s += float(secs)
    reprobe_s = len(reprobe_cells) * standup_s
    cells: Dict[str, Any] = {}
    scoring_s = 0.0
    for cell in scoring:
        tok_s = tok_s_unsteered if cell == base_cell else tok_s_steered
        per_task = {}
        cell_s = standup_s
        for task, c in counts["tasks"].items():
            waves = math.ceil(c["n_selected"] / seqs)
            wave_s = c["cap_to"] / tok_s
            per_task[task] = {"n_selected": c["n_selected"], "waves": waves, "wave_s": round(wave_s, 1), "seconds": round(waves * wave_s, 1)}
            cell_s += waves * wave_s
        cells[cell] = {"steered": cell != base_cell, "tok_s": tok_s, "standup_s": standup_s, "seconds": round(cell_s, 1), "tasks": per_task}
        scoring_s += cell_s
    total_s = probe_s + reprobe_s + scoring_s
    return {"hours": total_s / 3600.0, "seconds": round(total_s, 1), "seqs": seqs, "probes": probes, "probe_seconds": round(probe_s, 1),
            "reprobe_cells": list(reprobe_cells), "reprobe_seconds": round(reprobe_s, 1), "scoring_hours": scoring_s / 3600.0, "cells": cells,
            "scoring_cells": scoring,
            "basis": "one probe walk per fitted cell + one re-probe standup per cell whose accepted rung differs from the common rung + "
                     "scoring: per cell one standup + Σ_task ceil(n_selected / S) × cap_to / tok_s(cell)"}


def load_certificate(path: Union[str, Path], *, selection: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    Lm = _ladder()
    try:
        return Lm.load_certificate(path, selection=selection)
    except Lm.LadderError as e:
        raise MeterError(str(e)) from e


# --------------------------------------------------------------------------- #
# the check and the admission transaction                                      #
# --------------------------------------------------------------------------- #


def check(entries: Sequence[Mapping[str, Any]], *, stage: str, forecast_h: float, supersedes: Sequence[str] = ()) -> Dict[str, Any]:
    """The admission rule. An identity whose latest attempt is OPEN (forecast, no
    actual, not superseded) is CLAIMED by a launch the ledger has not heard back
    from — the check refuses (``ok`` False, ``claimed`` True) without touching the
    ceiling: a second launch never rides the first's reservation.
    Otherwise ``Σ committed of every attempt + forecast_h ≤ ceiling`` opens the NEXT
    attempt (the claim token); the open reservations of any identity this one
    ``supersedes`` are excluded (their forecast is replaced by this one); the spent
    hours of earlier attempts stay counted."""
    if not isinstance(forecast_h, (int, float)) or isinstance(forecast_h, bool) or forecast_h < 0:
        raise MeterError(f"forecast hours {forecast_h!r} is not a non-negative number")
    table = stage_table(entries)
    ceil = ceiling_hours(entries)
    held = open_attempts(table, stage)
    base = {"stage": stage, "forecast_h": float(forecast_h), "ceiling_h": ceil["ceiling_h"], "ceiling_source": ceil["source"],
            "stages": table}
    if held:
        n = held[-1][1]
        committed = committed_hours(table)
        return {**base, "attempt": n, "committed_h": committed, "total_h": committed, "remaining_h": ceil["ceiling_h"] - committed,
                "ok": False, "claimed": True, "excluded": [], "usd_estimate_total": committed * H200_USD_PER_H,
                "reason": (f"reservation {stage!r} attempt {n} is CLAIMED by a launch whose actual is not recorded "
                           f"(forecast {table[stage]['attempts'][n]['forecast']:.2f} h)")}
    attempt = next_attempt(table, stage)
    exclude: List[Tuple[str, int]] = []
    for other in supersedes:
        exclude.extend(open_attempts(table, other))
    committed = committed_hours(table, exclude=exclude)
    total = committed + float(forecast_h)
    return {**base, "attempt": attempt, "committed_h": committed, "total_h": total, "remaining_h": ceil["ceiling_h"] - total,
            "ok": total <= ceil["ceiling_h"] + 1e-9, "claimed": False, "excluded": exclude, "usd_estimate_total": total * H200_USD_PER_H}


def admit(ledger: Union[str, Path], *, wave: str, suffix: str, unit: str, stage_kind: str, forecast_h: Optional[float] = None,
          inputs: Optional[Mapping[str, Any]] = None, supersedes: Sequence[str] = (), note: Optional[str] = None,
          compute: Optional[Callable[[Sequence[Mapping[str, Any]]], Tuple[float, Dict[str, Any], Sequence[str]]]] = None
          ) -> Dict[str, Any]:
    """The admission TRANSACTION under the ledger lock: read → (``compute`` the
    forecast from the entries just read: the ladder's supersession decisions are
    made on the SAME ledger the check runs on, never one read earlier) → check →
    append the forecast with its structured supersession references. Returns the
    check plus ``inputs`` (``ok`` False: nothing appended; the caller prints the
    refusal — ``claimed`` names an open claim, else the ceiling)."""
    if stage_kind not in STAGE_KINDS:
        raise MeterError(f"--stage-kind {stage_kind!r} not in {list(STAGE_KINDS)}")
    if (forecast_h is None) == (compute is None):
        raise MeterError("admit: exactly one of forecast_h / compute")
    stage = reservation_id(wave, suffix, unit)
    with ledger_lock(ledger):
        entries = read_ledger(ledger)
        if compute is not None:
            forecast_h, computed, supersedes = compute(entries)
            inputs = dict(computed)
        c = check(entries, stage=stage, forecast_h=float(forecast_h), supersedes=supersedes)
        c["inputs"] = dict(inputs) if inputs else None
        if c["ok"]:
            superseded = [{"stage": st, "attempt": n} for (st, n) in c["excluded"] if st != stage]
            append_entry(ledger, {"kind": "forecast", "wave": wave, "suffix": suffix, "unit": unit, "stage_kind": stage_kind,
                                  "attempt": c["attempt"], "hours": float(forecast_h), "inputs": (dict(inputs) if inputs else None),
                                  "supersedes": superseded or None,
                                  "note": (note or f"ceiling {c['ceiling_h']:.2f} h ({c['ceiling_source']})")
                                          + (f"; supersedes {list(supersedes)}" if supersedes else "")})
    return c


def release(ledger: Union[str, Path], *, wave: str, suffix: str, unit: str, why: str = RELEASE_DEFAULT_WHY) -> Dict[str, Any]:
    """Close an OPEN claim whose launch never ran: an ``actual`` of 0.0 h whose note
    says so (under the lock; a claim with no open attempt refuses — a launch that ran
    records its hours with ``--record`` instead)."""
    stage = reservation_id(wave, suffix, unit)
    with ledger_lock(ledger):
        entries = read_ledger(ledger)
        table = stage_table(entries)
        held = open_attempts(table, stage)
        if not held:
            raise MeterError(f"nothing to release: reservation {stage!r} has no open claim "
                             f"({'not in the ledger' if stage not in table else 'its latest attempt is closed'})")
        n = held[-1][1]
        e = append_entry(ledger, {"kind": "actual", "wave": wave, "suffix": suffix, "unit": unit, "stage_kind": table[stage].get("stage_kind"),
                                  "attempt": n, "hours": 0.0, "note": f"{RELEASED_NOTE_PREFIX} {why}"})
    return {"stage": stage, "attempt": n, "released_forecast_h": table[stage]["attempts"][n]["forecast"], "entry": e}


def refusal_text(c: Mapping[str, Any]) -> str:
    if c.get("claimed"):
        return "\n".join([
            f"REFUSED launch of {c['stage']!r}: {c['reason']} — a second launch never rides an earlier launch's reservation.",
            "Close the open claim first, then re-run this admission (it opens attempt "
            f"{c['attempt'] + 1}):",
            f"  record the launch's hours if it ran (`--record --wave … --suffix … --unit … --actual-h …`), or",
            f"  release it if it never ran (`--release [why] --wave … --suffix … --unit …`)."])
    lines = [f"REFUSED launch of {c['stage']!r} (attempt {c['attempt']}): Σ committed {c['committed_h']:.2f} h + forecast "
             f"{c['forecast_h']:.2f} h = {c['total_h']:.2f} H200-h > ceiling {c['ceiling_h']:.2f} h "
             f"({c['ceiling_source']}); over by {-c['remaining_h']:.2f} h ≈ ${-c['remaining_h'] * H200_USD_PER_H:.0f}.",
             "A budget decision must be RECORDED before this stage runs (plan §S6, §S10; the spec's Constraints) — one of:"]
    lines.extend("  " + o for o in REFUSAL_OPTIONS)
    return "\n".join(lines)


def report_text(entries: Sequence[Mapping[str, Any]]) -> str:
    table = stage_table(entries)
    ceil = ceiling_hours(entries)
    committed = committed_hours(table)
    lines = ["| wave | suffix | unit | kind | attempt | forecast h | actual h | counted h |", "|---|---|---|---|---|---|---|---|"]
    for stage, row in sorted(table.items(), key=lambda kv: (str(kv[1].get("wave") or ""), kv[0])):
        for n, a in sorted(row["attempts"].items()):
            if a["actual"] is None and a.get("superseded_by"):
                counted = f"superseded (by {a['superseded_by']['stage']} #{a['superseded_by']['attempt']})"
            elif a.get("released"):
                counted = "released (never ran)"
            else:
                counted = _h(a["actual"] if a["actual"] is not None else a["forecast"])
            lines.append(f"| {row.get('wave') or '—'} | {row.get('suffix') or '—'} | {row.get('unit') or stage} | {row.get('stage_kind') or '—'} "
                         f"| {n} | {_h(a['forecast'])} | {_h(a['actual'])} | {counted} |")
    lines.append(f"| **Σ** | | | | | | | **{committed:.2f}** |")
    lines.append("")
    lines.append("by wave: " + "; ".join(f"{w} {h:.2f} h" for w, h in hours_by_wave(table).items()))
    deciding = ceiling_decision(entries) or {}
    lines.append(f"ceiling {ceil['ceiling_h']:.2f} H200-h ({ceil['source']}" + (f": {ceil['note']}" if ceil.get("note") else "")
                 + (f"; the user's ruling: {deciding['user_ruling']}" if deciding.get("user_ruling") else "")
                 + f"); remaining {ceil['ceiling_h'] - committed:.2f} h; ≈ ${committed * H200_USD_PER_H:.0f} of ≈ ${ceil['ceiling_h'] * H200_USD_PER_H:.0f} "
                 f"(resource-rate estimate at ${H200_USD_PER_H:.0f}/H200-h, not billing)")
    for d in (e for e in entries if e["kind"] == "decision"):
        lines.append(f"decision {d.get('at')}: ceiling_h {d.get('ceiling_h')} — {d.get('note')}"
                     + (f" (fired legs {d['fired_legs']})" if d.get("fired_legs") else "")
                     + (f" (the user's ruling: {d['user_ruling']})" if d.get("user_ruling") else ""))
    return "\n".join(lines)


def _h(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", default=LEDGER_DEFAULT)
    p.add_argument("--wave", default=None, help="the reservation's wave (roster wave name)")
    p.add_argument("--suffix", default=None, help="the reservation's store suffix (bcbw-<wave>-<sha7>[-tag])")
    p.add_argument("--unit", default=None, help="the reservation's unit: a cell id, probe:<cell>, reprobe:<cell>, budget:<model>, smoke, score")
    p.add_argument("--stage-kind", default=None, choices=STAGE_KINDS, help="the stage KIND (separate from the identity)")
    p.add_argument("--forecast-h", type=float, default=None, help="this reservation's H200-hour forecast (refused at the ceiling)")
    p.add_argument("--model", default=None, help=f"model key of the wave's row → rung_probe.ladder_spec (the {LADDER_PROBE_KIND} / "
                                                 f"{LADDER_SCORE_KIND} forecasts take S and the rung list from it)")
    p.add_argument("--selection", default=None, help="ladder_selection.json — the ladder forecast is COMPUTED from its n_selected per fired task")
    p.add_argument("--certificate", action="append", default=None, help=f"with {LADDER_SCORE_KIND}: a per-cell rung_probe.<cell>.json (repeat)")
    p.add_argument("--serving-engine", default=None, choices=list(SERVING_ENGINES),
                   help=f"the engine the ladder is walked on (default {SERVING_ENGINE_LENS}): a {LADDER_PROBE_KIND} forecast is sized "
                        f"from that engine's SERVED rungs; a {LADDER_SCORE_KIND} admission reads the engine off the certificates' "
                        "ladder blocks and, when given, holds this flag to it")
    p.add_argument("--fitted-cell", action="append", default=None, help=f"with {LADDER_PROBE_KIND}: the fitted cells whose probes this reservation "
                                                                          "budgets (default: the unit's probe:<cell>)")
    p.add_argument("--reprobe-cell", action="append", default=None, help=f"with {LADDER_SCORE_KIND}: cells re-probed at the common rung")
    p.add_argument("--tok-s-steered", type=float, default=TOK_S_STEERED)
    p.add_argument("--tok-s-unsteered", type=float, default=TOK_S_UNSTEERED)
    p.add_argument("--standup-s", type=float, default=STANDUP_S)
    p.add_argument("--base-cell", default=BASE_CELL)
    p.add_argument("--record", action="store_true", help="with the identity and --actual-h: record the reservation's actual hours")
    p.add_argument("--release", nargs="?", const=RELEASE_DEFAULT_WHY, default=None, metavar="WHY",
                   help="with the identity: close an OPEN claim whose launch never ran (an actual of 0.0 h, noted as released)")
    p.add_argument("--actual-h", type=float, default=None)
    p.add_argument("--attempt", type=int, default=None, help="with --record: the attempt measured (default: the latest)")
    p.add_argument("--decision", default=None, help="record a budget decision (with --ceiling-h); a RAISE names fired legs")
    p.add_argument("--ceiling-h", type=float, default=None)
    p.add_argument("--fired-leg", action="append", default=None, help="<wave>/<leg> the truncation rule fired on (≤ 1.5 h each, once)")
    p.add_argument("--user-ruling", default=None, metavar="WHO_WHEN_WHAT",
                   help="with --decision / --ceiling-h: the user's explicit ruling, cited (who, when, what; non-empty) — admits a raise "
                        "without a fired leg and without the per-leg cap; recorded verbatim on the entry as user_ruling")
    p.add_argument("--report", action="store_true")
    p.add_argument("--json", action="store_true", help="print the check / report as JSON too")
    return p


def _identity_args(a) -> Tuple[str, str, str]:
    if not (a.wave and a.suffix and a.unit):
        raise MeterError("a reservation needs --wave, --suffix and --unit (the identity P2-7 asked for; a stage name alone collides across waves)")
    return a.wave, a.suffix, a.unit


def _ladder_inputs(a, entries: Sequence[Mapping[str, Any]], *, unit: str) -> Tuple[float, Dict[str, Any], Tuple[str, ...]]:
    """The computed ladder forecast for a ``ladder-probe`` reservation (worst-case
    rung of the MODEL's ladder, a full walk per fitted cell) or a ``ladder-score``
    admission (the common S = min accepted over the per-cell certificates, each
    validated by the launcher's gate; probes recorded as actuals are not re-counted)."""
    if not a.model:
        raise MeterError(f"--stage-kind {a.stage_kind} needs --model: the rung list and the identity S come from the model's LadderSpec")
    # the ladder on the ENGINE — the flag's (default lens) for the pre-probe forecast; for a scoring
    # admission the engine the CERTIFICATES were walked on (read below, the flag held to it)
    engine = a.serving_engine or SERVING_ENGINE_LENS
    spec = ladder_spec(a.model, engine)
    seqs_list = tuple(int(s) for s in spec.seqs)
    counts = load_selection_counts(a.selection)
    if not counts["fired_any"]:
        raise MeterError(f"{a.selection}: no leg fired — nothing to launch")
    common = dict(base_cell=a.base_cell, tok_s_steered=a.tok_s_steered, tok_s_unsteered=a.tok_s_unsteered, standup_s=a.standup_s)
    if a.stage_kind == LADDER_PROBE_KIND:
        if a.certificate:
            raise MeterError(f"a certificate belongs to --stage-kind {LADDER_SCORE_KIND}, not the pre-probe reservation")
        fitted = list(a.fitted_cell or ([unit.split(":", 1)[1]] if unit.startswith("probe:") else []))
        if not fitted:
            raise MeterError(f"--stage-kind {LADDER_PROBE_KIND}: name the fitted cell(s) (--unit probe:<cell> or --fitted-cell)")
        # the probe reservation is the INDEPENDENT record of every probe standup the certify step may spend on the cell
        #: the worst-case full walk PLUS one re-probe standup at the wave's common footprint (a cell
        # whose walk accepted above the common S is re-probed there); `--reprobe-cell` names the cells to budget a
        # re-probe for (default: every fitted cell of this reservation). Its ACTUAL, recorded after the certify step,
        # replaces this forecast — nothing else ever does.
        reprobe = list(a.reprobe_cell) if a.reprobe_cell else list(fitted)
        fc = ladder_forecast(counts, seqs=min(seqs_list), fitted_cells=fitted, probe_rungs_by_cell={c: len(seqs_list) for c in fitted},
                             reprobe_cells=reprobe, scoring_cells=[], **common)
        fc["admission"] = (f"pre-probe (worst case S = {min(seqs_list)} of {list(seqs_list)}; a full walk + one re-probe standup per fitted "
                           "cell; no scoring — the scoring is its own claim)")
        fc["ladder"] = {"model_key": a.model, "seqs": list(seqs_list), "window": int(getattr(spec, "window", RP.LADDER_WINDOW)),
                        **_native_ladder_fields(spec, engine)}
        return fc["hours"], {"selection": str(Path(a.selection).expanduser()), "selection_sha256": counts["selection_sha256"], "forecast": fc}, ()
    if a.stage_kind == LADDER_SCORE_KIND:
        if not a.certificate:
            raise MeterError(f"--stage-kind {LADDER_SCORE_KIND} needs --certificate rung_probe.<cell>.json (one per fitted cell): the scoring "
                             "admission is recomputed from the CERTIFIED common S, never assumed")
        if a.reprobe_cell:
            raise MeterError(f"--reprobe-cell belongs to the {LADDER_PROBE_KIND} reservation: probe and re-probe spend are independent "
                             "records (probe:<cell>), never part of a scoring forecast")
        if unit == SCORE_UNIT:
            scoring_cells = None                                    # the wave's whole scoring
        elif unit.startswith(SCORE_CELL_PREFIX) and unit[len(SCORE_CELL_PREFIX):]:
            scoring_cells = [unit[len(SCORE_CELL_PREFIX):]]         # ONE cell's scoring launch (its own claim)
        else:
            raise MeterError(f"a {LADDER_SCORE_KIND} unit is {SCORE_UNIT!r} (the wave's scoring) or "
                             f"'{SCORE_CELL_PREFIX}<cell>' (one cell's), got {unit!r}")
        accepted: Dict[str, int] = {}
        reprobed: Dict[str, Optional[int]] = {}
        cert_engines: set = set()
        for path in a.certificate:
            cert = load_certificate(path, selection=counts["selection"])
            if counts["selection_sha256"] != cert["selection_sha256"]:
                raise MeterError(f"{path}: selection_sha256 {str(cert['selection_sha256'])[:12]}… != the selection file's — another selection")
            cell = str(cert["probed_cell"])
            accepted[cell] = int(cert["accepted"]["max_num_seqs"])
            rf = cert.get("reprobed_from")
            reprobed[cell] = int(rf["accepted_S"]) if isinstance(rf, Mapping) else None
            cert_engines.add(str(_ladder().certificate_serving_engine(cert)))
        # the certificates are the authority on the engine they were walked on (their `ladder` block,
        # validated by the gate above); several engines are not one ladder, and a flag that disagrees refuses
        if len(cert_engines) != 1:
            raise MeterError(f"the certificates were walked on several serving engines {sorted(cert_engines)} — not one ladder")
        cert_engine = cert_engines.pop()
        if a.serving_engine is not None and a.serving_engine != cert_engine:
            raise MeterError(f"--serving-engine {a.serving_engine} but the certificates were walked on the {cert_engine} engine "
                             "(their ladder blocks) — the scoring admission is sized from the engine that certified the rung")
        if cert_engine != engine:
            engine, spec = cert_engine, ladder_spec(a.model, cert_engine)
            seqs_list = tuple(int(s) for s in spec.seqs)
        seqs = min(accepted.values())
        if seqs not in seqs_list:
            raise MeterError(f"the common S {seqs} is not on {a.model}'s ladder {list(seqs_list)}"
                             + (f" ({engine} engine: the served rungs)" if engine == SERVING_ENGINE_NATIVE else ""))
        # SCORING ONLY. The probe walks and re-probes the certificates attest were spent under the
        # `probe:<cell>` reservations — independent ledger records with their own forecasts and actuals; this forecast
        # never carries, supersedes or replaces them, so recording a scoring actual (or a re-probe that folds the first
        # walk's timing into `reprobed_from`) cannot make probe spend vanish. The probe reservations' state is RECORDED
        # here for the report only (read under the admission lock).
        table = stage_table(entries)
        state = {c: probe_reservation_state(table, reservation_id(a.wave, a.suffix, f"{PROBE_UNIT_PREFIX}{c}")) for c in accepted}
        fc = ladder_forecast(counts, seqs=seqs, fitted_cells=[], scoring_cells=scoring_cells, **common)
        fc["admission"] = ("certified common S (min over the per-cell certificates); scoring only — probe / re-probe spend lives in the "
                           "probe:<cell> reservations" + ("" if scoring_cells is None else f"; scoring ONE cell ({scoring_cells[0]})"))
        fc["accepted_by_cell"] = accepted
        fc["reprobed_from_S_by_cell"] = reprobed
        fc["probe_reservations"] = state
        fc["ladder"] = {"model_key": a.model, "seqs": list(seqs_list), **_native_ladder_fields(spec, engine)}
        return fc["hours"], {"selection": str(Path(a.selection).expanduser()), "selection_sha256": counts["selection_sha256"],
                             "certificates": [str(Path(p).expanduser()) for p in a.certificate], "forecast": fc}, ()
    raise MeterError(f"--selection is the ladder's input; --stage-kind must be {LADDER_PROBE_KIND!r} or {LADDER_SCORE_KIND!r}, got {a.stage_kind!r}")


def _native_ladder_fields(spec, engine: str) -> Dict[str, Any]:
    """The forecast's `ladder` block gains the engine and the requested -> served mapping under the
    native engine; a lens forecast's block is byte-identical (no key added)."""
    if engine != SERVING_ENGINE_NATIVE:
        return {}
    block = spec.block() if hasattr(spec, "block") else {}
    return {"serving_engine": engine, "requested_seqs": list(getattr(spec, "requested_seqs", ())),
            "served_rungs": list(block.get("served_rungs", []))}


def probe_reservation_state(table: Mapping[str, Mapping[str, Any]], stage: str) -> str:
    """The latest attempt of a probe reservation, for the record: ``open`` (claimed, its
    forecast counts until its actual is recorded), ``absent`` (never reserved — the operator
    should reserve and record it), ``released`` (the certify launch never ran), ``measured``
    (its actual is the spent hours) or ``superseded`` (a library caller superseded it)."""
    row = table.get(stage)
    if row is None:
        return "absent"
    latest = row["attempts"][row["latest_attempt"]]
    if latest["actual"] is not None:
        return "released" if latest.get("released") else "measured"
    if latest.get("superseded_by") is not None:
        return "superseded"
    return "open"


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    ledger = Path(a.ledger).expanduser()
    if a.decision is not None or a.ceiling_h is not None or a.fired_leg or a.user_ruling is not None:
        if a.decision is None or a.ceiling_h is None:
            raise MeterError("--decision and --ceiling-h go together: a ceiling is recorded with its reason")
        with ledger_lock(ledger):
            entries = read_ledger(ledger)
            d = check_decision(entries, ceiling_h=a.ceiling_h, note=a.decision, fired_legs=a.fired_leg or (), user_ruling=a.user_ruling)
            entry: Dict[str, Any] = {"kind": "decision", "ceiling_h": float(a.ceiling_h), "note": a.decision, "fired_legs": d["fired_legs"] or None}
            if d["user_ruling"] is not None:            # the key rides ONLY a decision that cites a ruling: every other entry's bytes are unchanged
                entry["user_ruling"] = d["user_ruling"]
            e = append_entry(ledger, entry)
        print(f"[meter] decision recorded: ceiling {a.ceiling_h:.2f} H200-h ({d['previous_h']:.2f} → {a.ceiling_h:.2f}"
              + (f"; fired legs {d['new_legs']}" if d["new_legs"] else "")
              + (f"; {decision_verb(d['raise_h'])} by the user's ruling: {d['user_ruling']}" if d["user_ruling"] is not None else "")
              + f") — {a.decision}")
        if a.json:
            print(json.dumps(e, sort_keys=True))
        return 0
    if a.report:
        entries = read_ledger(ledger)
        print(report_text(entries))
        if a.json:
            print(json.dumps({"stages": stage_table(entries), "ceiling": ceiling_hours(entries), "by_wave": hours_by_wave(stage_table(entries))},
                             sort_keys=True))
        return 0
    if a.release is not None:
        if a.record or a.actual_h is not None or a.forecast_h is not None:
            raise MeterError("--release closes an open claim on its own (no --record / --actual-h / --forecast-h)")
        wave, suffix, unit = _identity_args(a)
        r = release(ledger, wave=wave, suffix=suffix, unit=unit, why=str(a.release))
        entries = read_ledger(ledger)
        table = stage_table(entries)
        ceil = ceiling_hours(entries)
        print(f"[meter] {r['stage']} attempt {r['attempt']}: claim RELEASED ({a.release}); its forecast {_h(r['released_forecast_h'])} h no "
              f"longer counts; Σ counted {committed_hours(table):.2f} h of {ceil['ceiling_h']:.2f} h ({ceil['source']})")
        if a.json:
            print(json.dumps(r["entry"], sort_keys=True))
        return 0
    if a.record:
        if a.actual_h is None:
            raise MeterError("--record needs --actual-h")
        if a.actual_h < 0:
            raise MeterError(f"--actual-h {a.actual_h} < 0")
        wave, suffix, unit = _identity_args(a)
        stage = reservation_id(wave, suffix, unit)
        with ledger_lock(ledger):
            entries = read_ledger(ledger)
            table = stage_table(entries)
            attempt = a.attempt if a.attempt is not None else (table[stage]["latest_attempt"] if stage in table else 1)
            if attempt < 1:
                raise MeterError(f"--attempt {attempt} < 1")
            kind = a.stage_kind or table.get(stage, {}).get("stage_kind")
            append_entry(ledger, {"kind": "actual", "wave": wave, "suffix": suffix, "unit": unit, "stage_kind": kind, "attempt": attempt,
                                  "hours": float(a.actual_h)})
            entries = read_ledger(ledger)
        table = stage_table(entries)
        committed = committed_hours(table)
        ceil = ceiling_hours(entries)
        print(f"[meter] {stage} attempt {attempt}: actual {a.actual_h:.2f} h recorded; Σ counted {committed:.2f} h of {ceil['ceiling_h']:.2f} h "
              f"({ceil['source']}); wave {wave} {committed_hours(table, wave=wave):.2f} h")
        if committed > ceil["ceiling_h"] + 1e-9:
            print(f"[meter] WARNING: Σ actual-or-forecast {committed:.2f} h EXCEEDS the ceiling {ceil['ceiling_h']:.2f} h — record a decision "
                  "before any further launch")
        return 0
    if not (a.wave or a.suffix or a.unit or a.stage_kind or a.forecast_h is not None or a.selection):
        build_parser().print_help()
        return 2
    wave, suffix, unit = _identity_args(a)
    if not a.stage_kind:
        raise MeterError("a reservation needs --stage-kind (the stage KIND is recorded apart from the identity)")
    if a.selection is not None or a.stage_kind in (LADDER_PROBE_KIND, LADDER_SCORE_KIND):
        if a.selection is None:
            raise MeterError(f"--stage-kind {a.stage_kind} needs --selection: the ladder's cost is computed from the actual selection, never assumed")
        if a.forecast_h is not None:
            raise MeterError("--selection computes the forecast; do not also pass --forecast-h")
        # the ladder inputs (which probe reservations this admission supersedes, which are measured) are computed
        # from the ledger read UNDER the admission lock — never from an earlier, unlocked read
        c = admit(ledger, wave=wave, suffix=suffix, unit=unit, stage_kind=a.stage_kind,
                  compute=lambda entries: _ladder_inputs(a, entries, unit=unit))
        fc = c["inputs"]["forecast"]
        print(f"[meter] {a.stage_kind} {c['stage']} forecast ({fc['admission']}): "
              + "; ".join(f"{t.split('/')[-1]} n_selected={cnt['n_selected']}" for t, cnt in load_selection_counts(a.selection)["tasks"].items())
              + f"; S={fc['seqs']}; probes {fc['probe_seconds'] / 3600:.2f} h; re-probes {fc['reprobe_seconds'] / 3600:.2f} h; "
              + "; ".join(f"{cell} {v['seconds'] / 3600:.2f} h" for cell, v in fc["cells"].items()) + f" → {c['forecast_h']:.2f} H200-h")
    else:
        if a.certificate:
            raise MeterError(f"--certificate belongs to --stage-kind {LADDER_SCORE_KIND}")
        if a.forecast_h is None:
            raise MeterError("a reservation needs --forecast-h (or --selection for the ladder kinds)")
        c = admit(ledger, wave=wave, suffix=suffix, unit=unit, stage_kind=a.stage_kind, forecast_h=float(a.forecast_h))
    if a.json:
        print(json.dumps({k: v for k, v in c.items() if k != "stages"}, sort_keys=True))
    if not c["ok"]:
        print(refusal_text(c), file=sys.stderr)
        return 2
    print(f"[meter] {c['stage']} attempt {c['attempt']}: forecast {c['forecast_h']:.2f} h admitted (the claim of this launch) — Σ "
          f"{c['total_h']:.2f} of {c['ceiling_h']:.2f} H200-h ({c['ceiling_source']}); remaining {c['remaining_h']:.2f} h ≈ "
          f"${c['remaining_h'] * H200_USD_PER_H:.0f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MeterError as e:
        print(f"[meter] REFUSED: {e}", file=sys.stderr)
        sys.exit(3)
