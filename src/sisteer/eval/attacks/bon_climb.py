"""The canonical best-of-N instrument: the 512 store PLUS the 4,096 CLIMB of its cap-coherent
at-cap draws — and a NATIVE 4,096 pass for a lane that has no 512 store.

WHY THE CLIMB EXISTS: the 512 reading is a FLOOR, not a measurement. A draw cut off at the cap is
judged jailbroken far more often than one that terminated — pooled over the six full22 lanes,
**~19x** more often (per lane 4.3-20.8x) — so a cell with meaningful at-cap volume reads
LOW at 512 by construction. The climb regenerates each cap-coherent at-cap draw at 4,096 with the
SAME seed, the SAME SAMPLER THE STORE RECORDS and the same served target, and folds the result in beside
the 512 numbers as a counterfactual best-of-N.

THE CLIMB'S SAMPLER IS THE STORE'S, NEVER A CONSTANT (:data:`CLIMB_DECODE` = ``as_drawn``).
Best-of-N draws at each served model's VENDOR PRESET (the registry's
``<model>|bon|off|vendor_preset`` row, ``bon_served.bon_sampler``); every store drawn before
2026-09-17 was drawn at the grandfathered temperature 1.0. A climb re-draws at-cap slots of an
EXISTING store, so it re-draws them under exactly the sampler those rows record (``decode`` +
``bon_decode_preset`` on every current-format row; a legacy store — rows carrying none — was drawn
under the grandfathered row and is climbed under it BY NAME), read by :func:`store_sampler` and
stamped on every manifest row. A store whose rows disagree is refused (a mixture is never a store).
A NATIVE pass has no store and draws at the vendor preset, like any new store. The store and its
climb are therefore never at different samplers.

THE TWO MODES ARE A REAL BRANCH, not a convention (:func:`climb_mode`):

* :data:`MODE_CLIMB` — a 512 store exists: escalate its cap-coherent at-cap draws to 4,096.
* :data:`MODE_NATIVE` — NO 512 store exists (the Llama lane): run best-of-N natively at 4,096.
  Getting this wrong in the silent direction is the dangerous one — a lane with no 512 store that
  tries to climb finds nothing to escalate and would report ZERO ESCALATIONS as "clean" rather than
  "not applicable". :func:`escalation_summary` therefore reports ``status`` =
  :data:`STATUS_NOT_APPLICABLE` for a native lane and refuses to call it clean, and
  :func:`climb_manifest_rows` REFUSES to build an empty climb manifest from a missing store.

THE DELIBERATE RESIDUAL: loop-flagged draws are NOT regenerated. A repeating
draw simply repeats further at a bigger cap, so escalating it buys nothing and costs the same GPU
time as a coherent one. Those draws keep their 512 outcome, are counted
(``n_loop`` / ``loop_share``), and are a KNOWN, ACCEPTED component of the residual floor — a cell's
residual is never "nothing left at cap", it is "nothing left at cap except the loops we chose not
to climb". The secondary degeneracy classifier is a FLAG, never a reason to skip a draw: a draw it
alone marks IS climbed, carrying ``subclass="secondary:<reason>"``.

THE CLASSIFICATION RULE is the campaign's, imported from the tree (never a vendored copy):
``capeval.tiers.cap_ladder`` for the repetition rule (4-gram ``rep_ratio`` over the whole text OR
its last 200 words, strict > 0.5, under-8-words = 0) and ``eval.degeneracy.classify`` for the
secondary flag. :func:`classifier_fingerprint` digests exactly those definitions so a store can
attest WHICH rule classified it.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ...capeval.tiers import cap_ladder as _cl
from .. import degeneracy as _dg

#: the canonical best-of-N escalation cap; the climb re-draws the SAME slot (same goal, same augmented
#: variant, same per-draw seed) at the SAME SAMPLER its 512 draw records — :data:`CLIMB_DECODE`
#: ``as_drawn`` (the roster's ``bon_climb_decode``; every rendered config's ``bon.climb_decode``). Not a
#: temperature: the sampler is per served model (the vendor preset) or per store (the grandfathered T=1 the
#: existing stores were drawn under), read from the store by :func:`store_sampler`.
CLIMB_CAP = 4096
CLIMB_DECODE = "as_drawn"
#: the cap the 512 store was drawn at — the base of the climb, and `bon.max_new_tokens` on every config
BASE_CAP = 512

MODE_CLIMB = "climb"
MODE_NATIVE = "native_4096"
MODES = (MODE_CLIMB, MODE_NATIVE)

CLASS_LOOP = "loop"
CLASS_COHERENT = "cap_coherent"

#: escalation_summary().status — a NATIVE lane has no escalation to report; it is NOT "clean"
STATUS_CLIMBED = "climbed"
STATUS_NOT_APPLICABLE = "not_applicable_native_4096"
STATUS_NOTHING_TO_CLIMB = "nothing_to_climb"

#: the names whose definitions ARE the classification rule (fingerprinted below)
_RULE_NAMES = ("REP_RATIO_THRESHOLD", "REP_NGRAM", "REP_MIN_WORDS", "TAIL_WORDS",
               "rep_ratio", "tail_rep_ratio", "degeneracy")

BON_RESPONSES = "bon_responses.jsonl"
BON_RESPONSES_CLIMBED = "bon_responses_cap4096.jsonl"


class BonClimbError(RuntimeError):
    """A refusal in the climb's own wiring — loud, never a silent empty manifest."""


# --------------------------------------------------------------------------- the mode branch

#: the 512 store's COMPLETION marker: bon_served_run.py writes it beside the responses when the cell's draws are all in
#: (its end-of-run record). The predicate below keys on this EVENT — never on the responses file's size, row count or
#: mtime — so a store still being written is refused by name instead of read as "no store" by one reader and "climb" by
#: another (the u0 wave on 2026-09-30: the climb launched while the store unit was writing).
BON_STORE_MANIFEST = "bon_served_run.json"
#: bon_served_run.py writes this when it STARTS a cell (job id + utc), so an in-progress store can name its job
BON_STORE_STARTED = "bon_served_run.started.json"
STORE_ABSENT, STORE_IN_PROGRESS, STORE_COMPLETE, STORE_PARTIAL = "absent", "in_progress", "complete", "partial"
#: the store's DURABLE manifest (bon_served.write_bon_served_manifest): `units` = the loop's slot accounting
BON_STORE_MANIFEST_UNITS = "bon_served_manifest.json"
#: a typed no-fit row's `error` value (served_window.PROMPT_OVER_WINDOW): the slot is DECIDED — the prompt does not fit the window
NOFIT_ERROR = "prompt_over_window"


#: the over-window row KINDS: only `nofit_proven` is a DECIDED slot — the exact prompt length (from the tokenize
#: endpoint or the server's exact message) plus the minimum completion cannot fit the authoritative effective window. A floor-only
#: rejection whose /tokenize was unavailable, a rejection inconsistent with a prompt that fits, or a clamped retry rejected again are
#: recoverable failures: those slots stay UNDECIDED and a resume of the store job draws them.
NOFIT_PROVEN, UNRESOLVED_TOKENIZE, INCONSISTENT_REJECTION, RETRY_REJECTED, UNRESOLVED_OTHER = (
    "nofit_proven", "unresolved_tokenize", "inconsistent_rejection", "retry_rejected", "unresolved_other")
NOFIT_KINDS = (NOFIT_PROVEN, UNRESOLVED_TOKENIZE, INCONSISTENT_REJECTION, RETRY_REJECTED, UNRESOLVED_OTHER)
#: served_window.MIN_CLAMPED_COMPLETION — the fewest completion tokens a clamped retry may ask for (a prompt leaving less cannot fit)
MIN_CLAMPED_COMPLETION = 64


def classify_over_window(row: Mapping[str, Any], *, min_completion: int = MIN_CLAMPED_COMPLETION) -> str:
    """The kind of one typed prompt_over_window row (served_window.PromptOverWindow.record()): `nofit_proven` ONLY when the exact prompt
    length + the minimum completion provably exceed the authoritative effective window; every other shape is an undecided kind by name."""
    reason = str(row.get("window_reason") or row.get("reason") or "")
    src = row.get("prompt_tokens_source")
    pt = row.get("prompt_tokens"); win = row.get("effective_window") or row.get("server_window") or row.get("window")
    if reason == "floor_only_unresolvable" or not isinstance(pt, int) or src not in ("tokenize_endpoint", "error_message"):
        return UNRESOLVED_TOKENIZE
    if "inconsistent" in reason:
        return INCONSISTENT_REJECTION
    if "rejected again" in reason or "clamped retry" in reason:
        return RETRY_REJECTED
    if isinstance(win, int) and pt + min_completion > win:
        return NOFIT_PROVEN
    return UNRESOLVED_OTHER


def store_slot_classification(store_dir) -> Dict[str, Any]:
    """The store's over-window SLOTS re-derived from their per-slot records (bon_errors.jsonl), never from a count: a slot is a DECIDED
    no-fit when any of its rows is `nofit_proven`; a slot with only undecided-kind rows is undecided; an unparsable line is counted (it
    decides nothing). Returns {n_nofit_proven, n_undecided_over_window, n_unparsable_rows, by_kind, slots: {proven: […], undecided: […]}}."""
    p = Path(store_dir) / "bon_errors.jsonl"
    proven: Dict[Tuple[Any, Any, Any], dict] = {}; undecided: Dict[Tuple[Any, Any, Any], str] = {}
    by_kind: Dict[str, int] = {k: 0 for k in NOFIT_KINDS}; bad = 0
    if p.is_file():
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                bad += 1; continue
            if r.get("error") != NOFIT_ERROR:
                continue
            key = (r.get("goal_id"), r.get("sample_idx"), r.get("trial"))
            if not (isinstance(key[0], str) and key[0] and isinstance(key[1], int) and isinstance(key[2], int)):
                bad += 1; continue                       # a row whose slot key is torn decides nothing (the store's ledger was written to by two processes)
            kind = classify_over_window(r); by_kind[kind] += 1
            if kind == NOFIT_PROVEN:
                proven.setdefault(key, {"goal_id": key[0], "sample_idx": key[1], "trial": key[2], "error": NOFIT_ERROR, "kind": kind,
                                        "prompt_tokens": r.get("prompt_tokens"), "window": r.get("effective_window") or r.get("window")})
                undecided.pop(key, None)
            elif key not in proven:
                undecided[key] = kind
    return {"n_nofit_proven": len(proven), "n_undecided_over_window": len(undecided), "n_unparsable_rows": bad, "by_kind": by_kind,
            "slots": {"proven": sorted(proven.values(), key=lambda d: (str(d["goal_id"]), d["sample_idx"] or 0, d["trial"] or 0)),
                      "undecided": sorted([{"goal_id": k[0], "sample_idx": k[1], "trial": k[2], "kind": v} for k, v in undecided.items()],
                                          key=lambda d: (str(d["goal_id"]), d["sample_idx"] or 0, d["trial"] or 0))}}


def store_planned_gaps(store_dir, planned_keys) -> Dict[str, Any]:
    """The EXACT completeness check a reader with the plan's goal universe can make (the planner has the config's goals x n_max):
    every planned (goal_id, sample_idx, trial) must be COVERED (a validated response row) or a PROVEN no-fit; anything else is a gap by
    name. Foreign keys (rows for slots outside the plan — another writer's) are listed, never counted toward the plan."""
    planned = {tuple(k) for k in planned_keys}
    covered = set()
    p = Path(store_dir) / BON_RESPONSES
    if p.is_file():
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            covered.add((r.get("goal_id"), r.get("sample_idx"), r.get("trial", 0)))
    cls = store_slot_classification(store_dir)
    proven = {(d["goal_id"], d["sample_idx"], d["trial"]) for d in cls["slots"]["proven"]}
    undecided = {(d["goal_id"], d["sample_idx"], d["trial"]): d["kind"] for d in cls["slots"]["undecided"]}
    gaps = sorted(k for k in planned if k not in covered and k not in proven)
    return {"n_planned": len(planned), "n_covered": len(planned & covered), "n_nofit_proven": len(planned & proven),
            "gaps": [{"goal_id": k[0], "sample_idx": k[1], "trial": k[2], "kind": undecided.get(k, "no_row")} for k in gaps],
            "foreign_keys": sorted(k for k in (covered | proven) - planned), "n_unparsable_rows": cls["n_unparsable_rows"]}


def store_completion(units: Optional[Mapping[str, Any]], classification: Optional[Mapping[str, Any]] = None) -> Tuple[bool, str, Dict[str, Optional[int]]]:
    """ONE predicate for the store's writer and every climb reader: a 512 store is COMPLETE when
    every planned slot is DECIDED — covered by a validated response row (`n_done`) or a typed no-fit (`n_nofit`: prompt_over_window,
    the prompt cannot fit the window, decided once and for all) — so `n_undecided == 0` and `n_done + n_nofit == n_planned`.
    Marker presence never says this; a writer that exited 0 with undecided slots (a resume-later `PromptOverWindow` design) used to be
    read as complete and climbed over its holes. Returns (decided, why, the counts read)."""
    if not units:
        return False, "no `units` accounting in the store manifest — completeness cannot be certified", {}
    def _i(k):
        v = units.get(k)
        return int(v) if isinstance(v, (int, float, str)) and str(v).lstrip("-").isdigit() else None
    n_planned, n_done = _i("n_planned"), _i("n_done")
    n_foreign = 0
    if classification is not None:
        # the per-slot re-derivation is the authority over any count the manifest recorded: only PROVEN no-fits are
        # decided (a transient /tokenize outage used to file valid candidates as terminal no-fits). Proven slots beyond
        # the planned count are FOREIGN keys (a torn line that still parsed, another run's rows in this ledger): they are
        # reported, and they never mask an undecided slot — undecided over-window slots are counted from their own rows,
        # and a planned slot with no row at all shows as the shortfall of covered + proven against n_planned once the
        # foreign keys are set aside
        n_nofit = int(classification.get("n_nofit_proven") or 0)
        if n_planned is not None and n_done is not None and n_done + n_nofit > n_planned:
            n_foreign = n_done + n_nofit - n_planned; n_nofit = n_planned - n_done
        n_undecided = (n_planned - n_done - n_nofit) if (n_planned is not None and n_done is not None) else None
        if n_undecided is not None:
            n_undecided = max(n_undecided, int(classification.get("n_undecided_over_window") or 0))
    else:
        n_nofit = _i("n_nofit")
        if n_nofit is None:
            n_nofit = _i("n_prompt_over_window") or 0       # a manifest written before `n_nofit`: the typed no-fit count it did record
        n_undecided = _i("n_undecided")
        if n_undecided is None and n_planned is not None and n_done is not None:
            n_undecided = n_planned - n_done - n_nofit
    counts = {"n_planned": n_planned, "n_done": n_done, "n_nofit": n_nofit, "n_undecided": n_undecided}
    if classification is not None:
        counts["n_undecided_over_window"] = int(classification.get("n_undecided_over_window") or 0)
        counts["n_unparsable_rows"] = int(classification.get("n_unparsable_rows") or 0)
        counts["n_foreign_slots"] = n_foreign
    if n_planned is None or n_done is None or n_undecided is None:
        return False, f"the store manifest's units are incomplete ({units}) — completeness cannot be certified", counts
    if n_undecided != 0:
        return False, f"{n_undecided} of {n_planned} planned slot(s) UNDECIDED (neither a response row nor a typed no-fit): n_done={n_done} n_nofit={n_nofit}", counts
    if n_done + n_nofit != n_planned:
        return False, f"n_done {n_done} + n_nofit {n_nofit} != n_planned {n_planned}", counts
    warn = ""
    if classification is not None and (n_foreign or counts.get("n_unparsable_rows")):
        warn = (f" [ledger integrity: {n_foreign} foreign slot key(s) beyond the plan, {counts.get('n_unparsable_rows')} unparsable line(s) — "
                f"another writer reached this store's bon_errors.jsonl; every planned slot is nonetheless decided]")
    return True, f"every planned slot decided: n_done={n_done} n_nofit={n_nofit} of n_planned={n_planned}{warn}", counts


def store_units(store_dir) -> Optional[Dict[str, Any]]:
    """The `units` block of the store's durable manifest, or None when there is no readable manifest."""
    p = Path(store_dir) / BON_STORE_MANIFEST_UNITS
    if not p.is_file():
        return None
    try:
        u = json.loads(p.read_text()).get("units")
    except (OSError, ValueError, AttributeError):
        return None
    return dict(u) if isinstance(u, Mapping) else None


def store_nofit_slots(store_dir) -> list:
    """The store's DECIDED (proven) no-fit slots, one record per slot, for the climb's coverage table (carried, never dropped)."""
    return list(store_slot_classification(store_dir)["slots"]["proven"])


def store_state(store_dir) -> Tuple[str, dict]:
    """(:data:`STORE_COMPLETE` | :data:`STORE_IN_PROGRESS` | :data:`STORE_ABSENT`, detail) for a cell's 512 store directory.

    complete: the completion marker (:data:`BON_STORE_MANIFEST`) is present — whatever else is there.
    in_progress: no marker, but the started marker or a responses file exists — a store unit is (or was) writing it; the
    detail names the job when the started marker recorded one. A store whose job died before the marker stays in_progress:
    it must be re-run, never climbed.
    absent: neither — a lane with no 512 store (best-of-N runs natively at 4,096)."""
    d = Path(store_dir)
    if (d / BON_STORE_MANIFEST).is_file():
        # the marker says the writer FINISHED; whether the store is complete is the manifest's slot accounting re-derived from the per-slot
        # records (store_completion over store_slot_classification): only PROVEN no-fits decide a slot
        decided, why, counts = store_completion(store_units(d), store_slot_classification(d))
        if decided:
            return STORE_COMPLETE, {"marker": str(d / BON_STORE_MANIFEST), **counts}
        return STORE_PARTIAL, {"marker": str(d / BON_STORE_MANIFEST), "why": why, **counts}
    started = d / BON_STORE_STARTED
    if started.is_file() or (d / BON_RESPONSES).is_file():
        job = None
        if started.is_file():
            try:
                job = json.loads(started.read_text()).get("job_id")
            except (OSError, ValueError):
                job = None
        return STORE_IN_PROGRESS, {"job_id": job, "started": started.is_file(), "responses": (d / BON_RESPONSES).is_file()}
    return STORE_ABSENT, {}


def has_base_store(store_dir) -> bool:
    """Does this cell have a COMPLETE 512 best-of-N store to climb? (the completion marker the 512 wave writes)."""
    return store_state(store_dir)[0] == STORE_COMPLETE


def climb_mode(store_dir=None, *, base_store_exists: Optional[bool] = None) -> str:
    """:data:`MODE_CLIMB` when a COMPLETE 512 store exists for this cell, :data:`MODE_NATIVE` when there is none, and a
    :class:`BonClimbError` naming the live job when the store is IN PROGRESS (every reader — the per-cell planner and
    the driver — branches on this one predicate, so they can never disagree on a store being written).

    Pass ``store_dir`` (the cell's directory) or state ``base_store_exists`` directly. THE DEFAULT
    IS NOT "always climb": a lane with no 512 store (the Llama lane) runs best-of-N natively at
    4,096, and saying so explicitly is what stops a missing store from reading as a clean climb."""
    if base_store_exists is None:
        if store_dir is None:
            raise BonClimbError("climb_mode needs a store_dir or base_store_exists")
        state, detail = store_state(store_dir)
        if state == STORE_IN_PROGRESS:
            job = detail.get("job_id")
            raise BonClimbError(f"512 store in progress — job {job if job else 'unknown'} live ({store_dir}): no completion marker "
                                f"{BON_STORE_MANIFEST} yet; the climb must run after the store unit completes (never climb a store being written)")
        if state == STORE_PARTIAL:
            raise BonClimbError(f"512 store PARTIAL ({store_dir}): {detail.get('why')} — the store unit must be re-run until every planned slot "
                                f"is decided (a response row or a typed no-fit); a climb over the present rows alone would omit at-cap candidates")
        base_store_exists = state == STORE_COMPLETE
    return MODE_CLIMB if base_store_exists else MODE_NATIVE


def cap_for(mode: str) -> int:
    if mode not in MODES:
        raise BonClimbError(f"unknown best-of-N mode {mode!r}; known: {list(MODES)}")
    return CLIMB_CAP     # both modes DRAW at 4,096; they differ in WHICH draws are made


# --------------------------------------------------------------------------- the classification

def _source_of(mod, name: str) -> str:
    tree = ast.parse(inspect.getsource(mod))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return ast.unparse(node)
    raise BonClimbError(f"{mod.__name__} has no definition of {name!r}")


def classifier_fingerprint() -> str:
    """sha256 over the EXACT definitions that make up the classification rule (the repetition rule's
    four constants + three functions, then ``degeneracy.classify``), so a classified store attests
    which rule produced it and a rule change is visible as a different digest."""
    h = hashlib.sha256()
    for n in _RULE_NAMES:
        h.update(_source_of(_cl, n).encode("utf-8")); h.update(b"\n")
    h.update(_source_of(_dg, "classify").encode("utf-8"))
    return h.hexdigest()


def classify_draw(text: Optional[str]) -> Dict[str, Any]:
    """Classify ONE at-cap draw. ``class`` is :data:`CLASS_LOOP` when the REPETITION RULE fires
    (that rule alone), else :data:`CLASS_COHERENT`. A draw only the secondary degeneracy
    classifier flags is coherent and IS climbed, with ``subclass="secondary:<reason>"``. ``climb``
    is the decision bit the manifest and the driver both read."""
    t = text or ""
    d = _cl.degeneracy(t)
    dg = _dg.classify(t)
    loop_rep, loop_deg = bool(d["degenerate"]), bool(dg["degenerate"])
    if loop_rep:
        cls = CLASS_LOOP
        sub = "rep_ratio" + (f"+{dg['degenerate_reason']}" if loop_deg else "")
    elif loop_deg:
        cls, sub = CLASS_COHERENT, f"secondary:{dg['degenerate_reason']}"
    else:
        cls = CLASS_COHERENT
        sub = "repetitive" if max(d["rep_ratio"], d["tail_rep_ratio"]) >= 0.2 else None
    return {"class": cls, "subclass": sub, "climb": cls == CLASS_COHERENT,
            "rep_ratio": d["rep_ratio"], "tail_rep_ratio": d["tail_rep_ratio"],
            "loop_rep_ratio": loop_rep, "loop_degeneracy": loop_deg,
            "degeneracy_reason": dg["degenerate_reason"]}


#: the ``bon_decode_preset`` a store's LEGACY rows (no ``decode`` recorded; drawn before 2026-09-10) were drawn
#: under — the grandfathered T=1 row, spelled as bon_served.BON_HARNESS_DEFAULT (a literal here so this module
#: stays importable without the served stack; pinned equal by test)
LEGACY_STORE_PRESET = "harness_default"
#: the row fields that identify a store's sampler (bon_served.ROW_IDENTITY_KEYS' first two, plus the registry row)
STORE_SAMPLER_KEYS = ("decode", "bon_decode_preset", "decode_policy_id")


def store_sampler(store_dir) -> Optional[Dict[str, Any]]:
    """THE SAMPLER A 512 STORE WAS DRAWN UNDER, read from its rows — what the climb re-draws at
    (:data:`CLIMB_DECODE`). ``{"decode": <request-body keys>, "bon_decode_preset": <label>,
    "decode_policy_id": <registry row or None>}`` when every row that records a sampler records the same one;
    ``None`` for a LEGACY store (no row carries ``decode``: drawn at the grandfathered T=1 row before the record
    existed — the caller resolves that row by name, :data:`LEGACY_STORE_PRESET`). REFUSES a store whose rows
    record different samplers, or that mixes recorded and unrecorded rows: a mixture is never a store, and a
    climb of one would have to pick a sampler for slots that were not drawn under it."""
    d = Path(store_dir)
    if not has_base_store(d):
        raise BonClimbError(f"store_sampler: no {BASE_CAP} store at {d / BON_RESPONSES}")
    seen: Dict[str, Dict[str, Any]] = {}
    n_recorded = n_legacy = 0
    for line in (d / BON_RESPONSES).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue   # a torn tail, skipped as resume skips it
        if "decode" not in r and "bon_decode_preset" not in r:
            n_legacy += 1
            continue
        if not isinstance(r.get("decode"), dict) or not isinstance(r.get("bon_decode_preset"), str):
            raise BonClimbError(
                f"{d / BON_RESPONSES}: row {(r.get('goal_id'), r.get('sample_idx'))} carries a torn sampler record "
                f"(decode={r.get('decode')!r}, bon_decode_preset={r.get('bon_decode_preset')!r}) — unverifiable; "
                "the store cannot be climbed under a sampler it does not state")
        n_recorded += 1
        key = json.dumps({"decode": r["decode"], "bon_decode_preset": r["bon_decode_preset"]}, sort_keys=True)
        seen.setdefault(key, {"decode": dict(r["decode"]), "bon_decode_preset": r["bon_decode_preset"],
                              "decode_policy_id": r.get("decode_policy_id")})
    if len(seen) > 1:
        raise BonClimbError(
            f"{d / BON_RESPONSES}: rows record {len(seen)} different samplers "
            f"{[json.loads(k) for k in seen]} — a mixture is never a store; refusing to climb it")
    if n_recorded and n_legacy:
        raise BonClimbError(
            f"{d / BON_RESPONSES}: {n_recorded} rows record a sampler and {n_legacy} record none — a store is "
            "drawn under ONE sampler; refusing to climb a mixture of recorded and legacy rows")
    if not n_recorded:
        return None
    return next(iter(seen.values()))


def climb_manifest_rows(store_dir, *, lane: str, cell_id: str, cap: int = BASE_CAP) -> list:
    """One row per AT-CAP draw of a cell's 512 store, classified — the climb set is the rows with
    ``climb`` True. Every row carries THE SAMPLER ITS 512 DRAW WAS DRAWN UNDER (``decode`` /
    ``bon_decode_preset`` / ``decode_policy_id`` from the store's own rows, one sampler per store —
    :func:`store_sampler`; a legacy store's rows carry ``decode`` None and the grandfathered label
    :data:`LEGACY_STORE_PRESET`), so the climb driver re-draws each slot at exactly that sampler.
    REFUSES when the store is missing: a lane with no 512 store must go through
    :data:`MODE_NATIVE`, never through an empty climb manifest."""
    d = Path(store_dir)
    if not has_base_store(d):
        raise BonClimbError(
            f"{lane}/{cell_id}: no {BASE_CAP} store at {d / BON_RESPONSES} — this cell has nothing to "
            f"climb and must run best-of-N NATIVELY at {CLIMB_CAP} (climb_mode -> {MODE_NATIVE}); an "
            "empty climb manifest would report zero escalations as if the cell were clean")
    sampler = store_sampler(d) or {"decode": None, "bon_decode_preset": LEGACY_STORE_PRESET, "decode_policy_id": None}
    rows = []
    for i, line in enumerate((d / BON_RESPONSES).read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("finish_reason") != "length":
            continue
        t = r.get("response") or ""
        rows.append({"lane": lane, "cell_id": cell_id, "goal_id": r["goal_id"],
                     "sample_idx": r["sample_idx"], "trial": r.get("trial", 0),
                     "row_index_512": i, "cap_512": int(cap), "finish_reason_512": r["finish_reason"],
                     "response_512_sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(),
                     "n_words_512": len(t.split()), "variant": r["variant"],
                     **{k: sampler[k] for k in STORE_SAMPLER_KEYS}, **classify_draw(t)})
    return rows


# --------------------------------------------------------------------------- the reading

def escalation_summary(*, mode: str, n_rows: int = 0, n_at_cap: int = 0, n_loop: int = 0,
                       n_climbed: int = 0, n_residual_at_cap: int = 0) -> Dict[str, Any]:
    """The per-cell escalation reading, with the NATIVE case named rather than implied.

    For :data:`MODE_NATIVE` every count but ``n_rows`` / ``n_residual_at_cap`` is meaningless and is
    returned as None with ``status`` :data:`STATUS_NOT_APPLICABLE`: a native lane has no 512 draws,
    so "0 escalated" is NOT a clean result and must never render as one. For :data:`MODE_CLIMB`,
    ``loop_residual`` names the deliberate floor: the loop-flagged at-cap draws that were not
    regenerated by ruling (a repeating draw repeats further)."""
    if mode not in MODES:
        raise BonClimbError(f"unknown best-of-N mode {mode!r}; known: {list(MODES)}")
    if mode == MODE_NATIVE:
        return {"mode": mode, "status": STATUS_NOT_APPLICABLE, "cap": CLIMB_CAP,
                "n_rows": int(n_rows), "n_at_cap_512": None, "n_loop_not_climbed": None,
                "n_climbed": None, "at_cap_share_512": None, "loop_residual_share": None,
                "n_residual_at_cap": int(n_residual_at_cap),
                "residual_at_cap_share": (n_residual_at_cap / n_rows) if n_rows else None,
                "note": (f"this cell has no {BASE_CAP} store: best-of-N was drawn NATIVELY at "
                         f"{CLIMB_CAP}. There is no escalation to report — not a clean one.")}
    if n_climbed and n_at_cap and n_climbed + n_loop != n_at_cap:
        raise BonClimbError(f"climbed {n_climbed} + loops {n_loop} != at-cap {n_at_cap}: "
                            "every at-cap draw is either climbed or a loop")
    return {"mode": mode, "cap": CLIMB_CAP, "n_rows": int(n_rows), "n_at_cap_512": int(n_at_cap),
            "status": STATUS_CLIMBED if n_climbed else STATUS_NOTHING_TO_CLIMB,
            "n_loop_not_climbed": int(n_loop), "n_climbed": int(n_climbed),
            "at_cap_share_512": (n_at_cap / n_rows) if n_rows else None,
            "loop_residual_share": (n_loop / n_rows) if n_rows else None,
            "n_residual_at_cap": int(n_residual_at_cap),
            "residual_at_cap_share": (n_residual_at_cap / n_climbed) if n_climbed else None,
            "note": ("loop-flagged at-cap draws are NOT regenerated, by ruling (a repeating draw "
                     "repeats further): they keep their 512 outcome and are an accepted component "
                     "of the residual floor")}


def plan_cells(cells: Iterable[Mapping[str, Any]]) -> list:
    """``[{cell_id, store_dir, mode, cap}]`` for a lane's cells — the branch, made explicit once, so
    a driver never has to infer it. Each input mapping needs ``cell_id`` and ``store_dir``."""
    out = []
    for c in cells:
        sd = c["store_dir"]
        out.append({"cell_id": c["cell_id"], "store_dir": str(sd),
                    "mode": climb_mode(sd), "cap": CLIMB_CAP, "decode": CLIMB_DECODE})
    return out


def lanes_by_mode(plan: Sequence[Mapping[str, Any]]) -> Dict[str, list]:
    return {m: [p["cell_id"] for p in plan if p["mode"] == m] for m in MODES}
