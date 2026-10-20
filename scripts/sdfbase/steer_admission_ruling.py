"""steer_admission_ruling.py — the RULED AlphaSteer operating point as a LOAD-TIME gate on the steer admission record.

Both consumers of ``configs/sdfbase/steer_admission.json`` (``scripts/sdfbase/gsm8k/g8k_cells.load_steer_admission`` and
``scripts/sdfbase/math500/cells.load_steer_admission``) call :func:`check_entries` after their own field validation. It
resolves the ruled operating points through the oracle ``scripts/alphasteer_f22/operating_points.py`` (``build``: the RUNG
from the ruling, lambda + object sha from that rung's row of ``ladder_canon/<target>/lambda_ladder.json``) and REFUSES, naming
the entry and the field, when an AlphaSteer entry's (rung, lambda, object_sha256, selection_outcome) or its bank paths do
not bind to that point. Why: on 2026-09-14 a user ruling moved the operating point on three of four models while every
checkout still carried the old pins, and a byte-identity check on those pins PASSES and serves the superseded dose. A pin
file can only be trusted against an authority it did not write itself.

A CANDIDATE POINT (``selection_outcome`` starting with ``candidate_``; the INTERIM gemma lam2 of 2026-09-14, active
while the first ruling still named lam3 and that rung stood disqualified on the measured readings) is admitted ONLY when the
entry's (rung, lambda, object_sha256) equal the ladder record's row AT THAT RUNG (by index, the oracle's own data path), its bank
paths bind to that rung, AND that rung is not the ruled one; a candidate that disagrees with its record refuses exactly like a
drifted ruled pin, and a candidate naming the ruled rung itself is a contradiction that refuses by name. Nothing here decides
between a candidate and the ruling — the record says which pin is ACTIVE and why. (History: when the SECOND ruling of
2026-09-14 moved gemma to lam2 and it was ported into the oracle, the committed candidate entry named the ruled rung and this
gate refused the record at import, as designed; the record was re-stamped to the oracle's ``ruled_2026-09-14_second`` on 2026-09-15
with lam3 under ``superseded``.)

THE ARDITI ENTRIES (2026-09-14, the second ruling): an Arditi point is the arditi strength battery's alpha pick, not a ladder rung, so
it carries no ladder word — ``selection_outcome`` is null, or EXACTLY ``ruled_2026-09-14`` where a user ruling overrode the battery's pick
(llama8b: alpha 0.1, the inclusive-boundary pick, -> 0.05). A RULED Arditi entry must bind by itself and to its lane's record
(:func:`arditi_entry_problems`): its ``ruling`` (ruled_on, alpha == the entry's, object_sha256 == the pin, ``supersedes`` = a DIFFERENT
alpha and 64-hex sha) and ``superseded`` (alpha / view / object_sha256 equal to ``ruling.supersedes``, ``superseded_on``, ``superseded_by``)
blocks agree; the view is the alpha SPELLED (``ar_r0p05``), the staged id ``<view>__arditi``, both bank roles end in ``cells_view/<view>``;
and where the arditi strength battery's record is present on this machine (``DEFAULT_ARDITI_RECORD``: targets.<key>.cells.<view>.alpha)
the entry's alpha equals the battery's for that view — a view the battery never measured is a refusal. The battery record carries no
object shas, so the sha binds to the bank's staged_from.json / VIEW_MANIFEST.json and the lane's pins record, hashed on disk at the
re-pin and cited on the entry. Any Arditi entry naming both ``alpha`` and ``view`` has the spelling checked; an entry naming neither
(a synthetic fixture) is left alone.

THE ALL-POSITIONS ARDITI ENTRY (2026-09-17; cell ``arditi_allpos``, qwen35_4b only — NEVER ``arditi``, the prefill-only arm: a different
phase AND a different dose unit): a ``vector_all`` object served at every position in BOTH phases on the native residual slot
(``serving_path`` lora_native; op add_vector at coeff 1.0, the dose baked into the vector's norm — a rung is different BYTES, never a serve-time
flag). Its dose is a RAW COEFFICIENT c, never an alpha; the view is c SPELLED (:func:`c_view`: ``ar_c0p3``), the staged id
``<view>__arditi_allpos``, both bank roles end in ``cells_view/<view>``. Its operating point is a DECLARED OVERRIDE of the selection rule —
``selection_outcome`` EXACTLY ``user_override_over_budget_2026-09-16`` — and the entry's ``selection`` block carries BOTH readings
(``rule_returns``: a DIFFERENT view, read within budget; ``override_reading_pp`` OVER ``budget_pp``), the ``ladder_cap`` that identifies the
reading (at another cap the rule returns a different point) and the single-trial flag (a margin is a ratio against a crude floor, never a
significance claim). A structured ``fit_provenance`` block must be complete (every key of :data:`FIT_PROVENANCE_KEYS`, non-null) with
``served_layer == fit_layer - 1``: a vector_all blob stores the SERVED block-output layer and is served as stored, the INVERSE of static_r at
site=pre, which stores the fit layer. :func:`arditi_allpos_entry_problems` refuses each of these by name — an alpha-shaped entry (an ``alpha``
key, an ``ar_r…`` view) first of all — and a lens key on the native route.

THE DISPATCH IS TOTAL: :func:`check_entries` sends every steer cell to a named gate (:data:`GATED_CELLS`) and REFUSES a cell no gate knows,
naming it. Before 2026-09-17 a cell outside {alphasteer, arditi} fell through UNCHECKED — a silent admission, which is itself a defect.

WHEN THE LADDER RECORD IS ABSENT ON THIS MACHINE (a cluster checkout: the grade dir lives on the analysis workstation) the check is SKIPPED
and returns None — the gate runs where the record is edited and the matrices are generated, never guesses where it cannot
read. Unresolvable WITH the record present (a target the ruling does not name, the oracle module missing) is a refusal.

TODO(operating_points): the oracle lives in its own module, loaded here by path wherever it sits in
the tree — nothing here duplicates its RULING or its derivation.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Mapping

REPO = Path(__file__).resolve().parents[2]
ORACLE_REL = Path("scripts/alphasteer_f22/operating_points.py")
#: mirrors operating_points.DEFAULT_LADDER (pinned equal by tests/test_steer_admission_ruling.py when the oracle is present):
#: the check must decide "skip vs run" BEFORE importing the oracle, so a checkout without the oracle file and without the
#: record (a cluster node) loads the admission exactly as before
#: Both defaults are in the grade root's placeholder form ($GRADE): :func:`_on_this_host` expands one at use, and a host where the
#: grade root is not set (a cluster node) skips that half exactly as a host where the record is absent does.
DEFAULT_LADDER_ROOT = Path("$GRADE/alphasteer_f22_rebake/ladder_canon")
CELL_ALPHASTEER = "alphasteer"
CELL_ARDITI = "arditi"
LAMBDA_TOL = 1e-9
#: the arditi strength battery's own record (targets.<key>.cells.<view>.alpha): the authority a RULED Arditi alpha binds to where present
DEFAULT_ARDITI_RECORD = Path("$GRADE/arditi_strength/battery_curves.json")
#: the ONE non-null selection_outcome an Arditi entry may carry (g8k_cells.ARDITI_SELECTION_OUTCOMES pins the same word)
ARDITI_RULED_OUTCOME = "ruled_2026-09-14"
#: THE ALL-POSITIONS ARDITI ARM (qwen35_4b only): a `vector_all` object at every position in BOTH phases on the native residual slot; its
#: dose a RAW COEFFICIENT c (never an alpha), its view the coefficient spelled (:func:`c_view`), its staged id `<view>__arditi_allpos`
CELL_ARDITI_ALLPOS = "arditi_allpos"
#: the ONE outcome word an all-positions entry may carry: the DECLARED OVERRIDE of the selection rule, over budget, filed in the
#: lane's own selection record with `rule_returns` naming the rule's (different) point — never the Arditi ruled word, never a ladder word
ALLPOS_OUTCOME = "user_override_over_budget_2026-09-16"
ALLPOS_DOSE_UNIT = "raw_coefficient"
ALLPOS_KIND = "vector_all"
ALLPOS_PHASE = "both"
ALLPOS_SERVING_PATH = "lora_native"
#: what the native residual slot attests for a constant-kind payload (native_residual: OP_ADD_VECTOR at coeff 1.0 — the dose is in the
#: vector's norm — position mode `all`, both phases): the entry's `attestation_expect` says exactly this, and carries no lens key
ALLPOS_ATTESTATION_EXPECT: Mapping[str, object] = {"serving_path": ALLPOS_SERVING_PATH, "op": "add_vector", "coeff": 1.0,
                                                   "positions": "all", "phase": ALLPOS_PHASE, "served_positions_expected": "all"}
#: every key a `fit_provenance` block carries, non-null — a relayed value is RECORDED or the record refuses; never guessed
#: `capture_positions` is REQUIRED as its own key (2026-09-17): the paper-row export refuses a fit provenance missing any key and
#: cannot parse `rendering`'s prose, so the capture positions stated only there were a fact the export did not have.
FIT_PROVENANCE_KEYS = ("fitted_on_model", "fitted_on_revision", "rendering", "capture_positions", "item_source", "item_source_repo_sha",
                       "realised_counts",
                       "direction_sha256", "direction_pt_sha256", "fit_layer", "fit_position", "direction_norm", "served_layer",
                       "fit_quality_gate", "fit_record_sha256")
FIT_REALISED_COUNT_KEYS = ("harmful_train", "harmful_val", "harmless_train", "harmless_val")
FIT_QUALITY_GATE_KEYS = ("harmful_mean", "harmless_mean", "harmful_frac_positive", "harmless_frac_negative")
#: the alpha spelling of the prefill-only arm's views (:func:`arditi_view`: ar_r0p05, ar_rm0p1) — refused on an all-positions entry
_ALPHA_VIEW_RE = re.compile(r"ar_rm?\d+(?:p\d+)?")
#: every steer cell an admission gate knows; :func:`check_entries` refuses any other cell BY NAME
GATED_CELLS = (CELL_ALPHASTEER, CELL_ARDITI, CELL_ARDITI_ALLPOS)


def load_oracle(repo: Path = REPO):
    """The operating_points module, loaded by PATH (scripts/ is not a package)."""
    p = Path(repo) / ORACLE_REL
    if not p.is_file():
        raise SystemExit(f"steer_admission_ruling: no oracle at {p} — the ruled operating points live in "
                         f"scripts/alphasteer_f22/operating_points.py (PR #118); a record cannot be checked without it")
    spec = importlib.util.spec_from_file_location("alphasteer_f22_operating_points", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


CANDIDATE_PREFIX = "candidate_"


def _on_this_host(default: Path) -> Path | None:
    """A default record path with its placeholder ($GRADE) expanded through sisteer.site, or None where the grade root is not
    set on this host (no site value, no environment variable): the caller then skips, as it does for an absent record."""
    import sys
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from sisteer import site as _site
    try:
        return _site.expand_path(default)
    except _site.PlaceholderError:
        return None


def _oracle_and_root(*, ladder_root: Path | None, repo: Path):
    root = Path(ladder_root) if ladder_root is not None else _on_this_host(DEFAULT_LADDER_ROOT)
    if root is None or not root.is_dir():
        return None, root
    return load_oracle(repo), root


def ruling_record(*, ladder_root: Path | None = None, curves: Path | None = None, repo: Path = REPO) -> dict | None:
    """The oracle's record (``operating_points.build``), or None when the ladder root is absent on this machine."""
    mod, root = _oracle_and_root(ladder_root=ladder_root, repo=repo)
    if mod is None:
        return None
    return mod.build(root, Path(curves) if curves is not None else mod.DEFAULT_CURVES)


def is_candidate(outcome) -> bool:
    return isinstance(outcome, str) and outcome.startswith(CANDIDATE_PREFIX)


def _bind_problems(where: str, entry: Mapping, *, rung_i: int, lam: float, sha: str, against: str) -> list[str]:
    out: list[str] = []
    for field, got, want in (("rung", entry.get("rung"), rung_i), ("lambda", entry.get("lambda"), lam),
                             ("object_sha256", entry.get("object_sha256"), sha)):
        if field == "lambda" and isinstance(got, (int, float)) and isinstance(want, (int, float)):
            ok = abs(float(got) - float(want)) < LAMBDA_TOL
        else:
            ok = got == want
        if not ok:
            out.append(f"{where}.{field}: record={got!r} {against}={want!r}")
    tail = f"ladder_canon/lam{rung_i}/cells_view/{entry.get('view')}"
    for role, rel in (entry.get("bank_view_rel") or {}).items():
        if rel and not str(rel).endswith(tail):
            out.append(f"{where}.bank_view_rel.{role}: {rel!r} does not end with {tail!r} — the bank path must bind the {against} rung")
    return out


def entry_problems(key: str, cell: str, entry: Mapping, target: Mapping | None, ladder: Mapping | None = None) -> list[str]:
    """Every way `entry` (an AlphaSteer admission entry for model `key`) fails to bind: a RULED entry to the oracle's `target`
    (rung / lambda / sha / outcome word); a CANDIDATE entry (`candidate_*`) to the ladder record's row AT ITS OWN RUNG (`ladder` = the
    record the oracle read for `key`; the row by index) — a candidate that disagrees with its record refuses like any drift."""
    where = f"entries.{key}.{cell}"
    if target is None:
        return [f"{where}: the ruling names no target {key!r} — unresolvable is a refusal, not a pass"]
    outcome = entry.get("selection_outcome")
    if is_candidate(outcome):
        if ladder is None:
            return [f"{where}: {outcome!r} needs the ladder record for {key!r} and none was readable — unresolvable is a refusal"]
        rungs = ladder.get("rungs") or []
        rung = entry.get("rung")
        if not (isinstance(rung, int) and 1 <= rung <= len(rungs)):
            return [f"{where}.rung: record={rung!r} — a candidate must name one of the ladder's {len(rungs)} rungs by index"]
        row = rungs[rung - 1]
        out = _bind_problems(where, entry, rung_i=rung, lam=row["lambda"], sha=row["object_sha256"], against="ladder-row")
        if rung == target["rung_i"] and entry.get("object_sha256") == target["object_sha256"]:
            out.append(f"{where}.selection_outcome: {outcome!r} names the RULED rung {rung} itself — a ruled point is not a candidate")
        return out
    out = _bind_problems(where, entry, rung_i=target["rung_i"], lam=target["lambda"], sha=target["object_sha256"], against="ruled")
    if outcome != target["selection_outcome"]:
        out.append(f"{where}.selection_outcome: record={outcome!r} ruled={target['selection_outcome']!r}")
    return out


def arditi_view(alpha: float) -> str:
    """The arditi strength lane's view name for a dose: the alpha SPELLED (0.05 -> ar_r0p05, 0.1 -> ar_r0p1, -0.1 -> ar_rm0p1)."""
    a = float(alpha)
    body = f"{abs(a):g}".replace(".", "p")
    return f"ar_r{'m' if a < 0 else ''}{body}"


def _is_sha(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def load_arditi_record(path: Path | None = None) -> dict | None:
    """The battery record (``battery_curves.json``), or None when absent on this machine (skipped, never guessed)."""
    p = Path(path) if path is not None else _on_this_host(DEFAULT_ARDITI_RECORD)
    if p is None or not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def arditi_entry_problems(key: str, cell: str, entry: Mapping, battery: Mapping | None) -> list[str]:
    """Every way an Arditi admission entry fails to bind: a ladder word as its outcome; a RULED entry whose `ruling` / `superseded`
    blocks disagree with each other or with the pin; a view that is not its alpha spelled, a staged id or a bank path bound to another
    view; an alpha the battery record (when present) does not carry for that view."""
    where = f"entries.{key}.{cell}"
    out: list[str] = []
    outcome = entry.get("selection_outcome")
    if outcome not in (None, ARDITI_RULED_OUTCOME):
        return [f"{where}.selection_outcome: record={outcome!r} — an Arditi entry carries null or exactly {ARDITI_RULED_OUTCOME!r}; the ladder's words are the AlphaSteer entries'"]
    alpha, view = entry.get("alpha"), entry.get("view")
    if outcome == ARDITI_RULED_OUTCOME:
        ruling, sup = entry.get("ruling"), entry.get("superseded")
        if not isinstance(ruling, Mapping) or not isinstance(sup, Mapping):
            return [f"{where}: {ARDITI_RULED_OUTCOME!r} needs a `ruling` block and a `superseded` block — a ruling that supersedes nothing is not a re-pin"]
        if not ruling.get("ruled_on"):
            out.append(f"{where}.ruling.ruled_on: missing")
        for field in ("alpha", "object_sha256"):
            if ruling.get(field) != entry.get(field):
                out.append(f"{where}.ruling.{field}: ruling={ruling.get(field)!r} entry={entry.get(field)!r}")
        old = ruling.get("supersedes")
        if not isinstance(old, Mapping) or not _is_sha(old.get("object_sha256")) or old.get("alpha") is None:
            out.append(f"{where}.ruling.supersedes: must name the superseded alpha and 64-hex object sha, got {old!r}")
        else:
            if old["object_sha256"] == entry.get("object_sha256") or old["alpha"] == alpha:
                out.append(f"{where}.ruling.supersedes: names the ACTIVE point itself (alpha {old['alpha']!r}, sha {old['object_sha256'][:12]}…) — a ruling supersedes a different dose")
            for field in ("alpha", "view", "object_sha256"):
                if sup.get(field) != old.get(field):
                    out.append(f"{where}.superseded.{field}: superseded={sup.get(field)!r} ruling.supersedes={old.get(field)!r}")
            if old.get("view") is not None and old.get("view") != arditi_view(old["alpha"]):
                out.append(f"{where}.ruling.supersedes.view: {old.get('view')!r} is not alpha {old['alpha']!r} spelled ({arditi_view(old['alpha'])!r})")
        for field in ("superseded_on", "superseded_by"):
            if not sup.get(field):
                out.append(f"{where}.superseded.{field}: missing")
        if alpha is None or view is None:
            out.append(f"{where}: a ruled entry names its alpha and view (got alpha={alpha!r}, view={view!r})")
    if alpha is not None and view is not None:
        want_view = arditi_view(alpha)
        if view != want_view:
            out.append(f"{where}.view: {view!r} is not alpha {alpha!r} spelled ({want_view!r}) — the view names the dose baked into the object")
        if entry.get("staged_id") is not None and entry.get("staged_id") != f"{view}__arditi":
            out.append(f"{where}.staged_id: {entry.get('staged_id')!r} does not bind the view {view!r} ({view}__arditi)")
        tail = f"cells_view/{view}"
        for role, rel in (entry.get("bank_view_rel") or {}).items():
            if rel and not str(rel).endswith(tail):
                out.append(f"{where}.bank_view_rel.{role}: {rel!r} does not end with {tail!r} — the bank path must bind the pinned view")
        if battery is not None:
            cells = ((battery.get("targets") or {}).get(key) or {}).get("cells") or {}
            if key not in (battery.get("targets") or {}):
                out.append(f"{where}: the arditi strength battery record names no target {key!r} — unresolvable is a refusal, not a pass")
            elif view not in cells:
                out.append(f"{where}.view: the battery never measured {view!r} for {key!r} (its cells: {sorted(cells)}) — a dose outside the battery cannot be its pick or a ruling over it")
            elif abs(float(cells[view].get("alpha")) - float(alpha)) >= LAMBDA_TOL:
                out.append(f"{where}.alpha: record={alpha!r} battery[{view}]={cells[view].get('alpha')!r}")
    return out


def c_view(c: float) -> str:
    """The all-positions lane's view name for a dose: the raw coefficient SPELLED, in :func:`arditi_view`'s style with the unit letter
    ``c`` (0.3 -> ar_c0p3, 0.2 -> ar_c0p2, 1.0 -> ar_c1, 0.5 -> ar_c0p5) — never ``r``, the alpha spelling."""
    x = float(c)
    body = f"{abs(x):g}".replace(".", "p")
    return f"ar_c{'m' if x < 0 else ''}{body}"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def arditi_allpos_entry_problems(key: str, cell: str, entry: Mapping) -> list[str]:
    """Every way an ALL-POSITIONS Arditi entry fails to bind, each named: an alpha-shaped entry (an ``alpha`` key, an ``ar_r…`` view); a
    dose unit, kind or phase that is not this arm's; a missing ``selection`` block or outcome, or any outcome word but the declared override;
    a view that is not the dose spelled (:func:`c_view`), a staged id or bank path bound to another view, a top-level ``coefficient`` that
    disagrees with the block's dose; a reading without its ``ladder_cap`` or its trial count; a ``rule_returns`` that is absent or the
    selected view itself (an override supersedes a DIFFERENT point); an override reading that is not over budget, or a rule reading that is;
    an absent or incomplete ``fit_provenance`` (every :data:`FIT_PROVENANCE_KEYS` key, the four realised counts, the four quality-gate
    statistics, 64-hex shas) or a ``served_layer`` that is not ``fit_layer - 1``; a ``serving_path`` / ``attestation_expect`` that does not
    name the native residual slot (op add_vector, coeff 1.0, positions all, phase both), or a lens key on that route."""
    where = f"entries.{key}.{cell}"
    out: list[str] = []
    if "alpha" in entry:
        out.append(f"{where}.alpha: present ({entry.get('alpha')!r}) — this arm's dose is a RAW COEFFICIENT, never an alpha; an `alpha` key is the "
                   f"prefill-only `{CELL_ARDITI}` arm's shape")
    view = entry.get("view")
    if isinstance(view, str) and _ALPHA_VIEW_RE.fullmatch(view):
        out.append(f"{where}.view: {view!r} is the ALPHA spelling (ar_r…, the `{CELL_ARDITI}` arm's) — an all-positions view is the coefficient spelled (ar_c…)")
    if entry.get("dose_unit") != ALLPOS_DOSE_UNIT:
        out.append(f"{where}.dose_unit: record={entry.get('dose_unit')!r} — exactly {ALLPOS_DOSE_UNIT!r} (never an alpha, never residual_norm_fraction)")
    if entry.get("kind") != ALLPOS_KIND:
        out.append(f"{where}.kind: record={entry.get('kind')!r} — the all-positions arm is a {ALLPOS_KIND!r} object (static_r is the prefill-only arm's kind)")
    if entry.get("phase") != ALLPOS_PHASE:
        out.append(f"{where}.phase: record={entry.get('phase')!r} — the all-positions arm steers in BOTH phases (prefill and every decode step), spelled {ALLPOS_PHASE!r}")
    if entry.get("selection_outcome") != ALLPOS_OUTCOME:
        out.append(f"{where}.selection_outcome: record={entry.get('selection_outcome')!r} — an all-positions entry carries exactly {ALLPOS_OUTCOME!r} "
                   f"(the user's declared override); never the Arditi ruled word, never a ladder word")
    sel = entry.get("selection")
    if not isinstance(sel, Mapping):
        out.append(f"{where}.selection: missing — an override is filed with BOTH readings (the rule's and the override's), the budget and the ladder cap "
                   "that identifies the reading, or it is a reading of nothing")
    else:
        if sel.get("outcome") is None:
            out.append(f"{where}.selection.outcome: missing — the block names the outcome it records")
        elif sel.get("outcome") != ALLPOS_OUTCOME:
            out.append(f"{where}.selection.outcome: record={sel.get('outcome')!r} — exactly {ALLPOS_OUTCOME!r}; any other word claims a different selection")
        dose = sel.get("dose")
        if not isinstance(dose, Mapping) or not _num(dose.get("value")) or dose.get("unit") != ALLPOS_DOSE_UNIT:
            out.append(f"{where}.selection.dose: {dose!r} — the dose is {{value: <c>, unit: {ALLPOS_DOSE_UNIT!r}}}, a raw coefficient, never an alpha")
        else:
            c = float(dose["value"])
            want_view = c_view(c)
            if view != want_view:
                out.append(f"{where}.view: {view!r} is not the dose c={c!r} spelled ({want_view!r}) — the view names the coefficient baked into the object")
            if "coefficient" in entry and (not _num(entry.get("coefficient")) or abs(float(entry["coefficient"]) - c) >= LAMBDA_TOL):
                out.append(f"{where}.coefficient: record={entry.get('coefficient')!r} selection.dose.value={c!r} — one dose, written once")
        if isinstance(view, str):
            if entry.get("staged_id") != f"{view}__{CELL_ARDITI_ALLPOS}":
                out.append(f"{where}.staged_id: {entry.get('staged_id')!r} does not bind the view {view!r} ({view}__{CELL_ARDITI_ALLPOS}) — "
                           f"`__{CELL_ARDITI}` is the prefill-only arm's suffix")
            tail = f"cells_view/{view}"
            for role, rel in (entry.get("bank_view_rel") or {}).items():
                if rel and not str(rel).endswith(tail):
                    out.append(f"{where}.bank_view_rel.{role}: {rel!r} does not end with {tail!r} — the bank path must bind the pinned view")
        cap = sel.get("ladder_cap")
        if cap is None:
            out.append(f"{where}.selection.ladder_cap: missing — the cap is part of the reading's IDENTITY (at another cap the rule returns a different "
                       "point); a selection block without it is not a reading of anything")
        elif not (isinstance(cap, int) and not isinstance(cap, bool) and cap > 0):
            out.append(f"{where}.selection.ladder_cap: {cap!r} is not a positive token cap")
        if not isinstance(sel.get("single_trial"), bool):
            out.append(f"{where}.selection.single_trial: {sel.get('single_trial')!r} — the ladder's trial count decides how its margins may be read "
                       "(a single trial is a ratio against a crude floor, never a significance claim)")
        rr = sel.get("rule_returns")
        if not rr:
            out.append(f"{where}.selection.rule_returns: missing — an override records the point the rule returned, or nothing was overridden")
        elif rr == view:
            out.append(f"{where}.selection.rule_returns: {rr!r} is the selected view itself — an override must supersede a DIFFERENT point; "
                       "if the rule chose this dose the outcome is not an override")
        readings = ("rule_reading_pp", "override_reading_pp", "budget_pp")
        for f in readings:
            if not _num(sel.get(f)):
                out.append(f"{where}.selection.{f}: {sel.get(f)!r} is not a number — both readings and the budget are recorded, never paraphrased")
        if all(_num(sel.get(f)) for f in readings):
            if float(sel["override_reading_pp"]) <= float(sel["budget_pp"]):
                out.append(f"{where}.selection.override_reading_pp: {sel['override_reading_pp']!r} is within the budget {sel['budget_pp']!r} — the outcome "
                           "word claims an OVER-budget override; a within-budget point is the rule's own return or a different outcome")
            if float(sel["rule_reading_pp"]) > float(sel["budget_pp"]):
                out.append(f"{where}.selection.rule_reading_pp: {sel['rule_reading_pp']!r} is over the budget {sel['budget_pp']!r} — the rule returns the "
                           "highest dose WITHIN budget, so this reading is not the rule's return")
    fp = entry.get("fit_provenance")
    if not isinstance(fp, Mapping):
        out.append(f"{where}.fit_provenance: missing — the direction's provenance (fitted-on checkpoint, rendering, items, shas, layers, quality gate) "
                   "is recorded or the entry refuses; never guessed")
    else:
        for k in FIT_PROVENANCE_KEYS:
            if fp.get(k) is None:
                out.append(f"{where}.fit_provenance.{k}: missing — a relayed value is recorded or the record refuses; never guessed")
        for sub, keys in (("realised_counts", FIT_REALISED_COUNT_KEYS), ("fit_quality_gate", FIT_QUALITY_GATE_KEYS)):
            blk = fp.get(sub)
            if isinstance(blk, Mapping):
                for k in keys:
                    if not _num(blk.get(k)):
                        out.append(f"{where}.fit_provenance.{sub}.{k}: {blk.get(k)!r} — missing or not a number")
            elif blk is not None:
                out.append(f"{where}.fit_provenance.{sub}: {blk!r} is not a block of {list(keys)}")
        for k in ("direction_sha256", "direction_pt_sha256", "fit_record_sha256"):
            if fp.get(k) is not None and not _is_sha(fp.get(k)):
                out.append(f"{where}.fit_provenance.{k}: {fp.get(k)!r} is not a 64-hex sha256")
        if fp.get("served_layer") is not None and fp.get("served_layer") != entry.get("served_layer"):
            out.append(f"{where}.served_layer: record={entry.get('served_layer')!r} fit_provenance.served_layer={fp.get('served_layer')!r} — one served layer, written once")
        if _num(fp.get("fit_layer")) and _num(entry.get("served_layer")) and int(fp["fit_layer"]) - 1 != int(entry["served_layer"]):
            out.append(f"{where}.served_layer: {entry.get('served_layer')!r} is not fit_layer - 1 ({fp['fit_layer']} - 1 = {int(fp['fit_layer']) - 1}) — a "
                       "vector_all blob stores the SERVED block-output layer and is served as stored (the INVERSE of static_r at site=pre, which stores "
                       "the fit layer); writing the fit layer here steers the wrong block")
    if entry.get("serving_path") != ALLPOS_SERVING_PATH:
        out.append(f"{where}.serving_path: record={entry.get('serving_path')!r} — this arm is served on the graph-captured native residual slot in every "
                   f"leg ({ALLPOS_SERVING_PATH!r}); the lens is not its route")
    att = entry.get("attestation_expect")
    if not isinstance(att, Mapping):
        out.append(f"{where}.attestation_expect: missing — the leg's attestation is compared against a declared expectation of the native residual slot")
    else:
        for k, want in ALLPOS_ATTESTATION_EXPECT.items():
            got = att.get(k)
            ok = abs(float(got) - float(want)) < LAMBDA_TOL if (k == "coeff" and _num(got)) else got == want
            if not ok:
                out.append(f"{where}.attestation_expect.{k}: record={got!r} native-slot={want!r}"
                           + (" — the dose lives in the vector's norm; the slot serves the constant kinds at coeff 1.0 exactly, so a rung is different "
                              "BYTES, never a serve-time flag" if k == "coeff" else ""))
        if "lens_reduction_mode" in att:
            out.append(f"{where}.attestation_expect.lens_reduction_mode: {att.get('lens_reduction_mode')!r} is a LENS key on a native route — a single "
                       f"add_vector op at the block output has nothing to reduce; a lens key here is the `{CELL_ARDITI}` entries' shape copied across")
        if att.get("served_layer") is not None and att.get("served_layer") != entry.get("served_layer"):
            out.append(f"{where}.attestation_expect.served_layer: record={att.get('served_layer')!r} entry={entry.get('served_layer')!r}")
    return out


def check_entries(entries: Mapping[tuple[str, str], Mapping], *, where: str, ladder_root: Path | None = None,
                  curves: Path | None = None, repo: Path = REPO, arditi_record: Path | None = None) -> dict | None:
    """Refuse (SystemExit, the entry named) when any AlphaSteer entry drifts from the ruled operating point, when an Arditi
    entry fails :func:`arditi_entry_problems` (its own blocks, its view / paths, the battery record where present), when an
    all-positions Arditi entry fails :func:`arditi_allpos_entry_problems`, or when a steer cell no gate knows is present (the dispatch is
    TOTAL over :data:`GATED_CELLS`: an unknown cell is named, never admitted unchecked); return the oracle's record when every entry
    binds; return None when the ladder record is absent on this machine (the AlphaSteer half skipped — the Arditi halves still run:
    they need no ladder, and the battery half skips on its own when that record is absent)."""
    battery = load_arditi_record(arditi_record)
    arditi_problems: list[str] = []
    allpos_problems: list[str] = []
    unknown: list[str] = []
    for (key, cell), entry in entries.items():
        if cell == CELL_ARDITI:
            arditi_problems += arditi_entry_problems(key, cell, entry, battery)
        elif cell == CELL_ARDITI_ALLPOS:
            allpos_problems += arditi_allpos_entry_problems(key, cell, entry)
        elif cell != CELL_ALPHASTEER:       # the AlphaSteer gate runs below, against the oracle's record
            unknown.append(f"entries.{key}.{cell}")
    if unknown:
        raise SystemExit(f"{where}: no admission gate knows the steer cell(s) {unknown} (gates: {list(GATED_CELLS)}) — a cell no gate knows is "
                         "refused by name, never admitted unchecked; add a gate for it in scripts/sdfbase/steer_admission_ruling.py")
    if arditi_problems:
        raise SystemExit(f"{where}: an Arditi entry does not bind to its ruling / its lane's record: " + "; ".join(arditi_problems)
                         + " — the view, staged id and bank paths must name the ruled alpha; a ruled entry keeps the superseded dose under `superseded`; never edit the pin by hand.")
    if allpos_problems:
        raise SystemExit(f"{where}: an all-positions Arditi entry does not bind to its declared override / its fit provenance / the native slot: "
                         + "; ".join(allpos_problems)
                         + " — the dose is a raw coefficient (the view is c spelled, the staged id `<view>__arditi_allpos`), the outcome is the user's "
                         "declared OVER-budget override with BOTH readings and the ladder cap on the entry, the fit provenance is complete, and the arm "
                         "serves on the native residual slot at coeff 1.0; never file it as an alpha, never edit the pin by hand.")
    mod, root = _oracle_and_root(ladder_root=ladder_root, repo=repo)
    if mod is None:
        return None
    rec = mod.build(root, Path(curves) if curves is not None else mod.DEFAULT_CURVES)
    problems: list[str] = []
    for (key, cell), entry in entries.items():
        if cell != CELL_ALPHASTEER:
            continue
        ladder = mod._ladder(key, root) if (is_candidate(entry.get("selection_outcome")) and key in rec["targets"]) else None
        problems += entry_problems(key, cell, entry, rec["targets"].get(key), ladder)
    if problems:
        raise SystemExit(f"{where}: the steer admission record DRIFTS from the ruled AlphaSteer operating points "
                         f"({rec['ruling']['ruled_on']}): " + "; ".join(problems) + " — a byte-identity check on these pins "
                         "would PASS and serve a superseded dose. Regenerate the record from the oracle "
                         "(scripts/alphasteer_f22/operating_points.py build/emit) or bind a candidate to its ladder row; never edit the pin by hand.")
    return rec
