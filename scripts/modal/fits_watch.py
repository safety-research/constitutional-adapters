#!/usr/bin/env python
"""Harvest/watch for a deployed-app fit wave (twosi_fits.py / sdfbase_fits.py
--mode fits-spawn): poll the `twosi-fits` volume for COMPLETE markers and
the spawned FunctionCalls for terminal states, print one status line per
cell per change, and EXIT at the wave's terminal state — so it runs under a
Monitor (persistent) or `setsid nohup ... &` exactly like watch-job.sh.

What "COMPLETE" means here: a cell counts complete only when its COMPLETE
marker's ``fit_fingerprint`` EQUALS the manifest row's
``expect_fingerprint`` — marker EXISTENCE alone lied: a round-2 refit
publishes into the same cell dir as the round-1 winner, so a stale
round-1 marker sits under a refit in flight and an existence check reads
the whole wave as done. Under the ``sdfbase-fits`` app every row MUST
carry an expected fingerprint (fits-spawn records it); a row without one
is reported as unverifiable, never counted. Manifests of other apps (the
twosi wave, whose rows predate the field) keep the historical
existence-only reading, announced once.

SUCCESSOR SPAWNING (the 24 h
Modal ceiling cannot be raised, so continuation is the design, not a
recovery): a cell whose call ended in a real function TIMEOUT or a FAILURE
(the call raised / the container was killed) without a fingerprint-matching
COMPLETE marker gets EXACTLY ONE successor call per event — the SAME item
tuple fits-spawn sent, on the SAME deployed function. Both come from the
ROW'S OWN SPAWN PIN: fits-spawn records on every manifest row the deployed
function it spawned the cell on (`app`, `function`, `function_id`) and the
item fields a successor must reproduce (`config_sha`, `overwrite`,
`checkpoint_every`). One manifest carries a whole campaign, so the
manifest's TOP-LEVEL copies of those values describe only the LAST
fits-spawn that wrote it — a merged gemma+qwen manifest read from the top
would send a qwen cell's successor through the std image with gemma's
config sha (or vice versa). A row without its own pin is refused a
successor ("re-run fits-spawn"), like a row without expect_fingerprint.
The version pin: a successor resumes a volume checkpoint, so it must run
the SAME deployed code as its predecessor. The row's `function_id` was the
original pin, and it is VACUOUS — a deployed function's id does not change
on `modal deploy` (sdfbase-fits v10 -> v11 kept fit_h200_std =
fu-usGQcK5EvStvNvwRSyrPhK and fit_h200_qwen = fu-ZwKNG6eg6YoLjxnaOcQdKj;
the stamps in $STAGE prove it). What a redeploy moves is the app's
production deployment VERSION (`modal app history`: v10 -> v11, one row per
deploy with its commit), so fits-spawn now records `app_id` + `app_version`
+ `deploy_commit` on every row (:func:`deployment_pin`, the same
AppDeploymentHistory call the modal CLI makes) and :func:`same_app_version`
refuses a successor whose live version differs. A row WITHOUT the
deployment pin (spawned before it existed) is refused too — the function id
alone proves nothing — unless the operator opts in with
`--infer-legacy-app-version`, which places EACH such row's `spawn_time` in
the app's deployment history (:func:`version_at`: the version whose deploy
immediately precedes the spawn) and requires the live production version to
equal it. Per row, from the authoritative history — never one number for
the whole manifest: a merged manifest can hold pin-less rows from different
deployments, and a scalar override would license a cross-version resume for
the older ones. The fit container itself does the resuming (the driver's
fitckpt seam: ckpt/<cell>/ckpt.pt on the volume).
Every successor is recorded on the cell (`successors: [{call_id,
predecessor_call_id, cause, spawn_time, app, function, function_id, app_id,
app_version, deploy_commit}]`) and the cell's `call_id` moves to the new
call, so the manifest stays the source of truth. Per-cell cap
`--max-successors` (default 3); at the cap the cell is a triage row.

Never respawned: a cell whose marker is complete (checked first), an
unverifiable row (no expectation to validate a completion against — re-run
fits-spawn), a row without a spawn pin (same fix), and a call that RETURNED
without a valid marker (`done` — the fit itself reported a failure or
no-pool: a deterministic outcome to triage, not a continuation).

The continuation drill (scripts/sdfbase/continuation_drill.py) imports
`can_respawn` / `spawn_successor` / `same_app_version` / `successor_item`
from this module, so the drill exercises the production successor path
rather than a copy of it.

CAP-RAISE CONTINUATION ROWS: a `fits-spawn --continue-cap N` row carries
`continue_cap` in its spawn pin and a `continued_from` block (the prior
fit's fingerprint + resume position); its `expect_fingerprint` is the
CONTINUED fit's (prior identity + continuation marker), so the cap-24
publication's marker reads `stale` under it — in flight while the call
runs, triage once it ends — and only the continued publication reads
`complete`. `successor_item` rebuilds such a row's item with the cap as an
8th element, so a successor of a continuation is itself the same
continuation (the container re-adopts nothing: it resumes its own later
transient checkpoint).

Terminal conditions (the manifest is the source of truth for what was
INTENDED — an unspawned cell never completes, so a launcher killed
mid-spawn shows up here as `pending-spawn` rows, the re-spawn list):

  exit 0  every manifest cell has a call id AND a fingerprint-matching
          COMPLETE marker
  exit 2  every spawned call reached a terminal state but some cells are
          not COMPLETE (failed fits, stale/foreign markers under a call
          that ended, capped/refused successors, unverifiable rows and/or
          never-spawned cells) — the re-spawn/triage list is printed
  exit 3  --max-hours elapsed with the wave still in flight

HPO TRIALS manifests (sdfbase_fits.py --mode hpo-spawn writes
doc["trials"]: model/arm/index/round rows) ride the same watcher and the
same contract, transposed: a trial counts PUBLISHED only when its
summary.json at out/hpo/<model>/<arm>_all/r<round>_t<NN>/ parses, carries a
TERMINAL trial status (ok / nan_diverged / gate_dead / floor_failed /
ratio_diverged — a diverged trial is an HPO outcome, not a failure to
respawn) AND echoes the row's ``expect`` identity (model, revision, box,
warmup levels, round, si, arm, position and the designed point — recorded
by hpo-spawn from the staged design). A summary of another design under
the same r<N>_tNN dir is `foreign`, never counted; a row without
``expect`` is unverifiable, never counted.

Usage:
  ~/venvs/modal/bin/python scripts/modal/fits_watch.py \
      [--manifest $STAGE/fits/fit_calls.json] [--poll 60]
      [--once] [--max-hours 48] [--respawn|--no-respawn]
      [--max-successors 3]

`modal` is imported only inside the volume/call probes and main(), so the
pure classification and successor functions import without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VOLUME = "twosi-fits"

#: Manifest ``app`` values whose rows MUST carry ``expect_fingerprint`` —
#: each together with its per-wave namesakes ``<app>-<wave>``: every later
#: sdfbase wave deploys under its own name, e.g. ``sdfbase-fits-ws3``, so
#: the frozen wave-A app is never redeployed; the manifests those waves
#: write are the same shape and just as strict.
STRICT_APPS = ("sdfbase-fits",)

#: Call states that END a call without a result: the two continuation
#: causes. `done` (returned) is deliberately NOT here — see the docstring.
RESPAWN_CAUSES = ("timeout", "failed")

#: Call states after which nothing more will happen to the cell on its own.
TERMINAL_CALL_STATES = ("done",) + RESPAWN_CAUSES

#: The per-row SPAWN PIN fits-spawn records on every manifest row: the
#: deployed function the cell ran on and the item fields a successor must
#: reproduce. Read from the ROW only — never from the manifest's top level,
#: which describes the last fits-spawn that wrote it.
ROW_PIN_FIELDS = ("app", "function", "function_id", "config_sha", "overwrite",
                  "checkpoint_every", "continue_cap", "continue_lr_mult",
                  "app_id", "app_version", "deploy_commit")


def is_strict_app(name) -> bool:
    """Is ``name`` a strict app (STRICT_APPS member or one of its
    ``<app>-<wave>`` namesakes)? Anything else — the twosi wave, a missing
    field — keeps the legacy existence-only reading."""
    return any(name == a or str(name).startswith(a + "-")
               for a in STRICT_APPS if name)


def respawn_hint(cell: dict) -> str:
    """The launcher flag that re-spawns THIS row on the app it was spawned
    on — a row carries ``app`` (sdfbase_fits.py records it per row), so a
    wave under an override name never gets a re-spawn aimed at the
    default app."""
    return f" --app-name {cell['app']}" if cell.get("app") else ""


def marker_path(cell: dict) -> str:
    return (f"{cell['out_root']}/cells_{cell['leg']}/"
            f"{cell['cell_id']}/COMPLETE")


def marker_verdict(raw: bytes | None, expected: str | None, *,
                   strict: bool) -> tuple[str, str]:
    """(state, detail) for one cell's marker bytes against the manifest
    row's expected fingerprint. States:

    ``complete``       marker parses and its fit_fingerprint == expected
                       (or, non-strict with no expectation: marker exists)
    ``absent``         no marker
    ``unparseable``    marker is not a JSON object
    ``no-fingerprint`` marker carries no fit_fingerprint (a pre-fingerprint
                       publication cannot be proven to be this work)
    ``stale``          marker carries ANOTHER fingerprint — a superseded
                       publication still sitting under the cell dir (the
                       refit is in flight, or failed and left it there)
    ``unverifiable``   strict manifest, row has no expect_fingerprint —
                       re-run fits-spawn so the row records one. Decided
                       BEFORE the marker is looked at: a row that can never
                       be validated is unverifiable whether or not a marker
                       happens to sit under its cell dir.
    """
    if strict and expected is None:
        return ("unverifiable", "manifest row records no "
                "expect_fingerprint — re-run fits-spawn for this cell")
    if raw is None:
        return "absent", "no COMPLETE marker"
    if expected is None:
        return "complete", "marker present (existence-only: legacy manifest)"
    try:
        marker = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return "unparseable", f"COMPLETE marker does not parse: {e}"
    if not isinstance(marker, dict):
        return "unparseable", "COMPLETE marker is not a JSON object"
    stored = marker.get("fit_fingerprint")
    if not stored:
        return ("no-fingerprint", "COMPLETE marker carries no "
                "fit_fingerprint — an unprovable publication")
    if str(stored) != str(expected):
        return ("stale", f"COMPLETE marker fingerprint {str(stored)[:12]}… "
                f"!= expected {str(expected)[:12]}… — a superseded "
                "publication under this cell dir, not this fit")
    return "complete", "fingerprint-matching COMPLETE marker"


def _read_marker(vol, cell: dict) -> bytes | None:
    """The cell's COMPLETE marker bytes off the volume, None when absent
    (the dir may not exist yet — any read failure is 'not there yet')."""
    try:
        return b"".join(vol.read_file(marker_path(cell)))
    except Exception:  # noqa: BLE001 — not there yet (or dir not created)
        return None


def classify_call(cell: dict) -> str:
    """One of: unspawned | running | done | timeout | failed | unknown.

    Verified against modal 1.5.4: ``FunctionCall.get(timeout=0)`` returns
    the result if finished; raises the BUILTIN ``TimeoutError`` while the
    call is still running; raises ``modal.exception.FunctionTimeoutError``
    when the function hit its real ``timeout=`` (a subclass of
    ``modal.exception.TimeoutError``, which is NOT the builtin); raises
    ``modal.exception.OutputExpiredError`` past the retention window; any
    other exception is the remote raise (a failed call). The order below
    is load-bearing: the modal timeout classes must be tested BEFORE the
    builtin one, and both before the generic except — a generic except
    first would file a real function timeout as `failed`.

    `done` means the call RETURNED (its fit RESULT may still be a failure —
    the COMPLETE marker is the success signal). `unknown` is fine — by
    then the COMPLETE marker decides. The error text of a timed-out or
    failed call is stored on the cell as `_call_error`."""
    import modal

    if not cell.get("call_id"):
        return "unspawned"
    try:
        fc = modal.FunctionCall.from_id(cell["call_id"])
        fc.get(timeout=0)
        return "done"
    except modal.exception.FunctionTimeoutError as e:
        cell["_call_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return "timeout"
    except modal.exception.OutputExpiredError:
        return "unknown"
    except TimeoutError:
        return "running"
    except Exception as e:  # noqa: BLE001 — remote raise == failed call
        cell["_call_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return "failed"


#: tune_arm.py's _TERMINAL_STATUSES — the only statuses run_trial writes at
#: the END of a trial. A summary carrying anything else is torn or foreign
#: (the trial itself re-runs on it), so the watcher never counts it.
#: Mirrored literally (tune_arm imports torch; the watcher must not) and
#: pinned against the source by tests/test_fits_watch.py.
TRIAL_TERMINAL_STATUSES = frozenset({"ok", "nan_diverged", "gate_dead",
                                     "floor_failed", "ratio_diverged"})

#: The ``expect`` fields a trial summary must ECHO verbatim. ``trial`` is
#: compared separately as a SUBSET (every designed axis must match; the
#: summary may carry more). ``config_sha`` rides the row for provenance
#: only — the summary does not carry it (tune_arm refused any other
#: --config before training, and its fit fingerprint hashes it).
TRIAL_EXPECT_KEYS = ("model", "model_revision", "arm", "si", "round",
                     "position", "box", "warmup_levels")


#: Consecutive polls that must agree on terminal-bad (nothing in flight,
#: rows missing) before main() declares WAVE-INCOMPLETE — see the note in
#: main(): a publication can become readable a little after its call
#: returns, and one poll cannot tell that from a lost result.
TERMINAL_BAD_CONFIRM_POLLS = 2


def summary_path(row: dict) -> str:
    return (f"out/hpo/{row['model']}/{row['arm']}_all/"
            f"r{int(row['round'])}_t{int(row['index']):02d}/summary.json")


def trial_verdict(raw: bytes | None, expect: dict | None
                  ) -> tuple[str, str]:
    """(state, detail) for one trial's summary.json bytes against the
    manifest row's ``expect`` identity. States:

    ``published``     parses, terminal status, identity matches the design
    ``absent``        no summary.json on the volume yet
    ``unparseable``   bytes that are not a JSON object
    ``non-terminal``  status outside TRIAL_TERMINAL_STATUSES (a torn write,
                      or a summary a still-running trial seeded)
    ``foreign``       identity differs from the design — a trial of ANOTHER
                      design (older config, other box) under this dir
    ``unverifiable``  the row records no ``expect`` — it can never be
                      validated; re-run hpo-spawn to record one
    """
    if expect is None:
        return ("unverifiable", "manifest row carries no expect identity "
                "— re-run hpo-spawn for this trial")
    if raw is None:
        return "absent", "no summary.json on the volume"
    try:
        summary = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return "unparseable", "summary.json is not JSON"
    if not isinstance(summary, dict):
        return "unparseable", "summary.json is not a JSON object"
    status = summary.get("status")
    if status not in TRIAL_TERMINAL_STATUSES:
        return ("non-terminal", f"summary status {status!r} is not a "
                "terminal trial outcome (torn write or in flight)")
    bad = [k for k in TRIAL_EXPECT_KEYS
           if k in expect and summary.get(k) != expect[k]]
    got = summary.get("trial") or {}
    if not isinstance(got, dict):
        got = {}
    bad += [f"trial.{k}" for k, v in (expect.get("trial") or {}).items()
            if got.get(k) != v]
    if bad:
        return ("foreign", f"summary identity differs from the design on "
                f"{bad} — a trial of another design under this dir")
    return ("published", f"terminal summary ({status}) matching the "
            "design identity")


def _read_summary(vol, row: dict) -> bytes | None:
    """The trial's summary.json bytes off the volume, None when absent."""
    try:
        return b"".join(vol.read_file(summary_path(row)))
    except Exception:  # noqa: BLE001 — not there yet (or dir not created)
        return None


def assess_trials(rows: list[dict], *, read_summary, call_state) -> dict:
    """One status pass over a TRIALS manifest — pure given the two probes
    (the assess() shape: ``n_complete`` counts published trials,
    ``n_unspawned``, ``needs_attention``, ``all_calls_terminal``,
    ``lines``). A published trial is done whatever its call did later; a
    non-published one is judged by its call: running/unknown holds the
    wave open; an ENDED call — done, timeout or failed, i.e.
    TERMINAL_CALL_STATES, the same reading classify_call gives the cells
    path — with no valid summary needs attention (a real function
    timeout is `timeout`, never "still running": the ctl-scale merge
    review caught this path knowing only done/failed); unspawned needs
    a re-spawn on the app the row names. Trials are NEVER respawned
    here — no continuation exists for an HPO trial — so every ended
    row is a triage line, not a successor."""
    n_complete = 0
    n_unspawned = 0
    needs_attention: list[str] = []
    all_calls_terminal = True
    lines: dict[str, str] = {}
    for r in rows:
        tid = f"{r['model']}/{r['arm']}/r{int(r['round'])}_t{int(r['index']):02d}"
        verdict, detail = trial_verdict(read_summary(r), r.get("expect"))
        if verdict == "published":
            n_complete += 1
            state = "published"
        elif verdict == "unverifiable":
            state = verdict
            needs_attention.append(f"{tid}: {detail}")
        else:
            call = call_state(r)
            if call == "unspawned":
                n_unspawned += 1
                state = "unspawned"
                needs_attention.append(f"{tid}: NEVER SPAWNED — re-run "
                                       f"hpo-spawn{respawn_hint(r)} for it")
            elif call in TERMINAL_CALL_STATES:
                state = f"call {call}, summary {verdict}"
                needs_attention.append(
                    f"{tid}: call {call} but {detail}"
                    + (f" ({r.get('_call_error')})"
                       if r.get("_call_error") else ""))
            else:
                all_calls_terminal = False
                state = f"{call}" + (f" (summary {verdict})"
                                     if verdict != "absent" else "")
        lines[tid] = f"[hpo-watch] {tid}: {state}"
    return {"n_complete": n_complete, "n_unspawned": n_unspawned,
            "needs_attention": needs_attention,
            "all_calls_terminal": all_calls_terminal, "lines": lines}


def manifest_kind(doc: dict) -> str:
    """``trials`` for an hpo-spawn manifest, ``cells`` otherwise — a doc
    carrying both is a cells manifest (the historical shape wins; nothing
    writes both)."""
    if doc.get("cells") is None and doc.get("trials") is not None:
        return "trials"
    return "cells"


#: The call probe under its historical name (assess's default probe).
_call_state = classify_call


def assess(cells: list[dict], *, strict: bool, read_marker, call_state,
           respawn=None) -> dict:
    """One status pass over the manifest rows — pure given the probes, so
    the classification is testable off-Modal. Returns ``n_complete``,
    ``n_unspawned``, ``needs_attention`` (triage lines),
    ``all_calls_terminal``, ``lines`` (one status line per cell) and
    ``calls`` (the probed call state per cell id).

    ``strict`` is the manifest-level reading (its ``app``); a row that
    records its own ``app`` is judged by THAT (one manifest can carry rows
    spawned on different apps — fits-spawn keeps foreign rows when it
    rewrites), and its re-spawn hint names that app.

    ``respawn`` (optional) is the continuation hook: called as
    ``respawn(cell, call, cid, marker) -> (spawned: bool, note: str)`` for a
    cell whose call ended in a RESPAWN_CAUSES state without a complete
    marker, ``marker`` being that cell's :func:`marker_verdict` state (so
    the hook's own can_respawn sees the same marker reading this pass did).
    When it spawned, the cell is back in flight (its line shows ``note``
    and it does not hold the wave as terminal-bad); when it did not, the
    cell is a triage row whose line carries ``note`` as the reason. Without
    the hook, every ended call without a valid marker is a triage row."""
    n_complete = 0
    n_unspawned = 0
    needs_attention: list[str] = []
    all_calls_terminal = True
    lines: dict[str, str] = {}
    calls: dict[str, str] = {}
    for c in cells:
        cid = f"{c['model']}/{c['cell_id']}/{c['leg']}"
        row_strict = is_strict_app(c["app"]) if c.get("app") else strict
        verdict, detail = marker_verdict(read_marker(c),
                                         c.get("expect_fingerprint"),
                                         strict=row_strict)
        if verdict == "complete":
            n_complete += 1
            state = "complete"
        elif verdict == "unverifiable":
            state = verdict
            needs_attention.append(f"{cid}: {detail}")
        else:
            call = call_state(c)
            calls[cid] = call
            err = (f" ({c.get('_call_error')})"
                   if c.get("_call_error") else "")
            if call == "unspawned":
                n_unspawned += 1
                state = "unspawned"
                needs_attention.append(f"{cid}: NEVER SPAWNED — re-run "
                                       f"fits-spawn{respawn_hint(c)} for it")
            elif call in RESPAWN_CAUSES and respawn is not None:
                spawned, note = respawn(c, call, cid, verdict)
                if spawned:
                    # the successor is in flight: the same picture as a
                    # running refit over a stale/absent marker
                    all_calls_terminal = False
                    state = note
                else:
                    state = f"call {call}, marker {verdict}"
                    needs_attention.append(
                        f"{cid}: call {call} but {detail}{err} — not "
                        f"respawned: {note}")
            elif call in TERMINAL_CALL_STATES:
                state = f"call {call}, marker {verdict}"
                needs_attention.append(f"{cid}: call {call} but {detail}{err}")
            else:
                # running / unknown: a stale or absent marker is the
                # expected picture of a refit in flight
                all_calls_terminal = False
                state = f"{call}" + (f" (marker {verdict})"
                                     if verdict != "absent" else "")
        n_succ = len(c.get("successors") or [])
        if n_succ and not state.startswith("successor #"):
            state += f" (successor #{n_succ})"
        lines[cid] = f"[fits-watch] {cid}: {state}"
    return {"n_complete": n_complete, "n_unspawned": n_unspawned,
            "needs_attention": needs_attention,
            "all_calls_terminal": all_calls_terminal, "lines": lines,
            "calls": calls}


# --------------------------------------------------------------------------- #
# Successor spawning — pure/injectable (the drill imports these)               #
# --------------------------------------------------------------------------- #


def row_pin(cell: dict) -> tuple[dict | None, str]:
    """The row's own spawn pin (:data:`ROW_PIN_FIELDS`) or ``(None, why)``
    for a row that records no `function` — a manifest row written before
    fits-spawn stamped the pin per row. The manifest's top-level values are
    deliberately NOT a fallback: they describe the last fits-spawn that
    wrote the file, which on a merged multi-model manifest is the WRONG
    function/config for every other model's rows."""
    if cell.get("function"):
        return {k: cell.get(k) for k in ROW_PIN_FIELDS}, "row"
    return None, ("manifest row records no spawn pin (function / "
                  "function_id / config_sha per row) — the manifest's "
                  "top-level pin describes only the last fits-spawn that "
                  "wrote it; re-run fits-spawn for this cell")


def successor_item(cell: dict) -> tuple:
    """EXACTLY the item tuple `fits-spawn` passes to ``fn.spawn`` —
    (model_key, si, arm, expect_sha, overwrite, out_root, checkpoint_every)
    — rebuilt from the ROW'S OWN pin, so the successor is the same fit
    identity (same fingerprint, same out_root, same checkpoint cadence) and
    the container resumes the predecessor's volume checkpoint instead of
    starting a new fit. A continuation row (``continue_cap`` in the pin)
    gets the cap as an 8th element, and a non-1.0 ``continue_lr_mult`` as
    a 9th; plain rows keep the 7-tuple and plain continuation rows the
    8-tuple, byte for byte. Raises ``ValueError`` on a pin-less row
    (callers check :func:`row_pin` first)."""
    pin, why = row_pin(cell)
    if pin is None:
        raise ValueError(why)
    item = (cell["model"], cell["si"], cell["arm"], pin["config_sha"],
            bool(pin.get("overwrite", False)), cell["out_root"],
            int(pin.get("checkpoint_every") or 0))
    if pin.get("continue_cap"):
        item += (int(pin["continue_cap"]),)
        m = pin.get("continue_lr_mult")
        if m is not None and float(m) != 1.0:
            item += (float(m),)
    return item


def can_respawn(cell: dict, cause: str, *, max_successors: int,
                marker: str = "absent",
                arms: tuple = ()) -> tuple[bool, str]:
    """Whether a successor may be spawned for `cell` given how its call
    ended and what its COMPLETE marker says (``marker`` = the
    :func:`marker_verdict` state, as read by the caller's own pass). Only a
    TIMEOUT or a FAILED call is a continuation event; a call that returned
    (`done`) without a COMPLETE marker is a deterministic outcome to
    triage; a COMPLETE cell is finished; an unverifiable row can never be
    harvested; and the per-cell cap bounds the number of successors.

    ``arms`` is the LAUNCHER'S arm list: when given, a row whose arm is not
    in it is never continued. A descoped arm's cancelled call classifies as
    `failed`, which is a continuation cause, so without this a watcher
    pointed at a manifest that still carries those rows revives exactly the
    work that was descoped (2026-09-12: four PSR cells, cancelled on a user
    ruling, respawned within seconds). Keyed on the launcher's arms rather
    than on any particular arm name, so the next descoped arm is covered
    with no code change; empty means unrestricted, the old behaviour."""
    if arms and cell.get("arm") not in arms:
        return False, (f"arm {cell.get('arm')!r} is not one the launcher "
                       f"runs ({'/'.join(arms)}) — descoped rows are never "
                       "continued")
    if marker == "complete":
        return False, "cell is COMPLETE — nothing to continue"
    if marker == "unverifiable":
        return False, ("manifest row records no expect_fingerprint — its "
                       "completion could never be validated; re-run "
                       "fits-spawn for this cell")
    if cause not in RESPAWN_CAUSES:
        return False, (f"call state {cause!r} is not a continuation event "
                       f"(only {'/'.join(RESPAWN_CAUSES)} are)")
    n = len(cell.get("successors") or [])
    if n >= int(max_successors):
        return False, (f"successor cap reached ({n}/{max_successors}) — "
                       "triage by hand before any further continuation")
    return True, "ok"


def deployment_pin(app_name: str) -> dict:
    """What a `modal deploy` actually changes: the app's PRODUCTION
    deployment version, from the same AppDeploymentHistory call the modal
    CLI's `modal app history` makes (modal 1.5.4:
    ``production_app_version`` + the head record's commit), plus the app
    id (constant across redeploys, recorded so a same-named app in another
    environment cannot alias). Returns ``{"app_id", "app_version",
    "deploy_commit", "deploy_commit_dirty", "deployed_at"}``; raises when
    the app cannot be resolved — the caller decides what an unknown
    deployment means (fits-spawn: warn + a pin-less row; the watcher:
    refuse the successor)."""
    import importlib

    modal = importlib.import_module("modal")
    if getattr(modal, "__file__", None) is None and not hasattr(modal, "App"):
        # Not the SDK: with scripts/ on sys.path and no modal installed,
        # `import modal` resolves to THIS directory (scripts/modal) as a
        # namespace package. Say so, and do not leave the shadow cached
        # where a later `import modal` would find it.
        sys.modules.pop("modal", None)
        raise RuntimeError(
            "`import modal` resolved to a namespace package, not the Modal "
            "SDK — the deployment pin needs the SDK in this interpreter")
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client
    from modal_proto import api_pb2

    app_id = str(modal.App.lookup(app_name).app_id)

    async def _history(aid: str):
        client = await _Client.from_env()
        return await client.stub.AppDeploymentHistory(
            api_pb2.AppDeploymentHistoryRequest(app_id=aid))

    resp = synchronizer.create_blocking(_history)(app_id)
    version = int(resp.production_app_version)
    head = None
    history = []
    for rec in resp.app_deployment_histories:
        if int(rec.version) == version:
            head = rec
        history.append({"version": int(rec.version),
                        "deployed_at": (float(rec.deployed_at)
                                        if rec.deployed_at else None)})
    commit = (head.commit_info.commit_hash if head is not None else "") or None
    return {"app_id": app_id, "app_version": version,
            "deploy_commit": (commit[:12] if commit else None),
            "deploy_commit_dirty": (bool(head.commit_info.dirty)
                                    if head is not None else None),
            "deployed_at": (float(head.deployed_at) if head is not None
                            else None),
            "history": sorted(history, key=lambda h: h["version"])}


def version_at(history, spawn_time) -> int | None:
    """The app version in PRODUCTION when a call was spawned: the history
    record with the greatest ``deployed_at`` <= the spawn instant (one
    record per `modal deploy`, :func:`deployment_pin`'s ``history``). None
    when the spawn time or the history cannot place it (unparseable time,
    no dated records, a spawn before the first recorded deploy) — the
    caller refuses; it never guesses."""
    if not spawn_time or not isinstance(history, list):
        return None
    try:
        t = datetime.fromisoformat(str(spawn_time).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    ts = t.timestamp()
    best = None
    for rec in history:
        try:
            v, at = int(rec["version"]), rec.get("deployed_at")
        except (KeyError, TypeError, ValueError):
            continue
        if at is None or float(at) > ts:
            continue
        if best is None or float(at) > best[1]:
            best = (v, float(at))
    return None if best is None else best[0]


def same_app_version(cell: dict, fn, live_deployment: dict | None = None,
                     infer_legacy: bool = False) -> tuple[bool, str]:
    """The version pin: a successor resumes a volume checkpoint, so it
    must run the SAME deployed code as its predecessor. Two layers:

    - the row's `function_id` (recorded at spawn) must equal the live
      deployed function's id — necessary, NOT sufficient: the id does not
      change on redeploy, so this alone never proves the code is the same.
    - the row's `app_version` (+ `app_id`) must equal the app's live
      production deployment (`live_deployment`, :func:`deployment_pin`).
      A row with NO `app_version` predates the deployment pin and is
      refused — unless `infer_legacy` is set, in which case THIS row's
      version is read from its own `spawn_time` against the live
      deployment's `history` (:func:`version_at`) and the live version must
      equal it. Per row: a merged manifest can carry pin-less rows from
      different deployments, so no single assumed version is ever applied
      across rows.

    `fn` / `live_deployment` are only consulted when the row carries the
    corresponding pin (a pin-less row never needs a hydrated handle)."""
    pinned = cell.get("function_id")
    if not pinned:
        return False, ("manifest row predates the version pin (no "
                       "function_id recorded at spawn) — respawn by hand")
    live = getattr(fn, "object_id", None)
    if live != pinned:
        return False, (f"the app was redeployed since the wave started "
                       f"(row function_id {pinned} != deployed {live}); a "
                       "successor would resume a volume checkpoint under "
                       "different code — refusing")
    row_version = cell.get("app_version")
    if row_version is None and not infer_legacy:
        return False, (
            "manifest row carries no deployment pin (app_version): the "
            "function id above does not change on redeploy, so it cannot "
            "prove the deployed code is the predecessor's — refusing; "
            "relaunch the watcher with --infer-legacy-app-version to place "
            "each such row's spawn_time in the app's deployment history")
    if not isinstance(live_deployment, dict) or \
            live_deployment.get("app_version") is None:
        return False, ("the app's live deployment version could not be "
                       "read — a successor cannot be proven to run the "
                       "predecessor's code; refusing")
    if row_version is None:
        inferred = version_at(live_deployment.get("history"),
                              cell.get("spawn_time"))
        if inferred is None:
            return False, (
                f"manifest row carries no deployment pin and its spawn_time "
                f"{cell.get('spawn_time')!r} cannot be placed in the app's "
                "deployment history — refusing")
        row_version = int(inferred)
        source = f"inferred from spawn_time {cell.get('spawn_time')}:"
    else:
        row_version = int(row_version)
        source = "row app_version"
    live_version = int(live_deployment["app_version"])
    row_app_id, live_app_id = cell.get("app_id"), live_deployment.get("app_id")
    if row_app_id and live_app_id and str(row_app_id) != str(live_app_id):
        return False, (f"row app_id {row_app_id} != the live app "
                       f"{live_app_id} — a different app under this name; "
                       "refusing")
    if live_version != row_version:
        return False, (f"the app was redeployed since the wave started "
                       f"({source} v{row_version} != live production "
                       f"v{live_version}, commit "
                       f"{live_deployment.get('deploy_commit')}); a "
                       "successor would resume a volume checkpoint under "
                       "different code — refusing")
    return True, "ok"


def spawn_successor(cell: dict, cause: str, fn, now_iso: str,
                    live_deployment: dict | None = None) -> dict:
    """Spawn the successor call on `fn` (the row's own deployed function)
    and record it on the cell: appends the record to `cell["successors"]`,
    then moves `call_id` / `spawn_time` to the new call and restates
    `status`. The predecessor's error text (classify_call's `_call_error`)
    travels into the record, never stays on the cell (it would mislabel
    the new call). `live_deployment` (the version the successor runs on)
    is recorded when known. Returns the record."""
    pin, _src = row_pin(cell)
    item = successor_item(cell)
    call = fn.spawn(item)
    new_id = call.object_id
    dep = live_deployment or {}
    rec = {
        "call_id": new_id,
        "predecessor_call_id": cell.get("call_id"),
        "cause": cause,
        "spawn_time": now_iso,
        "app": pin["app"],
        "function": pin["function"],
        "function_id": getattr(fn, "object_id", None),
        "app_id": dep.get("app_id"),
        "app_version": dep.get("app_version"),
        "deploy_commit": dep.get("deploy_commit"),
    }
    err = cell.pop("_call_error", None)
    if err:
        rec["predecessor_error"] = err
    cell.setdefault("successors", []).append(rec)
    n = len(cell["successors"])
    cell["call_id"] = new_id
    cell["spawn_time"] = now_iso
    cell["status"] = f"successor #{n} spawned ({cause})"
    return rec


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(doc: dict, mpath: Path) -> None:
    """Atomic rewrite (tmp + os.replace) — the sdfbase_fits._flush_manifest
    shape, so a watcher killed mid-write can never leave a torn manifest."""
    doc["written_at"] = _now_iso()
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, mpath)


class _FnCache:
    """One hydrated deployed-function handle per (app, function) — the
    ROW'S pin, so a merged manifest's gemma and qwen rows each get their
    own handle — fetched lazily on the first successor decision that
    needs the version pin."""

    def __init__(self):
        self._fns: dict[tuple[str, str], object] = {}

    def get(self, app: str, function: str):
        import modal

        key = (str(app), str(function))
        if key not in self._fns:
            fn = modal.Function.from_name(*key)
            fn.hydrate()
            self._fns[key] = fn
        return self._fns[key]


class _DeploymentCache:
    """The live deployment pin per app (:func:`deployment_pin`), read at
    every successor decision that needs it — never cached across polls,
    since a redeploy between two successor events must be seen. A lookup
    failure yields None (the version pin then refuses) and keeps the
    error for the log line."""

    def __init__(self, fetch=None):
        self._fetch = fetch or deployment_pin
        self.last_error: dict[str, str] = {}

    def get(self, app: str) -> dict | None:
        try:
            dep = self._fetch(str(app))
        except Exception as e:  # noqa: BLE001 — refuse, never guess
            self.last_error[str(app)] = f"{type(e).__name__}: {e}"
            return None
        self.last_error.pop(str(app), None)
        return dep


def main() -> int:
    import modal

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="$STAGE/fits/fit_calls.json")
    ap.add_argument("--poll", type=int, default=60,
                    help="seconds between polls (default 60)")
    ap.add_argument("--once", action="store_true",
                    help="print one status pass and exit 0/2 by state")
    ap.add_argument("--max-hours", type=float, default=48.0,
                    help="hard stop so nothing polls forever (default 48)")
    ap.add_argument("--respawn", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="spawn ONE successor call per timeout/failure of "
                         "a non-COMPLETE cell, same item + same app "
                         "version (default on; --no-respawn to only watch)")
    ap.add_argument("--arms", default="",
                    help="comma-separated arms the LAUNCHER runs; a row "
                         "whose arm is not listed is never respawned "
                         "(empty = unrestricted). Pass the launcher's own "
                         "ARMS so a descoped arm cannot be revived.")
    ap.add_argument("--max-successors", type=int, default=3,
                    help="per-cell successor cap (default 3); at the cap "
                         "the cell becomes a triage row")
    ap.add_argument("--infer-legacy-app-version", action="store_true",
                    help="for manifest rows spawned BEFORE the deployment "
                         "pin existed (no app_version on the row): infer "
                         "EACH row's version from its own spawn_time against "
                         "the app's deployment history, and refuse its "
                         "successor unless the live production version "
                         "equals it. Never needed for rows that carry the "
                         "pin; never one number for the whole manifest.")
    args = ap.parse_args()
    #: the launcher's arms, if it told us; empty = unrestricted.
    ARMS = tuple(a.strip() for a in args.arms.split(",") if a.strip())


    # a site placeholder ($STAGE, the staging directory) expands through sisteer.site from this checkout; an unset one
    # refuses by name
    _src = str(Path(__file__).resolve().parents[2] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from sisteer import site as _site
    mpath = _site.expand_path(args.manifest, key="--manifest")
    vol = modal.Volume.from_name(VOLUME)
    fns = _FnCache()
    deps = _DeploymentCache()
    t0 = time.time()
    last_line: dict[str, str] = {}
    announced_mode = False
    # A call's "done" and its publication becoming readable off the volume
    # are two systems with no ordering the client can see: the HPO
    # wave's first --once pass (2026-09-03 03:45Z) read one trial as "call
    # done, summary absent" and the next pass, under a minute later, as
    # published. So terminal-bad (nothing running, something missing) is
    # declared only when TWO consecutive polls agree; one observation is
    # announced and re-checked. --once keeps its single-pass reading.
    terminal_bad_polls = 0

    def say(key: str, line: str) -> None:
        if last_line.get(key) != line:
            print(line, flush=True)
            last_line[key] = line

    while True:
        try:
            doc = json.loads(mpath.read_text())
        except (OSError, ValueError) as e:
            print(f"[fits-watch] manifest {mpath} unreadable: {e}",
                  flush=True)
            return 2
        kind = manifest_kind(doc)
        rows = doc.get(kind) or []
        tag = "hpo-watch" if kind == "trials" else "fits-watch"
        if not rows:
            print(f"[{tag}] manifest {mpath} lists no {kind}", flush=True)
            return 2
        strict = is_strict_app(doc.get("app"))
        if not announced_mode:
            if kind == "trials":
                print("[hpo-watch] PUBLISHED = terminal summary.json "
                      "echoing the row's design identity (manifest rows "
                      "carry expect)", flush=True)
            else:
                print("[fits-watch] COMPLETE = "
                      + ("fingerprint-matching marker (manifest rows carry "
                         "expect_fingerprint)" if strict else
                         "marker existence (legacy manifest without "
                         "expected fingerprints)"), flush=True)
            apps = sorted({c.get("app") or doc.get("app") for c in rows}
                          - {None})
            print(f"[{tag}] app(s) the rows were spawned on: "
                  f"{', '.join(apps) or '(unrecorded)'}", flush=True)
            announced_mode = True
        spawned_this_pass: list[str] = []

        def respawn(c: dict, call: str, cid: str,
                    marker: str) -> tuple[bool, str]:
            """The continuation hook (assess calls it for a timeout/failed
            call without a complete marker): cap + the row's spawn pin +
            version pin, then ONE successor on the ROW'S deployed function,
            manifest rewritten atomically before returning."""
            if not args.respawn:
                return False, "respawning disabled (--no-respawn)"
            ok, why = can_respawn(c, call, max_successors=args.max_successors,
                                  marker=marker, arms=ARMS)
            fn = None
            live_dep = None
            if ok:
                pin, why = row_pin(c)
                ok = pin is not None
            if ok:
                fn = (fns.get(pin["app"], pin["function"])
                      if pin.get("function_id") else None)
                # the deployment pin is read only when the function-id
                # layer holds and the row (or --infer-legacy-app-version)
                # can name a version to compare against
                if (fn is not None and getattr(fn, "object_id", None)
                        == pin.get("function_id")
                        and (pin.get("app_version") is not None
                             or args.infer_legacy_app_version)):
                    live_dep = deps.get(pin["app"])
                    if live_dep is None:
                        err = deps.last_error.get(str(pin["app"]))
                        say(f"{cid}/deployment",
                            f"[fits-watch] deployment lookup failed for "
                            f"{pin['app']}: {err}")
                ok, why = same_app_version(
                    c, fn, live_deployment=live_dep,
                    infer_legacy=args.infer_legacy_app_version)
            if not ok:
                say(f"{cid}/no-respawn", f"[fits-watch] NO-RESPAWN {cid}: {why}")
                return False, why
            rec = spawn_successor(c, call, fn, _now_iso(),
                                  live_deployment=live_dep)
            rec["row_version_source"] = (
                "row" if c.get("app_version") is not None
                else "inferred_from_spawn_time")
            write_manifest(doc, mpath)
            print(f"[fits-watch] SUCCESSOR {cid}: cause={call} "
                  f"#{len(c['successors'])} -> {rec['call_id']} "
                  f"(on {rec['app']}/{rec['function']} "
                  f"v{rec.get('app_version')} {rec.get('deploy_commit')})",
                  flush=True)
            spawned_this_pass.append(rec["call_id"])
            return True, c["status"]

        if kind == "trials":
            res = assess_trials(rows,
                                read_summary=lambda r: _read_summary(vol, r),
                                call_state=_call_state)
            done_word, done_msg, missing_msg = (
                "PUBLISHED",
                "ALL-PUBLISHED — every manifest trial was spawned and "
                "published a terminal summary.json echoing its design "
                "identity",
                "these trials lack a valid published summary:")
        else:
            res = assess(rows, strict=strict,
                         read_marker=lambda c: _read_marker(vol, c),
                         call_state=_call_state, respawn=respawn)
            done_word, done_msg, missing_msg = (
                "COMPLETE",
                "ALL-COMPLETE — every manifest cell was spawned and "
                "published a COMPLETE marker carrying its expected fit "
                "fingerprint",
                "these cells lack a valid COMPLETE marker:")
        for cid, line in res["lines"].items():
            say(cid, line)
        needs_attention = res["needs_attention"]
        summary = (f"[{tag}] {res['n_complete']}/{len(rows)} {done_word}, "
                   f"{res['n_unspawned']} unspawned, "
                   f"{len(needs_attention)} needing attention")
        if spawned_this_pass:
            summary += (f", {len(spawned_this_pass)} successor(s) spawned "
                        "this pass")
        say("_summary", summary)
        if res["n_complete"] == len(rows):
            print(f"[{tag}] {done_msg}", flush=True)
            return 0
        if res["all_calls_terminal"] and needs_attention:
            # Nothing is running and something is missing: terminal-bad —
            # once confirmed by the next poll (publication visibility can
            # trail a call's return; see terminal_bad_polls above).
            terminal_bad_polls += 1
            if args.once or terminal_bad_polls >= TERMINAL_BAD_CONFIRM_POLLS:
                print(f"[{tag}] WAVE-INCOMPLETE — no calls in flight and "
                      f"{missing_msg}", flush=True)
                for line in needs_attention:
                    print(f"[{tag}]   {line}", flush=True)
                return 2
            print(f"[{tag}] terminal-bad observed ({len(needs_attention)} "
                  "row(s) missing with no call in flight) — confirming on "
                  "the next poll before declaring WAVE-INCOMPLETE",
                  flush=True)
        else:
            terminal_bad_polls = 0
        if args.once:
            return 0 if not needs_attention else 2
        if (time.time() - t0) / 3600 > args.max_hours:
            print(f"[{tag}] WATCH-TIMEOUT after {args.max_hours}h "
                  "with the wave still in flight — re-arm the watcher",
                  flush=True)
            return 3
        time.sleep(max(int(args.poll), 10))


if __name__ == "__main__":
    sys.exit(main())
