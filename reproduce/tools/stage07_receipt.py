#!/usr/bin/env python3
"""reproduce/tools/stage07_receipt.py — stage 07's RECEIPT: what a Modal run of the FULL BigCodeBench and BFCL sets (the paper's: 1,140 and
3,981 items under the `protocol` decoding) actually produced, from the two sources this host can reach, written under
<state>/07_store/ and summed up in receipt.json.

    python reproduce/tools/stage07_receipt.py --state-root S --view V --sha <40 hex> --cells <launcher tokens> --view-cells <table names>
                                              [--run-identity F] [--override-id ID] [--run-report F] [--app NAME] [--controls bigcodebench_full,bfcl_full]
                                              [--volume twosi-fits] [--environment ENV] [--modal BIN] [--dry-run]

Nothing of a run lands on this host: the stores sit under contracts.VOL_GEN_ROOT on the Modal volume (done markers included), and the launcher's
exit code alone says nothing about WHICH store satisfied it. So the receipt reads (0) THE LAUNCH'S IDENTITY, <state>/07_run_identity.json,
which `07 --phase run` writes beside the captured stdout BEFORE the launcher runs: the tree the launcher got as --sha,
the cells by both names, the object id of every fitted cell AT THAT TREE (the importer's plan over the committed objects, from the clean
checkout the launch required), the app, the volume and Modal environment, the run-overrides document (OVERRIDE_ID, its sha256, its path) and
the expect-hashes the run was held to. The receipt VERIFIES that file against its own call — `--sha`, `--view`, `--cells`, `--view-cells`,
`--volume`, `--environment`, `--controls`, and `--app` / `--override-id` when passed — refusing by name on any disagreement (a receipt taken
under another overrides document, another HEAD, other cells, other controls or another environment would relabel the execution; a launch
identity whose `sampling` is not the paper's `protocol` mode is refused the same way), and COPIES the launch's identity into the receipt:
`slug` (the launcher's registry row: the key on the wave, `<key>-<view>` on a dose view), `sampling`, `objects`, `run_overrides_id` /
`_sha256` / `_path`, `expect_hashes` (what the run was held to; never this call's state file) — so nothing in the receipt is this call's
reading of a later state; (1) the launcher's own RECs, captured by `07 --phase run` into
<state>/07_run_report.log (one JSON line per container, printed after `[tinycap] N/N container(s) OK`) — read ONCE, as one snapshot of bytes:
the snapshot is what is parsed, what is copied to <state>/07_store/run_report.log and what `run_report_sha256` hashes, and a live log (or a
live identity file) that no longer hashes to its snapshot when the receipt is about to be written (a `--phase run` wrote it during the pulls)
REFUSES, since the pulls could then be another execution's — for EVERY run cell a REC whose `status` is OK (`controls_still_remaining == []`,
`done_markers` naming both tasks' markers, `resume_after.controls[<control>].done_in` the trusted store key, `resume_after.expected.tree_sha`
the tree, `.cell` the token, `sha` the tree's short sha, `out` the store root under the volume's generation root, `sampling_mode` the paper's
`protocol` — the launcher stamps EVERY REC, an OK run's and a DONE reuse's alike, with the mode its plan resolved (SAMPLING_MODE_KEY; the
plan literal omits the key only for its `card` default, which the REC then spells out), so a REC reading `card`, or missing the key (an
older launcher's), is a run under the model card's sampling, not the paper's decoding, and is refused), and a REC for NO cell
outside `--cells` (the receipt is the whole execution's evidence, never a subset of it); and (2) a Modal-volume receipt pulled through the full22 stager's VolumeCLI, every
command scoped to the Modal ENVIRONMENT the launcher ran under (`--environment`, default $SISTEER_MODAL_ENVIRONMENT; a volume name is per
environment) — EVERY container path the REC names (`/vol/...`) converted to the volume-root path the `modal volume` CLI takes by a CHECKED
conversion, the original kept for every comparison and in the record: per run cell `<out>/cells/<done_in>/{done,summary}/
<marker>` for both tasks, `<out>/<cell>/controls_summary.json` (the per-cell merged file sits under the CELL id), the attempt pointers at
`<out>/watch/attempt_<fcid>.json`, BESIDE cells/ (tinycap_served.attempt_pointer_path): the REC's own `sandbox_pointer` pulled by exact path and
checked field by field against the REC (run_id, cell, function_call_id, out_dir, tree_sha, sandbox_record), and every OTHER attempt_*.json the
store root holds pulled under `prior_attempts`; and the completion records at `<out>/watch/completion_<fcid>.json` (tinycap_served.
completion_record_path; the paragraph on per-control attribution below). Checks on the pulled files: each done marker's `task`, `n_items ==
the frozen FULL item count` (contracts.BCB_FULL_N 1,140 / BFCL_FULL_N 3,981 — the launcher's FULL_ITEM_COUNT_OF_TASK; a limited run writes
partial/, never done/), the summary's `n == n_items`, no `partial/<marker>` beside the done marker, the pulled store key == the REC's `done_in`.

The launcher's REC branches: a FIRST attempt drives both controls, is `sandboxed: true` and carries `sandbox_record` +
`sandbox_pointer` — the pointer is pulled and verified. A BFCL-ONLY RESUMPTION (BigCodeBench already validly done; `driven = remaining`
leaves no sandboxed control) carries `sandboxed: false`, no pointer and no record but still its `sandbox_owner` — accepted, its call id
taken from the owner and its BigCodeBench attempt's pointer found in the `watch/` listing. A relaunch whose every task is validly done
returns `status: DONE` with `resume` / `controls_remaining` and no owner at all — accepted as REUSE (no new execution): the stores it credits
are pulled and checked exactly as above, its pointers are the earlier attempts' (recorded under `reused_attempts`), and THEIR call ids are the
execution the receipt names (`modal_call_ids`, with `execution: reuse` kept).
The pointer(s) a recovered run is CREDITED WITH are evidence only when they are this cell's, under this out dir, AT THIS TREE
(`tree_sha == --sha`, the tree the REC's resume block was held to), naming a call id and a run id, with the pointer file named for its call id
and `sandbox_record` in the launcher's own layout (`<out>/watch/sandboxes_<fcid>.jsonl`), the REC's model AND its mode, and — for BOTH
recovered shapes — `bigcodebench_full` among the controls they drove: the launcher writes a pointer per attempt into the same cell's watch/
whatever suite or mode it drove (the reduced `bigcodebench_200`, the `:open` mode), so a same-tree pointer that did not drive the full
BigCodeBench in the REC's mode cannot account for the full-set store a DONE relaunch reuses, and crediting it would attribute the result to
an unrelated attempt. A pointer for the cell at ANOTHER tree is history (`other_tree_attempts`), never the execution credited, and
a recovery with no qualifying pointer at the tree REFUSES by name (an older-tree or other-suite pointer in a persistent watch/ directory
cannot account for this run's BigCodeBench). `modal_call_ids` is never empty: an OK cell names its container's owner, a recovered cell the
attempts it credits.

PER-CONTROL ATTRIBUTION — the COMPLETION RECORDS: a pointer is written BEFORE the drive and only by a sandboxed attempt,
so a BFCL-only resumption left nothing on the volume naming the FunctionCall that produced its BFCL store, and a DONE relaunch after a
resume-then-complete sequence (one call finishes BigCodeBench, a later BFCL-only call finishes BFCL) was credited with the BigCodeBench
call alone. The launcher's solo path now writes `<out>/watch/completion_<fcid>.json` (tinycap_served.COMPLETION_FIELDS) at the END of EVERY
drive, sandboxed or not: the owner tuple, `controls` driven, `remaining` (the controls not validly done when the attempt started), `completed`
({control: the store key the attempt left it done in} — the subset of `remaining` done after the drive, so no attempt claims an earlier one's
work), the REC's status, `sandboxed`, the tree. The receipt (a) pulls an OK REC's own record by exact path (`completion_record`, required of
every OK REC of this launcher) and holds it to the REC field by field — owner, out dir, tree, model, mode, `controls == controls_driven`,
status, `sandboxed`, and `completed[<control>] == done_in[<control>]` for every control it drove (an OK attempt finished what it drove); and
(b) for a RECOVERED cell requires, for EVERY control it did not drive itself (a BFCL-only resumption: BigCodeBench; a DONE reuse: both),
a prior completion record for this cell under this out dir AT THE TREE, in the REC's model and mode, named for its call id, whose
`completed[<control>]` IS the store key the REC credits (`done_in`) — crediting every such record's call id (`credited_completions`, each
with the controls it is credited for) in `modal_call_ids` beside the pointer rule above, and REFUSING by name when a control has none
(the BFCL execution a relaunch reuses is then unaccounted for; nothing is written). Every other completion record under the cell's
watch/ is history (`prior_completions`).

THE JOIN: the pointer is written BEFORE the drive, so an attempt whose standup failed leaves a pointer — one that passes
every identity check above (this cell, this out dir, the tree, the model, the mode, the full BigCodeBench among the controls it was GOING to drive)
— and no completion record: it produced nothing. The pointers a recovered cell is credited with (`reused_attempts`, `bcb_attempts`) are
therefore the identity-qualifying pointers WHOSE CALL a credited completion record names (joined by `function_call_id`; each carries the
`credited_for` of its record), and every credited producer that was sandboxed — a BigCodeBench producer always is — must have such a
pointer, the launcher having published one before its drive: a producer with none REFUSES by name. A same-tree pointer for the cell that is
not credited — by identity, or because no completion record credits its call — is history under `uncredited_attempts`, each with why, beside
`prior_attempts`; its call id is never in `modal_call_ids`. The identity rule is still checked first, so every earlier refusal holds.

The HEADLINE is DERIVED, never typed: the `none` cell's `controls["inspect/bigcodebench_full"]["acc"]` of its pulled controls_summary.json
— its pass rate on the full 1,140-item set (the merged file is `{controls: {<task>: payload}, provenance, sources}`), accepted only when the
entry's `task` and `n == 1,140` agree, its `acc`/`n` equal the pulled `summary/inspect__bigcodebench_full.json`'s, and
`sources["inspect/bigcodebench_full"]["cell_key"]` is the REC's `done_in` — so the number the verdict records is the trusted store's.
receipt.json (schema stage07-receipt/7) carries every pulled file's
volume path + sha256, the matched REC fields, the app name, the Modal environment, `modal_call_ids`, `prior_attempts`, `reused_attempts`,
`other_tree_attempts`, `uncredited_attempts`, `completions`, `credited_completions`, `prior_completions`, `run_report_sha256`
(of the snapshot), the tree sha, the headline block, the bfcl block, AND THE LAUNCH'S IDENTITY the filing derives from: `cells`
(the launcher tokens), `view_cells` (the same cells by the models table's names), `objects` (token -> the committed object id,
resolved at launch from the run's tree), `run_overrides_id` / `run_overrides_sha256` (the document the run was launched under), the
`expect_hashes` it was held to, `slug`, `sampling`, `run_identity_sha256` — so `07 --phase file` composes from the receipt alone and a later call with other
cells, another overrides document, another checkout of objects/ or another HEAD cannot re-describe the execution — and NO wall-clock
field: its bytes are the execution's evidence, and `07 --phase file` names the analysis by their sha. Any missing or inconsistent item
REFUSES by name: no receipt, no verdict command. On success the verdict command is PRINTED with the headline and the store keys filled in;
`--range` is the one blank the user fills.

A RECEIPT IS NEVER REPLACED BEFORE ITS REPLACEMENT EXISTS: the pulls land in <state>/07_store.building/, and only
after every pull, every check and the receipt's own write succeed is that directory published as <state>/07_store/ — a refusal removes the
build and leaves the published receipt, its pulled evidence and its copy of the run log exactly as they were (after a `--phase run` has
overwritten the live log, that copy is the only evidence of the earlier execution). A previous receipt that is NOT byte-identical to the
new one is kept under <state>/07_store.archive/<its receipt sha12>/ (to file it, move it back to 07_store/ and the live log aside); a
byte-identical one (the same execution pulled again) is replaced by the fresh pulls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FULL22_STAGER = REPO / "scripts" / "modal" / "stage_full22_bank_modal.py"
STORE_DIR = "07_store"
#: the pulls land here and are published as STORE_DIR only once the receipt is written (a refusal removes it; the published receipt stands)
STORE_BUILD_DIR = "07_store.building"
#: a superseded receipt (another execution's) is kept under <state>/07_store.archive/<its receipt sha12>/, never deleted
STORE_ARCHIVE_DIR = "07_store.archive"
RECEIPT_NAME = "receipt.json"
RUN_REPORT_NAME = "07_run_report.log"
RUN_REPORT_COPY = "run_report.log"
#: the launch's identity, written by `07 --phase run` beside the captured stdout; the receipt verifies it against its call and copies it
RUN_IDENTITY_NAME = "07_run_identity.json"
RUN_IDENTITY_SCHEMA = "stage07-run-identity/1"
#: /4: a DONE reuse names the credited attempts' call ids in `modal_call_ids` and `recs[<cell>].mode` is recorded — a
#: /3 reuse receipt could credit a same-tree pointer of another suite or mode and filed no call id; `07 --phase file` refuses it (re-pull).
#: /5: every control a recovered cell reuses is attributed through the launcher's completion records (`completions`,
#: `credited_completions`, `prior_completions`; `recs[<cell>].completion_record`) — a /4 reuse receipt over a resume-then-complete sequence
#: named the BigCodeBench call alone and omitted the BFCL-only call that produced BFCL; `07 --phase file` refuses it (re-pull).
#: /6: a recovered cell's credited pointers (`reused_attempts`, `bcb_attempts`) are joined to the completion records by
#: call id, and the same-tree pointers not credited are listed under `uncredited_attempts` with why — a /5 receipt credited a failed standup's
#: pointer (written before the drive, no completion record) as a producer in `modal_call_ids`; `07 --phase file` refuses it (re-pull)
#: /7 (the demo follows the paper — the FULL sets, no reduced subsets): the controls are `bigcodebench_full,bfcl_full`
#: (n 1,140 / 3,981) under the `protocol` sampling mode every REC must carry, and the receipt copies the launch's `slug` (the registry row: a
#: dose view's is `<key>-<view>`) and `sampling`; a /6 receipt is a 200-item execution's and `07 --phase file` refuses it (re-run on the full sets)
RECEIPT_SCHEMA = "stage07-receipt/7"
HEADLINE_CELL = "none"
#: the launcher's attempt-pointer layout (tinycap_served.WATCH_DIR / attempt_pointer_path / sandbox_record_path) and the pointer's fields (POINTER_FIELDS)
WATCH_DIR = "watch"
POINTER_FIELDS = ("run_id", "cell", "function_call_id", "name", "model", "mode", "out_dir", "controls", "tree_sha", "sandbox_record", "sandbox_app_name", "started_at")
#: the launcher's per-attempt completion record (tinycap_served.COMPLETION_FIELDS / completion_record_path), written at the end of every solo drive
COMPLETION_FIELDS = ("run_id", "cell", "function_call_id", "name", "model", "mode", "out_dir", "controls", "remaining", "completed", "status", "sandboxed",
                     "tree_sha", "finished_at")
REC_DONE = "DONE"
EXIT_REFUSED = 2


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_by_path("_stage07_receipt_full22_stager", FULL22_STAGER)     # VolumeCLI (get / ls_json / exists), Refusal, sha256_file, VOL_MOUNT
Refusal = M.Refusal


def _contracts():
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.capeval.tinycap import contracts as C  # noqa: WPS433 — REC_OK_STATUSES, the task / marker grammar, the volume roots
    return C


_C = _contracts()                                        # torch-free; ONE spelling of the controls, their item counts and the sampling mode
#: the paper's two FULL legs (contracts.CONTROL_BIGCODEBENCH_FULL / CONTROL_BFCL_FULL): what 07 launches and what every REC must have driven
HEADLINE_CONTROL = _C.CONTROL_BIGCODEBENCH_FULL
BFCL_CONTROL = _C.CONTROL_BFCL_FULL
DEFAULT_CONTROLS = f"{HEADLINE_CONTROL},{BFCL_CONTROL}"
#: the frozen FULL item count of the two legs (contracts.BCB_FULL_N 1,140 / BFCL_FULL_N 3,981 — the launcher's FULL_ITEM_COUNT_OF_TASK for
#: these tasks; `invalid_done_marker_reason` holds a done marker to it) — a marker with another count is INCOMPLETE however it reads
FULL_ITEM_COUNT = {_C.inspect_task(HEADLINE_CONTROL): int(_C.BCB_FULL_N), _C.inspect_task(BFCL_CONTROL): int(_C.BFCL_FULL_N)}
#: the paper's decoding (contracts.SAMPLING_PROTOCOL): each benchmark's reference sampling — BigCodeBench greedy, BFCL T 0.001 — the mode
#: every paper row ran under and 07 passes as `--sampling protocol`; the REC carries it as `sampling_mode` (the launcher's SAMPLING_MODE_KEY,
#: stamped on every REC — OK and DONE alike — from the mode its plan resolved; tests/test_stage07_analysis_filing.py pins the key to the
#: launcher's literal)
SAMPLING_MODE = _C.SAMPLING_PROTOCOL
REC_SAMPLING_MODE_KEY = "sampling_mode"


def sha256_file(p: Path) -> str:
    return M.sha256_file(p)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def volume_rel(container_path: str, *, mount: str = M.VOL_MOUNT) -> str:
    """The CHECKED conversion: a container path under the volume mount (``/vol/tinycap/gen/...``) as the volume-root
    path the ``modal volume`` CLI takes (``/tinycap/gen/...``); anything else refuses by name — the original is kept for every comparison."""
    p = str(container_path)
    if p.startswith(mount + "/"):
        return p[len(mount):]
    raise Refusal(f"{p!r} is not under the volume mount {mount!r} — the launcher's store paths are container paths under it; nothing else is pulled")


def attempt_pointer_path(out_dir: str, function_call_id: str) -> str:
    """``<out_dir>/watch/attempt_<fcid>.json`` — the launcher's own layout (tinycap_served.attempt_pointer_path), beside ``cells/``."""
    return f"{str(out_dir).rstrip('/')}/{WATCH_DIR}/attempt_{function_call_id}.json"


def sandbox_record_path(out_dir: str, function_call_id: str) -> str:
    """``<out_dir>/watch/sandboxes_<fcid>.jsonl`` — the launcher's per-attempt event record (tinycap_served.sandbox_record_path)."""
    return f"{str(out_dir).rstrip('/')}/{WATCH_DIR}/sandboxes_{function_call_id}.jsonl"


def completion_record_path(out_dir: str, function_call_id: str) -> str:
    """``<out_dir>/watch/completion_<fcid>.json`` — the launcher's per-attempt completion record (tinycap_served.completion_record_path)."""
    return f"{str(out_dir).rstrip('/')}/{WATCH_DIR}/completion_{function_call_id}.json"


class Volume:
    """The pulls, every path converted once and recorded as the container path the REC named."""

    def __init__(self, cli, store: Path, log):
        self.cli, self.store, self.log = cli, store, log
        self.pulled: Dict[str, Dict[str, Any]] = {}      # container path -> {volume_path, local, sha256, bytes}

    def get(self, container_path: str, local_rel: str) -> Path:
        vpath = volume_rel(container_path)
        into = self.store / local_rel
        into.parent.mkdir(parents=True, exist_ok=True)
        try:
            got = self.cli.get(vpath, into.parent)
        except Refusal as e:
            raise Refusal(f"cannot pull {container_path} (volume path {vpath}): {e}") from None
        if got.name != into.name:
            got = got.rename(into)
        self.pulled[container_path] = {"volume_path": vpath, "local": local_rel, "sha256": sha256_file(into), "bytes": into.stat().st_size}
        self.log(f"  pulled {container_path} -> {into} ({self.pulled[container_path]['sha256'][:12]})")
        return into

    def exists(self, container_path: str) -> bool:
        return self.cli.exists(volume_rel(container_path).lstrip("/"))

    def ls(self, container_dir: str) -> List[dict]:
        try:
            return self.cli.ls_json(volume_rel(container_dir).lstrip("/"))
        except Refusal as e:
            raise Refusal(f"cannot list {container_dir}: {e}") from None


def read_json(p: Path, what: str) -> dict:
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal(f"{what}: {p} is not readable JSON ({e})") from None
    if not isinstance(doc, dict):
        raise Refusal(f"{what}: {p} is not a JSON object")
    return doc


def parse_recs(report_text: str) -> Tuple[Dict[str, dict], List[str]]:
    """The launcher's RECs out of the captured stdout: every line that parses as a JSON object carrying `cell` and `status` (the launcher
    prints one per container, every key but `evidence`); the LAST REC per cell wins. Returns ({cell: rec}, the `container(s) OK` lines)."""
    recs: Dict[str, dict] = {}
    ok_lines: List[str] = []
    for ln in report_text.splitlines():
        s = ln.strip()
        if "container(s) OK" in s:
            ok_lines.append(s)
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            doc = json.loads(s)
        except ValueError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("cell"), str) and "status" in doc:
            recs[doc["cell"]] = doc
    return recs, ok_lines


def tasks_of(control: str, C) -> List[str]:
    """The thinking-OFF row tasks of one --controls token (contracts.TASKS_OF_CONTROL over its atoms, the `:open` variants excluded)."""
    try:
        atoms = C.expand_control(control)
    except C.ControlNameError as e:
        raise Refusal(f"--controls {control!r}: {e}") from None
    return [t for a in atoms for t in C.TASKS_OF_CONTROL[a] if not t.endswith(C.OPEN_VARIANT_SUFFIX)]


def check_rec(tok: str, rec: dict, *, sha: str, controls: Sequence[str], C) -> Dict[str, Any]:
    """The REC's own completion statement, held to the tree and the token; returns the facts the receipt records. Three accepted shapes
    (the launcher's branches): OK (first attempt, sandboxed), OK with `sandboxed: false` (a BFCL-only resumption), DONE (a reuse)."""
    status = rec.get("status")
    out = rec.get("out")
    if not isinstance(out, str) or not out.startswith(C.VOL_GEN_ROOT + "/"):
        raise Refusal(f"REC {tok}: `out` {out!r} is not a store root under the volume's generation root {C.VOL_GEN_ROOT}")
    if status == REC_DONE:
        resume, remaining = rec.get("resume"), rec.get("controls_remaining")
        if not isinstance(resume, dict) or remaining != []:
            raise Refusal(f"REC {tok}: status DONE but controls_remaining {remaining!r} / no resume block — not a relaunch over validly done stores")
        execution = "reuse"
        done_markers = None
    elif status in C.REC_OK_STATUSES and status != C.LADDER_REC_STATUS:
        resume, remaining = rec.get("resume_after"), rec.get("controls_still_remaining")
        if remaining != []:
            raise Refusal(f"REC {tok}: status {status} but controls_still_remaining {remaining!r}")
        if not isinstance(resume, dict):
            raise Refusal(f"REC {tok}: no resume_after block")
        done_markers = rec.get("done_markers")
        if not isinstance(done_markers, list):
            raise Refusal(f"REC {tok}: no done_markers list")
        execution = "new"
    elif status == C.LADDER_REC_STATUS:
        raise Refusal(f"REC {tok}: status {status} is a truncation-ladder rung's, not a BCB/BFCL run's")
    else:
        raise Refusal(f"REC {tok}: status {status!r} is not OK/DONE ({rec.get('error') or 'no error text'}) — the run did not complete this cell")
    if rec.get("sha") != sha[:7]:
        raise Refusal(f"REC {tok}: sha {rec.get('sha')!r} is not the tree's {sha[:7]}")
    if rec.get(REC_SAMPLING_MODE_KEY) != SAMPLING_MODE:
        raise Refusal(f"REC {tok}: {REC_SAMPLING_MODE_KEY} {rec.get(REC_SAMPLING_MODE_KEY)!r} is not the paper's {SAMPLING_MODE!r} decoding (07 launches "
                      f"`--sampling {SAMPLING_MODE}`; the launcher stamps every REC with its plan's mode, so `card` is the model card's sampling and a missing key an older "
                      "launcher's) — the run is not on the paper's protocol; "
                      "record no verdict")
    expected = resume.get("expected") if isinstance(resume, dict) else None
    if not isinstance(expected, dict) or expected.get("tree_sha") != sha:
        raise Refusal(f"REC {tok}: resume.expected.tree_sha {str((expected or {}).get('tree_sha'))[:12]!r} is not the tree {sha[:12]} — the stores it credits "
                      "were held to another tree; run the receipt from the commit the run ran at")
    if expected.get("cell") != tok:
        raise Refusal(f"REC {tok}: resume.expected.cell {expected.get('cell')!r} is not the token")
    done_in: Dict[str, str] = {}
    for c in controls:
        entry = (resume.get("controls") or {}).get(c)
        key = entry.get("done_in") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not key:
            raise Refusal(f"REC {tok}: resume.controls[{c!r}].done_in is {key!r} — no trusted store for the control")
        done_in[c] = key
    facts: Dict[str, Any] = {"status": status, "sha": rec.get("sha"), "model": rec.get("model"), "mode": rec.get("mode"), "out": out, "done_in": done_in,
                             "done_markers": done_markers, "execution": execution, "sandbox_owner": None, "sandboxed": None, "sandbox_pointer": None,
                             "sandbox_record": None, "controls_driven": rec.get("controls_driven"), "completion_record": None,
                             REC_SAMPLING_MODE_KEY: rec.get(REC_SAMPLING_MODE_KEY)}
    if execution == "reuse":
        return facts
    owner = rec.get("sandbox_owner")
    if not isinstance(owner, dict) or not all(isinstance(owner.get(k), str) and owner.get(k) for k in ("run_id", "cell", "function_call_id")):
        raise Refusal(f"REC {tok}: no sandbox_owner {{run_id, cell, function_call_id}} — an OK REC of this launcher carries its container's owner")
    if owner["cell"] != tok:
        raise Refusal(f"REC {tok}: sandbox_owner.cell {owner['cell']!r} is not the token")
    facts["sandbox_owner"] = dict(owner)
    facts["sandboxed"] = bool(rec.get("sandboxed"))
    if facts["sandboxed"]:
        for k in ("sandbox_pointer", "sandbox_record"):
            if not isinstance(rec.get(k), str) or not rec.get(k):
                raise Refusal(f"REC {tok}: sandboxed but no {k} — a sandboxed attempt publishes its pointer and record")
        want = attempt_pointer_path(out, owner["function_call_id"])
        if rec["sandbox_pointer"] != want:
            raise Refusal(f"REC {tok}: sandbox_pointer {rec['sandbox_pointer']!r} is not the launcher's layout {want!r} (<out>/watch/attempt_<fcid>.json)")
        facts["sandbox_pointer"], facts["sandbox_record"] = rec["sandbox_pointer"], rec["sandbox_record"]
    else:
        driven = rec.get("controls_driven")
        if not isinstance(driven, list):
            raise Refusal(f"REC {tok}: not sandboxed and no controls_driven list — cannot tell a BFCL-only resumption from a missing pointer")
        if HEADLINE_CONTROL in driven:
            raise Refusal(f"REC {tok}: drove {HEADLINE_CONTROL} but is not sandboxed and carries no sandbox_pointer — a BigCodeBench attempt publishes one")
        facts["execution"] = "resumed_bfcl_only"
    # round 4: every OK attempt of this launcher, sandboxed or not, publishes its completion record at the end of its drive — the per-control
    # attribution a later DONE relaunch's receipt credits; an OK REC without one is another launcher's (or a drive that never reached its end)
    want = completion_record_path(out, owner["function_call_id"])
    cr = rec.get("completion_record")
    if not isinstance(cr, str) or not cr:
        raise Refusal(f"REC {tok}: no completion_record — an OK REC of this launcher names its attempt's completion record ({want}); the per-control "
                      "attribution of a later relaunch cannot be made without it")
    if cr != want:
        raise Refusal(f"REC {tok}: completion_record {cr!r} is not the launcher's layout {want!r} (<out>/watch/completion_<fcid>.json)")
    facts["completion_record"] = cr
    return facts


def pointer_identity_problem(a: Mapping[str, Any], *, tok: str, out: str, sha: str, model: Optional[str], mode: Optional[str]) -> Optional[str]:
    """Why a pointer for this cell / out dir is NOT evidence of the execution a recovered run is credited with (None when it is): the tree
    the REC's resume block was held to, a named call id and run id, the pointer file named for its call id, the record in the launcher's
    layout, the REC's model, the REC's mode, and the full BigCodeBench among the controls it drove — for a BFCL-only resumption AND a DONE reuse
    alike: the launcher writes a pointer per attempt into the cell's watch/ for every suite and mode it drives, and a
    same-tree reduced-set (`bigcodebench_200`) or `:open` pointer cannot account for the full-set store a relaunch reuses."""
    fcid = a.get("function_call_id")
    if a.get("tree_sha") != sha:
        return f"tree_sha {str(a.get('tree_sha'))[:12]!r} is not the tree {sha[:12]}"
    if not isinstance(fcid, str) or not fcid:
        return "no function_call_id"
    if not isinstance(a.get("run_id"), str) or not a.get("run_id"):
        return "no run_id"
    if Path(str(a.get("path"))).name != f"attempt_{fcid}.json":
        return f"the pointer file {Path(str(a.get('path'))).name} is not named for its function_call_id {fcid!r}"
    if a.get("sandbox_record") != sandbox_record_path(out, fcid):
        return f"sandbox_record {a.get('sandbox_record')!r} is not the launcher's layout {sandbox_record_path(out, fcid)!r}"
    if model is not None and isinstance(a.get("model"), str) and a.get("model") != model:
        return f"model {a.get('model')!r} is not the REC's {model!r}"
    if mode is not None and a.get("mode") != mode:
        return f"mode {a.get('mode')!r} is not the REC's {mode!r}"
    if HEADLINE_CONTROL not in (a.get("controls") or []):
        return f"did not drive {HEADLINE_CONTROL} (controls {a.get('controls')!r})"
    return None


def completion_identity_problem(a: Mapping[str, Any], *, tok: str, out: str, sha: str, model: Optional[str], mode: Optional[str]) -> Optional[str]:
    """Why a completion record for this cell / out dir is NOT this run's evidence at all (None when it may be): the tree the REC's resume block
    was held to, a named call id and run id, the file named for its call id, a `completed` map, the REC's model and its mode. WHICH control it
    accounts for is the caller's question (`completed[<control>] == done_in[<control>]`, per control), never a property of the record alone."""
    fcid = a.get("function_call_id")
    if a.get("tree_sha") != sha:
        return f"tree_sha {str(a.get('tree_sha'))[:12]!r} is not the tree {sha[:12]}"
    if not isinstance(fcid, str) or not fcid:
        return "no function_call_id"
    if not isinstance(a.get("run_id"), str) or not a.get("run_id"):
        return "no run_id"
    if Path(str(a.get("path"))).name != f"completion_{fcid}.json":
        return f"the record file {Path(str(a.get('path'))).name} is not named for its function_call_id {fcid!r}"
    if not isinstance(a.get("completed"), dict):
        return f"no `completed` map ({a.get('completed')!r})"
    if model is not None and isinstance(a.get("model"), str) and a.get("model") != model:
        return f"model {a.get('model')!r} is not the REC's {model!r}"
    if mode is not None and a.get("mode") != mode:
        return f"mode {a.get('mode')!r} is not the REC's {mode!r}"
    return None


def _completion_entry(path: str, doc: Mapping[str, Any]) -> Dict[str, Any]:
    return {"path": path, **{k: doc.get(k) for k in COMPLETION_FIELDS if k != "finished_at"}}


def pull_cell(vol: Volume, tok: str, facts: dict, *, sha: str, controls: Sequence[str], C) -> Dict[str, Any]:
    """The volume receipt of one cell: markers + summaries per control, no partial beside a done, the merged controls_summary.json, the
    pointers and the completion records under <out>/watch/. Returns the per-cell record (paths are the container paths the REC named). A
    recovered run (a BFCL-only resumption, a DONE reuse) is credited, per control it did not drive itself, ONLY with completion records at the
    tree whose `completed[<control>]` is the store key the REC credits (round 4), and ONLY with pointers that pass :func:`pointer_identity_problem`
    at the tree AND whose call one of those records names (round 5: a failed standup's pointer passes identity and produced nothing); every
    sandboxed producer must have such a pointer. The others are history (`other_tree_attempts`, `uncredited_attempts`, each with why)."""
    out = facts["out"]
    rec: Dict[str, Any] = {"out": out, "stores": {}, "controls_summary": None, "pointer": None, "prior_attempts": [], "reused_attempts": [], "other_tree_attempts": [],
                           "uncredited_attempts": [], "completion": None, "prior_completions": [], "credited_completions": [], "n_items": {}}
    for c in controls:
        key = facts["done_in"][c]
        for task in tasks_of(c, C):
            marker = C.marker_filename(task)
            done_p = f"{out}/cells/{key}/{C.DONE_DIR}/{marker}"
            summ_p = f"{out}/cells/{key}/{C.SUMMARY_DIR}/{marker}"
            part_p = f"{out}/cells/{key}/{C.PARTIAL_DIR}/{marker}"
            if facts["done_markers"] is not None and marker not in facts["done_markers"]:
                raise Refusal(f"REC {tok}: done_markers {facts['done_markers']} name no {marker} ({task})")
            done = read_json(vol.get(done_p, f"{tok}/{key}/{C.DONE_DIR}/{marker}"), f"{tok} done marker")
            summ = read_json(vol.get(summ_p, f"{tok}/{key}/{C.SUMMARY_DIR}/{marker}"), f"{tok} summary")
            if vol.exists(part_p):
                raise Refusal(f"{tok}: a partial marker sits beside the done marker at {part_p} (a store mix-up) — not a complete store")
            if done.get("task") != task:
                raise Refusal(f"{tok}: {done_p} names task {done.get('task')!r}, not {task!r}")
            n = done.get("n_items")
            full = FULL_ITEM_COUNT.get(task)
            if full is not None and n != full:
                raise Refusal(f"{tok}: {done_p} n_items {n!r} != the frozen full item count {full} — an INCOMPLETE store (a limited run writes partial/, never done/)")
            if summ.get(C.METRIC_N) != n:
                raise Refusal(f"{tok}: {summ_p} n {summ.get(C.METRIC_N)!r} != the done marker's n_items {n}")
            rec["stores"].setdefault(key, {})[task] = {"done": done_p, "summary": summ_p, "n_items": n, "acc": summ.get("acc"), "n_scored": summ.get("n_scored")}
            rec["n_items"][task] = n
    cs_p = f"{out}/{tok}/{C.CONTROLS_SUMMARY_FILENAME}"
    cs = read_json(vol.get(cs_p, f"{tok}/{C.CONTROLS_SUMMARY_FILENAME}"), f"{tok} controls_summary")
    if not isinstance(cs.get("controls"), dict):
        raise Refusal(f"{tok}: {cs_p} carries no `controls` map")
    rec["controls_summary"] = cs_p
    # the attempt pointers under the store ROOT's watch/ (never under cells/<key>/): the REC's own, then every other attempt the root holds
    try:
        listing = vol.ls(f"{out}/{WATCH_DIR}")
    except Refusal:                                    # no watch/ under the root: no attempt pointer was ever published there — an empty listing, named below
        listing = []
    names = sorted(Path(str(e.get("filename"))).name for e in listing if e.get("type") == "file")
    attempts = [n for n in names if re.fullmatch(r"attempt_.+\.json", n)]
    own_fcid = (facts.get("sandbox_owner") or {}).get("function_call_id")
    if facts["sandboxed"]:
        p = facts["sandbox_pointer"]
        if Path(p).name not in attempts:
            raise Refusal(f"{tok}: the REC's sandbox_pointer {p} is not in {out}/{WATCH_DIR}/ (listing: {attempts}) — the pointer the watcher keys on is missing")
        doc = read_json(vol.get(p, f"{tok}/{WATCH_DIR}/{Path(p).name}"), f"{tok} attempt pointer")
        want = {"run_id": facts["sandbox_owner"]["run_id"], "cell": tok, "function_call_id": own_fcid, "out_dir": out,
                "tree_sha": None, "sandbox_record": facts["sandbox_record"]}
        for k, v in want.items():
            if k == "tree_sha":
                continue
            if doc.get(k) != v:
                raise Refusal(f"{tok}: pointer {p} field {k} = {doc.get(k)!r}, the REC says {v!r}")
        rec["pointer"] = {"path": p, "function_call_id": own_fcid, "run_id": doc.get("run_id"), "tree_sha": doc.get("tree_sha"), "controls": doc.get("controls")}
        rec["pointer_tree_sha"] = doc.get("tree_sha")
    others = [n for n in attempts if not (facts["sandboxed"] and n == Path(facts["sandbox_pointer"]).name)]
    for n in others:
        p = f"{out}/{WATCH_DIR}/{n}"
        doc = read_json(vol.get(p, f"{tok}/{WATCH_DIR}/{n}"), f"{tok} prior attempt pointer")
        entry = {"path": p, "function_call_id": doc.get("function_call_id"), "run_id": doc.get("run_id"), "cell": doc.get("cell"), "out_dir": doc.get("out_dir"),
                 "tree_sha": doc.get("tree_sha"), "controls": doc.get("controls"), "mode": doc.get("mode"), "sandbox_record": doc.get("sandbox_record"),
                 "model": doc.get("model")}
        rec["prior_attempts"].append(entry)
    qualifying: List[dict] = []                         # the identity-qualifying pointers of a recovered cell; credited only through the join below (round 5)
    pointer_rejected: List[dict] = []
    if facts["execution"] in ("resumed_bfcl_only", "reuse"):
        # the execution a recovered run is credited with: this cell's pointers under this out dir, held to the tree, the launcher's identity layout,
        # the REC's mode and the full BigCodeBench among the controls driven — for a DONE reuse as for a BFCL-only resumption (an unguarded
        # reuse path took any same-tree pointer of the cell, another suite's attempt included, and then named no call id at all). Identity
        # is checked FIRST (every identity refusal holds); which of these pointers IS the execution is settled by the completion records
        reuse = facts["execution"] == "reuse"
        mine = [a for a in rec["prior_attempts"] if a["cell"] == tok and a["out_dir"] == out]
        for a in mine:
            why = pointer_identity_problem(a, tok=tok, out=out, sha=sha, model=facts.get("model"), mode=facts.get("mode"))
            (qualifying if why is None else pointer_rejected).append(a if why is None else dict(a, problem=why))
        rec["other_tree_attempts"] = [a for a in pointer_rejected if a.get("tree_sha") != sha]
        if not qualifying:
            shape = "a DONE relaunch (reuse)" if reuse else "a BFCL-only resumption"
            what = (f"{shape}, but {out}/{WATCH_DIR}/ holds no earlier attempt pointer for this cell AT THE TREE {sha[:12]} that drove {HEADLINE_CONTROL} "
                    f"in mode {facts.get('mode')!r}")
            raise Refusal(f"{tok}: {what} (this cell's pointers, each with why it is not evidence: {[(a['path'], a['problem']) for a in pointer_rejected]}; other pointers: "
                          f"{[a['path'] for a in rec['prior_attempts'] if a not in mine]}) — the {'execution it reuses is' if reuse else 'BigCodeBench execution is'} "
                          "unaccounted for")
    # the COMPLETION records under the same watch/ (round 4): the REC's own first — pulled by exact path and held to the REC — then every other
    completions = [n for n in names if re.fullmatch(r"completion_.+\.json", n)]
    own_cr = facts.get("completion_record")
    if own_cr is not None:
        if Path(own_cr).name not in completions:
            raise Refusal(f"{tok}: the REC's completion_record {own_cr} is not in {out}/{WATCH_DIR}/ (completion records there: {completions}) — the attempt's "
                          "per-control completion statement is missing; a later relaunch could not credit this call")
        doc = read_json(vol.get(own_cr, f"{tok}/{WATCH_DIR}/{Path(own_cr).name}"), f"{tok} completion record")
        want = {"run_id": facts["sandbox_owner"]["run_id"], "cell": tok, "function_call_id": own_fcid, "out_dir": out, "tree_sha": sha, "mode": facts.get("mode"),
                "controls": facts.get("controls_driven"), "status": facts["status"], "sandboxed": facts["sandboxed"]}
        if isinstance(facts.get("model"), str):
            want["model"] = facts["model"]
        for k, v in want.items():
            if doc.get(k) != v:
                raise Refusal(f"{tok}: completion record {own_cr} field {k} = {doc.get(k)!r}, the REC says {v!r}")
        completed = doc.get("completed")
        if not isinstance(completed, dict):
            raise Refusal(f"{tok}: completion record {own_cr} carries no `completed` map ({completed!r})")
        driven = list(facts.get("controls_driven") or [])
        outside = sorted(set(completed) - set(driven))
        if outside:
            raise Refusal(f"{tok}: completion record {own_cr} claims {outside} completed, controls the attempt did not drive ({driven})")
        for c in controls:
            if c in driven and completed.get(c) != facts["done_in"][c]:
                raise Refusal(f"{tok}: completion record {own_cr} says {c} completed in {completed.get(c)!r}, the REC credits store {facts['done_in'][c]!r} — an OK "
                              "attempt finished every control it drove in the store the REC trusts")
        rec["completion"] = _completion_entry(own_cr, doc)
    for n in completions:
        if own_cr is not None and n == Path(own_cr).name:
            continue
        p = f"{out}/{WATCH_DIR}/{n}"
        rec["prior_completions"].append(_completion_entry(p, read_json(vol.get(p, f"{tok}/{WATCH_DIR}/{n}"), f"{tok} prior completion record")))
    if facts["execution"] in ("resumed_bfcl_only", "reuse"):
        # per control the cell did NOT drive itself, the attempt that completed it in the store the REC credits — a BFCL-only resumption's own
        # record accounts for its BFCL; a DONE relaunch drove nothing, so both controls are prior attempts' (a reuse after a
        # resume-then-complete sequence named the BigCodeBench call alone, the BFCL-only call that produced BFCL having left no pointer)
        reuse = facts["execution"] == "reuse"
        shape = "a DONE relaunch (reuse)" if reuse else "a BFCL-only resumption"
        own_driven = set(facts.get("controls_driven") or []) if not reuse else set()
        mine_c = [a for a in rec["prior_completions"] if a.get("cell") == tok and a.get("out_dir") == out]
        credited: Dict[str, Dict[str, Any]] = {}
        for c in controls:
            if c in own_driven:
                continue
            key = facts["done_in"][c]
            rejected_c: List[Tuple[str, str]] = []
            found = False
            for a in mine_c:
                why = completion_identity_problem(a, tok=tok, out=out, sha=sha, model=facts.get("model"), mode=facts.get("mode"))
                if why is None and (a.get("completed") or {}).get(c) != key:
                    why = f"completed {a.get('completed')!r} does not name {c} in store {key!r}"
                if why is not None:
                    rejected_c.append((a["path"], why))
                    continue
                found = True
                credited.setdefault(a["path"], dict(a, credited_for=[]))["credited_for"].append(c)
            if not found:
                raise Refusal(f"{tok}: {shape}, but {out}/{WATCH_DIR}/ holds no completion record for this cell AT THE TREE {sha[:12]} in mode {facts.get('mode')!r} "
                              f"whose attempt completed {c} in the store the REC credits ({key}) (this cell's completion records, each with why it is not evidence: "
                              f"{rejected_c}; other completion records: {[a['path'] for a in rec['prior_completions'] if a not in mine_c]}) — the {c} execution "
                              f"{'it reuses' if reuse else 'it credits'} is unaccounted for")
        rec["credited_completions"] = [credited[p] for p in sorted(credited)]
        # THE JOIN: a pointer is written BEFORE the drive, so an attempt whose standup failed leaves a pointer that passes every
        # identity check above and NO completion record — it produced nothing. The pointers credited as the execution are the qualifying ones whose
        # call a credited completion record names (joined by function_call_id); and every credited producer that was sandboxed (a BigCodeBench
        # producer always is) must have such a pointer, the launcher having published one before its drive. The rest is history, each with why.
        producers = {a["function_call_id"]: a for a in rec["credited_completions"]}
        evidence: List[dict] = []
        for a in qualifying:
            producer = producers.get(a["function_call_id"])
            if producer is None:
                pointer_rejected.append(dict(a, problem=(f"no completion record for this cell at the tree credits call {a['function_call_id']!r} with a control the REC "
                                                        f"{'reuses' if reuse else 'credits'} — the pointer is written before the drive; this attempt completed nothing "
                                                        "in the stores the REC trusts (a standup that failed, or a drive that never reached its end)")))
            else:
                evidence.append(dict(a, credited_for=list(producer["credited_for"])))
        have = {a["function_call_id"] for a in evidence}
        for p in sorted(credited):
            a = credited[p]
            if (bool(a.get("sandboxed")) or HEADLINE_CONTROL in a["credited_for"]) and a["function_call_id"] not in have:
                raise Refusal(f"{tok}: {shape}, but the attempt {a['function_call_id']!r} whose completion record {p} credits {a['credited_for']} in the stores the REC "
                              f"trusts has no attempt pointer at {attempt_pointer_path(out, a['function_call_id'])} that is this run's evidence — a sandboxed attempt "
                              f"publishes one before its drive (this cell's pointers at the tree, each with why it is not evidence: "
                              f"{[(x['path'], x['problem']) for x in pointer_rejected if x.get('tree_sha') == sha]}; other-tree pointers: "
                              f"{[x['path'] for x in rec['other_tree_attempts']]}) — the {'execution it reuses is' if reuse else 'BigCodeBench execution is'} "
                              "unaccounted for")
        rec["reused_attempts" if reuse else "bcb_attempts"] = evidence
        rec["uncredited_attempts"] = [a for a in pointer_rejected if a.get("tree_sha") == sha]
    return rec


def headline_of(tok: str, cell_rec: dict, facts: dict, store: Path, *, C) -> Tuple[dict, dict]:
    """(the headline block, the bfcl block) from the pulled controls_summary.json, cross-checked three ways against the pulled summary and
    the store key the REC trusted."""
    cs = read_json(store / tok / C.CONTROLS_SUMMARY_FILENAME, f"{tok} controls_summary")
    task = tasks_of(HEADLINE_CONTROL, C)[0]
    entry = cs["controls"].get(task)
    if not isinstance(entry, dict):
        raise Refusal(f"{tok}: controls_summary.json has no {task!r} entry (controls: {sorted(cs['controls'])}) — no merged BigCodeBench result")
    if entry.get("task") != task:
        raise Refusal(f"{tok}: the merged {task} entry names task {entry.get('task')!r}")
    n = entry.get(C.METRIC_N)
    if n != FULL_ITEM_COUNT[task]:
        raise Refusal(f"{tok}: the merged {task} entry has n {n!r}, not {FULL_ITEM_COUNT[task]}")
    key = facts["done_in"][HEADLINE_CONTROL]
    pulled = cell_rec["stores"][key][task]
    if entry.get("acc") != pulled["acc"] or n != pulled["n_items"]:
        raise Refusal(f"{tok}: the merged {task} entry (acc {entry.get('acc')!r}, n {n!r}) disagrees with the pulled summary/{C.marker_filename(task)} "
                      f"(acc {pulled['acc']!r}, n {pulled['n_items']!r})")
    src = (cs.get(C.SUMMARY_KEY_SOURCES) or {}).get(task)
    if not isinstance(src, dict) or src.get("cell_key") != key:
        raise Refusal(f"{tok}: controls_summary.json sources[{task!r}].cell_key is {(src or {}).get('cell_key')!r}, not the REC's trusted store {key!r}")
    acc = entry.get("acc")
    if not isinstance(acc, (int, float)) or isinstance(acc, bool):
        raise Refusal(f"{tok}: the merged {task} acc {acc!r} is not a number")
    headline = {"cell": tok, "task": task, "metric": "acc", "value": float(acc), "n": n, "n_scored": entry.get("n_scored"), "store_key": key}
    btask = tasks_of(BFCL_CONTROL, C)[0]
    bentry = cs["controls"].get(btask)
    if not isinstance(bentry, dict) or bentry.get("task") != btask:
        raise Refusal(f"{tok}: controls_summary.json has no merged {btask!r} entry")
    bfcl = {"cell": tok, "task": btask, "acc": bentry.get("acc"), "n": bentry.get(C.METRIC_N), "n_scored": bentry.get("n_scored"), "store_key": facts["done_in"][BFCL_CONTROL]}
    return headline, bfcl


def _fmt(v: Any) -> str:
    return ",".join(v) if isinstance(v, list) else ("none" if v is None else str(v))


def read_run_identity(path: Path, *, sha: str, view: str, cells: Sequence[str], view_cells: Sequence[str], app: Optional[str], volume: str,
                      environment: Optional[str], override_id: Optional[str], C, controls: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], str]:
    """The LAUNCH's identity (written by `07 --phase run` beside the captured stdout), read once as a snapshot, VERIFIED against this call and
    returned with the snapshot's sha256. Every field this call also resolves must agree (the receipt describes the launch, never this call);
    `--app` and `--override-id` are cross-checks when passed, `--controls` too (the launch's `controls` string); a launch under a sampling mode
    other than the paper's `protocol` is refused by name; the objects map must cover exactly the fitted cells of the run."""
    if not path.is_file():
        raise Refusal(f"no run identity at {path} — `07 --phase run` writes the launch's identity (its tree, cells, objects, run-overrides document and "
                      "expect-hashes) beside its stdout before the launcher runs; a log with no identity beside it cannot be receipted (run --phase run)")
    snapshot = path.read_bytes()
    try:
        doc = json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise Refusal(f"{path} is not readable JSON ({e}) — not the launch's identity") from None
    if not isinstance(doc, dict) or doc.get("schema") != RUN_IDENTITY_SCHEMA:
        raise Refusal(f"{path} is not a {RUN_IDENTITY_SCHEMA!r} document (schema {doc.get('schema') if isinstance(doc, dict) else None!r}) — run --phase run again")
    checks = (("tree_sha", sha, "--sha", "the run was launched at tree {want}; run the receipt from that commit (every REC is held to it)"),
              ("view", view, "--view", "the run was launched for view {want}"),
              ("launcher_cells", list(cells), "--cells", "the run was launched with cells {want} (the launcher's tokens, in the launch's order)"),
              ("cells", list(view_cells), "--view-cells", "the run was launched with cells {want} (the table's names, in the launch's order)"),
              ("volume", volume, "--volume", "the run wrote volume {want}; a pull from another volume is another execution's evidence"),
              ("modal_environment", environment or None, "--environment", "the run ran under Modal environment {want}; a volume name is per environment"),
              *([("controls", ",".join(controls), "--controls", "the run was launched with controls {want} (the launcher's tokens; the paper's full sets)")]
                if controls is not None else []))
    for field, got, flag, why in checks:
        want = doc.get(field)
        if want != got:
            raise Refusal(f"{flag} {_fmt(got)} is not the run's: {why.format(want=_fmt(want))} ({path}) — pass {flag} {_fmt(want)}; the receipt records the "
                          "launch's identity, never this call's")
    if app is not None and doc.get("app") != app:
        raise Refusal(f"--app {app} is not the run's: the run was launched as Modal app {doc.get('app')!r} ({path})")
    if doc.get("sampling") != SAMPLING_MODE:
        raise Refusal(f"{path}: the run was launched under sampling {doc.get('sampling')!r}, not the paper's {SAMPLING_MODE!r} decoding (07 launches "
                      f"`--sampling {SAMPLING_MODE}`; an identity without the key is a run of an earlier 07) — the run is not on the paper's protocol")
    if not isinstance(doc.get("slug"), str) or not doc["slug"]:
        raise Refusal(f"{path} names no launcher slug (`slug`: the registry row the run was served from)")
    if not isinstance(doc.get("app"), str) or not doc["app"]:
        raise Refusal(f"{path} names no Modal app")
    ro = doc.get("run_overrides")
    if not isinstance(ro, dict) or not isinstance(ro.get("id"), str) or not ro["id"]:
        raise Refusal(f"{path} names no run-overrides document (run_overrides.id) — the launch records OVERRIDE_ID, its sha256 and its path")
    if override_id is not None and override_id != ro["id"]:
        raise Refusal(f"--override-id {override_id} is not the run's: the run was launched under run-overrides {ro['id']} ({ro.get('path')}; {path}) — run the "
                      f"receipt and the filing under that document (--overrides {ro.get('path')}); the receipt records the launch's, never this call's")
    objects = doc.get("objects")
    fitted = [t for t in cells if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    if objects is not None:
        if not isinstance(objects, dict) or not all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in objects.items()):
            raise Refusal(f"{path}: `objects` is not a map of launcher token -> 64-hex object id")
        missing = [t for t in fitted if t not in objects]
        extra = [t for t in objects if t not in fitted]
        if missing or extra:
            raise Refusal(f"{path}: `objects` does not cover exactly the run's fitted cells (fitted {fitted}; no object for {missing}; objects for cells the run "
                          f"did not launch {extra}) — the launch's identity is not this run's")
    eh = doc.get("expect_hashes")
    if eh is not None and not (isinstance(eh, dict) and isinstance(eh.get("hashes"), dict) and isinstance(eh.get("sha256"), str)):
        raise Refusal(f"{path}: `expect_hashes` is neither null nor {{path, sha256, hashes}}")
    return doc, sha256_bytes(snapshot)


def publish_store(state: Path, build: Path, receipt_sha: str, log) -> Path:
    """The finished build becomes <state>/07_store/. A previous published receipt is never lost: byte-identical to the new one (the same
    execution pulled again; it pins every file's sha) -> replaced by the fresh pulls; another receipt -> kept under 07_store.archive/<sha12>/;
    a directory with no receipt.json (an earlier version's in-place build that never finished) -> kept under 07_store.archive/incomplete/."""
    store = state / STORE_DIR
    archive = state / STORE_ARCHIVE_DIR
    if store.exists():
        prev = store / RECEIPT_NAME
        prev_sha = sha256_file(prev) if prev.is_file() else None
        if prev_sha == receipt_sha:
            shutil.rmtree(store)
            log(f"  the previous {store} held this same receipt ({receipt_sha[:12]}): replaced by the fresh pulls")
        else:
            tag = prev_sha[:12] if prev_sha else "incomplete"
            dest, k = archive / tag, 1
            while dest.exists():
                dest, k = archive / f"{tag}.{k}", k + 1
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(store), str(dest))
            log(f"  the previous receipt ({'receipt ' + prev_sha[:12] if prev_sha else 'no receipt.json: an unfinished build'}) is kept at {dest} — to file it, "
                f"move it back to {store} (and the live {RUN_REPORT_NAME} aside)")
    os.rename(build, store)
    return store


def verdict_command(state: Path, view: str, receipt: dict, receipt_sha: str, *, python: str = sys.executable) -> str:
    keys = ",".join(receipt["store_cell_keys"])
    note = f"headline: {receipt['headline']['cell']} {receipt['headline']['task']} {receipt['headline']['metric']}; receipt {receipt_sha[:12]}"
    return (f"{python} {REPO / 'reproduce' / 'lib' / 'stage.py'} verdict --state-root {state} --family bcb_bfcl/{view} "
            f"--headline {receipt['headline']['value']!r} --range <lo,hi> --run-ids {keys} --note \"{note}\"")


def run(*, state: Path, view: str, sha: str, cells: Sequence[str], view_cells: Sequence[str], controls: Sequence[str], run_report: Path, volume: str,
        modal_bin: Path, dry_run: bool, override_id: Optional[str] = None, app: Optional[str] = None, run_identity: Optional[Path] = None,
        environment: Optional[str] = None, runner=M.default_runner, log=print) -> Dict[str, Any]:
    C = _contracts()
    environment = environment or None
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Refusal(f"--sha {sha!r} is not a 40-hex tree sha")
    tokens = [c for c in cells if c]
    vcells = [c for c in view_cells if c]
    if HEADLINE_CELL not in tokens:
        raise Refusal(f"--cells {','.join(tokens)} has no {HEADLINE_CELL!r} cell — the headline is the none cell's BigCodeBench acc; this run cannot be ruled on")
    if len(vcells) != len(tokens):
        raise Refusal(f"--view-cells {','.join(vcells)} ({len(vcells)}) and --cells {','.join(tokens)} ({len(tokens)}) are not the same cells by two names — "
                      "07 hands the receipt the view's cells by the table's names and the launcher's tokens, in one order")
    # (0) the launch's identity: verified against this call, copied into the receipt (a receipt taken under another
    # overrides document, another HEAD or other cells must not relabel the execution; the objects are the launch tree's, never this checkout's)
    identity_path = run_identity or state / RUN_IDENTITY_NAME
    identity, identity_sha = read_run_identity(identity_path, sha=sha, view=view, cells=tokens, view_cells=vcells, app=app, volume=volume, environment=environment,
                                               override_id=override_id, C=C, controls=controls)
    if not run_report.is_file():
        raise Refusal(f"no run report at {run_report} — run `07 --phase run` first (the launcher's stdout is captured there); nothing to receipt")
    # the run report is read ONCE, as one snapshot of bytes: parsed, copied and hashed as the same bytes (a `--phase run`
    # that overwrites the live log during the pulls must never leave a receipt whose RECs are one execution's and whose copy is another's)
    snapshot = run_report.read_bytes()
    snapshot_sha = sha256_bytes(snapshot)
    try:
        text = snapshot.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Refusal(f"{run_report} is not UTF-8 text ({e}) — not the launcher's captured stdout") from None
    recs, ok_lines = parse_recs(text)
    if not recs:
        raise Refusal(f"{run_report} holds no REC line (the launcher prints one JSON object per container after `[tinycap] N/N container(s) OK`)")
    facts: Dict[str, dict] = {}
    for tok in tokens:
        if tok not in recs:
            raise Refusal(f"no REC for cell {tok!r} in {run_report} (RECs: {sorted(recs)})")
        facts[tok] = check_rec(tok, recs[tok], sha=sha, controls=controls, C=C)
    outside = sorted(set(recs) - set(tokens))
    if outside:
        raise Refusal(f"{run_report} carries RECs for cells outside --cells {','.join(tokens)}: {outside} — the receipt is the WHOLE execution's evidence, never a "
                      f"subset of it; pass the run's cells ({','.join(sorted(recs))})")
    store = state / STORE_DIR
    if dry_run:
        log(f"DRY-RUN: {len(tokens)} REC(s) OK in {run_report} ({', '.join(f'{t}:{facts[t]['status']}' for t in tokens)}); the launch's identity {identity_path} agrees "
            f"(overrides {identity['run_overrides']['id']}, {len(identity.get('objects') or {})} object(s)); would pull per cell "
            f"<out>/cells/<done_in>/{{done,summary}}/<marker> for {list(controls)}, <out>/<cell>/{C.CONTROLS_SUMMARY_FILENAME}, <out>/{WATCH_DIR}/attempt_*.json and "
            f"<out>/{WATCH_DIR}/completion_*.json "
            f"from volume {volume} (Modal environment {environment or 'the CLI default'}) into {state / STORE_BUILD_DIR}, copy the run report snapshot beside them, "
            f"write {RECEIPT_NAME} and publish the build as {store}; nothing pulled")
        for tok in tokens:
            log(f"DRY-RUN:   {tok}: out {facts[tok]['out']} done_in {facts[tok]['done_in']} execution {facts[tok]['execution']}")
        return {"dry": True, "recs": facts}
    # the pulls land in a BUILD directory; the published receipt (and its copy of an earlier run log) stands until the build is complete
    build = state / STORE_BUILD_DIR
    if build.exists():
        shutil.rmtree(build)                          # an earlier build that never finished — never a receipt
    build.mkdir(parents=True)
    try:
        receipt, rsha, n_pulled = _build(build, identity=identity, identity_path=identity_path, identity_sha=identity_sha, tokens=tokens, vcells=vcells, facts=facts,
                                         ok_lines=ok_lines, snapshot=snapshot, snapshot_sha=snapshot_sha, run_report=run_report, view=view, sha=sha, controls=controls,
                                         volume=volume, environment=environment, modal_bin=modal_bin, runner=runner, C=C, log=log)
    except Refusal:
        shutil.rmtree(build, ignore_errors=True)      # the published receipt, if any, is untouched
        raise
    store = publish_store(state, build, rsha, log)
    rpath = store / RECEIPT_NAME
    headline, bfcl = receipt["headline"], receipt["bfcl"]
    log(f"receipt: {rpath} (sha256 {rsha[:12]}; {n_pulled} file(s) pulled; call ids {receipt['modal_call_ids']}; execution {receipt['execution']})")
    log(f"headline: {headline['cell']} {headline['task']} {headline['metric']} = {headline['value']!r} (n {headline['n']}, n_scored {headline['n_scored']}, store {headline['store_key']}); "
        f"bfcl acc = {bfcl['acc']!r} (n {bfcl['n']})")
    log("verdict (fill --range, the one blank): " + verdict_command(state, view, receipt, rsha))
    return {"dry": False, "receipt": str(rpath), "receipt_sha256": rsha, "headline": headline, "verdict_command": verdict_command(state, view, receipt, rsha)}


def _build(build: Path, *, identity: dict, identity_path: Path, identity_sha: str, tokens: List[str], vcells: List[str], facts: Dict[str, dict], ok_lines: List[str],
           snapshot: bytes, snapshot_sha: str, run_report: Path, view: str, sha: str, controls: Sequence[str], volume: str, environment: Optional[str], modal_bin: Path,
           runner, C, log) -> Tuple[dict, str, int]:
    """Every pull, every check and the receipt's write, into the build directory; (the receipt, its sha256, the number of files pulled)."""
    vol = Volume(M.VolumeCLI(modal_bin, volume, runner=runner, log=log, environment=environment), build, log)
    cell_recs: Dict[str, dict] = {}
    for tok in tokens:
        log(f"== {tok} ({facts[tok]['execution']}) <- {facts[tok]['out']}")
        cell_recs[tok] = pull_cell(vol, tok, facts[tok], sha=sha, controls=controls, C=C)
    for tok in tokens:                                # the pointers' tree sha, once every pull is in
        p = cell_recs[tok].get("pointer")
        if p is not None and p.get("tree_sha") != sha:
            raise Refusal(f"{tok}: pointer {p['path']} tree_sha {str(p.get('tree_sha'))[:12]!r} is not the tree {sha[:12]}")
    headline, bfcl = headline_of(HEADLINE_CELL, cell_recs[HEADLINE_CELL], facts[HEADLINE_CELL], build, C=C)
    call_ids: List[str] = []
    prior: List[dict] = []
    reused: List[dict] = []
    other_tree: List[dict] = []
    uncredited: List[dict] = []
    credited_completions: List[dict] = []
    prior_completions: List[dict] = []
    for tok in tokens:
        f = facts[tok]; r = cell_recs[tok]
        if f["sandbox_owner"]:
            call_ids.append(f["sandbox_owner"]["function_call_id"])
        # a recovered cell's credited pointers (round 3) are, since round 5, joined by call id to its credited completion records — so the two
        # lists below name the same calls, and a pointer no record credits (a failed standup's) is `uncredited_attempts`, never a call id
        if f["execution"] == "resumed_bfcl_only":
            call_ids.extend(a["function_call_id"] for a in r["bcb_attempts"] if a.get("function_call_id"))
        if f["execution"] == "reuse":                 # the validated producing attempts ARE the execution a reuse names (round 3); `execution` stays reuse
            call_ids.extend(a["function_call_id"] for a in r["reused_attempts"] if a.get("function_call_id"))
        # round 4: the attempt that completed each control the cell did not drive itself — the BFCL-only call a reuse used to omit
        call_ids.extend(a["function_call_id"] for a in r["credited_completions"] if a.get("function_call_id"))
        prior.extend(dict(a, cell_of_receipt=tok) for a in r["prior_attempts"])
        reused.extend(dict(a, cell_of_receipt=tok) for a in r["reused_attempts"])
        other_tree.extend(dict(a, cell_of_receipt=tok) for a in r["other_tree_attempts"])
        uncredited.extend(dict(a, cell_of_receipt=tok) for a in r["uncredited_attempts"])
        credited_completions.extend(dict(a, cell_of_receipt=tok) for a in r["credited_completions"])
        prior_completions.extend(dict(a, cell_of_receipt=tok) for a in r["prior_completions"])
    call_ids = sorted(dict.fromkeys(c for c in call_ids if c))
    if not call_ids:
        raise Refusal("no Modal call id: an OK cell carries its container's owner and a recovered cell the pointer(s) and completion record(s) of the attempts it "
                      "credits — the receipt cannot name the execution")
    n_items: Dict[str, int] = {}
    for tok in tokens:
        for task, n in cell_recs[tok]["n_items"].items():
            if n_items.setdefault(task, n) != n:
                raise Refusal(f"{task}: n_items differs across cells ({n_items[task]} vs {n} on {tok})")
    # the live log and the live identity must still be the snapshots the RECs and the identity were read from: another `--phase run` meanwhile
    # may have changed which execution the pulled stores and pointers belong to — refuse, no receipt; the operator runs the receipt again
    live_sha = sha256_bytes(run_report.read_bytes()) if run_report.is_file() else None
    if live_sha != snapshot_sha:
        raise Refusal(f"{run_report} changed while the receipt was being built (read at sha {snapshot_sha[:12]}, now {str(live_sha)[:12]}): a `--phase run` wrote it "
                      "meanwhile, so the pulls above may be another execution's — run --phase receipt again over the log as it now stands")
    live_id = sha256_bytes(identity_path.read_bytes()) if identity_path.is_file() else None
    if live_id != identity_sha:
        raise Refusal(f"{identity_path} changed while the receipt was being built (read at sha {identity_sha[:12]}, now {str(live_id)[:12]}): a `--phase run` wrote it "
                      "meanwhile — run --phase receipt again")
    (build / RUN_REPORT_COPY).write_bytes(snapshot)
    ro = identity["run_overrides"]
    eh = identity.get("expect_hashes")
    receipt = {
        "schema": RECEIPT_SCHEMA, "view": view, "tree_sha": sha, "app": identity["app"], "slug": identity["slug"], "sampling": identity["sampling"],
        "volume": volume, "modal_environment": environment,
        "cells": list(tokens), "view_cells": list(vcells), "objects": identity.get("objects"),
        "run_overrides_id": ro["id"], "run_overrides_sha256": ro.get("sha256"), "run_overrides_path": ro.get("path"),
        "run_identity_sha256": identity_sha, "controls": list(controls),
        "tasks": {c: tasks_of(c, C) for c in controls},
        "recs": {t: {k: facts[t][k] for k in ("status", "sha", "model", "mode", "out", "done_in", "done_markers", "execution", "sandbox_owner", "sandboxed",
                                              "sandbox_pointer", "sandbox_record", "controls_driven", "completion_record", REC_SAMPLING_MODE_KEY)} for t in tokens},
        "stores": {t: cell_recs[t]["stores"] for t in tokens},
        "controls_summary": {t: cell_recs[t]["controls_summary"] for t in tokens},
        "pointers": {t: cell_recs[t]["pointer"] for t in tokens},
        "bcb_attempts": {t: cell_recs[t].get("bcb_attempts") for t in tokens if cell_recs[t].get("bcb_attempts")},
        "completions": {t: cell_recs[t]["completion"] for t in tokens},
        "pulled": {p: {"volume_path": v["volume_path"], "local": v["local"], "sha256": v["sha256"], "bytes": v["bytes"]} for p, v in sorted(vol.pulled.items())},
        "store_cell_keys": sorted({k for t in tokens for k in facts[t]["done_in"].values()}),
        "store_out_dirs": sorted({facts[t]["out"] for t in tokens}),
        "modal_call_ids": call_ids, "prior_attempts": prior, "reused_attempts": reused, "other_tree_attempts": other_tree, "uncredited_attempts": uncredited,
        "credited_completions": credited_completions, "prior_completions": prior_completions,
        "execution": "reuse" if all(facts[t]["execution"] == "reuse" for t in tokens) else "new",
        "n_items_per_task": n_items, "headline": headline, "bfcl": bfcl,
        "run_report": {"copy": RUN_REPORT_COPY, "container_ok_lines": ok_lines},
        "run_report_sha256": snapshot_sha,
        "expect_hashes": (eh or {}).get("hashes") if eh else None, "expect_hashes_sha256": (eh or {}).get("sha256") if eh else None,
        "expect_hashes_path": (eh or {}).get("path") if eh else None,
    }
    rpath = build / RECEIPT_NAME
    rpath.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return receipt, sha256_file(rpath), len(vol.pulled)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-root", type=Path, required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--sha", required=True, help="the FULL tree sha the run was launched at (every REC and the launch's identity are held to it)")
    ap.add_argument("--cells", required=True, help="the run's launcher tokens, comma list (07's LAUNCHER_CELLS); `none` among them; must be the launch's")
    ap.add_argument("--view-cells", required=True, help="the same cells by the models table's names, comma list, in the same order (07's CELLS); must be the launch's")
    ap.add_argument("--run-identity", type=Path, default=None, help=f"the launch's identity `07 --phase run` wrote (default <state-root>/{RUN_IDENTITY_NAME}); verified "
                                                                  "against this call and copied into the receipt")
    ap.add_argument("--override-id", default=None, help="a cross-check: OVERRIDE_ID of this call's run-overrides document; must be the one the run was launched under "
                                                       "(the receipt records the launch's)")
    ap.add_argument("--controls", default=DEFAULT_CONTROLS, help=f"the run's control tokens (default {DEFAULT_CONTROLS}: the paper's full sets); must be the launch's")
    ap.add_argument("--app", default=None, help="a cross-check: the Modal app name; must be the launch's (the receipt records the launch's)")
    ap.add_argument("--run-report", type=Path, default=None, help=f"default <state-root>/{RUN_REPORT_NAME}")
    ap.add_argument("--volume", default=None, help="default $SISTEER_MODAL_VOLUME, else contracts.MODAL_VOLUME; must be the volume the run wrote")
    ap.add_argument("--environment", default=None, help="the Modal environment every volume command is scoped to (`--env`; default $SISTEER_MODAL_ENVIRONMENT, "
                                                       "else the CLI's own resolution) — must be the one the launcher ran under")
    ap.add_argument("--modal", default=str(M.DEFAULT_MODAL), help="the Modal CLI (`<python> -m modal` accepted)")
    ap.add_argument("--dry-run", action="store_true", help="read the run report and the identity and say what would be pulled; pull nothing, write nothing")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    log = lambda s: print(f"[stage07-receipt] {s}", flush=True)  # noqa: E731
    try:
        C = _contracts()
        volume = a.volume or os.environ.get("SISTEER_MODAL_VOLUME") or C.MODAL_VOLUME
        environment = a.environment or os.environ.get("SISTEER_MODAL_ENVIRONMENT") or None
        parts = str(a.modal).split()
        runner = M.default_runner
        if len(parts) > 1:
            prefix = parts
            runner = lambda argv, **kw: M.default_runner([*prefix, *list(argv)[1:]], **kw)  # noqa: E731
        state = Path(a.state_root)
        run(state=state, view=a.view, sha=a.sha, cells=[c for c in a.cells.split(",") if c], view_cells=[c for c in a.view_cells.split(",") if c],
            override_id=a.override_id, controls=[c for c in a.controls.split(",") if c], app=a.app, run_report=a.run_report or state / RUN_REPORT_NAME,
            run_identity=a.run_identity, volume=volume, modal_bin=Path(parts[0]), dry_run=a.dry_run, environment=environment, runner=runner, log=log)
        return 0
    except Refusal as e:
        log(f"REFUSED: {e}")
        log("no receipt written; no verdict command")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
