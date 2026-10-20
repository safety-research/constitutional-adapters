#!/usr/bin/env python3
"""reproduce/tools/stage07_file.py — stage 07's FILING: the ruled BCB/BFCL verdict into the registry as an ANALYSIS (demo-import M8;
every run we execute reaches the registry; 07 registers no run rows (C10), so the gate verdict
is filed with `registry analysis file` — substrate modal, the hashed local report, the exercised objects as inputs, no cluster job ids, the
Modal app / call ids in `extra`).

    python reproduce/tools/stage07_file.py --state-root S --view V --model K --registry-root R [--cells <table names>] [--launcher-cells <tokens>]
                                           [--override-id ID] [--table T] [--repo R] [--dry-run]

Inputs, each REFUSED by name when absent: <state>/gate/bcb_bfcl/<view>.json (stage.py verdict, ruled from the command `07 --phase receipt`
printed) and <state>/07_store/receipt.json (run --phase receipt first; a receipt of an older schema — one that does not carry the run's
identity below — is refused the same way; a receipt describing ANOTHER view — `receipt.view` not --view — too, by name). <state> is THE
VIEW'S state directory as 07 passes it (the model's state dir for the baseline wave, <model state>/07_views/<view>/ for a dose view): the
receipt, the verdict, the mirror this filing points at, the spec it writes and the pointer records below are all that view's, so two views
of one model never share a receipt, a mirror or a pointer directory. Before anything is composed: every pulled file under <state>/07_store/ — the run report's copy
included — is re-hashed against the receipt (a local integrity check, no volume access), the LIVE <state>/07_run_report.log, when present,
must still be the receipt's (another sha: a later `--phase run` wrote it — refused by name, since the likeliest reading is an operator filing
the old execution as the new one), and the verdict is VERIFIED AGAINST THE RECEIPT: `headline == receipt.headline.value` exactly,
`sorted(run_ids) == the store keys`, `family == bcb_bfcl/<view>`, and `note` carrying `receipt <sha12>` of THIS receipt (a re-run that
resumed the same stores leaves the headline and keys unchanged; the note tells its verdict apart).

THE EXECUTION'S IDENTITY IS THE RECEIPT'S, never this call's (code review rounds 1-2): the cells the run exercised (`receipt.cells`, the
launcher tokens; `receipt.view_cells`, the same cells by the table's names — held equal through the table's own mapping, lib/models.py
launcher_cells), THE OBJECTS IT EXERCISED (`receipt.objects`, token -> committed object id, resolved AT LAUNCH from the run's own tree by `07
--phase run` and copied by the receipt — never re-planned from this checkout, whose objects/DEMO.json or models table may have moved since;
when this checkout IS at the receipt's tree, the live plan is cross-checked against the map and a disagreement refuses, and at any other
HEAD the map is filed as recorded, with a line saying so), the run-overrides document (`receipt.run_overrides_id`, its sha256), the Modal app
(`receipt.app`) and the tree (`receipt.tree_sha`, from which `code_ref = demo-import-smoke @ <sha7>` is spelled) are READ FROM THE RECEIPT;
`--cells`, `--launcher-cells` and `--override-id`, when passed (07 passes what its own call resolved), are CROSS-CHECKS that REFUSE by name
on a disagreement — so `07 --phase file` without the run's `--cells` (every cell of the view by default) cannot file seven objects as
exercised by a two-cell smoke, and a later HEAD, a later commit of objects/ or another overrides document cannot re-describe the execution.

The spec (sisteer.registry.analysis.SPEC_KEYS, every value a function of the receipt and the verdict file alone, so a file-only retry
composes byte-identical bytes): `name = demo_import_07_bcb_bfcl_<view>_<OVERRIDE_ID>_x<receipt12>` (the first 12 hex of sha256(receipt.json)
— the receipt IS the execution's identity: a new execution writes a new receipt, a new name, a new analysis_id; a retry a duplicate at the
fold), `code_sha` = the full tree sha the RECs name, `code_ref`, `substrate: modal`, `cluster_job_ids: []`, `report = {uri: the verdict
file, sha256: null}` (hashed at filing, registered as the report artifact), `inputs = {runs: [], objects: [the receipt's object id of every
fitted cell the run exercised, "none"]}`, `extra = {modal_app_name, modal_environment, modal_call_ids, modal_prior_attempts,
modal_reused_attempts, modal_other_tree_attempts, modal_uncredited_attempts, modal_credited_completions, launcher_slug, sampling, store_cell_keys, store_out_dirs,
cells, launcher_cells, objects_by_cell, controls,
n_items_per_task, done_markers, headline, bfcl, receipt_sha256, run_report_sha256 (the snapshot's, from the receipt), launcher_exit: 0,
expect_hashes, run_overrides_id, run_overrides_sha256, execution, verdict {ruled_utc, plausible_range, verdict}, spec_deviation, mirror}`.
Written to <state>/07_analysis_spec.json.

THE MIRROR (the stores are COPIED off the volume to the grading host and the cluster, sha-verified, and the
filing records where the verified copy lives): a filing REQUIRES the verified mirror of THIS receipt — <state>/07_mirror/<receipt sha12>/ with
its MIRROR.sha256, mirror.json (receipt_sha256 == this receipt's) and cluster.json (the cluster copy verified by `sha256sum -c` there, its
digest of MIRROR.sha256 equal to the local one), as `07 --phase mirror` (reproduce/tools/stage07_mirror.py) leaves them — and REFUSES by name,
with the exact mirror command, when none exists, so every new filing points at a durable copy; that path is held PLAIN under the state root
first (stage07_mirror.local_guard: a symlink at 07_mirror or at <receipt sha12>, or a directory resolving elsewhere, is REDIRECTED
and refused by name — before it the filing certified a link to the moved copy as the copy at the recorded path). `extra.mirror` carries the pointer: {local_path,
cluster_label, ssh_host, cluster_root, cluster_path, mirror_sha256sum_digest, mirror_json_sha256, receipt_sha256, files, bytes, tree_sha,
verified_utc, volume, modal_environment, store_volume_paths} — every value read from those two records, so a file-only retry still composes
byte-identical bytes; the read-back holds extra.mirror's FACTS (MIRROR_FACTS: local_path, cluster_label, ssh_host, cluster_root, cluster_path,
mirror_sha256sum_digest, mirror_json_sha256, receipt_sha256 — where EACH copy lives, the root it is guarded under and what it is; verified_utc alone is the grading host's bookkeeping, the same copy
re-verified) to the current verified mirror's, each disagreement a finding by name (comparing the digest alone let a
row naming the FORMER cluster path read back PASS after the documented retry pushed the same bytes to a new destination; treating local_path
as bookkeeping let a row, after the grading host's state root MOVED, keep naming a local copy that verified nowhere and read back PASS
on the cluster facts alone — now a finding by name, lifted the way the cluster-path one is, by a registry-supported correction of the row's
extra, or by the state root standing where the row says; nothing here rewrites a registry file). `--cluster-label` / `--ssh-host` /
`--cluster-root` / `--cluster-dest`, when passed (07 passes the site's primary cluster, its expanded reproduce root and <reproduce
root>/<model>/07_mirror — <reproduce root>/<model>/07_views/<view>/07_mirror for a dose view, the state tree's layout), are CROSS-CHECKS: the
cluster record's target must be this site's, else refused by name (otherwise a site whose destination had moved would
file the former path). THE CLUSTER COPY IS RE-VERIFIED THERE before anything is composed (stage07_mirror.verify_cluster_copy, over
the RECORDED ssh host: the guard under the recorded cluster root — no symlink at the copy or at any component below the root — before the cd,
then `sha256sum -c MIRROR.sha256` in the recorded cluster path and the three records' digests read back; earlier the filing read
cluster.json and never touched the copy) — a copy that is redirected, missing, altered or on a host that does not answer refuses by name;
`--dry-run` prints the check it would run and composes from the records as they stand.

A FILING THAT PREDATES THE MIRROR PHASE (the 2026-10-11 smoke's verdict was filed before `--phase mirror` existed, and
its row's extra carries no `mirror`): the registry's analysis rows are facts the fold never rewrites (INSERT OR IGNORE; a re-filing of the
same analysis_id is a duplicate), and this tool changes no registry file, so the pointer CANNOT be attached to that row here. The policy,
pinned: in the duplicate branch (same name, same report sha), when the filed row's extra has NO `mirror` key, the phase reports it BY NAME
as a filing made before the mirror phase — the analysis_id, where it stands (folded / pending), its filed_utc — prints the verified copy's
pointer, writes it to <state>/07_filing_pointers/<name>.json ({schema stage07-filing-pointer/2, name, analysis_id, filed_utc, filed_where,
status, mirror}) so the binding analysis -> copy is durable on the grading host, reads the row (or the pending header) back in every OTHER
field, and then ENDS IN A FINDING (exit 3), never a PASS: the registry record carries no pointer to the verified copy, so a registry reader
cannot locate the stores from it, and a success here would say otherwise (earlier cuts exited 0 with a note — a pass that
left the registry without the binding). The finding names the absent extra.mirror the way a hand-edited header is named, the pointer record,
and what lifts it: the row gaining extra.mirror through a registry mechanism (an amend of an analysis row's extra, the registry's owners'
call) — nothing here rewrites a registry file. A row whose extra DOES carry a `mirror` with another digest is a finding (exit 3) as before.
ONE RECORD PER ANALYSIS, WRITTEN ONCE (a single <state>/07_filing_pointer.json was overwritten by every later
pre-mirror filing of the model — the state root is per model, the name is per view, overrides id and receipt — so the first analysis lost
its only pointer): the record is keyed by the analysis name, created exclusively and never rewritten; a standing record whose name,
analysis_id and mirror are this run's stands as it is (its filed_where / filed_utc are the filing's standing when it was written), and one
that disagrees REFUSES by name, listing the fields, the record left intact — a deliberate change (the mirror's cluster record re-pushed to
a moved destination, a moved state root) is recorded and the old record moved OUT of the directory before the re-run. A
pointer attached to an existing row is a registry change (an amend of an analysis row's extra) the registry's owners would define; until
then the pointer records are its record.

BEFORE writing, the name is looked for in BOTH places a filing can be — the FOLD (`registry analysis list --name <name> --json`) and the
PENDING record log (every *-analysis.jsonl[.gz] header under <root>/records/analyses/*/*/ whose payload.name is the name; `analysis file`
writes records, never the database, and `analysis list` reads the fold alone). Same name + the same report_sha256 -> a duplicate, nothing
written (the read-back still runs); same name + ANOTHER report_sha256 -> REFUSED by name (a re-ruled verdict of an already-filed
execution: the registry holds one report per analysis — names the filed analysis_id, both shas, the filed verdict's ruled_utc and, for a
pending filing, its directory; a deliberate second ruling is filed by hand from the composed spec under a name the operator records);
neither -> `registry analysis file --spec <spec> --json` against the registry root, printing analysis_id / filing_dir. The READ-BACK is
three-way, from what the CLI returns: `analysis list --name <name> --json` for the row, its flat `inputs` and `artifacts`, and `registry
query --json "SELECT analysis_id, extra_json FROM analysis WHERE name = '<name>'"` for the parked extra — asserting THE IDENTITY (the row's
analysis_id == sisteer.registry.analysis.analysis_id(normalize_spec(the composed spec)) and its code_sha == this filing's, the receipt's
tree; a same-name, same-report filing at ANOTHER code_sha is another analysis_id, which the pre-write check took for this filing
and the read-back never compared), report_sha256 == sha256(the verdict file), substrate modal, cluster_job_ids_json == [], the object
inputs, extra.receipt_sha256, extra.headline and extra.mirror's MIRROR_FACTS against the current verified mirror (a filing that predates
the mirror phase is read back in every other field, and its absent extra.mirror is then the finding, above); a row -> PASS (exit 0); no row
but a pending header -> THE HEADER IS READ BACK THE SAME WAY (its payload's analysis_id and code_sha, report_sha256, substrate,
cluster_job_ids_json, the filing's analysis_input records against the object inputs, extra.receipt_sha256, extra.headline and the mirror
facts; a pending header was once accepted unread, so one carrying ANOTHER copy's digest filed as complete and the mismatch
surfaced only after the fold), agreeing -> "filed, not yet folded" (exit 0; the grading host's next `registry ingest` folds it; a re-run
reads the row back); neither, or differing fields in the row or the header -> a finding (exit 3). A duplicate whose analysis_id is not this
filing's gets no pointer record either (the predates branch binds a pointer to an analysis; it must be THIS one).
This phase never runs `registry ingest` and never extends `analysis list --json`. `--dry-run` composes and writes the spec, prints the
pre-write check and the filing command, and touches no registry.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMPORTER = HERE / "import_objects.py"
MIRROR_TOOL = HERE / "stage07_mirror.py"
STORE_DIR = "07_store"
RECEIPT_NAME = "receipt.json"
RUN_REPORT_NAME = "07_run_report.log"
SPEC_NAME = "07_analysis_spec.json"
FAMILY_PREFIX = "bcb_bfcl"
NAME_PREFIX = "demo_import_07_bcb_bfcl"
NAME_RE = re.compile(r"^[a-z0-9_]+$")
SPEC_DEVIATION = "07 registers no run rows (C10); filed as an analysis per Q13"
#: the receipt schema that carries the launch's identity (cells, view_cells, objects, run_overrides_id), since /4 names the attempts a DONE
#: reuse credits in modal_call_ids, since /5 attributes every reused control through the launcher's completion
#: records (a /4 reuse receipt omitted the BFCL-only call that produced BFCL), since /6 joins the credited pointers to those
#: records by call id (a /5 receipt credited a failed standup's pointer as a producer) and since /7 describes a run of the paper's
#: FULL sets under the protocol decoding with the launcher slug (a /6 receipt is a 200-item execution's); an older
#: receipt is re-made — or, for /6, the run re-done on the full sets — never guessed at
RECEIPT_SCHEMA = "stage07-receipt/7"
CODE_REF_LABEL = "demo-import-smoke"
EXIT_REFUSED, EXIT_FINDING = 2, 3


class FileRefusal(Exception):
    """A refusal by name: nothing written, nothing filed."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path, what: str) -> dict:
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise FileRefusal(f"{what}: {p} is not readable JSON ({e})") from None
    if not isinstance(doc, dict):
        raise FileRefusal(f"{what}: {p} is not a JSON object")
    return doc


def _registry(repo: Path):
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.registry import analysis as AN  # noqa: WPS433
    from sisteer.registry import cli  # noqa: WPS433
    from sisteer.registry import harvest as H  # noqa: WPS433
    from sisteer.registry import records as R  # noqa: WPS433
    return AN, cli, H, R


def _contracts(repo: Path):
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.capeval.tinycap import contracts as C  # noqa: WPS433
    return C


def _importer():
    spec = importlib.util.spec_from_file_location("_stage07_file_import_objects", IMPORTER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_MIRROR_TOOL_MODULE = None


def _mirror_tool():
    global _MIRROR_TOOL_MODULE
    if _MIRROR_TOOL_MODULE is None:
        spec = importlib.util.spec_from_file_location("_stage07_file_mirror", MIRROR_TOOL)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _MIRROR_TOOL_MODULE = mod
    return _MIRROR_TOOL_MODULE


POINTER_DIR = "07_filing_pointers"
POINTER_SCHEMA = "stage07-filing-pointer/2"
#: the fields that BIND a pointer record to one analysis and one copy: a standing record agreeing on these stands, one disagreeing refuses
POINTER_FACTS: Tuple[str, ...] = ("name", "analysis_id", "mirror")
PREDATES_STATUS = "filed before the mirror phase: the registry row carries no extra.mirror; this record is the pointer's — one per analysis, written once"
#: what lifts the predates finding — the one change this tool cannot make (header: A FILING THAT PREDATES THE MIRROR PHASE)
PREDATES_LIFT = ("the registry record gaining extra.mirror through a registry mechanism (an amend of an analysis row's extra, the registry's owners' call) — "
                 "nothing here rewrites a registry file")
#: the facts of a filed extra.mirror the read-back holds to the CURRENT verified mirror's — where each copy lives (the grading host's path, the
#: cluster's label, host and path) and what it is (the two digests, the receipt); verified_utc alone (the same copy re-verified) is the grading
#: host's bookkeeping and is not compared (comparing the digest alone let a row naming the former cluster path read back
#: PASS after the retry re-pushed the same bytes; leaving local_path out as bookkeeping let a row naming the former grading-host copy —
#: the state root moved — read back PASS while that recorded location verified nowhere)
MIRROR_FACTS: Tuple[str, ...] = ("local_path", "cluster_label", "ssh_host", "cluster_root", "cluster_path", "mirror_sha256sum_digest", "mirror_json_sha256", "receipt_sha256")


def _mirror_diffs(extra: Mapping[str, Any], mirror: Mapping[str, Any]) -> List[str]:
    """The MIRROR_FACTS on which a filed extra.mirror and the current verified mirror disagree, one line each (`extra.mirror.<fact> <filed> !=
    the verified mirror's <current>`); a filing with no `mirror` at all names that."""
    filed = extra.get("mirror")
    if not isinstance(filed, dict):
        return [f"extra.mirror {filed!r} != the verified mirror's pointer (no copy recorded)"]
    return [f"extra.mirror.{k} {filed.get(k)} != the verified mirror's {mirror.get(k)}" for k in MIRROR_FACTS if filed.get(k) != mirror.get(k)]


def mirror_pointer(state: Path, receipt_sha: str, *, key: str, view: str, cells: Optional[Sequence[str]], target: Optional[Mapping[str, str]] = None,
                   runner=None, dry_run: bool = False, log=print) -> Dict[str, Any]:
    """The verified mirror of THIS receipt (stage07_mirror.verified_mirror: the path <state>/07_mirror/<sha12> held plain under the state root
    first — a symlink at 07_mirror or at <sha12>, or a directory resolving elsewhere, is REDIRECTED, refused by name with the link and its
    target, round 7 — then every data file re-hashed against MIRROR.sha256, the two records and the cluster record consistent) as the pointer
    the filing records; refused by name, with the exact mirror command, when the receipt has none — a filing names a durable copy, at the path
    it records. `target` ({cluster_label, ssh_host, cluster_root, cluster_path}), when given, is the site's: a record
    naming another target refuses by name (stage07_mirror.check_target). Then THE CLUSTER COPY ITSELF is re-verified over the recorded ssh host
    (stage07_mirror.verify_cluster_copy: the guard under the recorded cluster root before the cd — a symlink at the copy or at any component
    below the root is REDIRECTED, round 6 — then `sha256sum -c MIRROR.sha256` in the recorded path, the three records' digests read back; round
    3) — redirected, missing, altered or unreachable refuses by name; a dry run prints the check and composes from the records as they stand."""
    MI = _mirror_tool()
    try:
        ptr = MI.verified_mirror(Path(state), receipt_sha)
    except MI.Refusal as e:
        if " is REDIRECTED: " in str(e):
            # round 7: the grading-host path is a link (or resolves elsewhere) — not a missing mirror; running the mirror again refuses the same way
            raise FileRefusal(f"{e} — nothing filed against a grading-host mirror whose path is redirected: the filing would record the link's path as the durable "
                              "copy (extra.mirror.local_path), and `07 --phase mirror` refuses it the same way until the copy stands at its path") from None
        sel = f" --cells {','.join(cells)}" if cells else ""
        raise FileRefusal(f"no verified mirror of receipt {receipt_sha[:12]} ({e}) — every filing points at a durable copy of the run's stores: run "
                          f"`reproduce/07_bcb_bfcl.sh --model {key} --view {view}{sel} --phase mirror` first (it pulls the whole stores off the volume into "
                          f"{Path(state) / MI.MIRROR_DIR / receipt_sha[:12]}/, verifies them against this receipt and pushes them to the primary cluster, "
                          f"`sha256sum -c` there), then file again") from None
    if target is not None:
        try:
            MI.check_target(ptr, target, what=f"this site's ({target.get('cluster_label')} {target.get('ssh_host')}:{target.get('cluster_path')})")
        except MI.Refusal as e:
            raise FileRefusal(f"{e}; nothing filed against a copy this site does not name") from None
    if dry_run:
        log(f"DRY-RUN: {MI.ssh_shown(ptr['ssh_host'], MI.copy_check_command(ptr['cluster_path'], root=ptr['cluster_root']))}   (the cluster copy re-verified there before a "
            f"real filing: the guard under the recorded cluster root before the cd — no symlink at the copy or at any component below the root — then {MI.SUMS_NAME}, "
            f"{MI.MIRROR_NAME} and {MI.CLUSTER_NAME} read back by digest; the inventory of every entry by type held to {MI.SUMS_NAME} plus the records and "
            f"{MI.MIRROR_NAME}'s directories)")
        return ptr
    try:
        MI.verify_cluster_copy(ptr, runner=runner or MI.sh_runner, log=log)
    except MI.Refusal as e:
        raise FileRefusal(f"{e} — nothing filed against a cluster copy that does not verify as recorded: when the copy is gone or altered, record it in decisions.md "
                          f"against receipt {receipt_sha[:12]}, move {ptr['local_path']}/{MI.CLUSTER_NAME} OUT of the mirror directory (anywhere else, named in the "
                          f"note) and run `reproduce/07_bcb_bfcl.sh --model {key} --view {view}{' --cells ' + ','.join(cells) if cells else ''} --phase mirror` again "
                          "to push the published mirror afresh, then file; when the host did not answer, file again once it does") from None
    log(f"cluster copy re-verified: {ptr['ssh_host']}:{ptr['cluster_path']} (sha256sum -c {MI.SUMS_NAME}: {ptr['files']} file(s) OK; the three records read back by digest)")
    return ptr


def filed_extra(cli, root: str, f: Tuple) -> Dict[str, Any]:
    """The `extra` of a filing found by the pre-write check: a pending header's payload.extra as read, a folded row's extra_json queried."""
    if f[0] == "pending":
        return dict((f[5].get("extra") or {}))
    rc, out, _err = _cli(cli, ["--root", root, "query", "--json", f"SELECT extra_json FROM analysis WHERE analysis_id = '{f[1]}'"])
    try:
        return dict(json.loads(json.loads(out)[0]["extra_json"]) if rc == 0 else {})
    except (ValueError, KeyError, IndexError, TypeError):
        return {}


def pointer_record_path(state: Path, name: str) -> Path:
    """<state>/07_filing_pointers/<name>.json — the analysis name ([a-z0-9_]+, NAME_RE) carries the view, the overrides id and the receipt."""
    return Path(state) / POINTER_DIR / f"{name}.json"


def _pointer_diffs(have: Mapping[str, Any], want: Mapping[str, Any]) -> List[str]:
    """The binding facts on which a standing record and this run disagree, one line each (`mirror` by its sub-keys)."""
    out: List[str] = []
    for k in POINTER_FACTS:
        a, b = have.get(k), want.get(k)
        if a == b:
            continue
        if k == "mirror" and isinstance(a, dict) and isinstance(b, dict):
            out += [f"mirror.{sk}: recorded {a.get(sk)!r}, this run {b.get(sk)!r}" for sk in sorted(set(a) | set(b)) if a.get(sk) != b.get(sk)]
        else:
            out.append(f"{k}: recorded {a!r}, this run {b!r}")
    return out


def write_pointer_record(state: Path, *, name: str, analysis_id: Optional[str], filed_utc: Optional[str], where: str, mirror: Mapping[str, Any]) -> Tuple[Path, bool]:
    """<state>/07_filing_pointers/<name>.json: the durable binding of ONE pre-mirror filing to the verified copy of its receipt (header: A
    FILING THAT PREDATES THE MIRROR PHASE). One record per analysis, created exclusively and NEVER rewritten (a single
    file per state root was overwritten by every later pre-mirror filing of the model): a standing record whose POINTER_FACTS — name,
    analysis_id, mirror — are this run's stands as it is (its filed_where / filed_utc are the filing's standing when it was written), and one
    that disagrees refuses by name with every differing field, the record left intact. Returns (path, True when written by this call)."""
    p = pointer_record_path(state, name)
    doc = {"schema": POINTER_SCHEMA, "name": name, "analysis_id": analysis_id, "filed_utc": filed_utc, "filed_where": where, "status": PREDATES_STATUS,
           "mirror": dict(mirror), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "written_by": "reproduce/tools/stage07_file.py"}
    if p.exists():
        have = read_json(p, "the filing pointer record")
        diffs = _pointer_diffs(have, doc)
        if diffs:
            raise FileRefusal(f"the pointer record {p} (written {have.get('written_utc')}) binds analysis {have.get('analysis_id')} to another copy than this run's: "
                              f"{'; '.join(diffs)} — a pointer record is written once and never rewritten; when the change is deliberate (the mirror's cluster "
                              f"record re-pushed to a moved destination, a moved state root), record it in decisions.md against the analysis_id, move {p} OUT of "
                              f"{p.parent} (anywhere else, named in the note) and run `07 --phase file` again")
        return p, False
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)       # exclusive: a record that appeared meanwhile is compared, never clobbered
    except FileExistsError:
        return write_pointer_record(state, name=name, analysis_id=analysis_id, filed_utc=filed_utc, where=where, mirror=mirror)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return p, True


def run_identity(receipt: dict, *, key: str, view: str, repo: Path, table: Optional[str]) -> Dict[str, Any]:
    """The execution's identity READ FROM THE RECEIPT: the launcher tokens and the table names of the cells it ran (held equal through the
    table's own mapping), the run-overrides id, the app, the tree and the code_ref spelled from it, the launcher row (`slug`: held to the
    table's launcher_slug for (key, view)) and the sampling mode (held to the paper's protocol). An older receipt, a missing field, a receipt
    whose two cell lists do not map onto each other, another row's slug or another sampling mode refuses by name."""
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise FileRefusal(f"the receipt's schema is {receipt.get('schema')!r}, not {RECEIPT_SCHEMA!r} — an older receipt carries no launch identity (cells / view_cells / "
                          "objects / run_overrides_id; before /3), may credit a DONE reuse to an unrelated attempt and name no call id (before /4), may omit "
                          "the BFCL-only call that produced a reused BFCL store (before /5), may credit a failed standup's pointer as a producer (before /6), "
                          "or describes a run of the 200-item subsets under the launcher's card sampling (before /7: the demo follows the paper's full sets "
                          "and protocol decoding — re-run 07 on them); run --phase receipt again over this execution's log")
    # the receipt describes ONE view's execution (`view`, copied from the launch identity the receipt verified): a receipt read from another
    # view's state directory — the wave's filed as a dose view, or the reverse — is refused by name before any other field is compared
    if receipt.get("view") != view:
        raise FileRefusal(f"the receipt describes view {receipt.get('view')!r}, not {view!r} — it is another view's execution (07 keeps each view's receipt, mirror "
                          "and filing under that view's state directory: the model's for the baseline wave, 07_views/<view>/ for a dose view); run --phase file "
                          "for the view the receipt names, from its own state directory")
    tokens = receipt.get("cells")
    vcells = receipt.get("view_cells")
    oid = receipt.get("run_overrides_id")
    tree = receipt.get("tree_sha")
    objects = receipt.get("objects")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) and t for t in tokens):
        raise FileRefusal("the receipt names no run cells (`cells`)")
    if not isinstance(vcells, list) or len(vcells) != len(tokens) or not all(isinstance(c, str) and c for c in vcells):
        raise FileRefusal(f"the receipt's view_cells {vcells!r} are not the run's cells by the table's names ({len(tokens)} tokens)")
    if not isinstance(oid, str) or not oid:
        raise FileRefusal("the receipt names no run_overrides_id")
    if not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise FileRefusal(f"the receipt's tree_sha {tree!r} is not a 40-hex sha")
    if objects is not None and not (isinstance(objects, dict) and all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in objects.items())):
        raise FileRefusal(f"the receipt's `objects` {objects!r} is not a map of launcher token -> 64-hex object id")
    MT = _importer()._load_models_module(repo)
    try:
        mapped = [tok for _c, tok in MT.launcher_cells(key, view, cells=list(vcells), path=table)]
    except (KeyError, ValueError) as e:
        raise FileRefusal(f"the receipt's view_cells {vcells} do not map to launcher tokens through the models table: {e}") from None
    if mapped != tokens:
        raise FileRefusal(f"the receipt's view_cells {vcells} map to the launcher tokens {mapped} through the models table, not the receipt's cells {tokens} — "
                          "the table changed under the receipt, or the receipt is not this table's")
    # the launcher row the run was served from (the receipt's `slug`, copied from the launch identity) must be the row the table names for
    # (key, view) — the key on the wave, `<key>-<view>` on a dose view (lib/models.py launcher_slug): a receipt served from another row (a dose
    # view's run filed under the wave, or the reverse) would file the wrong bank's objects under the view's name; and the run's `sampling` must
    # be the paper's protocol decoding (the receipt refuses any other at --phase receipt; held again here so a hand-edited receipt cannot pass)
    slug = receipt.get("slug")
    try:
        want_slug = MT.launcher_slug(key, view, table)
    except KeyError as e:
        raise FileRefusal(f"the models table names no launcher row for {key}/{view}: {e}") from None
    if slug != want_slug:
        raise FileRefusal(f"the receipt's slug {slug!r} is not the launcher row the models table serves {key}/{view} from ({want_slug!r}) — the run was served "
                          "from another row's bank (another view's, or an earlier 07 wrote no slug); the receipt is not this view's")
    C = _contracts(repo)
    if receipt.get("sampling") != C.SAMPLING_PROTOCOL:
        raise FileRefusal(f"the receipt's sampling {receipt.get('sampling')!r} is not the paper's {C.SAMPLING_PROTOCOL!r} decoding — not a run on the paper's protocol")
    return {"cells": list(vcells), "launcher_cells": list(tokens), "override_id": oid, "override_sha256": receipt.get("run_overrides_sha256"), "app": receipt.get("app"),
            "tree_sha": tree, "code_ref": f"{CODE_REF_LABEL} @ {tree[:7]}", "environment": receipt.get("modal_environment"), "objects": objects,
            "slug": slug, "sampling": receipt.get("sampling")}


def _git_head(repo: Path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    head = r.stdout.strip()
    return head if r.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def check_call_against_receipt(identity: Mapping[str, Any], *, cells: Optional[Sequence[str]], launcher_cells: Optional[Sequence[str]], override_id: Optional[str]) -> None:
    """The caller's arguments, when given, must be the receipt's — a disagreement refuses before anything is composed."""
    for what, given, want, flag in (("cells", cells, identity["cells"], "--cells"), ("launcher cells", launcher_cells, identity["launcher_cells"], "--launcher-cells"),
                                    ("run-overrides id", override_id, identity["override_id"], "--override-id")):
        if given is None:
            continue
        g = list(given) if not isinstance(given, str) else given
        if g != want:
            shown = ",".join(want) if isinstance(want, list) else want
            raise FileRefusal(f"{flag} {','.join(g) if isinstance(g, list) else g} is not the receipt's: the execution ran with {what} {shown} — pass "
                              f"{flag} {shown} (what the receipt records), or run --phase receipt for the execution you mean; the filing describes the receipt's run, never this call's")


def exercised_objects(identity: Mapping[str, Any], *, key: str, view: str, repo: Path, table: Optional[str], log) -> List[str]:
    """The object id of every FITTED launcher token the run exercised — THE RECEIPT'S map (resolved at launch from the run's tree), plus the
    `none` object row when the run had the none cell; prompted cells carry no object. The live checkout is consulted only as a cross-check,
    and only when its HEAD is the receipt's tree (the importer's plan by the table's cell name must then agree); at another HEAD the map is
    filed as recorded, since this checkout's objects/ may have moved since the run."""
    C = _contracts(repo)
    tokens, cells, objects, tree = identity["launcher_cells"], identity["cells"], identity["objects"], identity["tree_sha"]
    fitted = [t for t in tokens if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    if fitted and objects is None:
        raise FileRefusal(f"the receipt identifies no objects for the run's fitted cells {fitted} (`objects` is null: the launch resolved none — the model declares no "
                          "committed objects, so the run's object ids are not the importer's); a filing needs the exercised objects' ids")
    out: List[str] = ["none"] if "none" in tokens else []
    for tok in fitted:
        if tok not in (objects or {}):
            raise FileRefusal(f"the receipt's `objects` has no entry for the fitted cell {tok!r} (entries: {sorted(objects or {})}) — the receipt is not this run's")
        out.append(objects[tok])
    head = _git_head(repo)
    if head == tree:
        IO = _importer()
        try:
            plan = IO.Plan(key, views=[view], table=table, repo=repo)
        except SystemExit as e:
            raise FileRefusal(f"this checkout is at the receipt's tree {tree[:7]} but the committed objects of {key}/{view} do not plan here: {e}") from None
        by_view_cell = {r["view_cell"]: r["object_sha256_expected"] for r in plan.rows}
        for cell, tok in zip(cells, tokens):
            if tok in fitted and by_view_cell.get(cell) != objects[tok]:
                raise FileRefusal(f"the receipt's object for {tok} (cell {cell}) is {objects[tok][:12]}, but the checkout at the receipt's own tree {tree[:7]} maps the cell to "
                                  f"{str(by_view_cell.get(cell))[:12]} — the receipt's objects are not this tree's; run --phase receipt over the execution's log again")
        log(f"objects: the receipt's ({len(fitted)} fitted cell(s)), cross-checked against the importer's plan at HEAD {head[:7]} == the receipt's tree: agree")
    else:
        log(f"objects: the receipt's ({len(fitted)} fitted cell(s)), resolved at launch from tree {tree[:7]}; this checkout is at {head[:7] if head else 'no git HEAD'}, "
            "not consulted (its objects/ may have moved since the run)")
    return sorted(dict.fromkeys(out))


def verify_store(state: Path, receipt: dict) -> None:
    """Every pulled file and the run report's copy re-hashed against the receipt; the live run report, when present, must be the receipt's."""
    store = state / STORE_DIR
    for cpath, rec in (receipt.get("pulled") or {}).items():
        p = store / rec["local"]
        if not p.is_file():
            raise FileRefusal(f"{p} (the pull of {cpath}) is missing under {store} — the receipt's evidence is incomplete; run --phase receipt again")
        got = sha256_file(p)
        if got != rec["sha256"]:
            raise FileRefusal(f"{p} hashes to {got[:12]}, the receipt recorded {rec['sha256'][:12]} — the pulled evidence changed under the receipt")
    copy = store / (receipt.get("run_report") or {}).get("copy", "run_report.log")
    if not copy.is_file():
        raise FileRefusal(f"the receipt's run report copy {copy} is missing")
    if sha256_file(copy) != receipt.get("run_report_sha256"):
        raise FileRefusal(f"{copy} is not the receipt's run report (sha {sha256_file(copy)[:12]} vs {str(receipt.get('run_report_sha256'))[:12]})")
    live = state / RUN_REPORT_NAME
    if live.is_file() and sha256_file(live) != receipt.get("run_report_sha256"):
        raise FileRefusal(f"the run report on disk is not this receipt's: a later `--phase run` wrote {live} (sha {sha256_file(live)[:12]}, the receipt's "
                          f"{str(receipt.get('run_report_sha256'))[:12]}); run `07 --phase receipt` for that execution and rule its verdict, or move the live log "
                          "aside to file this receipt")


def verify_verdict(verdict: dict, receipt: dict, receipt_sha: str, view: str) -> None:
    want_family = f"{FAMILY_PREFIX}/{view}"
    if verdict.get("family") != want_family:
        raise FileRefusal(f"the verdict's family {verdict.get('family')!r} is not {want_family!r}")
    hv = receipt["headline"]["value"]
    if verdict.get("headline") != hv:
        raise FileRefusal(f"the verdict's headline {verdict.get('headline')!r} is not the receipt's none BCB acc {hv!r} — re-run stage.py verdict from the receipt's printed command")
    if sorted(verdict.get("run_ids") or []) != list(receipt["store_cell_keys"]):
        raise FileRefusal(f"the verdict's run_ids {sorted(verdict.get('run_ids') or [])} are not the receipt's store keys {receipt['store_cell_keys']}")
    tag = f"receipt {receipt_sha[:12]}"
    if tag not in str(verdict.get("note") or ""):
        raise FileRefusal(f"the verdict's note {verdict.get('note')!r} does not carry {tag!r} — it was ruled from another receipt (or by hand); re-run stage.py verdict "
                          "from this receipt's printed command")
    if "plausible_range" not in verdict or "verdict" not in verdict or "ruled_utc" not in verdict:
        raise FileRefusal("the verdict file lacks plausible_range / verdict / ruled_utc — not stage.py verdict's record")


def compose_spec(*, state: Path, view: str, key: str, identity: Mapping[str, Any], receipt: dict, receipt_sha: str, verdict: dict, verdict_path: Path,
                 objects: List[str], mirror: Mapping[str, Any]) -> Dict[str, Any]:
    override_id = identity["override_id"]
    name = f"{NAME_PREFIX}_{view}_{override_id}_x{receipt_sha[:12]}"
    if not NAME_RE.match(name):
        raise FileRefusal(f"the analysis name {name!r} carries characters outside [a-z0-9_] (the view or OVERRIDE_ID does) — the read-back queries it by name")
    extra = {
        "modal_app_name": identity["app"], "modal_environment": identity["environment"], "modal_call_ids": list(receipt["modal_call_ids"]),
        "launcher_slug": identity["slug"], "sampling": identity["sampling"],
        "modal_prior_attempts": [a.get("function_call_id") for a in receipt.get("prior_attempts") or []],
        "modal_reused_attempts": [a.get("function_call_id") for a in receipt.get("reused_attempts") or []],
        "modal_other_tree_attempts": [a.get("function_call_id") for a in receipt.get("other_tree_attempts") or []],
        "modal_uncredited_attempts": [a.get("function_call_id") for a in receipt.get("uncredited_attempts") or []],
        "modal_credited_completions": sorted({a.get("function_call_id") for a in receipt.get("credited_completions") or [] if a.get("function_call_id")}),
        "store_cell_keys": list(receipt["store_cell_keys"]), "store_out_dirs": list(receipt["store_out_dirs"]),
        "cells": list(identity["cells"]), "launcher_cells": list(identity["launcher_cells"]), "objects_by_cell": dict(identity["objects"] or {}),
        "controls": ",".join(receipt["controls"]),
        "n_items_per_task": dict(receipt["n_items_per_task"]), "done_markers": {t: r.get("done_markers") for t, r in receipt["recs"].items()},
        "headline": dict(receipt["headline"]), "bfcl": dict(receipt["bfcl"]), "receipt_sha256": receipt_sha, "run_report_sha256": receipt["run_report_sha256"],
        "launcher_exit": 0, "expect_hashes": receipt.get("expect_hashes"), "run_overrides_id": override_id, "run_overrides_sha256": identity.get("override_sha256"),
        "execution": receipt.get("execution"),
        "verdict": {"ruled_utc": verdict.get("ruled_utc"), "plausible_range": verdict.get("plausible_range"), "verdict": verdict.get("verdict")},
        "spec_deviation": SPEC_DEVIATION,
        "mirror": dict(mirror),                      # the verified copy of the stores this filing points at (header: THE MIRROR)
    }
    return {"name": name, "code_sha": identity["tree_sha"], "code_ref": identity["code_ref"], "cluster_job_ids": [], "substrate": "modal",
            "report": {"uri": str(verdict_path), "sha256": None}, "inputs": {"runs": [], "objects": objects}, "artifacts": [],
            "filed_by": "the user", "provenance": {"kind": "user_direct"},
            "notes": f"demo-import stage 07 ({key}/{view}): the gate verdict of a Modal run of the full BigCodeBench and BFCL sets (1,140 and 3,981 items, protocol "
                     "decoding), filed by 07 --phase file from the receipt",
            "extra": extra}


def _cli(cli, argv: List[str]) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = int(cli.main(argv))
        except SystemExit as e:
            rc = int(e.code or 0) if not isinstance(e.code, str) else 1
    return rc, out.getvalue(), err.getvalue()


def folded_rows(cli, root: str, name: str) -> List[dict]:
    rc, out, err = _cli(cli, ["--root", root, "analysis", "list", "--name", name, "--json"])
    if rc != 0:
        raise FileRefusal(f"registry analysis list under {root} failed ({rc}): {err.strip()[-400:]}")
    try:
        return [r for r in json.loads(out or "[]") if r.get("name") == name]
    except ValueError as e:
        raise FileRefusal(f"registry analysis list --json printed no JSON ({e})") from None


def _filing_inputs(R, filing_dir: str, analysis_id: Optional[str]) -> List[Tuple[str, str]]:
    """The (input_kind, input_id) pairs of a filing's `analysis_input` records (the files beside its header, by its analysis_id), sorted."""
    pairs: List[Tuple[str, str]] = []
    for path in R.iter_record_files(filing_dir):
        if not (path.endswith("-analysis_input.jsonl") or path.endswith("-analysis_input.jsonl.gz")):
            continue
        for _line, rec, _sha in R.read_record_file(path, compute_sha=False):
            payload = rec.get("payload") or {}
            if rec.get("record_type") == "analysis_input" and payload.get("analysis_id") == analysis_id:
                pairs.append((str(payload.get("input_kind")), str(payload.get("input_id"))))
    return sorted(pairs)


def pending_headers(AN, H, R, root: str, name: str) -> List[dict]:
    """Every pending `analysis` header under <root>/records/analyses/*/*/ whose payload.name is the name: {analysis_id, report_sha256, substrate,
    cluster_job_ids_json, code_sha, extra, filed_utc, dir, inputs} — `inputs` the filing's analysis_input records as (kind, id) pairs, so the
    header can be read back the way a folded row is (check_pending)."""
    base = os.path.join(H.records_dir(root), AN.SUBDIR)
    out: List[dict] = []
    if not os.path.isdir(base):
        return out
    for path in R.iter_record_files(base):
        if not (path.endswith("-analysis.jsonl") or path.endswith("-analysis.jsonl.gz")):
            continue
        for _line, rec, _sha in R.read_record_file(path, compute_sha=False):
            payload = rec.get("payload") or {}
            if rec.get("record_type") == "analysis" and payload.get("name") == name:
                d = os.path.dirname(path)
                out.append({"analysis_id": payload.get("analysis_id"), "report_sha256": payload.get("report_sha256"), "substrate": payload.get("substrate"),
                            "cluster_job_ids_json": payload.get("cluster_job_ids_json"), "code_sha": payload.get("code_sha"), "extra": payload.get("extra") or {},
                            "filed_utc": payload.get("filed_utc"), "dir": d, "inputs": _filing_inputs(R, d, payload.get("analysis_id"))})
    return out


def check_pending(h: Mapping[str, Any], *, expected_aid: str, report_sha: str, objects: List[str], receipt_sha: str, headline: dict,
                  mirror: Optional[Mapping[str, Any]], code_sha: Optional[str] = None) -> List[str]:
    """A pending header read back the way read_back reads a folded row (a pending header was once accepted unread): the
    problems, empty when the header is this filing's. The analysis_id is checked against the composed spec's — name, code_sha and inputs are
    the registry's identity (analysis.analysis_id), so one id covers the tree and the input set — the code_sha against this filing's (the
    receipt's tree) when given, the filing's analysis_input records against the object inputs, and extra.mirror's MIRROR_FACTS against the
    verified mirror (`mirror`; None for a filing that predates the mirror phase)."""
    problems: List[str] = []
    if str(h.get("analysis_id")) != expected_aid:
        problems.append(f"analysis_id {h.get('analysis_id')} != the composed spec's {expected_aid} (name, code_sha and inputs are the identity)")
    if code_sha is not None and h.get("code_sha") != code_sha:
        problems.append(f"code_sha {h.get('code_sha')} != this filing's {code_sha} (the receipt's tree)")
    if h.get("report_sha256") != report_sha:
        problems.append(f"report_sha256 {h.get('report_sha256')} != the verdict file's {report_sha}")
    if h.get("substrate") != "modal":
        problems.append(f"substrate {h.get('substrate')!r} != modal")
    jobs = h.get("cluster_job_ids_json")
    if isinstance(jobs, str):
        try:
            jobs = json.loads(jobs)
        except ValueError:
            pass
    if jobs != []:
        problems.append(f"cluster_job_ids_json {h.get('cluster_job_ids_json')!r} != []")
    if list(h.get("inputs") or []) != sorted(("object", o) for o in objects):
        problems.append(f"the filing's inputs {h.get('inputs')} != the object inputs {sorted(objects)} (and no run inputs)")
    extra = h.get("extra") or {}
    if extra.get("receipt_sha256") != receipt_sha:
        problems.append(f"extra.receipt_sha256 {extra.get('receipt_sha256')} != {receipt_sha}")
    if extra.get("headline") != headline:
        problems.append(f"extra.headline {extra.get('headline')} != the receipt's {headline}")
    if mirror is not None:
        problems += _mirror_diffs(extra, mirror)
    return problems


def predates_finding(name: str, aid: Optional[str], where: str, *, mirror: Mapping[str, Any], pointer: Path, log) -> int:
    """The end of the predates branch: the row or header is this filing's in every other field, the copy verifies, the
    pointer record stands — and the registry record still carries no pointer to the copy, so the phase ends in a FINDING (exit 3), never a
    PASS. The absent extra.mirror is named the way a hand-edited header is (_mirror_diffs), then the record, the pointer and what lifts it."""
    MI = _mirror_tool()                   # loaded by mirror_pointer before any filing reaches this point
    for p in _mirror_diffs({}, mirror):
        log(f"FINDING: {name}: {p}")
    log(f"FINDING: {name} is analysis {aid} ({where}), filed before the mirror phase: its registry record carries no pointer to the verified copy of receipt "
        f"{str(mirror.get('receipt_sha256'))[:12]} ({mirror.get('cluster_label')} {mirror.get('ssh_host')}:{mirror.get('cluster_path')}, {MI.SUMS_NAME} "
        f"{str(mirror.get('mirror_sha256sum_digest'))[:12]}), so a registry reader cannot locate the stores from it; every other field read back is this filing's and "
        f"the binding stands in this host's pointer record {pointer} — not a PASS: it is lifted by {PREDATES_LIFT}; record the pointer in decisions.md against analysis {aid}")
    return EXIT_FINDING


def read_back(cli, root: str, name: str, *, report_sha: str, objects: List[str], receipt_sha: str, headline: dict, log,
              mirror: Optional[Mapping[str, Any]] = None, expected_aid: Optional[str] = None, code_sha: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """('folded' | 'pending' | 'absent' | 'differs', analysis_id) — the three-way read-back from what the CLI returns. The folded row's
    IDENTITY is read back too (round 3): its analysis_id against `expected_aid` (the composed spec's) and its code_sha against this filing's,
    when given; and its extra.mirror's MIRROR_FACTS against `mirror`, the current verified mirror (None for a filing that predates the phase)."""
    rows = folded_rows(cli, root, name)
    if not rows:
        return "absent", None
    if len(rows) > 1:
        log(f"FINDING: {len(rows)} analysis rows carry the name {name!r}")
        return "differs", rows[0].get("analysis_id")
    row = rows[0]
    aid = str(row.get("analysis_id"))
    problems = []
    if expected_aid is not None and aid != expected_aid:
        problems.append(f"analysis_id {aid} != the composed spec's {expected_aid} (name, code_sha and inputs are the identity)")
    if code_sha is not None and row.get("code_sha") != code_sha:
        problems.append(f"code_sha {row.get('code_sha')} != this filing's {code_sha} (the receipt's tree)")
    if row.get("report_sha256") != report_sha:
        problems.append(f"report_sha256 {row.get('report_sha256')} != the verdict file's {report_sha}")
    if row.get("substrate") != "modal":
        problems.append(f"substrate {row.get('substrate')!r} != modal")
    try:
        jobs = json.loads(row.get("cluster_job_ids_json") or "[]")
    except ValueError:
        jobs = row.get("cluster_job_ids_json")
    if jobs != []:
        problems.append(f"cluster_job_ids_json {row.get('cluster_job_ids_json')!r} != []")
    got_objects = sorted(str(i.get("input_id")) for i in (row.get("inputs") or []) if i.get("input_kind") == "object")
    if got_objects != sorted(objects):
        problems.append(f"object inputs {got_objects} != {sorted(objects)}")
    run_inputs = [i for i in (row.get("inputs") or []) if i.get("input_kind") == "run"]
    if run_inputs:
        problems.append(f"run inputs {run_inputs} != []")
    arts = row.get("artifacts") or []
    if not any(a.get("role") == "report" and a.get("artifact_id") == report_sha for a in arts):
        problems.append(f"no report artifact {report_sha[:12]} linked ({arts})")
    rc, out, err = _cli(cli, ["--root", root, "query", "--json", f"SELECT analysis_id, extra_json FROM analysis WHERE name = '{name}'"])
    if rc != 0:
        problems.append(f"registry query failed ({rc}): {err.strip()[-300:]}")
    else:
        try:
            qrows = json.loads(out or "[]")
        except ValueError:
            qrows = []
        if len(qrows) != 1 or str(qrows[0].get("analysis_id")) != aid:
            problems.append(f"registry query returned {qrows!r} for the name")
        else:
            try:
                extra = json.loads(qrows[0].get("extra_json") or "{}")
            except ValueError:
                extra = {}
            if extra.get("receipt_sha256") != receipt_sha:
                problems.append(f"extra.receipt_sha256 {extra.get('receipt_sha256')} != {receipt_sha}")
            if extra.get("headline") != headline:
                problems.append(f"extra.headline {extra.get('headline')} != the receipt's {headline}")
            if mirror is not None:
                problems += _mirror_diffs(extra, mirror)
    if problems:
        for p in problems:
            log(f"FINDING: {name}: {p}")
        return "differs", aid
    identity = " (the composed spec's identity; code_sha this filing's)" if expected_aid is not None else ""
    facts = f", extra.mirror names the verified copy ({', '.join(MIRROR_FACTS)})" if mirror is not None else ""
    log(f"read-back: {name} -> analysis_id {aid}{identity}: report_sha256 == sha256(the verdict file), substrate modal, cluster_job_ids [], object inputs {sorted(objects)}, "
        f"extra.receipt_sha256 and extra.headline are the receipt's{facts}")
    return "folded", aid


def run(*, state: Path, view: str, key: str, registry_root: str, table: Optional[str], repo: Path, dry_run: bool, override_id: Optional[str] = None,
        cells: Optional[Sequence[str]] = None, launcher_cells: Optional[Sequence[str]] = None, cluster_label: Optional[str] = None, ssh_host: Optional[str] = None,
        cluster_dest: Optional[str] = None, cluster_root: Optional[str] = None, cluster_runner=None, log=print) -> int:
    """`cluster_runner` runs the cluster copy's re-check (stage07_mirror.ShRunner; the default runs ssh / the `local:` seam's bash for real)."""
    target = None
    if any(v is not None for v in (cluster_label, ssh_host, cluster_dest, cluster_root)):
        if not all(v for v in (cluster_label, ssh_host, cluster_dest, cluster_root)):
            raise FileRefusal("--cluster-label, --ssh-host, --cluster-root and --cluster-dest are passed together (the site's primary cluster, its expanded reproduce root "
                              "and its 07_mirror destination) or not at all")
    verdict_path = state / "gate" / FAMILY_PREFIX / f"{view}.json"
    receipt_path = state / STORE_DIR / RECEIPT_NAME
    if not verdict_path.is_file():
        raise FileRefusal(f"no verdict at {verdict_path} — rule it with stage.py verdict (the command `07 --phase receipt` printed; --range is yours) first")
    if not receipt_path.is_file():
        raise FileRefusal(f"no receipt at {receipt_path} — run --phase receipt first")
    receipt = read_json(receipt_path, "receipt")
    receipt_sha = sha256_file(receipt_path)
    verdict = read_json(verdict_path, "verdict")
    identity = run_identity(receipt, key=key, view=view, repo=repo, table=table)                  # the receipt's run, never this call's
    check_call_against_receipt(identity, cells=cells, launcher_cells=launcher_cells, override_id=override_id)
    verify_store(state, receipt)
    verify_verdict(verdict, receipt, receipt_sha, view)
    if cluster_label is not None:
        target = {"cluster_label": cluster_label, "ssh_host": ssh_host, "cluster_root": str(cluster_root).rstrip("/"), "cluster_path": f"{str(cluster_dest).rstrip('/')}/{receipt_sha[:12]}"}
    mirror = mirror_pointer(state, receipt_sha, key=key, view=view, cells=cells if cells is not None else identity["cells"], target=target,
                            runner=cluster_runner, dry_run=dry_run, log=log)                           # the durable copy, re-verified on the cluster, or refused by name
    objects = exercised_objects(identity, key=key, view=view, repo=repo, table=table, log=log)              # the receipt's map, never this checkout's plan
    spec = compose_spec(state=state, view=view, key=key, identity=identity, receipt=receipt, receipt_sha=receipt_sha, verdict=verdict, verdict_path=verdict_path,
                        objects=objects, mirror=mirror)
    spec_path = state / SPEC_NAME
    spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    report_sha = sha256_file(verdict_path)
    name = spec["name"]
    log(f"spec: {spec_path} — name {name}; code_sha {spec['code_sha']} ({spec['code_ref']}); cells {identity['cells']} -> {identity['launcher_cells']}; "
        f"inputs.objects {objects}; report {verdict_path} ({report_sha[:12]}); receipt {receipt_sha[:12]}; mirror {mirror['cluster_label']}:{mirror['cluster_path']} "
        f"(MIRROR.sha256 {mirror['mirror_sha256sum_digest'][:12]}, {mirror['files']} file(s), {mirror['bytes']} B)")
    root = os.path.expanduser(registry_root)
    if dry_run:
        log(f"DRY-RUN: registry analysis list --name {name} --json  (the fold) + the pending headers under {os.path.join(root, 'records', 'analyses')} — the pre-write check")
        log(f"DRY-RUN: registry analysis file --spec {spec_path} --dry-run --json  (root {root}); then the three-way read-back; nothing filed")
        return 0
    AN, cli, H, R = _registry(repo)
    expected_aid = AN.analysis_id(AN.normalize_spec(spec))          # the identity this filing would get: name, code_sha (the receipt's tree) and inputs
    rows = folded_rows(cli, root, name)
    pend = pending_headers(AN, H, R, root, name)
    filed = [("folded", r.get("analysis_id"), r.get("report_sha256"), r.get("filed_utc"), None, r) for r in rows] + \
            [("pending", p["analysis_id"], p["report_sha256"], p["filed_utc"], p["dir"], p) for p in pend]
    same = [f for f in filed if f[2] == report_sha]
    other = [f for f in filed if f[2] != report_sha]
    if other:
        f = other[0]
        ruled = ((f[5].get("extra") or {}) if f[0] == "pending" else {}).get("verdict", {}).get("ruled_utc")
        if f[0] == "folded":
            rc, out, _err = _cli(cli, ["--root", root, "query", "--json", f"SELECT extra_json FROM analysis WHERE analysis_id = '{f[1]}'"])
            try:
                ruled = (json.loads(json.loads(out)[0]["extra_json"]).get("verdict") or {}).get("ruled_utc") if rc == 0 else None
            except (ValueError, KeyError, IndexError, TypeError):
                ruled = None
        raise FileRefusal(f"{name} is already filed as analysis {f[1]} ({f[0]}{', ' + f[4] if f[4] else ''}; filed {f[3]}) with report_sha256 {f[2]} — this verdict "
                          f"file hashes to {report_sha}: a RE-RULED verdict of an already-filed execution (the filed verdict's ruled_utc {ruled}, this one's "
                          f"{verdict.get('ruled_utc')}); the registry holds ONE report per analysis and the filed ruling stands — record the re-ruling in decisions.md; "
                          f"a deliberate second ruling is filed by hand from {spec_path} under a name you choose and record")
    predates = False
    if same:
        f = same[0]
        log(f"already filed as {f[1]} ({f[0]}{', ' + f[4] if f[4] else ''}) — a duplicate; nothing written")
        if str(f[1]) != expected_aid:
            # round 3: a same-name, same-report filing at ANOTHER code_sha (or input set) is another analysis_id — not this filing; it gets no pointer
            # record (the predates branch binds a pointer to THIS analysis), and the read-back below names the fields
            log(f"FINDING: {name} is filed as analysis {f[1]} ({f[0]}), not this filing's identity {expected_aid} (name, code_sha and inputs; this filing's code_sha "
                f"{spec['code_sha']}, the receipt's tree) — another tree or input set under this name; no pointer record is written for it")
        elif "mirror" not in filed_extra(cli, root, f):
            # header: A FILING THAT PREDATES THE MIRROR PHASE — the row's extra cannot gain the pointer here; the pointer's record is this host's,
            # one per analysis, written once (a standing record that disagrees refuses above everything below; nothing is overwritten)
            predates = True
            ptr_p, written = write_pointer_record(state, name=name, analysis_id=f[1], filed_utc=f[3], where=f[0], mirror=mirror)
            log(f"FILED BEFORE THE MIRROR PHASE: {name} is analysis {f[1]} ({f[0]}{', ' + f[4] if f[4] else ''}; filed {f[3]}), whose extra carries no `mirror` — it was "
                "filed before `07 --phase mirror` existed, and the registry's analysis rows are facts the fold never rewrites (a re-filing of the same analysis is a "
                "duplicate), so this run attaches nothing to it; the verdict it holds stands")
            log(f"  the verified copy of receipt {receipt_sha[:12]}: {mirror['cluster_label']} {mirror['ssh_host']}:{mirror['cluster_path']} (MIRROR.sha256 "
                f"{mirror['mirror_sha256sum_digest']}, {mirror['files']} file(s), {mirror['bytes']} B, verified {mirror['verified_utc']}; local {mirror['local_path']})")
            log(f"  pointer record: {ptr_p} ({'written now' if written else 'stands from an earlier run, the same analysis and copy; not rewritten'}) — record it in "
                f"decisions.md against analysis {f[1]} (the registry row cannot carry it without a registry change; the read-back below ends in a finding, not a pass)")
    else:
        rc, out, err = _cli(cli, ["--root", root, "analysis", "file", "--spec", str(spec_path), "--json"])
        if rc != 0:
            raise FileRefusal(f"registry analysis file refused ({rc}): {(err or out).strip()[-600:]}")
        try:
            res = json.loads(out)
        except ValueError:
            raise FileRefusal(f"registry analysis file --json printed no JSON: {out[-400:]}") from None
        log(f"filed: analysis_id {res.get('analysis_id')} filing_dir {res.get('filing_dir')} (the harvester folds it on the grading host's next `registry ingest`)")
    # the current verified mirror the filed extra.mirror is held to (MIRROR_FACTS); a predates row is read back in every OTHER field here, and its
    # absent extra.mirror is then named as THE finding (predates_finding; round 4: the branch exited 0 — a pass that left the registry without the binding)
    mirror_facts = None if predates else mirror
    status, aid = read_back(cli, root, name, report_sha=report_sha, objects=objects, receipt_sha=receipt_sha, headline=receipt["headline"], log=log,
                            mirror=mirror_facts, expected_aid=expected_aid, code_sha=spec["code_sha"])
    if status == "folded":
        if predates:
            return predates_finding(name, aid, "folded", mirror=mirror, pointer=pointer_record_path(state, name), log=log)
        log(f"PASS: {name} analysis_id {aid}")
        return 0
    if status == "absent":
        # no row: the pending header is read back the way a row is (check_pending) before the filing is called complete — a header carrying another
        # copy's digest, another receipt or another identity is a finding here, not after the fold
        pend = pending_headers(AN, H, R, root, name)
        if not pend:
            log(f"FINDING: {name} is neither folded nor pending under {root}")
            return EXIT_FINDING
        problems = [(h, p) for h in pend for p in check_pending(h, expected_aid=expected_aid, report_sha=report_sha, objects=objects, receipt_sha=receipt_sha,
                                                                headline=receipt["headline"], mirror=mirror_facts, code_sha=spec["code_sha"])]
        if problems:
            for h, p in problems:
                log(f"FINDING: {name}: pending header {h['dir']} (analysis {h['analysis_id']}, filed {h['filed_utc']}): {p}")
            log(f"FINDING: the pending filing of {name} is not this filing's — the fold would keep it as filed; nothing here rewrites a record: resolve it by "
                "hand (the registry's owners' procedure for a mis-filed pending record) and record the call in decisions.md")
            return EXIT_FINDING
        log(f"read-back (pending): {name} -> analysis_id {pend[0]['analysis_id']} in {len(pend)} pending header(s): analysis_id == the composed spec's, code_sha this "
            f"filing's, report_sha256 == sha256(the verdict file), substrate modal, cluster_job_ids [], object inputs {sorted(objects)}, extra.receipt_sha256 and "
            f"extra.headline are the receipt's{', extra.mirror names the verified copy (' + ', '.join(MIRROR_FACTS) + ')' if mirror_facts is not None else ''}")
        if predates:
            return predates_finding(name, pend[0]["analysis_id"], f"pending, {pend[0]['dir']}", mirror=mirror, pointer=pointer_record_path(state, name), log=log)
        log(f"filed, not yet folded: {name} analysis_id {pend[0]['analysis_id']} (pending header {pend[0]['dir']}) — the grading host's next `registry ingest` "
            f"folds it; re-run `07 --phase file` to read the row back")
        return 0
    return EXIT_FINDING


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-root", type=Path, required=True, help="the VIEW's state directory (07: the model's state dir for the baseline wave, <state>/07_views/<view> for a "
                                                                      "dose view): the receipt, the verdict, the mirror, the spec and the pointer records are all read and written under it")
    ap.add_argument("--view", required=True)
    ap.add_argument("--model", required=True, help="the models-table key (the launcher slug)")
    ap.add_argument("--registry-root", required=True)
    ap.add_argument("--override-id", default=None, help="a cross-check: OVERRIDE_ID of this call's run-overrides document; must be the receipt's run_overrides_id")
    ap.add_argument("--cells", default=None, help="a cross-check: this call's cells by the table's names, comma list; must be the receipt's view_cells")
    ap.add_argument("--launcher-cells", default=None, help="a cross-check: this call's launcher tokens, comma list; must be the receipt's cells")
    ap.add_argument("--table", default=None)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--cluster-root", default=None, help="a cross-check (with the three below): the site's expanded SISTEER_REPRODUCE_ROOT on the cluster, the root the mirror "
                                                         "guarded the copy under; the record's cluster_root must be it")
    ap.add_argument("--cluster-label", default=None, help="a cross-check (with --ssh-host, --cluster-root and --cluster-dest): the site's primary cluster; the mirror's cluster "
                                                          "record must name it")
    ap.add_argument("--ssh-host", default=None, help="a cross-check: CLUSTER_SSH of that cluster, as the mirror recorded it")
    ap.add_argument("--cluster-dest", default=None, help="a cross-check: the site's <reproduce root>/<model>/07_mirror (<reproduce root>/<model>/07_views/<view>/07_mirror for a dose "
                                                         "view); the record's cluster_path must be <this>/<receipt sha12>")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    log = lambda s: print(f"[stage07-file] {s}", flush=True)  # noqa: E731
    split = lambda s: None if s is None else [c for c in s.split(",") if c]  # noqa: E731
    try:
        return run(state=Path(a.state_root), view=a.view, key=a.model, registry_root=a.registry_root, override_id=a.override_id, cells=split(a.cells),
                   launcher_cells=split(a.launcher_cells), table=a.table, repo=Path(a.repo).resolve(), dry_run=a.dry_run, cluster_label=a.cluster_label,
                   ssh_host=a.ssh_host, cluster_dest=a.cluster_dest, cluster_root=a.cluster_root, log=log)
    except FileRefusal as e:
        log(f"REFUSED: {e}")
        log("nothing filed")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
