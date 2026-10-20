"""Filing an ANALYSIS into the registry (2026-10-08): a prefill-only or offline computation -- a routing capture, a clamp ablation,
an efficiency study -- with results but no run-shaped home (no eval, no generation). ``registry analysis file --spec <json>`` reads a
spec, checks its shape, READ-checks that every input run and object is in the database the root names (a missing one refuses by name;
nothing is written), registers the report file and the page directories' files as ``artifact`` records through
:func:`publish.register_files` (kind by suffix; a remote report under its declared sha), and writes the three ledger-fact records --
``analysis``, ``analysis_input``, ``analysis_artifact`` -- under ``records/analyses/<day>/`` through one :class:`emit.RecordWriter` with the
filer's producer block and the standard provenance block, in that order and with strictly increasing stamps, so every FK parent folds
ahead of the row that names it by path order and by stamp order alike. The database is never written here: the harvester folds the
records on its next cycle (INSERT OR IGNORE, no ``row_history``: the rows ARE the facts). ``registry analysis list`` reads them back.

Every artifact record a filing writes names the ONE location the filing sees: the fold unions `artifact.locations_json` on EVERY record
of the table, from every producer (``reducer.UNION_COLUMNS``), adding the location to the standing list by uri instead of replacing the
list, so two filings of one digest in one window -- from two cluster paths, or a remote declaration beside a local read -- fold to every
location whatever their order, on live ingest and on rebuild alike, and an adapter revision re-registering the same bytes later keeps the
filed location too (the fourth review round, 2026-10-09: the third round's record restated the standing locations plus the filing's, read
from the database at filing time, so the second of two filings folded before an ingest replaced the first's path; the fourth closing pass:
the union was a per-record directive the publish writers alone carried, and an adapter's record without it erased the filed location, so
the filed copy refused to publish as unregistered). A digest the registry ALREADY holds is linked, not
re-registered (:func:`merge_known_artifacts`): its record, when one is written at all, is :func:`publish.merged_artifact_payload` -- the
standing row's kind and primary, this filing's location, the verification columns stated only by a local capture -- and a filing whose
location adds nothing to the standing list writes no artifact record (the link names the standing row). The same bytes under several
entries of one filing are one link whose ``extra.same_bytes`` lists the further entries, while every further distinct path writes its own
artifact record (:func:`_artifact_plans`; :data:`SAME_BYTES_KEY`).

The spec (JSON)::

    {"name": "moe_routing.capture",                       # dotted by study; free text
     "code_sha": "<40 hex>|unverified", "code_ref": "moe-routing @ 63cecaa7e (...)",
     "cluster_job_ids": ["949499", ...], "substrate": "cluster_a",
     "report": {"uri": "/path/report.json", "sha256": "<64 hex>|null, "substrate": "cluster_a"} | null,
     "inputs": {"runs": ["<run_id>", ...], "objects": ["<object_id>", ...]},
     "artifacts": [{"dir": "/path/figs/<page>/", "role": "page", "url": "https://..."},
                   {"path": "/path/file.json", "role": "report", "sha256": "<64 hex>|null"}],
     "filed_by": "<filer>", "provenance": {"kind": "relayed", "relayed_by": "<session>", "message_utc": "..."},
     "notes": "...", "extra": {...}}

Identity: ``analysis_id`` = uuid5 over canonical JSON of ``{name, code_sha, inputs}`` (:func:`analysis_id`), so a re-filing of the same
analysis is a duplicate at the fold and a changed input set or code sha is a new analysis; notes, artifacts and provenance are not identity.
The normalized spec's input lists are SORTED (:func:`normalize_spec`), so ``extra.spec_sha256`` -- the hash of the normalized spec the
header records -- is one hash for one input set whatever order a spec lists the ids in, and a re-filing with two ids swapped is a duplicate
of the filing, never a conflicting re-filing (the sixth closing pass, 2026-10-09: the id sorted the lists and the hash did not, so the
header was refused by name with every input and link bound to it). The artifacts (the first entry of a digest carries the link) and the
cluster job ids keep the spec's order: it has meaning.

A filing is PUBLISHED WHOLE or not at all (the seventh closing pass, 2026-10-09). :func:`write_filing` writes the four files under
``staging/analyses/<token>/`` -- the registry root's staging namespace, which the harvester never scans (the adapters' revisions are
assembled the same way, ``adapters.runner.staging_dir``) -- and, once every file is written and synced, renames the directory into
``records/analyses/<day>/<filing>/`` in ONE rename (:func:`publish_staged`; ``<filing>`` is the first twelve hex of the header's record
sha, the prefix the fold's refusals name a filing by). Before this the files were created one by one in the scanned tree, so a process
killed -- or a write that failed -- after the header and before the inputs or the links left a header the next cycle accepted as a
complete analysis with no inputs or no links, its ``report_uri`` and spec describing evidence the log did not hold. A filing interrupted
before its publish leaves its staging directory behind, holding nothing the record log knows: :func:`abandoned_stagings` lists such
directories and ``registry analysis file`` names them on every later run (a dry run included), so an interrupted filing is visible by
name; a write that fails inside the staging refuses by name (:class:`AnalysisRefusal`), the staged files left for inspection. Nothing is
recovered from a staging directory: the filing is re-run from its spec (a duplicate at the fold if the first one did publish), and the
directory is removed by hand.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import emit
from . import harvest as H
from . import identity
from . import publish as PUB
from . import records as R
from . import vocab as V
from .identity import canonical_json

#: ``records/<SUBDIR>/<day>/<filing>/`` -- where a filing's records go (beside ``records/publish/``, the other hand-driven artifact
#: records): one directory per filing, named by the first :data:`FILING_DIR_HEX` hex digits of the header's record sha, published whole by
#: one rename from ``<root>/<STAGING_DIR>/<SUBDIR>/<token>/`` (:func:`write_filing`, :func:`publish_staged`).
SUBDIR = "analyses"
STAGING_DIR = "staging"
FILING_DIR_HEX = 12
PRODUCER_SCRIPT = "sisteer.registry.cli analysis file"
DECISION = "D357"
DEFAULT_FILED_BY = "the user"
PROVENANCE_KINDS: Tuple[str, ...] = ("user_direct", "relayed")
#: The spec keys that make the analysis's IDENTITY (the uuid5 input); everything else may differ between two filings of one analysis.
IDENTITY_KEYS: Tuple[str, ...] = ("name", "code_sha", "inputs")
_SHA256_HEX = 64
_SHA1_HEX = 40


class AnalysisRefusal(ValueError):
    """A refusal by name at the CLI: nothing is written. ``field`` names what refused (a spec path such as ``inputs.runs[2]``), ``reason``
    says why."""

    def __init__(self, field_name: str, reason: str):
        super().__init__(f"{field_name}: {reason}")
        self.field = field_name
        self.reason = reason


def _is_hex(value: object, n: int) -> bool:
    return isinstance(value, str) and len(value) == n and all(c in "0123456789abcdef" for c in value)


def _str_list(value: object, where: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisRefusal(where, f"must be a list of strings, got {type(value).__name__}")
    out: List[str] = []
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            raise AnalysisRefusal(f"{where}[{i}]", f"must be a string, got {v!r}")
        text = str(v).strip()
        if not text:
            raise AnalysisRefusal(f"{where}[{i}]", "is empty")
        if text in out:
            raise AnalysisRefusal(f"{where}[{i}]", f"{text!r} is listed twice")
        out.append(text)
    return out


# --------------------------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------------------------


def load_spec(path: str) -> Dict[str, object]:
    """The spec file as a dict; a file that is not a JSON object refuses by name."""
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise AnalysisRefusal("spec", f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisRefusal("spec", f"{path} is not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(raw, dict):
        raise AnalysisRefusal("spec", f"{path} must hold a JSON object")
    return raw


#: Every key a spec may carry at its top level; an unknown one is refused by name (a misspelled `cluster_job_ids` must not pass silently).
SPEC_KEYS: Tuple[str, ...] = ("name", "code_sha", "code_ref", "cluster_job_ids", "substrate", "report", "inputs", "artifacts", "filed_by",
                              "provenance", "notes", "extra")


def normalize_spec(spec: Mapping[str, object]) -> Dict[str, object]:
    """The spec's shape, checked field by field (the first problem refuses by its spec path); returns the normalized spec."""
    unknown = sorted(k for k in spec if k not in SPEC_KEYS)
    if unknown:
        raise AnalysisRefusal(unknown[0], f"unknown spec key; the keys are {', '.join(SPEC_KEYS)} (free-form facts go under `extra`)")
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AnalysisRefusal("name", "a non-empty string is required")
    code_sha = spec.get("code_sha")
    if not (_is_hex(code_sha, _SHA1_HEX) or code_sha == "unverified"):
        raise AnalysisRefusal("code_sha", f"forty hex digits or 'unverified' required, got {code_sha!r} (a short sha resolves with `git rev-parse`)")
    code_ref = spec.get("code_ref")
    if code_ref is not None and not isinstance(code_ref, str):
        raise AnalysisRefusal("code_ref", "must be a string or null")
    jobs = _str_list(spec.get("cluster_job_ids"), "cluster_job_ids")
    substrate = spec.get("substrate")
    if substrate not in V.SUBSTRATES:
        raise AnalysisRefusal("substrate", f"must be one of {'|'.join(V.SUBSTRATES)}, got {substrate!r}")
    report = spec.get("report")
    if report is not None:
        if not isinstance(report, dict) or not isinstance(report.get("uri"), str) or not str(report["uri"]).strip():
            raise AnalysisRefusal("report.uri", "a report is {uri, sha256|null, substrate?} with a non-empty uri, or null")
        sha = report.get("sha256")
        if sha is not None and not _is_hex(sha, _SHA256_HEX):
            raise AnalysisRefusal("report.sha256", f"64 hex digits or null required, got {sha!r}")
        rsub = report.get("substrate")
        if rsub is not None and rsub not in V.SUBSTRATES:
            raise AnalysisRefusal("report.substrate", f"must be one of {'|'.join(V.SUBSTRATES)} or null, got {rsub!r}")
        report = {"uri": str(report["uri"]).strip(), "sha256": sha, "substrate": rsub}
    inputs = spec.get("inputs")
    if not isinstance(inputs, dict):
        raise AnalysisRefusal("inputs", "an object {runs: [...], objects: [...]} is required")
    unknown = sorted(set(inputs) - {"runs", "objects"})
    if unknown:
        raise AnalysisRefusal(f"inputs.{unknown[0]}", f"unknown input kind; the kinds are {'|'.join(V.ANALYSIS_INPUT_KINDS)} (`runs`, `objects`)")
    # the input lists are canonical in the normalized spec -- checked in the spec's order (a refusal names `inputs.runs[i]` as written),
    # then sorted -- so one input set is one `spec_sha256` and one `analysis_id` whatever order the spec lists the ids in, and the inputs
    # are emitted and listed in that order too (the sixth closing pass, 2026-10-09). The artifacts and the job ids keep the spec's order.
    runs = sorted(_str_list(inputs.get("runs"), "inputs.runs"))
    objects = sorted(_str_list(inputs.get("objects"), "inputs.objects"))
    if not runs and not objects:
        raise AnalysisRefusal("inputs", "an analysis names at least one input run or object")
    arts_raw = spec.get("artifacts")
    if arts_raw is None:
        arts_raw = []
    if not isinstance(arts_raw, list):
        raise AnalysisRefusal("artifacts", "must be a list")
    artifacts: List[Dict[str, object]] = []
    for i, a in enumerate(arts_raw):
        where = f"artifacts[{i}]"
        if not isinstance(a, dict) or (("dir" in a) == ("path" in a)):
            raise AnalysisRefusal(where, "an artifact entry names exactly one of `dir` (a published page directory) or `path` (one file)")
        role = a.get("role")
        if role is None:
            role = "page" if "dir" in a else PUB.kind_of(str(a["path"]))
            role = role if role in V.ANALYSIS_ARTIFACT_ROLES else "report"
        if role not in V.ANALYSIS_ARTIFACT_ROLES:
            raise AnalysisRefusal(f"{where}.role", f"must be one of {'|'.join(V.ANALYSIS_ARTIFACT_ROLES)}, got {role!r}")
        url = a.get("url")
        if url is not None and not isinstance(url, str):
            raise AnalysisRefusal(f"{where}.url", "must be a string or null")
        sha = a.get("sha256")
        if sha is not None and not _is_hex(sha, _SHA256_HEX):
            raise AnalysisRefusal(f"{where}.sha256", f"64 hex digits or null required, got {sha!r}")
        asub = a.get("substrate")
        if asub is not None and asub not in V.SUBSTRATES:
            raise AnalysisRefusal(f"{where}.substrate", f"must be one of {'|'.join(V.SUBSTRATES)} or null, got {asub!r}")
        key = "dir" if "dir" in a else "path"
        if not isinstance(a[key], str) or not a[key].strip():
            raise AnalysisRefusal(f"{where}.{key}", "a non-empty path is required")
        artifacts.append({key: a[key].strip(), "role": role, "url": url, "sha256": sha, "substrate": asub})
    filed_by = spec.get("filed_by", DEFAULT_FILED_BY)
    if not isinstance(filed_by, str) or not filed_by.strip():
        raise AnalysisRefusal("filed_by", "a non-empty string is required (default: the user)")
    prov = spec.get("provenance") or {"kind": "user_direct"}
    if not isinstance(prov, dict) or prov.get("kind") not in PROVENANCE_KINDS:
        raise AnalysisRefusal("provenance.kind", f"must be one of {'|'.join(PROVENANCE_KINDS)} (D83)")
    if prov["kind"] == "relayed" and not (isinstance(prov.get("relayed_by"), str) and prov["relayed_by"].strip()
                                          and isinstance(prov.get("message_utc"), str) and prov["message_utc"].strip()):
        raise AnalysisRefusal("provenance", "a relayed filing names the relaying session (`relayed_by`) and the message's time (`message_utc`), D83")
    notes = spec.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise AnalysisRefusal("notes", "must be a string or null")
    extra = spec.get("extra") or {}
    if not isinstance(extra, dict):
        raise AnalysisRefusal("extra", "must be an object")
    for k in ("provenance", "decision", "spec_sha256"):
        if k in extra:
            raise AnalysisRefusal(f"extra.{k}", "is the filing's own key")
    return {"name": name.strip(), "code_sha": code_sha, "code_ref": code_ref, "cluster_job_ids": jobs, "substrate": substrate,
            "report": report, "inputs": {"runs": runs, "objects": objects}, "artifacts": artifacts, "filed_by": filed_by.strip(),
            "provenance": {k: v for k, v in prov.items() if v is not None}, "notes": notes, "extra": extra}


def analysis_id(spec: Mapping[str, object]) -> str:
    """uuid5 over the analysis's IDENTITY: canonical JSON of its name, code sha and inputs (each list sorted here as well, so a raw
    spec's id is its normalized spec's). Notes, artifacts, job ids and provenance are not identity: a corrected note re-files the same
    analysis (a duplicate at the fold)."""
    inputs = spec["inputs"]
    ident = {"name": spec["name"], "code_sha": spec["code_sha"],
             "inputs": {"runs": sorted(inputs["runs"]), "objects": sorted(inputs["objects"])}}      # type: ignore[index]
    return str(uuid.uuid5(identity.RECORD_NAMESPACE, "analysis|" + canonical_json(ident)))


def spec_sha256(spec: Mapping[str, object]) -> str:
    """sha256 of the NORMALIZED spec's canonical JSON (:func:`normalize_spec`, its input lists sorted): what the analysis row records as
    the filing's input in ``extra.spec_sha256``. Two specs that differ only in the order of their input ids are one hash, as they are one
    ``analysis_id``; a spec that differs in anything else -- a note, a job id, an artifact -- is another hash, and the header's ordering
    rule names the difference when the rest of the row agrees."""
    return identity.sha256_hex(canonical_json(dict(spec)).encode("utf-8"))


# --------------------------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------------------------


@dataclass
class ArtifactPlan:
    uri: str                              # absolute local path, or the remote uri as declared
    kind: str                             # artifact.kind (by suffix)
    role: str                             # analysis_artifact.role
    sha256: str
    remote: bool
    payload: Optional[Dict[str, object]]  # the `artifact` record payload; None when the digest is known and this filing adds nothing to its row
    link_extra: Dict[str, object] = field(default_factory=dict)
    size: Optional[int] = None            # a local file's size and mtime at CAPTURE (the one read that hashed it); None for a remote uri
    mtime_ns: Optional[int] = None
    known: bool = False                   # the registry already holds an `artifact` row for this digest: linked, its row merged, never replaced
    same_bytes_as: Optional[str] = None   # the uri of the filing's FIRST entry of this digest, which carries the one link; None when this plan does


#: ``analysis_artifact.extra.same_bytes``: every further entry of ONE filing whose bytes are the link's digest -- ``{uri, role, page_dir?,
#: relpath?, page_url?, declared_sha256?}`` each, in the spec's order. The link's key is (analysis_id, artifact_id), so a filing has one link
#: per digest: its `role` and page keys are the first entry's, and the others are listed here instead of dropped (the closing pass,
#: 2026-10-09: two page directories with identical `index.html` bytes filed as one page -- the second's path, url and role nowhere). The
#: validator refuses a NEW link carrying the key in any other shape by name (``validate._check_same_bytes``, which spells the key too;
#: pinned equal by the tests), and the readers below skip an entry that is not an object (the fourth closing pass, 2026-10-09).
SAME_BYTES_KEY = "same_bytes"


def _capture(path: str, where: str) -> Tuple[str, int, int]:
    """ONE read of a local file: ``(sha256, size, mtime_ns)``. The file is stat'ed before and after the read; a size or mtime that moved
    around it is refused by name (nothing captured) -- the digest would name bytes the file no longer holds. The artifact record and the
    link to it both take THIS digest (2026-10-09: two reads could name different bytes)."""
    st = os.stat(path)
    sha = PUB.sha256_of(path)
    st2 = os.stat(path)
    if (st.st_size, st.st_mtime_ns) != (st2.st_size, st2.st_mtime_ns):
        raise AnalysisRefusal(where, f"{path} changed while being hashed (size {st.st_size} -> {st2.st_size}, mtime moved): nothing captured; "
                                     "re-run when the file is still")
    return sha, st.st_size, st.st_mtime_ns


def verify_captured(plans: Sequence[ArtifactPlan]) -> None:
    """Refuse by name when a local artifact's size or mtime moved since its capture: called right before the write, so the records never
    name bytes a file no longer holds (nothing is written)."""
    for p in plans:
        if p.remote or p.size is None:
            continue
        try:
            st = os.stat(p.uri)
        except OSError as exc:
            raise AnalysisRefusal("artifacts", f"{p.uri} changed between the plan and the write: {exc}; nothing written") from exc
        if (st.st_size, st.st_mtime_ns) != (p.size, p.mtime_ns):
            raise AnalysisRefusal("artifacts", f"{p.uri} changed between the plan and the write (size {p.size} -> {st.st_size}, mtime "
                                               f"{p.mtime_ns} -> {st.st_mtime_ns}): nothing written; re-run the filing")


@dataclass
class FilingPlan:
    analysis_id: str
    spec: Dict[str, object]
    inputs: List[Tuple[str, str]]         # (input_kind, input_id), runs first, in the spec's order
    artifacts: List[ArtifactPlan]
    analysis_payload: Dict[str, object]
    input_payloads: List[Dict[str, object]]
    link_payloads: List[Dict[str, object]]
    missing_inputs: List[Tuple[str, str]]  # (input_kind, input_id) the database does not hold -- a refusal
    day: str


def missing_inputs(conn: sqlite3.Connection, inputs: Mapping[str, Sequence[str]]) -> List[Tuple[str, str]]:
    """The input runs and objects the database does NOT hold, in the spec's order."""
    out: List[Tuple[str, str]] = []
    for rid in inputs["runs"]:
        if conn.execute("SELECT 1 FROM run WHERE run_id = ? LIMIT 1", (rid,)).fetchone() is None:
            out.append(("run", rid))
    for oid in inputs["objects"]:
        if conn.execute("SELECT 1 FROM object WHERE object_id = ? LIMIT 1", (oid,)).fetchone() is None:
            out.append(("object", oid))
    return out


def _local_file(path: str) -> Optional[str]:
    p = os.path.abspath(os.path.expanduser(path))
    return p if os.path.isfile(p) else None


_KNOWN_ARTIFACT_COLUMNS: Tuple[str, ...] = ("kind", "uri_primary", "locations_json", "bytes", "created_utc", "verified_utc", "extra_json")


def merge_known_artifacts(conn: sqlite3.Connection, plans: Sequence[ArtifactPlan]) -> int:
    """Mark every plan whose digest the database already holds as ``known`` and replace its payload with
    :func:`publish.merged_artifact_payload` over the standing row -- or None when the filing's location adds nothing to the standing list
    as it stands. A full `artifact` record overwrites the scalar columns it states (the third review round, 2026-10-09: a remote report
    declaring a known digest nulled the standing row's verification), so the known record keeps the standing kind and primary and states
    the verification only from a local read; the locations are the fold's union (on every artifact record, `reducer.UNION_COLUMNS`), so a
    record spooled but not yet folded when this reads the database costs nothing -- its location and this one both fold in. Returns how
    many plans are known. Reads the database only."""
    n = 0
    for p in plans:
        row = conn.execute(f"SELECT {', '.join(_KNOWN_ARTIFACT_COLUMNS)} FROM artifact WHERE artifact_id = ?", (p.sha256,)).fetchone()
        if row is None:
            continue
        existing = dict(zip(_KNOWN_ARTIFACT_COLUMNS, tuple(row)))
        assert p.payload is not None
        p.payload = PUB.merged_artifact_payload(existing, p.payload)
        p.known = True
        n += 1
    return n


def _artifact_plans(spec: Mapping[str, object], now: str) -> List[ArtifactPlan]:
    """The artifact records a filing registers: the report (local → hashed; remote → its declared sha, or a refusal by name), then
    each entry of ``artifacts`` -- a page directory's every publishable file (publish.page_files) or one file.

    The same bytes under several entries -- two page directories with an identical `index.html`, a figure both pages serve, a report also
    inside its page directory -- are ONE link ((analysis_id, artifact_id) is the link's key) carried by the FIRST entry of the digest, whose
    `role` and page keys the link states; every further entry is listed in the link's ``extra.same_bytes`` (:data:`SAME_BYTES_KEY`), and
    every further DISTINCT path still writes its own artifact record: its location is a fact the fold's union keeps, with the first path as
    the row's primary and the first entry's kind, so the row is the same whatever order the fold meets the records in. A further entry of
    the same path (the report inside its page directory) adds no record: its location is stated already. The closing pass, 2026-10-09:
    before this a digest seen once was dropped silently on its next path, so the second page's files were never registered and its url
    and role reached no link."""
    plans: List[ArtifactPlan] = []
    first: Dict[str, ArtifactPlan] = {}                # by digest: the plan that carries the link
    name = str(spec["name"])

    def add(p: ArtifactPlan) -> None:
        carrier = first.get(p.sha256)
        if carrier is None:
            first[p.sha256] = p
            plans.append(p)
            return
        entry: Dict[str, object] = {"uri": p.uri, "role": p.role, **p.link_extra}
        own = {"uri": carrier.uri, "role": carrier.role, **{k: v for k, v in carrier.link_extra.items() if k != SAME_BYTES_KEY}}
        listed: List[object] = list(carrier.link_extra.get(SAME_BYTES_KEY) or [])
        if entry == own or entry in listed:
            return                                   # the very same entry again (a directory listed twice): nothing new to say
        carrier.link_extra[SAME_BYTES_KEY] = listed + [entry]
        if any(q.uri == p.uri for q in plans if q.sha256 == p.sha256):
            return                                   # the same FILE under another role or page: its location is stated by its own plan
        assert p.payload is not None
        kept = ("analysis", "declared_sha256")                        # the row's page keys are the carrier's: this record states none
        p.payload = {**p.payload, "kind": carrier.kind, "uri_primary": carrier.uri,
                     "extra": {k: v for k, v in dict(p.payload.get("extra") or {}).items() if k in kept}}
        p.kind = carrier.kind
        p.same_bytes_as = carrier.uri
        plans.append(p)

    report = spec.get("report")
    if report:
        uri = str(report["uri"])                                                        # type: ignore[index]
        local = _local_file(uri)
        declared = report.get("sha256")                                                 # type: ignore[union-attr]
        if local is not None:
            sha, size, mtime = _capture(local, "report.uri")
            if declared is not None and declared != sha:
                raise AnalysisRefusal("report.sha256", f"the declared sha {str(declared)[:12]}… differs from the file's {sha[:12]}… ({local})")
            payload = PUB.artifact_payload(local, None, now, kind="report", sha=sha, size=size)
            payload["extra"] = {**payload["extra"], "analysis": name, "role": "report"}
            add(ArtifactPlan(local, "report", "report", sha, False, payload, size=size, mtime_ns=mtime))
        elif declared is not None:
            payload = PUB.remote_artifact_payload(uri, str(declared), now, kind="report", substrate=report.get("substrate") or spec.get("substrate"),  # type: ignore[union-attr]
                                                  extra={"analysis": name, "role": "report"})
            add(ArtifactPlan(uri, "report", "report", str(declared), True, payload, {"declared_sha256": str(declared)}))
        else:
            raise AnalysisRefusal("report.sha256", f"{uri} is not a readable local file and the spec declares no sha256: a remote report is "
                                                   "filed under its declared sha (hash it where it lives), never a guess")
    for i, a in enumerate(spec.get("artifacts") or []):                                 # type: ignore[union-attr]
        where = f"artifacts[{i}]"
        role = str(a["role"])
        if "dir" in a:
            d = os.path.abspath(os.path.expanduser(str(a["dir"])))
            if not os.path.isdir(d):
                raise AnalysisRefusal(f"{where}.dir", f"{d} is not a directory on this machine (a page directory is registered file by file)")
            files = PUB.page_files(d)
            if not files:
                raise AnalysisRefusal(f"{where}.dir", f"{d} holds no publishable file ({', '.join(PUB.FIGURE_SUFFIXES)})")
            for f in files:
                sha, size, mtime = _capture(f, f"{where}.dir")
                payload = PUB.artifact_payload(f, None, now, sha=sha, size=size)
                ex: Dict[str, object] = {"analysis": name, "page_dir": d, "relpath": os.path.relpath(f, d)}
                if a.get("url"):
                    ex["page_url"] = a["url"]
                payload["extra"] = {**payload["extra"], **ex}
                link_extra = {k: v for k, v in ex.items() if k != "analysis"}
                add(ArtifactPlan(f, str(payload["kind"]), role, sha, False, payload, link_extra, size=size, mtime_ns=mtime))
            continue
        uri = str(a["path"])
        local = _local_file(uri)
        declared = a.get("sha256")
        if local is not None:
            sha, size, mtime = _capture(local, f"{where}.path")
            if declared is not None and declared != sha:
                raise AnalysisRefusal(f"{where}.sha256", f"the declared sha {str(declared)[:12]}… differs from the file's {sha[:12]}… ({local})")
            payload = PUB.artifact_payload(local, None, now, sha=sha, size=size)
            ex = {"analysis": name}
            if a.get("url"):
                ex["page_url"] = a["url"]
            payload["extra"] = {**payload["extra"], **ex}
            add(ArtifactPlan(local, str(payload["kind"]), role, sha, False, payload, {k: v for k, v in ex.items() if k != "analysis"}, size=size, mtime_ns=mtime))
        elif declared is not None:
            payload = PUB.remote_artifact_payload(uri, str(declared), now, substrate=a.get("substrate") or spec.get("substrate"), extra={"analysis": name})
            add(ArtifactPlan(uri, str(payload["kind"]), role, str(declared), True, payload, {"declared_sha256": str(declared)}))
        else:
            raise AnalysisRefusal(f"{where}.sha256", f"{uri} is not a readable local file and the entry declares no sha256")
    return plans


def build_plan(spec_raw: Mapping[str, object], now: str, conn: Optional[sqlite3.Connection], producer_session: Optional[str] = None) -> FilingPlan:
    """Everything a filing would write, built WITHOUT writing: the normalized spec, the analysis id, the artifact records (local files
    hashed now), the three kinds' payloads, and -- when ``conn`` is given -- the inputs the database does not hold."""
    spec = normalize_spec(spec_raw)
    aid = analysis_id(spec)
    inputs = [("run", r) for r in spec["inputs"]["runs"]] + [("object", o) for o in spec["inputs"]["objects"]]   # type: ignore[index]
    arts = _artifact_plans(spec, now)
    if conn is not None:
        merge_known_artifacts(conn, arts)              # a digest the registry holds is linked, its row merged, never restated from here
    missing = missing_inputs(conn, spec["inputs"]) if conn is not None else []           # type: ignore[arg-type]
    report = spec.get("report")
    report_sha = None
    if report:
        report_sha = next(p.sha256 for p in arts if p.role == "report" and p.uri in (str(report["uri"]), _local_file(str(report["uri"])) or ""))  # type: ignore[index]
    extra: Dict[str, object] = dict(spec["extra"])                                       # type: ignore[arg-type]
    extra.update({"provenance": dict(spec["provenance"]), "decision": DECISION, "spec_sha256": spec_sha256(spec)})   # type: ignore[arg-type]
    analysis_payload: Dict[str, object] = {
        "analysis_id": aid, "name": spec["name"], "code_sha": spec["code_sha"], "code_ref": spec["code_ref"],
        "cluster_job_ids_json": list(spec["cluster_job_ids"]), "substrate": spec["substrate"],                   # type: ignore[arg-type]
        "report_uri": (_local_file(str(report["uri"])) or str(report["uri"])) if report else None,               # type: ignore[index]
        "report_sha256": report_sha, "filed_utc": now, "filed_by": spec["filed_by"], "notes": spec["notes"], "extra": extra}
    input_payloads = [{"analysis_id": aid, "input_kind": k, "input_id": i, "run_id": i if k == "run" else None,
                       "object_id": i if k == "object" else None} for k, i in inputs]
    link_payloads = [{"analysis_id": aid, "artifact_id": p.sha256, "role": p.role, "extra": dict(p.link_extra)} for p in arts if p.same_bytes_as is None]
    return FilingPlan(aid, spec, inputs, arts, analysis_payload, input_payloads, link_payloads, missing, now[:10])


def plan_lines(plan: FilingPlan) -> List[str]:
    spec = plan.spec
    out = [f"analysis {spec['name']} -> analysis_id {plan.analysis_id}",
           f"  code_sha {spec['code_sha']}" + (f" ({spec['code_ref']})" if spec.get("code_ref") else ""),
           f"  substrate {spec['substrate']}; cluster job ids {', '.join(spec['cluster_job_ids']) or '(none)'}",           # type: ignore[arg-type]
           f"  provenance {canonical_json(spec['provenance'])}; filed_by {spec['filed_by']}",
           f"  inputs: {len(spec['inputs']['runs'])} run(s), {len(spec['inputs']['objects'])} object(s)"]                   # type: ignore[index]
    for k, i in plan.inputs:
        mark = "  MISSING" if (k, i) in plan.missing_inputs else ""
        out.append(f"    {k} {i}{mark}")
    if spec.get("report"):
        out.append(f"  report {plan.analysis_payload['report_uri']} sha256 {str(plan.analysis_payload['report_sha256'])[:12]}…")
    known = [p for p in plan.artifacts if p.known]
    n_records = sum(1 for p in plan.artifacts if p.payload is not None)
    out.append(f"  artifacts: {len(plan.artifacts)} file(s) to register"
               + (f" ({len(known)} already registered: linked, the standing row merged, never replaced)" if known else ""))
    for p in plan.artifacts:
        out.append(f"    {p.role:<6} {p.kind:<6} {p.sha256[:12]}… {p.uri}" + (" (remote: declared sha, unverified)" if p.remote else "")
                   + _known_mark(p) + (f" [same bytes as {p.same_bytes_as}: one link, this path a location of the row]" if p.same_bytes_as else ""))
    out.append(f"  records: {1 if n_records else 0} artifact file ({n_records} record(s)"
               + (f"; {len(known)} already registered" if known else "") + f"), 1 analysis, "
               f"{len(plan.input_payloads)} analysis_input, {len(plan.link_payloads)} analysis_artifact -> records/{SUBDIR}/{plan.day}/<filing>/ "
               f"(staged under {STAGING_DIR}/{SUBDIR}/, published whole by one rename)")
    if plan.missing_inputs:
        out.append(f"REFUSED: {len(plan.missing_inputs)} input(s) not in this registry: " + ", ".join(f"{k} {i}" for k, i in plan.missing_inputs))
    return out


def _known_mark(p: ArtifactPlan) -> str:
    if not p.known:
        return ""
    if p.payload is None:
        return " [already registered at this uri: linked, no record]"
    if p.remote:
        return " [already registered: linked; this uri added to its locations, its verification kept]"
    return " [already registered: linked; verified now at this uri, its other locations kept]"


# --------------------------------------------------------------------------------------------
# the write
# --------------------------------------------------------------------------------------------


def staging_root(root: str) -> str:
    """``<root>/staging/analyses/`` -- where a filing is assembled: under the registry root but OUTSIDE ``records/`` and ``rulings/``, the
    two trees the harvester scans, so a half-written filing is never ingested (the adapters stage their revisions the same way)."""
    return os.path.join(root, STAGING_DIR, SUBDIR)


def filing_dir_name(filing_sha: str) -> str:
    """The name of a filing's directory under ``records/analyses/<day>/``: the first :data:`FILING_DIR_HEX` hex digits of its header's record
    sha -- the prefix the fold's refusals name a filing by (`kept filing <sha12>`), so an operator finds a filing's files from a refusal."""
    return filing_sha[:FILING_DIR_HEX]


def write_filing(root: str, plan: FilingPlan, producer: Mapping[str, object], now: str) -> List[str]:
    """Write the filing's record files through ONE RecordWriter into a fresh staging directory, ``staging/analyses/<token>/``, then
    publish the directory whole into ``records/analyses/<day>/<filing>/`` by one rename (:func:`publish_staged`): the artifact records
    first (through publish.register_files, ingest=False; a known digest's merged record, or none when the filing adds nothing to its row),
    then the analysis, its inputs, its artifact links -- four files at most, strictly increasing stamps and sequence numbers, so the fold
    meets every FK parent before the row that names it (a known digest's parent is the standing row). Returns the PUBLISHED paths, in
    sequence order. Never touches the database.

    Refuses (nothing written) when the plan names a missing input, or when a local artifact's size or mtime moved since the plan captured
    it. A failure inside the staging -- a write that raises, a header file that does not read back -- refuses by name too, the staging
    directory left in place for inspection and nothing under ``records/``: before the staging (the seventh closing pass, 2026-10-09) the
    files were created one by one in the scanned tree, and a process interrupted after the header left a header the next cycle accepted
    as a complete analysis with no inputs or no links. A process KILLED mid-staging leaves the same directory behind, which
    :func:`abandoned_stagings` names."""
    if plan.missing_inputs:
        k, i = plan.missing_inputs[0]
        raise AnalysisRefusal(f"inputs.{k}s", f"{k} {i!r} is not in this registry (and {len(plan.missing_inputs) - 1} more): an analysis is "
                                            "filed over ingested inputs; file it after they fold, or correct the id")
    verify_captured(plan.artifacts)                        # a local file that moved since its capture refuses the filing: nothing written
    H.ensure_layout(root)
    token = uuid.uuid4().hex
    w = emit.RecordWriter(os.path.join(root, STAGING_DIR), SUBDIR, token, producer, disabled=False, payload_extra={})
    staged = str(w.dir)
    assert staged == os.path.join(staging_root(root), token)
    # the writer's central stamps are a RUN producer's (`run_overrides`, `cluster_label`); a filing carries its own provenance
    w.payload_extra = {}
    try:
        written: List[str] = []
        written += PUB.register_files(root, [], now=now, ingest=False, payloads=[p.payload for p in plan.artifacts if p.payload is not None],
                                      write=w.write_many)
        path = w.write("analysis", plan.analysis_payload)
        if path:
            written.append(path)
        # the children name the FILING that wrote them (records.LEDGER_FILING_CHILDREN; the fifth closing pass, 2026-10-09): the header
        # record's sha, read back from the file just written -- the sha the harvester computes from the same line -- so the fold accepts an
        # input or a link only while this header is the analysis's kept filing or a duplicate of it, and a re-filing whose header is refused
        # as a conflicting re-filing takes its report and pages down with it instead of attaching them to the kept analysis
        filing_sha = filing_sha_of(path) if path else None
        if plan.input_payloads:
            path = w.write_many("analysis_input", [_bound(p, "analysis_input", filing_sha) for p in plan.input_payloads])
            if path:
                written.append(path)
        if plan.link_payloads:
            path = w.write_many("analysis_artifact", [_bound(p, "analysis_artifact", filing_sha) for p in plan.link_payloads])
            if path:
                written.append(path)
    except Exception as exc:
        raise AnalysisRefusal("analysis", f"the filing was interrupted while staged under {staged} ({type(exc).__name__}: {exc}): nothing "
                                          f"published, the record log holds none of it; the staged files stay there for inspection -- remove "
                                          f"the directory and re-run the filing") from exc
    assert filing_sha is not None
    final = os.path.join(H.records_dir(root), SUBDIR, plan.day, filing_dir_name(filing_sha))
    published = publish_staged(staged, final)
    assert [os.path.basename(p) for p in published] == [os.path.basename(p) for p in written], (published, written)
    return published


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_staged(staged: str, final: str) -> List[str]:
    """Publish a staged filing as ONE directory rename (atomic on one filesystem; the staging root is under the registry root for that):
    the harvester sees every file of the filing or none of them. Every staged file and the staging directory are synced first, and the
    destination's parent after, so the directory the harvester finds holds the bytes the filer wrote. A destination that already exists is
    refused by name -- another filing's header shares the twelve-hex prefix, which no two filings should -- the staging left in place.
    Returns the published paths in name (sequence) order."""
    names = sorted(os.listdir(staged))
    for fn in names:
        _fsync_path(os.path.join(staged, fn))
    _fsync_path(staged)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    if os.path.exists(final):
        raise AnalysisRefusal("analysis", f"{final} already exists (another filing's directory carries this header sha prefix): the staged "
                                          f"filing stays under {staged}, nothing published")
    os.replace(staged, final)
    _fsync_path(os.path.dirname(final))
    return [os.path.join(final, fn) for fn in names]


def abandoned_stagings(root: str) -> List[Tuple[str, List[str]]]:
    """Every staging directory under ``staging/analyses/`` with the files it holds, sorted: each a filing that was interrupted before its
    publish (or one another process is writing right now -- the two are told apart by the clock alone, so the caller names them and
    refuses nothing). Nothing in them is in the record log; a completed filing's directory is gone (renamed into ``records/``)."""
    base = staging_root(root)
    if not os.path.isdir(base):
        return []
    out: List[Tuple[str, List[str]]] = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            out.append((d, sorted(os.listdir(d))))
    return out


def abandoned_lines(stagings: Sequence[Tuple[str, Sequence[str]]]) -> List[str]:
    """The notice ``registry analysis file`` prints (to stderr, after any run) for :func:`abandoned_stagings`: empty when there are none."""
    if not stagings:
        return []
    out = [f"NOTE: {len(stagings)} staged filing(s) never published -- interrupted before the rename into records/{SUBDIR}/ (or still being "
           f"written by another process); nothing of them is in the record log:"]
    for d, files in stagings:
        out.append(f"  {d}: {len(files)} file(s) {', '.join(files) if files else '(empty)'}")
    out.append("  remove the directory by hand and re-run that filing (a filing that did publish is a duplicate at the fold)")
    return out


def filing_sha_of(header_path: str) -> str:
    """The record sha of the one `analysis` record in the header file a filing just wrote (``records.record_sha`` over the line as the
    harvester reads it): what the filing's children carry in their binding column."""
    shas = [sha for _line, _rec, sha in R.read_record_file(header_path)]
    if len(shas) != 1 or not shas[0]:
        raise AnalysisRefusal("analysis", f"{header_path} holds {len(shas)} records where the filing's one `analysis` header was expected")
    return str(shas[0])


def _bound(payload: Mapping[str, object], record_type: str, filing_sha: Optional[str]) -> Dict[str, object]:
    """A child payload with the filing's binding column (``records.LEDGER_FILING_CHILDREN``) set to the header's record sha."""
    _parent, col = R.LEDGER_FILING_CHILDREN[R.RECORD_TYPE_TABLE[record_type]]
    return dict(payload, **{col: filing_sha})


def producer_block(session: Optional[str] = None) -> Dict[str, object]:
    return emit.producer_block(PRODUCER_SCRIPT, session=session)


# --------------------------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------------------------

#: The page urls of the analysis ``a`` (a correlated subquery over the enclosing `analysis a`) as one space-separated, sorted, distinct
#: text: each link's own `extra_json.page_url` and the `page_url` of every OBJECT entry of its `extra_json.same_bytes` (a page whose every
#: file is another page's bytes is named there alone). The status page's Analyses section embeds the same text (`status.QUERIES`). The
#: entry's TYPE guards the extraction: `json_extract` raises `malformed JSON` on a bare string, so one link whose `same_bytes` held path
#: strings (free-form extra until the validator's shape check, the fourth closing pass, 2026-10-09) failed the whole listing and the
#: status section; an entry of any other type, or a `same_bytes` that is not a list, yields no url. An `extra_json` the fold did not write
#: as JSON reads as an empty object, so the scan never raises on a row's text.
PAGE_URLS_SQL = ("SELECT group_concat(u, ' ') FROM (SELECT DISTINCT u FROM ("
                 "SELECT json_extract(x.extra_json, '$.page_url') AS u FROM analysis_artifact x WHERE x.analysis_id = a.analysis_id "
                 "UNION ALL SELECT CASE WHEN s.type = 'object' THEN json_extract(s.value, '$.page_url') END FROM analysis_artifact x, "
                 "json_each(CASE WHEN json_valid(x.extra_json) THEN x.extra_json ELSE '{}' END, '$." + SAME_BYTES_KEY + "') s "
                 "WHERE x.analysis_id = a.analysis_id) WHERE u IS NOT NULL ORDER BY u)")

LIST_SQL = ("SELECT a.analysis_id, a.name, a.code_sha, a.code_ref, a.cluster_job_ids_json, a.substrate, a.report_uri, a.report_sha256, "
            "a.filed_utc, a.filed_by, a.notes, json_extract(a.extra_json, '$.provenance.kind') AS provenance, "
            "json_extract(a.extra_json, '$.provenance.relayed_by') AS relayed_by, "
            "(SELECT COUNT(*) FROM analysis_input i WHERE i.analysis_id = a.analysis_id AND i.input_kind = 'run') AS n_runs, "
            "(SELECT COUNT(*) FROM analysis_input i WHERE i.analysis_id = a.analysis_id AND i.input_kind = 'object') AS n_objects, "
            "(SELECT COUNT(*) FROM analysis_artifact x WHERE x.analysis_id = a.analysis_id) AS n_artifacts, "
            f"({PAGE_URLS_SQL}) AS page_urls "
            "FROM analysis a ORDER BY a.filed_utc DESC, a.name")


def list_analyses(conn: sqlite3.Connection, name: Optional[str] = None) -> List[Dict[str, object]]:
    rows = [dict(r) for r in conn.execute(LIST_SQL).fetchall()]
    return [r for r in rows if name is None or r["name"] == name]


def list_lines(rows: Sequence[Mapping[str, object]]) -> List[str]:
    if not rows:
        return ["(no analyses filed)"]
    out = []
    for r in rows:
        out.append(f"{r['name']}  {str(r['code_sha'])[:12]}  {r['substrate']}  filed {r['filed_utc']} by {r['filed_by']}"
                   f" ({r['provenance'] or 'in_filed_by_prose'}{', relayed by ' + str(r['relayed_by']) if r.get('relayed_by') else ''})"
                   f"  inputs {r['n_runs']} run(s) / {r['n_objects']} object(s); {r['n_artifacts']} artifact(s)")
        out.append(f"  analysis_id {r['analysis_id']}" + (f"; report {r['report_uri']}" if r.get("report_uri") else ""))
        if r.get("page_urls"):
            out.append(f"  pages {r['page_urls']}")
    return out


def inputs_of(conn: sqlite3.Connection, aid: str) -> List[Dict[str, object]]:
    return [dict(r) for r in conn.execute("SELECT input_kind, input_id FROM analysis_input WHERE analysis_id = ? ORDER BY input_kind, input_id", (aid,)).fetchall()]


def artifacts_of(conn: sqlite3.Connection, aid: str) -> List[Dict[str, object]]:
    return [dict(r) for r in conn.execute("SELECT x.role, x.artifact_id, ar.kind, ar.uri_primary, x.extra_json FROM analysis_artifact x "
                                          "LEFT JOIN artifact ar ON ar.artifact_id = x.artifact_id WHERE x.analysis_id = ? "
                                          "ORDER BY x.role, ar.uri_primary", (aid,)).fetchall()]
