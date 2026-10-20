"""BigCodeBench-200 / BFCL-200 paper rows -> registry (sources ``bcb_bfcl_200``, ``bcb_bfcl_200_widen``).

Reproduces ``paperlib.load.load_coding`` (pin 40997227): the lane emits its rows already in the paper's long schema,
one parquet per wave under the declared analysis roots (``paper_rows_*.parquet``); a DOTTED stem (``.before_D104`` …) is a
superseded snapshot the reader skips — here every snapshot is ingested with its runs ``superseded`` by the live file (§2.6),
never read as current. Producer identity fields (``object_artifact_sha256``, ``thinking``) map onto the paper's columns;
``held_out`` (a row held out of every pooled mean by ruling) is carried in the run's extra.

**Object identity from the store's own manifest.** The first wave's rows carry no object sha
(the lane predates the identity contract) but name their served store in the note (``store <store key>``) and the wave's store
directory in ``provenance``. For a fitted arm's row with no 64-hex sha, the adapter resolves
``<store_root>/<wave_dir>/<cell_id>/cells/<store_key>/manifest.json`` (a glob over every wave directory under the store root when
that exact path is absent, accepted only when it finds exactly ONE file) and reads ``provenance.artifact_sha256`` — the served
object — after cross-checking the manifest against the row: its ``cell_id`` equals the row's, its ``intervention_hash`` starts
with the store key's intervention segment and its ``harness_config_hash`` with the harness segment. The manifest's serving
identity / path, harness config hash, intervention hash and model revision are lifted onto the run. Every manifest the adapter
will read is discovered as an INPUT of the same comparison group as its parquet (its bytes are watermarked), so discovery
scans each parquet's rows for the store keys it will need. A manifest that is missing, ambiguous, malformed or disagrees with
the row leaves the row refused at the object step, by name, with the path tried in the reason — an object identity is never
guessed.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_glob, site_root
from . import declared_common as DC
from .declared_common import ARMS, HOME, LONG_COLUMNS, blank_row, finite, fnum
from .. import vocab as V

SPEC_GLOB = site_root("$WORK/bcb-bfcl-200*/analysis/paper_rows_*.parquet")
#: the lane's served stores: ``<store_root>/<wave dir>/<cell_id>/cells/<store key>/manifest.json``
STORE_ROOT = site_root("$GRADE/tinycap/bcb-bfcl-200")
MANIFEST_NAME = "manifest.json"
#: the note's store key: ``<object>__controls__t<trial>__a<attempt>__<intervention8>__<harness8>``
STORE_KEY_RE = re.compile(r"store (\S+__controls__t\d+__a\d+__[0-9a-f]{8}__[0-9a-f]{8})")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
HEX40_RE = re.compile(r"[0-9a-f]{40}")
#: The substrate the reader knows for its source as a whole (`assumed_default`, the basis filed): tinycap's Agentic runs serve on
#: Modal. Checked 2026-09-21 on every store directory under both store roots: each store manifest names its `usage_ledger` on the
#: `/vol/tinycap` volume, a `modal_owned` sandbox and `slurm_job_id: null`. The paper rows themselves carry no per-run marker (their
#: `provenance` is the grading host mirror path); a row whose manifest IS read derives the substrate from that manifest's ledger path instead.
#: The constant used to say `cluster_a` and came back `recorded` -- contradicted by every manifest.
SUBSTRATE_DEFAULT = ("modal", "tinycap's Agentic runs serve on Modal: every store manifest under the store directories names its usage_ledger "
                     "on the /vol/tinycap volume and a modal_owned sandbox (all store directories checked 2026-09-21); the paper rows carry no "
                     "per-run marker of their own")


def _list(pattern: str = SPEC_GLOB) -> List[str]:
    return sorted(store_glob(pattern))


def _hex64(x) -> Optional[str]:
    return str(x).lower() if isinstance(x, str) and HEX64_RE.fullmatch(str(x).lower()) else None


def _safe_segment(seg: str) -> bool:
    """A path segment taken from a record may not escape its directory."""
    return bool(seg) and "/" not in seg and os.sep not in seg and seg not in (".", "..")


def _read_parquet(path: str):
    """The parquet as a frame; an unreadable file refuses by name (the adapter contract) instead of crashing the pass."""
    import pandas as pd
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pyarrow raises its own hierarchy; the path and the error class are what the refusal needs
        raise AdapterRefusal(path, f"paper_rows parquet could not be read ({exc.__class__.__name__}: {exc})") from exc


class BcbBfclAdapter:
    name = "bcb_bfcl"
    #: the shared emitter's park change rides in (`paper_dose`) — see `declared_common.EMITTER_VERSION`
    #: own 2 (2026-09-21): the substrate is Modal (`assumed_default`, basis above), derived from the store manifest's usage_ledger where read
    EMISSION_VERSION = DC.emission_version(2)
    covers_sources: Tuple[str, ...] = ("bcb_bfcl_200", "bcb_bfcl_200_widen")
    rng_scope = "none"
    allow_symlinks = False
    campaign = "sdfbase_full22"

    def __init__(self, pattern: str = SPEC_GLOB, store_root: str = STORE_ROOT):
        self.pattern = pattern
        self.store_root = store_root
        self.roots = (os.path.dirname(os.path.dirname(pattern)), store_root)
        self.last_report: Optional[DC.EmitReport] = None
        self.last_facts: Dict[str, object] = {}
        #: store manifest uri -> the comparison group of the parquet whose rows name it (bound by ``discover``)
        self._manifest_group: Dict[str, str] = {}
        #: per-parquet discovery facts: which manifests its rows resolve, which they could not
        self.last_discovery: Dict[str, Dict[str, object]] = {}

    # -- discovery -------------------------------------------------------------------------------
    def discover(self) -> List[Input]:
        files = _list(self.pattern)
        if not files:
            raise AdapterRefusal(self.pattern, "no paper_rows parquet under the declared roots")
        inputs = [check_input(f, role="report", allow_symlink=self.allow_symlinks) for f in files]
        # every store manifest read() will consult is an input of the SAME group as its parquet, so a changed manifest re-runs
        # the group; discovery therefore scans the rows for the store keys of fitted arms lacking a sha (the exact rule read() applies)
        manifests: Dict[str, str] = {}
        self._manifest_group = {}
        self.last_discovery = {}
        for f in files:
            group = self._group_of_parquet(f)
            facts: Dict[str, object] = {"resolved": [], "unresolved": []}
            try:
                refs = self.store_refs(f)
            except AdapterRefusal as exc:
                # an unreadable parquet refuses ITS group when read() reaches it (same message); discovery of the other groups goes on
                facts["error"] = str(exc)
                self.last_discovery[os.path.basename(f)] = facts
                continue
            for ref in refs:
                path, how, tried = self.resolve_manifest(ref["wave_dir"], ref["cell_id"], ref["store_key"])
                if path is None:
                    facts["unresolved"].append(dict(ref, how=how, tried=tried))
                    continue
                prev = manifests.get(path)
                if prev is not None and prev != group:
                    raise AdapterRefusal(path, f"store manifest is named by parquets of two comparison groups ({prev!r}, {group!r}); "
                                               "an input belongs to exactly one group, so its watermark cannot cover both")
                manifests[path] = group
                facts["resolved"].append(dict(ref, manifest=path, how=how))
            self.last_discovery[os.path.basename(f)] = facts
        for path in sorted(manifests):
            inputs.append(check_input(path, role="report", allow_symlink=self.allow_symlinks))
            self._manifest_group[path] = manifests[path]
        return inputs

    def comparison_group(self, inp: Input) -> str:
        # the live file and its dotted snapshots form one group: a snapshot's runs name the live run they are superseded by;
        # a store manifest belongs to the group of the parquet whose rows name it (bound at discovery)
        g = self._manifest_group.get(inp.uri)
        if g is not None:
            return g
        if self.is_manifest(inp.uri):
            raise AdapterRefusal(inp.uri, "store manifest was not discovered by this adapter instance; discover() binds each manifest to its parquet's group")
        return self._group_of_parquet(inp.uri)

    def _group_of_parquet(self, path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(path))) + "/" + self.live_stem(path)

    @staticmethod
    def is_manifest(path: str) -> bool:
        return os.path.basename(path) == MANIFEST_NAME

    @staticmethod
    def live_stem(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0].split(".")[0]

    @staticmethod
    def snapshot_tag(path: str) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.split(".", 1)[1] if "." in stem else None

    @classmethod
    def is_snapshot(cls, path: str) -> bool:
        return cls.snapshot_tag(path) is not None

    # -- the store manifest ----------------------------------------------------------------------
    @staticmethod
    def row_store_ref(r: Mapping[str, object], columns) -> Dict[str, Optional[str]]:
        """What one parquet row says about its served store: the recorded object sha (64-hex or None), the note's store key and
        the wave's store directory (the first ``;`` segment of ``provenance``). ONE rule for discovery and read."""
        sha = _hex64(r.get("object_artifact_sha256")) if "object_artifact_sha256" in columns else None
        m = STORE_KEY_RE.search(str(r.get("note") or ""))
        store_key = m.group(1) if m else None
        prov = str(r.get("provenance") or "")
        wave_dir_legacy = os.path.basename(prov.split(";")[0].strip().rstrip("/")) if prov else ""
        return dict(sha=sha, store_key=store_key, wave_dir=wave_dir_legacy)

    @staticmethod
    def needs_object_sha(cell_id_legacy: str) -> bool:
        arm_legacy = ARMS.get(cell_id_legacy)
        return arm_legacy is not None and arm_legacy.object_kind not in ("none", "system_instruction")

    def store_refs(self, path: str) -> List[Dict[str, str]]:
        """The (cell_id, store_key, wave_dir) of every row of a fitted arm that names a store but carries no 64-hex object sha —
        the rows whose object the store manifest must supply."""
        raw = _read_parquet(path)
        out: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for _, r in raw.iterrows():
            cell_id_legacy = str(r.get("cell_id"))
            if not self.needs_object_sha(cell_id_legacy):
                continue
            ref = self.row_store_ref(r, raw.columns)
            if ref["sha"] or not ref["store_key"]:
                continue
            key = (ref["wave_dir"] or "", cell_id_legacy, ref["store_key"])
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(wave_dir=key[0], cell_id=cell_id_legacy, store_key=key[2], model=str(r.get("model")), eval_id=str(r.get("eval_id"))))
        return out

    def resolve_manifest(self, wave_dir_legacy: str, cell_id_legacy: str, store_key: str) -> Tuple[Optional[str], str, List[str]]:
        """``(path, how, tried)``: the manifest at the exact ``<store_root>/<wave_dir>/<cell_id>/cells/<store_key>/`` path
        (``how = exact``), else the ONE match of a glob over every wave directory (``glob``); ``missing`` / ``ambiguous`` / ``unsafe``
        with ``path = None`` otherwise. ``tried`` lists the paths (or candidates) so a refusal can name them."""
        if not (_safe_segment(cell_id_legacy) and _safe_segment(store_key)) or (wave_dir_legacy and not _safe_segment(wave_dir_legacy)):
            return None, "unsafe", [f"{wave_dir_legacy!r}/{cell_id_legacy!r}/cells/{store_key!r}"]
        tried: List[str] = []
        if wave_dir_legacy:
            exact = os.path.join(self.store_root, wave_dir_legacy, cell_id_legacy, "cells", store_key, MANIFEST_NAME)
            tried.append(exact)
            if os.path.isfile(exact):
                return exact, "exact", tried
        pattern = os.path.join(self.store_root, "*", cell_id_legacy, "cells", store_key, MANIFEST_NAME)
        tried.append(pattern)
        hits = sorted(p for p in store_glob(pattern) if os.path.isfile(p))
        if len(hits) == 1:
            return hits[0], "glob", tried
        if not hits:
            return None, "missing", tried
        return None, "ambiguous", hits

    @staticmethod
    def manifest_identity(path: str, cell_id_legacy: str, store_key: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
        """Read a store manifest and bind it to the row: ``(fields, None)`` when its ``provenance.artifact_sha256`` is 64-hex and
        its ``cell_id`` / ``intervention_hash`` / ``harness_config_hash`` agree with the row's cell and store key; ``(None, why)``
        otherwise. ``fields`` carries the sha and the serving fields lifted onto the run."""
        try:
            m = DC.read_json(path)
        except (OSError, ValueError) as exc:
            return None, f"store manifest {path} could not be read ({exc.__class__.__name__}: {exc})"
        if not isinstance(m, dict):
            return None, f"store manifest {path} is not a JSON object"
        prov = m.get("provenance") if isinstance(m.get("provenance"), dict) else {}
        sha = _hex64(prov.get("artifact_sha256"))
        if sha is None:
            return None, f"store manifest {path} records no 64-hex provenance.artifact_sha256 (the served object)"
        m_cell_legacy = m.get("cell_id", prov.get("cell_id"))
        if str(m_cell_legacy) != cell_id_legacy:
            return None, f"store manifest {path} names cell_id {m_cell_legacy!r}, the row names {cell_id_legacy!r}"
        if prov.get("cell_id") is not None and str(prov.get("cell_id")) != cell_id_legacy:
            return None, f"store manifest {path} provenance.cell_id {prov.get('cell_id')!r} disagrees with its cell_id {cell_id_legacy!r}"
        segs = store_key.split("__")
        intervention8, harness8 = segs[-2], segs[-1]
        ih = m.get("intervention_hash", prov.get("intervention_hash"))
        if not (isinstance(ih, str) and ih.lower().startswith(intervention8)):
            return None, f"store manifest {path} intervention_hash {ih!r} does not start with the store key's intervention segment {intervention8}"
        pih = prov.get("intervention_hash")
        if isinstance(pih, str) and pih.lower() != str(ih).lower():
            return None, f"store manifest {path} provenance.intervention_hash {pih!r} disagrees with its intervention_hash {ih!r}"
        hch = m.get("harness_config_hash")
        if isinstance(hch, str) and hch and not hch.lower().startswith(harness8):
            return None, f"store manifest {path} harness_config_hash {hch!r} does not start with the store key's harness segment {harness8}"
        serving = prov.get("serving") if isinstance(prov.get("serving"), dict) else {}
        serving_identity = m.get("serving_identity") or serving.get("serving_identity")
        serving_path = m.get("serving_path")
        if not (isinstance(serving_path, str) and serving_path in V.SERVING_PATHS and serving_path != "unverified"):
            serving_path = DC.serving_path_from_identity(serving_identity)[0] if serving_identity else "unverified"
        revision = m.get("model_revision") or prov.get("model_revision")
        revision = str(revision).lower() if isinstance(revision, str) and HEX40_RE.fullmatch(str(revision).lower()) else None
        producer = m.get("producer") if isinstance(m.get("producer"), dict) else {}
        route = m.get("route") if isinstance(m.get("route"), dict) else {}
        sandbox = route.get("bcb_sandbox") if isinstance(route.get("bcb_sandbox"), dict) else {}
        return dict(sha=sha, cell_id=str(m_cell_legacy), intervention_hash=str(ih), harness_config_hash=(str(hch) if isinstance(hch, str) and hch else None),
                    serving_identity=(str(serving_identity) if serving_identity else None), serving_path=str(serving_path), model_revision=revision,
                    producer_tree_sha=(str(producer.get("tree_sha")) if isinstance(producer.get("tree_sha"), str) else None),
                    engine=(str(m.get("engine")) if isinstance(m.get("engine"), str) else None),
                    # the substrate witnesses: the usage ledger's path (on the Modal volume for every tinycap store) and the sandbox provider
                    usage_ledger=(str(m.get("usage_ledger")) if isinstance(m.get("usage_ledger"), str) and m.get("usage_ledger") else None),
                    sandbox_provider=(str(sandbox.get("provider")) if isinstance(sandbox.get("provider"), str) else None)), None

    def _manifest_rel(self, path: str) -> str:
        root = self.store_root.rstrip("/") + "/"
        return path[len(root):] if path.startswith(root) else path

    # -- the rows --------------------------------------------------------------------------------
    def long_rows(self, path: str, registered: Optional[Set[str]] = None) -> Tuple[List[dict], Dict[str, object]]:
        """The parquet's rows in the paper's long schema plus the emitter's underscore keys. ``registered`` (from ``read``) is the
        set of store-manifest paths discovered as inputs: a manifest resolved here but absent from it is NOT watermarked, so the
        row is refused rather than read from an unregistered file. A row whose object the store cannot supply carries
        ``_manifest_refusal`` (the reason, naming the path tried); ``read`` withholds such a MEASURED row from the emitter and
        files the reason among its refusals."""
        raw = _read_parquet(path)
        has_held_out = "held_out" in raw.columns
        rows: List[dict] = []
        stem = os.path.splitext(os.path.basename(path))[0]
        live_stem = self.live_stem(path)
        live = os.path.join(os.path.dirname(path), live_stem + ".parquet")
        tag = self.snapshot_tag(path)
        superseded = (f"superseded snapshot {os.path.basename(path)} of the live file {os.path.basename(live)} (load_coding skips dotted stems)" if tag else None)
        when = DC.mtime_utc(path)
        manifest_cache: Dict[Tuple[str, str, str], Tuple[Optional[Dict[str, object]], Optional[str], Optional[str], str]] = {}
        n_from_manifest = 0
        manifest_refused: List[str] = []
        manifests_read: Set[str] = set()
        for _, r in raw.iterrows():
            row = blank_row()
            for c in LONG_COLUMNS:
                if c in raw.columns:
                    v = r[c]
                    row[c] = v if not (isinstance(v, float) and v != v) else float("nan")
            # producer identity fields -> the paper's columns (load.apply_producer_identity)
            ref = self.row_store_ref(r, raw.columns)
            sha = ref["sha"]
            row["object_sha256"] = sha if sha else float("nan")
            tk = r.get("thinking") if "thinking" in raw.columns else None
            row["thinking_mode"] = tk if tk in ("on", "off", "not_available") else float("nan")
            if not isinstance(row.get("source"), str) or not row["source"]:
                row["source"] = f"agentic/{stem}"
            row["held_out"] = bool(r["held_out"]) if has_held_out and r["held_out"] == r["held_out"] else False
            if isinstance(row.get("status"), str) and row["status"] == "graded":
                pass
            store_key = ref["store_key"]
            # the wave's store directory (the first provenance segment): a store key repeats across wave views, the directory does not
            wave_dir_legacy = ref["wave_dir"]
            run_key = (f"{wave_dir_legacy}/{store_key}" if wave_dir_legacy and store_key else None)
            cell_id_legacy = str(row.get("cell_id"))
            arm_legacy = ARMS.get(cell_id_legacy)
            eval_id = str(row.get("eval_id"))
            decode = str(r.get("decode") or "")
            dec: Dict[str, object] = {}
            for k in ("temperature", "top_p", "top_k"):
                mm = re.search(rf"{k.replace('_', '_')}[= ]([0-9.]+)", decode.replace("T=", "temperature="))
                if mm:
                    dec[k] = float(mm.group(1)) if k != "top_k" else int(float(mm.group(1)))
            source = "greedy" if eval_id == "bfcl_200" else ("model_card" if "model card" in decode else "unverified")
            # a fitted arm's row with no object sha names its served store; the store's own manifest supplies the sha
            manifest_fields: Optional[Dict[str, object]] = None
            manifest_refusal: Optional[str] = None
            manifest_path: Optional[str] = None
            manifest_how = ""
            if sha is None and store_key and self.needs_object_sha(cell_id_legacy):
                ck = (wave_dir_legacy or "", cell_id_legacy, store_key)
                if ck not in manifest_cache:
                    manifest_cache[ck] = self._resolve_and_bind(wave_dir_legacy, cell_id_legacy, store_key, registered)
                manifest_fields, manifest_refusal, manifest_path, manifest_how = manifest_cache[ck]
                if manifest_fields is not None:
                    sha = str(manifest_fields["sha"])
                    row["object_sha256"] = sha
                    n_from_manifest += 1
                    manifests_read.add(str(manifest_path))
                else:
                    manifest_refused.append(f"{cell_id_legacy}/{eval_id}: {manifest_refusal}")
            instrument: Dict[str, object] = {"harness": "bfcl_eval" if eval_id == "bfcl_200" else "inspect_evals", "harness_version": "unverified",
                                             "token_cap_tokens": int(fnum(row.get("decode_cap_tokens"))) if finite(fnum(row.get("decode_cap_tokens"))) else 0, "decode": dec,
                                             "decode_source": source, "decode_seed_rule": "per_item" if source == "model_card" else "none",
                                             "thinking_mode": tk if tk in ("on", "off", "not_available") else None, "serving_path": "unverified",
                                             "scoring_rule": "ast_match" if eval_id == "bfcl_200" else "pass_at_1_sandbox", "instrument_config_hash": f"{eval_id}_protocol", "fit_positions": "all"}
            confidence: Dict[str, Tuple[str, str]] = {"serving_path": ("unverified", "the paper rows name the served leg but not its engine (a store's evidence.json is not read here)")}
            extra: Dict[str, object] = {"held_out": row["held_out"], "wave_file": os.path.basename(path), "snapshot_tag": tag}
            model_revision = None
            if manifest_fields is not None:
                rel = self._manifest_rel(str(manifest_path))
                instrument.update(serving_identity=manifest_fields["serving_identity"], serving_path=manifest_fields["serving_path"],
                                  harness_config_hash=manifest_fields["harness_config_hash"], intervention_hash=manifest_fields["intervention_hash"])
                if manifest_fields["serving_path"] != "unverified":
                    confidence["serving_path"] = ("recorded", f"the store's manifest ({rel}) records the serving path / identity")
                else:
                    confidence["serving_path"] = ("unverified", f"the store's manifest ({rel}) names no serving path and its serving identity maps to none")
                model_revision = manifest_fields["model_revision"]
                extra.update(object_sha_source="store manifest provenance.artifact_sha256 (D60)", store_manifest=rel, store_manifest_resolution=manifest_how,
                             store_producer_tree_sha=manifest_fields["producer_tree_sha"], store_engine=manifest_fields["engine"],
                             store_usage_ledger=manifest_fields["usage_ledger"], store_sandbox_provider=manifest_fields["sandbox_provider"])
            # the substrate: the manifest's usage_ledger path (a per-run marker on the Modal volume) where the manifest is read, else the
            # reader's declared default for the source; the paper row's `provenance` is the grading host mirror and decides nothing
            substrate_path = str(manifest_fields["usage_ledger"] or "") if manifest_fields is not None else ""
            row.update(_when=when, _code_sha="unverified", _code_sha_reason="the paper rows name the launch sha only in the note", _object_id=sha,
                       _substrate_default=SUBSTRATE_DEFAULT, _substrate_path=substrate_path,
                       _producer_key=((run_key or store_key or f"{row['source']}|{row.get('model')}|{row.get('view')}|{cell_id_legacy}|{eval_id}") + (f"@{tag}" if tag else "")),
                       _object_key=("none" if cell_id_legacy == "none" else (f"{cell_id_legacy}@{store_key.split('__')[-2]}" if store_key else cell_id_legacy)),
                       _store_uri=str(r.get("provenance") or "") or None, _superseded=superseded, _model_revision=model_revision,
                       _instrument=instrument, _confidence=confidence,
                       _judge=None, _key=f"{row['source']}|{row.get('model')}|{row.get('view')}|{cell_id_legacy}|{eval_id}" + (f"@{tag}" if tag else ""), _n_items_planned=200,
                       _superseded_by_key=(f"{row['source']}|{row.get('model')}|{row.get('view')}|{cell_id_legacy}|{eval_id}" if tag else None),
                       _degeneration={"unit": "items"}, _extra=extra,
                       _strength_coefficient=(1.0 if arm_legacy and arm_legacy.object_kind not in ("none", "system_instruction") else None),
                       _is_operating_point=(1 if arm_legacy and arm_legacy.object_kind not in ("none", "system_instruction") else None))
            if manifest_refusal:
                row["_manifest_refusal"] = manifest_refusal
            if finite(row.get("delta")):
                row.update(_anchor_key=f"{row['source']}|{row.get('model')}|{row.get('view')}|none|{eval_id}" + (f"@{tag}" if tag else ""), _anchor_kind="same_standup_none",
                           _n_pairs=int(fnum(row.get("n"))) if finite(fnum(row.get("n"))) else None)
            rows.append(row)
        facts = dict(file=path, superseded=bool(superseded), n_rows=len(rows), n_held_out=int(sum(1 for r in rows if r.get("held_out"))),
                     n_object_from_manifest=n_from_manifest, manifests=sorted(self._manifest_rel(p) for p in manifests_read), manifest_refused=manifest_refused)
        return rows, facts

    def _resolve_and_bind(self, wave_dir_legacy: str, cell_id_legacy: str, store_key: str, registered: Optional[Set[str]]
                          ) -> Tuple[Optional[Dict[str, object]], Optional[str], Optional[str], str]:
        """``(fields, refusal, path, how)`` for one store reference: resolve the manifest, require it among the registered inputs
        when ``read`` supplied them, then bind it to the row."""
        path, how, tried = self.resolve_manifest(wave_dir_legacy, cell_id_legacy, store_key)
        if path is None:
            if how == "ambiguous":
                why = (f"fitted object with no 64-hex artifact sha in the record; the store key {store_key} resolves to {len(tried)} manifests under "
                       f"{self.store_root} ({', '.join(self._manifest_rel(p) for p in tried)}): one object is never picked from several")
            elif how == "unsafe":
                why = f"fitted object with no 64-hex artifact sha in the record; the row's store reference {tried[0]} is not a plain path segment"
            else:
                why = (f"fitted object with no 64-hex artifact sha in the record; no store manifest at {tried[0]}"
                       + (f" (nor one match of {tried[1]})" if len(tried) > 1 else ""))
            return None, why, None, how
        if registered is not None and path not in registered and os.path.realpath(path) not in registered:
            return None, (f"fitted object with no 64-hex artifact sha in the record; the store manifest {path} resolved at read time was not among the "
                          "discovered inputs (D12: its bytes are not watermarked) — re-run discovery"), path, how
        fields, why = self.manifest_identity(path, cell_id_legacy, store_key)
        if fields is None:
            return None, f"fitted object with no 64-hex artifact sha in the record; {why}", path, how
        return fields, None, path, how

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        # the live file first, then its snapshots (emit_rows orders superseded rows after live ones anyway); one batch per group;
        # the group's store manifests ride along as inputs and are read through long_rows, never iterated as parquets
        parquets = [i for i in inputs if not self.is_manifest(i.uri)]
        registered: Set[str] = set()
        for i in inputs:
            if self.is_manifest(i.uri):
                registered.update((i.uri, i.path))
        rows: List[dict] = []
        facts: Dict[str, object] = {}
        for inp in sorted(parquets, key=lambda i: (self.is_snapshot(i.path), i.path)):
            r, f = self.long_rows(inp.path, registered=registered)
            rows += r
            facts[os.path.basename(inp.path)] = f
        self.last_facts[group] = facts
        # a MEASURED row whose object the store manifest could not supply is withheld from the emitter — it would refuse the row at
        # the object step anyway — so the refusal filed carries the path tried; an unmeasured row is a disposition and needs no object
        keep: List[dict] = []
        withheld: List[str] = []
        for row in rows:
            why = row.pop("_manifest_refusal", None)
            if why and not DC.is_disposition(row.get("status")) and finite(row.get("value")):
                withheld.append(f"{row.get('model')}/{row.get('cell_id')}/{row.get('eval_id')}: {row.get('model')}/{row.get('cell_id')}: {why}")
            else:
                keep.append(row)
        batch = RecordBatch(f"{self.name}:{group}", group)
        rep = DC.emit_rows(batch, ctx, keep, adapter=self.name, campaign=self.campaign, reader="load.load_coding",
                           seed_provenance="rows already in the paper's long schema; nothing recomputed", code_sha="unverified",
                           code_sha_reason="the paper rows name the launch sha only in the note")
        rep.refusals.extend(withheld)
        self.last_report = rep
        yield batch

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        out: List[SourceStateRow] = []
        for i in inputs:
            member = self._manifest_rel(i.uri) if self.is_manifest(i.uri) else os.path.basename(i.uri)
            out.append(SourceStateRow(f"{self.name}:{member}", i.uri, "$", 1, 1, [member], i.sha256))
        return out


def register() -> None:
    from .base import register as _register
    _register(BcbBfclAdapter.name, lambda: BcbBfclAdapter())
