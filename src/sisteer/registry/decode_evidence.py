"""The request's OBSERVED sampler, read from store evidence -- ONE reader for the three writers; a ``decode_policy_id``
of ``'unrecorded'`` is a PRODUCER gap.

STDLIB ONLY (json / glob / os / hashlib), importable under ``python3 -S`` and ``python3 -I`` beside :mod:`hooks` (the store recorder), by
the full22 closer through its ``sys.path`` bootstrap, and as a bare script on a cluster node (the backfill's read pass:
``python3 -I decode_evidence.py --runs <json> --json``, or the module's source piped on stdin with a trailer calling
:func:`remote_main`). It never imports ``sisteer``: the literals it spells (:data:`PARAM_KEYS`, :data:`SAMPLED_CONTROL_SPELLINGS`,
:data:`MANIFEST_KEY_REQUEST_SAMPLERS`) are pinned equal to their owners by ``tests/test_decode_evidence.py``.

What is filed, per run, from the family's own artifact (never the plan, the instrument, the policy row or an attacker's preset line):

* ``decode_sent_json`` -- the SIX modelled keys the request body carried (``decode_policy.PARAM_KEYS``), numeric, ``top_k`` an int,
  ``temperature`` present whenever the set is non-empty (``bon_served.sampler_preset_problem``'s rule); every other key of the evidence
  block (``do_sample``, ``cap``, ``arm``, ``seed_base``, ``trial``, ``max_new_tokens``, ``batch_size``, ``escalated_from`` ...) is
  UNMODELLED and travels under ``extra_json.decode_evidence``, never into the sent set (the matcher's ``run_sent_unmodelled`` CTE makes
  any such key a ``no_match``);
* the typed ``decode_temperature_sent`` / ``decode_top_p_sent`` / ``decode_top_k_sent``, for the keys the sent set holds;
* ``decode_resolved_json`` -- ONLY a request recorder's server-resolved sampler (``<store root>/resolved_sampler.json``, the cellstore
  adapter's 27b rule), never the policy row, never a manifest's ``gen``;
* a ``field_confidence_json`` entry per column: ``recorded`` naming the artifact, its sha256 and the corroborating witnesses, or
  ``unverified`` with the reason when no evidence exists (the row keeps ``'{}'`` and reads ``unrecorded`` as before -- but says why).
  ONE exception (the registry owner, 2026-10-13 18:57 PT, Change 1): a request recorder whose client rows recorded NO per-request sampler
  (``sent_rows`` 0) proves the sent set only through its ``--expect`` REFERENCE -- a declaration of the keys the client builds its
  requests from, which cannot show which keys the client omitted (a ``min_p`` 0 or a ``presence_penalty`` 0 equals the engine's
  default). That sent set is filed as ``derived``, naming the pinned request builder (:data:`RECORDER_REQUEST_BUILDER` at the run's
  code sha) beside the recorder document and its sha256; the server-resolved values stay ``recorded``. A read that names no code sha
  files nothing for the sent columns and says why: a derived claim without its pin is a guess about which builder ran.

The rule behind the refusals (:class:`DecodeEvidenceContradiction` / :class:`DecodeEvidenceShape`): a store is drawn under ONE sampler --
two samplers across a store's rows or shards, a manifest disagreeing with its rows, a driver stating one request twice and differently, a
non-numeric or bool value, a ``top_k`` that is not an int -- refuse by name, never average, never pick. Absence is not a refusal: it files
the ``unverified`` entry and nothing else.
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: the six sampling parameters the policy table models -- ``decode_policy.PARAM_KEYS`` and the matcher's ``_params``, pinned equal by test
PARAM_KEYS: Tuple[str, ...] = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
#: the served driver's SAMPLED over-refusal controls (``served_capability_run.SAMPLED_CONTROLS``), the only spellings whose request
#: sampler is ``resolved_config.gen`` -- pinned equal by test
SAMPLED_CONTROL_SPELLINGS: Tuple[str, ...] = ("xstest", "or_bench_hard_1k")
#: the served driver's LEGACY controls (``served_capability_run.SERVED_CONTROLS``): the controls whose request the DRIVER builds and whose
#: sampler the cell MANIFEST states (``request_samplers``) -- never the store-level request recorder, which describes the tinycap atoms'
#: requests (Major 1: a completion patch must not hand one task another task's sampler) -- pinned equal by test
LEGACY_CONTROL_SPELLINGS: Tuple[str, ...] = ("mmlu",) + SAMPLED_CONTROL_SPELLINGS
#: the sampler the served driver's GREEDY judged-control request body carries (``served_capability_run.GREEDY_CHAT_SAMPLER``, the
#: object ``_chat_payload`` spreads under the ``--greedy`` opt-in; ``MMLU_SCORING_SAMPLER`` is the same number) -- pinned equal by test.
#: A ``gen_stats.json`` decode block with ``do_sample`` false and no sampler key is the GREEDY marker: it corroborates exactly this set.
GREEDY_CHAT_SAMPLER: Dict[str, float] = {"temperature": 0.0}
#: the served driver's top-level, UNHASHED manifest statement of each legacy control's request sampler
#: (``tinycap.contracts.MANIFEST_KEY_REQUEST_SAMPLERS``) -- pinned equal by test
MANIFEST_KEY_REQUEST_SAMPLERS = "request_samplers"
#: the typed run columns the sent set fills
TYPED_COLUMNS: Dict[str, str] = {"temperature": "decode_temperature_sent", "top_p": "decode_top_p_sent", "top_k": "decode_top_k_sent"}
#: how this reader names itself in ``extra_json.decode_evidence.reader``
READER = "sisteer.registry.decode_evidence"
RECORDER_FILE = "resolved_sampler.json"
#: the sent columns' confidence: a per-request OBSERVATION (the client rows' recorded sampler, a manifest the rows corroborate) is
#: ``recorded``; a recorder's DECLARED reference (``sent_rows`` 0) is ``derived`` against the pinned request builder
SENT_RECORDED = "recorded"
SENT_DERIVED = "derived"
#: the request builder a declared recorder reference derives the sent set from: the tinycap provider's ``sampling_params`` is the ONE
#: place a client request's sampling parameters are resolved onto the body (temperature / top_p, and presence_penalty when the row sets
#: it, as GenerateConfig fields; top_k / min_p / repetition_penalty in ``extra_body`` when the row sets them) -- the path and the
#: function are pinned to the tree by test
RECORDER_REQUEST_BUILDER = "src/sisteer/capeval/tiers/inspect_served.py sampling_params"
BON_MANIFEST = "bon_served_manifest.json"
BON_RUN = "bon_served_run.json"
BON_RESPONSES = "bon_responses.jsonl"
FAMILIES: Tuple[str, ...] = ("bon", "multiturn", "static", "padding", "controls", "misalign", "recorder")
#: the ``extra_json`` key the evidence block rides under
EXTRA_KEY = "decode_evidence"


class DecodeEvidenceProblem(ValueError):
    """The evidence cannot be filed: a contradiction or a shape problem (the writers REFUSE the run by name on either)."""


class DecodeEvidenceContradiction(DecodeEvidenceProblem):
    """Two samplers in one store, or a producer disagreeing with itself."""


class DecodeEvidenceShape(DecodeEvidenceProblem):
    """A value among the six that is not a number (bools included), a ``top_k`` that is not an int, an artifact that is not JSON, a
    torn row record, a BoN preset naming a key outside the sampler set."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except ValueError as exc:
        raise DecodeEvidenceShape(f"{path}: not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise DecodeEvidenceShape(f"{path}: not a JSON object ({type(doc).__name__})")
    return doc


def _num(key: str, value: object, where: str) -> float:
    """A modelled value as the registry compares it: a float, ``top_k`` an int; a bool or a non-number refuses by name."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecodeEvidenceShape(f"{where}: {key}={value!r} is not a number")
    if key == "top_k":
        if isinstance(value, float) and not value.is_integer():
            raise DecodeEvidenceShape(f"{where}: top_k={value!r} is not an integer")
        return int(value)
    return float(value)


def split_six(block: Mapping[str, object], where: str, *, require_temperature: bool = True) -> Tuple[Dict[str, float], Dict[str, object]]:
    """``(sent, unmodelled)``: the six modelled keys of ``block`` normalised (floats; ``top_k`` an int), and every other key with its value.
    A non-empty sent set without ``temperature`` is a shape problem when ``require_temperature`` (the request always carries one)."""
    if not isinstance(block, Mapping):
        raise DecodeEvidenceShape(f"{where}: the sampler block is {type(block).__name__}, not an object")
    sent: Dict[str, float] = {}
    unmodelled: Dict[str, object] = {}
    for k, v in block.items():
        if k in PARAM_KEYS:
            if v is None:
                continue                       # a key the producer spells but did not send (the canonical instrument's null slots)
            sent[k] = _num(k, v, where)
        else:
            unmodelled[str(k)] = v
    if sent and require_temperature and "temperature" not in sent:
        raise DecodeEvidenceShape(f"{where}: the sampler states {sorted(sent)} but no temperature -- the request always carries one")
    return sent, unmodelled


def recorded_entry(source: str) -> Dict[str, str]:
    return {"confidence": "recorded", "source": source}


def derived_entry(source: str) -> Dict[str, str]:
    return {"confidence": "derived", "source": source}


def unverified_entry(reason: str) -> Dict[str, str]:
    return {"confidence": "unverified", "reason": reason}


def _is_code_sha(value: object) -> bool:
    """A forty-hex commit sha (the run's ``code_sha``); ``unverified`` and anything else is not a pin."""
    s = str(value or "")
    return len(s) == 40 and all(c in "0123456789abcdef" for c in s)


@dataclasses.dataclass
class Evidence:
    """One run's decode evidence. ``why`` None = FILED (``sent`` non-empty: :meth:`recorded` is true); a string = UNVERIFIED by name
    (nothing filed for the SENT columns but the ``unverified`` entry). ``sent_confidence`` is the sent columns' entry:
    :data:`SENT_RECORDED` for a per-request observation, :data:`SENT_DERIVED` for a recorder's declared reference read against the
    pinned request builder (``derivation`` names it). ``resolved`` is the recorder's server-resolved sampler, an OBSERVATION of the
    document (one distinct sampler under an AGREES verdict) filed ``recorded`` whenever it was read (:meth:`resolved_read`) -- with the
    sent set or apart from it, when the same document proves no sent set (a declared reference with no code sha to derive against lost
    the resolved values with it, where the cellstore adapter kept them)."""

    family: str
    sent: Dict[str, float] = dataclasses.field(default_factory=dict)
    resolved: Dict[str, float] = dataclasses.field(default_factory=dict)
    source: Optional[str] = None                   # "<evidence path>#<field path>"
    evidence_path: Optional[str] = None
    sha256: Optional[str] = None
    unmodelled: Dict[str, Dict[str, object]] = dataclasses.field(default_factory=dict)    # per block: "<rel path>#<field>" -> {key: value}
    producer_policy_id: Optional[str] = None
    producer_policy_id_source: Optional[str] = None
    witnesses: Dict[str, object] = dataclasses.field(default_factory=dict)
    why: Optional[str] = None
    resolved_source: Optional[str] = None
    read_utc: str = dataclasses.field(default_factory=utc_now)
    sent_confidence: str = SENT_RECORDED
    derivation: Optional[str] = None               # the derived entry's basis: the pinned request builder and the code sha

    # -- the writers' shapes ---------------------------------------------------------------------------------------------------
    def recorded(self) -> bool:
        """FILED: a sent set with no refusal (``recorded`` OR ``derived`` sent columns -- see ``sent_confidence``)."""
        return self.why is None and bool(self.sent)

    def resolved_read(self) -> bool:
        """The recorder READ a server-resolved sampler: ``decode_resolved_json`` is filed ``recorded`` -- independent of :meth:`recorded`
        (a document that resolved one sampler but proves no sent set files the resolved column alone, beside the ``unverified`` sent
        entry naming why)."""
        return bool(self.resolved)

    def confidence_source(self) -> str:
        """The sent entry's source text: the artifact and field, its sha256, the witnesses' summary, and for a derived set its basis."""
        parts = [str(self.source)]
        if self.sha256:
            parts.append(f"sha256 {self.sha256[:12]}")
        summary = self.witnesses.get("summary")
        if summary:
            parts.append(str(summary))
        if self.sent_confidence == SENT_DERIVED and self.derivation:
            parts.append(str(self.derivation))
        return "; ".join(parts)

    def columns(self) -> Dict[str, object]:
        """The run columns the evidence sets: ``decode_sent_json`` and the typed columns of the keys carried when the sent set is FILED
        (:meth:`recorded`); ``decode_resolved_json`` whenever the recorder read one (:meth:`resolved_read`), with the sent set or
        alone -- the server-resolved sampler is an observation of the document, never withheld because the sent set could not be filed
        from it (Major 1)."""
        out: Dict[str, object] = {}
        if self.recorded():
            out["decode_sent_json"] = dict(self.sent)
            for k, col in TYPED_COLUMNS.items():
                if k in self.sent:
                    out[col] = self.sent[k]
        if self.resolved_read():
            out["decode_resolved_json"] = dict(self.resolved)
        return out

    def confidence_entries(self) -> Dict[str, Dict[str, str]]:
        """Per column (the fold reads a patch's confidence JSON per column): the sent entry (``recorded`` or ``derived`` per
        ``sent_confidence``) on every sent column set, else the ``unverified`` entry naming why on ``decode_sent_json``; and ``recorded``
        on the resolved column whenever one was read, in either case."""
        out: Dict[str, Dict[str, str]]
        if self.recorded():
            entry = (derived_entry if self.sent_confidence == SENT_DERIVED else recorded_entry)(self.confidence_source())
            out = {"decode_sent_json": dict(entry)}
            for k, col in TYPED_COLUMNS.items():
                if k in self.sent:
                    out[col] = dict(entry)
        else:
            out = {"decode_sent_json": unverified_entry(str(self.why or "no decode evidence"))}
        if self.resolved_read():
            out["decode_resolved_json"] = recorded_entry(str(self.resolved_source))
        return out

    def extra_block(self) -> Dict[str, object]:
        """``extra_json.decode_evidence``: the artifact, the unmodelled keys, the producer's own policy statement, the witnesses, the sent
        columns' confidence (None when no sent set is filed) and (derived) its basis."""
        return {"family": self.family, "source": self.source, "evidence_path": self.evidence_path, "sha256": self.sha256,
                "unmodelled_keys": {k: sorted(v) for k, v in self.unmodelled.items()},
                "producer_policy_id": self.producer_policy_id, "producer_policy_id_source": self.producer_policy_id_source,
                "witnesses": {k: v for k, v in self.witnesses.items() if k != "summary"}, "reader": READER, "read_utc": self.read_utc,
                "sent_confidence": self.sent_confidence if self.recorded() else None, "derivation": self.derivation}

    def patch_fields(self) -> Dict[str, object]:
        """The ``run_patch`` fields (``records.RUN_PATCHABLE_FIELDS`` members only): the columns, the per-column entries and the extra
        block. An absence contributes the ``unverified`` entry alone; a recorder that resolved a sampler but proves no sent set
        contributes the resolved column, its ``recorded`` entry and the block (the document, its sha256, the witnesses -- the declared
        keys among them when the pin was what was missing) beside the ``unverified`` sent entry."""
        out: Dict[str, object] = dict(self.columns())
        out["field_confidence_json"] = self.confidence_entries()
        if self.recorded() or self.resolved_read():
            out["extra_json"] = {EXTRA_KEY: self.extra_block()}
        return out

    def to_json(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------------------------------------------------------------------
# per-family readers
# ------------------------------------------------------------------------------------------------------------------------------


def _rows(path: str, *, where: str) -> List[Dict[str, Any]]:
    """Every JSON object line of a jsonl file; a line that is not a JSON object refuses by name (a store still being written is not
    evidence -- the closers refuse unparsable lines before any close, so a closed store has none)."""
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise DecodeEvidenceShape(f"{where}: line {i} is not JSON (a torn row)") from None
            if not isinstance(row, dict):
                raise DecodeEvidenceShape(f"{where}: line {i} is not a JSON object")
            out.append(row)
    return out


def _shard_dirs(store: str, shards: Optional[Sequence[str]], required: str) -> List[str]:
    """The shard directories to read: ``shards`` by name (the inspected subset) else every present ``shard*`` holding ``required``."""
    if shards:
        return [os.path.join(store, s) for s in shards]
    return sorted(d for d in glob.glob(os.path.join(store, "shard*")) if os.path.isdir(d) and os.path.isfile(os.path.join(d, required)))


def from_bon_store(store: str) -> Evidence:
    """A best-of-N store: ``bon_served_manifest.json#decode`` (the completion artifact), corroborated by every ``bon_responses.jsonl`` row's
    recorded sampler (the stdlib twin of ``bon_climb.store_sampler``'s one-sampler-per-store scan, pinned equal by test); the producer's
    own ``decode_policy_id`` / ``bon_decode_preset`` are carried as statements, never filed as the match."""
    ev = Evidence(family="bon")
    man = os.path.join(store, BON_MANIFEST)
    if not os.path.isfile(man):
        ev.why = f"{man}: no {BON_MANIFEST} (the unit has not finished writing, or a legacy store)"; return ev
    doc = _load_json(man)
    ev.evidence_path = man; ev.sha256 = sha256_file(man)
    ev.producer_policy_id = doc.get("decode_policy_id") if isinstance(doc.get("decode_policy_id"), str) else None
    ev.producer_policy_id_source = f"{BON_MANIFEST}#decode_policy_id" if ev.producer_policy_id else None
    ev.witnesses.update({"bon_decode_preset": doc.get("bon_decode_preset"), "n_legacy_unrecorded_rows": doc.get("n_legacy_unrecorded_rows"),
                         "seed_rule": doc.get("seed_rule")})
    run_doc = os.path.join(store, BON_RUN)
    if os.path.isfile(run_doc):
        try:
            ev.witnesses["bon_decode_preset_declared"] = _load_json(run_doc).get("bon_decode_preset_declared")
        except DecodeEvidenceProblem as exc:
            ev.witnesses["bon_decode_preset_declared"] = f"unreadable: {exc}"
    block = doc.get("decode")
    if block is None:
        ev.why = (f"{man}#decode absent: a legacy BoN store drawn before the sampler was recorded "
                  f"(n_legacy_unrecorded_rows={doc.get('n_legacy_unrecorded_rows')!r})")
        return ev
    sent, unmodelled = split_six(block, f"{man}#decode")
    if unmodelled:
        # bon_served.sampler_preset_problem: a preset carries sampler keys only -- a key outside the set is never a BoN store's sampler
        raise DecodeEvidenceShape(f"{man}#decode carries key(s) outside the sampler set {sorted(unmodelled)}; allowed: {list(PARAM_KEYS)}")
    if not sent:
        ev.why = f"{man}#decode is empty: the store states no sampler key"; return ev
    ev.witnesses["manifest_decode"] = dict(sent)
    # the rows: bon_climb.store_sampler's scan -- every row that records a sampler records the same one, and no mixture of recorded and
    # legacy rows
    rows_path = os.path.join(store, BON_RESPONSES)
    seen: Dict[str, Dict[str, Any]] = {}
    n_recorded = n_legacy = n_scanned = 0
    if os.path.isfile(rows_path):
        with open(rows_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue            # a torn tail, skipped as store_sampler and resume skip it
                if not isinstance(r, dict):
                    continue
                n_scanned += 1
                if "decode" not in r and "bon_decode_preset" not in r:
                    n_legacy += 1; continue
                if not isinstance(r.get("decode"), dict) or not isinstance(r.get("bon_decode_preset"), str):
                    raise DecodeEvidenceShape(f"{rows_path}: row {(r.get('goal_id'), r.get('sample_idx'))} carries a torn sampler record "
                                              f"(decode={r.get('decode')!r}, bon_decode_preset={r.get('bon_decode_preset')!r})")
                n_recorded += 1
                key = canonical({"decode": r["decode"], "bon_decode_preset": r["bon_decode_preset"]})
                seen.setdefault(key, {"decode": dict(r["decode"]), "bon_decode_preset": r["bon_decode_preset"], "decode_policy_id": r.get("decode_policy_id")})
    if len(seen) > 1:
        raise DecodeEvidenceContradiction(f"{rows_path}: rows record {len(seen)} different samplers {[json.loads(k) for k in seen]} -- a mixture is never a store")
    if n_recorded and n_legacy:
        raise DecodeEvidenceContradiction(f"{rows_path}: {n_recorded} rows record a sampler and {n_legacy} record none -- a store is drawn under ONE sampler")
    rows_sent: Optional[Dict[str, float]] = None
    if seen:
        only = next(iter(seen.values()))
        rows_sent, rows_unmodelled = split_six(only["decode"], f"{rows_path}#decode")
        if rows_unmodelled:
            raise DecodeEvidenceShape(f"{rows_path}#decode carries key(s) outside the sampler set {sorted(rows_unmodelled)}")
        if rows_sent != sent:
            raise DecodeEvidenceContradiction(f"{man}#decode states {canonical(sent)} while the {n_recorded} recorded {BON_RESPONSES} row(s) were drawn "
                                              f"under {canonical(rows_sent)} -- one store, one sampler")
        if only.get("bon_decode_preset") != doc.get("bon_decode_preset"):
            raise DecodeEvidenceContradiction(f"{man}#bon_decode_preset {doc.get('bon_decode_preset')!r} vs the rows' {only.get('bon_decode_preset')!r}")
        if only.get("decode_policy_id") not in (None, ev.producer_policy_id):
            raise DecodeEvidenceContradiction(f"{man}#decode_policy_id {ev.producer_policy_id!r} vs the rows' {only.get('decode_policy_id')!r}")
    ev.sent = sent
    ev.source = f"{man}#decode"
    ev.witnesses.update({"rows_scanned": n_scanned, "rows_recorded": n_recorded, "rows_legacy": n_legacy, "distinct_row_blocks": len(seen),
                         "rows_decode": rows_sent,
                         "summary": (f"{n_recorded}/{n_scanned} {BON_RESPONSES} rows carry the same decode block" if n_scanned
                                     else f"no {BON_RESPONSES} row (an all-no-fit store); the manifest states the sampler")})
    return ev


def from_multiturn_store(store: str, shards: Optional[Sequence[str]] = None) -> Evidence:
    """A multi-turn attack store: every inspected shard's ``progress.jsonl`` rows' ``target_decode`` (one distinct block, else refuse),
    corroborated by each shard's ``served_target.json#decode``; ``attacker_engine.json#target_decode.source`` is the producer's own
    policy statement."""
    ev = Evidence(family="multiturn")
    dirs = _shard_dirs(store, shards, "progress.jsonl")
    if not dirs:
        ev.why = f"{store}: no shard directory holds progress.jsonl"; return ev
    blocks: Dict[str, int] = {}
    first_sent: Optional[Dict[str, float]] = None
    n_with = n_without = 0
    st_agree: Optional[bool] = None
    st_unmodelled: Dict[str, object] = {}
    sources: set = set()
    for d in dirs:
        sh = os.path.basename(d)
        for r in _rows(os.path.join(d, "progress.jsonl"), where=f"{sh}/progress.jsonl"):
            td = r.get("target_decode")
            if td is None:
                n_without += 1; continue
            sent, _unm = split_six(td, f"{sh}/progress.jsonl#target_decode")
            key = canonical(sent); blocks[key] = blocks.get(key, 0) + 1; n_with += 1
            if first_sent is None:
                first_sent = sent
    if len(blocks) > 1:
        raise DecodeEvidenceContradiction(f"{store}: the progress rows state {len(blocks)} distinct target_decode blocks "
                                          f"({', '.join(f'{k}x{n}' for k, n in sorted(blocks.items()))}): one store, one sampler")
    if first_sent is None:
        ev.why = (f"{store}: no progress row states target_decode across {len(dirs)} shard(s) ({n_without} row(s)): a legacy unseeded multi-turn "
                  f"store, the server's generation config governed")
        return ev
    for d in dirs:
        sh = os.path.basename(d)
        stp = os.path.join(d, "served_target.json")
        if os.path.isfile(stp):
            st = _load_json(stp)
            for field in ("decode", "target_decode"):
                blk = st.get(field)
                if blk is None:
                    continue
                s_sent, s_unm = split_six(blk, f"{sh}/served_target.json#{field}")
                if s_sent != first_sent:
                    raise DecodeEvidenceContradiction(f"{store}: {sh}/served_target.json#{field} states {canonical(s_sent)} while the progress rows' "
                                                      f"target_decode is {canonical(first_sent)}")
                st_agree = True
                if field == "decode":
                    st_unmodelled = s_unm
        aep = os.path.join(d, "attacker_engine.json")
        if os.path.isfile(aep):
            ae = _load_json(aep)
            td = ae.get("target_decode") if isinstance(ae.get("target_decode"), dict) else {}
            if isinstance(td.get("source"), str):
                sources.add(td["source"])
    ev.sent = first_sent
    ev.evidence_path = os.path.join(dirs[0], "progress.jsonl")
    ev.sha256 = sha256_file(ev.evidence_path)
    ev.source = f"{ev.evidence_path}#target_decode"
    if st_unmodelled:
        ev.unmodelled[f"{os.path.basename(dirs[0])}/served_target.json#decode"] = st_unmodelled
    if len(sources) == 1:
        ev.producer_policy_id = next(iter(sources)); ev.producer_policy_id_source = f"{os.path.basename(dirs[0])}/attacker_engine.json#target_decode.source"
    ev.witnesses.update({"shards": [os.path.basename(d) for d in dirs], "rows_with_target_decode": n_with, "rows_without_target_decode": n_without,
                         "distinct_blocks": len(blocks), "served_target_decode_agrees": st_agree, "attacker_engine_sources": sorted(sources),
                         "summary": (f"{n_with}/{n_with + n_without} progress rows, {len(dirs)} present shard(s), 1 distinct block"
                                     + ("; served_target.json#decode agrees" if st_agree else ""))})
    return ev


def from_static_store(store: str, shards: Optional[Sequence[str]] = None, family: str = "static") -> Evidence:
    """A sampled static (or padded static) store: every shard's ``served_target.json#decode`` (one block across the shards, else refuse) and
    every ``responses.jsonl`` row's ``decode`` (each row's six-key subset equal to its shard's, else refuse)."""
    ev = Evidence(family=family)
    dirs = _shard_dirs(store, shards, "served_target.json")
    if not dirs:
        ev.why = f"{store}: no shard directory holds served_target.json"; return ev
    st_sent: Optional[Dict[str, float]] = None
    st_blocks: set = set()
    unmodelled: Dict[str, object] = {}
    n_rows = n_rows_without = 0
    row_blocks: set = set()
    absent: List[str] = []
    for d in dirs:
        sh = os.path.basename(d)
        st = _load_json(os.path.join(d, "served_target.json"))
        blk = st.get("decode")
        if blk is None:
            absent.append(sh); continue
        s_sent, s_unm = split_six(blk, f"{sh}/served_target.json#decode")
        st_blocks.add(canonical(s_sent))
        if st_sent is None:
            st_sent, unmodelled = s_sent, s_unm
    if absent and st_sent is not None:
        raise DecodeEvidenceContradiction(f"{store}: served_target.json states a decode block in {len(dirs) - len(absent)} shard(s) and none in {', '.join(absent)}")
    if st_sent is None:
        ev.why = f"{store}: no shard's served_target.json states a decode block ({len(dirs)} shard(s)): a greedy or legacy static store"; return ev
    if len(st_blocks) > 1:
        raise DecodeEvidenceContradiction(f"{store}: served_target.json#decode differs across the shards: {sorted(st_blocks)}")
    for d in dirs:
        sh = os.path.basename(d)
        rp = os.path.join(d, "responses.jsonl")
        if not os.path.isfile(rp):
            continue
        bad = 0; bad_block = None
        for r in _rows(rp, where=f"{sh}/responses.jsonl"):
            n_rows += 1
            dec = r.get("decode")
            if dec is None:
                n_rows_without += 1; continue
            r_sent, _unm = split_six(dec, f"{sh}/responses.jsonl#decode")
            row_blocks.add(canonical(r_sent))
            if r_sent != st_sent:
                bad += 1; bad_block = bad_block or r_sent
        if bad:
            raise DecodeEvidenceContradiction(f"{store}: {bad} response row(s) in {sh} were sent {canonical(bad_block)} while served_target.json states {canonical(st_sent)}")
    if n_rows_without and n_rows_without != n_rows:
        raise DecodeEvidenceContradiction(f"{store}: {n_rows - n_rows_without} response row(s) record a decode block and {n_rows_without} record none -- "
                                          f"a store is drawn under ONE sampler")
    ev.sent = st_sent
    ev.evidence_path = os.path.join(dirs[0], "served_target.json")
    ev.sha256 = sha256_file(ev.evidence_path)
    ev.source = f"{ev.evidence_path}#decode"
    if unmodelled:
        ev.unmodelled[f"{os.path.basename(dirs[0])}/served_target.json#decode"] = unmodelled
    ev.witnesses.update({"shards": len(dirs), "shard_names": [os.path.basename(d) for d in dirs], "served_target_blocks_distinct": len(st_blocks),
                         "response_rows": n_rows, "response_rows_without_decode": n_rows_without, "distinct_row_blocks": len(row_blocks),
                         "summary": f"{len(dirs)} shard(s) agree; {n_rows - n_rows_without}/{n_rows} responses.jsonl rows carry the same decode block"})
    return ev


def from_store_manifest(spec: Mapping[str, Any], spelling: Optional[str], *, store_dir: Optional[str] = None,
                       gen_stats: Optional[Mapping[str, Any]] = None, why_no_spelling: Optional[str] = None) -> Evidence:
    """A served-controls cell manifest (the store recorder's ``self.spec`` at ``on_manifest``; the closer's ``<cell>/manifest.json``),
    read for ONE eval -- ``spelling`` is the producer spelling of the route's eval (``xstest`` / ``or_bench_hard_1k`` / ``mmlu`` / an
    atom) -- never for what the shared manifest happens to carry:

    * FIRST the manifest's top-level ``request_samplers[<spelling>]`` (the driver's own statement of the request it sends, present on
      every current manifest of the served driver); for a sampled control under a SAMPLED ``gen`` it must equal
      ``resolved_config.gen``'s six-key subset, and under a GREEDY ``gen`` (``do_sample`` false, the ``--greedy`` opt-in) it must be
      :data:`GREEDY_CHAT_SAMPLER` (the greedy body's one key), else a contradiction.
    * ELSE ``resolved_config.gen`` ONLY when the spelling is a sampled over-refusal control and ``gen.do_sample`` is true.
    * ELSE nothing (``mmlu`` / ``gsm8k`` / an atom on a legacy manifest; a sampled control under ``--greedy`` on a LEGACY manifest, which
      states no request; a thinking_card cell with no recorder document yet) -- the ``unverified`` entry names why.

    ``gen_stats[<spelling>].decode`` (``<cell>/gen_stats.json``, written at the end of the run) corroborates the sent set when it exists --
    its six-key subset must agree, else a contradiction; the GREEDY block (``do_sample`` false, no sampler key) corroborates exactly
    :data:`GREEDY_CHAT_SAMPLER`; absent at manifest time it is the completion patch's and the backfill's witness."""
    ev = Evidence(family="controls")
    man_path = os.path.join(store_dir, "manifest.json") if store_dir else None
    rc = spec.get("resolved_config") if isinstance(spec.get("resolved_config"), Mapping) else {}
    gen = rc.get("gen") if isinstance(rc.get("gen"), Mapping) else None
    controls = spec.get("controls") if isinstance(spec.get("controls"), list) else None
    rs = spec.get(MANIFEST_KEY_REQUEST_SAMPLERS) if isinstance(spec.get(MANIFEST_KEY_REQUEST_SAMPLERS), Mapping) else None
    ev.witnesses.update({"manifest_controls": list(controls) if controls is not None else None, "spelling": spelling,
                         "gen_do_sample": (bool(gen.get("do_sample")) if gen is not None else None),
                         "max_new_tokens": (gen.get("max_new_tokens") if gen is not None else None),
                         "escalated_from_cap": ((gen.get("escalated_from") or {}).get("cap") if gen is not None and isinstance(gen.get("escalated_from"), Mapping) else None),
                         "request_samplers_present": rs is not None})
    if man_path and os.path.isfile(man_path):
        ev.evidence_path = man_path; ev.sha256 = sha256_file(man_path)
    where_man = man_path or "manifest"
    if spelling is None:
        ev.why = why_no_spelling or (f"the manifest names {len(controls)} controls ({', '.join(map(str, controls))}) under one identity: which request this run "
                                     f"sent is not stated" if controls else "the manifest names no controls list and the route names no task: which request this run sent is not stated")
        return ev
    sent: Dict[str, float] = {}
    unmodelled: Dict[str, object] = {}
    gen_sent: Optional[Dict[str, float]] = None
    if gen is not None and spelling in SAMPLED_CONTROL_SPELLINGS and gen.get("do_sample"):
        gen_sent, gen_unmodelled = split_six(gen, f"{where_man}#resolved_config.gen")
    if rs is not None and spelling in rs:
        sent, unmodelled = split_six(rs[spelling] if rs[spelling] is not None else {}, f"{where_man}#{MANIFEST_KEY_REQUEST_SAMPLERS}.{spelling}")
        if gen_sent is not None and gen_sent != sent:
            raise DecodeEvidenceContradiction(f"{where_man}: {spelling}'s request sampler is stated twice and differently: {MANIFEST_KEY_REQUEST_SAMPLERS}."
                                              f"{spelling} {canonical(sent)} vs resolved_config.gen {canonical(gen_sent)}")
        if gen is not None and spelling in SAMPLED_CONTROL_SPELLINGS and not gen.get("do_sample") and sent and sent != GREEDY_CHAT_SAMPLER:
            raise DecodeEvidenceContradiction(f"{where_man}: {spelling}'s request sampler is stated twice and differently: {MANIFEST_KEY_REQUEST_SAMPLERS}."
                                              f"{spelling} {canonical(sent)} vs resolved_config.gen do_sample false (the --greedy opt-in: the greedy body sends "
                                              f"{canonical(GREEDY_CHAT_SAMPLER)})")
        if not sent:
            ev.why = f"{where_man}#{MANIFEST_KEY_REQUEST_SAMPLERS}.{spelling} states no sampler key: the request names none, the server default applies"
            return ev
        ev.source = f"{where_man}#{MANIFEST_KEY_REQUEST_SAMPLERS}.{spelling}"
    elif gen_sent is not None:
        sent, unmodelled = gen_sent, gen_unmodelled
        if not sent:
            ev.why = f"{where_man}#resolved_config.gen has do_sample true but no sampler key"; return ev
        ev.source = f"{where_man}#resolved_config.gen"
    else:
        if gen is None:
            ev.why = f"{where_man} carries no resolved_config.gen and no {MANIFEST_KEY_REQUEST_SAMPLERS}.{spelling}: the {spelling} request's sampler is not stated"
        elif spelling in SAMPLED_CONTROL_SPELLINGS:
            ev.why = (f"{where_man}#resolved_config.gen has do_sample false (the --greedy opt-in): the {spelling} request names no sampler key, the "
                      f"server default applies")
        else:
            ev.why = (f"the {spelling} request's sampler is built by the driver, not by resolved_config.gen, and this manifest states no "
                      f"{MANIFEST_KEY_REQUEST_SAMPLERS} block (a producer before the decode-recording lane)")
        return ev
    # gen_stats.json[<spelling>].decode: the end-of-run corroboration
    gs = gen_stats
    gs_path = os.path.join(store_dir, "gen_stats.json") if store_dir else None
    if gs is None and gs_path and os.path.isfile(gs_path):
        gs = _load_json(gs_path)
    gs_agrees: Optional[bool] = None
    if isinstance(gs, Mapping) and isinstance(gs.get(spelling), Mapping) and isinstance(gs[spelling].get("decode"), Mapping):
        g_sent, g_unm = split_six(gs[spelling]["decode"], f"gen_stats.json#{spelling}.decode", require_temperature=False)
        if not g_sent and g_unm.get("do_sample") is False:
            # the GREEDY block ({do_sample: false, cap, arm}): the request it describes is the greedy body, GREEDY_CHAT_SAMPLER's one key
            if sent != GREEDY_CHAT_SAMPLER:
                raise DecodeEvidenceContradiction(f"{where_man}: {spelling}'s request sampler is stated twice and differently: {ev.source.split('#', 1)[1]} "
                                                  f"{canonical(sent)} vs gen_stats.json#{spelling}.decode the greedy block (do_sample false: the greedy body sends "
                                                  f"{canonical(GREEDY_CHAT_SAMPLER)})")
        elif g_sent != sent:
            raise DecodeEvidenceContradiction(f"{where_man}: {spelling}'s request sampler is stated twice and differently: {ev.source.split('#', 1)[1]} "
                                              f"{canonical(sent)} vs gen_stats.json#{spelling}.decode {canonical(g_sent)}")
        gs_agrees = True
    ev.sent = sent
    if unmodelled:
        ev.unmodelled[ev.source.split("#", 1)[1]] = unmodelled
    ev.witnesses.update({"gen_stats_decode_agrees": gs_agrees,
                         "summary": f"spelling {spelling}" + ("; gen_stats.json decode agrees" if gs_agrees else "")})
    return ev


def from_misalign_store(store: str) -> Evidence:
    """The misalign runner's agentic cell: ``meta.json#sampling`` (the ``SAMPLING_KEYS`` subset of the payload), corroborated by
    ``params.sampling``; ``sampler`` and ``finished`` are witnesses."""
    ev = Evidence(family="misalign")
    mp = os.path.join(store, "meta.json")
    if not os.path.isfile(mp):
        ev.why = f"{store}: no meta.json"; return ev
    meta = _load_json(mp)
    ev.evidence_path = mp; ev.sha256 = sha256_file(mp)
    ev.producer_policy_id = meta.get("decode_policy_id") if isinstance(meta.get("decode_policy_id"), str) else None
    ev.producer_policy_id_source = "meta.json#decode_policy_id" if ev.producer_policy_id else None
    params = meta.get("params") if isinstance(meta.get("params"), Mapping) else {}
    ev.witnesses.update({"sampler": meta.get("sampler"), "finished": bool(meta.get("finished"))})
    block = meta.get("sampling")
    if block is None:
        ev.why = f"{mp}#sampling absent: the runner recorded no sampling block (a legacy cell)"; return ev
    sent, unmodelled = split_six(block, f"{mp}#sampling")
    if not sent:
        ev.why = f"{mp}#sampling is empty"; return ev
    agree: Optional[bool] = None
    if params.get("sampling") is not None:
        p_sent, _u = split_six(params["sampling"], f"{mp}#params.sampling")
        if p_sent != sent:
            raise DecodeEvidenceContradiction(f"{mp}: sampling {canonical(sent)} vs params.sampling {canonical(p_sent)} -- the runner disagrees with itself")
        agree = True
    ev.sent = sent; ev.source = f"{mp}#sampling"
    if unmodelled:
        ev.unmodelled["meta.json#sampling"] = unmodelled
    ev.witnesses.update({"params_sampling_agrees": agree, "summary": f"sampler {meta.get('sampler')!r}" + ("; params.sampling agrees" if agree else "")})
    return ev


def recorder_path_for(cell_dir: str) -> str:
    """``<store root>/resolved_sampler.json``, two levels above the cell dir (``<root>/cells/<cell>``) -- the cellstore adapter's rule."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cell_dir))), RECORDER_FILE)


def _derive_from_reference(ev: Evidence, where: str, ref_src: str, n_rows: int, n: object, code_sha: Optional[str], task: Optional[str] = None) -> Evidence:
    """A recorder's DECLARED reference as the sent set (the owner's Change 1, 2026-10-13 18:57 PT): ``derived`` against the pinned request
    builder at the run's code sha, never ``recorded`` -- the reference states the keys the client BUILDS its requests from, and with no
    per-request row it cannot show which keys the client omitted (a 0 equals the engine's default). Without a forty-hex code sha the set
    is NOT filed: the derivation would name no pin -- the declared keys then ride the witnesses (``unfiled_declared_reference``, the
    cellstore adapter's shape) and the server-resolved sampler the caller read stays filed apart, ``recorded``. ``ev.sent`` /
    ``ev.source`` are set by the caller; this fills the confidence, the basis and the witnesses, or the refusal."""
    if not _is_code_sha(code_sha):
        ev.witnesses["unfiled_declared_reference"] = {"sent": dict(ev.sent), "reference_source": ref_src, "sent_rows": n_rows,
                                                      "request_builder": RECORDER_REQUEST_BUILDER}
        ev.sent = {}; ev.source = None
        ev.why = (f"{where}: the recorder's {ref_src} reference is a DECLARATION (sent_rows {n_rows}: the client recorded no per-request "
                  f"sampler); the sent set derives from it only against the pinned request builder, and this read names no code sha "
                  f"({code_sha!r}) -- nothing filed for the sent columns; the server-resolved sampler is recorded apart")
        return ev
    ev.sent_confidence = SENT_DERIVED
    ev.derivation = (f"DERIVED, not observed per request: the keys and values are the pinned request builder's -- {RECORDER_REQUEST_BUILDER} "
                     f"at code sha {code_sha} -- which the recorder's {ref_src} reference declares (sent_rows {n_rows}: no per-request record "
                     f"of the client's sampler; a key resolving to the engine's default cannot be told from one the client omitted); the "
                     f"server-resolved values are recorded apart")
    who = f"task {task}: " if task else ""
    ev.witnesses.update({"reference_source": ref_src, "sent_rows": n_rows, "request_builder": RECORDER_REQUEST_BUILDER, "request_builder_code_sha": code_sha,
                         "summary": f"{who}the producer's {ref_src} reference (the keys the client sends: {sorted(ev.sent)}), each resolved to its value "
                                    f"on every request{' of the task' if task else ''} (verdict AGREES, n={n})"})
    return ev


def from_recorder_document(cell_dir: str, *, path: Optional[str] = None, task: Optional[str] = None, code_sha: Optional[str] = None) -> Evidence:
    """The store-level request recorder (27b, the cellstore adapter's rule): SENT = the client rows' one recorded sampler when the driver
    recorded what it sent (``recorded``), else the producer's ``--expect`` reference (the keys the client sends) with every key resolved to
    its value (``derived`` against the pinned request builder at ``code_sha`` -- the owner's Change 1; no code sha, no sent column);
    RESOLVED = the client class's one server-resolved sampler (``recorded``) -- both only under the ``AGREES`` verdict, and the resolved
    sampler is filed whenever it was read, with the sent set or alone (:meth:`Evidence.resolved_read`). The resolved sampler's other
    keys (the engine's defaults) are NEVER filed as sent.

    ``task`` is the run's own task (the route's producer spelling). When the document states PER-TASK references
    (``classes.client.tasks``, the recorder's ``--expect-tasks`` shape) the evidence is that task's block alone -- a document that names
    tasks and is read for none, or for a task it does not name, is unverified by name, never the class-level answer (one store's
    recorder must not hand one task another task's sampler)."""
    ev = Evidence(family="recorder")
    p = path or recorder_path_for(cell_dir)
    if not os.path.isfile(p):
        ev.why = f"no recorder document ({RECORDER_FILE} absent at {p})"; return ev
    doc = _load_json(p)
    ev.evidence_path = p; ev.sha256 = sha256_file(p)
    verdict = str(doc.get("verdict") or "")
    client = (doc.get("classes") or {}).get("client") if isinstance(doc.get("classes"), Mapping) else None
    ev.witnesses["verdict"] = verdict or None
    ev.witnesses["task"] = task
    if verdict != "AGREES":
        ev.why = f"{p}: recorder verdict {verdict or 'absent'} (only AGREES fills the decode parameters)"; return ev
    if not isinstance(client, Mapping):
        ev.why = f"{p}: recorder verdict AGREES but no client class recorded"; return ev
    sent_block = doc.get("sent") if isinstance(doc.get("sent"), Mapping) else {}
    rows_distinct = [d for d in (sent_block.get("distinct") or []) if isinstance(d, Mapping) and isinstance(d.get("sampler"), Mapping)]
    n_rows = int(sent_block.get("n_rows") or 0)
    ev.witnesses["sent_rows"] = n_rows
    tasks = client.get("tasks") if isinstance(client.get("tasks"), Mapping) else None
    if tasks:
        ev.witnesses["recorder_tasks"] = sorted(str(t) for t in tasks)
        if task is None:
            ev.why = f"{p}: the recorder states per-task references ({sorted(tasks)}); this run's task is not named, so none of them is its"; return ev
        if task not in tasks or not isinstance(tasks[task], Mapping):
            ev.why = f"{p}: the recorder's client class covers {sorted(tasks)}, not {task!r}: no request of this run is described"; return ev
        return _from_recorder_task_block(ev, p, task, tasks[task], code_sha, n_rows)
    distinct = [d for d in (client.get("distinct_resolved") or []) if isinstance(d, Mapping) and isinstance(d.get("sampler"), Mapping)]
    if len(distinct) != 1:
        ev.why = f"{p}: recorder verdict AGREES but {len(distinct)} distinct client sampler(s) (exactly one fills the decode parameters)"; return ev
    resolved, _r_unm = split_six(distinct[0]["sampler"], f"{p}#classes.client.distinct_resolved[0].sampler")
    n = client.get("n")
    ev.resolved = resolved
    ev.resolved_source = f"{p}#classes.client.distinct_resolved[0].sampler; verdict AGREES (n={n})"
    ev.witnesses.update({"client_requests": n})
    if rows_distinct and n_rows > 0:
        if len(rows_distinct) != 1:
            ev.why = f"{p}: the client rows record {len(rows_distinct)} distinct sent samplers (exactly one fills the sent parameters)"; return ev
        sent, _u = split_six(rows_distinct[0]["sampler"], f"{p}#sent.distinct[0].sampler")
        ev.sent = sent; ev.source = f"{p}#sent.distinct[0].sampler"
        ev.witnesses.update({"reference_source": "client rows", "summary": f"the client rows' recorded sampler (n_rows={n_rows}); verdict AGREES"})
        return ev
    ref = client.get("reference") if isinstance(client.get("reference"), Mapping) else (doc.get("reference") if isinstance(doc.get("reference"), Mapping) else None)
    ref_src = str(client.get("reference_source") or doc.get("reference_source") or "")
    if ref and ref_src:
        sent, _u = split_six(ref, f"{p}#classes.client.reference")
        if not sent:
            ev.why = f"{p}: the recorder's reference names no decode key"; return ev
        disagree = [k for k, v in sent.items() if k not in resolved or float(resolved[k]) != float(v)]
        if disagree:
            ev.why = f"{p}: the recorder's reference keys {disagree} did not resolve to their values despite the AGREES verdict"; return ev
        ev.sent = sent; ev.source = f"{p}#classes.client.reference"
        return _derive_from_reference(ev, f"{p}#classes.client.reference", ref_src, n_rows, n, code_sha)
    ev.why = f"{p}: recorder verdict AGREES but neither the client rows nor a reference states the keys the client sent"
    return ev


def _from_recorder_task_block(ev: Evidence, p: str, task: str, tb: Mapping[str, Any], code_sha: Optional[str] = None, n_rows: int = 0) -> Evidence:
    """One task's block of a per-task recorder document (``classes.client.tasks[<task>]``): its own verdict, its one resolved sampler, its
    reference (the keys that task's client sends) with every key resolved to its value -- a declaration (the document's client rows, if
    any, are not attributed per task), so ``derived`` against the pinned request builder at ``code_sha`` (no code sha, no sent column;
    the task's resolved sampler is filed apart)."""
    where = f"{p}#classes.client.tasks[{task}]"
    verdict = str(tb.get("verdict") or "")
    if verdict != "AGREES":
        ev.why = f"{where}: verdict {verdict or 'absent'} (only AGREES fills the decode parameters)"; return ev
    distinct = [d for d in (tb.get("distinct_resolved") or []) if isinstance(d, Mapping) and isinstance(d.get("sampler"), Mapping)]
    if len(distinct) != 1:
        ev.why = f"{where}: verdict AGREES but {len(distinct)} distinct resolved sampler(s) for the task (exactly one fills the decode parameters)"; return ev
    resolved, _r_unm = split_six(distinct[0]["sampler"], f"{where}.distinct_resolved[0].sampler")
    n = tb.get("n")
    ev.resolved = resolved
    ev.resolved_source = f"{where}.distinct_resolved[0].sampler; verdict AGREES (n={n})"
    ev.witnesses.update({"client_requests": n})
    ref = tb.get("reference") if isinstance(tb.get("reference"), Mapping) else None
    ref_src = str(tb.get("reference_source") or "")
    if not ref:
        ev.why = f"{where}: verdict AGREES but the task's reference names no sampler (the keys its client sends are not stated)"; return ev
    sent, _u = split_six(ref, f"{where}.reference")
    if not sent:
        ev.why = f"{where}: the task's reference names no decode key"; return ev
    disagree = [k for k, v in sent.items() if k not in resolved or float(resolved[k]) != float(v)]
    if disagree:
        ev.why = f"{where}: the reference keys {disagree} did not resolve to their values despite the AGREES verdict"; return ev
    ev.sent = sent; ev.source = f"{where}.reference"
    return _derive_from_reference(ev, f"{where}.reference", ref_src or "--expect-tasks", n_rows, n, code_sha, task=task)


def for_family(fam: str, store: str, *, spelling: Optional[str] = None, shards: Optional[Sequence[str]] = None,
               artifact: Optional[str] = None, code_sha: Optional[str] = None) -> Evidence:
    """The family's reader over its store (the closer's ``fam`` / ``store`` / ``artifact``; the backfill's registry row). ``code_sha`` is
    the run's own (the registry row's / the producer block's): the recorder family's declared reference derives against it."""
    if fam == "bon":
        return from_bon_store(store)
    if fam == "multiturn":
        return from_multiturn_store(store, shards)
    if fam in ("static", "padding"):
        return from_static_store(store, shards, family=fam)
    if fam == "controls":
        man = os.path.join(store, "manifest.json")
        spec = _load_json(man) if os.path.isfile(man) else {}
        ev = from_store_manifest(spec, spelling, store_dir=store, why_no_spelling=None if spelling else f"no producer spelling for the control (store {store})")
        if not os.path.isfile(man) and ev.why is None:
            ev.why = f"{man}: no manifest.json"
        return ev
    if fam == "misalign":
        return from_misalign_store(store)
    if fam == "recorder":
        return from_recorder_document(store, task=spelling, code_sha=code_sha)
    return Evidence(family=str(fam), why=f"no decode evidence reader for family {fam!r}")


# ------------------------------------------------------------------------------------------------------------------------------
# the remote read pass and the write-time verification (the backfill's)
# ------------------------------------------------------------------------------------------------------------------------------

#: the per-run fields a frozen manifest line pins and the write pre-pass re-reads (one mismatch refuses the whole write); the sent
#: columns' confidence and its basis are pinned too -- client rows landing after the freeze would turn a derived set into a recorded one
VERIFY_FIELDS: Tuple[str, ...] = ("sha256", "sent", "resolved", "unmodelled", "producer_policy_id", "source", "evidence_path", "why",
                                  "sent_confidence", "derivation")


def read_run(req: Mapping[str, Any]) -> Dict[str, Any]:
    """One run's remote read: ``{run_id, family, store_uri, spelling?, shards?, code_sha?}`` -> the evidence as JSON plus a SECOND sha256
    read of the evidence file; a refusal is reported by kind (``contradiction`` / ``shape``) with its message, never raised."""
    rid = str(req.get("run_id")); fam = str(req.get("family")); store = str(req.get("store_uri"))
    out: Dict[str, Any] = {"run_id": rid, "family": fam, "store_uri": store, "read_utc": utc_now()}
    try:
        ev = for_family(fam, store, spelling=req.get("spelling"), shards=req.get("shards"), code_sha=req.get("code_sha"))
    except DecodeEvidenceContradiction as exc:
        out.update({"ok": False, "kind": "contradiction", "error": str(exc)}); return out
    except DecodeEvidenceShape as exc:
        out.update({"ok": False, "kind": "shape", "error": str(exc)}); return out
    except Exception as exc:  # noqa: BLE001 -- the read pass reports, the caller decides
        out.update({"ok": False, "kind": type(exc).__name__, "error": str(exc)}); return out
    out.update({"ok": True, "evidence": ev.to_json()})
    if ev.evidence_path and os.path.isfile(ev.evidence_path):
        out["sha256_read2"] = sha256_file(ev.evidence_path)
        out["evidence_bytes"] = os.path.getsize(ev.evidence_path)
    return out


def remote_main(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"reader": READER, "read_utc": utc_now(), "host": os.uname().nodename if hasattr(os, "uname") else None,
            "n": len(runs), "results": [read_run(r) for r in runs]}


def _witness_counts(w: Mapping[str, Any]) -> Dict[str, Any]:
    """The witness entries a verification compares: every number, bool, string and list (the ``summary`` prose excluded)."""
    return {k: v for k, v in w.items() if k != "summary" and isinstance(v, (int, float, bool, str, list, dict, type(None)))}


def verify_against(frozen: Mapping[str, Any], fresh: Mapping[str, Any]) -> List[str]:
    """The drift between a FROZEN evidence record (a manifest line of the backfill) and a fresh re-read of the same run: every
    :data:`VERIFY_FIELDS` member, the second sha256 read, and every witness count. ``[]`` = equal."""
    drifts: List[str] = []
    fe = frozen.get("evidence") if isinstance(frozen.get("evidence"), Mapping) else frozen
    ge = fresh.get("evidence") if isinstance(fresh.get("evidence"), Mapping) else fresh
    if bool(frozen.get("ok", True)) != bool(fresh.get("ok", True)):
        return [f"ok {frozen.get('ok', True)} -> {fresh.get('ok', True)}: {fresh.get('error')}"]
    for k in VERIFY_FIELDS:
        if canonical(fe.get(k)) != canonical(ge.get(k)):
            drifts.append(f"{k}: {canonical(fe.get(k))[:120]} -> {canonical(ge.get(k))[:120]}")
    for k in ("sha256_read2",):
        if frozen.get(k) is not None and frozen.get(k) != fresh.get(k):
            drifts.append(f"{k}: {frozen.get(k)} -> {fresh.get(k)}")
    fw, gw = _witness_counts(fe.get("witnesses") or {}), _witness_counts(ge.get("witnesses") or {})
    for k in sorted(set(fw) | set(gw)):
        if canonical(fw.get(k)) != canonical(gw.get(k)):
            drifts.append(f"witness {k}: {canonical(fw.get(k))[:120]} -> {canonical(gw.get(k))[:120]}")
    return drifts


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="read the request's observed sampler from store evidence (stdlib only)")
    ap.add_argument("--runs", required=True, help="a JSON file (or '-' for stdin) holding a list of {run_id, family, store_uri, spelling?, shards?}")
    ap.add_argument("--json", action="store_true", help="print the results as one JSON document")
    a = ap.parse_args(argv)
    raw = sys.stdin.read() if a.runs == "-" else open(a.runs, encoding="utf-8").read()
    runs = json.loads(raw)
    if not isinstance(runs, list):
        print("--runs must hold a JSON list", file=sys.stderr); return 2
    out = remote_main(runs)
    if a.json:
        print(json.dumps(out, sort_keys=True))
    else:
        for r in out["results"]:
            print(f"{r['run_id']}\t{r['family']}\t{'ok' if r.get('ok') else 'REFUSED ' + str(r.get('kind'))}\t"
                  f"{canonical((r.get('evidence') or {}).get('sent')) if r.get('ok') else r.get('error')}")
    return 0


#: the backfill ships this module's SOURCE on stdin (`DECODE_EVIDENCE_LIBRARY=1 python3 -I -`, a trailer calling :func:`remote_main`):
#: under that switch the CLI does not run, so the trailer's own invocation is the only one
LIBRARY_ENV = "DECODE_EVIDENCE_LIBRARY"

if __name__ == "__main__" and os.environ.get(LIBRARY_ENV) != "1":
    sys.exit(main())
