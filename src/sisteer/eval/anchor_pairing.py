"""The passthrough-anchor pairing allowance.

Every aggregator in this repo pairs a steered row against its unsteered ``none`` anchor only WITHIN one serving engine:
the lens (a vLLM plugin hook, eager decode) and the native multi-LoRA engine (a PEFT adapter at every position, CUDA
graphs on) are different engines, and a delta across them would measure the engine as well as the intervention
(``contracts.read_store_reduction_mode``, ``sweep.assert_pairable``, ``tinycap_gsm8k_paired200._comparable``). The
2026-09-15 user ruling reuses the EXISTING lens-era ``none`` anchors for the new natively served ReLU-steering (PSR) rows
instead of drawing new anchors, on one argument: an UNSTEERED PASSTHROUGH leg runs the same base model through the same
sampler on either engine, and the graphs-vs-eager numerics of an unsteered forward are far below sampling noise at
T >= 0.6. That is a decision, not a fact the records carry, so it is written HERE as an explicit, tested rule and STAMPED
on every paired row it admits (``pairing_allowance`` = :data:`PASSTHROUGH_ANCHOR_ALLOWANCE`, ``anchor_engine``,
``anchor_view``) — never a silent cross-engine join, and filterable by the stamp if the decision is ever reversed.

THE RULE (:func:`anchor_pairing_verdict`):

* same engine on both sides -> admitted, ``allowance`` None (the pairing every aggregator already makes);
* different engines -> admitted ONLY when the anchor row is an unsteered passthrough leg (cell ``none``, a
  ``proxy_vllm+passthrough+…`` serving identity, no object sha) — on ANY engine — and the steered row is served by the
  NATIVE engine (``serving_engine == "lora_native"``, a ``proxy_vllm+lora+…`` identity); the verdict then carries the
  allowance stamp;
* a steered LENS row against a native row, a steered anchor, a passthrough anchor that carries an object sha, or a
  ``prompted`` anchor across engines -> REFUSED, exactly as today;
* the chat template stays a pairing axis: two recorded template hashes that differ refuse under every engine
  combination (one recorded and one absent refuses too — an asymmetric record is not evidence of equality).

The GSM8K contrast relaxes the canonical reader's declared-tree-set clauses ONLY where they encode the engine:
:data:`ENGINE_LAUNCH_KEYS` names the vLLM startup-echo keys the two engines' launch composers set differently
(``vllm_server.build_server_launch``: the eager/graph token, the LoRA surface, the CUDA-graph capture surface) and
:data:`ENGINE_RECORDED_KEYS` the recorded server setting that names the lens arithmetic (``reduction_mode``:
``stacked`` on a lens store, ``not_applicable`` on a native one). A launch or recorded difference OUTSIDE these keys is
still a refusal (:func:`engine_only_differences`).

Nothing here reads a store: the adapters (:func:`stamp_from_served_target`, :func:`stamp_from_bon_run`,
:func:`stamp_from_capeval_manifest`, :func:`stamp_from_petri_cell`, :func:`stamp_from_sweep_stamp`,
:func:`stamp_from_gsm8k_leg`) take the RECORD the producer wrote and project it onto one :class:`PairingStamp`; the
verdict is a pure function of two stamps. Torch-free.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from sisteer.capeval import manifest as _cm
from sisteer.capeval.tinycap import contracts as _c

__all__ = [
    "PASSTHROUGH_ANCHOR_ALLOWANCE", "ENGINE_LENS", "ENGINE_NATIVE", "ENGINES", "CELL_NONE",
    "PATH_PASSTHROUGH", "PATH_LORA", "PATH_LENS", "ENGINE_LAUNCH_KEYS", "ENGINE_RECORDED_KEYS",
    "PairingStamp", "Verdict", "resolve_engine", "is_unsteered_passthrough", "is_native_steered",
    "anchor_pairing_verdict", "engine_only_differences",
    "stamp_from_served_target", "stamp_from_bon_run", "stamp_from_capeval_manifest", "stamp_from_petri_cell",
    "stamp_from_sweep_stamp", "stamp_from_gsm8k_leg",
    "PETRI_HARNESS_PREFIX", "petri_harness_cell_id", "petri_roster_cell_id_of_entry", "resolve_petri_entry",
]

#: the allowance stamp every admitted cross-engine paired row carries
PASSTHROUGH_ANCHOR_ALLOWANCE = "passthrough-anchor-engine-agnostic-2026-09-15"

#: the engine vocabulary, BY REFERENCE to the stored-record contract (never re-typed here)
ENGINE_LENS = _c.SERVING_ENGINE_LENS
ENGINE_NATIVE = _c.SERVING_ENGINE_LORA_NATIVE
ENGINES = _c.SERVING_ENGINES
CELL_NONE = "none"
#: the serving PATHS of a ``proxy_vllm`` identity (``manifest.SERVING_PATHS["proxy_vllm"]``)
PATH_LENS, PATH_LORA, PATH_PASSTHROUGH = "lens", "lora", "passthrough"

#: the vLLM startup-echo keys (``launch.<key>`` in the GSM8K reader's normalised server projection) whose difference
#: between a lens-era launch and a native launch IS the engine, per ``vllm_server.build_server_launch``: the lens pins
#: ``--enforce-eager`` and the native mode ``--no-enforce-eager``; the native mode composes the LoRA surface
#: (``--enable-lora --max-loras --max-lora-rank --max-cpu-loras --lora-target-modules --lora-modules``, the adapter
#: dtype / sharding toggles) that a lens passthrough launch never carries; the CUDA-graph capture surface is resolved
#: only with graphs on. ``worker_extension_cls`` / ``middleware`` are already excluded from that projection
#: (``LAUNCH_EXCLUDED_KEYS``). A difference in ANY OTHER launch key (dtype, revision, prefix caching, the sampling
#: override, the render kwargs, ...) is not an engine difference and stays a refusal.
ENGINE_LAUNCH_KEYS: frozenset[str] = frozenset({
    "enforce_eager",
    "enable_lora", "max_loras", "max_lora_rank", "max_cpu_loras", "lora_dtype", "fully_sharded_loras",
    "lora_target_modules", "lora_modules",
    "cudagraph_capture_sizes", "max_cudagraph_capture_size", "cuda_graph_sizes", "compilation_config",
})
#: the RECORDED server settings (the same projection, outside ``launch.*``) that name the engine: the lens reduction
#: mode (``stacked`` / ``per_row`` on a lens store, ``not_applicable`` on a native one — ``contracts.REDUCTION_MODE_FIELD``)
ENGINE_RECORDED_KEYS: frozenset[str] = frozenset({_c.REDUCTION_MODE_FIELD})


@dataclass(frozen=True)
class PairingStamp:
    """One row's pairing-relevant identity, projected from whatever record the producer wrote.

    ``cell`` is the cell id (``none`` for the unsteered anchor); ``serving_identity`` the ``<backend>+<path>+<mode>``
    string; ``serving_engine`` the STATED engine (``lens`` / ``lora_native``) or None when the record carries no stamp
    (a lens-era record — the native writers stamp every route, so absence is lens BY ERA, the contracts' rule);
    ``object_sha`` the served object's digest (None on an unsteered leg); ``chat_template`` the served template hash
    when the record carries one; ``view`` the roster view (``main`` / ``it``; stamped as ``anchor_view`` on the anchor
    side); ``model`` the served base model id when recorded; ``source`` names the record for the messages."""

    cell: str
    serving_identity: Optional[str]
    serving_engine: Optional[str] = None
    object_sha: Optional[str] = None
    chat_template: Optional[str] = None
    view: Optional[str] = None
    model: Optional[str] = None
    source: Optional[str] = None


@dataclass(frozen=True)
class Verdict:
    """The pairing verdict for (steered, anchor). ``admitted`` False carries the refusal in ``reason``; ``allowance`` is
    :data:`PASSTHROUGH_ANCHOR_ALLOWANCE` on an admitted CROSS-engine pair and None on a same-engine one."""

    admitted: bool
    anchor_engine: Optional[str]
    steered_engine: Optional[str]
    allowance: Optional[str]
    anchor_view: Optional[str]
    reason: str

    def as_record(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "anchor_engine": self.anchor_engine, "steered_engine": self.steered_engine,
                "pairing_allowance": self.allowance, "anchor_view": self.anchor_view, "reason": self.reason}


class PairingError(ValueError):
    """A record that cannot be read as either engine (torn, unknown identity) — never read around."""


def _path_of(identity: Optional[str], *, source: str) -> Optional[str]:
    """The serving path of an identity string, or None for an absent identity; a malformed identity refuses."""
    if identity is None:
        return None
    try:
        return _cm.parse_serving_identity(identity).path
    except Exception as e:  # manifest._refuse raises its own error class
        raise PairingError(f"{source}: serving_identity {identity!r} is not a serving identity ({e})") from e


def resolve_engine(stamp: PairingStamp) -> str:
    """The engine a stamp's record was served on: the STATED ``serving_engine`` when present (held to the identity —
    ``lora_native`` beside a ``+lens+`` identity or ``lens`` beside ``+lora+`` is a torn record and refuses,
    ``contracts.SERVING_PATHS_OF_ENGINE``); else BY PATH — ``lora`` is the native engine's own path, ``lens`` and
    ``passthrough`` without a stamp are lens by era (the native writers stamp every route; ``contracts.
    read_store_serving_engine``). A stamp with neither identity nor engine has no engine and refuses."""
    src = stamp.source or "record"
    path = _path_of(stamp.serving_identity, source=src)
    engine = stamp.serving_engine
    if engine is not None:
        if engine not in ENGINES:
            raise PairingError(f"{src}: serving_engine {engine!r} is not one of {list(ENGINES)}")
        if path is not None and path not in _c.SERVING_PATHS_OF_ENGINE[engine]:
            raise PairingError(
                f"{src}: serving_engine {engine!r} beside a {path!r} serving path ({stamp.serving_identity!r}) — a "
                f"{engine!r} store is written under {list(_c.SERVING_PATHS_OF_ENGINE[engine])} only; a torn or "
                "relabelled record, not read as either engine")
        return engine
    if path is None:
        raise PairingError(f"{src}: neither a serving_identity nor a serving_engine — no engine can be read")
    if path == PATH_LORA:
        return ENGINE_NATIVE
    return ENGINE_LENS


def is_unsteered_passthrough(stamp: PairingStamp) -> tuple[bool, str]:
    """Whether a stamp is the UNSTEERED PASSTHROUGH leg the allowance admits as an anchor: cell ``none``, a passthrough
    serving path, no object sha. Returns ``(ok, why_not)``."""
    if stamp.cell != CELL_NONE:
        return False, f"anchor cell is {stamp.cell!r}, not {CELL_NONE!r}"
    path = _path_of(stamp.serving_identity, source=stamp.source or "anchor")
    if path != PATH_PASSTHROUGH:
        return False, f"anchor serving_identity {stamp.serving_identity!r} is not a {PATH_PASSTHROUGH!r} route"
    if stamp.object_sha:
        return False, f"anchor carries an object sha ({str(stamp.object_sha)[:12]}…) — not an unsteered leg"
    return True, ""


def is_native_steered(stamp: PairingStamp) -> tuple[bool, str]:
    """Whether a stamp is a NATIVELY served steered row: a ``lora`` serving path resolved to the native engine."""
    path = _path_of(stamp.serving_identity, source=stamp.source or "steered")
    if path != PATH_LORA:
        return False, f"steered serving_identity {stamp.serving_identity!r} is not a {PATH_LORA!r} route"
    if resolve_engine(stamp) != ENGINE_NATIVE:
        return False, f"steered serving_engine {stamp.serving_engine!r} is not {ENGINE_NATIVE!r}"
    return True, ""


def _template_problem(steered: PairingStamp, anchor: PairingStamp) -> Optional[str]:
    """The chat template is a pairing axis under EVERY engine combination: two recorded hashes that differ refuse; one
    recorded and one absent refuses (an asymmetric record is not equality); both absent is not compared HERE (the
    caller's other axes — the served model pin — carry it) and is named in the verdict's reason."""
    a, b = steered.chat_template, anchor.chat_template
    if a is None and b is None:
        return None
    if a is None or b is None:
        return (f"chat_template recorded on one side only (steered {a!r}, anchor {b!r}) — the template axis cannot be "
                "read as equal")
    if str(a) != str(b):
        return f"chat_template differs: steered {a!r} vs anchor {b!r} — the template stays a pairing axis"
    return None


def _model_problem(steered: PairingStamp, anchor: PairingStamp) -> Optional[str]:
    a, b = steered.model, anchor.model
    if a is not None and b is not None and str(a) != str(b):
        return f"served model differs: steered {a!r} vs anchor {b!r}"
    return None


def anchor_pairing_verdict(steered_stamp: PairingStamp, anchor_stamp: PairingStamp) -> Verdict:
    """THE RULE (module docstring). Pure over two stamps; every refusal names its clause. A record that cannot be read
    as either engine (a torn stamp, a malformed identity) is a refusal too, never an exception the caller must catch."""
    try:
        steered_engine = resolve_engine(steered_stamp)
        anchor_engine = resolve_engine(anchor_stamp)
    except PairingError as e:
        return Verdict(False, None, None, None, anchor_stamp.view, f"refused: {e}")
    for problem in (_template_problem(steered_stamp, anchor_stamp), _model_problem(steered_stamp, anchor_stamp)):
        if problem:
            return Verdict(False, anchor_engine, steered_engine, None, anchor_stamp.view, f"refused: {problem}")
    tmpl_note = ("" if steered_stamp.chat_template is not None
                 else "; chat_template not recorded on either side (the served model pin carries the template axis)")
    if steered_engine == anchor_engine:
        return Verdict(True, anchor_engine, steered_engine, None, anchor_stamp.view,
                       f"admitted: same serving engine ({anchor_engine}){tmpl_note}")
    ok_anchor, why_anchor = is_unsteered_passthrough(anchor_stamp)
    ok_steered, why_steered = is_native_steered(steered_stamp)
    if ok_anchor and ok_steered:
        return Verdict(True, anchor_engine, steered_engine, PASSTHROUGH_ANCHOR_ALLOWANCE, anchor_stamp.view,
                       f"admitted under {PASSTHROUGH_ANCHOR_ALLOWANCE}: an unsteered passthrough {CELL_NONE!r} anchor "
                       f"served by the {anchor_engine!r} engine against a natively served steered row ({steered_engine!r}); "
                       "the anchor is engine-agnostic BY DECISION (spec Fixed decisions / decisions D0), stamped on the row"
                       f"{tmpl_note}")
    why = "; ".join(w for w in (why_anchor, why_steered) if w)
    return Verdict(False, anchor_engine, steered_engine, None, anchor_stamp.view,
                   f"refused: serving engines differ (steered {steered_engine!r} vs anchor {anchor_engine!r}) and the "
                   f"passthrough-anchor allowance does not apply — {why}")


def engine_only_differences(differing_keys: Mapping[str, Any] | list[str] | tuple[str, ...],
                            *, launch_prefix: str = "launch.") -> tuple[list[str], list[str]]:
    """Split a set of differing normalised-server-setting keys into the ENGINE ones the allowance may admit
    (``launch.<key>`` with ``<key>`` in :data:`ENGINE_LAUNCH_KEYS`, or a key in :data:`ENGINE_RECORDED_KEYS`) and the
    rest, which stay a refusal. ``(engine_keys, other_keys)``, each sorted."""
    engine: list[str] = []
    other: list[str] = []
    for k in differing_keys:            # a Mapping iterates its keys
        name = str(k)
        if (name.startswith(launch_prefix) and name[len(launch_prefix):] in ENGINE_LAUNCH_KEYS) or name in ENGINE_RECORDED_KEYS:
            engine.append(name)
        else:
            other.append(name)
    return sorted(engine), sorted(other)


# --------------------------------------------------------------------------- #
# adapters: each record shape a producer writes -> one PairingStamp            #
# --------------------------------------------------------------------------- #


def _m(x: Any) -> Mapping[str, Any]:
    return x if isinstance(x, Mapping) else {}


def stamp_from_served_target(doc: Mapping[str, Any], *, cell: str, view: Optional[str] = None,
                             source: str = "served_target.json") -> PairingStamp:
    """A multi-turn shard's ``served_target.json`` (``sisteer.eval.attacks.served_target.write_target_manifest``):
    ``serving_identity`` (top level, else ``evidence.serving_identity``), the row-facing ``serving_engine`` when the
    writer stamped it (absent on every lens-era shard), the object sha from ``object_artifact_sha256_by_cell`` /
    ``proxy_identity.object_artifact_sha256_by_cell`` under the harness cell id, the served ``base_model``."""
    ident = doc.get("serving_identity") or _m(doc.get("evidence")).get("serving_identity")
    harness = doc.get("harness_cell_id") or doc.get("tt_cell_id") or cell
    by_cell = _m(doc.get("object_artifact_sha256_by_cell")) or _m(_m(doc.get("proxy_identity")).get("object_artifact_sha256_by_cell"))
    sha = by_cell.get(str(harness)) or doc.get("object_artifact_sha256")
    return PairingStamp(cell=str(doc.get("tt_cell_id") or cell), serving_identity=ident, serving_engine=doc.get(_c.SERVED_ENGINE_FIELD),
                        object_sha=sha, chat_template=doc.get("chat_template_hash"), view=view,
                        model=doc.get("base_model") or _m(doc.get("proxy_identity")).get("base_model"), source=source)


def stamp_from_bon_run(doc: Mapping[str, Any], *, view: Optional[str] = None, source: str = "bon_served_run.json") -> PairingStamp:
    """A best-of-N cell's ``bon_served_run.json`` (``scripts/sdfbase/full22/bon_served_run.py``): ``serving_identity``,
    ``serving_engine`` (absent on a lens-era record), the object sha under the harness cell id in
    ``proxy_identity.object_artifact_sha256_by_cell``."""
    pid = _m(doc.get("proxy_identity"))
    harness = str(doc.get("harness_cell_id") or doc.get("cell_id"))
    sha = _m(pid.get("object_artifact_sha256_by_cell")).get(harness)
    return PairingStamp(cell=str(doc.get("cell_id")), serving_identity=doc.get("serving_identity"),
                        serving_engine=doc.get(_c.SERVED_ENGINE_FIELD), object_sha=sha, chat_template=None, view=view,
                        model=pid.get("base_model"), source=source)


def stamp_from_capeval_manifest(manifest: Mapping[str, Any], *, cell: Optional[str] = None, view: Optional[str] = None,
                                source: str = "manifest.json") -> PairingStamp:
    """A served capability store's ``manifest.json`` (controls / MMLU / GSM8K; ``served_capability_run``): the engine
    through THE ONE reader (``contracts.read_store_serving_engine`` — every stamp spelling, held to the identity; a torn
    manifest is a :class:`PairingError`), the identity from the ``route`` block (else top level), the object sha from
    ``fit_provenance.object_sha`` / ``provenance.artifact_sha256``, the served template hash from
    ``fit_provenance.served_template_hash`` / ``provenance.serving.chat_template_hash``."""
    reading = _c.read_store_serving_engine(manifest, where=source)
    if reading.refusal:
        raise PairingError(reading.refusal)
    route = _m(manifest.get(_c.MANIFEST_KEY_ROUTE))
    ident = route.get("serving_identity") or manifest.get("serving_identity")
    fp = _m(manifest.get("fit_provenance"))
    prov = _m(manifest.get(_c.MANIFEST_KEY_PROVENANCE))
    sha = fp.get("object_sha") or prov.get("artifact_sha256") or route.get(_c.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256)
    tmpl = fp.get("served_template_hash") or _m(prov.get("serving")).get("chat_template_hash")
    return PairingStamp(cell=str(cell or manifest.get("cell_id")), serving_identity=ident, serving_engine=reading.stated,
                        object_sha=sha, chat_template=tmpl, view=view,
                        model=manifest.get("model_id") or fp.get("served_model_id"), source=source)


def stamp_from_petri_cell(entry: Mapping[str, Any], *, view: Optional[str] = None, serving_engine: Optional[str] = None,
                          source: str = "cells.json") -> PairingStamp:
    """A Petri root's ``cells.json`` canonical entry (``cell_id``, ``serving_identity``, ``object_path``); the engine
    stamp comes from the root's standup record when the caller read one (``serving_engine``), else by identity path."""
    return PairingStamp(cell=str(entry.get("cell_id")), serving_identity=entry.get("serving_identity"),
                        serving_engine=serving_engine or entry.get(_c.SERVED_ENGINE_FIELD),
                        object_sha=entry.get("object_sha256") or entry.get("artifact_sha256"), chat_template=None, view=view,
                        model=entry.get("model_id"), source=source)


#: the Petri harness spells a roster cell id ``<si>-<...>__<arm>`` as ``si_<si>_<...>__<arm>`` (``cells.json`` ``cell_id``,
#: ``rows.parquet`` ``arm_name``); the roster spelling survives on the SAME entry as the parent directory of ``object_path``
#: (``.../cells_view/<view>/<roster id>/object.pt``). The two spellings are mapped THROUGH that entry, never by similarity.
PETRI_HARNESS_PREFIX = "si_"


def petri_harness_cell_id(cell: str) -> str:
    """The deterministic roster -> harness transform: ``si_`` + ``-`` -> ``_``. ``none`` and an id already in harness spelling
    (the ``si_`` prefix) are returned as they are. A caller must VERIFY the result against ``cells.json``
    (:func:`resolve_petri_entry`); the transform alone is a candidate, not a mapping."""
    cell = str(cell)
    if cell == CELL_NONE or cell.startswith(PETRI_HARNESS_PREFIX):
        return cell
    return PETRI_HARNESS_PREFIX + cell.replace("-", "_")


def petri_roster_cell_id_of_entry(entry: Mapping[str, Any]) -> Optional[str]:
    """The roster spelling a ``cells.json`` entry carries: the parent directory name of its ``object_path``
    (``.../<roster id>/object.pt``); ``none`` for the base cell; None when the entry carries no roster spelling."""
    if entry.get("cell_id") == CELL_NONE:
        return CELL_NONE
    op = entry.get("object_path")
    if not isinstance(op, str) or not op:
        return None
    parts = [x for x in op.replace("\\", "/").split("/") if x]
    return parts[-2] if len(parts) >= 2 else None


def resolve_petri_entry(entries: Any, cell: str) -> tuple[Optional[Mapping[str, Any]], dict[str, Any]]:
    """The ``cells.json`` canonical entry for ``cell`` in EITHER spelling, and the mapping record.

    Order: (1) an entry whose ``cell_id`` IS the requested id (harness spelling, or a root keyed by roster ids); else (2) the
    entry whose ``cell_id`` is :func:`petri_harness_cell_id` of the requested roster id, VERIFIED by the roster spelling that
    entry carries (:func:`petri_roster_cell_id_of_entry` == the request). Refuses by name (:class:`PairingError`) when the
    verification misses: the harness-id entry carries no roster spelling, or a different one; or an entry carries the roster
    spelling under a ``cell_id`` the transform does not produce (the transform is wrong for this root). ``(None, record)``
    when no entry names the cell in either spelling (absent — a survey's "missing", not a mapping failure). Two entries for one
    id refuse (a root is one entry per cell)."""
    ents = [e for e in (entries or []) if isinstance(e, Mapping)]
    cell = str(cell)
    harness = petri_harness_cell_id(cell)
    rec: dict[str, Any] = {"requested": cell, "harness": harness, "roster": None, "how": None}
    exact = [e for e in ents if e.get("cell_id") == cell]
    if len(exact) > 1:
        raise PairingError(f"petri cells.json: {len(exact)} canonical entries carry cell_id {cell!r} — a root is one entry per cell")
    if exact:
        e = exact[0]
        rec["roster"] = petri_roster_cell_id_of_entry(e) if cell.startswith(PETRI_HARNESS_PREFIX) else cell
        rec["how"] = "cell_id equals the requested id"
        return e, rec
    by_harness = [e for e in ents if e.get("cell_id") == harness]
    by_roster = [e for e in ents if petri_roster_cell_id_of_entry(e) == cell]
    if len(by_harness) > 1:
        raise PairingError(f"petri cells.json: {len(by_harness)} canonical entries carry cell_id {harness!r} — a root is one entry per cell")
    if by_roster and not (by_harness and by_roster[0] is by_harness[0]):
        raise PairingError(f"petri cells.json: the entry carrying the roster spelling {cell!r} (object_path {by_roster[0].get('object_path')!r}) "
                           f"names cell_id {by_roster[0].get('cell_id')!r}, not the transform {harness!r} — the roster/harness mapping is not "
                           "the deterministic one; refusing rather than inferring it from name similarity")
    if not by_harness:
        rec["how"] = "absent: no canonical entry names the cell in either spelling"
        return None, rec
    e = by_harness[0]
    roster = petri_roster_cell_id_of_entry(e)
    if roster != cell:
        raise PairingError(f"petri cells.json: entry {harness!r} (the transform of {cell!r}) carries "
                           + (f"the roster spelling {roster!r}, not {cell!r}" if roster else f"no roster spelling (object_path {e.get('object_path')!r})")
                           + " — the mapping cannot be verified against cells.json; refusing rather than inferring it from name similarity")
    rec["roster"] = cell
    rec["how"] = f"{PETRI_HARNESS_PREFIX!r} + '-'->'_' transform, verified against the entry's object_path"
    return e, rec


def stamp_from_sweep_stamp(stamp: Mapping[str, Any], *, cell: str, view: Optional[str] = None,
                           source: str = "engine_stamp") -> PairingStamp:
    """The sweep's per-row :func:`sisteer.eval.sweep.engine_stamp`: ``serving_path`` (``hf`` / ``lens`` / ``lora`` /
    ``passthrough``) becomes a ``proxy_vllm+<path>+interleaved`` identity for the ``vllm`` target engine (the only
    engine that has serving paths; an ``hf`` target has no proxy identity and refuses the allowance by construction),
    ``served_template_hash`` the template, ``object_sha`` the object; an optional ``serving_engine`` key on the stamp
    is honoured."""
    path = str(stamp.get("serving_path") or "")
    ident = f"proxy_vllm+{path}+interleaved" if path in (PATH_LENS, PATH_LORA, PATH_PASSTHROUGH) else None
    return PairingStamp(cell=cell, serving_identity=ident, serving_engine=stamp.get(_c.SERVED_ENGINE_FIELD),
                        object_sha=(str(stamp["object_sha"]) if stamp.get("object_sha") else None),
                        chat_template=(str(stamp["served_template_hash"]) if stamp.get("served_template_hash") else None),
                        view=stamp.get("view") if isinstance(stamp.get("view"), str) else view, source=source)


def stamp_from_gsm8k_leg(leg: Mapping[str, Any], *, source: Optional[str] = None) -> PairingStamp:
    """A GSM8K paired-200 LEG record as ``tinycap_gsm8k_paired200.collect_leg`` files it: ``identity.serving_identity``
    (the route's), ``identity.object_artifact_sha256``, ``identity.chat_template_hash``, ``identity.model_id``. The
    engine stamp is the leg's ``launch.engine`` when the collector filed one — the manifest's ``route.serving_engine``
    as ``tinycap_math500_thinking.launch_engine_of`` resolves it (PRESENT IFF NATIVE, so a lens-era manifest reads lens
    and a stamped one its engine; filed whether or not the launch record was then found) — so an UNSTEERED passthrough
    ``none`` served on the lora_native standup (vLLM's path-matched anchor, the real row 532495) reads NATIVE, as its
    manifest says. Before this (2026-09-16) the stamp read only the arm contract's ``steered_path`` and an unsteered
    native leg fell to lens-by-era: a same-engine pair was stamped as the cross-engine allowance's case. Without a
    filed launch engine the ``steered_path`` rule stands (``lora`` -> native, ``lens`` -> lens, None -> by identity
    path; :func:`resolve_engine`)."""
    ident = _m(leg.get("identity"))
    launch_engine = _m(leg.get("launch")).get("engine")
    if launch_engine in ENGINES:
        engine: Optional[str] = str(launch_engine)
    else:
        steered_path = _m(leg.get("arm_contract")).get("steered_path")
        engine = ENGINE_NATIVE if steered_path == PATH_LORA else (ENGINE_LENS if steered_path == PATH_LENS else None)
    return PairingStamp(cell=str(leg.get("cell")), serving_identity=ident.get("serving_identity"), serving_engine=engine,
                        object_sha=ident.get("object_artifact_sha256"), chat_template=ident.get("chat_template_hash"),
                        view=leg.get("view"), model=ident.get("model_id"), source=source or str(leg.get("leg")))
