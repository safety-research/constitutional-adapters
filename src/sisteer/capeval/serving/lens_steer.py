"""Worker-resident vllm-lens hook glue + the 8 fail-loud Phase-0
assertions.

THE WORKER-SIDE OBJECT-LOADING CONTRACT (H3), end to end:

1. **Export (enumeration time, launcher process).**
   :func:`export_worker_registry` writes a read-only registry directory:
   one entry per lens-served leg, carrying the object's ARTIFACT BYTES
   (copied verbatim — the same bytes the intervention hash covers), its
   kind, its resolved spec, and its ``expect_hash``. The export RECOMPUTES
   ``manifest.petri_intervention_hash`` over the copied bytes — REREAD from
   the registry file it just wrote, so every digest it records or checks is
   a digest of what the worker will load (R16-A) — and refuses any mismatch:
   a corrupted copy can never reach a worker. An optional
   ``validator`` (the launch script passes a ``methods.load_object``-based
   one post-rebase) hydrates each artifact on the export side; its
   verdict is recorded per entry.
2. **Preload (server start, INSIDE each vLLM worker).**
   :class:`WorkerObjectRegistry` loads every entry into immutable state:
   re-verifies the artifact content hash AND the intervention hash, then
   translates the artifact via the per-kind adapter
   (``lens_adapters.get_adapter``) into :class:`~.lens_adapters.HookOp`
   tuples. A missing adapter or a hash mismatch refuses loudly AT
   STARTUP, never at request time.
3. **Request time.** Requests carry ONLY the immutable selector
   ``{"petri_object": <cell_id>, "expect_hash": <intervention_hash>}``
   (+ ``"scale"`` for the no-op leg) — never tensors, never callables,
   never pickled objects. :meth:`WorkerObjectRegistry.select` refuses an
   unknown selector, a hash mismatch, or a scale that contradicts the
   registry. The HF ``apply(model, scale)`` contract NEVER crosses the
   process boundary.

The ``none__lens0`` leg: base served with the hook ARMED at
scale 0. The export registers it with a DONOR — the lexicographically
first lens-served canonical object (real tensors flow through the hook,
multiplied by 0.0) — or, when no lens-served object exists, the built-in
zero op (the hook still runs). Either way ``apply_hook_ops`` at scale 0.0
returns the hidden states bit-identically (assertion 8's contract).

WHAT vllm-lens 1.2.1 ACTUALLY PROVIDES, and why the wiring cannot just
use its stock hook API (read out of the shipped wheel, 2026-08-21 — the
pin file's "compatibility UNVERIFIED" note still stands for RUNTIME
behaviour, but the interface below is what the package contains):

* ``Hook(fn, layer_indices, pre=False)``; ``fn(ctx, hidden)`` is called
  ONCE PER REQUEST PER LAYER with that request's own row slice, so the
  plugin already does the continuous-batching segmentation. ``pre=True``
  hooks receive the layer INPUT, which means resid_pre is directly
  hookable and a resid_pre object need not be re-expressed as the
  previous block's output.
* BUT the context it hands the hook is only ``layer_idx``, ``seq_len``,
  ``saved`` and ``model`` — and ``seq_len`` is ``end - start``, the QUERY
  length of THIS forward, not the sequence's absolute length. So through
  the stock API a hook cannot know a row's absolute position, cannot
  distinguish a complete prefill from the 1-token remnant of a chunked
  one, and cannot see token ids at all. Position-dependent ops
  (BOS-skip, positions="last", and above all the response mask, which
  must match header TOKEN IDS) are therefore NOT expressible through it.
  That is why the position/phase context is built by our own worker-side
  segmentation (``serving/lens_batch.py``) from the forward metadata and
  the runner's input batch, rather than from ``HookContext.seq_len``.
* ``fn`` is cloudpickled when it arrives over HTTP — explicitly
  "arbitrary code execution on the server" in the plugin's own docs. The
  H3 selector-only contract (preload objects at startup; requests carry
  ids) is what keeps that door shut, and must stay shut.

TWO PERFORMANCE FACTS, both structural rather than tunable:

* the plugin sets ``enforce_eager = True`` unconditionally
  (``_activations_plugin.py``), so CUDA graphs are off process-wide for
  as long as it is loaded — the documented ~10-30% decode cost.
* it forces ``skip_reading_prefix_cache`` on EVERY request in a batch
  whenever hooks are needed, and persistent hooks make that permanent.
  So a lens server gets no prefix-cache reuse at all; keeping prefix
  caching off is an honest statement of what the path costs, not a
  restriction that could be lifted by configuration. Arms served through
  native multi-LoRA instead pay neither cost, which is the throughput
  argument for keeping that path genuine rather than folding the
  weight-level arms into hooks.

vllm-lens itself is NOT importable in the CPU venv — everything here is
import-guarded (:func:`require_vllm_lens`); the registry, adapters, and
assertion harness are pure-python/torch and CPU-testable. The actual
in-worker hook installation (`vllm_lens.Hook` wiring through
``SamplingParams.extra_args``) happens on a GPU node at server start and
is exercised by ``scripts/petri_serving_gate.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri.defense_cells import (CellSource,
                                                 NOOP_LENS_CELL_ID,
                                                 PetriCellSet,
                                                 parse_cell_id,
                                                 stacked_resolved_spec)
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import serve_declaration as sd

try:  # petri-bloom plan R2: vllm-lens is launch-time-only; guard it.
    import vllm_lens  # type: ignore  # noqa: F401
    HAVE_VLLM_LENS = True
except ImportError:
    HAVE_VLLM_LENS = False


class LensSteerError(RuntimeError):
    """A worker-side lens refusal (registry integrity, unknown selector,
    scale contradiction) — loud, never a silent no-op (vllm-lens's own
    documented failure mode is *silently capturing nothing*, so every
    seam here refuses instead of degrading)."""


def require_vllm_lens() -> None:
    if not HAVE_VLLM_LENS:
        raise LensSteerError(
            "vllm_lens is not importable in this process — the lens serving "
            "path runs only in the pinned GPU serving venv "
            "(configs/capeval/vllm-requirements.txt); CPU tests exercise the "
            "registry/adapters/assertions without it")


#: The registry index filename inside the registry dir.
REGISTRY_INDEX = "registry.json"
#: The selector keys a request may carry (H3: immutable IDs only —
#: plus the per-request FACTS the proxy injects: ``scale`` for the
#: no-op leg, and the proxy-VERIFIED declarations — two that relocate a
#: response op's boundary to the last header occurrence,
#: ``response_prefill`` (the prefill-seeded render of the served-
#: loglikelihood landing) and ``agent_loop`` (the Petri agent-loop
#: render of the multi-turn landing), and one that selects WHICH header
#: the boundary is located against, ``thinking_on`` (the declared
#: thinking-ON render of the fit-support-transfer landing) — each threaded
#: into the same-named ``HookContext`` field by the slot builder, never two
#: on one request).
SELECTOR_KEYS = ("petri_object", "expect_hash", "scale",
                 "response_prefill", "agent_loop", "thinking_on")
#: the declared FACTS among :data:`SELECTOR_KEYS` — strict bools on the
#: wire (see :func:`_wire_bool`), MUTUALLY EXCLUSIVE (at most one per
#: request: a context refuses any two).
SELECTOR_FACT_KEYS = ("response_prefill", "agent_loop", "thinking_on")
#: the registry-index key of the declared thinking-ON header
REGISTRY_THINKING_HEADER_KEY = "assistant_header_ids_thinking_on"


# --- export (launcher side) ---------------------------------------------------


@dataclass(frozen=True)
class RegistryExportEntry:
    """One exported registry entry (the JSON schema of REGISTRY_INDEX)."""

    cell_id: str
    kind: str
    resolved_spec: Mapping[str, Any]
    #: the selector contract: what a request's expect_hash must equal
    #: (the LEG's intervention hash — for the no-op leg that is the BASE
    #: cell's hash, while the artifact belongs to the donor).
    expect_hash: str
    artifact: Optional[str]  # registry-dir-relative path or None (zero op)
    sidecar: Optional[str]
    artifact_sha16: Optional[str]  # content integrity over the copied bytes
    #: the intervention hash recomputed over (artifact, sidecar, spec) —
    #: for arm cells equals expect_hash; for the donor-backed no-op leg it
    #: is the DONOR's hash (recorded so the worker can re-verify).
    artifact_intervention_hash: Optional[str]
    forced_scale: Optional[float]  # 0.0 on the no-op leg; None otherwise
    donor_cell: Optional[str]  # no-op leg: which object's tensors are armed
    validated_by: Optional[str]  # export-side validator provenance
    #: the fit-position regime the artifact's own hashed bytes record
    #: (lens_adapters.recorded_fit_positions) — "all" | "response" |
    #: "prompt" (loreft's kind-intrinsic identity fit) | None
    #: when nothing records one. INFORMATIONAL, not hashed: the guard
    #: that enforces fit==serve runs inside every adapter; this field
    #: makes the absent case a STATED absence in the durable index
    #: rather than a silent default. Stamped by the capeval exporter;
    #: older/petri-shim indexes simply lack the key.
    positions_at_fit: Optional[str] = None


def write_immutable(path: Path, data: bytes) -> bool:
    """Write an exported registry file as an IMMUTABLE snapshot (0440) —
    idempotently and atomically. Returns True when bytes were written.

    The export leaves every blob 0440 (NFS honors modes; the worker
    re-verifies hashes regardless, which is the real guarantee), so a bare
    ``write_bytes`` on a RE-EXPORT into the same dir hit EACCES against the
    previous run's file — a proxy re-submitted into its standup dir could
    never come back up (the sdfbase-mt Petri smoke, job 375604, 2026-09-04).
    Identical bytes are left untouched, so a re-export is PROVEN idempotent
    rather than merely tolerated; different bytes replace the DIRECTORY
    ENTRY through a temp file + ``os.replace`` (the writable parent permits
    that whatever the old file's mode), which is also crash-safe — a reader
    sees the old bytes or the new, never a torso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP)
    os.replace(tmp, path)
    return True


def export_worker_registry(
    cell_set: PetriCellSet,
    source: CellSource,
    out_dir: Path,
    *,
    validator: Optional[Callable[[Mapping[str, Any], Path], None]] = None,
    validator_name: Optional[str] = None,
    zero_op_hidden_size: Optional[int] = None,
    zero_op_layer: int = 0,
    assistant_header_ids: Optional[Sequence[int]] = None,
    assistant_turn_end_ids: Optional[Sequence[int]] = None,
    expected_artifact_sha256: Optional[Mapping[str, str]] = None,
    assistant_header_ids_thinking_on: Optional[Sequence[int]] = None,
) -> List[RegistryExportEntry]:
    """Write the worker registry dir for every lens-served leg.

    ``validator(entry_dict, artifact_path)`` — when given — hydrates each
    artifact on the export side (post-rebase: a ``methods.load_object``
    wrapper); a validator exception is a loud export refusal. Without one
    the entries record ``validated_by: null`` (the pre-rebase state; the
    §10 gate still exact-checks every kind end-to-end before any audit).

    ``expected_artifact_sha256`` — ``{cell_id: sha256 hex}`` — binds the
    export to bytes verified OUTSIDE this standup (tinycap E40 H1): the
    launcher's preflight derives an object's identity and records the
    artifact's sha256; the standup receives that digest and the bytes it
    copies here — the bytes the worker loads and re-verifies at startup,
    REREAD from the written ``artifact.bin`` rather than taken from the
    buffer that was written (R16-A / E45 H1-adjacent, so the comparison is
    against the copy and not against itself) — must hash to it, else the
    export refuses naming both digests, before the index is written. The
    recomputed intervention-hash check below cannot catch that case on its
    own: it compares the copied bytes with an identity ENUMERATED IN THIS
    SAME STANDUP, so an object swapped on the shared volume between the
    preflight and the bring-up is self-consistent and passes it. An
    expectation naming a cell this registry does not serve as a lens leg
    refuses too (a mis-keyed expectation must never silently check
    nothing); a malformed digest refuses before any file is read.

    ``zero_op_hidden_size`` is consulted ONLY when no lens-served donor
    object exists for the ``none__lens0`` leg; needing it and not having
    it is a refusal, never a skipped leg.

    ``assistant_header_ids`` are the served chat template's
    GENERATION-PROMPT assistant header tokens
    (``chat_markers.generation_header_ids(tok)`` — the id-form of the
    fit-time mask's width, NOT the terminator-anchored
    ``derive_assistant_markers().header``, which absorbs prompt-side
    scaffold on gemma-4-style templates and would steer outside the fit
    support).
    They are recorded ONCE for the whole registry rather than per entry,
    because one registry serves one serving identity and therefore one
    template — a second header width in the same registry would mean two
    templates, which is a different server. Cells fit at
    ``positions="response"`` cannot be adapted without them; cells that
    are not do not consult them.

    ``assistant_turn_end_ids`` are the served template's assistant
    turn-END SCAFFOLD tokens as they appear between turns — the terminator
    plus the glue before the next turn's opener
    (``chat_markers.assistant_turn_end_ids(tok)``; gemma-4 ``[<turn|>,
    \\n]``, Llama-3.1 ``[<|eot_id|>]``), the ALL-TURNS mask's second
    marker (lens_adapters, THE ALL-TURNS MASK): a cell served at
    ``positions="response_all_turns"`` cannot be adapted without them,
    every other cell ignores them. Recorded once per registry for the same
    one-template reason; ``None`` for a registry that serves no all-turns
    cell (and for a successor-bounded template, which has no terminator
    and cannot serve the mode).

    ``assistant_header_ids_thinking_on`` is the served template's
    THINKING-ON generation header (``chat_markers.generation_header_ids_
    under(tok, enable_thinking=True)``) — the DECLARED SECOND HEADER of a
    cell served out of support under a thinking-ON render
    (the cell's hashed spec carries
    ``serve_out_of_support: ["thinking"]``, spelled ``<cell>@thinking``).
    REQUIRED iff some lens cell declares it; REFUSED when given with no
    such cell (a header nothing serves is drift waiting to be depended on).
    EQUAL to ``assistant_header_ids`` is the header-INVARIANT template's case
    (gemma-4: the thinking switch leaves the assistant header in place) —
    admitted and recorded as such, the declared render locating the same
    boundary as the OFF one (plan P9's refusal withdrawn).
    The worker's startup verifier still holds the recorded header to the
    LIVE template's ON header, so a registry exported against a template
    that moves its header is caught there. Recorded once per registry like
    the other markers.

    THE NO-OP TWIN'S DONOR (plan P12): ``none__lens0`` arms a DONOR lens
    cell's exact tensors at scale 0 and inherits the donor's resolved spec
    VERBATIM, so it is declared iff its donor is. The donor is chosen by
    :func:`noop_twin_donor` — prefer a cell carrying a RENDER declaration
    (``serve_out_of_support``), never a POSITION variant
    (``serve_experiment``), then the lexicographically first — so the twin
    accepts every render its steered siblings accept (on a ``none`` +
    ``@thinking`` standup the twin carries the ON header; on a mixed
    experiment-2 registry it is the undeclared base cell's twin), and a
    position variant's changed refusal behaviour never becomes the
    control leg's."""
    if validator is not None and not validator_name:
        raise LensSteerError("export_worker_registry: a validator needs a "
                             "validator_name (provenance is recorded)")
    expected_sha256 = _validated_expected_digests(expected_artifact_sha256)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lens_cells = [c for c in cell_set.canonical
                  if cm.parse_serving_identity(c.serving_identity).path == "lens"]
    noop_legs = [c for c in cell_set.controls if c.cell_id == NOOP_LENS_CELL_ID]

    entries: List[RegistryExportEntry] = []

    def _write_blob(rel: str, data: bytes) -> str:
        write_immutable(out_dir / rel, data)
        return rel

    def _export_object(cell, entry) -> Tuple[str, Optional[str], bytes, bytes]:
        artifact_path = Path(entry.object_path)
        artifact_bytes = artifact_path.read_bytes()
        sidecar_bytes = b""
        sidecar_rel = None
        if entry.sidecar_path:
            sidecar_bytes = Path(entry.sidecar_path).read_bytes()
            sidecar_rel = _write_blob(f"{cell.cell_id}/sidecar.bin",
                                      sidecar_bytes)
        artifact_rel = _write_blob(f"{cell.cell_id}/artifact.bin",
                                   artifact_bytes)
        # R16-A (E45 H1-adjacent): return the bytes the COPY holds, REREAD from
        # disk — never the source buffer this function just read. Every hash
        # below (the preflight sha256 binding, the recomputed intervention
        # hash, the recorded artifact_sha16) is then a hash of what the worker
        # will load, which is what the binding claims to check; hashing the
        # buffer instead only proves this function is self-consistent, and a
        # writer that persisted something else — a short write, a filesystem
        # that transformed the bytes, a swap landing between the write and the
        # launch — passed the export and was caught only at worker startup, or
        # (for the outside digest, which the worker never sees) not at all.
        artifact_bytes = (out_dir / artifact_rel).read_bytes()
        if sidecar_rel is not None:
            sidecar_bytes = (out_dir / sidecar_rel).read_bytes()
        return artifact_rel, sidecar_rel, artifact_bytes, sidecar_bytes

    for cell in lens_cells:
        entry = source.get(cell.cell_id)
        if entry is None or not entry.object_path:
            raise LensSteerError(
                f"export: lens cell {cell.cell_id!r} has no registry "
                "object — enumerate_cells should have reported it ABSENT "
                "before export (harness bug)")
        parsed = parse_cell_id(cell.cell_id)
        if parsed.cell_class == "stacked":
            # the worker registry stores the WRAPPER spec (kind 'stacked',
            # prefix bound by sha16 — the SI text itself never lands in
            # the registry, for containment) so the worker's own
            # startup rehash re-verifies BOTH legs without the text; the
            # lens adapter for 'stacked' unwraps and serves the object
            # leg's exact ops.
            if not entry.si_prefix:
                raise LensSteerError(
                    f"export: stacked cell {cell.cell_id!r} has no "
                    "si_prefix bytes in the registry — the prompt leg IS "
                    "half its intervention; enumerate_cells should have "
                    "refused it (harness bug)")
            resolved = stacked_resolved_spec(
                parsed.si_id, bytes(entry.si_prefix),
                dict(entry.resolved_spec) if entry.resolved_spec is not None
                else {"kind": entry.kind, "si_id": entry.si_id})
            kind = cm.PETRI_STACKED_KIND
        else:
            resolved = (dict(entry.resolved_spec) if entry.resolved_spec
                        is not None
                        else {"kind": entry.kind, "si_id": entry.si_id})
            kind = entry.kind
        artifact_rel, sidecar_rel, abytes, sbytes = _export_object(cell, entry)
        if cell.cell_id in expected_sha256:
            # E40 H1: the bytes just copied — the bytes the worker loads,
            # REREAD from the registry file (R16-A / E45 H1-adjacent), never
            # the source buffer — against the digest verified BEFORE this
            # standup existed
            got = hashlib.sha256(abytes).hexdigest()
            want = expected_sha256.pop(cell.cell_id)
            if got != want:
                raise LensSteerError(
                    f"export: {cell.cell_id!r}: the artifact this standup reads "
                    f"({entry.object_path}) hashes to sha256 {got}, but the "
                    f"preflight verified sha256 {want} — the object was replaced "
                    "between the preflight and this bring-up; refusing to serve "
                    "bytes no preflight vouched for")
        if parsed.cell_class == "stacked":
            recomputed = cm.petri_intervention_hash(
                resolved, artifact_bytes=abytes, sidecar_bytes=sbytes,
                si_prefix_bytes=bytes(entry.si_prefix))
        else:
            recomputed = cm.petri_intervention_hash(
                resolved, artifact_bytes=abytes, sidecar_bytes=sbytes)
        if recomputed != cell.intervention_hash:
            raise LensSteerError(
                f"export: {cell.cell_id!r}: recomputed intervention hash "
                f"{recomputed} != cell's {cell.intervention_hash} — the "
                "copied bytes are not the enumerated bytes; refusing")
        exp = RegistryExportEntry(
            cell_id=cell.cell_id, kind=kind, resolved_spec=resolved,
            expect_hash=cell.intervention_hash,
            artifact=artifact_rel, sidecar=sidecar_rel,
            artifact_sha16=cm.sha256_hex16(abytes),
            artifact_intervention_hash=recomputed,
            forced_scale=None, donor_cell=None,
            validated_by=validator_name if validator is not None else None)
        if validator is not None:
            validator(_entry_dict(exp), out_dir / artifact_rel)  # loud
        entries.append(exp)

    if expected_sha256:
        raise LensSteerError(
            f"export: expected_artifact_sha256 names {sorted(expected_sha256)}, which this "
            f"registry does not serve as a lens leg (lens cells: "
            f"{sorted(c.cell_id for c in lens_cells)}) — a mis-keyed expectation would check "
            "nothing; refusing")

    on_header = _validated_header_ids(assistant_header_ids_thinking_on)
    header = _validated_header_ids(assistant_header_ids)
    declaring = sorted(e.cell_id for e in entries
                       if la.declares_thinking(e.resolved_spec))
    if declaring and on_header is None:
        raise LensSteerError(
            f"export: lens cell(s) {declaring} declare the thinking-ON render "
            f"(serve_out_of_support: ['thinking']) but no "
            f"{REGISTRY_THINKING_HEADER_KEY} was given — the declared render's "
            "header must be recorded for the worker to locate the boundary "
            "under it (chat_markers.generation_header_ids_under(tok, "
            "enable_thinking=True)); refusing")
    if on_header is not None and not declaring:
        raise LensSteerError(
            f"export: {REGISTRY_THINKING_HEADER_KEY} was given but no lens cell "
            "declares the thinking-ON render — a header nothing serves is "
            "refused, never recorded quietly")
    # the donor is a property of the CANONICAL entries (the twin appended
    # below is scale-forced and never eligible), so it is chosen once
    donor = noop_twin_donor(entries) if lens_cells else None
    for leg in noop_legs:
        if donor is not None:
            donor_entry = donor
            exp = RegistryExportEntry(
                cell_id=leg.cell_id, kind=donor_entry.kind,
                resolved_spec=donor_entry.resolved_spec,
                expect_hash=leg.intervention_hash,
                artifact=donor_entry.artifact, sidecar=donor_entry.sidecar,
                artifact_sha16=donor_entry.artifact_sha16,
                artifact_intervention_hash=donor_entry.artifact_intervention_hash,
                forced_scale=0.0, donor_cell=donor_entry.cell_id,
                validated_by=donor_entry.validated_by)
        else:
            if zero_op_hidden_size is None:
                raise LensSteerError(
                    "export: no lens-served donor object is eligible for the "
                    "none__lens0 leg (no lens cell, or every lens cell is a "
                    "POSITION variant — never a donor, plan P12) — pass "
                    "zero_op_hidden_size to arm the built-in zero op (a "
                    "skipped no-op leg would silently break Tier-1 pairing, "
                    "plan §3)")
            exp = RegistryExportEntry(
                cell_id=leg.cell_id, kind=la.ZERO_KIND,
                resolved_spec={"kind": la.ZERO_KIND,
                               "hidden_size": int(zero_op_hidden_size),
                               "layer": int(zero_op_layer)},
                expect_hash=leg.intervention_hash,
                artifact=None, sidecar=None, artifact_sha16=None,
                artifact_intervention_hash=None,
                forced_scale=0.0, donor_cell=None, validated_by=None)
        entries.append(exp)

    turn_end = _validated_turn_end_ids(assistant_turn_end_ids)
    index = {"version": 1,
             "assistant_header_ids": list(header) if header else None,
             "assistant_turn_end_ids": (list(turn_end) if turn_end
                                        else None),
             REGISTRY_THINKING_HEADER_KEY: (list(on_header) if on_header
                                            else None),
             "entries": [_entry_dict(e) for e in entries]}
    index_path = out_dir / REGISTRY_INDEX
    write_immutable(index_path,
                    json.dumps(index, indent=2, sort_keys=True).encode("utf-8"))
    return entries


def noop_twin_donor(entries: Sequence[RegistryExportEntry]
                    ) -> Optional[RegistryExportEntry]:
    """THE NO-OP TWIN'S DONOR RULE (fit-support-transfer plan P12), made
    explicit: among the CANONICAL lens entries (never a donor-backed or
    scale-forced one), prefer a cell whose hashed spec carries a RENDER
    declaration (``serve_out_of_support`` — the ``@thinking`` cell), never a
    POSITION variant (``serve_experiment`` — ``@all_turns`` /
    ``@all_positions``, whose changed mask would change the control leg's
    refusal behaviour), and break ties lexicographically by cell id. The
    twin inherits its donor's spec VERBATIM, so it accepts every render its
    steered siblings accept: declared with the ON header on a ``none`` +
    ``@thinking`` standup, the undeclared base cell's twin on experiment
    2's mixed registry. ``None`` when no entry is eligible (no lens cell,
    or position variants only) — the caller then arms the built-in zero
    op. A malformed declaration refuses (``lens_adapters.cell_declarations``)."""
    eligible = []
    for e in entries:
        if e.donor_cell is not None or e.forced_scale is not None:
            continue
        decl = la.cell_declarations(e.resolved_spec)
        if sd.SERVE_EXPERIMENT_KEY in decl:
            continue  # a position variant is never the control leg's donor
        rank = 0 if sd.SERVE_OUT_OF_SUPPORT_KEY in decl else 1
        eligible.append((rank, e.cell_id, e))
    if not eligible:
        return None
    return min(eligible, key=lambda t: (t[0], t[1]))[2]


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _validated_expected_digests(raw: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """``expected_artifact_sha256`` as a fresh ``{cell_id: 64-hex}`` dict (the
    export pops consumed keys), refusing anything that is not a mapping of
    cell ids to lowercase 64-hex sha256 digests — a digest of the wrong
    shape can never match, so it must not be accepted as an expectation."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise LensSteerError(
            f"expected_artifact_sha256 must be a mapping of cell id -> sha256 hex, got "
            f"{type(raw).__name__}")
    out: Dict[str, str] = {}
    for cell_id, digest in raw.items():
        if not isinstance(cell_id, str) or not cell_id:
            raise LensSteerError(f"expected_artifact_sha256: bad cell id {cell_id!r}")
        if not isinstance(digest, str) or not _SHA256_HEX.match(digest):
            raise LensSteerError(
                f"expected_artifact_sha256[{cell_id!r}] = {digest!r} is not a lowercase "
                "64-hex sha256 digest")
        out[cell_id] = digest
    return out


def _validated_header_ids(raw: Any) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    ids = tuple(int(t) for t in raw)
    if not ids:
        raise LensSteerError(
            "assistant_header_ids is empty — a collapsed assistant-header "
            "boundary silently disables the whole steered region of a "
            "response-mode cell, so it is a refusal rather than a "
            "header-less registry. Pass None if the registry genuinely "
            "serves no response-mode cell")
    if any(t < 0 for t in ids):
        raise LensSteerError(f"assistant_header_ids must be non-negative "
                             f"token ids, got {list(ids)}")
    return ids


def _validated_turn_end_ids(raw: Any) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    ids = tuple(int(t) for t in raw)
    if not ids:
        raise LensSteerError(
            "assistant_turn_end_ids is empty — a turn with no locatable "
            "end has no span, so an all-turns cell could not be served; "
            "pass None if the registry genuinely serves no "
            "response_all_turns cell (a successor-bounded template has no "
            "terminator at all)")
    if any(t < 0 for t in ids):
        raise LensSteerError(f"assistant_turn_end_ids must be non-negative "
                             f"token ids, got {list(ids)}")
    return ids


def verify_assistant_header(tokenizer: Any,
                            recorded: Optional[Sequence[int]]) -> None:
    """Cross-check a registry's recorded assistant header against the
    header the SERVED tokenizer actually produces — called at worker
    startup, before any request.

    This is the check that catches a registry exported against one chat
    template and served under another. Without it the per-request marker
    match would still refuse, but it would refuse at REQUEST time on
    every request, which reads as a serving outage rather than as the
    configuration error it is. Deriving both sides with the same probe
    means a disagreement is a finding rather than two conventions
    drifting apart. Both sides are the GENERATION-PROMPT header
    (``chat_markers.generation_header_ids`` — the id-form of the
    fit-time width), never the terminator-anchored located header,
    whose definition legitimately disagrees with the fit support on
    templates with post-terminator scaffold (gemma-4: located 4 vs
    fit/appended 3)."""
    want = _validated_header_ids(recorded)
    try:
        from sisteer.chat_markers import (derive_assistant_markers,
                                          generation_header_ids)
    except ImportError as e:
        raise LensSteerError(
            "sisteer.chat_markers is not importable, so the served chat "
            "template's assistant header cannot be derived — refusing to "
            "serve a response-mode registry whose header was never "
            "cross-checked against the live template") from e
    markers = derive_assistant_markers(tokenizer)
    got = (tuple(generation_header_ids(tokenizer, markers))
           if markers.header else ())
    if want is None:
        if got:
            raise LensSteerError(
                f"the served chat template has a {len(got)}-token assistant "
                "header but the registry recorded none — it was exported "
                "without one and cannot serve response-mode cells; "
                "re-export with assistant_header_ids")
        return
    if got != want:
        raise LensSteerError(
            f"assistant header mismatch: the served chat template produces "
            f"{list(got)} but the registry recorded {list(want)} — the "
            "registry was exported against a DIFFERENT template than this "
            "server loads. Every response-mode cell would steer the wrong "
            "region; refusing at STARTUP")


def verify_assistant_header_thinking_on(tokenizer: Any,
                                        recorded: Optional[Sequence[int]]
                                        ) -> None:
    """The :func:`verify_assistant_header` sibling for the DECLARED
    THINKING-ON header: cross-check a
    registry's recorded ``assistant_header_ids_thinking_on`` against
    ``chat_markers.generation_header_ids_under(tok, enable_thinking=True)``
    on the SERVED tokenizer — at worker startup, before any request, so a
    registry exported against another template (or another regime) is a
    configuration finding rather than a per-request refusal on every
    declared request.

    CALLED ONLY FOR A REGISTRY THAT SERVES THINKING-ON CELLS
    (``WorkerObjectRegistry.serves_thinking_on`` — LOADED ops carrying the
    second header, never metadata), so an ordinary registry never touches
    this. Refuses: a registry that serves such cells but recorded no ON
    header; a template with no thinking switch (no ON render exists —
    ``ChatMarkerError`` from the derivation, re-raised); and a mismatch. An
    ON header EQUAL to the OFF header is the header-invariant template's
    case (gemma-4) and passes when it matches the recorded one — the
    mismatch guard is what still catches a registry exported against a
    template that MOVES its header (the rule 2026-09-10; plan P9's refusal
    withdrawn)."""
    want = _validated_header_ids(recorded)
    if want is None:
        raise LensSteerError(
            "the registry serves thinking-ON declared cells (ops carrying a "
            f"second header) but recorded no {REGISTRY_THINKING_HEADER_KEY} — "
            "an inconsistent registry; refusing at STARTUP")
    try:
        from sisteer.chat_markers import (ChatMarkerError,
                                          generation_header_ids,
                                          generation_header_ids_under)
    except ImportError as e:
        raise LensSteerError(
            "sisteer.chat_markers is not importable, so the served chat "
            "template's thinking-ON assistant header cannot be derived — "
            "refusing to serve a thinking-ON declared registry whose header "
            "was never cross-checked against the live template") from e
    try:
        got = tuple(generation_header_ids_under(tokenizer, enable_thinking=True))
    except ChatMarkerError as e:
        raise LensSteerError(
            f"thinking-ON header check failed for a declared registry: {e}. "
            "The served template has no thinking-ON render to declare "
            "against; refusing at STARTUP") from e
    if got != want:
        raise LensSteerError(
            f"thinking-ON assistant header mismatch: the served chat template "
            f"produces {list(got)} but the registry recorded {list(want)} — "
            "the registry was exported against a DIFFERENT template (or "
            "regime) than this server loads. Every declared thinking-ON "
            "request would steer the wrong region; refusing at STARTUP")


def verify_assistant_turn_end(tokenizer: Any,
                              recorded: Optional[Sequence[int]]) -> None:
    """The :func:`verify_assistant_header` sibling for the ALL-TURNS
    mask's second marker: cross-check a registry's recorded turn-end
    scaffold against the one the SERVED tokenizer's template actually
    puts after a historical assistant turn
    (``chat_markers.assistant_turn_end_ids`` — terminator + glue) — at
    worker startup, before any request, so a registry exported against
    another template is a configuration finding rather than a
    per-request refusal on every multi-turn prompt.

    CALLED ONLY FOR A REGISTRY THAT SERVES ALL-TURNS OPS
    (``WorkerObjectRegistry.serves_all_turns``; the worker's preflight
    gates on that, never on the presence of scaffold metadata): a
    response-only registry — with or without stale ``assistant_turn_end_ids``
    — is not held to the all-turns template requirements, which are
    stricter (the role-header shape check refuses Nemotron) and irrelevant
    to what it serves. A registry that recorded NO scaffold has nothing to
    verify and is left alone without touching the tokenizer."""
    want = _validated_turn_end_ids(recorded)
    if want is None:
        return
    try:
        from sisteer.chat_markers import (ChatMarkerError,
                                          assistant_turn_end_ids,
                                          turn_opener_id)
    except ImportError as e:
        raise LensSteerError(
            "sisteer.chat_markers is not importable, so the served chat "
            "template's turn-end scaffold cannot be derived — refusing to "
            "serve an all-turns registry whose scaffold was never "
            "cross-checked against the live template") from e
    got = tuple(assistant_turn_end_ids(tokenizer) or ())
    if got != want:
        raise LensSteerError(
            f"turn-end scaffold mismatch: the served chat template closes a "
            f"historical assistant turn with {list(got) or 'no terminator (a '
            'successor-bounded template)'} but the registry recorded "
            f"{list(want)} — the registry was exported against a DIFFERENT "
            "template than this server loads. Every all-turns cell would "
            "end its earlier spans at the wrong tokens (or never); refusing "
            "at STARTUP")
    # the all-turns historical rule validates the header that follows a
    # scaffold by SHAPE (header_ids[0], one role token, header_ids[2:]):
    # verify, on the served template, that the successor user header has
    # exactly that shape (chat_markers.successor_role_header_ids, reached
    # through turn_opener_id)
    try:
        turn_opener_id(tokenizer)
    except ChatMarkerError as e:
        raise LensSteerError(
            f"turn-opener check failed for an all-turns registry: {e}. The "
            "all-turns mask's historical rule needs the template's role "
            "headers to share the generation header's shape, and this "
            "template does not provide that; refusing at STARTUP") from e


def _assert_ops_express_spec_positions(cell_id: str, raw: Mapping[str, Any],
                                        ops: Sequence[la.HookOp]) -> None:
    """The served ops must EXPRESS the spec's recorded position mode.

    The scale-12 psr regression shipped because nothing stood between
    "the registry records positions='response'" and "the ops came out
    positions='all'": the adapter dropped the recorded mask, and a full
    eval wave served a response-fit object at every position (the
    measured fit=response/serve=all model-destroyer corner) while every
    hash and readiness check passed — the intervention hash covers the
    SPEC, not the ops derived from it. An adapter that cannot express a
    recorded mode must refuse; a response spec yielding any non-response
    op is therefore an adapter bug, caught here at STARTUP rather than
    measured weeks later. (For a stacked cell the mode lives on the
    object leg's spec — the wrapper carries none of its own.)"""
    spec = raw.get("resolved_spec") or {}
    if spec.get("kind") == la.STACKED_KIND:
        spec = spec.get("object") or {}
    requested = spec.get("positions")
    if requested not in la.RESPONSE_MASK_MODES:
        return
    bad = sorted({op.positions for op in ops if op.positions != requested})
    if bad:
        raise LensSteerError(
            f"registry entry {cell_id!r}: resolved_spec records "
            f"positions={requested!r} but the adapter emitted op positions "
            f"{bad} — it dropped the recorded mask, so the server would "
            "silently serve a WIDER intervention than the registry claims "
            "(the scale-12 psr regression; for an all-turns spec, a "
            "different mask than the hashed one). Refusing at STARTUP")


def _assert_ops_express_declarations(cell_id: str, raw: Mapping[str, Any],
                                     ops: Sequence[la.HookOp]
                                     ) -> Dict[str, Any]:
    """Bind every serve DECLARATION of the entry's spec to the LOADED ops
    (the ``resumed_prefill_policy_cells``
    discipline: what is attested is derived from ops, never from metadata),
    returning the validated declaration (``{}`` for an undeclared cell)
    that ``lens_serve_declarations`` reports:

    - declares ``thinking`` ⇔ every header-located op carries the ON header
      (and at least one exists); an undeclared cell carries NO second header;
    - ``serve_experiment == fit_response_serve_all_turns`` ⇒
      ``{op.positions} == {"response_all_turns"}``;
    - ``serve_experiment == fit_response_serve_all_positions`` ⇒
      ``{op.positions} == {"all"}``;
    - any other combination refuses at STARTUP. (An UNNAMED all-turns or
      all-positions spec is the ordinary cell it always was — the name adds
      validation, never a requirement on unnamed cells.)"""
    spec = raw.get("resolved_spec") or {}
    try:
        decl = la.cell_declarations(spec)
    except la.AdapterError as e:
        raise LensSteerError(f"registry entry {cell_id!r}: {e}") from e
    served = {op.positions for op in ops}
    thinking = sd.OUT_OF_SUPPORT_THINKING in decl.get(sd.SERVE_OUT_OF_SUPPORT_KEY, [])
    header_ops = [op for op in ops if op.positions in la.RESPONSE_MASK_MODES]
    if thinking:
        if not header_ops or any(op.thinking_header_ids is None for op in header_ops):
            raise LensSteerError(
                f"registry entry {cell_id!r}: resolved_spec declares the "
                "thinking-ON render (serve_out_of_support: ['thinking']) but "
                f"its loaded ops do not all carry the ON header (positions="
                f"{sorted(served)}) — the declaration must be EXPRESSED by the "
                "ops it is attested for; refusing at STARTUP")
    elif any(op.thinking_header_ids is not None for op in ops):
        raise LensSteerError(
            f"registry entry {cell_id!r}: loaded ops carry a thinking-ON header "
            "but resolved_spec declares no thinking-ON render — an undeclared "
            "second header; refusing at STARTUP")
    experiment = decl.get(sd.SERVE_EXPERIMENT_KEY)
    if experiment is not None:
        _fit, want_served = sd.EXPERIMENT_FIT_SERVE_PAIRS[experiment]
        if served != {want_served}:
            raise LensSteerError(
                f"registry entry {cell_id!r}: resolved_spec declares "
                f"serve_experiment={experiment!r} (served positions "
                f"{want_served!r}) but the adapter emitted op positions "
                f"{sorted(served)} — the named experiment must be EXPRESSED by "
                "the ops it is attested for; refusing at STARTUP")
    return decl


def _entry_dict(e: RegistryExportEntry) -> Dict[str, Any]:
    return {"cell_id": e.cell_id, "kind": e.kind,
            "resolved_spec": dict(e.resolved_spec),
            "expect_hash": e.expect_hash, "artifact": e.artifact,
            "sidecar": e.sidecar, "artifact_sha16": e.artifact_sha16,
            "artifact_intervention_hash": e.artifact_intervention_hash,
            "forced_scale": e.forced_scale, "donor_cell": e.donor_cell,
            "validated_by": e.validated_by,
            "positions_at_fit": e.positions_at_fit}


# --- preload + select (worker side) -------------------------------------------


@dataclass(frozen=True)
class LoadedObject:
    """One preloaded, immutable worker-resident object."""

    cell_id: str
    kind: str
    expect_hash: str
    ops: Tuple[la.HookOp, ...]
    #: the selector-scale this entry serves at (0.0 = the no-op leg;
    #: 1.0 = a served arm — the object's own trained scale is baked into
    #: its ops by the adapter, never re-scaled here).
    effective_scale: float
    #: the cell's validated serve DECLARATION (``{}`` for an undeclared
    #: cell; else exactly one of ``serve_declaration``'s two closed shapes),
    #: bound to the loaded ops at load by
    #: :func:`_assert_ops_express_declarations` — what
    #: ``lens_serve_declarations`` attests.
    #: A donor-backed twin carries its donor's (same spec, same ops).
    serve_declarations: Mapping[str, Any] = field(default_factory=dict)


class WorkerObjectRegistry:
    """The per-worker immutable object registry: loaded ONCE
    at server start; requests select by immutable ID only."""

    def __init__(self, objects: Mapping[str, LoadedObject],
                 assistant_header_ids: Optional[Tuple[int, ...]] = None,
                 assistant_turn_end_ids: Optional[Tuple[int, ...]] = None,
                 assistant_header_ids_thinking_on: Optional[Tuple[int, ...]] = None):
        self._objects = dict(objects)
        #: the served template's assistant header, for the startup
        #: cross-check (:func:`verify_assistant_header`).
        self.assistant_header_ids = assistant_header_ids
        #: the served template's turn-end scaffold — the all-turns mask's
        #: second marker — for :func:`verify_assistant_turn_end`; None
        #: for a registry serving no ``response_all_turns`` cell.
        self.assistant_turn_end_ids = assistant_turn_end_ids
        #: the served template's DECLARED thinking-ON header — the second
        #: header of ``@thinking`` cells — for
        #: :func:`verify_assistant_header_thinking_on`; None for a registry
        #: serving no such cell.
        self.assistant_header_ids_thinking_on = assistant_header_ids_thinking_on
        #: (cell_id, hook site, layer) -> that object's ops at that site
        #: and layer, PRECOMPUTED once here (the registry is immutable;
        #: ``materialize`` builds a new registry and therefore a new
        #: index over the materialized ops). The request-time
        #: :meth:`apply` used to hand ``apply_hook_ops`` the object's
        #: WHOLE op list and let it skip the other layers — a 32-op scan
        #: per (request, layer), ~6% of the steered step in the 2026-09-04
        #: profile. A (cell, site, layer) with no op maps to ``()``.
        self._ops_index: Dict[Tuple[str, str, int], Tuple[la.HookOp, ...]] = {}
        for cid, obj in self._objects.items():
            by_key: Dict[Tuple[str, int], list] = {}
            for op in obj.ops:
                by_key.setdefault((la.op_site(op), op.layer), []).append(op)
            for (site, layer), ops in by_key.items():
                self._ops_index[(cid, site, layer)] = tuple(ops)

    @classmethod
    def load(cls, registry_dir: Path) -> "WorkerObjectRegistry":
        registry_dir = Path(registry_dir)
        index_path = registry_dir / REGISTRY_INDEX
        if not index_path.is_file():
            raise LensSteerError(f"no registry index at {index_path} — run "
                                 "export_worker_registry first")
        index = json.loads(index_path.read_text())
        if index.get("version") != 1:
            raise LensSteerError(
                f"registry index version {index.get('version')!r} != 1")
        header_ids = _validated_header_ids(index.get("assistant_header_ids"))
        turn_end_ids = _validated_turn_end_ids(
            index.get("assistant_turn_end_ids"))
        on_header_ids = _validated_header_ids(
            index.get(REGISTRY_THINKING_HEADER_KEY))
        raws: Dict[str, Mapping[str, Any]] = {}
        for raw in index["entries"]:
            if raw["cell_id"] in raws:
                raise LensSteerError(
                    f"duplicate registry entry {raw['cell_id']!r}")
            raws[raw["cell_id"]] = raw
        objects: Dict[str, LoadedObject] = {}
        for cell_id, raw in raws.items():
            artifact_bytes = b""
            sidecar_bytes = b""
            if raw["artifact"] is not None:
                artifact_bytes = (registry_dir / raw["artifact"]).read_bytes()
                got = cm.sha256_hex16(artifact_bytes)
                if got != raw["artifact_sha16"]:
                    raise LensSteerError(
                        f"registry entry {cell_id!r}: artifact content hash "
                        f"{got} != recorded {raw['artifact_sha16']} — "
                        "refusing at STARTUP (corrupt registry)")
                if raw["sidecar"] is not None:
                    sidecar_bytes = (registry_dir / raw["sidecar"]).read_bytes()
                rehash = cm.petri_intervention_hash(
                    raw["resolved_spec"], artifact_bytes=artifact_bytes,
                    sidecar_bytes=sidecar_bytes)
                if rehash != raw["artifact_intervention_hash"]:
                    raise LensSteerError(
                        f"registry entry {cell_id!r}: recomputed intervention "
                        f"hash {rehash} != recorded "
                        f"{raw['artifact_intervention_hash']} — refusing at "
                        "STARTUP")
            cls._verify_selector_binding(cell_id, raw, raws)
            adapter = la.get_adapter(raw["kind"])  # loud on unknown kind
            ops = adapter(raw["resolved_spec"], artifact_bytes, sidecar_bytes,
                          header_ids=header_ids, turn_end_ids=turn_end_ids)
            # the declared thinking-ON header meets the ops HERE and only
            # here (lens_adapters.attach_thinking_header): the registry's ON
            # header is handed to exactly the entries whose hashed spec
            # declares the render (a mixed registry — none, base, @thinking
            # — hands the undeclared base nothing), and attach refuses a
            # declaring entry the registry recorded no ON header for
            try:
                ops = la.attach_thinking_header(
                    ops, raw["resolved_spec"],
                    on_header_ids if la.declares_thinking(raw["resolved_spec"])
                    else None)
            except la.AdapterError as e:
                raise LensSteerError(
                    f"registry entry {cell_id!r}: {e}") from e
            _assert_ops_express_spec_positions(cell_id, raw, ops)
            declarations = _assert_ops_express_declarations(cell_id, raw, ops)
            forced = raw["forced_scale"]
            objects[cell_id] = LoadedObject(
                cell_id=cell_id, kind=raw["kind"],
                expect_hash=raw["expect_hash"], ops=tuple(ops),
                effective_scale=(float(forced) if forced is not None else 1.0),
                serve_declarations=declarations)
        return cls(objects, assistant_header_ids=header_ids,
                   assistant_turn_end_ids=turn_end_ids,
                   assistant_header_ids_thinking_on=on_header_ids)

    @staticmethod
    def _verify_selector_binding(cell_id: str, raw: Mapping[str, Any],
                                 raws: Mapping[str, Mapping[str, Any]]) -> None:
        """Bind the immutable request selector to the CANONICAL bytes (H3):
        for every ordinary arm entry the recomputed intervention hash must
        equal BOTH ``artifact_intervention_hash`` (checked by the caller)
        AND ``expect_hash`` — an index that swaps the artifact and updates
        its own recorded hashes while keeping the old selector would
        otherwise serve replacement tensors under the old intervention
        identity. Donor-backed no-op entries carry the DOCUMENTED hash
        difference and are validated against their donor explicitly."""
        donor_id = raw.get("donor_cell")
        if donor_id is None:
            if (raw["artifact"] is not None
                    and raw["artifact_intervention_hash"] != raw["expect_hash"]):
                raise LensSteerError(
                    f"registry entry {cell_id!r}: artifact intervention hash "
                    f"{raw['artifact_intervention_hash']} != selector "
                    f"expect_hash {raw['expect_hash']} — a non-no-op entry's "
                    "selector must bind the canonical artifact's recomputed "
                    "hash (H3); refusing at STARTUP (rebound registry)")
            return
        # donor-backed no-op leg: the ONE legal expect_hash/artifact-hash
        # split — validated against a canonical donor entry, field by field.
        if raw.get("forced_scale") != 0.0:
            raise LensSteerError(
                f"registry entry {cell_id!r}: donor-backed entries exist "
                "only for the scale-0 no-op leg, got forced_scale "
                f"{raw.get('forced_scale')!r}")
        donor = raws.get(donor_id)
        if donor is None:
            raise LensSteerError(
                f"registry entry {cell_id!r}: donor_cell {donor_id!r} is not "
                "in the registry — refusing at STARTUP")
        if donor.get("donor_cell") is not None or donor.get(
                "forced_scale") is not None:
            raise LensSteerError(
                f"registry entry {cell_id!r}: donor {donor_id!r} is not a "
                "canonical arm entry (it is itself donor-backed or "
                "scale-forced) — refusing at STARTUP")
        for fieldname in ("kind", "resolved_spec", "artifact", "sidecar",
                          "artifact_sha16", "artifact_intervention_hash"):
            if raw[fieldname] != donor[fieldname]:
                raise LensSteerError(
                    f"registry entry {cell_id!r}: {fieldname!r} differs from "
                    f"donor {donor_id!r}'s — the no-op leg arms the DONOR's "
                    "exact tensors at scale 0 (plan §3); refusing at STARTUP")

    def objects(self) -> Tuple[LoadedObject, ...]:
        """Every loaded object, in cell-id order — for STARTUP
        enumeration only (which layers to hook, which cells need the
        prefill ledger). Request-time access goes through
        :meth:`select`, which is where the selector cross-checks live;
        this accessor deliberately offers no way to reach an object by
        an unverified id."""
        return tuple(self._objects[cid] for cid in self.cell_ids())

    def cell_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._objects))

    def serves_all_turns(self) -> bool:
        """Whether any LOADED op serves ``positions="response_all_turns"`` —
        the fact the startup turn-end / role-header certification is gated
        on. Never the presence of ``assistant_turn_end_ids`` metadata: a
        response-only registry may carry scaffold metadata (older builds
        derived it whenever a tokenizer was given) and must preflight
        exactly as before, on every template — the 2026-09-07 fifth
        review's regression was a response-only registry refused at
        startup on Nemotron because that metadata existed."""
        return any(op.positions == la.POSITIONS_RESPONSE_ALL_TURNS
                   for obj in self._objects.values() for op in obj.ops)

    def serves_thinking_on(self) -> bool:
        """Whether any LOADED op carries a declared thinking-ON header —
        the fact the startup ON-header certification
        (:func:`verify_assistant_header_thinking_on`) is gated on, derived
        from the ops exactly as :meth:`serves_all_turns` is (never from the
        presence of ``assistant_header_ids_thinking_on`` metadata)."""
        return any(op.thinking_header_ids is not None
                   for obj in self._objects.values() for op in obj.ops)

    def serve_declarations_by_cell(self) -> Dict[str, Dict[str, Any]]:
        """cell id -> the cell's serve declaration, for DECLARATION-CARRYING
        cells only: the map
        ``lens_serve_declarations`` attests, keyed like
        :meth:`resumed_prefill_policy_by_cell` (every loaded cell, the
        ``none__lens0`` twin included under its OWN id when its donor
        declares — same spec, same ops). Every value was bound to the LOADED
        ops at load (:func:`_assert_ops_express_declarations`), so the map
        attests what runs, not what a cell name claims. Undeclared cells
        are ABSENT."""
        return {cid: dict(self._objects[cid].serve_declarations)
                for cid in self.cell_ids()
                if self._objects[cid].serve_declarations}

    def resumed_prefill_ok_by_cell(self) -> Dict[str, bool]:
        """cell id -> ``lens_adapters.resumed_prefill_ok(cell ops)``: which
        served cells may RESUME a prefill from the KV cache (lens_adapters
        module docstring, RESUMED PREFILLS) — derived from the adapted OPS
        of this registry, so a launcher-side reader (the routing proxy
        deciding a stable vs request-unique cache salt per cell) computes
        exactly what the worker will enforce. Startup enumeration only,
        like :meth:`objects`; a scale-0 twin reads as its donor (same
        ops)."""
        return {cid: la.resumed_prefill_ok(self._objects[cid].ops)
                for cid in self.cell_ids()}

    def resumed_prefill_policy_by_cell(self) -> Dict[str, str]:
        """cell id -> ``lens_adapters.resumed_prefill_policy(cell ops)`` —
        the THREE-VALUED form (``never`` / ``always`` / ``conditional``) the
        proxy half mints cache salts from: a stable per-(cell, intervention)
        salt for ``always``, a stable salt for SINGLE-TURN requests only on
        ``conditional``, request-unique otherwise. Derived from the adapted
        OPS like :meth:`resumed_prefill_ok_by_cell` (which is this map
        read as ``policy == "always"``)."""
        return {cid: la.resumed_prefill_policy(self._objects[cid].ops)
                for cid in self.cell_ids()}

    def ops_for(self, cell_id: str, site: str, layer: int
                ) -> Tuple[la.HookOp, ...]:
        """The ops of ``cell_id`` served at ``site`` and ``layer`` — the
        precomputed index (``()`` when none), equal by construction to
        ``tuple(op for op in ops_at_site(obj.ops, site) if op.layer ==
        layer)`` in the object's op order. An unknown site refuses like
        :func:`lens_adapters.ops_at_site`; an unknown cell is ``()`` —
        request-time access resolves the cell through :meth:`select`
        first, so a mis-typed id never reaches here unchecked."""
        if site not in (la.HOOK_SITE_BLOCK, la.HOOK_SITE_OPROJ):
            raise la.AdapterError(
                f"unknown hook site {site!r} (sites: "
                f"{[la.HOOK_SITE_BLOCK, la.HOOK_SITE_OPROJ]})")
        return self._ops_index.get((cell_id, site, int(layer)), ())

    def materialize(self, *, device: Any, dtype: Any) -> "WorkerObjectRegistry":
        """A NEW registry whose hook-op tensors live on ``device`` in
        ``dtype`` — called ONCE at worker startup, after the model device
        and serving dtype are known. The request-time hook then performs
        only the tensor op (``apply_hook_ops`` refuses a device/dtype
        mismatch rather than converting per generated token)."""
        return WorkerObjectRegistry(
            {cid: replace(obj, ops=la.materialize_ops(obj.ops, device=device,
                                                      dtype=dtype))
             for cid, obj in self._objects.items()},
            assistant_header_ids=self.assistant_header_ids,
            assistant_turn_end_ids=self.assistant_turn_end_ids,
            assistant_header_ids_thinking_on=self.assistant_header_ids_thinking_on)

    def select(self, petri_object: str, expect_hash: str,
               scale: Optional[float] = None) -> LoadedObject:
        """Resolve a request selector (H3). Refusals: unknown id, hash
        mismatch, or a scale contradicting the registry — the selector is
        a cross-check, never a control knob."""
        obj = self._objects.get(petri_object)
        if obj is None:
            raise LensSteerError(
                f"unknown petri_object selector {petri_object!r} "
                f"(registered: {list(self.cell_ids())}) — refusing loudly")
        if expect_hash != obj.expect_hash:
            raise LensSteerError(
                f"selector {petri_object!r}: expect_hash {expect_hash!r} != "
                f"registered {obj.expect_hash!r} — the request was built "
                "against a different object registry; refusing")
        if scale is not None and float(scale) != obj.effective_scale:
            raise LensSteerError(
                f"selector {petri_object!r}: request scale {scale!r} "
                f"contradicts the registered effective scale "
                f"{obj.effective_scale!r} — scale is set by the registry, "
                "not the client")
        return obj

    def apply(self, petri_object: str, expect_hash: str,
              hidden: Any, *, layer: int,
              scale: Optional[float] = None,
              ctx: Optional["la.RowContext"] = None,
              legacy_masks: bool = False,
              independent_rows: bool = False) -> Any:
        """The BLOCK-site hook body: select + apply at one layer. Scale
        0.0 returns ``hidden`` bit-identically
        (lens_adapters.apply_hook_ops).

        Applies exactly the object's BLOCK-site ops — an o_proj-site op
        (the lora kinds) is served by :meth:`apply_oproj` from the
        o_proj module hook instead, each op at exactly one site
        (``lens_adapters.ops_at_site``; ``apply_hook_ops`` additionally
        refuses a cross-site op outright, so an unrouted caller cannot
        apply wrong-site math).

        ``ctx`` is the per-forward position/phase context
        (:class:`lens_adapters.HookContext`) the GPU wiring builds from
        vllm-lens's per-request forward metadata; position/phase-dependent
        ops (prefill-only, positions="last", BOS-skip) REFUSE loudly
        without one rather than guessing rows. ``ctx`` may also be a
        :class:`lens_adapters.DecodeRows` (a vectorized decode group);
        ``legacy_masks`` selects the dense reference mask path;
        ``independent_rows`` declares every row a different request's
        single row (a vectorized group), so each reduction is issued per
        row (``lens_adapters._row_reduction``)."""
        obj = self.select(petri_object, expect_hash, scale)
        ops = self.ops_for(obj.cell_id, la.HOOK_SITE_BLOCK, layer)
        return la.apply_hook_ops(hidden, ops, obj.effective_scale,
                                 layer=layer, ctx=ctx,
                                 legacy_masks=legacy_masks,
                                 independent_rows=independent_rows)

    def apply_oproj(self, petri_object: str, expect_hash: str,
                    oproj_in: Any, oproj_out: Any, *, layer: int,
                    scale: Optional[float] = None,
                    ctx: Optional["la.RowContext"] = None,
                    legacy_masks: bool = False,
                    independent_rows: bool = False) -> Any:
        """The O_PROJ-site hook body: select + apply the object's
        oproj-site ops at one layer's attention output projection
        (``lens_adapters.apply_oproj_ops`` — the edit is computed from
        the o_proj INPUT rows and lands on the OUTPUT rows). The same
        selector cross-checks as :meth:`apply`; scale 0.0 and a
        fully-masked forward return ``oproj_out`` bit-identically."""
        obj = self.select(petri_object, expect_hash, scale)
        ops = self.ops_for(obj.cell_id, la.HOOK_SITE_OPROJ, layer)
        return la.apply_oproj_ops(oproj_in, oproj_out, ops,
                                  obj.effective_scale, layer=layer, ctx=ctx,
                                  legacy_masks=legacy_masks,
                                  independent_rows=independent_rows)


def parse_selector(extra_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a request's selector mapping (H3): only the immutable
    selector keys, only JSON scalars — tensors/callables/pickles refuse."""
    unknown = sorted(set(extra_args) - set(SELECTOR_KEYS))
    if unknown:
        raise LensSteerError(f"selector carries unknown key(s) {unknown} "
                             f"(allowed: {list(SELECTOR_KEYS)})")
    for req in ("petri_object", "expect_hash"):
        v = extra_args.get(req)
        if not isinstance(v, str) or not v:
            raise LensSteerError(f"selector {req!r} must be a non-empty "
                                 f"string, got {v!r}")
    out: Dict[str, Any] = {"petri_object": extra_args["petri_object"],
                           "expect_hash": extra_args["expect_hash"]}
    if "scale" in extra_args:
        s = extra_args["scale"]
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            raise LensSteerError(f"selector scale must be a number, got "
                                 f"{s!r}")
        out["scale"] = float(s)
    for fact in SELECTOR_FACT_KEYS:
        if fact in extra_args:
            out[fact] = _wire_bool(fact, extra_args[fact])
    declared = [fact for fact in SELECTOR_FACT_KEYS if out.get(fact)]
    if len(declared) > 1:
        raise LensSteerError(
            f"selector declares BOTH {declared[0]} and {declared[1]}"
            + (f" (and {declared[2:]})" if len(declared) > 2 else "")
            + " — the declared facts describe disjoint request shapes (a "
            "continued final message, a generation-prompt render with an open "
            "model turn, a thinking-ON tail render) and the proxy never emits "
            "two on one request, so this selector is malformed, refused")
    return out


def _wire_bool(name: str, value: Any) -> bool:
    """A declared selector FACT's wire value as a strict bool. The
    transport (``vllm_xargs``) types values ``str|int|float`` and
    lax-coerces JSON booleans to ints, so the legitimate wire forms are
    exactly ``{True, False, 1, 0}`` — anything else (strings, floats,
    other ints) is a malformed selector, refused. A bool-only parse
    killed the engine on the first live seeded request (2026-09-01); the
    proxy sends the transport's one honest spelling, int 1."""
    if isinstance(value, bool) or (isinstance(value, int)
                                   and value in (0, 1)):
        return bool(value)
    raise LensSteerError(
        f"selector {name} must be a bool or the transport ints 0/1, got "
        f"{value!r} — a declared fact is verified by the proxy, never a "
        "truthy coercion")


# --- the 8 fail-loud Phase-0 assertions ---------------------------------------
#
# The harness separates EVIDENCE GATHERING (GPU, against the live
# vllm-lens server — scripts/petri_serving_gate.py collects it) from
# ASSERTION LOGIC (pure, CPU-testable — below). Every check refuses
# loudly; there are no warnings and no tolerances except where an
# explicitly STATED one is part of the check's own contract (assertion 4,
# whose atol the caller must state — never a default).

PHASE0_ASSERTIONS = (
    "server_provenance",         # 1
    "saved_sentinel",            # 2 ctx.saved prefill-vs-decode
    "mixed_batch_isolation",     # 3
    "returned_modification",     # 4 vs fp32 reference
    "disable_flag",              # 5 VLLM_LENS_DISABLE=1
    "single_persistent_hook",    # 6
    "prefix_cache_isolation",    # 7 (caching ON + salt isolation PROVEN)
    "scale0_noop",               # 8
)
#: TP-equivalence (two-SI's slot) is N/A on a single H200 — the harness
#: keeps the slot and records this note verbatim.
TP_SLOT_NOTE = "TP=1 only"


class Phase0Failure(LensSteerError):
    """One or more Phase-0 assertions failed — the gate blocks the audit."""


@dataclass(frozen=True)
class Phase0Report:
    passed: Tuple[str, ...]
    tp_note: str = TP_SLOT_NOTE
    details: Mapping[str, Any] = field(default_factory=dict)


def _req(ev: Mapping[str, Any], name: str, *keys: str) -> List[Any]:
    out = []
    for k in keys:
        if k not in ev:
            raise Phase0Failure(f"phase0 {name}: evidence lacks {k!r} — "
                                "no evidence is a failure, not a pass")
        out.append(ev[k])
    return out


def _check_server_provenance(ev: Mapping[str, Any]) -> None:
    expected, reported, plugin, eager = _req(
        ev, "server_provenance", "expected_packages", "reported_packages",
        "plugin_registered", "enforce_eager")
    for pkg, want in dict(expected).items():
        got = dict(reported).get(pkg)
        if got != want:
            raise Phase0Failure(
                f"phase0 server_provenance: {pkg} reported {got!r}, pinned "
                f"{want!r}")
    if plugin is not True:
        raise Phase0Failure("phase0 server_provenance: vllm-lens plugin not "
                            "registered — its documented failure is "
                            "silently capturing nothing")
    if eager is not True:
        raise Phase0Failure("phase0 server_provenance: enforce_eager is not "
                            "set (vllm-lens requires it process-wide)")


def _nonempty(name: str, label: str, tokens: Any) -> List[Any]:
    """Vacuous evidence is a FAILURE, not a pass: ``[] == []`` proves
    nothing about serving, so every token-comparison probe must have
    actually generated output."""
    out = list(tokens)
    if not out:
        raise Phase0Failure(
            f"phase0 {name}: {label} is empty — a probe that produced no "
            "output tokens is vacuous evidence, a failure, never a pass")
    return out


def _check_saved_sentinel(ev: Mapping[str, Any]) -> None:
    prefill, decode, generated, steps = _req(
        ev, "saved_sentinel", "prefill_fires", "decode_fires",
        "generated_tokens", "decode_step_fires")
    if int(generated) < 1:
        raise Phase0Failure(
            "phase0 saved_sentinel: the probe generated 0 tokens — "
            "0 == 0 decode fires proves nothing; the probe must generate")
    if int(prefill) < 1:
        raise Phase0Failure("phase0 saved_sentinel: hook never fired at "
                            "prefill")
    if int(decode) != int(generated):
        raise Phase0Failure(
            f"phase0 saved_sentinel: hook fired on {decode} decode steps "
            f"for {generated} generated tokens — the hook must fire on "
            "EVERY generated token")
    per_step = [int(s) for s in steps]
    if len(per_step) != int(generated) or any(s != 1 for s in per_step):
        raise Phase0Failure(
            f"phase0 saved_sentinel: per-token decode-step evidence must "
            f"record EVERY generated token's step exactly once, got "
            f"{len(per_step)} steps with fire counts {sorted(set(per_step))} "
            f"for {generated} tokens — an aggregate count alone can hide a "
            "double-fired step masking a skipped one")


def _check_mixed_batch_isolation(ev: Mapping[str, Any]) -> None:
    ub, us, hb, hs = _req(
        ev, "mixed_batch_isolation", "unhooked_in_batch_tokens",
        "unhooked_solo_tokens", "hooked_in_batch_tokens",
        "hooked_solo_tokens")
    name = "mixed_batch_isolation"
    ub = _nonempty(name, "unhooked_in_batch_tokens", ub)
    us = _nonempty(name, "unhooked_solo_tokens", us)
    hb = _nonempty(name, "hooked_in_batch_tokens", hb)
    hs = _nonempty(name, "hooked_solo_tokens", hs)
    if ub != us:
        raise Phase0Failure(
            "phase0 mixed_batch_isolation: an UNHOOKED request's output "
            "changed when batched with hooked requests — cross-request "
            "contamination")
    if hb != hs:
        raise Phase0Failure(
            "phase0 mixed_batch_isolation: a HOOKED request's output "
            "changed when batched with unhooked requests")


def _check_returned_modification(ev: Mapping[str, Any]) -> None:
    delta, atol = _req(ev, "returned_modification", "max_abs_delta",
                       "stated_atol")
    d = float(delta)
    if not (math.isfinite(d) and d >= 0):
        raise Phase0Failure(
            f"phase0 returned_modification: max_abs_delta {d!r} is not a "
            "finite non-negative number — a maximum ABSOLUTE delta can "
            "never be negative or infinite; impossible evidence is a "
            "failed probe, not a pass")
    if atol is None:
        raise Phase0Failure("phase0 returned_modification: the comparison "
                            "atol must be STATED by the gate, never "
                            "defaulted")
    if d > float(atol):
        raise Phase0Failure(
            f"phase0 returned_modification: |returned - fp32 reference| = "
            f"{d} > stated atol {atol}")


def _check_disable_flag(ev: Mapping[str, Any]) -> None:
    saved, tokens, baseline = _req(
        ev, "disable_flag", "saved_count_with_disable",
        "tokens_with_disable", "baseline_tokens")
    if int(saved) != 0:
        raise Phase0Failure(
            f"phase0 disable_flag: VLLM_LENS_DISABLE=1 process still saved "
            f"{saved} activations")
    tokens = _nonempty("disable_flag", "tokens_with_disable", tokens)
    baseline = _nonempty("disable_flag", "baseline_tokens", baseline)
    if tokens != baseline:
        raise Phase0Failure("phase0 disable_flag: disabled process's output "
                            "differs from the unhooked baseline")


def _check_single_persistent_hook(ev: Mapping[str, Any]) -> None:
    regs, per_request, n = _req(
        ev, "single_persistent_hook", "registration_count",
        "per_request_registrations", "requests_observed")
    if int(n) < 1:
        raise Phase0Failure(
            "phase0 single_persistent_hook: 0 requests observed — "
            "'no per-request registrations' over no requests is vacuous "
            "evidence, a failure, never a pass")
    if int(regs) != 1:
        raise Phase0Failure(
            f"phase0 single_persistent_hook: {regs} hook registrations "
            "(must be exactly 1, GPU-resident, at server start)")
    if int(per_request) != 0:
        raise Phase0Failure(
            f"phase0 single_persistent_hook: {per_request} per-request "
            f"(re-)registrations across {n} requests — requests must carry "
            "selector IDs only (H3), never per-request hook state")


def _check_prefix_cache_isolation(ev: Mapping[str, Any]) -> None:
    """Assertion 7, REWORKED with the leakage question settled
    (2026-08-25). Prefix caching is ON by default, made safe by the
    proxy's per-request ``cache_salt`` (cell-stable for unhooked/LoRA
    legs; REQUEST-UNIQUE for selector-carrying lens legs, whose cached
    rows would otherwise skip the hooks). The assertion now PROVES the
    isolation on hardware instead of asserting the flag off:

    - ``prefix_caching_enabled`` — the RESOLVED engine value, a stated
      bool. ``False`` composes the legacy OFF launch and needs only the
      repeat-determinism pair below.
    - ``first_run_tokens`` / ``second_run_tokens`` — one selector-
      carrying request served twice IDENTICALLY. Under the salt
      discipline both serves are cold (request-unique salt), so the
      shapes match and the outputs must be bit-equal — repeat
      determinism, and a live check that the lens leg is not quietly
      reading cache (a warm second run changes prefill shape).
    - when enabled, ADDITIONALLY REQUIRED:
      ``lens_salts_unique`` — the gatherer states it issued a fresh
      salt per lens request (it minted them; nothing else can know);
      ``arm_tokens`` — a BITING arm served on probe prompt P (its
      steered blocks populate the cache pool first);
      ``twin_cold_tokens`` — the arm's scale-0 twin on the SAME P,
      served with nothing relevant cached;
      ``twin_warm_tokens`` — the twin on P served AFTER the arm.
      ``arm != twin_cold`` is the DISCRIMINATION requirement (an inert
      arm/prompt pair could never expose a leak — vacuous, a failure);
      ``twin_warm == twin_cold`` is the isolation proof: were the twin
      reading the arm's blocks (a salt collision / no salt), it would
      inherit the arm's steered KV and its shapes, and its output would
      move toward ``arm_tokens``."""
    enabled, first, second = _req(
        ev, "prefix_cache_isolation", "prefix_caching_enabled",
        "first_run_tokens", "second_run_tokens")
    if not isinstance(enabled, bool):
        raise Phase0Failure(
            f"phase0 prefix_cache_isolation: prefix_caching_enabled is "
            f"{enabled!r}, not a bool — the RESOLVED engine value must be "
            "stated; unreadable cache state can never pass")
    first = _nonempty("prefix_cache_isolation", "first_run_tokens", first)
    second = _nonempty("prefix_cache_isolation", "second_run_tokens", second)
    if first != second:
        raise Phase0Failure("phase0 prefix_cache_isolation: repeated "
                            "identical request produced different tokens — "
                            "cross-request state (or a lens leg quietly "
                            "reading cache: a warm re-serve changes the "
                            "prefill shape)")
    if not enabled:
        return  # the legacy OFF launch: nothing to isolate
    salted, arm, cold, warm = _req(
        ev, "prefix_cache_isolation", "lens_salts_unique", "arm_tokens",
        "twin_cold_tokens", "twin_warm_tokens")
    if salted is not True:
        raise Phase0Failure(
            "phase0 prefix_cache_isolation: prefix caching is ON but "
            "lens_salts_unique is not attested True — without the "
            "per-request salt a steered request can resume from cache "
            "and serve rows its hooks never touched; refusing")
    arm = _nonempty("prefix_cache_isolation", "arm_tokens", arm)
    cold = _nonempty("prefix_cache_isolation", "twin_cold_tokens", cold)
    warm = _nonempty("prefix_cache_isolation", "twin_warm_tokens", warm)
    if arm == cold:
        raise Phase0Failure(
            "phase0 prefix_cache_isolation: the probe arm's output equals "
            "its scale-0 twin's — this arm/prompt pair cannot DISCRIMINATE "
            "a cache leak (a twin reading the arm's blocks would produce "
            "the same equality); probe with an arm that bites, or this "
            "evidence is vacuous — a failure, never a pass")
    if warm != cold:
        raise Phase0Failure(
            "phase0 prefix_cache_isolation: the scale-0 twin's output "
            "CHANGED after the arm served the same prompt — the twin is "
            "reading the arm's cached (steered) blocks across cells: the "
            "per-cell cache_salt isolation is NOT holding; every cell's "
            "identity is corrupt under caching")


def _check_scale0_noop(ev: Mapping[str, Any]) -> None:
    scale0, baseline = _req(ev, "scale0_noop", "scale0_tokens",
                            "baseline_tokens")
    scale0 = _nonempty("scale0_noop", "scale0_tokens", scale0)
    baseline = _nonempty("scale0_noop", "baseline_tokens", baseline)
    if scale0 != baseline:
        raise Phase0Failure(
            "phase0 scale0_noop: a scale-0 lens request's output tokens "
            "differ from the undefended base — the armed hook is not a "
            "no-op at zero")


_PHASE0_CHECKS: Dict[str, Callable[[Mapping[str, Any]], None]] = {
    "server_provenance": _check_server_provenance,
    "saved_sentinel": _check_saved_sentinel,
    "mixed_batch_isolation": _check_mixed_batch_isolation,
    "returned_modification": _check_returned_modification,
    "disable_flag": _check_disable_flag,
    "single_persistent_hook": _check_single_persistent_hook,
    "prefix_cache_isolation": _check_prefix_cache_isolation,
    "scale0_noop": _check_scale0_noop,
}
assert tuple(_PHASE0_CHECKS) == PHASE0_ASSERTIONS


def run_phase0(evidence: Mapping[str, Mapping[str, Any]]) -> Phase0Report:
    """Evaluate the 8 assertions over gathered evidence. ALL of them are
    evaluated (so one failure does not mask another) and any failure set
    raises :class:`Phase0Failure` naming every failed assertion.

    Malformed evidence (a probe emitting the wrong type — ``int(...)`` /
    ``list(...)`` raising inside a checker) is normalized into that
    assertion's OWN Phase0Failure at the assertion boundary, so the loop
    still evaluates every remaining check; process-level interrupts
    (KeyboardInterrupt/SystemExit) are deliberately NOT caught."""
    failures: List[str] = []
    for name in PHASE0_ASSERTIONS:
        ev = evidence.get(name)
        try:
            if ev is None:
                raise Phase0Failure(f"phase0 {name}: no evidence gathered — "
                                    "a missing probe is a failure, not a "
                                    "pass")
            _PHASE0_CHECKS[name](ev)
        except Phase0Failure as e:
            failures.append(str(e))
        except Exception as e:  # malformed evidence — never aborts the loop
            failures.append(
                f"phase0 {name}: evidence could not be decoded "
                f"({type(e).__name__}: {e}) — malformed evidence is a "
                "failure of this assertion, not an abort of the rest")
    if failures:
        raise Phase0Failure(
            f"{len(failures)}/{len(PHASE0_ASSERTIONS)} Phase-0 assertions "
            "FAILED:\n- " + "\n- ".join(failures))
    return Phase0Report(passed=PHASE0_ASSERTIONS,
                        details={"tp_slot": TP_SLOT_NOTE})
