"""The REAL defense-cell registry adapter: the training side's
``sisteer.eval.defense_cell`` registry behind the harness's
:class:`~sisteer.capeval.petri.defense_cells.CellSource` protocol
(the seam this module closes).

The two sides name the same 22-cell matrix in DIFFERENT vocabularies, and
this module owns the ONE translation between them (a mismatch is a loud
refusal naming both vocabularies, never a silent skip — R5):

===============================  =====================================
harness id (petri manifest)      defense_cell registry id
===============================  =====================================
``none``                         ``none``
``prompted__safety``             ``prompted__safety_refusal``
``prompted__constitution``       ``prompted__si_constitution``
``prompted__irrelevant``         ``prompted__si_irrelevant``
``si_safety__<method>``          ``safety_refusal__<method>``
``si_constitution__<method>``    ``si_constitution__<method>``
``si_irrelevant__<method>``      ``si_irrelevant__<method>``
``antidote`` / ``alphasteer`` /  the same bare Class B arm names
``circuit_breaker``
===============================  =====================================

(The ``si_constitution__<m>`` / ``si_irrelevant__<m>`` rows coincide
textually by accident of the registry's SI ids being ``si_constitution`` /
``si_irrelevant``; the safety rows do not, so the translation is always
explicit, never a pass-through.)

A SERVE VARIANT rides through the translation untouched:
``<tt object cell>@<variant>`` ↔ ``<harness object cell>@<variant>`` —
``safety_refusal__psr_kl_all@thinking`` ↔
``si_safety__psr_kl_all@thinking`` — the base translates exactly as above
and the suffix is re-appended (``serve_declaration.SERVE_VARIANTS`` is the
closed vocabulary; the harness grammar admits it on object cells only).
The registry holds BASE cells only, so ``TrainTestCellSource.get``
resolves the base entry — same object_path, sidecar and artifact bytes —
and overlays the variant's hashed serve declaration onto the resolved spec
(``serve_declaration.apply_variant``): the ONE overlay on the
artifact-content-only resolution, named in the id and hashed into the
identity.

What each translated :class:`RegistryEntry` carries:

- **kind** — the ``methods.py`` OBJECT-KIND vocabulary (the string each
  object class's ``save()`` writes and ``load_object`` dispatches on),
  via :data:`ARM_OBJECT_KINDS`; verified against the artifact's own stored
  kind whenever the artifact exists (a mismatch refuses — the map is
  documented registry knowledge, and first contact with a real artifact
  proves it).
- **object_path / sidecar_path** — ``defense_cell.object_path_for`` (the
  registry's own resolution, cross-checked) plus the ``<artifact>.json``
  sidecar convention (``manifest.sidecar_path``): ALWAYS declared for the
  SteeringVector-container arms (``methods.VECTOR_SAFETENSOR_ARMS`` —
  their loader requires it), declared-when-present otherwise (a stray
  sidecar's bytes must enter the hash, never be silently ignored).
- **resolved_spec** — built ONLY from the artifact's own stored content
  (the exact fields ``load_object``'s per-kind ``load_blob`` consumes):
  the torch blob's JSON-safe non-tensor fields, or the SteeringVector
  JSON sidecar. Tensor payloads are deliberately omitted — their bytes
  are hashed verbatim as ``artifact_bytes``. When the artifact (or a
  declared sidecar) is missing/unreadable there IS no authoritative
  resolution: ``resolved_spec`` is None and enumeration reports the cell
  ABSENT from the missing component (never a fabricated minimal spec).
  Present-but-malformed content refuses loudly.
- **serving_path** — decided by the source's ``serving_path_policy``
  (:data:`SERVING_PATH_POLICIES`), a property of the STANDUP, never of
  the arm (the per-arm ``LORA_EXPORTED_ARMS`` set this replaced was empty
  for its whole life).
  ``"lens"`` — every lens standup; the DATACLASS default, so every existing
  construction site is byte-identical — puts EVERY object on the lens
  hook, byte-identical to the pre-policy adapter. ``"lora_native"`` — a
  plugin-free vLLM multi-LoRA standup, and since 2026-09-13 THE STANDUP
  ENTRY POINTS' default (:data:`DEFAULT_STANDUP_SERVING_PATH`: ONE native
  standup per view serves ``none``, the prompted cells and every LoRA
  cell; a PSR / vector cell is SKIPPED BY NAME upstream by
  :func:`partition_cells_for_native`, which supersedes the earlier
  two-standup split) —
  marks an object ``lora`` when ``defense_cells.native_lora_unservable_reason``
  is None for its kind and resolved spec, and REFUSES BY NAME otherwise
  (a PSR, position-masked, declared-variant or non-unit-scaled cell is
  never silently dropped or lens-served on a lora-native table: the source
  is the BACKSTOP, the partition is what keeps a skip-class cell from
  reaching it). A cell whose artifact is not on disk is judged by KIND
  only and reports ABSENT downstream as it always did. The ``none__lora0``
  control leg derives in ``enumerate_cells`` iff a ``lora`` route is live
  (or ``lora_controls=True`` names a zero-adapter-only server), on either
  policy.
- **si_prefix** — the exact SI text bytes, resolved through the training
  side's own resolver (``eval.jailbreak.resolve_instruction``: the
  ``SAFETY_SI`` module constant for ``safety_refusal``, the instruction
  registry for ``si_constitution``). The petri manifest's recorded
  ``prompted_prefix_hashes`` are verified against these bytes at
  enumeration (H8), so a drifted SI text refuses rather than auditing the
  wrong prompt.

The compiled-objects dir is never read as the two-SI config's RELATIVE
default (the per-commit immutable checkout — a directory the object
writers no longer write, since they moved to an absolute,
commit-independent store precisely because the in-checkout location
voided resume on every code change). An explicit ``defenses_dir``
override wins; otherwise ``defense_cell.resolve_defenses_dir`` (the ONE
resolution shared with the writers) is delegated to; and
when neither is available construction refuses loudly rather than
enumerating a store where every object cell would report ABSENT forever
(:func:`_resolve_defenses_dir`).

Import discipline: this module imports the registry/methods stack
(hence torch), so ``scripts/petri_run.py`` imports it lazily inside
``load_cell_source`` — the driver still parses on the bare capeval venv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import serve_declaration as sd


class RegistryAdapterError(dc.DefenseCellError):
    """A registry-seam refusal (untranslatable id, vocabulary drift,
    inconsistent registry/artifact) — loud, never a silent skip (R5)."""


# --- the SI vocabulary translation (verified against both sides' pins) -------

#: harness SI id (manifest.PETRI_SI_IDS) -> registry SI id
#: (data.instructions ids; defense_cell.DEFAULT_SI_IDS).
HARNESS_TO_TRAINTEST_SI: Dict[str, str] = {
    "safety": "safety_refusal",
    "constitution": "si_constitution",
    # irrelevant (2026-08-28): the off-target CONTROL SI joins the served
    # vocabulary. It was previously OUTSIDE the audit on purpose ("the
    # sweep's si_irrelevant control SI simply has no harness cells"),
    # which blocked prompted__si_irrelevant and every
    # si_irrelevant__<method> cell from the served path — the control is
    # compared on the SAME engine as everything else, so the vocabulary
    # is extended rather than special-cased. As with si_constitution, the
    # Class A ids coincide textually (harness `si_irrelevant__<m>` ==
    # registry `si_irrelevant__<m>`); the translation is still explicit,
    # never a pass-through.
    "irrelevant": "si_irrelevant",
    # the FULL constitution (sdfbase-llama8b, 2026-09-10): the whole document as a
    # prompted baseline, distinct from the `constitution` EXCERPT above
    "constitution_full": "si_constitution_full",
    # sdf-base-steering (2026-09-04): the two OBJECT-ONLY pseudo-
    # instructions the full-corpus base-fit cells are named through
    # (manifest.PETRI_OBJECT_ONLY_SI_IDS). Registry ids carry hyphens
    # (data.instructions), harness ids cannot (the `__` grammar), so the
    # translation is explicit here and nowhere else. Never a prompted
    # baseline: _traintest_si_text_bytes is never asked for these
    # (parse_cell_id refuses `prompted__constv2_*` first).
    "constv2_mix_full": "constv2-mix-full",
    "constv2ctl2_full": "constv2ctl2-full",
}
TRAINTEST_TO_HARNESS_SI: Dict[str, str] = {
    v: k for k, v in HARNESS_TO_TRAINTEST_SI.items()}

#: registry arm name -> the methods.py OBJECT-KIND string its trained
#: artifact stores (each class's ``kind`` attribute — what ``save()``
#: writes and ``load_object`` dispatches on). Covers exactly the arms the
#: petri matrix audits (defense_cells.CLASS_A_METHODS + CLASS_B_IDS);
#: pinned against the real classes by tests/test_petri_registry_adapter.py
#: and re-verified against every PRESENT artifact's own stored kind.
ARM_OBJECT_KINDS: Dict[str, str] = {
    "psr_kl_all": "psr",
    "psr_mse_all": "psr",
    "vector_all": "vector_all",
    "lora_r1": "lora",
    # lora_r4 = the canonical name of the all-layer low-rank arm (renamed
    # 2026-08-28; the legacy spelling lora_r1_all lives only at the
    # ARTIFACT layer — methods.LEGACY_ARM_ALIASES — never as a harness id).
    # Its artifacts store MultiLayerLoRAObject.kind = "lora_all".
    "lora_r4": "lora_all",
    # psr_xfer_* (2026-08-28): cross-model TRANSFERRED PSR units — the
    # artifacts are ordinary PSRObject blobs (PSRObject.kind = "psr", the
    # same per-layer relu-gated block-output units psr_kl_all stores), so
    # they dispatch through the existing psr loader and lens adapter; the
    # distinct arm names carry the transfer identity only.
    "psr_xfer_from_e2b": "psr",
    "psr_xfer_from_e2b_rw1": "psr",
    "psr_xfer_from_e4b": "psr",
    "psr_xfer_from_e4b_rw1": "psr",
    # sdf-base-steering document-CE arms (2026-09-04): psr_ce_all trains
    # psr_kl_all's exact object (PSRObject.kind = "psr"), lora_r4_ce trains
    # lora_r4's exact object (MultiLayerLoRAObject.kind = "lora_all") — the
    # loss and data differ, the artifact classes do not (methods.ARMS).
    "psr_ce_all": "psr",
    "lora_r4_ce": "lora_all",
    # sdf-base-steering DERIVED arms (round 2, 2026-09-06): the corpus fit
    # MINUS its register-matched control fit, as servable objects.
    # psr_ce_all_minus_ctl_distilled distils the functional residual back
    # into psr_ce_all's own geometry, so its artifact IS a PSRObject blob
    # (kind "psr") and it serves through the existing psr adapter, with
    # only the arm name carrying the derivation identity — the psr_xfer_*
    # precedent exactly. lora_r4_ce_minus_ctl stores BOTH lora_r4_ce
    # parents' factors bit-exact (multilayer.LoRASubtractionObject, kind
    # "lora_subtraction") and serves as that same o_proj translation twice
    # per layer, at coeff +1 / -1.
    "psr_ce_all_minus_ctl_distilled": "psr",
    "lora_r4_ce_minus_ctl": "lora_subtraction",
    # psr_ce_all_minus_ctl (relu-differenced-native-qwen, 2026-09-17): the
    # UNDISTILLED functional residual as extraction/paired_write.
    # PSRPairedWriteObject — kind "psr_paired_write": BOTH parents' (w, b, d) stored
    # bit-exact, served as `+u_pos(z) - u_neg(z)` on the shared pre-write state
    # through the native residual slot's second gate term (op
    # relu_gated_vector_pair). Its own kind, never "psr": the artifact is a
    # two-sided blob no single-gate adapter can read, and the served function
    # is not a one-gate write.
    "psr_ce_all_minus_ctl": "psr_paired_write",
    "loreft": "loreft",
    "lofit": "lofit",
    "token_bos": "token",
    "cast": "cast",
    "antidote": "antidote",
    "alphasteer": "alphasteer",
    # caa_static_r (E45 H2): the CAA / diff-in-means Class B floor. Its
    # artifacts store CaaStaticRObject.kind = "caa_static_r" — a kind of its
    # OWN, not the peer's "static_r": the two save different blobs (this one
    # writes {kind, rs, base_scale, metadata} and records neither
    # steer_positions nor site), and static_r's loader defaults would read it
    # as a one-layer, last-prompt-token block-OUTPUT edit instead of the
    # ten-layer, all-prompt-position resid_pre one it is. Its lens adapter is
    # registered under this kind (serving/lens_adapters._adapt_caa_static_r).
    "caa_static_r": "caa_static_r",
    # arditi (defense_cells.SERVED_ONLY_CLASS_B_IDS): the Arditi refusal direction is
    # staged as a StaticRObject blob — kind "static_r" (one stored direction `rs` at the
    # served block layer, `base_scale`, `steer_positions` prefill_all, `site` pre), served
    # through lens_adapters._adapt_static_r. NOT "caa_static_r" (a different blob; see
    # above). No trainer in this repository writes it, so it is a SERVED-ONLY arm: the
    # explicit-cell registry admits it by name, the Cartesian registry never mints it.
    "arditi": "static_r",
    # arditi_allpos (arditi-qwen-all-positions, 2026-09-16): the SAME fitted direction served in the
    # PAPER'S OWN configuration — the raw difference of means at coefficient c, added at every
    # position of every forward — which is a "vector_all" blob (layers under `vectors`, keyed by the
    # SERVED block layer; `positions` / `fit_positions` "all"; `skip_bos` False; no base_scale),
    # served through lens_adapters._adapt_vector_all at PHASE_BOTH, or natively on the residual
    # slot's constant path. A SECOND id and not a re-dose of `arditi`: the two differ in phase AND
    # in dose unit, so one name for both would alias two interventions everywhere a row is joined.
    # The KIND is what makes it natively servable (native_residual.CONSTANT_KINDS) and what routes
    # its liveness to the logprob instrument (contracts.LOGPROB_LIVENESS_KINDS).
    "arditi_allpos": "vector_all",
    # circuit_breaker is deliberately ABSENT: it left methods.ARMS on
    # 2026-08-20 and is no longer a cell id at all
    # (defense_cells.RETIRED_CLASS_B_IDS), so no arm maps to it. Its KIND
    # survives for hydration (methods._KIND_LOADERS) and the genuine PEFT
    # export (serving/lora_export.GENUINE_EXPORT_KINDS); this table is
    # arm -> kind, and there is no such arm.
}

#: The two serving-path policies a :class:`TrainTestCellSource` can be built
#: under (module docstring, "serving_path"): ``"lens"`` serves every object
#: on the lens hook; ``"lora_native"`` serves every object as a vLLM-native
#: PEFT adapter and refuses, by cell name, any object that cannot be one
#: (``defense_cells.native_lora_unservable_reason``). Closed vocabulary — an
#: unknown policy refuses at construction, never at the first cell.
SERVING_PATH_POLICIES = ("lens", "lora_native")
#: THE ENTRY-POINT DEFAULT (since 2026-09-13): what ``mt_serving_up.py`` /
#: ``capctrl_serving_up.py`` / ``common.sh f22_standup`` / the Modal tinycap
#: launcher stand a view up on when no ``--serving-path`` is given. The
#: DATACLASS default (``TrainTestCellSource.serving_path_policy``) stays
#: ``"lens"`` on purpose: a source is built by a caller that has already
#: decided the path, and every existing construction site (the lens standups,
#: the tinycap object-identity source, the partition's own lens-policy read)
#: keeps its bytes.
DEFAULT_STANDUP_SERVING_PATH = "lora_native"
#: the ``--serving-path`` flag every standup entry point spells, named here so
#: a refusal can tell the operator the exact escape hatch
SERVING_PATH_FLAG = "--serving-path"

_CLASS_OF_PARSED = {"base": "baseline", "prompted": "baseline",
                    "class_a": "A", "class_b": "B"}


class SkippedCell(NamedTuple):
    """One cell a native standup SKIPPED BY NAME: its harness id,
    the object kind the registry maps it to, and the quotable reason
    (:func:`defense_cells.native_serving_disposition`'s). Recorded in
    ``cells_served.json`` / ``endpoint.json`` under ``cells_skipped`` and
    logged once per cell — never a silent drop."""

    cell_id: str
    kind: str
    reason: str


def partition_cells_for_native(harness_cells: Sequence[str], lens_source: Any
                               ) -> Tuple[List[str], List[SkippedCell]]:
    """Split a view's harness cell ids into ``(served, skipped)`` for ONE native
    standup. Pure: it READS kind and resolved spec off
    ``lens_source`` — a LENS-policy :class:`TrainTestCellSource` (or any
    ``CellSource`` whose entries carry their own path) — and exports,
    hashes and serves nothing. Order is preserved.

    Per cell, by :func:`defense_cells.parse_cell_id`:

    - ``none`` and every ``prompted__<si>`` cell -> served (passthrough legs).
    - a STACKED cell -> skipped (the wrapper has no PEFT form), by name,
      without consulting the source (the registry never mints a stacked
      cell).
    - an object cell -> ``lens_source.get(cell)``; a cell the registry LACKS is
      passed through as served (enumeration reports it ABSENT with its reason,
      exactly as today — the partition removes skip-class cells and nothing
      else); a present entry goes by
      :func:`defense_cells.native_serving_disposition` over its kind and
      resolved spec: ``serve`` -> served; ``skip`` -> a :class:`SkippedCell`
      (kind and reason recorded) — a kind with no native form (AlphaSteer,
      antidote, LoReFT, …; the residual-slot kinds psr / vector / vector_all /
      diffmean / static_r SERVE since native-psr-serving)
      OR a position-masked fit of a LoRA kind (``positions`` present and not
      ``"all"``: native multi-LoRA has no position mask, so the cell is
      skipped by name with the mode in its reason, never served at every
      token under its hash); ``refuse`` -> :class:`RegistryAdapterError`
      naming the cell, its kind, the reason and the escape hatch
      (``--serving-path lens``): a LoRA kind whose record the engine cannot
      judge (``positions`` missing, a serve declaration, ``circuit_breaker``
      ``alpha != r``) is never dropped and never guessed at.

    Why a LENS-policy source: a ``lora_native``-policy source REFUSES a
    skip-class cell (an AlphaSteer cell, a position-masked fit) at ``get`` (by
    name, the backstop), before this function could skip it; the lens read
    exports nothing and classifies everything. A source that
    names another policy refuses here. A duplicate id refuses (enumeration
    would refuse the duplicate virtual model later; saying so here is cheaper)."""
    policy = getattr(lens_source, "serving_path_policy", "lens")
    if policy != "lens":
        raise RegistryAdapterError(
            f"partition_cells_for_native needs a LENS-policy cell source (kind and spec are "
            f"read, nothing is exported); this source's serving_path_policy is {policy!r}, "
            "which refuses a skip-class cell at get() before it could be skipped by name")
    served: List[str] = []
    skipped: List[SkippedCell] = []
    seen: set = set()
    for cell_id in harness_cells:
        parsed = dc.parse_cell_id(cell_id)  # loud on a malformed / derived id
        if cell_id in seen:
            raise RegistryAdapterError(
                f"cell {cell_id!r} is named twice in the standup's cell list — one vmodel per "
                "cell; refusing rather than serving a duplicate route")
        seen.add(cell_id)
        if parsed.cell_class in ("base", "prompted"):
            served.append(cell_id)
            continue
        if parsed.cell_class == "stacked":
            disposition, reason = dc.native_serving_disposition(cm.PETRI_STACKED_KIND, None)
            assert disposition == dc.NATIVE_DISPOSITION_SKIP, disposition  # by construction
            skipped.append(SkippedCell(cell_id, cm.PETRI_STACKED_KIND, str(reason)))
            continue
        entry = lens_source.get(cell_id)  # RegistryAdapterError on an inconsistent registry
        if entry is None:
            served.append(cell_id)        # ABSENT downstream, with its reason, as today
            continue
        if entry.cell_id != cell_id:
            raise RegistryAdapterError(
                f"registry entry id {entry.cell_id!r} != requested {cell_id!r} — inconsistent "
                "registry")
        disposition, reason = dc.native_serving_disposition(entry.kind, entry.resolved_spec)
        if disposition == dc.NATIVE_DISPOSITION_SERVE:
            served.append(cell_id)
        elif disposition == dc.NATIVE_DISPOSITION_SKIP:
            skipped.append(SkippedCell(cell_id, str(entry.kind), str(reason)))
        else:
            raise RegistryAdapterError(
                f"cell {cell_id!r} (kind {entry.kind!r}) cannot be served on the native "
                f"multi-LoRA engine: {reason}. Its kind IS natively servable and the refusing "
                "fact is not a position mask (a masked fit would be skipped by name), so this is "
                "a LoRA cell the view meant to serve whose record the engine cannot judge — "
                "refusing the standup rather than dropping, guessing at or lens-serving it; a "
                f"lens standup ({SERVING_PATH_FLAG} lens) serves it as before")
    return served, skipped


# --- THE KIND-AWARE SERVING-PATH DEFAULT (the rule of 2026-09-15) -----------

#: the key every standup record carries the selection under (``standup_jobs.jsonl``,
#: ``cells_served.json`` / ``endpoint.json`` through ``mt_serving_up.py
#: --serving-path-selection``) — additive; a record without it predates the rule
SERVING_PATH_SELECTION_KEY = "serving_path_selection"
#: how the path was chosen: ``explicit`` = the caller set it (``F22_SERVING_PATH`` /
#: ``--serving-path``) and it wins over every auto verdict; ``auto`` = the caller set
#: NONE and the rule below picked it from the launch's steered cells
SERVING_PATH_SELECTION_MODES = ("auto", "explicit")
SERVING_PATH_SELECTION_AUTO, SERVING_PATH_SELECTION_EXPLICIT = SERVING_PATH_SELECTION_MODES
#: the rule's name, recorded so a later reader knows WHICH default picked the path
SERVING_PATH_SELECTION_RULE = "kind-aware-2026-09-15"
#: the ONE log-line prefix (grep this in a standup log to recover the selection)
SERVING_PATH_LOG_PREFIX = "[serving-path]"
#: what the classification READ per cell: ``registry`` = a cell source carrying the
#: objects' resolved specs (the standup's partition sees position masks and malformed
#: records); ``kind_only`` = the KIND by name alone (:func:`kind_only_cell_source` — what
#: the wave shell can read at init, before any venv or bank view exists: a position-masked
#: fit of a LoRA kind reads as natively servable here and is then skipped BY NAME by the
#: standup's own partition, exactly as before)
SPEC_SOURCE_REGISTRY, SPEC_SOURCE_KIND_ONLY = "registry", "kind_only"
#: the reason an explicit selection records
SERVING_PATH_EXPLICIT_REASON = "F22_SERVING_PATH set by caller"


class ServingPathSelection(NamedTuple):
    """ONE standup's serving-path decision, recoverable: the mode, the path, a quotable
    reason, the cells by disposition (``passthrough`` = ``none`` / prompted legs, ``serve`` =
    natively servable steered cells, ``skip`` = lens-only steered cells — None on an
    explicit selection, which classifies nothing), the skipped cells with kind and reason,
    and what the classification read. :meth:`record` is the JSON shape every record
    carries; :attr:`log_line` the ONE line the standup log carries at selection time."""

    mode: str
    selected: str
    reason: str
    cells_by_disposition: Optional[Dict[str, List[str]]]
    skipped: Tuple[SkippedCell, ...]
    spec_source: Optional[str]
    cells_source: Optional[str]

    @property
    def log_line(self) -> str:
        if self.mode == SERVING_PATH_SELECTION_EXPLICIT:
            return f"{SERVING_PATH_LOG_PREFIX} EXPLICIT {self.selected} ({self.reason})"
        return (f"{SERVING_PATH_LOG_PREFIX} AUTO-selected {self.selected}: {self.reason}; "
                "explicit F22_SERVING_PATH would override")

    def record(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "selected": self.selected, "reason": self.reason,
            "cells_by_disposition": (None if self.cells_by_disposition is None
                                     else {k: list(v) for k, v in self.cells_by_disposition.items()}),
            "skipped": [dict(s._asdict()) for s in self.skipped],
            "spec_source": self.spec_source, "cells_source": self.cells_source,
            "rule": SERVING_PATH_SELECTION_RULE,
        }


def kind_only_cell_source(harness_cells: Sequence[str]) -> dc.MappingCellSource:
    """A cell source that knows each cell's KIND by name and nothing else (``resolved_spec``
    None): ``none`` / ``prompted__<si>`` as themselves, a stacked id as the stacked wrapper
    kind, an object cell by its arm through :data:`ARM_OBJECT_KINDS` (an arm the table does
    not map refuses — extend the table deliberately, never guess). What the wave shell can
    read at ``f22_init`` time: no venv, no bank view, no torch — the kind is the ruling's
    unit (\"lens-only kinds\"), and :func:`native_serving_disposition` on a ``None`` spec files
    a lens-only kind (AlphaSteer, antidote, the stacked wrapper) as ``skip`` and a LoRA or
    residual-slot kind as ``serve``."""
    entries: Dict[str, dc.RegistryEntry] = {}
    for cell_id in harness_cells:
        parsed = dc.parse_cell_id(cell_id)  # loud on a malformed id
        if parsed.cell_class == "base":
            entries[cell_id] = dc.RegistryEntry(cell_id=cell_id, kind="none")
        elif parsed.cell_class == "prompted":
            entries[cell_id] = dc.RegistryEntry(cell_id=cell_id, kind="prompted", si_id=parsed.si_id)
        elif parsed.cell_class == "stacked":
            entries[cell_id] = dc.RegistryEntry(cell_id=cell_id, kind=cm.PETRI_STACKED_KIND, si_id=parsed.si_id)
        else:
            kind = ARM_OBJECT_KINDS.get(str(parsed.method))
            if kind is None:
                raise RegistryAdapterError(
                    f"cell {cell_id!r}: arm {parsed.method!r} has no object-kind mapping (mapped: "
                    f"{sorted(ARM_OBJECT_KINDS)}) — the serving path cannot be selected by kind; "
                    "extend ARM_OBJECT_KINDS deliberately, never guess")
            entries[cell_id] = dc.RegistryEntry(cell_id=cell_id, kind=kind, si_id=parsed.si_id)
    return dc.MappingCellSource(entries)


def select_serving_path(harness_cells: Sequence[str], lens_source: Any, *,
                        explicit: Optional[str] = None,
                        cells_source: Optional[str] = None,
                        spec_source: str = SPEC_SOURCE_REGISTRY) -> ServingPathSelection:
    """THE serving-path default, kind-aware (the rule of 2026-09-15: use native
    when possible; select the lens route by itself when a launch's steered cells are all
    lens-only kinds; make the selection recoverable). The ONE rule every shell caller
    (``common.sh f22_init``, ``petri_drive_full22.sh``) reaches through
    ``scripts/sdfbase/full22/serving_path_select.py`` — never a parallel shell predicate.

    ``explicit`` (the caller's ``F22_SERVING_PATH`` / ``--serving-path``) WINS: it is
    validated against :data:`SERVING_PATH_POLICIES` and returned as an ``explicit``
    selection with nothing classified (the whole-view path byte-identical to before).
    Otherwise every cell goes through :func:`partition_cells_for_native` over
    ``lens_source`` — the SAME disposition the standup applies (``serve`` / ``skip`` /
    ``refuse``, :func:`defense_cells.native_serving_disposition`) — and:

    - no steered cell at all (an empty list, or ``none`` / prompted cells only) ->
      ``lora_native``, the entry-point default (a passthrough-only launch);
    - at least one steered cell ``serve`` -> ``lora_native``; the ``skip`` cells are
      skipped by name at the standup exactly as today (recorded here too);
    - EVERY steered cell ``skip`` (lens-only kinds; with a registry source also a
      position-masked fit) -> ``lens``;
    - any ``refuse`` (malformed: ``positions`` missing, a serve declaration, a
      ``circuit_breaker`` with ``alpha != r``), a malformed or duplicate id -> the
      partition's :class:`RegistryAdapterError`, propagated: refuse as today.

    ``cells_source`` names where the cell list came from (a config path, a launcher's
    list) and ``spec_source`` what was read per cell (:data:`SPEC_SOURCE_REGISTRY` /
    :data:`SPEC_SOURCE_KIND_ONLY`); both ride the record."""
    if explicit is not None:
        if explicit not in SERVING_PATH_POLICIES:
            raise RegistryAdapterError(
                f"serving path {explicit!r} is not one of {SERVING_PATH_POLICIES}")
        return ServingPathSelection(SERVING_PATH_SELECTION_EXPLICIT, explicit, SERVING_PATH_EXPLICIT_REASON,
                                    None, (), None, cells_source)
    if spec_source not in (SPEC_SOURCE_REGISTRY, SPEC_SOURCE_KIND_ONLY):
        raise RegistryAdapterError(
            f"spec_source {spec_source!r} is not one of {(SPEC_SOURCE_REGISTRY, SPEC_SOURCE_KIND_ONLY)}")
    cells = list(harness_cells)
    served, skipped = partition_cells_for_native(cells, lens_source)   # RegistryAdapterError = refuse, as today
    passthrough = [c for c in served if dc.parse_cell_id(c).cell_class in ("base", "prompted")]
    serve = [c for c in served if c not in passthrough]
    skip = [s.cell_id for s in skipped]
    by_disposition = {"passthrough": passthrough, "serve": serve, "skip": skip}
    n_steered = len(serve) + len(skip)
    if n_steered == 0:
        if not cells:
            reason = (f"no cell list (source: {cells_source}): " if cells_source else "no cell list: ")
        else:
            reason = "passthrough-only launch (none / prompted cells only, no steered cell): "
        reason += f"the entry-point default {DEFAULT_STANDUP_SERVING_PATH}"
        return ServingPathSelection(SERVING_PATH_SELECTION_AUTO, DEFAULT_STANDUP_SERVING_PATH, reason,
                                    by_disposition, tuple(skipped), spec_source, cells_source)
    if serve:
        skipped_txt = ", ".join(f"{s.cell_id} ({s.kind})" for s in skipped) or "none"
        reason = (f"{len(serve)} of {n_steered} steered cells natively servable; skipped by name: {skipped_txt}")
        return ServingPathSelection(SERVING_PATH_SELECTION_AUTO, "lora_native", reason,
                                    by_disposition, tuple(skipped), spec_source, cells_source)
    tally: Dict[str, int] = {}
    for s in skipped:
        tally[s.kind] = tally.get(s.kind, 0) + 1
    kinds = ", ".join(f"{k}×{n}" for k, n in tally.items())
    reason = f"every steered cell is lens-only ({kinds})"
    return ServingPathSelection(SERVING_PATH_SELECTION_AUTO, "lens", reason,
                                by_disposition, tuple(skipped), spec_source, cells_source)


# --- id translation, both directions ------------------------------------------


def traintest_cell_id(harness_id: str) -> str:
    """The ``defense_cell`` registry id for a canonical harness cell id. A
    malformed harness id refuses via :func:`defense_cells.parse_cell_id`;
    an SI outside the mapped vocabulary refuses naming both sides. A serve
    variant (``@<variant>``) rides through verbatim."""
    parsed = dc.parse_cell_id(harness_id)
    if parsed.cell_class == "base":
        return "none"
    if parsed.cell_class == "class_b":
        return parsed.cell_id  # bare Class B ids coincide by design (variant included)
    tt_si = HARNESS_TO_TRAINTEST_SI.get(parsed.si_id)
    if tt_si is None:
        raise RegistryAdapterError(
            f"cell {harness_id!r}: harness SI {parsed.si_id!r} has no "
            f"TrainTest translation (harness vocabulary "
            f"{sorted(HARNESS_TO_TRAINTEST_SI)} -> TrainTest "
            f"{sorted(TRAINTEST_TO_HARNESS_SI)})")
    if parsed.cell_class == "prompted":
        return f"prompted__{tt_si}"
    # class_a — the serve variant, if any, re-appended unchanged
    return sd.join_variant(f"{tt_si}__{parsed.method}", parsed.serve_variant)


def harness_cell_id(traintest_id: str) -> str:
    """The harness cell id for a registry id — the inverse translation,
    refusing loudly on any registry cell the petri matrix cannot express
    (e.g. the alphakl ladder arms, an unmapped SI): a registry cell must
    never be silently renamed or dropped.

    A serve variant (``safety_refusal__psr_kl_all@thinking``) is split off,
    the BASE translates exactly as before, and the variant is re-appended
    (``si_safety__psr_kl_all@thinking``) — then validated by the harness
    grammar, which admits it on object cells only."""
    if not isinstance(traintest_id, str) or not traintest_id:
        raise RegistryAdapterError(
            f"TrainTest cell id must be a non-empty string, got "
            f"{traintest_id!r}")
    try:
        base, variant = sd.split_variant(traintest_id)
    except sd.ServeDeclarationError as e:
        raise RegistryAdapterError(
            f"TrainTest cell {traintest_id!r}: {e}") from e
    harness = sd.join_variant(_harness_base_cell_id(base, traintest_id),
                              variant)
    # EVERY translation is validated by the harness grammar (the variant
    # rule — object cells only — and, since the object-only SI joined the
    # vocabulary on 2026-09-10, the rule that a corpus pseudo-instruction
    # has no prompted spelling: the Cartesian registry mints a
    # `prompted__constv2-mix-full` row for the sdfbase config, and that row
    # must never translate into a harness id nothing can serve).
    try:
        dc.parse_cell_id(harness)
    except dc.DefenseCellError as e:
        raise RegistryAdapterError(
            f"TrainTest cell {traintest_id!r}: {e}") from e
    return harness


def _harness_base_cell_id(base: str, traintest_id: str) -> str:
    """The variant-free translation; ``traintest_id`` (as given) names the
    offender in every refusal."""
    from sisteer.eval import defense_cell as tt

    if base == "none":
        return "none"
    tt_si = tt.si_of(base)
    arm = tt.arm_of(base)
    if arm is None:  # a prompted baseline (or the bare legacy "prompted")
        if tt_si is None or tt_si not in TRAINTEST_TO_HARNESS_SI:
            raise RegistryAdapterError(
                f"TrainTest cell {traintest_id!r}: SI {tt_si!r} has no "
                f"harness translation (TrainTest vocabulary "
                f"{sorted(TRAINTEST_TO_HARNESS_SI)} -> harness "
                f"{sorted(HARNESS_TO_TRAINTEST_SI)})")
        return f"prompted__{TRAINTEST_TO_HARNESS_SI[tt_si]}"
    if tt_si is None:  # bare Class B (or a served-only Class B id)
        if arm not in dc.CLASS_B_IDS and arm not in dc.SERVED_ONLY_CLASS_B_IDS:
            raise RegistryAdapterError(
                f"TrainTest Class B cell {traintest_id!r} is not in the "
                f"harness's Class B vocabulary {dc.CLASS_B_IDS} (nor the "
                f"served-only ids {dc.SERVED_ONLY_CLASS_B_IDS}) — update "
                "the petri contract deliberately (R5), never drop the cell")
        return arm
    if tt_si not in TRAINTEST_TO_HARNESS_SI:
        raise RegistryAdapterError(
            f"TrainTest cell {traintest_id!r}: SI {tt_si!r} has no harness "
            f"translation (TrainTest vocabulary "
            f"{sorted(TRAINTEST_TO_HARNESS_SI)} -> harness "
            f"{sorted(HARNESS_TO_TRAINTEST_SI)})")
    if arm not in dc.CLASS_A_METHODS:
        raise RegistryAdapterError(
            f"TrainTest Class A cell {traintest_id!r}: method {arm!r} is "
            f"outside the harness's audited set {dc.CLASS_A_METHODS} (e.g. "
            "the alphakl ladder is TrainTest-only) — no harness id exists; "
            "widen defense_cells.CLASS_A_METHODS deliberately (R5) if this "
            "cell should join the audit")
    return f"si_{TRAINTEST_TO_HARNESS_SI[tt_si]}__{arm}"


# --- JSON-safe extraction of an artifact's stored (non-tensor) fields --------

_TENSOR = object()  # sentinel: a tensor-bearing value (bytes are hashed)


def _json_safe(value: Any, where: str) -> Any:
    """``value`` as a JSON-safe structure, the :data:`_TENSOR` sentinel if
    it (or anything inside it) is a tensor payload, or a LOUD refusal for
    any representation this adapter does not understand — an unknown field
    type must never be guessed into the intervention identity."""
    import torch

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise RegistryAdapterError(
                f"{where}: non-finite float {value!r} cannot enter the "
                "resolved spec (canonical_json refuses NaN/inf)")
        return value
    if isinstance(value, torch.Tensor):
        return _TENSOR
    if isinstance(value, (list, tuple)):
        out = [_json_safe(v, f"{where}[{i}]") for i, v in enumerate(value)]
        return _TENSOR if any(v is _TENSOR for v in out) else out
    if isinstance(value, Mapping):
        out_d = {}
        for k, v in value.items():
            if not isinstance(k, (str, int)):
                raise RegistryAdapterError(
                    f"{where}: mapping key {k!r} is neither str nor int — "
                    "refusing to guess a canonical serialization")
            sv = _json_safe(v, f"{where}[{k!r}]")
            if sv is _TENSOR:
                return _TENSOR
            out_d[str(k)] = sv
        return out_d
    raise RegistryAdapterError(
        f"{where}: unhandled stored field type {type(value).__name__} — "
        "refusing to guess how it enters the resolved spec (extend the "
        "adapter deliberately for new representations)")


# --- the CellSource over the defense_cell registry ----------------------------


@dataclass(frozen=True)
class TrainTestCellSource:
    """:class:`defense_cells.CellSource` over the ``DefenseCell``
    registry. ``get`` returns None ONLY for a cell the registry genuinely
    lacks (an unregistered arm — cast/circuit_breaker before their
    ``methods.ARMS`` registration, for instance); every inconsistency
    between the two vocabularies or between the registry and the on-disk
    store refuses loudly."""

    #: TrainTest cell id -> DefenseCell (defense_cell.by_id output).
    cells: Mapping[str, Any]
    #: the RESOLVED compiled-objects dir — ABSOLUTE, from an explicit
    #: override or ``defense_cell.resolve_defenses_dir``; the registry
    #: the ``cells`` mapping came from was built over this SAME dir,
    #: so the object_path cross-check below can never disagree.
    defenses_dir: Path
    #: which serving path every OBJECT entry is registered on
    #: (:data:`SERVING_PATH_POLICIES`, module docstring). The default keeps
    #: every existing construction site — the lens standups, the tinycap
    #: object-identity source — byte-identical.
    serving_path_policy: str = "lens"

    def __post_init__(self) -> None:
        _check_serving_path_policy(self.serving_path_policy)

    def get(self, cell_id: str) -> Optional[dc.RegistryEntry]:
        parsed = dc.parse_cell_id(cell_id)  # loud on malformed ids
        # the BASE TrainTest cell: the registry never mints a serve variant
        # (they are serve-time cells) — a variant re-serves its base's
        # artifact under the overlay _object_entry applies
        tt_id = traintest_cell_id(parsed.base_id)
        tt_cell = self.cells.get(tt_id)
        if tt_cell is None:
            return None  # ABSENT: the registry has no such cell
        self._check_identity(parsed, tt_id, tt_cell)

        if parsed.cell_class == "base":
            return dc.RegistryEntry(cell_id=cell_id, kind=cm.NONE_KIND)
        if parsed.cell_class == "prompted":
            return dc.RegistryEntry(
                cell_id=cell_id, kind=cm.PETRI_PROMPTED_KIND,
                si_id=parsed.si_id,
                si_prefix=_traintest_si_text_bytes(tt_cell.si_id))
        return self._object_entry(parsed, tt_id, tt_cell)

    # -- internals -------------------------------------------------------

    def _check_identity(self, parsed: dc.ParsedCellId, tt_id: str,
                        tt_cell: Any) -> None:
        """The translated cell and the registry cell must agree on class,
        SI, and arm — a disagreement is an inconsistent registry, refused
        before any hashing or serving decision."""
        expected_class = _CLASS_OF_PARSED[parsed.cell_class]
        if tt_cell.class_ != expected_class:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r} (TrainTest {tt_id!r}): registry "
                f"class {tt_cell.class_!r} != the id-derived "
                f"{expected_class!r} — inconsistent registry")
        expected_si = (HARNESS_TO_TRAINTEST_SI[parsed.si_id]
                       if parsed.si_id is not None else None)
        if parsed.cell_class == "class_b":
            expected_si = None
        if tt_cell.si_id != expected_si:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r} (TrainTest {tt_id!r}): registry "
                f"si_id {tt_cell.si_id!r} != the id-derived {expected_si!r} "
                "— an object must never be attributed to the wrong SI")
        if parsed.cell_class in ("class_a", "class_b"):
            expected_arm = parsed.method
            if tt_cell.arm != expected_arm:
                raise RegistryAdapterError(
                    f"cell {parsed.cell_id!r} (TrainTest {tt_id!r}): "
                    f"registry arm {tt_cell.arm!r} != the id-derived "
                    f"{expected_arm!r} — inconsistent registry")

    def _object_entry(self, parsed: dc.ParsedCellId, tt_id: str,
                      tt_cell: Any) -> dc.RegistryEntry:
        from sisteer.eval.defense_cell import object_path_for
        from sisteer.methods import VECTOR_SAFETENSOR_ARMS

        arm = tt_cell.arm
        kind = ARM_OBJECT_KINDS.get(arm)
        if kind is None:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: TrainTest arm {arm!r} has no "
                f"object-kind mapping (mapped: {sorted(ARM_OBJECT_KINDS)}) "
                "— extend ARM_OBJECT_KINDS deliberately, never guess")
        artifact = tt_cell.object_path
        if artifact is None:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r} (TrainTest {tt_id!r}): an object "
                "cell with no registry object_path is an inconsistent "
                "registry (build_registry always resolves one)")
        artifact = Path(artifact)
        expected = object_path_for(self.defenses_dir, tt_id)
        if artifact != expected:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: registry object_path {artifact} "
                f"!= object_path_for's {expected} — inconsistent registry")
        self._check_meta_json(parsed, tt_id, tt_cell, artifact)

        # the <artifact>.json sidecar (manifest.sidecar_path convention):
        # REQUIRED for SteeringVector-container arms (their loader reads
        # it), declared-when-present otherwise so stray sidecar bytes can
        # never be silently ignored.
        sc = cm.sidecar_path(artifact)
        declare_sidecar = arm in VECTOR_SAFETENSOR_ARMS or sc.is_file()
        sidecar: Optional[Path] = sc if declare_sidecar else None

        resolved = self._resolve_spec(parsed, arm, kind, artifact, sidecar)
        if resolved is not None and parsed.serve_variant is not None:
            # THE ONE OVERLAY on the artifact-content-only spec:
            # named in the id, hashed into the identity; object_path,
            # sidecar and the artifact bytes stay the base cell's.
            try:
                resolved = sd.apply_variant(resolved, parsed.serve_variant)
            except sd.ServeDeclarationError as e:
                raise RegistryAdapterError(
                    f"cell {parsed.cell_id!r}: {e}") from e
        return dc.RegistryEntry(
            cell_id=parsed.cell_id, kind=kind,
            si_id=parsed.si_id if parsed.cell_class == "class_a" else None,
            object_path=str(artifact),
            sidecar_path=str(sidecar) if sidecar is not None else None,
            serving_path=self._serving_path_for(parsed, kind, resolved),
            resolved_spec=resolved)

    def _serving_path_for(self, parsed: dc.ParsedCellId, kind: str,
                          resolved: Optional[Mapping[str, Any]]) -> str:
        """The registered serving path under this source's policy (module
        docstring). ``lens``: every object, unconditionally. ``lora_native``:
        ``lora`` iff ``defense_cells.native_lora_unservable_reason`` is None
        for the object's kind and its RESOLVED spec (the variant overlay
        applied — a declared variant is refused there by name); a None
        ``resolved`` (artifact not on disk) is judged by kind alone and the
        cell reports ABSENT downstream. Anything unservable is a loud
        refusal naming cell, kind and reason — a lora-native standup never
        silently drops or lens-serves a cell it cannot serve. A PSR /
        plain-vector cell with a judgeable record passes this predicate —
        the residual slot — instead of refusing."""
        if self.serving_path_policy == "lens":
            return "lens"
        reason = dc.native_lora_unservable_reason(kind, resolved)
        if reason is not None:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: kind {kind!r} cannot be served on "
                f"a lora-native (plugin-free vLLM multi-LoRA) server — "
                f"{reason}. A lora-native standup serves LoRA-only views "
                "(D2), so this cell belongs on a lens standup "
                "(serving_path_policy='lens'); it is refused here rather "
                "than silently dropped or lens-served")
        return "lora"

    def _check_meta_json(self, parsed: dc.ParsedCellId, tt_id: str,
                         tt_cell: Any, artifact: Path) -> None:
        """TrainTest writes a meta.json identity sidecar per cell dir
        (defense_cell.cell_meta). When present it must agree with the
        registry cell — a disagreement means an object landed in the wrong
        cell dir, exactly the mis-attribution hazard this guards, refused."""
        meta_path = artifact.parent / "meta.json"
        try:
            if not meta_path.is_file():
                return
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except OSError:
            return  # unreadable meta -> the artifact checks still govern
        except json.JSONDecodeError as e:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: unparseable {meta_path}: {e} — "
                "refusing to serve an object whose identity record is "
                "corrupt") from e
        from sisteer.eval.defense_cell import legacy_cell_id
        from sisteer.methods import CANONICAL_TO_LEGACY_ARM

        # The ONE accepted mismatch per key: the LEGACY spelling of this
        # same cell/arm (methods.LEGACY_ARM_ALIASES — a lora_r4 cell served
        # from a staged `<si>__lora_r1_all/` dir whose meta records the
        # name the fitting step emitted). Table-driven, so the alias never
        # passes silently; any OTHER value still refuses as
        # mis-attribution.
        accepted_alias = {
            "cell_id": legacy_cell_id(tt_id),
            "arm": CANONICAL_TO_LEGACY_ARM.get(tt_cell.arm or ""),
        }
        for key, expected in (("cell_id", tt_id), ("arm", tt_cell.arm),
                              ("si_id", tt_cell.si_id),
                              ("class", tt_cell.class_)):
            if key in meta and meta[key] != expected:
                alias = accepted_alias.get(key)
                if alias is not None and meta[key] == alias:
                    continue  # the recorded legacy spelling of the SAME cell
                raise RegistryAdapterError(
                    f"cell {parsed.cell_id!r}: {meta_path} records "
                    f"{key}={meta[key]!r} but the registry cell says "
                    f"{expected!r} — an object in the wrong cell dir (D10 "
                    "mis-attribution); refusing")

    def _resolve_spec(self, parsed: dc.ParsedCellId, arm: str, kind: str,
                      artifact: Path, sidecar: Optional[Path]
                      ) -> Optional[Mapping[str, Any]]:
        """The authoritative resolution from the artifact's OWN stored
        content — None when the artifact (or a declared sidecar) is
        missing/unreadable (no object, no resolution: enumeration reports
        ABSENT from the missing component); a loud refusal when content is
        present but malformed. Dispatch is by the ARM's declared storage
        format (methods.VECTOR_SAFETENSOR_ARMS), with the artifact's
        actual content verified against it — a mis-saved artifact would
        mis-dispatch in load_object and must refuse, never hash.

        Artifact content ONLY — with ONE overlay, applied by the caller:
        a serve variant named in the cell id (``@<variant>``) adds its
        hashed serve declaration (``serve_declaration.apply_variant``),
        so the same bytes served as a declared transfer cell are a
        distinct identity."""
        from sisteer.methods import VECTOR_SAFETENSOR_ARMS, _is_safetensors_file

        try:
            if not artifact.is_file():
                return None
            if sidecar is not None and not sidecar.is_file():
                return None
            artifact_bytes = artifact.read_bytes()
            sidecar_text = (sidecar.read_text(encoding="utf-8")
                            if sidecar is not None else None)
        except OSError:
            return None  # unreadable -> ABSENT (enumerate re-reads + reports)

        is_safetensors = _is_safetensors_file(artifact)
        if arm in VECTOR_SAFETENSOR_ARMS:
            if not is_safetensors:
                raise RegistryAdapterError(
                    f"cell {parsed.cell_id!r}: arm {arm!r} saves a "
                    f"SteeringVector safetensors container but {artifact} "
                    "is not one — a mis-saved artifact would mis-dispatch "
                    "in load_object; refusing")
            return self._resolve_safetensors(parsed, arm, kind, artifact,
                                             sidecar, sidecar_text)
        if is_safetensors:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: arm {arm!r} saves a torch blob "
                f"but {artifact} is a safetensors container — a mis-saved "
                "artifact would mis-dispatch in load_object; refusing")
        return self._resolve_torch_blob(parsed, arm, kind, artifact,
                                        artifact_bytes)

    def _resolve_safetensors(self, parsed: dc.ParsedCellId, arm: str,
                             kind: str, artifact: Path,
                             sidecar: Optional[Path],
                             sidecar_text: Optional[str]
                             ) -> Mapping[str, Any]:
        """A SteeringVector-container artifact: the JSON sidecar IS the
        loader's stored spec (SteeringVector.load reads method/layers/
        metadata from it; load_object dispatches on the stored method)."""
        if sidecar is None or sidecar_text is None:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: SteeringVector-container arm "
                f"{arm!r} reached spec resolution without its required "
                "sidecar — adapter bug")
        try:
            stored = json.loads(sidecar_text)
        except json.JSONDecodeError as e:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: unparseable SteeringVector "
                f"sidecar {sidecar}: {e}") from e
        if not isinstance(stored, dict):
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: sidecar {sidecar} is not a JSON "
                "object")
        method = stored.get("method")
        if method != kind:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: sidecar method {method!r} != the "
                f"arm's object kind {kind!r} (ARM_OBJECT_KINDS[{arm!r}]) — "
                "load_object would hydrate a different intervention than "
                "this registry claims; refusing")
        spec = {k: _json_safe(v, f"sidecar[{k!r}]") for k, v in stored.items()}
        spec.update(kind=kind, arm=arm)
        if parsed.cell_class == "class_a":
            spec["si_id"] = parsed.si_id
        return spec

    def _resolve_torch_blob(self, parsed: dc.ParsedCellId, arm: str,
                            kind: str, artifact: Path,
                            artifact_bytes: bytes) -> Mapping[str, Any]:
        """A torch-blob artifact: the blob's stored non-tensor fields are
        the exact configuration its ``load_blob`` consumes; tensor payloads
        are omitted here because their bytes are hashed verbatim."""
        import io

        import torch

        try:
            blob = torch.load(io.BytesIO(artifact_bytes),
                              map_location="cpu", weights_only=True)
        except Exception as e:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: artifact {artifact} exists but "
                f"is not a loadable torch blob: {e} — a corrupt artifact is "
                "refused, never hashed into a servable identity") from e
        if not isinstance(blob, dict) or "kind" not in blob:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: artifact {artifact} is not a "
                "torch blob with a 'kind' field (the load_object contract)")
        if blob["kind"] != kind:
            raise RegistryAdapterError(
                f"cell {parsed.cell_id!r}: artifact stores kind "
                f"{blob['kind']!r} but the adapter maps arm {arm!r} to "
                f"{kind!r} (ARM_OBJECT_KINDS) — vocabulary drift between "
                "TrainTest's trainer and this map; fix the map "
                "deliberately, never here-and-now")
        spec: Dict[str, Any] = {}
        for key, value in blob.items():
            if key == "kind":
                continue
            safe = _json_safe(value, f"blob[{key!r}]")
            if safe is _TENSOR:
                continue  # tensor payloads are hashed as artifact_bytes
            spec[key] = safe
        spec.update(kind=kind, arm=arm)
        if parsed.cell_class == "class_a":
            spec["si_id"] = parsed.si_id
        return spec


def _check_serving_path_policy(policy: Any) -> str:
    """The closed :data:`SERVING_PATH_POLICIES` vocabulary, refused loudly
    at construction (a typo must never fall through to a default path)."""
    if policy not in SERVING_PATH_POLICIES:
        raise RegistryAdapterError(
            f"serving_path_policy must be one of {list(SERVING_PATH_POLICIES)}, "
            f"got {policy!r}")
    return policy


def _traintest_si_text_bytes(tt_si_id: str) -> bytes:
    """The exact SI text bytes for a TrainTest SI id, via TrainTest's own
    resolver (``eval.jailbreak.resolve_instruction`` — the SAFETY_SI module
    constant for ``safety_refusal``, the instruction registry otherwise)."""
    from sisteer.eval.jailbreak import resolve_instruction

    try:
        instruction = resolve_instruction(tt_si_id)
    except KeyError as e:
        raise RegistryAdapterError(
            f"SI {tt_si_id!r} does not resolve through TrainTest's "
            f"instruction registry: {e}") from e
    text = instruction.text
    if not isinstance(text, str) or not text:
        raise RegistryAdapterError(
            f"SI {tt_si_id!r} resolved to empty/non-string text — the SI "
            "bytes ARE the prompted intervention (plan §5 H4); refusing")
    return text.encode("utf-8")


# --- construction from the two-SI config (the petri_run wiring) ---------------

TWO_SI_CONFIG = Path("configs") / "two_si.yaml"


def _resolve_defenses_dir(cfg: Mapping[str, Any], *, repo_root: Path,
                          override: Optional[Path],
                          config_source: Path) -> Path:
    """The compiled-objects dir this adapter enumerates — ABSOLUTE, always.

    1. an explicit ``override`` wins (a test's tmp store, a launch-time
       ``defenses_dir=`` override).
    2. otherwise DELEGATE to ``defense_cell.resolve_defenses_dir`` — the
       ONE resolution of the objects dir, shared by the object writers
       (``train_two_si_objects`` / ``fit_class_b``) and readers — when
       this branch has it.
    3. otherwise REFUSE loudly. Never the config's relative default: that
       is the per-commit immutable checkout, a directory the writers no
       longer write, and enumerating it would report every object cell
       ABSENT forever — even after training lands — which is
       indistinguishable from 'not trained yet'. This adapter also never
       re-implements that resolution locally: a duplicate would drift
       from the writers and silently disagree about where objects live.
    """
    if override is not None:
        return Path(override).expanduser().absolute()
    from sisteer.eval import defense_cell as tt

    resolver = getattr(tt, "resolve_defenses_dir", None)
    if callable(resolver):
        return Path(resolver(cfg, repo_root=repo_root)).absolute()
    raise RegistryAdapterError(
        "cannot resolve the compiled-objects dir: the authoritative "
        "resolver sisteer.eval.defense_cell.resolve_defenses_dir — "
        "TrainTest's ONE resolution of the objects dir, shared by the "
        "object writers (train_two_si_objects / fit_class_b) and readers "
        "(D58) — has not landed on this branch, and no explicit "
        "defenses_dir override was given. The writers now place objects "
        "in an ABSOLUTE, commit-independent directory (exported as "
        "TT_TWO_SI_DEFENSES_DIR in the two-SI sbatch bodies), so falling "
        f"back to the two-SI config's defenses_dir="
        f"{cfg['defenses_dir']!r} ({config_source}) would enumerate a "
        "directory the writers never write: every object cell would "
        "report ABSENT forever, even after training lands. Merge "
        "TrainTest's defense_cell change (so this adapter can delegate) "
        "or pass defenses_dir= explicitly; this adapter will not "
        "duplicate the D58 resolution (a duplicate would silently drift "
        "from the writers).")


def traintest_cell_source(spec: cm.PetriSpec, *, repo_root: Path,
                          config_path: Optional[Path] = None,
                          defenses_dir: Optional[Path] = None,
                          serving_path_policy: str = "lens",
                          ) -> TrainTestCellSource:
    """Build the real CellSource from the two-SI config (the submission-
    time record of the matrix: si_ids, class arms). Loud refusals when
    the config is missing/malformed, when its SI vocabulary is not the
    one this adapter maps, or when its model pin disagrees with the petri
    manifest's — never a silent default.

    ``serving_path_policy`` (:data:`SERVING_PATH_POLICIES`; module
    docstring) is the standup's serving path for every object cell — the
    lens standups take the default; a lora-native standup passes
    ``"lora_native"`` and gets a by-name refusal for any cell that cannot
    be a native adapter. Validated FIRST, before the config is read.

    The compiled-objects dir is NOT the config's relative default: an
    explicit ``defenses_dir`` wins, else TrainTest's own
    ``resolve_defenses_dir`` resolves it, else this refuses loudly
    (:func:`_resolve_defenses_dir`)."""
    import yaml

    from sisteer.eval import defense_cell as tt

    _check_serving_path_policy(serving_path_policy)
    repo_root = Path(repo_root)
    path = Path(config_path) if config_path is not None else (
        repo_root / TWO_SI_CONFIG)
    if not path.is_file():
        raise RegistryAdapterError(
            f"two-SI config {path} is missing — the DefenseCell registry "
            "derives from it (si_ids, defenses_dir, class arms); refusing "
            "to enumerate against nothing")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RegistryAdapterError(f"unparseable two-SI config {path}: {e}"
                                   ) from e
    if not isinstance(cfg, dict) or "defenses_dir" not in cfg:
        raise RegistryAdapterError(
            f"two-SI config {path} is not a mapping with a defenses_dir — "
            "refusing to guess the object store location")

    # the audit and the object store must pin the SAME model: serving
    # another model's objects would be silent wrongness. Checked FIRST —
    # before the SI vocabulary — because a wrong model is the more
    # fundamental refusal, whatever the cells say (and since the
    # object-only SI joined the harness vocabulary, 2026-09-10, a manifest
    # naming every canonical cell puts an SI in play that most two-SI
    # configs never declare: that drift must not mask a model mismatch).
    for cfg_key, spec_val, spec_key in (
            ("model", spec.target_model, "target_model"),
            ("model_revision", spec.target_revision, "target_revision")):
        if cfg_key in cfg and cfg[cfg_key] != spec_val:
            raise RegistryAdapterError(
                f"two-SI config {path} pins {cfg_key}={cfg[cfg_key]!r} but "
                f"the petri manifest pins {spec_key}={spec_val!r} — the "
                "audit must serve the objects' own base model; refusing")

    explicit = tt.EXPLICIT_CELLS_KEY in cfg
    if explicit:
        # An EXPLICIT-cell config: the registry is exactly the enumerated
        # cell ids, so the SI vocabulary in play is whatever those ids
        # name — checked below against the manifest's in-play SIs;
        # ``si_ids`` is refused next to explicit_cells by
        # registry_from_explicit_cells itself.
        si_ids = tuple(dict.fromkeys(
            tt.si_of(c) for c in cfg[tt.EXPLICIT_CELLS_KEY]
            if isinstance(c, str) and tt.si_of(c) is not None))
    else:
        si_ids = tuple(cfg.get("si_ids", tt.DEFAULT_SI_IDS))
    # Every IN-PLAY mapped SI must be declared by the config (a missing or
    # renamed one is real vocabulary drift, refused loudly); EXTRA
    # TrainTest-only SIs are fine — TrainTest may carry more than the audit
    # maps (the same precedent as its TrainTest-only arms, e.g. the alphakl
    # ladder). "In play" = the SIs the spec's OWN cells name (prompted /
    # Class A / stacked): since si_irrelevant joined the mapped vocabulary
    # (2026-08-28) the full map spans three SIs, and requiring every config
    # to declare all three would wrongly refuse a deliberately narrower
    # serving config whose cells never touch the missing SI — a cell whose
    # SI the config omits already reports ABSENT from the registry, which
    # is the correct per-cell outcome, not a source-wide refusal.
    in_play = []
    for cell_id in spec.cells:
        p_cell = dc.parse_cell_id(cell_id)
        if p_cell.si_id is not None and p_cell.si_id not in in_play:
            in_play.append(p_cell.si_id)
    expected = tuple(HARNESS_TO_TRAINTEST_SI[si] for si in in_play)
    missing = [si for si in expected if si not in si_ids]
    if missing:
        raise RegistryAdapterError(
            f"two-SI config {path} declares si_ids {list(si_ids)} but the "
            f"manifest's cells put the harness SIs {in_play} in play, "
            f"mapping onto {list(expected)} (missing: {missing}) — "
            "vocabulary drift between the configs; update "
            "HARNESS_TO_TRAINTEST_SI / the config deliberately (R5)")

    ddir = _resolve_defenses_dir(cfg, repo_root=repo_root,
                                 override=defenses_dir, config_source=path)
    # Build the registry DIRECTLY over the one resolved dir —
    # ``build_registry`` takes the dir as-is and never re-resolves, so the
    # dir resolution runs exactly ONCE (in _resolve_defenses_dir above),
    # never re-entered via registry_from_cfg with its own output fed back
    # as cfg["defenses_dir"]. The cfg -> kwargs mapping mirrors
    # registry_from_cfg's exactly (si_ids above; class arm lists here,
    # module defaults when absent); the drift guard in
    # tests/test_petri_registry_adapter.py pins the two builds identical
    # over an absolute-defenses_dir cfg, so a new cfg key that
    # registry_from_cfg honors but this mirroring misses fails loudly
    # instead of being silently ignored. Both sides of _object_entry's
    # object_path cross-check share this single ddir, so they can never
    # disagree about the store.
    if explicit:
        # The explicit registry validates every listed cell against the
        # STAGED bank (artifact + budget sidecar + the shared identity
        # verdict) — the same seam every other consumer of these configs
        # uses (registry_from_cfg). It resolves the dir through
        # resolve_defenses_dir itself, so the ONE resolution above is
        # handed in as an absolute defenses_dir and must come back
        # unchanged: a TT_TWO_SI_DEFENSES_DIR export pointing elsewhere is
        # a refusal here, never two stores silently disagreeing.
        cfg_abs = {k: v for k, v in cfg.items() if k != "defenses_dir"}
        cfg_abs["defenses_dir"] = str(ddir)
        resolved_again = Path(tt.resolve_defenses_dir(cfg_abs, repo_root=repo_root))
        if resolved_again.resolve() != Path(ddir).resolve():
            raise RegistryAdapterError(
                f"explicit-cell config {path}: the objects dir resolved to "
                f"{ddir} but TrainTest's resolve_defenses_dir (the "
                f"TT_TWO_SI_DEFENSES_DIR export, D58) yields {resolved_again} "
                "— one wave, one bank; align the export with --defenses-dir")
        registry = tt.registry_from_explicit_cells(cfg_abs, repo_root=repo_root)
        return TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=ddir,
                                   serving_path_policy=serving_path_policy)
    registry = tt.build_registry(
        ddir, si_ids=si_ids,
        class_a_arms=tuple(cfg.get("class_a_arms", tt.CLASS_A_ARMS)),
        class_b_arms=tuple(cfg.get("class_b_arms", tt.CLASS_B_ARMS)))
    return TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=ddir,
                               serving_path_policy=serving_path_policy)
