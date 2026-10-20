"""Petri defense-cell registry adapter.

Maps the audit's DefenseCell ids onto served legs: for each cell it
resolves ``(kind, si_id, object_path, virtual_model_name,
serving_identity)`` and the three separated keys
(``intervention_hash`` / ``harness_config_hash`` / ``petri_run_key``),
derives the path-matched no-op CONTROL LEGS (§3), and enforces the
two-tier pairing semantics (``check_pairing``).

THE ``DefenseCell`` CONTRACT. The registry lives in
``sisteer.eval.defense_cell`` and the real adapter is
``sisteer.capeval.petri.registry_adapter`` — it translates the two id
vocabularies (harness ``si_safety__<m>`` ↔ registry
``safety_refusal__<m>``, ``prompted__constitution`` ↔
``prompted__si_constitution``, …) and refuses loudly on anything it cannot
map. The injectable :class:`CellSource` below remains the seam (tests use
a synthetic one). The contract as the harness consumes it:

- **Cell ids** (the 22 canonical cells of the two-SI audit):
  - ``none`` — the undefended base.
  - ``prompted__safety`` / ``prompted__constitution`` — base weights + the
    verified SI prefix.
  - ``si_<si>__<method>`` — the 16 Class A prompt-compiled objects,
    ``<si> ∈ {safety, constitution}``, ``<method>`` one of
    :data:`CLASS_A_METHODS` (the two-SI spec's 8 prompt-compiling arms,
    e.g. ``si_constitution__psr_kl_all``).
  - ``antidote`` / ``alphasteer`` / ``circuit_breaker`` — the 3 Class B
    SI-agnostic baselines (:data:`CLASS_B_IDS`).
  A landed registry whose ids differ from this grammar is a LOUD REFUSAL
  at enumeration time (update the constants deliberately), never a silent
  skip (R5).
- **Serve variants**: an OBJECT cell id
  (Class A / Class B / stacked) may carry ONE ``@<variant>`` suffix from
  the closed vocabulary ``serving.serve_declaration.SERVE_VARIANTS``
  (``@thinking``, ``@all_turns``, ``@all_positions``) — the SAME artifact
  served under a hashed, cell-level serve declaration that the resolved
  spec carries and the intervention hash covers, so
  ``si_safety__psr_kl_all@thinking`` is a distinct served identity from
  ``si_safety__psr_kl_all``, with its own vmodel
  (``petri__si_safety__psr_kl_all@thinking``). ``none``, the prompted
  cells and the derived control legs refuse the suffix. Variant cells are
  SERVE-TIME cells: the cell registry never mints them; the adapter
  resolves the BASE entry and overlays the variant
  (``registry_adapter.TrainTestCellSource.get``), and enumeration refuses
  an entry whose name and spec disagree about the declaration.
- **object_path scheme**: the trainer's on-disk ``(si_id, arm)`` /
  ``(arm)`` cell dir or artifact blob, hydrated by ``methods.load_object``.
  At rebase, the real adapter validates each object via ``load_object`` at
  registry-export time and supplies the RESOLVED defense spec
  (``RegistryEntry.resolved_spec``) plus the artifact/sidecar paths; the
  intervention hash covers the artifact bytes, the sidecar bytes, and the
  resolved spec — nothing else.
- **serving_path**: the cell source's SERVING-PATH POLICY decides it
  (``registry_adapter.TrainTestCellSource(serving_path_policy=...)``):
  ``"lens"`` — every lens standup,
  the default — puts EVERY object on the lens hook, byte-identical to
  before the policy existed; ``"lora_native"`` — a plugin-free vLLM
  multi-LoRA standup (LoRA-only views run lora-native, PSR views stay
  on the lens, mixed servers never) — marks an object ``lora`` iff
  :func:`native_lora_unservable_reason` is None for its kind and resolved
  spec (:data:`NATIVE_LORA_SERVABLE_KINDS`, an all-position fit, no serve
  declaration, unit LoRA scaling) and REFUSES BY NAME otherwise — a PSR,
  position-masked or stacked cell is never silently dropped or lens-served
  on a lora-native table. ``enumerate_cells(lens_controls=False)`` (a table
  for a server with NO lens hook) re-checks every ``lora`` route against
  the same predicate; a STACKED cell marked ``lora`` refuses on every
  table. A kind that fails the §10 gate's canary on EVERY committed vLLM
  path is a gate FAILURE that blocks the audit — there is no HF-shim
  serving fallback and no derived ``*__hf`` leg; the HF shim survives only
  as the gate's fp32 reference oracle.

ABSENT vs REFUSAL: a cell that is missing from the registry or
whose object artifact is missing/unreadable is reported **ABSENT** with its
reason (the completeness map lists it; it is never silently dropped). A
**malformed id**, an inconsistent registry entry, or an unfrozen spec is a
loud refusal — those are harness bugs, not missing data.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, List, Mapping, Optional, Protocol, Sequence,
                    Tuple)

from sisteer.capeval import manifest as cm
from sisteer.capeval.engine_identity import identity_diff
from sisteer.capeval.serving import serve_declaration as sd


class DefenseCellError(ValueError):
    """A defense-cell refusal (malformed id, inconsistent registry entry,
    duplicate virtual model, …) — loud, never a default."""


class PairingError(ValueError):
    """A pairing refusal: the two legs may not be compared."""


# --- The expected cell id vocabulary ---------------------------------------

BASE_CELL_ID = "none"
#: the 8 Class A prompt-compiling methods (two-SI spec table). A landed
#: registry with a different set refuses loudly at parse time — update
#: deliberately at rebase (R5).
CLASS_A_METHODS = (
    "psr_kl_all", "psr_mse_all", "vector_all", "lora_r1",
    "loreft", "lofit", "token_bos", "cast",
    # lora_r4 (2026-08-28): the all-layer low-rank LoRA arm of the scale-12
    # campaign, under its CANONICAL name (the staged objects are rank 4,
    # "all" is implicit). Only the canonical name is a harness id — the
    # legacy spelling lora_r1_all is translated at the ARTIFACT layer
    # (methods.LEGACY_ARM_ALIASES / defense_cell.object_path_for), never
    # parsed as a cell id, so one method can never wear two harness
    # identities. Appended LAST (R5: a deliberate widening of the audited
    # set).
    "lora_r4",
    # psr_xfer_* (2026-08-28): the cross-model TRANSFERRED PSR units
    # (kind 'psr', same hook site as psr_kl_all;
    # registry_adapter.ARM_OBJECT_KINDS). Distinct names carry the
    # transfer identity; _rw1 = ridge-ablation variant. Appended LAST
    # (R5: a deliberate widening of the audited set).
    "psr_xfer_from_e2b",
    "psr_xfer_from_e2b_rw1",
    "psr_xfer_from_e4b",
    "psr_xfer_from_e4b_rw1",
    # psr_ce_all / lora_r4_ce (sdf-base-steering, 2026-09-04): the
    # document-CE arms — psr_kl_all's exact object (PSRObject, kind 'psr')
    # and lora_r4's exact object (MultiLayerLoRAObject, kind 'lora_all')
    # trained by hard-target LM CE on document pools; served at their
    # recorded fit leg (positions=all) like any Class A cell. Named
    # through the OBJECT-ONLY SIs (manifest.PETRI_OBJECT_ONLY_SI_IDS).
    # Appended LAST (R5: a deliberate widening of the audited set).
    "psr_ce_all",
    "lora_r4_ce",
    # the sdf-base-steering DERIVED arms (round 2, 2026-09-06): the
    # decomposition products of the full-corpus fit MINUS its
    # register-matched control fit — `<corpus SI>__psr_ce_all_minus_ctl_
    # distilled` (a PSRObject, kind 'psr') and `<corpus SI>__lora_r4_ce_
    # minus_ctl` (a LoRASubtractionObject, kind 'lora_subtraction').
    # Class A in exactly the sense the other corpus arms are: named
    # through an OBJECT-ONLY pseudo-instruction (manifest.
    # PETRI_OBJECT_ONLY_SI_IDS — parse_cell_id already refuses a
    # `prompted__`/`stacked__` spelling over those), one object per corpus
    # cell, served at the recorded fit leg (positions=all). Their objects
    # are never TRAINED — methods.ARMS refuses both — which changes
    # nothing here: the harness audits what a cell SERVES.
    # Appended LAST (R5: a deliberate widening of the audited set).
    "psr_ce_all_minus_ctl_distilled",
    "lora_r4_ce_minus_ctl",
    # psr_ce_all_minus_ctl (relu-differenced-native-qwen, 2026-09-17): the
    # UNDISTILLED functional residual — the corpus fit MINUS its register-matched
    # control fit as the PAIRED WRITE `relu(w1.z+b1) d1 - relu(w2.z+b2) d2` on
    # the shared pre-write state (extraction/paired_write.PSRPairedWriteObject,
    # kind 'psr_paired_write'; registry_adapter.ARM_OBJECT_KINDS), served on the
    # native residual slot's second gate term. A NEW id, never a relabel of the
    # distilled cell: the distilled student is a one-gate PSRObject fit to
    # approximate this; this is the thing itself. Object-only like its siblings.
    # Appended LAST (R5: a deliberate widening of the audited set).
    "psr_ce_all_minus_ctl",
)
#: Class A methods whose objects are staged ONLY under an object-only
#: pseudo-instruction (``manifest.PETRI_OBJECT_ONLY_SI_IDS``): the serving
#: grammar admits them (``CLASS_A_METHODS`` above — a cell id, a launcher
#: token, a route), but they have NO prompt-SI cells and are NOT legs of any
#: prompt-SI hypothesis. Appending the derived arm to CLASS_A_METHODS alone
#: also widens petri_stats' FIXED confirmatory family
#: (constitution-vs-safety contrasts, the max-T band, the Holm slots, the
#: per-SI aggregate denominators) 13 -> 14 with legs that never exist, so a
#: re-analysed 13-method audit loses its simultaneous band and aggregates.
#: The statistical family is :data:`PROMPTABLE_CLASS_A_METHODS`; this tuple
#: names what it excludes.
OBJECT_ONLY_METHODS = (
    # the sdf-base-steering corpus arms (document-CE fits 2026-09-01, the two
    # DERIVED arms 2026-09-06; united with main's port at the math500 merge,
    # 2026-09-11): every one is staged under a corpus pseudo-instruction only
    # (configs/capeval/manifests/petri_full22_*.yaml), never under a prompt SI
    "psr_ce_all", "lora_r4_ce", "psr_ce_all_minus_ctl_distilled",
    "lora_r4_ce_minus_ctl",
    # the paired write (2026-09-17): staged under the corpus pseudo-instruction only
    "psr_ce_all_minus_ctl")
#: THE FIXED confirmatory family — the prompt-SI Class A methods, in
#: CLASS_A_METHODS order (13 as of 2026-09-10). Every statistical family in
#: petri_stats is over THIS tuple: the constitution-vs-safety contrasts, the
#: pre-registered max-T band, the Holm slots, the arm-vs-prompted references,
#: the per-SI class aggregates. Object-only methods are enumerated separately
#: (``petri_stats.object_only_cell_ids``) and never enter it.
PROMPTABLE_CLASS_A_METHODS = tuple(m for m in CLASS_A_METHODS
                                   if m not in OBJECT_ONLY_METHODS)
assert set(OBJECT_ONLY_METHODS) <= set(CLASS_A_METHODS)
#: THE Class B tuple — the 3 SI-agnostic baselines (two-SI spec table), and
#: the ONE place they are written down (E45 H2). This module is torch-free, so
#: it can be the authority for all three consumers: the trainer's registry
#: (``sisteer.eval.defense_cell.CLASS_B_ARMS`` imports THIS tuple under its own
#: historical name), the harness id vocabulary (``parse_cell_id`` /
#: ``registry_adapter.harness_cell_id``, here), and the tinycap launcher's
#: grammar (``tinycap.contracts.launchable_class_b_ids``, which runs in a bare
#: modal venv and could never import the trainer's torch-importing copy). Two
#: hand-kept copies is what E40 H2 found and what E45 H2 found again in the
#: other direction: the sets disagreed, so ``caa_static_r`` — a real
#: trainer-side arm — was unlaunchable while ``circuit_breaker`` parsed and
#: then died after an H200 had been allocated. One tuple cannot disagree with
#: itself.
CLASS_B_IDS = ("antidote", "alphasteer", "caa_static_r")
#: Class B ids that were audited once and are not cells any more:
#: ``circuit_breaker`` left ``methods.ARMS`` entirely on 2026-08-20, so the
#: registry builds no cell for it and a standup naming it would enumerate
#: it ABSENT after the GPU was allocated. Kept BY NAME so a refusal can
#: state why instead of reading as an unknown id. This is about cell IDS
#: only — its fitted artifacts stay hydratable (``methods._KIND_LOADERS``)
#: and its object kind stays exportable
#: (``serving/lora_export.GENUINE_EXPORT_KINDS``).
RETIRED_CLASS_B_IDS = ("circuit_breaker",)
#: Class B ids SERVED from an external, pre-staged bank on this tree with NO trainer here
#: (math-eight-arms-extension plan P8): ``arditi`` — the Arditi refusal direction (arXiv 2406.11717),
#: fitted by ``scripts/arditi/fit_arditi.py`` and staged under
#: ``exp/objects/arditi_strength/<target>/strength/cells_view/<view>/arditi`` (a ``static_r`` blob:
#: one direction, prefill-only, the dose baked into the bytes). The id parses as ``class_b`` — the harness
#: serves it exactly like ``alphasteer`` (one object per model, SI-agnostic, no SI prefix) — but it is
#: deliberately NOT in :data:`CLASS_B_IDS`: that tuple also defines the trainer's Cartesian registry
#: (``defense_cell.CLASS_B_ARMS``), ``fit_class_b``'s arm list and petri_stats' Class B family, none of
#: which may widen for an object no trainer on this tree produces. Consumers that must admit it name it
#: here: ``parse_cell_id`` (this module), ``registry_adapter.harness_cell_id`` / ``ARM_OBJECT_KINDS``,
#: ``defense_cell.registry_from_explicit_cells`` and the launcher grammar
#: (``tinycap.contracts.launchable_class_b_ids``).
#:
#: ``arditi_allpos`` (arditi-qwen-all-positions, 2026-09-16) is the SAME refusal direction served in
#: the PAPER'S OWN configuration: the RAW difference of means at coefficient ``c``, added at every
#: position of every forward (prefill AND every decode step), staged under
#: ``exp/objects/arditi_allpos/<target>/ladder/cells_view/<view>/arditi_allpos`` as a ``vector_all``
#: blob. It is a SECOND ID rather than a re-dose of the first because the two differ in PHASE and in
#: DOSE UNIT at once: ``arditi`` is prefill-only at a fraction of the residual norm — a DECLARED
#: deviation — and this one is neither. Aliasing them would put two interventions under one name
#: across the export's supersession mechanisms and the Petri intervention-hash check, so a row could
#: no longer say which it measured. Same reason as ``arditi`` for being served-only: no trainer on
#: this tree mints either.
SERVED_ONLY_CLASS_B_IDS = ("arditi", "arditi_allpos")
#: every virtual model name is petri__<...> (one vmodel per served leg).
VMODEL_PREFIX = "petri__"
#: control-leg id suffixes (derived legs — never legal in spec.cells).
_CONTROL_SUFFIXES = ("lens0", "lora0")
#: the derived hook-armed scale-0 control leg (§3) — the one id the
#: serving modules (proxy, lens_steer) special-case.
NOOP_LENS_CELL_ID = "none__lens0"
#: the derived native-multi-LoRA zero-adapter (B=0) control leg (§3) — the
#: ONE constant for the id (it was a bare literal at the derivation site
#: until native-multilora-serving). It MUST equal
#: ``serving.lora_export.ZERO_ADAPTER_NAME``, the adapter name/dir the zero
#: export writes and the frontend serves it under; this module is torch-free
#: and ``lora_export`` imports torch, so the two are pinned equal by tests
#: (``tests/test_lora_native_enumeration.py``,
#: ``tests/test_native_lora_constants_agree.py``), never by import.
NOOP_LORA_CELL_ID = "none__lora0"
#: object KINDS a plugin-free vLLM multi-LoRA server can serve as a REAL PEFT
#: adapter — ``serving.lora_export.GENUINE_EXPORT_KINDS`` (the exporter's
#: dispatch table) spelled here torch-free, and pinned equal to it by test.
#: Kind is NECESSARY, not sufficient: :func:`native_lora_unservable_reason`
#: is the per-SPEC predicate (fit positions, serve declarations, scaling).
NATIVE_LORA_SERVABLE_KINDS = ("circuit_breaker", "lora", "lora_all",
                              "lora_subtraction",
                              "psr", "vector", "vector_all", "diffmean", "static_r",
                              # THE PAIRED WRITE (relu-differenced-native-qwen): the two-gate
                              # kind, served on the residual slot alone (the lens has no adapter)
                              "psr_paired_write")
#: the RESIDUAL-SLOT kinds among them: served at
#: the decoder block OUTPUT through the native residual slot (a rank-1 ZERO
#: o_proj LoRA carries the routing, the payload rides beside it), never as an
#: o_proj LoRA — so the o_proj rules (positions present + ``all``, unit scaling)
#: do not apply; what is judged is the object's NATIVE POSITION MODE
#: (:func:`native_position_mode`). Pinned equal to
#: ``serving.native_residual.RESIDUAL_SLOT_KINDS`` by test (this module is
#: torch-free).
RESIDUAL_SLOT_KINDS = ("psr", "vector", "vector_all", "diffmean", "static_r", "psr_paired_write")
#: the position modes the residual slot knows (``native_residual.POSITION_MODES``,
#: pinned equal by test): ``all`` = every token (stage 1); ``response`` = the
#: assistant header + every generated position; ``prompt_last`` = the last
#: prompt token; ``prefill_all`` = every prompt token at the block INPUT
#: (``static_r`` ``site='pre'``, one boundary earlier). NOT the lens's
#: ``served_positions_of`` vocabulary.
NATIVE_POSITION_MODES = ("all", "response", "prompt_last", "prefill_all")
#: stage 2 (the per-token position class) IS landed: EVERY
#: mode the lens names is served natively on a residual kind, so the
#: ``position_mode_stage2`` SKIP retires — no residual cell is skipped for its
#: mode any more, while a mode OUTSIDE the vocabulary still refuses by name
#: (``residual_record``), never a silent all-position serve. While this was
#: False (stage 1) a non-``all`` residual cell was skipped by name and the lens
#: served it. Pinned equal to ``native_residual.STAGE2_POSITION_CLASSES`` by
#: test — the worker refuses to LOAD a restricted payload while its own flag is
#: False, so the two arm together; tests pin both worlds by monkeypatching this.
NATIVE_STAGE2_ARMED = True

_PROMPTED_PREFIX = "prompted__"
_SI_ID_PREFIX = "si_"
#: stacked prompt+object cells: ``stacked__<si>__<method>`` — the SI
#: prompt ``prompted__<si>``'s exact prefix served TOGETHER with a
#: steered object: the same-SI Class A object ``si_<si>__<method>``
#: when ``<method>`` is a Class A method, or the SI-agnostic Class B
#: baseline ``<method>`` when it is a Class B id (the two sets are
#: disjoint, so the id parses uniquely). Cross-SI stacks (one SI's
#: prompt over the other SI's compiled object) are deliberately not in
#: the grammar — adding them is a recorded decision, not a parser edit.
_STACKED_PREFIX = "stacked__"

CELL_CLASSES = ("base", "prompted", "class_a", "class_b", "stacked",
                "control")


@dataclass(frozen=True)
class ParsedCellId:
    """A validated canonical cell id."""

    cell_id: str  # the FULL id as given (serve variant included)
    cell_class: str  # base | prompted | class_a | class_b | stacked
    si_id: Optional[str]  # safety | constitution | None (base, Class B)
    method: Optional[str]  # Class A method or Class B id; None otherwise
    #: the serve variant named by an ``@<variant>`` suffix
    #: (``serve_declaration.SERVE_VARIANTS``), or None — object cells only.
    serve_variant: Optional[str] = None

    @property
    def base_id(self) -> str:
        """The id without its serve variant — the artifact-bearing cell a
        variant re-serves (``cell_id`` itself when there is none)."""
        return _split_variant(self.cell_id)[0]


#: the cell classes a ``@<variant>`` suffix is legal on — OBJECT cells only:
#: a variant re-serves a fitted artifact under a declaration, so there is
#: nothing for it to mean on ``none``, a prompted cell, or a derived control
#: leg (which inherits its donor's spec).
_VARIANT_CELL_CLASSES = ("class_a", "class_b", "stacked")


def _split_variant(cell_id: str) -> Tuple[str, Optional[str]]:
    """``serve_declaration.split_variant`` in this module's refusal
    vocabulary: a malformed or unknown variant is a malformed cell id."""
    try:
        return sd.split_variant(cell_id)
    except sd.ServeDeclarationError as e:
        raise DefenseCellError(f"malformed cell id {cell_id!r}: {e}") from e


def parse_cell_id(cell_id: Any) -> ParsedCellId:
    """Validate a CANONICAL cell id against the grammar above.
    Malformed ids refuse loudly — including derived control-leg ids,
    which are never manifest inputs.

    A serve variant (``<object cell>@<variant>``, module docstring) is split
    off FIRST and validated against the closed vocabulary; the base is then
    parsed exactly as before and the variant re-attached on
    :attr:`ParsedCellId.serve_variant` — legal on object cells only. ``@``
    never occurred in a cell id before the grammar gained it, so no
    existing id changes meaning."""
    if not isinstance(cell_id, str) or not cell_id:
        raise DefenseCellError(f"cell id must be a non-empty string, got "
                               f"{cell_id!r}")
    base, variant = _split_variant(cell_id)
    parsed = _parse_base_cell_id(base, cell_id)
    if variant is None:
        return parsed
    if parsed.cell_class not in _VARIANT_CELL_CLASSES:
        raise DefenseCellError(
            f"cell id {cell_id!r}: a serve variant ({sd.VARIANT_SEP}{variant}) "
            f"is legal on OBJECT cells only ({_VARIANT_CELL_CLASSES}), never "
            f"on a {parsed.cell_class!r} cell — a variant re-serves a fitted "
            "artifact under a hashed serve declaration; 'none' and the "
            "prompted cells serve no artifact, and control legs inherit "
            "their donor's spec (fit-support-transfer plan §1.2)")
    return dataclasses.replace(parsed, cell_id=cell_id, serve_variant=variant)


def _parse_base_cell_id(base: str, cell_id: str) -> ParsedCellId:
    """The variant-free grammar. ``base`` is parsed; ``cell_id`` (the id as
    given, variant included) names the offender in every refusal."""
    if base == BASE_CELL_ID:
        return ParsedCellId(base, "base", None, None)
    if base in CLASS_B_IDS or base in SERVED_ONLY_CLASS_B_IDS:
        return ParsedCellId(base, "class_b", None, base)
    parts = base.split("__")
    if parts[-1] in _CONTROL_SUFFIXES:
        raise DefenseCellError(
            f"cell id {cell_id!r} names a path-matched control leg — control "
            "legs are DERIVED by enumerate_cells (plan §3), never manifest "
            "inputs")
    if base.startswith(_STACKED_PREFIX):
        rest = base[len(_STACKED_PREFIX):]
        si, sep, method = rest.partition("__")
        if not sep or si not in cm.PETRI_SI_IDS:
            raise DefenseCellError(
                f"malformed stacked cell id {cell_id!r}: expected "
                f"'stacked__<si>__<method>' with <si> in {cm.PETRI_SI_IDS}")
        if si in cm.PETRI_OBJECT_ONLY_SI_IDS:
            raise DefenseCellError(
                f"cell id {cell_id!r} stacks a prompted prefix over the "
                f"OBJECT-ONLY SI {si!r} — a document-corpus pseudo-"
                "instruction has no system-prompt text; only "
                f"'si_{si}__<method>' is legal for it")
        if si in cm.PETRI_PROMPTED_ONLY_SI_IDS:
            raise DefenseCellError(
                f"cell id {cell_id!r} stacks over the PROMPTED-ONLY SI {si!r} — a "
                f"prompted baseline with no fitted object; only 'prompted__{si}' "
                "is legal for it")
        if method not in CLASS_A_METHODS and method not in CLASS_B_IDS:
            raise DefenseCellError(
                f"malformed stacked cell id {cell_id!r}: method {method!r} "
                f"is neither a Class A method ({CLASS_A_METHODS}) nor a "
                f"Class B baseline ({CLASS_B_IDS})")
        return ParsedCellId(base, "stacked", si, method)
    if base.startswith(_PROMPTED_PREFIX):
        si = base[len(_PROMPTED_PREFIX):]
        if si not in cm.PETRI_SI_IDS:
            raise DefenseCellError(
                f"malformed prompted cell id {cell_id!r}: SI {si!r} not in "
                f"{cm.PETRI_SI_IDS}")
        if si in cm.PETRI_OBJECT_ONLY_SI_IDS:
            raise DefenseCellError(
                f"prompted cell id {cell_id!r} names the OBJECT-ONLY SI "
                f"{si!r} — a document-corpus pseudo-instruction is never a "
                f"system prompt; only 'si_{si}__<method>' is legal for it")
        return ParsedCellId(base, "prompted", si, None)
    if base.startswith(_SI_ID_PREFIX):
        head, sep, method = base.partition("__")
        si = head[len(_SI_ID_PREFIX):]
        if sep and si in cm.PETRI_PROMPTED_ONLY_SI_IDS:
            raise DefenseCellError(
                f"cell id {cell_id!r} names the PROMPTED-ONLY SI {si!r} — a prompted "
                f"baseline with no fitted object; only 'prompted__{si}' is legal for it")
        if not sep or si not in cm.PETRI_SI_IDS:
            raise DefenseCellError(
                f"malformed Class A cell id {cell_id!r}: expected "
                f"'si_<si>__<method>' with <si> in {cm.PETRI_SI_IDS}")
        if method not in CLASS_A_METHODS:
            raise DefenseCellError(
                f"malformed Class A cell id {cell_id!r}: method {method!r} "
                f"not in the expected TrainTest set {CLASS_A_METHODS} — if "
                "the landed registry legitimately differs, update "
                "CLASS_A_METHODS deliberately (R5), never here-and-now")
        return ParsedCellId(base, "class_a", si, method)
    raise DefenseCellError(
        f"malformed cell id {cell_id!r}: expected 'none', "
        f"'prompted__<si>', 'si_<si>__<method>', or one of "
        f"{CLASS_B_IDS + SERVED_ONLY_CLASS_B_IDS}")


def stacked_object_cell_id(parsed: ParsedCellId) -> str:
    """The OBJECT leg's canonical cell id for a stacked cell: the
    same-SI Class A object, or the SI-agnostic Class B baseline. What a
    gate serves as the 'object-only single' of the two-sided identity
    assertion."""
    if parsed.cell_class != "stacked":
        raise DefenseCellError(
            f"stacked_object_cell_id: {parsed.cell_id!r} is "
            f"{parsed.cell_class!r}, not a stacked cell")
    if parsed.method in CLASS_B_IDS:
        base = parsed.method
    else:
        base = f"{_SI_ID_PREFIX}{parsed.si_id}__{parsed.method}"
    # the object leg is what a serve variant re-serves, so it carries the
    # variant; the prompt leg (stacked_prompted_cell_id) never does
    return sd.join_variant(base, parsed.serve_variant)


def stacked_prompted_cell_id(parsed: ParsedCellId) -> str:
    """The PROMPT leg's canonical cell id for a stacked cell — the
    prompt-only single of the two-sided identity assertion."""
    if parsed.cell_class != "stacked":
        raise DefenseCellError(
            f"stacked_prompted_cell_id: {parsed.cell_id!r} is "
            f"{parsed.cell_class!r}, not a stacked cell")
    return f"{_PROMPTED_PREFIX}{parsed.si_id}"


def stacked_resolved_spec(si_id: str, si_prefix: bytes,
                          object_resolved: Mapping[str, Any]
                          ) -> Dict[str, Any]:
    """The stacked cell's WRAPPER resolved spec — the ONE construction
    shared by enumeration, the worker-registry export, and any shim, so
    the intervention hash can never be computed over two different
    shapes. The prefix rides as its sha16: the SI text stays out of
    worker registries while the hash still binds the exact bytes."""
    # PROMPTABLE SIs only: a stack's prompt leg IS a system prompt, and an
    # object-only pseudo-instruction (cm.PETRI_OBJECT_ONLY_SI_IDS) has none —
    # the grammar (parse_cell_id) refuses `stacked__<object-only>__…` and
    # this constructor must not be able to build what the grammar cannot
    # name
    if si_id not in cm.PETRI_PROMPTABLE_SI_IDS:
        raise DefenseCellError(
            f"stacked_resolved_spec: si_id {si_id!r} not in the promptable "
            f"SIs {cm.PETRI_PROMPTABLE_SI_IDS}"
            + (" (an OBJECT-ONLY pseudo-instruction has no system-prompt "
               "text to stack)" if si_id in cm.PETRI_OBJECT_ONLY_SI_IDS
               else ""))
    if not si_prefix:
        raise DefenseCellError(
            "stacked_resolved_spec: empty si_prefix bytes — the prompt "
            "leg IS half the intervention; a stacked cell without it is "
            "a harness bug, never a valid stack")
    if not isinstance(object_resolved, Mapping) or not object_resolved:
        raise DefenseCellError(
            "stacked_resolved_spec: the object leg needs its own "
            "non-empty resolved spec (the authoritative load_object "
            "resolution — plan §5 H4)")
    obj = dict(object_resolved)
    okind = obj.get("kind")
    if (not isinstance(okind, str) or not okind
            or okind in (cm.NONE_KIND, cm.PETRI_PROMPTED_KIND,
                         cm.PETRI_STACKED_KIND)):
        raise DefenseCellError(
            f"stacked_resolved_spec: object leg kind {okind!r} must be a "
            "defense-object kind — never none/prompted, and never a "
            "nested stack")
    return {"kind": cm.PETRI_STACKED_KIND, "si_id": si_id,
            "si_prefix_sha16": cm.sha256_hex16(bytes(si_prefix)),
            "object": obj}


def served_positions_of(resolved_spec: Any) -> Any:
    """The position MODE a cell's resolved spec asks to SERVE
    (``serve_declaration.POSITIONS_KEY``) — for a STACKED spec the OBJECT
    leg's (``resolved_spec["object"]["positions"]``; the wrapper
    :func:`stacked_resolved_spec` builds carries none of its own), exactly
    as ``lens_steer._assert_ops_express_spec_positions`` unwraps it — or
    ``None`` when the spec states none (the served mode then defaults to
    the artifact's own, which only the worker's adapter reads:
    ``lens_adapters._served_positions``). Never an exception: a non-mapping
    spec, or a stacked wrapper with no object leg, reads ``None``.

    THE ONE UNWRAPPER, beside the one wrapper: ``registry_shim``'s marker
    gates (which header / terminator a build must carry) and the proxy's
    routing-table stamp (``proxy.lens_served_positions`` — the fact the
    thinking-ON gate reads a cell's header-locatedness from) both decide
    from THIS, so no two readers can unwrap a stacked cell two ways (the
    2026-09-07 sixth review's P2 was a stacked all-turns condition read as
    response-only off the top-level key). Torch-free on purpose: the proxy
    imports no adapter module."""
    if not isinstance(resolved_spec, Mapping):
        return None
    if resolved_spec.get("kind") == cm.PETRI_STACKED_KIND:
        obj = resolved_spec.get("object")
        return _served_positions_of_leg(obj) if isinstance(obj, Mapping) else None
    return _served_positions_of_leg(resolved_spec)


#: The two DEPLOYMENT-SHAPE vocabularies that state a served mask without the ``positions`` key (math-eight-arms-extension
#: plan: the first thinking-ON serving of these objects on the eight-arm launcher): a ``static_r`` blob (the Arditi refusal
#: direction; lens_adapters._adapt_static_r) states ``steer_positions`` — ``prefill_all`` = the edit at EVERY prompt row,
#: ``prompt_last`` = at the LAST prompt row — and an ``alphasteer`` blob (lens_adapters._adapt_alphasteer) states
#: ``steer_positions: prefill`` = every prompt row. Read as the served mode the thinking-ON gate classifies (``all`` /
#: ``last``: every row / the last row BY CONSTRUCTION — no header is matched, no thinking_on fact is read, so an undeclared
#: ON render can mis-mask nothing), exactly what those ops do at prefill; the adapters never read this map (they build their
#: masks from ``steer_positions`` themselves), so the STAMP moves and the served intervention does not. A value outside the
#: table reads None — fail closed, as before, never a silent ``all``.
SERVED_MODE_BY_STEER_POSITIONS: Mapping[str, Mapping[str, str]] = {
    "static_r": {"prefill_all": "all", "prompt_last": "last"},
    "alphasteer": {"prefill": "all"},
}
STEER_POSITIONS_KEY = "steer_positions"


def _served_positions_of_leg(spec: Mapping[str, Any]) -> Any:
    """One (unwrapped) leg: the stated ``positions`` when present, else the deployment-shape derivation of
    :data:`SERVED_MODE_BY_STEER_POSITIONS` for the two kinds that carry ``steer_positions``, else None."""
    stated = spec.get(sd.POSITIONS_KEY)
    if stated is not None:
        return stated
    table = SERVED_MODE_BY_STEER_POSITIONS.get(spec.get("kind"))
    if table is None:
        return None
    return table.get(spec.get(STEER_POSITIONS_KEY))


# --- The injectable registry interface ---------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    """What the cell registry supplies per canonical cell. Produced by
    ``registry_adapter.TrainTestCellSource`` from the landed
    ``DefenseCell`` registry (module docstring); tests build synthetic
    ones."""

    cell_id: str
    #: "none" | "prompted" | an object kind (methods.py vocabulary)
    kind: str
    si_id: Optional[str] = None
    #: object cells only: the on-disk artifact whose BYTES enter the
    #: intervention hash. Missing/unreadable -> the cell reports ABSENT.
    object_path: Optional[str] = None
    #: object cells only: optional JSON sidecar next to the artifact;
    #: absent sidecar hashes as the EXPLICIT b"" (a stated input).
    sidecar_path: Optional[str] = None
    #: object cells only: "lens" | "lora" (the registered serving path).
    serving_path: Optional[str] = None
    #: prompted AND stacked cells: the exact SI text bytes (the prompt
    #: leg of the intervention). EXCLUSIVITY is enforced at enumeration,
    #: two-sided: a prompted entry must carry NO object_path, a plain
    #: object entry must carry NO si_prefix, and a STACKED entry
    #: ('stacked__<si>__<method>') must carry BOTH — the one cell class
    #: where prefix and object legitimately coexist.
    si_prefix: Optional[bytes] = None
    #: object cells only: the RESOLVED defense spec (post-load_object
    #: defaults applied — every effective field: layers, scale, converter
    #: version, …). Must carry kind == .kind. REQUIRED whenever the
    #: object artifact (and any declared sidecar) is PRESENT: a None then
    #: refuses loudly at enumeration — the adapter must supply the
    #: authoritative load_object resolution, never a guessed minimal spec
    #: (two loaders applying different resolved interventions under one
    #: intervention_hash is a serving-invariance violation).
    #: A cell whose artifact/sidecar is missing has no resolution to give
    #: (nothing exists to resolve) and reports ABSENT from the missing
    #: component before this refusal is ever reached.
    resolved_spec: Optional[Mapping[str, Any]] = None


class CellSource(Protocol):
    """The injectable registry interface. The REAL implementation is
    ``registry_adapter.TrainTestCellSource`` over the landed
    DefenseCell registry; tests inject synthetic sources."""

    def get(self, cell_id: str) -> Optional[RegistryEntry]:
        """The entry for a canonical cell id, or None if the registry has
        no such cell (-> ABSENT, never an exception)."""
        ...


@dataclass(frozen=True)
class MappingCellSource:
    """A CellSource over a plain {cell_id: RegistryEntry} mapping."""

    entries: Mapping[str, RegistryEntry]

    def get(self, cell_id: str) -> Optional[RegistryEntry]:
        return self.entries.get(cell_id)


# --- Resolved cells ----------------------------------------------------------


@dataclass(frozen=True)
class PetriCell:
    """One SERVED LEG: a canonical cell or a derived control leg, fully
    resolved — vmodel, serving identity, and the three keys."""

    cell_id: str
    cell_class: str  # base | prompted | class_a | class_b | control
    kind: str
    si_id: Optional[str]
    object_path: Optional[str]
    virtual_model_name: str
    serving_identity: str  # <backend>+<path>+<mode>
    intervention_hash: str
    harness_config_hash: str
    petri_run_key: str
    #: control legs: the canonical cell this leg is the path-matched no-op
    #: of. None on canonical cells.
    control_of: Optional[str] = None
    #: prompted cells: sha256_hex16 of the exact SI-prefix bytes — carried
    #: on the DISTINCT prompted vmodel so the proxy can verify the
    #: client-injected prefix per request (H8).
    si_prefix_hash: Optional[str] = None


@dataclass(frozen=True)
class AbsentCell:
    """A canonical cell that could not be served — reported, never
    silently dropped."""

    cell_id: str
    reason: str


@dataclass(frozen=True)
class PetriCellSet:
    """enumerate_cells' output: canonical cells, derived control legs, and
    the ABSENT report."""

    canonical: Tuple[PetriCell, ...]
    controls: Tuple[PetriCell, ...]
    absent: Tuple[AbsentCell, ...]

    def all_served(self) -> Tuple[PetriCell, ...]:
        return self.canonical + self.controls

    def virtual_model_map(self) -> Dict[str, PetriCell]:
        """vmodel -> cell, for the proxy's routing table. A duplicate
        vmodel is a refusal (one vmodel per served leg, H8)."""
        out: Dict[str, PetriCell] = {}
        for cell in self.all_served():
            if cell.virtual_model_name in out:
                raise DefenseCellError(
                    f"duplicate virtual model {cell.virtual_model_name!r} "
                    f"({out[cell.virtual_model_name].cell_id!r} vs "
                    f"{cell.cell_id!r})")
            out[cell.virtual_model_name] = cell
        return out


def _vmodel(cell_id: str, cell_class: str) -> str:
    if cell_class == "base":
        return VMODEL_PREFIX + "base"
    # prompted cells keep their DISTINCT names petri__prompted__<si> (H8);
    # objects and control legs are petri__<cell_id> — a serve variant's
    # vmodel therefore carries its suffix (petri__si_safety__psr_kl_all@thinking).
    return VMODEL_PREFIX + cell_id


def virtual_model_name(cell_id: str) -> str:
    """The virtual model the proxy serves a CANONICAL cell id as — the ONE
    mapping :func:`enumerate_cells` writes into every route entry
    (``_vmodel``): the base route is ``petri__base`` (never
    ``petri__none``), a prompted cell ``petri__prompted__<si>``, an object
    ``petri__<cell id>``. Public so the launcher-side contracts derive a
    cell's spellings through this mapping instead of re-deriving the prefix
    rule (which is how ``none`` was once spelled ``petri__none`` there).
    Refuses a malformed id (:func:`parse_cell_id`)."""
    return _vmodel(cell_id, parse_cell_id(cell_id).cell_class)


def _check_variant_expressed(cell_id: str, variant: Optional[str],
                             resolved: Mapping[str, Any]) -> None:
    """The id's serve variant and the resolved spec's declarations must
    AGREE: the variant is visible in the NAME so vmodels, transcript
    dirs and manifests cannot confuse the settings, and hashed in the
    SPEC so the served identity differs — a spec that declares what its
    name does not (or the reverse) is an inconsistent registry entry,
    refused before hashing."""
    try:
        declared = sd.declarations_of(resolved)
    except sd.ServeDeclarationError as e:
        raise DefenseCellError(
            f"cell {cell_id!r}: malformed serve declaration in "
            f"resolved_spec: {e}") from e
    if variant is None:
        if declared:
            raise DefenseCellError(
                f"cell {cell_id!r}: resolved_spec carries serve "
                f"declaration(s) {declared} but the id names no serve "
                "variant — a declaration must be visible in the cell's NAME "
                f"(spell the cell '<id>{sd.VARIANT_SEP}<variant>', one of "
                f"{sorted(sd.SERVE_VARIANTS)}); a declaration the name does "
                "not carry is an inconsistent registry entry")
        return
    expected = sd.variant_overlay(variant)
    mismatched = {k: (resolved.get(k), v) for k, v in expected.items()
                  if resolved.get(k) != v}
    if mismatched:
        raise DefenseCellError(
            f"cell {cell_id!r}: the id names serve variant {variant!r} but "
            "the resolved_spec does not express it — (got, expected) per "
            f"key: {mismatched}; the adapter applies the variant with "
            "serve_declaration.apply_variant, so a spec that lacks it was "
            "not built for this cell (inconsistent registry entry)")


#: the two ``static_r`` deployment vocabularies (``lens_adapters._adapt_static_r``)
_STATIC_R_STEER_POSITIONS = ("prompt_last", "prefill_all")
_STATIC_R_SITES = ("post", "pre")


def _mode_header(kind: str, mode: Any, header: Any, *, header_required: bool
                 ) -> Tuple[str, Optional[int]]:
    """Normalise ``(mode, header_len)`` for one residual record: the mode must be
    a native mode; ``response`` needs a positive header width when the kind
    records one (psr / diffmean / vector_all record their fit header; a
    ``static_r`` cross-phase cell locates the header from the served template
    at export, so its width may be absent here)."""
    if not isinstance(mode, str) or mode not in NATIVE_POSITION_MODES:
        raise DefenseCellError(
            f"kind {kind!r}: position mode {mode!r} is not a native residual-slot mode "
            f"({list(NATIVE_POSITION_MODES)}); the lens serves it, the slot refuses to guess")
    if mode != "response":
        return mode, None
    if header is None:
        if header_required:
            raise DefenseCellError(
                f"kind {kind!r}: position mode 'response' but the record carries no "
                "response_header_len — the served region starts at the assistant header, "
                "whose width the fit recorded; a record without it cannot be judged")
        return mode, None
    if isinstance(header, bool) or not isinstance(header, int) or header < 1:
        raise DefenseCellError(
            f"kind {kind!r}: response_header_len={header!r} is not a positive int")
    return mode, int(header)


def native_position_mode(kind: str, resolved_spec: Mapping[str, Any], *,
                         header_required: bool = True) -> Tuple[str, Optional[int]]:
    """The NATIVE POSITION MODE of one residual-kind record and its response
    header width — the fact stage assignment is
    made on (``all`` -> stage 1; anything else -> stage 2). Read from the
    object's OWN fields, per kind, exactly as its loader / lens adapter reads
    them; a record that cannot be judged raises :class:`DefenseCellError`:

    - ``psr`` / ``diffmean``: the blob's FIT leg ``positions`` (missing -> ``all``,
      the loaders' own default: ``PSRObject.load_blob`` / ``DiffMeanObject.
      load_blob`` and the lens's ``_fit_leg_positions`` all read ``blob.get(
      "positions", "all")``); ``response`` needs ``response_header_len``;
    - ``vector_all``: the DEPLOYMENT ``positions`` (default ``all``) with the FIT
      leg ``fit_positions`` cross-checked — a masked fit served at every
      position is the fit=response / serve=all corner (a measured
      model-destroyer, ``_adapt_psr``'s docstring) and refuses; ``response``
      takes its header from ``fit_response_header_len`` (a response fit) or
      ``response_header_len``;
    - ``vector``: the sidecar's ``positions`` (a manifest-level override) and
      ``metadata.vector_object.positions`` (what ``VectorObject.save`` persists)
      must agree when both are present; default ``all``;
    - ``static_r``: ``steer_positions`` (default ``prompt_last``) and ``site``
      (default ``post``) validated against the lens vocabulary; a spec
      ``positions == "response"`` is the cross-phase Arditi cell -> ``response``;
      any other stated ``positions`` refuses (the lens would ignore it).

    ``header_required`` (default True — the exporter's and stage 2's reading):
    a ``response`` mode on a kind that records its fit header must carry the
    width. The DISPOSITION passes ``NATIVE_STAGE2_ARMED``: while stage 2 is
    unarmed the mode alone decides the skip and a response record without its
    width is still skipped by name (the width is needed to SERVE, which stage 1
    never does); once armed, the record must be complete or it refuses.

    ``resolved_spec`` must be a mapping (an absent artifact has no mode)."""
    if kind not in RESIDUAL_SLOT_KINDS:
        raise DefenseCellError(
            f"native_position_mode: kind {kind!r} is not a residual-slot kind "
            f"({list(RESIDUAL_SLOT_KINDS)})")
    if not isinstance(resolved_spec, Mapping):
        raise DefenseCellError(
            f"kind {kind!r}: native_position_mode needs the resolved spec, got "
            f"{type(resolved_spec).__name__}")
    spec = resolved_spec
    if kind in ("psr", "diffmean", "psr_paired_write"):
        # psr_paired_write: the fitted artifact's `positions` (always "all" — PSRPairedWriteObject
        # refuses masked parents and its blob writes "all"), read exactly as psr's FIT leg is
        return _mode_header(kind, spec.get(sd.POSITIONS_KEY, "all"),
                            spec.get("response_header_len"), header_required=header_required)
    if kind == "vector_all":
        mode = spec.get(sd.POSITIONS_KEY, "all")
        fit = spec.get("fit_positions", "all")
        if fit not in ("all", "response"):
            raise DefenseCellError(
                f"kind 'vector_all': fit_positions={fit!r} is not a fit leg ('all' / 'response')")
        if fit != "all" and mode == "all":
            raise DefenseCellError(
                "kind 'vector_all': the object was fit at 'response' positions (fit_positions) "
                "but the record deploys it at every position — the fit=response / serve=all "
                "corner is a different intervention from the one that was fit; refusing")
        header = (spec.get("fit_response_header_len") if fit == "response"
                  else spec.get("response_header_len"))
        return _mode_header(kind, mode, header, header_required=header_required)
    if kind == "vector":
        stated = spec.get(sd.POSITIONS_KEY)
        meta = spec.get("metadata")
        vo = meta.get("vector_object") if isinstance(meta, Mapping) else None
        persisted = vo.get(sd.POSITIONS_KEY) if isinstance(vo, Mapping) else None
        if stated is not None and persisted is not None and stated != persisted:
            raise DefenseCellError(
                f"kind 'vector': the sidecar states positions={stated!r} but its "
                f"vector_object record persists positions={persisted!r} — the object's two "
                "records disagree about its deployment; refusing to pick one")
        mode = stated if stated is not None else (persisted if persisted is not None else "all")
        return _mode_header(kind, mode, spec.get("response_header_len"),
                            header_required=header_required)
    # static_r
    steer = spec.get(STEER_POSITIONS_KEY, "prompt_last")
    site = spec.get("site", "post")
    if steer not in _STATIC_R_STEER_POSITIONS:
        raise DefenseCellError(
            f"kind 'static_r': unknown steer_positions {steer!r} (expected one of "
            f"{list(_STATIC_R_STEER_POSITIONS)}) — the mask is the intervention; refusing to "
            "guess it")
    if site not in _STATIC_R_SITES:
        raise DefenseCellError(
            f"kind 'static_r': unknown hook site {site!r} (expected 'pre' or 'post') — the site "
            "decides which block's output carries the edit")
    stated = spec.get(sd.POSITIONS_KEY)
    if stated is not None:
        if stated != "response":
            raise DefenseCellError(
                f"kind 'static_r': stated positions={stated!r} — the lens honours only "
                "positions='response' (the cross-phase cell) on this kind and would serve "
                "steer_positions otherwise; refusing an ambiguous record")
        return _mode_header(kind, "response", spec.get("response_header_len"),
                            header_required=False)
    return _mode_header(kind, steer, None, header_required=False)


def native_lora_unservable_reason(
        kind: str, resolved_spec: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Why an object of ``kind`` with this RESOLVED spec cannot be served as
    a vLLM-native PEFT adapter — ``None`` when it can. The ONE predicate
    behind both the cell source's
    ``lora_native`` policy (``registry_adapter._object_entry``) and the
    enumeration-time re-check of every ``lora`` route on a lora-native table
    (:func:`enumerate_cells` with ``lens_controls=False``). Every reason is a
    sentence a refusal can quote; the caller prefixes the cell id.

    Judged, in order:

    - the STACKED wrapper kind (``manifest.PETRI_STACKED_KIND``) — never
      (no stacked SI-plus-LoRA cell is served in this mode).
    - the kind must be in :data:`NATIVE_LORA_SERVABLE_KINDS` (a LoReFT, AlphaSteer,
      antidote, … object is a lens hook with no native form; the residual-slot
      kinds — psr, vector, vector_all, diffmean, static_r — are servable since
      native-psr-serving).
    - ``resolved_spec is None`` — no artifact on disk, so there is nothing
      to judge beyond the kind: returns ``None`` (the cell reports ABSENT
      downstream; nothing is served). A present artifact always has a
      resolution (enumeration refuses otherwise), so the spec-level checks
      below always run for anything that could reach a server.
    - a SERVE DECLARATION (``serve_declaration.declarations_of`` non-empty —
      an ``@thinking`` / ``@all_turns`` / ``@all_positions`` variant): the
      lens path enforces those declarations (render, positions); a native
      adapter has no declaration surface, and the position variants REPLACE
      ``positions`` in the spec, so judging the overlaid ``positions`` would
      admit a response-fit object the exporter refuses from its bytes.
      Refused before ``positions`` is read, so the positions rule below only
      ever sees the FIT leg.
    - a RESIDUAL-SLOT kind (:data:`RESIDUAL_SLOT_KINDS`): the o_proj rules
      below do not apply (the write is at the block output, routed per token);
      judged instead by :func:`native_position_mode` — a record it cannot judge
      (an unknown mode, a response mode without its header width, a masked
      ``vector_all`` fit deployed at every position, an unknown ``static_r``
      site / steer_positions) REFUSES; a ``static_r`` ``site='pre'`` whose
      layers include block 0 (the embedding output) REFUSES (the lens's
      rule); EVERY mode of :data:`NATIVE_POSITION_MODES` is served since
      :data:`NATIVE_STAGE2_ARMED` (step 7) — while that flag is False a mode
      other than ``all`` is stage-2 work and the cell is SKIPPED by name with
      the mode in the sentence (the lens serves it).
    - ``circuit_breaker``: all-position by construction (its hook has no
      mask; the export writes ``petri_fit_positions="all"``), but vLLM's
      ``LoRALayerWeights.optimize()`` folds ``lora_alpha / r`` into
      ``lora_b`` in place at load, so the bf16 attestation fingerprint is
      defined only at unit scaling — a spec with ``lora_alpha != rank`` (or
      without both fields) is refused.
    - ``lora`` / ``lora_all`` / ``lora_subtraction``: ``positions`` must be
      present and ``"all"`` — native multi-LoRA has no position mask, so a
      response-only (or any masked) fit served as a plain adapter would run
      a DIFFERENT intervention under the cell's hash (exactly what
      ``lora_export._fit_positions`` refuses at export). A missing field is
      refused too (the export refuses a blob without it).

    The sentence is the whole public verdict; WHICH fact refused is
    :func:`_native_lora_verdict`'s second element, read only by
    :func:`native_serving_disposition` (never by matching the sentence).
    """
    return _native_lora_verdict(kind, resolved_spec)[0]


#: the FACT that refused an object in :func:`_native_lora_verdict` — one tag
#: per rule of :func:`native_lora_unservable_reason`, in its judged order.
#: :func:`native_serving_disposition` classifies by these, so the reason's
#: wording is free to change without moving a cell between skip and refuse.
UNSERVABLE_STACKED = "stacked"
UNSERVABLE_KIND = "kind"
UNSERVABLE_DECLARATION = "serve_declaration"
UNSERVABLE_SCALING = "lora_scaling"
UNSERVABLE_POSITIONS_MISSING = "positions_missing"
UNSERVABLE_POSITIONS_MASKED = "positions_masked"
#: a residual-kind record :func:`native_position_mode` cannot judge, or a
#: ``static_r`` ``site='pre'`` at block 0 — REFUSES
UNSERVABLE_RESIDUAL_RECORD = "residual_record"
#: a residual-kind cell whose native position mode is not ``all`` while stage 2
#: is not armed — SKIPPED by name (the lens serves it). RETIRED in effect since
#: :data:`NATIVE_STAGE2_ARMED` (step 7): with the flag True this fact cannot
#: fire. The tag and its rule stay so the stage-1 world remains expressible
#: (the tests pin it by monkeypatching the flag) and so a record whose fact was
#: already filed keeps its meaning.
UNSERVABLE_POSITION_MODE_STAGE2 = "position_mode_stage2"
UNSERVABLE_FACTS = (UNSERVABLE_STACKED, UNSERVABLE_KIND, UNSERVABLE_DECLARATION,
                    UNSERVABLE_SCALING, UNSERVABLE_POSITIONS_MISSING,
                    UNSERVABLE_POSITIONS_MASKED, UNSERVABLE_RESIDUAL_RECORD,
                    UNSERVABLE_POSITION_MODE_STAGE2)


def _native_lora_verdict(
        kind: str, resolved_spec: Optional[Mapping[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    """``(reason, fact)`` — the body of :func:`native_lora_unservable_reason`
    (its docstring is the rule; judged in that order), with the refusing
    fact tagged from :data:`UNSERVABLE_FACTS`. ``(None, None)`` when
    servable."""
    if kind == cm.PETRI_STACKED_KIND:
        return "stacked cells serve on the lens only (D6)", UNSERVABLE_STACKED
    if kind not in NATIVE_LORA_SERVABLE_KINDS:
        return (f"kind {kind!r} is not natively exportable (natively "
                f"servable kinds: {list(NATIVE_LORA_SERVABLE_KINDS)})"), UNSERVABLE_KIND
    if resolved_spec is None:
        return None, None  # no artifact -> ABSENT downstream; only the kind is judged
    try:
        declared = sd.declarations_of(resolved_spec)
    except sd.ServeDeclarationError as e:
        return f"malformed serve declaration in resolved_spec: {e}", UNSERVABLE_DECLARATION
    if declared:
        return (f"serve declaration {declared} is lens-only (a native "
                "adapter has no declaration surface, and a position variant "
                "replaces the spec's positions, hiding the fit leg)"), UNSERVABLE_DECLARATION
    if kind in RESIDUAL_SLOT_KINDS:
        try:
            mode, _ = native_position_mode(kind, resolved_spec,
                                           header_required=NATIVE_STAGE2_ARMED)
        except DefenseCellError as e:
            return (f"kind {kind!r}: residual-slot record cannot be judged: {e}"), UNSERVABLE_RESIDUAL_RECORD
        if kind == "static_r" and resolved_spec.get("site", "post") == "pre":
            layers = resolved_spec.get("layers")
            if isinstance(layers, (list, tuple)) and 0 in layers:
                return ("kind 'static_r': fit at resid_pre of layer 0, which is the embedding output — "
                        "there is no preceding decoder block whose output could carry the "
                        "edit; the lens refuses it and so does the residual slot (D6)"
                        ), UNSERVABLE_RESIDUAL_RECORD
        if mode != "all" and not NATIVE_STAGE2_ARMED:
            return (f"kind {kind!r}: native position mode {mode!r} is stage-2 (the per-token "
                    "position class is not armed on this tree); the lens serves it"
                    ), UNSERVABLE_POSITION_MODE_STAGE2
        return None, None
    if kind == "circuit_breaker":
        rank, alpha = resolved_spec.get("rank"), resolved_spec.get("lora_alpha")
        if rank is None or alpha is None:
            return ("resolved_spec carries no 'rank'/'lora_alpha' — the LoRA "
                    "scaling cannot be judged (a CircuitBreakerObject blob "
                    "stores both)"), UNSERVABLE_SCALING
        if float(alpha) != float(rank):
            return (f"non-unit LoRA scaling alpha={float(alpha)} r={rank} is "
                    "folded into lora_b by vLLM at load; the native "
                    "attestation is undefined for it"), UNSERVABLE_SCALING
        return None, None
    positions = resolved_spec.get(sd.POSITIONS_KEY)
    if positions is None:
        return ("resolved_spec records no 'positions' — the fit leg is "
                "unknown, and native multi-LoRA has no position mask (the "
                "export refuses the same blob)"), UNSERVABLE_POSITIONS_MISSING
    if positions != "all":
        return (f"positions={positions!r} is position-masked (native "
                "multi-LoRA applies a loaded adapter to every token; the "
                "honest serving path for this fit is the lens o_proj site)"), UNSERVABLE_POSITIONS_MASKED
    return None, None


#: the three things a NATIVE standup can do with one object cell of a view
#: (ONE native standup per view serves
#: none + prompted + every LoRA cell and skips the cells the engine cannot
#: serve BY NAME, superseding native-multilora-serving's two-standup
#: split): ``serve`` it as a PEFT adapter; ``skip`` it, recorded, because
#: the engine has no way to serve it honestly — its KIND has no PEFT form
#: at all, or its fit leg is position-masked (native multi-LoRA has no
#: position mask); ``refuse`` the whole standup, because the kind IS
#: natively servable and the OBJECT's record is not one the engine can
#: judge or attest (malformed / declared / non-unit scaling).
NATIVE_DISPOSITIONS = ("serve", "skip", "refuse")
NATIVE_DISPOSITION_SERVE, NATIVE_DISPOSITION_SKIP, NATIVE_DISPOSITION_REFUSE = NATIVE_DISPOSITIONS

#: the refusing facts a native standup SKIPS the cell for (by name, recorded
#: in ``cells_skipped``); every other fact REFUSES the standup.
NATIVE_SKIP_FACTS = (UNSERVABLE_STACKED, UNSERVABLE_KIND, UNSERVABLE_POSITIONS_MASKED,
                     UNSERVABLE_POSITION_MODE_STAGE2)


def native_serving_disposition(
        kind: str, resolved_spec: Optional[Mapping[str, Any]]
) -> Tuple[str, Optional[str]]:
    """``(disposition, reason)`` for ONE object under a native standup — one
    of :data:`NATIVE_DISPOSITIONS`, with the quotable sentence
    :func:`native_lora_unservable_reason` gives when it is not ``serve``
    (``None`` on ``serve``). The ONE predicate stays the judge; this only
    CLASSIFIES its verdict by WHICH fact refused (:data:`UNSERVABLE_FACTS`,
    tagged by :func:`_native_lora_verdict` — never by matching the sentence):

    - ``skip`` (:data:`NATIVE_SKIP_FACTS`) — the engine has no honest way to
      serve the cell, and the view's author can see why by name, so the
      standup SKIPS the cell, records it (``cells_skipped``) and serves the
      rest — the lens standup behind ``--serving-path lens`` is where such a
      cell still serves. Two facts land here:

      - the KIND has no native form: the stacked wrapper
        (``manifest.PETRI_STACKED_KIND``) or anything outside
        :data:`NATIVE_LORA_SERVABLE_KINDS` (loreft, alphasteer, antidote, ...).
        Judged by kind alone: the artifact's presence does not change what
        kind of thing it is (native-serving-default — since native-psr-
        serving the residual-slot kinds are servable, so an ABSENT psr cell
        reports ABSENT downstream like an absent LoRA cell).
      - a RESIDUAL-SLOT kind whose native position mode is not ``all`` while
        :data:`NATIVE_STAGE2_ARMED` is False (native-psr-serving §1.1): the
        response-fit ``psr_kl_all`` objects, the Arditi ``static_r`` rungs
        (``prompt_last`` / ``prefill_all``) were skipped by name with the mode
        in the sentence. RETIRED with step 7: the flag is True, the per-token
        position class serves every mode the lens names, and those cells now
        SERVE. The rule stays for the stage-1 world (a worker whose own
        ``STAGE2_POSITION_CLASSES`` is False refuses to load such a payload,
        so the two flags arm together).
      - a POSITION-MASKED fit of a natively servable kind — ``positions`` is
        PRESENT and not ``"all"`` (``"response"``: the scale-12 objects, e.g.
        ``safety_refusal__lora_r4`` on the gemma4e4b / qwen35_4b ``main``
        views). Native multi-LoRA applies an adapter to every token, so
        serving it would run a DIFFERENT intervention under the cell's hash;
        the honest path is the lens, and the rule is ONE native standup per
        view with the cells it cannot serve skipped by name and recorded
        (native-serving-default: before this, the fact REFUSED and took the
        whole gemma / qwen main-view standup down with it — never caught by
        a GPU job). Never served natively at the wrong positions, never
        dropped silently: the skip carries the predicate's position-masked
        sentence naming the mode.
    - ``refuse`` — the kind IS natively servable and the OBJECT's record is
      one the engine cannot even judge or attest: a MISSING ``positions``
      (malformed — the fit leg is unknown; the export refuses the same blob),
      a serve declaration (``@thinking`` / a position variant: no declaration
      surface on a native adapter, and a position variant hides the fit leg),
      a ``circuit_breaker`` with ``lora_alpha != r`` (or without both fields).
      A LoRA cell the view MEANT to serve whose record does not say what it
      is — never dropped, never lens-served on the side: the standup refuses
      by name.
    - ``serve`` — :func:`native_lora_unservable_reason` is None. A ``None``
      ``resolved_spec`` on a servable kind (no artifact on disk) is ``serve``
      too: only the kind can be judged, and the cell reports ABSENT
      downstream exactly as it always did.

    ``native_lora_unservable_reason`` itself is unchanged and stays the
    backstop at the cell source (``registry_adapter._serving_path_for``), at
    enumeration (``lens_controls=False``) and in the proxy: a skip-class cell
    (a PSR, a stacked cell, a position-masked fit) that reaches any of them
    is still a loud refusal there — the partition
    (``registry_adapter.partition_cells_for_native``) is what keeps it from
    reaching them."""
    reason, fact = _native_lora_verdict(kind, resolved_spec)
    if reason is None:
        return NATIVE_DISPOSITION_SERVE, None
    if fact in NATIVE_SKIP_FACTS:
        return NATIVE_DISPOSITION_SKIP, reason
    return NATIVE_DISPOSITION_REFUSE, reason


def enumerate_cells(
    spec: cm.PetriSpec,
    source: CellSource,
    *,
    behavior_files: Mapping[str, bytes],
    prompt_files: Mapping[str, bytes],
    lens_controls: bool = True,
    lora_controls: bool = False,
) -> PetriCellSet:
    """Resolve every canonical cell in ``spec.cells`` and derive the
    path-matched control legs.

    ``prompt_files`` carries the exact auditor/judge prompt BYTES in use
    (resolved from the provenance the spec declares) — they bind every
    cell's harness_config_hash.

    ``lens_controls`` (native-multilora-serving; default ``True`` keeps
    every existing table byte-identical) declares whether the server this
    table is for HAS a lens hook. ``False`` — a plugin-free lora-native
    standup — has three consequences, all of the one fact "no lens on this
    server": (1) the ``none__lens0`` control leg is NOT derived, and (2) a
    canonical ``lens`` route REFUSES — either would be a request vLLM
    silently serves as base under a ``+lens+`` identity; (3) every ``lora``
    route is re-checked against :func:`native_lora_unservable_reason` and
    an unservable one REFUSES (a PSR cell marked lora would otherwise 404
    at serve time). On a lens table (``True``) a ``lora`` route is the legacy
    lens+lora hybrid server the §10 gate's registry-dir path still admits
    (adapters from its ``lora_adapters`` scan), so the kind predicate is
    NOT applied there — the gate's per-kind canary is that server's check.
    The ``none__lora0`` leg (:data:`NOOP_LORA_CELL_ID`) derives iff a lora
    route is live OR ``lora_controls`` is True, under either setting.
    ``lora_controls`` (default ``False`` keeps every existing enumeration
    byte-identical) declares that the
    server this table is for REGISTERS THE ZERO ADAPTER even though no lora
    route is live: the zero-adapter-only native standup of a view with no
    LoRA cell (a ``none``-only cell set, a prompted-only view, a view whose
    objects were all skipped by name) and the Petri client's
    enumeration of such a server both need the control leg without a
    genuine lora route, so that the base and the steered configurations
    share an engine. A STACKED cell marked ``lora`` refuses under either
    setting.

    Refusals (loud): malformed ids, inconsistent registry entries (a
    registry SI contradicting the id-derived SI, an SI-bearing Class B
    entry, a missing resolved_spec), an unfrozen spec (any UNFROZEN hash
    input), a prompted-prefix mismatch against the spec's recorded hash, a
    stacked cell registered ``lora``, and — on a lora-native table —
    a ``lora`` route whose object is not natively servable.
    ABSENT (reported, not raised): a cell the registry lacks, or whose
    object artifact/sidecar is missing, unlocatable, or unreadable."""
    validate_petri_spec_for_enumeration(spec)

    canonical: List[PetriCell] = []
    absent: List[AbsentCell] = []
    lora_live = False

    def _harness_and_key(cell_id: str, identity: str, ihash: str) -> Tuple[str, str]:
        chash = cm.petri_harness_config_hash(
            spec, behavior_files=behavior_files, prompt_files=prompt_files,
            serving_identity=identity)
        return chash, cm.petri_run_key(chash, cell_id, ihash)

    for cell_id in spec.cells:
        parsed = parse_cell_id(cell_id)  # loud refusal on malformed
        entry = source.get(cell_id)
        if entry is None:
            absent.append(AbsentCell(
                cell_id, "not in the defense-cell registry"))
            continue
        if entry.cell_id != cell_id:
            raise DefenseCellError(
                f"registry entry id {entry.cell_id!r} != requested "
                f"{cell_id!r} — inconsistent registry")

        # set for PROMPTED and STACKED cells (both carry a verified
        # prompt leg the proxy must check per request, H8); None elsewhere.
        si_prefix_hash: Optional[str] = None
        if parsed.cell_class == "base":
            if entry.kind != cm.NONE_KIND:
                raise DefenseCellError(
                    f"cell 'none' must have kind 'none', got {entry.kind!r}")
            if entry.si_prefix or entry.object_path:
                raise DefenseCellError(
                    "cell 'none': the undefended base carries neither an "
                    "SI prefix nor an object — a base entry with either is "
                    "an inconsistent registry")
            ihash = cm.petri_intervention_hash({"kind": cm.NONE_KIND})
            path = "passthrough"
        elif parsed.cell_class == "prompted":
            if entry.kind != cm.PETRI_PROMPTED_KIND:
                raise DefenseCellError(
                    f"cell {cell_id!r} must have kind 'prompted', got "
                    f"{entry.kind!r}")
            if entry.si_id != parsed.si_id:
                raise DefenseCellError(
                    f"cell {cell_id!r}: registry si_id {entry.si_id!r} != "
                    f"id-derived {parsed.si_id!r}")
            if not entry.si_prefix:
                raise DefenseCellError(
                    f"cell {cell_id!r}: registry entry carries no si_prefix "
                    "bytes — the SI text IS the prompted intervention "
                    "(plan §5 H4); a missing prefix is a harness bug, not an "
                    "absent cell")
            if entry.object_path:
                raise DefenseCellError(
                    f"cell {cell_id!r}: a PROMPTED entry carries "
                    f"object_path {entry.object_path!r} — prompt-only "
                    "cells serve no object (the prefix/object exclusivity "
                    "guard; a prompt+object combination is a STACKED cell, "
                    "'stacked__<si>__<method>', never an overloaded "
                    "prompted entry)")
            si_prefix_hash = cm.sha256_hex16(bytes(entry.si_prefix))
            recorded = spec.prompted_prefix_hashes.get(parsed.si_id)
            if recorded != si_prefix_hash:
                raise DefenseCellError(
                    f"cell {cell_id!r}: SI-prefix bytes hash to "
                    f"{si_prefix_hash} but the manifest records {recorded!r} "
                    "for this SI — a wrong prefix can never masquerade as a "
                    "valid prompted baseline (H8)")
            ihash = cm.petri_intervention_hash(
                {"kind": cm.PETRI_PROMPTED_KIND, "si_id": parsed.si_id},
                si_prefix_bytes=bytes(entry.si_prefix))
            path = "passthrough"
        else:  # class_a | class_b | stacked — a defense object
            # --- identity refusals FIRST (harness bugs, never ABSENT),
            # all before any hashing or serving decision ------------------
            if entry.kind in (cm.NONE_KIND, cm.PETRI_PROMPTED_KIND,
                              cm.PETRI_STACKED_KIND):
                # NB even a stacked cell's registry kind is the OBJECT
                # leg's kind — the "stacked" wrapper kind is DERIVED at
                # enumeration (stacked_resolved_spec), never supplied.
                raise DefenseCellError(
                    f"cell {cell_id!r}: object cell has non-object kind "
                    f"{entry.kind!r}")
            if parsed.cell_class == "class_a":
                if entry.si_id != parsed.si_id:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: registry si_id {entry.si_id!r} "
                        f"!= id-derived {parsed.si_id!r} — a Class A object "
                        "must never be attributed to the wrong SI (it would "
                        "be executed and hashed as one SI's compilation but "
                        "reported as the other's)")
            elif parsed.cell_class == "stacked":
                if entry.si_id != parsed.si_id:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: registry si_id {entry.si_id!r} "
                        f"!= id-derived {parsed.si_id!r} — a stacked cell's "
                        "SI names its PROMPT leg; attributing it to the "
                        "wrong SI would verify one SI's prefix while "
                        "reporting the other's")
            else:  # class_b — SI-agnostic by definition
                if entry.si_id is not None:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: Class B baselines are SI-agnostic "
                        f"but the registry entry carries si_id "
                        f"{entry.si_id!r} — inconsistent registry")
            if entry.serving_path not in ("lens", "lora"):
                raise DefenseCellError(
                    f"cell {cell_id!r}: registry serving_path must be "
                    f"'lens' or 'lora', got {entry.serving_path!r}")
            if entry.serving_path == "lens" and not lens_controls:
                # a lora-native table has NO lens hook: a lens route on
                # it is a request vLLM would serve as base under a '+lens+'
                # identity — a plausible-looking wrong number. The proxy
                # refuses it again beside an engine attestation; refusing at
                # enumeration names the cell before any GPU is allocated.
                raise DefenseCellError(
                    f"cell {cell_id!r}: registered serving_path='lens' on a "
                    "table for a server with no lens hook "
                    "(lens_controls=False) — vLLM would serve it as the base "
                    "model under a lens identity (D15); a lens-served object "
                    "belongs on a lens standup, a lora-native standup serves "
                    "LoRA-only views (D2)")
            if parsed.cell_class == "stacked" and entry.serving_path == "lora":
                # no stacked SI-plus-LoRA cell is
                # served natively — on ANY table. A stacked cell's prompt
                # leg is verified by the proxy and its object leg is the
                # lens wrapper adapter; there is no PEFT form of "prefix +
                # adapter", and the id grammar admits the combination only
                # as a lens-served stack.
                raise DefenseCellError(
                    f"cell {cell_id!r}: a STACKED cell is registered "
                    f"serving_path='lora' — "
                    f"{native_lora_unservable_reason(cm.PETRI_STACKED_KIND, None)}"
                    "; register the stack on the lens (serving_path='lens')")
            # --- the prefix/object EXCLUSIVITY guard, made explicit and
            # two-sided: a stacked cell REQUIRES both legs; every
            # other object cell REFUSES a prefix. The guard catches the
            # exact mistakes that invalidated the earlier stacking
            # attempt — one leg silently dropped, or a prefix smuggled
            # onto a cell whose identity does not include it. -------------
            if parsed.cell_class == "stacked":
                if not entry.si_prefix:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: STACKED entry carries no "
                        "si_prefix bytes — the prompt leg IS half the "
                        "intervention; a stack without it would silently "
                        "serve object-only under a stacked identity "
                        "(exactly the one-leg failure that invalidated "
                        "the earlier stacking attempt)")
                si_prefix_hash = cm.sha256_hex16(bytes(entry.si_prefix))
                recorded = spec.prompted_prefix_hashes.get(parsed.si_id)
                if recorded != si_prefix_hash:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: SI-prefix bytes hash to "
                        f"{si_prefix_hash} but the manifest records "
                        f"{recorded!r} for SI {parsed.si_id!r} — a wrong "
                        "prefix can never masquerade as this stack's "
                        "prompt leg (H8)")
            elif entry.si_prefix:
                raise DefenseCellError(
                    f"cell {cell_id!r}: a NON-stacked object entry carries "
                    "si_prefix bytes — an object cell's identity includes "
                    "no prompt leg, so a prefix here is a mistake (route a "
                    "prompt+object combination through a STACKED cell, "
                    "'stacked__<si>__<method>')")
            # --- availability NEXT: missing/unreadable components are
            # ABSENT (reported, never raised). Deliberately
            # BEFORE the resolved_spec refusals: a not-yet-trained cell is
            # missing DATA, and an object that does not exist has no
            # authoritative load_object resolution to demand — the adapter
            # supplies resolved_spec=None for it, and the cell reports
            # ABSENT from the missing component. A PRESENT artifact with
            # no resolved_spec is still a loud refusal below. -------------
            if not entry.object_path:
                absent.append(AbsentCell(
                    cell_id, "registry entry carries no object_path — no "
                             "artifact to serve"))
                continue
            # Every ABSENT reason names the FULLY-RESOLVED ABSOLUTE path
            # that was actually checked (.absolute() = how the OS resolved
            # it), so a wrong-directory mistake — the registry enumerating
            # a store the object writers never write — is visible at a
            # glance in print-cells output instead of masquerading as
            # 'not trained yet'.
            artifact = Path(entry.object_path)
            if not artifact.is_file():
                absent.append(AbsentCell(
                    cell_id,
                    f"object artifact missing: {artifact.absolute()}"))
                continue
            try:
                artifact_bytes = artifact.read_bytes()
            except OSError as e:
                absent.append(AbsentCell(
                    cell_id, f"object artifact unreadable: "
                             f"{artifact.absolute()}: {e}"))
                continue
            sidecar_bytes = b""
            if entry.sidecar_path:
                sc = Path(entry.sidecar_path)
                if not sc.is_file():
                    absent.append(AbsentCell(
                        cell_id,
                        f"declared sidecar missing: {sc.absolute()}"))
                    continue
                try:
                    sidecar_bytes = sc.read_bytes()
                except OSError as e:
                    absent.append(AbsentCell(
                        cell_id, f"declared sidecar unreadable: "
                                 f"{sc.absolute()}: {e}"))
                    continue
            # --- resolution refusals (harness bugs, never ABSENT): the
            # artifact is in hand, so the authoritative resolution must be
            # too, and it must agree with the entry's identity -------------
            if entry.resolved_spec is None:
                raise DefenseCellError(
                    f"cell {cell_id!r}: object artifact is present but the "
                    "registry entry carries no resolved_spec — the adapter "
                    "must supply the authoritative load_object resolution "
                    "(every effective field); a guessed minimal spec would "
                    "let two loaders apply different resolved interventions "
                    "under one intervention_hash (plan §5 H4)")
            resolved = dict(entry.resolved_spec)
            if resolved.get("kind") != entry.kind:
                raise DefenseCellError(
                    f"cell {cell_id!r}: resolved_spec kind "
                    f"{resolved.get('kind')!r} != registry kind "
                    f"{entry.kind!r}")
            # a serve variant must be expressed by the spec it names (and a
            # declaration must be named) — the object leg's spec for a stack
            _check_variant_expressed(cell_id, parsed.serve_variant, resolved)
            if parsed.cell_class == "class_a":
                if "si_id" in resolved and resolved["si_id"] != parsed.si_id:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: resolved_spec si_id "
                        f"{resolved['si_id']!r} contradicts the id-derived "
                        f"{parsed.si_id!r} — refused before hashing (H4)")
            elif parsed.cell_class == "stacked":
                # the OBJECT leg's spec: for a Class A method it is the
                # same-SI compilation; a Class B object leg is SI-agnostic.
                if parsed.method in CLASS_A_METHODS:
                    if ("si_id" in resolved
                            and resolved["si_id"] != parsed.si_id):
                        raise DefenseCellError(
                            f"cell {cell_id!r}: object-leg resolved_spec "
                            f"si_id {resolved['si_id']!r} contradicts the "
                            f"id-derived {parsed.si_id!r} — refused before "
                            "hashing (H4)")
                elif resolved.get("si_id") is not None:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: a Class B object leg is "
                        f"SI-agnostic but its resolved_spec carries si_id "
                        f"{resolved['si_id']!r} — inconsistent registry")
            elif resolved.get("si_id") is not None:
                raise DefenseCellError(
                    f"cell {cell_id!r}: Class B resolved_spec carries si_id "
                    f"{resolved['si_id']!r} — an SI-bearing spec on an "
                    "SI-agnostic baseline is an inconsistent registry")
            if entry.serving_path == "lora" and not lens_controls:
                # a lora-native table (no lens hook): every lora route
                # IS a real PEFT adapter or it is nothing — the cell source's
                # lora_native policy already refused unservable kinds by
                # name; this is the enumeration-time re-check for ANY source,
                # so "psr marked lora -> 404 later" is a loud
                # refusal here instead.
                reason = native_lora_unservable_reason(entry.kind, resolved)
                if reason is not None:
                    raise DefenseCellError(
                        f"cell {cell_id!r}: registered serving_path='lora' "
                        "on a table for a server with no lens hook "
                        "(lens_controls=False), but its object is not "
                        f"natively servable — {reason}; a lora route there "
                        "is served by vLLM as a PEFT adapter or 404s, and a "
                        "lora-native standup serves LoRA-only views (D2) — "
                        "this cell belongs on a lens standup")
            if parsed.cell_class == "stacked":
                # the hash covers BOTH legs (H4): the wrapper spec
                # names the exact prefix by sha16 and the object leg's
                # bytes enter raw; passing the bytes makes the hash verify
                # the recorded sha against the real text.
                wrapper = stacked_resolved_spec(
                    parsed.si_id, bytes(entry.si_prefix), resolved)
                ihash = cm.petri_intervention_hash(
                    wrapper, artifact_bytes=artifact_bytes,
                    sidecar_bytes=sidecar_bytes,
                    si_prefix_bytes=bytes(entry.si_prefix))
            else:
                ihash = cm.petri_intervention_hash(
                    resolved, artifact_bytes=artifact_bytes,
                    sidecar_bytes=sidecar_bytes)
            path = entry.serving_path

        identity = cm.serving_identity(spec.serving_backend, path,
                                       spec.serving_mode)
        lora_live = lora_live or path == "lora"
        chash, run_key = _harness_and_key(cell_id, identity, ihash)
        canonical.append(PetriCell(
            cell_id=cell_id, cell_class=parsed.cell_class, kind=entry.kind,
            si_id=parsed.si_id, object_path=entry.object_path,
            virtual_model_name=_vmodel(cell_id, parsed.cell_class),
            serving_identity=identity, intervention_hash=ihash,
            harness_config_hash=chash, petri_run_key=run_key,
            control_of=None, si_prefix_hash=si_prefix_hash))

    by_id = {c.cell_id: c for c in canonical}

    # --- derived path-matched control legs ----------------------------------
    controls: List[PetriCell] = []

    def _control(control_id: str, control_of: str, *, backend: str,
                 path: str) -> None:
        target = by_id[control_of]
        identity = cm.serving_identity(backend, path, spec.serving_mode)
        # same intervention, different serving: the control leg's
        # intervention hash EQUALS its canonical counterpart's (§3).
        ihash = target.intervention_hash
        chash, run_key = _harness_and_key(control_id, identity, ihash)
        controls.append(PetriCell(
            cell_id=control_id, cell_class="control", kind=target.kind,
            si_id=target.si_id, object_path=None,
            virtual_model_name=VMODEL_PREFIX + control_id,
            serving_identity=identity, intervention_hash=ihash,
            harness_config_hash=chash, petri_run_key=run_key,
            control_of=control_of, si_prefix_hash=None))

    if BASE_CELL_ID in by_id:
        if lens_controls:
            # base served through the lens path, hook armed, scale 0 — the
            # pairing partner for every lens-served arm (emitted on every
            # table for a server WITH a lens hook, §3; suppressed on a
            # lora-native table, where it would be base served under a
            # lens identity).
            _control(NOOP_LENS_CELL_ID, BASE_CELL_ID,
                     backend=spec.serving_backend, path="lens")
        if lora_live or lora_controls:
            # base through native multi-LoRA with the zero (B=0) adapter —
            # when a LoRA-path arm is live (§3), on either table, or when the
            # caller says the server registers the zero adapter regardless
            # (``lora_controls``: the zero-adapter-only native standup).
            _control(NOOP_LORA_CELL_ID, BASE_CELL_ID,
                     backend=spec.serving_backend, path="lora")

    cell_set = PetriCellSet(canonical=tuple(canonical),
                            controls=tuple(controls), absent=tuple(absent))
    cell_set.virtual_model_map()  # refuse duplicate vmodels loudly
    return cell_set


def validate_petri_spec_for_enumeration(spec: cm.PetriSpec) -> None:
    """Structural validation + the enumeration-time requirement that the
    spec is fully validated (UNFROZEN inputs surface as refusals inside
    petri_harness_config_hash, which enumeration calls per cell)."""
    cm.validate_petri_spec(spec)
    # every prompted AND stacked cell in the manifest needs its SI's
    # recorded prefix hash — both carry a proxy-verified prompt leg (H8)
    for cell_id in spec.cells:
        # the prompt leg's SI sits on the BASE id; whether a variant is
        # legal on this cell is parse_cell_id's call, in enumerate_cells
        base, _variant = _split_variant(cell_id)
        si = None
        if base.startswith(_PROMPTED_PREFIX):
            si = base[len(_PROMPTED_PREFIX):]
        elif base.startswith(_STACKED_PREFIX):
            si = base[len(_STACKED_PREFIX):].partition("__")[0]
        if si is not None and si not in spec.prompted_prefix_hashes:
            raise DefenseCellError(
                f"manifest cell {cell_id!r} has no "
                f"prompted_prefix_hashes[{si!r}] entry — the recorded "
                "prefix hash is what the proxy verifies against (H8)")


# --- Pairing semantics (the two-tier rule) ----------------------------------


@dataclass(frozen=True)
class NoOpCertificate:
    """One §10 gate exact no-op certificate, stored under its FULL key
    (H5): a certificate licenses Tier-2 cross-path references
    only for rows whose determinism mode (and revision/commit world)
    matches this key exactly."""

    backend: str
    path: str
    mode: str
    model_revision: str
    serving_impl_commit: str
    gate_prompt_set_hash: str


@dataclass(frozen=True)
class PairingOK:
    """An ACCEPTED pairing. ``cross_path`` labels a Tier-2 cross-path
    reference (the report must carry the label, §3)."""

    tier: int
    cross_path: bool
    cell_id_a: str
    cell_id_b: str
    identity_a: str
    identity_b: str


@dataclass(frozen=True)
class PairingLeg:
    """One leg of a pairing check: the ROLE METADATA check_pairing needs
    (a bare serving-identity string is not enough — two unrelated defended
    cells can share an identity). Build from a :class:`PetriCell` via
    :meth:`from_cell`, or directly from stored-row metadata (cell_id +
    engine column + the enumeration record's control_of)."""

    cell_id: str
    serving_identity: str
    #: derived control legs only: the canonical cell this leg re-serves.
    control_of: Optional[str] = None

    @classmethod
    def from_cell(cls, cell: PetriCell) -> "PairingLeg":
        return cls(cell_id=cell.cell_id,
                   serving_identity=cell.serving_identity,
                   control_of=cell.control_of)


#: derived-control-leg suffix -> the (backend, path) it must be served on.
_CONTROL_LEG_SERVING = {
    "lens0": ("proxy_vllm", "lens"),
    "lora0": ("proxy_vllm", "lora"),
}


@dataclass(frozen=True)
class _LegRole:
    """A resolved pairing leg: its parsed CANONICAL cell (for a control
    leg, the cell it re-serves) and whether it is a no-op base leg."""

    leg: PairingLeg
    ident: cm.ServingIdentity
    parsed: ParsedCellId
    is_noop_base: bool


def _as_leg(obj: Any, label: str) -> PairingLeg:
    if isinstance(obj, PairingLeg):
        return obj
    if isinstance(obj, PetriCell):
        return PairingLeg.from_cell(obj)
    raise PairingError(
        f"check_pairing: {label} must be a PetriCell or PairingLeg carrying "
        "role metadata (cell_id, serving_identity, control_of) — a bare "
        "serving-identity string is refused because two unrelated defended "
        f"cells can share one identity (plan §3); got {type(obj).__name__}: "
        f"{obj!r}")


def _resolve_leg(leg: PairingLeg, label: str) -> _LegRole:
    """Validate a leg's internal consistency and derive its role. Loud on
    contradictory metadata — a leg whose id, control_of, and serving
    identity disagree is a harness bug, never a silent role guess."""
    ident = cm.parse_serving_identity(leg.serving_identity)
    if leg.control_of is None:
        parsed = parse_cell_id(leg.cell_id)  # loud on malformed/control ids
        if parsed.cell_class == "base":
            if ident.path != "passthrough":
                raise PairingError(
                    f"check_pairing: {label} is the canonical base "
                    f"{leg.cell_id!r} but carries serving path "
                    f"{ident.path!r} — the base is served passthrough; a "
                    "path-matched no-op is a DERIVED control leg "
                    f"({NOOP_LENS_CELL_ID} / {NOOP_LORA_CELL_ID}), never the "
                    "base itself (plan §3)")
            return _LegRole(leg, ident, parsed, is_noop_base=True)
        return _LegRole(leg, ident, parsed, is_noop_base=False)
    # a derived control leg: <control_of>__<lens0|lora0>
    parsed_target = parse_cell_id(leg.control_of)
    head, sep, suffix = leg.cell_id.rpartition("__")
    if (not sep or head != leg.control_of
            or suffix not in _CONTROL_LEG_SERVING):
        raise PairingError(
            f"check_pairing: {label} claims control_of="
            f"{leg.control_of!r} but its id {leg.cell_id!r} is not "
            f"'{leg.control_of}__<{'|'.join(_CONTROL_LEG_SERVING)}>' — "
            "contradictory control-leg metadata")
    expected = _CONTROL_LEG_SERVING[suffix]
    if (ident.backend, ident.path) != expected:
        raise PairingError(
            f"check_pairing: control leg {leg.cell_id!r} must be served on "
            f"{expected[0]}+{expected[1]} but carries "
            f"{leg.serving_identity!r} — contradictory control-leg metadata")
    if parsed_target.cell_class == "base":
        return _LegRole(leg, ident, parsed_target, is_noop_base=True)
    raise PairingError(
        f"check_pairing: {label} claims to be a control leg of "
        f"{leg.control_of!r} ({parsed_target.cell_class}) — only the base "
        "derives lens0/lora0 legs (plan §3); contradictory control-leg "
        "metadata")


def _validate_certificate(c: NoOpCertificate) -> None:
    if (c.backend not in cm.SERVING_BACKENDS
            or c.path not in cm.SERVING_PATHS.get(c.backend, ())
            or c.mode not in cm.SERVING_MODES
            or not cm._is_full_sha(c.model_revision)
            or not cm._is_full_sha(c.serving_impl_commit)
            or not cm._is_hex16(c.gate_prompt_set_hash)):
        raise PairingError(
            f"malformed no-op certificate on record: {c!r} — the §10 store "
            "is corrupt; refusing rather than treating a bad certificate as "
            "merely non-matching")


def check_pairing(
    leg_a: Any,
    leg_b: Any,
    *,
    tier: int,
    certificates: Sequence[NoOpCertificate] = (),
    model_revision: Optional[str] = None,
    serving_impl_commit: Optional[str] = None,
    gate_prompt_set_hash: Optional[str] = None,
) -> PairingOK:
    """The §3 two-tier pairing rule. Returns :class:`PairingOK` or raises
    :class:`PairingError` — a refused pairing is an error, never a silent
    drop.

    Both legs are :class:`PetriCell` or :class:`PairingLeg` — ROLE-AWARE
    on purpose: a bare identity string refuses, because two unrelated
    defended cells sharing a serving identity must never pass as a
    confirmatory pair.

    The engine/serving-identity comparison itself is the axis-agnostic
    primitive ``sisteer.capeval.engine_identity.identity_diff`` — this
    function is a CALLER of it, mapping its verdict (which of backend /
    path / mode differ) onto the §3 pairing prose and the tier rules
    below. Reuse the primitive, not this function, for a non-petri axis.

    - **Always refused, both tiers:** differing ``backend`` (proxy ↔ HF —
      a DEFENSIVE invariant: no audit cell is hf-served anymore, so an
      hf_shim-tagged row can only be foreign/corrupt), differing ``mode``
      (interleaved ↔ exact_replay), a leg paired against itself, and
      internally contradictory leg metadata.
    - **Tier 1 (confirmatory deltas):** EXACT serving-identity
      equality AND exactly one of the two legal shapes:
      (a) a defense arm ↔ its path-matched NO-OP BASE leg (the canonical
      passthrough base, or a derived ``NOOP_LENS_CELL_ID`` /
      ``NOOP_LORA_CELL_ID`` leg whose ``control_of`` is the base — validated
      against its own id and serving path); or
      (b) the within-method two-SI contrast: two CANONICAL Class A cells
      of the same method compiled from different SIs
      (``si_constitution__<m>`` ↔ ``si_safety__<m>``), served under the
      SAME serve variant (an ``@thinking`` arm never pairs confirmatorily
      with an in-support one — the render differs).
      Anything else — two unrelated defenses on one identity included —
      refuses.
    - **Tier 2 (cross-path descriptive references; arm ↔ prompted
      baseline / passthrough reference):** equal ``(backend, mode)``;
      ``path`` may differ ONLY when a §10 exact no-op certificate for
      EACH path is on record matching the FULL key — ``(backend, path,
      mode, model_revision, serving_impl_commit,
      gate_prompt_set_hash)``. A certificate produced in a different
      mode, or against a different gate prompt set, licenses nothing
      (H5).

    ``model_revision`` / ``serving_impl_commit`` /
    ``gate_prompt_set_hash`` are REQUIRED for a Tier-2 cross-path lookup
    (the full-key match): pass the spec's pinned values and the current
    gate prompt-set hash.
    """
    if tier not in (1, 2):
        raise PairingError(f"unknown pairing tier {tier!r} (plan §3 defines "
                           "Tier 1 and Tier 2)")
    ra = _resolve_leg(_as_leg(leg_a, "leg_a"), "leg_a")
    rb = _resolve_leg(_as_leg(leg_b, "leg_b"), "leg_b")
    la, lb = ra.leg, rb.leg
    identity_a, identity_b = la.serving_identity, lb.serving_identity
    a, b = ra.ident, rb.ident
    if la.cell_id == lb.cell_id:
        raise PairingError(
            f"pairing refused: {la.cell_id!r} paired against itself — a "
            "delta needs two distinct legs (plan §3)")
    # the shared engine-identity primitive decides WHAT differs; the §3
    # prose below is petri's. Refusal precedence is unchanged: backend,
    # then mode, then (per tier) path.
    differing = identity_diff(
        {"backend": a.backend, "path": a.path, "mode": a.mode},
        {"backend": b.backend, "path": b.path, "mode": b.mode})
    if "backend" in differing:
        raise PairingError(
            f"pairing refused: differing serving backend ({identity_a!r} vs "
            f"{identity_b!r}) — proxy-served rows never pair against "
            "HF-served rows, either tier (plan §3)")
    if "mode" in differing:
        raise PairingError(
            f"pairing refused: differing determinism mode ({identity_a!r} vs "
            f"{identity_b!r}) — interleaved and exact_replay rows never mix "
            "(plan §3)")

    def _ok(cross_path: bool) -> PairingOK:
        return PairingOK(tier=tier, cross_path=cross_path,
                         cell_id_a=la.cell_id, cell_id_b=lb.cell_id,
                         identity_a=identity_a, identity_b=identity_b)

    if tier == 1:
        if differing:  # backend/mode equal above, so this is the path
            raise PairingError(
                f"Tier-1 pairing refused: serving identities differ "
                f"({identity_a!r} vs {identity_b!r}) — confirmatory deltas "
                "require EXACT identity; pair against the path-matched "
                f"no-op control leg ({NOOP_LENS_CELL_ID} / {NOOP_LORA_CELL_ID}), "
                "not the passthrough base (plan §3)")
        # identity strings are now equal (identity = backend+path+mode)
        if ra.is_noop_base and rb.is_noop_base:
            raise PairingError(
                f"Tier-1 pairing refused: {la.cell_id!r} vs {lb.cell_id!r} "
                "are both no-op base legs — a confirmatory delta needs a "
                "defense arm on one side (plan §3)")
        if ra.is_noop_base or rb.is_noop_base:
            return _ok(cross_path=False)  # shape (a): arm ↔ its no-op base
        pa, pb = ra.parsed, rb.parsed
        if (la.control_of is None and lb.control_of is None
                and pa.cell_class == pb.cell_class == "class_a"
                and pa.method == pb.method and pa.si_id != pb.si_id
                and pa.serve_variant == pb.serve_variant):
            return _ok(cross_path=False)  # shape (b): within-method two-SI
        raise PairingError(
            f"Tier-1 pairing refused: {la.cell_id!r} vs {lb.cell_id!r} is "
            "neither an arm ↔ its path-matched no-op base nor the "
            "within-method two-SI contrast (si_constitution__<m> ↔ "
            "si_safety__<m>, same serve variant) — a shared serving "
            "identity alone never licenses a confirmatory delta between "
            "unrelated cells (plan §3)")

    # Tier 2
    if not differing:  # backend/mode equal above; empty diff = same path
        return _ok(cross_path=False)
    if (model_revision is None or serving_impl_commit is None
            or gate_prompt_set_hash is None):
        raise PairingError(
            "Tier-2 cross-path pairing refused: certificate lookup needs "
            "the full key — pass model_revision, serving_impl_commit, and "
            "gate_prompt_set_hash (plan §10)")
    if not cm._is_full_sha(model_revision):
        raise PairingError(
            f"Tier-2 cross-path pairing refused: model_revision must be a "
            f"full 40-hex SHA, got {model_revision!r}")
    if not cm._is_full_sha(serving_impl_commit):
        raise PairingError(
            f"Tier-2 cross-path pairing refused: serving_impl_commit must "
            f"be a full 40-hex SHA, got {serving_impl_commit!r}")
    if not cm._is_hex16(gate_prompt_set_hash):
        raise PairingError(
            f"Tier-2 cross-path pairing refused: gate_prompt_set_hash must "
            f"be a {cm.HASH_HEX_LEN}-hex hash, got {gate_prompt_set_hash!r}")
    for c in certificates:
        _validate_certificate(c)

    def _has_certificate(ident) -> bool:
        return any(
            c.backend == ident.backend and c.path == ident.path
            and c.mode == ident.mode
            and c.model_revision == model_revision
            and c.serving_impl_commit == serving_impl_commit
            and c.gate_prompt_set_hash == gate_prompt_set_hash
            for c in certificates)

    missing = [ident for ident in (a, b) if not _has_certificate(ident)]
    if missing:
        names = [f"{i.backend}+{i.path}+{i.mode}" for i in missing]
        raise PairingError(
            f"Tier-2 cross-path pairing refused: no exact no-op certificate "
            f"on record for {names} matching the FULL key (need one per "
            f"path, in the rows' own mode {a.mode!r}, at model_revision="
            f"{model_revision!r} / serving_impl_commit="
            f"{serving_impl_commit!r} / gate_prompt_set_hash="
            f"{gate_prompt_set_hash!r}) — a certificate from a different "
            "mode or a different gate prompt set licenses nothing "
            "(plan §3/§10 H5)")
    return _ok(cross_path=True)
