"""DefenseCell — defense identity for the two-SI comparison.

The single-SI jailbreak lane (``eval/jailbreak.py``) identifies a defense by
ARM NAME alone: ``COMPILED = tuple(ARMS)``, ``DEFENSES = ("none", "prompted",
*COMPILED)``, and ``jailbreak_run._defense_object_path`` resolves
``<defenses_dir>/<arm>/object...``. That is a defect for the two-SI matrix,
which trains TWO objects per compiled arm (one per system instruction) plus TWO
prompted baselines: keyed on the arm alone, the two SIs' objects for one arm
share a path AND a results key, so the second silently OVERWRITES the first
while the aggregate still looks complete (verified in
``eval/jailbreak.py:43-52`` and ``scripts/jailbreak_run.py:112-118``). There is
also only ONE ``prompted`` identity where the two-SI matrix needs two.

A :class:`DefenseCell` makes identity a CELL, not an arm name. Every column of
the matrix is one cell with a unique :attr:`~DefenseCell.id`; everything the
pipeline keys on — the CLI ``--defense`` surface, private output dirs, sbatch
array decode, ``meta.json``, the aggregation dicts, and the completeness check
— keys on ``cell.id``, and ONLY object construction (which trainer to call,
which artifact filename to read/write) uses ``cell.arm`` (and ``cell.si_id`` to
pick the instruction).

Id scheme (stable, readable, documented)
-----------------------------------------
::

    none                the undefended base            (class "baseline", no object, no SI)
    prompted__<si_id>   a prompted baseline            (class "baseline", no object, SI as system prompt)
    <si_id>__<arm>      a Class A prompt-compiling cell (class "A", one per (SI, arm))
    <arm>               a Class B SI-agnostic object    (class "B", bare arm name)
    <object cell>@<variant>   a SERVE-TIME variant of an object cell (never minted here)

The Class B id is the BARE arm name on purpose. A Class B object is fit once,
SI-free, so it has no ``si_id`` — and a bare id degenerates EXACTLY to the
string the single-SI lane already uses. That degeneration is what keeps the
existing lane (and jobs already running against it) working unchanged:
:func:`object_path_for` resolves a bare arm to
``<defenses_dir>/<arm>/object...`` exactly as the old ``_defense_object_path``
did, so the two paths are byte-identical. (A ``si_agnostic__<arm>`` prefix for
Class B was considered; the bare form is used here for that
backwards-compatible degeneration. The distinction is recorded on
``cell.class_``/``cell.si_id``, not smuggled into the id.)

Because ``<si_id>`` values (``safety_refusal``, ``si_constitution``) and arm
names (``psr_kl_all``, …) never contain a double underscore, splitting a cell
id on the FIRST ``__`` unambiguously recovers ``(si_id, arm)``.

A ``@<variant>`` suffix (``capeval.serving.serve_declaration.SERVE_VARIANTS``
— the fit-support-transfer serve declarations, e.g.
``safety_refusal__psr_kl_all@thinking``) names a SERVE-TIME cell that
re-serves its base cell's exact artifact under a hashed declaration.
:func:`build_registry` never mints one; the id splitters below strip the
suffix FIRST, so :func:`si_of` / :func:`arm_of` / :func:`object_path_for` see
the base (same object, same path) and :func:`legacy_cell_id` re-appends it.
An unknown variant refuses (the vocabulary is closed).

Deriving the matrix from the registry, never a literal
------------------------------------------------------
The two-SI matrix is a SUBSET of ``methods.ARMS`` — the Class A arms and the
Class B arms — not every registered arm. The registry FILTERS to arms that
actually exist in ``methods.ARMS`` and every count DERIVES from the filtered
registry (``build_registry`` / :func:`matrix_counts`), so an arm leaving
``methods.ARMS`` (cast and circuit_breaker, removed entirely 2026-08-20 —
their fitted artifacts and ``load_object`` hydration survive) simply stops
contributing a cell.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..capeval.serving.serve_declaration import join_variant, split_variant
from ..methods import (ARMS, CANONICAL_TO_LEGACY_ARM, object_artifact_name)

# The two-SI matrix method sets (Class A / Class B). These are the
# arms the two-SI comparison uses — a SUBSET of methods.ARMS, in matrix order.
# cast (Class A) and circuit_breaker (Class B) left methods.ARMS entirely on
# 2026-08-20 and are no longer named here either: nothing
# enumerates them, while their fitted artifacts stay loadable through
# methods.load_object (_KIND_LOADERS keeps both kinds).
CLASS_A_ARMS: tuple[str, ...] = (
    "psr_kl_all", "psr_mse_all", "vector_all", "lora_r1",
    "loreft", "lofit", "token_bos",
    # Three-rung probe ladder.
    # CLASS A, not B: the ladder's actuation direction r is a diff-in-means
    # between SI-PRESENT and SI-ABSENT renderings, so the SI enters the fit and
    # there is one object per SI. `alphakl_null` is included alongside the two
    # additional rungs because it is the DEPLOYMENT-MATCHED stage 1
    # (per-position ReLU-clamped gate at prompt AND decode) whereas
    # `alphasteer` is prefill-only / last-token-broadcast / UNCLAMPED — without
    # it an alphasteer-vs-alphakl_soft gap conflates the penalty with the
    # deployment. It costs one capture + eigh + one solve.
    "alphakl_null", "alphakl_soft", "alphakl_fisher",
    # diffmean: the diff-in-means / CAA-family
    # floor -- one fixed RAW (un-normalised) vector
    # r = diff-in-means(SI-present - SI-absent), EXTRACTED from last-prompt-token
    # activations and applied additively; no gate, no probe, no ridge.
    # Class A because r is SI-derived (one object per SI). Appended LAST so
    # every existing arm-major array decode keeps its meaning.
    #
    # DEPLOYMENT (2026-08-21 -- an earlier version of this comment said "at the
    # last prompt token" without saying it meant EXTRACTION, and was read as the
    # apply site; it cost a live analysis, so it is spelled out here). The apply
    # site is much broader than the extraction site: DiffMeanObject.apply routes
    # through hooks.steering(positions="all", skip_bos=True) and generate.py
    # holds it open across the whole generate call, so r lands on EVERY position
    # -- all prompt tokens during prefill AND every decode step directly -- with
    # only the true BOS excluded. It therefore does NOT have alphasteer's
    # prefill-only reach (which touches decode only through the steered KV
    # cache). The real contrast against the all-layer arms (psr_*, alphakl_*) is
    # SINGLE-LAYER vs all-layer, not position coverage: diffmean steers the
    # config's one `layer` (15), with no `layers` override anywhere in the
    # two-SI driver.
    "diffmean",
    # bidpo: the DPO preference arm, re-added to the two-SI trained set.
    # Appended LAST so existing arm-major array indices are unchanged.
    "bidpo",
    # vector + bidpo_lora: the two cells completing
    # the objective x substrate factorial around the matrix's single DPO cell
    # (KL: vector / lora_r1; DPO: bidpo / bidpo_lora). `vector` is the
    # long-registered distill_soft arm, newly in the two-SI matrix;
    # `bidpo_lora` is bidpo's objective on lora_r1's exact substrate
    # (extraction/bidpo_lora.py). Appended LAST, in sequence, so every
    # earlier arm's array index is unchanged.
    "vector",
    "bidpo_lora",
    # token_faithful: the Radevski faithful Steering-Token arm
    # (semantic-init soft token appended at the last prompt position,
    # T=10 / loss_scale=100 — long registered in methods.ARMS; the name is
    # kept rather than renamed to steering_token). Appended LAST so every
    # earlier arm's arm-major array index is unchanged.
    "token_faithful",
    # lora_r4 (2026-08-28): the ALL-LAYER low-rank LoRA arm
    # (methods.ARMS -> MultiLayerLoRAObject, o_proj on every layer) — first
    # registered here as "lora_r1_all" (it had been in methods.ARMS for a
    # while but was never listed here, so `<si>__lora_r1_all` was not a cell
    # id and the scale-12 campaign's staged objects — fit with
    # arm=lora_r1_all, target_layers 0..31 — could not be served: KeyError
    # at cell_controls_run --cell), then RENAMED the same day: the staged
    # objects are RANK 4, so the rank is named and "all" is implicit. The
    # legacy spelling stays registered in methods.ARMS and its staged
    # artifact dirs stay readable (object_path_for below resolves
    # `<si>__lora_r1_all/` when the canonical dir holds no artifact —
    # methods.LEGACY_ARM_ALIASES). NOT an alias of lora_r1: lora_r1 is the
    # SINGLE-layer rank-1 arm, a genuinely different method
    # (train_budget's _ARM_TRAINER_SOURCES lists them separately). Appended
    # LAST so every earlier arm's arm-major array index is unchanged.
    "lora_r4",
    # psr_xfer_* (2026-08-28, psr-transfer behavioral eval): CROSS-MODEL
    # TRANSFERRED PSR units (methods.ARMS note) — same object kind 'psr'
    # and hook site as psr_kl_all, distinct arm names so a transferred
    # artifact can never collide with (or be mistaken for) the target's
    # native <si>__psr_kl_all fit. Class A: each transfer is derived from
    # ONE SI's source fit, so there is one object per SI. The _rw1 names
    # are the ridge-ablation variants. Appended LAST so every earlier
    # arm's arm-major array index is unchanged.
    "psr_xfer_from_e2b",
    "psr_xfer_from_e2b_rw1",
    "psr_xfer_from_e4b",
    "psr_xfer_from_e4b_rw1",
)
# caa_static_r: the CAA / diff-in-means floor as a Class B arm — an earlier
# development branch's static_r diff-in-means fit, reused under the
# AlphaSteer-MATCHED deployment: the paper's TEN layers, resid_pre,
# prefill-only, RAW per-layer lam·r_l broadcast to ALL prompt positions — NOT
# the withdrawn single-layer/last-token/unit-norm form, which measured below
# the unsteered baseline and is not a floor. With the deployment matched,
# alphasteer-vs-caa_static_r isolates what the learned input-dependent d×d
# map buys over a fixed direction. Appended LAST so the existing Class B
# cell order keeps its meaning. See extraction/caa_static_r.py for the full
# provenance, the documented deviation from the literal CAA form, and the
# capability caveat.
# IMPORTED, never retyped: the Class B tuple lives in ONE torch-free module
# (capeval.petri.defense_cells) so this registry, the harness id vocabulary
# and the tinycap launcher's grammar cannot drift apart — they did, in both
# directions. The NAME stays CLASS_B_ARMS here (this module's own vocabulary;
# a rename sweep is a separate change) but the object is that one tuple. The
# import is safe in both directions: defense_cells imports nothing from
# sisteer.eval.
from ..capeval.petri.defense_cells import CLASS_B_IDS as CLASS_B_ARMS
# Class B ids served from an external bank with no trainer on this tree —
# admitted by the EXPLICIT registry only (never the Cartesian one, never fit_class_b)
from ..capeval.petri.defense_cells import SERVED_ONLY_CLASS_B_IDS

# The two system instructions being compared (configs/two_si.yaml
# si_ids). Both resolve through sisteer.data.instructions.
DEFAULT_SI_IDS: tuple[str, ...] = ("safety_refusal", "si_constitution")

_BASELINE = "baseline"
# "stacked" is a fourth cell class: an SI system prompt AND a compiled object
# together (see eval/jailbreak.STACKED). It carries BOTH an arm (which object)
# and an si_id (which SI) like Class A, but is served as mode='stacked'. Added
# LAST so nothing that enumerates the historical three classes changes.
_STACKED = "stacked"
_CLASSES = ("A", "B", _BASELINE, _STACKED)


@dataclass(frozen=True)
class DefenseCell:
    """One column of the two-SI defense matrix.

    ``id`` is the ONLY key the pipeline uses (CLI, dirs, array decode, meta,
    aggregation, completeness). ``arm`` is used ONLY to construct/serve the
    object (``methods.ARMS[arm].train`` and :func:`methods.object_artifact_name`);
    ``si_id`` picks the instruction to distil/prompt with. Baselines and Class B
    have ``si_id is None`` and Class B / ``none`` have no ``arm`` distinction
    beyond the object it builds.
    """

    id: str
    class_: str  # "A" | "B" | "baseline"
    arm: str | None  # None for none/prompted baselines
    si_id: str | None  # None for Class B objects and the `none` baseline
    object_path: Path | None = None  # None for baselines (no object on disk)
    # The over-refusal non-inferiority reference cell id(s): a Class A
    # cell references its matching prompted arm; a Class B object references
    # BOTH prompted arms; a baseline references nothing (it IS a reference).
    ni_reference_ids: tuple[str, ...] = ()
    stacked: bool = False  # serve the object AND its si_id's system together (2026-08-25)

    def __post_init__(self) -> None:
        if self.class_ not in _CLASSES:
            raise ValueError(f"class_ must be one of {_CLASSES}, got {self.class_!r}")
        if self.class_ == "A" and (self.arm is None or self.si_id is None):
            raise ValueError(f"Class A cell {self.id!r} needs both arm and si_id")
        if self.class_ == "B" and (self.arm is None or self.si_id is not None):
            raise ValueError(
                f"Class B cell {self.id!r} needs an arm and must have si_id=None "
                "(SI-agnostic, fit once)")
        if self.class_ == _STACKED and (self.arm is None or self.si_id is None):
            raise ValueError(
                f"stacked cell {self.id!r} needs both arm and si_id (it stacks "
                "that arm's object on that SI's system prompt)")


# --------------------------------------------------------------------------- #
# Id ⇄ (si_id, arm) — pure structural parsing, the inverse of the id scheme    #
# --------------------------------------------------------------------------- #


def arm_of(cell_id: str) -> str | None:
    """The ARM component of a cell id (what object construction / the artifact
    filename key on), or ``None`` for a baseline (``none`` / ``prompted`` /
    ``prompted__<si>``). Splitting on the first ``__`` is unambiguous because no
    si_id or arm name contains a double underscore.

    A stacked cell is ``stacked__<si>__<arm>`` (three segments): the ``stacked__``
    mode prefix is stripped FIRST, then the remaining ``<si>__<arm>`` splits on
    its own first ``__`` exactly like a Class A id — so the arm is recovered
    even though the raw id has two ``__`` (a plain split-on-first would return
    ``<si>__<arm>``, not the arm; hence the explicit prefix strip).

    A serve variant (``<cell>@<variant>``) is stripped first — it changes
    how the object is served, never which object."""
    cell_id, _variant = split_variant(cell_id)
    if cell_id in ("none", "prompted") or cell_id.startswith("prompted__"):
        return None
    if cell_id.startswith("stacked__"):
        rest = cell_id[len("stacked__"):]
        return rest.split("__", 1)[1] if "__" in rest else rest
    if "__" in cell_id:
        return cell_id.split("__", 1)[1]
    return cell_id  # bare arm (Class B in two-SI; a compiled arm in the old lane)


def si_of(cell_id: str) -> str | None:
    """The SI component of a cell id, or ``None`` for ``none`` / bare Class B /
    a bare old-lane arm. A stacked cell ``stacked__<si>__<arm>`` yields ``<si>``
    (prefix stripped, then the first ``__`` of the remainder). A serve
    variant (``<cell>@<variant>``) is stripped first."""
    cell_id, _variant = split_variant(cell_id)
    if cell_id == "prompted":
        return None
    if cell_id.startswith("prompted__"):
        return cell_id[len("prompted__"):]
    if cell_id.startswith("stacked__"):
        rest = cell_id[len("stacked__"):]
        return rest.split("__", 1)[0] if "__" in rest else None
    if "__" in cell_id:
        return cell_id.split("__", 1)[0]
    return None


def sha256_file(path) -> str:
    """Streaming sha256 of an artifact file — the object identity the
    trainer stamps into ``train_budget_meta.json`` (``object_sha256``) and
    the readers (train resume, ``two_si_run generate``) verify before
    trusting an object/sidecar pair. Compiled objects are small (vectors /
    LoRA deltas), so this is milliseconds."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def legacy_cell_id(cell_id: str) -> str | None:
    """The LEGACY spelling of ``cell_id`` when its arm has a legacy alias
    (``methods.LEGACY_ARM_ALIASES``), else None. ``safety_refusal__lora_r4``
    -> ``safety_refusal__lora_r1_all``; a non-aliased cell -> None. This is
    the ONE derivation the artifact-layer alias uses — the provenance
    checks accept exactly this spelling (a recorded alias of the SAME
    cell), never an arbitrary mismatch. A serve variant rides through:
    ``safety_refusal__lora_r4@thinking`` -> ``safety_refusal__lora_r1_all@thinking``."""
    base, variant = split_variant(cell_id)
    arm = arm_of(base)
    if arm is None:
        return None
    legacy_arm = CANONICAL_TO_LEGACY_ARM.get(arm)
    if legacy_arm is None:
        return None
    si = si_of(base)
    legacy = f"{si}__{legacy_arm}" if si is not None else legacy_arm
    return join_variant(legacy, variant)


def object_path_for(defenses_dir, cell_id: str) -> Path | None:
    """The on-disk artifact path for a cell id: directory keyed on ``cell.id``,
    filename keyed on ``cell.arm`` (:func:`methods.object_artifact_name`).
    Returns ``None`` for a baseline (no object).

    This is the fix at the exact site the defect lives. For the single-SI
    lane a bare arm ``x`` resolves to ``<defenses_dir>/x/object...`` — identical
    to the old ``_defense_object_path`` — so that lane is byte-for-byte
    unchanged. For a two-SI cell ``<si>__<arm>`` the DIRECTORY carries the SI,
    so the two SIs' objects for one arm never collide.

    LEGACY-ALIAS RESOLUTION (rename 2026-08-28, lora_r1_all ->
    lora_r4): for a cell whose arm has a legacy spelling
    (``methods.LEGACY_ARM_ALIASES``), the CANONICAL directory wins whenever
    its artifact exists; otherwise, when the LEGACY directory
    (``<si>__lora_r1_all/``) holds the artifact — the staged scale-12
    objects, and anything the upstream fitting path emits until it switches
    names — the legacy path is returned. When neither artifact exists the
    canonical path is returned, so ABSENT reporting names the path new fits
    will write. The sidecar checks downstream
    (``two_si_run._check_budget_sidecar``,
    ``registry_adapter._check_meta_json``) accept the legacy directory's
    recorded ids as the DECLARED alias of this same cell — a deliberate,
    table-driven acceptance, so the name mismatch between directory and
    embedded metadata can never pass a provenance check silently.

    A SERVE VARIANT (``<cell>@<variant>``) resolves the SAME path as its
    base: it re-serves the base cell's exact bytes under a hashed serve
    declaration and owns no artifact of its own, so the directory is
    keyed on the base id."""
    cell_id, _variant = split_variant(cell_id)
    arm = arm_of(cell_id)
    if arm is None:
        return None
    if cell_id.startswith("stacked__"):
        # A stacked cell REUSES the underlying Class A object — no new artifact
        # is trained or written. The object dir is keyed on the ``<si>__<arm>``
        # base (the mode prefix stripped), so ``stacked__X`` and ``X`` resolve
        # to the SAME object.pt and the stack serves the exact bytes the Class A
        # single served. Stripped FIRST so the legacy-alias fallback below
        # applies to the stacked spelling of an aliased arm too.
        cell_id = cell_id[len("stacked__"):]
    canonical = Path(defenses_dir) / cell_id / object_artifact_name(arm)
    legacy_id = legacy_cell_id(cell_id)
    if legacy_id is None or canonical.is_file():
        return canonical
    legacy = Path(defenses_dir) / legacy_id / object_artifact_name(arm)
    return legacy if legacy.is_file() else canonical


def resolve_defenses_dir(cfg: dict, *, repo_root=None) -> Path:
    """The ONE resolution of the compiled-objects dir, shared by the writers
    (``train_two_si_objects`` / ``fit_class_b``) and the readers
    (:func:`registry_from_cfg`, hence ``two_si_run`` / ``jailbreak_run``).

    Objects used to land inside the per-commit IMMUTABLE checkout
    (``<checkout>/out/two_si/defenses/cells``): any code change meant a new
    sha, a new checkout, an EMPTY objects dir — resume silently voided, every
    object refit, and eval could read a different checkout's dir than the
    fitters wrote. Resolution order:

    1. an ABSOLUTE ``cfg["defenses_dir"]`` (a submit-time ``--out`` lands
       here) — used as-is;
    2. ``TT_TWO_SI_DEFENSES_DIR`` — the one absolute, commit-independent dir
       under the private remote root, exported in every two-SI sbatch BODY
       (never via ``--export``, hard rule 1). Two-SI jobs only: other lanes
       (stage-2) keep their own ``defenses_dir`` and never export it;
    3. the relative config default rooted at ``repo_root`` — local/offline
       runs and tests, unchanged.

    A site placeholder in the value (``$SHARED/...``) is expanded first through
    ``sisteer.site`` (an unset one refuses by name), so a placeholder is never
    taken for a relative path and rooted under the checkout as a literal
    ``$SHARED`` directory.
    """
    from sisteer import site as _site

    ddir = _site.expand_path(cfg["defenses_dir"], key="defenses_dir")
    if ddir.is_absolute():
        return ddir
    env = os.environ.get("TT_TWO_SI_DEFENSES_DIR")
    if env:
        return Path(env).expanduser()
    if repo_root is not None:
        ddir = Path(repo_root) / ddir
    return ddir


# --------------------------------------------------------------------------- #
# Eval-time serving coefficient (alpha / dose) — namespace keying              #
# --------------------------------------------------------------------------- #
#
# ``scale`` in configs/two_si.yaml is the PRE-REGISTERED FIT-TIME operating
# point (1.0 — "no scale sweep") and never changes here. The eval-time
# ``--alpha`` grid re-serves the SAME fitted bytes at other APPLICATION-TIME
# coefficients (the ``scale`` argument of every object's ``apply()``, applied
# in the hooks at serving time, never baked into the artifact) — no refit
# anywhere. Read the grid within its bound:
#
#   Serving at alpha != fit scale is an UNDER-/OVER-APPLICATION of an object
#   optimised for a different operating point. A good number at, say,
#   alpha=0.25 supports "the deployed dose is too strong for THIS object";
#   it does NOT support "0.25 is the right target for the fit objective" —
#   an object properly refit at 0.25 could land better or worse and is
#   UNMEASURED.
#
# Corollary (observed in practice, twice): a defense metric can look BEST
# exactly where the dose has wrecked the model's capability. A dose-grid
# score must therefore never be readable without the alpha it was served
# at — which is why the served alpha is keyed into the transcript namespace
# (below), recorded in every per-shard sidecar, and stated in the aggregate
# next to the fit-time scale.


def format_alpha(alpha: float) -> str:
    """Canonical short form of a serving coefficient for keys and filenames
    (``0.25`` -> ``"0.25"``, ``1.0`` -> ``"1"``)."""
    return format(float(alpha), "g")


def alpha_run_namespace(private_run: str, alpha: float, fit_scale: float) -> str:
    """The private-store namespace for transcripts SERVED at ``alpha``.

    The pre-registered dose (``alpha == fit_scale``, the config ``scale``)
    keeps the namespace — and hence every path — byte-identical to a run
    with no override: no migration of existing stores, nothing orphaned.
    Any other dose gets its own sibling tree
    (``<private_run>_alpha<A>``), so transcripts served at different doses
    can never collide on the ``cell.id``-keyed layout inside it. The dose
    is NEVER parsed back out of the directory name — the recorded
    ``served_alpha`` in the ``cell_meta`` sidecars is the authoritative
    dose record (and what the aggregate's dose-consistency gate checks).
    """
    if float(alpha) == float(fit_scale):
        return private_run
    return f"{private_run}_alpha{format_alpha(alpha)}"


# --------------------------------------------------------------------------- #
# Registry — the authoritative, ordered two-SI matrix                          #
# --------------------------------------------------------------------------- #


def _registered(arms: tuple[str, ...]) -> list[str]:
    """The arms of ``arms`` that actually exist in ``methods.ARMS``, in order.
    Filters out unregistered arms (e.g. a config still naming the removed
    cast / circuit_breaker) so counts derive from what is real."""
    return [a for a in arms if a in ARMS]


def build_registry(
    defenses_dir,
    *,
    si_ids: tuple[str, ...] = DEFAULT_SI_IDS,
    class_a_arms: tuple[str, ...] = CLASS_A_ARMS,
    class_b_arms: tuple[str, ...] = CLASS_B_ARMS,
    stacked_cells: tuple[str, ...] = (),
    require_registered: bool = True,
) -> list[DefenseCell]:
    """The ordered two-SI defense-cell registry.

    Order (matches the on-disk layout, so the sbatch array index and
    the aggregate read the SAME sequence): ``none``; the prompted baselines in
    ``si_ids`` order; the Class A cells arm-major with the two SIs adjacent
    (``safety_refusal__psr_kl_all``, ``si_constitution__psr_kl_all``, …); the
    Class B objects. With ``require_registered`` (default) the Class A/B arm
    lists are filtered to arms present in ``methods.ARMS`` — an unregistered
    arm (a stale config naming the removed cast / circuit_breaker)
    contributes no cell, and the matrix count DERIVES from the result rather
    than a literal 16/19.

    ``stacked_cells`` is OPT-IN and defaults to empty, so a registry built
    without it is byte-for-byte the historical matrix (every existing run and
    every peer's live jobs are untouched). Each entry is a Class A base id
    ``<si>__<arm>``; it appends a ``stacked__<si>__<arm>`` cell served as
    mode='stacked' (the SI system prompt AND that arm's object together). The
    stacked cell REUSES the underlying Class A object (object_path_for keys the
    object dir on the base ``<si>__<arm>``), so nothing new is trained. Its
    over-refusal references are BOTH singles it stacks — the matching prompted
    baseline AND the Class A object cell — because a stack is expected to
    over-refuse no more than either part alone (the compounded-over-refusal
    check). Appended LAST so every earlier cell's array index is unchanged.
    """
    a_arms = _registered(class_a_arms) if require_registered else list(class_a_arms)
    b_arms = _registered(class_b_arms) if require_registered else list(class_b_arms)

    prompted_ids = tuple(f"prompted__{si}" for si in si_ids)
    cells: list[DefenseCell] = [
        DefenseCell("none", _BASELINE, None, None, None, ())
    ]
    for si in si_ids:
        cells.append(
            DefenseCell(f"prompted__{si}", _BASELINE, None, si, None, ()))
    # Class A: arm-major, both SIs adjacent, each referencing its matching
    # prompted arm (the over-refusal reference).
    for arm in a_arms:
        for si in si_ids:
            cid = f"{si}__{arm}"
            cells.append(DefenseCell(
                cid, "A", arm, si, object_path_for(defenses_dir, cid),
                (f"prompted__{si}",)))
    # Class B: bare arm id, SI-agnostic, referencing BOTH prompted arms
    # (non-inferiority against each prompted arm separately, not the base).
    for arm in b_arms:
        cells.append(DefenseCell(
            arm, "B", arm, None, object_path_for(defenses_dir, arm),
            prompted_ids))
    # Stacked cells (opt-in): SI prompt + compiled object together. Each base
    # is a Class A id ``<si>__<arm>``; the stacked id prefixes it with
    # ``stacked__``. The arm must be registered (same require_registered gate as
    # Class A/B) so a stale name contributes no cell. References are the two
    # singles it stacks.
    for base in stacked_cells:
        b_si, b_arm = si_of(base), arm_of(base)
        if b_si is None or b_arm is None:
            raise ValueError(
                f"stacked_cells entry {base!r} must be a Class A base id "
                "'<si>__<arm>' (got no si/arm split)")
        if require_registered and b_arm not in ARMS:
            continue
        cid = f"stacked__{base}"
        cells.append(DefenseCell(
            cid, _STACKED, b_arm, b_si, object_path_for(defenses_dir, cid),
            (f"prompted__{b_si}", base)))
    return cells


def registry_from_cfg(cfg: dict, *, repo_root=None) -> list[DefenseCell]:
    """Build the registry from a two-SI config dict (``si_ids`` + ``defenses_dir``).
    The dir resolves through :func:`resolve_defenses_dir`: ``repo_root``
    roots a relative ``defenses_dir`` locally, while on the cluster the
    ``TT_TWO_SI_DEFENSES_DIR`` export moves every reader to the one
    commit-independent dir the fitters wrote. Optional ``class_a_arms`` /
    ``class_b_arms`` config lists override
    the module defaults (additive: absent keys keep :data:`CLASS_A_ARMS` /
    :data:`CLASS_B_ARMS`), so ``configs/two_si.yaml`` is the submission-time
    record of the matrix and both drivers derive the same registry from it.

    A config carrying :data:`EXPLICIT_CELLS_KEY` takes the EXPLICIT branch
    instead (:func:`registry_from_explicit_cells`): the registry is exactly
    the enumerated cell ids, never a Cartesian product. A config WITHOUT
    that key is byte-for-byte the historical path.
    """
    if EXPLICIT_CELLS_KEY in cfg:
        return registry_from_explicit_cells(cfg, repo_root=repo_root)
    si_ids = tuple(cfg.get("si_ids", DEFAULT_SI_IDS))
    ddir = resolve_defenses_dir(cfg, repo_root=repo_root)
    return build_registry(
        ddir, si_ids=si_ids,
        class_a_arms=tuple(cfg.get("class_a_arms", CLASS_A_ARMS)),
        class_b_arms=tuple(cfg.get("class_b_arms", CLASS_B_ARMS)),
        stacked_cells=tuple(cfg.get("stacked_cells", ())))


# --------------------------------------------------------------------------- #
# Explicit-cell registry                                                      #
# --------------------------------------------------------------------------- #
#
# The Cartesian derivation above builds ``si_ids × class_a_arms`` plus a
# prompted baseline PER SI. For the instruct-transfer leg that construction is
# a defect: the base-fit cells are named through corpus
# PSEUDO-instructions (``constv2-*``), so "extend si_ids" would fabricate
# ``prompted__constv2-*`` baselines (an SI that is never a system prompt),
# cross every pseudo-id with EVERY registered Class A arm (garbage cells with
# no artifacts), and still never include the CE arms (not in CLASS_A_ARMS).
# The explicit mode replaces derivation with ENUMERATION: the config lists the
# exact, ordered cell ids of the run, each validated at build time, and
# exactly those cells — nothing else — are built.

#: The config key that switches :func:`registry_from_cfg` to the explicit
#: branch. Mutually exclusive with the Cartesian-defining keys below.
EXPLICIT_CELLS_KEY = "explicit_cells"

#: Config keys that DEFINE the Cartesian registry. Present next to
#: ``explicit_cells`` they would read as set and behave as unset — the exact
#: silently-dropped-configuration bug class — so the combination is refused.
_CARTESIAN_REGISTRY_KEYS = ("si_ids", "class_a_arms", "class_b_arms",
                            "stacked_cells")

#: Sentinel prefix for an ``explicit_cells`` slot the orchestrator has not
#: filled in yet (the base-fit winner cells are only known after winner
#: selection). An unfilled slot REFUSES the whole registry at run time —
#: never a silent skip that would publish a complete-looking, smaller run.
UNFILLED_CELL_PREFIX = "TBD"

#: The instruction category the fitting path gives its document-corpus
#: pseudo-instructions (data cells named through the SI slot). Their text is
#: never rendered as a system prompt, so a prompted/stacked cell over one is
#: refused — the fabricated-baseline hazard the explicit mode exists to kill.
CORPUS_CELL_CATEGORY = "corpus_cell"


def _resolve_si_or_error(si_id: str):
    """The Instruction for ``si_id`` via the SAME lookup the serving path
    uses (``eval.jailbreak.resolve_instruction`` → SAFETY_SI +
    ``instructions.BY_ID``), imported lazily so registration happens at RUN
    time — pseudo-instructions registered by the fitting path are honored
    without this module snapshotting the registry at import. Returns
    ``(instruction, None)`` or ``(None, error_string)``."""
    from .jailbreak import resolve_instruction

    try:
        return resolve_instruction(si_id), None
    except KeyError:
        return None, (f"SI id {si_id!r} does not resolve through "
                      "resolve_instruction (not SAFETY_SI and not in "
                      "instructions.BY_ID)")


def registry_from_explicit_cells(cfg: dict, *, repo_root=None) -> list[DefenseCell]:
    """The explicit-cell registry: EXACTLY the ordered cell ids in
    ``cfg["explicit_cells"]``, validated at build time — never a Cartesian
    product, never a fabricated baseline.

    Accepted id shapes (the standard grammar): ``none``,
    ``prompted__<si>``, ``<si>__<arm>`` (Class A), a bare registered
    ``<arm>`` (Class B), and ``stacked__<si>__<arm>``. Validation, per id:

    - an unfilled ``TBD…`` placeholder refuses the WHOLE registry, naming
      every unfilled slot (checked FIRST, so an unfilled shipped config can
      never limp past on other errors — and never silently skips)
    - duplicates are refused (a winner cell that IS the 1M cell must be
      deduplicated in the config, not listed twice)
    - the arm must be registered in ``methods.ARMS`` and the SI must resolve
      through ``resolve_instruction`` — both looked up at CALL time, so arms
      and pseudo-instructions registered by the fitting path are honored
    - a prompted/stacked cell whose SI is a document-corpus pseudo-instruction
      (category ``corpus_cell``) is refused: that text is never a system
      prompt, so ``prompted__constv2-*`` would be a fabricated baseline
    - every compiled cell's artifact must be STAGED under the resolved
      ``defenses_dir`` (:func:`object_path_for`, legacy-alias aware), with
      its ``train_budget_meta.json`` sidecar for Class A / stacked cells
      (Class B fits write no sidecar, mirroring the generate-side check;
      baselines are exempt — no object). An explicit-cell registry is an
      EVAL registry: everything it serves already exists.

    ``ni_reference_ids`` keep the non-inferiority semantics but reference
    only cells that are themselves LISTED (a reference to a cell outside the
    run would point the stats layer at a column that does not exist).

    Downstream contracts are untouched: ``defenses_dir`` resolves through
    :func:`resolve_defenses_dir`, object paths through
    :func:`object_path_for` (same dir naming, same legacy-alias fallback),
    and the returned cells are ordinary :class:`DefenseCell` rows — the
    intervention-hash, sidecar-gate and judge/aggregate plumbing read them
    exactly as they read a Cartesian registry's.
    """
    conflicting = [k for k in _CARTESIAN_REGISTRY_KEYS if k in cfg]
    if conflicting:
        raise ValueError(
            f"config sets {EXPLICIT_CELLS_KEY!r} AND the Cartesian registry "
            f"key(s) {conflicting} — these are mutually exclusive (the "
            "Cartesian keys would read as set and behave as unset). Remove "
            "one side.")
    listed = cfg.get(EXPLICIT_CELLS_KEY)
    if (not isinstance(listed, (list, tuple)) or not listed
            or not all(isinstance(x, str) and x.strip() for x in listed)):
        raise ValueError(
            f"{EXPLICIT_CELLS_KEY!r} must be a non-empty list of cell-id "
            f"strings, got {listed!r}")
    ids = [x.strip() for x in listed]

    unfilled = [x for x in ids if x.startswith(UNFILLED_CELL_PREFIX)]
    if unfilled:
        raise ValueError(
            f"{EXPLICIT_CELLS_KEY!r} has {len(unfilled)} UNFILLED "
            f"placeholder slot(s): {unfilled}. These are filled by the "
            "orchestrator after winner selection (replace each with the "
            "exact cell id; if the winner IS the 1M cell, remove the "
            "duplicate slot). Refusing to build a partial registry — an "
            "unfilled slot never silently skips.")

    dupes = sorted(x for x, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(
            f"{EXPLICIT_CELLS_KEY!r} lists duplicate cell id(s): {dupes} — "
            "each cell appears once (a winner cell that coincides with the "
            "1M cell is listed once, not twice).")

    id_set = set(ids)
    ddir = resolve_defenses_dir(cfg, repo_root=repo_root)
    prompted_listed = tuple(x for x in ids if x.startswith("prompted__"))
    cells: list[DefenseCell] = []
    problems: list[str] = []

    def _corpus_si_problem(cid: str, si: str) -> str | None:
        inst, err = _resolve_si_or_error(si)
        if err is not None:
            return f"{cid!r}: {err}"
        if getattr(inst, "category", None) == CORPUS_CELL_CATEGORY:
            return (f"{cid!r}: SI {si!r} is a document-corpus "
                    f"pseudo-instruction (category={CORPUS_CELL_CATEGORY!r}) "
                    "— its text is never served as a system prompt, so a "
                    "prompted/stacked cell over it is a fabricated baseline")
        return None

    def _staged_problem(cid: str, opath: Path, *, need_sidecar: bool) -> str | None:
        if not Path(opath).is_file():
            return (f"{cid!r}: no staged artifact at {opath} — an "
                    "explicit-cell registry is an eval registry; stage the "
                    "object under the resolved defenses_dir before building")
        # WHICH IDENTITY CONTRACT the dir declares decides which verdict
        # runs — the same dispatch train_budget.assert_object_identity
        # makes for the script-side consumers, folded into this
        # aggregation. Three sidecars, one each:
        # train_budget_meta.json (a training fit),
        # fit_provenance.json (the HPO-promotion lineage's fit — the SAME
        # fit-time facts under the stage-3 gate's native spelling, read
        # through sweep.read_stage3_fit_provenance, the one normaliser),
        # derived_meta.json (a derived object — never trained, so it has
        # no budget sidecar by construction; held to the stricter derived
        # verdict: bytes ↔ sidecar sha, sidecar ↔ the provenance embedded
        # in the bytes, ≥ 2 complete and mutually consistent parents). A
        # dir declaring more than one is ambiguous and refuses.
        from .train_budget import (DERIVED_META_NAME, FIT_PROVENANCE_NAME,
                                   IDENTITY_SIDECARS,
                                   derived_object_problems,
                                   fit_provenance_problems,
                                   object_identity_kind)

        identity = object_identity_kind(opath)
        base_cid = (cid[len("stacked__"):] if cid.startswith("stacked__")
                    else cid)
        if identity == "ambiguous":
            d = Path(opath).parent
            return (f"{cid!r}: {d} carries more than one identity sidecar "
                    f"({[s for s in IDENTITY_SIDECARS if (d / s).is_file()]})"
                    " — a dir declares exactly one; refusing to guess which "
                    "identity to validate")
        if identity == "derived":
            shared = derived_object_problems(opath, derived_id=base_cid)
            if shared:
                return f"{cid!r}: " + "; ".join(shared)
            return None
        if identity == "fit_provenance":
            shared = fit_provenance_problems(opath, base_cid)
            if shared:
                return f"{cid!r}: " + "; ".join(shared)
            return None
        if need_sidecar and not (Path(opath).parent
                                 / "train_budget_meta.json").is_file():
            return (f"{cid!r}: artifact present but no identity sidecar "
                    f"next to {opath} — a Class A / stacked cell must carry "
                    "its budget sidecar (D9), the HPO-promotion lineage's "
                    f"{FIT_PROVENANCE_NAME}, or a derived object's "
                    f"{DERIVED_META_NAME}")
        # The shared artifact-identity verdict (train_budget.
        # staged_cell_problems): sidecar names THIS cell (legacy alias
        # accepted), object bytes match the sidecar's sha, budget kind
        # valid (document schema through the one shared verdict), and any
        # PRESENT COMPLETE marker coherent. Registry build keeps
        # resume-grade strictness (a stated-absence sample_manifest_sha
        # passes here; the serving gates apply their own sidecar bar per
        # cell). A stacked cell validates the BASE cell dir it reuses, so
        # the recorded cell_id is compared against the base id, not the
        # stacked spelling.
        from .train_budget import staged_cell_problems

        base_cid = (cid[len("stacked__"):] if cid.startswith("stacked__")
                    else cid)
        shared = staged_cell_problems(opath, base_cid,
                                      require_sidecar=need_sidecar)
        if shared:
            return f"{cid!r}: " + "; ".join(shared)
        return None

    for cid in ids:
        if cid == "none":
            cells.append(DefenseCell("none", _BASELINE, None, None, None, ()))
            continue
        if cid.startswith("prompted__"):
            si = cid[len("prompted__"):]
            problem = _corpus_si_problem(cid, si)
            if problem:
                problems.append(problem)
                continue
            cells.append(DefenseCell(cid, _BASELINE, None, si, None, ()))
            continue
        if cid.startswith("stacked__"):
            base = cid[len("stacked__"):]
            b_si, b_arm = si_of(base), arm_of(base)
            if b_si is None or b_arm is None:
                problems.append(
                    f"{cid!r}: a stacked id must be 'stacked__<si>__<arm>'")
                continue
            if b_arm not in ARMS:
                problems.append(
                    f"{cid!r}: arm {b_arm!r} is not registered in methods.ARMS")
                continue
            problem = _corpus_si_problem(cid, b_si)
            if problem:
                problems.append(problem)
                continue
            opath = object_path_for(ddir, cid)
            problem = _staged_problem(cid, opath, need_sidecar=True)
            if problem:
                problems.append(problem)
                continue
            refs = tuple(r for r in (f"prompted__{b_si}", base)
                         if r in id_set)
            cells.append(DefenseCell(cid, _STACKED, b_arm, b_si, opath, refs,
                                     stacked=True))
            continue
        if "__" in cid:
            # The shared structural parse (identical to split("__", 1)
            # here: 'none'/'prompted__'/'stacked__' were dispatched above).
            si, arm = si_of(cid), arm_of(cid)
            if arm not in ARMS:
                problems.append(
                    f"{cid!r}: arm {arm!r} is not registered in methods.ARMS")
                continue
            _, err = _resolve_si_or_error(si)
            if err is not None:
                problems.append(f"{cid!r}: {err}")
                continue
            opath = object_path_for(ddir, cid)
            problem = _staged_problem(cid, opath, need_sidecar=True)
            if problem:
                problems.append(problem)
                continue
            ref = f"prompted__{si}"
            cells.append(DefenseCell(
                cid, "A", arm, si, opath,
                (ref,) if ref in id_set else ()))
            continue
        if cid in ARMS or cid in SERVED_ONLY_CLASS_B_IDS:
            # a served-only Class B id (e.g. `arditi`) is a Class B cell of an EVAL registry: the object
            # is staged upstream, the arm has no trainer here, and the blob's own stored kind is what
            # the serving adapter dispatches on (registry_adapter.ARM_OBJECT_KINDS)
            opath = object_path_for(ddir, cid)
            problem = _staged_problem(cid, opath, need_sidecar=False)
            if problem:
                problems.append(problem)
                continue
            cells.append(DefenseCell(cid, "B", cid, None, opath,
                                     prompted_listed))
            continue
        problems.append(
            f"{cid!r}: not a recognized cell id — expected 'none', "
            "'prompted__<si>', '<si>__<arm>', a bare registered Class-B "
            "arm, or 'stacked__<si>__<arm>'")

    if problems:
        raise ValueError(
            f"{EXPLICIT_CELLS_KEY!r} validation failed for "
            f"{len(problems)} cell id(s):\n  - " + "\n  - ".join(problems))
    return cells


def by_id(registry: list[DefenseCell]) -> dict[str, DefenseCell]:
    """``{cell.id: cell}`` — and a loud guard against a duplicate id (a dup would
    silently drop a column, the exact hazard this prevents)."""
    out: dict[str, DefenseCell] = {}
    for cell in registry:
        if cell.id in out:
            raise ValueError(f"duplicate cell id {cell.id!r} in registry")
        out[cell.id] = cell
    return out


def matrix_counts(registry: list[DefenseCell]) -> dict[str, int]:
    """Class/baseline/object/total counts, DERIVED from the registry (never a
    hard-coded 16/19)."""
    n_a = sum(1 for c in registry if c.class_ == "A")
    n_b = sum(1 for c in registry if c.class_ == "B")
    n_base = sum(1 for c in registry if c.class_ == _BASELINE)
    n_stacked = sum(1 for c in registry if c.class_ == _STACKED)
    return {
        "class_A": n_a,
        "class_B": n_b,
        "stacked": n_stacked,
        "baselines": n_base,
        # A stacked cell reuses an existing object, so it publishes NO new
        # artifact — it is not counted in "objects".
        "objects": n_a + n_b,  # the published objects (Class A + Class B)
        "total": len(registry),
    }


# --------------------------------------------------------------------------- #
# sbatch array decode — the index enumerates cell.ids, not arm names           #
# --------------------------------------------------------------------------- #


def array_decode(task_id: int, n_shards: int) -> tuple[int, int]:
    """Decode a Slurm array ``SLURM_ARRAY_TASK_ID`` into ``(cell_idx, shard)``
    using the layout ``cell_idx * n_shards + shard`` — the SAME shape the
    single-SI lane uses for ``defense_idx``, but the outer index now
    enumerates :class:`DefenseCell` ids. Keeping the two SI versions of an arm
    on DISTINCT cell indices (hence distinct shards) is what stops them
    colliding."""
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    return divmod(task_id, n_shards)


def cell_for_array_index(registry: list[DefenseCell], task_id: int,
                         n_shards: int) -> tuple[DefenseCell, int]:
    """The (cell, shard) a given array task owns. Raises loudly if ``cell_idx``
    is out of range for the registry (a stale ``--array`` upper bound must fail,
    never silently wrap)."""
    cell_idx, shard = array_decode(task_id, n_shards)
    if not 0 <= cell_idx < len(registry):
        raise IndexError(
            f"array task {task_id} decodes to cell_idx {cell_idx} with "
            f"n_shards={n_shards}, but the registry has {len(registry)} cells")
    return registry[cell_idx], shard


# --------------------------------------------------------------------------- #
# Run metadata (meta.json sidecar) — keyed on cell.id                         #
# --------------------------------------------------------------------------- #


def cell_meta(cell: DefenseCell, **extra) -> dict:
    """The ``meta.json`` sidecar for a cell's object. Carries the
    identity (``cell_id``/``class``/``si_id``/``arm``) plus whatever the trainer
    records (``fit_data``, ``positive_control``, ``gate``, geometry/``contrast``
    label, recorded ``completion_tokens``, …) passed as ``extra``. Keyed on
    ``cell_id`` so a Class B baseline can never be read back as an SI cell."""
    meta = {
        "cell_id": cell.id,
        "class": cell.class_,
        "si_id": cell.si_id,
        "arm": cell.arm,
        "ni_reference_ids": list(cell.ni_reference_ids),
    }
    meta.update(extra)
    return meta


# --------------------------------------------------------------------------- #
# Completeness — a cell can never silently drop from the aggregate              #
# --------------------------------------------------------------------------- #


def missing_cells(registry: list[DefenseCell], present_ids) -> list[str]:
    """Cell ids in the registry with no result present."""
    have = set(present_ids)
    return [c.id for c in registry if c.id not in have]


def assert_matrix_complete(registry: list[DefenseCell], present_ids,
                           *, refused=None) -> None:
    """Every cell id must be either PRESENT in the aggregate or explicitly
    REFUSED-with-reason (a gate refusal is a first-class, recorded outcome).
    A cell that is neither is a SILENTLY-DROPPED column, which is the
    exact failure the arm-only key produced — so raise.

    ``refused`` may be a mapping ``{cell_id: reason}`` or an iterable of ids.
    """
    refused_ids = set(refused or ())
    if isinstance(refused, dict):
        refused_ids = set(refused)
    have = set(present_ids)
    dropped = [c.id for c in registry
               if c.id not in have and c.id not in refused_ids]
    if dropped:
        raise ValueError(
            "matrix incomplete — these cells are neither published nor "
            f"refused-with-reason (silently dropped, D10): {dropped}")


# --------------------------------------------------------------------------- #
# Over-refusal / non-inferiority identity plumbing — the seam only            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ControlComparison:
    """One non-inferiority comparison the stats layer will fill in:
    ``cell_id`` over-refuses no more than its ``reference_id`` on ``control``.
    Identity/plumbing only — no statistics here."""

    cell_id: str
    reference_id: str
    control: str


def over_refusal_comparisons(registry: list[DefenseCell],
                             controls) -> list[ControlComparison]:
    """The (cell, reference, control) non-inferiority comparisons for every
    published object across the six controls. A Class A cell is compared
    against its MATCHING prompted arm; a Class B object against BOTH prompted
    arms separately — the reference is each prompted arm, never the undefended
    base. Baselines are references, not compared.

    This is the seam: it expresses "this cell, compared against that reference,
    on that control". The statistics (paired NI test, CI, margin) are the stats
    layer's job and are deliberately NOT implemented here.
    """
    out: list[ControlComparison] = []
    for cell in registry:
        if cell.arm is None:  # baselines are references, not compared
            continue
        for ref in cell.ni_reference_ids:
            for control in controls:
                out.append(ControlComparison(cell.id, ref, control))
    return out


def stacked_cell(base: "DefenseCell") -> "DefenseCell":
    """A stacked variant of a compiled Class-A cell: same object (arm, si_id,
    object_path) but served WITH its si_id's system instruction (mode=stacked).
    id is prefixed 'stacked__' so it resolves distinctly. Requires a real arm +
    si_id (a baseline cannot be stacked)."""
    import dataclasses
    if base.arm is None or base.si_id is None:
        raise ValueError(f"cannot stack cell {base.id!r}: needs both an arm and an si_id")
    return dataclasses.replace(base, id=f"stacked__{base.id}", class_=_STACKED,
                               stacked=True)
