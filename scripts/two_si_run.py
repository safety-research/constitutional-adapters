"""Two-SI comparison driver: per-CELL transcript generation + the aggregate.

The two-SI matrix evaluates every
:class:`~sisteer.eval.defense_cell.DefenseCell` — ``none``, the TWO prompted
baselines, the Class A ``<si>__<arm>`` cells and the Class B SI-agnostic
objects — under the same attack battery the single-SI driver runs. This driver
is the cell-id-keyed twin of ``scripts/jailbreak_run.py`` and REUSES its
machinery rather than forking it: the attack contracts (``bon_generate`` /
``run_gcg`` / ``_run_gated_attack``), the mode-0700 private-store layout
(``private_run_dir`` via ``_bon_defense_dir`` / ``_gated_shard_dir``), the
per-defense grading readers, and the sanitized-artifact containment. What is
NEW here is the identity layer: every CLI surface, private dir, array index,
meta sidecar, aggregation key and completeness check keys on ``cell.id``
(never the arm), and the aggregate carries the two-SI-specific
structure the statistics will consume.

Explicit-cell configs: a config carrying
``explicit_cells`` derives its registry as EXACTLY that ordered cell list
(``defense_cell.registry_from_cfg``'s explicit branch — validated ids, no
Cartesian extras, no fabricated ``prompted__<pseudo>`` baselines). generate/
aggregate/cells consume it through the same ``_registry`` seam; such runs
mix training provenances by design, so the aggregate's pool-regime block
becomes the ``explicit_cells`` statement (per-cell ``train_budget_meta.json``
is the authoritative record) and no pool manifest is required. Configs
without the key behave byte-identically to before the mode existed.

Three subcommands:

  generate  — on a GPU node. ``--cell <cell.id>`` picks the DefenseCell
              (validated against the config-derived registry);
              ``--attack`` picks the tier exactly as in jailbreak_run.
              Builds the target via ``build_defense_cell`` (object from the
              cell-id-keyed path; SI from ``cell.si_id``), runs the attack,
              persists to the private dir keyed on ``cell.id``, writes a
              ``cell_meta`` sidecar (identity + model pins + the
              target-forward count from the ``LLMHFSteered`` boundary), and
              resumes by goal_id (inherited from the attack contracts).
              ``--xstest`` runs the benign over-refusal pass for the cell.
  aggregate — on the grading host. Grades through the EXISTING StrongREJECT
              machinery — ``--transport live`` (default; the Low-Prio
              ``LiveStrongREJECTGrader``) or ``--transport batch`` — keyed on
              ``cell.id``, and calls ``assert_matrix_complete`` FIRST so a
              missing cell fails loudly instead of producing a
              complete-looking aggregate (§10; a gate-refused cell is a
              first-class recorded outcome read from its object-store
              meta.json). Emits, per cell: the config's control slots, the
              over-refusal non-inferiority reference ids (Class A -> its
              matching prompted arm, Class B -> BOTH prompted arms),
              and the training-pool regime; plus the top-level
              ``pool_regime`` block (matched_intersection or
              native_per_si, with both pool sizes, stated in BOTH cases) and
              the ``over_refusal_comparisons`` seam. The statistics
              themselves are deliberately NOT implemented here — this emits
              the structure the stats layer consumes.
  cells     — the sbatch array decode surface. With ``--task-id``/
              ``--nshards`` prints ``<cell.id> <shard>`` (via
              ``cell_for_array_index`` — a stale ``--array`` bound raises,
              never wraps); without, lists the registry with indices and the
              array upper bound.

The eval-time dose grid (``--alpha`` / ``--coeff``)
----------------------------------------------------
``generate`` takes ``--alpha`` (alias ``--coeff``; repeatable and/or
comma-separated) to serve the SAME fitted object at other application-time
coefficients, one full attack pass per dose — the coefficient reaches every
object's ``apply(scale=...)`` (hence both hook shapes in ``sisteer.hooks``
and the Class B arms' own hooks) exactly the way the config ``scale`` does
today. The default is the config's pre-registered fit-time ``scale``, which
keeps every path and byte of a flag-less invocation identical to before the
flag existed. A non-default dose writes to its own private namespace
(``defense_cell.alpha_run_namespace``) and records ``served_alpha`` next to
``fit_scale`` in every ``cell_meta`` sidecar; ``aggregate --alpha <A>``
grades that dose's tree, verifies the recorded ``served_alpha`` of every
transcript it is about to grade (the dose-consistency gate — before any
grading spend), and states both numbers in the published aggregate.

READ THE GRID WITHIN ITS BOUND: nothing is refit at any dose, and the
config ``scale`` (pre-registered, fixed) never changes. Serving at
alpha != fit scale under-/over-applies an object optimised for a different
operating point, so a good number at alpha=0.25 supports "the deployed dose
is too strong for this object" — NOT "0.25 is the right target for the fit
objective"; a refit-at-0.25 object is unmeasured and could land better or
worse. And a defense metric can look BEST exactly where the dose has
wrecked the model (observed twice), so no score from this lane is served
without the alpha next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # jailbreak_run, judge_batch

import jailbreak_run  # the single-SI driver whose contracts this lane reuses

REPO = Path(__file__).resolve().parents[1]

# The only two regimes the frozen pool manifest may declare, and the fixed
# (allowlist-safe) reader-facing notes for each. native_per_si is the
# FALLBACK and restates the headline as confounded — never silently.
_POOL_REGIMES = ("matched_intersection", "native_per_si")
# The explicit-cell registry mode: the config
# enumerates exact cell ids instead of deriving si_ids × arms, and the cells
# deliberately MIX training provenances (instruct-pool KL fits next to
# document-corpus CE fits), so no single pool manifest can describe the
# run. The regime says so, and points at the per-cell authoritative record.
_EXPLICIT_REGIME = "explicit_cells"
_EXPLICIT_POOL_NOTE = (
    "cells enumerated explicitly (explicit_cells config): training "
    "provenance is PER CELL by design — each object's train_budget_meta.json "
    "sidecar is the authoritative record; no shared pool regime applies")
# The budget-kind branch (sdf-base-steering §5.3) is NOT re-implemented
# here: kind resolution and the document-kind schema verdict live in
# sisteer.eval.train_budget (resolve_budget_kind / document_budget_problems,
# imported inside _check_budget_sidecar) — one verdict for every consumer,
# never a second copy that can drift.
_POOL_NOTE = {
    "matched_intersection": (
        "Class A objects trained on the MATCHED INTERSECTION: identical "
        "ordered prompt ids for both SIs, only the pos completion differs (D8)"),
    "native_per_si": (
        "Class A objects trained on NATIVE per-SI pools: sizes and "
        "composition differ, so the headline SI contrast is CONFOUNDED by "
        "pool composition (D8 fallback, flagged per D26)"),
}
# Class B objects are fit once on harmful/benign data with no SI in the
# signal — the pair-pool regime does not apply to them.
_POOL_SI_FREE = "si_free_harmful_benign_fit_once"
# The explicit not-yet-run marker for control slots: a pending control
# is stated, never silently absent — the aggregate must not look complete.
_CONTROL_PENDING = "pending_control_not_yet_run"
# Keys a validated controls summary (cell_controls_run.py judge) may put in a
# slot payload — a fixed enum extended into allowed_strings so the sanitizer
# accepts filled slots; payload VALUES (hashes/labels, our own recorded
# strings) are allowlisted from the validated summaries themselves, like the
# gate-refusal reasons below.
_CONTROL_PAYLOAD_KEYS = (
    "acc", "n", "extraction_failures", "full_compliance_frac",
    "partial_refusal_frac", "full_refusal_frac", "safe", "unsafe",
    "winrate_vs_base", "tie_frac", "order_consistency",
    "winrate_by_len_stratum", "pref_len_assoc",
    "rubric_sha", "provenance", "source",
    # judge-abstention accounting (bound + cap, tiers/judged.py): counts
    # and bounds ride every judged payload; the by-reason breakdown's
    # keys are the two reason literals
    "n_graded", "n_abstained", "abstained_by_reason",
    "refusal", "max_tokens",
    "orb_refused_lo", "orb_refused_hi",
    "refused_lo", "refused_hi",
    "winrate_lo", "winrate_hi",
    # the served STANDARD GSM8K's slot payload (PW3; the paired-200 subset
    # since U4(a); the literal twin of
    # sisteer.capeval.tinycap.contracts.GSM8K_SLOT_PAYLOAD_FIELDS, pinned
    # equal by test_two_si_controls_fill): the payload names its instrument
    "task", "item_set", "framework", "engine", "thinking", "temperature",
)
# The keys a loaded summary may carry BESIDE the control slots for a number that
# is deliberately NOT a slot any more (PW5-1): the retired s2parity/gsm8k_0shot
# payload of a pre-PW3 summary, which cell_controls.load_controls_summaries
# reports under this key (the literal twin of
# sisteer.capeval.tinycap.contracts.GSM8K_LEGACY_SLOT, pinned equal by
# test_two_si_controls_fill) while the `gsm8k` slot reads pending. Carried in
# the aggregate under `legacy_controls`, never under `controls`.
_LEGACY_CONTROL_KEYS = ("gsm8k_legacy_s2parity",)


def _registry(cfg: dict):
    """The config-derived cell registry (si_ids + class_a_arms/class_b_arms +
    defenses_dir — the one source of identity truth)."""
    from sisteer.eval.defense_cell import registry_from_cfg

    return registry_from_cfg(cfg, repo_root=REPO)


#: The goal-set restrictions ``bon.goal_subset`` may name. ``jbb`` keeps only
#: the JBB half of the eval set (whose SIZE is still ``--confirmatory``'s call:
#: 100 confirmatory / 50 pilot) and drops the HarmBench half.
_GOAL_SUBSETS = ("jbb",)


def _goal_subset(cfg: dict, attack: str | None) -> str | None:
    """The goal-set restriction in force for ``attack``, or None.

    ATTACK-SCOPED on purpose, and today only ``bon`` can declare one. BoN
    draws ``bon.n_max`` responses PER GOAL, so it is the one family whose cost
    scales with the goal count steeply enough to be worth trading breadth for
    depth (best-of-75 on 100 goals reads the refusal-vs-robustness separation
    that best-of-29 on 259 does not). Every other family stays on the full
    set, so restricting BoN can never silently shrink their denominators."""
    if attack != "bon":
        return None
    sub = (cfg.get("bon") or {}).get("goal_subset")
    if sub is None:
        return None
    if sub not in _GOAL_SUBSETS:
        raise SystemExit(f"[two-si] bon.goal_subset: {sub!r} is not one of "
                         f"{list(_GOAL_SUBSETS)}")
    return str(sub)


def _eval_goals(cfg: dict, *, confirmatory: bool, attack: str | None = None):
    """The eval goal set — jailbreak_run's recipe, plus the loaders' ``path=``
    override read from each dataset block so offline tests point at
    repo-local fixtures (the existing test convention).

    ``attack`` opts the caller into the attack-scoped ``bon.goal_subset``
    restriction (see ``_goal_subset``). It must be threaded through at BOTH
    the generate and the aggregate call sites: ``bon_curves`` refuses any
    expected goal with fewer than ``n_max`` responses, so an aggregate that
    expected a wider goal set than generation ran would fail loudly — the
    restriction is a property of the run, not of one command."""
    from sisteer.eval.jailbreak_data import load_harmbench, load_jbb, stratify_jbb

    hb = cfg["datasets"]["harmbench"]
    jbb = cfg["datasets"]["jbb"]
    subset = _goal_subset(cfg, attack)
    goals = [] if subset == "jbb" else load_harmbench(
        split=hb["eval_split"], revision=hb["revision"],
        path=hb.get("path"),
        functional=hb["eval_functional"],
        exclude_copyright=hb["exclude_copyright"])
    n_jbb = cfg["prereg"]["confirmatory_goals"] if confirmatory else jbb["harmful_subset"]
    goals += stratify_jbb(load_jbb("harmful", revision=jbb["revision"],
                                   path=jbb.get("path")),
                          n=n_jbb, seed=jbb["stratify_seed"])
    return goals


# --------------------------------------------------------------------------- #
# the dose grid — served alphas, per-dose namespaces, the consistency gate     #
# --------------------------------------------------------------------------- #


def _served_alphas(args, cfg: dict) -> list[float]:
    """The serving-coefficient grid of this invocation, from ``--alpha`` /
    ``--coeff`` (repeatable and/or comma-separated), order-preserving and
    deduplicated. No flag — including every pre-existing caller, whose args
    namespace has no ``alpha`` attribute at all — resolves to the config's
    pre-registered fit-time ``scale``, so the default path is unchanged."""
    fit = float(cfg.get("scale", 1.0))
    raw = getattr(args, "alpha", None) or []
    alphas: list[float] = []
    for entry in raw:
        for part in str(entry).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                a = float(part)
            except ValueError:
                raise SystemExit(
                    f"[two-si] --alpha: {part!r} is not a number")
            if a not in alphas:
                alphas.append(a)
    return alphas or [fit]


def _alpha_cfg(cfg: dict, alpha: float) -> dict:
    """``cfg`` with the private-store namespace keyed to the served dose.
    The pre-registered dose (== config ``scale``) keeps the namespace — and
    every derived path — byte-identical to today (no migration, nothing
    orphaned); any other dose gets its own sibling tree. See
    ``defense_cell.alpha_run_namespace`` for the bound on what a non-default
    dose's numbers do and do not support."""
    from sisteer.eval.defense_cell import alpha_run_namespace

    return {**cfg, "private_run": alpha_run_namespace(
        cfg["private_run"], alpha, float(cfg.get("scale", 1.0)))}


# The published statement accompanying any off-fit-dose aggregate. Fixed
# text (allowlisted for the sanitizer), so the bound travels WITH the
# numbers, not just in this file's comments.
_ALPHA_NOTE = (
    "served_alpha differs from fit_scale: the SAME fitted object bytes, "
    "re-served at an eval-time coefficient the object was NOT optimised "
    "for (an under-/over-application; nothing was refit at any dose). "
    "Numbers at this dose support claims about the SERVING dose of this "
    "object only — never that this alpha is the right FIT target; an "
    "object refit at this dose is unmeasured. Read every defense score "
    "next to served_alpha: a defense metric can look best exactly where "
    "the dose has destroyed capability.")


def _assert_served_alpha(cfg: dict, root, attack: str, present_ids,
                         *, alpha: float, fit_scale: float) -> None:
    """The dose-consistency gate, run BEFORE any grading spend: every
    transcript this aggregate is about to grade must have been SERVED at the
    dose the aggregate claims.

    Reads the ``cell_meta*`` sidecars the generate phase wrote next to the
    transcripts (the authoritative dose record — never the directory name)
    and refuses on any recorded ``served_alpha`` other than ``alpha``.
    Sidecars written before the alpha seam carry no ``served_alpha``; that
    is acceptable ONLY at the pre-registered dose (``alpha == fit_scale``),
    because pre-seam code could only ever serve the config scale — at any
    other dose the tree can only have been written by alpha-aware code, so
    a missing record is itself a mismatch (e.g. transcripts copied between
    dose trees). This is the two-SI twin of the capability battery's
    equivalence gate (which reads its serving coefficient via
    ``SISTEER_EQUIV_SCALE``): it keys on recorded evidence, so it still
    catches a genuine mismatch and has no override that could turn it into
    a rubber stamp."""
    from sisteer.eval.defense_cell import format_alpha

    at_fit_dose = float(alpha) == float(fit_scale)
    for cell_id in present_ids:
        metas = sorted(
            (Path(root) / cfg["private_run"] / cell_id).glob("cell_meta*.json"))
        metas += sorted(
            (Path(root) / cfg["private_run"] / attack / cell_id).glob(
                "shard*/cell_meta.json"))
        if not metas and not at_fit_dose:
            raise SystemExit(
                f"[two-si] {cell_id}: transcripts in the "
                f"alpha={format_alpha(alpha)} namespace have no cell_meta "
                "sidecar recording a served_alpha — cannot verify they were "
                "served at this dose. Regenerate them with `generate --alpha "
                f"{format_alpha(alpha)}` rather than copying transcripts "
                "between dose trees.")
        for m in metas:
            try:
                rec = json.loads(m.read_text()).get("served_alpha")
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"[two-si] unreadable cell_meta at {m}: {e} — the dose "
                    "record is authoritative; fix it rather than grading "
                    "around it.")
            if rec is None:
                if at_fit_dose:
                    continue  # pre-seam sidecar; only the config scale existed
                raise SystemExit(
                    f"[two-si] {m}: no served_alpha recorded, but this "
                    f"aggregate claims alpha={format_alpha(alpha)} != the "
                    f"fit scale {format_alpha(fit_scale)} — a dose-unverified "
                    "transcript cannot be graded as this dose.")
            if float(rec) != float(alpha):
                raise SystemExit(
                    f"[two-si] {m}: transcripts were SERVED at "
                    f"alpha={format_alpha(float(rec))} but this aggregate "
                    f"claims alpha={format_alpha(alpha)} — refusing to grade "
                    "a dose mismatch (a score published here would be "
                    "attributed to the wrong dose).")


# --------------------------------------------------------------------------- #
# generate — keyed on cell.id                                                  #
# --------------------------------------------------------------------------- #


def _write_cell_meta(run_dir, cell, cfg: dict, alpha: float, *,
                     filename: str, defense=None, **extra) -> Path:
    """The ``cell_meta`` sidecar in the PRIVATE run dir — the identity record
    (cell_id/class/si_id/arm/ni_reference_ids) plus the model pins and
    whatever the run measured (attack, shard, ``n_target_forwards`` counted at
    the ``LLMHFSteered`` boundary). ``filename`` varies so concurrent array
    shards never clobber: the BoN layout shares ONE dir per cell (per-shard
    filename), the gated layout is per-shard already (fixed filename).
    Written inside the existing mode-0700 dir under the private umask, so the
    file lands 0600.

    Records BOTH coefficients, always: ``fit_scale`` (the pre-registered
    config ``scale`` the object was optimised at) and ``served_alpha`` (the
    application-time coefficient THIS run's transcripts were generated
    under). They are different numbers whenever the dose grid is in play,
    and a transcript must never be readable as having been served at the
    dose it was fit at — the aggregate's ``_assert_served_alpha`` gate keys
    on this record. (For a baseline cell the coefficient is inert — there
    is no object to scale — but it is recorded uniformly so every sidecar
    in a dose tree states its dose.)

    Also records the positions legs, read off the DEFENSE the transcripts
    were generated with (``build_defense_cell`` stamps ``positions_served``
    on every build and ``positions_at_fit`` on compiled cells — the designed
    source, never re-derived here): equal by construction on this path,
    because ``cmd_generate`` serves with ``positions="fit"`` and the
    fit==serve guard raises on a mismatch. Recorded anyway so a transcript
    states the leg it was generated under instead of leaving it to be
    inferred from the code version that wrote it. (Absent only for a test
    fake that bypasses the real build.)"""
    from sisteer.eval.defense_cell import cell_meta
    from sisteer.eval.jailbreak_artifacts import private_umask

    for key in ("positions_served", "positions_at_fit"):
        leg = getattr(defense, key, None)
        if leg is not None:
            extra.setdefault(key, leg)
    meta = cell_meta(cell,
                     model=cfg["model"],
                     model_revision=cfg.get("model_revision"),
                     private_run=cfg["private_run"],
                     fit_scale=float(cfg.get("scale", 1.0)),
                     served_alpha=float(alpha),
                     **extra)
    path = Path(run_dir) / filename
    with private_umask():
        path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return path


def _gated_seams(cfg, attack: str) -> dict:
    """The attacker-side seams of a gated tier (pair/tap/crescendo/
    actorattack), built ONCE per invocation and reused across the dose grid —
    the attacker model must not be re-loaded per alpha."""
    attacker_model, attacker_tok = jailbreak_run._load_attacker(cfg)
    samp = jailbreak_run._attacker_sampling(cfg, attack)
    return {
        "scorer_fn": jailbreak_run._make_scorer_fn(
            attacker_model, attacker_tok,
            max_new_tokens=jailbreak_run._scorer_max_new_tokens(cfg)),
        "attacker_factory": jailbreak_run._make_attacker_factory(
            attacker_model, attacker_tok, temperature=samp["temperature"],
            top_p=samp["top_p"],
            max_new_tokens=jailbreak_run._attacker_max_new_tokens(cfg)),
        "attacker_fn": jailbreak_run._make_attacker_fn(
            attacker_model, attacker_tok, temperature=samp["temperature"],
            top_p=samp["top_p"],
            max_new_tokens=jailbreak_run._attacker_max_new_tokens(cfg)),
    }


def _check_budget_sidecar(cell, cfg: dict) -> None:
    """Refuse to generate from an object whose budget provenance cannot be
    verified, or was trained under a DIFFERENT completion-token budget than
    the config now declares: the eval-time decode budget deliberately
    equals the shared training cap (``attacks.target_max_new_tokens ==
    completion_tokens``), so a mixed-budget object would silently break the
    supervision-vs-decode parity between arms.

    Class semantics:

    - baselines (no object) and a Class A object file that does not exist
      yet are skipped — there is nothing to mis-pair, and the object
      loader fails on its own terms;
    - a **Class A** object that EXISTS must have a readable, COMPLETE
      sidecar: exact ``cell_id``, ``completion_tokens`` equal to the
      config's, and an ``object_sha256`` matching the on-disk bytes. A
      missing or pre-stamp legacy sidecar fails LOUDLY with the explicit
      migration path (retrain via train_two_si_objects.py --overwrite, or
      hand-stamp after verifying provenance) — never silent acceptance;
    - a **Class B** object (``fit_class_b`` — SI-free by construction, no
      ``train_budget_meta.json``) is exempt when the sidecar is absent;
      if one IS present it must validate like Class A's.

    BUDGET-KIND BRANCH (sdf-base-steering §5.3/§12): a sidecar carrying
    ``training_budget_kind: "document_tokens"`` records a fit trained on
    document tokens under a sequence cap — it has NO completion-token budget
    by construction (``completion_tokens`` is explicitly null), so the
    completion-budget equality above cannot apply. Such a sidecar must
    instead be complete on ITS schema (``sequence_max_len`` +
    ``sample_manifest_sha``), must record ``completion_tokens: null``
    (a document fit recording a completion budget is a schema violation,
    refused), and still passes the cell-id and object-hash checks. A sidecar
    WITHOUT the key keeps today's SI-pair path byte-for-byte, the legacy
    absent-``completion_tokens``-config 512 default included; an unknown
    kind is refused, never silently routed.

    The kind resolution and the document-kind schema verdict are
    ``train_budget.resolve_budget_kind`` / ``document_budget_problems`` —
    the ONE shared implementation every sidecar consumer routes through.
    This call site adds only its own serving strictness
    (``require_sample_manifest_sha=True``: a null sha is a stated absence,
    fine to resume/retrain over, but never to GENERATE from) and keeps the
    identity checks (cell_id, object_sha256) that apply to both kinds.

    CPU-cheap and called BEFORE the model loads, so a stale object fails
    in seconds, not after a GPU allocation."""
    from sisteer.eval.defense_cell import sha256_file
    from sisteer.eval.train_budget import (document_budget_problems,
                                           resolve_budget_kind)

    from sisteer.eval.train_budget import (IDENTITY_SIDECARS,
                                           assert_derived_object,
                                           assert_fit_provenance_cell,
                                           object_identity_kind)

    if getattr(cell, "object_path", None) is None:
        return                          # baselines: no object, no budget
    obj_path = Path(cell.object_path)
    if not obj_path.exists():
        return                          # nothing on disk yet: loader's problem
    identity = object_identity_kind(obj_path)
    if identity == "ambiguous":
        raise SystemExit(
            f"[two-si] cell {cell.id!r}: {obj_path.parent} carries more than "
            "one identity sidecar "
            f"({[s for s in IDENTITY_SIDECARS if (obj_path.parent / s).is_file()]})"
            " — a dir declares exactly one; refusing to guess which to "
            "validate.")
    if identity == "fit_provenance":
        # The HPO-promotion lineage's fit record (qwen's scale-12 si12
        # cells): the SAME fit-time facts as a budget sidecar under the
        # stage-3 gate's native spelling, but it records no completion
        # budget, so the completion-budget equality below is undefined
        # for it — not waived. Its identity is proven by the shared
        # fit-record verdict, and the DECODE budget the config declares
        # governs generation exactly as it does for a baseline.
        base_cid = (cell.id[len("stacked__"):]
                    if cell.id.startswith("stacked__") else cell.id)
        assert_fit_provenance_cell(obj_path, base_cid, consumer="two-si")
        return
    if identity == "derived":
        # A DERIVED object (sdf-base-steering round 2) was never trained,
        # so it has NO completion-token budget to compare against the
        # config — the equality below is undefined for it, not waived. Its
        # identity is proven instead by the shared derived verdict (bytes
        # <-> sidecar sha, sidecar <-> the provenance embedded in the
        # bytes, complete and mutually consistent parent identities), and
        # the DECODE budget the config declares still governs generation
        # exactly as it does for a baseline.
        base_cid = (cell.id[len("stacked__"):]
                    if cell.id.startswith("stacked__") else cell.id)
        assert_derived_object(obj_path, consumer="two-si",
                              derived_id=base_cid)
        return
    meta_path = obj_path.parent / "train_budget_meta.json"
    if not meta_path.exists():
        if getattr(cell, "class_", None) == "B":
            return                      # Class B objects carry no sidecar
        raise SystemExit(
            f"[two-si] cell {cell.id!r}: object at {obj_path} has NO budget "
            f"sidecar ({meta_path} missing) — a Class A object must carry "
            "its train_budget_meta.json. A legacy object must be migrated "
            "deliberately: retrain it (train_two_si_objects.py --overwrite) "
            "or restore its sidecar; never generate from an unverifiable "
            "object.")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"[two-si] unreadable budget sidecar for cell {cell.id!r} at "
            f"{meta_path}: {e} — fix or retrain the cell; the sidecar is "
            "the authoritative budget record.")
    try:
        kind = resolve_budget_kind(meta)   # unknown kind refuses loudly
    except ValueError as e:
        raise SystemExit(
            f"[two-si] cell {cell.id!r}: budget sidecar {meta_path} — {e}")
    if kind is None:
        required = ("cell_id", "completion_tokens", "object_sha256")
    else:
        # document_tokens: the schema fields are the shared verdict's job
        # (document_budget_problems, below); only the identity fields every
        # sidecar must carry are required here.
        required = ("cell_id", "object_sha256")
    missing = [k for k in required if meta.get(k) is None]
    if missing:
        raise SystemExit(
            f"[two-si] cell {cell.id!r}: INCOMPLETE budget sidecar "
            f"{meta_path} — missing {missing}. A sidecar predating the "
            "object-hash stamp must be migrated deliberately: retrain "
            "(train_two_si_objects.py --overwrite), or after verifying "
            f'provenance add "object_sha256": "{sha256_file(obj_path)}".')
    if meta["cell_id"] != cell.id:
        # The ONE accepted mismatch: the LEGACY spelling of this same cell
        # (methods.LEGACY_ARM_ALIASES — lora_r4 cells reading a staged
        # `<si>__lora_r1_all/` directory whose sidecar records the name the
        # fitting lane emitted). Table-driven and announced, so the alias
        # never passes silently; ANY other cell_id still refuses.
        from sisteer.eval.defense_cell import legacy_cell_id

        legacy = legacy_cell_id(cell.id)
        if legacy is not None and meta["cell_id"] == legacy:
            print(f"[two-si] cell {cell.id!r}: budget sidecar records the "
                  f"LEGACY spelling {legacy!r} (accepted alias — "
                  "methods.LEGACY_ARM_ALIASES; the artifact predates the "
                  "lora_r4 rename).")
        else:
            raise SystemExit(
                f"[two-si] budget sidecar at {meta_path} records cell_id="
                f"{meta['cell_id']!r} but this cell is {cell.id!r} — the object "
                "store is corrupted; refusing to generate from it.")
    if kind is None:
        current = int(cfg.get("completion_tokens", 512))
        if int(meta["completion_tokens"]) != current:
            raise SystemExit(
                f"[two-si] cell {cell.id!r}: object at {obj_path} was "
                f"trained with completion_tokens={meta['completion_tokens']}, "
                f"but the config now says {current} — refusing to mix training "
                "budgets in one matrix (D9). Retrain the cell at the current "
                "budget (train_two_si_objects.py --overwrite) or restore the "
                "config it was trained under.")
    else:
        # The ONE document-kind schema verdict (train_budget), with this
        # call site's serving strictness: a null sample_manifest_sha is a
        # stated absence — resumable, but never served.
        problems = document_budget_problems(
            meta, require_sample_manifest_sha=True)
        if problems:
            raise SystemExit(
                f"[two-si] cell {cell.id!r}: INCOMPLETE or invalid "
                f"document-token budget sidecar {meta_path} — "
                + "; ".join(problems) +
                ". Retrain the cell (train_two_si_objects.py --overwrite) "
                "or fix its provenance deliberately; never generate from "
                "an unverifiable object.")
    actual_sha = sha256_file(obj_path)
    if meta["object_sha256"] != actual_sha:
        raise SystemExit(
            f"[two-si] cell {cell.id!r}: object_sha256 mismatch (sidecar "
            f"{meta['object_sha256'][:12]}… != object {actual_sha[:12]}…) — "
            "a stale object/sidecar pair (crashed save or out-of-band "
            "overwrite). Retrain the cell rather than generating from "
            "unverifiable bytes.")


def cmd_generate(args, cfg) -> None:
    from sisteer.eval.attacks import REPLAYABLE_FAMILIES
    from sisteer.eval.defense_cell import by_id, format_alpha
    from sisteer.eval.jailbreak import build_defense_cell
    from sisteer.models import load_model_and_tokenizer

    cells = by_id(_registry(cfg))
    if args.cell not in cells:
        # A bare single-SI arm name (e.g. `vector`) is NOT a two-SI cell —
        # the surface accepts registry cell.ids only, loudly.
        raise SystemExit(
            f"[{args.attack}] unknown --cell {args.cell!r}; known cell ids: "
            f"{sorted(cells)}")
    cell = cells[args.cell]
    _check_budget_sidecar(cell, cfg)  # Fail BEFORE the model loads

    # The dose grid (default: the ONE pre-registered fit-time scale). Every
    # dose serves the SAME object bytes; only the application-time
    # coefficient — hooks-applied, never baked into the artifact — differs.
    alphas = _served_alphas(args, cfg)

    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )

    shard_goals = None
    if not args.xstest:
        goals = _eval_goals(cfg, confirmatory=args.confirmatory,
                            attack=args.attack)
        shard_goals = goals[args.shard::args.nshards]

    seams = None
    if (not args.xstest and args.attack not in ("bon", "gcg")
            and args.attack not in REPLAYABLE_FAMILIES):
        seams = _gated_seams(cfg, args.attack)

    for alpha in alphas:
        if len(alphas) > 1:
            print(f"[{args.attack}] === dose alpha={format_alpha(alpha)} "
                  f"(fit scale {format_alpha(float(cfg.get('scale', 1.0)))}; "
                  "same object bytes, serving coefficient only) ===")
        # Object from the cell-id-keyed path, SI from cell.si_id, serving
        # coefficient = this dose (default: the pre-registered config scale).
        # vLLM for the none/prompted cells when the config asks for it. The
        # positive side already runs on vLLM (sampling.engine); leaving this
        # side on HF makes a pool whose two halves come from different
        # engines, and the measured HF/vLLM acceptance gap on this workload
        # is +25.8pp -- larger than most effects being measured with it.
        # for_cell REFUSES a compiled cell rather than silently dropping its
        # forward hooks and labelling undefended text as defended.
        _gfn = None
        if _neg_engine(cfg) == "vllm":
            from sisteer.sampling.generate_fn import VllmGenerateFn
            _gfn = _vllm_generate_fn(cfg, cell)
        # positions="fit": serve each object at the leg it was FITTED on
        # (the artifact's persisted positions_at_fit — decode-only for the
        # Class C fits, all-position for token_faithful), with the
        # fit==serve guard raising on any mismatch. Before this the default
        # was "all", which silently unmasked every response-fit object
        # served through this path (audit 2026-08-27).
        defense = build_defense_cell(model, tok, cell, scale=alpha,
                                     positions="fit", generate_fn=_gfn)
        _generate_one(args, _alpha_cfg(cfg, alpha), cell, defense,
                      alpha=alpha, shard_goals=shard_goals, seams=seams)



def _pool(cfg) -> dict:
    """The pool block. EVERY neg-side vLLM key is read through this, because
    reading some of them from the top level is how job 212618 failed: the
    engine switch resolved from cfg["pool"] and turned vLLM ON, while the
    settings beside it resolved from the top level, missed, and fell back to
    defaults. gpu_memory_utilization went out at 0.90 instead of the declared
    0.85 and vLLM refused to start -- a config value that is present, correct,
    and never applied."""
    return cfg.get("pool") or {}


def _neg_engine(cfg) -> str:
    """Engine for the none/prompted generation path. Declared in config so a
    pool's two halves cannot silently disagree; defaults to hf, which is what
    every pool before 2026-08-26 used."""
    return str(_pool(cfg).get("neg_generation_engine", "hf"))


def neg_vllm_settings(cfg) -> dict:
    """The declared neg-side engine settings, resolved from ONE scope.

    Returns only keys the config actually declares, so anything unset falls
    through to ``sampling.engine.VLLM_ENGINE_DEFAULTS`` rather than being
    restated here -- a second copy of a default is a second thing to drift.
    """
    pool, out = _pool(cfg), {}
    if "vllm_max_model_len" in pool:
        out["max_model_len"] = int(pool["vllm_max_model_len"])
    if "vllm_gpu_memory_utilization" in pool:
        out["gpu_memory_utilization"] = float(
            pool["vllm_gpu_memory_utilization"])
    return out


_VLLM_GFN_CACHE = {}


def _vllm_generate_fn(cfg, cell):
    """One vLLM engine per process, reused across doses -- constructing a
    second LLM in the same process would try to claim the GPU twice."""
    from sisteer.sampling.generate_fn import VllmGenerateFn

    mode = getattr(cell, "arm", None) or getattr(cell, "mode", None) or "none"
    # The config flag means "vLLM WHERE IT IS SAFE", not "vLLM everywhere".
    # A compiled cell falls back to HF because vLLM cannot run its forward
    # hooks -- and it says so, because an engine that varies by cell and is
    # not stated is the whole failure this work exists to remove. Pool
    # generation is unaffected: the pool is built from none/prompted/pos,
    # all of which are vLLM. Compiled cells appear in eval sweeps, not here.
    if mode not in ("none", "prompted"):
        print(f"[generate] neg-side engine=hf cell={mode} "
              f"(compiled cell: vLLM cannot run its forward hooks)",
              flush=True)
        return None
    key = (cfg["model"], cfg.get("model_revision"), mode)
    if key not in _VLLM_GFN_CACHE:
        gfn = VllmGenerateFn.for_cell(
            mode, cfg["model"], revision=cfg.get("model_revision"),
            seed=int(cfg.get("seed", 0)), **neg_vllm_settings(cfg))
        _VLLM_GFN_CACHE[key] = gfn
        # Echo the RESOLVED settings, not the declared ones. 212618 printed
        # "engine=vllm" and nothing else, so a run whose settings had silently
        # reverted to defaults looked identical to one that honoured the
        # config. The engine banner two seconds later held the real number.
        shown = {k: v for k, v in gfn._cfg.items()
                 if k not in ("model", "revision")}
        print(f"[generate] neg-side engine=vllm cell={mode} settings={shown}",
              flush=True)
    return _VLLM_GFN_CACHE[key]


def _generate_one(args, cfg, cell, defense, *, alpha, shard_goals,
                  seams=None) -> None:
    """One (cell, attack, dose) generation pass. ``cfg`` is already the
    dose-keyed config (``_alpha_cfg``), so every private dir derived from
    ``cfg['private_run']`` lands in this dose's tree — the pre-registered
    dose keeps today's tree byte-identically."""
    root = jailbreak_run._private_root(cfg, args.private_root)

    if args.xstest:
        # Benign over-refusal pass: per-CELL, attack-independent (BoN-layout
        # dir, resume by goal_id) — the xstest slot of the config's controls.
        run_dir = jailbreak_run._bon_defense_dir(cfg, root, cell.id)
        jailbreak_run._xstest_generate(defense, cfg, run_dir)
        _write_cell_meta(run_dir, cell, cfg, alpha,
                         filename="cell_meta_xstest.json",
                         defense=defense, attack="xstest",
                         n_target_forwards=defense.n_target_forwards)
        return

    if args.attack == "bon":
        from sisteer.eval.attacks.bon import bon_generate

        run_dir = jailbreak_run._bon_defense_dir(cfg, root, cell.id)
        bcfg = cfg["bon"]
        print(f"[bon] cell={cell.id} shard {args.shard}/{args.nshards}: "
              f"{len(shard_goals)} goals x N={bcfg['n_max']} -> {run_dir}")
        bon_generate(defense, shard_goals, out_dir=run_dir, n_max=bcfg["n_max"],
                     sigma=bcfg["sigma"], trial=args.trial,
                     max_new_tokens=bcfg["max_new_tokens"],
                     batch_size=getattr(args, "bon_batch_size", None))
        _write_cell_meta(
            run_dir, cell, cfg, alpha,
            filename=f"cell_meta_bon_shard{args.shard:02d}.json",
            defense=defense, attack="bon", shard=args.shard,
            nshards=args.nshards, trial=args.trial,
            n_target_forwards=defense.n_target_forwards)
        print(f"[bon] done; n_target_forwards={defense.n_target_forwards}")
        return

    if args.attack == "gcg":
        # White-box, target-only; EXEMPT from forward_cap (optimizer forwards
        # are the third cost class) — identical contract to jailbreak_run.
        from sisteer.eval.attacks.gcg import run_gcg

        out_dir = jailbreak_run._gated_shard_dir(cfg, root, "gcg", cell.id,
                                                 args.shard)
        gcfg = {k: v
                for k, v in (jailbreak_run._attacks_cfg(cfg).get("gcg", {}) or {}).items()
                if k in jailbreak_run._GCG_RUN_KEYS}
        print(f"[gcg] cell={cell.id} shard {args.shard}/{args.nshards}: "
              f"{len(shard_goals)} goals -> {out_dir}")
        result = run_gcg(defense, shard_goals, out_dir=out_dir,
                         target_max_new_tokens=jailbreak_run._target_max_new_tokens(cfg),
                         **gcfg)
        _write_cell_meta(
            out_dir, cell, cfg, alpha, filename="cell_meta.json",
            defense=defense, attack="gcg", shard=args.shard,
            nshards=args.nshards,
            n_target_forwards=result["n_target_forwards_total"],
            n_optimizer_forwards=result["n_optimizer_forwards_total"])
        print(f"[gcg] done; goals_run={result['n_goals_run']}/"
              f"{result['n_goals_total']}; n_target_forwards="
              f"{result['n_target_forwards_total']} (REAL generations only); "
              f"optimizer_forwards~{result['n_optimizer_forwards_total']}")
        return

    from sisteer.eval.attacks import REPLAYABLE_FAMILIES

    if args.attack in REPLAYABLE_FAMILIES:
        # Static/replayable + sub-env families: NO attacker model, forward_cap-
        # EXEMPT (same accounting exemption as GCG). Written to the SAME sharded
        # gated layout keyed on cell.id, so the existing aggregate reads
        # them unchanged; sub-env families skip LOUDLY when their sub-env is
        # absent. One contract, two drivers — reuse jailbreak_run's dispatch.
        out_dir = jailbreak_run._gated_shard_dir(cfg, root, args.attack, cell.id,
                                                 args.shard)
        print(f"[{args.attack}] cell={cell.id} shard {args.shard}/{args.nshards}: "
              f"{len(shard_goals)} goals -> {out_dir}")
        result = jailbreak_run._run_replayable_family(
            args.attack, defense, shard_goals, cfg, out_dir, trial=args.trial)
        if result.get("skipped"):
            print(f"[{args.attack}] SKIPPED — {result.get('reason')}; nothing "
                  "generated (see stderr).")
            return
        cost = result.get("amplegcg_optimizer_cost")
        meta_extra = {} if cost is None else {"n_generator_forwards": cost}
        _write_cell_meta(
            out_dir, cell, cfg, alpha, filename="cell_meta.json",
            defense=defense, attack=args.attack, shard=args.shard,
            nshards=args.nshards,
            n_target_forwards=result["n_target_forwards_total"], **meta_extra)
        print(f"[{args.attack}] done; goals_run={result['n_goals_run']}/"
              f"{result['n_goals_total']}; n_target_forwards_total="
              f"{result['n_target_forwards_total']}"
              + (f"; generator_forwards~{cost}" if cost is not None else ""))
        return

    # Gated tiers (pair/tap/crescendo/actorattack): attacker seams + dispatch
    # reused verbatim from jailbreak_run — one contract, two drivers. The
    # seams were built once in cmd_generate and are shared across the grid.
    out_dir = jailbreak_run._gated_shard_dir(cfg, root, args.attack, cell.id,
                                             args.shard)
    print(f"[{args.attack}] cell={cell.id} shard {args.shard}/{args.nshards}: "
          f"{len(shard_goals)} goals -> {out_dir}")
    result = jailbreak_run._run_gated_attack(
        args.attack, defense, shard_goals,
        attacker_factory=seams["attacker_factory"],
        attacker_fn=seams["attacker_fn"], scorer_fn=seams["scorer_fn"],
        out_dir=out_dir,
        forward_cap=jailbreak_run._forward_cap(cfg),
        target_max_new_tokens=jailbreak_run._target_max_new_tokens(cfg),
        # GREEDY, STATED: this cell-keyed lane's filed stores are greedy, and it must not drift
        # when jailbreak_run's CLI default changed to "sample at the served model's own generation
        # config" (2026-09-11). It was greedy before only because the kwarg was not threaded —
        # an omission a future mechanical edit could flip; now it is a decision.
        target_temperature=0.0)
    _write_cell_meta(
        out_dir, cell, cfg, alpha, filename="cell_meta.json",
        defense=defense, attack=args.attack, shard=args.shard,
        nshards=args.nshards,
        n_target_forwards=result["n_target_forwards_total"])
    print(f"[{args.attack}] done; goals_run={result['n_goals_run']}/"
          f"{result['n_goals_total']}; "
          f"n_target_forwards_total={result['n_target_forwards_total']}")


# --------------------------------------------------------------------------- #
# aggregate — completeness first, then grade, keyed on cell.id                 #
# --------------------------------------------------------------------------- #


def _load_pool_regime(cfg: dict) -> dict:
    """The pool-regime block, read from the FROZEN counts-only manifest
    the §3a pool construction writes (the authoritative record — never
    re-derived here). Fails loudly when the manifest is missing or malformed:
    the aggregate must state which training pool produced the Class A
    objects, in BOTH regimes, and refuses to guess.

    EXPLICIT-cell configs (``explicit_cells`` — sdf-base-steering §12) get
    their own regime block instead: those runs mix training provenances by
    construction (instruct KL fits next to document-corpus CE fits), so a
    single pool manifest cannot describe them and is not required — the
    per-cell ``train_budget_meta.json`` sidecar (validated at registry build
    and by the generate gate) is the authoritative record, and the block
    says exactly that rather than inventing pool sizes."""
    from sisteer.eval.defense_cell import EXPLICIT_CELLS_KEY

    if EXPLICIT_CELLS_KEY in cfg:
        return {
            "regime": _EXPLICIT_REGIME,
            "pool_sizes": {},
            "headline_confounded_by_pool_composition": None,
            "note": _EXPLICIT_POOL_NOTE,
        }
    path = Path(cfg["pool_manifest"])
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise SystemExit(
            f"[two-si] pool manifest missing at {path} — D26 requires the "
            "aggregate to state the training-pool regime (matched_intersection "
            "or native_per_si) with both pool sizes, in BOTH cases. The pool "
            "construction (plan §3a step 6) writes this counts-only manifest; "
            "run it rather than bypassing this check.")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"[two-si] pool manifest {path} is not valid JSON: {e}")
    regime = manifest.get("regime")
    if regime not in _POOL_REGIMES:
        raise SystemExit(
            f"[two-si] pool manifest {path}: regime must be one of "
            f"{_POOL_REGIMES}, got {regime!r}")
    si_ids = list(cfg["si_ids"])
    sizes = manifest.get("pool_sizes")
    if not isinstance(sizes, dict) or sorted(sizes) != sorted(si_ids):
        raise SystemExit(
            f"[two-si] pool manifest {path}: pool_sizes must map exactly the "
            f"config si_ids {si_ids}, got "
            f"{sorted(sizes) if isinstance(sizes, dict) else sizes!r}")
    sizes = {si: int(n) for si, n in sizes.items()}
    if regime == "matched_intersection" and len(set(sizes.values())) != 1:
        raise SystemExit(
            f"[two-si] pool manifest {path}: matched_intersection pools must "
            f"have IDENTICAL sizes (D8 step 5 — identical ordered prompt ids), "
            f"got {sizes}")
    return {
        "regime": regime,
        "pool_sizes": sizes,
        # Machine-readable fallback flag: native pools confound the
        # headline SI contrast with pool composition.
        "headline_confounded_by_pool_composition": regime == "native_per_si",
        "note": _POOL_NOTE[regime],
    }


def _controls_fill(cfg: dict, registry) -> tuple[dict, list[str]]:
    """``({cell_id: {control: payload}}, string values to allowlist)`` from
    PROVENANCE-VALIDATED controls summaries (written by cell_controls_run.py
    judge and the served driver's completion merge; any stale component
    raises). Absent dir -> empty, aggregate unchanged. Fill order at the
    slot site: the slot's own value wins -> validated summary -> the
    pending sentinel; a filled slot is never overwritten. The loader
    applies the ``gsm8k`` slot's acceptance rule: a legacy bespoke payload
    comes back under ``_LEGACY_CONTROL_KEYS``, not the slot."""
    from sisteer.eval.cell_controls import load_controls_summaries, payload_strings

    droot = Path(cfg.get("controls_dir") or Path(cfg["out_dir"]) / "controls")
    if not droot.is_absolute():
        droot = REPO / droot
    filled = load_controls_summaries(droot, registry, cfg)
    return filled, payload_strings(filled)


def _training_pool_for(cell, pool_regime: dict):
    """The per-cell training-pool statement (a reader looking at ANY
    single cell must see which pool regime trained it, never infer it). In
    the explicit-cell mode every object cell points at the same per-cell
    truth: its own budget sidecar (no shared pool sizes exist to quote)."""
    if pool_regime["regime"] == _EXPLICIT_REGIME:
        return None if cell.arm is None else {"regime": _EXPLICIT_REGIME}
    if cell.class_ == "A":
        return {"regime": pool_regime["regime"],
                "size": pool_regime["pool_sizes"][cell.si_id]}
    if cell.class_ == "B":
        return {"regime": _POOL_SI_FREE}
    return None  # baselines train nothing


def _si_id_strings(cfg: dict, registry) -> list[str]:
    """The SI-id strings for the aggregate's sanitizer allowlist: in
    explicit-cell mode — which has no ``si_ids`` by mutual exclusion —
    every si component the enumerated cells actually carry; otherwise the
    config's ``si_ids``, byte-identical to the historical read (including
    the loud KeyError a Cartesian config without the key always got)."""
    from sisteer.eval.defense_cell import EXPLICIT_CELLS_KEY

    if EXPLICIT_CELLS_KEY in cfg:
        return sorted({c.si_id for c in registry if c.si_id})
    return list(cfg["si_ids"])


def _refused_cells(cfg: dict, registry) -> dict[str, str]:
    """``{cell_id: reason}`` for cells whose object-store ``meta.json``
    records a gate refusal — a first-class, recorded outcome, read
    from the object sidecar the trainer/gate writes next to
    ``cell.object_path``. An unreadable sidecar fails loudly (it is the
    authoritative refusal record, never guessed around)."""
    refused: dict[str, str] = {}
    for cell in registry:
        if cell.object_path is None:
            continue  # baselines have no object, hence no gate
        meta_path = Path(cell.object_path).parent / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"[two-si] unreadable object meta for cell {cell.id!r} at "
                f"{meta_path}: {e} — the refusal record is authoritative; fix "
                "it rather than aggregating around it.")
        reason = meta.get("refused")
        if reason:
            refused[cell.id] = str(reason)
    return refused


def _present_cell_ids(cfg: dict, root, attack: str, registry, refused) -> list[str]:
    """Registry cells with transcripts on disk for ``attack`` (registry
    order). PRESENCE only — per-goal completeness is still enforced by the
    downstream readers (A2/B3), which fail loudly on partial work."""
    present: list[str] = []
    for cell in registry:
        if cell.id in refused:
            continue
        if attack == "bon":
            has = (Path(root) / cfg["private_run"] / cell.id
                   / "bon_responses.jsonl").exists()
        else:
            droot = jailbreak_run._gated_defense_root(cfg, root, attack, cell.id)
            has = any(droot.glob("shard*/responses.jsonl"))
        if has:
            present.append(cell.id)
    return present


def _aggregate_bon_cells(cfg: dict, root, by_goal, grade_fn, trial, per_cell,
                         leaked_needles, *, expected_goal_ids,
                         group_of=None) -> None:
    """The BoN per-cell read: jailbreak_run's ``_aggregate_bon`` with the
    grade function INJECTED (that one hardwires the Batch judges; the curve /
    summary machinery — ``load_bon_responses`` / ``bon_curves`` /
    ``_summarize_bon`` — is reused, so the numbers agree by construction).
    Iterates ``cfg["defenses"]`` (= the present cell ids) and FAILS LOUD on a
    fully empty cell (A2/B3)."""
    from sisteer.eval.attacks.bon import bon_curves, load_bon_responses
    from sisteer.eval.jailbreak_artifacts import grade_items

    expected_goals = [by_goal[gid] for gid in expected_goal_ids if gid in by_goal]
    for cell_id in cfg["defenses"]:
        jsonl = Path(root) / cfg["private_run"] / cell_id / "bon_responses.jsonl"
        if not jsonl.exists():
            raise SystemExit(
                f"[bon] {cell_id}: no responses at {jsonl} — a fully empty "
                "cell is an error, not SKIP (A2/B3/D10). Refusing to publish.")
        responses = load_bon_responses(jsonl, trial=trial)
        items = grade_items(expected_goal_ids, by_goal, responses)
        scores = grade_fn(items)

        def score_fn(goal, response, _s=scores):
            return _s[(goal.dataset, goal.revision, goal.text, response)]

        for g in expected_goals:
            leaked_needles.append(g.text)
            leaked_needles.extend(responses.get(g.goal_id, []))
        curves = bon_curves(
            expected_goals, responses, score_fn, n_max=cfg["bon"]["n_max"],
            threshold=cfg["prereg"]["threshold_grid"][1],  # 0.5 — the endpoint
        )
        per_cell[cell_id] = jailbreak_run._summarize_bon(curves)
        if group_of is not None:
            per_cell[cell_id]["by_category_group"] = {
                grp: jailbreak_run._summarize_bon(
                    [c for c in curves if group_of.get(c.goal_id) == grp])
                for grp in ("held_in", "held_out")
            }


def _build_graders(args, cfg: dict, root, out_root):
    """``(grade_fn, xstest_grade_fn)`` for the chosen transport. live (the
    default) = the Low-Prio ``LiveStrongREJECTGrader``; batch = the Message
    Batches judges. Both share the JB1 contract: every item scored or a loud
    raise — never a silent 0."""
    if args.transport == "live":
        from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader

        grader = LiveStrongREJECTGrader(
            model=cfg["grader"]["model"], concurrency=args.concurrency,
            cache_path=Path(root) / cfg["private_run"]
            / "live_two_si_grade_cache.jsonl")
        return grader.grade, grader.grade_xstest

    from judge_batch import grade_strongreject, grade_xstest
    from sisteer.eval.attacks.grading import BatchStrongREJECTGrader

    haiku, sonnet = jailbreak_run._grade_judges(cfg, out_root)
    if args.attack == "bon":
        def grade_fn(items):
            return grade_strongreject(items, judge=haiku, sonnet_judge=sonnet,
                                      poll_interval=args.poll_interval)
    else:
        grade_fn = BatchStrongREJECTGrader(
            judge=haiku, sonnet_judge=sonnet,
            poll_interval=args.poll_interval).grade

    def xstest_grade_fn(items):
        return grade_xstest(items, judge=haiku, sonnet_judge=sonnet,
                            poll_interval=args.poll_interval)

    return grade_fn, xstest_grade_fn


def cmd_aggregate(args, cfg, *, grade_fn=None, xstest_grade_fn=None) -> None:
    """The grading host aggregate. ``grade_fn`` / ``xstest_grade_fn`` are test
    seams (offline, CPU-only tests inject fakes); when absent the real
    transport is built from ``--transport``."""
    from sisteer.eval.attacks import SUBENV_ATTACKS
    from sisteer.eval.defense_cell import (
        assert_matrix_complete,
        format_alpha,
        matrix_counts,
        over_refusal_comparisons,
    )
    from sisteer.eval.jailbreak_artifacts import (
        allowed_strings,
        grep_gate,
        write_sanitized_json,
    )

    registry = _registry(cfg)
    # ONE served dose per aggregate: the output is a per-dose artifact, and a
    # multi-dose grade would smear doses into one file. Grade each dose with
    # its own invocation.
    alphas = _served_alphas(args, cfg)
    if len(alphas) != 1:
        raise SystemExit(
            "[two-si] aggregate grades ONE served dose per invocation — got "
            f"--alpha {', '.join(format_alpha(a) for a in alphas)}; run one "
            "aggregate per dose.")
    alpha = alphas[0]
    fit_scale = float(cfg.get("scale", 1.0))
    # Everything below reads the dose-keyed tree; the pre-registered dose
    # (== config scale) is today's tree, byte-identically.
    cfg = _alpha_cfg(cfg, alpha)
    # Pool regime FIRST: no grading spend before the regime is known — the
    # aggregate cannot be published without it, so fail before the bill.
    pool_regime = _load_pool_regime(cfg)
    root = jailbreak_run._private_root(cfg, args.private_root)
    out_root = REPO / cfg["out_dir"]

    # Completeness gate, BEFORE grading: every cell is either present
    # (transcripts on disk) or refused-with-reason — anything else raises with
    # the dropped cell ids named.
    refused = _refused_cells(cfg, registry)
    present = _present_cell_ids(cfg, root, args.attack, registry, refused)
    assert_matrix_complete(registry, present, refused=refused)
    # Dose-consistency gate, also BEFORE grading: every transcript about to
    # be graded must record the served_alpha this aggregate claims.
    _assert_served_alpha(cfg, root, args.attack, present,
                         alpha=alpha, fit_scale=fit_scale)

    goals = _eval_goals(cfg, confirmatory=args.confirmatory,
                        attack=args.attack)
    by_goal = {g.goal_id: g for g in goals}
    expected_goal_ids = [g.goal_id for g in goals]
    group_of, held_in, held_out, ch = jailbreak_run._category_groups(cfg, goals)

    if grade_fn is None or xstest_grade_fn is None:
        built_grade, built_xs = _build_graders(args, cfg, root, out_root)
        grade_fn = grade_fn or built_grade
        xstest_grade_fn = xstest_grade_fn or built_xs

    # Reuse the per-defense readers keyed on cell.id: they iterate
    # cfg["defenses"], so handing them the present cell ids makes every
    # results key a cell.id at the aggregation site.
    cfg_cells = {**cfg, "defenses": present}
    per_cell: dict[str, dict] = {}
    leaked_needles: list[str] = []
    if args.attack == "bon":
        _aggregate_bon_cells(cfg_cells, root, by_goal, grade_fn, args.trial,
                             per_cell, leaked_needles,
                             expected_goal_ids=expected_goal_ids,
                             group_of=group_of)
        primary = cfg["prereg"]["primary_statistic"]
        metric_keys = [*jailbreak_run._BON_METRIC_KEYS,
                       *[str(n) for n in cfg["bon"]["schedule"]]]
    else:
        jailbreak_run._aggregate_gated(
            args.attack, cfg_cells, root, by_goal, grade_fn, per_cell,
            leaked_needles, expected_goal_ids=expected_goal_ids,
            group_of=group_of)
        primary = jailbreak_run._GATED_PRIMARY_STATISTIC
        metric_keys = [*jailbreak_run._GATED_METRIC_KEYS,
                       *[str(t) for t in cfg["prereg"]["threshold_grid"]]]
        if args.attack == "gcg":
            jailbreak_run._augment_gcg_optimizer_cost(cfg_cells, root, per_cell)
            metric_keys = [*metric_keys, *jailbreak_run._GCG_METRIC_KEYS]
        elif args.attack in SUBENV_ATTACKS:
            jailbreak_run._augment_subenv_optimizer_cost(
                args.attack, cfg_cells, root, per_cell)
            metric_keys = [*metric_keys, *jailbreak_run._SUBENV_METRIC_KEYS]

    # The xstest control slot (the one with a runner in this lane) — the
    # same fail-loud 250-safe-prompt read as the single-SI lane, per cell.
    xs_goal_ids = jailbreak_run._aggregate_xstest(
        cfg_cells, root, None, None, args.poll_interval, per_cell,
        leaked_needles, xstest_grade_fn=xstest_grade_fn)

    # -- assemble: identity + controls + pool statement, keyed on cell.id ----
    controls = list(cfg["controls"])
    # capeval-controls seam: provenance-validated per-cell summaries (r1 #4).
    summary_fill, fill_strings = _controls_fill(cfg, registry)
    cells_block: dict[str, dict] = {}
    for cell in registry:
        if cell.id in refused:
            continue
        metrics = dict(per_cell.get(cell.id, {}))
        xstest_result = metrics.pop("xstest", None)
        fill = summary_fill.get(cell.id, {})
        entry = {
            "class": cell.class_,
            "si_id": cell.si_id,
            "arm": cell.arm,
            # The NI reference(s) the stats layer compares against —
            # Class A: its matching prompted arm; Class B: BOTH.
            "ni_reference_ids": list(cell.ni_reference_ids),
            # The config's control slots, fill order: this lane's own
            # machinery's value wins (xstest today), else a provenance-valid
            # controls summary, else explicitly pending — never absent, and
            # a filled slot is never overwritten.
            "controls": {
                c: (xstest_result if c == "xstest" and xstest_result is not None
                    else fill.get(c, _CONTROL_PENDING))
                for c in controls
            },
            "training_pool": _training_pool_for(cell, pool_regime),
            # The dose next to the score, PER CELL (a lifted entry must
            # carry its own dose): the serving coefficient these transcripts
            # were generated under, verified against the sidecars above.
            "served_alpha": alpha,
            "attack_metrics": metrics,
        }
        # A number the loader retained as explicitly LEGACY (PW5-1: the
        # retired gsm8k instrument's payload of a pre-PW3 summary) rides
        # beside the control slots, never inside them — so the slot says
        # pending and the old number is still on the page, labelled.
        legacy = {k: fill[k] for k in _LEGACY_CONTROL_KEYS if k in fill}
        if legacy:
            entry["legacy_controls"] = legacy
        cells_block[cell.id] = entry

    aggregate = {
        "attack": args.attack,
        # BOTH coefficients, always: the pre-registered fit-time scale and
        # the dose this aggregate's transcripts were SERVED at. Different
        # numbers whenever the dose grid is in play — see dose_note.
        "fit_scale": fit_scale,
        "served_alpha": alpha,
        # top level, present in BOTH regimes, with both pool sizes.
        "pool_regime": pool_regime,
        "cells": cells_block,
        "refused": refused,
        "controls": controls,
        "over_refusal_comparisons": [
            {"cell_id": c.cell_id, "reference_id": c.reference_id,
             "control": c.control}
            for c in over_refusal_comparisons(registry, controls)
        ],
        "matrix_counts": matrix_counts(registry),
        "primary_statistic": primary,
        "category_holdout": {"seed": ch["seed"], "frac": ch["frac"],
                             "held_in": held_in, "held_out": held_out},
    }
    if alpha != fit_scale:
        # The bound travels WITH the numbers: an off-fit-dose aggregate says
        # what its scores do and do not support (see _ALPHA_NOTE).
        aggregate["dose_note"] = _ALPHA_NOTE
    allowed = allowed_strings(
        defenses=[c.id for c in registry],
        categories=held_in + held_out,
        goal_ids=expected_goal_ids + xs_goal_ids,
        extra=[
            "attack", args.attack, "cells", "refused", "controls",
            "over_refusal_comparisons", "matrix_counts", "pool_regime",
            "primary_statistic", primary,
            "class", "si_id", "arm", "ni_reference_ids", "attack_metrics",
            "training_pool", "regime", "pool_sizes", "size", "note",
            "fit_scale", "served_alpha", "dose_note", _ALPHA_NOTE,
            "headline_confounded_by_pool_composition",
            "cell_id", "reference_id", "control",
            "A", "B", "baseline",
            *_si_id_strings(cfg, registry),
            *(c.arm for c in registry if c.arm),
            *controls, _CONTROL_PENDING,
            "legacy_controls", *_LEGACY_CONTROL_KEYS,
            *_POOL_REGIMES, _POOL_SI_FREE,
            pool_regime["regime"], pool_regime["note"],
            # Gate-refusal reasons are OUR OWN gate text (from the object
            # meta.json this lane writes) — allowlisted deliberately, and the
            # grep gate below still scans them against the harmful needles.
            *refused.values(),
            # Controls-summary payloads: fixed key enum + the validated
            # summaries' own string values (hashes/labels) — same deliberate
            # allowlisting as the refusal reasons; grep gate still applies.
            *_CONTROL_PAYLOAD_KEYS, *fill_strings,
            *metric_keys, *jailbreak_run._XSTEST_KEYS,
            *jailbreak_run._CATEGORY_HOLDOUT_KEYS,
            "held_in", "held_out",
            "class_A", "class_B", "baselines", "objects", "total",
        ])
    # Per-dose output file: the pre-registered dose keeps today's filename
    # (no migration); any other dose is named by its alpha so dose aggregates
    # never overwrite each other.
    dose_tag = "" if alpha == fit_scale else f"_alpha{format_alpha(alpha)}"
    out_override = getattr(args, "out", None)
    out_path = (Path(out_override) if out_override
                else out_root / f"two_si_{args.attack}{dose_tag}_aggregate.json")
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_sanitized_json(aggregate, out_path, allowed=allowed)

    hits = grep_gate([out_root], leaked_needles)
    if hits:
        raise SystemExit(f"[{args.attack}] GREP GATE FAILED — harmful text in "
                         f"served files: {hits}. Refusing to publish.")
    # state the regime PROMINENTLY, in both cases — not only in the JSON.
    # (Explicit-cell runs have no shared pool sizes; the note carries the
    # per-cell-sidecar statement instead.)
    sizes = ", ".join(f"{si}={n}" for si, n in pool_regime["pool_sizes"].items())
    size_part = f" ({sizes})" if sizes else ""
    print(f"[two-si] POOL REGIME: {pool_regime['regime']}{size_part} — "
          f"{pool_regime['note']}")
    # The dose, PROMINENTLY, in both cases — a defense score from this lane
    # is never reported without the alpha it was served at.
    if alpha == fit_scale:
        print(f"[two-si] SERVED ALPHA: {format_alpha(alpha)} — the "
              "pre-registered fit-time scale (the default dose).")
    else:
        print(f"[two-si] SERVED ALPHA: {format_alpha(alpha)} != fit scale "
              f"{format_alpha(fit_scale)} — OFF-FIT DOSE: same object bytes, "
              "eval-time serving coefficient only. Supports serving-dose "
              "claims for THIS object; says nothing about a refit at this "
              "dose (unmeasured).")
    print(f"[{args.attack}] wrote {out_path} (aggregate, sanitized, keyed on "
          f"cell.id); grep gate clean.")


# --------------------------------------------------------------------------- #
# cells — the sbatch array decode surface                                      #
# --------------------------------------------------------------------------- #


def cmd_cells(args, cfg) -> None:
    registry = _registry(cfg)
    if args.task_id is not None:
        from sisteer.eval.defense_cell import cell_for_array_index

        # Raises (never wraps) on a stale --array bound — the sbatch guard
        # treats empty output as fatal.
        cell, shard = cell_for_array_index(registry, args.task_id, args.nshards)
        print(f"{cell.id} {shard}")
        return
    from sisteer.eval.defense_cell import matrix_counts

    for i, c in enumerate(registry):
        print(f"{i:3d}  {c.id}  (class {c.class_})")
    counts = matrix_counts(registry)
    print(f"# {counts['total']} cells = {counts['baselines']} baselines + "
          f"{counts['class_A']} Class A + {counts['class_B']} Class B "
          f"(derived from the registry, never a literal); "
          f"--array=0-({counts['total']}*NSHARDS - 1)")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def main() -> None:
    from sisteer.eval.attacks import JB_ATTACKS

    attacks = sorted(JB_ATTACKS)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    alpha_help = (
        "eval-time serving coefficient (dose) — repeatable and/or "
        "comma-separated (e.g. --alpha 0.25 --alpha 1, or --alpha 0.25,1); "
        "--coeff is an alias. Default: the config's pre-registered fit-time "
        "`scale`, which keeps every path and behaviour identical to an "
        "invocation without the flag. Serves the SAME fitted object at each "
        "dose (application-time coefficient only — NOTHING is refit, and "
        "configs/two_si.yaml `scale` is untouched); a non-default dose "
        "writes to its own <private_run>_alpha<A> namespace and records "
        "served_alpha next to fit_scale in every cell_meta sidecar. A good "
        "number at alpha<1 means the deployed dose is too strong for THIS "
        "object — it does NOT mean alpha is the right fit target (a refit "
        "at that dose is unmeasured). To run the §4.1 engine-vs-lane "
        "equivalence gate at a served dose, export SISTEER_EQUIV_SCALE=<A> "
        "(the gate otherwise compares at the config scale).")

    g = sub.add_parser("generate")
    g.add_argument("--config", default="configs/two_si.yaml")
    g.add_argument("--attack", default="bon", choices=attacks)
    g.add_argument("--cell", required=True,
                   help="DefenseCell id: none, prompted__<si>, <si>__<arm> "
                        "(Class A) or a bare Class-B <arm> — the ONLY "
                        "identity surface (D10; `cells` lists them)")
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--nshards", type=int, default=1)
    g.add_argument("--trial", type=int, default=0)
    g.add_argument("--bon-batch-size", type=int, default=None,
                   help="fixed GPU width for batched BoN generation (bon.py "
                        "batches the augmentations at this width; None = the "
                        "legacy width-1 per-variant-seed path). ~7x at 64; "
                        "width>1 is NOT reproducible against width-1 and is "
                        "recorded per row (bon_batch_size) as a comparability "
                        "axis — regenerate consistently, never mix widths.")
    g.add_argument("--confirmatory", action="store_true",
                   help="the 100-goal confirmatory JBB subset; default is "
                        "the 50-goal pilot")
    g.add_argument("--private-root", default=None)
    g.add_argument("--xstest", action="store_true",
                   help="run the 250-safe-prompt over-refusal pass for this "
                        "cell instead of the attack sweep")
    g.add_argument("--alpha", "--coeff", dest="alpha", action="append",
                   default=None, metavar="A[,A...]", help=alpha_help)

    a = sub.add_parser("aggregate")
    a.add_argument("--config", default="configs/two_si.yaml")
    a.add_argument("--attack", default="bon", choices=attacks)
    a.add_argument("--trial", type=int, default=0)
    a.add_argument("--confirmatory", action="store_true")
    a.add_argument("--private-root", default=None)
    a.add_argument("--transport", choices=["live", "batch"], default="live",
                   help="StrongREJECT grade transport: live Low-Prio "
                        "concurrent requests (default) or the Batch API")
    a.add_argument("--concurrency", type=int, default=200,
                   help="peak concurrent live requests (default 200)")
    a.add_argument("--poll-interval", type=int, default=120)
    a.add_argument("--out", default=None,
                   help="output json path override (default "
                        "<out_dir>/two_si_<attack>_aggregate.json; a "
                        "non-default --alpha inserts _alpha<A> before "
                        "_aggregate)")
    a.add_argument("--alpha", "--coeff", dest="alpha", action="append",
                   default=None, metavar="A",
                   help="the ONE served dose this aggregate grades (default: "
                        "the config `scale`, i.e. today's tree). Reads the "
                        "<private_run>_alpha<A> transcript tree and refuses — "
                        "before any grading spend — unless every transcript's "
                        "cell_meta records served_alpha == A (the "
                        "dose-consistency gate). One dose per invocation.")

    c = sub.add_parser("cells")
    c.add_argument("--config", default="configs/two_si.yaml")
    c.add_argument("--task-id", type=int, default=None,
                   help="decode one Slurm array task id to '<cell.id> <shard>'")
    c.add_argument("--nshards", type=int, default=1)

    args = ap.parse_args()
    cfg = jailbreak_run._load_cfg(args.config)
    {"generate": cmd_generate, "aggregate": cmd_aggregate,
     "cells": cmd_cells}[args.cmd](args, cfg)


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed
    # error -- ends the process with the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[two-si] "):
        main()
