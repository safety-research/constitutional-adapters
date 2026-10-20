"""Multi-source ownership of ONE measurement (the paper's ``config.MULTI_SOURCE_CELL_OWNERSHIP`` in the registry's nouns).

Two sources can read the same measurement: the r2 export (source ``r2_export``) and the source's own reader of the same grade
roots (source ``arditi_allpos``; the Llama-3.1-8B reader ``llama8b_lane`` for that model's Petri stores). The registry keeps BOTH
provenances — two ``run`` rows, two ``run_metric`` rows — because the readers mint different campaigns, so no key ever collides
and none of the key-level rules (native first, declared priority, source id, revision) ever sees the two as one
key. Without a declaration ``v_paper_long`` carries the measurement TWICE, the silent failover the paper refuses everywhere.

A declaration names, per SCOPE (model × object label × eval category × transfer kind, at the served operating point or unfitted,
minus any eval ids it excludes by name):

* the OWNER source and the SECONDARY source (``run.lane_source_legacy`` spellings). The export owns the Qwen scope; the Llama
  reader owns the Llama Petri scopes (paperlib's rule of 2026-09-14: ``load_r2`` drops the export's Llama Petri rows at read; the
  export's ``n`` there counts scenarios PLANNED — ``petri_agg``'s ``n_items``, no transfer-stats join for the Llama settings — where
  the reader counts scenarios SCORED).
* the TRANSFER KIND in scope (``run.transfer_kind``): a declaration is per transfer kind, because one paper label can carry two
  fitted objects served under two kinds (the Llama constitution / control / difference LoRAs each exist as a fit on the base
  checkpoint served on the instruct one, ``base_to_instruct``, and as a fit on the instruct checkpoint, ``instruct_direct``); a
  scope that ignored the kind would see two owner runs at one identity and refuse every measurement.
* the STRENGTH RULE that admits a run to the scope: ``served_operating_point`` (the declared ``served_operating_point_id``, or a run
  filing no id at nominal 1.0 — the fitted objects) or ``unfitted`` (no id and no nominal — the ``none`` object and the prompted
  system instructions).
* the EVAL IDS EXCLUDED from the category (``exclude_eval_ids``), each with a reason in the ruling: the Llama Petri scopes leave
  ``petri_pooled`` out because the reader's pooled runs are ``not_in_roster`` (their instruments carry ``thinking_mode`` unverified
  and miss the roster's ``thinking_mode = off`` selector), so an owner row there would sit outside the paper frame while the fold
  removed the export's row — the pooled measurement would EMPTY. Outside the scope both rows stand as today (the export's selected, the
  reader's not in the roster, no double) until the roster admits the reader's pooled instrument.
* the PAIRING RULE that tells two runs of ONE source apart inside the scope: ``producer_run_key`` matches an owner row to a
  secondary row only when both runs carry the same ``run.producer_run_key_legacy`` (the two Llama readers spell it alike, so a row
  folds only onto the row read from the same store); ``identity_only`` pairs on the measurement identity alone (the Qwen pair's
  two readers spell the key differently — the export files the store's directory name, the reader a
  ``<source>/<ladder>/<store>/<model>`` path — and the scope holds one object with one run per eval, so the identity is already
  unique per side). The rendered pairing column is ``pairing_key``.
* the IDENTITY a measurement is matched on — ``(pairing_key, eval_id, metric_id, threshold)``; the threshold sentinel ``-1`` stands
  for "no threshold" on both sides, so the match is null-safe by construction.
* the counts that must agree EXACTLY — ``k`` and ``n``; a difference REFUSES the measurement (never averaged, never picked from).
* the readings the secondary CONTRIBUTES onto the owner's row, only where the owner has none — ``degenerate_share`` and
  ``loop_share`` (the secondary's degeneracy sidecar; the export carries none for these scopes); an owner already carrying a
  DIFFERENT value refuses.
* the readings that NEVER travel — ``at_cap_share``: the owner's is the export's per-measurement at-cap figure, the secondary's
  is the sidecar's re-tokenized reading, two statistics under one paper column.

BOTH sides count only their source's OWN computation rows (``run_metric.computation_id <> ''``). The stats readers of the same
stores (``petri_stats``, ``multiturn_grades``, ``bon_stores``) mint the export's run id and file a producer-filed blank-computation
twin of every export row onto it — the row ``v_paper_long``'s metric role ranks second ("a computed row before a blank one";
migration 0004) — and a twin counted on either side would read as two rows at one identity and refuse every measurement. Every
declared source files its rows under a computation (the export carries no blank-computation key anywhere: 8,231 computed, 0 blank
on the live registry, 2026-09-20; the Llama reader and the all-positions reader likewise), so the rule keeps exactly the two
readers' own rows; a blank row on a scoped run is never in the scope at all.

While the owner has no row for a measurement the secondary's row stands as a STOPGAP and says so.

The declarations are rendered into SQL by :mod:`schema_def` (``v_multi_source_rows`` — every metric row inside a declared scope
with its role — and ``v_multi_source_check`` — one row per matched identity with ``state`` ∈ {stopgap, folded, refused}, both
sides' k / n and the carried readings) and read by ``v_paper_long`` (a folded or refused secondary row leaves the frame; a refused
owner row keeps its identity with value / k / n / ci NULL and the reason), by :mod:`paper` (the carry, the status and the note)
and by the status page. The SQL is generic over exactly the match / check / carry sets and the strength / pairing rules above;
:meth:`Ownership.__post_init__` refuses a declaration that asks for others, so nothing is declared that the SQL would silently ignore.

Adapter priorities are a SEPARATE mechanism — they order provenances of ONE key — and cannot resolve this class; they are
still declared explicitly (the owner's adapter above the secondary's: ``PRIORITY_OWNING_READER`` > ``PRIORITY_PAPER_SOURCE`` >
``PRIORITY_SECONDARY_SOURCE``) so the ranking is stated rather than left to the source-id tiebreak, and :func:`check_priorities`
pins the order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import vocab as V

#: the identity columns the SQL matches a measurement on (fixed; a declaration naming others is refused). `pairing_key` is the
#: declaration's PAIRING rule rendered per row (the producer run key, or '' under `identity_only`).
MATCH_COLUMNS: Tuple[str, ...] = ("pairing_key", "eval_id", "metric_id", "threshold")
#: the counts the SQL requires to be EXACTLY equal (null-safe: two nulls agree, a null against a number does not)
CHECK_COLUMNS: Tuple[str, ...] = ("k", "n")
#: the paper columns the secondary may contribute onto the owner's row where the owner has none
CARRY_COLUMNS: Tuple[str, ...] = ("degenerate_share", "loop_share")
#: the paper columns that never travel between the two sources, whatever either carries
NEVER_CARRY_COLUMNS: Tuple[str, ...] = ("at_cap_share",)
#: the states `v_multi_source_check.state` takes
STATES: Tuple[str, ...] = ("stopgap", "folded", "refused")
#: how a run is admitted to a scope by its dose: at the declared served operating point (a run filing no id counts at nominal 1.0),
#: unfitted (no operating point, no nominal: the `none` object, a prompted system instruction), or `nominal_one` (migration 0026: no
#: operating-point id and nominal 1.0 -- a fitted object whose served point the roster states on the kind alone, with no legacy id any
#: run of either source files: the ReLU steers of the twin scopes)
STRENGTH_RULES: Tuple[str, ...] = ("served_operating_point", "unfitted", "nominal_one")
#: how two runs of one source are told apart inside a scope: by the producer run key both sources spell alike, not at all, or
#: (migration 0026) by `run.view_legacy` -- the tag both the export and the pooled reader file, when the two spell the producer run key
#: differently and the scope holds several tags' runs at one identity
PAIRING_RULES: Tuple[str, ...] = ("identity_only", "producer_run_key", "view_legacy")
#: how the two readings at a matched identity are checked (migration 0026): `counts` -- k and n EXACTLY equal, null-safe (today's rule);
#: `value_with_draws` -- the values agree to 1e-9, the secondary's n equals the owner's `extra.n_draws` and the secondary carries k (a
#: pooled owner files k NULL and n as goals; the per-draw secondary files k and n over the draws, so the exact-count check is the wrong
#: predicate for that pair, not a disagreement)
CHECK_RULES: Tuple[str, ...] = ("counts", "value_with_draws")


@dataclass(frozen=True)
class Ownership:
    declaration_id: str
    model_ids: Tuple[str, ...]                 # `run.model_id` spellings in scope
    paper_label: str                           # `object.paper_label` of the object in scope
    eval_category: str                         # `eval.eval_category` in scope
    transfer_kind: str                         # `run.transfer_kind` in scope (vocab.TRANSFER_KINDS): a declaration is per transfer kind
    owner_source: str                          # `run.lane_source_legacy` of the owner
    secondary_source: str                      # `run.lane_source_legacy` of the secondary
    served_operating_point_id: Optional[str]   # `run.operating_point_id_legacy` at the served point (strength_rule served_operating_point; a row filing no id is in scope at nominal 1.0); None under `unfitted`
    since: str
    ruling: str
    pairing_rule: str                          # PAIRING_RULES: how an owner run and a secondary run are paired inside the scope
    strength_rule: str = "served_operating_point"   # STRENGTH_RULES
    check_rule: str = "counts"                 # CHECK_RULES: how the two readings at a matched identity are compared
    token_cap_tokens: Optional[int] = None     # the ONE token cap the scope holds (both sources' `run.token_cap_tokens`); None = every cap
    owner_pass_labels: Tuple[str, ...] = ()    # the owner's rows must carry one of these `run.pass_label`s (its pooled pass); () = any
    exclude_eval_ids: Tuple[str, ...] = ()     # `run.eval_id`s of the category left OUT of the scope, each for a reason the ruling states
    match: Tuple[str, ...] = MATCH_COLUMNS
    check: Tuple[str, ...] = CHECK_COLUMNS
    carry: Tuple[str, ...] = CARRY_COLUMNS
    never_carry: Tuple[str, ...] = NEVER_CARRY_COLUMNS

    def __post_init__(self):
        if tuple(self.match) != MATCH_COLUMNS or tuple(self.check) != CHECK_COLUMNS:
            raise ValueError(f"{self.declaration_id}: the SQL matches on {MATCH_COLUMNS} and checks {CHECK_COLUMNS}; "
                             f"a declaration naming other columns is not rendered and is refused")
        if not set(self.carry) <= set(CARRY_COLUMNS) or set(self.carry) & set(self.never_carry):
            raise ValueError(f"{self.declaration_id}: carry {self.carry} must be a subset of {CARRY_COLUMNS} and disjoint from never_carry {self.never_carry}")
        if not self.model_ids or not self.owner_source or not self.secondary_source or self.owner_source == self.secondary_source:
            raise ValueError(f"{self.declaration_id}: a declaration names at least one model and two DIFFERENT sources")
        if self.transfer_kind not in V.TRANSFER_KINDS:
            raise ValueError(f"{self.declaration_id}: transfer_kind {self.transfer_kind!r} is not one of vocab.TRANSFER_KINDS {V.TRANSFER_KINDS}")
        if self.strength_rule not in STRENGTH_RULES:
            raise ValueError(f"{self.declaration_id}: strength_rule {self.strength_rule!r} is not one of {STRENGTH_RULES}; the SQL renders no other")
        if self.pairing_rule not in PAIRING_RULES:
            raise ValueError(f"{self.declaration_id}: pairing_rule {self.pairing_rule!r} is not one of {PAIRING_RULES}; the SQL renders no other")
        if self.check_rule not in CHECK_RULES:
            raise ValueError(f"{self.declaration_id}: check_rule {self.check_rule!r} is not one of {CHECK_RULES}; the SQL renders no other")
        if self.check_rule == "value_with_draws" and tuple(self.check) != CHECK_COLUMNS:
            raise ValueError(f"{self.declaration_id}: value_with_draws compares the value, the secondary's n against the owner's draws and the secondary's k; "
                             f"the declared check columns stay {CHECK_COLUMNS}")
        if any(not isinstance(e, str) or not e.strip() for e in self.exclude_eval_ids) or len(set(self.exclude_eval_ids)) != len(self.exclude_eval_ids):
            raise ValueError(f"{self.declaration_id}: exclude_eval_ids names distinct, non-blank eval ids (got {self.exclude_eval_ids!r})")
        op = (self.served_operating_point_id or "").strip()
        if self.strength_rule == "served_operating_point" and not op:
            raise ValueError(f"{self.declaration_id}: the served operating point id is the scope's dose anchor and must be named")
        if self.strength_rule in ("unfitted", "nominal_one") and op:
            raise ValueError(f"{self.declaration_id}: a scope under strength_rule {self.strength_rule} names no operating point (got {op!r})")
        if self.token_cap_tokens is not None and (isinstance(self.token_cap_tokens, bool) or not isinstance(self.token_cap_tokens, int) or self.token_cap_tokens <= 0):
            raise ValueError(f"{self.declaration_id}: token_cap_tokens is a positive int or None (got {self.token_cap_tokens!r})")
        if not set(self.owner_pass_labels) <= set(V.PASS_LABELS) or len(set(self.owner_pass_labels)) != len(self.owner_pass_labels):
            raise ValueError(f"{self.declaration_id}: owner_pass_labels names distinct vocab.PASS_LABELS (got {self.owner_pass_labels!r})")

    def strength_sql(self, run: str = "r") -> str:
        """The dose predicate of the scope over an aliased `run` row, per `strength_rule`."""
        if self.strength_rule == "unfitted":
            return f"({run}.operating_point_id_legacy IS NULL AND {run}.strength_nominal IS NULL)"
        if self.strength_rule == "nominal_one":
            return f"({run}.operating_point_id_legacy IS NULL AND ABS({run}.strength_nominal - 1.0) <= 1e-9)"
        return (f"({run}.operating_point_id_legacy = {_lit(self.served_operating_point_id)}"
                f" OR ({run}.operating_point_id_legacy IS NULL AND {run}.strength_nominal = 1.0))")

    def pairing_key_sql(self, run: str = "r") -> str:
        """The `pairing_key` expression over an aliased `run` row, per `pairing_rule`: the producer run key; the run's `view_legacy` AND its
        instrument (`<tag>|<instrument_id>`: a NULL tag pairs as '', so two unlabeled runs of one instrument pair as under `identity_only`;
        two instruments never group -- the export's 4,096-token static runs sit under instruments of their own beside the 8,192 ones the
        pooled reader shares); or '' (every row of the scope pairs on the measurement identity alone)."""
        if self.pairing_rule == "producer_run_key":
            return f"{run}.producer_run_key_legacy"
        if self.pairing_rule == "view_legacy":
            return f"(COALESCE({run}.view_legacy, '') || '|' || {run}.instrument_id)"
        return "''"

    def scope_sql(self, run: str = "r", obj: str = "o", ev: str = "e") -> str:
        """The scope predicate over aliased `run` / `object` / `eval` rows: model, object label, eval category (minus the excluded
        eval ids), transfer kind, the two sources, the strength rule (the declared operating point — a row filing no id counts at
        nominal 1.0 — unfitted, or nominal_one), and, when declared, the ONE token cap both sources' rows must carry and the pass labels
        the OWNER's rows must carry (the pooled reader's `pooled_t0_t1`: its per-trial `base_cap` / `ladder_rung` members at the same cap
        and instrument would otherwise stand beside the pooled row as a second owner row and refuse the identity)."""
        models = ", ".join(_lit(m) for m in self.model_ids)
        excluded = (f"\n    AND {run}.eval_id NOT IN ({', '.join(_lit(e) for e in self.exclude_eval_ids)})") if self.exclude_eval_ids else ""
        cap = f"\n    AND {run}.token_cap_tokens = {int(self.token_cap_tokens)}" if self.token_cap_tokens is not None else ""
        owner_pass = (f"\n    AND ({run}.lane_source_legacy = {_lit(self.secondary_source)} OR {run}.pass_label IN ({', '.join(_lit(p) for p in self.owner_pass_labels)}))"
                      if self.owner_pass_labels else "")
        return (f"{run}.model_id IN ({models}) AND {obj}.paper_label = {_lit(self.paper_label)} AND {ev}.eval_category = {_lit(self.eval_category)}{excluded}\n"
                f"    AND {run}.transfer_kind = {_lit(self.transfer_kind)}\n"
                f"    AND {run}.lane_source_legacy IN ({_lit(self.owner_source)}, {_lit(self.secondary_source)})\n"
                f"    AND {self.strength_sql(run)}{cap}{owner_pass}")


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


#: the Llama-3.1-8B Petri scopes: the fitted labels × the two transfer kinds under which each label's object was fit, the model's own
#: reader owning and the export secondary. The served operating point ids are the roster's spellings (`run.operating_point_id_legacy`
#: of that model's agentic runs, 12 each); the Petri runs of both sources file no id and enter the scope at nominal 1.0. `petri_pooled`
#: is out of scope until the roster admits the reader's pooled instrument (LLAMA_PETRI_EXCLUDED_EVALS).
LLAMA_PETRI_SERVED_OPERATING_POINT: Dict[str, str] = {"base_to_instruct": "base__constv2-mix-full__lora_r4_ce",
                                                      "instruct_direct": "itdoc__constv2-mix-full__lora_r4_ce"}
LLAMA_PETRI_LABELS: Tuple[str, ...] = ("constitution_lora", "control_lora", "difference_lora")
LLAMA_PETRI_EXCLUDED_EVALS: Tuple[str, ...] = ("petri_pooled",)
_LLAMA_PETRI_RULING = (
    "paperlib's rule since 2026-09-14: the model's own reader (source llama8b_lane) owns the Llama-3.1-8B Petri measurements and load_r2 drops "
    "the export's Llama Petri rows at read; the export stands as the secondary. Ruled 2026-09-21 01:30Z (verified from the files): the "
    "export's n for these measurements is the count of scenarios PLANNED (build_export.py has no transfer-stats join for the Llama settings and "
    "takes petri_agg's n_items) where the reader counts scenarios SCORED (Petri's judge refused one finished transcript, recorded in the store as "
    "terminal) — the export's own pooled n of 59 for the same object proves the planned count wrong (10 x 6 is not 59), and elsewhere the export "
    "uses the scored denominator. Verified read-only on the live registry 2026-09-20/21: paired by producer run key inside one transfer kind over "
    "the six Petri sub-evals, every one of the 36 identities (three labels x two transfer kinds x six evals) matches the reader's one computed row to "
    "exactly one export computed row, values equal to 1e-9 and n agreeing on 35 of 36 (k is unfiled on both sides: petri_score is a mean); the one "
    "disagreement, control_lora fit on the base checkpoint x petri_cyber_bio_chem_capabilities (value 7.333 on both sides, n 9 in the reader against "
    "10 in the export), stays REFUSED per identity, never swallowed by a tolerance — the paper takes neither value silently. petri_pooled is out of "
    "scope: the reader's pooled runs are not_in_roster (their instruments carry thinking_mode unverified and miss the roster's thinking_mode = off "
    "selector), so an owner row there would sit outside the paper frame while the fold removed the export's row and the pooled measurement would empty; "
    "outside the scope the pooled measurement stays as today (the export's row selected, the reader's not in the roster, no double) until the roster admits "
    "the reader's pooled instrument. The stats reader's blank-computation twin of every export row is not counted (both sides: their own computation "
    "rows only). The secondary contributes the sidecar's degenerate_share / loop_share only; at_cap_share never travels (two statistics under one column)")


def _llama_petri(label: str, kind: str) -> "Ownership":
    return Ownership(
        declaration_id=f"llama8b_it/{label}/petri/{kind}",
        model_ids=("llama8b_it",), paper_label=label, eval_category="petri", transfer_kind=kind,
        owner_source="llama8b_lane", secondary_source="r2_export",
        served_operating_point_id=LLAMA_PETRI_SERVED_OPERATING_POINT[kind],
        since="2026-09-21", ruling=_LLAMA_PETRI_RULING, pairing_rule="producer_run_key",
        exclude_eval_ids=LLAMA_PETRI_EXCLUDED_EVALS,
    )


#: The `r2_export` <-> `sampled_lane` TWINS (adapter re-anchoring, migration 0026): the export's 8,192-token
#: static rows of three models are its reading of the pooled `pooled_t0_t1` value of the SAME tag -- per (run identity, metric, threshold)
#: the export files asr@{0.25, 0.5, 0.75} with k and n = the pooled draws, the pooled reader asr@{0.5, 0.75} with k NULL, n counting goals
#: and the draws in `extra.n_draws`; so the 0.5 / 0.75 identities are duplicates (the pooled row is the paper row, the export's folds) and
#: asr@0.25 plus the export's degeneration shares are UNIQUE and stand as stopgaps -- never drop a uniquely filed measurement.
#: Paired by `view_legacy` (each export run twins SEVERAL pooled nones at its key -- the tags -- and exactly one shares its tag,
#: the one whose value equals its own; the two readers spell the producer run key differently), checked by `value_with_draws`. One
#: declaration per paper label x transfer kind: the three models' `none` (unfitted) and Qwen's ReLU constitution steer on
#: `instruct_direct` (`nominal_one`: no roster operating-point id exists for the ReLU steers -- the roster admits them on the kind, both
#: sources file no id at nominal 1.0; the 0.5-nominal dose runs stay out). The scope is the 8,192-TOKEN static readings alone
#: (`token_cap_tokens` 8192 on both sides: the export also files 4,096-token runs of the same tags under instruments of their own, with
#: `n` 259 -- a different measurement, which the pairing by tag + instrument keeps apart and the cap keeps out of the scope, so it stands as
#: a plain export row) and the owner's POOLED pass alone (`owner_pass_labels` pooled_t0_t1: the reader's base_cap / ladder_rung members
#: share the pooled row's cap and instrument and would otherwise read as a second owner row). Registry-side only until the paper side cuts
#: over: `paperlib.config.MULTI_SOURCE_CELL_OWNERSHIP` carries no mirror of these two.
_TWIN_RULING = ("the owner's ruling on the plan's section 5 (2026-10-03): fold the duplicate (identity, metric, threshold) pairs of the "
                "r2_export <-> sampled_lane twins at the 8192 static instrument onto the pooled reading of the same tag, keep every uniquely "
                "filed measurement (asr@0.25, the degeneration shares) as the export's stopgap; the pooled reader files k NULL and n as goals, so "
                "the readings are checked by value (1e-9) and by the export's n against the pooled extra.n_draws, never by exact counts; "
                "D661 (i) / D671: the unsteered baseline is counted once")
TWIN_OWNERSHIP: Tuple[Ownership, ...] = (
    Ownership(
        declaration_id="sampled_static_8192/none/static_attack/none",
        model_ids=("gemma4e4b_it", "qwen35_4b_it", "llama8b_it"), paper_label="none", eval_category="static_attack", transfer_kind="none",
        owner_source="sampled_lane", secondary_source="r2_export", served_operating_point_id=None, strength_rule="unfitted",
        pairing_rule="view_legacy", check_rule="value_with_draws", token_cap_tokens=8192, owner_pass_labels=("pooled_t0_t1",),
        since="2026-10-05", ruling=_TWIN_RULING,
    ),
    Ownership(
        declaration_id="sampled_static_8192/constitution_relu/static_attack/instruct_direct",
        model_ids=("qwen35_4b_it",), paper_label="constitution_relu", eval_category="static_attack", transfer_kind="instruct_direct",
        owner_source="sampled_lane", secondary_source="r2_export", served_operating_point_id=None, strength_rule="nominal_one",
        pairing_rule="view_legacy", check_rule="value_with_draws", token_cap_tokens=8192, owner_pass_labels=("pooled_t0_t1",),
        since="2026-10-05", ruling=_TWIN_RULING,
    ),
)

#: The declarations in force. The Qwen scope mirrors `paperlib.config.MULTI_SOURCE_CELL_OWNERSHIP` (paper-figures 982c26c8, 2026-09-18);
#: the Llama Petri scopes mirror paperlib's rule of 2026-09-14 (the reader owns) as restated on 2026-09-21 01:30Z;
#: the two twin scopes (`TWIN_OWNERSHIP`) are the registry's own since migration 0026.
MULTI_SOURCE_OWNERSHIP: Tuple[Ownership, ...] = (
    Ownership(
        declaration_id="qwen35_4b_it/arditi_allpos/multi_turn",
        model_ids=("qwen35_4b_it",), paper_label="arditi_allpos", eval_category="multi_turn", transfer_kind="instruct_direct",
        owner_source="r2_export", secondary_source="arditi_allpos",
        served_operating_point_id="ar_c0p3__arditi_allpos",
        since="2026-09-18",
        ruling=("ruled 2026-09-17 19:54 (paper-figures 982c26c8): the export owns the all-positions refusal steer's multi-turn "
                "measurements once its ingress (Scale PR #204, 015489fe) refreshes; the source's own rows were a stopgap read from the same "
                "grade roots and must agree on k / n or the measurement is refused; the source contributes the sidecar's degenerate_share / "
                "loop_share only; at_cap_share never travels (two statistics under one column). Paired on the measurement identity alone: "
                "the two readers spell the producer run key differently (the export files the store's directory name, the reader a "
                "<source>/<ladder>/<store>/<model> path) and the scope holds one object with one run per eval"),
        pairing_rule="identity_only",
    ),
) + tuple(_llama_petri(label, kind) for label in LLAMA_PETRI_LABELS for kind in ("base_to_instruct", "instruct_direct")) + TWIN_OWNERSHIP

BY_ID: Dict[str, Ownership] = {d.declaration_id: d for d in MULTI_SOURCE_OWNERSHIP}
if len(BY_ID) != len(MULTI_SOURCE_OWNERSHIP):
    raise ValueError("duplicate multi-source declaration ids")


def declarations() -> Tuple[Ownership, ...]:
    return MULTI_SOURCE_OWNERSHIP


def sources_declared() -> Dict[str, Tuple[str, str]]:
    """``declaration_id -> (owner_source, secondary_source)``."""
    return {d.declaration_id: (d.owner_source, d.secondary_source) for d in MULTI_SOURCE_OWNERSHIP}


def adapters_by_source(registry: Mapping[str, object]) -> Dict[str, List[object]]:
    """``lane source string -> [adapter instance, ...]`` over a loaded adapter registry (``adapters.load_all()``)."""
    out: Dict[str, List[object]] = {}
    for name in sorted(registry):
        inst = registry[name]()
        for attr in ("covers_sources", "declared_sources"):
            for s in (getattr(inst, attr, ()) or ()):
                lst = out.setdefault(str(s), [])
                if not any(getattr(a, "name", None) == getattr(inst, "name", name) for a in lst):
                    lst.append(inst)
    return out


def check_priorities(registry: Mapping[str, object]) -> List[Dict[str, object]]:
    """Pure over a loaded registry: for every declaration, the owner's adapter must declare a HIGHER priority than the
    secondary's (the spelling of "the owner outranks"), and both must declare one (an undeclared 0 would leave the ranking
    to the source-id tiebreak). Returns one row per declaration; raises ``ValueError`` naming the first
    declaration whose adapters break the order."""
    by_source = adapters_by_source(registry)
    from .adapters.runner import adapter_priority
    rows: List[Dict[str, object]] = []
    for d in MULTI_SOURCE_OWNERSHIP:
        owners = by_source.get(d.owner_source, [])
        seconds = by_source.get(d.secondary_source, [])
        if not owners or not seconds:
            raise ValueError(f"{d.declaration_id}: no adapter covers {d.owner_source if not owners else d.secondary_source!r}")
        op = {getattr(a, "name", "?"): adapter_priority(a) for a in owners}
        sp = {getattr(a, "name", "?"): adapter_priority(a) for a in seconds}
        row = {"declaration_id": d.declaration_id, "owner_source": d.owner_source, "owner_priorities": op,
               "secondary_source": d.secondary_source, "secondary_priorities": sp}
        rows.append(row)
        if min(op.values()) <= max(sp.values()):
            raise ValueError(f"{d.declaration_id}: the owner's adapter(s) {op} do not outrank the secondary's {sp}; declare the priorities")
        if min(op.values()) <= 0 or min(sp.values()) <= 0:
            raise ValueError(f"{d.declaration_id}: an adapter in the declaration declares no priority ({op}, {sp}); the source-id tiebreak would decide")
    return rows
