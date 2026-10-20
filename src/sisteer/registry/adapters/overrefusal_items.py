"""The paper's per-item over-refusal source ``overrefusal_items``.

Roots (``paperlib.load.OR_ROOTS`` at the parity pin): ``$GRADE/sdfbase_full22/controls/<target>/``,
``$STAGE/sdfbase/fullscale/consolidated/judge/<target>/`` and ``run-state/full22/controls_halfdose/<target>/``,
each holding ``cells/<cell_key>/`` served-controls stores (read by :mod:`cellstore`) plus the producer's own paired
verdict ``judge_paired_vs_none.json``.

Two computations ride beside the store rows:

* the RELOCATED reader (``paperlib.load.load_overrefusal_graded`` @ the pin, quoted verbatim in
  :func:`relocated_graded`): per (target dir, cell, control) the mean graded score over judged items (XSTest
  restricted to its 250 safe prompts; at-cap = the intermediate grade), the at-cap share, and the paired delta
  against the same target dir's unsteered cell over shared item ids — percentile bootstrap (5,000) + Wilcoxon, drawn
  from ONE ``np.random.default_rng(0)`` in dict-iteration order over a ``glob`` traversal of the three roots with
  the newest ``judge_items.jsonl`` winning a (target, cell) that appears under several roots. Equality at 1e-9 with
  the paper needs the same code over the same inputs in the same order, so ``rng_scope = "reader_traversal"``: a
  change to ANY input re-runs the whole traversal. Filed under ``computation_id =
  identity.computation_id("paperlib.load.load_overrefusal_graded", PAPER_PIN, "shared_rng seed=0 n_boot=5000 iteration-order")``.
* the PRODUCER's paired verdict (``judge_paired_vs_none.json``: McNemar exact p, Wald 95 % on the refusal-rate
  difference, discordant counts) as ``run_contrast`` rows under its own computation id.

Item rows, flags, judgments and per-store metrics come from the shared :class:`cellstore.StoreReader`.

Run identity: every run of a store is keyed ``<target dir>/<store key>`` (``StoreReader(run_key_scope="target_dir")``,
``run.producer_run_key_legacy`` carries the same string), never the store key alone. The unsteered store of one model is served
again under its ``_halfdose``, ``_it`` and ``_it_halfdose`` target directories under the SAME store key
(``none__controls__t0__a0__2ff9cdff__<harness8>``: no object, so the intervention hash is constant, and the harness hash does not
name the served model or dose), so the bare key minted ONE run per eval for two to four stores and the fold's last writer mixed
their fields — ``n_items_judged`` from one store, ``store_uri`` from another, ``k_graded_sum`` rewritten from three record dirs
(b2, 2026-09-19). With the target directory in the key each store's control is its own run, and the steered runs' contrasts
(:meth:`OverrefusalItemsAdapter._none_store`: the ``none__controls__`` store under the SAME ``cells/``) anchor on the control
of their own store. The other readers of served stores key the same way — ``relu_paired`` by the store directory name,
``strength_ladder`` by ``declared_common.store_run_key`` (the path above the store key), ``sampled_controls`` by the tag.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import identity
from . import _items_common as IC
from . import cellstore as CS
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_glob, site_root

ADAPTER_NAME = "overrefusal_items"
OR_ROOTS: Tuple[str, ...] = (
    site_root("$GRADE/sdfbase_full22/controls"),
    site_root("$STAGE/sdfbase/fullscale/consolidated/judge"),
    os.path.join(IC.RUN_STATE, "controls_halfdose"),
)
#: ``OR_ROOTS[1]`` -- the consolidated judge root as the watermarks of the earlier reads spelled it -- RETIRED. The producer moved the
#: tree WHOLE to ``$STAGE/sdfbase/fullscale/_retired/judge/`` on 2026-10-07T08:53:14Z, bytes
#: unchanged, nothing deleted; ``README_retired.md`` there is the provenance. The one input this root ever gave discovery,
#: ``nemotron3nano4b/judge_paired_vs_none.json`` (sha256 cbc93e0273da3434…, 2026-09-07), names a ``none__controls__…__5603bdf0`` store under
#: a ``cells/`` directory that never existed there (its six sibling directories hold only ``judge_summary.json``, the 09-07 aggregate), so the
#: reader refused the group identically five times since 09-19; the real stores and the NEWER paired verdict (09-08, 8 stores) live under
#: ``OR_ROOTS[0]``'s ``nemotron3nano4b/`` and are in the registry; ``glm46vflash/`` was empty. Paper impact none:
#: ``paperlib.load.load_overrefusal_graded`` reads only ``<root>/*/cells/*/judge_items.jsonl`` (73 files before and after the move).
#: ``OR_ROOTS`` keeps the spelling for paperlib parity (discovery skips a missing root); nothing is re-pointed, no symlink, nothing is read
#: from ``_retired/``. Declared through :meth:`OverrefusalItemsAdapter.inputs_retired` so the runner's inputs-dropped comparison counts the
#: vanished input `inputs_retired`, never `inputs_dropped` -- on a CLUSTER shard ANY undeclared dropped input refuses the group
#: (``runner.ClusterInputDropRefusal``, ``staged_copy_incomplete``), which is where this group sat in every window since the move.
RETIRED_JUDGE_ROOT = OR_ROOTS[1]
#: Stores under :data:`OR_ROOTS` the producer moved OUT of their ``cells/`` directory because no numbers were ever filed -- the OLD path, the
#: one the watermarks recorded -- RETIRED. ``gemma4e4b_halfdose``'s ``constv2-mix-full--lora_r4_ce_minus_ctl`` store held only
#: ``manifest.json`` (sha256 c67fb014f0a610d4…) and ``evidence.json`` (f1fab2b74a5d52b2…), both of 2026-09-08: Slurm job 463796 died on
#: ``FidelityError: logprob_steering_bites`` (the steer could not be certified to reach the scored positions), and
#: ``gemma4e4b_halfdose_controls_summary.json`` records the store ``complete: false``; the paired verdict omits it. The producer
#: moved it to ``gemma4e4b_halfdose/_incomplete/<same name>/`` on 2026-10-07T08:53:14Z (``README_incomplete.md`` there is the
#: provenance), bytes unchanged. No accepted revision ever existed for the group (the reader refused it; nothing accepted is withdrawn), and
#: ``_incomplete/`` is enumerated by nobody (discovery lists ``cells/`` alone); nothing is re-pointed. Declared through
#: :meth:`OverrefusalItemsAdapter.inputs_retired` for the same reason as :data:`RETIRED_JUDGE_ROOT`.
INCOMPLETE_STORES: Tuple[str, ...] = (
    os.path.join(IC.RUN_STATE, "controls_halfdose", "gemma4e4b_halfdose", "cells",
                 "constv2-mix-full--lora_r4_ce_minus_ctl__controls__t0__a0__8ebbf872__e5868612"),
)
OR_CONTROLS = {"xstest": "xstest", "or_bench_hard_1k": "or_bench_hard_1k"}
N_BOOT = 5000
SEED = 0
RELOCATED_ALGORITHM = "paperlib.load.load_overrefusal_graded"
RELOCATED_SEED_PROVENANCE = f"shared_rng seed={SEED} n_boot={N_BOOT} iteration-order"
PRODUCER_PAIRED_ALGORITHM = "served_capability_judge.judge_paired_vs_none"


def relocated_computation_id() -> str:
    return identity.computation_id(RELOCATED_ALGORITHM, IC.PAPER_PIN, RELOCATED_SEED_PROVENANCE)


def producer_paired_computation_id() -> str:
    return identity.computation_id(PRODUCER_PAIRED_ALGORITHM, "producer", "exact (no rng)")


def reader_files(roots: Sequence[str]) -> Dict[Tuple[str, str], str]:
    """``files`` exactly as the pinned reader builds it: glob order per root, newest judge_items.jsonl wins a (target, cell)."""
    files: Dict[Tuple[str, str], str] = {}
    for root in roots:
        for f in store_glob(f"{root}/*/cells/*/judge_items.jsonl"):
            view_legacy = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
            store = os.path.basename(os.path.dirname(f))
            cid = store.split("__controls__")[0].replace("--", "__")
            k = (view_legacy, cid)
            if k not in files or os.path.getmtime(f) > os.path.getmtime(files[k]):
                files[k] = f
    return files


def relocated_graded(files: Mapping[Tuple[str, str], str], n_boot: int = N_BOOT, seed: int = SEED) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """paperlib.load.load_overrefusal_graded @ the pin, minus the frame assembly: {(target, cell, control): stats}.

    ``value`` = mean graded score, ``k`` = its sum, ``n`` items, ``at_cap_share``; ``delta`` (pp), ``delta_lo`` /
    ``delta_hi`` (percentile bootstrap over shared items), ``delta_p`` (Wilcoxon), ``n_pairs`` when the target dir has
    an unsteered cell. The RNG is drawn in the reader's dict-iteration order; nothing here may be reordered.
    """
    import json
    import numpy as np
    per: Dict[Tuple[str, str, str], Dict[str, Tuple[float, bool]]] = {}
    for (view, cid), f in files.items():
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                ctl = r.get("control")
                if ctl not in OR_CONTROLS or (ctl == "xstest" and r.get("subset") != "safe"):
                    continue
                sc = IC.or_item_score(r)
                if sc is None:
                    continue
                per.setdefault((view, cid, ctl), {})[r["item_id"]] = (sc, bool(r.get("at_cap")))
    rng = np.random.default_rng(seed)
    out: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for (view, cid, ctl), items in per.items():
        vals = np.array([v[0] for v in items.values()]); caps = np.array([v[1] for v in items.values()])
        rec: Dict[str, object] = dict(value=float(vals.mean()), k=float(vals.sum()), n=len(vals), at_cap_share=float(caps.mean()),
                                      provenance=files[(view, cid)])
        base = per.get((view, "none", ctl))
        if cid != "none" and base:
            shared = [i for i in items if i in base]
            if shared:
                d = np.array([items[i][0] - base[i][0] for i in shared])
                idx = rng.integers(0, len(d), size=(n_boot, len(d))); bs = d[idx].mean(axis=1)
                rec.update(delta=100 * float(d.mean()), delta_lo=100 * float(np.percentile(bs, 2.5)), delta_hi=100 * float(np.percentile(bs, 97.5)),
                           delta_p=IC.wilcoxon_p(d), n_pairs=len(shared),
                           delta_method=f"paired on {len(shared)} benign items (graded score): percentile bootstrap {n_boot}; Wilcoxon signed-rank p")
        out[(view, cid, ctl)] = rec
    return out


class OverrefusalItemsAdapter:
    name = ADAPTER_NAME
    covers_sources: Tuple[str, ...] = ("overrefusal_items",)
    #: 5 = the shared reader derives a fitted object's `transfer_kind` from the manifest's `fit_provenance.model_id` OR the ingested object
    #: row's `fit_model_id`, and refuses a store with neither (or an unresolvable / disagreeing fit model) by name instead of filing the
    #: `unverified` sentinel (2026-09-30; a default change of :class:`cellstore.StoreReader`, so every constructor bumps -- every store of
    #: this source carries `fit_provenance.model_id`, so its bytes do not move);
    #: 4 = a fitted object's runs file the dose the target directory spells (`_halfdose` 0.5, else 1.0) with `is_operating_point` =
    #: (nominal == 1.0) through the shared :class:`cellstore.StoreReader` (2026-09-20; every fitted run here had carried NULL there);
    #: 3 = runs keyed `<target dir>/<store key>` (`StoreReader(run_key_scope="target_dir")`): the unsteered store's key recurs under
    #: a model's base / `_halfdose` / `_it` / `_it_halfdose` target dirs and minted one run whose fields the fold mixed (b2, 2026-09-19);
    #: 2 = `run.lane_source_legacy` filed on the benign controls' runs (`StoreReader(source_by_control_legacy)`)
    #: NOT bumped for the 2026-10-07 retirement declarations (:data:`RETIRED_JUDGE_ROOT`, :data:`INCOMPLETE_STORES`): the read of
    #: every remaining group is unchanged byte for byte -- the declaration only tells the inputs-dropped comparison that the two vanished
    #: groups' inputs left on purpose -- and a bump would re-read every group of this reader_traversal adapter for nothing.
    EMISSION_VERSION = 7        # 27b: the shared StoreReader's sent/resolved decode split + the recorder as a per-store revision input
    rng_scope = "reader_traversal"
    allow_symlinks = False
    priority = IC.PRIORITY_PAPER_SOURCE        # The paper's over-refusal source outranks the generic reader of its mirrors

    def __init__(self, roots: Sequence[str] = OR_ROOTS, dims: Optional[IC.Dims] = None, campaign: str = "full22"):
        self.roots = tuple(roots)
        self.dims = dims
        self.campaign = campaign
        self._relocated: Optional[Tuple[frozenset, Dict[Tuple[str, str, str], Dict[str, object]]]] = None
        self._store_cache: Dict[str, CS.StoreRecords] = {}

    # -- discovery: store groups + one view-level group per target dir ------------------------------
    def discover(self) -> List[Input]:
        inputs: List[Input] = []
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for view in IC.listdir(root):
                vdir = os.path.join(root, view)
                if not os.path.isdir(vdir):
                    continue
                for name, role in CS.TARGET_DIR_FILES:
                    p = os.path.join(vdir, name)
                    if os.path.lexists(p):
                        inputs.append(check_input(p, role=role, allow_symlink=self.allow_symlinks))
                cells = os.path.join(vdir, "cells")
                for d in IC.listdir(cells):
                    store = os.path.join(cells, d)
                    if os.path.isdir(store):
                        inputs.extend(CS.store_inputs(store, self.allow_symlinks))
        return inputs

    def comparison_group(self, inp: Input) -> str:
        return CS.group_of_input(inp)       # the shared recorder's per-store alias groups with ITS store (27b)

    def inputs_retired(self, group: str) -> List[str]:
        """The inputs a read of ``group`` drops ON PURPOSE (:func:`base.inputs_retired_of`): everything under
        :data:`RETIRED_JUDGE_ROOT` for a group under that root, the files of a store in :data:`INCOMPLETE_STORES` for that store's own
        group, nothing for any other group -- a RENAME is a retirement and is declared, never inferred (runner.INPUTS_DROPPED_RULE). Both
        moves are the producer's of 2026-10-07: the files stand byte-identical at paths no root of this adapter reaches, so the read
        drops nothing a reader could miss, and the sibling stores of the same ``cells/`` are not covered (an absent file of theirs is still
        a drop). Asked with the EMPTY group -- `stores coverage` reading the declaration with no comparison group (``stores._retired_globs``)
        -- it answers with the whole declaration. The runner canonicalizes these patterns through the pass's identity map and matches each
        absent input against them (``fnmatch``: ``*`` crosses ``/``)."""
        judge = [os.path.join(RETIRED_JUDGE_ROOT, "*")]
        if group == "":
            return judge + [os.path.join(store, "*") for store in INCOMPLETE_STORES]
        if group == RETIRED_JUDGE_ROOT or group.startswith(RETIRED_JUDGE_ROOT.rstrip("/") + "/"):
            return judge
        if group in INCOMPLETE_STORES:
            return [os.path.join(group, "*")]
        return []

    def _dims(self, ctx: AdapterContext) -> IC.Dims:
        if self.dims is None:
            self.dims = IC.Dims(conn=ctx.conn)
            self.dims.models.update(IC.load_legacy_model_map(ctx.legacy_spellings))
        elif self.dims.conn is None and ctx.conn is not None:
            self.dims.conn = ctx.conn
        return self.dims

    def _is_store(self, group: str) -> bool:
        return os.path.basename(os.path.dirname(group)) == "cells"

    def _campaign_for(self, group: str) -> str:
        return self.campaign

    def relocated(self) -> Dict[Tuple[str, str, str], Dict[str, object]]:
        """The whole-traversal computation, cached on the set of judge_items shas it read."""
        files = reader_files(self.roots)
        key = frozenset((k, os.path.getmtime(f), os.path.getsize(f)) for k, f in files.items())
        if self._relocated is None or self._relocated[0] != key:
            self._relocated = (key, relocated_graded(files))
        return self._relocated[1]

    def _store_records(self, ctx: AdapterContext, store: str, inputs: Optional[Sequence[Input]] = None) -> CS.StoreRecords:
        if store in self._store_cache:
            return self._store_cache[store]
        ins = list(inputs) if inputs is not None else CS.store_inputs(store, self.allow_symlinks)
        # the two benign controls the pinned reader files ride under its source (`run.lane_source_legacy = overrefusal_items`); a
        # store's other controls (gsm8k) are read into the registry with no paper source — no paper reader files them from here
        # `run_key_scope="target_dir"`: the store key alone does not identify a run here (the module docstring's identity paragraph)
        reader = CS.StoreReader(self._dims(ctx), self._campaign_for(store), ctx.now,
                                source_by_control_legacy={ctl: ADAPTER_NAME for ctl in OR_CONTROLS}, run_key_scope="target_dir")
        sr = reader.read(store, ins, view=CS._store_view_legacy(store, "targets"))
        self._store_cache[store] = sr
        return sr

    # -- reading ---------------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        batch = RecordBatch(f"{self.name}:{group}", group)
        if self._is_store(group):
            sr = self._store_records(ctx, group, inputs)
            CS.emit(batch, sr, ctx.producer, ctx.now)
            self._emit_relocated(ctx, batch, group, sr)
        else:
            self._emit_producer_paired(ctx, batch, group, inputs)
        yield batch

    def _emit_relocated(self, ctx: AdapterContext, batch: RecordBatch, store: str, sr: CS.StoreRecords) -> None:
        view = CS._store_view_legacy(store, "targets")
        stats = self.relocated()
        cid = sr.key.cell_id
        comp = relocated_computation_id()
        batch.add("computation", {"computation_id": comp, "algorithm": RELOCATED_ALGORITHM, "version": IC.PAPER_PIN,
                                  "seed_provenance": RELOCATED_SEED_PROVENANCE, "code_sha": IC.PAPER_PIN,
                                  "notes": "relocated from paperlib/load.py at the parity pin; verbatim statistics"}, ctx.producer, ctx.now)
        for ctl, cr in sr.controls.items():
            if ctl not in OR_CONTROLS:
                continue
            st = stats.get((view, cid, ctl))
            if not st:
                continue
            # the reader's newest-wins rule: a store that lost the (target, cell) to a newer root carries no relocated row
            if os.path.dirname(str(st["provenance"])) != store:
                continue
            jid = cr.judgment_id
            batch.add("run_metric", {"run_id": cr.run_id, "judgment_id": jid, "metric_id": "over_refusal_graded", "threshold": -1.0,
                                     "computation_id": comp, "producer_metric_spelling": "over_refusal_graded", "value": st["value"],
                                     "k": None, "n": int(st["n"]),
                                     "frac_at_token_cap": st["at_cap_share"], "computed_by": "producer", "computed_utc": ctx.now,
                                     "source_uri": str(st["provenance"]),
                                     "extra": {"k_graded_sum": st["k"], "reader": RELOCATED_ALGORITHM, "pin": IC.PAPER_PIN,
                                               # the reader's at-cap share is the judge driver's RE-TOKENIZED flag (judge_items.at_cap)
                                               "at_cap_estimand": IC.AT_CAP_RETOKENIZED}}, ctx.producer, ctx.now)
            if "delta" in st:
                none_store = self._none_store(store)
                anchor = self._store_records(ctx, none_store) if none_store else None
                if anchor is None or ctl not in anchor.controls:
                    continue
                acr = anchor.controls[ctl]
                kind = "same_standup_none" if anchor.key.harness8 == sr.key.harness8 else "sibling_none"
                batch.add("run_contrast", {"run_id": cr.run_id, "anchor_run_id": acr.run_id, "judgment_id": jid, "metric_id": "over_refusal_graded",
                                           "threshold": -1.0, "computation_id": comp, "anchor_judgment_id": acr.judgment_id, "anchor_kind": kind,
                                           "delta": st["delta"], "delta_lo": st["delta_lo"], "delta_hi": st["delta_hi"],
                                           "delta_p": IC.nan_to_none(st["delta_p"]), "delta_method": "paired_bootstrap", "n_pairs": int(st["n_pairs"]),
                                           "computed_by": "producer", "computed_utc": ctx.now,
                                           "extra": {"unit": "pp", "method_text": st["delta_method"], "reader": RELOCATED_ALGORITHM}}, ctx.producer, ctx.now)

    def _none_store(self, store: str) -> Optional[str]:
        cells = os.path.dirname(store)
        for d in IC.listdir(cells):
            if d.startswith("none__controls__") and os.path.isdir(os.path.join(cells, d)):
                return os.path.join(cells, d)
        return None

    def _emit_producer_paired(self, ctx: AdapterContext, batch: RecordBatch, view_dir: str, inputs: Sequence[Input]) -> None:
        paired = next((i for i in inputs if os.path.basename(i.uri) == "judge_paired_vs_none.json"), None)
        if paired is None:
            return
        d = IC.read_json(paired.path)
        if not isinstance(d, Mapping) or not isinstance(d.get("cells"), Mapping):
            raise AdapterRefusal(paired.uri, "judge_paired_vs_none.json has no cells block")
        none_key = d.get("none_cell")
        cells = os.path.join(view_dir, "cells")
        if not isinstance(none_key, str) or not os.path.isdir(os.path.join(cells, none_key)):
            raise AdapterRefusal(paired.uri, f"none_cell {none_key!r} names no store under {cells}")
        anchor = self._store_records(ctx, os.path.join(cells, none_key))
        comp = producer_paired_computation_id()
        batch.add("computation", {"computation_id": comp, "algorithm": PRODUCER_PAIRED_ALGORITHM, "version": "producer", "seed_provenance": "exact (no rng)",
                                  "code_sha": "unverified", "notes": "the producer's paired verdict file beside the target dir (McNemar exact, Wald 95 %)",
                                  "field_confidence_json": {"code_sha": {"confidence": "unverified", "reason": "judge_paired_vs_none.json records no code sha"}}},
                  ctx.producer, ctx.now)
        for store_key, blk in d["cells"].items():
            if store_key == none_key or not isinstance(blk, Mapping) or not os.path.isdir(os.path.join(cells, store_key)):
                continue
            sr = self._store_records(ctx, os.path.join(cells, store_key))
            for ctl, cr in sr.controls.items():
                pb = blk.get(ctl)
                if not isinstance(pb, Mapping) or ctl not in anchor.controls:
                    continue
                acr = anchor.controls[ctl]
                wald = pb.get("delta_wald95") if isinstance(pb.get("delta_wald95"), list) and len(pb["delta_wald95"]) == 2 else [None, None]
                batch.add("run_contrast", {"run_id": cr.run_id, "anchor_run_id": acr.run_id, "judgment_id": cr.judgment_id, "metric_id": "orb_refused" if ctl != "xstest" else "xst_refused",
                                           "threshold": -1.0, "computation_id": comp, "anchor_judgment_id": acr.judgment_id,
                                           "anchor_kind": "same_standup_none" if anchor.key.harness8 == sr.key.harness8 else "sibling_none",
                                           "delta": IC.nan_to_none(pb.get("delta")), "delta_lo": IC.nan_to_none(wald[0]), "delta_hi": IC.nan_to_none(wald[1]),
                                           "delta_p": IC.nan_to_none(pb.get("mcnemar_exact_p")), "delta_method": "exact_mcnemar",
                                           "n_pairs": int(pb.get("n_judged_both") or 0), "computed_by": "producer", "computed_utc": ctx.now,
                                           "extra": {k: v for k, v in pb.items() if not isinstance(v, (dict, list))} | {"source_uri": paired.uri,
                                                     "terminated_in_both": pb.get("terminated_in_both")}}, ctx.producer, ctx.now)

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        out: List[SourceStateRow] = []
        for i in inputs:
            if os.path.basename(i.uri) == "judge_paired_vs_none.json":
                try:
                    d = IC.read_json(i.path)
                    members = sorted(d.get("cells", {}).keys()) if isinstance(d, Mapping) and isinstance(d.get("cells"), Mapping) else []
                except AdapterRefusal:
                    members = []
                out.append(SourceStateRow(f"{self.name}:{os.path.dirname(i.uri)}", i.uri, "$.cells", 1, 1, members, i.sha256))
        return out


def register() -> None:
    from .base import register as _register
    _register(OverrefusalItemsAdapter.name, lambda: OverrefusalItemsAdapter())
