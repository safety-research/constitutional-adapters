"""`bon_stores` adapter: the best-of-N escalation statistics and the vs-unsteered deltas.

Inputs on the grading host: `bon_cap4096/analysis/bon_cap4096_stats.json` (per producer group x store: the 512 store's at-cap / loop /
coherent-at-cap counts over its 7,500 draws, the 4,096 climb's residuals and the counterfactual best-of-75 at 0.5 / 0.75 paired
against the store's own 512 reading), `bon_siconst/all_bon_deltas.json` (each campaign store's counterfactual best-of-75 paired against
its view's UNSTEERED store at its final rung: delta, bootstrap interval, exact McNemar), the producer's registrations
`bon_cap4096/cluster/manifests/lanes.json` (per group, in the producer's own spelling: the model `key`, the point `view`, the store `roots`,
the `cluster` that climbed) and, for every store whose draws are on this host, its `bon_served_manifest.json` / `bon_served_run.json`
(the sampler the draws were requested under -- `decode_policy_id`, the decode numbers, `bon_decode_preset` -- and the serving identity).
Every snapshot beside the statistics (`.pre-*`) is an artifact.

Two INSTRUMENT families, one per decode-policy MODE, chosen per store from its RECORDED sampler and never defaulted: the grandfathered
T = 1.0 stores (`bon_512_only_T1` / `bon_512store_4096climb_T1`) and the stores drawn under a served model's vendor preset
(`bon_512_only_vendor_preset` / `bon_512store_4096climb_vendor_preset`, ReLU's paired-write points since 2026-09-19). A store's
served manifest on this host decides (its `decode_policy_id`'s mode, cross-checked against `bon_decode_preset` and the instrument's decode
numbers); with no manifest on this host, a campaign view's stores are the grandfathered ones (every store drawn before 2026-09-17,
decode_policy_seed.yaml) and any other group is REFUSED by name.

Two IDENTITIES, one rule: EVERY reading this reader files lands on a run THIS revision emits. A campaign view's two runs per store (the
512 store, superseded by the 512-floor ruling; the 512-store-plus-4,096-climb reading, the canonical best-of-N) are minted with the r2
export's row key so the rows attach to the export's own runs. A registered point of another source (relu_paired's `paired_d<D>_s<N>`)
has its 512 run in the registry only where relu_paired graded the store and filed it under ITS instrument; this reader files NO second
512 run row for the same draws (version 2's contract -- and a 512-only vendor-preset run on Qwen would be pulled into the paper by the
2026-09-17 include ruling over the 512-floor supersede), and since version 4 it files NO reading on relu_paired's run either: version 2
attached the 512 counts to that run id, rebuilt from the same inputs through :func:`relu_paired.bon_instrument_id`, and the harvester
holds a revision until every parent exists -- relu_paired files no best-of-N run for a point whose stores it never graded (the gemma /
nemotron paired writes, drawn under this producer's own root), so the 2026-09-22 read carrying those points never activated, its
paper-facing climb rows included. The registered point's climb run (the vendor-preset climb instrument, the registration's `key` for the
model, its `view` as ``view_legacy``, its `cluster` as the substrate, the point's dose as the nominal strength at the operating point) is
the paper-facing row and this reader's own; the 512 store's counts ride its `extra.store_512_counts`, relu_paired's run id its
`extra.relu_paired_512_run_id` as a cross-reference, and the rows not filed are counted and named in the read's facts
(`degeneration_rows_dropped_unemitted_runs`, `metric_rows_dropped_unemitted_runs`, `readings_dropped_unemitted_runs`). Before a batch is
yielded the reader checks that every degeneration, metric, judgment, state and contrast record names a run (and a judgment) of the batch
and REFUSES the revision by name otherwise. A campaign store's counts become `run_degeneration` rows (`at_token_cap_finish_reason`,
`rep_ratio_loop` over `draws`) and `run_metric` rows (the export's metric spellings kept) on its 512 run; the store-vs-its-own-512 delta
rides the climb metric's `extra_json` (not a contrast against an anchor); the best-of-N CURVE rides every climb run as seven `asr_at_n<N>`
rows (N in 1, 2, 4, 8, 16, 32, 64 at 0.5 -- seedrep_cluster's and bon_native4096's spelling, `paper.METRIC_LEGACY`'s `asr` at `bon_n` N)
read off the grid `asr_at_n_4096_cf_t0.5`: the producer's p, k = p x n where an integer, n the headline's goals, no interval, the grid's 75
the headline row and not filed twice (version 5, 2026-09-23); the vs-unsteered delta is a `run_contrast` (anchor_kind
same_standup_none) -- the deltas file's for a campaign view (a key the statistics file carries no store for is refused by name, never a
contrast on a run nobody emits), and for a registered point the difference of the two counterfactual best-of-75 ASRs read off the
statistics file (no per-goal pairing there: no interval, no McNemar, said so on the row). The per-draw stores of the campaign views live on
the clusters (`exp/sdfbase_full22/bon_cap4096`) and are declared in :attr:`BonStoresAdapter.cluster_roots` for the mirror pass (§2.8);
nothing here reads a draw.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .. import identity as I
from .. import vocab as V
from . import declared_common as DC
from . import lineage_common as LC
from . import relu_paired as RP
from . import seedrep_cluster as SC
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register, store_glob
from .runner import source_id_for

NAME = "bon_stores"
STATS = "bon_cap4096/analysis/bon_cap4096_stats.json"
DELTAS = "bon_siconst/all_bon_deltas.json"
#: the producer's registrations of the groups the statistics file keys on (the first segment of a statistics key, in its own spelling)
REGISTRATIONS = "bon_cap4096/cluster/manifests/lanes.json"
#: a store's own record of the sampler its draws were requested under, and of the standup that served them
STORE_MANIFEST = "bon_served_manifest.json"
STORE_RUN = "bon_served_run.json"
INST_512 = "bon_512_only_T1"
INST_CLIMB = "bon_512store_4096climb_T1"
INST_512_VENDOR = "bon_512_only_vendor_preset"
INST_CLIMB_VENDOR = "bon_512store_4096climb_vendor_preset"
GRANDFATHERED_MODE = "grandfathered_T1"
#: decode-policy MODE (the last segment of a `<model>|bon|<thinking>|<mode>` ref) -> (the 512-only seed, the 512-store-plus-4,096-climb seed)
INSTRUMENTS_BY_DECODE_MODE: Dict[str, Tuple[str, str]] = {GRANDFATHERED_MODE: (INST_512, INST_CLIMB), "vendor_preset": (INST_512_VENDOR, INST_CLIMB_VENDOR)}
#: the 512-only seed -> the climb seed that supersedes it (the 512-floor ruling's target)
CLIMB_OF: Dict[str, str] = {floor: climb for floor, climb in INSTRUMENTS_BY_DECODE_MODE.values()}
#: the producer's `bon_decode_preset` spellings -> decode-policy mode (bon_served.resolve_bon_sampler; decode_policy_seed.yaml routes)
PRESET_MODE: Dict[str, str] = {"vendor_preset": "vendor_preset", "harness_default": GRANDFATHERED_MODE}
#: a store manifest's decode keys -> the instrument row's decode columns (the numbers a filed reading must have been drawn under)
DECODE_COLS: Dict[str, str] = {"temperature": "decode_temperature", "top_p": "decode_top_p", "top_k": "decode_top_k", "min_p": "decode_min_p",
                               "repetition_penalty": "decode_repetition_penalty", "presence_penalty": "decode_presence_penalty"}
JUDGE_SEED = "strongreject_v1_haiku45"
#: the day the `<model>|bon|off|vendor_preset` rows became the best-of-N default (ruling decode-bon-vendor-preset-2026-09-17, decode_policy_seed.yaml):
#: every store drawn before it was drawn under `<model>|bon|off|grandfathered_T1`, and store manifests written before it record no sampler at all
VENDOR_PRESET_BON_EFFECTIVE_DATE = "2026-09-17"
#: the campaign views whose climbs ran on cluster A (bon_cap4096 README); every other campaign view climbed on cluster B
CLUSTER_A_VIEWS_LEGACY = ("glm46vflash", "gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b")
#: the best-of-N CURVE (the appendix figure of defense success within N attempts): one
#: `asr_at_n<N>` row per N at the headline threshold on EVERY climb run this reader files, in seedrep_cluster's spelling (bon_native4096
#: spells the same; `paper.METRIC_LEGACY` folds exactly these seven ids to the paper's `asr` at `bon_n` N, read off the row's
#: `extra.bon_n`), the value the statistics file's per-N grid `asr_at_n_4096_cf_t0.5` (the producer's p per N, string keys; the
#: counterfactual best-of-N within the first N draws). The grid's N beyond the seven -- 75, the headline -- is the `asr_bon_cap4096` row
#: already and is not filed twice
BON_CURVE_METRIC = SC.BON_CURVE_METRIC
BON_CURVE_THRESHOLD = SC.HEADLINE_THRESHOLD
BON_CURVE_NS = (1, 2, 4, 8, 16, 32, 64)
CURVE_GRID_KEY = "asr_at_n_4096_cf_t{t:g}"


@dataclass
class Standup:
    """One producer group of the statistics file: a CAMPAIGN view (a legacy model spelling the seeds resolve; its stores are the r2 export's)
    or a REGISTERED point of another source (relu_paired's paired writes: the registration's `key` names the model, its `view` the point, its
    `roots` the stores on this host, its `cluster` the substrate)."""
    producer_key: str                  # the statistics file's first key segment, the producer's own spelling
    view_legacy: str                   # the run row's view_legacy: the campaign spelling, or the registration's point spelling
    model_view_legacy: str             # the spelling the seeds resolve to a model: the campaign spelling, or the registration's `key`
    is_campaign: bool                  # True -> the r2 export's own runs (producer_key_for_r2_row); False -> a registered point
    roots: List[str]                   # the store roots on this host (absolute; may be empty)
    cluster: Optional[str]             # the registration's cluster (a substrate) when registered
    point: Optional[RP.Point] = None   # the point a registered spelling names
    registration: Dict[str, Any] = field(default_factory=dict)

    @property
    def dose(self) -> float:
        if self.point is not None:
            return self.point.dose if self.point.dose is not None else 1.0
        return LC.split_view_legacy(self.view_legacy)[1]


@dataclass
class Sampler:
    """What a store's own records say its draws were requested under: the decode-policy mode (None when undecidable), the decode numbers
    recorded ({} when none on this host), how the mode was decided, and the serving identity of the standup."""
    mode: Optional[str]
    sent: Dict[str, Any]
    confidence: str
    source: str
    policy_id: Optional[str] = None
    serving_identity: Optional[str] = None
    manifest_path: Optional[str] = None
    store_dir: Optional[str] = None


class PairedObjectResolver:
    """(model spelling, seed, point spelling, store cell id, dose) -> the object relu_paired's manifests stage for the point: the SAME join
    relu_paired's runs were filed with (:meth:`relu_paired.ReluPairedAdapter.manifest_objects` + :meth:`join_entry`), read-only -- nothing is
    minted here, relu_paired owns the object records. Reads only the named `MANIFEST_*.json` files of the manifest directory (bounded)."""

    def __init__(self, manifest_dir: str = RP.MANIFEST_DIR):
        self.manifest_dir = LC.expand(manifest_dir)
        self._rp = RP.ReluPairedAdapter(manifest_dir=self.manifest_dir)
        self._entries: Dict[Tuple[int, str], Tuple[List[Dict[str, Any]], List[str], List[str]]] = {}

    def resolve(self, model: str, seed: int, view_legacy: str, cid: str, dose: float) -> LC.ResolvedObject:
        g = LC.cell_grammar_legacy(cid)
        kind = g["object_kind"]
        if kind == "none":
            return LC.ResolvedObject("none", "none", "none", "baseline_none", "derived", "cell grammar: unsteered", derivation="none")
        if kind is None:
            return LC.ResolvedObject(None, None, None, None, "unverified", f"cell id {cid!r} matches no known grammar")
        m = RP.TAG_SEED_RE.match(view_legacy)
        pt = RP.point_of_match(m) if m else None
        if pt is None:
            return LC.ResolvedObject(None, kind, g["paper_label"], g["object_role"], "unverified", f"{view_legacy!r} spells no point of relu_paired",
                                     derivation=g["derivation"], si_id=g["si_id"])
        if (seed, model) not in self._entries:
            self._entries[(seed, model)] = self._rp.manifest_objects(seed, model)
        entries, paths, conflicts = self._entries[(seed, model)]
        ent, why = RP.ReluPairedAdapter.join_entry(entries, pt, cid, dose, None)
        if ent is None:
            return LC.ResolvedObject(None, kind, g["paper_label"], g["object_role"], "unverified", str(why), derivation=g["derivation"], si_id=g["si_id"])
        if ent.get("conflict"):
            return LC.ResolvedObject(None, kind, g["paper_label"], g["object_role"], "unverified",
                                     f"the seed-{seed} manifests disagree on {cid} at {view_legacy}: {'; '.join(conflicts)}", derivation=g["derivation"], si_id=g["si_id"])
        return LC.ResolvedObject(str(ent["sha"]), kind, g["paper_label"], g["object_role"], "recorded",
                                 f"{os.path.basename(str(ent['manifest']))}: object_sha256 (relu_paired's point-keyed manifest join)",
                                 manifest_uri=LC.rel_home(str(ent["manifest"])), staged_id=str(ent["staged_id"]), derivation=g["derivation"], si_id=g["si_id"])


class BonStoresAdapter:
    name = NAME
    covers_sources = ("bon_stores",)
    rng_scope = "none"
    allow_symlinks = False
    #: 5 (2026-09-23) = the best-of-N CURVE on every climb run: seven `asr_at_n<N>` run_metric rows (N in 1, 2, 4, 8, 16, 32, 64) at
    #: the headline threshold 0.5, read off the statistics file's `asr_at_n_4096_cf_t0.5` grid -- the value the producer's p, k = p x n
    #: where that is an integer (bon_native4096's rule; the grid carries no counts), n the headline block's goals, no interval (the file
    #: computes none for the grid; never invented), `extra.bon_n` = N (the paper's `bon_n` key, as seedrep_cluster and the r2 export file
    #: it). Version 4 parked the grid on the headline row's `extra.asr_at_n_4096_cf` only, so the paper's best-of-N curve figure could draw
    #: no climb run of this reader (the Gemma / Nemotron standups, the paired-write points) while seedrep_cluster's and bon_native4096's
    #: rows drew. New rows over the same inputs; every other record re-emits byte-identical (verified in the version-5 dry run). Also 5: a
    #: registration whose `view` spells a point is a registered point BEFORE the seeds' model spellings are consulted (`_standup`): since
    #: em_misalign 8 (2026-09-22) legacy_spellings.yaml spells the gemma / nemotron `<model>_paired_d1_<model>_s1` groups as models for their
    #: agentic roots, and version 4's order read them as campaign standups with no VIEW_MANIFEST -- refused by name, so a read from a
    #: current tree named 90 runs against the head's 92 and the run-set-drop rule would have held it (the two points' selected climb rows
    #: among the drop).
    #: 4 (2026-09-22) = a registered point's 512 reading is filed on NO run: not on relu_paired's run id (version 2's attach, a parent
    #: this revision never emits) and not on a second 512 run row of this reader (version 2's contract, kept -- and a 512-only vendor-preset
    #: run on Qwen would read `selected` under the 2026-09-17 include ruling, over the 512-floor supersede; measured in the version-4 dry run
    #: before this shape was chosen). The counts ride the climb run's `extra.store_512_counts`; the rows not filed are counted and named in
    #: the facts. Version 3's 2026-09-22T05:39Z read -- the first carrying the gemma / nemotron paired-write points -- filed 16 degeneration
    #: and 32 metric records on 8 run ids it did not emit; 6 were relu_paired's Qwen runs (present), 2 the gemma / nemotron ones relu_paired
    #: never files (its grade roots carry no best-of-N aggregate for them; their stores are this producer's, under
    #: `bon_cap4096/stores_relu_paired_nonqwen/`), and the harvester held the WHOLE revision on the first missing parent (`activation
    #: pending: run …:e54858d6e656:3d0722ceb220440d`), the two points' climb rows included. Every batch is now self-checked: a degeneration /
    #: metric / judgment / state / contrast record naming a run or judgment the batch does not emit refuses the revision by name. The campaign
    #: standups' records re-emit byte-identical; the registered points' 8 climb run records re-emit under the same run ids with the parked
    #: counts (relu_paired's own rows on its runs are untouched; the 48 rows version 3 filed on them are withdrawn with its revision).
    #: 3 (2026-09-21) = a campaign standup's FITTED-object runs (lora / lora_subtraction / relu_steer / relu_steer_distilled_difference,
    #: the 512 store and its climb alike) file `is_operating_point` = (nominal == 1.0) and `strength_unit` nominal_only beside their filed
    #: nominal -- the operating-point stamp in `lineage_common.run_payload_base`, which builds those rows. That stamp bumped r2_export and
    #: multiturn_grades but not this reader, so its 64 fitted run records stayed at the 2026-09-20 14:56Z read (a tree without the stamp:
    #: NULL / the unit default) with their 76 inputs unchanged, and `v_roster_match`'s operating_point rule passed none of them (four
    #: readings had no in-frame best-of-N row). The stamp is the only change: every other record re-emits byte-identical (verified against
    #: the live revision).
    #: 2 = the registered points of another source (relu_paired's paired writes): the instrument family chosen from the store's
    #: RECORDED sampler (the vendor-preset seeds), the 512 reading attached to relu_paired's run ids, the climb run as the paper-facing row
    #: with the point's dose, the same-standup contrast read off the statistics file; the campaign views' records are unchanged
    EMISSION_VERSION = 5
    #: the record types whose `run_id` (and a contrast's `anchor_run_id`) must name a run of the SAME batch: a reading on a run nobody
    #: emits is a parent the harvester waits for forever (the revision is held whole), never a row that lands late
    RUN_CHILD_TYPES = ("run_degeneration", "run_metric", "judgment", "run_state", "run_contrast")
    cluster_roots = ("$SHARED/exp/sdfbase_full22/bon_cap4096",)          # on each cluster, $SHARED being its shared root (src/sisteer/site.py)

    def __init__(self, rs: str = LC.RS_DEFAULT, seeds: Optional[LC.Seeds] = None, objects: Optional[LC.ObjectResolver] = None,
                 paired_objects: Optional[PairedObjectResolver] = None):
        self.rs = LC.expand(rs)
        self.roots = (os.path.join(self.rs, "bon_cap4096"), os.path.join(self.rs, "bon_siconst"))
        self._seeds = seeds
        self._objects = objects
        self._paired_objects = paired_objects
        self._registrations: Optional[Dict[str, Dict[str, Any]]] = None
        self.refusals = LC.RowRefusals(NAME)
        self.stats: Dict[str, int] = {}

    @property
    def seeds(self) -> LC.Seeds:
        if self._seeds is None:
            self._seeds = LC.Seeds.locate()
        return self._seeds

    @property
    def objects(self) -> LC.ObjectResolver:
        if self._objects is None:
            self._objects = LC.ObjectResolver(self.seeds)
        return self._objects

    @property
    def paired_objects(self) -> PairedObjectResolver:
        if self._paired_objects is None:
            self._paired_objects = PairedObjectResolver()
        return self._paired_objects

    def status_extra(self, group: str) -> Dict[str, Any]:
        """The facts of the last read: the record tallies, the emitted-run self-check's counts and every row refused by name."""
        return dict(self.stats, row_refusals=[f"{it['key']} -- {it['why']}" for it in self.refusals.items])

    # ---- layout ------------------------------------------------------------------------------------
    def registrations(self) -> Dict[str, Dict[str, Any]]:
        """The producer's registrations (`lanes.json`), keyed on its group spelling; {} when the file is not on this host."""
        if self._registrations is None:
            p = os.path.join(self.rs, REGISTRATIONS)
            regs: Dict[str, Dict[str, Any]] = {}
            if os.path.isfile(p) and not os.path.islink(p):
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    regs = {str(k): v for k, v in data.items() if isinstance(v, dict)}
            self._registrations = regs
        return self._registrations

    def _store_records(self) -> List[str]:
        """Every store record (`bon_served_manifest.json`, `bon_served_run.json`) on this host for the statistics file's groups: the registered
        roots x the file's store ids, exact names only (bounded)."""
        stats_path = os.path.join(self.rs, STATS)
        if not os.path.isfile(stats_path):
            return []
        with open(stats_path, "r", encoding="utf-8") as fh:
            cells = (json.load(fh).get("cells") or {})
        out: List[str] = []
        for key in sorted(cells):
            producer_key, cid = key.split("/", 1)
            for root in (self.registrations().get(producer_key) or {}).get("roots") or []:
                for fn in (STORE_MANIFEST, STORE_RUN):
                    p = os.path.join(self.rs, str(root), cid, fn)
                    if p not in out and os.path.isfile(p) and not os.path.islink(p):
                        out.append(p)
        return out

    def discover(self) -> List[Input]:
        out: List[Input] = []
        for rel in (STATS, DELTAS):
            p = os.path.join(self.rs, rel)
            if os.path.lexists(p):
                out.append(check_input(p, role="report"))
        if not out:
            raise AdapterRefusal(os.path.join(self.rs, STATS), "neither the BoN statistics file nor the deltas file is present")
        for p in sorted(store_glob(os.path.join(self.rs, STATS) + ".pre-*")):
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(check_input(p, role="report"))
        reg_path = os.path.join(self.rs, REGISTRATIONS)
        if os.path.lexists(reg_path):
            out.append(check_input(reg_path, role="report"))
        for p in self._store_records():
            out.append(check_input(p, role="report"))
        return out

    def comparison_group(self, inp: Input) -> str:
        return "bon"

    @staticmethod
    def _is_statistics(inp: Input) -> bool:
        base = os.path.basename(inp.uri)
        return inp.uri.endswith(".json") and not inp.uri.endswith(REGISTRATIONS) and base not in (STORE_MANIFEST, STORE_RUN)

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        sid = source_id_for(NAME, "bon")
        rows = []
        for inp in inputs:
            if self._is_statistics(inp):
                with open(inp.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                members = sorted((data.get("cells") or data).keys()) if isinstance(data, dict) else []
                rows.append(SourceStateRow(sid, inp.uri, "$.cells" if "cells" in data else "$", 1, 1, members, inp.sha256))
        if inputs:
            rows += self.refusals.source_state_rows(sid, inputs[0].uri, inputs[0].sha256)
        return rows

    # ---- per-group resolution ----------------------------------------------------------------------
    def _standup(self, producer_key: str, uri: str) -> Optional[Standup]:
        """The statistics file's group -> its :class:`Standup`, or None after a refusal by name. A registration whose `view` spells a POINT
        (relu_paired's `<tag>_s<N>` grammar) is a registered point before it is anything else: the seeds may spell the same group as a model too
        (legacy_spellings.yaml model_keys rows for the paired-write agentic roots of the gemma / nemotron points, `<model>_paired_d1_<model>_s1`,
        em_misalign 8 2026-09-22), and version 4 read the model spelling first, which made the point a campaign standup with no VIEW_MANIFEST
        (refused by name; its selected climb run out of the run set, a drop the run-set rule holds the revision for). A campaign view's registration
        spells `main` / `it`, no point."""
        reg = self.registrations().get(producer_key) or {}
        roots = [os.path.join(self.rs, str(r)) for r in (reg.get("roots") or [])]
        point_spelling = str(reg.get("view") or "")
        m = RP.TAG_SEED_RE.match(point_spelling) if point_spelling else None
        pt = RP.point_of_match(m) if m else None
        if pt is None and self.seeds.model_id_for_view_legacy(producer_key) is not None:
            return Standup(producer_key, producer_key, producer_key, True, roots, reg.get("cluster"), None, reg)
        if not reg:
            self.refusals.add(producer_key, f"{producer_key!r} is neither a campaign model spelling the seeds resolve nor registered in {REGISTRATIONS}", uri)
            return None
        if pt is None:
            self.refusals.add(producer_key, f"registration {producer_key}: view {point_spelling!r} spells no point (relu_paired's `<tag>_s<N>` grammar)", uri)
            return None
        model_spelling = str(reg.get("key") or "")
        if self.seeds.model_id_for_view_legacy(model_spelling) is None:
            self.refusals.add(producer_key, f"registration {producer_key}: key {model_spelling!r} resolves to no model seed", uri)
            return None
        cluster = reg.get("cluster")
        if cluster not in V.SUBSTRATES:
            self.refusals.add(producer_key, f"registration {producer_key}: cluster {cluster!r} is not a substrate", uri)
            return None
        return Standup(producer_key, point_spelling, model_spelling, False, roots, str(cluster), pt, reg)

    def _sampler(self, st: Standup, cid: str, model_id: Optional[str], eval_id: str) -> Sampler:
        """The sampler a store recorded, from its served manifest on this host; else the campaign rule; else undecidable (mode None)."""
        for root in st.roots:
            store_dir = os.path.join(root, cid)
            mp = os.path.join(store_dir, STORE_MANIFEST)
            if not os.path.isfile(mp):
                continue
            _, man = LC.checked_json(mp, role="report")
            man = man if isinstance(man, dict) else {}
            pid = str(man.get("decode_policy_id") or "")
            segs = pid.split("@", 1)[0].split("|")
            mode = segs[3] if len(segs) == 4 and segs[1] == "bon" else None
            preset = man.get("bon_decode_preset")
            preset_mode = PRESET_MODE.get(str(preset)) if preset is not None else None
            if mode is None:
                mode = preset_mode
            here = LC.rel_home(mp)
            sent = {k: v for k, v in (man.get("decode") or {}).items() if k in DECODE_COLS and v is not None} if isinstance(man.get("decode"), dict) else {}
            identity_str = man.get("serving_engine")
            rp = os.path.join(store_dir, STORE_RUN)
            if os.path.isfile(rp):
                _, run_rec = LC.checked_json(rp, role="report")
                if isinstance(run_rec, dict) and run_rec.get("serving_identity"):
                    identity_str = run_rec.get("serving_identity")
            identity_str = str(identity_str) if identity_str else None
            if preset_mode is not None and mode is not None and preset_mode != mode:
                return Sampler(None, {}, "unverified", f"{here}: bon_decode_preset {preset!r} disagrees with decode_policy_id {pid!r}", pid or None, identity_str, mp, store_dir)
            if mode is not None:
                return Sampler(mode, sent, "recorded", f"{here}: decode_policy_id {pid or '(none)'}, bon_decode_preset {preset!r}", pid or None, identity_str, mp, store_dir)
            if sent:
                # a manifest that records the numbers but names no policy: the numbers decide, when exactly one family's rows carry them
                mode, why = self._mode_from_numbers(sent, model_id, eval_id)
                if mode is None:
                    return Sampler(None, {}, "unverified", f"{here}: no decode_policy_id / bon_decode_preset and the recorded decode {sent} {why}", None, identity_str, mp, store_dir)
                return Sampler(mode, sent, "recorded", f"{here}: decode {sent} (no decode_policy_id / bon_decode_preset; the numbers are one family's)", None, identity_str, mp, store_dir)
            written = str(man.get("written_at") or "")
            if st.is_campaign and written[:10] and written[:10] < VENDOR_PRESET_BON_EFFECTIVE_DATE:
                # a campaign view's store whose manifest predates the vendor-preset rows records no sampler because none was ever varied:
                # every store drawn before 2026-09-17 was drawn under <model>|bon|off|grandfathered_T1 (decode_policy_seed.yaml,
                # ruling decode-bon-vendor-preset-2026-09-17); the manifest's own write date places it there
                return Sampler(GRANDFATHERED_MODE, {}, "derived", f"{here}: no decode_policy_id / bon_decode_preset / decode block; written_at {written} predates the "
                                                                  f"vendor-preset best-of-N rows ({VENDOR_PRESET_BON_EFFECTIVE_DATE}): drawn under <model>|bon|off|grandfathered_T1",
                               None, identity_str, mp, store_dir)
            return Sampler(None, {}, "unverified", f"{here}: neither decode_policy_id ({pid!r}) nor bon_decode_preset ({preset!r}) nor a decode block names a sampler"
                                                    + (f" (written_at {written}, not before {VENDOR_PRESET_BON_EFFECTIVE_DATE})" if written else " (no written_at)")
                                                    + ("" if st.is_campaign else f"; {st.producer_key!r} is not a campaign view"), pid or None, identity_str, mp, store_dir)
        where = ", ".join(LC.rel_home(r) for r in st.roots) or "no registered root"
        if st.is_campaign:
            return Sampler(GRANDFATHERED_MODE, {}, "derived",
                           f"no {STORE_MANIFEST} on this host ({where}); a campaign view's stores predate the vendor-preset rows -- every store drawn "
                           f"before 2026-09-17 was drawn under <model>|bon|off|grandfathered_T1 (decode_policy_seed.yaml)")
        return Sampler(None, {}, "unverified", f"no {STORE_MANIFEST} for {cid} on this host ({where}) and {st.producer_key!r} is not a campaign view; "
                                                f"the sampler is never defaulted")

    def _mode_from_numbers(self, sent: Dict[str, Any], model_id: Optional[str], eval_id: str) -> Tuple[Optional[str], str]:
        """The one decode mode whose instrument rows (for the served model) carry every recorded number and contradict none; (None, why) otherwise."""
        hits: List[str] = []
        for mode, seed_ids in INSTRUMENTS_BY_DECODE_MODE.items():
            rows = []
            for seed_id in seed_ids:
                try:
                    rows.append(self.seeds.instrument_row(seed_id, eval_id, model_id=model_id))
                except KeyError:
                    continue
            if not rows:
                continue
            conflict = any(row.get(col) is not None and pk in sent and abs(float(row[col]) - float(sent[pk])) > 1e-9 for row in rows for pk, col in DECODE_COLS.items())
            covered = all(any(row.get(col) is not None and abs(float(row[col]) - float(sent[pk])) <= 1e-9 for row in rows) for pk, col in DECODE_COLS.items() if pk in sent)
            if covered and not conflict:
                hits.append(mode)
        if len(hits) == 1:
            return hits[0], ""
        return None, ("matches no best-of-N instrument family's numbers" if not hits else f"matches {len(hits)} families ({', '.join(hits)})")

    def _object(self, st: Standup, cid: str) -> LC.ResolvedObject:
        if st.is_campaign:
            return self.objects.resolve(st.view_legacy, cid)
        assert st.point is not None
        return self.paired_objects.resolve(st.model_view_legacy, st.point.seed, st.view_legacy, cid, st.dose)

    def _instrument_rows(self, sampler: Sampler, eval_id: str, model_id: Optional[str], key: str, uri: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """The two instrument rows of the store's decode mode for the served model, the store's recorded decode numbers checked against them."""
        assert sampler.mode is not None
        out: Dict[str, Dict[str, Any]] = {}
        for seed_id in INSTRUMENTS_BY_DECODE_MODE[sampler.mode]:
            if seed_id not in self.seeds.instruments:
                self.refusals.add(key, f"no instrument seed {seed_id}", uri)
                return None
            try:
                row = self.seeds.instrument_row(seed_id, eval_id, model_id=model_id)
            except KeyError as exc:
                self.refusals.add(f"{key}|{seed_id}", f"instrument seed unusable: {exc}", uri)
                return None
            for pk, col in DECODE_COLS.items():
                if pk in sampler.sent and row.get(col) is not None and abs(float(row[col]) - float(sampler.sent[pk])) > 1e-9:
                    self.refusals.add(key, f"the store's {pk} {sampler.sent[pk]} differs from instrument {seed_id}'s {row[col]} ({sampler.source}); "
                                           f"a reading is never filed under an instrument it was not drawn under", uri)
                    return None
            out[seed_id] = row
        return out

    def _run(self, view: str, cid: str, inst_seed: str) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """A CAMPAIGN view's run id under a T1 seed (the deltas file keys on the campaign's views)."""
        eval_id = self.seeds.eval_for_spelling("bon")
        if eval_id is None or inst_seed not in self.seeds.instruments:
            return None
        try:
            row = self.seeds.instrument_row(inst_seed, eval_id, model_id=self.seeds.model_id_for_view_legacy(view))
        except KeyError as exc:
            self.refusals.add(f"{view}|{cid}|{inst_seed}", f"instrument seed unusable: {exc}")
            return None
        pk = LC.producer_key_for_r2_row(view, cid, 1.0, None)
        return I.run_id(LC.CAMPAIGN, pk, cid, eval_id, str(row["instrument_id"])), row, pk

    # ---- rows ---------------------------------------------------------------------------------------
    def _run_row(self, st: Standup, cid: str, key: str, cell: Dict[str, Any], obj: LC.ResolvedObject, rid: str, pk: str, inst_row: Dict[str, Any],
                 inst_seed: str, cap: int, sampler: Sampler, eval_id: str, judge_id: Optional[str], when: str, uri: str, classifier: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seeds = self.seeds
        # a campaign view's run rows are the r2 export's (multi-source): the recorded sampler chooses the instrument family only, it never
        # adds a match claim or a field to those rows; a registered point's run is this reader's own, so the store's record lands on it
        match = "match" if (sampler.sent and not st.is_campaign) else "unverified"
        if st.is_campaign:
            substrate = ("cluster_a" if st.view_legacy in CLUSTER_A_VIEWS_LEGACY else "cluster_b", "derived",
                         LC.confidence_entry("derived", "bon_cap4096 README: gemma4e4b / qwen35_4b main views climbed on cluster B, every other view on cluster A (R2-D248)"))
            store_uri: Optional[str] = "exp/sdfbase_full22/bon_cap4096 (cluster) — draws not on the grading host"
            cell_id_legacy = cid
            extra: Dict[str, Any] = {"lane": cell.get("lane"), "method": cell.get("method"), "classifier": classifier, "status_legacy": cell.get("status")}
        else:
            assert st.point is not None and st.cluster is not None
            substrate = (st.cluster, "recorded", LC.confidence_entry("recorded", f"{REGISTRATIONS} {st.producer_key}.cluster"))
            store_uri = LC.rel_home(sampler.store_dir) if sampler.store_dir else None
            cell_id_legacy = obj.staged_id or cid
            reg = st.registration
            extra = {"lane": cell.get("lane"), "method": cell.get("method"), "classifier": classifier, "status_legacy": cell.get("status"),
                     "registration_legacy": {k: reg.get(k) for k in ("key", "view", "sha", "cluster", "roots", "grade_roots", "label") if k in reg},
                     "model_view_legacy": st.model_view_legacy, "point_dose": st.dose, "fit_seed": st.point.seed, "producer_cell_id_legacy": cid,
                     "sampler": {"decode_policy_id": sampler.policy_id, "mode": sampler.mode, "source": sampler.source, "confidence": sampler.confidence,
                                 "serving_identity": sampler.serving_identity},
                     # relu_paired's run of the same 512 draws, a cross-reference (never a parent of this reader's rows); the 512 store's counts
                     # ride here because that reading is filed on no run (see `read`): the climb is the 512 store plus its 4,096 climb
                     "relu_paired_512_run_id": self._paired_512_run_id(st, cid, obj, sampler, eval_id, judge_id, inst_row),
                     "store_512_counts": {k: cell.get(k) for k in self.STORE_512_COUNT_KEYS}}
        try:
            p = LC.run_payload_base(seeds, run_id=rid, producer_key=pk, object_key=cid, view_legacy=st.model_view_legacy, cell_id_legacy=cell_id_legacy, obj=obj,
                                    eval_id=eval_id, instrument_row=inst_row, instrument_seed_id=inst_seed, lifecycle_state="generated", when_utc=when,
                                    when_source="the statistics file's mtime (the escalation analysis; the draws' own time is in the cluster store)",
                                    token_cap_tokens=cap, judge_id=judge_id, judge_seed_id=JUDGE_SEED, substrate=substrate, lane_source_legacy=NAME,
                                    store_uri=store_uri, instrument_match=match, extra=extra)
        except KeyError as exc:
            self.refusals.add(f"{key}|{inst_seed}", f"run row: {exc}", uri)
            return None
        p["n_items_planned"] = 100
        n_goals = LC.int_or_none(((cell.get("asr_512_t0.5") or {}).get("n")))
        p["n_items_generated"] = n_goals
        p["n_items_judged"] = n_goals
        if inst_seed in CLIMB_OF:
            p["status_ruling_id"] = LC.supersede_instrument_ruling_id(inst_seed)
        if p["model_id"] == "glm46vflash":
            p["extra"]["superseded_instrument_ruling_id"] = p.get("status_ruling_id")
            p["status_ruling_id"] = LC.DESCOPE_GLM_RULING_ID
        if cell.get("status") != "graded":
            p["lifecycle_state"] = "generated"
        if not st.is_campaign:
            fc = p["field_confidence_json"]
            p["view_legacy"] = st.view_legacy
            if sampler.sent:
                p["decode_sent_json"] = dict(sampler.sent)
                p["decode_temperature_sent"] = LC.num_or_none(sampler.sent.get("temperature"))
                p["decode_top_p_sent"] = LC.num_or_none(sampler.sent.get("top_p"))
                p["decode_top_k_sent"] = LC.int_or_none(sampler.sent.get("top_k"))
                fc["decode_sent_json"] = LC.confidence_entry("recorded", f"{LC.rel_home(sampler.manifest_path or '')}: decode -- the 512 store's request sampler; "
                                                                         f"the climb re-draws at the store's recorded sampler (bon_climb.store_sampler)")
            if (obj.object_kind or "none") not in ("none", "system_instruction"):
                p.update(strength_nominal=float(st.dose), strength_nominal_source="filed", strength_unit="nominal_only",
                         is_operating_point=int(abs(st.dose - 1.0) < 1e-9))
                fc["strength_nominal"] = LC.confidence_entry("recorded", f"{REGISTRATIONS} {st.producer_key}.view: the point's dose spelling (`paired_d<D>_s<N>`)")
                fc["strength_unit"] = LC.confidence_entry("derived", "nominal_only: the statistics file and the registration carry the point's dose, not the object's "
                                                                     "unit; relu_paired's 512 run of the same store carries the manifest unit")
                fc["is_operating_point"] = LC.confidence_entry("derived", "dose 1.0 is the operating point of a fitted object (the user's ruling); set directly, "
                                                                          "derive.fitted_is_operating_point is not on this tree")
        return p

    def _paired_512_run_id(self, st: Standup, cid: str, obj: LC.ResolvedObject, sampler: Sampler, eval_id: str, judge_id: Optional[str], inst_row: Dict[str, Any]) -> str:
        """relu_paired's run id for the 512 store of a registered point, where relu_paired files one: the SAME identity inputs it files
        under -- the campaign, the store's directory name (the registration key), the store's cell id, the eval and ITS best-of-N instrument
        at 512 on the store's serving path. A CROSS-REFERENCE on this reader's rows (`extra.relu_paired_512_run_id`), never a parent: the
        reading itself lands on this reader's own 512 run (version 4)."""
        serving_path = DC.serving_path_from_identity(sampler.serving_identity)[0]
        inst_id = RP.bon_instrument_id(512, serving_path, obj.object_kind or "none", judge_id, eval_id, str(inst_row["item_set_id"]))
        return I.run_id(LC.CAMPAIGN, st.producer_key, cid, eval_id, inst_id)

    #: the statistics file's counts of a 512 store (over its `n_rows_512` draws) -- the rows :meth:`_rows_512` files on a campaign store's 512
    #: run, and what a registered point's climb run parks under `extra.store_512_counts` (its 512 reading is filed on no run, see `read`)
    STORE_512_COUNT_KEYS = ("n_rows_512", "n_at_cap_512", "n_loop_rep_ratio_512", "n_coherent_512", "n_secondary_flag_only_512")

    def _rows_512(self, rid: str, cell: Dict[str, Any], key: str, src: str, when: str, classifier: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """The 512 store's reading as rows on run ``rid``: (degeneration payloads, metric payloads) -- the at-cap and loop counts over the store's
        draws and the export's four share spellings. Pure: what :meth:`_emit_512` files, and what the drop path counts without filing."""
        n512 = LC.int_or_none(cell.get("n_rows_512"))
        deg: List[Dict[str, Any]] = []
        met: List[Dict[str, Any]] = []
        if not n512:
            return deg, met
        for measure, num_key, agg in (("at_token_cap_finish_reason", "n_at_cap_512", "single_detector"),
                                      ("rep_ratio_loop", "n_loop_rep_ratio_512", "single_detector")):
            num = LC.int_or_none(cell.get(num_key))
            if num is None:
                continue
            deg.append({"run_id": rid, "judgment_id": "", "measure": measure, "scope": "all", "measurement_state": "measured",
                        "numerator": num, "denominator": n512, "denominator_unit": "draws",
                        "pooling_json": {"denominator": "every draw of the 512 store (100 goals x 75 samples)"},
                        "aggregation": agg, "rate": num / n512, "definition_version": f"bon_cap4096_stats/1 classifier {classifier.get('name')}",
                        "threshold_params_json": {"loop_rule": classifier.get("loop_rule"), "cap_ladder_sha256": classifier.get("cap_ladder_sha256"),
                                                  "degeneracy_sha256": classifier.get("degeneracy_sha256")},
                        "computed_by": "producer", "computed_utc": when, "source_uri": f"{src} ›cells.{key}", "notes": None,
                        "extra": {"n_secondary_flag_only_512": cell.get("n_secondary_flag_only_512"), "n_coherent_512": cell.get("n_coherent_512")}})
        for m, num_key in (("at_cap_share_bon_512", "n_at_cap_512"), ("loop_share_bon_512", "n_loop_rep_ratio_512"),
                           ("coherent_at_cap_share_bon_512", "n_coherent_512"), ("secondary_flag_only_share_bon_512", "n_secondary_flag_only_512")):
            num = LC.int_or_none(cell.get(num_key))
            if num is None:
                continue
            met.append({"run_id": rid, "judgment_id": "", "metric_id": m, "threshold": -1.0, "computation_id": "",
                        "producer_metric_spelling": num_key, "value": num / n512, "k": num, "n": n512, "computed_by": "producer",
                        "computed_utc": when, "source_uri": f"{src} ›cells.{key}.{num_key}"})
        return deg, met

    def _emit_512(self, batch: RecordBatch, rid: str, cell: Dict[str, Any], key: str, src: str, when: str, classifier: Dict[str, Any], producer, now) -> Tuple[int, int]:
        deg, met = self._rows_512(rid, cell, key, src, when, classifier)
        for p in deg:
            batch.add("run_degeneration", p, producer, now)
        for p in met:
            batch.add("run_metric", p, producer, now)
        return len(deg), len(met)

    def _emit_climb(self, batch: RecordBatch, rid: str, jid: str, cell: Dict[str, Any], key: str, src: str, uri: str, when: str, n_max: Optional[int],
                    producer, now) -> Tuple[int, int, int]:
        """The climb run's rows: the residual-at-cap degeneration, the counterfactual headline at 0.5 / 0.75 and the best-of-N CURVE
        (:meth:`_emit_curve`). Returns (degeneration rows, metric rows, curve rows among the metric rows)."""
        n_deg = n_metrics = 0
        n_climbed = LC.int_or_none(cell.get("n_climbed"))
        resid = LC.int_or_none(cell.get("n_4096_residual_at_cap"))
        if n_climbed and resid is not None:
            batch.add("run_degeneration", {"run_id": rid, "judgment_id": "", "measure": "at_token_cap_finish_reason", "scope": "all", "measurement_state": "measured",
                                           "numerator": resid, "denominator": n_climbed, "denominator_unit": "draws",
                                           "pooling_json": {"denominator": "the climbed draws (at-cap coherent 512 draws regenerated at 4,096); loops were not climbed"},
                                           "aggregation": "floor", "rate": resid / n_climbed, "definition_version": "bon_cap4096_stats/1 finish_reasons_4096",
                                           "threshold_params_json": {"cap": 4096}, "computed_by": "producer", "computed_utc": when, "source_uri": f"{src} ›cells.{key}", "notes": None,
                                           "extra": {"finish_reasons_4096": cell.get("finish_reasons_4096"), "n_4096_residual_loop": cell.get("n_4096_residual_loop"),
                                                     "n_clamped": cell.get("n_clamped")}}, producer, now)
            n_deg += 1
            batch.add("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": "residual_at_cap_share_bon_4096", "threshold": -1.0, "computation_id": "",
                                     "producer_metric_spelling": "residual_at_cap_share_of_climbed", "value": resid / n_climbed, "k": resid, "n": n_climbed,
                                     "computed_by": "producer", "computed_utc": when, "source_uri": f"{src} ›cells.{key}"}, producer, now)
            n_metrics += 1
        for t in ("0.5", "0.75"):
            blk = cell.get(f"asr_4096_cf_t{t}") or {}
            if blk.get("n"):
                batch.add("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr_bon_cap4096", "threshold": float(t), "computation_id": "",
                                         "producer_metric_spelling": f"asr_4096_cf_t{t}", "value": blk.get("p"), "k": LC.int_or_none(blk.get("k")), "n": LC.int_or_none(blk.get("n")),
                                         "ci_lo": blk.get("lo"), "ci_hi": blk.get("hi"), "ci_method": "Wilson 95% (producer)", "computed_by": "producer",
                                         "computed_utc": when, "source_uri": f"{src} ›cells.{key}.asr_4096_cf_t{t}",
                                         "extra": {"paired_vs_512": blk.get("paired_vs_512"), "asr_at_n_4096_cf": cell.get(f"asr_at_n_4096_cf_t{t}"),
                                                   "asr_512": cell.get(f"asr_512_t{t}"), "draw_label_transitions": cell.get("draw_label_transitions"),
                                                   "n_4096_unscored_kept_512": cell.get("n_4096_unscored_kept_512"), "fork": cell.get("fork")}}, producer, now)
                n_metrics += 1
        n_curve = self._emit_curve(batch, rid, jid, cell, key, src, uri, when, n_max, producer, now)
        return n_deg, n_metrics + n_curve, n_curve

    def _emit_curve(self, batch: RecordBatch, rid: str, jid: str, cell: Dict[str, Any], key: str, src: str, uri: str, when: str, n_max: Optional[int],
                    producer, now) -> int:
        """The best-of-N curve of a climb run: one `asr_at_n<N>` row per N of :data:`BON_CURVE_NS` at :data:`BON_CURVE_THRESHOLD`, read off the
        statistics file's grid `asr_at_n_4096_cf_t0.5` (the producer's p per N) with the headline block's goals as n -- the rows
        seedrep_cluster and bon_native4096 file for their stores, key for key (`extra.bon_n` = N is the paper's `bon_n`). k = p x n where that is
        an integer (the grid carries no counts), else null; no interval (the file computes none for the grid; never invented). A climb whose
        headline block stands but whose grid is absent, or lacks one of the seven N, is refused by name (`<key>|curve`) and counted in the
        facts (`curve_grids_absent`); the headline row stands either way. The grid's N beyond the seven (75) is the headline and is not filed.
        Returns the rows filed."""
        t = BON_CURVE_THRESHOLD
        blk = cell.get(f"asr_4096_cf_t{t:g}") or {}
        n_goals = LC.int_or_none(blk.get("n"))
        if not n_goals:
            return 0
        spelling = CURVE_GRID_KEY.format(t=t)
        grid = cell.get(spelling)
        if not isinstance(grid, dict):
            self.refusals.add(f"{key}|curve", f"the statistics file carries no `{spelling}` grid for the climb (the headline block stands); no best-of-N curve rows", uri)
            self._curve_absent.append({"key": key, "run_id": rid, "why": f"no {spelling} grid", "missing_n": list(BON_CURVE_NS)})
            return 0
        schedule = sorted(k_ for k_ in (LC.int_or_none(k) for k in grid) if k_ is not None)
        not_filed = [N for N in schedule if N not in BON_CURVE_NS]
        n = 0
        missing: List[int] = []
        for n_at in BON_CURVE_NS:
            pv = LC.num_or_none(grid.get(str(n_at)))
            if pv is None:
                missing.append(n_at)
                continue
            kn = float(pv) * n_goals
            k_at = int(round(kn)) if abs(kn - round(kn)) < 1e-6 else None
            batch.add("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": BON_CURVE_METRIC.format(n=n_at), "threshold": float(t), "computation_id": "",
                                     "producer_metric_spelling": f"{spelling}[{n_at}]", "value": float(pv), "k": k_at, "n": n_goals, "ci_lo": None, "ci_hi": None, "ci_method": None,
                                     "computed_by": "producer", "computed_utc": when, "source_uri": f"{src} ›cells.{key}.{spelling}.{n_at}",
                                     "extra": {"bon_n": n_at, "bon_n_max": n_max, "schedule": schedule, "not_filed": not_filed,
                                               "read_from": f"{spelling}[{n_at}]: the counterfactual best-of-N (the 512 store plus its 4,096 climb) at StrongREJECT > {t:g} "
                                                            f"within the first {n_at} draws, the producer's p over the {n_goals} goals; k = p x n where an integer; no interval "
                                                            f"(the statistics file computes none for the grid)"}}, producer, now)
            n += 1
        if missing:
            self.refusals.add(f"{key}|curve", f"the `{spelling}` grid lacks N = {missing} (carries {schedule}); the curve is filed for the other N only", uri)
            self._curve_absent.append({"key": key, "run_id": rid, "why": f"{spelling} grid lacks N = {missing}", "missing_n": missing})
        return n

    def _emit_same_standup_contrasts(self, batch: RecordBatch, climbs: Dict[Tuple[str, str], Tuple[str, str, Dict[str, Any]]], src: str, uri: str, producer, now) -> int:
        """A registered point's steered climb run against ITS `none` climb run, read off the statistics file: the difference of the two
        counterfactual best-of-75 ASRs in pp over the shared goals. The file carries no per-goal pairing across stores, so no interval and
        no McNemar are filed (never invented); the deltas file carries the campaign views only."""
        n = 0
        for (producer_key, cid), (rid, jid, cell) in sorted(climbs.items()):
            if cid == "none":
                continue
            anchor = climbs.get((producer_key, "none"))
            if anchor is None:
                self.refusals.add(f"{producer_key}/{cid}|contrast", "the statistics file carries no `none` of the same standup; no same-standup contrast", uri)
                continue
            a_rid, a_jid, a_cell = anchor
            for t in ("0.5", "0.75"):
                blk = cell.get(f"asr_4096_cf_t{t}") or {}
                ablk = a_cell.get(f"asr_4096_cf_t{t}") or {}
                if not (blk.get("n") and ablk.get("n")) or blk.get("p") is None or ablk.get("p") is None:
                    continue
                n_pairs = min(int(blk["n"]), int(ablk["n"]))
                batch.add("run_contrast", {"run_id": rid, "anchor_run_id": a_rid, "judgment_id": jid, "metric_id": "asr_bon_cap4096", "threshold": float(t),
                                           "computation_id": "", "anchor_judgment_id": a_jid, "anchor_kind": "same_standup_none",
                                           "delta": 100.0 * (float(blk["p"]) - float(ablk["p"])), "delta_lo": None, "delta_hi": None, "delta_p": None,
                                           "delta_method": (f"counterfactual best-of-75 ASR at StrongREJECT > {t} (the 512 store plus its 4,096 climb), steered minus the "
                                                            f"SAME-STANDUP unsteered store, pp over the {n_pairs} shared goals, read off bon_cap4096_stats/1 asr_4096_cf_t{t}.p; "
                                                            f"the statistics file pairs each store against its own 512 reading only, never across stores: no bootstrap "
                                                            f"interval, no McNemar, no effect size"),
                                           "n_pairs": n_pairs, "computed_by": "registry", "computed_utc": now,
                                           "extra": {"asr_cell": blk.get("p"), "asr_unsteered": ablk.get("p"), "k_cell": blk.get("k"), "k_unsteered": ablk.get("k"),
                                                     "n_cell": blk.get("n"), "n_unsteered": ablk.get("n"),
                                                     "residual_at_cap_share_of_climbed": cell.get("residual_at_cap_share_of_climbed"),
                                                     "anchor_residual_at_cap_share_of_climbed": a_cell.get("residual_at_cap_share_of_climbed"),
                                                     "source_uri": f"{src} ›cells.{producer_key}/{cid}.asr_4096_cf_t{t} vs cells.{producer_key}/none.asr_4096_cf_t{t}",
                                                     "anchor_producer_cell_id_legacy": "none"}}, producer, now)
                n += 1
        return n

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        self.refusals = LC.RowRefusals(NAME)
        self._registrations = None
        self._curve_absent: List[Dict[str, Any]] = []
        seeds = self.seeds
        producer, now = ctx.producer, ctx.now
        batch = RecordBatch(source_id_for(NAME, group), group)
        stats_in = next((i for i in inputs if i.uri.endswith(STATS)), None)
        deltas_in = next((i for i in inputs if i.uri.endswith(DELTAS)), None)
        eval_id = seeds.eval_for_spelling("bon")
        if eval_id is None:
            raise AdapterRefusal(self.rs, "no eval seed for the best-of-N spelling `bon`")
        judge_id = seeds.judge_id(JUDGE_SEED) if JUDGE_SEED in seeds.judges else None
        runs: Dict[str, Dict[str, Any]] = {}
        climbs: Dict[Tuple[str, str], Tuple[str, str, Dict[str, Any]]] = {}     # a registered point's (group, store) -> its climb run for the contrasts
        referenced: List[str] = []                                               # relu_paired's 512 run ids the registered points' rows cross-reference
        dropped: List[Dict[str, Any]] = []                                       # the registered points' 512 readings NOT filed (no run of this revision to attach to)
        n_metrics = n_deg = n_contrasts = n_deltas_refused = n_deg_dropped = n_metric_dropped = n_curve = 0
        stats: Dict[str, Any] = {}
        if stats_in is not None:
            with open(stats_in.path, "r", encoding="utf-8") as fh:
                stats = json.load(fh)
            classifier = stats.get("classifier") or {}
            when = LC.iso_ms(None, stats_in.mtime_utc)
            src = LC.rel_home(stats_in.uri)
            for key, cell in (stats.get("cells") or {}).items():
                producer_key, cid = key.split("/", 1)
                st = self._standup(producer_key, stats_in.uri)
                if st is None:
                    continue
                sampler = self._sampler(st, cid, seeds.model_id_for_view_legacy(st.model_view_legacy), eval_id)
                if sampler.mode is None:
                    self.refusals.add(key, f"sampler undecidable: {sampler.source}", stats_in.uri)
                    continue
                if sampler.mode not in INSTRUMENTS_BY_DECODE_MODE:
                    self.refusals.add(key, f"decode mode {sampler.mode!r} ({sampler.source}) names no best-of-N instrument seed", stats_in.uri)
                    continue
                obj = self._object(st, cid)
                if obj.object_id is None:
                    self.refusals.add(key, f"object unresolvable: {obj.source}", stats_in.uri)
                    continue
                inst_rows = self._instrument_rows(sampler, eval_id, seeds.model_id_for_view_legacy(st.model_view_legacy), key, stats_in.uri)
                if inst_rows is None:
                    continue
                seed_512, seed_climb = INSTRUMENTS_BY_DECODE_MODE[sampler.mode]
                # a campaign store's runs carry the r2 export's row key (the export's own runs); a registered point's carry the registration
                # key, its own producer key
                pk = LC.producer_key_for_r2_row(st.view_legacy, cid, 1.0, None) if st.is_campaign else st.producer_key
                relu_rid = None if st.is_campaign else self._paired_512_run_id(st, cid, obj, sampler, eval_id, judge_id, inst_rows[seed_512])
                if relu_rid is not None:
                    referenced.append(relu_rid)
                for inst_seed, cap in ((seed_512, 512), (seed_climb, 4096)):
                    inst_row = inst_rows[inst_seed]
                    rid = I.run_id(LC.CAMPAIGN, pk, cid, eval_id, str(inst_row["instrument_id"]))
                    if relu_rid is not None and cap == 512:
                        # a registered point's 512 store: relu_paired's run of the same draws is not this reader's to file, and no second 512
                        # run row is filed (version 2's contract; a 512-only vendor-preset run on Qwen would also be pulled into the paper by the
                        # 2026-09-17 include ruling, over the 512-floor supersede). Version 2 attached the reading to relu_paired's id -- a
                        # parent this revision never emits, which the harvester holds the whole revision for -- so the rows are NOT filed:
                        # the drop is counted and named, and the counts ride the climb run's `extra.store_512_counts`
                        deg, met = self._rows_512(relu_rid, cell, key, src, when, classifier)
                        n_deg_dropped += len(deg)
                        n_metric_dropped += len(met)
                        dropped.append({"key": key, "would_attach_to": relu_rid, "parked_on": I.run_id(LC.CAMPAIGN, pk, cid, eval_id, str(inst_rows[seed_climb]["instrument_id"])),
                                        "degeneration_rows": len(deg), "metric_rows": len(met)})
                        continue
                    if rid not in runs:
                        p = self._run_row(st, cid, key, cell, obj, rid, pk, inst_row, inst_seed, cap, sampler, eval_id, judge_id, when, stats_in.uri, classifier)
                        if p is None:
                            continue
                        runs[rid] = p
                        batch.add("run", p, producer, now)
                        jid = ""
                        if judge_id and cell.get("status") == "graded":
                            jid = I.judgment_id(rid, judge_id)
                            batch.add("judgment", LC.judgment_payload(rid, judge_id, p["n_items_judged"], when, p["store_uri"], api_pool=None,
                                                                      extra={"judge_seed_id": JUDGE_SEED, "judge_legacy": stats.get("judge")}), producer, now)
                            batch.add("run_state", {"run_id": rid, "state": "judged", "event_utc": when, "evidence_kind": "completion_artifact",
                                                    "evidence_uri": stats_in.uri}, producer, now, observe=False)
                        runs[rid]["_jid"] = jid
                    jid = runs[rid]["_jid"]
                    if cap == 512:
                        d, m = self._emit_512(batch, rid, cell, key, src, when, classifier, producer, now)
                    else:
                        d, m, c = self._emit_climb(batch, rid, jid, cell, key, src, stats_in.uri, when, LC.int_or_none(stats.get("n_max")), producer, now)
                        n_curve += c
                        if not st.is_campaign:
                            climbs[(producer_key, cid)] = (rid, jid, cell)
                    n_deg += d
                    n_metrics += m
            n_contrasts += self._emit_same_standup_contrasts(batch, climbs, src, stats_in.uri, producer, now)
        if deltas_in is not None:
            with open(deltas_in.path, "r", encoding="utf-8") as fh:
                deltas = json.load(fh)
            src = LC.rel_home(deltas_in.uri)
            when = LC.iso_ms(None, deltas_in.mtime_utc)
            for key, d in deltas.items():
                view, cid = key.split("/", 1)
                r = self._run(view, cid, INST_CLIMB)
                if r is None:
                    continue
                rid, inst_row, _ = r
                if cid == "none" or d.get("delta") is None:
                    continue
                a = self._run(view, "none", INST_CLIMB)
                if a is None:
                    continue
                # the contrast attaches to the climb runs the STATISTICS file filed above; a deltas key those runs do not cover (a store the
                # statistics file dropped or never carried, a spelling the two files disagree on) is refused by name -- a contrast on a run
                # this revision does not emit would hold the whole revision in the harvester, never land late
                absent = [x for x in ((rid, key), (a[0], f"{view}/none")) if x[0] not in runs]
                if absent:
                    self.refusals.add(f"{key}|contrast", "the deltas file names a store whose climb run the statistics file does not file in this read ("
                                           + ", ".join(f"{k} -> {r_[:40]}…" for r_, k in absent)
                                           + "); the vs-unsteered contrast has no run of this revision to attach to", deltas_in.uri)
                    n_deltas_refused += 1
                    continue
                jid = runs[rid]["_jid"]
                batch.add("run_contrast", {"run_id": rid, "anchor_run_id": a[0], "judgment_id": jid, "metric_id": "asr_bon_cap4096", "threshold": 0.5, "computation_id": "",
                                           "anchor_judgment_id": "", "anchor_kind": "same_standup_none", "delta": d.get("delta"), "delta_lo": d.get("delta_lo"),
                                           "delta_hi": d.get("delta_hi"), "delta_p": d.get("delta_p"), "delta_method": str(d.get("delta_method") or "paired vs the view's unsteered cell at its final rung"),
                                           "n_pairs": LC.int_or_none(d.get("n_goals")) or 0, "computed_by": "producer", "computed_utc": when,
                                           "extra": {"asr_cell": d.get("asr_cell"), "asr_unsteered": d.get("asr_unsteered"), "discordant_base1_cell0": d.get("discordant_base1_cell0"),
                                                     "discordant_base0_cell1": d.get("discordant_base0_cell1"), "at_cap_share": d.get("at_cap_share"), "source_uri": f"{src} ›{key}"}},
                          producer, now)
                n_contrasts += 1
        rulings = {p.get("status_ruling_id") for p in runs.values()} | {p["extra"].get("superseded_instrument_ruling_id") for p in runs.values()}
        for rul in sorted(r for r in rulings if r and r.startswith("ruling:r2:supersede:instrument:")):
            seed_id = rul.rsplit(":", 1)[1]
            inst = seeds.instruments.get(seed_id) or {}
            batch.add("ruling", {"ruling_id": rul, "round_id": LC.ROUND_ID, "action": "supersede", "scope_kind": "key", "scope_json": {"instrument_seed_id": seed_id},
                                 "target_json": {"superseded_by": CLIMB_OF.get(seed_id, INST_CLIMB)}, "reason": str(inst.get("paper_status_ruling_id") or "superseded instrument (instruments.yaml paper_status)"),
                                 "ruled_utc": now, "ruled_utc_precision": "date", "ruled_by": "the user (instruments.yaml paper_status_ruling_id)",
                                 "source_ref": "configs/registry/instruments.yaml; plan §2.6; PR-7's rulings seed adopts this id", "extra": {"placeholder_for_pr7": True}}, producer, now)
        for p in runs.values():
            p.pop("_jid", None)
        checked = self._check_children_name_emitted_runs(batch, runs, stats_in.uri if stats_in is not None else (deltas_in.uri if deltas_in is not None else self.rs))
        self.stats.update(runs=len(runs), metrics=n_metrics, degeneration=n_deg, contrasts=n_contrasts, row_refusals=len(self.refusals),
                          deltas_keys_refused=n_deltas_refused, cells=len((stats.get("cells") or {})), registered_points=len({k[0] for k in climbs}),
                          relu_paired_512_runs_referenced=len(set(referenced)), degeneration_rows_dropped_unemitted_runs=n_deg_dropped,
                          metric_rows_dropped_unemitted_runs=n_metric_dropped, readings_dropped_unemitted_runs=dropped,
                          run_child_records_checked=checked, run_child_records_orphaned=0,
                          curve_rows=n_curve, curve_grids_absent=list(self._curve_absent))
        yield batch

    def _check_children_name_emitted_runs(self, batch: RecordBatch, runs: Dict[str, Dict[str, Any]], uri: str) -> int:
        """Every run-child record of the batch (:attr:`RUN_CHILD_TYPES`) names a run THIS batch emits -- its `run_id`, a contrast's
        `anchor_run_id`, and a non-empty `judgment_id` / `anchor_judgment_id` a judgment of the batch -- else the revision is REFUSED by name
        with every offending record: a child whose parent no revision of this reader files is one the harvester holds the whole revision for
        (`activation pending`), so nothing of the read would ever land. Returns the number of records checked."""
        judgments = {str(rec["payload"].get("judgment_id")) for rec in batch.records if rec["record_type"] == "judgment"}
        orphans: List[str] = []
        n = 0
        for rec in batch.records:
            rt = rec["record_type"]
            if rt not in self.RUN_CHILD_TYPES:
                continue
            n += 1
            p = rec["payload"]
            where = str(p.get("source_uri") or (p.get("extra") or {}).get("source_uri") or p.get("store_uri") or p.get("evidence_uri") or "")
            for col in ("run_id", "anchor_run_id"):
                rid = p.get(col)
                if rid is not None and str(rid) not in runs:
                    orphans.append(f"{rt}.{col} {rid}" + (f" [{where}]" if where else ""))
            for col in ("judgment_id", "anchor_judgment_id"):
                jid = p.get(col)
                if jid and str(jid) not in judgments:
                    orphans.append(f"{rt}.{col} {jid} (run {p.get('run_id')})" + (f" [{where}]" if where else ""))
        if orphans:
            self.stats.update(run_child_records_checked=n, run_child_records_orphaned=len(orphans))
            shown = orphans[:20]
            raise AdapterRefusal(uri, f"{len(orphans)} record(s) name a run or judgment this revision does not emit -- the harvester would hold the "
                                      f"revision on the first missing parent and nothing of the read would land: " + "; ".join(shown)
                                      + (f"; … {len(orphans) - len(shown)} more" if len(orphans) > len(shown) else ""))
        return n


def factory() -> BonStoresAdapter:
    return BonStoresAdapter()


register(NAME, factory)
