"""The ``paired_allmodels`` GSM8K document -> registry (source ``gsm8k_paired_allmodels``; a second source of the GSM8K reader).

WHAT IS RUN (2026-09-22 04:24Z, for the all-model appendix): the paired-write difference ReLU objects
(the ``relu_paired`` objects, kind ``psr_paired_write`` in the producer's vocabulary — in the registry the ``composed`` objects labelled
``difference_relu``) served on gemma4e4b_it (object ``87863240…``), llama8b_it (``5066333f…``) and nemotron3nano4b (``ad3e1922…``), plus
nemotron3nano4b x the constitution ReLU object (the roster's ``base__constv2-mix-full__psr_ce_all``, ``81321803…``; its lens-era GSM8K row never
completed). Arrival order: nemotron (both keys), then gemma, then llama. Every row is the appendix column's PER-MODEL instrument: inspect_evals'
stock ``gsm8k`` task as ``inspect/gsm8k_paired200`` over the 200 Platinum-paired items, 10-shot, k = 1, cap 8,192 with the ONE 16,384 step above
1 % truncated, served LoRA-natively, under the DEFAULT Inspect seed rule (one trial per model, no replicate policy — NOT the seed-replicate
document's decode-seed trial rule), the vendor thinking card ON for gemma / nemotron and the no-switch family's ``not_available`` for llama
(``gen_stats.json:thinking`` spells it; the store directory is ``thinking_card`` for a thinking model and ``card`` for llama —
``g8k_cells.SAMPLING_MODE_BY_MODEL``).

THE MIRROR (``run-state/full22/gsm8k_paired_allmodels/``, the reader's input root; configurable like the seed-replicate roots): the cluster's
``exp/gsm8k_paired200_paper_objects_paired_allmodels/controls_paired_allmodels/`` rsync'd so that ``<root>/<tag>/<object dir>/<sampling>/`` is the
BASE store (cap 8,192) and ``<root>/<tag>/<object dir>/<sampling>__rung16384/`` its CEILING store (cap 16,384, present iff the ladder escalated);
the mirror may keep the ``controls_paired_allmodels/`` head under the root (both spellings are read). ``<tag>`` is the producer's standup spelling
— ``gemma4e4b_it`` / ``llama8b_it`` (the ``it`` panel) and ``nemotron3nano4b`` (the ``main`` panel), the model key plus the panel's suffix
(:data:`DOCUMENT`); ``<object dir>`` the producer's slug (``constv2-mix-full--psr_ce_all_minus_ctl``: ``--`` for the harness key's ``__``);
``<sampling>`` ``thinking_card`` or ``card`` per model (:data:`SAMPLING_DIR_BY_THINKING`). The store shape is the seed-replicate document's, byte
for byte the same files (:mod:`gsm8k_seedrep`): ``cells/<store key>/{manifest.json, gen_stats.json, rows.jsonl, outcomes/inspect__gsm8k_paired200.json}``,
``cap_ladder.json`` (base only), ``ladder_state.json``, ``RESULT``, the evidence files as artifacts; optionally the producer's
``<sampling>/<object dir>/controls_summary.json`` and a ``PROVENANCE.md`` beside the object dir. No store pins and no producer count record sit
beside these stores: the tree is attested by ``ladder_state.json``'s ACCEPTED attempt (both stores; ``RESULT``'s last line must read
``result=COMPLETE``), the rung's ``item_subset.source_identity_hash`` must be the base store's ``harness_config_hash``, and the final state is the
reader's item-wise merge alone, checked against ``gen_stats.json`` and ``controls_summary.json`` when present — the seed-replicate reader's
default-seed target precedent (the Arditi all-positions point, ruling ``r2v3-include-gsm8k-arditi-allpos-default-seed-point-own-instrument``).

THE OBJECT is attested THREE ways by the store — ``manifest.json:fit_provenance.object_sha`` (a 16-hex prefix), the rows' aggregate
``object_artifact_sha256`` and ``manifest.json:provenance.artifact_sha256`` (64-hex, both required, equal) — and the attested sha must then be
(a) a row of the connected registry's ``object`` table whose kind / label are the key's grammar (``composed`` / ``difference_relu`` for the paired
write, ``relu_steer`` / ``constitution_relu`` for the constitution key; :func:`lineage_common.cell_grammar_legacy`) — an absent row REFUSES the pair
by name (the four are present and ``approved`` live, verified 2026-09-22; this reader consults the registry, unlike the seed-replicate reader,
because the composed objects are staged in no roster yet: the ``relu_paired`` staging manifests say ``configs_status: PENDING``), (b) the roster / pins' sha
when the roster's ``targets.<model key>.views.<panel>.cells`` stages the key (the constitution key on nemotron: ``jobplan.object_id_for``; a
staged key whose pin disagrees refuses), and (c) the ``relu_paired`` staging manifest's sha when one is on this host (``$STAGE/sdfbase/relu_paired/
MANIFEST_paired_d1_<model key>.json``, schema ``relu_paired_manifest/1``: ``object.cell_id`` the key, ``object.kind psr_paired_write``,
``object.staged_id`` ``paired_eb<B>__…``, the fit / served model pins) — a disagreement refuses by name, an absent manifest is recorded. The
``transfer_kind`` is derived from the manifest's ``fit_provenance`` (the fit checkpoint's HF id -> the fit model; ``transfer`` must equal
fit != served) through :func:`derive.transfer_kind`, never from the standup spelling: the paired writes are fit on the BASE checkpoints and served
on the instruct ones (``base_to_instruct``; the ``it`` spelling alone would read ``instruct_direct``), nemotron's is one chat checkpoint
(``chat_checkpoint``), the same spellings relu_paired files on the same objects.

A ``none`` object dir is REFUSED BY NAME: the document declares no unsteered run (the panel pools the in-frame anchors), and an unsteered run filed
here would enter another source's collapse key. Any other undeclared key, any tag outside the document, a store under the other model's sampling
directory, a replicate seed block, an item count other than the item set's 200 (unless the base manifest declares an ``item_subset``, whose ``n``
is then the count), a ceiling store without its base, and every disagreement between the store's files refuse the pair by name
(:class:`lineage_common.RowRefusals`; the tag's other pairs file).

THE PRODUCER'S RECORDS beside the stores at the MIRROR ROOT (never under the controls head; the mirror of 2026-09-22 carries all three) are bound
field by field WHEN PRESENT: ``store_pins_paired_allmodels.json`` (``g8k_store_pins/2``: ``<model key>/<panel>/<object key>@<cap>`` -> tree; the base
manifest's tree must be its ``@8192`` pin, a ceiling store present iff its ``@16384`` pin is and at it; ``_meta.refused`` names a leg the driver refused,
which may carry no completed store), ``three_numbers_paired_allmodels.json`` (``g8k_three_numbers_offtask/1``: ``rows[<tag>/<object dir>]`` — the
value, the bound, the third state, the base and ceiling readings, the object and the role, every one re-derived and compared; the off-task share is
the producer's own reading, carried on ``extra.three_numbers.producer_off_task``) and ``final_state_paired_allmodels.json`` (the seed-replicate
producer's per-pair shape, the shared cross-check). A disagreement refuses the pair by name, a present record of another schema refuses the tag,
an absent record is recorded (``producer_record_version`` says which was read; the bound row then carries the producer's spelling
``bound_correct_over_attempted``).

A LEG THE DRIVER REFUSED files as a REFUSAL STATE, never as nothing: a store whose ``RESULT`` ends
``result=DRIVER_FAILED`` and whose ``ladder_state.json`` accepts no attempt has no run, no metric and no item — it files ONE ``disposition`` row
(``state not_served``, the schema's vocabulary) whose ``reason`` and ``extra.refusal_state`` spell WHICH refusal: ``thinking_unservable_predicate_a``
when every ON-arm render-probe draw ended inside the prompt-opened think block (``evidence.json:render_probe.on_attempts[*].class == unclosed_block``,
``render_certified.think_shape_certified false``; :data:`PREDICATE_A_RULE` — nemotron3nano4b x the constitution ReLU, RESULT ``rc=2 driver_rc=1
result=DRIVER_FAILED``, 3/3 draws), else ``driver_failed_unclassified``. The row is bound to the model and to the manifest's attested object (a row
of the registry's ``object`` table when the connection carries it; else the disposition names no object and says why) and carries the RESULT line
and path, the failed attempt (job, node, sha, times), the evidence path with every draw's class and the certification problems, the store pins'
``_meta.refused`` entry and the declaration row when present. Its id is minted from (source, model, tag, key, refusal state) — the same under any
mirror root. A completed pin for a refused leg, a refused entry whose RESULT is not the store's, or a ``ladder_state.json`` that accepts an attempt
beside a DRIVER_FAILED RESULT refuses by name.

THE DECLARATION (the record ``configs/sdfbase/gsm8k/paired_allmodels_rows.json``, schema ``g8k_paired_allmodels_rows/1``, role
``paired_allmodels_value``; :data:`DECLARATION_REL`) is bound field by field WHEN PRESENT in the checkout — exactly one row per (tag, key): the
model key and panel, the out roots the mirror's directories spell, the OOS tuples per cap the manifests' ``oos_experiment`` (where the row carries
them; a no-switch fitted row carries none), the thinking mode, the sampling mode, the object sha and staged id when named — a disagreement refuses
by name; while the record is absent the run says so (``extra.declaration None``). The role rides every run and metric row (``extra.declared_role``
/ ``extra.role``: ``paired_allmodels_value``).

REGISTRY GRAIN, THE THREE NUMBERS, THE THIRD ITEM STATE: as the seed-replicate document (the completion-cap rule, applied item-wise).
ONE run per (tag, object key) at the ladder's final state — the ceiling store's items REPLACE the base store's item by item, never a second run,
never a sum of aggregates — ``run_id = identity.run_id(sdfbase_full22, <tag>/<the BASE store key>, <object key>, gsm8k, <the run's instrument>)``
(:data:`PRODUCER_KEY_RULE`: the tag prefixes the store's own NAME so two models' stores can never share a producer key), ``token_cap_tokens`` the
final cap, ``token_cap_rung_index`` its index in ``[8192, 16384]``, ``extra.escalation_state`` ``final_at_ceiling`` / ``final_at_base``. THE VALUE is
``run_metric acc`` = correct / SCORED at the final state under the reader's merge computation; THE BOUND a second ``acc`` row = correct / 200
attempted (blank ``computation_id``, ``computed_by producer`` — the producer's own counts, no producer spelling since no count record exists) with
the unresolved items' interval; THE THIRD STATE an item at the cap with no answer (``at_cap_unresolved`` with its class ``loop`` / ``ceiling`` /
``not_escalated``), the run-level shares as ``run_degeneration`` measures (``at_token_cap_finish_reason`` / ``rep_ratio_loop`` / ``truncation_only``)
— every count, label and measure the seed-replicate reader's (:meth:`gsm8k_seedrep.Gsm8kSeedrepAdapter._merge_items`, shared).

THE INSTRUMENT is the appendix column's PER-MODEL instrument: the canonical seed ``gsm8k_paired200_inspect_10shot_thinking_on_8192`` bound to the
model — expanded per model (gemma / nemotron thinking ON, llama ``not_available``, each model's shipped decode row), so the three models sit
on three instrument ids and a model's two keys (nemotron) on ONE. Nothing is minted: the harness configuration hash differs per (object, cap) store
(the live seed-replicate keys: ``…__d02ab83e`` for ``none``, ``…__d6f67623`` for the LoRA on one tag), so a mint with it would give every key its own
instrument and the panel could pool nothing; the all-positions default-seed point minted only to stay off the instrument the Qwen seed TRIALS sit
on, and no replicate row of these three models sits on their canonical rows. The default seed rule files on that row the way the seed-replicate
reader documents it: ``decode_seed_rule per_item`` with ``seed_base`` NULL, ``trial_index`` 0, ``extra.decode_seed_policy default``
(:data:`gsm8k_seedrep.DEFAULT_SEED_TRIAL_RULE`); the store's own ``harness_config_hash`` / window / versions ride the RUN (``run.harness_config_hash``,
``context_window_tokens``, ``harness_version``). The sent numbers are compared to the row's decode columns (``instrument_match``) and to the
model's registered policy row at ITS thinking mode (``gemma4e4b_it|gsm8k|on|thinking_card``, ``nemotron3nano4b|gsm8k|on|thinking_card``,
``llama8b_it|gsm8k|not_available|card``).

THE ANCHORS: the document runs no ``none``, so NO ``run_contrast`` is filed (``run_contrast.anchor_run_id`` is NOT NULL and this source carries no
anchor run). Every metric row parks the contrast block instead (``extra.delta_block``, the emitter's convention for an anchor another source owns;
em_misalign 7's ``anchor_state`` precedent): ``anchor.anchor_state = pooled_in_frame_anchors``, naming the IN-FRAME unsteered GSM8K run ids of the
model — derived BY IDENTITY from the registry rows at read time (``run.model_id`` = the model, ``eval_id gsm8k``, ``object_id none``, and
``v_paper_current.in_current_paper = 1``; :meth:`Gsm8kPairedAllmodelsAdapter.in_frame_unsteered`), one query per pass for the document's three models,
as an INFORMATIONAL reference: the panel pools them, the reader never picks one as a delta anchor, ``delta`` is null and ``delta_method`` says why.
The block is mirrored on the run (``extra.anchor_state``, ``extra.in_frame_unsteered_reference``); a registry without the paper flag (a scratch root
without the paper views, no connection) parks the block ``unresolved`` naming why — never a refusal.

COEXISTENCE: the ruling ``r2v4-coexist-gsm8k-paired-allmodels-beside-in-frame-gsm8k-rows`` (filed 2026-09-22 04:24Z; match: campaign
``sdfbase_full22``, ``lane_sources_legacy [gsm8k_paired_allmodels]``, the three models, eval ``gsm8k``, ``object_id_prefixes`` 87863240 / 5066333f /
ad3e1922 / 81321803) keeps these rows beside every in-frame GSM8K row of the models; the reader records on each run whether its attested object
carries one of the ruling's prefixes (``extra.coexist_ruling``) and files whatever the store attests — a re-fit object outside the ruling's prefixes
needs a ruling of its own, not a reader default.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import decode_policy as DP
from .. import derive as DV
from .. import identity as I
from .. import jobplan as JP
from .. import vocab as V
from . import _items_common as IC
from . import declared_common as DC
from . import lineage_common as LC
from . import gsm8k_seedrep as G
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register, site_root
from .runner import source_id_for

NAME = "gsm8k_paired_allmodels"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for this document
SOURCE = "gsm8k_paired_allmodels"
OWNER = "Overflow"
PRODUCER_SESSION = "Overflow"
CAMPAIGN = LC.CAMPAIGN
EVAL_ID = G.EVAL_ID
TASK = G.TASK
INSTRUMENT_SEED = G.INSTRUMENT_SEED
CAP, CEILING, LADDER, N_ITEMS = G.CAP, G.CEILING, G.LADDER, G.N_ITEMS
ROOT_DEFAULT = os.path.join(LC.RS_DEFAULT, "gsm8k_paired_allmodels")
#: the cluster's wave root and the controls head the mirror may keep under the reader's root
CLUSTER_ROOT_REL = "exp/gsm8k_paired200_paper_objects_paired_allmodels"
CONTROLS_HEAD = "controls_paired_allmodels"
#: the role every run of the document carries (g8k_cells.ROLE_PAIRED_ALLMODELS)
ROLE = "paired_allmodels_value"
COMPOSED_KEY = "constv2-mix-full__psr_ce_all_minus_ctl"
CONSTITUTION_KEY = "constv2-mix-full__psr_ce_all"
#: THE DOCUMENT: standup tag -> (model key, the roster panel the tag spells, the object keys the document declares on it). A tag outside it, or a
#: key outside its tag's tuple, refuses by name (a fourth model or a fifth key needs a declaration first, then one line here).
DOCUMENT: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "gemma4e4b_it": ("gemma4e4b", "it", (COMPOSED_KEY,)),
    "llama8b_it": ("llama8b", "it", (COMPOSED_KEY,)),
    "nemotron3nano4b": ("nemotron3nano4b", "main", (COMPOSED_KEY, CONSTITUTION_KEY)),
}
#: the store directory per thinking spelling (g8k_cells.SAMPLING_MODE_BY_MODEL: the thinking families' `thinking_card`, the no-switch family's `card`)
SAMPLING_DIR_BY_THINKING = {"on": "thinking_card", "not_available": "card"}
THINKING_NOT_AVAILABLE = "not_available"
STORE_DIR_RE = re.compile(r"^(?P<sampling>thinking_card|card)(?P<ceiling>__rung16384)?$")
#: the coexist ruling over this source (filed 2026-09-22 04:24Z) and the object prefixes it names
COEXIST_RULING_ID = "r2v4-coexist-gsm8k-paired-allmodels-beside-in-frame-gsm8k-rows"
COEXIST_OBJECT_PREFIXES = ("87863240", "5066333f", "ad3e1922", "81321803")
#: the declaration record of the document's rows (optional while it lands; bound field by field when present)
DECLARATION_REL = ("configs", "sdfbase", "gsm8k", "paired_allmodels_rows.json")
DECLARATION_SCHEMA = "g8k_paired_allmodels_rows/1"
#: ReLU's staging manifests of the paired writes (relu_paired.MANIFEST_DIR; machine-local, read when present)
STAGING_DIR_DEFAULT = site_root("$STAGE/sdfbase/relu_paired")
STAGING_MANIFEST_NAME = "MANIFEST_paired_d1_{model_key}.json"
STAGING_SCHEMA = "relu_paired_manifest/1"
STAGING_KIND = "psr_paired_write"
PRODUCER_KEY_RULE = "the producer key is `<tag>/<base store key>`: the standup tag prefixes the store's own NAME, so two models' stores never share a key"
#: the producer's records beside the stores at the MIRROR ROOT (never under the controls head), bound field by field WHEN PRESENT: the store pins
#: (`g8k_store_pins/2`, `<model key>/<panel>/<object key>@<cap>` -> tree, `_meta.refused` naming a leg that never completed), the per-pair final-state
#: record (keyed `<tag>/<object dir>`, the seed-replicate producer's shape, cross-checked by the shared reader) and the three-number record
#: (`g8k_three_numbers_offtask/1`: `rows[<tag>/<object dir>]` — the value, the bound, the third state and the off-task share beside them)
PINS_FILE = "store_pins_paired_allmodels.json"
THREE_NUMBERS_FILE = "three_numbers_paired_allmodels.json"
FINAL_STATE_FILE = "final_state_paired_allmodels.json"
SHARED_FILES = (PINS_FILE, THREE_NUMBERS_FILE, FINAL_STATE_FILE)
PINS_SCHEMA = G.PINS_SCHEMA
THREE_NUMBERS_SCHEMA = "g8k_three_numbers_offtask/1"
RECORD_VERSION_PRODUCER = f"2026-09-22: {THREE_NUMBERS_FILE} ({THREE_NUMBERS_SCHEMA}) and / or {FINAL_STATE_FILE} beside the stores at the mirror root"
#: A LEG THAT NEVER COMPLETED files as a REFUSAL STATE, never as nothing: a `disposition` row (state `not_served`; `reason` and `extra.refusal_state`
#: spell WHICH refusal) for the (tag, key), bound to the model and the attested object. The spellings: `thinking_unservable_predicate_a` — the
#: driver's THINKING_UNSERVABLE predicate A, the steered model ends its turn inside the prompt-opened think block on every render-probe draw
#: (`evidence.json:render_probe.on_attempts[*].class == unclosed_block`, `render_certified.think_shape_certified false`; nemotron3nano4b x the
#: constitution ReLU, 2026-09-22, RESULT `rc=2 driver_rc=1 result=DRIVER_FAILED`); `driver_failed_unclassified` — a DRIVER_FAILED leg whose evidence
#: names no predicate (still a state, never null).
RESULT_DRIVER_FAILED = "result=DRIVER_FAILED"
REFUSAL_STATE_THINKING_UNSERVABLE_A = "thinking_unservable_predicate_a"
REFUSAL_STATE_DRIVER_FAILED = "driver_failed_unclassified"
REFUSAL_DISPOSITION_STATE = "not_served"
PROBE_CLASS_UNCLOSED = "unclosed_block"
PREDICATE_A_RULE = ("THINKING_UNSERVABLE predicate A (served_capability_run): the steered model ends its turn inside the prompt-opened think block on every render-probe "
                    "draw (render_probe.on_attempts[*].class == unclosed_block, finish_reason stop, reasoning absent), so the thinking-ON instrument cannot be served for "
                    "this (model, object); the driver refuses the leg (rc 2, DRIVER_FAILED) and no store is generated")
INSTRUMENT_RULE = ("one instrument per MODEL: the canonical seed row bound to the served model (WS-B0's per-model expansion), nothing minted — the harness configuration hash "
                   "differs per (object, cap) store, so a mint with it would put every key on its own instrument and the panel could pool nothing; the default seed rule "
                   "files on the row as per_item with seed_base NULL (the seed-replicate reader's documented reading of a default-seed store)")
ANCHOR_STATE = "pooled_in_frame_anchors"
ANCHOR_DERIVATION = ("by identity from the registry rows at read time: run.model_id = the served model, eval_id gsm8k, object_id none, and "
                     "v_paper_current.in_current_paper = 1 (the frame's flag); informational — the panel pools these anchors, none is a delta anchor")
NO_CONTRAST_WHY = ("the document runs no `none`: no anchor run of this source exists to point a run_contrast at (anchor_run_id is NOT NULL), and an in-frame "
                   "none of another source is never picked as the delta anchor — the panel pools the in-frame anchors; the block names them as a reference")
OBJECT_ROW_RULE = ("the attested object sha must be a row of the connected registry's object table with the key's grammar kind / label: the composed objects "
                   "are staged in no roster yet (ReLU's staging manifests: configs_status PENDING), so the object row is the registry-side identity")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA16 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def producer_key_for(tag: str, base_store_key: str) -> str:
    return f"{tag}/{base_store_key}"


def run_id_for(tag: str, base_store_key: str, object_key: str, instrument_id: str) -> str:
    return I.run_id(CAMPAIGN, producer_key_for(tag, base_store_key), object_key, EVAL_ID, instrument_id)


def thinking_for(seeds: LC.Seeds, model_id: str) -> str:
    """The model's thinking spelling: ``not_available`` for a no-switch family (models.yaml thinking_channel), else ``on``."""
    return THINKING_NOT_AVAILABLE if seeds.model_thinking_channel(model_id) == THINKING_NOT_AVAILABLE else "on"


def store_dirs_for(thinking: str) -> Tuple[str, str]:
    """``(base store dir, ceiling store dir)`` of a model's stores."""
    d = SAMPLING_DIR_BY_THINKING[thinking]
    return d, f"{d}__rung{CEILING}"


class Gsm8kPairedAllmodelsAdapter(G.Gsm8kSeedrepAdapter):
    """The ``paired_allmodels`` document's reader: the seed-replicate reader's store grammar, item-wise merge and three numbers over another
    mirror root, another tag grammar (the document's standups), per-model thinking / sampling / decode policy, the object attested by the store
    and bound to the registry's object row, and the in-frame unsteered reference parked on every metric (module docstring)."""

    name = NAME
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 2 (2026-09-22): the producer's records at the mirror root (store pins, the off-task three-number record, the final-state record) are bound
    #: when present, and a leg the driver refused files as a REFUSAL STATE (a `not_served` disposition spelling `thinking_unservable_predicate_a`)
    #: instead of a bare pair refusal; the run / metric / item rows of a completed pair read without the records are byte-identical to 1, but the
    #: mirror carries the records, so every row's extra moves (producer_record, store_pins, the bound's spelling) -- the version moves with it.
    #: 1 (2026-09-22): the first version — the paired_allmodels document (module docstring)
    EMISSION_VERSION = 2
    rng_scope = "none"
    allow_symlinks = False
    owner = OWNER

    def __init__(self, root: str = ROOT_DEFAULT, seeds: Optional[LC.Seeds] = None, roster_path: Optional[str] = None, pins_path: Optional[str] = None,
                 policy_path: Optional[str] = None, declaration_path: Optional[str] = None, staging_dir: Optional[str] = STAGING_DIR_DEFAULT):
        super().__init__(root=root, seeds=seeds, roster_path=roster_path, pins_path=pins_path, policy_path=policy_path)
        self.declaration_path = declaration_path or DC.repo_path(*DECLARATION_REL)
        self.staging_dir = LC.expand(staging_dir) if staging_dir else None
        self._frame_cache: Dict[str, Dict[str, Any]] = {}
        self._declaration: Optional[Tuple[Optional[Dict[str, Any]], Optional[str]]] = None

    # ---- discovery ----------------------------------------------------------------------------
    def _tags_root(self) -> str:
        head = os.path.join(self.root, CONTROLS_HEAD)
        return head if os.path.isdir(head) else self.root

    def discover(self) -> List[Input]:
        """Named files of every ``<tag>/<object dir>/<store dir>/`` under the root (or under its ``controls_paired_allmodels/`` head), one directory
        level at a time; the producer's ``controls_summary.json`` under the base store and a ``PROVENANCE.md`` beside the object dir when present.
        An absent root is a SKIP (the mirror is machine-local); a root holding no store refuses. Every store-shaped directory is listed — an
        undeclared tag or key is refused BY NAME at read time, never silently skipped."""
        out: List[Input] = []
        if not os.path.isdir(self.root):
            return out
        troot = self._tags_root()
        for tag in IC.listdir(troot):
            td = os.path.join(troot, tag)
            if not os.path.isdir(td) or tag == CONTROLS_HEAD:
                continue
            for slug in IC.listdir(td):
                od = os.path.join(td, slug)
                if not os.path.isdir(od):
                    continue
                n_before = len(out)
                for store_dir in IC.listdir(od):
                    m = STORE_DIR_RE.match(store_dir)
                    d = os.path.join(od, store_dir)
                    if m and os.path.isdir(d):
                        out += self._discover_store(d, summary_slug=(slug if not m.group("ceiling") else None))
                p = os.path.join(od, G.TARGET_PROVENANCE_FILE)
                if len(out) > n_before and os.path.isfile(p):
                    out.append(check_input(p, role="log", allow_symlink=self.allow_symlinks))
        if not out:
            raise AdapterRefusal(self.root, f"no GSM8K store found (<tag>/<object dir>/<thinking_card|card>[__rung{CEILING}]/cells/<store key>/ under the root"
                                            f" or its {CONTROLS_HEAD}/ head)")
        # the producer's records at the MIRROR ROOT, aliased per tag (`#<tag>`) so every group's read sees them (the seed-replicate reader's convention)
        tags = sorted({self.comparison_group(i) for i in out})
        for name in SHARED_FILES:
            p = os.path.join(self.root, name)
            if os.path.isfile(p):
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(t) for t in tags]
        return out

    def comparison_group(self, inp: Input) -> str:
        """One group per standup TAG (every object key, base and ceiling store of the tag); a shared record rides a ``#<tag>`` alias."""
        base = os.path.basename(inp.uri)
        if "#" in base and base.rsplit("#", 1)[0] in SHARED_FILES:
            return base.rsplit("#", 1)[1]
        rel = os.path.relpath(inp.uri, self._tags_root()).split(os.sep)
        if len(rel) >= 3 and rel[0] not in ("..", ""):
            return rel[0]
        raise AdapterRefusal(inp.uri, f"input is not under <tag>/<object dir>/ of {self._tags_root()}")

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        for inp in inputs:
            base = os.path.basename(inp.uri).rsplit("#", 1)[0]
            if base == PINS_FILE and "#" in os.path.basename(inp.uri):
                d = self._json(inp) or {}
                group = self.comparison_group(inp)
                doc = DOCUMENT.get(group)
                head = f"{doc[0]}/{doc[1]}/" if doc else f"{group}/"
                members = sorted(k for k in d if k != "_meta" and str(k).startswith(head))
                members += sorted(f"refused:{k}" for k in ((d.get("_meta") or {}).get("refused") or {}) if str(k).startswith(head))
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.pins", 1, 1, members, inp.sha256))
        if inputs and self.refusals.items:
            rows += self.refusals.source_state_rows(source_id_for(NAME, self.comparison_group(inputs[0])), inputs[0].uri, inputs[0].sha256)
        return rows

    def status_sources(self, group: str) -> Tuple[str, ...]:
        return (SOURCE,)

    # ---- the in-frame unsteered reference (by identity, from the registry rows) ---------------
    def in_frame_unsteered(self, conn: Any, now: str) -> Dict[str, Dict[str, Any]]:
        """Per document model: the unsteered GSM8K runs of the model in the connected registry (``run`` by identity) and, of those, the ones the
        frame flags in the current paper (``v_paper_current.in_current_paper``). ONE pass over the three models, cached on the instance (the frame
        flag's view materialises the roster match: ~90 s live, so it is paid once per pass, and only when a group is (re)read). A registry without
        the flag view, or no connection, leaves the reference ``unresolved`` naming why — never a refusal."""
        if self._frame_cache:
            return self._frame_cache
        models = sorted({self.seeds.model_for_view_legacy(t).get("model_id") or "" for t in DOCUMENT} - {""})
        out: Dict[str, Dict[str, Any]] = {}
        base = {"anchor_state": ANCHOR_STATE, "derivation": ANCHOR_DERIVATION, "derived_at": now, "object_id": "none", "eval_id": EVAL_ID}
        if conn is None:
            for m in models:
                out[m] = dict(base, model_id=m, state="unresolved", why="no registry connection: the frame's unsteered runs cannot be named", in_frame_unsteered_run_ids=[],
                              anchors=[], n_candidates=None)
            self._frame_cache = out
            return out
        cands: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
        try:
            for m in models:
                for r in conn.execute("SELECT run_id, view_legacy, lane_source_legacy, instrument_id, token_cap_tokens, thinking_mode, lifecycle_state FROM run "
                                      "WHERE model_id = ? AND object_id = 'none' AND eval_id = ? ORDER BY run_id", (m, EVAL_ID)):
                    cands[m].append({"run_id": r[0], "view_legacy": r[1], "lane_source_legacy": r[2], "instrument_id": r[3], "token_cap_tokens": r[4],
                                     "thinking_mode": r[5], "lifecycle_state": r[6]})
        except sqlite3.Error as exc:
            for m in models:
                out[m] = dict(base, model_id=m, state="unresolved", why=f"the connected registry's run table could not be read: {exc}", in_frame_unsteered_run_ids=[],
                              anchors=[], n_candidates=None)
            self._frame_cache = out
            return out
        ids = [c["run_id"] for m in models for c in cands[m]]
        flags: Dict[str, Tuple[Any, Any]] = {}
        flag_why: Optional[str] = None
        if ids:
            try:
                q = "SELECT run_id, in_current_paper, selection_state FROM v_paper_current WHERE run_id IN (%s)" % ",".join("?" * len(ids))
                for r in conn.execute(q, ids):
                    flags[str(r[0])] = (r[1], r[2])
            except sqlite3.Error as exc:
                flag_why = f"the connected registry has no readable paper flag (v_paper_current): {exc}"
        for m in models:
            if flag_why is not None:
                out[m] = dict(base, model_id=m, state="unresolved", why=flag_why, in_frame_unsteered_run_ids=[], anchors=[], n_candidates=len(cands[m]),
                              candidates=[c["run_id"] for c in cands[m]])
                continue
            anchors = [dict(c, selection_state=flags.get(c["run_id"], (None, None))[1]) for c in cands[m] if flags.get(c["run_id"], (0, None))[0] == 1]
            out[m] = dict(base, model_id=m, state="derived", why=(None if anchors else f"no in-frame unsteered GSM8K run of {m} in the connected registry"),
                          in_frame_unsteered_run_ids=[a["run_id"] for a in anchors], anchors=anchors, n_candidates=len(cands[m]))
        self._frame_cache = out
        return out

    @staticmethod
    def delta_block(ref: Mapping[str, Any], model_id: str) -> Dict[str, Any]:
        """The parked contrast block of a metric row: the in-frame anchors bound by identity, no delta (module docstring, THE ANCHORS)."""
        anchor = {"kind_legacy": None, "anchor_state": ANCHOR_STATE, "anchor_run_id": None, "anchor_judgment_id": None, "object_id": "none", "campaign": CAMPAIGN,
                  "model_id": model_id, "eval_id": EVAL_ID, "state": ref.get("state"), "why": ref.get("why"), "derivation": ref.get("derivation"),
                  "derived_at": ref.get("derived_at"), "in_frame_unsteered_run_ids": list(ref.get("in_frame_unsteered_run_ids") or []),
                  "anchors": list(ref.get("anchors") or []), "n_candidates": ref.get("n_candidates"),
                  "note": "informational: the panel pools the in-frame unsteered anchors of the model; the reader never picks one as the delta anchor"}
        return {"anchor": anchor, "paired": False, "n_pairs": None, "delta": None, "delta_lo": None, "delta_hi": None, "delta_p": None, "effect_size": None,
                "effect_kind": None, "delta_method": f"not computed: {ANCHOR_STATE}", "why_no_run_contrast": NO_CONTRAST_WHY,
                "computed_by": f"registry ({NAME} reader): the binding only; no delta"}

    # ---- the declaration record (optional while it lands) -------------------------------------
    def declaration(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """``(the record, None)`` when the checkout carries the declaration record, ``(None, why)`` when it is absent; a present record with another
        schema or role is a refusal of every pair (raised as :class:`_items_common.Unresolved` by :meth:`declaration_row`)."""
        if self._declaration is None:
            if not os.path.isfile(self.declaration_path):
                self._declaration = (None, f"the declaration record {LC.rel_home(self.declaration_path)} is not in this checkout (Overflow's record lands with the wave)")
            else:
                with open(self.declaration_path, "rb") as fh:
                    raw = fh.read()
                try:
                    d = json.loads(raw.decode("utf-8"))
                except ValueError as exc:
                    d = {"_error": f"not JSON: {exc}"}
                if not isinstance(d, Mapping):
                    d = {"_error": "not a mapping"}
                self._declaration = (dict(d, _path=LC.rel_home(self.declaration_path), _sha256=I.sha256_hex(raw)), None)
        return self._declaration

    def declaration_row(self, tag: str, okey: str) -> Optional[Dict[str, Any]]:
        """The record's ONE row for (tag, key), or None when the record is absent; a present record that is malformed, carries another schema or
        role, or names no / several rows for the pair raises :class:`_items_common.Unresolved`."""
        d, _why = self.declaration()
        if d is None:
            return None
        path = d["_path"]
        if d.get("_error"):
            raise IC.Unresolved(path, f"the declaration record is {d['_error']}")
        if str(d.get("schema") or "") != DECLARATION_SCHEMA:
            raise IC.Unresolved(path, f"the declaration record carries schema {d.get('schema')!r}, not {DECLARATION_SCHEMA}")
        if str(d.get("role") or "") != ROLE:
            raise IC.Unresolved(path, f"the declaration record declares role {d.get('role')!r}, not {ROLE!r}; another role needs its own reading")
        rows = [r for r in (d.get("rows") or []) if isinstance(r, Mapping) and str(r.get("tag") or "") == tag and str(r.get("cell") or "") == okey]
        if len(rows) != 1:
            raise IC.Unresolved(path, f"the declaration record names {len(rows)} row(s) for tag {tag} / {okey} (expected exactly one)")
        return dict(rows[0], _path=path, _schema=DECLARATION_SCHEMA, _sha256=d["_sha256"], _role=ROLE)

    @staticmethod
    def _check_declaration(decl: Mapping[str, Any], tag: str, model_key: str, panel: str, slug: str, sampling_dir: str, thinking: str, base: Mapping[str, Any],
                           ceil: Optional[Mapping[str, Any]], attested: str, staged_id: Optional[str], block: Mapping[str, Any]) -> List[str]:
        bad: List[str] = []
        if str(decl.get("model_key") or "") != model_key:
            bad.append(f"model_key {decl.get('model_key')!r} is not {model_key!r}")
        for key in ("view", "panel_view"):
            if decl.get(key) is not None and str(decl.get(key)) != panel:
                bad.append(f"{key} {decl.get(key)!r} is not the tag's panel {panel!r}")
        base_dir, ceil_dir = sampling_dir, f"{sampling_dir}__rung{CEILING}"
        for key, store_dir, present in (("out_root", base_dir, True), ("rung_out_root", ceil_dir, ceil is not None)):
            want_tail = os.path.join(tag, slug, store_dir)
            got = str(decl.get(key) or "")
            if not got:
                if present:
                    bad.append(f"{key} is not declared")
            elif not (got == want_tail or got.endswith(os.sep + want_tail)):
                bad.append(f"{key} {got!r} does not spell this store's directory (…/{want_tail})")
        tuples = decl.get("tuple_names") if isinstance(decl.get("tuple_names"), Mapping) else {}
        if tuples:
            for cap, m in ((CAP, base["manifest"]), (CEILING, ceil["manifest"] if ceil is not None else None)):
                if m is None:
                    continue
                want_t, got_t = tuples.get(str(cap)), m.get("oos_experiment")
                if want_t is not None and str(got_t or "") != str(want_t):
                    bad.append(f"the manifest at {cap:,} names oos_experiment {got_t!r}, the declaration's tuple_names[{cap}] {want_t!r}")
        if decl.get("thinking_mode") is not None and str(decl.get("thinking_mode")) != thinking:
            bad.append(f"thinking_mode {decl.get('thinking_mode')!r} is not the model's {thinking!r}")
        if decl.get("sampling_mode") is not None and str(decl.get("sampling_mode")) != sampling_dir:
            bad.append(f"sampling_mode {decl.get('sampling_mode')!r} is not the store directory {sampling_dir!r}")
        if decl.get("object_sha256") is not None and str(decl.get("object_sha256")) != attested:
            bad.append(f"object_sha256 {str(decl.get('object_sha256'))[:16]}… is not the store's attested object {attested[:16]}…")
        if decl.get("staged_id") is not None and staged_id is not None and str(decl.get("staged_id")) != staged_id:
            bad.append(f"staged_id {decl.get('staged_id')!r} is not the staging's {staged_id!r}")
        if decl.get("fitted") is not None and decl.get("fitted") is not True:
            bad.append(f"fitted {decl.get('fitted')!r} is not true (a paired write / a ReLU fit is a fitted object)")
        sampling = decl.get("sampling") if isinstance(decl.get("sampling"), Mapping) else None
        if sampling is not None and not G.Gsm8kSeedrepAdapter._numbers_agree(block, {k: sampling.get(k) for k in DP.PARAM_KEYS}):
            bad.append(f"the declared sampling row {G.RP._numbers(sampling)} is not the sent block {G.RP._numbers(block)}")
        return bad

    # ---- the staging manifest (ReLU's, machine-local; read when present) ----------------------
    def staging_manifest(self, model_key: str, okey: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """ReLU's ``MANIFEST_paired_d1_<model key>.json`` object block when it is on this host and names this key: ``(block + _path / _sha256, None)``;
        ``(None, why)`` when absent or naming another key (a constitution ReLU has no paired-write manifest)."""
        if not self.staging_dir:
            return None, "no staging directory configured"
        p = os.path.join(self.staging_dir, STAGING_MANIFEST_NAME.format(model_key=model_key))
        if not os.path.isfile(p):
            return None, f"no staging manifest {LC.rel_home(p)} on this host"
        with open(p, "rb") as fh:
            raw = fh.read()
        try:
            d = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise IC.Unresolved(p, f"the staging manifest is not JSON: {exc}") from None
        if not isinstance(d, Mapping) or str(d.get("_schema") or "") != STAGING_SCHEMA or not isinstance(d.get("object"), Mapping):
            raise IC.Unresolved(p, f"the staging manifest carries schema {(d or {}).get('_schema') if isinstance(d, Mapping) else None!r}, not {STAGING_SCHEMA}, or no object block")
        o = dict(d["object"])
        if str(o.get("cell_id") or "") != okey:
            return None, f"the staging manifest {LC.rel_home(p)} stages {o.get('cell_id')!r}, not {okey}"
        o.update(_path=LC.rel_home(p), _sha256=I.sha256_hex(raw), _written_at=d.get("_written_at"))
        return o, None

    # ---- the object: attested by the store, bound to the registry's object row ----------------
    def resolve_attested_object(self, conn: Any, okey: str, model_key: str, model_id: str, panel: str, base: Mapping[str, Any]) -> Tuple[LC.ResolvedObject, Dict[str, Any]]:
        """The object the store attests three ways, checked against the registry's ``object`` row (required), the roster / pins (when the panel
        stages the key) and ReLU's staging manifest (when present); returns the resolved object and the attestation facts. Raises
        :class:`gsm8k_seedrep._PairRefusal`."""
        man = base["manifest"]
        where = base["files"]["manifest.json"].uri
        g = LC.cell_grammar_legacy(okey)
        kind = g["object_kind"]
        if kind in (None, "none", "system_instruction"):
            raise G._PairRefusal(f"object key {okey!r} is not a fitted object's key", where)
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        sha16 = str(fitp.get("object_sha") or "")
        attested = str(base["agg_extra"].get("object_artifact_sha256") or "")
        prov_sha = str((man.get("provenance") or {}).get("artifact_sha256") or "") if isinstance(man.get("provenance"), Mapping) else ""
        if not _SHA256.match(attested):
            raise G._PairRefusal(f"the rows' aggregate attests no 64-hex object_artifact_sha256 ({attested[:16]!r}); the served object is unattested", base["files"]["rows.jsonl"].uri)
        if not _SHA256.match(prov_sha):
            raise G._PairRefusal(f"manifest.json:provenance.artifact_sha256 {prov_sha[:16]!r} names no 64-hex object", where)
        if prov_sha != attested:
            raise G._PairRefusal(f"manifest.json:provenance.artifact_sha256 {prov_sha[:16]}… and the rows' aggregate {attested[:16]}… name two objects", where)
        if not _SHA16.match(sha16) or not attested.startswith(sha16):
            raise G._PairRefusal(f"manifest.json:fit_provenance.object_sha {sha16!r} is no prefix of the attested object {attested[:16]}…", where)
        facts: Dict[str, Any] = {"attested_sha256": attested, "attested_by": ["rows.jsonl aggregate object_artifact_sha256", "manifest.json:provenance.artifact_sha256",
                                                                            "manifest.json:fit_provenance.object_sha (16-hex prefix)"]}
        # (a) the registry's object row: REQUIRED (module docstring, THE OBJECT)
        if conn is None:
            raise G._PairRefusal(f"object {attested[:16]}…: no registry connection, so the object row cannot be verified ({OBJECT_ROW_RULE})", where)
        try:
            row = conn.execute("SELECT object_kind, paper_label, object_role, fit_model_id, derivation, status, campaign_cell_id_legacy FROM object WHERE object_id = ?",
                               (attested,)).fetchone()
        except sqlite3.Error as exc:
            raise G._PairRefusal(f"object {attested[:16]}…: the connected registry's object table could not be read ({exc})", where)
        if row is None:
            raise G._PairRefusal(f"object {attested[:16]}… is not in the registry's object table ({OBJECT_ROW_RULE})", where)
        o_kind, o_label, o_role, o_fit, o_deriv, o_status, o_staged = (str(row[0] or ""), str(row[1] or ""), str(row[2] or ""), row[3], row[4], row[5], row[6])
        if o_kind != kind or o_label != g["paper_label"]:
            raise G._PairRefusal(f"the object row of {attested[:16]}… is kind {o_kind!r} / label {o_label!r}; the key {okey} spells {kind!r} / {g['paper_label']!r}", where)
        facts["object_row"] = {"object_kind": o_kind, "paper_label": o_label, "object_role": o_role, "fit_model_id": o_fit, "derivation": o_deriv, "status": o_status,
                               "campaign_cell_id_legacy": o_staged}
        staged_id: Optional[str] = str(o_staged) if o_staged else None
        sources = ["the registry's object row"]
        # (b) the roster / pins, when the panel stages the key
        staged_roster = None
        try:
            roster, pins = self._roster_pins()
            staged_roster = ((((roster.get("targets") or {}).get(model_key) or {}).get("views") or {}).get(panel) or {}).get("cells", {}).get(okey)
        except IC.Unresolved:
            roster, pins = None, None
        if staged_roster:
            try:
                sha = JP.object_id_for(okey, model_key=model_key, roster_selection=panel, roster=roster, pins=pins, si_sha={})
            except LookupError as exc:
                raise G._PairRefusal(f"object: the roster stages {okey} on {model_key}/{panel} as {staged_roster!r} but the pins name no sha for it: {exc}", self.pins_path)
            if sha != attested:
                raise G._PairRefusal(f"the roster / pins name object {sha[:16]}… for {model_key}/{panel}/{okey} (staged {staged_roster}), the store attests {attested[:16]}…", where)
            staged_id = str(staged_roster)
            facts["roster_staging"] = {"staged_id": staged_id, "selection": panel, "pins": LC.rel_home(self.pins_path)}
            sources.append(f"roster targets.{model_key}.views.{panel}.cells[{okey}] -> {staged_id}; pins {model_key}.{staged_id}.object_sha256 (jobplan.object_id_for)")
        else:
            facts["roster_staging"] = None
        # (c) ReLU's staging manifest, when present on this host
        try:
            stg, stg_why = self.staging_manifest(model_key, okey)
        except IC.Unresolved as exc:
            raise G._PairRefusal(f"object: {exc.why}", exc.path)
        if stg is not None:
            if str(stg.get("object_sha256") or "") != attested:
                raise G._PairRefusal(f"ReLU's staging manifest {stg['_path']} stages object {str(stg.get('object_sha256'))[:16]}…, the store attests {attested[:16]}…", where)
            if str(stg.get("kind") or "") != STAGING_KIND:
                raise G._PairRefusal(f"ReLU's staging manifest {stg['_path']} stages kind {stg.get('kind')!r}, not {STAGING_KIND!r}", where)
            served = stg.get("served_model") if isinstance(stg.get("served_model"), Mapping) else {}
            if served.get("id") and self.seeds.model_for_hf_id(str(served["id"])) != model_id:
                raise G._PairRefusal(f"ReLU's staging manifest serves {served.get('id')!r}, not the tag's model {model_id}", where)
            if served.get("revision") and str(man.get("model_revision") or "") and str(served["revision"]) != str(man.get("model_revision")):
                raise G._PairRefusal(f"ReLU's staging manifest pins served revision {str(served['revision'])[:12]}, the store {str(man.get('model_revision'))[:12]}", where)
            fitm = stg.get("fit_model") if isinstance(stg.get("fit_model"), Mapping) else {}
            if fitm.get("id") and fitp.get("model_id") and str(fitm["id"]) != str(fitp.get("model_id")):
                raise G._PairRefusal(f"ReLU's staging manifest fits on {fitm.get('id')!r}, the store's fit_provenance.model_id is {fitp.get('model_id')!r}", where)
            if staged_id is not None and stg.get("staged_id") and str(stg["staged_id"]) != staged_id:
                raise G._PairRefusal(f"ReLU's staging manifest stages {stg.get('staged_id')!r}, the registry / roster {staged_id!r}: one object, two staged spellings", where)
            staged_id = str(stg.get("staged_id") or staged_id or "") or None
            facts["staging_manifest"] = {k: stg.get(k) for k in ("_path", "_sha256", "_written_at", "staged_id", "view", "seed", "kind", "dose", "bank", "source")}
            facts["staging_manifest"]["fit_model"] = dict(fitm) if fitm else None
            facts["staging_manifest"]["served_model"] = dict(served) if served else None
            facts["staging_manifest"]["parents"] = ({k: (v.get("object_sha256") if isinstance(v, Mapping) else None) for k, v in (stg.get("parents") or {}).items()}
                                                    if isinstance(stg.get("parents"), Mapping) else None)
            sources.append(f"ReLU's staging manifest {stg['_path']} (object.object_sha256, kind {STAGING_KIND}, staged {stg.get('staged_id')})")
        else:
            facts["staging_manifest"] = None
            facts["staging_manifest_absent_why"] = stg_why
        facts["coexist_ruling"] = {"ruling_id": COEXIST_RULING_ID, "object_id_prefixes": list(COEXIST_OBJECT_PREFIXES),
                                   "matches_object_prefix": any(attested.startswith(p) for p in COEXIST_OBJECT_PREFIXES)}
        obj = LC.ResolvedObject(attested, kind, g["paper_label"], g["object_role"], "recorded",
                                "attested by the store (" + "; ".join(facts["attested_by"]) + ") and bound to " + "; ".join(sources),
                                staged_id=staged_id, derivation=g["derivation"], si_id=g["si_id"])
        return obj, facts

    # ---- read ---------------------------------------------------------------------------------
    def _sort_paired_stores(self, inputs: Sequence[Input], tag: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Input], Dict[str, Input]]:
        """The tag's inputs sorted into stores ``(object slug, store dir) -> {"top", "keys", "dir", ["summary"]}``, the PROVENANCE.md per slug, and the
        shared records (``#<tag>`` aliases) by file name."""
        stores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        prov: Dict[str, Input] = {}
        shared: Dict[str, Input] = {}
        troot = self._tags_root()
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" in base and base.rsplit("#", 1)[0] in SHARED_FILES:
                shared[base.rsplit("#", 1)[0]] = inp
                continue
            rel = os.path.relpath(inp.uri, troot).split(os.sep)
            if rel[0] != tag or len(rel) < 3:
                continue
            slug = rel[1]
            if len(rel) == 3 and rel[2] == G.TARGET_PROVENANCE_FILE:
                prov[slug] = inp
                continue
            if not STORE_DIR_RE.match(rel[2]):
                continue
            st = stores.setdefault((slug, rel[2]), {"top": {}, "keys": {}, "dir": os.path.join(troot, *rel[:3])})
            body = rel[3:]
            if len(body) == 1:
                st["top"][body[0]] = inp
            elif len(body) == 2 and body[0] == slug and body[1] == G.TARGET_SUMMARY_FILE:
                st["summary"] = inp
            elif len(body) >= 3 and body[0] == "cells":
                files = st["keys"].setdefault(body[1], {})
                if len(body) == 3:
                    files[body[2]] = inp
                elif len(body) == 4 and body[2] == "outcomes":
                    files["outcomes"] = inp
        return stores, prov, shared

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        batch = RecordBatch(source_id_for(NAME, group), group)
        self.refusals = LC.RowRefusals(NAME)
        facts: Dict[str, Any] = {"tag": group, "source": SOURCE, "role": ROLE, "runs": 0, "pairs": {}, "ceiling_merged": 0, "refusals": [], "instrument_seed_id": INSTRUMENT_SEED,
                                 "document": {t: {"model_key": v[0], "panel": v[1], "object_keys": list(v[2])} for t, v in DOCUMENT.items()},
                                 "producer_key_rule": PRODUCER_KEY_RULE, "anchor_state": ANCHOR_STATE, "coexist_ruling": COEXIST_RULING_ID}
        doc = DOCUMENT.get(group)
        if doc is None:
            self.refusals.add(group, f"tag {group!r} is not one of the document's standups ({', '.join(sorted(DOCUMENT))}); a fourth model needs Overflow's declaration and a line in DOCUMENT",
                              inputs[0].uri if inputs else self.root)
            self._finish(group, facts)
            yield batch
            return
        model_key, panel, declared_keys = doc
        seeds = self.seeds
        mv = seeds.model_for_view_legacy(group)
        model_id = mv.get("model_id")
        if model_id is None or mv.get("base") != model_key or mv.get("view_key") != panel:
            self.refusals.add(group, f"tag {group!r} spells model {mv.get('base')!r} at panel {mv.get('view_key')!r} ({mv.get('source')}); the document declares {model_key}/{panel}", self.root)
            self._finish(group, facts)
            yield batch
            return
        thinking = thinking_for(seeds, model_id)
        base_dir, ceil_dir = store_dirs_for(thinking)
        try:
            inst_row = seeds.instrument_row(INSTRUMENT_SEED, EVAL_ID, model_id=model_id)
        except KeyError as exc:
            self.refusals.add(group, f"instrument row: {exc}", self.root)
            self._finish(group, facts)
            yield batch
            return
        facts.update(model_key=model_key, panel=panel, model_id=model_id, thinking=thinking, sampling_dir=base_dir, seed_instrument_id=str(inst_row["instrument_id"]),
                     declared_keys=list(declared_keys), trial_rule=G.DEFAULT_SEED_TRIAL_RULE)
        decl_rec, decl_why = self.declaration()
        facts["declaration"] = (decl_rec["_path"] if decl_rec is not None else None)
        facts["declaration_state"] = ("present" if decl_rec is not None else f"absent: {decl_why}")
        ref = self.in_frame_unsteered(ctx.conn, ctx.now).get(model_id) or {"state": "unresolved", "why": f"no reference derived for {model_id}", "anchor_state": ANCHOR_STATE,
                                                                             "in_frame_unsteered_run_ids": [], "anchors": [], "n_candidates": None, "derivation": ANCHOR_DERIVATION,
                                                                             "derived_at": ctx.now}
        facts["in_frame_unsteered_reference"] = {k: ref.get(k) for k in ("state", "why", "in_frame_unsteered_run_ids", "n_candidates")}
        stores, prov, shared = self._sort_paired_stores(inputs, group)
        records = self._shared_records(shared, facts)
        for slug in sorted({s for s, _d in stores}):
            okey = G.slug_to_object_key(slug)
            rkey = f"{group}|{okey}"
            dirs_here = {d for s, d in stores if s == slug}
            if okey == "none":
                self.refusals.add(rkey, "a `none` object dir: the document declares no unsteered run (the panel pools the in-frame anchors); an unsteered run filed here would "
                                        "enter another source's collapse key, so it is refused by name", stores[(slug, next(iter(sorted(dirs_here))))]["dir"])
                continue
            if okey not in declared_keys:
                self.refusals.add(rkey, f"object key {okey!r} is not declared on {group} (the document declares {', '.join(declared_keys)})", stores[(slug, next(iter(sorted(dirs_here))))]["dir"])
                continue
            stray = sorted(d for d in dirs_here if d not in (base_dir, ceil_dir))
            if stray:
                self.refusals.add(rkey, f"store directories {stray} are not the model's ({model_id} thinking {thinking!r}: {base_dir} / {ceil_dir}); a store under another "
                                        "model's sampling directory is refused by name", stores[(slug, stray[0])]["dir"])
                continue
            base_st, ceil_st = stores.get((slug, base_dir)), stores.get((slug, ceil_dir))
            if base_st is None:
                self.refusals.add(rkey, f"a ceiling store without its base store ({base_dir}/ is absent)", ceil_st["dir"] if ceil_st else self.root)
                continue
            try:
                if self._result_last_line(base_st).endswith(RESULT_DRIVER_FAILED) or RESULT_DRIVER_FAILED in self._result_last_line(base_st):
                    # a leg the driver refused: a REFUSAL STATE (a `not_served` disposition spelling which refusal), never nothing
                    self._emit_refused_pair(batch, ctx, group, model_key, panel, model_id, thinking, base_dir, okey, slug, base_st, ceil_st, records, facts)
                else:
                    self._emit_paired_pair(batch, ctx, group, model_key, panel, model_id, thinking, base_dir, inst_row, okey, slug, base_st, ceil_st, prov.get(slug), ref, records, facts)
            except G._PairRefusal as exc:
                self.refusals.add(rkey, exc.why, exc.where)
        self._finish(group, facts)
        yield batch

    # ---- the producer's records beside the stores (optional; bound when present) --------------
    def _shared_records(self, shared: Mapping[str, Input], facts: Dict[str, Any]) -> Dict[str, Any]:
        """The three producer records at the mirror root as parsed mappings (``None`` when absent or malformed, the reason on the facts): the
        store pins (schema-checked), the three-number record (schema-checked, ``rows`` keyed ``<tag>/<object dir>``), the final-state record."""
        out: Dict[str, Any] = {"pins": None, "three": None, "fstate": None, "pins_inp": shared.get(PINS_FILE), "three_inp": shared.get(THREE_NUMBERS_FILE),
                               "fstate_inp": shared.get(FINAL_STATE_FILE)}
        why: Dict[str, str] = {}
        if out["pins_inp"] is not None:
            d = self._json(out["pins_inp"])
            if isinstance(d, dict) and str((d.get("_meta") or {}).get("schema")) == PINS_SCHEMA:
                out["pins"] = d
            else:
                why["pins"] = f"{PINS_FILE} is not a {PINS_SCHEMA} mapping"
        else:
            why["pins"] = f"{PINS_FILE} is absent from the mirror root"
        if out["three_inp"] is not None:
            d = self._json(out["three_inp"])
            if isinstance(d, dict) and str(d.get("schema")) == THREE_NUMBERS_SCHEMA and isinstance(d.get("rows"), dict):
                out["three"] = d
            else:
                why["three"] = f"{THREE_NUMBERS_FILE} is not a {THREE_NUMBERS_SCHEMA} record with rows"
        else:
            why["three"] = f"{THREE_NUMBERS_FILE} is absent from the mirror root"
        if out["fstate_inp"] is not None:
            d = self._json(out["fstate_inp"])
            if isinstance(d, dict) and d and all(isinstance(v, Mapping) and "n_scored_final" in v for v in d.values()):
                out["fstate"] = d
            else:
                why["fstate"] = f"{FINAL_STATE_FILE} is not a per-pair final-state mapping"
        else:
            why["fstate"] = f"{FINAL_STATE_FILE} is absent from the mirror root"
        facts["producer_records"] = {"store_pins": (LC.rel_home(out["pins_inp"].uri.rsplit("#", 1)[0]) if out["pins"] is not None else None),
                                     "three_numbers": (LC.rel_home(out["three_inp"].uri.rsplit("#", 1)[0]) if out["three"] is not None else None),
                                     "final_state": (LC.rel_home(out["fstate_inp"].uri.rsplit("#", 1)[0]) if out["fstate"] is not None else None), "absent_or_malformed": why}
        for name, inp in (("pins", out["pins_inp"]), ("three", out["three_inp"]), ("fstate", out["fstate_inp"])):
            if inp is not None and out[name] is None:
                # a present record that is malformed refuses the whole tag: nothing can be cross-checked and the record is not what it claims
                self.refusals.add(f"{facts['tag']}|*", why[name], inp.uri.rsplit("#", 1)[0])
        return out

    @staticmethod
    def _result_last_line(st: Mapping[str, Any]) -> str:
        top: Mapping[str, Input] = st.get("top") or {}
        if "RESULT" not in top:
            return ""
        with open(top["RESULT"].path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _pin_key(model_key: str, panel: str, okey: str, cap: int) -> str:
        return f"{model_key}/{panel}/{okey}@{cap}"

    def _check_pins(self, pins: Optional[Mapping[str, Any]], pins_inp: Optional[Input], model_key: str, panel: str, okey: str, tree: str, ceil: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        """The store pins against the manifests when the record is present: the base tree its ``@8192`` pin, a ceiling store present iff pinned and at
        its ``@16384`` pin; a leg the pins name under ``_meta.refused`` may not carry a completed store. Returns the pins read (None when absent)."""
        if pins is None:
            return None
        kb, kc = self._pin_key(model_key, panel, okey, CAP), self._pin_key(model_key, panel, okey, CEILING)
        refused = ((pins.get("_meta") or {}).get("refused") or {}) if isinstance(pins.get("_meta"), Mapping) else {}
        if kb in refused:
            raise G._PairRefusal(f"the store pins name this leg under _meta.refused ({kb}: {str((refused.get(kb) or {}).get('RESULT'))[:120]}) but a completed store is present", pins_inp.uri)
        pin_base, pin_ceiling = pins.get(kb), pins.get(kc)
        if pin_base != tree:
            raise G._PairRefusal(f"the base store's tree {tree[:12]} is not its store pin {str(pin_base)[:12] if pin_base else '<absent>'} ({kb})", pins_inp.uri)
        if (ceil is not None) != (pin_ceiling is not None):
            raise G._PairRefusal(f"a ceiling store is {'present' if ceil else 'absent'} but its store pin {kc} is {'absent' if ceil else 'present'}", ceil["dir"] if ceil else pins_inp.uri)
        if ceil is not None:
            rtree = str((ceil["manifest"].get("producer") or {}).get("tree_sha") or "")
            if rtree != pin_ceiling:
                raise G._PairRefusal(f"the ceiling store's tree {rtree[:12] or '<none>'} is not its store pin {str(pin_ceiling)[:12]} ({kc})", ceil["files"]["manifest.json"].uri)
        return {str(CAP): pin_base, str(CEILING): pin_ceiling, "record": LC.rel_home(pins_inp.uri.rsplit("#", 1)[0])}

    @staticmethod
    def _cross_check_three_offtask(tn: Mapping[str, Any], ceil: Optional[Mapping[str, Any]], items: Mapping[str, Any], n_items: int, attested: str, exit_: str) -> List[str]:
        """The producer's ``g8k_three_numbers_offtask/1`` row against the re-derivation: the value, the bound, the third state, the base and ceiling readings,
        the object and the role (the off-task share is the producer's own reading beside the value and is carried, never re-derived)."""
        c = items["counts"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False) -> None:
            ok = G.rounded_equal(theirs, mine) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        eq("n_attempted", LC.int_or_none(tn.get("n_attempted")), n_items)
        eq("n_scored_final", LC.int_or_none(tn.get("n_scored_final")), c["n_scored_final"])
        eq("correct_final", LC.int_or_none(tn.get("correct_final")), c["correct_final"])
        eq("value_correct_over_scored", LC.num_or_none(tn.get("value_correct_over_scored")), (c["correct_final"] / c["n_scored_final"]) if c["n_scored_final"] else None, rounded=True)
        eq("bound_correct_over_attempted", LC.num_or_none(tn.get("bound_correct_over_attempted")), c["correct_final"] / n_items, rounded=True)
        third = tn.get("third_state") if isinstance(tn.get("third_state"), Mapping) else {}
        eq("third_state.n_unresolved_at_cap", LC.int_or_none(third.get("n_unresolved_at_cap")), c["n_unresolved"])
        eq("third_state.loops", LC.int_or_none(third.get("loops")), c["n_loops"])
        eq("third_state.still_at_ceiling_after_rung", LC.int_or_none(third.get("still_at_ceiling_after_rung")), c["n_ceiling"])
        eq("third_state.eligible_not_escalated", LC.int_or_none(third.get("eligible_not_escalated")), c["n_not_escalated"])
        eq("third_state.n_unscored_below_cap", LC.int_or_none(third.get("n_unscored_below_cap")), c["n_unscored_below_cap"])
        base = tn.get("base_8192") if isinstance(tn.get("base_8192"), Mapping) else {}
        eq("base_8192.n_scored", LC.int_or_none(base.get("n_scored")), c["n_scored_base"])
        eq("base_8192.correct", LC.int_or_none(base.get("correct")), c["correct_base"])
        eq("base_8192.n_at_cap", LC.int_or_none(base.get("n_at_cap")), c["n_at_cap_base"])
        eq("base_8192.n_eligible", LC.int_or_none(base.get("n_eligible")), c["n_eligible"])
        eq("base_8192.ladder_exit", str(base.get("ladder_exit")), str(exit_))
        r = tn.get("rung_16384")
        if ceil is None:
            eq("rung_16384", r, None)
        elif not isinstance(r, Mapping):
            bad.append(f"rung_16384: producer {r!r}, a ceiling store is present")
        else:
            eq("rung_16384.n_drawn", LC.int_or_none(r.get("n_drawn")), c["n_ceil_drawn"])
            eq("rung_16384.n_scored", LC.int_or_none(r.get("n_scored")), c["n_scored_ceil"])
            eq("rung_16384.correct", LC.int_or_none(r.get("correct")), c["correct_ceil"])
            eq("rung_16384.still_at_cap", LC.int_or_none(r.get("still_at_cap")), c["n_ceiling"])
        if tn.get("object_sha256") is not None:
            eq("object_sha256", str(tn.get("object_sha256")), attested)
        if tn.get("record_role") is not None:
            eq("record_role", str(tn.get("record_role")), ROLE)
        return bad

    # ---- a leg the driver refused: a refusal state, never nothing -----------------------------
    def _emit_refused_pair(self, batch: RecordBatch, ctx: AdapterContext, tag: str, model_key: str, panel: str, model_id: str, thinking: str, sampling_dir: str, okey: str,
                           slug: str, base_st: Mapping[str, Any], ceil_st: Optional[Mapping[str, Any]], records: Mapping[str, Any], facts: Dict[str, Any]) -> None:
        """A store whose ``RESULT`` ends ``result=DRIVER_FAILED``: no run, no metric, no item — a ``disposition`` row (state ``not_served``) whose
        ``reason`` and ``extra.refusal_state`` spell WHICH refusal (module constants), carrying the RESULT line, the attempt ``ladder_state.json`` records,
        the render-probe evidence (path + every draw's class) and the store pins' ``_meta.refused`` entry when the record is present. The object is the
        manifest's attested sha, bound to the registry's object row when the connection carries it (else the disposition names no object and says so)."""
        top: Mapping[str, Input] = base_st["top"]
        result_line = self._result_last_line(base_st)
        if ceil_st is not None:
            raise G._PairRefusal("a ceiling store beside a leg the driver refused (no base generation to escalate from)", ceil_st["dir"])
        if "ladder_state.json" not in top:
            raise G._PairRefusal("the refused leg carries no ladder_state.json; the attempt that failed cannot be named", base_st["dir"])
        ladder_state = self._json(top["ladder_state.json"]) or {}
        if str(ladder_state.get("schema")) != G.LADDER_STATE_SCHEMA:
            raise G._PairRefusal(f"ladder_state.json schema is {ladder_state.get('schema')!r}, not {G.LADDER_STATE_SCHEMA}", top["ladder_state.json"].uri)
        if ladder_state.get("accepted_generation_attempt") is not None:
            raise G._PairRefusal(f"RESULT ends {RESULT_DRIVER_FAILED} but ladder_state.json accepts attempt {ladder_state.get('accepted_generation_attempt')!r}", top["ladder_state.json"].uri)
        attempts = [a for a in (ladder_state.get("attempts") or []) if isinstance(a, Mapping)]
        failed = [a for a in attempts if str(a.get("result")) == "DRIVER_FAILED"]
        if not failed:
            raise G._PairRefusal("ladder_state.json names no DRIVER_FAILED attempt for a RESULT that ends DRIVER_FAILED", top["ladder_state.json"].uri)
        last = failed[-1]
        job = str(last.get("job_id") or "")
        if job and f"job={job}" not in result_line:
            raise G._PairRefusal(f"the RESULT line names another job than ladder_state.json's failed attempt {job}", top["RESULT"].uri)
        tree = str(last.get("sha") or "")
        # the manifest (written before generation) attests the object and the served model
        keys = base_st["keys"]
        if len(keys) != 1:
            raise G._PairRefusal(f"the refused leg holds {len(keys)} store key directories under cells/ (expected exactly one)", base_st["dir"])
        store_key = next(iter(keys))
        files: Mapping[str, Input] = keys[store_key]
        if "manifest.json" not in files:
            raise G._PairRefusal(f"the refused leg's store {store_key} carries no manifest.json; the object cannot be named", os.path.join(base_st["dir"], "cells", store_key))
        man = self._json(files["manifest.json"]) or {}
        if str(man.get("cell_id") or "") != okey:
            raise G._PairRefusal(f"manifest.json names object key {man.get('cell_id')!r}, not {okey!r}", files["manifest.json"].uri)
        mtree = str((man.get("producer") or {}).get("tree_sha") or "")
        if _SHA1.match(mtree) and tree and mtree != tree:
            raise G._PairRefusal(f"manifest.json:producer.tree_sha {mtree[:12]} is not the failed attempt's sha {tree[:12]}", files["manifest.json"].uri)
        tree = tree or mtree
        served_hf = str(man.get("model_id") or "")
        if served_hf and self.seeds.model_for_hf_id(served_hf) != model_id:
            raise G._PairRefusal(f"the refused leg's manifest serves {served_hf!r}, not the tag's model {model_id}", files["manifest.json"].uri)
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        prov_sha = str((man.get("provenance") or {}).get("artifact_sha256") or "") if isinstance(man.get("provenance"), Mapping) else ""
        sha16 = str(fitp.get("object_sha") or "")
        attested = prov_sha if _SHA256.match(prov_sha) else None
        if attested is not None and _SHA16.match(sha16) and not attested.startswith(sha16):
            raise G._PairRefusal(f"manifest.json:fit_provenance.object_sha {sha16!r} is no prefix of provenance.artifact_sha256 {attested[:16]}…", files["manifest.json"].uri)
        g = LC.cell_grammar_legacy(okey)
        object_id: Optional[str] = None
        object_why = "the manifest attests no 64-hex object"
        if attested is not None:
            if ctx.conn is None:
                object_why = f"the manifest attests {attested[:16]}… but no registry connection can verify the object row; the disposition names no object"
            else:
                try:
                    row = ctx.conn.execute("SELECT object_kind, paper_label FROM object WHERE object_id = ?", (attested,)).fetchone()
                except sqlite3.Error as exc:
                    row = None
                    object_why = f"the object table could not be read: {exc}"
                if row is not None:
                    if str(row[0]) != g["object_kind"] or str(row[1]) != g["paper_label"]:
                        raise G._PairRefusal(f"the object row of {attested[:16]}… is kind {row[0]!r} / label {row[1]!r}; the key {okey} spells {g['object_kind']!r} / {g['paper_label']!r}",
                                             files["manifest.json"].uri)
                    object_id = attested
                    object_why = "the manifest's provenance.artifact_sha256, a row of the registry's object table"
                elif ctx.conn is not None and object_why.startswith("the manifest attests no"):
                    object_why = f"the manifest attests {attested[:16]}… but the registry carries no object row for it; the disposition names no object"
        # the render-probe evidence: predicate A iff every ON-arm draw ended inside the prompt-opened think block
        ev_inp = top.get("evidence.json") or files.get("evidence.json")
        ev = self._json(ev_inp) if ev_inp is not None else None
        probe = (ev or {}).get("render_probe") if isinstance((ev or {}).get("render_probe"), Mapping) else {}
        on_attempts = [a for a in (probe.get("on_attempts") or []) if isinstance(a, Mapping)]
        classes = [str(a.get("class") or "") for a in on_attempts]
        certified = (ev or {}).get("render_certified") if isinstance((ev or {}).get("render_certified"), Mapping) else {}
        predicate_a = bool(classes) and all(c == PROBE_CLASS_UNCLOSED for c in classes)
        state = REFUSAL_STATE_THINKING_UNSERVABLE_A if predicate_a else REFUSAL_STATE_DRIVER_FAILED
        # the store pins' refused entry, when the record names the leg
        pins, pins_inp = records.get("pins"), records.get("pins_inp")
        kb = self._pin_key(model_key, panel, okey, CAP)
        pins_refused = None
        if pins is not None:
            refused = ((pins.get("_meta") or {}).get("refused") or {}) if isinstance(pins.get("_meta"), Mapping) else {}
            if kb in pins:
                raise G._PairRefusal(f"the store pins carry a completed pin for {kb} but the store's RESULT ends {RESULT_DRIVER_FAILED}", pins_inp.uri)
            ent = refused.get(kb)
            if isinstance(ent, Mapping):
                if str(ent.get("RESULT") or "").strip() != result_line:
                    raise G._PairRefusal(f"the store pins' _meta.refused[{kb}].RESULT is not the store's RESULT line", pins_inp.uri)
                if ent.get("tree") is not None and tree and str(ent.get("tree")) != tree:
                    raise G._PairRefusal(f"the store pins' _meta.refused[{kb}].tree {str(ent.get('tree'))[:12]} is not the failed attempt's {tree[:12]}", pins_inp.uri)
                pins_refused = {"key": kb, "RESULT": ent.get("RESULT"), "tree": ent.get("tree"), "record": LC.rel_home(pins_inp.uri.rsplit("#", 1)[0])}
        decl: Optional[Dict[str, Any]] = None
        try:
            decl = self.declaration_row(tag, okey)
        except IC.Unresolved as exc:
            raise G._PairRefusal(f"declaration: {exc.why}", exc.path)
        when = IC.utc_from_iso(last.get("ended")) or IC.utc_from_iso(last.get("started")) or ctx.now
        reason = (f"{state}: {result_line}" if result_line else state)
        extra = {"refusal_state": state, "refusal_rule": (PREDICATE_A_RULE if predicate_a else "the driver refused the leg (RESULT DRIVER_FAILED) and its evidence names no predicate"),
                 "document": "paired_allmodels", "declared_role": ROLE, "lane_source_legacy": SOURCE, "view_legacy": tag, "campaign_cell_id_legacy": okey, "model_key": model_key,
                 "panel": panel, "thinking_spelling": thinking, "sampling_dir": sampling_dir, "store_uri": LC.rel_home(base_st["dir"]), "store_key": store_key,
                 "result_line": result_line, "result_uri": LC.rel_home(top["RESULT"].uri), "ladder_state_uri": LC.rel_home(top["ladder_state.json"].uri),
                 "failed_attempt": {k: last.get(k) for k in ("attempt_id", "job_id", "job_name", "node", "cluster", "rc", "result", "started", "ended", "sha", "rung")},
                 "n_attempts": len(attempts), "tree_sha": tree or None,
                 "evidence_uri": (LC.rel_home(ev_inp.uri) if ev_inp is not None else None), "render_probe_on_attempts": on_attempts, "render_probe_classes": classes,
                 "render_certified": {k: certified.get(k) for k in ("think_shape_certified", "render_regime_certified", "header_certified", "render_certified", "split_mode",
                                                                        "evidence_status", "certification_problems", "standup_thinking_mode")},
                 "served_model_id_hf": served_hf or None, "served_model_revision": man.get("model_revision"), "serving_identity": man.get("serving_identity"),
                 "object_attested_sha256": attested, "object_sha16": (sha16 or None), "object_resolution": object_why, "object_kind": g["object_kind"], "paper_label": g["paper_label"],
                 "fit_model_id_hf": fitp.get("model_id"), "oos_tuple": man.get("oos_experiment"), "store_pins_refused": pins_refused,
                 "declaration": ({k: decl.get(k) for k in ("_path", "_schema", "_sha256", "index", "tag", "job_id", "role", "expected_refusal", "expected_refusal_reason", "held_reason")}
                                 if decl is not None else None),
                 "coexist_ruling": {"ruling_id": COEXIST_RULING_ID, "matches_object_prefix": bool(attested and any(attested.startswith(p) for p in COEXIST_OBJECT_PREFIXES))},
                 "note": ("a leg that never completed files as a refusal state so the panel reads WHY the (model, object) has no value here: the thinking-ON instrument "
                          "cannot be served for it (predicate A) — not an absent row, not a null" if predicate_a else
                          "a leg that never completed files as a refusal state so the panel reads that the (model, object) has no value here — not an absent row, not a null")}
        payload = {"disposition_id": str(uuid.uuid5(I.RECORD_NAMESPACE, I.canonical_json([NAME, SOURCE, model_id, tag, okey, EVAL_ID, state]))),
                   "round_id": LC.ROUND_ID, "roster_entry_id": None, "comparison_id": f"{SOURCE}/{tag}/{okey}", "model_id": model_id, "object_id": object_id, "eval_id": EVAL_ID,
                   "state": REFUSAL_DISPOSITION_STATE, "reason": reason, "source_ref": LC.rel_home(top["RESULT"].uri), "ruled_utc": when, "extra": extra}
        batch.add("disposition", payload, ctx.producer, ctx.now)
        facts.setdefault("refused_legs", {})[okey] = {"refusal_state": state, "disposition_id": payload["disposition_id"], "result_line": result_line, "job": job,
                                                     "node": last.get("node"), "tree": tree or None, "object_id": object_id, "object_attested_sha256": attested,
                                                     "render_probe_classes": classes, "pins_refused_bound": pins_refused is not None, "declaration_bound": decl is not None}

    # ---- one (tag, object key) pair -----------------------------------------------------------
    def _emit_paired_pair(self, batch: RecordBatch, ctx: AdapterContext, tag: str, model_key: str, panel: str, model_id: str, thinking: str, sampling_dir: str,
                          inst_row: Mapping[str, Any], okey: str, slug: str, base_st: Mapping[str, Any], ceil_st: Optional[Mapping[str, Any]], prov_inp: Optional[Input],
                          ref: Mapping[str, Any], records: Mapping[str, Any], facts: Dict[str, Any]) -> None:
        seeds = self.seeds
        producer, now = ctx.producer, ctx.now
        base = self._store_bundle(base_st, False, okey)
        ceil = self._store_bundle(ceil_st, True, okey) if ceil_st is not None else None
        man = base["manifest"]
        mfile = base["files"]["manifest.json"].uri
        if base["ck"].cell_id != okey:
            raise G._PairRefusal(f"the base store key spells object {base['ck'].cell_id!r} but the directory slug spells {okey!r}", base["dir"])
        if str(man.get("cell_id") or "") != okey:
            raise G._PairRefusal(f"manifest.json names object key {man.get('cell_id')!r}, not {okey!r}", mfile)
        # -- the tree: attested by the accepted attempt (no store pins beside this document)
        tree = str((man.get("producer") or {}).get("tree_sha") or "")
        if not _SHA1.match(tree):
            raise G._PairRefusal("manifest.json:producer.tree_sha names no 40-hex tree", mfile)
        att_sha = str(base["attempt"].get("sha") or "")
        if att_sha != tree:
            raise G._PairRefusal(f"the base store's tree {tree[:12]} is not its accepted attempt's sha {att_sha[:12] or '<none>'} (ladder_state.json attempts[{base['attempt'].get('attempt_id')}])",
                                 base["top"]["ladder_state.json"].uri)
        if ceil is not None:
            rtree = str((ceil["manifest"].get("producer") or {}).get("tree_sha") or "")
            ratt = str(ceil["attempt"].get("sha") or "")
            if not _SHA1.match(rtree) or rtree != ratt:
                raise G._PairRefusal(f"the ceiling store's tree {rtree[:12] or '<none>'} is not its accepted attempt's sha {ratt[:12] or '<none>'} (ladder_state.json)", ceil["files"]["manifest.json"].uri)
            sub_src = str(((ceil["manifest"].get("item_subset") or {}) if isinstance(ceil["manifest"].get("item_subset"), Mapping) else {}).get("source_identity_hash") or "")
            if sub_src != str(man.get("harness_config_hash") or ""):
                raise G._PairRefusal(f"the ceiling manifest's item_subset.source_identity_hash {sub_src!r} is not the base store's harness_config_hash {man.get('harness_config_hash')!r}; "
                                     "the ceiling was not drawn from this base store", ceil["files"]["manifest.json"].uri)
            if str(ceil["manifest"].get("cell_id") or "") != okey or ceil["ck"].cell_id != okey:
                raise G._PairRefusal(f"the ceiling store names object key {ceil['manifest'].get('cell_id')!r}, not {okey!r}", ceil["files"]["manifest.json"].uri)
        # -- the store pins, when the producer's record is present: the base tree its pin, a ceiling store present iff pinned
        pins_read = self._check_pins(records.get("pins"), records.get("pins_inp"), model_key, panel, okey, tree, ceil)
        # -- the served model and identity
        served_hf = str(man.get("model_id") or "")
        if seeds.model_for_hf_id(served_hf) != model_id:
            raise G._PairRefusal(f"the store serves {served_hf!r}, not the tag's model {model_id}", mfile)
        rev = str(man.get("model_revision") or "")
        pinned_rev = str((seeds.models.get(model_id) or {}).get("hf_revision") or "")
        if _SHA1.match(pinned_rev) and rev != pinned_rev:
            raise G._PairRefusal(f"the store serves revision {rev[:12] or '<none>'}, not the model seed's {pinned_rev[:12]}", mfile)
        ident = str(man.get("serving_identity") or "")
        if ident != G.STEERED_IDENTITY:
            raise G._PairRefusal(f"serving_identity {ident!r} is not the LoRA-native object's {G.STEERED_IDENTITY!r}", mfile)
        if str(man.get("serving_engine") or "") != "lora_native":
            raise G._PairRefusal(f"manifest.json:serving_engine {man.get('serving_engine')!r} is not 'lora_native'", mfile)
        # -- the object, attested by the store and bound to the registry's row (+ roster / pins, + the staging manifest)
        obj, ofacts = self.resolve_attested_object(ctx.conn, okey, model_key, model_id, panel, base)
        # -- the transfer: from the fit provenance, never the standup spelling
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        fit_hf = str(fitp.get("model_id") or "")
        fit_model_id = seeds.model_for_hf_id(fit_hf) if fit_hf else None
        if fit_hf and fit_model_id is None:
            raise G._PairRefusal(f"manifest.json:fit_provenance.model_id {fit_hf!r} is no model of models.yaml; the transfer cannot be derived", mfile)
        if fit_model_id is not None and fitp.get("transfer") is not None and bool(fitp.get("transfer")) != (fit_model_id != model_id):
            raise G._PairRefusal(f"manifest.json:fit_provenance.transfer {fitp.get('transfer')!r} disagrees with fit {fit_model_id} vs served {model_id}", mfile)
        tk = DV.transfer_kind(fit_model_id, model_id, obj.object_kind, run_checkpoint_kind=(seeds.models.get(model_id) or {}).get("checkpoint_kind"),
                              fit_checkpoint_kind=((seeds.models.get(fit_model_id) or {}).get("checkpoint_kind") if fit_model_id else None))
        if tk[0] not in V.TRANSFER_KINDS:
            raise G._PairRefusal(f"the transfer kind cannot be derived from fit_provenance ({tk[2]})", mfile)
        row_fit = ofacts["object_row"].get("fit_model_id")
        if fit_model_id is not None and row_fit and str(row_fit) != fit_model_id:
            raise G._PairRefusal(f"the registry's object row was fit on {row_fit}, the store's fit_provenance names {fit_model_id} ({fit_hf})", mfile)
        # -- the seed rule: the default Inspect rule (no replicate block), spelled by the store
        rep = ((man.get("seeds") or {}).get("inspect") or {}).get("replicate") if isinstance(man.get("seeds"), Mapping) else None
        if isinstance(rep, Mapping) and rep:
            raise G._PairRefusal(f"manifest.json:seeds.inspect carries a replicate policy ({rep.get('policy')!r} at seed_base {rep.get('seed_base')!r}); the document is the default "
                                 "Inspect seed rule — a replicate store belongs to a seed-replicate document", mfile)
        seed_rule_text = str(((man.get("seeds") or {}).get("inspect") or {}).get("rule") or "") if isinstance(man.get("seeds"), Mapping) else ""
        if not seed_rule_text:
            raise G._PairRefusal("manifest.json:seeds.inspect.rule names no seed rule; the default rule must be spelled by the store", mfile)
        # -- the item count: the item set's 200 unless the base manifest declares an item_subset
        n_items = N_ITEMS
        subset = man.get("item_subset") if isinstance(man.get("item_subset"), Mapping) else None
        if subset is not None:
            n_items = LC.int_or_none(subset.get("n")) or len(subset.get("ids") or [])
            if not n_items or (subset.get("ids") is not None and len(subset["ids"]) != n_items):
                raise G._PairRefusal(f"manifest.json:item_subset declares n {subset.get('n')!r} over {len(subset.get('ids') or [])} ids", mfile)
        items = self._merge_items(base, ceil, okey, n_items=n_items, thinking=thinking)
        counts = items["counts"]
        # -- the producer's own count of the same numbers, when its records are present (the three-number row, the final-state row): every count must agree
        tn_row = fs_row = None
        producer_record: Optional[str] = None
        if records.get("three") is not None:
            tn_row = records["three"]["rows"].get(f"{tag}/{slug}")
            if isinstance(tn_row, Mapping):
                bad = self._cross_check_three_offtask(tn_row, ceil, items, n_items, obj.object_id, items["exit"])
                if bad:
                    raise G._PairRefusal(f"{THREE_NUMBERS_FILE} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), records["three_inp"].uri)
                producer_record = LC.rel_home(records["three_inp"].uri.rsplit("#", 1)[0])
            else:
                tn_row = None
        if records.get("fstate") is not None:
            fs_row = records["fstate"].get(f"{tag}/{slug}")
            if isinstance(fs_row, Mapping):
                bad = self._cross_check_final_state(fs_row, ceil, items)
                if bad:
                    raise G._PairRefusal(f"{FINAL_STATE_FILE} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), records["fstate_inp"].uri)
                producer_record = producer_record or LC.rel_home(records["fstate_inp"].uri.rsplit("#", 1)[0])
            else:
                fs_row = None
        if (records.get("three") is not None or records.get("fstate") is not None) and tn_row is None and fs_row is None:
            raise G._PairRefusal(f"the producer's records are present but neither {THREE_NUMBERS_FILE} nor {FINAL_STATE_FILE} carries a row for {tag}/{slug}",
                                 (records.get("three_inp") or records.get("fstate_inp")).uri)
        producer_version = RECORD_VERSION_PRODUCER if producer_record is not None else G.RECORD_VERSION_NONE
        summary_inp = base_st.get("summary")
        if summary_inp is not None:
            bad = self._cross_check_target_summary(self._json(summary_inp) or {}, base, items, obj.object_id, tree, okey=okey, n_items=n_items, thinking=thinking)
            if bad:
                raise G._PairRefusal(f"{G.TARGET_SUMMARY_FILE} disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), summary_inp.uri)
        # -- decode: the sent numbers against the model's registered policy row at ITS thinking mode, and the instrument's columns
        gs = base["gen_stats"]
        block = {k: gs.get(k) for k in DP.PARAM_KEYS}
        sent = {k: v for k, v in block.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        seed_inst_match = "match" if self._numbers_agree(block, {k: inst_row.get(f"decode_{k}") for k in DP.PARAM_KEYS}) else "mismatch"
        policy, policy_why = self._match_policy(model_id, block, thinking_mode=thinking)
        if ceil is not None:
            rblock = {k: ceil["gen_stats"].get(k) for k in DP.PARAM_KEYS}
            if not self._numbers_agree(block, rblock):
                raise G._PairRefusal(f"the ceiling store sent decode {G.RP._numbers(rblock)}, the base {G.RP._numbers(block)}; one ladder must send one row", ceil["files"]["gen_stats.json"].uri)
        # -- the declaration record, when present
        try:
            decl = self.declaration_row(tag, okey)
        except IC.Unresolved as exc:
            raise G._PairRefusal(f"declaration: {exc.why}", exc.path)
        if decl is not None:
            bad = self._check_declaration(decl, tag, model_key, panel, slug, sampling_dir, thinking, base, ceil, obj.object_id, obj.staged_id, block)
            if bad:
                raise G._PairRefusal(f"the declaration {decl['_path']} (row {decl.get('index')}) disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), decl["_path"])
        # -- time, job, substrate
        final = ceil if ceil is not None else base
        when_base_start = IC.utc_from_iso(base["attempt"].get("started")) or self._rows_time(base)
        when_final = IC.utc_from_iso(final["attempt"].get("ended")) or self._rows_time(final)
        if when_final is None or when_base_start is None:
            raise G._PairRefusal("ladder_state.json names no start / end for the accepted attempt and rows.jsonl carries no created_at", base["top"]["ladder_state.json"].uri)
        job = str((man.get("row_constants") or {}).get("slurm_job_id") or "")
        att_job = str(base["attempt"].get("job_id") or "")
        if not job or job != att_job:
            raise G._PairRefusal(f"manifest.json:row_constants.slurm_job_id {job!r} and ladder_state.json's accepted attempt job {att_job!r} disagree", mfile)
        ceiling_job = (str((ceil["manifest"].get("row_constants") or {}).get("slurm_job_id") or "") if ceil is not None else None)
        witness = G.SC.substrate_of_routing_table(str(man.get("routing_table") or ""))
        if witness is None:
            raise G._PairRefusal(f"manifest.json:routing_table {str(man.get('routing_table'))!r} is on no cluster's shared root (/workspace-vast/ = cluster A, /workspace/ = cluster B)", mfile)
        substrate = (witness[0], "derived", witness[2])
        # -- the instrument: the model's canonical seed row (the per-model expansion); nothing minted (module docstring, THE INSTRUMENT)
        final_cap = CEILING if ceil is not None else CAP
        state = "final_at_ceiling" if ceil is not None else "final_at_base"
        geometry = man.get("geometry") if isinstance(man.get("geometry"), Mapping) else {}
        rgeometry = (ceil["manifest"].get("geometry") if ceil is not None and isinstance(ceil["manifest"].get("geometry"), Mapping) else {})
        frameworks = man.get("frameworks") if isinstance(man.get("frameworks"), Mapping) else {}
        if str(inst_row.get("thinking_mode") or "") != thinking:
            raise G._PairRefusal(f"the seed instrument {INSTRUMENT_SEED} bound to {model_id} declares thinking {inst_row.get('thinking_mode')!r}, the model's spelling is {thinking!r}", mfile)
        inst_match = seed_inst_match
        instrument_id = str(inst_row["instrument_id"])
        producer_key = producer_key_for(tag, base["store_key"])
        rid = run_id_for(tag, base["store_key"], okey, instrument_id)
        n_scored, correct = counts["n_scored_final"], counts["correct_final"]
        acc = (correct / n_scored) if n_scored else None
        bound_lo = correct / n_items
        bound_hi = (correct + counts["n_unresolved"]) / n_items
        three_block = {"n_attempted": n_items, "n_scored_final": n_scored, "correct_final": correct, "acc_over_scored": acc, "acc_over_attempted": bound_lo,
                       "acc_over_attempted_upper_bound": bound_hi, "n_unresolved_neither_hit_nor_miss": counts["n_unresolved"], "n_unresolved_ladder_only": counts["n_unresolved_ladder_only"],
                       "n_rung_drawn_that_were_scored_at_base": counts["n_redrawn_scored_at_base"], "n_loops": counts["n_loops"],
                       "n_at_cap_ceiling": counts["n_ceiling"], "n_at_cap_not_escalated": counts["n_not_escalated"], "n_unscored_final": counts["n_unscored_final"],
                       "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_scored_at_cap_final": counts["n_scored_at_cap"], "n_at_cap_final": counts["n_at_cap_final"],
                       "n_at_cap_8192": counts["n_at_cap_base"], "n_eligible_for_rung": counts["n_eligible"], "ladder_exit": items["exit"],
                       "base_reading": {"acc": counts["acc_base"], "correct": counts["correct_base"], "n_scored": counts["n_scored_base"]},
                       "ceiling_reading": ({"n": counts["n_ceil_drawn"], "n_scored": counts["n_scored_ceil"], "correct": counts["correct_ceil"], "still_at_cap": counts["n_ceiling"]} if ceil else None),
                       "loop_rule": (base["cap_ladder"].get("rules") or {}).get("degenerate"), "at_cap_rule": (base["cap_ladder"].get("rules") or {}).get("at_cap"),
                       "eligible_rule": (base["cap_ladder"].get("rules") or {}).get("trigger"), "producer_record": producer_record, "producer_record_version": producer_version,
                       "producer_record_agrees": True,
                       "producer_off_task": ((tn_row.get("off_task") or {}).get("counts") if tn_row is not None and isinstance(tn_row.get("off_task"), Mapping) else None),
                       "producer_off_task_share_structural": ((tn_row.get("off_task") or {}).get("share_off_task_structural") if tn_row is not None and isinstance(tn_row.get("off_task"), Mapping) else None),
                       "unresolved_definition": "item-wise: an at-cap item with no answer at its final draw — a loop never re-drawn, a truncated item the ladder did not escalate, "
                                                "or an item still at cap at the ceiling (the completion-cap rule, D476, applied item-wise)"}
        blk = self.delta_block(ref, model_id)
        extra: Dict[str, Any] = {
            "document": "paired_allmodels", "declared_role": ROLE, "standup_tag_legacy": tag, "panel": panel, "model_key": model_key,
            "seed_replicate_label": None, "fit_seed": None, "inspect_seed_base": None, "decode_seed_policy": G.INSPECT_SEED_POLICY_DEFAULT, "decode_seed_rule_text": seed_rule_text,
            "trial_rule": G.DEFAULT_SEED_TRIAL_RULE, "escalation_state": state, "token_cap_ladder": list(LADDER), "final_cap": final_cap,
            "ceiling_item_ids": sorted(items["ceiling_ids"]), "n_ceiling_items_merged": len(items["ceiling_ids"]),
            "store_uri_base": LC.rel_home(base["dir"]), "store_uri_ceiling": (LC.rel_home(ceil["dir"]) if ceil else None),
            "store_key_base": base["store_key"], "store_key_ceiling": (ceil["store_key"] if ceil else None), "sampling_dir": sampling_dir, "thinking_spelling": thinking,
            "store_pins": ({str(CAP): pins_read[str(CAP)], str(CEILING): pins_read[str(CEILING)]} if pins_read else None), "store_pins_record": (pins_read["record"] if pins_read else None),
            "tree_attestation": G.TREE_ATTESTATION_TARGET + ("; the store pins agree" if pins_read else "; no store pins beside this leg"), "producer_key_rule": PRODUCER_KEY_RULE,
            "accepted_attempt": {str(CAP): base["attempt"].get("attempt_id"), str(CEILING): (ceil["attempt"].get("attempt_id") if ceil else None)},
            "attempts_before_accepted": {str(CAP): [str(a.get("attempt_id")) for a in (base["ladder_state"].get("attempts") or []) if isinstance(a, Mapping) and str(a.get("result")) != "COMPLETE"]},
            "cluster_job_ids": {str(CAP): job, str(CEILING): ceiling_job}, "job_names": {str(CAP): base["attempt"].get("job_name"), str(CEILING): (ceil["attempt"].get("job_name") if ceil else None)},
            "node_by_cap": {str(CAP): base["attempt"].get("node"), str(CEILING): (ceil["attempt"].get("node") if ceil else None)},
            "harness_config_hash_by_cap": {str(CAP): man.get("harness_config_hash"), str(CEILING): (ceil["manifest"].get("harness_config_hash") if ceil else None)},
            "served_window_by_cap": {str(CAP): LC.int_or_none(geometry.get("max_model_len")), str(CEILING): LC.int_or_none(rgeometry.get("max_model_len"))},
            "served_max_num_seqs_by_cap": {str(CAP): LC.int_or_none(geometry.get("max_num_seqs")), str(CEILING): LC.int_or_none(rgeometry.get("max_num_seqs"))},
            "three_numbers": three_block, "n_items": n_items, "n_items_declared_by_manifest": (n_items if subset is not None else None),
            "item_state_counts": {"correct": correct, "incorrect": n_scored - correct, G.ITEM_LABEL_AT_CAP_UNRESOLVED: counts["n_at_cap_unresolved"], G.ITEM_LABEL_UNSCORED: counts["n_unscored_below_cap"],
                                  "scored_at_cap": counts["n_scored_at_cap"]},
            "at_cap_classes": {"loop": counts["n_loops_unresolved"], "ceiling": counts["n_ceiling"], "not_escalated": counts["n_not_escalated"]},
            "served_model_id_hf": served_hf, "served_model_revision": rev or None, "serving_identity": ident, "served_tree_sha": tree,
            "object_sha16": str(fitp.get("object_sha") or "") or None, "object_attested_sha256": obj.object_id, "object_attestation": ofacts,
            "fit_model_id_hf": fitp.get("model_id"), "fit_model_id": fit_model_id, "fit_transfer": fitp.get("transfer"), "transfer_kind_derivation": tk[2],
            "steer_positions_legacy": man.get("steer_positions"), "producer_cell_id_legacy": okey, "producer_slug_legacy": slug, "producer_view_legacy": panel,
            "inspect_ai_version": frameworks.get("inspect_ai"), "inspect_evals_version": frameworks.get("inspect_evals"), "framework_legacy": gs.get("framework"),
            "decode_policy_ref_legacy": (policy.policy_id if policy else None), "decode_policy_reader_match": ("exact by the stated sampling numbers" if policy else policy_why),
            "sampling_mode_legacy": gs.get("sampling_mode") or base["agg_extra"].get("sampling_mode") or (man.get("sampling") or {}).get("mode") if isinstance(man.get("sampling"), Mapping) else gs.get("sampling_mode"),
            "generation_state": "generated; rule-scored at generation (inspect_evals gsm8k numeric match), no judge", "source_legacy": SOURCE,
            "oos_tuple": {str(CAP): man.get("oos_experiment"), str(CEILING): (ceil["manifest"].get("oos_experiment") if ceil else None)},
            "instrument_minted": None, "instrument_rule": INSTRUMENT_RULE, "mirror_provenance": (LC.rel_home(prov_inp.uri) if prov_inp is not None else None),
            "producer_summary": (LC.rel_home(summary_inp.uri) if summary_inp is not None else None),
            "declaration": ({k: decl.get(k) for k in ("_path", "_schema", "_sha256", "index", "tag", "job_id", "role")} if decl is not None else None),
            "declaration_state": ("present" if decl is not None else "absent"),
            "cluster_root_legacy": f"{CLUSTER_ROOT_REL}/{CONTROLS_HEAD}/{tag}/{slug}/{sampling_dir}",
            "anchor_state": ANCHOR_STATE, "in_frame_unsteered_reference": blk["anchor"], "why_no_run_contrast": NO_CONTRAST_WHY,
            "coexist_ruling": ofacts["coexist_ruling"],
        }
        lead = f"GSM8K paired-200 thinking {thinking}, {tag} {okey} (the paired_allmodels document: default Inspect seed, one trial per model; role {ROLE})"
        notes = (f"{lead}: {correct}/{n_scored} correct over scored = {acc:.4f}" if acc is not None else f"{lead}: no scored item")
        notes += (f"; {counts['n_at_cap_unresolved']}/{n_items} at the cap and unresolved (neither hit nor miss: {counts['n_loops']} loops, {counts['n_ceiling']} still at cap at {CEILING:,}, "
                  f"{counts['n_not_escalated']} truncated at {CAP:,} without a ceiling)"
                  + (f", {counts['n_unscored_below_cap']} unscored below the cap (unterminated think block)" if counts["n_unscored_below_cap"] else "")
                  + f"; correct/{n_items} = {bound_lo:.3f} is the BOUND, not the value; escalation_state {state}"
                  + (f", {len(items['ceiling_ids'])} item(s) merged from the {CEILING:,} ceiling store" if ceil else "")
                  + f"; no `none` in the document: the in-frame unsteered anchors ride extra.in_frame_unsteered_reference ({ANCHOR_STATE})")
        try:
            payload = LC.run_payload_base(seeds, run_id=rid, producer_key=producer_key, object_key=okey, view_legacy=tag, cell_id_legacy=okey, obj=obj, eval_id=EVAL_ID,
                                          instrument_row=inst_row, instrument_seed_id=INSTRUMENT_SEED, lifecycle_state="generated", when_utc=when_final,
                                          when_source="ladder_state.json: the accepted attempt's ended (the final store's; the base attempt's started for queued / started)",
                                          token_cap_tokens=final_cap, judge_id=None, judge_seed_id=None, substrate=substrate, lane_source_legacy=SOURCE,
                                          store_uri=LC.rel_home(base["dir"]), instrument_match=inst_match, extra=extra, notes=notes)
        except KeyError as exc:
            raise G._PairRefusal(f"run row: {exc}", mfile)
        fc = payload["field_confidence_json"]
        payload.update(producer_session=PRODUCER_SESSION, trial_index=0, seed_base=None, queued_utc=when_base_start, started_utc=when_base_start,
                       n_items_planned=n_items, n_items_generated=n_items, n_items_judged=n_scored, model_revision=(rev if _SHA1.match(rev) else None),
                       intervention_hash=man.get("intervention_hash"), harness_config_hash=man.get("harness_config_hash"),
                       env_digest=(next(iter(base["consts"].get("env_digest") or []), None)), cluster_job_id=job, job_name=base["attempt"].get("job_name"),
                       node_hostname=base["attempt"].get("node"), serving_identity=ident, token_cap_rung_index=LADDER.index(final_cap), token_cap_ladder_json=list(LADDER),
                       decode_seed_rule=G.DECODE_SEED_RULE, decode_sent_json=sent, decode_temperature_sent=sent.get("temperature"), decode_top_p_sent=sent.get("top_p"),
                       decode_top_k_sent=(int(sent["top_k"]) if sent.get("top_k") is not None else None), code_sha=tree, transfer_kind=tk[0],
                       harness_version=(f"inspect-ai {frameworks.get('inspect_ai')} / inspect_evals {frameworks.get('inspect_evals')} ({gs.get('framework')})"
                                        if frameworks.get("inspect_ai") else payload["harness_version"]),
                       context_window_tokens=LC.int_or_none(geometry.get("max_model_len")),
                       store_completion_artifact_uri=final["files"]["gen_stats.json"].uri, store_completion_sha256=final["files"]["gen_stats.json"].sha256)
        if LC.int_or_none(geometry.get("max_num_seqs")) is not None:
            payload.update(served_max_num_seqs=LC.int_or_none(geometry.get("max_num_seqs")), served_max_num_seqs_source="attested")
            fc["served_max_num_seqs"] = LC.confidence_entry("recorded", f"manifest.json:geometry.max_num_seqs (source {geometry.get('source')}); the ceiling store's rides extra.served_max_num_seqs_by_cap")
        if policy is not None:
            payload["decode_source"] = policy.decode_source
            fc["decode_source"] = LC.confidence_entry("recorded", f"the registered policy row {policy.policy_id} the sent numbers equal (its decode_source)")
        else:
            fc["decode_source"] = LC.unverified(policy_why)
        fc["transfer_kind"] = LC.confidence_entry("derived", f"manifest.json:fit_provenance (model_id {fit_hf!r} -> {fit_model_id}, transfer {fitp.get('transfer')!r}) vs the served model "
                                                             f"{model_id}: {tk[2]}; never the standup spelling's ({seeds.model_for_view_legacy(tag).get('transfer_kind')!r})")
        fc["code_sha"] = LC.confidence_entry("recorded", "manifest.json:producer.tree_sha = ladder_state.json's accepted attempt sha (no store pins beside this document)")
        fc["model_revision"] = LC.confidence_entry("recorded", "manifest.json:model_revision (= models.yaml hf_revision)")
        fc["cluster_job_id"] = LC.confidence_entry("recorded", "manifest.json:row_constants.slurm_job_id = ladder_state.json's accepted attempt (the base store's job; the ceiling's rides extra.cluster_job_ids)")
        fc["node_hostname"] = LC.confidence_entry("recorded", "ladder_state.json: the accepted attempt's node (the base store's)")
        fc["token_cap_tokens"] = LC.confidence_entry("recorded", f"the ladder's final state: {final_cap} ({state}); gen_stats.json:max_tokens of the final store")
        fc["seed_base"] = LC.confidence_entry("derived", "the default Inspect seed rule (manifest.json:seeds.inspect carries the rule and no replicate block): no seed base")
        fc["decode_seed_rule"] = LC.confidence_entry("derived", f"one seed per item: {seed_rule_text} (k = 1, so the closure is the item; the default rule has no base, "
                                                                "extra.decode_seed_policy default; the rule's text extra.decode_seed_rule_text)")
        fc["trial_index"] = LC.confidence_entry("derived", G.DEFAULT_SEED_TRIAL_RULE + f"; manifest.json:seeds.inspect carries no replicate block ({seed_rule_text}); one trial per model")
        fc["decode_sent_json"] = LC.confidence_entry("recorded", "gen_stats.json (the client's sampler)")
        fc["instrument_match"] = LC.confidence_entry("derived", f"the sent numbers compared to the seed instrument {INSTRUMENT_SEED}'s decode columns bound to {model_id} (WS-B0's per-model "
                                                                f"row, the appendix column's instrument): {inst_match}")
        fc["instrument_id"] = LC.confidence_entry("derived", f"seed instrument {INSTRUMENT_SEED} bound to {model_id} (WS-B0's per-model expansion; extra.instrument_seed_id); nothing minted "
                                                             "(extra.instrument_rule)")
        fc["thinking_mode"] = LC.confidence_entry("recorded", f"gen_stats.json:thinking {gs.get('thinking')!r} (= the model's spelling {thinking!r}: models.yaml thinking_channel)")
        fc["harness_version"] = LC.confidence_entry("recorded", "manifest.json:frameworks (inspect_ai / inspect_evals) and gen_stats.json:framework")
        fc["context_window_tokens"] = LC.confidence_entry("recorded", "manifest.json:geometry.max_model_len (the base store's; the ceiling store's rides extra.served_window_by_cap)")
        fc["n_items_planned"] = LC.confidence_entry("derived", f"the instrument's item set (n_items {n_items}); gen_stats.json:n and the outcome record's n_attempted_items agree")
        fc["n_items_generated"] = LC.confidence_entry("recorded", "the outcome record: every attempted item has a stop reason (scored, or at the cap)")
        fc["n_items_judged"] = LC.confidence_entry("derived", "the SCORED items at the ladder's final state (the user's rule: unresolved items are neither hit nor miss); rule-scored at generation")
        fc["queued_utc"] = fc["started_utc"] = LC.confidence_entry("recorded", "ladder_state.json: the base store's accepted attempt started")
        fc["generated_utc"] = LC.confidence_entry("recorded", "ladder_state.json: the final store's accepted attempt ended")
        fc["object_id"] = LC.confidence_entry("recorded", obj.source)
        if man.get("steer_positions") in V.APPLIED_POSITIONS_MODES:
            payload["applied_positions_mode"] = str(man["steer_positions"])
            fc["applied_positions_mode"] = LC.confidence_entry("recorded", "manifest.json:steer_positions")
        payload["thinking_mode"] = thinking
        batch.add("run", payload, producer, now)
        facts["runs"] += 1
        if ceil is not None:
            facts["ceiling_merged"] += 1
        # NO state event (the seed-replicate precedent): the harvester's rule-scored completion stamps `ingested` off the bound row's n
        comp_id = G.merge_computation_id()
        code_sha = str(producer.get("code_sha") or "")
        batch.add("computation", {"computation_id": comp_id, "algorithm": G.MERGE_ALGORITHM, "version": G.MERGE_VERSION, "seed_provenance": G.MERGE_SEED_PROVENANCE,
                                  "code_sha": (code_sha if _SHA1.match(code_sha) else "unverified"),
                                  "notes": "the ceiling store's items replace the base store's item-wise (never averaged); correct / scored recounted over the merged outcomes"},
                  producer, now)
        rows_uri = LC.rel_home(base["files"]["rows.jsonl"].uri)
        lo, hi = IC.wilson(correct, n_scored) if n_scored else (None, None)
        batch.add("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": G.HEADLINE_METRIC, "threshold": -1.0, "computation_id": comp_id,
                                 "producer_metric_spelling": "acc", "value": acc, "k": correct, "n": n_scored, "ci_lo": lo, "ci_hi": hi, "ci_method": ("wilson_95" if n_scored else None),
                                 "n_abstentions": None, "n_incomplete": counts["n_unscored_final"], "computed_by": "registry", "computed_utc": when_final, "source_uri": rows_uri,
                                 "extra": {"role": "THE VALUE: correct over SCORED at the ladder's final state (the user's rule: degeneration is scored separately)", "declared_role": ROLE,
                                           "recounted_from": "the base outcome record with the ceiling store's items merged item-wise", "base_reading": three_block["base_reading"],
                                           "ceiling_reading": three_block["ceiling_reading"], "n_unresolved_neither_hit_nor_miss": counts["n_unresolved"],
                                           "n_unscored_final": counts["n_unscored_final"], "equals_gen_stats_acc": (ceil is None), "producer_record": producer_record,
                                           "delta_block": blk}}, producer, now)
        batch.add("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": G.HEADLINE_METRIC, "threshold": -1.0, "computation_id": "",
                                 "producer_metric_spelling": ("bound_correct_over_attempted" if tn_row is not None else ("acc_over_attempted_final" if fs_row is not None else None)),
                                 "value": bound_lo, "k": correct, "n": n_items, "ci_lo": bound_lo, "ci_hi": bound_hi, "ci_method": G.BOUND_CI_METHOD,
                                 "n_abstentions": None, "n_incomplete": counts["n_unresolved"], "computed_by": "producer", "computed_utc": when_final,
                                 "source_uri": (producer_record or LC.rel_home(base["files"]["gen_stats.json"].uri)),
                                 "extra": {"role": f"THE BOUND, never the value: correct over the {n_items} attempted; the interval's lower end counts the unresolved items wrong, the upper end right",
                                           "declared_role": ROLE, "n_unresolved_neither_hit_nor_miss": counts["n_unresolved"], "n_unresolved_ladder_only": counts["n_unresolved_ladder_only"],
                                           "n_at_cap_not_escalated": counts["n_not_escalated"], "n_unscored_below_cap_counted_as_miss": counts["n_unscored_below_cap"],
                                           "unresolved_definition": three_block["unresolved_definition"],
                                           "producer_spellings": ({"bound_correct_over_attempted": tn_row.get("bound_correct_over_attempted")} if tn_row is not None else
                                                                  ({"acc_over_attempted_final": fs_row.get("acc_over_attempted_final"), "n_unresolved_final": fs_row.get("n_unresolved_final")} if fs_row is not None else None)),
                                           "producer_counts": ("the producer's own counts: gen_stats.json n / n_scored / n_at_cap (and controls_summary.json when present) over the attempted "
                                                               "items, the correct items the rows the producer scored; no producer record spells this ratio, so the row carries no producer spelling"),
                                           "metric_role_expected": "secondary (a blank computation_id ranks after the value's computed row at one identity)",
                                           "delta_block": blk}}, producer, now)
        rows_sha = {CAP: base["files"]["rows.jsonl"].sha256, CEILING: (ceil["files"]["rows.jsonl"].sha256 if ceil else None)}
        for iid in sorted(items["final"]):
            it = items["final"][iid]
            label = None if it["value"] is not None else (G.ITEM_LABEL_UNSCORED if it["at_cap_class"] == G.UNSCORED_BELOW_CAP_CLASS else G.ITEM_LABEL_AT_CAP_UNRESOLVED)
            iextra = {"final_cap": it["cap"], "source_store": it["source"], "stop_reason": it["stop_reason"], "completion_tokens": it["completion_tokens"], "decode_seed": it["seed"],
                      "at_cap_class": (it["at_cap_class"] if it["at_cap_class"] in G.AT_CAP_CLASSES else None), "scored_at_cap": bool(it["value"] is not None and it["at_cap"]),
                      "unscored_reason": (G.UNSCORED_BELOW_CAP_CLASS if it["at_cap_class"] == G.UNSCORED_BELOW_CAP_CLASS else None),
                      "think_block_unterminated": it.get("think_block_unterminated"), "unparseable": it.get("unparseable"),
                      "base_value_replaced": it.get("base_value_replaced"), "base_at_cap": it.get("base_at_cap"), "text_sha256": it.get("text_sha256")}
            batch.add("item_result", IC.item_result_payload(rid, iid, G.HEADLINE_METRIC, it["value"], label, it["created"] or when_final, judgment_id="", trial_index=0, extra=iextra),
                      producer, now)
            gen_art, gen_off = (rows_sha[it["cap"]], it["line"]) if it["line"] is not None else (None, None)
            batch.add("item_flags", IC.item_flags_payload(rid, iid, gen_art, gen_off, at_token_cap=(1 if it["at_cap"] else 0), finish_reason=it["stop_reason"],
                                                          response_tokens=it["completion_tokens"], decode_seed=(str(it["seed"]) if it["seed"] is not None else None), judgment_id="",
                                                          trial_index=0, degenerate_rep_ratio_loop=(1 if it.get("classifier_loop") else (0 if it["degenerate"] is False else None)),
                                                          rep_ratio=it.get("rep_ratio"), tail_loop=it.get("tail_loop"),
                                                          extra={"at_token_cap_source": "the outcome record's at_cap (stop_reason == max_tokens at the request's cap; never re-tokenized)",
                                                                 "generation_pointer": ("rows.jsonl of the store the item's final draw came from (sha, line)" if it["line"] is not None
                                                                                        else "no row: an unscored item's generation stays in the store's private/ spool on the cluster"),
                                                                 "outcome_record": LC.rel_home((ceil if it["source"] == "ceiling" else base)["files"]["outcomes"].uri), "tail_rep_ratio": it.get("tail_rep_ratio")}),
                      producer, now)
        src_uri = LC.rel_home(base["top"]["cap_ladder.json"].uri)
        rules = base["cap_ladder"].get("rules") if isinstance(base["cap_ladder"].get("rules"), Mapping) else {}
        measured = {
            G.RAW_AT_CAP_MEASURE: dict(numerator=counts["n_at_cap_final"], denominator=n_items, unit="items", threshold_params={"cap_final": final_cap, "ladder": list(LADDER)},
                                       extra={"rule": rules.get("at_cap"), "at_ladder_final_state": True,
                                              "classes": {"loop": counts["n_loops_unresolved"], "ceiling": counts["n_ceiling"], "not_escalated": counts["n_not_escalated"], "scored_at_cap": counts["n_scored_at_cap"]},
                                              "n_at_cap_unresolved": counts["n_at_cap_unresolved"], "n_unscored_below_cap_not_counted": counts["n_unscored_below_cap"],
                                              "n_at_cap_8192_before_ceiling": counts["n_at_cap_base"], "source": src_uri}),
            G.LOOP_MEASURE: dict(numerator=counts["n_loops"], denominator=n_items, unit="items", threshold_params={"rep_ratio_gt": 0.5, "tail_tokens": 200, "cap": CAP},
                                 extra={"rule": rules.get("degenerate"), "periodic_block_rule": rules.get("periodic_block"), "class": "loop: at cap at 8,192 AND degenerate by cap_ladder.json — never re-drawn", "source": src_uri}),
        }
        notes_nm = {"classifier": "the paper's GSM8K reader files the loop count under `classifier`; this store's classification is the rep_ratio rule (rep_ratio_loop)",
                    "contentless": "no contentless reading in the ladder gate's classification", "zlib_collapse": "no zlib reading in the ladder gate's classification",
                    "union5": "the five-detector union is not computed for these stores (the degeneracy-union reader's input, not here)",
                    IC.AT_CAP_RETOKENIZED: "the ladder gate reads the provider's stop reason, never a re-tokenized length"}
        for row in IC.declared_degeneration_rows(rid, "", measured, source_uri=src_uri, computed_utc=when_final, not_measured_notes=notes_nm):
            batch.add("run_degeneration", row, producer, now)
        batch.add("run_degeneration", IC.degeneration_payload(rid, G.COHERENT_MEASURE, "measured", judgment_id="", numerator=counts["n_ceiling"] + counts["n_not_escalated"], denominator=n_items,
                                                               unit="items", threshold_params={"cap_final": final_cap}, computed_utc=when_final, source_uri=src_uri,
                                                               extra={"class": "unresolved and not a loop: still at cap at the ceiling, or truncated at 8,192 without a ceiling store",
                                                                      "classes": {"ceiling": counts["n_ceiling"], "not_escalated": counts["n_not_escalated"]}, "source": src_uri}), producer, now)
        facts["pairs"][okey] = {"run_id": rid, "instrument_id": instrument_id, "instrument_minted": False, "object_id": obj.object_id, "staged_id": obj.staged_id,
                                "transfer_kind": tk[0], "fit_model_id": fit_model_id, "acc_over_scored": acc, "correct": correct, "n_scored": n_scored, "acc_over_attempted": bound_lo,
                                "upper_bound": bound_hi, "n_unresolved": counts["n_unresolved"], "n_unresolved_ladder_only": counts["n_unresolved_ladder_only"], "n_loops": counts["n_loops"],
                                "n_ceiling": counts["n_ceiling"], "n_not_escalated": counts["n_not_escalated"], "n_unscored_below_cap": counts["n_unscored_below_cap"],
                                "n_scored_at_cap": counts["n_scored_at_cap"], "ceiling_items_merged": len(items["ceiling_ids"]), "escalation_state": state, "ladder_exit": items["exit"],
                                "instrument_match": inst_match, "decode_policy": (policy.policy_id if policy else None), "jobs": {str(CAP): job, str(CEILING): ceiling_job}, "tree": tree,
                                "trial_index": 0, "producer_key": producer_key, "store_key_base": base["store_key"], "thinking": thinking, "sampling_dir": sampling_dir,
                                "declaration_bound": decl is not None, "roster_staged": ofacts.get("roster_staging") is not None, "staging_manifest_bound": ofacts.get("staging_manifest") is not None,
                                "producer_record_version": producer_version, "store_pins_bound": pins_read is not None, "three_numbers_bound": tn_row is not None, "final_state_bound": fs_row is not None,
                                "coexist_prefix_match": ofacts["coexist_ruling"]["matches_object_prefix"], "n_items": n_items,
                                "in_frame_unsteered_run_ids": list(ref.get("in_frame_unsteered_run_ids") or []), "anchor_reference_state": ref.get("state")}


def factory() -> Gsm8kPairedAllmodelsAdapter:
    return Gsm8kPairedAllmodelsAdapter()


register(NAME, factory)
