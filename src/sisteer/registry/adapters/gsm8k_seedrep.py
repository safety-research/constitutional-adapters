"""The batch-64 three-seed Qwen family's GSM8K paired-200 stores -> registry (source ``gsm8k_seedrep``).

WHAT WAS RUN: the scored GSM8K stores of the batch-64 three-seed Qwen family are the registry's to ingest. Slurm on cluster A at tree
``b54f5b9c`` (jobs 585909 / 585910 / 585911, rungs 585997 / 586005), driven by the
matrix ``configs/sdfbase/gsm8k/matrix_qwen_seedrep.json`` (24 rows: seed tags ``qwen35_4b_s1`` / ``_s2`` / ``_s3`` x the unsteered anchor, the
five admitted staged objects and the two prompted system instructions; the composed ``difference_relu`` object is NOT among them). Every row
is the paper's ONE GSM8K instrument — inspect_evals' stock ``gsm8k`` task as ``inspect/gsm8k_paired200`` over the 200 Platinum-paired items,
10-shot, k = 1, thinking ON, the Qwen thinking-card decode row (T 0.6 / top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 0), cap 8,192 with
ONE 16,384 rung above 1 % truncated — served LoRA-natively (``proxy_vllm+lora+interleaved``; the unsteered and prompted keys ride the passthrough
route) on ``Qwen/Qwen3.5-4B @ 851bf6e8``, under the Inspect REPLICATE seed rule at seed base s (the tag's index): ``manifest.json:seeds.inspect``
spells ``sha256(f'{seed_base}|inspect-replicate|{item_id}|{sample_idx}')[:4] big-endian mod 2^31-1`` (``contracts.SEED_RULE_INSPECT_REPLICATE``).

THE PAIRED TAGS: ``qwen35_4b_s<N>p`` is a STANDUP REPLICATE of seed tag ``s<N>`` — the same objects and the same Inspect replicate
seed base N, re-served on cluster A — carrying two object keys: its OWN native ``none`` (base store at tree ``9865993c7``, arrays
589570 / 589607 / 589634; ceiling store at ``c03d60a0e``) and the composed ``difference_relu`` object ``constv2-mix-full__psr_ce_all_minus_ctl``
(``paired_eb64s<N>__…`` in the roster's ``s<N>p`` selection; tree ``c03d60a0e``, array 590975; no ceiling store — every at-cap
item is a loop, exit ``triggered_no_eligible``). The served object is attested three ways (``fit_provenance.object_sha``, the rows' aggregate
``object_artifact_sha256``, ``provenance.artifact_sha256``) and each must be the roster / pins' sha or the pair refuses. The paired ``none``'s base
store REUSES the seed tag's store key byte for byte, so a paired tag's producer key is ``<tag>/<base store key>`` (:data:`PAIRED_KEY_RULE`; the
seed tags keep the bare key, their ids are landed). Every steered run of a tag files a ``run_contrast`` against ITS OWN tag's ``none``
(``anchor_kind same_standup_none``; the composed object is paired against the paired ``none``, never ``s<N>``'s), item-wise over the items scored
on both sides: pp difference, McNemar's exact p, Cohen's h. The paired run carries ``extra.paired_of`` / ``extra.standup_replicate_of_view`` (the
roster's ``paired_of`` is checked when it carries one) and ``extra.oos_tuple`` (the manifest's ``oos_experiment``, ``gsm8k_paired200_qwen35_4b_native_paired_8192``
on the composed legs).

THE MIRROR (``run-state/full22/gsm8k_seedrep/``, rsync'd from the producer's root's ``controls_qwen_seedrep/`` on cluster A, the ``private/``
generation spools left behind): ``<tag>/<object dir>/thinking_card/`` is a BASE store (cap 8,192) and ``<tag>/<object dir>/thinking_card__rung16384/``
its RUNG store (cap 16,384, present for the 13 (tag, object) pairs whose ladder escalated; each holds only that pair's eligible items, 1-5). The
object dir is the producer's slug (``constv2-mix-full--lora_r4_ce``: ``--`` for the harness id's ``__``). Beside them the two records the
producer wrote when the runs finished: ``store_pins_qwen_seedrep.json`` (``g8k_store_pins/2``: ``<model>/<seed>/<object key>@<cap> -> tree``, one
key per store that ran) and ``three_numbers_qwen_seedrep.json`` (``g8k_three_numbers/1``: the producer's own count of every number this reader
re-derives). What is read, every file named:

* ``thinking_card/cells/<store key>/manifest.json`` — served identity (``model_id`` / ``model_revision`` / ``serving_identity`` /
  ``fit_provenance.object_sha`` a 16-hex prefix of the object / ``resolved_config.system_sha`` the prompt's / ``producer.tree_sha`` /
  ``row_constants.slurm_job_id`` / ``routing_table`` / ``harness_config_hash`` / ``intervention_hash`` / ``steer_positions`` / ``geometry`` /
  ``seeds.inspect.replicate.seed_base`` / ``frameworks``); a rung manifest adds ``item_subset`` (the eligible ids it re-drew);
* ``…/cells/<store key>/outcomes/inspect__gsm8k_paired200.json`` — the per-item OUTCOME record over EVERY attempted item (``at_cap`` /
  ``scored`` / ``value`` / ``stop_reason`` / ``completion_tokens`` / ``seed``): the item universe (``rows.jsonl`` holds scored items only);
* ``…/cells/<store key>/rows.jsonl`` — the scored ``acc`` rows (value, the generation text, the item's seed, the row's ``slurm_job_id`` /
  ``env_digest``) plus the producer's aggregate rows; ``gen_stats.json`` — the producer's aggregate (``acc`` over SCORED, ``n`` 200, ``n_scored``,
  ``n_unscored``, ``n_at_cap``, the sent decode numbers, the frameworks);
* ``thinking_card/cap_ladder.json`` (``cap_ladder/1``, base only) — the ladder gate's per-item classification (``item_facts``: ``at_cap`` /
  ``degenerate`` / ``kind`` / ``rep_ratio`` / ``tail_rep_ratio`` / ``eligible`` / ``text_sha256``), its ``eligible_ids`` / ``non_terminating_ids``,
  the ``exit`` it declared and the RULES it applied (at-cap = ``stop_reason == max_tokens``; loop = whitespace-token 4-gram ``rep_ratio > 0.5``
  on the text or its last 200 tokens; eligible = at cap minus degenerate; trigger above 1 %);
* ``…/ladder_state.json`` (``g8k_ladder_state/1``) — the accepted generation attempt (its job id, node, cluster, tree, start / end), ``RESULT``
  (its last line must read ``result=COMPLETE``); ``evidence.json`` / ``resolved_sampler.json`` / ``request_evidence.jsonl`` / ``usage_ledger.jsonl``
  are registered as the store's artifacts (the sampler the client sent, per request) and not otherwise read.

REGISTRY GRAIN: ONE run per (model, seed tag, object key) at the LADDER'S FINAL STATE. The rung's items REPLACE the base store's items
ITEM-WISE (never a second run, never an average of two stores): ``run_id = identity.run_id(sdfbase_full22, <the BASE store key —
``<object>__controls__t0__a0__<ihash8>__<chash8>``, the store's own NAME, never its path>, <object key>, gsm8k, <the canonical seeded
instrument bound to qwen35_4b_it>)``, ``token_cap_tokens`` the final cap the ladder reached (16,384 with a rung, else 8,192), ``token_cap_rung_index``
its index in the ladder ``[8192, 16384]``, ``extra.escalation_state`` in relu_paired's vocabulary — ``final_at_ceiling`` when the 16,384 store's items were merged (that
step is the ladder's ceiling: ``cap_ladder.json`` declares no step above it), ``final_at_base`` otherwise — and ``extra.ceiling_item_ids`` /
``extra.store_uri_base`` / ``extra.store_uri_ceiling`` naming what came from where.
The campaign is ``sdfbase_full22`` (:data:`lineage_common.CAMPAIGN`): these are the campaign's own objects on its canonical instrument, the way
``seedrep_cluster`` files the same objects' statics and MMLU; the Modal premints under ``seedrep_battery`` were minted from another matrix
and are expected to differ (a separate ruling retires them).

THE THREE NUMBERS (standing rule — degeneration is scored separately):

1. THE VALUE — ``run_metric acc``, the run's headline: correct / SCORED at the ladder's final state, ``k`` the correct items, ``n`` the scored
   items, ``n_items_judged`` = ``n``. Filed under the reader's ladder-merge computation (:data:`MERGE_ALGORITHM`, a non-blank ``computation_id``,
   ``computed_by registry``): for a pair without a rung it equals ``gen_stats.json:acc`` exactly (checked); with a rung it is the merged recount
   the producer's ``three_numbers`` record spells as ``correct_after_rung`` over ``200 - n_unresolved`` (checked).
2. THE BOUND — a SECOND ``run_metric acc`` row on the same run: correct / 200 ATTEMPTED (``k`` correct, ``n`` 200, the producer's own spelling
   ``acc_over_attempted_after_rung`` as ``producer_metric_spelling``, ``computed_by producer``, a BLANK ``computation_id``) with the interval the
   unresolved items leave open in ``ci_lo`` / ``ci_hi`` (``ci_method unresolved_items_interval``: the lower end counts them wrong, the upper end
   right). The frame's metric role (``v_paper_long.metric_role``, migration 0004) ranks a computed row before a blank one at one identity, so the
   value is the OWNER row the paper counts and the bound is SECONDARY evidence beside it — never the value. Its ``n = 200`` is also the row that
   reaches the item set's coverage in the harvester's rule-scored completion (``completion_requirements_hold`` reads ``MAX(n)`` over the headline
   rows): every attempted item is a hit, a miss or an unresolved item, so the run IS complete.
3. THE THIRD ITEM STATE — an item whose final draw ended at the token cap is NEITHER correct NOR incorrect: its ``item_result`` row carries
   ``value NULL`` and the label :data:`_items_common.ITEM_LABEL_AT_CAP_UNRESOLVED` (``at_cap_unresolved``), with ``extra.at_cap_class`` naming
   which: ``loop`` (degenerate by ``cap_ladder.json``, never re-drawn), ``ceiling`` (still at cap at 16,384), ``not_escalated`` (truncated at 8,192,
   eligible, but the ladder did not trigger). The run-level shares are ``run_degeneration`` measures over the 200 attempted items:
   ``at_token_cap_finish_reason`` = every item at the cap at the ladder's final state (the finish-reason reading), ``rep_ratio_loop`` = the
   loops, ``truncation_only`` = the unresolved items that are not loops; the other declared measures are ``not_measured`` naming why. ALL THREE
   classes are unresolved — the not-escalated items included (the producer widened the third state to them, 2026-09-21 09:23Z): the bound's
   interval is ``[correct / 200, (correct + loops + ceiling + not_escalated) / 200]``, ``n_unresolved_neither_hit_nor_miss`` is that item-wise
   count, and ``n_unresolved_ladder_only`` (loops + ceiling, the record's first definition) rides beside it.
   Two rarer states the outcome records carry are filed as what they are: an item SCORED at the cap (its think block closed and the scorer ran —
   its value counts; ``item_flags.at_token_cap 1``, ``extra.scored_at_cap``; the ladder re-draws it like any at-cap item) and an item UNSCORED
   below the cap (the think block never closed before EOS, the instrument's ``unterminated_policy``): ``value NULL`` under the paper reader's label
   ``unscored`` with ``extra.unscored_reason unterminated_below_cap`` — not the third state, since it ran out of nothing; it is outside ``scored``, a
   miss in the bound (the producer's ``n_unscored_below_cap_final``), and it enters no degeneration measure.

CROSS-CHECKS, every one a refusal of the (tag, object key) PAIR by name (:class:`lineage_common.RowRefusals`; the other pairs of the tag file):
the base manifest's tree must equal the store pin ``@8192`` and a rung's the pin ``@16384``, a rung store present iff its pin is; the outcome record,
``cap_ladder.json`` and ``rows.jsonl`` must agree item by item (200 distinct items; at cap iff unscored; the scored values); the rung's
``item_subset.ids`` must equal the base's ``eligible_ids`` and its outcome record cover exactly them; ``gen_stats.json`` must agree with the
recount (``n``, ``n_scored``, ``n_unscored``, ``n_at_cap``, ``acc``); ``cap_ladder.json:exit`` must be ``escalate`` iff a rung ran; the served model
must be the seed tag's (``qwen35_4b_it``, revision the model seed's), the serving identity the object kind's, the attested object sha (the rows'
aggregate ``object_artifact_sha256``, the manifest's 16-hex prefix, the prompt's ``system_sha``) the roster / pins' (``jobplan.object_id_for``, the
harness's own resolution) or ``si_texts.yaml``'s; ``seeds.inspect.replicate.seed_base`` must be the tag's index; and EVERY number of the producer's
``three_numbers`` record for the pair must equal the re-derivation. That record exists in three versions, told apart by the keys a cell carries
(:func:`record_version`): the mirror root's first write (07:05Z: ``acc_over_scored`` the base reading, ``n_unresolved_neither_hit_nor_miss``
= loops + ceiling, the bounds from it), the additive final-state write (09:14Z: ``n_scored_final`` / ``correct_final`` / ``acc_over_scored_final`` /
``headline``) and the item-wise re-derivation (09:23Z, ``g8k_final_state.py``: ``n_unresolved_neither_hit_nor_miss`` item-wise with
``n_unresolved_ladder_only`` beside it, ``n_unscored_below_cap_final``, ``n_rung_drawn``, ``n_rung_drawn_that_were_scored_at_base``, the bounds
widened). Each version is checked under ITS OWN definitions — the final-state keys when present, the base-reading derivation otherwise — and the
reader files the widened definition whatever the record's version (``extra.three_numbers.producer_record_version`` names which was read). The
producer's per-pair ``final_state_qwen_seedrep.json`` (14:40Z; keyed ``s<N>[p]/<object dir>``) is the SECOND cross-check record: a pair needs at
least one of the two, every one present must agree on every count it carries, and a pair only the per-pair record names (the paired tags, while
the three-number record stays at its 24 seed cells) is checked against it alone (``producer_record_version`` says so). A LOOP the scorer scored
(its think block closed at the cap — s1p's composed object has two) keeps its value: the classifier's loop count (``n_loops``, the ``rep_ratio_loop``
share) and the unresolved loops (``at_cap_classes.loop``) are two numbers, and ``n_unresolved_ladder_only`` keeps the producer's ``n_loops + ceiling``.
The reader never consults the live registry: it emits on the attested identity and lets ``blocked_records`` hold a run whose object row is absent.

DECODE: the sent numbers (``gen_stats.json``) are compared to the registered policy row ``qwen35_4b_it|gsm8k|on|thinking_card@v1`` (the committed
export) and to the instrument's decode columns (``instrument_match``); ``decode_policy_match`` stays the producer placeholder. ``decode_seed_rule``
is ``per_item`` — one seed per item, the closure the replicate rule hashes over at k = 1 — with ``seed_base`` the tag's index and the rule's exact
text on ``extra.decode_seed_rule_text``; the default Inspect rule (no base) files ``per_item`` with ``seed_base NULL``, so the pair tells them apart.
The unsteered and prompted keys name ONE object on every seed tag: as in ``seedrep_cluster`` they file as decode-seed TRIALS of one run key
(``trial_index`` = the seed index, ``extra.decode_replicate``); the fitted keys are one object per seed and stay at trial 0. Their coexistence in
the paper frame needs the roster's ``coexist`` ruling over ``lane_sources_legacy [gsm8k_seedrep]`` x the three tags (not a reader default).

THE ALL-POSITIONS OBJECT (v4, 2026-09-21). The Arditi all-positions refusal steer is ONE fit
(harness key ``arditi_allpos``, staged ``ar_c0p3__arditi_allpos``, object ``ba2212f0…``, operating point ``ar_c0p3``: c = 0.3 ``raw_coefficient``,
served natively at every position), and the standing rule — the two replication axes never mix — makes one fit drawn at three seed bases
three DECODE-SEED TRIALS of one object, never three fits (:data:`DECODE_TRIAL_RULE`, ``seedrep_cluster`` v8's rule, mirrored here). Two inputs:

* THE THREE SEED ROWS: ``<tag>/arditi_allpos/thinking_card[__rung16384]/`` under the mirror root for ``qwen35_4b_s1`` / ``_s2`` / ``_s3`` (the
  seed-replicate wave at the vendor thinking-card decode, thinking ON, the 8,192 -> 16,384 ladder, the replicate seed rule at base s). The key
  resolves through the ALL-POSITIONS roster / pins (:data:`ARDITI_ROSTER_REL` / :data:`ARDITI_PINS_REL`; ``jobplan.object_id_for`` at the rung the
  declaration names), never the seed-replicate roster, which stages no such key; its DECLARATION is the producer's eligibility record
  ``configs/sdfbase/gsm8k/arditi_rows_qwen_seedrep.json`` (:data:`ARDITI_SEED_ROWS_SCHEMA`; one row per seed tag: the staged id and object sha
  the pins must reproduce, the bank view ``ar_c0p3``, the out roots the mirror's directories must spell, the OOS tuples
  ``gsm8k_paired200_qwen35_4b_steer_allpos_<cap>`` the manifests' ``oos_experiment`` must equal, the role ``arditi_unpaired_value``) — REQUIRED,
  checked field by field, a disagreement refuses the pair by name. The store's attested sha (``fit_provenance.object_sha``, the rows' aggregate,
  ``provenance.artifact_sha256``) must be the pin's or the pair refuses. The three rows file as trials of ONE object on the seed views:
  ``trial_index`` = ``seed_base`` = the seed index, ``fit_seed`` unset, one object id across the views, ``extra.decode_replicate`` naming the trial
  and the object; the strength columns are the operating point through dose_grammar (``strength_coefficient`` 0.3, ``strength_unit``
  ``raw_coefficient``, ``strength_nominal`` 1.0 derived, ``is_operating_point`` 1, ``operating_point_id_legacy ar_c0p3__arditi_allpos``) and
  ``transfer_kind`` the rung's own legacy spelling (``instruct_direct``: the direction is fit on and served on the instruct checkpoint), so the
  paper keys agree to the letter with the ``arditi_allpos`` reader's rows on the same object. The instrument is the seed views' canonical
  GSM8K instrument (the one the fitted keys sit on; ``instrument_match`` from the sent numbers as everywhere), the store pins and the producer's
  count records are required as for every other key, the three numbers and the item-wise final state are the same rule, and the run is
  contrasted against the seed view's ``none`` as every steered key is (the declared role rides on the run and the contrast). Their coexistence
  in the paper frame needs the roster's ``coexist`` ruling (match: ``lane_sources_legacy [gsm8k_seedrep]`` x ``campaigns [sdfbase_full22]`` x
  ``model_ids [qwen35_4b_it]`` x ``object_ids [ba2212f0…]`` x ``eval_ids [gsm8k]`` x ``instrument_ids [<the canonical instrument>]`` x
  ``views_legacy [qwen35_4b_s1, _s2, _s3]``) — an explicit ruling, never a reader default.
* THE DEFAULT-SEED ROW (the all-positions matrix's own GSM8K row: cluster A 544615 / rung 544730, 2026-09-17, standup ``qwen35_4b_it``, the
  DEFAULT Inspect seed rule — no replicate block — thinking ON), mirrored byte for byte under the same root at
  ``arditi_allpos_paper_objects/controls_allpos/qwen35_4b_it/arditi_allpos/{thinking_card, thinking_card__rung16384}`` (+ ``PROVENANCE.md``;
  :data:`TARGET_MIRROR_DIR`). Its group is ``arditi_allpos_paper_objects/qwen35_4b_it``, its declaration the matrix's own row
  (:data:`ARDITI_MATRIX_REL`, the same fields), and it files as a single-object NON-seed target: ``view_legacy qwen35_4b_it`` (the
  ``arditi_allpos`` source's spelling), ``trial_index`` 0, ``fit_seed`` unset, ``seed_base`` NULL with ``decode_seed_rule per_item`` and
  ``extra.decode_seed_policy default`` (:data:`DEFAULT_SEED_TRIAL_RULE`), the same strength columns, and its OWN instrument MINTED from the
  record (:meth:`Gsm8kSeedrepAdapter._mint_target_instrument`: the canonical seed row with the store's harness configuration hash — the
  harness's own configuration identity, which the seed rule enters — its inspect versions, its served window, its ladder rule and its sent
  numbers, through ``identity.instrument_id``; marked on the run as ``extra.instrument_minted`` with the seed instrument's decode match beside
  it). The default seed rule is not the seed views' replicate rule, so the point is never forced onto their instrument and never reads as one of
  their trials. No store pins and no producer count record sit beside it: the tree is attested by ``ladder_state.json``'s accepted attempt
  (:data:`TREE_ATTESTATION_TARGET`; ``RESULT`` may list earlier failed attempts — the LAST line and the ACCEPTED attempt are read, never the first),
  the base reading is checked against ``gen_stats.json`` and the producer's ``controls_summary.json``, and the final state is the reader's
  item-wise merge alone (``producer_record_version`` :data:`RECORD_VERSION_NONE`; the bound row carries no producer spelling). No ``none`` sits
  beside it, so no same-standup contrast is formed — the roster pairs it. The seed views' rows are byte-identical to v3.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import decode_policy as DP
from .. import identity as I
from .. import jobplan as JP
from .. import vocab as V
from . import _items_common as IC
from . import declared_common as DC
from . import dose_grammar as DG
from . import lineage_common as LC
from . import relu_paired as RP
from . import seedrep_cluster as SC
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register
from .runner import source_id_for

NAME = "gsm8k_seedrep"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "gsm8k_seedrep"
OWNER = "Overflow"
PRODUCER_SESSION = "Overflow"
CAMPAIGN = LC.CAMPAIGN
ROOT_DEFAULT = os.path.join(LC.RS_DEFAULT, "gsm8k_seedrep")
MODEL_KEY = SC.MODEL_KEY
#: a seed view `qwen35_4b_s<N>` or its PAIRED standup replicate `qwen35_4b_s<N>p` (the paired views: the same seed base N,
#: re-served beside the composed difference_relu object; the view's own native `none` is its anchor)
TAG_RE = re.compile(rf"^(?P<model>{MODEL_KEY})_s(?P<seed>\d+)(?P<paired>p?)$")
EVAL_ID = "gsm8k"
TASK = "inspect/gsm8k_paired200"
#: the paper's ONE GSM8K instrument, bound per model by the dimensions loader
INSTRUMENT_SEED = "gsm8k_paired200_inspect_10shot_thinking_on_8192"
CAP, CEILING = 8192, 16384
LADDER: Tuple[int, ...] = (CAP, CEILING)
N_ITEMS = 200
BASE_DIR = "thinking_card"
CEILING_DIR = f"thinking_card__rung{CEILING}"
STORE_DIRS = (BASE_DIR, CEILING_DIR)
PINS_FILE = "store_pins_qwen_seedrep.json"
THREE_NUMBERS_FILE = "three_numbers_qwen_seedrep.json"
#: the producer's per-pair item-wise record (g8k_final_state.py; keyed `s<N>[p]/<object dir>`): a second cross-check record beside the
#: three-number one — a pair needs at least ONE of the two, and every one present must agree
FINAL_STATE_FILE = "final_state_qwen_seedrep.json"
SHARED_FILES = (PINS_FILE, THREE_NUMBERS_FILE, FINAL_STATE_FILE)
PINS_SCHEMA = "g8k_store_pins/2"
THREE_NUMBERS_SCHEMA = "g8k_three_numbers/1"
CAP_LADDER_SCHEMA = "cap_ladder/1"
LADDER_STATE_SCHEMA = "g8k_ladder_state/1"
OUTCOMES_FILE = "inspect__gsm8k_paired200.json"
#: the files of a store directory (``cap_ladder.json`` on the base store only) and of its ``cells/<store key>/`` directory
STORE_TOP_FILES = ("RESULT", "ladder_state.json", "cap_ladder.json", "evidence.json", "resolved_sampler.json", "request_evidence.jsonl", "usage_ledger.jsonl")
STORE_KEY_FILES = ("gen_stats.json", "manifest.json", "rows.jsonl", "evidence.json")
ROLE_OF_FILE = {"rows.jsonl": "rows", "gen_stats.json": "completion", "RESULT": "log", "request_evidence.jsonl": "log", "usage_ledger.jsonl": "log"}
RESULT_COMPLETE = "result=COMPLETE"
#: ``cap_ladder.json:exit`` spellings: a ceiling store exists iff the ladder declared ``escalate``
LADDER_EXIT_ESCALATE = "escalate"
LADDER_EXITS_FINAL_AT_BASE = ("below_threshold", "clean", "triggered_no_eligible")
#: relu_paired's vocabulary for the two states a ladder-merged run can be in
ESCALATION_STATES = ("final_at_base", "final_at_ceiling")
#: the third item state's label and the class of each unresolved item (module docstring, number 3)
ITEM_LABEL_AT_CAP_UNRESOLVED = IC.ITEM_LABEL_AT_CAP_UNRESOLVED
AT_CAP_CLASSES = ("loop", "ceiling", "not_escalated")
#: an item the instrument left UNSCORED below the cap (its think block never closed before EOS, `unterminated_policy: unscored`): not the
#: third state (it did not run out of budget), never a hit or a miss either — the paper's GSM8K reader's spelling for a NULL item
ITEM_LABEL_UNSCORED = "unscored"
UNSCORED_BELOW_CAP_CLASS = "unterminated_below_cap"
#: the run-level shares (run_degeneration measures over the 200 attempted items; the spellings the sampled readers use)
RAW_AT_CAP_MEASURE = "at_token_cap_finish_reason"
LOOP_MEASURE = "rep_ratio_loop"
COHERENT_MEASURE = "truncation_only"
#: the headline's computation: the item-wise ladder merge and recount (no rng)
MERGE_ALGORITHM = "sisteer.registry.adapters.gsm8k_seedrep.ladder_merge"
MERGE_VERSION = "1"
MERGE_SEED_PROVENANCE = "none: a deterministic item-wise merge of the base and ceiling stores' outcome records and a recount"
#: the bound row's spellings (the producer's own, three_numbers_qwen_seedrep.json)
BOUND_SPELLING = "acc_over_attempted_after_rung"
BOUND_CI_METHOD = "unresolved_items_interval"
HEADLINE_METRIC = "acc"
DECODE_SEED_RULE = "per_item"
INSPECT_SEED_POLICY = "replicate"
UNFITTED_TRIAL_RULE = SC.UNFITTED_TRIAL_RULE
#: the served identities the proxy records for a LoRA-native object and for the passthrough route
STEERED_IDENTITY = SC.STEERED_IDENTITY
PASSTHROUGH_IDENTITY = SC.PASSTHROUGH_IDENTITY
ROSTER_REL = SC.ROSTER_REL
PINS_REL = SC.PINS_REL
#: three_numbers' `acc_over_scored` etc. are rounded to four places; the recount is compared at that precision
ROUNDED_TOL = 5e-5 + 1e-12
#: the versions of the producer's three-number record (module docstring), by the keys a cell carries
RECORD_VERSION_BASE = "2026-09-21T07:05Z: base reading, unresolved = loops + ceiling"
RECORD_VERSION_FINAL_STATE = "2026-09-21T09:14Z: final-state keys added, unresolved = loops + ceiling"
RECORD_VERSION_ITEMWISE = "2026-09-21T09:23Z: item-wise (g8k_final_state.py), unresolved = loops + ceiling + not escalated"
RECORD_VERSION_FINAL_STATE_FILE = "2026-09-21T14:40Z: final_state_qwen_seedrep.json (per pair, item-wise; the three-number record carries no entry)"
HEADLINE_KEY_FINAL = "acc_over_scored_final"
#: the paired views' stores REUSE the seed views' store keys (`none__controls__t0__a0__2ff9cdff__d02ab83e` is s1's AND s1p's base store), so a
#: paired view's producer key is the store key prefixed with the tag; the seed views keep the bare key (their 24 ids are landed)
PAIRED_KEY_RULE = "a paired view's producer key is `<tag>/<base store key>`: its stores reuse the seed view's store keys byte for byte"
#: the contrast every steered run files against ITS OWN view's `none` (the paired view's anchor is the paired `none`, never the seed view's)
CONTRAST_METHOD = "exact_mcnemar"
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA16 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# ---- the Arditi all-positions refusal steer (v4): ONE fit on every seed view, and its own default-seed row (module docstring, "THE ALL-POSITIONS OBJECT") --
#: the harness object key of the all-positions refusal steer (dose_grammar): a `vector_all` object whose dose is the raw coefficient c
ARDITI_OBJECT_KEY = DG.OBJECT_KEY_LEGACY
#: the all-positions roster / pins the harness resolved the served object through (common.sh f22_arm_bank_view -> --expect-object-sha256; the
#: seed-replicate roster stages no such key, so the key never resolves through it)
ARDITI_ROSTER_REL = SC.SINGLE_OBJECT_ROSTER_REL
ARDITI_PINS_REL = SC.SINGLE_OBJECT_PINS_REL
#: the DECLARATION of the three seed rows (g8k_seedrep_arditi.py: the all-positions matrix's one row re-keyed onto the seed views at seed bases
#: 1..3; `rows[*].tag` = the seed tag, `bank_view` the rung the object resolves through, `out_root` / `rung_out_root` the store directories,
#: `tuple_names` the OOS tuples per cap, `role` how the value is filed) — REQUIRED for a seed view's all-positions pair, checked field by field
ARDITI_SEED_ROWS_REL = ("configs", "sdfbase", "gsm8k", "arditi_rows_qwen_seedrep.json")
ARDITI_SEED_ROWS_SCHEMA = "g8k_seedrep_arditi_rows/1"
#: the DECLARATION of the default-seed row: the all-positions matrix's own row (`rows[*].tag` = `qwen35_4b_it`, the same fields)
ARDITI_MATRIX_REL = ("configs", "sdfbase", "gsm8k", "matrix_arditi_allpos.json")
ARDITI_MATRIX_SCHEMAS = ("g8k_matrix/1", "g8k_matrix/2")
#: the declared role of a seed row's value (g8k_seedrep_arditi.ROLE): filed unpaired by the producer; this reader contrasts it against the seed
#: view's `none` as it does every steered key and parks the role beside the contrast (the anchor engine differs: `max_lora_rank` 1 vs 8)
ARDITI_ROLE = "arditi_unpaired_value"
#: the standing rule (seedrep_cluster v8): the two replication axes never mix — one fit drawn at three seed bases is three decode-seed
#: trials of one object, never three fits
DECODE_TRIAL_RULE = SC.DECODE_TRIAL_RULE
DEFAULT_SEED_TRIAL_RULE = ("a default-seed point of one object (the all-positions matrix's own row under the default Inspect seed rule, no seed base) is "
                           "trial 0 on its own instrument — never one of the seed trials, never their anchor; the roster pairs it")
#: the NON-seed target's mirror under the same root: `<root>/arditi_allpos_paper_objects/controls_allpos/<standup>/<object dir>/<store dir>/`
#: (the matrix's `out_root` with its `controls_allpos/` head; PROVENANCE.md beside the object dir; the producer's `controls_summary.json`
#: under `<base store dir>/<object dir>/`)
TARGET_MIRROR_DIR = "arditi_allpos_paper_objects"
TARGET_CONTROLS_DIR = "controls_allpos"
TARGET_PROVENANCE_FILE = "PROVENANCE.md"
TARGET_SUMMARY_FILE = "controls_summary.json"
TARGET_STANDUP_RE = re.compile(rf"^(?P<model>{MODEL_KEY})_it$")
TARGET_GROUP_RE = re.compile(rf"^{re.escape(TARGET_MIRROR_DIR)}/(?P<standup>{MODEL_KEY}_it)$")
#: the default Inspect seed rule (no replicate block on `manifest.json:seeds.inspect`): `per_item` with `seed_base` NULL
INSPECT_SEED_POLICY_DEFAULT = "default"
#: a non-seed target has no store pins and no producer count record beside it: the final state is the reader's item-wise merge of the two
#: stores alone; the base reading is checked against `gen_stats.json` (as everywhere) and the producer's `controls_summary.json` when present
RECORD_VERSION_NONE = "none: no producer count record beside the target; the final state is derived from the two stores alone (gen_stats.json and controls_summary.json check the base reading)"
#: what a non-seed target attests its tree with, in place of the store pins: the accepted attempt's sha in ladder_state.json
TREE_ATTESTATION_TARGET = "ladder_state.json: the accepted attempt's sha = manifest.json:producer.tree_sha (both stores); the rung's item_subset.source_identity_hash = the base store's harness_config_hash"


def slug_to_object_key(slug: str) -> str:
    """The producer's directory slug -> the harness object key (``constv2-mix-full--lora_r4_ce`` -> ``constv2-mix-full__lora_r4_ce``)."""
    return slug.replace("--", "__", 1)


def selection_key(seed_n: int, paired: bool = False) -> str:
    """The roster's view key of a tag: ``s<N>`` or ``s<N>p``."""
    return f"s{seed_n}{'p' if paired else ''}"


def pin_key(selection: str, object_key: str, cap: int) -> str:
    return f"{MODEL_KEY}/{selection}/{object_key}@{cap}"


def producer_key_for(base_store_key: str, tag: str, paired: bool) -> str:
    """The producer key of a (tag, object key): the BASE store's NAME (never its path); a paired view's is prefixed with the tag (:data:`PAIRED_KEY_RULE`)."""
    return f"{tag}/{base_store_key}" if paired else base_store_key


def merged_run_id(base_store_key: str, object_key: str, instrument_id: str, tag: Optional[str] = None, paired: bool = False) -> str:
    """The ONE run of a (tag, object key) — the ceiling store's items merge into it."""
    return I.run_id(CAMPAIGN, producer_key_for(base_store_key, tag or "", paired), object_key, EVAL_ID, instrument_id)


def tag_parts(tag: str) -> Optional[Tuple[int, bool]]:
    """``(seed index, paired?)`` of a tag, None when it is not one."""
    m = TAG_RE.match(tag)
    return (int(m.group("seed")), bool(m.group("paired"))) if m else None


def merge_computation_id() -> str:
    return I.computation_id(MERGE_ALGORITHM, MERGE_VERSION, MERGE_SEED_PROVENANCE)


def record_version(tn: Mapping[str, Any]) -> str:
    """Which write of the producer's three-number record a cell is: the item-wise one carries ``n_unresolved_ladder_only``, the additive
    final-state one ``n_scored_final`` without it, the first write neither."""
    if "n_unresolved_ladder_only" in tn:
        return RECORD_VERSION_ITEMWISE
    if "n_scored_final" in tn:
        return RECORD_VERSION_FINAL_STATE
    return RECORD_VERSION_BASE


def rounded_equal(a: Optional[float], b: Optional[float], tol: float = ROUNDED_TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


class Gsm8kSeedrepAdapter:
    name = NAME
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 4 (2026-09-21): the Arditi all-positions object — its three seed rows (`<tag>/arditi_allpos/`) file as DECODE-SEED TRIALS of ONE object on the
    #: seed views (DECODE_TRIAL_RULE: trial_index = seed_base = the seed index, fit_seed unset, the operating point's strength columns, resolved through
    #: the all-positions roster / pins and bound to the producer's eligibility record), and the matrix's own default-seed row under
    #: `arditi_allpos_paper_objects/` files as a single-object NON-seed target on a reader-minted instrument; the seed views' rows are byte-identical to 3.
    #: 3 (2026-09-21): the PAIRED views s<N>p (their own `none` anchor and the composed difference_relu object; tag-prefixed producer keys),
    #: a `run_contrast` per steered run against its own view's `none`, the producer's per-pair final-state record as a cross-check record, and a
    #: scored-at-cap loop kept out of the unresolved count (the classifier's loop count and the unresolved loops are two numbers); the seed views'
    #: run / metric / item rows are unchanged. 2: the third state widened to the at-cap items the ladder never escalated (the producer's item-wise
    #: re-derivation): the bound row's interval and `n_unresolved_neither_hit_nor_miss` count them, `n_unresolved_ladder_only` keeps the first
    #: definition, and the cross-check binds to the record's final-state keys where present; 1: the first version
    EMISSION_VERSION = 4
    rng_scope = "none"
    allow_symlinks = False
    owner = OWNER
    #: the `gen_stats.json:thinking` spelling every store of this document carries (the Qwen thinking card is ON everywhere); a subclass reading
    #: another model's stores binds it per model (the no-switch family's `not_available`)
    THINKING_SPELLING = "on"

    def __init__(self, root: str = ROOT_DEFAULT, seeds: Optional[LC.Seeds] = None, roster_path: Optional[str] = None,
                 pins_path: Optional[str] = None, policy_path: Optional[str] = None, arditi_roster_path: Optional[str] = None,
                 arditi_pins_path: Optional[str] = None, arditi_seed_rows_path: Optional[str] = None, arditi_matrix_path: Optional[str] = None):
        self.root = LC.expand(root)
        self.roots = (self.root,)
        self._seeds = seeds
        self.roster_path = roster_path or DC.repo_path(*ROSTER_REL)
        self.pins_path = pins_path or DC.repo_path(*PINS_REL)
        self.policy_path = policy_path
        # the all-positions object's resolution inputs and the two declarations of its rows (v4)
        self.arditi_roster_path = arditi_roster_path or DC.repo_path(*ARDITI_ROSTER_REL)
        self.arditi_pins_path = arditi_pins_path or DC.repo_path(*ARDITI_PINS_REL)
        self.arditi_seed_rows_path = arditi_seed_rows_path or DC.repo_path(*ARDITI_SEED_ROWS_REL)
        self.arditi_matrix_path = arditi_matrix_path or DC.repo_path(*ARDITI_MATRIX_REL)
        self.refusals = LC.RowRefusals(NAME)
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._roster: Optional[Dict[str, Any]] = None
        self._pins: Optional[Dict[str, Any]] = None
        self._arditi_roster: Optional[Dict[str, Any]] = None
        self._arditi_pins: Optional[Dict[str, Any]] = None
        self._policy: Optional[DP.PolicyFile] = None

    @property
    def seeds(self) -> LC.Seeds:
        if self._seeds is None:
            self._seeds = LC.Seeds.locate()
        return self._seeds

    # ---- discovery ----------------------------------------------------------------------------
    def discover(self) -> List[Input]:
        """Named files of every ``<tag>/<object dir>/<store dir>/`` under the mirror root, one directory level at a time, plus the shared
        records aliased per tag (``#<tag>``); and (v4) the non-seed target's stores under ``arditi_allpos_paper_objects/controls_allpos/<standup>/``
        with the producer's summary and the mirror's PROVENANCE.md beside them. An absent root is a SKIP (the mirror is machine-local); a root
        holding no store refuses."""
        out: List[Input] = []
        if not os.path.isdir(self.root):
            return out
        tags: List[str] = []
        for tag in IC.listdir(self.root):
            if not TAG_RE.match(tag) or not os.path.isdir(os.path.join(self.root, tag)):
                continue
            n_before = len(out)
            for slug in IC.listdir(os.path.join(self.root, tag)):
                for store_dir in STORE_DIRS:
                    d = os.path.join(self.root, tag, slug, store_dir)
                    if os.path.isdir(d):
                        out += self._discover_store(d)
            if len(out) > n_before:
                tags.append(tag)
        # the NON-seed target's mirror (v4): `arditi_allpos_paper_objects/controls_allpos/<standup>/<object dir>/<store dir>/`, PROVENANCE.md beside
        # the object dir and the producer's controls_summary.json under the base store dir — its own group per standup, no shared records
        tdir = os.path.join(self.root, TARGET_MIRROR_DIR, TARGET_CONTROLS_DIR)
        if os.path.isdir(tdir):
            for standup in IC.listdir(tdir):
                sd = os.path.join(tdir, standup)
                if not TARGET_STANDUP_RE.match(standup) or not os.path.isdir(sd):
                    continue
                for slug in IC.listdir(sd):
                    od = os.path.join(sd, slug)
                    if not os.path.isdir(od):
                        continue
                    n_before = len(out)
                    for store_dir in STORE_DIRS:
                        d = os.path.join(od, store_dir)
                        if os.path.isdir(d):
                            out += self._discover_store(d, summary_slug=(slug if store_dir == BASE_DIR else None))
                    p = os.path.join(od, TARGET_PROVENANCE_FILE)
                    if len(out) > n_before and os.path.isfile(p):
                        out.append(check_input(p, role="log", allow_symlink=self.allow_symlinks))
        if not out:
            raise AdapterRefusal(self.root, f"no GSM8K seed-replicate store found ({MODEL_KEY}_s<N>/<object dir>/{BASE_DIR}[__rung{CEILING}]/cells/<store key>/ under the root)")
        for name in SHARED_FILES:
            p = os.path.join(self.root, name)
            if os.path.isfile(p):
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(t) for t in tags]
        return out

    def _discover_store(self, d: str, summary_slug: Optional[str] = None) -> List[Input]:
        out: List[Input] = []
        for name in STORE_TOP_FILES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append(check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks))
        if summary_slug is not None:
            # a non-seed target's base store: the producer's controls_summary.json under `<store dir>/<object dir>/` (never listed for a seed view)
            p = os.path.join(d, summary_slug, TARGET_SUMMARY_FILE)
            if os.path.isfile(p):
                out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        for key in IC.listdir(os.path.join(d, "cells")):
            kd = os.path.join(d, "cells", key)
            if not os.path.isdir(kd):
                continue
            for name in STORE_KEY_FILES:
                p = os.path.join(kd, name)
                if os.path.isfile(p):
                    out.append(check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks))
            p = os.path.join(kd, "outcomes", OUTCOMES_FILE)
            if os.path.isfile(p):
                out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: Input) -> str:
        """One group per seed TAG: every object key, base and ceiling store of ``qwen35_4b_s<N>``; a shared file rides a ``#<tag>`` alias."""
        if "#" in os.path.basename(inp.uri) and TAG_RE.match(inp.uri.rsplit("#", 1)[1]):
            return inp.uri.rsplit("#", 1)[1]
        rel = os.path.relpath(inp.uri, self.root)
        parts = rel.split(os.sep)
        head = parts[0]
        if TAG_RE.match(head):
            return head
        if head == TARGET_MIRROR_DIR and len(parts) > 3 and parts[1] == TARGET_CONTROLS_DIR and TARGET_STANDUP_RE.match(parts[2]):
            # the non-seed target: one group per standup under the mirror (`arditi_allpos_paper_objects/qwen35_4b_it`)
            return f"{TARGET_MIRROR_DIR}/{parts[2]}"
        raise AdapterRefusal(inp.uri, f"input names no seed tag ({MODEL_KEY}_s<N>) and no non-seed target ({TARGET_MIRROR_DIR}/{TARGET_CONTROLS_DIR}/<standup>/) under {self.root}")

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        for inp in inputs:
            base = os.path.basename(inp.uri).rsplit("#", 1)[0]
            if base == THREE_NUMBERS_FILE:
                d = self._json(inp) or {}
                group = self.comparison_group(inp)
                members = sorted(k for k in (d.get("cells") or {}) if str(k).startswith(f"s{TAG_RE.match(group).group('seed')}/"))
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.cells", 1, 1, members, inp.sha256))
            elif base == PINS_FILE:
                d = self._json(inp) or {}
                group = self.comparison_group(inp)
                members = sorted(k for k in d if k != "_meta" and str(k).startswith(f"{MODEL_KEY}/s{TAG_RE.match(group).group('seed')}/"))
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.pins", 1, 1, members, inp.sha256))
        if inputs and self.refusals.items:
            rows += self.refusals.source_state_rows(source_id_for(NAME, self.comparison_group(inputs[0])), inputs[0].uri, inputs[0].sha256)
        return rows

    def status_extra(self, group: str) -> Dict[str, Any]:
        return dict(self._facts.get(group) or {})

    def status_sources(self, group: str) -> Tuple[str, ...]:
        return (SOURCE,)

    # ---- small readers ------------------------------------------------------------------------
    @staticmethod
    def _json(inp: Optional[Input]) -> Optional[Dict[str, Any]]:
        if inp is None:
            return None
        with open(inp.path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None

    def _roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._roster is None or self._pins is None:
            if not os.path.isfile(self.roster_path):
                raise IC.Unresolved(self.roster_path, "the roster is not in this checkout; no staged id can be named")
            if not os.path.isfile(self.pins_path):
                raise IC.Unresolved(self.pins_path, "the pins are not in this checkout; no object sha can be named")
            import yaml
            with open(self.roster_path, "r", encoding="utf-8") as fh:
                roster = yaml.safe_load(fh)
            with open(self.pins_path, "r", encoding="utf-8") as fh:
                pins = json.load(fh)
            if not isinstance(roster, dict) or not isinstance(pins, dict):
                raise IC.Unresolved(self.roster_path, "the roster / pins are not mappings")
            self._roster, self._pins = roster, pins
        return self._roster, self._pins

    def _policy_file(self) -> Optional[DP.PolicyFile]:
        if self._policy is None:
            try:
                self._policy = DP.load(self.policy_path)
            except (OSError, ValueError, KeyError):
                self._policy = None
        return self._policy

    # ---- objects ------------------------------------------------------------------------------
    def resolve_object(self, object_key: str, selection: str) -> LC.ResolvedObject:
        """The object a store's key names at the seed selection: ``none``, a declared prompt (``si:<sha>`` from si_texts), or the staged
        seed-replicate object the roster's ``s<N>`` selection maps the key to and the pins sha (:func:`jobplan.object_id_for`). Never the live
        registry: the run is emitted on this identity and ``blocked_records`` holds it while the object row is absent."""
        g = LC.cell_grammar_legacy(object_key)
        kind = g["object_kind"]
        if kind == "none":
            return LC.ResolvedObject("none", "none", "none", "baseline_none", "derived", "the unsteered object key", derivation="none")
        if kind == "system_instruction":
            oid = self.seeds.si_object_id(g["si_id"] or "")
            if oid is None:
                raise IC.Unresolved(object_key, f"si_texts.yaml carries no sha256 for the declared prompt {g['si_id']!r}")
            return LC.ResolvedObject(oid, kind, g["paper_label"], "prompt", "recorded", "si_texts.yaml si_text_sha256", si_id=g["si_id"], derivation="none")
        if kind is None:
            raise IC.Unresolved(object_key, "the key matches no object grammar this reader knows")
        roster, pins = self._roster_pins()
        try:
            sha = JP.object_id_for(object_key, model_key=MODEL_KEY, roster_selection=selection, roster=roster, pins=pins, si_sha={})
        except LookupError as exc:
            raise IC.Unresolved(object_key, f"the roster's {selection!r} selection / the pins name no object for it: {exc}") from None
        staged = roster["targets"][MODEL_KEY]["views"][selection]["cells"][object_key]
        return LC.ResolvedObject(str(sha), kind, g["paper_label"], g["object_role"], "recorded",
                                 f"roster targets.{MODEL_KEY}.views.{selection}.cells[{object_key}] -> {staged}; pins {MODEL_KEY}.{staged}.object_sha256 "
                                 f"(jobplan.object_id_for, the harness's own resolution); attested by the store's rows aggregate object_artifact_sha256",
                                 staged_id=str(staged), derivation=g["derivation"], si_id=g["si_id"])

    # ---- the all-positions object (v4) --------------------------------------------------------
    def _arditi_roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """The all-positions refusal steer's roster and pins (the harness's own resolution inputs: common.sh f22_arm_bank_view)."""
        if self._arditi_roster is None or self._arditi_pins is None:
            if not os.path.isfile(self.arditi_roster_path):
                raise IC.Unresolved(self.arditi_roster_path, "the all-positions roster is not in this checkout; no staged id can be named")
            if not os.path.isfile(self.arditi_pins_path):
                raise IC.Unresolved(self.arditi_pins_path, "the all-positions pins are not in this checkout; no object sha can be named")
            import yaml
            with open(self.arditi_roster_path, "r", encoding="utf-8") as fh:
                roster = yaml.safe_load(fh)
            with open(self.arditi_pins_path, "r", encoding="utf-8") as fh:
                pins = json.load(fh)
            if not isinstance(roster, dict) or not isinstance(pins, dict):
                raise IC.Unresolved(self.arditi_roster_path, "the all-positions roster / pins are not mappings")
            self._arditi_roster, self._arditi_pins = roster, pins
        return self._arditi_roster, self._arditi_pins

    def _arditi_declaration(self, tag: str, selection: str, target: bool) -> Dict[str, Any]:
        """The declaration row of the all-positions object's store on ``tag``: the producer's eligibility record (a seed view; :data:`ARDITI_SEED_ROWS_REL`)
        or the all-positions matrix's own row (the non-seed target; :data:`ARDITI_MATRIX_REL`), read as DATA — exactly one row naming the tag and
        the key, with ``_path`` / ``_schema`` / ``_sha256`` / ``_role`` beside its fields. Raises :class:`_items_common.Unresolved`."""
        path = self.arditi_matrix_path if target else self.arditi_seed_rows_path
        what = "the all-positions matrix" if target else "the seed rows' eligibility record"
        if not os.path.isfile(path):
            raise IC.Unresolved(path, f"{what} is not in this checkout; the all-positions object's store on {tag} has no declaration to bind to")
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            d = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise IC.Unresolved(path, f"{what} is not JSON: {exc}") from None
        schema = str((d or {}).get("schema") or "") if isinstance(d, Mapping) else ""
        want = ARDITI_MATRIX_SCHEMAS if target else (ARDITI_SEED_ROWS_SCHEMA,)
        if not isinstance(d, Mapping) or schema not in want:
            raise IC.Unresolved(path, f"{what} carries schema {schema!r}, not {' / '.join(want)}")
        rows = [r for r in (d.get("rows") or []) if isinstance(r, Mapping) and str(r.get("tag") or "") == tag and str(r.get("cell") or "") == ARDITI_OBJECT_KEY]
        if len(rows) != 1:
            raise IC.Unresolved(path, f"{what} names {len(rows)} row(s) for tag {tag} / {ARDITI_OBJECT_KEY} (expected exactly one)")
        row = dict(rows[0])
        if str(row.get("model_key") or "") != MODEL_KEY or str(row.get("view") or "") != selection:
            raise IC.Unresolved(path, f"{what}'s row for {tag} is model {row.get('model_key')!r} at {row.get('view')!r}, not {MODEL_KEY} / {selection}")
        role = d.get("role") if not target else None
        if not target and str(role or "") != ARDITI_ROLE:
            raise IC.Unresolved(path, f"{what} declares role {role!r}, not {ARDITI_ROLE!r}; another role needs its own reading here")
        row.update(_path=LC.rel_home(path), _schema=schema, _sha256=I.sha256_hex(raw), _role=role)
        return row

    def resolve_arditi_object(self, decl: Mapping[str, Any]) -> Tuple[LC.ResolvedObject, DG.ArditiAllposRung]:
        """The ONE object the declaration names, resolved the way the harness resolved it before serving (:func:`jobplan.object_id_for` over the
        all-positions roster's rung selection — the declaration's ``bank_view`` — and its pins); the pin's kind / c / unit / rung must be the rung's,
        the roster's staged id and the declaration's ``staged_id`` / ``object_sha256`` the pin's. Never the live registry: the run is emitted on
        this identity and ``blocked_records`` holds it while the object row is absent (the ``arditi_allpos`` reader mints it)."""
        bank = str(decl.get("bank_view") or "")
        op = DG.parse_arditi_allpos_tag(f"{MODEL_KEY}_{bank}") if bank else None
        if op is None or op.trial is not None or op.model_key != MODEL_KEY:
            raise IC.Unresolved(ARDITI_OBJECT_KEY, f"the declaration's bank_view {bank!r} spells no all-positions rung (dose_grammar `ar_c<tok>`)")
        if str(decl.get("staged_id") or "") != op.staged_id:
            raise IC.Unresolved(ARDITI_OBJECT_KEY, f"the declaration stages {decl.get('staged_id')!r}, not the rung's {op.staged_id!r} (dose_grammar)")
        roster, pins = self._arditi_roster_pins()
        try:
            sha = JP.object_id_for(ARDITI_OBJECT_KEY, model_key=MODEL_KEY, roster_selection=op.rung, roster=roster, pins=pins, si_sha={})
        except LookupError as exc:
            raise IC.Unresolved(ARDITI_OBJECT_KEY, f"the all-positions roster's {op.rung!r} selection / the pins name no object for it: {exc}") from None
        staged = str(roster["targets"][MODEL_KEY]["views"][op.rung]["cells"][ARDITI_OBJECT_KEY])
        if staged != op.staged_id:
            raise IC.Unresolved(ARDITI_OBJECT_KEY, f"the roster stages {op.rung}/{ARDITI_OBJECT_KEY} as {staged!r}, not the rung's {op.staged_id!r} (dose_grammar)")
        pin = (pins.get(MODEL_KEY) or {}).get(staged) or {}
        bad: List[str] = []
        if str(pin.get("kind") or "") != DG.OBJECT_KIND:
            bad.append(f"kind {pin.get('kind')!r} is not {DG.OBJECT_KIND!r}")
        if LC.num_or_none(pin.get("c")) is None or abs(float(pin["c"]) - op.coefficient) > 1e-9:
            bad.append(f"c {pin.get('c')!r} is not the rung's {op.coefficient:g}")
        if str(pin.get("dose_unit") or "") != op.unit:
            bad.append(f"dose_unit {pin.get('dose_unit')!r} is not {op.unit!r}")
        if pin.get("view") is not None and str(pin.get("view")) != op.rung:
            bad.append(f"view {pin.get('view')!r} is not the rung {op.rung!r}")
        if bad:
            raise IC.Unresolved(ARDITI_OBJECT_KEY, f"the pin {MODEL_KEY}.{staged} is not the rung's operating point: " + "; ".join(bad))
        if str(decl.get("object_sha256") or "") != sha:
            raise IC.Unresolved(ARDITI_OBJECT_KEY, f"the declaration names object {str(decl.get('object_sha256'))[:16]}…, the pins {sha[:16]}… (staged {staged}); one object, two spellings")
        obj = LC.ResolvedObject(str(sha), DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded",
                                f"the declaration {decl.get('_path')} (row {decl.get('index')}: staged_id {staged}, bank_view {op.rung}) -> all-positions roster targets.{MODEL_KEY}.views.{op.rung}."
                                f"cells[{ARDITI_OBJECT_KEY}] -> {staged}; pins {MODEL_KEY}.{staged}.object_sha256 (jobplan.object_id_for, the harness's own resolution: common.sh "
                                "f22_arm_bank_view -> --expect-object-sha256); attested by the store's manifest fit_provenance.object_sha / provenance.artifact_sha256 and the rows aggregate",
                                staged_id=staged, derivation="none")
        return obj, op

    def _check_arditi_declaration(self, decl: Mapping[str, Any], op: DG.ArditiAllposRung, tag: str, selection: str, slug: str, base: Mapping[str, Any],
                                  ceil: Optional[Mapping[str, Any]], block: Mapping[str, Any]) -> List[str]:
        """The declaration against the store, field by field: the out roots must spell this store's directories, the OOS tuples per cap the manifests'
        ``oos_experiment``, the sampling row the sent numbers, the positions the manifest's ``steer_positions``, thinking ON, the object attested."""
        bad: List[str] = []
        man = base["manifest"]
        for key, store_dir, present in (("out_root", BASE_DIR, True), ("rung_out_root", CEILING_DIR, ceil is not None)):
            want_tail = os.path.join(tag, slug, store_dir)
            got = str(decl.get(key) or "")
            if not got:
                bad.append(f"{key} is not declared")
            elif not (got == want_tail or got.endswith(os.sep + want_tail)):
                bad.append(f"{key} {got!r} does not spell this store's directory (…/{want_tail})")
        tuples = decl.get("tuple_names") if isinstance(decl.get("tuple_names"), Mapping) else {}
        for cap, m in ((CAP, man), (CEILING, ceil["manifest"] if ceil is not None else None)):
            if m is None:
                continue
            want_t = tuples.get(str(cap))
            got_t = m.get("oos_experiment")
            if not want_t or str(got_t or "") != str(want_t):
                bad.append(f"the manifest at {cap:,} names oos_experiment {got_t!r}, the declaration's tuple_names[{cap}] {want_t!r}")
        if str(decl.get("thinking_mode") or "") != "on":
            bad.append(f"thinking_mode {decl.get('thinking_mode')!r} is not 'on'")
        if str(decl.get("served_positions_expected") or "") != str(man.get("steer_positions") or ""):
            bad.append(f"served_positions_expected {decl.get('served_positions_expected')!r} is not the manifest's steer_positions {man.get('steer_positions')!r}")
        sampling = decl.get("sampling") if isinstance(decl.get("sampling"), Mapping) else None
        if sampling is None or not self._numbers_agree(block, {k: sampling.get(k) for k in DP.PARAM_KEYS}):
            bad.append(f"the declared sampling row {RP._numbers(sampling or {})} is not the sent block {RP._numbers(block)}")
        if str(decl.get("harness_cell") or ARDITI_OBJECT_KEY) != ARDITI_OBJECT_KEY or str(decl.get("cell_slug") or slug) != slug:
            bad.append(f"harness_cell {decl.get('harness_cell')!r} / cell_slug {decl.get('cell_slug')!r} are not {ARDITI_OBJECT_KEY} / {slug}")
        if decl.get("fitted") is not True:
            bad.append(f"fitted {decl.get('fitted')!r} is not true (the all-positions object is a fitted direction)")
        return bad

    @staticmethod
    def _cross_check_target_summary(summary: Mapping[str, Any], base: Mapping[str, Any], items: Mapping[str, Any], object_id: str, tree: str,
                                    okey: str = ARDITI_OBJECT_KEY, n_items: int = N_ITEMS, thinking: str = "on") -> List[str]:
        """The producer's ``controls_summary.json`` beside a non-seed target's base store against the store: the key, the tree, the configuration hash,
        the object and the base reading (``controls.gsm8k``: acc over scored, n, the cap, thinking, the task)."""
        bad: List[str] = []
        c = items["counts"]
        src = (summary.get("sources") or {}).get(TASK) if isinstance(summary.get("sources"), Mapping) else None
        if not isinstance(src, Mapping):
            bad.append(f"sources[{TASK}] is absent")
        else:
            if str(src.get("cell_key") or "") != base["store_key"]:
                bad.append(f"sources[{TASK}].cell_key {src.get('cell_key')!r} is not the base store key {base['store_key']}")
            if str(src.get("producer_tree_sha") or "") != tree:
                bad.append(f"sources[{TASK}].producer_tree_sha {str(src.get('producer_tree_sha'))[:12]!r} is not the manifest's tree {tree[:12]}")
            if str(src.get("harness_config_hash") or "") != str(base["manifest"].get("harness_config_hash") or ""):
                bad.append(f"sources[{TASK}].harness_config_hash {src.get('harness_config_hash')!r} is not the manifest's {base['manifest'].get('harness_config_hash')!r}")
        prov_sha = str(((summary.get("provenance") or {}) if isinstance(summary.get("provenance"), Mapping) else {}).get("artifact_sha256") or "")
        if prov_sha != object_id:
            bad.append(f"provenance.artifact_sha256 {prov_sha[:16]!r} is not the resolved object {object_id[:16]}")
        if str(summary.get("cell_id") or "") != okey:
            bad.append(f"cell_id {summary.get('cell_id')!r} is not {okey}")
        g = (summary.get("controls") or {}).get(EVAL_ID) if isinstance(summary.get("controls"), Mapping) else None
        if not isinstance(g, Mapping):
            bad.append(f"controls.{EVAL_ID} is absent")
            return bad
        acc_b = c["acc_base"]
        gacc = LC.num_or_none(g.get("acc"))
        if (acc_b is None) != (gacc is None) or (acc_b is not None and abs(float(gacc) - acc_b) > 1e-9):
            bad.append(f"controls.{EVAL_ID}.acc {gacc} is not the base reading correct / scored {acc_b}")
        if LC.int_or_none(g.get("n")) != n_items or LC.int_or_none(g.get("max_tokens")) != CAP or str(g.get("thinking") or "") != thinking or str(g.get("task") or "") != TASK:
            bad.append(f"controls.{EVAL_ID} (n {g.get('n')}, max_tokens {g.get('max_tokens')}, thinking {g.get('thinking')!r}, task {g.get('task')!r}) is not the base store's "
                       f"(n {n_items}, max_tokens {CAP}, thinking {thinking!r}, task {TASK!r})")
        return bad

    def _mint_target_instrument(self, inst_row: Mapping[str, Any], base: Mapping[str, Any], man: Mapping[str, Any], block: Mapping[str, Any], policy: Optional[DP.Policy],
                                policy_why: str, frameworks: Mapping[str, Any], geometry: Mapping[str, Any], model_id: str, tag: str, slug: str, seed_rule_text: str,
                                seed_inst_match: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """``(instrument row, instrument record payload or None, the mint's facts or None)``: the non-seed target's OWN instrument — the canonical seed row
        (:data:`INSTRUMENT_SEED` bound to the model) with the store's own facts (``manifest.json:harness_config_hash``, the harness's configuration
        identity the seed rule enters; its inspect versions; its served window; the ladder's trigger rule; the sent numbers and the registered row's
        decode source they equal; thinking ON, the native path, the manifest's positions), through :func:`identity.instrument_id`. Bounded: every fact
        must be on the record (a missing one refuses the pair, nothing is guessed). When the facts hash to the seed instrument's own id nothing is
        minted and the seed row is returned with ``None`` twice (they ARE the same instrument)."""
        where = base["files"]["manifest.json"].uri
        chash = str(man.get("harness_config_hash") or "")
        if not _SHA16.match(chash):
            raise _PairRefusal(f"manifest.json:harness_config_hash {chash!r} names no 16-hex configuration hash; no instrument is minted from a record without its harness identity", where)
        if str(man.get("serving_engine") or "") != "lora_native":
            raise _PairRefusal(f"manifest.json:serving_engine {man.get('serving_engine')!r} is not 'lora_native'; the all-positions object is served natively", where)
        positions = str(man.get("steer_positions") or "")
        if positions not in V.APPLIED_POSITIONS_MODES:
            raise _PairRefusal(f"manifest.json:steer_positions {positions!r} is not an applied-positions mode", where)
        window = LC.int_or_none(geometry.get("max_model_len"))
        if window is None:
            raise _PairRefusal("manifest.json:geometry.max_model_len names no served window", where)
        if not frameworks.get("inspect_ai") or not frameworks.get("inspect_evals"):
            raise _PairRefusal("manifest.json:frameworks names no inspect_ai / inspect_evals version", where)
        rule = str((base["cap_ladder"].get("rules") or {}).get("trigger") or base["cap_ladder"].get("rule") or "")
        if not rule:
            raise _PairRefusal("cap_ladder.json names no trigger rule for the ladder", base["top"]["cap_ladder.json"].uri)
        if policy is None:
            raise _PairRefusal(f"the sent numbers match no registered decode policy row ({policy_why}); a decode block matching no row is never minted onto an instrument", where)
        fields = {k: v for k, v in inst_row.items() if k not in ("instrument_id", "extra", "field_confidence_json", "notes")}
        changes: Dict[str, Any] = {"harness_config_hash": chash,
                                   "harness_version": f"inspect-ai {frameworks.get('inspect_ai')} / inspect_evals {frameworks.get('inspect_evals')} ({base['gen_stats'].get('framework')})",
                                   "context_window_tokens": window, "ladder_promotion_rule": rule, "decode_source": str(policy.decode_source), "decode_seed_rule": DECODE_SEED_RULE,
                                   "thinking_mode": "on", "serving_path": "lora_native", "applied_positions_mode": positions, "token_cap_tokens": CAP,
                                   "token_cap_ladder_json": list(LADDER), "read_at": "ladder_final"}
        for k in DP.PARAM_KEYS:
            v = block.get(k)
            changes[f"decode_{k}"] = (v if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
        changed = {k: {"seed": inst_row.get(k), "record": v} for k, v in changes.items() if I.canonical_json(inst_row.get(k)) != I.canonical_json(v)}
        fields.update(changes)
        seed_iid = str(inst_row["instrument_id"])
        iid = I.instrument_id(fields)
        if iid == seed_iid:
            return dict(inst_row), None, None
        fields["instrument_id"] = iid
        why = ("the default Inspect seed rule (no replicate block) is not the seed views' replicate rule, so the point is never forced onto the instrument their rows sit on; "
               "the harness configuration hash (which the seed rule enters) and the record's own versions, window, ladder rule and numbers mint its own")
        minted = {"instrument_id": iid, "derived_from_instrument_id": seed_iid, "derived_from_seed_instrument": INSTRUMENT_SEED, "changed_fields": changed, "why": why,
                  "seed_instrument_decode_match": seed_inst_match, "decode_seed_policy": INSPECT_SEED_POLICY_DEFAULT, "seed_rule_text": seed_rule_text,
                  "decode_policy_id": policy.policy_id, "minted_by": NAME, "emission_version": str(self.EMISSION_VERSION)}
        payload = dict(fields)
        payload["notes"] = (f"Derived GSM8K instrument of the all-positions object's default-seed store ({tag} {slug}): the canonical seed instrument {INSTRUMENT_SEED} "
                            f"({seed_iid[:12]}…, bound to {model_id}) with the store's own facts ({', '.join(sorted(changed))}); minted by the {NAME} reader through "
                            f"identity.instrument_id because {why}. Its rows are marked (extra.instrument_minted) and form no contrast against the seed views' `none`.")
        payload["extra"] = {"derived_from_instrument_id": seed_iid, "derived_from_seed_instrument": INSTRUMENT_SEED, "changed_fields": changed, "model_id": model_id,
                            "decode_policy_ref": policy.policy_id, "decode_seed_policy": INSPECT_SEED_POLICY_DEFAULT, "seed_rule_text": seed_rule_text,
                            "store": LC.rel_home(base["dir"]), "standup_legacy": tag, "minted_by": NAME, "emission_version": str(self.EMISSION_VERSION), "why": why}
        payload["field_confidence_json"] = {"prompt_template_sha256": LC.unverified("the served path renders the chat template server-side; no prompt-template hash is recorded for this harness"),
                                            "harness_config_hash": LC.confidence_entry("recorded", "manifest.json:harness_config_hash (the base store's; the store key's <chash8>)"),
                                            "harness_version": LC.confidence_entry("recorded", "manifest.json:frameworks (inspect_ai / inspect_evals) and gen_stats.json:framework"),
                                            "context_window_tokens": LC.confidence_entry("recorded", "manifest.json:geometry.max_model_len (the base store's)"),
                                            "ladder_promotion_rule": LC.confidence_entry("recorded", "cap_ladder.json:rules.trigger"),
                                            "decode_source": LC.confidence_entry("recorded", f"the registered policy row {policy.policy_id} the sent numbers equal")}
        return fields, payload, minted

    # ---- read ---------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        batch = RecordBatch(source_id_for(NAME, group), group)
        self.refusals = LC.RowRefusals(NAME)
        facts: Dict[str, Any] = {"tag": group, "runs": 0, "pairs": {}, "ceiling_merged": 0, "contrasts": {}, "refusals": [], "instrument_seed_id": INSTRUMENT_SEED}
        m = TAG_RE.match(group)
        if not m:
            tm = TARGET_GROUP_RE.match(group)
            if tm:
                self._read_target(batch, ctx, group, tm.group("standup"), inputs, facts)
            else:
                self.refusals.add(group, f"group {group!r} does not spell a seed tag ({MODEL_KEY}_s<N> or its paired view {MODEL_KEY}_s<N>p) nor the non-seed target "
                                         f"({TARGET_MIRROR_DIR}/{MODEL_KEY}_it)", inputs[0].uri if inputs else self.root)
            self._finish(group, facts)
            yield batch
            return
        seed_n = int(m.group("seed"))
        paired = bool(m.group("paired"))
        selection = f"s{seed_n}{'p' if paired else ''}"
        facts.update(paired=paired, paired_of=(f"s{seed_n}" if paired else None), producer_key_rule=(PAIRED_KEY_RULE if paired else "the base store key"))
        seeds = self.seeds
        mv = seeds.model_for_view_legacy(group)
        model_id = mv.get("model_id")
        if model_id is None:
            self.refusals.add(group, f"no legacy model spelling for {group!r} ({mv.get('source')}); a fourth seed needs its own row in legacy_spellings.yaml", self.root)
            self._finish(group, facts)
            yield batch
            return
        try:
            inst_row = seeds.instrument_row(INSTRUMENT_SEED, EVAL_ID, model_id=model_id)
        except KeyError as exc:
            self.refusals.add(group, f"instrument row: {exc}", self.root)
            self._finish(group, facts)
            yield batch
            return
        facts.update(seed=seed_n, selection=selection, model_id=model_id, instrument_id=str(inst_row["instrument_id"]))
        stores, shared, _prov = self._sort_stores(inputs, (group,))
        pins_inp, tn_inp, fs_inp = shared.get(PINS_FILE), shared.get(THREE_NUMBERS_FILE), shared.get(FINAL_STATE_FILE)
        pins = self._json(pins_inp) if pins_inp is not None else None
        three = self._json(tn_inp) if tn_inp is not None else None
        fstate = self._json(fs_inp) if fs_inp is not None else None
        pins_ok = isinstance(pins, dict) and str((pins.get("_meta") or {}).get("schema")) == PINS_SCHEMA
        three_ok = isinstance(three, dict) and str(three.get("schema")) == THREE_NUMBERS_SCHEMA and isinstance(three.get("cells"), dict)
        fstate_ok = isinstance(fstate, dict) and all(isinstance(v, Mapping) and "n_scored_final" in v for v in fstate.values()) and bool(fstate)
        facts["store_pins"] = (LC.rel_home(pins_inp.uri.rsplit("#", 1)[0]) if pins_inp is not None else None)
        facts["three_numbers"] = (LC.rel_home(tn_inp.uri.rsplit("#", 1)[0]) if tn_inp is not None else None)
        facts["final_state"] = (LC.rel_home(fs_inp.uri.rsplit("#", 1)[0]) if fs_inp is not None else None)
        filed: Dict[str, Dict[str, Any]] = {}
        slugs = sorted({slug for slug, _sd in stores})
        for slug in slugs:
            okey = slug_to_object_key(slug)
            rkey = f"{group}|{okey}"
            if not pins_ok:
                self.refusals.add(rkey, f"{PINS_FILE} ({PINS_SCHEMA}) is absent or malformed beside the stores; no store pin can be checked", self.root)
                continue
            if not three_ok and not fstate_ok:
                self.refusals.add(rkey, f"neither {THREE_NUMBERS_FILE} ({THREE_NUMBERS_SCHEMA}) nor {FINAL_STATE_FILE} is present and well-formed beside the stores; nothing can be cross-checked", self.root)
                continue
            base_st = stores.get((slug, BASE_DIR))
            ceil_st = stores.get((slug, CEILING_DIR))
            if base_st is None:
                self.refusals.add(rkey, f"a ceiling store without its base store ({BASE_DIR}/ is absent)", ceil_st["dir"] if ceil_st else self.root)
                continue
            try:
                filed[okey] = self._emit_pair(batch, ctx, group, seed_n, paired, selection, model_id, inst_row, okey, slug, base_st, ceil_st, pins,
                                              (three if three_ok else None), (fstate if fstate_ok else None), pins_inp, tn_inp, fs_inp, facts)
            except _PairRefusal as exc:
                self.refusals.add(rkey, exc.why, exc.where)
        self._emit_contrasts(batch, ctx, group, filed, facts)
        self._finish(group, facts)
        yield batch

    def _sort_stores(self, inputs: Sequence[Input], prefix: Tuple[str, ...]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Input], Dict[str, Input]]:
        """The group's inputs sorted into stores — ``(object slug, store dir) -> {"top": {name: Input}, "keys": {store key: {name: Input}},
        "dir", ["summary"]}`` under ``<root>/<prefix…>/<slug>/<store dir>/`` — plus the shared records (``#<tag>`` aliases) and, for the non-seed
        target, the PROVENANCE.md beside each object dir (slug -> Input)."""
        stores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        shared: Dict[str, Input] = {}
        prov: Dict[str, Input] = {}
        p = len(prefix)
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" in base and base.rsplit("#", 1)[0] in SHARED_FILES:
                shared[base.rsplit("#", 1)[0]] = inp
                continue
            rel = os.path.relpath(inp.uri, self.root).split(os.sep)
            if tuple(rel[:p]) != prefix:
                continue
            if len(rel) == p + 2 and rel[p + 1] == TARGET_PROVENANCE_FILE:
                prov[rel[p]] = inp
                continue
            if len(rel) < p + 3 or rel[p + 1] not in STORE_DIRS:
                continue
            st = stores.setdefault((rel[p], rel[p + 1]), {"top": {}, "keys": {}, "dir": os.path.join(self.root, *rel[:p + 2])})
            body = rel[p + 2:]
            if len(body) == 1:
                st["top"][body[0]] = inp
            elif len(body) == 2 and body[0] == "cells":
                pass
            elif len(body) == 2 and body[0] == rel[p] and body[1] == TARGET_SUMMARY_FILE:
                st["summary"] = inp                     # the non-seed target's producer summary, `<store dir>/<object dir>/controls_summary.json`
            elif len(body) >= 3 and body[0] == "cells":
                files = st["keys"].setdefault(body[1], {})
                if len(body) == 3:
                    files[body[2]] = inp
                elif len(body) == 4 and body[2] == "outcomes":
                    files["outcomes"] = inp
        return stores, shared, prov

    def _read_target(self, batch: RecordBatch, ctx: AdapterContext, group: str, standup: str, inputs: Sequence[Input], facts: Dict[str, Any]) -> None:
        """The NON-seed target's group (``arditi_allpos_paper_objects/<standup>``): the all-positions matrix's own GSM8K row under the default
        Inspect seed rule, filed as a single-object target on its own minted instrument (module docstring, "THE ALL-POSITIONS OBJECT"). Only the
        all-positions key has a declaration here; any other object dir refuses by name. No ``none`` sits beside it, so no contrast is formed."""
        facts.pop("tag", None)
        facts.update(target=group, view_legacy=standup, selection="it", declaration=LC.rel_home(self.arditi_matrix_path), trial_rule=DEFAULT_SEED_TRIAL_RULE)
        seeds = self.seeds
        mv = seeds.model_for_view_legacy(standup)
        model_id = mv.get("model_id")
        if model_id is None:
            self.refusals.add(group, f"no legacy model spelling for {standup!r} ({mv.get('source')})", self.root)
            return
        try:
            inst_row = seeds.instrument_row(INSTRUMENT_SEED, EVAL_ID, model_id=model_id)
        except KeyError as exc:
            self.refusals.add(group, f"instrument row: {exc}", self.root)
            return
        facts.update(model_id=model_id, seed_instrument_id=str(inst_row["instrument_id"]))
        stores, _shared, prov = self._sort_stores(inputs, (TARGET_MIRROR_DIR, TARGET_CONTROLS_DIR, standup))
        for slug in sorted({slug for slug, _sd in stores}):
            okey = slug_to_object_key(slug)
            rkey = f"{group}|{okey}"
            if okey != ARDITI_OBJECT_KEY:
                self.refusals.add(rkey, f"the non-seed target mirror declares only the all-positions object ({ARDITI_OBJECT_KEY}); {okey!r} has no declaration here", self.root)
                continue
            base_st = stores.get((slug, BASE_DIR))
            ceil_st = stores.get((slug, CEILING_DIR))
            if base_st is None:
                self.refusals.add(rkey, f"a ceiling store without its base store ({BASE_DIR}/ is absent)", ceil_st["dir"] if ceil_st else self.root)
                continue
            target = {"summary": base_st.get("summary"), "provenance": prov.get(slug), "standup": standup}
            try:
                self._emit_pair(batch, ctx, standup, None, False, "it", model_id, inst_row, okey, slug, base_st, ceil_st, None, None, None, None, None, None, facts, target=target)
            except _PairRefusal as exc:
                self.refusals.add(rkey, exc.why, exc.where)
                continue
            facts["contrasts"][okey] = {"anchor": None, "why": "no `none` beside the non-seed target (the all-positions matrix's own row): the roster pairs it; no same-standup "
                                                                "contrast is formed and none is invented"}

    def _emit_contrasts(self, batch: RecordBatch, ctx: AdapterContext, tag: str, filed: Mapping[str, Mapping[str, Any]], facts: Dict[str, Any]) -> None:
        """One ``run_contrast`` per steered run of the tag against the SAME view's ``none`` (the paired view's own native anchor, never the seed
        view's), paired per item over the items SCORED on both sides at the ladder's final state: the accuracy difference in pp, McNemar's exact p
        on the discordant pairs, Cohen's h. An unresolved or unscored item on either side leaves the pair (its count rides extra)."""
        anchor = filed.get("none")
        if anchor is None:
            for okey in sorted(k for k in filed if k != "none"):
                facts["contrasts"][okey] = {"anchor": None, "why": f"the view's own `none` did not file ({tag}|none refused or absent); no contrast is formed"}
            return
        for okey in sorted(k for k in filed if k != "none"):
            st = filed[okey]
            a_items, s_items = anchor["items"], st["items"]
            shared = sorted(i for i in s_items if i in a_items and s_items[i]["value"] is not None and a_items[i]["value"] is not None)
            if not shared:
                facts["contrasts"][okey] = {"anchor": anchor["run_id"], "why": "no item scored on both sides"}
                continue
            b = sum(1 for i in shared if s_items[i]["value"] == 1.0 and a_items[i]["value"] == 0.0)
            c = sum(1 for i in shared if s_items[i]["value"] == 0.0 and a_items[i]["value"] == 1.0)
            p_s = sum(s_items[i]["value"] for i in shared) / len(shared)
            p_a = sum(a_items[i]["value"] for i in shared) / len(shared)
            payload = {"run_id": st["run_id"], "anchor_run_id": anchor["run_id"], "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0,
                       "computation_id": merge_computation_id(), "anchor_judgment_id": "", "anchor_kind": "same_standup_none",
                       "delta": 100.0 * (p_s - p_a), "delta_p": DC.mcnemar_exact(b, c), "delta_method": CONTRAST_METHOD,
                       "effect_size": DC.cohen_h(p_s, p_a), "effect_kind": "cohens_h", "n_pairs": len(shared), "computed_by": "registry", "computed_utc": st["when"],
                       "extra": {"unit": "pp", "anchor_view_legacy": tag, "anchor_object_key": "none", "discordant_steered_only": b, "discordant_anchor_only": c,
                                 "acc_steered_over_pairs": p_s, "acc_anchor_over_pairs": p_a,
                                 "n_excluded_steered_unscored": sum(1 for i in s_items if s_items[i]["value"] is None),
                                 "n_excluded_anchor_unscored": sum(1 for i in a_items if a_items[i]["value"] is None),
                                 "method_text": "paired per item over the items SCORED on both sides at the ladder's final state (the same 200 Platinum pairs, one instrument, "
                                                "one seed base): the accuracy difference, pp; p from McNemar's exact test on the discordant pairs; Cohen's h of the two accuracies; "
                                                "an unresolved or unscored item on either side leaves the pair"}}
            if st.get("declared_role") is not None:
                # the all-positions object's seed row: the producer declared its value unpaired (the single-object engine attests another max_lora_rank than the
                # seed view's engines); the contrast is filed as every steered key's is, and the declaration rides beside it for the reader of the frame
                payload["extra"].update(declared_role=st["declared_role"], anchor_standup_note=("the declaration files the value as " + str(st["declared_role"]) + ": the seed "
                                                                                                "view's `none` was served by another engine than the single-object standup; the "
                                                                                                "contrast is the reader's, formed as for every steered key"))
            batch.add("run_contrast", payload, ctx.producer, ctx.now)
            facts["contrasts"][okey] = {"anchor": anchor["run_id"], "n_pairs": len(shared), "delta_pp": payload["delta"], "delta_p": payload["delta_p"], "cohens_h": payload["effect_size"]}

    def _finish(self, group: str, facts: Dict[str, Any]) -> None:
        facts["refusals"] = [f"{it['key']} -- {it['why']}" for it in self.refusals.items]
        facts["n_refusals"] = len(self.refusals)
        self._facts[group] = facts

    # ---- one (tag, object key) pair -----------------------------------------------------------
    def _store_bundle(self, st: Mapping[str, Any], is_ceiling: bool, where_key: str) -> Dict[str, Any]:
        """The named files of one store directory, checked for presence and shape; raises :class:`_PairRefusal`."""
        top: Dict[str, Input] = st["top"]
        need_top = ["RESULT", "ladder_state.json"] + ([] if is_ceiling else ["cap_ladder.json"])
        missing = [n for n in need_top if n not in top]
        if missing:
            raise _PairRefusal(f"{'ceiling' if is_ceiling else 'base'} store lacks {missing}", st["dir"])
        keys = st["keys"]
        if len(keys) != 1:
            raise _PairRefusal(f"{'ceiling' if is_ceiling else 'base'} store holds {len(keys)} store key directories under cells/ (expected exactly one)", st["dir"])
        store_key = next(iter(keys))
        files: Dict[str, Input] = keys[store_key]
        missing = [n for n in ("gen_stats.json", "manifest.json", "rows.jsonl", "outcomes") if n not in files]
        if missing:
            raise _PairRefusal(f"{'ceiling' if is_ceiling else 'base'} store {store_key} lacks {missing}", os.path.join(st["dir"], "cells", store_key))
        try:
            ck = IC.parse_cell_key(store_key)
        except AdapterRefusal as exc:
            raise _PairRefusal(str(exc.why), os.path.join(st["dir"], "cells", store_key))
        with open(top["RESULT"].path, "r", encoding="utf-8") as fh:
            result_lines = [ln.strip() for ln in fh if ln.strip()]
        if not result_lines or RESULT_COMPLETE not in result_lines[-1]:
            raise _PairRefusal(f"RESULT's last line does not read {RESULT_COMPLETE} ({(result_lines[-1] if result_lines else '<empty>')[:120]})", top["RESULT"].uri)
        ladder_state = self._json(top["ladder_state.json"]) or {}
        if str(ladder_state.get("schema")) != LADDER_STATE_SCHEMA:
            raise _PairRefusal(f"ladder_state.json schema is {ladder_state.get('schema')!r}, not {LADDER_STATE_SCHEMA}", top["ladder_state.json"].uri)
        accepted = str(ladder_state.get("accepted_generation_attempt") or "")
        attempt = next((a for a in (ladder_state.get("attempts") or []) if isinstance(a, Mapping) and str(a.get("attempt_id")) == accepted), None)
        if attempt is None or str(attempt.get("result")) != "COMPLETE":
            raise _PairRefusal(f"ladder_state.json names no COMPLETE accepted attempt ({accepted!r})", top["ladder_state.json"].uri)
        man = self._json(files["manifest.json"]) or {}
        gs_all = self._json(files["gen_stats.json"]) or {}
        gs = gs_all.get(TASK) if isinstance(gs_all.get(TASK), Mapping) else None
        if gs is None:
            raise _PairRefusal(f"gen_stats.json carries no {TASK} block", files["gen_stats.json"].uri)
        outcomes = self._json(files["outcomes"]) or {}
        if str(outcomes.get("task")) != TASK or not isinstance(outcomes.get("items"), list):
            raise _PairRefusal(f"the outcome record is not {TASK}'s (task {outcomes.get('task')!r})", files["outcomes"].uri)
        cap_ladder = None
        if not is_ceiling:
            cap_ladder = self._json(top["cap_ladder.json"]) or {}
            if str(cap_ladder.get("schema")) != CAP_LADDER_SCHEMA:
                raise _PairRefusal(f"cap_ladder.json schema is {cap_ladder.get('schema')!r}, not {CAP_LADDER_SCHEMA}", top["cap_ladder.json"].uri)
            if str(cap_ladder.get("cell_id")) != store_key or bool((cap_ladder.get("source") or {}).get("is_ceiling")):
                raise _PairRefusal(f"cap_ladder.json names store {cap_ladder.get('cell_id')!r} (is_ceiling {(cap_ladder.get('source') or {}).get('is_ceiling')}), not this base store {store_key}",
                                   top["cap_ladder.json"].uri)
        # the scored rows: item -> (value, line, extra); the aggregate rows' extra (object sha, tree)
        scored: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
        agg_extra: Dict[str, Any] = {}
        consts: Dict[str, Any] = {}
        for ln, r in IC.iter_jsonl(files["rows.jsonl"].path):
            if str(r.get("task") or "") != TASK:
                continue
            ex = r.get("extra_json")
            try:
                exd = json.loads(ex) if isinstance(ex, str) and ex else (ex if isinstance(ex, Mapping) else {})
            except ValueError:
                exd = {}
            if r.get("item_id") == "__aggregate__":
                agg_extra.update({k: v for k, v in exd.items() if k in ("object_artifact_sha256", "producer_tree_sha", "route_kind", "sampling_mode", "view")})
                continue
            if str(r.get("metric")) != HEADLINE_METRIC:
                continue
            iid = str(r.get("item_id"))
            if iid in scored:
                raise _PairRefusal(f"rows.jsonl carries item {iid} twice", files["rows.jsonl"].uri)
            scored[iid] = (float(r.get("value")), ln, exd, LC.num_or_none(r.get("created_at")))
            for k in ("slurm_job_id", "env_digest", "model_id", "model_revision", "harness_config_hash", "intervention_hash", "created_at"):
                v = r.get(k)
                if v is not None:
                    consts.setdefault(k, set()).add(v if k != "created_at" else float(v))
        return {"dir": st["dir"], "store_key": store_key, "ck": ck, "top": top, "files": files, "manifest": man, "gen_stats": gs, "outcomes": outcomes,
                "cap_ladder": cap_ladder, "ladder_state": ladder_state, "attempt": attempt, "scored": scored, "agg_extra": agg_extra, "consts": consts,
                "result_last": result_lines[-1], "is_ceiling": is_ceiling}

    def _emit_pair(self, batch: RecordBatch, ctx: AdapterContext, tag: str, seed_n: int, paired: bool, selection: str, model_id: str, inst_row: Mapping[str, Any],
                   okey: str, slug: str, base_st: Mapping[str, Any], ceil_st: Optional[Mapping[str, Any]], pins: Mapping[str, Any], three: Optional[Mapping[str, Any]],
                   fstate: Optional[Mapping[str, Any]], pins_inp: Optional[Input], tn_inp: Optional[Input], fs_inp: Optional[Input], facts: Dict[str, Any],
                   target: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Files one (tag, object key) pair; returns what the tag's contrasts need (the run id, the final items, whether steered). ``target`` names
        the NON-seed target (the all-positions matrix's own row, v4): no store pins and no producer count record beside it, the default Inspect
        seed rule, its own minted instrument; ``seed_n`` is then None and ``selection`` the matrix row's spelling (``it``)."""
        seeds = self.seeds
        producer, now = ctx.producer, ctx.now
        base = self._store_bundle(base_st, False, okey)
        ceil = self._store_bundle(ceil_st, True, okey) if ceil_st is not None else None
        if base["ck"].cell_id != okey:
            raise _PairRefusal(f"the base store key spells object {base['ck'].cell_id!r} but the directory slug spells {okey!r}", base["dir"])
        man = base["manifest"]
        if str(man.get("cell_id") or "") != okey:
            raise _PairRefusal(f"manifest.json names object key {man.get('cell_id')!r}, not {okey!r}", base["files"]["manifest.json"].uri)
        # -- the store pins: the base tree, the ceiling store's tree, a ceiling store present iff pinned
        tree = str((man.get("producer") or {}).get("tree_sha") or "")
        if not _SHA1.match(tree):
            raise _PairRefusal("manifest.json:producer.tree_sha names no 40-hex tree", base["files"]["manifest.json"].uri)
        if target is None:
            pin_base = pins.get(pin_key(selection_key(seed_n, paired), okey, CAP))
            pin_ceiling = pins.get(pin_key(selection_key(seed_n, paired), okey, CEILING))
            if pin_base != tree:
                raise _PairRefusal(f"the base store's tree {tree[:12]} is not its store pin {str(pin_base)[:12] if pin_base else '<absent>'} ({pin_key(selection_key(seed_n, paired), okey, CAP)})", pins_inp.uri)
            if (ceil is not None) != (pin_ceiling is not None):
                raise _PairRefusal(f"a ceiling store is {'present' if ceil else 'absent'} but its store pin {pin_key(selection_key(seed_n, paired), okey, CEILING)} is {'absent' if ceil else 'present'}",
                                   ceil["dir"] if ceil else pins_inp.uri)
            if ceil is not None:
                rtree = str((ceil["manifest"].get("producer") or {}).get("tree_sha") or "")
                if rtree != pin_ceiling:
                    raise _PairRefusal(f"the ceiling store's tree {rtree[:12] or '<none>'} is not its store pin {str(pin_ceiling)[:12]}", ceil["files"]["manifest.json"].uri)
        else:
            # a non-seed target has no store pins: the tree is attested by ladder_state.json's ACCEPTED attempt (never the first line of RESULT), on
            # both stores, and the rung's declared source identity must be the base store's configuration hash (TREE_ATTESTATION_TARGET)
            pin_base = pin_ceiling = None
            att_sha = str(base["attempt"].get("sha") or "")
            if att_sha != tree:
                raise _PairRefusal(f"no store pins beside a non-seed target, and the base store's tree {tree[:12]} is not its accepted attempt's sha "
                                   f"{att_sha[:12] or '<none>'} (ladder_state.json attempts[{base['attempt'].get('attempt_id')}])", base["top"]["ladder_state.json"].uri)
            if ceil is not None:
                rtree = str((ceil["manifest"].get("producer") or {}).get("tree_sha") or "")
                ratt = str(ceil["attempt"].get("sha") or "")
                if not _SHA1.match(rtree) or rtree != ratt:
                    raise _PairRefusal(f"the ceiling store's tree {rtree[:12] or '<none>'} is not its accepted attempt's sha {ratt[:12] or '<none>'} (ladder_state.json)",
                                       ceil["files"]["manifest.json"].uri)
                sub_src = str(((ceil["manifest"].get("item_subset") or {}) if isinstance(ceil["manifest"].get("item_subset"), Mapping) else {}).get("source_identity_hash") or "")
                if sub_src != str(man.get("harness_config_hash") or ""):
                    raise _PairRefusal(f"the ceiling manifest's item_subset.source_identity_hash {sub_src!r} is not the base store's harness_config_hash "
                                       f"{man.get('harness_config_hash')!r}; the rung was not drawn from this base store", ceil["files"]["manifest.json"].uri)
        if ceil is not None:
            if str(ceil["manifest"].get("cell_id") or "") != okey or ceil["ck"].cell_id != okey:
                raise _PairRefusal(f"the ceiling store names object key {ceil['manifest'].get('cell_id')!r}, not {okey!r}", ceil["files"]["manifest.json"].uri)
        # -- the served model and identity
        served_hf = str(man.get("model_id") or "")
        if seeds.model_for_hf_id(served_hf) != model_id:
            raise _PairRefusal(f"the store serves {served_hf!r}, not the seed tag's model {model_id}", base["files"]["manifest.json"].uri)
        rev = str(man.get("model_revision") or "")
        pinned_rev = str((seeds.models.get(model_id) or {}).get("hf_revision") or "")
        if _SHA1.match(pinned_rev) and rev != pinned_rev:
            raise _PairRefusal(f"the store serves revision {rev[:12] or '<none>'}, not the model seed's {pinned_rev[:12]}", base["files"]["manifest.json"].uri)
        # -- the object: the all-positions key through ITS declaration and the all-positions roster / pins (v4); every other key as before
        arditi = okey == ARDITI_OBJECT_KEY
        decl: Optional[Dict[str, Any]] = None
        op: Optional[DG.ArditiAllposRung] = None
        try:
            if arditi:
                decl = self._arditi_declaration(tag, selection, target is not None)
                obj, op = self.resolve_arditi_object(decl)
            else:
                obj = self.resolve_object(okey, selection)
        except IC.Unresolved as exc:
            raise _PairRefusal(f"object: {exc.why}", base["files"]["manifest.json"].uri)
        steered = obj.object_kind not in ("none", "system_instruction")
        ident = str(man.get("serving_identity") or "")
        want_ident = STEERED_IDENTITY if steered else PASSTHROUGH_IDENTITY
        if ident != want_ident:
            raise _PairRefusal(f"serving_identity {ident!r} is not the {obj.object_kind} object's {want_ident!r}", base["files"]["manifest.json"].uri)
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        sha16 = str(fitp.get("object_sha") or "")
        attested = str(base["agg_extra"].get("object_artifact_sha256") or "")
        prov_sha = str((man.get("provenance") or {}).get("artifact_sha256") or "")
        if steered:
            if not _SHA16.match(sha16) or not str(obj.object_id).startswith(sha16):
                raise _PairRefusal(f"manifest.json:fit_provenance.object_sha {sha16!r} is no prefix of the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})",
                                   base["files"]["manifest.json"].uri)
            if attested and attested != obj.object_id:
                raise _PairRefusal(f"the rows' aggregate attests object {attested[:16]}…, the roster / pins {str(obj.object_id)[:16]}…", base["files"]["rows.jsonl"].uri)
            if prov_sha and prov_sha != obj.object_id:
                raise _PairRefusal(f"manifest.json:provenance.artifact_sha256 {prov_sha[:16]}… is not the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})",
                                   base["files"]["manifest.json"].uri)
        elif obj.object_kind == "system_instruction":
            sys_sha = str((man.get("resolved_config") or {}).get("system_sha") or "")
            si_sha = str(obj.object_id).split(":", 1)[1]
            if not (_SHA16.match(sys_sha) and si_sha.startswith(sys_sha)):
                raise _PairRefusal(f"manifest.json:resolved_config.system_sha {sys_sha!r} is no prefix of the declared prompt's text sha {si_sha[:16]}…", base["files"]["manifest.json"].uri)
        elif sha16 or attested:
            raise _PairRefusal("the unsteered store attests an object sha", base["files"]["manifest.json"].uri)
        roster_paired_of = None
        if paired:
            try:
                roster, _pins = self._roster_pins()
                blk = ((roster.get("targets") or {}).get(MODEL_KEY) or {}).get("views", {}).get(selection) or {}
                roster_paired_of = blk.get("paired_of")
            except IC.Unresolved:
                roster_paired_of = None
            if roster_paired_of is not None and str(roster_paired_of) != f"s{seed_n}":
                raise _PairRefusal(f"the roster's view {selection} says paired_of {roster_paired_of!r}; the tag says s{seed_n}", self.roster_path)
        # -- the seed rule: the replicate policy at the tag's base (a non-seed target: the default Inspect rule, no replicate block)
        rep = ((man.get("seeds") or {}).get("inspect") or {}).get("replicate") if isinstance(man.get("seeds"), Mapping) else None
        rep = rep if isinstance(rep, Mapping) else {}
        sb = LC.int_or_none(rep.get("seed_base"))
        if target is None:
            if str(rep.get("policy")) != INSPECT_SEED_POLICY or sb != seed_n:
                raise _PairRefusal(f"manifest.json:seeds.inspect.replicate is policy {rep.get('policy')!r} at seed_base {rep.get('seed_base')!r}; the tag's rule is "
                                   f"{INSPECT_SEED_POLICY} at {seed_n}", base["files"]["manifest.json"].uri)
        elif rep:
            raise _PairRefusal(f"manifest.json:seeds.inspect carries a replicate policy ({rep.get('policy')!r} at seed_base {rep.get('seed_base')!r}); the non-seed target is the "
                               "default Inspect seed rule — a replicate store is one of the seed views' and belongs under its tag", base["files"]["manifest.json"].uri)
        seed_rule_text = str(((man.get("seeds") or {}).get("inspect") or {}).get("rule") or "")
        if target is not None and not seed_rule_text:
            raise _PairRefusal("manifest.json:seeds.inspect.rule names no seed rule; the default rule must be spelled by the store", base["files"]["manifest.json"].uri)
        # -- items: the base outcome record x cap_ladder x rows, then the rung's items replace theirs
        items = self._merge_items(base, ceil, okey)
        counts = items["counts"]
        # -- the producer's own count of the same numbers: the three-number record's cell and / or the per-pair final-state record's entry
        # (a non-seed target has neither: the base reading is checked against gen_stats.json — everywhere — and its controls_summary.json when present)
        tn = fs_row = None
        if target is None:
            tn = self._three_numbers_row(three, selection, okey) if three is not None else None
            fs_row = self._final_state_row(fstate, selection, slug) if fstate is not None else None
            if tn is None and fs_row is None:
                raise _PairRefusal(f"neither {THREE_NUMBERS_FILE} nor {FINAL_STATE_FILE} carries an entry for {selection} / {okey}", (tn_inp or fs_inp or pins_inp).uri)
            if tn is not None:
                bad = self._cross_check_three(tn, base, ceil, items)
                if bad:
                    raise _PairRefusal(f"{THREE_NUMBERS_FILE} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), tn_inp.uri)
            if fs_row is not None:
                bad = self._cross_check_final_state(fs_row, ceil, items)
                if bad:
                    raise _PairRefusal(f"{FINAL_STATE_FILE} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), fs_inp.uri)
            producer_record = LC.rel_home((tn_inp if tn is not None else fs_inp).uri.rsplit("#", 1)[0])
            producer_version = record_version(tn) if tn is not None else RECORD_VERSION_FINAL_STATE_FILE
        else:
            producer_record = None
            producer_version = RECORD_VERSION_NONE
            summary_inp = target.get("summary")
            if summary_inp is not None:
                bad = self._cross_check_target_summary(self._json(summary_inp) or {}, base, items, obj.object_id, tree)
                if bad:
                    raise _PairRefusal(f"{TARGET_SUMMARY_FILE} disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), summary_inp.uri)
        # -- decode: the sent numbers against the registered policy row and the instrument's columns
        gs = base["gen_stats"]
        block = {k: gs.get(k) for k in DP.PARAM_KEYS}
        sent = {k: v for k, v in block.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        inst_numbers = {k: inst_row.get(f"decode_{k}") for k in DP.PARAM_KEYS}
        inst_match = "match" if self._numbers_agree(block, inst_numbers) else "mismatch"
        policy, policy_why = self._match_policy(model_id, block)
        if ceil is not None:
            rblock = {k: ceil["gen_stats"].get(k) for k in DP.PARAM_KEYS}
            if not self._numbers_agree(block, rblock):
                raise _PairRefusal(f"the ceiling store sent decode {RP._numbers(rblock)}, the base {RP._numbers(block)}; one ladder must send one row", ceil["files"]["gen_stats.json"].uri)
        # -- the all-positions declaration binds the store: its out roots, its tuples per cap, its sampling row, its positions, the standup
        if arditi:
            bad = self._check_arditi_declaration(decl, op, tag, selection, slug, base, ceil, block)
            if bad:
                raise _PairRefusal(f"the declaration {decl['_path']} (row {decl.get('index')}) disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), decl["_path"])
        # -- time, job, substrate
        final = ceil if ceil is not None else base
        when_base_start = IC.utc_from_iso(base["attempt"].get("started")) or self._rows_time(base)
        when_final = IC.utc_from_iso(final["attempt"].get("ended")) or self._rows_time(final)
        if when_final is None or when_base_start is None:
            raise _PairRefusal("ladder_state.json names no start / end for the accepted attempt and rows.jsonl carries no created_at", base["top"]["ladder_state.json"].uri)
        job = str((man.get("row_constants") or {}).get("slurm_job_id") or "")
        att_job = str(base["attempt"].get("job_id") or "")
        if not job or job != att_job:
            raise _PairRefusal(f"manifest.json:row_constants.slurm_job_id {job!r} and ladder_state.json's accepted attempt job {att_job!r} disagree", base["files"]["manifest.json"].uri)
        ceiling_job = (str((ceil["manifest"].get("row_constants") or {}).get("slurm_job_id") or "") if ceil is not None else None)
        witness = SC.substrate_of_routing_table(str(man.get("routing_table") or ""))
        if witness is None:
            raise _PairRefusal(f"manifest.json:routing_table {str(man.get('routing_table'))!r} is on no cluster's shared root (/workspace-vast/ = cluster A, /workspace/ = cluster B)",
                               base["files"]["manifest.json"].uri)
        substrate = (witness[0], "recorded", LC.confidence_entry("recorded", f"manifest.json:routing_table (the /{'workspace-vast' if witness[0] == 'cluster_a' else 'workspace'}/ shared root); "
                                                                             f"ladder_state.json attempts[{base['attempt'].get('attempt_id')}].cluster = {base['attempt'].get('cluster')}"))
        # -- the run
        final_cap = CEILING if ceil is not None else CAP
        state = "final_at_ceiling" if ceil is not None else "final_at_base"
        geometry = man.get("geometry") if isinstance(man.get("geometry"), Mapping) else {}
        rgeometry = (ceil["manifest"].get("geometry") if ceil is not None and isinstance(ceil["manifest"].get("geometry"), Mapping) else {})
        frameworks = man.get("frameworks") if isinstance(man.get("frameworks"), Mapping) else {}
        minted: Optional[Dict[str, Any]] = None
        if target is not None:
            # the non-seed target sits on ITS OWN instrument, minted from the record (the default seed rule is not the seed views' replicate rule);
            # the seed instrument's decode match rides beside it, and the run's own match is against the minted row
            inst_row, inst_payload, minted = self._mint_target_instrument(inst_row, base, man, block, policy, policy_why, frameworks, geometry, model_id, tag, slug,
                                                                          seed_rule_text, inst_match)
            if inst_payload is not None:
                batch.add("instrument", inst_payload, producer, now, observe=False)
                facts["instrument_minted"] = {"instrument_id": str(inst_row["instrument_id"]), "changed_fields": sorted(minted["changed_fields"])}
            inst_match = "match" if self._numbers_agree(block, {k: inst_row.get(f"decode_{k}") for k in DP.PARAM_KEYS}) else "mismatch"
        instrument_id = str(inst_row["instrument_id"])
        producer_key = producer_key_for(base["store_key"], tag, paired)
        rid = merged_run_id(base["store_key"], okey, instrument_id, tag=tag, paired=paired)
        n_scored, correct = counts["n_scored_final"], counts["correct_final"]
        acc = (correct / n_scored) if n_scored else None
        bound_lo = correct / N_ITEMS
        bound_hi = (correct + counts["n_unresolved"]) / N_ITEMS
        three_block = {"n_attempted": N_ITEMS, "n_scored_final": n_scored, "correct_final": correct, "acc_over_scored": acc, "acc_over_attempted": bound_lo,
                       "acc_over_attempted_upper_bound": bound_hi, "n_unresolved_neither_hit_nor_miss": counts["n_unresolved"], "n_unresolved_ladder_only": counts["n_unresolved_ladder_only"],
                       "n_rung_drawn_that_were_scored_at_base": counts["n_redrawn_scored_at_base"], "n_loops": counts["n_loops"],
                       "n_at_cap_ceiling": counts["n_ceiling"], "n_at_cap_not_escalated": counts["n_not_escalated"], "n_unscored_final": counts["n_unscored_final"],
                       "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_scored_at_cap_final": counts["n_scored_at_cap"], "n_at_cap_final": counts["n_at_cap_final"],
                       "n_at_cap_8192": counts["n_at_cap_base"], "n_eligible_for_rung": counts["n_eligible"], "ladder_exit": items["exit"],
                       "base_reading": {"acc": counts["acc_base"], "correct": counts["correct_base"], "n_scored": counts["n_scored_base"]},
                       "ceiling_reading": ({"n": counts["n_ceil_drawn"], "n_scored": counts["n_scored_ceil"], "correct": counts["correct_ceil"], "still_at_cap": counts["n_ceiling"]} if ceil else None),
                       "loop_rule": (base["cap_ladder"].get("rules") or {}).get("degenerate"), "at_cap_rule": (base["cap_ladder"].get("rules") or {}).get("at_cap"),
                       "eligible_rule": (base["cap_ladder"].get("rules") or {}).get("trigger"), "producer_record": producer_record,
                       "producer_record_version": producer_version, "producer_record_agrees": True,
                       "unresolved_definition": "item-wise: an at-cap item with no answer at its final draw — a loop never re-drawn, a truncated item the ladder did not escalate, "
                                                "or an item still at cap at the ceiling (the producer's widened third state, 2026-09-21 09:23Z)"}
        extra: Dict[str, Any] = {
            "seed_replicate_label": selection, "fit_seed": (seed_n if steered else None), "inspect_seed_base": seed_n, "decode_seed_policy": INSPECT_SEED_POLICY,
            "decode_seed_rule_text": seed_rule_text, "escalation_state": state, "token_cap_ladder": list(LADDER), "final_cap": final_cap,
            "ceiling_item_ids": sorted(items["ceiling_ids"]), "n_ceiling_items_merged": len(items["ceiling_ids"]),
            "store_uri_base": LC.rel_home(base["dir"]), "store_uri_ceiling": (LC.rel_home(ceil["dir"]) if ceil else None),
            "store_key_base": base["store_key"], "store_key_ceiling": (ceil["store_key"] if ceil else None),
            "store_pins": {str(CAP): pin_base, str(CEILING): pin_ceiling}, "store_pins_record": (LC.rel_home(pins_inp.uri.rsplit("#", 1)[0]) if pins_inp is not None else None),
            "cluster_job_ids": {str(CAP): job, str(CEILING): ceiling_job}, "job_names": {str(CAP): base["attempt"].get("job_name"), str(CEILING): (ceil["attempt"].get("job_name") if ceil else None)},
            "node_by_cap": {str(CAP): base["attempt"].get("node"), str(CEILING): (ceil["attempt"].get("node") if ceil else None)},
            "harness_config_hash_by_cap": {str(CAP): man.get("harness_config_hash"), str(CEILING): (ceil["manifest"].get("harness_config_hash") if ceil else None)},
            "served_window_by_cap": {str(CAP): LC.int_or_none(geometry.get("max_model_len")), str(CEILING): LC.int_or_none(rgeometry.get("max_model_len"))},
            "served_max_num_seqs_by_cap": {str(CAP): LC.int_or_none(geometry.get("max_num_seqs")), str(CEILING): LC.int_or_none(rgeometry.get("max_num_seqs"))},
            "three_numbers": three_block,
            "item_state_counts": {"correct": correct, "incorrect": n_scored - correct, ITEM_LABEL_AT_CAP_UNRESOLVED: counts["n_at_cap_unresolved"], ITEM_LABEL_UNSCORED: counts["n_unscored_below_cap"],
                                  "scored_at_cap": counts["n_scored_at_cap"]},
            "at_cap_classes": {"loop": counts["n_loops_unresolved"], "ceiling": counts["n_ceiling"], "not_escalated": counts["n_not_escalated"]},
            "served_model_id_hf": served_hf, "served_model_revision": rev or None, "serving_identity": ident, "served_tree_sha": tree, "object_sha16": (sha16 or None),
            "object_attested_sha256": (attested or None), "fit_model_id_hf": fitp.get("model_id"), "fit_transfer": fitp.get("transfer"),
            "steer_positions_legacy": man.get("steer_positions"), "producer_cell_id_legacy": okey, "producer_slug_legacy": slug, "producer_view_legacy": selection,
            "inspect_ai_version": frameworks.get("inspect_ai"), "inspect_evals_version": frameworks.get("inspect_evals"), "framework_legacy": gs.get("framework"),
            "decode_policy_ref_legacy": (policy.policy_id if policy else None), "decode_policy_reader_match": ("exact by the stated sampling numbers" if policy else policy_why),
            "sampling_mode_legacy": gs.get("sampling_mode") or base["agg_extra"].get("sampling_mode") or (man.get("sampling") or {}).get("mode"),
            "generation_state": "generated; rule-scored at generation (inspect_evals gsm8k numeric match), no judge", "source_legacy": SOURCE,
            "matrix": "configs/sdfbase/gsm8k/matrix_qwen_seedrep.json (rows keyed by (tag, object key), never by index)",
        }
        if not steered:
            extra["decode_replicate"] = {"trial_index": seed_n, "seed_base": seed_n, "twin_tags_legacy": [f"{MODEL_KEY}_s{k}" for k in (1, 2, 3) if k != seed_n], "rule": UNFITTED_TRIAL_RULE,
                                         "coexist_ruling_needed": f"a roster coexist ruling over lane_sources_legacy [{SOURCE}] x the seed tags keeps the trials selected together"}
        if paired:
            # a standup replicate of seed view s<N>: the same objects and the same seed base, re-served; its own `none` is its anchor
            extra.update(paired_of=f"s{seed_n}", standup_replicate_of_view=f"{MODEL_KEY}_s{seed_n}", roster_paired_of=roster_paired_of, producer_key_rule=PAIRED_KEY_RULE,
                         oos_tuple=man.get("oos_experiment"), anchor_rule="the paired view's OWN native `none` (this tag's), never the seed view's")
            if not steered:
                extra["decode_replicate"]["standup_replicate_of_view"] = f"{MODEL_KEY}_s{seed_n}"
                extra["decode_replicate"]["twin_tags_legacy"] = sorted({f"{MODEL_KEY}_s{k}" for k in (1, 2, 3)} | {f"{MODEL_KEY}_s{k}p" for k in (1, 2, 3)} - {tag})
                extra["decode_replicate"]["coexist_ruling_needed"] += " and the paired views (the same seed base, re-served)"
        lead = f"GSM8K paired-200 thinking ON, seed tag {tag} {okey}"
        if arditi:
            # ONE object on every seed view (DECODE_TRIAL_RULE): the seed rows are decode-seed trials of it, the matrix's own row a default-seed point
            declared_role = decl.get("_role")
            single_object = {"object_key": okey, "staged_id": str(obj.staged_id), "object_id": obj.object_id, "operating_point": op.rung, "standup_legacy": op.standup_tag,
                             "c": float(op.coefficient), "unit": op.unit, "declaration": {k: decl.get(k) for k in ("_path", "_schema", "_sha256", "index", "tag", "job_id", "role")},
                             "declared_role": declared_role, "rule": DECODE_TRIAL_RULE}
            extra.update(fit_seed=None, oos_tuple={str(CAP): man.get("oos_experiment"), str(CEILING): (ceil["manifest"].get("oos_experiment") if ceil else None)},
                         declared_role=declared_role, single_object=single_object, matrix=f"{decl['_path']} ({decl['_schema']}; the row for tag {tag}, index {decl.get('index')})")
            if target is None:
                extra["decode_replicate"] = {"trial_index": seed_n, "seed_base": seed_n, "twin_tags_legacy": [f"{MODEL_KEY}_s{k}" for k in (1, 2, 3) if k != seed_n], "rule": DECODE_TRIAL_RULE,
                                             "single_object": {"object_key": okey, "staged_id": str(obj.staged_id), "object_id": obj.object_id, "operating_point": op.rung},
                                             "coexist_ruling_needed": (f"a roster coexist ruling over lane_sources_legacy [{SOURCE}] x campaigns [{CAMPAIGN}] x model_ids [{model_id}] x object_ids "
                                                                       f"[{obj.object_id}] x eval_ids [{EVAL_ID}] x instrument_ids [{instrument_id}] x views_legacy "
                                                                       f"[{', '.join(f'{MODEL_KEY}_s{k}' for k in (1, 2, 3))}] keeps the three trials selected together (the user's, never a reader default)")}
                lead = f"GSM8K paired-200 thinking ON, seed tag {tag} {okey} (ONE object on every seed view: decode-seed trial {seed_n}, {op.rung} c = {op.coefficient:g} {op.unit})"
            else:
                extra.update(seed_replicate_label=None, inspect_seed_base=None, decode_seed_policy=INSPECT_SEED_POLICY_DEFAULT, store_pins=None, store_pins_record=None,
                             tree_attestation=TREE_ATTESTATION_TARGET, instrument_minted=minted, trial_rule=DEFAULT_SEED_TRIAL_RULE,
                             producer_summary=(LC.rel_home(target["summary"].uri) if target.get("summary") is not None else None),
                             mirror_provenance=(LC.rel_home(target["provenance"].uri) if target.get("provenance") is not None else None),
                             target_mirror=f"{TARGET_MIRROR_DIR}/{TARGET_CONTROLS_DIR}/{tag}/{slug}", accepted_attempt={str(CAP): base["attempt"].get("attempt_id"), str(CEILING): (ceil["attempt"].get("attempt_id") if ceil else None)},
                             attempts_before_accepted={str(CAP): [str(a.get("attempt_id")) for a in (base["ladder_state"].get("attempts") or []) if isinstance(a, Mapping) and str(a.get("result")) != "COMPLETE"]})
                lead = f"GSM8K paired-200 thinking ON, {tag} {okey} (the all-positions matrix's own row, default Inspect seed, {op.rung} c = {op.coefficient:g} {op.unit}; a single default-seed point, not a seed trial)"
        notes = (f"{lead}: {correct}/{n_scored} correct over scored = {acc:.4f}" if acc is not None else f"{lead}: no scored item")
        notes += (f"; {counts['n_at_cap_unresolved']}/{N_ITEMS} at the cap and unresolved (neither hit nor miss: {counts['n_loops']} loops, {counts['n_ceiling']} still at cap at {CEILING:,}, "
                  f"{counts['n_not_escalated']} truncated at {CAP:,} without a ceiling)"
                  + (f", {counts['n_unscored_below_cap']} unscored below the cap (unterminated think block)" if counts["n_unscored_below_cap"] else "")
                  + f"; correct/200 = {bound_lo:.3f} is the BOUND, not the value; escalation_state {state}"
                  + (f", {len(items['ceiling_ids'])} item(s) merged from the {CEILING:,} ceiling store" if ceil else ""))
        try:
            payload = LC.run_payload_base(seeds, run_id=rid, producer_key=producer_key, object_key=okey, view_legacy=tag, cell_id_legacy=okey, obj=obj, eval_id=EVAL_ID,
                                          instrument_row=inst_row, instrument_seed_id=INSTRUMENT_SEED, lifecycle_state="generated", when_utc=when_final,
                                          when_source="ladder_state.json: the accepted attempt's ended (the final store's; the base attempt's started for queued / started)",
                                          token_cap_tokens=final_cap, judge_id=None, judge_seed_id=None, substrate=substrate, lane_source_legacy=SOURCE,
                                          store_uri=LC.rel_home(base["dir"]), instrument_match=inst_match, extra=extra, notes=notes)
        except KeyError as exc:
            raise _PairRefusal(f"run row: {exc}", base["files"]["manifest.json"].uri)
        fc = payload["field_confidence_json"]
        # a decode-seed trial (an unfitted key; the all-positions object on a seed view) files the seed index; a fit replicate and the non-seed target trial 0
        trial_index = seed_n if (not steered or (arditi and target is None)) else 0
        payload.update(producer_session=PRODUCER_SESSION, trial_index=trial_index, seed_base=seed_n, queued_utc=when_base_start, started_utc=when_base_start,
                       n_items_planned=N_ITEMS, n_items_generated=N_ITEMS, n_items_judged=n_scored, model_revision=(rev if _SHA1.match(rev) else None),
                       intervention_hash=man.get("intervention_hash"), harness_config_hash=man.get("harness_config_hash"),
                       env_digest=(next(iter(base["consts"].get("env_digest") or []), None)), cluster_job_id=job, job_name=base["attempt"].get("job_name"),
                       node_hostname=base["attempt"].get("node"), serving_identity=ident, token_cap_rung_index=LADDER.index(final_cap), token_cap_ladder_json=list(LADDER),
                       decode_seed_rule=DECODE_SEED_RULE, decode_sent_json=sent, decode_temperature_sent=sent.get("temperature"), decode_top_p_sent=sent.get("top_p"),
                       decode_top_k_sent=(int(sent["top_k"]) if sent.get("top_k") is not None else None), code_sha=tree,
                       harness_version=(f"inspect-ai {frameworks.get('inspect_ai')} / inspect_evals {frameworks.get('inspect_evals')} ({gs.get('framework')})"
                                        if frameworks.get("inspect_ai") else payload["harness_version"]),
                       context_window_tokens=LC.int_or_none(geometry.get("max_model_len")),
                       store_completion_artifact_uri=final["files"]["gen_stats.json"].uri, store_completion_sha256=final["files"]["gen_stats.json"].sha256)
        if LC.int_or_none(geometry.get("max_num_seqs")) is not None:
            payload.update(served_max_num_seqs=LC.int_or_none(geometry.get("max_num_seqs")), served_max_num_seqs_source="attested")
            fc["served_max_num_seqs"] = LC.confidence_entry("recorded", f"manifest.json:geometry.max_num_seqs (source {geometry.get('source')}); the rung's rides extra.served_max_num_seqs_by_cap")
        if policy is not None:
            payload["decode_source"] = policy.decode_source
            fc["decode_source"] = LC.confidence_entry("recorded", f"the registered policy row {policy.policy_id} the sent numbers equal (its decode_source)")
        else:
            fc["decode_source"] = LC.unverified(policy_why)
        fc["code_sha"] = LC.confidence_entry("recorded", "manifest.json:producer.tree_sha = the store pin")
        fc["model_revision"] = LC.confidence_entry("recorded", "manifest.json:model_revision (= models.yaml hf_revision)")
        fc["cluster_job_id"] = LC.confidence_entry("recorded", "manifest.json:row_constants.slurm_job_id = ladder_state.json's accepted attempt (the base store's job; the rung's rides extra.cluster_job_ids)")
        fc["node_hostname"] = LC.confidence_entry("recorded", "ladder_state.json: the accepted attempt's node (the base store's)")
        fc["token_cap_tokens"] = LC.confidence_entry("recorded", f"the ladder's final state: {final_cap} ({state}); gen_stats.json:max_tokens of the final store")
        fc["seed_base"] = LC.confidence_entry("recorded", "manifest.json:seeds.inspect.replicate.seed_base (the replicate policy at the tag's index)")
        fc["decode_seed_rule"] = LC.confidence_entry("derived", f"one seed per item: {seed_rule_text or 'contracts.SEED_RULE_INSPECT_REPLICATE'} (k = 1, so the closure is the item; "
                                                                "the base rides seed_base and the rule's text extra.decode_seed_rule_text)")
        fc["decode_sent_json"] = LC.confidence_entry("recorded", "gen_stats.json (the client's sampler, resolved_sampler.json classes.client AGREES over the 200 requests)")
        fc["instrument_match"] = LC.confidence_entry("derived", f"the sent numbers compared to instrument seed {INSTRUMENT_SEED}'s decode columns (bound to {model_id}): {inst_match}")
        if target is not None:
            fc["seed_base"] = LC.confidence_entry("derived", "the default Inspect seed rule (manifest.json:seeds.inspect carries the rule and no replicate block): no seed base")
            fc["decode_seed_rule"] = LC.confidence_entry("derived", f"one seed per item: {seed_rule_text} (k = 1, so the closure is the item; the default rule has no base, "
                                                                    "extra.decode_seed_policy default; the rule's text extra.decode_seed_rule_text)")
            fc["instrument_match"] = LC.confidence_entry("derived", (f"the run sits on its own instrument {instrument_id[:12]}…, minted from this record (extra.instrument_minted): the sent "
                                                                     f"numbers compared to its decode columns: {inst_match}; against the seed instrument {INSTRUMENT_SEED} they read "
                                                                     f"{(minted or {}).get('seed_instrument_decode_match')}, and the seed rule differs (default, not replicate), which is why the point is never forced onto it")
                                                         if minted is not None else
                                                         f"the record's facts hash to the seed instrument {INSTRUMENT_SEED}'s own id (nothing to mint); the sent numbers compared to its decode columns: {inst_match}")
            fc["instrument_id"] = LC.confidence_entry("derived", (f"minted by this reader from the record through identity.instrument_id (the seed instrument {INSTRUMENT_SEED}'s row with the store's "
                                                                  f"harness_config_hash, inspect versions, served window, ladder rule and sent numbers; extra.instrument_minted.changed_fields)")
                                                      if minted is not None else fc["instrument_id"]["source"])
        fc["harness_version"] = LC.confidence_entry("recorded", "manifest.json:frameworks (inspect_ai / inspect_evals) and gen_stats.json:framework")
        fc["context_window_tokens"] = LC.confidence_entry("recorded", "manifest.json:geometry.max_model_len (the base store's; the rung's rides extra.served_window_by_cap)")
        fc["n_items_planned"] = LC.confidence_entry("derived", f"the instrument's item set (n_items {N_ITEMS}); gen_stats.json:n and the outcome record's n_attempted_items agree")
        fc["n_items_generated"] = LC.confidence_entry("recorded", "the outcome record: every attempted item has a stop reason (scored, or at the cap)")
        fc["n_items_judged"] = LC.confidence_entry("derived", "the SCORED items at the ladder's final state (the user's rule: unresolved items are neither hit nor miss); rule-scored at generation")
        fc["queued_utc"] = fc["started_utc"] = LC.confidence_entry("recorded", "ladder_state.json: the base store's accepted attempt started")
        fc["generated_utc"] = LC.confidence_entry("recorded", "ladder_state.json: the final store's accepted attempt ended")
        if steered and man.get("steer_positions") in V.APPLIED_POSITIONS_MODES:
            payload["applied_positions_mode"] = str(man["steer_positions"])
            fc["applied_positions_mode"] = LC.confidence_entry("recorded", "manifest.json:steer_positions")
        if not steered:
            payload["strength_unit"] = "unverified"
            fc["strength_unit"] = LC.unverified("an unsteered / prompted object key: no object on the served path, so no dose and no unit (the column default)")
            fc["trial_index"] = LC.confidence_entry("derived", UNFITTED_TRIAL_RULE + f"; manifest.json:seeds.inspect.replicate.seed_base = {seed_n}")
        if arditi:
            # the operating point's ONE strength stamp (dose_grammar: c in raw_coefficient, the nominal c / c_served, the operating-point id `<rung>__arditi_allpos`),
            # the spelling the all-positions reader files, so the paper keys agree to the letter with its rows on the same object; never the seed view's filed 1.0
            payload.update(op.strength_columns())
            fc.update(op.strength_confidence(f"the declaration's bank view `{op.rung}` ({decl['_path']}, row {decl.get('index')}) and the pins' staged object {op.staged_id}"))
            payload["extra"].update(op.extra())
            # the direction is fit on and served on the instruct checkpoint: the rung's own legacy spelling (instruct_direct), never the seed view's base_to_instruct
            op_mv = seeds.model_for_view_legacy(op.standup_tag)
            if op_mv.get("transfer_kind") in V.TRANSFER_KINDS:
                payload["transfer_kind"] = op_mv["transfer_kind"]
                fc["transfer_kind"] = LC.confidence_entry("recorded", f"legacy_spellings.yaml model_keys[{op.standup_tag}].transfer_kind (the direction is fit on and served on the instruct "
                                                                      f"checkpoint: manifest.json:fit_provenance.transfer {fitp.get('transfer')!r}; the standup {tag}'s own spelling is "
                                                                      f"{seeds.model_for_view_legacy(tag).get('transfer_kind')!r})")
            if target is None:
                fc["trial_index"] = LC.confidence_entry("derived", DECODE_TRIAL_RULE + f"; the declaration {decl['_path']} names the one object {str(obj.object_id)[:12]}… on every seed "
                                                                   f"view, so this row is trial {seed_n} (manifest.json:seeds.inspect.replicate.seed_base = {seed_n})")
            else:
                fc["trial_index"] = LC.confidence_entry("derived", DEFAULT_SEED_TRIAL_RULE + f"; manifest.json:seeds.inspect carries no replicate block ({seed_rule_text})")
        batch.add("run", payload, producer, now)
        facts["runs"] += 1
        if ceil is not None:
            facts["ceiling_merged"] += 1
        # NO state event: the record's `generated` is the state and the harvester's rule-scored completion stamps `ingested` (seedrep_cluster's MMLU
        # precedent: a re-emitted `generated` event would fold after that stamp and regress the row)
        comp_id = merge_computation_id()
        code_sha = str(producer.get("code_sha") or "")
        batch.add("computation", {"computation_id": comp_id, "algorithm": MERGE_ALGORITHM, "version": MERGE_VERSION, "seed_provenance": MERGE_SEED_PROVENANCE,
                                  "code_sha": (code_sha if _SHA1.match(code_sha) else "unverified"),
                                  "notes": "the ceiling store's items replace the base store's item-wise (never averaged); correct / scored recounted over the merged 200 outcomes"},
                  producer, now)
        rows_uri = LC.rel_home(base["files"]["rows.jsonl"].uri)
        lo, hi = IC.wilson(correct, n_scored) if n_scored else (None, None)
        batch.add("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": comp_id,
                                 "producer_metric_spelling": "acc", "value": acc, "k": correct, "n": n_scored, "ci_lo": lo, "ci_hi": hi, "ci_method": ("wilson_95" if n_scored else None),
                                 "n_abstentions": None, "n_incomplete": counts["n_unscored_final"], "computed_by": "registry", "computed_utc": when_final, "source_uri": rows_uri,
                                 "extra": {"role": "THE VALUE: correct over SCORED at the ladder's final state (the user's rule: degeneration is scored separately)",
                                           "recounted_from": "the base outcome record with the ceiling store's items merged item-wise", "base_reading": three_block["base_reading"],
                                           "ceiling_reading": three_block["ceiling_reading"], "n_unresolved_neither_hit_nor_miss": counts["n_unresolved"],
                                           "n_unscored_final": counts["n_unscored_final"], "equals_gen_stats_acc": (ceil is None),
                                           "producer_record": three_block["producer_record"]}}, producer, now)
        if tn is not None:
            producer_spellings: Optional[Dict[str, Any]] = {"acc_over_attempted_after_rung": tn.get("acc_over_attempted_after_rung"), "acc_over_attempted_lower_bound": tn.get("acc_over_attempted_lower_bound"),
                                                            "acc_over_attempted_upper_bound": tn.get("acc_over_attempted_upper_bound")}
        elif fs_row is not None:
            producer_spellings = {"acc_over_attempted_final": fs_row.get("acc_over_attempted_final"), "n_unresolved_final": fs_row.get("n_unresolved_final")}
        else:
            producer_spellings = None                   # a non-seed target: no producer count record spells the bound
        bound_extra: Dict[str, Any] = {"role": "THE BOUND, never the value: correct over the 200 attempted; the interval's lower end counts the unresolved items wrong, the upper end right",
                                       "n_unresolved_neither_hit_nor_miss": counts["n_unresolved"], "n_unresolved_ladder_only": counts["n_unresolved_ladder_only"],
                                       "n_at_cap_not_escalated": counts["n_not_escalated"], "n_unscored_below_cap_counted_as_miss": counts["n_unscored_below_cap"],
                                       "unresolved_definition": three_block["unresolved_definition"],
                                       "producer_spellings": producer_spellings,
                                       "metric_role_expected": "secondary (a blank computation_id ranks after the value's computed row at one identity)"}
        if target is not None:
            bound_extra["producer_counts"] = ("the producer's own counts: gen_stats.json n / n_scored / n_at_cap (and controls_summary.json when present) over the 200 attempted, the "
                                              "correct items the rows the producer scored; no producer record spells this ratio, so the row carries no producer spelling")
        batch.add("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": "",
                                 "producer_metric_spelling": (BOUND_SPELLING if target is None else None), "value": bound_lo, "k": correct, "n": N_ITEMS, "ci_lo": bound_lo, "ci_hi": bound_hi,
                                 "ci_method": BOUND_CI_METHOD,
                                 "n_abstentions": None, "n_incomplete": counts["n_unresolved"], "computed_by": "producer", "computed_utc": when_final,
                                 "source_uri": (three_block["producer_record"] if target is None else LC.rel_home(base["files"]["gen_stats.json"].uri)),
                                 "extra": bound_extra}, producer, now)
        # -- items (whole rows): value 1 / 0, or NULL with the third state's label
        rows_sha = {CAP: base["files"]["rows.jsonl"].sha256, CEILING: (ceil["files"]["rows.jsonl"].sha256 if ceil else None)}
        for iid in sorted(items["final"]):
            it = items["final"][iid]
            label = None if it["value"] is not None else (ITEM_LABEL_UNSCORED if it["at_cap_class"] == UNSCORED_BELOW_CAP_CLASS else ITEM_LABEL_AT_CAP_UNRESOLVED)
            iextra = {"final_cap": it["cap"], "source_store": it["source"], "stop_reason": it["stop_reason"], "completion_tokens": it["completion_tokens"], "decode_seed": it["seed"],
                      "at_cap_class": (it["at_cap_class"] if it["at_cap_class"] in AT_CAP_CLASSES else None), "scored_at_cap": bool(it["value"] is not None and it["at_cap"]),
                      "unscored_reason": (UNSCORED_BELOW_CAP_CLASS if it["at_cap_class"] == UNSCORED_BELOW_CAP_CLASS else None),
                      "think_block_unterminated": it.get("think_block_unterminated"), "unparseable": it.get("unparseable"),
                      "base_value_replaced": it.get("base_value_replaced"), "base_at_cap": it.get("base_at_cap"), "text_sha256": it.get("text_sha256")}
            batch.add("item_result", IC.item_result_payload(rid, iid, HEADLINE_METRIC, it["value"], label, it["created"] or when_final, judgment_id="", trial_index=trial_index, extra=iextra),
                      producer, now)
            gen_art, gen_off = (rows_sha[it["cap"]], it["line"]) if it["line"] is not None else (None, None)
            batch.add("item_flags", IC.item_flags_payload(rid, iid, gen_art, gen_off, at_token_cap=(1 if it["at_cap"] else 0), finish_reason=it["stop_reason"],
                                                          response_tokens=it["completion_tokens"], decode_seed=(str(it["seed"]) if it["seed"] is not None else None), judgment_id="",
                                                          trial_index=trial_index, degenerate_rep_ratio_loop=(1 if it.get("classifier_loop") else (0 if it["degenerate"] is False else None)),
                                                          rep_ratio=it.get("rep_ratio"), tail_loop=it.get("tail_loop"),
                                                          extra={"at_token_cap_source": "the outcome record's at_cap (stop_reason == max_tokens at the request's cap; never re-tokenized)",
                                                                 "generation_pointer": ("rows.jsonl of the store the item's final draw came from (sha, line)" if it["line"] is not None
                                                                                        else "no row: an unscored item's generation stays in the store's private/ spool on cluster A"),
                                                                 "outcome_record": LC.rel_home((ceil if it["source"] == "ceiling" else base)["files"]["outcomes"].uri), "tail_rep_ratio": it.get("tail_rep_ratio")}),
                      producer, now)
        # -- the run-level shares (run_degeneration over the 200 attempted items)
        src_uri = LC.rel_home(base["top"]["cap_ladder.json"].uri)
        rules = base["cap_ladder"].get("rules") if isinstance(base["cap_ladder"].get("rules"), Mapping) else {}
        measured = {
            RAW_AT_CAP_MEASURE: dict(numerator=counts["n_at_cap_final"], denominator=N_ITEMS, unit="items", threshold_params={"cap_final": final_cap, "ladder": list(LADDER)},
                                     extra={"rule": rules.get("at_cap"), "at_ladder_final_state": True,
                                            "classes": {"loop": counts["n_loops_unresolved"], "ceiling": counts["n_ceiling"], "not_escalated": counts["n_not_escalated"], "scored_at_cap": counts["n_scored_at_cap"]},
                                            "n_at_cap_unresolved": counts["n_at_cap_unresolved"], "n_unscored_below_cap_not_counted": counts["n_unscored_below_cap"],
                                            "n_at_cap_8192_before_ceiling": counts["n_at_cap_base"], "source": src_uri}),
            LOOP_MEASURE: dict(numerator=counts["n_loops"], denominator=N_ITEMS, unit="items", threshold_params={"rep_ratio_gt": 0.5, "tail_tokens": 200, "cap": CAP},
                               extra={"rule": rules.get("degenerate"), "periodic_block_rule": rules.get("periodic_block"), "class": "loop: at cap at 8,192 AND degenerate by cap_ladder.json — never re-drawn", "source": src_uri}),
        }
        notes_nm = {"classifier": "the paper's GSM8K reader files the loop count under `classifier`; this store's classification is the rep_ratio rule (rep_ratio_loop)",
                    "contentless": "no contentless reading in the ladder gate's classification", "zlib_collapse": "no zlib reading in the ladder gate's classification",
                    "union5": "the five-detector union is not computed for these stores (the degeneracy-union reader's input, not here)",
                    IC.AT_CAP_RETOKENIZED: "the ladder gate reads the provider's stop reason, never a re-tokenized length"}
        for row in IC.declared_degeneration_rows(rid, "", measured, source_uri=src_uri, computed_utc=when_final, not_measured_notes=notes_nm):
            batch.add("run_degeneration", row, producer, now)
        batch.add("run_degeneration", IC.degeneration_payload(rid, COHERENT_MEASURE, "measured", judgment_id="", numerator=counts["n_ceiling"] + counts["n_not_escalated"], denominator=N_ITEMS,
                                                               unit="items", threshold_params={"cap_final": final_cap}, computed_utc=when_final, source_uri=src_uri,
                                                               extra={"class": "unresolved and not a loop: still at cap at the ceiling, or truncated at 8,192 without a rung",
                                                                      "classes": {"ceiling": counts["n_ceiling"], "not_escalated": counts["n_not_escalated"]}, "source": src_uri}), producer, now)
        facts["pairs"][okey] = {"run_id": rid, "acc_over_scored": acc, "correct": correct, "n_scored": n_scored, "acc_over_attempted": bound_lo, "upper_bound": bound_hi,
                                "n_unresolved": counts["n_unresolved"], "n_unresolved_ladder_only": counts["n_unresolved_ladder_only"], "producer_record_version": producer_version,
                                "n_loops": counts["n_loops"], "n_ceiling": counts["n_ceiling"], "n_not_escalated": counts["n_not_escalated"],
                                "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_scored_at_cap": counts["n_scored_at_cap"],
                                "ceiling_items_merged": len(items["ceiling_ids"]), "escalation_state": state, "ladder_exit": items["exit"], "instrument_match": inst_match,
                                "decode_policy": (policy.policy_id if policy else None), "jobs": {str(CAP): job, str(CEILING): ceiling_job}, "tree": tree,
                                "trial_index": trial_index, "object_id": obj.object_id, "store_key_base": base["store_key"], "producer_key": producer_key,
                                "n_loops_unresolved": counts["n_loops_unresolved"], "paired": paired}
        if arditi:
            facts["pairs"][okey].update(single_object=extra["single_object"], seed_base=seed_n, fit_seed=None, trial_rule=("DECODE_TRIAL_RULE" if target is None else "DEFAULT_SEED_TRIAL_RULE"),
                                        strength={k: payload.get(k) for k in ("strength_coefficient", "strength_unit", "strength_nominal", "is_operating_point", "operating_point_id_legacy")},
                                        transfer_kind=payload["transfer_kind"], instrument_id=instrument_id, instrument_minted=(minted is not None), oos_tuple=extra["oos_tuple"],
                                        declared_role=extra["declared_role"], base_reading=three_block["base_reading"])
        return {"run_id": rid, "items": items["final"], "steered": steered, "when": when_final, "declared_role": (extra.get("declared_role") if arditi else None)}

    # ---- items ---------------------------------------------------------------------------------
    @staticmethod
    def _outcome_items(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for it in bundle["outcomes"]["items"]:
            if not isinstance(it, Mapping) or not it.get("item_id"):
                raise _PairRefusal("the outcome record carries an item without an id", bundle["files"]["outcomes"].uri)
            iid = str(it["item_id"])
            if iid in out:
                raise _PairRefusal(f"the outcome record carries item {iid} twice", bundle["files"]["outcomes"].uri)
            at_cap, scored = bool(it.get("at_cap")), bool(it.get("scored"))
            v = it.get("value")
            if scored and v not in (0, 1, 0.0, 1.0):
                raise _PairRefusal(f"item {iid}: a scored item with value {v!r}", bundle["files"]["outcomes"].uri)
            if not scored and v is not None:
                raise _PairRefusal(f"item {iid}: an unscored item with value {v!r}", bundle["files"]["outcomes"].uri)
            out[iid] = {"at_cap": at_cap, "scored": scored, "value": (float(v) if scored else None), "stop_reason": it.get("stop_reason"),
                        "completion_tokens": LC.int_or_none(it.get("completion_tokens")), "seed": LC.int_or_none(it.get("seed")),
                        "think_block_unterminated": it.get("think_block_unterminated"), "unparseable": it.get("unparseable")}
        return out

    def _merge_items(self, base: Mapping[str, Any], ceil: Optional[Mapping[str, Any]], okey: str, n_items: int = N_ITEMS, thinking: Optional[str] = None) -> Dict[str, Any]:
        """The 200 items at the ladder's final state, from the base outcome record x ``cap_ladder.json`` x ``rows.jsonl``, the rung's items replacing
        theirs; every disagreement between the files is a pair refusal. ``n_items`` is the item set's size (200; a subclass whose base manifest
        declares another count passes it); ``thinking`` the ``gen_stats.json:thinking`` spelling the stores must carry (the class's, unless a
        subclass reading another model's stores passes the model's)."""
        thinking = thinking or self.THINKING_SPELLING
        cl = base["cap_ladder"]
        b_items = self._outcome_items(base)
        if len(b_items) != n_items:
            raise _PairRefusal(f"the base outcome record holds {len(b_items)} items, not {n_items}", base["files"]["outcomes"].uri)
        facts_by: Dict[str, Mapping[str, Any]] = {}
        for f in (cl.get("item_facts") or []):
            if isinstance(f, Mapping) and f.get("item_id"):
                facts_by[str(f["item_id"])] = f
        if set(facts_by) != set(b_items):
            raise _PairRefusal(f"cap_ladder.json:item_facts names {len(facts_by)} items, the outcome record {len(b_items)}; the two do not name the same items", base["top"]["cap_ladder.json"].uri)
        for iid, it in b_items.items():
            f = facts_by[iid]
            if bool(f.get("at_cap")) != it["at_cap"] or bool(f.get("scored")) != it["scored"] or (it["scored"] and float(f.get("value")) != it["value"]):
                raise _PairRefusal(f"item {iid}: cap_ladder.json:item_facts and the outcome record disagree (at_cap / scored / value)", base["top"]["cap_ladder.json"].uri)
        scored_rows = base["scored"]
        want = {iid: it["value"] for iid, it in b_items.items() if it["scored"]}
        if set(scored_rows) != set(want) or any(scored_rows[i][0] != want[i] for i in want):
            raise _PairRefusal(f"rows.jsonl's {len(scored_rows)} scored acc rows do not reproduce the outcome record's {len(want)} scored items", base["files"]["rows.jsonl"].uri)
        loops = {str(x) for x in (cl.get("non_terminating_ids") or [])}
        eligible = {str(x) for x in (cl.get("eligible_ids") or [])}
        at_cap = {iid for iid, it in b_items.items() if it["at_cap"]}
        deg = {iid for iid, f in facts_by.items() if bool(f.get("degenerate")) and bool(f.get("at_cap"))}
        if loops != deg:
            raise _PairRefusal(f"cap_ladder.json:non_terminating_ids ({len(loops)}) is not the set of at-cap items its item_facts mark degenerate ({len(deg)})", base["top"]["cap_ladder.json"].uri)
        if not loops <= at_cap or not eligible <= at_cap or (loops & eligible):
            raise _PairRefusal("cap_ladder.json: loops and eligible items are not disjoint subsets of the at-cap items", base["top"]["cap_ladder.json"].uri)
        if eligible != (at_cap - loops):
            raise _PairRefusal(f"cap_ladder.json:eligible_ids ({len(eligible)}) is not at-cap minus degenerate ({len(at_cap - loops)}); the boundary class is empty at k = 1", base["top"]["cap_ladder.json"].uri)
        if LC.int_or_none(cl.get("n_at_cap")) != len(at_cap) or LC.int_or_none(cl.get("n")) != n_items or LC.int_or_none(cl.get("cap")) != CAP or LC.int_or_none(cl.get("ceiling")) != CEILING:
            raise _PairRefusal(f"cap_ladder.json counts (n {cl.get('n')}, n_at_cap {cl.get('n_at_cap')}, cap {cl.get('cap')}, ceiling {cl.get('ceiling')}) disagree with the items", base["top"]["cap_ladder.json"].uri)
        exit_ = str(cl.get("exit") or "")
        if ceil is not None and exit_ != LADDER_EXIT_ESCALATE:
            raise _PairRefusal(f"a ceiling store is present but cap_ladder.json declares exit {exit_!r}, not {LADDER_EXIT_ESCALATE!r}", base["top"]["cap_ladder.json"].uri)
        if ceil is None and exit_ not in LADDER_EXITS_FINAL_AT_BASE:
            raise _PairRefusal(f"cap_ladder.json declares exit {exit_!r} but no ceiling store is present (the ladder is incomplete)", base["top"]["cap_ladder.json"].uri)
        gs = base["gen_stats"]
        n_scored_b = len(want)
        correct_b = int(round(sum(want.values())))
        acc_b = (correct_b / n_scored_b) if n_scored_b else None
        self._check_gen_stats(gs, n_items, n_scored_b, len(at_cap), acc_b, base["files"]["gen_stats.json"].uri, thinking=thinking)
        final: Dict[str, Dict[str, Any]] = {}
        ceiling_ids: set = set()
        r_items: Dict[str, Dict[str, Any]] = {}
        n_ceil_drawn = n_scored_r = correct_r = n_ceiling = 0
        if ceil is not None:
            subset = {str(x) for x in ((ceil["manifest"].get("item_subset") or {}).get("ids") or [])}
            if subset != eligible:
                raise _PairRefusal(f"the ceiling manifest's item_subset.ids ({len(subset)}) is not the base's eligible_ids ({len(eligible)})", ceil["files"]["manifest.json"].uri)
            r_items = self._outcome_items(ceil)
            if set(r_items) != subset:
                raise _PairRefusal(f"the ceiling outcome record covers {len(r_items)} items, not the {len(subset)} of item_subset", ceil["files"]["outcomes"].uri)
            r_want = {iid: it["value"] for iid, it in r_items.items() if it["scored"]}
            if set(ceil["scored"]) != set(r_want) or any(ceil["scored"][i][0] != r_want[i] for i in r_want):
                raise _PairRefusal("the ceiling store's rows.jsonl does not reproduce its outcome record's scored items", ceil["files"]["rows.jsonl"].uri)
            n_ceil_drawn, n_scored_r, correct_r = len(subset), len(r_want), int(round(sum(r_want.values())))
            self._check_gen_stats(ceil["gen_stats"], n_ceil_drawn, n_scored_r, sum(1 for x in r_items.values() if x["at_cap"]), ((correct_r / n_scored_r) if n_scored_r else None),
                                  ceil["files"]["gen_stats.json"].uri, cap=CEILING, thinking=thinking)
            ceiling_ids = subset
        for iid, it in sorted(b_items.items()):
            f = facts_by[iid]
            common = {"rep_ratio": LC.num_or_none(f.get("rep_ratio")), "tail_rep_ratio": LC.num_or_none(f.get("tail_rep_ratio")), "text_sha256": f.get("text_sha256"),
                      "tail_loop": (1 if bool(((f.get("periodic_block") or {}) if isinstance(f.get("periodic_block"), Mapping) else {}).get("loop")) else 0), "degenerate": bool(f.get("degenerate")),
                      "classifier_loop": (iid in loops)}
            if ceil is not None and iid in ceiling_ids:
                r = r_items[iid]
                line = ceil["scored"][iid][1] if r["scored"] else None
                created = IC.utc_from_epoch(ceil["scored"][iid][3]) if (r["scored"] and ceil["scored"][iid][3] is not None) else None
                final[iid] = dict(common, value=r["value"], cap=CEILING, source="ceiling", at_cap=r["at_cap"], stop_reason=r["stop_reason"], completion_tokens=r["completion_tokens"],
                                  seed=r["seed"], at_cap_class=self._item_class(r, loops, iid, "ceiling"), line=line, created=created,
                                  think_block_unterminated=r["think_block_unterminated"], unparseable=r["unparseable"], base_value_replaced=it["value"], base_at_cap=it["at_cap"])
            else:
                line = base["scored"][iid][1] if it["scored"] else None
                created = IC.utc_from_epoch(base["scored"][iid][3]) if (it["scored"] and base["scored"][iid][3] is not None) else None
                final[iid] = dict(common, value=it["value"], cap=CAP, source="base", at_cap=it["at_cap"], stop_reason=it["stop_reason"], completion_tokens=it["completion_tokens"],
                                  seed=it["seed"], at_cap_class=self._item_class(it, loops, iid, "not_escalated"), line=line, created=created,
                                  think_block_unterminated=it["think_block_unterminated"], unparseable=it["unparseable"], base_value_replaced=None, base_at_cap=it["at_cap"])
        n_scored_f = sum(1 for x in final.values() if x["value"] is not None)
        correct_f = int(round(sum(x["value"] for x in final.values() if x["value"] is not None)))
        n_loops = len(loops)                                                                    # the ladder gate's classifier: at cap AND degenerate
        n_loops_unresolved = sum(1 for x in final.values() if x["at_cap_class"] == "loop")      # of those, the ones with no answer (a scored loop's value counts)
        n_not_escalated = sum(1 for x in final.values() if x["at_cap_class"] == "not_escalated")
        n_ceiling = sum(1 for x in final.values() if x["at_cap_class"] == "ceiling")
        n_unscored_below_cap = sum(1 for x in final.values() if x["at_cap_class"] == UNSCORED_BELOW_CAP_CLASS)
        n_scored_at_cap = sum(1 for x in final.values() if x["value"] is not None and x["at_cap"])
        n_at_cap_unresolved = n_loops_unresolved + n_ceiling + n_not_escalated
        if n_at_cap_unresolved + n_unscored_below_cap != n_items - n_scored_f:
            raise _PairRefusal("the item classes do not partition the unscored items", base["files"]["outcomes"].uri)
        n_redrawn_scored_at_base = sum(1 for x in final.values() if x["source"] == "ceiling" and x.get("base_value_replaced") is not None)
        counts = {"n_scored_final": n_scored_f, "correct_final": correct_f, "n_unscored_final": n_items - n_scored_f, "n_loops": n_loops, "n_loops_unresolved": n_loops_unresolved, "n_ceiling": n_ceiling,
                  "n_not_escalated": n_not_escalated, "n_at_cap_unresolved": n_at_cap_unresolved, "n_unscored_below_cap": n_unscored_below_cap, "n_scored_at_cap": n_scored_at_cap,
                  "n_at_cap_final": n_at_cap_unresolved + n_scored_at_cap,
                  # the third state, item-wise (loops + ceiling + not escalated: the bound's interval) and the record's first definition beside it
                  "n_unresolved": n_at_cap_unresolved, "n_unresolved_ladder_only": n_loops + n_ceiling, "n_redrawn_scored_at_base": n_redrawn_scored_at_base,
                  "n_at_cap_base": len(at_cap), "n_eligible": len(eligible),
                  "acc_base": acc_b, "correct_base": correct_b, "n_scored_base": n_scored_b, "n_ceil_drawn": n_ceil_drawn, "n_scored_ceil": n_scored_r, "correct_ceil": correct_r}
        return {"final": final, "counts": counts, "ceiling_ids": ceiling_ids, "loops": loops, "eligible": eligible, "exit": exit_}

    @staticmethod
    def _item_class(it: Mapping[str, Any], loops: set, iid: str, at_cap_unscored: str) -> Optional[str]:
        """None for a scored item (its value counts, at the cap or not); the third state's class for an unscored item at the cap (``loop`` when the
        ladder gate classified it degenerate, else ``at_cap_unscored`` — ``not_escalated`` on the base store, ``ceiling`` on the ceiling store); the
        below-cap class for an item the instrument left unscored without reaching the cap (an unterminated think block before EOS)."""
        if it["scored"]:
            return None
        if it["at_cap"]:
            return "loop" if iid in loops else at_cap_unscored
        return UNSCORED_BELOW_CAP_CLASS

    @staticmethod
    def _check_gen_stats(gs: Mapping[str, Any], n: int, n_scored: int, n_at_cap: int, acc: Optional[float], uri: str, cap: int = CAP, thinking: str = "on") -> None:
        got = (LC.int_or_none(gs.get("n")), LC.int_or_none(gs.get("n_scored")), LC.int_or_none(gs.get("n_unscored")), LC.int_or_none(gs.get("n_at_cap")), LC.int_or_none(gs.get("max_tokens")))
        want = (n, n_scored, n - n_scored, n_at_cap, cap)
        gacc = LC.num_or_none(gs.get("acc"))
        acc_ok = (acc is None and (gacc is None or n_scored == 0)) or (acc is not None and gacc is not None and abs(float(gacc) - acc) < 1e-9)
        if got != want or not acc_ok:
            raise _PairRefusal(f"gen_stats.json (n {got[0]}, n_scored {got[1]}, n_unscored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}, acc {gacc}) does not reproduce the outcome "
                               f"record (n {want[0]}, n_scored {want[1]}, n_unscored {want[2]}, n_at_cap {want[3]}, cap {want[4]}, acc {acc})", uri)
        if str(gs.get("thinking") or "") != thinking:
            raise _PairRefusal(f"gen_stats.json:thinking is {gs.get('thinking')!r}, not {thinking!r}", uri)

    @staticmethod
    def _rows_time(bundle: Mapping[str, Any]) -> Optional[str]:
        created = bundle["consts"].get("created_at")
        return IC.utc_from_epoch(max(created)) if created else None

    @staticmethod
    def _three_numbers_row(three: Mapping[str, Any], selection: str, okey: str) -> Optional[Dict[str, Any]]:
        for k, row in (three.get("cells") or {}).items():
            if isinstance(row, Mapping) and str(row.get("view")) == selection and str(row.get("cell_id")) == okey:
                return dict(row, _key=k)
        return None

    @staticmethod
    def _final_state_row(fstate: Mapping[str, Any], selection: str, slug: str) -> Optional[Dict[str, Any]]:
        row = fstate.get(f"{selection}/{slug}")
        return dict(row) if isinstance(row, Mapping) else None

    @staticmethod
    def _cross_check_final_state(fs: Mapping[str, Any], ceil: Optional[Mapping[str, Any]], items: Mapping[str, Any]) -> List[str]:
        """The producer's per-pair item-wise record (g8k_final_state.py) against the re-derivation: every count it carries."""
        c = items["counts"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False) -> None:
            ok = rounded_equal(theirs, mine) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        eq("n_items", LC.int_or_none(fs.get("n_items")), N_ITEMS)
        eq("n_scored_8192", LC.int_or_none(fs.get("n_scored_8192")), c["n_scored_base"])
        eq("correct_8192", LC.int_or_none(fs.get("correct_8192")), c["correct_base"])
        eq("rung_present", bool(fs.get("rung_present")), ceil is not None)
        eq("n_rung_drawn", LC.int_or_none(fs.get("n_rung_drawn")), c["n_ceil_drawn"])
        eq("n_rung_drawn_that_were_scored_at_base", LC.int_or_none(fs.get("n_rung_drawn_that_were_scored_at_base")), c["n_redrawn_scored_at_base"])
        eq("n_scored_final", LC.int_or_none(fs.get("n_scored_final")), c["n_scored_final"])
        eq("correct_final", LC.int_or_none(fs.get("correct_final")), c["correct_final"])
        eq(HEADLINE_KEY_FINAL, LC.num_or_none(fs.get(HEADLINE_KEY_FINAL)), (c["correct_final"] / c["n_scored_final"]) if c["n_scored_final"] else None, rounded=True)
        eq("n_unresolved_final", LC.int_or_none(fs.get("n_unresolved_final")), c["n_unresolved"])
        eq("n_unscored_below_cap_final", LC.int_or_none(fs.get("n_unscored_below_cap_final")), c["n_unscored_below_cap"])
        eq("n_scored_at_cap_final", LC.int_or_none(fs.get("n_scored_at_cap_final")), c["n_scored_at_cap"])
        eq("acc_over_attempted_final", LC.num_or_none(fs.get("acc_over_attempted_final")), c["correct_final"] / N_ITEMS, rounded=True)
        if "rung_items_not_in_base" in fs:
            eq("rung_items_not_in_base", LC.int_or_none(fs.get("rung_items_not_in_base")), 0)
        return bad

    @staticmethod
    def _cross_check_three(tn: Mapping[str, Any], base: Mapping[str, Any], ceil: Optional[Mapping[str, Any]], items: Mapping[str, Any]) -> List[str]:
        c = items["counts"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False) -> None:
            ok = rounded_equal(theirs, mine) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        eq("n_attempted", LC.int_or_none(tn.get("n_attempted")), N_ITEMS)
        eq("n_scored_8192", LC.int_or_none(tn.get("n_scored_8192")), c["n_scored_base"])
        eq("correct_8192", LC.int_or_none(tn.get("correct_8192")), c["correct_base"])
        eq("acc_over_scored", LC.num_or_none(tn.get("acc_over_scored")), c["acc_base"], rounded=True)
        eq("n_at_cap_8192", LC.int_or_none(tn.get("n_at_cap_8192")), c["n_at_cap_base"])
        eq("n_loops", LC.int_or_none(tn.get("n_loops")), c["n_loops"])
        eq("n_eligible_for_rung", LC.int_or_none(tn.get("n_eligible_for_rung")), c["n_eligible"])
        eq("ladder_exit", str(tn.get("ladder_exit")), str(items["exit"]))
        r = tn.get("rung_16384")
        if ceil is None:
            eq("rung_16384", r, None)
        else:
            if not isinstance(r, Mapping):
                bad.append(f"rung_16384: producer {r!r}, a ceiling store is present")
            else:
                eq("rung_16384.n", LC.int_or_none(r.get("n")), c["n_ceil_drawn"])
                eq("rung_16384.n_scored", LC.int_or_none(r.get("n_scored")), c["n_scored_ceil"])
                eq("rung_16384.correct", LC.int_or_none(r.get("correct")), c["correct_ceil"])
                eq("rung_16384.still_at_cap", LC.int_or_none(r.get("still_at_cap")), c["n_ceiling"])
        eq("correct_after_rung", LC.int_or_none(tn.get("correct_after_rung")), c["correct_final"])
        eq("acc_over_attempted_after_rung", LC.num_or_none(tn.get("acc_over_attempted_after_rung")), c["correct_final"] / N_ITEMS, rounded=True)
        eq("acc_over_attempted_lower_bound", LC.num_or_none(tn.get("acc_over_attempted_lower_bound")), c["correct_final"] / N_ITEMS, rounded=True)
        version = record_version(tn)
        if version != RECORD_VERSION_BASE:
            # the final-state keys (09:14Z and later): the merged reading itself, and the headline named
            eq("n_scored_final", LC.int_or_none(tn.get("n_scored_final")), c["n_scored_final"])
            eq("correct_final", LC.int_or_none(tn.get("correct_final")), c["correct_final"])
            eq(HEADLINE_KEY_FINAL, LC.num_or_none(tn.get(HEADLINE_KEY_FINAL)), (c["correct_final"] / c["n_scored_final"]) if c["n_scored_final"] else None, rounded=True)
            eq("headline", str(tn.get("headline")), HEADLINE_KEY_FINAL)
            if "acc_over_scored_8192" in tn:
                eq("acc_over_scored_8192", LC.num_or_none(tn.get("acc_over_scored_8192")), c["acc_base"], rounded=True)
        if version == RECORD_VERSION_ITEMWISE:
            # the item-wise write: unresolved counts the not-escalated items too; the first definition rides beside it
            eq("n_unresolved_neither_hit_nor_miss", LC.int_or_none(tn.get("n_unresolved_neither_hit_nor_miss")), c["n_unresolved"])
            eq("n_unresolved_ladder_only", LC.int_or_none(tn.get("n_unresolved_ladder_only")), c["n_unresolved_ladder_only"])
            eq("acc_over_attempted_upper_bound", LC.num_or_none(tn.get("acc_over_attempted_upper_bound")), (c["correct_final"] + c["n_unresolved"]) / N_ITEMS, rounded=True)
            if "n_unscored_below_cap_final" in tn:
                eq("n_unscored_below_cap_final", LC.int_or_none(tn.get("n_unscored_below_cap_final")), c["n_unscored_below_cap"])
            if "n_rung_drawn" in tn:
                eq("n_rung_drawn", LC.int_or_none(tn.get("n_rung_drawn")), c["n_ceil_drawn"])
            if "n_rung_drawn_that_were_scored_at_base" in tn:
                eq("n_rung_drawn_that_were_scored_at_base", LC.int_or_none(tn.get("n_rung_drawn_that_were_scored_at_base")), c["n_redrawn_scored_at_base"])
            if "acc_over_attempted_final" in tn:
                eq("acc_over_attempted_final", LC.num_or_none(tn.get("acc_over_attempted_final")), c["correct_final"] / N_ITEMS, rounded=True)
        else:
            # the record's FIRST definition of the third state (loops + ceiling; a not-escalated item a miss in its bound) — checked as the
            # producer wrote it; the reader files the widened definition whatever the record's version
            eq("n_unresolved_neither_hit_nor_miss", LC.int_or_none(tn.get("n_unresolved_neither_hit_nor_miss")), c["n_unresolved_ladder_only"])
            eq("acc_over_attempted_upper_bound", LC.num_or_none(tn.get("acc_over_attempted_upper_bound")), (c["correct_final"] + c["n_unresolved_ladder_only"]) / N_ITEMS, rounded=True)
        if "n_scored_at_cap_final" in tn:
            eq("n_scored_at_cap_final", LC.int_or_none(tn.get("n_scored_at_cap_final")), c["n_scored_at_cap"])
        return bad

    # ---- decode --------------------------------------------------------------------------------
    @staticmethod
    def _numbers_agree(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        """The stated sampling numbers agree where BOTH state one (a number one side leaves unstated — repetition_penalty, which the store does not
        send — is no disagreement)."""
        na, nb = RP._numbers(a), RP._numbers(b)
        return all(na[k] == nb[k] for k in DP.PARAM_KEYS if na[k] is not None and nb[k] is not None) and any(na[k] is not None and nb[k] is not None for k in DP.PARAM_KEYS)

    def _match_policy(self, model_id: str, block: Mapping[str, Any], thinking_mode: str = "on") -> Tuple[Optional[DP.Policy], str]:
        pf = self._policy_file()
        if pf is None:
            return None, "the committed decode-policy export could not be loaded"
        try:
            p = pf.resolve(model_id, eval_id=EVAL_ID, thinking_mode=thinking_mode)
        except (DP.NoPolicy, KeyError, ValueError) as exc:
            return None, f"no registered decode policy row for {model_id} / {EVAL_ID} thinking {thinking_mode}: {exc}"
        if self._numbers_agree(block, p.numbers()):
            return p, "exact by the stated sampling numbers"
        return None, f"the sent numbers {RP._numbers(block)} differ from the default policy row {p.policy_id} {RP._numbers(p.numbers())}"


class _PairRefusal(Exception):
    """A (tag, object key) pair cannot be filed: carries why and where (a RowRefusals entry, never a group refusal)."""

    def __init__(self, why: str, where: str):
        super().__init__(f"{where}: {why}")
        self.why = why
        self.where = where


def factory() -> Gsm8kSeedrepAdapter:
    return Gsm8kSeedrepAdapter()


register(NAME, factory)
