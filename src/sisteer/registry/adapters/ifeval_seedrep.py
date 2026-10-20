"""The batch-64 three-seed Qwen family's IFEval stores -> registry (source ``ifeval_seedrep``).

WHAT WAS RUN (the seed-replicate IFEval run; Modal, the ``twosi-fits`` volume, at tree
``8365992042b6a4836a27593ea4364b78871132d9``; the 30 rows the launch set names, every one complete): seed tags ``qwen35_4b_s1`` / ``_s2`` /
``_s3`` x {the unsteered anchor ``none``, the five admitted staged objects (``base64s<N>__…`` / ``derived_base64s<N>__…``: constitution_relu,
control_relu, constitution_lora, control_lora, difference_lora), the two prompted system instructions (safety_si, constitution_si) and the
Arditi all-positions refusal steer at its operating point (``ar_c0p3__arditi_allpos``, ONE fit on every seed tag)}, plus the composed
``difference_relu`` paired write on the PAIRED tags ``qwen35_4b_s<N>p`` (``paired_eb64s<N>__constv2-mix-full__psr_ce_all_minus_ctl``: its own
standup — a STANDUP REPLICATE of ``s<N>``; its OWN ``none`` follows one per paired tag (since 2026-09-23), and reads through
the same completion-cap merge as every ``none``). Every row is the IFEval instrument — inspect_evals 0.18.0's stock ``ifeval`` task as row task ``inspect/ifeval``
over the 541 prompts, k = 2 seeded trials per prompt (1,082 draws), thinking OFF, the vendor-preset decode row (T 0.7 / top_p 0.8 / top_k 20 /
min_p 0 / presence_penalty 1.5; the registered row ``qwen35_4b_it|ifeval|off|vendor_preset@v1``), the BASE cap 2,048 with the cap ladder
2,048 -> 4,096 -> 8,192 (trigger: more than 1 % of the draws at the cap; the truncated non-loop draws are re-drawn at the next cap; a loop is
never re-drawn; no cap above 8,192) — served LoRA-natively (``proxy_vllm+lora+interleaved``; the unsteered and prompted keys ride the
passthrough route) on ``Qwen/Qwen3.5-4B @ 851bf6e8``. Capability evaluations carry their OWN instrument (never the statics' 4,096 cap):
every row files on ``ifeval_inspect_evals_sampled_vendor_preset_ladder`` bound to the served model, at the cap its ladder REACHED.

SEED READING: IFEval keeps its own fixed-base per-draw seed rule (``sha256(f'{20260909}|{trial}|ifeval|{item_key}')[:4]``, ``seed_base``
20260909 on every store, no per-seed replicate block): the seed axis is the fitted OBJECT, and every row — fitted, unsteered, prompted,
all-positions — files the documented default: ``decode_seed_rule per_item``, ``seed_base NULL``, ``trial_index 0``. A replicate block on
a store refuses the row.

THE MIRROR (``run-state/full22/ifeval_seedrep/``, a byte-for-byte publish of ``twosi-fits:tinycap/ifeval_seedrep/8365992``;
``PROVENANCE.md`` at the root; the mirror root IS the producer's root). What is read, every file named:

* ``controls/<tag>/<object dir>/vendor_preset/`` — the BASE store (cap 2,048) of a (tag, object key); ``…/vendor_preset__rung<cap>/`` — the
  cap-step store at 4,096 / 8,192 (present for the rows the ladder promoted: 20 / 15 of 30). The object dir is the producer's slug
  (``constv2-mix-full--lora_r4_ce``: ``--`` for the harness id's ``__``). Under each store: ``RESULT`` (its LAST line must read
  ``tinycap-modal … rc=0 result=COMPLETE status=OK sha=<tree> at=<ts>``; earlier lines are failed attempts), ``cells/<store key>/manifest.json``
  (served identity, ``producer.tree_sha``, ``harness_config_hash``, ``geometry``, ``seeds.inspect``, ``tinycap.max_tokens_by_task``,
  ``resume_attempts``, ``usage_ledger``; a step store adds ``item_subset`` — the draws it re-drew, its source store's identity hash),
  ``gen_stats.json`` / ``summary/inspect__ifeval.json`` / ``done/inspect__ifeval.json`` (the producer's aggregate: ``acc`` = the prompt-level
  strict mean, ``n`` items, ``n_requests`` draws, ``n_at_cap``, ``max_tokens``, the sent numbers, the four per-trial means),
  ``outcomes/inspect__ifeval.json`` (the per-DRAW outcome record — ``item_id`` x ``sample_idx`` — ``at_cap`` / ``scored`` / ``value`` (=
  ``prompt_level_strict``) / ``prompt_level_loose`` / ``inst_level_strict`` / ``inst_level_loose`` / ``num_instructions`` / ``stop_reason`` /
  ``completion_tokens`` / ``seed``: the draw universe), ``rows.jsonl`` (the per-draw ``acc@<trial>`` rows — the generation text, the row's
  ``created_at`` — the per-item ``acc`` rows and the aggregates attesting the object sha and the tree), ``served_geometry.json`` (the standup's
  endpoint: ``lora_config.max_lora_rank`` — RECORDED, never compared), ``cap_ladder.json`` (``cap_ladder/1``: present on a store
  the ladder PROMOTED FROM — its per-draw classification: ``eligible_draws`` the truncated non-loop draws the next cap re-draws, ``loop_draws``
  the loops, ``exit``), ``ladder_state.json`` (a step store's plan: ``ifeval_ladder_state/1``, the cap, the draw count, the source store, the
  subset's sha), ``<object key>/controls_summary.json`` (the base store's producer summary); ``evidence.json`` / ``resolved_sampler.json`` /
  ``request_evidence.jsonl`` / ``usage_ledger.jsonl`` / the logs are registered as the store's artifacts and not otherwise read;
* ``rung<cap>/<tag>/<object dir>/rung_subset.json`` (``rung_subset/2``): the (item, trial) draws re-drawn at that cap, its source cap and store;
* ``records/`` — ``dispatch/dispatch/<tag>/<object dir>/rung2048.json`` (the pass-1 dispatcher's record per base store),
  ``dispatch_rung<cap>/dispatch/<tag>/<object dir>/rung<cap>.json`` (the cap-step dispatchers' records) and
  ``dispatch_paired_none/dispatch/<tag>/none/rung2048.json`` (the paired tags' own ``none`` rows, dispatched from a sibling matrix — mirrored
  2026-09-23 15:24Z; read only for a paired tag's ``none`` and only when the pass-1 root names none: :data:`DISPATCH_ROOTS_RULE`); every
  record ``ifeval_dispatch_record/1`` (the requested vs ATTESTED serving path, the state, the object sha, the Modal function call), each dispatcher
  root's ``dispatch_state.json``; the parked ``dispatch*.failed-<n>/`` / ``dispatch*.incomplete-<n>/`` roots are the ATTEMPT-1 records of dispatches
  that were stopped, fixed and re-dispatched from a fresh root — history, NEVER read as a row's state (PROVENANCE.md names them so; discovery opens
  only the live roots by name); ``ladder_<cap>/plan.json`` + ``promoted.json`` (``ifeval_seedrep_rung_plan/1``: the ladder's DECIDING record per
  cap step — the merged reading it decided on, the rows it promoted and the rows it did not, with their exits); ``launch_sets.json``
  (``launch_sets/v1``: the ROW TABLE — ``evals.ifeval.launch``, the rows the current pass launched, plus ``carried`` / ``carried_late``, the rows
  carried from the earlier pass; :data:`_bcbw_common.ROW_TABLE_RULE`, the census of 2026-09-23 14:19Z — the PIN KEY and EXPECTED object sha
  per row, the identity declaration, the census's own anchor declaration ``anchor_tuple`` / ``anchor_fallback``); ``sdfbase_full22_pins.json``
  (the pins at the pin sha); ``matrix_qwen_seedrep.json`` (``ifeval_matrix/1``, the row matrix, cross-checked when present);
  ``three_numbers_ifeval_qwen_seedrep.json`` (``w4_three_numbers/1``: the producer's own count of every PER-STORE number this reader re-derives —
  65 rows, one per store; the MERGED reading has no producer record).

THE COMPLETION-CAP MERGE IS THE READER'S (capability rows are read at the completion cap; the GSM8K reader's
item-wise merge, here at the DRAW grain the ladder itself works at). A row's final state is the DRAW-WISE merge: for every (item, trial) draw,
the deepest cap's draw REPLACES its truncated shallower draw (``rung_subset.json`` names the draws re-drawn at each cap; each must have been at
the cap in the state it replaced, and — where the source store carries ``cap_ladder.json`` — must be exactly its ``eligible_draws``); a draw the
ladder classified a LOOP is never re-drawn and stays at its cap; a draw never at cap keeps its base reading. Everything is RECOUNTED from the
merged per-draw outcomes — never a sum of a base aggregate and a step aggregate (the additive merge double-counts the draws the step re-drew).
The merged reading is checked against the ladder plan's own merged decision at every cap step (``plan.json:planned[].decision.n_at_cap`` over the
1,082 draws, its ``exit``, its ``eligible_draws``) and the per-store readings against the producer's 65-row record; the merged value has no
producer record and is derived here alone, its derivation on the row (which draws came from which cap).

REGISTRY GRAIN (:mod:`gsm8k_seedrep`'s): ONE run per (model, tag, object key) at the ladder's FINAL STATE — ``run_id = identity.run_id(sdfbase_full22,
<tag>/<base store key>, <object key>, ifeval, <the IFEval instrument bound to qwen35_4b_it>)`` (the producer key is the tag-prefixed BASE store
NAME, never a path: :data:`PRODUCER_KEY_RULE`); ``token_cap_tokens`` the cap the ladder reached, ``token_cap_rung_index`` its index on
``[2048, 4096, 8192]``, ``extra.escalation_state`` ``final_at_base`` / ``final_at_step`` / ``final_at_ceiling``; the base and intermediate caps'
readings ride ``extra.three_numbers.per_cap`` (never a second run, never an average of two stores). ONE GROUP PER SEED (:data:`GROUP_RULE`):
the seed tag and its paired tag read in one batch, because the anchor rule pairs the composed row ACROSS tags with ``s<N>/none`` until the
paired tag's own ``none`` files.

THE THREE NUMBERS (the GSM8K precedent; degeneration is scored separately):

1. THE VALUE — ``run_metric prompt_level_strict`` (the eval's headline: ``configs/registry/evals.yaml``): the strict prompt-level passes over the
   SCORED draws of the merged final state (= the mean over items of each item's two-trial mean, every item carrying k = 2 draws), ``k`` the
   passing draws, ``n`` the scored draws, under the reader's merge computation (:data:`MERGE_ALGORITHM`, a non-blank ``computation_id``,
   ``computed_by registry``); for a row with no cap step it equals ``gen_stats.json:acc`` (checked). The loose prompt-level rate files beside it
   as a SECONDARY ``run_metric prompt_level_loose`` under the same computation; the instruction-level micro means ride ``extra``.
2. THE BOUND — a SECOND ``run_metric prompt_level_strict`` row: passes over ALL 1,082 draws (541 x k), ``producer_metric_spelling bound`` (the
   producer's own spelling per store: ``acc x n_scored / n``), ``computed_by producer``, a BLANK ``computation_id``, ``ci_lo`` / ``ci_hi`` the
   interval the unscored draws leave open (``unresolved_items_interval``; IFEval scores every draw, so the interval closes on the value). The
   frame's metric role ranks the computed row first: the value is the owner row, the bound secondary evidence.
3. THE THIRD STATE — the draws at the token cap at the FINAL state, over the 1,082 draws (``run_degeneration at_token_cap_finish_reason``, unit
   ``draws``, the finish-reason reading: ``stop_reason == max_tokens``). An at-cap draw IS scored on IFEval (the checker fails what it did not
   complete), so its value counts and ``item_flags.at_token_cap 1`` marks it; its class rides ``item_result.extra.at_cap_class``: ``loop`` (the
   ladder's rep-ratio classifier, never re-drawn), ``ceiling`` (still at cap at 8,192, the ladder's last cap), ``not_escalated`` (truncated below
   8,192, the ladder did not trigger; the plan's ``eligible_draws``). ``rep_ratio_loop`` = the classified loops over the draws (the classification
   of ``cap_ladder.json`` where the ladder promoted from a store, the plan's ``at_cap minus eligible`` where it stopped), ``truncation_only`` =
   ceiling + not_escalated; the other declared measures file ``not_measured`` naming why.

DISPATCH: a row files only if EVERY store of its chain has its dispatcher's record reading ``state done`` with
``serving_path_attested lora_native`` AND the store's own ``RESULT`` reads COMPLETE / OK and its manifest attests ``lora_native`` (four spellings);
a record whose attestation read was RATE-LIMITED (``SERVING_PATH_MISMATCH … rate limit``, ``serving_path_attested`` null) is judged by the STORE
(the one ``rung8192`` record of ``qwen35_4b_s2/constv2-mix-full--psr_ce_all``); any other failed record, or a record absent for a present store,
refuses the row by name. A record is read from the live root its cap names; a paired tag's ``none`` may carry its base-cap record under
``records/dispatch_paired_none/`` instead (:data:`DISPATCH_ROOTS_RULE`; the pass-1 root wins when both name it). The parked ``*.failed-*`` /
``*.incomplete-*`` roots are attempt-1 history and are never inputs.

ROWS OUTSIDE THE LADDER PLAN (:data:`PLAN_ABSENT_RULE`; the paired tags' own ``none`` rows of 2026-09-23): a row the ladder's deciding records never
planned files ONLY as a base-only row — no cap-step store, subset or promotion may exist for it, and its base at-cap share must not exceed the trigger
threshold the plan records themselves declare (``planned[].decision.threshold``, one value across the planned rows), so no step was due under the
producer's own rule. Nothing is reconstructed: its at-cap draws are ``unclassified`` (no rep-ratio reading exists, so ``rep_ratio_loop`` and
``truncation_only`` file ``not_measured``), and ``extra.completion_cap_merge.plan_decisions`` names the absence (exit ``not_planned``, ``planned``
false). A share above the threshold, or any step artifact, refuses the row: the ladder is undecided there.

OBJECT IDENTITY per (selection, object key) = the launch set's pin key, which must be the harness roster's staged id for the key at the tag's
selection (``configs/sdfbase_full22_roster.yaml`` -> ``configs/sdfbase_full22_pins.json``; the all-positions key through
``configs/arditi_allpos_roster.yaml`` / ``_pins.json`` at the operating point the pin key spells), whose pinned sha must equal the launch set's
expected sha, the mirrored pins' entry, every dispatcher record's ``object_sha256``, and every store's ``route.object_artifact_sha256`` /
``provenance.artifact_sha256`` / the rows' aggregate / the 16-hex ``fit_provenance.object_sha``; a prompted key resolves through ``si_texts.yaml``
and ``resolved_config.system_sha``. Any disagreement refuses the (tag, object key) by name (:class:`lineage_common.RowRefusals`; the group's other
rows file). The reader never consults the live registry for the object: the run is emitted on the attested identity and ``blocked_records`` holds
it while the object row is absent.

ANCHORS: every steered and prompted row of a seed tag files a ``run_contrast`` against ITS tag's ``none`` at the SAME final state (the
``none`` rows never escalated, so their base reading is their final), paired per DRAW (item, trial — the same seeds on both sides under the fixed
seed rule) over the draws scored on both sides: the strict pass-rate difference in pp, McNemar's exact p on the discordant pairs, Cohen's h; the
composed ``s<N>p`` row pairs against ITS OWN tag's ``none`` once that files (since 2026-09-23: the paired tag is a standup
replicate of ``s<N>``, its ``none`` launch-set class ``anchor``, a normal run through the same merge, ``trial_index 0``; ``anchor_view_legacy`` the
paired tag, ``cross_standup`` false) and ACROSS tags with ``s<N>/none`` until then (``anchor_kind sibling_none`` — the same model's ``none`` served
by another standup, :func:`_bcbw_common.anchor_kind_for`; ``extra.cross_standup``; the fallback,
:data:`PAIRED_ANCHOR_RULE` / :func:`_bcbw_common.anchor_plan`; the run's ``extra.anchor.own_none_state`` names the pending ``none`` once the launch
set declares it; a seed tag's rows never pair with the paired tag's ``none``); the Arditi rows are declared UNPAIRED (role
``arditi_unpaired_value``, the GSM8K / MATH-500 / BCB precedent) — the value files regardless, the contrast is formed as for every steered key and
the role rides on the run and the contrast. The unsteered, prompted and all-positions keys name ONE object on every seed tag; their coexistence in
the paper frame needs the roster's ``coexist`` ruling, never a reader default.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .. import decode_policy as DP
from .. import identity as I
from .. import jobplan as JP
from .. import vocab as V
from . import _bcbw_common as B
from . import _items_common as IC
from . import anchor_current as AC
from . import declared_common as DC
from . import dose_grammar as DG
from . import gsm8k_seedrep as G
from . import lineage_common as LC
from . import relu_paired as RP
from . import seedrep_cluster as SC
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register
from .runner import source_id_for

NAME = "ifeval_seedrep"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "ifeval_seedrep"
OWNER = "Overflow"
PRODUCER_SESSION = "Overflow"
CAMPAIGN = LC.CAMPAIGN
ROOT_DEFAULT = os.path.join(LC.RS_DEFAULT, "ifeval_seedrep")
MODEL_KEY = SC.MODEL_KEY
TAG_RE = re.compile(rf"^(?P<model>{MODEL_KEY})_s(?P<seed>\d+)(?P<paired>p?)$")
EVAL_ID = "ifeval"
TASK = "inspect/ifeval"
OUTCOMES_FILE = "inspect__ifeval.json"
#: the IFEval instrument (configs/registry/instruments.yaml), bound per model by the dimensions loader
INSTRUMENT_SEED = "ifeval_inspect_evals_sampled_vendor_preset_ladder"
LADDER: Tuple[int, ...] = (2048, 4096, 8192)
CAP_BASE, CAP_CEILING = LADDER[0], LADDER[-1]
N_ITEMS = 541
K_TRIALS = 2
N_DRAWS = N_ITEMS * K_TRIALS
#: the fixed per-draw seed base every store of this instrument spells (contracts IFEVAL_SEED_BASE): the instrument's, never a per-seed base
IFEVAL_SEED_BASE = 20260909
STORES_HEAD = "controls"
BASE_DIR = "vendor_preset"
STEP_DIR_FMT = "vendor_preset__rung{cap}"
SUBSET_HEAD_FMT = "rung{cap}"
SUBSET_FILE = "rung_subset.json"
RECORDS_DIR = "records"
DISPATCH_HEAD_BASE = "dispatch"
DISPATCH_HEAD_FMT = "dispatch_rung{cap}"
DISPATCH_SUB = "dispatch"
DISPATCH_FILE_FMT = "rung{cap}.json"
DISPATCH_STATE_FILE = "dispatch_state.json"
#: the paired tags' own `none` rows were dispatched from a sibling matrix (mirrored 2026-09-23 15:24Z): their base-cap records sit under THIS
#: root (`records/dispatch_paired_none/dispatch/<tag>/none/rung2048.json`, its own dispatch_state.json beside them); the pass-1 root stays authoritative
#: for every other row, and for a paired `none` it names too (DISPATCH_ROOTS_RULE)
DISPATCH_HEAD_PAIRED_NONE = "dispatch_paired_none"
#: the parked dispatcher roots (`dispatch*.failed-<n>` / `dispatch*.incomplete-<n>`: attempt history of a dispatch that was stopped, fixed and re-dispatched
#: from a fresh root; PROVENANCE.md "Attempt history present under records/ — NOT inputs"): named on the run, NEVER read — discovery opens only the live
#: roots by name, so a parked root is never an input whatever it holds
FAILED_DISPATCH_RE = re.compile(r"^dispatch(?:_[a-z0-9_]+)?\.(?:failed|incomplete)-[A-Za-z0-9_]+$")
HISTORY_DISPATCH_RE = FAILED_DISPATCH_RE
PLAN_HEAD_FMT = "ladder_{cap}"
PLAN_FILE = "plan.json"
PROMOTED_FILE = "promoted.json"
LAUNCH_SET_FILE = "launch_sets.json"
PINS_FILE = "sdfbase_full22_pins.json"
MATRIX_FILE = "matrix_qwen_seedrep.json"
THREE_NUMBERS_FILE = "three_numbers_ifeval_qwen_seedrep.json"
SHARED_FILES = (LAUNCH_SET_FILE, PINS_FILE, MATRIX_FILE, THREE_NUMBERS_FILE)
CONTROLS_SUMMARY_FILE = "controls_summary.json"
GEOMETRY_FILE = "served_geometry.json"
CAP_LADDER_FILE = "cap_ladder.json"
LADDER_STATE_FILE = "ladder_state.json"
STORE_TOP_FILES = ("RESULT", CAP_LADDER_FILE, LADDER_STATE_FILE, GEOMETRY_FILE, "evidence.json", "resolved_sampler.json", "request_evidence.jsonl",
                   "usage_ledger.jsonl", "ledger.log", "driver_stdout.log", "driver_stderr.log", "oos_prompt_budget.json")
STORE_KEY_FILES = ("gen_stats.json", "manifest.json", "rows.jsonl", "evidence.json")
TASK_SUBDIRS = ("outcomes", "summary", "done")
ROLE_OF_FILE = {"rows.jsonl": "rows", "gen_stats.json": "completion", "RESULT": "log", "usage_ledger.jsonl": "log", "ledger.log": "log",
                "request_evidence.jsonl": "log", "driver_stdout.log": "log", "driver_stderr.log": "log"}
LAUNCH_SET_SCHEMA = "launch_sets/v1"
THREE_NUMBERS_SCHEMA = "w4_three_numbers/1"
CAP_LADDER_SCHEMA = "cap_ladder/1"
LADDER_STATE_SCHEMA = "ifeval_ladder_state/1"
SUBSET_SCHEMA = "rung_subset/2"
DISPATCH_SCHEMA = "ifeval_dispatch_record/1"
DISPATCH_STATE_SCHEMA = "ifeval_dispatch_state/1"
PLAN_SCHEMA = "ifeval_seedrep_rung_plan/1"
MATRIX_SCHEMA = "ifeval_matrix/1"
RESULT_COMPLETE = B.RESULT_COMPLETE
RESULT_HEAD = B.RESULT_HEAD
SERVING_ENGINE = "lora_native"
STEERED_IDENTITY = SC.STEERED_IDENTITY
PASSTHROUGH_IDENTITY = SC.PASSTHROUGH_IDENTITY
STEERED_PATH_LEGACY = "lora"
PASSTHROUGH_PATH_LEGACY = "passthrough"
#: the volume path every store's manifest places its usage ledger under (PROVENANCE.md: held equal at publish); the substrate witness
VOLUME_HEAD = "/vol/tinycap/ifeval_seedrep/"
SUBSTRATE = "modal"
SAMPLING_MODE = "vendor_preset"
DISPATCH_DONE = "done"
DISPATCH_FAILED = "failed"
DISPATCH_OK = "OK"
#: a dispatcher record whose attestation READ failed on the volume's list rate limit: the store decides (module docstring, DISPATCH)
DISPATCH_RATE_LIMITED_STATUS = "SERVING_PATH_MISMATCH"
DISPATCH_RATE_LIMITED_MARK = "rate limit"
#: the ladder's exits (cap_ladder.json / plan.json): a step store exists iff the deciding record says `escalate`
EXIT_ESCALATE = "escalate"
EXITS_STOP = ("below_threshold", "triggered_no_eligible", "clean")
#: the headline (evals.yaml ifeval.headline_metric_id) and the secondary rate filed beside it; the store's rows.jsonl spells the per-draw
#: strict value `acc@<trial>` and the per-item mean `acc` (the producer's row metric, never the registry's id)
HEADLINE_METRIC = "prompt_level_strict"
HEADLINE_ROW_METRIC = "acc"
SECONDARY_METRIC = "prompt_level_loose"
BOUND_SPELLING = "bound"
BOUND_CI_METHOD = G.BOUND_CI_METHOD
ITEM_LABEL_AT_CAP_UNRESOLVED = IC.ITEM_LABEL_AT_CAP_UNRESOLVED
ITEM_LABEL_UNSCORED = G.ITEM_LABEL_UNSCORED
UNSCORED_BELOW_CAP_CLASS = "unscored_below_cap"
#: `unclassified`: an at-cap draw of a row the ladder never planned (PLAN_ABSENT_RULE) — no rep-ratio classification exists for it
AT_CAP_CLASSES = ("loop", "ceiling", "not_escalated", "unclassified")
RAW_AT_CAP_MEASURE = G.RAW_AT_CAP_MEASURE
LOOP_MEASURE = G.LOOP_MEASURE
COHERENT_MEASURE = G.COHERENT_MEASURE
ESCALATION_STATES = ("final_at_base", "final_at_step", "final_at_ceiling")
#: the headline's computation: the draw-wise completion-cap merge and recount (no rng)
MERGE_ALGORITHM = "sisteer.registry.adapters.ifeval_seedrep.completion_cap_merge"
MERGE_VERSION = "1"
MERGE_SEED_PROVENANCE = "none: a deterministic draw-wise merge of the base store's outcome record with each cap step's re-drawn draws, and a recount over the 1,082 draws"
DECODE_SEED_RULE = "per_item"
INSPECT_SEED_POLICY_DEFAULT = "default"
CONTRAST_METHOD = G.CONTRAST_METHOD
ROUNDED_TOL = G.ROUNDED_TOL
ROSTER_REL = SC.ROSTER_REL
PINS_REL = SC.PINS_REL
ARDITI_ROSTER_REL = SC.SINGLE_OBJECT_ROSTER_REL
ARDITI_PINS_REL = SC.SINGLE_OBJECT_PINS_REL
ARDITI_OBJECT_KEY = DG.OBJECT_KEY_LEGACY
ARDITI_ROLE = G.ARDITI_ROLE
LAUNCH_CLASSES = ("anchor", "arm", "prompted", "composed", "arditi")
#: the rules every row cites
PRODUCER_KEY_RULE = "the producer key is `<tag>/<base store key>`: the tag prefixes the BASE store's own NAME (never its path), so a paired tag's store never shares a key with a seed tag's"
GROUP_RULE = ("one group per SEED: the seed tag `qwen35_4b_s<N>` and its paired tag `qwen35_4b_s<N>p` read in one batch — the anchor rule (D65) pairs the "
              "composed row across tags with `s<N>/none`, and a contrast names a run its own batch emits")
ANCHOR_RULE = ("D65: every steered and prompted row pairs with ITS seed tag's `none` at the SAME final state, per draw (item, trial); the composed `s<N>p` row pairs "
               "ACROSS tags with `s<N>/none`; the all-positions rows are declared unpaired (arditi_unpaired_value) — the value files regardless and the contrast carries the role")
SEED_AXIS_RULE = ("D69: no per-seed decode rule on this evaluation — IFEval keeps its own fixed-base per-draw seed rule (seed_base 20260909 on every store, no replicate "
                  "block), so the seed axis is the fitted OBJECT; every row files decode_seed_rule per_item, seed_base NULL, trial_index 0 (D66)")
MERGE_RULE = ("the completion-cap merge is the reader's: for every (item, trial) draw the deepest cap's draw REPLACES its truncated shallower draw (rung_subset.json names "
              "the draws re-drawn at each cap; each was at the cap in the state it replaced), a loop is never re-drawn and stays at its cap, a draw never at cap keeps "
              "its base reading; every number is recounted over the merged 1,082 draws — never a base aggregate plus a step aggregate")
DISPATCH_RULE = ("a row files only if every store of its chain has its dispatcher's record reading state done with serving_path_attested lora_native AND the store's RESULT "
                 "reads COMPLETE / OK and its manifest attests lora_native; a record whose attestation read was rate-limited is judged by the store; any other failed "
                 "record, or a record absent for a present store, refuses the row; the `.failed-D<n>` roots are attempt-1 history and never inputs")
NOT_RUN_RULE = "a launch-set row with no base store files NOTHING here — no run, no refusal, no disposition (the producer files those); a later mirror adds the store idempotently"
DISPATCH_ROOTS_RULE = ("a store's dispatcher record is read from the live root its cap names (records/dispatch/ for the base cap, records/dispatch_rung<cap>/ for a cap step); "
                       "a paired tag's `none` (a standup replicate dispatched from a sibling matrix, D243) may carry its base-cap record under records/dispatch_paired_none/ "
                       "instead — read only for a paired tag's `none`, and only when the pass-1 root names none (the pass-1 root wins when both do); each root's own "
                       "dispatch_state.json is the one consulted; the parked *.failed-* / *.incomplete-* roots are attempt history, never inputs")
PLAN_ABSENT_RULE = ("a row the ladder's deciding records never planned (no plan.json entry at any cap step) files ONLY as a base-only row: no cap-step store, subset or "
                    "promotion may exist for it, and its base at-cap share must not exceed the trigger threshold the plan records themselves declare (the same rule the "
                    "planned rows were judged by), so no step was due; its at-cap draws are unclassified (no rep-ratio reading exists: rep_ratio_loop and truncation_only "
                    "file not_measured, at_cap_class `unclassified`) and extra.completion_cap_merge.plan_decisions names the absence (exit not_planned, planned false); a "
                    "share above the threshold, or any step artifact, refuses the row — the ladder is undecided there")
PLAN_EXIT_NOT_PLANNED = "not_planned"
ROW_TABLE_RULE = B.ROW_TABLE_RULE
ROW_TABLE_LISTS = B.ROW_TABLE_LISTS
MAX_LORA_RANK_RULE = "RECORDED, never compared: served_geometry.json endpoint.lora_config.max_lora_rank per store (D71 / GSM8K C4)"
ANCHOR_NOT_YET_RUN = B.ANCHOR_NOT_YET_RUN
ANCHOR_REFUSED = B.ANCHOR_REFUSED
ANCHOR_CONTRASTED = B.ANCHOR_CONTRASTED
#: every contrast is decided by the pass's anchor currency (the six capability readers' shared rule)
CURRENCY_RULE = B.CURRENCY_RULE
#: the paired tag's OWN `none` is its anchor once it files (the rule relayed 2026-09-23): the shared rule and resolution (_bcbw_common)
PAIRED_ANCHOR_RULE = B.PAIRED_ANCHOR_RULE
anchor_plan = B.anchor_plan
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA16 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PairRefusal = G._PairRefusal
rounded_equal = G.rounded_equal
slug_to_object_key = G.slug_to_object_key
parse_result_line = B.parse_result_line
tag_parts = B.tag_parts
seed_tag_of = B.seed_tag_of
selection_key = B.selection_key

DrawKey = Tuple[str, int]


def step_dir(cap: int) -> str:
    """The store directory of a cap: the base store's name at the base cap, ``vendor_preset__rung<cap>`` above it."""
    return BASE_DIR if cap == CAP_BASE else STEP_DIR_FMT.format(cap=cap)


def cap_of_store_dir(name: str) -> Optional[int]:
    if name == BASE_DIR:
        return CAP_BASE
    for cap in LADDER[1:]:
        if name == step_dir(cap):
            return cap
    return None


def subset_head(cap: int) -> str:
    return SUBSET_HEAD_FMT.format(cap=cap)


def dispatch_head(cap: int) -> str:
    return DISPATCH_HEAD_BASE if cap == CAP_BASE else DISPATCH_HEAD_FMT.format(cap=cap)


#: every live dispatcher root a record or a dispatch_state.json is read from (DISPATCH_ROOTS_RULE)
LIVE_DISPATCH_HEADS: Tuple[str, ...] = tuple(dispatch_head(c) for c in LADDER) + (DISPATCH_HEAD_PAIRED_NONE,)


def dispatch_heads(cap: int, tag: str, okey: str) -> List[str]:
    """The live roots a store's dispatcher record may sit under, the pass-1 root first (DISPATCH_ROOTS_RULE): the cap's root for every row; for a
    paired tag's ``none`` at the base cap, ``records/dispatch_paired_none/`` as well."""
    heads = [dispatch_head(cap)]
    parts = tag_parts(tag)
    if cap == CAP_BASE and parts is not None and parts[1] and okey == "none":
        heads.append(DISPATCH_HEAD_PAIRED_NONE)
    return heads


def dispatch_file(cap: int) -> str:
    return DISPATCH_FILE_FMT.format(cap=cap)


def plan_head(cap: int) -> str:
    return PLAN_HEAD_FMT.format(cap=cap)


def configuration_id(tag: str, slug: str, cap: int) -> str:
    """The dispatchers' and the plan's row spelling: ``<tag>/<object dir>/rung<cap>``."""
    return f"{tag}/{slug}/rung{cap}"


def rel_store(tag: str, slug: str, cap: int) -> str:
    return f"{STORES_HEAD}/{tag}/{slug}/{step_dir(cap)}"


def rel_subset(tag: str, slug: str, cap: int) -> str:
    return f"{subset_head(cap)}/{tag}/{slug}/{SUBSET_FILE}"


def producer_key_for(tag: str, base_store_key: str) -> str:
    return f"{tag}/{base_store_key}"


def run_id_for(tag: str, base_store_key: str, okey: str, instrument_id: str) -> str:
    return I.run_id(CAMPAIGN, producer_key_for(tag, base_store_key), okey, EVAL_ID, instrument_id)


def merge_computation_id() -> str:
    return I.computation_id(MERGE_ALGORITHM, MERGE_VERSION, MERGE_SEED_PROVENANCE)


def draw_key(item_id: Any, sample_idx: Any) -> DrawKey:
    return (str(item_id), int(sample_idx))


def draw_keys(pairs: Any) -> Set[DrawKey]:
    """A record's ``[[item_id, trial], …]`` list as a set of draw keys."""
    out: Set[DrawKey] = set()
    for p in (pairs or []):
        if isinstance(p, (list, tuple)) and len(p) == 2:
            out.add(draw_key(p[0], p[1]))
    return out


def completion_cap_merge(base: Mapping[DrawKey, Mapping[str, Any]], steps: Sequence[Tuple[int, Set[DrawKey], Mapping[DrawKey, Mapping[str, Any]]]]
                         ) -> Tuple[Dict[DrawKey, Dict[str, Any]], Dict[int, List[DrawKey]]]:
    """THE MERGE (:data:`MERGE_RULE`), pure: ``base`` is the base store's draws (``at_cap`` on each), ``steps`` the cap steps in ladder order — ``(cap, the
    draws re-drawn at that cap, that cap's draws)``. Each re-drawn draw must exist in the state it replaces and have been at the cap there; the step's
    draws must be exactly the re-drawn set. Returns the final draws (each carrying ``source_cap``, the replaced value and whether it replaced an at-cap
    draw) and, per cap, the draws it contributed. Raises ``ValueError`` naming the first draw that breaks the rule."""
    final: Dict[DrawKey, Dict[str, Any]] = {k: dict(v, source_cap=CAP_BASE, replaced_value=None, replaced_at_cap=None) for k, v in base.items()}
    from_cap: Dict[int, List[DrawKey]] = {CAP_BASE: sorted(final)}
    prev_cap = CAP_BASE
    for cap, redrawn, draws in steps:
        if cap <= prev_cap:
            raise ValueError(f"cap step {cap} does not climb above {prev_cap}")
        if set(draws) != set(redrawn):
            missing = sorted(set(redrawn) - set(draws))[:3]
            extra = sorted(set(draws) - set(redrawn))[:3]
            raise ValueError(f"the {cap} store's draws are not the re-drawn set (missing {missing}, extra {extra})")
        for k in sorted(redrawn):
            if k not in final:
                raise ValueError(f"draw {k} re-drawn at {cap} is not a draw of the state it replaces")
            if not final[k].get("at_cap"):
                raise ValueError(f"draw {k} re-drawn at {cap} was not at the cap in the state it replaces (a terminated draw is never re-drawn)")
        for k in sorted(redrawn):
            old = final[k]
            final[k] = dict(draws[k], source_cap=cap, replaced_value=old.get("value"), replaced_at_cap=bool(old.get("at_cap")))
        from_cap[cap] = sorted(redrawn)
        for c in list(from_cap):
            if c != cap:
                from_cap[c] = [k for k in from_cap[c] if k not in redrawn]
        prev_cap = cap
    return final, from_cap


def store_reading(draws: Mapping[DrawKey, Mapping[str, Any]]) -> Dict[str, Any]:
    """The producer's per-STORE aggregate over a store's own draws (the ``three_numbers`` contract): ``acc`` = the mean over the store's items of
    each item's mean strict value over the store's draws of it (the base store carries every item twice, a step store one or two draws of the items it
    re-drew), ``n`` the items, ``draws`` the draws, ``n_at_cap`` the at-cap draws; the loose / instruction-level per-trial means beside it."""
    by_item: Dict[str, List[float]] = {}
    for (iid, _t), d in draws.items():
        if d.get("value") is not None:
            by_item.setdefault(iid, []).append(float(d["value"]))
    n_items = len({k[0] for k in draws})
    n_scored_items = len(by_item)
    acc = (sum(sum(v) / len(v) for v in by_item.values()) / n_scored_items) if n_scored_items else None
    n_at_cap = sum(1 for d in draws.values() if d.get("at_cap"))
    # one slot per trial of the instrument (k = 2), None for a trial the store re-drew no draw of: the producer's spelling of a step store's per-trial means
    trials = list(range(K_TRIALS))

    def by_trial(field: str, denom: Optional[str] = None) -> List[Optional[float]]:
        """The per-trial mean of a draw field: the plain mean over the trial's scored draws, or — with ``denom`` — the MICRO ratio (the field summed over the
        trial's draws / the denominator summed), the instruction-level convention."""
        out: List[Optional[float]] = []
        for t in trials:
            ds = [d for (i, tt), d in draws.items() if tt == t and d.get(field) is not None]
            if not ds:
                out.append(None)
            elif denom is None:
                out.append(sum(float(d[field]) for d in ds) / len(ds))
            else:
                dn = sum(float(d.get(denom) or 0) for d in ds)
                out.append((sum(float(d[field]) for d in ds) / dn) if dn else None)
        return out
    return {"n_items": n_items, "n_scored_items": n_scored_items, "n_draws": len(draws), "n_scored_draws": sum(1 for d in draws.values() if d.get("value") is not None),
            "acc": acc, "n_at_cap": n_at_cap, "share_at_cap": (n_at_cap / len(draws)) if draws else None, "n_unscored_items": n_items - n_scored_items,
            "prompt_strict_mean_by_trial": by_trial("value"), "prompt_loose_mean_by_trial": by_trial("prompt_level_loose"),
            "inst_strict_mean_by_trial": by_trial("inst_level_strict", "num_instructions"), "inst_loose_mean_by_trial": by_trial("inst_level_loose", "num_instructions")}


#: the per-trial means a store's records spell beside `acc` (gen_stats.json / the summary / the producer's three-number row), checked against the recount
TRIAL_MEAN_FIELDS = ("prompt_strict_mean_by_trial", "prompt_loose_mean_by_trial", "inst_strict_mean_by_trial", "inst_loose_mean_by_trial")


def trial_means_agree(theirs: Any, mine: Sequence[Optional[float]]) -> bool:
    return isinstance(theirs, list) and len(theirs) == len(mine) and all(rounded_equal(LC.num_or_none(a), b, 1e-9) for a, b in zip(theirs, mine))


def merged_counts(final: Mapping[DrawKey, Mapping[str, Any]]) -> Dict[str, Any]:
    """The recount over the merged draws: the strict and loose passes over the scored draws, the instruction-level micro means, the at-cap draws."""
    scored = [d for d in final.values() if d.get("value") is not None]
    n_inst = sum(int(d.get("num_instructions") or 0) for d in scored)
    items_scored = {k[0] for k, d in final.items() if d.get("value") is not None}
    return {"n_draws": len(final), "n_scored_draws": len(scored), "n_unscored_draws": len(final) - len(scored), "strict_passes": int(round(sum(float(d["value"]) for d in scored))),
            "loose_passes": int(round(sum(float(d.get("prompt_level_loose") or 0) for d in scored))), "n_instructions": n_inst,
            "inst_strict_followed": int(round(sum(float(d.get("inst_level_strict") or 0) for d in scored))), "inst_loose_followed": int(round(sum(float(d.get("inst_level_loose") or 0) for d in scored))),
            "n_at_cap_draws": sum(1 for d in final.values() if d.get("at_cap")), "n_items": len({k[0] for k in final}), "n_items_scored": len(items_scored),
            "n_at_cap_unscored": sum(1 for d in final.values() if d.get("at_cap") and d.get("value") is None),
            "n_unscored_below_cap": sum(1 for d in final.values() if not d.get("at_cap") and d.get("value") is None)}


class IfevalSeedrepAdapter:
    name = NAME
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-23): the first version — the seed-replicate IFEval stores from the Modal volume, read at the completion cap (module docstring)
    #: 2 (2026-09-23, the mirror of 15:24Z): the launch set's ROW TABLE is launch + carried (ROW_TABLE_RULE; emission 1 read `launch` alone and would have
    #: refused every carried row); a paired tag's `none` reads its dispatcher record from records/dispatch_paired_none/ (DISPATCH_ROOTS_RULE) and files
    #: outside the ladder plan under PLAN_ABSENT_RULE (its at-cap draws unclassified, no loop reading invented); every row's extra.dispatch[<cap>] names
    #: the root it was read from; the paired tag's own `none` is that tag's anchor (the same day's rule); the parked *.incomplete-* roots join the
    #: *.failed-* roots as history
    #: (2026-09-26, no bump: the composed row's fallback contrast -- ACROSS tags against s<N>/none, extra.cross_standup true -- now files anchor_kind
    #: sibling_none through _bcbw_common.anchor_kind_for instead of same_standup_none; the live mirror re-reads byte for byte because every composed row's
    #: own `none` has filed and no contrast crosses standups any more (the dry run over the live mirror))
    #: 3 (2026-10-04): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`; the row's own pairing stands while the frame holds it current, a retired or absent own
    #: `none` re-anchors on a current unsteered run of the exact key -- paired per draw when it is a row of the batch, else the registry's unpaired
    #: rate difference from the two readings' k / n); a comparison with no current anchor parks on the VALUE row's extra.delta_block
    EMISSION_VERSION = 3
    rng_scope = "none"
    allow_symlinks = False
    owner = OWNER
    THINKING_SPELLING = "off"

    def __init__(self, root: str = ROOT_DEFAULT, seeds: Optional[LC.Seeds] = None, roster_path: Optional[str] = None, pins_path: Optional[str] = None,
                 policy_path: Optional[str] = None, arditi_roster_path: Optional[str] = None, arditi_pins_path: Optional[str] = None):
        self.root = LC.expand(root)
        self.roots = (self.root,)
        self._seeds = seeds
        self.roster_path = roster_path or DC.repo_path(*ROSTER_REL)
        self.pins_path = pins_path or DC.repo_path(*PINS_REL)
        self.arditi_roster_path = arditi_roster_path or DC.repo_path(*ARDITI_ROSTER_REL)
        self.arditi_pins_path = arditi_pins_path or DC.repo_path(*ARDITI_PINS_REL)
        self.policy_path = policy_path
        self.refusals = LC.RowRefusals(NAME)
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._reanchor_computations: Dict[str, Dict[str, Any]] = {}
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
    def _stores_root(self) -> str:
        return os.path.join(self.root, STORES_HEAD)

    def _records_dir(self) -> str:
        return os.path.join(self.root, RECORDS_DIR)

    def discover(self) -> List[Input]:
        """Named files of every ``controls/<tag>/<object dir>/<store dir>/`` store under the mirror root, the ``rung<cap>/<tag>/<object dir>/rung_subset.json``
        beside each cap step, the dispatcher records per row (``records/<live root>/dispatch/<tag>/<object dir>/rung<cap>.json`` for the roots
        :func:`dispatch_heads` names) and, aliased per group (``#<seed tag>``), the shared records (the launch set, the pins, the matrix, the three-number
        record, each live dispatcher root's state, each cap step's plan and promoted list). The parked ``*.failed-*`` / ``*.incomplete-*`` roots are never
        listed: only the live roots are opened, by name. An absent root is a SKIP (the mirror is machine-local); a root holding no store refuses."""
        out: List[Input] = []
        if not os.path.isdir(self.root):
            return out
        groups: Set[str] = set()
        for tag in IC.listdir(self._stores_root()):
            group = seed_tag_of(tag)
            if group is None or not os.path.isdir(os.path.join(self._stores_root(), tag)):
                continue
            for slug in IC.listdir(os.path.join(self._stores_root(), tag)):
                okey = slug_to_object_key(slug)
                for cap in LADDER:
                    d = os.path.join(self._stores_root(), tag, slug, step_dir(cap))
                    if not os.path.isdir(d):
                        continue
                    n_before = len(out)
                    out += self._discover_store(d, okey, cap == CAP_BASE)
                    if len(out) > n_before:
                        groups.add(group)
                        p = os.path.join(self.root, rel_subset(tag, slug, cap))
                        if cap != CAP_BASE and os.path.isfile(p):
                            out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
                        for head in dispatch_heads(cap, tag, okey):
                            p = os.path.join(self._records_dir(), head, DISPATCH_SUB, tag, slug, dispatch_file(cap))
                            if os.path.isfile(p):
                                out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        if not out:
            raise AdapterRefusal(self.root, f"no IFEval seed-replicate store found ({STORES_HEAD}/{MODEL_KEY}_s<N>[p]/<object dir>/{BASE_DIR}[__rung<cap>]/cells/<store key>/ under the root)")
        shared: List[str] = [os.path.join(self._records_dir(), n) for n in SHARED_FILES]
        for head in LIVE_DISPATCH_HEADS:
            shared.append(os.path.join(self._records_dir(), head, DISPATCH_STATE_FILE))
        for cap in LADDER[1:]:
            shared += [os.path.join(self._records_dir(), plan_head(cap), PLAN_FILE), os.path.join(self._records_dir(), plan_head(cap), PROMOTED_FILE)]
        for p in shared:
            if os.path.isfile(p):
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(g) for g in sorted(groups)]
        return out

    def _discover_store(self, d: str, okey: str, is_base: bool) -> List[Input]:
        out: List[Input] = []
        for name in STORE_TOP_FILES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append(check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks))
        if is_base:
            p = os.path.join(d, okey, CONTROLS_SUMMARY_FILE)
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
            for sub in TASK_SUBDIRS:
                p = os.path.join(kd, sub, OUTCOMES_FILE)
                if os.path.isfile(p):
                    out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def comparison_group(self, inp: Input) -> str:
        """One group per SEED (:data:`GROUP_RULE`): a store or subset under ``<head>/<tag>/`` maps to the tag's seed tag, a dispatcher record through the
        tag its path spells, a shared record through its ``#<seed tag>`` alias."""
        base = os.path.basename(inp.uri)
        if "#" in base and seed_tag_of(base.rsplit("#", 1)[1]) == base.rsplit("#", 1)[1]:
            return base.rsplit("#", 1)[1]
        rel = os.path.relpath(inp.uri, self.root).split(os.sep)
        if len(rel) > 2 and (rel[0] == STORES_HEAD or rel[0].startswith("rung")) and seed_tag_of(rel[1]) is not None:
            return seed_tag_of(rel[1])
        if len(rel) > 4 and rel[0] == RECORDS_DIR and rel[1].startswith(DISPATCH_HEAD_BASE) and rel[2] == DISPATCH_SUB and seed_tag_of(rel[3]) is not None:
            return seed_tag_of(rel[3])
        raise AdapterRefusal(inp.uri, f"input names no seed tag ({STORES_HEAD}/{MODEL_KEY}_s<N>[p]/, rung<cap>/{MODEL_KEY}_s<N>[p]/ or {RECORDS_DIR}/dispatch*/dispatch/{MODEL_KEY}_s<N>[p]/) under {self.root}")

    def _group_selections(self, group: str) -> Tuple[str, str]:
        seed_n = tag_parts(group)[0]
        return selection_key(seed_n), selection_key(seed_n, True)

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" not in base:
                continue
            name = base.rsplit("#", 1)[0]
            group = self.comparison_group(inp)
            sels = self._group_selections(group)
            tags = tuple(f"{MODEL_KEY}_{s}" for s in sels)
            if name == THREE_NUMBERS_FILE:
                d = self._json(inp) or {}
                members = sorted(k for k in (d.get("rows") or {}) if str(k).split("/", 1)[0] in tags)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.rows", 1, 1, members, inp.sha256))
            elif name == LAUNCH_SET_FILE:
                doc = self._json(inp)
                launch = self._launch_rows(doc)
                members = sorted(str(r.get("tuple_id")) for r in (launch or []) if str(r.get("view")) in sels)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, B.BcbwSeedrepReader.row_table_json_path(doc, EVAL_ID), 1, 1, members, inp.sha256))
            elif name == PROMOTED_FILE:
                d = self._json(inp) or {}
                members = sorted(str(r.get("configuration_id")) for r in (d.get("promoted") or []) if isinstance(r, Mapping) and str(r.get("configuration_id") or "").split("/", 1)[0] in tags)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.promoted", 1, 1, members, inp.sha256))
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

    _load_pair = staticmethod(B.BcbwSeedrepReader._load_pair)

    def _roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._roster is None or self._pins is None:
            self._roster, self._pins = self._load_pair(self.roster_path, self.pins_path, "seed-replicate")
        return self._roster, self._pins

    def _arditi_roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self._arditi_roster is None or self._arditi_pins is None:
            self._arditi_roster, self._arditi_pins = self._load_pair(self.arditi_roster_path, self.arditi_pins_path, "all-positions")
        return self._arditi_roster, self._arditi_pins

    def _policy_file(self) -> Optional[DP.PolicyFile]:
        if self._policy is None:
            try:
                self._policy = DP.load(self.policy_path)
            except (OSError, ValueError, KeyError, DP.NoPolicy):
                self._policy = None
        return self._policy

    @staticmethod
    def _launch_rows(doc: Optional[Mapping[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """The eval's ROW TABLE (:data:`ROW_TABLE_RULE`): ``launch`` plus ``carried`` / ``carried_late`` (the shared reading in ``_bcbw_common``)."""
        return B.BcbwSeedrepReader._launch_rows_of(doc, EVAL_ID)

    # ---- objects ------------------------------------------------------------------------------
    resolve_object = B.BcbwSeedrepReader.resolve_object

    # ---- read ---------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        batch = RecordBatch(source_id_for(NAME, group), group)
        self.refusals = LC.RowRefusals(NAME)
        facts: Dict[str, Any] = {"group": group, "runs": 0, "rows": {}, "contrasts": {}, "not_run": [], "refusals": [], "instrument_seed_id": INSTRUMENT_SEED,
                                 "group_rule": GROUP_RULE, "anchor_rule": ANCHOR_RULE, "producer_key_rule": PRODUCER_KEY_RULE, "seed_axis_rule": SEED_AXIS_RULE,
                                 "merge_rule": MERGE_RULE, "dispatch_rule": DISPATCH_RULE, "dispatch_roots_rule": DISPATCH_ROOTS_RULE, "plan_absent_rule": PLAN_ABSENT_RULE,
                                 "row_table_rule": ROW_TABLE_RULE, "not_run_rule": NOT_RUN_RULE, "failed_dispatch_roots_never_read": []}
        parts = tag_parts(group)
        if parts is None or parts[1]:
            self.refusals.add(group, f"group {group!r} does not spell a seed tag ({MODEL_KEY}_s<N>)", inputs[0].uri if inputs else self.root)
            self._finish(group, facts)
            yield batch
            return
        seed_n = parts[0]
        sel_seed, sel_paired = self._group_selections(group)
        tags = {sel_seed: group, sel_paired: f"{MODEL_KEY}_s{seed_n}p"}
        facts.update(seed=seed_n, selections=[sel_seed, sel_paired], tags=sorted(tags.values()))
        if os.path.isdir(self._records_dir()):
            facts["failed_dispatch_roots_never_read"] = sorted(n for n in IC.listdir(self._records_dir()) if FAILED_DISPATCH_RE.match(n))
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
        facts.update(model_id=model_id, instrument_id=str(inst_row["instrument_id"]))
        sorted_in = self._sort_inputs(inputs)
        shared = sorted_in["shared"]
        ls_inp, pins_inp, matrix_inp, tn_inp = (shared.get(n) for n in SHARED_FILES)
        launch_rows = self._launch_rows(self._json(ls_inp)) if ls_inp is not None else None
        pins_doc = self._json(pins_inp) if pins_inp is not None else None
        three = self._json(tn_inp) if tn_inp is not None else None
        matrix = self._json(matrix_inp) if matrix_inp is not None else None
        pins_ok = isinstance(pins_doc, dict) and isinstance(pins_doc.get(MODEL_KEY), Mapping)
        three_ok = isinstance(three, dict) and str(three.get("schema")) == THREE_NUMBERS_SCHEMA and isinstance(three.get("rows"), dict) and str(three.get("eval")) == EVAL_ID
        matrix_ok = isinstance(matrix, dict) and str(matrix.get("schema")) == MATRIX_SCHEMA and isinstance(matrix.get("rows"), list)
        plans = sorted_in["plans"]
        for name, inp in (("launch_set", ls_inp), ("pins", pins_inp), ("matrix", matrix_inp), ("three_numbers", tn_inp)):
            facts[name] = LC.rel_home(inp.uri.rsplit("#", 1)[0]) if inp is not None else None
        facts["plans"] = {str(cap): {k: (LC.rel_home(v.uri.rsplit("#", 1)[0]) if v is not None else None) for k, v in (plans.get(cap) or {}).items()} for cap in LADDER[1:]}
        expected: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in launch_rows or []:
            sel = str(r.get("view") or "")
            if sel in tags and r.get("cell"):
                expected[(tags[sel], str(r["cell"]))] = r
        stores = sorted_in["stores"]
        rows_present = sorted({(tag, slug) for (tag, slug, _cap) in stores})
        filed: Dict[Tuple[str, str], Dict[str, Any]] = {}
        seen: Set[Tuple[str, str]] = set()
        for tag, slug in rows_present:
            okey = slug_to_object_key(slug)
            seen.add((tag, okey))
            rkey = f"{tag}|{okey}"
            if (tag, slug, CAP_BASE) not in stores:
                self.refusals.add(rkey, f"a cap-step store without its base store ({rel_store(tag, slug, CAP_BASE)} is absent)", self.root)
                continue
            if launch_rows is None:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{LAUNCH_SET_FILE} ({LAUNCH_SET_SCHEMA}, evals.{EVAL_ID}.launch) is absent or malformed beside the stores; no pin key and no expected object sha can be named", self.root)
                continue
            if not pins_ok:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{PINS_FILE} is absent or malformed beside the stores; the pins at the pin sha cannot be checked", self.root)
                continue
            if not three_ok:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{THREE_NUMBERS_FILE} ({THREE_NUMBERS_SCHEMA}, eval {EVAL_ID}) is absent or malformed beside the stores; nothing can be cross-checked", self.root)
                continue
            missing_plans = [f"{RECORDS_DIR}/{plan_head(cap)}/{n}" for cap in LADDER[1:] for n in (PLAN_FILE, PROMOTED_FILE) if (plans.get(cap) or {}).get(n) is None]
            if missing_plans:
                self.refusals.add(rkey, f"the ladder's deciding records are absent beside the stores ({', '.join(missing_plans)}); which caps a row reached cannot be confirmed", self.root)
                continue
            lrow = expected.get((tag, okey))
            if lrow is None:
                self.refusals.add(rkey, f"the launch set names no row for {_sel_of(tags, tag)}/{okey}; a store the launch set does not declare has no identity to bind to", ls_inp.uri)
                continue
            try:
                filed[(tag, okey)] = self._emit_row(ctx, group, tag, seed_n, _sel_of(tags, tag), model_id, inst_row, okey, slug, sorted_in, lrow, pins_doc, three,
                                                    (matrix if matrix_ok else None), ls_inp, pins_inp, tn_inp, matrix_inp, facts)
            except _PairRefusal as exc:
                self.refusals.add(rkey, exc.why, exc.where)
        for (tag, okey), lrow in sorted(expected.items()):
            if (tag, okey) in seen:
                continue
            slug = okey.replace("__", "--", 1)
            by_root = sorted_in["dispatch"].get((tag, slug, CAP_BASE)) or {}
            root = next((r for r in dispatch_heads(CAP_BASE, tag, okey) if r in by_root), None)
            dp = by_root.get(root) if root is not None else None
            dd = self._json(dp) if dp is not None else None
            facts["not_run"].append({"tuple_id": lrow.get("tuple_id"), "tag": tag, "object_key": okey, "paper_label": lrow.get("paper_label"), "class": lrow.get("class"),
                                     "dispatch_state": (dd.get("state") if isinstance(dd, Mapping) else None), "dispatch_status": (dd.get("status") if isinstance(dd, Mapping) else None),
                                     "dispatch_record": (LC.rel_home(dp.uri) if dp is not None else None), "rule": NOT_RUN_RULE})
        # the anchor blocks first (they ride the run payloads), then every row's records, then the contrasts -- a record is built once, never re-stamped.
        # Every contrast is DECIDED by the pass's anchor currency (CURRENCY_RULE; the frame loads once per pass through ctx.anchor_currency() and is
        # asked on the first comparison; a group without one never asks): the parks land on the VALUE rows' extra.delta_block before the records are added
        self._reanchor_computations = {}
        facts.update(contrasts_filed=0, contrasts_own=0, contrasts_reanchored=0, contrasts_parked=0, anchor_states={})
        contrasts = self._emit_contrasts(ctx, group, filed, expected, seen, facts)
        facts["dispatch_roots_read"] = sorted({str(d.get("root")) for st in filed.values() for d in (st["payload"]["extra"].get("dispatch") or {}).values() if d.get("root")})
        facts["rows_outside_ladder_plan"] = sorted(f"{k[0]}|{k[1]}" for k, st in filed.items() if st.get("plan_absent"))
        for key in sorted(filed):
            for rt, payload in filed[key]["records"]:
                batch.add(rt, payload, ctx.producer, ctx.now)
        for comp_row in self._reanchor_computations.values():
            batch.add("computation", comp_row, ctx.producer, ctx.now)      # the registry's statistic of a re-anchored contrast (extra.statistic_computation_id names it)
        for payload in contrasts:
            batch.add("run_contrast", payload, ctx.producer, ctx.now)
        facts["anchor_states"] = dict(sorted(facts["anchor_states"].items()))
        facts["reanchor_computations"] = len(self._reanchor_computations)
        self._finish(group, facts)
        yield batch

    def _sort_inputs(self, inputs: Sequence[Input]) -> Dict[str, Any]:
        """The group's inputs sorted: ``stores[(tag, slug, cap)] = {"top", "keys", "dir", ["summary"]}``, ``subsets[(tag, slug, cap)]``,
        ``dispatch[(tag, slug, cap)] = {<live root>: record}`` (the roots :func:`dispatch_heads` admits for the row), ``dispatch_states[<live root>]``,
        ``plans[cap] = {plan.json, promoted.json}`` and ``shared[name]`` (the ``#<group>`` aliases)."""
        stores: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        subsets: Dict[Tuple[str, str, int], Input] = {}
        dispatch: Dict[Tuple[str, str, int], Dict[str, Input]] = {}
        dispatch_states: Dict[str, Input] = {}
        plans: Dict[int, Dict[str, Input]] = {}
        shared: Dict[str, Input] = {}
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" in base:
                real = inp.uri.rsplit("#", 1)[0]
                name = os.path.basename(real)
                rel = os.path.relpath(real, self.root).split(os.sep)
                if len(rel) == 2 and rel[0] == RECORDS_DIR and name in SHARED_FILES:
                    shared[name] = inp
                elif len(rel) == 3 and rel[0] == RECORDS_DIR and name == DISPATCH_STATE_FILE and rel[1] in LIVE_DISPATCH_HEADS:
                    dispatch_states[rel[1]] = inp
                elif len(rel) == 3 and rel[0] == RECORDS_DIR and name in (PLAN_FILE, PROMOTED_FILE):
                    for cap in LADDER[1:]:
                        if rel[1] == plan_head(cap):
                            plans.setdefault(cap, {})[name] = inp
                continue
            rel = os.path.relpath(inp.uri, self.root).split(os.sep)
            if len(rel) == 4 and rel[3] == SUBSET_FILE and seed_tag_of(rel[1]) is not None:
                for cap in LADDER[1:]:
                    if rel[0] == subset_head(cap):
                        subsets[(rel[1], rel[2], cap)] = inp
                continue
            if len(rel) == 6 and rel[0] == RECORDS_DIR and rel[2] == DISPATCH_SUB and seed_tag_of(rel[3]) is not None:
                for cap in LADDER:
                    if rel[5] == dispatch_file(cap) and rel[1] in dispatch_heads(cap, rel[3], slug_to_object_key(rel[4])):
                        dispatch.setdefault((rel[3], rel[4], cap), {})[rel[1]] = inp
                continue
            if len(rel) < 5 or rel[0] != STORES_HEAD or seed_tag_of(rel[1]) is None:
                continue
            cap = cap_of_store_dir(rel[3])
            if cap is None:
                continue
            tag, slug = rel[1], rel[2]
            st = stores.setdefault((tag, slug, cap), {"top": {}, "keys": {}, "dir": os.path.join(self.root, *rel[:4])})
            body = rel[4:]
            if len(body) == 1:
                st["top"][body[0]] = inp
            elif len(body) == 2 and body[0] == slug_to_object_key(slug) and body[1] == CONTROLS_SUMMARY_FILE:
                st["summary"] = inp
            elif len(body) >= 3 and body[0] == "cells":
                files = st["keys"].setdefault(body[1], {})
                if len(body) == 3:
                    files[body[2]] = inp
                elif len(body) == 4 and body[2] in TASK_SUBDIRS and body[3] == OUTCOMES_FILE:
                    files[body[2]] = inp
        return {"stores": stores, "subsets": subsets, "dispatch": dispatch, "dispatch_states": dispatch_states, "plans": plans, "shared": shared}

    def _finish(self, group: str, facts: Dict[str, Any]) -> None:
        facts["refusals"] = [f"{it['key']} -- {it['why']}" for it in self.refusals.items]
        facts["n_refusals"] = len(self.refusals)
        facts["n_not_run"] = len(facts.get("not_run") or [])
        self._facts[group] = facts

    # ---- one store -----------------------------------------------------------------------------
    def _store_bundle(self, st: Mapping[str, Any], okey: str, slug: str, tag: str, cap: int) -> Dict[str, Any]:
        """The named files of one store directory (the base or a cap step), checked for presence and shape; raises :class:`_PairRefusal`."""
        top: Dict[str, Input] = st["top"]
        where = st["dir"]
        what = "base store" if cap == CAP_BASE else f"the {cap:,} store"
        missing = [n for n in ("RESULT", GEOMETRY_FILE) + (() if cap == CAP_BASE else (LADDER_STATE_FILE,)) if n not in top]
        if missing:
            raise _PairRefusal(f"{what} lacks {missing}", where)
        keys = st["keys"]
        if len(keys) != 1:
            raise _PairRefusal(f"{what} holds {len(keys)} store key directories under cells/ (expected exactly one)", where)
        store_key = next(iter(keys))
        files: Dict[str, Input] = keys[store_key]
        missing = [n for n in ("gen_stats.json", "manifest.json", "rows.jsonl") + TASK_SUBDIRS if n not in files]
        if missing:
            raise _PairRefusal(f"{what} {store_key} lacks {missing} (the task files are {'/'.join(TASK_SUBDIRS)}/{OUTCOMES_FILE})", os.path.join(where, "cells", store_key))
        try:
            ck = IC.parse_cell_key(store_key)
        except AdapterRefusal as exc:
            raise _PairRefusal(str(exc.why), os.path.join(where, "cells", store_key))
        if ck.cell_id != okey:
            raise _PairRefusal(f"{what}'s key spells object {ck.cell_id!r} but the directory spells {okey!r}", where)
        with open(top["RESULT"].path, "r", encoding="utf-8") as fh:
            result_lines = [ln.strip() for ln in fh if ln.strip()]
        if not result_lines or RESULT_COMPLETE not in result_lines[-1]:
            raise _PairRefusal(f"{what}: RESULT's last line does not read {RESULT_COMPLETE} ({(result_lines[-1] if result_lines else '<empty>')[:160]})", top["RESULT"].uri)
        result = parse_result_line(result_lines[-1])
        if result is None or result.get("result") != "COMPLETE" or result.get("rc") != "0" or result.get("status") != DISPATCH_OK:
            raise _PairRefusal(f"{what}: RESULT's last line does not parse as a {RESULT_HEAD} COMPLETE / {DISPATCH_OK} line with rc=0 ({result_lines[-1][:160]})", top["RESULT"].uri)
        man = self._json(files["manifest.json"]) or {}
        gs_all = self._json(files["gen_stats.json"]) or {}
        gs = gs_all.get(TASK) if isinstance(gs_all.get(TASK), Mapping) else None
        if gs is None:
            raise _PairRefusal(f"{what}: gen_stats.json carries no {TASK} block", files["gen_stats.json"].uri)
        outcomes = self._json(files["outcomes"]) or {}
        if str(outcomes.get("task")) != TASK or not isinstance(outcomes.get("items"), list):
            raise _PairRefusal(f"{what}: the outcome record is not {TASK}'s (task {outcomes.get('task')!r})", files["outcomes"].uri)
        summary = self._json(files["summary"]) or {}
        done = self._json(files["done"]) or {}
        if str(done.get("task")) != TASK:
            raise _PairRefusal(f"{what}: done/{OUTCOMES_FILE} names task {done.get('task')!r}, not {TASK!r}", files["done"].uri)
        geometry_doc = self._json(top[GEOMETRY_FILE]) or {}
        ladder_state = None
        if cap != CAP_BASE:
            ladder_state = self._json(top[LADDER_STATE_FILE]) or {}
            if str(ladder_state.get("schema")) != LADDER_STATE_SCHEMA:
                raise _PairRefusal(f"{what}: {LADDER_STATE_FILE} schema is {ladder_state.get('schema')!r}, not {LADDER_STATE_SCHEMA}", top[LADDER_STATE_FILE].uri)
            plan_blk = ladder_state.get("rung") if isinstance(ladder_state.get("rung"), Mapping) else {}
            if LC.int_or_none(plan_blk.get("cap")) != cap:
                raise _PairRefusal(f"{what}: {LADDER_STATE_FILE} plans cap {plan_blk.get('cap')!r}, not this store's {cap}", top[LADDER_STATE_FILE].uri)
        cap_ladder = None
        if CAP_LADDER_FILE in top:
            cap_ladder = self._json(top[CAP_LADDER_FILE]) or {}
            if str(cap_ladder.get("schema")) != CAP_LADDER_SCHEMA:
                raise _PairRefusal(f"{what}: {CAP_LADDER_FILE} schema is {cap_ladder.get('schema')!r}, not {CAP_LADDER_SCHEMA}", top[CAP_LADDER_FILE].uri)
            if str(cap_ladder.get("cell_id")) != store_key or LC.int_or_none(cap_ladder.get("cap")) != cap:
                raise _PairRefusal(f"{what}: {CAP_LADDER_FILE} classifies store {cap_ladder.get('cell_id')!r} at cap {cap_ladder.get('cap')!r}, not this store ({store_key} at {cap})",
                                   top[CAP_LADDER_FILE].uri)
        # the rows: the per-draw `acc@<trial>` rows (item, trial) -> (value, line, created_at); the per-item `acc` rows; the aggregates
        draw_rows: Dict[DrawKey, Tuple[float, int, Optional[float]]] = {}
        item_rows: Dict[str, float] = {}
        agg_extra: Dict[str, Any] = {}
        agg_values: Dict[str, Any] = {}
        consts: Dict[str, set] = {}
        for ln, r in IC.iter_jsonl(files["rows.jsonl"].path):
            if str(r.get("task") or "") != TASK:
                continue
            metric = str(r.get("metric"))
            if r.get("item_id") == "__aggregate__":
                ex = r.get("extra_json")
                try:
                    exd = json.loads(ex) if isinstance(ex, str) and ex else (ex if isinstance(ex, Mapping) else {})
                except ValueError:
                    exd = {}
                agg_extra.update({k: v for k, v in exd.items() if k in ("object_artifact_sha256", "producer_tree_sha", "route_kind", "sampling_mode", "view", "bank_view", "samples_per_item")})
                agg_values[metric] = LC.num_or_none(r.get("value"))
                continue
            iid = str(r.get("item_id"))
            if metric.startswith(f"{HEADLINE_ROW_METRIC}@"):
                try:
                    trial = int(metric.split("@", 1)[1])
                except ValueError:
                    continue
                k = (iid, trial)
                if k in draw_rows:
                    raise _PairRefusal(f"{what}: rows.jsonl carries draw {k} twice", files["rows.jsonl"].uri)
                draw_rows[k] = (float(r.get("value")), ln, LC.num_or_none(r.get("created_at")))
            elif metric == HEADLINE_ROW_METRIC:
                if iid in item_rows:
                    raise _PairRefusal(f"{what}: rows.jsonl carries item {iid} twice", files["rows.jsonl"].uri)
                item_rows[iid] = float(r.get("value"))
            else:
                continue
            for c in ("env_digest", "model_id", "model_revision", "harness_config_hash", "intervention_hash", "created_at"):
                v = r.get(c)
                if v is not None:
                    consts.setdefault(c, set()).add(v if c != "created_at" else float(v))
        controls_summary = self._json(st["summary"]) if st.get("summary") is not None else None
        return {"dir": st["dir"], "tag": tag, "slug": slug, "cap": cap, "store_key": store_key, "ck": ck, "top": top, "files": files, "manifest": man, "gen_stats": gs,
                "outcomes": outcomes, "summary": summary, "done": done, "geometry_doc": geometry_doc, "ladder_state": ladder_state, "cap_ladder": cap_ladder,
                "controls_summary": controls_summary, "draw_rows": draw_rows, "item_rows": item_rows, "agg_extra": agg_extra, "agg_values": agg_values, "consts": consts,
                "result": result, "result_last": result_lines[-1], "result_earlier": result_lines[:-1], "what": what}

    @staticmethod
    def _draws(bundle: Mapping[str, Any]) -> Dict[DrawKey, Dict[str, Any]]:
        """The store's draws off its outcome record: ``(item_id, sample_idx) -> {at_cap, scored, value (strict), loose, inst counts, stop_reason, …}``."""
        out: Dict[DrawKey, Dict[str, Any]] = {}
        where = bundle["files"]["outcomes"].uri
        for it in bundle["outcomes"]["items"]:
            if not isinstance(it, Mapping) or it.get("item_id") in (None, "") or LC.int_or_none(it.get("sample_idx")) is None:
                raise _PairRefusal(f"{bundle['what']}: the outcome record carries a draw without an item id / sample_idx", where)
            k = draw_key(it["item_id"], it["sample_idx"])
            if k in out:
                raise _PairRefusal(f"{bundle['what']}: the outcome record carries draw {k} twice", where)
            at_cap, scored = bool(it.get("at_cap")), bool(it.get("scored"))
            v = it.get("value")
            if scored and v not in (0, 1, 0.0, 1.0, True, False):
                raise _PairRefusal(f"{bundle['what']}: draw {k} is scored with value {v!r}", where)
            if not scored and v is not None:
                raise _PairRefusal(f"{bundle['what']}: draw {k} is unscored with value {v!r}", where)
            if scored and "prompt_level_strict" in it and float(bool(it.get("prompt_level_strict"))) != float(v):
                raise _PairRefusal(f"{bundle['what']}: draw {k}'s value {v!r} is not its prompt_level_strict {it.get('prompt_level_strict')!r}", where)
            if at_cap and str(it.get("stop_reason") or "") not in IC.AT_CAP_FINISH_REASONS:
                raise _PairRefusal(f"{bundle['what']}: draw {k} is at_cap with stop_reason {it.get('stop_reason')!r}", where)
            out[k] = {"at_cap": at_cap, "scored": scored, "value": (float(v) if scored else None), "prompt_level_loose": (float(bool(it.get("prompt_level_loose"))) if scored else None),
                      "inst_level_strict": (LC.int_or_none(it.get("inst_level_strict")) if scored else None), "inst_level_loose": (LC.int_or_none(it.get("inst_level_loose")) if scored else None),
                      "num_instructions": LC.int_or_none(it.get("num_instructions")), "stop_reason": it.get("stop_reason"), "completion_tokens": LC.int_or_none(it.get("completion_tokens")),
                      "seed": LC.int_or_none(it.get("seed")), "unparseable": it.get("unparseable"), "cap": bundle["cap"]}
        return out

    def _check_store(self, bundle: Mapping[str, Any], draws: Mapping[DrawKey, Mapping[str, Any]], expect_items: Optional[int]) -> Dict[str, Any]:
        """The store's own records against its draws: ``rows.jsonl`` per draw and per item, ``gen_stats.json`` / the summary / the outcome header / the
        completion marker, the aggregates, the manifest's cap and k; returns the producer's per-store reading (:func:`store_reading`)."""
        what = bundle["what"]
        reading = store_reading(draws)
        rows = bundle["draw_rows"]
        want = {k: d["value"] for k, d in draws.items() if d["value"] is not None}
        if set(rows) != set(want) or any(rows[k][0] != want[k] for k in want):
            raise _PairRefusal(f"{what}: rows.jsonl's {len(rows)} per-draw {HEADLINE_ROW_METRIC}@<trial> rows do not reproduce the outcome record's {len(want)} scored draws", bundle["files"]["rows.jsonl"].uri)
        by_item: Dict[str, List[float]] = {}
        for (iid, _t), v in want.items():
            by_item.setdefault(iid, []).append(v)
        item_rows = bundle["item_rows"]
        # the base store writes a per-item `acc` row beside the per-draw rows (the mean of the item's two draws); a cap-step store writes the per-draw rows alone
        if (item_rows or bundle["cap"] == CAP_BASE) and (set(item_rows) != set(by_item) or any(abs(item_rows[i] - sum(v) / len(v)) > 1e-9 for i, v in by_item.items())):
            raise _PairRefusal(f"{what}: rows.jsonl's {len(item_rows)} per-item {HEADLINE_ROW_METRIC} rows are not the means of the items' draws", bundle["files"]["rows.jsonl"].uri)
        if expect_items is not None and reading["n_items"] != expect_items:
            raise _PairRefusal(f"{what}: the outcome record holds {reading['n_items']} items, not {expect_items}", bundle["files"]["outcomes"].uri)
        gs = bundle["gen_stats"]
        cap = bundle["cap"]
        for name, g, uri in (("gen_stats.json", gs, bundle["files"]["gen_stats.json"].uri), (f"summary/{OUTCOMES_FILE}", bundle["summary"], bundle["files"]["summary"].uri)):
            got = (LC.int_or_none(g.get("n")), LC.int_or_none(g.get("n_scored")), LC.int_or_none(g.get("n_unscored")), LC.int_or_none(g.get("n_at_cap")), LC.int_or_none(g.get("max_tokens")),
                   LC.int_or_none(g.get("n_requests")), LC.int_or_none(g.get("samples_per_item")))
            want_t = (reading["n_items"], reading["n_scored_items"], reading["n_unscored_items"], reading["n_at_cap"], cap, reading["n_draws"], K_TRIALS)
            gacc = LC.num_or_none(g.get("acc"))
            acc_ok = (reading["acc"] is None and gacc is None) or (reading["acc"] is not None and gacc is not None and abs(float(gacc) - reading["acc"]) < 1e-9)
            if got != want_t or not acc_ok:
                raise _PairRefusal(f"{what}: {name} (n {got[0]}, n_scored {got[1]}, n_unscored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}, n_requests {got[5]}, samples_per_item {got[6]}, acc {gacc}) "
                                   f"does not reproduce the outcome record (n {want_t[0]}, n_scored {want_t[1]}, n_unscored {want_t[2]}, n_at_cap {want_t[3]}, cap {want_t[4]}, draws {want_t[5]}, k {want_t[6]}, acc {reading['acc']})", uri)
            if str(g.get("thinking") or "") != self.THINKING_SPELLING or str(g.get("task") or "") != TASK:
                raise _PairRefusal(f"{what}: {name} thinking {g.get('thinking')!r} / task {g.get('task')!r} are not {self.THINKING_SPELLING!r} / {TASK!r}", uri)
            for field in TRIAL_MEAN_FIELDS:
                if field in g and not trial_means_agree(g.get(field), reading[field]):
                    raise _PairRefusal(f"{what}: {name}:{field} {g.get(field)} is not the recount {reading[field]}", uri)
        oc = bundle["outcomes"]
        got = (LC.int_or_none(oc.get("n_attempted_items")), LC.int_or_none(oc.get("n_attempted_samples")), LC.int_or_none(oc.get("n_scored")), LC.int_or_none(oc.get("n_at_cap")),
               LC.int_or_none(oc.get("max_tokens")), LC.int_or_none(oc.get("samples_per_item")))
        if got != (reading["n_items"], reading["n_draws"], reading["n_scored_items"], reading["n_at_cap"], cap, K_TRIALS):
            raise _PairRefusal(f"{what}: the outcome record's header (n_attempted_items {got[0]}, n_attempted_samples {got[1]}, n_scored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}, "
                               f"samples_per_item {got[5]}) does not reproduce its draws (items {reading['n_items']}, draws {reading['n_draws']}, scored {reading['n_scored_items']}, "
                               f"at cap {reading['n_at_cap']}, cap {cap}, k {K_TRIALS})", bundle["files"]["outcomes"].uri)
        if str(oc.get("thinking") or "") != self.THINKING_SPELLING:
            raise _PairRefusal(f"{what}: the outcome record's thinking is {oc.get('thinking')!r}, not {self.THINKING_SPELLING!r}", bundle["files"]["outcomes"].uri)
        done = bundle["done"]
        dgot = (LC.int_or_none(done.get("n_items")), LC.int_or_none(done.get("n_draws")), LC.int_or_none(done.get("n_expected_draws")))
        if dgot[0] != reading["n_items"] or dgot[1] != reading["n_draws"] or dgot[2] not in (None, reading["n_draws"]):
            raise _PairRefusal(f"{what}: done/{OUTCOMES_FILE} (n_items {dgot[0]}, n_draws {dgot[1]}, n_expected_draws {dgot[2]}) is not this store's completion marker "
                               f"({reading['n_items']} items, {reading['n_draws']} draws)", bundle["files"]["done"].uri)
        agg_acc, agg_n = bundle["agg_values"].get("acc_aggregate"), bundle["agg_values"].get("n")
        if "acc_aggregate" in bundle["agg_values"] and not ((agg_acc is None and reading["acc"] is None) or (agg_acc is not None and reading["acc"] is not None and abs(agg_acc - reading["acc"]) < 1e-9)):
            raise _PairRefusal(f"{what}: rows.jsonl's acc_aggregate {agg_acc} is not the store's reading {reading['acc']}", bundle["files"]["rows.jsonl"].uri)
        if "n" in bundle["agg_values"] and LC.int_or_none(agg_n) != reading["n_items"]:
            raise _PairRefusal(f"{what}: rows.jsonl's aggregate n {agg_n} is not {reading['n_items']}", bundle["files"]["rows.jsonl"].uri)
        man = bundle["manifest"]
        tc = man.get("tinycap") if isinstance(man.get("tinycap"), Mapping) else {}
        cap_by_task = tc.get("max_tokens_by_task") if isinstance(tc.get("max_tokens_by_task"), Mapping) else {}
        k_by_task = tc.get("samples_by_task") if isinstance(tc.get("samples_by_task"), Mapping) else {}
        if LC.int_or_none(cap_by_task.get(TASK)) != cap or LC.int_or_none(k_by_task.get(TASK)) != K_TRIALS:
            raise _PairRefusal(f"{what}: manifest.json:tinycap max_tokens_by_task[{TASK}] {cap_by_task.get(TASK)!r} / samples_by_task[{TASK}] {k_by_task.get(TASK)!r} are not the store's cap {cap} / k {K_TRIALS}",
                               bundle["files"]["manifest.json"].uri)
        return reading

    # ---- the producer's records --------------------------------------------------------------------
    def _dispatch_record(self, sorted_in: Mapping[str, Any], tag: str, slug: str, cap: int, bundle: Mapping[str, Any], obj_sha: Optional[str], tree: str) -> Dict[str, Any]:
        """The dispatcher's record of one store (:data:`DISPATCH_RULE`): ``state done`` attesting ``lora_native`` files; a record whose attestation read was
        rate-limited is judged by the store (its manifest attests ``lora_native`` in four spellings and its RESULT read COMPLETE / OK — both already
        checked); anything else refuses. Its identity fields must be this store's. Raises :class:`_PairRefusal`."""
        okey = slug_to_object_key(slug)
        by_root = sorted_in["dispatch"].get((tag, slug, cap)) or {}
        roots = dispatch_heads(cap, tag, okey)
        root = next((r for r in roots if r in by_root), None)
        if root is None:
            named = " or ".join(f"{RECORDS_DIR}/{r}/{DISPATCH_SUB}/{tag}/{slug}/{dispatch_file(cap)}" for r in roots)
            raise _PairRefusal(f"no dispatcher record ({named}) beside the stores for this PRESENT {bundle['what']}; a row files only under a done record attesting {SERVING_ENGINE}", self._records_dir())
        inp = by_root[root]
        head = f"{RECORDS_DIR}/{root}/{DISPATCH_SUB}/{tag}/{slug}/{dispatch_file(cap)}"
        d = self._json(inp) or {}
        where = inp.uri
        if str(d.get("schema") or "") != DISPATCH_SCHEMA:
            raise _PairRefusal(f"{head} carries schema {d.get('schema')!r}, not {DISPATCH_SCHEMA}", where)
        cid = configuration_id(tag, slug, cap)
        if str(d.get("configuration_id") or "") != cid or LC.int_or_none(d.get("cap")) != cap or str(d.get("cell") or "") != slug_to_object_key(slug):
            raise _PairRefusal(f"{head} names configuration {d.get('configuration_id')!r} / cap {d.get('cap')!r} / key {d.get('cell')!r}, not this store's ({cid}, {cap}, {slug_to_object_key(slug)})", where)
        if str(d.get("requested_serving_path") or "") != SERVING_ENGINE:
            raise _PairRefusal(f"{head} requested serving path {d.get('requested_serving_path')!r}, not {SERVING_ENGINE}", where)
        out_dir = str(d.get("out_dir") or "")
        if not out_dir.endswith(rel_store(tag, slug, cap)):
            raise _PairRefusal(f"{head}:out_dir {out_dir!r} does not spell this store's path (…/{rel_store(tag, slug, cap)})", where)
        if cap != CAP_BASE and not str(d.get("item_subset") or "").endswith(rel_subset(tag, slug, cap)):
            raise _PairRefusal(f"{head}:item_subset {d.get('item_subset')!r} does not spell this cap's subset (…/{rel_subset(tag, slug, cap)})", where)
        if (d.get("object_sha256") or None) != obj_sha:
            raise _PairRefusal(f"{head}:object_sha256 {str(d.get('object_sha256'))[:16]!r} is not the resolved object {str(obj_sha)[:16]!r}", where)
        rec = d.get("rec") if isinstance(d.get("rec"), Mapping) else {}
        if rec and str(rec.get("tree_sha") or "") != tree:
            raise _PairRefusal(f"{head}:rec.tree_sha {str(rec.get('tree_sha'))[:12]!r} is not the store's tree {tree[:12]}", where)
        state, status, attested = str(d.get("state") or ""), str(d.get("status") or ""), d.get("serving_path_attested")
        error = str(d.get("error") or "")
        rate_limited = (state == DISPATCH_FAILED and status == DISPATCH_RATE_LIMITED_STATUS and attested is None and DISPATCH_RATE_LIMITED_MARK in error.lower())
        if rate_limited:
            verdict, judged_by = "store", "the store: the dispatcher's attestation READ hit the volume's list rate limit; the store's manifest attests lora_native and its RESULT reads COMPLETE / OK"
        elif state == DISPATCH_DONE and status == DISPATCH_OK and str(attested or "") == SERVING_ENGINE:
            verdict, judged_by = "record", "the dispatcher's record: state done, status OK, serving_path_attested lora_native"
        else:
            raise _PairRefusal(f"{head} reads state {state!r} / status {status!r} / serving_path_attested {attested!r} ({error[:200] or 'no error named'}); a row files only under a done record "
                               f"attesting {SERVING_ENGINE} (or a rate-limited attestation read, judged by the store)", where)
        st_inp = sorted_in["dispatch_states"].get(root)
        state_doc = self._json(st_inp) if st_inp is not None else None
        in_done = in_failed = None
        if isinstance(state_doc, Mapping) and str(state_doc.get("schema") or "") == DISPATCH_STATE_SCHEMA:
            in_done = cid in (state_doc.get("done") or {})
            in_failed = cid in (state_doc.get("failed") or {})
            if verdict == "record" and not in_done:
                raise _PairRefusal(f"{RECORDS_DIR}/{root}/{DISPATCH_STATE_FILE} does not list {cid} under done, the record says done", st_inp.uri.rsplit("#", 1)[0])
        return {"record": LC.rel_home(where), "root": root, "roots_considered": list(roots), "other_root_records": sorted(set(by_root) - {root}),
                "state": state, "status": status, "serving_path_requested": d.get("requested_serving_path"), "serving_path_attested": attested,
                "serving_path_attested_source": d.get("serving_path_attested_source"), "error": (error or None), "verdict": verdict, "judged_by": judged_by, "rate_limited_read": rate_limited,
                "configuration_id": cid, "launch_key": d.get("launch_key"), "run_id_legacy": d.get("run_id"), "app_id": d.get("app_id"), "app_name": d.get("app_name"),
                "function_call_id": d.get("fc_id"), "geometry_eval_legacy": d.get("geometry_eval"), "spawned_at": IC.utc_from_epoch(LC.num_or_none(d.get("spawned_at"))),
                "finished_at": IC.utc_from_epoch(LC.num_or_none(d.get("t"))), "attempt_id": rec.get("attempt_id"), "driver_rc": rec.get("driver_rc"),
                "dispatch_state_lists_done": in_done, "dispatch_state_lists_failed": in_failed, "rule": DISPATCH_RULE}

    def _plan_records(self, plans: Mapping[int, Mapping[str, Input]], cap: int) -> Tuple[Dict[str, Dict[str, Any]], Set[str], Dict[str, Dict[str, Any]], Input, Input]:
        """The ladder's deciding records of one cap step: ``plan.json:planned`` by ``cell_key``, ``promoted.json:promoted`` configuration ids and ``not_promoted``
        by ``job_id``; both records must spell the schema, the sha agreement and this cap."""
        pi, qi = plans[cap][PLAN_FILE], plans[cap][PROMOTED_FILE]
        plan = self._json(pi) or {}
        prom = self._json(qi) or {}
        if str(prom.get("schema") or "") != PLAN_SCHEMA or LC.int_or_none(prom.get("cap")) != cap or LC.int_or_none(prom.get("from_cap")) != LADDER[LADDER.index(cap) - 1]:
            raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}/{PROMOTED_FILE} (schema {prom.get('schema')!r}, from_cap {prom.get('from_cap')!r}, cap {prom.get('cap')!r}) is not the {cap:,} step's plan",
                               qi.uri.rsplit("#", 1)[0])
        if LC.int_or_none(plan.get("from_cap")) != LADDER[LADDER.index(cap) - 1] or not isinstance(plan.get("planned"), list):
            raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}/{PLAN_FILE} (from_cap {plan.get('from_cap')!r}) is not the {cap:,} step's plan", pi.uri.rsplit("#", 1)[0])
        if str(plan.get("sha") or "") != str(prom.get("sha") or ""):
            raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}: plan.json (sha {str(plan.get('sha'))[:12]}) and promoted.json (sha {str(prom.get('sha'))[:12]}) are not one plan", pi.uri.rsplit("#", 1)[0])
        planned = {str(e.get("cell_key")): dict(e) for e in plan["planned"] if isinstance(e, Mapping) and e.get("cell_key")}
        promoted = {str(e.get("configuration_id")) for e in (prom.get("promoted") or []) if isinstance(e, Mapping)}
        not_promoted = {str(e.get("job_id")): dict(e) for e in (prom.get("not_promoted") or []) if isinstance(e, Mapping)}
        return planned, promoted, not_promoted, pi, qi

    def _plan_decision(self, plans: Mapping[int, Mapping[str, Input]], cap: int, selection: str, okey: str, tag: str, slug: str, merged_at_cap: int, n_draws: int,
                       step_present: bool) -> Dict[str, Any]:
        """The plan's decision for the step INTO ``cap`` against the merged reading the reader derived before it: the entry's ``exit`` must match the step
        store's presence (``escalate`` iff present), its merged ``n_at_cap`` / ``n_draws`` the recount; returns the decision (its ``eligible_draws`` classify
        the at-cap draws the ladder left in place). Raises :class:`_PairRefusal`."""
        planned, promoted, not_promoted, pi, qi = self._plan_records(plans, cap)
        cid = configuration_id(tag, slug, cap)
        key = f"{MODEL_KEY}/{selection}/{okey}"
        entry = planned.get(key)
        where = pi.uri.rsplit("#", 1)[0]
        if entry is None:
            # PLAN_ABSENT_RULE: a row the ladder never planned files only as a base-only row whose share could not have triggered the step, judged by the
            # threshold the plan records themselves declare; nothing of the ladder's classification is reconstructed for it
            if step_present or cid in promoted:
                raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}/{PLAN_FILE} plans no row {key} but "
                                   f"{f'the {cap:,} store is present' if step_present else f'{PROMOTED_FILE} promotes {cid}'}; whether the ladder promoted this row to {cap:,} is undecided", where)
            declared = {(LC.num_or_none(e["decision"].get("threshold")), str(e["decision"].get("rule") or "")) for e in planned.values() if isinstance(e.get("decision"), Mapping)}
            if len(declared) != 1 or next(iter(declared))[0] is None:
                raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}/{PLAN_FILE} plans no row {key} and declares no single trigger threshold across its planned rows "
                                   f"({sorted(str(t) for t in declared)}); an unplanned row cannot be judged (PLAN_ABSENT_RULE)", where)
            threshold, rule = next(iter(declared))
            share = (merged_at_cap / n_draws) if n_draws else 0.0
            if share > float(threshold):
                raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}/{PLAN_FILE} plans no row {key} and its at-cap share before {cap:,} ({merged_at_cap} of {n_draws} draws = {share:.4f}) "
                                   f"exceeds the declared trigger {float(threshold):g}: a step to {cap:,} was due and the ladder is undecided here (PLAN_ABSENT_RULE)", where)
            return {"cap": cap, "record": LC.rel_home(where), "promoted_record": LC.rel_home(qi.uri.rsplit("#", 1)[0]), "job_id": None, "exit": PLAN_EXIT_NOT_PLANNED, "status": PLAN_EXIT_NOT_PLANNED,
                    "note": (f"the ladder's deciding records plan no row {key} (dispatched outside the ladder matrix); its at-cap share {share:.4f} ({merged_at_cap} of {n_draws} draws) is at or "
                             f"below the declared trigger {float(threshold):g}, so no step to {cap:,} was due; its at-cap draws stay unclassified (PLAN_ABSENT_RULE)"),
                    "share_at_cap": share, "n_at_cap": merged_at_cap, "n_draws": n_draws, "n_eligible": None, "eligible_draws": None, "reading": None, "chain": None,
                    "threshold": float(threshold), "rule": rule, "trigger_provenance": None, "promoted": False, "planned": False, "decided_by": None, "rule_applied": PLAN_ABSENT_RULE}
        exit_ = str(entry.get("exit") or "")
        status = str(entry.get("status") or "")
        if (cid in promoted) != step_present:
            raise _PairRefusal(f"{RECORDS_DIR}/{plan_head(cap)}/{PROMOTED_FILE} {'promotes' if cid in promoted else 'does not promote'} {cid} but the {cap:,} store is {'present' if step_present else 'absent'}",
                               qi.uri.rsplit("#", 1)[0])
        if step_present and exit_ != EXIT_ESCALATE:
            raise _PairRefusal(f"the {cap:,} store is present but the plan's exit for {key} is {exit_!r}, not {EXIT_ESCALATE!r}", where)
        if not step_present and exit_ not in EXITS_STOP:
            raise _PairRefusal(f"no {cap:,} store, but the plan's exit for {key} is {exit_!r} (status {status!r}), not one of {EXITS_STOP}; the ladder is undecided or incomplete here", where)
        dec = entry.get("decision") if isinstance(entry.get("decision"), Mapping) else {}
        if LC.int_or_none(dec.get("n_at_cap")) != merged_at_cap or LC.int_or_none(dec.get("n_draws")) != n_draws:
            raise _PairRefusal(f"the plan's merged reading before {cap:,} for {key} (n_at_cap {dec.get('n_at_cap')!r} of {dec.get('n_draws')!r} draws) is not the reader's merge ({merged_at_cap} of {n_draws})", where)
        if str(dec.get("exit") or "") != exit_ or bool(dec.get("escalate")) != step_present:
            raise _PairRefusal(f"the plan's decision block for {key} (exit {dec.get('exit')!r}, escalate {dec.get('escalate')!r}) disagrees with its entry (exit {exit_!r}) / the store's presence", where)
        eligible = draw_keys(dec.get("eligible_draws"))
        if LC.int_or_none(dec.get("n_eligible")) not in (None, len(eligible)):
            raise _PairRefusal(f"the plan's decision for {key} counts n_eligible {dec.get('n_eligible')!r} but names {len(eligible)} eligible draws", where)
        np_entry = not_promoted.get(str(entry.get("job_id") or ""))
        if not step_present and np_entry is not None and str(np_entry.get("exit") or "") != exit_:
            raise _PairRefusal(f"{PROMOTED_FILE}:not_promoted[{entry.get('job_id')}] exit {np_entry.get('exit')!r} is not plan.json's {exit_!r}", qi.uri.rsplit("#", 1)[0])
        return {"cap": cap, "record": LC.rel_home(where), "promoted_record": LC.rel_home(qi.uri.rsplit("#", 1)[0]), "job_id": entry.get("job_id"), "exit": exit_, "status": status,
                "note": entry.get("note"), "share_at_cap": LC.num_or_none(entry.get("share_at_cap")), "n_at_cap": LC.int_or_none(dec.get("n_at_cap")), "n_draws": LC.int_or_none(dec.get("n_draws")),
                "n_eligible": len(eligible), "eligible_draws": sorted(eligible), "reading": dec.get("reading"), "chain": dec.get("chain"), "threshold": LC.num_or_none(dec.get("threshold")),
                "rule": dec.get("rule"), "trigger_provenance": dec.get("trigger_provenance"), "promoted": step_present, "decided_by": dec.get("decided_by")}

    def _cross_check_three(self, tn: Mapping[str, Any], bundle: Mapping[str, Any], reading: Mapping[str, Any], obj_sha: Optional[str], tag: str, selection: str, seed_n: int,
                           paired: bool, okey: str, slug: str, tree: str) -> List[str]:
        """The producer's per-STORE three-number row (``w4_three_numbers/1``) against the store's recount: the value (acc over the store's items), the bound
        (acc x n_scored / n), the at-cap count and share over the store's draws, the counts, the identity and path fields, the RESULT line, the per-trial means."""
        c = reading
        man = bundle["manifest"]
        cap = bundle["cap"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False, tol: float = 1e-9) -> None:
            ok = rounded_equal(theirs, mine, tol) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        eq("status", str(tn.get("status")), "complete")
        eq("tag", str(tn.get("tag")), tag)
        eq("view", str(tn.get("view")), selection)
        eq("seed", LC.int_or_none(tn.get("seed")), seed_n)
        eq("paired", bool(tn.get("paired")), paired)
        eq("cell_id", str(tn.get("cell_id")), okey)
        eq("cell_slug", str(tn.get("cell_slug")), slug)
        eq("arm", str(tn.get("arm")), str(LC.cell_grammar_legacy(okey)["paper_label"]))
        eq("cap", LC.int_or_none(tn.get("cap")), cap)
        eq("max_tokens", LC.int_or_none(tn.get("max_tokens")), cap)
        eq("is_rung", bool(tn.get("is_rung")), cap != CAP_BASE)
        eq("merged", bool(tn.get("merged")), False)
        eq("store", str(tn.get("store")), rel_store(tag, slug, cap))
        eq("summary", str(tn.get("summary")), f"cells/{bundle['store_key']}/summary/{OUTCOMES_FILE}")
        eq("result_line", str(tn.get("result_line")), bundle["result_last"])
        eq("engine", str(tn.get("engine")), str(man.get("serving_engine")))
        eq("serving_identity", str(tn.get("serving_identity")), str(man.get("serving_identity")))
        eq("producer_tree", str(tn.get("producer_tree")), tree)
        eq("bank_view", tn.get("bank_view"), (man.get("route") or {}).get("bank_view") if isinstance(man.get("route"), Mapping) else None)
        eq("n_attempted", LC.int_or_none(tn.get("n_attempted")), c["n_items"])
        eq("n_scored", LC.int_or_none(tn.get("n_scored")), c["n_scored_items"])
        eq("n_unscored", LC.int_or_none(tn.get("n_unscored")), c["n_unscored_items"])
        eq("draws", LC.int_or_none(tn.get("draws")), c["n_draws"])
        eq("samples_per_item", LC.int_or_none(tn.get("samples_per_item")), K_TRIALS)
        eq("value", LC.num_or_none(tn.get("value")), c["acc"], rounded=True)
        eq(BOUND_SPELLING, LC.num_or_none(tn.get(BOUND_SPELLING)), ((c["acc"] * c["n_scored_items"] / c["n_items"]) if (c["acc"] is not None and c["n_items"]) else None), rounded=True)
        eq("at_cap_n", LC.int_or_none(tn.get("at_cap_n")), c["n_at_cap"])
        eq("at_cap_share", LC.num_or_none(tn.get("at_cap_share")), c["share_at_cap"], rounded=True)
        eq("object_sha256", tn.get("object_sha256"), obj_sha)
        if obj_sha is not None:
            eq("object_sha256_source", str(tn.get("object_sha256_source")), "route")
        blk = tn.get("ifeval") if isinstance(tn.get("ifeval"), Mapping) else {}
        for field in TRIAL_MEAN_FIELDS:
            if not trial_means_agree(blk.get(field), c[field]):
                bad.append(f"ifeval.{field}: producer {blk.get(field)!r}, re-derived {c[field]!r}")
        return bad

    def _cross_check_summary(self, summary: Mapping[str, Any], base: Mapping[str, Any], reading: Mapping[str, Any], obj_sha: Optional[str], okey: str, tree: str) -> List[str]:
        """The producer's ``controls_summary.json`` beside the base store against the store: the key, the object, the tree and the base reading."""
        bad: List[str] = []
        if str(summary.get("cell_id") or "") != okey:
            bad.append(f"cell_id {summary.get('cell_id')!r} is not {okey}")
        prov = summary.get("provenance") if isinstance(summary.get("provenance"), Mapping) else None
        if prov is not None and (prov.get("artifact_sha256") or None) != obj_sha:
            bad.append(f"provenance.artifact_sha256 {str(prov.get('artifact_sha256'))[:16]!r} is not the resolved object {str(obj_sha)[:16]}")
        src = (summary.get("sources") or {}).get(TASK) if isinstance(summary.get("sources"), Mapping) else None
        if isinstance(src, Mapping):
            if str(src.get("cell_key") or "") not in ("", base["store_key"]):
                bad.append(f"sources[{TASK}].cell_key {src.get('cell_key')!r} is not the base store key {base['store_key']}")
            if str(src.get("producer_tree_sha") or "") not in ("", tree):
                bad.append(f"sources[{TASK}].producer_tree_sha {str(src.get('producer_tree_sha'))[:12]!r} is not the manifest's tree {tree[:12]}")
        g = (summary.get("controls") or {}).get(TASK) if isinstance(summary.get("controls"), Mapping) else None
        if not isinstance(g, Mapping):
            bad.append(f"controls[{TASK}] is absent")
            return bad
        gacc = LC.num_or_none(g.get("acc"))
        if (reading["acc"] is None) != (gacc is None) or (reading["acc"] is not None and abs(float(gacc) - reading["acc"]) > 1e-9):
            bad.append(f"controls[{TASK}].acc {gacc} is not the base reading {reading['acc']}")
        got = (LC.int_or_none(g.get("n")), LC.int_or_none(g.get("n_scored")), LC.int_or_none(g.get("n_at_cap")), LC.int_or_none(g.get("max_tokens")), str(g.get("thinking") or ""), str(g.get("task") or ""))
        if got != (reading["n_items"], reading["n_scored_items"], reading["n_at_cap"], CAP_BASE, self.THINKING_SPELLING, TASK):
            bad.append(f"controls[{TASK}] (n {got[0]}, n_scored {got[1]}, n_at_cap {got[2]}, max_tokens {got[3]}, thinking {got[4]!r}, task {got[5]!r}) is not the base store's "
                       f"(n {reading['n_items']}, n_scored {reading['n_scored_items']}, n_at_cap {reading['n_at_cap']}, max_tokens {CAP_BASE}, thinking {self.THINKING_SPELLING!r}, task {TASK!r})")
        return bad

    @staticmethod
    def _matrix_row(matrix: Optional[Mapping[str, Any]], tag: str, okey: str) -> Optional[Dict[str, Any]]:
        if matrix is None:
            return None
        hits = [r for r in (matrix.get("rows") or []) if isinstance(r, Mapping) and str(r.get("tag")) == tag and str(r.get("cell")) == okey]
        return dict(hits[0]) if len(hits) == 1 else None

    @staticmethod
    def _check_matrix_row(row: Mapping[str, Any], tag: str, slug: str, obj: LC.ResolvedObject, steered: bool, chain: Sequence[int], where: str) -> Dict[str, Any]:
        """The matrix's row against the store: the staged id and object sha, the slug and out roots, thinking OFF, k, the caps, the serving path."""
        bad: List[str] = []
        if steered and (str(row.get("staged_id") or "") != str(obj.staged_id) or str(row.get("object_sha256") or "") != obj.object_id):
            bad.append(f"staged_id {row.get('staged_id')!r} / object_sha256 {str(row.get('object_sha256'))[:16]!r} are not the resolved {obj.staged_id!r} / {str(obj.object_id)[:16]}")
        if not steered and (row.get("staged_id") or row.get("object_sha256")):
            bad.append("an unsteered or prompted row names a staged id / object sha")
        if str(row.get("cell_slug") or "") != slug or str(row.get("thinking") or "") != "off" or LC.int_or_none(row.get("samples_per_item")) != K_TRIALS:
            bad.append(f"cell_slug {row.get('cell_slug')!r} / thinking {row.get('thinking')!r} / samples_per_item {row.get('samples_per_item')!r} are not ({slug}, off, {K_TRIALS})")
        if [LC.int_or_none(x) for x in (row.get("caps") or [])] != list(LADDER) or LC.int_or_none(row.get("cap")) != CAP_BASE or LC.int_or_none(row.get("cap_ceiling")) != CAP_CEILING:
            bad.append(f"caps {row.get('caps')!r} / cap {row.get('cap')!r} / cap_ceiling {row.get('cap_ceiling')!r} are not the ladder {list(LADDER)}")
        if str(row.get("serving_path") or "") != SERVING_ENGINE or str(row.get("sampling_mode") or "") != SAMPLING_MODE:
            bad.append(f"serving_path {row.get('serving_path')!r} / sampling_mode {row.get('sampling_mode')!r} are not {SERVING_ENGINE} / {SAMPLING_MODE}")
        out_root = str(row.get("out_root") or "")
        if out_root and not out_root.endswith(rel_store(tag, slug, CAP_BASE)):
            bad.append(f"out_root {out_root!r} does not spell the base store's path")
        step_roots = row.get("rung_out_roots") if isinstance(row.get("rung_out_roots"), Mapping) else {}
        for cap in chain[1:]:
            got = str(step_roots.get(str(cap)) or "")
            if got and not got.endswith(rel_store(tag, slug, cap)):
                bad.append(f"rung_out_roots[{cap}] {got!r} does not spell the {cap:,} store's path")
        if bad:
            raise _PairRefusal(f"{MATRIX_FILE} row {row.get('job_id')!r} disagrees with the store on " + "; ".join(bad), where)
        return {"record": LC.rel_home(where), "job_id": row.get("job_id"), "index": row.get("index"), "route_kind_legacy": row.get("route_kind"), "prompt_class": row.get("prompt_class"),
                "budget_class": row.get("budget_class"), "status": row.get("status"), "fitted": row.get("fitted"), "configuration_ids": row.get("configuration_ids")}

    # ---- decode ---------------------------------------------------------------------------------
    _numbers_agree = staticmethod(G.Gsm8kSeedrepAdapter._numbers_agree)

    def _match_policy(self, model_id: str, block: Mapping[str, Any]) -> Tuple[Optional[DP.Policy], str]:
        pf = self._policy_file()
        if pf is None:
            return None, "the committed decode-policy export could not be loaded"
        try:
            p = pf.resolve(model_id, eval_id=EVAL_ID, thinking_mode=self.THINKING_SPELLING)
        except (DP.NoPolicy, KeyError, ValueError) as exc:
            return None, f"no registered decode policy row for {model_id} / {EVAL_ID} thinking {self.THINKING_SPELLING}: {exc}"
        if self._numbers_agree(block, p.numbers()):
            return p, "exact by the stated sampling numbers"
        return None, f"the sent numbers {RP._numbers(block)} differ from the default policy row {p.policy_id} {RP._numbers(p.numbers())}"

    # ---- one (tag, object key) row -------------------------------------------------------------
    def _identity_checks(self, man: Mapping[str, Any], where_man: str, okey: str, obj: LC.ResolvedObject, steered: bool, model_id: str, agg_extra: Mapping[str, Any], rows_uri: str,
                         what: str) -> Tuple[str, str, str, str, Dict[str, Any]]:
        """The served model / revision / engine / identity / path and the attested object shas of one store's manifest, by object kind; returns
        ``(served hf id, revision, identity, 16-hex object sha, fit provenance)``."""
        seeds = self.seeds
        served_hf = str(man.get("model_id") or "")
        if seeds.model_for_hf_id(served_hf) != model_id:
            raise _PairRefusal(f"{what} serves {served_hf!r}, not the seed tag's model {model_id}", where_man)
        rev = str(man.get("model_revision") or "")
        pinned_rev = str((seeds.models.get(model_id) or {}).get("hf_revision") or "")
        if _SHA1.match(pinned_rev) and rev != pinned_rev:
            raise _PairRefusal(f"{what} serves revision {rev[:12] or '<none>'}, not the model seed's {pinned_rev[:12]}", where_man)
        route = man.get("route") if isinstance(man.get("route"), Mapping) else {}
        resolved = (man.get("resolved_config") or {}).get("serving") if isinstance(man.get("resolved_config"), Mapping) else None
        served = (man.get("tinycap") or {}).get("served") if isinstance(man.get("tinycap"), Mapping) else None
        engines = {"serving_engine": man.get("serving_engine"), "route.serving_engine": route.get("serving_engine"),
                   "resolved_config.serving.serving_engine": (resolved or {}).get("serving_engine") if isinstance(resolved, Mapping) else None,
                   "tinycap.served.serving_engine": (served or {}).get("serving_engine") if isinstance(served, Mapping) else None}
        bad_eng = [k for k, v in engines.items() if str(v or "") != SERVING_ENGINE]
        if bad_eng:
            raise _PairRefusal(f"{what}: manifest.json attests {', '.join(f'{k} {engines[k]!r}' for k in bad_eng)}, not {SERVING_ENGINE} (the four spellings must agree)", where_man)
        ident = str(man.get("serving_identity") or "")
        want_ident = STEERED_IDENTITY if steered else PASSTHROUGH_IDENTITY
        if ident != want_ident:
            raise _PairRefusal(f"{what}: serving_identity {ident!r} is not the {obj.object_kind} object's {want_ident!r}", where_man)
        want_path = STEERED_PATH_LEGACY if steered else PASSTHROUGH_PATH_LEGACY
        if str(man.get("serving_path") or "") != want_path:
            raise _PairRefusal(f"{what}: manifest.json:serving_path {man.get('serving_path')!r} is not the {obj.object_kind} object's {want_path!r}", where_man)
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        sha16 = str(fitp.get("object_sha") or "")
        prov = man.get("provenance") if isinstance(man.get("provenance"), Mapping) else {}
        prov_sha = str(prov.get("artifact_sha256") or "")
        by_key = route.get("object_artifact_sha256_by_cell") if isinstance(route.get("object_artifact_sha256_by_cell"), Mapping) else {}
        route_sha = str(route.get("object_artifact_sha256") or "") or str(by_key.get(str(man.get("harness_cell_id") or "")) or "")
        attested = str(agg_extra.get("object_artifact_sha256") or "")
        if steered:
            if not _SHA16.match(sha16) or not str(obj.object_id).startswith(sha16):
                raise _PairRefusal(f"{what}: manifest.json:fit_provenance.object_sha {sha16!r} is no prefix of the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})", where_man)
            if prov_sha != obj.object_id:
                raise _PairRefusal(f"{what}: manifest.json:provenance.artifact_sha256 {prov_sha[:16] or '<none>'}… is not the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})", where_man)
            if route_sha != obj.object_id:
                raise _PairRefusal(f"{what}: manifest.json:route.object_artifact_sha256[_by_cell] {route_sha[:16] or '<none>'}… is not the pinned object {str(obj.object_id)[:16]}…", where_man)
            if attested != obj.object_id:
                raise _PairRefusal(f"{what}: the rows' aggregate attests object {attested[:16] or '<none>'}…, the roster / pins {str(obj.object_id)[:16]}…", rows_uri)
        elif obj.object_kind == "system_instruction":
            sys_sha = str((man.get("resolved_config") or {}).get("system_sha") or "")
            si_sha = str(obj.object_id).split(":", 1)[1]
            if not (_SHA16.match(sys_sha) and si_sha.startswith(sys_sha)):
                raise _PairRefusal(f"{what}: manifest.json:resolved_config.system_sha {sys_sha!r} is no prefix of the declared prompt's text sha {si_sha[:16]}…", where_man)
            if sha16 or prov_sha or route_sha or attested:
                raise _PairRefusal(f"{what}: the prompted store attests an object sha", where_man)
        elif sha16 or prov_sha or route_sha or attested:
            raise _PairRefusal(f"{what}: the unsteered store attests an object sha", where_man)
        return served_hf, rev, ident, sha16, dict(fitp)

    def _seed_and_decode_checks(self, man: Mapping[str, Any], gs: Mapping[str, Any], where_man: str, what: str, cap: int) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """The store's seed rule (the fixed IFEval base, no replicate block — SEED_AXIS_RULE), thinking OFF, the sent decode numbers against the manifest's
        sampling row; returns ``(the seed rule's text, the ifeval seed block, the sent block)``."""
        inspect_seeds = (man.get("seeds") or {}).get("inspect") if isinstance(man.get("seeds"), Mapping) else None
        inspect_seeds = inspect_seeds if isinstance(inspect_seeds, Mapping) else {}
        if isinstance(inspect_seeds.get("replicate"), Mapping):
            raise _PairRefusal(f"{what}: manifest.json:seeds.inspect carries a replicate block ({json.dumps(inspect_seeds.get('replicate'))[:120]}); this instrument has no per-seed rule (D69)", where_man)
        ife = inspect_seeds.get("ifeval") if isinstance(inspect_seeds.get("ifeval"), Mapping) else {}
        seed_rule_text = str(ife.get("rule") or inspect_seeds.get("rule") or "")
        if not seed_rule_text:
            raise _PairRefusal(f"{what}: manifest.json:seeds.inspect names no seed rule", where_man)
        if LC.int_or_none(ife.get("seed_base")) != IFEVAL_SEED_BASE:
            raise _PairRefusal(f"{what}: manifest.json:seeds.inspect.ifeval.seed_base {ife.get('seed_base')!r} is not the instrument's fixed base {IFEVAL_SEED_BASE}", where_man)
        thinking = man.get("thinking") if isinstance(man.get("thinking"), Mapping) else {}
        by_task = thinking.get("by_task") if isinstance(thinking.get("by_task"), Mapping) else {}
        if str(thinking.get("mode") or "") != self.THINKING_SPELLING or str(by_task.get(TASK) or "") != self.THINKING_SPELLING or thinking.get("enable_thinking") is True:
            raise _PairRefusal(f"{what}: manifest.json:thinking (mode {thinking.get('mode')!r}, by_task[{TASK}] {by_task.get(TASK)!r}) is not {self.THINKING_SPELLING!r}", where_man)
        sampling = man.get("sampling") if isinstance(man.get("sampling"), Mapping) else {}
        if str(sampling.get("mode") or "") != SAMPLING_MODE:
            raise _PairRefusal(f"{what}: manifest.json:sampling.mode {sampling.get('mode')!r} is not {SAMPLING_MODE!r}", where_man)
        block = {k: gs.get(k) for k in DP.PARAM_KEYS}
        if not self._numbers_agree(block, {k: sampling.get(k) for k in DP.PARAM_KEYS}):
            raise _PairRefusal(f"{what}: gen_stats.json sent {RP._numbers(block)}, manifest.json:sampling spells {RP._numbers(sampling)}", where_man)
        ledger = str(man.get("usage_ledger") or "")
        if not ledger.startswith(VOLUME_HEAD) or not ledger.endswith("/usage_ledger.jsonl"):
            raise _PairRefusal(f"{what}: manifest.json:usage_ledger {ledger!r} is not a store path on the volume ({VOLUME_HEAD}…/usage_ledger.jsonl)", where_man)
        return seed_rule_text, dict(ife), block

    def _emit_row(self, ctx: AdapterContext, group: str, tag: str, seed_n: int, selection: str, model_id: str, inst_row: Mapping[str, Any], okey: str, slug: str,
                  sorted_in: Mapping[str, Any], lrow: Mapping[str, Any], pins_doc: Mapping[str, Any], three: Mapping[str, Any], matrix: Optional[Mapping[str, Any]],
                  ls_inp: Input, pins_inp: Input, tn_inp: Input, matrix_inp: Optional[Input], facts: Dict[str, Any]) -> Dict[str, Any]:
        """Reads one (tag, object key) — its base store and every cap step it reached — merges the draws at the completion cap and BUILDS its records
        (``records``: ``(record type, payload)`` in emission order, the run first; the caller adds them once the group's anchor blocks are stamped)."""
        seeds = self.seeds
        producer = ctx.producer
        records: List[Tuple[str, Dict[str, Any]]] = []
        paired = bool(tag_parts(tag)[1])
        stores = sorted_in["stores"]
        plans = sorted_in["plans"]
        # -- the chain: the base store and every consecutive cap step present (a gap in the chain refuses)
        chain: List[int] = [CAP_BASE]
        for cap in LADDER[1:]:
            if (tag, slug, cap) in stores:
                if chain[-1] != LADDER[LADDER.index(cap) - 1]:
                    raise _PairRefusal(f"a {cap:,} store without the {LADDER[LADDER.index(cap) - 1]:,} store before it; the ladder has a gap", stores[(tag, slug, cap)]["dir"])
                chain.append(cap)
        bundles = {cap: self._store_bundle(stores[(tag, slug, cap)], okey, slug, tag, cap) for cap in chain}
        base = bundles[CAP_BASE]
        man = base["manifest"]
        where_man = base["files"]["manifest.json"].uri
        if str(man.get("cell_id") or "") != okey:
            raise _PairRefusal(f"manifest.json names object key {man.get('cell_id')!r}, not {okey!r}", where_man)
        # -- the tree: the manifest's, RESULT's and the outcome record's, on every store of the chain
        tree = str((man.get("producer") or {}).get("tree_sha") or "")
        if not _SHA1.match(tree):
            raise _PairRefusal("manifest.json:producer.tree_sha names no 40-hex tree", where_man)
        for cap, b in bundles.items():
            t = str((b["manifest"].get("producer") or {}).get("tree_sha") or "")
            if t != tree or str(b["result"].get("sha") or "") != tree or str(b["outcomes"].get("producer_tree_sha") or "") != tree:
                raise _PairRefusal(f"{b['what']}: the manifest's tree {t[:12]!r}, RESULT's {str(b['result'].get('sha'))[:12]!r} and the outcome record's {str(b['outcomes'].get('producer_tree_sha'))[:12]!r} "
                                   f"are not one tree ({tree[:12]}, the base store's)", b["files"]["manifest.json"].uri)
            if str(b["manifest"].get("cell_id") or "") != okey:
                raise _PairRefusal(f"{b['what']}: manifest.json names object key {b['manifest'].get('cell_id')!r}, not {okey!r}", b["files"]["manifest.json"].uri)
        # -- the object: the launch set's pin key and expected sha, the harness roster / pins, the mirrored pins
        expected_sha = lrow.get("object_sha256_expected") or None
        launch_pin_key = lrow.get("pin_key") or None
        try:
            obj, op = self.resolve_object(okey, selection, pin_key=launch_pin_key)
        except IC.Unresolved as exc:
            raise _PairRefusal(f"object: {exc.why}", ls_inp.uri)
        arditi = okey == ARDITI_OBJECT_KEY
        steered = obj.object_kind not in ("none", "system_instruction")
        lclass = str(lrow.get("class") or "")
        # the launch set's class grammar: the paired tag's `none` is its ANCHOR (PAIRED_ANCHOR_RULE) and a prompted key is `prompted` on either tag; only a
        # fitted key under the paired tag is the `composed` object
        want_class = "arditi" if arditi else ("anchor" if obj.object_kind == "none" else ("prompted" if obj.object_kind == "system_instruction" else ("composed" if paired else "arm")))
        if lclass != want_class:
            raise _PairRefusal(f"the launch set files this row as class {lclass!r}, the key's grammar says {want_class!r}", ls_inp.uri)
        if LC.int_or_none(lrow.get("seed_base")) != seed_n or str(lrow.get("tag") or "") != tag or str(lrow.get("cell_slug") or "") != slug:
            raise _PairRefusal(f"the launch set's row (tag {lrow.get('tag')!r}, seed_base {lrow.get('seed_base')!r}, cell_slug {lrow.get('cell_slug')!r}) is not this store's ({tag}, {seed_n}, {slug})", ls_inp.uri)
        if str(lrow.get("paper_label") or "") != str(obj.paper_label):
            raise _PairRefusal(f"the launch set labels the row {lrow.get('paper_label')!r}, the key's grammar {obj.paper_label!r}", ls_inp.uri)
        if str(lrow.get("out_root") or "") not in ("", rel_store(tag, slug, CAP_BASE)):
            raise _PairRefusal(f"the launch set's out_root {lrow.get('out_root')!r} does not spell this base store's path", ls_inp.uri)
        cids = lrow.get("configuration_ids") if isinstance(lrow.get("configuration_ids"), Mapping) else {}
        for cap in LADDER:
            if str(cap) in cids and str(cids[str(cap)]) != configuration_id(tag, slug, cap):
                raise _PairRefusal(f"the launch set's configuration_ids[{cap}] {cids[str(cap)]!r} is not {configuration_id(tag, slug, cap)}", ls_inp.uri)
        if steered:
            if launch_pin_key != obj.staged_id:
                raise _PairRefusal(f"the launch set's pin key {launch_pin_key!r} is not the roster's staged id {obj.staged_id!r} at {op.rung if op else selection}", ls_inp.uri)
            if expected_sha != obj.object_id:
                raise _PairRefusal(f"the launch set expects object {str(expected_sha)[:16]}…, the pins name {str(obj.object_id)[:16]}… for {obj.staged_id}", ls_inp.uri)
            mirrored = (pins_doc.get(MODEL_KEY) or {}).get(obj.staged_id) if isinstance(pins_doc.get(MODEL_KEY), Mapping) else None
            if not isinstance(mirrored, Mapping) or str(mirrored.get("object_sha256") or "") != obj.object_id:
                raise _PairRefusal(f"the mirrored pins ({RECORDS_DIR}/{PINS_FILE}) name {str((mirrored or {}).get('object_sha256'))[:16]!r} for {obj.staged_id}, the harness pins {str(obj.object_id)[:16]}",
                                   pins_inp.uri.rsplit("#", 1)[0])
        elif launch_pin_key is not None or expected_sha is not None:
            raise _PairRefusal(f"the launch set carries a pin key / expected sha ({launch_pin_key!r} / {str(expected_sha)[:16]!r}) for an unsteered or prompted key", ls_inp.uri)
        obj_sha = obj.object_id if steered else None
        roster_paired_of = None
        if paired:
            roster, _pins = self._roster_pins()
            blk = ((roster.get("targets") or {}).get(MODEL_KEY) or {}).get("views", {}).get(selection) or {}
            roster_paired_of = blk.get("paired_of")
            if str(roster_paired_of or "") != f"s{seed_n}":
                raise _PairRefusal(f"the roster's selection {selection} says paired_of {roster_paired_of!r}; the tag says s{seed_n}", self.roster_path)
        # -- every store of the chain: the served identity, the seed rule, the decode numbers, the dispatcher's record, the draws and the per-store reading
        per_cap: Dict[int, Dict[str, Any]] = {}
        draws_by_cap: Dict[int, Dict[DrawKey, Dict[str, Any]]] = {}
        served_hf = rev = ident = sha16 = ""
        fitp: Dict[str, Any] = {}
        seed_rule_text = ""
        ife_seed: Dict[str, Any] = {}
        block: Dict[str, Any] = {}
        for cap in chain:
            b = bundles[cap]
            m = b["manifest"]
            wm = b["files"]["manifest.json"].uri
            served_hf, rev, ident, s16, fp = self._identity_checks(m, wm, okey, obj, steered, model_id, b["agg_extra"], b["files"]["rows.jsonl"].uri, b["what"])
            srt, ife, blk = self._seed_and_decode_checks(m, b["gen_stats"], wm, b["what"], cap)
            if cap == CAP_BASE:
                sha16, fitp, seed_rule_text, ife_seed, block = s16, fp, srt, ife, blk
            elif not self._numbers_agree(block, blk):
                raise _PairRefusal(f"{b['what']} sent decode {RP._numbers(blk)}, the base store {RP._numbers(block)}; one ladder must send one row", b["files"]["gen_stats.json"].uri)
            draws_by_cap[cap] = self._draws(b)
            reading = self._check_store(b, draws_by_cap[cap], N_ITEMS if cap == CAP_BASE else None)
            dispatch = self._dispatch_record(sorted_in, tag, slug, cap, b, obj_sha, tree)
            key = f"{tag}/{slug}" if cap == CAP_BASE else f"{tag}/{slug}/rung{cap}"
            tn = (three.get("rows") or {}).get(key)
            if not isinstance(tn, Mapping):
                raise _PairRefusal(f"{THREE_NUMBERS_FILE} carries no row for {key}", tn_inp.uri)
            bad = self._cross_check_three(tn, b, reading, obj_sha, tag, selection, seed_n, paired, okey, slug, tree)
            if bad:
                raise _PairRefusal(f"{THREE_NUMBERS_FILE} row {key} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), tn_inp.uri)
            endpoint = b["geometry_doc"].get("endpoint") if isinstance(b["geometry_doc"].get("endpoint"), Mapping) else {}
            lora_cfg = endpoint.get("lora_config") if isinstance(endpoint.get("lora_config"), Mapping) else {}
            geometry = m.get("geometry") if isinstance(m.get("geometry"), Mapping) else {}
            attempts = [a for a in (m.get("resume_attempts") or []) if isinstance(a, Mapping)]
            completed = [a for a in attempts if bool(a.get("completed"))]
            if not completed:
                raise _PairRefusal(f"{b['what']}: manifest.json:resume_attempts names no completed attempt", wm)
            accepted = completed[-1]
            if str(accepted.get("tree_sha") or "") != tree:
                raise _PairRefusal(f"{b['what']}: the completed attempt's tree {str(accepted.get('tree_sha'))[:12]} is not the store's {tree[:12]}", wm)
            per_cap[cap] = {"store": LC.rel_home(b["dir"]), "store_key": b["store_key"], "harness_config_hash": m.get("harness_config_hash"), "reading": reading,
                            "producer_row": key, "producer_value": LC.num_or_none(tn.get("value")), "producer_bound": LC.num_or_none(tn.get(BOUND_SPELLING)),
                            "producer_at_cap_n": LC.int_or_none(tn.get("at_cap_n")), "producer_delta_value_vs_none": LC.num_or_none(tn.get("delta_value_vs_none")),
                            "producer_record_agrees": True, "dispatch": dispatch, "result_line": b["result_last"], "attempts_before_accepted": b["result_earlier"],
                            "accepted_attempt": {k: accepted.get(k) for k in ("attempt_id", "started", "ended", "generation_calls", "ts")},
                            "standup": {"record": LC.rel_home(b["top"][GEOMETRY_FILE].uri), "engine": endpoint.get("engine"), "serving_path_legacy": b["geometry_doc"].get("serving_path"),
                                        "max_lora_rank": LC.int_or_none(lora_cfg.get("max_lora_rank")), "max_loras": LC.int_or_none(lora_cfg.get("max_loras")), "node": endpoint.get("node"),
                                        "max_lora_rank_rule": MAX_LORA_RANK_RULE},
                            "served_window": LC.int_or_none(geometry.get("max_model_len")), "served_max_num_seqs": LC.int_or_none(geometry.get("max_num_seqs")),
                            "geometry_eval_legacy": (geometry.get("eval") or ((b["geometry_doc"].get("geometry") or {}).get("eval") if isinstance(b["geometry_doc"].get("geometry"), Mapping) else None)),
                            "generated_utc": IC.utc_from_iso(b["result"].get("at")), "job_name": b["result"].get("name"),
                            "cap_ladder": (LC.rel_home(b["top"][CAP_LADDER_FILE].uri) if b["cap_ladder"] is not None else None),
                            "ladder_state": (LC.rel_home(b["top"][LADDER_STATE_FILE].uri) if b["ladder_state"] is not None else None)}
        # -- the base store's producer summary
        if base.get("controls_summary") is not None:
            bad = self._cross_check_summary(base["controls_summary"], base, per_cap[CAP_BASE]["reading"], obj_sha, okey, tree)
            if bad:
                raise _PairRefusal(f"{CONTROLS_SUMMARY_FILE} disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), stores[(tag, slug, CAP_BASE)]["summary"].uri)
        # -- THE MERGE: the cap steps' subsets against the source store's classification and the step manifest, then the draw-wise replacement
        steps: List[Tuple[int, Set[DrawKey], Dict[DrawKey, Dict[str, Any]]]] = []
        loops: Dict[int, Set[DrawKey]] = {}
        classified_at: Dict[int, str] = {}
        subset_facts: Dict[int, Dict[str, Any]] = {}
        for i, cap in enumerate(chain[1:], start=1):
            prev = chain[i - 1]
            b, pb = bundles[cap], bundles[prev]
            sub_inp = sorted_in["subsets"].get((tag, slug, cap))
            if sub_inp is None:
                raise _PairRefusal(f"no {rel_subset(tag, slug, cap)} beside the {cap:,} store; the draws it re-drew are unnamed", self.root)
            sub = self._json(sub_inp) or {}
            if str(sub.get("schema") or "") != SUBSET_SCHEMA or LC.int_or_none(sub.get("cap")) != cap or LC.int_or_none(sub.get("source_cap")) != prev:
                raise _PairRefusal(f"{rel_subset(tag, slug, cap)} (schema {sub.get('schema')!r}, cap {sub.get('cap')!r}, source_cap {sub.get('source_cap')!r}) is not the {prev:,} -> {cap:,} subset", sub_inp.uri)
            redrawn = draw_keys(sub.get("draws"))
            if not redrawn or LC.int_or_none(sub.get("n_draws")) != len(redrawn) or LC.int_or_none(sub.get("n_items")) != len({k[0] for k in redrawn}):
                raise _PairRefusal(f"{rel_subset(tag, slug, cap)} counts n_draws {sub.get('n_draws')!r} / n_items {sub.get('n_items')!r} but names {len(redrawn)} draws of {len({k[0] for k in redrawn})} items", sub_inp.uri)
            if str(sub.get("source_identity_hash") or "") != str(pb["manifest"].get("harness_config_hash") or "") or not str(sub.get("source_out_root") or "").endswith(rel_store(tag, slug, prev)):
                raise _PairRefusal(f"{rel_subset(tag, slug, cap)} names source {sub.get('source_identity_hash')!r} at {sub.get('source_out_root')!r}, not the {prev:,} store ({pb['manifest'].get('harness_config_hash')!r})", sub_inp.uri)
            isub = b["manifest"].get("item_subset") if isinstance(b["manifest"].get("item_subset"), Mapping) else {}
            if draw_keys(isub.get("draws")) != redrawn or LC.int_or_none(isub.get("cap")) != cap or LC.int_or_none(isub.get("source_cap")) != prev \
                    or str(isub.get("source_identity_hash") or "") != str(pb["manifest"].get("harness_config_hash") or ""):
                raise _PairRefusal(f"{b['what']}: manifest.json:item_subset (cap {isub.get('cap')!r}, source_cap {isub.get('source_cap')!r}, {len(draw_keys(isub.get('draws')))} draws, source "
                                   f"{isub.get('source_identity_hash')!r}) is not the subset record ({cap}, {prev}, {len(redrawn)} draws, {pb['manifest'].get('harness_config_hash')!r})", b["files"]["manifest.json"].uri)
            plan_blk = b["ladder_state"].get("rung") if isinstance(b["ladder_state"].get("rung"), Mapping) else {}
            if LC.int_or_none(plan_blk.get("n_draws")) != len(redrawn) or str(plan_blk.get("source_identity_hash") or "") != str(pb["manifest"].get("harness_config_hash") or "") \
                    or (isub.get("subset_sha16") and plan_blk.get("subset_sha16") and str(isub.get("subset_sha16")) != str(plan_blk.get("subset_sha16"))):
                raise _PairRefusal(f"{b['what']}: {LADDER_STATE_FILE} plans {plan_blk.get('n_draws')!r} draws from {plan_blk.get('source_identity_hash')!r} (subset {plan_blk.get('subset_sha16')!r}), the manifest "
                                   f"{len(redrawn)} from {pb['manifest'].get('harness_config_hash')!r} (subset {isub.get('subset_sha16')!r})", b["top"][LADDER_STATE_FILE].uri)
            cl = pb["cap_ladder"]
            if cl is None:
                raise _PairRefusal(f"{pb['what']} carries no {CAP_LADDER_FILE} but the ladder promoted from it to {cap:,}; the loops and the eligible draws are unclassified", pb["dir"])
            eligible = draw_keys(cl.get("eligible_draws"))
            loop_set = draw_keys(cl.get("loop_draws"))
            if eligible != redrawn:
                raise _PairRefusal(f"{pb['what']}: {CAP_LADDER_FILE}:eligible_draws ({len(eligible)}) is not the {cap:,} subset ({len(redrawn)}); the ladder re-drew other draws than it classified eligible",
                                   pb["top"][CAP_LADDER_FILE].uri)
            if loop_set & redrawn:
                raise _PairRefusal(f"{pb['what']}: {len(loop_set & redrawn)} loop draw(s) were re-drawn at {cap:,}; a loop is never re-drawn", pb["top"][CAP_LADDER_FILE].uri)
            if str(cl.get("exit") or "") != EXIT_ESCALATE or LC.int_or_none(cl.get("next_cap")) != cap:
                raise _PairRefusal(f"{pb['what']}: {CAP_LADDER_FILE} declares exit {cl.get('exit')!r} / next_cap {cl.get('next_cap')!r} but a {cap:,} store follows", pb["top"][CAP_LADDER_FILE].uri)
            loops[prev] = loop_set
            classified_at[prev] = f"{CAP_LADDER_FILE} of the {prev:,} store (loop_draws)"
            steps.append((cap, redrawn, draws_by_cap[cap]))
            subset_facts[cap] = {"record": LC.rel_home(sub_inp.uri), "n_draws": len(redrawn), "n_items": len({k[0] for k in redrawn}), "n_loop_source": LC.int_or_none(sub.get("n_loop")),
                                 "n_truncated_source": LC.int_or_none(sub.get("n_truncated")), "n_boundary_source": LC.int_or_none(sub.get("n_boundary")), "subset_sha16": isub.get("subset_sha16"),
                                 "source_classification": {"record": LC.rel_home(pb["top"][CAP_LADDER_FILE].uri), "n_loop": len(loop_set), "n_eligible": len(eligible), "exit": cl.get("exit"),
                                                           "share_at_cap": LC.num_or_none(cl.get("share_at_cap")), "rules": (cl.get("rules") if isinstance(cl.get("rules"), Mapping) else None)}}
        try:
            final, from_cap = completion_cap_merge(draws_by_cap[CAP_BASE], steps)
        except ValueError as exc:
            raise _PairRefusal(f"the completion-cap merge refuses: {exc}", base["files"]["outcomes"].uri)
        final_cap = chain[-1]
        # -- the ladder's deciding record at every step INTO a cap: the step taken (escalate) and the step NOT taken after the final cap
        decisions: Dict[int, Dict[str, Any]] = {}
        state_draws: Dict[DrawKey, Dict[str, Any]] = dict(draws_by_cap[CAP_BASE])
        unclassified: Set[DrawKey] = set()
        plan_absent = False
        for i, cap in enumerate(LADDER[1:], start=1):
            prev = LADDER[i - 1]
            if prev not in chain:
                break
            merged_at_cap = sum(1 for d in state_draws.values() if d.get("at_cap"))
            dec = self._plan_decision(plans, cap, selection, okey, tag, slug, merged_at_cap, len(state_draws), cap in chain)
            decisions[cap] = dec
            if cap in chain:
                for k in steps[i - 1][1]:
                    state_draws[k] = draws_by_cap[cap][k]
            elif dec["exit"] == PLAN_EXIT_NOT_PLANNED:
                # the ladder never planned this row (PLAN_ABSENT_RULE): its at-cap draws carry no classification — unclassified, never loops, never the plan's eligible draws
                unclassified = {k for k, d in state_draws.items() if d.get("at_cap")}
                classified_at[prev] = None
                plan_absent = True
                break
            else:
                # the ladder stopped here: the plan's eligible draws are the truncated non-loop draws it left in place; the other at-cap draws are loops
                at_cap_here = {k for k, d in state_draws.items() if d.get("at_cap")}
                eligible = set(dec["eligible_draws"])
                if not eligible <= at_cap_here:
                    raise _PairRefusal(f"the plan's eligible draws for the step to {cap:,} name {len(eligible - at_cap_here)} draw(s) not at the cap in the reader's merge", dec["record"])
                implied = at_cap_here - eligible
                loops[prev] = {k for k in implied if k in draws_by_cap[prev]}
                classified_at[prev] = f"the ladder plan's decision for the step to {cap:,} (at cap minus eligible_draws; exit {dec['exit']})"
                known: Set[DrawKey] = set()
                for ks in loops.values():
                    known |= ks
                if implied != known:
                    raise _PairRefusal(f"the merged state's {len(implied)} at-cap non-eligible draw(s) at the stop before {cap:,} are not the {len(known)} loop draw(s) the records classify", dec["record"])
                if bundles[prev]["cap_ladder"] is not None:
                    cl_loops = draw_keys(bundles[prev]["cap_ladder"].get("loop_draws"))
                    if cl_loops != loops[prev]:
                        raise _PairRefusal(f"{bundles[prev]['what']}: {CAP_LADDER_FILE} names {len(cl_loops)} loop draws, the plan's decision implies {len(loops[prev])}", bundles[prev]["top"][CAP_LADDER_FILE].uri)
                break
        # -- the classes of the final at-cap draws
        loop_keys: Set[DrawKey] = set()
        for cap, ks in loops.items():
            loop_keys |= {k for k in ks if final[k]["source_cap"] == cap}
        not_escalated: Set[DrawKey] = set()
        ceiling: Set[DrawKey] = set()
        for k, d in final.items():
            if not d.get("at_cap"):
                continue
            if k in loop_keys:
                d["at_cap_class"] = "loop"
            elif k in unclassified:
                d["at_cap_class"] = "unclassified"
            elif d["source_cap"] == CAP_CEILING:
                d["at_cap_class"] = "ceiling"
                ceiling.add(k)
            else:
                d["at_cap_class"] = "not_escalated"
                not_escalated.add(k)
        # the plan's eligible draws at the stop ARE the not-escalated draws; a ladder stopping at the ceiling names no step; an unplanned row has no such draws
        if final_cap != CAP_CEILING and not plan_absent:
            dec = decisions.get(LADDER[LADDER.index(final_cap) + 1])
            if dec is not None and set(dec["eligible_draws"]) != not_escalated:
                raise _PairRefusal(f"the not-escalated draws ({len(not_escalated)}) are not the plan's eligible draws at the stop ({len(dec['eligible_draws'])})", dec["record"])
        counts = merged_counts(final)
        for cap in chain:
            per_cap[cap]["draws_in_final"] = len(from_cap.get(cap, []))
        # -- the run
        instrument_id = str(inst_row["instrument_id"])
        producer_key = producer_key_for(tag, base["store_key"])
        rid = run_id_for(tag, base["store_key"], okey, instrument_id)
        n_scored, passes = counts["n_scored_draws"], counts["strict_passes"]
        value = (passes / n_scored) if n_scored else None
        loose = (counts["loose_passes"] / n_scored) if n_scored else None
        bound_lo = passes / N_DRAWS
        bound_hi = (passes + counts["n_unscored_draws"]) / N_DRAWS
        state = "final_at_base" if final_cap == CAP_BASE else ("final_at_ceiling" if final_cap == CAP_CEILING else "final_at_step")
        inst_numbers = {k: inst_row.get(f"decode_{k}") for k in DP.PARAM_KEYS}
        inst_match = "match" if self._numbers_agree(block, inst_numbers) else "mismatch"
        policy, policy_why = self._match_policy(model_id, block)
        sent = {k: v for k, v in block.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        base_dispatch = per_cap[CAP_BASE]["dispatch"]
        when_final = per_cap[final_cap]["generated_utc"]
        if when_final is None:
            raise _PairRefusal(f"the {final_cap:,} store's RESULT carries no parsable `at=` timestamp ({bundles[final_cap]['result_last'][:120]})", bundles[final_cap]["top"]["RESULT"].uri)
        when_start = IC.utc_from_iso(per_cap[CAP_BASE]["accepted_attempt"].get("started")) or base_dispatch.get("spawned_at") or when_final
        matrix_row = self._matrix_row(matrix, tag, okey)
        matrix_facts = self._check_matrix_row(matrix_row, tag, slug, obj, steered, chain, matrix_inp.uri.rsplit("#", 1)[0]) if matrix_row is not None else None
        geometry = man.get("geometry") if isinstance(man.get("geometry"), Mapping) else {}
        frameworks = man.get("frameworks") if isinstance(man.get("frameworks"), Mapping) else {}
        substrate = (SUBSTRATE, "derived", LC.confidence_entry("derived", f"manifest.json:usage_ledger on the Modal volume ({VOLUME_HEAD}…, held equal to the store's path at publish); RESULT's `{RESULT_HEAD}` head; "
                                                                          f"the dispatcher's record names Modal app {base_dispatch.get('app_id')} / function call {base_dispatch.get('function_call_id')}"))
        per_cap_block = {str(cap): {"store": pc["store"], "store_key": pc["store_key"], "value": pc["reading"]["acc"], "n_items": pc["reading"]["n_items"], "n_scored_items": pc["reading"]["n_scored_items"],
                                    "n_draws": pc["reading"]["n_draws"], "n_at_cap_draws": pc["reading"]["n_at_cap"], "share_at_cap": pc["reading"]["share_at_cap"],
                                    "bound": ((pc["reading"]["acc"] * pc["reading"]["n_scored_items"] / pc["reading"]["n_items"]) if pc["reading"]["acc"] is not None else None),
                                    "prompt_strict_mean_by_trial": pc["reading"]["prompt_strict_mean_by_trial"], "prompt_loose_mean_by_trial": pc["reading"]["prompt_loose_mean_by_trial"],
                                    "producer_row": pc["producer_row"], "producer_value": pc["producer_value"], "producer_bound": pc["producer_bound"], "producer_at_cap_n": pc["producer_at_cap_n"],
                                    "producer_record_agrees": True, "draws_in_final": pc["draws_in_final"], "n_loops_classified_here": len(loops.get(cap, ())),
                                    "loop_classification": classified_at.get(cap), "generated_utc": pc["generated_utc"]} for cap, pc in per_cap.items()}
        three_block = {"n_items": N_ITEMS, "k_trials": K_TRIALS, "n_draws": N_DRAWS, "n_scored_draws": n_scored, "strict_passes": passes, "loose_passes": counts["loose_passes"],
                       "acc_over_scored": value, "loose_over_scored": loose, "acc_over_attempted": bound_lo, "acc_over_attempted_upper_bound": bound_hi,
                       "inst_strict_micro": ((counts["inst_strict_followed"] / counts["n_instructions"]) if counts["n_instructions"] else None),
                       "inst_loose_micro": ((counts["inst_loose_followed"] / counts["n_instructions"]) if counts["n_instructions"] else None), "n_instructions": counts["n_instructions"],
                       "n_at_cap_draws_final": counts["n_at_cap_draws"], "share_at_cap_final": counts["n_at_cap_draws"] / N_DRAWS, "n_at_cap_draws_base": per_cap[CAP_BASE]["reading"]["n_at_cap"],
                       "n_loops": len(loop_keys), "n_ceiling": len(ceiling), "n_not_escalated": len(not_escalated), "n_unscored_draws": counts["n_unscored_draws"],
                       "n_at_cap_unscored": counts["n_at_cap_unscored"], "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_items_scored": counts["n_items_scored"],
                       "final_cap": final_cap, "chain": list(chain), "per_cap": per_cap_block, "merged_has_producer_record": False,
                       "producer_record": LC.rel_home(tn_inp.uri.rsplit("#", 1)[0]), "producer_record_schema": THREE_NUMBERS_SCHEMA, "producer_records_agree": True,
                       "producer_contract": three.get("contract"), "producer_seed_reading": three.get("seed_reading"),
                       "at_cap_rule": "the outcome record's at_cap (stop_reason == max_tokens at the request's cap; never re-tokenized); an at-cap draw is SCORED on IFEval (the checker fails what it did not complete)",
                       "unresolved_definition": "none on IFEval: every draw is scored; the at-cap draws at the final state are the third state's count, classified loop / ceiling / not_escalated"}
        merge_block = {"rule": MERGE_RULE, "algorithm": MERGE_ALGORITHM, "version": MERGE_VERSION, "chain": list(chain), "final_cap": final_cap, "escalation_state": state,
                       "draws_from_cap": {str(cap): len(ks) for cap, ks in from_cap.items()}, "replaced_draws": {str(cap): [list(k) for k in sorted(redrawn)] for cap, redrawn, _d in steps},
                       "loop_draws_by_cap": {str(cap): [list(k) for k in sorted(ks)] for cap, ks in loops.items()}, "loop_classification_by_cap": {str(c): t for c, t in classified_at.items()},
                       "not_escalated_draws": [list(k) for k in sorted(not_escalated)], "ceiling_draws": [list(k) for k in sorted(ceiling)], "subsets": {str(c): s for c, s in subset_facts.items()},
                       "plan_decisions": {str(c): d for c, d in decisions.items()}, "n_replaced_that_passed_before": sum(1 for d in final.values() if d["source_cap"] != CAP_BASE and d.get("replaced_value") == 1.0)}
        if plan_absent:
            merge_block.update(ladder_planned=False, plan_absent_rule=PLAN_ABSENT_RULE, unclassified_draws=[list(k) for k in sorted(unclassified)])
            three_block["n_unclassified"] = len(unclassified)
        extra: Dict[str, Any] = {
            "seed_replicate_label": selection, "fit_seed": (seed_n if steered and not arditi else None),
            "seed_axis": {"rule": SEED_AXIS_RULE, "trial_index": 0, "seed_base": None, "decode_seed_policy": INSPECT_SEED_POLICY_DEFAULT, "ifeval_seed_base_fixed": IFEVAL_SEED_BASE,
                          "same_object_on_every_seed_tag": (not steered) or arditi},
            "decode_seed_policy": INSPECT_SEED_POLICY_DEFAULT, "decode_seed_rule_text": seed_rule_text, "ifeval_seed_block": ife_seed, "final_cap": final_cap, "token_cap_ladder": list(LADDER),
            "escalation_state": state, "completion_cap_merge": merge_block, "store_uri": LC.rel_home(base["dir"]), "store_key": base["store_key"],
            "store_uri_by_cap": {str(c): pc["store"] for c, pc in per_cap.items()}, "store_key_by_cap": {str(c): pc["store_key"] for c, pc in per_cap.items()},
            "harness_config_hash_by_cap": {str(c): pc["harness_config_hash"] for c, pc in per_cap.items()}, "served_window_by_cap": {str(c): pc["served_window"] for c, pc in per_cap.items()},
            "served_max_num_seqs_by_cap": {str(c): pc["served_max_num_seqs"] for c, pc in per_cap.items()}, "producer_key_rule": PRODUCER_KEY_RULE, "group_rule": GROUP_RULE,
            "launch_set": {"record": LC.rel_home(ls_inp.uri.rsplit("#", 1)[0]), "tuple_id": lrow.get("tuple_id"), "class": lclass, "pin_key": launch_pin_key, "object_sha256_expected": expected_sha,
                           "block": lrow.get("block"), "table_index": lrow.get("table_index"), "census_status": lrow.get("census_status"), "configuration_ids": dict(cids),
                           **{k: lrow[k] for k in ("anchor_tuple", "anchor_fallback") if k in lrow}},
            "pins_record": LC.rel_home(pins_inp.uri.rsplit("#", 1)[0]), "matrix": matrix_facts, "dispatch": {str(c): pc["dispatch"] for c, pc in per_cap.items()}, "dispatch_rule": DISPATCH_RULE,
            "failed_dispatch_roots_never_read": facts.get("failed_dispatch_roots_never_read"), "standup": {str(c): pc["standup"] for c, pc in per_cap.items()}, "max_lora_rank_rule": MAX_LORA_RANK_RULE,
            "result_line": bundles[final_cap]["result_last"], "result_line_by_cap": {str(c): pc["result_line"] for c, pc in per_cap.items()},
            "attempts_before_accepted": {str(c): pc["attempts_before_accepted"] for c, pc in per_cap.items()}, "accepted_attempt_by_cap": {str(c): pc["accepted_attempt"] for c, pc in per_cap.items()},
            "served_window": LC.int_or_none(geometry.get("max_model_len")), "served_max_num_seqs": LC.int_or_none(geometry.get("max_num_seqs")), "geometry_source_legacy": geometry.get("source"),
            "harness_cell_id_legacy": man.get("harness_cell_id"), "three_numbers": three_block,
            "item_state_counts": {"strict_pass": passes, "strict_fail": n_scored - passes, ITEM_LABEL_AT_CAP_UNRESOLVED: counts["n_at_cap_unscored"], ITEM_LABEL_UNSCORED: counts["n_unscored_below_cap"],
                                  "scored_at_cap": counts["n_at_cap_draws"] - counts["n_at_cap_unscored"], "unit": "draws"},
            "at_cap_classes": {"loop": len(loop_keys), "ceiling": len(ceiling), "not_escalated": len(not_escalated), "unit": "draws", **({"unclassified": len(unclassified)} if plan_absent else {})},
            "served_model_id_hf": served_hf, "served_model_revision": rev or None, "serving_identity": ident, "serving_engine": man.get("serving_engine"), "serving_path_legacy": man.get("serving_path"),
            "served_tree_sha": tree, "object_sha16": (sha16 or None), "fit_model_id_hf": fitp.get("model_id"), "fit_transfer": fitp.get("transfer"), "fit_kind_legacy": fitp.get("fit_kind"),
            "fit_status_legacy": fitp.get("status"), "steer_positions_legacy": man.get("steer_positions"), "bank_view_legacy": (man.get("route") or {}).get("bank_view") if isinstance(man.get("route"), Mapping) else None,
            "producer_cell_id_legacy": okey, "producer_slug_legacy": slug, "producer_view_legacy": selection, "producer_tag_legacy": tag,
            "inspect_ai_version": frameworks.get("inspect_ai"), "inspect_evals_version": frameworks.get("inspect_evals"), "framework_legacy": base["gen_stats"].get("framework"),
            "client_lock_sha256": frameworks.get("client_lock_sha256"), "sampling_ref_legacy": base["gen_stats"].get("sampling_ref"), "ifeval_task_version": base["gen_stats"].get("ifeval_task_version"),
            "sampling_mode_legacy": (man.get("sampling") or {}).get("mode"), "sampling_source_legacy": (man.get("sampling") or {}).get("source"),
            "decode_policy_ref_legacy": (policy.policy_id if policy else None), "decode_policy_reader_match": ("exact by the stated sampling numbers" if policy else policy_why),
            "generation_state": "generated; rule-scored at generation (inspect_evals ifeval: the instruction checker, prompt-level strict as the headline), no judge", "source_legacy": SOURCE,
            "modal": {"function_call_id": base_dispatch.get("function_call_id"), "function_call_id_by_cap": {str(c): pc["dispatch"].get("function_call_id") for c, pc in per_cap.items()},
                      "app_id_by_cap": {str(c): pc["dispatch"].get("app_id") for c, pc in per_cap.items()}, "run_id_legacy": base_dispatch.get("run_id_legacy"), "usage_ledger": man.get("usage_ledger"),
                      "volume_head": VOLUME_HEAD}, "anchor_rule": ANCHOR_RULE, "secondary_metric": SECONDARY_METRIC,
        }
        if paired:
            extra.update(paired_of=f"s{seed_n}", seed_tag_legacy=group, roster_paired_of=roster_paired_of)
            if obj.object_kind == "none":
                # the paired tag's OWN anchor: a standup replicate of the seed tag's `none` at the same seed base, re-served (PAIRED_ANCHOR_RULE); no anchor block of its own
                extra.update(standup_replicate_of_view=group, anchor_rule=PAIRED_ANCHOR_RULE,
                             anchor_role=f"the paired tag's OWN anchor: a standup replicate of {group}/none (D429); the composed row pairs against it once it files")
            else:
                # the fallback's spelling; _emit_contrasts re-stamps the row once its own `none` filed (extra.anchor names the anchor used either way)
                extra["cross_standup_anchor"] = f"{group}/none"
        declared_role = None
        lead = f"IFEval thinking OFF, k = {K_TRIALS}, completion-cap merge over {' -> '.join(f'{c:,}' for c in chain)} ({state}), seed tag {tag} {okey}"
        if arditi:
            declared_role = ARDITI_ROLE
            extra.update(fit_seed=None, declared_role=declared_role,
                         single_object={"object_key": okey, "staged_id": str(obj.staged_id), "object_id": obj.object_id, "operating_point": op.rung, "standup_legacy": op.standup_tag,
                                        "c": float(op.coefficient), "unit": op.unit, "declared_role": declared_role, "seed_axis_rule": SEED_AXIS_RULE,
                                        "coexist_ruling_needed": (f"a roster coexist ruling over lane_sources_legacy [{SOURCE}] x campaigns [{CAMPAIGN}] x model_ids [{model_id}] x object_ids "
                                                                  f"[{obj.object_id}] x eval_ids [{EVAL_ID}] x instrument_ids [{instrument_id}] x views_legacy "
                                                                  f"[{', '.join(f'{MODEL_KEY}_s{k}' for k in (1, 2, 3))}] keeps the three rows selected together (the user's, never a reader default)")})
            lead += f" (ONE object on every seed tag, {op.rung} c = {op.coefficient:g} {op.unit}; declared {declared_role})"
        elif not steered:
            extra["same_object_rows"] = {"twin_tags_legacy": [f"{MODEL_KEY}_s{k}" for k in (1, 2, 3) if k != seed_n], "rule": SEED_AXIS_RULE,
                                         "coexist_ruling_needed": f"a roster coexist ruling over lane_sources_legacy [{SOURCE}] x the seed tags keeps the rows of one object selected together"}
            if paired:
                extra["same_object_rows"]["standup_replicate_of_view"] = group
                extra["same_object_rows"]["coexist_ruling_needed"] += " and the paired tag (the same seed base, re-served: r2v4-coexist-<lane>-paired-view-none-anchors)"
        notes = (f"{lead}: {passes}/{n_scored} strict passes over the scored draws = {value:.4f}" if value is not None else f"{lead}: no scored draw")
        notes += (f" (loose {loose:.4f})" if loose is not None else "")
        notes += (f"; {counts['n_at_cap_draws']}/{N_DRAWS} draws at the cap at the final state ({len(loop_keys)} loops, {len(ceiling)} at the {CAP_CEILING:,} ceiling, "
                  f"{len(not_escalated)} truncated without a step; {per_cap[CAP_BASE]['reading']['n_at_cap']} at the {CAP_BASE:,} base)")
        notes += f"; passes/{N_DRAWS} = {bound_lo:.4f} is the BOUND, not the value"
        try:
            payload = LC.run_payload_base(seeds, run_id=rid, producer_key=producer_key, object_key=okey, view_legacy=tag, cell_id_legacy=okey, obj=obj, eval_id=EVAL_ID,
                                          instrument_row=inst_row, instrument_seed_id=INSTRUMENT_SEED, lifecycle_state="generated", when_utc=when_final,
                                          when_source=f"the final store's RESULT ({RESULT_COMPLETE}) `at=` for generated; the base store's manifest.json:resume_attempts[completed].started for queued / started",
                                          token_cap_tokens=final_cap, judge_id=None, judge_seed_id=None, substrate=substrate, lane_source_legacy=SOURCE,
                                          store_uri=LC.rel_home(base["dir"]), instrument_match=inst_match, extra=extra, notes=notes)
        except KeyError as exc:
            raise _PairRefusal(f"run row: {exc}", where_man)
        fc = payload["field_confidence_json"]
        payload.update(producer_session=PRODUCER_SESSION, trial_index=0, seed_base=None, queued_utc=when_start, started_utc=when_start,
                       n_items_planned=N_ITEMS, n_items_generated=N_ITEMS, n_items_judged=counts["n_items_scored"], model_revision=(rev if _SHA1.match(rev) else None),
                       intervention_hash=man.get("intervention_hash"), harness_config_hash=man.get("harness_config_hash"),
                       env_digest=(next(iter(base["consts"].get("env_digest") or []), None)), modal_call_id=base_dispatch.get("function_call_id"), job_name=base["result"].get("name"),
                       node_hostname=(per_cap[CAP_BASE]["standup"].get("node") or None), serving_identity=ident, token_cap_rung_index=LADDER.index(final_cap), token_cap_ladder_json=list(LADDER),
                       decode_seed_rule=DECODE_SEED_RULE, decode_sent_json=sent, decode_temperature_sent=sent.get("temperature"), decode_top_p_sent=sent.get("top_p"),
                       decode_top_k_sent=(int(sent["top_k"]) if sent.get("top_k") is not None else None), code_sha=tree,
                       harness_version=(f"inspect-ai {frameworks.get('inspect_ai')} / inspect_evals {frameworks.get('inspect_evals')} ({base['gen_stats'].get('framework')})"
                                        if frameworks.get("inspect_ai") else payload["harness_version"]),
                       context_window_tokens=LC.int_or_none(geometry.get("max_model_len")),
                       store_completion_artifact_uri=bundles[final_cap]["files"]["gen_stats.json"].uri, store_completion_sha256=bundles[final_cap]["files"]["gen_stats.json"].sha256)
        if LC.int_or_none(geometry.get("max_num_seqs")) is not None:
            payload.update(served_max_num_seqs=LC.int_or_none(geometry.get("max_num_seqs")), served_max_num_seqs_source="attested")
            fc["served_max_num_seqs"] = LC.confidence_entry("recorded", f"manifest.json:geometry.max_num_seqs (source {geometry.get('source')}; the base store's — the steps' ride extra.served_max_num_seqs_by_cap)")
        if policy is not None:
            payload["decode_source"] = policy.decode_source
            fc["decode_source"] = LC.confidence_entry("recorded", f"the registered policy row {policy.policy_id} the sent numbers equal (its decode_source)")
        else:
            fc["decode_source"] = LC.unverified(policy_why)
        fc["code_sha"] = LC.confidence_entry("recorded", "manifest.json:producer.tree_sha = RESULT's sha = the outcome record's producer_tree_sha = the dispatcher record's rec.tree_sha, on every store of the chain")
        fc["model_revision"] = LC.confidence_entry("recorded", "manifest.json:model_revision (= models.yaml hf_revision)")
        fc["modal_call_id"] = (LC.confidence_entry("recorded", f"{base_dispatch['record']}:fc_id (the dispatcher's Modal function call for the base store; the steps' ride extra.modal)")
                               if base_dispatch.get("function_call_id") else LC.unverified("the dispatcher's record names no function call id"))
        fc["node_hostname"] = (LC.confidence_entry("recorded", f"{GEOMETRY_FILE}:endpoint.node (the base standup's attested endpoint)") if per_cap[CAP_BASE]["standup"].get("node")
                               else LC.unverified(f"{GEOMETRY_FILE}:endpoint names no node (a Modal container)"))
        fc["token_cap_tokens"] = LC.confidence_entry("recorded", f"the ladder's final state: {final_cap} ({state}); gen_stats.json:max_tokens = the outcome record's max_tokens of the final store; the ladder's deciding records agree")
        fc["seed_base"] = LC.confidence_entry("derived", SEED_AXIS_RULE)
        fc["trial_index"] = LC.confidence_entry("derived", SEED_AXIS_RULE)
        fc["decode_seed_rule"] = LC.confidence_entry("derived", f"one seed per draw under IFEval's fixed-base rule: {seed_rule_text} (seed_base {IFEVAL_SEED_BASE} is the instrument's, not a per-seed base; extra.ifeval_seed_block)")
        fc["decode_sent_json"] = LC.confidence_entry("recorded", "gen_stats.json (the client's sampler; manifest.json:sampling agrees on every store of the chain; the stores' resolved_sampler.json are registered beside them)")
        fc["instrument_match"] = LC.confidence_entry("derived", f"the sent numbers compared to instrument seed {INSTRUMENT_SEED}'s decode columns (bound to {model_id}): {inst_match}")
        fc["harness_version"] = LC.confidence_entry("recorded", "manifest.json:frameworks (inspect_ai / inspect_evals) and gen_stats.json:framework")
        fc["context_window_tokens"] = LC.confidence_entry("recorded", "manifest.json:geometry.max_model_len (the base store's; the steps' ride extra.served_window_by_cap)")
        fc["n_items_planned"] = LC.confidence_entry("derived", f"the instrument's item set (n_items {N_ITEMS}); gen_stats.json:n and the outcome record's n_attempted_items agree ({K_TRIALS} draws each)")
        fc["n_items_generated"] = LC.confidence_entry("recorded", "the base outcome record: every attempted draw has a stop reason; done/<task>.json n_draws agrees")
        fc["n_items_judged"] = LC.confidence_entry("derived", "the items with a SCORED draw at the final state (rule-scored at generation; an at-cap draw is scored on IFEval)")
        fc["queued_utc"] = fc["started_utc"] = LC.confidence_entry("recorded", "manifest.json:resume_attempts — the base store's completed attempt's started")
        fc["generated_utc"] = LC.confidence_entry("recorded", f"the final store's RESULT last line ({RESULT_COMPLETE}): its `at=` timestamp")
        fc["thinking_mode"] = LC.confidence_entry("recorded", f"gen_stats.json:thinking = manifest.json:thinking.by_task[{TASK}] = the outcome record's thinking = off, on every store of the chain")
        fc["substrate"] = substrate[2]
        if steered and man.get("steer_positions") in V.APPLIED_POSITIONS_MODES:
            payload["applied_positions_mode"] = str(man["steer_positions"])
            fc["applied_positions_mode"] = LC.confidence_entry("recorded", "manifest.json:steer_positions")
        if not steered:
            payload["strength_unit"] = "unverified"
            fc["strength_unit"] = LC.unverified("an unsteered / prompted object key: no object on the served path, so no dose and no unit (the column default)")
        if arditi:
            payload.update(op.strength_columns())
            fc.update(op.strength_confidence(f"the launch set's pin key {op.staged_id} (the all-positions pins' staged object at the operating point)"))
            payload["extra"].update(op.extra())
            op_mv = seeds.model_for_view_legacy(op.standup_tag)
            if op_mv.get("transfer_kind") in V.TRANSFER_KINDS:
                payload["transfer_kind"] = op_mv["transfer_kind"]
                fc["transfer_kind"] = LC.confidence_entry("recorded", f"legacy_spellings.yaml model_keys[{op.standup_tag}].transfer_kind (the direction is fit on and served on the instruct "
                                                                      f"checkpoint: manifest.json:fit_provenance.transfer {fitp.get('transfer')!r})")
        records.append(("run", payload))
        facts["runs"] += 1
        # NO state event: the record's `generated` is the state and the harvester's rule-scored completion stamps `ingested`
        comp_id = merge_computation_id()
        code_sha = str(producer.get("code_sha") or "")
        records.append(("computation", {"computation_id": comp_id, "algorithm": MERGE_ALGORITHM, "version": MERGE_VERSION, "seed_provenance": MERGE_SEED_PROVENANCE,
                                        "code_sha": (code_sha if _SHA1.match(code_sha) else "unverified"),
                                        "notes": "the draw-wise completion-cap merge: each cap step's re-drawn draws replace their truncated shallower draws (a loop is never re-drawn); the "
                                                 f"prompt-level strict and loose pass rates are recounted over the merged {N_DRAWS} draws"}))
        rows_uri = LC.rel_home(bundles[final_cap]["files"]["rows.jsonl"].uri)
        lo, hi = IC.wilson(passes, n_scored) if n_scored else (None, None)
        records.append(("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": comp_id, "producer_metric_spelling": "value",
                                       "value": value, "k": passes, "n": n_scored, "ci_lo": lo, "ci_hi": hi, "ci_method": ("wilson_95" if n_scored else None), "n_abstentions": None,
                                       "n_incomplete": counts["n_unscored_draws"], "computed_by": "registry", "computed_utc": when_final, "source_uri": rows_uri,
                                       "extra": {"role": "THE VALUE: strict prompt-level passes over the SCORED draws at the ladder's final state (the user's rule: degeneration is scored separately)",
                                                 "unit": "draws", "recounted_from": "the base outcome record with each cap step's re-drawn draws merged draw-wise", "equals_gen_stats_acc": (len(chain) == 1),
                                                 "base_reading": per_cap_block[str(CAP_BASE)]["value"], "final_cap": final_cap, "n_at_cap_draws_final": counts["n_at_cap_draws"],
                                                 "producer_record": three_block["producer_record"], "merged_has_producer_record": False, "producer_value_per_cap": {c: p["producer_value"] for c, p in per_cap_block.items()}}}))
        records.append(("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": "", "producer_metric_spelling": BOUND_SPELLING,
                                       "value": bound_lo, "k": passes, "n": N_DRAWS, "ci_lo": bound_lo, "ci_hi": bound_hi, "ci_method": BOUND_CI_METHOD, "n_abstentions": None,
                                       "n_incomplete": counts["n_unscored_draws"], "computed_by": "producer", "computed_utc": when_final, "source_uri": three_block["producer_record"],
                                       "extra": {"role": f"THE BOUND, never the value: strict passes over ALL {N_DRAWS} draws ({N_ITEMS} x {K_TRIALS}); the interval's lower end counts an unscored draw failed, the upper end passed",
                                                 "unit": "draws", "n_unscored_draws": counts["n_unscored_draws"], "unresolved_definition": three_block["unresolved_definition"],
                                                 "producer_spellings": {f"{BOUND_SPELLING}@{c}": p["producer_bound"] for c, p in per_cap_block.items()},
                                                 "producer_formula": "the producer's per-store spelling `bound = acc x n_scored / n` (three_numbers contract), applied here to the merged draws; the merged number has no producer record",
                                                 "metric_role_expected": "secondary (a blank computation_id ranks after the value's computed row at one identity)"}}))
        lo_l, hi_l = IC.wilson(counts["loose_passes"], n_scored) if n_scored else (None, None)
        records.append(("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": SECONDARY_METRIC, "threshold": -1.0, "computation_id": comp_id, "producer_metric_spelling": "prompt_loose_mean",
                                       "value": loose, "k": counts["loose_passes"], "n": n_scored, "ci_lo": lo_l, "ci_hi": hi_l, "ci_method": ("wilson_95" if n_scored else None), "n_abstentions": None,
                                       "n_incomplete": counts["n_unscored_draws"], "computed_by": "registry", "computed_utc": when_final, "source_uri": rows_uri,
                                       "extra": {"role": "SECONDARY: loose prompt-level passes over the scored draws at the final state (the headline is prompt_level_strict)", "unit": "draws",
                                                 "inst_strict_micro": three_block["inst_strict_micro"], "inst_loose_micro": three_block["inst_loose_micro"], "n_instructions": counts["n_instructions"]}}))
        # -- items (whole rows): one item_result per DRAW (sample_index = the trial), value 1 / 0 (strict), or NULL with the state's label
        for k in sorted(final):
            d = final[k]
            iid, trial = k
            cap = d["source_cap"]
            b = bundles[cap]
            row = b["draw_rows"].get(k)
            created = IC.utc_from_epoch(row[2]) if (row is not None and row[2] is not None) else None
            if d["value"] is not None:
                label, cls = None, d.get("at_cap_class")
            elif d.get("at_cap"):
                label, cls = ITEM_LABEL_AT_CAP_UNRESOLVED, d.get("at_cap_class")
            else:
                label, cls = ITEM_LABEL_UNSCORED, UNSCORED_BELOW_CAP_CLASS
            iextra = {"final_cap": cap, "source_store": f"cap{cap}", "stop_reason": d["stop_reason"], "completion_tokens": d["completion_tokens"], "decode_seed": d["seed"],
                      "at_cap": bool(d.get("at_cap")), "at_cap_class": (cls if cls in AT_CAP_CLASSES else None), "scored_at_cap": bool(d["value"] is not None and d.get("at_cap")),
                      "unscored_reason": (UNSCORED_BELOW_CAP_CLASS if cls == UNSCORED_BELOW_CAP_CLASS else None), "prompt_level_loose": d.get("prompt_level_loose"),
                      "inst_level_strict": d.get("inst_level_strict"), "inst_level_loose": d.get("inst_level_loose"), "num_instructions": d.get("num_instructions"),
                      "replaced_value": d.get("replaced_value"), "replaced_at_cap": d.get("replaced_at_cap"), "unparseable": d.get("unparseable")}
            records.append(("item_result", IC.item_result_payload(rid, iid, HEADLINE_METRIC, d["value"], label, created or when_final, judgment_id="", trial_index=0, sample_index=trial, extra=iextra)))
            records.append(("item_flags", IC.item_flags_payload(rid, iid, (b["files"]["rows.jsonl"].sha256 if row is not None else None), (row[1] if row is not None else None),
                                                                at_token_cap=(1 if d.get("at_cap") else 0), finish_reason=d["stop_reason"], response_tokens=d["completion_tokens"],
                                                                decode_seed=(str(d["seed"]) if d["seed"] is not None else None), judgment_id="", trial_index=0, sample_index=trial,
                                                                degenerate_rep_ratio_loop=((1 if k in loop_keys else 0) if (d.get("at_cap") and cls not in ("ceiling", "unclassified")) else None),
                                                                extra={"at_token_cap_source": three_block["at_cap_rule"], "generation_pointer": f"rows.jsonl of the {cap:,} store (sha, line): the draw's {HEADLINE_ROW_METRIC}@{trial} row",
                                                                       "outcome_record": LC.rel_home(b["files"]["outcomes"].uri), "loop_source": classified_at.get(cap)})))
        # -- the run-level shares (run_degeneration over the 1,082 draws at the final state)
        src_uri = LC.rel_home(bundles[final_cap]["files"]["outcomes"].uri)
        measured = {
            RAW_AT_CAP_MEASURE: dict(numerator=counts["n_at_cap_draws"], denominator=N_DRAWS, unit="draws", threshold_params={"cap_final": final_cap, "ladder": list(LADDER)}, computed_by="registry",
                                     extra={"rule": three_block["at_cap_rule"], "at_ladder_final_state": True,
                                            "classes": {"loop": len(loop_keys), "ceiling": len(ceiling), "not_escalated": len(not_escalated), **({"unclassified": len(unclassified)} if plan_absent else {})},
                                            "n_at_cap_draws_base": per_cap[CAP_BASE]["reading"]["n_at_cap"], "per_cap_store_at_cap": {c: p["n_at_cap_draws"] for c, p in per_cap_block.items()}, "source": src_uri}),
        }
        if not plan_absent:
            measured[LOOP_MEASURE] = dict(numerator=len(loop_keys), denominator=N_DRAWS, unit="draws", threshold_params={"rep_ratio_gt": 0.5, "tail_tokens": 200, "caps_classified": sorted(loops)}, computed_by="registry",
                               extra={"class": "loop: at the cap AND degenerate by the ladder's rep-ratio rule (cap_ladder.json where the ladder promoted from a store; the plan's at-cap minus eligible where it stopped) — never re-drawn",
                                      "loop_classification_by_cap": {str(c): t for c, t in classified_at.items()}, "n_unclassified_at_ceiling": len(ceiling),
                                      "ceiling_note": (f"{len(ceiling)} draw(s) at the {CAP_CEILING:,} ceiling carry no loop classification (the ladder has no step above it) and are not counted here" if ceiling else None),
                                      "source": src_uri})
        notes_nm = {"classifier": "the paper's IFEval reader files no classifier count; this reader carries the ladder's rep-ratio rule (rep_ratio_loop)",
                    "contentless": "no contentless reading in the ladder's classification", "zlib_collapse": "no zlib reading in the ladder's classification",
                    "union5": "the five-detector union is not computed for these stores (the degeneracy-union reader's input, not here)",
                    IC.AT_CAP_RETOKENIZED: "the outcome record reads the provider's stop reason, never a re-tokenized length"}
        if plan_absent:
            notes_nm[LOOP_MEASURE] = (f"the ladder never planned this row (PLAN_ABSENT_RULE): no rep-ratio classification exists for its {len(unclassified)} at-cap draw(s), which file "
                                      "at_cap_class `unclassified`; the raw at-cap share stands")
        for drow in IC.declared_degeneration_rows(rid, "", measured, source_uri=src_uri, computed_utc=when_final, not_measured_notes=notes_nm):
            records.append(("run_degeneration", drow))
        if plan_absent:
            records.append(("run_degeneration", IC.degeneration_payload(rid, COHERENT_MEASURE, "not_measured", judgment_id="", threshold_params={"cap_final": final_cap}, computed_by="registry",
                                                                        computed_utc=when_final, source_uri=src_uri,
                                                                        notes=f"the ladder never planned this row (PLAN_ABSENT_RULE): the loop / truncation split of its {len(unclassified)} at-cap draw(s) does not exist",
                                                                        extra={"class": "at the cap and not a classified loop: still at cap at the ceiling, or truncated below it without a step",
                                                                               "classes": {"ceiling": len(ceiling), "not_escalated": len(not_escalated), "unclassified": len(unclassified)}, "source": src_uri})))
        else:
            records.append(("run_degeneration", IC.degeneration_payload(rid, COHERENT_MEASURE, "measured", judgment_id="", numerator=len(ceiling) + len(not_escalated), denominator=N_DRAWS, unit="draws",
                                                                        threshold_params={"cap_final": final_cap}, computed_by="registry", computed_utc=when_final, source_uri=src_uri,
                                                                        extra={"class": "at the cap and not a classified loop: still at cap at the ceiling, or truncated below it without a step",
                                                                               "classes": {"ceiling": len(ceiling), "not_escalated": len(not_escalated)}, "source": src_uri})))
        facts["rows"][f"{tag}|{okey}"] = {"run_id": rid, "tag": tag, "selection": selection, "paper_label": obj.paper_label, "object_id": obj.object_id, "staged_id": obj.staged_id,
                                          "instrument_id": instrument_id, "final_cap": final_cap, "chain": list(chain), "escalation_state": state, "value": value, "loose": loose, "strict_passes": passes,
                                          "n_scored_draws": n_scored, "bound": bound_lo, "upper_bound": bound_hi, "n_at_cap_draws_final": counts["n_at_cap_draws"], "share_at_cap_final": counts["n_at_cap_draws"] / N_DRAWS,
                                          "n_loops": len(loop_keys), "n_ceiling": len(ceiling), "n_not_escalated": len(not_escalated), "n_unclassified": len(unclassified), "ladder_planned": (not plan_absent),
                                          "draws_from_cap": merge_block["draws_from_cap"],
                                          "per_cap": {c: {"value": p["value"], "producer_value": p["producer_value"], "n_at_cap_draws": p["n_at_cap_draws"], "producer_at_cap_n": p["producer_at_cap_n"], "n_draws": p["n_draws"]}
                                                      for c, p in per_cap_block.items()},
                                          "dispatch": {c: {"verdict": p["dispatch"]["verdict"], "state": p["dispatch"]["state"], "attested": p["dispatch"]["serving_path_attested"], "rate_limited_read": p["dispatch"]["rate_limited_read"]}
                                                       for c, p in per_cap.items()},
                                          "plan_exits": {c: d["exit"] for c, d in decisions.items()}, "instrument_match": inst_match, "decode_policy": (policy.policy_id if policy else None),
                                          "tree": tree, "store_key": base["store_key"], "producer_key": producer_key, "paired": paired, "steered": steered, "declared_role": declared_role,
                                          "max_lora_rank_recorded": {c: p["standup"]["max_lora_rank"] for c, p in per_cap.items()}, "modal_call_id": base_dispatch.get("function_call_id")}
        return {"run_id": rid, "draws": final, "steered": steered, "when": when_final, "declared_role": declared_role, "tag": tag, "paired": paired, "okey": okey, "final_cap": final_cap,
                "payload": payload, "records": records, "plan_absent": plan_absent}

    # ---- contrasts (PAIRED_ANCHOR_RULE) ------------------------------------------------------
    @staticmethod
    def _value_row(st: Mapping[str, Any], comp_id: str) -> Optional[Dict[str, Any]]:
        """The row's VALUE reading among its built records (the merged recount the contrast is keyed on): a park lands on its ``extra.delta_block``."""
        for rt, payload in st["records"]:
            if rt == "run_metric" and payload.get("metric_id") == HEADLINE_METRIC and payload.get("computation_id") == comp_id:
                return payload
        return None

    @staticmethod
    def _anchor_key(st: Mapping[str, Any], own_run_id: Optional[str], own_kind: Optional[str]) -> AC.AnchorKey:
        """The steered row's exact key for the pass's anchor currency, from the run payload the row built: the standup is the
        registry spelling of its tag under this source (``view_legacy``, ``lane_source_legacy``)."""
        p = st["payload"]
        return AC.AnchorKey(model_id=str(p["model_id"]), eval_id=str(p["eval_id"]), instrument_id=str(p["instrument_id"]),
                            token_cap_tokens=LC.int_or_none(p.get("token_cap_tokens")), trial_index=LC.int_or_none(p.get("trial_index")),
                            pass_label=p.get("pass_label"), decode_policy_id=p.get("decode_policy_id"), lane_source_legacy=p.get("lane_source_legacy"),
                            standup={"view_legacy": p.get("view_legacy"), "lane_source_legacy": p.get("lane_source_legacy")},
                            own_anchor_run_id=own_run_id, own_anchor_kind=(own_kind if own_run_id is not None else None))

    def _emit_contrasts(self, ctx: AdapterContext, group: str, filed: Mapping[Tuple[str, str], Mapping[str, Any]], expected: Mapping[Tuple[str, str], Mapping[str, Any]],
                        seen: Set[Tuple[str, str]], facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """One ``run_contrast`` per steered / prompted run of the group, DECIDED by the pass's anchor currency (:data:`CURRENCY_RULE`).
        The row's OWN pairing is the plan's (:func:`_bcbw_common.anchor_plan`: its own tag's ``none`` at the same final state,
        the paired tag's own ``none`` once it files, else the fallback across tags): it is the own anchor the currency is asked about and it stands
        while the frame holds it current with a usable reading -- paired per DRAW (item, trial) over the draws scored on both sides (the strict pass-rate
        difference in pp, McNemar's exact p on the discordant pairs, Cohen's h), plus ``extra.anchor_selection``. A retired, unfolded or absent own anchor
        searches the exact key: a re-picked unsteered run that is a row of this batch is paired per draw like the own one, any other files the registry's
        UNPAIRED rate difference from the two readings' k / n (``computed_by registry``, its own computation row named by
        ``extra.statistic_computation_id``). The kind :func:`_bcbw_common.anchor_kind_for` yields is checked against the currency's on every filed
        contrast (a disagreement refuses the group by name). No current anchor at the exact key parks the comparison by name -- the VALUE row's
        ``extra.delta_block``, a refusal line -- and the previous revision's key is withdrawn by the head. Stamps every row's ``extra.anchor`` block on
        its (not yet added) run payload and returns the contrast payloads."""
        out: List[Dict[str, Any]] = []
        comp_id = merge_computation_id()
        frame: Optional[AC.AnchorCurrency] = None
        by_run = {st_["run_id"]: st_ for st_ in filed.values()}
        for key in sorted(k for k in filed if k[1] != "none"):
            tag, okey = key
            st = filed[key]
            plan = anchor_plan(group, tag, filed, expected, seen)
            anchor = plan["anchor"]
            block = {"rule": ANCHOR_RULE, "anchor_view_legacy": (plan["anchor_tag"] or group), "anchor_object_key": "none", "cross_standup": plan["cross_standup"],
                     "declared_role": st.get("declared_role"), "pair_unit": "draw (item, trial)", "row_final_cap": st.get("final_cap"), "currency_rule": CURRENCY_RULE}
            if plan["own_none_state"] is not None:
                block.update(own_none_state=plan["own_none_state"], own_none_note=plan["own_none_note"])
            if plan["own"] and st.get("paired"):
                # the paired tag's row against ITS OWN `none`: the run's spelling moves from the fallback to the own anchor
                block.update(rule=PAIRED_ANCHOR_RULE, own_standup_replicate_of_view=group)
                st["payload"]["extra"].pop("cross_standup_anchor", None)
                st["payload"]["extra"].update(own_anchor=f"{tag}/none", standup_replicate_of_view=group, anchor_rule=PAIRED_ANCHOR_RULE)
            # ---- the decision (CURRENCY_RULE): the plan's anchor is the OWN anchor the pass's frame is asked about; the frame loads on the first comparison
            if frame is None:
                frame = ctx.anchor_currency()
            own_id = anchor["run_id"] if anchor is not None else None
            own_kind = B.anchor_kind_for(plan) if anchor is not None else None
            akey = self._anchor_key(st, own_id, own_kind)
            choice = frame.choose(akey, HEADLINE_METRIC, -1.0, None, computation_id=comp_id, steered_run_id=st["run_id"], steered_judgment_id="")
            selection = choice.selection_json()
            facts["anchor_states"][choice.state] = facts["anchor_states"].get(choice.state, 0) + 1
            value_row = self._value_row(st, comp_id)
            k_s, n_s = ((value_row or {}).get("k"), (value_row or {}).get("n"))
            own_text = (f"{own_id} ({own_kind}; {plan['anchor_tag']}/none)" if own_id is not None else f"none declared ({plan['state']}: {plan['why']})")
            key_text = f"key (cap {akey.token_cap_tokens}, trial {akey.trial_index}, pass {akey.pass_label}, instrument {akey.instrument_id})"
            fact = {"anchor": None, "state": plan["state"], "why": plan["why"], "own_none_state": plan["own_none_state"], "currency_state": choice.state,
                    "currency_own_reason": choice.own_reason, "own_anchor_run_id": own_id, "n_candidates": choice.n_candidates, "row_final_cap": st.get("final_cap")}

            def park(why: str, refusal: str) -> None:
                if value_row is not None:
                    value_row.setdefault("extra", {})["delta_block"] = dict(
                        {f: None for f in ("delta", "delta_lo", "delta_hi", "delta_p", "effect_size", "effect_kind")}, n_pairs=0,
                        delta_method=f"none: {why}", computed_by="registry", why_not_a_contrast_row=why,
                        anchor=dict(own_anchor_run_id=own_id, own_anchor_kind=own_kind, state=choice.state, own_reason=choice.own_reason, anchor_run_id=choice.run_id,
                                    anchor_kind=choice.anchor_kind, registry_contrast_filed_against=None),
                        anchor_selection=selection, producer_block={"producer_delta_value_vs_none": (st["payload"]["extra"].get("three_numbers") or {}).get("producer_delta_value_vs_none"),
                                                                    "note": "the producer's three-number record's delta is RECORDED on the run, never a contrast row"},
                        rule=AC.RULE)
                facts["contrasts_parked"] += 1
                self.refusals.add(f"{tag}|{okey}|contrast", refusal, st["payload"]["extra"].get("store_uri") or self.root)

            if choice.state == "no_current_anchor":
                # no current anchor at the exact key: the plan's own state (a `none` not yet run / refused) keeps its spelling on the run; the currency's reason beside it
                reason = plan["why"] if plan["why"] else choice.why
                block.update(state=(plan["state"] or "no_current_anchor"), reason=reason, anchor_run_id=None, currency_state=choice.state, currency_own_reason=choice.own_reason,
                             currency_why=choice.why, anchor_selection=selection)
                st["payload"]["extra"]["anchor"] = block
                facts["contrasts"][f"{tag}|{okey}"] = dict(fact, state=(plan["state"] or "no_current_anchor"), why=reason, currency_why=choice.why)
                park(choice.why, f"contrast has no current anchor at {key_text} [own anchor {own_text}: {choice.own_reason}]; filed without a contrast: {choice.why}")
                continue
            assert choice.run_id is not None and choice.measurement is not None
            reanchored = choice.state != "own_current"
            same_standup = False
            tier_key: Optional[str] = None
            cand = None
            if reanchored:
                # a RE-PICKED anchor (the own one retired, unfolded or absent), by the currency's tier: the row's own standup's unsteered run the registry
                # holds current from an earlier read; the same model's unsteered run of another standup under this source; the paper's roster-matched baseline of
                # another source. The kind flows through the plan (same_standup_current / sibling_current / borrowed_reference, anchor_kind_for) and is checked below
                cand = next((c for c in choice.candidates if c.run_id == choice.run_id), None)
                same_standup = choice.anchor_kind == AC.KIND_SAME_STANDUP
                tier_key = "borrowed_reference" if choice.state == "borrowed_reference" else ("same_standup_current" if same_standup else "sibling_current")
                plan = dict(plan, cross_standup=not same_standup, anchor_tag=(cand.view_legacy if cand is not None else None), **{tier_key: True})
                anchor = by_run.get(choice.run_id)          # a row of THIS batch (another tag's `none`): paired per draw like the own one; else unpaired from k / n
                block.update(anchor_view_legacy=plan["anchor_tag"], cross_standup=plan["cross_standup"], reanchored=True, reanchor_tier=tier_key, own_anchor_run_id=own_id,
                             own_anchor_state=(plan["state"] or "retired"),
                             anchor_standup={"lane_source_legacy": cand.lane_source_legacy, "view_legacy": cand.view_legacy, "cluster_job_id": cand.cluster_job_id} if cand is not None else None)
            if B.anchor_kind_for(plan) != choice.anchor_kind:
                raise AdapterRefusal(self.root, f"{tag}/{okey}: the contrast's kind by anchor_kind_for ({B.anchor_kind_for(plan)}) disagrees with the anchor currency's "
                                                f"({choice.anchor_kind}; state {choice.state}, anchor {choice.run_id}); refused rather than filed under either label")
            block.update(currency_state=choice.state, currency_own_reason=choice.own_reason, anchor_selection=selection)
            m = choice.measurement
            retired = f"own anchor {own_text}: {choice.own_reason}" + (f" ({choice.own_why})" if choice.own_why else "")
            chosen = f"{choice.run_id} ({choice.anchor_kind}" + (", another source" if choice.cross_source else "") + ")"
            common_extra: Dict[str, Any] = {"unit": "pp", "pair_unit": "draw (item, trial)", "anchor_view_legacy": plan["anchor_tag"], "anchor_object_key": "none", "anchor_rule": ANCHOR_RULE,
                                            "row_final_cap": st.get("final_cap"), "cross_standup": plan["cross_standup"], "row_tag_legacy": st.get("tag"),
                                            "anchor_selection": selection, "currency_rule": CURRENCY_RULE}
            if anchor is not None:
                # the anchor is a row of this batch: paired per DRAW over the draws scored on both sides
                a_d, s_d = anchor["draws"], st["draws"]
                shared = sorted(k for k in s_d if k in a_d and s_d[k]["value"] is not None and a_d[k]["value"] is not None)
                block["anchor_final_cap"] = anchor.get("final_cap")
                if not shared:
                    why = (f"no draw scored on both sides (anchor {chosen})" if reanchored else "no draw scored on both sides")
                    block.update(state="no_shared_items", reason=why, anchor_run_id=anchor["run_id"])
                    st["payload"]["extra"]["anchor"] = block
                    facts["contrasts"][f"{tag}|{okey}"] = dict(fact, anchor=anchor["run_id"], state="no_shared_items", why=why)
                    park(why, f"contrast's anchor {chosen} is current but shares no scored draw with the row at {key_text}; filed without a contrast")
                    continue
                b = sum(1 for k in shared if s_d[k]["value"] == 1.0 and a_d[k]["value"] == 0.0)
                c = sum(1 for k in shared if s_d[k]["value"] == 0.0 and a_d[k]["value"] == 1.0)
                p_s = sum(s_d[k]["value"] for k in shared) / len(shared)
                p_a = sum(a_d[k]["value"] for k in shared) / len(shared)
                stat = {"delta": 100.0 * (p_s - p_a), "delta_p": DC.mcnemar_exact(b, c), "delta_method": CONTRAST_METHOD, "effect_size": DC.cohen_h(p_s, p_a), "effect_kind": "cohens_h",
                        "n_pairs": len(shared)}
                stat_extra = {"discordant_steered_only": b, "discordant_anchor_only": c, "acc_steered_over_pairs": p_s, "acc_anchor_over_pairs": p_a, "anchor_final_cap": anchor.get("final_cap"),
                              "n_excluded_steered_unscored": sum(1 for k in s_d if s_d[k]["value"] is None), "n_excluded_anchor_unscored": sum(1 for k in a_d if a_d[k]["value"] is None),
                              "method_text": (f"paired per draw (item, trial) over the {N_DRAWS} draws scored on both sides at each row's FINAL state (the same {N_ITEMS} prompts x {K_TRIALS} trials, one "
                                              "instrument, the same fixed-base per-draw seeds): the strict pass-rate difference, pp; p from McNemar's exact test on the discordant pairs; Cohen's h of "
                                              "the two rates; an unscored draw on either side leaves the pair")}
                if reanchored:
                    stat_extra["pairing"] = f"paired per draw: the re-picked anchor {chosen} is a row of this batch ({plan['anchor_tag']}/none)"
                n_pairs = len(shared)
            else:
                # the anchor is a run of the registry, not of this batch: the registry's UNPAIRED rate difference from the two readings' k / n (rule-scored, no judge)
                try:
                    ust = DC.unpaired_rate_contrast(k_s, n_s, m.k, m.n)
                except DC.ContrastRefusal as exc:
                    why = f"the pairing against {retired}; the registry's statistic against the re-picked {chosen} is not derived: {exc.why}"
                    block.update(state="no_current_anchor", reason=why, anchor_run_id=None)
                    st["payload"]["extra"]["anchor"] = block
                    facts["contrasts"][f"{tag}|{okey}"] = dict(fact, state="no_current_anchor", why=why)
                    park(why, f"contrast's {retired}; no registry-derived statistic against the re-picked {chosen}: {exc.why}; filed without a contrast")
                    continue
                comp_row = DC.reanchor_computation("unpaired_rates", str(ctx.producer.get("code_sha") or "unverified"))
                self._reanchor_computations.setdefault(str(comp_row["computation_id"]), comp_row)
                stat = {k: ust[k] for k in ("delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind", "n_pairs")}
                stat_extra = {**dict(ust["extra"]), "statistic_computation_id": comp_row["computation_id"], "anchor_reading": m.reading_json(),
                              "pairing": f"unpaired: the re-picked anchor {chosen} is not a row of this batch; the registry's rate difference from the two readings' k / n",
                              "method_text": (f"unpaired rate difference of the row's strict prompt-level passes over its scored draws (k {k_s} of n {n_s}) against the anchor's registry "
                                              f"reading (k {m.k:g} of n {m.n}) in pp; Newcombe hybrid score interval; two-proportion z; Cohen's h")}
                n_pairs = 0
            if reanchored:
                if same_standup:
                    picked = f"its own standup's `none` the registry holds current from an earlier read ({plan['anchor_tag']}/none, {chosen}); the pair stays within the standup"
                elif choice.cross_source:
                    picked = f"the same model's `none` served by another standup of another source, the paper's roster-matched baseline ({plan['anchor_tag']}, {chosen})"
                else:
                    picked = f"the same model's `none` served by another standup ({plan['anchor_tag']}, {chosen})"
                reanchor_note = f"the row's own anchor ({own_text}) is {choice.own_reason}: the pass's anchor currency re-anchors the row on {picked}; {CURRENCY_RULE}"
                common_extra.update(cross_source=bool(choice.cross_source), contrast_caps={"steered": akey.token_cap_tokens, "anchor": akey.token_cap_tokens},
                                    anchor_standup=({"lane_source_legacy": cand.lane_source_legacy, "view_legacy": cand.view_legacy, "cluster_job_id": cand.cluster_job_id} if cand is not None else None),
                                    anchor_rule=CURRENCY_RULE, reanchor_tier=tier_key,
                                    own_pairing={"own_anchor_run_id": own_id, "own_anchor_kind": own_kind, "state": plan["state"], "why": plan["why"],
                                                 "own_none_state": plan["own_none_state"], "own_reason": choice.own_reason, "own_why": choice.own_why},
                                    reanchor_note=reanchor_note, **({"cross_standup_note": reanchor_note} if plan["cross_standup"] else {}))
            payload = {"run_id": st["run_id"], "anchor_run_id": choice.run_id, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0,
                       "computation_id": comp_id, "anchor_judgment_id": choice.anchor_judgment_id, "anchor_kind": B.anchor_kind_for(plan),
                       **stat, "computed_by": "registry", "computed_utc": st["when"], "extra": {**common_extra, **stat_extra}}
            if reanchored:
                pass        # the re-anchoring text is on `reanchor_note` (and `cross_standup_note` across standups) / `own_pairing` above
            elif plan["cross_standup"]:
                payload["extra"]["cross_standup_note"] = B.cross_standup_note(group, str(st.get("tag")), plan)
                if plan["own_none_state"] is not None:
                    payload["extra"].update(own_none_state=plan["own_none_state"], own_none_note=plan["own_none_note"])
            elif st.get("paired"):
                payload["extra"].update(anchor_rule=PAIRED_ANCHOR_RULE, own_standup_note=B.own_standup_note(group, str(st.get("tag"))))
            if st.get("declared_role") is not None:
                payload["extra"].update(declared_role=st["declared_role"],
                                        anchor_standup_note=(f"the value is declared {st['declared_role']}: the single-object standup composes another max_lora_rank than the seed tag's "
                                                             "(RECORDED on both runs, never compared); the contrast is the reader's, formed as for every steered key, and the role rides on it"))
            out.append(payload)
            if value_row is not None:
                value_row.setdefault("extra", {})["delta_block"] = None
            facts["contrasts_filed"] += 1
            facts["contrasts_reanchored" if reanchored else "contrasts_own"] += 1
            block.update(state=ANCHOR_CONTRASTED, anchor_run_id=choice.run_id, n_pairs=n_pairs)
            st["payload"]["extra"]["anchor"] = block
            facts["contrasts"][f"{tag}|{okey}"] = dict(fact, anchor=choice.run_id, state=ANCHOR_CONTRASTED, n_pairs=n_pairs, delta_pp=payload["delta"], delta_p=payload["delta_p"],
                                                       cohens_h=payload["effect_size"], cross_standup=plan["cross_standup"], declared_role=st.get("declared_role"),
                                                       anchor_final_cap=(anchor.get("final_cap") if anchor is not None else None), anchor_view_legacy=plan["anchor_tag"],
                                                       own_standup=(plan["own"] and not reanchored), reanchored=reanchored, **({"reanchor_tier": tier_key} if reanchored else {}))
        return out

def _sel_of(tags: Mapping[str, str], tag: str) -> str:
    """The selection of a tag in the group's ``{selection: tag}`` map."""
    for sel, t in tags.items():
        if t == tag:
            return sel
    raise KeyError(tag)


def factory() -> IfevalSeedrepAdapter:
    return IfevalSeedrepAdapter()


register(NAME, factory)
