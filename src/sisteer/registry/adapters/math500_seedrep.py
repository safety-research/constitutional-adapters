"""The batch-64 three-seed Qwen family's MATH-500 stores -> registry (source ``math500_seedrep``).

WHAT WAS RUN (the seed-replicate MATH-500 run, cluster B at tree ``4c8268da``, jobs 128075 /
128177 / 128180 (array rows 56-83) / 129051 on the cluster's QoS; the 30 rows the launch set names): seed tags ``qwen35_4b_s1`` / ``_s2`` / ``_s3``
x {the unsteered anchor ``none``, the five admitted staged objects (``base64s<N>__…`` / ``derived_base64s<N>__…``: constitution_relu,
control_relu, constitution_lora, control_lora, difference_lora), the two prompted system instructions (safety_si, constitution_si) and the
Arditi all-positions refusal steer at its operating point (``ar_c0p3__arditi_allpos``, ONE fit drawn at the three seed bases)}, plus the
composed ``difference_relu`` paired write on the PAIRED tags ``qwen35_4b_s<N>p`` (``paired_eb64s<N>__constv2-mix-full__psr_ce_all_minus_ctl``:
the same seed base N, its own standup — a STANDUP REPLICATE of ``s<N>``; its OWN ``none`` follows one per paired tag).
Every row is the paper's MATH-500 instrument — inspect_evals' ``math`` task as
row task ``inspect/math`` over the 500 MATH-500 ids, 0-shot, k = 1, thinking ON, the Qwen thinking-card decode row (T 0.6 / top_p 0.95 /
top_k 20 / min_p 0 / presence_penalty 0; the registered policy ``qwen35_4b_it|math500|on|thinking_card@v1``), ONE pass at 16,384 (the
standing truncation rule's ceiling: no cap step above, final == base) — served LoRA-natively (``proxy_vllm+lora+interleaved``; the
unsteered and prompted keys ride the passthrough route) on ``Qwen/Qwen3.5-4B @ 851bf6e8``, under the Inspect REPLICATE seed rule at seed
base s (the tag's index): ``manifest.json:seeds.inspect.replicate`` spells ``sha256(f'{seed_base}|inspect-replicate|{item_id}|{sample_idx}')[:4]
big-endian mod 2^31-1`` — read off EACH store, never assumed. Capability evaluations carry their OWN instrument (never the statics' 4,096
cap): every row files on the canonical MATH-500 instrument seed bound to the served model.

THE MIRROR (``run-state/full22/math500_seedrep/``, the mirror script's publish of cluster B's ``$SHARED/exp/
math500_thinking_on_full22``, byte for byte; ``PROVENANCE.md`` at the root): ``controls_qwen_seedrep/<tag>/<object dir>/thinking_card/`` is the
ONE store of a (tag, object key) — the object dir is the producer's slug (``constv2-mix-full--lora_r4_ce``: ``--`` for the harness id's ``__``).
What is read, every file named:

* ``thinking_card/cells/<store key>/manifest.json`` — served identity (``model_id`` / ``model_revision`` / ``serving_identity`` /
  ``serving_engine`` / ``fit_provenance.object_sha`` a 16-hex prefix of the object / ``route.object_artifact_sha256_by_cell[<harness id>]`` and
  ``provenance.artifact_sha256`` the full sha / ``route.bank_view`` the harness selection the key resolved through / ``resolved_config.system_sha``
  the prompt's / ``producer.tree_sha`` / ``row_constants.slurm_job_id`` / ``routing_table`` / ``harness_config_hash`` / ``intervention_hash`` /
  ``steer_positions`` / ``geometry`` / ``seeds.inspect`` / ``frameworks`` / ``oos_experiment`` / ``resume_attempts``);
* ``…/cells/<store key>/outcomes/inspect__math.json`` — the per-item OUTCOME record over EVERY attempted item (``at_cap`` / ``scored`` /
  ``value`` / ``stop_reason`` / ``completion_tokens`` / ``seed`` / ``think_block_unterminated``): the item universe;
* ``…/cells/<store key>/rows.jsonl`` — the scored ``acc`` rows and the ``unscored`` rows (the generation text of every item, the item's seed, the
  row's ``slurm_job_id`` / ``env_digest``) plus the producer's aggregate rows (``acc_aggregate``, ``n``; their ``extra_json`` attests the object
  sha and the tree); ``gen_stats.json`` and ``summary/inspect__math.json`` — the producer's aggregate (``acc`` over SCORED, ``n`` 500,
  ``n_scored``, ``n_unscored``, ``n_at_cap``, the sent decode numbers, the framework);
* ``thinking_card/RESULT`` (its LAST line must read ``result=COMPLETE``; earlier lines are failed attempts), ``row_identity.json``
  (``m5t_row_identity/3``, the launch stamp: the row's declared identity — expected object sha, staged id, prompt sha, tag, out root),
  ``<object dir>/controls_summary.json`` (the producer's per-store summary, cross-checked when present); ``oos_prompt_budget.json`` /
  ``evidence.json`` / ``resolved_sampler.json`` / ``request_evidence.jsonl`` / ``usage_ledger.jsonl`` are registered as the store's artifacts
  and not otherwise read;
* ``serving/<job dir>/math500/serving_attestation.json`` + ``standup_jobs.jsonl`` — the standup the manifest's ``routing_table`` names: the
  node, the cluster, and ``lora_config.max_lora_rank`` (the single-object Arditi standup composes 1, the seed standups 8) — RECORDED on the run,
  never compared;
* the wave-root records: ``store_pins_qwen_seedrep.json`` (``m5t_store_pins/2``: ``<model>/<object key>/thinking_card@<selection>`` -> the admitted
  producer tree; ``_meta.refused`` / ``_meta.partial`` / ``_meta.identity_refused`` name every refused row — a named row never files),
  ``three_numbers_math500_qwen_seedrep.json`` (``m5t_three_numbers_seedrep/1``: ``rows[<tag>/<object dir>]``, the producer's own count of every
  number this reader re-derives, the loop / eligible split of the at-cap items and the off-task classification beside them),
  ``final_state_math500_qwen_seedrep.json`` (``m5t_final_state_seedrep/1``: ``stores[<tag>/<object dir>/thinking_card]``, the per-item final state
  and the identity RE-DERIVED by the reader tree ``fc5bfc35``), and ``records/launch_sets.json`` (``launch_sets/v1``: the ROW TABLE —
  ``evals.math500.launch``, the rows the current pass launched, plus ``carried`` / ``carried_late``, the rows carried from the earlier pass
  (:data:`_bcbw_common.ROW_TABLE_RULE`; the census of 2026-09-23 moved the 30 pass-1 rows to ``carried``) — one row per (selection, object
  key) with its PIN KEY and EXPECTED object sha; the identity declaration).

OBJECT IDENTITY per (selection, object key) = the launch set's pin key, which must be the harness roster's staged id for the key at that selection
(``configs/sdfbase_full22_roster.yaml`` ``targets.qwen35_4b.views.s<N>[p].cells`` -> ``configs/sdfbase_full22_pins.json``; the all-positions key
through ``configs/arditi_allpos_roster.yaml`` / ``_pins.json`` at the selection the manifest's ``route.bank_view`` spells), whose pinned sha must
equal the launch set's expected sha AND every sha the store attests (``provenance.artifact_sha256``, ``route.object_artifact_sha256_by_cell``,
the rows' aggregate ``object_artifact_sha256``, the launch stamp's ``identity.object_artifact_sha256``, the 16-hex ``fit_provenance.object_sha``);
a prompted key resolves through ``si_texts.yaml`` and ``resolved_config.system_sha``. Any disagreement refuses the (tag, object key) by name
(:class:`lineage_common.RowRefusals`; the group's other rows file). The reader never consults the live registry for the object: the run is
emitted on the attested identity and ``blocked_records`` holds it while the object row is absent.

REGISTRY GRAIN: ONE run per (model, tag, object key): ``run_id = identity.run_id(sdfbase_full22, <tag>/<store key>, <object key>, math500,
<the canonical MATH-500 instrument bound to qwen35_4b_it>)`` — the producer key is the tag-prefixed store NAME (:data:`PRODUCER_KEY_RULE`),
never a path. ``token_cap_tokens`` 16,384, ``token_cap_rung_index`` 0 on the one-step ladder ``[16384]``. ONE GROUP PER SEED: the seed tag
``qwen35_4b_s<N>`` and its paired tag ``qwen35_4b_s<N>p`` read in one batch (:data:`GROUP_RULE`), because the anchor rule pairs the
composed row ACROSS tags with ``s<N>/none`` until the paired tag's own ``none`` files, and a contrast must name a run its own batch emits.

THE THREE NUMBERS (degeneration is scored separately; the GSM8K precedent):

1. THE VALUE — ``run_metric acc``: correct / SCORED, ``k`` the correct items, ``n`` the scored items, ``n_items_judged`` = ``n``, under the reader's
   recount computation (:data:`RECOUNT_ALGORITHM`, a non-blank ``computation_id``, ``computed_by registry``); it equals ``gen_stats.json:acc``
   (checked) and the producer's ``value_correct_over_scored`` (checked at four places).
2. THE BOUND — a SECOND ``run_metric acc`` row: correct / 500 ATTEMPTED (``k`` correct, ``n`` 500, the producer's spelling
   ``bound_correct_over_attempted`` as ``producer_metric_spelling``, ``computed_by producer``, a BLANK ``computation_id``) with the interval the
   unresolved items leave open in ``ci_lo`` / ``ci_hi`` (``ci_method unresolved_items_interval``). The frame's metric role ranks a computed row
   before a blank one at one identity: the value is the owner row, the bound secondary evidence.
3. THE THIRD ITEM STATE — an item whose draw ended at the token cap without an answer is NEITHER correct NOR incorrect: ``item_result.value NULL``
   with the label ``at_cap_unresolved`` and ``extra.at_cap_class`` ``loop`` (the producer's rep-ratio rule over the at-cap text marked it
   degenerate) or ``ceiling`` (at the one pass's cap, 16,384 — the standing truncation rule's ceiling). The at-cap SHARE = EVERY at-cap item
   (unresolved + scored at the cap) over the 500 attempted (``run_degeneration at_token_cap_finish_reason``, the finish-reason reading);
   the loop / eligible split is INFORMATIONAL (no cap step exists on MATH-500) and files as ``rep_ratio_loop`` from the PRODUCER's
   count (``computed_by producer``), ``truncation_only`` = the unresolved items that are not loops. An item SCORED at the cap keeps its value
   (``item_flags.at_token_cap 1``, ``extra.scored_at_cap``); an item UNSCORED below the cap (the think block never closed before EOS, the
   instrument's ``unterminated_policy``) is ``value NULL`` under the label ``unscored`` with ``extra.unscored_reason unterminated_below_cap`` — a
   miss in the bound, in no degeneration measure. The producer's off-task classification (``off_task``: scorer-anchored, over the not-correct
   items) is CARRIED on the run (``extra.three_numbers.producer_off_task``), never re-derived and never a measure the vocabulary lacks.

CROSS-CHECKS, every one a refusal of the (tag, object key) by name: the manifest's tree against the store pin (``@<selection>``); the outcome record
against ``rows.jsonl`` item by item (500 distinct items; the scored values; the unscored rows) and against ``gen_stats.json`` /
``summary/inspect__math.json`` / the record's own header counts; ``RESULT``'s last line; the served model / revision / serving identity /
engine; the attested object shas (above); the replicate ``seed_base`` against the tag's index; thinking ON and the 16,384 cap on every record
that spells them; the launch stamp's declared identity; the producer's ``controls_summary.json`` when present; and EVERY number of the producer's
three-number row (the value, the counts, the bound, the third state, the loop / eligible split, the store block's identity fields) and of the
final-state record's entry (the counts and the 500 items) against the re-derivation.

ANCHORS: every steered and prompted row of a seed tag files a ``run_contrast`` against ITS tag's ``none`` (``anchor_kind same_standup_none``),
item-wise over the items scored on both sides (pp difference, McNemar's exact p, Cohen's h). THE PAIRED TAG'S OWN ``none`` IS ITS ANCHOR
(the GSM8K precedent; :data:`PAIRED_ANCHOR_RULE`, :func:`_bcbw_common.anchor_plan`): a
``none`` under ``qwen35_4b_s<N>p`` is launch-set class ``anchor`` and files as a normal run of the paired tag — object ``none``, the MATH-500
instrument, a decode-seed TRIAL at the seed index like the seed tag's ``none`` (``extra.decode_replicate.standup_replicate_of_view`` names the
replicated tag), no contrast of its own — and the composed ``s<N>p`` row then pairs against IT (``cross_standup`` false, ``anchor_view_legacy`` the
paired tag). Until that ``none`` FILES the composed row keeps the fallback ACROSS tags against ``s<N>/none`` (``anchor_kind sibling_none`` —
the same model's ``none`` served by another standup, :func:`_bcbw_common.anchor_kind_for`; ``extra.cross_standup``,
``cross_standup_note``; the run's ``extra.own_none_state`` names the pending ``none`` once the launch set declares it). The anchor is resolved from
the rows that FILED, never from a store's presence — the ``none`` rows of both tags read first, so a paired row's run spells the anchor its contrast
pairs against — and a seed tag's rows never pair with the paired tag's ``none`` (the ``(tag, none)`` keys differ). A prompted key under the paired
tag is class ``prompted`` and pairs like any row of its tag. The Arditi rows are declared UNPAIRED (role ``arditi_unpaired_value``, the GSM8K precedent: the
single-object standup attests another ``max_lora_rank`` than the seed standups) — the value files regardless, the contrast is formed as for
every steered key and the declared role rides on the run and the contrast for the frame's reader. The unsteered and prompted keys name ONE
object on every seed tag and file as decode-seed TRIALS (``trial_index`` = the seed index, ``extra.decode_replicate``); the all-positions object
likewise (one fit, three seed bases: :data:`DECODE_TRIAL_RULE`); the fitted keys are one object per seed and stay at trial 0. Coexistence in
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

NAME = "math500_seedrep"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "math500_seedrep"
OWNER = "Overflow"
PRODUCER_SESSION = "Overflow"
CAMPAIGN = LC.CAMPAIGN
ROOT_DEFAULT = os.path.join(LC.RS_DEFAULT, "math500_seedrep")
#: the mirror's store head: `<root>/controls_qwen_seedrep/<tag>/<object dir>/thinking_card/`
STORES_HEAD = "controls_qwen_seedrep"
SERVING_HEAD = "serving"
RECORDS_DIR = "records"
MODEL_KEY = SC.MODEL_KEY
#: a seed tag `qwen35_4b_s<N>` or its PAIRED tag `qwen35_4b_s<N>p` (the composed difference_relu object at the same seed base, its own standup)
TAG_RE = re.compile(rf"^(?P<model>{MODEL_KEY})_s(?P<seed>\d+)(?P<paired>p?)$")
EVAL_ID = "math500"
TASK = "inspect/math"
#: the paper's MATH-500 instrument (configs/registry/instruments.yaml), bound per model by the dimensions loader
INSTRUMENT_SEED = "math500_inspect_thinking_on_16384_onepass"
CAP = 16384
LADDER: Tuple[int, ...] = (CAP,)
N_ITEMS = 500
STORE_DIR = "thinking_card"
OUTCOMES_FILE = "inspect__math.json"
SUMMARY_FILE = "inspect__math.json"
PINS_FILE = "store_pins_qwen_seedrep.json"
THREE_NUMBERS_FILE = "three_numbers_math500_qwen_seedrep.json"
FINAL_STATE_FILE = "final_state_math500_qwen_seedrep.json"
LAUNCH_SET_FILE = "launch_sets.json"
SHARED_FILES = (PINS_FILE, THREE_NUMBERS_FILE, FINAL_STATE_FILE, LAUNCH_SET_FILE)
PINS_SCHEMA = "m5t_store_pins/2"
THREE_NUMBERS_SCHEMA = "m5t_three_numbers_seedrep/1"
FINAL_STATE_SCHEMA = "m5t_final_state_seedrep/1"
LAUNCH_SET_SCHEMA = "launch_sets/v1"
ROW_IDENTITY_SCHEMA = "m5t_row_identity/3"
CONTROLS_SUMMARY_FILE = "controls_summary.json"
ATTESTATION_FILE = "serving_attestation.json"
STANDUP_JOBS_FILE = "standup_jobs.jsonl"
SERVING_FILES = (ATTESTATION_FILE, STANDUP_JOBS_FILE)
#: the files of a store directory and of its `cells/<store key>/` directory
STORE_TOP_FILES = ("RESULT", "row_identity.json", "oos_prompt_budget.json", "evidence.json", "resolved_sampler.json", "request_evidence.jsonl", "usage_ledger.jsonl")
STORE_KEY_FILES = ("gen_stats.json", "manifest.json", "rows.jsonl", "evidence.json")
ROLE_OF_FILE = {"rows.jsonl": "rows", "gen_stats.json": "completion", "RESULT": "log", "request_evidence.jsonl": "log", "usage_ledger.jsonl": "log",
                "vllm_server.log": "log", STANDUP_JOBS_FILE: "log"}
RESULT_COMPLETE = "result=COMPLETE"
_RESULT_RE = re.compile(r"^(?P<ts>\S+)\s+job=(?P<job>\S+)\s+array=(?P<array>\S+)\s+row=(?P<row>\S+)\s+(?P<job_name>\S+)\s+m5t\s+(?P<model>\S+)\s+(?P<okey>\S+)\s+(?P<mode>\S+)"
                        r"\s+fitted=(?P<fitted>\S+)\s+rc=(?P<rc>\S+)\s+driver_rc=(?P<driver_rc>\S+)\s+result=(?P<result>\S+)(?P<rest>.*)$")
#: the third item state's label and the classes of an unresolved at-cap item: `loop` by the producer's rep-ratio rule, else `ceiling` — the one
#: pass IS at the standing truncation rule's ceiling, so nothing is "not escalated"
ITEM_LABEL_AT_CAP_UNRESOLVED = IC.ITEM_LABEL_AT_CAP_UNRESOLVED
AT_CAP_CLASSES = ("loop", "ceiling")
ITEM_LABEL_UNSCORED = G.ITEM_LABEL_UNSCORED
UNSCORED_BELOW_CAP_CLASS = G.UNSCORED_BELOW_CAP_CLASS
#: the run-level shares (run_degeneration measures over the 500 attempted items; the spellings the GSM8K reader uses)
RAW_AT_CAP_MEASURE = G.RAW_AT_CAP_MEASURE
LOOP_MEASURE = G.LOOP_MEASURE
COHERENT_MEASURE = G.COHERENT_MEASURE
#: the headline's computation: the recount of the outcome record (no rng, no merge — one pass, final == base)
RECOUNT_ALGORITHM = "sisteer.registry.adapters.math500_seedrep.recount"
RECOUNT_VERSION = "1"
RECOUNT_SEED_PROVENANCE = "none: a deterministic recount of the store's outcome record (correct / scored over the 500 attempted items; one pass, no cap step)"
#: the bound row's spellings (the producer's own, three_numbers_math500_qwen_seedrep.json)
BOUND_SPELLING = "bound_correct_over_attempted"
BOUND_CI_METHOD = G.BOUND_CI_METHOD
HEADLINE_METRIC = "acc"
DECODE_SEED_RULE = "per_item"
INSPECT_SEED_POLICY = "replicate"
UNFITTED_TRIAL_RULE = SC.UNFITTED_TRIAL_RULE
DECODE_TRIAL_RULE = SC.DECODE_TRIAL_RULE
STEERED_IDENTITY = SC.STEERED_IDENTITY
PASSTHROUGH_IDENTITY = SC.PASSTHROUGH_IDENTITY
SERVING_ENGINE = "lora_native"
ROSTER_REL = SC.ROSTER_REL
PINS_REL = SC.PINS_REL
ARDITI_ROSTER_REL = SC.SINGLE_OBJECT_ROSTER_REL
ARDITI_PINS_REL = SC.SINGLE_OBJECT_PINS_REL
ARDITI_OBJECT_KEY = DG.OBJECT_KEY_LEGACY
#: the declared role of an Arditi row's value (the GSM8K C4 precedent): the single-object standup attests another max_lora_rank than
#: the seed standups, so the producer files the value UNPAIRED; this reader forms the contrast as for every steered key and parks the role on it
ARDITI_ROLE = "arditi_unpaired_value"
#: the launch set's row classes (records/launch_sets.json evals.math500.launch[*].class)
LAUNCH_CLASSES = ("anchor", "arm", "prompted", "composed", "arditi")
#: the producer's three-number record rounds to four places; the recount is compared at that precision
ROUNDED_TOL = G.ROUNDED_TOL
CONTRAST_METHOD = G.CONTRAST_METHOD
PRODUCER_KEY_RULE = "the producer key is `<tag>/<store key>`: the tag prefixes the store's own NAME (never its path), so a paired tag's store never shares a key with a seed tag's"
GROUP_RULE = ("one group per SEED: the seed tag `qwen35_4b_s<N>` and its paired tag `qwen35_4b_s<N>p` read in one batch — the anchor rule (D65) pairs the "
              "composed row across tags with `s<N>/none`, and a contrast names a run its own batch emits")
ANCHOR_RULE = ("D65: every steered and prompted row pairs with ITS seed tag's `none`; the composed `s<N>p` row pairs ACROSS tags with `s<N>/none`; the "
               "Arditi rows are declared unpaired (arditi_unpaired_value) — the value files regardless and the contrast carries the role")
#: the paired tag's OWN `none` is its anchor once it files (the rule relayed 2026-09-23): the rule text, the own-`none` states and the
#: resolution shared with the BigCodeBench / BFCL / IFEval readers (_bcbw_common); ANCHOR_RULE above stays byte for byte on the rows it describes
PAIRED_ANCHOR_RULE = B.PAIRED_ANCHOR_RULE
OWN_NONE_NOT_YET_RUN = B.ANCHOR_NOT_YET_RUN
OWN_NONE_REFUSED = B.ANCHOR_REFUSED
ANCHOR_CONTRASTED = B.ANCHOR_CONTRASTED
#: every contrast is decided by the pass's anchor currency (the six capability readers' shared rule)
CURRENCY_RULE = B.CURRENCY_RULE
anchor_plan = B.anchor_plan
#: the launch set's row table: `launch` + `carried` + `carried_late` (shared with the BigCodeBench / BFCL / IFEval readers)
ROW_TABLE_RULE = B.ROW_TABLE_RULE
ROW_TABLE_LISTS = B.ROW_TABLE_LISTS
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA16 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

slug_to_object_key = G.slug_to_object_key
rounded_equal = G.rounded_equal
_PairRefusal = G._PairRefusal


def selection_key(seed_n: int, paired: bool = False) -> str:
    """The harness roster's selection of a tag: ``s<N>`` or ``s<N>p``."""
    return f"s{seed_n}{'p' if paired else ''}"


def tag_parts(tag: str) -> Optional[Tuple[int, bool]]:
    m = TAG_RE.match(tag)
    return (int(m.group("seed")), bool(m.group("paired"))) if m else None


def seed_tag_of(tag: str) -> Optional[str]:
    """The GROUP of a tag: the seed tag itself, or the seed tag a paired tag stands beside (:data:`GROUP_RULE`)."""
    p = tag_parts(tag)
    return f"{MODEL_KEY}_s{p[0]}" if p else None


def pin_key(selection: str, object_key: str) -> str:
    return f"{MODEL_KEY}/{object_key}/{STORE_DIR}@{selection}"


def producer_key_for(tag: str, store_key: str) -> str:
    return f"{tag}/{store_key}"


def run_id_for(tag: str, store_key: str, object_key: str, instrument_id: str) -> str:
    return I.run_id(CAMPAIGN, producer_key_for(tag, store_key), object_key, EVAL_ID, instrument_id)


def recount_computation_id() -> str:
    return I.computation_id(RECOUNT_ALGORITHM, RECOUNT_VERSION, RECOUNT_SEED_PROVENANCE)


def parse_result_line(line: str) -> Optional[Dict[str, str]]:
    """One line of a store's ``RESULT``: ``<ts> job=<id> array=<k> row=<k> <job name> m5t <model> <object key> <mode> fitted=<0|1> rc=<rc>
    driver_rc=<rc> result=<STATE> [wave=… thinking=… geometry=…]``; None when the line spells something else."""
    m = _RESULT_RE.match(line.strip())
    if not m:
        return None
    d = {k: v for k, v in m.groupdict().items() if k != "rest"}
    for tok in (m.group("rest") or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d.setdefault(k, v)
    return d


def serving_dir_rel(routing_table: Optional[str]) -> Optional[str]:
    """The mirror-relative standup directory the manifest's ``routing_table`` names: ``serving/<job dir>/math500`` (the last two directories
    of the recorded path under the ``serving`` head); None when the path spells no such shape."""
    p = str(routing_table or "")
    d = os.path.dirname(p)
    if not d:
        return None
    parts = d.split(os.sep)
    if len(parts) < 3 or parts[-3] != SERVING_HEAD:
        return None
    return os.path.join(SERVING_HEAD, parts[-2], parts[-1])


class Math500SeedrepAdapter:
    name = NAME
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-23): the first version — the seed-replicate MATH-500 run (module docstring)
    #: (2026-09-26, no bump: the composed row's fallback contrast -- ACROSS tags against s<N>/none, extra.cross_standup true -- now files anchor_kind
    #: sibling_none through _bcbw_common.anchor_kind_for instead of same_standup_none; the live mirror re-reads byte for byte because every composed row's
    #: own `none` has filed and no contrast crosses standups any more (the dry run over the live mirror))
    #: 2 (2026-10-04): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`; the row's own pairing stands while the frame holds it current, a retired or absent own
    #: `none` re-anchors on a current unsteered run of the exact key -- paired per item when it is a row of the batch, else the registry's unpaired
    #: rate difference from the two readings' k / n); a comparison with no current anchor parks on the VALUE row's extra.delta_block; every steered
    #: run carries extra.anchor (the block the other seedrep readers stamp)
    EMISSION_VERSION = 2
    rng_scope = "none"
    allow_symlinks = False
    owner = OWNER
    #: the `gen_stats.json:thinking` spelling every store of this run carries (the Qwen thinking card is ON everywhere)
    THINKING_SPELLING = "on"

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

    def discover(self) -> List[Input]:
        """Named files of every ``controls_qwen_seedrep/<tag>/<object dir>/thinking_card/`` under the mirror root, the standup records each store's
        manifest names (aliased into the store's group), and the wave-root records aliased per group (``#<seed tag>``). An absent root is a SKIP
        (the mirror is machine-local); a root holding no store refuses."""
        out: List[Input] = []
        if not os.path.isdir(self.root):
            return out
        groups: Set[str] = set()
        for tag in IC.listdir(self._stores_root()):
            group = seed_tag_of(tag)
            if group is None or not os.path.isdir(os.path.join(self._stores_root(), tag)):
                continue
            for slug in IC.listdir(os.path.join(self._stores_root(), tag)):
                d = os.path.join(self._stores_root(), tag, slug, STORE_DIR)
                if not os.path.isdir(d):
                    continue
                n_before = len(out)
                out += self._discover_store(d, slug)
                if len(out) > n_before:
                    groups.add(group)
                    out += self._discover_serving(d, group)
        if not out:
            raise AdapterRefusal(self.root, f"no MATH-500 seed-replicate store found ({STORES_HEAD}/{MODEL_KEY}_s<N>[p]/<object dir>/{STORE_DIR}/cells/<store key>/ under the root)")
        for name in SHARED_FILES:
            p = os.path.join(self.root, RECORDS_DIR, name) if name == LAUNCH_SET_FILE else os.path.join(self.root, name)
            if os.path.isfile(p):
                base = check_input(p, role="report", allow_symlink=self.allow_symlinks)
                out += [base.with_alias(g) for g in sorted(groups)]
        return out

    def _discover_store(self, d: str, slug: str) -> List[Input]:
        out: List[Input] = []
        for name in STORE_TOP_FILES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out.append(check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks))
        p = os.path.join(d, slug, CONTROLS_SUMMARY_FILE)
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
            for sub, fname in (("outcomes", OUTCOMES_FILE), ("summary", SUMMARY_FILE)):
                p = os.path.join(kd, sub, fname)
                if os.path.isfile(p):
                    out.append(check_input(p, role="report", allow_symlink=self.allow_symlinks))
        return out

    def _discover_serving(self, d: str, group: str) -> List[Input]:
        """The standup records the store's manifest names (``serving/<job dir>/math500/``), aliased into the store's group: a standup directory
        spells no tag of its own."""
        out: List[Input] = []
        for key in IC.listdir(os.path.join(d, "cells")):
            mp = os.path.join(d, "cells", key, "manifest.json")
            if not os.path.isfile(mp):
                continue
            try:
                with open(mp, "r", encoding="utf-8") as fh:
                    man = json.load(fh)
            except (OSError, ValueError):
                continue
            rel = serving_dir_rel((man or {}).get("routing_table")) if isinstance(man, dict) else None
            if rel is None:
                continue
            for name in SERVING_FILES:
                p = os.path.join(self.root, rel, name)
                if os.path.isfile(p):
                    base = check_input(p, role=ROLE_OF_FILE.get(name, "report"), allow_symlink=self.allow_symlinks)
                    out.append(base.with_alias(group))
        return out

    def comparison_group(self, inp: Input) -> str:
        """One group per SEED (:data:`GROUP_RULE`): a store under ``<tag>/`` maps to the tag's seed tag; a wave-root record or standup record
        rides a ``#<seed tag>`` alias."""
        base = os.path.basename(inp.uri)
        if "#" in base and seed_tag_of(base.rsplit("#", 1)[1]) == base.rsplit("#", 1)[1]:
            return base.rsplit("#", 1)[1]
        rel = os.path.relpath(inp.uri, self.root).split(os.sep)
        if len(rel) > 1 and rel[0] == STORES_HEAD and seed_tag_of(rel[1]) is not None:
            return seed_tag_of(rel[1])
        raise AdapterRefusal(inp.uri, f"input names no seed tag ({STORES_HEAD}/{MODEL_KEY}_s<N>[p]/) under {self.root}")

    def _group_selections(self, group: str) -> Tuple[str, str]:
        seed_n = tag_parts(group)[0]
        return selection_key(seed_n), selection_key(seed_n, True)

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        for inp in inputs:
            base = os.path.basename(inp.uri).rsplit("#", 1)[0]
            if base not in SHARED_FILES or "#" not in os.path.basename(inp.uri):
                continue
            group = self.comparison_group(inp)
            sels = self._group_selections(group)
            tags = tuple(f"{MODEL_KEY}_{s}" for s in sels)
            d = self._json(inp) or {}
            if base == THREE_NUMBERS_FILE:
                members = sorted(k for k in (d.get("rows") or {}) if str(k).split("/", 1)[0] in tags)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.rows", 1, 1, members, inp.sha256))
            elif base == PINS_FILE:
                members = sorted(k for k in d if k != "_meta" and str(k).rsplit("@", 1)[-1] in sels)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.pins", 1, 1, members, inp.sha256))
            elif base == FINAL_STATE_FILE:
                members = sorted(k for k in (d.get("stores") or {}) if str(k).split("/", 1)[0] in tags)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, "$.stores", 1, 1, members, inp.sha256))
            elif base == LAUNCH_SET_FILE:
                launch = self._launch_rows(d)
                members = sorted(str(r.get("tuple_id")) for r in (launch or []) if str(r.get("view")) in sels)
                rows.append(SourceStateRow(source_id_for(NAME, group), inp.uri, B.BcbwSeedrepReader.row_table_json_path(d, EVAL_ID), 1, 1, members, inp.sha256))
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

    @staticmethod
    def _load_pair(roster_path: str, pins_path: str, what: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not os.path.isfile(roster_path):
            raise IC.Unresolved(roster_path, f"the {what} roster is not in this checkout; no staged id can be named")
        if not os.path.isfile(pins_path):
            raise IC.Unresolved(pins_path, f"the {what} pins are not in this checkout; no object sha can be named")
        import yaml
        with open(roster_path, "r", encoding="utf-8") as fh:
            roster = yaml.safe_load(fh)
        with open(pins_path, "r", encoding="utf-8") as fh:
            pins = json.load(fh)
        if not isinstance(roster, dict) or not isinstance(pins, dict):
            raise IC.Unresolved(roster_path, f"the {what} roster / pins are not mappings")
        return roster, pins

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

    # ---- objects ------------------------------------------------------------------------------
    def resolve_object(self, object_key: str, selection: str, bank: Optional[str] = None) -> Tuple[LC.ResolvedObject, Optional[DG.ArditiAllposRung]]:
        """The object a store's key names at the harness selection: ``none``, a declared prompt (``si:<sha>`` from si_texts), a staged
        seed-replicate object (the roster's ``s<N>[p]`` selection -> staged id -> the pins' sha, :func:`jobplan.object_id_for`), or the
        all-positions object through ITS roster / pins at the selection the manifest's ``route.bank_view`` spells (``bank``). Never the live
        registry. Returns the object and, for the all-positions key, its operating point."""
        g = LC.cell_grammar_legacy(object_key)
        kind = g["object_kind"]
        if kind == "none":
            return LC.ResolvedObject("none", "none", "none", "baseline_none", "derived", "the unsteered object key", derivation="none"), None
        if kind == "system_instruction":
            oid = self.seeds.si_object_id(g["si_id"] or "")
            if oid is None:
                raise IC.Unresolved(object_key, f"si_texts.yaml carries no sha256 for the declared prompt {g['si_id']!r}")
            return LC.ResolvedObject(oid, kind, g["paper_label"], "prompt", "recorded", "si_texts.yaml si_text_sha256", si_id=g["si_id"], derivation="none"), None
        if kind is None:
            raise IC.Unresolved(object_key, "the key matches no object grammar this reader knows")
        if object_key == ARDITI_OBJECT_KEY:
            op = DG.parse_arditi_allpos_tag(f"{MODEL_KEY}_{bank}") if bank else None
            if op is None or op.trial is not None or op.model_key != MODEL_KEY:
                raise IC.Unresolved(object_key, f"the manifest's route.bank_view {bank!r} spells no all-positions operating point (dose_grammar `ar_c<tok>`)")
            roster, pins = self._arditi_roster_pins()
            try:
                sha = JP.object_id_for(object_key, model_key=MODEL_KEY, roster_selection=op.rung, roster=roster, pins=pins, si_sha={})
            except LookupError as exc:
                raise IC.Unresolved(object_key, f"the all-positions roster's {op.rung!r} selection / the pins name no object for it: {exc}") from None
            staged = str(roster["targets"][MODEL_KEY]["views"][op.rung]["cells"][object_key])
            if staged != op.staged_id:
                raise IC.Unresolved(object_key, f"the all-positions roster stages {op.rung}/{object_key} as {staged!r}, not the operating point's {op.staged_id!r} (dose_grammar)")
            pin = (pins.get(MODEL_KEY) or {}).get(staged) or {}
            bad: List[str] = []
            if str(pin.get("kind") or "") != DG.OBJECT_KIND:
                bad.append(f"kind {pin.get('kind')!r} is not {DG.OBJECT_KIND!r}")
            if LC.num_or_none(pin.get("c")) is None or abs(float(pin["c"]) - op.coefficient) > 1e-9:
                bad.append(f"c {pin.get('c')!r} is not the operating point's {op.coefficient:g}")
            if str(pin.get("dose_unit") or "") != op.unit:
                bad.append(f"dose_unit {pin.get('dose_unit')!r} is not {op.unit!r}")
            if bad:
                raise IC.Unresolved(object_key, f"the pin {MODEL_KEY}.{staged} is not the operating point: " + "; ".join(bad))
            obj = LC.ResolvedObject(str(sha), DG.OBJECT_KIND, DG.PAPER_LABEL, DG.OBJECT_ROLE, "recorded",
                                    f"manifest.json:route.bank_view {op.rung} -> all-positions roster targets.{MODEL_KEY}.views.{op.rung}.cells[{object_key}] -> {staged}; pins "
                                    f"{MODEL_KEY}.{staged}.object_sha256 (jobplan.object_id_for, the harness's own resolution); the launch set's pin key and expected sha; "
                                    "attested by the store's provenance.artifact_sha256, route.object_artifact_sha256_by_cell, the rows aggregate and the launch stamp",
                                    staged_id=staged, derivation="none")
            return obj, op
        roster, pins = self._roster_pins()
        try:
            sha = JP.object_id_for(object_key, model_key=MODEL_KEY, roster_selection=selection, roster=roster, pins=pins, si_sha={})
        except LookupError as exc:
            raise IC.Unresolved(object_key, f"the roster's {selection!r} selection / the pins name no object for it: {exc}") from None
        staged = roster["targets"][MODEL_KEY]["views"][selection]["cells"][object_key]
        obj = LC.ResolvedObject(str(sha), kind, g["paper_label"], g["object_role"], "recorded",
                                f"roster targets.{MODEL_KEY}.views.{selection}.cells[{object_key}] -> {staged}; pins {MODEL_KEY}.{staged}.object_sha256 (jobplan.object_id_for, "
                                "the harness's own resolution); the launch set's pin key and expected sha; attested by the store's provenance.artifact_sha256, "
                                "route.object_artifact_sha256_by_cell, the rows aggregate and the launch stamp",
                                staged_id=str(staged), derivation=g["derivation"], si_id=g["si_id"])
        return obj, None

    # ---- read ---------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        batch = RecordBatch(source_id_for(NAME, group), group)
        self.refusals = LC.RowRefusals(NAME)
        facts: Dict[str, Any] = {"group": group, "runs": 0, "rows": {}, "contrasts": {}, "refusals": [], "instrument_seed_id": INSTRUMENT_SEED,
                                 "group_rule": GROUP_RULE, "anchor_rule": ANCHOR_RULE, "producer_key_rule": PRODUCER_KEY_RULE}
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
        stores, shared, serving = self._sort_inputs(inputs)
        pins_inp, tn_inp, fs_inp, ls_inp = (shared.get(n) for n in SHARED_FILES)
        pins = self._json(pins_inp) if pins_inp is not None else None
        three = self._json(tn_inp) if tn_inp is not None else None
        fstate = self._json(fs_inp) if fs_inp is not None else None
        launch_doc = self._json(ls_inp) if ls_inp is not None else None
        pins_ok = isinstance(pins, dict) and str((pins.get("_meta") or {}).get("schema")) == PINS_SCHEMA
        three_ok = isinstance(three, dict) and str(three.get("schema")) == THREE_NUMBERS_SCHEMA and isinstance(three.get("rows"), dict)
        fstate_ok = isinstance(fstate, dict) and str(fstate.get("schema")) == FINAL_STATE_SCHEMA and isinstance(fstate.get("stores"), dict)
        launch_rows = self._launch_rows(launch_doc)
        for name, inp in (("store_pins", pins_inp), ("three_numbers", tn_inp), ("final_state", fs_inp), ("launch_set", ls_inp)):
            facts[name] = LC.rel_home(inp.uri.rsplit("#", 1)[0]) if inp is not None else None
        # the rows the group is expected to hold: every launch-set row at the group's two selections
        expected: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in launch_rows or []:
            sel = str(r.get("view") or "")
            if sel in tags and r.get("cell"):
                expected[(tags[sel], str(r["cell"]))] = r
        filed: Dict[Tuple[str, str], Dict[str, Any]] = {}
        seen: Set[Tuple[str, str]] = set()
        # the `none` rows of both tags file FIRST (PAIRED_ANCHOR_RULE): a paired tag's row spells the anchor its contrast pairs against — its own `none`
        # when that FILED, else the seed tag's — and the run record is added as the row is read, so the anchors' status must be final by then
        for (tag, slug), st in sorted(stores.items(), key=lambda kv: (slug_to_object_key(kv[0][1]) != "none", kv[0])):
            okey = slug_to_object_key(slug)
            seen.add((tag, okey))
            rkey = f"{tag}|{okey}"
            if not pins_ok:
                self.refusals.add(rkey, f"{PINS_FILE} ({PINS_SCHEMA}) is absent or malformed at the mirror root; no store pin can be checked", self.root)
                continue
            if not three_ok:
                self.refusals.add(rkey, f"{THREE_NUMBERS_FILE} ({THREE_NUMBERS_SCHEMA}) is absent or malformed at the mirror root; nothing can be cross-checked", self.root)
                continue
            if launch_rows is None:
                self.refusals.add(rkey, f"{RECORDS_DIR}/{LAUNCH_SET_FILE} ({LAUNCH_SET_SCHEMA}) is absent or malformed at the mirror root; no pin key and no expected object sha can be named", self.root)
                continue
            lrow = expected.get((tag, okey))
            if lrow is None:
                self.refusals.add(rkey, f"the launch set names no row for {tags_inv(tags, tag)}/{okey}; a store the launch set does not declare has no identity to bind to", ls_inp.uri)
                continue
            paired = tag_parts(tag)[1]
            plan = anchor_plan(group, tag, filed, expected, seen) if (paired and okey != "none") else None
            try:
                filed[(tag, okey)] = self._emit_row(batch, ctx, group, tag, seed_n, paired, tags_inv(tags, tag), model_id, inst_row, okey, slug, st, pins, three,
                                                    (fstate if fstate_ok else None), lrow, serving, pins_inp, tn_inp, fs_inp, ls_inp, facts, anchor_plan=plan)
            except _PairRefusal as exc:
                self.refusals.add(rkey, exc.why, exc.where)
        for (tag, okey), lrow in sorted(expected.items()):
            if (tag, okey) not in seen:
                self.refusals.add(f"{tag}|{okey}", f"the launch set names this row (index {lrow.get('m5t_index')}, {lrow.get('m5t_out_root')}) but no store is mirrored at "
                                                   f"{STORES_HEAD}/{tag}/{okey.replace('__', '--', 1)}/{STORE_DIR}", ls_inp.uri if ls_inp is not None else self.root)
        # the anchor blocks first (they ride the run payloads), then every row's run and VALUE records, then the contrasts -- a record is built once, never
        # re-stamped. Every contrast is DECIDED by the pass's anchor currency (CURRENCY_RULE; the frame loads once per pass through ctx.anchor_currency()
        # and is asked on the first comparison; a group without one never asks): the parks land on the VALUE rows' extra.delta_block before they are added
        self._reanchor_computations = {}
        facts.update(contrasts_filed=0, contrasts_own=0, contrasts_reanchored=0, contrasts_parked=0, anchor_states={})
        contrasts = self._emit_contrasts(ctx, group, filed, expected, seen, facts)
        for key in sorted(filed):
            batch.add("run", filed[key]["payload"], ctx.producer, ctx.now)
            batch.add("run_metric", filed[key]["value_row"], ctx.producer, ctx.now)
        for comp_row in self._reanchor_computations.values():
            batch.add("computation", comp_row, ctx.producer, ctx.now)      # the registry's statistic of a re-anchored contrast (extra.statistic_computation_id names it)
        for payload in contrasts:
            batch.add("run_contrast", payload, ctx.producer, ctx.now)
        facts["anchor_states"] = dict(sorted(facts["anchor_states"].items()))
        facts["reanchor_computations"] = len(self._reanchor_computations)
        self._finish(group, facts)
        yield batch

    @staticmethod
    def _launch_rows(doc: Optional[Mapping[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """The eval's ROW TABLE (:data:`ROW_TABLE_RULE`): ``launch`` plus ``carried`` / ``carried_late`` (the shared reading in ``_bcbw_common``)."""
        return B.BcbwSeedrepReader._launch_rows_of(doc, EVAL_ID)

    def _sort_inputs(self, inputs: Sequence[Input]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Input], Dict[str, Dict[str, Input]]]:
        """The group's inputs sorted into stores — ``(tag, object slug) -> {"top": {name: Input}, "keys": {store key: {name: Input}}, "dir",
        ["summary"]}`` — the shared records (``#<group>`` aliases) and the standup records (``serving/<job dir>/math500`` -> {name: Input})."""
        stores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        shared: Dict[str, Input] = {}
        serving: Dict[str, Dict[str, Input]] = {}
        for inp in inputs:
            base = os.path.basename(inp.uri)
            if "#" in base:
                real = inp.uri.rsplit("#", 1)[0]
                name = os.path.basename(real)
                rel = os.path.relpath(real, self.root).split(os.sep)
                if name in SHARED_FILES and (len(rel) == 1 or (len(rel) == 2 and rel[0] == RECORDS_DIR)):
                    shared[name] = inp
                elif len(rel) == 4 and rel[0] == SERVING_HEAD and name in SERVING_FILES:
                    serving.setdefault(os.path.join(*rel[:3]), {})[name] = inp
                continue
            rel = os.path.relpath(inp.uri, self.root).split(os.sep)
            if len(rel) < 5 or rel[0] != STORES_HEAD or rel[3] != STORE_DIR:
                continue
            tag, slug = rel[1], rel[2]
            st = stores.setdefault((tag, slug), {"top": {}, "keys": {}, "dir": os.path.join(self.root, *rel[:4])})
            body = rel[4:]
            if len(body) == 1:
                st["top"][body[0]] = inp
            elif len(body) == 2 and body[0] == slug and body[1] == CONTROLS_SUMMARY_FILE:
                st["summary"] = inp
            elif len(body) >= 3 and body[0] == "cells":
                files = st["keys"].setdefault(body[1], {})
                if len(body) == 3:
                    files[body[2]] = inp
                elif len(body) == 4 and body[2] in ("outcomes", "summary"):
                    files[body[2]] = inp
        return stores, shared, serving

    def _finish(self, group: str, facts: Dict[str, Any]) -> None:
        facts["refusals"] = [f"{it['key']} -- {it['why']}" for it in self.refusals.items]
        facts["n_refusals"] = len(self.refusals)
        self._facts[group] = facts

    # ---- one store -----------------------------------------------------------------------------
    def _store_bundle(self, st: Mapping[str, Any], okey: str, slug: str, tag: str) -> Dict[str, Any]:
        """The named files of one store directory, checked for presence and shape; raises :class:`_PairRefusal`."""
        top: Dict[str, Input] = st["top"]
        missing = [n for n in ("RESULT", "row_identity.json") if n not in top]
        if missing:
            raise _PairRefusal(f"the store lacks {missing}", st["dir"])
        keys = st["keys"]
        if len(keys) != 1:
            raise _PairRefusal(f"the store holds {len(keys)} store key directories under cells/ (expected exactly one)", st["dir"])
        store_key = next(iter(keys))
        files: Dict[str, Input] = keys[store_key]
        missing = [n for n in ("gen_stats.json", "manifest.json", "rows.jsonl", "outcomes") if n not in files]
        if missing:
            raise _PairRefusal(f"store {store_key} lacks {missing}", os.path.join(st["dir"], "cells", store_key))
        try:
            ck = IC.parse_cell_key(store_key)
        except AdapterRefusal as exc:
            raise _PairRefusal(str(exc.why), os.path.join(st["dir"], "cells", store_key))
        if ck.cell_id != okey:
            raise _PairRefusal(f"the store key spells object {ck.cell_id!r} but the directory slug spells {okey!r}", st["dir"])
        with open(top["RESULT"].path, "r", encoding="utf-8") as fh:
            result_lines = [ln.strip() for ln in fh if ln.strip()]
        if not result_lines or RESULT_COMPLETE not in result_lines[-1]:
            raise _PairRefusal(f"RESULT's last line does not read {RESULT_COMPLETE} ({(result_lines[-1] if result_lines else '<empty>')[:160]})", top["RESULT"].uri)
        result = parse_result_line(result_lines[-1])
        if result is None or result.get("result") != "COMPLETE":
            raise _PairRefusal(f"RESULT's last line does not parse as a m5t result line ({result_lines[-1][:160]})", top["RESULT"].uri)
        if result.get("okey") != okey or result.get("mode") != STORE_DIR or result.get("model") != MODEL_KEY:
            raise _PairRefusal(f"RESULT names {result.get('model')} {result.get('okey')} {result.get('mode')}, not {MODEL_KEY} {okey} {STORE_DIR}", top["RESULT"].uri)
        earlier = [parse_result_line(ln) or {"raw": ln} for ln in result_lines[:-1]]
        stamp = self._json(top["row_identity.json"]) or {}
        if str(stamp.get("schema")) != ROW_IDENTITY_SCHEMA:
            raise _PairRefusal(f"row_identity.json schema is {stamp.get('schema')!r}, not {ROW_IDENTITY_SCHEMA}", top["row_identity.json"].uri)
        man = self._json(files["manifest.json"]) or {}
        gs_all = self._json(files["gen_stats.json"]) or {}
        gs = gs_all.get(TASK) if isinstance(gs_all.get(TASK), Mapping) else None
        if gs is None:
            raise _PairRefusal(f"gen_stats.json carries no {TASK} block", files["gen_stats.json"].uri)
        outcomes = self._json(files["outcomes"]) or {}
        if str(outcomes.get("task")) != TASK or not isinstance(outcomes.get("items"), list):
            raise _PairRefusal(f"the outcome record is not {TASK}'s (task {outcomes.get('task')!r})", files["outcomes"].uri)
        summary = self._json(files["summary"]) if "summary" in files else None
        # the rows: scored `acc` rows and `unscored` rows (item -> (value, line, extra, created_at)); the aggregate rows' extra and values
        scored: Dict[str, Tuple[Optional[float], int, Dict[str, Any], Optional[float]]] = {}
        unscored: Dict[str, Tuple[Optional[float], int, Dict[str, Any], Optional[float]]] = {}
        agg_extra: Dict[str, Any] = {}
        agg_values: Dict[str, Any] = {}
        consts: Dict[str, set] = {}
        for ln, r in IC.iter_jsonl(files["rows.jsonl"].path):
            if str(r.get("task") or "") != TASK:
                continue
            ex = r.get("extra_json")
            try:
                exd = json.loads(ex) if isinstance(ex, str) and ex else (ex if isinstance(ex, Mapping) else {})
            except ValueError:
                exd = {}
            metric = str(r.get("metric"))
            if r.get("item_id") == "__aggregate__":
                agg_extra.update({k: v for k, v in exd.items() if k in ("object_artifact_sha256", "producer_tree_sha", "route_kind", "sampling_mode", "view", "bank_view")})
                agg_values[metric] = LC.num_or_none(r.get("value"))
                continue
            iid = str(r.get("item_id"))
            if metric == HEADLINE_METRIC:
                if iid in scored:
                    raise _PairRefusal(f"rows.jsonl carries item {iid} twice", files["rows.jsonl"].uri)
                scored[iid] = (float(r.get("value")), ln, exd, LC.num_or_none(r.get("created_at")))
            elif metric == ITEM_LABEL_UNSCORED:
                if iid in unscored:
                    raise _PairRefusal(f"rows.jsonl carries unscored item {iid} twice", files["rows.jsonl"].uri)
                unscored[iid] = (None, ln, exd, LC.num_or_none(r.get("created_at")))
            else:
                continue
            for k in ("slurm_job_id", "env_digest", "model_id", "model_revision", "harness_config_hash", "intervention_hash", "created_at"):
                v = r.get(k)
                if v is not None:
                    consts.setdefault(k, set()).add(v if k != "created_at" else float(v))
        controls_summary = self._json(st["summary"]) if st.get("summary") is not None else None
        return {"dir": st["dir"], "tag": tag, "slug": slug, "store_key": store_key, "ck": ck, "top": top, "files": files, "manifest": man, "gen_stats": gs,
                "outcomes": outcomes, "summary": summary, "controls_summary": controls_summary, "scored": scored, "unscored": unscored, "agg_extra": agg_extra,
                "agg_values": agg_values, "consts": consts, "result": result, "result_last": result_lines[-1], "result_earlier": earlier, "stamp": stamp}

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
            if at_cap and str(it.get("stop_reason") or "") not in IC.AT_CAP_FINISH_REASONS:
                raise _PairRefusal(f"item {iid}: at_cap with stop_reason {it.get('stop_reason')!r}", bundle["files"]["outcomes"].uri)
            out[iid] = {"at_cap": at_cap, "scored": scored, "value": (float(v) if scored else None), "stop_reason": it.get("stop_reason"),
                        "completion_tokens": LC.int_or_none(it.get("completion_tokens")), "seed": LC.int_or_none(it.get("seed")),
                        "think_block_unterminated": it.get("think_block_unterminated"), "unparseable": it.get("unparseable"),
                        "sample_idx": LC.int_or_none(it.get("sample_idx"))}
        return out

    def _items(self, base: Mapping[str, Any]) -> Dict[str, Any]:
        """The 500 items of the one pass from the outcome record x ``rows.jsonl`` x the producer's aggregates; every disagreement between the
        files is a refusal of the (tag, object key)."""
        items = self._outcome_items(base)
        if len(items) != N_ITEMS:
            raise _PairRefusal(f"the outcome record holds {len(items)} items, not {N_ITEMS}", base["files"]["outcomes"].uri)
        want = {iid: it["value"] for iid, it in items.items() if it["scored"]}
        rows_scored = base["scored"]
        if set(rows_scored) != set(want) or any(rows_scored[i][0] != want[i] for i in want):
            raise _PairRefusal(f"rows.jsonl's {len(rows_scored)} scored acc rows do not reproduce the outcome record's {len(want)} scored items", base["files"]["rows.jsonl"].uri)
        want_unscored = {iid for iid, it in items.items() if not it["scored"]}
        if set(base["unscored"]) != want_unscored:
            raise _PairRefusal(f"rows.jsonl's {len(base['unscored'])} unscored rows do not name the outcome record's {len(want_unscored)} unscored items", base["files"]["rows.jsonl"].uri)
        n_scored = len(want)
        correct = int(round(sum(want.values())))
        acc = (correct / n_scored) if n_scored else None
        at_cap = {iid for iid, it in items.items() if it["at_cap"]}
        n_at_cap_unresolved = sum(1 for it in items.values() if it["at_cap"] and not it["scored"])
        n_scored_at_cap = sum(1 for it in items.values() if it["at_cap"] and it["scored"])
        n_unscored_below_cap = sum(1 for it in items.values() if not it["at_cap"] and not it["scored"])
        if n_at_cap_unresolved + n_unscored_below_cap != N_ITEMS - n_scored:
            raise _PairRefusal("the item classes do not partition the unscored items", base["files"]["outcomes"].uri)
        gs = base["gen_stats"]
        self._check_gen_stats(gs, N_ITEMS, n_scored, len(at_cap), acc, base["files"]["gen_stats.json"].uri)
        oc = base["outcomes"]
        got = (LC.int_or_none(oc.get("n_attempted_items")), LC.int_or_none(oc.get("n_scored")), LC.int_or_none(oc.get("n_unscored")), LC.int_or_none(oc.get("n_at_cap")),
               LC.int_or_none(oc.get("max_tokens")))
        if got != (N_ITEMS, n_scored, N_ITEMS - n_scored, len(at_cap), CAP):
            raise _PairRefusal(f"the outcome record's header (n_attempted_items {got[0]}, n_scored {got[1]}, n_unscored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}) does not "
                               f"reproduce its items (n {N_ITEMS}, n_scored {n_scored}, n_unscored {N_ITEMS - n_scored}, n_at_cap {len(at_cap)}, cap {CAP})", base["files"]["outcomes"].uri)
        if str(oc.get("thinking") or "") != self.THINKING_SPELLING:
            raise _PairRefusal(f"the outcome record's thinking is {oc.get('thinking')!r}, not {self.THINKING_SPELLING!r}", base["files"]["outcomes"].uri)
        agg_acc, agg_n = base["agg_values"].get("acc_aggregate"), base["agg_values"].get("n")
        if "acc_aggregate" in base["agg_values"] and not ((agg_acc is None and acc is None) or (agg_acc is not None and acc is not None and abs(agg_acc - acc) < 1e-9)):
            raise _PairRefusal(f"rows.jsonl's acc_aggregate {agg_acc} is not correct / scored {acc}", base["files"]["rows.jsonl"].uri)
        if "n" in base["agg_values"] and LC.int_or_none(agg_n) != N_ITEMS:
            raise _PairRefusal(f"rows.jsonl's aggregate n {agg_n} is not {N_ITEMS}", base["files"]["rows.jsonl"].uri)
        if base.get("summary") is not None:
            s = base["summary"]
            sgot = (LC.int_or_none(s.get("n")), LC.int_or_none(s.get("n_scored")), LC.int_or_none(s.get("n_at_cap")), LC.int_or_none(s.get("max_tokens")), LC.num_or_none(s.get("acc")))
            s_ok = sgot[:4] == (N_ITEMS, n_scored, len(at_cap), CAP) and ((sgot[4] is None and acc is None) or (sgot[4] is not None and acc is not None and abs(sgot[4] - acc) < 1e-9))
            if not s_ok:
                raise _PairRefusal(f"summary/{SUMMARY_FILE} (n {sgot[0]}, n_scored {sgot[1]}, n_at_cap {sgot[2]}, max_tokens {sgot[3]}, acc {sgot[4]}) does not reproduce the outcome record",
                                   base["files"]["summary"].uri)
        counts = {"n_scored": n_scored, "correct": correct, "acc": acc, "n_unscored": N_ITEMS - n_scored, "n_at_cap": len(at_cap), "n_at_cap_unresolved": n_at_cap_unresolved,
                  "n_scored_at_cap": n_scored_at_cap, "n_unscored_below_cap": n_unscored_below_cap}
        return {"items": items, "counts": counts, "at_cap_ids": at_cap}

    def _check_gen_stats(self, gs: Mapping[str, Any], n: int, n_scored: int, n_at_cap: int, acc: Optional[float], uri: str) -> None:
        got = (LC.int_or_none(gs.get("n")), LC.int_or_none(gs.get("n_scored")), LC.int_or_none(gs.get("n_unscored")), LC.int_or_none(gs.get("n_at_cap")), LC.int_or_none(gs.get("max_tokens")))
        want = (n, n_scored, n - n_scored, n_at_cap, CAP)
        gacc = LC.num_or_none(gs.get("acc"))
        acc_ok = (acc is None and (gacc is None or n_scored == 0)) or (acc is not None and gacc is not None and abs(float(gacc) - acc) < 1e-9)
        if got != want or not acc_ok:
            raise _PairRefusal(f"gen_stats.json (n {got[0]}, n_scored {got[1]}, n_unscored {got[2]}, n_at_cap {got[3]}, max_tokens {got[4]}, acc {gacc}) does not reproduce the outcome "
                               f"record (n {want[0]}, n_scored {want[1]}, n_unscored {want[2]}, n_at_cap {want[3]}, cap {want[4]}, acc {acc})", uri)
        if str(gs.get("thinking") or "") != self.THINKING_SPELLING:
            raise _PairRefusal(f"gen_stats.json:thinking is {gs.get('thinking')!r}, not {self.THINKING_SPELLING!r}", uri)
        if str(gs.get("task") or "") != TASK:
            raise _PairRefusal(f"gen_stats.json:task is {gs.get('task')!r}, not {TASK!r}", uri)

    # ---- the producer's records -------------------------------------------------------------------
    @staticmethod
    def _three_numbers_row(three: Mapping[str, Any], tag: str, slug: str) -> Optional[Dict[str, Any]]:
        row = (three.get("rows") or {}).get(f"{tag}/{slug}")
        return dict(row, _key=f"{tag}/{slug}") if isinstance(row, Mapping) else None

    @staticmethod
    def _final_state_row(fstate: Mapping[str, Any], tag: str, slug: str) -> Optional[Dict[str, Any]]:
        row = (fstate.get("stores") or {}).get(f"{tag}/{slug}/{STORE_DIR}")
        return dict(row, _key=f"{tag}/{slug}/{STORE_DIR}") if isinstance(row, Mapping) else None

    def _loop_split(self, tn: Mapping[str, Any], items: Mapping[str, Any], where: str) -> Dict[str, Any]:
        """The producer's loop / eligible split of the at-cap items (INFORMATIONAL): ``third_state.loop_eligible_split`` and
        ``degeneracy_by_item`` — checked against the outcome record's at-cap set; the rule's text carried."""
        ts = tn.get("third_state") if isinstance(tn.get("third_state"), Mapping) else {}
        split = ts.get("loop_eligible_split") if isinstance(ts.get("loop_eligible_split"), Mapping) else None
        at_cap: Set[str] = items["at_cap_ids"]
        if split is None:
            return {"present": False, "loops": set(), "eligible": set(), "rule": None, "by_item": {}, "informational": None, "note": None}
        loops = {str(x) for x in (split.get("loop_ids") or [])}
        eligible = {str(x) for x in (split.get("eligible_ids") or [])}
        bad: List[str] = []
        if LC.int_or_none(split.get("loops")) != len(loops) or LC.int_or_none(split.get("eligible")) != len(eligible):
            bad.append(f"loops {split.get('loops')} / eligible {split.get('eligible')} do not count loop_ids ({len(loops)}) / eligible_ids ({len(eligible)})")
        if not loops <= at_cap or not eligible <= at_cap or (loops & eligible) or (loops | eligible) != at_cap:
            bad.append(f"loop_ids ({len(loops)}) and eligible_ids ({len(eligible)}) do not partition the outcome record's at-cap items ({len(at_cap)})")
        by_item = tn.get("degeneracy_by_item") if isinstance(tn.get("degeneracy_by_item"), Mapping) else {}
        if by_item:
            if set(map(str, by_item)) != at_cap:
                bad.append(f"degeneracy_by_item names {len(by_item)} items, the outcome record {len(at_cap)} at-cap items")
            else:
                flagged = {iid for iid, f in by_item.items() if isinstance(f, Mapping) and bool(f.get("degenerate"))}
                if flagged != loops:
                    bad.append(f"degeneracy_by_item marks {len(flagged)} items degenerate, loop_ids names {len(loops)}")
        if bad:
            raise _PairRefusal(f"{THREE_NUMBERS_FILE} row {tn.get('_key')}: the loop / eligible split disagrees with the store: " + "; ".join(bad), where)
        return {"present": True, "loops": loops, "eligible": eligible, "rule": split.get("rule"), "by_item": dict(by_item), "informational": split.get("informational"),
                "note": split.get("note")}

    @staticmethod
    def _cross_check_three(tn: Mapping[str, Any], base: Mapping[str, Any], items: Mapping[str, Any], split: Mapping[str, Any], obj_sha: Optional[str], tag: str, selection: str,
                           seed_n: int, okey: str, tree: str) -> List[str]:
        """The producer's three-number row against the re-derivation and the store: every count, the value, the bound, the third state, and the
        store block's identity fields."""
        c = items["counts"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False) -> None:
            ok = rounded_equal(theirs, mine) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        eq("status", str(tn.get("status")), "complete")
        eq("tag", str(tn.get("tag")), tag)
        eq("view", str(tn.get("view")), selection)
        eq("cell_id", str(tn.get("cell_id")), okey)
        eq("cell_slug", str(tn.get("cell_slug")), base["slug"])
        eq("out_root", str(tn.get("out_root")), f"{STORES_HEAD}/{tag}/{base['slug']}/{STORE_DIR}")
        eq("n_attempted", LC.int_or_none(tn.get("n_attempted")), N_ITEMS)
        eq("n_scored_final", LC.int_or_none(tn.get("n_scored_final")), c["n_scored"])
        eq("correct_final", LC.int_or_none(tn.get("correct_final")), c["correct"])
        eq("value_correct_over_scored", LC.num_or_none(tn.get("value_correct_over_scored")), c["acc"], rounded=True)
        eq(BOUND_SPELLING, LC.num_or_none(tn.get(BOUND_SPELLING)), c["correct"] / N_ITEMS, rounded=True)
        ts = tn.get("third_state") if isinstance(tn.get("third_state"), Mapping) else {}
        eq("third_state.n_at_cap", LC.int_or_none(ts.get("n_at_cap")), c["n_at_cap"])
        eq("third_state.share_at_cap", LC.num_or_none(ts.get("share_at_cap")), c["n_at_cap"] / N_ITEMS, rounded=True)
        eq("third_state.n_unresolved_at_cap", LC.int_or_none(ts.get("n_unresolved_at_cap")), c["n_at_cap_unresolved"])
        eq("third_state.share_unresolved", LC.num_or_none(ts.get("share_unresolved")), c["n_at_cap_unresolved"] / N_ITEMS, rounded=True)
        eq("third_state.n_scored_at_cap", LC.int_or_none(ts.get("n_scored_at_cap")), c["n_scored_at_cap"])
        eq("third_state.n_unscored_below_cap", LC.int_or_none(ts.get("n_unscored_below_cap")), c["n_unscored_below_cap"])
        if "unscored_below_cap_ids" in ts:
            eq("third_state.unscored_below_cap_ids", sorted(str(x) for x in (ts.get("unscored_below_cap_ids") or [])),
               sorted(iid for iid, it in items["items"].items() if not it["at_cap"] and not it["scored"]))
        od = tn.get("outcome_doc") if isinstance(tn.get("outcome_doc"), Mapping) else {}
        eq("outcome_doc.n_scored", LC.int_or_none(od.get("n_scored")), c["n_scored"])
        eq("outcome_doc.n_at_cap", LC.int_or_none(od.get("n_at_cap")), c["n_at_cap"])
        eq("outcome_doc.max_tokens", LC.int_or_none(od.get("max_tokens")), CAP)
        eq("outcome_doc.producer_tree_sha", str(od.get("producer_tree_sha")), tree)
        st = tn.get("store") if isinstance(tn.get("store"), Mapping) else {}
        man = base["manifest"]
        eq("store.producer_tree_sha", str(st.get("producer_tree_sha")), tree)
        eq("store.serving_engine", str(st.get("serving_engine")), str(man.get("serving_engine")))
        eq("store.serving_identity", str(st.get("serving_identity")), str(man.get("serving_identity")))
        eq("store.cell_id", str(st.get("cell_id")), okey)
        eq("store.harness_cell_id", str(st.get("harness_cell_id")), str(man.get("harness_cell_id")))
        eq("store.oos_experiment", st.get("oos_experiment"), man.get("oos_experiment"))
        eq("store.thinking_mode", str(st.get("thinking_mode")), "on")
        eq("store.sampling_mode", str(st.get("sampling_mode")), STORE_DIR)
        eq("store.seed_policy", str(st.get("seed_policy")), INSPECT_SEED_POLICY)
        eq("store.seed_base", LC.int_or_none(st.get("seed_base")), seed_n)
        eq("store.seed_rule", str(st.get("seed_rule")), str(((man.get("seeds") or {}).get("inspect") or {}).get("rule")))
        eq("store.model_id", str(st.get("model_id")), str(man.get("model_id")))
        eq("store.model_revision", str(st.get("model_revision")), str(man.get("model_revision")))
        eq("store.bank_view", str(st.get("bank_view")), str((man.get("route") or {}).get("bank_view")))
        eq("store.slurm_job_id", str(st.get("slurm_job_id")), str((man.get("row_constants") or {}).get("slurm_job_id")))
        eq("store.result", str(st.get("result")), "COMPLETE")
        sampling = st.get("sampling") if isinstance(st.get("sampling"), Mapping) else {}
        gs = base["gen_stats"]
        for k in DP.PARAM_KEYS:
            if k in sampling and gs.get(k) is not None:
                eq(f"store.sampling.{k}", LC.num_or_none(sampling.get(k)), LC.num_or_none(gs.get(k)))
        design = tn.get("design") if isinstance(tn.get("design"), Mapping) else {}
        eq("design.view", str(design.get("view")), selection)
        eq("design.inspect_seed_base", LC.int_or_none(design.get("inspect_seed_base")), seed_n)
        eq("design.object_sha256", design.get("object_sha256"), obj_sha)
        eq("design.tuple_name", design.get("tuple_name"), man.get("oos_experiment"))
        eq("design.bank_view", str(design.get("bank_view")), str((man.get("route") or {}).get("bank_view")))
        return bad

    @staticmethod
    def _cross_check_final_state(fs: Mapping[str, Any], items: Mapping[str, Any], obj_sha: Optional[str], tag: str, selection: str, seed_n: int, tree: str) -> List[str]:
        """The producer's final-state entry (the reader tree's re-derivation) against the re-derivation: the counts, the identity and the 500 items."""
        c = items["counts"]
        bad: List[str] = []

        def eq(name: str, theirs, mine, rounded: bool = False) -> None:
            ok = rounded_equal(theirs, mine) if rounded else (theirs == mine)
            if not ok:
                bad.append(f"{name}: producer {theirs!r}, re-derived {mine!r}")

        eq("status", str(fs.get("status")), "complete")
        eq("tag", str(fs.get("tag")), tag)
        eq("view", str(fs.get("view")), selection)
        eq("n_items", LC.int_or_none(fs.get("n_items")), N_ITEMS)
        eq("n_scored", LC.int_or_none(fs.get("n_scored")), c["n_scored"])
        eq("n_correct", LC.int_or_none(fs.get("n_correct")), c["correct"])
        eq("acc_over_scored", LC.num_or_none(fs.get("acc_over_scored")), c["acc"], rounded=True)
        eq("acc_over_attempted", LC.num_or_none(fs.get("acc_over_attempted")), c["correct"] / N_ITEMS, rounded=True)
        eq("n_at_cap_unresolved", LC.int_or_none(fs.get("n_at_cap_unresolved")), c["n_at_cap_unresolved"])
        eq("n_unscored_below_cap", LC.int_or_none(fs.get("n_unscored_below_cap")), c["n_unscored_below_cap"])
        eq("n_scored_at_cap", LC.int_or_none(fs.get("n_scored_at_cap")), c["n_scored_at_cap"])
        eq("n_at_cap_total", LC.int_or_none(fs.get("n_at_cap_total")), c["n_at_cap"])
        eq("rung", fs.get("rung"), None)
        eq("producer_tree_sha", str(fs.get("producer_tree_sha")), tree)
        eq("seed_base", LC.int_or_none(fs.get("seed_base")), seed_n)
        if "row_identity_complete" in fs:
            eq("row_identity_complete", bool(fs.get("row_identity_complete")), True)
        if "row_identity_manifest_object_sha256" in fs:
            eq("row_identity_manifest_object_sha256", fs.get("row_identity_manifest_object_sha256"), obj_sha)
        fitems = fs.get("items") if isinstance(fs.get("items"), Mapping) else None
        if fitems is not None:
            mine = items["items"]
            if set(map(str, fitems)) != set(mine):
                bad.append(f"items: the record names {len(fitems)} items, the outcome record {len(mine)}")
            else:
                n_bad = 0
                for iid, f in fitems.items():
                    it = mine[str(iid)]
                    theirs = (bool(f.get("scored")), bool(f.get("at_cap")), (bool(f.get("correct")) if it["scored"] else None), str(f.get("stop_reason")))
                    ours = (it["scored"], it["at_cap"], (it["value"] == 1.0 if it["scored"] else None), str(it["stop_reason"]))
                    if theirs != ours:
                        n_bad += 1
                if n_bad:
                    bad.append(f"items: {n_bad} item(s) disagree with the outcome record on scored / at_cap / correct / stop_reason")
        return bad

    @staticmethod
    def _cross_check_summary(summary: Mapping[str, Any], base: Mapping[str, Any], items: Mapping[str, Any], obj_sha: Optional[str], okey: str, tree: str) -> List[str]:
        """The producer's ``controls_summary.json`` beside the store against the store: the key, the tree, the configuration hash, the object and the reading."""
        bad: List[str] = []
        c = items["counts"]
        src = (summary.get("sources") or {}).get(TASK) if isinstance(summary.get("sources"), Mapping) else None
        if not isinstance(src, Mapping):
            bad.append(f"sources[{TASK}] is absent")
        else:
            if str(src.get("cell_key") or "") != base["store_key"]:
                bad.append(f"sources[{TASK}].cell_key {src.get('cell_key')!r} is not the store key {base['store_key']}")
            if str(src.get("producer_tree_sha") or "") != tree:
                bad.append(f"sources[{TASK}].producer_tree_sha {str(src.get('producer_tree_sha'))[:12]!r} is not the manifest's tree {tree[:12]}")
            if str(src.get("harness_config_hash") or "") != str(base["manifest"].get("harness_config_hash") or ""):
                bad.append(f"sources[{TASK}].harness_config_hash {src.get('harness_config_hash')!r} is not the manifest's {base['manifest'].get('harness_config_hash')!r}")
        prov = summary.get("provenance") if isinstance(summary.get("provenance"), Mapping) else {}
        if (prov.get("artifact_sha256") or None) != obj_sha:
            bad.append(f"provenance.artifact_sha256 {str(prov.get('artifact_sha256'))[:16]!r} is not the resolved object {str(obj_sha)[:16]}")
        if str(summary.get("cell_id") or "") != okey:
            bad.append(f"cell_id {summary.get('cell_id')!r} is not {okey}")
        g = (summary.get("controls") or {}).get(TASK) if isinstance(summary.get("controls"), Mapping) else None
        if not isinstance(g, Mapping):
            bad.append(f"controls[{TASK}] is absent")
            return bad
        gacc = LC.num_or_none(g.get("acc"))
        if (c["acc"] is None) != (gacc is None) or (c["acc"] is not None and abs(float(gacc) - c["acc"]) > 1e-9):
            bad.append(f"controls[{TASK}].acc {gacc} is not correct / scored {c['acc']}")
        if (LC.int_or_none(g.get("n")), LC.int_or_none(g.get("n_scored")), LC.int_or_none(g.get("n_at_cap")), LC.int_or_none(g.get("max_tokens")), str(g.get("thinking") or ""), str(g.get("task") or "")) != \
                (N_ITEMS, c["n_scored"], c["n_at_cap"], CAP, "on", TASK):
            bad.append(f"controls[{TASK}] (n {g.get('n')}, n_scored {g.get('n_scored')}, n_at_cap {g.get('n_at_cap')}, max_tokens {g.get('max_tokens')}, thinking {g.get('thinking')!r}, "
                       f"task {g.get('task')!r}) is not the store's (n {N_ITEMS}, n_scored {c['n_scored']}, n_at_cap {c['n_at_cap']}, max_tokens {CAP}, thinking 'on', task {TASK!r})")
        return bad

    @staticmethod
    def _check_launch_stamp(stamp: Mapping[str, Any], okey: str, slug: str, tag: str, selection: str, obj: LC.ResolvedObject, tree: str, where: str) -> Dict[str, Any]:
        """The launch stamp (``row_identity.json``, written at launch and never rewritten) against the store: its declared identity must be the
        resolved object's; its own verdict (``complete`` / ``problems``) is CARRIED, not trusted — the launch's check read ``route.object_artifact_sha256``,
        which a multi-adapter standup leaves null, and the reader tree's re-derivation (the final-state record) is the verdict that stands."""
        ident = stamp.get("identity") if isinstance(stamp.get("identity"), Mapping) else {}
        row = stamp.get("row") if isinstance(stamp.get("row"), Mapping) else {}
        bad: List[str] = []
        steered = obj.object_kind not in ("none", "system_instruction")
        want_sha = obj.object_id if steered else None
        if (ident.get("object_artifact_sha256") or None) != want_sha:
            bad.append(f"identity.object_artifact_sha256 {str(ident.get('object_artifact_sha256'))[:16]!r} is not the resolved object {str(want_sha)[:16]!r}")
        for k, want in (("view", selection), ("producer_tree_sha", tree), ("pin", tree), ("thinking", "on"), ("sampling_mode", STORE_DIR)):
            if str(ident.get(k) or "") != want:
                bad.append(f"identity.{k} {ident.get(k)!r} is not {want!r}")
        for k, want in (("cell", okey), ("model_key", MODEL_KEY), ("sampling_mode", STORE_DIR), ("out_root", f"{STORES_HEAD}/{tag}/{slug}/{STORE_DIR}")):
            if str(row.get(k) or "") != want:
                bad.append(f"row.{k} {row.get(k)!r} is not {want!r}")
        if steered and str(row.get("staged_id") or "") != str(obj.staged_id):
            bad.append(f"row.staged_id {row.get('staged_id')!r} is not the resolved staged id {obj.staged_id!r}")
        if obj.object_kind == "system_instruction":
            si_sha = str(obj.object_id).split(":", 1)[1]
            if str(row.get("si_sha256") or "") != si_sha:
                bad.append(f"row.si_sha256 {str(row.get('si_sha256'))[:16]!r} is not the declared prompt's text sha {si_sha[:16]}")
        if bad:
            raise _PairRefusal("row_identity.json (the launch stamp) disagrees with the store on " + "; ".join(bad), where)
        return {"complete": stamp.get("complete"), "problems": list(stamp.get("problems") or []), "schema": stamp.get("schema"), "index": row.get("index"), "job_id": row.get("job_id"),
                "sampling_ref": ident.get("sampling_ref"), "route_kind": ident.get("route_kind"), "budget_class": row.get("budget_class"), "tuple_name": row.get("tuple_name"),
                "verdict_rule": "the stamp's own verdict is carried, never trusted: the launch's check read route.object_artifact_sha256 (null on a multi-adapter standup); the "
                                "reader tree's re-derivation (final_state_*.json row_identity_*) and this reader's own sha checks are the verdicts that stand"}

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

    # ---- the standup records -------------------------------------------------------------------
    def _standup(self, man: Mapping[str, Any], serving: Mapping[str, Mapping[str, Input]]) -> Dict[str, Any]:
        """The standup the manifest's ``routing_table`` names: ``serving_attestation.json`` (the node, the engine kind, ``lora_config.max_lora_rank`` —
        RECORDED, never compared) and ``standup_jobs.jsonl`` (the cluster the job ran on, the tree). Absent records are named, never guessed."""
        rel = serving_dir_rel(man.get("routing_table"))
        out: Dict[str, Any] = {"serving_dir": rel, "attestation": None, "standup_job": None, "max_lora_rank": None, "lora_target_modules": None, "max_loras": None,
                               "node": None, "cluster": None, "kind": None, "attested_max_model_len": None, "attested_max_num_seqs": None, "absent": []}
        files = serving.get(rel or "", {})
        att_inp = files.get(ATTESTATION_FILE)
        if att_inp is None:
            out["absent"].append(f"{rel or '<no routing_table>'}/{ATTESTATION_FILE}")
        else:
            att = self._json(att_inp) or {}
            lc = att.get("lora_config") if isinstance(att.get("lora_config"), Mapping) else {}
            exp = att.get("expected") if isinstance(att.get("expected"), Mapping) else {}
            out.update(attestation=LC.rel_home(att_inp.uri.rsplit("#", 1)[0]), max_lora_rank=LC.int_or_none(lc.get("max_lora_rank")), lora_target_modules=lc.get("target_modules"),
                       max_loras=LC.int_or_none(lc.get("max_loras")), node=att.get("hostname"), kind=att.get("kind"), attested_model=att.get("model"),
                       attested_max_model_len=LC.int_or_none(exp.get("max_model_len")), attested_max_num_seqs=LC.int_or_none(exp.get("max_num_seqs")),
                       lora_loaded_count=LC.int_or_none(exp.get("lora_loaded_count")), adapters=sorted((att.get("adapters") or {}).keys()) if isinstance(att.get("adapters"), Mapping) else None)
        sj_inp = files.get(STANDUP_JOBS_FILE)
        if sj_inp is None:
            out["absent"].append(f"{rel or '<no routing_table>'}/{STANDUP_JOBS_FILE}")
        else:
            last = None
            for _ln, r in IC.iter_jsonl(sj_inp.path):
                last = r
            if isinstance(last, Mapping):
                out.update(standup_job=LC.rel_home(sj_inp.uri.rsplit("#", 1)[0]), cluster=last.get("cluster"), standup_job_id=last.get("job_id"), standup_node=last.get("node"),
                           standup_tree=last.get("sha"), standup_engine=last.get("engine"))
        return out

    # ---- one (tag, object key) row -------------------------------------------------------------
    def _emit_row(self, batch: RecordBatch, ctx: AdapterContext, group: str, tag: str, seed_n: int, paired: bool, selection: str, model_id: str, inst_row: Mapping[str, Any],
                  okey: str, slug: str, st: Mapping[str, Any], pins: Mapping[str, Any], three: Mapping[str, Any], fstate: Optional[Mapping[str, Any]], lrow: Mapping[str, Any],
                  serving: Mapping[str, Mapping[str, Input]], pins_inp: Input, tn_inp: Input, fs_inp: Optional[Input], ls_inp: Input, facts: Dict[str, Any],
                  anchor_plan: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Files one (tag, object key); returns what the group's contrasts need (the run id, the items, whether steered, the declared role). ``anchor_plan``
        (:func:`_bcbw_common.anchor_plan`, given for a paired tag's non-``none`` row) names the anchor the row's contrast will pair against, so the run's
        extra spells it — the ``none`` rows file first, so the plan is final when the row is read."""
        seeds = self.seeds
        producer, now = ctx.producer, ctx.now
        base = self._store_bundle(st, okey, slug, tag)
        man = base["manifest"]
        if str(man.get("cell_id") or "") != okey:
            raise _PairRefusal(f"manifest.json names object key {man.get('cell_id')!r}, not {okey!r}", base["files"]["manifest.json"].uri)
        # -- the store pin: the admitted producer tree of the (selection, object key); a refused / partial row never files
        tree = str((man.get("producer") or {}).get("tree_sha") or "")
        if not _SHA1.match(tree):
            raise _PairRefusal("manifest.json:producer.tree_sha names no 40-hex tree", base["files"]["manifest.json"].uri)
        pkey = pin_key(selection, okey)
        meta = pins.get("_meta") if isinstance(pins.get("_meta"), Mapping) else {}
        for bucket in ("refused", "partial", "identity_refused"):
            named = meta.get(bucket) if isinstance(meta.get(bucket), Mapping) else {}
            if pkey in named:
                raise _PairRefusal(f"the store pins' _meta.{bucket} names this row ({pkey}): {json.dumps(named[pkey])[:200]}; a refused row never files", pins_inp.uri)
        pin = pins.get(pkey)
        if pin is None:
            raise _PairRefusal(f"no store pin {pkey} beside the stores; the admitted tree of this row is unnamed", pins_inp.uri)
        if str(pin) != tree:
            raise _PairRefusal(f"the store's tree {tree[:12]} is not its store pin {str(pin)[:12]} ({pkey})", pins_inp.uri)
        # -- the served model and identity
        served_hf = str(man.get("model_id") or "")
        if seeds.model_for_hf_id(served_hf) != model_id:
            raise _PairRefusal(f"the store serves {served_hf!r}, not the seed tag's model {model_id}", base["files"]["manifest.json"].uri)
        rev = str(man.get("model_revision") or "")
        pinned_rev = str((seeds.models.get(model_id) or {}).get("hf_revision") or "")
        if _SHA1.match(pinned_rev) and rev != pinned_rev:
            raise _PairRefusal(f"the store serves revision {rev[:12] or '<none>'}, not the model seed's {pinned_rev[:12]}", base["files"]["manifest.json"].uri)
        if str(man.get("serving_engine") or "") != SERVING_ENGINE:
            raise _PairRefusal(f"manifest.json:serving_engine {man.get('serving_engine')!r} is not {SERVING_ENGINE!r}", base["files"]["manifest.json"].uri)
        # -- the object: the launch set's pin key and expected sha, the harness roster / pins, the store's attestations
        route = man.get("route") if isinstance(man.get("route"), Mapping) else {}
        bank = str(route.get("bank_view") or "") or None
        try:
            obj, op = self.resolve_object(okey, selection, bank=bank)
        except IC.Unresolved as exc:
            raise _PairRefusal(f"object: {exc.why}", base["files"]["manifest.json"].uri)
        arditi = okey == ARDITI_OBJECT_KEY
        steered = obj.object_kind not in ("none", "system_instruction")
        lclass = str(lrow.get("class") or "")
        # the launch set's class grammar: the paired tag's `none` is its ANCHOR (PAIRED_ANCHOR_RULE) and a prompted key is `prompted` on either tag; only a
        # fitted key under the paired tag is the `composed` object
        want_class = "arditi" if arditi else ("anchor" if obj.object_kind == "none" else ("prompted" if obj.object_kind == "system_instruction" else ("composed" if paired else "arm")))
        if lclass != want_class:
            raise _PairRefusal(f"the launch set files this row as class {lclass!r}, the key's grammar says {want_class!r}", ls_inp.uri)
        if LC.int_or_none(lrow.get("seed_base")) != seed_n or str(lrow.get("tag") or "") != tag or str(lrow.get("cell_slug") or "") != slug:
            raise _PairRefusal(f"the launch set's row (tag {lrow.get('tag')!r}, seed_base {lrow.get('seed_base')!r}, cell_slug {lrow.get('cell_slug')!r}) is not this store's "
                               f"({tag}, {seed_n}, {slug})", ls_inp.uri)
        if str(lrow.get("m5t_out_root") or "") not in ("", f"{STORES_HEAD}/{tag}/{slug}/{STORE_DIR}"):
            raise _PairRefusal(f"the launch set's m5t_out_root {lrow.get('m5t_out_root')!r} does not spell this store's directory", ls_inp.uri)
        if str(lrow.get("paper_label") or "") != str(obj.paper_label):
            raise _PairRefusal(f"the launch set labels the row {lrow.get('paper_label')!r}, the key's grammar {obj.paper_label!r}", ls_inp.uri)
        expected_sha = lrow.get("object_sha256_expected") or None
        launch_pin_key = lrow.get("pin_key") or None
        if steered:
            if launch_pin_key != obj.staged_id:
                raise _PairRefusal(f"the launch set's pin key {launch_pin_key!r} is not the roster's staged id {obj.staged_id!r} at {op.rung if op else selection}", ls_inp.uri)
            if expected_sha != obj.object_id:
                raise _PairRefusal(f"the launch set expects object {str(expected_sha)[:16]}…, the pins name {str(obj.object_id)[:16]}… for {obj.staged_id}", ls_inp.uri)
        elif launch_pin_key is not None or expected_sha is not None:
            raise _PairRefusal(f"the launch set carries a pin key / expected sha ({launch_pin_key!r} / {str(expected_sha)[:16]!r}) for an unsteered or prompted key", ls_inp.uri)
        ident = str(man.get("serving_identity") or "")
        want_ident = STEERED_IDENTITY if steered else PASSTHROUGH_IDENTITY
        if ident != want_ident:
            raise _PairRefusal(f"serving_identity {ident!r} is not the {obj.object_kind} object's {want_ident!r}", base["files"]["manifest.json"].uri)
        fitp = man.get("fit_provenance") if isinstance(man.get("fit_provenance"), Mapping) else {}
        sha16 = str(fitp.get("object_sha") or "")
        attested = str(base["agg_extra"].get("object_artifact_sha256") or "")
        prov_sha = str((man.get("provenance") or {}).get("artifact_sha256") or "")
        by_harness_id = route.get("object_artifact_sha256_by_cell") if isinstance(route.get("object_artifact_sha256_by_cell"), Mapping) else {}
        route_sha = str(route.get("object_artifact_sha256") or "") or str(by_harness_id.get(str(man.get("harness_cell_id") or "")) or "")
        if steered:
            if not _SHA16.match(sha16) or not str(obj.object_id).startswith(sha16):
                raise _PairRefusal(f"manifest.json:fit_provenance.object_sha {sha16!r} is no prefix of the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})",
                                   base["files"]["manifest.json"].uri)
            if attested != obj.object_id:
                raise _PairRefusal(f"the rows' aggregate attests object {attested[:16] or '<none>'}…, the roster / pins {str(obj.object_id)[:16]}…", base["files"]["rows.jsonl"].uri)
            if prov_sha != obj.object_id:
                raise _PairRefusal(f"manifest.json:provenance.artifact_sha256 {prov_sha[:16] or '<none>'}… is not the pinned object {str(obj.object_id)[:16]}… (staged {obj.staged_id})",
                                   base["files"]["manifest.json"].uri)
            if route_sha != obj.object_id:
                raise _PairRefusal(f"manifest.json:route.object_artifact_sha256[_by_cell] {route_sha[:16] or '<none>'}… is not the pinned object {str(obj.object_id)[:16]}…",
                                   base["files"]["manifest.json"].uri)
        elif obj.object_kind == "system_instruction":
            sys_sha = str((man.get("resolved_config") or {}).get("system_sha") or "")
            si_sha = str(obj.object_id).split(":", 1)[1]
            if not (_SHA16.match(sys_sha) and si_sha.startswith(sys_sha)):
                raise _PairRefusal(f"manifest.json:resolved_config.system_sha {sys_sha!r} is no prefix of the declared prompt's text sha {si_sha[:16]}…", base["files"]["manifest.json"].uri)
            if sha16 or attested or prov_sha or route_sha:
                raise _PairRefusal("the prompted store attests an object sha", base["files"]["manifest.json"].uri)
        elif sha16 or attested or prov_sha or route_sha:
            raise _PairRefusal("the unsteered store attests an object sha", base["files"]["manifest.json"].uri)
        if not arditi and steered and bank != selection:
            raise _PairRefusal(f"manifest.json:route.bank_view {bank!r} is not the tag's selection {selection!r}", base["files"]["manifest.json"].uri)
        roster_paired_of = None
        if paired:
            roster, _pins = self._roster_pins()
            blk = ((roster.get("targets") or {}).get(MODEL_KEY) or {}).get("views", {}).get(selection) or {}
            roster_paired_of = blk.get("paired_of")
            if str(roster_paired_of or "") != f"s{seed_n}":
                raise _PairRefusal(f"the roster's selection {selection} says paired_of {roster_paired_of!r}; the tag says s{seed_n}", self.roster_path)
        # -- the seed rule: the replicate policy at the tag's base, read off the store
        rep = ((man.get("seeds") or {}).get("inspect") or {}).get("replicate") if isinstance(man.get("seeds"), Mapping) else None
        rep = rep if isinstance(rep, Mapping) else {}
        sb = LC.int_or_none(rep.get("seed_base"))
        if str(rep.get("policy")) != INSPECT_SEED_POLICY or sb != seed_n:
            raise _PairRefusal(f"manifest.json:seeds.inspect.replicate is policy {rep.get('policy')!r} at seed_base {rep.get('seed_base')!r}; the tag's rule is {INSPECT_SEED_POLICY} at {seed_n}",
                               base["files"]["manifest.json"].uri)
        seed_rule_text = str(((man.get("seeds") or {}).get("inspect") or {}).get("rule") or "")
        if not seed_rule_text:
            raise _PairRefusal("manifest.json:seeds.inspect.rule names no seed rule", base["files"]["manifest.json"].uri)
        # -- thinking ON and the one cap on every record that spells them
        thinking = man.get("thinking") if isinstance(man.get("thinking"), Mapping) else {}
        if str(thinking.get("mode") or "") != self.THINKING_SPELLING or thinking.get("enable_thinking") is False:
            raise _PairRefusal(f"manifest.json:thinking.mode {thinking.get('mode')!r} is not {self.THINKING_SPELLING!r}", base["files"]["manifest.json"].uri)
        tc = man.get("tinycap") if isinstance(man.get("tinycap"), Mapping) else {}
        cap_by_task = tc.get("max_tokens_by_task") if isinstance(tc.get("max_tokens_by_task"), Mapping) else {}
        if LC.int_or_none(cap_by_task.get(TASK)) not in (None, CAP) or LC.int_or_none(tc.get("math500_max_tokens")) not in (None, CAP):
            raise _PairRefusal(f"manifest.json:tinycap caps {cap_by_task.get(TASK)!r} / {tc.get('math500_max_tokens')!r} are not the one pass at {CAP}", base["files"]["manifest.json"].uri)
        if LC.int_or_none(tc.get("math500_retry_max_tokens")) not in (None, 0):
            raise _PairRefusal(f"manifest.json:tinycap.math500_retry_max_tokens {tc.get('math500_retry_max_tokens')!r} spells a retry; the instrument is one pass", base["files"]["manifest.json"].uri)
        # -- the launch stamp's declared identity
        stamp = self._check_launch_stamp(base["stamp"], okey, slug, tag, selection, obj, tree, base["top"]["row_identity.json"].uri)
        # -- items: the outcome record x rows.jsonl x the aggregates
        items = self._items(base)
        counts = items["counts"]
        # -- the producer's records: the three-number row (REQUIRED), the final-state entry (when present), the controls summary (when present)
        tn = self._three_numbers_row(three, tag, slug)
        if tn is None:
            raise _PairRefusal(f"{THREE_NUMBERS_FILE} carries no row for {tag}/{slug}", tn_inp.uri)
        split = self._loop_split(tn, items, tn_inp.uri)
        obj_sha = obj.object_id if steered else None
        bad = self._cross_check_three(tn, base, items, split, obj_sha, tag, selection, seed_n, okey, tree)
        if bad:
            raise _PairRefusal(f"{THREE_NUMBERS_FILE} row {tn['_key']} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), tn_inp.uri)
        fs_row = self._final_state_row(fstate, tag, slug) if fstate is not None else None
        if fs_row is not None:
            bad = self._cross_check_final_state(fs_row, items, obj_sha, tag, selection, seed_n, tree)
            if bad:
                raise _PairRefusal(f"{FINAL_STATE_FILE} entry {fs_row['_key']} disagrees with the re-derivation on {len(bad)} number(s): " + "; ".join(bad), fs_inp.uri)
        if base.get("controls_summary") is not None:
            bad = self._cross_check_summary(base["controls_summary"], base, items, obj_sha, okey, tree)
            if bad:
                raise _PairRefusal(f"{CONTROLS_SUMMARY_FILE} disagrees with the store on {len(bad)} fact(s): " + "; ".join(bad), st["summary"].uri)
        # -- decode: the sent numbers against the registered policy row and the instrument's columns
        gs = base["gen_stats"]
        block = {k: gs.get(k) for k in DP.PARAM_KEYS}
        sent = {k: v for k, v in block.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        inst_numbers = {k: inst_row.get(f"decode_{k}") for k in DP.PARAM_KEYS}
        inst_match = "match" if self._numbers_agree(block, inst_numbers) else "mismatch"
        policy, policy_why = self._match_policy(model_id, block)
        man_sampling = man.get("sampling") if isinstance(man.get("sampling"), Mapping) else {}
        if man_sampling and not self._numbers_agree(block, {k: man_sampling.get(k) for k in DP.PARAM_KEYS}):
            raise _PairRefusal(f"gen_stats.json sent {RP._numbers(block)}, manifest.json:sampling spells {RP._numbers(man_sampling)}", base["files"]["gen_stats.json"].uri)
        # -- time, job, substrate, standup
        attempts = [a for a in (man.get("resume_attempts") or []) if isinstance(a, Mapping)]
        completed = [a for a in attempts if bool(a.get("completed"))]
        if attempts and not completed:
            raise _PairRefusal("manifest.json:resume_attempts names no completed attempt", base["files"]["manifest.json"].uri)
        accepted = completed[-1] if completed else None
        if accepted is not None and str(accepted.get("tree_sha") or "") != tree:
            raise _PairRefusal(f"the completed attempt's tree {str(accepted.get('tree_sha'))[:12]} is not the manifest's {tree[:12]}", base["files"]["manifest.json"].uri)
        when_final = IC.utc_from_iso(base["result"].get("ts")) or self._rows_time(base, max)
        when_start = (IC.utc_from_iso(accepted.get("ts")) if accepted is not None else None) or self._rows_time(base, min) or when_final
        if when_final is None:
            raise _PairRefusal("RESULT carries no parsable timestamp and rows.jsonl no created_at", base["top"]["RESULT"].uri)
        job = str((man.get("row_constants") or {}).get("slurm_job_id") or "")
        if not job or job != str(base["result"].get("job")):
            raise _PairRefusal(f"manifest.json:row_constants.slurm_job_id {job!r} and RESULT's job {base['result'].get('job')!r} disagree", base["files"]["manifest.json"].uri)
        rows_jobs = base["consts"].get("slurm_job_id") or set()
        if rows_jobs and rows_jobs != {job}:
            raise _PairRefusal(f"rows.jsonl carries slurm_job_id {sorted(rows_jobs)}, the manifest {job}", base["files"]["rows.jsonl"].uri)
        witness = SC.substrate_of_routing_table(str(man.get("routing_table") or ""))
        if witness is None:
            raise _PairRefusal(f"manifest.json:routing_table {str(man.get('routing_table'))!r} is on no cluster's shared root (/workspace-vast/ = cluster A, /workspace/ = cluster B)",
                               base["files"]["manifest.json"].uri)
        standup = self._standup(man, serving)
        if standup.get("cluster") and str(standup["cluster"]) == witness[0]:
            substrate = (witness[0], "recorded", LC.confidence_entry("recorded", f"{standup['standup_job']}:cluster = {standup['cluster']} (the standup's own record); manifest.json:routing_table "
                                                                                   f"on the /{'workspace-vast' if witness[0] == 'cluster_a' else 'workspace'}/ shared root agrees"))
        else:
            substrate = witness
        # -- the run
        geometry = man.get("geometry") if isinstance(man.get("geometry"), Mapping) else {}
        frameworks = man.get("frameworks") if isinstance(man.get("frameworks"), Mapping) else {}
        instrument_id = str(inst_row["instrument_id"])
        producer_key = producer_key_for(tag, base["store_key"])
        rid = run_id_for(tag, base["store_key"], okey, instrument_id)
        n_scored, correct = counts["n_scored"], counts["correct"]
        acc = counts["acc"]
        bound_lo = correct / N_ITEMS
        bound_hi = (correct + counts["n_at_cap_unresolved"]) / N_ITEMS
        loops_unresolved = sum(1 for iid, it in items["items"].items() if it["at_cap"] and not it["scored"] and iid in split["loops"])
        n_ceiling = counts["n_at_cap_unresolved"] - loops_unresolved
        off_task = tn.get("off_task") if isinstance(tn.get("off_task"), Mapping) else {}
        three_block = {"n_attempted": N_ITEMS, "n_scored_final": n_scored, "correct_final": correct, "acc_over_scored": acc, "acc_over_attempted": bound_lo,
                       "acc_over_attempted_upper_bound": bound_hi, "n_unresolved_neither_hit_nor_miss": counts["n_at_cap_unresolved"], "n_at_cap_final": counts["n_at_cap"],
                       "share_at_cap": counts["n_at_cap"] / N_ITEMS, "n_scored_at_cap_final": counts["n_scored_at_cap"], "n_unscored_final": counts["n_unscored"],
                       "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_loops": len(split["loops"]), "n_loops_unresolved": loops_unresolved, "n_at_cap_ceiling": n_ceiling,
                       "n_eligible_informational": len(split["eligible"]), "loop_rule": split["rule"], "loop_split_informational": split["informational"], "loop_split_note": split["note"],
                       "at_cap_rule": "the outcome record's at_cap (stop_reason == max_tokens at the request's cap; never re-tokenized)",
                       "producer_record": LC.rel_home(tn_inp.uri.rsplit("#", 1)[0]), "producer_record_schema": THREE_NUMBERS_SCHEMA, "producer_record_agrees": True,
                       "final_state_record": (LC.rel_home(fs_inp.uri.rsplit("#", 1)[0]) if fs_row is not None else None), "final_state_record_agrees": (True if fs_row is not None else None),
                       "final_state_reader_tree_sha": (fstate.get("reader_tree_sha") if fs_row is not None else None),
                       "producer_off_task": (off_task.get("counts") if isinstance(off_task.get("counts"), Mapping) else None),
                       "producer_off_task_share_structural": LC.num_or_none(off_task.get("share_off_task_structural")),
                       "producer_off_task_rule": (off_task.get("rule") if isinstance(off_task.get("rule"), Mapping) else None),
                       "producer_contrast": (tn.get("contrast") if isinstance(tn.get("contrast"), Mapping) else None),
                       "unresolved_definition": "item-wise: an at-cap item with no answer at the one pass's cap (16,384, the standing truncation rule's ceiling); no cap step exists on MATH-500 "
                                                "(D73 (e)), so final == base and the loop / eligible split is informational"}
        extra: Dict[str, Any] = {
            "seed_replicate_label": selection, "fit_seed": (seed_n if steered and not arditi else None), "inspect_seed_base": seed_n, "decode_seed_policy": INSPECT_SEED_POLICY,
            "decode_seed_rule_text": seed_rule_text, "final_cap": CAP, "token_cap_ladder": list(LADDER), "escalation_state": "final_at_base",
            "store_uri": LC.rel_home(base["dir"]), "store_key": base["store_key"], "producer_key_rule": PRODUCER_KEY_RULE, "group_rule": GROUP_RULE,
            "store_pin": {"key": pkey, "tree": pin}, "store_pins_record": LC.rel_home(pins_inp.uri.rsplit("#", 1)[0]),
            "launch_set": {"record": LC.rel_home(ls_inp.uri.rsplit("#", 1)[0]), "tuple_id": lrow.get("tuple_id"), "class": lclass, "pin_key": launch_pin_key, "object_sha256_expected": expected_sha,
                           "m5t_index": lrow.get("m5t_index"), "block": lrow.get("block"), "table_index": lrow.get("table_index"),
                           # the census of 2026-09-23 spells these — the row's census_status and the census's own anchor declaration (anchor_tuple / anchor_fallback); recorded
                           # only when the row carries a census signal (a status other than `run`, or an anchor declaration), so the first launch set's rows (every row `run`,
                           # nothing carried, no declaration) re-read byte for byte
                           **({k: lrow[k] for k in ("census_status", "anchor_tuple", "anchor_fallback") if k in lrow}
                              if ("anchor_tuple" in lrow or "anchor_fallback" in lrow or lrow.get("census_status") not in (None, "run")) else {})},
            "launch_stamp": stamp, "result_line": base["result_last"], "attempts_before_accepted": base["result_earlier"], "resume_attempts": attempts,
            "standup": standup, "max_lora_rank_rule": "RECORDED, never compared: the single-object Arditi standup composes max_lora_rank 1, the seed standups 8 (D71 / GSM8K C4)",
            "served_window": LC.int_or_none(geometry.get("max_model_len")), "served_max_num_seqs": LC.int_or_none(geometry.get("max_num_seqs")), "geometry_eval_legacy": geometry.get("eval"),
            "harness_cell_id_legacy": man.get("harness_cell_id"), "three_numbers": three_block,
            "item_state_counts": {"correct": correct, "incorrect": n_scored - correct, ITEM_LABEL_AT_CAP_UNRESOLVED: counts["n_at_cap_unresolved"], ITEM_LABEL_UNSCORED: counts["n_unscored_below_cap"],
                                  "scored_at_cap": counts["n_scored_at_cap"]},
            "at_cap_classes": {"loop": loops_unresolved, "ceiling": n_ceiling},
            "served_model_id_hf": served_hf, "served_model_revision": rev or None, "serving_identity": ident, "serving_engine": man.get("serving_engine"), "serving_path_legacy": man.get("serving_path"),
            "served_tree_sha": tree, "object_sha16": (sha16 or None), "object_attested_sha256": (attested or None), "fit_model_id_hf": fitp.get("model_id"), "fit_transfer": fitp.get("transfer"),
            "fit_kind_legacy": fitp.get("fit_kind"), "steer_positions_legacy": man.get("steer_positions"), "bank_view_legacy": bank, "oos_tuple": man.get("oos_experiment"),
            "producer_cell_id_legacy": okey, "producer_slug_legacy": slug, "producer_view_legacy": selection, "producer_tag_legacy": tag,
            "inspect_ai_version": frameworks.get("inspect_ai"), "inspect_evals_version": frameworks.get("inspect_evals"), "framework_legacy": gs.get("framework"),
            "client_lock_sha256": frameworks.get("client_lock_sha256"), "sampling_ref_legacy": gs.get("sampling_ref"),
            "decode_policy_ref_legacy": (policy.policy_id if policy else None), "decode_policy_reader_match": ("exact by the stated sampling numbers" if policy else policy_why),
            "sampling_mode_legacy": man_sampling.get("mode") or base["agg_extra"].get("sampling_mode") or STORE_DIR,
            "generation_state": "generated; rule-scored at generation (inspect_evals math: sympy equivalence on the boxed answer), no judge", "source_legacy": SOURCE,
            "row_config_legacy": f"configs/sdfbase/math500/sdfbase_full22_{tag}.math500_{slug}.yaml", "anchor_rule": ANCHOR_RULE,
        }
        if not steered:
            extra["decode_replicate"] = {"trial_index": seed_n, "seed_base": seed_n, "twin_tags_legacy": [f"{MODEL_KEY}_s{k}" for k in (1, 2, 3) if k != seed_n], "rule": UNFITTED_TRIAL_RULE,
                                         "coexist_ruling_needed": f"a roster coexist ruling over lane_sources_legacy [{SOURCE}] x the seed tags keeps the trials selected together"}
            if paired:
                # the paired tag's `none` (or prompted key): a standup replicate of the seed tag's row at the same seed base, re-served (PAIRED_ANCHOR_RULE)
                extra["decode_replicate"]["standup_replicate_of_view"] = group
                extra["decode_replicate"]["coexist_ruling_needed"] += " and the paired tag (the same seed base, re-served: r2v4-coexist-<lane>-paired-view-none-anchors)"
        if paired:
            extra.update(paired_of=f"s{seed_n}", seed_tag_legacy=group, roster_paired_of=roster_paired_of)
            if obj.object_kind == "none":
                # the paired tag's OWN anchor (PAIRED_ANCHOR_RULE): no cross-standup anchor of its own
                extra.update(standup_replicate_of_view=group, anchor_rule=PAIRED_ANCHOR_RULE,
                             anchor_role=f"the paired tag's OWN anchor: a standup replicate of {group}/none (D429); the composed row pairs against it once it files")
            elif anchor_plan is not None and anchor_plan.get("own"):
                # its own tag's `none` filed: the contrast pairs against it (the run is added now, so the spelling is decided here)
                extra.update(standup_replicate_of_view=group, own_anchor=f"{tag}/none",
                             anchor_rule=PAIRED_ANCHOR_RULE + f"; this row pairs with {tag}/none (its own standup's `none`: extra.cross_standup false on the contrast)")
            else:
                # the fallback: ACROSS tags against the seed tag's `none`, the standing spelling; the own `none`'s state rides beside it once the launch set names one
                extra.update(cross_standup_anchor=f"{group}/none",
                             anchor_rule=ANCHOR_RULE + f"; this {'composed ' if steered else ''}row pairs with {group}/none (another standup: extra.cross_standup on the contrast)")
                if anchor_plan is not None and anchor_plan.get("own_none_state") is not None:
                    extra.update(own_none_state=anchor_plan["own_none_state"], own_none_note=anchor_plan["own_none_note"])
        lead = f"MATH-500 thinking ON, one pass at {CAP:,}, seed tag {tag} {okey}"
        declared_role = None
        if arditi:
            declared_role = ARDITI_ROLE
            extra.update(fit_seed=None, declared_role=declared_role,
                         single_object={"object_key": okey, "staged_id": str(obj.staged_id), "object_id": obj.object_id, "operating_point": op.rung, "standup_legacy": op.standup_tag,
                                        "c": float(op.coefficient), "unit": op.unit, "declared_role": declared_role, "rule": DECODE_TRIAL_RULE},
                         decode_replicate={"trial_index": seed_n, "seed_base": seed_n, "twin_tags_legacy": [f"{MODEL_KEY}_s{k}" for k in (1, 2, 3) if k != seed_n], "rule": DECODE_TRIAL_RULE,
                                           "single_object": {"object_key": okey, "staged_id": str(obj.staged_id), "object_id": obj.object_id, "operating_point": op.rung},
                                           "coexist_ruling_needed": (f"a roster coexist ruling over lane_sources_legacy [{SOURCE}] x campaigns [{CAMPAIGN}] x model_ids [{model_id}] x object_ids "
                                                                     f"[{obj.object_id}] x eval_ids [{EVAL_ID}] x instrument_ids [{instrument_id}] x views_legacy "
                                                                     f"[{', '.join(f'{MODEL_KEY}_s{k}' for k in (1, 2, 3))}] keeps the three trials selected together (the user's, never a reader default)")})
            lead += f" (ONE object on every seed tag: decode-seed trial {seed_n}, {op.rung} c = {op.coefficient:g} {op.unit}; declared {declared_role})"
        notes = (f"{lead}: {correct}/{n_scored} correct over scored = {acc:.4f}" if acc is not None else f"{lead}: no scored item")
        notes += (f"; {counts['n_at_cap']}/{N_ITEMS} at the cap ({counts['n_at_cap_unresolved']} unresolved — neither hit nor miss: {loops_unresolved} loops by the producer's rule, "
                  f"{n_ceiling} at the ceiling — and {counts['n_scored_at_cap']} scored at the cap)"
                  + (f", {counts['n_unscored_below_cap']} unscored below the cap (unterminated think block)" if counts["n_unscored_below_cap"] else "")
                  + f"; correct/{N_ITEMS} = {bound_lo:.3f} is the BOUND, not the value")
        try:
            payload = LC.run_payload_base(seeds, run_id=rid, producer_key=producer_key, object_key=okey, view_legacy=tag, cell_id_legacy=okey, obj=obj, eval_id=EVAL_ID,
                                          instrument_row=inst_row, instrument_seed_id=INSTRUMENT_SEED, lifecycle_state="generated", when_utc=when_final,
                                          when_source="RESULT's last line (result=COMPLETE) for generated; manifest.json:resume_attempts[completed].ts for queued / started",
                                          token_cap_tokens=CAP, judge_id=None, judge_seed_id=None, substrate=substrate, lane_source_legacy=SOURCE,
                                          store_uri=LC.rel_home(base["dir"]), instrument_match=inst_match, extra=extra, notes=notes)
        except KeyError as exc:
            raise _PairRefusal(f"run row: {exc}", base["files"]["manifest.json"].uri)
        fc = payload["field_confidence_json"]
        trial_index = seed_n if (not steered or arditi) else 0
        payload.update(producer_session=PRODUCER_SESSION, trial_index=trial_index, seed_base=seed_n, queued_utc=when_start, started_utc=when_start,
                       n_items_planned=N_ITEMS, n_items_generated=N_ITEMS, n_items_judged=n_scored, model_revision=(rev if _SHA1.match(rev) else None),
                       intervention_hash=man.get("intervention_hash"), harness_config_hash=man.get("harness_config_hash"),
                       env_digest=(next(iter(base["consts"].get("env_digest") or []), None)), cluster_job_id=job, job_name=base["result"].get("job_name"),
                       node_hostname=(standup.get("node") or None), serving_identity=ident, token_cap_rung_index=0, token_cap_ladder_json=list(LADDER),
                       decode_seed_rule=DECODE_SEED_RULE, decode_sent_json=sent, decode_temperature_sent=sent.get("temperature"), decode_top_p_sent=sent.get("top_p"),
                       decode_top_k_sent=(int(sent["top_k"]) if sent.get("top_k") is not None else None), code_sha=tree,
                       harness_version=(f"inspect-ai {frameworks.get('inspect_ai')} / inspect_evals {frameworks.get('inspect_evals')} ({gs.get('framework')})"
                                        if frameworks.get("inspect_ai") else payload["harness_version"]),
                       context_window_tokens=LC.int_or_none(geometry.get("max_model_len")),
                       store_completion_artifact_uri=base["files"]["gen_stats.json"].uri, store_completion_sha256=base["files"]["gen_stats.json"].sha256)
        if LC.int_or_none(geometry.get("max_num_seqs")) is not None:
            payload.update(served_max_num_seqs=LC.int_or_none(geometry.get("max_num_seqs")), served_max_num_seqs_source="attested")
            fc["served_max_num_seqs"] = LC.confidence_entry("recorded", f"manifest.json:geometry.max_num_seqs (source {geometry.get('source')})")
        if policy is not None:
            payload["decode_source"] = policy.decode_source
            fc["decode_source"] = LC.confidence_entry("recorded", f"the registered policy row {policy.policy_id} the sent numbers equal (its decode_source)")
        else:
            fc["decode_source"] = LC.unverified(policy_why)
        fc["code_sha"] = LC.confidence_entry("recorded", f"manifest.json:producer.tree_sha = the store pin {pkey}")
        fc["model_revision"] = LC.confidence_entry("recorded", "manifest.json:model_revision (= models.yaml hf_revision)")
        fc["cluster_job_id"] = LC.confidence_entry("recorded", "manifest.json:row_constants.slurm_job_id = RESULT's job = rows.jsonl slurm_job_id")
        fc["node_hostname"] = (LC.confidence_entry("recorded", f"{standup.get('attestation')}:hostname (the standup the manifest's routing_table names)") if standup.get("node")
                               else LC.unverified("the standup's serving_attestation.json is not mirrored beside the store"))
        fc["token_cap_tokens"] = LC.confidence_entry("recorded", f"gen_stats.json:max_tokens = the outcome record's max_tokens = {CAP} (one pass, no cap step)")
        fc["seed_base"] = LC.confidence_entry("recorded", "manifest.json:seeds.inspect.replicate.seed_base (the replicate policy at the tag's index)")
        fc["decode_seed_rule"] = LC.confidence_entry("derived", f"one seed per item: {seed_rule_text} (k = 1, so the closure is the item; the base rides seed_base and the rule's text extra.decode_seed_rule_text)")
        fc["decode_sent_json"] = LC.confidence_entry("recorded", f"gen_stats.json (the client's sampler; manifest.json:sampling agrees; the store's resolved_sampler.json is registered beside it)")
        fc["instrument_match"] = LC.confidence_entry("derived", f"the sent numbers compared to instrument seed {INSTRUMENT_SEED}'s decode columns (bound to {model_id}): {inst_match}")
        fc["harness_version"] = LC.confidence_entry("recorded", "manifest.json:frameworks (inspect_ai / inspect_evals) and gen_stats.json:framework")
        fc["context_window_tokens"] = LC.confidence_entry("recorded", "manifest.json:geometry.max_model_len")
        fc["n_items_planned"] = LC.confidence_entry("derived", f"the instrument's item set (n_items {N_ITEMS}); gen_stats.json:n and the outcome record's n_attempted_items agree")
        fc["n_items_generated"] = LC.confidence_entry("recorded", "the outcome record: every attempted item has a stop reason (scored, or at the cap)")
        fc["n_items_judged"] = LC.confidence_entry("derived", "the SCORED items (the user's rule: an at-cap item without an answer is neither hit nor miss); rule-scored at generation")
        fc["queued_utc"] = fc["started_utc"] = LC.confidence_entry("recorded", "manifest.json:resume_attempts — the completed attempt's ts")
        fc["generated_utc"] = LC.confidence_entry("recorded", "RESULT's last line (result=COMPLETE): its timestamp")
        fc["thinking_mode"] = LC.confidence_entry("recorded", "gen_stats.json:thinking = manifest.json:thinking.mode = the outcome record's thinking = on")
        if steered and man.get("steer_positions") in V.APPLIED_POSITIONS_MODES:
            payload["applied_positions_mode"] = str(man["steer_positions"])
            fc["applied_positions_mode"] = LC.confidence_entry("recorded", "manifest.json:steer_positions")
        if not steered:
            payload["strength_unit"] = "unverified"
            fc["strength_unit"] = LC.unverified("an unsteered / prompted object key: no object on the served path, so no dose and no unit (the column default)")
            fc["trial_index"] = LC.confidence_entry("derived", UNFITTED_TRIAL_RULE + f"; manifest.json:seeds.inspect.replicate.seed_base = {seed_n}")
        if arditi:
            payload.update(op.strength_columns())
            fc.update(op.strength_confidence(f"the manifest's route.bank_view `{op.rung}` and the launch set's pin key {op.staged_id} (the all-positions pins' staged object)"))
            payload["extra"].update(op.extra())
            op_mv = seeds.model_for_view_legacy(op.standup_tag)
            if op_mv.get("transfer_kind") in V.TRANSFER_KINDS:
                payload["transfer_kind"] = op_mv["transfer_kind"]
                fc["transfer_kind"] = LC.confidence_entry("recorded", f"legacy_spellings.yaml model_keys[{op.standup_tag}].transfer_kind (the direction is fit on and served on the instruct "
                                                                      f"checkpoint: manifest.json:fit_provenance.transfer {fitp.get('transfer')!r})")
            fc["trial_index"] = LC.confidence_entry("derived", DECODE_TRIAL_RULE + f"; the launch set names the one object {str(obj.object_id)[:12]}… on every seed tag, so this row is trial "
                                                               f"{seed_n} (manifest.json:seeds.inspect.replicate.seed_base = {seed_n})")
        facts["runs"] += 1          # the run record is added by read() once the contrasts are decided: its extra.anchor block rides it (a record is built once)
        # NO state event: the record's `generated` is the state and the harvester's rule-scored completion stamps `ingested`
        comp_id = recount_computation_id()
        code_sha = str(producer.get("code_sha") or "")
        batch.add("computation", {"computation_id": comp_id, "algorithm": RECOUNT_ALGORITHM, "version": RECOUNT_VERSION, "seed_provenance": RECOUNT_SEED_PROVENANCE,
                                  "code_sha": (code_sha if _SHA1.match(code_sha) else "unverified"),
                                  "notes": "correct / scored recounted from the store's outcome record over the 500 attempted items (one pass at 16,384, no cap step: final == base)"},
                  producer, now)
        rows_uri = LC.rel_home(base["files"]["rows.jsonl"].uri)
        lo, hi = IC.wilson(correct, n_scored) if n_scored else (None, None)
        value_payload = {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": comp_id,
                                 "producer_metric_spelling": "value_correct_over_scored", "value": acc, "k": correct, "n": n_scored, "ci_lo": lo, "ci_hi": hi,
                                 "ci_method": ("wilson_95" if n_scored else None), "n_abstentions": None, "n_incomplete": counts["n_unscored"], "computed_by": "registry",
                                 "computed_utc": when_final, "source_uri": rows_uri,
                                 "extra": {"role": "THE VALUE: correct over SCORED (the user's rule: degeneration is scored separately)", "recounted_from": "the store's outcome record",
                                           "equals_gen_stats_acc": True, "n_unresolved_neither_hit_nor_miss": counts["n_at_cap_unresolved"], "n_unscored_final": counts["n_unscored"],
                                           "producer_record": three_block["producer_record"], "producer_value_correct_over_scored": LC.num_or_none(tn.get("value_correct_over_scored"))}}        # added by read() after the contrasts: a park lands on its extra.delta_block
        batch.add("run_metric", {"run_id": rid, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0, "computation_id": "",
                                 "producer_metric_spelling": BOUND_SPELLING, "value": bound_lo, "k": correct, "n": N_ITEMS, "ci_lo": bound_lo, "ci_hi": bound_hi, "ci_method": BOUND_CI_METHOD,
                                 "n_abstentions": None, "n_incomplete": counts["n_at_cap_unresolved"], "computed_by": "producer", "computed_utc": when_final,
                                 "source_uri": three_block["producer_record"],
                                 "extra": {"role": "THE BOUND, never the value: correct over the 500 attempted; the interval's lower end counts the unresolved items wrong, the upper end right",
                                           "n_unresolved_neither_hit_nor_miss": counts["n_at_cap_unresolved"], "n_unscored_below_cap_counted_as_miss": counts["n_unscored_below_cap"],
                                           "unresolved_definition": three_block["unresolved_definition"],
                                           "producer_spellings": {BOUND_SPELLING: LC.num_or_none(tn.get(BOUND_SPELLING))},
                                           "metric_role_expected": "secondary (a blank computation_id ranks after the value's computed row at one identity)"}}, producer, now)
        # -- items (whole rows): value 1 / 0, or NULL with the third state's label
        rows_sha = base["files"]["rows.jsonl"].sha256
        for iid in sorted(items["items"]):
            it = items["items"][iid]
            if it["scored"]:
                label, cls = None, None
            elif it["at_cap"]:
                label, cls = ITEM_LABEL_AT_CAP_UNRESOLVED, ("loop" if iid in split["loops"] else "ceiling")
            else:
                label, cls = ITEM_LABEL_UNSCORED, UNSCORED_BELOW_CAP_CLASS
            row = base["scored"].get(iid) or base["unscored"].get(iid)
            created = IC.utc_from_epoch(row[3]) if (row is not None and row[3] is not None) else None
            deg = split["by_item"].get(iid) if isinstance(split["by_item"].get(iid), Mapping) else {}
            iextra = {"final_cap": CAP, "stop_reason": it["stop_reason"], "completion_tokens": it["completion_tokens"], "decode_seed": it["seed"],
                      "at_cap_class": (cls if cls in AT_CAP_CLASSES else None), "scored_at_cap": bool(it["scored"] and it["at_cap"]),
                      "unscored_reason": (UNSCORED_BELOW_CAP_CLASS if cls == UNSCORED_BELOW_CAP_CLASS else None),
                      "think_block_unterminated": it.get("think_block_unterminated"), "unparseable": it.get("unparseable")}
            batch.add("item_result", IC.item_result_payload(rid, iid, HEADLINE_METRIC, it["value"], label, created or when_final, judgment_id="", trial_index=trial_index, extra=iextra),
                      producer, now)
            loop_flag = (1 if iid in split["loops"] else 0) if (split["present"] and it["at_cap"]) else None
            batch.add("item_flags", IC.item_flags_payload(rid, iid, rows_sha, (row[1] if row is not None else None), at_token_cap=(1 if it["at_cap"] else 0), finish_reason=it["stop_reason"],
                                                          response_tokens=it["completion_tokens"], decode_seed=(str(it["seed"]) if it["seed"] is not None else None), judgment_id="",
                                                          trial_index=trial_index, degenerate_rep_ratio_loop=loop_flag, rep_ratio=LC.num_or_none(deg.get("rep_ratio")),
                                                          extra={"at_token_cap_source": "the outcome record's at_cap (stop_reason == max_tokens at the request's cap; never re-tokenized)",
                                                                 "generation_pointer": "rows.jsonl (sha, line): the scored `acc` row or the `unscored` row of the item",
                                                                 "outcome_record": LC.rel_home(base["files"]["outcomes"].uri), "tail_rep_ratio": LC.num_or_none(deg.get("tail_rep_ratio")),
                                                                 "loop_source": (three_block["producer_record"] if split["present"] else None)}),
                      producer, now)
        # -- the run-level shares (run_degeneration over the 500 attempted items)
        measured = {
            RAW_AT_CAP_MEASURE: dict(numerator=counts["n_at_cap"], denominator=N_ITEMS, unit="items", threshold_params={"cap_final": CAP, "ladder": list(LADDER)}, computed_by="registry",
                                     extra={"rule": three_block["at_cap_rule"], "classes": {"loop": loops_unresolved, "ceiling": n_ceiling, "scored_at_cap": counts["n_scored_at_cap"]},
                                            "n_at_cap_unresolved": counts["n_at_cap_unresolved"], "n_unscored_below_cap_not_counted": counts["n_unscored_below_cap"], "source": LC.rel_home(base["files"]["outcomes"].uri)}),
        }
        if split["present"]:
            measured[LOOP_MEASURE] = dict(numerator=len(split["loops"]), denominator=N_ITEMS, unit="items", threshold_params={"rep_ratio_gt": 0.5, "tail_tokens": 200, "cap": CAP}, computed_by="producer",
                                          extra={"rule": split["rule"], "class": "loop: at the cap AND degenerate by the producer's rep-ratio rule over the at-cap text (informational, D73 (e): no cap step exists)",
                                                 "n_loops_unresolved": loops_unresolved, "n_loops_scored_at_cap": len(split["loops"]) - loops_unresolved, "source": three_block["producer_record"]})
        notes_nm = {"classifier": "the paper's MATH-500 reader files the loop count under `classifier`; this reader carries the producer's rep-ratio rule (rep_ratio_loop)",
                    "contentless": "no contentless reading in the producer's record", "zlib_collapse": "no zlib reading in the producer's record",
                    "union5": "the five-detector union is not computed for these stores (the degeneracy-union reader's input, not here)",
                    IC.AT_CAP_RETOKENIZED: "the outcome record reads the provider's stop reason, never a re-tokenized length",
                    LOOP_MEASURE: "the producer's three-number row carries no loop / eligible split for this store"}
        for drow in IC.declared_degeneration_rows(rid, "", measured, source_uri=LC.rel_home(base["files"]["outcomes"].uri), computed_utc=when_final, not_measured_notes=notes_nm):
            batch.add("run_degeneration", drow, producer, now)
        batch.add("run_degeneration", IC.degeneration_payload(rid, COHERENT_MEASURE, "measured", judgment_id="", numerator=n_ceiling, denominator=N_ITEMS, unit="items",
                                                               threshold_params={"cap_final": CAP}, computed_by="registry", computed_utc=when_final, source_uri=LC.rel_home(base["files"]["outcomes"].uri),
                                                               extra={"class": "unresolved at the cap and not a loop by the producer's rule (at the one pass's ceiling)", "classes": {"ceiling": n_ceiling},
                                                                      "source": LC.rel_home(base["files"]["outcomes"].uri)}), producer, now)
        facts["rows"][f"{tag}|{okey}"] = {"run_id": rid, "tag": tag, "selection": selection, "paper_label": obj.paper_label, "object_id": obj.object_id, "staged_id": obj.staged_id,
                                          "instrument_id": instrument_id, "acc_over_scored": acc, "correct": correct, "n_scored": n_scored, "acc_over_attempted": bound_lo, "upper_bound": bound_hi,
                                          "n_at_cap": counts["n_at_cap"], "share_at_cap": counts["n_at_cap"] / N_ITEMS, "n_at_cap_unresolved": counts["n_at_cap_unresolved"],
                                          "n_scored_at_cap": counts["n_scored_at_cap"], "n_unscored_below_cap": counts["n_unscored_below_cap"], "n_loops": len(split["loops"]),
                                          "n_eligible_informational": len(split["eligible"]), "instrument_match": inst_match, "decode_policy": (policy.policy_id if policy else None),
                                          "job": job, "tree": tree, "trial_index": trial_index, "store_key": base["store_key"], "producer_key": producer_key, "paired": paired,
                                          "max_lora_rank_recorded": standup.get("max_lora_rank"), "node": standup.get("node"), "declared_role": declared_role, "steered": steered}
        return {"run_id": rid, "items": items["items"], "steered": steered, "when": when_final, "declared_role": declared_role, "tag": tag, "paired": paired, "okey": okey,
                "payload": payload, "value_row": value_payload}

    @staticmethod
    def _rows_time(bundle: Mapping[str, Any], pick) -> Optional[str]:
        created = bundle["consts"].get("created_at")
        return IC.utc_from_epoch(pick(created)) if created else None

    # ---- contrasts (PAIRED_ANCHOR_RULE) ------------------------------------------------------
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
        The row's OWN pairing is the plan's (:func:`_bcbw_common.anchor_plan`, the same resolution the rows' extras were stamped
        from: its own tag's ``none``, the paired tag's own ``none`` once it files, else the fallback across tags): it is the own anchor the currency
        is asked about and it stands while the frame holds it current with a usable reading -- item-wise over the items SCORED on both sides (the
        accuracy difference in pp, McNemar's exact p on the discordant pairs, Cohen's h; an unresolved or unscored item on either side leaves the pair),
        plus ``extra.anchor_selection``. A retired, unfolded or absent own anchor searches the exact key: a re-picked unsteered run that is a row of this
        batch is paired per item like the own one, any other files the registry's UNPAIRED rate difference from the two readings' k / n (``computed_by
        registry``, its own computation row named by ``extra.statistic_computation_id``). The kind :func:`_bcbw_common.anchor_kind_for` yields is checked
        against the currency's on every filed contrast (a disagreement refuses the group by name). No current anchor at the exact key parks the comparison
        by name -- the VALUE row's ``extra.delta_block``, a refusal line -- and the previous revision's key is withdrawn by the head. Stamps every steered
        row's ``extra.anchor`` block on its (not yet added) run payload and returns the contrast payloads. The Arditi rows carry their declared role."""
        out: List[Dict[str, Any]] = []
        comp_id = recount_computation_id()
        frame: Optional[AC.AnchorCurrency] = None
        by_run = {st_["run_id"]: st_ for st_ in filed.values()}
        for key in sorted(k for k in filed if k[1] != "none"):
            tag, okey = key
            st = filed[key]
            plan = anchor_plan(group, tag, filed, expected, seen)
            anchor = plan["anchor"]
            plan_why = None
            if anchor is None:
                plan_why = f"the seed tag's `none` did not file ({group}|none refused or absent); no contrast is formed"
                if tag != group:
                    plan_why += f"; the paired tag's own `none` ({tag}|none) did not file either"
            block = {"rule": ANCHOR_RULE, "anchor_view_legacy": (plan["anchor_tag"] or group), "anchor_object_key": "none", "cross_standup": plan["cross_standup"],
                     "declared_role": st.get("declared_role"), "currency_rule": CURRENCY_RULE}
            if plan["own_none_state"] is not None:
                block.update(own_none_state=plan["own_none_state"], own_none_note=plan["own_none_note"])
            if plan["own"] and st.get("paired"):
                block.update(rule=PAIRED_ANCHOR_RULE, own_standup_replicate_of_view=group)
            # ---- the decision (CURRENCY_RULE): the plan's anchor is the OWN anchor the pass's frame is asked about; the frame loads on the first comparison
            if frame is None:
                frame = ctx.anchor_currency()
            own_id = anchor["run_id"] if anchor is not None else None
            own_kind = B.anchor_kind_for(plan) if anchor is not None else None
            akey = self._anchor_key(st, own_id, own_kind)
            choice = frame.choose(akey, HEADLINE_METRIC, -1.0, None, computation_id=comp_id, steered_run_id=st["run_id"], steered_judgment_id="")
            selection = choice.selection_json()
            facts["anchor_states"][choice.state] = facts["anchor_states"].get(choice.state, 0) + 1
            value_row = st["value_row"]
            k_s, n_s = value_row.get("k"), value_row.get("n")
            own_text = (f"{own_id} ({own_kind}; {plan['anchor_tag']}/none)" if own_id is not None else f"none declared ({plan['state']}: {plan['why'] or plan_why})")
            key_text = f"key (cap {akey.token_cap_tokens}, trial {akey.trial_index}, pass {akey.pass_label}, instrument {akey.instrument_id})"
            fact = {"anchor": None, "state": plan["state"], "why": (plan["why"] or plan_why), "own_none_state": plan["own_none_state"], "currency_state": choice.state,
                    "currency_own_reason": choice.own_reason, "own_anchor_run_id": own_id, "n_candidates": choice.n_candidates}

            def park(why: str, refusal: str) -> None:
                value_row.setdefault("extra", {})["delta_block"] = dict(
                    {f: None for f in ("delta", "delta_lo", "delta_hi", "delta_p", "effect_size", "effect_kind")}, n_pairs=0,
                    delta_method=f"none: {why}", computed_by="registry", why_not_a_contrast_row=why,
                    anchor=dict(own_anchor_run_id=own_id, own_anchor_kind=own_kind, state=choice.state, own_reason=choice.own_reason, anchor_run_id=choice.run_id,
                                anchor_kind=choice.anchor_kind, registry_contrast_filed_against=None),
                    anchor_selection=selection, producer_block={"producer_record": (st["payload"]["extra"].get("three_numbers") or {}).get("producer_record"),
                                                                "note": "the producer's three-number record files no delta against `none`; the comparison is the reader's alone"},
                    rule=AC.RULE)
                facts["contrasts_parked"] += 1
                self.refusals.add(f"{tag}|{okey}|contrast", refusal, st["payload"]["extra"].get("store_uri") or self.root)

            if choice.state == "no_current_anchor":
                # no current anchor at the exact key: the plan's own spelling of why (a `none` not yet run / refused / absent) stays; the currency's reason beside it
                reason = plan["why"] or plan_why or choice.why
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
                cand = next((c for c in choice.candidates if c.run_id == choice.run_id), None)
                same_standup = choice.anchor_kind == AC.KIND_SAME_STANDUP
                tier_key = "borrowed_reference" if choice.state == "borrowed_reference" else ("same_standup_current" if same_standup else "sibling_current")
                plan = dict(plan, cross_standup=not same_standup, anchor_tag=(cand.view_legacy if cand is not None else None), **{tier_key: True})
                anchor = by_run.get(choice.run_id)          # a row of THIS batch (another tag's `none`): paired per item like the own one; else unpaired from k / n
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
            common_extra: Dict[str, Any] = {"unit": "pp", "anchor_view_legacy": plan["anchor_tag"], "anchor_object_key": "none", "anchor_rule": ANCHOR_RULE,
                                            "cross_standup": plan["cross_standup"], "row_tag_legacy": st.get("tag"), "anchor_selection": selection, "currency_rule": CURRENCY_RULE}
            if anchor is not None:
                a_items, s_items = anchor["items"], st["items"]
                shared = sorted(i for i in s_items if i in a_items and s_items[i]["value"] is not None and a_items[i]["value"] is not None)
                if not shared:
                    why = (f"no item scored on both sides (anchor {chosen})" if reanchored else "no item scored on both sides")
                    block.update(state="no_shared_items", reason=why, anchor_run_id=anchor["run_id"])
                    st["payload"]["extra"]["anchor"] = block
                    facts["contrasts"][f"{tag}|{okey}"] = dict(fact, anchor=anchor["run_id"], state="no_shared_items", why=why)
                    park(why, f"contrast's anchor {chosen} is current but shares no scored item with the row at {key_text}; filed without a contrast")
                    continue
                b = sum(1 for i in shared if s_items[i]["value"] == 1.0 and a_items[i]["value"] == 0.0)
                c = sum(1 for i in shared if s_items[i]["value"] == 0.0 and a_items[i]["value"] == 1.0)
                p_s = sum(s_items[i]["value"] for i in shared) / len(shared)
                p_a = sum(a_items[i]["value"] for i in shared) / len(shared)
                stat = {"delta": 100.0 * (p_s - p_a), "delta_p": DC.mcnemar_exact(b, c), "delta_method": CONTRAST_METHOD, "effect_size": DC.cohen_h(p_s, p_a), "effect_kind": "cohens_h",
                        "n_pairs": len(shared)}
                stat_extra = {"discordant_steered_only": b, "discordant_anchor_only": c, "acc_steered_over_pairs": p_s, "acc_anchor_over_pairs": p_a,
                              "n_excluded_steered_unscored": sum(1 for i in s_items if s_items[i]["value"] is None),
                              "n_excluded_anchor_unscored": sum(1 for i in a_items if a_items[i]["value"] is None),
                              "method_text": "paired per item over the items SCORED on both sides (the same 500 MATH-500 items, one instrument, one seed base): the accuracy "
                                             "difference, pp; p from McNemar's exact test on the discordant pairs; Cohen's h of the two accuracies; an unresolved or unscored item "
                                             "on either side leaves the pair"}
                if reanchored:
                    stat_extra["pairing"] = f"paired per item: the re-picked anchor {chosen} is a row of this batch ({plan['anchor_tag']}/none)"
                n_pairs = len(shared)
            else:
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
                              "method_text": (f"unpaired rate difference of the row's recount (correct / scored, k {k_s} of n {n_s}) against the anchor's registry reading "
                                              f"(k {m.k:g} of n {m.n}) in pp; Newcombe hybrid score interval; two-proportion z; Cohen's h")}
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
                                    own_pairing={"own_anchor_run_id": own_id, "own_anchor_kind": own_kind, "state": plan["state"], "why": (plan["why"] or plan_why),
                                                 "own_none_state": plan["own_none_state"], "own_reason": choice.own_reason, "own_why": choice.own_why},
                                    reanchor_note=reanchor_note, **({"cross_standup_note": reanchor_note} if plan["cross_standup"] else {}))
            payload = {"run_id": st["run_id"], "anchor_run_id": choice.run_id, "judgment_id": "", "metric_id": HEADLINE_METRIC, "threshold": -1.0,
                       "computation_id": comp_id, "anchor_judgment_id": choice.anchor_judgment_id, "anchor_kind": B.anchor_kind_for(plan),
                       **stat, "computed_by": "registry", "computed_utc": st["when"], "extra": {**common_extra, **stat_extra}}
            if reanchored:
                pass        # the re-anchoring text is on `reanchor_note` (and `cross_standup_note` across standups) / `own_pairing` above
            elif plan["cross_standup"]:
                if plan["own_none_state"] is None:
                    payload["extra"]["cross_standup_note"] = (f"the composed row's tag {st.get('tag')} has no `none` of its own: D65 pairs it with the seed tag's `none` ({group}/none), served by "
                                                              "another standup at the same seed base")
                else:
                    payload["extra"]["cross_standup_note"] = (f"the composed row's tag {st.get('tag')} has its own `none` in the launch set but it did not file ({plan['own_none_state']}): D65 pairs "
                                                              f"the row with the seed tag's `none` ({group}/none), served by another standup at the same seed base, until its own `none` files")
                    payload["extra"].update(own_none_state=plan["own_none_state"], own_none_note=plan["own_none_note"])
            elif st.get("paired"):
                payload["extra"].update(anchor_rule=PAIRED_ANCHOR_RULE, own_standup_note=B.own_standup_note(group, str(st.get("tag"))))
            if st.get("declared_role") is not None:
                payload["extra"].update(declared_role=st["declared_role"],
                                        anchor_standup_note=(f"the value is declared {st['declared_role']}: the single-object standup attests another max_lora_rank than the seed tag's "
                                                             "(RECORDED on both runs, never compared); the contrast is the reader's, formed as for every steered key, and the role rides on it"))
            out.append(payload)
            value_row.setdefault("extra", {})["delta_block"] = None
            facts["contrasts_filed"] += 1
            facts["contrasts_reanchored" if reanchored else "contrasts_own"] += 1
            block.update(state=ANCHOR_CONTRASTED, anchor_run_id=choice.run_id, n_pairs=n_pairs)
            st["payload"]["extra"]["anchor"] = block
            facts["contrasts"][f"{tag}|{okey}"] = dict(fact, anchor=choice.run_id, state=ANCHOR_CONTRASTED, n_pairs=n_pairs, delta_pp=payload["delta"], delta_p=payload["delta_p"],
                                                       cohens_h=payload["effect_size"], cross_standup=plan["cross_standup"], declared_role=st.get("declared_role"),
                                                       anchor_view_legacy=plan["anchor_tag"], own_standup=(plan["own"] and not reanchored), reanchored=reanchored,
                                                       **({"reanchor_tier": tier_key} if reanchored else {}))
        return out

def tags_inv(tags: Mapping[str, str], tag: str) -> str:
    """The selection of a tag in the group's ``{selection: tag}`` map."""
    for sel, t in tags.items():
        if t == tag:
            return sel
    raise KeyError(tag)


def factory() -> Math500SeedrepAdapter:
    return Math500SeedrepAdapter()


register(NAME, factory)
