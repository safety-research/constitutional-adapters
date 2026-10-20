"""The batch-64 three-seed Qwen family's BigCodeBench-200 stores -> registry (source ``bigcodebench_seedrep``).

WHAT WAS RUN (the seed-replicate BigCodeBench-200 / BFCL-200 run, b64-capability-agentic-queue; Modal, the ``twosi-fits`` volume, at
tree ``8365992042b6a4836a27593ea4364b78871132d9``; the 30 rows the launch set names, 25 of which have stores at the first mirror): seed tags
``qwen35_4b_s1`` / ``_s2`` / ``_s3`` x {the unsteered anchor ``none``, the five admitted staged objects (``base64s<N>__…`` / ``derived_base64s<N>__…``:
constitution_relu, control_relu, constitution_lora, control_lora, difference_lora), the two prompted system instructions (safety_si,
constitution_si) and the Arditi all-positions refusal steer at its operating point (``ar_c0p3__arditi_allpos``, ONE fit on every seed tag)}, plus
the composed ``difference_relu`` paired write on the PAIRED tags ``qwen35_4b_s<N>p`` (``paired_eb64s<N>__constv2-mix-full__psr_ce_all_minus_ctl``: its
own standup, no ``none`` of its own). Every row is the BigCodeBench-200 instrument — inspect_evals 0.18.0's ``bigcodebench`` task as row task
``inspect/bigcodebench_200`` over the 200 pinned items (of 992; ``manifest.json:tinycap.bcb_item_set``), pass@1 in the pinned sandbox image,
thinking OFF, the benchmark's reference protocol sampler (greedy: T 0 / top_p 1 / top_k 0, the registered row
``qwen35_4b_it|bigcodebench|off|protocol@v1``), ONE pass at 4,096 (``contracts.BCB200_MAX_TOKENS_OFF``; no cap step) — served LoRA-natively
(``proxy_vllm+lora+interleaved``; the unsteered and prompted keys ride the passthrough route) on ``Qwen/Qwen3.5-4B @ 851bf6e8``, under the DEFAULT
Inspect seed rule (no replicate block: the seed axis is the fitted object). Capability evaluations carry their OWN instrument (never the
statics' 4,096-cap instruments, which share the number and nothing else): every row files on ``bigcodebench_200_protocol_greedy_4096`` bound to the
served model.

Everything else — the mirror's layout, the gate records, the launch set, the pins, the launched roster, the three-number record, the recount, the
cross-checks, the anchors and the not-run rule — is the shared reading in :mod:`_bcbw_common` (the BFCL-200 reader ``bfcl_seedrep`` binds the same
code to the other task). This module binds the :class:`_bcbw_common.TaskSpec` alone.
"""
from __future__ import annotations

from typing import Tuple

from . import _bcbw_common as B
from .base import register

NAME = "bigcodebench_seedrep"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "bigcodebench_seedrep"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
MODEL_KEY = B.MODEL_KEY
EVAL_ID = "bigcodebench"
TASK = "inspect/bigcodebench_200"
RECORD_TASK = "bigcodebench_200"
OUTCOMES_FILE = "inspect__bigcodebench_200.json"
#: the BigCodeBench-200 instrument (configs/registry/instruments.yaml), bound per model by the dimensions loader
INSTRUMENT_SEED = "bigcodebench_200_protocol_greedy_4096"
#: contracts.BCB200_MAX_TOKENS_OFF: the one pass's cap
CAP = 4096
LADDER: Tuple[int, ...] = (CAP,)
N_ITEMS = B.N_ITEMS
#: BigCodeBench's reference protocol: greedy pass@1
PROTOCOL_TEMPERATURE = 0.0
THREE_NUMBERS_FILE = "three_numbers_bigcodebench_qwen_seedrep.json"
ITEM_SET_KEY = "bcb_item_set"
GENERATION_STATE = "generated; rule-scored at generation (inspect_evals bigcodebench: pass@1 of the emitted code in the pinned sandbox image), no judge"
SPEC = B.TaskSpec(name=NAME, source=SOURCE, eval_id=EVAL_ID, task=TASK, record_task=RECORD_TASK, outcomes_file=OUTCOMES_FILE, instrument_seed=INSTRUMENT_SEED, cap=CAP,
                  protocol_temperature=PROTOCOL_TEMPERATURE, three_numbers_file=THREE_NUMBERS_FILE, item_set_key=ITEM_SET_KEY, generation_state=GENERATION_STATE,
                  clamp_override=False, tool_calls_required=False, root_name=SOURCE)   # the tool-call route is BFCL's; here it is RECORDED (extra.tool_calls), never required
ROOT_DEFAULT = SPEC.root_default
RECOUNT_ALGORITHM = SPEC.recount_algorithm
RECOUNT_VERSION = B.RECOUNT_VERSION
RECOUNT_SEED_PROVENANCE = SPEC.recount_seed_provenance
ARDITI_ROLE = B.ARDITI_ROLE
DECODE_SEED_RULE = B.DECODE_SEED_RULE
ITEM_LABEL_AT_CAP_UNRESOLVED = B.ITEM_LABEL_AT_CAP_UNRESOLVED
ITEM_LABEL_UNSCORED = B.ITEM_LABEL_UNSCORED
RAW_AT_CAP_MEASURE = B.RAW_AT_CAP_MEASURE
BOUND_SPELLING = B.BOUND_SPELLING
BOUND_CI_METHOD = B.BOUND_CI_METHOD
GROUP_RULE = B.GROUP_RULE
ANCHOR_RULE = B.ANCHOR_RULE
SEED_AXIS_RULE = B.SEED_AXIS_RULE
NOT_RUN_RULE = B.NOT_RUN_RULE
GATE_RULE = B.GATE_RULE
tag_parts = B.tag_parts
seed_tag_of = B.seed_tag_of
selection_key = B.selection_key
tag_of_row_dir = B.tag_of_row_dir
row_dir_of = B.row_dir_of
producer_key_for = B.producer_key_for
parse_result_line = B.parse_result_line


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BigcodebenchSeedrepAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-23): the first version — the seed-replicate BigCodeBench-200 stores from the Modal volume (module docstring)
    #: 2 (2026-09-23): `decode_seed_rule none` — the instrument's and the protocol policy row's rule (_bcbw_common, DECODE POLICY); emission 1 filed
    #: `per_item`, which kept the derived decode-policy match at no_match on every row. The paired tag's own `none` as its anchor (the same day)
    #: changes nothing over the 2026-09-23 mirror on its own
    #: (2026-09-26, no bump: the composed row's fallback contrast -- ACROSS tags against s<N>/none, extra.cross_standup true -- now files anchor_kind
    #: sibling_none through _bcbw_common.anchor_kind_for instead of same_standup_none; the live mirror re-reads byte for byte because every composed row's
    #: own `none` has filed and no contrast crosses standups any more (the dry run over the live mirror))
    #: 3 (2026-10-03, adapter re-anchoring): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`); a comparison with no current anchor parks on the VALUE row's extra.delta_block
    EMISSION_VERSION = 3
    owner = OWNER


def factory() -> BigcodebenchSeedrepAdapter:
    return BigcodebenchSeedrepAdapter()


register(NAME, factory)
