"""The batch-64 three-seed Qwen family's BFCL-200 stores -> registry (source ``bfcl_seedrep``).

WHAT WAS RUN: the SAME stores as ``bigcodebench_seedrep`` (the seed-replicate run on the b64-capability-agentic-queue; Modal, the ``twosi-fits``
volume, tree ``8365992042b6a4836a27593ea4364b78871132d9``; the launch set names 30 rows, 25 with stores at the first mirror) — one store per
(seed tag, object key) holds both tasks, and this reader owns its OWN byte-for-byte copy under ``run-state/full22/bfcl_seedrep/`` (never a
symlink, never a cross-read). Every row is the BFCL-200 instrument — the BFCL suite (V1 U V2 minus rest) as row task ``inspect/bfcl_200`` over the
200 pinned items (of 3,981; ``manifest.json:tinycap.bfcl_item_set``), one turn of tool calls scored by AST match, thinking OFF, the benchmark's
reference protocol sampler (T 0.001 / top_p 1 / top_k 0, executed LITERALLY under the served stack's clamp override — vLLM's ``_MAX_TEMP`` 0.01 ->
0.001, ``manifest.json:tinycap.served.sampling_clamp_override``; the registered row ``qwen35_4b_it|bfcl|off|protocol@v1``), ONE pass at
1,024 (``contracts.BFCL200_MAX_TOKENS_OFF``, the registry's BFCL row; no cap step) — served LoRA-natively on ``Qwen/Qwen3.5-4B @ 851bf6e8``
with the tool-call route (``route.tool_calls``), under the DEFAULT Inspect seed rule. Every row files on ``bfcl_200_protocol_T0001_1024``
bound to the served model — its own instrument, never the statics' caps.

THE TOOL PARSER: the qwen3_5 family's live tool-parser verdict (``TOOL_PARSER_LIVE_PASS`` at the producer's tree: ``qwen3_5 @ 80df464``)
gated the launch, but neither the store nor the gate record spells the parser's name — the store attests the tool-call route (``route.tool_calls``,
``tinycap.served.tool_calls``, the standup's ``--tool-calls``) and no parser spelling, so the run carries ``extra.tool_calls`` and nothing is
invented about the parser here.

Everything else — the layout, the gate records, the launch set, the pins, the launched roster, the three-number record, the recount, the
cross-checks, the anchors and the not-run rule — is the shared reading in :mod:`_bcbw_common`. This module binds the :class:`_bcbw_common.TaskSpec` alone.
"""
from __future__ import annotations

from typing import Tuple

from . import _bcbw_common as B
from .base import register

NAME = "bfcl_seedrep"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these stores
SOURCE = "bfcl_seedrep"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
MODEL_KEY = B.MODEL_KEY
EVAL_ID = "bfcl"
TASK = "inspect/bfcl_200"
RECORD_TASK = "bfcl_200"
OUTCOMES_FILE = "inspect__bfcl_200.json"
#: the BFCL-200 instrument (configs/registry/instruments.yaml), bound per model by the dimensions loader
INSTRUMENT_SEED = "bfcl_200_protocol_T0001_1024"
#: contracts.BFCL200_MAX_TOKENS_OFF: the one pass's cap (= the registry's BFCL row)
CAP = 1024
LADDER: Tuple[int, ...] = (CAP,)
N_ITEMS = B.N_ITEMS
#: BFCL's reference protocol: the generation script's default temperature 0.001, executed literally under the clamp override
PROTOCOL_TEMPERATURE = 0.001
THREE_NUMBERS_FILE = "three_numbers_bfcl_qwen_seedrep.json"
ITEM_SET_KEY = "bfcl_item_set"
GENERATION_STATE = "generated; rule-scored at generation (BFCL: AST match of the emitted tool call, one turn), no judge"
SPEC = B.TaskSpec(name=NAME, source=SOURCE, eval_id=EVAL_ID, task=TASK, record_task=RECORD_TASK, outcomes_file=OUTCOMES_FILE, instrument_seed=INSTRUMENT_SEED, cap=CAP,
                  protocol_temperature=PROTOCOL_TEMPERATURE, three_numbers_file=THREE_NUMBERS_FILE, item_set_key=ITEM_SET_KEY, generation_state=GENERATION_STATE,
                  clamp_override=True, root_name=SOURCE)
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


class BfclSeedrepAdapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-23): the first version — the seed-replicate BFCL-200 stores from the Modal volume (module docstring)
    #: 2 (2026-09-23): `decode_seed_rule none` — the instrument's and the protocol policy row's rule (_bcbw_common, DECODE POLICY); emission 1 filed
    #: `per_item`, which kept the derived decode-policy match at no_match on every row. The paired tag's own `none` as its anchor (the same day)
    #: changes nothing over the 2026-09-23 mirror on its own
    #: (2026-09-26, no bump: the composed row's fallback contrast -- ACROSS tags against s<N>/none, extra.cross_standup true -- now files anchor_kind
    #: sibling_none through _bcbw_common.anchor_kind_for instead of same_standup_none; the live mirror re-reads byte for byte because every composed row's
    #: own `none` has filed and no contrast crosses standups any more (the dry run over the live mirror))
    #: 3 (2026-10-03): every contrast is decided by the pass's anchor currency and carries
    #: extra.anchor_selection (`_bcbw_common.CURRENCY_RULE`); a comparison with no current anchor parks on the VALUE row's extra.delta_block
    EMISSION_VERSION = 3
    owner = OWNER


def factory() -> BfclSeedrepAdapter:
    return BfclSeedrepAdapter()


register(NAME, factory)
