"""LANDING 3 of the SHARED-STANDUP comparison stores of the FULL-suite BigCodeBench rows -> registry (source ``bigcodebench_full_shared_l3``).

WHAT IS RUN (the fan-out test, published 2026-10-01 02:28Z): the gemma4e4b s1 row's FOUR tuples -- ``none``, ``prompted__safety_refusal``,
``prompted__si_constitution``, ``constv2-mix-full__psr_ce_all`` -- on tree ``f1347e9cf990ab7dbfa4801ee72be51403bf229e``, driven once more by the row-engine harness
(one standup per row launch, ``fedf6a32``; the CPU verification pass ``M2_replay_vehicle``, state ``verified``) and published under ``run-state/full22/bigcodebench_full_shared_l3/``
as a THIRD source beside landing 1's ``bigcodebench_full_shared`` and landing 2's ``bigcodebench_full_shared_l2``, never into either: the landed reader keys rows by
``(tag, key[@rep])`` and mints run ids without the store suffix, so a third run of one tuple under an earlier landing's source would collide with its ingested row.
Four gate records: ``none`` PASS ``cross_standup`` against the SOLO ``none`` record (tree ``8ba0b48``); the other three PASS ``same_standup`` against this run's ``none``.
``launch_sets_shared.json`` is landing 3's own bound set (8 of 8 standup ids bound; ``evals.<eval>.own_source`` spells the BASE source, as landing 2's does);
``three_numbers_bigcodebench_full_shared.json`` keeps landing 1's FILE NAME and spells ``source`` as THIS source.

THE READER IS :mod:`bigcodebench_full_shared` PARAMETRIZED AS ON LANDING 2 (:mod:`bigcodebench_full_shared_l2`; ``dataclasses.replace`` on landing 1's spec with
:class:`_bcbw_common.SharedSpec` ``harness_variant = "l3"`` / ``base_source``, plus landing 3's own :data:`_bcbw_common.SHARED_L3_RULE` as ``landing_rule``): the run-id
discriminator is the producer key's SOURCE segment, ``<tag>_shared_l3/<store key>[@rep<N>]`` (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`, the rule landing 2 wrote
down -- a run id is a function of the producer's identity and another harness landing is another producer; never a suffix on the finished id, never the instrument,
never the store key's hash), so every landing-3 id differs from landing 1's, landing 2's and the solo reader's over one store key while the registry spelling stays
``<tag>_shared`` (``view_legacy``; legacy_spellings.yaml unchanged) and the launch row's ``solo_run_id`` is what the SOLO reader mints. The three-number record must
spell this source (:data:`_bcbw_common.THREE_NUMBERS_SOURCE_RULE`): the ``_l2`` reader over this root refuses every row by name on that header alone (checked
2026-10-01: 4 refusals, 0 runs), and so does this reader over a landing-2 root.

THREE SHAPES DIFFER FROM LANDING 2's 19 STORES, each checked against the reader before this module was written, none a gate
(:data:`_bcbw_common.SHARED_L3_RULE`, stamped on every row as ``extra.shared_standup.landing_rule`` and on the status as ``landing_rule``):

* THE VERIFIED BLOCK HAS FOURTEEN KEYS -- the V1 nine plus ``verifier_tree_sha``, ``generation_tree_sha``, ``executed_image``, ``executed_verify_timeout_s``,
  ``binding_version`` (``contracts.VERIFICATION_PASS_VERIFIED_FIELDS``; ``VERIFICATION_PASS_VERIFIED_FIELD_SETS`` admits either set). The registry reader pins the
  ``row_engine`` block's EIGHT keys (:data:`_bcbw_common.ROW_ENGINE_KEYS`; these manifests carry exactly those), reads the verified block's ``state`` and
  ``function_call_id`` alone, and RECORDS the block verbatim (``extra.row_engine.verification_pass``, ``extra.gate.verification_pass``) -- it imports nothing from
  ``capeval.tinycap.contracts`` and pins neither nine nor fourteen. Both shapes read on every landing's reader (pinned by the tests).
* EVERY RESULT RE-STAMP CARRIES ``standup=<manifest id>`` BEFORE ``verified_by=`` (the later shape; the first ``phase=generate`` line carries no token and is
  skipped, the row runner's GENERATED line carries the same id). The standup id is read off the LAST line as on landing 1 (``extra.row_engine.result_standup_source``
  = ``last line``); :data:`_bcbw_common.SHARED_VERIFIED_STANDUP_RULE`'s absent-token branch is not taken on any landing-3 store, and ``verified_by`` is held equal to
  ``verification_pass.function_call_id`` as on landing 2.
* ``manifest.route.object_artifact_sha256`` (THE LEGACY SCALAR) IS NON-NULL ON ALL FOUR STORES, ``none`` and the two prompted ones included: the row served exactly
  one fitted object, and the harness spelled its digest into the scalar of every store. The per-store identity is right -- ``provenance.artifact_sha256`` and
  ``fit_provenance.object_sha`` are null on ``none`` / prompted and the pinned digest on the fitted store; the by-harness-key map (``served_geometry.json:endpoint``
  = ``route.object_artifact_sha256_by_cell``) carries the fitted key alone; the three-number rows spell ``object_sha256`` None on ``none`` / prompted. The reader never
  reads the scalar as identity (the only read in :mod:`_bcbw_common` RECORDS it as ``extra.row_engine.object_artifact_sha256_scalar``; the unsteered / prompted
  decision fires on ``fit_provenance.object_sha`` / ``provenance.artifact_sha256`` and on the map entry for the store's own harness key), so ``none`` files as
  ``none`` / passthrough, the prompted stores as ``prompted_only``, and the fitted store as the pinned object -- proved in the dry read of 2026-10-01 and pinned by
  the tests (a fixture whose scalar spells the fitted digest on every store).
* (the fourth note) ``row_engine.row_dir`` is volume-absolute, ``/vol/tinycap/gen/rows/gemma4e4b-f22s1/bcbw-gemma4e4b_s1-f1347e9-shared``, the spelling landing 2
  carried: :data:`_bcbw_common.SHARED_ROW_DIR_RULE` binds by the ``rows/<row>/<suffix>`` tail.

THE FRAME: a COMPARISON source, never in the paper frame. The exclude ruling ``r2v4-exclude-shared-standup-comparison-sources-from-the-paper-frame`` COVERS this
source: its same-id re-file of 2026-10-01T02:31:02Z (the registry's ``ruling-add`` record and the live ``ruling`` row carry the filed scope) widened
``match.lane_sources_legacy`` to the six shared-standup names, landings 1, 2 and 3. The paper frame matches that list by exact equality against
``run.lane_source_legacy``, so a new landing's name must be added there before its first ingest or its rows would read ``selected`` beside their solo twins --
a ruling, not a reader default, as on landing 1; this pair was, before this reader landed.

DRY READ (2026-10-01, this reader over the published root, read-only, no ingest): 95 inputs, group ``gemma4e4b_s1``, 4 runs / 0 refusals / 0 malformed; the contrasts
3 ``same_standup_none`` anchored on this landing's ``none`` + 4 ``borrowed`` anchored on the launch rows' ``solo_run_id``s (all present in the live database under
``bigcodebench_full``); the four run ids disjoint from the landing-1, landing-2 and solo readers' ids over the same store keys and absent from the live database.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from . import _bcbw_common as B
from . import bigcodebench_full as SOLO
from . import bigcodebench_full_shared as L1
from . import bigcodebench_full_shared_l2 as L2
from .base import register

NAME = "bigcodebench_full_shared_l3"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): landing 3 of the COMPARISON source, never a paper spelling
SOURCE = "bigcodebench_full_shared_l3"
OWNER = B.OWNER
PRODUCER_SESSION = B.PRODUCER_SESSION
CAMPAIGN = B.CAMPAIGN
MODEL_KEYS: Tuple[str, ...] = tuple(SOLO.MODEL_KEYS)
EVAL_ID = SOLO.EVAL_ID
TASK = SOLO.TASK
RECORD_TASK = SOLO.RECORD_TASK
OUTCOMES_FILE = SOLO.OUTCOMES_FILE
INSTRUMENT_SEED = SOLO.INSTRUMENT_SEED
ITEM_SET_ID = SOLO.ITEM_SET_ID
#: the solo source the comparison stands against (unchanged from landing 1) and the landing-1 source whose name the bound launch set spells (recorded, never bound)
SOLO_SOURCE = SOLO.SOURCE
BASE_SOURCE = L1.SOURCE
HARNESS_VARIANT = B.SHARED_L3_HARNESS_VARIANT
LANDING_RULE = B.SHARED_L3_RULE
SHARED = dataclasses.replace(L1.SHARED, harness_variant=HARNESS_VARIANT, base_source=BASE_SOURCE, landing_rule=LANDING_RULE)
#: the producer's three-number record beside the stores keeps landing 1's FILE NAME (its `source` field spells this source); the launch set's ONE name; the envelope
THREE_NUMBERS_FILE = L1.THREE_NUMBERS_FILE
LAUNCH_SET_FILES = L1.LAUNCH_SET_FILES
SCHEDULE_FILE = L1.SCHEDULE_FILE
CAP = SOLO.CAP
LADDER: Tuple[int, ...] = SOLO.LADDER
PROTOCOL_TEMPERATURE = SOLO.PROTOCOL_TEMPERATURE
SPEC = dataclasses.replace(L1.SPEC, name=NAME, source=SOURCE, root_name=SOURCE, shared=SHARED)
ROOT_DEFAULT = SPEC.root_default
SOLO_ROOT_DEFAULT = SHARED.solo_root_default
RECOUNT_ALGORITHM = SPEC.recount_algorithm
RECOUNT_VERSION = B.RECOUNT_VERSION
RECOUNT_SEED_PROVENANCE = SPEC.recount_seed_provenance
SHARED_SOURCE_RULE = B.SHARED_SOURCE_RULE
SHARED_RUN_ID_RULE = B.SHARED_RUN_ID_RULE
SHARED_L2_RUN_ID_RULE = B.SHARED_L2_RUN_ID_RULE
SHARED_L3_RULE = B.SHARED_L3_RULE
SHARED_VERIFIED_STANDUP_RULE = B.SHARED_VERIFIED_STANDUP_RULE
THREE_NUMBERS_SOURCE_RULE = B.THREE_NUMBERS_SOURCE_RULE
SHARED_ROW_KEY_RULE = B.SHARED_ROW_KEY_RULE
SHARED_SPELLING_RULE = B.SHARED_SPELLING_RULE
SHARED_IT_NONE_RULE = B.SHARED_IT_NONE_RULE
SHARED_STANDUP_RULE = B.SHARED_STANDUP_RULE
SHARED_COMPARISON_RULE = B.SHARED_COMPARISON_RULE
SHARED_ROW_DIR_RULE = B.SHARED_ROW_DIR_RULE
SHARED_ENDPOINT_TUPLE_RULE = B.SHARED_ENDPOINT_TUPLE_RULE
SHARED_IT_REPEAT_RULE = B.SHARED_IT_REPEAT_RULE
VERIFICATION_STATE_VERIFIED = B.VERIFICATION_STATE_VERIFIED
COMPARISON_KINDS = B.COMPARISON_KINDS
ANCHOR_KIND_BORROWED = B.ANCHOR_KIND_BORROWED


def spelling_of(tag: str) -> str:
    """``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared``: the registry spelling a landing-3 row files under -- landing 1's, unchanged (the landing is not a spelling)."""
    return SPEC.spelling_of(tag)


def producer_spelling_of(tag: str) -> str:
    """``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared_l3``: the producer key's spelling segment (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`, the variant ``l3``)."""
    return SHARED.producer_spelling_of(tag)


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The LANDING-3 run id (producer key ``<tag>_shared_l3/<store key>[@rep<N>]``): never landing 1's or landing 2's over the same store key, never the solo reader's."""
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id, repeat=repeat)


def landing1_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The id landing 1's reader mints over the same store key -- what this reader's id must never equal."""
    return L1.run_id_for(tag, store_key, okey, instrument_id, repeat=repeat)


def landing2_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The id landing 2's reader mints over the same store key -- what this reader's id must never equal either."""
    return L2.run_id_for(tag, store_key, okey, instrument_id, repeat=repeat)


def solo_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    """The id the SOLO reader mints for the solo store -- what a launch row's ``solo_run_id`` must equal."""
    return SOLO.run_id_for(tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BigcodebenchFullSharedL3Adapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-10-01): the first version -- landing 3's four gemma stores through landing 1's reader parametrized as on landing 2 (module docstring): the `_l3`
    #: producer key, the fourteen-key verified block recorded, the token-present re-stamp read off the last line, the non-null route scalar recorded and never read
    #: as identity (SHARED_L3_RULE stamped as extra.shared_standup.landing_rule)
    #: 2 (2026-10-03, adapter re-anchoring): the anchor currency on every contrast (`_bcbw_common.CURRENCY_RULE`)
    #: 3 (2026-10-06, migration 0027): the stand-in contrast files anchor_kind `solo_stand_in` (landing 1's rule)
    EMISSION_VERSION = 3
    owner = OWNER


def factory() -> BigcodebenchFullSharedL3Adapter:
    return BigcodebenchFullSharedL3Adapter()


register(NAME, factory)
