"""LANDING 2 of the SHARED-STANDUP comparison stores of the FULL-suite BigCodeBench rows -> registry (source ``bigcodebench_full_shared_l2``).

WHAT IS RUN (2026-09-30): the SAME comparison tuples as landing 1 (:mod:`bigcodebench_full_shared`) -- the gemma4e4b s1 row (8 tuples + the
``none@rep2`` repeat) and the qwen35_4b s1 row (8 tuples + ``none@rep2`` + ``arditi_allpos``), 19 stores on tree ``082f583`` -- driven again by a DIFFERENT
harness: K = 2 drivers per row on one vLLM standup per row launch (standups ``88f8913d`` gemma, ``5f51cd86`` qwen), followed by a CPU VERIFICATION PASS
(``manifest.row_engine.verification_pass``: mechanism ``M2_replay_vehicle``, state ``verified``, the verifier's Modal function call) that re-stamps RESULT's last
line ``result=COMPLETE status=OK sha=<tree> verified_by=<function call id>`` WITHOUT a ``standup=`` token. Published under
``run-state/full22/bigcodebench_full_shared_l2/`` as a SEPARATE source beside landing 1's ``bigcodebench_full_shared`` (66 rows already filed, byte-identical
before and after), never into it: the landed reader keys rows by ``(tag, key[@rep])`` and mints run ids without the store suffix, so a second run of one tuple
under the landing-1 source would collide with the ingested landing-1 row.

THE READER IS :mod:`bigcodebench_full_shared` PARAMETRIZED ONCE MORE (``dataclasses.replace`` on its :class:`_bcbw_common.TaskSpec`; the same
:class:`_bcbw_common.SharedSpec` with two fields set): the same layout, the same records (``records/launch_sets_shared.json``, the gate records with the standup
fields, ``three_numbers_bigcodebench_full_shared.json`` whose ``source`` spells THIS source, the roster set / pins / item-set index), the same cross-checks and the same
store-attestation rules (:data:`_bcbw_common.SHARED_ROW_DIR_RULE` / :data:`_bcbw_common.SHARED_ENDPOINT_TUPLE_RULE` / :data:`_bcbw_common.SHARED_IT_REPEAT_RULE`),
plus two things landing 2 needs:

* THE RUN-ID DISCRIMINATOR (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`; the choice, written down): a run id is ``identity.run_id(sdfbase_full22, <producer key>,
  <object key>, <eval>, <instrument>)`` -- ``campaign:eval:instrument[:12]:sha16(producer key | object key)`` -- so the source or harness variant enters through the
  PRODUCER KEY and nowhere else (landing 1 discriminated from the solo rows exactly there: ``<tag>_shared/<store key>`` against the solo ``<tag>/<store key>``; the
  instrument is the eval's decode policy + item set, shared by every landing). Landing 2 is another PRODUCER of the same tuples (another harness), so its
  discriminator is the producer key's SOURCE segment: ``SharedSpec.harness_variant = "l2"`` folds into the spelling half, giving the producer key
  ``<tag>_shared_l2/<store key>[@rep<N>]`` (:meth:`_bcbw_common.SharedSpec.producer_spelling_of`; the store-key half and the repeat segment as on landing 1).
  Rejected: a suffix bolted on the finished id (not a function of the producer's identity, unparseable by ``identity.parse_run_id``); a new instrument (the same
  decode policy and item set: one instrument, the rule that instrument identity is measurement identity); leaning on the store key's trailing hash (it differs
  from landing 1's under K = 2 -- ``none__controls__t0__a0__2ff9cdff__1dd7b444`` against the solo ``...__a03d0582`` -- but an identity that relied on it would collide
  the day a landing hashed alike, and :data:`_bcbw_common.SHARED_RUN_ID_RULE`'s "byte-identical store key" premise is landing 1's, not a rule for landing 2).
  The REGISTRY SPELLING stays ``<tag>_shared`` (``view_legacy``; legacy_spellings.yaml resolves the same model / fit / transfer as landing 1 -- the landing is
  producer identity, not a spelling), so the launch row's ``view`` and the three-number row's ``view`` keep spelling ``gemma4e4b_s1_shared`` and
  :data:`_bcbw_common.SHARED_SPELLING_RULE` holds unchanged; the launch row's ``solo_run_id`` is still what the SOLO reader mints. Every landing-2 run id is
  distinct from every landing-1 id by construction (the producer key differs in its spelling segment) -- the dry read of 2026-09-30 checked all 19 against the 66
  ingested landing-1 ids: no collision.
* THE STANDUP RULE ON A VERIFIED STORE (:data:`_bcbw_common.SHARED_VERIFIED_STANDUP_RULE`; refined 2026-09-30):
  landing 1's rule reads the standup id off RESULT's LAST line (``standup=`` must equal ``manifest.row_engine.standup_id``). On a landing-2 store the
  last line is the verifier's re-stamp without the token, so an ABSENT token on the last line is admitted ONLY when ``verification_pass.state == "verified"`` AND
  the manifest's and the endpoint's standup ids agree; the id is then the manifest's, cross-checked against every RESULT line that carries ``standup=`` (the row
  runner's earlier ``result=GENERATED`` line); a PRESENT token that differs refuses by name, verified or not; the re-stamp's ``verified_by`` must equal
  ``verification_pass.function_call_id``; a landing-1 store keeps the last-line rule byte for byte. ``extra.row_engine`` records ``verification_state``,
  ``result_standup_source`` (``earlier line`` / ``manifest only`` / ``last line``), ``result_verified_by``. Later harness re-stamps carry
  ``standup=<manifest id>`` before ``verified_by=``, so the absent-token path covers exactly these 19 stores.

THE LAUNCH SET'S ``own_source``: landing 2's BOUND ``launch_sets_shared.json`` spells ``evals.bigcodebench.own_source`` as the BASE source's name
(``bigcodebench_full_shared``; its PROVENANCE says so). The reader binds by ROOT -- the mirror root IS the source -- and records the spelled name as provenance
(``extra.launch_set.own_source_spelled`` / ``own_source_is_base``; the status extra ``launch_set_own_source_spelled``), never a refusal.

THE THREE-NUMBER RECORD NAMES THE SOURCE (:data:`_bcbw_common.THREE_NUMBERS_SOURCE_RULE`): because the root binds the source and mints the
producer keys, ``own_source`` legitimately spells landing 1, and an unverified landing-1 store passes the standup rule, a landing-2 root accidentally populated with
landing-1 stores + records would otherwise file duplicate landing-1 measurements under new ``_l2`` ids. So the record's top-level ``source`` (the SAME file name in both
landings) must equal this reader's source before any row files; another name refuses every row by name ("the three-numbers record spells source X; this reader is Y --
wrong root or wrong mirror"). Every bcbw reader holds the rule (every real record spells its reader's source, so landing 1's records and status stay byte for byte);
the landing-2 readers record the header as the status extra ``three_numbers_source``.

THE FRAME: the exclude ruling ``r2v4-exclude-shared-standup-comparison-sources-from-the-paper-frame`` is scoped ``match.lane_sources_legacy [bigcodebench_full_shared,
bfcl_full_shared]`` and the paper views match that list by EXACT equality against ``run.lane_source_legacy``, so it does NOT cover this source's rows: a same-id
re-file widened to the two ``_l2`` names is drafted for the registry's owner to file (a ruling, not a reader's
decision, as on landing 1).

MEASURED on the REAL mirror (2026-09-30, ``run-state/full22/{bigcodebench,bfcl}_full_shared_l2/``, 19 stores each, read-only under strace: the one write was the
diagnostic's own output file): the discriminator, the verified standup rule, the row-dir tail, the endpoint map (every endpoint AGREES with its store's tuple: landing 2
fixed :data:`_bcbw_common.ENDPOINT_SERVED_KEY_BUG`, no note filed) and the base-source ``own_source`` all pass on all 19 stores of both readers; every run id is
distinct from the 66 ingested landing-1 ids. 11 runs per reader filed as published; the 8 refusals per reader (the four unsteered / prompted rows of each model row:
``none``, ``none@rep2``, ``prompted__safety_refusal``, ``prompted__si_constitution``) are STORE-SIDE, in the producer's three-number record: landing 2's writer stamps
``object_sha256`` = the row's FIRST fitted object's digest (``object_sha256_source: route.object_artifact_sha256_by_cell``; gemma ``8b2cd115...`` = the
``si_constv2_mix_full__lora_r4_ce`` entry, qwen ``ba2212f0...`` = the ``arditi_allpos`` entry) on rows that serve NO fitted object, where landing 1's record spelled
null and the store's own ``provenance.artifact_sha256`` is null -- ``_cross_check_three`` refuses by name (``object_sha256: producer '8b2cd115...', re-derived None``).
With those eight rows' two fields nulled in a read-only in-memory diagnostic the readers file 19 runs / 0 refusals each (17 ``same_standup_none`` + 19 ``borrowed``
contrasts per reader; 0 collisions). The fix is the producer's re-emission of the two three-number records; no reader change admits a producer record that names an
object the store did not serve.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from . import _bcbw_common as B
from . import bigcodebench_full as SOLO
from . import bigcodebench_full_shared as L1
from .base import register

NAME = "bigcodebench_full_shared_l2"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): landing 2 of the COMPARISON source, never a paper spelling
SOURCE = "bigcodebench_full_shared_l2"
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
HARNESS_VARIANT = B.SHARED_L2_HARNESS_VARIANT
SHARED = dataclasses.replace(L1.SHARED, harness_variant=HARNESS_VARIANT, base_source=BASE_SOURCE)
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
SHARED_VERIFIED_STANDUP_RULE = B.SHARED_VERIFIED_STANDUP_RULE
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
    """``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared``: the registry spelling a landing-2 row files under -- landing 1's, unchanged (the landing is not a spelling)."""
    return SPEC.spelling_of(tag)


def producer_spelling_of(tag: str) -> str:
    """``gemma4e4b_s1`` -> ``gemma4e4b_s1_shared_l2``: the producer key's spelling segment (:data:`_bcbw_common.SHARED_L2_RUN_ID_RULE`)."""
    return SHARED.producer_spelling_of(tag)


def run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The LANDING-2 run id (producer key ``<tag>_shared_l2/<store key>[@rep<N>]``): never landing 1's over the same store key, never the solo reader's."""
    return B.run_id_for(SPEC, tag, store_key, okey, instrument_id, repeat=repeat)


def landing1_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str, repeat: Optional[int] = None) -> str:
    """The id landing 1's reader mints over the same store key -- what this reader's id must never equal."""
    return L1.run_id_for(tag, store_key, okey, instrument_id, repeat=repeat)


def solo_run_id_for(tag: str, store_key: str, okey: str, instrument_id: str) -> str:
    """The id the SOLO reader mints for the solo store -- what a launch row's ``solo_run_id`` must equal."""
    return SOLO.run_id_for(tag, store_key, okey, instrument_id)


def recount_computation_id() -> str:
    return SPEC.recount_computation_id()


class BigcodebenchFullSharedL2Adapter(B.BcbwSeedrepReader):
    name = NAME
    SPEC = SPEC
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 1 (2026-09-30): the first version -- landing 2's stores through landing 1's reader parametrized (module docstring): the `_l2` producer key
    #: (SHARED_L2_RUN_ID_RULE), the verified store's standup rule (SHARED_VERIFIED_STANDUP_RULE), the bound launch set's base-source `own_source` recorded
    #: 2 (2026-10-03): the anchor currency on every contrast (`_bcbw_common.CURRENCY_RULE`)
    #: 3 (2026-10-06, migration 0027): the stand-in contrast files anchor_kind `solo_stand_in` (landing 1's rule)
    EMISSION_VERSION = 3
    owner = OWNER


def factory() -> BigcodebenchFullSharedL2Adapter:
    return BigcodebenchFullSharedL2Adapter()


register(NAME, factory)
