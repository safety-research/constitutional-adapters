"""The Petri seed-1 store -> the registry (source ``seedrep_petri``): the legacy-store reader that folds the
completed Petri audit of the seed-replicate battery's first seed onto the runs the premint queued for it.

THE STORE (``STORE_ROOT_DEFAULT``, the ONE root this reader is pinned to; never a glob): the ``petri_native`` root the Modal drive
``20260919T111239Z.110143`` wrote (proxy attempt ``01M2WNTNMW8AQ7CJHTC1D3Y5SR-79e88ba275a1``, tree ``c65b47ba…``), ``cells.json`` listing
the eight audited objects with their ``petri_run_key`` (plus the harness's derived no-op control ``none__lora0``, never a run),
``modal_drive.json`` (the drive record: the tree, the attempt, the schedule, the verdict — 8 complete, 0 failed), and per object
``cells/<petri_run_key>/{manifest.json, results.json, rows.jsonl, petri_incomplete.json[, petri_judge_refused.json]}``. A complete audit is
6 behaviours x 10 scenarios x 3 dimensions = 180 rows; every scenario RAN (``petri_incomplete.json`` names no withheld item anywhere) and the
13 missing rows across the store are TERMINAL JUDGE REFUSALS (``petri_judge_refused.json``: the opus-5 judge's content filter refused to
score a finished transcript) — refusals to score, never unrun scenarios.

WHAT IS FILED. One run per (object, behaviour eval), on the identity premint-queued under producer key
``seedrep_battery:6fdd5e5:petri:s1`` (:data:`PREMINT_PRODUCER_KEY`; the plan the roster selects — the earlier premint of the same rows,
``…:6d69006:…``, was retired by ruling and nothing is emitted on it): ``identity.run_id(seedrep_battery, that key, <the roster's staged id of
the object: base64s1__<stem> / derived_base64s1__<stem>, none, prompted__<si>>, <eval>, <the premint's instrument for the eval>)``. The
instrument and judge ids are READ from the premint-queued rows of the connected registry (the battery's derived Petri instruments are not
seeds: their identity folds the tree, so this reader binds to the premint rather than re-deriving a harness_version string), the object
through :func:`jobplan.object_id_for` over the roster's ``s1`` selection and the pins (the harness's own resolution), and where a queued row
exists its run id, object and trial must equal what the reader derives — a disagreement refuses the object by name. The three objects the
premint did not queue for Petri (the derived difference object and the two prompted system instructions) file as NEW runs under the same
producer key and instrument, flagged ``extra.admission.premint_row = absent``. The fold advances each queued row ``queued -> generated ->
judged`` (two ``run_state`` events, evidence the object's ``results.json``); the harvester's completion stamps ``ingested`` once the judgment
covers the run (``n_items_judged + n_judge_refusals >= 10``).

COUNTS AND VALUES (re-verified from the files): ``n_items_planned`` 10 per behaviour eval;
``n_items_generated`` 10 (scored + judge-refused — every scenario ran); ``n_items_judged`` = the SCORED count; the metric ``petri_score`` is
the mean over scored scenarios with ``n`` = the scored count and ``n_abstentions`` = the refusals (never 10 by default), checked against the
producer's own ``results.json`` mean, and a percentile bootstrap over the scored scenarios beside it. Item rows: one per (behaviour, scenario)
scored (``item_id <behaviour>/<scenario>``, ``turn_index -1``, ``metric_id petri_score``, plus the two shared dimensions ``eval_awareness`` /
``scenario_realism`` on their own metric ids), and one per REFUSED scenario with ``value NULL`` and ``label judge_refused`` (relu_paired's label
for the same store shape). Contrasts: each steered object against the store's own ``none`` per behaviour, ``anchor_kind same_standup_none``,
PAIRED ON SCENARIO ID over the scenarios scored in BOTH objects (``n_pairs`` = the shared count, the unshared counts in ``extra.pairing``):
paired mean difference, percentile bootstrap interval (10,000 draws, seed 0), exact two-sided sign test, d_z. No unpaired delta is filed —
the refusals are structured by object (4 on the prompted constitution, 3 on ``none``, 0 on the ReLU constitution and the difference), so an
unpaired difference would compare different scenario sets.

THE POOLED RUN (v3): the seed-replicate Petri runs file ``petri_pooled`` beside the six behaviour evals, the way relu_paired and the r2
export file theirs. ONE run per (seed, object) with eval ``petri_pooled`` (:data:`POOLED_EVAL`), DERIVED by
this reader from the SAME per-object files the six behaviour runs are read from — never from a producer "pooled" artifact (no
``petri_analysis.json``, no ``petri_stats`` sidecar). Item set ``petri_scenarios_6x10``, metric ``petri_score``: the value is the MEAN OF THE SIX
BEHAVIOUR MEANS (relu_paired's ``mean_of_behaviour_means``; equal to the digit to the six behaviour runs' values averaged), ``n`` = the scored
(behaviour, scenario) items, ``n_items_planned`` 60 (``results.json`` ``audited.expected``), ``n_items_generated`` = scored + judge-refused +
withheld, ``n_abstentions`` = the judge refusals PLUS the withheld items of ``petri_incomplete.json`` (a refused item is an abstention, a withheld
item is unresolved — neither is silently dropped; ``n_incomplete`` counts the withheld alone), a STRATIFIED percentile bootstrap (scenarios
resampled within each behaviour, the mean of behaviour means recomputed; 10,000 draws) beside it. Every (behaviour, scenario) item is filed
AGAIN under the pooled run (``petri_score``, the two shared dimensions, the refused items as ``judge_refused``) so the pooled contrast can be
re-derived from items; the pooled contrast pairs each steered object's pooled run against the seed's ``none`` pooled run on
``<behaviour>/<scenario>`` over the items scored in both (the same ``anchor_kind`` rule as the behaviour contrasts). The pooled INSTRUMENT is
minted per seed the way the six per-eval instruments are: for the minted binding from the seeded ``petri_full22_as_run`` row of ``petri_pooled``
with the store's declared changes; for the premint binding from the premint's own six instrument rows (they agree on every identity field but
``eval_id``; the pooled one is that row with ``eval_id = petri_pooled``, ``extra.premint_family`` = the producer key) — unless the premint key set
itself queues a ``petri_pooled`` row, in which case the reader binds to it exactly as it binds the six. The premint queues none (checked
2026-09-21), so the s1 pooled runs are this reader's own ids under the premint's producer key (``extra.premint_family``; single-writer rule: no
other reader writes them). The pooled JUDGMENT rolls the per-item judge-chain markers up: ``judge_models_used`` = the count of items per judge
model (a mix of opus-5 and opus-4-8 records the mix, never one of them), the fallback / auditor-fallback counts and scenarios per behaviour,
``refusal_kinds``, the re-judged / kept counts, and the six behaviour judgment ids. A withheld audit files its pending disposition for
``petri_pooled`` too (seven per object). An object whose six behaviour runs did not all file has no pooled run (refused by name).

PETRIWAVE'S PER-ROW JUDGE PROVENANCE (tolerated, never required): the producer re-judges its opus-4-8-scored items with opus-5 and rewrites each
object's ``rows.jsonl`` / ``results.json`` IN PLACE (same row ids and keys, new values, a new ``results.json`` sha and a rewritten ``leg_<n>.rc``
join line — the v2 pin / join mechanics are unchanged, so a stale sha still refuses the object, never folds it), adding to every score row's
``extra_json`` the fields ``judge_model_used`` / ``judge_chain_display`` / ``judge_chain_advanced`` and either ``rejudged_by`` / ``rejudged_at`` /
``previous_judge`` / ``previous_value`` (the opus-5 score stands) or ``rejudge_attempted_by`` / ``rejudge_attempted_at`` / ``rejudge_outcome`` /
``rejudge_detail`` (the opus-4-8 score kept). Every ``extra_json`` key under the prefixes :data:`PRODUCER_JUDGE_FIELD_PREFIXES` is carried onto the
item's ``extra`` verbatim; a row's own ``judge_model_used`` is the producer's record and wins over the mirror's derivation
(``extra.judge_model_used_source`` names which; a disagreeing mirror value is parked as ``judge_model_used_mirror``). Absent fields are fine.

PROVENANCE THAT TRAVELS ON EVERY RUN (``extra.admission``, ``field_confidence_json.admission``; both facts required): (1) the in-container
determinism spot-check was WAIVED for these carried legs (:data:`WAIVER_DEFAULT`, read and checked against the drive's tree
and seed; the sampled Petri auditor chain never byte-reproduces, and re-running completed work is refused); (2) the waiver's compensating
control — a re-audit by later readers — NEVER RAN: no output exists. The rows do not read as if they passed that spot-check. ``code_sha`` is
the tree that DROVE the audit (``modal_drive.json`` W = proxy_sha, ``c65b47ba…``); the premint's carrying tree (``6fdd5e52…``) rides
``extra.admission.carried_under_tree``.

REFUSED BY NAME (:class:`lineage_common.RowRefusals`, in ``source_state`` and ``status_extra``): a listed object whose files are missing or
whose manifest / results / drive schedule disagree on the run key; an object ``cells.json`` names ``absent``; an audit with withheld items;
a behaviour whose rows do not reproduce its own ``results.json`` (count or mean), whose scored and refused scenarios overlap or do not sum to
the bundle's 10; an object the roster / pins / si_texts cannot name or whose object row is not in the connected registry; a queued row whose
run id, object or trial differ from the derived identity; a decode the store sent that differs from the registered policy row; a premint
instrument that disagrees with the store's manifest (cap, decode, seed rule, serving path, scenario bundle, auditor). REFUSED AS A
WHOLE (:class:`AdapterRefusal`): a root that is not the pinned s1 store (another s1 store is a second instrument on a completed
measurement, not a replicate: s1 is DONE); two configured roots declaring the same seed; a missing waiver file; no
premint-queued Petri rows under the producer key in the connected registry (nothing to fold onto — this reader mints no identity of its own); a
producer key whose premint instruments an ``exclude`` ruling has retired (folding a completed store onto excluded identities would ingest rows
the paper frame drops in silence).
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import decode_policy as DP
from .. import identity as I
from .. import jobplan as JP
from .. import records as R
from . import _items_common as IC
from . import declared_common as DC
from . import lineage_common as LC
from . import relu_paired as RP
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, register
from .petri_stats import PCELL_REV
from .runner import source_id_for
from ...vault import wif_org_of_ids_sha12


def drive_pool_key(drive: Mapping[str, Any]) -> str:
    """The pool a DRIVE_START drive counts under in `_pools_of`: a key drive's `auditor_org=` label (the daemon proved it against the vault item's
    hash at drive time); a federated drive's VERIFIED org -- the one whose current ids file digests to the attested `ids_sha12` -- or, when no
    org's file does, `wif-unverified:<digest>` (so the judgment's api_pool reads None and the extra names the digest; never the label); a drive
    whose DRIVE_START says `secrets_mode=wif` but attests NO federation digest at all (a wave sha from before the attestation, or one still on
    the key path -- an inherited key of any org) is `wif-unverified:unattested` (round 1 on 7778694d0): never the label either."""
    if drive.get("auth") != "wif":
        return str(drive["org"])
    return str(drive.get("org_verified") or f"{WIF_UNVERIFIED_POOL_PREFIX}{drive.get('ids_sha12') or WIF_UNATTESTED_MARK}")

NAME = "seedrep_petri"
#: PROVISIONAL, registry-declared (configs/registry/sources_r2_v1.yaml meta.additions): the paper has declared no spelling for these rows
SOURCE = "seedrep_petri"
OWNER = "Reseed"
PRODUCER_SESSION = "Reseed"
CAMPAIGN = "seedrep_battery"
SEED_LABEL = "s1"                 # the value of `view_legacy` on every row (the seed-replicate battery's first seed)
SEED_INDEX = 1
MODEL_ID = "qwen35_4b_it"
MODEL_KEY = "qwen35_4b"
ROSTER_SELECTION = "s1"
#: the producer key of the premint-queued Petri s1 runs the ROSTER selects (the plan step under tree 6fdd5e52); every run id this reader
#: mints is under it. The earlier premint of the same rows (`seedrep_battery:6d69006:petri:s1`) was retired by ruling
#: `premint-retire-instruments-W6d690064-aeee6c30f4a314af` (its 20 instruments superseded by the current ones, which differ only in the tree
#: suffix of harness_version); nothing is emitted on it, and :meth:`_premint` refuses any key whose instruments a ruling has retired
PREMINT_PRODUCER_KEY = "seedrep_battery:6fdd5e5:petri:s1"
#: the tree the premint filed as the run's code_sha: the legs were CARRIED (first under the earlier premint's tree, then under this one
#: when that premint was retired); the store itself was driven by the tree in STORE_PIN["W"]
CARRIED_UNDER_TREE = "6fdd5e52bd6aec62b2389c185bd1d094d902703d"
STORE_ROOT_DEFAULT = os.path.join(LC.HOME, ".local", "state", "petri_native",
                                  "petri_seedrep_c65b47b_c3b864ed_01M2WNTNMW8AQ7CJHTC1D3Y5SR-79e88ba275a1_qwen35_4b_s1")
#: what the pinned store's drive record must say for a root to be THE s1 store (anything else is refused, never folded)
STORE_PIN: Dict[str, str] = {"W": "c65b47ba4e3def0fda7159f497e60900b8ec4631", "drive_id": "20260919T111239Z.110143",
                             "attempt_id": "01M2WNTNMW8AQ7CJHTC1D3Y5SR-79e88ba275a1", "seed_legacy": SEED_LABEL, "audit": "petri_full22_qwen35_4b_s1"}
WAIVER_DEFAULT = os.path.join(DC.RUN_STATE, "seedrep_battery", "criterion3_waiver_w19_petri.json")
WAIVER_DECISION_REF = "D118"
W23_REAUDIT_STATE = "absent"
W23_REAUDIT_NOTE = ("the waiver's compensating control ('carry them under W23 with their stores re-audited by W23's readers') NEVER RAN: "
                    "no output exists (Reseed, 2026-09-21)")
BEHAVIOURS: Tuple[str, ...] = tuple(DC.PETRI_BEHAVIOURS)
SHARED_DIMENSIONS = tuple(RP.PETRI_SHARED_DIMENSIONS)
METRIC = RP.PETRI_METRIC
#: the pooled eval (v3): one run per (seed, object), derived by this reader from the six behaviour runs' own files (relu_paired's precedent)
POOLED_EVAL = RP.PETRI_POOLED
JUDGE_REFUSED_LABEL = "judge_refused"
N_SCENARIOS = 10
#: the pooled run's planned items: the whole scenario bundle (results.json audited.expected; item_sets.yaml petri_scenarios_6x10 n_items)
N_ITEMS_POOLED = len(BEHAVIOURS) * N_SCENARIOS
#: the producer's per-row judge provenance (its add_judge_fields step writes these into every score row's extra_json): every key under one of
#: these prefixes is carried onto the item's extra verbatim — judge_model_used / judge_chain_display / judge_chain_advanced / judge_refusal, the
#: re-judge fields rejudged_by / rejudged_at / previous_judge / previous_value, the kept-score fields rejudge_attempted_by / rejudge_attempted_at /
#: rejudge_outcome / rejudge_detail, judge_fields_added_at; a prefix match so a spelling the producer adds later still travels
PRODUCER_JUDGE_FIELD_PREFIXES: Tuple[str, ...] = ("judge_", "rejudged_", "rejudge_", "previous_")
#: the producer's own per-item judge marker (the model whose score stands); wins over the mirror's derivation when a row carries it
PRODUCER_JUDGE_USED_KEY = "judge_model_used"
PRODUCER_REJUDGED_KEY = "rejudged_by"
PRODUCER_REJUDGE_KEPT_KEY = "rejudge_attempted_by"
N_BOOT = 10000
BOOT_SEED = 0
CONTROL_PREFIX = "none__"
STORE_FILES_REQUIRED = ("manifest.json", "results.json", "rows.jsonl", "petri_incomplete.json")
STORE_FILES_OPTIONAL = ("petri_judge_refused.json",)
ROOT_FILES = ("cells.json", "modal_drive.json")
ROSTER_REL = ("configs", "sdfbase_full22_roster.yaml")
PINS_REL = ("configs", "sdfbase_full22_pins.json")
# ---- the Slurm stores for seeds 2 and 3 (v2): the same layout, NO premint — the reader mints the instrument and the runs
#: the campaign of the cluster-run seed replicates (the seedrep_cluster reader files ReLU's seed-2 / seed-3 statics, MMLU and multi-turn
#: there; the roster's coexist rulings for the unfitted keys are written against it)
CLUSTER_CAMPAIGN = "sdfbase_full22"
PETRI_NATIVE_ROOT = os.path.join(LC.HOME, ".local", "state", "petri_native")
#: the standup state the producer's driver writes per seed and pass: `drive.log` (DRIVE_START lines: drive id, tree sha, cluster, auditor_org),
#: `leg_<n>.rc` (`<rc> drive=<id> results=<sha256 of results.json>`: the byte-level join of a completed audit to the drive that completed it),
#: `endpoint.json` (attested_max_model_len: the served window), `models.json` (petri_proxy: thinking_mode, engine kind)
STANDUP_STATE_ROOT = os.path.join(LC.HOME, ".local", "state", "sdfbase_full22_native", "petri")
PETRIWAVE_STATE_ROOT = os.path.join(LC.HOME, ".local", "state", "petri-wave")
MANIFEST_CONFIG_REL = ("configs", "capeval", "manifests", "petri_full22_qwen35_4b_{seed}.yaml")
PETRI_INSTRUMENT_SEED = "petri_full22_as_run"
JUDGE_SEED = "petri_judge_opus5"
#: petri_runner.OVERFLOW_REASON_PREFIX: the runner's machine-checkable classification of a withheld scenario as a context-overflow casualty
OVERFLOW_REASON_PREFIX = "target context overflow"
RERUN_WINDOW_TOKENS = 32768
#: the driver marks the context-overflow re-run pass on its DRIVE_START line; its standup serves the widened window (the seeded Petri instrument's
#: own declaration: window 16,384, the overflow re-run at 32,768, cap unchanged — one instrument, the re-run is provenance on the run)
RERUN_MARK = "OVERFLOW-RERUN"
#: the producer's per-scenario mirror of the judge chain (`petri_judge_chain` of every sample in the newest .eval per behaviour, which the stdlib cannot
#: read): `fallback_judged` = [[behaviour, scenario, "opus-5 -> opus-4-8 @turn N"], ...] per (seed, paper label); the source of truth for WHICH judge
#: scored each scenario (the fallback-judged ones can be re-scored by a later ruling), so the marker rides every item and judgment
FALLBACK_SUMMARY_DEFAULT = os.path.join(PETRIWAVE_STATE_ROOT, "summary.json")
#: the refusal kinds a terminal judge refusal is classified into from its reason text (the two kinds stay distinguishable on the item)
REFUSAL_KIND_FALLBACK_NO_SCORE = "judge_fallback_no_score"
REFUSAL_KIND_CONTENT_FILTER = "content_filter"
REFUSAL_KIND_OTHER = "other"
DRIVE_START_RE = re.compile(r"^\[(?P<utc>[^\]]+)\]\s+DRIVE_START\s+(?P<target>\S+)\s+drive_id=(?P<drive>\S+).*?\bcluster=(?P<cluster>\S+).*?\bauditor_org=(?P<org>\S+).*?\bsha=(?P<sha>[0-9a-f]{40})")
#: a drive under SISTEER_SECRETS_MODE=wif attests the org it FEDERATED as and the digest of that org's ids file (petri_drive_full22.sh:
#: `auditor_auth=wif:<org> ids_sha12=<sha256(~/.anthropic/wif-<stem>.env)[:12]>`) in place of `auditor_key_sha12=`. The reader VALIDATES the
#: digest against the vault's own over every org's current ids file (`sisteer.vault.wif_org_of_ids_sha12`) and takes THAT org as the pool --
#: the `auditor_org=` label is never trusted as written under federation: an unproven digest makes the pool `wif-unverified:<digest>`
WIF_ATTEST_RE = re.compile(r"\bauditor_auth=wif:(?P<org>\S+)\s+ids_sha12=(?P<sha12>[0-9a-f]{12})\b")
WIF_UNVERIFIED_POOL_PREFIX = "wif-unverified:"
#: the drive's secrets mode as DRIVE_START records it (`secrets_mode=<op|env|wif>`): a `wif` drive WITHOUT the federated attestation (a wave sha from
#: before the field, or one still on the key path under a wif daemon) is a federated drive the reader can prove nothing about -- its pool is
#: `wif-unverified:unattested`, never the `auditor_org=` label (round 1 on 7778694d0: such a log was read as a key drive and its label trusted)
SECRETS_MODE_RE = re.compile(r"\bsecrets_mode=(?P<mode>\w+)\b")
WIF_UNATTESTED_MARK = "unattested"
RC_RE = re.compile(r"^(?P<rc>-?\d+)\s+drive=(?P<drive>\S+)\s+results=(?P<results>[0-9a-f]{64})")
DRIVE_FILES = ("drive.log", "endpoint.json", "models.json")
#: every run files trial 0: the minted instrument names the seed's manifest config in its harness_version, so the unfitted keys (none, the prompted
#: system instructions) sit on a DIFFERENT instrument per seed and never collide across seeds in the frame's collapse key (no decode-seed trials, no
#: coexist ruling needed for them; the seed-1 premint rows are trial 0 on the premint's instruments likewise)


@dataclass(frozen=True)
class StoreDeclaration:
    """One declared store: WHERE it is and HOW its runs are identified. ``premint`` binds to the premint-queued rows of ``producer_key``
    (the instrument and judge read from them; the s1 store). ``minted`` mints the instrument from the seeded Petri row plus the store's
    declared changes and the runs under ``producer_key`` = the store's audit name (the producer's seed-2 / seed-3 stores; no premint exists)."""
    seed_label: str
    seed_index: int
    root: str
    binding: str
    campaign: str
    producer_key: str
    audit: str
    drive_dirs: Tuple[str, ...] = ()
    manifest_config: Optional[str] = None
    waiver_path: Optional[str] = None
    store_pin: Optional[Mapping[str, str]] = None
    fallback_summary: Optional[str] = None

    @property
    def roster_selection(self) -> str:
        return self.seed_label


def premint_store(root: str = STORE_ROOT_DEFAULT, producer_key: str = PREMINT_PRODUCER_KEY, waiver_path: str = WAIVER_DEFAULT,
                  store_pin: Optional[Mapping[str, str]] = None) -> StoreDeclaration:
    pin = dict(store_pin or STORE_PIN)
    return StoreDeclaration(seed_label=str(pin["seed_legacy"]), seed_index=int(str(pin["seed_legacy"]).lstrip("s")), root=LC.expand(root), binding="premint",
                            campaign=CAMPAIGN, producer_key=producer_key, audit=str(pin["audit"]), waiver_path=LC.expand(waiver_path), store_pin=pin)


def minted_store(seed_index: int, root: Optional[str] = None, drive_dirs: Optional[Sequence[str]] = None, manifest_config: Optional[str] = None,
                 campaign: str = CLUSTER_CAMPAIGN, fallback_summary: Optional[str] = FALLBACK_SUMMARY_DEFAULT) -> StoreDeclaration:
    lbl = f"s{int(seed_index)}"
    audit = f"petri_full22_{MODEL_KEY}_{lbl}"
    return StoreDeclaration(seed_label=lbl, seed_index=int(seed_index), root=LC.expand(root or os.path.join(PETRI_NATIVE_ROOT, audit)), binding="minted",
                            campaign=campaign, producer_key=audit, audit=audit,
                            drive_dirs=tuple(LC.expand(d) for d in (drive_dirs or (os.path.join(STANDUP_STATE_ROOT, f"{MODEL_KEY}_{lbl}"),
                                                                                   os.path.join(STANDUP_STATE_ROOT, f"{MODEL_KEY}_{lbl}_second"),
                                                                                   os.path.join(STANDUP_STATE_ROOT, f"{MODEL_KEY}_{lbl}_rerun")))),
                            manifest_config=(manifest_config or DC.repo_path(*[x.format(seed=lbl) for x in MANIFEST_CONFIG_REL])),
                            fallback_summary=(LC.expand(fallback_summary) if fallback_summary else None))


#: the declared stores: the seed-1 store on its premint, the producer's seed-2 and seed-3 stores minted
STORES: Tuple[StoreDeclaration, ...] = (premint_store(), minted_store(2), minted_store(3))
DECODE_KEYS = tuple(DP.PARAM_KEYS)
SIGN_TEXT = "steered minus unsteered (positive = MORE of the behaviour)"


def _is_sha1(s: object) -> bool:
    return isinstance(s, str) and len(s) == 40 and all(c in "0123456789abcdef" for c in s)


def _is_hex16(s: object) -> bool:
    return isinstance(s, str) and len(s) == 16 and all(c in "0123456789abcdef" for c in s)


def epoch_utc(x: object) -> Optional[str]:
    """A producer epoch (seconds, float) -> the registry's UTC-with-ms string; None when absent / not a number."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return None
    return R.epoch_ms_to_utc(int(round(float(x) * 1000)))


def campaign_object_key_legacy(store_key_legacy: str) -> Optional[str]:
    """The store's harness spelling of an object -> the campaign's (petri_stats' table: ``si_constv2_mix_full__psr_ce_all`` ->
    ``constv2-mix-full__psr_ce_all``, ``prompted__safety`` -> ``prompted__safety_refusal``); None for a spelling the table does not carry."""
    if store_key_legacy == "none":
        return "none"
    out = PCELL_REV.get(store_key_legacy)
    return out if out and out != store_key_legacy else None


def refusal_kind(reason: object) -> str:
    """A terminal judge refusal's kind from its reason text: the fallback judge returned no score (a chain opus-5 -> opus-4-8 that ended in
    tool_calls), the judge's content filter refused the transcript, or another reason."""
    t = str(reason or "").lower()
    if "fallback judge produced no score" in t:
        return REFUSAL_KIND_FALLBACK_NO_SCORE
    if "content filter" in t:
        return REFUSAL_KIND_CONTENT_FILTER
    return REFUSAL_KIND_OTHER


def sign_test_p(d: Sequence[float]) -> Optional[float]:
    """Exact two-sided sign test over the NON-ZERO paired differences (binomial, p = 1/2); None with fewer than one non-zero difference."""
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_scenario_contrast(steered: Mapping[str, float], anchor: Mapping[str, float], rng, n_boot: int = N_BOOT) -> Dict[str, object]:
    """Steered minus anchor over the scenarios SCORED IN BOTH readings: the mean difference, a percentile bootstrap interval, the exact sign
    test, d_z (mean / sd of the differences), and the pairing counts (shared, only steered, only anchor)."""
    shared = sorted(k for k in set(steered) & set(anchor) if DC.finite(steered[k]) and DC.finite(anchor[k]))
    out: Dict[str, object] = {"n_pairs": len(shared), "pairing": {"shared": len(shared), "only_steered": len(set(steered) - set(anchor)),
                                                                  "only_anchor": len(set(anchor) - set(steered))}, "shared_scenarios": shared}
    if not shared:
        return out
    x = np.asarray([float(steered[k]) for k in shared])
    y = np.asarray([float(anchor[k]) for k in shared])
    d = x - y
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    sd = float(d.std(ddof=1)) if len(d) > 1 else None
    dz = float(d.mean() / sd) if sd and sd > 0 else None
    try:
        wp = DC.wilcoxon_p(d)
        wilcoxon = float(wp) if wp == wp else None
    except ImportError:   # pragma: no cover — scipy absent: the sign test is the filed p, Wilcoxon is a companion reading
        wilcoxon = None
    out.update(delta=float(d.mean()), delta_lo=float(np.percentile(bs, 2.5)), delta_hi=float(np.percentile(bs, 97.5)), delta_p=sign_test_p(d.tolist()),
               d_z=dz, sd_diff=sd, wilcoxon_p=wilcoxon, mean_steered=float(x.mean()), mean_anchor=float(y.mean()))
    return out


def scenario_bootstrap(scores: Sequence[float], rng, n_boot: int = N_BOOT) -> Tuple[Optional[float], Optional[float]]:
    """A percentile bootstrap interval of the mean over scored scenarios; (None, None) below two scores."""
    v = np.asarray([float(s) for s in scores])
    if len(v) < 2:
        return None, None
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pooled_bootstrap(per_behaviour_scores: Mapping[str, Sequence[float]], rng, n_boot: int = N_BOOT) -> Tuple[Optional[float], Optional[float]]:
    """A STRATIFIED percentile bootstrap interval of the mean of behaviour means: the scored scenarios are resampled WITHIN each behaviour
    (the strata the pooled value averages over) and the mean of behaviour means recomputed per draw; (None, None) when any behaviour holds
    fewer than two scores (its stratum cannot be resampled) or none."""
    strata = [np.asarray([float(s) for s in v]) for v in per_behaviour_scores.values()]
    if not strata or any(len(v) < 2 for v in strata):
        return None, None
    draws = np.zeros(n_boot)
    for v in strata:
        draws += v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    draws /= len(strata)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mean_of_behaviour_means(per_behaviour_means: Mapping[str, float]) -> float:
    """The pooled value: the plain mean of the six behaviour means (relu_paired's ``mean_of_behaviour_means``, petri_stats'
    ``mean_of_behavior_means``) — never the mean over all scored items, which would weight a behaviour by its refusal count."""
    vals = [float(v) for v in per_behaviour_means.values()]
    return sum(vals) / len(vals)


def producer_judge_fields(extra: Mapping[str, Any]) -> Dict[str, Any]:
    """The producer's per-row judge provenance: every ``extra_json`` key under :data:`PRODUCER_JUDGE_FIELD_PREFIXES`, verbatim
    (``judge_model_used``, the re-judge / kept-score fields); ``{}`` for a row that carries none (the v2 stores)."""
    return {str(k): v for k, v in extra.items() if str(k).startswith(PRODUCER_JUDGE_FIELD_PREFIXES)}


class SeedrepPetriAdapter:
    name = NAME
    covers_sources: Tuple[str, ...] = (SOURCE,)
    #: 3 (2026-09-21): the pooled run per (seed, object) on eval `petri_pooled`, derived from the six behaviour runs' own files (its instrument
    #: minted per seed, its items filed again under it, its contrast against the seed's `none` pooled run, the judge-chain markers rolled up onto
    #: its judgment, a seventh pending disposition for a withheld audit), and the producer's per-row judge provenance carried onto every item;
    #: 2 (2026-09-21): the producer's seed-2 / seed-3 stores (the minted binding); 1: the first version, the seed-1 store on its premint
    EMISSION_VERSION = 3
    rng_scope = "group"
    allow_symlinks = False
    owner = OWNER
    #: the store's own reader; nothing else files measurements on these run ids (the premint rows are native plan rows, not a source)
    priority = IC.PRIORITY_OWNING_READER

    def __init__(self, roots: Optional[Sequence[str]] = None, seeds: Optional[LC.Seeds] = None, roster_path: Optional[str] = None,
                 pins_path: Optional[str] = None, waiver_path: Optional[str] = None, policy_path: Optional[str] = None,
                 premint_producer_key: str = PREMINT_PRODUCER_KEY, store_pin: Optional[Mapping[str, str]] = None,
                 stores: Optional[Sequence[StoreDeclaration]] = None, escalation_log: Optional[str] = None):
        if stores is not None:
            self.stores: Tuple[StoreDeclaration, ...] = tuple(stores)
        elif roots is not None:
            # the seed-1 premint binding on the given root(s) only (tests; a second root declaring the same seed refuses)
            self.stores = tuple(premint_store(r, premint_producer_key, waiver_path or WAIVER_DEFAULT, store_pin) for r in roots)
        else:
            self.stores = STORES
        self.roots = tuple(d.root for d in self.stores)
        self.waiver_path = LC.expand(waiver_path or WAIVER_DEFAULT)
        self._seeds = seeds
        # the roster and the pins are THIS checkout's (declared_common.repo_path; never a home-relative worktree path)
        self.roster_path = roster_path or DC.repo_path(*ROSTER_REL)
        self.pins_path = pins_path or DC.repo_path(*PINS_REL)
        self.policy_path = policy_path
        self.premint_producer_key = premint_producer_key
        self.store_pin: Dict[str, str] = dict(store_pin or STORE_PIN)
        self.escalation_log = LC.expand(escalation_log or os.path.join(PETRIWAVE_STATE_ROOT, "escalation.log"))
        self.refusals = LC.RowRefusals(NAME)
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._drive_by_root: Dict[str, Dict[str, Any]] = {}

    @property
    def seeds(self) -> LC.Seeds:
        if self._seeds is None:
            self._seeds = LC.Seeds.locate()
        return self._seeds

    # ---- discovery ----------------------------------------------------------------------------
    def _drive_of(self, root: str) -> Dict[str, Any]:
        """The root's drive record, read once; a present root without one refuses."""
        if root not in self._drive_by_root:
            p = os.path.join(root, "modal_drive.json")
            if not os.path.isfile(p):
                raise AdapterRefusal(root, "a present store root carries no modal_drive.json; the drive record names the tree, the attempt and the seed")
            _inp, drive = LC.checked_json(p, role="report")
            if not isinstance(drive, Mapping):
                raise AdapterRefusal(p, "the drive record is not a JSON object")
            self._drive_by_root[root] = dict(drive)
        return self._drive_by_root[root]

    def _check_pin(self, root: str, drive: Mapping[str, Any]) -> None:
        """A root is read only when its drive record IS the pinned s1 store; another s1 store is refused, never folded (s1 is done: a
        second store for the same seed is a second instrument on a completed measurement, not a replicate)."""
        seen = {"W": drive.get("W"), "drive_id": drive.get("drive_id"), "attempt_id": drive.get("attempt_id"), "seed_legacy": drive.get("view")}
        want = {k: self.store_pin[k] for k in ("W", "drive_id", "attempt_id", "seed_legacy")}
        if seen != want:
            diff = {k: (seen[k], want[k]) for k in want if seen[k] != want[k]}
            raise AdapterRefusal(root, f"not the pinned s1 store: the drive record differs from the declaration on {diff} (seen, declared); a second "
                                       f"store for seed {self.store_pin['seed_legacy']} is a second instrument on a completed measurement, not a "
                                       f"replicate, and is refused — declare it as its own source or rule on it, never fold it here")

    def discover(self) -> List[Input]:
        """Named files of every declared store whose root is present (one directory level at a time; no recursive walk): the root's
        declaration files, every listed object's files, and per binding the waiver (premint) or the drive ledgers + the checkout's manifest
        config (minted). An absent root is a skip (the stores are grading-host-local); a present root is checked against its declaration."""
        present = [d for d in self.stores if os.path.isdir(d.root)]
        if not present:
            return []
        by_seed: Dict[str, List[str]] = {}
        for d in present:
            by_seed.setdefault(d.seed_label, []).append(d.root)
        for seed_legacy, rs in sorted(by_seed.items()):
            if len(rs) > 1:
                raise AdapterRefusal(rs[1], f"two store roots declare seed {seed_legacy!r} ({', '.join(rs)}): the second is a second instrument on a "
                                            f"completed measurement, not a replicate (the user 2026-09-21: s1 is done); refused, not folded")
        out: List[Input] = []
        for d in present:
            cj_path = os.path.join(d.root, "cells.json")
            if not os.path.isfile(cj_path):
                raise AdapterRefusal(d.root, "a present store root carries no cells.json; the store's own declaration of its objects is the reader's inventory")
            if d.binding == "premint":
                self._check_pin(d.root, self._drive_of(d.root))
                for fn in ROOT_FILES:
                    out.append(check_input(os.path.join(d.root, fn), role="report"))
            else:
                out.append(check_input(cj_path, role="report"))
                cj0 = self._json_of(cj_path)
                if str(cj0.get("audit") or "") != d.audit:
                    raise AdapterRefusal(cj_path, f"cells.json names audit {cj0.get('audit')!r}; the declared store for seed {d.seed_label} is {d.audit!r} — another "
                                                  f"store under this root is not read")
                for dd in d.drive_dirs:
                    if not os.path.isdir(dd):
                        continue
                    for fn in DRIVE_FILES:
                        p = os.path.join(dd, fn)
                        if os.path.lexists(p):
                            out.append(check_input(p, role="report"))
                    for fn in IC.listdir(dd):
                        if re.fullmatch(r"leg_\d+\.rc", fn):
                            out.append(check_input(os.path.join(dd, fn), role="report"))
                if not d.manifest_config or not os.path.isfile(d.manifest_config):
                    raise AdapterRefusal(str(d.manifest_config), f"the checkout carries no Petri manifest config for seed {d.seed_label}; the auditor / judge / serving "
                                                                 f"declarations the minted instrument needs are not guessed")
                out.append(check_input(d.manifest_config, role="report"))
                if d.fallback_summary and os.path.lexists(d.fallback_summary):
                    out.append(check_input(d.fallback_summary, role="report"))
            cj = self._json_of(cj_path)
            for e in (cj.get("canonical") or []):
                key = e.get("petri_run_key") if isinstance(e, Mapping) else None
                if not isinstance(key, str) or not key:
                    continue
                cdir = os.path.join(d.root, "cells", key)
                for fn in STORE_FILES_REQUIRED + STORE_FILES_OPTIONAL:
                    p = os.path.join(cdir, fn)
                    if os.path.lexists(p):
                        out.append(check_input(p, role=("rows" if fn == "rows.jsonl" else "report")))
        if any(d.binding == "premint" for d in present):
            if not os.path.lexists(self.waiver_path):
                raise AdapterRefusal(self.waiver_path, "the criterion-3 waiver record is absent; it is required provenance on every run of the seed-1 store (D118) "
                                                       "and the rows are not filed without it")
            out.append(check_input(self.waiver_path, role="report"))
        return out

    @staticmethod
    def _json_of(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise AdapterRefusal(path, f"not JSON: {exc.msg}") from exc
        return data if isinstance(data, Mapping) else {}

    def _store_of(self, uri: str) -> Optional[StoreDeclaration]:
        for d in self.stores:
            under = [d.root] + list(d.drive_dirs) + ([d.manifest_config] if d.manifest_config else [])
            if any(uri == r or uri.startswith(r.rstrip(os.sep) + os.sep) for r in under):
                return d
        # the judge-chain mirror is shared by the minted seeds: it rides the FIRST minted group (every minted group reads it through by_uri)
        minted = [d for d in self.stores if d.binding == "minted" and d.fallback_summary == uri]
        return minted[0] if minted else None

    def comparison_group(self, inp: Input) -> str:
        if inp.uri == self.waiver_path:
            prem = [d for d in self.stores if d.binding == "premint"]
            if prem:
                return prem[0].seed_label
        d = self._store_of(inp.uri)
        if d is None:
            raise AdapterRefusal(inp.uri, "not under a declared store root, drive ledger, manifest config, nor the waiver path")
        return d.seed_label

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        rows: List[SourceStateRow] = []
        first_by_group: Dict[str, Input] = {}
        for inp in inputs:
            try:
                g = self.comparison_group(inp)
            except AdapterRefusal:
                continue
            first_by_group.setdefault(g, inp)
            if os.path.basename(inp.uri) == "cells.json":
                sid = source_id_for(NAME, g)
                cj = self._json_of(inp.path)
                members = sorted(f"{e.get('cell_id')}|{e.get('petri_run_key')}" for e in (cj.get("canonical") or []) if isinstance(e, Mapping))
                rows.append(SourceStateRow(sid, inp.uri, "$.canonical[*]", 1, 1, members, inp.sha256))
                rows.append(SourceStateRow(sid, inp.uri, "$.absent", 1, 1, sorted(str(a) for a in (cj.get("absent") or [])), inp.sha256))
        for g, inp in sorted(first_by_group.items()):
            rows += self.refusals.source_state_rows(source_id_for(NAME, g), inp.uri, inp.sha256)
        return rows

    def status_extra(self, group: str) -> Dict[str, Any]:
        return dict(self._facts.get(group) or {})

    def status_sources(self, group: str) -> Tuple[str, ...]:
        return (SOURCE,)

    # ---- the connected registry -----------------------------------------------------------------
    @staticmethod
    def _rows(conn, sql: str, params: Sequence[object] = ()) -> List[Dict[str, Any]]:
        cur = conn.execute(sql, tuple(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _premint(self, ctx: AdapterContext, root: str, producer_key: Optional[str] = None) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
        """The premint-queued Petri rows under the producer key: ``{(campaign object spelling, eval): row}``, the instrument rows by eval
        (``{eval: instrument row}``) and the judge per eval; refuses when the connected registry holds none (nothing to fold onto)."""
        conn = ctx.conn
        pk = producer_key or self.premint_producer_key
        if conn is None:
            raise AdapterRefusal(root, "this reader folds onto the premint-queued rows and needs the registry connection to read their instrument and "
                                       "judge identities; none was given")
        rows = self._rows(conn, "SELECT run_id, eval_id, instrument_id, judge_id, object_id, producer_object_key_legacy, campaign_cell_id_legacy, "
                                "trial_index, seed_base, lifecycle_state FROM run WHERE campaign = ? AND producer_run_key_legacy = ? AND eval_id LIKE 'petri_%'",
                          (CAMPAIGN, pk))
        if not rows:
            raise AdapterRefusal(root, f"no premint-queued Petri rows under producer key {pk!r} in the connected registry; the reader "
                                       f"folds onto them and mints no identity of its own")
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        inst_of_eval: Dict[str, set] = {}
        judge_of_eval: Dict[str, set] = {}
        for r in rows:
            by_key[(str(r["campaign_cell_id_legacy"]), str(r["eval_id"]))] = r
            inst_of_eval.setdefault(str(r["eval_id"]), set()).add(str(r["instrument_id"]))
            judge_of_eval.setdefault(str(r["eval_id"]), set()).add(str(r["judge_id"]))
        insts: Dict[str, Dict[str, Any]] = {}
        judges: Dict[str, str] = {}
        for ev in sorted(inst_of_eval):
            if len(inst_of_eval[ev]) != 1 or len(judge_of_eval[ev]) != 1:
                raise AdapterRefusal(root, f"the premint-queued rows of {ev} name {len(inst_of_eval[ev])} instrument(s) and {len(judge_of_eval[ev])} judge(s); "
                                           f"one of each is the battery's declaration")
            iid = next(iter(inst_of_eval[ev]))
            irows = self._rows(conn, "SELECT * FROM instrument WHERE instrument_id = ?", (iid,))
            if not irows:
                raise AdapterRefusal(root, f"instrument {iid} of the premint-queued {ev} rows is not in the connected registry (the premint's own record)")
            retired_by = self._retiring_ruling(conn, iid)
            if retired_by:
                raise AdapterRefusal(root, f"the premint-queued {ev} rows under producer key {pk!r} bind instrument {iid[:12]}…, which ruling "
                                           f"{retired_by} retired (an exclude ruling naming it in target_json.superseded_by): a completed store is not folded onto "
                                           f"excluded identities — fold onto the successor premint's producer key instead")
            insts[ev] = irows[0]
            judges[ev] = next(iter(judge_of_eval[ev]))
            if str(irows[0].get("judge_id") or "") != judges[ev]:
                raise AdapterRefusal(root, f"the premint-queued {ev} rows name judge {judges[ev][:12]}… but their instrument {iid[:12]}… names judge "
                                           f"{str(irows[0].get('judge_id') or '')[:12]}…; the completion predicate reads the instrument's judge, so the plan rows are not one plan")
            if not self._rows(conn, "SELECT 1 AS ok FROM judge WHERE judge_id = ?", (judges[ev],)):
                raise AdapterRefusal(root, f"judge {judges[ev]} of the premint-queued {ev} rows is not in the connected registry")
        return by_key, insts, judges

    def _pooled_instrument_from_premint(self, batch: RecordBatch, ctx: AdapterContext, decl: StoreDeclaration, insts: Dict[str, Dict[str, Any]],
                                        judges: Dict[str, str], facts: Dict[str, Any]) -> None:
        """The premint binding's pooled instrument (v3). When the premint key set queues a ``petri_pooled`` row its instrument is already in
        ``insts`` and the reader binds to it exactly as it binds the six. Otherwise the six premint instruments — which must agree on every
        identity field but ``eval_id`` (one declaration, six evals; anything else refuses the group: the pooled instrument is not guessed) — mint
        the pooled one: the same row with ``eval_id = petri_pooled``, through ``identity.instrument_id``, written as an ``instrument`` record with
        ``extra.premint_family`` = the producer key and the six sibling ids beside it."""
        producer, now = ctx.producer, ctx.now
        if POOLED_EVAL in insts and POOLED_EVAL in judges:
            facts["premint"]["pooled_row"] = "present"
            facts["instruments"][POOLED_EVAL] = {"instrument_id": str(insts[POOLED_EVAL]["instrument_id"]), "source": "premint_row"}
            return
        beh_evals = [self.seeds.eval_for_spelling(b) or b for b in BEHAVIOURS]
        missing = [ev for ev in beh_evals if ev not in insts]
        if missing:
            raise AdapterRefusal(decl.root, f"the premint-queued rows under {decl.producer_key!r} name no instrument for {missing}; the pooled instrument is derived from "
                                            f"all six behaviour instruments and is not guessed")
        rows = [insts[ev] for ev in beh_evals]
        fields = [self._identity_fields(r) for r in rows]
        agree = {name for name, _t in I.INSTRUMENT_ID_FIELDS if name != "eval_id" and len({I.canonical_json(f.get(name)) for f in fields}) == 1}
        differ = sorted({name for name, _t in I.INSTRUMENT_ID_FIELDS if name != "eval_id"} - agree)
        if differ:
            raise AdapterRefusal(decl.root, f"the premint's six behaviour instruments under {decl.producer_key!r} differ on {differ} beyond eval_id; one declaration "
                                            f"across six evals is the premint's shape, so the pooled instrument is not guessed")
        if len(set(judges[ev] for ev in beh_evals)) != 1:
            raise AdapterRefusal(decl.root, f"the premint's six behaviour instruments name {len(set(judges[ev] for ev in beh_evals))} judges; one is the premint's shape")
        payload: Dict[str, Any] = dict(fields[0], eval_id=POOLED_EVAL)
        payload["instrument_id"] = I.instrument_id(payload)
        base_id = str(rows[0]["instrument_id"])
        payload["notes"] = (f"Derived pooled instrument of the seed-{decl.seed_index} Petri audit ({decl.audit}) on premint family {decl.producer_key}: the premint's "
                            f"behaviour instrument {base_id[:12]}… with eval_id {POOLED_EVAL} (the six premint instruments agree on every other identity field); "
                            f"minted by the {NAME} reader through identity.instrument_id because the premint key set queues no {POOLED_EVAL} row")
        payload["extra"] = {"derived_from_instrument_id": base_id, "premint_sibling_instrument_ids": sorted(str(r["instrument_id"]) for r in rows),
                            "premint_family": decl.producer_key, "derived_from_seed_instrument": PETRI_INSTRUMENT_SEED, "model_id": MODEL_ID,
                            "judge_seed_id": JUDGE_SEED, "audit": decl.audit, "changed_fields": {"eval_id": POOLED_EVAL}, "minted_by": NAME,
                            "emission_version": self.EMISSION_VERSION}
        payload["field_confidence_json"] = {"prompt_template_sha256": LC.unverified("the served path renders the chat template server-side; no prompt-template "
                                                                                     "hash is recorded for this harness (the premint's own note)")}
        batch.add("instrument", payload, producer, now, observe=False)
        insts[POOLED_EVAL] = payload
        judges[POOLED_EVAL] = judges[beh_evals[0]]
        facts["premint"]["pooled_row"] = "absent"
        facts["instruments"][POOLED_EVAL] = {"instrument_id": payload["instrument_id"], "derived_from": base_id, "source": "minted_from_premint", "premint_family": decl.producer_key}

    @staticmethod
    def _identity_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
        """A registry instrument row's identity fields with the JSON columns parsed (the connected registry stores them serialised)."""
        out: Dict[str, Any] = {}
        for name, ctype in I.INSTRUMENT_ID_FIELDS:
            v = row.get(name)
            if ctype == "JSON" and isinstance(v, str):
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            out[name] = v
        return out

    def _retiring_ruling(self, conn, instrument_id: str) -> Optional[str]:
        """The id of an ``exclude`` ruling whose ``target_json.superseded_by`` names the instrument (the premint-retirement rulings' shape:
        ``{old instrument id: successor instrument id}``), else None."""
        for r in self._rows(conn, "SELECT ruling_id, target_json FROM ruling WHERE action = 'exclude'"):
            tj = r.get("target_json")
            try:
                tj = json.loads(tj) if isinstance(tj, str) else tj
            except ValueError:
                continue
            sup = tj.get("superseded_by") if isinstance(tj, Mapping) else None
            if isinstance(sup, Mapping) and instrument_id in sup:
                return str(r["ruling_id"])
        return None

    def _object_row(self, ctx: AdapterContext, object_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows(ctx.conn, "SELECT object_id, object_kind, paper_label, object_role FROM object WHERE object_id = ?", (object_id,))
        return rows[0] if rows else None

    # ---- the checkout's roster / pins / seeds -----------------------------------------------------
    def _roster_pins(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import yaml
        for p, what in ((self.roster_path, "roster"), (self.pins_path, "pins")):
            if not os.path.isfile(p):
                raise AdapterRefusal(p, f"the {what} is not in this checkout; no staged object id can be named")
        with open(self.roster_path, "r", encoding="utf-8") as fh:
            roster = yaml.safe_load(fh) or {}
        with open(self.pins_path, "r", encoding="utf-8") as fh:
            pins = json.load(fh)
        if not isinstance(roster, Mapping) or not isinstance(pins, Mapping):
            raise AdapterRefusal(self.roster_path, "the roster / pins are not mappings")
        return dict(roster), dict(pins)

    def resolve_object(self, campaign_key_legacy: str, roster_selection: str = ROSTER_SELECTION) -> Tuple[str, str]:
        """``(object_id, producer object key)`` of a campaign spelling through the harness's own resolution (jobplan.object_id_for over the roster's
        s1 selection and the pins; ``none``; ``prompted__<si>`` -> ``si:<sha>``); the producer object key is the roster's staged id, else the spelling."""
        roster, pins = self._roster_pins()
        si_sha = {k: str(v.get("si_text_sha256")) for k, v in (self.seeds.si_texts or {}).items() if isinstance(v, Mapping) and v.get("si_text_sha256")}
        oid = JP.object_id_for(campaign_key_legacy, model_key=MODEL_KEY, roster_selection=roster_selection, roster=roster, pins=pins, si_sha=si_sha)
        staged = None
        try:
            staged = roster["targets"][MODEL_KEY]["views"][roster_selection]["cells"].get(campaign_key_legacy)
        except (KeyError, TypeError):
            staged = None
        return oid, (str(staged) if isinstance(staged, str) and staged else campaign_key_legacy)

    # ---- read ------------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        self.refusals = LC.RowRefusals(NAME)
        decls = [d for d in self.stores if d.seed_label == group]
        if len(decls) != 1:
            raise AdapterRefusal(group, f"{len(decls)} declared stores carry seed label {group!r}; exactly one is read per group")
        decl = decls[0]
        batch = RecordBatch(source_id_for(NAME, group), group)
        facts: Dict[str, Any] = {"store_root": LC.rel_home(decl.root), "binding": decl.binding, "campaign": decl.campaign, "producer_key": decl.producer_key,
                                 "pin": dict(decl.store_pin or {}), "objects": {}, "per_behaviour": {}, "judged_runs": 0, "pooled_runs": 0, "pooled": {}, "items": 0,
                                 "metrics": 0, "contrasts": 0, "pairing": {}, "controls_skipped": [],
                                 "premint": {"producer_key": decl.producer_key, "queued_rows_seen": 0, "matched": 0, "runs_without_premint": [], "pooled_row": None},
                                 "admission": {}, "pending": {}, "enumerated_not_run": [], "instruments": {}, "ledger": {}, "refusals": [], "n_refusals": 0}
        self._facts[group] = facts
        rng = np.random.default_rng(BOOT_SEED)
        comp_id = self._computation(batch, ctx)
        if decl.binding == "premint":
            self._read_premint(ctx, decl, inputs, batch, facts, rng, comp_id)
        else:
            self._read_minted(ctx, decl, inputs, batch, facts, rng, comp_id)
        facts["refusals"] = [f"{it['key']} -- {it['why']}" for it in self.refusals.items]
        facts["n_refusals"] = len(self.refusals)
        yield batch

    def _computation(self, batch: RecordBatch, ctx: AdapterContext) -> str:
        comp_id = I.computation_id(f"sisteer.registry.adapters.{NAME}", f"emission_version={self.EMISSION_VERSION}",
                                   f"numpy default_rng({BOOT_SEED}), {N_BOOT} draws, one generator per group in object / behaviour order")
        batch.add("computation", {"computation_id": comp_id, "algorithm": f"sisteer.registry.adapters.{NAME}", "version": f"emission_version={self.EMISSION_VERSION}",
                                  "seed_provenance": f"numpy default_rng({BOOT_SEED}), {N_BOOT} draws, one generator per group in object / behaviour order",
                                  "code_sha": "unverified", "notes": ("the reader's statistics: the mean over scored scenarios (checked equal to the producer's "
                                                                       "results.json mean), a percentile bootstrap over scored scenarios, and the paired-on-scenario "
                                                                       "contrast against the store's own `none` (mean difference, percentile bootstrap, exact two-sided "
                                                                       "sign test, d_z); the pooled run's mean of the six behaviour means with a stratified percentile "
                                                                       "bootstrap (scenarios resampled within each behaviour) and its contrast paired on (behaviour, "
                                                                       "scenario); code_sha unverified: the registry's own code at an unrecorded tree, "
                                                                       "the emission version is the reproducibility handle")}, ctx.producer, ctx.now)
        return comp_id

    def _declared_objects(self, cj: Mapping[str, Any], cells_inp: Input, seed_label: str, facts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """The store's own declaration of its objects, checked against itself: the listed (canonical) entries keyed by the store spelling; an
        `absent` entry and a disagreement between expected_cells and canonical refuse THAT object; the controls are recorded, never runs."""
        canonical = [e for e in (cj.get("canonical") or []) if isinstance(e, Mapping) and e.get("petri_run_key")]
        listed = {str(e["cell_id"]): e for e in canonical}
        expected = [str(x) for x in (cj.get("expected_cells") or [])]
        for a in (str(x) for x in (cj.get("absent") or [])):
            self.refusals.add(f"{seed_label}|{a}", "cells.json lists this object under `absent`: no audit to read", cells_inp.uri)
        for x in sorted(set(expected) ^ set(listed)):
            self.refusals.add(f"{seed_label}|{x}", f"cells.json disagrees with itself: expected_cells {sorted(expected)} vs canonical {sorted(listed)}", cells_inp.uri)
        facts["controls_skipped"] = sorted(str(e.get("cell_id")) for e in (cj.get("controls") or []) if isinstance(e, Mapping))
        return listed

    def _read_premint(self, ctx: AdapterContext, decl: StoreDeclaration, inputs: Sequence[Input], batch: RecordBatch, facts: Dict[str, Any], rng, comp_id: str) -> None:
        """The seed-1 store on its premint: the instrument and judge come from the queued rows, the fold advances them."""
        producer, now = ctx.producer, ctx.now
        by_uri: Dict[str, Input] = {i.uri: i for i in inputs}
        root = decl.root
        pin = dict(decl.store_pin or self.store_pin)
        if os.path.join(root, "cells.json") not in by_uri or os.path.join(root, "modal_drive.json") not in by_uri:
            raise AdapterRefusal(root, "the store's cells.json / modal_drive.json are not among the group's inputs")
        cells_inp, drive_inp = by_uri[os.path.join(root, "cells.json")], by_uri[os.path.join(root, "modal_drive.json")]
        cj = self._json_of(cells_inp.path)
        drive = self._json_of(drive_inp.path)
        waiver_path = decl.waiver_path or self.waiver_path
        waiver_inp = by_uri.get(waiver_path)
        if waiver_inp is None:
            raise AdapterRefusal(waiver_path, "the criterion-3 waiver record is not among the inputs")
        admission = self._admission(waiver_inp, drive)
        facts["admission"] = admission
        # -- the store's own declaration of its objects
        audit = str(cj.get("audit") or "")
        if audit != pin["audit"]:
            raise AdapterRefusal(cells_inp.uri, f"cells.json names audit {audit!r}, the pin {pin['audit']!r}")
        listed = self._declared_objects(cj, cells_inp, decl.seed_label, facts)
        verdict = (drive.get("verdict") or {}) if isinstance(drive.get("verdict"), Mapping) else {}
        counts = verdict.get("counts") if isinstance(verdict.get("counts"), Mapping) else {}
        schedule = {str(s.get("cell_id")): s for s in (drive.get("schedule") or []) if isinstance(s, Mapping)}
        verdict_state = {str(d[1]): str(d[2]) for d in (verdict.get("detail") or []) if isinstance(d, (list, tuple)) and len(d) >= 3}
        facts["drive"] = {"drive_id": drive.get("drive_id"), "W": drive.get("W"), "attempt_id": drive.get("attempt_id"), "verdict_counts": dict(counts),
                          "started_at": drive.get("started_at"), "finished_at": drive.get("finished_at"), "substrate": drive.get("substrate"),
                          "auditor": (dict(drive.get("auditor")) if isinstance(drive.get("auditor"), Mapping) else None)}
        # -- the premint identities and the decode policy
        queued, insts, judges = self._premint(ctx, root, decl.producer_key)
        facts["premint"]["queued_rows_seen"] = len(queued)
        # the pooled instrument: the premint's own when the key set queues a `petri_pooled` row, else minted from the premint's six (v3)
        self._pooled_instrument_from_premint(batch, ctx, decl, insts, judges, facts)
        pf = DP.load(self.policy_path)
        launches = {str(l.get("record_name")): l for l in (drive.get("launches") or []) if isinstance(l, Mapping)}
        queued_utc = LC.iso_ms(drive.get("started_at"), now)
        drive_w = str(drive.get("W") or "")
        bd: Dict[str, Any] = {
            "campaign": decl.campaign, "producer_key": decl.producer_key, "seed_label": decl.seed_label, "seed_index": decl.seed_index, "binding": "premint",
            "premint_family": decl.producer_key,
            "roster_selection": decl.roster_selection, "queued": queued, "admission": admission, "drive": facts["drive"],
            "code_sha": drive_w if _is_sha1(drive_w) else "unverified", "substrate": str(drive.get("substrate") or "modal"), "cluster_job_id": None,
            "trial_of": (lambda oid: 0),
            "provenance_extra": {"modal_attempt": {"drive_id": drive.get("drive_id"), "attempt_id": drive.get("attempt_id"), "W": drive.get("W")}},
            "auditor_pool_legacy": ((drive.get("auditor") or {}).get("pool") if isinstance(drive.get("auditor"), Mapping) else None),
            "api_pool": None, "api_pools": None, "judge_model": "anthropic/claude-opus-5",
            "instrument_source": f"the premint-queued rows under producer key {decl.producer_key} (one instrument per eval)",
            "judge_source": "the premint-queued rows' judge (the instrument's judge; petri_judge_opus5)",
            "code_sha_source": (f"modal_drive.json W == proxy_sha (the tree that drove the audit); the premint filed the carrying tree {CARRIED_UNDER_TREE[:12]} "
                                f"(W24; extra.admission.carried_under_tree)"),
            "substrate_source_text": "modal_drive.json substrate",
            "queued_source": "modal_drive.json started_at (the premint's own queued event stands on the live row)",
            "started_source": "modal_drive.json launches[].t of this object",
            "admission_fc": {"confidence": "recorded", "reason": (f"criterion 3 (the determinism spot-check) WAIVED under {WAIVER_DECISION_REF} ({admission['waiver_ref']}); "
                                                                  f"the compensating W23 re-audit is {W23_REAUDIT_STATE}: {W23_REAUDIT_NOTE}")},
            "notes_tail": f"criterion 3 waived ({WAIVER_DECISION_REF}), W23 re-audit {W23_REAUDIT_STATE}",
        }
        # -- every listed object, `none` first (the contrasts pair against it)
        readings: Dict[str, Dict[str, Any]] = {}
        order = sorted(listed, key=lambda k: (0 if k == "none" else 1, k))
        for store_key_legacy in order:
            entry = listed[store_key_legacy]
            if store_key_legacy.startswith(CONTROL_PREFIX):
                continue   # the harness's derived no-op control legs are never runs (legacy_spellings.yaml; petri_stats does the same)
            key = str(entry["petri_run_key"])
            rkey = f"{decl.seed_label}|{store_key_legacy}"
            campaign_key_legacy = campaign_object_key_legacy(store_key_legacy)
            if campaign_key_legacy is None:
                self.refusals.add(rkey, f"the store spelling {store_key_legacy!r} has no campaign spelling (petri_stats.PCELL_REV); the object is not guessed", cells_inp.uri)
                continue
            cdir = os.path.join(root, "cells", key)
            files = {fn: by_uri.get(os.path.join(cdir, fn)) for fn in STORE_FILES_REQUIRED + STORE_FILES_OPTIONAL}
            missing = [fn for fn in STORE_FILES_REQUIRED if files[fn] is None]
            if missing:
                self.refusals.add(rkey, f"store files missing under cells/{key}: {', '.join(missing)}", cdir)
                continue
            sched = schedule.get(store_key_legacy) or {}
            if str(sched.get("run_key") or key) != key:
                self.refusals.add(rkey, f"modal_drive.json schedules run key {sched.get('run_key')!r} for this object, cells.json {key!r}", drive_inp.uri)
                continue
            if verdict_state.get(key) not in (None, "complete"):
                self.refusals.add(rkey, f"the drive's verdict names this audit {verdict_state.get(key)!r}, not complete", drive_inp.uri)
                continue
            rec = self._read_object(root, entry, key, campaign_key_legacy, files, rkey, audit=decl.audit, seed_index=decl.seed_index)
            if rec is None:
                continue
            if not self._bind_object(ctx, decl, rec, entry, campaign_key_legacy, store_key_legacy, rkey, cdir, pf, files):
                continue
            launch = launches.get(f"leg_{sched.get('cell_index')}_{decl.seed_label}") if sched.get("cell_index") is not None else None
            rec["started_utc"] = epoch_utc((launch or {}).get("t")) or queued_utc
            rec["queued_utc"] = queued_utc
            rec["judge_chain"] = {"state": "not_applicable"}
            self._emit_object(batch, ctx, rec, bd, insts, judges, comp_id, rng, facts, files, cdir)
            readings[store_key_legacy] = rec
        self._contrasts(batch, ctx, readings, comp_id, rng, facts, decl.seed_label, cells_inp)

    def _bind_object(self, ctx: AdapterContext, decl: StoreDeclaration, rec: Dict[str, Any], entry: Mapping[str, Any], campaign_key_legacy: str, store_key_legacy: str,
                     rkey: str, cdir: str, pf, files: Mapping[str, Optional[Input]]) -> bool:
        """The object through the harness's own resolution (the roster's selection for the seed, the pins, si_texts), its row in the connected
        registry, the store's kind agreeing with it, and the decode the store sent equal to the registered policy row the manifest names.
        False with the refusal filed when any of it fails."""
        try:
            object_id, object_key = self.resolve_object(campaign_key_legacy, decl.roster_selection)
        except (LookupError, AdapterRefusal) as exc:
            self.refusals.add(rkey, f"object unresolvable through the roster's {decl.roster_selection} selection / pins / si_texts: {exc}", self.roster_path)
            return False
        orow = self._object_row(ctx, object_id)
        if orow is None:
            self.refusals.add(rkey, f"object row {object_id} is not in the connected registry (the objects reader mints the premints); the run is not filed against an absent object", cdir)
            return False
        kind_store = str(entry.get("kind") or "")
        if (kind_store == "none") != (object_id == "none") or (kind_store == "prompted") != object_id.startswith("si:"):
            self.refusals.add(rkey, f"the store's object kind {kind_store!r} does not agree with the resolved object {object_id[:16]}…", cdir)
            return False
        rec.update(object_id=object_id, object_key=object_key, object_row=orow, campaign_key_legacy=campaign_key_legacy, store_key_legacy=store_key_legacy, entry=dict(entry))
        policy_id = str(((rec["manifest"].get("target_decode") or {}).get("policy_id")) or "")
        prow = pf.by_id.get(policy_id)
        if prow is None:
            self.refusals.add(rkey, f"the manifest names decode policy {policy_id!r}, which the registered decode_policy export does not carry", files["manifest.json"].uri)
            return False
        sent = {k: rec["request_params"].get(k) for k in DECODE_KEYS}
        stated = {k: prow.get(k) for k in DECODE_KEYS}
        if any(sent[k] is None or float(sent[k]) != float(stated[k]) for k in DECODE_KEYS if stated.get(k) is not None):
            self.refusals.add(rkey, f"the decode the store sent {sent} differs from the registered policy row {policy_id} {stated}", files["manifest.json"].uri)
            return False
        rec["policy_id"] = policy_id
        return True

    def _contrasts(self, batch: RecordBatch, ctx: AdapterContext, readings: Mapping[str, Dict[str, Any]], comp_id: str, rng, facts: Dict[str, Any], seed_label: str,
                   cells_inp: Input) -> None:
        """Every steered object against the store's own `none`, paired on scenario id per behaviour; an anchor not (yet) filed leaves the contrasts
        unformed by name (they form when the anchor's audit consolidates and the group re-reads)."""
        anchor = readings.get("none")
        for store_key_legacy, rec in sorted(readings.items()):
            if store_key_legacy == "none" or not rec.get("runs"):
                continue
            if anchor is None or not anchor.get("runs"):
                self.refusals.add(f"{seed_label}|{store_key_legacy}|contrast", "the store's `none` audit was not filed; no paired contrast is formed", cells_inp.uri)
                continue
            self._emit_contrasts(batch, ctx, rec, anchor, comp_id, rng, facts, seed_label)


    # ---- the minted binding: the producer's seed-2 / seed-3 stores --------------------------------
    def _read_minted(self, ctx: AdapterContext, decl: StoreDeclaration, inputs: Sequence[Input], batch: RecordBatch, facts: Dict[str, Any], rng, comp_id: str) -> None:
        """A cluster-run store: the instrument is MINTED from the seeded Petri row plus the store's declared changes (the checkout's manifest config, the
        manifest's decode, the drive ledger's tree, the standup's attested window), the runs under the audit's name; a complete audit files, an
        audit withheld by context-overflow casualties files a pending disposition per behaviour eval, an enumerated-only audit files nothing."""
        producer, now = ctx.producer, ctx.now
        by_uri: Dict[str, Input] = {i.uri: i for i in inputs}
        root = decl.root
        cells_inp = by_uri.get(os.path.join(root, "cells.json"))
        if cells_inp is None:
            raise AdapterRefusal(root, "the store's cells.json is not among the group's inputs")
        cfg_inp = by_uri.get(str(decl.manifest_config))
        if cfg_inp is None:
            raise AdapterRefusal(str(decl.manifest_config), "the checkout's Petri manifest config is not among the group's inputs")
        cj = self._json_of(cells_inp.path)
        if str(cj.get("audit") or "") != decl.audit:
            raise AdapterRefusal(cells_inp.uri, f"cells.json names audit {cj.get('audit')!r}, the declaration {decl.audit!r}")
        listed = self._declared_objects(cj, cells_inp, decl.seed_label, facts)
        cfg = self._config(cfg_inp, decl)
        ledger = self._ledger(decl, by_uri)
        facts["ledger"] = {"drive_dirs": [LC.rel_home(d) for d in decl.drive_dirs], "drives": {k: {kk: vv for kk, vv in v.items() if kk != "dir"} for k, v in ledger["drives"].items()},
                           "completed_results": len(ledger["rc"]), "trees": sorted(ledger["trees"]), "windows": sorted(ledger["windows"]), "rerun_windows": sorted(ledger["rerun_windows"]),
                           "thinking_modes": sorted(ledger["thinking"]),
                           "engine_kinds": sorted(ledger["engine_kinds"]), "proxy_jobs": ledger["proxy_jobs"], "escalation": ledger["escalation"]}
        facts["config"] = {k: cfg.get(k) for k in ("judge_model", "judge_fallback_models", "auditor_model", "auditor_fallback_models", "scenario_bundle_hash", "serving_impl_commit",
                                                   "serving_packages", "target_model", "target_revision", "policy_id", "bloom_options", "prompt_provenance")}
        pf = DP.load(self.policy_path)
        admission = {"criterion3_determinism": "not_applicable", "premint_row": "absent", "carried": False,
                     "note": (f"PetriWave's Slurm audit of seed {decl.seed_index} under tree {ledger['tree'] or 'unrecorded'} (no carry, no waiver, no premint: the reader "
                              f"mints the instrument from the seeded {PETRI_INSTRUMENT_SEED} row and the store's declared changes)")}
        facts["admission"] = admission
        # the instrument is minted PER COMPLETING TREE (its harness_version names the tree): an audit completed on a per-object fix tree
        # (s2 control_relu, 3dfdddd6…) files on its own instrument beside the seed's others, never refused and never folded onto theirs
        insts_by_tree: Dict[str, Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]] = {}
        readings: Dict[str, Dict[str, Any]] = {}
        fallback = self._fallback_summary(decl, by_uri, facts)
        order = sorted(listed, key=lambda k: (0 if k == "none" else 1, k))
        for store_key_legacy in order:
            entry = listed[store_key_legacy]
            if store_key_legacy.startswith(CONTROL_PREFIX):
                continue
            key = str(entry["petri_run_key"])
            rkey = f"{decl.seed_label}|{store_key_legacy}"
            campaign_key_legacy = campaign_object_key_legacy(store_key_legacy)
            if campaign_key_legacy is None:
                self.refusals.add(rkey, f"the store spelling {store_key_legacy!r} has no campaign spelling (petri_stats.PCELL_REV); the object is not guessed", cells_inp.uri)
                continue
            cdir = os.path.join(root, "cells", key)
            files = {fn: by_uri.get(os.path.join(cdir, fn)) for fn in STORE_FILES_REQUIRED + STORE_FILES_OPTIONAL}
            if files["manifest.json"] is None:
                self.refusals.add(rkey, f"store files missing under cells/{key}: manifest.json", cdir)
                continue
            if files["rows.jsonl"] is None and files["results.json"] is None:
                facts["enumerated_not_run"].append(store_key_legacy)      # enumerated by the standup, not yet audited: nothing to file, nothing to refuse
                continue
            inc = self._json_of(files["petri_incomplete.json"].path) if files["petri_incomplete.json"] is not None else {}
            unresolved = [u for u in (inc.get("unresolved") or []) if isinstance(u, Mapping)] if isinstance(inc, Mapping) else []
            if files["results.json"] is None:
                # an audit the runner left INCOMPLETE: withheld scenarios, no results.json. Context-overflow casualties await the widened-window
                # re-run pass and file as PENDING by name; any other withholding is refused by name (never read as a refusal to score)
                other = [u for u in unresolved if not str(u.get("reason") or "").startswith(OVERFLOW_REASON_PREFIX)]
                if not unresolved:
                    self.refusals.add(rkey, "rows.jsonl present, no results.json and no withheld item in petri_incomplete.json: an audit in flight or torn; not read", cdir)
                    continue
                if other:
                    self.refusals.add(rkey, f"{len(other)} withheld item(s) whose reason is not a context overflow ({sorted({str(u.get('reason'))[:60] for u in other})}); "
                                            f"the audit is incomplete for an unclassified cause and is not filed as pending", cdir)
                    continue
                self._emit_pending(batch, ctx, decl, store_key_legacy, campaign_key_legacy, key, unresolved, files["petri_incomplete.json"], facts)
                continue
            if files["petri_incomplete.json"] is None:
                self.refusals.add(rkey, f"store files missing under cells/{key}: petri_incomplete.json (the withheld-item ledger must be present and empty for a complete audit)", cdir)
                continue
            rec = self._read_object(root, entry, key, campaign_key_legacy, files, rkey, audit=decl.audit, seed_index=decl.seed_index)
            if rec is None:
                continue
            join = ledger["rc"].get(files["results.json"].sha256)
            if join is None:
                self.refusals.add(rkey, f"no leg_*.rc line under {', '.join(LC.rel_home(d) for d in decl.drive_dirs)} names this results.json ({files['results.json'].sha256[:12]}…); "
                                        f"the audit is not joined to the drive that completed it", cdir)
                continue
            drive = ledger["drives"].get(join["drive"])
            if drive is None or int(join["rc"]) != 0:
                self.refusals.add(rkey, f"the rc line names drive {join['drive']!r} (rc {join['rc']}), which drive.log does not start or which did not complete", join["file"])
                continue
            tree = str(drive["sha"])
            if tree not in insts_by_tree:
                insts_by_tree[tree] = self._mint_instruments(batch, ctx, decl, cfg, dict(ledger, tree=tree), rec, facts)
            insts, judges = insts_by_tree[tree]
            if not self._bind_object(ctx, decl, rec, entry, campaign_key_legacy, store_key_legacy, rkey, cdir, pf, files):
                continue
            rec["tree"] = tree
            rec["judge_chain"] = self._judge_chain_of(fallback, decl.seed_label, rec["object_row"], store_key_legacy, rec, facts)
            pools = self._pools_of(ledger, drive)
            rec["queued_utc"] = LC.iso_ms(min(d["utc"] for d in ledger["drives"].values()), now)
            rec["started_utc"] = LC.iso_ms(drive["utc"], now)
            rerun: Optional[Dict[str, Any]] = None
            if drive.get("rerun"):
                # the audit was CONSOLIDATED by the widened-window re-run pass: the scenarios whose rows the re-run drive filed (created after its
                # start) ran at the re-run window; everything else at the main pass's. Recorded per run, never a second instrument
                t_rerun = LC.iso_ms(drive["utc"], now)
                redone = {b: sorted(sc for sc, cu in pb["created"].items() if cu >= t_rerun) for b, pb in rec["per_beh"].items()}
                rerun = {"drive_id": drive["id"], "window_tokens": ledger.get("rerun_window"), "main_window_tokens": ledger.get("window"),
                         "scenarios": {b: v for b, v in redone.items() if v}, "n_scenarios": sum(len(v) for v in redone.values()),
                         "rule": "the context-overflow casualties of the main pass re-run at the widened window (the instrument's own ladder_promotion_rule); "
                                 "identified as the scenarios whose score rows the re-run drive filed"}
                facts.setdefault("overflow_reruns", {})[store_key_legacy] = rerun
            bd: Dict[str, Any] = {
                "campaign": decl.campaign, "producer_key": decl.producer_key, "seed_label": decl.seed_label, "seed_index": decl.seed_index, "binding": "minted",
                "roster_selection": decl.roster_selection, "queued": {}, "admission": admission, "drive": None,
                "code_sha": drive["sha"], "substrate": drive["cluster"], "cluster_job_id": ledger["proxy_jobs"].get(os.path.basename(drive["dir"])),
                "trial_of": (lambda oid: 0),
                "provenance_extra": {"drive_ledger": {"drive_id": drive["id"], "drive_dir": LC.rel_home(drive["dir"]), "rc_file": LC.rel_home(join["file"]), "rc": join["rc"],
                                                      "tree_sha": drive["sha"], "cluster": drive["cluster"], "auditor_org": drive["org"], "started_utc": drive["utc"],
                                                      "results_sha256": files["results.json"].sha256, "window_tokens": ledger["window"], "completed_by_rerun": bool(drive.get("rerun")),
                                                      "trees_of_seed": sorted(ledger["trees"]),
                                                      # a federated drive: the auth, the digest it attested and the org that digest PROVES (None: no
                                                      # org's current ids file digests to it -- the pool then reads wif-unverified:<digest>, never the label)
                                                      **({"auditor_auth": "wif", "auditor_auth_org": drive.get("auth_org"), "auditor_ids_sha12": drive.get("ids_sha12"),
                                                          "auditor_org_verified": drive.get("org_verified")} if drive.get("auth") == "wif" else {})},
                                     **({"overflow_rerun": rerun} if rerun else {})},
                "auditor_pool_legacy": drive["org"], "api_pool": pools["api_pool"], "api_pools": pools["api_pools"],
                "judge_model": cfg.get("judge_model"), "judge_fallback_models": cfg.get("judge_fallback_models"),
                "instrument_source": (f"minted by this reader from seed instrument {PETRI_INSTRUMENT_SEED} with the store's declared changes (extra.derived_from_instrument_id, "
                                      f"extra.changed_fields on the instrument row)"),
                "judge_source": f"seed judge {JUDGE_SEED} (the manifest config's judge_model resolves to it; the fallback chain rides the judgment's extra)",
                "code_sha_source": "drive.log DRIVE_START sha of the drive whose leg_*.rc names this results.json (the byte-level join)",
                "substrate_source_text": "drive.log DRIVE_START cluster of the completing drive",
                "queued_source": "the first DRIVE_START of the completing drive's pass at the same tree",
                "started_source": "drive.log DRIVE_START utc of the completing drive (the resumed audit's last launch)",
                "admission_fc": {"confidence": "recorded", "reason": admission["note"]},
                "notes_tail": (f"PetriWave Slurm audit, tree {drive['sha'][:12]}, auditor / judge org {pools['api_pool'] or 'mixed (' + ', '.join(sorted(pools['api_pools'])) + ')'}"
                               + (f"; {rerun['n_scenarios']} scenario(s) consolidated by the {ledger.get('rerun_window')}-token re-run pass" if rerun else "")),
            }
            self._emit_object(batch, ctx, rec, bd, insts, judges, comp_id, rng, facts, files, cdir)
            readings[store_key_legacy] = rec
        self._contrasts(batch, ctx, readings, comp_id, rng, facts, decl.seed_label, cells_inp)

    def _config(self, cfg_inp: Input, decl: StoreDeclaration) -> Dict[str, Any]:
        """The checkout's Petri manifest config for the seed (`petri:` block): the audit name must be the store's; the auditor / judge and their
        fallback chains, the bundle, the serving packages and the target are the declarations the minted instrument carries."""
        import yaml
        with open(cfg_inp.path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        pc = data.get("petri") if isinstance(data, Mapping) and isinstance(data.get("petri"), Mapping) else None
        if pc is None:
            raise AdapterRefusal(cfg_inp.uri, "the manifest config carries no `petri:` block")
        if str(pc.get("audit") or "") != decl.audit:
            raise AdapterRefusal(cfg_inp.uri, f"the manifest config names audit {pc.get('audit')!r}, the declared store {decl.audit!r}")
        td = pc.get("target_decode") if isinstance(pc.get("target_decode"), Mapping) else {}
        out = {"judge_model": pc.get("judge_model"), "judge_fallback_models": list(pc.get("judge_fallback_models") or []), "auditor_model": pc.get("auditor_model"),
               "auditor_fallback_models": list(pc.get("auditor_fallback_models") or []), "scenario_bundle_hash": pc.get("scenario_bundle_hash"),
               "serving_impl_commit": pc.get("serving_impl_commit"), "serving_packages": dict(pc.get("serving_packages") or {}), "target_model": pc.get("target_model"),
               "target_revision": pc.get("target_revision"), "policy_id": td.get("policy"), "bloom_options": dict(pc.get("bloom_options") or {}),
               "prompt_provenance": pc.get("prompt_provenance"), "config_uri": LC.rel_home(cfg_inp.uri), "config_sha256": cfg_inp.sha256}
        for k in ("judge_model", "auditor_model", "scenario_bundle_hash", "policy_id", "target_model"):
            if not out.get(k):
                raise AdapterRefusal(cfg_inp.uri, f"the manifest config declares no {k}")
        return out

    def _ledger(self, decl: StoreDeclaration, by_uri: Mapping[str, Input]) -> Dict[str, Any]:
        """The producer's drive ledger over the declared standup dirs: every DRIVE_START (drive id -> utc, tree sha, cluster, auditor org, dir), every
        completed results.json (leg_*.rc: results sha -> rc, drive, file), the attested window and thinking mode, the producer's proxy jobs and
        escalation lines when its state dir is on this host. The group's tree is the ONE sha every completing drive ran; two refuse."""
        drives: Dict[str, Dict[str, Any]] = {}
        rc: Dict[str, Dict[str, Any]] = {}
        windows: set = set()
        rerun_windows: set = set()
        thinking: set = set()
        engine_kinds: set = set()
        for dd in decl.drive_dirs:
            rerun_dir = False
            log_inp = by_uri.get(os.path.join(dd, "drive.log"))
            if log_inp is not None:
                with open(log_inp.path, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        m = DRIVE_START_RE.match(line.strip())
                        if m:
                            is_rerun = RERUN_MARK in line
                            rerun_dir = rerun_dir or is_rerun
                            drive_rec = {"id": m.group("drive"), "utc": m.group("utc"), "sha": m.group("sha"), "cluster": m.group("cluster"),
                                         "org": m.group("org"), "dir": dd, "rerun": is_rerun}
                            mw = WIF_ATTEST_RE.search(line)
                            ms = SECRETS_MODE_RE.search(line)
                            if mw:      # a federated drive: the pool is the org whose ids file digests to what the drive attested, never the label alone
                                drive_rec.update({"auth": "wif", "auth_org": mw.group("org"), "ids_sha12": mw.group("sha12"), "org_verified": wif_org_of_ids_sha12(mw.group("sha12"))})
                            elif ms and ms.group("mode") == "wif":
                                # a wif drive that attests NO digest (round 1 on 7778694d0): nothing proves which identity its auditor / judge calls went
                                # through -- the label is not trusted; the pool reads wif-unverified:unattested and the ledger says so
                                drive_rec.update({"auth": "wif", "auth_org": None, "ids_sha12": None, "org_verified": None, "unattested": True})
                            drives[m.group("drive")] = drive_rec
            for uri, inp in by_uri.items():
                if os.path.dirname(uri) == dd and re.fullmatch(r"leg_\d+\.rc", os.path.basename(uri)):
                    with open(inp.path, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            m = RC_RE.match(line.strip())
                            if m:
                                rc[m.group("results")] = {"rc": int(m.group("rc")), "drive": m.group("drive"), "file": uri}
            ep = by_uri.get(os.path.join(dd, "endpoint.json"))
            if ep is not None:
                w = self._json_of(ep.path).get("attested_max_model_len")
                if isinstance(w, int) and not isinstance(w, bool):
                    (rerun_windows if rerun_dir else windows).add(int(w))
            mj = by_uri.get(os.path.join(dd, "models.json"))
            if mj is not None:
                proxy = self._json_of(mj.path).get("petri_proxy")
                if isinstance(proxy, Mapping):
                    if proxy.get("thinking_mode"):
                        thinking.add(str(proxy["thinking_mode"]))
                    eng = proxy.get("engine") if isinstance(proxy.get("engine"), Mapping) else {}
                    if eng.get("kind"):
                        engine_kinds.add(str(eng["kind"]))
        completing = {drives[e["drive"]]["sha"] for e in rc.values() if e["rc"] == 0 and e["drive"] in drives}
        if len(windows) > 1:
            raise AdapterRefusal(decl.root, f"the main-pass standup dirs attest {len(windows)} different served windows {sorted(windows)}; one instrument cannot name two")
        if len(rerun_windows) > 1 or (rerun_windows and windows and min(rerun_windows) <= max(windows)):
            raise AdapterRefusal(decl.root, f"the overflow re-run standup attests window(s) {sorted(rerun_windows)} against the main pass's {sorted(windows)}; the re-run pass "
                                            f"widens the window ({RERUN_WINDOW_TOKENS:,}), anything else is another instrument")
        proxy_jobs: Dict[str, str] = {}
        launch = os.path.join(PETRIWAVE_STATE_ROOT, "launch.log")
        if os.path.isfile(launch):
            with open(launch, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = re.search(rf"\b{re.escape(decl.seed_label)} PROXY-READY job=(\d+) dir=(\S+)", line)
                    if m:
                        proxy_jobs[os.path.basename(m.group(2)).replace("serving_second", f"{MODEL_KEY}_{decl.seed_label}_second").replace("serving", f"{MODEL_KEY}_{decl.seed_label}")] = m.group(1)
        escalation: List[str] = []
        if os.path.isfile(self.escalation_log):
            with open(self.escalation_log, "r", encoding="utf-8", errors="replace") as fh:
                escalation = [ln.strip() for ln in fh if ln.strip()][:20]
        return {"drives": drives, "rc": rc, "windows": windows, "window": (next(iter(windows)) if windows else None), "rerun_windows": rerun_windows,
                "rerun_window": (next(iter(rerun_windows)) if rerun_windows else None), "thinking": thinking, "engine_kinds": engine_kinds,
                "trees": completing, "tree": (next(iter(completing)) if len(completing) == 1 else None), "proxy_jobs": proxy_jobs, "escalation": escalation}

    @staticmethod
    def _pools_of(ledger: Mapping[str, Any], drive: Mapping[str, Any]) -> Dict[str, Any]:
        """The API org(s) an audit's auditor / judge calls went through: every drive of the seed that started at or before the completing drive,
        whatever its tree (the inspect resume dir keeps the samples earlier drives scored; a fix drive on another tree re-converts them, a re-run
        drive consolidates a main-pass audit). One org -> the judgment's api_pool; several -> None with the drives per org in extra (cellstore's
        rule for a mixed pool)."""
        same = [d for d in ledger["drives"].values() if d["utc"] <= drive["utc"]]
        by_org: Dict[str, List[str]] = {}
        for d in sorted(same, key=lambda x: x["utc"]):
            by_org.setdefault(drive_pool_key(d), []).append(d["id"])
        pool = next(iter(by_org)) if len(by_org) == 1 and next(iter(by_org)) in ("lowprio", "highprio", "batch") else None
        return {"api_pool": pool, "api_pools": by_org}

    def _fallback_summary(self, decl: StoreDeclaration, by_uri: Mapping[str, Input], facts: Dict[str, Any]) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
        """The producer's judge-chain mirror indexed by (seed label, paper label): the fallback-judged (behaviour, scenario) pairs with their chain note,
        the auditor-fallback pairs; None (recorded as unverified on every item) when the file is not on this host."""
        inp = by_uri.get(str(decl.fallback_summary)) if decl.fallback_summary else None
        if inp is None and decl.fallback_summary and os.path.isfile(decl.fallback_summary):
            # the mirror is one file shared by the minted seeds and an input belongs to one group (the first minted seed's); the other seeds read
            # the same bytes here, hashed again — its sha is already part of that group's revision
            inp = check_input(decl.fallback_summary, role="report")
        if inp is None:
            facts["judge_chain_mirror"] = None
            return None
        data = self._load_any(inp.path)
        if not isinstance(data, list):
            raise AdapterRefusal(inp.uri, "the judge-chain mirror is not a list of per-(seed, object) entries")
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for e in data:
            if not isinstance(e, Mapping) or not e.get("view") or not e.get("cell"):
                continue
            # the mirror spells a prompted object's behaviour as the behaviour DIRECTORY (`<behaviour>/PROMPTED_<SI>`), as the rows do: the behaviour is the head
            fb = {}
            for row in (e.get("fallback_judged") or []):
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    fb[(str(row[0]).split("/", 1)[0], str(row[1]))] = str(row[2]) if len(row) > 2 else ""
            af = {(str(r[0]).split("/", 1)[0], str(r[1])): (str(r[2]) if len(r) > 2 else "") for r in (e.get("auditor_fallback") or []) if isinstance(r, (list, tuple)) and len(r) >= 2}
            out[(str(e["view"]), str(e["cell"]))] = {"fallback_judged": fb, "auditor_fallback": af, "status": e.get("status"), "uri": LC.rel_home(inp.uri), "sha256": inp.sha256}
        facts["judge_chain_mirror"] = {"uri": LC.rel_home(inp.uri), "sha256": inp.sha256, "entries": len(out)}
        return out

    def _judge_chain_of(self, fallback: Optional[Mapping[Tuple[str, str], Dict[str, Any]]], seed_label: str, orow: Mapping[str, Any], store_key_legacy: str,
                        rec: Mapping[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
        """Which judge scored each scenario of one audit, from the mirror's entry for (seed, the object's paper label): the primary judge unless the
        (behaviour, scenario) is fallback-judged. An entry naming a scenario the store did not score is a mismatch, recorded and refused by name."""
        label = "none" if orow.get("object_id") == "none" else str(orow.get("paper_label") or "")
        if fallback is None:
            return {"state": "unverified", "why": "PetriWave's judge-chain mirror (summary.json) is not on this host; which judge scored each scenario is unrecorded"}
        entry = fallback.get((seed_label, label))
        if entry is None:
            return {"state": "unverified", "why": f"the judge-chain mirror carries no entry for ({seed_label}, {label})"}
        scored = {(b, sc) for b, pb in rec["per_beh"].items() for sc in pb["scored"]} | {(b, sc) for b, pb in rec["per_beh"].items() for sc in pb["refused"]}
        by_scenario: Dict[str, List[str]] = {}
        for b, sc in scored:
            by_scenario.setdefault(sc, []).append(b)

        def resolve(pairs: Mapping[Tuple[str, str], str]) -> Tuple[Dict[Tuple[str, str], str], List[Tuple[str, str]]]:
            # a prompted object's mirror rows spell the behaviour DIRECTORY variant alone (`PROMPTED_SAFETY`): the behaviour is then the ONE behaviour
            # of the store that carries the scenario (scenario names are unique within the bundle's behaviours); anything else is a mismatch
            out: Dict[Tuple[str, str], str] = {}
            bad: List[Tuple[str, str]] = []
            for (btok, sc), note in pairs.items():
                if (btok, sc) in scored:
                    out[(btok, sc)] = note
                elif btok not in BEHAVIOURS and len(by_scenario.get(sc, [])) == 1:
                    out[(by_scenario[sc][0], sc)] = note
                else:
                    bad.append((btok, sc))
            return out, sorted(bad)
        fb, unknown = resolve(entry["fallback_judged"])
        af, unknown_af = resolve(entry["auditor_fallback"])
        facts.setdefault("judge_chain", {})[store_key_legacy] = {"label": label, "n_fallback_judged": len(fb), "n_auditor_fallback": len(af),
                                                                 "unknown_scenarios": unknown + unknown_af}
        if unknown or unknown_af:
            return {"state": "mismatch", "why": f"the mirror names {len(unknown) + len(unknown_af)} fallback scenario(s) the store neither scored nor refused: {(unknown + unknown_af)[:3]}"}
        return {"state": "recorded", "fallback_judged": fb, "auditor_fallback": af, "uri": entry["uri"], "sha256": entry["sha256"]}

    def _mint_instruments(self, batch: RecordBatch, ctx: AdapterContext, decl: StoreDeclaration, cfg: Mapping[str, Any], ledger: Mapping[str, Any], rec: Mapping[str, Any],
                          facts: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """One instrument per behaviour eval, the way the premint minted its derived instruments: the seeded `petri_full22_as_run` row for the served
        model with the store's declared changes (the vendor decode the manifest sent, per-scenario-turn seeds, the attested window, the tree the
        drive ran, the auditor chain the config declares), minted through identity.instrument_id and written as an `instrument` record. The judge
        is the seeded Petri judge, which the config's judge_model must resolve to (an ad-hoc judge is never minted)."""
        producer, now = ctx.producer, ctx.now
        seeds = self.seeds
        jrow = seeds.judge_row(JUDGE_SEED)
        cfg_judge = str(cfg.get("judge_model") or "").split("/")[-1]
        if cfg_judge != str(jrow.get("judge_model")):
            raise AdapterRefusal(str(decl.manifest_config), f"the manifest config's judge_model {cfg.get('judge_model')!r} does not resolve to seed judge {JUDGE_SEED} "
                                                            f"({jrow.get('judge_model')!r}); no ad-hoc judge is minted — seed the judge first")
        judge_id = seeds.judge_id(JUDGE_SEED)
        man = rec["manifest"]
        td = man.get("target_decode") if isinstance(man.get("target_decode"), Mapping) else {}
        req = dict(rec["request_params"])
        if str(td.get("policy_id") or "") != str(cfg.get("policy_id") or ""):
            raise AdapterRefusal(str(decl.manifest_config), f"the manifest config declares decode policy {cfg.get('policy_id')!r}, the store's manifest sent {td.get('policy_id')!r}")
        if str(man.get("scenario_bundle_hash") or "") != str(cfg.get("scenario_bundle_hash") or ""):
            raise AdapterRefusal(str(decl.manifest_config), f"the manifest config declares scenario bundle {cfg.get('scenario_bundle_hash')!r}, the store {man.get('scenario_bundle_hash')!r}")
        if str(man.get("model_id") or "") != str(cfg.get("target_model") or "") or (cfg.get("target_revision") and man.get("model_revision") != cfg.get("target_revision")):
            raise AdapterRefusal(str(decl.manifest_config), f"the manifest config declares target {cfg.get('target_model')} @ {cfg.get('target_revision')}, the store "
                                                            f"{man.get('model_id')} @ {man.get('model_revision')}")
        tree = ledger.get("tree")
        if not _is_sha1(tree):
            raise AdapterRefusal(decl.root, "no completed audit joins to a drive with a 40-hex tree sha; the instrument's harness version is not guessed")
        window = ledger.get("window")
        if window is None:
            raise AdapterRefusal(decl.root, "no standup dir attests the served window (endpoint.json attested_max_model_len); the instrument's window is not guessed")
        thinking = sorted(ledger.get("thinking") or [])
        if len(thinking) > 1:
            raise AdapterRefusal(decl.root, f"the standup dirs declare {thinking} thinking modes; one instrument cannot name two")
        pkgs = cfg.get("serving_packages") or {}
        harness_version = (f"petri-bloom audit via scripts/petri_run.py @ {tree}; serving " + ", ".join(f"{k} {v}" for k, v in sorted(pkgs.items()))
                           + (f"; serving_impl_commit {cfg['serving_impl_commit']}" if cfg.get("serving_impl_commit") else "")
                           + f" ({os.path.basename(str(decl.manifest_config))})")
        bloom = dict(cfg.get("bloom_options") or {})
        attacker = {"auditor": cfg.get("auditor_model"), "auditor_fallback": list(cfg.get("auditor_fallback_models") or []),
                    "max_turns": bloom.get("max_turns"), "enable_prefill": bloom.get("enable_prefill"), "enable_rollback": bloom.get("enable_rollback"),
                    "compaction": bloom.get("compaction"), "realism_filter": bloom.get("realism_filter")}
        cap = LC.int_or_none(td.get("max_tokens")) or 1024
        insts: Dict[str, Dict[str, Any]] = {}
        judges: Dict[str, str] = {}
        # the six behaviour evals and the pooled one (v3): the seeded Petri row declares all seven, so the pooled instrument is the same
        # derivation with eval_id petri_pooled
        for b in BEHAVIOURS + (POOLED_EVAL,):
            ev = seeds.eval_for_spelling(b) or b
            base = seeds.instrument_row(PETRI_INSTRUMENT_SEED, ev, model_id=MODEL_ID)
            payload: Dict[str, Any] = {name: base.get(name) for name, _t in I.INSTRUMENT_ID_FIELDS}
            changed = {"harness_version": harness_version, "context_window_tokens": int(window), "token_cap_tokens": cap, "reply_cap_tokens": cap,
                       "decode_seed_rule": "per_scenario_turn", "decode_source": "model_card", "attacker_decode_json": attacker, "judge_id": judge_id,
                       "serving_path": str(man.get("serving_engine") or "lora_native"), "max_turns": LC.int_or_none(bloom.get("max_turns")) or base.get("max_turns"),
                       "thinking_mode": (thinking[0] if thinking else str(base.get("thinking_mode") or "off")), "n_trials": 1}
            for k in DECODE_KEYS:
                changed[f"decode_{k}"] = req.get(k)
            payload.update(changed)
            payload["instrument_id"] = I.instrument_id(payload)
            payload["notes"] = (f"Derived instrument of PetriWave's Slurm Petri audit of seed {decl.seed_index} ({decl.audit}): the canonical seeded instrument "
                                f"{PETRI_INSTRUMENT_SEED} ({base.get('instrument_id')}) with the store's declared changes (extra.changed_fields); minted by the "
                                f"{NAME} reader through identity.instrument_id, the way the seed-1 premint minted its W24 instruments")
            payload["extra"] = {"derived_from_instrument_id": base.get("instrument_id"), "derived_from_seed_instrument": PETRI_INSTRUMENT_SEED,
                                "changed_fields": {k: v for k, v in changed.items() if base.get(k) != v}, "model_id": MODEL_ID, "judge_seed_id": JUDGE_SEED,
                                "judge_fallback_models": list(cfg.get("judge_fallback_models") or []), "decode_policy_ref": cfg.get("policy_id"), "tree_sha": tree,
                                "manifest_config": cfg.get("config_uri"), "manifest_config_sha256": cfg.get("config_sha256"), "audit": decl.audit,
                                "overflow_rerun_window_tokens": ledger.get("rerun_window"), "minted_by": NAME, "emission_version": self.EMISSION_VERSION}
            payload["field_confidence_json"] = {"prompt_template_sha256": LC.unverified("the served path renders the chat template server-side; no prompt-template "
                                                                                         "hash is recorded for this harness")}
            batch.add("instrument", payload, producer, now, observe=False)
            insts[ev] = payload
            judges[ev] = judge_id
            facts["instruments"][f"{ev}@{tree[:12]}"] = {"instrument_id": payload["instrument_id"], "derived_from": base.get("instrument_id"), "tree_sha": tree, "window": int(window)}
        return insts, judges

    def _emit_pending(self, batch: RecordBatch, ctx: AdapterContext, decl: StoreDeclaration, store_key_legacy: str, campaign_key_legacy: str, key: str,
                      unresolved: Sequence[Mapping[str, Any]], inc_inp: Input, facts: Dict[str, Any]) -> None:
        """ONE pending disposition per behaviour eval of an audit the runner withheld for context-overflow casualties: the widened-window re-run pass
        consolidates it afterwards; nothing is filed until its results.json exists."""
        try:
            object_id, _okey = self.resolve_object(campaign_key_legacy, decl.roster_selection)
        except (LookupError, AdapterRefusal) as exc:
            self.refusals.add(f"{decl.seed_label}|{store_key_legacy}", f"object unresolvable through the roster's {decl.roster_selection} selection / pins / si_texts: {exc}", self.roster_path)
            return
        if object_id != "none" and self._object_row(ctx, object_id) is None:
            self.refusals.add(f"{decl.seed_label}|{store_key_legacy}", f"object row {object_id} is not in the connected registry; no disposition is filed against an absent object", inc_inp.uri)
            return
        casualties = sorted({(str(u.get("behavior") or u.get("behaviour") or ""), str(u.get("scenario_id") or u.get("sample_id") or "")) for u in unresolved})
        by_b: Dict[str, List[str]] = {}
        for b, sc in casualties:
            by_b.setdefault(b, []).append(sc)
        reason = (f"pending: {len(casualties)} scenario(s) withheld by the runner as target context overflow ({'; '.join(f'{b}: {', '.join(v)}' for b, v in sorted(by_b.items()))}); "
                  f"the {RERUN_WINDOW_TOKENS:,}-token-window re-run pass consolidates the audit afterwards — no results.json yet, nothing is filed until it exists")
        facts["pending"][store_key_legacy] = {"petri_run_key": key, "casualties": {b: v for b, v in sorted(by_b.items())}, "n_unresolved_entries": len(unresolved)}
        # one disposition per behaviour eval AND one for the pooled eval (v3): the pooled run is derived from all six, so it waits with them
        for b in BEHAVIOURS + (POOLED_EVAL,):
            ev = self.seeds.eval_for_spelling(b) or b
            payload = {"disposition_id": str(uuid.uuid5(I.RECORD_NAMESPACE, I.canonical_json([NAME, SOURCE, decl.seed_label, key, ev, "pending", "overflow_rerun"]))),
                       "round_id": LC.ROUND_ID, "roster_entry_id": None, "comparison_id": f"{SOURCE}/petri/{ev}", "model_id": MODEL_ID,
                       "object_id": (object_id if object_id != "none" else None), "eval_id": ev, "state": "pending", "reason": reason,
                       "source_ref": LC.rel_home(inc_inp.uri), "ruled_utc": ctx.now,
                       "extra": {"view_legacy": decl.seed_label, "campaign_cell_id_legacy": campaign_key_legacy, "store_object_key_legacy": store_key_legacy,
                                 "lane_source_legacy": SOURCE, "petri_run_key": key, "object_id": object_id, "casualties": {bb: v for bb, v in sorted(by_b.items())},
                                 "behaviour_affected": (bool(by_b) if b == POOLED_EVAL else b in by_b), "pooled": b == POOLED_EVAL,
                                 "campaign": decl.campaign, "producer_key": decl.producer_key}}
            batch.add("disposition", payload, ctx.producer, ctx.now)

    # ---- one object's audit ---------------------------------------------------------------------
    def _read_object(self, root: str, entry: Mapping[str, Any], key: str, campaign_key_legacy: str, files: Mapping[str, Optional[Input]], rkey: str,
                     audit: str = STORE_PIN["audit"], seed_index: int = SEED_INDEX) -> Optional[Dict[str, Any]]:
        """The audit's four (five) files read and cross-checked: the manifest, the results, the rows folded per (behaviour, scenario, dimension),
        the withheld list (must be empty) and the terminal judge refusals. None with the refusal filed when the files do not agree."""
        man = self._json_of(files["manifest.json"].path)
        res = self._json_of(files["results.json"].path)
        inc = self._json_of(files["petri_incomplete.json"].path)
        jr = self._json_of(files["petri_judge_refused.json"].path) if files.get("petri_judge_refused.json") is not None else {}
        store_key_legacy = str(entry.get("cell_id"))
        problems: List[str] = []
        if str(man.get("cell_id")) != store_key_legacy or str(man.get("petri_run_key")) != key:
            problems.append(f"manifest names ({man.get('cell_id')}, {man.get('petri_run_key')}), cells.json ({store_key_legacy}, {key})")
        if str(res.get("cell_id")) != store_key_legacy or str(res.get("petri_run_key")) != key:
            problems.append(f"results.json names ({res.get('cell_id')}, {res.get('petri_run_key')}), cells.json ({store_key_legacy}, {key})")
        if str(man.get("sweep") or "") != audit:
            problems.append(f"manifest sweep {man.get('sweep')!r} is not the audit {audit!r}")
        for f in ("intervention_hash", "harness_config_hash"):
            if entry.get(f) and man.get(f) != entry.get(f):
                problems.append(f"manifest {f} {man.get(f)!r} vs cells.json {entry.get(f)!r}")
        if entry.get("serving_identity") and man.get("engine") != entry.get("serving_identity"):
            problems.append(f"manifest engine {man.get('engine')!r} vs cells.json serving_identity {entry.get('serving_identity')!r}")
        hf_model = self.seeds.model_for_hf_id(str(man.get("model_id") or ""))
        if hf_model != MODEL_ID:
            problems.append(f"manifest model_id {man.get('model_id')!r} resolves to {hf_model!r}, not {MODEL_ID}")
        if man.get("target_seed_base") != seed_index:
            problems.append(f"manifest target_seed_base {man.get('target_seed_base')!r} is not the seed index {seed_index} of the declared store")
        td = man.get("target_decode") if isinstance(man.get("target_decode"), Mapping) else {}
        if td.get("seed_rule") != "per_scenario_turn":
            problems.append(f"manifest target_decode.seed_rule {td.get('seed_rule')!r} is not per_scenario_turn")
        request_params = dict(td.get("request_params") or {}) if isinstance(td.get("request_params"), Mapping) else {}
        bundle = str(man.get("scenario_bundle_hash") or "")
        if not _is_hex16(bundle) or not os.path.isdir(os.path.join(root, f"behaviors-{bundle}")):
            problems.append(f"manifest scenario_bundle_hash {bundle!r} names no behaviours directory of the store")
        unresolved = [u for u in (inc.get("unresolved") or []) if isinstance(u, Mapping)] if isinstance(inc, Mapping) else []
        if unresolved:
            problems.append(f"petri_incomplete.json names {len(unresolved)} withheld item(s): the audit is incomplete, not a refusal to score")
        audited = res.get("audited") if isinstance(res.get("audited"), Mapping) else {}
        if LC.int_or_none(audited.get("expected")) != len(BEHAVIOURS) * N_SCENARIOS:
            problems.append(f"results.json audited.expected {audited.get('expected')!r} is not {len(BEHAVIOURS) * N_SCENARIOS}")
        when = epoch_utc(res.get("completed_at"))
        if when is None:
            problems.append("results.json carries no completed_at")
        if problems:
            self.refusals.add(rkey, "; ".join(problems), files["results.json"].uri)
            return None
        refused_all = [x for x in (jr.get("judge_refused") or []) if isinstance(x, Mapping)] if isinstance(jr, Mapping) else []
        n_ref_total = LC.int_or_none((res.get("judge_refused") or {}).get("total")) if isinstance(res.get("judge_refused"), Mapping) else None
        if n_ref_total is not None and n_ref_total != len(refused_all):
            self.refusals.add(rkey, f"results.json counts {n_ref_total} judge refusals, petri_judge_refused.json lists {len(refused_all)}", files["results.json"].uri)
            return None
        latest, n_superseded = RP.read_petri_rows(files["rows.jsonl"].path)
        scores: Dict[str, Dict[str, float]] = {b: {} for b in BEHAVIOURS}
        shared: Dict[str, Dict[str, Dict[str, float]]] = {d: {b: {} for b in BEHAVIOURS} for d in SHARED_DIMENSIONS}
        created: Dict[str, Dict[str, str]] = {b: {} for b in BEHAVIOURS}
        judge_fields: Dict[str, Dict[str, Dict[str, Any]]] = {b: {} for b in BEHAVIOURS}
        dims: Dict[str, Optional[str]] = {}
        auditor_models: set = set()
        variants: set = set()
        for (bkey, item, metric), r in latest.items():
            # the behaviour is the task's (`petri/<behaviour>`); a prompted object's rows spell the behaviour DIRECTORY with the bundle
            # variant that injects the system instruction (`<behaviour>/PROMPTED_<SI>`), which is recorded, never a seventh behaviour
            task = str(r.get("task") or "")
            b = task.split("/", 1)[1] if task.startswith("petri/") else bkey.split("/", 1)[0]
            if bkey.split("/", 1)[0] != b:
                self.refusals.add(rkey, f"rows.jsonl row ({bkey}, {item}, {metric}) names task {task!r} for behaviour directory {bkey!r}", files["rows.jsonl"].uri)
                return None
            variants.add(bkey.split("/", 1)[1] if "/" in bkey else None)
            if b not in scores:
                self.refusals.add(rkey, f"rows.jsonl names behaviour {b!r}, none of the six", files["rows.jsonl"].uri)
                return None
            v = r.get("value")
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                self.refusals.add(rkey, f"rows.jsonl row ({b}, {item}, {metric}) carries no finite value", files["rows.jsonl"].uri)
                return None
            ex = RP.petri_extra(r)
            chain = ex.get("auditor_chain") if isinstance(ex.get("auditor_chain"), Mapping) else {}
            for u in (chain.get("used") or []):
                if isinstance(u, Mapping) and u.get("model"):
                    auditor_models.add(str(u["model"]))
            if metric == b:
                scores[b][item] = float(v)
                created[b][item] = epoch_utc(r.get("created_at")) or when
                dims.setdefault(b, ex.get("dimension"))
                # the producer's per-row judge provenance (the behaviour's own score row carries it; absent on the v2 stores)
                pj = producer_judge_fields(ex)
                if pj:
                    judge_fields[b][item] = pj
            elif metric in shared:
                shared[metric][b][item] = float(v)
            else:
                self.refusals.add(rkey, f"rows.jsonl names metric {metric!r} under {b}: neither the behaviour nor a shared dimension", files["rows.jsonl"].uri)
                return None
        per_beh: Dict[str, Dict[str, Any]] = {}
        beh_res = res.get("behaviors") if isinstance(res.get("behaviors"), Mapping) else {}
        for b in BEHAVIOURS:
            rb = beh_res.get(b) if isinstance(beh_res.get(b), Mapping) else {}
            refused_b = [x for x in refused_all if str(x.get("behavior")) == b]
            refused_ids = {str(x.get("scenario_id") or x.get("sample_id")) for x in refused_b}
            scored_ids = set(scores[b])
            n_scored, n_refused = len(scored_ids), len(refused_ids)
            probs: List[str] = []
            if n_scored != LC.int_or_none(rb.get("audited")):
                probs.append(f"{n_scored} scored scenarios in rows.jsonl, results.json audited {rb.get('audited')!r}")
            if LC.int_or_none(rb.get("n_items")) not in (None, N_SCENARIOS):
                probs.append(f"results.json n_items {rb.get('n_items')!r} is not the bundle's {N_SCENARIOS} scenarios")
            if scored_ids & refused_ids:
                probs.append(f"scenarios both scored and judge-refused: {sorted(scored_ids & refused_ids)}")
            if n_scored + n_refused != N_SCENARIOS or LC.int_or_none(rb.get("n_scenarios")) not in (None, N_SCENARIOS):
                probs.append(f"scored {n_scored} + refused {n_refused} != {N_SCENARIOS} scenarios (results.json n_scenarios {rb.get('n_scenarios')!r})")
            jr_items = rb.get("judge_refused_items")
            if isinstance(jr_items, list) and len(jr_items) != n_refused:
                probs.append(f"results.json lists {len(jr_items)} judge-refused items for the behaviour, petri_judge_refused.json {n_refused}")
            mean_prod = (rb.get("mean") or {}).get(b) if isinstance(rb.get("mean"), Mapping) else None
            mean_read = (sum(scores[b].values()) / n_scored) if n_scored else None
            if mean_read is None or not isinstance(mean_prod, (int, float)) or abs(float(mean_prod) - mean_read) > 1e-9:
                probs.append(f"rows.jsonl mean over scored {mean_read!r} does not reproduce results.json mean {mean_prod!r}")
            for d in SHARED_DIMENSIONS:
                if set(shared[d][b]) != scored_ids:
                    probs.append(f"the {d} rows of {b} cover {sorted(set(shared[d][b]) ^ scored_ids)} differently from the scored scenarios")
            if probs:
                self.refusals.add(f"{rkey}|{b}", "; ".join(probs), files["rows.jsonl"].uri)
                return None
            per_beh[b] = {"scored": scores[b], "refused": {str(x.get("scenario_id") or x.get("sample_id")): x for x in refused_b}, "n_scored": n_scored, "n_refused": n_refused,
                          "mean": mean_read, "created": created[b], "shared": {d: shared[d][b] for d in SHARED_DIMENSIONS}, "dimension": dims.get(b),
                          "judge_fields": judge_fields[b],
                          "producer": {"mean": mean_prod, "n_items": rb.get("n_items"), "audited": rb.get("audited"), "auditor_chain": rb.get("auditor_chain"),
                                       "auditor_refused": rb.get("auditor_refused"), "shared_means": {d: (rb.get("mean") or {}).get(d) for d in SHARED_DIMENSIONS}}}
        if len(variants) != 1:
            self.refusals.add(rkey, f"rows.jsonl names {len(variants)} behaviour-directory variants {sorted(str(v) for v in variants)}; one audit runs one bundle", files["rows.jsonl"].uri)
            return None
        return {"key": key, "manifest": man, "results": res, "request_params": request_params, "when": when, "per_beh": per_beh, "n_superseded_rows": n_superseded,
                "auditor_models": sorted(auditor_models), "refused_total": len(refused_all), "n_unresolved": len(unresolved), "runs": {}, "run_payloads": {},
                "bundle_variant": next(iter(variants)), "audited_expected": LC.int_or_none(audited.get("expected")), "audited_items": LC.int_or_none(audited.get("items"))}

    # ---- emission ---------------------------------------------------------------------------------
    def _admission(self, waiver_inp: Input, drive: Mapping[str, Any]) -> Dict[str, Any]:
        """The two provenance facts every run carries: the waiver of the determinism spot-check (read from the file and checked against
        the drive's tree and seed) and the absent re-audit."""
        entries = self._load_any(waiver_inp.path)
        if isinstance(entries, Mapping):
            entries = [entries]
        if not isinstance(entries, list) or not entries:
            raise AdapterRefusal(waiver_inp.uri, "the waiver record carries no entries")
        hit = None
        for e in entries:
            if not isinstance(e, Mapping):
                continue
            scope = e.get("scope") if isinstance(e.get("scope"), Mapping) else {}
            if scope.get("W") == drive.get("W") and scope.get("view") == drive.get("view") and scope.get("leg") == "petri":
                hit = e
                break
        if hit is None:
            raise AdapterRefusal(waiver_inp.uri, f"no waiver entry covers tree {drive.get('W')!r} / seed {drive.get('view')!r} / petri; the store is not filed as if it "
                                                 f"passed criterion 3 and not without its waiver")
        scope = hit.get("scope") if isinstance(hit.get("scope"), Mapping) else {}
        return {"criterion3_determinism": "waived", "waiver_ref": LC.rel_home(waiver_inp.uri), "waiver_sha256": waiver_inp.sha256, "waiver_decision": WAIVER_DECISION_REF,
                "waiver_by": hit.get("by"), "waiver_at": hit.get("at"), "waiver_reason": hit.get("reason"), "waiver_decision_text": hit.get("decision"),
                "waiver_scope_legs": list(scope.get("legs") or []), "w23_reaudit": W23_REAUDIT_STATE, "w23_reaudit_note": W23_REAUDIT_NOTE,
                "carried_under_tree": CARRIED_UNDER_TREE, "drove_tree": drive.get("W"), "drive_id": drive.get("drive_id"), "attempt_id": drive.get("attempt_id")}

    @staticmethod
    def _load_any(path: str):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError as exc:
                raise AdapterRefusal(path, f"not JSON: {exc.msg}") from exc

    def _instrument_match(self, inst: Mapping[str, Any], rec: Mapping[str, Any]) -> Tuple[str, List[str]]:
        """The store's recorded settings against the instrument row the premint bound: the cap, the six decode numbers, the seed rule, the
        serving path, the scenario bundle (named in the instrument's harness_config_hash text) and the auditor model."""
        man, req = rec["manifest"], rec["request_params"]
        td = man.get("target_decode") if isinstance(man.get("target_decode"), Mapping) else {}
        mism: List[str] = []
        if LC.int_or_none(td.get("max_tokens")) != LC.int_or_none(inst.get("token_cap_tokens")):
            mism.append(f"max_tokens {td.get('max_tokens')!r} vs instrument token_cap_tokens {inst.get('token_cap_tokens')!r}")
        for k in DECODE_KEYS:
            iv = inst.get(f"decode_{k}")
            if iv is not None and (req.get(k) is None or float(req[k]) != float(iv)):
                mism.append(f"{k} {req.get(k)!r} vs instrument {iv!r}")
        if td.get("seed_rule") != inst.get("decode_seed_rule"):
            mism.append(f"seed_rule {td.get('seed_rule')!r} vs instrument {inst.get('decode_seed_rule')!r}")
        if man.get("serving_engine") != inst.get("serving_path"):
            mism.append(f"serving_engine {man.get('serving_engine')!r} vs instrument serving_path {inst.get('serving_path')!r}")
        if str(man.get("scenario_bundle_hash") or "") not in str(inst.get("harness_config_hash") or ""):
            mism.append(f"scenario_bundle_hash {man.get('scenario_bundle_hash')!r} is not named by the instrument's harness_config_hash")
        adj = inst.get("attacker_decode_json")
        try:
            adj = json.loads(adj) if isinstance(adj, str) else adj
        except ValueError:
            adj = None
        auditor = (adj or {}).get("auditor") if isinstance(adj, Mapping) else None
        if auditor and rec["auditor_models"] and rec["auditor_models"] != [auditor]:
            mism.append(f"auditor models used {rec['auditor_models']} vs the instrument's auditor {auditor!r}")
        return ("mismatch" if mism else "match"), mism

    def _emit_object(self, batch: RecordBatch, ctx: AdapterContext, rec: Dict[str, Any], bd: Mapping[str, Any], insts: Mapping[str, Mapping[str, Any]],
                     judges: Mapping[str, str], comp_id: str, rng, facts: Dict[str, Any], files: Mapping[str, Optional[Input]], cdir: str) -> None:
        """One object's six runs (judgment, metric, items, states) under the binding ``bd`` (campaign, producer key, seed, identities, provenance)."""
        producer, now = ctx.producer, ctx.now
        queued: Mapping[Tuple[str, str], Mapping[str, Any]] = bd.get("queued") or {}
        admission: Mapping[str, Any] = bd["admission"]
        seed_label, seed_index = str(bd["seed_label"]), int(bd["seed_index"])
        man, res, when = rec["manifest"], rec["results"], rec["when"]
        object_id, object_key, orow = rec["object_id"], rec["object_key"], rec["object_row"]
        campaign_key_legacy, store_key_legacy, entry = rec["campaign_key_legacy"], rec["store_key_legacy"], rec["entry"]
        fitted = object_id != "none" and not object_id.startswith("si:")
        kind = "none" if object_id == "none" else ("system_instruction" if object_id.startswith("si:") else str(orow.get("object_kind") or "unverified"))
        transfer_kind = "none" if object_id == "none" else ("prompted_only" if object_id.startswith("si:") else "base_to_instruct")
        rel_dir = LC.rel_home(cdir)
        n_scored_total = sum(pb["n_scored"] for pb in rec["per_beh"].values())
        ofacts: Dict[str, Any] = {"petri_run_key": rec["key"], "campaign_object_key_legacy": campaign_key_legacy, "object_id": object_id, "producer_object_key": object_key,
                                  "n_scored": n_scored_total, "n_judge_refused": rec["refused_total"], "per_behaviour_scored": {b: pb["n_scored"] for b, pb in rec["per_beh"].items()},
                                  "per_behaviour_refused": {b: pb["n_refused"] for b, pb in rec["per_beh"].items()}, "runs": {}, "premint_rows": {}}
        facts["objects"][store_key_legacy] = ofacts
        for b in BEHAVIOURS:
            ev = self.seeds.eval_for_spelling(b) or b
            inst = insts.get(ev)
            judge_id = judges.get(ev)
            if inst is None or judge_id is None:
                self.refusals.add(f"{seed_label}|{store_key_legacy}|{ev}", f"the premint-queued rows name no instrument / judge for {ev}; no run is minted without them", cdir)
                continue
            iid = str(inst["instrument_id"])
            rid = I.run_id(str(bd["campaign"]), str(bd["producer_key"]), object_key, ev, iid)
            trial = int(bd["trial_of"](object_id))
            q = queued.get((campaign_key_legacy, ev))
            if q is not None:
                bad = []
                if str(q["run_id"]) != rid:
                    bad.append(f"run_id {q['run_id']} != derived {rid}")
                if str(q["object_id"]) != object_id:
                    bad.append(f"object {q['object_id']} != resolved {object_id}")
                if str(q["producer_object_key_legacy"]) != object_key:
                    bad.append(f"producer object key {q['producer_object_key_legacy']!r} != {object_key!r}")
                if LC.int_or_none(q.get("trial_index")) != trial or LC.int_or_none(q.get("seed_base")) != seed_index:
                    bad.append(f"trial_index / seed_base ({q.get('trial_index')}, {q.get('seed_base')}) != ({trial}, {seed_index})")
                if bad:
                    self.refusals.add(f"{seed_label}|{store_key_legacy}|{ev}", "the premint-queued row disagrees with the derived identity: " + "; ".join(bad), cdir)
                    continue
                facts["premint"]["matched"] += 1
                ofacts["premint_rows"][ev] = {"run_id": rid, "lifecycle_state": q.get("lifecycle_state")}
            else:
                facts["premint"]["runs_without_premint"].append(rid)
                ofacts["premint_rows"][ev] = None
            pb = rec["per_beh"][b]
            inst_match, mism = self._instrument_match(inst, rec)
            if mism:
                # the premint instrument and the store's manifest must be ONE declaration: the object is refused by name rather than filed
                # on an instrument it did not run under (never re-pointed at another premint)
                self.refusals.add(f"{seed_label}|{store_key_legacy}|{ev}", f"the premint instrument {iid[:12]}… disagrees with the store's manifest: {'; '.join(mism)}", cdir)
                continue
            jid = I.judgment_id(rid, judge_id)
            store_uri = rel_dir
            fc: Dict[str, Any] = {
                "model_id": LC.confidence_entry("recorded", f"manifest model_id {man.get('model_id')} -> models.yaml hf id"),
                "model_revision": LC.confidence_entry("recorded", "manifest model_revision"),
                "object_id": LC.confidence_entry("derived", f"jobplan.object_id_for over the roster's {bd['roster_selection']} selection and the pins / si_texts"
                                                            + ("; equal to the premint-queued row's" if q is not None else "; no premint-queued row for this (object, eval)")),
                "instrument_id": LC.confidence_entry("recorded", str(bd["instrument_source"])),
                "instrument_match": LC.confidence_entry("derived", "the store's cap, decode numbers, seed rule, serving path, scenario bundle and auditor against the instrument row"
                                                                   + (f": {'; '.join(mism)}" if mism else "")),
                "judge_id": LC.confidence_entry("recorded", str(bd["judge_source"])),
                "code_sha": LC.confidence_entry("recorded", str(bd["code_sha_source"])),
                "substrate": LC.confidence_entry("recorded", str(bd["substrate_source_text"])),
                "n_items_planned": LC.confidence_entry("recorded", "results.json audited.expected / 6 behaviours; the item set's 10 scenarios"),
                "n_items_generated": LC.confidence_entry("derived", "scored + terminal judge refusals (petri_judge_refused.json); petri_incomplete.json withholds nothing: every scenario ran"),
                "n_items_judged": LC.confidence_entry("recorded", "the scored scenarios of rows.jsonl (checked against results.json n_items)"),
                "decode_sent_json": LC.confidence_entry("recorded", "manifest target_decode.request_params (equal to the registered policy row)"),
                "decode_seed_rule": LC.confidence_entry("recorded", "manifest target_decode.seed_rule"),
                "seed_base": LC.confidence_entry("recorded", "manifest target_seed_base"),
                "serving_identity": LC.confidence_entry("recorded", "manifest engine"),
                "queued_utc": LC.confidence_entry("derived", str(bd["queued_source"])),
                "started_utc": LC.confidence_entry("recorded", str(bd["started_source"])),
                "generated_utc": LC.confidence_entry("recorded", "results.json completed_at (the audit scores as it runs: generated = judged)"),
                "judged_utc": LC.confidence_entry("recorded", "results.json completed_at (set by the `judged` state event this reader files)"),
                "prompt_template_sha256": LC.unverified("the served path renders the chat template server-side; no prompt-template hash is recorded for this harness"),
                "admission": dict(bd["admission_fc"]),
            }
            payload: Dict[str, Any] = {
                "run_id": rid, "campaign": str(bd["campaign"]), "producer_run_key_legacy": str(bd["producer_key"]), "producer_object_key_legacy": object_key,
                "lifecycle_state": "generated", "queued_utc": rec["queued_utc"], "started_utc": rec["started_utc"], "generated_utc": when,
                "walltime_s": None, "model_id": MODEL_ID, "model_revision": (man.get("model_revision") if _is_sha1(man.get("model_revision")) else None),
                "object_id": object_id, "transfer_kind": transfer_kind,
                "eval_id": ev, "item_set_id": inst.get("item_set_id"), "instrument_id": iid, "instrument_match": inst_match,
                "harness": inst.get("harness"), "harness_version": inst.get("harness_version") or "unverified", "n_shot": inst.get("n_shot"),
                "cot_prompting": inst.get("cot_prompting"), "scoring_rule": inst.get("scoring_rule"), "prompt_template_sha256": "unverified",
                "judge_id": judge_id, "pass_label": None, "trial_index": trial, "n_items_planned": N_SCENARIOS, "n_items_generated": pb["n_scored"] + pb["n_refused"],
                "n_items_judged": pb["n_scored"], "seed_base": seed_index, "decode_seed_rule": "per_scenario_turn", "decode_policy_match": "no_policy",
                "decode_sent_json": dict(rec["request_params"], max_tokens=(man.get("target_decode") or {}).get("max_tokens")), "decode_resolved_json": {},
                "decode_temperature_sent": rec["request_params"].get("temperature"), "decode_top_p_sent": rec["request_params"].get("top_p"),
                "decode_top_k_sent": rec["request_params"].get("top_k"), "decode_source": inst.get("decode_source") or "unverified",
                "token_cap_tokens": int(inst.get("token_cap_tokens") or 1024), "token_cap_rung_index": 0, "token_cap_ladder_json": inst.get("token_cap_ladder_json") or "[]",
                "reply_cap_tokens": inst.get("reply_cap_tokens"), "context_window_tokens": inst.get("context_window_tokens"),
                "thinking_mode": inst.get("thinking_mode") or "off", "serving_path": inst.get("serving_path") or "unverified", "serving_identity": man.get("engine"),
                "applied_positions_mode": "all" if fitted else "none", "multi_turn_steering_scope": "every_turn_prefill_and_decode" if fitted else "not_applicable",
                "intervention_hash": man.get("intervention_hash") if _is_hex16(man.get("intervention_hash")) else None,
                "harness_config_hash": man.get("harness_config_hash") if _is_hex16(man.get("harness_config_hash")) else None,
                "env_digest": man.get("env_digest") if _is_hex16(man.get("env_digest")) else None,
                "code_sha": str(bd["code_sha"]) if _is_sha1(bd["code_sha"]) else "unverified", "substrate": str(bd["substrate"]), "substrate_source": "recorded",
                "cluster_job_id": bd.get("cluster_job_id"), "attacker_model_id": None, "store_uri": store_uri, "store_completion_artifact_uri": LC.rel_home(files["results.json"].uri),
                "store_completion_sha256": files["results.json"].sha256, "view_legacy": seed_label, "lane_source_legacy": SOURCE,
                "campaign_cell_id_legacy": campaign_key_legacy, "producer_session": PRODUCER_SESSION, "field_confidence_json": fc,
                "notes": (f"Petri seed-{seed_index} audit of {campaign_key_legacy} ({b}): {pb['n_scored']} of {N_SCENARIOS} scenarios scored, {pb['n_refused']} judge refusal(s) "
                          f"filed as unscored items; {bd['notes_tail']}"),
                "extra": {"petri_run_key": rec["key"], "behaviour": b, "dimension": pb["dimension"], "store_object_key_legacy": store_key_legacy,
                          "behaviour_bundle_variant_legacy": rec.get("bundle_variant"),
                          "object_staged_id": object_key if object_key != campaign_key_legacy else None, "object_kind_store_legacy": entry.get("kind"),
                          "object_class_store_legacy": entry.get("cell_class"), "virtual_model_name": man.get("virtual_model_name"),
                          "auditor_chain": res.get("auditor_chain"), "auditor_refused": res.get("auditor_refused"), "auditor_models_used": rec["auditor_models"],
                          "auditor_pool_legacy": bd.get("auditor_pool_legacy"),
                          "judge_refused_scenarios": sorted(pb["refused"]), "n_superseded_rows": rec["n_superseded_rows"],
                          "decode_policy_ref": rec["policy_id"], "seed_replicate_label": seed_label,
                          "premint": {"producer_key": str(bd["producer_key"]), "row": ("present" if q is not None else "absent"),
                                      "queued_lifecycle_state": (q.get("lifecycle_state") if q is not None else None)},
                          "admission": dict(admission, premint_row=("present" if q is not None else "absent")),
                          "instrument_mismatches": mism, **dict(bd.get("provenance_extra") or {})},
            }
            if fitted:
                payload.update(strength_nominal=1.0, strength_nominal_source="filed", strength_unit="nominal_only", is_operating_point=1)
                fc["strength_nominal"] = LC.confidence_entry("recorded", "a fitted seed-replicate object served at its campaign nominal (the premint's filing)")
                fc["strength_unit"] = LC.confidence_entry("unverified", "the record declares no dose unit; the dose is the campaign nominal")
            else:
                payload.update(strength_unit="unverified")
                fc["strength_unit"] = LC.unverified("an unsteered / prompted object has no dose")
            if payload["model_revision"] is None:
                fc["model_revision"] = LC.unverified("manifest model_revision is not a 40-hex sha")
            if payload["code_sha"] == "unverified":
                fc["code_sha"] = LC.unverified("modal_drive.json W is not a 40-hex sha")
            if payload["harness_version"] == "unverified":
                fc["harness_version"] = LC.unverified("the instrument row records no harness version")
            if payload["decode_source"] == "unverified":
                fc["decode_source"] = LC.unverified("the instrument row records no decode source")
            if payload["serving_path"] == "unverified":
                fc["serving_path"] = LC.unverified("the instrument row declares no serving path")
            launch_t = rec.get("started_utc")
            if launch_t and when:
                try:
                    payload["walltime_s"] = (R.utc_to_epoch_ms(when) - R.utc_to_epoch_ms(launch_t)) / 1000.0
                    payload["walltime_source"] = "store_span"
                except Exception:   # noqa: BLE001 — a malformed stamp leaves the walltime unfiled
                    payload["walltime_s"] = None
            batch.add("run", payload, producer, now)
            ev_uri = LC.rel_home(files["results.json"].uri)
            batch.add("run_state", {"run_id": rid, "state": "generated", "event_utc": when, "evidence_kind": "completion_artifact", "evidence_uri": ev_uri,
                                    "note": f"audit complete: {pb['n_scored'] + pb['n_refused']} of {N_SCENARIOS} scenarios ran ({pb['n_refused']} judge refusal(s))"},
                      producer, now, observe=False)
            batch.add("run_state", {"run_id": rid, "state": "judged", "event_utc": when, "evidence_kind": "completion_artifact", "evidence_uri": ev_uri,
                                    "note": f"{pb['n_scored']} scenarios scored by the audit's judge; {pb['n_refused']} refused to score"}, producer, now, observe=False)
            jpay = LC.judgment_payload(rid, judge_id, pb["n_scored"], when, store_uri, api_pool=bd.get("api_pool"),
                                       extra={"judge_seed_id": JUDGE_SEED, "judge_model": str(bd.get("judge_model") or "anthropic/claude-opus-5"),
                                              "judge_fallback_models": bd.get("judge_fallback_models"), "api_pools": bd.get("api_pools"),
                                              "refusal_reasons": sorted({str(x.get("reason")) for x in pb["refused"].values()}),
                                              "auditor_chain": pb["producer"]["auditor_chain"], "auditor_refused": pb["producer"]["auditor_refused"]})
            chain = rec.get("judge_chain") or {}
            primary_judge = str(bd.get("judge_model") or "anthropic/claude-opus-5")
            fb_b = {sc: note for (bb, sc), note in (chain.get("fallback_judged") or {}).items() if bb == b}
            af_b = {sc: note for (bb, sc), note in (chain.get("auditor_fallback") or {}).items() if bb == b}
            kinds = collections.Counter(refusal_kind(x.get("reason")) for x in pb["refused"].values())
            jpay["extra"].update({"refusal_kinds": dict(kinds), "judge_chain_state": chain.get("state", "not_applicable"), "judge_chain_note": chain.get("why"),
                                  "judge_fallback_count": (len(fb_b) if chain.get("state") == "recorded" else None), "judge_fallback_scenarios": (sorted(fb_b) if chain.get("state") == "recorded" else None),
                                  "judge_fallback_model": (list(bd.get("judge_fallback_models") or [None])[0] if fb_b else None),
                                  "auditor_fallback_count": (len(af_b) if chain.get("state") == "recorded" else None), "auditor_fallback_scenarios": (sorted(af_b) if chain.get("state") == "recorded" else None),
                                  "judge_chain_mirror": ({"uri": chain.get("uri"), "sha256": chain.get("sha256")} if chain.get("state") == "recorded" else None)})
            jpay.update(n_judge_refusals=pb["n_refused"], n_errors=0, producer_session=PRODUCER_SESSION, grading_code_sha=(payload["code_sha"] if payload["code_sha"] != "unverified" else None),
                        started_utc=rec["started_utc"])
            # the per-item judge markers (the mirror's chain, the producer's own per-row fields) and their roll-up onto the judgment (v3)
            judge_used = self._judge_used_fn(bd, chain, fb_b, primary_judge)
            markers = {sc: self._item_judge_marker(sc, pb, chain, fb_b, af_b, judge_used) for sc in list(pb["scored"]) + list(pb["refused"])}
            jpay["extra"].update(self._judge_marker_rollup(markers.values()))
            batch.add("judgment", jpay, producer, now)
            rec["run_payloads"][b] = payload
            rec.setdefault("markers", {})[b] = markers
            lo, hi = scenario_bootstrap(list(pb["scored"].values()), rng)
            sd = float(np.std(list(pb["scored"].values()), ddof=1)) if pb["n_scored"] > 1 else None
            batch.add("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": METRIC, "threshold": -1.0, "computation_id": comp_id,
                                     "producer_metric_spelling": f"behaviors.{b}.mean", "value": pb["mean"], "k": None, "n": pb["n_scored"], "ci_lo": lo, "ci_hi": hi,
                                     "ci_method": (f"percentile bootstrap over the {pb['n_scored']} scored scenarios ({N_BOOT} draws, seed {BOOT_SEED}; registry)" if lo is not None else "none filed"),
                                     "n_abstentions": pb["n_refused"], "n_incomplete": 0, "computed_by": "registry", "computed_utc": when, "source_uri": ev_uri,
                                     "source_producers_json": [{"producer": "results.json", "uri": ev_uri, "sha256": files["results.json"].sha256, "field_group": "behaviour mean, counts, judge refusals"},
                                                               {"producer": "rows.jsonl", "uri": LC.rel_home(files["rows.jsonl"].uri), "sha256": files["rows.jsonl"].sha256, "field_group": "per-scenario scores"}],
                                     "extra": {"denominator": "scored scenarios (judge refusals excluded, never imputed)", "sd": sd, "producer_mean": pb["producer"]["mean"],
                                               "producer_n_items": pb["producer"]["n_items"], "judge_refused_scenarios": sorted(pb["refused"]),
                                               "eval_awareness_mean": (sum(pb["shared"]["eval_awareness"].values()) / pb["n_scored"]) if pb["n_scored"] else None,
                                               "scenario_realism_mean": (sum(pb["shared"]["scenario_realism"].values()) / pb["n_scored"]) if pb["n_scored"] else None,
                                               "sign": "higher = MORE of the flagged behaviour (judge score 1-10)"}}, producer, now)
            facts["metrics"] += 1
            facts["items"] += self._emit_items(batch, producer, now, rid, jid, b, pb, when, markers)
            rec["runs"][b] = {"run_id": rid, "judgment_id": jid, "eval_id": ev}
            ofacts["runs"][ev] = {"run_id": rid, "n_scored": pb["n_scored"], "n_refused": pb["n_refused"], "mean": pb["mean"], "instrument_match": inst_match}
            facts["judged_runs"] += 1
            facts["per_behaviour"].setdefault(b, {})[store_key_legacy] = {"scored": pb["n_scored"], "refused": pb["n_refused"], "mean": pb["mean"]}
        # -- the pooled run (v3): derived from the six behaviour runs just filed, on the pooled instrument
        self._emit_pooled(batch, ctx, rec, bd, insts, judges, comp_id, rng, facts, files, cdir, ofacts)

    @staticmethod
    def _judge_used_fn(bd: Mapping[str, Any], chain: Mapping[str, Any], fb_b: Mapping[str, str], primary_judge: str):
        """Which judge the MIRROR says scored a scenario of one behaviour: None when the mirror is not recorded, the first fallback model when the
        scenario is fallback-judged, else the primary judge."""
        def judge_used(sc: str) -> Optional[str]:
            if chain.get("state") != "recorded":
                return None
            if sc in fb_b:
                return str(list(bd.get("judge_fallback_models") or [None])[0] or "fallback")
            return primary_judge
        return judge_used

    @staticmethod
    def _item_judge_marker(sc: str, pb: Mapping[str, Any], chain: Mapping[str, Any], fb_b: Mapping[str, str], af_b: Mapping[str, str], judge_used) -> Dict[str, Any]:
        """One item's judge-chain markers: v2's mirror-derived `judge_model_used` / chain notes, and (v3) the producer's per-row fields carried
        verbatim — the row's own `judge_model_used` is the producer's record and wins, the mirror's derivation is parked when it disagrees.
        A refused item's marker is the refusal entry's own judge (else the mirror's)."""
        out: Dict[str, Any] = {}
        recorded_chain = chain.get("state") is not None and chain.get("state") != "not_applicable"
        if sc in pb["refused"]:
            x = pb["refused"][sc]
            out.update({"reason": x.get("reason"), "refusal_kind": refusal_kind(x.get("reason")), "judge_model": x.get("judge_model"), "outcome": x.get("outcome"),
                        "auditor_turns": ((x.get("auditor_chain") or {}).get("auditor_turns") if isinstance(x.get("auditor_chain"), Mapping) else None)})
            if recorded_chain:
                out["judge_model_used"] = x.get("judge_model") or judge_used(sc)
                if sc in fb_b:
                    out["judge_chain_note"] = fb_b[sc]
            return out
        out["dimension"] = pb["dimension"]
        mirror_used: Optional[str] = None
        if recorded_chain:
            mirror_used = judge_used(sc)
            out["judge_model_used"] = mirror_used
            if sc in fb_b:
                out["judge_chain_note"] = fb_b[sc]
            if sc in af_b:
                out["auditor_fallback_note"] = af_b[sc]
        pj = (pb.get("judge_fields") or {}).get(sc) or {}
        if pj:
            row_used = pj.get(PRODUCER_JUDGE_USED_KEY)
            out.update(pj)
            if row_used:
                out["judge_model_used_source"] = "rows.jsonl (the producer's per-row judge provenance)"
                if mirror_used is not None and str(mirror_used) != str(row_used):
                    out["judge_model_used_mirror"] = mirror_used
            elif out.get("judge_model_used"):
                out["judge_model_used_source"] = "judge_chain_mirror"
        elif out.get("judge_model_used"):
            out["judge_model_used_source"] = "judge_chain_mirror"
        return out

    @staticmethod
    def _judge_marker_rollup(markers) -> Dict[str, Any]:
        """The per-item judge markers rolled up onto a judgment: the count of items per judge model (a mix stays a mix — never one of them),
        the items whose judge is unrecorded, the producer's re-judged / kept counts."""
        ms = list(markers)
        used = collections.Counter(str(m["judge_model_used"]) for m in ms if m.get("judge_model_used"))
        return {"judge_models_used": dict(sorted(used.items())), "n_items_judge_unrecorded": sum(1 for m in ms if not m.get("judge_model_used")),
                "n_items_rejudged": sum(1 for m in ms if m.get(PRODUCER_REJUDGED_KEY)), "n_items_rejudge_kept": sum(1 for m in ms if m.get(PRODUCER_REJUDGE_KEPT_KEY)),
                "producer_judge_fields_present": any(str(m.get("judge_model_used_source") or "").startswith("rows.jsonl") or m.get(PRODUCER_REJUDGED_KEY)
                                                     or m.get(PRODUCER_REJUDGE_KEPT_KEY) for m in ms)}

    def _emit_items(self, batch: RecordBatch, producer: Mapping[str, Any], now: str, rid: str, jid: str, b: str, pb: Mapping[str, Any], when: str,
                    markers: Mapping[str, Dict[str, Any]], pooled: bool = False) -> int:
        """One behaviour's item rows under a run: every scored scenario on `petri_score` and the two shared dimensions, every refused scenario
        as a NULL `judge_refused` item; the judge markers on each. Under the pooled run the same rows are filed again with the behaviour named.
        Returns the number of rows added."""
        n = 0
        for sc in sorted(pb["scored"]):
            cu = pb["created"].get(sc) or when
            it_extra: Dict[str, Any] = dict(markers.get(sc) or {"dimension": pb["dimension"]})
            if pooled:
                it_extra["behaviour"] = b
            batch.add("item_result", IC.item_result_payload(rid, f"{b}/{sc}", METRIC, pb["scored"][sc], None, cu, judgment_id=jid, judged_utc=cu, extra=it_extra), producer, now)
            n += 1
            for d in SHARED_DIMENSIONS:
                v = pb["shared"][d].get(sc)
                if v is not None:
                    batch.add("item_result", IC.item_result_payload(rid, f"{b}/{sc}", d, v, None, cu, judgment_id=jid, judged_utc=cu,
                                                                    extra=({"behaviour": b} if pooled else None)), producer, now)
                    n += 1
        for sc in sorted(pb["refused"]):
            rx: Dict[str, Any] = dict(markers.get(sc) or {})
            if pooled:
                rx["behaviour"] = b
            batch.add("item_result", IC.item_result_payload(rid, f"{b}/{sc}", METRIC, None, JUDGE_REFUSED_LABEL, when, judgment_id=jid, judged_utc=when, extra=rx), producer, now)
            n += 1
        return n

    def _emit_pooled(self, batch: RecordBatch, ctx: AdapterContext, rec: Dict[str, Any], bd: Mapping[str, Any], insts: Mapping[str, Mapping[str, Any]],
                     judges: Mapping[str, str], comp_id: str, rng, facts: Dict[str, Any], files: Mapping[str, Optional[Input]], cdir: str, ofacts: Dict[str, Any]) -> None:
        """The object's pooled run (v3): eval `petri_pooled` on the seed's pooled instrument, DERIVED from the six behaviour runs just filed — the
        value the mean of the six behaviour means, `n` the scored items, `n_items_planned` the bundle's 60, `n_abstentions` the judge refusals plus
        the withheld items, a stratified bootstrap; the judgment rolls the per-item judge markers up; every (behaviour, scenario) item is filed
        again under it. Refused by name when a behaviour run did not file, when the pooled instrument is absent, or when it disagrees with the
        manifest — the pooled run is never filed on a partial reading."""
        producer, now = ctx.producer, ctx.now
        seed_label, seed_index = str(bd["seed_label"]), int(bd["seed_index"])
        store_key_legacy, campaign_key_legacy, object_key, object_id = rec["store_key_legacy"], rec["campaign_key_legacy"], rec["object_key"], rec["object_id"]
        rkey = f"{seed_label}|{store_key_legacy}|{POOLED_EVAL}"
        missing = [b for b in BEHAVIOURS if b not in rec["runs"]]
        if missing:
            self.refusals.add(rkey, f"the pooled run is derived from all six behaviour runs; {len(missing)} did not file ({', '.join(missing)})", cdir)
            return
        inst = insts.get(POOLED_EVAL)
        judge_id = judges.get(POOLED_EVAL)
        if inst is None or judge_id is None:
            self.refusals.add(rkey, f"no instrument / judge for {POOLED_EVAL}; the pooled run is not minted without them", cdir)
            return
        inst_match, mism = self._instrument_match(inst, rec)
        if mism:
            self.refusals.add(rkey, f"the pooled instrument {str(inst['instrument_id'])[:12]}… disagrees with the store's manifest: {'; '.join(mism)}", cdir)
            return
        expected = rec.get("audited_expected")
        if expected != N_ITEMS_POOLED:
            self.refusals.add(rkey, f"results.json audited.expected {expected!r} is not the bundle's {N_ITEMS_POOLED} items; the pooled denominator is not guessed", cdir)
            return
        iid = str(inst["instrument_id"])
        rid = I.run_id(str(bd["campaign"]), str(bd["producer_key"]), object_key, POOLED_EVAL, iid)
        trial = int(bd["trial_of"](object_id))
        queued: Mapping[Tuple[str, str], Mapping[str, Any]] = bd.get("queued") or {}
        q = queued.get((campaign_key_legacy, POOLED_EVAL))
        if q is not None:
            bad = []
            if str(q["run_id"]) != rid:
                bad.append(f"run_id {q['run_id']} != derived {rid}")
            if str(q["object_id"]) != object_id:
                bad.append(f"object {q['object_id']} != resolved {object_id}")
            if str(q["producer_object_key_legacy"]) != object_key:
                bad.append(f"producer object key {q['producer_object_key_legacy']!r} != {object_key!r}")
            if LC.int_or_none(q.get("trial_index")) != trial or LC.int_or_none(q.get("seed_base")) != seed_index:
                bad.append(f"trial_index / seed_base ({q.get('trial_index')}, {q.get('seed_base')}) != ({trial}, {seed_index})")
            if bad:
                self.refusals.add(rkey, "the premint-queued pooled row disagrees with the derived identity: " + "; ".join(bad), cdir)
                return
            facts["premint"]["matched"] += 1
            ofacts["premint_rows"][POOLED_EVAL] = {"run_id": rid, "lifecycle_state": q.get("lifecycle_state")}
        else:
            facts["premint"]["runs_without_premint"].append(rid)
            ofacts["premint_rows"][POOLED_EVAL] = None
        per_beh = rec["per_beh"]
        when = rec["when"]
        n_scored = sum(pb["n_scored"] for pb in per_beh.values())
        n_refused = sum(pb["n_refused"] for pb in per_beh.values())
        n_unresolved = int(rec.get("n_unresolved") or 0)
        means = {b: float(per_beh[b]["mean"]) for b in BEHAVIOURS}
        value = mean_of_behaviour_means(means)
        jid = I.judgment_id(rid, judge_id)
        # -- the run: the behaviour run's payload shape (every column the six carry), re-pointed at the pooled identity and denominators
        template = rec["run_payloads"][BEHAVIOURS[0]]
        payload: Dict[str, Any] = json.loads(json.dumps(template))
        fc: Dict[str, Any] = payload["field_confidence_json"]
        payload.update({
            "run_id": rid, "eval_id": POOLED_EVAL, "item_set_id": inst.get("item_set_id"), "instrument_id": iid, "instrument_match": inst_match,
            "harness": inst.get("harness"), "harness_version": inst.get("harness_version") or "unverified", "n_shot": inst.get("n_shot"),
            "cot_prompting": inst.get("cot_prompting"), "scoring_rule": inst.get("scoring_rule"), "judge_id": judge_id, "trial_index": trial,
            "n_items_planned": N_ITEMS_POOLED, "n_items_generated": n_scored + n_refused + n_unresolved, "n_items_judged": n_scored,
            "token_cap_tokens": int(inst.get("token_cap_tokens") or 1024), "token_cap_ladder_json": inst.get("token_cap_ladder_json") or "[]",
            "reply_cap_tokens": inst.get("reply_cap_tokens"), "context_window_tokens": inst.get("context_window_tokens"),
            "thinking_mode": inst.get("thinking_mode") or "off", "serving_path": inst.get("serving_path") or "unverified", "decode_source": inst.get("decode_source") or "unverified",
            "notes": (f"Petri seed-{seed_index} audit of {campaign_key_legacy} (pooled): the mean of the six behaviour means {value:.4f} over {n_scored} of {N_ITEMS_POOLED} "
                      f"(behaviour, scenario) items scored, {n_refused} judge refusal(s) and {n_unresolved} withheld item(s) filed as abstentions; derived by the registry "
                      f"from the six behaviour runs' files; {bd['notes_tail']}"),
        })
        fc.update({
            "instrument_id": LC.confidence_entry("recorded" if facts["instruments"].get(POOLED_EVAL, {}).get("source") == "premint_row" else "derived",
                                                 (facts["instruments"].get(POOLED_EVAL, {}).get("source") or "minted") + f": the seed's pooled instrument ({str(bd['instrument_source'])})"),
            "instrument_match": LC.confidence_entry("derived", "the store's cap, decode numbers, seed rule, serving path, scenario bundle and auditor against the pooled instrument row"),
            "object_id": LC.confidence_entry("derived", f"jobplan.object_id_for over the roster's {bd['roster_selection']} selection and the pins / si_texts"
                                                        + ("; equal to the premint-queued pooled row's" if q is not None else "; no premint-queued row for the pooled eval")),
            "n_items_planned": LC.confidence_entry("recorded", "results.json audited.expected (the bundle's 60 (behaviour, scenario) items)"),
            "n_items_generated": LC.confidence_entry("derived", "scored + terminal judge refusals + withheld items over the six behaviours"),
            "n_items_judged": LC.confidence_entry("derived", "the scored (behaviour, scenario) items of rows.jsonl over the six behaviours"),
        })
        if payload["harness_version"] == "unverified":
            fc["harness_version"] = LC.unverified("the instrument row records no harness version")
        ex = payload["extra"]
        ex.update({"behaviour": None, "dimension": None, "pooled": True, "pooled_from_behaviour_runs": {b: rec["runs"][b]["run_id"] for b in BEHAVIOURS},
                   "mean_of_behaviour_means": value, "per_behaviour_means": means, "per_behaviour_n": {b: per_beh[b]["n_scored"] for b in BEHAVIOURS},
                   "per_behaviour_refused": {b: per_beh[b]["n_refused"] for b in BEHAVIOURS},
                   "judge_refused_scenarios": sorted(f"{b}/{sc}" for b in BEHAVIOURS for sc in per_beh[b]["refused"]), "n_unresolved": n_unresolved,
                   "premint": {"producer_key": str(bd["producer_key"]), "row": ("present" if q is not None else "absent"),
                               "queued_lifecycle_state": (q.get("lifecycle_state") if q is not None else None)},
                   "admission": dict(bd["admission"], premint_row=("present" if q is not None else "absent")), "instrument_mismatches": mism,
                   "derivation": "registry-derived from the six behaviour runs' own files (no producer pooled artifact is read)"})
        if str(bd.get("binding") or "") == "premint" or bd.get("premint_family"):
            ex["premint_family"] = bd.get("premint_family") or str(bd["producer_key"])
        batch.add("run", payload, producer, now)
        ev_uri = LC.rel_home(files["results.json"].uri)
        batch.add("run_state", {"run_id": rid, "state": "generated", "event_utc": when, "evidence_kind": "completion_artifact", "evidence_uri": ev_uri,
                                "note": f"audit complete: {n_scored + n_refused} of {N_ITEMS_POOLED} (behaviour, scenario) items ran ({n_refused} judge refusal(s), {n_unresolved} withheld)"},
                  producer, now, observe=False)
        batch.add("run_state", {"run_id": rid, "state": "judged", "event_utc": when, "evidence_kind": "completion_artifact", "evidence_uri": ev_uri,
                                "note": f"{n_scored} (behaviour, scenario) items scored by the audit's judge; {n_refused} refused to score"}, producer, now, observe=False)
        # -- the judgment: the six behaviours' markers rolled up (a judge mix stays a mix)
        chain = rec.get("judge_chain") or {}
        all_markers = [m for b in BEHAVIOURS for m in (rec.get("markers") or {}).get(b, {}).values()]
        fb_all = {b: sorted(sc for (bb, sc) in (chain.get("fallback_judged") or {}) if bb == b) for b in BEHAVIOURS} if chain.get("state") == "recorded" else None
        af_all = {b: sorted(sc for (bb, sc) in (chain.get("auditor_fallback") or {}) if bb == b) for b in BEHAVIOURS} if chain.get("state") == "recorded" else None
        kinds = collections.Counter(refusal_kind(x.get("reason")) for b in BEHAVIOURS for x in per_beh[b]["refused"].values())
        jpay = LC.judgment_payload(rid, judge_id, n_scored, when, LC.rel_home(cdir), api_pool=bd.get("api_pool"),
                                   extra={"judge_seed_id": JUDGE_SEED, "judge_model": str(bd.get("judge_model") or "anthropic/claude-opus-5"),
                                          "judge_fallback_models": bd.get("judge_fallback_models"), "api_pools": bd.get("api_pools"), "pooled": True,
                                          "behaviour_judgment_ids": {b: rec["runs"][b]["judgment_id"] for b in BEHAVIOURS},
                                          "refusal_reasons": sorted({str(x.get("reason")) for b in BEHAVIOURS for x in per_beh[b]["refused"].values()}),
                                          "refusal_kinds": dict(kinds), "auditor_chain": rec["results"].get("auditor_chain"), "auditor_refused": rec["results"].get("auditor_refused"),
                                          "judge_chain_state": chain.get("state", "not_applicable"), "judge_chain_note": chain.get("why"),
                                          "judge_fallback_count": (sum(len(v) for v in fb_all.values()) if fb_all is not None else None),
                                          "judge_fallback_scenarios": ({b: v for b, v in fb_all.items() if v} if fb_all is not None else None),
                                          "judge_fallback_model": (list(bd.get("judge_fallback_models") or [None])[0] if fb_all and any(fb_all.values()) else None),
                                          "auditor_fallback_count": (sum(len(v) for v in af_all.values()) if af_all is not None else None),
                                          "auditor_fallback_scenarios": ({b: v for b, v in af_all.items() if v} if af_all is not None else None),
                                          "judge_chain_mirror": ({"uri": chain.get("uri"), "sha256": chain.get("sha256")} if chain.get("state") == "recorded" else None),
                                          **self._judge_marker_rollup(all_markers)})
        jpay.update(n_judge_refusals=n_refused, n_errors=0, producer_session=PRODUCER_SESSION, grading_code_sha=(payload["code_sha"] if payload["code_sha"] != "unverified" else None),
                    started_utc=rec["started_utc"])
        batch.add("judgment", jpay, producer, now)
        # -- the metric: the mean of the six behaviour means; a stratified bootstrap beside it
        lo, hi = pooled_bootstrap({b: list(per_beh[b]["scored"].values()) for b in BEHAVIOURS}, rng)
        all_scores = [v for b in BEHAVIOURS for v in per_beh[b]["scored"].values()]
        shared_means = {d: ((sum(v for b in BEHAVIOURS for v in per_beh[b]["shared"][d].values()) / n_scored) if n_scored else None) for d in SHARED_DIMENSIONS}
        batch.add("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": METRIC, "threshold": -1.0, "computation_id": comp_id,
                                 "producer_metric_spelling": "mean of behaviors.<b>.mean over the six behaviours (registry-derived)", "value": value, "k": None, "n": n_scored,
                                 "ci_lo": lo, "ci_hi": hi,
                                 "ci_method": (f"stratified percentile bootstrap: scenarios resampled within each of the six behaviours, the mean of behaviour means recomputed "
                                               f"({N_BOOT} draws, seed {BOOT_SEED}; registry)" if lo is not None else "none filed"),
                                 "n_abstentions": n_refused + n_unresolved, "n_incomplete": n_unresolved, "computed_by": "registry", "computed_utc": when, "source_uri": ev_uri,
                                 "source_producers_json": [{"producer": "results.json", "uri": ev_uri, "sha256": files["results.json"].sha256, "field_group": "behaviour means, counts, judge refusals"},
                                                           {"producer": "rows.jsonl", "uri": LC.rel_home(files["rows.jsonl"].uri), "sha256": files["rows.jsonl"].sha256, "field_group": "per-scenario scores"}],
                                 "extra": {"denominator": "the mean of the six behaviour means, each over its scored scenarios (judge refusals and withheld items excluded, never imputed)",
                                           "per_behaviour_means": means, "per_behaviour_n": {b: per_beh[b]["n_scored"] for b in BEHAVIOURS},
                                           "per_behaviour_refused": {b: per_beh[b]["n_refused"] for b in BEHAVIOURS}, "n_judge_refused": n_refused, "n_withheld": n_unresolved,
                                           "mean_over_all_scored_items": (sum(all_scores) / len(all_scores)) if all_scores else None,
                                           "sd_over_all_scored_items": (float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else None),
                                           "producer_audited": {"expected": expected, "items": rec.get("audited_items")},
                                           "eval_awareness_mean": shared_means["eval_awareness"], "scenario_realism_mean": shared_means["scenario_realism"],
                                           "sign": "higher = MORE of the flagged behaviour (judge score 1-10)"}}, producer, now)
        facts["metrics"] += 1
        # -- every (behaviour, scenario) item again under the pooled run
        n_items = 0
        for b in BEHAVIOURS:
            n_items += self._emit_items(batch, producer, now, rid, jid, b, per_beh[b], when, (rec.get("markers") or {}).get(b, {}), pooled=True)
        facts["items"] += n_items
        rec["runs"][POOLED_EVAL] = {"run_id": rid, "judgment_id": jid, "eval_id": POOLED_EVAL}
        ofacts["runs"][POOLED_EVAL] = {"run_id": rid, "n_scored": n_scored, "n_refused": n_refused, "mean": value, "instrument_match": inst_match}
        facts["judged_runs"] += 1
        facts["pooled_runs"] += 1
        facts["pooled"][store_key_legacy] = {"run_id": rid, "value": value, "per_behaviour_means": means, "n": n_scored, "n_abstentions": n_refused + n_unresolved,
                                             "n_items": n_items, "judge_models_used": jpay["extra"]["judge_models_used"]}

    def _emit_contrasts(self, batch: RecordBatch, ctx: AdapterContext, rec: Mapping[str, Any], anchor: Mapping[str, Any], comp_id: str, rng, facts: Dict[str, Any],
                        seed_label: str = SEED_LABEL) -> None:
        """The steered object's contrast against the store's own `none` per behaviour, PAIRED ON SCENARIO ID over the scenarios scored in both."""
        producer, now = ctx.producer, ctx.now
        store_key_legacy = rec["store_key_legacy"]
        for b in BEHAVIOURS:
            s_run, a_run = rec["runs"].get(b), anchor["runs"].get(b)
            if s_run is None or a_run is None:
                continue
            ct = paired_scenario_contrast(rec["per_beh"][b]["scored"], anchor["per_beh"][b]["scored"], rng)
            pkey = f"{store_key_legacy}/{b}"
            facts["pairing"][pkey] = dict(ct["pairing"], n_pairs=ct["n_pairs"])
            if not ct["n_pairs"]:
                self.refusals.add(f"{seed_label}|{store_key_legacy}|{s_run['eval_id']}|contrast", "no scenario scored in both the steered and the `none` audit; no paired contrast is formed", "")
                continue
            when = rec["when"] if rec["when"] >= anchor["when"] else anchor["when"]
            # the same standup when both audits were completed on one tree (the s1 store: one attempt; a cluster-run seed: its main / re-run passes serve
            # one tree); an audit completed on a per-object fix tree pairs against the seed's `none` as a SIBLING standup
            same_tree = rec.get("tree") is None or anchor.get("tree") is None or rec.get("tree") == anchor.get("tree")
            batch.add("run_contrast", {"run_id": s_run["run_id"], "anchor_run_id": a_run["run_id"], "judgment_id": s_run["judgment_id"], "metric_id": METRIC, "threshold": -1.0,
                                       "computation_id": comp_id, "anchor_judgment_id": a_run["judgment_id"], "anchor_kind": ("same_standup_none" if same_tree else "sibling_none"),
                                       "delta": ct["delta"], "delta_lo": ct["delta_lo"], "delta_hi": ct["delta_hi"], "delta_p": ct["delta_p"],
                                       "delta_method": (f"paired mean difference of the judge score over {ct['n_pairs']} scenarios scored in BOTH the steered and the `none` audit "
                                                        f"(paired on scenario id; {ct['pairing']['only_steered']} scored only steered, {ct['pairing']['only_anchor']} only in `none`, "
                                                        f"excluded); percentile bootstrap {N_BOOT} draws (seed {BOOT_SEED}); exact two-sided sign test (registry)"),
                                       "effect_size": ct["d_z"], "effect_kind": ("dz" if ct["d_z"] is not None else None), "n_pairs": ct["n_pairs"], "computed_by": "registry",
                                       "computed_utc": when,
                                       "extra": {"pairing": ct["pairing"], "shared_scenarios": ct["shared_scenarios"], "sign": SIGN_TEXT, "sd_diff": ct["sd_diff"],
                                                 "wilcoxon_p": ct["wilcoxon_p"], "mean_steered": ct["mean_steered"], "mean_anchor": ct["mean_anchor"],
                                                 "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED, "anchor_store_object_key_legacy": "none", "steered_tree_sha": rec.get("tree"),
                                                 "anchor_tree_sha": anchor.get("tree"), "unpaired_delta": "not filed: the refusals are "
                                                 "structured by object, so an unpaired difference would compare different scenario sets"}}, producer, now)
            facts["contrasts"] += 1
        # -- the pooled contrast (v3): the steered pooled run against the seed's `none` pooled run, paired on <behaviour>/<scenario> over the items
        # scored in both; the same anchor rule as the behaviour contrasts
        s_run, a_run = rec["runs"].get(POOLED_EVAL), anchor["runs"].get(POOLED_EVAL)
        if s_run is None or a_run is None:
            if s_run is not None:
                self.refusals.add(f"{seed_label}|{store_key_legacy}|{POOLED_EVAL}|contrast", "the `none` pooled run was not filed; no paired pooled contrast is formed", "")
            return
        pooled_s = {f"{b}/{sc}": v for b in BEHAVIOURS for sc, v in rec["per_beh"][b]["scored"].items()}
        pooled_a = {f"{b}/{sc}": v for b in BEHAVIOURS for sc, v in anchor["per_beh"][b]["scored"].items()}
        ct = paired_scenario_contrast(pooled_s, pooled_a, rng)
        facts["pairing"][f"{store_key_legacy}/{POOLED_EVAL}"] = dict(ct["pairing"], n_pairs=ct["n_pairs"])
        if not ct["n_pairs"]:
            self.refusals.add(f"{seed_label}|{store_key_legacy}|{POOLED_EVAL}|contrast", "no (behaviour, scenario) item scored in both the steered and the `none` audit; no paired pooled contrast is formed", "")
            return
        when = rec["when"] if rec["when"] >= anchor["when"] else anchor["when"]
        same_tree = rec.get("tree") is None or anchor.get("tree") is None or rec.get("tree") == anchor.get("tree")
        per_b_pairs = {b: sum(1 for k in ct["shared_scenarios"] if k.split("/", 1)[0] == b) for b in BEHAVIOURS}
        s_means = {b: rec["per_beh"][b]["mean"] for b in BEHAVIOURS}
        a_means = {b: anchor["per_beh"][b]["mean"] for b in BEHAVIOURS}
        batch.add("run_contrast", {"run_id": s_run["run_id"], "anchor_run_id": a_run["run_id"], "judgment_id": s_run["judgment_id"], "metric_id": METRIC, "threshold": -1.0,
                                   "computation_id": comp_id, "anchor_judgment_id": a_run["judgment_id"], "anchor_kind": ("same_standup_none" if same_tree else "sibling_none"),
                                   "delta": ct["delta"], "delta_lo": ct["delta_lo"], "delta_hi": ct["delta_hi"], "delta_p": ct["delta_p"],
                                   "delta_method": (f"paired mean difference of the judge score over {ct['n_pairs']} (behaviour, scenario) items scored in BOTH the steered and the `none` "
                                                    f"audit (paired on <behaviour>/<scenario>; {ct['pairing']['only_steered']} scored only steered, {ct['pairing']['only_anchor']} only in "
                                                    f"`none`, excluded); percentile bootstrap {N_BOOT} draws (seed {BOOT_SEED}); exact two-sided sign test (registry)"),
                                   "effect_size": ct["d_z"], "effect_kind": ("dz" if ct["d_z"] is not None else None), "n_pairs": ct["n_pairs"], "computed_by": "registry",
                                   "computed_utc": when,
                                   "extra": {"pairing": ct["pairing"], "shared_items": ct["shared_scenarios"], "n_pairs_per_behaviour": per_b_pairs, "sign": SIGN_TEXT,
                                             "sd_diff": ct["sd_diff"], "wilcoxon_p": ct["wilcoxon_p"], "mean_steered": ct["mean_steered"], "mean_anchor": ct["mean_anchor"],
                                             "mean_of_behaviour_means_steered": mean_of_behaviour_means(s_means), "mean_of_behaviour_means_anchor": mean_of_behaviour_means(a_means),
                                             "delta_of_behaviour_means": mean_of_behaviour_means(s_means) - mean_of_behaviour_means(a_means),
                                             "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED, "anchor_store_object_key_legacy": "none", "steered_tree_sha": rec.get("tree"),
                                             "anchor_tree_sha": anchor.get("tree"), "pooled": True,
                                             "unpaired_delta": "not filed: the refusals are structured by object, so an unpaired difference would compare different item sets; "
                                                               "the difference of the two pooled values rides extra.delta_of_behaviour_means"}}, producer, now)
        facts["contrasts"] += 1


def factory() -> SeedrepPetriAdapter:
    return SeedrepPetriAdapter()


register(NAME, factory)
