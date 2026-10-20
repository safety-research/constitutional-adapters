"""scripts/modal/seedrep_carry_over.py -- completed attempts are CARRIED across wave trees; nothing that completed once is re-run.

The accepted-attempt map is keyed `<wave>|wave_uuid|leg|view|stage|shard` (`seedrep_launch_plan.accepted_key`): two waves never share a key and
nothing admits a prior wave's attempt, so a W(n+1) forced by one defect re-ran every cell, and a proving tree's FULL-shape rows (complete
production-shaped stores) were re-run although their producer files did not change. This module is the carry-over ADMISSION the launcher
(`seedrep_served.launch_production`, CLI `--carry-over <prior RS or plan>[,<...>]`) and the certificate (`seedrep_launch_plan.certificate_problems`)
share. Stdlib only; imports its sibling `seedrep_launch_plan` for the key / root / evidence spellings so ONE reader judges a carried root exactly
as it judges a fresh one; every Modal / git / filesystem access is INJECTED (`read_text`, `accepted_get_for_dict`, `git`, `readiness_problems`), so
the evaluation is pure and testable on synthetic roots.

A candidate attempt of a prior tree is CARRIED into THIS plan's row iff EVERY clause holds -- each recorded by name with its facts:

  (a) ``row_identity``   -- the prior row spells identically to this plan's row once the tree and wave identity is replaced by
                            tokens: standup argv, children argv (decode, budget flags, caps), geometry, objects / pins, readiness expectations,
                            adapters, route set, ladder / rung consumer, `modal` function -- with the `smoke` block ABSENT on both. The admitted
                            smoke sources are the explicit :data:`S5_FULL_CLAUSE` (the reference smoke: `smoke.budget.kind == full_shape`) and
                            :data:`S3_FULL_CLAUSE` (a smoke whose shape IS the production row's -- an eight-leg Petri row) -- EVERY production
                            child present, no budget flag on any child, compared under the production standup id. THE FULL LIST of identity
                            fields clause (a) SETS ASIDE for a smoke source: (1) the `smoke` block itself; (2) the `smoke-<S>__` prefix of
                            `standup_id` (compared under `smoke.production_standup_id`); (3) `adapters[*].expected_reused` (the smoke's §8
                            adapter-role rewrite; the served hashes are held to THIS plan by clause (d)); (4) `modal.timeout_s` (the smoke's
                            container bound, not the row's shape); (5) the tree / wave identity, tokenised on BOTH sides -- each identity field
                            to its placeholder, the production wave -> {WAVE}, and for a full-shape smoke its smoke wave -> {WAVE} too (the
                            smoke wave stood in for production: the six-axis keys, the ladder root and the rung consumer's --wave-uuid carry
                            it). Everything else -- argv, geometry, objects, route set, readiness, children (count, arm, cell, s, argv incl.
                            caps and budget flags) -- is compared byte for byte. Budgeted smokes -- every smoke id outside
                            :data:`FULL_SHAPE_SMOKE_SOURCES` -- are refused by name, as is an `s5_probe` row.
  (b) ``producer_set``   -- the leg's producer FILE SET is byte-identical between the two tree shas: derived from the row itself (the standup
                            script and every tree file its argv names, every child argv's script and config, the rung consumer's, the modules the
                            standup starts -- the routing proxy and, for GSM8K, the usage ledger -- the audit tools the row's evidence names as
                            writers, and the in-container helpers of the leg), then `git ls-tree` blobs at BOTH shas (a path absent at either sha
                            refuses) and `git diff --quiet <prev> <this> -- <files>` (recorded with its rc). The container app module
                            `scripts/modal/seedrep_served.py` and the plan module are READERS here: their blobs are RECORDED under
                            `container_modules` (informational, not gating) -- what they produced is re-judged by clauses (c) and (d) under
                            THIS tree's readers, and the one thing they produce that a rung consumes (the ladder queue) is not carried (clause (e)).
  (c) ``complete``       -- the attempt is COMPLETE under THIS tree's readers: `RESULT` DONE rc 0 naming the attempt, `COMPLETE.json` /
                            `modal_evidence.json` naming it, readiness problems [], the object verification ok, every child rc 0, producer
                            evidence bound (`lp._smoke_row_facts`), and the carry-source check set re-derived on the root through `lp._smoke_check`
                            (`SMOKE_CHECKS[S3_SMOKE_ID]`: resolved sampler AGREES, the request-evidence audit AGREES per child with its
                            qualification, the determinism spot-check per table leg, the pool endpoint snapshot on a pool-pinned row) plus the
                            ladder inventory (`ladder_enqueued.json` complete) on a producer row. The producer bindings are held to the PRIOR
                            tree's sha (the tree the evidence was produced at -- byte-identical to this one by (b)).
  (d) ``engine_identity`` -- the engine the attempt served IS the one this plan's row describes: THIS tree's readiness contract
                            (`seedrep_served.readiness_problems`, injected) over the attempt's READY records (`exp/serve/endpoint.json`,
                            `cells_served.json`, `serving_attestation.json`) against THIS plan's row -- geometry (L, S, kernel adjustment, cudagraph
                            surface), thinking class, cells served, object pins, adapter reuse -- and the served adapter hashes (`exp/serve/cells.json`)
                            against THIS plan's `adapter_hash_set` (`lp._smoke_check("adapter_hashes_banked")`).
  (e) ``ladder_queue``   -- a PRODUCER row (`ladder.cap_next` set) whose consumer rung is a row of THIS plan carries ONLY through the queue REPLAY:
                            its at-cap queue lives in the prior tree and wave's ladder root, which this wave's `rung_redraw.py consume` never
                            reads. :func:`replay_ladder_queue` re-reads the prior attempt's `queue_inventory.json`, re-validates EVERY listed
                            queue file (sha vs the inventory, the six-axis record check incl. `check_redraw_sampling`, the producer binding)
                            against the PRIOR spec, re-keys ONLY the tree identity and `wave_uuid` to THIS wave, validates again against THIS
                            spec, writes the attempt-unique files under THIS wave's ladder root (identical bytes already there skip; different
                            bytes refuse; nothing is written before every record validated -- never a partial replay), writes the re-keyed
                            inventory + a carried-producer stub under `<ladder root>/carried/<shard>__<aid>/`, accepts under THIS wave's key
                            (`accept_attempt`), writes the producer's CLOSED marker and runs `reconcile_queue` with a carry-aware root resolver
                            so `CLOSED.json` lands when every declared producer is closed. Recorded under `clauses.ladder_queue.replay`
                            (`{n_records, source_keys, target_keys, closed, ...}`). Without a `replay` callable (a pure evaluation) such a row is
                            refused by name; a row whose rungs are not in this plan, a rung, a ceiling and every multi-turn / Petri row pass
                            without one.

  Petri: the `petri_fallback_chain` admitted kind covers (i) a `*_fallback_models` list membership change in a manifest and (ii)
  the retired-leg code removals in :data:`D111_ADMITTED_FILES`, PINNED by content hash of every changed hunk (:data:`D111_ADMITTED_HUNKS`; an unrelated
  change to those files is a hunk outside the pin and is NOT admitted). A Petri row with an admitted chain difference additionally needs the
  SIDECAR PROOF (:func:`petri_sidecar_proof`): over the carried store's cells, ZERO uses of the removed auditor leg (the `results.json` chain tally
  and every `petri_auditor_refused.json` record's used legs), recorded under `clauses.producer_set.petri_sidecar_proof`.

  The determinism spot-check ABSENT from a carried root refuses clause (c) unless an explicit, named operator WAIVER
  (`criterion3_waivers[standup_id] = {by, reason}`) is given: recorded on the entry as `criterion3_waiver` and in the check as
  `not_spot_checked (waived)`; the certificate refuses a waived entry whose waiver is missing. Default absent -> refused.

  Every evidence file clauses (c) / (d) read is bound by digest on the entry (`evidence_inventory`); :func:`carry_over_problems` re-reads every
  one (absent or changed -> refused), so a source removed after evaluation cannot pass certification on the recorded flags alone.

A CARRIED attempt enters THIS wave's accepted Dict under THIS wave's key (the plain attempt id, so every existing reader works), with its
provenance under the companion key :func:`carried_provenance_key` and in `RS/carry_over.json` (:data:`CARRY_OVER_FILE`); `select_launch_rows`
skips it by name (`CARRIED <standup_id> from <W7>/<attempt>`), so it is never spawned; the certificate carries the whole record
(`carry_over`) and :func:`carry_over_problems` refuses a carried entry whose provenance or clause record is missing, whose root no longer reads
COMPLETE, or whose key the live accepted Dict no longer names. A row failing any clause is RE-RUN, the failing clause named under `refused`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedrep_launch_plan as lp  # noqa: E402

CARRY_OVER_SCHEMA = "seedrep_carry_over/1"
CARRY_OVER_FILE = "carry_over.json"
#: the five clauses, in the order they are evaluated; every carried entry carries all five `ok: true`
CLAUSES: Tuple[str, ...] = ("row_identity", "producer_set", "complete", "engine_identity", "ladder_queue")
#: fj3: the queue replay's record schema, the carried-producer stub the reconciler finds under `<ladder root>/carried/<shard>__<aid>/COMPLETE.json`
REPLAY_SCHEMA = "seedrep_carry_replay/1"
CARRIED_PRODUCER_SCHEMA = "seedrep_carried_producer/1"
CARRIED_DIR = "carried"
REPLAY_SYNTHESISED_BY = "carry_over_replay"
#: the determinism waiver is an explicit, SCOPED operator document `{by, at, scope: {W, leg, view, legs}, reason, decision}`
#: (`--criterion3-waiver <json | file>`), copied verbatim into the carry record and the certificate; never a default, never inferred
CRITERION3_WAIVER_KEYS: Tuple[str, ...] = ("by", "at", "scope", "reason", "decision")
CRITERION3_SCOPE_KEYS: Tuple[str, ...] = ("W", "leg", "view", "legs")
DETERMINISM_WAIVED = "not_spot_checked (waived)"
#: the determinism spot-check (byte-identical replay) is NOT APPLICABLE to a row whose decode is SAMPLED (the vendor preset, temperature > 0: vLLM
#: sampling under continuous batching is not bit-reproducible; the row's reproducibility guarantee is its pinned decode identity, which the
#: resolved_sampler check already holds). A PRESENT record with reproduces=False on a sampled row is recorded as this QUALIFICATION on the carried
#: entry (surfaced in the certificate's carry block), never a refusal; a GREEDY row (temperature 0) with reproduces=False refuses as before.
CRITERION3_NOT_APPLICABLE_SAMPLED = lp.DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED      # "not_applicable_sampled_decode" -- ONE spelling, the plan's (fg1 / F-AW)
CRITERION3_RULING = "D180"
#: fj3: the retired Petri auditor leg as the chain records spell it -- `auditor_fallback.auditor_label` -> `qwen-abl`, the model id
PETRI_REMOVED_LEG_LABELS: Tuple[str, ...] = ("qwen-abl",)
PETRI_REMOVED_LEG_MODELS: Tuple[str, ...] = ("openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated",)
PETRI_RESULTS_JSON = "results.json"
PETRI_REFUSED_LEDGER = "petri_auditor_refused.json"
#: fj5: the JUDGE-refused ledger -- a judge-refused conversation is EXCLUDED from the `auditor_chain` tally by the summary
#: producer (`petri_runner._behavior_summaries`) and keeps its auditor-chain provenance under `behaviors[*].judge_refused_items` + here
PETRI_JUDGE_REFUSED_LEDGER = "petri_judge_refused.json"
#: the chain-tally buckets that mean the removed third leg WAS reached (`auditor_fallback.CHAIN_KINDS`): continued by it, or every configured leg
#: exhausted (the third leg was tried) -- pinned equal to the auditor module's spelling by test
PETRI_REMOVED_LEG_BUCKETS: Tuple[str, ...] = ("continued_by_qwen", "auditor_refused_all_legs")
#: the explicit, named clause under which a proving tree's S5 FULL-shape row may carry into the production row (admitted when the row
#: identity is byte-equal apart from the `smoke` block and the row ran the FULL shape)
S5_FULL_CLAUSE = "s5_full_row"
S5_SMOKE_ID = lp.S5_WINDOW_SMOKE_ID
#: a smoke row is a carry source too -- iff its shape IS the production row's: every production child
#: present, NO child carrying a budget flag (a bon / misalign smoke row carries `--limit-goals` / `--limit-items` and stays refused by name),
#: same objects / evals / argv modulo the admitted hunks -- so the eight Petri legs accepted under the smoke wave's Dict carry
S3_FULL_CLAUSE = "s3_full_row"
S3_SMOKE_ID = "S3"
#: the smoke ids admissible as carry sources, with the `carried_from.source` each is recorded under
FULL_SHAPE_SMOKE_SOURCES: Dict[str, str] = {S5_SMOKE_ID: "s5_smoke", S3_SMOKE_ID: "s3_smoke"}
#: a completed, accepted FULL-shape smoke row on the CURRENT tree (tree identity equal to the plan's; the smoke wave's accepted Dict) IS the
#: production row's canonical attempt -- admitted through the same S5_FULL_CLAUSE machinery, recorded under this source, never re-run
CURRENT_S5_SOURCE = "current_s5_full"
REAUDIT_NOT_REQUIRED_SAME_TREE = "not_required_same_tree"
SMOKE_ID_OF_SOURCE: Dict[str, str] = {**{v: k for k, v in FULL_SHAPE_SMOKE_SOURCES.items()}, CURRENT_S5_SOURCE: S5_SMOKE_ID}
#: the order carry sources are tried: the prior production acceptance, the CURRENT tree's S5 FULL, then the prior S5 / S3 smokes
SOURCE_RANK: Dict[str, int] = {"production": 0, CURRENT_S5_SOURCE: 1, "s5_smoke": 2, "s3_smoke": 3}
FULL_CLAUSE_OF_SMOKE: Dict[str, str] = {S5_SMOKE_ID: S5_FULL_CLAUSE, S3_SMOKE_ID: S3_FULL_CLAUSE}
#: the S3 check set re-derived on a carried root (clause (c)); `adapter_hashes_banked` is clause (d)'s hash check against THIS plan
COMPLETE_CHECKS: Tuple[str, ...] = tuple(lp.SMOKE_CHECKS["S3"])
#: the id `accepted_get_for` resolves to THIS wave's (production) accepted Dict -- the smoke getters are keyed by smoke id; this one never collides
PRODUCTION_ACCEPTED_ID = "production"
#: the prior tree's plan, in preference order: the FINAL plan a production wave ran; the DRAFT a proving tree's smokes ran against
PRIOR_PLAN_FILES: Tuple[str, ...] = ("launch_plan_final.json", "launch_plan.json", "launch_plan_draft.json")
CONTAINER_TREE_PREFIX = "/workspace/seedrep/si-steering-eval-repo/"
#: the modules the standup process starts in-container beside the engine, on the tree at W (the proxy is part of `mt_serving_up.py`'s
#: bring-up; the usage ledger is the GSM8K children's `python -m sisteer.capeval.serving.usage_ledger`)
PROXY_REL = "src/sisteer/capeval/serving/proxy.py"
LEDGER_REL = "src/sisteer/capeval/serving/usage_ledger.py"
#: in-container helpers of a leg that write what the row's evidence reads (the misalign tailer's message builder runs under the driver venv)
LEG_HELPERS: Dict[str, Tuple[str, ...]] = {"misalign": ("scripts/modal/seedrep_ladder_helpers.py", "src/sisteer/capeval/misalign/runner.py"),
                                            # the Petri DRIVE (grading-host side) and the auditor chain ARE the eval's identity for a Petri row (the retired leg changed both)
                                            "petri": ("scripts/modal/seedrep_petri_drive.py", "src/sisteer/capeval/petri/auditor_fallback.py")}
IMPORT_DEPTH = 1                # clause (b) walks the DIRECT imports of every producer script (ast over `git show <sha>:<path>`), recorded as such
#: clause (b)'s producer set is the code that SHAPES THE STORE'S ROWS (items, prompts, decode, target serving, the
#: driver's sampling / generation path) -- NOT pre-run probes, certification checks or evidence writers, which clause (c) re-derives under THIS
#: tree's readers. These are EXCLUDED from gating by name and RECORDED (`non_eval_modules`) when they differ. NOTE the granularity: a module that
#: holds BOTH a probe and the generation path (`scripts/served_capability_run.py`: the render probe AND the GSM8K / controls driver) is a
#: producer and stays gating -- a file-level byte check cannot carve the probe out of the driver; the record names the file so a rule can.
NON_EVAL_MODULES: Tuple[str, ...] = (
    "scripts/sdfbase/record_resolved_sampler.py",            # evidence writer: the resolved-sampler recorder (clause (c) re-reads its record)
    "scripts/sdfbase/seedrep/request_evidence_audit.py",     # evidence writer: the request-evidence audit (clause (c))
    "scripts/sdfbase/seedrep/determinism_check.py",          # post-run determinism spot-check (clause (c))
    "scripts/modal/seedrep_s5_measure.py",                   # evidence writer: the window measurement
    "scripts/sdfbase/seedrep/audit_common.py",               # the audits' shared reader
    "src/sisteer/capeval/tinycap/contracts.py",              # render CERTIFICATION (fd3: the OFF-header reference) -- a check, not the eval
    # NOT here: `src/sisteer/registry/hooks.py` -- the driver PROBES `StoreRecorder.run_payload()` for the row constants it
    # reads and lifts them into the persisted cell manifest (`served_capability_run._manifest_row_constant_lift`), so the recorder shapes store
    # bytes; it stays GATING, its reviewed emission-only changes admitted through exact pinned hunks (the hooks.py entry in D136_ADMITTED_HUNKS)
)
#: a Petri run's auditor / judge FALLBACK CHAIN is not the eval's identity (target cell, scenarios,
#: behaviours, judge model, prompts, decode are). A producer file whose ONLY difference between the trees is the membership of a
#: `*_fallback_models` list is an ADMITTED difference, recorded line by line with the old / new list; the store's run key is never re-stamped.
#: A Petri ROW additionally needs the sidecar proof that the removed / added legs were never reached in the carried store -- NOT implemented here
#: (no sidecar reader in this module; named gap): a Petri row with an admitted chain difference is refused by name until that reader lands.
ADMITTED_KIND_CHAIN = "petri_fallback_chain"
ADMITTED_KIND_REGISTRY = "registry_emission"          # a store-byte-neutral recorder call
ADMITTED_KIND_PLUMBING = "grading_host_plumbing"             # the grading host's relay-port plumbing
#: the two producer files whose difference between the two trees IS the retired third leg's code. Every changed hunk (a zero-context unified
#: diff, hashed over its `-` / `+` body lines) must be one of the pinned hashes below; the labels say what each hunk is. Computed from
#: `git diff 56aff576e 8024980d2 -- <file>`; a future change to either file adds a hunk outside the pin and is REFUSED.
D111_ADMITTED_FILES: Tuple[str, ...] = ("scripts/modal/seedrep_petri_drive.py", "src/sisteer/capeval/petri/auditor_fallback.py")
D111_ADMITTED_HUNKS: Dict[str, Dict[str, str]] = {
    "scripts/modal/seedrep_petri_drive.py": {
        "38d248527b6b3bbc49abd73d5eaa62efb563940e5b20032b3dac748ee0cf8371": "REMOVED_QWEN_LEG_MSG added (+5: the refusal text for the retired flags)",
        # the SAME +5 block under the other blank-line boundary difflib picks once the relay-port hunks sit in the same file:
        # a zero-context hunk's boundary can shift by a blank line without any new change -- pinned as the same hunk, second alignment
        "4d9f2798f6c794778e41e43ce0ed25c453f1d1aa3ed1f7ff5b9415cab36a1b26": "REMOVED_QWEN_LEG_MSG added (+5; blank-line boundary alignment beside D137's hunks)",
        "42800e710bd4e8e72810e128cb79122ba5f4ccadcb1671c560ab868545505390": "argparse (-3 +5): --qwen-auditor-pool / --qwen-auditor-url / --allow-unreachable-qwen-auditor suppressed, default None",
        "83ca2c93b8ffafa1cf916c2651ccc5070027a1abdd98cc3823d59367d20dfcfa": "main() (+3): a command line naming the retired flags is refused by name (DriveRefused REMOVED_QWEN_LEG_MSG)",
        "0f60a06b6e1f484079438e8a8d00e4b7344fae62a22789acbb541d3e4240bb87": "main() (-18): the third-leg pool record / URL env + the /models reachability probe + its DriveRefused removed",
        "cfe012878da0346fb9166dbb76e7352c7f53c1e3b1180b0cbb7fcbfe4d3f4c93": "the drive record (-1 +1): qwen_auditor_url None, reachable False, auditor_chain 'sonnet->grok (D111: the Qwen third leg is retired)'",
    },
    "src/sisteer/capeval/petri/auditor_fallback.py": {
        "f8a2921973e39c6aee3958ffc23e5d6d3b34cacae38b75f0c9aa094385c9ab4c": "DEFAULT_FALLBACK_MODELS (-1 +7) drops DEFAULT_QWEN_MODEL; the comment block records the 231-cell / 21,399-row zero-use finding",
    },
    # manifest.py's ONE retired-leg hunk (the Qwen item of the Python default `auditor_fallback_models` list); alone it is also the
    # chain-list kind, but beside the judge-chain hunks it must be pinned like them
    "src/sisteer/capeval/manifest.py": {
        "590e57ea32e00dbb31114092ea655eaf099aa649ee4bf3b11f36fb2196d57ec6": "PetriSpec.auditor_fallback_models default (-1) drops the abliterated-Qwen item",
    },
}
#: the pin was computed from BOTH carry sources -- 56aff576e and c65b47ba4 (the Petri carry source): the three
#: gating files are byte-identical at the two (pinned by test), so one hunk set admits both
D111_PIN_SOURCES: Tuple[str, ...] = ("56aff576e", "c65b47ba4")
#: the JUDGE fallback chain (181ac97f4 cherry-picked onto the battery branch) -- judge chain code ONLY; filled from the merged tip,
#: whose conflict resolution in manifest.py / petri_runner.py is what the trees carry; a hunk outside both tables refuses
#: computed from `git diff 56aff576e b6754f7ba` (the assembly tip with that branch merged; byte-identical to its tip 1862aea8c on these files:
#: 181ac97f4 cherry-picked as dbf326ead plus the generator rule); every hunk's
#: +/- lines are lines of 181ac97f4 / 3931984bf except the two manifest.py resolution hunks named as such; the carry sources' blobs are equal, so one set admits both
D113_ADMITTED_HUNKS: Dict[str, Dict[str, str]] = {
    "src/sisteer/capeval/petri/auditor_fallback.py": {
        "f412da8b78174afdb1691bd7a54d499e2cef51ca89ebb9ca0cb464808b823824": "(-0 +9) +#: the JUDGE role's chain record, one key per role so the two provenance record",
        "4410f29760b1c7f9d47a570d2e10f0d27b847026bc69d0a9b8ba4a48d8cce9a5": "(-0 +2) +    if \"opus-5\" in low:",
        "09f2e4b8fa06bf125e948cd898705d0391c75d1b8b9234260b886ba3d2792bf3": "(-0 +3) +        role: str = \"auditor\",",
        "d41e4c23eac0fa3f51896084ac165f6d59d1eda56c96d7e15822c906f5ba4d18": "(-1 +1) -            raise AuditorFallbackError(\"auditor chain must contain a primary le",
        "34f03fb169e75e01203c8cf7ed31b04e65079ade16b95942c8156c14ee9abec4": "(-1 +1) -            raise AuditorFallbackError(f\"auditor chain repeats a model: {names}",
        "f291bc4efc9993d973fa47d85c2c9a7c4d5f18c8c853ecbb53688d4002f35ce0": "(-0 +11) +        #: which role this chain serves and under which store key its provenanc",
        "3566079bafcb97bb49124ee4e7a8a5ed0107581d5134e8d41002defc9fb7f55e": "(-1 +1) -        state = _store_get(store, AUDITOR_CHAIN_STORE_KEY)",
        "5f7f5ea8350f0d0fcc42dc9e451e0e42eaaf77530cee8bb299b655ab6ca0272a": "(-1 +1) -            _store_set(store, AUDITOR_CHAIN_STORE_KEY, state)",
        "520fa8ce3a00b90064ddf3ac8393f1872d23c590af7780ba21d984cf14d96c9e": "(-1 +1) -            raise AuditorFallbackError(\"recorded auditor_chain is not a mapping",
        "d5c04414ca0132c5818b2b05ce57ed1f356ed2cbab4b243abd0f6af115968e13": "(-1 +1) -                \"recorded auditor_chain configuration differs from this run\"",
        "c78c580197f4904ed989e26568c001d6c82c788031b0e6f241d13758715a3fec": "(-1 +1) -                    _store_set(store, AUDITOR_CHAIN_STORE_KEY, state)",
        "2251a83b5318f2c31c7d23246583e2298aa1a2466594e1190538eedb4ff3c8eb": "(-1 +1) -                _store_set(store, AUDITOR_CHAIN_STORE_KEY, state)",
        "278bb070b70f6bd4dbc4f4f22792e146a4fd3d6f6cdb5a872eb694bb12c05ca6": "(-0 +48) +                stop_reason = getattr(output, \"stop_reason\", None)",
        "003d4a96fa392c75dff5c7bb7cb9e70be9b7c2f7fea21cc8cc598f35ba4a8245": "(-1 +1) -                metadata[AUDITOR_CHAIN_STORE_KEY] = copy.deepcopy(state)",
        "c438e5fd1b144292e9e287271e75e98d3cd57edc8f0517ddbcd78baf25354f14": "(-0 +2) +    *,",
        "3e90b9d2f2cd66c70cc508e7818615680edd8ca6313382cc63c830e0deb13f27": "(-0 +2) +    ``role`` is the inspect model role every leg is stamped with (\"auditor\",",
        "5f6047fc50762bdb9129770ab5a3b57edb67b97042e20b457deaf84b1acd1070": "(-1 +1) -        model._set_role(\"auditor\")",
        "1e6c10fc120ebd781ca7154d3de58fe5e7f3f8c220568fa768ef6538ee60be01": "(-0 +3) +    role: str = \"auditor\",",
        "20e788e78578407e31dbfd2f2eea1b822ee1b3cd9b83d7326bc168bbbb22b14a": "(-1 +2) -    \"\"\"Return a concrete ``inspect_ai.model.Model`` auditor role wrapper.\"\"\"",
        "49f997ffd3b7649f962d00b597de459a7110370c88ab4e11f7bd8382e76094db": "(-1 +1) -        primary_model, fallback_models",
        "1620948b8ebc409ef68041cbe4ee1b56b5c30377f5e82fe35fef42fc6a476d84": "(-1 +3) -        resolved, discriminator=discriminator, state_getter=state_getter",
        "be1e31e08e96ff96c4bad670816b9adec0d2941ae2fd9ebae8975ddb0c1d287b": "(-1 +1) -    wrapped._set_role(\"auditor\")",
        "4f613a193c1e327392c8e7b1777cfc5ba583bb73b22192d4a2f2f28b101155e2": "(-0 +38) +",
    },
    "src/sisteer/capeval/manifest.py": {
        "894596d45d7f1bcd6ebf9da9dbb0d5004c6fa338f70d3aa280b47e5ccd3a1736": "(-2 +3) -    additive ``auditor_fallback_models`` are required; omitting that field",
        "4ddf2edc0a5e46c770d6d6750f5475423f2bdf40d045e2f83acafdd6a991c4e8": "(-0 +13) +    # Ordered fallback JUDGES after judge_model (2026-09-19, the user: Grok",
        "f9e7f820520040718a3a29fa77d0444cb9a966c9c1f0837ce2c46e191af2c250": "(-0 +8) +    if (not isinstance(s.judge_fallback_models, list)",
        "b06cadf8bab12403a4d078f03ea1302fb434863acee776931357165c54fe6dcd": "(-0 +10) +    # the judge chain's RAW value, the same rule; here omission and [] both",
        "2c9ae8e54835b69b3d345faa8429d30ead9bb70d518b07a219b3cad7f7574ea2": "(-0 +1) +    spec.judge_fallback_models = list(spec.judge_fallback_models)",
        "808fbc5c0897e94be861347e2107c0fe43e293f4adab4556c247912ecb7edf16": "(-1 +4) -    - the auditor / judge / scenarios role model IDs;",
        "0b8eb85cea20afcaa146b15cb2d4bfb592474c8111e479833d97b52abc3b4d0a": "(-0 +4) +            # the judge chain, the identical ADDITIVE rule (2026-09-19): a",
        "068d2129130a428ccb5190dfe78740572151883701338c7d67b6afc944615ca1": "(-1 +1) load_petri_spec optional field set gains judge_fallback_models (fk2b's resolution of 181ac97f4 over the D111 pick)",
        "b565cb782c14c908658231ec29b250e9e3292eef541dbb6c7a066739f285a957": "(-2 +2) load_petri_spec required-fields message names judge_fallback_models (fk2b's resolution)",
    },
    "src/sisteer/capeval/tiers/petri_runner.py": {
        "5adc2f58aa69789adcb2ef3e1abce02796056833dd20b9d15d6fdc4eeb81aa58": "(-0 +3) +#: the judge chain's twin (2026-09-19): stamped only when NON-EMPTY, and on",
        "575ce4d5805028fb94336755fd2c381e2dfc6c99705804499b9b886f9706ac13": "(-0 +5) +    if spec.judge_fallback_models:",
        "8b1052d7d8d3b3a06d7f5b45e8180ec4c79d623f9d057de1ad9bc7eb968a62f9": "(-0 +1) +    jfb_key = JUDGE_FALLBACK_METADATA_KEY",
        "e801ea1ae4648051a163a8b481cc718466db34249146a5e6dd0758189fb15901": "(-1 +1) -             if k != fb_key and md.get(k) != v}",
        "e6d6bb32eedaea36f66a469941a3e5652974473c54b1e23ebc6b5e63b898ceef": "(-0 +5) +    # the judge chain is compared by the same MEANING rule (absent == [])",
        "51d2a3e73a12e34b843908fad6dcd8461ebf0a4141eae210f1a4c474493a557c": "(-0 +9) +    # main: a role is wrapped in the manifest's fallback chain ONLY when a",
        "a1424a0a6ded9172965ed2235f9daa5974503075dacacd03e7d0f946a2f9c4df": "(-0 +8) +    if spec.judge_fallback_models:",
        "b89ff9025d3a6e2e77849abc8c4369b9114db7417b24d30024b52e640e821fd6": "(-1 +1) -                \"judge\": spec.judge_model,",
    },
}
#: claude-opus-4-8 goes BEFORE grok-4.6 in BOTH Petri chains (1f2fd34bb cherry-picked, the generator rules 7e322a766, tip 8681c01bd):
#: the ONE code hunk is `auditor_label`'s two new labels -- the zero-context diff merges the
#: `opus-5` label lines with the `opus-4-8` ones into one hunk, pinned here; the manifests' chain lines are the chain-list kind
D120_ADMITTED_HUNKS: Dict[str, Dict[str, str]] = {
    "src/sisteer/capeval/petri/auditor_fallback.py": {
        "547afca63b00901f1d8756105fbbba3717503036bddb657ba5dc2e495f7d3009": "auditor_label (+4): the `opus-5` (D113) and `opus-4-8` (D120) report labels -- one hunk, adjacent lines",
    },
}
#: completed rows carry across a tree change that leaves the store bytes untouched:
#: the THREE store-side hunks of 6ce849aaa -- the judged drivers tell the store's recorder which items they GENERATED
#: (`store.note_generated(...)`, one call each) and `CellStore.note_generated` itself, which writes NO store byte (its own docstring). Pinned
#: by content hash from `git diff ff96b7210 5e0b3c9b0` (the base -> the registry-completion tip); an unrelated change to any of the three is
#: an extra hunk and refuses. The recorder module the calls reach (`src/sisteer/registry/hooks.py`) is GATING with its own pinned entry below.
D136_ADMITTED_HUNKS: Dict[str, Dict[str, str]] = {
    "scripts/served_capability_run.py": {
        "e66f2f7d4caf976374102fcc789b39129f47a032b207b981a2c63eb99716020a": "cmd_generate (+3): after _run_judged_served, `store.note_generated(spec.task, sorted(spool.entries()), str(spool.path))` + its comment",
    },
    "scripts/cell_controls_run.py": {
        "0346b0dfcfc217be709836630710d70f77005acbb46a074292dece761eb67baf": "cmd_generate (+2): after judged.generate, `store.note_generated(spec.task, sorted(spool.entries()), str(spool.path))` + its comment",
    },
    "src/sisteer/capeval/store.py": {
        "a108dee87ca949b26ea14e252087eb282bda1292a0791d2b6c6db958f28ab704": "CellStore.note_generated (+11): tells the recorder the generated items -- writes no store byte",
    },
    # the recorder's REVIEWED emission-only changes between the carry sources and this tree -- the list-per-served-key `emit_serving_patches`
    # (f408d4d1f, three hunks) and on_generated / aggregate run_metric / code_sha (6ce849aaa, f725af8b7, 5e0b3c9b0, ten hunks) -- pinned
    # from `git diff 56aff576e 5e0b3c9b0` (the two carry sources' blobs are equal; the other base's diff is the same ten hashes). None touches
    # `run_payload` / `_seed_base` / the row-constant reads the driver probes; a historical recorder change to those (pre-c98306e60) refuses.
    "src/sisteer/registry/hooks.py": {
        "d521f9ac69cb564ba13801a83fbec6c90d0ef386f1baaef7fe2518d88cc0bf96": "fr5 2a/fr5 2b (-0 +10): #: fr5 (W24, 2a): the metric id of the null-valued ``item_result`` a j",
        "6c6a6539051f9e5fcecdfb91b28a1cf329b24851cd6c12e069d0072174b81b7d": "fr5 2b (-0 +1): self._metrics: Dict[str, List[Dict[str, object]]] = {}     # route -> ",
        "06ede4c491d3401a8bc4c0fdc58d10323443d40eaa13f0e4897d075db72fc69e": "fr5 2a/fr5 2b (-0 +7): if item == AGGREGATE_ITEM_ID or metric.endswith(AGGREGATE_SUFFIX):",
        "a28e5f4f61786cfdbbba3393f0a0c2f9e4e203be1c841ec8bc6f296085c0f7f6": "fr2 D123 (W23)/fr5 2a/fr5 2b (-0 +34): def on_generated(self, task: str, item_ids: Sequence[str], uri: str) -",
        "1f6b4d5c813c6bb8871cd5653ae70a7ae2a0a59c47f50b59fb619e72e591e2ae": "fr5 2b (-1 +1): routes = [route] if route else list(self._items) + [r for r in self._f",
        "dfdbb73ef80a22c6021805affaf100b34f77ea8590d30f3c4a0c665ef88f9a70": "fr5 2b (-1 +4): if rows or flags:",
        "f828431fb01e0906c3da29da2e5ed7f077fbd33261c207c20007f2fbc23a0403": "fr5 2b (-1 +1): if any(self._items.values()) or any(self._flags.values()):",
        "700b37512939cc953e68ca7e2617683bd9fe5a338f6ca4e5d36631419905e1b5": "fr5 2b (-0 +21): def aggregate_metric_row(run_id: str, metric: str, value: Optional[flo",
        "0a21b27283b2101bce30be4c6d3450e1194dece7a54d984444fc521d83af0b72": "fr5 5 (-1 +2): session: Optional[str] = None, extra: Optional[Mapping[str, object]] =",
        "5f86196379c6e17aea707758c89d6cda6e60af4245da3bcd3e1e5f185093a18c": "fr5 5 (-1 +1): producer = emit.producer_block(script, session=session)",
        "86870cfe6f572adc3c3a0836d022d6b5f408421a5075a56bfaff1d3e2d588ca5": "fr2 D123 (W23) (-1 +4): the launching shell from the planned runs; None when absent.\"\"\"",
        "c94736b95ee2f00fe38a49a7665e4cb315d04af67c26b9408040816a8ace024b": "fr2 D123 (W23)/fr5 2a (-13 +10): for served_key, run in runs.items():",
        "23dc86d7be533baf183a8a6e260d077df3a3c5b64f96611a137f09e91e1aa609": "fr2 D123 (W23) (-0 +14): def _serving_patch_for(endpoint: Mapping[str, Any], served_key: str, r",
    },
}
#: the Petri DRIVE's local relay-port plumbing (44aa5b1d1: `free_local_port` /
#: `local_port_of` / `open_local_relay`) -- the port never enters a store, a transcript or a key (the legs get PETRI_PROXY_BASE_URL with
#: whatever port was bound; the watcher does not parse it). Pinned from `git diff 8681c01bd 5e0b3c9b0`; an unrelated drive hunk refuses.
D137_ADMITTED_HUNKS: Dict[str, Dict[str, str]] = {
    "scripts/modal/seedrep_petri_drive.py": {
        "0c39dc33c8328ade5a02b5d8832b3fb2937c491ee1e510df71679bfe0ae05f59": "LOCAL_PORT_OFFSET -> LOCAL_PORT_MIN (-1 +2)",
        "c5f90eb14546789cc5d67b1bc88e21e41c6a0a597eaca4fbf44f9e325332c645": "free_local_port() added (+7): a loopback port the OS hands out free",
        "2e478eb11ebddfa9ad222b7882193a2d400d44702bf832bc790956a4cdc15a2a": "local_port_of (-1 +11): a free OS port in range, never proxy_port + 10000 (67204 on W23)",
        "d254ce80aff222b3f7b4ddc902ee483f2ece1739ca0a3cbcf33849c0bead8f18": "open_local_relay() added (+11): choose the port, refuse a held one, start the TLS relay",
        "a81a32bc9a794ef72c1d4cdbe2fffd737a8c330feaba59250b294a3ae2f38fa7": "main() (-4 +1): the relay opened through open_local_relay",
    },
}
ADMITTED_REASON_D137 = ("grading_host_plumbing (D137; fs2 F-AR: the Petri drive's local relay port is the grading host plumbing that never enters a store, a transcript or a key; "
                        "hunk set pinned by content hash from 8681c01bd -> 5e0b3c9b0)")
ADMITTED_REASON_D136 = ("registry_emission (D136; fr5 W24 2a: the judged drivers tell the results registry which items they generated -- a recorder call "
                        "that writes no store byte is not the eval's identity; hunk set pinned by content hash from ff96b7210 -> 5e0b3c9b0)")
ADMITTED_REASON_D120 = ("petri_fallback_chain (D120; the user 2026-09-19: claude-opus-4-8 before grok-4.6 in both chains -- a chain-member label is the "
                        "fallback chain, not the eval's identity; hunk pinned by content hash from fk2c's tip 8681c01bd)")
ADMITTED_REASON_D113 = ("petri_fallback_chain (D113; the user 18:48Z: Grok 4.6 is the fallback for BOTH roles -- the judge chain's code is the "
                        "fallback chain, not the eval's identity; hunk set pinned by content hash from fk2's tip)")
ADMITTED_REASON_D111 = ("petri_fallback_chain (D111; user ruling 18:55Z / D115 (2): the retired abliterated-Qwen auditor leg's code -- flags, probe, "
                        "default chain item -- is the fallback chain, not the eval's identity; hunk set pinned by content hash)")
FALLBACK_CHAIN_KEY_RE = re.compile(r"^\s*(?:\"?)(auditor_fallback_models|judge_fallback_models)(?:\"?)\s*[:=]")
FALLBACK_CHAIN_ITEM_RE = re.compile(r"^\s*(?:-\s*)?[\"']?[A-Za-z0-9_./:-]+[\"']?,?\s*(?:#.*)?$")
ADMITTED_REASON_CHAIN = "petri_fallback_chain (D111 / D113; user ruling 18:55Z: the auditor / judge fallback chain is not the eval's identity)"
#: READERS by ruling (module docstring, clause (b)): recorded, never gating
CONTAINER_MODULES: Tuple[str, ...] = ("scripts/modal/seedrep_served.py", "scripts/modal/seedrep_launch_plan.py", "scripts/modal/seedrep_ladder.py")
TREE_FILE_RE = re.compile(r"^(?:configs|scripts|src)/[^\s]+\.(?:py|sh|yaml|yml|json|txt|toml|jsonl)$")
READY_RECORDS: Tuple[str, ...] = ("endpoint.json", "cells_served.json")
ATTESTATION_REL = f"{lp.SERVE_DIR_REL}/serving_attestation.json"
LADDER_ENQUEUED_REL = "ladder_enqueued.json"
PROVENANCE_KEYS: Tuple[str, ...] = ("W", "W7", "wave_uuid", "attempt_id", "tree_sha", "rs", "accepted_key", "dict", "standup_id", "source")
W_TOKEN, W7_TOKEN, WAVE_TOKEN, SMOKE_WAVE_TOKEN = "{W}", "{W7}", "{WAVE}", "{SMOKE_WAVE}"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# the prior tree's source                                                       #
# --------------------------------------------------------------------------- #

def load_prior_source(path: Path) -> Dict[str, Any]:
    """A prior tree's run state: a directory holding its plan (:data:`PRIOR_PLAN_FILES`, first present wins) and, when present, its minted smoke
    waves (`smokes/<id>/wave.json`), or the plan FILE itself (smoke waves from the sibling `smokes/`). Refuses by name otherwise. NOTE: the lane's
    RS root is shared across W trees (its plan is regenerated per W), so a prior tree's plan must be archived under its own directory before the
    next tree's chain regenerates it -- this loader reads what it is given and never guesses a tree from the shared root."""
    p = Path(path)
    if p.is_file():
        plan_path, base = p, p.parent
    elif p.is_dir():
        found = [p / n for n in PRIOR_PLAN_FILES if (p / n).is_file()]
        if not found:
            raise ValueError(f"carry-over source {p}: none of {list(PRIOR_PLAN_FILES)} is present")
        plan_path, base = found[0], p
    else:
        raise ValueError(f"carry-over source {p} is neither a directory nor a plan file")
    plan = json.loads(plan_path.read_text())
    for k in ("W", "W7", "wave_uuid", "standups", "rungs", "results_root", "materialisation"):
        if k not in plan:
            raise ValueError(f"carry-over source {plan_path}: not a launch plan (lacks {k!r})")
    if lp.plan_sha(plan) != plan.get("launch_plan_sha"):
        raise ValueError(f"carry-over source {plan_path}: launch_plan_sha {plan.get('launch_plan_sha')!r} is not its content digest {lp.plan_sha(plan)!r} -- edited after generation")
    waves: Dict[str, str] = {}
    for sid in lp.SMOKE_DERIVABLE_IDS:
        try:
            w = lp.smoke_execution_wave(base, sid)
        except ValueError:
            w = None
        if w:
            waves[sid] = w
    return {"path": str(p), "plan_path": str(plan_path), "plan": plan, "W": plan["W"], "W7": plan["W7"], "wave_uuid": plan["wave_uuid"],
            "launch_plan_sha": plan.get("launch_plan_sha"), "materialisation": plan["materialisation"], "smoke_waves": waves,
            "smokes_launched": read_smokes_launched(base, plan)}


#: fj19: the OPTIONAL per-prior LAUNCH record, `<prior dir>/smokes_launched.json` -- which smoke waves (and whether the production wave) were
#: actually LAUNCHED on that tree, hence which accepted Dicts MUST exist. The archive's `smokes/<Sid>/wave.json` files cannot say this: the
#: smoke waves are minted ONCE in RS and shared by every W (every prior archive carries all six, identical), so a wave record there is not a
#: launch. Without this file a missing Dict is LAUNCH HISTORY UNKNOWN (fj20): the candidate refuses by name and the row is held -- the record is
#: mandatory for any prior whose Dict is missing; a Dict that reads needs none.
SMOKES_LAUNCHED_REL = "smokes_launched.json"
SMOKES_LAUNCHED_SCHEMA = "seedrep_prior_smokes_launched/1"


def read_smokes_launched(base: Path, plan: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``{"launched": {smoke_id: wave_uuid}, "production": wave_uuid | None}`` from `<base>/smokes_launched.json`, None when the archive has no
    launch record; a malformed record, or one for another tree, refuses by name (never read as 'nothing launched')."""
    p = Path(base) / SMOKES_LAUNCHED_REL
    if not p.is_file():
        return None
    doc = json.loads(p.read_text())
    if not isinstance(doc, Mapping) or doc.get("schema") != SMOKES_LAUNCHED_SCHEMA:
        raise ValueError(f"{p}: schema {doc.get('schema') if isinstance(doc, Mapping) else None!r} != {SMOKES_LAUNCHED_SCHEMA!r}")
    if doc.get("W") != plan["W"]:
        raise ValueError(f"{p}: records tree {str(doc.get('W'))[:12]!r}, the archived plan is {plan['W'][:12]}")
    launched = doc.get("launched")
    if not isinstance(launched, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in launched.items()):
        raise ValueError(f"{p}: `launched` must map smoke id -> wave uuid")
    prod = doc.get("production")
    if prod is not None and (not isinstance(prod, str) or not prod):
        raise ValueError(f"{p}: `production` must be the launched production wave uuid or null")
    return {"launched": dict(launched), "production": prod, "path": str(p)}


#: fj20: the three verdicts on a source whose accepted Dict does not exist
DICT_VANISHED, DICT_HISTORY_UNKNOWN, DICT_NEVER_LAUNCHED = "vanished", "launch_history_unknown", "never_launched"


def missing_dict_verdict(prior: Mapping[str, Any], *, source: str, wave_uuid: str) -> Tuple[str, str]:
    """For a source whose accepted Dict does not exist -- ``(verdict, archive_note)``:
    :data:`DICT_VANISHED` -- the archive's launch record names this very wave for the source: its Dict was created at launch and is gone; the
    evaluation refuses by name (its acceptances are canonical work it cannot see).
    :data:`DICT_HISTORY_UNKNOWN` -- the prior carries no launch record, or its record names ANOTHER wave for the source: nothing says whether a
    completed row sits behind the missing Dict, so the candidate is refused by name and the row HELD (`carry_refused_holds`), never re-spawned
    (a 'no record = empty' default would let a COMPLETE historical row spawn again).
    :data:`DICT_NEVER_LAUNCHED` -- the record omits the smoke id (production: null): the source was never launched on that tree, an empty acceptance."""
    rec = prior.get("smokes_launched")
    smoke_id = SMOKE_ID_OF_SOURCE.get(source)
    label = "production" if smoke_id is None else smoke_id
    if rec is None:
        return DICT_HISTORY_UNKNOWN, (f"{prior.get('path')} carries no {SMOKES_LAUNCHED_REL}: launch history unknown -- write the record ({SMOKES_LAUNCHED_SCHEMA}: "
                                      f"`launched` {{smoke id: wave uuid}} for every smoke this tree launched, `production` its wave or null) before evaluating; "
                                      "the shared smoke-wave mints are not launch records")
    recorded = rec.get("production") if smoke_id is None else (rec.get("launched") or {}).get(smoke_id)
    if recorded == wave_uuid:
        return DICT_VANISHED, f"the archive records wave {wave_uuid} launched for {label} ({rec.get('path')})"
    if recorded:
        return DICT_HISTORY_UNKNOWN, (f"{rec.get('path')} records {label} launched under wave {recorded}, the candidate wave is {wave_uuid} -- an inconsistent archive; "
                                      "launch history unknown")
    return DICT_NEVER_LAUNCHED, f"{rec.get('path')} records no {label} launch for this tree"


def _production_rows(plan: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return list(plan.get("standups") or []) + list(plan.get("rungs") or [])


def archived_row(prior: Mapping[str, Any], standup_id: str) -> Mapping[str, Any]:
    """The archived plan's production row named ``standup_id`` (a standup or a rung) -- the row `_candidates` resolves for the consuming row of the
    same id, and the row every consumer-side derivation reads (the chain policy's manifests and target cells, the expected store inputs); the
    smoke rows `lp.smoke_row` reconstructs from it keep those fields (argv, children argv, cells_served_expected), so a derivation over this row
    IS the derivation over any of its smoke rows. Refuses by name when the archive has none (fj33: the proof CLI reads its row here)."""
    row = next((r for r in _production_rows(prior["plan"]) if r.get("standup_id") == standup_id), None)
    if row is None:
        raise ValueError(f"the archived plan {prior.get('plan_path')} (tree {str(prior.get('W'))[:12]}) has no row {standup_id!r}")
    return row


# --------------------------------------------------------------------------- #
# clause (a): row identity                                                      #
# --------------------------------------------------------------------------- #

def _substitute(obj: Any, subs: Sequence[Tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        for src, tok in subs:
            if src:
                obj = obj.replace(src, tok)
        return obj
    if isinstance(obj, Mapping):
        return {k: _substitute(v, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, subs) for v in obj]
    return obj


def normalise_row(row: Mapping[str, Any], *, W: str, W7: str, wave_uuid: str, smoke_wave: Optional[str] = None, s5_full: bool = False) -> Dict[str, Any]:
    """The row's SPELLING with the tree identity tokenised (W -> {W}, W7 -> {W7}, the production wave -> {WAVE}, a smoke wave -> {SMOKE_WAVE}) and
    the `smoke` block removed. Under the S5 FULL clause the smoke's own rewrites are undone: the `smoke-S5__` standup-id prefix, the adapter-role
    `expected_reused` flag (set aside on EVERY adapter, both sides compare without it) and the smoke's container `timeout_s` bound."""
    r = copy.deepcopy(dict(row))
    smoke = r.pop("smoke", None)
    if s5_full:
        if isinstance(smoke, Mapping) and smoke.get("production_standup_id"):
            r["standup_id"] = smoke["production_standup_id"]
        for a in (r.get("adapters") or {}).values():
            if isinstance(a, dict):
                a.pop("expected_reused", None)
        if isinstance(r.get("modal"), dict):
            r["modal"] = {k: v for k, v in r["modal"].items() if k != "timeout_s"}
    # under the S5 FULL clause the smoke wave STOOD IN for the production wave (the smoke row's keys / ladder root carry it): one token for both
    subs = [(W, W_TOKEN), (smoke_wave or "", WAVE_TOKEN if s5_full else SMOKE_WAVE_TOKEN), (wave_uuid, WAVE_TOKEN), (W7, W7_TOKEN)]
    return _substitute(r, subs)


def _diff_paths(a: Any, b: Any, path: str) -> List[str]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing in prior' if k not in a else 'missing in this plan'}")
            else:
                out += _diff_paths(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_paths(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


EXECUTED_SHAPE_FULL, EXECUTED_SHAPE_LIMITED, EXECUTED_SHAPE_UNKNOWN = "full", "limited", "unknown"


def _argv_flag_value(argv: Sequence[Any], flag: str) -> Tuple[bool, Optional[str]]:
    """``(present, value)`` of a budget flag in an executed argv: `--flag v` or `--flag=v`."""
    for i, a in enumerate(argv):
        a = str(a)
        if a == flag:
            return True, (str(argv[i + 1]) if i + 1 < len(argv) else None)
        if a.startswith(flag + "="):
            return True, a.split("=", 1)[1]
    return False, None


def executed_shape(*, evidence: Optional[Mapping[str, Any]], prod_row: Mapping[str, Any], prev_row: Mapping[str, Any], root: str,
                   read_text: Callable[[str], Optional[str]], aid: Optional[str] = None, W: Optional[str] = None,
                   wave_uuid: Optional[str] = None) -> Dict[str, Any]:
    """The shape the accepted attempt EXECUTED, from evidence BOUND to it -- never from a candidate today's
    `smoke_row` reconstructs (an older generator added no budget flags where today's adds `--limit-goals 10` to the same row).

    1. IDENTITY first: `modal_evidence.json` must name THIS attempt, tree, wave and standup (`attempt_id`, `W`, `wave_uuid`,
       `standup_id`); another attempt's evidence classifies nothing -> ``unknown``.
    2. the executed CHILDREN (`children`) against the PRIOR tree's production row's: fewer -> ``limited``.
    3. the executed BUDGET, for a leg with a budget flag (`SMOKE_BUDGET_FLAG_OF_LEG`): EVERY executed child's argv as recorded
       (`children/<eval_leg>__<arm>__s<s>.argv.json`) is read -- unreadable -> ``unknown``; the flag / value must agree across the children --
       mixed -> ``unknown``; and when the container recorded a `smoke.budget` it must AGREE with the executed argv (a `full_shape` budget with a
       flagged argv, an `item_budget` whose value the argv does not carry, or the reverse) -- contradiction -> ``unknown``. METADATA NEVER WINS
       OVER RECORDED EXECUTION: the verdict is the argv's.
    ``full`` (a production attempt, or a smoke that ran every production child without a budget flag) HOLDS on a refusal; ``limited`` releases
    the hold; ``unknown`` RETAINS it with the reason named -- missing, inconsistent or unbound history is never permission to re-run. Every
    read goes through ``read_text`` (a RecordingReader in the evaluator), so the inputs are digest-bound and re-checked at certify."""
    if not isinstance(evidence, Mapping):
        return {"shape": EXECUTED_SHAPE_UNKNOWN, "why": f"{root}/modal_evidence.json absent or not an object -- the executed shape cannot be read", "source": None}
    want = {"attempt_id": aid, "W": W, "wave_uuid": wave_uuid, "standup_id": prev_row.get("standup_id")}
    bad = [f"{k} {evidence.get(k)!r} != {v!r}" for k, v in want.items() if v is not None and evidence.get(k) != v]
    if bad:
        return {"shape": EXECUTED_SHAPE_UNKNOWN, "why": f"{root}/modal_evidence.json is not this attempt's evidence ({'; '.join(bad)}) -- nothing it says classifies the shape",
                "source": "modal_evidence.json identity"}
    smoke = evidence.get("smoke")
    kids = evidence.get("children")
    if smoke is None:
        return {"shape": EXECUTED_SHAPE_FULL, "why": "a production attempt (no smoke block on its evidence)", "source": "modal_evidence.json"}
    if not isinstance(smoke, Mapping) or not isinstance(kids, list):
        return {"shape": EXECUTED_SHAPE_UNKNOWN, "why": "the evidence's smoke block / children list is malformed", "source": "modal_evidence.json"}
    n_prod = len(prod_row.get("children") or [])
    if len(kids) != n_prod:
        return {"shape": EXECUTED_SHAPE_LIMITED, "why": f"the attempt ran {len(kids)} child(ren); the prior tree's production row has {n_prod}", "source": "modal_evidence.json children"}
    spec = lp.SMOKE_BUDGET_FLAG_OF_LEG.get(str(prev_row.get("leg")))
    if spec is None:
        return {"shape": EXECUTED_SHAPE_FULL, "why": f"every production child ran and the {prev_row.get('leg')} leg carries no D107 item budget", "source": "modal_evidence.json children"}
    flag, budget_key = spec
    # the EXECUTED argv of every child -- the record of what ran; unreadable or mixed -> unknown
    values: List[Tuple[Any, bool, Optional[str]]] = []
    for k in kids:
        if not isinstance(k, Mapping):
            return {"shape": EXECUTED_SHAPE_UNKNOWN, "why": "a child record on the evidence is not an object", "source": "modal_evidence.json children"}
        argv = k.get("argv")
        if not isinstance(argv, list):
            text = read_text(f"{root}/children/{k.get('eval_leg')}__{k.get('arm')}__s{k.get('s')}.argv.json")
            try:
                argv = (json.loads(text) or {}).get("argv") if text is not None else None
            except ValueError:
                argv = None
        if not isinstance(argv, list):
            return {"shape": EXECUTED_SHAPE_UNKNOWN, "source": "children/*.argv.json",
                    "why": f"the executed argv of child {k.get('arm')!r} is not readable -- the executed budget cannot be established (metadata alone never decides)"}
        present, val = _argv_flag_value(argv, flag)
        values.append((k.get("arm"), present, val))
    kinds = {(present, val) for _arm, present, val in values}
    if len(kinds) != 1:
        return {"shape": EXECUTED_SHAPE_UNKNOWN, "source": "children/*.argv.json",
                "why": f"the executed argv disagree on {flag} across the children ({sorted((str(a), p, v) for a, p, v in values)}) -- an inconsistent execution record"}
    present, val = next(iter(kinds))
    executed = {"flag": flag, "present": present, "value": val, "n_children": len(values)}
    budget = smoke.get("budget")
    if isinstance(budget, Mapping) and budget.get("kind"):
        rec_kind, rec_val = budget.get("kind"), budget.get(budget_key)
        agree = ((rec_kind == lp.SMOKE_BUDGET_KIND_FULL or rec_val is None) and not present) or \
                (rec_kind == lp.SMOKE_BUDGET_KIND_ITEM and rec_val is not None and present and str(val) == str(rec_val))
        if not agree:
            return {"shape": EXECUTED_SHAPE_UNKNOWN, "source": "modal_evidence.json smoke.budget vs children/*.argv.json", "executed": executed,
                    "why": f"the recorded smoke.budget ({rec_kind}: {budget_key} = {rec_val!r}) contradicts the executed argv ({flag} {'= ' + str(val) if present else 'absent'}) -- "
                           "metadata never wins over recorded execution; the shape is unknown"}
    if present:
        return {"shape": EXECUTED_SHAPE_LIMITED, "why": f"every executed child carries {flag} {val} (D107){'; the recorded smoke.budget agrees' if isinstance(budget, Mapping) and budget.get('kind') else ''}",
                "source": "children/*.argv.json", "executed": executed}
    return {"shape": EXECUTED_SHAPE_FULL, "why": f"every production child ran and none of the {len(values)} executed argv carries {flag}"
                                                  f"{'; the recorded smoke.budget agrees' if isinstance(budget, Mapping) and budget.get('kind') else ''}",
            "source": "children/*.argv.json", "executed": executed}


def smoke_shape_problems(prev_row: Mapping[str, Any], this_row: Mapping[str, Any]) -> List[str]:
    """Why a CANDIDATE smoke row (today's `lp.smoke_row` reconstruction) is not the production row's full shape: fewer children than the
    production row, or any child carrying a budget flag. INFORMATIONAL for the hold (recorded as `candidate_shape_problems`):
    the hold is derived from :func:`executed_shape` -- evidence bound to the accepted attempt -- because today's generator can add flags the
    prior tree's never executed."""
    if prev_row.get("smoke") is None:
        return []
    out: List[str] = []
    n_prev, n_this = len(prev_row.get("children") or []), len(this_row.get("children") or [])
    if n_prev != n_this:
        out.append(f"the smoke ran {n_prev} child(ren), the production row has {n_this}")
    for k in prev_row.get("children") or []:
        flagged = [f for f in lp.SMOKE_BUDGET_FLAGS if any(a == f or str(a).startswith(f + "=") for a in (k.get("argv") or []))]
        if flagged:
            out.append(f"child {k.get('arm')} s{k.get('s')} carries the smoke-only budget flag(s) {flagged} (D107)")
    return out


def row_identity_verdict(prev_row: Mapping[str, Any], this_row: Mapping[str, Any], *, prev: Mapping[str, Any], this: Mapping[str, Any],
                         prev_smoke_wave: Optional[str] = None) -> Dict[str, Any]:
    """Clause (a). ``prev`` / ``this`` carry W, W7, wave_uuid. A prior smoke row is admitted ONLY under the S5 FULL clause; this plan's row must
    be a production row (no `smoke` block)."""
    if this_row.get("smoke") is not None:
        return {"ok": False, "why": f"this plan's row {this_row.get('standup_id')} carries a smoke block -- only production rows are carry targets"}
    if this_row.get("kind") == lp.S5_PROBE_KIND or prev_row.get("kind") == lp.S5_PROBE_KIND:
        return {"ok": False, "why": f"an {lp.S5_PROBE_KIND} row is a smoke-only probe, never a production row"}
    smoke = prev_row.get("smoke")
    s5_full = False
    if smoke is not None:
        if not isinstance(smoke, Mapping):
            return {"ok": False, "why": "the prior row's smoke block is not an object"}
        sid = smoke.get("smoke_id")
        if sid not in FULL_SHAPE_SMOKE_SOURCES:
            return {"ok": False, "budgeted": True, "why": f"the prior row ran as smoke {sid!r}: an S1 / S1b / S2 / S4 smoke is BUDGETED (D107: bounded children / items) and is never a carry source"}
        budget = smoke.get("budget")
        if sid == S5_SMOKE_ID and (not isinstance(budget, Mapping) or budget.get("kind") != lp.SMOKE_BUDGET_KIND_FULL or budget.get("bon_limit_goals") is not None
                                   or budget.get("misalign_limit_items") is not None):
            return {"ok": False, "why": f"the prior S5 row's smoke.budget {budget!r} is not the FULL shape ({lp.SMOKE_BUDGET_KIND_FULL}, no item budget)"}
        if len(prev_row.get("children") or []) != len(this_row.get("children") or []):
            return {"ok": False, "why": f"the prior {sid} row ran {len(prev_row.get('children') or [])} child(ren), this plan's row has {len(this_row.get('children') or [])} -- "
                                        "a bounded smoke, not the FULL shape"}
        for k in prev_row.get("children") or []:
            flagged = [f for f in lp.SMOKE_BUDGET_FLAGS if any(a == f or str(a).startswith(f + "=") for a in (k.get("argv") or []))]
            if flagged:
                # the item budget IS the shape difference -- a bon / misalign S3 row is BUDGETED by construction and never a carry source
                return {"ok": False, "budgeted": True, "why": f"the prior {sid} row's child {k.get('arm')} s{k.get('s')} carries the smoke-only budget flag(s) {flagged} -- BUDGETED (D107), never a carry source"}
        if smoke.get("production_standup_id") != this_row.get("standup_id"):
            return {"ok": False, "why": f"the prior {sid} row is the smoke of {smoke.get('production_standup_id')!r}, not of {this_row.get('standup_id')!r}"}
        s5_full = True
    a = normalise_row(prev_row, W=prev["W"], W7=prev["W7"], wave_uuid=prev["wave_uuid"], smoke_wave=prev_smoke_wave, s5_full=s5_full)
    b = normalise_row(this_row, W=this["W"], W7=this["W7"], wave_uuid=this["wave_uuid"], s5_full=s5_full)
    diffs = _diff_paths(a, b, "")
    smoke_id = (smoke or {}).get("smoke_id") if s5_full else None
    out = {"ok": not diffs, "s5_full_row": bool(s5_full and smoke_id == S5_SMOKE_ID), "full_smoke_row": s5_full, "smoke_id": smoke_id,
           "clause": (FULL_CLAUSE_OF_SMOKE[smoke_id] if s5_full else "production_row"),
           "tokens": {"W": W_TOKEN, "W7": W7_TOKEN, "wave_uuid": WAVE_TOKEN, "smoke_wave": (WAVE_TOKEN if s5_full else SMOKE_WAVE_TOKEN)},
           "set_aside": (["smoke", "standup_id smoke- prefix", "adapters[*].expected_reused", "modal.timeout_s"] if s5_full else ["smoke"]),
           "row_sha256": _sha_text(json.dumps(b, sort_keys=True, default=str))}
    if diffs:
        out["why"] = f"the row spells differently in the prior plan: {diffs[:6]}{' ...' if len(diffs) > 6 else ''}"
        out["diff_paths"] = diffs
    return out


# --------------------------------------------------------------------------- #
# clause (b): the producer file set                                             #
# --------------------------------------------------------------------------- #

def _tree_rel(token: Any, W: str) -> Optional[str]:
    if not isinstance(token, str):
        return None
    for prefix in (f"{CONTAINER_TREE_PREFIX}{W}/",):
        if token.startswith(prefix):
            rel = token[len(prefix):]
            return rel if rel and "/" in rel or rel.endswith(".py") else None
    if TREE_FILE_RE.match(token):
        return token
    return None


def _module_tokens(argv: Sequence[Any]) -> List[str]:
    """`python -m sisteer.x.y` -> `src/sisteer/x/y.py` (the misalign runner / the usage ledger are launched as modules, not scripts)."""
    out = []
    for i, tok in enumerate(argv):
        if tok == "-m" and i + 1 < len(argv) and isinstance(argv[i + 1], str) and argv[i + 1].startswith("sisteer."):
            out.append("src/" + argv[i + 1].replace(".", "/") + ".py")
    return out


def producer_set(row: Mapping[str, Any], evidence: Optional[Mapping[str, Any]], *, W: str) -> Dict[str, List[str]]:
    """The leg's producer FILE SET of one row (tree-relative), grouped by how each file was derived; ``all`` is the sorted union the git check
    runs over, ``container_modules`` the readers recorded but not gating."""
    argv_files, child_files, writer_files = set(), set(), set()
    for tok in row.get("argv") or []:
        rel = _tree_rel(tok, W)
        if rel:
            argv_files.add(rel)
    for tok in (evidence or {}).get("standup_argv") or []:               # the argv as it actually ran (the container's record)
        rel = _tree_rel(tok, W)
        if rel:
            argv_files.add(rel)
    for k in row.get("children") or []:
        for tok in k.get("argv") or []:
            rel = _tree_rel(tok, W)
            if rel:
                child_files.add(rel)
        child_files.update(_module_tokens(k.get("argv") or []))
    for tok in row.get("rung_consumer") or []:
        rel = _tree_rel(tok, W)
        if rel:
            child_files.add(rel)
    asm = row.get("rung_assembly") or {}
    if isinstance(asm, Mapping) and asm.get("helper"):
        child_files.add(str(asm["helper"]))
    modules = {PROXY_REL}
    if row.get("leg") == "gsm8k":
        modules.add(LEDGER_REL)
    modules.update(LEG_HELPERS.get(str(row.get("leg")), ()))
    for name in COMPLETE_CHECKS + ("adapter_hashes_banked",):
        if name == "pool_endpoint" and not row.get("pool"):
            continue
        writer_files.add(lp.PRODUCER_WRITERS[name])
    ev = evidence or {}
    for b in (ev.get(lp.PRODUCER_EVIDENCE_KEY) or {}).values():
        if isinstance(b, Mapping) and isinstance(b.get("writer"), str):
            writer_files.add(b["writer"])
    for b in (ev.get(lp.DETERMINISM_EVIDENCE_KEY) or {}).values():
        if isinstance(b, Mapping):
            if isinstance(b.get("writer"), str):
                writer_files.add(b["writer"])
            for tok in b.get("argv") or []:
                rel = _tree_rel(tok, W)
                if rel:
                    writer_files.add(rel)
    gating = (argv_files | child_files | modules | writer_files) - set(CONTAINER_MODULES)
    return {"standup": sorted(argv_files - set(CONTAINER_MODULES)), "children": sorted(child_files - set(CONTAINER_MODULES)), "modules": sorted(modules),
            "writers": sorted(writer_files - set(CONTAINER_MODULES)), "all": sorted(gating), "container_modules": sorted(CONTAINER_MODULES)}


def direct_imports(git: Callable[[Sequence[str]], Tuple[int, str]], sha: str, scripts: Sequence[str]) -> Dict[str, List[str]]:
    """Clause (b)'s import walk, depth :data:`IMPORT_DEPTH`: for every `.py` producer script, its DIRECT `import` / `from ... import` targets that
    resolve to a tree file at ``sha`` -- `sisteer.a.b` -> `src/sisteer/a/b.py` (or the package's `__init__.py`), a bare name -> a sibling of the
    importing script or `scripts/<name>.py`. ``{script: [resolved paths]}``; a script that is not parseable at the sha contributes nothing (the
    blob check on the script itself still gates)."""
    import ast
    out: Dict[str, List[str]] = {}
    for path in sorted(set(scripts)):
        if not path.endswith(".py"):
            continue
        rc, text = git(["show", f"{sha}:{path}"])
        if rc != 0:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.append(node.module)
                names += [f"{node.module}.{a.name}" for a in node.names]
        cands: List[str] = []
        parent = str(Path(path).parent)
        for n in sorted(set(names)):
            if n.startswith("sisteer."):
                base = "src/" + n.replace(".", "/")
                cands += [base + ".py", base + "/__init__.py"]
            elif "." not in n:
                cands += [f"{parent}/{n}.py", f"scripts/{n}.py"]
        if not cands:
            out[path] = []
            continue
        present = _blobs_at(git, sha, cands)
        out[path] = sorted(p for p, b in present.items() if b is not None and p != path)
    return out


def default_git(repo: Path) -> Callable[[Sequence[str]], Tuple[int, str]]:
    def git(args: Sequence[str]) -> Tuple[int, str]:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.returncode, r.stdout
    return git


def _blobs_at(git: Callable[[Sequence[str]], Tuple[int, str]], sha: str, files: Sequence[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {f: None for f in files}
    if not files:
        return out
    rc, text = git(["ls-tree", "-r", "-z", sha, "--", *files])
    if rc != 0:
        raise ValueError(f"git ls-tree {sha[:12]} failed (rc {rc}) -- is the sha in this repo?")
    for entry in text.split("\0"):
        if not entry.strip():
            continue
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and path in out:
            out[path] = parts[2]
    return out


def diff_hunks(a: str, b: str) -> List[Dict[str, Any]]:
    """The ZERO-CONTEXT unified-diff hunks of ``a -> b``, each hashed over its `-` / `+` body lines (never the `@@` header, so an unrelated
    change elsewhere in the file moves line numbers without changing a pinned hunk's hash -- it shows up as an EXTRA hunk instead)."""
    import difflib
    hunks: List[Dict[str, Any]] = []
    cur: Optional[List[str]] = None
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm=""):
        if line.startswith(("---", "+++")) and cur is None:
            continue
        if line.startswith("@@"):
            if cur is not None:
                hunks.append(cur)
            cur = []
            continue
        if cur is not None:
            cur.append(line)
    if cur is not None:
        hunks.append(cur)
    out = []
    for body in hunks:
        out.append({"sha256": _sha_text("\n".join(body)), "n_minus": sum(1 for x in body if x.startswith("-")), "n_plus": sum(1 for x in body if x.startswith("+")),
                    "head": (body[0][:96] if body else "")})
    return out


#: every NAMED chain-code admission: (decision, its pinned hunk table, its reason). Adding one (claude-opus-4-8 before grok-4.6 in
#: both chains, say) is ONE table + ONE entry here; `d111_admitted_difference` reads them all and labels each hunk by its decision.
#: The `D138_ADMITTED_HUNKS` table is MOOT: the prior controls rows have no COMPLETE.json / acceptance, nothing to carry; controls s1 runs
#: fresh. The table stays EMPTY and unused; the machinery is kept. It would hold eecba9039's hunks in scripts/served_capability_run.py for
#: the CONTROLS leg's carry only. Verification from the diff: the `cmd_generate` call-site rewiring (`_bites_probe_render_arm`) and
#: `_served_evidence`'s three additive `steering_bites` keys execute on EVERY served run and land in evidence.json inside the store's cell
#: dir; the local header measurement (`_render_token_ids`, `_header_ids_under_arm`, the OFF-reference refusal) runs for a controls row
#: whenever `early_header is None` (the controls argv carries no --tinycap-identity-version). So no hunk of that commit is confined to the
#: gsm8k / thinking-ON path; the table is EMPTY until a rule names the hunks. The machinery is in place: a table restricted to LEGS is
#: consulted only for a carry of one of those legs (a gsm8k carry never sees it).
D138_ADMITTED_HUNKS: Dict[str, Dict[str, str]] = {"scripts/served_capability_run.py": {}}
ADMITTED_REASON_D138 = "other_leg_path (D138; PENDING: fd3's served_capability_run.py hunks for the controls carry, once a ruling names them)"
ADMITTED_KIND_OTHER_LEG = "other_leg_path"
#: (decision, table, reason, kind, legs) -- `legs` None = every leg; a frozenset restricts the admission to carries of those legs
#: the three FULL carries (pasttense ...-ccf6feacddc8, misalign ...-b4cbfce42621, controls
#: ...-dc4bd2f90303) passed clauses (a) and (c) and refused at (b) on producer files whose changes cannot alter a static / controls / misalign
#: row's generations or scoring. Each table below pins EVERY zero-context hunk of the real (prior tree -> launch tree 0f00f7fae) diff of one file,
#: labelled with the commit(s) `git blame` attributes its added lines to (a hunk-level match against the actual diff; an extra
#: or altered hunk refuses the file). Tables are LEG-restricted: a carry of any other leg (a Petri row, whose producer petri_runner.py IS) never
#: sees them. NOT admitted: eecba9039's served_capability_run.py hunks -- they execute on the controls path (the bites-probe render derivation
#: and the header measurement) -- and they are not needed: the controls carry source already carries that commit, and no prior-tree row's
#: producer set names that file.
D183_LEGS_STATIC_CONTROLS_MISALIGN: frozenset = frozenset(lp.STATIC_LEGS) | frozenset({"controls", "misalign"})
ADMITTED_KIND_OFF_PATH = "off_path_code"                # changed code the carried row's child never executes
ADMITTED_REASON_D183_PETRI = ("off_path_code (D183; petri_runner.py: W22 dbf326ead (judge fallback chain) + W22/W24 f29562721 (re-key guard): the module is imported by "
                              "served_capability_run.py, capeval.manifest / runtime and the audit reader, but every changed path runs only inside the Petri drive -- a "
                              "static / controls / misalign child never calls it; its generations and scoring cannot change)")
ADMITTED_REASON_D183_POOL = ("off_path_code (D183; attacker_pool.py: W22 fm2 b926916ff + fm2b 6ae1fe544 (D116): attacker-pool geometry sizing for the multi-turn pools, "
                             "never executed by a static / controls / misalign row)")
ADMITTED_REASON_D183_HELPERS = ("off_path_code (D183; seedrep_ladder_helpers.py: W22 fc3 117475163 + fc4 ba33eb96a (D133): the `expected-samplers` reader command and the "
                                "controls-sampling refactor (a controls-only helper); `misalign-messages` and `_queue_sampler_keys` -- the misalign / static queue-producer "
                                "path -- untouched)")
ADMITTED_REASON_D183_HOOKS = ("registry_emission (D183; registry/hooks.py: W23 fr2 f408d4d1f, W24 fr5 5e0b3c9b0 / f725af8b7 / 6ce849aaa, fr7 c31eed2e5, fr8 30df37038, "
                              "fr9 319b81bfb, cda105326 relu_paired reader: registry emission / recording and a reader -- store bytes, generations and scoring untouched)")
ADMITTED_REASON_D183_SERVED = ("off_path_code (D183; served_capability_run.py: W24 fs4 ac0401136 (F-AS): the inspect-leg (gsm8k) bearer -- `_inspect_api_key` and its two "
                               "call sites in cmd_generate's Inspect branch, never executed for xstest / or_bench / mmlu; fd3 eecba9039 is NOT admitted)")

D183_PETRI_RUNNER_HUNKS: Dict[str, Dict[str, str]] = {
    'src/sisteer/capeval/tiers/petri_runner.py': {
        '5adc2f58aa69789adcb2ef3e1abce02796056833dd20b9d15d6fdc4eeb81aa58': "W22->launch dbf326ead: \"+#: the judge chain's twin (2026-09-19): stamped only when NON-EMPTY, an\" (-0/+3)",
        '575ce4d5805028fb94336755fd2c381e2dfc6c99705804499b9b886f9706ac13': "W22->launch dbf326ead: '+    if spec.judge_fallback_models:' (-0/+5)",
        '8b1052d7d8d3b3a06d7f5b45e8180ec4c79d623f9d057de1ad9bc7eb968a62f9': "W22->launch dbf326ead: '+    jfb_key = JUDGE_FALLBACK_METADATA_KEY' (-0/+1)",
        'e801ea1ae4648051a163a8b481cc718466db34249146a5e6dd0758189fb15901': "W22->launch dbf326ead: '-             if k != fb_key and md.get(k) != v}' (-1/+1)",
        'e6d6bb32eedaea36f66a469941a3e5652974473c54b1e23ebc6b5e63b898ceef': "W22->launch dbf326ead: '+    # the judge chain is compared by the same MEANING rule (absent == [' (-0/+5)",
        '51d2a3e73a12e34b843908fad6dcd8461ebf0a4141eae210f1a4c474493a557c': "W22->launch dbf326ead: \"+    # main: a role is wrapped in the manifest's fallback chain ONLY whe\" (-0/+9)",
        'a1424a0a6ded9172965ed2235f9daa5974503075dacacd03e7d0f946a2f9c4df': "W22->launch dbf326ead: '+    if spec.judge_fallback_models:' (-0/+8)",
        'b89ff9025d3a6e2e77849abc8c4369b9114db7417b24d30024b52e640e821fd6': "W22->launch dbf326ead: '-                \"judge\": spec.judge_model,' (-1/+1)",
        'e89f1670b5a5ea5079f34a6dc4092358e0a6dbd76c39e7124fb14a2a0b1863e2': "W22/W24->launch f29562721: '+#: Opt-out for the re-key guard below. Set to \"1\" only when a fresh mea' (-0/+39)",
        'c4edfb4781659c4be2a7013b85726c9f73c98ade5a8dc7eb79e07756c7be8de8': "W22/W24->launch f29562721: '+    # RE-KEY GUARD: refuse to silently re-audit a cell already complete' (-0/+14)",
    },
}

D183_ATTACKER_POOL_HUNKS: Dict[str, Dict[str, str]] = {
    'src/sisteer/eval/attacks/attacker_pool.py': {
        'b0f652c08427ca88d1d9e9d3b7327b49ddb5c4c7d3c766b6ffa1e5a873cb0e18': "W22->launch 6ae1fe544: '+    #: the KV rule this geometry was fit under. ``kv_seq_tokens`` None ' (-0/+8)",
        'f4c7dfd5df883bb266a76bfd3db10940e338912b3e32f243242812af1dcc7119': "W22->launch 6ae1fe544+b926916ff: '-                f\"{self.max_seqs_fitting}){(\\' \u2014 \\' + self.note) if self.' (-1/+30)",
        '429dd59ff57f5ce84cc64c96e4e32991f6a162acd3ffc1e93dae53fae6794419': "W22->launch 6ae1fe544: '-          reserve_bytes: int = RESERVE_BYTES) -> Tuple[int, int, int, i' (-2/+2)",
        'e1378dc1fc024d2e6c527d69f637160f61484c627ee61b43100a86e90242ab14': "W22->launch 6ae1fe544+b926916ff: '-        arch: AttackerArch = QWEN36_27B, reserve_bytes: int = RESERVE_B' (-1/+2)",
        '5d8e66b1cccf4e792a3061cf1ec14a33a1976dbcc260452a226bc533974d2cd3': "W22->launch 6ae1fe544+b926916ff: '-    communicators need, and the geometry records the value it was fit w' (-1/+9)",
        '88abb025e0efee38615f45af22461dc5188a63d54ace2dbc143bdce59b03bcf4': "W22->launch 6ae1fe544: '+    kv_seq_tokens, kv_rule_name = _resolve_kv_rule(kv_seq_tokens, kv_ru' (-0/+1)",
        '8dd3d0c42c6f4ec121abb03d434221c0c5c199da4f3da7387f11bf2ac64fbcac': "W22->launch 6ae1fe544: '-    kv, state, act, total = _need(arch, L, S, batched, reserve_bytes)' (-1/+1)",
        '6627a765384dc9176635426069436f03d6d8867ba243195658d6319b0a5f5a6c': "W22->launch b926916ff: '-    # the largest S that fits at this window (monotone in S)' (-1/+1)",
        '4b84813dc691b2f8be5536d6f9927040572bcc78c428bef2af5a6bd08296d5a0': "W22->launch 6ae1fe544: '-        if _need(arch, L, s, b, reserve_bytes)[3] <= budget:' (-1/+1)",
        'e8c35b83c4c62d846ff6b511ba970cd402978fb02f744dff466f6801c9d1eecc': "W22->launch 6ae1fe544+b926916ff: '-        max_seqs_fitting=s_max, seqs_source=\"explicit\", arch=arch.name)' (-1/+2)",
        'ce315ccc40c13e872faa69f6d2271521763afa3d32f687a89d6866efde458afc': "W22->launch 6ae1fe544+b926916ff: '-                          reserve_bytes: int = RESERVE_BYTES) -> PoolGe' (-1/+2)",
        '536f2b30c580b7f506b505bf1d9fb2dfbecd7444f2fdd896e0ed8da2df386c71': "W22->launch 6ae1fe544+b926916ff: '-    memory model fits at the window (``seqs_source=\"kv_fit\"``); an expl' (-4/+7)",
        '8ca0de76b290ab236493f4564267dc8c96aafc4e811da72e3953548a98bc7920': "W22->launch b926916ff: '-        probe = fit(max_model_len, 1, util=util, chunked_prefill=chunke' (-2/+1)",
        '5a4ca282281d2ba1fc7427f42477ed4909c1a0753fad598f4cf37a07d7c24af9': "W22->launch b926916ff: '-        g = fit(max_model_len, probe.max_seqs_fitting, util=util,' (-3/+1)",
        'e2c444163f0dbd90580858c2411f1b2c8a2ab17e98f17eb263e095747417094f': "W22->launch b926916ff: '-    g = fit(max_model_len, max_num_seqs, util=util, chunked_prefill=chu' (-2/+1)",
        '34774acb458faf346ca3a838b0b4d75ca23b06deac27e8139a0c26ad89f8dc4a': "W22->launch 6ae1fe544: '+        rule = (\"S x L identity\" if g.kv_seq_tokens is None' (-0/+2)",
        'ca0a1ec2b1314b24935dbcf08b2a8a32c8dbd3b8ed3dc878c90fbe00cf9803d3': "W22->launch b926916ff: '-            g.describe() + f\" \u2014 refusing: set max_num_seqs <= {g.max_se' (-1/+1)",
        '4f60f0e7f0ea55bd9e7548e0d1e5b94468ded7efa604d31bccce51503b00067b': "W22->launch 6ae1fe544+b926916ff: '-def coverage_problem(cov: Dict[str, Any], *, max_num_seqs: int, max_mod' (-3/+33)",
        '75351d832ea6f6e8d1843c452b1c0eeb08f6377f431e3cbebedc53109cd2ad1c': "W22->launch 6ae1fe544+b926916ff: '-    if conc + 1e-9 < max_num_seqs:' (-5/+18)",
    },
}

D183_LADDER_HELPERS_HUNKS: Dict[str, Dict[str, str]] = {
    'scripts/modal/seedrep_ladder_helpers.py': {
        'f24fc31712071d744cfa4f1fe9e656c79a854853258035dbf135d903816494d2': "W22->launch 117475163: '+  expected-samplers --row <plan row json> --tree <W checkout> --out <js' (-0/+8)",
        '12ff38611397e7e5f967bdeb0f113f3aa9d9878213db22fcc94cb6b1793ed364': "W22->launch 117475163: '-def controls_sampling(config: Path, out: Path, *, gen_of=None) -> int:' (-7/+2)",
        'cde6b8245c8dc5405e17cd45f002f670d819227d1f7f548743a36dd894865bb0': "W22->launch 117475163: '-        sys.stderr.write(f\"controls-sampling: {config}: controls_gen_co' (-2/+1)",
        '2c7775ba650588b559a81085778d7195e16f32b8d7fe61a92c6cd03175a7a04b': "W22->launch 117475163+ba33eb96a: '-        sys.stderr.write(f\"controls-sampling: {config}: the controls de' (-2/+170)",
        'c847f0572e9fdc4a680a736aa4187ecfbc6301c13b2649d514e05da54061f0ea': "W22->launch 117475163: '+    e = sub.add_parser(\"expected-samplers\")' (-0/+4)",
        'a2686cf77b2e068e65b1c7fd618c46c1ddec310ca29dca7af91f86ce537c7861': "W22->launch 117475163: '+    if a.cmd == \"expected-samplers\":' (-0/+10)",
    },
}

D183_REGISTRY_HOOKS_HUNKS: Dict[str, Dict[str, str]] = {
    'src/sisteer/registry/hooks.py': {
        'd521f9ac69cb564ba13801a83fbec6c90d0ef386f1baaef7fe2518d88cc0bf96': "W22->launch 6ce849aaa+f725af8b7: '+#: fr5 (W24, 2a): the metric id of the null-valued ``item_result`` a ju' (-0/+10)",
        '6c6a6539051f9e5fcecdfb91b28a1cf329b24851cd6c12e069d0072174b81b7d': "W22->launch f725af8b7: '+        self._metrics: Dict[str, List[Dict[str, object]]] = {}     # ro' (-0/+1)",
        '538a92e9a0c2284b5ee54c42cd491efbbc22d2f2296de95cf9f34f9297bda4fe': "W22/W24->launch 319b81bfb+c31eed2e5: '-        \"\"\"``run`` (running) + ``run_state running`` for one route\\'s ru' (-2/+12)",
        '4c6368e7ae74bbc74429741061e1fbe074f04e80a1c6576beb2cc18c42c29809': "W22/W24->launch c31eed2e5: '-        w.emit_run(payload)' (-1/+2)",
        '74b4a66971a6e17a4a37467436434890e7f9df4040d260628db465a00c46cdaa': "W22/W24->launch 30df37038+c31eed2e5: '+    @staticmethod' (-0/+30)",
        '06ede4c491d3401a8bc4c0fdc58d10323443d40eaa13f0e4897d075db72fc69e': "W22->launch f725af8b7: '+        if item == AGGREGATE_ITEM_ID or metric.endswith(AGGREGATE_SUFFI' (-0/+7)",
        'a28e5f4f61786cfdbbba3393f0a0c2f9e4e203be1c841ec8bc6f296085c0f7f6': "W22->launch 6ce849aaa: '+    def on_generated(self, task: str, item_ids: Sequence[str], uri: str' (-0/+34)",
        '1f6b4d5c813c6bb8871cd5653ae70a7ae2a0a59c47f50b59fb619e72e591e2ae': "W22->launch f725af8b7: '-        routes = [route] if route else list(self._items) + [r for r in ' (-1/+1)",
        'dfdbb73ef80a22c6021805affaf100b34f77ea8590d30f3c4a0c665ef88f9a70': "W22->launch f725af8b7: '-            if rows or flags:' (-1/+4)",
        'f828431fb01e0906c3da29da2e5ed7f077fbd33261c207c20007f2fbc23a0403': "W22->launch f725af8b7: '-            if any(self._items.values()) or any(self._flags.values()):' (-1/+1)",
        '700b37512939cc953e68ca7e2617683bd9fe5a338f6ca4e5d36631419905e1b5': "W22->launch f725af8b7: '+def aggregate_metric_row(run_id: str, metric: str, value: Optional[floa' (-0/+21)",
        '0a21b27283b2101bce30be4c6d3450e1194dece7a54d984444fc521d83af0b72': "W22->launch 5e0b3c9b0: '-                             session: Optional[str] = None, extra: Opti' (-1/+2)",
        '5f86196379c6e17aea707758c89d6cda6e60af4245da3bcd3e1e5f185093a18c': "W22->launch 5e0b3c9b0: '-    producer = emit.producer_block(script, session=session)' (-1/+1)",
        '86870cfe6f572adc3c3a0836d022d6b5f408421a5075a56bfaff1d3e2d588ca5': "W22->launch f408d4d1f: '-    the launching shell from the planned runs; None when absent.\"\"\"' (-1/+4)",
        'c94736b95ee2f00fe38a49a7665e4cb315d04af67c26b9408040816a8ace024b': "W22->launch f408d4d1f: '-    for served_key, run in runs.items():' (-13/+10)",
        '23dc86d7be533baf183a8a6e260d077df3a3c5b64f96611a137f09e91e1aa609': "W22->launch f408d4d1f: '+def _serving_patch_for(endpoint: Mapping[str, Any], served_key: str, ru' (-0/+14)",
        '54f6ee25af72deb59db5746d0acdb23d4dcadbca8a7a681a915c00fe3180e249': "W22/W24->launch cda105326: '+    if object_kind == \"relu_steer_distilled_difference\" and role == \"di' (-0/+4)",
    },
}

D183_SERVED_FS4_HUNKS: Dict[str, Dict[str, str]] = {
    'scripts/served_capability_run.py': {
        '0d9013b2700cc92b638df70639b7467fef49321a45fab8f593659a7673ed42a2': "W24->launch ac0401136: '-                                           load_routing_table)' (-1/+2)",
        '1bacf2f048a3b7266206bf6d3b5e504ad5f5c7ad6d36ea69c2eeab708ca0c67b': "W24->launch ac0401136: '+def _inspect_api_key(standup_endpoint: \"dict | None\",' (-0/+33)",
        '13f14c7e8a76667fe907fec079db04e19361ccf6596a3f7a81fa4a0bf27824bc': "W24->launch ac0401136: '-                    api_key=os.environ.get(tc.INSPECT_API_KEY_ENV),' (-1/+1)",
        'fef036b439346aca0d57c850c6fdb4878c5346983ef35369b7e97159794303a1': "W24->launch ac0401136: '-                api_key=os.environ.get(tc.INSPECT_API_KEY_ENV),' (-1/+1)",
    },
}

#: the GSM8K landing cut (the assembly plus a960b9ea8 / 86277ee7f) adds a narrowly scoped off-path admission for the carry-dependent tree:
#: the controls carry (...-dc4bd2f90303) re-derived against the merged tree refuses at clause (b) on exactly two producer files those commits
#: changed. Each table pins EVERY zero-context hunk of the real (prior tree -> merged tree; the two prior trees agree on these files) diff of
#: one file by content hash, labelled with the commit(s) `git blame` attributes its added lines to; the
#: tables are restricted to the CONTROLS leg (a gsm8k carry, whose runner inspect_served.py IS, never sees them). The other modules those
#: commits touch (audit_common.py, record_resolved_sampler.py, tinycap/contracts.py) are NON_EVAL_MODULES -- readers / probes, not producers.
ADMITTED_REASON_D194_INSPECT_SERVED = ("off_path_code (D194-fg; inspect_served.py: fg1 a960b9ea8 + fg2 86277ee7f: five hunks, all inside run_one_task -- the tinycap done "
                                       "marker's n_planned / samples_per_item accounting and the per-sample filed-row count (F-AU); the archived W24 controls children request "
                                       "xstest, or_bench_hard_1k and mmlu through served_capability_run.py's controls path, which never enters this Inspect (gsm8k) runner; a "
                                       "controls row's generations and scoring cannot change)")
ADMITTED_REASON_D194_COMPLETION = ("off_path_code (D194-fg; completion.py: fg1 a960b9ea8: three hunks in finish_task -- the OPTIONAL n_planned field on the done marker; every "
                                   "existing caller omits n_planned, so the base and current completion payloads are byte-identical; a controls row's generations and scoring "
                                   "cannot change)")
D194_INSPECT_SERVED_HUNKS: Dict[str, Dict[str, str]] = {
    'src/sisteer/capeval/tiers/inspect_served.py': {
        '6c1f29619fdc1e57ff11e4e03aeda5bca0e0d005a4882b6728cd18a3376fbe81': "W22/W24->launch fg1 a960b9ea8 (blame also 0309c7677 / 5d18576a7: moved lines): '+            elif plan.samples_per_item > 1:' (-0/+11)",
        'aa690c7becca7e0a3ea28284e1fd33355407b827eb73e162173deaad98be8754': "W22/W24->launch fg1 a960b9ea8: '+            #' (-0/+17)",
        '523f8c80ff1c0c67f848969e4a6acbfe3e382ec81f6a627aaa11a7e8623693a7': "W22/W24->launch fg1 a960b9ea8: '-        # under a done marker by way of this count.' (-1/+3)",
        'd6c4a1ee072b187ab3f93f5761886f3db1ed1f56d10b0036c13bf74adfac5466': "W22/W24->launch fg1 a960b9ea8: '-        # the per-item / per-sample rows actually FILED (an unscored ' (-3/+4)",
        '5771f868e3fe59b2431ae2d9361e24396d443cbfd75c46f9f0d7bd519d2e0f66': "W22/W24->launch fg1 a960b9ea8: \"+        # F-AU (D194): the plan's count on the done marker beside the\" (-0/+2)",
    },
}
D194_COMPLETION_HUNKS: Dict[str, Dict[str, str]] = {
    'src/sisteer/capeval/tinycap/completion.py': {
        '8224c21ebc5ebf59eb85da91ece37822819351aebf7fa7145f1ba67834a1409d': "W22/W24->launch fg1 a960b9ea8: '-                n_expected_draws: int | None = None) -> dict[str, Any' (-1/+2)",
        'b3cfdbc5ff3d2588763dfb78f6c0882a7a310910b3030adb7d8ec4b965dadbb0': "W22/W24->launch fg1 a960b9ea8 (docstring; blame spans older commits): '-    passes neither and writes a byte-identical marker.\"\"\"' (-1/+8)",
        'd030635ae028f58d64a5184a044bfc0e31cdc77cd011d500ccb1cc1e989a8be2': "W22/W24->launch fg1 a960b9ea8: '+        if n_planned is not None:' (-0/+3)",
    },
}
D194_FG_LEGS: frozenset = frozenset({"controls"})


ADMITTED_HUNK_TABLES: Tuple[Tuple[str, Dict[str, Dict[str, str]], str, str, Optional[frozenset]], ...] = (
    ("D111", D111_ADMITTED_HUNKS, ADMITTED_REASON_D111, ADMITTED_KIND_CHAIN, None),
    ("D113", D113_ADMITTED_HUNKS, ADMITTED_REASON_D113, ADMITTED_KIND_CHAIN, None),
    ("D120", D120_ADMITTED_HUNKS, ADMITTED_REASON_D120, ADMITTED_KIND_CHAIN, None),
    ("D136", D136_ADMITTED_HUNKS, ADMITTED_REASON_D136, ADMITTED_KIND_REGISTRY, None),
    ("D137", D137_ADMITTED_HUNKS, ADMITTED_REASON_D137, ADMITTED_KIND_PLUMBING, None),
    ("D138", D138_ADMITTED_HUNKS, ADMITTED_REASON_D138, ADMITTED_KIND_OTHER_LEG, frozenset({"controls"})),
    # one table per file, each pinned to the real prior -> launch diff; see the block above
    ("D183-petri_runner", D183_PETRI_RUNNER_HUNKS, ADMITTED_REASON_D183_PETRI, ADMITTED_KIND_OFF_PATH, D183_LEGS_STATIC_CONTROLS_MISALIGN),
    ("D183-attacker_pool", D183_ATTACKER_POOL_HUNKS, ADMITTED_REASON_D183_POOL, ADMITTED_KIND_OFF_PATH, D183_LEGS_STATIC_CONTROLS_MISALIGN),
    ("D183-ladder_helpers", D183_LADDER_HELPERS_HUNKS, ADMITTED_REASON_D183_HELPERS, ADMITTED_KIND_OFF_PATH, frozenset(lp.STATIC_LEGS) | frozenset({"misalign"})),
    ("D183-registry_hooks", D183_REGISTRY_HOOKS_HUNKS, ADMITTED_REASON_D183_HOOKS, ADMITTED_KIND_REGISTRY, D183_LEGS_STATIC_CONTROLS_MISALIGN),
    ("D183-served_fs4", D183_SERVED_FS4_HUNKS, ADMITTED_REASON_D183_SERVED, ADMITTED_KIND_OFF_PATH, frozenset({"controls"})),
    # the GSM8K landing cut; controls carry only
    ("D194-fg-inspect_served", D194_INSPECT_SERVED_HUNKS, ADMITTED_REASON_D194_INSPECT_SERVED, ADMITTED_KIND_OFF_PATH, D194_FG_LEGS),
    ("D194-fg-completion", D194_COMPLETION_HUNKS, ADMITTED_REASON_D194_COMPLETION, ADMITTED_KIND_OFF_PATH, D194_FG_LEGS),
)


def pinned_hunks(path: str, leg: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    """``{hunk sha256: (decision, label)}`` over every table of :data:`ADMITTED_HUNK_TABLES` for ``path`` -- a leg-restricted table only when
    ``leg`` is one of its legs (a carry that names no leg never sees a restricted table)."""
    out: Dict[str, Tuple[str, str]] = {}
    for name, table, _reason, _kind, legs in ADMITTED_HUNK_TABLES:
        if legs is not None and (leg is None or leg not in legs):
            continue
        for h, label in (table.get(path) or {}).items():
            out[h] = (name, label)
    return out


def d111_admitted_difference(a: str, b: str, path: str, leg: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The pinned-hunk kind (every table of :data:`ADMITTED_HUNK_TABLES`): every hunk of ``a -> b`` is in :func:`pinned_hunks` for ``path``
    (and ``leg``, for a leg-restricted table); else None. The record names each hunk's table, so a reader sees which ruling admitted what."""
    pin = pinned_hunks(path, leg)
    if not pin or a == b:
        return None
    hunks = diff_hunks(a, b)
    if not hunks or any(h["sha256"] not in pin for h in hunks):
        return None
    tables = sorted({pin[h["sha256"]][0] for h in hunks})
    reasons = {name: reason for name, _t, reason, _k, _l in ADMITTED_HUNK_TABLES}
    kinds = {name: kind for name, _t, _r, kind, _l in ADMITTED_HUNK_TABLES}
    reason = " + ".join(reasons[t] for t in tables)
    kind = "+".join(sorted({kinds[t] for t in tables}))
    return {"path": path, "kind": kind, "reason": reason, "tables": tables,
            "hunks": [{**h, "table": pin[h["sha256"]][0], "label": pin[h["sha256"]][1]} for h in hunks], "n_hunks": len(hunks), "removed": [], "added": []}


def admitted_difference(git: Callable[[Sequence[str]], Tuple[int, str]], prev_sha: str, this_sha: str, path: str, leg: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The ONE admitted KIND of producer-file difference (`petri_fallback_chain`): (i) every changed line is a
    member of an auditor / judge `*_fallback_models` list (YAML `- item` under the key, or a quoted item inside a Python default list), or (ii)
    for :data:`D111_ADMITTED_FILES`, every changed hunk is in the pinned hunk set (:func:`d111_admitted_difference`). Returns the record
    (the changed lines / hunks, the reason) or None when the difference is anything else / cannot be read."""
    import difflib
    rc_a, a = git(["show", f"{prev_sha}:{path}"])
    rc_b, b = git(["show", f"{this_sha}:{path}"])
    if rc_a != 0 or rc_b != 0 or a == b:
        return None
    if pinned_hunks(path, leg):
        adm = d111_admitted_difference(a, b, path, leg)
        if adm is not None:
            return adm
        # a file with pinned hunks whose diff is NOT the pinned set falls through to the chain-list kind (manifest.py: a pure chain-item change)
    al, bl = a.splitlines(), b.splitlines()

    def in_chain_block(lines: List[str], idx: int) -> bool:
        j = idx - 1
        while j >= 0:
            if FALLBACK_CHAIN_KEY_RE.match(lines[j]):
                return True
            if lines[j].strip() and not FALLBACK_CHAIN_ITEM_RE.match(lines[j]):
                return False
            j -= 1
        return False
    def line_kind(lines: List[str], idx: int) -> Optional[str]:
        """`item` (a member of a chain list), `key` (a `*_fallback_models` key line -- re-spelled with a trailing comment, or the judge chain's key
        added), `comment` (a pure comment or blank line: inert in YAML / Python), else None (anything else is the eval's identity)."""
        line = lines[idx]
        if FALLBACK_CHAIN_KEY_RE.match(line):
            return "key"
        if not line.strip() or line.strip().startswith("#"):
            return "comment"
        if FALLBACK_CHAIN_ITEM_RE.match(line) and in_chain_block(lines, idx):
            return "item"
        return None
    changed: List[Dict[str, Any]] = []
    sm = difflib.SequenceMatcher(a=al, b=bl, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for i in range(i1, i2):
            k = line_kind(al, i)
            if k is None:
                return None
            changed.append({"side": "prev", "line": al[i].strip(), "kind": k})
        for j in range(j1, j2):
            k = line_kind(bl, j)
            if k is None:
                return None
            changed.append({"side": "this", "line": bl[j].strip(), "kind": k})
    if not changed or not any(c["kind"] == "item" for c in changed):
        return None                       # a comment-only / key-only change is not a chain difference; the byte check decides it as before
    items = [c for c in changed if c["kind"] == "item"]
    return {"path": path, "kind": ADMITTED_KIND_CHAIN, "reason": ADMITTED_REASON_CHAIN, "changed_lines": changed,
            "removed": sorted(c["line"] for c in items if c["side"] == "prev"), "added": sorted(c["line"] for c in items if c["side"] == "this"),
            "keys_changed": sorted({c["line"].split(":")[0].strip().strip('"') for c in changed if c["kind"] == "key"}),
            "n_comment_lines": sum(1 for c in changed if c["kind"] == "comment")}


def producer_set_verdict(files: Sequence[str], *, prev_sha: str, this_sha: str, git: Callable[[Sequence[str]], Tuple[int, str]],
                         container_modules: Sequence[str] = CONTAINER_MODULES, leg: Optional[str] = None) -> Dict[str, Any]:
    """Clause (b): every producer file's blob at both shas (absent at either refuses), `git diff --quiet` over the set (rc recorded),
    and the container modules' blobs recorded beside (informational)."""
    non_eval = sorted(set(files) & set(NON_EVAL_MODULES))
    files = sorted(set(files) - set(NON_EVAL_MODULES))
    prev_b, this_b = _blobs_at(git, prev_sha, files), _blobs_at(git, this_sha, files)
    per = [{"path": f, "prev_blob": prev_b[f], "this_blob": this_b[f], "identical": (prev_b[f] is not None and prev_b[f] == this_b[f])} for f in files]
    absent = [p["path"] for p in per if p["prev_blob"] is None or p["this_blob"] is None]
    differing = [p["path"] for p in per if p["prev_blob"] is not None and p["this_blob"] is not None and not p["identical"]]
    admitted: List[Dict[str, Any]] = []
    changed: List[str] = []
    for f in differing:
        adm = admitted_difference(git, prev_sha, this_sha, f, leg)
        if adm is None:
            changed.append(f)
        else:
            adm = {**adm, "prev_blob": prev_b[f], "this_blob": this_b[f]}          # fj25: the blobs the admission was derived over
            admitted.append(adm)
            next(p for p in per if p["path"] == f)["admitted_difference"] = adm["reason"]
    cmd = ["diff", "--quiet", prev_sha, this_sha, "--", *files]
    rc, _ = git(cmd) if files else (0, "")
    cm_prev, cm_this = _blobs_at(git, prev_sha, list(container_modules)), _blobs_at(git, this_sha, list(container_modules))
    cm = [{"path": f, "prev_blob": cm_prev[f], "this_blob": cm_this[f], "identical": (cm_prev[f] is not None and cm_prev[f] == cm_this[f])} for f in container_modules]
    ne_prev, ne_this = _blobs_at(git, prev_sha, non_eval), _blobs_at(git, this_sha, non_eval)
    ne = [{"path": f, "prev_blob": ne_prev[f], "this_blob": ne_this[f], "identical": (ne_prev[f] is not None and ne_prev[f] == ne_this[f])} for f in non_eval]
    ok = bool(files) and not absent and not changed and (rc == 0 or (rc == 1 and bool(admitted)))
    out = {"ok": ok, "prev_sha": prev_sha, "this_sha": this_sha, "files": per, "n_files": len(files), "git_diff_quiet": {"cmd": ["git", *cmd], "rc": rc},
           "admitted_differences": admitted, "non_eval_modules": ne,
           "non_eval_rule": "user ruling 18:55Z: probes / certification checks / evidence writers are not the store's producers; excluded by the explicit NON_EVAL_MODULES list, recorded, re-derived under clause (c)",
           "container_modules": cm, "container_modules_rule": "readers by ruling: recorded, not gating (clauses (c)/(d) re-judge their products; the ladder queue is clause (e))"}
    if not files:
        out["why"] = "no producer file could be derived from the row (no tree path in its argv / children / evidence)"
    elif absent:
        out["why"] = f"producer file(s) absent at one of the shas: {absent[:4]}"
    elif changed:
        out["why"] = f"producer file(s) differ between {prev_sha[:12]} and {this_sha[:12]}: {changed[:4]}{' ...' if len(changed) > 4 else ''} (git diff --quiet rc {rc}; not an admitted difference)"
    elif rc not in (0, 1):
        out["why"] = f"git diff --quiet failed (rc {rc})"
    return out


# --------------------------------------------------------------------------- #
# clause (c): COMPLETE under this tree's readers                                #
# --------------------------------------------------------------------------- #

def criterion3_waiver_problems(w: Any, sid: str = "") -> List[str]:
    """Why ``w`` is not a determinism waiver document: an object with non-empty `by` / `at` / `reason` / `decision` and a `scope`
    `{W (sha or >= 7-hex prefix), leg, view, legs: [int, ...]}` -- an explicit operator input, never inferred."""
    who = f"{sid}: " if sid else ""
    if not isinstance(w, Mapping):
        return [f"{who}criterion3_waiver is not an object"]
    out = [f"{who}criterion3_waiver.{k} missing or empty" for k in CRITERION3_WAIVER_KEYS if k != "scope" and (not isinstance(w.get(k), str) or not w.get(k).strip())]
    sc = w.get("scope")
    if not isinstance(sc, Mapping):
        return out + [f"{who}criterion3_waiver.scope is not an object"]
    for k in ("W", "leg", "view"):
        if not isinstance(sc.get(k), str) or not sc.get(k).strip():
            out.append(f"{who}criterion3_waiver.scope.{k} missing or empty")
    if isinstance(sc.get("W"), str) and not re.fullmatch(r"[0-9a-f]{7,40}", sc["W"].strip()):
        out.append(f"{who}criterion3_waiver.scope.W {sc.get('W')!r} is not a tree sha (>= 7 hex)")
    legs = sc.get("legs")
    if not isinstance(legs, list) or not legs or any(isinstance(x, bool) or not isinstance(x, int) for x in legs):
        out.append(f"{who}criterion3_waiver.scope.legs must be a non-empty list of leg indices")
    return out


#: the registry's site alias module once loaded (``[module]``), or ``[None]`` where the package is not importable
_SITE_ALIASES: List[Any] = []


def _site_aliases() -> Any:
    """``sisteer.registry.site_aliases`` with the table of ``$REGISTRY_ROOT`` (else ``~/registry``) installed, loaded once per process;
    None where the package is not importable (this module is stdlib-only and also runs in a container beside the plan module). A
    configuration the registry refuses is raised, never skipped."""
    if not _SITE_ALIASES:
        try:
            from sisteer.registry import config as CFG  # noqa: PLC0415
            from sisteer.registry import site_aliases as SA  # noqa: PLC0415
        except ImportError:
            _SITE_ALIASES.append(None)
        else:
            root = os.path.abspath(os.environ.get("REGISTRY_ROOT") or os.path.join(os.path.expanduser("~"), "registry"))
            SA.install(CFG.load(root))
            _SITE_ALIASES.append(SA)
    return _SITE_ALIASES[0]


def drive_block(child: Any) -> Optional[Mapping[str, Any]]:
    """A Petri child's grading-host drive block (``grading_host_drive``): a launch plan written before the site renamed its grading
    host spells the key under the older name, which the site alias table maps (:func:`_site_aliases`); where the package is not
    importable only the current key is read. None when the child carries no block."""
    if not isinstance(child, Mapping):
        return None
    sa = _site_aliases()
    dd = sa.get_compat(child, "grading_host_drive") if sa is not None else child.get("grading_host_drive")
    return dd if isinstance(dd, Mapping) else None


def row_petri_legs(row: Mapping[str, Any]) -> List[int]:
    """The leg (cell) indices a row's children spell (the drive block's ``cell_index`` on a Petri child, :func:`drive_block`), else the
    child ordinals."""
    kids = row.get("children") or []
    out = []
    for i, k in enumerate(kids):
        dd = drive_block(k)
        out.append(int(dd["cell_index"]) if dd is not None and dd.get("cell_index") is not None else i)
    return sorted(set(out))


def waiver_covers(w: Mapping[str, Any], *, W: str, leg: str, view: str, legs: Sequence[int]) -> bool:
    """The waiver's scope names THIS (tree, leg, view) and every one of ``legs``: `scope.W` equals ``W`` or one is a >= 7-hex prefix of the other."""
    if criterion3_waiver_problems(w):
        return False
    sc = w["scope"]
    sw, W = sc["W"].strip().lower(), str(W).lower()
    w_ok = sw == W or (len(sw) >= 7 and W.startswith(sw)) or (len(W) >= 7 and sw.startswith(W))
    return bool(w_ok and sc["leg"] == leg and sc["view"] == view and set(int(x) for x in legs) <= set(int(x) for x in sc["legs"]))


def covering_waiver(waivers: Optional[Sequence[Mapping[str, Any]]], *, W: str, row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The first supplied waiver whose scope covers ``row`` of tree ``W`` (its leg, engine / view and every leg index), else None."""
    legs = row_petri_legs(row)
    for w in waivers or ():
        if isinstance(w, Mapping) and waiver_covers(w, W=W, leg=str(row.get("leg")), view=str(row.get("engine")), legs=legs):
            return w
    return None


REAUDIT_KEYS: Tuple[str, ...] = ("reaudited_by", "evidence_root", "producer_evidence", "rc", "argv", "store_root")


def reaudit_problems(ra: Any, *, this_sha: str) -> List[str]:
    """Why ``ra`` is not a re-audit record of THIS tree (fj7): the keys, `reaudited_by == this tree`, a volume evidence root, rc 0."""
    if not isinstance(ra, Mapping):
        return ["reaudit: not an object"]
    out = [f"reaudit: {k} missing" for k in REAUDIT_KEYS if k not in ra]
    if ra.get("reaudited_by") != this_sha:
        out.append(f"reaudit: reaudited_by {str(ra.get('reaudited_by'))[:12]!r} is not this tree {this_sha[:12]}")
    if not (isinstance(ra.get("evidence_root"), str) and ra["evidence_root"].startswith(lp.VOL_PATH + "/")):
        out.append(f"reaudit: evidence_root {ra.get('evidence_root')!r} is not a volume path")
    if ra.get("rc") != 0:
        out.append(f"reaudit: the audit tool exited rc {ra.get('rc')!r}, not 0")
    if not isinstance(ra.get("producer_evidence"), Mapping) or not ra["producer_evidence"]:
        out.append("reaudit: no producer_evidence bindings")
    return out


def resolved_decode_temperatures(doc: Any) -> Optional[List[float]]:
    """fj22: the CLIENT class's resolved sampling temperatures from a `resolved_sampler.json` record (`distinct_resolved[*].sampler.temperature`,
    the writer's top-level list, else `classes.client.distinct_resolved`); None when the record states none -- the decode is then unknown and
    the determinism spot-check applies in full (fail closed)."""
    if not isinstance(doc, Mapping):
        return None
    rows = doc.get("distinct_resolved")
    if not isinstance(rows, list) or not rows:
        client = (doc.get("classes") or {}).get("client") if isinstance(doc.get("classes"), Mapping) else None
        rows = client.get("distinct_resolved") if isinstance(client, Mapping) else None
    if not isinstance(rows, list) or not rows:
        return None
    out: List[float] = []
    for r in rows:
        t = ((r.get("sampler") or {}) if isinstance(r, Mapping) else {}).get("temperature")
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            return None
        out.append(float(t))
    return out


#: which record the determinism exemption's decode was read from
DECODE_AUTHORITY_REAUDIT = "reaudit_rederived"                    # the FRESH record this tree's recorder derived, under the re-audit evidence root (fc11d)
DECODE_AUTHORITY_HISTORICAL = "historical_producer_binding"       # the attempt's own record, its producer binding verified by the resolved_sampler check


def verified_resolved_decode(sampler_check: Any, *, root: str, prev_row: Mapping[str, Any], evidence: Mapping[str, Any],
                             read_text: Callable[[str], Optional[str]]) -> Dict[str, Any]:
    """fj23: the row's resolved decode temperatures from the SAME record the resolved-sampler check VERIFIED for this row -- its
    recorded `sources[*].path` re-read and held to the `sha256` the check hashed (under fc11d the fresh re-derived record in the re-audit evidence
    root, bound through the re-audit's producer_evidence; without one the producer-bound historical record). When the fresh record was the
    authority the HISTORICAL producer-bound record is cross-checked too: its binding must verify and its temperatures must agree -- a tampered
    historical record (bytes changed, digest unchanged) or a disagreement refuses. ``{"ok": False, "why"}`` when no verified source exists."""
    if not isinstance(sampler_check, Mapping) or not isinstance(sampler_check.get("sources"), list) or not sampler_check["sources"]:
        return {"ok": False, "why": "no verified resolved-sampler source: the resolved_sampler check recorded none for this row"}
    authority = DECODE_AUTHORITY_REAUDIT if sampler_check.get("reaudit") else DECODE_AUTHORITY_HISTORICAL
    temps: List[float] = []
    sources: List[Dict[str, Any]] = []
    for fact in sampler_check["sources"]:
        path, sha = (fact.get("path"), fact.get("sha256")) if isinstance(fact, Mapping) else (None, None)
        if not isinstance(path, str) or not isinstance(sha, str):
            return {"ok": False, "why": "the resolved_sampler check's source facts name no path / sha256"}
        text = read_text(path)
        if text is None or _sha_text(text) != sha:
            return {"ok": False, "why": f"the verified resolved-sampler record {path} no longer reads the bytes the check verified (sha {sha[:12]})"}
        try:
            ts = resolved_decode_temperatures(json.loads(text))
        except ValueError:
            ts = None
        if ts is None:
            return {"ok": False, "why": f"the verified resolved-sampler record {path} states no client decode temperature"}
        temps += ts
        sources.append({"path": path, "sha256": sha, "authority": authority})
    cross: Optional[Dict[str, Any]] = None
    if authority == DECODE_AUTHORITY_REAUDIT:
        pm = lp.producer_map_for("resolved_sampler", evidence)
        hist_paths = lp.smoke_evidence_paths(prev_row, root, pm)["resolved_sampler"]
        hist: List[Dict[str, Any]] = []
        htemps: List[float] = []
        for hp in hist_paths:
            ht = read_text(hp)
            rel = hp[len(root) + 1:] if hp.startswith(root + "/") else hp
            b = pm.get(rel) if isinstance(pm, Mapping) else None
            if ht is None:
                return {"ok": False, "why": f"the historical resolved-sampler record {hp} the producer bound is absent -- the fresh record cannot be cross-checked"}
            hsha = _sha_text(ht)
            if not isinstance(b, Mapping) or b.get("sha256") != hsha:
                return {"ok": False, "why": f"the historical resolved-sampler record {hp} is not the bytes its producer bound (sha {hsha[:12]} != bound "
                                            f"{str((b or {}).get('sha256'))[:12]}) -- tampered evidence never grounds the D180 exemption"}
            try:
                hts = resolved_decode_temperatures(json.loads(ht))
            except ValueError:
                hts = None
            if hts is None:
                return {"ok": False, "why": f"the historical resolved-sampler record {hp} states no client decode temperature -- the fresh record cannot be cross-checked"}
            htemps += hts
            hist.append({"path": hp, "sha256": hsha})
        if sorted(htemps) != sorted(temps):
            return {"ok": False, "why": f"the re-derived resolved decode (temperatures {sorted(temps)}) and the historical producer-bound one ({sorted(htemps)}) disagree"}
        cross = {"historical": hist, "agrees": True}
    return {"ok": True, "temperatures": temps, "source": sources, "authority": authority, "cross_check": cross}


class D180Undeterminable(ValueError):
    """A PRESENT, DIVERGED spot-check record whose exemption cannot be judged -- no VERIFIED resolved decode for the row, or the
    record's bytes are not the ones its producer bound -- refuses by name; nothing is relaxed."""


def determinism_d180_relaxation(text: str, *, path: str, root: str, producer: Mapping[str, Any], sampler_check: Any, prev_row: Mapping[str, Any],
                                evidence: Mapping[str, Any], read_text: Callable[[str], Optional[str]]) -> Optional[Dict[str, Any]]:
    """The ONE qualification of a determinism spot-check record, shared by the carry-over's clause (c)
    (`complete_verdict`) and the smoke-PASS derivations (`seedrep_launch_plan._determinism_source`: `derive_smoke_pass`, `derive_smoke_pass_from_priors`
    and their certify-time re-checks).

    A PRESENT record whose replay did not reproduce byte-for-byte (`reproduces: false`, `identical < n_requests`) on a SAMPLED row is a QUALIFICATION,
    not a refusal: byte-identical replay is not a sampled decode's guarantee -- its pinned decode identity is (resolved_sampler AGREES on the verified
    record). The decode is read from the SAME resolved-sampler record the row's resolved_sampler check VERIFIED (`verified_resolved_decode`: the
    fresh re-derived record under fc11d's re-audit, else the producer-bound historical one; the historical record cross-checked when the fresh one is
    the authority). Returns None when ``text`` is not such a record (it reproduces, or lacks the counts) OR the verified decode is GREEDY (any
    temperature 0): the determinism spot-check then applies in full and the caller's check refuses the real record by name. Otherwise returns the relaxation
    ``{"text": <the record with its byte-identity assertion satisfied>, "producer": <the binding re-pointed at those bytes>, "diverged": <the
    qualification record>, "real_sha": <the real bytes' digest>}`` -- the caller runs its check over ``text`` / ``producer`` (every OTHER property
    of the record -- the writer's schema, the table leg, counts, the row it is of -- is still held) and stamps the REAL digest and counts back on
    the facts. Raises :class:`D180Undeterminable` when the exemption cannot be judged (no verified decode) or the bytes are not the producer's."""
    try:
        ddoc = json.loads(text)
    except ValueError:
        return None
    if not (isinstance(ddoc, Mapping) and ddoc.get("reproduces") is False and isinstance(ddoc.get("n_requests"), int) and isinstance(ddoc.get("identical"), int)):
        return None
    decode = verified_resolved_decode(sampler_check, root=root, prev_row=prev_row, evidence=evidence, read_text=read_text)
    if not decode["ok"]:
        raise D180Undeterminable(f"{path}: {ddoc['identical']} of {ddoc['n_requests']} re-sent request(s) reproduced byte-identically -- "
                                 f"criterion 3 requires every one, and the D180 exemption cannot be judged: {decode['why']}")
    temps = decode["temperatures"]
    if not (temps and all(t > 0 for t in temps)):
        return None
    diverged = {"verdict": CRITERION3_NOT_APPLICABLE_SAMPLED, "ruling": CRITERION3_RULING, "reproduced": ddoc["identical"], "resent": ddoc["n_requests"],
                "misses": list(ddoc.get("misses") or []), "record": path,
                "decode": {"temperatures": temps, "source": decode["source"], "authority": decode["authority"], "cross_check": decode["cross_check"],
                           "why": "sampled decode (temperature > 0): byte-identical replay is not the row's guarantee -- its pinned decode identity is "
                                  "(resolved_sampler AGREES on the verified record)"}}
    real_sha = _sha_text(text)
    rel = path[len(root) + 1:] if path.startswith(root + "/") else path
    bound = producer.get(rel) if isinstance(producer, Mapping) else None
    if not isinstance(bound, Mapping) or bound.get("sha256") != real_sha:
        raise D180Undeterminable(f"{path} is not the bytes the producer bound (sha {real_sha[:12]} != bound {str((bound or {}).get('sha256'))[:12]})")
    relaxed = json.dumps({**ddoc, "identical": ddoc["n_requests"], "reproduces": True, "misses": []})
    return {"text": relaxed, "producer": {**producer, rel: {**bound, "sha256": _sha_text(relaxed)}}, "diverged": diverged, "real_sha": real_sha}


def complete_verdict(*, prev_plan: Mapping[str, Any], prev_row: Mapping[str, Any], identity: Mapping[str, Any], aid: str,
                     read_text: Callable[[str], Optional[str]], full_children: Optional[int] = None,
                     criterion3_waiver: Optional[Mapping[str, Any]] = None, reaudit: Optional[Mapping[str, Any]] = None,
                     this_plan: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Clause (c): the root facts (`lp._smoke_row_facts`) and the S3 check set (`lp._smoke_check` over `lp.smoke_evidence_paths`), the producer
    bindings held to the PRIOR tree (`plan=prev_plan`), the ladder inventory on a producer row, the FULL child count when the source is an S5 row.
    ``reaudit`` (fj7): THIS tree's `request_evidence_audit.py` was run over the carried root's stores into the carried entry's own evidence root
    (never the historical root) -- the audit check then reads the FRESH verdicts there, bound through the re-audit's `producer_evidence`
    (`arm_binding` and all) against THIS plan (`this_plan`), the stores still held to the historical root (`store_root`)."""
    sid = prev_row["standup_id"]
    try:
        facts = lp._smoke_row_facts(sid=sid, row=prev_row, identity=identity, accepted_get=lambda _k: aid, read_text=read_text)
    except ValueError as e:
        return {"ok": False, "why": f"root facts: {e}"}
    root, ev = facts["attempt_root"], facts["_evidence"]
    if full_children is not None and facts.get("n_children") != full_children:
        return {"ok": False, "why": f"the attempt ran {facts.get('n_children')} child(ren), the production row has {full_children} -- not the FULL shape", "attempt_root": root}
    checks: Dict[str, Any] = {}
    for name in COMPLETE_CHECKS:
        producer = lp.producer_map_for(name, ev)
        check_root, check_plan = root, prev_plan
        # With a re-audit that also RE-DERIVED the sampler verdict (`seedrep_served.rederive_sampler_verdict`: this tree's recorder over the
        # carried root's engine log, bound under the evidence root), the resolved-sampler check reads THAT record held to this plan -- the
        # historical record (UNCOMPARED on every two-sampler row) never stands in; without the binding the historical one is read as before
        rederived_sampler = name == "resolved_sampler" and reaudit is not None and isinstance(reaudit.get("producer_evidence"), Mapping) \
            and isinstance(reaudit["producer_evidence"].get(lp.RESOLVED_SAMPLER_REL), Mapping) and not reaudit["producer_evidence"][lp.RESOLVED_SAMPLER_REL].get("absent")
        if (name == "request_evidence_audit" and reaudit is not None) or rederived_sampler:
            probs = reaudit_problems(reaudit, this_sha=str((this_plan or {}).get("W")))
            if probs:
                return {"ok": False, "why": f"check {name!r}: the re-audit record is not this tree's: {probs}", "attempt_root": root}
            producer, check_root, check_plan = reaudit["producer_evidence"], str(reaudit["evidence_root"]), this_plan
        paths = lp.smoke_evidence_paths(prev_row, check_root, producer)[name]       # F-AO: the audit's verdicts at the FILES the producer bound
        if not paths:
            if name == "pool_endpoint" and not prev_row.get("pool"):
                continue
            return {"ok": False, "why": f"check {name!r}: the row names no producer file to read", "attempt_root": root}
        aligned: Dict[str, Tuple[Any, Any]] = {}
        if name == "request_evidence_audit":
            aligned = {p: (k, leg) for p, k, leg in lp.smoke_audit_sources(prev_row, check_root, producer)}
        if name == "determinism":
            aligned = {f"{root}/{rel}": (None, leg) for leg, rel in lp.determinism_rels(prev_row)}
        sources = []
        if name == "determinism" and criterion3_waiver is not None:
            # The determinism spot-check ABSENT is admitted ONLY under an explicit, scoped operator waiver -- recorded verbatim, never inferred;
            # the waiver covers ONLY the determinism record's absence, every other check still runs
            probs = criterion3_waiver_problems(criterion3_waiver, sid)
            if probs:
                return {"ok": False, "why": f"check {name!r}: the criterion-3 waiver is malformed: {probs}", "attempt_root": root}
            absent = [p for p in paths if read_text(p) is None]
            if absent:
                waived = f"criterion3 ({criterion3_waiver['decision'].strip()})"
                checks[name] = {"ok": True, "waived": waived, "verdict": DETERMINISM_WAIVED, "absent": absent, "sources": []}
                continue
        criterion3: Optional[Dict[str, Any]] = None
        for path in paths:
            text = read_text(path)
            if text is None:
                return {"ok": False, "why": f"check {name!r}: {path} is absent on the volume" + (" (criterion 3: a waiver `criterion3_waivers[standup_id] = {by, reason}` admits its absence by name)" if name == "determinism" else ""),
                        "attempt_root": root}
            child, table_leg = aligned.get(path, (None, None))
            diverged: Optional[Dict[str, Any]] = None
            producer_ = producer
            if name == "determinism":
                # fj22 / fj23 / fg1 (F-AW): the qualification is ONE function, `determinism_d180_relaxation` -- shared with the smoke-PASS
                # derivations (`seedrep_launch_plan._determinism_source`), so a sampled gsm8k row reads the same on every path
                try:
                    relax = determinism_d180_relaxation(text, path=path, root=root, producer=producer, sampler_check=checks.get("resolved_sampler"),
                                                        prev_row=prev_row, evidence=ev, read_text=read_text)
                except D180Undeterminable as e:
                    return {"ok": False, "why": f"check {name!r}: {e}", "attempt_root": root}
                if relax is not None:
                    diverged, real_sha, text, producer_ = relax["diverged"], relax["real_sha"], relax["text"], relax["producer"]
            try:
                facts_ = lp._smoke_check(name, text, path=path, plan=check_plan, accepted_attempts=[aid], row=prev_row, aid=aid, root=check_root,
                                         producer=producer_, child=child, table_leg=table_leg, store_root=(root if check_root != root else None))
            except ValueError as e:
                return {"ok": False, "why": f"check {name!r}: {e}", "attempt_root": root}
            if diverged is not None:
                facts_ = {**facts_, "sha256": real_sha, "reproduces": False, "identical": diverged["reproduced"], "criterion3": CRITERION3_NOT_APPLICABLE_SAMPLED}
                if criterion3 is None:
                    criterion3 = {**diverged, "records": [path]}
                else:
                    criterion3 = {**criterion3, "reproduced": criterion3["reproduced"] + diverged["reproduced"], "resent": criterion3["resent"] + diverged["resent"],
                                  "misses": criterion3["misses"] + diverged["misses"], "records": criterion3["records"] + [path]}
            sources.append(facts_)
        checks[name] = {"ok": True, "sources": sources}
        if criterion3 is not None:
            checks[name]["criterion3"] = criterion3
        if check_root != root:
            checks[name]["reaudit"] = {"reaudited_by": reaudit["reaudited_by"], "evidence_root": check_root, "rc": reaudit.get("rc"), "held_to_plan": "this"}
            if name == "resolved_sampler":
                checks[name]["reaudit"]["sampler"] = dict(reaudit.get("sampler") or {})       # fc11d: historical vs re-derived verdict, the recorder's rc
    ladder: Optional[Dict[str, Any]] = None
    if (prev_row.get("ladder") or {}).get("cap_next") is not None:
        text = read_text(f"{root}/{LADDER_ENQUEUED_REL}")
        if text is None:
            return {"ok": False, "why": f"a producer row without its {LADDER_ENQUEUED_REL} on the volume", "attempt_root": root}
        doc = json.loads(text)
        if doc.get("schema") != "seedrep_ladder_enqueued/1" or doc.get("complete") is not True or doc.get("attempt_id") != aid \
                or not isinstance(doc.get("entries"), list) or doc.get("n") != len(doc["entries"]):
            return {"ok": False, "why": f"{LADDER_ENQUEUED_REL} is not this attempt's complete inventory (schema {doc.get('schema')!r}, complete {doc.get('complete')!r}, "
                                        f"attempt {doc.get('attempt_id')!r}, n {doc.get('n')!r})", "attempt_root": root}
        ladder = {"path": f"{root}/{LADDER_ENQUEUED_REL}", "sha256": _sha_text(text), "n": doc["n"], "complete": True}
    out = {"ok": True, "attempt_root": root, "result_line": facts["result_line"], "sources": facts["sources"], "n_children": facts.get("n_children"),
           "attested_max_num_seqs": facts.get("attested_max_num_seqs"), "checks": checks, "ladder_inventory": ladder,
           "evidence_smoke": ev.get("smoke"), "_evidence": ev}
    det = checks.get("determinism") or {}
    if det.get("waived"):
        out["waived"] = det["waived"]
    return out


# --------------------------------------------------------------------------- #
# clause (d): engine identity                                                   #
# --------------------------------------------------------------------------- #

def engine_identity_verdict(*, this_plan: Mapping[str, Any], this_row: Mapping[str, Any], root: str, aid: str, prev_W: str, evidence: Mapping[str, Any],
                            read_text: Callable[[str], Optional[str]],
                            readiness_problems: Callable[..., List[str]], served_standup_id: Optional[str] = None) -> Dict[str, Any]:
    """Clause (d): THIS tree's readiness contract over the attempt's READY records against THIS plan's row, plus the served adapter hashes against
    THIS plan's adapter set (the cells.json binding is the producer's, at the PRIOR tree)."""
    texts = {n: read_text(f"{root}/{lp.SERVE_DIR_REL}/{n}") for n in READY_RECORDS}
    missing = [n for n, t in texts.items() if t is None]
    if missing:
        return {"ok": False, "why": f"READY record(s) {missing} absent under {root}/{lp.SERVE_DIR_REL} -- no attestation to hold the row to"}
    att_text = read_text(f"{root}/{ATTESTATION_REL}")
    try:
        endpoint, cells_served = json.loads(texts["endpoint.json"]), json.loads(texts["cells_served.json"])
        attestation = json.loads(att_text) if att_text else None
    except ValueError as e:
        return {"ok": False, "why": f"READY record is not JSON: {e}"}
    try:
        problems = list(readiness_problems(endpoint, cells_served, this_row, attestation=attestation))
    except Exception as e:  # noqa: BLE001 -- a reader that cannot judge is a refusal, never a pass
        return {"ok": False, "why": f"this tree's readiness contract could not judge the attestation: {type(e).__name__}: {e}"}
    if problems:
        return {"ok": False, "why": f"the attempt's engine is not the one this plan's row describes: {problems[:4]}{' ...' if len(problems) > 4 else ''}", "problems": problems}
    cells_path = f"{root}/{lp.CELLS_JSON_REL}"
    cells_text = read_text(cells_path)
    if cells_text is None:
        return {"ok": False, "why": f"{cells_path} absent -- the served adapter hashes cannot be held to this plan"}
    plan_prev_tree = dict(this_plan)
    plan_prev_tree["W"] = prev_W          # the binding's tree is the PRODUCER's (clause (b) makes it this tree's bytes); the adapter set compared is THIS plan's
    # cells.json's `serving.standup_id` is the row the container SERVED (a smoke source's `smoke-S5__<id>`); the route set / objects held to are THIS plan's
    row_for_cells = dict(this_row)
    row_for_cells["standup_id"] = served_standup_id or this_row["standup_id"]
    try:
        hashes = lp._smoke_check("adapter_hashes_banked", cells_text, path=cells_path, plan=plan_prev_tree, accepted_attempts=[aid], row=row_for_cells, aid=aid, root=root,
                                 producer=lp.producer_map_for("adapter_hashes_banked", evidence))
    except ValueError as e:
        return {"ok": False, "why": f"served adapter hashes vs this plan: {e}"}
    return {"ok": True, "readiness_problems": [], "ready_records": {n: {"path": f"{root}/{lp.SERVE_DIR_REL}/{n}", "sha256": _sha_text(t)} for n, t in texts.items()},
            "attestation": ({"path": f"{root}/{ATTESTATION_REL}", "sha256": _sha_text(att_text)} if att_text else None),
            "thinking_mode": endpoint.get("thinking_mode"), "attested_max_model_len": endpoint.get("attested_max_model_len"),
            "attested_max_num_seqs": endpoint.get("attested_max_num_seqs"), "adapter_hashes": {k: hashes.get(k) for k in ("n_banked", "n_planned", "cells")}}


# --------------------------------------------------------------------------- #
# clause (e): the ladder queue                                                  #
# --------------------------------------------------------------------------- #

def ladder_queue_verdict(prev_row: Mapping[str, Any], this_plan: Mapping[str, Any], *, replay: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Clause (e). ``replay`` (fj3) performs :func:`replay_ladder_queue` for this row when its rung is consumed in THIS plan; its record decides."""
    lad = prev_row.get("ladder") or {}
    cap = lad.get("cap_next") if isinstance(lad, Mapping) else None
    if cap is None:
        return {"ok": True, "producer_row": False}
    consumers = [r["standup_id"] for r in (this_plan.get("rungs") or []) if r.get("leg") == prev_row.get("leg") and r.get("engine") == prev_row.get("engine")
                 and str(r.get("stage")) == str(cap)]
    if not consumers:
        return {"ok": True, "producer_row": True, "cap_next": cap, "consumers": [], "note": "no consumer of its rung in this plan"}
    if replay is None:
        return {"ok": False, "producer_row": True, "cap_next": cap, "consumers": consumers,
                "why": f"a producer row of the {cap} rung consumed by {consumers} in this plan: its queue lives in the prior (W7, wave) ladder root, which this "
                       "wave's consumer never reads -- carried only through the queue REPLAY (replay_ladder_queue), which this evaluation was not given"}
    try:
        rec = replay()
    except Exception as e:  # noqa: BLE001 -- a replay that cannot complete refuses the row by name; nothing partial is ever admitted
        return {"ok": False, "producer_row": True, "cap_next": cap, "consumers": consumers, "why": f"queue replay refused: {type(e).__name__}: {e}"}
    ok = isinstance(rec, Mapping) and rec.get("ok") is True and rec.get("closed") is True
    out = {"ok": ok, "producer_row": True, "cap_next": cap, "consumers": consumers, "replay": rec}
    if not ok:
        out["why"] = f"queue replay incomplete: {(rec or {}).get('why') if isinstance(rec, Mapping) else rec!r}"
    return out


def _vol_rel(path: str) -> str:
    """`/vol/<rel>` -> `<rel>` (a VolumeFS path); a relative path is returned as given."""
    return path[len(lp.VOL_PATH) + 1:] if path.startswith(lp.VOL_PATH + "/") else path.lstrip("/")


def carried_producer_root(ladder_root: str, shard: str, attempt_id: str) -> str:
    """Where the replay puts the carried producer's re-keyed `queue_inventory.json` + `COMPLETE.json` stub (the reconciler's two hard-coded
    names), under THIS wave's ladder root -- never under the results root, so no store reader mistakes it for an attempt root."""
    return f"{ladder_root}/{CARRIED_DIR}/{shard}__{attempt_id}"


def carry_aware_root_of(L: Any, vol: Any, spec: Any, default_root_of: Callable[[Any, str], str]) -> Callable[[Any, str], str]:
    """A `reconcile_queue(attempt_root_of=...)` for ONE queue: a producer whose carried stub exists under `<ladder root>/carried/<shard>__<aid>/`
    resolves there (the replay's re-keyed inventory beside its COMPLETE stub); every other producer through ``default_root_of`` (the app's
    spelling under this wave). The steward uses it per queue, the replay itself for the closure it runs."""
    def root_of(p: Any, aid: str) -> str:
        cr = carried_producer_root(spec.ladder_root, p.shard, aid)
        return cr if vol.exists(L.complete_rel(cr)) else default_root_of(p, aid)
    return root_of


def replay_ladder_queue(*, L: Any, vol: Any, accepted_kv: Any, prev_row: Mapping[str, Any], prior: Mapping[str, Any], this_plan: Mapping[str, Any],
                        aid: str, prev_root: str, expected_n: Optional[int] = None, default_root_of: Optional[Callable[[Any, str], str]] = None,
                        utc: Optional[str] = None, prev_wave_uuid: Optional[str] = None) -> Dict[str, Any]:
    """The ladder-queue REPLAY (module docstring, clause (e)). ``L`` is `seedrep_ladder`; ``vol`` a VolumeFS with the volume's real commit /
    reload (a RemoteVolume on the grading host); ``accepted_kv`` THIS wave's accepted KV; ``prev_root`` the carried attempt's root (`/vol/...` or
    volume-relative). Raises `LadderRefusal` / `ValueError` on ANY refusal before a single write (a torn record, a sha that differs from the
    inventory, a record of another producer / queue, a target file with different bytes) -- never a partial replay. Idempotent: a second run
    over the same prior finds identical bytes and writes nothing new.

    fj32 -- BULK, O(1) volume operations in the queue's size (the per-file form took ~5 s a record through the Volume API: ~30 min for a
    200-item queue, 8-15 h for controls' 5,872): ONE recursive listing of the prior ``queue/`` (the same listing the inventory is verified
    against), ONE bulk read of every listed record (a RemoteVolume pulls the directory once), the re-keying local and byte-identical to the
    per-file form (the two fields, the canonical JSON, the attempt-unique target path), ONE recursive listing of THIS wave's ladder root for
    what is already there, ONE bulk read of the present files (identical bytes skip, different bytes refuse), then ONE bulk upload + ONE
    commit carrying every new queue file AND the carried-producer inventory + stub -- nothing is written before every record validates, and a
    bulk upload that fails midway commits nothing, so the re-run finds the same world (or a subset of the files, which it skips as identical).
    The report gains ``counts`` {read, rekeyed, skipped_identical, uploaded} and ``volume_ops`` (the calls by kind, from a `CountingVolume`
    over ``vol``); every other field is as before."""
    lad = prev_row.get("ladder") or {}
    kind, cap_next, shard = str(lad["kind"]), int(lad["cap_next"]), str(lad["shard"])
    shards = [str(x) for x in (lad.get("shards") or [shard])]
    leg, view = str(prev_row["leg"]), str(prev_row["engine"])
    # The prior queue lives under the wave the attempt was ACCEPTED under (``prev_wave_uuid``: a smoke source's own wave), never
    # blindly the prior plan's production wave -- for a same-tree smoke source that would be THIS wave, the very root the replay writes
    prev_spec = L.QueueSpec.build(prior["W7"], prev_wave_uuid or prior["wave_uuid"], leg, view, kind, cap_next, shards)
    this_spec = L.QueueSpec.build(this_plan["W7"], this_plan["wave_uuid"], leg, view, kind, cap_next, shards)
    prev_ref = next(p for p in prev_spec.producers if p.shard == shard)
    this_ref = next(p for p in this_spec.producers if p.shard == shard)
    root_rel = _vol_rel(prev_root)
    vol = L.CountingVolume(vol)                    # fj32: every volume call of this run, tallied into the report
    vol.reload()
    # 1. the prior inventory, verified EXACTLY as the reconciler verifies a producer (every listed file present with its sha, none stray) --
    #    from ONE recursive listing of the prior queue/ and ONE bulk read of the listed entries
    inv_rel = L.inventory_rel(root_rel)
    if not vol.exists(inv_rel):
        raise ValueError(f"{inv_rel}: the carried attempt has no queue_inventory.json -- nothing to replay")
    inv_bytes = vol.read_bytes(inv_rel)
    inv = json.loads(inv_bytes.decode("utf-8"))
    if inv.get("queue_root") != prev_spec.ladder_root or inv.get("shard") != shard or inv.get("attempt_id") != aid:
        raise ValueError(f"{inv_rel}: names queue {inv.get('queue_root')!r} / shard {inv.get('shard')!r} / attempt {inv.get('attempt_id')!r}, "
                         f"not {prev_spec.ladder_root!r} / {shard!r} / {aid!r}")
    prev_files = L._queue_files(vol, prev_spec)
    on_volume = {rel for _, _, _, rel in prev_files}
    entries_in = inv.get("entries") if isinstance(inv.get("entries"), list) else []
    to_read = sorted({str(e.get("path", "")) for e in entries_in if isinstance(e, Mapping) and str(e.get("path", "")) in on_volume})
    blobs: Dict[str, bytes] = vol.read_many(to_read) if to_read else {}
    problems = L._verify_inventory_files(vol, inv, prev_spec, aid, files=prev_files, blobs=blobs)
    if problems:
        raise ValueError(f"the prior inventory does not verify: {problems[:4]}")
    if expected_n is not None and int(inv.get("n", -1)) != int(expected_n):
        raise ValueError(f"queue_inventory.json n {inv.get('n')} != ladder_enqueued.json n {expected_n}")
    complete_rel = L.complete_rel(root_rel)
    if not vol.exists(complete_rel):
        raise ValueError(f"{complete_rel} absent -- the carried attempt is not COMPLETE on the volume")
    prior_complete_sha = L.sha256_hex(vol.read_bytes(complete_rel))
    # 2. every record: bytes from the bulk read, sha, validate against the PRIOR spec, bind to this producer, re-key, validate against THIS
    #    spec -- before ANY write; what is already under THIS wave's ladder root comes from ONE recursive listing of it
    already = set(vol.walk(this_spec.ladder_root))
    plan_writes: List[Tuple[str, bytes, Dict[str, Any]]] = []
    present: List[str] = []
    targets_seen: set = set()
    for e in inv["entries"]:
        rel = str(e["path"])
        data = blobs[rel]
        if L.sha256_hex(data) != e.get("sha256"):
            raise ValueError(f"{rel}: sha differs from the inventory (torn record)")
        rec = json.loads(data.decode("utf-8"))
        L._validate_record(rec, prev_spec, where=rel)
        if rec.get("producer") != {"shard": shard, "attempt_id": aid}:
            raise ValueError(f"{rel}: record producer {rec.get('producer')} is not {{'shard': {shard!r}, 'attempt_id': {aid!r}}}")
        if rec.get("arm_s") != e.get("arm_s") or rec.get("item_key") != e.get("item_key"):
            raise ValueError(f"{rel}: content names {rec.get('arm_s')!r}/{rec.get('item_key')!r}, the inventory {e.get('arm_s')!r}/{e.get('item_key')!r}")
        rec2 = dict(rec)
        rec2["W7"], rec2["wave_uuid"] = this_spec.W7, this_spec.wave_uuid       # the ONLY two fields that change
        target = L.queue_entry_rel(this_spec.ladder_root, rec2["arm_s"], rec2["item_key"], aid)
        L._validate_record(rec2, this_spec, where=target)
        data2 = (L.canonical_json(rec2) + "\n").encode("utf-8")
        entry = {"item_key": rec2["item_key"], "arm_s": rec2["arm_s"], "slot": L.slot(rec2["arm_s"], rec2["item_key"]), "path": target, "sha256": L.sha256_hex(data2)}
        if target in targets_seen:
            raise ValueError(f"{target}: two inventory entries re-key to ONE queue path -- refusing to replay a duplicated item")
        targets_seen.add(target)
        if target in already:
            present.append(target)
        plan_writes.append((target, data2, entry))
    entries = [entry for _t, _d, entry in plan_writes]
    croot = carried_producer_root(this_spec.ladder_root, shard, aid)
    inv2 = {"schema": L.SCHEMA_INVENTORY, "key": L.key_str(this_ref.key), "queue_root": this_spec.ladder_root, "attempt_id": aid, "shard": shard,
            "n": len(entries), "item_keys_sha256": L.item_keys_sha256(L.slot(e["arm_s"], e["item_key"]) for e in entries), "entries": entries,
            "replayed_from": {"W7": prev_spec.W7, "wave_uuid": prev_spec.wave_uuid, "queue_root": prev_spec.ladder_root, "attempt_root": root_rel,
                              "inventory": inv_rel, "inventory_sha256": L.sha256_hex(inv_bytes), "key": L.key_str(prev_ref.key)}}
    stub = {"schema": CARRIED_PRODUCER_SCHEMA, "attempt_id": aid, "shard": shard, "key": L.key_str(this_ref.key), "queue_root": this_spec.ladder_root,
            "carried_from": {"W7": prev_spec.W7, "wave_uuid": prev_spec.wave_uuid, "attempt_root": root_rel, "complete_sha256": prior_complete_sha, "tree_sha": prior["W"]},
            "note": "a CARRIED producer (D112 / fj3 replay): the store and its evidence live under carried_from.attempt_root; this stub exists so the reconciler "
                    "finds the re-keyed inventory beside a COMPLETE marker"}
    carried_docs = [(L.inventory_rel(croot), inv2), (L.complete_rel(croot), stub)]
    # 3. what is already there, read in ONE bulk read: a queue file with different bytes refuses (before any write), identical bytes skip;
    #    an existing carried-producer record must agree with the one this run derived
    present_carried = [rel for rel, _doc in carried_docs if rel in already]
    have: Dict[str, bytes] = vol.read_many(present + present_carried) if (present or present_carried) else {}
    for target, data2, _entry in plan_writes:
        if target in already and have[target] != data2:
            raise ValueError(f"{target}: a queue file with DIFFERENT bytes already exists under this wave -- refusing to replay over it")
    to_write: Dict[str, bytes] = {target: data2 for target, data2, _entry in plan_writes if target not in already}
    for rel, doc in carried_docs:
        data = (L.canonical_json(doc) + "\n").encode("utf-8")
        if rel in already:
            old = json.loads(have[rel].decode("utf-8"))
            core_old = {k: v for k, v in old.items() if k not in ("written_utc", "replayed_utc")}
            if core_old != doc:
                raise ValueError(f"{rel}: an existing carried-producer record disagrees; refusing to overwrite")
            continue
        to_write[rel] = data
    # 4. ONE bulk upload + ONE commit for every new file (queue records, then the carried-producer inventory + stub); nothing when all present.
    #    Then the READ-BACK: every queue file under this ladder root (the ones found by the listing and the ones just uploaded) in ONE bulk read
    #    of the committed volume -- the closure below verifies the inventory's shas and builds the accepted projection from THOSE bytes, never
    #    from this run's local plan (a torn or missed upload is caught here, exactly as the per-file form's read-back caught it)
    queue_prefix = f"{this_spec.ladder_root}/queue/"
    if to_write:
        vol.write_many(to_write)
        vol.commit()
        readback = vol.read_many(sorted({r for r in already if r.startswith(queue_prefix)} | {t for t, _d, _e in plan_writes}))
    else:
        readback = {r: b for r, b in have.items() if r.startswith(queue_prefix)}
    for target, data2, _entry in plan_writes:
        if readback.get(target) != data2:
            raise ValueError(f"{target}: the read-back after the upload does not carry the re-keyed bytes -- the upload did not land; re-run")
    # 5. the acceptance under THIS wave's key (the same primitive `seed_accepted` uses), then the marker, then closure
    if L.accept_attempt(accepted_kv, this_ref.key, aid):
        accepted = "seeded"
    else:
        holder = L.accepted_attempt(accepted_kv, this_ref.key)
        if holder != aid:
            raise ValueError(f"this wave's accepted Dict names {holder!r} under {L.key_str(this_ref.key)}, not the carried {aid!r} -- a carry never overwrites an acceptance")
        accepted = "present"
    marker = L.write_closed_marker(vol, this_spec, this_ref, aid, inv2, synthesised_by=REPLAY_SYNTHESISED_BY)
    default_root_of = default_root_of or (lambda p, a: L.producer_attempt_root(p, a))
    status = L.reconcile_queue(vol, accepted_kv, this_spec, attempt_root_of=carry_aware_root_of(L, vol, this_spec, default_root_of), blobs=readback)
    return {"schema": REPLAY_SCHEMA, "ok": True, "closed": True, "n_records": len(entries), "n_written": len(entries) - len(present), "n_already_present": len(present),
            "source_keys": {"queue": L.key_str(prev_spec.queue_key), "producer": L.key_str(prev_ref.key), "queue_root": prev_spec.ladder_root, "inventory": inv_rel},
            "target_keys": {"queue": L.key_str(this_spec.queue_key), "producer": L.key_str(this_ref.key), "queue_root": this_spec.ladder_root,
                            "carried_root": croot, "marker": L.producer_marker_rel(this_spec.ladder_root, shard, aid)},
            "accepted": accepted, "marker": {k: marker.get(k) for k in ("n_enqueued", "item_keys_sha256", "synthesised_by")},
            "queue_closed": bool(status.get("queue_closed")), "producers": {s: st.get("state") for s, st in (status.get("producers") or {}).items()},
            "closed_json": (L.closed_rel(this_spec.ladder_root) if status.get("queue_closed") else None), "replayed_utc": utc or _utc(),
            "counts": {"read": len(blobs), "rekeyed": len(entries), "skipped_identical": len(present), "uploaded": len(to_write),
                       "uploaded_queue_files": len(entries) - len(present), "uploaded_carried_records": len(to_write) - (len(entries) - len(present))},
            "volume_ops": vol.report()}


# --------------------------------------------------------------------------- #
# the Petri sidecar proof                                                     #
# --------------------------------------------------------------------------- #

def petri_cell_dirs(evidence: Mapping[str, Any], *, home: Optional[Path] = None) -> List[str]:
    """The carried Petri store's cell directories from the attempt's completion bundle (`modal_evidence.json` `petri.evidence`, the drive's
    `drive_verdict_record`: `verdict.detail = [[record_name, run_key, why], ...]`, `view`, `W7`, `wave_uuid`, `attempt_id`) through the drive's
    OWN out-root spelling (`seedrep_petri_drive.out_root_for` / `run_tag`; the stores are grading-host-local). Refuses a bundle without them."""
    petri = evidence.get("petri") if isinstance(evidence.get("petri"), Mapping) else None
    bundle = (petri or {}).get("evidence")
    if not isinstance(bundle, Mapping):
        raise ValueError("the Petri attempt's evidence carries no completion bundle (petri.evidence)")
    detail = ((bundle.get("verdict") or {}).get("detail")) or []
    if not detail:
        raise ValueError("the completion bundle's verdict.detail names no (record, run_key) -- no cell to read")
    import seedrep_petri_drive as drive  # noqa: E402 -- the sibling's out-root spelling, never re-spelled here
    wu8 = str(bundle["wave_uuid"]).replace("-", "")[:8]
    out = []
    for item in detail:
        record_name, run_key = str(item[0]), str(item[1])
        m = re.fullmatch(r"leg_(\d+)_s(\d+)", record_name)
        if not m:
            raise ValueError(f"completion detail record {record_name!r} is not leg_<i>_s<seed>")
        tag = drive.run_tag(str(bundle["view"]), int(m.group(2)))
        root = drive.out_root_for(Path(home or Path.home()), str(bundle["W7"]), wu8, str(bundle["attempt_id"]), tag)
        out.append(f"{root}/cells/{run_key}")
    return out


PETRI_JUDGE_CHAIN_KEY = "judge_chain"          # a results.json tally under this name when the runner tallies the judge chain (forward-compatible)


def petri_judge_fallback_reading(cells: Sequence[str], read_text: Callable[[str], Optional[str]]) -> Dict[str, Any]:
    """The JUDGE chain's reading over a store of a tree that HAS one: uses per judge leg from a `results.json` `judge_chain` tally
    (top-level or folded from `behaviors`) when the runner wrote one, else the `judge_refused` totals it does write (petri_stats' footing);
    the per-sample `petri_judge_chain` records live inside the Inspect logs and are named, not parsed, here."""
    uses: Dict[str, int] = {}
    refused = 0
    tallied = False
    for cell in cells:
        text = read_text(f"{cell}/{PETRI_RESULTS_JSON}")
        if text is None:
            continue
        res = json.loads(text)
        chain = res.get(PETRI_JUDGE_CHAIN_KEY)
        if not isinstance(chain, Mapping):
            chain = {}
            for b in (res.get("behaviors") or {}).values():
                for k, n in ((b or {}).get(PETRI_JUDGE_CHAIN_KEY) or {}).items():
                    chain[k] = chain.get(k, 0) + int(n)
        if chain:
            tallied = True
        for k, n in chain.items():
            uses[k] = uses.get(k, 0) + int(n)
        jr = res.get("judge_refused")
        if isinstance(jr, Mapping):
            refused += int(jr.get("total", 0) or 0)
        else:
            for b in (res.get("behaviors") or {}).values():
                refused += int((((b or {}).get("judge_refused") or {}).get("total")) or 0)
    return {"configured": True, "uses_per_leg": (uses if tallied else None), "judge_refused_total": refused,
            "source": ("results.json judge_chain tally" if tallied else "results.json carries no judge_chain tally; the per-sample petri_judge_chain records "
                                                                        "are in the Inspect logs (not parsed here)")}


def _chain_used_removed(chain: Any, removed_labels: Sequence[str], removed_models: Sequence[str], uses: Dict[str, int]) -> int:
    """The removed leg's uses in ONE per-sample auditor-chain record (`{configured, used: [{label, model, ...}]}`), every used leg tallied
    into ``uses`` by label (or model when unlabelled)."""
    n = 0
    for u in ((chain or {}).get("used") or []) if isinstance(chain, Mapping) else []:
        label, model = str((u or {}).get("label", "")), str((u or {}).get("model", ""))
        uses[label or model] = uses.get(label or model, 0) + 1
        if label in removed_labels or model in removed_models:
            n += 1
    return n


def petri_sidecar_proof(cells: Sequence[str], read_text: Callable[[str], Optional[str]], *, removed_labels: Sequence[str] = PETRI_REMOVED_LEG_LABELS,
                        removed_models: Sequence[str] = PETRI_REMOVED_LEG_MODELS, judge_chain_in_prior: bool = False) -> Dict[str, Any]:
    """ZERO uses of the removed auditor leg across the carried store, over EVERY outcome a conversation can land in:

    * the AUDITED items: `results.json`'s `auditor_chain` tally per behavior (:data:`PETRI_REMOVED_LEG_BUCKETS` count as uses);
    * the AUDITOR-refused items: `behaviors[*].auditor_refused_items[*].auditor_chain` and the `petri_auditor_refused.json` ledger's
      `shed.auditor_chain` (these are also in the tally, under the terminal buckets);
    * the JUDGE-refused items: `behaviors[*].judge_refused_items[*].auditor_chain` and the `petri_judge_refused.json` ledger's
      `auditor_chain` -- the summary producer EXCLUDES them from the auditor-chain tally (the conversation was carried but not scored),
      so without this reading a Qwen-carried, judge-refused conversation proved as zero uses.

    Outcome COVERAGE is verified per behavior (petri_stats' conservation law): `n_items == audited + auditor_refused.total +
    judge_refused.total`, the item lists as long as their totals, the chain tally summing to audited + auditor-refused, and each ledger
    holding exactly the records the summary counts -- an item that lands in no tally, or in two, refuses the proof. Every input, the
    optional ledgers included, is read through ``read_text`` (absent -> None), so a recording reader binds the whole input set.
    A cell without `results.json` refuses. `judge_fallback` is `n/a (not in prior tree)` when the prior tree had no judge chain."""
    uses: Dict[str, int] = {}
    tally: Dict[str, int] = {}
    n_refused = n_judge_refused = 0
    audited = 0
    per_cell = []
    coverage: List[str] = []
    for cell in cells:
        res_text = read_text(f"{cell}/{PETRI_RESULTS_JSON}")
        led_text = read_text(f"{cell}/{PETRI_REFUSED_LEDGER}")                 # both ledgers ALWAYS read (recorded absent when absent)
        jled_text = read_text(f"{cell}/{PETRI_JUDGE_REFUSED_LEDGER}")
        if res_text is None:
            return {"ok": False, "why": f"{cell}/{PETRI_RESULTS_JSON} absent -- the carried Petri cell is not complete", "cells": len(cells)}
        res = json.loads(res_text)
        behaviors = res.get("behaviors") if isinstance(res.get("behaviors"), Mapping) else {}
        chain = res.get("auditor_chain")
        if not isinstance(chain, Mapping):
            chain = {}
            for b in behaviors.values():
                for k, n in ((b or {}).get("auditor_chain") or {}).items():
                    chain[k] = chain.get(k, 0) + int(n)
        for k, n in chain.items():
            tally[k] = tally.get(k, 0) + int(n)
        cell_audited = int(((res.get("audited") or {}).get("items")) or 0)
        audited += cell_audited
        tally_uses = sum(int(chain.get(k, 0) or 0) for k in PETRI_REMOVED_LEG_BUCKETS)
        # per behavior: the excluded items' provenance + the conservation law
        cell_ar = cell_jr = 0
        judge_item_uses = auditor_item_uses = 0
        for bname, b in behaviors.items():
            b = b if isinstance(b, Mapping) else {}
            n_items = int(b.get("n_items") or 0)
            b_aud = int(b.get("audited") or 0)
            ar = int(((b.get("auditor_refused") or {}).get("total")) or 0)
            jr = int(((b.get("judge_refused") or {}).get("total")) or 0)
            ar_items = b.get("auditor_refused_items") or []
            jr_items = b.get("judge_refused_items") or []
            cell_ar += ar
            cell_jr += jr
            where = f"{cell} behavior {bname}"
            if n_items and n_items != b_aud + ar + jr:
                coverage.append(f"{where}: n_items {n_items} != audited {b_aud} + auditor_refused {ar} + judge_refused {jr} -- an item lands in no tally or in two")
            if len(ar_items) != ar:
                coverage.append(f"{where}: {len(ar_items)} auditor_refused_items listed, auditor_refused.total {ar}")
            if ("judge_refused" in b) and len(jr_items) != jr:
                coverage.append(f"{where}: {len(jr_items)} judge_refused_items listed, judge_refused.total {jr}")
            b_chain = b.get("auditor_chain") or {}
            if isinstance(b_chain, Mapping) and sum(int(v) for v in b_chain.values()) != b_aud + ar:
                coverage.append(f"{where}: auditor_chain tally sums to {sum(int(v) for v in b_chain.values())}, audited {b_aud} + auditor_refused {ar}")
            for it in jr_items:
                judge_item_uses += _chain_used_removed((it or {}).get("auditor_chain"), removed_labels, removed_models, uses)
            for it in ar_items:                                    # an outage-classified refusal may still have REACHED the leg: its chain counts
                auditor_item_uses += _chain_used_removed((it or {}).get("auditor_chain"), removed_labels, removed_models, {})
        top_jr = res.get("judge_refused")
        if isinstance(top_jr, Mapping) and int(top_jr.get("total", 0) or 0) != cell_jr:
            coverage.append(f"{cell}: judge_refused.total {top_jr.get('total')} != the behaviors' {cell_jr}")
        # the ledgers: every record's chain read; the record counts must be the summary's
        led_n = None
        ledger_auditor_uses = 0
        if led_text is not None:
            led = json.loads(led_text)
            recs = led.get("auditor_refused") or []
            led_n = len(recs)
            n_refused += led_n
            for r in recs:
                ledger_auditor_uses += _chain_used_removed(((r or {}).get("shed") or {}).get("auditor_chain"), removed_labels, removed_models, uses)
        if cell_ar and led_n != cell_ar:
            coverage.append(f"{cell}: the summary counts {cell_ar} auditor-refused item(s) but {PETRI_REFUSED_LEDGER} holds {led_n if led_n is not None else 'no ledger'}")
        jled_n = None
        ledger_judge_uses = 0
        if jled_text is not None:
            jled = json.loads(jled_text)
            recs = jled.get("judge_refused") or []
            jled_n = len(recs)
            n_judge_refused += jled_n
            for r in recs:
                ledger_judge_uses += _chain_used_removed((r or {}).get("auditor_chain"), removed_labels, removed_models, {})
        if cell_jr and jled_n != cell_jr:
            coverage.append(f"{cell}: the summary counts {cell_jr} judge-refused item(s) but {PETRI_JUDGE_REFUSED_LEDGER} holds {jled_n if jled_n is not None else 'no ledger'}")
        # the removed leg's uses, by outcome: the audited + auditor-refused items through the tally's buckets OR their item / ledger chains
        # (whichever saw more -- an `auditor_unavailable` refusal's chain can still name the leg), plus the judge-refused conversations'
        # chains (summary items or ledger, whichever saw more). The ok/refuse decision is exact (any source > 0 refuses); the count is the
        # best-informed reading, never a double count of one conversation across two sources of the same outcome
        auditor_uses = max(tally_uses, auditor_item_uses, ledger_auditor_uses)
        judge_uses = max(judge_item_uses, ledger_judge_uses)
        cell_uses = auditor_uses + judge_uses
        per_cell.append({"cell": cell, "removed_leg_uses": cell_uses, "chain": dict(chain), "auditor_refused": cell_ar, "judge_refused": cell_jr,
                         "tally_removed_leg_uses": tally_uses, "auditor_refused_removed_leg_uses": max(auditor_item_uses, ledger_auditor_uses),
                         "judge_refused_removed_leg_uses": judge_uses})
    removed_uses = sum(c["removed_leg_uses"] for c in per_cell)
    for k in PETRI_REMOVED_LEG_BUCKETS:
        if tally.get(k):
            uses[k] = uses.get(k, 0) + int(tally[k])
    out = {"ok": removed_uses == 0 and bool(cells) and not coverage, "cells": len(cells), "audited_items": audited, "chain_tally": tally, "used_legs": uses,
           "removed_leg": {"labels": list(removed_labels), "models": list(removed_models), "uses": removed_uses},
           "refused_records": n_refused, "judge_refused_records": n_judge_refused, "per_cell": per_cell,
           "coverage": {"ok": not coverage, "problems": coverage, "rule": "n_items == audited + auditor_refused.total + judge_refused.total per behavior; "
                                                                          "item lists and ledgers hold exactly the counted records"},
           "judge_fallback": (petri_judge_fallback_reading(cells, read_text) if judge_chain_in_prior else "n/a (not in prior tree)")}
    if not cells:
        out["why"] = "no Petri cell to read"
    elif removed_uses:
        out["why"] = f"the removed auditor leg {list(removed_labels)} was reached {removed_uses} time(s) in the carried store (judge-refused conversations included) -- the chain difference is not inert here"
    elif coverage:
        out["why"] = f"outcome coverage incomplete: {coverage[:3]}"
    return out


# --------------------------------------------------------------------------- #
# the evaluation                                                                #
# --------------------------------------------------------------------------- #

def _candidates(prior: Mapping[str, Any], this_row: Mapping[str, Any], *, current_tree: bool = False) -> List[Dict[str, Any]]:
    """The prior tree's attempts that may stand for ``this_row``: its production row under the prior production identity, then its reference
    smoke row (reconstructed with `lp.smoke_row` at the FULL shape) under the prior reference smoke identity, then its budgeted smoke row.
    ``current_tree``: the source IS this tree -- only its reference FULL-shape smoke can stand for the production row (its production key
    is the row itself, never a source), recorded as :data:`CURRENT_S5_SOURCE`."""
    plan = prior["plan"]
    out: List[Dict[str, Any]] = []
    sid = this_row["standup_id"]
    prev_row = next((r for r in _production_rows(plan) if r["standup_id"] == sid), None)
    if prev_row is None:
        return out
    if current_tree:
        s5 = (prior.get("smoke_waves") or {}).get(S5_SMOKE_ID)
        if prev_row.get("kind") == "rung":
            return [{"source": CURRENT_S5_SOURCE, "row": None, "why": "the prior tree IS this tree; a rung row has no S5 smoke of its own -- no source for it here"}]
        if not s5:
            return [{"source": CURRENT_S5_SOURCE, "row": None, "why": "the prior tree IS this tree and has minted no S5 smoke wave -- no S5 FULL row can stand for the production row"}]
        try:
            sident = lp.smoke_identity(plan, S5_SMOKE_ID, s5)
            srow = lp.smoke_row(prev_row, sident, max_children=len(prev_row.get("children") or []), max_timeout_s=int(prev_row["modal"]["timeout_s"]))
        except (ValueError, KeyError) as e:
            return [{"source": CURRENT_S5_SOURCE, "row": None, "why": f"this tree's S5 smoke row could not be reconstructed: {e}"}]
        return [{"source": CURRENT_S5_SOURCE, "row": srow, "identity": sident, "smoke_wave": s5,
                 "accepted_key": lp.accepted_key(plan["W7"], s5, prev_row["leg"], prev_row["engine"], prev_row["stage"], prev_row["shard"])}]
    ident = {"W": plan["W"], "W7": plan["W7"], "wave_uuid": plan["wave_uuid"], "results_root": plan["results_root"],
             "dicts": {"accepted": lp.dict_name(plan["W7"], plan["wave_uuid"], "accepted")}}
    out.append({"source": "production", "row": prev_row, "identity": ident, "smoke_wave": None,
                "accepted_key": lp.accepted_key(plan["W7"], plan["wave_uuid"], prev_row["leg"], prev_row["engine"], prev_row["stage"], prev_row["shard"])})
    # The reference smoke AND the budgeted smoke (the Petri legs are accepted under the budgeted smoke WAVE's Dict) -- each reconstructed at
    # the FULL child count and judged by clause (a) under its named clause (a budgeted row -- bon / misalign -- refuses there by name)
    for smoke_id, source in FULL_SHAPE_SMOKE_SOURCES.items():
        wave = (prior.get("smoke_waves") or {}).get(smoke_id)
        if not wave or prev_row.get("kind") == "rung":
            continue
        try:
            sident = lp.smoke_identity(plan, smoke_id, wave)
            srow = lp.smoke_row(prev_row, sident, max_children=len(prev_row.get("children") or []), max_timeout_s=int(prev_row["modal"]["timeout_s"]))
        except (ValueError, KeyError) as e:
            out.append({"source": source, "row": None, "why": f"the {smoke_id} smoke row could not be reconstructed: {e}"})
        else:
            out.append({"source": source, "row": srow, "identity": sident, "smoke_wave": wave,
                        "accepted_key": lp.accepted_key(plan["W7"], wave, prev_row["leg"], prev_row["engine"], prev_row["stage"], prev_row["shard"])})
    return out


def default_local_read_text(path: str) -> Optional[str]:
    """A grading-host-local file's text, None when absent -- the Petri stores the drive wrote live under ~/.local/state, not on the volume."""
    p = Path(path).expanduser()
    return p.read_text(errors="replace") if p.is_file() else None


#: fj18: the exception a never-created Modal Dict raises at its first read (`modal.exception.NotFoundError`), matched by NAME -- this module imports
#: no modal (the evaluator runs under injected getters in tests and on the grading host alike)
DICT_MISSING_EXCEPTION_NAMES: Tuple[str, ...] = ("NotFoundError",)


def is_dict_missing(e: BaseException) -> bool:
    """True when ``e`` is a Modal "Dict not found" (:data:`DICT_MISSING_EXCEPTION_NAMES`), whichever module defines it."""
    return type(e).__name__ in DICT_MISSING_EXCEPTION_NAMES


class RecordingReader:
    """A `read_text` that RECORDS every path it served with the sha256 of the text (None when absent) -- the evidence inventory a carried entry
    binds (the certificate re-reads every one)."""

    def __init__(self, read_text: Callable[[str], Optional[str]]) -> None:
        self._read, self.seen = read_text, {}

    def __call__(self, path: str) -> Optional[str]:
        text = self._read(path)
        self.seen[path] = None if text is None else _sha_text(text)
        return text


def evaluate_carry_over(*, this_plan: Mapping[str, Any], this_sha: str, priors: Sequence[Mapping[str, Any]], read_text: Callable[[str], Optional[str]],
                        accepted_get_for_dict: Callable[[str], Callable[[str], Any]], git: Callable[[Sequence[str]], Tuple[int, str]],
                        readiness_problems: Callable[..., List[str]], utc: Optional[str] = None,
                        replay: Optional[Callable[..., Dict[str, Any]]] = None, petri_proof: Optional[Callable[..., Dict[str, Any]]] = None,
                        criterion3_waivers: Optional[Sequence[Mapping[str, Any]]] = None,
                        local_read_text: Optional[Callable[[str], Optional[str]]] = None,
                        reaudit: Optional[Callable[..., Dict[str, Any]]] = None,
                        generation_proof_doc: Optional[Callable[[str, str], Optional[Mapping[str, Any]]]] = None) -> Dict[str, Any]:
    """The carry record for THIS plan (FINAL, production): every production row judged against every prior source's candidates in order; the first
    CARRIED candidate wins; every refusal is recorded by clause. Nothing is written or seeded here EXCEPT through ``replay`` (the ladder-queue
    replay, `replay(prev_row=, prior=, this_plan=, aid=, prev_root=, expected_n=) -> record`, invoked for a producer row whose rung is consumed in
    this plan once (a)-(d) hold -- it writes this wave's queue and its acceptance for that row). ``petri_proof(prev_row=, prior=, aid=, evidence=,
    read_text=) -> record`` is the sidecar proof a Petri row with an admitted chain difference needs; it MUST read every local input through the
    ``read_text`` it is handed -- a :class:`RecordingReader` over ``local_read_text`` (default: the grading host's filesystem) -- so the proof's
    inputs, absent sidecars included, enter the entry's `evidence_inventory_local` and are re-read at certification. ``criterion3_waivers`` are
    the explicit, SCOPED operator waiver documents (`{by, at, scope: {W, leg, view, legs}, reason, decision}`) of an ABSENT determinism
    spot-check -- a candidate is admitted under one only when the prior (W, leg, view, legs) fall inside its scope, with the document copied
    verbatim onto the entry; default none. ``reaudit(prev_row=, this_row=, prior=, aid=, root=, this_plan=) -> record``: THIS tree's
    request-evidence audit run over the carried root's stores into the carried entry's own evidence root, its verdicts bound
    (`producer_evidence`) -- clause (c) then reads those fresh verdicts (the historical ones predate `arm_binding`; the fitted cells have none)
    and the entry records `reaudited_by: <this tree>`. Without it the historical verdicts are read as they are."""
    if this_plan.get("materialisation") != "final":
        raise ValueError("carry-over admits into the FINAL plan only (production rows are launched from it)")
    if this_plan.get("W") != this_sha:
        raise ValueError(f"the plan is for W {this_plan.get('W')}, not the tree {this_sha}")
    this_ids = {"W": this_plan["W"], "W7": this_plan["W7"], "wave_uuid": this_plan["wave_uuid"]}
    carried: Dict[str, Any] = {}
    refused: Dict[str, Any] = {}
    for row in _production_rows(this_plan):
        sid = row["standup_id"]
        this_key = row["accepted_key"]
        attempts: List[Dict[str, Any]] = []
        won: Optional[Dict[str, Any]] = None
        # every source's candidates, ordered -- prior production, THIS tree's S5 FULL, prior S5, prior S3 (stable in operator order)
        ordered: List[Tuple[Mapping[str, Any], Dict[str, Any]]] = []
        for prior in priors:
            same_tree = prior["W"] == this_plan["W"]
            cands = _candidates(prior, row, current_tree=same_tree)
            if same_tree and not cands:
                attempts.append({"prior": prior["path"], "source": "-", "clause": "row_identity", "why": "the prior tree IS this tree and its plan lacks this row -- nothing to carry"})
            ordered += [(prior, c) for c in cands]
        ordered.sort(key=lambda pc: SOURCE_RANK.get(pc[1]["source"], 9))
        for prior, cand in ordered:
            prev_ids = {"W": prior["W"], "W7": prior["W7"], "wave_uuid": prior["wave_uuid"]}
            same_tree = prior["W"] == this_plan["W"]
            rec: Dict[str, Any] = {"prior": prior["path"], "prior_W": prior["W"], "source": cand["source"]}
            if cand.get("row") is None:
                rec.update(clause="row_identity", why=cand.get("why"))
                attempts.append(rec)
                continue
            prev_row, ident = cand["row"], cand["identity"]
            clauses: Dict[str, Any] = {}
            proof_inventory: Optional[Dict[str, Optional[str]]] = None      # the Petri proof's local inputs (set in the Petri branch)
            # The prior source's ACCEPTANCE is discovered and RECORDED first, independently of carry eligibility -- an acceptance
            # follows the store's COMPLETE.json by protocol, so a candidate that any clause below refuses still names the completed attempt
            # it refused (`prior_accepted_attempt`), and the launch belt (`carry_refused_holds`) holds the row on it whatever the clause
            # This is the ONE place a source's accepted Dict is read. A Dict that does not exist (`modal.Dict.from_name` is lazy; the
            # NotFoundError surfaces at the first read) is judged by the prior's LAUNCH record (`missing_dict_verdict`; the shared smoke-wave mints
            # decide nothing): NEVER LAUNCHED -> an empty acceptance map (`dict_missing` + `archive` say why the source offered nothing);
            # VANISHED -> refused by name, the evaluation stops; LAUNCH HISTORY UNKNOWN (no record) -> the candidate is refused by name and the
            # row HELD (a COMPLETE historical row must never re-spawn because its acceptance could not be read).
            dict_name = ident["dicts"]["accepted"]
            try:
                get = accepted_get_for_dict(dict_name)
                aid = get(cand["accepted_key"])
            except Exception as e:  # noqa: BLE001 -- classified by name below; anything else propagates unchanged
                if not is_dict_missing(e):
                    raise
                verdict, note = missing_dict_verdict(prior, source=cand["source"], wave_uuid=ident["wave_uuid"])
                if verdict == DICT_VANISHED:
                    raise ValueError(f"accepted Dict {dict_name} of the {cand['source']} source of {prior['path']} vanished: {note} -- its acceptances are "
                                     f"canonical work this evaluation cannot see; refusing rather than re-spawning") from e
                rec.update(accepted_key=cand["accepted_key"], dict=dict_name, attempt_id=None, dict_missing=dict_name, archive=note)
                if verdict == DICT_HISTORY_UNKNOWN:
                    rec.update(clause="complete", launch_history_unknown=True,
                               why=f"accepted Dict {dict_name} is missing and {note} -- the row is HELD ({CARRY_REFUSED_HOLD}), never re-spawned on an unread history")
                    attempts.append(rec)
                    continue
                aid = None
            rec.update(accepted_key=cand["accepted_key"], dict=dict_name, attempt_id=aid)
            if isinstance(aid, str) and aid:
                # An accepted BUDGET-LIMITED smoke (a budget flag on a child, fewer children than production) is completed
                # SMOKE work, never completed production work -- recorded here, and `carry_refused_holds` never holds the row on it; a
                # production acceptance and a FULL-shape smoke's (an S5 reference, the eight-leg Petri S3) establish the hold as before
                paa_root = lp.attempt_root_for(ident["results_root"], leg=prev_row["leg"], view=prev_row["engine"], stage=prev_row["stage"], shard=prev_row["shard"],
                                               attempt_id=aid, kind=prev_row.get("kind"))
                # fj14: the EXECUTED shape from evidence bound to the attempt (its modal_evidence.json children / recorded budget /
                # per-child argv) against the PRIOR tree's production row -- never from the candidate today's smoke_row reconstructs
                shape_reader = RecordingReader(read_text)                    # fj15: every shape input digest-bound, re-checked at certify
                ev_text = shape_reader(f"{paa_root}/modal_evidence.json")
                try:
                    ev_doc = json.loads(ev_text) if ev_text is not None else None
                except ValueError:
                    ev_doc = None
                prod_row = next((r for r in _production_rows(prior["plan"]) if r["standup_id"] == sid), prev_row)
                xs = executed_shape(evidence=ev_doc, prod_row=prod_row, prev_row=prev_row, root=paa_root, read_text=shape_reader,
                                    aid=aid, W=prior["W"], wave_uuid=ident["wave_uuid"])
                limited = xs["shape"] == EXECUTED_SHAPE_LIMITED
                rec["prior_accepted_attempt"] = {"attempt_id": aid, "dict": ident["dicts"]["accepted"], "accepted_key": cand["accepted_key"], "attempt_root": paa_root,
                                                 "prior": prior["path"], "prior_W": prior["W"], "wave_uuid": ident["wave_uuid"], "source": cand["source"],
                                                 "standup_id": prev_row["standup_id"], "executed_shape": xs, "full_shape": (not limited),
                                                 "evidence_inventory": dict(sorted(shape_reader.seen.items())),
                                                 "candidate_shape_problems": smoke_shape_problems(prev_row, row),      # today's reconstruction, informational only
                                                 "holds": (not limited),
                                                 "hold_note": ({EXECUTED_SHAPE_FULL: "completed production-equivalent work (evidence-bound): a refusal below HOLDS the row",
                                                                EXECUTED_SHAPE_UNKNOWN: "the executed shape could not be established from the attempt's evidence: the hold is RETAINED",
                                                                EXECUTED_SHAPE_LIMITED: "a BUDGETED smoke's acceptance (evidence-bound): its refusal stays recorded, it never holds the production row"}[xs["shape"]])}
            # (a)
            clauses["row_identity"] = row_identity_verdict(prev_row, row, prev=prev_ids, this=this_ids, prev_smoke_wave=cand["smoke_wave"])
            if not clauses["row_identity"]["ok"]:
                rec.update(clause="row_identity", why=clauses["row_identity"]["why"], clauses=clauses)
                if clauses["row_identity"].get("budgeted"):
                    rec["budgeted"] = True                 # a budgeted smoke is "never a carry source" -- no candidate, no hold (see carry_refused_holds)
                attempts.append(rec)
                continue
            if not isinstance(aid, str) or not aid:
                rec.update(clause="complete", why=(f"accepted Dict {dict_name} does not exist ({rec.get('archive')}) -- the {cand['source']} source offers nothing"
                                                   if rec.get("dict_missing") else f"no ACCEPTED attempt under {dict_name} key {cand['accepted_key']}"))
                attempts.append(rec)
                continue
            # (c) first: the evidence names the producer files (b) reads; every path read is RECORDED with its digest
            full = len(row.get("children") or []) if cand["source"] in SMOKE_ID_OF_SOURCE else None
            reader = RecordingReader(read_text)
            waiver = covering_waiver(criterion3_waivers, W=prior["W"], row=prev_row)
            ra: Optional[Dict[str, Any]] = None
            if reaudit is not None and not same_tree:              # fj17: a same-tree source's verdicts are already this tree's readers' work
                root_abs = lp.attempt_root_for(ident["results_root"], leg=prev_row["leg"], view=prev_row["engine"], stage=prev_row["stage"], shard=prev_row["shard"],
                                               attempt_id=aid, kind=prev_row.get("kind"))
                try:
                    ra = reaudit(prev_row=prev_row, this_row=row, prior=prior, aid=aid, root=root_abs, this_plan=this_plan)
                except Exception as e:  # noqa: BLE001 -- a re-audit that cannot run refuses the row by name, never falls back to the stale verdicts
                    rec.update(clause="complete", why=f"re-audit under this tree failed: {type(e).__name__}: {e}", clauses=clauses)
                    attempts.append(rec)
                    continue
            clauses["complete"] = complete_verdict(prev_plan=prior["plan"], prev_row=prev_row, identity=ident, aid=aid, read_text=reader, full_children=full,
                                                   criterion3_waiver=waiver, reaudit=ra, this_plan=this_plan)
            if not clauses["complete"]["ok"]:
                rec.update(clause="complete", why=clauses["complete"]["why"], clauses=clauses)
                attempts.append(rec)
                continue
            ev = clauses["complete"].pop("_evidence")
            root = clauses["complete"]["attempt_root"]
            if cand["source"] in SMOKE_ID_OF_SOURCE:
                want_smoke = SMOKE_ID_OF_SOURCE[cand["source"]]
                es = ev.get("smoke") if isinstance(ev.get("smoke"), Mapping) else {}
                n_kids = len(row.get("children") or [])
                if es.get("smoke_id") != want_smoke or (want_smoke == S5_SMOKE_ID and isinstance(es.get("budget"), Mapping) and es["budget"].get("kind") != lp.SMOKE_BUDGET_KIND_FULL) \
                        or (want_smoke == S3_SMOKE_ID and int(es.get("max_children") or 0) < n_kids):
                    rec.update(clause="row_identity", why=f"the attempt's own evidence says it ran as smoke {es.get('smoke_id')!r} with budget {es.get('budget')!r} / "
                                                          f"max_children {es.get('max_children')!r}, not the {want_smoke} FULL shape ({n_kids} children)", clauses=clauses, budgeted=True)
                    attempts.append(rec)
                    continue
            # (b)
            pset = producer_set(prev_row, ev, W=prior["W"])
            imports = direct_imports(git, prior["W"], pset["all"])
            gating = sorted((set(pset["all"]) | {p for ps in imports.values() for p in ps}) - set(CONTAINER_MODULES))
            clauses["producer_set"] = {**producer_set_verdict(gating, prev_sha=prior["W"], this_sha=this_sha, git=git, leg=str(prev_row.get("leg"))),
                                       "derived": {**{k: pset[k] for k in ("standup", "children", "modules", "writers")}, "imports": imports, "import_depth": IMPORT_DEPTH}}
            if not clauses["producer_set"]["ok"]:
                rec.update(clause="producer_set", why=clauses["producer_set"]["why"], clauses=clauses)
                attempts.append(rec)
                continue
            if petri_proof_required(prev_row, clauses["producer_set"]["admitted_differences"]):
                # A Petri-MODE store (the petri leg, the misalign leg) carries across a chain
                # difference or a petri_runner.py difference ONLY with the sidecar proof that the removed /
                # added legs were never reached in it (`petri_sidecar_proof`, injected as `petri_proof`); refused by name, never carried on faith
                proof: Dict[str, Any]
                gen_doc = None
                if generation_proof_scope_problem(prev_row, source=cand["source"], evidence=ev) is None:
                    # A generation-only proof document for a petri-MODE row without a Petri completion bundle -- re-derived from the
                    # volume bytes through the RECORDING reader (its inputs enter the evidence inventory) and held to the document field by field
                    try:
                        gen_doc = generation_proof_doc(row["standup_id"], aid) if generation_proof_doc is not None else None      # keyed by the CONSUMING row's id
                    except Exception as e:  # noqa: BLE001 -- a malformed document is a refusal by name
                        gen_doc = {"_unreadable": f"{type(e).__name__}: {e}"}
                if gen_doc is not None and "_unreadable" in gen_doc:
                    proof = {"ok": False, "kind": GENERATION_PROOF_KIND, "why": f"the {GENERATION_PROOF_RULING} proof document is unreadable: {gen_doc['_unreadable']}"}
                elif gen_doc is not None:
                    grec, gprobs = generation_proof_record(gen_doc, prev_row=prev_row, source=cand["source"], root=root, evidence=ev, read_text=reader, git=git, prev_W=prior["W"], this_W=this_sha)
                    proof = grec if grec is not None else {"ok": False, "kind": GENERATION_PROOF_KIND, "ruling": GENERATION_PROOF_RULING, "why": "; ".join(gprobs)}
                elif petri_proof is None:
                    proof = {"ok": False, "why": "a Petri row with an admitted fallback-chain difference needs the sidecar proof (auditor: zero uses of the "
                                                 "removed leg; judge: n/a) -- this evaluation was given no petri_proof reader"
                                                 + (f" and no {GENERATION_PROOF_RULING} proof document at RS/{GENERATION_PROOF_REL}/{row['standup_id']}__{aid}.json"
                                                    if generation_proof_scope_problem(prev_row, source=cand["source"], evidence=ev) is None else "")}
                else:
                    local_reader = RecordingReader(local_read_text or default_local_read_text)
                    try:
                        proof = petri_proof(prev_row=prev_row, prior=prior, aid=aid, evidence=ev, read_text=local_reader)
                    except Exception as e:  # noqa: BLE001 -- a proof that cannot be read is a refusal, never a pass
                        proof = {"ok": False, "why": f"sidecar proof unreadable: {type(e).__name__}: {e}"}
                    proof_inventory = dict(sorted(local_reader.seen.items()))
                    if proof.get("ok") is True and not proof_inventory:
                        proof = {**proof, "ok": False, "why": "the sidecar proof read no local input through the recording reader -- its inputs cannot be bound"}
                    if proof.get("kind") == GENERATION_PROOF_KIND:
                        proof = {**proof, "ok": False, "why": f"a {GENERATION_PROOF_KIND} record from the Petri store proof reader -- only a {GENERATION_PROOF_RULING} "
                                                              "document over a misalign attempt's own bytes may carry that kind"}
                clauses["producer_set"]["petri_sidecar_proof"] = proof
                if proof.get("ok") is not True:
                    clauses["producer_set"]["ok"] = False
                    clauses["producer_set"]["why"] = f"Petri sidecar proof: {proof.get('why')}"
                    rec.update(clause="producer_set", why=clauses["producer_set"]["why"], clauses=clauses)
                    attempts.append(rec)
                    continue
            # (d)
            clauses["engine_identity"] = engine_identity_verdict(this_plan=this_plan, this_row=row, root=root, aid=aid, prev_W=prior["W"], evidence=ev, read_text=reader,
                                                                 readiness_problems=readiness_problems, served_standup_id=prev_row["standup_id"])
            if not clauses["engine_identity"]["ok"]:
                rec.update(clause="engine_identity", why=clauses["engine_identity"]["why"], clauses=clauses)
                attempts.append(rec)
                continue
            # (e) -- through the replay when this plan consumes the row's rung (fj3)
            lad_inv = (clauses["complete"].get("ladder_inventory") or {})
            do_replay = None
            if replay is not None:
                do_replay = (lambda _pr=prev_row, _prior=prior, _aid=aid, _root=root, _n=lad_inv.get("n"), _wave=ident["wave_uuid"]:
                             replay(prev_row=_pr, prior=_prior, this_plan=this_plan, aid=_aid, prev_root=_root, expected_n=_n, prev_wave_uuid=_wave))
            clauses["ladder_queue"] = ladder_queue_verdict(prev_row, this_plan, replay=do_replay)
            if not clauses["ladder_queue"]["ok"]:
                rec.update(clause="ladder_queue", why=clauses["ladder_queue"]["why"], clauses=clauses)
                attempts.append(rec)
                continue
            won = {"accepted_key": this_key, "attempt_id": aid, "attempt_root": root,
                   "carried_from": {"W": prior["W"], "W7": prior["W7"], "wave_uuid": ident["wave_uuid"], "attempt_id": aid, "tree_sha": prior["W"], "rs": prior["path"],
                                    "accepted_key": cand["accepted_key"], "dict": ident["dicts"]["accepted"], "standup_id": prev_row["standup_id"], "source": cand["source"]},
                   "clauses": clauses, "s5_full_row": bool(clauses["row_identity"].get("s5_full_row")), "full_smoke_row": bool(clauses["row_identity"].get("full_smoke_row")),
                   "smoke_source": clauses["row_identity"].get("smoke_id"), "refused_before": attempts,
                   "evidence_inventory": dict(sorted(reader.seen.items()))}
            c3 = ((clauses["complete"].get("checks") or {}).get("determinism") or {}).get("criterion3")
            if c3 is not None:
                won["criterion3"] = copy.deepcopy(c3)                              # fj22: the QUALIFICATION, surfaced in the certificate's carry block
            if clauses["complete"].get("waived"):
                won["criterion3_waiver"] = copy.deepcopy(dict(waiver))          # verbatim: the operator's document, not a summary of it
            if proof_inventory:
                won["evidence_inventory_local"] = proof_inventory              # the Petri proof's inputs, absent sidecars as None
            if rec.get("prior_accepted_attempt") is not None:
                won["prior_accepted_attempt"] = rec["prior_accepted_attempt"]  # fj15: the acceptance's executed shape, as classified at discovery
            if same_tree:
                won["reaudited_by"] = REAUDIT_NOT_REQUIRED_SAME_TREE          # fj17: the verdicts are already this tree's readers' work
                won["reaudit"] = None
            if ra is not None:
                won["reaudited_by"] = ra["reaudited_by"]                       # fj7: the tree whose audit wrote the verdicts clause (c) read
                won["reaudit_evidence_dir"] = ra["evidence_root"]
                won["reaudit"] = {k: ra.get(k) for k in ("reaudited_by", "evidence_root", "store_root", "rc", "argv", "tool", "n_files", "index_sha256", "local_evidence_root")}
                won["reaudit"]["producer_evidence"] = dict(ra["producer_evidence"])
                won["reaudit"]["original"] = {"attempt_id": aid, "attempt_root": root, "W": prior["W"], "wave_uuid": ident["wave_uuid"]}
            break
        if won:
            carried[sid] = won
        else:
            refused[sid] = {"accepted_key": this_key, "attempts": attempts, "why": (attempts[-1]["why"] if attempts else "no prior source holds this row"),
                            "clause": (attempts[-1].get("clause") if attempts else None)}
    return {"schema": CARRY_OVER_SCHEMA, "W": this_plan["W"], "W7": this_plan["W7"], "wave_uuid": this_plan["wave_uuid"], "launch_plan_sha": this_plan.get("launch_plan_sha"),
            "this_sha": this_sha, "priors": [{k: p[k] for k in ("path", "plan_path", "W", "W7", "wave_uuid", "launch_plan_sha", "materialisation", "smoke_waves")} for p in priors],
            "clauses": list(CLAUSES), "s5_full_clause": S5_FULL_CLAUSE, "s3_full_clause": S3_FULL_CLAUSE, "carried": carried, "refused": refused,
            "n_carried": len(carried), "n_refused": len(refused),
            "container_modules_rule": "recorded per carried entry under clauses.producer_set.container_modules; readers by ruling, not gating",
            "evaluated_utc": utc or _utc()}


# --------------------------------------------------------------------------- #
# seeding, the record file, the launcher's view                                 #
# --------------------------------------------------------------------------- #

def carried_provenance_key(accepted_key: str) -> str:
    return f"carried|{accepted_key}"


def carried_accepted_entries(record: Mapping[str, Any]) -> Dict[str, str]:
    return {e["accepted_key"]: e["attempt_id"] for e in (record.get("carried") or {}).values() if isinstance(e, Mapping)}


CARRY_REFUSED_HOLD = "carry_refused_hold"


def carry_refused_holds(record: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{standup_id: {reason, attempt_id, attempt_root, prior, clause, why}}`` of every REFUSED row whose prior source holds an ACCEPTED attempt
    for it (on `prior_accepted_attempt`, recorded BEFORE clause (a), so the hold stands whichever clause refused -- row identity included):
    an acceptance follows the store's COMPLETE.json by protocol, so such a row is completed canonical work the carry
    could not admit -- it is HELD by name (`carry_refused_hold`), never a spawn candidate. A refusal with no accepted attempt anywhere (nothing
    completed) is not a hold -- EXCEPT a source whose accepted Dict is missing with no launch record on the prior (`launch_history_unknown`:
    whether something completed cannot be read, and unknown holds); neither is a BUDGETED smoke's acceptance (`full_shape: False` -- a 10-goal
    S3 BoN smoke is completed smoke work, not the production row's; its carry refusal stays recorded and the production row runs)."""
    if not isinstance(record, Mapping):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sid, e in (record.get("refused") or {}).items():
        if not isinstance(e, Mapping):
            continue
        acc = []
        for a in (e.get("attempts") or []):
            if not isinstance(a, Mapping):
                continue
            if a.get("launch_history_unknown") is True:
                # fj20: the source's accepted Dict is missing and the prior carries no launch record -- whether a completed row sits
                # behind it is UNKNOWN, and unknown is never permission to re-spawn: held until the record is written
                acc.append((a, {"attempt_id": None, "attempt_root": None, "hold_kind": DICT_HISTORY_UNKNOWN}))
                continue
            paa = a.get("prior_accepted_attempt")
            if isinstance(paa, Mapping) and isinstance(paa.get("attempt_id"), str) and paa["attempt_id"]:
                if paa.get("full_shape") is False:
                    continue                    # fj13 / fj14: an acceptance whose EXECUTED shape is proven limited (evidence-bound) is no hold; unknown RETAINS it
                # A candidate refused ONLY as a BUDGETED smoke ("never a carry source", `budgeted: true` on the attempt) is no candidate
                # -- and its acceptance holds nothing when the EVIDENCE proves the budget (executed shape `limited`, released above). An `unknown`
                # executed shape (argv unreadable) RETAINS the hold even on a budgeted candidate: missing, inconsistent or unbound history is
                # never permission to re-run; a proven-full acceptance refused as budgeted is a contradiction: held.
                acc.append((a, paa))
            elif isinstance(a.get("attempt_id"), str) and a.get("attempt_id"):        # a record written before fj8: the acceptance read after clause (a)
                acc.append((a, {"attempt_id": a["attempt_id"], "attempt_root": None}))
        if not acc:
            continue
        a, paa = acc[-1]
        unknown = paa.get("hold_kind") == DICT_HISTORY_UNKNOWN
        out[str(sid)] = {"reason": CARRY_REFUSED_HOLD, "hold_kind": (DICT_HISTORY_UNKNOWN if unknown else "accepted_attempt"),
                         "attempt_id": paa.get("attempt_id"), "attempt_root": paa.get("attempt_root"), "prior": a.get("prior"),
                         "prior_W": a.get("prior_W"), "source": a.get("source"), "clause": a.get("clause"), "why": a.get("why"),
                         "executed_shape": paa.get("executed_shape"), "dict_missing": a.get("dict_missing"),
                         "note": ("the source's accepted Dict is missing and the prior carries no launch record (fj20): whether a completed attempt sits behind it is "
                                  f"unknown -- held until {SMOKES_LAUNCHED_REL} is written, never re-spawned on an unread history" if unknown else
                                  "a completed attempt of a prior source exists for this row and the carry refused it -- held, never re-spawned (the user: completed work is canonical)")}
    return out


def carried_standup_ids(record: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{standup_id: carried_from}`` of a record (empty for None) -- what `select_launch_rows` skips by name."""
    if not isinstance(record, Mapping):
        return {}
    return {sid: dict(e["carried_from"]) for sid, e in (record.get("carried") or {}).items() if isinstance(e, Mapping) and isinstance(e.get("carried_from"), Mapping)}


def carried_attempt_roots(record: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """``{attempt_id: attempt_root}`` of every carried entry -- the REAL root under the prior tree's results root (retrieval and every
    downstream reader resolve a carried attempt here, never under the current results root where it does not exist)."""
    if not isinstance(record, Mapping):
        return {}
    return {e["attempt_id"]: e["attempt_root"] for e in (record.get("carried") or {}).values()
            if isinstance(e, Mapping) and isinstance(e.get("attempt_id"), str) and isinstance(e.get("attempt_root"), str)}


def attempt_root_resolver(record: Optional[Mapping[str, Any]], default: Callable[..., str]) -> Callable[..., str]:
    """``resolve(results_root, row, attempt_id, **kw)``: a CARRIED attempt's recorded root, else ``default(results_root, row, attempt_id, **kw)``."""
    roots = carried_attempt_roots(record)

    def resolve(results_root: str, row: Mapping[str, Any], attempt_id: str, **kw: Any) -> str:
        return roots.get(attempt_id) or default(results_root, row, attempt_id, **kw)
    return resolve


def seed_accepted(record: Mapping[str, Any], put: Callable[..., bool], get: Callable[[str], Any]) -> Dict[str, List[str]]:
    """Write every carried attempt into THIS wave's accepted Dict under THIS wave's key -- `put(key, attempt_id, skip_if_exists=True)`, the
    container's own acceptance primitive -- and its provenance under the companion key. A key already naming ANOTHER attempt refuses (a fresh
    attempt won it; the carry does not overwrite an acceptance)."""
    seeded, present, conflict = [], [], []
    for sid, e in (record.get("carried") or {}).items():
        key, aid = e["accepted_key"], e["attempt_id"]
        if put(key, aid, skip_if_exists=True):
            seeded.append(sid)
        else:
            holder = get(key)
            (present if holder == aid else conflict).append(sid)
            if holder != aid:
                continue
        put(carried_provenance_key(key), dict(e["carried_from"]), skip_if_exists=True)
    if conflict:
        raise ValueError(f"carry-over refused: this wave's accepted Dict already names another attempt for {conflict} -- a carried attempt never overwrites an acceptance")
    return {"seeded": seeded, "already_present": present}


def write_carry_over(rs: Path, record: Mapping[str, Any]) -> Path:
    p = Path(rs) / CARRY_OVER_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, p)
    return p


def read_carry_over(rs: Path) -> Optional[Dict[str, Any]]:
    p = Path(rs) / CARRY_OVER_FILE
    return json.loads(p.read_text()) if p.is_file() else None


def record_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# the certificate's re-validation                                               #
# --------------------------------------------------------------------------- #

#: fj28: a petri-MODE row (the petri leg, the misalign leg: `mode: petri`) carries a difference in these producers -- or any chain-list-kind
#: admission -- only with the covering sidecar proof (zero uses of the removed / added auditor and judge legs in the store)
PETRI_PROOF_TRIGGER_PATHS: Tuple[str, ...] = ("src/sisteer/capeval/tiers/petri_runner.py", "src/sisteer/capeval/petri/auditor_fallback.py")
#: the legs the plan generator runs in Petri MODE (`seedrep_launch_plan.leg_table(...)[leg]["mode"] == "petri"`; pinned equal by test). A plan ROW
#: carries no `mode` field -- the archived misalign row reads `mode: None` -- so the mode is the LEG's, read here, never the row's absence.
PETRI_MODE_LEGS: Tuple[str, ...] = ("petri", "misalign")


def is_petri_mode_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("leg")) in PETRI_MODE_LEGS or str(row.get("mode")) == "petri"


def petri_proof_required(row: Mapping[str, Any], admitted: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the sidecar proof is REQUIRED for ``row`` given its admitted differences -- derived from the row's leg / mode and the differences
    themselves, never from whether a record happens to contain a proof. The petri LEG requires it for ANY admitted
    difference; a petri-MODE row needs it for a chain-kind admission or a difference in :data:`PETRI_PROOF_TRIGGER_PATHS`."""
    adm = [a for a in (admitted or []) if isinstance(a, Mapping)]
    if not adm or not is_petri_mode_row(row):
        return False
    if str(row.get("leg")) == "petri":
        return True
    return any(a.get("kind") == ADMITTED_KIND_CHAIN or str(a.get("path")) in PETRI_PROOF_TRIGGER_PATHS for a in adm)


#: A GENERATION-ONLY sidecar proof for a petri-MODE row whose attempt has no Petri completion bundle (the misalign leg: its children
#: generate through the proxy, its judging is a grading-host step downstream) -- the operator supplies the document derived from the pulled bytes
#: (`scripts/sdfbase/seedrep/misalign_generation_proof.py`) at RS/carry_over/proofs/<standup_id>__<attempt_id>.json; the evaluator and the
#: certificate RE-DERIVE it from the volume bytes through their readers and hold the document to the re-derivation field by field.
GENERATION_PROOF_KIND = "generation_only"
GENERATION_PROOF_RULING = "D191"
GENERATION_PROOF_LEGS: Tuple[str, ...] = ("misalign",)
GENERATION_PROOF_SOURCES: Tuple[str, ...] = (CURRENT_S5_SOURCE, "s5_smoke")
GENERATION_PROOF_JUDGE_SITE = "scripts/sdfbase/full22/misalign/misalign_judge.py (the grading host, D8)"
GENERATION_PROOF_REL = "carry_over/proofs"
GENERATION_PROOF_ENTRY_KEYS: Tuple[str, ...] = ("kind", "ruling", "ok", "removed_leg", "added_legs", "served_models", "inputs", "proof_sha256", "judge", "bytes_read", "target_models", "policy")


def _generation_proof_module() -> Any:
    """`scripts/sdfbase/seedrep/misalign_generation_proof.py`, loaded by path (a sibling tree, stdlib-only)."""
    import importlib.util as ilu
    path = Path(__file__).resolve().parents[1] / "sdfbase" / "seedrep" / "misalign_generation_proof.py"
    name = "seedrep_misalign_generation_proof"
    mod = sys.modules.get(name)
    if mod is None:
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def generation_proof_scope_problem(prev_row: Mapping[str, Any], *, source: str, evidence: Mapping[str, Any]) -> Optional[str]:
    """Why a generation-only proof may NOT stand for this row (None = in scope): the misalign leg only (a Petri-LEG row keeps the store proof),
    a `current_s5_full` / `s5_smoke` source, an attempt whose evidence carries NO Petri completion bundle."""
    if str(prev_row.get("leg")) not in GENERATION_PROOF_LEGS:
        return (f"a {GENERATION_PROOF_KIND} proof stands only for the {list(GENERATION_PROOF_LEGS)} leg(s); a {prev_row.get('leg')!r} row keeps the D115 store proof "
                "(the Petri drive's cells)")
    if str(source) not in GENERATION_PROOF_SOURCES:
        return f"a {GENERATION_PROOF_KIND} proof stands only for a {list(GENERATION_PROOF_SOURCES)} source, not {source!r}"
    if evidence.get("petri") is not None:
        return f"the attempt's evidence carries a Petri completion bundle -- the D115 store proof applies, not a {GENERATION_PROOF_KIND} one"
    return None


def rs_generation_proof_reader(rs: Path) -> Callable[[str, str], Optional[Mapping[str, Any]]]:
    """The operator's documents at `RS/carry_over/proofs/<standup_id>__<attempt_id>.json` (None when absent; a malformed one raises)."""
    def read(standup_id: str, attempt_id: str) -> Optional[Mapping[str, Any]]:
        p = Path(rs) / GENERATION_PROOF_REL / f"{standup_id}__{attempt_id}.json"
        if not p.is_file():
            return None
        doc = json.loads(p.read_text())
        if not isinstance(doc, Mapping):
            raise ValueError(f"{p}: not an object")
        return {**doc, "_document_path": str(p)}
    return read


def generation_proof_policy(prev_row: Mapping[str, Any], *, git: Callable[[Sequence[str]], Tuple[int, str]], prev_W: str, this_W: str) -> Dict[str, Any]:
    """The proof's POLICY, derived from authorities the document cannot alter -- the REMOVED / ADDED chain identities from the
    archived row's manifests (`--manifest` on its standup / children argv) read at the prior and launch trees through git and diffed per role
    (`misalign_generation_proof.chain_policy`), aliased through THIS tree's `auditor_label`; the TARGET set from the archived row's
    `cells_served_expected` (canonical + controls) as THIS tree spells their virtual models (`defense_cells.virtual_model_name`). Any input that
    cannot be read refuses (ValueError) -- no constant stands in."""
    mgp = _generation_proof_module()
    manifests: List[str] = []
    for argv in [prev_row.get("argv") or []] + [k.get("argv") or [] for k in (prev_row.get("children") or [])]:
        for i_, a in enumerate(argv):
            a = str(a)
            if a == "--manifest" and i_ + 1 < len(argv):
                manifests.append(str(argv[i_ + 1]))
            elif a.startswith("--manifest="):
                manifests.append(a[len("--manifest="):])
    if not manifests:
        raise ValueError("the archived row's argv names no --manifest -- the chain policy cannot be derived")
    try:
        from sisteer.capeval.petri.auditor_fallback import auditor_label  # noqa: PLC0415 -- this tree's alias rule (src on the path)
        from sisteer.capeval.petri import defense_cells as dc  # noqa: PLC0415 -- this tree's vmodel spelling
    except ImportError as e:
        raise ValueError(f"this tree's auditor_label / defense_cells are not importable ({e}) -- aliases and the target set cannot be derived") from e
    policy = mgp.chain_policy(git, prev_W=prev_W, this_W=this_W, manifests=manifests, label_of=auditor_label)
    cse = prev_row.get("cells_served_expected")
    if not isinstance(cse, Mapping) or not isinstance(cse.get("canonical"), list) or not cse.get("canonical"):
        raise ValueError("the archived row records no cells_served_expected.canonical -- the target set cannot be derived")
    canonical = [str(c) for c in cse.get("canonical") or []]
    controls = [str(c) for c in cse.get("controls") or []]
    try:
        # canonical cells as `enumerate_cells` spells them (the base route is petri__base); the DERIVED control legs (§3: never manifest inputs, so
        # `virtual_model_name` refuses them by design) through the same `_vmodel` with the class enumerate_cells gives them (defense_cells:1501)
        target = sorted({str(dc.virtual_model_name(c)) for c in canonical} | {str(dc._vmodel(c, "control")) for c in controls})
    except Exception as e:  # noqa: BLE001 -- a cell id this tree cannot spell refuses by name
        raise ValueError(f"this tree cannot spell a virtual model for the archived row's cells ({type(e).__name__}: {e})") from e
    return {**policy, "cells": sorted(set(canonical) | set(controls)), "target": target}


def generation_proof_record(doc: Any, *, prev_row: Mapping[str, Any], source: str, root: str, evidence: Mapping[str, Any],
                            read_text: Callable[[str], Optional[str]], git: Callable[[Sequence[str]], Tuple[int, str]], prev_W: str, this_W: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The entry record ``(record, problems)`` for a generation-only proof DOCUMENT over the attempt at ``root``: the scope guard; the
    document's kind / ruling / schema; its `inputs` EQUAL to the set derived from the archived row's children under the canonical root (a missing
    input, an extra one, or one outside the root refuses); the proof RE-DERIVED from the volume bytes through ``read_text`` (never the document's
    `ok`) and held to the document field by field (VERDICT_KEYS); the record carries the re-derived verdict, the document's sha256, the judge site
    and the chain basis. Any problem -> (None, [why, ...])."""
    mgp = _generation_proof_module()
    who = f"{GENERATION_PROOF_KIND} proof ({GENERATION_PROOF_RULING})"
    scope = generation_proof_scope_problem(prev_row, source=source, evidence=evidence)
    if scope is not None:
        return None, [f"{who}: out of scope -- {scope}"]
    if not isinstance(doc, Mapping):
        return None, [f"{who}: the document is not an object"]
    doc_body = {k: v for k, v in doc.items() if k != "_document_path"}
    probs: List[str] = []
    if doc_body.get("kind", GENERATION_PROOF_KIND) != GENERATION_PROOF_KIND:
        probs.append(f"{who}: the document's kind is {doc_body.get('kind')!r}, not {GENERATION_PROOF_KIND!r}")
    if doc_body.get("ruling", GENERATION_PROOF_RULING) != GENERATION_PROOF_RULING:
        probs.append(f"{who}: the document's ruling label is {doc_body.get('ruling')!r}, not {GENERATION_PROOF_RULING!r}")
    if doc_body.get("schema") != mgp.SCHEMA:
        probs.append(f"{who}: the document's schema is {doc_body.get('schema')!r}, not {mgp.SCHEMA!r}")
    if probs:
        return None, probs
    try:
        expected = mgp.expected_inputs(prev_row.get("children") or [], root)
    except ValueError as e:
        return None, [f"{who}: the archived row's children do not locate the store ({e})"]
    try:
        policy = generation_proof_policy(prev_row, git=git, prev_W=prev_W, this_W=this_W)
    except ValueError as e:
        return None, [f"{who}: the chain policy cannot be derived from the trees ({e}) -- refusing"]
    # fj31: the document's lists must EQUAL the tree-derived policy -- the document never decides what the derivation searches for
    for field, path_, want in (("removed legs", ("removed_leg", "legs"), policy["removed"]), ("added legs", ("added_legs", "legs"), policy["added"]),
                               ("target models", ("target_models",), policy["target"])):
        got = doc_body
        for k_ in path_:
            got = got.get(k_) if isinstance(got, Mapping) else None
        if sorted(str(x) for x in (got or [])) != list(want):
            probs.append(f"{who}: the document's {field} {sorted(str(x) for x in (got or []))[:6]} are not the tree-derived policy {want[:6]} "
                         f"(manifests {policy['manifests']} at {prev_W[:12]} -> {this_W[:12]}); refusing")
    if doc_body.get("policy") != policy:
        probs.append(f"{who}: the document's policy block is not the one derived from the trees; refusing")
    if not isinstance(doc_body.get("bytes_read"), int) or isinstance(doc_body.get("bytes_read"), bool):
        probs.append(f"{who}: the document records no bytes_read -- a proof without its byte count is incomplete; refusing")
    if probs:
        return None, probs
    rec_inputs = doc_body.get("inputs") if isinstance(doc_body.get("inputs"), Mapping) else None
    if rec_inputs is None:
        return None, [f"{who}: the document records no inputs"]
    outside = sorted(p_ for p_ in rec_inputs if not str(p_).startswith(str(root).rstrip("/") + "/"))
    missing = sorted(p_ for p_ in expected if p_ not in rec_inputs)
    extra = sorted(p_ for p_ in rec_inputs if p_ not in expected)
    if outside:
        probs.append(f"{who}: {len(outside)} recorded input(s) lie outside the canonical root {root}: {outside[:3]}")
    if missing:
        probs.append(f"{who}: {len(missing)} store file(s) the archived row's children name are not among the document's inputs: {missing[:3]}")
    if extra:
        probs.append(f"{who}: {len(extra)} recorded input(s) are not files the archived row's children name: {[e for e in extra if e not in outside][:3]}")
    if probs:
        return None, probs
    try:
        derived = mgp.derive_over(read_text, root=root, files=expected, removed=policy["removed"], added=policy["added"], target=policy["target"], policy=policy,
                                  chain_note=doc_body.get("chain"))
    except (FileNotFoundError, ValueError) as e:
        return None, [f"{who}: the proof cannot be re-derived from the volume bytes ({type(e).__name__}: {str(e)[:160]}) -- refusing"]
    for k in mgp.VERDICT_KEYS:
        if derived.get(k) != doc_body.get(k):
            probs.append(f"{who}: re-derived from the volume bytes, {k!r} is not the document's (the re-derivation wins: "
                         f"{str(derived.get(k))[:100]} != {str(doc_body.get(k))[:100]})")
    if probs:
        return None, probs
    if derived.get("ok") is not True:
        return None, [f"{who}: the re-derived proof does NOT hold: {derived.get('why')}"]
    body_text = json.dumps(doc_body, sort_keys=True, separators=(",", ":"))
    record = {"kind": GENERATION_PROOF_KIND, "ruling": GENERATION_PROOF_RULING, "ok": True,
              **{k: derived[k] for k in ("cells", "n_rows", "n_response_evidence", "removed_leg", "added_legs", "served_models", "models_reported", "per_cell", "inputs", "bytes_read",
                                          "target_models", "foreign_models", "policy")},
              "proof_sha256": _sha_text(body_text), "judge": {"in_attempt": False, "site": GENERATION_PROOF_JUDGE_SITE},
              "basis": {"derivation": derived.get("basis"), "chain": {"prev": policy["prev"], "this": policy["this"], "manifests": policy["manifests"]}, "chain_note": doc_body.get("chain")},
              "document_path": doc.get("_document_path"), "why": None}
    return record, []


def default_accepted_get_for_dict() -> Callable[[str], Callable[[str], Any]]:
    """The certificate's default reader of the PRIOR acceptance: the LIVE Modal Dicts by name (fc11b's precedent for prior smoke PASS records).
    `modal` is imported lazily -- where it is not importable the reader raises and the recheck refuses by name (a caller injects a reader)."""
    def for_dict(name: str) -> Callable[[str], Any]:
        try:
            import modal  # noqa: PLC0415 -- lazy: this module is otherwise modal-free
            from_name = modal.Dict.from_name
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"no live accepted-Dict reader: modal is not importable here ({type(e).__name__}: {e})") from e
        return from_name(name, create_if_missing=False).get
    return for_dict


def default_petri_proof(*, prev_row: Mapping[str, Any], prior: Mapping[str, Any], aid: str, evidence: Mapping[str, Any], read_text: Callable[[str], Optional[str]],
                        git: Callable[[Sequence[str]], Tuple[int, str]], home: Optional[Path] = None) -> Dict[str, Any]:
    """The sidecar proof re-derived from its inputs: the store's cell dirs from the attempt's completion bundle (`petri_cell_dirs`), read through
    ``read_text`` (a RecordingReader binds every input), the judge chain's presence in the prior tree from its `auditor_fallback.py`."""
    rc, text = git(["show", f"{prior['W']}:src/sisteer/capeval/petri/auditor_fallback.py"])
    return petri_sidecar_proof(petri_cell_dirs(evidence, home=home), read_text, judge_chain_in_prior=(rc == 0 and "judge_fallback" in text))


REPO_ROOT = Path(__file__).resolve().parents[2]      # scripts/modal/<this file> -> the repository (the certificate's default git objects)


def producer_set_recheck(entry: Mapping[str, Any], *, row: Mapping[str, Any], plan: Mapping[str, Any], git: Callable[[Sequence[str]], Tuple[int, str]],
                         read_text: Callable[[str], Optional[str]], accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None,
                         local_read_text: Optional[Callable[[str], Optional[str]]] = None, petri_proof: Optional[Callable[..., Dict[str, Any]]] = None,
                         petri_home: Optional[Path] = None, load_prior: Callable[[Path], Mapping[str, Any]] = None) -> List[str]:
    """The carried entry's clauses (b) and the identity they rest on, RE-DERIVED at certification from authorities carry_over.json cannot alter.

    AUTHORITY MAP (fj28) -- every input the recheck consumes: its WRITER, its LOCATION, and WHAT BINDS IT. The only binding
    roots are the CONSUMING plan row, the ACCEPTED Dict record the producer wrote at acceptance, the volume bytes under the canonical root the
    accepted record + consuming row spell (held by digest), the RS/prior archives, and the git objects. Nothing below bottoms out in the entry.

      prior W / W7        writer: the prior tree's plan generator (archived, launch_plan_sha-bound)       location: carried_from.W (a claim)
                          bound by: the ACCEPTED Dict `seedrep-<W7>-<wave8>-accepted` at key (W7|wave|leg|view|stage|shard from the CONSUMING row)
                          naming the attempt id -- a (W7, wave) the producer never accepted under has no such record -- and by the evidence's W.
      wave_uuid           writer: RS/wave.json / smokes/<Sid>/wave.json (minted)                          location: carried_from.wave_uuid (a claim)
                          bound by: the accepted Dict's name / key, and the evidence's wave_uuid.
      standup_id (hist.)  writer: the prior plan (archived) / `lp.smoke_row`'s spelling                   location: carried_from.standup_id (a claim)
                          bound by: DERIVED from the consuming row + source; the entry's must equal; the evidence's standup_id must equal.
      attempt_id          writer: the producer at acceptance (`accepted.put(key, attempt_id)`)             location: the ACCEPTED Dict record (read LIVE)
                          bound by: the accepted record itself -- the entry's and the provenance's must equal it. The record carries the attempt
                          id ONLY (no evidence digest), so the evidence is bound through the canonical root below, never through a digest it names.
      attempt_root        writer: the app (`lp.attempt_root_for`)                                          location: derived, never read from the entry
                          bound by: (prior W7, source results root, consuming row axes, the accepted attempt id); the entry's must equal.
      modal_evidence.json writer: the container at completion                                              location: the volume under the canonical root
      + completion sources (RESULT, COMPLETE.json, ladder_enqueued.json for a producer row)
                          bound by: read at the CANONICAL root through the certificate's reader; each path must carry a non-null digest in the
                          entry's evidence_inventory EQUAL to the bytes read (the inventory is an entry claim -- it is trusted only as re-verified
                          against the canonical root's bytes, and only for the canonical root's paths).
      archived plan row   writer: the prior tree's plan generator                                          location: RS/prior/<W7>/launch_plan_*.json
                          bound by: launch_plan_sha (content), the archive's W == the accepted-record-bound W; the row re-derived (production, or
                          smoke row under the carried wave) and `row_identity_verdict` RE-RUN against the consuming row.
      producer file list  writer: derived (`producer_set` over the archived row + evidence, `direct_imports` at the prior tree via git)
                          bound by: recomputed; the record's list must equal it.
      blob ids            writer: git                                                                      bound by: git objects at prior W / this W.
      hunks / chain lines writer: `diff_hunks` / `admitted_difference` over git objects                    bound by: recomputed; pinned tables in this tree's code.
      Petri sidecar proof writer: `petri_sidecar_proof` over the store cells the evidence's completion bundle names (grading-host-local bytes)
                          bound by: the REQUIREMENT derived from the archived row's leg / mode and the recomputed differences; the proof re-derived
                          from the re-read evidence's cells through the local reader, its inputs equal to evidence_inventory_local by digest.

    ORDER: (1) the prior accepted record (P1-a); (2) the canonical root; (3) the evidence and completion sources at that root, bound to the
    provenance and the inventory (P1-b); (4) the archive, the archived row, its identity re-run (fj26 P1-B); (5) the producer set, blobs, the
    three-set equality and every actual differing file's admission (fj25 / fj26 P1-A); (6) the Petri sidecar proof (P2). Every failure refuses
    by name; nothing is skipped for want of a reader (a missing reader is itself a refusal)."""
    who = f"carry_over: {row.get('standup_id')}: producer_set recheck"
    out: List[str] = []
    prov = entry.get("carried_from") if isinstance(entry.get("carried_from"), Mapping) else {}
    ps = ((entry.get("clauses") or {}).get("producer_set") or {}) if isinstance(entry.get("clauses"), Mapping) else {}
    if not isinstance(ps, Mapping) or not isinstance(ps.get("files"), list):
        return [f"{who}: no producer_set record to re-derive against"]
    prev_W, this_W = str(prov.get("W")), str(plan.get("W"))
    wave = str(prov.get("wave_uuid"))
    aid = entry.get("attempt_id")
    root = str(entry.get("attempt_root"))
    smoke_id = SMOKE_ID_OF_SOURCE.get(str(prov.get("source")))
    if ps.get("prev_sha") != prev_W or ps.get("this_sha") != this_W:
        out.append(f"{who}: the record's shas ({str(ps.get('prev_sha'))[:12]} -> {str(ps.get('this_sha'))[:12]}) are not the provenance's / this plan's ({prev_W[:12]} -> {this_W[:12]})")
    # ---- (1) P1-a: the PRIOR ACCEPTED RECORD, resolved from the consuming row and the (wave) the entry names -- the producer's own write
    try:
        dict_name = lp.dict_name(prev_W[:7], wave, "accepted")
        key = lp.accepted_key(prev_W[:7], wave, str(row["leg"]), str(row["engine"]), row["stage"], row["shard"])
    except (KeyError, ValueError, TypeError) as e:
        return out + [f"{who}: the consuming row cannot spell the prior accepted key ({e})"]
    try:
        getter = (accepted_get_for_dict or default_accepted_get_for_dict())
        live_aid = getter(dict_name)(key)
    except Exception as e:  # noqa: BLE001 -- no record, no Dict, no reader: each refuses by name
        return out + [f"{who}: the prior accepted Dict {dict_name} names no acceptance under {key} ({type(e).__name__}: {str(e)[:120]}) -- the provenance "
                      f"(W {prev_W[:12]}, wave {wave[:8]}) is not one the producer accepted under; refusing"]
    if not isinstance(live_aid, str) or not live_aid:
        return out + [f"{who}: the prior accepted Dict {dict_name} names no attempt under {key} -- the provenance (W {prev_W[:12]}, wave {wave[:8]}) is not one the producer accepted under"]
    if live_aid != aid or prov.get("attempt_id") != aid:
        return out + [f"{who}: the prior accepted record names attempt {live_aid!r} under {key}; the entry names {aid!r}, the provenance {prov.get('attempt_id')!r} -- refusing"]
    # ---- (2) the CANONICAL root, derived -- never the entry's
    expected_results_root = f"{lp.LANE}/{prev_W[:7]}/exp/{lp.RESULTS_ROOT_NAME}" + (f"{lp.SMOKE_RESULTS_SUFFIX}/{smoke_id}" if smoke_id else "")
    try:
        expected_root = lp.attempt_root_for(expected_results_root, leg=str(row["leg"]), view=str(row["engine"]), stage=row["stage"], shard=row["shard"], attempt_id=str(aid), kind=row.get("kind"))
    except (KeyError, ValueError, TypeError) as e:
        return out + [f"{who}: the consuming row's axes cannot spell an attempt root ({e})"]
    if root != expected_root:
        return out + [f"{who}: the attempt root {root} is not the canonical root the accepted record and the consuming row spell ({expected_root}: lane / W7 {prev_W[:7]} / "
                      f"{'smoke ' + smoke_id if smoke_id else 'production'} results / {row.get('leg')} / {row.get('engine')} / shard {row.get('shard')} / {aid})"]
    # ---- (3) P1-b: the evidence and the completion sources AT the canonical root, bound to the provenance and held by the inventory's digests
    inv = entry.get("evidence_inventory") if isinstance(entry.get("evidence_inventory"), Mapping) else {}
    ev_path = f"{root}/modal_evidence.json"
    text = read_text(ev_path)
    try:
        ev = json.loads(text) if text is not None else None
    except ValueError:
        ev = None
    if not isinstance(ev, Mapping):
        return out + [f"{who}: {ev_path} is absent or not an object at the canonical root -- the provenance cannot be bound, the producer set cannot be re-derived"]
    binds = {"W": (ev.get("W"), prev_W), "wave_uuid": (ev.get("wave_uuid"), wave), "attempt_id": (ev.get("attempt_id"), aid), "standup_id": (ev.get("standup_id"), prov.get("standup_id"))}
    bad = [f"{k}: evidence {str(a)[:20]!r} != provenance {str(b)[:20]!r}" for k, (a, b) in binds.items() if a != b]
    if bad:
        return out + [f"{who}: the attempt's own evidence ({ev_path}) does not bind the provenance -- {'; '.join(bad)}; a redirected provenance refuses by name"]
    for name in ("modal_evidence.json", "RESULT", "COMPLETE.json"):
        pth = f"{root}/{name}"
        t = text if name == "modal_evidence.json" else read_text(pth)
        digest = inv.get(pth)
        if t is None:
            out.append(f"{who}: {pth} is absent at the canonical root -- a completion source the certificate cannot read")
        elif not isinstance(digest, str) or not digest:
            out.append(f"{who}: {pth} is not bound in the evidence inventory (no digest) -- an inventory that does not list the canonical root's {name} binds nothing")
        elif _sha_text(t) != digest:
            out.append(f"{who}: {pth}: the bytes at the canonical root (sha {_sha_text(t)[:12]}) are not the inventory's ({digest[:12]})")
    # ---- (4) the archive the provenance names, held to the accepted-record-bound tree; the archived row and its identity re-run
    try:
        prior = (load_prior or load_prior_source)(Path(str(prov.get("rs"))))
    except Exception as e:  # noqa: BLE001 -- the archive is an input; unreadable refuses by name
        return out + [f"{who}: the prior archive {prov.get('rs')!r} cannot be loaded ({type(e).__name__}: {e}) -- the producer set cannot be re-derived"]
    if prior.get("W") != prev_W or str(prior.get("W7")) != prev_W[:7]:
        return out + [f"{who}: the prior archive {prov.get('rs')!r} is tree {str(prior.get('W'))[:12]}, the accepted record and the evidence name {prev_W[:12]}"]
    prev_row = next((r for r in _production_rows(prior["plan"]) if r.get("standup_id") == row.get("standup_id")), None)
    if prev_row is None:
        return out + [f"{who}: the prior plan has no row {row.get('standup_id')!r}"]
    ident_root = str(prior["plan"].get("results_root"))
    if smoke_id is not None:
        try:
            sident = lp.smoke_identity(prior["plan"], smoke_id, wave)
            prev_row = lp.smoke_row(prev_row, sident, max_children=len(prev_row.get("children") or []), max_timeout_s=int(prev_row["modal"]["timeout_s"]))
            ident_root = str(sident["results_root"])
        except (ValueError, KeyError) as e:
            return out + [f"{who}: the archived {smoke_id} row cannot be reconstructed ({e})"]
    if prev_row.get("standup_id") != prov.get("standup_id"):
        out.append(f"{who}: the re-derived archived row is {prev_row.get('standup_id')!r}, the provenance names {prov.get('standup_id')!r}")
    exact_root = lp.attempt_root_for(ident_root, leg=str(prev_row["leg"]), view=str(prev_row["engine"]), stage=prev_row["stage"], shard=prev_row["shard"], attempt_id=str(aid), kind=prev_row.get("kind"))
    if exact_root != root:
        out.append(f"{who}: the attempt root {root} is not the archived identity's {exact_root}")
    if (prev_row.get("ladder") or {}).get("cap_next") is not None:
        pth = f"{root}/{LADDER_ENQUEUED_REL}"
        t = read_text(pth); digest = inv.get(pth)
        if t is None or not isinstance(digest, str) or _sha_text(t) != digest:
            out.append(f"{who}: {pth}: a producer row's ladder inventory must be at the canonical root and bound in the evidence inventory by digest")
    prev_ids = {"W": prior["W"], "W7": prior["W7"], "wave_uuid": prior["wave_uuid"]}
    this_ids = {"W": plan.get("W"), "W7": plan.get("W7"), "wave_uuid": plan.get("wave_uuid")}
    ri = row_identity_verdict(prev_row, row, prev=prev_ids, this=this_ids, prev_smoke_wave=(wave if smoke_id else None))
    rec_ri = ((entry.get("clauses") or {}).get("row_identity") or {}) if isinstance(entry.get("clauses"), Mapping) else {}
    if not ri.get("ok"):
        out.append(f"{who}: the archived row's identity against the consuming row does NOT re-derive ok: {ri.get('why')}")
    elif ri.get("clause") != rec_ri.get("clause") or bool(ri.get("full_smoke_row")) != bool(rec_ri.get("full_smoke_row")):
        out.append(f"{who}: the re-derived row identity ({ri.get('clause')!r}, full_smoke_row {bool(ri.get('full_smoke_row'))}) is not the recorded clause ({rec_ri.get('clause')!r})")
    # ---- (5) the producer set, re-derived exactly as the evaluator derives it; blobs; the three sets; every actual differing file's admission
    pset = producer_set(prev_row, ev, W=prev_W)
    imports = direct_imports(git, prev_W, pset["all"])
    gating = sorted((set(pset["all"]) | {p for ps_ in imports.values() for p in ps_}) - set(CONTAINER_MODULES) - set(NON_EVAL_MODULES))
    files = [f for f in ps["files"] if isinstance(f, Mapping)]
    recorded = sorted(str(f.get("path")) for f in files)
    if recorded != gating:
        return out + [f"{who}: the producer set re-derived at certify {gating[:6]}{' ...' if len(gating) > 6 else ''} is not the record's {recorded[:6]}{' ...' if len(recorded) > 6 else ''}"]
    try:
        prev_b, this_b = _blobs_at(git, prev_W, gating), _blobs_at(git, this_W, gating)
    except ValueError as e:                                                     # a sha the repository does not hold: refuse by name, never raise
        return out + [f"{who}: git objects are not available ({e}) -- the differences cannot be re-derived; refusing"]
    missing = [f for f in gating if prev_b[f] is None or this_b[f] is None]
    if missing:
        return out + [f"{who}: git objects for {missing[:4]} are not available at {prev_W[:12]} / {this_W[:12]} -- the differences cannot be re-derived; refusing"]
    for f in files:
        pth = str(f.get("path"))
        if f.get("prev_blob") != prev_b[pth] or f.get("this_blob") != this_b[pth]:
            out.append(f"{who}: {pth}: recorded blobs {str(f.get('prev_blob'))[:12]} -> {str(f.get('this_blob'))[:12]} are not the trees' {str(prev_b[pth])[:12]} -> {str(this_b[pth])[:12]}")
    differing = sorted(f for f in gating if prev_b[f] != this_b[f])
    marked = sorted(str(f.get("path")) for f in files if f.get("admitted_difference"))
    not_identical = sorted(str(f.get("path")) for f in files if f.get("identical") is not True)
    adms = [a for a in (ps.get("admitted_differences") or []) if isinstance(a, Mapping)]
    record_paths = [str(a.get("path")) for a in adms]
    dupes = sorted({p_ for p_ in record_paths if record_paths.count(p_) > 1})
    if dupes:
        out.append(f"{who}: more than one admission record for {dupes[:4]} -- exactly one per differing file")
    records = sorted(set(record_paths))
    if records != differing:
        out.append(f"{who}: admission RECORDS {records[:6]} are not the files that actually differ between {prev_W[:12]} and {this_W[:12]} {differing[:6]} "
                   f"(missing record: {[f for f in differing if f not in records][:4]}; record for a non-differing file: {[f for f in records if f not in differing][:4]})")
    if marked != differing:
        out.append(f"{who}: files MARKED admitted {marked[:6]} are not the files that actually differ {differing[:6]} "
                   f"(differing but unmarked: {[f for f in differing if f not in marked][:4]}; marked but not differing: {[f for f in marked if f not in differing][:4]})")
    if marked != records:
        out.append(f"{who}: files MARKED admitted {marked[:6]} are not the admission RECORDS' paths {records[:6]} -- a mark without a record (or a record without a mark) certifies nothing")
    if not_identical != differing:
        out.append(f"{who}: files recorded not identical {not_identical[:6]} are not the files that actually differ {differing[:6]}")
    for adm in adms:
        if adm.get("hunks") is None and adm.get("kind") != ADMITTED_KIND_CHAIN:
            out.append(f"{who}: {adm.get('path')}: an admission of kind {adm.get('kind')!r} carries no hunks -- only the chain-list kind ({ADMITTED_KIND_CHAIN!r}) is validated by lines; refusing")
    by_path = {str(a.get("path")): a for a in adms}
    real_adms: List[Dict[str, Any]] = []
    for path in differing:
        adm = by_path.get(path)
        if adm is None:
            continue                                                            # refused above by name (no record)
        kind = adm.get("kind")
        if adm.get("hunks") is None and kind != ADMITTED_KIND_CHAIN:
            continue                                                            # refused above by name
        if adm.get("prev_blob") != prev_b[path] or adm.get("this_blob") != this_b[path]:
            out.append(f"{who}: {path}: the admission's blobs {str(adm.get('prev_blob'))[:12]} -> {str(adm.get('this_blob'))[:12]} are not the trees' "
                       f"{str(prev_b[path])[:12]} -> {str(this_b[path])[:12]} (a stale admission)")
        adm2 = admitted_difference(git, prev_W, this_W, path, str(prev_row.get("leg")))
        if adm2 is None:
            out.append(f"{who}: {path}: re-derived over the trees, the difference is NOT an admitted one under leg {prev_row.get('leg')!r} (the record claims {kind!r} {adm.get('tables')})")
            continue
        real_adms.append(adm2)
        if adm2.get("kind") != kind or sorted(adm2.get("tables") or []) != sorted(adm.get("tables") or []):
            out.append(f"{who}: {path}: the admission's kind / tables ({kind!r} {adm.get('tables')}) are not the re-derived ({adm2.get('kind')!r} {adm2.get('tables')})")
            continue
        if adm.get("hunks") is not None:
            rec_h = sorted(str(h.get("sha256")) for h in adm["hunks"] if isinstance(h, Mapping))
            real_h = sorted(h["sha256"] for h in adm2.get("hunks") or [])
            if rec_h != real_h:
                out.append(f"{who}: {path}: the recorded hunks ({len(rec_h)}) are not the real zero-context hunks ({len(real_h)}) of {prev_W[:12]} -> {this_W[:12]}")
            pin = pinned_hunks(path, str(row.get("leg")))
            unpinned = [h["sha256"] for h in adm2.get("hunks") or [] if pin.get(h["sha256"], (None,))[0] not in (adm2.get("tables") or [])]
            if unpinned:
                out.append(f"{who}: {path}: {len(unpinned)} real hunk(s) are not pinned for the consuming row's leg {row.get('leg')!r}")
        else:
            if sorted(adm.get("removed") or []) != sorted(adm2.get("removed") or []) or sorted(adm.get("added") or []) != sorted(adm2.get("added") or []):
                out.append(f"{who}: {path}: the recorded chain-list lines are not the re-derived ones")
    # ---- (6) P2: the Petri sidecar proof -- REQUIRED by the archived row's leg / mode and the RECOMPUTED differences, re-derived from its inputs
    if petri_proof_required(prev_row, real_adms or [{"path": f, "kind": (by_path.get(f) or {}).get("kind")} for f in differing]):
        proof = ps.get("petri_sidecar_proof")
        if not isinstance(proof, Mapping) or proof.get("ok") is not True:
            return out + [f"{who}: a Petri-mode row ({prev_row.get('leg')}, mode {prev_row.get('mode')!r}) with an admitted chain / petri_runner difference needs the covering "
                          f"sidecar proof; the record carries {'none' if not isinstance(proof, Mapping) else 'a FAILING one (' + str(proof.get('why'))[:120] + ')'} -- refusing"]
        if proof.get("kind") == GENERATION_PROOF_KIND:
            # The generation-only proof, RE-DERIVED from the volume bytes through the certificate's reader and held to the RECORDED
            # verdict field by field; the scope guard first (a petri-LEG row never carries one); every input under the canonical root
            gp = f"{who}: {GENERATION_PROOF_KIND} proof ({GENERATION_PROOF_RULING})"
            scope = generation_proof_scope_problem(prev_row, source=str(prov.get("source")), evidence=ev)
            if scope is not None:
                return out + [f"{gp}: out of scope at certify -- {scope}; refusing"]
            if proof.get("ruling") != GENERATION_PROOF_RULING:
                return out + [f"{gp}: the recorded proof's ruling label is {proof.get('ruling')!r}, not {GENERATION_PROOF_RULING!r}; refusing"]
            missing_keys = [k for k in GENERATION_PROOF_ENTRY_KEYS if k not in proof]
            if missing_keys:
                return out + [f"{gp}: the recorded proof lacks {missing_keys}; refusing"]
            if (proof.get("judge") or {}).get("in_attempt") is not False or (proof.get("judge") or {}).get("site") != GENERATION_PROOF_JUDGE_SITE:
                out.append(f"{gp}: the recorded judge site is not {GENERATION_PROOF_JUDGE_SITE!r} / in_attempt false")
            mgp = _generation_proof_module()
            rec_doc = {**{k: proof.get(k) for k in mgp.VERDICT_KEYS if k in proof}, "schema": mgp.SCHEMA, "kind": GENERATION_PROOF_KIND, "ruling": GENERATION_PROOF_RULING}
            rederived, gprobs = generation_proof_record(rec_doc, prev_row=prev_row, source=str(prov.get("source")), root=root, evidence=ev, read_text=read_text, git=git, prev_W=prev_W, this_W=this_W)
            if rederived is None:
                return out + [f"{gp}: {p_}" for p_ in gprobs] + [f"{gp}: the volume bytes at the canonical root do not reproduce the recorded proof; refusing"]
            if rederived.get("bytes_read") != proof.get("bytes_read"):
                out.append(f"{gp}: re-read {rederived.get('bytes_read')} bytes at certify, the record says {proof.get('bytes_read')}")
            return out
        local_inv = entry.get("evidence_inventory_local")
        if not isinstance(local_inv, Mapping) or not local_inv:
            return out + [f"{who}: the sidecar proof's inputs are not bound (no evidence_inventory_local) -- the proof cannot be re-derived; refusing"]
        reader = RecordingReader(local_read_text or default_local_read_text)
        try:
            proof2 = (petri_proof or (lambda **kw: default_petri_proof(git=git, home=petri_home, **kw)))(prev_row=prev_row, prior=prior, aid=str(aid), evidence=ev, read_text=reader)
        except Exception as e:  # noqa: BLE001 -- a proof that cannot be re-derived refuses by name
            return out + [f"{who}: the sidecar proof cannot be re-derived from the attempt's evidence ({type(e).__name__}: {str(e)[:160]}) -- refusing"]
        uses = ((proof2.get("removed_leg") or {}).get("uses")) if isinstance(proof2, Mapping) else None
        if not isinstance(proof2, Mapping) or proof2.get("ok") is not True or (isinstance(uses, int) and uses != 0):
            out.append(f"{who}: the sidecar proof re-derived at certify does NOT hold (ok {proof2.get('ok') if isinstance(proof2, Mapping) else None!r}, removed-leg uses {uses!r}): "
                       f"{str((proof2 or {}).get('why'))[:160] if isinstance(proof2, Mapping) else proof2!r}")
        seen = dict(sorted(reader.seen.items()))
        if set(seen) != set(local_inv) or any(seen[k_] != local_inv[k_] for k_ in seen):
            out.append(f"{who}: the sidecar proof's re-derived inputs ({len(seen)}) are not the recorded evidence_inventory_local ({len(local_inv)}) by path and digest")
    return out


def carry_over_problems(record: Any, *, plan: Mapping[str, Any], read_text: Optional[Callable[[str], Optional[str]]] = None,
                        accepted_get: Optional[Callable[[str], Any]] = None,
                        local_read_text: Optional[Callable[[str], Optional[str]]] = None,
                        git: Optional[Callable[[Sequence[str]], Tuple[int, str]]] = None,
                        accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None,
                        petri_proof: Optional[Callable[..., Dict[str, Any]]] = None, petri_home: Optional[Path] = None) -> List[str]:
    """Why ``record`` is not THIS plan's carry-over record (empty = it is; ``None`` = nothing carried, no problem): the schema, the binding (W /
    W7 / wave / launch_plan_sha), every carried entry a plan row under its accepted key with a full provenance (:data:`PROVENANCE_KEYS`) and all
    five clauses recorded `ok`; with ``read_text`` the root must STILL read COMPLETE (RESULT DONE rc 0 naming the attempt, COMPLETE.json and
    modal_evidence.json naming it); with ``accepted_get`` (THIS wave's LIVE accepted Dict) the key must name the carried attempt NOW.
    ``local_read_text`` (default the grading host's filesystem) re-reads every LOCAL Petri-proof input the entry bound
    (`evidence_inventory_local`): a source recorded present must still hold its digest, one recorded ABSENT must still be absent (a refusal
    ledger that appeared after the evaluation is new evidence the proof never saw); a Petri entry carrying a sidecar proof without a bound
    local inventory refuses. A `launch_history_unknown` carry hold is a named problem per held
    row -- no certificate issues while one stands."""
    if record is None:
        return []
    if not isinstance(record, Mapping):
        return ["carry_over: not an object"]
    out: List[str] = []
    if record.get("schema") != CARRY_OVER_SCHEMA:
        out.append(f"carry_over: schema {record.get('schema')!r} != {CARRY_OVER_SCHEMA}")
    for k in ("W", "W7", "wave_uuid"):
        if record.get(k) != plan.get(k):
            out.append(f"carry_over: {k} {record.get(k)!r} is not this plan's {plan.get(k)!r}")
    if record.get("launch_plan_sha") != plan.get("launch_plan_sha"):
        out.append(f"carry_over: launch_plan_sha {record.get('launch_plan_sha')!r} is not this plan's {plan.get('launch_plan_sha')!r} -- evaluated against another plan")
    carried = record.get("carried")
    if not isinstance(carried, Mapping):
        out.append("carry_over: carried is not a mapping")
        return out
    rows = {r["standup_id"]: r for r in _production_rows(plan)}
    for sid, e in carried.items():
        who = f"carry_over: {sid}"
        row = rows.get(sid)
        if row is None:
            out.append(f"{who}: not a production row of this plan")
            continue
        if not isinstance(e, Mapping):
            out.append(f"{who}: entry is not an object")
            continue
        aid = e.get("attempt_id")
        if e.get("accepted_key") != row["accepted_key"]:
            out.append(f"{who}: accepted_key {e.get('accepted_key')!r} is not the plan row's {row['accepted_key']}")
        if not isinstance(aid, str) or not aid:
            out.append(f"{who}: no attempt id")
            continue
        prov = e.get("carried_from")
        if not isinstance(prov, Mapping) or [k for k in PROVENANCE_KEYS if k not in prov]:
            out.append(f"{who}: provenance missing or incomplete (carried_from needs {list(PROVENANCE_KEYS)})")
        else:
            if prov.get("attempt_id") != aid:
                out.append(f"{who}: provenance names attempt {prov.get('attempt_id')!r}, the entry {aid!r}")
            if prov.get("W") == plan.get("W") and prov.get("source") != CURRENT_S5_SOURCE:
                out.append(f"{who}: provenance names THIS tree {str(plan.get('W'))[:12]} -- a carried attempt comes from a prior tree (or is this tree's S5 FULL row, named as such)")
            if prov.get("source") == CURRENT_S5_SOURCE and prov.get("W") != plan.get("W"):
                out.append(f"{who}: a {CURRENT_S5_SOURCE} source must be THIS tree's ({str(prov.get('W'))[:12]} != {str(plan.get('W'))[:12]})")
        clauses = e.get("clauses")
        if not isinstance(clauses, Mapping):
            out.append(f"{who}: no clause record")
        else:
            for c in CLAUSES:
                v = clauses.get(c)
                if not isinstance(v, Mapping) or v.get("ok") is not True:
                    out.append(f"{who}: clause {c!r} missing or not ok")
            ri = clauses.get("row_identity") if isinstance(clauses.get("row_identity"), Mapping) else {}
            if (ri.get("s5_full_row") or ri.get("full_smoke_row")) and ri.get("clause") not in (S5_FULL_CLAUSE, S3_FULL_CLAUSE):
                out.append(f"{who}: a smoke source must be admitted under a named clause ({S5_FULL_CLAUSE!r} / {S3_FULL_CLAUSE!r})")
            if ri.get("full_smoke_row") and isinstance(prov, Mapping) and SMOKE_ID_OF_SOURCE.get(str(prov.get("source"))) != ri.get("smoke_id"):
                out.append(f"{who}: the provenance source {prov.get('source')!r} is not the smoke {ri.get('smoke_id')!r} the row identity admitted")
            ps = clauses.get("producer_set") if isinstance(clauses.get("producer_set"), Mapping) else {}
            if isinstance(ps, Mapping) and ps.get("ok") is True:
                if ps.get("this_sha") != plan.get("W") or not ps.get("files") or any(not f.get("identical") and not f.get("admitted_difference") for f in ps.get("files") or []):
                    out.append(f"{who}: the producer-set record is not a byte-identical set at this tree ({str(ps.get('this_sha'))[:12]}, {len(ps.get('files') or [])} file(s))")
                if any(f.get("admitted_difference") for f in ps.get("files") or []) and not ps.get("admitted_differences"):
                    out.append(f"{who}: a producer file is marked as an admitted difference but the record carries no admitted_differences block")
                # fj24: every recorded pinned-hunk admission must be THIS tree's -- its tables known, every hunk pinned for the path under the row's
                # leg and labelled with the table that pins it (a record claiming an unpinned hunk, or another leg's table, refuses at certify)
                known = {name for name, _t, _r, _k, _l in ADMITTED_HUNK_TABLES}
                for adm in ps.get("admitted_differences") or []:
                    if not isinstance(adm, Mapping) or not isinstance(adm.get("path"), str):
                        out.append(f"{who}: an admitted difference record names no path")
                        continue
                    if adm.get("hunks") is None:
                        # ONLY the chain-list kind is validated by lines -- any other kind without hunks refuses by name
                        if adm.get("kind") != ADMITTED_KIND_CHAIN or not isinstance(adm.get("changed_lines"), list):
                            out.append(f"{who}: admitted difference {adm['path']}: an admission of kind {adm.get('kind')!r} carries no hunks -- only the chain-list kind "
                                       f"({ADMITTED_KIND_CHAIN!r}, with its changed_lines) is validated by lines; refusing")
                        continue
                    tables = adm.get("tables") or []
                    if not tables or any(t not in known for t in tables):
                        out.append(f"{who}: admitted difference {adm['path']} names unknown table(s) {tables}")
                        continue
                    pin = pinned_hunks(adm["path"], str(row.get("leg")))
                    bad = [h for h in adm["hunks"] if not isinstance(h, Mapping) or pin.get(h.get("sha256"), (None, None))[0] != h.get("table")]
                    if bad or not adm["hunks"]:
                        out.append(f"{who}: admitted difference {adm['path']}: {len(bad)} hunk(s) are not pinned for leg {row.get('leg')!r} by the table they claim "
                                   f"({[str(h.get('sha256'))[:12] for h in bad[:3]]})")
            cc = clauses.get("complete") if isinstance(clauses.get("complete"), Mapping) else {}
            det = ((cc.get("checks") or {}).get("determinism") or {}) if isinstance(cc, Mapping) else {}
            det_absent = [p for p in (e.get("evidence_inventory") or {}) if (e.get("evidence_inventory") or {}).get(p) is None
                          and any(p.endswith("/" + rel) for _leg, rel in lp.determinism_rels(row))] if isinstance(e.get("evidence_inventory"), Mapping) else []
            c3 = det.get("criterion3") if isinstance(det, Mapping) else None
            if c3 is not None or e.get("criterion3") is not None:
                # fj22: the qualification must be the ruling's shape, the entry's copy the check's, and the decode it rests on sampled
                dec = (c3.get("decode") or {}) if isinstance(c3, Mapping) and isinstance(c3.get("decode"), Mapping) else {}
                temps, srcs = dec.get("temperatures"), dec.get("source")
                good_src = isinstance(srcs, list) and srcs and all(isinstance(x, Mapping) and isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str)
                                                                     and x.get("authority") in (DECODE_AUTHORITY_REAUDIT, DECODE_AUTHORITY_HISTORICAL) for x in srcs)
                if not isinstance(c3, Mapping) or c3.get("verdict") != CRITERION3_NOT_APPLICABLE_SAMPLED or c3.get("ruling") != CRITERION3_RULING \
                        or not isinstance(c3.get("reproduced"), int) or not isinstance(c3.get("resent"), int) or not (0 <= c3["reproduced"] < c3["resent"]) \
                        or not isinstance(temps, list) or not temps or any(not isinstance(t, (int, float)) or isinstance(t, bool) or t <= 0 for t in temps) \
                        or not good_src or dec.get("authority") not in (DECODE_AUTHORITY_REAUDIT, DECODE_AUTHORITY_HISTORICAL):
                    out.append(f"{who}: the criterion-3 qualification is not {CRITERION3_RULING}'s ({CRITERION3_NOT_APPLICABLE_SAMPLED}: reproduced < resent on a sampled decode, "
                               "temperatures > 0, read from a verified resolved-sampler source (path, sha256, authority))")
                elif e.get("criterion3") != c3:
                    out.append(f"{who}: the entry's criterion3 qualification is not the complete clause's")
                elif read_text is not None:
                    # fj23: the decode source(s) the exemption rests on still read the bytes the check verified (the entry's inventory holds them too)
                    for x in srcs:
                        t = read_text(x["path"])
                        if t is None or _sha_text(t) != x["sha256"]:
                            out.append(f"{who}: the criterion-3 qualification's decode source {x['path']} no longer reads the verified bytes")
                if isinstance(det, Mapping) and det.get("waived"):
                    out.append(f"{who}: a determinism record cannot be both waived (absent) and qualified (present, diverged)")
            if (isinstance(det, Mapping) and det.get("waived")) or det_absent:
                # The determinism record ABSENT carries ONLY under a covering waiver, present verbatim on the entry
                w = e.get("criterion3_waiver")
                probs = criterion3_waiver_problems(w, sid)
                out += [f"{who}: {p}" for p in probs]
                if not probs:
                    prov_W = str((prov or {}).get("W") or "") if isinstance(prov, Mapping) else ""
                    if not waiver_covers(w, W=prov_W, leg=str(row.get("leg")), view=str(row.get("engine")), legs=row_petri_legs(row)):
                        out.append(f"{who}: the criterion-3 waiver's scope {dict(w.get('scope') or {})} does not cover this row (W {prov_W[:12]}, {row.get('leg')}/{row.get('engine')}, legs {row_petri_legs(row)})")
                if isinstance(det, Mapping) and det.get("waived") and det.get("verdict") != DETERMINISM_WAIVED:
                    out.append(f"{who}: a waived determinism check must read {DETERMINISM_WAIVED!r}")
                if det_absent and not (isinstance(det, Mapping) and det.get("waived")):
                    out.append(f"{who}: determinism record(s) {det_absent} absent at evaluation but the complete clause does not record the waiver")
            lq = clauses.get("ladder_queue") if isinstance(clauses.get("ladder_queue"), Mapping) else {}
            if isinstance(lq, Mapping) and lq.get("producer_row") and lq.get("consumers"):
                rp = lq.get("replay")
                if not isinstance(rp, Mapping) or rp.get("ok") is not True or rp.get("closed") is not True or rp.get("schema") != REPLAY_SCHEMA:
                    out.append(f"{who}: a producer row consumed in this plan carries no complete queue replay record")
                elif read_text is not None:
                    tk = rp.get("target_keys") or {}
                    if read_text(f"{lp.VOL_PATH}/{tk.get('marker')}") is None:
                        out.append(f"{who}: the replayed producer's CLOSED marker {tk.get('marker')!r} is not on the volume")
                    if rp.get("queue_closed") and read_text(f"{lp.VOL_PATH}/{rp.get('closed_json')}") is None:
                        out.append(f"{who}: the replay recorded the queue CLOSED but {rp.get('closed_json')!r} is not on the volume")
        root = e.get("attempt_root")
        if read_text is not None and isinstance(e.get("clauses"), Mapping):
            # fj25: clause (b) re-derived from the trusted git objects and the archived plan row -- never trusted from the record
            out += producer_set_recheck(e, row=row, plan=plan, git=(git or default_git(REPO_ROOT)), read_text=read_text, accepted_get_for_dict=accepted_get_for_dict,
                                        local_read_text=local_read_text, petri_proof=petri_proof, petri_home=petri_home)
        if not isinstance(root, str) or not root.startswith(lp.VOL_PATH + "/"):
            out.append(f"{who}: attempt_root {root!r} is not a volume path")
        elif read_text is not None:
            # EVERY evidence source clauses (c) / (d) read is re-read and held to its recorded digest -- a source removed or changed after
            # the evaluation refuses here, on the bytes, not on the recorded flags
            inv = e.get("evidence_inventory")
            if not isinstance(inv, Mapping) or not inv:
                out.append(f"{who}: no evidence inventory -- the clauses cannot be re-validated on their sources")
            else:
                for path, digest in sorted(inv.items()):
                    if digest is None:
                        continue                                                    # recorded ABSENT at evaluation (a waived source): absence is its state
                    text = read_text(path)
                    if text is None:
                        out.append(f"{who}: evidence source {path} is no longer on the volume")
                    elif _sha_text(text) != digest:
                        out.append(f"{who}: evidence source {path} changed since the evaluation (sha {_sha_text(text)[:12]} != {str(digest)[:12]})")
            result = read_text(f"{root}/RESULT")
            m = lp.RESULT_DONE_RE.search(result or "")
            if not m or m.group("aid") != aid:
                out.append(f"{who}: the root no longer reads COMPLETE (RESULT {(result or '').strip()[:120]!r} is not `status=DONE rc=0 attempt={aid}`)")
            for name in ("COMPLETE.json", "modal_evidence.json"):
                text = read_text(f"{root}/{name}")
                try:
                    doc = json.loads(text) if text is not None else None
                except ValueError:
                    doc = None
                if not isinstance(doc, Mapping) or doc.get("attempt_id") != aid:
                    out.append(f"{who}: {name} under the carried root no longer names attempt {aid!r}")
        if accepted_get is not None:
            live = accepted_get(row["accepted_key"])
            if live != aid:
                out.append(f"{who}: this wave's accepted Dict names {live!r} under {row['accepted_key']}, the record {aid!r} -- the carry was not seeded or was superseded")
        # fj7 (restored inside the carried loop by fj16): a re-audited entry's verdicts are THIS tree's readers' work -- the record
        # must say so and point at a volume evidence root; checked for EVERY carried entry, independent of read_text
        if e.get("reaudited_by") == REAUDIT_NOT_REQUIRED_SAME_TREE:
            # fj17: admitted only for a same-tree source (its verdicts ARE this tree's readers' work)
            if not (isinstance(prov, Mapping) and prov.get("source") == CURRENT_S5_SOURCE and prov.get("W") == plan.get("W")) or e.get("reaudit") is not None:
                out.append(f"{who}: reaudited_by {REAUDIT_NOT_REQUIRED_SAME_TREE!r} is admitted only for this tree's own S5 FULL source, with no re-audit record")
        elif e.get("reaudit") is not None or e.get("reaudited_by") is not None:
            out += [f"{who}: {p}" for p in reaudit_problems(e.get("reaudit"), this_sha=str(plan.get("W")))]
            if e.get("reaudited_by") != plan.get("W"):
                out.append(f"{who}: reaudited_by {str(e.get('reaudited_by'))[:12]!r} is not this tree {str(plan.get('W'))[:12]}")
        # The Petri sidecar proof's LOCAL inputs, re-read on the bytes -- presence, absence and digest all held to the evaluation's
        ps_ = clauses.get("producer_set") if isinstance(clauses, Mapping) and isinstance(clauses.get("producer_set"), Mapping) else {}
        has_proof = isinstance(ps_, Mapping) and isinstance(ps_.get("petri_sidecar_proof"), Mapping)
        local_inv = e.get("evidence_inventory_local")
        gen_proof = has_proof and ps_["petri_sidecar_proof"].get("kind") == GENERATION_PROOF_KIND     # Its inputs are VOLUME paths, bound in evidence_inventory
        if has_proof and not gen_proof and not (isinstance(local_inv, Mapping) and local_inv):
            out.append(f"{who}: the Petri sidecar proof's local inputs are not bound (no evidence_inventory_local) -- the chain exception cannot be re-validated")
        if has_proof and ps_["petri_sidecar_proof"].get("ok") is not True:
            # fj28: a recorded proof that FAILED is a refusal wherever it appears (the requirement itself is derived in producer_set_recheck)
            out.append(f"{who}: the recorded Petri sidecar proof FAILED ({str(ps_['petri_sidecar_proof'].get('why'))[:160]}) -- never a carry")
        if isinstance(local_inv, Mapping) and local_inv:
            lread = local_read_text or default_local_read_text
            for path, digest in sorted(local_inv.items()):
                text = lread(path)
                if digest is None and text is not None:
                    out.append(f"{who}: local source {path} was ABSENT at the evaluation and exists now -- evidence the sidecar proof never read")
                elif digest is not None and text is None:
                    out.append(f"{who}: local source {path} is no longer present")
                elif digest is not None and _sha_text(text) != digest:
                    out.append(f"{who}: local source {path} changed since the evaluation (sha {_sha_text(text)[:12]} != {str(digest)[:12]})")
    # fj15: a REFUSED row whose hold was RELEASED (an acceptance classified `limited`) rests on shape evidence -- re-read on the bytes;
    # evidence that changed or disappeared since the evaluation voids the release (the row would spawn on stale grounds): refuse, re-evaluate
    if read_text is not None:
        for sid, e in (record.get("refused") or {}).items():
            if not isinstance(e, Mapping):
                continue
            for a in e.get("attempts") or []:
                paa = a.get("prior_accepted_attempt") if isinstance(a, Mapping) else None
                if not isinstance(paa, Mapping) or paa.get("holds") is not False:
                    continue
                who = f"carry_over: refused {sid}"
                inv = paa.get("evidence_inventory")
                if not isinstance(inv, Mapping) or not inv:
                    out.append(f"{who}: the released hold ({paa.get('attempt_id')}) binds no shape evidence -- nothing to re-validate; the release is void")
                    continue
                for path, digest in sorted(inv.items()):
                    text = read_text(path)
                    if digest is None and text is not None:
                        out.append(f"{who}: shape evidence {path} was ABSENT at the evaluation and exists now -- the released hold rests on stale grounds")
                    elif digest is not None and text is None:
                        out.append(f"{who}: shape evidence {path} is no longer on the volume -- the released hold rests on stale grounds")
                    elif digest is not None and _sha_text(text) != digest:
                        out.append(f"{who}: shape evidence {path} changed since the evaluation (sha {_sha_text(text)[:12]} != {str(digest)[:12]}) -- the released hold is void")
    # fj21: a hold of kind launch_history_unknown blocks `select_launch_rows`, but a retry pass / --reattempt spawns without reading holds --
    # so CERTIFICATION refuses while any such hold stands (a certificate cannot issue; --go's re-validation refuses), until the prior's launch record
    # is written and the carry re-evaluated. Metadata-level: runs with or without read_text. Other hold kinds (an accepted attempt refused) are the
    # launcher's belt as before; vanished-Dict refusals already stop the evaluation.
    for sid, h in sorted(carry_refused_holds(record).items()):
        if h.get("hold_kind") != DICT_HISTORY_UNKNOWN:
            continue
        e = (record.get("refused") or {}).get(sid) or {}
        priors_unknown = sorted({str(a.get("prior")) for a in (e.get("attempts") or []) if isinstance(a, Mapping) and a.get("launch_history_unknown") is True})
        out.append(f"carry hold {DICT_HISTORY_UNKNOWN} on {sid}: write " + ", ".join(f"{pd}/{SMOKES_LAUNCHED_REL}" for pd in priors_unknown)
                   + f" ({SMOKES_LAUNCHED_SCHEMA}) and re-evaluate before certify")
    return out
