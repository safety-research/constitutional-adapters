#!/usr/bin/env python3
"""File the Petri re-judge's verdicts (the ``--judge`` profile of ``judges.py``, default GLM-5.3) into the registry as a NEW judge beside
the Opus 5 one (records -> inbox, never the DB).

    PYTHONPATH=src:scripts python scripts/rejudge/petri/file_records.py --scope SCOPE --enum ENUM --run-dir RUN [--run-dir ...] \
        --staging STAGING --registry-root ROOT [--runs RUNS.txt] [--allow-partial] [--dry-run]
    PYTHONPATH=src:scripts python scripts/rejudge/petri/file_records.py --staging STAGING --registry-root ROOT --deliver judge --deliver-only
    (a harvester cycle folds the judge)
    PYTHONPATH=src:scripts python scripts/rejudge/petri/file_records.py --staging STAGING --registry-root ROOT --run-dir RUN [--run-dir ...] \
        --deliver runs --deliver-only [--allow-partial]

``--scope`` is the FULL paper-row scope (``scope.py``'s jobs.jsonl and units.tsv.gz) and ``--enum`` the FULL enumeration (its
inputs.jsonl.gz maps each transcript to its judge-input key, its tools.json holds the six answer tools), so a run judged only in
part files a ``partial`` judgment. Stage and validate once (``--dry-run`` or no ``--deliver``), then deliver those same bytes judge first.

THE VERDICT LEDGERS. Every ``--run-dir`` is a verdict ledger of the SAME judge protocol: its ``track.json`` must be one of the
profile's ledger tracks (model, endpoint, parser), carry no ``id_suffix`` (a probe re-sends inputs another ledger holds, under other
settings) and name the profile's request settings (``judges.ledger_identity``) -- anything else is refused, so verdicts made at another
effort or under another layout never file under this judge id. ``max_tokens`` is not part of the identity (the Opus judge's identity
leaves its reply cap out too); each ledger's cap is recorded in the judgment. A key with a FILEABLE terminal outcome (a verdict or an
item outcome: moderation, judge_refusal, parse_error, truncated) in more than one ledger is refused (the smoke, the pilot and the full
run are built disjoint with ``build_manifests.py --exclude-keys``): there is never a choice between two verdicts, or between a verdict
and a rejection, of the same input. An INFRASTRUCTURE terminal outcome (provider_error, missing, batch_*: the provider failed, not the
item) is settled by a fileable outcome of the same key in another ledger -- a follow-up run directory that re-sent the keys a failed
batch stranded -- and between two infrastructure outcomes the later harvest is kept; neither choice depends on the ledgers' order.

THE DELIVERY GATE. Runs are delivered (``--deliver runs|all``, with or without ``--deliver-only``) only when every ``--run-dir`` has
STOPPED MOVING (``state.readiness``): not PAUSED, no batch pending / queued / in flight, and no request still waiting on a re-ask or a
person. A run that would file ``partial`` (a unit with no terminal outcome, or only an infrastructure one), or a ledger holding a
set-aside batch, is refused unless ``--allow-partial`` says that is intended (the pilot is partial by design). ``--deliver-only`` also
refuses when the ``--run-dir`` set is not the one staged or any ledger changed since the files were staged (restage instead). Pass
``--registry-root`` when STAGING (read only): a run already filed is then staged as a FOLLOW-UP whose file names start above the
registry's (``filing.Filer``), so a completed or corrected filing can be delivered later; the registry folds the later judgment and
item rows over the earlier ones of the same key and touches no other judge's rows. The judge record is not staged again when the
registry already holds it.

ONE judge row per protocol: ``petri_bloom_judge`` on the profile's model (``judge_backend`` batch). ``rubric_sha256`` = sha256 of the
JSON of the rubric material every request carries -- inspect_petri's ``JUDGE_PROMPT`` template and ``HIGHLIGHTS_DESCRIPTION`` and the six
answer tools as sent (their property order kept), so every filing of the same inputs names the same judge (it reads the installed
``inspect-petri``, refused unless it is the pinned 3.0.11); ``protocol_sha256`` = sha256 of the request protocol (endpoint, model slug,
the body template without the texts, and the provider the verdicts were served by). ``threshold_rule`` and ``score_scale`` are the
Opus judge's. A SECOND judge beside the Opus 5 one: nothing it files supersedes, withdraws or re-points an Opus row.

Per run, one judgment (plain-form id over (run, judge); ``complete`` once every unit of the run has a verdict or a filed item
outcome, else ``partial``; counts, the OpenRouter batch ids, the code sha, each ledger's settings) and, per UNIT (a registry item:
run, behaviour/scenario, trial, sample, turn) these item rows, every one keyed by the new judgment id and the Opus item's own key, so
``v_judge_side_by_side`` pairs the judges:
  petri_score       the behaviour's own dimension (1-10; ``extra.own_dimension`` names it: cyber is scored as
                    cyber_bio_chem_operational_uplift, as the Opus rows are)
  eval_awareness    1-10
  scenario_realism  1-10
  -- or, for a unit whose terminal outcome is an ITEM outcome other than a verdict (moderation, judge_refusal, parse_error, truncated):
  ONE ``petri_score`` row, value NULL, ``label`` = the outcome (the Opus judge files its own terminal refusals as ``petri_score`` NULL /
  ``judge_refused``), ``extra`` the error kind, status and message. A moderation rejection is therefore recorded per item, never dropped
  and never scored. An INFRASTRUCTURE outcome (provider_error, missing, batch_*: the provider failed, not the item) is NOT filed as a row
  by default -- the unit counts as unjudged (``infra_unfiled``, so the judgment is ``partial``) and a later filing can still give it a
  verdict; ``--file-infra-outcomes`` files them as well. Account and balance failures are never terminal, so they never reach a filing.
Every row's ``extra`` carries ``request_key`` (the judge-input key: the verdict, highlights, summary and justification are in the run
directory's ledger under it), ``batch_id``, ``attempt``, the transcript key and the Opus judgment it re-measures. A verdict whose own
dimension is not the request tool's (the exact-schema check the generic parse defers) files as ``parse_error``.
A run directory marked ``SYNTHETIC`` (the scratch proof's fake verdicts) is refused for any registry root but a scratch one
(``--allow-synthetic``).

``--judge glm-5.3`` files the GLM-5.3 verdicts: ``petri_bloom_judge`` on ``glm-5.3`` (``judge_backend`` batch), ``protocol_sha256`` over
the candidate's body template (``judges.arm_body`` without the texts and ``max_tokens``) and the provider the verdicts were served by
(``served_by``: DeepInfra), records under ``rejudge/petri_glm53__<judge_id[:12]>/``. Its ledgers are the full run's (``track``
``petri_glm53``) and the judge comparison's (``track`` ``petri_bakeoff``, ``arm`` ``glm-5.3``), both checked against the candidate's
request settings (``judges.ledger_identity``); the provider PIN differs between them (the comparison ran unpinned) and is recorded per
ledger, not in the identity. Every verdict and item outcome to be filed must have been served by DeepInfra, else the staging is
refused (``judges.served_problems``).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import filing as F  # noqa: E402
from rejudge.common import state as S  # noqa: E402
from rejudge.petri import judges as J  # noqa: E402
from rejudge.petri import request as R  # noqa: E402
from rejudge.petri.scope import PETRI_JUDGE  # noqa: E402

JUDGE_MODEL = J.PROFILES[J.DEFAULT]["judge_model"]
UNSCORED_LABELS = ("moderation", "judge_refusal", "parse_error", "truncated", "provider_error", "missing", "batch_failed",
                   "batch_expired", "batch_cancelled")
#: terminal outcomes that say something about the ITEM (filed as NULL petri_score rows); every other non-verdict terminal outcome is the
#: provider's or the batch's failure and is filed only with --file-infra-outcomes
ITEM_UNSCORED = ("moderation", "judge_refusal", "parse_error", "truncated")
#: terminal outcomes that say something about the input (a verdict or an item outcome): two ledgers holding one for the same key are refused
FILEABLE = ("verdict",) + ITEM_UNSCORED
OPUS_RUBRIC_VERSION = "petri-bloom 0.2.6 verbatim prompt bytes, override@c7fb238c0b17a812"
OPUS_THRESHOLD_RULE = "none: headline is the mean judge score (paperlib HEADLINE['petri'] petri_score, lower is better)"
OPUS_SCORE_SCALE = "1-10 misalignment score per (behaviour, scenario) transcript"
SYNTHETIC_MARK = "SYNTHETIC"
EVALS = ("petri_pooled", "petri_cyber_bio_chem_capabilities", "petri_self_preservation", "petri_delusion_sycophancy",
         "petri_defer_to_users", "petri_reasoning_unfaithfulness", "petri_instruction_evasion")


#: the judge template's package as the Opus judge ran it (pyproject pin); the rubric hash reads the template from the installed package,
#: so a different install would mint a different judge id -- refused instead
RUBRIC_PACKAGE = ("inspect-petri", "3.0.11")


def rubric_material(tools: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The rubric every request carries: the judge template and highlights text (inspect_petri, as installed -- refused unless it is the
    pinned version) and the six answer tools as sent, ordered by their own dimension (each tool's keys in their own order)."""
    from importlib.metadata import version
    have = version(RUBRIC_PACKAGE[0])
    if have != RUBRIC_PACKAGE[1]:
        raise SystemExit(f"{RUBRIC_PACKAGE[0]} {have} is installed; the Petri judge template is {RUBRIC_PACKAGE[0]} {RUBRIC_PACKAGE[1]}'s")
    import inspect_petri._judge.judge as J
    ordered = sorted((R.canonical_tool(t) for t in tools.values()), key=R.own_dimension)
    return {"judge_prompt_template": J.JUDGE_PROMPT, "highlights_description": J.HIGHLIGHTS_DESCRIPTION, "answer_tools": ordered,
            "prompt_provenance": "override@c7fb238c0b17a812"}


def rubric_sha256(tools: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha256(R._ordered_json(rubric_material(tools)).encode("ascii")).hexdigest()


def arm_protocol(prof: Mapping[str, Any]) -> Dict[str, Any]:
    """A profile's request protocol: the candidate's body template (no texts, no max_tokens) and the provider its verdicts were served
    by -- hashed into ``protocol_sha256`` (``judges.py``, IDENTITY)."""
    tool = {"name": R.TOOL_NAME, "description": "<the answer tool's description>", "input_schema": {"<the behaviour's AlignmentAnswer schema>": True}}
    body = J.arm_body(prof, R.BEGIN + "x" + R.END, tool)
    body.pop("max_tokens")
    body["messages"][0]["content"] = "<the judge prompt, one string>"
    return {"route": "openrouter_batch_api", "endpoint": prof["endpoint"], "model": prof["model"], "batch_model_variant": prof["model"] + ":batch",
            "body_template": body, "system_prompt": None, "temperature": None, "served_by": list(prof["served_by"])}


def protocol(judge: Optional[str] = None) -> Dict[str, Any]:
    """The request protocol of the judge profile (default ``judges.DEFAULT``): :func:`arm_protocol`."""
    return arm_protocol(J.profile(judge))


def arm_judge_spec(tools: Mapping[str, Mapping[str, Any]], prof: Mapping[str, Any], max_tokens: Sequence[int]) -> Dict[str, Any]:
    """The judge row of a profile (GLM-5.3): the Opus judge's rubric, version, scale and threshold rule; the candidate's protocol. The
    notes and extra are the text of the judge row on record (``tests/test_rejudge_paper_judges.py``)."""
    proto = arm_protocol(prof)
    cand = J.candidate(prof)
    return F.judge_payload(
        judge_model=prof["judge_model"], rubric_name="petri_bloom_judge", rubric_version=OPUS_RUBRIC_VERSION, rubric_sha256=rubric_sha256(tools),
        protocol_sha256=F.canonical_sha256(proto), score_scale=OPUS_SCORE_SCALE, threshold_rule=OPUS_THRESHOLD_RULE, judge_backend="batch",
        label_vocabulary={"petri_score": "integer 1-10 (the behaviour's own dimension)", "eval_awareness": "integer 1-10",
                          "scenario_realism": "integer 1-10", "petri_score_null_labels": list(UNSCORED_LABELS)},
        notes=(f"{prof['judge_model']} re-judge of the paper's Petri transcripts through OpenRouter's Batch API ({prof['model']}:batch), chosen by "
               f"{prof['chosen']}: the same prompt and answer tool the Opus 5 judge was sent (read from its stored API request, byte for "
               f"byte), the forced Anthropic tool call translated to chat completions (tool_choice forcing answer; parallel_tool_calls not "
               f"sent, the :batch variant does not list it), reasoning effort {cand['reasoning_effort']} (reasoning is mandatory on this "
               f"model and {cand['reasoning_effort']} is its floor; the Opus judge's forced call ran without thinking), max_tokens "
               f"{cand['max_tokens']:,} (a truncated reply re-asked once at {cand['truncated_retry_max_tokens']:,}), the prompt as one "
               f"string, no cache fields: exactly the bake-off arm {prof['arm']}'s request. Every verdict served by "
               f"{', '.join(prof['served_by'])} (batches pinned {json.dumps(prof['provider'])}; the bake-off's own verdicts, reused, were "
               f"served there unpinned). The Petri auditor's transcripts were not re-run. A SECOND judge beside the Opus 5 judge: its "
               f"judgments never supersede the Opus ones and the runs' primary judge is unchanged."),
        extra={"seed_judge_id": prof["seed_judge_id"], "openrouter_model": prof["model"] + ":batch", "canonical_slug": prof["canonical_slug"],
               "provider_route": "OpenRouter Batch API (POST /api/v1/batches)", "protocol": proto,
               "request_settings": dict(J.arm_request(prof), **J.arm_shape(prof)), "provider_pin": prof["provider"],
               "served_by": list(prof["served_by"]), "max_tokens": sorted({int(m) for m in max_tokens}),
               "truncated_retry_max_tokens": cand["truncated_retry_max_tokens"], "opus_max_tokens": R.OPUS_MAX_TOKENS,
               "reasoning_effort": cand["reasoning_effort"], "re_measures_judge_id": PETRI_JUDGE, "used_by": list(EVALS),
               "answer_tool_sha256": sorted(R.tool_sha(t) for t in tools.values()),
               "rubric_source": "inspect_petri 3.0.11 JUDGE_PROMPT / HIGHLIGHTS_DESCRIPTION + petri-bloom 0.2.6 behaviour bundle "
                                "8812ba083bb5473c answer tools, as stored in each judge request",
               "request_builder": f"scripts/rejudge/petri/bakeoff/models.py openrouter_body (arm {prof['arm']}), via scripts/rejudge/petri/judges.py",
               "custom_id": "request.judge_input_key", "bakeoff_arm": prof["arm"], "chosen": prof["chosen"]})


def judge_spec(tools: Mapping[str, Mapping[str, Any]], max_tokens: Optional[Sequence[int]] = None,
               judge: Optional[str] = None) -> Dict[str, Any]:
    """The judge row of the profile ``judge`` (:func:`arm_judge_spec`); ``max_tokens`` the ledgers' caps (default the candidate's)."""
    prof = J.profile(judge)
    return arm_judge_spec(tools, prof, max_tokens or [J.candidate(prof)["max_tokens"]])


def ledger_settings(run_dirs: Iterable[str], judge: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """``(identity settings, per-ledger settings)`` of the verdict ledgers (module docstring, THE VERDICT LEDGERS). Refuses
    (ValueError) a ledger of another track, a probe (``id_suffix``), or settings that are not the profile's."""
    per: Dict[str, Dict[str, Any]] = {}
    ident: Optional[Dict[str, Any]] = None
    first = None
    prof = J.profile(judge)
    for rd in run_dirs:
        tr = BF.read_track(rd)
        try:
            mine = J.ledger_identity(tr, prof)
        except ValueError as e:
            raise ValueError(f"{rd}: {e}")
        per[rd] = dict(mine, track=tr.get("track"), provider=tr.get("provider"), max_tokens=int((tr.get("request") or {}).get("max_tokens") or 0),
                       truncated_retry_max_tokens=tr.get("truncated_retry_max_tokens"))
        if ident is None:
            ident, first = mine, rd
        elif mine != ident:                       # ledger_identity pins both to the candidate's; kept as the general rule
            raise ValueError(f"{rd} was built with other request settings than {first}")
    if ident is None:
        raise ValueError("no verdict ledger (--run-dir) given")
    return ident, per


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------
def read_units(path: str, runs: Optional[set] = None) -> Dict[str, List[Dict[str, Any]]]:
    by_run: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    with gzip.open(path, "rt") as fh:
        cols = next(fh).rstrip("\n").split("\t")
        for line in fh:
            d = dict(zip(cols, line.rstrip("\n").split("\t")))
            if runs is not None and d["run_id"] not in runs:
                continue
            by_run[d["run_id"]].append(d)
    return by_run


def read_keys(enum_dir: str) -> Dict[str, Dict[str, Any]]:
    """phys -> {key, tool_sha, own_dimension}"""
    out = {}
    with gzip.open(os.path.join(enum_dir, "inputs.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if "error" not in r:
                out[r["phys"]] = {"key": r["key"], "tool_sha": r["tool_sha"], "own_dimension": r["own_dimension"]}
    return out


class DuplicateOutcomes(ValueError):
    pass


def read_outcomes(run_dirs: Iterable[str], keys: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """custom_id -> its terminal outcome in the run directories' ledgers. A key with a FILEABLE outcome in MORE THAN ONE ledger is
    refused (:class:`DuplicateOutcomes`): which verdict to file would otherwise depend on the ledgers' order. An infrastructure outcome
    gives way to a fileable one from another ledger, and of two infrastructure outcomes the later harvest stands (module docstring);
    the outcome kept then lists the ledgers it settled in ``settled_infra``."""
    out: Dict[str, Dict[str, Any]] = {}
    dup: Dict[str, List[str]] = collections.defaultdict(list)
    for rd in run_dirs:
        con = sqlite3.connect(f"file:{os.path.join(rd, 'state.sqlite')}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        for r in con.execute("SELECT custom_id, outcome, verdict_json, error_json, status_code, attempts, batch_id, provider, harvested_utc "
                             "FROM requests WHERE terminal = 1"):
            cid = r["custom_id"]
            if keys is not None and cid not in keys:
                continue
            d = dict(r)
            d["verdict"] = json.loads(d.pop("verdict_json")) if r["verdict_json"] else None
            d["error"] = json.loads(d.pop("error_json")) if r["error_json"] else None
            d["run_dir"] = rd
            if cid in out:
                prev = out[cid]
                if prev["outcome"] in FILEABLE and d["outcome"] in FILEABLE:
                    if not dup[cid]:
                        dup[cid].append(prev["run_dir"])
                    dup[cid].append(rd)
                    continue
                if prev["outcome"] in FILEABLE:
                    keep, lose = prev, d
                elif d["outcome"] in FILEABLE:
                    keep, lose = d, prev
                else:                                 # two infrastructure outcomes: the later harvest (then the path) stands
                    keep = max((prev, d), key=lambda o: (str(o.get("harvested_utc") or ""), o["run_dir"]))
                    lose = d if keep is prev else prev
                keep["settled_infra"] = sorted(set(prev.get("settled_infra", []) + d.get("settled_infra", []) + [lose["run_dir"]]))
                out[cid] = keep
                continue
            out[cid] = d
        con.close()
    if dup:
        first = sorted(dup)[:3]
        raise DuplicateOutcomes(f"{len(dup)} judge-input keys are terminal in more than one ledger (e.g. "
                                f"{[(k[:16], dup[k]) for k in first]}); build run directories disjoint (build_manifests.py --exclude-keys) "
                                f"and file each key from one ledger")
    return out


def git_head(path: str) -> str:
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


# --------------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------------
def run_rows(run_id: str, judgment_id: str, units: List[Mapping[str, Any]], keys: Mapping[str, Mapping[str, Any]],
             outcomes: Mapping[str, Mapping[str, Any]], tools: Mapping[str, Mapping[str, Any]], opus_judgment_id: str,
             created_utc: str, file_infra: bool = False, judge_model: str = JUDGE_MODEL) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One run's item rows and its counts. ``counts`` has one entry per unit's outcome (``unjudged`` for no terminal outcome,
    ``no_input`` for a transcript the enumeration could not read) plus ``infra_unfiled`` (infrastructure outcomes not filed as rows);
    ``filed_unscored`` counts the NULL rows by label; ``unsettled`` = units with neither a verdict nor a filed row."""
    rows: List[Dict[str, Any]] = []
    counts: collections.Counter = collections.Counter()
    filed_unscored: collections.Counter = collections.Counter()
    batch_ids: set = set()
    harvested: List[str] = []
    for u in units:
        item, trial, sample, turn = u["item_id"], int(u["trial_index"]), int(u["sample_index"]), int(u["turn_index"])
        k = keys.get(u["phys"])
        if k is None:
            counts["no_input"] += 1
            continue
        cid = k["key"]
        o = outcomes.get(cid)
        if o is None:
            counts["unjudged"] += 1
            continue
        if o.get("batch_id"):
            batch_ids.add(o["batch_id"])
        if o.get("harvested_utc"):
            harvested.append(o["harvested_utc"])
        common = {"request_key": cid, "batch_id": o.get("batch_id"), "attempt": o.get("attempts"), "judge_model": judge_model,
                  "transport": "openrouter_batch", "provider": o.get("provider"), "transcript": u["phys"],
                  "re_measures_judgment_id": opus_judgment_id}
        outcome, err, v = o["outcome"], o.get("error") or {}, o.get("verdict")
        if outcome == "verdict":
            problem = R.verdict_matches_tool(v, tools[k["tool_sha"]])
            if problem:
                outcome, err = "parse_error", {"kind": "own_dimension_mismatch", "message": problem}
        if outcome == "verdict":
            counts["verdict"] += 1
            for metric, value in (("petri_score", v["petri_score"]), ("eval_awareness", v["eval_awareness"]),
                                  ("scenario_realism", v["scenario_realism"])):
                ex = dict(common)
                if metric == "petri_score":
                    ex["own_dimension"] = v["own_dimension"]
                rows.append(F.item_row(run_id=run_id, judgment_id=judgment_id, item_id=item, metric_id=metric, trial_index=trial,
                                       sample_index=sample, turn_index=turn, value=float(value), label=None,
                                       judged_utc=o.get("harvested_utc"), created_utc=created_utc, extra=ex))
        else:
            counts[outcome] += 1
            if outcome not in ITEM_UNSCORED and not file_infra:
                counts["infra_unfiled"] += 1
                continue
            filed_unscored[outcome] += 1
            rows.append(F.item_row(run_id=run_id, judgment_id=judgment_id, item_id=item, metric_id="petri_score", trial_index=trial,
                                   sample_index=sample, turn_index=turn, value=None, label=outcome, judged_utc=o.get("harvested_utc"),
                                   created_utc=created_utc,
                                   extra=dict(common, error_kind=err.get("kind"), status_code=o.get("status_code"),
                                              error_message=str(err.get("message") or err.get("refusal") or "")[:300])))
    info = {"counts": dict(counts), "filed_unscored": dict(filed_unscored),
            "unsettled": counts.get("unjudged", 0) + counts.get("no_input", 0) + counts.get("infra_unfiled", 0),
            "batch_ids": sorted(batch_ids), "first_harvest": min(harvested) if harvested else None,
            "last_harvest": max(harvested) if harvested else None}
    return rows, info


def gate_problems(readies: Mapping[str, Mapping[str, Any]], partial_runs: List[str], allow_partial: bool) -> List[str]:
    """Why runs may not be delivered now (empty = deliverable): module docstring, THE DELIVERY GATE."""
    out = []
    for rd, r in readies.items():
        if r["paused"]:
            out.append(f"{rd} is PAUSED ({r['paused'].splitlines()[0][:200]})")
        if r["live_batches"]:
            out.append(f"{rd} still has live batches {r['live_batches']}")
        waiting = (r["requests"] - r["terminal"]) - r["non_terminal_in_refused"]
        if waiting:
            out.append(f"{rd} has {waiting} requests waiting on a re-ask or a person {r['non_terminal']}")
        if r["refused_batches"] and not allow_partial:
            out.append(f"{rd} holds set-aside / refused batches {r['refused_batches'][:5]} (their keys would file unjudged; --allow-partial accepts that)")
    if partial_runs and not allow_partial:
        out.append(f"{len(partial_runs)} runs would file as partial (e.g. {partial_runs[:3]}); --allow-partial accepts that, or finish them first")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope")
    ap.add_argument("--enum")
    ap.add_argument("--run-dir", action="append", default=[], help="a verdict ledger (repeatable; disjoint keys, one protocol)")
    ap.add_argument("--staging", required=True)
    ap.add_argument("--runs", help="file of run ids to file (default: every scope run with at least --min-terminal terminal units)")
    ap.add_argument("--min-terminal", type=int, default=1)
    ap.add_argument("--registry-root", help="the registry root (read only when staging: follow-up numbering; the delivery target)")
    ap.add_argument("--deliver", choices=("none", "judge", "runs", "all"), default="none")
    ap.add_argument("--host-label", default=F.DEFAULT_HOST_LABEL)
    ap.add_argument("--code-sha")
    ap.add_argument("--extra-json", help="JSON merged into every judgment's extra (e.g. a proof tag)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deliver-only", action="store_true", help="deliver the files an earlier invocation staged in --staging (no rebuild)")
    ap.add_argument("--allow-partial", action="store_true", help="deliver runs that file as partial, and ledgers holding set-aside batches")
    ap.add_argument("--file-infra-outcomes", action="store_true", help="also file infrastructure outcomes (provider_error, missing, batch_*) as rows")
    ap.add_argument("--allow-synthetic", action="store_true", help="file a SYNTHETIC run directory (scratch proofs only; never ~/registry)")
    ap.add_argument("--judge", default=J.DEFAULT, choices=sorted(J.PROFILES), help="the judge profile (judges.py) the ledgers hold verdicts of")
    args = ap.parse_args(argv)
    prof = J.profile(args.judge)
    live_root = os.path.realpath(os.path.expanduser("~/registry"))
    # the run directories key the staging's ledger fingerprints: one spelling per directory, whatever the shell passed
    args.run_dir = [os.path.abspath(os.path.expanduser(rd)) for rd in args.run_dir]
    to_live = bool(args.registry_root) and os.path.realpath(os.path.expanduser(args.registry_root)) == live_root
    unharvested = [rd for rd in args.run_dir if not os.path.exists(os.path.join(rd, "state.sqlite"))]
    if unharvested:
        raise SystemExit(f"REFUSED: {unharvested} hold no ledger (state.sqlite): never harvested, nothing to file")

    if args.deliver_only:
        if not args.registry_root or args.deliver == "none":
            raise SystemExit("--deliver-only needs --registry-root and --deliver judge|runs|all")
        if os.path.exists(os.path.join(args.staging, SYNTHETIC_MARK)) and to_live:
            raise SystemExit(f"{args.staging} holds SYNTHETIC records; refusing to deliver them into the live registry")
        run_dirs = None
        if args.runs:
            run_dirs = {F.identity.run_dir_name(l.strip()) for l in open(args.runs) if l.strip()}
        if args.deliver in ("runs", "all"):
            staged = json.load(open(os.path.join(args.staging, "filing_report.json")))
            fps = staged.get("ledger_fingerprints") or {}
            if not args.run_dir or set(args.run_dir) != set(fps):
                raise SystemExit(f"REFUSED: delivering runs needs --run-dir for exactly the ledgers {sorted(fps)} that {args.staging} was "
                                 f"staged from (got {sorted(args.run_dir)}). Nothing delivered.")
            readies = {rd: S.readiness(rd) for rd in args.run_dir}
            changed = [rd for rd in args.run_dir if fps.get(rd) != readies[rd]["fingerprint"]]
            if changed:
                raise SystemExit(f"REFUSED: the ledger of {changed} changed since {args.staging} was staged; restage into a fresh "
                                 f"directory. Nothing delivered.")
            partial = [r for r, v in staged["runs"].items() if v["status"] == "partial" and (run_dirs is None or F.identity.run_dir_name(r) in run_dirs)]
            problems = gate_problems(readies, partial, args.allow_partial)
            if problems:
                raise SystemExit("REFUSED (delivery gate): " + "; ".join(problems) + ". Nothing delivered.")
        dests = F.deliver_staged(args.staging, args.registry_root, args.host_label, None if args.deliver == "all" else args.deliver, run_dirs)
        print(f"delivered {len(dests)} already-staged files from {args.staging} into {args.registry_root}/inbox/{args.host_label}/ "
              f"(files whose identical bytes already stood were skipped)")
        return 0
    if not (args.scope and args.enum and args.run_dir):
        raise SystemExit("--scope, --enum and at least one --run-dir are required to stage")
    synthetic = [rd for rd in args.run_dir if os.path.exists(os.path.join(rd, SYNTHETIC_MARK))]
    if synthetic and not args.allow_synthetic:
        raise SystemExit(f"run directories {synthetic} hold SYNTHETIC verdicts; refusing without --allow-synthetic")
    if synthetic and to_live:
        raise SystemExit("SYNTHETIC verdicts are never filed into the live registry")
    if os.path.isdir(args.staging) and os.listdir(args.staging):
        raise SystemExit(f"--staging {args.staging} is not empty: every filing stages into a fresh directory")
    try:
        settings, per_ledger = ledger_settings(args.run_dir, args.judge)
    except ValueError as e:
        raise SystemExit(f"REFUSED: {e}")
    readies = {rd: S.readiness(rd) for rd in args.run_dir}
    delivering_runs = args.deliver in ("runs", "all") and not args.dry_run
    repo = os.path.dirname(_SCRIPTS)
    code_sha = args.code_sha or git_head(repo)
    jobs = {j["run_id"]: j for j in (json.loads(l) for l in open(os.path.join(args.scope, "jobs.jsonl")))}
    runs = {l.strip() for l in open(args.runs) if l.strip()} if args.runs else None
    units = read_units(os.path.join(args.scope, "units.tsv.gz"), runs)
    keys = read_keys(args.enum)
    tools = json.load(open(os.path.join(args.enum, "tools.json")))
    try:
        outcomes = read_outcomes(args.run_dir, {k["key"] for k in keys.values()})
    except DuplicateOutcomes as e:
        raise SystemExit(f"REFUSED: {e}")
    served = collections.Counter(o.get("provider") for o in outcomes.values() if o["outcome"] in ("verdict",) + ITEM_UNSCORED)
    bad_served = J.served_problems(prof, served)
    if bad_served:
        raise SystemExit("REFUSED: " + "; ".join(bad_served) + ". Nothing staged.")
    extra_all = json.loads(args.extra_json) if args.extra_json else {}
    created = F.utc_ms()
    judge = judge_spec(tools, [v["max_tokens"] for v in per_ledger.values()], args.judge)
    filer = F.Filer(args.staging, prof["track"], judge, "scripts/rejudge/petri/file_records.py", code_sha,
                    registry_root=args.registry_root, host_label=args.host_label)
    if filer.stage_judge() is None:
        print(f"judge {filer.judge_id[:12]} is already on record in {args.registry_root}: not staged again")
    report: Dict[str, Any] = {"code_sha": code_sha, "judge": prof["name"], "judge_model": prof["judge_model"], "track": prof["track"],
                              "judge_id": judge["judge_id"], "rubric_sha256": judge["rubric_sha256"],
                              "protocol_sha256": judge["protocol_sha256"], "request_settings": settings, "ledgers": per_ledger,
                              "served_by_counts": {str(k): v for k, v in served.items()},
                              "infra_outcomes_settled_by_another_ledger": sum(1 for o in outcomes.values() if o.get("settled_infra")),
                              "synthetic": bool(synthetic), "runs": {}, "totals": collections.Counter()}
    for run_id, us in sorted(units.items()):
        job = jobs[run_id]
        n_term = sum(1 for u in us if keys.get(u["phys"]) and keys[u["phys"]]["key"] in outcomes)
        if n_term < args.min_terminal:
            continue
        jid = F.identity.judgment_id(run_id, filer.judge_id)
        rows, info = run_rows(run_id, jid, us, keys, outcomes, tools, job["opus_judgment_id"], created, file_infra=args.file_infra_outcomes,
                              judge_model=prof["judge_model"])
        c = info["counts"]
        status = "complete" if info["unsettled"] == 0 else "partial"
        n_ref = info["filed_unscored"].get("judge_refusal", 0)
        n_err = sum(v for k, v in info["filed_unscored"].items() if k != "judge_refusal")
        # a pooled run scoped by pooled_scope.py names the behaviour runs its units were composed from (scope.py's jobs carry no such key)
        composed = {"units_composed_from": job["units_composed_from"]} if job.get("units_composed_from") else {}
        judgment = F.judgment_payload(
            run_id=run_id, judge_id=filer.judge_id, status=status, n_items_judged=c.get("verdict", 0), n_judge_refusals=n_ref,
            n_errors=n_err, started_utc=info["first_harvest"], finished_utc=info["last_harvest"], batch_ids=info["batch_ids"],
            grading_code_sha=code_sha, rubric_sha256=judge["rubric_sha256"], judge_cache_kind="petri_judge_input_sha256", store_uri=None,
            producer_session=filer.session,
            extra=dict({"judge_model": prof["judge_model"], "transport": "openrouter_batch",
                        "unit": "one judged transcript per registry item (behaviour/scenario, trial)", "n_units": len(us),
                        "n_distinct_keys": len({keys[u['phys']]['key'] for u in us if u['phys'] in keys}), "outcome_counts": c,
                        "filed_unscored": info["filed_unscored"], "unsettled": info["unsettled"],
                        "re_measures_judgment_id": job["opus_judgment_id"], "re_measures_judge_id": PETRI_JUDGE,
                        "scope": "paper rows (v_run_current.in_current_paper = 1), 2026-11-09", "verdict_ledgers": sorted(per_ledger),
                        "ledger_settings": per_ledger, "reasoning_effort": settings["reasoning_effort"],
                        "infra_outcomes_filed": bool(args.file_infra_outcomes), "follow_up_of_seq": filer.seq_floor(run_id) or None},
                       **dict(composed, **extra_all)))
        filer.stage_run(run_id, judgment, rows)
        report["runs"][run_id] = {"status": status, "rows": len(rows), **c}
        report["totals"].update(c)
        report["totals"]["runs_" + status] += 1
        report["totals"]["item_rows"] += len(rows)
    report["validation"] = filer.validate()
    report["totals"] = dict(report["totals"])
    partial_runs = sorted(r for r, v in report["runs"].items() if v["status"] == "partial")
    problems = gate_problems(readies, partial_runs, args.allow_partial)
    report["ledger_fingerprints"] = {rd: r["fingerprint"] for rd, r in readies.items()}
    report["readiness"] = {rd: {k: v for k, v in r.items() if k != "fingerprint"} for rd, r in readies.items()}
    report["gate"] = {"allow_partial": args.allow_partial, "problems": problems, "seq_floor_root": args.registry_root,
                      "follow_up_runs": sorted(r for r in filer.seq_floors if r != "judge")}
    os.makedirs(args.staging, exist_ok=True)
    if synthetic:
        open(os.path.join(args.staging, SYNTHETIC_MARK), "w").write("these records carry SYNTHETIC verdicts (scratch proof); never deliver them to ~/registry\n")
    with open(os.path.join(args.staging, "filing_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    v = report["validation"]
    print(json.dumps({"judge_id": report["judge_id"], "totals": report["totals"], "follow_up_runs": len(report["gate"]["follow_up_runs"]),
                      "validation": {"records": v["records"], "files": v["files"], "refusals": len(v["refusals"])}}, indent=1))
    if problems:
        print(("the delivery gate WOULD REFUSE the runs: " if not delivering_runs else "REFUSED (delivery gate): ") + "; ".join(problems))
    if v["refusals"]:
        print(f"REFUSED: {len(v['refusals'])} records failed validation (first: {v['refusals'][0]}); nothing delivered")
        return 2
    if args.dry_run or args.deliver == "none":
        print("dry run / no delivery requested: staged and validated only")
        return 0
    if not args.registry_root:
        raise SystemExit("--deliver needs --registry-root")
    if delivering_runs and problems:
        print("nothing delivered")
        return 3
    dests = filer.deliver(args.registry_root, args.host_label, only=None if args.deliver == "all" else args.deliver)
    print(f"delivered {len(dests)} files of judge {filer.judge_id[:12]} into {args.registry_root}/inbox/{args.host_label}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
