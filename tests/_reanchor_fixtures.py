"""The anchor-decision harness (commit 2 of the sequencing): a synthetic adapter that files a contrast on
every steered store against the anchor the pass's currency chooses, over the synthetic store world of `test_registry_adapter_pass`.
Shared by the re-read, the cluster-read and the no-re-emit tests. Everything here is invented (synthetic ids, TESTHOST producers, tiny
numeric shapes); nothing is a real prompt, transcript or generation.

Why a harness adapter and not a production one: commit 2 wires the decision set into the runner's identity machinery and hooks NO adapter
(the per-family hooks are commits 3-7), so the end-to-end path (read -> own contrast -> a ruling retires the anchor -> the next pass plans
exactly that group, mints a new revision id with no file change and no bump, re-anchors, withdraws the old key -> quiet) needs a reader
that calls `ctx.anchor_currency().choose(...)` the way the hooked adapters will.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store  # noqa: E402

#: the harness's unsteered stores: s2 is the steered store's OWN anchor (its `anchor_key`), s4 a second unsteered run of the same source at
#: the same key (equal timestamps, so the two coexist under the later-run rule); the steered store is `_setup`'s s1
STEERED, OWN, SIBLING = "legacy_store_1", "legacy_store_2", "legacy_store_4"
PARK_PHRASE = "contrast has no current anchor at key"


def run_id_of(key: str, obj: str = "none") -> str:
    return I.run_id(synth.CAMPAIGN, key, obj, synth.EVAL_ID, synth.INSTRUMENT_ID)


def store_with_anchor(store_root: str, name: str, key: str, obj: str, scores: Sequence[float], anchor_key: Optional[str]) -> str:
    """A steered store naming its own unsteered store (the adapter composes its own anchor id from it); None declares no own anchor."""
    d = os.path.join(store_root, name)
    os.makedirs(d, exist_ok=True)
    body: Dict[str, object] = {"key": key, "object": obj, "items": [{"id": f"q{i}", "score": s} for i, s in enumerate(scores)]}
    if anchor_key is not None:
        body["anchor_key"] = anchor_key
    with open(os.path.join(d, "store.json"), "w") as fh:
        json.dump(body, fh)
    return os.path.join(d, "store.json")


class ContrastStoreAdapter(SyntheticStoreAdapter):
    """`SyntheticStoreAdapter` (the same name, so the plain adapter's heads are THIS adapter's heads: the first read under the bump is the
    first-landing case) plus, on every steered store, the anchor consultation the hooked adapters will make: the key
    from the steered run payload, the own anchor composed from the store's `anchor_key` (none declared without one), the standup = the
    own unsteered store's producer key. A chosen anchor files a `run_contrast` (an unpaired difference of the two values; the harness
    statistic) with `extra.anchor_selection`; a parked comparison files no contrast and names the park in `row_refusals`."""
    EMISSION_VERSION = 2
    #: whether the harness statistic is paired over registry items (e4 asked of every candidate): `PairedContrastStoreAdapter`
    NEEDS_ITEMS = False

    def __init__(self, store_root: str):
        super().__init__(store_root)
        self.last_report = SimpleNamespace(refusals=[])
        self.choices: Dict[str, AC.AnchorChoice] = {}        # group -> the last choice (tests read it)

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        self.last_report = SimpleNamespace(refusals=[])
        for inp, batch in zip(inputs, super().read(ctx, group, inputs)):
            run = next(r for r in batch.records if r["record_type"] == "run")["payload"]
            if run["object_id"] != "none":
                with open(inp.path) as fh:
                    store = json.load(fh)
                metric = next(r for r in batch.records if r["record_type"] == "run_metric")["payload"]
                anchor_key = store.get("anchor_key")
                key = AC.AnchorKey(model_id=run["model_id"], eval_id=run["eval_id"], instrument_id=run["instrument_id"],
                                   token_cap_tokens=run.get("token_cap_tokens"), trial_index=run.get("trial_index", 0), pass_label=run.get("pass_label"),
                                   decode_policy_id=None, lane_source_legacy=run["lane_source_legacy"],
                                   standup=({"producer_run_key_legacy": anchor_key} if anchor_key else {}),
                                   own_anchor_run_id=(run_id_of(anchor_key) if anchor_key else None),
                                   own_anchor_kind=("same_standup_none" if anchor_key else None))
                choice = ctx.anchor_currency().choose(key, metric["metric_id"], metric["threshold"], run["judge_id"], computation_id="",
                                                      needs_items=self.NEEDS_ITEMS, steered_run_id=run["run_id"], steered_judgment_id=metric["judgment_id"])
                self.choices[group] = choice
                if choice.run_id is None:
                    self.last_report.refusals.append(f"{run['run_id']}: {PARK_PHRASE} cap {key.token_cap_tokens} trial {key.trial_index} "
                                                     f"pass {key.pass_label}; filed without a contrast: {choice.why}")
                else:
                    assert choice.measurement is not None
                    batch.add("run_contrast", {"run_id": run["run_id"], "anchor_run_id": choice.run_id, "judgment_id": metric["judgment_id"],
                                               "metric_id": metric["metric_id"], "threshold": metric["threshold"], "computation_id": "",
                                               "anchor_judgment_id": choice.anchor_judgment_id, "anchor_kind": choice.anchor_kind,
                                               "delta": float(metric["value"]) - float(choice.measurement.value or 0.0),
                                               "delta_method": "unpaired difference of values (synthetic harness)", "n_pairs": 0,
                                               "computed_by": "registry", "extra": {"anchor_selection": choice.selection_json()}},
                              ctx.producer, ctx.now)
            yield batch


class PairedContrastStoreAdapter(ContrastStoreAdapter):
    """The same harness with a statistic paired over registry items: e4 asked of every candidate, so the anchor must carry current item rows
    at the metric under its judgment and none of them may carry conflicting visible provenances (the item rows every synthetic store emits)."""
    NEEDS_ITEMS = True


class StoppedFilingAdapter(SyntheticStoreAdapter):
    """The same reader, bumped once more, filing no contrast and consulting no anchor: the head's decision set has no successor."""
    EMISSION_VERSION = 3


def ingest(root: str):
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked, rep.refusals)
    return rep


def seed_contrast_metric(root: str) -> None:
    """`asr` enters the contrast metric set: the frame reads the unsteered runs' readings at the metrics SOME contrast files,
    so a registry with no contrast yet loads no reading. The seed is a judged pair at ANOTHER key (`pass_label` 'seed'): never a candidate
    at the harness's key, never its later run."""
    none_id, none_recs = synth.run_records("seed_none", 500, "none", "none", 4, pass_label="seed")
    lora_id, lora_recs = synth.run_records("seed_lora", 520, "constitution_lora@seed", synth.OBJECT_SHA, 4, pass_label="seed")
    contrast = synth.rec("run_contrast", {"run_id": lora_id, "anchor_run_id": none_id, "judgment_id": I.judgment_id(lora_id, synth.JUDGE_ID),
                                          "metric_id": "asr", "threshold": 0.5, "anchor_judgment_id": I.judgment_id(none_id, synth.JUDGE_ID),
                                          "anchor_kind": "same_standup_none", "delta": 0.0, "delta_method": "seed", "n_pairs": 4,
                                          "computed_by": "registry"}, synth.t(540))
    R.write_records(os.path.join(root, "records", "seed", "000001-seed-contrast.jsonl"), none_recs + lora_recs + [contrast])
    ingest(root)


def file_ruling(root: str, ruling_id: str, action: str, run_id: str, i: int) -> None:
    """One ruling of `action` on one run (the pre-match-form run scope), filed and folded. A later call with the SAME ruling id restates
    the ruling -- the lift of a hold-out is its re-file as `reference`."""
    payload = {"ruling_id": ruling_id, "round_id": "r_synth", "action": action, "scope_kind": "run", "scope_json": {"run_id": run_id},
               "target_json": {}, "reason": f"synthetic: {action} {run_id}", "ruled_utc": synth.t(i), "ruled_utc_precision": "datetime",
               "ruled_by": "test", "source_ref": "synthetic"}
    R.write_records(os.path.join(root, "records", "rulings", f"{i:06d}-{ruling_id}.jsonl"), [synth.rec("ruling", payload, synth.t(i))])
    ingest(root)


def conflicting_provenance(root: str, run_id: str, judgment_id: str, item_id: str, value: float, i: int, source_id: Optional[str] = None) -> str:
    """A SECOND accepted provenance of one of a run's item keys at `asr` that disagrees with the adapter's row, ranked BELOW it (equal
    priority 0, a lexically later source id -- the tie-break), filed and folded: `v_item_result_current` keeps selecting the adapter's row
    while `v_item_conflict` reports the key. Each call is its OWN source (`zz_second_source_<i>` unless named): a second revision of one
    source would supersede its first and lift that conflict. Returns the revision id of the marker it wrote."""
    source_id = source_id or f"zz_second_source_{i}"
    rev = f"{source_id}:rev{i}"
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": source_id, "adapter": "synthetic_second", "observed_utc": synth.t(i),
                                           "inputs_json": {f"synthetic:///second/{i}": "b" * 64}, "accepted": 1, "priority": 0}, synth.t(i))
    row = synth.rec("item_result", {"run_id": run_id, "item_id": item_id, "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1,
                                    "judgment_id": judgment_id, "value": value, "label": None, "api_pool": None, "created_utc": synth.t(i),
                                    "judged_utc": None, "source_revision_id": rev, "withdrawn_by_revision_id": None}, synth.t(i + 1))
    R.write_records(os.path.join(root, "records", f"second_{i}", "000001-source_revision.jsonl"), [marker])
    R.write_records(os.path.join(root, "records", f"second_{i}", "000002-item_result.jsonl.gz"), [row], gz=True)
    ingest(root)
    return rev


def contrast_world(tmp_path, stores_dir: Optional[str] = None):
    """The registry root (dims, roster, the seed contrast metric) and a store root with the steered store naming s2 as its own anchor,
    s2 and s4 unsteered at the same key. Returns `(root, store_root)`."""
    root, default_stores = _setup(tmp_path)
    seed_contrast_metric(root)
    store_root = stores_dir or default_stores
    if stores_dir:
        os.makedirs(store_root, exist_ok=True)
    store_with_anchor(store_root, "s1", STEERED, "constitution_lora@aaaa", [0.2, 0.8, 0.4], anchor_key=OWN)
    _store(store_root, "s2", OWN, "none", [0.1, 0.9])
    _store(store_root, "s4", SIBLING, "none", [0.3, 0.7])
    return root, store_root


def pass_once(root: str, adapter, now: str, **kw) -> RUN.PassReport:
    with DB.reading(DB.db_path(root)) as conn:
        return RUN.run_pass(root, [adapter], conn=conn, now=now, producer=synth.producer(), **kw)


def pass_and_fold(root: str, adapter, now: str, **kw):
    rep = pass_once(root, adapter, now, **kw)
    return rep, ingest(root)


def q(root: str, sql: str, *params):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def head_of(root: str, source_id: str):
    """`(revision_id, extra)` of the source's accepted head."""
    rows = q(root, "SELECT revision_id, extra_json FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1",
             source_id)
    assert rows, source_id
    return rows[0][0], json.loads(rows[0][1] or "{}")


def decision_watermark(root: str, source_id: str):
    """`(sha256, last_error)` of the source's anchor-decision watermark, or None."""
    rows = q(root, "SELECT sha256, last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", source_id, AC.decision_uri(source_id))
    return rows[0] if rows else None


def paper_contrast(root: str, run_id: str) -> List[tuple]:
    """`(anchor_run_id, anchor_kind, n_current_contrasts)` of the run's `asr` row in the paper frame."""
    return q(root, "SELECT anchor_run_id, anchor_kind, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr'", run_id)


def current_contrast_keys(root: str, source_id: str) -> List[str]:
    """The source's CURRENT run_contrast observation keys (current = 1 under its head)."""
    return [r[0] for r in q(root, "SELECT observation_key FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1 "
                                  "ORDER BY observation_key", source_id)]
