"""THE PROOF ON REAL RUNS (slow, opt-in: SISTEER_SLOW_TESTS=1): a row synthesized for a run that DOES carry run_metric rows, from its
policy judge's items and with its own filed measurement identities (`synthesized_rows(force_run_ids=...)`), equals the re-derived row
of the same run after the loader's spelling and judge selection, in every reading column and every run-level column the paper view
gives, and in its roster match (comparison and roster-entry ids, as sets: the policy judge standing in for each entry's judge yields the
entries the filed judge satisfies). Read on the committed paper frame (or REGISTRY_NB_SNAPSHOT_ROOT's), which must carry the paper
policy's item judges; about ten minutes (the frame is reassembled under the notebook's cache and every policy judge's item rows are read).

Columns that belong to a FILED measurement and have no source for a run without one (the
producer's spelling, computation, source and extra json, writer-filed degeneration mirrors, filed contrasts, multi-source ownership,
the metric-role bookkeeping and the filed copies `primary_*`) are not compared; `judge_source` differs by design (it names the
synthesis). A row neither side can read (no rule for its metric) keeps the judgment it came in with on each side."""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

pytestmark = pytest.mark.skipif(os.environ.get("SISTEER_SLOW_TESTS") != "1", reason="reads every policy judge's item rows of the "
                                "committed frame (~10 min, ~4 GB cache); SISTEER_SLOW_TESTS=1 opts in")

COMPARED = ["value", "k", "n", "ci_lo", "ci_hi", "ci_method", "judgment_id", "judge_id", "judge_model", "judge_ok", "seed", "lane", "dose",
            "object_label", "at_operating_point", "draw", "headline", "in_current_paper", "selection_state", "held_out", "is_complete",
            "operating_point_basis", "decode_family_default_policy_id", "decode_is_family_default", "token_cap_tokens", "item_set_id",
            "instrument_id", "serving_path", "decode_source", "paper_label", "object_id", "eval_category", "headline_metric_id", "campaign",
            "lane_source_legacy", "view_legacy", "transfer_kind", "strength_nominal", "is_operating_point", "pass_label", "lifecycle_state",
            "n_trials", "trial_index", "seed_base", "judgment_n_judge_refusals", "bon_n", "max_turns", "harness", "on_lane", "lane_pulled"]
#: compared as sets: a synthesized row's roster match (`synth_roster_matches`, the policy judge standing in for each entry's judge) equals
#: the filed row's (`v_roster_match`) whatever order the view's json_group_array lists it in
COMPARED_SETS = ["comparison_ids_json", "roster_entry_ids_json"]
EVALS = ["grok420", "pasttense", "crescendo", "xstest", "or_bench_hard"]     # no agentic eval: the paper omits them, and none is synthesized


def _same(p, q):
    if pd.isna(p) and pd.isna(q):
        return True
    try:
        return abs(float(p) - float(q)) < 1e-9
    except (TypeError, ValueError):
        return str(p) == str(q)


def test_a_synthesized_row_equals_the_rederived_row_of_the_same_run(monkeypatch):
    import registry_nb as R
    root = os.environ.get("REGISTRY_NB_SNAPSHOT_ROOT") or str(pathlib.Path(R.SNAPSHOT_ROOT))
    from sisteer.registry import snapshot as SN
    sid, man = SN.paper_frame_current(root)
    if not man.get("item_judges"):
        pytest.skip(f"the frame {sid} carries no item judges (read_set_version {man.get('read_set_version')}): nothing to re-derive from")
    monkeypatch.setattr(R, "SOURCE", "snapshot")
    monkeypatch.setattr(R, "SNAPSHOT_ROOT", root)
    monkeypatch.setattr(R, "JUDGES", "rejudge")
    for f in (R.judge_item_rows, R._judge_runs, R.resolve_judge, R.judge_model_of):
        f.cache_clear()
    R.JUDGE_REPORT.clear()
    ml = ",".join(f"'{m}'" for m in R.MODEL_ORDER)
    raw = R.query(f"SELECT v.*, r.seed_base, r.trial_index FROM v_paper_long v JOIN run r ON r.run_id = v.run_id WHERE v.model_id IN ({ml}) "
                  f"AND (v.in_current_paper = 1 OR (v.selection_state = '{R.PROVISIONAL_STATE}' AND v.metric_role = 'owner'))", "paper_all")
    sub = raw[raw.model_id.eq(R.MODEL_ID) & raw.eval_id.isin(EVALS)].copy()
    sub.attrs = dict(raw.attrs)
    pick = []
    for e in EVALS:
        pick += list(dict.fromkeys(sub[sub.eval_id.eq(e) & sub.paper_label.ne("none")].run_id.astype(str)))[:3]
    pick += list(dict.fromkeys(sub[sub.eval_id.eq("grok420") & sub.paper_label.eq("none")].run_id.astype(str)))[:2]
    syn = R.synthesized_rows(raw, pol=R.judge_policy("rejudge"), force_run_ids=pick)
    assert len(syn) and set(syn.run_id) == set(pick) and list(syn.columns[:len(raw.columns)]) == list(raw.columns)
    a = R.spell(sub)
    b_in = pd.concat([sub[~sub.run_id.astype(str).isin(pick)], syn.reindex(columns=list(sub.columns) + ["synthesized", "synth_carries"])],
                     ignore_index=True)
    b_in.attrs = dict(sub.attrs)
    b = R.spell(b_in)
    for f in (a, b):                                 # the paper lane as the load cell admits it (`admit_lane`, over the whole frame)
        f["on_lane"], f["lane_pulled"] = R.admit_lane(f)
    key = ["run_id", "metric_id", "threshold"]
    a, b = (f[f.run_id.astype(str).isin(pick)].assign(threshold=lambda x: pd.to_numeric(x.threshold, errors="coerce").round(6)) for f in (a, b))
    m = a.merge(b, on=key, suffixes=("_a", "_b"), how="outer", indicator=True)
    assert m["_merge"].eq("both").all(), m.loc[~m["_merge"].eq("both"), key + ["_merge"]].to_dict("records")[:5]
    assert b["synthesized"].all() and b["judge_source"].astype(str).str.contains("synthesized|not re-derivable|no complete").all()
    bad = {}
    for c in COMPARED:
        blank_both = (m["value_a"].isna() & m["value_b"].isna() & m["judge_ok_a"].eq(False) & m["judge_ok_b"].eq(False)
                      if c in ("judgment_id", "judgment_n_judge_refusals") else pd.Series(False, index=m.index))
        off = [(r, p, q) for r, p, q, bb in zip(m["run_id"], m[c + "_a"], m[c + "_b"], blank_both) if not (_same(p, q) or bb)]
        if off:
            bad[c] = off[:3]
    as_set = lambda v: frozenset(json.loads(v)) if isinstance(v, str) and v.strip() else frozenset()
    for c in COMPARED_SETS:
        off = [(r, p, q) for r, p, q in zip(m["run_id"], m[c + "_a"], m[c + "_b"]) if as_set(p) != as_set(q)]
        if off:
            bad[c] = off[:3]
    assert not bad, json.dumps(bad, default=str)
    read = m[m["value_a"].notna()]
    assert len(read) >= 15 and read["eval_id_a"].nunique() >= 5, "the proof reads real readings on most of the families"
