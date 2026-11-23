"""The Petri judge is a loader parameter (`registry_nb.select_petri_judge`, REGISTRY_NB_PETRI_JUDGE / REGISTRY_NB_PETRI_EXCLUDE, 2026-11-11).

A re-judge filed as a new judge carries item rows only, so the loader re-derives each Petri `petri_score` row from the selected judge's
items by the filed row's own rules. Pins, on a synthetic frame and synthetic item rows (no registry): `primary` with no exclusion file is
the identity; a behaviour run's value is the mean over scored items and its `n` the scored count; the pooled run is the mean of the
behaviour means, composed from the six behaviour runs when it has no items of its own; an interval and a contrast are recomputed only
where the filed row carries one, the contrast paired over the items both runs scored against the anchor the filed contrast names; an
exclusion leaves both sides of a contrast; a run with no items under the judge reads NaN and is named, never the primary judge's number."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import registry_nb as R  # noqa: E402

PRIMARY = R.PETRI_JUDGE_ALIASES["claude-opus-5"]
NEW = R.PETRI_JUDGE_ALIASES["glm-5.3"]
B = R.PETRI_BEHAVIORS


def run(eval_id: str, obj: str) -> str:
    return f"camp:{eval_id}:aaaaaaaaaaaa:{obj}"


def items(judge: str) -> pd.DataFrame:
    """Two objects (steered `objs`, unsteered `objn`), every behaviour with three scenarios; the new judge scores 1 lower on `s1`."""
    rows = []
    for obj, base in (("objs", 3.0), ("objn", 6.0)):
        for b in B:
            for k, sc in enumerate(("s1", "s2", "s3")):
                v = base + k - (1.0 if (judge == NEW and sc == "s1") else 0.0)
                rows.append(dict(run_id=run(b, obj), item_id=f"{b[6:]}/{sc}", trial_index=0, sample_index=0,
                                 judgment_id=f"{judge[:4]}-{obj}-{b}", value=v, label=None,
                                 judgment_status="complete", judgment_finished_utc="2026-11-11T00:00:00.000Z"))
    # the primary judge refused one steered defer-to-users transcript (s2, rows 3-5 are that run): a null score
    if judge == PRIMARY:
        rows[4]["value"] = np.nan
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def fake_items(monkeypatch):
    monkeypatch.setattr(R, "petri_item_scores", lambda jid: items(jid))


def frame() -> pd.DataFrame:
    """Filed rows: the steered defer-to-users run (an interval and a contrast against the unsteered run), its anchor, and the steered
    pooled run, which has no item rows of its own (an export that filed the mean alone)."""
    b = B[1]
    return pd.DataFrame([
        dict(run_id=run(b, "objs"), eval_id=b, metric_id="petri_score", value=4.0, n=3.0, ci_lo=3.0, ci_hi=5.0, ci_method="10000 draws",
             delta=-3.0, delta_lo=-4.0, delta_hi=-2.0, delta_method="percentile bootstrap 5000", anchor_run_id=run(b, "objn"),
             judgment_id=f"{PRIMARY[:4]}-objs-{b}", judge_id=PRIMARY, judge_model="claude-opus-5"),
        dict(run_id=run(b, "objn"), eval_id=b, metric_id="petri_score", value=7.0, n=3.0, ci_lo=np.nan, ci_hi=np.nan, ci_method=None,
             delta=np.nan, delta_lo=np.nan, delta_hi=np.nan, delta_method=None, anchor_run_id=None,
             judgment_id=f"{PRIMARY[:4]}-objn-{b}", judge_id=PRIMARY, judge_model="claude-opus-5"),
        dict(run_id=run(R.PETRI_POOLED, "objs"), eval_id=R.PETRI_POOLED, metric_id="petri_score", value=4.0, n=np.nan, ci_lo=np.nan,
             ci_hi=np.nan, ci_method=None, delta=np.nan, delta_lo=np.nan, delta_hi=np.nan, delta_method=None, anchor_run_id=None,
             judgment_id="filed-pooled", judge_id=PRIMARY, judge_model="claude-opus-5"),
        dict(run_id="camp:pasttense:aaaaaaaaaaaa:objs", eval_id="pasttense", metric_id="asr", value=0.4, n=100.0, ci_lo=np.nan, ci_hi=np.nan,
             ci_method=None, delta=np.nan, delta_lo=np.nan, delta_hi=np.nan, delta_method=None, anchor_run_id=None, judgment_id="j-sr",
             judge_id="other", judge_model="claude-haiku-4-5"),
    ])


def test_primary_without_exclusions_is_the_identity():
    f = frame()
    out = R.select_petri_judge(f, spec="primary", exclude="")
    assert out is f


def test_new_judge_rederives_value_n_interval_and_contrast():
    out = R.select_petri_judge(frame(), spec="glm-5.3", exclude="").set_index("run_id")
    b = B[1]
    s, n = out.loc[run(b, "objs")], out.loc[run(b, "objn")]
    assert s.value == pytest.approx(np.mean([2.0, 4.0, 5.0])) and s.n == 3.0       # 3 - 1, 4, 5
    assert n.value == pytest.approx(np.mean([5.0, 7.0, 8.0]))
    assert s.delta == pytest.approx(s.value - n.value)                             # paired on the three shared scenarios
    assert s.delta_lo <= s.delta <= s.delta_hi and s.ci_lo <= s.value <= s.ci_hi
    assert np.isnan(n.ci_lo) and np.isnan(n.delta)                                  # nothing filed there, nothing invented
    assert s.judge_id == NEW and s.judge_model == "glm-5.3" and s.judgment_id == f"{NEW[:4]}-objs-{b}"
    assert out.loc["camp:pasttense:aaaaaaaaaaaa:objs"].value == 0.4                # a non-Petri row is untouched


def test_pooled_run_without_items_is_composed_from_its_behaviour_runs():
    out = R.select_petri_judge(frame(), spec="glm-5.3", exclude="").set_index("run_id")
    p = out.loc[run(R.PETRI_POOLED, "objs")]
    assert p.value == pytest.approx(np.mean([np.mean([2.0, 4.0, 5.0])] * len(B)))  # the mean of the six behaviour means
    assert "composed" in p.petri_judge_source and np.isnan(p.n)


def test_primary_rederivation_skips_the_refused_item():
    d = R.select_petri_judge(frame(), spec="claude-opus-5", exclude="").set_index("run_id").loc[run(B[1], "objs")]
    assert d.value == pytest.approx(4.0) and d.n == 2.0                          # 3 and 5 scored, the refusal never imputed
    assert d.delta == pytest.approx(-3.0)                                         # paired on s1, s3 only: (3 - 6, 5 - 8)


def test_exclusion_leaves_both_sides_of_the_contrast(tmp_path):
    b = B[1]
    x = tmp_path / "excl.tsv"
    x.write_text(f"run_id\titem_id\ttrial_index\n{run(b, 'objs')}\t{b[6:]}/s3\t0\n{run(b, 'objn')}\t{b[6:]}/s3\t0\n")
    out = R.select_petri_judge(frame(), spec="glm-5.3", exclude=str(x)).set_index("run_id")
    s = out.loc[run(b, "objs")]
    assert s.value == pytest.approx(3.0) and s.n == 2.0 and s.petri_n_excluded == 1.0
    assert s.delta == pytest.approx(np.mean([2.0 - 5.0, 4.0 - 7.0]))


def test_a_run_the_judge_never_scored_reads_nan_and_is_named(monkeypatch):
    monkeypatch.setattr(R, "petri_item_scores", lambda jid: items(jid)[lambda t: ~t.run_id.str.contains(":objn")])
    R.PETRI_JUDGE_REPORT.clear()
    out = R.select_petri_judge(frame(), spec="glm-5.3", exclude="").set_index("run_id")
    assert np.isnan(out.loc[run(B[1], "objn")].value)
    assert np.isnan(out.loc[run(B[1], "objs")].delta)                              # its contrast lost the anchor's items
    rep = R.PETRI_JUDGE_REPORT["frames"][-1]
    assert run(B[1], "objn") in rep["missing"] and run(B[1], "objs") in rep["contrast_missing_anchor"]
