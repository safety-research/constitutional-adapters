"""The decode-only conditions are ONE dot per cell, the mean of their matched runs (the owner's rule, 2026-11-09, final).

The Refusal steer (`arditi_allpos`), the Safety SI and the Constitution SI have no fit: every run of one of them is a decode replicate
of one object. The panels drew one dot per instrument such a condition had been run under, so a cell showed one, two or three dots of
one condition: the SIs split by lane (campaign readings at another cap or decode beside their seed-replicate runs), the Refusal steer
by item set (a vllm_default campaign run on jailbreak_goals_259 beside three vendor-preset multi-turn trials on multiturn_goals_259 on
PAIR, Crescendo and ActorAttack), and the best-of-N figure drew each SI run as a faint curve under their mean.

The rule pinned here, for every figure and table: one dot per condition per cell, the mean of the condition's runs matched to the fitted
objects' instrument (decode, judge, metric, threshold, item set, padding, cap; a declared ladder is one cap only where the fitted runs
span it); on the main figures an SI also shares the fitted objects' serving setup where such runs exist, the Refusal steer counting its
own trial views; in the appendix any serving setup counts (disclosed); where the fitted objects sit on several instruments only one
instrument is averaged; where no run matches, the closest readings plot; nothing unmatched is averaged beside the matched runs; two runs
on one standup are both averaged when their generation stores differ, and one generation filed twice is one reading, in every path;
Figure 3's padding panel is one dot per rung; and a table's first row averages the fitted arms' unsteered readings over seeds, so the
selection cannot move it, and no cell is flagged where a condition's dot reads another anchor (the owner, 2026-11-14). Synthetic
frames throughout, shaped like the Qwen static, multi-turn, capability and best-of-N cells and the Gemma and Nemotron appendix cells."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import registry_nb as R  # noqa: E402

SI = ("safety_si", "constitution_si")
RS = "arditi_allpos"
DEC = (RS,) + SI
STATIC_LADDER = json.dumps({"token_cap_ladder": [4096, 8192, 16384]})
SEED_VIEWS = (("s1", "seedrep_battery", "d1/0"), ("qwen35_4b_s2", "seedrep_cluster", "d2/2"), ("qwen35_4b_s3", "seedrep_cluster", "d3/3"))
BASE = dict(model_id=R.MODEL_ID, seed="", draw="", lane=None, dose=1.0, eval_id="pasttense", metric_id="asr", headline=True, judge_ok=True,
            item_set_id="static_pasttense_259x2", value=0.3, k=None, n=259, delta=None, ingest_seq=1, token_cap_tokens=4096, pass_label="base_cap",
            selection_state="selected", object_id="obj", run_id="r", view_legacy="v", lane_source_legacy="seedrep_cluster", judgment_id="j",
            threshold=0.5, bon_n=None, metric_role="owner", computation_id="c", ci_lo=None, ci_hi=None, anchor_run_id=None, anchor_kind=None,
            in_current_paper=1, held_out=0, decisive_action="include", comparison_ids_json="[]",
            decode_policy_id="qwen35_4b_it|static|off|vendor_preset@v1", decode_source="model_card", decode_sent_json=None,
            run_extra_json=STATIC_LADDER, serving_engine=None, escalation_state=None, two_term_dev=None, judge_id="judge-50095",
            at_operating_point=True, pad=0.0, pad_contract=False)


def row(**kw):
    r = {**BASE, **kw}
    r["k"] = round(r["value"] * r["n"]) if r.get("k") is None else r["k"]
    return r


def static_cell(eval_id="pasttense", offset=0.0, refusal=True):
    """One Qwen static cell: the unsteered anchors, a fitted object's three seeds (seed 1 at the base cap, seeds 2-3 escalated to
    8,192 - rungs of the declared ladder), each SI's three seed-standup runs at 4,096 and the campaign readings the old reader drew
    beside them (Safety SI on the instruct-direct view at 8,192, Constitution SI on that view AND the base->instruct one), and the
    Refusal steer's one campaign run, on the fitted objects' decode policy but under another decode-source label."""
    iset = f"static_{eval_id}_259x2"
    rows = []
    for v, src, dr in SEED_VIEWS:
        rows.append(row(eval_id=eval_id, item_set_id=iset, object_label="none", run_id=f"{eval_id}:none:{v}", view_legacy=v, lane_source_legacy=src, value=0.60, draw=dr))
    for i, (v, src, dr) in enumerate(SEED_VIEWS):
        rows.append(row(eval_id=eval_id, item_set_id=iset, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"lora{i}",
                        run_id=f"{eval_id}:lora:{v}", view_legacy=v, lane_source_legacy=src, value=0.40 + 0.01 * i,
                        token_cap_tokens=4096 if v == "s1" else 8192, pass_label="base_cap" if v == "s1" else "ladder_rung"))
    for a, base in (("safety_si", 0.20), ("constitution_si", 0.45)):
        for i, (v, src, dr) in enumerate(SEED_VIEWS):
            rows.append(row(eval_id=eval_id, item_set_id=iset, object_label=a, object_id=a, run_id=f"{eval_id}:{a}:{v}", view_legacy=v, lane_source_legacy=src,
                            value=base + offset + 0.02 * i, draw=dr))
    camp = dict(eval_id=eval_id, item_set_id=iset, token_cap_tokens=8192, decode_source="unverified", decode_policy_id=None,
                lane_source_legacy="sampled_lane", pass_label="pooled_t0_t1", value=0.95)
    rows.append(row(object_label="safety_si", object_id="safety_si", run_id=f"{eval_id}:safety_si:camp_it", view_legacy="qwen35_4b_it", lane="instruct_direct", **camp))
    rows.append(row(object_label="constitution_si", object_id="constitution_si", run_id=f"{eval_id}:constitution_si:camp_it", view_legacy="qwen35_4b_it", lane="instruct_direct", **camp))
    rows.append(row(object_label="constitution_si", object_id="constitution_si", run_id=f"{eval_id}:constitution_si:camp_base", view_legacy="qwen35_4b", lane="base_to_instruct", **camp))
    if refusal:
        rows.append(row(eval_id=eval_id, item_set_id=iset, object_label=RS, object_id=RS, run_id=f"{eval_id}:{RS}:camp", view_legacy="qwen35_4b_it",
                        lane="instruct_direct", lane_source_legacy="arditi_allpos", decode_source="vendor_recommendation", value=0.30 + offset, dose=0.3))
    return rows


MT_POLICY = "qwen35_4b_it|multi_turn|off|vendor_preset@v1"
MT_TRIALS = (("qwen35_4b_ar_c0p3", "d20260909/0", 0.80), ("qwen35_4b_ar_c0p3_t1", "d20260909/1", 0.82), ("qwen35_4b_ar_c0p3_t2", "d20260909/2", 0.84))


def multiturn_cell(eval_id="crescendo"):
    """One Qwen multi-turn cell: the unsteered anchors and a fitted object's three seeds on the vendor-preset multi-turn instrument
    (multiturn_goals_259), and the Refusal steer's three vendor-preset trials on that instrument beside its vllm_default campaign run on
    jailbreak_goals_259 - the cell that drew two Refusal-steer dots."""
    mt = dict(eval_id=eval_id, item_set_id="multiturn_goals_259", decode_policy_id=MT_POLICY, run_extra_json=None, n=259)
    rows = []
    for i, (v, src, dr) in enumerate(SEED_VIEWS):
        rows.append(row(object_label="none", run_id=f"{eval_id}:none:{v}", view_legacy=v, lane_source_legacy=src, value=0.95, draw=dr, **mt))
        rows.append(row(object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"lora{i}", run_id=f"{eval_id}:lora:{v}",
                        view_legacy=v, lane_source_legacy=src, value=0.70, **mt))
    rows.append(row(object_label="none", run_id=f"{eval_id}:none:camp", view_legacy="qwen35_4b_it", lane_source_legacy="arditi_allpos", value=0.97,
                    **{**mt, "item_set_id": "jailbreak_goals_259", "decode_source": "vllm_default", "decode_policy_id": None}))
    for v, dr, val in MT_TRIALS:
        rows.append(row(object_label=RS, object_id=RS, run_id=f"{eval_id}:{RS}:{v}", view_legacy=v, lane="instruct_direct", lane_source_legacy="arditi_vendor_mt",
                        draw=dr, value=val, dose=0.3, **mt))
    rows.append(row(object_label=RS, object_id=RS, run_id=f"{eval_id}:{RS}:camp", view_legacy="qwen35_4b_it", lane="instruct_direct", lane_source_legacy="arditi_allpos",
                    value=0.40, dose=0.3, **{**mt, "item_set_id": "jailbreak_goals_259", "decode_source": "vllm_default", "decode_policy_id": None}))
    return rows


def seed_runs(eval_id, arm):
    return {f"{eval_id}:{arm}:{v}" for v, _, _ in SEED_VIEWS}


def cells(d):
    return d.attrs["decode_only_cells"].set_index(["object_label", "eval_id"])


def dots(d):
    """The dots a main panel draws for the decode-only conditions: `delta_figure` hands `dodge` the pooled draws."""
    return R.pool_draws(d[d.object_label.isin(DEC) & d.d_y.notna()])


class TestOneDotPerConditionPerCell:
    def test_every_condition_is_one_dot_per_eval_in_a_mixed_frame(self):
        d = R.anchored(pd.DataFrame(static_cell("pasttense") + static_cell("codeattack", offset=0.05) + multiturn_cell("crescendo") + multiturn_cell("pair")))
        n = dots(d).groupby(["object_label", "eval_id"]).size()
        assert n.eq(1).all(), n
        assert set(n.index) == {(o, e) for o in DEC for e in ("pasttense", "codeattack")} | {(RS, "crescendo"), (RS, "pair")}

    def test_the_si_dot_is_its_three_seed_runs_and_the_campaign_readings_are_set_aside(self):
        d = R.anchored(pd.DataFrame(static_cell()))
        for a in SI:
            assert set(d[d.object_label.eq(a)].run_id) == seed_runs("pasttense", a)
            assert d[d.object_label.eq(a)].decode_only_match.eq("matched").all()
        aside = d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match.to_dict()
        assert aside == {"pasttense:safety_si:camp_it": "other instrument", "pasttense:constitution_si:camp_it": "other instrument",
                         "pasttense:constitution_si:camp_base": "other instrument"}, aside
        assert "_standup" not in d.columns

    def test_the_dot_is_the_mean_of_its_runs_with_the_pooled_interval(self):
        d = R.anchored(pd.DataFrame(static_cell()))
        mk = dots(d).set_index("object_label")
        for a in SI:
            src = d[d.object_label.eq(a)]
            c = R.cell_interval(src)                                # the cell's item-clustered interval (pooled_ci)
            assert int(mk.loc[a, "n_draws"]) == 3
            assert mk.loc[a, "d_y"] == pytest.approx(src.d_y.mean()) and mk.loc[a, "d_y"] == pytest.approx(c.dy)
            assert (mk.loc[a, "d_lo"], mk.loc[a, "d_hi"]) == (pytest.approx(c.lo), pytest.approx(c.hi))
            # a synthetic frame has no item rows: the dot takes the identical-draws bound, never narrower than one draw's interval
            assert c.method.startswith(R.BOUND_NOTE)
            assert (mk.loc[a, "d_hi"] - mk.loc[a, "d_lo"]) / 3.92 >= ((src.d_hi - src.d_lo) / 3.92).min() - 1e-9

    def test_no_field_of_a_kept_run_splits_a_decode_only_dot(self):
        # whatever lane, seed, item-set or metric spelling the kept rows carry, a decode-only condition is one dot; a fitted object keeps its key
        d = R.anchored(pd.DataFrame(static_cell()))
        d.attrs = {}
        x = d[d.object_label.eq("safety_si")]
        f = pd.concat([x.iloc[:1].assign(lane="instruct_direct"), x.iloc[1:2].assign(seed="s2"), x.iloc[2:].assign(item_set_id="other", metric_id="asr2")])
        assert len(R.pool_draws(f)) == 1
        fit = d[d.object_label.eq("control_lora")].head(1)
        assert len(R.pool_draws(pd.concat([fit, fit.assign(lane="instruct_direct")]))) == 2

    def test_the_main_figure_draws_one_dot_per_condition_per_eval(self, tmp_path, monkeypatch):
        """End to end through `delta_figure`: count the per-eval dots of each decode-only colour on the static panel."""
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        d = R.anchored(pd.DataFrame(static_cell("pasttense") + static_cell("codeattack", offset=0.05)))
        cats = {"static": (["pasttense", "codeattack"], -1, "x")}
        cm = R.category_means(d, categories=cats)
        gaps = pd.DataFrame(columns=["model_id", "object_label", "eval_id", "status"])
        fig = R.delta_figure(cm, d, [["static"]], list(DEC) + ["control_lora"], gaps, "dots_test", categories=cats)
        ax = fig.axes[0]
        for o in DEC:
            hexc = mcolors.to_hex(R.COLOR[o])
            per_eval = [c for c in ax.collections if c.get_offsets().shape[0] == 1 and len(c.get_facecolor())
                        and mcolors.to_hex(c.get_facecolor()[0]) == hexc and float(c.get_sizes()[0]) < 100]
            xs = sorted(round(float(c.get_offsets()[0][0])) for c in per_eval)
            assert xs == [0, 1], (o, xs)                                    # one dot in each of the two eval columns
        plt.close(fig)


class TestTheRefusalSteerPoolsItsTrials:
    def test_the_multi_turn_cell_is_one_dot_from_the_three_vendor_trials(self):
        d = R.anchored(pd.DataFrame(multiturn_cell()))
        got = d[d.object_label.eq(RS)]
        assert set(got.run_id) == {f"crescendo:{RS}:{v}" for v, _, _ in MT_TRIALS}, "the trials are its own views: they count as matched"
        assert got.decode_only_match.eq("matched").all()
        aside = d.attrs["decode_only_set_aside"]
        assert aside.set_index("run_id").decode_only_match.to_dict() == {f"crescendo:{RS}:camp": "other instrument"}
        mk = dots(d)
        assert len(mk) == 1 and int(mk.n_draws.iat[0]) == 3
        assert mk.y.iat[0] == pytest.approx(100 * np.mean([v for _, _, v in MT_TRIALS]))

    def test_a_refusal_steer_with_no_matched_run_still_plots_its_closest_reading(self):
        d = R.anchored(pd.DataFrame(static_cell()))
        got = d[d.object_label.eq(RS)]
        assert got.run_id.tolist() == [f"pasttense:{RS}:camp"] and got.decode_only_match.eq("fallback").all()
        c = cells(d).loc[(RS, "pasttense")]
        assert c.status == "fallback" and "decode source vendor_recommendation vs model_card" in c.why, c.why
        assert len(dots(d)[dots(d).object_label.eq(RS)]) == 1

    def test_its_category_mean_and_table_read_the_dot(self):
        d = R.anchored(pd.DataFrame(multiturn_cell("crescendo") + multiturn_cell("actorattack")))
        cats = {"multi-turn": (["crescendo", "actorattack"], -1, "x")}
        cm = R.category_means(d, categories=cats)
        mk = dots(d)
        got = cm[cm.object_label.eq(RS)]
        assert len(got) == 1 and got.y.iat[0] == pytest.approx(-mk.d_y.mean())
        se = np.sqrt((((mk.d_hi - mk.d_lo) / 3.92) ** 2).sum()) / 2
        assert (got.hi.iat[0] - got.lo.iat[0]) / 3.92 == pytest.approx(se), "the mean's interval is the dots' intervals combined"
        tex = R.table_evals_seeds(d, [["multi-turn"]], [RS, "control_lora"], "cap", "tab:t", categories=cats)
        for e in ("crescendo", "actorattack"):
            r = mk[mk.eval_id.eq(e)].iloc[0]
            lo, hi = sorted((-r.d_lo, -r.d_hi))
            assert R._cell(-r.d_y, lo, hi, signed=True) in tex, e
        assert tex.count(R._tex(R.LABEL[RS])) == 1


class TestPaddingIsPartOfTheInstrument:
    def test_a_padded_run_at_the_matched_cap_and_view_is_set_aside(self):
        rows = static_cell(refusal=False)
        rows += [row(object_label=a, object_id=a, run_id=f"pasttense:{a}:pad1024", view_legacy="s1", lane_source_legacy="seedrep_battery",
                     value=0.9, pad=1024.0, pad_contract=True) for a in SI]
        d = R.anchored(pd.DataFrame(rows))
        for a in SI:
            assert set(d[d.object_label.eq(a)].run_id) == seed_runs("pasttense", a)
        aside = d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match
        assert aside[[f"pasttense:{a}:pad1024" for a in SI]].eq("other instrument").all()
        assert dots(d).groupby("object_label").size().eq(1).all()

    def test_the_fallback_prefers_an_unpadded_reading(self):
        rows = [r for r in static_cell(refusal=False) if r["object_label"] not in SI]
        rows += [row(object_label="safety_si", object_id="s", run_id="pad", view_legacy="s1", value=0.9, pad=1024.0, pad_contract=True),
                 row(object_label="safety_si", object_id="s", run_id="other_cap", view_legacy="s1", value=0.2, token_cap_tokens=2048)]
        d = R.anchored(pd.DataFrame(rows))
        assert d[d.object_label.eq("safety_si")].run_id.tolist() == ["other_cap"]
        assert cells(d).loc[("safety_si", "pasttense"), "status"] == "fallback"


def padding_frame(pads=(0.0, 8192.0)):
    """Figure 3's padding panel at two rungs: the unsteered anchor, the difference LoRA's three seeds on the fitted view family
    (`pad_static_qwen35_4b_obj{,_s2,_s3}_p<pad>`), and each SI on that family AND on the prompted-only family (`..._p<pad>`)."""
    common = dict(eval_id="grok420", item_set_id="static_grok420_259x2", pad_contract=True, lane_source_legacy="static_padding", token_cap_tokens=16384,
                  draw="", pass_label="single_pass")
    rows = []
    for p in pads:
        P = int(p)
        rows.append(row(object_label="none", run_id=f"none:{P}", view_legacy=f"pad_static_qwen35_4b_obj_p{P}", value=0.5, pad=p, **common))
        for i, v in enumerate((f"pad_static_qwen35_4b_obj_p{P}", f"pad_static_qwen35_4b_obj_s2_p{P}", f"pad_static_qwen35_4b_obj_s3_p{P}")):
            rows.append(row(object_label="difference_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"dl{i}", run_id=f"dl:{i}:{P}",
                            view_legacy=v, value=0.1, pad=p, **common))
        for a in SI:
            rows.append(row(object_label=a, object_id=a, run_id=f"{a}:obj:{P}", view_legacy=f"pad_static_qwen35_4b_obj_p{P}", value=0.3, pad=p, **common))
            rows.append(row(object_label=a, object_id=a, run_id=f"{a}:prompted:{P}", view_legacy=f"pad_static_qwen35_4b_p{P}", value=0.6, pad=p, **common))
    return pd.DataFrame(rows)


class TestThePaddingPanelIsOneDotPerRung:
    """Figure 3's padding panel pooled every SI row at a rung, so each SI dot averaged its run on the fitted view family with its run
    on the prompted-only family, a serving setup no fitted object is on."""

    def test_each_si_dot_is_its_run_on_the_fitted_family_at_each_rung(self):
        pts = R.padding_points(padding_frame())
        si = pts[pts.object_label.isin(SI)]
        assert len(si) == 4 and si.groupby(["object_label", "pad"]).size().eq(1).all()
        assert si.k1.eq(round(0.3 * 259)).all() and si.defence.round(6).eq(round(100 * (1 - round(0.3 * 259) / 259), 6)).all(), si
        assert pts[pts.object_label.eq("difference_lora")].groupby("pad").size().eq(3).all(), "the fitted seeds are untouched"

    def test_the_prompted_only_runs_are_set_aside_per_rung_and_listed(self):
        pts = R.padding_points(padding_frame())
        aside = pts.attrs["decode_only_set_aside"]
        assert sorted(zip(aside.run_id, aside.pad)) == sorted((f"{a}:prompted:{int(p)}", p) for a in SI for p in (0.0, 8192.0))
        assert aside.decode_only_match.eq("other serving setup").all()
        c = pts.attrs["decode_only_cells"]
        assert len(c) == 4 and c.status.eq("matched").all() and c.n_runs.eq(1).all() and sorted(set(c.pad)) == [0.0, 8192.0]
        text = R.decode_only_disclosure(c, aside, title="Figure 3, padding panel")
        assert "set-aside readings (4; never drawn)" in text and "Pliny · Safety SI at 8,192 filler tokens" in text
        assert "pad_static_qwen35_4b_p8192" in text


class TestThePerInstrumentChoice:
    def _petri(self, si_rows):
        m = "llama8b_it"
        common = dict(model_id=m, eval_id="petri_defer_to_users", metric_id="petri_score", item_set_id="petri_scenarios_6x10", threshold=None,
                      judge_id="judge-17e4", decode_source="generation_config", token_cap_tokens=1024, run_extra_json=None, draw="", n=60, value=7.0,
                      view_legacy="llama8b")
        spelled = "llama8b_it|petri|not_available|generation_config@v1"
        rows = [row(object_label="none", run_id="none", decode_policy_id=spelled, **common)]
        rows += [row(object_label=o, object_id=o, run_id=o, decode_policy_id=spelled, **common) for o in ("control_lora", "constitution_lora", "difference_lora")]
        rows += [row(object_label=o, object_id=o, run_id=o, decode_policy_id=None, **common) for o in ("control_relu", "constitution_relu")]
        rows += [row(object_label="safety_si", object_id="s", run_id=rid, decode_policy_id=(spelled if pol == "spelled" else None), **{**common, "value": v})
                 for rid, pol, v in si_rows]
        return pd.DataFrame(rows)

    def test_one_dot_reads_one_instrument_the_one_most_fitted_objects_use(self):
        """Llama's Petri cell: the LoRAs file the decode policy spelled, the ReLU steers the same runs' older filing without it, and the
        Safety SI has one filing under each - two instruments of the cell. The dot reads one, never both."""
        f = self._petri([("s:spelled", "spelled", 7.7), ("s:older", "older", 5.1)])
        for setup in ("main", "any"):
            kept, aside, cl = R.select_decode_only(f, setup=setup)
            assert kept[kept.object_label.eq("safety_si")].run_id.tolist() == ["s:spelled"]
            assert aside.set_index("run_id").decode_only_match.to_dict() == {"s:older": "second instrument"}
            c = cl.set_index("object_label").loc["safety_si"]
            assert int(c.fitted_instruments) == 2 and int(c.instrument_rank) == 1
        d = R.anchored(f, setup="any")
        assert d[d.object_label.eq("safety_si")].y.tolist() == [pytest.approx(7.7)]

    def test_a_condition_matching_only_the_minority_instrument_reads_it_and_says_so(self):
        f = self._petri([("s:older", "older", 5.1)])
        kept, aside, cl = R.select_decode_only(f)
        c = cl.set_index("object_label").loc["safety_si"]
        assert c.status == "matched" and int(c.instrument_rank) == 2 and "instrument 2 of 2" in c.why
        assert "instrument 2 of 2" in R.decode_only_disclosure(cl, aside)


class TestServingSetup:
    def _gsm8k(self):
        """A Qwen GSM8K cell: the fitted seeds and the Safety SI's seed runs on the seed views, plus one SI run on the campaign view on the
        same decode at a rung of the same declared ladder - the fitted objects' instrument on another serving setup."""
        lad = json.dumps({"token_cap_ladder": [8192, 16384]})
        common = dict(eval_id="gsm8k", metric_id="acc", item_set_id="gsm8k_platinum_paired_200", decode_policy_id="qwen35_4b_it|gsm8k|on|thinking_card@v1",
                      judge_id=None, threshold=None, run_extra_json=lad, n=200)
        rows = []
        for i, (v, src, dr) in enumerate(SEED_VIEWS):
            rows.append(row(object_label="none", run_id=f"none:{v}", view_legacy=v, value=0.9, draw=dr, token_cap_tokens=16384, **common))
            rows.append(row(object_label="control_lora", seed=f"s{i + 1}", object_id=f"l{i}", run_id=f"lora:{v}", view_legacy=v, value=0.88,
                            token_cap_tokens=8192 if i else 16384, **common))
            rows.append(row(object_label="safety_si", object_id="s", run_id=f"s:{v}", view_legacy=v, value=0.85 + 0.01 * i, draw=dr, token_cap_tokens=16384, **common))
        rows.append(row(object_label="safety_si", object_id="s", run_id="s:camp", view_legacy="qwen35_4b_it", value=0.70, draw="d-1/0", token_cap_tokens=8192, **common))
        return pd.DataFrame(rows)

    def test_qwen_main_figures_keep_the_si_on_the_fitted_serving_setup(self):
        d = R.anchored(self._gsm8k())
        assert set(d[d.object_label.eq("safety_si")].run_id) == {"s:s1", "s:qwen35_4b_s2", "s:qwen35_4b_s3"}
        assert d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match.to_dict() == {"s:camp": "other serving setup"}

    def test_the_appendix_counts_another_serving_setup_as_matched_and_discloses_it(self):
        d = R.anchored(self._gsm8k(), setup="any")
        assert set(d[d.object_label.eq("safety_si")].run_id) == {"s:s1", "s:qwen35_4b_s2", "s:qwen35_4b_s3", "s:camp"}
        c = cells(d).loc[("safety_si", "gsm8k")]
        assert c.status == "matched, cross-setup" and int(c.cross_setup) == 1
        text = R.decode_only_disclosure(d.attrs["decode_only_cells"], d.attrs["decode_only_set_aside"])
        assert "cross-setup matches (1;" in text and "1 of 4 run(s)" in text

    def test_a_ladder_counts_as_one_cap_only_where_the_fitted_runs_span_it(self):
        f = self._gsm8k()
        f.loc[f.object_label.eq("control_lora"), "token_cap_tokens"] = 16384       # every fitted run at one rung: the ladder is not spanned
        d = R.anchored(f, setup="any")
        assert "s:camp" not in set(d[d.object_label.eq("safety_si")].run_id)
        assert d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match["s:camp"] == "other instrument"


GEMMA = "gemma4e4b_it"
GEMMA_CAMP = dict(model_id=GEMMA, token_cap_tokens=8192, decode_source="unverified", decode_policy_id=None, lane_source_legacy="sampled_lane",
                  pass_label="pooled_t0_t1", draw="", run_extra_json=None)


def gemma_cell():
    """A Gemma appendix cell: the fitted objects on the base->instruct view (gemma4e4b); the Constitution SI served there and on the
    instruct-direct view, the Safety SI on the instruct-direct view only, plus a greedy 512-token reading on another instrument."""
    return [row(object_label="none", run_id="none", view_legacy="gemma4e4b", lane="base_to_instruct", value=0.6, **GEMMA_CAMP),
            row(object_label="control_lora", object_id="lora", run_id="lora", view_legacy="gemma4e4b", lane="base_to_instruct", value=0.4, **GEMMA_CAMP),
            row(object_label="constitution_si", object_id="c", run_id="c:base", view_legacy="gemma4e4b", value=0.30, **GEMMA_CAMP),
            row(object_label="constitution_si", object_id="c", run_id="c:it", view_legacy="gemma4e4b_it", value=0.34, **GEMMA_CAMP),
            row(object_label="safety_si", object_id="s", run_id="s:it", view_legacy="gemma4e4b_it", value=0.20, **GEMMA_CAMP),
            row(object_label="safety_si", object_id="s", run_id="s:greedy", view_legacy="gemma4e4b_it", value=0.90,
                **{**GEMMA_CAMP, "decode_source": "greedy", "token_cap_tokens": 512})]


class TestTheAppendix:
    def test_a_run_from_another_serving_setup_counts_as_matched(self):
        d = R.anchored(pd.DataFrame(gemma_cell()), setup="any")
        assert set(d[d.object_label.eq("constitution_si")].run_id) == {"c:base", "c:it"}, "both setups average into the one dot"
        assert d[d.object_label.eq("safety_si")].run_id.tolist() == ["s:it"]
        assert d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match.to_dict() == {"s:greedy": "other instrument"}
        c = cells(d)
        assert c.loc[("safety_si", "pasttense"), "status"] == "matched, cross-setup"
        assert c.loc[("constitution_si", "pasttense"), "status"] == "matched, cross-setup" and int(c.loc[("constitution_si", "pasttense"), "cross_setup"]) == 1

    def test_the_absolute_points_read_the_same_runs_and_draw_one_dot_per_model(self):
        pts = R.headline_points(pd.DataFrame(gemma_cell()), ["pasttense"], setup="any")
        assert set(pts[pts.object_label.eq("constitution_si")].run_id) == {"c:base", "c:it"}
        assert set(pts[pts.object_label.eq("safety_si")].run_id) == {"s:it"}
        pooled = R.pool_cell(pts[pts.object_label.isin(SI)])
        assert pooled.groupby("object_label").size().eq(1).all()
        assert pooled.set_index("object_label").loc["constitution_si", "y"] == pytest.approx(32.0)

    def test_two_selections_share_one_figure(self):
        """Over-refusal and capability share one appendix figure: two `headline_points` frames, each carrying its disclosure in attrs,
        are concatenated for the panels (pd.concat compares the inputs' attrs and a frame-valued attr raises) and their disclosures
        merged for the lines under the figure."""
        a = R.headline_points(pd.DataFrame(gemma_cell()), ["pasttense"], setup="any")
        b = R.headline_points(pd.DataFrame(gemma_cell()).assign(eval_id="codeattack"), ["codeattack"], setup="any")
        both = R.concat_plain(a, b)
        assert len(both) == len(a) + len(b) and not both.attrs and len(a.attrs["decode_only_cells"])
        merged = R.concat_selected(a, b)
        assert set(merged.attrs["decode_only_cells"].eval_id) == {"pasttense", "codeattack"}
        assert set(merged.attrs["decode_only_set_aside"].run_id) == {"s:greedy"}
        assert merged.attrs["decode_only_set_aside"].dose.eq("op").all(), "an aside reading names its cell, as the cells frame does"

    def test_where_no_run_matches_the_closest_readings_plot(self):
        """Nemotron's Constitution SI best-of-N: read at cap 8,192 only, while the fitted objects sit at 4,096."""
        m = "nemotron3nano4b"
        common = dict(model_id=m, eval_id="strongreject_bon", metric_id="asr_bon_cap4096", item_set_id="jbb_goals_100", decode_source="grandfathered_T1",
                      decode_policy_id=None, run_extra_json=None, n=100, draw="", view_legacy="nemotron3nano4b", judge_id="judge-c10c", lane="chat_checkpoint")
        rows = [row(object_label="none", run_id="none", value=0.7, **common),
                row(object_label="control_lora", object_id="lora", run_id="lora", value=0.6, **common),
                row(object_label="constitution_si", object_id="c", run_id="c:8192", value=0.5, **{**common, "token_cap_tokens": 8192}),
                row(object_label="constitution_si", object_id="c", run_id="c:greedy512", value=0.9,
                    **{**common, "token_cap_tokens": 512, "decode_source": "greedy", "item_set_id": "other_goals"})]
        d = R.anchored(pd.DataFrame(rows), setup="any")
        got = d[d.object_label.eq("constitution_si")]
        assert got.run_id.tolist() == ["c:8192"] and got.decode_only_match.eq("fallback").all()
        c = cells(d).loc[("constitution_si", "strongreject_bon")]
        assert c.status == "fallback" and "cap 8192 vs 4096" in c.why, c.why
        aside = d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match.to_dict()
        assert aside == {"c:greedy512": "fallback, not closest"}
        text = R.decode_only_disclosure(d.attrs["decode_only_cells"], d.attrs["decode_only_set_aside"])
        assert "fallback dots (1;" in text and "Nemotron-3-Nano-4B · Best-of-75 · Constitution SI" in text and "cap 8192 vs 4096" in text
        assert "fallback, not closest" in text

    def test_a_cell_with_no_fitted_object_still_plots_one_dot(self):
        rows = [r for r in gemma_cell() if r["object_label"] != "control_lora"]
        d = R.anchored(pd.DataFrame(rows), setup="any")
        assert len(d[d.object_label.eq("safety_si")]) == 1
        assert cells(d).loc[("safety_si", "pasttense"), "status"] == "fallback, no fitted object"
        assert R.pool_cell(d[d.object_label.isin(SI)]).groupby("object_label").size().eq(1).all()


def two_run_cell(stores, values, eval_id="actorattack", deltas=(None, None)):
    """A Llama appendix cell: the fitted object on the instruct view, and the Constitution SI with two runs on one other view at one cap,
    decode, judge and item set - two grading waves (two generation stores), or one generation filed twice (one store, or none recorded)."""
    bon = eval_id == R.BON_EVAL
    common = dict(model_id="llama8b_it", eval_id=eval_id, item_set_id="jbb_goals_100" if bon else "multiturn_goals_259",
                  metric_id="asr_bon_cap4096" if bon else "asr", bon_n=75 if bon else None, n=100 if bon else 259, decode_source="generation_config",
                  decode_policy_id=None, run_extra_json=None, draw="", pass_label=None, lane="instruct_direct")
    rows = [row(object_label="none", run_id="none", view_legacy="llama8b_it", lane_source_legacy="r2_export", value=0.9, **common),
            row(object_label="control_lora", object_id="lora", run_id="lora", view_legacy="llama8b_it", lane_source_legacy="r2_export", value=0.7, **common)]
    for i, (st, v, dl) in enumerate(zip(stores, values, deltas)):
        rows.append(row(object_label="constitution_si", object_id="c", run_id=f"c:{i}", view_legacy="llama8b_shortconst", value=v, store_uri=st,
                        ingest_seq=10 + i, delta=dl, lane_source_legacy=("multiturn_grades", "r2_export")[i], **common))
    return pd.DataFrame(rows)


class TestOneReadingPerIndependentGeneration:
    """Two matched runs of one condition on one standup were one cell key, so `anchored` kept one (the one with a contrast) and
    `headline_points` another, and neither drew their mean (Llama's Constitution SI on PAIR, Crescendo and ActorAttack)."""

    def test_two_generations_on_one_standup_are_averaged_in_both_paths(self):
        f = two_run_cell(("sha256:aaa", "sha256:bbb"), (0.78, 0.80), deltas=(None, -9.0))
        d = R.anchored(f, setup="any")
        assert set(d[d.object_label.eq("constitution_si")].run_id) == {"c:0", "c:1"}
        mk = dots(d)
        assert len(mk) == 1 and int(mk.n_draws.iat[0]) == 2 and mk.y.iat[0] == pytest.approx(79.0)
        assert int(cells(d).loc[("constitution_si", "actorattack"), "n_runs"]) == 2
        hp = R.headline_points(f, ["actorattack"], setup="any")
        assert set(hp[hp.object_label.eq("constitution_si")].run_id) == {"c:0", "c:1"}
        assert R.pool_cell(hp[hp.object_label.eq("constitution_si")]).y.tolist() == [pytest.approx(79.0)]
        text = R.decode_only_disclosure_pair(hp, d, ["actorattack"], ["constitution_si"], "abs", "delta")
        assert "(the same dots)" in text and "2 of 2 run(s) on llama8b_shortconst" in text

    @pytest.mark.parametrize("stores", [("sha256:aaa", None), ("sha256:aaa", "sha256:aaa"), (None, None)])
    def test_one_generation_filed_twice_is_one_reading_and_every_path_keeps_the_same_run(self, stores):
        f = two_run_cell(stores, (0.38, 0.38), eval_id=R.BON_EVAL, deltas=(None, -9.0))
        d = R.anchored(f, setup="any")
        assert d[d.object_label.eq("constitution_si")].run_id.tolist() == ["c:1"], "the run with a registry contrast"
        assert d.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match.to_dict() == {"c:0": "same measurement"}
        hp = R.headline_points(f, [R.BON_EVAL], setup="any")
        assert hp[hp.object_label.eq("constitution_si")].run_id.tolist() == ["c:1"]
        assert "same measurement" in set(hp.attrs["decode_only_set_aside"].decode_only_match)
        long = pd.DataFrame([dict(run_id=r, eval_id=R.BON_EVAL, metric_id=f"asr_at_n{n}", threshold=0.5, value=0.38 * n / 75)
                             for r in ("c:0", "c:1", "lora") for n in (1, 4, 16, 64)])
        bp = R.bon_curve_points(f, long, setup="any")
        assert set(bp[bp.object_label.eq("constitution_si")].run_id) == {"c:1"}, "the best-of-N figure draws the run the dot reads"
        assert set(bp.attrs["decode_only_set_aside"].run_id) == {"c:0"}

    def test_a_base_pass_beside_its_escalated_rung_stays_one_reading(self):
        f = two_run_cell(("sha256:aaa", "sha256:bbb"), (0.70, 0.78))
        f.loc[f.run_id.eq("c:0"), "pass_label"] = "base_cap"
        f.loc[f.run_id.eq("c:1"), "pass_label"] = "ladder_rung"
        d = R.anchored(f, setup="any")
        assert d[d.object_label.eq("constitution_si")].run_id.tolist() == ["c:1"]


class TestTheGenerationKeyIsTheSameLiveAndFromTheFrame:
    """The live registry records a run's store as a raw path, which two readers of ONE generation spell differently (Llama's Constitution
    SI best-of-N cell: the multi-turn grades' directory and the r2 export's stats file); the frame's export writes `sha256:<the store's
    completion hash>` where the run records one and nothing where it does not. Keyed on the path, a live render counted that generation
    twice (two draws, a narrower interval) where the frame counts it once. The key is now the completion hash in both
    (`generation_store`), so a live render and a frame render keep the same runs."""

    def _live_and_frame(self, hashes, paths):
        live = two_run_cell(paths, (0.38, 0.38), eval_id=R.BON_EVAL, deltas=(None, -9.0))
        live["store_completion_sha256"] = None
        frame = live.copy()
        for i, h in enumerate(hashes):
            live.loc[live.run_id.eq(f"c:{i}"), "store_completion_sha256"] = h
            frame.loc[frame.run_id.eq(f"c:{i}"), "store_completion_sha256"] = h
            frame.loc[frame.run_id.eq(f"c:{i}"), "store_uri"] = f"sha256:{h}" if h else None      # what the export's scrub writes
        return live, frame

    def test_one_generation_under_two_raw_paths_is_one_reading_live_as_in_the_frame(self):
        live, frame = self._live_and_frame(("5357cbbc", None), ("/grades/llama8b_shortconst", "/stats/transfer_stats_llama8b.json"))
        assert R.generation_store(live).tolist() == R.generation_store(frame).tolist()
        for f in (live, frame):
            d = R.anchored(f, setup="any")
            assert d[d.object_label.eq("constitution_si")].run_id.tolist() == ["c:1"], "one reading, the run with a registry contrast"
            hp = R.headline_points(f, [R.BON_EVAL], setup="any")
            assert hp[hp.object_label.eq("constitution_si")].run_id.tolist() == ["c:1"]

    def test_two_generations_with_their_own_hashes_stay_two_readings_live_as_in_the_frame(self):
        live, frame = self._live_and_frame(("aaaa1111", "bbbb2222"), ("/grades/wave1", "/grades/wave2"))
        for f in (live, frame):
            d = R.anchored(f, setup="any")
            assert set(d[d.object_label.eq("constitution_si")].run_id) == {"c:0", "c:1"}

    def test_a_raw_path_is_never_a_key(self):
        f = pd.DataFrame(dict(store_uri=["/a/x", "sha256:abc", None, "/b/y"], store_completion_sha256=[None, None, None, "def"]))
        assert R.generation_store(f).tolist() == ["", "sha256:abc", "", "sha256:def"]
        assert R.generation_store(f.drop(columns="store_completion_sha256")).tolist() == ["", "sha256:abc", "", ""]


class TestTheBestOfNCurves:
    """The best-of-N figure read every admitted best-of-N run, so an SI drew its campaign reading beside its seed runs, and drew each seed run as
    a faint line under their mean."""

    def _frames(self):
        e = R.BON_EVAL
        common = dict(eval_id=e, metric_id="asr_bon_cap4096", item_set_id="jbb_goals_100", judge_id="judge-c10c", threshold=0.5, n=100, bon_n=75,
                      decode_policy_id="qwen35_4b_it|bon|off|vendor_preset@v1", decode_source="model_card", run_extra_json=None, draw="")
        op, long = [], []
        for i, (v, src, _dr) in enumerate((("qwen35_4b_s1",) + SEED_VIEWS[0][1:], SEED_VIEWS[1], SEED_VIEWS[2])):
            op.append(row(object_label="control_lora", seed=f"s{i + 1}", object_id=f"lora{i}", run_id=f"lora:{v}", view_legacy=v, lane_source_legacy=src,
                          value=0.5, **common))
            for a in SI:
                op.append(row(object_label=a, object_id=a, run_id=f"{a}:{v}", view_legacy=v, lane_source_legacy=src, value=0.3 + 0.05 * i, **common))
        for a in SI:
            op.append(row(object_label=a, object_id=a, run_id=f"{a}:camp", view_legacy="qwen35_4b_it", lane_source_legacy="bon_stores", value=0.9,
                          **{**common, "token_cap_tokens": 8192, "decode_source": "grandfathered_T1", "decode_policy_id": None}))
        for r in op:
            for n in (1, 4, 16, 64):
                long.append(dict(run_id=r["run_id"], eval_id=e, metric_id=f"asr_at_n{n}", threshold=0.5, value=r["value"] * n / 75))
        return pd.DataFrame(op), pd.DataFrame(long)

    def test_an_si_is_one_curve_from_the_runs_of_its_dot(self, tmp_path, monkeypatch):
        op, long = self._frames()
        pts = R.bon_curve_points(op, long, setup="any")
        for a in SI:
            assert set(pts[pts.object_label.eq(a)].run_id) == {f"{a}:{v}" for v in ("qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3")}
        assert set(pts.attrs["decode_only_set_aside"].run_id) == {f"{a}:camp" for a in SI}
        kept, _dropped = R.bon_curve_select(pts)
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        fig = R.bon_curves_figure(kept, ["control_lora", *SI], [R.MODEL_ID], "bon_test")
        colors = [mcolors.to_hex(ln.get_color()) for ln in fig.axes[0].lines]
        for a in SI:
            assert colors.count(mcolors.to_hex(R.COLOR[a])) == 1, (a, colors)
        assert colors.count(mcolors.to_hex(R.COLOR["control_lora"])) == 4, "a fitted object keeps its seeds as faint lines under the mean"
        mean = [ln for ln in fig.axes[0].lines if mcolors.to_hex(ln.get_color()) == mcolors.to_hex(R.COLOR["safety_si"])][0]
        assert mean.get_ydata()[-1] == pytest.approx(100 * (1 - 0.35))
        plt.close(fig)

    def test_the_curves_disclosure_counts_the_runs_a_curve_draws(self):
        """One store filed by two readers is one reading: the selection collapses it as the panels' best-of-N dot does
        (`collapse_decode_only`) and lists the other filing as set aside, so the line under the best-of-N figure counts the runs the curve averages
        (the curve rules then find no duplicate left to drop)."""
        op, long = self._frames()
        dup = op[op.run_id.eq("safety_si:qwen35_4b_s1")].assign(run_id="safety_si:qwen35_4b_s1:dup")
        op = pd.concat([op, dup], ignore_index=True)
        long = pd.concat([long, long[long.run_id.eq("safety_si:qwen35_4b_s1")].assign(run_id="safety_si:qwen35_4b_s1:dup")], ignore_index=True)
        pts = R.bon_curve_points(op, long, setup="any")
        pair = {"safety_si:qwen35_4b_s1", "safety_si:qwen35_4b_s1:dup"}
        assert len(pair & set(pts.run_id)) == 1
        aside = pts.attrs["decode_only_set_aside"].set_index("run_id").decode_only_match
        assert [aside[r] for r in pair - set(pts.run_id)] == ["same measurement"]
        assert int(pts.attrs["decode_only_cells"].set_index("object_label").loc["safety_si", "n_runs"]) == 3
        kept, dropped = R.bon_curve_select(pts)
        assert dropped.empty, dropped
        c = kept.attrs["decode_only_cells"].set_index("object_label")
        assert int(c.loc["safety_si", "n_runs"]) == 3 and int(c.loc["constitution_si", "n_runs"]) == 3
        assert "runs averaged per dot: 3 runs x 2" in R.decode_only_disclosure(kept.attrs["decode_only_cells"], kept.attrs["decode_only_set_aside"])


class TestTheFrontierDiamondIsTheDot:
    """Figure 7 keyed an SI's points by lane and drew their mean, so the campaign readings at 8,192 were averaged into the diamond."""

    def _frame(self):
        rows = []
        evals = [(e, f"static_{e}_259x2", "asr", "judge-50095", STATIC_LADDER) for e in R.FRONTIER_STATIC[1]] + \
                [("xstest", "xstest_safe_250", "over_refusal_label_mean", "judge-or", None),
                 ("or_bench_hard", "or_bench_hard_1319", "over_refusal_label_mean", "judge-or", None)]
        for e, iset, met, judge, lad in evals:
            c = dict(eval_id=e, item_set_id=iset, metric_id=met, judge_id=judge, run_extra_json=lad, lifecycle_state="ingested",
                     coefficient=None, decode_policy_id="vp", draw="")
            for i, (v, src, _dr) in enumerate(SEED_VIEWS):
                rows.append(row(object_label="none", run_id=f"{e}:none:{v}", view_legacy=v, lane_source_legacy=src, value=0.60, **c))
                rows.append(row(object_label="difference_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"dl{i}", run_id=f"{e}:dl:{v}",
                                view_legacy=v, lane_source_legacy=src, value=0.30, **c))
                for a, base in (("safety_si", 0.20), ("constitution_si", 0.45)):
                    rows.append(row(object_label=a, object_id=a, run_id=f"{e}:{a}:{v}", view_legacy=v, lane_source_legacy=src, value=base + 0.02 * i, **c))
            camp = {**c, "token_cap_tokens": 8192, "decode_source": "unverified", "decode_policy_id": None, "lane_source_legacy": "sampled_lane", "value": 0.95}
            rows.append(row(object_label="safety_si", object_id="safety_si", run_id=f"{e}:safety_si:it", view_legacy="qwen35_4b_it", lane="instruct_direct", **camp))
            rows.append(row(object_label="constitution_si", object_id="constitution_si", run_id=f"{e}:constitution_si:it", view_legacy="qwen35_4b_it",
                            lane="instruct_direct", **camp))
            rows.append(row(object_label="constitution_si", object_id="constitution_si", run_id=f"{e}:constitution_si:base", view_legacy="qwen35_4b",
                            lane="base_to_instruct", **camp))
        return pd.DataFrame(rows)

    def test_one_point_per_si_from_its_three_seed_runs(self):
        pts, _held, meta = R.frontier_points(self._frame(), R.FRONTIER_LANE, R.FRONTIER_STATIC[1], pull_lane=R.FRONTIER_PULL_LANE,
                                             objects=R.FRONTIER_OBJECTS, panel_name="static")
        si = pts[pts.object_label.isin(SI)].set_index("object_label")
        assert sorted(si.index) == sorted(SI) and si.lane.eq("prompt (no lane)").all()
        assert si.loc["safety_si", "dy"] == pytest.approx(38.0) and si.loc["safety_si", "dx"] == pytest.approx(38.0)
        assert si.loc["constitution_si", "dy"] == pytest.approx(13.0)
        aside = meta["decode_only_set_aside"].groupby(["object_label", "view_legacy"]).run_id.nunique().to_dict()
        assert aside == {("constitution_si", "qwen35_4b"): 6, ("constitution_si", "qwen35_4b_it"): 6, ("safety_si", "qwen35_4b_it"): 6}
        assert meta["decode_only_cells"].status.eq("matched").all()
        assert float(pts.set_index("object_label").loc["difference_lora", "dy"]) == pytest.approx(30.0), "a fitted ladder is untouched"

    def test_a_seed_rung_carries_its_own_unsteered_levels_and_its_levels_after(self):
        """The frontier table's first row averages the rungs' own levels over seeds, and a rung's value after is its own level plus its
        change: each per-seed rung and pooled point carries its own levels (100 - its anchors: defense on the battery,
        compliance on the benign pair) and its levels after (100 - its readings), averaged as its dy / dx are; nothing draws them."""
        pts, _held, meta = R.frontier_points(self._frame(), R.FRONTIER_LANE, R.FRONTIER_STATIC[1], pull_lane=R.FRONTIER_PULL_LANE,
                                             objects=R.FRONTIER_OBJECTS, panel_name="static")
        ps = meta["per_seed"]
        assert {"ay_abs", "ax_abs", "ay_after", "ax_after"} <= set(ps.columns) and {"ay_after", "ax_after"} <= set(pts.columns)
        dl = ps[ps.object_label.eq("difference_lora")]
        assert len(dl) == 3 and dl.ay_abs.round(6).eq(40.0).all() and dl.ax_abs.round(6).eq(40.0).all(), dl[["seed", "ay_abs", "ax_abs"]]
        assert dl.ay_after.round(6).eq(70.0).all() and dl.ax_after.round(6).eq(70.0).all(), "100 - the 30 % readings"
        for f in (ps, pts):
            assert np.allclose(f.ay_abs + f.dy, f.ay_after) and np.allclose(f.ax_abs + f.dx, f.ax_after), f[["object_label", "dy", "ay_abs", "ay_after"]]


def first_row(tex):
    return next(ln for ln in tex.splitlines() if R.BASELINE_ROW_LABEL in ln)


def keep_every_reading(monkeypatch):
    """No selection at all: every reading of a decode-only condition is drawn (the frame as it stood before the one-dot rule)."""
    def keep_all(frame, fitted=None, setup="main"):
        f = frame.assign(decode_only_match="")
        f.attrs = {}
        return f, f.iloc[0:0], pd.DataFrame(columns=R.DECODE_ONLY_CELL_COLS)
    monkeypatch.setattr(R, "select_decode_only", keep_all)


class TestTheFirstRowIsTheFittedArmsSeedAverage:
    """A table's first row ('unsteered (absolute)') is the unsteered level averaged over seeds (`seed_average`, the owner,
    2026-11-14): the mean over the fit seeds of the unsteered readings each seed's arms were differenced against, which a seedless
    condition's dot never enters while the column has seed replicates. A rule of 2026-11-09 required that the decode-only selection
    never move the first row; the 2026-11-09 mean over every arm's anchors moved with it, so the tables read every decode-only reading
    as it stood before the selection (`baseline_rows`). There is nothing left for that to protect: the fitted arms' seed average holds
    the row whichever runs a dot keeps, and no cell is flagged for the dot's own anchor."""

    CATS = {"multi-turn": (["crescendo"], -1, "x")}

    def test_the_per_eval_and_absolute_tables_first_row_is_the_fitted_arms_anchor_whatever_the_selection(self, monkeypatch):
        f = pd.DataFrame(multiturn_cell())
        d = R.anchored(f)
        t = R.table_evals_seeds(d, [["multi-turn"]], [RS, "control_lora"], "cap", "tab:t", categories=self.CATS)
        a = R.table_absolute(d, ["crescendo"], ["none", RS, "control_lora"], [R.MODEL_ID], "cap", "tab:a")
        keep_every_reading(monkeypatch)
        d0 = R.anchored(f)
        assert set(d0.run_id) - set(d.run_id) == {f"crescendo:{RS}:camp"}
        fitted = float(d[d.object_label.eq("control_lora")].anchor_value.mean())
        rs0 = float(d0[d0.object_label.eq(RS)].anchor_value.mean())
        assert abs(rs0 - fitted) > 0.05, "the set-aside reading has its own anchor: averaging it in moved the old first row"
        t0 = R.table_evals_seeds(d0, [["multi-turn"]], [RS, "control_lora"], "cap", "tab:t", categories=self.CATS)
        a0 = R.table_absolute(d0, ["crescendo"], ["none", RS, "control_lora"], [R.MODEL_ID], "cap", "tab:a")
        for tex in (t, a, t0, a0):
            assert R._cell(100.0 - fitted) in first_row(tex), first_row(tex)              # as defense success, 100 - ASR
        assert first_row(t) == first_row(t0) and first_row(a) == first_row(a0)
        rs_cell = next(ln for ln in t0.splitlines() if ln.startswith("Refusal steer"))
        assert "vs " not in rs_cell and all("scriptsize vs" not in tex for tex in (t, a, t0, a0)), "no cell is flagged for its own anchor"
        assert not hasattr(R, "baseline_frame") and "baseline_rows" not in d.attrs

    def test_a_frame_rebuilt_by_concatenation_needs_no_attrs(self):
        """The control-pool multi-turn figure's table reads a frame rebuilt from Figure 4's rows plus the refit slots' rows, whose attrs
        are empty, and pandas drops attrs that differ across a concatenation; the first row no longer reads any, so the rebuilt frame's
        table is Figure 4's, and the notebook cell carries none."""
        d = R.anchored(pd.DataFrame(multiturn_cell()))
        new = d[d.object_label.eq("control_lora")].copy()
        new.attrs = {}
        rebuilt = pd.concat([d[d.object_label.ne("control_lora")], new], ignore_index=True)
        table = lambda f: R.table_evals_seeds(f, [["multi-turn"]], [RS, "control_lora"], "cap", "tab:t", categories=self.CATS)
        assert table(rebuilt) == table(d)
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        cell = src[src.index('code("7fae15d6"'):src.index('md("de9a82d2"')]
        assert "baseline_rows" not in cell

    def test_the_category_tables_first_row_is_the_fitted_arms_anchor_whatever_the_selection(self, monkeypatch):
        f = pd.DataFrame(multiturn_cell())
        d = R.anchored(f)
        tc = R.table_categories_models(R.category_means(d, categories=self.CATS), "multi-turn", [RS, "control_lora"], [R.MODEL_ID], "cap", "tab:c",
                                       categories=self.CATS)
        keep_every_reading(monkeypatch)
        cm0 = R.category_means(R.anchored(f), categories=self.CATS)
        tc0 = R.table_categories_models(cm0, "multi-turn", [RS, "control_lora"], [R.MODEL_ID], "cap", "tab:c", categories=self.CATS)
        assert R._cell(100.0 - 95.0) in first_row(tc) and first_row(tc) == first_row(tc0), (first_row(tc), first_row(tc0))

    def test_the_frontier_tables_first_row_is_the_fitted_rungs_seed_average_whatever_the_selection(self):
        f = TestTheFrontierDiamondIsTheDot()._frame()
        # the campaign readings at 8,192 have an unsteered run of their own on the benign evals, so their origin differs
        extra = [row(object_label="none", run_id=f"{e}:none:8192", view_legacy="qwen35_4b_it", lane_source_legacy="sampled_lane", value=0.80,
                     eval_id=e, item_set_id=iset, metric_id="over_refusal_label_mean", judge_id="judge-or", run_extra_json=None,
                     lifecycle_state="ingested", coefficient=None, token_cap_tokens=8192, decode_source="unverified", decode_policy_id=None, draw="")
                 for e, iset in (("xstest", "xstest_safe_250"), ("or_bench_hard", "or_bench_hard_1319"))]
        f = pd.concat([f, pd.DataFrame(extra)], ignore_index=True)
        kw = dict(pull_lane=R.FRONTIER_PULL_LANE, objects=R.FRONTIER_OBJECTS, panel_name="static")
        cells_ = lambda tex: [c.strip().rstrip("\\").strip() for c in first_row(tex).split("&")]
        for select in (True, False):
            pts, _held, meta = R.frontier_points(f, R.FRONTIER_LANE, R.FRONTIER_STATIC[1], decode_select=select, **kw)
            tex = R.table_frontier([dict(title="static", pts=pts, per_seed=meta.get("per_seed"))], R.FRONTIER_OBJECTS, "{panel}", "tab:f")
            # every fitted rung's own origin is 40 % defense and 40 % compliance; before the selection the SIs' campaign points read 20 %
            # compliance against their own 8,192 run, which the seed average over the fitted ladders never reads
            assert cells_(tex)[3:5] == [R._cell(40.0), R._cell(40.0)], (select, cells_(tex))
            si = [ln for ln in tex.splitlines() if ln.startswith(("Safety SI", "Constitution SI"))]
            assert si and not any("vs " in ln for ln in si), (select, si)


class TestWhatTheSelectionKeepsFromBefore:
    """Behaviour the first matched-runs change introduced that the unified rule keeps."""

    def test_identical_draw_labels_do_not_collapse_three_seed_runs(self):
        """BigCodeBench / BFCL / MMLU: the SIs' seed runs carry no seed base and trial 0, so one draw label; three standups are three
        runs of the one dot, not one measurement selected three times."""
        rows = []
        for i, v in enumerate(("qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3")):
            common = dict(eval_id="bigcodebench", metric_id="acc", item_set_id="bigcodebench_full_1140", lane_source_legacy="bigcodebench_full",
                          decode_source="greedy", decode_policy_id=None, judge_id=None, threshold=None, run_extra_json=None, view_legacy=v,
                          token_cap_tokens=4096, pass_label=None, draw="", n=1140)
            rows.append(row(object_label="none", run_id=f"none:{v}", value=0.40, **common))
            rows.append(row(object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"lora{i}", run_id=f"lora:{v}", value=0.38, **common))
            for a in SI:
                rows.append(row(object_label=a, object_id=a, run_id=f"{a}:{v}", value=0.35 + 0.01 * i, **common))
        d = R.anchored(pd.DataFrame(rows))
        for a in SI:
            assert len(d[d.object_label.eq(a)]) == 3, d[d.object_label.eq(a)].run_id.tolist()
        assert dots(d).n_draws.tolist() == [3, 3]
        assert cells(d).n_runs.tolist() == [3, 3]

    def test_two_runs_of_one_standup_are_still_one_measurement(self):
        extra = row(object_label="safety_si", object_id="safety_si", run_id="pasttense:safety_si:qwen35_4b_s2:rung", view_legacy="qwen35_4b_s2",
                    lane_source_legacy="seedrep_cluster", value=0.25, draw="d2/2", token_cap_tokens=8192, pass_label="ladder_rung")
        d = R.anchored(pd.DataFrame(static_cell(refusal=False) + [extra]))
        got = d[d.object_label.eq("safety_si")]
        assert len(got) == 3 and got.view_legacy.value_counts().max() == 1, got[["run_id", "view_legacy"]].to_dict("records")
        assert int(cells(d).loc[("safety_si", "pasttense"), "n_runs"]) == 3, "the disclosure counts what the dot averages"

    def test_one_metric_under_two_spellings_is_one_instrument(self, monkeypatch):
        """Gemma's Constitution SI best-of-N run files `asr`, the fitted objects `asr_bon_cap4096`: one quantity (`metric_family`)."""
        monkeypatch.setattr(R, "headline_metric_ids", lambda: {"strongreject_bon": ["asr", "asr_bon_cap4096"]})
        common = dict(model_id=GEMMA, eval_id="strongreject_bon", item_set_id="jbb_goals_100", decode_source="grandfathered_T1",
                      decode_policy_id=None, run_extra_json=None, n=100, draw="")
        rows = [row(object_label="control_lora", object_id="lora", run_id="lora", view_legacy="gemma4e4b", metric_id="asr_bon_cap4096", **common),
                row(object_label="constitution_si", object_id="c", run_id="c", view_legacy="gemma4e4b_it", metric_id="asr", **common),
                row(object_label="constitution_si", object_id="c", run_id="c:8192", view_legacy="gemma4e4b_it", metric_id="asr",
                    **{**common, "token_cap_tokens": 8192})]
        kept, aside, cl = R.select_decode_only(pd.DataFrame(rows), setup="any")
        assert kept[kept.object_label.eq("constitution_si")].run_id.tolist() == ["c"]
        assert aside.set_index("run_id").decode_only_match.to_dict() == {"c:8192": "other instrument"}
        assert cl.status.tolist() == ["matched, cross-setup"]

    def test_where_seed_replicates_exist_the_single_fit_does_not_define_the_instrument(self):
        """The earlier single fit at 8,192 under the campaign's decode would make the SIs' campaign readings 'matched'; it is retired
        wherever seed replicates exist, so those readings stay set aside."""
        canon = row(object_label="control_lora", object_id="lora_canon", run_id="pasttense:lora:canon", view_legacy="qwen35_4b_it",
                    lane="instruct_direct", token_cap_tokens=8192, decode_source="unverified", decode_policy_id=None,
                    lane_source_legacy="sampled_lane", pass_label="pooled_t0_t1")
        for setup in ("main", "any"):
            kept, aside, _cl = R.select_decode_only(pd.DataFrame(static_cell(refusal=False) + [canon]), setup=setup)
            assert set(aside.run_id) == {"pasttense:safety_si:camp_it", "pasttense:constitution_si:camp_it", "pasttense:constitution_si:camp_base"}
            assert aside.decode_only_match.eq("other instrument").all()

    def test_rows_without_a_decode_only_condition_pass_through(self):
        rows = [r for r in static_cell(refusal=False) if r["object_label"] not in SI]
        kept, aside, cl = R.select_decode_only(pd.DataFrame(rows))
        assert len(kept) == len(rows) and len(aside) == 0 and len(cl) == 0


class TestTheRungTablesReadTheDiamondsRuns:
    """Figure 7's rung tables chose a prompted condition's runs again through `anchored`, and where no run matched the fitted
    objects' instrument that choice could differ from the diamond's; given the panels' `frontier_decode_runs` they read those runs."""

    def _drawn(self):
        f = TestTheFrontierDiamondIsTheDot()._frame()
        pts, _held, meta = R.frontier_points(f, R.FRONTIER_LANE, R.FRONTIER_STATIC[1], pull_lane=R.FRONTIER_PULL_LANE,
                                             objects=R.FRONTIER_OBJECTS, panel_name="static")
        return f, pts.set_index("object_label"), meta

    def test_the_runs_are_the_cells_runs(self):
        f, _pts, meta = self._drawn()
        runs = R.frontier_decode_runs(meta, None, {})
        want = {r for c in meta["decode_only_cells"].runs for r in c.split(",")}
        assert runs == want and runs and all(r.split(":")[1] in SI for r in runs)

    def test_rung_eval_table_prints_the_diamonds_readings_and_follows_the_given_runs(self):
        f, pts, meta = self._drawn()
        t = R.rung_eval_table(f, R.FRONTIER_OBJECTS, decode_runs=R.frontier_decode_runs(meta)).set_index("arm")
        assert t.loc["safety_si", "ST mean"] == pytest.approx(round(float(pts.loc["safety_si", "dy"]), 1))       # 38.0, the diamond's
        # a selection the table's own rule would not make (one standup's runs): the table reads exactly those
        one = {r for r in R.frontier_decode_runs(meta) if r.endswith(f":{SEED_VIEWS[0][0]}")}
        t1 = R.rung_eval_table(f, R.FRONTIER_OBJECTS, decode_runs=one).set_index("arm")
        assert t1.loc["safety_si", "ST mean"] == pytest.approx(40.0) and t1.loc["constitution_si", "ST mean"] == pytest.approx(15.0)
        ladder = lambda x: x.loc["difference_lora"].reset_index(drop=True)
        assert ladder(t1).equals(ladder(t)), "a fitted ladder is untouched"
        assert R.rung_eval_table(f, R.FRONTIER_OBJECTS).loc[lambda x: x.arm.eq("safety_si"), "ST mean"].tolist() == [38.0]

    def test_rung_quality_table_follows_the_given_runs(self):
        f, _pts, meta = self._drawn()
        f = f.assign(frac_at_token_cap=0.1)
        first = f.run_id.str.endswith(f":{SEED_VIEWS[0][0]}") & f.object_label.isin(SI)
        f.loc[first, "frac_at_token_cap"] = 0.9                                     # the first standup's SI runs: 90 % at cap
        runs = R.frontier_decode_runs(meta)
        rest = {r for r in runs if not r.endswith(f":{SEED_VIEWS[0][0]}")}
        col = R.EVAL_LABEL.get(R.FRONTIER_STATIC[1][0], R.FRONTIER_STATIC[1][0])
        q = lambda dr: R.rung_quality_table(f, "frac_at_token_cap", R.FRONTIER_OBJECTS, decode_runs=dr).set_index("arm").loc["safety_si", col]
        assert q(runs) == 90.0 and q(rest) == 10.0


class TestTheQualityGridReadsTheSameRuns:
    def test_rung_quality_table_keeps_the_runs_rung_eval_table_draws(self):
        """Figure 7's at-cap grid read every reading of a decode-only condition; it reads the runs of its one dot, as the score grid."""
        f = pd.DataFrame(multiturn_cell("crescendo")).assign(frac_at_token_cap=0.1)
        f.loc[f.run_id.eq(f"crescendo:{RS}:camp"), "frac_at_token_cap"] = 0.9            # the set-aside campaign run: 90 % at cap
        q = R.rung_quality_table(f, "frac_at_token_cap", [RS])
        assert q[R.EVAL_LABEL.get("crescendo", "crescendo")].tolist() == [10.0], q


class TestTheTextSaysIt:
    def test_the_readme_and_the_generator_state_the_rule(self):
        readme = (NOTEBOOKS / "README.md").read_text()
        para = " ".join(readme[readme.index("**Seeds and draws.**"):readme.index("**Lanes.**")].split())
        for phrase in ("one dot per cell", "matched", "padding", "closest", "Refusal steer", "serving setup", "padding panel",
                       "generation store", "averaged over seeds", "the unsteered reading the most of that seed's arms",
                       "a plurality that need not be a majority", "point here for ties"):
            assert phrase in para, phrase
        assert "vs a" not in para and "anchor most of its column's arms" not in para and "distinct unsteered readings" not in para
        assert "not drawn there" not in para and "no matched run" not in para
        gen = " ".join((NOTEBOOKS / "make_registry_notebook.py").read_text().split())
        assert "R.decode_only_disclosure(" in gen and 'setup="any"' in gen
        assert 'title="Figure 3, padding panel"' in gen, "the padding panel prints its set-aside readings under Figure 3"
        assert "unmatched=UNM" not in gen and "prompted_set_aside" not in gen


class TestTheDrawnBarOfAPooledCell:
    """A decode-only dot that averages several draws is drawn with the POOLED interval (`pool_draws`: the draws' own intervals combined
    plus the variance between them), the interval the per-seed tables print. `delta_figure` used to set each row's drawn bar before
    pooling, so a multi-draw cell drew its first draw's interval around the pooled mean (2026-11-14)."""

    def test_the_bar_handed_to_dodge_is_the_pooled_interval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        d = R.anchored(pd.DataFrame(static_cell("pasttense") + static_cell("codeattack", offset=0.05)))
        cats = {"static": (["pasttense", "codeattack"], -1, "x")}
        cm = R.category_means(d, categories=cats)
        gaps = pd.DataFrame(columns=["model_id", "object_label", "eval_id", "status"])
        handed = []
        real = R.dodge

        def spy(ax, pts, *a, **k):
            handed.append(pts.copy())
            return real(ax, pts, *a, **k)

        monkeypatch.setattr(R, "dodge", spy)
        fig = R.delta_figure(cm, d, [["static"]], list(DEC) + ["control_lora"], gaps, "bars_test", categories=cats)
        plt.close(fig)
        pts = pd.concat(handed, ignore_index=True)
        multi = pts[pts.n_draws > 1]
        assert len(multi) >= 4                                          # both SIs pool three seed-standup runs on each eval
        sign = -1
        for r in multi.itertuples():
            assert r.lo == pytest.approx(min(sign * r.d_lo, sign * r.d_hi)) and r.hi == pytest.approx(max(sign * r.d_lo, sign * r.d_hi))
            assert r.lo < r.y < r.hi
        # the pooled interval is not the first draw's: one SI cell, by hand
        raw = d[d.object_label.eq("safety_si") & d.eval_id.eq("pasttense") & d.d_y.notna()]
        c = R.cell_interval(raw)
        plo, phi = c.lo, c.hi
        si = multi[multi.object_label.eq("safety_si") & multi.eval_id.eq("pasttense")].iloc[0]
        assert (si.lo, si.hi) == pytest.approx((min(-plo, -phi), max(-plo, -phi)))
        first = raw.iloc[0]
        assert (si.lo, si.hi) != pytest.approx((min(-first.d_lo, -first.d_hi), max(-first.d_lo, -first.d_hi)))
        one = pts[pts.n_draws == 1]                                      # a one-draw cell keeps its own interval
        assert len(one) and all(r.lo <= r.y <= r.hi for r in one.itertuples())
